"""Load Spark-X2.6's official mixed INT4/INT8 checkpoint on Ascend."""
import os

import torch
import torch_npu
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.utils import set_weight_attrs
from vllm_ascend.quantization.compressed_tensors_config import AscendCompressedTensorsConfig
from vllm_ascend.quantization.method_adapters import AscendLinearMethod
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod

from utils.int4_packing import repack_int4, int4_scales
from utils.model_config import validate_config
from utils.spark_fused_decode import SparkFusedForCausalLM


class CallAudit:
    def __init__(self):
        self.calls = 0
        self.input_dtypes = set()
        self.token_counts = set()

    def record(self, x):
        if x.dtype != torch.bfloat16:
            raise ValueError("Spark INT4 requires BF16 inputs")
        self.calls += 1
        self.input_dtypes.add(str(x.dtype))
        self.token_counts.add(x.numel() // x.shape[-1])


class SparkW4A16LinearMethod(CallAudit, LinearMethodBase):
    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **attrs):
        n, k = sum(output_partition_sizes), input_size_per_partition
        if params_dtype != torch.bfloat16 or n % 8 or k % 32:
            raise ValueError("Expected aligned BF16 W4A16 G32 dimensions")
        specs = {
            "weight_packed": ((n, k // 8), torch.int32, dict(input_dim=1, output_dim=0, packed_dim=1, packed_factor=8)),
            "weight_scale": ((n, k // 32), torch.bfloat16, dict(output_dim=0)),
            "weight_zero_point": ((n // 8, k // 32), torch.int32, dict(output_dim=0, packed_dim=0, packed_factor=8)),
        }
        for name, (shape, dtype, layout) in specs.items():
            param = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(param, {**attrs, **layout})
            layer.register_parameter(name, param)

    def process_weights_after_loading(self, layer):
        device = layer.weight_packed.device
        packed, scale, offset = repack_int4(layer.weight_packed.cpu(), layer.weight_scale.cpu(),
                                           layer.weight_zero_point.cpu())
        layer.weight_packed.data = packed.to(device)
        layer.weight_scale.data = scale.to(device)
        layer.register_buffer("weight_offset", offset.to(device))
        del layer.weight_zero_point

    def apply(self, layer, x, bias=None):
        self.record(x)
        y = torch_npu.npu_weight_quant_batchmatmul(
            x.reshape(-1, x.shape[-1]), layer.weight_packed, layer.weight_scale,
            antiquant_offset=layer.weight_offset, antiquant_group_size=32, bias=bias)
        return y.reshape(*x.shape[:-1], y.shape[-1])


class SparkW8A8Scheme(CallAudit, AscendW8A8DynamicLinearMethod):
    def get_perchannel_param(self, output_size, params_dtype):
        # Upstream INT8 scales are FP32. Loading them into BF16 loses precision.
        return {"weight_scale": torch.empty(output_size, 1, dtype=torch.float32),
                "weight_offset": torch.zeros(output_size, 1, dtype=torch.float32)}

    def apply(self, layer, x, bias=None, tp_rank=0):
        self.record(x)
        return super().apply(layer, x, bias, tp_rank)


class SparkInt4Config(AscendCompressedTensorsConfig):
    def get_quant_method(self, layer, prefix, **kwargs):
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            weight, activation, fmt = self._get_quant_args(layer, prefix)
            if weight is not None:
                layer.ascend_quant_method = "compressed-tensors"
                if weight.num_bits == 4:
                    return SparkW4A16LinearMethod()
                return AscendLinearMethod(SparkW8A8Scheme())
        return super().get_quant_method(layer, prefix, **kwargs)


class SparkInt4ForCausalLM(SparkFusedForCausalLM):
    # Q/K/V are already one tensor in the checkpoint; only gate/up are separate.
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_config.to_dict()
        validate_config(config)
        if (vllm_config.parallel_config.tensor_parallel_size != 1
                or vllm_config.parallel_config.pipeline_parallel_size != 1):
            raise ValueError("Spark INT4 adapter currently supports TP=PP=1")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("Spark INT4 adapter currently requires enforce_eager=True")
        if os.environ.get("SPARK_FUSED_DECODE") == "1":
            raise ValueError("BF16/G128 fused libraries cannot execute the upstream mixed INT4 G32/INT8 model")
        quant = SparkInt4Config.from_config(config["quantization_config"])
        quant.packed_modules_mapping = self.packed_modules_mapping
        vllm_config.quant_config = quant
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        modules = dict(self.named_modules())
        params = dict(self.named_parameters())
        shapes = set()
        def checked():
            for name, tensor in weights:
                if name.endswith(".weight_shape"):
                    module_name = name.removesuffix(".weight_shape")
                    mapped = module_name.replace(".gate_proj", ".gate_up_proj").replace(".up_proj", ".gate_up_proj")
                    module = modules[mapped]
                    n, packed_k = module.weight_packed.shape
                    if module_name != mapped:
                        n //= 2
                    if tensor.tolist() != [n, packed_k * 8]:
                        raise ValueError(f"Invalid upstream weight_shape: {name}={tensor.tolist()}")
                    shapes.add(module_name)
                else:
                    if name.endswith(".weight_scale") and not name.startswith("lm_head.") and ".out_proj." not in name:
                        tensor = int4_scales(tensor)
                    mapped = name.replace(".gate_proj.", ".gate_up_proj.").replace(".up_proj.", ".gate_up_proj.")
                    if tensor.dtype != params[mapped].dtype:
                        raise ValueError(f"Unexpected checkpoint dtype for {name}: {tensor.dtype}")
                    yield name, tensor
        loaded = super().load_weights(checked())
        required = {name for name, _ in self.named_parameters() if not name.endswith(".weight_offset")}
        if missing := required - loaded:
            raise ValueError(f"Missing checkpoint parameters: {sorted(missing)}")
        expected_shapes = {f"model.layers.{i}.{suffix}" for i in range(36)
                           for suffix in ("self_attn.q_k_v_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
        if shapes != expected_shapes:
            raise ValueError("Missing or unexpected INT4 weight_shape metadata")
        return loaded


class Int4WorkerExtension:
    def spark_quant_audit(self):
        model = self.get_model()
        rows = []
        for name, layer in model.named_modules():
            method = getattr(layer, "quant_method", None)
            scheme = getattr(method, "quant_method", method)
            if isinstance(scheme, (SparkW4A16LinearMethod, SparkW8A8Scheme)):
                int4 = isinstance(scheme, SparkW4A16LinearMethod)
                weight = layer.weight_packed if int4 else layer.weight
                rows.append(dict(name=name, bits=4 if int4 else 8, weight_dtype=str(weight.dtype),
                                 weight_shape=list(weight.shape), scale_dtype=str(layer.weight_scale.dtype),
                                 scale_shape=list(layer.weight_scale.shape), calls=scheme.calls,
                                 input_dtypes=sorted(scheme.input_dtypes), token_counts=sorted(scheme.token_counts),
                                 device=str(weight.device)))
        return dict(quantized_linear_layers=rows, embedding_dtype=str(model.model.embedding.weight.dtype),
                    tied_lm_head=model.lm_head is model.model.embedding,
                    parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()))

