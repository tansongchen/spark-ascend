"""Spark-only Dense W4A16 using the native Ascend weight-only matmul.

Checkpoint weight: signed INT4, eight output rows per int32, [N/8, K].
Checkpoint scale: BF16 [N, K/group_size]. Runtime transposes both tensors.
This is a local format, not an AWQ, GPTQ or msModelSlim-exported checkpoint.
"""
import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm_ascend.quantization.methods.base import AscendLinearScheme
from vllm_ascend.quantization.methods.registry import register_scheme
from vllm_ascend.quantization.modelslim_config import packed_modules_model_mapping
from vllm_spark2_5_plugin.spark2_5 import Spark2_5ForCausalLM
from utils.spark_fused_decode import SparkFusedForCausalLM, SparkFusedWorkerExtension
from utils.spark_w4a16_fused import W4A16FusedState

FORMAT = "spark_int4_signed_output_pack_v1"
SCHEME = "SPARK_W4A16"


@register_scheme(SCHEME, "linear")
class SparkW4A16LinearScheme(AscendLinearScheme):
    def __init__(self):
        config = get_current_vllm_config()
        desc = config.quant_config.quant_description
        if desc.get("spark_format") != FORMAT:
            raise ValueError("Unexpected Spark INT4 checkpoint layout")
        self.group_size = int(desc["group_size"])
        if self.group_size not in (32, 64, 128):
            raise ValueError("Supported group sizes: 32, 64, 128")
        if config.parallel_config.tensor_parallel_size != 1 or config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("This Spark W4A16 adapter has only been validated for TP=PP=1")
        self.calls = 0
        self.input_dtypes = set()
        self.token_counts = set()

    def get_weight(self, input_size, output_size, params_dtype):
        if params_dtype != torch.bfloat16:
            raise ValueError("This checkpoint requires BF16 activations")
        if output_size % 8 or input_size % self.group_size:
            raise ValueError("Matrix dimensions do not align with INT4 packing/groups")
        return {
            "weight": torch.empty(output_size // 8, input_size, dtype=torch.int32),
            "_packed_dim": 0,
            "_packed_factor": 8,
        }

    def get_pergroup_param(self, input_size, output_size, params_dtype, layer_type=None):
        return {"weight_scale": torch.empty(output_size, input_size // self.group_size, dtype=params_dtype)}

    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.t().contiguous()
        layer.weight_scale.data = layer.weight_scale.data.t().contiguous()

    def apply(self, layer, x, bias=None, tp_rank=0):
        if x.dtype != torch.bfloat16 or layer.weight.dtype != torch.int32:
            raise ValueError("Expected BF16 activation and packed INT4/int32 weight")
        self.calls += 1
        self.input_dtypes.add(str(x.dtype))
        self.token_counts.add(x.numel() // x.shape[-1])
        shape = x.shape[:-1]
        y = torch_npu.npu_weight_quant_batchmatmul(
            x.reshape(-1, x.shape[-1]), layer.weight, layer.weight_scale,
            bias=bias, antiquant_group_size=self.group_size,
        )
        return y.reshape(*shape, y.shape[-1])


class SparkW4A16ForCausalLM(SparkFusedForCausalLM):
    fused_state_class = W4A16FusedState

    def __init__(self, *, vllm_config, prefix=""):
        if vllm_config.quant_config is None:
            raise ValueError("SPARK_W4A16=1 requires a quantized checkpoint")
        if not vllm_config.model_config.enforce_eager:
            raise ValueError("This Spark W4A16 adapter requires enforce_eager=True")
        # The source checkpoint already stores Q/K/V together.
        packed_modules_model_mapping["spark2_5"] = {"gate_up_proj": ["gate_proj", "up_proj"]}
        super().__init__(vllm_config=vllm_config, prefix=prefix)


class W4A16WorkerExtension(SparkFusedWorkerExtension):
    def spark_quant_audit(self):
        model = self.model_runner.model
        rows = []
        for name, layer in model.named_modules():
            scheme = getattr(getattr(layer, "quant_method", None), "quant_method", None)
            if isinstance(scheme, SparkW4A16LinearScheme):
                rows.append({
                    "name": name, "weight_dtype": str(layer.weight.dtype),
                    "weight_shape": list(layer.weight.shape),
                    "scale_dtype": str(layer.weight_scale.dtype),
                    "scale_shape": list(layer.weight_scale.shape),
                    "calls": scheme.calls, "input_dtypes": sorted(scheme.input_dtypes),
                    "token_counts": sorted(scheme.token_counts),
                    "device": str(layer.weight.device),
                })
        return {
            "quantized_linear_layers": rows,
            "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "embedding_dtype": str(model.model.embedding.weight.dtype),
            "tied_lm_head": model.lm_head is model.model.embedding,
        }
