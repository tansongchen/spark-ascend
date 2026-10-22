"""Shared defaults and the supported upstream Spark mixed-precision contract."""
import re
from pathlib import Path

MODEL_ID = "XHToken/Spark-X2.6-4B-INT4"
MODEL_REVISION = "4cd05c9e3bd23b08331fcc70fea6dbd5139ef29f"
DEFAULT_MODEL = Path("/data1/t00995472/models/Spark-X2.6-4B-INT4")


def is_official_int4(config):
    return (config.get("model_type") == "spark2_5"
            and (config.get("quantization_config") or {}).get("quant_method") == "compressed-tensors")


def quantization_for(config, name):
    """Resolve exact/regex targets without changing the upstream config."""
    matches = []
    for group in config["quantization_config"]["config_groups"].values():
        if any((re.fullmatch(target[3:], name) is not None if target.startswith("re:")
                else target == name) for target in group["targets"]):
            matches.append(group)
    if len(matches) != 1:
        raise ValueError(f"Expected one quantization scheme for {name}, got {len(matches)}")
    return matches[0]


def validate_config(config):
    expected = dict(model_type="spark2_5", hidden_size=2560, intermediate_size=10240,
                    num_hidden_layers=36, num_attention_heads=16, num_key_value_heads=4,
                    head_dim=256, vocab_size=131072, tie_word_embeddings=False)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported Spark configuration: {key}={config.get(key)!r}")
    if not is_official_int4(config):
        raise ValueError("Expected the upstream compressed-tensors INT4 checkpoint")
    for i in range(36):
        for suffix in ("self_attn.q_k_v_proj", "mlp.gate_proj", "mlp.up_proj",
                       "mlp.gate_up_proj", "mlp.down_proj", "self_attn.out_proj"):
            name = f"model.layers.{i}.{suffix}"
            validate_scheme(quantization_for(config, name), name)
    validate_scheme(quantization_for(config, "lm_head"), "lm_head")
    return config


def validate_scheme(scheme, name):
    w, a = scheme["weights"], scheme.get("input_activations")
    int8 = name == "lm_head" or name.endswith("self_attn.out_proj")
    expected = dict(num_bits=8 if int8 else 4, type="int", symmetric=int8,
                    strategy="channel" if int8 else "group", dynamic=False,
                    scale_dtype="torch.float32" if int8 else "torch.bfloat16")
    if any(w.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Unsupported weight scheme for {name}: {w}")
    if scheme.get("format") != ("int-quantized" if int8 else "pack-quantized"):
        raise ValueError(f"Unsupported compression format for {name}")
    if int8:
        expected_a = dict(num_bits=8, type="int", symmetric=True, strategy="token", dynamic=True)
        if a is None or any(a.get(k) != v for k, v in expected_a.items()):
            raise ValueError(f"Expected dynamic per-token INT8 activations for {name}")
    elif a is not None or w.get("group_size") != 32 or w.get("actorder") not in (None, "static"):
        raise ValueError(f"Expected asymmetric W4A16 G32 without a runtime permutation: {name}")


def validate_audit(audit, minimum_calls=1):
    if not audit:
        raise RuntimeError("No Spark INT4 worker audit returned")
    for worker in audit:
        layers = worker["quantized_linear_layers"]
        expected = {f"model.layers.{i}.{suffix}": bits for i in range(36)
                    for suffix, bits in (("self_attn.q_k_v_proj", 4), ("mlp.gate_up_proj", 4),
                                         ("mlp.down_proj", 4), ("self_attn.out_proj", 8))}
        expected["lm_head"] = 8
        if (len(layers) != 145 or {row["name"]: row["bits"] for row in layers} != expected
                or worker["tied_lm_head"] or worker["embedding_dtype"] != "torch.bfloat16"):
            raise RuntimeError("Expected 108 INT4 layers, 36 INT8 out projections and an independent INT8 LM head")
        for row in layers:
            if (row["weight_dtype"] != ("torch.int32" if row["bits"] == 4 else "torch.int8")
                    or row["scale_dtype"] != ("torch.bfloat16" if row["bits"] == 4 else "torch.float32")
                    or row["input_dtypes"] != ["torch.bfloat16"] or row["calls"] < minimum_calls):
                raise RuntimeError(f"Invalid INT4/INT8 runtime audit: {row}")
