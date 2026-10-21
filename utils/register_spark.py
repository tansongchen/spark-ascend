"""Lazy, opt-in registration; ordinary BF16 model registration is unaffected."""
import os


def register_w4a16():
    if os.environ.get("SPARK_W4A16") != "1":
        return
    from vllm_spark2_5_plugin import register as register_spark
    from vllm import ModelRegistry

    register_spark()
    ModelRegistry.register_model(
        "Spark2_5ForCausalLM",
        "utils.spark_w4a16:SparkW4A16ForCausalLM",
    )
