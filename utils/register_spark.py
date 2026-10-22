"""Lazy registration of the upstream Spark INT4 Ascend adapter."""
import os


def register_int4():
    if os.environ.get("SPARK_INT4") != "1":
        return
    from vllm_spark2_5_plugin import register as register_spark
    from vllm import ModelRegistry

    register_spark()
    ModelRegistry.register_model(
        "Spark2_5ForCausalLM",
        "utils.spark_int4:SparkInt4ForCausalLM",
    )
