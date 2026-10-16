"""Compatibility module used by the installed Spark model registration."""
from integration.spark_fused_decode import (
    FusedState, SparkFusedForCausalLM, SparkFusedWorkerExtension, fused_status,
)

__all__ = ['FusedState', 'SparkFusedForCausalLM', 'SparkFusedWorkerExtension', 'fused_status']
