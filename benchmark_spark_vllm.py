#!/usr/bin/env python3
"""Compatibility entry point; implementation lives in benchmarks/."""
from benchmarks.benchmark_spark_vllm import arguments, main, preflight

if __name__ == '__main__':
    main()
