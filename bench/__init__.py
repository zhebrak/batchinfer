"""Benchmark workloads built from public datasets, and a runner that feeds them to any backend.

    python -m bench.workload build --preset mixed --size quick --model Qwen/Qwen3-8B
    python -m bench.run workloads/mixed-quick.jsonl --backend vllm
    python -m bench.report results/
"""
