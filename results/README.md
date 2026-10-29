# Benchmark results

All published benchmark runs: Qwen3-14B on an NVIDIA H100 80GB HBM3 (SXM), on the suite's workloads, with
batchinfer's defaults, every configuration in the suite (`ARMS` in `bench/suite.py`) and vLLM 0.30. `index.html` lists
every run, with a table of every configuration on every workload behind a dropdown at the top. Each run links to its
own `report.html` with charts. `RUN.md` has the commands that produced them and the workload files' hashes.

The main comparisons (vLLM with and without its prefix cache, batchinfer's defaults, batchinfer without prefix
sharing, and batchinfer without overlap) are the run with the median wall time out of three. On `docqa` and `rag`
only batchinfer's defaults and vLLM with its cache are medians. Everything else ran once. `runs.json` has every
run's wall time for every configuration and which run was kept.

Each run is a directory `<workload>/<model>/<gpu>/<label>/`, where the label lists the options that differ from the
engine's defaults. It contains:

- `metrics.json`: what ran (engine, options, configuration name, commit, package versions, GPU, clock and throttling),
  the wall time of the timed pass, accuracy per source, how the outputs match the same GPU's vLLM run, and the
  engine's own stats (`backend_stats`, including `kv_capacity_tokens`, the usable KV cache as each engine reports
  it).
- `details.json`: the engine's full record, including the per-step trace (for vLLM, its engine iterations).
- `outputs.jsonl`: every request's generated text and token ids.

`python -m bench.report --html results/` regenerates the pages. The commits recorded in the results are from the
development repository and don't exist in the released history.
