# Benchmark results

Every row behind the tables in the top-level README, as `bench.run` and `bench.suite` wrote them on 2026-09-28:
Qwen3-8B and Qwen3-1.7B on an H100 PCIe (80 GB) and an A100-SXM4 (40 GB), on five workloads each, with batchinfer's
defaults and every option, vLLM 0.30 (prefix cache on, on with `max_num_seqs=512`, on with `enforce_eager`, and off)
and the naive baseline. The README's tables cite every configuration on `mixed` and the main ones on all five
workloads; the other option rows on the other four workloads are kept for reference. Each suite row is the run with
the median wall time of three runs of the suite on its GPU, and naive ran once per workload (`RUN.md` gives the
calls). `scripts/median_rows.py` assembled the tree: `runs.json` lists every run's wall time of every row and which
run the row is, and each row's `compare` was recomputed against the vLLM row beside it, since vLLM's greedy
generations differ from run to run at bf16 near-ties. One directory per row, `<workload>/<model>/<gpu>/<label>/`,
where the label names the options the row set beyond its engine's defaults:

- `metrics.json`: what ran (engine, options, commit, package versions, GPU), wall time of the timed pass, accuracy
  per source, the match against this GPU's vLLM row, and the engine's own stats (`backend_stats`).
- `details.json`: the engine's full record, including its per-step trace (vLLM's: its engine iterations).
- `outputs.jsonl`: every request's generated text and token ids.

The workload files are not included: `python -m bench.workload build --preset <mixed|classify|generate|sweep0|sweep90>
--size quick --model Qwen/Qwen3-8B` rebuilds them from the public datasets, and every row records the workload's
sha256 to check against. `index.html` is the browsable index and each row's `report.html` its page with charts
(open them from a clone, or at https://zhebrak.github.io/batchinfer/results/); `python -m bench.report --html results/` regenerates them, and
the table below is `python -m bench.report results/`. `RUN.md` lists the hardware, versions and workload hashes, and
the command behind each README table.

The `commit` column is the development commit a row ran at, which is not in this repository's history. Every row ran
at `0d19c51`, with the engines and the benchmark harness published here; only the report's rendering and the
documentation changed after it.

| engine | workload | model | GPU | prefix reuse | order | prefill budget | layers | graphs | max seqs | wall s | × vLLM | accuracy % | prefix hit % | steps | ms/step | decode batch | GPU util % | MFU % | MBU % | commit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | off | max_tokens_desc | - | HF | off | - | 82.20 | 15.82 | 53.6 | 0.0 | - | - | 19.0 | 90.1 | 4.2 | 5.7 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | off | - | - | compiled | on | - | 9.66 | 1.86 | 53.1 | - | 549 | 17.6 | 48.3 | 96.4 | 35.3 | 28.7 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | eager | off | - | 12.94 | 2.49 | 54.0 | - | 520 | 24.9 | 51.0 | 43.4 | 7.0 | 20.9 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | 512 | 5.21 | 1.00 | 53.1 | - | 521 | 10.0 | 50.9 | 93.4 | 17.3 | 51.9 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | - | 5.20 | 1.00 | 53.1 | - | 521 | 10.0 | 50.9 | 93.5 | 17.4 | 52.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | decode_ratio_desc | 16,384 | fused | decode 99% | - | 5.67 | 1.09 | 53.6 | 80.5 | 512 | 11.1 | 51.7 | 91.2 | 15.9 | 47.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | input | 16,384 | fused | decode 99% | - | 5.62 | 1.08 | 53.6 | 80.5 | 515 | 10.9 | 51.4 | 91.2 | 16.0 | 47.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | max_tokens_desc | 16,384 | fused | decode 99% | - | 5.64 | 1.08 | 53.6 | 80.5 | 512 | 11.0 | 51.7 | 91.1 | 16.0 | 47.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 98% | - | 7.08 | 1.36 | 53.1 | 80.5 | 778 | 9.1 | 34.3 | 92.0 | 12.8 | 46.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 5.69 | 1.10 | 53.6 | 80.5 | 512 | 11.1 | 51.7 | 91.2 | 15.9 | 47.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | HF | off | - | 19.20 | 3.70 | 53.6 | 80.5 | 512 | 37.5 | 51.7 | 48.0 | 4.7 | 14.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 583 | fused | off | - | 7.10 | 1.37 | 54.0 | 80.5 | 534 | 13.3 | 49.6 | 80.1 | 12.7 | 38.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | off | - | 6.78 | 1.30 | 53.6 | 80.5 | 512 | 13.2 | 51.7 | 81.0 | 13.3 | 39.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 583 | fused | decode 81% | - | 6.09 | 1.17 | 54.0 | 80.5 | 534 | 11.4 | 49.6 | 88.6 | 14.8 | 44.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 583 | fused | decode 81% | - | 6.08 | 1.17 | 54.0 | 80.5 | 534 | 11.4 | 49.6 | 88.9 | 14.8 | 44.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 5.69 | 1.10 | 53.6 | 80.5 | 512 | 11.1 | 51.7 | 90.9 | 15.9 | 47.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | A100-SXM4-40GB | off | prefix_dfs | 16,384 | fused | decode 96% | - | 10.09 | 1.94 | 53.1 | 0.0 | 512 | 19.7 | 51.7 | 93.8 | 33.8 | 26.6 | 0d19c51 · 09-28 02:09 |
| naive | mixed-quick | Qwen3-1.7B | H100-PCIe | off | max_tokens_desc | - | HF | off | - | 93.12 | 26.33 | 54.0 | 0.0 | - | - | 19.0 | 54.9 | 1.5 | 3.9 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-1.7B | H100-PCIe | off | - | - | compiled | on | - | 5.86 | 1.66 | 53.1 | - | 532 | 11.0 | 49.9 | 94.4 | 24.0 | 36.2 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | eager | off | - | 6.87 | 1.94 | 53.1 | - | 535 | 12.8 | 49.6 | 50.1 | 5.4 | 31.0 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | compiled | on | 512 | 3.54 | 1.00 | 53.1 | - | 527 | 6.7 | 50.3 | 82.6 | 10.5 | 59.6 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | compiled | on | - | 3.54 | 1.00 | 53.1 | - | 534 | 6.6 | 49.7 | 83.7 | 10.5 | 60.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | global trie | decode_ratio_desc | 16,384 | fused | decode 99% | - | 4.19 | 1.19 | 52.7 | 80.5 | 512 | 8.2 | 51.7 | 90.2 | 8.9 | 49.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | global trie | input | 16,384 | fused | decode 99% | - | 4.23 | 1.20 | 53.1 | 80.5 | 515 | 8.2 | 51.4 | 91.0 | 8.8 | 49.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | global trie | max_tokens_desc | 16,384 | fused | decode 99% | - | 4.17 | 1.18 | 52.7 | 80.5 | 512 | 8.1 | 51.7 | 93.1 | 8.9 | 50.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 98% | - | 5.26 | 1.49 | 53.1 | 80.5 | 778 | 6.8 | 34.3 | 92.9 | 7.1 | 48.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 4.20 | 1.19 | 53.1 | 80.5 | 512 | 8.2 | 51.7 | 91.9 | 8.9 | 49.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | HF | off | - | 12.27 | 3.47 | 53.1 | 80.5 | 512 | 24.0 | 51.7 | 59.3 | 3.0 | 17.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 583 | fused | off | - | 5.13 | 1.45 | 53.1 | 80.5 | 534 | 9.6 | 49.6 | 82.7 | 7.3 | 41.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | off | - | 5.51 | 1.56 | 52.7 | 80.5 | 512 | 10.8 | 51.7 | 75.9 | 6.8 | 37.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 583 | fused · FA3 | off | - | 4.27 | 1.21 | 53.1 | 80.5 | 534 | 8.0 | 49.6 | 81.0 | 8.7 | 49.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused · FA3 | off | - | 5.15 | 1.46 | 53.1 | 80.5 | 512 | 10.1 | 51.7 | 66.7 | 7.2 | 40.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 583 | fused | decode 81% | - | 4.48 | 1.27 | 53.1 | 80.5 | 534 | 8.4 | 49.6 | 87.3 | 8.3 | 47.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 583 | fused | decode 81% | - | 4.45 | 1.26 | 53.1 | 80.5 | 534 | 8.3 | 49.6 | 89.2 | 8.4 | 47.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 4.24 | 1.20 | 52.7 | 80.5 | 512 | 8.3 | 51.7 | 90.6 | 8.8 | 49.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-1.7B | H100-PCIe | off | prefix_dfs | 16,384 | fused | decode 96% | - | 6.95 | 1.96 | 53.1 | 0.0 | 512 | 13.6 | 51.7 | 94.2 | 20.3 | 30.1 | 0d19c51 · 09-28 02:09 |
| naive | mixed-quick | Qwen3-8B | A100-SXM4-40GB | off | max_tokens_desc | - | HF | off | - | 176.79 | 12.92 | 76.8 | 0.0 | - | - | 19.0 | 97.4 | 8.8 | 8.8 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-8B | A100-SXM4-40GB | off | - | - | compiled | on | - | 30.67 | 2.24 | 76.3 | - | 549 | 55.9 | 48.3 | 99.5 | 51.0 | 23.9 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | eager | off | - | 18.89 | 1.38 | 77.2 | - | 520 | 36.3 | 51.0 | 74.9 | 21.7 | 37.4 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | 512 | 13.69 | 1.00 | 76.3 | - | 521 | 26.3 | 50.9 | 98.7 | 29.9 | 51.6 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | - | 13.69 | 1.00 | 76.3 | - | 521 | 26.3 | 50.9 | 98.9 | 29.9 | 51.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | A100-SXM4-40GB | global trie | decode_ratio_desc | 16,384 | fused | decode 99% | - | 14.30 | 1.04 | 76.3 | 80.5 | 512 | 27.9 | 51.7 | 96.3 | 28.6 | 48.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | A100-SXM4-40GB | global trie | input | 16,384 | fused | decode 99% | - | 14.20 | 1.04 | 76.3 | 80.5 | 515 | 27.6 | 51.4 | 96.4 | 28.8 | 49.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | A100-SXM4-40GB | global trie | max_tokens_desc | 16,384 | fused | decode 99% | - | 14.23 | 1.04 | 76.3 | 80.5 | 512 | 27.8 | 51.7 | 96.5 | 28.7 | 49.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 21.13 | 1.54 | 76.8 | 80.5 | 1035 | 20.4 | 25.8 | 97.1 | 19.4 | 57.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 14.32 | 1.05 | 76.3 | 80.5 | 512 | 28.0 | 51.7 | 95.4 | 28.6 | 48.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | HF | off | - | 27.90 | 2.04 | 76.8 | 80.5 | 512 | 54.5 | 51.7 | 70.4 | 14.7 | 25.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 583 | fused | off | - | 16.14 | 1.18 | 76.3 | 80.5 | 534 | 30.2 | 49.6 | 93.4 | 25.3 | 44.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | off | - | 15.05 | 1.10 | 76.3 | 80.5 | 512 | 29.4 | 51.7 | 93.4 | 27.2 | 46.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 583 | fused | decode 81% | - | 15.56 | 1.14 | 76.3 | 80.5 | 534 | 29.1 | 49.6 | 95.5 | 26.3 | 46.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 583 | fused | decode 81% | - | 15.54 | 1.14 | 76.3 | 80.5 | 534 | 29.1 | 49.6 | 95.5 | 26.3 | 46.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 14.29 | 1.04 | 76.3 | 80.5 | 512 | 27.9 | 51.7 | 96.2 | 28.6 | 48.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | A100-SXM4-40GB | off | prefix_dfs | 16,384 | fused | decode 96% | - | 31.14 | 2.28 | 76.3 | 0.0 | 512 | 60.8 | 51.7 | 98.0 | 50.2 | 22.4 | 0d19c51 · 09-28 02:09 |
| naive | mixed-quick | Qwen3-8B | H100-PCIe | off | max_tokens_desc | - | HF | off | - | 156.08 | 17.79 | 76.3 | 0.0 | - | - | 19.0 | 75.5 | 4.1 | 7.8 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-8B | H100-PCIe | off | - | - | compiled | on | - | 18.30 | 2.09 | 75.9 | - | 532 | 34.4 | 49.9 | 98.9 | 35.3 | 30.5 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | eager | off | - | 9.75 | 1.11 | 76.8 | - | 519 | 18.8 | 51.1 | 92.6 | 17.3 | 56.2 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | compiled | on | 512 | 8.83 | 1.01 | 76.3 | - | 518 | 17.0 | 51.2 | 95.7 | 19.1 | 62.0 | 0d19c51 · 09-28 02:09 |
| vllm | mixed-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | compiled | on | - | 8.77 | 1.00 | 75.9 | - | 518 | 16.9 | 51.2 | 97.7 | 19.2 | 62.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | global trie | decode_ratio_desc | 16,384 | fused | decode 99% | - | 10.03 | 1.14 | 76.8 | 80.5 | 512 | 19.6 | 51.7 | 95.6 | 16.8 | 54.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | global trie | input | 16,384 | fused | decode 99% | - | 10.05 | 1.15 | 76.3 | 80.5 | 515 | 19.5 | 51.4 | 95.1 | 16.8 | 54.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | global trie | max_tokens_desc | 16,384 | fused | decode 99% | - | 10.01 | 1.14 | 76.8 | 80.5 | 512 | 19.5 | 51.7 | 96.1 | 16.9 | 54.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 98% | - | 12.76 | 1.45 | 76.8 | 80.5 | 777 | 16.4 | 34.4 | 96.5 | 13.2 | 58.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 10.05 | 1.15 | 76.8 | 80.5 | 512 | 19.6 | 51.7 | 95.0 | 16.8 | 54.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | HF | off | - | 16.01 | 1.83 | 76.3 | 80.5 | 512 | 31.3 | 51.7 | 88.2 | 10.5 | 33.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 583 | fused | off | - | 11.44 | 1.30 | 76.8 | 80.5 | 534 | 21.4 | 49.6 | 91.8 | 14.8 | 48.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | off | - | 10.92 | 1.24 | 76.8 | 80.5 | 512 | 21.3 | 51.7 | 91.7 | 15.5 | 49.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 583 | fused · FA3 | off | - | 9.89 | 1.13 | 76.8 | 80.5 | 534 | 18.5 | 49.6 | 87.9 | 17.1 | 56.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused · FA3 | off | - | 9.72 | 1.11 | 76.3 | 80.5 | 512 | 19.0 | 51.7 | 89.6 | 17.4 | 55.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 583 | fused | decode 81% | - | 10.74 | 1.22 | 76.8 | 80.5 | 534 | 20.1 | 49.6 | 94.3 | 15.7 | 52.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 583 | fused | decode 81% | - | 10.74 | 1.22 | 76.8 | 80.5 | 534 | 20.1 | 49.6 | 95.4 | 15.7 | 52.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 10.10 | 1.15 | 76.8 | 80.5 | 512 | 19.7 | 51.7 | 95.3 | 16.7 | 53.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | mixed-quick | Qwen3-8B | H100-PCIe | off | prefix_dfs | 16,384 | fused | decode 96% | - | 20.13 | 2.29 | 76.3 | 0.0 | 512 | 39.3 | 51.7 | 97.6 | 32.1 | 27.0 | 0d19c51 · 09-28 02:09 |
| naive | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | off | max_tokens_desc | - | HF | off | - | 15.27 | 12.43 | 54.5 | 0.0 | - | - | 23.2 | 95.5 | 27.4 | 0.5 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | off | - | - | compiled | on | - | 6.71 | 5.47 | 54.0 | - | 53 | 126.7 | 5.5 | 93.6 | 62.2 | 2.3 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | eager | off | - | 1.26 | 1.03 | 54.5 | - | 13 | 97.2 | 25.4 | 78.6 | 51.3 | 5.3 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | 512 | 1.23 | 1.00 | 54.3 | - | 14 | 87.5 | 23.0 | 88.7 | 53.0 | 5.6 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | - | 1.23 | 1.00 | 54.3 | - | 14 | 87.7 | 23.0 | 92.4 | 52.9 | 5.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | decode_ratio_desc | 16,384 | fused | decode 17% | - | 1.37 | 1.11 | 53.7 | 84.7 | 6 | 227.8 | 36.4 | 81.1 | 47.5 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | input | 16,384 | fused | decode 38% | - | 1.36 | 1.11 | 53.7 | 84.7 | 8 | 170.5 | 26.0 | 81.0 | 47.6 | 4.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | max_tokens_desc | 16,384 | fused | decode 17% | - | 1.35 | 1.10 | 53.7 | 84.7 | 6 | 225.3 | 36.4 | 81.3 | 48.0 | 3.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 29% | - | 1.44 | 1.18 | 54.0 | 84.7 | 24 | 60.1 | 20.2 | 81.1 | 45.0 | 6.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 0% | - | 1.37 | 1.12 | 53.7 | 84.7 | 6 | 229.0 | 36.4 | 80.3 | 47.2 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | HF | off | - | 1.87 | 1.52 | 54.5 | 84.7 | 6 | 311.8 | 36.6 | 85.1 | 34.7 | 2.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 16,384 | fused | off | - | 1.36 | 1.11 | 53.7 | 84.7 | 6 | 227.5 | 36.4 | 78.3 | 47.6 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | off | - | 1.36 | 1.11 | 53.7 | 84.7 | 6 | 227.5 | 36.4 | 80.0 | 47.5 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 16,384 | fused | decode 0% | - | 1.37 | 1.11 | 53.7 | 84.7 | 6 | 227.8 | 36.4 | 79.4 | 47.5 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 0% | - | 1.38 | 1.12 | 53.7 | 84.7 | 6 | 229.7 | 36.4 | 79.9 | 47.1 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 0% | - | 1.37 | 1.12 | 53.7 | 84.7 | 6 | 228.8 | 36.4 | 78.7 | 47.3 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | A100-SXM4-40GB | off | prefix_dfs | 16,384 | fused | decode 0% | - | 6.92 | 5.64 | 53.7 | 0.0 | 25 | 276.9 | 26.0 | 94.2 | 60.3 | 1.3 | 0d19c51 · 09-28 02:09 |
| naive | classify-quick | Qwen3-1.7B | H100-PCIe | off | max_tokens_desc | - | HF | off | - | 9.67 | 6.76 | 54.5 | 0.0 | - | - | 22.9 | 84.7 | 17.8 | 0.6 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-1.7B | H100-PCIe | off | - | - | compiled | on | - | 3.84 | 2.68 | 54.3 | - | 29 | 132.3 | 10.4 | 90.2 | 44.9 | 2.1 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | eager | off | - | 1.40 | 0.98 | 53.7 | - | 35 | 40.1 | 8.4 | 40.0 | 19.1 | 6.4 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | compiled | on | 512 | 1.39 | 0.97 | 54.3 | - | 30 | 46.5 | 10.0 | 39.9 | 19.2 | 5.8 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | compiled | on | - | 1.43 | 1.00 | 54.3 | - | 30 | 47.7 | 10.0 | 38.7 | 18.7 | 5.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | global trie | decode_ratio_desc | 16,384 | fused | decode 17% | - | 0.97 | 0.68 | 53.2 | 84.7 | 6 | 162.2 | 36.6 | 62.6 | 27.5 | 4.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | global trie | input | 16,384 | fused | decode 38% | - | 0.90 | 0.63 | 53.2 | 84.7 | 8 | 112.6 | 26.1 | 79.0 | 29.7 | 4.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | global trie | max_tokens_desc | 16,384 | fused | decode 17% | - | 0.89 | 0.62 | 53.2 | 84.7 | 6 | 147.8 | 36.6 | 71.6 | 30.2 | 4.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 27% | - | 1.03 | 0.72 | 53.2 | 84.7 | 22 | 47.0 | 20.3 | 73.8 | 25.9 | 6.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 0% | - | 0.98 | 0.68 | 53.2 | 84.7 | 6 | 163.2 | 36.6 | 58.2 | 27.4 | 4.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | HF | off | - | 1.33 | 0.93 | 53.7 | 84.7 | 6 | 221.8 | 37.0 | 75.6 | 20.1 | 3.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 16,384 | fused | off | - | 0.90 | 0.63 | 53.2 | 84.7 | 6 | 150.0 | 36.6 | 77.0 | 29.8 | 4.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | off | - | 0.91 | 0.63 | 53.2 | 84.7 | 6 | 150.8 | 36.6 | 80.0 | 29.6 | 4.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 16,384 | fused · FA3 | off | - | 0.83 | 0.58 | 53.2 | 84.7 | 6 | 138.7 | 36.6 | 67.5 | 32.2 | 4.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused · FA3 | off | - | 0.82 | 0.57 | 53.2 | 84.7 | 6 | 136.8 | 36.6 | 68.0 | 32.6 | 4.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 16,384 | fused | decode 0% | - | 0.88 | 0.62 | 53.2 | 84.7 | 6 | 147.2 | 36.6 | 78.6 | 30.3 | 4.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 0% | - | 0.98 | 0.69 | 53.2 | 84.7 | 6 | 163.7 | 36.6 | 58.0 | 27.3 | 4.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 0% | - | 0.90 | 0.63 | 53.2 | 84.7 | 6 | 149.8 | 36.6 | 79.2 | 29.8 | 4.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-1.7B | H100-PCIe | off | prefix_dfs | 16,384 | fused | decode 0% | - | 4.37 | 3.05 | 53.2 | 0.0 | 25 | 174.6 | 26.1 | 91.4 | 39.5 | 1.7 | 0d19c51 · 09-28 02:09 |
| naive | classify-quick | Qwen3-8B | A100-SXM4-40GB | off | max_tokens_desc | - | HF | off | - | 48.66 | 11.16 | 79.3 | 0.0 | - | - | 19.9 | 98.7 | 39.9 | 0.5 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-8B | A100-SXM4-40GB | off | - | - | compiled | on | - | 27.31 | 6.26 | 79.5 | - | 52 | 525.2 | 6.2 | 98.3 | 71.2 | 2.0 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | eager | off | - | 4.46 | 1.02 | 79.8 | - | 12 | 371.9 | 31.5 | 93.4 | 67.2 | 3.7 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | 512 | 4.37 | 1.00 | 79.5 | - | 12 | 364.1 | 31.4 | 90.0 | 68.7 | 3.8 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | - | 4.36 | 1.00 | 79.5 | - | 12 | 363.4 | 31.4 | 94.8 | 68.8 | 3.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | A100-SXM4-40GB | global trie | decode_ratio_desc | 16,384 | fused | decode 44% | - | 4.66 | 1.07 | 79.3 | 84.7 | 9 | 518.2 | 27.4 | 94.0 | 64.3 | 3.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | A100-SXM4-40GB | global trie | input | 16,384 | fused | decode 50% | - | 4.66 | 1.07 | 79.3 | 84.7 | 10 | 465.6 | 24.3 | 92.4 | 64.4 | 3.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | A100-SXM4-40GB | global trie | max_tokens_desc | 16,384 | fused | decode 44% | - | 4.65 | 1.07 | 79.3 | 84.7 | 9 | 516.4 | 27.4 | 93.8 | 64.5 | 3.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 34% | - | 4.83 | 1.11 | 79.3 | 84.7 | 29 | 166.6 | 20.0 | 93.6 | 62.1 | 6.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 33% | - | 4.67 | 1.07 | 79.3 | 84.7 | 9 | 518.9 | 27.4 | 93.7 | 64.2 | 3.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | HF | off | - | 5.83 | 1.34 | 79.5 | 84.7 | 9 | 647.3 | 27.4 | 94.8 | 51.5 | 2.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 16,384 | fused | off | - | 4.67 | 1.07 | 79.3 | 84.7 | 9 | 519.1 | 27.4 | 93.6 | 64.2 | 3.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | off | - | 4.67 | 1.07 | 79.3 | 84.7 | 9 | 519.0 | 27.4 | 93.6 | 64.2 | 3.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 16,384 | fused | decode 33% | - | 4.67 | 1.07 | 79.3 | 84.7 | 9 | 518.9 | 27.4 | 93.6 | 64.2 | 3.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 33% | - | 4.66 | 1.07 | 79.3 | 84.7 | 9 | 517.7 | 27.4 | 93.9 | 64.4 | 3.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 33% | - | 4.67 | 1.07 | 79.3 | 84.7 | 9 | 518.6 | 27.4 | 93.4 | 64.3 | 3.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | A100-SXM4-40GB | off | prefix_dfs | 16,384 | fused | decode 0% | - | 27.35 | 6.27 | 79.3 | 0.0 | 25 | 1,094.0 | 31.3 | 98.4 | 71.1 | 1.1 | 0d19c51 · 09-28 02:09 |
| naive | classify-quick | Qwen3-8B | H100-PCIe | off | max_tokens_desc | - | HF | off | - | 28.11 | 11.16 | 79.3 | 0.0 | - | - | 19.9 | 96.9 | 28.5 | 0.7 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-8B | H100-PCIe | off | - | - | compiled | on | - | 15.44 | 6.13 | 79.0 | - | 28 | 551.4 | 12.0 | 98.5 | 51.9 | 1.6 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | eager | off | - | 2.66 | 1.06 | 79.5 | - | 11 | 241.7 | 35.0 | 88.7 | 46.6 | 4.6 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | compiled | on | 512 | 2.56 | 1.02 | 79.3 | - | 11 | 232.9 | 35.0 | 88.6 | 48.3 | 4.8 | 0d19c51 · 09-28 02:09 |
| vllm | classify-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | compiled | on | - | 2.52 | 1.00 | 79.0 | - | 10 | 251.9 | 39.4 | 90.6 | 49.1 | 4.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | global trie | decode_ratio_desc | 16,384 | fused | decode 44% | - | 2.88 | 1.14 | 79.5 | 84.7 | 9 | 320.2 | 27.5 | 92.0 | 43.0 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | global trie | input | 16,384 | fused | decode 50% | - | 2.88 | 1.14 | 79.5 | 84.7 | 10 | 288.3 | 24.4 | 92.0 | 42.9 | 4.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | global trie | max_tokens_desc | 16,384 | fused | decode 44% | - | 2.88 | 1.14 | 79.5 | 84.7 | 9 | 319.6 | 27.5 | 91.5 | 43.1 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 31% | - | 2.96 | 1.18 | 79.5 | 84.7 | 26 | 114.0 | 20.0 | 91.9 | 41.8 | 8.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 33% | - | 2.85 | 1.13 | 79.5 | 84.7 | 9 | 316.6 | 27.5 | 86.6 | 43.5 | 3.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | HF | off | - | 3.75 | 1.49 | 79.5 | 84.7 | 9 | 417.1 | 27.4 | 91.4 | 33.0 | 2.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 16,384 | fused | off | - | 2.91 | 1.16 | 79.5 | 84.7 | 9 | 323.6 | 27.5 | 88.0 | 42.5 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | off | - | 2.87 | 1.14 | 79.5 | 84.7 | 9 | 319.2 | 27.5 | 90.2 | 43.1 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 16,384 | fused · FA3 | off | - | 2.70 | 1.07 | 79.5 | 84.7 | 9 | 299.6 | 27.5 | 86.5 | 45.9 | 4.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused · FA3 | off | - | 2.64 | 1.05 | 79.5 | 84.7 | 9 | 293.0 | 27.5 | 87.9 | 47.0 | 4.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 16,384 | fused | decode 33% | - | 2.88 | 1.14 | 79.5 | 84.7 | 9 | 320.0 | 27.5 | 91.9 | 43.0 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 33% | - | 2.88 | 1.14 | 79.5 | 84.7 | 9 | 319.9 | 27.5 | 92.1 | 43.0 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 33% | - | 2.88 | 1.14 | 79.5 | 84.7 | 9 | 319.4 | 27.5 | 90.3 | 43.1 | 3.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | classify-quick | Qwen3-8B | H100-PCIe | off | prefix_dfs | 16,384 | fused | decode 0% | - | 16.62 | 6.60 | 79.5 | 0.0 | 25 | 664.9 | 31.4 | 97.8 | 48.2 | 1.4 | 0d19c51 · 09-28 02:09 |
| naive | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | off | max_tokens_desc | - | HF | off | - | 107.85 | 17.47 | 78.1 | 0.0 | - | - | 28.0 | 95.7 | 1.5 | 6.7 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | off | - | - | compiled | on | - | 8.11 | 1.31 | 75.0 | - | 526 | 15.4 | 101.8 | 97.0 | 20.1 | 51.1 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | eager | off | - | 13.28 | 2.15 | 76.6 | - | 518 | 25.6 | 103.3 | 49.6 | 7.3 | 31.1 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | 512 | 6.18 | 1.00 | 75.0 | - | 519 | 11.9 | 103.1 | 94.8 | 15.8 | 66.9 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | - | 6.17 | 1.00 | 75.0 | - | 518 | 11.9 | 103.3 | 94.6 | 15.8 | 66.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | decode_ratio_desc | 16,384 | fused | decode 99% | - | 6.67 | 1.08 | 76.6 | 64.8 | 513 | 13.0 | 104.3 | 91.8 | 14.6 | 61.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | input | 16,384 | fused | decode 99% | - | 6.70 | 1.09 | 78.1 | 64.8 | 514 | 13.0 | 104.1 | 91.7 | 14.6 | 61.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | max_tokens_desc | 16,384 | fused | decode 99% | - | 6.58 | 1.07 | 76.6 | 64.8 | 513 | 12.8 | 104.3 | 91.7 | 14.8 | 62.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 100% | - | 10.43 | 1.69 | 79.7 | 64.8 | 1329 | 7.9 | 40.3 | 93.2 | 9.3 | 56.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 6.66 | 1.08 | 76.6 | 64.8 | 513 | 13.0 | 104.3 | 91.8 | 14.6 | 61.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | HF | off | - | 19.01 | 3.08 | 76.6 | 64.8 | 513 | 37.1 | 104.3 | 53.4 | 5.1 | 21.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 256 | fused | off | - | 8.59 | 1.39 | 73.4 | 64.8 | 601 | 14.3 | 89.0 | 82.0 | 11.3 | 50.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | off | - | 7.69 | 1.25 | 76.6 | 64.8 | 513 | 15.0 | 104.3 | 83.2 | 12.7 | 53.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 256 | fused | decode 75% | - | 7.48 | 1.21 | 73.4 | 64.8 | 601 | 12.4 | 89.0 | 89.1 | 13.0 | 57.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 256 | fused | decode 75% | - | 7.46 | 1.21 | 73.4 | 64.8 | 601 | 12.4 | 89.0 | 89.6 | 13.1 | 57.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 6.68 | 1.08 | 76.6 | 64.8 | 513 | 13.0 | 104.3 | 91.3 | 14.6 | 61.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | A100-SXM4-40GB | off | prefix_dfs | 16,384 | fused | decode 99% | - | 8.52 | 1.38 | 76.6 | 0.0 | 513 | 16.6 | 104.3 | 93.2 | 19.2 | 48.3 | 0d19c51 · 09-28 02:09 |
| naive | generate-quick | Qwen3-1.7B | H100-PCIe | off | max_tokens_desc | - | HF | off | - | 126.40 | 31.40 | 75.0 | 0.0 | - | - | 28.0 | 57.9 | 0.5 | 4.4 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-1.7B | H100-PCIe | off | - | - | compiled | on | - | 5.24 | 1.30 | 78.1 | - | 520 | 10.1 | 102.9 | 93.3 | 12.9 | 61.4 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | eager | off | - | 6.71 | 1.67 | 76.6 | - | 518 | 13.0 | 103.3 | 59.9 | 6.0 | 47.8 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | compiled | on | 512 | 4.03 | 1.00 | 78.1 | - | 519 | 7.8 | 103.1 | 91.6 | 10.0 | 79.8 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | compiled | on | - | 4.03 | 1.00 | 78.1 | - | 518 | 7.8 | 103.3 | 90.2 | 10.0 | 79.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | global trie | decode_ratio_desc | 16,384 | fused | decode 99% | - | 4.99 | 1.24 | 75.0 | 64.8 | 513 | 9.7 | 104.3 | 93.9 | 8.1 | 64.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | global trie | input | 16,384 | fused | decode 99% | - | 5.09 | 1.26 | 75.0 | 64.8 | 514 | 9.9 | 104.1 | 89.9 | 7.9 | 63.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | global trie | max_tokens_desc | 16,384 | fused | decode 99% | - | 4.91 | 1.22 | 75.0 | 64.8 | 513 | 9.6 | 104.3 | 94.3 | 8.2 | 65.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 100% | - | 7.74 | 1.92 | 75.0 | 64.8 | 1329 | 5.8 | 40.3 | 93.8 | 5.2 | 59.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 4.99 | 1.24 | 75.0 | 64.8 | 513 | 9.7 | 104.3 | 92.5 | 8.1 | 64.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | HF | off | - | 10.49 | 2.61 | 75.0 | 64.8 | 513 | 20.5 | 104.3 | 75.2 | 3.8 | 30.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 256 | fused | off | - | 6.26 | 1.55 | 78.1 | 64.8 | 601 | 10.4 | 89.0 | 81.8 | 6.4 | 53.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | off | - | 6.14 | 1.53 | 75.0 | 64.8 | 513 | 12.0 | 104.3 | 79.1 | 6.5 | 52.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 256 | fused · FA3 | off | - | 5.42 | 1.35 | 75.0 | 64.8 | 601 | 9.0 | 89.0 | 79.2 | 7.4 | 61.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused · FA3 | off | - | 5.06 | 1.26 | 75.0 | 64.8 | 513 | 9.9 | 104.3 | 79.5 | 8.0 | 63.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 256 | fused | decode 75% | - | 5.52 | 1.37 | 78.1 | 64.8 | 601 | 9.2 | 89.0 | 89.6 | 7.3 | 60.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 256 | fused | decode 75% | - | 5.51 | 1.37 | 78.1 | 64.8 | 601 | 9.2 | 89.0 | 91.3 | 7.3 | 60.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 5.08 | 1.26 | 75.0 | 64.8 | 513 | 9.9 | 104.3 | 89.3 | 7.9 | 63.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-1.7B | H100-PCIe | off | prefix_dfs | 16,384 | fused | decode 99% | - | 6.08 | 1.51 | 78.1 | 0.0 | 513 | 11.8 | 104.3 | 93.2 | 11.1 | 52.7 | 0d19c51 · 09-28 02:09 |
| naive | generate-quick | Qwen3-8B | A100-SXM4-40GB | off | max_tokens_desc | - | HF | off | - | 231.81 | 15.66 | 90.6 | 0.0 | - | - | 28.0 | 98.4 | 3.3 | 9.7 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-8B | A100-SXM4-40GB | off | - | - | compiled | on | - | 20.39 | 1.38 | 89.1 | - | 526 | 38.8 | 101.8 | 98.6 | 37.1 | 43.9 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | eager | off | - | 17.84 | 1.21 | 87.5 | - | 518 | 34.4 | 103.3 | 85.1 | 24.6 | 49.8 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | 512 | 14.82 | 1.00 | 89.1 | - | 518 | 28.6 | 103.3 | 98.7 | 29.6 | 59.9 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | - | 14.80 | 1.00 | 89.1 | - | 518 | 28.6 | 103.3 | 99.2 | 29.7 | 60.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | A100-SXM4-40GB | global trie | decode_ratio_desc | 16,384 | fused | decode 99% | - | 15.22 | 1.03 | 87.5 | 64.8 | 513 | 29.7 | 104.3 | 96.3 | 28.8 | 58.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | A100-SXM4-40GB | global trie | input | 16,384 | fused | decode 99% | - | 15.27 | 1.03 | 89.1 | 64.8 | 514 | 29.7 | 104.1 | 96.2 | 28.7 | 57.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | A100-SXM4-40GB | global trie | max_tokens_desc | 16,384 | fused | decode 99% | - | 15.12 | 1.02 | 87.5 | 64.8 | 513 | 29.5 | 104.3 | 96.3 | 29.0 | 58.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 100% | - | 28.38 | 1.92 | 87.5 | 64.8 | 1527 | 18.6 | 35.1 | 97.3 | 15.5 | 65.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 15.23 | 1.03 | 87.5 | 64.8 | 513 | 29.7 | 104.3 | 96.2 | 28.8 | 58.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | HF | off | - | 26.35 | 1.78 | 90.6 | 64.8 | 513 | 51.4 | 104.3 | 76.6 | 16.7 | 33.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 256 | fused | off | - | 18.16 | 1.23 | 90.6 | 64.8 | 601 | 30.2 | 89.0 | 92.4 | 24.2 | 53.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | off | - | 16.22 | 1.10 | 87.5 | 64.8 | 513 | 31.6 | 104.3 | 91.8 | 27.1 | 54.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 256 | fused | decode 75% | - | 17.37 | 1.17 | 90.6 | 64.8 | 601 | 28.9 | 89.0 | 95.3 | 25.3 | 55.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 256 | fused | decode 75% | - | 17.34 | 1.17 | 90.6 | 64.8 | 601 | 28.8 | 89.0 | 95.4 | 25.3 | 55.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 15.22 | 1.03 | 87.5 | 64.8 | 513 | 29.7 | 104.3 | 96.2 | 28.8 | 58.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | A100-SXM4-40GB | off | prefix_dfs | 16,384 | fused | decode 98% | - | 21.60 | 1.46 | 90.6 | 0.0 | 561 | 38.5 | 95.4 | 97.1 | 35.0 | 43.0 | 0d19c51 · 09-28 02:09 |
| naive | generate-quick | Qwen3-8B | H100-PCIe | off | max_tokens_desc | - | HF | off | - | 211.77 | 22.77 | 90.6 | 0.0 | - | - | 28.0 | 75.9 | 1.5 | 8.2 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-8B | H100-PCIe | off | - | - | compiled | on | - | 12.71 | 1.37 | 89.1 | - | 520 | 24.5 | 102.9 | 97.0 | 24.5 | 54.4 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | eager | off | - | 9.95 | 1.07 | 89.1 | - | 516 | 19.3 | 103.7 | 96.8 | 18.2 | 69.3 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | compiled | on | 512 | 9.37 | 1.01 | 89.1 | - | 516 | 18.2 | 103.7 | 96.0 | 19.3 | 73.5 | 0d19c51 · 09-28 02:09 |
| vllm | generate-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | compiled | on | - | 9.30 | 1.00 | 89.1 | - | 516 | 18.0 | 103.7 | 97.5 | 19.5 | 74.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | global trie | decode_ratio_desc | 16,384 | fused | decode 99% | - | 10.70 | 1.15 | 90.6 | 64.8 | 513 | 20.9 | 104.3 | 96.3 | 16.9 | 64.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | global trie | input | 16,384 | fused | decode 99% | - | 10.79 | 1.16 | 90.6 | 64.8 | 514 | 21.0 | 104.1 | 96.8 | 16.8 | 63.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | global trie | max_tokens_desc | 16,384 | fused | decode 99% | - | 10.61 | 1.14 | 90.6 | 64.8 | 513 | 20.7 | 104.3 | 97.2 | 17.1 | 64.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 100% | - | 18.74 | 2.01 | 90.6 | 64.8 | 1329 | 14.1 | 40.3 | 97.5 | 9.7 | 69.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 10.69 | 1.15 | 90.6 | 64.8 | 513 | 20.8 | 104.3 | 96.2 | 16.9 | 64.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | HF | off | - | 16.01 | 1.72 | 89.1 | 64.8 | 513 | 31.2 | 104.3 | 90.5 | 11.3 | 42.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 256 | fused | off | - | 13.07 | 1.41 | 90.6 | 64.8 | 601 | 21.7 | 89.0 | 91.6 | 13.9 | 57.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | off | - | 11.62 | 1.25 | 90.6 | 64.8 | 513 | 22.7 | 104.3 | 91.6 | 15.6 | 59.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 256 | fused · FA3 | off | - | 11.15 | 1.20 | 90.6 | 64.8 | 601 | 18.6 | 89.0 | 90.3 | 16.2 | 67.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused · FA3 | off | - | 10.67 | 1.15 | 90.6 | 64.8 | 513 | 20.8 | 104.3 | 89.0 | 17.0 | 64.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 256 | fused | decode 75% | - | 12.35 | 1.33 | 90.6 | 64.8 | 601 | 20.6 | 89.0 | 95.0 | 14.7 | 61.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 256 | fused | decode 75% | - | 12.33 | 1.33 | 90.6 | 64.8 | 601 | 20.5 | 89.0 | 95.6 | 14.7 | 61.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 99% | - | 10.69 | 1.15 | 90.6 | 64.8 | 513 | 20.8 | 104.3 | 96.1 | 16.9 | 64.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | generate-quick | Qwen3-8B | H100-PCIe | off | prefix_dfs | 16,384 | fused | decode 99% | - | 13.85 | 1.49 | 90.6 | 0.0 | 513 | 27.0 | 104.3 | 96.9 | 22.5 | 49.6 | 0d19c51 · 09-28 02:09 |
| naive | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | off | max_tokens_desc | - | HF | off | - | 23.47 | 2.99 | - | 0.0 | - | - | 53.3 | 92.8 | 14.3 | 10.5 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | off | - | - | compiled | on | - | 7.87 | 1.00 | - | - | 139 | 56.6 | 146.1 | 97.5 | 42.7 | 24.4 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | eager | off | - | 8.68 | 1.10 | - | - | 139 | 62.4 | 146.1 | 89.7 | 38.8 | 22.1 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | 512 | 8.05 | 1.02 | - | - | 137 | 58.8 | 148.2 | 98.4 | 41.8 | 23.8 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | - | 7.86 | 1.00 | - | - | 139 | 56.5 | 146.1 | 97.3 | 42.8 | 24.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | decode_ratio_desc | 16,384 | fused | decode 83% | - | 8.13 | 1.03 | - | 0.0 | 131 | 62.1 | 155.1 | 93.5 | 41.4 | 23.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | input | 16,384 | fused | decode 83% | - | 8.12 | 1.03 | - | 0.0 | 131 | 62.0 | 155.1 | 93.4 | 41.4 | 23.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | max_tokens_desc | 16,384 | fused | decode 83% | - | 8.15 | 1.04 | - | 0.0 | 131 | 62.2 | 155.1 | 93.3 | 41.3 | 23.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 93% | - | 8.81 | 1.12 | - | 0.0 | 340 | 25.9 | 60.2 | 92.9 | 38.2 | 26.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 82% | - | 8.23 | 1.05 | - | 0.0 | 131 | 62.9 | 155.1 | 92.9 | 40.8 | 23.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | HF | off | - | 12.66 | 1.61 | - | 0.0 | 130 | 97.4 | 156.3 | 87.0 | 26.6 | 15.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 5,267 | fused | off | - | 8.99 | 1.14 | - | 0.0 | 136 | 66.1 | 149.3 | 92.4 | 37.4 | 21.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | off | - | 8.30 | 1.06 | - | 0.0 | 130 | 63.9 | 156.3 | 91.8 | 40.5 | 22.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 5,267 | fused | decode 53% | - | 8.88 | 1.13 | - | 0.0 | 137 | 64.8 | 148.2 | 93.1 | 37.9 | 21.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 5,267 | fused | decode 53% | - | 8.87 | 1.13 | - | 0.0 | 137 | 64.7 | 148.2 | 93.2 | 37.9 | 21.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 83% | - | 8.17 | 1.04 | - | 0.0 | 131 | 62.4 | 155.1 | 92.9 | 41.2 | 23.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | A100-SXM4-40GB | off | prefix_dfs | 16,384 | fused | decode 83% | - | 8.13 | 1.03 | - | 0.0 | 131 | 62.1 | 155.1 | 93.3 | 41.4 | 23.4 | 0d19c51 · 09-28 02:09 |
| naive | sweep0-quick | Qwen3-1.7B | H100-PCIe | off | max_tokens_desc | - | HF | off | - | 21.75 | 4.76 | - | 0.0 | - | - | 53.3 | 69.3 | 6.4 | 8.8 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-1.7B | H100-PCIe | off | - | - | compiled | on | - | 4.58 | 1.00 | - | - | 89 | 51.4 | 229.1 | 94.3 | 30.3 | 30.8 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | eager | off | - | 4.91 | 1.07 | - | - | 90 | 54.5 | 226.5 | 89.1 | 28.3 | 28.7 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | compiled | on | 512 | 4.55 | 1.00 | - | - | 87 | 52.3 | 234.4 | 91.5 | 30.5 | 30.9 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | compiled | on | - | 4.57 | 1.00 | - | - | 87 | 52.5 | 234.4 | 93.1 | 30.4 | 30.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | global trie | decode_ratio_desc | 16,384 | fused | decode 75% | - | 5.11 | 1.12 | - | 0.0 | 84 | 60.8 | 242.9 | 92.5 | 27.2 | 27.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | global trie | input | 16,384 | fused | decode 75% | - | 5.10 | 1.12 | - | 0.0 | 84 | 60.8 | 242.9 | 93.5 | 27.2 | 27.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | global trie | max_tokens_desc | 16,384 | fused | decode 75% | - | 5.11 | 1.12 | - | 0.0 | 84 | 60.8 | 242.9 | 89.9 | 27.2 | 27.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 93% | - | 5.87 | 1.29 | - | 0.0 | 340 | 17.3 | 60.2 | 91.1 | 23.6 | 31.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 74% | - | 5.14 | 1.13 | - | 0.0 | 85 | 60.5 | 240.0 | 92.3 | 27.0 | 27.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | HF | off | - | 7.73 | 1.69 | - | 0.0 | 84 | 92.0 | 242.9 | 92.5 | 18.0 | 18.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 5,267 | fused | off | - | 5.75 | 1.26 | - | 0.0 | 126 | 45.6 | 161.3 | 91.5 | 24.1 | 25.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | off | - | 5.20 | 1.14 | - | 0.0 | 84 | 61.9 | 242.9 | 91.7 | 26.7 | 26.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 5,267 | fused · FA3 | off | - | 5.02 | 1.10 | - | 0.0 | 126 | 39.8 | 161.3 | 88.2 | 27.7 | 29.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused · FA3 | off | - | 4.92 | 1.08 | - | 0.0 | 84 | 58.6 | 242.9 | 87.8 | 28.2 | 28.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 5,267 | fused | decode 50% | - | 5.67 | 1.24 | - | 0.0 | 126 | 45.0 | 161.3 | 91.3 | 24.5 | 26.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 5,267 | fused | decode 50% | - | 5.72 | 1.25 | - | 0.0 | 126 | 45.4 | 161.3 | 90.6 | 24.3 | 25.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 75% | - | 5.13 | 1.12 | - | 0.0 | 84 | 61.1 | 242.9 | 92.1 | 27.0 | 27.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-1.7B | H100-PCIe | off | prefix_dfs | 16,384 | fused | decode 75% | - | 5.11 | 1.12 | - | 0.0 | 84 | 60.8 | 242.9 | 92.2 | 27.2 | 27.4 | 0d19c51 · 09-28 02:09 |
| naive | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | off | max_tokens_desc | - | HF | off | - | 55.59 | 1.98 | - | 0.0 | - | - | 53.3 | 97.8 | 29.0 | 10.5 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | off | - | - | compiled | on | - | 28.41 | 1.01 | - | - | 203 | 139.9 | 99.7 | 99.6 | 56.7 | 14.3 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | eager | off | - | 29.23 | 1.04 | - | - | 202 | 144.7 | 100.2 | 98.1 | 55.1 | 13.8 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | 512 | 28.15 | 1.00 | - | - | 203 | 138.7 | 99.8 | 99.6 | 57.3 | 14.4 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | - | 28.12 | 1.00 | - | - | 203 | 138.5 | 99.8 | 99.6 | 57.3 | 14.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | global trie | decode_ratio_desc | 16,384 | fused | decode 89% | - | 26.97 | 0.96 | - | 0.0 | 196 | 137.6 | 103.4 | 97.9 | 59.8 | 14.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | global trie | input | 16,384 | fused | decode 89% | - | 26.97 | 0.96 | - | 0.0 | 196 | 137.6 | 103.4 | 97.8 | 59.8 | 14.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | global trie | max_tokens_desc | 16,384 | fused | decode 89% | - | 26.95 | 0.96 | - | 0.0 | 196 | 137.5 | 103.4 | 97.9 | 59.8 | 14.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 93% | - | 28.42 | 1.01 | - | 0.0 | 340 | 83.6 | 60.2 | 97.8 | 56.7 | 18.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 88% | - | 27.13 | 0.96 | - | 0.0 | 196 | 138.4 | 103.4 | 97.8 | 59.4 | 14.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | HF | off | - | 35.80 | 1.27 | - | 0.0 | 196 | 182.6 | 103.4 | 95.6 | 45.0 | 11.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 5,267 | fused | off | - | 28.95 | 1.03 | - | 0.0 | 204 | 141.9 | 99.3 | 97.4 | 55.7 | 14.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | off | - | 27.32 | 0.97 | - | 0.0 | 196 | 139.4 | 103.4 | 97.2 | 59.0 | 14.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 5,267 | fused | decode 69% | - | 28.71 | 1.02 | - | 0.0 | 205 | 140.0 | 98.8 | 97.8 | 56.1 | 14.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 5,267 | fused | decode 69% | - | 28.71 | 1.02 | - | 0.0 | 205 | 140.1 | 98.8 | 97.9 | 56.1 | 14.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 89% | - | 27.02 | 0.96 | - | 0.0 | 196 | 137.8 | 103.4 | 97.8 | 59.7 | 14.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | A100-SXM4-40GB | off | prefix_dfs | 16,384 | fused | decode 89% | - | 26.98 | 0.96 | - | 0.0 | 196 | 137.6 | 103.4 | 97.9 | 59.7 | 14.8 | 0d19c51 · 09-28 02:09 |
| naive | sweep0-quick | Qwen3-8B | H100-PCIe | off | max_tokens_desc | - | HF | off | - | 41.71 | 2.84 | - | 0.0 | - | - | 53.3 | 86.7 | 15.9 | 10.8 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-8B | H100-PCIe | off | - | - | compiled | on | - | 14.91 | 1.01 | - | - | 86 | 173.3 | 237.2 | 97.7 | 44.6 | 15.2 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | eager | off | - | 15.13 | 1.03 | - | - | 85 | 178.0 | 240.0 | 97.4 | 44.0 | 14.9 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | compiled | on | 512 | 14.82 | 1.01 | - | - | 86 | 172.3 | 237.2 | 97.3 | 44.9 | 15.3 | 0d19c51 · 09-28 02:09 |
| vllm | sweep0-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | compiled | on | - | 14.69 | 1.00 | - | - | 86 | 170.8 | 237.2 | 97.8 | 45.3 | 15.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | global trie | decode_ratio_desc | 16,384 | fused | decode 75% | - | 16.06 | 1.09 | - | 0.0 | 84 | 191.2 | 242.9 | 97.1 | 41.4 | 14.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | global trie | input | 16,384 | fused | decode 75% | - | 16.07 | 1.09 | - | 0.0 | 84 | 191.3 | 242.9 | 96.8 | 41.4 | 14.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | global trie | max_tokens_desc | 16,384 | fused | decode 75% | - | 16.07 | 1.09 | - | 0.0 | 84 | 191.3 | 242.9 | 97.7 | 41.4 | 14.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 93% | - | 17.89 | 1.22 | - | 0.0 | 340 | 52.6 | 60.2 | 97.6 | 37.2 | 23.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 74% | - | 16.13 | 1.10 | - | 0.0 | 85 | 189.7 | 240.0 | 96.9 | 41.2 | 14.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | HF | off | - | 21.25 | 1.45 | - | 0.0 | 84 | 252.9 | 242.9 | 97.2 | 31.3 | 10.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 5,267 | fused | off | - | 18.02 | 1.23 | - | 0.0 | 126 | 143.0 | 161.3 | 97.3 | 36.9 | 14.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | off | - | 16.15 | 1.10 | - | 0.0 | 84 | 192.3 | 242.9 | 96.8 | 41.2 | 13.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 5,267 | fused · FA3 | off | - | 15.30 | 1.04 | - | 0.0 | 126 | 121.4 | 161.3 | 97.1 | 43.5 | 16.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused · FA3 | off | - | 14.91 | 1.01 | - | 0.0 | 84 | 177.5 | 242.9 | 95.6 | 44.6 | 15.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 5,267 | fused | decode 50% | - | 17.95 | 1.22 | - | 0.0 | 126 | 142.4 | 161.3 | 97.2 | 37.1 | 14.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 5,267 | fused | decode 50% | - | 17.93 | 1.22 | - | 0.0 | 126 | 142.3 | 161.3 | 97.5 | 37.1 | 14.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 75% | - | 16.08 | 1.09 | - | 0.0 | 84 | 191.4 | 242.9 | 97.5 | 41.4 | 14.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep0-quick | Qwen3-8B | H100-PCIe | off | prefix_dfs | 16,384 | fused | decode 75% | - | 16.07 | 1.09 | - | 0.0 | 84 | 191.4 | 242.9 | 97.7 | 41.4 | 14.0 | 0d19c51 · 09-28 02:09 |
| naive | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | off | max_tokens_desc | - | HF | off | - | 23.91 | 7.91 | - | 0.0 | - | - | 53.3 | 95.5 | 14.1 | 10.3 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | off | - | - | compiled | on | - | 7.92 | 2.62 | - | - | 139 | 57.0 | 146.1 | 95.2 | 42.7 | 24.4 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | eager | off | - | 4.05 | 1.34 | - | - | 141 | 28.7 | 144.0 | 74.1 | 16.5 | 47.8 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | 512 | 2.83 | 0.94 | - | - | 116 | 24.4 | 175.3 | 95.9 | 23.5 | 66.3 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | - | 3.02 | 1.00 | - | - | 143 | 21.1 | 142.0 | 92.0 | 22.0 | 64.1 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | decode_ratio_desc | 16,384 | fused | decode 94% | - | 2.69 | 0.89 | - | 86.8 | 67 | 40.2 | 305.4 | 86.4 | 24.8 | 65.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | input | 16,384 | fused | decode 94% | - | 2.69 | 0.89 | - | 86.8 | 67 | 40.1 | 305.4 | 86.9 | 24.8 | 65.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | max_tokens_desc | 16,384 | fused | decode 94% | - | 2.69 | 0.89 | - | 86.8 | 67 | 40.1 | 305.4 | 86.4 | 24.8 | 65.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 97% | - | 3.85 | 1.27 | - | 86.8 | 325 | 11.9 | 63.0 | 88.8 | 17.3 | 60.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 94% | - | 2.70 | 0.89 | - | 86.8 | 67 | 40.2 | 305.4 | 86.2 | 24.7 | 65.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | HF | off | - | 3.74 | 1.24 | - | 86.8 | 67 | 55.9 | 305.4 | 84.2 | 17.8 | 47.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 5,296 | fused | off | - | 2.90 | 0.96 | - | 86.8 | 72 | 40.3 | 283.9 | 84.7 | 23.0 | 61.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | off | - | 2.81 | 0.93 | - | 86.8 | 67 | 41.9 | 305.4 | 84.1 | 23.7 | 63.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 5,296 | fused | decode 88% | - | 2.79 | 0.92 | - | 86.8 | 72 | 38.8 | 283.9 | 86.6 | 23.9 | 63.8 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 5,296 | fused | decode 88% | - | 2.79 | 0.92 | - | 86.8 | 72 | 38.7 | 283.9 | 86.4 | 23.9 | 63.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 94% | - | 2.70 | 0.89 | - | 86.8 | 67 | 40.3 | 305.4 | 86.3 | 24.7 | 65.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | A100-SXM4-40GB | off | prefix_dfs | 16,384 | fused | decode 83% | - | 8.12 | 2.68 | - | 0.0 | 131 | 62.0 | 155.1 | 93.6 | 41.7 | 23.6 | 0d19c51 · 09-28 02:09 |
| naive | sweep90-quick | Qwen3-1.7B | H100-PCIe | off | max_tokens_desc | - | HF | off | - | 22.00 | 8.66 | - | 0.0 | - | - | 53.3 | 71.5 | 6.3 | 8.7 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-1.7B | H100-PCIe | off | - | - | compiled | on | - | 4.57 | 1.80 | - | - | 89 | 51.3 | 229.1 | 94.2 | 30.5 | 31.0 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | eager | off | - | 3.37 | 1.33 | - | - | 121 | 27.8 | 168.0 | 47.3 | 8.2 | 43.6 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | compiled | on | 512 | 2.58 | 1.02 | - | - | 119 | 21.7 | 170.8 | 55.0 | 10.6 | 56.7 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-1.7B | H100-PCIe | prefix cache | - | - | compiled | on | - | 2.54 | 1.00 | - | - | 123 | 20.7 | 165.2 | 57.8 | 10.8 | 58.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | global trie | decode_ratio_desc | 16,384 | fused | decode 94% | - | 2.01 | 0.79 | - | 86.8 | 67 | 30.0 | 305.4 | 81.9 | 13.7 | 68.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | global trie | input | 16,384 | fused | decode 94% | - | 2.01 | 0.79 | - | 86.8 | 67 | 30.0 | 305.4 | 81.3 | 13.7 | 68.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | global trie | max_tokens_desc | 16,384 | fused | decode 94% | - | 2.01 | 0.79 | - | 86.8 | 67 | 30.0 | 305.4 | 79.5 | 13.7 | 68.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 97% | - | 2.91 | 1.15 | - | 86.8 | 325 | 9.0 | 63.0 | 85.0 | 9.4 | 62.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 94% | - | 2.01 | 0.79 | - | 86.8 | 67 | 30.0 | 305.4 | 81.1 | 13.7 | 68.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | HF | off | - | 2.68 | 1.06 | - | 86.8 | 67 | 40.0 | 305.4 | 80.9 | 10.3 | 51.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 5,296 | fused | off | - | 2.14 | 0.84 | - | 86.8 | 72 | 29.8 | 283.9 | 80.6 | 12.8 | 64.6 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | off | - | 2.14 | 0.84 | - | 86.8 | 67 | 32.0 | 305.4 | 78.4 | 12.8 | 64.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 5,296 | fused · FA3 | off | - | 1.79 | 0.70 | - | 86.8 | 72 | 24.8 | 283.9 | 82.4 | 15.4 | 77.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused · FA3 | off | - | 1.75 | 0.69 | - | 86.8 | 67 | 26.2 | 305.4 | 82.4 | 15.7 | 78.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | adaptive from 5,296 | fused | decode 88% | - | 2.07 | 0.81 | - | 86.8 | 72 | 28.7 | 283.9 | 85.2 | 13.3 | 66.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 5,296 | fused | decode 88% | - | 2.07 | 0.81 | - | 86.8 | 72 | 28.7 | 283.9 | 85.7 | 13.3 | 67.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 94% | - | 2.01 | 0.79 | - | 86.8 | 67 | 30.0 | 305.4 | 86.1 | 13.7 | 68.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-1.7B | H100-PCIe | off | prefix_dfs | 16,384 | fused | decode 75% | - | 5.12 | 2.02 | - | 0.0 | 84 | 61.0 | 242.9 | 92.4 | 27.2 | 27.4 | 0d19c51 · 09-28 02:09 |
| naive | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | off | max_tokens_desc | - | HF | off | - | 58.17 | 8.48 | - | 0.0 | - | - | 53.3 | 98.5 | 27.9 | 10.0 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | off | - | - | compiled | on | - | 28.39 | 4.14 | - | - | 203 | 139.9 | 99.8 | 99.5 | 57.1 | 14.3 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | eager | off | - | 7.61 | 1.11 | - | - | 133 | 57.2 | 152.7 | 90.8 | 40.2 | 44.4 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | 512 | 6.28 | 0.92 | - | - | 72 | 87.2 | 283.9 | 97.4 | 48.7 | 44.4 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | prefix cache | - | - | compiled | on | - | 6.86 | 1.00 | - | - | 133 | 51.6 | 152.7 | 97.0 | 44.6 | 49.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | global trie | decode_ratio_desc | 16,384 | fused | decode 94% | - | 6.49 | 0.95 | - | 86.8 | 67 | 96.9 | 305.4 | 94.2 | 47.1 | 42.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | global trie | input | 16,384 | fused | decode 94% | - | 6.49 | 0.95 | - | 86.8 | 67 | 96.8 | 305.4 | 94.1 | 47.1 | 42.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | global trie | max_tokens_desc | 16,384 | fused | decode 94% | - | 6.49 | 0.95 | - | 86.8 | 67 | 96.9 | 305.4 | 94.2 | 47.1 | 42.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 97% | - | 9.37 | 1.37 | - | 86.8 | 325 | 28.8 | 63.0 | 95.1 | 32.6 | 56.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 94% | - | 6.48 | 0.95 | - | 86.8 | 67 | 96.7 | 305.4 | 94.0 | 47.2 | 42.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | HF | off | - | 8.29 | 1.21 | - | 86.8 | 67 | 123.8 | 305.4 | 93.2 | 36.9 | 33.0 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 5,296 | fused | off | - | 6.84 | 1.00 | - | 86.8 | 72 | 95.0 | 283.9 | 92.9 | 44.7 | 40.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | off | - | 6.59 | 0.96 | - | 86.8 | 67 | 98.4 | 305.4 | 92.7 | 46.4 | 41.5 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | adaptive from 5,296 | fused | decode 88% | - | 6.74 | 0.98 | - | 86.8 | 72 | 93.6 | 283.9 | 94.0 | 45.4 | 41.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 5,296 | fused | decode 88% | - | 6.74 | 0.98 | - | 86.8 | 72 | 93.7 | 283.9 | 94.2 | 45.3 | 41.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | global trie | prefix_dfs | 16,384 | fused | decode 94% | - | 6.49 | 0.95 | - | 86.8 | 67 | 96.9 | 305.4 | 94.1 | 47.1 | 42.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | A100-SXM4-40GB | off | prefix_dfs | 16,384 | fused | decode 89% | - | 27.07 | 3.95 | - | 0.0 | 196 | 138.1 | 103.4 | 98.0 | 59.9 | 14.8 | 0d19c51 · 09-28 02:09 |
| naive | sweep90-quick | Qwen3-8B | H100-PCIe | off | max_tokens_desc | - | HF | off | - | 43.59 | 10.96 | - | 0.0 | - | - | 53.3 | 87.2 | 15.3 | 10.4 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-8B | H100-PCIe | off | - | - | compiled | on | - | 15.03 | 3.78 | - | - | 86 | 174.7 | 237.2 | 98.3 | 44.5 | 15.1 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | eager | off | - | 4.12 | 1.03 | - | - | 83 | 49.6 | 245.8 | 91.7 | 30.7 | 54.7 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | compiled | on | 512 | 3.98 | 1.00 | - | - | 83 | 48.0 | 245.8 | 92.8 | 31.7 | 56.5 | 0d19c51 · 09-28 02:09 |
| vllm | sweep90-quick | Qwen3-8B | H100-PCIe | prefix cache | - | - | compiled | on | - | 3.98 | 1.00 | - | - | 84 | 47.4 | 242.9 | 91.8 | 31.7 | 56.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | global trie | decode_ratio_desc | 16,384 | fused | decode 94% | - | 4.40 | 1.11 | - | 86.8 | 67 | 65.7 | 305.4 | 94.6 | 28.7 | 48.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | global trie | input | 16,384 | fused | decode 94% | - | 4.40 | 1.11 | - | 86.8 | 67 | 65.6 | 305.4 | 91.6 | 28.7 | 48.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | global trie | max_tokens_desc | 16,384 | fused | decode 94% | - | 4.40 | 1.11 | - | 86.8 | 67 | 65.6 | 305.4 | 92.6 | 28.7 | 48.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 97% | - | 6.78 | 1.70 | - | 86.8 | 325 | 20.9 | 63.0 | 95.0 | 18.6 | 60.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 94% | - | 4.40 | 1.10 | - | 86.8 | 67 | 65.6 | 305.4 | 92.8 | 28.7 | 48.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | HF | off | - | 5.65 | 1.42 | - | 86.8 | 67 | 84.3 | 305.4 | 93.1 | 22.3 | 37.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 5,296 | fused | off | - | 4.64 | 1.17 | - | 86.8 | 72 | 64.4 | 283.9 | 91.4 | 27.2 | 46.7 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | off | - | 4.51 | 1.13 | - | 86.8 | 67 | 67.3 | 305.4 | 90.8 | 28.0 | 47.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 5,296 | fused · FA3 | off | - | 4.02 | 1.01 | - | 86.8 | 72 | 55.8 | 283.9 | 91.8 | 31.4 | 53.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused · FA3 | off | - | 3.95 | 0.99 | - | 86.8 | 67 | 59.0 | 305.4 | 91.5 | 31.9 | 53.9 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | adaptive from 5,296 | fused | decode 88% | - | 4.59 | 1.15 | - | 86.8 | 72 | 63.8 | 283.9 | 93.6 | 27.5 | 47.2 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 5,296 | fused | decode 88% | - | 4.57 | 1.15 | - | 86.8 | 72 | 63.5 | 283.9 | 93.1 | 27.6 | 47.4 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | global trie | prefix_dfs | 16,384 | fused | decode 94% | - | 4.41 | 1.11 | - | 86.8 | 67 | 65.9 | 305.4 | 94.2 | 28.6 | 48.3 | 0d19c51 · 09-28 02:09 |
| batchinfer | sweep90-quick | Qwen3-8B | H100-PCIe | off | prefix_dfs | 16,384 | fused | decode 75% | - | 16.14 | 4.06 | - | 0.0 | 84 | 192.2 | 242.9 | 97.6 | 41.4 | 14.0 | 0d19c51 · 09-28 02:09 |
