# Reproducing the benchmarks

Every number in the README comes from a row under `results/` (`results/README.md` describes what a row holds). This
file says what the rows ran on and which commands make each table.

## Setup

```
git clone https://github.com/zhebrak/batchinfer && cd batchinfer
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PATH=$PWD/.venv/bin:$PATH   # FlashInfer's JIT-built kernels (vLLM's top-k sampler among them) need its ninja
```

Everything runs from the repository root. Models and datasets download on first use into `~/.cache/huggingface`
and `~/.cache/bench`; no Hugging Face token is needed. Without the venv's `bin` on the `PATH`, vLLM dies at warmup
with `FileNotFoundError: ... 'ninja'`.

## Measured on

| | H100 host | A100 host |
|---|---|---|
| GPU | H100 PCIe, 80 GB | A100-SXM4, 40 GB |
| CPU | Intel Xeon Platinum 8480+, 26 vCPUs | AMD EPYC 7J13, 30 vCPUs |
| RAM | 221 GiB | 216 GiB |
| OS, NVIDIA driver | Ubuntu 24.04, 580.105.08 | Ubuntu 24.04, 580.105.08 |

Both ran Python 3.12.3, torch 2.13.0 (CUDA 13.0), transformers 5.17.0 and vLLM 0.30.0; every row records its
versions, GPU and commit. The host matters: with HF's layers (and vLLM `enforce_eager`) a step is bound by kernel
launches, so a slower CPU slows those rows most, while the defaults' steps are GPU-bound. Timings are only valid with
one job on the GPU.

## What makes each table

Build the five workloads (`bench.suite` builds any that are missing):

```
python -m bench.workload build --preset <mixed|classify|generate|sweep0|sweep90> --size quick --model Qwen/Qwen3-8B
```

The datasets are downloaded from the Hugging Face Hub at their latest revision (`bench/sources.py`), so check each
file against the hash the rows recorded (`workload_sha256` in `metrics.json`); a different hash is a different job.
The prompts are rendered with Qwen3-8B's chat template, which Qwen3-1.7B shares, so one file serves both models.

| Workload file | sha256 |
|---|---|
| `mixed-quick.jsonl` | `54c82627830a6cceffdcbc136e74ae7b812f097605846e613149a9d117ddc9f5` |
| `classify-quick.jsonl` | `c49ce11968ac90b739dc2456b7a1295f3f7974dd1cd838995b624cfda1bbca89` |
| `generate-quick.jsonl` | `6c20a3e6b76372d190a7557f83b1a19290d73162e571b8ee0d8ef4da6c8726b5` |
| `sweep0-quick.jsonl` | `cae946b966237039be28c817c178b8beace7e9c990ee5cd418eccebe02392166` |
| `sweep90-quick.jsonl` | `0a34b5845716072c9ddb2c3ce046ddf4f891ef601762ccb26c4896383187fe6d` |

Then, for every table (each model on each GPU):

- **Every column but naive, and the engine-options table**: one suite call per model and GPU with every variant,
  `python -m bench.suite --model Qwen/Qwen3-8B --variant prefix_sharing=false --variant fused_layers=false,cuda_graphs=false --variant cuda_graphs=false --variant prefill_budget=adaptive --variant cuda_graphs=false,prefill_budget=adaptive --variant order=input --variant order=max_tokens_desc --variant order=decode_ratio_desc --variant admission=groups --variant chunk_prefill=false --variant prefill_budget=auto --variant fa_version=3 --variant fa_version=3,prefill_budget=adaptive`
  (`--model Qwen/Qwen3-1.7B` for the 1.7B tables; `fa_version=3` needs a Hopper GPU, so the A100 calls leave its two
  variants out). The suite runs vLLM first with its prefix cache on, off, `enforce_eager` and `max_num_seqs=512` (its
  default on the A100 is 256 running requests), then batchinfer's defaults and each variant; every row compares its
  outputs against this GPU's vLLM cache-on row.
- **The naive column**, one `bench.run` per workload, since the suite runs naive only with its defaults and their
  arrival order does not finish `mixed` in 18 minutes:
  `python -m bench.run workloads/mixed-quick.jsonl --backend naive --model Qwen/Qwen3-8B --opt order=max_tokens_desc --opt max_batch_tokens=65536 --compare vllm`.
  8B `generate` takes about 4 minutes of load and timed pass.
- **Three runs, one row each.** Run-to-run noise depends on the host (README, Results), so every suite call ran three
  times and `results/` holds each row's median run. Rows go to `results/`, or to `$BENCH_RESULTS_ROOT` when set, so
  give each run a directory of its own and export it before all of that run's commands:

  ```
  export BENCH_RESULTS_ROOT=runs/run1-h100   # both models' suite calls, then the naive rows (only in run 1)
  export BENCH_RESULTS_ROOT=runs/run2-h100   # both models' suite calls again, vLLM included
  export BENCH_RESULTS_ROOT=runs/run3-h100   # and once more
  ```

  The naive rows run once, since they take minutes each where a suite row takes seconds; their `--compare vllm`
  reads run 1's vLLM rows. Then, with the workload files in `workloads/`, `unset BENCH_RESULTS_ROOT` and a `results/`
  without rows:

  ```
  python scripts/median_rows.py runs/run1-h100 runs/run2-h100 runs/run3-h100 runs/run1-a100 runs/run2-a100 runs/run3-a100
  python -m bench.report --html results/      # index.html and each row's report.html
  python -m bench.report results/             # the table in results/README.md
  ```

  `median_rows.py` copies each row from the run with its median wall time (a row only one run holds, such as naive,
  as it is), and recomputes each copied row's `compare` against the vLLM row now beside it, since vLLM's greedy
  generations differ from run to run at bf16 near-ties. `results/runs.json` lists every run's wall time of every row
  and which run the row is.
- **Decode-step times** (~23 ms with HF's layers, ~10 ms fused and graphed, 8B at 1 row):
  `python scripts/decode_step_profile.py --model Qwen/Qwen3-8B --rows 1 16 64 320 --context 1024`, with
  `--fused-layers --cuda-graphs` for the defaults' step; its docstring has the rest.

`python -m bench.suite --dry-run` lists the rows a call would run and which baselines it would reuse, without running
anything.

## Results

Rows live in `results/<workload>/<model>/<gpu>/<label>/`, where `<gpu>` is the recorded GPU name shortened
(`A100-SXM4-40GB`, `H100-PCIe`; `cpu` without one; `bench/results.py`) and the label names the options a row set
beyond its engine's defaults. bench writes the latest run of a label into its row directory; the published `results/`
holds each row's median of three runs. `python -m batchinfer run` writes wherever `--output` says, so put the GPU in
that path too.

`--compare` refuses a reference from another GPU. `--compare vllm` takes the `vllm` row of the GPU the run is on from
the results tree (or `$BENCH_REFERENCE_ROOT` when set) and records it by its path in the tree, `results/...`; the run
stops before loading the model if that row is missing. A row's `git_sha` is `git describe` of the tree it ran in,
ending in `-dirty` only when a tracked file outside `results/` differed from it.

The fixed-groups budget (`max_batch_tokens`, used by the naive engine and `admission=groups`) is `auto` unless given:
what the engine measured for its model on its card at load (the naive engine's fit, or the batchinfer engine's KV
pool). The two engines' values differ, so like-for-like rows pass one explicit budget to both, and so do rows compared
across cards: `--opt max_batch_tokens=65536`. Rows record it as a column, and `details.json` records what was measured
under `memory`: `in_use_after_load_gb` and `probe_peak_device_gb` for both engines, `max_batch_tokens_fit`,
`padded_token_kib` and `row_length_bytes` for the naive engine, and `step_activation_gb`, `kv_pool_tokens`,
`graph_capture_s`, `graph_sized_gb` and `graph_held_gb` for batchinfer.

## Iterating

Iterate on Qwen3-1.7B, the CLI's and the suite's default: every command finishes in a few minutes, vLLM's cold start
included (vLLM loads in about 200 s cold and 40 s warm on the A100, 105 s and 40 s on the H100, before a sub-minute
timed pass). Keep 8B for the headline rows and the comparison with vLLM: a speedup measured on 1.7B can overstate the
one on 8B, since a launch-bound step barely shrinks with the model while vLLM's graphed step does. Always pass
`--model` to `bench.run`: the workload files' sidecars name Qwen3-8B, so a command without it runs 8B.

Engine build options (`--opt` / `--variant` for bench, flags for `python -m batchinfer run`): `fused_layers` and
`cuda_graphs` default to `auto`, on wherever they can run (Qwen3 on a GPU; graphs with FlashAttention-2), and every
row records what ran (`fused_layers`, `cuda_graphs` and `fa_version` in `backend_stats`; the index's layers and graphs
columns). `fused_layers=false,cuda_graphs=false` is the non-optimised setup (HF's layers, eager), which `bench.suite`
runs as a variant by default. `cuda_graphs` replays decode-only steps from graphs captured at load (a few seconds of
`load_s`; the step trace's `graph_rows` says which steps replayed). `fa_version=3` is FlashAttention-3 (sm90 only),
which runs without graphs. Each option is its own load group in `bench.suite`. The profiler takes the same switches.

## Tests and reproducibility

`python -m pytest tests/ -q` runs the CPU tests anywhere and, on a GPU, the tiny-model engine tests and the executor
gate on Qwen3-0.6B, in about 70 s. The gate accepts a token that is HF's argmax or within two bf16 ulps of HF's top
logit, checked at every generated position on the engine's own history: bf16 near-ties (a top-2 margin of 0.125 at
logits near 17) flip between kernels. `scripts/reference_check.py` checks the naive engine against HF's
`model.generate` and a full recompute at batch size 1.

batchinfer reproduces every token run to run. The naive engine does not on long padded generations: repeated
smoke-quick runs on the H100 differed pairwise on 2-4 of 20 rows (the long WildChat answers, from token 33-405 on)
with identical groups and memory. Compare naive rows by accuracy and `--compare` match rate, never byte equality.

## Browsing the reports

`python -m bench.report --html results/` writes a `report.html` beside every `metrics.json` and `results/index.html`
linking them; open them from the checkout. To browse from another machine, serve them:
`python -m bench.report --serve 8765 --bind <address> results/`. `--bind` defaults to 127.0.0.1; bind an address
only machines you trust can reach.
