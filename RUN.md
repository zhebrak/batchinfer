# Reproducing the benchmarks

All published results are stored under `results/` (see `results/README.md` for the format). This file describes the
hardware they ran on and the commands that produced them.

## Setup

You need an NVIDIA driver 580 or newer (for torch 2.13's cu130 wheels), Python 3.12, a C compiler (for Triton) and
a CUDA 13 toolkit: FlashInfer builds vLLM's top-k sampler at warmup against the wheels' CUDA. torch looks for the
toolkit in `$CUDA_HOME`, then the `nvcc` on your `PATH`, then `/usr/local/cuda`, so on a machine with an older `nvcc`
on its `PATH` set `CUDA_HOME`.

```
git clone https://github.com/zhebrak/batchinfer && cd batchinfer
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
PATH=$PWD/.venv/bin:$PATH   # FlashInfer's JIT kernels need the venv's ninja
```

Run everything from the repository root. Models and datasets are downloaded on first use to `~/.cache/huggingface`
and `~/.cache/bench`, and no Hugging Face token is needed. If the venv's `bin` is not on your `PATH`, vLLM fails at
warmup with `FileNotFoundError: ... 'ninja'`. In a container that sees more CPUs than its quota allows, set
`OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `TORCHINDUCTOR_COMPILE_THREADS` to the quota or lower.

## Hardware

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (SXM), 700 W cap, 1,980 MHz max SM clock |
| CPU | Intel Xeon Platinum 8568Y+ in a container limited to 23 CPUs (192 visible), OMP, MKL and Inductor thread pools set to 8 |
| OS, driver | Ubuntu 24.04.5 LTS, driver 610.57.04, CUDA 13.1 toolkit |
| Packages | Python 3.12, torch 2.13.0 (CUDA 13.0), transformers 5.17.0, vLLM 0.30.0 |

Each result records the package versions, GPU, clock and throttling (`gpu_clock_mhz_mean`, `gpu_throttled_pct`) and
the commit it ran at. The CPU matters for the eager runs (HF layers, vLLM with `enforce_eager`), which are limited by
kernel launches. The default runs are limited by the GPU. Only run one job on the GPU at a time.

## Running the benchmarks

`ARMS` in `bench/suite.py` lists every configuration the suite can run: the engine, its options, the workloads it
runs on and the column it fills. `--campaign` picks the ones behind the published results. `--dry-run` prints the
runs, the load groups and which existing results it would reuse, without running anything. vLLM runs with its
defaults (prefix cache on, 1,024 running sequences on an 80 GB GPU, 16,384 batched tokens), and every result
compares its outputs against vLLM's cache-on result on the same GPU. The capped runs give both engines the same
usable KV tokens, a quarter of what the workload needs, and every result records `kv_capacity_tokens` as its engine
reports it.

Timings vary between runs, so the published numbers are medians of three. Each run writes to its own directory
(`$BENCH_RESULTS_ROOT`, default `results/`):

```
M=Qwen/Qwen3-14B
for n in 1 2 3; do
  BENCH_RESULTS_ROOT=runs/run$n python -m bench.suite --model $M --campaign
  BENCH_RESULTS_ROOT=runs/run$n python -m bench.suite --model $M --workloads diverse --once --engines vllm,batchinfer
done
BENCH_RESULTS_ROOT=runs/run1 python -m bench.suite --model $M --workloads sweep10,sweep25,sweep50,sweep75 \
  --column "vLLM cache on" --column "vLLM cache off" --column batchinfer --column "no trie"
python scripts/median_rows.py runs/run1 runs/run2 runs/run3 --out results
python -m bench.report --html results/                # writes index.html and a page per run
```

For the published results, only the main comparisons were run three times: vLLM with and without its prefix cache,
batchinfer's defaults, batchinfer without prefix sharing, and batchinfer without overlap. On `docqa` and `rag` only
batchinfer's defaults and vLLM with its cache were repeated. Everything else ran once. The loop above runs
everything three times. `median_rows.py` copies each configuration from the run with its median wall time (one that
only a single run has is copied as is) and recomputes its `compare` against the vLLM result now next to it, because
vLLM's greedy outputs differ between runs at bf16 near-ties. `results/runs.json` has every run's wall time for every
configuration and which run was kept.

The suite builds any missing workload from its preset. To build one yourself, run
`python -m bench.workload build --preset <name> --size quick --model Qwen/Qwen3-14B` (add `--lengths natural` for
`generate-natural`). The datasets are pulled from the Hugging Face Hub at their latest revision (`bench/sources.py`),
so check each file against the hash in the results (`workload_sha256`). A different hash means a different workload.
The published workloads were built from `cais/mmlu@c30699e8`, `clinc/clinc_oos@155b9c71`, `openai/gsm8k@740312ad`,
`allenai/WildChat-1M@7d6490e4` and QuALITY v1.0.1 (htmlstripped, dev), with Qwen3-14B's chat template:

| Workload file | sha256 |
|---|---|
| `mixed-quick.jsonl` | `54c82627830a6cceffdcbc136e74ae7b812f097605846e613149a9d117ddc9f5` |
| `classify-quick.jsonl` | `c49ce11968ac90b739dc2456b7a1295f3f7974dd1cd838995b624cfda1bbca89` |
| `generate-quick.jsonl` | `6c20a3e6b76372d190a7557f83b1a19290d73162e571b8ee0d8ef4da6c8726b5` |
| `generate-quick-natural.jsonl` | `d1b6b98c3c0c84721b962fd0f08d546aa1c9555ca34446a4732f70c7bef4eefe` |
| `sweep0-quick.jsonl` | `cae946b966237039be28c817c178b8beace7e9c990ee5cd418eccebe02392166` |
| `sweep10-quick.jsonl` | `5590e799189160f20a8fe4a12e530c29f319b483b46b59a3264140de05c24cf8` |
| `sweep25-quick.jsonl` | `3a26589a87837d932ebd41936497da014ebc93cf2b9655917122f70f6bed4c3f` |
| `sweep50-quick.jsonl` | `dfc512b66b0f4c7b54fa6b62a514572a5c7e06e37e2f1b7deeb2f7958ffa1a61` |
| `sweep75-quick.jsonl` | `36641888c8d6f83f997e4851c05daa68b6cbeb966298d4190c139fffa411ce7e` |
| `sweep90-quick.jsonl` | `0a34b5845716072c9ddb2c3ce046ddf4f891ef601762ccb26c4896383187fe6d` |
| `pool-quick.jsonl` | `8e270eb344d6c0ded00cce61493f1cff30281b858d9b030ef734dad086675ac2` |
| `long0x8-quick.jsonl` | `555866e5d41fd2853dfb9fc04f59bcbffca0328c4eca23c92842d512441068d3` |
| `long50x8-quick.jsonl` | `673fde7f9f52f2440db9d01049d2d8055f0753f49b885706b80609486b5d6f49` |
| `docqa-quick.jsonl` | `deed1df45db1482ea929dbc2ae6aeb343a535152812bc26f8538d421b4751580` |
| `rag-quick.jsonl` | `3ea1825f8951ab14a519ee9e78ec474742c17d7508c1925f2ed4d15458be6626` |
| `dec50x32-quick.jsonl` | `8c1a9e250dd488daabc112068da65739324261d02c5d59920989193100a17c79` |
| `pf50x32-quick.jsonl` | `fdb2637fab71c0c6e6a591997772493c707b96174ec7124dddcfae648525c7ba` |

Two scripts cover things the suite doesn't:

- `python scripts/decode_step_profile.py --model Qwen/Qwen3-14B --rows 64 256 512 --context 1024 --layers fused
  --cuda-graphs` times decode steps. `--pool contiguous` adds the paged attention overhead, and `--block-size`,
  `--attention cascade` and `--layers compiled` cover the other modes. See its docstring for the rest.
- `python scripts/ridge_calibration.py --model Qwen/Qwen3-14B` measures the GPU's ridge for
  `--prefill-budget ridge` and prints the `metrics.CARDS` entry to add.

## Result files

Results go to `results/<workload>/<model>/<gpu>/<label>/`. `<gpu>` is the short GPU name (`cpu` if there is none) and
the label lists the options that differ from the engine's defaults. A new run of the same label replaces the old one.
`--compare vllm` compares against the `vllm` result for the same GPU in the results directory (or
`$BENCH_REFERENCE_ROOT`) and refuses one from a different GPU. `git_sha` is `git describe` of the checkout, with
`-dirty` only if a tracked file outside `results/` was modified.

For quick experiments use Qwen3-1.7B, the default for the CLI and the suite. Speedups on 1.7B can look bigger than on
larger models, because a launch-bound step barely gets faster with a smaller model while vLLM's graphed step does.
Always pass `--model` to `bench.run`, otherwise it uses the model the workload was built for.

## Tests

`python -m pytest tests/ -q` runs the CPU tests anywhere. On a GPU it also runs the engine tests on a tiny model and
the executor check on Qwen3-0.6B. That check accepts a token if it is HF's argmax or its logit is within two bf16 ulps
of HF's top logit, at every generated position on the engine's own history, since bf16 near-ties (a top-2 margin of
0.125 at logits near 17) flip between kernels. `scripts/reference_check.py` checks the
naive engine against HF's `model.generate` and a full recompute at batch size 1.

batchinfer produces the same tokens on every run. The naive engine doesn't on long padded generations, so compare
naive results by accuracy and `--compare` match rate, not byte for byte.

`python -m bench.report --serve 8765 results/` serves the report. `--bind` defaults to 127.0.0.1, so only bind
another address on a network you trust.
