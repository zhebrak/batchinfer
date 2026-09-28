# batchinfer

An offline batch-inference engine for decoder-only Transformers.
Scheduling, batching and KV-cache management are hand-rolled. HF transformers supplies the model weights; vLLM
supplies kernels only (FlashAttention and a handful of fused ops), no engine, scheduler or cache code.

On a 288-request mix of classification and generation (Qwen3-8B, H100), it takes 10.1 s against 8.8 s for vLLM with
its prefix cache on and 156.1 s for naive padded HF batching.

**[Browse all 360 benchmark runs](https://zhebrak.github.io/batchinfer/results/)**: five workloads, Qwen3-8B and 1.7B,
H100 and A100, against vLLM and naive HF, each with its trace and charts.

## Quick start

```
pip install -r requirements.txt                    # torch 2.13, transformers 5.17, vllm 0.30 (kernels, baseline)

python -m bench.workload build --preset mixed --size quick --model Qwen/Qwen3-8B    # workloads/mixed-quick.jsonl
python -m batchinfer analyze --input workloads/mixed-quick.jsonl                    # job facts and decisions, no GPU
python -m batchinfer run --input workloads/mixed-quick.jsonl --output out.jsonl --model Qwen/Qwen3-8B
python -m bench.suite --model Qwen/Qwen3-8B        # five workloads against vLLM; HTML report under results/
python -m pytest tests/                            # CPU tests everywhere, GPU tests where CUDA is available
```

Input is JSONL, one request per line: `prompt` (required), `id`, `max_tokens` (default 128), `ignore_eos` and
`labels` (classification answers). Results stream to `--output` as requests finish; `metrics.json` beside it holds
the per-step trace. `python -m batchinfer run --help` lists the knobs; the main ones:

| Option | Default | |
|---|---|---|
| `--prefix-sharing` | on | the global prefix trie |
| `--order` | `prefix_dfs` | admission order (`input`, `max_tokens_desc`, `decode_ratio_desc`) |
| `--prefill-budget` | `16384` | prefill tokens per step; `auto` and `adaptive` spread the prompts over the decode chain |
| `--fused-layers`, `--cuda-graphs` | on where they can run | `--no-fused-layers --no-cuda-graphs` runs HF's layers eagerly |
| `--fa-version` | 2 | 3 for FlashAttention-3 on Hopper (runs without graphs) |

## How it works

```
JSONL --io--> Request --analysis--> JobAnalysis --policy.decide--> Policy --engine.run--> Result per request
                        tokens, prefix trie      order, budgets      every step: Scheduler -> Executor -> commit
```

The whole job is known up front: analysis measures it once, policy decides once, the scheduler decides per step and
the executor only runs what it is given. The engine sees only what a client sends with each request, never the
benchmark's own metadata, and adapts from that alone.

- **Global prefix trie.** Every prompt is tokenized once and put in a trie of 16-token KV blocks before the first
  step. A block's first user writes it, later users read it, and it is freed exactly when its last user finishes,
  since the trie knows every block's users: nothing is kept on speculation and nothing still needed is evicted.
- **Paged KV, reserved for a request's life.** Admission reserves `prompt_len + max_tokens - 1` tokens, so the pool
  never runs dry mid-decode and nothing is preempted or recomputed.
- **One packed forward per step.** Every decode row plus prefill chunks in one `[1, N]` forward without padding,
  attention through per-row block tables, one host sync.
- **Order and admission.** `prefix_dfs` walks the trie depth first, the subtree with the highest decode ratio
  `(max_tokens - 1) / prompt_len` first: generations start early, classification fills prefill chunks, and requests
  sharing a prefix run back to back. A request is admitted as soon as its reservation fits. A step prefills up to
  16,384 prompt tokens, the widest step the engine sizes for: with fused layers and graphs the step is GPU-bound,
  so fewer, larger prefill steps cost less, and the decode-only steps between them replay a graph.
- **Sized on the card.** At load the engine runs its widest possible step once and gives the KV pool the rest, up
  to 95% of memory. No constant names a model or GPU.
- **Fused layers and CUDA graphs.** HF's Qwen3 layer launches ~57 kernels, so an 8B step makes ~2,200 launches and
  the host sets its pace. Our layer forward over HF's weights (`qwen3_fused.py`) runs the same arithmetic in ~11 of
  vLLM's kernels, and decode-only steps are captured as CUDA graphs at load and replayed as one launch. Steps that
  carry prefill run eagerly.

## Results

Five quick workloads from public datasets (`bench/sources.py`): `mixed` is 288 requests (192 classification, 96
generation, 297k prompt tokens); `classify` (MMLU, CLINC, QuALITY) has long shared prefixes; `generate` (GSM8K,
WildChat) is decode-bound; `sweep0` and `sweep90` are 320 synthetic prompts with 0% and 90% of each prompt shared.
Only the pass that sends the whole job is timed, and each cell is the median of three runs of the suite (naive ran
once). Every row is in `results/` with its per-step trace and outputs (`results/README.md`, browsable at
https://zhebrak.github.io/batchinfer/results/); `RUN.md` reproduces them.

*batchinfer* is the defaults; *HF layers, eager* is the same engine on HF's own layers without CUDA graphs; *no trie*
turns prefix sharing off, and *trie gain* is its time over batchinfer's. *Shared* is the share of prompt tokens a trie
can reuse. *naive* is padded HF batches of 65,536 tokens, longest `max_tokens` first. vLLM runs with its defaults and
its prefix cache on or off; *vLLM eager* is cache on with `enforce_eager` (no torch.compile or CUDA graphs). *× vLLM*
is batchinfer's time over vLLM cache on's; † gives it against vLLM with `max_num_seqs=512` where that is noticeably
faster (its default is 256 on the A100 and 1,024 on the H100).

**Qwen3-8B bf16, H100 PCIe** (wall s)

| Workload | Shared | naive | HF layers, eager | batchinfer | no trie | Trie gain | vLLM eager | vLLM cache on | vLLM cache off | × vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixed` | 81% | 156.1 | 16.0 | **10.1** | 20.1 | 1.99x | 9.8 | 8.8 | 18.3 | 1.15 |
| `classify` | 86% | 28.1 | 3.8 | **2.9** | 16.6 | 5.78x | 2.7 | 2.5 | 15.4 | 1.14 |
| `generate` | 66% | 211.8 | 16.0 | **10.7** | 13.8 | 1.29x | 9.9 | 9.3 | 12.7 | 1.15 |
| `sweep0` | 0% | 41.7 | 21.2 | **16.1** | 16.1 | 1.00x | 15.1 | 14.7 | 14.9 | 1.09 |
| `sweep90` | 87% | 43.6 | 5.6 | **4.4** | 16.1 | 3.66x | 4.1 | 4.0 | 15.0 | 1.11 |

**Qwen3-8B bf16, A100-40GB** (wall s)

| Workload | Shared | naive | HF layers, eager | batchinfer | no trie | Trie gain | vLLM eager | vLLM cache on | vLLM cache off | × vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixed` | 81% | 176.8 | 27.9 | **14.3** | 31.1 | 2.18x | 18.9 | 13.7 | 30.7 | 1.04 |
| `classify` | 86% | 48.7 | 5.8 | **4.7** | 27.3 | 5.86x | 4.5 | 4.4 | 27.3 | 1.07 |
| `generate` | 66% | 231.8 | 26.3 | **15.2** | 21.6 | 1.42x | 17.8 | 14.8 | 20.4 | 1.03 |
| `sweep0` | 0% | 55.6 | 35.8 | **27.0** | 27.0 | 1.00x | 29.2 | 28.1 | 28.4 | 0.96 |
| `sweep90` | 87% | 58.2 | 8.3 | **6.5** | 27.1 | 4.17x | 7.6 | 6.9 | 28.4 | 0.95 (1.03 †) |

**Qwen3-1.7B bf16, H100 PCIe** (wall s)

| Workload | Shared | naive | HF layers, eager | batchinfer | no trie | Trie gain | vLLM eager | vLLM cache on | vLLM cache off | × vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixed` | 81% | 93.1 | 12.3 | **4.2** | 6.9 | 1.64x | 6.9 | 3.5 | 5.9 | 1.20 |
| `classify` | 86% | 9.7 | 1.3 | **0.9** | 4.4 | 4.86x | 1.4 | 1.4 | 3.8 | 0.63 |
| `generate` | 66% | 126.4 | 10.5 | **5.1** | 6.1 | 1.20x | 6.7 | 4.0 | 5.2 | 1.26 |
| `sweep0` | 0% | 21.8 | 7.7 | **5.1** | 5.1 | 1.00x | 4.9 | 4.6 | 4.6 | 1.12 |
| `sweep90` | 87% | 22.0 | 2.7 | **2.0** | 5.1 | 2.55x | 3.4 | 2.5 | 4.6 | 0.79 |

**Qwen3-1.7B bf16, A100-40GB** (wall s)

| Workload | Shared | naive | HF layers, eager | batchinfer | no trie | Trie gain | vLLM eager | vLLM cache on | vLLM cache off | × vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixed` | 81% | 82.2 | 19.2 | **5.7** | 10.1 | 1.77x | 12.9 | 5.2 | 9.7 | 1.10 |
| `classify` | 86% | 15.3 | 1.9 | **1.4** | 6.9 | 5.04x | 1.3 | 1.2 | 6.7 | 1.12 |
| `generate` | 66% | 107.9 | 19.0 | **6.7** | 8.5 | 1.28x | 13.3 | 6.2 | 8.1 | 1.08 |
| `sweep0` | 0% | 23.5 | 12.7 | **8.2** | 8.1 | 0.99x | 8.7 | 7.9 | 7.9 | 1.04 |
| `sweep90` | 87% | 23.9 | 3.7 | **2.7** | 8.1 | 3.01x | 4.0 | 3.0 | 7.9 | 0.89 (0.95 †) |

Accuracy is within two questions of vLLM's per source in every table cell except 1.7B H100 MMLU, where batchinfer
answers 153 of 256 with or without the trie against vLLM's 156 (naive 157). batchinfer's classification outputs are
96.8-100% identical to vLLM's; generations diverge at bf16 near-ties, as vLLM's own do from run to run. On the A100 a
cell's three runs agree within 6.6% (median 0.2%). The H100's host is noisier (median 0.9%; vLLM's defaults up to
23%, launch-bound rows up to 45%, such as vLLM `enforce_eager` on 1.7B `mixed` at 6.8, 6.9 and 9.9 s), so read small
differences between H100 cells as ties.

- **Sharing is the largest lever.** Prefill drops from 297k to 58k tokens on `mixed` (80.45% hit rate, the ceiling for
  whole 16-token blocks). The trie makes `classify` 4.9-5.9x faster, `sweep90` 2.6-4.2x and `mixed` 1.6-2.2x, and a
  job with nothing to share pays nothing (`sweep0`).
- **Prefill all at once.** A step prefills up to 16,384 prompt tokens, so `mixed` is prefilled in 5 steps and
  `generate` in 3, every generation starts decoding at once, and 99% of `mixed`'s 512 steps replay a CUDA graph.
  `--prefill-budget adaptive` spreads the prompts over the longest decode chain instead, but that chain's requests
  start only once their own prompts are in: on `generate` it starts at step 89 instead of 1 and the job takes 601
  steps instead of 513 (12.4 against 10.7 s on 8B H100). Adaptive is slower in 16 of the 20 workload cells and within
  2% in the rest.
- **The engine is GPU-bound.** With HF's layers a step is bound by kernel launches (on the H100 a decode-only 8B step
  takes ~23 ms whether 1 or 64 rows are live, `scripts/decode_step_profile.py`); fused layers and graphs bring it to
  ~10 ms at 1 row, about one read of the weights. Every option on `mixed` (wall s, and × vLLM cache on):

  | Configuration | 8B H100 | 8B A100 | 1.7B H100 | 1.7B A100 |
  |---|---:|---:|---:|---:|
  | HF layers, eager (`--no-fused-layers --no-cuda-graphs`) | 16.0 (1.83x) | 27.9 (2.04x) | 12.3 (3.47x) | 19.2 (3.70x) |
  | fused layers, eager (`--no-cuda-graphs`) | 10.9 (1.24x) | 15.1 (1.10x) | 5.5 (1.56x) | 6.8 (1.30x) |
  | fused layers, eager, `--prefill-budget adaptive` | 11.4 (1.30x) | 16.1 (1.18x) | 5.1 (1.45x) | 7.1 (1.37x) |
  | fused layers, CUDA graphs: **the defaults** | 10.1 (1.15x) | 14.3 (1.04x) | 4.2 (1.20x) | 5.7 (1.10x) |
  | the defaults, `--prefill-budget adaptive` | 10.7 (1.22x) | 15.6 (1.14x) | 4.5 (1.27x) | 6.1 (1.17x) |
  | the defaults, `--prefill-budget auto` | 10.7 (1.22x) | 15.5 (1.14x) | 4.5 (1.26x) | 6.1 (1.17x) |
  | fused layers, FlashAttention-3, eager (`--fa-version 3`) | 9.7 (1.11x) | - | 5.1 (1.46x) | - |
  | `--fa-version 3 --prefill-budget adaptive` | 9.9 (1.13x) | - | 4.3 (1.21x) | - |
  | the defaults, `--no-prefix-sharing` | 20.1 (2.29x) | 31.1 (2.28x) | 6.9 (1.96x) | 10.1 (1.94x) |
  | the defaults, `--order input` | 10.1 (1.15x) | 14.2 (1.04x) | 4.2 (1.20x) | 5.6 (1.08x) |
  | the defaults, `--order max_tokens_desc` | 10.0 (1.14x) | 14.2 (1.04x) | 4.2 (1.18x) | 5.6 (1.08x) |
  | the defaults, `--order decode_ratio_desc` | 10.0 (1.14x) | 14.3 (1.04x) | 4.2 (1.19x) | 5.7 (1.09x) |
  | the defaults, `--admission groups` | 12.8 (1.45x) | 21.1 (1.54x) | 5.3 (1.49x) | 7.1 (1.36x) |
  | the defaults, `--no-chunk-prefill` | 10.0 (1.15x) | 14.3 (1.05x) | 4.2 (1.19x) | 5.7 (1.10x) |
  | vLLM, prefix cache on (its defaults) | 8.8 | 13.7 | 3.5 | 5.2 |
  | vLLM, prefix cache on, `max_num_seqs=512` | 8.8 (1.01x) | 13.7 (1.00x) | 3.5 (1.00x) | 5.2 (1.00x) |
  | vLLM, prefix cache on, `enforce_eager` | 9.8 (1.11x) | 18.9 (1.38x) | 6.9 (1.94x) | 12.9 (2.49x) |
  | vLLM, prefix cache off | 18.3 (2.09x) | 30.7 (2.24x) | 5.9 (1.66x) | 9.7 (1.86x) |

  Fused layers are the largest step (16.0 to 10.9 s on 8B H100, 19.2 to 6.8 s on 1.7B A100, both eager), and graphs
  take 8B H100 on to 10.1 s. FlashAttention-3 (Hopper only) runs eagerly, which makes it the fastest 8B configuration
  (9.7 s) but costs 1.7B more than it gains (5.1 against 4.2 s). The admission order changes `mixed` by at most 2%
  once everything is prefilled at once, while fixed groups (`--admission groups`) cost 24-48%.
- **Where time goes after that.** `mixed` runs 512 steps with or without the trie, bound by its longest decode chain,
  and attention is most of a wide decode step (27 of 42 ms at 320 rows x 1k tokens, 8B on the H100).

## Compared with existing solutions

- **Naive HF batching** (`naive.py`): 2.1-19.8x slower across the five workloads on 8B and 2.9-24.9x on 1.7B,
  least on `sweep0` (nothing to share, equal output lengths) and most on `generate`.
- **vLLM 0.30**, with its defaults and its prefix cache on, is faster where both engines are bound by the GPU:
  batchinfer takes 1.09-1.15x its time on 8B and 1.12-1.26x on 1.7B on the H100, 1.03-1.07x and 1.04-1.12x on the
  A100. Sharing is not the gap: outside the cells below, vLLM's cache computes exactly the prompt tokens our trie
  does. The difference is engine work: vLLM also graphs the steps that carry prefill and fuses with torch.compile.
  Eager against eager, batchinfer is level or ahead on `mixed`: 9.7 s (fused, FlashAttention-3) against vLLM
  `enforce_eager`'s 9.8 s on 8B H100, 15.1 against 18.9 s on 8B A100, and 5.5 against 6.9 s and 6.8 against 12.9 s
  on 1.7B. Where batchinfer is ahead, the reason is on vLLM's side, not a faster kernel:
  - *Running-request cap* (A100 `sweep90`). vLLM runs at most 256 requests at once by default, so the 320 run in two
    waves (133 iterations on 8B and 143 on 1.7B, against batchinfer's 67 steps). With `max_num_seqs=512` it runs
    them at once (72 and 116 iterations) and batchinfer's ratio goes from 0.95 to 1.03 on 8B and from 0.89 to 0.95
    on 1.7B (†). Everywhere else that setting moves vLLM's time by 3% at most.
  - *Preemption* (8B A100 `sweep0`, nothing shared). vLLM's KV pool runs full and it preempts, computing 341,692
    prompt tokens against the workload's 331,791; batchinfer reserves a request's KV when it admits it and never
    preempts (0.96).
  - *Host-bound scheduling* (1.7B H100). A small model leaves the GPU waiting on vLLM's host work: on `classify`
    (0.63) it runs 30 iterations with at most 43 of its 376 requests running (its running-request cap there is
    1,024) at 39% GPU utilisation, against batchinfer's 6 steps at 79%; on `sweep90` (0.79) it runs 123 iterations
    at 58%, against batchinfer's 67 steps at 86%.
- **BatchLLM** (arXiv 2412.03594), the closest published design: global prefix identification, block lifetimes tied
  to their users, decode-ratio ordering. batchinfer shares every trie level rather than one, and reserves KV for a
  request's life rather than allocating it as written; it has no fused shared-prefix attention kernel.

## Limitations

- One GPU, bf16, greedy decoding. Tested on Qwen3 (0.6B to 8B). Fused layers are Qwen3-only; other decoder-only
  HF models should run on HF's layers through the same paged attention, but are untested.
- KV is reserved for a request's whole life: simple and never preempts, but it holds space not yet written, so a
  pool that binds admits fewer requests at a time.
- CUDA graphs cover decode-only steps with FlashAttention-2; FlashAttention-3 runs eagerly.
- The naive baseline in arrival order is impractically slow (past 18 minutes on 8B `mixed` on the H100, in an
  earlier run of the same engine); its rows above sort by `max_tokens`.

## Layout

- `batchinfer/`: the engine. `flow.py` chains the stages; `analysis.py` and `prefix.py` (the trie), `policy.py`,
  `scheduler.py`, `kv.py`, `executor.py` (paged attention, CUDA graphs), `qwen3_fused.py`, `step_engine.py`, and
  `naive.py` (the baseline).
- `bench/`: workload sources and builder, the vLLM backend, `run.py` and `suite.py`, and the HTML report.
- `results/`: every benchmark row behind the tables above, and the ablation rows beside them.
- `scripts/`: the decode-step profiler, a reference check against HF, and the median-run assembly of `results/`
  (`median_rows.py`).
- `tests/`: CPU tests for every stage (exact token equality in fp32 against HF) and GPU tests (every token HF's
  pick or within two bf16 ulps). `RUN.md` reproduces the numbers above.
