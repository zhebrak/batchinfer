# batchinfer

Offline batch inference for decoder-only transformers. batchinfer does its own scheduling, batching and KV cache
management. It loads model weights with HF transformers and uses vLLM only for kernels (FlashAttention and a few
fused ops), not for its engine, scheduler or cache.

Since the whole job is known before the first step, batchinfer computes each shared prompt prefix once and frees it
as soon as the last request that uses it is done. This helps most when a job has more shared documents than fit in
the KV cache, or when memory is tight. When everything fits, it runs about as fast as vLLM.

See the [benchmark results](https://zhebrak.github.io/batchinfer/results/), and `RUN.md` for how to reproduce them.

## Quick start

```
pip install -r requirements.txt                    # torch 2.13, transformers 5.17, vllm 0.30

python -m bench.workload build --preset mixed --size quick --model Qwen/Qwen3-14B   # writes workloads/mixed-quick.jsonl
python -m batchinfer analyze --input workloads/mixed-quick.jsonl                    # job stats and decisions, no GPU
python -m batchinfer run --input workloads/mixed-quick.jsonl --output out.jsonl --model Qwen/Qwen3-14B
python -m bench.suite --model Qwen/Qwen3-14B       # benchmark against vLLM, HTML report in results/
python -m pytest tests/                            # CPU tests, plus GPU tests if CUDA is available
```

Input is JSONL with one request per line. `prompt` is required. `id`, `max_tokens` (default 128), `ignore_eos` and
`labels` (for classification) are optional. Results are written to `--output` as requests finish, and
`metrics.json` next to it has the per-step trace. `python -m batchinfer run --help` lists all options. The main ones:

| Option | Default | |
|---|---|---|
| `--sharing` | block | Prefix sharing: `none`, `block`, or `token`, which also copies the leading tokens of a request's first unshared block from a sibling that already computed them |
| `--cache` | job | `job` frees a shared block after its last user. `radix` keeps it for the next job in the same process and evicts the oldest kept blocks when space is needed |
| `--memory` | reserve | `reserve` allocates a request's full KV at admission. `grow` and `preempt` allocate blocks as decode writes them. `preempt` recomputes the newest request when memory runs out |
| `--watermark-steps` | 1 | With `preempt`: how many decode steps of growth admission leaves room for. 1 admits as soon as the prompt fits, like vLLM. Larger values approach `reserve` |
| `--order` | `prefix_dfs` | Admission order: `prefix_dfs`, `input`, `max_tokens_desc` or `decode_ratio_desc` |
| `--prefill-budget` | 16384 | Max prefill tokens per step. `auto` spreads prefill across the decode phase. `ridge` fills steps that have decode rows up to the GPU's measured ridge (`--ridge-tokens`, or the GPU's entry from `scripts/ridge_calibration.py`) |
| `--block-size` | 16 | Tokens per KV block, a multiple of 16 up to 256 |
| `--quant` | none | `fp8`: the fused layers' projection weights in e4m3 with a scale per output channel, activations quantized per token. Needs FP8 tensor cores |
| `--kv-dtype` | auto | `fp8`: the KV cache in e4m3 at static unit scales, so the same memory holds twice the tokens. Needs FlashAttention-3 |
| `--attention` | fa | `cascade`: requests behind one shared prefix read its KV once per step (two FlashAttention calls, merged exactly). `torch`: SDPA per row |
| `--pool` | static | `elastic`: the KV pool on CUDA virtual memory, with physical memory only under live blocks and returned as requests finish. `contiguous`: each sequence reads its KV as one contiguous range in its own virtual region, without a block table (FlashAttention-2 only) |
| `--overlap` | on | Plan and launch the next step while the current one runs, so CPU work is hidden behind the GPU. Decode tokens are passed from the previous step's output on the GPU. `--no-overlap` runs one step at a time |
| `--layers` | auto | `fused` (Qwen3 layers on vLLM kernels), `compiled` (HF layers under `torch.compile`, any decoder that passes the checks) or `hf` (eager). `auto` picks the first that can run |
| `--cuda-graphs` | on | Replay steps from CUDA graphs. `--layers hf --no-cuda-graphs` runs plain HF |
| `--fa-version` | auto | FlashAttention-3 on Hopper, otherwise 2. With FA3 the kernel's tile schedule is computed per step outside the graph, so steps with prefill also replay graphs |
| `--resume` | off | Keep the results already in `--output` and only run what is missing |

## How it works

```
JSONL --io--> Request --analysis--> JobAnalysis --policy.decide--> Policy --engine.run--> Result per request
                        tokens, prefix trie      order, budgets      every step: Scheduler -> Executor -> commit
```

Analysis runs once over the whole job and the policy is decided once. After that the scheduler decides what goes
into each step and the executor runs it. The engine only uses what is in the requests, never the benchmark's
metadata.

- **Global prefix trie.** All prompts are tokenized once, before the first step, into a trie of KV blocks (16 tokens
  each by default). The first request to reach a block computes it, later ones read it, and the block is freed when
  its last user finishes. Because the trie knows every block's users, nothing is cached speculatively and nothing
  that is still needed gets evicted.
- **Ordering and admission.** `prefix_dfs` walks the trie depth-first, visiting the subtree with the highest decode
  ratio `(max_tokens - 1) / prompt_len` first. Generations start early, classification requests fill the prefill
  chunks, and all requests behind a shared prefix run back to back, so its KV is written once and read by all of
  them before it is freed. If an order would need to hold more shared blocks than the pool has, the run is rejected
  before the first step instead of stalling. A request is admitted as soon as its KV fits, and a step prefills up
  to 16,384 prompt tokens, the largest step the engine sizes for.
- **Paged KV with three memory modes.** `reserve` (the default) reserves `prompt_len + max_tokens - 1` tokens at
  admission, so the pool never runs out and nothing is recomputed. `grow` allocates decode blocks as decode reaches
  them and admits against a projection of every running request's future demand. `preempt` admits on a watermark.
  When the pool runs out it first frees parked blocks, then pinned shared blocks, then preempts the newest request
  and recomputes it when it is readmitted (outputs are unchanged under greedy decoding).
- **One forward pass per step.** All decode rows and prefill chunks go into a single `[1, N]` forward pass without
  padding, with attention through per-row block tables. With overlap on, the one CPU-GPU sync per step happens while
  the GPU is still busy.
- **Fused layers and CUDA graphs.** HF's Qwen3 layer launches about 57 kernels, so a step makes thousands of launches
  and the CPU becomes the bottleneck. Our layer (`qwen3_fused.py`) runs the same math on HF's weights in about 11 of
  vLLM's kernels. Steps are captured as CUDA graphs at load time and replayed with one launch: decode-only steps per
  batch size, and with FlashAttention-3 also steps with prefill, bucketed by rows and tokens. Other models use HF
  layers with `torch.compile` (`--layers compiled`).
- **Sizing at load time.** At startup the engine runs its largest possible step once and gives the remaining memory
  to the KV pool, up to 95% of the GPU's memory. Nothing is hard-coded per model or GPU.

The closest published design is BatchLLM (arXiv 2412.03594): global prefix identification, block lifetimes tied to
their users, and ordering by decode ratio. batchinfer shares prefixes at every level of the trie instead of one,
offers reserve, grow and preempt as memory modes, and runs shared-prefix attention as two FlashAttention calls merged
exactly (`--attention cascade`) instead of a fused kernel.

## Limitations

- Single GPU, greedy decoding only, bf16 or fp8. The fused layers are Qwen3-specific. Other models run HF's layers,
  eager or under `torch.compile` (tested on TinyLlama-1.1B).
- Memory is planned for `max_tokens`, with no output length prediction. vLLM, which allocates as tokens are written,
  does better when replies are much shorter than their limit and memory is tight.
- `--pool contiguous` only works with FlashAttention-2 (FlashAttention-3's read over the regions faults on steps with
  prefill). FP8 KV needs FlashAttention-3, and FP8 weights need FP8 tensor cores.
- With FlashAttention-2 (before Hopper), steps that include prefill run without CUDA graphs.

## Layout

- `batchinfer/`: the engine. `flow.py` connects the stages: `analysis.py` and `prefix.py` (the trie), `policy.py`,
  `scheduler.py`, `kv.py`, `executor.py` (one step's forward pass), `pool.py` (where the KV lives), `attention.py`,
  `graphs.py` (CUDA graph layout and staging), `qwen3_fused.py` and `step_engine.py`. `naive.py` is a padded HF
  baseline.
- `bench/`: workload sources and builder, the vLLM backend, `run.py`, the benchmark suite (`suite.py`) and the HTML
  report.
- `results/`: the raw benchmark results the results site is built from (see `results/README.md`).
- `scripts/`: the decode step profiler, ridge calibration, a correctness check against HF, and the script that picks
  the median of three runs (`median_rows.py`).
- `tests/`: CPU tests for every stage (exact token match against HF in fp32) and GPU tests (every token is HF's top
  choice or within two bf16 ulps of it).
