"""One line per column or metric key: what it measures and how to read it. bench.report shows these as the hover
text of index columns and run-page rows, whose names it underlines dotted. A key without an entry is not underlined.

GLOSSARY is bench.run's meaning of a key. ENGINE overrides the keys that mean something else in the engine's
own record (details.json, and python -m batchinfer run's metrics.json): its rates divide by its own
inference time, not bench.run's wall s."""

GLOSSARY = {
    # the index's own columns (bench.report.index_columns)
    "run_dir": "The run's directory under results/. Opens its page, which lists every recorded number.",
    "prefix_reuse": "How the run reused the KV of shared prompt prefixes. global trie: batchinfer's prefix_sharing, "
                    "each shared block computed once and read by every request that has it. prefix cache: vLLM's "
                    "enable_prefix_caching, blocks cached as they are computed and hit by later requests. off: every "
                    "request prefills its whole prompt (the naive engine never reuses).",
    "other_opts": "The run's --opt values that have no column of their own.",
    "accuracy_pct": "Correct answers over every request of a scored source (MMLU, CLINC, QuALITY, GSM8K), in %. "
                    "Accuracy by source and the match against the reference run are on the run's page.",
    "engine": "batchinfer: ours (paged KV, one packed forward per step, continuous admission, prefix sharing). "
              "naive: the baseline (fixed batches in arrival order, left-padded, HF's DynamicCache). vllm: vLLM's "
              "offline engine, the reference. Opens the run's page; hover for its directory under results/.",
    "engine_config": "The engine, its order, and the flags that differ between the engine runs listed. Flags that "
                     "are the same in every run are on each run's page.",
    "commit": "git describe of the code that ran, @ the branch it came from when that was not main, and when the "
              "commit was made (UTC); sorts by that time.",
    "ran": "When the row's timed pass finished (UTC).",
    "vs_vllm": "wall s ÷ the wall s of vLLM with its prefix cache on, for the same workload file, model and GPU. "
               "1.00 is parity, 2.00 is twice as slow. Blank when that vLLM row is missing. On a run's page it links "
               "that vLLM row.",
    "max_num_seqs": "vLLM's cap on requests running at once, when the run set it (max_num_seqs). Blank: vLLM's "
                    "default, 256 on the A100 and 1,024 on the H100 for vLLM 0.30's offline engine.",
    "block_ceiling_pct": "The most prefix hit % can reach when KV is shared in whole 16-token blocks, from the "
                         "workload's prompts. hit % well under it is sharing left on the table.",
    "ms_per_step": "wall s ÷ steps, in ms, so scheduling and commit are in it too. A decode step reads every weight "
                   "once, so it cannot drop below model bytes ÷ memory bandwidth whatever the batch (Qwen3-8B bf16 is "
                   "~16.4 GB: ~10.5 ms on an A100-40GB, ~8 ms on an H100 PCIe). Only the time above that floor is "
                   "host or launch overhead. Prefill-heavy work (classify) runs few large steps, so there it "
                   "measures prefill chunks.",
    # bench.run: what the run was asked to do
    "workload": "The workload file: which requests ran, in what order, with what max_tokens.",
    "workload_sha256": "Hash of the workload file. --compare refuses a reference that ran another file.",
    "preset": "Workload preset (bench/workload.py PRESETS): the source mix it was built from.",
    "size": "quick or full: the request-count multiplier and the time budget.",
    "mix": "source:count list the workload was built from.",
    "seed": "Seed for drawing each source's requests.",
    "backend": "The backend bench.run drove.",
    "opts": "Options given with --opt, passed to the backend's constructor.",
    "model": "Hugging Face model id.",
    "gpu": "GPU the run used, as nvidia-smi names it.",
    "GPU": "GPU the run used, shortened.",
    "git_sha": "git describe of the code that ran; -dirty when a tracked file outside results/ differed from it.",
    "timestamp": "When the run finished.",
    "target_s": "The workload's time budget for the timed pass.",
    # bench.run: what happened
    "load_s": "Model load plus one warmup request. Not part of wall s. bench.suite runs every row of one engine "
              "configuration through one load, and those rows share its load s.",
    "wall_s": "The timed pass: the whole workload in one generate() call. Lower is better.",
    "requests": "Requests in the workload.",
    "req_per_s": "Requests ÷ wall s.",
    "input_tokens": "Prompt tokens over all requests.",
    "unique_input_tokens": "Prompt tokens left once shared prefixes are counted once (token level).",
    "output_tokens": "Generated tokens over all requests.",
    "input_tok_per_s": "Prompt tokens ÷ wall s.",
    "unique_input_tok_per_s": "Unique prompt tokens ÷ wall s: prefill work that no cache could skip.",
    "output_tok_per_s": "Generated tokens ÷ wall s.",
    "total_tok_per_s": "(Prompt + generated tokens) ÷ wall s.",
    "ideal_prefix_reuse": "Share of prompt tokens that repeat an earlier prompt's prefix (token level): the "
                          "ceiling for any prefix cache.",
    "gpu_util_mean": "nvidia-smi utilization.gpu over the timed pass: the share of time any kernel ran. Stays near "
                     "100% when launch gaps dominate, so it does not show a host-bound engine.",
    "mfu_pct": "Model FLOPs utilisation, the same count for every engine: the FLOPs the model needs for the tokens "
               "the engine computed (2 × matmul weights per query token, lm_head per sampled token, causal "
               "attention; prefix hits, padding and dead slots are not work) ÷ wall s ÷ this GPU's dense bf16 peak, "
               "in %. Compute-bound prefill drives it; decode-heavy rows stay low even at full bandwidth, so read "
               "it next to MBU %. Blank without a request trace or on a card with no listed peak.",
    "mbu_pct": "Model bandwidth utilisation, the same count for every engine: bytes the model must move (every "
               "matmul weight once per forward pass, plus the KV cache each query token reads and writes) ÷ wall s ÷ "
               "this GPU's HBM bandwidth, in %. A lower bound: chunked-prefill re-reads and activations are left "
               "out. Decode drives it. Blank without a forward-pass count or on a card with no listed bandwidth.",
    "utilisation": "The counts behind MFU % and MBU % (bench/utilisation.py).",
    "model_tflop": "FLOPs the model needs for the tokens computed (MFU's numerator), in TFLOP.",
    "attention_flop_pct": "Share of model_tflop that is causal attention (QKᵀ and PV); grows with context length.",
    "model_gb_moved": "Bytes the model must move (MBU's numerator), in GB: weights per forward pass plus KV reads "
                      "and writes.",
    "forward_passes": "Forward passes the engine ran: steps (batchinfer, vLLM iterations), or a prefill plus decode "
                      "steps per group (naive).",
    "query_tokens": "Tokens the engine ran through the model: prompt tokens minus prefix hits, plus fed-back "
                    "output tokens.",
    "peak_bf16_tflops": "This GPU's dense bf16 tensor-core peak (datasheet), MFU's denominator.",
    "peak_hbm_gb_per_s": "This GPU's HBM bandwidth (datasheet), MBU's denominator.",
    "shape_layers": "Decoder layers, from the model's config.",
    "shape_hidden": "Hidden size, from the model's config.",
    "shape_q_dim": "Attention heads × head dim, from the model's config.",
    "shape_kv_dim": "KV heads × head dim, from the model's config: one token's K (or V) per layer.",
    "shape_intermediate": "MLP intermediate size, from the model's config.",
    "shape_vocab": "Vocabulary size, from the model's config: lm_head is hidden × vocab.",
    "shape_dtype_bytes": "Bytes per weight and KV element (2 for bf16).",
    "gpu_mem_peak_mb": "Peak GPU memory in use during the timed pass (nvidia-smi), preallocated KV pool included.",
    "length_violations": "Requests with no output, more than max_tokens, finish_reason length short of max_tokens, "
                         "ignore_eos short of it, token_ids disagreeing with output_tokens, a finish_reason other "
                         "than stop or length, or (EOS honoured, with the workload's stop ids) a stop that does not "
                         "end on a stop id or a stop id before the last token. An output ends at max_tokens or at "
                         "its first stop token. Anything but 0 is a bug, and bench.run exits 1.",
    "over_budget": "wall s went over the workload's time budget (target_s).",
    # by source and compare
    "n": "Requests in this row.",
    "kind": "classify: answers of a few tokens (prefill-bound). generate: long outputs (decode-bound).",
    "trace": "vLLM backend: whether vLLM's per-iteration and per-request trace was recorded (details.json), so the "
             "page draws the same charts as for batchinfer. On unless --opt trace=false.",
    "mean_output_tokens": "Mean generated tokens per request.",
    "accuracy": "Share of scored requests answered correctly (choice, label or number scorer), in %. Blank for a "
                "source with no scorer (WildChat's open-ended chat).",
    "match": "classify: share with the same normalized answer as the reference. generate: mean share of the longer "
             "output (token ids up to the first stop) both share as a prefix. In %.",
    "identical": "Share of requests whose output matches the reference's in full: the same normalized answer "
                 "(classify), or the same token ids up to the first stop (generate). In %.",
    # batchinfer engine counters (backend_stats; batchinfer/metrics.py FLAT_KEYS and STEP_FLAT_KEYS)
    "prefill_s": "Time in prefill (naive engine).",
    "decode_s": "Time in decode (naive engine).",
    "prefill_tok_per_s": "Prompt tokens ÷ prefill time.",
    "decode_tok_per_s": "Decode tokens (live slots) ÷ decode time.",
    "mean_decode_batch": "Mean live decode rows per step that decoded anything (vLLM: per engine iteration, from its "
                         "trace). Higher amortizes each step's fixed cost over more tokens.",
    "wasted_decode_slots": "Decode slots spent on rows that had already finished (naive engine).",
    "padding_waste_pct": "Share of padded prompt tokens that were padding (naive engine).",
    "groups": "Static batches the plan made.",
    "max_batch_tokens": "Padded-token budget per fixed group (naive engine, and batchinfer with admission=groups).",
    "prefix_hit_pct": "Prompt tokens read from KV another request computed ÷ all prompt tokens, in %.",
    "peak_mem_reserved_gb": "Peak memory the torch allocator reserved, KV pool included, in GiB.",
    "steps": "Forward passes the engine ran (vLLM: its engine iterations, from its trace). With wall s it gives "
             "ms/step: the same step count at a higher ms/step is per-step overhead.",
    "decode_only_steps": "Steps that carried decode rows and no prefill tokens.",
    "mean_step_tokens": "Mean query tokens per step (decode rows + prefill tokens).",
    "dense_mfu_pct": "The batchinfer engine's own MFU: matmul FLOPs only (2 × parameters × tokens processed, plus "
                     "lm_head, no attention) ÷ inference time ÷ this GPU's dense bf16 peak, in %; mfu_pct is the "
                     "count every engine gets. Telling on prefill-heavy rows (classify), where compute is the bound. "
                     "A decode step reads every weight for a few rows' worth of FLOPs, so decode-heavy rows stay low "
                     "even at full memory bandwidth: compare those within one workload, and read ms/step for "
                     "per-step overhead.",
    "gpu_busy_pct": "CUDA-event time around each forward ÷ inference time. The events also count gaps where the "
                    "GPU waits for the host, so this reads high even for a launch-bound step.",
    "layers": "The forward a row ran. batchinfer: fused (its Qwen3 layers on vLLM's fused kernels, fused_layers) or HF "
              "(HF's layer modules, the non-optimised setup), · FA3 for FlashAttention-3. vLLM: compiled (torch.compile, "
              "its default) or eager (enforce_eager). naive: HF.",
    "graphs": "CUDA graphs. batchinfer: decode N% (cuda_graphs: the share of steps replayed from a graph, its "
              "decode-only steps; steps carrying prefill run eagerly) or off. vLLM: on (full and piecewise graphs) "
              "unless enforce_eager. naive: off.",
    "graph_step_pct": "Share of steps replayed from a captured CUDA graph (cuda_graphs=true): decode-only steps up to "
                      "the largest graph size. Every other step, including any step carrying prefill, ran eagerly.",
    "graph_rows": "Row count of the graph a step replayed (0: eager). The rows above decode_rows were pad rows.",
    "graph_capture_s": "Load-time cost of capturing every graph size, in s, over both captures when the pool is sized "
                       "on the card (the sizing's own and the run's); part of load_s, not of the timed pass.",
    "graph_sized_gb": "Device memory the pool sizing set aside for the graphs, GiB: every size captured over its scratch "
                      "pool, read from the card's free memory (the graphs' pool plus the driver's instantiated graphs).",
    "graph_held_gb": "Device memory the run's graphs hold, GiB, measured the same way at the real capture.",
    "fa_version": "FlashAttention version of the batchinfer engine: 2, or 3 (vLLM's choice on Hopper, sm90 only).",
    "fused_layers": "Qwen3's layers as one fused forward over HF's weights with vLLM's kernels (~11 launches a layer "
                    "instead of HF's ~57); off: HF's own layer modules.",
    "cuda_graphs": "Decode-only steps replayed from CUDA graphs captured at load (one launch per step); mixed steps "
                   "run eagerly.",
    "chain_start_step": "Step at which the longest decode chain started. The job cannot end before this plus its "
                        "length.",
    "prefill_budget": "Most prefill tokens a step may carry, decided before the first step. With "
                      "prefill_budget=adaptive it is the floor the scheduler raises from.",
    "prefill_budget_peak": "Largest prefill budget any step ran under. Above prefill_budget only with "
                           "prefill_budget=adaptive, which raises it when the decode work left shrinks (a long-capped "
                           "generation that stopped early).",
    "kv_occupancy_pct": "KV positions written ÷ KV held for the admitted requests, averaged over steps, in %. "
                        "Requests reserve prompt + max_tokens when admitted, so the rest is held for prompts admitted "
                        "ahead of their prefill chunks (most of it while prefill runs) and for tokens not generated "
                        "yet, or never when a request stops early.",
    "kv_reserved_peak_pct": "Most KV held for admitted requests at any step start ÷ the KV pool, in %. Requests "
                            "reserve prompt + max_tokens when admitted, so this is the pool admission needs.",
    "kv_written_peak_pct": "Most KV positions written by the end of any step, before finished requests release "
                           "theirs, ÷ the KV pool, in %, a shared block counted once: what an engine allocating KV as "
                           "it writes holds, to within a partly filled block per request. vLLM's KV cache usage and "
                           "BatchLLM's peak KV usage compare with this, not with kv_reserved_peak_pct.",
    "prefill_phase_steps": "Steps up to and including the last one that carried prefill tokens. The decode-only tail "
                           "after it has no prompt left to fill its steps with.",
    "prefill_small_step_pct": "Prefill-phase steps with fewer query tokens than MIN_PREFILL_BUDGET (256) ÷ "
                              "prefill_phase_steps, in %. Below that a step's GEMMs stay bound by reading the weights, "
                              "so the GPU's compute sits idle while prompts wait: BatchLLM's \"valleys\" (its Figs 2 "
                              "and 10).",
    "prefill_budget_adaptive": "Whether the batchinfer engine recomputed the prefill budget every step "
                               "(prefill_budget=adaptive) instead of holding prefill_budget for the whole job.",
    "order": "The order our engines admit requests in (vLLM takes them first come, first served).",
    "prefix_sharing": "Whether requests share KV blocks for common prompt prefixes, through the global prefix trie.",
    "prefix_wait_steps": "(Sequence, step) pairs with budget left that took no chunk because a shared block was "
                         "still being computed elsewhere.",
    "pinned_blocks_peak": "Most KV blocks held for requests not yet admitted.",
    "trie_blocks_peak": "Most KV blocks held by the prefix trie at once.",
    "ideal_prefix_reuse_page16": "Prefix reuse reachable in whole 16-token blocks: the ceiling for prefix_hit_pct.",
    # engine page sections (python -m batchinfer run and details.json)

    "analysis_s": "Time analysing the workload before scheduling.",
    "inference_s": "Time from the first step to the last.",
    "step_ms_mean": "Mean time per step in the executor's forward, in ms; scheduling (sched_ms_per_step) and "
                    "commit are outside it.",
    "sched_ms_per_step": "Mean scheduling time per step, in ms.",
}

# The engine's own record: the same key, another clock or another span.
ENGINE = {
    "req_per_s": "Requests ÷ inference_s, the engine's own clock (not bench.run's wall s).",
    "total_tok_per_s": "(Prompt + generated tokens) ÷ inference_s.",
    "load_s": "Tokenizer and model load (python -m batchinfer run).",
    "timestamp": "When the run started.",
    "probe_s": "The naive engine's memory probe: one forward at max_batch_tokens padded tokens, before the run.",
    "total_s": "From the start of this record to its end. python -m batchinfer run starts it before load; a "
               "bench.run engine's details.json covers its generate() call only.",
}
