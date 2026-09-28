"""How this invocation performed: phase timers, counters, a per-group trace (naive engine) or a
per-step trace (batchinfer engine), and a per-request trace (both).

One Metrics object per flow.run call, so the bench runner's warmup call never leaks into the timed pass.
to_dict() is everything (metrics.json, details.json). flat_stats() is a fixed whitelist of report columns
for a backend's stats(), because bench.report makes one column per key.

Who writes what:
- analysis (JobAnalysis.metrics) and policy (Policy.to_dict()): flow.run, through set_analysis and set_policy;
- meta: the CLI;
- counts, memory, model_shape, sync, flat_keys, groups (naive engine) or steps (batchinfer engine), request_trace:
  the engine that ran. set_analysis also copies unique_prompt_tokens into counts, a volume figure no engine counts;
- timing: one key per phase, written by whoever runs that phase: load and probe (the CLI), analysis and
  policy (flow.run), inference, prefill and decode (the engine).
flat_stats() reads the decided sections first (analysis, policy) and the measured ones after (counts,
timings, rates, memory). Decided and measured keys never share a name; tests/test_flow.py checks it.
"""
import time
from collections import Counter

from .kv import BLOCK_SIZE
from .schema import MIN_PREFILL_BUDGET

FLAT_KEYS = ("prefill_s", "decode_s", "prefill_tok_per_s", "decode_tok_per_s", "mean_decode_batch",
             "wasted_decode_slots", "padding_waste_pct", "groups", "order", "max_batch_tokens", "prefix_hit_pct",
             "peak_mem_reserved_gb")
# Batchinfer engine columns. mean_decode_batch, max_batch_tokens (admission="groups" only), prefix_hit_pct and
# peak_mem_reserved_gb mean the same as in FLAT_KEYS. The block-level reuse ceiling (an analysis fact) sits next
# to prefix_hit_pct: it is what the hit rate is judged against, where the runner's token-level
# ideal_prefix_reuse is not reachable with whole blocks. order and prefix_sharing are the policy's own knobs, so a
# row's label never has to carry them for the table to tell rows apart; they and prefill_budget / max_batch_tokens
# come from the policy section, prefill_budget_adaptive too. prefill_budget_peak is measured (the largest per-step
# budget in the trace): it differs from prefill_budget only when the adaptive budget rose. prefill_small_step_pct is
# the share of the prefill phase's steps under MIN_PREFILL_BUDGET query tokens (BatchLLM's "valleys"), and the two KV
# peaks are shares of the pool: reserved is what admission holds, written is what a pool allocated on demand holds.
PAGED_REUSE_KEY = f"ideal_prefix_reuse_page{BLOCK_SIZE}"
STEP_FLAT_KEYS = ("steps", "decode_only_steps", "mean_step_tokens", "prefill_small_step_pct", "mean_decode_batch",
                  "dense_mfu_pct", "gpu_busy_pct", "fused_layers", "cuda_graphs", "fa_version", "graph_step_pct",
                  "chain_start_step", "prefill_budget", "prefill_budget_adaptive",
                  "prefill_budget_peak", "max_batch_tokens", "order", "prefix_sharing", "prefix_hit_pct",
                  PAGED_REUSE_KEY, "prefix_wait_steps", "pinned_blocks_peak", "trie_blocks_peak", "kv_occupancy_pct",
                  "kv_reserved_peak_pct", "kv_written_peak_pct", "peak_mem_reserved_gb")
# end_ms: ms from the start of the engine's inference timer (the origin of inference_s) to the end of the step, after
# its commit and its results; a step starts at the previous step's end_ms (0 for step 0).
STEP_FIELDS = ("decode_rows", "prefill_tokens", "prefill_rows", "admitted", "free_blocks", "step_ms", "gpu_ms",
               "sched_ms", "trie_blocks", "pinned_blocks", "step_prefill_budget", "kv_reserved_tokens",
               "kv_written_tokens", "unprefilled_tokens", "end_ms", "graph_rows")
# One entry per request, in finish order. analysis_kind is AnalyzedRequest.kind (prefill_only | label | generate),
# named apart from bench's classify/generate kind. prefix_hit_tokens is the request's share of the prefix_hit_tokens
# count. The *_step fields index the step trace: the step that reserved the request's KV, ran its first prefill
# chunk, sampled its first token, finished it (None from an engine without steps). The *_ms fields are the same
# events in ms since the inference timer started: the start or end of that step (batchinfer engine); the group's
# start, its prefill's end and a finish apportioned within its decode time as latency_s is (naive engine).
REQUEST_FIELDS = ("idx", "id", "analysis_kind", "group", "prompt_len", "max_tokens", "output_tokens",
                  "prefix_hit_tokens", "finish_reason", "admitted_step", "prefill_start_step", "first_token_step",
                  "finished_step", "admitted_ms", "prefill_start_ms", "first_token_ms", "finished_ms")
# Dense bf16 tensor-core peak by torch.cuda.get_device_name(). dense_mfu_pct is relative to the run's own GPU,
# and None on a GPU missing here rather than measured against another card's peak.
BF16_DENSE_FLOPS = {"NVIDIA A100-SXM4-40GB": 312e12, "NVIDIA A100-SXM4-80GB": 312e12, "NVIDIA H100 PCIe": 756e12,
                    "NVIDIA H100 80GB HBM3": 989e12, "NVIDIA H200": 989e12}
# HBM bandwidth by the same name, bytes/s from the datasheets: bench/utilisation.py's mbu_pct.
HBM_BYTES_PER_S = {"NVIDIA A100-SXM4-40GB": 1555e9, "NVIDIA A100-SXM4-80GB": 2039e9, "NVIDIA H100 PCIe": 2000e9,
                   "NVIDIA H100 80GB HBM3": 3350e9, "NVIDIA H200": 4800e9}


class _Timer:
    """Context manager that synchronises the device (when the engine set a sync) at both ends and
    accumulates into metrics.timing[phase]; .elapsed holds this one interval."""

    def __init__(self, metrics, phase):
        self.metrics, self.phase, self.elapsed = metrics, phase, 0.0

    def __enter__(self):
        self.metrics.sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.metrics.sync()
        self.elapsed = time.perf_counter() - self.t0
        self.metrics.timing[self.phase] = self.metrics.timing.get(self.phase, 0.0) + self.elapsed


class Metrics:
    def __init__(self, sync=None, flat_keys=FLAT_KEYS):
        self.sync = sync or (lambda: None)  # the engine sets torch.cuda.synchronize when on a GPU
        self.flat_keys = flat_keys
        self.timing = {}  # phase -> seconds, accumulated across groups
        self.counts = Counter()
        self.memory = {}
        self.analysis = {}  # JobAnalysis.metrics: facts about the job
        self.policy = {}  # Policy.to_dict(): what policy decided for the job
        self.executor = {}  # the batchinfer executor's resolved build: fused_layers, cuda_graphs, fa_version
        self.meta = {}
        self.groups = []  # per-group trace (naive engine)
        self.steps = []  # per-step trace (batchinfer engine), tuples in STEP_FIELDS order
        self.request_trace = []  # per-request trace (both engines), tuples in REQUEST_FIELDS order
        self.model_shape = {}  # n_body_params, hidden, vocab: for dense_mfu_pct
        self._t0 = time.perf_counter()
        self._total = None

    def timer(self, phase):
        return _Timer(self, phase)

    def add(self, **counts):
        self.counts.update(counts)

    def add_group(self, **record):
        self.groups.append(record)

    def set_analysis(self, metrics):
        """JobAnalysis.metrics; the job-level count the engine does not count itself is copied."""
        self.analysis = dict(metrics)
        if "unique_prompt_tokens" in metrics:
            self.counts["unique_prompt_tokens"] = metrics["unique_prompt_tokens"]

    def set_policy(self, decided):
        """Policy.to_dict(): the decisions, recorded as made."""
        self.policy = dict(decided)

    def record_step(self, **fields):
        """One step of the trace, every STEP_FIELDS entry by name."""
        assert fields.keys() == set(STEP_FIELDS), f"record_step needs exactly {STEP_FIELDS}, got {sorted(fields)}"
        self.steps.append(tuple(fields[k] for k in STEP_FIELDS))

    def record_request(self, **fields):
        """One finished request of the trace, every REQUEST_FIELDS entry by name."""
        assert fields.keys() == set(REQUEST_FIELDS), \
            f"record_request needs exactly {REQUEST_FIELDS}, got {sorted(fields)}"
        self.request_trace.append(tuple(fields[k] for k in REQUEST_FIELDS))

    def finish(self):
        self._total = time.perf_counter() - self._t0

    # derived ------------------------------------------------------------------------------------------

    def timings(self):
        t = {f"{k}_s": round(v, 4) for k, v in self.timing.items()}
        t["total_s"] = round(self._total if self._total is not None else time.perf_counter() - self._t0, 4)
        return t

    def rates(self):
        c, t = self.counts, self.timing
        prefill_s, decode_s = t.get("prefill", 0.0), t.get("decode", 0.0)
        inference_s = t.get("inference") or (prefill_s + decode_s)
        steps = c["decode_steps"]
        return {
            "prefill_tok_per_s": _rate(c["prompt_tokens"], prefill_s),
            "padded_prefill_tok_per_s": _rate(c["padded_prompt_tokens"], prefill_s),
            # every live decode slot yields one token; the first token of each row comes from prefill,
            # so output_tokens == requests + live_slot_steps and only the latter is decode work
            "decode_tok_per_s": _rate(c["live_slot_steps"], decode_s),
            "total_tok_per_s": _rate(c["prompt_tokens"] + c["output_tokens"], inference_s),
            "req_per_s": _rate(c["requests"], inference_s, 2),
            "mean_decode_batch": round(c["live_slot_steps"] / steps, 2) if steps else None,
            "decode_step_ms_mean": round(1000 * decode_s / steps, 2) if steps else None,
            "wasted_decode_slots": c["decode_slot_steps"] - c["live_slot_steps"],
            "padding_waste_pct": (round(100 * (1 - c["prompt_tokens"] / c["padded_prompt_tokens"]), 2)
                                  if c["padded_prompt_tokens"] else None),
            # prompt positions read from KV another request computed, over all prompt positions. The batchinfer engine
            # counts prefill_tokens (what it computed itself); the naive engine computes every prompt token and reports 0
            "prefix_hit_pct": (round(100 * (1 - c["prefill_tokens"] / c["prompt_tokens"]), 2)
                               if c["prefill_tokens"] and c["prompt_tokens"] else 0.0),
        }

    def step_rates(self):
        """Derived from the per-step trace. Step tokens are query tokens (decode rows + prefill tokens)."""
        if not self.steps:
            return {}
        c, t = self.counts, self.timing
        cols = dict(zip(STEP_FIELDS, zip(*self.steps)))
        n = len(self.steps)
        with_decode = sum(1 for d in cols["decode_rows"] if d)
        inference_s = t.get("inference", 0.0)
        gpu_ms = [g for g in cols["gpu_ms"] if g is not None]  # None off the GPU
        shape = self.model_shape
        flops, peak = None, BF16_DENSE_FLOPS.get(self.memory.get("gpu_name"))
        occupied = [(w, r) for w, r in zip(cols["kv_written_tokens"], cols["kv_reserved_tokens"]) if r]
        tokens = [d + p for d, p in zip(cols["decode_rows"], cols["prefill_tokens"])]  # query tokens per step
        # the prefill phase: every step up to the last one carrying prefill tokens. The decode tail after it has no
        # prompt left to fill its steps with, so only the prefill phase's small steps are valleys a scheduler owns
        phase = tokens[:1 + max((i for i, p in enumerate(cols["prefill_tokens"]) if p), default=-1)]
        pool = BLOCK_SIZE * (c["kv_blocks"] - 1) if c.get("kv_blocks", 0) > 1 else None  # block 0 is the pad block
        # KV written by the end of each step, before commit releases finished requests: the step-start sample plus the
        # step's own writes, one position per query token (readers skip computed blocks, so nothing is written twice)
        written = [w + t for w, t in zip(cols["kv_written_tokens"], tokens)]
        if shape and inference_s:
            flops = (2 * shape["n_body_params"] * (sum(cols["prefill_tokens"]) + sum(cols["decode_rows"]))
                     + 2 * shape["hidden"] * shape["vocab"] * c["sampled_rows"])
        return {
            "steps": n,
            "decode_only_steps": sum(1 for d, p in zip(cols["decode_rows"], cols["prefill_tokens"]) if d and not p),
            "prefill_only_steps": sum(1 for d, p in zip(cols["decode_rows"], cols["prefill_tokens"]) if p and not d),
            "mean_step_tokens": round((sum(cols["decode_rows"]) + sum(cols["prefill_tokens"])) / n, 1),
            "prefill_phase_steps": len(phase),
            # prefill-phase steps too small to leave the weight-read bound (schema.MIN_PREFILL_BUDGET): the valleys
            "prefill_small_step_pct": (round(100 * sum(1 for x in phase if x < MIN_PREFILL_BUDGET) / len(phase), 1)
                                       if phase else None),
            "mean_decode_batch": round(sum(cols["decode_rows"]) / with_decode, 2) if with_decode else None,
            "dense_mfu_pct": round(100 * flops / inference_s / peak, 1) if flops and peak else None,
            "gpu_busy_pct": round(100 * sum(gpu_ms) / 1000 / inference_s, 1) if gpu_ms and inference_s else None,
            # steps replayed from a captured CUDA graph (decode-only steps within the largest graph size)
            "graph_step_pct": round(100 * sum(1 for g in cols["graph_rows"] if g) / n, 1),
            "step_ms_mean": round(sum(cols["step_ms"]) / n, 2),
            "sched_ms_per_step": round(sum(cols["sched_ms"]) / n, 2),
            "chain_start_step": c.get("chain_start_step"),
            "prefix_wait_steps": c.get("prefix_wait_steps"),
            "pinned_blocks_peak": max(cols["pinned_blocks"]) if "pinned_blocks" in cols else None,
            "trie_blocks_peak": max(cols["trie_blocks"]) if "trie_blocks" in cols else None,
            "prefill_budget_peak": max(cols["step_prefill_budget"]),  # above prefill_budget only when adaptive raised it
            # KV held for admitted requests' whole lives against KV written so far, both at step start. Held but
            # unwritten is prompts admitted ahead of their prefill chunks (unprefilled_tokens in the trace) plus
            # decode slots not generated yet; the first dominates while prefill runs.
            "kv_reserved_tokens_peak": max(cols["kv_reserved_tokens"]),
            "kv_occupancy_pct": round(100 * sum(w / r for w, r in occupied) / len(occupied), 1) if occupied else None,
            # peaks as shares of the pool: reserved at step start is what admission holds (nothing is reserved inside
            # a step); written at step end is what an engine allocating KV as it writes holds, to within a partly
            # filled block per sequence: the like-for-like with BatchLLM's and vLLM's KV usage
            "kv_reserved_peak_pct": round(100 * max(cols["kv_reserved_tokens"]) / pool, 1) if pool else None,
            "kv_written_peak_pct": round(100 * max(written) / pool, 1) if pool else None,
        }

    def to_dict(self):
        d = {"meta": dict(self.meta), "timing": self.timings(),
             "volume": {k: self.counts[k] for k in sorted(self.counts)},
             "rates": {**self.rates(), **self.step_rates()}, "memory": dict(self.memory),
             "analysis": dict(self.analysis), "policy": dict(self.policy), "groups": list(self.groups)}
        if self.executor:
            d["executor"] = dict(self.executor)
        if self.steps:
            d["steps"] = {name: list(col) for name, col in zip(STEP_FIELDS, zip(*self.steps))}
        if self.request_trace:
            d["request_trace"] = {name: list(col) for name, col in zip(REQUEST_FIELDS, zip(*self.request_trace))}
        return d

    def flat_stats(self):
        merged = {**self.analysis, **self.policy, **self.counts, **self.timings(), **self.rates(), **self.step_rates(),
                  **self.memory, **self.executor}
        return {k: merged.get(k) for k in self.flat_keys}

    def summary(self):
        t, r, c, m, a = self.timings(), self.rates(), self.counts, self.memory, self.analysis

        def s(key):
            return f"{t.get(key, 0.0):.2f}"

        paged = next((a[k] for k in a if k.startswith("ideal_prefix_reuse_page")), None)
        prefix = (f"prefix: hit {r['prefix_hit_pct']}%, ideal reuse {a.get('ideal_prefix_reuse', 0.0):.1%}"
                  + (f" ({paged:.1%} in whole blocks)" if paged is not None else "")
                  + f", in-batch {self.policy.get('in_batch_reuse', 0.0):.1%}")
        memory = (f"memory: peak reserved {m.get('peak_mem_reserved_gb')} GiB (allocated {m.get('peak_mem_allocated_gb')}, "
                  f"probe {m.get('probe_peak_reserved_gb')}) on {m.get('gpu_name')}")
        if self.steps:
            sr, p = self.step_rates(), self.policy
            return "\n".join([
                f"{c['requests']} requests | load {s('load_s')} s, probe {s('probe_s')} s, analysis {s('analysis_s')} s, "
                f"inference {s('inference_s')} s, total {s('total_s')} s",
                f"policy: order={p.get('order')} admission={p.get('admission')} chunk_prefill={p.get('chunk_prefill')} "
                f"prefill_budget={p.get('prefill_budget')} prefix_sharing={p.get('prefix_sharing')}",
                f"tokens: prompt {c['prompt_tokens']:,}, output {c['output_tokens']:,}; total {r['total_tok_per_s']} tok/s, "
                f"{r['req_per_s']} req/s; finished: {c['finish_stop']} stop, {c['finish_length']} length",
                f"steps: {sr['steps']} ({sr['decode_only_steps']} decode-only, {sr['prefill_only_steps']} prefill-only), "
                f"mean {sr['mean_step_tokens']} tokens ({sr['prefill_small_step_pct']}% of the "
                f"{sr['prefill_phase_steps']} prefill-phase steps under {MIN_PREFILL_BUDGET}), "
                f"mean decode batch {sr['mean_decode_batch']}, "
                f"{sr['step_ms_mean']} ms/step ({sr['sched_ms_per_step']} ms scheduling); GPU busy {sr['gpu_busy_pct']}%, "
                f"dense MFU {sr['dense_mfu_pct']}%; chain start step {sr['chain_start_step']}, "
                f"head blocked {c['head_blocked_steps']} steps, prefix waits {c['prefix_wait_steps']}",
                f"{memory}; KV pool {c['kv_blocks']} blocks, occupancy {sr['kv_occupancy_pct']}% of reserved, "
                f"peak {sr['kv_reserved_peak_pct']}% reserved and {sr['kv_written_peak_pct']}% written, "
                f"trie blocks peak {sr['trie_blocks_peak']}, pinned peak {sr['pinned_blocks_peak']}; {prefix}",
            ])
        return "\n".join([
            f"{c['requests']} requests in {self.policy.get('groups')} groups | load {s('load_s')} s, probe {s('probe_s')} s, "
            f"analysis {s('analysis_s')} s, inference {s('inference_s')} s (prefill {s('prefill_s')}, decode {s('decode_s')}), "
            f"total {s('total_s')} s",
            f"tokens: prompt {c['prompt_tokens']:,} (padded {c['padded_prompt_tokens']:,}, waste {r['padding_waste_pct']}%), "
            f"output {c['output_tokens']:,}; prefill {r['prefill_tok_per_s']} tok/s, decode {r['decode_tok_per_s']} tok/s, "
            f"total {r['total_tok_per_s']} tok/s, {r['req_per_s']} req/s",
            f"decode: {c['decode_steps']} steps, mean batch {r['mean_decode_batch']}, {r['wasted_decode_slots']} dead-slot steps, "
            f"{r['decode_step_ms_mean']} ms/step; finished: {c['finish_stop']} stop, {c['finish_length']} length",
            f"{memory}; {prefix}",
        ])


def _rate(n, seconds, digits=1):
    return round(n / seconds, digits) if seconds else None
