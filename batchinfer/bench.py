"""Adapters so bench.run and bench.suite can drive either engine like any other backend:

    python -m bench.run workloads/mixed-quick.jsonl --backend naive
    python -m bench.run workloads/mixed-quick.jsonl --backend batchinfer \\
        --opt order=decode_ratio_desc --opt prefill_budget=auto --opt prefix_sharing=false
    python -m bench.run workloads/mixed-quick.jsonl --backend batchinfer \\
        --opt prefix_sharing=true --opt order=prefix_dfs --opt prefill_budget=583 --opt num_blocks=3750

(`batchinfer` and `naive` are bench/backends.py's names for batchinfer.bench:BatchinferBackend and :NaiveBackend.)
The runner times the constructor plus one warmup call as load_s, then for every timed pass calls reset(), and
configure() when the suite changes policy options between passes, then generate(); it copies stats() into
metrics.json as backend_stats (one report column per key) and writes details() (the full metrics, per-step trace
included) to details.json.

A backend's options are of two kinds. Load options (load_opts) shape what is loaded and need a fresh construction.
Every other option is a policy option: read only by policy.decide through a PolicyConfig, built by the one
_policy_config() of each backend, which both the constructor and configure() call, so a policy option means the
same whether it came with the load or with a later pass."""
from . import flow
from .io import from_rows
from .metrics import Metrics
from .schema import DEFAULT_BATCH_SIZE, NAIVE_ORDER, PolicyConfig


def to_rows(results):
    return [{"id": r.id, "text": r.text, "output_tokens": r.output_tokens, "finish_reason": r.finish_reason,
             "token_ids": r.token_ids} for r in results]


def budget_tokens(max_batch_tokens):
    """--opt max_batch_tokens: an int, or "auto" for what the engine measures on its card (policy.auto_max_batch_tokens)."""
    return max_batch_tokens if max_batch_tokens == "auto" else int(max_batch_tokens)


class _Backend:
    """What bench.run needs from an engine: its constructor sets tok, engine and cfg (a validated PolicyConfig,
    built before the model loads so a bad option fails fast); generate() runs one job through flow.run."""
    name = None  # the engine's name in results and reports (schema.ENGINES)
    load_opts = ()  # constructor options that need a fresh load; every other option is a policy option
    tok = engine = cfg = metrics = None

    @staticmethod
    def _policy_config(**policy_opts):
        raise NotImplementedError

    def configure(self, **policy_opts):
        """The policy options for the next generate(), from this engine's defaults, never from the previous call's."""
        self.cfg = self._policy_config(**policy_opts)

    def reset(self):
        """Before every timed pass, outside the timer: back to the state a fresh process is in after its load-time
        probe. No job state survives a run (the prefix trie and every KV block live and die with one flow.run), but
        the torch allocator keeps the previous pass's cached blocks, and the engines' reset_peak_memory_stats()
        starts the peak from what is still reserved. The naive engine's probe ends with an empty cache, so emptying
        it is that state; BatchinferBackend also re-runs its executor's probe."""
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, requests):
        self.metrics = Metrics()  # fresh per call: the warmup pass must not leak into the timed one
        results = []
        flow.run(from_rows(requests), self.tok, self.engine, self.cfg, results.append, self.metrics)
        return to_rows(results)

    def stats(self):
        return self.metrics.flat_stats() if self.metrics else {}

    def details(self):
        return self.metrics.to_dict() if self.metrics else {}


class NaiveBackend(_Backend):
    """The naive baseline: fixed groups, in arrival order by default, run as left-padded HF batches (naive.py)."""
    name = "naive"
    load_opts = ("attn", "dtype", "probe")

    def __init__(self, model, attn="sdpa", dtype="bfloat16", probe=True, **policy_opts):
        """max_batch_tokens="auto" (the default) runs NaiveEngine.fit(), which measures the budget on this card and
        probes at it; an explicit budget is probed unless probe=false. A later pass may not configure a larger budget
        than the one probed."""
        from transformers import AutoTokenizer

        from .naive import NaiveEngine
        self.cfg = self._policy_config(**policy_opts)
        auto = self.cfg.max_batch_tokens == "auto"
        if auto and not probe:
            raise ValueError("probe=false needs an explicit max_batch_tokens: 'auto' is what the probe measures")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.engine = NaiveEngine(model, self.tok, dtype=dtype, attn=attn)
        if auto:
            self.engine.fit()
        elif probe:
            self.engine.probe(self.cfg.max_batch_tokens)

    @staticmethod
    def _policy_config(order=NAIVE_ORDER, max_batch_tokens="auto", batch_size=DEFAULT_BATCH_SIZE):
        return PolicyConfig(engine="naive", order=order, admission="groups", prefix_sharing=False,
                            max_batch_tokens=budget_tokens(max_batch_tokens),
                            max_batch_size=int(batch_size)).validate()


def _flag(name, value):
    """A build option as given, never coerced (bool('off') is True): true, false or 'auto'."""
    if not isinstance(value, bool) and value != "auto":
        raise ValueError(f"{name} must be true, false or auto, not {value!r} (--opt/--variant take true/false, on/off "
                         f"or auto)")
    return value


class BatchinferBackend(_Backend):
    """The batchinfer engine (ours): every forward carries all decode rows plus prefill chunks, under a Policy."""
    name = "batchinfer"
    load_opts = ("reserve_gb", "num_blocks", "dtype", "seed", "fa_version", "fused_layers", "cuda_graphs")

    def __init__(self, model, reserve_gb="auto", num_blocks=None, dtype="bfloat16", seed=None, fa_version=2,
                 fused_layers="auto", cuda_graphs="auto", **policy_opts):
        """num_blocks sets the KV pool size directly (block 0 is the pad block), so a row that must bind on
        memory means the same thing on every card; by default the pool is sized on the card (reserve_gb="auto":
        the largest step measured, the pool up to sizing.MAX_PEAK_SHARE), so it differs between a 40 GB A100 and
        an 80 GB H100. fa_version: FlashAttention 2, or 3 on sm90 (executor.check_fa_version refuses anything else,
        strings and bools included). fused_layers: Qwen3's layers fused with vLLM's kernels (qwen3_fused.py). cuda_graphs: decode-only
        steps replayed from CUDA graphs captured at load. Both default to "auto", on exactly where they can run (Qwen3
        on a GPU; graphs with FlashAttention-2); false runs HF's layers eagerly, the non-optimised setup."""
        from transformers import AutoTokenizer

        from .executor import Executor
        from .step_engine import StepEngine
        self.cfg = self._policy_config(**policy_opts)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.engine = StepEngine(Executor(model, dtype=dtype,
                                          reserve_gb=reserve_gb if reserve_gb == "auto" else float(reserve_gb),
                                          num_blocks=int(num_blocks) if num_blocks else None, seed=seed,
                                          fa_version=fa_version, fused_layers=_flag("fused_layers", fused_layers),
                                          cuda_graphs=_flag("cuda_graphs", cuda_graphs)),
                           self.tok)

    def reset(self):
        """A fresh load leaves the allocator holding the executor's probe (the widest step), which it does not
        release, so after emptying the cache the probe runs again: every pass starts from that same state."""
        super().reset()
        ex = self.engine.executor
        if ex.cuda and ex.probe_peak_reserved_gb is not None:
            ex.probe()

    @staticmethod
    def _policy_config(order=PolicyConfig.order, admission=PolicyConfig.admission,
                       chunk_prefill=PolicyConfig.chunk_prefill, prefill_budget=PolicyConfig.prefill_budget,
                       max_batch_tokens="auto", batch_size=DEFAULT_BATCH_SIZE,
                       prefix_sharing=PolicyConfig.prefix_sharing):
        return PolicyConfig(engine="batchinfer", order=order, admission=admission, chunk_prefill=chunk_prefill,
                            prefill_budget=prefill_budget, max_batch_tokens=budget_tokens(max_batch_tokens),
                            max_batch_size=int(batch_size),
                            prefix_sharing=prefix_sharing).validate()  # strings refused, never coerced
