"""Adapters so bench.run and bench.suite can drive either engine like any other backend:

    python -m bench.run workloads/mixed-quick.jsonl --backend naive
    python -m bench.run workloads/mixed-quick.jsonl --backend batchinfer \\
        --opt order=decode_ratio_desc --opt prefill_budget=auto --opt sharing=none
    python -m bench.run workloads/mixed-quick.jsonl --backend batchinfer \\
        --opt sharing=token --opt order=prefix_dfs --opt prefill_budget=583 --opt num_blocks=3750

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
        return PolicyConfig(engine="naive", order=order, admission="groups", sharing="none",
                            max_batch_tokens=budget_tokens(max_batch_tokens),
                            max_batch_size=int(batch_size)).validate()


def _flag(name, value):
    """A build option as given, never coerced (bool('off') is True): true, false or 'auto'."""
    if not isinstance(value, bool) and value != "auto":
        raise ValueError(f"{name} must be true, false or auto, not {value!r} (--opt/--variant take true/false, on/off "
                         f"or auto)")
    return value


# How a load option's value is read from --opt text before it reaches the executor; the rest pass through as parsed.
COERCE = {"reserve_gb": lambda v: v if v == "auto" else float(v), "num_blocks": lambda v: int(v) if v else None,
          "cuda_graphs": lambda v: _flag("cuda_graphs", v),
          "block_size": int, "fa_version": lambda v: v if v == "auto" else int(v),
          "ridge_tokens": lambda v: int(v) if v else None, "overlap": lambda v: v}


class BatchinferBackend(_Backend):
    """The batchinfer engine (ours): every forward carries all decode rows plus prefill chunks, under a Policy."""
    name = "batchinfer"
    # The executor's keyword parameters, every one of them but the internal ones (tests/test_step_engine_cpu.py
    # checks this list against Executor.__init__'s signature, so a new executor knob reaches --opt and the suite's
    # arms through this one list). Their defaults live in Executor.__init__ alone: an option not given is not passed.
    load_opts = ("dtype", "reserve_gb", "num_blocks", "seed", "fa_version", "layers", "cuda_graphs", "graph_max_rows",
                 "block_size", "ridge_tokens", "quant", "kv_dtype", "overlap", "attention", "pool")
    internal = ("model_name", "device", "model", "probe", "build", "compile_backend")  # tests' and scripts' knobs

    def __init__(self, model, **opts):
        """Load options (load_opts) build the executor: num_blocks sets the KV pool size directly (block 0 is the pad
        block), so a row that must bind on memory means the same thing on every card; by default the pool is sized
        on the card (reserve_gb="auto": the largest step measured, the pool up to sizing.MAX_PEAK_SHARE). fa_version:
        FlashAttention 2, 3, or auto (3 where it runs: sm90). layers: fused (Qwen3's layers with vLLM's kernels,
        qwen3_fused.py), compiled (HF's layers under torch.compile) or hf (HF's layers eagerly, the non-optimised
        setup), auto by default (the first that can run); cuda_graphs: steps replayed from CUDA graphs captured at
        load, auto by default. Every other option is a policy option (_policy_config)."""
        from transformers import AutoTokenizer

        from .executor import Executor
        from .step_engine import StepEngine
        build = {k: COERCE.get(k, lambda v: v)(v) for k, v in opts.items() if k in self.load_opts}
        self.cfg = self._policy_config(**{k: v for k, v in opts.items() if k not in self.load_opts})
        self.tok = AutoTokenizer.from_pretrained(model)
        self.engine = StepEngine(Executor(model, **build), self.tok)

    def stats(self):
        """The engine's flat stats plus its usable KV capacity in tokens: the fairness column beside vLLM's."""
        return {**super().stats(), "kv_capacity_tokens": self.engine.hardware.kv_pool_tokens}

    def reset(self):
        """A fresh load leaves the allocator holding the executor's probe (the widest step), which it does not
        release, so after emptying the cache the probe runs again: every pass starts from that same state. Blocks a
        cache=radix job kept for the next one are dropped too: a pass never reads KV an earlier pass computed."""
        super().reset()
        self.engine.drop_retained()
        ex = self.engine.executor
        if ex.cuda and ex.probe_peak_reserved_gb is not None:
            ex.probe()

    @staticmethod
    def _policy_config(order=PolicyConfig.order, admission=PolicyConfig.admission,
                       chunk_prefill=PolicyConfig.chunk_prefill, prefill_budget=PolicyConfig.prefill_budget,
                       max_batch_tokens="auto", batch_size=DEFAULT_BATCH_SIZE,
                       sharing=PolicyConfig.sharing, cache=PolicyConfig.cache, memory=PolicyConfig.memory,
                       watermark_steps=PolicyConfig.watermark_steps):
        return PolicyConfig(engine="batchinfer", order=order, admission=admission, chunk_prefill=chunk_prefill,
                            prefill_budget=prefill_budget, max_batch_tokens=budget_tokens(max_batch_tokens),
                            max_batch_size=int(batch_size), sharing=sharing, cache=cache, memory=memory,
                            watermark_steps=int(watermark_steps)).validate()
