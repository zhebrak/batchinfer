"""Backends run a whole workload in one call.

A backend is any class constructed as Backend(model, **opts) with
    generate(requests) -> results
where each request is {id, prompt, max_tokens, ignore_eos, labels} and each result is
{id, text, output_tokens, finish_reason, token_ids (optional)}. Decoding is greedy.
token_ids are every generated id, including a stop token if one was sampled, and
output_tokens == len(token_ids) <= max_tokens. text is decoded without special tokens
(no <|im_end|>), as vLLM's skip_special_tokens=True does. Optional:
- stats() -> dict (prefix hit rate, peak KV, ...), copied into the metrics; details() -> dict, details.json: the
  backend's own record, which the run's page draws as charts (bench/charts.py);
- name: the engine's name in rows and reports (batchinfer, naive, vllm);
- load_opts: the constructor options that need a fresh load. Absent (vLLM) means every option does;
  the others are policy options, which configure(**policy_opts) sets for the next pass, starting from
  the backend's defaults, never from the previous pass's (bench.suite runs many passes per load);
- reset(): before every timed pass, outside the timer, back to the state a fresh load leaves.
Pass a registry name or any importable package.module:Class as --backend.
"""
import importlib
import sys
import time
from contextlib import contextmanager
from typing import Protocol

from batchinfer.analysis import kind_of  # torch-free: the same request kinds the engine's trace records
from batchinfer.metrics import REQUEST_FIELDS
from batchinfer.schema import Request

BACKENDS = {"dummy": "bench.backends:Dummy", "vllm": "bench.backends:VLLM",
            "batchinfer": "batchinfer.bench:BatchinferBackend", "naive": "batchinfer.bench:NaiveBackend"}


class Backend(Protocol):
    def __init__(self, model: str, **opts): ...

    def generate(self, requests: list[dict]) -> list[dict]: ...


def resolve(spec):
    module, _, name = BACKENDS.get(spec, spec).partition(":")
    if not name:
        raise ValueError(f"backend {spec!r} is neither one of {', '.join(BACKENDS)} nor package.module:Class")
    return getattr(importlib.import_module(module), name)


class Dummy:
    """No model: answers labels[0] or max_tokens copies of 'x'. For harness tests without a GPU."""
    name = "dummy"

    def __init__(self, model=None, drop=0, delay_s=0.0):
        self.drop, self.delay_s = int(drop), float(delay_s)

    def generate(self, requests):
        time.sleep(self.delay_s)
        return [{"id": r["id"], "text": r["labels"][0] if r["labels"] else "x" * r["max_tokens"],
                 "output_tokens": r["max_tokens"], "finish_reason": "length", "token_ids": [0] * r["max_tokens"]}
                for r in requests[self.drop:]]


class VLLM:
    """Reference row: vLLM's own offline engine and scheduler, not part of our engine.
    opts go to vllm.LLM, e.g. enable_prefix_caching=false, gpu_memory_utilization=0.9.

    trace (default on; --opt trace=false turns it off) records what our engines record, so the page draws the same
    charts: one row per engine iteration from vLLM's own iteration details (prefill tokens and requests, decode
    requests, running and waiting requests, KV cache usage) and one row per request from its request metrics
    (scheduled, first token, last token). It turns vLLM's stats logging on, replaces vLLM's stat loggers (the
    per-iteration log lines and Prometheus) with one collector, and wraps the engine core client's get_output to
    read each iteration's engine-core timestamp: the frontend receives nothing while generate() is still adding
    requests (about 0.9 s for mixed-quick), so its own clock would stamp those iterations late. The wrapper is in
    place during generate() only: left on the client it keeps a reference to it past generate(), and vLLM's
    shutdown then logs "engine core exited unexpectedly". Those are vLLM's unstable interfaces (checked on 0.30.0):
    when one is missing the run goes on untraced and details() says why, since the reference row must never break
    for a chart. Its cost is below run-to-run noise: four alternating pairs on mixed-quick, Qwen3-0.6B, H100
    (2026-09-27) timed 2.93 / 2.90 / 3.45 / 2.90 s traced against 2.86 / 3.40 / 2.95 / 2.91 s untraced."""
    name = "vllm"

    def __init__(self, model, trace=True, **opts):
        from vllm import LLM
        self.trace, self.trace_error, self.log, self.t0, self.request_rows = bool(trace), None, IterationLog(), None, []
        if self.trace:
            try:
                self.llm = LLM(model=model, seed=0, disable_log_stats=False, enable_logging_iteration_details=True,
                               **opts)
            except TypeError as e:  # a vLLM without these engine arguments
                self._untraced(f"vllm.LLM refused the trace arguments ({e})")
        if not self.trace:
            self.llm = LLM(model=model, seed=0, **opts)
        elif getattr(getattr(self.llm.llm_engine, "logger_manager", None), "stat_loggers", None) is None:
            self._untraced("vLLM's engine has no logger_manager.stat_loggers to collect iterations from")
        elif not callable(getattr(getattr(self.llm.llm_engine, "engine_core", None), "get_output", None)):
            self._untraced("vLLM's engine has no engine_core.get_output to read iteration timestamps from")
        else:
            self.llm.llm_engine.logger_manager.stat_loggers[:] = [self.log]

    def _untraced(self, reason):
        self.trace, self.trace_error = False, reason
        print(f"WARN: vllm trace unavailable, running untraced: {reason}", file=sys.stderr)

    def reset(self):
        """Drop the prefix cache, so a pass never reads blocks an earlier pass (or the warmup) computed."""
        if not self.llm.reset_prefix_cache():
            raise RuntimeError("vLLM refused to reset its prefix cache (requests still running?)")

    def generate(self, requests):
        from vllm import SamplingParams
        params = [SamplingParams(temperature=0, max_tokens=r["max_tokens"], ignore_eos=r["ignore_eos"]) for r in requests]
        self.t0 = time.monotonic()  # the trace's origin; vLLM's request timestamps are monotonic too
        self.log.start(self.t0)
        if self.trace:
            with self.log.timing(self.llm.llm_engine.engine_core):
                outputs = self.llm.generate([r["prompt"] for r in requests], params, use_tqdm=False)
        else:
            outputs = self.llm.generate([r["prompt"] for r in requests], params, use_tqdm=False)
        self.request_rows = []
        if self.trace:
            try:
                self.request_rows = [request_row(i, r, o, self.t0) for i, (r, o) in enumerate(zip(requests, outputs))]
            except Exception as e:  # RequestOutput's metrics are vLLM's too
                self.log.error = self.log.error or f"request metrics: {type(e).__name__}: {e}"
            if self.log.error:  # no partial trace: the rows so far would read as the whole run
                self.request_rows = []
                self._untraced(f"collecting the trace failed ({self.log.error})")
        return [{"id": r["id"], "text": o.outputs[0].text, "output_tokens": len(o.outputs[0].token_ids),
                 "finish_reason": o.outputs[0].finish_reason, "token_ids": list(o.outputs[0].token_ids)}
                for r, o in zip(requests, outputs)]

    def stats(self):
        return {"trace": self.trace}

    def details(self):
        """The last generate() call's trace, in batchinfer's record shape (bench/charts.py reads both alike)."""
        meta = {"engine": "vllm", "trace": self.trace}
        if self.trace_error:
            meta["trace_error"] = self.trace_error
        d = {"meta": meta}
        if self.trace and self.log.rows:
            d["steps"] = {k: [row[k] for row in self.log.rows] for k in ITERATION_FIELDS}
        if self.request_rows:
            d["request_trace"] = {k: [row[k] for row in self.request_rows] for k in REQUEST_FIELDS}
        return d


# One row per vLLM iteration. decode_rows, prefill_tokens, prefill_rows and end_ms mean what they mean in
# batchinfer's step trace (metrics.STEP_FIELDS): requests decoding (one token each without speculative decoding),
# prompt tokens computed (vLLM leaves prefix-cache hits out, as our count leaves shared blocks out), requests
# prefilling, and ms from generate()'s start to when the engine core built the iteration's outputs (its own
# monotonic timestamp, the clock of the request timestamps too). running, waiting and kv_cache_usage_pct are
# vLLM's own. vLLM's elapsed_ms is left out: with async scheduling it times only the wait on the previous batch's
# result, not a step.
ITERATION_FIELDS = ("decode_rows", "prefill_tokens", "prefill_rows", "end_ms", "running", "waiting",
                    "kv_cache_usage_pct")


class IterationLog:
    """A vLLM stat logger that keeps one row per engine iteration. Duck-typed to vLLM's StatLoggerBase: the logger
    manager calls only these four methods. LLMEngine.step() gets one iteration's outputs, then records its stats;
    timing() wraps the get_output that fetches them, so record() knows when the engine core built them. record()
    runs inside vLLM's engine loop, so it never raises: the first failure is kept as error (VLLM then goes on
    untraced), and from then on it records nothing."""

    def __init__(self):
        self.rows, self.t0, self.core_ts, self.error = [], None, None, None

    def start(self, t0):
        self.rows, self.t0, self.core_ts = [], t0, None

    @contextmanager
    def timing(self, core):
        """Wrap core.get_output while the block runs, keeping each fetched iteration's engine-core timestamp; then
        leave core exactly as it was."""
        own = "get_output" in vars(core)  # an instance attribute, or the class's method
        get_output = core.get_output

        def get_output_timed(*args, **kwargs):
            outputs = get_output(*args, **kwargs)
            self.core_ts = getattr(outputs, "timestamp", None) or None
            return outputs
        core.get_output = get_output_timed
        try:
            yield
        finally:
            if own:
                core.get_output = get_output
            else:
                del core.get_output

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        if self.error is not None:
            return
        try:
            details = getattr(scheduler_stats, "iteration_details", None)
            if details is None or details.is_dummy or self.t0 is None:
                return
            ts = self.core_ts or time.monotonic()  # the receive time only if vLLM gave no core timestamp
            if ts < self.t0:  # left over from the previous generate(): async scheduling runs one step past a stop
                return
            self.rows.append({"decode_rows": details.num_generation_requests, "prefill_tokens": details.num_ctx_tokens,
                              "prefill_rows": details.num_ctx_requests, "end_ms": round(1000 * (ts - self.t0), 3),
                              "running": scheduler_stats.num_running_reqs, "waiting": scheduler_stats.num_waiting_reqs,
                              "kv_cache_usage_pct": round(100 * scheduler_stats.kv_cache_usage, 2)})
        except Exception as e:  # vLLM's stats objects are not a stable interface; the row must not break for them
            self.error = f"iteration stats: {type(e).__name__}: {e}"

    def log(self):
        pass

    def log_engine_initialized(self):
        pass

    def record_sleep_state(self, *args, **kwargs):
        pass


def request_row(idx, request, output, t0):
    """One request_trace entry (metrics.REQUEST_FIELDS) from a vLLM RequestOutput. vLLM reserves KV when it first
    schedules a request, so admitted and prefill start are the same moment. Times are None when vLLM gave none."""
    stats = getattr(output, "metrics", None)

    def ms(name):
        ts = getattr(stats, name, None)
        return round(1000 * (ts - t0), 3) if ts else None

    completion = output.outputs[0]
    return {"idx": idx, "id": request["id"],
            "analysis_kind": kind_of(Request(**{k: request[k] for k in ("id", "prompt", "max_tokens", "ignore_eos",
                                                                         "labels")})),
            "group": None, "prompt_len": len(output.prompt_token_ids or ()), "max_tokens": request["max_tokens"],
            "output_tokens": len(completion.token_ids), "prefix_hit_tokens": output.num_cached_tokens or 0,
            "finish_reason": completion.finish_reason, "admitted_step": None, "prefill_start_step": None,
            "first_token_step": None, "finished_step": None, "admitted_ms": ms("scheduled_ts"),
            "prefill_start_ms": ms("scheduled_ts"), "first_token_ms": ms("first_token_ts"),
            "finished_ms": ms("last_token_ts")}
