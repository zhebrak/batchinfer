"""The batchinfer engine (ours), as StepEngine: the loop scheduler -> executor -> scheduler.commit, then emit results
and record metrics.

Every forward is one step: all decode rows plus prefill chunks, packed. What goes into a step is the
scheduler's decision under the job's Policy; the executor only runs it.
"""
import time

import torch

from .kv import BlockAllocator
from .metrics import STEP_FLAT_KEYS
from .scheduler import Scheduler
from .schema import Hardware, Result

GiB = 2**30


class StepEngine:
    flat_keys = STEP_FLAT_KEYS  # the report columns this engine fills

    def __init__(self, executor, tokenizer):
        """The tokenizer only decodes finished sequences; analysis owns encoding. The allocator and the blocks a
        cache=radix job keeps for the next one live here, across jobs."""
        from collections import OrderedDict
        if getattr(executor, "pool_mode", None) == "contiguous" and getattr(executor, "fa_version", None) == 3:
            # FlashAttention-3's varlen read over the regions (each row's keys at its slot, cu_seqlens_k out of order,
            # seqused_k) faults on steps with prefill on sm90 (an illegal memory access in the kernel itself, with
            # or without graphs; decode-only steps read correctly, which scripts/decode_step_profile.py measures)
            raise ValueError("pool=contiguous with FlashAttention-3 faults on steps with prefill on sm90; run it "
                             "with fa_version=2, or the default paged pool")
        self.executor, self.tokenizer = executor, tokenizer
        self.kv = None  # the pool's allocator, made at the first run and kept: kept blocks stay allocated between jobs
        self.retained = OrderedDict()  # cache=radix: prefix key -> block kept from an earlier job, oldest first

    def drop_retained(self):
        """Releases every block kept from earlier jobs (a cache=job run, or the bench's reset between passes)."""
        if self.kv is not None and self.retained:
            self.kv.release(list(self.retained.values()))
            self.retained.clear()

    @property
    def block_size(self):
        """The executor's KV block size: analysis builds the trie at it (flow.run)."""
        return self.executor.block_size

    def prepare(self, job):
        """Between analysis and policy (flow.run): the executor built for this job, when its load left that for
        here (Executor(build=False)), so the graphs it captures fit the job's decoding requests."""
        if self.executor.limits is None:
            self.executor.build(decoding=job.metrics["requests"] - job.metrics["n_prefill_only"])

    @property
    def hardware(self):
        """What this engine measured on its card (schema.Hardware), for policy.decide: its KV pool in token slots,
        and the card off which it was sized (None off the GPU, where the pool is whatever num_blocks said)."""
        ex, lim = self.executor, self.executor.limits
        return Hardware(gpu_name=torch.cuda.get_device_name(ex.device) if ex.cuda else None,
                        total_gb=round(torch.cuda.mem_get_info(ex.device)[1] / GiB, 2) if ex.cuda else None,
                        max_batch_tokens_fit=None,
                        kv_pool_tokens=(lim.num_blocks - 1) * lim.block_size if lim else None,
                        kv_block_size=lim.block_size if lim else None, ridge_tokens=lim.ridge_tokens if lim else None)

    def run(self, job, policy, on_result, metrics):
        """Steps until every request finished, calling on_result with each Result as its request finishes."""
        if policy.engine != "batchinfer":
            raise ValueError(f"the batchinfer engine runs a policy decided for engine='batchinfer', "
                             f"not {policy.engine!r}")
        ex, cuda, lim = self.executor, self.executor.cuda, self.executor.limits
        if lim is None:
            raise RuntimeError("the executor is not built: flow.run calls StepEngine.prepare(job) before run")
        if self.kv is None:  # pool=elastic: lowest id first, live blocks counted per group (kv.py)
            self.kv = BlockAllocator(lim.num_blocks, seed=ex.seed, group=lim.blocks_per_group)
        if policy.cache != "radix":
            self.drop_retained()
        kv = self.kv
        sched = Scheduler(job, policy, kv, ex.stop_ids, lim, retained=self.retained)
        metrics.flat_keys = self.flat_keys
        metrics.model_shape = dict(ex.model_shape)
        metrics.executor = dict(ex.facts)
        if cuda:
            metrics.sync = torch.cuda.synchronize
            metrics.memory.setdefault("gpu_name", torch.cuda.get_device_name())
            metrics.memory["probe_peak_reserved_gb"] = ex.probe_peak_reserved_gb
            metrics.memory.update(ex.memory_facts, kv_pool_tokens=(lim.num_blocks - 1) * lim.block_size)
            torch.cuda.reset_peak_memory_stats()
        run = _Run(job, sched, on_result, metrics)
        try:
            with metrics.timer("inference") as inference:
                run.t0 = inference.t0
                self._loop(run)
        except BaseException:  # a job that died mid-step leaves blocks allocated: the next job gets a fresh pool
            self.kv, self.retained = None, type(self.retained)()
            ex.reset()
            raise
        ex.finish_job()
        metrics.add(requests=len(job.requests), prompt_tokens=sum(it.prompt_len for it in job.requests),
                    output_tokens=metrics.counts["sampled_rows"],  # every sampled id is an output token
                    finish_stop=run.finish["stop"], finish_length=run.finish["length"], kv_blocks=ex.num_blocks,
                    kv_block_size=lim.block_size,
                    **sched.counters())
        # every prompt position was read from shared KV or prefilled, and what a preemption prefilled again on top
        assert (metrics.counts["prefix_hit_tokens"] + metrics.counts["prefill_tokens"]
                == metrics.counts["prompt_tokens"] + metrics.counts["recomputed_tokens"])
        metrics.counts["chain_start_step"] = sched.chain_start_step  # None when no request ran to its first token
        if cuda:
            metrics.memory.update(peak_mem_allocated_gb=round(torch.cuda.max_memory_allocated() / GiB, 2),
                                  peak_mem_reserved_gb=round(torch.cuda.max_memory_reserved() / GiB, 2))

    def _loop(self, run):
        """The steps, until every request finished. Each turn plans a step and launches it (executor.launch), then
        settles one: this one, or under overlap the one launched the turn before, so the host plans and stages a
        step while the previous one runs (the scheduler then returns None when everything live has its last token
        in flight, and the turn only settles)."""
        ex, sched, pending = self.executor, run.sched, None
        while pending is not None or not sched.done:
            planned = None
            if not sched.done:
                t0 = time.perf_counter()
                step = sched.next_step()
                if step is not None:  # with the groups its new blocks need (elastic)
                    planned = (step, ex.launch(step, self.kv.take_map()), t0, time.perf_counter())
            if sched.overlap:  # settle the step before; this one stays in flight
                planned, pending = pending, planned
            assert planned is not None or pending is not None, "nothing planned, nothing in flight"
            if planned is not None:
                self._settle(run, planned)

    def _settle(self, run, planned):
        """Waits for a launched step's ids, commits them, emits the finished requests and records the step."""
        job, sched, metrics, ends = run.job, run.sched, run.metrics, run.ends
        step, handle, t0, t1 = planned
        sampled, gpu_ms = handle.wait()
        t2 = time.perf_counter()
        discarded_before = sched.discarded
        finished = sched.commit(step, sampled)
        self.executor.release(self.kv.take_unmap())  # the groups the commit freed go after the last step that may touch them
        now = time.perf_counter()
        for s in finished:
            run.finish[s.finish_reason] += 1
            run.on_result(Result(id=job.requests[s.idx].req.id,
                                 text=self.tokenizer.decode(s.outputs, skip_special_tokens=True),
                                 token_ids=s.outputs, prompt_tokens=s.prompt_len, output_tokens=len(s.outputs),
                                 finish_reason=s.finish_reason, group=s.group,
                                 latency_s=round(now - s.admitted_at, 4)))
        ends.append(round(1000 * (time.perf_counter() - run.t0), 3))
        decode_rows, prefill_tokens, prefill_rows = step.counts()
        graph_rows, graph_tokens = handle.key
        # step_ms: from the launch to the ids on the host (under overlap, the wait behind the step before included)
        metrics.record_step(decode_rows=decode_rows, prefill_tokens=prefill_tokens, prefill_rows=prefill_rows,
                            **sched.occupancy(), step_ms=round(1000 * (t2 - t1), 3),
                            gpu_ms=None if gpu_ms is None else round(gpu_ms, 3),
                            sched_ms=round(1000 * (t1 - t0), 3), end_ms=ends[-1], graph_rows=graph_rows,
                            graph_tokens=graph_tokens)
        metrics.add(prefill_tokens=prefill_tokens, decode_rows_total=decode_rows,
                    sampled_rows=len(sampled) - (sched.discarded - discarded_before))  # a discarded id is no output
        for s in finished:
            it = job.requests[s.idx]
            metrics.record_request(
                idx=s.idx, id=it.req.id, analysis_kind=it.kind, group=s.group, prompt_len=s.prompt_len,
                max_tokens=s.max_tokens, output_tokens=len(s.outputs), prefix_hit_tokens=s.hit_share,
                finish_reason=s.finish_reason, admitted_step=s.admitted_step,
                prefill_start_step=s.prefill_start_step, first_token_step=s.first_token_step,
                finished_step=s.finished_step, admitted_ms=_start(ends, s.admitted_step),
                prefill_start_ms=_start(ends, s.prefill_start_step), first_token_ms=ends[s.first_token_step],
                finished_ms=ends[s.finished_step])


class _Run:
    """One job's run: what the loop and the settling of every step read and fill."""

    def __init__(self, job, sched, on_result, metrics):
        self.job, self.sched, self.on_result, self.metrics = job, sched, on_result, metrics
        self.finish = {"stop": 0, "length": 0}
        self.ends = []  # end_ms of every step so far: step k runs from ends[k - 1] (0 for step 0) to ends[k]
        self.t0 = None  # when inference started


def _start(ends, k):
    """When step k started, in ms since inference started: the previous step's end."""
    return ends[k - 1] if k else 0.0
