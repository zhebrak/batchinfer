"""The batchinfer engine (ours), as StepEngine: the loop scheduler -> executor -> scheduler.commit, then emit results
and record metrics.

Every forward is one step: all decode rows plus prefill chunks, packed. What goes into a step is the
scheduler's decision under the job's Policy; the executor only runs it.
"""
import time

import torch

from .kv import BLOCK_SIZE, BlockAllocator
from .metrics import STEP_FLAT_KEYS
from .scheduler import Scheduler
from .schema import Hardware, Result

GiB = 2**30


class StepEngine:
    flat_keys = STEP_FLAT_KEYS  # the report columns this engine fills

    def __init__(self, executor, tokenizer):
        """The tokenizer only decodes finished sequences; analysis owns encoding."""
        self.executor, self.tokenizer = executor, tokenizer

    @property
    def hardware(self):
        """What this engine measured on its card (schema.Hardware), for policy.decide: its KV pool in token slots,
        and the card off which it was sized (None off the GPU, where the pool is whatever num_blocks said)."""
        ex = self.executor
        return Hardware(gpu_name=torch.cuda.get_device_name(ex.device) if ex.cuda else None,
                        total_gb=round(torch.cuda.mem_get_info(ex.device)[1] / GiB, 2) if ex.cuda else None,
                        max_batch_tokens_fit=None, kv_pool_tokens=(ex.num_blocks - 1) * BLOCK_SIZE)

    def run(self, job, policy, on_result, metrics):
        """Steps until every request finished, calling on_result with each Result as its request finishes."""
        if policy.engine != "batchinfer":
            raise ValueError(f"the batchinfer engine runs a policy decided for engine='batchinfer', "
                             f"not {policy.engine!r}")
        ex, cuda = self.executor, self.executor.cuda
        kv = BlockAllocator(ex.num_blocks, seed=ex.seed)
        sched = Scheduler(job, policy, kv, ex.stop_ids, ex.max_positions)
        metrics.flat_keys = self.flat_keys
        metrics.model_shape = dict(ex.model_shape)
        metrics.executor = dict(ex.facts)
        if cuda:
            metrics.sync = torch.cuda.synchronize
            metrics.memory.setdefault("gpu_name", torch.cuda.get_device_name())
            metrics.memory["probe_peak_reserved_gb"] = ex.probe_peak_reserved_gb
            metrics.memory.update(ex.memory_facts, kv_pool_tokens=(ex.num_blocks - 1) * BLOCK_SIZE)
            torch.cuda.reset_peak_memory_stats()
        finish = {"stop": 0, "length": 0}
        ends = []  # end_ms of every step so far: step k runs from ends[k - 1] (0 for step 0) to ends[k]
        with metrics.timer("inference") as inference:
            while not sched.done:
                t0 = time.perf_counter()
                step = sched.next_step()
                t1 = time.perf_counter()
                sampled, gpu_ms = ex.forward(step)
                t2 = time.perf_counter()
                finished = sched.commit(step, sampled)
                now = time.perf_counter()
                for s in finished:
                    finish[s.finish_reason] += 1
                    on_result(Result(id=job.requests[s.idx].req.id,
                                     text=self.tokenizer.decode(s.outputs, skip_special_tokens=True),
                                     token_ids=s.outputs, prompt_tokens=s.prompt_len, output_tokens=len(s.outputs),
                                     finish_reason=s.finish_reason, group=s.group,
                                     latency_s=round(now - s.admitted_at, 4)))
                ends.append(round(1000 * (time.perf_counter() - inference.t0), 3))
                decode_rows, prefill_tokens, prefill_rows = step.counts()
                metrics.record_step(decode_rows=decode_rows, prefill_tokens=prefill_tokens, prefill_rows=prefill_rows,
                                    **sched.occupancy(), step_ms=round(1000 * (t2 - t1), 3),
                                    gpu_ms=None if gpu_ms is None else round(gpu_ms, 3),
                                    sched_ms=round(1000 * (t1 - t0), 3), end_ms=ends[-1], graph_rows=ex.graph_rows)
                metrics.add(prefill_tokens=prefill_tokens, decode_rows_total=decode_rows,
                            sampled_rows=len(sampled))
                for s in finished:
                    it = job.requests[s.idx]
                    metrics.record_request(
                        idx=s.idx, id=it.req.id, analysis_kind=it.kind, group=s.group, prompt_len=s.prompt_len,
                        max_tokens=s.max_tokens, output_tokens=len(s.outputs), prefix_hit_tokens=s.hits,
                        finish_reason=s.finish_reason, admitted_step=s.admitted_step,
                        prefill_start_step=s.prefill_start_step, first_token_step=s.first_token_step,
                        finished_step=s.finished_step, admitted_ms=_start(ends, s.admitted_step),
                        prefill_start_ms=_start(ends, s.prefill_start_step), first_token_ms=ends[s.first_token_step],
                        finished_ms=ends[s.finished_step])
        metrics.add(requests=len(job.requests), prompt_tokens=sum(it.prompt_len for it in job.requests),
                    output_tokens=metrics.counts["sampled_rows"],  # every sampled id is an output token
                    finish_stop=finish["stop"], finish_length=finish["length"], kv_blocks=ex.num_blocks,
                    **sched.counters())
        assert metrics.counts["prefix_hit_tokens"] == metrics.counts["prompt_tokens"] - metrics.counts["prefill_tokens"]
        metrics.counts["chain_start_step"] = sched.chain_start_step  # None when no request ran to its first token
        if cuda:
            metrics.memory.update(peak_mem_allocated_gb=round(torch.cuda.max_memory_allocated() / GiB, 2),
                                  peak_mem_reserved_gb=round(torch.cuda.max_memory_reserved() / GiB, 2))


def _start(ends, k):
    """When step k started, in ms since inference started: the previous step's end."""
    return ends[k - 1] if k else 0.0
