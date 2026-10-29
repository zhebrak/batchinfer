"""scheduler: every knob combination keeps the invariants a paged batchinfer engine depends on, with and without
prefix sharing. Pure Python: a fake executor returns a fixed id (or a stop id after a set number of outputs)."""
import dataclasses
from collections import OrderedDict
import itertools
import random

import pytest

from batchinfer.analysis import analyze
from batchinfer.kv import DEFAULT_BLOCK_SIZE as BLOCK_SIZE, PAD_BLOCK, BlockAllocator, reservation_blocks
from batchinfer.metrics import STEP_FIELDS
from batchinfer.policy import decide
from batchinfer.scheduler import COMPUTED, FREE, PENDING, Scheduler, Sequence
from batchinfer.schema import MAX_PREFILL_BUDGET, MAX_SEQS, Hardware, Limits, PolicyConfig, Request


def limits(num_blocks, max_positions=None, graph_rows=(), block_size=BLOCK_SIZE, graph_tokens=(), ridge_tokens=None,
           overlap=False, attention="fa", pool="static"):
    """What the executor would publish for a pool of num_blocks (schema.Limits), for tests that need one."""
    return Limits(num_blocks, block_size, MAX_SEQS, MAX_PREFILL_BUDGET, max_positions, graph_rows, graph_tokens,
                  ridge_tokens, "bf16", overlap, attention, pool)


RIDGE = Hardware(gpu_name=None, total_gb=None, max_batch_tokens_fit=None, kv_pool_tokens=None, ridge_tokens=24)
# a card calibrated at a ridge of 24 tokens, small enough for the harness's jobs to fill
from stub import StubTokenizer

TOKEN, STOP = 7, 99
BUDGET = 8


def workload(n=24):
    """Prompts of 3-37 words, max_tokens 1-12; every third request stops naturally, every fourth ignores EOS."""
    reqs = []
    for i in range(n):
        plen, mt = 3 + (i * 11) % 35, 1 + (i * 5) % 12
        reqs.append(Request(id=f"r{i}", prompt=" ".join(f"w{i}_{j}" for j in range(plen)), max_tokens=mt,
                            ignore_eos=i % 4 == 0))
    return analyze(reqs, StubTokenizer())


def shared_workload(n=18, groups=3, shared=35):
    """`groups` prefixes of `shared` words (two full blocks and a bit) with unique suffixes of 2-14 words and
    max_tokens 1-9, plus a duplicate of the first prompt. Every third request stops naturally, every fourth
    ignores EOS."""
    reqs = []
    for i in range(n):
        words = [f"p{i % groups}_{j}" for j in range(shared)] + [f"u{i}_{j}" for j in range(2 + (i * 5) % 13)]
        reqs.append(Request(id=f"s{i}", prompt=" ".join(words), max_tokens=1 + (i * 5) % 9, ignore_eos=i % 4 == 0))
    reqs.append(Request(id="dup", prompt=reqs[0].prompt, max_tokens=3))
    return analyze(reqs, StubTokenizer())


def stops_after(job):
    """Outputs after which the fake model emits STOP: every third request stops at half its max_tokens."""
    return {i: max(1, it.req.max_tokens // 2) for i, it in enumerate(job.requests) if i % 3 == 0}


def check_blocks(sched, kv):
    """Never the pad block, never double-allocated, never over-allocated; a shared block is counted once."""
    private = [b for s in sched.admitted.values() for b in s.private_blocks]
    shared = [b for b, st in zip(sched.node_block, sched.node_state) if st != FREE]
    kept = list(sched.retained.values()) if sched.retained is not None else []  # cache=radix: for the next job
    parked = [b for blocks, _, _ in sched.parked.values() for b in blocks]  # preempt: a preempted row's, until needed
    live = private + shared + kept + parked
    assert PAD_BLOCK not in live and len(live) == len(set(live)) and kv.free >= 0
    assert kv.free + len(live) == kv.capacity and sched.trie_blocks == len(shared)
    pinned = sum(1 for nid, st in enumerate(sched.node_state) if st != FREE and sched.node_admitted[nid] == 0)
    assert sched.pinned_blocks == pinned
    for s in sched.admitted.values():
        assert s.block_ids == [sched.node_block[nid] for nid in s.path] + s.private_blocks


def check_rows(sched, step, policy):
    """Rows read only computed nodes, nodes an earlier row of the step completes, or blocks they wrote
    themselves; write only their own pending nodes or private blocks, and never the same slot twice in a step;
    the budget holds; the earliest admitted sequence with prompt left always gets a chunk."""
    B = sched.B
    written, completed_now = set(), set()
    for r in step.rows:
        s = sched.admitted[r.seq]
        end = r.start + len(r.token_ids)
        for k, nid in enumerate(s.path):
            lo, hi = k * B, (k + 1) * B
            if lo < r.start:  # read by this row
                assert sched.node_state[nid] == COMPUTED or sched.node_owner[nid] == r.seq or nid in completed_now
            if lo < end and hi > r.start:  # written by this row (under overlap, marked computed as it was planned)
                assert sched.node_owner[nid] == r.seq and (sched.node_state[nid] == PENDING or sched.overlap)
            if lo < end and hi <= end and sched.node_owner[nid] == r.seq:
                completed_now.add(nid)  # whole by the end of this row (begun earlier or not): the rows after it may read it
        for p in range(r.start, end):
            slot = (r.block_ids[p // B], p % B)
            assert slot not in written, "two rows write one slot"
            written.add(slot)
    for src, dst, k in step.copies:  # sharing=token: from a block computed before this step, into a row's own block
        assert 0 < k < B and policy.sharing == "token"
        assert any(src == sched.node_block[nid] for nid, st in enumerate(sched.node_state) if st == COMPUTED) \
            or any(src in s.private_blocks and s.computed > 0 for s in sched.admitted.values())
        owner = next(s for s in sched.admitted.values() if dst in s.block_ids)
        d = owner.block_ids.index(dst)
        start = next((r.start for r in step.rows if r.seq == owner.idx), None)  # its chunk may wait for budget
        assert start is None or start == d * B + k, "the chunk starts after the copy"
        for slot in range(k):
            assert (dst, slot) not in written, "a copied slot is not written by a row of the same step"
    _, prefill_tokens, _ = step.counts()
    chunks = [r for r in step.rows if not r.decode]
    for r in step.rows:  # overlap: a decode row's token is given, or gathered from the step in flight by its slot
        assert (r.prev is None) == (len(r.token_ids) > 1 or not sched.overlap or r.token_ids != [0]) or r.prev is not None
    budget = sched.step_budget  # the policy's, or under ridge between the card's ridge and the probed ceiling
    assert budget == policy.prefill_budget or (policy.prefill_budget_ridge
                                               and policy.ridge_tokens <= budget <= MAX_PREFILL_BUDGET)
    if policy.chunk_prefill:
        assert prefill_tokens <= budget
    else:
        for r in chunks:  # the whole prompt after its prefix hits, in one row (a resumed feed is chunked regardless)
            s = sched.admitted[r.seq]
            assert s.outputs or (r.start == s.hits and r.start + len(r.token_ids) == s.target)
        assert prefill_tokens <= budget or len(chunks) == 1
    for g in step.groups:  # attention=cascade: a group's prefix is whole and computed, before every member's queries
        assert g.node is None or sched.node_state[g.node] == COMPUTED
        for i in g.rows:
            r = step.rows[i]
            assert r.start >= g.shared and r.block_ids[:len(g.blocks)] == g.blocks
            assert all(sched.node_state[nid] == COMPUTED for nid in sched._path(r.seq)[:len(g.blocks)])
    assert sorted(i for g in step.groups for i in g.rows) == (list(range(len(step.rows))) if step.groups else [])
    if sched.slots is not None:  # pool=contiguous: one region per live sequence, never the pad slot, cooling apart
        live = {s.slot for s in sched.admitted.values()}
        assert 0 not in live and len(live) == len(sched.admitted) and all(r.slot == sched.admitted[r.seq].slot for r in step.rows)
        assert not (live & set(sched.slots)) and not (live & set(sched.cooling)) and (sched.overlap or not sched.cooling)
        assert len(live) + len(sched.slots) + len(sched.cooling) == sched.limits.max_seqs
        assert set(step.slot_resets) <= live
    else:
        assert all(r.slot == 0 for r in step.rows) and not step.slot_resets
    chunk_of = {r.seq: len(r.token_ids) for r in chunks}  # under overlap a planned chunk already moved computed
    first = next((s for s in sched.admitted.values()
                  if s.computed - (chunk_of.get(s.idx, 0) if sched.overlap else 0) < s.target), None)
    if first is not None and policy.memory != "preempt":  # a resumed owner can re-enter behind its own waiters
        assert any(r.seq == first.idx for r in chunks), "the earliest admitted unfinished prefill never waits"


def drive(job, policy, num_blocks, seed=0, stop_at=None, max_seqs=MAX_SEQS, kv=None, retained=None, overlap=False,
          attention="fa", pool="static"):
    """stop_at: request index -> the output count at which the fake model emits STOP (default stops_after). kv and
    retained: an allocator and a cache=radix kept-block map carried over from an earlier job, as the engine does.
    The block size is the job's trie's, as the executor's limits would publish. overlap: the engine's order, a step
    planned before the one before it is committed, its gathered tokens checked against the step in flight."""
    B = job.prefix.block_size
    kv = kv or BlockAllocator(num_blocks, seed=seed)
    sched = Scheduler(job, policy, kv, stop_ids=[STOP], retained=retained,
                      limits=dataclasses.replace(limits(num_blocks, block_size=B, overlap=overlap, attention=attention,
                                                        pool=pool), max_seqs=max_seqs))
    stop_at = stops_after(job) if stop_at is None else stop_at
    covered = {i: {} for i in range(len(job.requests))}  # admission -> positions each sequence fed, in order
    copied = {i: set() for i in range(len(job.requests))}  # sharing=token: positions taken by a tail copy, not fed
    sched.test_copied = copied  # check_results reads it: under preempt a resumed row may never feed a copied position
    # (block, slot) -> who held the block when it was written: a sequence (and which admission of it, since a
    # preempted one comes back) or a trie node (and its owner); a slot whose block changed hands no longer holds KV
    written, admissions = {}, {}
    acquired = {}  # (sequence, block) -> the admission in which it first held the block: a parked block keeps its KV
    results, steps, decoding, prefill_total = {}, [], set(), 0
    pending = None  # overlap: the step in flight, (step, its sampled ids, its sampling rows' sequences by slot)

    def holders():
        h = {b: ("seq", s.idx, acquired.setdefault((s.idx, b), admissions[s.idx]))
             for s in sched.admitted.values() for b in s.private_blocks}
        h.update({b: ("seq", idx, acquired[(idx, b)]) for idx, (blocks, _, _) in sched.parked.items() for b in blocks})
        h.update({b: ("node", nid, sched.node_owner[nid]) for nid, (b, st) in enumerate(zip(sched.node_block, sched.node_state))
                  if st != FREE})
        for key in [k for k in acquired if h.get(k[1], (None, None))[1] != k[0]]:
            del acquired[key]  # released: the next holder starts afresh
        return h

    def settle(step, sampled):
        nonlocal written
        live = set(sched.admitted) | set(sched.preempted)  # a row for a sequence an earlier commit ended: discarded
        for _, dst, k in step.copies:  # a block's leading k positions, copied for the sequence that holds dst
            for seq in sched.admitted.values():
                if dst in seq.block_ids:
                    d = seq.block_ids.index(dst)
                    copied[seq.idx].update(range(d * B, d * B + k))
        for r in step.rows:
            if r.seq in live:
                covered[r.seq].setdefault(admissions[r.seq], []).extend(range(r.start, r.start + len(r.token_ids)))
        for s in sched.commit(step, sampled):
            assert s.idx not in results, "finished twice"
            results[s.idx] = s
        held = holders()
        written = {w: who for w, who in written.items() if held.get(w[0]) == who}  # a released block's KV is gone

    while pending is not None or not sched.done:
        admitted_before, preempted_before = set(sched.admitted), set(sched.preempted)
        step = sched.next_step() if not sched.done else None
        if step is None:  # overlap: everything live has its last token in flight
            assert pending is not None
            settle(*pending[:2])
            pending = None
            decoding = {i for i, s in sched.admitted.items() if s.computed >= s.target
                        and len(s.outputs) + s.inflight < s.max_tokens}
            continue
        newly = (set(sched.admitted) - admitted_before) | {i for i in preempted_before if i in sched.admitted}
        for i in newly:
            admissions[i] = admissions.get(i, 0) + 1
        for nid, st in enumerate(sched.node_state):  # cache=radix: a node computed by an earlier job, taken whole
            if st == COMPUTED and sched.node_owner[nid] is None:
                for slot in range(B):
                    written.setdefault((sched.node_block[nid], slot), ("node", nid, None))
        steps.append((step, newly, admitted_before))
        check_blocks(sched, kv)
        check_rows(sched, step, policy)
        held = holders()  # a preemption or an eviction releases blocks inside next_step, and they may be reused at once
        written = {w: who for w, who in written.items() if held.get(w[0]) == who}
        # a preempted owner's pending nodes are rewritten when it resumes: their partial KV no longer counts
        stale = {sched.node_block[nid] for nid, st in enumerate(sched.node_state)
                 if st == PENDING and sched.node_owner[nid] in sched.preempted}
        written = {w: who for w, who in written.items() if w[0] not in stale}
        at_step = sched.occupancy()  # written counts each shared block once, and never exceeds what is held
        assert at_step["kv_written_tokens"] == len(written) <= at_step["kv_reserved_tokens"]
        assert at_step["kv_reserved_tokens"] == B * (kv.capacity - kv.free)
        for src, dst, k in step.copies:  # the executor copies before the forward: the source was written before this step
            assert all((src, slot) in written for slot in range(k)), "a copy reads slots holding KV"
            for slot in range(k):
                written[(dst, slot)] = held[dst]
        for r in step.rows:
            for p in range(r.start, r.start + len(r.token_ids)):
                written[(r.block_ids[p // B], p % B)] = held[r.block_ids[p // B]]
        # every sequence that was decoding after the last step is a decode row in this one, unless preempted
        assert {i for i in decoding if i in sched.admitted} <= {r.seq for r in step.rows if r.decode}
        assert not decoding - set(sched.admitted) - set(sched.preempted)
        prefill_total += step.counts()[1]
        sampled, slots = [], []
        for r in step.rows:
            if r.prev is not None:  # the token is the step in flight's sampled id at that slot, for this sequence
                assert overlap and pending is not None and pending[2][r.prev] == r.seq and r.token_ids == [0]
            elif r.decode:
                assert r.token_ids == [sched.admitted[r.seq].outputs[-1]]
            if r.sample:  # the fake model: STOP at the output count stop_at names, by the row's position
                s = sched.admitted[r.seq]
                n_sample = r.start + len(r.token_ids) - s.prompt_len + 1
                sampled.append(STOP if stop_at.get(r.seq) == n_sample else TOKEN)
                slots.append(r.seq)
        if overlap:
            if pending is not None:
                settle(*pending[:2])
            pending = (step, sampled, slots)
        else:
            settle(step, sampled)
        decoding = {i for i, s in sched.admitted.items() if s.computed >= s.target
                    and len(s.outputs) + s.inflight < s.max_tokens}
    kept = len(sched.retained) if sched.retained is not None else 0  # cache=radix: held for the next job
    assert kv.free + kept == kv.capacity and sched.trie_blocks == 0 == sched.pinned_blocks
    assert all(st == FREE for st in sched.node_state) and all(b is None for b in sched.node_block)
    assert sched.prefix_hit_tokens + prefill_total == sum(it.prompt_len for it in job.requests) + sched.recomputed_tokens
    assert (sched.recomputed_tokens == 0 or sched.preemptions > 0) and not sched.preempted
    return sched, results, covered, steps


def check_results(job, policy, sched, results, covered):
    B = sched.B
    assert sorted(results) == list(range(len(job.requests)))  # every request finished exactly once
    stop_at = stops_after(job)
    for i, s in results.items():
        it = job.requests[i]
        n = len(s.outputs)
        if it.req.ignore_eos:
            assert n == it.req.max_tokens and s.finish_reason == "length"
        elif i in stop_at and stop_at[i] <= it.req.max_tokens:
            assert n == stop_at[i] and s.outputs[-1] == STOP and s.finish_reason == "stop"
        else:
            assert n == it.req.max_tokens and s.finish_reason == "length"
        # fed positions tile [hits, prompt_len + outputs - 1): no gap, no overlap, in order; after a preemption the
        # sequence feeds again from a hit-aligned position: one increasing run per admission, the last one whole
        runs = [covered[i][k] for k in sorted(covered[i])]
        if policy.memory == "preempt":  # a resumed row feeds from where its parked blocks end: every position fed once at
            # least, or copied (sharing=token: a position copied in one admission and parked can come back in the next)
            fed = set().union(*runs) | getattr(sched, "test_copied", {}).get(i, set())
            assert fed >= set(range(s.hits, it.prompt_len + n - 1)), (i, runs)
        else:
            assert runs[-1] == list(range(s.hits, it.prompt_len + n - 1)), (i, runs)
        assert all(r == list(range(r[0], r[0] + len(r))) and (r[0] % B == 0 or policy.sharing == "token" or policy.memory == "preempt")
                   for r in runs)
        assert len(runs) == 1 or policy.memory == "preempt"
        assert (s.hits % B == 0 or policy.sharing == "token") and s.hits < it.prompt_len
        assert reservation_blocks(it.prompt_len, it.req.max_tokens, B) * B >= it.prompt_len + n - 1
    assert sched.admission_log == list(policy.admission_order)
    # the request trace's hits are each request's share of the run's: the same positions counted once each side
    assert sum(s.hit_share for s in results.values()) == sched.prefix_hit_tokens


CASES = [(order, admission, chunk, memory)
         for order, admission, chunk in itertools.product(["input", "max_tokens_desc", "decode_ratio_desc"],
                                                          ["continuous", "groups"], [True, False])
         for memory in (("reserve", "grow", "preempt") if admission == "continuous" else ("reserve",))]


@pytest.mark.parametrize("order,admission,chunk,memory", CASES)
def test_invariants(order, admission, chunk, memory):
    job = workload()
    policy = decide(job, PolicyConfig(order=order, admission=admission, chunk_prefill=chunk, prefill_budget=BUDGET,
                                      max_batch_tokens=80, max_batch_size=4, sharing="none", memory=memory))
    sched, results, covered, steps = drive(job, policy, num_blocks=12)
    check_results(job, policy, sched, results, covered)
    assert sched.prefix_hit_tokens == 0 and all(not s.path for s in results.values())
    if admission == "groups":
        admitted_sets = [newly for _, newly, before in steps if newly]
        assert all(not before for _, newly, before in steps if newly)  # a group waits for the previous one
        assert [sorted(g) for g in admitted_sets] == [sorted(g.members) for g in policy.groups]


SHARING_CASES = [(order, admission, chunk, sharing, budget, memory)
                 for order, admission, chunk, sharing, budget
                 in itertools.product(["input", "decode_ratio_desc", "prefix_dfs"], ["continuous", "groups"],
                                      [True, False], ["none", "block", "token"], [BUDGET, 1, "ridge"])
                 for memory in (("reserve", "grow", "preempt") if admission == "continuous" else ("reserve",))]


@pytest.mark.parametrize("order,admission,chunk,sharing,budget,memory", SHARING_CASES)
def test_sharing_invariants(order, admission, chunk, sharing, budget, memory):
    job = shared_workload()
    policy = decide(job, PolicyConfig(order=order, admission=admission, chunk_prefill=chunk, prefill_budget=budget,
                                      sharing=sharing, max_batch_tokens=400, max_batch_size=5, memory=memory), RIDGE)
    assert policy.prefill_budget_ridge is (budget == "ridge")
    sched, results, covered, _ = drive(job, policy, num_blocks=32)
    check_results(job, policy, sched, results, covered)
    dup = len(job.requests) - 1
    if sharing != "none":
        # each shared prefix is computed once; the duplicate shares every block but its last with request 0
        assert sched.prefix_hit_tokens >= 2 * BLOCK_SIZE * (len(job.requests) - 3)
        assert job.prefix.paths[dup] == job.prefix.paths[0]  # at most one of the two computed those blocks
        assert results[dup].hits + results[0].hits >= BLOCK_SIZE * len(job.prefix.paths[0])
    else:
        assert sched.prefix_hit_tokens == 0 and sched.prefix_wait_steps == 0
        assert all(not s.path and s.block_ids == s.private_blocks for s in results.values())


def test_children_wait_for_pending_nodes_and_hit_computed_ones():
    job = shared_workload()
    # budget 40: the owner's whole prompt fits one step with budget to spare, and its siblings read the nodes it
    # completes in that same step (no wait), since every row's KV is written before the step's attention
    policy = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=40, sharing="block"))
    sched, results, _, steps = drive(job, policy, num_blocks=64)
    assert sched.prefix_wait_steps == 0  # the budget, not a pending node, is what the siblings wait for
    # the first admitted user of a shared node owns it and has no hits; every later user hits every shared node
    pos = {i: k for k, i in enumerate(policy.admission_order)}
    owners = {min(n.users, key=pos.get) for n in job.prefix.nodes if len(n.users) > 1}
    for i, s in results.items():
        shared = sum(BLOCK_SIZE for nid in job.prefix.paths[i] if len(job.prefix.nodes[nid].users) > 1)
        assert s.hits == (0 if i in owners else shared), i


def test_waits_are_counted_only_while_budget_remains():
    """With the budget spent by the owner's chunk, the siblings behind it are not waiting for a node, they
    are waiting for budget like everyone else; counting them would depend on where the order puts them."""
    job = shared_workload()
    saturated = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=4, sharing="block"))
    sched, _, _, steps = drive(job, saturated, num_blocks=64)
    assert sched.prefix_wait_steps == 0
    assert all(step.counts()[1] == 4 for step, _, _ in steps if any(not r.decode for r in step.rows) and step is not steps[-1][0])
    spare = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=40, sharing="block"))
    sched, _, _, _ = drive(job, spare, num_blocks=64)
    assert sched.prefix_wait_steps == 0  # a node the owner completes this step is read by its siblings this step
    # a node still pending after the owner's chunk (the chunk ends inside it) is a wait, when budget is left for
    # the siblings: the owner's second block is half written by a 24-token chunk on step 0 with budget 100
    job2 = analyze([Request(id="o", prompt=" ".join(f"x{j}" for j in range(40)), max_tokens=2),
                    Request(id="s", prompt=" ".join(f"x{j}" for j in range(35)) + " y", max_tokens=2)], StubTokenizer())
    half = decide(job2, PolicyConfig(order="input", prefill_budget=100, sharing="block", chunk_prefill=True))
    sched2 = Scheduler(job2, half, BlockAllocator(16), [STOP])
    sched2.budget = sched2.step_budget = 24  # the step's budget, below the owner's prompt: its chunk ends mid-node
    step = sched2.next_step()
    assert [r.seq for r in step.rows] == [0] and sched2.prefix_wait_steps == 0  # budget spent: not a counted wait
    sched2.budget = sched2.step_budget = 100
    sched2.commit(step, [])
    step = sched2.next_step()  # the owner finishes its prompt; the sibling reads both nodes in the same step
    assert sorted(r.seq for r in step.rows) == [0, 1] and sched2.admitted[1].hits == 2 * BLOCK_SIZE


def test_a_second_scheduler_on_the_same_job_starts_clean():
    job = shared_workload()
    policy = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=BUDGET, sharing="block"))
    drive(job, policy, num_blocks=32)
    fresh = Scheduler(job, policy, BlockAllocator(32, seed=1), [STOP])
    assert fresh.node_remaining == [len(n.users) for n in job.prefix.nodes]
    assert all(st == FREE for st in fresh.node_state) and fresh.trie_blocks == 0 and fresh.prefix_hit_tokens == 0
    assert not hasattr(job.prefix.nodes[0], "state")


def interleaved_job():
    """Two prefix groups of three classification requests (48 shared tokens = three full blocks, then two
    unique ones), interleaved in input order: X0 Y0 X1 Y1 X2 Y2."""
    reqs = []
    for i in range(3):
        for g in "XY":
            words = [f"{g}_{j}" for j in range(48)] + [f"{g}{i}_a", f"{g}{i}_b"]
            reqs.append(Request(id=f"{g}{i}", prompt=" ".join(words), max_tokens=1))
    return analyze(reqs, StubTokenizer())


def test_pinned_prefixes_under_an_interleaving_order_are_refused_before_the_first_step():
    job = interleaved_job()
    # capacity 6: X0 takes 3 shared + 1 private blocks; when it finishes, its 3 stay for X1 and X2 and Y0 cannot fit.
    # Refused when the scheduler is built, so no request has run
    policy = decide(job, PolicyConfig(order="input", prefill_budget=BUDGET, sharing="block"))
    with pytest.raises(ValueError, match="request Y0 needs 4 KV blocks while 3 are held for prefixes shared with "
                                         "requests later in the order; the pool has 6"):
        Scheduler(job, policy, BlockAllocator(7), [STOP])
    assert len(drive(job, policy, num_blocks=8)[1]) == 6  # one block more: Y0 fits beside X's held prefix
    plain = decide(job, PolicyConfig(order="input", prefill_budget=BUDGET, sharing="none"))
    assert len(drive(job, plain, num_blocks=7)[1]) == 6  # without sharing nothing is held
    # the same pool runs to completion when a group's requests are adjacent (budget 64: X0's 50 tokens fit one
    # step with budget left, so X1 and X2 wait on its pending nodes and are counted)
    policy = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=64, sharing="block"))
    assert list(policy.admission_order) == [0, 2, 4, 1, 3, 5]
    sched, results, _, _ = drive(job, policy, num_blocks=7)
    assert len(results) == 6 and sched.prefix_hit_tokens == 4 * 48
    assert sched.head_blocked_steps > 0 and sched.prefix_wait_steps == 0  # X1 and X2 read X0's nodes in its step


def test_fixed_groups_are_checked_exactly_with_shared_nodes():
    job = interleaved_job()
    # groups [X0 X1 X2] [Y0 Y1 Y2]: a group needs 3 shared + 3 private = 6 blocks and holds nothing across the boundary
    policy = decide(job, PolicyConfig(order="prefix_dfs", admission="groups", prefill_budget=BUDGET, sharing="block",
                                      max_batch_tokens=1000, max_batch_size=3))
    assert [len(g.members) for g in policy.groups] == [3, 3]
    with pytest.raises(ValueError, match="fixed group 0 .* needs 6 KV blocks while 0 are held"):
        Scheduler(job, policy, BlockAllocator(6), [STOP])
    sched, results, _, _ = drive(job, policy, num_blocks=7)
    assert len(results) == 6 and sched.prefix_hit_tokens == 4 * 48
    # groups of one in input order X0 Y0 X1 ...: after X0, its 3 blocks stay held for X1 and X2, so Y0 needs 4 + 3.
    # Refused before step 1 with 6 blocks (without sharing the same job completes), runs with 7
    policy = decide(job, PolicyConfig(order="input", admission="groups", prefill_budget=BUDGET, sharing="block",
                                      max_batch_tokens=1000, max_batch_size=1))
    with pytest.raises(ValueError, match="fixed group 1 .* needs 4 KV blocks while 3 are held"):
        Scheduler(job, policy, BlockAllocator(7), [STOP])
    plain = decide(job, PolicyConfig(order="input", admission="groups", prefill_budget=BUDGET, sharing="none",
                                     max_batch_tokens=1000, max_batch_size=1))
    assert len(drive(job, plain, num_blocks=7)[1]) == 6
    sched, results, _, _ = drive(job, policy, num_blocks=8)
    assert len(results) == 6 and sched.prefix_hit_tokens == 4 * 48



def random_shared_job(rng):
    """Up to 14 requests over up to 4 prefixes (0 to 50 shared words, so 0 to 3 full blocks), in input order."""
    groups, reqs = rng.randint(1, 4), []
    for i in range(rng.randint(2, 14)):
        words = [f"p{i % groups}_{j}" for j in range(rng.choice([0, 10, 17, 33, 50]))]
        words += [f"u{i}_{j}" for j in range(rng.randint(1, 20))]
        reqs.append(Request(id=f"r{i}", prompt=" ".join(words), max_tokens=rng.randint(1, 20),
                            ignore_eos=rng.random() < 0.3))
    return analyze(reqs, StubTokenizer())


def test_the_continuous_preflight_refuses_exactly_the_jobs_that_would_stall(monkeypatch):
    """Over random orders and pools: the scheduler refuses a job up front if and only if, run without that check,
    it would reach a head it cannot admit with nothing running. Never a job that would have finished."""
    rng, refused_n, ran_n = random.Random(0), 0, 0
    for trial in range(200):
        job = random_shared_job(rng)
        base = decide(job, PolicyConfig(order="input", prefill_budget=rng.choice([8, 16, 64]), sharing="block",
                                        chunk_prefill=rng.random() < 0.7))
        order = list(base.admission_order)
        rng.shuffle(order)
        policy, num_blocks = dataclasses.replace(base, admission_order=tuple(order)), rng.randint(3, 25)
        try:
            Scheduler(job, policy, BlockAllocator(num_blocks), [STOP])
            refused = False
        except ValueError as e:
            if "requests later in the order" not in str(e):
                continue  # one request alone over the pool: refused before either check applies
            refused = True
        with monkeypatch.context() as m:
            m.setattr(Scheduler, "_overflow", lambda self, groups: None)
            try:
                drive(job, policy, num_blocks, seed=trial)
                stalled = False
            except MemoryError as e:
                assert "nothing is running" in str(e)
                stalled = True
        assert refused == stalled, (trial, order, num_blocks)
        refused_n, ran_n = refused_n + refused, ran_n + (not refused)
    assert refused_n >= 5 and ran_n >= 100, (refused_n, ran_n)  # both outcomes occur: 7 and 171 at seed 0

def test_scheduler_reports_occupancy_counters_and_chain_start():
    """What the batchinfer engine records comes from the scheduler's own reports, not from reading its internals."""
    job = shared_workload()
    policy = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=BUDGET, sharing="block"))
    sched, _, _, steps = drive(job, policy, num_blocks=32)
    trace = {"decode_rows", "prefill_tokens", "prefill_rows", "step_ms", "gpu_ms", "sched_ms", "end_ms", "graph_rows",
             "graph_tokens"}  # mapped_groups comes from the allocator through occupancy()
    assert set(sched.occupancy()) | trace == set(STEP_FIELDS)  # the engine fills the rest from the step and clocks
    assert sched.occupancy()["free_blocks"] == sched.kv.capacity and sched.commits == len(steps)
    assert sched.counters() == {"head_blocked_steps": sched.head_blocked_steps, "prefix_hit_tokens":
                                sched.prefix_hit_tokens, "prefix_wait_steps": sched.prefix_wait_steps,
                                "preemptions": 0, "recomputed_tokens": 0, "retained_hits": 0, "retained_blocks": 0,
                                "discarded_rows": 0, "parked_tokens": 0}
    # the step in which the last request with the job's longest max_tokens sampled its first token
    longest = max(it.req.max_tokens for it in job.requests)
    first = [k for k, (step, _, _) in enumerate(steps)
             if any(r.sample and r.start + len(r.token_ids) == job.requests[r.seq].prompt_len
                    and job.requests[r.seq].req.max_tokens == longest for r in step.rows)]
    assert sched.chain_start_step == max(first)


@pytest.mark.parametrize("config", [
    PolicyConfig(order="prefix_dfs", prefill_budget=BUDGET, sharing="block"),
    PolicyConfig(order="input", prefill_budget=BUDGET, sharing="none"),
    PolicyConfig(order="max_tokens_desc", admission="groups", max_batch_tokens=64, prefill_budget=BUDGET,
                 sharing="none"),
], ids=["sharing", "private", "groups"])
def test_finished_sequences_say_in_which_step_each_event_happened(config):
    """The request trace's *_step fields, recounted from the steps the scheduler built: KV reserved, first prefill
    chunk, first token, finish. On a pool this size admission runs far ahead of prefill."""
    job = shared_workload()
    sched, results, _, steps = drive(job, decide(job, config), num_blocks=64)
    rows = {i: [(k, r) for k, (step, _, _) in enumerate(steps) for r in step.rows if r.seq == i] for i in results}
    for i, s in results.items():
        assert s.admitted_step == next(k for k, (_, newly, _) in enumerate(steps) if i in newly)
        assert s.prefill_start_step == next(k for k, r in rows[i] if not r.decode)
        assert s.first_token_step == next(k for k, r in rows[i] if r.sample)
        assert s.finished_step == rows[i][-1][0]
        assert s.admitted_step <= s.prefill_start_step <= s.first_token_step <= s.finished_step < len(steps)
    if config.admission == "continuous":  # why charts plot prefill start, never admission
        assert any(s.admitted_step < s.prefill_start_step for s in results.values())


def test_chunking_splits_prompts_and_fills_the_budget():
    job = workload()
    policy = decide(job, PolicyConfig(order="input", chunk_prefill=True, prefill_budget=BUDGET))
    _, _, _, steps = drive(job, policy, num_blocks=64)
    chunked = [r for step, _, _ in steps for r in step.rows if not r.decode and r.start > 0]
    assert chunked, "some prompt must span several steps"
    full = [step.counts()[1] for step, _, _ in steps[:-1] if step.counts()[1]]
    assert full.count(BUDGET) >= len(full) // 2  # while prompts remain, steps are filled to the budget


def test_continuous_admission_waits_for_blocks_in_order():
    job = workload()
    policy = decide(job, PolicyConfig(order="input", prefill_budget=BUDGET))
    sched, _, _, _ = drive(job, policy, num_blocks=6)
    assert sched.head_blocked_steps > 0 and sched.admission_log == list(range(len(job.requests)))


def test_seeded_allocator_scatters_blocks_and_never_hands_out_the_pad_block():
    kv = BlockAllocator(64, seed=1)
    ids = kv.reserve(10)
    assert PAD_BLOCK not in ids and ids != sorted(ids)
    assert BlockAllocator(64).reserve(3) == [1, 2, 3]
    with pytest.raises(MemoryError):
        kv.reserve(54)


def _job(prompt_words, max_tokens):
    return analyze([Request(id="big", prompt=" ".join(f"w{j}" for j in range(prompt_words)), max_tokens=max_tokens)],
                   StubTokenizer())


@pytest.mark.parametrize("sharing", [False, True])
def test_refuses_a_reservation_larger_than_the_pool(sharing):
    job = _job(40, 10)
    with pytest.raises(ValueError, match="reserves 4 KV blocks; the pool has 3"):
        Scheduler(job, decide(job, PolicyConfig(sharing=("block" if sharing else "none"))), BlockAllocator(4), [STOP])


def test_refuses_a_fixed_group_larger_than_the_pool():
    job = workload(8)
    policy = decide(job, PolicyConfig(admission="groups", max_batch_tokens=10_000, max_batch_size=8))
    with pytest.raises(ValueError, match="fixed group 0"):
        Scheduler(job, policy, BlockAllocator(5), [STOP])


def test_refuses_an_unchunkable_prompt_over_one_step():
    job = _job(MAX_PREFILL_BUDGET + 1, 1)
    with pytest.raises(ValueError, match="must fit one step"):
        Scheduler(job, decide(job, PolicyConfig(chunk_prefill=False, prefill_budget=512)), BlockAllocator(2000), [STOP])


def test_refuses_a_request_past_the_models_positions():
    job = _job(40, 10)  # feeds positions 0 .. 48
    policy = decide(job, PolicyConfig())
    Scheduler(job, policy, BlockAllocator(64), [STOP], limits(64, max_positions=49))
    with pytest.raises(ValueError, match="request big needs 49 positions .* the model has 48"):
        Scheduler(job, policy, BlockAllocator(64), [STOP], limits(64, max_positions=48))


def long_cap_then_classification(n=50):
    """One generation capped at 8,192 tokens (64 prompt words) that stops at its first token, then n 512-word
    classification prompts: 25,664 prompt tokens behind an 8,191-step chain that never happens."""
    reqs = [Request(id="long", prompt=" ".join(f"l{j}" for j in range(64)), max_tokens=8192)]
    reqs += [Request(id=f"c{i}", prompt=" ".join(f"c{i}_{j}" for j in range(512)), max_tokens=1) for i in range(n)]
    return analyze(reqs, StubTokenizer())


def test_ridge_budget_follows_its_formula():
    """ridge 256 with mixed graphs to 2,048 tokens: a step with decode rows, few or many, carries the measured ridge
    (what the step absorbs for free on top of its decode rows); a 200k-token queue over a 512-step chain needs more
    than the ridge and gets it; a step with no decode rows gets the widest step; the largest mixed graph caps a step
    with decode rows, the widest step caps one without mixed graphs."""
    from batchinfer.graphs import MIXED_TOKENS
    from batchinfer.scheduler import ridge_budget
    assert ridge_budget(256, 200, 100, 10, MIXED_TOKENS, MAX_PREFILL_BUDGET) == 256
    assert ridge_budget(256, 50, 100, 10, MIXED_TOKENS, MAX_PREFILL_BUDGET) == 256
    assert ridge_budget(256, 50, 200_000, 512, MIXED_TOKENS, MAX_PREFILL_BUDGET) == 391
    assert ridge_budget(256, 0, 200_000, 512, MIXED_TOKENS, MAX_PREFILL_BUDGET) == MAX_PREFILL_BUDGET
    assert ridge_budget(256, 50, 200_000, 1, MIXED_TOKENS, MAX_PREFILL_BUDGET) == MIXED_TOKENS[-1]
    assert ridge_budget(256, 50, 200_000, 1, (), MAX_PREFILL_BUDGET) == MAX_PREFILL_BUDGET


def test_ridge_opens_wide_and_fills_decode_steps_by_the_formula_recomputed_from_scratch():
    """Under ridge the opening step, with no decode rows, prefills the widest step; every later step's budget is the
    formula recomputed here from the queue, the admitted prompts' positions left and the decode rows."""
    from batchinfer.graphs import MIXED_TOKENS
    from batchinfer.scheduler import ridge_budget
    reqs = [Request(id="long", prompt=" ".join(f"l{j}" for j in range(64)), max_tokens=8192)]
    reqs += [Request(id=f"g{i}", prompt=" ".join(f"g{i}_{j}" for j in range(30)), max_tokens=300, ignore_eos=True)
             for i in range(40)]
    reqs += [Request(id=f"c{i}", prompt=" ".join(f"c{i}_{j}" for j in range(500)), max_tokens=1) for i in range(300)]
    job = analyze(reqs, StubTokenizer())
    hw = Hardware(gpu_name=None, total_gb=None, max_batch_tokens_fit=None, kv_pool_tokens=None, ridge_tokens=256)
    policy = decide(job, PolicyConfig(order="decode_ratio_desc", prefill_budget="ridge", sharing="none"), hw)
    assert policy.prefill_budget_ridge and policy.ridge_tokens == 256 and policy.prefill_budget == MAX_PREFILL_BUDGET
    sched = Scheduler(job, policy, BlockAllocator(20_000), [STOP], limits(20_000, graph_tokens=MIXED_TOKENS, ridge_tokens=256))
    budgets, filled = [], False
    while not sched.done:
        live_before = list(sched.admitted.values())
        step = sched.next_step()
        live = list(sched.admitted.values())
        queued = [job.requests[i] for i in sched.queue]
        remaining = sum(it.prompt_len for it in queued) + sum(max(0, s.prompt_len - s.computed) for s in live)
        chain = max([it.req.max_tokens - 1 for it in queued] + [s.max_tokens - max(len(s.outputs), 1) for s in live])
        decoding = sum(1 for s in live if s.computed >= s.target and s.computed > s.prompt_len - 1 and len(s.outputs) > 0) \
            or sum(1 for r in step.rows if r.decode)
        want = ridge_budget(256, sum(1 for r in step.rows if r.decode) or decoding, remaining, chain, MIXED_TOKENS, MAX_PREFILL_BUDGET)
        assert sched.step_budget == want or sum(1 for r in step.rows if r.decode) == 0 and sched.step_budget == MAX_PREFILL_BUDGET
        budgets.append(sched.step_budget)
        filled |= 0 < step.counts()[0] and 0 < step.counts()[1] <= 2048
        sched.commit(step, [STOP if r.seq == 0 else TOKEN for r in step.rows if r.sample])
    assert budgets[0] == MAX_PREFILL_BUDGET and filled and max(budgets[1:]) <= MAX_PREFILL_BUDGET
    assert decide(job, PolicyConfig(prefill_budget="ridge")).prefill_budget_ridge is False  # no card: the default budget


def test_ridge_takes_the_prefill_at_once_when_nothing_decodes():
    """The long-cap-then-classification job: with ridge the whole prefill goes in the first step (no decode rows yet:
    the widest step), every classification samples there, and the job takes 2 steps, where the auto budget spread it
    over 101."""
    job = long_cap_then_classification()
    hw = Hardware(gpu_name=None, total_gb=None, max_batch_tokens_fit=None, kv_pool_tokens=None, ridge_tokens=256)
    steps = {}
    for budget in ("auto", "ridge"):
        policy = decide(job, PolicyConfig(order="decode_ratio_desc", prefill_budget=budget, sharing="none"), hw)
        _, results, _, trace = drive(job, policy, num_blocks=4000, stop_at={0: 1})
        assert results[0].finish_reason == "stop" and len(results[0].outputs) == 1
        steps[budget] = len(trace)
    assert steps == {"auto": 101, "ridge": 2}


def _grow_job(prompts, max_tokens):
    reqs = [Request(id=f"g{i}", prompt=" ".join(f"g{i}_{j}" for j in range(p)), max_tokens=m, ignore_eos=True)
            for i, (p, m) in enumerate(zip(prompts, max_tokens))]
    return analyze(reqs, StubTokenizer())


def test_grow_admits_a_request_as_soon_as_the_live_set_can_never_exceed_the_pool():
    """Three requests of 8-token prompts on a pool of 7 blocks: A (max_tokens 64, 5 blocks for its life) and B
    (max_tokens 16, 2 blocks) fill it under reserve, so C (max_tokens 16, 2 blocks) waits for B to finish at step
    15. Under grow, A's and B's blocks are counted when they will be held: B is gone before A needs its fourth
    block, so A, B and C never exceed 7 together and C is admitted at step 1, with nothing preempted."""
    job = _grow_job([8, 8, 8], [64, 16, 16])
    walls = {}
    for memory in ("reserve", "grow"):
        policy = decide(job, PolicyConfig(order="input", sharing="none", memory=memory))
        sched, results, covered, _ = drive(job, policy, num_blocks=8, stop_at={})
        check_results(job, policy, sched, results, covered)
        walls[memory] = (results[2].admitted_step, results[2].finished_step, sched.commits)
    assert walls["reserve"][0] == 16 and walls["grow"][0] == 1
    assert walls["grow"][1] < walls["reserve"][1]  # C finishes earlier; the job's wall is A's 64-step chain either way
    assert walls["grow"][2] == walls["reserve"][2] == 64


def test_grow_never_runs_dry_and_never_admits_later_than_reserve():
    """Over the random jobs and pools the preflight test runs: grow finishes every job reserve finishes (its
    timeline keeps the pool from running dry, so _cover never fails) and admits every request no later."""
    rng, compared = random.Random(1), 0
    for trial in range(120):
        job = random_shared_job(rng)
        base = decide(job, PolicyConfig(order=rng.choice(["input", "prefix_dfs"]), prefill_budget=rng.choice([8, 16, 64]),
                                        sharing="block" if rng.random() < 0.7 else "none", chunk_prefill=rng.random() < 0.7))
        num_blocks = rng.randint(3, 25)
        try:
            _, reserve_results, _, _ = drive(job, base, num_blocks, seed=trial)
        except ValueError:
            continue  # refused up front: the same request alone over the pool, under either mode
        grow = dataclasses.replace(base, memory="grow")
        sched, results, covered, _ = drive(job, grow, num_blocks, seed=trial)
        check_results(job, grow, sched, results, covered)
        for i, s in results.items():
            assert s.admitted_step <= reserve_results[i].admitted_step, (trial, i)
        compared += 1
    assert compared >= 80


def test_the_scheduler_reads_its_caps_from_the_limits_it_is_given():
    """A Limits with max_seqs below the module constant caps admission, and the pool must be the allocator's."""
    job = workload(12)
    policy = decide(job, PolicyConfig(order="input", sharing="none", prefill_budget=BUDGET))
    sched, results, covered, steps = drive(job, policy, num_blocks=64, max_seqs=4)
    check_results(job, policy, sched, results, covered)
    assert max(len(before) + len(newly) for _, newly, before in steps) == 4 and sched.head_blocked_steps > 0
    with pytest.raises(AssertionError, match="allocator's pool"):
        Scheduler(job, policy, BlockAllocator(64), [STOP], limits(32))


@pytest.mark.parametrize("sharing", [False, True])
def test_preempt_recovers_with_a_tiny_pool_and_gives_the_same_outputs(sharing):
    """A pool too small for the live set under preempt: the youngest request is preempted when a decode row finds
    no block, re-enters at the head of the queue, feeds its prompt and outputs again, and finishes with the same
    outputs and finish reason as under reserve (the fake model is a function of the output count)."""
    # eight prompts whose tails end a block, so every request needs a second block one step after its first token,
    # together, which the watermark (a step of growth for the live set) cannot cover for all of them at once
    shared = ["s0_%d" % j for j in range(16)] if sharing else []
    reqs = [Request(id=f"q{i}", prompt=" ".join(shared + [f"q{i}_{j}" for j in range(15)]), max_tokens=20,
                    ignore_eos=i % 2 == 0) for i in range(8)]
    job = analyze(reqs, StubTokenizer())
    runs = {}
    for memory in ("reserve", "preempt"):
        policy = decide(job, PolicyConfig(order="input", sharing=("block" if sharing else "none"), prefill_budget=BUDGET, memory=memory))
        sched, results, covered, _ = drive(job, policy, num_blocks=10 if sharing else 9)
        check_results(job, policy, sched, results, covered)
        runs[memory] = (sched, {i: (s.outputs, s.finish_reason) for i, s in results.items()})
    assert runs["preempt"][0].preemptions > 0 and runs["preempt"][0].recomputed_tokens > 0
    assert runs["preempt"][1] == runs["reserve"][1]


def test_preempt_evicts_pinned_nodes_deepest_first_before_preempting():
    """With sharing, a node held for requests not yet admitted (pinned) is released before any request is
    preempted, deepest first and never above a live child; its next user allocates it again."""
    job = shared_workload()
    # a shuffled order pins nodes of the first group while the others run; reserve refuses it, preempt runs it
    policy = decide(job, PolicyConfig(order="input", sharing="block", prefill_budget=BUDGET, memory="preempt"))
    order = list(policy.admission_order)
    order = order[::2] + order[1::2]
    policy = dataclasses.replace(policy, admission_order=tuple(order))
    sched, results, covered, steps = drive(job, policy, num_blocks=9)
    check_results(job, policy, sched, results, covered)
    assert results and sched.pinned_blocks == 0 and sched.trie_blocks == 0


def test_the_grow_preflight_refuses_exactly_the_jobs_that_would_stall(monkeypatch):
    """The preflight validates every mode against a request's whole life, so under grow a lone oversize request is
    refused with the reservation message and a stalling order with the shared-prefix one, exactly as under reserve."""
    rng, agreed = random.Random(2), 0
    for trial in range(150):
        job = random_shared_job(rng)
        base = decide(job, PolicyConfig(order="input", prefill_budget=rng.choice([8, 16, 64]), sharing="block",
                                        chunk_prefill=rng.random() < 0.7))
        order = list(base.admission_order)
        rng.shuffle(order)
        num_blocks = rng.randint(3, 25)
        messages = {}
        for memory in ("reserve", "grow"):
            policy = dataclasses.replace(base, admission_order=tuple(order), memory=memory)
            try:
                Scheduler(job, policy, BlockAllocator(num_blocks), [STOP])
                messages[memory] = None
            except ValueError as e:
                messages[memory] = str(e)
        assert messages["grow"] == messages["reserve"], (trial, messages)
        agreed += 1
    assert agreed == 150


def test_the_timeline_caches_its_peak_between_changes():
    from batchinfer.scheduler import Timeline
    t = Timeline(8, BLOCK_SIZE)
    s = Sequence(idx=0, prompt_ids=[1] * 20, max_tokens=6, ignore_eos=True, block_ids=[], private_blocks=[],
                 path=[], group=None, admitted_at=0.0, feed=[1] * 20, target=20)
    t.flat += 3
    assert t.peak() == 3
    t.stair(s, +1, age=0)  # positions 21.. : 2 blocks, then a third block at position 32 (k = 11 > end): never
    assert t.peak() == 3 + 2 and t._peak == 2
    t.flat -= 3
    assert t.peak() == 2  # flat is added on top of the cached staircases' peak
    t.advance()
    assert t._peak is None and t.peak() == 2


def test_token_sharing_copies_the_leading_tokens_of_a_block_from_a_computed_sibling():
    """Two prompts share 20 tokens: block 0 as a trie node, and 4 tokens of their second block. The 40-token prompt's
    second block is a trie node; once it is computed, the 22-token prompt copies those 4 tokens into its private tail
    instead of recomputing them (block mode recomputes them: 16 hits against 20). A third prompt sharing the 22-token
    prompt's whole tail but its last token copies from that tail once written, a request-to-request copy."""
    words = [f"t{j}" for j in range(40)]
    reqs = [Request(id="long", prompt=" ".join(words), max_tokens=3, ignore_eos=True),
            Request(id="short", prompt=" ".join(words[:20] + ["u1", "u2"]), max_tokens=3, ignore_eos=True),
            Request(id="twin", prompt=" ".join(words[:20] + ["u1", "u3"]), max_tokens=3, ignore_eos=True),
            Request(id="later", prompt=" ".join(words[:20] + ["u1", "u4"]), max_tokens=3, ignore_eos=True)]
    job = analyze(reqs, StubTokenizer())
    hits = {}
    for sharing in ("block", "token"):
        policy = decide(job, PolicyConfig(order="input", prefill_budget=4, sharing=sharing))
        sched, results, covered, steps = drive(job, policy, num_blocks=32, stop_at={})
        check_results(job, policy, sched, results, covered)
        hits[sharing] = {r.id: results[i].hits for i, r in enumerate(reqs)}
        copies = [c for step, _, _ in steps for c in step.copies]
        assert (len(copies) == 3) == (sharing == "token")
    assert hits["block"] == {"long": 0, "short": 16, "twin": 16, "later": 16}
    # short and twin copy 4 from long's second node; twin reaches its block while short's tail is still being
    # written, so only the node counts; later finds short's tail written and copies 5 of it (t16..t19 and u1)
    assert hits["token"] == {"long": 0, "short": 20, "twin": 20, "later": 21}


def test_preempt_gives_reserves_outputs_on_random_jobs_and_tiny_pools():
    """Over random jobs, shuffled orders, budgets and pools from the smallest the preflight accepts: preempt finishes
    every job with reserve's outputs and finish reasons, preempting on a good share of them."""
    rng, compared, preempted = random.Random(3), 0, 0
    for trial in range(200):
        job = random_shared_job(rng)
        base = decide(job, PolicyConfig(order="input", prefill_budget=rng.choice([1, 4, 8, 16, 64]),
                                        sharing=rng.choice(["none", "block", "token"]), chunk_prefill=rng.random() < 0.7))
        order = list(base.admission_order)
        rng.shuffle(order)
        num_blocks = rng.randint(3, 25)
        reserve = dataclasses.replace(base, admission_order=tuple(order), memory="reserve")
        try:
            _, want, _, _ = drive(job, reserve, num_blocks, seed=trial)
        except ValueError:
            continue  # refused up front under reserve; preempt may run it (pinned nodes are evictable) or refuse too
        policy = dataclasses.replace(base, admission_order=tuple(order), memory="preempt")
        sched, results, covered, _ = drive(job, policy, num_blocks, seed=trial)
        check_results(job, policy, sched, results, covered)
        assert {i: (s.outputs, s.finish_reason) for i, s in results.items()} == {i: (s.outputs, s.finish_reason) for i, s in want.items()}
        compared += 1
        preempted += sched.preemptions > 0
    assert compared >= 100 and preempted >= 5, (compared, preempted)  # the watermark keeps most jobs from preempting



def test_preempt_with_token_sharing_resumes_past_its_prompt_on_random_jobs():
    """Outputs several blocks long under tight pools: a preempted request feeds its prompt and outputs again, and its
    parked blocks, partly evicted, can leave it at a block boundary past the prompt's trie blocks, where no sibling
    shares anything; token sharing takes no copy there. Preempt finishes with reserve's outputs wherever reserve
    runs the job."""
    rng, ran, preempted = random.Random(11), 0, 0
    for trial in range(200):
        groups, reqs = rng.randint(1, 3), []
        for i in range(rng.randint(3, 12)):
            words = [f"p{i % groups}_{j}" for j in range(rng.choice([10, 17, 33]))] + [f"u{i}_{j}" for j in range(rng.randint(1, 20))]
            reqs.append(Request(id=f"r{i}", prompt=" ".join(words), max_tokens=rng.randint(20, 70), ignore_eos=rng.random() < 0.5))
        job = analyze(reqs, StubTokenizer())
        base = decide(job, PolicyConfig(order="input", prefill_budget=rng.choice([4, 16, 64]), sharing="token",
                                        chunk_prefill=rng.random() < 0.7))
        num_blocks = rng.randint(6, 30)
        policy = dataclasses.replace(base, memory="preempt", watermark_steps=rng.choice([1, 4, 32]))
        try:
            sched, results, covered, _ = drive(job, policy, num_blocks, seed=trial)
        except ValueError:
            continue  # refused up front: a request longer than the pool
        check_results(job, policy, sched, results, covered)
        ran += 1
        preempted += sched.preemptions > 0
        try:
            _, want, _, _ = drive(job, dataclasses.replace(base, memory="reserve"), num_blocks, seed=trial)
        except ValueError:
            continue
        assert {i: (s.outputs, s.finish_reason) for i, s in results.items()} == {i: (s.outputs, s.finish_reason) for i, s in want.items()}
    assert ran >= 100 and preempted >= 20, (ran, preempted)

def test_radix_cache_keeps_shared_blocks_for_the_next_job_and_evicts_the_oldest_first():
    """Three jobs on one allocator with cache=radix. A job's shared prefix blocks are kept when their last user
    finishes, so the next job sharing that prefix admits with the nodes computed: every request hits the whole
    prefix, nothing prefills it. A job that needs the room evicts kept blocks oldest first, so the newest kept
    prefix still serves it."""
    from collections import OrderedDict
    s1, s2 = [f"s{j}" for j in range(32)], [f"t{j}" for j in range(32)]  # two 2-block system prompts
    tok = StubTokenizer()  # one vocabulary across the jobs, as a real tokenizer has

    def job(tag, system, n):
        return analyze([Request(id=f"{tag}{i}", prompt=" ".join(system + [f"{tag}{i}"]), max_tokens=2, ignore_eos=True)
                        for i in range(n)], tok)

    def run(j, kv, retained):
        policy = decide(j, PolicyConfig(order="input", prefill_budget=64, sharing="block", cache="radix"))
        sched, results, covered, _ = drive(j, policy, 12, kv=kv, retained=retained, stop_at={})
        check_results(j, policy, sched, results, covered)
        return sched, results

    kv, retained = BlockAllocator(12, seed=0), OrderedDict()
    sched, results = run(job("a", s1, 3), kv, retained)  # cold: the owner computes the prefix, the others hit it
    assert sched.retained_hits == 0 and len(retained) == 2 and kv.free == 11 - 2
    assert sorted(r.hits for r in results.values()) == [0, 32, 32]
    sched, results = run(job("b", s2, 3), kv, retained)  # another prefix: kept behind the first
    assert sched.retained_hits == 0 and len(retained) == 4 and kv.free == 11 - 4
    sched, results = run(job("c", s1, 3), kv, retained)  # warm: all three hit the kept prefix
    assert sched.retained_hits == 2 and all(r.hits == 32 for r in results.values())
    assert sched.prefix_hit_tokens == 3 * 32 and len(retained) == 4  # re-kept, now the newest
    # 8 requests sharing s2 need 8 private blocks of the 7 free: the oldest kept block (one of s1's) makes way,
    # one block at a time and only as needed; s2's serve the whole job
    sched, results = run(job("d", s2, 8), kv, retained)
    assert sched.retained_hits == 2 and all(r.hits == 32 for r in results.values())
    assert len(retained) == 3 and kv.free == 11 - 3


def test_radix_cache_needs_sharing_and_changes_nothing_within_one_job():
    with pytest.raises(ValueError, match="needs sharing"):
        PolicyConfig(sharing="none", cache="radix").validate()
    job = random_shared_job(random.Random(5))
    walls = {}
    for cache in ("job", "radix"):
        policy = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=16, cache=cache))
        sched, results, covered, steps = drive(job, policy, 24, retained=OrderedDict() if cache == "radix" else None)
        check_results(job, policy, sched, results, covered)
        walls[cache] = (len(steps), {i: (s.outputs, s.hits) for i, s in results.items()})
    assert walls["job"] == walls["radix"]


@pytest.mark.parametrize("block_size", [32, 64])
def test_the_invariants_hold_at_other_block_sizes(block_size):
    """The scheduler reads the block size from its limits and the trie: at 32 and 64 every memory x sharing mode
    keeps the harness's invariants on shared jobs (prefixes of 70 words, so a node exists at 64 too), the outputs
    equal the block-16 run's, and the hit counts are whole blocks of the run's own size."""
    reqs = [Request(id=f"s{i}", prompt=" ".join([f"p{i % 3}_{j}" for j in range(70)] + [f"u{i}_{j}" for j in range(2 + (i * 5) % 13)]),
                    max_tokens=1 + (i * 5) % 9, ignore_eos=i % 4 == 0) for i in range(12)]
    tok = StubTokenizer()
    jobs = {b: analyze(reqs, tok, block_size=b) for b in (16, block_size)}
    assert jobs[block_size].prefix.block_size == block_size
    for memory, sharing in itertools.product(["reserve", "grow", "preempt"], ["none", "block", "token"]):
        outputs = {}
        for b, job in jobs.items():
            policy = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=48, sharing=sharing, memory=memory))
            sched, results, covered, _ = drive(job, policy, num_blocks=64 * 16 // b + 1)  # the same tokens of pool
            check_results(job, policy, sched, results, covered)
            outputs[b] = {i: (s.outputs, s.finish_reason) for i, s in results.items()}
            if sharing == "block":
                assert all(s.hits % b == 0 for s in results.values())
            if sharing != "none" and b == block_size:
                assert sched.prefix_hit_tokens >= b * sum(len(n.users) - 1 for n in job.prefix.nodes)
        assert outputs[16] == outputs[block_size], (memory, sharing)


def test_overlap_plans_a_step_ahead_and_gives_the_synchronous_outputs():
    """The engine's overlap order on random shared jobs over every memory x sharing mode: the outputs and finish
    reasons equal the synchronous run's; rows gather their token from the step in flight (checked by the harness);
    the one decode row planned past an EOS is discarded and counted; a sequence whose last token is in flight gets
    no row; a preempted sequence's in-flight sample still lands."""
    rng, discarded, preempted_inflight = random.Random(11), 0, 0
    for trial in range(60):
        job = random_shared_job(rng)
        memory = rng.choice(["reserve", "grow", "preempt"])
        cfg = PolicyConfig(order=rng.choice(["input", "prefix_dfs", "decode_ratio_desc"]), prefill_budget=rng.choice([4, 16, 64]),
                           sharing=rng.choice(["none", "block", "token"]), memory=memory, chunk_prefill=True)
        policy = decide(job, cfg)
        num_blocks = rng.randint(12, 40) if memory == "preempt" else 64
        try:
            sync_sched, want, _, sync_steps = drive(job, policy, num_blocks, seed=trial)
        except ValueError:
            continue
        sched, results, covered, steps = drive(job, policy, num_blocks, seed=trial, overlap=True)
        check_results(job, policy, sched, results, covered)
        assert {i: (s.outputs, s.finish_reason) for i, s in results.items()} == {i: (s.outputs, s.finish_reason) for i, s in want.items()}
        assert sched.discarded == sum(1 for s in results.values() if s.finish_reason == "stop" and len(s.outputs) < s.max_tokens
                                      and s.outputs[-1] == STOP)  # every EOS finish had one row in flight past it
        # steps: a discarded row's step and an in-flight last token cost one each; a sharer reads a node the step
        # before it wrote one step earlier than under commit-time marks, which can save one
        assert abs(len(steps) - len(sync_steps)) <= 2 + sched.discarded + sched.preemptions
        discarded += sched.discarded
        preempted_inflight += sched.preemptions > 0
    assert discarded > 0 and preempted_inflight > 0


def test_cascade_groups_every_step_behind_shared_computed_prefixes():
    """Under attention=cascade the scheduler hands each step its groups (checked by the harness): on the shared
    workload the decode steps group the sharers behind their prefix, the outputs are unchanged, and under overlap
    a node planned in the step before counts as computed for the grouping."""
    job = shared_workload()
    policy = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=64, sharing="block"))
    plain = drive(job, policy, 64)[1]
    for overlap in (False, True):
        sched, results, covered, steps = drive(job, policy, 64, attention="cascade", overlap=overlap)
        check_results(job, policy, sched, results, covered)
        assert {i: s.outputs for i, s in results.items()} == {i: s.outputs for i, s in plain.items()}
        grouped = [g for step, _, _ in steps for g in step.groups if len(g.rows) > 1]
        assert grouped and all(g.shared >= BLOCK_SIZE for g in grouped)


def test_contiguous_slots_follow_the_sequences_and_cool_for_a_commit():
    """pool=contiguous: an admitted sequence takes a region for its life, the step says which regions new admissions
    took, a freed region cools for one commit before it is handed out again, the pad slot is never handed out, and
    with as many regions as the cap a tiny cap admits one sequence per cooled slot; outputs are unchanged under both
    loop orders and every memory mode."""
    job = shared_workload()
    policy = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=64, sharing="block"))
    plain = drive(job, policy, 64)[1]
    for overlap in (False, True):
        for memory in ("reserve", "grow", "preempt"):
            pol = decide(job, dataclasses.replace(PolicyConfig(order="prefix_dfs", prefill_budget=64, sharing="block",
                                                               memory=memory)))
            sched, results, covered, steps = drive(job, pol, 64, overlap=overlap, pool="contiguous")
            check_results(job, pol, sched, results, covered)
            assert {i: s.outputs for i, s in results.items()} == {i: s.outputs for i, s in plain.items()}
            assert sum(len(step.slot_resets) for step, _, _ in steps) == len(job.requests) + sched.preemptions
            assert sorted(sched.slots + sched.cooling) == list(range(1, MAX_SEQS + 1))
    for overlap in (False, True):  # three regions: a tiny cap, the cooling costing a step at most under overlap
        small, results, covered, _ = drive(job, policy, 64, max_seqs=3, pool="contiguous", overlap=overlap)
        check_results(job, policy, small, results, covered)
        assert {i: s.outputs for i, s in results.items()} == {i: s.outputs for i, s in plain.items()}


def test_overlap_keeps_grow_within_the_pool_and_releases_a_preempted_sequence_that_ends_in_flight():
    """Under overlap a finish is committed one step after its last row, when the next step is already planned: the
    grow timeline holds each staircase one step longer, so tight pools with no EOS never run dry; and a sequence
    preempted with its last token in flight, which ends while preempted, releases its path nodes and pays back the
    recompute it never did, so the pool and the token identity come out whole."""
    rng, preempted_runs, grow_runs = random.Random(21), 0, 0
    for trial in range(120):
        job = random_shared_job(rng)
        base = PolicyConfig(order=rng.choice(["input", "prefix_dfs"]), prefill_budget=rng.choice([4, 16, 64]),
                            sharing=rng.choice(["none", "block"]), chunk_prefill=True)
        for memory, blocks in (("grow", rng.randint(10, 30)), ("preempt", rng.randint(8, 24))):
            policy = decide(job, dataclasses.replace(base, memory=memory))
            try:
                sync = drive(job, policy, blocks, seed=trial)[1]
            except ValueError:
                continue
            sched, results, covered, _ = drive(job, policy, blocks, seed=trial, overlap=True)
            check_results(job, policy, sched, results, covered)  # the pool whole, the token identity, every finish
            assert {i: s.outputs for i, s in results.items()} == {i: s.outputs for i, s in sync.items()}
            preempted_runs += memory == "preempt" and sched.preemptions > 0
            grow_runs += memory == "grow"
    assert preempted_runs > 5 and grow_runs > 20
    # a pinned node evicted while its last user waits preempted with a sample in flight: when that sample ends the
    # sequence, its path release finds the node gone already and leaves it alone (trial 127 of this seed reaches
    # it; 195 is a sequence that ends preempted, whose record keeps the hits of its one admission)
    rng = random.Random(5)
    for trial in range(200):
        job = random_shared_job(rng)
        base = PolicyConfig(order=rng.choice(["input", "prefix_dfs"]), prefill_budget=rng.choice([4, 16, 64]),
                            sharing=rng.choice(["none", "block"]), chunk_prefill=True)
        policy, blocks = decide(job, dataclasses.replace(base, memory="preempt")), rng.randint(6, 24)
        try:
            sync = drive(job, policy, blocks, seed=trial)[1]
        except ValueError:
            continue
        sched, results, covered, _ = drive(job, policy, blocks, seed=trial, overlap=True)
        check_results(job, policy, sched, results, covered)
        assert {i: s.outputs for i, s in results.items()} == {i: s.outputs for i, s in sync.items()}


def test_every_knob_combination_gives_the_same_outputs_on_random_jobs():
    """The scheduler's modes compose: random jobs under random memory, sharing, cache, loop order, attention and
    pool settings, with tiny pools and few regions, all give the outputs and finish reasons of the plain run and
    keep every harness invariant (the pool whole at the end, the token identity, the groups and slots)."""
    rng, compared = random.Random(31), 0
    for trial in range(80):
        job = random_shared_job(rng)
        plain = decide(job, PolicyConfig(order="input", prefill_budget=64, sharing="none"))
        want = {i: (s.outputs, s.finish_reason) for i, s in drive(job, plain, 64)[1].items()}
        memory = rng.choice(["reserve", "grow", "preempt"])
        watermark = rng.choice([1, 32])  # preempt: the lookahead changes admission, never outputs
        sharing = rng.choice(["none", "block", "token"])
        cfg = PolicyConfig(order=rng.choice(["input", "prefix_dfs", "decode_ratio_desc"]), prefill_budget=rng.choice([4, 16, 64]),
                           sharing=sharing, memory=memory, cache=rng.choice(["job", "radix"]) if sharing != "none" else "job",
                           watermark_steps=watermark)
        policy = decide(job, cfg)
        overlap, attention = rng.random() < 0.5, rng.choice(["fa", "cascade"])
        pool = rng.choice(["static", "contiguous"])
        blocks = rng.randint(10, 40) if memory == "preempt" else rng.randint(24, 64)
        max_seqs = rng.choice([MAX_SEQS, 3, 6])
        try:
            drive(job, policy, blocks, seed=trial, max_seqs=max_seqs)  # refused up front (reserve or grow): skip
        except ValueError:
            continue
        sched, results, covered, steps = drive(job, policy, blocks, seed=trial, max_seqs=max_seqs, overlap=overlap,
                                               attention=attention, pool=pool, retained=OrderedDict() if cfg.cache == "radix" else None)
        check_results(job, policy, sched, results, covered)
        assert {i: (s.outputs, s.finish_reason) for i, s in results.items()} == want, (trial, cfg, overlap, attention, pool)
        compared += 1
    assert compared >= 40
