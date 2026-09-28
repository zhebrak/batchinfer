"""scheduler: every knob combination keeps the invariants a paged batchinfer engine depends on, with and without
prefix sharing. Pure Python: a fake executor returns a fixed id (or a stop id after a set number of outputs)."""
import dataclasses
import itertools
import random

import pytest

from batchinfer.analysis import analyze
from batchinfer.kv import BLOCK_SIZE, PAD_BLOCK, BlockAllocator, reservation_blocks
from batchinfer.metrics import STEP_FIELDS
from batchinfer.policy import decide
from batchinfer.scheduler import COMPUTED, FREE, PENDING, Scheduler
from batchinfer.schema import MAX_PREFILL_BUDGET, PolicyConfig, Request
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
    live = private + shared
    assert PAD_BLOCK not in live and len(live) == len(set(live)) and kv.free >= 0
    assert kv.free + len(live) == kv.capacity and sched.trie_blocks == len(shared)
    pinned = sum(1 for nid, st in enumerate(sched.node_state) if st != FREE and sched.node_admitted[nid] == 0)
    assert sched.pinned_blocks == pinned
    for s in sched.admitted.values():
        assert s.block_ids == [sched.node_block[nid] for nid in s.path] + s.private_blocks


def check_rows(sched, step, policy):
    """Rows read only computed nodes or blocks they wrote themselves, write only their own pending nodes or
    private blocks, and never the same slot twice in a step; the budget holds; the earliest admitted
    sequence with prompt left always gets a chunk."""
    written = set()
    for r in step.rows:
        s = sched.admitted[r.seq]
        end = r.start + len(r.token_ids)
        for k, nid in enumerate(s.path):
            lo, hi = k * BLOCK_SIZE, (k + 1) * BLOCK_SIZE
            if lo < r.start:  # read by this row
                assert sched.node_state[nid] == COMPUTED or sched.node_owner[nid] == r.seq
            if lo < end and hi > r.start:  # written by this row
                assert sched.node_owner[nid] == r.seq and sched.node_state[nid] == PENDING
        for p in range(r.start, end):
            slot = (r.block_ids[p // BLOCK_SIZE], p % BLOCK_SIZE)
            assert slot not in written, "two rows write one slot"
            written.add(slot)
    _, prefill_tokens, _ = step.counts()
    chunks = [r for r in step.rows if not r.decode]
    budget = sched.step_budget  # the policy's, or at least it and at most the probed ceiling when adaptive
    assert budget == policy.prefill_budget or (policy.prefill_budget_adaptive
                                               and policy.prefill_budget <= budget <= MAX_PREFILL_BUDGET)
    if policy.chunk_prefill:
        assert prefill_tokens <= budget
    else:
        for r in chunks:  # the whole prompt after its prefix hits, in one row
            s = sched.admitted[r.seq]
            assert r.start == s.hits and r.start + len(r.token_ids) == s.prompt_len
        assert prefill_tokens <= budget or len(chunks) == 1
    first = next((s for s in sched.admitted.values() if s.computed < s.prompt_len), None)
    if first is not None:
        assert any(r.seq == first.idx for r in chunks), "the earliest admitted unfinished prefill never waits"


def drive(job, policy, num_blocks, seed=0, stop_at=None):
    """stop_at: request index -> the output count at which the fake model emits STOP (default stops_after)."""
    kv = BlockAllocator(num_blocks, seed=seed)
    sched = Scheduler(job, policy, kv, stop_ids=[STOP])
    stop_at = stops_after(job) if stop_at is None else stop_at
    covered = {i: [] for i in range(len(job.requests))}  # positions each sequence fed, in order
    written = set()  # (block, slot) pairs holding KV, recounted from the rows themselves
    results, steps, decoding, prefill_total = {}, [], set(), 0
    while not sched.done:
        admitted_before = set(sched.admitted)
        step = sched.next_step()
        newly = set(sched.admitted) - admitted_before
        steps.append((step, newly, admitted_before))
        check_blocks(sched, kv)
        check_rows(sched, step, policy)
        at_step = sched.occupancy()  # written counts each shared block once, and never exceeds what is held
        assert at_step["kv_written_tokens"] == len(written) <= at_step["kv_reserved_tokens"]
        assert at_step["kv_reserved_tokens"] == BLOCK_SIZE * (kv.capacity - kv.free)
        written |= {(r.block_ids[p // BLOCK_SIZE], p % BLOCK_SIZE) for r in step.rows
                    for p in range(r.start, r.start + len(r.token_ids))}
        # every sequence that was decoding after the last step is a decode row in this one
        assert decoding <= {r.seq for r in step.rows if r.decode}
        prefill_total += step.counts()[1]
        sampled = []
        for r in step.rows:
            covered[r.seq] += range(r.start, r.start + len(r.token_ids))
            if r.sample:
                n_out = len(sched.admitted[r.seq].outputs)
                sampled.append(STOP if stop_at.get(r.seq) == n_out + 1 else TOKEN)
        for s in sched.commit(step, sampled):
            assert s.idx not in results, "finished twice"
            results[s.idx] = s
        live = ({b for s in sched.admitted.values() for b in s.private_blocks}
                | {b for b, st in zip(sched.node_block, sched.node_state) if st != FREE})
        written = {w for w in written if w[0] in live}  # a released block's stale KV no longer counts
        decoding = {i for i, s in sched.admitted.items() if s.computed >= s.prompt_len}
    assert kv.free == kv.capacity and sched.trie_blocks == 0 == sched.pinned_blocks
    assert all(st == FREE for st in sched.node_state) and all(b is None for b in sched.node_block)
    assert sched.prefix_hit_tokens == sum(it.prompt_len for it in job.requests) - prefill_total
    return sched, results, covered, steps


def check_results(job, policy, sched, results, covered):
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
        # fed positions tile [hits, prompt_len + outputs - 1): no gap, no overlap, in order
        assert covered[i] == list(range(s.hits, it.prompt_len + n - 1))
        assert s.hits % BLOCK_SIZE == 0 and s.hits < it.prompt_len
        assert reservation_blocks(it.prompt_len, it.req.max_tokens) * BLOCK_SIZE >= it.prompt_len + n - 1
    assert sched.admission_log == list(policy.admission_order)


CASES = list(itertools.product(["input", "max_tokens_desc", "decode_ratio_desc"], ["continuous", "groups"],
                               [True, False]))


@pytest.mark.parametrize("order,admission,chunk", CASES)
def test_invariants(order, admission, chunk):
    job = workload()
    policy = decide(job, PolicyConfig(order=order, admission=admission, chunk_prefill=chunk, prefill_budget=BUDGET,
                                      max_batch_tokens=80, max_batch_size=4, prefix_sharing=False))
    sched, results, covered, steps = drive(job, policy, num_blocks=12)
    check_results(job, policy, sched, results, covered)
    assert sched.prefix_hit_tokens == 0 and all(not s.path for s in results.values())
    if admission == "groups":
        admitted_sets = [newly for _, newly, before in steps if newly]
        assert all(not before for _, newly, before in steps if newly)  # a group waits for the previous one
        assert [sorted(g) for g in admitted_sets] == [sorted(g.members) for g in policy.groups]


SHARING_CASES = list(itertools.product(["input", "decode_ratio_desc", "prefix_dfs"], ["continuous", "groups"],
                                       [True, False], [True, False], [BUDGET, 1, "adaptive"]))


@pytest.mark.parametrize("order,admission,chunk,sharing,budget", SHARING_CASES)
def test_sharing_invariants(order, admission, chunk, sharing, budget):
    job = shared_workload()
    policy = decide(job, PolicyConfig(order=order, admission=admission, chunk_prefill=chunk, prefill_budget=budget,
                                      prefix_sharing=sharing, max_batch_tokens=400, max_batch_size=5))
    sched, results, covered, _ = drive(job, policy, num_blocks=32)
    check_results(job, policy, sched, results, covered)
    dup = len(job.requests) - 1
    if sharing:
        # each shared prefix is computed once; the duplicate shares every block but its last with request 0
        assert sched.prefix_hit_tokens >= 2 * BLOCK_SIZE * (len(job.requests) - 3)
        assert job.prefix.paths[dup] == job.prefix.paths[0]  # at most one of the two computed those blocks
        assert results[dup].hits + results[0].hits >= BLOCK_SIZE * len(job.prefix.paths[0])
    else:
        assert sched.prefix_hit_tokens == 0 and sched.prefix_wait_steps == 0
        assert all(not s.path and s.block_ids == s.private_blocks for s in results.values())


def test_children_wait_for_pending_nodes_and_hit_computed_ones():
    job = shared_workload()
    # budget 40: the owner's whole prompt fits one step with budget to spare, so its siblings wait, counted
    policy = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=40, prefix_sharing=True))
    sched, results, _, steps = drive(job, policy, num_blocks=64)
    assert sched.prefix_wait_steps > 0
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
    saturated = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=4, prefix_sharing=True))
    sched, _, _, steps = drive(job, saturated, num_blocks=64)
    assert sched.prefix_wait_steps == 0
    assert all(step.counts()[1] == 4 for step, _, _ in steps if any(not r.decode for r in step.rows) and step is not steps[-1][0])
    spare = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=40, prefix_sharing=True))
    sched, _, _, _ = drive(job, spare, num_blocks=64)
    assert sched.prefix_wait_steps > 0


def test_a_second_scheduler_on_the_same_job_starts_clean():
    job = shared_workload()
    policy = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=BUDGET, prefix_sharing=True))
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
    # Refused when the scheduler is built, so no request has run (before, one result came out, then a MemoryError)
    policy = decide(job, PolicyConfig(order="input", prefill_budget=BUDGET, prefix_sharing=True))
    with pytest.raises(ValueError, match="request Y0 needs 4 KV blocks while 3 are held for prefixes shared with "
                                         "requests later in the order; the pool has 6"):
        Scheduler(job, policy, BlockAllocator(7), [STOP])
    assert len(drive(job, policy, num_blocks=8)[1]) == 6  # one block more: Y0 fits beside X's held prefix
    plain = decide(job, PolicyConfig(order="input", prefill_budget=BUDGET, prefix_sharing=False))
    assert len(drive(job, plain, num_blocks=7)[1]) == 6  # without sharing nothing is held
    # the same pool runs to completion when a group's requests are adjacent (budget 64: X0's 50 tokens fit one
    # step with budget left, so X1 and X2 wait on its pending nodes and are counted)
    policy = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=64, prefix_sharing=True))
    assert list(policy.admission_order) == [0, 2, 4, 1, 3, 5]
    sched, results, _, _ = drive(job, policy, num_blocks=7)
    assert len(results) == 6 and sched.prefix_hit_tokens == 4 * 48
    assert sched.head_blocked_steps > 0 and sched.prefix_wait_steps > 0


def test_fixed_groups_are_checked_exactly_with_shared_nodes():
    job = interleaved_job()
    # groups [X0 X1 X2] [Y0 Y1 Y2]: a group needs 3 shared + 3 private = 6 blocks and holds nothing across the boundary
    policy = decide(job, PolicyConfig(order="prefix_dfs", admission="groups", prefill_budget=BUDGET, prefix_sharing=True,
                                      max_batch_tokens=1000, max_batch_size=3))
    assert [len(g.members) for g in policy.groups] == [3, 3]
    with pytest.raises(ValueError, match="fixed group 0 .* needs 6 KV blocks while 0 are held"):
        Scheduler(job, policy, BlockAllocator(6), [STOP])
    sched, results, _, _ = drive(job, policy, num_blocks=7)
    assert len(results) == 6 and sched.prefix_hit_tokens == 4 * 48
    # groups of one in input order X0 Y0 X1 ...: after X0, its 3 blocks stay held for X1 and X2, so Y0 needs 4 + 3.
    # Refused before step 1 with 6 blocks (without sharing the same job completes), runs with 7
    policy = decide(job, PolicyConfig(order="input", admission="groups", prefill_budget=BUDGET, prefix_sharing=True,
                                      max_batch_tokens=1000, max_batch_size=1))
    with pytest.raises(ValueError, match="fixed group 1 .* needs 4 KV blocks while 3 are held"):
        Scheduler(job, policy, BlockAllocator(7), [STOP])
    plain = decide(job, PolicyConfig(order="input", admission="groups", prefill_budget=BUDGET, prefix_sharing=False,
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
        base = decide(job, PolicyConfig(order="input", prefill_budget=rng.choice([8, 16, 64]), prefix_sharing=True,
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
    policy = decide(job, PolicyConfig(order="prefix_dfs", prefill_budget=BUDGET, prefix_sharing=True))
    sched, _, _, steps = drive(job, policy, num_blocks=32)
    trace = {"decode_rows", "prefill_tokens", "prefill_rows", "step_ms", "gpu_ms", "sched_ms", "end_ms", "graph_rows"}
    assert set(sched.occupancy()) | trace == set(STEP_FIELDS)  # the engine fills the rest from the step and clocks
    assert sched.occupancy()["free_blocks"] == sched.kv.capacity and sched.commits == len(steps)
    assert sched.counters() == {"head_blocked_steps": sched.head_blocked_steps, "prefix_hit_tokens":
                                sched.prefix_hit_tokens, "prefix_wait_steps": sched.prefix_wait_steps}
    # the step in which the last request with the job's longest max_tokens sampled its first token
    longest = max(it.req.max_tokens for it in job.requests)
    first = [k for k, (step, _, _) in enumerate(steps)
             if any(r.sample and r.start + len(r.token_ids) == job.requests[r.seq].prompt_len
                    and job.requests[r.seq].req.max_tokens == longest for r in step.rows)]
    assert sched.chain_start_step == max(first)


@pytest.mark.parametrize("config", [
    PolicyConfig(order="prefix_dfs", prefill_budget=BUDGET, prefix_sharing=True),
    PolicyConfig(order="input", prefill_budget=BUDGET, prefix_sharing=False),
    PolicyConfig(order="max_tokens_desc", admission="groups", max_batch_tokens=64, prefill_budget=BUDGET,
                 prefix_sharing=False),
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
        Scheduler(job, decide(job, PolicyConfig(prefix_sharing=sharing)), BlockAllocator(4), [STOP])


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
    Scheduler(job, policy, BlockAllocator(64), [STOP], max_positions=49)
    with pytest.raises(ValueError, match="request big needs 49 positions .* the model has 48"):
        Scheduler(job, policy, BlockAllocator(64), [STOP], max_positions=48)


def long_cap_then_classification(n=50):
    """The review's case for the adaptive budget: one generation capped at 8,192 tokens (64 prompt words) that
    stops at its first token, then n 512-word classification prompts. The auto budget spreads 25,664 prompt
    tokens over an 8,191-step chain that never happens: 256 per step."""
    reqs = [Request(id="long", prompt=" ".join(f"l{j}" for j in range(64)), max_tokens=8192)]
    reqs += [Request(id=f"c{i}", prompt=" ".join(f"c{i}_{j}" for j in range(512)), max_tokens=1) for i in range(n)]
    return analyze(reqs, StubTokenizer())


def test_adaptive_budget_takes_the_prefill_once_the_long_chain_stops():
    job = long_cap_then_classification()
    steps = {}
    for budget in ("auto", "adaptive"):
        policy = decide(job, PolicyConfig(order="decode_ratio_desc", prefill_budget=budget, prefix_sharing=False))
        assert policy.prefill_budget == 256 and policy.prefill_budget_adaptive is (budget == "adaptive")
        _, results, _, trace = drive(job, policy, num_blocks=4000, stop_at={0: 1})
        assert results[0].finish_reason == "stop" and len(results[0].outputs) == 1
        steps[budget] = len(trace)
    assert steps == {"auto": 101, "adaptive": 3}  # 256; then 16,384 once no decode is left; then the rest


def test_adaptive_budget_follows_its_formula_while_requests_decode():
    """Every step's budget is the formula recomputed here from scratch: prompt tokens not yet computed (the
    queue's, plus each admitted request's, never its decode positions) over the longest decode chain left. A
    long cap stops at once while 40 generations keep decoding for 300 steps, so the budget rises above the
    floor with decode rows in the step, not only once decode is over."""
    reqs = [Request(id="long", prompt=" ".join(f"l{j}" for j in range(64)), max_tokens=8192)]
    reqs += [Request(id=f"g{i}", prompt=" ".join(f"g{i}_{j}" for j in range(30)), max_tokens=300, ignore_eos=True)
             for i in range(40)]
    reqs += [Request(id=f"c{i}", prompt=" ".join(f"c{i}_{j}" for j in range(500)), max_tokens=1) for i in range(300)]
    job = analyze(reqs, StubTokenizer())
    policy = decide(job, PolicyConfig(order="decode_ratio_desc", prefill_budget="adaptive", prefix_sharing=False))
    sched = Scheduler(job, policy, BlockAllocator(20_000), [STOP])
    raised_while_decoding = False
    while not sched.done:
        step = sched.next_step()
        live = list(sched.admitted.values())
        queued = [job.requests[i] for i in sched.queue]
        remaining = sum(it.prompt_len for it in queued) + sum(max(0, s.prompt_len - s.computed) for s in live)
        chain = max([it.req.max_tokens - 1 for it in queued] + [s.max_tokens - max(len(s.outputs), 1) for s in live])
        assert sched.step_budget == min(max(-(-remaining // max(chain, 1)), policy.prefill_budget), MAX_PREFILL_BUDGET)
        raised_while_decoding |= sched.step_budget > policy.prefill_budget and step.counts()[0] > 0
        sampled = [STOP if r.seq == 0 else TOKEN for r in step.rows if r.sample]
        sched.commit(step, sampled)
    assert raised_while_decoding


def test_adaptive_budget_is_auto_when_the_caps_are_the_lengths():
    """With every request running to max_tokens the chain the auto budget assumed is the one that runs, so the
    adaptive budget stays near it and never costs a step."""
    reqs = [Request(id=f"g{i}", prompt=" ".join(f"g{i}_{j}" for j in range(40)), max_tokens=60, ignore_eos=True)
            for i in range(4)]
    reqs += [Request(id=f"c{i}", prompt=" ".join(f"c{i}_{j}" for j in range(600)), max_tokens=1) for i in range(40)]
    job = analyze(reqs, StubTokenizer())  # 24,160 prompt tokens over a 59-step chain: auto is 410, off the clamps
    runs = {}
    for budget in ("auto", "adaptive"):
        policy = decide(job, PolicyConfig(order="decode_ratio_desc", prefill_budget=budget, prefix_sharing=False))
        _, _, _, trace = drive(job, policy, num_blocks=2000, stop_at={})
        runs[budget] = (policy.prefill_budget, len(trace), max(step.counts()[1] for step, _, _ in trace))
    assert runs["adaptive"][0] == runs["auto"][0] == 410 and runs["adaptive"][1] <= runs["auto"][1]
    assert runs["adaptive"][2] <= 2 * runs["auto"][2]  # rounding only: remaining / chain stays near the floor
