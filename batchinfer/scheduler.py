"""Scheduler: decisions made at every step of the batchinfer engine. Pure Python, no torch.

It follows a Policy (decided once, up front, in policy.py) and decides, per step:
1. admission: which requests get their KV reservation now. admission="continuous" admits the head of
   the order whenever its reservation fits the free pool; admission="groups" admits the next fixed
   group whole once every sequence of the previous group has finished. Strict order, no skipping.
2. composition: every decode row (1 query token each; never deferred), then prefill tokens in
   admission order up to the step's prefill budget. With chunk_prefill a prompt may be split across steps
   (a prefill chunk); without it a prompt is computed in exactly one step, and a lone prompt larger
   than the budget runs alone. The budget is the policy's, unless policy.prefill_budget_adaptive: then each
   step spreads the prefill still to compute over the longest decode chain still to run, never below the
   policy's budget. A chain is counted to its request's max_tokens, so when a long-capped generation stops
   early the prefill behind it is no longer spread over steps that will not happen.
3. prefix sharing (policy.prefix_sharing): the prompt blocks the job's prefix trie marks as shared are KV
   blocks with one writer and many readers. The first admitted user of a trie node allocates its block
   and owns it; the node is `pending` until the owner's prefill has covered it, `computed` from then on.
   Later users skip computed nodes (prefix hits) and take no chunk while their next node is pending
   under someone else (a prefix wait). Rows never read a block written in the same step. A node's block
   is released when its last user finishes, admitted or not: the lifetime is known before the first
   step, so nothing is ever evicted. Because order is strict, a block held for a later user can leave the
   head of the order no room while nothing runs; _check refuses such an order before the first step, for
   fixed groups and continuous admission alike. Waits only point at earlier-admitted owners, and an owner
   cannot finish before computing its nodes, so the earliest admitted unfinished prefill never waits.
The executor runs the Step (schema.py: the scheduler -> executor record) and returns one sampled id per sampling
row; commit() applies them. The engine records the scheduler through occupancy() (each step) and counters() (the
run), never through its internals; per request it reads only the finished Sequences commit() returns, whose
*_step fields say in which step each event happened.
"""
import math
import time
from collections import deque
from dataclasses import dataclass, field

from .kv import BLOCK_SIZE, reservation_blocks
from .schema import MAX_PREFILL_BUDGET, Row, Step

MAX_SEQS = 1024  # admitted sequences; bounds decode rows per step
# Hard cap on step tokens. The widest step the scheduler can build (MAX_SEQS sampled decode rows and a prefill chunk
# of MAX_PREFILL_BUDGET) is the executor's memory probe, and what its auto pool sizing measures.
STEP_LIMIT = MAX_PREFILL_BUDGET + MAX_SEQS
FREE, PENDING, COMPUTED = "free", "pending", "computed"  # trie node states, owned by the scheduler


@dataclass
class Sequence:
    """An admitted request."""
    idx: int
    prompt_ids: list[int]
    max_tokens: int
    ignore_eos: bool
    block_ids: list[int]  # blocks of the path's trie nodes, then private_blocks
    private_blocks: list[int]  # the rest of the prompt and the decode slots; released when it finishes
    path: list[int]  # trie node ids covering positions [0, BLOCK_SIZE * len(path)); empty without sharing
    group: int | None  # fixed-group index, None under continuous admission
    admitted_at: float
    computed: int = 0  # positions whose KV is written
    marked: int = 0  # path nodes already checked for completion at commit
    hits: int = 0  # prompt positions skipped because another sequence computed them (all at the front)
    outputs: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    # step indices (Scheduler.commits at the time) of the request's events, for the engine's request trace:
    # KV reserved (next_step's admit), first prefill chunk (next_step), first token and finish (commit)
    admitted_step: int | None = None
    prefill_start_step: int | None = None
    first_token_step: int | None = None
    finished_step: int | None = None

    @property
    def prompt_len(self):
        return len(self.prompt_ids)


class Scheduler:
    def __init__(self, job, policy, allocator, stop_ids, max_positions=None):
        """max_positions: the model's position limit (config.max_position_embeddings); None skips the check."""
        self.items, self.policy, self.kv, self.stop = job.requests, policy, allocator, set(stop_ids)
        self.max_positions = max_positions
        self.budget = policy.prefill_budget  # every step's, or the floor when adaptive
        self.step_budget = self.budget  # the budget of the step next_step() built last
        self.adaptive = bool(policy.prefill_budget_adaptive)
        order = policy.admission_order
        self.queued_prompt_tokens = sum(self.items[i].prompt_len for i in order)  # prompts not yet admitted
        self.chain_after = [0] * (len(order) + 1)  # chain_after[k]: longest max_tokens - 1 over order[k:]
        for k in range(len(order) - 1, -1, -1):
            self.chain_after[k] = max(self.chain_after[k + 1], self.items[order[k]].req.max_tokens - 1)
        self.queue = deque(order)  # consumed front to back
        self.groups = deque(enumerate(policy.groups)) if policy.admission == "groups" else None  # (index, Group)
        self.admitted = {}  # idx -> Sequence, in admission order
        self.admission_log = []  # idx in the order they were admitted
        self.head_blocked_steps = 0  # steps in which the next request waited for KV blocks or a sequence slot
        # prefix sharing: per-run state of the trie's nodes, indexed by node id (the trie itself is read-only)
        self.sharing, self.trie = policy.prefix_sharing, job.prefix
        n = len(self.trie.nodes) if self.sharing else 0
        self.node_state = [FREE] * n
        self.node_block = [None] * n
        self.node_owner = [None] * n  # the sequence that allocated and writes the block
        self.node_remaining = [len(x.users) for x in self.trie.nodes[:n]]  # users not yet finished
        self.node_admitted = [0] * n  # users admitted and not yet finished
        self.trie_blocks = 0  # blocks held by trie nodes now, shared or not (a node may have one user)
        self.pinned_blocks = 0  # of those, held for users none of which is admitted
        self.prefix_hit_tokens = 0
        self.shared_written = 0  # KV positions written inside trie nodes; only a node's owner writes it, so once each
        self.kv_at_step = {"kv_reserved_tokens": 0, "kv_written_tokens": 0, "unprefilled_tokens": 0}  # at next_step
        self.prefix_wait_steps = 0  # (sequence, step) pairs with budget left that took no chunk: a node was pending elsewhere
        self.longest = max((it.req.max_tokens for it in self.items), default=0)
        self.commits = 0
        self.chain_start_step = None  # the step in which the last longest-max_tokens request sampled its first token
        self._check()

    # reservations ---------------------------------------------------------------------------------

    def _path(self, i):
        return self.trie.paths[i] if self.sharing else ()

    def _private(self, i):
        """Blocks for the prompt after the path and the max_tokens - 1 decode slots. The last prompt token is
        never on the path, so a non-empty prompt always has at least one private token."""
        it = self.items[i]
        return reservation_blocks(it.prompt_len - BLOCK_SIZE * len(self._path(i)), it.req.max_tokens)

    def _need(self, i):
        """Blocks admitting request i takes now: path nodes nobody has allocated, plus its private blocks.
        Before anything is admitted this equals reservation_blocks(prompt_len, max_tokens)."""
        return sum(1 for nid in self._path(i) if self.node_state[nid] == FREE) + self._private(i)

    def _group_need(self, members):
        seen, need = set(), 0
        for i in members:
            for nid in self._path(i):
                if self.node_state[nid] == FREE and nid not in seen:
                    seen.add(nid)
                    need += 1
            need += self._private(i)
        return need

    def _check(self):
        """Refuse up front what could never run, instead of stalling or overrunning mid-job. Fixed groups
        are admitted whole and in order, so their needs are exact: a group takes the nodes no earlier
        group allocated plus its private blocks, from a pool that still holds every node whose users
        span the group boundary (shared with a later group). Without sharing this is the plain sum.
        Continuous admission is the same check with every request a group of its own: admission only stalls
        with nothing running, and then every request before the head has finished, so the pool holds exactly
        the nodes whose users span the head. A job that passes never reaches _stuck."""
        cap = self.kv.capacity
        for i in self.policy.admission_order:
            it = self.items[i]
            if self.max_positions is not None and it.prompt_len + it.req.max_tokens - 1 > self.max_positions:
                raise ValueError(f"request {it.req.id} needs {it.prompt_len + it.req.max_tokens - 1} positions "
                                 f"({it.prompt_len} prompt + {it.req.max_tokens} max_tokens - 1); the model has "
                                 f"{self.max_positions}")
            if self._need(i) > cap:
                raise ValueError(f"request {self.items[i].req.id} reserves {self._need(i)} KV blocks; the pool has {cap}")
            if not self.policy.chunk_prefill and self.items[i].prompt_len > MAX_PREFILL_BUDGET:
                raise ValueError(f"request {self.items[i].req.id} has {self.items[i].prompt_len} prompt tokens; without "
                                 f"chunked prefill a prompt must fit one step ({MAX_PREFILL_BUDGET})")
        if self.groups is None:
            order = self.policy.admission_order
            over = self._overflow([[i] for i in order])  # without sharing nothing is held: _need covered it
            if over is not None:
                k, need, held = over
                raise ValueError(f"request {self.items[order[k]].req.id} needs {need} KV blocks while {held} are held "
                                 f"for prefixes shared with requests later in the order; the pool has {cap} "
                                 f"(order=prefix_dfs keeps them adjacent, prefix_sharing=off holds none, or enlarge "
                                 f"the pool)")
            return
        groups = [g.members for _, g in self.groups]
        over = self._overflow(groups)
        if over is not None:
            gi, need, held = over
            raise ValueError(f"fixed group {gi} ({len(groups[gi])} requests) needs {need} KV blocks while {held} are held for "
                             f"prefixes shared with later groups; the pool has {cap}")

    def _overflow(self, groups):
        """The first group, as (index, need, held), that does not fit the pool when groups (member lists in
        admission order) run one after another, each starting once every earlier one has finished; None when all
        fit. need: the nodes the group allocates first plus its private blocks. held: the nodes an earlier group
        allocated that a later one still uses."""
        group_of = {i: k for k, members in enumerate(groups) for i in members}
        new, delta = [0] * len(groups), [0] * (len(groups) + 1)  # nodes first used by group k; held-count changes
        for nid in range(len(self.node_remaining)):
            gs = [group_of[u] for u in self.trie.nodes[nid].users]
            first, last = min(gs), max(gs)
            new[first] += 1
            delta[first + 1] += 1  # held from the group after its first user's ...
            delta[last + 1] -= 1  # ... through its last user's group
        held = 0
        for k, members in enumerate(groups):
            held += delta[k]
            need = new[k] + sum(self._private(i) for i in members)
            if need + held > self.kv.capacity or len(members) > MAX_SEQS:
                return k, need, held
        return None

    @property
    def done(self):
        return not self.queue and not self.admitted

    def occupancy(self):
        """This moment's state, for the per-step trace (metrics.STEP_FIELDS)."""
        return {"admitted": len(self.admitted), "free_blocks": self.kv.free, "trie_blocks": self.trie_blocks,
                "pinned_blocks": self.pinned_blocks, "step_prefill_budget": self.step_budget, **self.kv_at_step}

    def counters(self):
        """Totals over the run so far, for the engine's counts."""
        return {"head_blocked_steps": self.head_blocked_steps, "prefix_hit_tokens": self.prefix_hit_tokens,
                "prefix_wait_steps": self.prefix_wait_steps}

    # admission ------------------------------------------------------------------------------------

    def _admit(self, i, group):
        it, path = self.items[i], list(self._path(i))
        new = [nid for nid in path if self.node_state[nid] == FREE]
        blocks = self.kv.reserve(len(new) + self._private(i))
        for nid, b in zip(new, blocks):
            self.node_state[nid], self.node_block[nid], self.node_owner[nid] = PENDING, b, i
        self.trie_blocks += len(new)
        new = set(new)
        for nid in path:
            if nid not in new and self.node_admitted[nid] == 0:
                self.pinned_blocks -= 1  # allocated earlier, held for this request; now in use again
            self.node_admitted[nid] += 1
        s = Sequence(idx=i, prompt_ids=it.token_ids, max_tokens=it.req.max_tokens, ignore_eos=it.req.ignore_eos,
                     block_ids=[self.node_block[nid] for nid in path] + blocks[len(new):],
                     private_blocks=blocks[len(new):], path=path, group=group, admitted_at=time.perf_counter(),
                     admitted_step=self.commits)
        self._advance(s)
        self.admitted[i] = s
        self.admission_log.append(i)
        self.queued_prompt_tokens -= it.prompt_len

    def _stuck(self, ident, need):
        """The head cannot be admitted and nothing runs. _check refuses every job that would get here before its
        first step, so this is a guard, not a path."""
        return MemoryError(f"request {ident} needs {need} KV blocks, {self.kv.free} are free and nothing is running; "
                           f"{self.pinned_blocks} blocks hold shared prefixes for requests further down the order "
                           f"(order=prefix_dfs keeps them adjacent; or enlarge the pool)")

    def admit(self):
        if self.groups is not None:
            if not self.admitted and self.groups:
                gi, g = self.groups.popleft()
                need = self._group_need(g.members)
                if need > self.kv.free:
                    raise self._stuck(f"group {gi}", need)
                for i in g.members:
                    assert self.queue.popleft() == i
                    self._admit(i, gi)
            return
        while self.queue:
            head, need = self.queue[0], self._need(self.queue[0])
            if len(self.admitted) >= MAX_SEQS or need > self.kv.free:
                if not self.admitted:
                    raise self._stuck(self.items[head].req.id, need)
                self.head_blocked_steps += 1
                return
            self._admit(self.queue.popleft(), None)

    # composition ----------------------------------------------------------------------------------

    def _advance(self, s):
        """Move s.computed over path nodes other sequences have computed (prefix hits). Returns False when the
        next node is pending under another owner, so s takes no prefill chunk this step. Once the next node
        is s's own, every later node on the path is too: an earlier owner of a later node would have
        allocated this one as well."""
        while s.computed < BLOCK_SIZE * len(s.path):
            nid = s.path[s.computed // BLOCK_SIZE]
            if self.node_owner[nid] == s.idx:
                return True
            assert self.node_state[nid] != FREE, "a path node of an admitted sequence is always allocated"
            if self.node_state[nid] != COMPUTED:
                return False
            assert s.computed % BLOCK_SIZE == 0
            s.computed += BLOCK_SIZE
            s.hits += BLOCK_SIZE
            self.prefix_hit_tokens += BLOCK_SIZE
        return True

    def _step_budget(self, unprefilled):
        """Rule 2's budget for the step being built: the policy's, or under adaptive the prefill still to compute
        (the queue's prompts plus the admitted ones' unprefilled tokens) over the longest decode chain still to
        run, within [the policy's budget, MAX_PREFILL_BUDGET]. Prefix hits not yet taken are counted as prefill,
        which only over-estimates; the ceiling is what the executor probed."""
        if not self.adaptive:
            return self.budget
        live = self.admitted.values()
        # decode forwards left: the first output comes from the prefill forward, each later one from a decode row
        chain = max([self.chain_after[len(self.admission_log)]] + [s.max_tokens - max(len(s.outputs), 1) for s in live])
        remaining = self.queued_prompt_tokens + unprefilled
        return min(max(math.ceil(remaining / max(chain, 1)), self.budget), MAX_PREFILL_BUDGET)

    def next_step(self):
        self.admit()
        # admitted prompt positions not yet computed; computed runs past prompt_len as a sequence decodes
        unprefilled = sum(max(0, s.prompt_len - s.computed) for s in self.admitted.values())
        budget = self.step_budget = self._step_budget(unprefilled)
        # KV while this step runs: blocks held (reservations are for a request's whole life) against positions
        # already written in them, a shared block counted once, and the admitted prompt tokens not yet prefilled,
        # which is most of what is held but unwritten while prompts wait for their chunks. free_blocks in
        # occupancy() is sampled after commit.
        private = sum(max(0, s.computed - BLOCK_SIZE * len(s.path)) for s in self.admitted.values())
        self.kv_at_step = {"kv_reserved_tokens": BLOCK_SIZE * (self.kv.capacity - self.kv.free),
                           "kv_written_tokens": self.shared_written + private, "unprefilled_tokens": unprefilled}
        rows = [Row(s.idx, s.computed, [s.outputs[-1]], s.block_ids, sample=True, decode=True)
                for s in self.admitted.values() if s.computed >= s.prompt_len]
        prefill = 0
        for s in self.admitted.values():
            if s.computed >= s.prompt_len:
                continue
            if prefill >= budget:
                break  # budget spent: the sequences behind neither get a chunk nor count as waiting
            if not self._advance(s):
                self.prefix_wait_steps += 1
                continue
            remaining = s.prompt_len - s.computed
            left = budget - prefill
            if self.policy.chunk_prefill:
                n = min(remaining, left)
            elif remaining <= left or prefill == 0:
                n = remaining
            else:
                n = 0
            if n <= 0:
                break
            rows.append(Row(s.idx, s.computed, s.prompt_ids[s.computed:s.computed + n], s.block_ids,
                            sample=s.computed + n == s.prompt_len, decode=False))
            if s.prefill_start_step is None:
                s.prefill_start_step = self.commits
            prefill += n
        step = Step(rows)
        assert rows, "an empty step would never make progress"
        assert step.tokens <= STEP_LIMIT, f"step of {step.tokens} tokens over the step limit {STEP_LIMIT}"
        return step

    def _mark(self, s):
        """Owned path nodes that s.computed now covers become computed, readable by later steps."""
        done = min(len(s.path), s.computed // BLOCK_SIZE)
        for k in range(s.marked, done):
            nid = s.path[k]
            if self.node_owner[nid] == s.idx:
                self.node_state[nid] = COMPUTED
        s.marked = done

    def commit(self, step, sampled):
        """Apply one step's sampled ids (one per sampling row, in row order). Returns the sequences that
        finished, their KV blocks already released."""
        assert len(sampled) == sum(r.sample for r in step.rows), "one sampled id per sampling row"
        it, finished = iter(sampled), []
        for row in step.rows:
            s = self.admitted[row.seq]
            assert row.start == s.computed, "rows must tile each sequence with no gap or overlap"
            top = BLOCK_SIZE * len(s.path)  # a row below it writes s's own pending nodes (rule 3)
            self.shared_written += max(0, min(row.start + len(row.token_ids), top) - row.start)
            s.computed += len(row.token_ids)
            self._mark(s)
            if row.sample:
                t = next(it)
                s.outputs.append(t)
                if len(s.outputs) == 1:
                    s.first_token_step = self.commits
                    if s.max_tokens == self.longest:
                        self.chain_start_step = self.commits
                if t in self.stop and not s.ignore_eos:
                    s.finish_reason = "stop"
                elif len(s.outputs) == s.max_tokens:
                    s.finish_reason = "length"
                if s.finish_reason:
                    s.finished_step = self.commits
                    finished.append(s)
        for s in finished:
            del self.admitted[s.idx]
            self.kv.release(s.private_blocks)
            for nid in s.path:
                self.node_admitted[nid] -= 1
                self.node_remaining[nid] -= 1
                if self.node_remaining[nid] == 0:
                    assert self.node_state[nid] == COMPUTED, "an owner finishes only after computing its nodes"
                    self.kv.release([self.node_block[nid]])
                    self.node_state[nid], self.node_block[nid], self.node_owner[nid] = FREE, None, None
                    self.trie_blocks -= 1
                    self.shared_written -= BLOCK_SIZE
                elif self.node_admitted[nid] == 0:
                    self.pinned_blocks += 1
        self.commits += 1
        return finished
