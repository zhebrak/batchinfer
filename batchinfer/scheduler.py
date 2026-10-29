"""Scheduler: decisions made at every step of the batchinfer engine. Pure Python, no torch.

It follows a Policy (decided once, up front, in policy.py) and decides, per step:
1. admission: which requests get their KV reservation now. admission="continuous" admits the head of
   the order whenever its reservation fits the free pool; admission="groups" admits the next fixed
   group whole once every sequence of the previous group has finished. Strict order, no skipping.
2. composition: every decode row (1 query token each; never deferred), then prefill tokens in
   admission order up to the step's prefill budget. With chunk_prefill a prompt may be split across steps
   (a prefill chunk); without it a prompt is computed in exactly one step, and a lone prompt larger
   than the budget runs alone. The budget is the policy's, unless policy.prefill_budget_ridge: then a step
   with decode rows carries prefill up to the card's ridge (ridge_budget), the prefill a graphed decode step
   absorbs at about no cost, and never less than what spreads the prefill still to compute over the
   longest decode chain still to run, so the prefill never lags the chain; a step with no decode rows (the
   opening prefill, a classification job) runs the widest step the executor sized, where prefill is
   compute-bound and nothing waits.
3. prefix sharing (policy.sharing): the prompt blocks the job's prefix trie marks as shared are KV
   blocks with one writer and many readers. The first admitted user of a trie node allocates its block
   and owns it; the node is `pending` until the owner's prefill has covered it, `computed` from then on.
   Later users skip computed nodes (prefix hits), and a node the owner's chunk completes in this very step
   counts as computed for the rows after it (every backend writes a layer's KV for all of a step's rows before
   that layer's attention, so a later row reads what an earlier row wrote); they take no chunk while their
   next node is pending under someone else and not completed this step (a prefix wait). sharing=token adds what
   a radix tree shares and whole blocks miss: when a request reaches the first block it must write itself, the
   leading tokens that block shares with a sibling block already computed in an earlier step (another child of
   the same trie node, or another live request's private tail under the same node) are copied from it (Step.copies,
   run by the executor before the forward) and counted as hits, and the request's chunk starts after them; never
   the block's last token, which the request computes itself, and never more than a block. A node's block
   is released when its last user finishes, admitted or not (cache=radix keeps it instead, keyed by its prefix,
   for the next job this engine runs, and evicts the oldest kept block when a job needs the room: a node of a
   later job's trie with the same prefix starts computed): the lifetime is known before the first
   step, so nothing is ever evicted. Because order is strict, a block held for a later user can leave the
   head of the order no room while nothing runs; _check refuses such an order before the first step, for
   fixed groups and continuous admission alike. Waits only point at earlier-admitted owners, and an owner
   cannot finish before computing its nodes, so the earliest admitted unfinished prefill never waits.
4. memory (policy.memory): how a request's KV blocks are held. reserve: its whole life's blocks (prompt plus
   max_tokens - 1 decode slots) at admission, so the pool can never run dry and nothing is ever preempted. grow:
   the prompt's blocks at admission and a decode block as decode reaches it (_cover), with admission checked
   against a Timeline, the private blocks every live request can hold at every future step: a request still
   prefilling projects its whole life (its finish step is unknown), a decoding one the staircase its decode
   writes, which ends at its first token's step plus max_tokens - 1, the latest it can finish (every decoding
   sequence gets a row every step and nothing is preempted). The projection bounds what is held at every future
   step, so the pool never runs dry under grow either; what it buys is that a live request's blocks are counted
   when it will hold them, not for its whole life, so requests that end before others peak make room. preempt:
   grow's allocation under a watermark (admission leaves a step of decode growth for every live request free),
   and when a decode row still finds the pool dry, room is made: shared blocks held for requests not yet admitted
   (pinned nodes) are released deepest first, then the youngest request is preempted: its private blocks go back,
   its trie nodes stay (a computed node stays readable; a pending one it owned is rewritten when it resumes), and
   it re-enters at the head of the queue to be admitted again, prefilling its prompt and the tokens it had
   produced (feed), then decoding on; greedy decoding makes the recomputed KV the same, so its outputs are too.
The executor runs the Step (schema.py: the scheduler -> executor record) and returns one sampled id per sampling
row; commit() applies them. The engine records the scheduler through occupancy() (each step) and counters() (the
run), never through its internals; per request it reads only the finished Sequences commit() returns, whose
*_step fields say in which step each event happened.
"""
import math
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from . import cascade
from .kv import DEFAULT_BLOCK_SIZE, reservation_blocks
from .schema import MAX_PREFILL_BUDGET, MAX_SEQS, Limits, Row, Step
FREE, PENDING, COMPUTED = "free", "pending", "computed"  # trie node states, owned by the scheduler


def ridge_budget(ridge, decoding, remaining, chain, graph_tokens, max_prefill):
    """prefill_budget=ridge: the prefill tokens the step being built may carry. A step with no decode rows gets the
    widest step the executor sized (max_prefill): the opening prefill, and prefill-only jobs, are compute-bound
    and nothing waits on them. A step with decode rows carries the card's ridge, the prefill tokens a graphed decode
    step absorbs at about no cost as scripts/ridge_calibration.py measured it on top of the fewest decode rows it
    sweeps (more decode rows read more KV, which buys more free compute, so the number is conservative above that
    count), and never less than what spreads the `remaining` prefill over the `chain` decode steps still to run, so
    the prefill never lags the chain; capped at the largest mixed graph's tokens (graph_tokens[-1]) so the step
    still replays a graph, or at max_prefill without mixed graphs."""
    if not decoding:
        return max_prefill
    cap = graph_tokens[-1] if graph_tokens else max_prefill
    need = math.ceil(remaining / max(chain, 1))
    return min(max(ridge, need), cap)


@dataclass
class Sequence:
    """An admitted request."""
    idx: int
    prompt_ids: list[int]
    max_tokens: int
    ignore_eos: bool
    block_ids: list[int]  # blocks of the path's trie nodes, then private_blocks
    private_blocks: list[int]  # the rest of the prompt and the decode slots; released when it finishes
    path: list[int]  # trie node ids covering positions [0, self.B * len(path)); empty without sharing
    group: int | None  # fixed-group index, None under continuous admission
    admitted_at: float
    feed: list[int] = None  # what prefill rows read: the prompt (after a preemption, the prompt and the outputs so far)
    target: int = 0  # len(feed): the position from which the sequence decodes
    tail_copied: bool = False  # sharing=token: the one copy into its first own block was tried
    slot: int = 0  # pool=contiguous: the region its rows read through, for this admission
    inflight: int = 0  # overlap: sampling rows of this sequence planned and not yet committed (at most two: the step
    #                    in flight's and the one just planned, until the first commits)
    high: int = 0  # positions it held when last preempted: re-hitting them is neither a recompute nor a new hit
    parked: tuple | None = None  # preempt: [lo, hi) positions whose KV its parked blocks still hold at re-admission
    computed: int = 0  # positions whose KV is written
    marked: int = 0  # path nodes already checked for completion at commit
    hits: int = 0  # prompt positions skipped because another sequence computed them (all at the front), this admission
    hit_share: int = 0  # its share of the run's prefix_hit_tokens: never reset, so re-reading what it held before a
    #                     preemption (neither a recompute nor a new hit) does not count; what the request trace records
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


class Timeline:
    """memory="grow": the private blocks the live requests can hold at every future step, as a difference array over
    a ring of `horizon` steps ahead (the longest max_tokens plus one is enough: every staircase ends within it).
    flat holds the whole-life blocks of requests still prefilling; now the staircase blocks at the current step;
    the ring the changes at each later step. peak() is the most the live set can hold at any future step."""

    def __init__(self, horizon, block_size, lag=0):
        self.B = block_size
        self.lag = lag  # steps between a sequence's last row and the commit that releases its blocks
        self.flat, self.now, self.base = 0, 0, 0
        self.ring = [0] * (horizon + 2)
        self._peak = None  # the staircases' peak, cached between changes (flat is added on top, the same at every step)

    def _at(self, k):
        return (self.base + k) % len(self.ring)

    def advance(self):
        """To the next step: its change lands in now."""
        self.base = self._at(1)
        self.now += self.ring[self.base]
        self.ring[self.base] = 0
        self._peak = None

    def stair(self, s, sign, age):
        """Add (+1) or remove (-1) the staircase of a decoding sequence whose first token was sampled at the step
        before the one the timeline stands at minus `age` (0 at its first token): at k steps ahead its decode row
        writes position target + age + k, so it holds ceil((target + age + k + 1) / B) - path blocks of its own,
        until its last possible step, max_tokens - 2 - age ahead (the row there samples its max_tokens-th token);
        after it, nothing. Under overlap (lag 1) its finish is committed, and its blocks released, one step after
        that row, when the next step has already been planned: the staircase holds one step longer."""
        self._peak = None
        end = s.max_tokens - 2 - age + self.lag  # k of its last decode row, plus the commit's lag
        if end < 0:
            return  # finished: with its first token (max_tokens 1), or by now
        blocks = lambda k: -(-(s.target + age + k + 1) // self.B) - len(s.path)  # noqa: E731
        held = blocks(0)
        self.now += sign * held
        for k in range(1, end + 1):
            b = blocks(k)
            if b != held:
                self.ring[self._at(k)] += sign * (b - held)
                held = b
        self.ring[self._at(end + 1)] -= sign * held

    def peak(self):
        if self._peak is None:
            cur = best = self.now
            for k in range(1, len(self.ring)):
                cur += self.ring[self._at(k)]
                best = max(best, cur)
            self._peak = best
        return self._peak + self.flat


class Scheduler:
    def __init__(self, job, policy, allocator, stop_ids, limits=None, retained=None):
        """limits.overlap: the engine plans step N+1 before step N's ids are back. Then next_step advances what a
        row covers (computed, the shared-write count, the node marks: the stream runs N+1 after N, so N's writes are
        done before anything of N+1 reads them) and commit only applies ids and finishes; a decode row whose last
        token is still on the device names its slot in the previous step's output (Row.prev); a sequence with a
        sample in flight gets no row that would run past its max_tokens, and a row planned for one that an EOS then
        ends is discarded at its commit (counted in discarded); a preempted sequence's in-flight sample lands on its
        record; next_step returns None when nothing can be planned until the step in flight commits.
        limits: what the executor built (schema.Limits): the step caps, the model's position limit (None skips the
        check) and the pool; without one, the module caps over the allocator's pool, which is what the tests run.
        retained: cache=radix's blocks kept from earlier jobs on this allocator, an OrderedDict prefix key -> block
        the engine owns across jobs (oldest first); this job takes the ones its trie has and adds its own at the end."""
        self.items, self.policy, self.kv, self.stop = job.requests, policy, allocator, set(stop_ids)
        self.limits = limits or Limits(allocator.num_blocks, DEFAULT_BLOCK_SIZE, MAX_SEQS, MAX_PREFILL_BUDGET, None)
        assert self.limits.num_blocks == allocator.num_blocks, "the allocator's pool is the executor's"
        self.B = self.limits.block_size  # tokens per KV block: the trie's granularity too
        assert job.prefix.block_size == self.B, "the trie is built at the executor's block size (analysis.analyze)"
        self.max_positions = self.limits.max_positions
        self.budget = policy.prefill_budget  # every step's, or under ridge the cap of a step without decode rows
        self.step_budget = self.budget  # the budget of the step next_step() built last
        self.ridge = policy.ridge_tokens if policy.prefill_budget_ridge else None
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
        self.sharing, self.trie = policy.sharing != "none", job.prefix
        self.token_sharing = policy.sharing == "token"
        n = len(self.trie.nodes) if self.sharing else 0
        self.node_state = [FREE] * n
        self.node_block = [None] * n
        self.node_owner = [None] * n  # the sequence that allocated and writes the block
        self.node_remaining = [len(x.users) for x in self.trie.nodes[:n]]  # users not yet finished
        self.node_admitted = [0] * n  # users admitted and not yet finished
        self.trie_blocks = 0  # blocks held by trie nodes now, shared or not (a node may have one user)
        self.pinned_blocks = 0  # of those, held for users none of which is admitted
        self.pinned = set()  # their node ids: what _evict_pinned scans, not the whole trie
        self.prefix_hit_tokens = 0
        self.shared_written = 0  # KV positions written inside trie nodes; only a node's owner writes it, so once each
        self.kv_at_step = {"kv_reserved_tokens": 0, "kv_written_tokens": 0, "unprefilled_tokens": 0}  # at next_step
        self.prefix_wait_steps = 0  # (sequence, step) pairs with budget left that took no chunk: a node was pending elsewhere
        self.longest = max((it.req.max_tokens for it in self.items), default=0)
        self.memory = policy.memory or "reserve"
        self.watermark_steps = policy.watermark_steps or 1
        self.overlap = bool(self.limits.overlap)
        self.timeline = Timeline(self.longest + 1, self.B, lag=int(self.overlap)) if self.memory == "grow" else None
        self.preempted = {}  # idx -> its Sequence, between a preemption and its re-admission (preempt mode)
        self.preemptions = 0
        # preempt mode: a preempted sequence's private blocks stay allocated, parked, idx -> (blocks, lo, hi): the
        # positions [lo, hi) they hold. They go back to the pool only as the pool needs them, the oldest parking
        # first and a row's last block first, so a sequence re-admitted soon (it is at the head of the queue) takes
        # them back and recomputes only what was given up: what vLLM's prefix cache does for its preempted requests
        self.parked = OrderedDict()
        self.parked_tokens = 0  # positions taken back from parked blocks instead of recomputed
        self.planned = 0  # steps planned (== commits unless a step is in flight): what admission and chunks are stamped with
        self.prev_slots = {}  # overlap: sequence -> its slot in the step in flight's sampled output
        self.discarded = 0  # overlap: sampled ids of rows planned for sequences an earlier commit had finished
        # pool=contiguous: one slot per live sequence (0 is the pad slot); a freed slot cools for one commit before
        # it is handed out again, so the last step that read it (under overlap, the one in flight when its sequence
        # finished) has been settled by then, and the executor may drop its mappings without waiting
        self.slots = list(range(self.limits.max_seqs, 0, -1)) if self.limits.pool == "contiguous" else None
        self.cooling = []  # overlap: freed slots, free again at the next commit, by which their last reader has settled
        self.slot_resets = []
        self.finished_preempted = []  # overlap: sequences whose in-flight sample ended them while preempted
        self.recomputed_tokens = 0  # prompt and output positions prefilled again after a preemption
        # cache=radix: blocks kept from earlier jobs, taken when a request whose path has the node is admitted (a
        # node is never pinned by a job that has not reached it), the oldest evicted when admission needs the room
        self.retained = retained if policy.cache == "radix" and self.sharing else None
        self.node_key = self._node_keys() if self.retained is not None else None
        self.retained_hits = 0  # nodes taken computed from an earlier job's blocks
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
        return reservation_blocks(it.prompt_len - self.B * len(self._path(i)), it.req.max_tokens, self.B)

    def _tail(self, i):
        """Blocks for the prompt after the path alone (for a preempted request, its prompt and the outputs it feeds
        back): what grow and preempt take at admission, the decode blocks coming through _cover as decode reaches
        them."""
        target = self.preempted[i].target if i in self.preempted else self.items[i].prompt_len
        held = self.B * (len(self._path(i)) + (len(self.parked[i][0]) if i in self.parked else 0))  # path, parked
        return reservation_blocks(target - held, 1, self.B) if target > held else 0

    def _watermark(self):
        """preempt mode: the blocks admission leaves free, watermark_steps of decode growth for every live request
        (a request crosses a block boundary every self.B steps) plus one; nothing when nothing is live."""
        return -(-len(self.admitted) * self.watermark_steps // self.B) + 1 if self.admitted else 0

    def _new_nodes(self, i):
        """Path nodes admitting i allocates: FREE ones, less those an earlier job's kept block fills (_takes)."""
        return sum(1 for nid in self._path(i) if self.node_state[nid] == FREE) - len(self._takes(i))

    def _need(self, i):
        """Blocks request i needs for its whole life beyond what is held: path nodes nobody has allocated, plus its
        private blocks. Before anything is admitted this equals reservation_blocks(prompt_len, max_tokens); the
        preflight (_check, _overflow) validates every mode against it, since a request must finish in the pool
        whatever mode holds its blocks."""
        return self._new_nodes(i) + self._private(i)

    def _need_now(self, i):
        """Blocks admitting request i takes from the free list: _need under reserve; the new path nodes and the
        prompt tail under grow and preempt, the decode blocks coming through _cover."""
        return self._need(i) if self.memory == "reserve" else self._new_nodes(i) + self._tail(i)

    def _fits(self, i):
        """Can request i be admitted now: its blocks are free, and under grow the live set with it never exceeds the
        pool at any future step (its own demand projected flat, over its whole life, since it has not started)."""
        if self._need_now(i) + (self._watermark() if self.memory == "preempt" else 0) > self.kv.free:
            return False
        if self.slots is not None and not self.slots:
            return False  # every region is live or cooling
        if self.timeline is None:
            return True
        kept = len(self.retained) if self.retained else 0  # held by no one in this job, released on demand
        return self.timeline.peak() + self._private(i) + self.trie_blocks + self._new_nodes(i) + kept <= self.kv.capacity

    def _cover(self, s, upto):
        """Blocks for positions [0, upto): already held under reserve; appended from the free list as decode reaches
        them under grow and preempt. False when the pool is dry, which only preempt allows (grow's timeline keeps
        every future step within the pool)."""
        need = -(-upto // self.B) - len(s.block_ids)
        if need <= 0:
            return True
        if need > self.kv.free:
            assert self.memory == "preempt", "grow's timeline keeps the pool from running dry"
            if not self._make_room(s, need):
                return False
        new = self.kv.reserve(need)
        s.block_ids += new
        s.private_blocks += new
        return True

    def _group_need(self, members):
        seen, need = set(), 0
        for i in members:
            for nid in self._path(i):
                if self.node_state[nid] == FREE and nid not in seen:
                    seen.add(nid)
                    need += 1
            need += self._private(i)
        return need

    def _make_room(self, s, need):
        """preempt mode: frees blocks until `need` are, first by releasing pinned nodes (shared blocks held for
        requests not yet admitted, not on s's path, deepest first), then by preempting the youngest request, s
        itself when it is the youngest (False then: s has been preempted and gets no row)."""
        keep, keys = set(s.path), self._keep_keys(s.idx)
        while self.kv.free < need:
            if self._evict_retained(keys):
                continue
            if self._evict_parked():
                continue
            if self.pinned and self._evict_pinned(keep):
                continue
            victim = next(reversed(self.admitted.values()))  # the youngest, s itself included: every row built so
            self._preempt(victim)  # far in this step is older than it, so no built row is ever invalidated
            if victim is s:
                return False
        return True

    def _evict_pinned(self, keep=()):
        """Releases one pinned node, the deepest whose children are all free (so a node is never released above a
        live child, and an owner's later nodes stay its own), none on `keep`. Its users are still to come: the next
        one admitted allocates and owns it again. True when one was released."""
        best = None
        for nid in self.pinned:
            if nid in keep:
                continue
            node = self.trie.nodes[nid]
            if any(self.node_state[c.id] != FREE for c in node.children.values()):
                continue
            if best is None or node.depth > self.trie.nodes[best].depth:
                best = nid
        if best is None:
            return False
        self.kv.release([self.node_block[best]])
        if self.node_state[best] == COMPUTED:
            self.shared_written -= self.B
        self.node_state[best], self.node_block[best], self.node_owner[best] = FREE, None, None
        self.trie_blocks -= 1
        self.pinned_blocks -= 1
        self.pinned.discard(best)
        return True

    def _preempt(self, s):
        """Takes s out of the live set: its private blocks go back, its path nodes stay allocated (a computed node
        stays readable by everyone; a pending one it owned is rewritten when s resumes; a node left with no admitted
        user is pinned, and may be evicted), and s goes to the head of the queue with the prompt and the outputs it
        had produced as its next feed, to be admitted again like any request."""
        self._mark(s)  # its hit nodes are not its writes: only its own pending nodes' positions come off below
        del self.admitted[s.idx]
        feed_len = len(s.prompt_ids) + (len(s.outputs) if s.inflight else len(s.outputs) - 1)
        lo, hi = self.B * len(s.path), min(s.computed, max(feed_len, 0))
        if s.private_blocks and hi > lo:  # its written positions stay in their blocks until the pool needs them
            self.parked[s.idx] = (list(s.private_blocks), lo, hi)
        else:
            self.kv.release(s.private_blocks)
        if self.slots is not None:
            (self.cooling if self.overlap else self.slots).append(s.slot)
        for nid in s.path:
            self.node_admitted[nid] -= 1
            if self.node_admitted[nid] == 0:
                self.pinned_blocks += 1
                self.pinned.add(nid)
        top = self.B * len(s.path)
        self.shared_written -= max(0, min(s.computed, top) - self.B * s.marked)  # its pending nodes' positions
        s.high = min(s.computed, s.target)  # positions it had: re-hitting them on resume is not a recompute
        # the last output is not written yet: it is the next decode row's query; under overlap a sample in flight is
        # not an output yet, and lands on this record at its commit (then feed == prompt + outputs[:-1] again)
        feed = s.prompt_ids + (s.outputs if s.inflight else s.outputs[:-1])
        self.recomputed_tokens += min(s.computed, s.target) + len(feed) - s.target
        self.queued_prompt_tokens += len(feed)
        s.feed, s.target = feed, len(feed)
        s.block_ids, s.private_blocks = [], []
        s.computed = s.marked = 0  # hits stay until it is admitted again: a sequence that ends preempted keeps its record
        s.tail_copied = False  # its new blocks may take a copy again
        self.preempted[s.idx] = s
        self.queue.appendleft(s.idx)
        self.preemptions += 1

    def _node_keys(self):
        """One key per trie node for cache=radix: a digest of its parent's key and its block's tokens, so two jobs'
        nodes with the same prefix of tokens have the same key (the way vLLM hashes prefix-cache blocks)."""
        from array import array
        from hashlib import blake2b
        keys, stack = [None] * len(self.trie.nodes), [(self.trie.root, b"")]
        while stack:
            node, key = stack.pop()
            for blk, child in node.children.items():
                keys[child.id] = blake2b(key + array("q", blk).tobytes(), digest_size=16).digest()
                stack.append((child, keys[child.id]))
        return keys

    def _takes(self, i):
        """cache=radix: the path nodes admitting i fills from kept blocks: the FREE ones whose key is kept, down to
        the first FREE node whose key is not (below it i computes its own nodes, and a kept block there would be
        rewritten by its chunk)."""
        out = []
        if self.retained is not None:
            for nid in self._path(i):
                if self.node_state[nid] == FREE:
                    if self.node_key[nid] not in self.retained:
                        break
                    out.append(nid)
        return out

    def _keep_keys(self, i):
        """cache=radix: the kept keys on i's path, which evicting cannot help admit it."""
        return {self.node_key[nid] for nid in self._path(i)} if self.retained else ()

    def _evict_parked(self, protect=None):
        """preempt mode: releases one parked block, the oldest parking's last block (its surviving prefix stays
        usable), never protect's; False when none."""
        for idx, (blocks, lo, hi) in self.parked.items():
            if idx == protect:
                continue
            self.kv.release([blocks.pop()])
            if not blocks:
                del self.parked[idx]
            return True
        return False

    def _evict_retained(self, keep=()):
        """cache=radix: releases the oldest block kept from earlier jobs whose key is not in keep; False when none."""
        if self.retained:
            for key in self.retained:
                if key not in keep:
                    self.kv.release([self.retained.pop(key)])
                    return True
        return False

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
            if not self.policy.chunk_prefill and self.items[i].prompt_len > self.limits.max_prefill:
                raise ValueError(f"request {self.items[i].req.id} has {self.items[i].prompt_len} prompt tokens; without "
                                 f"chunked prefill a prompt must fit one step ({self.limits.max_prefill})")
        if self.memory == "preempt":
            return  # pinned nodes are evictable, so nothing held for a later request can block the head
        if self.groups is None:
            order = self.policy.admission_order
            over = self._overflow([[i] for i in order])  # without sharing nothing is held: _need covered it
            if over is not None:
                k, need, held = over
                raise ValueError(f"request {self.items[order[k]].req.id} needs {need} KV blocks while {held} are held "
                                 f"for prefixes shared with requests later in the order; the pool has {cap} "
                                 f"(order=prefix_dfs keeps them adjacent, sharing=none holds none, or enlarge "
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
            if need + held > self.kv.capacity or len(members) > self.limits.max_seqs:
                return k, need, held
        return None

    @property
    def done(self):
        return not self.queue and not self.admitted

    def occupancy(self):
        """This moment's state, for the per-step trace (metrics.STEP_FIELDS)."""
        return {"admitted": len(self.admitted), "free_blocks": self.kv.free, "trie_blocks": self.trie_blocks,
                "pinned_blocks": self.pinned_blocks, "step_prefill_budget": self.step_budget, **self.kv_at_step,
                "mapped_groups": self.kv.mapped_groups}

    def counters(self):
        """Totals over the run so far, for the engine's counts."""
        return {"head_blocked_steps": self.head_blocked_steps, "prefix_hit_tokens": self.prefix_hit_tokens,
                "prefix_wait_steps": self.prefix_wait_steps, "preemptions": self.preemptions,
                "recomputed_tokens": self.recomputed_tokens, "retained_hits": self.retained_hits,
                "retained_blocks": len(self.retained) if self.retained is not None else 0,
                "discarded_rows": self.discarded, "parked_tokens": self.parked_tokens}

    # admission ------------------------------------------------------------------------------------

    def _admit(self, i, group):
        it, path = self.items[i], list(self._path(i))
        taken = set(self._takes(i))  # cache=radix: nodes an earlier job computed and kept start computed, held for i
        for nid in taken:
            self.node_state[nid], self.node_block[nid] = COMPUTED, self.retained.pop(self.node_key[nid])
            self.trie_blocks += 1
            self.shared_written += self.B
            self.retained_hits += 1
        new = [nid for nid in path if self.node_state[nid] == FREE]
        need = self._need_now(i)  # with its parked blocks counted as held, as _fits counted them
        parked, lo, hi = self.parked.pop(i, (None, 0, 0))
        blocks = self.kv.reserve(need)
        for nid, b in zip(new, blocks):
            self.node_state[nid], self.node_block[nid], self.node_owner[nid] = PENDING, b, i
        self.trie_blocks += len(new)
        placed = set(new) | taken  # put in place by this admission, so never pinned before it
        for nid in path:
            if nid not in placed and self.node_admitted[nid] == 0:
                self.pinned_blocks -= 1  # allocated earlier, held for this request; now in use again
                self.pinned.discard(nid)
            self.node_admitted[nid] += 1
        resumed = self.preempted.pop(i, None)
        if resumed is None:
            s = Sequence(idx=i, prompt_ids=it.token_ids, max_tokens=it.req.max_tokens, ignore_eos=it.req.ignore_eos,
                         block_ids=[self.node_block[nid] for nid in path] + blocks[len(new):],
                         private_blocks=blocks[len(new):], path=path, group=group, admitted_at=time.perf_counter(),
                         admitted_step=self.planned, feed=it.token_ids, target=it.prompt_len)
            self.admission_log.append(i)
        else:  # the same request, admitted again: its feed, outputs and timestamps kept; its parked blocks back, then new
            s = resumed
            s.private_blocks = (parked or []) + blocks[len(new):]
            s.block_ids = [self.node_block[nid] for nid in path] + s.private_blocks
            s.hits = 0  # counted again from the start of its feed
            if parked:
                hi = min(hi, lo + self.B * len(parked))  # what survived eviction, from the front
                s.parked = (lo, hi)
                self.recomputed_tokens -= hi - lo  # booked at its preemption; these positions are not fed again
                self.parked_tokens += hi - lo
        if self.timeline is not None:
            self.timeline.flat += self._private(i)  # until its first token: its whole life, flat
        if self.slots is not None:
            s.slot = self.slots.pop()
            self.slot_resets.append(s.slot)
        self._advance(s)
        self.admitted[i] = s
        self.queued_prompt_tokens -= s.target

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
                while need > self.kv.free and self._evict_retained():  # blocks kept from earlier jobs go first
                    pass
                if need > self.kv.free:
                    raise self._stuck(f"group {gi}", need)
                for i in g.members:
                    assert self.queue.popleft() == i
                    self._admit(i, gi)
            return
        while self.queue:
            head = self.queue[0]
            keep = self._keep_keys(head)  # kept blocks go first, but never the head's own: that frees nothing it needs
            while not self._fits(head) and (self._evict_retained(keep) or self._evict_parked(head)):
                pass
            if self.memory == "preempt":  # shared blocks held for requests further down make way for the head
                keep = set(self._path(head))
                while not self._fits(head) and self.pinned and self._evict_pinned(keep):
                    pass
            if len(self.admitted) >= self.limits.max_seqs or not self._fits(head):
                cooling = self.slots is not None and not self.slots  # every region freed is cooling: next commit
                if not self.admitted and not cooling:
                    raise self._stuck(self.items[head].req.id, self._need(head))
                self.head_blocked_steps += 1
                return
            self._admit(self.queue.popleft(), None)

    # composition ----------------------------------------------------------------------------------

    def _advance(self, s, covered_now=()):
        """Move s.computed over path nodes other sequences have computed (prefix hits), covered_now among them:
        the nodes an earlier row of the step being built completes. Returns False when the next node is pending
        under another owner, so s takes no prefill chunk this step. Once the next node is s's own, every later
        node on the path is too: an earlier owner of a later node would have allocated this one as well."""
        while s.computed < self.B * len(s.path):
            nid = s.path[s.computed // self.B]
            assert self.node_state[nid] != FREE, "a path node of an admitted sequence is always allocated"
            if self.node_state[nid] != COMPUTED and nid not in covered_now:  # pending: s's own, or a wait
                return self.node_owner[nid] == s.idx
            assert s.computed % self.B == 0
            rehit = min(self.B, max(0, s.high - s.computed))  # had before a preemption: not a recompute, not a new hit
            s.computed += self.B
            s.hits += self.B
            s.hit_share += self.B - rehit
            self.prefix_hit_tokens += self.B - rehit
            self.recomputed_tokens -= rehit
        if s.parked and s.computed == s.parked[0]:  # its parked blocks hold the positions from here: no recompute
            s.computed, s.parked = s.parked[1], None
        return True

    def _tail_copy(self, s, copies, marked_now, planned_now):
        """sharing=token, once, when s stands at the start of the first block it writes itself: the leading tokens
        that block shares with a sibling block computed in an earlier step are copied from it rather than
        recomputed. Sources: another child of the same trie node that is COMPUTED (marked at a commit, so written
        before this step), or, for a private tail, another live request ending at the same node whose tail is
        written that far (its computed count moves at commit too). Never the block's last feed token, never a whole
        block. Books the copy, the hits, and the positions written inside an own node."""
        B, d = self.B, s.computed // self.B
        if s.tail_copied or s.computed % B or s.computed >= s.target:
            return
        s.tail_copied = True
        if d > len(s.path):  # past the prompt's trie blocks (a resumed request's own outputs, parked blocks restored
            return  # up to there): no trie node or live request shares them
        lo, hi = d * B, min((d + 1) * B, s.target)
        tokens, cap = s.feed[lo:hi], hi - lo - 1  # the block's last token stays s's own
        parent = self.trie.nodes[s.path[d - 1]] if d else self.trie.root
        best, src = 0, None
        for key, child in parent.children.items():  # never a node this step's own chunks write (marked_now, overlap)
            if self.node_state[child.id] == COMPUTED and self.node_owner[child.id] != s.idx and child.id not in marked_now:
                k = self._common(key, tokens, cap)
                if k > best:
                    best, src = k, self.node_block[child.id]
        if d == len(s.path):  # a private tail: live requests ending at the same node, their tails written past k
            for j in self.trie.ends.get(parent.id, ()):
                o = self.admitted.get(j)
                if o is None or o is s or len(o.path) != d:
                    continue
                written = o.computed - planned_now.get(j, 0)  # under overlap computed counts this step's chunk too
                k = self._common(o.feed[lo:lo + B], tokens, min(cap, written - lo))
                if k > best:
                    best, src = k, o.block_ids[d]
        if best > 0:
            copies.append((src, s.block_ids[d], best))
            rehit = min(best, max(0, s.high - s.computed))
            s.computed += best
            s.hits += best
            s.hit_share += best - rehit
            self.prefix_hit_tokens += best - rehit
            self.recomputed_tokens -= rehit
            if d < len(s.path):
                self.shared_written += best  # positions written inside s's own node, like a chunk's

    @staticmethod
    def _common(a, b, cap):
        k = 0
        while k < cap and k < len(a) and k < len(b) and a[k] == b[k]:
            k += 1
        return k

    def _step_budget(self, unprefilled, decoding):
        """Rule 2's budget for the step being built: the policy's, or under ridge ridge_budget over the prefill still
        to compute (the queue's prompts plus the admitted ones' unprefilled tokens; prefix hits not yet taken count
        as prefill, which only over-estimates), the longest decode chain still to run and the step's decode rows."""
        if self.ridge is None:
            return self.budget
        live = self.admitted.values()
        # decode forwards left: the first output comes from the prefill forward, each later one from a decode row
        chain = max([self.chain_after[len(self.admission_log)]] + [s.max_tokens - max(len(s.outputs), 1) for s in live])
        return ridge_budget(self.ridge, decoding, self.queued_prompt_tokens + unprefilled, chain,
                            self.limits.graph_tokens, self.limits.max_prefill)

    def next_step(self):
        """The step to run next: every decoding sequence's row (its growth first, as vLLM schedules its running
        requests before its waiting ones: an admission never takes the block a live row needs this step), then
        admission, then prefill chunks up to the budget; None under overlap when nothing can be planned until the
        step in flight commits."""
        if self.slots is not None and self.cooling and self.planned == self.commits:
            self.slots += self.cooling  # nothing in flight: every reader of the freed regions has settled
            self.cooling = []
        live = set(self.admitted)
        rows = self._decode_rows()
        self.admit()
        rows += self._decode_rows(skip=live)  # a request re-admitted with its parked positions whole decodes at once
        # admitted prompt positions not yet computed; computed runs past target as a sequence decodes
        unprefilled = sum(max(0, s.target - s.computed) for s in self.admitted.values())
        decoding = sum(1 for s in self.admitted.values() if s.computed >= s.target)
        budget = self.step_budget = self._step_budget(unprefilled, decoding)
        # KV while this step runs: blocks held (a request's whole life under reserve; what decode has reached under
        # grow and preempt, this step's new blocks included: the decode rows above took them, prefill chunks never
        # do) against positions written before it (private ones here, shared ones in shared_written; under overlap
        # the decode rows above already count as written, so they come off), and the admitted prompt tokens not yet
        # prefilled, which is most of what is held but unwritten while prompts wait for their chunks. Taken here,
        # after any preemption above and before the chunks below. free_blocks in occupancy() is after commit.
        private = sum(max(0, s.computed - self.B * len(s.path)) for s in self.admitted.values())
        private += sum(min(hi, lo + self.B * len(blocks)) - lo for blocks, lo, hi in self.parked.values())  # parked KV
        self.kv_at_step = {"kv_reserved_tokens": self.B * (self.kv.capacity - self.kv.free),
                           "kv_written_tokens": self.shared_written + private - (len(rows) if self.overlap else 0),
                           "unprefilled_tokens": unprefilled}
        copies, marked_now = self._prefill_rows(rows, budget)
        if not rows and self.overlap:
            return None  # every live sequence has its last token in flight: nothing to plan until it commits
        return self._step(rows, copies, marked_now)

    def _decode_rows(self, skip=()):
        """One row per decoding sequence not in skip: its last sampled id as the query at position computed (under
        overlap, the id still in flight, gathered from the previous step's output: Row.prev); a new block at a
        block boundary under grow and preempt (rows share s.block_ids, so it is taken before the row is built)."""
        rows = []
        for s in list(self.admitted.values()):
            if s.idx in skip or s.idx not in self.admitted or s.computed < s.target:
                continue  # already built; preempted while an older row made room; or still prefilling
            if len(s.outputs) + s.inflight >= s.max_tokens:
                continue  # overlap: its last token is in flight; the commit finishes it
            if not self._cover(s, s.computed + 1):
                continue  # s was the youngest and made way itself: it is back in the queue
            prev = self.prev_slots.get(s.idx) if self.overlap else None
            rows.append(Row(s.idx, s.computed, [0 if prev is not None else s.outputs[-1]], s.block_ids, sample=True,
                            decode=True, prev=prev, slot=s.slot))
            if self.overlap:
                s.computed += 1
        return rows

    def _prefill_rows(self, rows, budget):
        """Prefill chunks in admission order up to the budget, appended to rows: each sequence first moves over the
        path nodes others computed (_advance), takes a tail copy (sharing=token), then feeds what the budget leaves.
        Returns the step's copies and, under overlap, the nodes its own chunks complete (computed for later steps,
        not for this one)."""
        prefill, covered_now, copies = 0, set(), []  # covered_now: nodes the chunks built so far complete this step
        marked_now, planned_now = set(), {}  # overlap: nodes marked by this step's chunks; the chunk each sequence got
        for s in self.admitted.values():
            if s.computed >= s.target:
                continue
            if prefill >= budget:
                break  # budget spent: the sequences behind neither get a chunk nor count as waiting
            if not self._advance(s, covered_now):
                self.prefix_wait_steps += 1
                continue
            if self.token_sharing:
                self._tail_copy(s, copies, marked_now, planned_now)
            n = self._chunk(s, budget - prefill, prefill)
            if n <= 0:
                break
            covered = self._cover(s, s.computed + n)  # within the feed's blocks, taken at admission: never new
            assert covered, "a prefill chunk writes blocks taken at admission"
            rows.append(Row(s.idx, s.computed, s.feed[s.computed:s.computed + n], s.block_ids,
                            sample=s.computed + n == s.target and not s.outputs, decode=False, slot=s.slot))
            for k in range(s.computed // self.B, min(len(s.path), (s.computed + n) // self.B)):
                if self.node_owner[s.path[k]] == s.idx:
                    covered_now.add(s.path[k])  # whole by the end of this chunk: readable by the rows after it
            if s.prefill_start_step is None:
                s.prefill_start_step = self.planned
            if self.overlap:
                self._written(rows[-1], s)
                s.computed += n
                planned_now[s.idx] = n
                marked_now.update(self._mark(s))
            prefill += n
        return copies, marked_now

    def _chunk(self, s, left, prefill):
        """The tokens s feeds this step with `left` of the budget unspent and `prefill` already taken: what remains of
        its feed, chunked to the budget (chunk_prefill, and a resumed feed regardless, since it may exceed the step
        cap); unchunked, the whole remainder when it fits or when the step has no prefill yet, else nothing."""
        end = s.parked[0] if s.parked and s.computed < s.parked[0] else s.target  # up to its parked positions
        remaining = end - s.computed
        if self.policy.chunk_prefill or s.outputs:
            return min(remaining, left)
        return remaining if remaining <= left or prefill == 0 else 0

    def _step(self, rows, copies, marked_now):
        """The Step over these rows, planned: the regions reset this step, the cascade groups, the in-flight counts."""
        assert rows, "an empty step would never make progress"
        step = Step(rows, copies)
        step.slot_resets, self.slot_resets = self.slot_resets, []
        if self.limits.attention == "cascade":  # rows behind a prefix computed before this step read it once
            step.groups = cascade.groups(rows, self._path, self.node_state, self.node_block, self.B, COMPUTED, marked_now)
        limit = self.limits.max_prefill + self.limits.max_seqs  # the widest step the executor probed for
        assert step.tokens <= limit, f"step of {step.tokens} tokens over the step limit {limit}"
        self.prev_slots = {}
        for j, row in enumerate(r for r in rows if r.sample):
            self.admitted[row.seq].inflight += 1
            if self.overlap:
                self.prev_slots[row.seq] = j
        self.planned += 1
        return step

    def _written(self, row, s):
        """The shared positions a row writes: inside s's own pending nodes (rule 3), the part of the row below the
        path's top; counted when the row is planned under overlap, at its commit otherwise."""
        top = self.B * len(s.path)
        self.shared_written += max(0, min(row.start + len(row.token_ids), top) - row.start)

    def _release_path(self, s, admitted=False):
        """A finished sequence leaves its path: its nodes' remaining users drop, and a node whose last user it was
        goes back to the pool (or is kept, cache=radix); admitted: it counted among the node's admitted users (a
        sequence that ends while preempted does not). Leaf first: a kept chain is evicted from its leaf, never from
        its root."""
        for nid in reversed(s.path):
            if admitted:
                self.node_admitted[nid] -= 1
            self.node_remaining[nid] -= 1
            if self.node_remaining[nid] == 0:
                if self.node_state[nid] == FREE:  # evicted (pinned) while s, its last user, waited preempted: gone already
                    assert not admitted, "an admitted sequence's path node is never pinned, so never evicted"
                    continue
                computed = self.node_state[nid] == COMPUTED
                assert computed or not admitted, "an owner finishes only after computing its nodes"
                if not admitted:  # the node was pinned: held for users none of which was admitted
                    self.pinned_blocks -= 1
                    self.pinned.discard(nid)
                if computed:
                    self.shared_written -= self.B
                if computed and self.retained is not None and self.node_key[nid] not in self.retained:
                    self.retained[self.node_key[nid]] = self.node_block[nid]  # kept, the newest last
                else:  # or pending: it owned the node and never came back, nobody else needs it
                    self.kv.release([self.node_block[nid]])
                self.node_state[nid], self.node_block[nid], self.node_owner[nid] = FREE, None, None
                self.trie_blocks -= 1
            elif admitted and self.node_admitted[nid] == 0:
                self.pinned_blocks += 1
                self.pinned.add(nid)

    def _mark(self, s):
        """Owned path nodes that s.computed now covers become computed, readable by later steps; returns them."""
        done, marked = min(len(s.path), s.computed // self.B), []
        for k in range(s.marked, done):
            nid = s.path[k]
            if self.node_owner[nid] == s.idx:
                self.node_state[nid] = COMPUTED
                marked.append(nid)
        s.marked = done
        return marked

    def commit(self, step, sampled):
        """Apply one step's sampled ids (one per sampling row, in row order). Returns the sequences that
        finished, their KV blocks already released."""
        assert len(sampled) == sum(r.sample for r in step.rows), "one sampled id per sampling row"
        if self.slots is not None and self.cooling:  # freed at the commit before: this step was their last reader
            self.slots += self.cooling
            self.cooling = []
        it, finished = iter(sampled), []
        for row in step.rows:
            s = self._landing(row)
            if s is None:  # overlap: planned for a sequence an earlier commit had ended
                if row.sample:
                    next(it)
                    self.discarded += 1
                continue
            if not self.overlap:
                assert row.start == s.computed, "rows must tile each sequence with no gap or overlap"
                self._written(row, s)
                s.computed += len(row.token_ids)
                self._mark(s)
            if not row.sample:
                continue
            t = next(it)
            s.inflight -= 1
            s.outputs.append(t)
            if len(s.outputs) == 1:
                s.first_token_step = self.commits
                if s.max_tokens == self.longest:
                    self.chain_start_step = self.commits
            if t in self.stop and not s.ignore_eos:
                s.finish_reason = "stop"
            elif len(s.outputs) == s.max_tokens:
                s.finish_reason = "length"
            else:
                continue
            s.finished_step = self.commits
            if row.seq in self.preempted:  # overlap: it ended while waiting to be admitted again
                self._end_preempted(s)
            else:
                finished.append(s)
        if self.timeline is not None:
            self.timeline.advance()  # the projection now stands at the next step
            for row in step.rows:
                s = self.admitted.get(row.seq)
                if s is None:
                    continue
                if row.sample and len(s.outputs) == 1:  # its finish step is known now: the staircase replaces the flat
                    self.timeline.flat -= self._private(s.idx)
                    if not s.finish_reason:
                        self.timeline.stair(s, +1, age=0)
            for s in finished:
                if len(s.outputs) > 1:  # a decoding sequence that ended early (EOS): the rest of its staircase goes
                    self.timeline.stair(s, -1, age=self.commits - s.first_token_step)
        for s in finished:
            del self.admitted[s.idx]
            self.kv.release(s.private_blocks)
            if self.slots is not None:
                (self.cooling if self.overlap else self.slots).append(s.slot)
            self._release_path(s, admitted=True)
        self.commits += 1
        done, self.finished_preempted = self.finished_preempted, []
        return finished + done

    def _landing(self, row):
        """The sequence a row's result lands on: admitted, or (overlap) preempted after it was planned with its
        sample still open; None when an earlier commit had ended it, so its sample is discarded."""
        s = self.admitted.get(row.seq)
        if s is None:
            s = self.preempted.get(row.seq)
            if s is None or len(s.outputs) >= s.max_tokens or s.finish_reason:
                return None
        return s

    def _end_preempted(self, s):
        """A sequence whose in-flight sample ended it while it waited to be admitted again: out of the queue, the
        recompute booked at its preemption never runs, its path released as a sequence that was not admitted."""
        del self.preempted[s.idx]
        self.queue.remove(s.idx)
        if s.idx in self.parked:
            self.kv.release(self.parked.pop(s.idx)[0])
        self.queued_prompt_tokens -= s.target
        self.recomputed_tokens -= s.target
        self._release_path(s)
        self.finished_preempted.append(s)
