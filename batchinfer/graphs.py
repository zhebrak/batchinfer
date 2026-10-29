"""Graph buckets, and the layout of one step inside a captured CUDA graph (executor.py captures and replays them).

A graph bakes its shapes: a row count (batch) and a token count (total_tokens). Two kinds are captured:
- decode-only graphs, keyed (rows, 0): rows rows of one token each (executor.graph_row_sizes picks the sizes). A
  decode-only step replays the smallest at or above its row count, pad rows after its own.
- mixed graphs, keyed (rows, tokens), for steps that carry prefill chunks: MIXED_ROWS x MIXED_TOKENS up to the
  largest decode-only size. rows + 1 rows and rows + tokens tokens: the step's decode rows, then its prefill rows,
  then pad decode rows up to rows, then one pad row holding whatever tokens are left, so the total is constant. Only
  FlashAttention-3 captures them: its tile schedule is computed per step outside the graph (executor), so a row's
  query length is not baked; FlashAttention-2's grid is, so under it a mixed step runs eagerly (vLLM's rule too).
Pad rows carry token 0 at position 0, write slot 0 (block 0, the pad block, which no real row reads; several pad
tokens writing it is benign) and attend over nothing (seqused_k 0: FlashAttention writes zeros). Sampling reads the
logits at `rows` sample slots (a step never samples more rows than it has), the unused ones pointing at the last
token, a pad token in a mixed graph. Steps past the largest bucket run eagerly: more rows than any graph, or more
prefill tokens than MIXED_TOKENS[-1], where the step is compute-bound and the launch overhead is small.
"""
from collections import namedtuple

import numpy as np
import torch

MIXED_ROWS = tuple(range(32, 513, 32))
MIXED_TOKENS = (64, 128, 256, 512, 1024, 2048)
# FlashAttention-3 under a graph: its tile schedule (get_scheduler_metadata) is computed per step outside the graph
# into a persistent buffer the captured call reads, and the KV split count is fixed, since the kernel's own heuristic
# would pick one per step from the real lengths. NUM_SPLITS is vLLM's flash_attn_max_num_splits_for_cuda_graph
# (vllm/config/attention.py), applied as vLLM applies it, to graphs of at most SPLIT_TOKENS tokens (its largest full
# graph): the split accumulator is [splits, heads, tokens, head_dim] fp32 in the graph's pool, 1.25 GiB for 32 splits
# over 2,560 tokens. Neither number is measured here.
NUM_SPLITS, SPLIT_TOKENS = 32, 512


def splits(key):
    """The KV split count baked into the graph with this key: NUM_SPLITS up to SPLIT_TOKENS tokens, else none."""
    return NUM_SPLITS if shape(key)[1] <= SPLIT_TOKENS else 1


def sched_entries(batch):
    """The int32 entries the scheduler metadata of a graph with this batch can take, sized as vLLM sizes its buffer
    (1 + round_up(batch, 4) * 4: the tile counter, then the varlen preparation, the dynamic splits, the batch sort
    and the head swizzle, one entry per row each, padded to 4); the call returns how many it used (executor)."""
    return 1 + -(-batch // 4) * 4 * 4


def keys(decode_sizes, mixed):
    """Every graph to capture, decode-only sizes first: (rows, 0) per size, then with mixed the (rows, tokens) keys
    up to the largest decode-only size."""
    out = [(r, 0) for r in decode_sizes]
    if mixed and decode_sizes:
        out += [(r, t) for r in MIXED_ROWS if r <= decode_sizes[-1] for t in MIXED_TOKENS]
    return tuple(out)


def shape(key):
    """(batch, total_tokens, sample slots) the graph with this key bakes."""
    rows, tokens = key
    return (rows, rows, rows) if tokens == 0 else (rows + 1, rows + tokens, rows)


def key_for(step, keys_):
    """The graph a step replays: the one with the fewest rows, then the fewest tokens, that holds it; None when none
    does (eager). A decode-only step takes a decode-only graph, a step with prefill rows a mixed one."""
    n, mixed = len(step.rows), any(not r.decode for r in step.rows)
    pre = sum(len(r.token_ids) for r in step.rows if not r.decode)
    best = None
    for k in keys_:
        if k[0] >= n and (k[1] > 0) == mixed and k[1] >= pre and (best is None or k < best):
            best = k
    return best


Laid = namedtuple("Laid", "batch total width sampling max_q max_k")  # what layout wrote: the counts of the step laid out


def exact_shape(step):
    """(batch, total_tokens, sample slots) of a step laid out as it is, no padding: the eager forward's inputs."""
    return len(step.rows), sum(len(r.token_ids) for r in step.rows), sum(1 for r in step.rows if r.sample)


def layout(step, key, block_size, out, cascade=False, region=None):
    """Writes the step into out, laid out for the graph with this key, and returns a Laid: the graph's batch and
    total_tokens, the table width, the sampling rows, and the longest query and the longest context of any row as
    laid out (at least 1 each: FlashAttention-3's schedule is computed from them per step). out: the graph's input
    arrays (numpy views of the executor's pinned buffers, or plain arrays in tests), each at least the key's size:
    tokens, positions, slots and prev (overlap: the slot of the previous step's output a row's token is gathered
    from, -1 for a token given), used, cu, sample, and table, a flat array read as [batch, width] (width: the blocks
    the longest real row reaches, so the copy is as wide as the step, not the pool). cascade: also the two-call
    arrays (cascade.plan) padded to the graph: perm and inv over every token, the pad tokens as one last group over no
    keys, then empty groups up to groups_of(key); used_s and table_s per row, the pad rows over no keys. The counts a
    graph bakes come from here alone (shape); nothing in a captured graph varies with the step but the contents of
    these arrays."""
    if key is None:  # exact: the step as it is, no pad row, no pad token (the eager forward)
        batch, total, slots_n = exact_shape(step)
        rows_b, tokens_b = batch, 0
    else:
        rows_b, tokens_b = key
        batch, total, slots_n = shape(key)
        assert not step.rows or key_for(step, (key,)) == key, "the step does not fit this graph"  # a capture stages no row
    bs, n = block_size, len(step.rows)
    t, width, j, max_q, max_k = 0, 1, 0, 1, 1
    out.cu[0] = 0
    out.prev[:total] = -1  # overlap: a row's token gathered from the previous step's output, by slot
    for i, r in enumerate(step.rows):
        m = len(r.token_ids)
        out.tokens[t:t + m] = r.token_ids
        if r.prev is not None:
            out.prev[t] = r.prev
        if m == 1:  # a decode row: scalar work, cheaper than numpy's per-call overhead over a thousand rows
            out.positions[t] = r.start
            out.slots[t] = r.block_ids[r.start // bs] * bs + r.start % bs
        else:  # a prefill chunk: numpy over its positions (a 16k-token chunk in Python cost milliseconds a step)
            pos = np.arange(r.start, r.start + m)
            out.positions[t:t + m] = pos
            blocks = np.asarray(r.block_ids[r.start // bs:(r.start + m - 1) // bs + 1], dtype=np.int64)
            out.slots[t:t + m] = blocks[pos // bs - r.start // bs] * bs + pos % bs
        t += m
        out.cu[i + 1] = t
        out.used[i] = r.start + m
        width, max_q, max_k = max(width, (r.start + m - 1) // bs + 1), max(max_q, m), max(max_k, r.start + m)
        if r.sample:
            out.sample[j] = t - 1
            j += 1
    pad = total - t  # pad tokens: one per pad row, the rest on the last row of a mixed graph
    out.tokens[t:total] = 0
    out.positions[t:total] = 0
    out.slots[t:total] = 0
    out.used[n:batch] = 0
    for i in range(n, batch):
        m = 1 if i < rows_b else pad - (rows_b - n)
        t += m
        out.cu[i + 1] = t
        max_q = max(max_q, m)
    assert t == total, (t, total)
    out.sample[j:slots_n] = total - 1
    table = out.table[:batch * width].reshape(batch, width)
    table[:] = 0
    for i, r in enumerate(step.rows):
        k = (r.start + len(r.token_ids) - 1) // bs + 1
        table[i, :k] = r.block_ids[:k]
    if cascade:
        _cascade_layout(step, batch, total, width, bs, out, n, t_real=sum(len(r.token_ids) for r in step.rows),
                        exact=key is None)
    if region is not None:  # pool=contiguous: each row's keys start at its region (region: tokens per slot)
        out.cu_k[:n] = [r.slot * region for r in step.rows]
        out.cu_k[n:batch + 1] = 0  # the pad rows read the pad slot, over no keys
    return Laid(batch, total, width, j, max_q, max_k)


def groups_of(key):
    """The group entries a cascade graph with this key bakes: one per row at most, plus the pad tokens' group."""
    return shape(key)[0] + 1


def _cascade_layout(step, batch, total, width, bs, out, n, t_real, exact=False):
    from .cascade import plan
    if step.rows:
        perm, inv, cu_g, used_g, table_g, used_s, table_s, _ = plan(step, bs)
    else:
        perm, inv, cu_g, used_g, table_g, used_s, table_s = [], [], [0], [], [], [], []
    if exact:  # the groups as they are
        g_max = len(used_g)
    else:  # the pad tokens as one last group over no keys, then empty groups up to the graph's count
        g_max = batch + 1
        pads = list(range(t_real, total))
        perm, cu_g, used_g, table_g = perm + pads, cu_g + [total], used_g + [0], table_g + [[]]
    inv = [0] * total
    for j_, t_ in enumerate(perm):
        inv[t_] = j_
    out.perm[:total], out.inv[:total] = perm, inv
    g = len(used_g)
    assert g <= g_max, (g, g_max)
    out.cu_g[:g + 1] = cu_g
    out.cu_g[g + 1:g_max + 1] = total  # empty groups after the pad group
    out.used_g[:g] = used_g
    out.used_g[g:g_max] = 0
    out.used_s[:n] = used_s
    out.used_s[n:batch] = 0
    tg = out.table_g[:g_max * width].reshape(g_max, width)
    tg[:] = 0
    for i, blocks in enumerate(table_g):
        tg[i, :len(blocks)] = blocks
    ts = out.table_s[:batch * width].reshape(batch, width)
    ts[:] = 0
    for i, blocks in enumerate(table_s):
        k = min(len(blocks), width)
        ts[i, :k] = blocks[:k]


class Arrays:
    """The arrays layout fills: numpy views of a Staging's pinned buffers, or fresh arrays at a step's exact sizes."""

    def __init__(self, **arrays):
        self.__dict__.update(arrays)

    @classmethod
    def exact(cls, step, block_size):
        """Arrays at the step's exact sizes; the tables as wide as the blocks the longest row reaches (layout's width)."""
        batch, total, slots = exact_shape(step)
        width = max(((r.start + len(r.token_ids) - 1) // block_size + 1 for r in step.rows), default=1)
        groups = len(step.groups)
        i64, i32 = np.int64, np.int32
        return cls(tokens=np.zeros(total, i64), positions=np.zeros(total, i64), slots=np.zeros(total, i64),
                   prev=np.zeros(total, i64), used=np.zeros(batch, i32), cu=np.zeros(batch + 1, i32),
                   sample=np.zeros(slots, i64), table=np.zeros(batch * width, i32), cu_k=np.zeros(batch + 1, i32),
                   perm=np.zeros(total, i64), inv=np.zeros(total, i64), cu_g=np.zeros(groups + 1, i32),
                   used_g=np.zeros(groups, i32), table_g=np.zeros(groups * width, i32), used_s=np.zeros(batch, i32),
                   table_s=np.zeros(batch * width, i32))


class Inputs:
    """One step's device inputs, the fields the forward and the attention read, made one of two ways: exact tensors
    for an eager step (Inputs.exact), or views of a Staging's buffers at a graph's shape (Inputs.graph), where the
    tokens are gathered inside the captured forward from the previous step's output (Row.prev)."""

    def __init__(self, executor, **fields):
        self.executor, self._input_ids = executor, None  # executor: what the HF attention hook attends through
        self.__dict__.update(fields)

    @property
    def input_ids(self):
        if self._input_ids is not None:
            return self._input_ids
        prev = self.g_prev
        return torch.where(prev >= 0, self.g_out[prev.clamp(min=0)], self.g_tokens).view(1, -1)

    @classmethod
    def exact(cls, step, ex):
        """The step laid out as it is (layout with no key), as fresh device tensors."""
        cascade_on, token = bool(step.groups), ex.pool.layout == "token"
        a = Arrays.exact(step, ex.block_size)
        laid = layout(step, None, ex.block_size, a, cascade=cascade_on, region=ex.pool.max_ctx if token else None)
        batch, total, w = laid.batch, laid.total, laid.width
        assert a.table.shape == (batch * w,), "the exact arrays and the layout agree on the table's width"
        t = lambda x, dt: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=ex.device)  # noqa: E731
        tokens = t(a.tokens, torch.long)
        if (a.prev >= 0).any():  # overlap: a row's token is the previous step's sampled id, still on the device
            prev = t(a.prev, torch.long)
            tokens = torch.where(prev >= 0, ex.g_out[prev.clamp(min=0)], tokens)
        cu, used = a.cu.tolist(), a.used.tolist()
        self = cls(ex, position_ids=t(a.positions, torch.long).view(1, total), slot_mapping=t(a.slots, torch.long),
                   cu_seqlens_q=t(a.cu, torch.int32), seqused_k=t(a.used, torch.int32),
                   block_table=t(a.table, torch.int32).view(batch, w), sample_idx=t(a.sample, torch.long),
                   all_sample=laid.sampling == total, max_seqlen_q=laid.max_q, max_seqlen_k=laid.max_k,
                   scheduler_metadata=None, num_splits=0, groups=None, cu_k=t(a.cu_k, torch.int32) if token else None,
                   rows=[(cu[i], cu[i + 1], used[i], r.block_ids) for i, r in enumerate(step.rows)])
        self._input_ids = tokens.view(1, total)
        if cascade_on:
            g = len(step.groups)
            self.perm, self.inv = t(a.perm, torch.long), t(a.inv, torch.long)
            self.cu_g, self.used_g, self.used_s = t(a.cu_g, torch.int32), t(a.used_g, torch.int32), t(a.used_s, torch.int32)
            self.table_g, self.table_s = t(a.table_g, torch.int32).view(g, w), t(a.table_s, torch.int32).view(batch, w)
            self.max_k_g, self.max_k_s = max(int(a.used_g.max()), 1), max(int(a.used_s.max()), 1)
            self.max_seqlen_q_g = max(int(np.diff(a.cu_g).max()), 1)
            self.scheduler_metadata_g = None
            of = {i: grp for grp in step.groups for i in grp.rows}  # the torch oracle: per row, its prefix and suffix
            self.groups = [(cu[i], cu[i + 1], used[i], r.block_ids, of[i].shared, of[i].blocks) for i, r in enumerate(step.rows)]
        return self

    @classmethod
    def graph(cls, ex, key):
        """Views of the executor's staging buffers at the key's shape. A decode-only graph samples every row; a mixed
        one gathers its sample slots. The FlashAttention-3 schedule is read through the slice the capture's own call
        returned."""
        st, token_layout, cascade_on, fa3 = ex.staging, ex.pool.layout == "token", ex.attention == "cascade", ex.fa_version == 3
        rows_b, tokens_b = key
        batch, total, slots = shape(key)
        big = st.total_max
        self = cls(ex, g_prev=st.g_prev[:total], g_out=st.g_out, g_tokens=st.g_tokens[:total],
                   position_ids=st.g_tokens[big:big + total].view(1, total), slot_mapping=st.g_tokens[2 * big:2 * big + total],
                   cu_seqlens_q=st.g_cu[:batch + 1], seqused_k=st.g_used[:batch], block_table=st.g_table[:batch],
                   cu_k=st.g_cu_k[:batch + 1] if token_layout else None,
                   sample_idx=st.g_sample[:slots], all_sample=tokens_b == 0,  # decode-only: every row, in order
                   # the longest query a row of this graph can carry: one token, or the mixed graph's pad row at most
                   max_seqlen_q=1 if tokens_b == 0 else tokens_b + rows_b, max_seqlen_k=st.k_cap,
                   scheduler_metadata=st.g_sched[:st.sched_n[key]] if fa3 else None, num_splits=splits(key) if fa3 else 0,
                   groups=None, rows=None)
        if cascade_on:  # the two-call arrays, at the graph's group count (groups_of)
            g = groups_of(key)
            self.groups = True
            self.perm, self.inv = st.g_perm[:total], st.g_inv[:total]
            self.cu_g, self.used_g, self.table_g = st.g_cu_g[:g + 1], st.g_used_g[:g], st.g_table_g[:g]
            self.used_s, self.table_s = st.g_used_s[:batch], st.g_table_s[:batch]
            self.max_k_g = self.max_k_s = st.k_cap
            self.max_seqlen_q_g = total  # a group may hold every token
            self.scheduler_metadata_g = st.g_sched_g[:st.sched_g_n[key]] if fa3 else None
        return self


class Staging:
    """The persistent inputs every graph reads, at the largest graph's shape over every key, and their pinned host
    copies in two sets (under overlap the step in flight may still be copying one in): token ids, positions and KV
    slots in one int64 buffer, the gather slots, cu_seqlens_q, seqused_k, the sample slots, a block table as wide
    as the longest context the pool holds (k_cap, the max_seqlen_k baked into every graph, as vLLM bakes
    max_model_len), the cascade's arrays, the regions' key starts, and FlashAttention-3's schedule buffers. stage()
    fills a set from a step through layout and copies in every array the graph reads; a replay reads the same
    addresses it was captured with."""

    def __init__(self, keys, block_size, k_cap, device, g_out, cascade=False, region=None, schedule=None, heads=None):
        """schedule: get_scheduler_metadata (FlashAttention-3), with heads = (num_heads_q, num_heads_kv, head_dim,
        qkv_dtype, page_size or None); region: the regions' tokens per slot under pool=contiguous."""
        self.keys, self.block_size, self.k_cap, self.device, self.g_out = keys, block_size, k_cap, device, g_out
        self.cascade, self.region, self.schedule, self.heads = cascade, region, schedule, heads
        shapes = [shape(k) for k in keys]
        batch, total, rows = (max(s[i] for s in shapes) for i in range(3))
        self.total_max, width = total, -(-k_cap // block_size)
        dev, i32, i64 = device, torch.int32, torch.long
        z = lambda *s, dtype: torch.zeros(s, dtype=dtype, device=dev)  # noqa: E731
        self.g_tokens, self.g_prev = z(3 * total, dtype=i64), z(total, dtype=i64)  # ids | positions | slots
        self.g_used, self.g_cu, self.g_sample = z(batch, dtype=i32), z(batch + 1, dtype=i32), z(rows, dtype=i64)
        self.g_table, self.g_table_in = z(batch, width, dtype=i32), z(batch * width, dtype=i32)  # the step's [R, w] block, packed
        self.g_sched, self.sched_n = z(sched_entries(batch), dtype=i32), {}  # FlashAttention-3's; key -> entries returned
        g = batch + 1  # cascade: the groups' arrays (groups_of) and the per-row suffix arrays
        self.g_perm, self.g_inv = z(total, dtype=i64), z(total, dtype=i64)
        self.g_cu_g, self.g_used_g = z(g + 1, dtype=i32), z(g, dtype=i32)
        self.g_table_g, self.g_table_s = z(g, width, dtype=i32), z(batch, width, dtype=i32)
        self.g_table_g_in, self.g_table_s_in = z(g * width, dtype=i32), z(batch * width, dtype=i32)
        self.g_used_s, self.g_cu_k = z(batch, dtype=i32), z(batch + 1, dtype=i32)
        self.g_sched_g, self.sched_g_n = z(sched_entries(g), dtype=i32), {}
        pin = lambda *s, dtype: torch.zeros((2, *s), dtype=dtype, pin_memory=True)  # noqa: E731
        self.h_tokens, self.h_prev = pin(3 * total, dtype=i64), pin(total, dtype=i64)
        self.h_used, self.h_cu, self.h_sample = pin(batch, dtype=i32), pin(batch + 1, dtype=i32), pin(rows, dtype=i64)
        self.h_table, self.h_cu_k = pin(batch * width, dtype=i32), pin(batch + 1, dtype=i32)
        self.h_perm, self.h_inv = pin(total, dtype=i64), pin(total, dtype=i64)
        self.h_cu_g, self.h_used_g = pin(g + 1, dtype=i32), pin(g, dtype=i32)
        self.h_table_g, self.h_table_s, self.h_used_s = pin(g * width, dtype=i32), pin(batch * width, dtype=i32), pin(batch, dtype=i32)
        self.views = []
        for k in range(2):
            tk = self.h_tokens[k].numpy()
            self.views.append(Arrays(tokens=tk[:total], positions=tk[total:2 * total], slots=tk[2 * total:],
                                     prev=self.h_prev[k].numpy(), used=self.h_used[k].numpy(), cu=self.h_cu[k].numpy(),
                                     sample=self.h_sample[k].numpy(), table=self.h_table[k].numpy(),
                                     perm=self.h_perm[k].numpy(), inv=self.h_inv[k].numpy(), cu_g=self.h_cu_g[k].numpy(),
                                     used_g=self.h_used_g[k].numpy(), table_g=self.h_table_g[k].numpy(),
                                     used_s=self.h_used_s[k].numpy(), table_s=self.h_table_s[k].numpy(),
                                     cu_k=self.h_cu_k[k].numpy()))
        self.h2d = [torch.cuda.Event() for _ in range(2)]  # each set's copies' end: the set is free again after it
        self.parity = 0

    def stage(self, step, key):
        """Fills the next pinned set from the step (once the step that last used it has copied it in), copies every
        array the graph reads in up to the key's shape (the table up to the widest real row; the rest is left as it
        was, past every row's seqused_k), marks the set's event, and computes FlashAttention-3's schedule(s) for the
        step's real lengths where the captured call reads them."""
        k = self.parity = 1 - self.parity
        self.h2d[k].synchronize()
        laid = layout(step, key, self.block_size, self.views[k], cascade=self.cascade, region=self.region)
        batch, width, total = laid.batch, laid.width, laid.total
        copy = lambda g, h, n: g[:n].copy_(h[k, :n], non_blocking=True)  # noqa: E731
        self.g_tokens.copy_(self.h_tokens[k], non_blocking=True)
        copy(self.g_prev, self.h_prev, total)
        copy(self.g_used, self.h_used, batch)
        copy(self.g_cu, self.h_cu, batch + 1)
        copy(self.g_sample, self.h_sample, key[0])
        if self.region is not None:
            copy(self.g_cu_k, self.h_cu_k, batch + 1)
        self._table(self.g_table, self.g_table_in, self.h_table[k], batch, width)
        if self.cascade:
            g = groups_of(key)
            copy(self.g_perm, self.h_perm, total)
            copy(self.g_inv, self.h_inv, total)
            copy(self.g_cu_g, self.h_cu_g, g + 1)
            copy(self.g_used_g, self.h_used_g, g)
            copy(self.g_used_s, self.h_used_s, batch)
            self._table(self.g_table_g, self.g_table_g_in, self.h_table_g[k], g, width)
            self._table(self.g_table_s, self.g_table_s_in, self.h_table_s[k], batch, width)
        self.h2d[k].record()
        if self.schedule is not None:
            if self.cascade:  # the rows' call runs over their suffixes, the groups' call over the groups
                max_k = max(int(self.h_used_s[k, :batch].max()), 1)
                self._schedule(key, self.g_sched, self.sched_n, batch, laid.max_q, max_k, self.g_used_s, self.g_cu, True)
                g = groups_of(key)
                self._schedule(key, self.g_sched_g, self.sched_g_n, g, total, laid.max_k, self.g_used_g, self.g_cu_g, False)
            else:
                self._schedule(key, self.g_sched, self.sched_n, batch, laid.max_q, laid.max_k, self.g_used, self.g_cu, True)

    @staticmethod
    def _table(g, g_in, h, rows, width):
        g_in[:rows * width].copy_(h[:rows * width], non_blocking=True)
        g[:rows, :width].copy_(g_in[:rows * width].view(rows, width))

    def _schedule(self, key, buf, sizes, batch, max_q, max_k, used, cu, causal):
        """FlashAttention-3's tile schedule for this step's real lengths, into the buffer the captured call reads;
        the capture's own call fixes the slice the graph reads, checked every step, the tail zeroed."""
        heads_q, heads_kv, head_dim, dtype, page_size = self.heads
        sched = self.schedule(batch_size=batch, max_seqlen_q=max_q, max_seqlen_k=max_k, num_heads_q=heads_q, num_heads_kv=heads_kv,
                              headdim=head_dim, cache_seqlens=used[:batch], qkv_dtype=dtype, cu_seqlens_q=cu[:batch + 1],
                              causal=causal, page_size=page_size, num_splits=splits(key))
        n = sizes.setdefault(key, sched.numel())
        assert sched.numel() == n <= buf.numel(), \
            f"FlashAttention-3 returned {sched.numel()} schedule entries for {batch} rows; the graph reads {n}"
        buf[:n].copy_(sched)
        buf[n:].zero_()
