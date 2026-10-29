"""batchinfer.graphs: the graph buckets and the layout of a step inside a captured graph. CPU only: the executor
copies what layout writes into its pinned buffers; nothing here needs a GPU."""
import numpy as np
import pytest

from batchinfer import graphs
from batchinfer.executor import graph_row_sizes
from batchinfer.schema import Row, Step

B = 16


class Arrays:
    """Plain arrays at the key's shape, as the executor's pinned views are (the table as wide as 8 blocks)."""

    def __init__(self, key):
        batch, total, rows = graphs.shape(key)
        self.tokens, self.positions, self.slots = (np.full(total, -1, dtype=np.int64) for _ in range(3))
        self.used, self.cu = np.full(batch, -1, dtype=np.int32), np.full(batch + 1, -1, dtype=np.int32)
        self.sample, self.table = np.full(rows, -1, dtype=np.int64), np.full(batch * 8, -1, dtype=np.int32)
        self.prev = np.full(total, -7, dtype=np.int64)
        g = batch + 1  # cascade: the groups' arrays and the per-row suffix arrays
        self.perm, self.inv = np.full(total, -1, dtype=np.int64), np.full(total, -1, dtype=np.int64)
        self.cu_g, self.used_g = np.full(g + 1, -1, dtype=np.int32), np.full(g, -1, dtype=np.int32)
        self.table_g, self.table_s = np.full(g * 8, -1, dtype=np.int32), np.full(batch * 8, -1, dtype=np.int32)
        self.used_s = np.full(batch, -1, dtype=np.int32)


def decode(seq, pos, block_ids):
    return Row(seq=seq, start=pos, token_ids=[100 + seq], block_ids=block_ids, sample=True, decode=True)


def chunk(seq, start, n, block_ids, sample):
    return Row(seq=seq, start=start, token_ids=[200 + seq + i for i in range(n)], block_ids=block_ids, sample=sample,
               decode=False)


def test_keys_are_the_decode_sizes_then_the_mixed_grid_up_to_the_largest_one():
    sizes = graph_row_sizes(None, None, graph_max_rows=64)
    assert graphs.keys(sizes, mixed=False) == tuple((r, 0) for r in sizes)
    with_mixed = graphs.keys(sizes, mixed=True)
    assert with_mixed[:len(sizes)] == graphs.keys(sizes, mixed=False)
    assert with_mixed[len(sizes):] == tuple((r, t) for r in (32, 64) for t in graphs.MIXED_TOKENS)
    assert graphs.keys((), mixed=True) == ()


def test_a_graph_bakes_its_batch_and_token_count():
    assert graphs.shape((8, 0)) == (8, 8, 8)  # decode-only: one token per row, every row samples
    assert graphs.shape((32, 256)) == (33, 32 + 256, 32)  # mixed: one pad row more, rows + tokens tokens, rows slots
    assert graphs.sched_entries(33) == 1 + 36 * 4  # FlashAttention-3's schedule buffer, sized as vLLM sizes it
    assert graphs.sched_entries(1) == 17 and graphs.sched_entries(513) == 1 + 516 * 4
    assert graphs.splits((512, 0)) == 32 and graphs.splits((32, 64)) == 32 and graphs.splits((32, 512)) == 1
    assert graphs.splits((544, 0)) == 1  # past vLLM's largest full graph: no split accumulator in the pool


def test_a_step_takes_the_smallest_graph_that_holds_it_or_runs_eagerly():
    keys = graphs.keys((1, 2, 4, 8, 16, 32, 64), mixed=True)
    rows = [decode(i, 5, [1 + i]) for i in range(3)]
    assert graphs.key_for(Step(rows), keys) == (4, 0)
    assert graphs.key_for(Step(rows + [chunk(9, 0, 70, [20, 21, 22, 23, 24], False)]), keys) == (32, 128)
    assert graphs.key_for(Step(rows + [chunk(9, 0, 2048, [20] * 128, True)]), keys) == (32, 2048)
    assert graphs.key_for(Step(rows + [chunk(9, 0, 2049, [20] * 129, True)]), keys) is None  # past the largest bucket
    assert graphs.key_for(Step([decode(i, 0, [1 + i]) for i in range(65)]), keys) is None  # more rows than any graph
    without = graphs.keys((1, 2, 4, 8, 16, 32, 64), mixed=False)
    assert graphs.key_for(Step(rows + [chunk(9, 0, 7, [20], False)]), without) is None  # FA2: mixed steps run eagerly
    assert graphs.key_for(Step(rows), without) == (4, 0)


@pytest.mark.parametrize("key", [(4, 0), (32, 64), (32, 128)])
def test_the_layout_tiles_the_step_exactly_and_pads_the_rest(key):
    """Every real token once, in row order, then pads; cu over batch + 1 entries ending at the baked total; sample
    slots in sampling order, the unused ones at the last token; pad rows over nothing (used 0, block 0, slot 0)."""
    rows = [decode(0, 17, [3, 4]), decode(1, 2, [5])]
    if key[1]:
        rows += [chunk(2, 16, 30, [6, 7, 8], False), chunk(3, 0, 10, [9], True)]
    step = Step(rows)
    out = Arrays(key)
    batch, total, width, sampling, max_q, max_k = graphs.layout(step, key, B, out)
    assert (batch, total) == graphs.shape(key)[:2] and sampling == sum(r.sample for r in rows)
    assert max_k == max(r.start + len(r.token_ids) for r in rows)
    pad_row = total - sum(len(r.token_ids) for r in rows) - (key[0] - len(rows)) if key[1] else 1
    assert max_q == max(max(len(r.token_ids) for r in rows), pad_row)
    real = sum(len(r.token_ids) for r in rows)
    assert list(out.tokens[:real]) == [t for r in rows for t in r.token_ids] and not out.tokens[real:total].any()
    assert list(out.positions[:real]) == [p for r in rows for p in range(r.start, r.start + len(r.token_ids))]
    assert list(out.slots[:real]) == [r.block_ids[p // B] * B + p % B for r in rows
                                      for p in range(r.start, r.start + len(r.token_ids))]
    assert not out.slots[real:total].any() and not out.positions[real:total].any()
    assert all(out.prev[:total] == -1)  # no row gathers its token: every token given
    assert out.cu[0] == 0 and out.cu[batch] == total and all(out.cu[i] <= out.cu[i + 1] for i in range(batch))
    assert list(out.cu[1:len(rows) + 1]) == list(np.cumsum([len(r.token_ids) for r in rows]))
    assert list(out.used[:len(rows)]) == [r.start + len(r.token_ids) for r in rows] and not out.used[len(rows):batch].any()
    ends = [out.cu[i + 1] - 1 for i, r in enumerate(rows) if r.sample]
    assert list(out.sample[:sampling]) == ends and all(out.sample[sampling:key[0]] == total - 1)
    assert width == max((r.start + len(r.token_ids) - 1) // B + 1 for r in rows)
    table = out.table[:batch * width].reshape(batch, width)
    for i, r in enumerate(rows):
        k = (r.start + len(r.token_ids) - 1) // B + 1
        assert list(table[i, :k]) == r.block_ids[:k] and not table[i, k:].any()
    assert not table[len(rows):].any()
    if key[1]:  # the pad decode rows carry one token each; the last pad row carries the rest
        pads = [out.cu[i + 1] - out.cu[i] for i in range(len(rows), batch)]
        assert pads[:-1] == [1] * (key[0] - len(rows)) and pads[-1] == total - real - (key[0] - len(rows))


def test_an_empty_step_lays_out_as_all_pads():
    """What a capture stages: no real row, every row a pad row, the total still baked."""
    key = (8, 64)
    out = Arrays(key)
    batch, total, width, sampling, max_q, max_k = graphs.layout(Step([]), key, B, out)
    assert (batch, total, width, sampling, max_q, max_k) == (9, 72, 1, 0, 64, 1)  # the pad row holds the 64 tokens
    assert not out.tokens[:total].any() and not out.used[:batch].any() and out.cu[batch] == total
    assert list(out.cu[:batch + 1]) == list(range(9)) + [72]
    assert all(out.sample[:8] == total - 1)


def test_the_layout_refuses_a_step_its_graph_cannot_hold():
    out = Arrays((2, 0))
    with pytest.raises(AssertionError):
        graphs.layout(Step([decode(i, 0, [1 + i]) for i in range(3)]), (2, 0), B, out)
    with pytest.raises(AssertionError):  # a mixed step in a decode-only graph
        graphs.layout(Step([chunk(0, 0, 4, [1], True)]), (2, 0), B, out)


def test_a_row_that_gathers_its_token_names_the_slot_at_its_token():
    """overlap: a decode row whose token is the previous step's sampled id at slot prev puts that slot at its one
    token's index in the prev array; every other entry is -1 (the token given)."""
    key = (4, 0)
    rows = [decode(0, 3, [1]), Row(seq=1, start=9, token_ids=[0], block_ids=[2], sample=True, decode=True, prev=2),
            decode(2, 1, [3])]
    out = Arrays(key)
    graphs.layout(Step(rows), key, B, out)
    assert list(out.prev[:4]) == [-1, 2, -1, -1]


def test_the_cascade_layout_pads_the_groups_and_keeps_perm_a_permutation():
    """Under cascade the layout packs each group's tokens together (perm, its inverse), gives the pad tokens one
    last group over no keys, pads the groups to groups_of(key) with empty ones, and lays the suffix lengths and
    tables per row with the pad rows over no keys."""
    from batchinfer.cascade import Group
    key = (4, 64)
    rows = [decode(0, 40, [3, 4, 5]), decode(1, 33, [3, 4, 6]), chunk(2, 32, 10, [3, 4, 7, 8], True)]
    step = Step(rows)
    step.groups = [Group(9, [3, 4], 32, [0, 2]), Group(None, [], 0, [1])]  # rows 0 and 2 behind blocks 3, 4
    out = Arrays(key)
    laid = graphs.layout(step, key, B, out, cascade=True)
    batch, total = laid.batch, laid.total
    g = graphs.groups_of(key)
    assert sorted(out.perm[:total]) == list(range(total)) and all(out.perm[out.inv[t]] == t for t in range(total))
    assert list(out.perm[:3]) == [0, 2, 3]  # group 0: row 0's token, then row 2's ten... first three of them
    assert list(out.cu_g[:4]) == [0, 11, 12, total] and all(out.cu_g[4:g + 1] == total)  # group 0, group 1, pads, empties
    assert list(out.used_g[:3]) == [32, 0, 0] and not out.used_g[3:g].any()
    tg = out.table_g[:g * laid.width].reshape(g, laid.width)
    assert list(tg[0, :2]) == [3, 4] and not tg[1:].any()
    assert list(out.used_s[:3]) == [41 - 32, 34, 42 - 32] and not out.used_s[3:batch].any()
    ts = out.table_s[:batch * laid.width].reshape(batch, laid.width)
    assert list(ts[0, :1]) == [5] and list(ts[1, :3]) == [3, 4, 6] and list(ts[2, :2]) == [7, 8] and not ts[3:].any()
