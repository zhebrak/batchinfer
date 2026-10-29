"""batchinfer.cascade: the grouping of a step's rows behind shared computed prefixes, the two-call plan, and the
exact merge, on the CPU. The GPU kernels (FlashAttention with return_softmax_lse, vLLM's merge_attn_states) run the
same arithmetic; tests/test_executor_gpu.py checks them."""
import torch

from batchinfer import cascade
from batchinfer.schema import Row, Step

B = 16
COMPUTED, PENDING = "computed", "pending"


def row(seq, start, n, blocks, decode=False):
    return Row(seq=seq, start=start, token_ids=[seq] * n, block_ids=blocks, sample=True, decode=decode)


def test_rows_group_by_the_deepest_computed_node_at_least_two_share():
    """Paths: rows 0 and 1 share nodes a, b; row 2 shares a only; row 3 shares nothing; row 4 shares a, b but b is
    still pending for it to read, so it groups under a with row 2; every row lands in exactly one group, a group's
    prefix is a prefix of each member's blocks, singletons have no prefix."""
    paths = {0: [10, 11], 1: [10, 11], 2: [10, 12], 3: [], 4: [10, 11, 13]}
    state = {10: COMPUTED, 11: COMPUTED, 12: COMPUTED, 13: PENDING}
    block = {10: 100, 11: 101, 12: 102, 13: 103}
    rows = [row(0, 40, 1, [100, 101, 5], True), row(1, 33, 1, [100, 101, 6], True), row(2, 32, 4, [100, 102, 7]),
            row(3, 2, 3, [8]), row(4, 48, 1, [100, 101, 103, 9], True)]
    gs = cascade.groups(rows, paths.__getitem__, state, block, B)
    assert [(g.node, g.blocks, g.shared, g.rows) for g in gs] == [(11, [100, 101], 32, [0, 1, 4]), (10, [100], 16, [2]),
                                                                  (None, [], 0, [3])]
    assert sorted(i for g in gs for i in g.rows) == list(range(5))
    for g in gs:
        for i in g.rows:
            assert rows[i].block_ids[:len(g.blocks)] == g.blocks and rows[i].start >= g.shared
    gs = cascade.groups(rows, paths.__getitem__, state, block, B)
    singles = [g for g in gs if len(g.rows) == 1]
    assert all(g.shared == 0 or g.node == 10 for g in singles)  # row 2 alone behind a: a prefix read once is no loss


def test_the_plan_packs_group_queries_and_slices_the_suffixes():
    rows = [row(0, 40, 1, [100, 101, 5], True), row(1, 33, 1, [100, 101, 6], True), row(2, 2, 3, [8])]
    step = Step(rows)
    step.groups = [cascade.Group(11, [100, 101], 32, [0, 1]), cascade.Group(None, [], 0, [2])]
    perm, inv, cu_g, used_g, table_g, used_s, table_s, shared = cascade.plan(step, B)
    assert perm == [0, 1, 2, 3, 4] and inv == [0, 1, 2, 3, 4]
    assert cu_g == [0, 2, 5] and used_g == [32, 0] and table_g == [[100, 101], []]
    assert used_s == [41 - 32, 34 - 32, 5] and table_s == [[5], [6], [8]] and shared == [32, 32, 0]
    step.groups = [cascade.Group(None, [], 0, [2]), cascade.Group(11, [100, 101], 32, [0, 1])]  # another order
    perm, inv, cu_g, *_ = cascade.plan(step, B)
    assert perm == [2, 3, 4, 0, 1] and inv == [3, 4, 0, 1, 2] and cu_g == [0, 3, 5]


def test_the_merge_of_two_partial_attentions_is_the_whole_attention():
    """Random q against keys split into a prefix (no mask) and a causal suffix, merged by log-sum-exp, equals one
    causal attention over all the keys; a prefix of no keys merges to the suffix alone."""
    torch.manual_seed(1)
    n, m, h, d = 5, 23, 4, 8
    q, k, v = (torch.randn(x, h, d) for x in (n, m, m))
    scaling = d ** -0.5
    whole, _ = cascade.lse_attention(q, k, v, scaling, causal=True)
    for shared in (0, 7, 18):
        out_p, lse_p = cascade.lse_attention(q, k[:shared], v[:shared], scaling, causal=False)
        out_s, lse_s = cascade.lse_attention(q, k[shared:], v[shared:], scaling, causal=True)
        merged = cascade.merge(out_p, lse_p, out_s, lse_s)
        assert torch.allclose(merged, whole, atol=1e-5), shared
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), is_causal=False, scale=scaling,
        attn_mask=torch.ones(n, m, dtype=torch.bool).tril(diagonal=m - n)).transpose(0, 1)
    assert torch.allclose(whole, ref, atol=1e-5)
