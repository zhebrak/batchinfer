"""attention=cascade: rows of a step that share a computed prefix read its KV once per step, not once per row.

The scheduler groups a step's rows by the deepest trie node that at least two of them share among the leading
computed nodes of their paths (groups: a group's prefix is whole, written in an earlier step, and lies before every
member's first query position); every other row is a group of its own with no prefix. The executor then runs two
attention calls per layer, the way vLLM's FlashAttention backend runs cascade attention: one over the groups, each
group's queries packed together attending to its prefix without a causal mask (every query lies after the whole
prefix), and one per row over the row's own suffix with the causal mask; the two partial results are merged by their
log-sum-exps (merge, vLLM's merge_attn_states on the GPU), which is exact. The gain is memory reads at decode and
batched suffix prefill, largest with many rows behind a few prefixes; a step without shared prefixes costs one extra
zero-length call. Under graphs every per-step array of it lives in a refilled buffer, padded to the graph's rows.
"""
from collections import Counter, namedtuple

import torch

Group = namedtuple("Group", "node blocks shared rows")  # the trie node (None: no prefix), its chain's block ids,
#                                                         the prefix length in tokens, the step's row indices


def groups(rows, path_of, node_state, node_block, block_size, computed="computed", exclude=()):
    """The step's rows grouped by the deepest node at least two of them share among the leading computed nodes of
    their paths (exclude: nodes marked computed for this step's own writes, under overlap), in row order; a row
    sharing nothing is its own group with no prefix. Readers of a node are a subset of its parent's, so the shared
    nodes of a row are a prefix of its leading run."""
    runs = []
    for r in rows:
        run = []
        for nid in path_of(r.seq):
            if node_state[nid] != computed or nid in exclude:
                break
            run.append(nid)
        runs.append(run)
    readers = Counter(nid for run in runs for nid in run)
    out, by_node = [], {}
    for i, run in enumerate(runs):
        shared = [nid for nid in run if readers[nid] >= 2]
        if not shared:
            out.append(Group(None, [], 0, [i]))
            continue
        key = shared[-1]
        if key not in by_node:
            by_node[key] = Group(key, [node_block[nid] for nid in shared], block_size * len(shared), [])
            out.append(by_node[key])
        by_node[key].rows.append(i)
    return out


def plan(step, block_size):
    """The arrays the two calls take, from the step's groups: perm (token indices in group order) and its inverse,
    cu_g and used_g (the groups' query ranges in perm order and their prefix lengths), table_g (each group's prefix
    blocks), and per row used_s and table_s (the suffix keys and blocks) and shared_s (tokens of prefix the row's
    keys start after). A row's queries are all past its prefix: start >= shared."""
    rows, cu = step.rows, [0]
    for r in rows:
        cu.append(cu[-1] + len(r.token_ids))
    perm, cu_g, used_g, table_g = [], [0], [], []
    shared_s = [0] * len(rows)
    for g in step.groups:
        for i in g.rows:
            assert rows[i].start >= g.shared, "a cascade prefix lies before every query of its group"
            perm.extend(range(cu[i], cu[i + 1]))
            shared_s[i] = g.shared
        cu_g.append(len(perm))
        used_g.append(g.shared)
        table_g.append(list(g.blocks))
    inv = [0] * len(perm)
    for j, t in enumerate(perm):
        inv[t] = j
    used_s = [r.start + len(r.token_ids) - shared_s[i] for i, r in enumerate(rows)]
    table_s = [r.block_ids[shared_s[i] // block_size:] for i, r in enumerate(rows)]
    return perm, inv, cu_g, used_g, table_g, used_s, table_s, shared_s


def lse_attention(q, k, v, scaling, causal):
    """Attention with its log-sum-exp, in float: q [n, H, D], k and v [m, H, D] -> (out [n, H, D], lse [H, n]). For
    the torch oracle of the cascade; m may be 0 (out zeros, lse -inf)."""
    n, h, d = q.shape
    if k.shape[0] == 0:
        return torch.zeros_like(q), torch.full((h, n), float("-inf"), dtype=torch.float32, device=q.device)
    s = torch.einsum("nhd,mhd->hnm", q.float(), k.float()) * scaling
    if causal:
        m = k.shape[0]
        s = s.masked_fill(~torch.ones(n, m, dtype=torch.bool, device=q.device).tril(diagonal=m - n), float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    out = torch.einsum("hnm,mhd->nhd", torch.exp(s - lse[..., None]), v.float())
    return out.to(q.dtype), lse


def merge(out_p, lse_p, out_s, lse_s):
    """The exact merge of two partial attentions over disjoint keys: [n, H, D] outputs with [H, n] log-sum-exps
    (-inf for no keys) -> [n, H, D]. What vLLM's merge_attn_states computes, in torch for the oracle."""
    m = torch.maximum(lse_p, lse_s)
    wp, ws = torch.exp(lse_p - m), torch.exp(lse_s - m)
    wp, ws = wp.t()[..., None], ws.t()[..., None]  # [n, H, 1]
    return ((out_p.float() * wp + out_s.float() * ws) / (wp + ws)).to(out_s.dtype)
