"""prefix trie: full-block nodes, the last prompt token never shared, a DFS order with contiguous subtrees,
and nodes that hold facts only. Pure Python; block size 4 so the cases stay small."""
from batchinfer.prefix import Node, PrefixTrie
from bench.workload import unique_tokens as bench_unique_tokens

BS = 4
A = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # two nodes, tail 1
B = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # shares both nodes with A, tail 2
C = [1, 2, 3, 4, 9, 9, 9, 9, 9]  # shares the first node, has its own second node, tail 1
D = [1, 2, 3, 4, 5, 6, 7, 8]  # a block-aligned prefix of A: one node, its last block stays private
E = list(A)  # a duplicate of A: shares all but the last block
F = [7]  # no node, tail 1
G = []  # nothing
SEQS = [A, B, C, D, E, F, G]


def trie():
    return PrefixTrie.build(SEQS, BS)


def test_nodes_are_full_blocks_and_the_last_prompt_token_is_never_shared():
    t = trie()
    assert [len(p) for p in t.paths] == [2, 2, 2, 1, 2, 0, 0] == [max(0, (len(s) - 1) // BS) for s in SEQS]
    assert t.tails == [1, 2, 1, 4, 1, 1, 0]
    assert t.paths[0] == t.paths[1] == t.paths[4]  # A, B and the duplicate E
    assert t.paths[2][0] == t.paths[0][0] and t.paths[2][1] != t.paths[0][1]  # C branches at block 2
    assert t.paths[3] == t.paths[0][:1]  # D keeps [5, 6, 7, 8] private
    assert [n.users for n in t.nodes] == [[0, 1, 2, 3, 4], [0, 1, 4], [2]]
    assert [n.depth for n in t.nodes] == [1, 2, 2] and [n.id for n in t.nodes] == [0, 1, 2]
    assert t.ends == {1: [0, 1, 4], 2: [2], 0: [3], -1: [5, 6]}


def test_metrics_count_every_node_once_plus_the_tails():
    assert trie().metrics() == {"trie_nodes": 3, "shared_nodes": 2, "branch_points": 1, "shared_subtrees": 1,
                                "unique_prefill_tokens": 3 * BS + (1 + 2 + 1 + 4 + 1 + 1 + 0)}


def test_unique_prefill_matches_bench_unless_a_prompt_is_a_block_aligned_prefix_of_another():
    seqs = [A, B, C, E, F, G]
    assert PrefixTrie.build(seqs, BS).metrics()["unique_prefill_tokens"] == bench_unique_tokens(seqs, page=BS) == 18
    # D's last block is shared in the token-level count; the trie keeps it private (one block more)
    assert trie().metrics()["unique_prefill_tokens"] == bench_unique_tokens(SEQS, page=BS) + BS


def test_segments_end_at_branches_and_where_a_path_ends():
    t = PrefixTrie.build([list(range(1, 14)), list(range(1, 10))], BS)  # 3 nodes and 2 nodes, the second a prefix
    assert sorted(t.segments()) == [(1, 1, 3), (2, 2, 1)]
    text = trie().describe()
    assert "3 nodes, 2 shared" in text and "5 requests share    1 blocks" in text and "3 requests share" in text


def test_dfs_order_keeps_subtrees_contiguous_and_visits_the_best_rank_first():
    t = trie()
    rank = {0: 5, 1: 3, 2: 1, 3: 4, 4: 6, 5: 0, 6: 2}  # smaller first, unique
    order = t.dfs_order([(rank[i], i) for i in range(7)])
    # root: F (0), then the subtree under [1,2,3,4] (best rank 1, C), then G (2).
    # in that subtree: C's branch (1) before A/B/E's branch (best 3) before D, which ends at the subtree root (4).
    assert order == [5, 2, 1, 0, 4, 3, 6]
    assert sorted(order) == list(range(7))
    for node in t.nodes:
        positions = sorted(order.index(u) for u in node.users)
        assert positions == list(range(positions[0], positions[0] + len(positions))), "subtree not contiguous"


def test_dfs_is_iterative_on_paths_deeper_than_the_recursion_limit():
    long = list(range(2, 2 + 2000 * BS + 1))
    t = PrefixTrie.build([long, long[:1000 * BS] + [99] * 5], BS)
    assert len(t.nodes) == 2001 and len(t.paths[0]) == 2000 and len(t.paths[1]) == 1001
    assert t.dfs_order([(0, 0), (1, 1)]) == [0, 1] and t.dfs_order([(1, 0), (0, 1)]) == [1, 0]


def test_nodes_hold_facts_only():
    assert Node.__slots__ == ("id", "depth", "users", "children")
    t = trie()
    for name in ("state", "block", "owner", "remaining"):
        assert not hasattr(t.nodes[0], name)
    assert not hasattr(t, name)
