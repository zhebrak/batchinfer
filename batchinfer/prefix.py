"""Global prefix trie: the job's prompts keyed by full KV blocks, built once by analysis. Pure Python.

One node is one block's worth of token ids at one depth, so a node is exactly one KV block that can be
computed once and read by every request whose prompt passes through it. A request's path stops before
its last prompt token, because that token's logits must come from the request's own forward; identical
prompts therefore share all but their last block. The trie is a fact about the job and holds no per-run
state: block ids, owners and remaining-user counts live in the scheduler, in lists indexed by node id.

Sizes on the mixed-quick workload (2026-09-27): 3,481 nodes over 297k prompt tokens, 17 branch points,
paths up to ~500 nodes deep. Per-block nodes and an iterative walk are enough; no radix compression.
"""


class Node:
    __slots__ = ("id", "depth", "users", "children")

    def __init__(self, id, depth):
        self.id = id  # index into PrefixTrie.nodes; -1 for the root
        self.depth = depth  # 1-based block index: the node covers prompt positions [(depth-1)*bs, depth*bs)
        self.users = []  # requests whose path passes through this node, in input order
        self.children = {}  # block token tuple -> Node


class PrefixTrie:
    def __init__(self, block_size):
        self.block_size = block_size
        self.root = Node(-1, 0)
        self.nodes = []  # by id
        self.paths = []  # per request: node ids from the root down, one per full block before the last token
        self.tails = []  # per request: prompt tokens after its path (>= 1 for a non-empty prompt)
        self.ends = {}  # node id (-1: root) -> requests whose path ends at that node

    @classmethod
    def build(cls, seqs, block_size):
        trie = cls(block_size)
        for i, seq in enumerate(seqs):
            node, path = trie.root, []
            for k in range(max(0, (len(seq) - 1) // block_size)):
                key = tuple(seq[k * block_size:(k + 1) * block_size])
                child = node.children.get(key)
                if child is None:
                    child = Node(len(trie.nodes), k + 1)
                    trie.nodes.append(child)
                    node.children[key] = child
                child.users.append(i)
                path.append(child.id)
                node = child
            trie.paths.append(path)
            trie.tails.append(len(seq) - len(path) * block_size)
            trie.ends.setdefault(node.id, []).append(i)
        return trie

    # facts ------------------------------------------------------------------------------------------

    def metrics(self):
        """unique_prefill_tokens is what an engine that shares whole blocks must compute: every node once
        plus every tail. It is at least unique_prompt_tokens (token-level, analysis.py), and larger exactly
        when a prompt is a block-aligned prefix of another, since the last block stays private."""
        return {
            "trie_nodes": len(self.nodes),
            "shared_nodes": sum(1 for n in self.nodes if len(n.users) > 1),
            "branch_points": sum(1 for n in (self.root, *self.nodes) if len(n.children) > 1),
            "shared_subtrees": sum(1 for c in self.root.children.values() if len(c.users) > 1),
            "unique_prefill_tokens": self.block_size * len(self.nodes) + sum(self.tails),
        }

    def segments(self):
        """Maximal runs of nodes with the same users and a single child: (users, blocks, first depth).
        A run ends at a branch point, at a leaf, or where a request's path ends."""
        out, stack = [], list(self.root.children.values())
        while stack:
            node = stack.pop()
            users, blocks = len(node.users), 1
            while len(node.children) == 1:
                (child,) = node.children.values()
                if len(child.users) != users:
                    break
                node, blocks = child, blocks + 1
            out.append((users, blocks, node.depth - blocks + 1))
            stack.extend(node.children.values())
        return out

    def describe(self, top=8):
        m = self.metrics()
        singles = sum(1 for c in self.root.children.values() if len(c.users) == 1)
        lines = [f"prefix trie ({self.block_size}-token blocks): {m['trie_nodes']:,} nodes, {m['shared_nodes']:,} shared "
                 f"by >1 request, {m['branch_points']} branch points; at the root {m['shared_subtrees']} shared subtrees "
                 f"and {singles} single-request prompts; unique prefill {m['unique_prefill_tokens']:,} tokens"]
        shared = sorted((s for s in self.segments() if s[0] > 1), key=lambda s: (-(s[0] - 1) * s[1], -s[0]))
        for users, blocks, depth in shared[:top]:
            lines.append(f"  {users:5} requests share {blocks:4} blocks ({blocks * self.block_size:,} tokens) from block {depth}")
        return "\n".join(lines)

    # order ------------------------------------------------------------------------------------------

    def dfs_order(self, ranks):
        """Admission order: an iterative depth-first walk. ranks[i] is request i's sort key, smallest first,
        and must be unique (pair it with the input index). At every node, the requests whose path ends
        there and the child subtrees are visited in rank order, a subtree ranked by the best of its users.
        So every subtree's requests are contiguous, and the best-ranked subtree goes first."""
        best = [min(ranks[u] for u in n.users) for n in self.nodes]
        order, stack = [], [self._visits(self.root, ranks, best)]
        while stack:
            item = next(stack[-1], None)
            if item is None:
                stack.pop()
            elif isinstance(item, Node):
                stack.append(self._visits(item, ranks, best))
            else:
                order.append(item)
        return order

    def _visits(self, node, ranks, best):
        items = [(ranks[i], 0, i) for i in self.ends.get(node.id, ())]
        items += [(best[c.id], 1, c) for c in node.children.values()]
        items.sort(key=lambda t: t[:2])
        return iter(t[2] for t in items)
