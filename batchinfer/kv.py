"""Paged KV block bookkeeping, pure Python: which physical blocks are free, and how many a request
reserves. The tensors themselves live in executor.py; this module never imports torch.

A sequence's KV lives in fixed-size blocks of block_size token slots (the executor's, DEFAULT_BLOCK_SIZE unless set). Position p of a sequence is
slot block_ids[p // block_size] * block_size + p % block_size of the pool. Block 0 is the pad block:
never handed out, so padded block-table entries can never alias live KV.
"""
import heapq
import random

DEFAULT_BLOCK_SIZE = 16  # the executor's default; every reader takes the run's block size from Limits or the executor
PAD_BLOCK = 0


def reservation_blocks(prompt_len, max_tokens, block_size):
    """Blocks a request needs for its whole life: every prompt position plus every sampled token that
    is fed back. The last sampled token is never fed back, hence max_tokens - 1. Reserving this at
    admission means the pool can never run dry mid-decode, so nothing is ever preempted."""
    return -(-(prompt_len + max_tokens - 1) // block_size)


class BlockAllocator:
    def __init__(self, num_blocks, seed=None, group=None):
        """num_blocks counts the pad block, so num_blocks - 1 are usable. With a seed the free list
        is shuffled, so tests and the executor gate see scattered, non-contiguous block ids. group: pool=elastic,
        the blocks that share one set of physical pieces; then the lowest free id goes first (a heap), so live blocks
        cluster in few groups, and the allocator counts live blocks per group: to_map holds the groups that gained
        their first live block since the executor last took them, to_unmap those that lost their last (the pad
        block keeps group 0 live for good)."""
        if num_blocks < 2:
            raise ValueError(f"need at least 2 blocks (one is the pad block), got {num_blocks}")
        self.num_blocks, self.group = num_blocks, group
        self.free_ids = list(range(num_blocks - 1, PAD_BLOCK, -1))  # pop() hands out 1, 2, 3, ...
        self.to_map, self.to_unmap = set(), set()  # stay empty without groups
        if group is not None:
            self.free_ids = list(range(1, num_blocks))  # a heap: the lowest id first
            self.live = [0] * -(-num_blocks // group)
            self.live[0] = 1  # the pad block
        elif seed is not None:
            random.Random(seed).shuffle(self.free_ids)

    @property
    def capacity(self):
        return self.num_blocks - 1

    @property
    def free(self):
        return len(self.free_ids)

    @property
    def mapped_groups(self):
        """pool=elastic: the groups with a live block (physical memory under them); None for a static pool."""
        return sum(1 for n in self.live if n) if self.group is not None else None

    def reserve(self, n):
        if n > len(self.free_ids):
            raise MemoryError(f"asked for {n} blocks, {len(self.free_ids)} free")
        if self.group is None:
            return [self.free_ids.pop() for _ in range(n)]
        ids = [heapq.heappop(self.free_ids) for _ in range(n)]
        for b in ids:
            g = b // self.group
            if self.live[g] == 0:
                if g in self.to_unmap:
                    self.to_unmap.discard(g)  # freed and taken again before the executor acted: still mapped
                else:
                    self.to_map.add(g)
            self.live[g] += 1
        return ids

    def release(self, ids):
        if self.group is None:
            self.free_ids.extend(ids)
            return
        for b in ids:
            heapq.heappush(self.free_ids, b)
            g = b // self.group
            self.live[g] -= 1
            if self.live[g] == 0:
                if g in self.to_map:
                    self.to_map.discard(g)  # never mapped: nothing to unmap
                else:
                    self.to_unmap.add(g)

    def take_map(self):
        """The groups to map before the next step runs; cleared."""
        out, self.to_map = sorted(self.to_map), set()
        return out

    def take_unmap(self):
        """The groups free since the last take, to unmap once the last step that could touch them is done; cleared."""
        out, self.to_unmap = sorted(self.to_unmap), set()
        return out
