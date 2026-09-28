"""Paged KV block bookkeeping, pure Python: which physical blocks are free, and how many a request
reserves. The tensors themselves live in executor.py; this module never imports torch.

A sequence's KV lives in fixed-size blocks of BLOCK_SIZE token slots. Position p of a sequence is
slot block_ids[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE of the pool. Block 0 is the pad block:
never handed out, so padded block-table entries can never alias live KV.
"""
import random

BLOCK_SIZE = 16  # small pages keep few-shot prefixes shareable later; vLLM's FA2 accepts 16
PAD_BLOCK = 0


def reservation_blocks(prompt_len, max_tokens, block_size=BLOCK_SIZE):
    """Blocks a request needs for its whole life: every prompt position plus every sampled token that
    is fed back. The last sampled token is never fed back, hence max_tokens - 1. Reserving this at
    admission means the pool can never run dry mid-decode, so nothing is ever preempted."""
    return -(-(prompt_len + max_tokens - 1) // block_size)


class BlockAllocator:
    def __init__(self, num_blocks, seed=None):
        """num_blocks counts the pad block, so num_blocks - 1 are usable. With a seed the free list
        is shuffled, so tests and the executor gate see scattered, non-contiguous block ids."""
        if num_blocks < 2:
            raise ValueError(f"need at least 2 blocks (one is the pad block), got {num_blocks}")
        self.num_blocks = num_blocks
        self.free_ids = list(range(num_blocks - 1, PAD_BLOCK, -1))  # pop() hands out 1, 2, 3, ...
        if seed is not None:
            random.Random(seed).shuffle(self.free_ids)

    @property
    def capacity(self):
        return self.num_blocks - 1

    @property
    def free(self):
        return len(self.free_ids)

    def reserve(self, n):
        if n > len(self.free_ids):
            raise MemoryError(f"asked for {n} blocks, {len(self.free_ids)} free")
        return [self.free_ids.pop() for _ in range(n)]

    def release(self, ids):
        self.free_ids.extend(ids)
