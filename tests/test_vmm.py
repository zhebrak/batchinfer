"""pool=elastic off the GPU: the grouped allocator's bookkeeping over random reserve and release sequences, the fake
driver's piece accounting, and the engine mapping and unmapping through it. The driver calls themselves (vmm.Vmm over
libcuda) run on the GPU: tests/test_executor_gpu.py."""
import random

import pytest
import torch

from batchinfer import vmm
from batchinfer.kv import BlockAllocator


def test_the_grouped_allocator_hands_out_the_lowest_ids_and_counts_live_blocks_per_group():
    kv = BlockAllocator(65, group=8)  # 64 usable blocks in 9 groups, group 0 holding the pad block
    assert kv.live[0] == 1 and kv.mapped_groups == 1 and kv.take_map() == [] and kv.take_unmap() == []
    a = kv.reserve(10)
    assert a == list(range(1, 11)) and kv.take_map() == [1] and kv.live[:2] == [8, 3]  # ids 1..7 in group 0, 8..10 in 1
    kv.release([8, 9, 10])
    assert kv.take_unmap() == [1] and kv.mapped_groups == 1
    b = kv.reserve(2)
    assert b == [8, 9] and kv.take_map() == [1]  # the lowest ids again: the group comes back
    kv.release([8, 9])
    kv.reserve(1)  # 8 again, before the executor acted on the unmap: still mapped, nothing to do
    assert kv.take_unmap() == [] and kv.take_map() == []
    kv.release([8])
    kv.reserve(1)
    assert kv.take_unmap() == [] and kv.take_map() == []
    c = kv.reserve(20)
    assert kv.take_map() == [2, 3]  # 11..30 fill groups 1 (has 8), 2 and 3
    kv.release(c)
    assert kv.take_unmap() == [2, 3]  # 11..15 leave group 1, which keeps block 8; groups 2 and 3 are empty
    assert kv.mapped_groups == 2


def test_random_sequences_never_unmap_a_live_group_nor_map_above_the_pool():
    rng = random.Random(4)
    for trial in range(50):
        group, n = rng.choice([4, 8, 16]), rng.randint(8, 100)
        kv, held, mapped = BlockAllocator(n, group=group), [], {0}
        for _ in range(200):
            if held and rng.random() < 0.5:
                k = rng.randint(1, len(held))
                rng.shuffle(held)
                kv.release(held[:k])
                held = held[k:]
            elif kv.free:
                held += kv.reserve(rng.randint(1, kv.free))
            for g in kv.take_map():
                assert g not in mapped and g < len(kv.live)
                mapped.add(g)
            for g in kv.take_unmap():
                assert g in mapped and g != 0
                mapped.remove(g)
            live = {0} | {b // group for b in held}
            assert mapped == live, (trial, mapped, live)
            assert kv.mapped_groups == len(live)
            assert sorted(held + kv.free_ids) == list(range(1, n))  # every usable id exactly once
        if kv.free:
            assert kv.reserve(1) == [min(set(range(1, n)) - set(held))]  # lowest first


def test_the_fake_driver_keeps_pieces_by_offset():
    fake = vmm.Fake(granularity=1024)
    fake.reserve(8192)
    fake.map(1024, 2048)
    with pytest.raises(AssertionError):
        fake.map(1024, 1024)  # twice
    with pytest.raises(AssertionError):
        fake.map(512, 1024)  # off the granularity
    assert fake.mapped_bytes == 2048
    fake.unmap(1024)
    with pytest.raises(KeyError):
        fake.unmap(1024)
    assert fake.tensor((2, 4), torch.bfloat16).shape == (2, 4) and fake.mapped_bytes == 0
