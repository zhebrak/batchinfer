"""Where the KV lives: the pool tensor the kernels write and read, in one of three modes, behind one interface the
executor runs over (the scheduler never sees it: it holds block ids, which are rows of these tensors).

- static: one tensor, [layers, K|V, blocks, block slots, kv heads, head dim], allocated at build.
- elastic: the same shape over CUDA virtual memory (vmm.py): the blocks that share one set of physical pieces form a
  group, mapped before a step writes its first live block and unmapped once the last step that could touch it is
  done, so a card shared with another process gets memory back as rows finish; the address never changes, so the
  graphs and admission are untouched. The allocator counts live blocks per group (kv.BlockAllocator(group=...)).
- contiguous: the vAttention arrangement. The block is the mapping unit (page_tokens); writes go through a
  token-major home, [blocks, unit tokens, layers, K|V, kv heads, head dim], one physical allocation per unit created
  once; reads go through per-slot regions, a second virtual range where every live sequence's units are mapped
  again, in order, so its KV is one contiguous run and attention needs no block table (the scheduler hands each
  admitted sequence a slot, Row.slot). The pad unit sits after a row's last unit as the guard FlashAttention-3's
  tile loads need. Off the GPU the home is a plain tensor, the regions are bookkeeping over vmm.Fake, and the torch
  backend gathers from the home.

The hooks the executor calls, no-ops where a mode has nothing to do: before_step (the step's blocks have memory
and its rows their regions), release (freed memory queued to go after the last step that may touch it), finish (a
job's last step is done), reset (a job died: a fresh allocator comes), and the probe's begin and end (the widest
step writes across the whole pool).
"""
import math
from collections import namedtuple

import torch

from . import vmm
from .kv import PAD_BLOCK
from .schema import Step

Spec = namedtuple("Spec", "layers kv_heads head_dim block_size page_tokens dtype device max_seqs max_positions")
MODES = ("static", "elastic", "contiguous")


def make(mode, spec, num_blocks):
    return {"static": StaticPool, "elastic": ElasticPool, "contiguous": ContiguousPool}[mode](spec, num_blocks)


class StaticPool:
    mode, layout = "static", "layer"  # layout: layer-major [L, 2, blocks, ...], or token-major (contiguous)

    def __init__(self, spec, num_blocks):
        self.spec, self.num_blocks = spec, num_blocks
        self.cuda = spec.device.type == "cuda"
        self.blocks_per_group = None  # elastic: what the allocator counts live blocks by
        self.kv = self._alloc(num_blocks)

    def _alloc(self, num_blocks):
        s = self.spec
        return torch.zeros((s.layers, 2, num_blocks, s.block_size, s.kv_heads, s.head_dim), dtype=s.dtype, device=s.device)

    @property
    def bytes_per_block(self):
        s = self.spec
        return 2 * s.layers * s.block_size * s.kv_heads * s.head_dim * s.dtype.itemsize

    def k_cache(self, layer):
        """This layer's K as the write kernels and the paged reads take it: [blocks, block slots, kv heads, head dim]."""
        return self.kv[layer, 0]

    def v_cache(self, layer):
        return self.kv[layer, 1]

    def write(self, layer, key, value, slots):
        """key and value [N, kv heads, head dim] into the slots (block * block_size + position), the HF path's write."""
        s = self.spec
        self.kv[layer, 0].view(-1, s.kv_heads, s.head_dim)[slots] = key.to(s.dtype)
        self.kv[layer, 1].view(-1, s.kv_heads, s.head_dim)[slots] = value.to(s.dtype)

    def copy(self, src, dst, k):
        """sharing=token: the first k slots of block src into block dst, every layer's K and V."""
        self.kv[:, :, dst, :k].copy_(self.kv[:, :, src, :k])

    def before_step(self, step, maps):
        """Before the step launches: memory under its blocks (maps: the groups the allocator handed out since the
        last step, elastic) and its rows' regions (contiguous)."""

    def release(self, groups, last):
        """Groups the allocator freed; last: the step launched last, the last that may still touch them."""

    def finish(self):
        """The job's last step is done."""

    def reset(self):
        """A job died mid-step and the allocator is replaced: every block's memory but the pad block's goes back."""

    def probe_begin(self):
        """The probe writes every decode row into its own block and a chunk across the rest: whatever that needs."""
        return None

    def probe_end(self, token):
        pass


class ElasticPool(StaticPool):
    mode = "elastic"

    def _alloc(self, num_blocks):
        s = self.spec
        self.vmm = vmm.Vmm(s.device.index or 0) if self.cuda else vmm.Fake()
        slab = s.block_size * s.kv_heads * s.head_dim * s.dtype.itemsize  # one block of one layer's K or V
        gran = self.vmm.granularity
        self.piece_bytes = gran * slab // math.gcd(gran, slab)  # the smallest size both divide: whole slabs
        self.blocks_per_group = self.piece_bytes // slab
        self.groups = -(-num_blocks // self.blocks_per_group)
        padded = self.groups * self.blocks_per_group  # the blocks dimension in whole groups: no piece straddles rows
        self.row_bytes = padded * slab
        self.mapped, self.unmap_queue = set(), []  # groups with memory now; (groups, the handle they wait on)
        self.vmm.reserve(self.row_bytes * 2 * s.layers)
        self.map_groups([0])  # the pad block's group, live for good
        return self.vmm.tensor((s.layers, 2, padded, s.block_size, s.kv_heads, s.head_dim), s.dtype)

    def pieces(self, group):
        """The byte offsets of a group's pieces: its slab run in every layer's K and V."""
        return [(l * 2 + j) * self.row_bytes + group * self.piece_bytes for l in range(self.spec.layers) for j in range(2)]

    def map_groups(self, groups):
        """Physical memory under these groups' blocks. A group still mapped (freed, queued to go, and taken again
        before its last reader finished) stays as it is and leaves the queue."""
        for g in groups:
            if g in self.mapped:
                self.unmap_queue = [(gs - {g}, h) for gs, h in self.unmap_queue if gs - {g}]
                continue
            for off in self.pieces(g):
                self.vmm.map(off, self.piece_bytes)
            self.mapped.add(g)

    def unmap(self, groups):
        for g in groups:
            if g in self.mapped:
                for off in self.pieces(g):
                    self.vmm.unmap(off)
                self.mapped.discard(g)

    def reap(self, wait=False):
        """Unmaps the queued groups whose step has finished (all of them, waiting, with wait)."""
        kept = []
        for groups, handle in self.unmap_queue:
            if wait or handle.done():
                self.unmap(groups)
            else:
                kept.append((groups, handle))
        self.unmap_queue = kept

    def before_step(self, step, maps):
        self.reap()
        self.map_groups(maps)

    def release(self, groups, last):
        if groups:
            self.unmap_queue.append((set(groups), last))

    def finish(self):
        self.reap(wait=True)

    def reset(self):
        if self.cuda:
            torch.cuda.synchronize(self.spec.device)
        self.unmap_queue = []
        self.unmap([g for g in sorted(self.mapped) if g != 0])

    def probe_begin(self):
        extra = [g for g in range(1, self.groups) if g not in self.mapped]
        self.map_groups(extra)
        return extra

    def probe_end(self, extra):
        if extra:  # only what the probe mapped goes back
            if self.cuda:
                torch.cuda.synchronize(self.spec.device)
            self.unmap(extra)


class ContiguousPool(StaticPool):
    mode, layout = "contiguous", "token"

    def _alloc(self, num_blocks):
        s = self.spec
        self.unit_bytes = s.page_tokens * 2 * s.layers * s.kv_heads * s.head_dim * s.dtype.itemsize
        self.vmm = vmm.Vmm(s.device.index or 0) if self.cuda else vmm.Fake(granularity=self.unit_bytes)
        assert self.unit_bytes % self.vmm.granularity == 0, "page_tokens makes a unit a whole number of pieces"
        self.vmm.reserve(num_blocks * self.unit_bytes)
        for b in range(num_blocks):  # one physical unit per block, for good: every write and copy goes through here
            self.vmm.map(b * self.unit_bytes, self.unit_bytes)
        # the regions: a slot per live sequence, the pad slot 0, and one more for the probe's extra row, each of
        # max_ctx tokens: the longest context the pool holds, in whole units, plus the guard unit
        units = min(num_blocks - 1, -(-(s.max_positions or num_blocks * s.page_tokens) // s.page_tokens)) + 1
        self.max_ctx, self.region_slots = units * s.page_tokens, s.max_seqs + 2
        self.region = vmm.Vmm(s.device.index or 0) if self.cuda else vmm.Fake(granularity=self.unit_bytes)
        self.region.reserve(self.region_slots * units * self.unit_bytes)
        self.region.alias(0, self.vmm, 0)  # the pad slot reads the pad unit
        self.slot_ids = {}  # slot -> the block ids mapped into its region, by index
        shape = (num_blocks, s.page_tokens, s.layers, 2, s.kv_heads, s.head_dim)
        if not self.cuda:
            self.region_kv = None  # the torch backend gathers from the home
            return torch.zeros(shape, dtype=s.dtype)
        self.region_kv = self.region.tensor((self.region_slots * self.max_ctx, s.layers, 2, s.kv_heads, s.head_dim), s.dtype)
        return self.vmm.tensor(shape, s.dtype)

    def k_cache(self, layer):
        return self.kv[:, :, layer, 0]  # strided: the write kernels take block, page and head strides

    def v_cache(self, layer):
        return self.kv[:, :, layer, 1]

    def region_k(self, layer):
        """This layer's K over every region, [slots x max_ctx, kv heads, head dim] with a token stride."""
        return self.region_kv[:, layer, 0]

    def region_v(self, layer):
        return self.region_kv[:, layer, 1]

    def write(self, layer, key, value, slots):
        s = self.spec
        flat = self.kv.view(-1, s.layers, 2, s.kv_heads, s.head_dim)  # token-major: a token's K and V are one run
        flat[slots, layer, 0] = key.to(s.dtype)
        flat[slots, layer, 1] = value.to(s.dtype)

    def copy(self, src, dst, k):
        self.kv[dst, :k].copy_(self.kv[src, :k])

    def before_step(self, step, maps):
        """The rows' regions hold their blocks: a slot an admission took this step drops the mappings of the sequence
        before (its last reader settled a commit ago: the scheduler's cooling), every row's blocks beyond what its
        region holds are mapped at their index, and the pad unit sits after the last one as the guard."""
        unit, per_slot = self.unit_bytes, self.max_ctx // self.spec.page_tokens
        for s in step.slot_resets:
            have = self.slot_ids.get(s)
            if have:  # the sequence before: its units and the guard after them
                for k in range(len(have) + 1):
                    self.region.unalias(s * per_slot * unit + k * unit)
            self.slot_ids[s] = []
        for r in step.rows:
            have = self.slot_ids.setdefault(r.slot, [])
            if len(r.block_ids) <= len(have):
                continue
            base = r.slot * per_slot * unit
            if have:
                self.region.unalias(base + len(have) * unit)  # the guard after the units it had
            for k in range(len(have), len(r.block_ids)):
                self.region.alias(base + k * unit, self.vmm, r.block_ids[k] * unit)
            self.region.alias(base + len(r.block_ids) * unit, self.vmm, PAD_BLOCK * unit)  # the guard: the pad unit
            have += r.block_ids[len(have):]

    def probe_end(self, token):
        if self.slot_ids:  # the probe's regions, dropped
            if self.cuda:
                torch.cuda.synchronize(self.spec.device)
            self.before_step(Step([], slot_resets=list(self.slot_ids)), ())
