"""How attention is computed over the pool: one call per layer and step, for HF's layers and the fused layers alike.

Three ways, the executor's attention knob: fa, vLLM's FlashAttention varlen kernel (2, or 3 on sm90) through the
step's block table, or without one through the rows' regions under pool=contiguous (the kernels take each row's key
start from cu_seqlens_k and its length from seqused_k, which vLLM's wrapper refuses together, so that is a call into
the same ops with the wrapper's argument lists at 0.30.0); cascade, two FlashAttention calls merged exactly
(cascade.py); torch, SDPA per row over KV gathered through the block table, the CPU path and the oracle for the
other two. An fp8 pool is read with q quantised at the pool's unit scale and the kernel's descales expanded from it
(FlashAttention-3), or converted on the gather (torch).
"""
import torch
import torch.nn.functional as F

from . import cascade


class Attention:
    def __init__(self, backend, fa_version, kv_heads, n_rep, block_size, pool_dtype, model_dtype, kv_scale):
        """backend: "fa" (the kernel) or "torch"; fa_version 2 or 3 under fa; kv_scale: the fp8 pool's unit scale."""
        self.backend, self.fa_version = backend, fa_version
        self.kv_heads, self.n_rep, self.block_size = kv_heads, n_rep, block_size
        self.fp8 = pool_dtype != model_dtype
        self.kv_scale = kv_scale
        # pool=contiguous reads through the regions; False reads the same token-major units through the paged call
        # with the block table, the measurement arm that isolates the indirection (scripts/decode_step_profile.py)
        self.regions = True
        if backend == "fa":
            from vllm.vllm_flash_attn import flash_attn_varlen_func
            self._fa = flash_attn_varlen_func

    def run(self, q, pool, layer, b, scaling):
        """q [N, H, D] (last dim contiguous) -> [N, H, D] over this layer's KV, written already. b: the step's inputs
        (graphs.Inputs)."""
        if self.backend == "fa":
            if b.groups is not None:
                return self._cascade(q, pool, layer, b, scaling)
            return self._flash(q, pool, layer, b, scaling)
        k_pool, v_pool = pool.k_cache(layer), pool.v_cache(layer)
        if b.groups is not None:
            return torch.cat([self._cascade_torch(q[q0:q1], k_pool, v_pool, used, blocks, shared, prefix, scaling)
                              for q0, q1, used, blocks, shared, prefix in b.groups])
        return torch.cat([self._rows_torch(q[q0:q1], k_pool, v_pool, used, blocks, scaling) for q0, q1, used, blocks in b.rows])

    def _fp8(self, q, b):
        """An fp8 pool: q quantised at the unit scale, and the descales (per row and kv head) the kernel takes."""
        from vllm import _custom_ops as ops
        n, h, d = q.shape
        q = ops.scaled_fp8_quant(q.reshape(n, h * d).contiguous(), self.kv_scale)[0].view(n, h, d)
        return q, self.kv_scale.expand(b.seqused_k.shape[0], self.kv_heads)

    def _flash(self, q, pool, layer, b, scaling):
        extra = {} if b.scheduler_metadata is None else {"scheduler_metadata": b.scheduler_metadata, "num_splits": b.num_splits}
        if self.fp8:
            q, descale = self._fp8(q, b)
            extra.update(q_descale=descale, k_descale=descale, v_descale=descale)
        if pool.layout == "token" and self.regions:  # one key run per row, no block table
            return self._regions(q, pool.region_k(layer), pool.region_v(layer), b, scaling, extra)
        return self._fa(q, pool.k_cache(layer), pool.v_cache(layer), max_seqlen_q=b.max_seqlen_q, cu_seqlens_q=b.cu_seqlens_q,
                        max_seqlen_k=b.max_seqlen_k, seqused_k=b.seqused_k, causal=True, softmax_scale=scaling,
                        block_table=b.block_table, fa_version=self.fa_version, **extra)

    def _regions(self, q, k, v, b, scaling, extra):
        """FlashAttention over the regions: each row's keys start at cu_k[row] (its region) and run seqused_k[row]
        long; the argument lists mirror vllm/vllm_flash_attn/flash_attn_interface.py at 0.30.0. max_seqlen_k is the
        step's longest context (a graph's k_cap), as the paged call passes: the kernel sizes its KV split from it,
        and the region's full length there made the read slower than the paged one (FlashAttention-2)."""
        out, max_k = torch.empty_like(q), b.max_seqlen_k
        if self.fa_version == 2:
            assert not extra, "FlashAttention-2 takes no schedule and no descales"
            out, _ = torch.ops._vllm_fa2_C.varlen_fwd(q, k, v, out, b.cu_seqlens_q, b.cu_k, b.seqused_k, None, None, None,
                                                      b.max_seqlen_q, max_k, 0.0, scaling, False, True, -1, -1, 0.0, False, 0, None)
            return out
        out, _, _, _ = torch.ops._vllm_fa3_C.fwd(
            q, k, v, None, None, None, out, b.cu_seqlens_q, b.cu_k, None, None, b.seqused_k, b.max_seqlen_q, max_k,
            None, None, None, None, None, None, extra.get("q_descale"), extra.get("k_descale"), extra.get("v_descale"),
            scaling, True, -1, -1, 0.0, True, extra.get("scheduler_metadata"), extra.get("num_splits", 0), None, 0, None, 1, 0, None)
        return out

    def _cascade(self, q, pool, layer, b, scaling):
        """The groups' call over the queries in group order (each group's packed queries against its prefix, no causal
        mask), the rows' call over each row's own suffix, and vLLM's exact merge of the two."""
        from vllm import _custom_ops as ops
        k, v = pool.k_cache(layer), pool.v_cache(layer)
        extra = {} if b.scheduler_metadata is None else {"num_splits": b.num_splits}
        descales = lambda rows: {}  # noqa: E731
        if self.fp8:
            q, _ = self._fp8(q, b)
            descales = lambda rows: dict(zip(("q_descale", "k_descale", "v_descale"), [self.kv_scale.expand(rows, self.kv_heads)] * 3))  # noqa: E731
        sched = lambda m: {} if b.scheduler_metadata is None else {"scheduler_metadata": m}  # noqa: E731
        # max_seqlen_q is a bound; at 1 with GQA FlashAttention-2 takes its single-query specialisation, which reshapes
        # q by the number of cu_seqlens_q entries, one query each: the group call has groups with none (a graph's pad
        # group, a group whose rows all fell to pads), so the bound stays at 2 or more and the specialisation off
        out_g, lse_g = self._fa(q.index_select(0, b.perm), k, v, max_seqlen_q=max(b.max_seqlen_q_g, 2), cu_seqlens_q=b.cu_g,
                                max_seqlen_k=b.max_k_g, seqused_k=b.used_g, causal=False, softmax_scale=scaling,
                                block_table=b.table_g, fa_version=self.fa_version, return_softmax_lse=True,
                                **sched(b.scheduler_metadata_g), **extra, **descales(b.used_g.shape[0]))
        out_s, lse_s = self._fa(q, k, v, max_seqlen_q=b.max_seqlen_q, cu_seqlens_q=b.cu_seqlens_q, max_seqlen_k=b.max_k_s,
                                seqused_k=b.used_s, causal=True, softmax_scale=scaling, block_table=b.table_s,
                                fa_version=self.fa_version, return_softmax_lse=True,
                                **sched(b.scheduler_metadata), **extra, **descales(b.used_s.shape[0]))
        # a group over no keys (a row behind no prefix; the pad tokens): the non-causal kernel returns garbage for its
        # queries, not zeros with an infinite log-sum-exp (measured on FlashAttention-2), so they are made absent
        # here: lse +inf, which the merge reads as no weight, and a zero output under it. Tensor ops only (captured)
        group = torch.searchsorted(b.cu_g[1:], torch.arange(q.shape[0], device=q.device, dtype=b.cu_g.dtype), right=True)
        empty = b.used_g.index_select(0, group) == 0  # per query, in group order
        lse_g = lse_g.masked_fill(empty.unsqueeze(0), float("inf"))
        out_g = out_g.masked_fill(empty.unsqueeze(1).unsqueeze(2), 0)
        out = torch.empty_like(out_s)
        ops.merge_attn_states(out, out_g.index_select(0, b.inv), lse_g.index_select(1, b.inv), out_s, lse_s)
        return out

    def _gather(self, pool_t, ids, upto, dtype):
        """KV of positions 0..upto-1 from these blocks, as [upto, H, D] in the model's dtype (an fp8 pool converts)."""
        t = pool_t[torch.tensor(ids, device=pool_t.device, dtype=torch.long)].flatten(0, 1)[:upto]
        return t.to(dtype).repeat_interleave(self.n_rep, dim=1)

    def _rows_torch(self, q, k_pool, v_pool, used, blocks, scaling):
        """One row: queries at positions used - n .. used - 1 attend to KV positions 0 .. their own."""
        n = q.shape[0]
        ids = blocks[:-(-used // self.block_size)]
        k, v = self._gather(k_pool, ids, used, q.dtype), self._gather(v_pool, ids, used, q.dtype)
        mask = torch.ones(n, used, dtype=torch.bool, device=q.device).tril(diagonal=used - n)
        out = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask, scale=scaling)
        return out.transpose(0, 1)

    def _cascade_torch(self, q, k_pool, v_pool, used, blocks, shared, prefix, scaling):
        """One row of the cascade: its queries against the group's prefix (no mask) and against its own suffix
        (causal), merged by log-sum-exp. Equal to _rows_torch over the whole context."""
        bs = self.block_size
        kp, vp = (self._gather(k_pool, prefix, shared, q.dtype), self._gather(v_pool, prefix, shared, q.dtype)) if shared else (q[:0], q[:0])
        ks, vs = self._gather(k_pool, blocks[shared // bs:], used - shared, q.dtype), self._gather(v_pool, blocks[shared // bs:], used - shared, q.dtype)
        out_p, lse_p = cascade.lse_attention(q, kp, vp, scaling, causal=False)
        out_s, lse_s = cascade.lse_attention(q, ks, vs, scaling, causal=True)
        return cascade.merge(out_p, lse_p, out_s, lse_s)
