"""Fused layers: our Qwen3 decoder forward over HF's loaded weights, calling vLLM's fused CUDA ops. Makes no decisions.

HF's Qwen3 layer launches ~57 kernels (each RMSNorm is 8 elementwise kernels and a layer has four; RoPE is 10; q, k
and v are three GEMMs), so an 8B step is ~2,100 launches and the host, not the GPU, sets its time. This forward runs
the same arithmetic in ~11 launches a layer:

    fused_add_rms_norm (residual add + input norm) -> one QKV GEMM -> fused_qk_norm_rope (both head norms and RoPE,
    in place) -> reshape_and_cache_flash (the K/V write) -> FlashAttention -> o_proj -> fused_add_rms_norm ->
    one gate/up GEMM -> silu_and_mul -> down_proj

The ops are vLLM's `_C` kernels (vllm._custom_ops and torch.ops._C), imported as kernels only: no vLLM engine, model
or scheduler code. Weights stay HF's: fuse_projections concatenates q/k/v and gate/up once and points HF's Linear
weights at views of the concatenated tensors, so nothing is held twice and HF's forward still runs, unchanged, as the
oracle. Qwen3 as released only; check_fusable refuses anything else by name.
"""
import torch
import torch.nn.functional as F

FUSABLE_HEAD_DIMS = (64, 128, 256)  # the head sizes vLLM's fused_qk_norm_rope is built for


def fusion_problems(cfg):
    """How a config's layers differ from the Qwen3 layers this forward implements; empty when they do not."""
    rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    problems = [what for what, bad in (
        (f"model_type {cfg.model_type!r} (fused layers implement qwen3)", cfg.model_type != "qwen3"),
        (f"hidden_act {getattr(cfg, 'hidden_act', None)!r} (silu)", getattr(cfg, "hidden_act", None) != "silu"),
        ("attention_bias (none)", bool(getattr(cfg, "attention_bias", False))),
        (f"rope_type {rope.get('rope_type', rope.get('type'))!r} (default)",
         rope.get("rope_type", rope.get("type", "default")) != "default"),
        ("sliding-window layers (full attention only)", bool(getattr(cfg, "use_sliding_window", False))
         or any(t != "full_attention" for t in getattr(cfg, "layer_types", None) or ())),
        (f"head_dim {head_dim} (one of {FUSABLE_HEAD_DIMS})", head_dim not in FUSABLE_HEAD_DIMS),
    ) if bad]
    return problems


def check_fusable(cfg):
    """Refuses, naming every difference, a config whose layers are not the Qwen3 layers this forward implements."""
    problems = fusion_problems(cfg)
    if problems:
        raise ValueError(f"fused_layers cannot run this model: {'; '.join(problems)}")


def fuse_projections(model):
    """Concatenates each layer's q/k/v and gate/up weights along the output dim (vLLM's [q; k; v] and [gate; up]
    layouts) and re-points HF's Linear weights to views of them. Idempotent, so executors can share a model."""
    for layer in model.model.layers:
        attn, mlp = layer.self_attn, layer.mlp
        if not hasattr(attn, "qkv_weight"):
            attn.qkv_weight = _concat(attn.q_proj, attn.k_proj, attn.v_proj)
        if not hasattr(mlp, "gate_up_weight"):
            mlp.gate_up_weight = _concat(mlp.gate_proj, mlp.up_proj)


def _concat(*linears):
    w = torch.cat([lin.weight.detach() for lin in linears])
    offset = 0
    for lin in linears:
        n = lin.weight.shape[0]
        lin.weight = torch.nn.Parameter(w[offset:offset + n], requires_grad=False)  # a view: the old tensor is freed
        offset += n
    return w


class FusedQwen3:
    """The forward. It reads the executor's KV pool and FlashAttention call at every step (the pool is replaced once,
    after the sizing probe), and takes the same step inputs as HF layers do (executor._Batch)."""

    def __init__(self, model, executor, qk_norm_rope="fused"):
        """qk_norm_rope: "fused" (one kernel for both head norms and RoPE) or "split" (two rms_norm launches and
        rotary_embedding), the check that the fused kernel matches."""
        from vllm import _custom_ops as ops  # loads vLLM's kernel library only
        self.ops, self.silu_and_mul = ops, torch.ops._C.silu_and_mul
        self.ex, self.qk_norm_rope = executor, qk_norm_rope
        cfg, inner = model.config, model.model
        self.eps = cfg.rms_norm_eps
        self.heads, self.kv_heads, self.head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, executor.head_dim
        self.intermediate = cfg.intermediate_size
        self.embed, self.final_norm, self.lm_head = inner.embed_tokens.weight, inner.norm.weight, model.lm_head.weight
        self.scaling = inner.layers[0].self_attn.scaling
        self.layers = [(l.input_layernorm.weight, l.self_attn.qkv_weight, l.self_attn.q_norm.weight,
                        l.self_attn.k_norm.weight, l.self_attn.o_proj.weight, l.post_attention_layernorm.weight,
                        l.mlp.gate_up_weight, l.mlp.down_proj.weight) for l in inner.layers]
        rot = inner.rotary_emb
        if rot.rope_type != "default" or float(rot.attention_scaling) != 1.0:
            raise ValueError(f"fused layers build plain RoPE; the model's is {rot.rope_type!r} scaled by "
                             f"{rot.attention_scaling}")
        # vLLM's cos/sin cache: [positions, head_dim], the cos half then the sin half, from HF's own inverse frequencies
        # (the neox layout, which is HF's rotate_half), in the model's dtype as vLLM keeps it
        device = self.embed.device
        freqs = torch.outer(torch.arange(executor.max_positions, dtype=torch.float32, device=device),
                            rot.inv_freq.float().to(device))
        self.cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(self.embed.dtype)
        self.one = torch.ones((), dtype=torch.float32, device=device)  # the KV write's scales (no quantisation)

    def logits(self, b):
        """[sampling rows, vocab] logits of one step; writes the step's K/V into the pool."""
        ops, eps = self.ops, self.eps
        ids, pos = b.input_ids[0], b.position_ids[0]
        n = ids.shape[0]
        h = F.embedding(ids, self.embed)
        act = torch.empty(n, self.intermediate, dtype=h.dtype, device=h.device)
        res = None
        for i, (w_in, w_qkv, w_qn, w_kn, w_o, w_post, w_gu, w_down) in enumerate(self.layers):
            if res is None:  # layer 0: the embedding is the residual; the norm writes a new tensor
                res, h = h, torch.empty_like(h)
                ops.rms_norm(h, res, w_in, eps)
            else:  # res += h (the previous layer's MLP output); h = norm(res) * w
                ops.fused_add_rms_norm(h, res, w_in, eps)
            q, k, v = self._qkv(F.linear(h, w_qkv), pos, w_qn, w_kn, n)
            ops.reshape_and_cache_flash(k, v, self.ex.kv[i, 0], self.ex.kv[i, 1], b.slot_mapping, "auto", self.one,
                                        self.one)
            o = self.ex._flash(q, i, b, self.scaling)
            h = F.linear(o.view(n, -1), w_o)
            ops.fused_add_rms_norm(h, res, w_post, eps)
            self.silu_and_mul(act, F.linear(h, w_gu))
            h = F.linear(act, w_down)
        ops.fused_add_rms_norm(h, res, self.final_norm, eps)
        if not b.all_sample:
            h = h[b.sample_idx]
        return F.linear(h, self.lm_head)

    def _qkv(self, qkv, pos, w_qn, w_kn, n):
        """Per-head RMSNorm on q and k, then RoPE; returns q [n, H, D], k and v [n, KVH, D] (views where possible)."""
        hq, hk, d = self.heads, self.kv_heads, self.head_dim
        if self.qk_norm_rope == "fused":
            self.ops.fused_qk_norm_rope(qkv, hq, hk, hk, d, self.eps, w_qn, w_kn, self.cos_sin, True, pos)
            q, k, v = qkv.split([hq * d, hk * d, hk * d], dim=-1)
            return q.view(n, hq, d), k.view(n, hk, d), v.view(n, hk, d)
        q, k, v = qkv.split([hq * d, hk * d, hk * d], dim=-1)
        qn, kn = torch.empty(n * hq, d, dtype=qkv.dtype, device=qkv.device), torch.empty(n * hk, d, dtype=qkv.dtype,
                                                                                            device=qkv.device)
        self.ops.rms_norm(qn, q.reshape(n * hq, d), w_qn, self.eps)
        self.ops.rms_norm(kn, k.reshape(n * hk, d), w_kn, self.eps)
        self.ops.rotary_embedding(pos, qn.view(n, hq * d), kn.view(n, hk * d), d, self.cos_sin, True)
        return qn.view(n, hq, d), kn.view(n, hk, d), v.view(n, hk, d)
