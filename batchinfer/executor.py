"""Executor: runs one Step on the model and returns the sampled ids. Makes no decisions.

The weights are HF transformers' own. By default so are the layers (norms, q/k-norm, RoPE, MLP, lm_head), and only
attention is ours: it is registered with HF as "batchinfer_paged", writes each step's new K/V into a paged pool and
attends over the pool through a block table. fused_layers=True runs our Qwen3 layer forward over the same weights
with vLLM's fused kernels instead (qwen3_fused.py). A step is one packed [1, N] forward over every row, prefill
chunks and decode rows alike; logits are computed only at sampling rows. cuda_graphs=True captures the forward of a
decode-only step once per row count (GRAPH_ROWS) at load and replays it: a replay is one launch instead of ~400 (fused
layers) or ~2,100 (HF layers). Mixed steps, whose shape changes every step, run eagerly.

Two backends for the same attention:
- fa: vLLM's vendored FlashAttention-2 varlen kernel with block_table (page size 16). Only the kernel
  is imported: no vLLM engine, scheduler or cache code.
- torch: gather each row's KV through its block table and run SDPA. For CPU tests, and the oracle for fa.
"""
from .model import load_model, stop_token_ids  # before torch: sets the CUDA allocator config

import bisect  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .kv import BLOCK_SIZE, PAD_BLOCK  # noqa: E402
from .scheduler import MAX_SEQS  # noqa: E402
from .schema import MAX_PREFILL_BUDGET, Row, Step  # noqa: E402
from .sizing import pool_blocks  # noqa: E402

GiB = 2**30
ATTN = "batchinfer_paged"
# The auto sizing's measurement pool: the pad block, one scratch block every decode row writes, the prefill chunk's.
SCRATCH_POOL_BLOCKS = 2 + -(-MAX_PREFILL_BUDGET // BLOCK_SIZE)
# Decode-only row counts a graph is captured for (vLLM's capture sizes). A step with R decode rows replays the smallest
# one >= R; the extra rows are pad rows. Steps with more rows than the largest run eagerly.
GRAPH_ROWS = (1, 2, 4, *range(8, 256, 8), *range(256, 513, 16))


# What an HF attention layer can ask of its attention function per call, beyond full causal attention. Qwen3 passes
# sliding_window=None on full-attention layers; a layer that passes a value is asking for something this kernel does
# not do, so it is refused rather than given full causal attention without a word.
UNSUPPORTED_KWARGS = ("sliding_window", "softcap", "s_aux")  # windowed attention, logit soft-capping, attention sinks


def _paged_attention(module, query, key, value, attention_mask, scaling=None, batchinfer_batch=None, **kwargs):
    """HF calls this per layer with the kwargs given to the model's forward, so the step's batch arrives as one
    of them (Executor.logits passes it); nothing about the step lives outside that call. The executor's probe is
    the first such call, so an unsupported layer fails at load."""
    asked = {k: kwargs[k] for k in UNSUPPORTED_KWARGS if kwargs.get(k) is not None}
    if asked:
        raise NotImplementedError(f"{ATTN} implements full causal attention only; layer {module.layer_idx} asks "
                                  f"for {asked}")
    if batchinfer_batch is None:
        raise RuntimeError(f"{ATTN} attention needs the step's batch: run the model through Executor.logits")
    return batchinfer_batch.executor.attend(module.layer_idx, query, key, value, scaling, batchinfer_batch), None


def register():
    """Registers the attention function only. Causality lives in the kernel, and no mask function is registered on
    purpose: HF skips mask creation for an attention implementation it has no mask function for (masking_utils.py,
    create_causal_mask), while a registered one, even a no-op, makes it check position_ids for packed sequences with a
    host sync on every forward."""
    from transformers import AttentionInterface
    AttentionInterface.register(ATTN, _paged_attention)


def check_fa_version(fa_version, device):
    """FlashAttention 2 everywhere; 3 (what vLLM runs on Hopper) only on an sm90 card where vLLM's build has it."""
    if fa_version not in (2, 3) or isinstance(fa_version, bool):
        raise ValueError(f"fa_version must be 2 or 3, not {fa_version!r}")
    if fa_version == 3:
        from vllm.vllm_flash_attn import is_fa_version_supported
        if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 9 or not is_fa_version_supported(3):
            raise ValueError("fa_version=3 needs an sm90 (Hopper) GPU and vLLM's FlashAttention-3 build")


def check_supported(cfg, model_name):
    """What the per-call check in _paged_attention cannot see: query heads must split evenly over KV heads."""
    if cfg.num_attention_heads % cfg.num_key_value_heads:
        raise ValueError(f"{model_name}: {ATTN} needs query heads in whole groups per KV head; the config has "
                         f"{cfg.num_attention_heads} heads over {cfg.num_key_value_heads} KV heads")


class _Batch:
    """Device tensors for one step, built from a Step's rows, and the executor that runs it."""

    def __init__(self, step, executor):
        self.executor, device = executor, executor.device
        ids, pos, slots, cu, used, table, sample = [], [], [], [0], [], [], []
        width = max(len(r.block_ids) for r in step.rows)
        for r in step.rows:
            n = len(r.token_ids)
            ids += r.token_ids
            for p in range(r.start, r.start + n):
                slots.append(r.block_ids[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE)
            pos += range(r.start, r.start + n)
            cu.append(cu[-1] + n)
            used.append(r.start + n)
            table.append(r.block_ids + [PAD_BLOCK] * (width - len(r.block_ids)))
            if r.sample:
                sample.append(cu[-1] - 1)
        t = lambda x, dt: torch.tensor(x, dtype=dt, device=device)  # noqa: E731
        self.input_ids, self.position_ids = t([ids], torch.long), t([pos], torch.long)
        self.slot_mapping = t(slots, torch.long)
        self.cu_seqlens_q, self.seqused_k = t(cu, torch.int32), t(used, torch.int32)
        self.block_table = t(table, torch.int32)
        self.sample_idx = t(sample, torch.long)
        self.all_sample = len(sample) == len(ids)  # every row one sampled token: a decode-only step
        self.max_seqlen_q = max(len(r.token_ids) for r in step.rows)
        self.max_seqlen_k = max(used)
        self.rows = [(cu[i], cu[i + 1], used[i], r.block_ids) for i, r in enumerate(step.rows)]  # torch backend


class _GraphInputs:
    """A captured graph's step inputs: views of the executor's persistent buffers, the same fields as _Batch."""

    def __init__(self, ex, rows):
        big = ex.graph_sizes[-1]
        self.executor = ex
        self.input_ids = ex.g_tokens[:rows].view(1, rows)
        self.position_ids = ex.g_tokens[big:big + rows].view(1, rows)
        self.slot_mapping = ex.g_tokens[2 * big:2 * big + rows]
        self.cu_seqlens_q, self.seqused_k = ex.g_cu[:rows + 1], ex.g_used[:rows]
        self.block_table = ex.g_table[:rows]
        self.sample_idx, self.all_sample = None, True
        self.max_seqlen_q, self.max_seqlen_k = 1, ex.k_cap


def graph_problems(model, backend, fa_version):
    """What a captured graph cannot hold: the torch backend's per-row Python loop, FlashAttention-3 (whose split
    scheduling vLLM precomputes outside the graph every step; not built here, so not captured), and RoPE that reads the
    largest position back to the host (dynamic and longrope scaling). Empty when a graph can hold the step."""
    rope = getattr(getattr(getattr(model, "model", None), "rotary_emb", None), "rope_type", "default")
    return [what for what, bad in (
        ("cuda_graphs captures the fa backend's kernels: it needs the GPU and the fa backend", backend != "fa"),
        ("cuda_graphs captures FlashAttention-2 only: FlashAttention-3 under graphs needs its scheduler metadata "
         "computed per step outside the graph, which is not built", fa_version != 2),
        (f"cuda_graphs cannot capture {rope!r} RoPE: it reads the largest position back to the host",
         "dynamic" in rope or rope == "longrope"),
    ) if bad]


def check_graphable(model, backend, fa_version):
    problems = graph_problems(model, backend, fa_version)
    if problems:
        raise ValueError("; ".join(problems))


def resolve_build(name, value, problems):
    """True / False as given (True refused with the reasons it cannot run), or "auto": on exactly where it can run."""
    if value == "auto":
        return not problems
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true, false or 'auto', not {value!r}")
    if value and problems:
        raise ValueError(f"{name}: {'; '.join(problems)}")
    return value


class Executor:
    def __init__(self, model_name, *, dtype="bfloat16", device=None, backend=None, reserve_gb="auto", num_blocks=None,
                 seed=None, model=None, probe=True, fa_version=2, fused_layers=False, cuda_graphs=False,
                 graph_max_rows=GRAPH_ROWS[-1]):
        """num_blocks: the KV pool size in blocks (block 0 is the pad block), the same on every card. Without it the
        pool is sized on this card: reserve_gb="auto" measures the largest step's activations with the worst-step
        probe over a small pool and sizes the pool so the run's device peak is at most sizing.MAX_PEAK_SHARE of the
        card (sizing.pool_blocks); a number leaves that many GiB of the free memory after load for activations.
        probe: run the worst-step probe once more over the final pool (auto measures either way). seed shuffles
        the block free list (see kv.BlockAllocator) for every run; tests and the gate use it. fa_version: the
        FlashAttention version of the fa backend (check_fa_version). fused_layers: Qwen3's layers as one fused forward
        over HF's weights with vLLM's kernels (qwen3_fused.py; GPU and fa backend only). cuda_graphs: replay decode-only
        steps of up to graph_max_rows rows from graphs captured at load (GPU, fa backend, FlashAttention-2). Both take
        True, False or "auto" (on exactly where they can run); self.facts records what was resolved."""
        register()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cuda = self.device.type == "cuda"
        if num_blocks is None and not self.cuda:
            raise ValueError("num_blocks is required off the GPU: by default the KV pool is sized from free GPU memory")
        self.backend = backend or ("fa" if self.cuda else "torch")
        self.fa_version = fa_version
        if self.backend == "fa":
            check_fa_version(fa_version, self.device)
            from vllm.vllm_flash_attn import flash_attn_varlen_func
            self._fa = flash_attn_varlen_func
        if model is None:
            model = load_model(model_name, getattr(torch, dtype), ATTN)
        elif hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation(ATTN)
        else:
            model.config._attn_implementation = ATTN
        check_supported(model.config, model_name)
        from .qwen3_fused import fusion_problems
        fused_layers = resolve_build("fused_layers", fused_layers, fusion_problems(model.config) + (
            [] if self.backend == "fa" else ["fused_layers runs vLLM's CUDA kernels: it needs the GPU and the fa backend"]))
        cuda_graphs = resolve_build("cuda_graphs", cuda_graphs, graph_problems(model, self.backend, fa_version))
        # what this executor runs, as the engine records it (metrics.executor): the resolved settings, never "auto"
        self.facts = {"fused_layers": fused_layers, "cuda_graphs": cuda_graphs,
                      "fa_version": fa_version if self.backend == "fa" else None}
        self.model = model.to(self.device).eval()
        self.seed = seed
        cfg = self.model.config
        self.layers, self.kv_heads = cfg.num_hidden_layers, cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.n_rep = cfg.num_attention_heads // self.kv_heads
        self.stop_ids = stop_token_ids(self.model)
        self.max_positions = getattr(cfg, "max_position_embeddings", None)  # the scheduler refuses requests past it
        self.model_shape = {"hidden": cfg.hidden_size, "vocab": cfg.vocab_size,
                            "n_body_params": sum(p.numel() for n, p in self.model.named_parameters()
                                                 if "embed_tokens" not in n and not n.startswith("lm_head"))}
        self.dtype = next(self.model.parameters()).dtype
        self.bytes_per_block = 2 * self.layers * BLOCK_SIZE * self.kv_heads * self.head_dim * self.dtype.itemsize
        self.memory_facts = {}  # what the auto sizing measured; the batchinfer engine records it in metrics.memory
        self.fused = None
        if fused_layers:  # before sizing, so the probe measures the fused forward's activations
            from .qwen3_fused import FusedQwen3, fuse_projections
            fuse_projections(self.model)
            torch.cuda.empty_cache()  # the unfused projection tensors, freed
            self.fused = FusedQwen3(self.model, self)
        self.graph_sizes = tuple(r for r in GRAPH_ROWS if r <= min(graph_max_rows, MAX_SEQS)) if cuda_graphs else ()
        self.graphs = {}  # graph rows -> (CUDAGraph, its sampled ids [rows]); filled after the pool is allocated
        self.graph_rows = 0  # the last forward's: the graph rows it replayed, 0 if it ran eagerly
        if num_blocks is None and reserve_gb == "auto":
            num_blocks = self._measure_pool()
        elif num_blocks is None:
            free, _ = torch.cuda.mem_get_info(self.device)
            num_blocks = int((free - float(reserve_gb) * GiB) // self.bytes_per_block)
        self.num_blocks = num_blocks
        self.kv = self._pool(num_blocks)
        if self.graph_sizes:  # before the probe, so it checks the card with the graphs' memory held
            self._graph_buffers(num_blocks)
            self._capture()
        self.probe_peak_reserved_gb = self.probe() if probe and self.cuda else None

    def _pool(self, num_blocks):
        # [layers, K|V, blocks, block slots, kv heads, head dim]: kv[l, 0] and kv[l, 1] are contiguous pools
        return torch.zeros((self.layers, 2, num_blocks, BLOCK_SIZE, self.kv_heads, self.head_dim),
                           dtype=self.dtype, device=self.device)

    def _probe_step(self, scratch=False):
        """The widest step the scheduler can build: MAX_SEQS sampled decode rows (a [rows, vocab] logits tensor, ~0.3
        GiB for Qwen3 at 1,024 rows) plus a MAX_PREFILL_BUDGET-token prefill chunk. Over the real pool every decode row
        writes a block of its own, and a pool too small for that gets the largest such step it can hold. scratch: every
        decode row writes block 1 instead, so the whole step fits SCRATCH_POOL_BLOCKS; the probe's KV is garbage either
        way, and its activations do not depend on which blocks the rows write."""
        rows = MAX_SEQS if scratch else min(MAX_SEQS, (self.num_blocks - 1) // 2)
        first = 2 if scratch else 1 + rows  # the prefill chunk's first block
        n = min(MAX_PREFILL_BUDGET, (self.num_blocks - first) * BLOCK_SIZE)
        return Step([Row(seq=i, start=0, token_ids=[0], block_ids=[1 if scratch else 1 + i], sample=True, decode=True)
                     for i in range(rows)]
                    + [Row(seq=rows, start=0, token_ids=[0] * n, block_ids=list(range(first, first + -(-n // BLOCK_SIZE))),
                           sample=True, decode=False)])

    def _step_peak(self, scratch=False):
        """Runs the widest-step probe. Returns (torch's peak reserved, what the process holds outside torch, the
        card's total), in bytes."""
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.forward(self._probe_step(scratch))
        torch.cuda.synchronize(self.device)
        peak = torch.cuda.max_memory_reserved(self.device)
        free, total = torch.cuda.mem_get_info(self.device)
        outside = total - free - torch.cuda.memory_reserved(self.device)  # CUDA context and library handles
        torch.cuda.reset_peak_memory_stats(self.device)
        return peak, outside, total

    def _measure_pool(self):
        """reserve_gb="auto": the widest-step probe over a scratch pool (SCRATCH_POOL_BLOCKS, so the measurement fits
        where the model barely does) measures the largest step's activations on this card; sizing.pool_blocks then
        sizes the real pool around them."""
        self.num_blocks, self.kv = SCRATCH_POOL_BLOCKS, self._pool(SCRATCH_POOL_BLOCKS)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        before = torch.cuda.memory_reserved(self.device)
        peak, outside, total = self._step_peak(scratch=True)
        activation = peak - before
        graph = 0
        if self.graph_sizes:  # every graph captured as the run will, over the scratch pool: their memory, held all run
            self._graph_buffers(SCRATCH_POOL_BLOCKS)
            graph = self._capture()
            self.graphs, self.graph_pool = {}, None
            self.memory_facts["graph_sized_gb"] = round(graph / GiB, 2)
        self.kv = None
        torch.cuda.empty_cache()
        in_use = torch.cuda.memory_reserved(self.device) + outside
        self.memory_facts.update(in_use_after_load_gb=round(in_use / GiB, 2), step_activation_gb=round(activation / GiB, 2))
        return pool_blocks(total / GiB, in_use / GiB, activation / GiB, self.bytes_per_block / GiB, graph_gb=graph / GiB)

    def probe(self):
        """The widest-step probe over the real pool before any real work, so activation memory at the largest step
        the scheduler can build is checked against this card now rather than mid-job. It is the run's own worst
        step, so it records its peak (torch's reserved GiB) and fails only on OOM."""
        try:
            peak, outside, _ = self._step_peak()
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(f"memory probe at {MAX_SEQS} sampled decode rows + a {MAX_PREFILL_BUDGET}-token prefill "
                               f"chunk ran out of memory with a {self.num_blocks}-block KV pool; pass a smaller num_blocks "
                               f"or a larger reserve_gb") from e
        self.memory_facts["probe_peak_device_gb"] = round((peak + outside) / GiB, 2)  # what sizing's limit is about
        return round(peak / GiB, 2)

    # decode-only steps from CUDA graphs --------------------------------------------------------------

    def _graph_buffers(self, num_blocks):
        """The persistent inputs every graph reads, at the largest graph's size: token ids, positions and KV slots in one
        int64 buffer, seqused_k, and a block table as wide as the longest context the pool can hold (k_cap, the
        max_seqlen_k baked into every graph, as vLLM bakes max_model_len for FlashAttention 2). Pinned host copies are
        filled per step and copied in; a replay reads the same addresses it was captured with."""
        rows, dev = self.graph_sizes[-1], self.device
        self.k_cap = min(self.max_positions or BLOCK_SIZE * (num_blocks - 1), BLOCK_SIZE * (num_blocks - 1))
        width = -(-self.k_cap // BLOCK_SIZE)
        self.g_tokens = torch.zeros(3 * rows, dtype=torch.long, device=dev)  # ids | positions | slots
        self.g_used = torch.zeros(rows, dtype=torch.int32, device=dev)
        self.g_cu = torch.arange(rows + 1, dtype=torch.int32, device=dev)  # one query token per row, always
        self.g_table = torch.zeros((rows, width), dtype=torch.int32, device=dev)
        self.g_table_in = torch.zeros(rows * width, dtype=torch.int32, device=dev)  # the step's [R, w] block, packed
        self.h_tokens = torch.zeros(3 * rows, dtype=torch.long, pin_memory=True)
        self.h_used = torch.zeros(rows, dtype=torch.int32, pin_memory=True)
        self.h_table = torch.zeros(rows * width, dtype=torch.int32, pin_memory=True)
        self.h_tokens_np, self.h_used_np, self.h_table_np = self.h_tokens.numpy(), self.h_used.numpy(), self.h_table.numpy()

    def _capture(self):
        """Captures every graph size, largest first, into one shared memory pool. Returns the device memory it added:
        the pool's segments plus what the driver holds for the instantiated graphs (~0.3 GiB for 51 graphs of HF
        layers' ~2,200 nodes each), so it is read from the card's free memory, not torch's reserved. Records that as
        graph_held_gb (the auto sizing's own capture, over its scratch pool, is graph_sized_gb) and adds its time to
        graph_capture_s: with reserve_gb="auto" a load captures twice."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()  # torch.cuda.graph empties the cache as it starts: measure from an empty one
        t0, free_before = time.perf_counter(), torch.cuda.mem_get_info(self.device)[0]
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.graphs = {rows: self._capture_one(rows) for rows in reversed(self.graph_sizes)}
        torch.cuda.synchronize(self.device)
        self._graph_kv_ptr = self.kv.data_ptr()
        added = free_before - torch.cuda.mem_get_info(self.device)[0]
        self.memory_facts.update(graph_capture_s=round(self.memory_facts.get("graph_capture_s", 0)
                                                       + time.perf_counter() - t0, 2),
                                 graph_held_gb=round(added / GiB, 2))
        return added

    @torch.inference_mode()
    def _capture_one(self, rows):
        """One warmup, then the capture of this step (every row a pad row, which the graph does not depend on), both on
        the executor's one capture stream: library handles and lazy initialisation happen in the warmup, outside the
        graph, and once per stream (cuBLAS keeps a workspace per stream, ~32 MiB each)."""
        if getattr(self, "_capture_stream", None) is None:
            self._capture_stream = torch.cuda.Stream(self.device)
        side = self._capture_stream
        self._stage_pads(rows)
        b = _GraphInputs(self, rows)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            self._logits(b).argmax(-1)
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool, stream=side):
            ids = self._logits(b).argmax(-1)
        return graph, ids

    def _stage_pads(self, rows):
        """All `rows` rows as pad rows, straight on the device: token 0 at position 0, writing slot 0 of the pad block
        (never read by a real row), attending over nothing (seqused_k 0; FlashAttention writes zeros)."""
        big = self.graph_sizes[-1]
        for k in range(3):
            self.g_tokens[k * big:k * big + rows].zero_()
        self.g_used[:rows].zero_()

    def _graph_size(self, step):
        """The graph rows this step replays: the smallest captured size >= its row count if every row is a decode row,
        else 0 (eager)."""
        n = len(step.rows)
        if not self.graphs or n > self.graph_sizes[-1] or not all(r.decode for r in step.rows):
            return 0
        return self.graph_sizes[bisect.bisect_left(self.graph_sizes, n)]

    def _stage(self, step, rows):
        """Copies a decode-only step into the graph inputs: its rows first, pad rows after, and each row's table up to
        the block holding its position (FlashAttention reads no further; the rest of a row is left as it was)."""
        big, n = self.graph_sizes[-1], len(step.rows)
        t, u = self.h_tokens_np, self.h_used_np
        width = 0
        for i, r in enumerate(step.rows):
            p = r.start
            t[i], t[big + i], t[2 * big + i] = r.token_ids[0], p, r.block_ids[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE
            u[i] = p + 1
            width = max(width, p // BLOCK_SIZE + 1)
        for k in range(3):
            t[k * big + n:k * big + rows] = 0
        u[n:rows] = 0
        table = self.h_table_np[:n * width].reshape(n, width)
        for i, r in enumerate(step.rows):
            k = r.start // BLOCK_SIZE + 1
            table[i, :k] = r.block_ids[:k]
        self.g_tokens.copy_(self.h_tokens, non_blocking=True)
        self.g_used[:rows].copy_(self.h_used[:rows], non_blocking=True)
        self.g_table_in[:n * width].copy_(self.h_table[:n * width], non_blocking=True)
        self.g_table[:n, :width].copy_(self.g_table_in[:n * width].view(n, width))

    # ---------------------------------------------------------------------------------------------

    def attend(self, layer, query, key, value, scaling, b):
        """query [1, H, N, D], key/value [1, KVH, N, D] (HF layout, RoPE applied) -> [1, N, H, D]. b: the step's _Batch."""
        q = query[0].transpose(0, 1)
        k_pool, v_pool = self.kv[layer, 0], self.kv[layer, 1]
        k_pool.view(-1, self.kv_heads, self.head_dim)[b.slot_mapping] = key[0].transpose(0, 1)
        v_pool.view(-1, self.kv_heads, self.head_dim)[b.slot_mapping] = value[0].transpose(0, 1)
        if self.backend == "fa":
            out = self._flash(q, layer, b, scaling)
        else:
            out = torch.cat([self._attend_torch(q[q0:q1], k_pool, v_pool, used, blocks, scaling)
                             for q0, q1, used, blocks in b.rows])
        return out.unsqueeze(0)

    def _flash(self, q, layer, b, scaling):
        """q [N, H, D] (last dim contiguous) -> [N, H, D]: FlashAttention over this layer's pool through the step's block
        table. The one attention call of the fa backend, for HF layers and fused layers alike."""
        return self._fa(q, self.kv[layer, 0], self.kv[layer, 1], max_seqlen_q=b.max_seqlen_q, cu_seqlens_q=b.cu_seqlens_q,
                        max_seqlen_k=b.max_seqlen_k, seqused_k=b.seqused_k, causal=True, softmax_scale=scaling,
                        block_table=b.block_table, fa_version=self.fa_version)

    def _attend_torch(self, q, k_pool, v_pool, used, blocks, scaling):
        """One row: queries at positions used - n .. used - 1 attend to KV positions 0 .. their own."""
        n = q.shape[0]
        ids = torch.tensor(blocks[:-(-used // BLOCK_SIZE)], device=q.device)
        k = k_pool[ids].flatten(0, 1)[:used].repeat_interleave(self.n_rep, dim=1)
        v = v_pool[ids].flatten(0, 1)[:used].repeat_interleave(self.n_rep, dim=1)
        mask = torch.ones(n, used, dtype=torch.bool, device=q.device).tril(diagonal=used - n)
        out = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
                                             attn_mask=mask, scale=scaling)
        return out.transpose(0, 1)

    @torch.inference_mode()
    def logits(self, step):
        """[sampling rows, vocab] logits of one step, in row order. Writes the step's K/V into the pool."""
        return self._logits(_Batch(step, self))

    def _logits(self, b):
        """The forward over one step's inputs b: fused layers or HF's."""
        if self.fused is not None:
            return self.fused.logits(b)
        return self.model(input_ids=b.input_ids, position_ids=b.position_ids, use_cache=False,
                          logits_to_keep=0 if b.all_sample else b.sample_idx, batchinfer_batch=b).logits[0]

    def forward(self, step):
        """Returns (one greedy id per sampling row in row order, GPU milliseconds or None on CPU). self.graph_rows says
        whether the step replayed a graph (its row count) or ran eagerly (0)."""
        if not self.cuda:
            return self.logits(step).argmax(-1).tolist(), None
        self.graph_rows = self._graph_size(step)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        if self.graph_rows:
            assert self.kv.data_ptr() == self._graph_kv_ptr, "the KV pool moved after capture: the graphs read the old one"
            self._stage(step, self.graph_rows)  # the pinned buffers are free: the previous forward ended in a sync
            graph, ids = self.graphs[self.graph_rows]
            graph.replay()
            sampled = ids[:len(step.rows)]
        else:
            sampled = self.logits(step).argmax(-1)
        end.record()
        ids = sampled.tolist()  # the one host sync per step: stop checks need the ids
        # ... except when no row samples (long prompts mid-chunk, nothing decoding): an empty tolist() copies nothing
        # and does not wait for the GPU, so wait for the end event itself before timing it
        end.synchronize()
        return ids, start.elapsed_time(end)
