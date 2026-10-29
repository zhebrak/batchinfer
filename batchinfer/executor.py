"""Executor: runs one Step on the model and returns the sampled ids. Makes no decisions.

The weights are HF transformers' own. By default so are the layers (norms, q/k-norm, RoPE, MLP, lm_head), and only
attention is ours: it is registered with HF as "batchinfer_paged", writes each step's new K/V into a paged pool and
attends over the pool through a block table. layers=fused runs our Qwen3 layer forward over the same weights with
vLLM's fused kernels instead (qwen3_fused.py); layers=compiled runs HF's layers under torch.compile, the paged
attention an opaque custom op (paged_attention), for any decoder the checks accept; layers=hf runs HF's layers
eagerly, the non-optimised setup. A step is one packed [1, N] forward over every row, prefill
chunks and decode rows alike; logits are computed only at sampling rows. cuda_graphs=True captures graphs at load
and replays them: one launch instead of ~400 (fused layers) or ~2,100 (HF layers). Decode-only steps replay a graph
per row count (GRAPH_ROWS); under FlashAttention-3 steps carrying prefill replay a mixed graph too (graphs.py), the
kernel's tile schedule computed per step outside the graph; under FlashAttention-2 they run eagerly, as does any
step past the largest graph.

Three ways for the same attention:
- fa: vLLM's vendored FlashAttention varlen kernel (2, or 3 on sm90) with block_table. Only the kernel
  is imported: no vLLM engine, scheduler or cache code.
- torch: gather each row's KV through its block table and run SDPA. For CPU tests, and the oracle for fa.
- cascade: the two-call cascade (cascade.py) over the fa kernel, or over SDPA off the GPU as its oracle.
"""
from .model import load_model, stop_token_ids  # before torch: sets the CUDA allocator config

import math  # noqa: E402
import time  # noqa: E402
import weakref  # noqa: E402

import torch  # noqa: E402

from . import graphs  # noqa: E402
from .attention import Attention  # noqa: E402
from . import pool as pool_module  # noqa: E402
from .kv import DEFAULT_BLOCK_SIZE  # noqa: E402
from .schema import MAX_PREFILL_BUDGET, MAX_SEQS, Limits, Row, Step  # noqa: E402
from .sizing import pool_blocks  # noqa: E402

GiB = 2**30
ATTN = "batchinfer_paged"
# The auto sizing's measurement pool: the pad block, one scratch block every decode row writes, the prefill chunk's.
# Decode-only row counts a graph is captured for: vLLM's capture sizes up to 512, then every 32 rows up to MAX_SEQS,
# so a decode-only step of any admissible size can replay a graph (whether the sizes past 512 pay for their capture
# is what graph_max_rows, a bench load option, measures). A step with R decode rows replays the smallest one >= R; the
# extra rows are pad rows. Steps with more rows than the largest captured run eagerly.
GRAPH_ROWS = (1, 2, 4, *range(8, 256, 8), *range(256, 512, 16), *range(512, 1025, 32))


def graph_row_sizes(num_blocks=None, decoding=None, graph_max_rows=GRAPH_ROWS[-1]):
    """The decode-only row counts to capture graphs for: GRAPH_ROWS up to the smallest of graph_max_rows, MAX_SEQS,
    the smallest size that holds every decode row the pool can feed (num_blocks - 1 rows at a block each; None before
    the pool is sized) and, when the job is known, the smallest size that holds every request of it that decodes at
    all (decoding), so a classify job captures a few graphs and a job with no decode captures none."""
    cap = min(graph_max_rows, MAX_SEQS)
    for bound in (num_blocks and num_blocks - 1, decoding):
        if bound is None:
            continue
        if bound <= 0:
            return ()
        cap = min(cap, next((r for r in GRAPH_ROWS if r >= bound), GRAPH_ROWS[-1]))
    return tuple(r for r in GRAPH_ROWS if r <= cap)


# What an HF attention layer can ask of its attention function per call, beyond full causal attention. Qwen3 passes
# sliding_window=None on full-attention layers; a layer that passes a value is asking for something this kernel does
# not do, so it is refused rather than given full causal attention without a word.
UNSUPPORTED_KWARGS = ("sliding_window", "softcap", "s_aux")  # windowed attention, logit soft-capping, attention sinks


def _paged_attention(module, query, key, value, attention_mask, scaling=None, batchinfer_batch=None, **kwargs):
    """HF calls this per layer with the kwargs given to the model's forward, so the step's batch arrives as one
    of them (Executor.logits passes it); nothing about the step lives outside that call. The executor's probe is
    the first such call, so an unsupported layer fails at load. Under layers=compiled the call goes through the
    custom op, which Dynamo keeps opaque (the executor's attend is Python over kernels it cannot trace)."""
    asked = {k: kwargs[k] for k in UNSUPPORTED_KWARGS if kwargs.get(k) is not None}
    if asked:
        raise NotImplementedError(f"{ATTN} implements full causal attention only; layer {module.layer_idx} asks "
                                  f"for {asked}")
    if batchinfer_batch is None:
        raise RuntimeError(f"{ATTN} attention needs the step's batch: run the model through Executor.logits")
    b, ex = batchinfer_batch, batchinfer_batch.executor
    if ex.layers_mode == "compiled":
        out = torch.ops.batchinfer.paged_attention(query, key, value, b.slot_mapping, b.cu_seqlens_q, b.seqused_k,
                                                   b.block_table, module.layer_idx, float(scaling), ex.handle)
        return out.unsqueeze(0), None
    return ex.attend(module.layer_idx, query, key, value, scaling, b), None


_EXECUTORS = weakref.WeakValueDictionary()  # handle -> Executor: what the custom op reaches its executor by


@torch.library.custom_op("batchinfer::paged_attention", mutates_args=())
def paged_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor,
                    cu_seqlens_q: torch.Tensor, seqused_k: torch.Tensor, block_table: torch.Tensor, layer: int,
                    scaling: float, executor: int) -> torch.Tensor:
    """The paged attention of one layer as one opaque op for torch.compile: writes the layer's K and V into the pool
    at slot_mapping and attends through the block table (Executor.attend, with the rest of the step read off the
    executor's current batch). query [1, H, N, D], key and value [1, KVH, N, D] -> [N, H, D]. The pool is the
    executor's, reached by handle and never an argument: nothing in the compiled graph reads it, and an argument
    declared mutated would be cloned and copied back by functionalisation, erasing the step's writes."""
    ex = _EXECUTORS[executor]
    return ex.attend(layer, query, key, value, scaling, ex.batch)[0]


@paged_attention.register_fake
def _(query, key, value, slot_mapping, cu_seqlens_q, seqused_k, block_table, layer, scaling, executor):
    return query.new_empty((query.shape[2], query.shape[1], query.shape[3]))


def register():
    """Registers the attention function only. Causality lives in the kernel, and no mask function is registered on
    purpose: HF skips mask creation for an attention implementation it has no mask function for (masking_utils.py,
    create_causal_mask), while a registered one, even a no-op, makes it check position_ids for packed sequences with a
    host sync on every forward."""
    from transformers import AttentionInterface
    AttentionInterface.register(ATTN, _paged_attention)


def require(ok, message):
    """A knob's rule: ValueError with the message where it does not hold."""
    if not ok:
        raise ValueError(message)


def fa3_available(device):
    """FlashAttention-3 runs on an sm90 (Hopper) card where vLLM's build has it; any other compute capability runs
    FlashAttention-2."""
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 9:
        return False
    from vllm.vllm_flash_attn import is_fa_version_supported
    return bool(is_fa_version_supported(3))


def resolve_fa_version(fa_version, device):
    """The FlashAttention version the fa backend runs: 2 or 3 as given (3 refused where it cannot run), or "auto":
    3 where it runs, what vLLM picks on Hopper, else 2."""
    if fa_version == "auto":
        return 3 if fa3_available(device) else 2
    if fa_version not in (2, 3) or isinstance(fa_version, bool):
        raise ValueError(f"fa_version must be 2, 3 or 'auto', not {fa_version!r}")
    if fa_version == 3 and not fa3_available(device):
        raise ValueError("fa_version=3 needs an sm90 (Hopper) GPU and vLLM's FlashAttention-3 build")
    return fa_version


def check_supported(cfg, model_name):
    """What the per-call check in _paged_attention cannot see: query heads must split evenly over KV heads."""
    if cfg.num_attention_heads % cfg.num_key_value_heads:
        raise ValueError(f"{model_name}: {ATTN} needs query heads in whole groups per KV head; the config has "
                         f"{cfg.num_attention_heads} heads over {cfg.num_key_value_heads} KV heads")


class _Handle:
    """A launched step: wait() blocks until its ids are on the host and returns (ids, GPU ms). key: the graph it
    replayed, (0, 0) when eager. done(): whether the step has finished on the device (no wait)."""

    def __init__(self, key, ids, pending):
        self.key, self.ids, self.pending = key or (0, 0), ids, pending

    def done(self):
        return self.pending is None or self.pending[2].query()

    def wait(self):
        if self.pending is not None:
            out, start, end = self.pending
            end.synchronize()  # the one host sync per step: stop checks need the ids
            self.ids, self.pending = out.tolist(), None
            self.gpu_ms = start.elapsed_time(end)
        else:
            self.gpu_ms = None
        return self.ids, self.gpu_ms


def graph_problems(model, backend):
    """What a captured graph cannot hold: the torch backend's per-row Python loop, and RoPE that reads the largest
    position back to the host (dynamic and longrope scaling). Empty when a graph can hold the step. FlashAttention-3
    is captured too: its tile schedule is computed per step outside the graph (_stage), as vLLM does."""
    rope = getattr(getattr(getattr(model, "model", None), "rotary_emb", None), "rope_type", "default")
    return [what for what, bad in (
        ("cuda_graphs captures the fa backend's kernels: it needs the GPU and the fa backend", backend != "fa"),
        (f"cuda_graphs cannot capture {rope!r} RoPE: it reads the largest position back to the host",
         "dynamic" in rope or rope == "longrope"),
    ) if bad]


def check_graphable(model, backend):
    problems = graph_problems(model, backend)
    if problems:
        raise ValueError("; ".join(problems))


LAYERS = ("auto", "fused", "compiled", "hf")
PAGE_TOKENS_MIN = 128  # pool=contiguous: a unit of at least this many tokens, whatever the model's KV bytes per token
PIECE_BYTES = 2 << 20  # the driver's usual mapping granularity (vmm asks; this sizes the unit)


def page_tokens(layers, kv_heads, head_dim, itemsize):
    """pool=contiguous's unit in tokens: the fewest tokens whose KV across every layer is a whole number of pieces,
    raised to a multiple of itself of at least PAGE_TOKENS_MIN (128 for Qwen3-8B in bf16, 256 with an fp8 pool)."""
    per_token = 2 * layers * kv_heads * head_dim * itemsize
    unit = PIECE_BYTES // math.gcd(PIECE_BYTES, per_token)
    return unit * -(-PAGE_TOKENS_MIN // unit)


def resolve_layers(layers, fusion, cuda):
    """The forward to run: fused where the layers are Qwen3's over the fa kernel (fusion: the problems, empty when it
    can), compiled (torch.compile over HF's layers) on the GPU, hf eagerly; auto takes the first that can run; fused
    asked where it cannot run is refused with the reasons."""
    if layers not in LAYERS:
        raise ValueError(f"layers must be one of {LAYERS}, not {layers!r}")
    if layers == "auto":
        return "fused" if not fusion else "compiled" if cuda else "hf"
    if layers == "fused" and fusion:
        raise ValueError(f"layers=fused: {'; '.join(fusion)}")
    return layers


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
    def __init__(self, model_name, *, dtype="bfloat16", device=None, attention=None, reserve_gb="auto", num_blocks=None,
                 seed=None, model=None, probe=True, fa_version="auto", layers="auto", cuda_graphs="auto",
                 graph_max_rows=GRAPH_ROWS[-1], build=True, block_size=None, ridge_tokens=None, quant="none",
                 kv_dtype="auto", overlap=True, compile_backend="inductor", pool="static"):
        """num_blocks: the KV pool size in blocks (block 0 is the pad block), the same on every card. Without it the
        pool is sized on this card: reserve_gb="auto" measures the largest step's activations with the worst-step
        probe over a small pool and sizes the pool so the run's device peak is at most sizing.MAX_PEAK_SHARE of the
        card (sizing.pool_blocks); a number leaves that many GiB of the free memory after load for activations.
        probe: run the worst-step probe once more over the final pool (auto measures either way). seed shuffles
        the block free list (see kv.BlockAllocator) for every run; tests and the gate use it. fa_version: the
        FlashAttention version of the fa backend, 2, 3 or "auto" (resolve_fa_version: 3 where it runs). layers: the
        forward (resolve_layers): "fused", Qwen3's layers as one fused forward over HF's weights with vLLM's kernels
        (qwen3_fused.py; GPU and fa backend only); "compiled", HF's layers under torch.compile (fullgraph, the paged
        attention an opaque op; compile_backend, "inductor", is for tests); "hf", HF's layers eagerly; "auto", the
        first that can run. cuda_graphs: replay steps from graphs captured at load (GPU, fa backend): decode-only
        steps of up to graph_max_rows rows, and under FlashAttention-3 steps with prefill too (graphs.py); True, False
        or "auto" (on exactly where it can run); self.facts records what was resolved. quant: "none", or "fp8" for the
        fused layers' projections in fp8 (fp8.py; a GPU with FP8 tensor cores). kv_dtype: the KV pool's dtype, "auto"
        (the model's), "bf16" or "fp8" (e4m3, half the bytes per token, so the pool holds twice the tokens: static
        unit scales, the way vLLM runs kv_cache_dtype=fp8 without calibration; FlashAttention-3 reads it with
        descales, the torch backend converts, FlashAttention-2 refuses it). overlap: the engine plans and launches
        a step while the previous one runs (launch / _Handle; the scheduler's overlap mode); a row's token can then
        be the previous step's sampled id still on the device, gathered from g_out (Row.prev). pool: "static" (one
        tensor) or "elastic" (the same shape over CUDA virtual memory, vmm.py: physical pieces mapped under blocks
        as the allocator hands them out and released when their last live block goes, so a card shared with
        another process gets the memory back as rows finish; the address never changes, so graphs and admission are
        untouched)."""
        register()
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cuda = self.device.type == "cuda"
        require(num_blocks is not None or self.cuda,
                "num_blocks is required off the GPU: by default the KV pool is sized from free GPU memory")
        # attention: fa (vLLM's FlashAttention kernel, the GPU default), torch (SDPA per row, the CPU default and
        # the oracle), or cascade (the two-call cascade over the same kernel; over SDPA off the GPU, cascade.py)
        self.attention = attention or ("fa" if self.cuda else "torch")
        require(self.attention in ("fa", "torch", "cascade"), f"attention must be 'fa', 'torch' or 'cascade', not {attention!r}")
        self.backend = "fa" if self.attention != "torch" and self.cuda else "torch"  # the kernel underneath
        self.fa_version = None
        if self.backend == "fa":
            self.fa_version = resolve_fa_version(fa_version, self.device)
            if self.fa_version == 3:
                from vllm.vllm_flash_attn import get_scheduler_metadata
                self._fa_schedule = get_scheduler_metadata
        if model is None:
            model = load_model(model_name, getattr(torch, dtype), ATTN)
        elif hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation(ATTN)
        else:
            model.config._attn_implementation = ATTN
        check_supported(model.config, model_name)
        from .qwen3_fused import fusion_problems
        self.layers_mode = resolve_layers(layers, fusion_problems(model.config) + (
            [] if self.backend == "fa" else ["fused layers run vLLM's CUDA kernels: they need the GPU and the fa kernel"]),
            self.cuda)
        fused_layers = self.layers_mode == "fused"
        cuda_graphs = resolve_build("cuda_graphs", cuda_graphs, graph_problems(model, self.backend))
        require(quant in ("none", "fp8"), f"quant must be 'none' or 'fp8', not {quant!r}")
        if quant == "fp8":
            from .fp8 import fp8_problems
            problems = fp8_problems(self.device, fused_layers)
            require(not problems, "; ".join(problems))
        self.quant = quant
        require(kv_dtype in ("auto", "bf16", "fp8"), f"kv_dtype must be 'auto', 'bf16' or 'fp8', not {kv_dtype!r}")
        require(kv_dtype != "fp8" or self.backend != "fa" or self.fa_version == 3,
                "kv_dtype=fp8 needs the attention kernel that reads an fp8 cache: FlashAttention-3 (fa_version=3, sm90) "
                "or the torch backend")
        self.kv_dtype = kv_dtype
        require(pool in pool_module.MODES, f"pool must be 'static', 'elastic' or 'contiguous', not {pool!r}")
        require(pool != "contiguous" or self.attention != "cascade",
                "pool=contiguous reads KV without a block table: attention=cascade needs one")
        self.pool_mode = pool
        self.pool = None  # the KV pool (pool.py), made at build; the sizing's scratch pool is a static one whatever the mode
        require(isinstance(overlap, bool), f"overlap must be true or false, not {overlap!r}")
        self.overlap = overlap
        # what this executor runs, as the engine records it (metrics.executor): the resolved settings, never "auto"
        # the card's ridge for prefill_budget=ridge: the table's measurement for this card (metrics.CARDS, None until
        # scripts/ridge_calibration.py has run on it), or the number given
        from .metrics import ridge_tokens as card_ridge
        self.ridge_tokens = (card_ridge(torch.cuda.get_device_name(self.device)) if self.cuda else None) \
            if ridge_tokens is None else int(ridge_tokens)
        self.facts = {"layers": self.layers_mode, "cuda_graphs": cuda_graphs, "fa_version": self.fa_version,
                      "ridge_tokens": self.ridge_tokens, "quant": quant, "kv_dtype": kv_dtype, "overlap": self.overlap,
                      "attention": self.attention, "pool": pool}
        self.handle = max(_EXECUTORS.keys(), default=0) + 1  # the custom op's way back to this executor
        _EXECUTORS[self.handle] = self
        self.batch = None  # the step the forward is running, for the op (set by _logits)
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
        # the KV pool's dtype: the model's, or fp8 (e4m3) with static unit scales (one float32 scalar the KV write and
        # the attention descales expand from, so nothing per step), what vLLM runs as kv_cache_dtype=fp8 uncalibrated
        self.pool_dtype = torch.float8_e4m3fn if kv_dtype == "fp8" else self.dtype
        self.kv_scale = torch.ones((), dtype=torch.float32, device=self.device)
        # the last step's sampled ids, at a fixed address outside any graph pool: the next step's rows gather their
        # token from it (Row.prev), and the host copy for commit is taken from it (two pinned slots, one per step in
        # flight); the previous step's events say when its pinned staging buffers and its ids are free to reuse
        self.g_out = torch.zeros(MAX_SEQS + 1, dtype=torch.long, device=self.device)  # + 1: the probe's prefill row
        self.h_out = torch.zeros((2, MAX_SEQS + 1), dtype=torch.long, pin_memory=True) if self.cuda else None
        self.parity = 0  # which pinned output slot the next launch takes
        self.last_handle = None  # the step launched last: the last that may touch what a commit then frees
        # tokens per KV block: the paged attention path takes multiples of 16 (vLLM's FlashAttention backend enforces
        # the same); the one place the default lives, published in Limits for every other reader. pool=contiguous
        # maps whole units of the driver's granularity, so its block is the unit: page_tokens, the smallest multiple
        # of (2 MiB over the token's KV bytes, reduced) of at least PAGE_TOKENS_MIN tokens; another value is refused
        self.page_tokens = page_tokens(self.layers, self.kv_heads, self.head_dim, self.pool_dtype.itemsize)
        require(pool != "contiguous" or block_size in (None, self.page_tokens),
                f"pool=contiguous maps units of {self.page_tokens} tokens for this model and dtype: block_size must be "
                f"that or unset, not {block_size!r}")
        self.block_size = (self.page_tokens if pool == "contiguous" else DEFAULT_BLOCK_SIZE) if block_size is None \
            else int(block_size)
        require(self.block_size % 16 == 0 and (16 <= self.block_size <= 256 or pool == "contiguous"),
                f"block_size must be a multiple of 16 up to 256 (the paged attention kernels' rule), not {block_size!r}")
        self.scratch_blocks = 2 + -(-MAX_PREFILL_BUDGET // self.block_size)  # a scratch pool the widest step fits
        self.spec = pool_module.Spec(self.layers, self.kv_heads, self.head_dim, self.block_size, self.page_tokens,
                                     self.pool_dtype, self.device, MAX_SEQS, self.max_positions)
        self.bytes_per_block = 2 * self.layers * self.block_size * self.kv_heads * self.head_dim * self.pool_dtype.itemsize
        self.attn = Attention(self.backend, self.fa_version, self.kv_heads, self.n_rep, self.block_size,
                                        self.pool_dtype, self.dtype, self.kv_scale)
        self.memory_facts = {}  # what the auto sizing measured; the batchinfer engine records it in metrics.memory
        self.fused = None
        if fused_layers:  # before sizing, so the probe measures the fused forward's activations
            from .qwen3_fused import FusedQwen3, fuse_projections
            fuse_projections(self.model)
            torch.cuda.empty_cache()  # the unfused projection tensors, freed
            self.fused = FusedQwen3(self.model, self, quant=quant)
        self.model_fn = self.model
        if self.layers_mode == "compiled":  # the whole forward, no graph break allowed; the op keeps attention opaque
            torch._dynamo.config.cache_size_limit = max(getattr(torch._dynamo.config, "cache_size_limit", 8), 64)
            torch._dynamo.config.fail_on_recompile_limit_hit = True  # never a silent fall back to eager
            self.model_fn = torch.compile(self.model, fullgraph=True, backend=compile_backend)
        # the second half of a load, build(): the pool sized, the graphs captured, the probe run. build=True does it
        # now (the bench adapter, which serves many jobs, and the tests); the CLI leaves it for StepEngine.prepare,
        # once the job is analysed, so the graphs captured fit the job
        self.cuda_graphs, self.graph_max_rows = cuda_graphs, graph_max_rows
        self.max_seqs, self.max_prefill = MAX_SEQS, MAX_PREFILL_BUDGET  # the step caps: probed for, then published
        self.reserve_gb, self.num_blocks_asked, self.probe_asked = reserve_gb, num_blocks, probe
        # graph_sizes: the decode-only row sizes; graph_keys: every graph (graphs.keys: those, and under FA3 the mixed
        # keys); graphs: key -> (CUDAGraph, its sampled ids), once built
        self.graph_sizes, self.graph_keys, self.graphs = (), (), {}
        self.staging = None  # the graphs' buffers (graphs.Staging), once built
        self.graph_rows = self.graph_tokens = 0  # the last forward's graph key, (0, 0) if it ran eagerly
        self.num_blocks = self.limits = self.probe_peak_reserved_gb = None
        if build:
            self.build()

    def build(self, decoding=None):
        """Sizes the pool, captures the graphs and runs the probe; returns the Limits the scheduler reads (also
        self.limits). decoding: the job's requests that decode at all, which bounds the graphs captured
        (graph_row_sizes); None captures for any job. Once per executor."""
        if self.limits is not None:
            raise RuntimeError("the executor is already built")
        num_blocks = self.num_blocks_asked
        if num_blocks is None and self.reserve_gb == "auto":
            # the sizing capture measures the graphs' memory over a scratch pool, so it is not bounded by the pool
            self._graph_keys(graph_row_sizes(None, decoding, self.graph_max_rows) if self.cuda_graphs else ())
            num_blocks = self._measure_pool()
        elif num_blocks is None:
            free, _ = torch.cuda.mem_get_info(self.device)
            num_blocks = int((free - float(self.reserve_gb) * GiB) // self.bytes_per_block)
        self.num_blocks = num_blocks
        self._graph_keys(graph_row_sizes(num_blocks, decoding, self.graph_max_rows) if self.cuda_graphs else ())
        self.pool = pool_module.make(self.pool_mode, self.spec, num_blocks)
        if self.graph_keys:  # before the probe, so it checks the card with the graphs' memory held
            self._staging(num_blocks)
            self._capture()
        self.probe_peak_reserved_gb = self.probe() if self.probe_asked and self.cuda else None
        self.limits = Limits(num_blocks, self.block_size, self.max_seqs, self.max_prefill, self.max_positions,
                             self.graph_sizes, graphs.MIXED_TOKENS if any(t for _, t in self.graph_keys) else (),
                             self.ridge_tokens, "fp8" if self.kv_dtype == "fp8" else "bf16", self.overlap, self.attention,
                             self.pool_mode, self.pool.blocks_per_group)
        return self.limits

    def _graph_keys(self, sizes):
        """The graphs to capture for these decode-only sizes: them, and under FlashAttention-3 the mixed graphs."""
        self.graph_sizes = sizes
        self.graph_keys = graphs.keys(sizes, mixed=self.fa_version == 3)

    @property
    def kv(self):
        """The pool tensor (pool.py): [layers, K|V, blocks, ...] layer-major, or token-major under pool=contiguous."""
        return self.pool.kv

    def _probe_step(self, scratch=False):
        """The widest step the scheduler can build, the caps build() publishes in Limits: max_seqs sampled decode rows
        (a [rows, vocab] logits tensor, ~0.3 GiB for Qwen3 at 1,024 rows) plus max_prefill tokens of prefill. Over
        the real pool every decode row writes a block of its own, and a pool too small for that gets the largest such
        step it can hold. scratch: every decode row writes block 1 instead, so the whole step fits scratch_blocks;
        the probe's KV is garbage either way, and its activations do not depend on which blocks the rows write."""
        bs = self.block_size
        # a prefill row never runs past the model's positions (the scheduler refuses longer prompts), so the chunk
        # splits into rows of at most that many tokens (whole blocks), each in its own region slot after the decode
        # rows' (pool=contiguous sizes a slot from max_positions); the rows they take come off the decode rows
        per = max(bs, min(self.max_prefill, self.max_positions or self.max_prefill) // bs * bs)
        k = -(-self.max_prefill // per)  # prefill rows at most
        rows = min(self.max_seqs + 1 - k, self.max_seqs if scratch else (self.num_blocks - 1) // 2)
        first = 2 if scratch else 1 + rows  # the prefill chunk's first block
        n = min(self.max_prefill, (self.num_blocks - first) * bs)
        chunks = [per] * (n // per) + ([n % per] if n % per else [])
        out = [Row(seq=i, start=0, token_ids=[0], block_ids=[1 if scratch else 1 + i], sample=True, decode=True, slot=1 + i)
               for i in range(rows)]
        for j, m in enumerate(chunks):
            out.append(Row(seq=rows + j, start=0, token_ids=[0] * m, block_ids=list(range(first, first + -(-m // bs))),
                           sample=True, decode=False, slot=1 + rows + j))
            first += -(-m // bs)
        step = Step(out)
        step.slot_resets = list(range(1, len(out) + 1))
        return step

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
        """reserve_gb="auto": the widest-step probe over a scratch pool (scratch_blocks, so the measurement fits
        where the model barely does) measures the largest step's activations on this card; sizing.pool_blocks then
        sizes the real pool around them."""
        self.num_blocks, self.pool = self.scratch_blocks, pool_module.make("static", self.spec, self.scratch_blocks)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        before = torch.cuda.memory_reserved(self.device)
        peak, outside, total = self._step_peak(scratch=True)
        activation = peak - before
        graph = 0
        if self.graph_keys:  # every graph captured as the run will, over the scratch pool: their memory, held all run
            self._staging(self.scratch_blocks)
            graph = self._capture()
            self.graphs, self.graph_pool = {}, None
            self.memory_facts["graph_sized_gb"] = round(graph / GiB, 2)
        self.pool = None
        torch.cuda.empty_cache()
        in_use = torch.cuda.memory_reserved(self.device) + outside
        self.memory_facts.update(in_use_after_load_gb=round(in_use / GiB, 2), step_activation_gb=round(activation / GiB, 2))
        return pool_blocks(total / GiB, in_use / GiB, activation / GiB, self.bytes_per_block / GiB, graph_gb=graph / GiB)

    def probe(self):
        """The widest-step probe over the real pool before any real work, so activation memory at the largest step
        the scheduler can build is checked against this card now rather than mid-job. It is the run's own worst
        step, so it records its peak (torch's reserved GiB) and fails only on OOM."""
        token = self.pool.probe_begin()  # the probe writes across the pool: whatever that needs mapped
        try:
            peak, outside, _ = self._step_peak()
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(f"memory probe at {self.max_seqs} sampled decode rows + a {self.max_prefill}-token prefill "
                               f"chunk ran out of memory with a {self.num_blocks}-block KV pool; pass a smaller num_blocks "
                               f"or a larger reserve_gb") from e
        self.pool.probe_end(token)
        self.memory_facts["probe_peak_device_gb"] = round((peak + outside) / GiB, 2)  # what sizing's limit is about
        return round(peak / GiB, 2)

    # decode-only steps from CUDA graphs --------------------------------------------------------------

    def _staging(self, num_blocks):
        """The graphs' persistent buffers (graphs.Staging) for this pool: the block table as wide as the longest
        context the pool holds (k_cap), FlashAttention-3's schedule call where it runs, the cascade's and the
        regions' arrays where those modes run."""
        bs = self.block_size
        k_cap = min(self.max_positions or bs * (num_blocks - 1), bs * (num_blocks - 1))
        token = self.pool.layout == "token"
        heads = (self.model.config.num_attention_heads, self.kv_heads, self.head_dim, self.pool_dtype, None if token else bs)
        self.staging = graphs.Staging(self.graph_keys, bs, k_cap, self.device, self.g_out, cascade=self.attention == "cascade",
                                      region=self.pool.max_ctx if token else None,
                                      schedule=self._fa_schedule if self.fa_version == 3 else None, heads=heads)

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
        self.graphs = {key: self._capture_one(key) for key in reversed(self.graph_keys)}
        torch.cuda.synchronize(self.device)
        self._graph_kv_ptr = self.pool.kv.data_ptr()
        added = free_before - torch.cuda.mem_get_info(self.device)[0]
        self.memory_facts.update(graph_capture_s=round(self.memory_facts.get("graph_capture_s", 0)
                                                       + time.perf_counter() - t0, 2),
                                 graph_held_gb=round(added / GiB, 2))
        return added

    @torch.inference_mode()
    def _capture_one(self, key):
        """One warmup, then the capture of this graph (every row a pad row, which the graph does not depend on), both on
        the executor's one capture stream: library handles and lazy initialisation happen in the warmup, outside the
        graph, and once per stream (cuBLAS keeps a workspace per stream, ~32 MiB each)."""
        if getattr(self, "_capture_stream", None) is None:
            self._capture_stream = torch.cuda.Stream(self.device)
        side = self._capture_stream
        self.staging.stage(Step([]), key)  # all pad rows
        b = graphs.Inputs.graph(self, key)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            self._logits(b).argmax(-1)
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.graph_pool, stream=side):
            ids = self._logits(b).argmax(-1)
        return graph, ids

    def attend(self, layer, query, key, value, scaling, b):
        """query [1, H, N, D], key/value [1, KVH, N, D] (HF layout, RoPE applied) -> [1, N, H, D]: the K and V written
        into the pool, then the attention (attention.py). b: the step's inputs."""
        self.pool.write(layer, key[0].transpose(0, 1), value[0].transpose(0, 1), b.slot_mapping)
        return self.attn.run(query[0].transpose(0, 1), self.pool, layer, b, scaling).unsqueeze(0)

    @torch.inference_mode()
    def logits(self, step):
        """[sampling rows, vocab] logits of one step, in row order. Writes the step's K/V into the pool."""
        return self._logits(graphs.Inputs.exact(step, self))

    def _logits(self, b):
        """The forward over one step's inputs b: fused layers, HF's compiled, or HF's eager."""
        if self.fused is not None:
            return self.fused.logits(b)
        self.batch = b
        ids = b.input_ids  # once: a graph input gathers it anew on every read, and the marks below live on the tensor
        keep = b.sample_idx  # the rows sampled; 0 (every row) where that is all of them, but under compile always
        if self.layers_mode == "compiled":  # a tensor (one graph), every per-step dimension symbolic: no recompiles
            for x, dim in ((ids, 1), (b.position_ids, 1), (b.slot_mapping, 0), (b.cu_seqlens_q, 0),
                           (b.seqused_k, 0), (b.block_table, 0), (b.block_table, 1), (b.sample_idx, 0)):
                torch._dynamo.mark_dynamic(x, dim)
        elif b.all_sample:
            keep = 0
        return self.model_fn(input_ids=ids, position_ids=b.position_ids, use_cache=False, logits_to_keep=keep,
                             batchinfer_batch=b).logits[0]

    def forward(self, step):
        """Runs one step and waits for it: (one greedy id per sampling row in row order, GPU milliseconds or None on
        CPU). self.graph_rows says whether the step replayed a graph (its row count) or ran eagerly (0)."""
        return self.launch(step).wait()

    def launch(self, step, maps=()):
        """Enqueues one step and returns a _Handle whose wait() gives (the sampled ids in row order, GPU ms). maps:
        the groups the allocator handed out since the last step (pool=elastic). On the CPU the step runs here. On
        the GPU nothing waits: the step's sampled ids are copied on its stream into g_out (where the next step's
        rows gather their token) and from there into a pinned slot, and an event marks the end; a next step may be
        staged and launched before this one's ids are read (StepEngine, overlap)."""
        self.pool.before_step(step, maps)  # memory under its blocks, its rows' regions
        for src, dst, k in step.copies:  # sharing=token: the first k slots of a computed block, into a row's own block
            self.pool.copy(src, dst, k)
        n = sum(1 for r in step.rows if r.sample)
        key = graphs.key_for(step, self.graph_keys) if self.graphs else None
        self.graph_rows, self.graph_tokens = key or (0, 0)
        self.last_handle = self._launch(step, key, n)
        return self.last_handle

    def release(self, groups):
        """Groups the allocator freed at a commit (pool=elastic): their memory goes once the last step launched is
        done, the last that may touch them."""
        self.pool.release(groups, self.last_handle)

    def finish_job(self):
        """The job's last step is done: whatever waited on it goes (pool=elastic: the last groups)."""
        self.pool.finish()

    def reset(self):
        """A job died mid-step and its allocator is dropped: the pool's memory goes back to the pad block's."""
        self.pool.reset()

    def _launch(self, step, key, n):
        if not self.cuda:
            ids = self.logits(step).argmax(-1)
            self.g_out[:n].copy_(ids)
            return _Handle(key, ids.tolist(), None)
        k = self.parity = 1 - self.parity
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        if key:
            assert self.pool.kv.data_ptr() == self._graph_kv_ptr, "the KV pool moved after capture: the graphs read the old one"
            self.staging.stage(step, key)
            graph, ids = self.graphs[key]
            graph.replay()
        else:
            ids = self.logits(step).argmax(-1)
        self.g_out[:n].copy_(ids[:n])  # before the next step is launched: it gathers from here and then overwrites it
        self.h_out[k, :n].copy_(self.g_out[:n], non_blocking=True)
        end.record()
        return _Handle(key, None, (self.h_out[k, :n], start, end))
