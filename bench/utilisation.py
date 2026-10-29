"""How much of the card a bench.run row used: model FLOPs utilisation (mfu_pct) and model bandwidth utilisation
(mbu_pct), counted the same way for every engine, so vLLM, naive and batchinfer rows compare on one scale.

Both count the work the model needs for what the engine computed, from its request trace (prompt_len,
prefix_hit_tokens, output_tokens per request): padding, dead decode slots and re-reads are waste, not work, so an
engine is not credited for them. Per request with prompt length L, h prefix-hit tokens and n output tokens:
- query tokens: prefill computes positions h..L-1, decode feeds back n-1 tokens at positions L..L+n-2 (the
  first output token comes from prefill);
- FLOPs: 2 x body matmul weights per query token, 2 x hidden x vocab per sampled token (n), and causal
  attention, 4 x layers x q_dim per (query, key) pair, a query at position p reading p+1 keys;
- bytes: every forward pass reads every matmul weight once (lm_head included, the embedding lookup is a few rows);
  prefill moves L tokens of KV (h read, L-h written) and a decode token at position p moves p+1 (reads p, writes 1).
Chunked-prefill re-reads, activations and the embedding rows are left out, so mbu_pct is a lower bound.

mfu_pct is those FLOPs over wall_s over the card's dense bf16 peak; mbu_pct those bytes over wall_s over its HBM
bandwidth. Prefill is compute-bound and decode bandwidth-bound, so a job reads on both: a decode-heavy row stays low
on MFU at full bandwidth. The shape comes from the model's config (dense decoder only: a mixture of experts gets
None, as does a card missing from the peak tables, rather than a wrong number).
"""
from dataclasses import asdict, dataclass

from batchinfer.metrics import peak_bandwidth, peak_flops

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


@dataclass(frozen=True)
class Shape:
    """What the FLOP and byte counts need from a dense decoder's config. q_dim and kv_dim are heads x head_dim."""
    layers: int
    hidden: int
    q_dim: int
    kv_dim: int
    intermediate: int
    vocab: int
    dtype_bytes: int
    kv_dtype_bytes: int = None  # the KV pool's bytes per element when it differs from the weights' (an fp8 pool)

    @property
    def body_params(self):
        """Matmul weights per layer (q, k, v, o, gate, up, down) over all layers: norms and biases are not matmuls."""
        h = self.hidden
        return self.layers * (2 * h * self.q_dim + 2 * h * self.kv_dim + 3 * h * self.intermediate)

    @property
    def head_params(self):
        return self.hidden * self.vocab

    @property
    def weight_bytes(self):
        return (self.body_params + self.head_params) * self.dtype_bytes

    @property
    def kv_bytes_per_token(self):
        return 2 * self.layers * self.kv_dim * (self.kv_dtype_bytes or self.dtype_bytes)  # K and V


def shape_of(config):
    """A Shape from a transformers config, or None when it is not a dense decoder this module can count."""
    get = lambda k, default=None: getattr(config, k, default)  # noqa: E731
    if any(get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts")):
        return None
    try:
        heads, kv_heads = config.num_attention_heads, get("num_key_value_heads") or config.num_attention_heads
        head_dim = get("head_dim") or config.hidden_size // heads
        dtype = str(get("torch_dtype") or get("dtype") or "bfloat16").removeprefix("torch.")
        return Shape(layers=config.num_hidden_layers, hidden=config.hidden_size, q_dim=heads * head_dim,
                     kv_dim=kv_heads * head_dim, intermediate=config.intermediate_size, vocab=config.vocab_size,
                     dtype_bytes=DTYPE_BYTES[dtype])
    except (AttributeError, KeyError):
        return None


def load_shape(model):
    """The Shape of a model from its config in the local Hugging Face cache, or None: never downloads."""
    try:
        from transformers import AutoConfig
        return shape_of(AutoConfig.from_pretrained(model, local_files_only=True))
    except Exception:  # no transformers, no cached snapshot, or not a hub model: like model_revision, best effort
        return None


def _span(a, b):
    """Sum of p + 1 over positions a..b-1: the keys read by queries at those positions (causal)."""
    return (b * (b + 1) - a * (a + 1)) // 2 if b > a else 0


def work(shape, requests):
    """{query_tokens, dense_flops, attention_flops, kv_tokens_moved} for (prompt_len, prefix_hit_tokens,
    output_tokens) per request."""
    query = dense = attention = kv = 0
    for prompt, hit, out in requests:
        decode = max(out - 1, 0)
        q = (prompt - hit) + decode
        query += q
        dense += 2 * shape.body_params * q + 2 * shape.head_params * out
        attention += 4 * shape.layers * shape.q_dim * (_span(hit, prompt) + _span(prompt, prompt + decode))
        kv += prompt + _span(prompt, prompt + decode)
    return {"query_tokens": query, "dense_flops": dense, "attention_flops": attention, "kv_tokens_moved": kv}


def forward_passes(details):
    """Forward passes the engine ran, from its record: one per step (batchinfer, traced vLLM), or one prefill plus
    its decode steps per group (naive). None when the record has neither."""
    steps = (details or {}).get("steps") or {}
    if steps:
        return len(next(iter(steps.values())))
    groups = (details or {}).get("groups") or []
    if groups and all("decode_steps" in g for g in groups):
        return sum(1 + g["decode_steps"] for g in groups)
    return None


def requests_of(details):
    trace = (details or {}).get("request_trace") or {}
    if not trace:
        return None
    return list(zip(trace["prompt_len"], trace["prefix_hit_tokens"], trace["output_tokens"]))


def bytes_of(details, opts):
    """(weight bytes per parameter, KV bytes per element) a row moved, by what it ran, the same rule for both
    engines: fp8 weights (batchinfer quant=fp8, vLLM quantization=fp8) and an fp8 cache (batchinfer kv_dtype=fp8,
    vLLM kv_cache_dtype=fp8) are one byte; None where the row says nothing (the model's dtype then)."""
    ex, opts = details.get("executor") or {}, opts or {}
    fp8_w = ex.get("quant") == "fp8" or opts.get("quant") == "fp8" or str(opts.get("quantization", "")).startswith("fp8")
    fp8_kv = ex.get("kv_dtype") == "fp8" or opts.get("kv_dtype") == "fp8" or str(opts.get("kv_cache_dtype", "")).startswith("fp8")
    return (1 if fp8_w else None), (1 if fp8_kv else None)


def utilisation(shape, details, gpu, wall_s, opts=None):
    """(mfu_pct, mbu_pct, record): record holds the counts behind them, for the run page; each is None when an
    input is missing (no shape, no request trace, no forward count, or a card without a listed peak). The bytes
    moved follow what the row ran (bytes_of): fp8 weights and an fp8 cache are one byte each, on both engines."""
    requests = requests_of(details)
    if shape is None or requests is None or not wall_s:
        return None, None, None
    w_bytes, kv_bytes = bytes_of(details, opts)
    if w_bytes is not None or kv_bytes is not None:
        shape = Shape(**{**asdict(shape), "dtype_bytes": w_bytes or shape.dtype_bytes,
                         "kv_dtype_bytes": kv_bytes or shape.kv_dtype_bytes or shape.dtype_bytes})
    w = work(shape, requests)
    passes = forward_passes(details)
    flops = w["dense_flops"] + w["attention_flops"]
    moved = None if passes is None else passes * shape.weight_bytes + w["kv_tokens_moved"] * shape.kv_bytes_per_token
    flops_peak, peak_bw = peak_flops(gpu), peak_bandwidth(gpu)
    mfu = round(100 * flops / wall_s / flops_peak, 1) if flops_peak else None
    mbu = round(100 * moved / wall_s / peak_bw, 1) if peak_bw and moved is not None else None
    record = {"model_tflop": round(flops / 1e12, 2),
              "attention_flop_pct": round(100 * w["attention_flops"] / flops, 1) if flops else None,
              "model_gb_moved": None if moved is None else round(moved / 1e9, 1),
              "forward_passes": passes, "query_tokens": w["query_tokens"],
              "peak_bf16_tflops": flops_peak and flops_peak / 1e12, "peak_hbm_gb_per_s": peak_bw and peak_bw / 1e9,
              **{f"shape_{k}": v for k, v in asdict(shape).items()}}
    return mfu, mbu, record
