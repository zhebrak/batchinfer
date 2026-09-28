"""Batch analysis: tokenize once and characterise the job. Facts and metrics only, no decisions.

What runs together, in what order and under which budget is decided in policy.py (once, before
the first step) and scheduler.py (at every step), from the facts produced here. No torch here: this
runs on a CPU.
"""
from collections import Counter

from .kv import BLOCK_SIZE, reservation_blocks
from .prefix import PrefixTrie
from .schema import AnalyzedRequest, JobAnalysis


def analyze(requests, tokenizer):
    items = tokenize(requests, tokenizer)
    trie = PrefixTrie.build([it.token_ids for it in items], BLOCK_SIZE)
    return JobAnalysis(requests=items, tokenizer=str(getattr(tokenizer, "name_or_path", type(tokenizer).__name__)),
                       metrics=job_metrics(items, trie), prefix=trie)


# per request --------------------------------------------------------------------------------------

def kind_of(req):
    """Exact, not a threshold: the v2 fast path keys on max_tokens == 1; label sets may be multi-token."""
    if req.max_tokens == 1:
        return "prefill_only"
    if req.labels:
        return "label"
    return "generate"


def decode_ratio(prompt_len, max_tokens):
    """Decode forwards per prefill token, at most: the first output token comes from the prefill forward,
    each later one from a decode forward. Exact for ignore_eos rows, an upper bound otherwise."""
    return (max_tokens - 1) / max(prompt_len, 1)


def tokenize(requests, tok):
    """The one tokenization. Prompts are already chat-templated text, so no special tokens are added
    (the same call bench.workload uses for prompt_tokens). Label sets are tokenized too, and cached
    per distinct set, so the v2 classification path needs no contract change."""
    encoded = tok([r.prompt for r in requests], add_special_tokens=False)["input_ids"] if requests else []
    label_cache, items = {}, []
    for req, ids in zip(requests, encoded):
        if not ids:  # the first output token is sampled from the last prompt position; there is none
            raise ValueError(f"request {req.id}: the prompt encodes to no tokens")
        label_ids = None
        if req.labels:
            key = tuple(req.labels)
            if key not in label_cache:
                label_cache[key] = [list(x) for x in tok(list(key), add_special_tokens=False)["input_ids"]]
            label_ids = label_cache[key]
        items.append(AnalyzedRequest(req=req, token_ids=list(ids), prompt_len=len(ids), label_ids=label_ids,
                                     kind=kind_of(req), decode_ratio=decode_ratio(len(ids), req.max_tokens)))
    return items


# prefix structure ---------------------------------------------------------------------------------

def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def unique_prompt_tokens(seqs, page=1):
    """Size of a token-level prefix trie over seqs without building one: sort, and each sequence adds what
    its predecessor does not cover. With page > 1 only whole pages are shared. Same arithmetic as
    bench.workload.unique_tokens. The block trie's unique_prefill_tokens (prefix.py) is the same at
    page=BLOCK_SIZE except that it keeps every prompt's last block private."""
    unique, prev = 0, ()
    for s in sorted(seqs):
        unique += len(s) - lcp(s, prev) // page * page
        prev = s
    return unique


# job metrics --------------------------------------------------------------------------------------

def pcts(values):
    v = sorted(values)
    if not v:
        return {"p50": 0, "p95": 0, "max": 0}
    return {"p50": v[len(v) // 2], "p95": v[int(0.95 * (len(v) - 1))], "max": v[-1]}


def job_metrics(items, trie):
    raw = sum(it.prompt_len for it in items)
    unique = unique_prompt_tokens([tuple(it.token_ids) for it in items])
    kinds = Counter(it.kind for it in items)
    plen, mtok = pcts([it.prompt_len for it in items]), pcts([it.req.max_tokens for it in items])
    decode_max = sum(it.req.max_tokens - 1 for it in items)
    reserved = [BLOCK_SIZE * reservation_blocks(it.prompt_len, it.req.max_tokens) for it in items]
    prefix = trie.metrics()
    return {
        "requests": len(items),
        "n_prefill_only": kinds.get("prefill_only", 0),
        "n_label": kinds.get("label", 0),
        "n_generate": kinds.get("generate", 0),
        "n_ignore_eos": sum(it.req.ignore_eos for it in items),
        "prompt_tokens": raw,
        "unique_prompt_tokens": unique,
        "ideal_prefix_reuse": round(1 - unique / raw, 4) if raw else 0.0,  # token-level ceiling
        # what an engine sharing whole blocks can reach: prefix_hit_pct is judged against this one
        f"ideal_prefix_reuse_page{trie.block_size}": round(1 - prefix["unique_prefill_tokens"] / raw, 4) if raw else 0.0,
        **prefix,
        "prompt_len_p50": plen["p50"], "prompt_len_p95": plen["p95"], "prompt_len_max": plen["max"],
        "max_tokens_p50": mtok["p50"], "max_tokens_max": mtok["max"],
        "max_tokens_sum": sum(it.req.max_tokens for it in items),
        "decode_tokens_max": decode_max,  # decode forwards if every request runs to max_tokens
        "decode_chain": max(mtok["max"] - 1, 0),  # decode steps the longest request needs
        "job_decode_ratio": round(decode_max / raw, 4) if raw else 0.0,
        "kv_reservation_tokens_sum": sum(reserved),  # rounded up to whole blocks; not bench's kv_demand_tokens
        "max_reservation_tokens": max(reserved, default=0),
    }


def describe(job):
    """Human-readable job metrics, for `python -m batchinfer analyze`."""
    m = job.metrics
    return "\n".join([
        f"{m['requests']} requests: {m['n_prefill_only']} prefill_only, {m['n_label']} label, "
        f"{m['n_generate']} generate; {m['n_ignore_eos']} with ignore_eos",
        f"prompt tokens: raw {m['prompt_tokens']:,}, unique {m['unique_prompt_tokens']:,} "
        f"(ideal prefix reuse {m['ideal_prefix_reuse']:.1%}, {job.prefix.block_size}-token blocks "
        f"{m[f'ideal_prefix_reuse_page{job.prefix.block_size}']:.1%}); length p50/p95/max "
        f"{m['prompt_len_p50']}/{m['prompt_len_p95']}/{m['prompt_len_max']}",
        f"max_tokens: p50/max {m['max_tokens_p50']}/{m['max_tokens_max']}, sum {m['max_tokens_sum']:,}; "
        f"decode forwards <= {m['decode_tokens_max']:,} (job decode ratio {m['job_decode_ratio']}), "
        f"decode chain {m['decode_chain']} steps",
        f"KV reservations: sum {m['kv_reservation_tokens_sum']:,} tokens, largest {m['max_reservation_tokens']:,}",
        job.prefix.describe(),
    ])
