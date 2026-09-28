"""Policy: decisions made once, before the first step, from analysis facts and the user's config.

decide(job, cfg, hardware) -> Policy is the one entry point; the records live in schema.py. It resolves the
admission order, the fixed groups when admission="groups" (the naive engine runs each as one padded batch; the
batchinfer engine admits each whole after the previous one finished) and the prefill budget. Per-step decisions
(who is admitted now, which tokens go into this step) live in scheduler.py. Hardware arrives only as the Hardware
record the engine measured at load; nothing here calls CUDA.
"""
import math
from collections import Counter

from .analysis import lcp
from .kv import BLOCK_SIZE
from .schema import DEFAULT_MAX_BATCH_TOKENS, MAX_PREFILL_BUDGET, MIN_PREFILL_BUDGET, ORDERS, Group, Policy

# Under max_tokens_desc a naive group runs until its longest max_tokens. A row whose max_tokens is under
# half of that would spend more than half its decode steps as a dead slot, so the group closes there.
DEAD_SLOT_RATIO = 2


# order ---------------------------------------------------------------------------------------------

def ordered(job, order):
    """Admission order as indices into job.requests. Python's sort is stable, so ties keep input order.
    input:             as read; the shuffled workload file is the neutral reference.
    max_tokens_desc:   (max_tokens, prompt_len) descending. For static batches, equal max_tokens together
                       avoids dead slots and equal lengths minimise padding.
    decode_ratio_desc: (max_tokens - 1) / prompt_len descending. Decode-heavy requests start first, so
                       decode rows exist throughout the job; prefill-heavy ones (classification, ratio 0)
                       come last and fill steps as prefill chunks. BatchLLM's ordering, per request.
    prefix_dfs:        depth first over the job's prefix trie; at every branch the subtree holding the
                       highest decode ratio goes first (ties: lowest input index). Requests sharing a
                       prefix are admitted one after another, so a shared block lives from its first
                       user's prefill to its last user's finish and is never held for requests far down
                       the order. BatchLLM's ordering, per prefix group."""
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}, not {order!r}")
    items = job.requests
    idx = list(range(len(items)))
    if order == "max_tokens_desc":
        return sorted(idx, key=lambda i: (items[i].req.max_tokens, items[i].prompt_len), reverse=True)
    if order == "decode_ratio_desc":
        return sorted(idx, key=lambda i: -items[i].decode_ratio)
    if order == "prefix_dfs":
        return job.prefix.dfs_order([(-it.decode_ratio, i) for i, it in enumerate(items)])
    return idx


# fixed groups ---------------------------------------------------------------------------------------

def padded_tokens(members, items):
    """End-state footprint of a static batch: every row padded to the longest prompt and run to the
    longest max_tokens. This is what the naive engine's cache holds at the last decode step, and the one formula
    the naive engine checks against the budget."""
    return len(members) * (max(items[i].prompt_len for i in members) + max(items[i].req.max_tokens for i in members))


def partition(order, items, max_batch_tokens, max_batch_size, dead_slot_ratio=None):
    """Greedy fill in the given order. A group closes when the next request would break either budget,
    or, with dead_slot_ratio set (max_tokens_desc order, so the first member holds the group's longest
    max_tokens), when that longest max_tokens is at least dead_slot_ratio times the next request's.
    A single request over the token budget still gets its own group (counted as oversized in stats): the naive
    engine refuses it before the first forward, and the batchinfer engine checks it against its KV pool instead
    (scheduler._check)."""
    groups, cur = [], []
    for i in order:
        if cur and (len(cur) >= max_batch_size or padded_tokens(cur + [i], items) > max_batch_tokens
                    or (dead_slot_ratio and items[cur[0]].req.max_tokens >= dead_slot_ratio * items[i].req.max_tokens)):
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return [Group(members=g, shared_prefix_len=shared_prefix(g, items)) for g in groups]


def shared_prefix(members, items):
    """Longest common token prefix over the members; 0 when there is nobody to share with."""
    if len(members) < 2:
        return 0
    first = items[members[0]].token_ids
    n = len(first)
    for i in members[1:]:
        n = min(n, lcp(first, items[i].token_ids))
    return n


def fixed_groups(job, order, max_batch_tokens, max_batch_size):
    """The fixed-groups partition of the job in the given order, in execution order."""
    return tuple(partition(ordered(job, order), job.requests, max_batch_tokens, max_batch_size,
                           dead_slot_ratio=DEAD_SLOT_RATIO if order == "max_tokens_desc" else None))


def group_stats(items, groups, max_batch_tokens, order, padded):
    """Facts about a fixed-groups decision. The order, the budget and the group count are Policy fields, so they
    are not repeated here. With padded (the naive engine runs each group as one padded batch) also the padding
    the groups will cost, under groups_ names: the engine measures the same quantities itself as
    padded_prompt_tokens and padding_waste_pct, and a decided value never shares a key with a measured one."""
    raw = sum(it.prompt_len for it in items)
    shared = sum((len(g.members) - 1) * g.shared_prefix_len for g in groups)
    stats = {
        "oversized": sum(1 for g in groups if padded_tokens(g.members, items) > max_batch_tokens),
        "in_batch_reuse": round(shared / raw, 4) if raw else 0.0,
        "dead_slot_ratio": DEAD_SLOT_RATIO if order == "max_tokens_desc" else None,
    }
    if padded:
        tokens = sum(len(g.members) * max(items[i].prompt_len for i in g.members) for g in groups)
        stats.update(padded_groups=sum(1 for g in groups if len({items[i].prompt_len for i in g.members}) > 1),
                     groups_padded_prompt_tokens=tokens,
                     groups_padding_waste_pct=round(100 * (1 - raw / tokens), 2) if tokens else 0.0)
    return stats


def auto_max_batch_tokens(cfg, hardware):
    """max_batch_tokens="auto": what the engine measured on its card at load (schema.Hardware), as (budget, where it
    came from). The two engines' values differ, so like-for-like engine rows pass one explicit int to both.
    naive:      the padded-token budget NaiveNaiveEngine.fit() measured on this card.
    batchinfer: its KV pool less (BLOCK_SIZE - 1) tokens per row. A row reserves whole blocks, at most 14 tokens past
                its share of the padded footprint, so a group within this budget fits the pool. With prefix
                sharing, blocks held for later groups can still overrun it; the scheduler refuses such a job before
                step 1.
    No card at all (analyze, CPU tests): DEFAULT_MAX_BATCH_TOKENS. A card with nothing measured is refused, so an
    unverified budget never reaches a group."""
    if cfg.engine == "naive" and hardware is not None and hardware.max_batch_tokens_fit:
        return hardware.max_batch_tokens_fit, f"measured fit on {hardware.gpu_name}"
    if cfg.engine == "batchinfer" and hardware is not None and hardware.kv_pool_tokens:
        tokens = hardware.kv_pool_tokens - cfg.max_batch_size * (BLOCK_SIZE - 1)
        if tokens < BLOCK_SIZE:
            raise ValueError(f"a KV pool of {hardware.kv_pool_tokens:,} tokens leaves {tokens} for groups of "
                             f"{cfg.max_batch_size} rows; lower max_batch_size or pass max_batch_tokens")
        return tokens, f"KV pool of {hardware.kv_pool_tokens:,} tokens"
    if hardware is None or hardware.total_gb is None:
        return DEFAULT_MAX_BATCH_TOKENS, "no card to measure: the default"
    raise ValueError(f"max_batch_tokens='auto' on {hardware.gpu_name} with nothing measured: call the naive engine's "
                     f"fit() before the job, or pass an int")


# prefill budget --------------------------------------------------------------------------------------

def auto_prefill_budget(metrics):
    """Spread the job's prefill evenly over the longest decode chain: every step then carries
    compute-bound prefill work while the memory-bound decode rows ride along, instead of a burst of
    big prefill steps followed by a tail of decode-only steps. Only mixed jobs land between the clamps:
    classify-only jobs (chain ~0) hit the ceiling, generate-only jobs the floor. The formula uses the
    raw prompt tokens even with prefix sharing on (mixed-quick: 583 per step): sizing it from the unique
    prefill (256) only delayed the start of the longest chain and cost steps (562 vs 534 vs 512 at 16384)
    for the same wall time, measured 2026-09-27."""
    chain = metrics["decode_chain"]
    if chain == 0:
        return MAX_PREFILL_BUDGET
    return min(max(math.ceil(metrics["prompt_tokens"] / chain), MIN_PREFILL_BUDGET), MAX_PREFILL_BUDGET)


# the decision ----------------------------------------------------------------------------------------

def decide(job, cfg, hardware=None):
    """JobAnalysis + PolicyConfig + Hardware -> Policy, the one constructor of Policy. hardware: what the engine
    measured on its card at load (engine.hardware), None without an engine (analyze, tests). Settings the chosen
    engine does not apply are None, so the recorded policy is exactly what governed the run."""
    cfg.validate()
    ours = cfg.engine == "batchinfer"
    if cfg.admission == "groups":
        budget_auto = cfg.max_batch_tokens == "auto"
        max_batch_tokens, source = (auto_max_batch_tokens(cfg, hardware) if budget_auto
                                    else (cfg.max_batch_tokens, None))
        groups = fixed_groups(job, cfg.order, max_batch_tokens, cfg.max_batch_size)
        admission_order = tuple(i for g in groups for i in g.members)
        stats = group_stats(job.requests, groups, max_batch_tokens, cfg.order, padded=not ours)
        if source:
            stats["max_batch_tokens_from"] = source
        max_batch_size = cfg.max_batch_size
    else:
        groups, stats, max_batch_tokens, max_batch_size, budget_auto = None, {}, None, None, None
        admission_order = tuple(ordered(job, cfg.order))
    if ours:
        adaptive = cfg.prefill_budget == "adaptive"  # starts from the auto value, the floor the scheduler raises from
        auto = adaptive or cfg.prefill_budget == "auto"
        chunk, budget = cfg.chunk_prefill, auto_prefill_budget(job.metrics) if auto else cfg.prefill_budget
        sharing = cfg.prefix_sharing
    else:  # the naive engine prefills each group in one forward and shares nothing
        chunk = budget = auto = adaptive = sharing = None
    return Policy(engine=cfg.engine, order=cfg.order, admission_order=admission_order, groups=groups,
                  admission=cfg.admission, chunk_prefill=chunk, prefill_budget=budget, prefill_budget_auto=auto,
                  prefill_budget_adaptive=adaptive, prefix_sharing=sharing, max_batch_tokens=max_batch_tokens,
                  max_batch_tokens_auto=budget_auto, max_batch_size=max_batch_size, stats=stats)


def describe_policy(policy, job, show=0):
    """Human-readable decisions: the knobs the engine applies, then, for fixed groups, their stats and the first
    `show` groups."""
    knobs = f"order={policy.order}, admission={policy.admission}"
    if policy.engine == "batchinfer":
        how =(" (adaptive: the auto value, raised per step when less decode work is left)"
               if policy.prefill_budget_adaptive else " (auto)" if policy.prefill_budget_auto else "")
        knobs += (f", chunk_prefill={policy.chunk_prefill}, prefill_budget={policy.prefill_budget}{how}, "
                  f"prefix_sharing={policy.prefix_sharing}")
    lines = [f"policy ({policy.engine} engine): {knobs}; admission order head {list(policy.admission_order[:10])}"]
    if policy.groups is None:
        return lines[0]
    st, items = policy.stats, job.requests
    padding = (f"; {st['padded_groups']} padded, padded prompt tokens {st['groups_padded_prompt_tokens']:,} "
               f"(waste {st['groups_padding_waste_pct']}%)" if "groups_padded_prompt_tokens" in st else "")
    source = f" (tokens auto: {st['max_batch_tokens_from']})" if "max_batch_tokens_from" in st else ""
    lines.append(f"fixed groups: {len(policy.groups)} groups ({st['oversized']} oversized) under "
                 f"{policy.max_batch_tokens:,} tokens x {policy.max_batch_size} rows{source}; in-batch shared prefix "
                 f"{st['in_batch_reuse']:.1%}{padding}")
    for gi, g in enumerate(policy.groups[:show]):
        rows = [items[i] for i in g.members]
        lens, outs = [r.prompt_len for r in rows], [r.req.max_tokens for r in rows]
        lines.append(f"  group {gi}: {len(rows)} rows {dict(Counter(r.kind for r in rows))}, prompt_len {min(lens)}-{max(lens)}, "
                     f"max_tokens {min(outs)}-{max(outs)}, shared prefix {g.shared_prefix_len} tok, "
                     f"footprint {padded_tokens(g.members, items):,} tok")
    return "\n".join(lines)
