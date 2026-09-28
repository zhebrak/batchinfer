"""policy: orders are permutations, fixed groups respect their budget, decide() resolves the config into one Policy."""
from pathlib import Path

import pytest

from batchinfer.analysis import analyze, lcp
from batchinfer.io import read_requests
from batchinfer.kv import BLOCK_SIZE, reservation_blocks
from batchinfer.policy import (DEAD_SLOT_RATIO, auto_prefill_budget, decide, describe_policy, fixed_groups, ordered,
                               padded_tokens)
from batchinfer.schema import (DEFAULT_BATCH_SIZE, DEFAULT_MAX_BATCH_TOKENS, MAX_PREFILL_BUDGET, MIN_PREFILL_BUDGET,
                               Hardware, PolicyConfig, Request)
from stub import StubTokenizer

FIXTURE = Path(__file__).parent / "fixtures" / "smoke.jsonl"


def req(i, prompt, max_tokens, **kw):
    return Request(id=str(i), prompt=prompt, max_tokens=max_tokens, **kw)


def job_of(reqs):
    return analyze(reqs, StubTokenizer())


def many(n=40):
    return [req(i, " ".join(f"w{i}_{j}" for j in range(3 + (i * 7) % 11)), 1 + (i * 5) % 17) for i in range(n)]


def groups_policy(job, order="max_tokens_desc", max_batch_tokens=DEFAULT_MAX_BATCH_TOKENS,
                  max_batch_size=DEFAULT_BATCH_SIZE, engine="naive"):
    """A fixed-groups decision; by default the one the naive engine runs, which shares nothing."""
    return decide(job, PolicyConfig(engine=engine, order=order, admission="groups", max_batch_tokens=max_batch_tokens,
                                    max_batch_size=max_batch_size, prefix_sharing=engine == "batchinfer"))


# order -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("order", ["input", "max_tokens_desc", "decode_ratio_desc", "prefix_dfs"])
def test_every_order_is_a_permutation(order):
    job = job_of(many())
    assert sorted(ordered(job, order)) == list(range(40))


def shared_groups():
    """Three prefix groups of 40 words (two full blocks) with 4-word suffixes, interleaved in input order.
    Group a holds one long generation, group c short ones, group b classification only."""
    max_tokens = {"a": [1, 200, 1, 1], "b": [1, 1, 1, 1], "c": [8, 8, 8, 8]}
    reqs = []
    for k in range(4):
        for g in "abc":
            reqs.append(req(f"{g}{k}", " ".join([f"{g}_{j}" for j in range(40)] + [f"{g}{k}_{j}" for j in range(4)]),
                            max_tokens[g][k]))
    return job_of(reqs)


def test_prefix_dfs_keeps_prefix_groups_adjacent_and_decode_heavy_first():
    job = shared_groups()
    order = ordered(job, "prefix_dfs")
    ids = [job.requests[i].req.id for i in order]
    # a (ratio 199/44 in it) first, its highest-ratio member first; then c (7/44); then b (0), in input order
    assert ids == ["a1", "a0", "a2", "a3", "c0", "c1", "c2", "c3", "b0", "b1", "b2", "b3"]
    for g in "abc":
        pos = [k for k, i in enumerate(ids) if i.startswith(g)]
        assert pos == list(range(pos[0], pos[0] + 4)), g


def test_max_tokens_desc_is_max_tokens_then_length_descending():
    job = job_of([req(0, "a b", 4), req(1, "a b c", 4), req(2, "a", 9), req(3, "a b c d", 1)])
    assert ordered(job, "max_tokens_desc") == [2, 1, 0, 3]


def test_decode_ratio_desc_is_stable_on_ties():
    # ratios: 3/2, 0, 3/2, 8/1, 0
    job = job_of([req(0, "a b", 4), req(1, "a b c", 1), req(2, "c d", 4), req(3, "e", 9), req(4, "f g", 1)])
    assert ordered(job, "decode_ratio_desc") == [3, 0, 2, 1, 4]


def test_unknown_order_raises():
    with pytest.raises(ValueError):
        ordered(job_of([]), "random")


# fixed groups ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("order", ["max_tokens_desc", "input"])
def test_partition_is_a_budgeted_partition(order):
    job = job_of(many())
    p = groups_policy(job, order, max_batch_tokens=120, max_batch_size=6)
    flat = list(p.admission_order)
    assert sorted(flat) == list(range(40)) and len(p.groups) > 1
    if order == "input":
        assert flat == list(range(40))
    else:
        keys = [(job.requests[i].req.max_tokens, job.requests[i].prompt_len) for i in flat]
        assert keys == sorted(keys, reverse=True)
    for g in p.groups:
        assert len(g.members) <= 6
        assert padded_tokens(g.members, job.requests) <= 120
    assert p.stats["oversized"] == 0 and p.order == order
    assert p.to_dict()["groups"] == len(p.groups)


def test_oversized_request_gets_its_own_group_and_is_counted():
    reqs = [req(0, "a b", 2), req(1, " ".join(str(j) for j in range(50)), 100), req(2, "c d", 2)]
    p = groups_policy(job_of(reqs), "input", max_batch_tokens=40, max_batch_size=8)
    assert [g.members for g in p.groups] == [[0], [1], [2]]
    assert p.stats["oversized"] == 1
    assert p.stats["padded_groups"] == 0


def test_fixture_groups_and_shared_prefix_with_batch_size_5():
    job = analyze(read_requests(FIXTURE), StubTokenizer())
    p = groups_policy(job, max_batch_tokens=100000, max_batch_size=5)
    ids = {it.req.id: i for i, it in enumerate(job.requests)}
    # sorted by max_tokens: 32, 24 | 16 | 8, 6 | 1, 1, 1. The dead-slot cut closes a group when its longest
    # max_tokens is at least DEAD_SLOT_RATIO times the next row's (32 vs 16, 16 vs 8, 8 vs 1).
    assert p.groups[0].members == [ids["g-2"], ids["g-3"]]
    assert p.groups[1].members == [ids["g-1"]]
    assert p.groups[2].members == [ids["g-4"], ids["l-1"]]
    # c-1 and c-3 tie on stub length; a stable sort keeps input order, so compare as a set
    assert set(p.groups[3].members) == {ids[x] for x in ("c-1", "c-2", "c-3")} and len(p.groups) == 4
    c = [job.requests[i].token_ids for i in p.groups[3].members]
    expected = min(lcp(c[0], c[1]), lcp(c[0], c[2]))
    assert p.groups[3].shared_prefix_len == expected == len("you are a strict sentiment judge . answer positive or negative only . review :".split())
    assert p.groups[1].shared_prefix_len == 0
    st = p.stats
    assert st["in_batch_reuse"] > 0 and st["padded_groups"] == 3 and st["dead_slot_ratio"] == DEAD_SLOT_RATIO
    assert st["groups_padded_prompt_tokens"] > sum(it.prompt_len for it in job.requests)
    text = describe_policy(p, job, show=4)
    assert text.startswith("policy (naive engine): order=max_tokens_desc, admission=groups;")
    assert "fixed groups: 4 groups" in text and "under 100,000 tokens x 5 rows" in text and "group 3: 3 rows" in text
    assert "3 padded, padded prompt tokens" in text and "chunk_prefill" not in text and "prefill_budget" not in text


def test_dead_slot_cut_only_under_max_tokens_desc():
    job = job_of([req(i, "a b", m) for i, m in enumerate([8, 512, 1, 256, 8])])
    p = groups_policy(job, "max_tokens_desc", max_batch_tokens=100000, max_batch_size=64)
    assert [[job.requests[i].req.max_tokens for i in g.members] for g in p.groups] == [[512], [256], [8, 8], [1]]
    p = groups_policy(job, "input", max_batch_tokens=100000, max_batch_size=64)
    assert [g.members for g in p.groups] == [[0, 1, 2, 3, 4]] and p.stats["dead_slot_ratio"] is None


def test_single_member_group_has_no_shared_prefix():
    p = groups_policy(job_of([req(0, "a b c", 2)]), max_batch_tokens=100, max_batch_size=1)
    assert p.groups[0].shared_prefix_len == 0 and p.stats["in_batch_reuse"] == 0.0


# decide ----------------------------------------------------------------------------------------------

def test_auto_budget_spreads_prefill_over_the_decode_chain_within_clamps():
    assert auto_prefill_budget({"prompt_tokens": 297_536, "decode_chain": 511}) == 583
    assert auto_prefill_budget({"prompt_tokens": 50_000, "decode_chain": 511}) == MIN_PREFILL_BUDGET  # generate-like
    assert auto_prefill_budget({"prompt_tokens": 250_000, "decode_chain": 3}) == MAX_PREFILL_BUDGET  # classify-like
    assert auto_prefill_budget({"prompt_tokens": 1000, "decode_chain": 0}) == MAX_PREFILL_BUDGET
    # prefix sharing does not change the formula (see the docstring: the unique-token variant only cost steps)
    mixed = {"prompt_tokens": 297_536, "unique_prefill_tokens": 58_160, "decode_chain": 511}
    assert auto_prefill_budget(mixed) == 583


def test_decide_resolves_the_config():
    job = job_of(many())
    p = decide(job, PolicyConfig(order="decode_ratio_desc", admission="continuous", chunk_prefill=True,
                                 prefill_budget=512))
    assert p.order == "decode_ratio_desc" and p.admission_order == tuple(ordered(job, "decode_ratio_desc"))
    assert p.groups is None and p.stats == {}
    assert p.prefill_budget == 512 and not p.prefill_budget_auto
    assert p.chunk_prefill is True and p.admission == "continuous"
    assert p.prefix_sharing is True  # the default; off must be asked for
    assert decide(job, PolicyConfig(prefix_sharing=False)).prefix_sharing is False
    assert p.max_batch_tokens is None and p.max_batch_size is None  # the fixed-groups budget plays no part
    auto = decide(job, PolicyConfig(prefill_budget="auto"))
    assert auto.order == "prefix_dfs" and auto.prefill_budget == auto_prefill_budget(job.metrics)
    assert auto.prefill_budget_auto and auto.prefill_budget_adaptive is False and p.prefill_budget_adaptive is False
    assert "(auto)" in describe_policy(auto, job)
    default = decide(job, PolicyConfig())  # the widest step the engine sizes for, held for the whole job
    assert default.prefill_budget == MAX_PREFILL_BUDGET and not default.prefill_budget_auto
    assert default.prefill_budget_adaptive is False and default.to_dict()["prefill_budget_adaptive"] is False
    adaptive = decide(job, PolicyConfig(prefill_budget="adaptive"))  # the auto value, as the floor it raises from
    assert adaptive.prefill_budget == auto.prefill_budget and adaptive.prefill_budget_auto
    assert adaptive.prefill_budget_adaptive and "(adaptive:" in describe_policy(adaptive, job)
    shared = decide(job, PolicyConfig(order="prefix_dfs", prefix_sharing=True))
    assert shared.prefix_sharing is True and shared.admission_order == tuple(ordered(job, "prefix_dfs"))
    text = describe_policy(shared, job)
    assert "order=prefix_dfs" in text and "prefix_sharing=True" in text and f"prefill_budget={MAX_PREFILL_BUDGET}," in text
    assert "fixed groups" not in text


def test_groups_admission_uses_the_fixed_groups():
    job = job_of(many())
    p = groups_policy(job, "max_tokens_desc", max_batch_tokens=120, max_batch_size=6)
    assert p.groups == fixed_groups(job, "max_tokens_desc", 120, 6)
    assert list(p.admission_order) == [i for g in p.groups for i in g.members]
    assert (p.max_batch_tokens, p.max_batch_size) == (120, 6)
    d = p.to_dict()
    assert d["order"] == "max_tokens_desc" and d["groups"] == len(p.groups) and d["max_batch_tokens"] == 120
    assert d["admission_order_head"] == list(p.admission_order[:20]) and d["padded_groups"] == p.stats["padded_groups"]


def test_a_policy_holds_only_what_its_engine_applies():
    job = job_of(many())
    naive = groups_policy(job, "max_tokens_desc", max_batch_tokens=120, max_batch_size=6)
    assert naive.engine == "naive" and naive.prefix_sharing is None
    assert naive.chunk_prefill is None and naive.prefill_budget is None and naive.prefill_budget_auto is None
    assert naive.prefill_budget_adaptive is None
    assert {"padded_groups", "groups_padded_prompt_tokens", "groups_padding_waste_pct"} <= set(naive.stats)
    ours = groups_policy(job, "max_tokens_desc", max_batch_tokens=120, max_batch_size=6, engine="batchinfer")
    assert ours.groups == naive.groups and ours.admission_order == naive.admission_order  # the same partition
    assert ours.chunk_prefill is True and ours.prefill_budget == PolicyConfig.prefill_budget
    assert set(ours.stats) == {"oversized", "in_batch_reuse", "dead_slot_ratio"}  # the batchinfer engine never pads


@pytest.mark.parametrize("cfg", [PolicyConfig(prefill_budget=MAX_PREFILL_BUDGET + 1), PolicyConfig(prefill_budget=0),
                                 PolicyConfig(admission="lazy"), PolicyConfig(order="random"),
                                 PolicyConfig(prefix_sharing="off"), PolicyConfig(chunk_prefill="true"),  # bool('off') is True
                                 PolicyConfig(max_batch_tokens="Auto"), PolicyConfig(max_batch_tokens="65536"),
                                 PolicyConfig(max_batch_tokens=0), PolicyConfig(max_batch_size=0),
                                 PolicyConfig(engine="gpu"), PolicyConfig(prefill_budget=True),  # True is an int
                                 PolicyConfig(max_batch_tokens=True), PolicyConfig(max_batch_size=True)])
def test_decide_rejects_bad_config(cfg):
    with pytest.raises(ValueError):
        decide(job_of(many(4)), cfg)


@pytest.mark.parametrize("kw,match", [({"admission": "continuous"}, "fixed groups only"),
                                      ({"prefix_sharing": True}, "prefix_sharing needs the batchinfer engine"),
                                      ({"prefill_budget": 512}, "batchinfer-engine settings"),
                                      ({"chunk_prefill": False}, "batchinfer-engine settings")])
def test_naive_engine_configs_it_cannot_run_are_refused(kw, match):
    cfg = PolicyConfig(**{"engine": "naive", "order": "max_tokens_desc", "admission": "groups",
                          "prefix_sharing": False, **kw})
    with pytest.raises(ValueError, match=match):
        cfg.validate()


@pytest.mark.parametrize("budget", ["adaptive", "auto", MAX_PREFILL_BUDGET])  # the modes and the default: left to the engine
def test_naive_engine_takes_a_budget_mode_it_has_nothing_to_apply_to(budget):
    job = job_of(many())
    cfg = PolicyConfig(engine="naive", order="max_tokens_desc", admission="groups", prefix_sharing=False,
                       prefill_budget=budget)
    policy = decide(job, cfg)
    assert policy.prefill_budget is None and policy.prefill_budget_adaptive is None and policy.prefill_budget_auto is None


A100 = {"gpu_name": "NVIDIA A100-SXM4-40GB", "total_gb": 39.39}
NAIVE_GROUPS = {"engine": "naive", "order": "max_tokens_desc", "admission": "groups", "prefix_sharing": False}


def test_auto_budget_is_what_the_engine_measured():
    job = job_of(many())
    fit = Hardware(**A100, max_batch_tokens_fit=77_824, kv_pool_tokens=None)
    static = decide(job, PolicyConfig(**NAIVE_GROUPS), fit)
    assert static.max_batch_tokens == 77_824 and static.max_batch_tokens_auto is True
    assert static.stats["max_batch_tokens_from"] == "measured fit on NVIDIA A100-SXM4-40GB"
    assert "77,824 tokens x 64 rows (tokens auto: measured fit on NVIDIA A100-SXM4-40GB)" in describe_policy(static, job)
    assert static.to_dict()["max_batch_tokens_auto"] is True
    pool = Hardware(**A100, max_batch_tokens_fit=None, kv_pool_tokens=144_176)
    step = decide(job, PolicyConfig(admission="groups", max_batch_size=8), pool)
    assert step.max_batch_tokens == 144_176 - 8 * (BLOCK_SIZE - 1) and step.max_batch_tokens_auto is True
    explicit = decide(job, PolicyConfig(**NAIVE_GROUPS, max_batch_tokens=500), fit)
    assert explicit.max_batch_tokens == 500 and explicit.max_batch_tokens_auto is False
    assert "max_batch_tokens_from" not in explicit.stats
    for no_card in (None, Hardware(gpu_name=None, total_gb=None, max_batch_tokens_fit=None, kv_pool_tokens=None)):
        p = decide(job, PolicyConfig(**NAIVE_GROUPS), no_card)  # analyze, or the naive engine on the CPU
        assert p.max_batch_tokens == DEFAULT_MAX_BATCH_TOKENS and p.stats["max_batch_tokens_from"].startswith("no card")
    continuous = decide(job, PolicyConfig(), pool)
    assert continuous.max_batch_tokens is None and continuous.max_batch_tokens_auto is None


def test_auto_budget_refuses_an_unmeasured_card_and_a_pool_too_small_for_its_rows():
    job = job_of(many(4))
    with pytest.raises(ValueError, match="nothing measured"):  # an unverified budget never reaches a group
        decide(job, PolicyConfig(**NAIVE_GROUPS), Hardware(**A100, max_batch_tokens_fit=None, kv_pool_tokens=None))
    tiny = Hardware(gpu_name=None, total_gb=None, max_batch_tokens_fit=None, kv_pool_tokens=39 * BLOCK_SIZE)
    with pytest.raises(ValueError, match="lower max_batch_size"):
        decide(job, PolicyConfig(admission="groups", max_batch_size=64), tiny)


def test_step_auto_groups_fit_the_pool():
    """A row reserves whole blocks, up to 14 tokens past its share of the padded footprint: the batchinfer engine's auto
    budget leaves that per row, so every group it forms fits the pool whole."""
    job, pool = job_of(many(200)), 400
    p = decide(job, PolicyConfig(admission="groups", max_batch_size=8, prefix_sharing=False),
               Hardware(gpu_name=None, total_gb=None, max_batch_tokens_fit=None, kv_pool_tokens=pool))
    items = job.requests
    assert len(p.groups) > 10 and p.stats["oversized"] == 0
    for g in p.groups:
        assert sum(reservation_blocks(items[i].prompt_len, items[i].req.max_tokens) for i in g.members) * BLOCK_SIZE <= pool


def test_engine_defaults_have_one_source():
    """The bench adapter and the CLI take the batchinfer engine's defaults from PolicyConfig, so flipping a default
    is one edit and every entry point runs the same policy. The naive engine's are its own: fixed groups in arrival
    order (NAIVE_ORDER), no sharing."""
    import argparse
    import inspect

    from batchinfer import cli
    from batchinfer.bench import BatchinferBackend, NaiveBackend
    from batchinfer.schema import NAIVE_ORDER
    params = inspect.signature(BatchinferBackend._policy_config).parameters
    assert params["order"].default == PolicyConfig.order == "prefix_dfs"
    assert params["prefix_sharing"].default is PolicyConfig.prefix_sharing is True
    assert params["prefill_budget"].default == PolicyConfig.prefill_budget == MAX_PREFILL_BUDGET
    assert params["max_batch_tokens"].default == PolicyConfig.max_batch_tokens == "auto"
    assert inspect.signature(BatchinferBackend.__init__).parameters["reserve_gb"].default == "auto"  # sized on the card
    assert inspect.signature(NaiveBackend._policy_config).parameters["order"].default == NAIVE_ORDER == "input"
    parser = argparse.ArgumentParser()
    cli.common(parser)
    unset = parser.parse_args(["--input", "w.jsonl"])
    assert unset.model == "Qwen/Qwen3-1.7B"
    assert cli.resolve_defaults(argparse.Namespace(**vars(unset))).engine == PolicyConfig.engine == "batchinfer"
    ours = cli.resolve_defaults(argparse.Namespace(**{**vars(unset), "engine": "batchinfer"}))
    assert (ours.order, ours.admission, ours.prefix_sharing) == (PolicyConfig.order, PolicyConfig.admission,
                                                                PolicyConfig.prefix_sharing)
    assert ours.max_batch_tokens == PolicyConfig.max_batch_tokens  # "auto": what the engine measures on its card
    naive = cli.resolve_defaults(argparse.Namespace(**{**vars(unset), "engine": "naive"}))
    assert (naive.order, naive.admission, naive.prefix_sharing) == ("input", "groups", False)
