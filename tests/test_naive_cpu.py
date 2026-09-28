"""The naive engine on a tiny random Qwen3 in fp32 on CPU. Needs torch + transformers; skipped without them.

Three comparisons on a group [short, long] with different max_tokens: (a) batched equals each request
decoded alone by the engine; (b) both equal an independent reference that recomputes the full
sequence with no cache and no padding; (c) logits are finite and the engine never called encode.
(a) alone passes with a cache-slot off-by-one, because both paths share it; (b) catches it."""
import dataclasses

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from batchinfer import flow  # noqa: E402
from batchinfer.analysis import analyze  # noqa: E402
from batchinfer.metrics import Metrics  # noqa: E402
from batchinfer.naive import NaiveEngine  # noqa: E402
from batchinfer.policy import decide  # noqa: E402
from batchinfer.schema import Hardware, PolicyConfig, Request  # noqa: E402

VOCAB, EOS, PAD = 64, 1, 0
PROMPTS = {"short": [5, 9, 17], "long": [3, 8, 21, 34, 2, 40, 11, 27, 19]}


class FixedTok:
    """Maps the two prompt names to fixed ids inside the tiny vocab; counts calls."""
    name_or_path = "fixed"
    pad_token_id = PAD

    def __init__(self):
        self.calls = 0

    def __call__(self, texts, add_special_tokens=False):
        self.calls += 1
        return {"input_ids": [list(PROMPTS[t]) for t in texts]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


@pytest.fixture(scope="module")
def model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
                      tie_word_embeddings=False, eos_token_id=EOS, pad_token_id=PAD, bos_token_id=None,
                      attn_implementation="sdpa")
    return Qwen3ForCausalLM(cfg).eval()


def reference(model, prompt_ids, max_new, ignore_eos):
    """Greedy decode by full recompute: no cache, no padding, no position bookkeeping to get wrong."""
    ids, out = list(prompt_ids), []
    with torch.inference_mode():
        for _ in range(max_new):
            logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
            assert torch.isfinite(logits).all()
            t = int(logits.argmax())
            out.append(t)
            ids.append(t)
            if t == EOS and not ignore_eos:
                break
    return out


def groups(batch_size, max_batch_tokens=10000):
    return PolicyConfig(engine="naive", order="max_tokens_desc", admission="groups", prefix_sharing=False,
                        max_batch_tokens=max_batch_tokens, max_batch_size=batch_size)


def run(model, tok, reqs, batch_size):
    engine = NaiveEngine("tiny", tok, device="cpu", model=model)
    out, m = [], Metrics()
    calls = tok.calls
    flow.run(reqs, tok, engine, groups(batch_size), out.append, m)  # as the CLI and NaiveBackend do
    assert tok.calls == calls + 1, "analysis tokenizes once; the engine never does"
    return {r.id: r for r in out}, m


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_batched_equals_single_equals_reference(model, ignore_eos):
    tok = FixedTok()
    reqs = [Request(id="short", prompt="short", max_tokens=4, ignore_eos=ignore_eos),
            Request(id="long", prompt="long", max_tokens=7, ignore_eos=ignore_eos)]
    batched, m = run(model, tok, reqs, batch_size=2)
    single, m_single = run(model, tok, reqs, batch_size=1)
    for record, out in ((m, batched), (m_single, single)):
        check_request_trace(record, out)
    for r in reqs:
        ref = reference(model, PROMPTS[r.prompt], r.max_tokens, ignore_eos)
        assert batched[r.id].token_ids == single[r.id].token_ids == ref, r.id
        assert batched[r.id].output_tokens == len(ref) <= r.max_tokens
        assert batched[r.id].prompt_tokens == len(PROMPTS[r.prompt])
        if ignore_eos:
            assert len(ref) == r.max_tokens and batched[r.id].finish_reason == "length"
        else:
            assert batched[r.id].finish_reason == ("stop" if ref[-1] == EOS else "length")
    assert m.counts["requests"] == 2 and m.policy["groups"] == 1
    # every live decode slot yields one token; each row's first token comes from prefill
    assert m.counts["output_tokens"] == m.counts["requests"] + m.counts["live_slot_steps"]
    # one group: short (4 tokens) dies before long (7), so dead slots appear unless EOS cut long first
    assert m.counts["decode_slot_steps"] >= m.counts["live_slot_steps"]
    assert m.counts["decode_steps"] == max(len(batched["short"].token_ids), len(batched["long"].token_ids)) - 1


def check_request_trace(m, out):
    """Each request's times sit inside its group's measured span; groups carry their real prompt tokens and run
    one after another; there are no steps to index."""
    d = m.to_dict()
    trace = [dict(zip(d["request_trace"], row)) for row in zip(*d["request_trace"].values())]
    assert sorted(t["id"] for t in trace) == sorted(out)
    starts = [g["start_ms"] for g in d["groups"]]
    assert starts == sorted(starts)
    for t in trace:
        g = d["groups"][t["group"]]
        end = g["start_ms"] + 1000 * (g["prefill_s"] + g["decode_s"]) + 0.1  # both rounded to 0.1 ms in the record
        assert t["admitted_ms"] == t["prefill_start_ms"] == g["start_ms"]
        assert g["start_ms"] <= t["first_token_ms"] <= t["finished_ms"] <= end
        assert {t[f"{e}_step"] for e in ("admitted", "prefill_start", "first_token", "finished")} == {None}
        assert t["output_tokens"] == out[t["id"]].output_tokens and t["prefix_hit_tokens"] == 0
    for g in d["groups"]:
        assert g["prompt_tokens"] == sum(t["prompt_len"] for t in trace if t["group"] == g["group"])


def test_prefill_only_group_takes_no_decode_step(model):
    tok = FixedTok()
    reqs = [Request(id="short", prompt="short", max_tokens=1), Request(id="long", prompt="long", max_tokens=1)]
    out, m = run(model, tok, reqs, batch_size=2)
    for r in reqs:
        assert out[r.id].token_ids == reference(model, PROMPTS[r.prompt], 1, False)
    assert m.counts["decode_steps"] == 0 and m.counts["output_tokens"] == 2


def test_engine_refuses_a_policy_it_cannot_run(model):
    tok = FixedTok()
    reqs = [Request(id="short", prompt="short", max_tokens=2), Request(id="long", prompt="long", max_tokens=2)]
    job = analyze(reqs, tok)
    policy = decide(job, groups(batch_size=2))
    engine = NaiveEngine("tiny", tok, device="cpu", model=model)
    tampered = dataclasses.replace(policy, max_batch_tokens=5)  # after deciding: the engine must notice, not re-split
    with pytest.raises(ValueError, match="never re-splits"):
        engine.run(job, tampered, lambda r: None, Metrics())
    with pytest.raises(ValueError, match="decided for engine='naive', not 'batchinfer'"):
        engine.run(job, decide(job, PolicyConfig()), lambda r: None, Metrics())  # a batchinfer-engine policy
    m = Metrics()
    engine.run(job, policy, lambda r: None, m)
    assert "probe_peak_reserved_gb" not in m.memory  # recorded by the engine on a GPU, next to its run peaks



def test_a_lone_request_over_the_budget_is_refused_before_any_forward(model):
    """The budget binds a group of one too: the probe exercised max_batch_tokens and nothing past it. Before, one
    request alone ran at any size (a 70,001-token footprint passed under a 4,096-token budget)."""
    tok = FixedTok()
    job = analyze([Request(id="long", prompt="long", max_tokens=8)], tok)
    footprint = job.requests[0].prompt_len + 8
    policy = decide(job, groups(batch_size=1, max_batch_tokens=footprint - 1))
    assert [g.members for g in policy.groups] == [[0]] and policy.stats["oversized"] == 1
    engine, results = NaiveEngine("tiny", tok, device="cpu", model=model), []
    with pytest.raises(ValueError, match=f"{footprint} padded tokens vs 1 rows, {footprint - 1} tokens.*larger"):
        engine.run(job, policy, results.append, Metrics())
    assert results == []
    engine.run(job, decide(job, groups(batch_size=1, max_batch_tokens=footprint)), results.append, Metrics())
    assert [r.id for r in results] == ["long"]

def test_off_the_gpu_nothing_is_measured(model):
    """Hardware reaches policy only as what the engine measured on its card. On the CPU there is no card: fit()
    measures nothing and an auto budget is policy's no-card default."""
    engine = NaiveEngine("tiny", FixedTok(), device="cpu", model=model)
    assert engine.fit() is None
    assert engine.hardware == Hardware(gpu_name=None, total_gb=None, max_batch_tokens_fit=None, kv_pool_tokens=None)


def test_a_group_with_rows_longer_than_the_fit_priced_is_refused_before_any_forward(model):
    """fit() priced padded tokens in 4,096-token rows and measured what longer rows add; a group predicted past 95%
    of the card fails before the first forward, naming the group. The costs are set by hand: this runs on the CPU."""
    tok = FixedTok()
    reqs = [Request(id="short", prompt="short", max_tokens=2), Request(id="long", prompt="long", max_tokens=2)]
    job = analyze(reqs, tok)
    engine = NaiveEngine("tiny", tok, device="cpu", model=model)
    policy = decide(job, groups(batch_size=2))
    engine._cost = (1.0, 0.0, 0.0, 2.0)  # 1 of 2 GiB in use, tokens free: fits
    engine.run(job, policy, lambda r: None, Metrics())
    engine._cost = (1.0, 0.1, 0.0, 2.0)  # 0.1 GiB per padded token: this group's footprint alone passes 95% of 2 GiB
    with pytest.raises(ValueError, match="group 0 .* over the 95% limit"):
        engine.run(job, policy, lambda r: None, Metrics())
