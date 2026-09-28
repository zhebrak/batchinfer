"""batchinfer engine on a tiny random Qwen3 in fp32 on CPU, through the torch attention backend with scattered
KV blocks. Needs torch + transformers; skipped without them.

Token ids must equal an independent reference exactly (greedy by full recompute: no cache, no paging,
no positions to get wrong) and the naive engine, for every knob combination, including budgets so
small that every prompt is split into several prefill chunks."""
import copy
import dataclasses

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from batchinfer import flow  # noqa: E402
from batchinfer.naive import NaiveEngine  # noqa: E402
from batchinfer.executor import Executor, _paged_attention, check_supported  # noqa: E402
from batchinfer.kv import BLOCK_SIZE  # noqa: E402
from batchinfer.metrics import Metrics  # noqa: E402
from batchinfer.schema import Hardware, PolicyConfig, Request, Row, Step  # noqa: E402
from batchinfer.step_engine import StepEngine  # noqa: E402

VOCAB, EOS, PAD = 64, 1, 0
PREFIX = [3, 8, 21, 34, 2, 40, 11, 27, 19, 5, 9, 17, 4, 6, 12, 30, 22, 7, 13, 44, 18, 25, 36, 50, 29, 31, 60, 14, 23, 41,
          16, 33]  # 32 tokens: two full KV blocks, shareable
PROMPTS = {"short": [5, 9, 17], "long": [3, 8, 21, 34, 2, 40, 11, 27, 19],
           "mid": [4, 6, 12, 30, 22, 7], "longer": [13, 3, 44, 9, 18, 25, 36, 5, 50, 29, 31, 60, 2, 14, 23, 41, 8, 16, 33],
           "sharedA": PREFIX + [5, 9, 17, 2], "sharedB": PREFIX + [40, 11, 27],
           "sharedC": PREFIX[:16] + [45, 46, 47, 48, 49, 51]}  # shares the first block only


class FixedTok:
    """Maps prompt names to fixed ids inside the tiny vocab; counts calls."""
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
def ref_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256,
                      tie_word_embeddings=False, eos_token_id=EOS, pad_token_id=PAD, bos_token_id=None,
                      attn_implementation="sdpa")
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def executor(ref_model):
    return Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False)


def reference(model, prompt_ids, max_new, ignore_eos):
    ids, out = list(prompt_ids), []
    with torch.inference_mode():
        for _ in range(max_new):
            t = int(model(input_ids=torch.tensor([ids])).logits[0, -1].argmax())
            out.append(t)
            ids.append(t)
            if t == EOS and not ignore_eos:
                break
    return out


def requests(ignore_eos):
    """Four unrelated prompts, three sharing one or two KV blocks, and a duplicate of one of them."""
    reqs = [Request(id=name, prompt=name, max_tokens=mt, ignore_eos=ignore_eos)
            for name, mt in (("short", 6), ("long", 9), ("mid", 1), ("longer", 12),
                             ("sharedA", 7), ("sharedB", 5), ("sharedC", 4))]
    return reqs + [Request(id="dupA", prompt="sharedA", max_tokens=3, ignore_eos=ignore_eos)]


def run_step(executor, reqs, cfg):
    tok = FixedTok()
    out, m = [], Metrics()
    flow.run(reqs, tok, StepEngine(executor, tok), cfg, out.append, m)
    assert tok.calls == 1, "analysis tokenizes once; the engine never does"
    return {r.id: r for r in out}, m


OFF = dict(prefix_sharing=False)  # the default is on; CONFIGS holds both
BASE = [PolicyConfig(order="input", admission="continuous", chunk_prefill=True, prefill_budget=4, **OFF),
        PolicyConfig(order="decode_ratio_desc", admission="continuous", chunk_prefill=True, prefill_budget=1, **OFF),
        PolicyConfig(order="max_tokens_desc", admission="continuous", chunk_prefill=False, prefill_budget=8, **OFF),
        PolicyConfig(order="max_tokens_desc", admission="groups", chunk_prefill=False, prefill_budget=16384,
                     max_batch_tokens=40, max_batch_size=2, **OFF),
        PolicyConfig(order="prefix_dfs", admission="continuous", chunk_prefill=True, prefill_budget=4, **OFF)]
CONFIGS = BASE + [dataclasses.replace(c, prefix_sharing=True) for c in BASE]


def cfg_id(c):
    return f"{c.order}-{c.admission}-chunk{c.chunk_prefill}-{c.prefill_budget}-share{c.prefix_sharing}"


@pytest.mark.parametrize("ignore_eos", [False, True])
@pytest.mark.parametrize("cfg", CONFIGS, ids=cfg_id)
def test_step_engine_equals_reference(ref_model, executor, cfg, ignore_eos):
    reqs = requests(ignore_eos)
    got, m = run_step(executor, reqs, cfg)
    for r in reqs:
        ref = reference(ref_model, PROMPTS[r.prompt], r.max_tokens, ignore_eos)
        assert got[r.id].token_ids == ref, r.id
        assert got[r.id].finish_reason == ("stop" if ref[-1] == EOS and not ignore_eos else "length")
    assert m.counts["output_tokens"] == sum(len(g.token_ids) for g in got.values())
    assert len(m.steps) == m.step_rates()["steps"] > 0
    prompt_tokens = sum(len(PROMPTS[r.prompt]) for r in reqs)
    assert m.counts["prompt_tokens"] == prompt_tokens
    check_request_trace(m, got)
    if cfg.prefix_sharing:
        # sharedB and dupA hit both PREFIX blocks, sharedC the first: 5 blocks never computed twice
        assert m.counts["prefix_hit_tokens"] == 5 * 16 and m.counts["prefill_tokens"] == prompt_tokens - 5 * 16
        assert m.rates()["prefix_hit_pct"] == round(100 * 5 * 16 / prompt_tokens, 2)
    else:
        assert m.counts["prefix_hit_tokens"] == 0 and m.counts["prefill_tokens"] == prompt_tokens


def check_request_trace(m, got):
    """One trace entry per result, its events in step order, its times read off the step trace's end_ms."""
    d = m.to_dict()
    trace = [dict(zip(d["request_trace"], row)) for row in zip(*d["request_trace"].values())]
    ends = d["steps"]["end_ms"]
    assert sorted(t["id"] for t in trace) == sorted(got)
    assert all(a < b for a, b in zip(ends, ends[1:]))  # one origin for the whole run, not a per-step one
    for t in trace:
        assert t["admitted_step"] <= t["prefill_start_step"] <= t["first_token_step"] <= t["finished_step"] < len(ends)
        assert t["finished_ms"] == ends[t["finished_step"]] and t["first_token_ms"] == ends[t["first_token_step"]]
        assert t["prefill_start_ms"] == (ends[t["prefill_start_step"] - 1] if t["prefill_start_step"] else 0.0)
        assert t["admitted_ms"] <= t["prefill_start_ms"]
        assert t["output_tokens"] == got[t["id"]].output_tokens and t["prompt_len"] == got[t["id"]].prompt_tokens
        assert t["finish_reason"] == got[t["id"]].finish_reason
    assert sum(t["prefix_hit_tokens"] for t in trace) == m.counts["prefix_hit_tokens"]


def test_batchinfer_engine_equals_naive_engine(ref_model, executor):
    reqs = requests(ignore_eos=True)
    got, _ = run_step(executor, reqs, CONFIGS[0])
    tok = FixedTok()
    cfg = PolicyConfig(engine="naive", order="max_tokens_desc", admission="groups", prefix_sharing=False,
                       max_batch_tokens=10_000, max_batch_size=4)
    naive = []
    flow.run(reqs, tok, NaiveEngine("tiny", tok, device="cpu", model=copy.deepcopy(ref_model)), cfg, naive.append,
             Metrics())
    for r in naive:
        assert got[r.id].token_ids == r.token_ids, r.id
    with pytest.raises(ValueError, match="decided for engine='batchinfer', not 'naive'"):
        flow.run(reqs, tok, StepEngine(executor, tok), cfg, naive.append, Metrics())  # a naive-engine policy


def test_the_step_engine_reports_its_pool(executor):
    """Its fixed-groups budget comes from the pool, which exists off the GPU too (sizing the pool needs a card:
    test_executor_needs_its_batch_and_a_pool_size_off_the_gpu)."""
    hw = StepEngine(executor, FixedTok()).hardware
    assert hw == Hardware(gpu_name=None, total_gb=None, max_batch_tokens_fit=None, kv_pool_tokens=39 * BLOCK_SIZE)


def test_both_engines_refuse_a_request_past_the_models_positions(ref_model, executor):
    """The tiny model has 256 positions; 'long' (9 prompt tokens) with 250 new tokens needs 258 (0 .. 257). It is
    second in input order and in a group of its own, and still nothing runs: the refusal comes before any forward."""
    reqs = [Request(id="short", prompt="short", max_tokens=2), Request(id="long", prompt="long", max_tokens=250)]
    naive = PolicyConfig(engine="naive", order="input", admission="groups", prefix_sharing=False,
                         max_batch_tokens=10_000, max_batch_size=1)
    engines = [(StepEngine(executor, FixedTok()), PolicyConfig(order="input")),
               (NaiveEngine("tiny", FixedTok(), device="cpu", model=copy.deepcopy(ref_model)), naive)]
    for engine, cfg in engines:
        out = []
        with pytest.raises(ValueError, match="request long needs 258 positions.* the model has 256"):
            flow.run(reqs, FixedTok(), engine, cfg, out.append, Metrics())
        assert out == [], type(engine).__name__


@pytest.mark.parametrize("kw", [{"sliding_window": 8}, {"softcap": 50.0}, {"s_aux": torch.zeros(4)}])
def test_attention_refuses_what_a_layer_asks_for_beyond_full_causal(kw):
    import types
    layer = types.SimpleNamespace(layer_idx=3)
    with pytest.raises(NotImplementedError, match=f"layer 3 asks for .*{next(iter(kw))}"):
        _paged_attention(layer, None, None, None, None, **kw)
    with pytest.raises(RuntimeError, match="needs the step's batch"):  # None is what a full-attention layer passes
        _paged_attention(layer, None, None, None, None, sliding_window=None, softcap=None)


def test_a_model_with_a_sliding_window_layer_fails_on_its_first_forward(ref_model):
    """End to end: HF passes the layer's window to the attention function, which refuses it."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(**{**ref_model.config.to_dict(), "use_sliding_window": True, "sliding_window": 8,
                         "layer_types": ["full_attention", "sliding_attention"]})
    ex = Executor("windowed", device="cpu", model=Qwen3ForCausalLM(cfg).eval(), num_blocks=8, probe=False)
    with pytest.raises(NotImplementedError, match="layer 1 asks for .*sliding_window"):
        ex.forward(Step([Row(seq=0, start=0, token_ids=[5, 9, 17], block_ids=[1], sample=True, decode=False)]))


def test_executor_refuses_heads_that_do_not_split_over_kv_heads():
    import types
    with pytest.raises(ValueError, match="6 heads over 4 KV heads"):
        check_supported(types.SimpleNamespace(num_attention_heads=6, num_key_value_heads=4), "some/model")


def test_executor_needs_its_batch_and_a_pool_size_off_the_gpu(ref_model, executor):
    with pytest.raises(RuntimeError, match="needs the step's batch"):
        _paged_attention(None, None, None, None, None)  # e.g. the model called directly, outside Executor.logits
    with pytest.raises(ValueError, match="num_blocks is required off the GPU"):
        Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), probe=False)
    assert executor.cuda is False and executor.max_positions == 256


def test_build_options_resolve_to_what_can_run(ref_model):
    """fused_layers and cuda_graphs default to "auto" at the edges: on exactly where they can run. Off the GPU (the
    torch backend) neither can, so auto resolves both off and the engine records that; asking for either outright is
    refused with the reason, and anything but a bool or "auto" is refused."""
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False,
                  fused_layers="auto", cuda_graphs="auto")
    assert ex.facts == {"fused_layers": False, "cuda_graphs": False, "fa_version": None}
    for option in ("fused_layers", "cuda_graphs"):
        with pytest.raises(ValueError, match=f"{option}: .*fa backend"):
            Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False, **{option: True})
        with pytest.raises(ValueError, match="true, false or 'auto'"):
            Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False, **{option: "on"})
