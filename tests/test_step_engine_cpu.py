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
from batchinfer.metrics import Metrics  # noqa: E402
from batchinfer.analysis import analyze  # noqa: E402
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


@pytest.fixture(scope="module", params=[16, 32], ids=["block16", "block32"])
def executor(request, ref_model):
    """The engine under test at two block sizes: the shared PREFIX is two blocks at 16 and one at 32, and sharedC's
    half-block share is a block share at 16 and a token-level one at 32."""
    return Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False,
                    block_size=request.param)


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


OFF = dict(sharing="none")  # the default is on; CONFIGS holds both
BASE = [PolicyConfig(order="input", admission="continuous", chunk_prefill=True, prefill_budget=4, **OFF),
        PolicyConfig(order="decode_ratio_desc", admission="continuous", chunk_prefill=True, prefill_budget=1, **OFF),
        PolicyConfig(order="max_tokens_desc", admission="continuous", chunk_prefill=False, prefill_budget=8, **OFF),
        PolicyConfig(order="max_tokens_desc", admission="groups", chunk_prefill=False, prefill_budget=16384,
                     max_batch_tokens=40, max_batch_size=2, **OFF),
        PolicyConfig(order="prefix_dfs", admission="continuous", chunk_prefill=True, prefill_budget=4, **OFF)]
CONFIGS = BASE + [dataclasses.replace(c, sharing="block") for c in BASE]
CONFIGS += [dataclasses.replace(c, memory=m) for c in (BASE[0], BASE[4], CONFIGS[5], CONFIGS[9]) for m in ("grow", "preempt")]
CONFIGS += [dataclasses.replace(c, sharing="token") for c in BASE[:2] + BASE[4:5]]


def cfg_id(c):
    return f"{c.order}-{c.admission}-chunk{c.chunk_prefill}-{c.prefill_budget}-share{c.sharing}-{c.memory}"


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
    if cfg.sharing != "none":
        # every shared trie node is computed once and read by its other users (at 16: sharedB and dupA hit both PREFIX
        # blocks, sharedC the first, 5 blocks); token sharing copies partial blocks on top (dupA's tail from sharedA's
        # when written before its turn; at 32, sharedC's 16 shared tokens from the PREFIX node)
        B = executor.block_size
        job = analyze(reqs, FixedTok(), block_size=B)
        block_hits = B * sum(len(n.users) - 1 for n in job.prefix.nodes)
        hits = m.counts["prefix_hit_tokens"]
        assert hits == block_hits if cfg.sharing == "block" else block_hits <= hits < block_hits + 2 * B
        assert m.counts["prefill_tokens"] == prompt_tokens - hits
        assert m.rates()["prefix_hit_pct"] == round(100 * hits / prompt_tokens, 2)
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
    cfg = PolicyConfig(engine="naive", order="max_tokens_desc", admission="groups", sharing="none",
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
    assert hw == Hardware(gpu_name=None, total_gb=None, max_batch_tokens_fit=None, kv_pool_tokens=39 * executor.block_size,
                          kv_block_size=executor.block_size)  # the capacity column follows the block size


def test_both_engines_refuse_a_request_past_the_models_positions(ref_model, executor):
    """The tiny model has 256 positions; 'long' (9 prompt tokens) with 250 new tokens needs 258 (0 .. 257). It is
    second in input order and in a group of its own, and still nothing runs: the refusal comes before any forward."""
    reqs = [Request(id="short", prompt="short", max_tokens=2), Request(id="long", prompt="long", max_tokens=250)]
    naive = PolicyConfig(engine="naive", order="input", admission="groups", sharing="none",
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
    """layers and cuda_graphs default to "auto" at the edges: on exactly where they can run. Off the GPU (the torch
    backend) neither fused layers nor graphs can, so auto resolves to HF's layers eagerly and the engine records that;
    asking for either outright is refused with the reason, and a bad value is refused."""
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False,
                  layers="auto", cuda_graphs="auto")
    assert ex.facts == {"layers": "hf", "cuda_graphs": False, "fa_version": None, "ridge_tokens": None,
                        "quant": "none", "kv_dtype": "auto", "overlap": True, "attention": "torch", "pool": "static"}
    with pytest.raises(ValueError, match="'none' or 'fp8'"):
        Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False, quant="int8")
    with pytest.raises(ValueError, match="fused layers"):  # off the GPU the fused path cannot run, so neither can fp8
        Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False, quant="fp8")
    assert Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False,
                    ridge_tokens=300).limits.ridge_tokens == 300  # given outright; else the card table's entry
    from batchinfer.executor import resolve_fa_version
    assert resolve_fa_version("auto", torch.device("cpu")) == 2  # off the GPU, auto is 2 (the torch backend runs none)
    with pytest.raises(ValueError, match="sm90"):
        resolve_fa_version(3, torch.device("cpu"))
    with pytest.raises(ValueError, match="2, 3 or 'auto'"):
        resolve_fa_version(True, torch.device("cpu"))
    with pytest.raises(ValueError, match="cuda_graphs: .*fa backend"):
        Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False, cuda_graphs=True)
    with pytest.raises(ValueError, match="true, false or 'auto'"):
        Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False, cuda_graphs="on")
    with pytest.raises(ValueError, match="layers=fused: .*fa kernel"):
        Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False, layers="fused")
    with pytest.raises(ValueError, match="layers must be one of"):
        Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False, layers="jit")


def test_the_bench_load_options_are_the_executors_keyword_parameters():
    """A new executor knob reaches --opt and the suite's arms through BatchinferBackend.load_opts, so the list must
    name every keyword parameter of Executor.__init__ except the internal ones."""
    import inspect

    from batchinfer.bench import BatchinferBackend
    params = {name for name, p in inspect.signature(Executor.__init__).parameters.items() if name != "self"}
    assert set(BatchinferBackend.load_opts) == params - set(BatchinferBackend.internal)
    assert set(BatchinferBackend.internal) <= params


def test_the_executor_publishes_its_limits_and_the_engine_reads_only_them(ref_model):
    """What the scheduler knows of the executor is one record (schema.Limits): the pool, the block size, the step
    caps, the model's position limit and the graph sizes (none off the GPU)."""
    from batchinfer.schema import MAX_PREFILL_BUDGET, MAX_SEQS, Limits
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False)
    assert ex.limits == Limits(40, 16, MAX_SEQS, MAX_PREFILL_BUDGET, 256, (), attention="torch", overlap=True)
    assert StepEngine(ex, FixedTok()).hardware.kv_pool_tokens == 39 * 16
    with pytest.raises(RuntimeError, match="already built"):
        ex.build()


def test_a_deferred_build_happens_in_prepare_once_the_job_is_analysed(ref_model):
    """The CLI loads with build=False: analysis runs first (it needs only the tokenizer), then flow.run's prepare
    builds the executor for the job, before the policy and the run; running without it is refused."""
    from batchinfer.analysis import analyze
    from batchinfer.policy import decide
    tok = FixedTok()
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False, build=False)
    engine = StepEngine(ex, tok)
    assert ex.limits is None and engine.hardware.kv_pool_tokens is None
    requests = [Request(id="a", prompt="short", max_tokens=3, ignore_eos=True, labels=None)]
    with pytest.raises(RuntimeError, match="not built"):
        engine.run(analyze(requests, tok), decide(analyze(requests, tok), PolicyConfig()), lambda r: None, Metrics())
    results, m = [], Metrics()
    flow.run(requests, tok, engine, PolicyConfig(), results.append, m)
    assert ex.limits is not None and ex.limits.num_blocks == 40 and len(results) == 1
    assert m.timing["load"] >= 0  # the build is load time, not analysis, policy or inference


def test_graph_row_sizes_follow_the_pool_and_the_job():
    """Graph sizes are derived, never a constant: bounded by the largest capture size, MAX_SEQS, the pool (a block
    per decode row beside a prefill row) and the job's decoding requests."""
    from batchinfer.executor import GRAPH_ROWS, graph_row_sizes
    assert graph_row_sizes() == GRAPH_ROWS and GRAPH_ROWS[-1] == 1024 and 496 in GRAPH_ROWS and 544 in GRAPH_ROWS
    # the pool bound is what a pool can feed, a decode row per usable block, rounded up to a capture size
    assert graph_row_sizes(25807)[-1] == 1024 and graph_row_sizes(201)[-1] == 200 and graph_row_sizes(9) == (1, 2, 4, 8)
    assert graph_row_sizes(1) == ()
    assert graph_row_sizes(None, 96)[-1] == 96 and graph_row_sizes(None, 3) == (1, 2, 4)
    assert graph_row_sizes(None, 0) == () and graph_row_sizes(None, 5000)[-1] == 1024
    assert graph_row_sizes(None, None, graph_max_rows=16) == (1, 2, 4, 8, 16)


@pytest.mark.parametrize("sharing", [False, True])
def test_preemption_rebuilds_the_same_kv_and_ids_as_the_reference(ref_model, sharing):
    """A 9-block pool (eight usable) holds every request alone but not the set: under preempt the youngest is
    preempted and prefills its prompt and outputs again; the ids still equal the no-cache greedy reference exactly,
    which proves the recomputed KV is the one the sequence would have had."""
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=9, seed=7, probe=False)
    cfg = PolicyConfig(order="input", prefill_budget=4, sharing=("block" if sharing else "none"), memory="preempt")
    # six 15-token prompts (a block each at admission) decoding 20 tokens: all need a second block one step after
    # their first token, more than the watermark leaves free, so the youngest are preempted and resume
    for i in range(6):
        PROMPTS[f"edge{i}"] = [(7 * i + 3 * j) % 60 + 2 for j in range(15)]
    reqs = [Request(id=f"edge{i}", prompt=f"edge{i}", max_tokens=20, ignore_eos=True) for i in range(6)]
    got, m = run_step(ex, reqs, cfg)
    assert m.counts["preemptions"] > 0 and m.counts["recomputed_tokens"] > 0
    for r in reqs:
        assert got[r.id].token_ids == reference(ref_model, PROMPTS[r.prompt], r.max_tokens, True), r.id


@pytest.mark.parametrize("sharing", [False, True])
def test_grow_on_a_tight_pool_gives_reserves_ids_in_fewer_or_equal_steps(ref_model, sharing):
    """Grow through the real executor on a 6-block pool: the rows' block lists grow in place as decode reaches a
    boundary, the ids still equal the reference, and the job takes no more steps than under reserve."""
    steps = {}
    for memory in ("reserve", "grow"):
        ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=6, seed=7, probe=False)
        cfg = PolicyConfig(order="prefix_dfs", prefill_budget=4, sharing=("block" if sharing else "none"), memory=memory)
        reqs = requests(ignore_eos=True)
        got, m = run_step(ex, reqs, cfg)
        for r in reqs:
            assert got[r.id].token_ids == reference(ref_model, PROMPTS[r.prompt], r.max_tokens, True), (memory, r.id)
        steps[memory] = m.counts["sampled_rows"] and len(m.steps)
    assert steps["grow"] <= steps["reserve"]


def test_token_sharing_copies_kv_through_the_executor_and_matches_the_reference(ref_model):
    """A 22-token prompt sharing 20 tokens with a 40-token prompt: under sharing=token the 4 tokens of its tail that
    the long prompt's second block holds are copied by the executor from that block, the chunk continues from
    position 20, and the ids still equal the no-cache greedy reference exactly."""
    head = [(5 * j + 3) % 60 + 2 for j in range(40)]
    PROMPTS["tok_long"], PROMPTS["tok_short"] = head, head[:20] + [61, 62]
    reqs = [Request(id="tok_long", prompt="tok_long", max_tokens=3, ignore_eos=True),
            Request(id="tok_short", prompt="tok_short", max_tokens=3, ignore_eos=True)]
    hits = {}
    for sharing in ("block", "token"):
        ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False)
        got, m = run_step(ex, reqs, PolicyConfig(order="input", prefill_budget=4, sharing=sharing))
        for r in reqs:
            assert got[r.id].token_ids == reference(ref_model, PROMPTS[r.prompt], r.max_tokens, True), (sharing, r.id)
        hits[sharing] = m.counts["prefix_hit_tokens"]
    assert hits == {"block": 16, "token": 20}


def test_radix_cache_serves_the_next_job_through_one_engine_and_matches_the_reference(ref_model):
    """Two jobs through one StepEngine with cache=radix: the second job's prompts share the first's 32-token system
    prompt; its two nodes come computed from the kept blocks (no prefill of them), and the ids still equal the
    no-cache greedy reference. cache=job drops the kept blocks before the next job."""
    system = [(7 * j + 1) % 60 + 2 for j in range(32)]
    PROMPTS["rad_a"], PROMPTS["rad_b"], PROMPTS["rad_c"] = system + [3], system + [4, 5], system + [6]
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False)
    tok = FixedTok()
    engine = StepEngine(ex, tok)

    def run(ids, cache):
        reqs = [Request(id=i, prompt=i, max_tokens=3, ignore_eos=True) for i in ids]
        out, m = [], Metrics()
        flow.run(reqs, tok, engine, PolicyConfig(order="input", prefill_budget=8, cache=cache), out.append, m)
        for r in reqs:
            assert out[[o.id for o in out].index(r.id)].token_ids == reference(ref_model, PROMPTS[r.prompt], 3, True)
        return m.counts

    first = run(["rad_a"], "radix")
    assert first["prefix_hit_tokens"] == 0 and first["retained_blocks"] == 2 and engine.kv.free == 39 - 2
    second = run(["rad_b", "rad_c"], "radix")  # both hit the kept system prompt; nothing of it is prefilled
    assert second["retained_hits"] == 2 and second["prefix_hit_tokens"] == 2 * 32
    assert second["prefill_tokens"] == 2 + 1 and second["retained_blocks"] == 2
    third = run(["rad_a"], "job")  # cache=job: the kept blocks are dropped first, the prefix is computed again
    assert third["prefix_hit_tokens"] == 0 and third["retained_blocks"] == 0 and engine.kv.free == 39


def test_an_fp8_kv_pool_halves_the_bytes_and_runs_the_torch_backend_close_to_bf16(ref_model):
    """kv_dtype=fp8 off the GPU: the torch backend writes the pool in e4m3 and converts on read. The pool's bytes per
    block halve against the model's dtype, Limits says fp8, and the greedy ids of the test prompts match the fp32
    reference on a short run on most prompts (the tiny model's logits are wide apart); bad values are refused."""
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False, kv_dtype="fp8")
    plain = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False)
    assert ex.kv.dtype == torch.float8_e4m3fn and ex.limits.kv_dtype == "fp8" and plain.limits.kv_dtype == "bf16"
    assert ex.bytes_per_block * plain.dtype.itemsize == plain.bytes_per_block and ex.facts["kv_dtype"] == "fp8"
    reqs = [Request(id=n, prompt=n, max_tokens=2, ignore_eos=True) for n in ("short", "sharedA", "sharedB")]
    got, m = run_step(ex, reqs, PolicyConfig(order="input", prefill_budget=16))
    agree = sum(got[r.id].token_ids == reference(ref_model, PROMPTS[r.prompt], 2, True) for r in reqs)
    assert agree >= 2 and m.executor["kv_dtype"] == "fp8"
    with pytest.raises(ValueError, match="'auto', 'bf16' or 'fp8'"):
        Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False, kv_dtype="int8")


@pytest.mark.parametrize("cfg", CONFIGS, ids=cfg_id)
def test_overlap_gives_the_reference_ids_through_the_engine(ref_model, cfg):
    """overlap=True on the CPU executor: the loop plans and launches a step before settling the one before it, a
    decode row's token is gathered from the executor's output buffer, and the ids still equal the no-cache greedy
    reference for every config; a discarded row never reaches a result and output_tokens excludes it."""
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False, overlap=True)
    assert ex.limits.overlap and ex.facts["overlap"] is True
    reqs = requests(False)
    got, m = run_step(ex, reqs, cfg)
    for r in reqs:
        ref = reference(ref_model, PROMPTS[r.prompt], r.max_tokens, False)
        assert got[r.id].token_ids == ref, r.id
    assert m.counts["output_tokens"] == sum(len(g.token_ids) for g in got.values())
    assert m.counts["discarded_rows"] == sum(1 for g in got.values() if g.finish_reason == "stop")
    check_request_trace(m, got)


@pytest.mark.parametrize("cfg", [c for c in CONFIGS if c.sharing != "none"][:6], ids=cfg_id)
def test_cascade_attention_gives_the_reference_ids_through_the_engine(ref_model, cfg):
    """attention=cascade off the GPU: the scheduler groups each step's rows behind their shared computed prefixes,
    the executor runs the two-call cascade over SDPA per row and merges by log-sum-exp, and the ids equal the
    no-cache reference exactly; steps with shared rows did group."""
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False,
                  attention="cascade")
    assert ex.limits.attention == "cascade" and ex.facts["attention"] == "cascade" and ex.backend == "torch"
    reqs = requests(False)
    got, m = run_step(ex, reqs, cfg)
    for r in reqs:
        assert got[r.id].token_ids == reference(ref_model, PROMPTS[r.prompt], r.max_tokens, False), r.id
    assert m.counts["prefix_hit_tokens"] > 0


def test_compiled_layers_trace_whole_and_give_the_reference_ids(ref_model):
    """layers=compiled over the aot_eager backend, which functionalises the graph as Inductor does: Dynamo traces
    HF's forward as one graph (fullgraph: a break at the paged attention would raise), the custom op is the attention
    and writes the pool through its handle (an argument declared mutated would be cloned and copied back, erasing
    the writes), and the ids equal the reference for the sharing configs; the op's fake implementation passes
    torch's op check."""
    from batchinfer.executor import paged_attention
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False,
                  layers="compiled", compile_backend="aot_eager")
    assert ex.layers_mode == "compiled" and ex.facts["layers"] == "compiled" and ex.handle > 0
    reqs = requests(False)
    for cfg in (BASE[0], CONFIGS[5]):
        got, m = run_step(ex, reqs, cfg)
        for r in reqs:
            assert got[r.id].token_ids == reference(ref_model, PROMPTS[r.prompt], r.max_tokens, False), r.id
    b = ex.batch  # the last step's inputs, as the op saw them
    q = torch.randn(1, 4, b.input_ids.shape[1], 8)
    kv = torch.randn(1, 2, b.input_ids.shape[1], 8)
    torch.library.opcheck(paged_attention, (q, kv, kv, b.slot_mapping, b.cu_seqlens_q, b.seqused_k, b.block_table, 0, 0.35,
                                            ex.handle))


def test_the_elastic_pool_maps_groups_as_rows_take_blocks_and_unmaps_them_after_their_last_step(ref_model):
    """pool=elastic off the GPU runs over the fake driver: the pool's blocks dimension is padded to whole groups, the
    pad block's group is mapped at build, every group a step's new blocks need is mapped before it launches and
    unmapped once the last step that could touch it is done, the trace records the mapped groups, and the ids are
    the reference's. The tiny model's slab is 1 KiB (fp32), so a 2 MiB piece holds 2,048 blocks: one group; with a
    4 KiB fake granularity a piece holds 4 blocks, and the engine maps and unmaps through the job."""
    from batchinfer import vmm as vmm_module
    real = vmm_module.Fake
    vmm_module.Fake = lambda: real(granularity=4096)
    try:
        ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False, pool="elastic")
    finally:
        vmm_module.Fake = real
    assert ex.facts["pool"] == "elastic" and ex.pool.blocks_per_group == 4 and ex.pool.groups == 10 and ex.kv.shape[2] == 40
    assert ex.pool.vmm.log == [("map", off, 4096) for off in ex.pool.pieces(0)] and len(ex.pool.vmm.handles) == 2 * ex.layers
    reqs = requests(False)
    got, m = run_step(ex, reqs, CONFIGS[5])
    for r in reqs:
        assert got[r.id].token_ids == reference(ref_model, PROMPTS[r.prompt], r.max_tokens, False), r.id
    cols = m.to_dict()["steps"]
    assert max(cols["mapped_groups"]) > 1 and cols["mapped_groups"][0] >= 1
    assert len(ex.pool.vmm.handles) == 2 * ex.layers  # the job's groups went back: only the pad block's group stays
    maps = sum(1 for e in ex.pool.vmm.log if e[0] == "map")
    unmaps = sum(1 for e in ex.pool.vmm.log if e[0] == "unmap")
    assert maps == unmaps + 2 * ex.layers and maps > 2 * ex.layers
    with pytest.raises(ValueError, match="'static', 'elastic' or 'contiguous'"):
        Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False, pool="paged")


def test_the_contiguous_pool_runs_the_torch_backend_over_the_home_layout_and_maps_regions(ref_model):
    """pool=contiguous off the GPU: the block size is the unit (the tiny fp32 model's token is 256 B of KV, so a 2 MiB
    piece is 8,192 tokens: one unit of 8,192 tokens), the home pool is token-major, every unit is mapped once at
    build and never unmapped, each row's region gets its units at their index with the pad unit as the guard after
    them, a region's mappings are dropped when a new admission takes it, and the ids are the reference's. An
    explicit block size that is not the unit, and cascade, are refused."""
    from batchinfer.executor import page_tokens
    assert page_tokens(36, 8, 128, 2) == 128 and page_tokens(36, 8, 128, 1) == 256 and page_tokens(2, 2, 8, 4) == 8192
    with pytest.raises(ValueError, match="block_size must be that or unset"):
        Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False, pool="contiguous", block_size=16)
    with pytest.raises(ValueError, match="cascade"):
        Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=8, probe=False, pool="contiguous",
                 attention="cascade")
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=6, seed=7, probe=False, pool="contiguous")
    assert ex.block_size == ex.page_tokens == 8192 and ex.limits.block_size == 8192 and ex.limits.pool == "contiguous"
    assert ex.pool.layout == "token" and ex.pool.region_slots == ex.max_seqs + 2
    assert ex.kv.shape == (6, 8192, 2, 2, 2, 8) and ex.pool.vmm.handles.keys() == {b * ex.pool.unit_bytes for b in range(6)}
    assert ex.pool.max_ctx == 2 * 8192  # the model's 256 positions round up to one unit, plus the guard
    reqs = requests(False)
    got, m = run_step(ex, reqs, CONFIGS[5])
    for r in reqs:
        assert got[r.id].token_ids == reference(ref_model, PROMPTS[r.prompt], r.max_tokens, False), r.id
    assert len(ex.pool.vmm.handles) == 6 and not any(e[0] == "unmap" for e in ex.pool.vmm.log)  # the home is for good
    aliases = [e for e in ex.pool.region.log if e[0] == "alias"]
    assert aliases and all(e[2] % ex.pool.unit_bytes == 0 for e in aliases)
    assert sum(1 for e in ex.pool.region.log if e[0] == "unalias") > 0  # regions reused by later admissions were dropped
    live = set(ex.pool.region.aliases)
    assert 0 in live  # the pad slot's unit


def test_the_widest_probe_step_fits_the_output_buffers(ref_model):
    """The probe samples max_seqs decode rows plus one prefill row: the sampled-id buffers hold max_seqs + 1."""
    from batchinfer.schema import MAX_PREFILL_BUDGET, MAX_SEQS
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=2 * MAX_SEQS + 40, seed=7, probe=False)
    step = ex._probe_step()
    assert sum(1 for r in step.rows if r.sample) == MAX_SEQS + 1 and ex.g_out.shape[0] == MAX_SEQS + 1
    prefill = [r for r in step.rows if not r.decode]  # the budget in rows of at most the model's positions (256: 64 rows)
    assert len(prefill) == 64 and all(len(r.token_ids) == ex.max_positions for r in prefill)
    assert sum(len(r.token_ids) for r in prefill) == MAX_PREFILL_BUDGET
    ids, _ = ex.forward(step)
    assert len(ids) == MAX_SEQS + 1


def test_the_probe_keeps_every_prefill_row_inside_its_region(ref_model):
    """pool=contiguous sizes a region slot from the model's positions; the probe's prefill rows each fit a slot of
    their own (blocks plus the guard unit), after the decode rows' slots, and the probe's regions are dropped at
    its end. The tiny model's unit (8,192 tokens) exceeds its 256 positions, so a row is one unit."""
    from batchinfer.schema import MAX_SEQS
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=200, seed=7, probe=False, pool="contiguous")
    step = ex._probe_step()
    units = ex.pool.max_ctx // ex.page_tokens
    prefill = [r for r in step.rows if not r.decode]
    assert len(prefill) == 2 and all(len(r.block_ids) + 1 <= units for r in prefill)
    assert len({r.slot for r in step.rows}) == len(step.rows) and max(r.slot for r in step.rows) < ex.pool.region_slots
    assert sum(1 for r in step.rows if r.sample) <= MAX_SEQS + 1
    token = ex.pool.probe_begin()
    ex.forward(step)
    assert len(ex.pool.region.aliases) == 1 + sum(len(r.block_ids) + 1 for r in step.rows)  # every unit and guard
    ex.pool.probe_end(token)
    assert list(ex.pool.region.aliases) == [0]  # the pad slot's unit stays


def test_overlap_with_token_sharing_never_copies_from_a_block_this_step_writes(ref_model):
    """Under overlap an earlier chunk's node is marked computed as it is planned; a later sharer in the same step must
    not copy from it (the copy runs before the forward): tok_short's ids equal the reference with overlap on."""
    head = [(5 * j + 3) % 60 + 2 for j in range(40)]
    PROMPTS["tok_long"], PROMPTS["tok_short"] = head, head[:20] + [61, 62]
    reqs = [Request(id="tok_long", prompt="tok_long", max_tokens=3, ignore_eos=True),
            Request(id="tok_short", prompt="tok_short", max_tokens=3, ignore_eos=True)]
    for overlap in (False, True):
        ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False, overlap=overlap)
        got, m = run_step(ex, reqs, PolicyConfig(order="input", prefill_budget=64, sharing="token"))
        for r in reqs:
            assert got[r.id].token_ids == reference(ref_model, PROMPTS[r.prompt], r.max_tokens, True), (overlap, r.id)


def test_the_elastic_pool_never_maps_a_group_twice_and_the_probe_maps_only_what_it_needs(ref_model):
    """Handles that count as done only once waited, as on the GPU: under overlap a group freed at one commit and
    taken again by the next planned step is still queued to go when it is needed, so it stays mapped and leaves the
    queue instead of being mapped twice; a failed job's allocator hands its groups back; the probe maps the groups
    it lacks and unmaps exactly those."""
    from batchinfer import executor as executor_module, vmm as vmm_module
    real = vmm_module.Fake
    vmm_module.Fake = lambda: real(granularity=4096)
    try:
        ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False, pool="elastic",
                      overlap=True)
    finally:
        vmm_module.Fake = real
    waited = []
    orig_wait = executor_module._Handle.wait
    executor_module._Handle.done = lambda self: self in waited
    executor_module._Handle.wait = lambda self: (waited.append(self), orig_wait(self))[1]
    try:
        reqs = requests(False)
        got, m = run_step(ex, reqs, CONFIGS[5])
        for r in reqs:
            assert got[r.id].token_ids == reference(ref_model, PROMPTS[r.prompt], r.max_tokens, False), r.id
        assert ex.pool.mapped == {0} and len(ex.pool.vmm.handles) == 2 * ex.layers and not ex.pool.unmap_queue
        ex.pool.map_groups([3, 4])  # a probe's extras, then a job that dies: the pool comes back to the pad group
        assert ex.pool.mapped == {0, 3, 4}
        ex.pool.reset()
        assert ex.pool.mapped == {0} and len(ex.pool.vmm.handles) == 2 * ex.layers
    finally:
        del executor_module._Handle.done
        executor_module._Handle.wait = orig_wait


def test_every_executor_attribute_the_gpu_only_code_reads_exists(executor):
    """The fused layers, the profiler and the GPU tests run only on a GPU; what they read off the executor must be
    there on the CPU build too, so a rename shows up here and not at the first forward on the card."""
    import re
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    files = ["batchinfer/qwen3_fused.py", "scripts/decode_step_profile.py", "scripts/ridge_calibration.py",
             "tests/test_executor_gpu.py", "tests/test_memory_gpu.py"]
    names = set()
    for f in files:
        src = (root / f).read_text()
        names |= set(re.findall(r"\b(?:self\.ex|ex|graphed|executor|fa|paged)\.([a-zA-Z_][a-zA-Z0-9_]*)\b", src))
    gpu_only = {"graph_pool", "_graph_kv_ptr", "_capture_stream", "h_out"}  # set at capture or on the card only
    missing = sorted(n for n in names - gpu_only if not hasattr(executor, n))
    assert not missing, missing


def test_the_engine_refuses_contiguous_kv_under_flashattention_3(ref_model):
    """pool=contiguous with FlashAttention-3 faults on steps with prefill on sm90: the engine refuses the
    pair up front, naming the way out; the executor alone (decode-only profiling) is not refused."""
    ex = Executor("tiny", device="cpu", model=copy.deepcopy(ref_model), num_blocks=40, seed=7, probe=False)
    ex.pool_mode, ex.fa_version = "contiguous", 3  # what an sm90 executor with pool=contiguous reports
    with pytest.raises(ValueError, match="fa_version=2"):
        StepEngine(ex, FixedTok())
