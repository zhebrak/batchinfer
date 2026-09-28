"""The executor gate on a real model (Qwen3-0.6B, bf16, CUDA): the fa path (vLLM's FA2 kernel with a
block table), the torch path and HF's own SDPA forward agree, with scattered KV blocks, packed rows and
chunked prefill; and greedy decoding through the batchinfer engine follows HF, with and without prefix sharing,
batched or alone, through EOS, and after blocks are released and reused.

bf16 makes exact equality too strict: two kernels that sum in a different order can flip an argmax whose
top-2 margin is within a couple of bf16 steps. So a pick is accepted when it is HF's argmax, or within two
bf16 ulps of HF's top logit (bf16_tie). Every generated position is checked on the engine's own history,
so a tolerated flip never ends the comparison. The exact check lives in test_step_engine_cpu.py (fp32)."""
import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from batchinfer import flow  # noqa: E402
from batchinfer.executor import Executor  # noqa: E402
from batchinfer.metrics import Metrics  # noqa: E402
from batchinfer.schema import PolicyConfig, Request, Row, Step  # noqa: E402
from batchinfer.step_engine import StepEngine  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
TEXT = ("Batch inference is incredibly important for many offline processing pipelines. Many external APIs offer "
        "batch inference at a fixed discount, and it is the provider's job to worry about efficiency. ") * 40


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


@pytest.fixture(scope="module")
def hf():
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()


@pytest.fixture(scope="module")
def executors():
    fa = Executor(MODEL, num_blocks=512, seed=11, backend="fa", probe=False)
    th = Executor(MODEL, model=fa.model, num_blocks=512, seed=11, backend="torch", probe=False)  # same weights
    return fa, th


def prompts(tok):
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    return [ids[:17], ids[5:105], ids[50:350]]


def hf_last_logits(hf, ids):
    with torch.inference_mode():
        return hf(input_ids=torch.tensor([ids], device="cuda")).logits[0, -1].float()


def bf16_tie(top):
    """Two bf16 ulps at the magnitude of the top logit (0.25 for logits in [16, 32)): how close two logits can be
    before a kernel that sums in another order may swap them."""
    return 2 * 2 ** (math.floor(math.log2(abs(top))) - 7) if top else 0.0


def agrees(got, ref):
    """got's logits against HF's for the same position: close as vectors, and the same pick unless HF's own top
    two are within a tie, in which case either of them."""
    got, ref = got.float(), ref.float()
    if torch.nn.functional.cosine_similarity(got, ref, dim=0) <= 0.999:
        return False
    top = ref.topk(2)
    pick = int(got.argmax())
    if pick == int(top.indices[0]):
        return True
    return pick == int(top.indices[1]) and float(top.values[0] - top.values[1]) <= bf16_tie(float(top.values[0]))


def blocks_for(ex, n, rng_offset):
    """Scattered physical blocks for a sequence of n positions, like a mid-job allocation."""
    need = -(-n // 16)
    return [1 + (rng_offset + 37 * k) % (ex.num_blocks - 1) for k in range(need)]


def test_packed_prefill_matches_hf(tok, hf, executors):
    ps = prompts(tok)
    for ex in executors:
        rows, off = [], 0
        for i, p in enumerate(ps):
            rows.append(Row(seq=i, start=0, token_ids=p, block_ids=blocks_for(ex, len(p), off), sample=True,
                            decode=False))
            off += 100  # disjoint block sets per row
        assert len({b for r in rows for b in r.block_ids}) == sum(len(r.block_ids) for r in rows)
        got = ex.logits(Step(rows))
        for i, p in enumerate(ps):
            assert agrees(got[i], hf_last_logits(hf, p)), (ex.backend, len(p))


def test_chunked_prefill_matches_hf(tok, hf, executors):
    p = prompts(tok)[2]
    for ex in executors:
        blocks = blocks_for(ex, len(p), 7)
        last = None
        for start, end in ((0, 128), (128, 256), (256, len(p))):
            step = Step([Row(seq=0, start=start, token_ids=p[start:end], block_ids=blocks, sample=end == len(p),
                             decode=False)])
            out = ex.logits(step)
            last = out[0] if end == len(p) else last
        assert agrees(last, hf_last_logits(hf, p)), ex.backend


_checked = {}


def follows_hf(hf, prompt_ids, got):
    """Every generated id against HF's logits on the same history (the prompt plus got's own earlier ids): each
    must be HF's argmax or within bf16_tie of its top logit. Returns how many were HF's exact argmax."""
    key = (tuple(prompt_ids), tuple(got))
    if key not in _checked:
        exact = 0
        for k, g in enumerate(got):
            logits = hf_last_logits(hf, list(prompt_ids) + got[:k])
            top = float(logits.max())
            assert float(logits[g]) >= top - bf16_tie(top), (
                f"position {k}: picked {g} at logit {float(logits[g]):.3f}, HF's top is {int(logits.argmax())} at "
                f"{top:.3f}, beyond the bf16 tie {bf16_tie(top)}")
            exact += int(logits.argmax()) == g
        _checked[key] = exact
    return _checked[key]


def run_engine(ex, tok, reqs, budget=64, **cfg):
    out, m = [], Metrics()
    cfg = PolicyConfig(**{"order": "decode_ratio_desc", "prefill_budget": budget, **cfg})
    flow.run(reqs, tok, StepEngine(ex, tok), cfg, out.append, m)
    assert sorted(r.id for r in out) == sorted(r.id for r in reqs)
    return {r.id: r for r in out}, m


def assert_ran_to_length(results, reqs):
    for r in reqs:
        res = results[r.id]
        assert len(res.token_ids) == res.output_tokens == r.max_tokens and res.finish_reason == "length", r.id


def ids_of(tok, r):
    return tok(r.prompt, add_special_tokens=False)["input_ids"]


def test_engine_with_prefix_sharing_follows_hf(tok, hf, executors):
    """Four prompts with a 64-token shared prefix (four full blocks): the first computes it, the others read
    it. Both runs follow HF, and exactly 3 x 64 prompt tokens are never computed."""
    fa, _ = executors
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    ps = [ids[:64] + ids[100 + 30 * i:130 + 30 * i] for i in range(4)]
    reqs = [Request(id=str(i), prompt=tok.decode(p), max_tokens=16, ignore_eos=True) for i, p in enumerate(ps)]
    shared, m = run_engine(fa, tok, reqs, budget=32, order="prefix_dfs", prefix_sharing=True)
    plain, _ = run_engine(fa, tok, reqs, budget=32, order="prefix_dfs", prefix_sharing=False)
    assert m.counts["prefix_hit_tokens"] == 3 * 64 and m.rates()["prefix_hit_pct"] > 0
    assert_ran_to_length(shared, reqs)
    assert_ran_to_length(plain, reqs)
    for r in reqs:
        follows_hf(hf, ids_of(tok, r), shared[r.id].token_ids)
        follows_hf(hf, ids_of(tok, r), plain[r.id].token_ids)


def test_engine_follows_hf_batched_and_alone(tok, hf, executors):
    """Batch invariance up to bf16: together and alone each follow HF at every position, so where they differ
    both picks are near-ties on the same history."""
    fa, _ = executors
    ps = prompts(tok) + [tok("Name three prime numbers.", add_special_tokens=False)["input_ids"]]
    reqs = [Request(id=str(i), prompt=tok.decode(p), max_tokens=32, ignore_eos=True) for i, p in enumerate(ps)]
    together, _ = run_engine(fa, tok, reqs)
    assert_ran_to_length(together, reqs)
    for r in reqs:
        alone, _ = run_engine(fa, tok, [r])
        assert_ran_to_length(alone, [r])
        a, b = together[r.id].token_ids, alone[r.id].token_ids
        exact = follows_hf(hf, ids_of(tok, r), a), follows_hf(hf, ids_of(tok, r), b)
        first = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
        print(f"request {r.id}: HF's exact argmax at {exact[0]} / {exact[1]} of 32 (together / alone); "
              f"first difference {first}")


def test_engine_stops_at_eos_where_hf_does(tok, hf, executors):
    fa, _ = executors
    prompt = tok.apply_chat_template([{"role": "user", "content": "Name three prime numbers, comma separated."}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    r = Request(id="eos", prompt=prompt, max_tokens=64)
    got, m = run_engine(fa, tok, [r])
    res = got["eos"]
    assert res.finish_reason == "stop" and res.token_ids[-1] in fa.stop_ids and len(res.token_ids) < 64
    assert m.counts["finish_stop"] == 1
    follows_hf(hf, ids_of(tok, r), res.token_ids)  # the EOS id included: HF picks it at the same position


def test_released_blocks_are_reused(tok, hf, executors):
    """A 64-block pool holds two of the four requests (25 blocks each) at a time, so the later two run in blocks
    the earlier two released, which still hold their stale KV."""
    fa, _ = executors
    small = Executor(MODEL, model=fa.model, num_blocks=64, seed=5, backend="fa", probe=False)
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    reqs = [Request(id=str(i), prompt=tok.decode(ids[37 * i:37 * i + 380]), max_tokens=16, ignore_eos=True)
            for i in range(4)]
    got, m = run_engine(small, tok, reqs, prefix_sharing=False)  # private blocks only, so the arithmetic above holds
    assert m.counts["head_blocked_steps"] > 0
    assert_ran_to_length(got, reqs)
    for r in reqs:
        follows_hf(hf, ids_of(tok, r), got[r.id].token_ids)


def test_a_step_where_no_row_samples(tok, executors):
    """A long prompt's middle chunk with nothing decoding samples no row; the step still runs and is timed. It
    crashed on the H100 (a classification-heavy job with 8k-token prompts): the empty result never synchronised."""
    p = prompts(tok)[2]
    for ex in executors:
        ids, ms = ex.forward(Step([Row(seq=0, start=0, token_ids=p[:128], block_ids=blocks_for(ex, len(p), 3),
                                       sample=False, decode=False)]))
        assert ids == [] and ms > 0, ex.backend


def test_probe_runs_the_widest_step(executors):
    """255 sampled decode rows and a 4,096-token prefill chunk on the 512-block pool."""
    fa, _ = executors
    assert isinstance(fa.probe(), float)


# fused layers: our Qwen3 forward over the same weights, with vLLM's fused kernels ------------------------------------

@pytest.fixture(scope="module")
def fused(executors):
    fa, _ = executors
    return Executor(MODEL, model=fa.model, num_blocks=512, seed=11, backend="fa", probe=False, fused_layers=True)


def test_fused_projections_are_views_of_one_tensor(fused):
    """q/k/v and gate/up are concatenated once and HF's Linear weights point into the result: nothing is held twice,
    and HF's forward (the other executors share this model) still reads the same values."""
    attn, mlp = fused.model.model.layers[0].self_attn, fused.model.model.layers[0].mlp
    assert attn.q_proj.weight.data_ptr() == attn.qkv_weight.data_ptr()
    assert attn.v_proj.weight.data_ptr() == attn.qkv_weight[-attn.v_proj.weight.shape[0]:].data_ptr()
    assert mlp.up_proj.weight.data_ptr() == mlp.gate_up_weight[mlp.gate_proj.weight.shape[0]:].data_ptr()


def matches_hf_on_prefill_and_decode(tok, hf, ex):
    """A packed prefill of three prompts, then a decode-only step on the KV it wrote, both against HF."""
    ps, rows, blocks = prompts(tok), [], []
    for i, p in enumerate(ps):
        blocks.append(blocks_for(ex, len(p) + 1, 100 * i))
        rows.append(Row(seq=i, start=0, token_ids=p, block_ids=blocks[i], sample=True, decode=False))
    got = ex.logits(Step(rows))
    nxt = [int(g.argmax()) for g in got]
    for i, p in enumerate(ps):
        assert agrees(got[i], hf_last_logits(hf, p)), ("prefill", len(p))
    dec = ex.logits(Step([Row(seq=i, start=len(p), token_ids=[nxt[i]], block_ids=blocks[i], sample=True,
                              decode=True) for i, p in enumerate(ps)]))
    for i, p in enumerate(ps):
        assert agrees(dec[i], hf_last_logits(hf, p + [nxt[i]])), ("decode", len(p))


def test_fused_layers_match_hf_on_prefill_and_decode(tok, hf, fused):
    matches_hf_on_prefill_and_decode(tok, hf, fused)


@pytest.mark.parametrize("fused_layers", [False, True])
def test_flash_attention_3_matches_hf(tok, hf, executors, fused_layers):
    """fa_version=3, eagerly, over HF layers and over fused layers (sm90 only)."""
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FlashAttention-3 needs an sm90 GPU")
    fa, _ = executors
    ex = Executor(MODEL, model=fa.model, num_blocks=512, seed=11, backend="fa", probe=False, fa_version=3,
                  fused_layers=fused_layers)
    matches_hf_on_prefill_and_decode(tok, hf, ex)


def test_cuda_graphs_refuse_flash_attention_3(executors):
    """The graph check itself, so it reads the same on any card (off sm90, fa_version=3 is refused earlier still)."""
    from batchinfer.executor import check_graphable
    fa, _ = executors
    with pytest.raises(ValueError, match="FlashAttention-2 only"):
        check_graphable(fa.model, "fa", 3)


def test_fused_layers_match_hf_on_chunked_prefill(tok, hf, fused):
    p = prompts(tok)[2]
    blocks = blocks_for(fused, len(p), 7)
    for start, end in ((0, 128), (128, 256), (256, len(p))):
        out = fused.logits(Step([Row(seq=0, start=start, token_ids=p[start:end], block_ids=blocks,
                                     sample=end == len(p), decode=False)]))
    assert agrees(out[0], hf_last_logits(hf, p))


def test_fused_qk_norm_rope_matches_the_split_ops(tok, hf, fused):
    """vLLM's one-kernel q/k norm + RoPE against rms_norm twice and rotary_embedding, on the same prefill."""
    p = prompts(tok)[1]
    step = Step([Row(seq=0, start=0, token_ids=p, block_ids=blocks_for(fused, len(p), 3), sample=True, decode=False)])
    one = fused.logits(step)[0]
    fused.fused.qk_norm_rope = "split"
    try:
        split = fused.logits(step)[0]
    finally:
        fused.fused.qk_norm_rope = "fused"
    assert agrees(one, split) and agrees(split, hf_last_logits(hf, p))


def test_engine_with_fused_layers_follows_hf_and_repeats_itself(tok, hf, fused):
    """Greedy decoding through the engine follows HF at every position, with prefix sharing, and two runs give the
    same ids (the README's run-to-run claim holds for fused layers too)."""
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    ps = [ids[:64] + ids[100 + 30 * i:130 + 30 * i] for i in range(3)] + prompts(tok)[:1]
    reqs = [Request(id=str(i), prompt=tok.decode(p), max_tokens=24, ignore_eos=True) for i, p in enumerate(ps)]
    first, m = run_engine(fused, tok, reqs, budget=32, order="prefix_dfs", prefix_sharing=True)
    again, _ = run_engine(fused, tok, reqs, budget=32, order="prefix_dfs", prefix_sharing=True)
    assert m.counts["prefix_hit_tokens"] > 0
    assert_ran_to_length(first, reqs)
    for r in reqs:
        assert first[r.id].token_ids == again[r.id].token_ids, r.id
        follows_hf(hf, ids_of(tok, r), first[r.id].token_ids)


# CUDA graphs: decode-only steps replayed from graphs captured at load, over HF layers and over fused layers ----------

@pytest.fixture(scope="module", params=["hf_layers", "fused_layers"])
def graphed(request, executors):
    fa, _ = executors
    return Executor(MODEL, model=fa.model, num_blocks=512, seed=11, backend="fa", probe=False,
                    fused_layers=request.param == "fused_layers", cuda_graphs=True, graph_max_rows=16)


def prefilled_decode_step(tok, ex):
    """Three prompts prefilled eagerly (their KV written), then the decode-only step of their next tokens."""
    ps, rows, blocks = prompts(tok), [], []
    for i, p in enumerate(ps):
        blocks.append(blocks_for(ex, len(p) + 1, 100 * i))
        rows.append(Row(seq=i, start=0, token_ids=p, block_ids=blocks[i], sample=True, decode=False))
    first, _ = ex.forward(Step(rows))
    assert ex.graph_rows == 0  # a prefill step runs eagerly
    return Step([Row(seq=i, start=len(p), token_ids=[first[i]], block_ids=blocks[i], sample=True, decode=True)
                 for i, p in enumerate(ps)])


def test_graph_replay_agrees_with_the_eager_step(tok, graphed):
    """Three decode rows replay the 4-row graph; each id is the eager step's pick or within a bf16 tie of its top (the
    graph's GEMMs run at 4 rows, the eager ones at 3, and cuBLAS may pick another kernel)."""
    step = prefilled_decode_step(tok, graphed)
    ids, ms = graphed.forward(step)
    assert graphed.graph_rows == 4 and ms > 0
    eager = graphed.logits(step).float()
    for i, g in enumerate(ids):
        top = float(eager[i].max())
        assert float(eager[i][g]) >= top - bf16_tie(top), (i, g, int(eager[i].argmax()))


def test_pad_rows_do_not_touch_real_rows(tok, graphed):
    """The same three real rows at the same graph size, once with the pad row zeroed and once with it holding a
    real-looking token and position over a real row's blocks: the real rows' ids are identical."""
    step = prefilled_decode_step(tok, graphed)
    ids, _ = graphed.forward(step)
    graph, out = graphed.graphs[4]
    graphed._stage(step, 4)
    big = graphed.graph_sizes[-1]
    graphed.g_tokens[3], graphed.g_tokens[big + 3] = 1234, 40  # slot stays 0: the pad block
    graphed.g_used[3] = 41
    graphed.g_table[3, :3] = graphed.g_table[0, :3]
    graph.replay()
    assert out[:3].tolist() == ids


def test_mixed_and_oversized_steps_run_eagerly(graphed):
    graphed.forward(Step([Row(seq=i, start=0, token_ids=[5], block_ids=[1 + i], sample=True, decode=True)
                          for i in range(17)]))  # more rows than the largest graph (16)
    assert graphed.graph_rows == 0
    graphed.forward(Step([Row(seq=0, start=0, token_ids=[5], block_ids=[1], sample=True, decode=True),
                          Row(seq=1, start=0, token_ids=[5, 6], block_ids=[2], sample=True, decode=False)]))
    assert graphed.graph_rows == 0


def test_graphs_survive_the_probe_and_an_emptied_cache(tok, graphed):
    """bench's reset() empties the allocator's cache and re-runs the probe between passes; a graph's private pool and
    the KV pool it reads must survive both."""
    ids, _ = graphed.forward(prefilled_decode_step(tok, graphed))
    graphed.probe()  # writes throwaway KV across the pool, as it does at load
    torch.cuda.empty_cache()
    again, _ = graphed.forward(prefilled_decode_step(tok, graphed))  # the prompts' KV written again
    assert graphed.graph_rows == 4 and again == ids


def test_engine_with_graphs_follows_hf_and_repeats_itself(tok, hf, graphed):
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    ps = [ids[:64] + ids[100 + 30 * i:130 + 30 * i] for i in range(3)] + prompts(tok)[:1]
    reqs = [Request(id=str(i), prompt=tok.decode(p), max_tokens=24, ignore_eos=True) for i, p in enumerate(ps)]
    first, m = run_engine(graphed, tok, reqs, budget=32, order="prefix_dfs", prefix_sharing=True)
    again, _ = run_engine(graphed, tok, reqs, budget=32, order="prefix_dfs", prefix_sharing=True)
    assert m.step_rates()["graph_step_pct"] > 0
    assert_ran_to_length(first, reqs)
    for r in reqs:
        assert first[r.id].token_ids == again[r.id].token_ids, r.id
        follows_hf(hf, ids_of(tok, r), first[r.id].token_ids)
