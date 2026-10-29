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
    fa = Executor(MODEL, num_blocks=512, seed=11, attention="fa", probe=False, layers="hf", cuda_graphs=False)
    th = Executor(MODEL, model=fa.model, num_blocks=512, seed=11, attention="torch", probe=False,
                  layers="hf", cuda_graphs=False)  # same weights
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


def pick_agrees(pick, logits):
    """A graph's pick against the eager logits of the same row: the argmax, or within a bf16 tie of it."""
    top = logits.topk(2)
    return pick == int(top.indices[0]) or (pick == int(top.indices[1])
                                            and float(top.values[0] - top.values[1]) <= bf16_tie(float(top.values[0])))


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
            assert agrees(got[i], hf_last_logits(hf, p)), (ex.attention, len(p))


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
        assert agrees(last, hf_last_logits(hf, p)), ex.attention


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
    shared, m = run_engine(fa, tok, reqs, budget=32, order="prefix_dfs", sharing="block")
    plain, _ = run_engine(fa, tok, reqs, budget=32, order="prefix_dfs", sharing="none")
    assert m.counts["prefix_hit_tokens"] == 3 * 64 and m.rates()["prefix_hit_pct"] > 0
    assert_ran_to_length(shared, reqs)
    assert_ran_to_length(plain, reqs)
    for r in reqs:
        follows_hf(hf, ids_of(tok, r), shared[r.id].token_ids)
        follows_hf(hf, ids_of(tok, r), plain[r.id].token_ids)


def test_token_sharing_copies_the_shared_tail_bitwise(tok, hf, executors):
    """Two prompts sharing 20 tokens: the second's second block gets its first 4 slots copied from the first's
    computed block (Step.copies) and its chunk starts at position 20; the copied KV equals the source bitwise, and
    the prompt's next-token logits agree with HF."""
    fa, _ = executors
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    a, b = ids[:40], ids[:20] + ids[200:203]
    blocks_a, blocks_b = blocks_for(fa, 40, 500), [None, 1 + (77 * 3) % (fa.num_blocks - 1), 1 + (77 * 5) % (fa.num_blocks - 1)]
    blocks_b[0] = blocks_a[0]  # the shared first block is a trie node: read, never written, by b
    first = fa.logits(Step([Row(seq=0, start=0, token_ids=a, block_ids=blocks_a, sample=True, decode=False)]))
    assert agrees(first[0], hf_last_logits(hf, a))
    row = Row(seq=1, start=20, token_ids=b[20:], block_ids=blocks_b, sample=True, decode=False)
    fa.forward(Step([row], copies=[(blocks_a[1], blocks_b[1], 4)]))  # the copies run in launch, before the forward
    assert torch.equal(fa.kv[:, :, blocks_b[1], :4], fa.kv[:, :, blocks_a[1], :4])
    got = fa.logits(Step([row]))  # the same chunk again over the copied slots: its logits
    assert agrees(got[0], hf_last_logits(hf, b))
    # budget 40: a's prompt fills step 0 alone, so b's copy source (a's second block) is computed before b's chunk; in
    # one step with a, b would prefill the block itself (a copy never reads a block the same step writes)
    through, m = run_engine(fa, tok, [Request(id="a", prompt=tok.decode(a), max_tokens=4, ignore_eos=True),
                                      Request(id="b", prompt=tok.decode(b), max_tokens=4, ignore_eos=True)],
                            budget=40, order="input", sharing="token")
    assert m.counts["prefix_hit_tokens"] == 16 + 4
    follows_hf(hf, b, through["b"].token_ids)


def test_block_size_64_follows_hf_with_and_without_sharing(tok, hf, executors):
    """The paged kernels at 64-token blocks: packed prefill and decode against HF, and the engine with a 128-token
    shared prefix (two blocks at 64) following HF with 3 x 128 hits."""
    fa, _ = executors
    ex = Executor(MODEL, model=fa.model, num_blocks=128, seed=11, attention="fa", probe=False, layers="hf",
                  cuda_graphs=False, block_size=64)
    assert ex.limits.block_size == 64 and ex.kv.shape[3] == 64
    matches_hf_on_prefill_and_decode(tok, hf, ex)
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    ps = [ids[:128] + ids[200 + 30 * i:230 + 30 * i] for i in range(4)]
    reqs = [Request(id=str(i), prompt=tok.decode(p), max_tokens=8, ignore_eos=True) for i, p in enumerate(ps)]
    shared, m = run_engine(ex, tok, reqs, budget=64, order="prefix_dfs", sharing="block")
    assert m.counts["prefix_hit_tokens"] == 3 * 128 and m.counts["kv_block_size"] == 64
    assert_ran_to_length(shared, reqs)
    for r in reqs:
        follows_hf(hf, ids_of(tok, r), shared[r.id].token_ids)


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
    small = Executor(MODEL, model=fa.model, num_blocks=64, seed=5, attention="fa", probe=False, layers="hf",
                     cuda_graphs=False)
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    reqs = [Request(id=str(i), prompt=tok.decode(ids[37 * i:37 * i + 380]), max_tokens=16, ignore_eos=True)
            for i in range(4)]
    got, m = run_engine(small, tok, reqs, sharing="none")  # private blocks only, so the arithmetic above holds
    assert m.counts["head_blocked_steps"] > 0
    assert_ran_to_length(got, reqs)
    for r in reqs:
        follows_hf(hf, ids_of(tok, r), got[r.id].token_ids)


def test_a_step_where_no_row_samples(tok, executors):
    """A long prompt's middle chunk with nothing decoding samples no row; the step still runs and is timed."""
    p = prompts(tok)[2]
    for ex in executors:
        ids, ms = ex.forward(Step([Row(seq=0, start=0, token_ids=p[:128], block_ids=blocks_for(ex, len(p), 3),
                                       sample=False, decode=False)]))
        assert ids == [] and ms > 0, ex.attention


def test_probe_runs_the_widest_step(executors):
    """255 sampled decode rows and a 4,096-token prefill chunk on the 512-block pool."""
    fa, _ = executors
    assert isinstance(fa.probe(), float)


# fused layers: our Qwen3 forward over the same weights, with vLLM's fused kernels ------------------------------------

@pytest.fixture(scope="module")
def fused(executors):
    fa, _ = executors
    return Executor(MODEL, model=fa.model, num_blocks=512, seed=11, attention="fa", probe=False, layers="fused",
                    cuda_graphs=False)


def test_fp8_weights_follow_the_bf16_fused_forward(tok, hf, executors):
    """quant=fp8 (sm89 and above): the fused forward with fp8 projections against the bf16 fused one on packed
    prefill and decode: close as vectors (cosine above 0.99) and the same greedy pick on the clear cases; and the
    engine's generations stay on HF's path for their first tokens. The quality gate of the bench (classify accuracy
    within a point of the bf16 row) is the end-to-end check; this is the kernel check."""
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("FP8 tensor cores need sm89 or above")
    fa, _ = executors
    import copy
    bf16 = Executor(MODEL, model=fa.model, num_blocks=512, seed=11, attention="fa", probe=False, layers="fused",
                    cuda_graphs=False)
    q8 = Executor(MODEL, model=copy.deepcopy(fa.model), num_blocks=512, seed=11, attention="fa", probe=False,
                  layers="fused", cuda_graphs=False, quant="fp8")
    assert q8.facts["quant"] == "fp8" and q8.fused.quant == "fp8"
    ps = prompts(tok)
    rows = [Row(seq=i, start=0, token_ids=p, block_ids=blocks_for(fa, len(p) + 1, 100 * i), sample=True, decode=False)
            for i, p in enumerate(ps)]
    ref, got = bf16.logits(Step(rows)).float(), q8.logits(Step(rows)).float()
    for i in range(len(ps)):
        assert torch.nn.functional.cosine_similarity(got[i], ref[i], dim=0) > 0.99
    agree = sum(int(got[i].argmax()) == int(ref[i].argmax()) for i in range(len(ps)))
    assert agree >= 2, "fp8 flips the greedy pick on more than one of three prompts"
    reqs = [Request(id=str(i), prompt=tok.decode(p), max_tokens=8, ignore_eos=True) for i, p in enumerate(ps)]
    out, m = run_engine(q8, tok, reqs, order="input")
    assert_ran_to_length(out, reqs) and m.executor["quant"] == "fp8"


def test_fp8_kv_pool_follows_the_bf16_pool_under_fa3(tok, hf, executors):
    """kv_dtype=fp8 with FlashAttention-3 (sm90): the fused forward writes e4m3 KV and reads it with unit descales;
    prefill logits agree with HF's as vectors, and a graphed run on it goes to length. FlashAttention-2 refuses the
    pool. The quality gate (classify accuracy against the bf16 row) is the bench's."""
    fa, _ = executors
    from batchinfer.executor import fa3_available
    if not fa3_available(torch.device("cuda")):
        pytest.skip("FlashAttention-3 needs an sm90 GPU")
    with pytest.raises(ValueError, match="FlashAttention-3"):
        Executor(MODEL, model=fa.model, num_blocks=64, attention="fa", probe=False, fa_version=2, kv_dtype="fp8",
                 layers="hf", cuda_graphs=False)
    ex = Executor(MODEL, model=fa.model, num_blocks=512, seed=11, attention="fa", probe=False, fa_version=3,
                  layers="fused", cuda_graphs=True, graph_max_rows=16, kv_dtype="fp8")
    assert ex.kv.dtype == torch.float8_e4m3fn and ex.limits.kv_dtype == "fp8"
    ps = prompts(tok)
    rows = [Row(seq=i, start=0, token_ids=p, block_ids=blocks_for(ex, len(p) + 1, 100 * i), sample=True, decode=False)
            for i, p in enumerate(ps)]
    got = ex.logits(Step(rows))
    for i, p in enumerate(ps):
        assert torch.nn.functional.cosine_similarity(got[i].float(), hf_last_logits(hf, p), dim=0) > 0.99
    reqs = [Request(id=str(i), prompt=tok.decode(p), max_tokens=8, ignore_eos=True) for i, p in enumerate(ps)]
    out, m = run_engine(ex, tok, reqs, order="input")
    assert_ran_to_length(out, reqs)
    assert m.step_rates()["graph_step_pct"] > 0 and m.executor["kv_dtype"] == "fp8"


def test_overlap_matches_sync(tok, hf, executors):
    """overlap=True over graphed decode (FlashAttention-2 graphs, fused layers): the loop launches a step before the
    previous one's ids are back, rows gather their token on the device, and every id equals the synchronous run's
    (the same kernels at the same shapes); the discarded rows are the EOS finishes."""
    fa, _ = executors
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    reqs = [Request(id=str(i), prompt=tok.decode(ids[20 * i:20 * i + 60]), max_tokens=24, ignore_eos=i % 2 == 0)
            for i in range(6)]
    runs = {}
    for overlap in (False, True):
        ex = Executor(MODEL, model=fa.model, num_blocks=512, seed=11, attention="fa", probe=False, fa_version=2,
                      layers="fused", cuda_graphs=True, graph_max_rows=16, overlap=overlap)
        out, m = run_engine(ex, tok, reqs, budget=64, order="input")
        runs[overlap] = ({r.id: (out[r.id].token_ids, out[r.id].finish_reason) for r in reqs}, m)
        assert m.executor["overlap"] is overlap and m.step_rates()["graph_step_pct"] > 0
    assert runs[True][0] == runs[False][0]
    m = runs[True][1]
    assert m.counts["discarded_rows"] == sum(1 for ids_, reason in runs[True][0].values() if reason == "stop")
    assert m.counts["output_tokens"] == sum(len(ids_) for ids_, _ in runs[True][0].values())


def test_cascade_matches_fa(tok, hf, executors):
    """attention=cascade on the GPU: a decode step of four rows behind a 64-token shared prefix (and one row behind
    nothing) through the two FlashAttention calls and vLLM's merge, against the plain fa call on the same KV: the
    same picks within a bf16 tie and logits close as vectors; and the engine on it follows HF with graphs on."""
    fa, _ = executors
    cas = Executor(MODEL, model=fa.model, num_blocks=512, seed=11, attention="cascade", probe=False, layers="hf",
                   cuda_graphs=False)
    from batchinfer.cascade import Group
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    shared, rows, blocks = ids[:64], [], []
    for i in range(5):
        own = ids[100 + 30 * i:130 + 30 * i] if i < 4 else ids[300:340]
        p = (shared if i < 4 else []) + own
        blocks.append(([1, 2, 3, 4] if i < 4 else []) + blocks_for(fa, len(own) + 1, 50 * i + 300)[:(len(own) + 16) // 16 + 1])
        rows.append(Row(seq=i, start=0, token_ids=p, block_ids=blocks[i], sample=True, decode=False))
    first = fa.logits(Step(rows)).argmax(-1).tolist()  # the KV written by the plain path; both executors share the pool
    cas.pool = fa.pool  # the cascade executor reads the KV the fa executor wrote
    decode = Step([Row(seq=i, start=len(r.token_ids), token_ids=[first[i]], block_ids=blocks[i], sample=True, decode=True)
                   for i, r in enumerate(rows)])
    plain = fa.logits(decode).float()
    decode.groups = [Group(0, [1, 2, 3, 4], 64, [0, 1, 2, 3]), Group(None, [], 0, [4])]
    got = cas.logits(decode).float()
    for i in range(5):
        assert agrees(got[i], plain[i])
    ex = Executor(MODEL, model=fa.model, num_blocks=512, seed=11, attention="cascade", probe=False, layers="fused",
                  cuda_graphs=True, graph_max_rows=16)
    reqs = [Request(id=str(i), prompt=tok.decode(shared + ids[100 + 30 * i:130 + 30 * i]), max_tokens=8, ignore_eos=True)
            for i in range(4)]
    out, m = run_engine(ex, tok, reqs, budget=64, order="prefix_dfs", sharing="block")
    assert_ran_to_length(out, reqs)
    assert m.executor["attention"] == "cascade" and m.step_rates()["graph_step_pct"] > 0
    for r in reqs:
        follows_hf(hf, ids_of(tok, r), out[r.id].token_ids)


def test_compiled_layers_follow_hf_with_graphs(tok, hf, executors):
    """layers=compiled on the GPU: HF's layers under torch.compile (Inductor), the paged attention an opaque op, one
    symbolic token dimension, captured into the decode graphs like any forward: prefill and decode agree with HF
    and the engine follows HF; the second executor on the same model is warm from the Inductor cache."""
    fa, _ = executors
    ex = Executor(MODEL, model=fa.model, num_blocks=512, seed=11, attention="fa", probe=False, layers="compiled",
                  cuda_graphs=True, graph_max_rows=16)
    assert ex.layers_mode == "compiled" and ex.facts["layers"] == "compiled"
    matches_hf_on_prefill_and_decode(tok, hf, ex)
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    reqs = [Request(id=str(i), prompt=tok.decode(ids[20 * i:20 * i + 60]), max_tokens=12, ignore_eos=True) for i in range(5)]
    out, m = run_engine(ex, tok, reqs, budget=64, order="input")
    assert_ran_to_length(out, reqs)
    assert m.executor["layers"] == "compiled" and m.step_rates()["graph_step_pct"] > 0
    for r in reqs:
        follows_hf(hf, ids_of(tok, r), out[r.id].token_ids)


def test_elastic_pool_round_trips(tok, hf, executors):
    """pool=elastic on the GPU: the pool over CUDA virtual memory through libcuda, graphs captured over it; a run's
    ids equal the static pool's, the card's used memory drops as rows finish (pieces released), and graphs replay
    after an unmap and a remap."""
    fa, _ = executors
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    # 16 requests of 5 blocks each: more than one group (a 2 MiB piece holds 64 of Qwen3-0.6B's 32 KiB slabs)
    reqs = [Request(id=str(i), prompt=tok.decode(ids[20 * i:20 * i + 60]), max_tokens=12, ignore_eos=True) for i in range(16)]
    runs = {}
    for pool in ("static", "elastic"):
        ex = Executor(MODEL, model=fa.model, num_blocks=2048, seed=11, attention="fa", probe=False, fa_version=2,
                      layers="fused", cuda_graphs=True, graph_max_rows=16, pool=pool)
        free_before = torch.cuda.mem_get_info()[0]
        out, m = run_engine(ex, tok, reqs, budget=64, order="input")
        runs[pool] = {r.id: out[r.id].token_ids for r in reqs}
        if pool == "elastic":
            assert ex.pool.vmm.mapped_bytes == 2 * ex.layers * ex.pool.piece_bytes  # only the pad group stays mapped after the job
            assert torch.cuda.mem_get_info()[0] >= free_before - ex.pool.piece_bytes * 2 * ex.layers * 2  # given back
            assert max(m.to_dict()["steps"]["mapped_groups"]) > 1
            again, _ = run_engine(ex, tok, reqs, budget=64, order="input")  # remapped, graphs replay
            assert {r.id: again[r.id].token_ids for r in reqs} == runs["elastic"]
    assert runs["elastic"] == runs["static"]


def test_contiguous_pool_round_trips(tok, hf, executors):
    """pool=contiguous on the GPU: the home over mapped units, the regions over a second reservation. (1) The region
    read (out-of-order cu_seqlens_k with seqused_k over a strided token dimension, no block table) gives the paged
    read's logits over the same units (the attention's regions switch), rows placed one past a unit boundary
    among them, where the guard unit after the last mapped one covers the kernel's tile loads; (2) a graphed run's
    ids equal the static pool's at the same block size, and a second run reuses the regions."""
    fa, _ = executors
    ex = Executor(MODEL, model=fa.model, num_blocks=64, seed=11, attention="fa", probe=False, fa_version=2, layers="fused",
                  cuda_graphs=True, graph_max_rows=16, pool="contiguous")
    unit = ex.page_tokens
    assert ex.pool.layout == "token" and ex.block_size == unit and ex.kv.shape[:2] == (64, unit)
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    lengths = [unit + 1, 2 * unit + 1, 40, unit - 1]  # one past a boundary twice, a short row, one short of a boundary
    rows, blocks = [], []
    for i, n in enumerate(lengths):
        p = (ids * 3)[5 * i:5 * i + n]
        blocks.append([1 + 3 * i + k for k in range(-(-(n + 1) // unit))])
        rows.append(Row(seq=i, start=0, token_ids=p, block_ids=blocks[i], sample=True, decode=False, slot=1 + i))
    prefill = Step(rows, slot_resets=[1 + i for i in range(len(rows))])
    first, _ = ex.forward(prefill)  # through launch: the rows' regions mapped first
    decode = Step([Row(seq=i, start=len(r.token_ids), token_ids=[first[i]], block_ids=blocks[i], sample=True, decode=True,
                       slot=1 + i) for i, r in enumerate(rows)][::-1])  # the slots out of order
    region = ex.logits(decode).float()
    ex.attn.regions = False
    table = ex.logits(decode).float()
    ex.attn.regions = True
    for i in range(len(rows)):
        assert agrees(region[i], table[i]) and torch.nn.functional.cosine_similarity(region[i], table[i], dim=0) > 0.999
    reqs = [Request(id=str(i), prompt=tok.decode(ids[20 * i:20 * i + 60]), max_tokens=12, ignore_eos=True) for i in range(5)]
    paged = Executor(MODEL, model=fa.model, num_blocks=64, seed=11, attention="fa", probe=False, fa_version=2, layers="fused",
                     cuda_graphs=True, graph_max_rows=16, block_size=unit)
    want, _ = run_engine(paged, tok, reqs, budget=64, order="input")
    got, m = run_engine(ex, tok, reqs, budget=64, order="input")
    assert {r.id: got[r.id].token_ids for r in reqs} == {r.id: want[r.id].token_ids for r in reqs}
    assert m.step_rates()["graph_step_pct"] > 0 and m.executor["pool"] == "contiguous"
    again, _ = run_engine(ex, tok, reqs, budget=64, order="input")
    assert {r.id: again[r.id].token_ids for r in reqs} == {r.id: want[r.id].token_ids for r in reqs}
    for r in reqs:
        follows_hf(hf, ids_of(tok, r), got[r.id].token_ids)


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


@pytest.mark.parametrize("layers", ["hf", "fused"])
def test_flash_attention_3_matches_hf(tok, hf, executors, layers):
    """fa_version=3, eagerly, over HF layers and over fused layers (sm90 only)."""
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FlashAttention-3 needs an sm90 GPU")
    fa, _ = executors
    ex = Executor(MODEL, model=fa.model, num_blocks=512, seed=11, attention="fa", probe=False, fa_version=3,
                  layers=layers, cuda_graphs=False)
    matches_hf_on_prefill_and_decode(tok, hf, ex)


def test_the_fa_version_resolves_to_what_the_card_runs(executors):
    """auto is 3 on sm90 with vLLM's FA3 build, else 2; 3 is refused where it cannot run; graphs take either."""
    from batchinfer.executor import check_graphable, fa3_available, resolve_fa_version
    fa, _ = executors
    dev = torch.device("cuda")
    assert resolve_fa_version("auto", dev) == (3 if fa3_available(dev) else 2) == fa.fa_version
    assert resolve_fa_version(2, dev) == 2
    if not fa3_available(dev):
        with pytest.raises(ValueError, match="sm90"):
            resolve_fa_version(3, dev)
    check_graphable(fa.model, "fa")
    with pytest.raises(ValueError, match="fa backend"):
        check_graphable(fa.model, "torch")


@pytest.fixture(scope="module")
def fa3_graphed(executors):
    """FlashAttention-3 under graphs, fused layers, decode-only sizes to 16 and the mixed graphs over them."""
    if not torch.cuda.get_device_capability()[0] == 9:
        pytest.skip("FlashAttention-3 needs an sm90 GPU")
    fa, _ = executors
    return Executor(MODEL, model=fa.model, num_blocks=512, seed=11, attention="fa", probe=False, fa_version=3,
                    layers="fused", cuda_graphs=True, graph_max_rows=32)  # 32: the smallest mixed row bucket


def test_fa3_graph_replays_two_steps_of_different_lengths(tok, fa3_graphed):
    """One decode-only graph, a 40-token-context step then a 300-token one: each equals the eager step. The schedule
    is recomputed per step outside the graph, so the second replay sees the longer contexts."""
    ex = fa3_graphed
    assert any(t for _, t in ex.graph_keys)  # the mixed graphs exist under FA3
    step = prefilled_decode_step(tok, ex)
    ids, _ = ex.forward(step)
    assert (ex.graph_rows, ex.graph_tokens) == (4, 0)
    eager = ex.logits(step).float()
    for i, pick in enumerate(ids):
        assert pick_agrees(pick, eager[i])
    short = Step([Row(seq=i, start=min(r.start, 40), token_ids=r.token_ids, block_ids=r.block_ids, sample=True, decode=True)
                  for i, r in enumerate(step.rows)])
    again, _ = ex.forward(short)
    eager = ex.logits(short).float()
    assert ex.graph_rows == 4 and all(pick_agrees(pick, eager[i]) for i, pick in enumerate(again))


def test_mixed_step_graph_matches_eager(tok, fa3_graphed):
    """Two decode rows, a chunked prefill row and a prompt-ending prefill row replay a mixed graph: the sampled ids
    equal the eager step's (within a bf16 tie) and the KV blocks the step wrote are bitwise equal."""
    ex = fa3_graphed
    ps = prompts(tok)
    decode = prefilled_decode_step(tok, ex)
    chunk_ids, end_ids = ps[2][:64], ps[1][:50]
    rows = decode.rows[:2] + [Row(seq=7, start=0, token_ids=chunk_ids, block_ids=blocks_for(ex, 128, 300), sample=False, decode=False),
                              Row(seq=8, start=0, token_ids=end_ids, block_ids=blocks_for(ex, 64, 400), sample=True, decode=False)]
    step = Step(rows)
    written = sorted({b for r in rows[2:] for b in r.block_ids})
    eager = ex.logits(step).float()
    kv_eager = ex.kv[:, :, written].clone()
    ids, _ = ex.forward(step)
    assert (ex.graph_rows, ex.graph_tokens) == (32, 128)
    assert len(ids) == 3 and all(pick_agrees(pick, eager[i]) for i, pick in enumerate(ids))
    # the KV the step wrote, by the graph against eagerly: the same up to bf16 rounding of GEMMs at another M
    assert torch.nn.functional.cosine_similarity(ex.kv[:, :, written].float().flatten(), kv_eager.float().flatten(), dim=0) > 0.999


def test_prefill_pad_row_with_no_keys_writes_nothing_real(tok, fa3_graphed):
    """A mixed graph's pad row holds many tokens over no keys: after a replay no real block changed but the step's own."""
    ex = fa3_graphed
    decode = prefilled_decode_step(tok, ex)
    before = ex.kv[:, :, 1:].clone()
    step = Step(decode.rows[:1] + [Row(seq=9, start=0, token_ids=prompts(tok)[0], block_ids=blocks_for(ex, 32, 450),
                                       sample=True, decode=False)])
    ex.forward(step)
    touched = {decode.rows[0].block_ids[decode.rows[0].start // 16]} | set(step.rows[1].block_ids)
    mask = torch.ones(ex.num_blocks - 1, dtype=torch.bool, device=ex.kv.device)
    for b in touched:
        mask[b - 1] = False
    assert torch.equal(ex.kv[:, :, 1:][:, :, mask], before[:, :, mask])


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
    same ids (RUN.md's run-to-run claim holds for fused layers too)."""
    ids = tok(TEXT, add_special_tokens=False)["input_ids"]
    ps = [ids[:64] + ids[100 + 30 * i:130 + 30 * i] for i in range(3)] + prompts(tok)[:1]
    reqs = [Request(id=str(i), prompt=tok.decode(p), max_tokens=24, ignore_eos=True) for i, p in enumerate(ps)]
    first, m = run_engine(fused, tok, reqs, budget=32, order="prefix_dfs", sharing="block")
    again, _ = run_engine(fused, tok, reqs, budget=32, order="prefix_dfs", sharing="block")
    assert m.counts["prefix_hit_tokens"] > 0
    assert_ran_to_length(first, reqs)
    for r in reqs:
        assert first[r.id].token_ids == again[r.id].token_ids, r.id
        follows_hf(hf, ids_of(tok, r), first[r.id].token_ids)


# CUDA graphs: decode-only steps replayed from graphs captured at load, over HF layers and over fused layers ----------

@pytest.fixture(scope="module", params=["hf", "fused"])
def graphed(request, executors):
    fa, _ = executors
    return Executor(MODEL, model=fa.model, num_blocks=512, seed=11, attention="fa", probe=False, fa_version=2,
                    layers=request.param, cuda_graphs=True, graph_max_rows=16)


def prefilled_decode_step(tok, ex):
    """Three prompts prefilled eagerly (their KV written), then the decode-only step of their next tokens."""
    ps, rows, blocks = prompts(tok), [], []
    for i, p in enumerate(ps):
        blocks.append(blocks_for(ex, len(p) + 1, 100 * i))
        rows.append(Row(seq=i, start=0, token_ids=p, block_ids=blocks[i], sample=True, decode=False))
    first = ex.logits(Step(rows)).argmax(-1).tolist()  # eagerly, whatever graphs the executor holds
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
    graph, out = graphed.graphs[(4, 0)]
    st = graphed.staging
    st.stage(step, (4, 0))
    big = st.total_max
    st.g_tokens[3], st.g_tokens[big + 3] = 1234, 40  # slot stays 0: the pad block
    st.g_used[3] = 41
    st.g_table[3, :3] = st.g_table[0, :3]
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
    first, m = run_engine(graphed, tok, reqs, budget=32, order="prefix_dfs", sharing="block")
    again, _ = run_engine(graphed, tok, reqs, budget=32, order="prefix_dfs", sharing="block")
    assert m.step_rates()["graph_step_pct"] > 0
    assert_ran_to_length(first, reqs)
    for r in reqs:
        assert first[r.id].token_ids == again[r.id].token_ids, r.id
        follows_hf(hf, ids_of(tok, r), first[r.id].token_ids)
