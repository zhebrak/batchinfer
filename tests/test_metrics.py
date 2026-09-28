"""metrics: rates from counters, timers accumulate, flat_stats() is exactly the report columns."""
import pytest

from batchinfer.metrics import FLAT_KEYS, REQUEST_FIELDS, STEP_FIELDS, STEP_FLAT_KEYS, Metrics


def step(*values):
    """A step record in STEP_FIELDS order, as the keywords record_step takes."""
    return dict(zip(STEP_FIELDS, values, strict=True))


def test_rates_and_flat_stats():
    m = Metrics()
    m.timing.update(prefill=2.0, decode=4.0, inference=8.0)
    m.add(requests=4, prompt_tokens=1000, padded_prompt_tokens=1250, output_tokens=400,
          decode_steps=100, decode_slot_steps=400, live_slot_steps=300, finish_stop=1, finish_length=3)
    m.set_analysis({"unique_prompt_tokens": 700, "ideal_prefix_reuse": 0.3})
    m.set_policy({"groups": 2, "padded_groups": 1, "oversized": 0, "in_batch_reuse": 0.1, "max_batch_tokens": 65536})
    m.memory.update(peak_mem_reserved_gb=12.5, gpu_name="test")
    r = m.rates()
    assert r["prefill_tok_per_s"] == 500.0 and r["padded_prefill_tok_per_s"] == 625.0
    assert r["decode_tok_per_s"] == 75.0 and r["total_tok_per_s"] == 175.0 and r["req_per_s"] == 0.5  # 300 live slots / 4 s
    assert r["mean_decode_batch"] == 3.0 and r["wasted_decode_slots"] == 100
    assert r["decode_step_ms_mean"] == 40.0 and r["padding_waste_pct"] == 20.0 and r["prefix_hit_pct"] == 0.0
    m.add(prefill_tokens=250)  # the batchinfer engine counts what it computed itself: the rest was read from shared KV
    assert m.rates()["prefix_hit_pct"] == 75.0
    m.counts["prefill_tokens"] = 0
    flat = m.flat_stats()
    assert tuple(flat) == FLAT_KEYS
    assert all(isinstance(v, (int, float)) or v is None for v in flat.values())
    assert flat["groups"] == 2 and flat["peak_mem_reserved_gb"] == 12.5 and flat["wasted_decode_slots"] == 100
    assert flat["max_batch_tokens"] == 65536  # from the policy, so rows on different GPUs show their budget
    assert flat["prefill_s"] == 2.0 and flat["decode_s"] == 4.0
    d = m.to_dict()
    assert set(d) == {"meta", "timing", "volume", "rates", "memory", "analysis", "policy", "groups"}
    assert d["policy"]["groups"] == 2 and "groups" not in d["volume"]  # a decision, not a count
    assert "total_s" in d["timing"] and d["volume"]["unique_prompt_tokens"] == 700
    assert "4 requests in 2 groups" in m.summary()


def test_timer_accumulates_and_reports_elapsed():
    m = Metrics()
    with m.timer("prefill") as t1:
        pass
    with m.timer("prefill") as t2:
        pass
    assert m.timing["prefill"] >= t1.elapsed + t2.elapsed - 1e-9
    m.finish()
    assert m.timings()["total_s"] >= m.timings()["prefill_s"]


def test_empty_metrics_do_not_divide_by_zero():
    m = Metrics()
    r = m.rates()
    assert r["prefill_tok_per_s"] is None and r["mean_decode_batch"] is None and r["padding_waste_pct"] is None
    assert m.flat_stats()["prefix_hit_pct"] == 0.0
    m.summary()


def test_step_rates_from_the_trace():
    m = Metrics(flat_keys=STEP_FLAT_KEYS)
    m.timing["inference"] = 2.0
    m.model_shape = {"n_body_params": 10**12, "hidden": 10, "vocab": 50}  # big enough for MFU to show
    m.set_policy({"prefill_budget": 8, "max_batch_tokens": 196608})
    m.memory["gpu_name"] = "NVIDIA A100-SXM4-40GB"
    # decode_rows, prefill_tokens, prefill_rows, admitted, free_blocks, step_ms, gpu_ms, sched_ms, trie_blocks,
    # pinned_blocks, step_prefill_budget, kv_reserved_tokens, kv_written_tokens, unprefilled_tokens, end_ms, graph_rows
    m.record_step(**step(0, 8, 2, 2, 10, 100.0, 90.0, 1.0, 2, 0, 8, 64, 0, 40, 102.0, 0))  # prefill only
    m.record_step(**step(2, 6, 1, 3, 9, 100.0, 90.0, 1.0, 3, 1, 8, 64, 16, 30, 204.0, 0))  # mixed
    m.record_step(**step(3, 0, 0, 3, 9, 50.0, 40.0, 1.0, 1, 0, 16, 80, 40, 0, 256.0, 4))  # decode only, budget raised
    m.add(sampled_rows=7, chain_start_step=1, prefix_wait_steps=2, prompt_tokens=20, prefill_tokens=14)
    m.set_analysis({"ideal_prefix_reuse": 0.35, "ideal_prefix_reuse_page16": 0.3})  # the block-level ceiling is a column
    r = m.step_rates()
    assert r["steps"] == 3 and r["decode_only_steps"] == 1 and r["prefill_only_steps"] == 1
    assert r["mean_step_tokens"] == round((5 + 14) / 3, 1) and r["mean_decode_batch"] == 2.5
    assert r["gpu_busy_pct"] == round(100 * 0.22 / 2.0, 1) and r["graph_step_pct"] == round(100 / 3, 1)
    flops_per_s = (2 * 10**12 * 19 + 2 * 10 * 50 * 7) / 2.0
    assert r["dense_mfu_pct"] == round(100 * flops_per_s / 312e12, 1) == 6.1
    m.memory["gpu_name"] = "NVIDIA H100 PCIe"  # each GPU against its own peak
    assert m.step_rates()["dense_mfu_pct"] == round(100 * flops_per_s / 756e12, 1) == 2.5
    m.memory["gpu_name"] = "NVIDIA RTX 6000 Ada"  # no known peak: no number rather than a wrong one
    assert m.step_rates()["dense_mfu_pct"] is None
    assert r["prefix_wait_steps"] == 2 and r["pinned_blocks_peak"] == 1 and r["trie_blocks_peak"] == 3
    flat = m.flat_stats()
    assert tuple(flat) == STEP_FLAT_KEYS and flat["prefill_budget"] == 8 and flat["max_batch_tokens"] == 196608
    assert flat["prefill_budget_peak"] == 16  # measured from the trace, next to the decided floor
    assert flat["kv_occupancy_pct"] == round(100 * (0 + 16 / 64 + 40 / 80) / 3, 1)  # mean over steps of written/held
    assert r["kv_reserved_tokens_peak"] == 80
    assert flat["prefix_hit_pct"] == 30.0 and flat["ideal_prefix_reuse_page16"] == 0.3 and "ideal_prefix_reuse" not in flat
    assert set(m.to_dict()["steps"]) >= {"decode_rows", "gpu_ms", "pinned_blocks"}
    m.record_step(**step(1, 0, 0, 1, 9, 10.0, None, 1.0, 0, 0, 8, 16, 16, 0, 268.0, 0))  # off the GPU, gpu_ms is None
    assert m.step_rates()["steps"] == 4
    assert "steps: 4" in m.summary() and "pinned peak 1" in m.summary()



def trace_step(**given):
    """A step record by field name, 0 for every field not given, so a trace column added later needs no edit here."""
    assert set(given) <= set(STEP_FIELDS), sorted(set(given) - set(STEP_FIELDS))
    return {k: given.get(k, 0) for k in STEP_FIELDS}


def test_valleys_and_kv_peaks_from_the_trace():
    """prefill_small_step_pct counts steps under MIN_PREFILL_BUDGET query tokens (BatchLLM's valleys) within the prefill
    phase only; the KV peaks are shares of the pool (kv_blocks less the pad block): reserved as sampled at step start,
    written at step end, before finished requests release their blocks."""
    m = Metrics(flat_keys=STEP_FLAT_KEYS)
    # decode rows, prefill tokens, KV reserved and written at step start. Step 0 prefills 400 tokens of max_tokens=1
    # prompts that finish in that step, so no step-start sample ever sees them written; steps 3 and 4 are the decode
    # tail after the last prefill, which no scheduler could have filled
    for decode, prefill, reserved, written in [(0, 400, 420, 0), (2, 0, 64, 40), (2, 260, 400, 42), (4, 0, 200, 150),
                                               (1, 0, 128, 100)]:
        m.record_step(**trace_step(decode_rows=decode, prefill_tokens=prefill, kv_reserved_tokens=reserved,
                                   kv_written_tokens=written))
    r = m.step_rates()  # query tokens per step: 400, 2, 262 | 4, 1
    assert r["prefill_phase_steps"] == 3 and r["prefill_small_step_pct"] == 33.3  # step 1's 2 decode rows only
    assert r["kv_reserved_peak_pct"] is None and r["kv_written_peak_pct"] is None  # no pool size, no share
    m.add(kv_blocks=41)  # 40 usable blocks of 16: a 640-token pool
    r = m.step_rates()
    assert r["kv_reserved_peak_pct"] == round(100 * 420 / 640, 1)
    assert r["kv_written_peak_pct"] == 62.5  # step 0's 400 written tokens; the step-start samples peak at 150
    flat = m.flat_stats()
    assert tuple(flat) == STEP_FLAT_KEYS
    assert flat["prefill_small_step_pct"] == 33.3 and flat["kv_written_peak_pct"] == 62.5
    assert "33.3% of the 3 prefill-phase steps under 256" in m.summary()
    assert "peak 65.6% reserved and 62.5% written" in m.summary()
    decode_only = Metrics()
    decode_only.record_step(**trace_step(decode_rows=3))
    r = decode_only.step_rates()  # no prefill at all: no prefill phase to judge
    assert r["prefill_phase_steps"] == 0 and r["prefill_small_step_pct"] is None


def test_record_step_takes_every_field_by_name():
    m = Metrics()
    fields = step(1, 0, 0, 1, 9, 10.0, None, 1.0, 0, 0, 8, 16, 16, 0, 12.0, 0)
    with pytest.raises(AssertionError, match="record_step needs exactly"):
        m.record_step(**{k: v for k, v in fields.items() if k != "pinned_blocks"})  # a column added in one place only
    m.record_step(**dict(reversed(fields.items())))  # keyword order does not matter
    assert m.steps == [tuple(fields.values())]


def request(**overrides):
    """A request_trace entry, every REQUEST_FIELDS key, as record_request takes it."""
    fields = dict(idx=0, id="r0", analysis_kind="generate", group=None, prompt_len=10, max_tokens=4, output_tokens=4,
                  prefix_hit_tokens=0, finish_reason="length", admitted_step=0, prefill_start_step=0,
                  first_token_step=0, finished_step=3, admitted_ms=0.0, prefill_start_ms=0.0, first_token_ms=5.0,
                  finished_ms=20.0)
    assert fields.keys() == set(REQUEST_FIELDS)
    return {**fields, **overrides}


def test_record_request_takes_every_field_by_name_and_to_dict_is_columnar():
    m = Metrics()
    with pytest.raises(AssertionError, match="record_request needs exactly"):
        m.record_request(**{k: v for k, v in request().items() if k != "prefill_start_step"})
    assert "request_trace" not in m.to_dict()  # nothing recorded, no section
    m.record_request(**request())
    m.record_request(**dict(reversed(request(idx=1, id="r1", finished_ms=30.0).items())))  # keyword order is free
    trace = m.to_dict()["request_trace"]
    assert list(trace) == list(REQUEST_FIELDS)
    assert trace["id"] == ["r0", "r1"] and trace["finished_ms"] == [20.0, 30.0]
    assert "request_trace" not in m.flat_stats()  # a trace, never a report column
