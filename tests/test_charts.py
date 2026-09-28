"""bench/charts.py: the rows each chart draws come from the record as recorded, on one time origin. CPU only."""
import json

from stub import StubTokenizer

from batchinfer.analysis import analyze
from batchinfer.metrics import Metrics
from batchinfer.policy import decide
from batchinfer.schema import PolicyConfig, Request
from bench.charts import PALETTES, charts, log_ticks

STEP_IDS = ["tokens_per_step", "in_flight", "kv", "step_time", "timeline", "prompt_len", "output_len", "decode_ratio",
            "prompt_hist", "hit_hist", "output_hist"]


def policy(**cfg):
    """The policy section as the engines record it, from policy.decide itself: a renamed engine or knob fails here."""
    job = analyze([Request(id="r0", prompt="a b c", max_tokens=3), Request(id="r1", prompt="a b d e", max_tokens=3)],
                  StubTokenizer())
    return decide(job, PolicyConfig(**cfg)).to_dict()


def step_record():
    """Three batchinfer steps: the first two carry prefill, and every request's KV is reserved at step 0."""
    m = Metrics()
    m.timing.update(analysis=0.5, policy=0.25, inference=0.03)  # the job started 750 ms before the first step
    m.set_policy(policy(prefill_budget=8))
    m.add(kv_blocks=11)  # 10 usable blocks
    for k, (d, p, end) in enumerate([(0, 8, 10.0), (1, 4, 18.0), (2, 0, 30.0)]):
        m.record_step(decode_rows=d, prefill_tokens=p, prefill_rows=1 if p else 0, admitted=2, free_blocks=6,
                      step_ms=end / 2, gpu_ms=None, sched_ms=0.1, trie_blocks=1, pinned_blocks=0,
                      step_prefill_budget=8, kv_reserved_tokens=64, kv_written_tokens=8 * (k + 1),
                      unprefilled_tokens=0, end_ms=end, graph_rows=0)
    for idx, (kind, prompt, start, first, hits) in enumerate([("generate", 8, 0, 0, 0), ("label", 12, 1, 1, 8)]):
        m.record_request(idx=idx, id=f"r{idx}", analysis_kind=kind, group=None, prompt_len=prompt, max_tokens=3,
                         output_tokens=3 - idx, prefix_hit_tokens=hits, finish_reason="length" if idx == 0 else "stop",
                         admitted_step=0, prefill_start_step=start, first_token_step=first, finished_step=2,
                         admitted_ms=0.0, prefill_start_ms=[0.0, 10.0][start], first_token_ms=[10.0, 18.0][first],
                         finished_ms=30.0)
    return m.to_dict()


def by_id(drawn):
    return {c["id"]: c for c in drawn}


def data_names(spec):
    names = {spec["data"]["name"]} if "data" in spec else set()
    return names | {n for layer in spec.get("layer", []) for n in data_names(layer)}


def test_step_record_draws_every_chart_on_the_jobs_clock():
    drawn, data, notes = charts(step_record())
    assert [c["id"] for c in drawn] == STEP_IDS and notes == []
    steps, requests = data["steps"], data["requests"]
    assert len(steps) == 3 and len(requests) == 2
    # seconds since the job started: the engine's end_ms shifted by analysis + policy
    assert [(s["x0"], s["x1"]) for s in steps] == [(0.75, 0.76), (0.76, 0.768), (0.768, 0.78)]
    assert [s["interval_ms"] for s in steps] == [10.0, 8.0, 12.0]
    assert [s["kind"] for s in steps] == ["prefill-only", "mixed", "decode-only"]
    assert [s["waiting"] for s in steps] == [0, 0, 0]  # both reserved KV at step 0
    # held is sampled with written, at step start: 64 of 160 token slots; written grows
    assert [s["kv_held_pct"] for s in steps] == [40.0] * 3 and [s["kv_written_pct"] for s in steps] == [5.0, 10.0, 15.0]
    assert requests[1]["prefill_start_s"] == 0.76 and requests[1]["computed_prompt_tokens"] == 4
    for c in drawn:  # every spec reads only datasets the page ships
        assert data_names(c["light"]) <= set(data), c["id"]
        assert json.dumps(c["light"]).count("#") == json.dumps(c["dark"]).count("#")  # same marks, other colours
    json.dumps(data)  # the page embeds it as JSON


def test_prompt_length_is_plotted_where_prefill_started_not_at_admission():
    """Every request reserved its KV at step 0; the chart must still spread them by when their prefill ran."""
    drawn, data, _ = charts(step_record())
    spec = by_id(drawn)["prompt_len"]["light"]
    assert spec["encoding"]["x"]["field"] == "prefill_start_s"
    assert sorted({r["prefill_start_s"] for r in data["requests"]}) == [0.75, 0.76]
    assert by_id(drawn)["decode_ratio"]["light"]["encoding"]["x"]["field"] == "prefill_start_s"
    phases = [(t["rank"], t["phase"]) for t in data["timeline"]]
    assert (1, "KV reserved, waiting") in phases and (0, "KV reserved, waiting") not in phases


def test_budget_follows_the_trace_and_colours_follow_the_entity():
    drawn, data, _ = charts(step_record())
    assert {s["budget"] for s in data["steps"]} == {8}
    tokens = by_id(drawn)["tokens_per_step"]
    for theme in ("light", "dark"):
        scale = tokens[theme]["layer"][0]["encoding"]["color"]["scale"]
        assert scale["domain"] == ["decode tokens", "prefill tokens"]
        assert scale["range"] == [PALETTES[theme]["blue"], PALETTES[theme]["orange"]]


def test_static_record_draws_measured_group_bars_and_its_requests():
    m = Metrics()
    m.set_policy(policy(engine="naive", order="input", admission="groups", prefix_sharing=False))
    m.add_group(group=0, n=2, kinds={"generate": 2}, padded=True, padded_len=10, T=4, prompt_tokens=15, start_ms=1.0,
                shared_prefix_len=0, prefill_s=0.01, decode_steps=3, decode_s=0.03, wasted_slots=2)
    for i, (prompt, out) in enumerate([(10, 4), (5, 2)]):
        m.record_request(idx=i, id=f"r{i}", analysis_kind="generate", group=0, prompt_len=prompt, max_tokens=4,
                         output_tokens=out, prefix_hit_tokens=0, finish_reason="length", admitted_step=None,
                         prefill_start_step=None, first_token_step=None, finished_step=None, admitted_ms=1.0,
                         prefill_start_ms=1.0, first_token_ms=11.0, finished_ms=11.0 + 10.0 * (out - 1))
    drawn, data, notes = charts(m.to_dict())
    ids = [c["id"] for c in drawn]
    assert ids[0] == "tokens_per_group" and "in_flight" not in ids and "hit_hist" not in ids and notes == []
    assert data["groups"] == [{"group": 0, "n": 2, "padded_len": 10, "prefill": 15, "padding": 5, "decode": 4,
                               "dead_slots": 2, "wasted": 7, "prefill_s": 0.01, "decode_s": 0.03, "decode_steps": 3}]
    assert "apportioned" in by_id(drawn)["output_len"]["description"]


def vllm_record():
    return {"meta": {"engine": "vllm", "trace": True},
            "steps": {"decode_rows": [0, 2, 2], "prefill_tokens": [20, 5, 0], "prefill_rows": [2, 1, 0],
                      "end_ms": [5.0, 9.0, 12.0], "running": [2, 2, 2], "waiting": [1, 0, 0],
                      "kv_cache_usage_pct": [3.0, 3.5, 3.5]},
            "request_trace": {"idx": [0], "id": ["r0"], "analysis_kind": ["generate"], "group": [None],
                              "prompt_len": [20], "max_tokens": [3], "output_tokens": [3], "prefix_hit_tokens": [16],
                              "finish_reason": ["length"], "admitted_step": [None], "prefill_start_step": [None],
                              "first_token_step": [None], "finished_step": [None], "admitted_ms": [0.5],
                              "prefill_start_ms": [0.5], "first_token_ms": [5.0], "finished_ms": [12.0]}}


def test_vllm_record_uses_its_own_names_and_clock():
    drawn, data, _ = charts(vllm_record())
    assert "kv" in [c["id"] for c in drawn]
    first = data["steps"][0]
    assert (first["x0"], first["x1"]) == (0.0, 0.005)  # vLLM's origin is generate()'s start: no shift
    assert (first["running"], first["waiting"], first["kv_in_use_pct"]) == (2, 1, 3.0) and "kv_held_pct" not in first
    kv = json.dumps(by_id(drawn)["kv"]["light"])
    assert "in use (allocated as written)" in kv and '"held"' not in kv  # never labelled like our reservation
    assert "admitted" not in first  # our term, not vLLM's
    in_flight = json.dumps(by_id(drawn)["in_flight"]["light"])
    assert '"running"' in in_flight and "admitted (KV reserved)" not in in_flight


def test_reference_trace_shares_the_time_axis():
    drawn, data, _ = charts(step_record(), {"record": vllm_record(), "prefix_cache": True})
    ours, theirs = by_id(drawn)["tokens_per_step"], by_id(drawn)["tokens_per_step_vllm"]
    domain = ours["light"]["layer"][0]["encoding"]["x"]["scale"]["domain"]
    assert domain == theirs["light"]["layer"][0]["encoding"]["x"]["scale"]["domain"] == [0, 0.78]
    assert theirs["light"]["data"]["name"] == "reference_steps" and len(data["reference_steps"]) == 3


def test_a_record_from_before_end_ms_falls_back_to_the_step_index():
    old = step_record()
    del old["steps"]["end_ms"], old["request_trace"]
    for k in ("step_prefill_budget", "kv_reserved_tokens", "kv_written_tokens"):
        del old["steps"][k]
    old["policy"]["engine"] = "step"  # the engine's name before the rename
    drawn, data, notes = charts(old)
    assert [c["id"] for c in drawn] == ["tokens_per_step", "in_flight", "kv"] and "predates end_ms" in notes[0]
    assert [(s["x0"], s["x1"]) for s in data["steps"]] == [(0, 1), (1, 2), (2, 3)]
    assert data["steps"][0]["kv_held_pct"] == 40.0  # 10 usable blocks, 6 free after commit
    assert by_id(drawn)["tokens_per_step"]["light"]["layer"][0]["encoding"]["x"]["title"] == "step"


def test_no_trace_no_charts():
    assert charts({"meta": {}, "timing": {}}) == ([], {}, [])


def test_log_ticks_stay_sparse():
    assert log_ticks([1, 7, 250]) == [1, 3, 10, 30, 100, 300]
    assert log_ticks([0, None]) is None


def test_tokens_per_step_marks_where_the_run_ends():
    drawn, data, _ = charts(step_record(), {"record": vllm_record(), "prefix_cache": True})
    for cid, last in (("tokens_per_step", 2), ("tokens_per_step_vllm", 2)):
        done = next(layer for layer in by_id(drawn)[cid]["light"]["layer"] if "layer" in layer)
        assert done["transform"][0] == {"filter": f"datum.step === {last}"}, cid
    assert data["steps"][-1]["x1"] == 0.78 and data["reference_steps"][-1]["x1"] == 0.012
