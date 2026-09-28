"""bench/utilisation.py: model FLOPs and bytes for MFU and MBU, the same count for every engine."""
from types import SimpleNamespace as NS

import bench.run
from batchinfer.metrics import REQUEST_FIELDS
from bench.run import Job, timed_pass
from bench.utilisation import Shape, forward_passes, shape_of, utilisation, work

# 2 layers, hidden 4, heads x head dim 4 for queries and 2 for keys and values
TINY = Shape(layers=2, hidden=4, q_dim=4, kv_dim=2, intermediate=8, vocab=10, dtype_bytes=2)
A100 = "NVIDIA A100-SXM4-40GB"


def trace(*requests):
    return {"prompt_len": [r[0] for r in requests], "prefix_hit_tokens": [r[1] for r in requests],
            "output_tokens": [r[2] for r in requests]}


def test_shape_counts_matmul_weights_and_kv():
    assert TINY.body_params == 2 * (2 * 4 * 4 + 2 * 4 * 2 + 3 * 4 * 8) == 288  # q+o, k+v, gate+up+down
    assert TINY.head_params == 40 and TINY.weight_bytes == (288 + 40) * 2
    assert TINY.kv_bytes_per_token == 2 * 2 * 2 * 2  # K and V, layers, kv_dim, bf16


def test_work_counts_only_what_the_engine_computed():
    # prompt 5 with 2 prefix-hit tokens, 3 output tokens: prefill computes positions 2..4, decode feeds back 2
    # tokens at positions 5 and 6; a query at position p reads p + 1 keys
    w = work(TINY, [(5, 2, 3)])
    assert w["query_tokens"] == 3 + 2
    assert w["dense_flops"] == 2 * 288 * 5 + 2 * 40 * 3  # lm_head for every sampled token, the first from prefill
    assert w["attention_flops"] == 4 * 2 * 4 * ((3 + 4 + 5) + (6 + 7))
    assert w["kv_tokens_moved"] == 5 + (6 + 7)  # the prompt's KV once (2 read, 3 written), then each decode's context
    assert work(TINY, [(5, 0, 1)])["query_tokens"] == 5  # one output token: prefill only, no decode


def test_forward_passes_per_engine_record():
    assert forward_passes({"steps": {"decode_rows": [0, 1, 1]}}) == 3  # batchinfer, traced vLLM
    assert forward_passes({"groups": [{"decode_steps": 4}, {"decode_steps": 0}]}) == 5 + 1  # naive: prefill + decode
    assert forward_passes({"meta": {}}) is None and forward_passes(None) is None


def test_utilisation_against_the_cards_own_peaks():
    details = {"steps": {"decode_rows": [0, 1, 1]}, "request_trace": trace((5, 2, 3))}
    flops = 3120 + 800
    moved = 3 * TINY.weight_bytes + 18 * TINY.kv_bytes_per_token
    mfu, mbu, record = utilisation(TINY, details, A100, flops / 312e12)  # wall_s such that MFU is exactly 100%
    assert mfu == 100.0 and mbu == round(100 * moved / (flops / 312e12) / 1555e9, 1)
    assert record["forward_passes"] == 3 and record["attention_flop_pct"] == round(100 * 800 / flops, 1)
    assert record["peak_bf16_tflops"] == 312 and record["peak_hbm_gb_per_s"] == 1555 and record["shape_layers"] == 2
    mfu, _, _ = utilisation(TINY, details, "NVIDIA H100 PCIe", flops / 312e12)
    assert mfu == round(100 * 312 / 756, 1)  # each card against its own peak
    mfu, mbu, record = utilisation(TINY, details, "NVIDIA RTX 6000 Ada", 1.0)  # no listed peak: no number
    assert mfu is None and mbu is None and record["model_tflop"] is not None
    _, mbu, record = utilisation(TINY, {"request_trace": trace((5, 2, 3))}, A100, 1.0)  # no forward count
    assert mbu is None and record["model_gb_moved"] is None
    assert utilisation(TINY, {"steps": {"decode_rows": [0]}}, A100, 1.0) == (None, None, None)  # no request trace
    assert utilisation(None, details, A100, 1.0) == (None, None, None)  # no model shape


def test_shape_of_a_dense_config_only():
    qwen3 = NS(num_hidden_layers=28, hidden_size=2048, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
               intermediate_size=6144, vocab_size=151936, torch_dtype="bfloat16")
    s = shape_of(qwen3)
    assert (s.q_dim, s.kv_dim, s.dtype_bytes) == (2048, 1024, 2)
    assert s.body_params == 1_409_286_144  # Qwen3-1.7B's matmul weights outside embeddings and lm_head
    assert shape_of(NS(**{**vars(qwen3), "head_dim": None})).q_dim == 2048  # hidden / heads without head_dim
    assert shape_of(NS(**{**vars(qwen3), "torch_dtype": "torch.float32"})).dtype_bytes == 4
    assert shape_of(NS(**vars(qwen3), num_experts=64)) is None  # a mixture of experts is not counted as dense
    assert shape_of(NS(hidden_size=8)) is None


class Traced:
    """An engine whose record carries a request trace and steps, as batchinfer's and traced vLLM's do."""

    def generate(self, requests):
        self.requests = requests
        return [{"id": r["id"], "text": "x", "output_tokens": r["max_tokens"], "finish_reason": "length",
                 "token_ids": [0] * r["max_tokens"]} for r in requests]

    def details(self):
        rows = [dict(dict.fromkeys(REQUEST_FIELDS), idx=i, id=r["id"], analysis_kind="generate", prompt_len=5,
                     max_tokens=r["max_tokens"], output_tokens=r["max_tokens"], prefix_hit_tokens=0,
                     finish_reason="length") for i, r in enumerate(self.requests)]
        return {"steps": {"decode_rows": [0, 1]}, "request_trace": {k: [row[k] for row in rows] for k in REQUEST_FIELDS}}


def test_timed_pass_records_mfu_and_mbu(tmp_path, monkeypatch):
    requests = [{"id": "a", "prompt": "p", "max_tokens": 2, "ignore_eos": True, "labels": None, "kind": "generate",
                 "source": "s", "prompt_tokens": 5}]
    workload = tmp_path / "w.jsonl"
    workload.write_text("{}\n")
    monkeypatch.setattr(bench.run, "load_shape", lambda model: TINY)
    job = Job(workload, requests, {}, "tiny", A100, "sha")
    m = timed_pass(Traced(), job, "dummy", {}, 0.0, tmp_path / "out")
    assert m["mfu_pct"] is not None and m["mbu_pct"] is not None
    assert m["utilisation"]["forward_passes"] == 2 and m["utilisation"]["query_tokens"] == 5 + 1
    assert "Utilisation" in (tmp_path / "out" / "report.html").read_text()
