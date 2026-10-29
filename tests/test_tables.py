"""bench.tables: the results index's arms matrix from rows. CPU only."""
import pytest

from bench import tables
from bench.suite import Arm


def row(workload, column, wall, engine="batchinfer", opts=None, reuse=0.8115, gpu="NVIDIA H100 80GB HBM3"):
    return {"workload": f"workloads/{workload}-quick.jsonl", "size": "quick", "model": "Qwen/Qwen3-8B", "gpu": gpu,
            "engine": engine, "opts": opts or {}, "column": column, "wall_s": wall, "ideal_prefix_reuse": reuse,
            "backend_stats": {}}


ROWS = [row("mixed", "vLLM cache on", 8.77, "vllm"),
        row("mixed", "vLLM cache off", 18.3, "vllm", {"enable_prefix_caching": False}),
        row("mixed", "batchinfer", 10.104), row("mixed", "no trie", 20.133, opts={"sharing": "none"}),
        row("mixed", "naive", 156.1, "naive", {"order": "max_tokens_desc", "max_batch_tokens": 65536}),
        row("classify", "vLLM cache on", 2.5, "vllm", reuse=0.86), row("classify", "batchinfer", 2.9, reuse=0.86),
        row("classify", "vLLM capped", 3.1, "vllm", {"kv_cache_memory_bytes": 1})]


def matrix(rows):
    """The arms matrix of rows of one model and GPU: (workload names, {arm column: cells})."""
    (_, rows_by_workload), = tables.by_job(rows).items()
    stems, arms = tables.arms(rows_by_workload)
    return [tables.short(s) for s in stems], {a.column: cells for a, cells in arms}


def test_the_arms_matrix_has_one_row_per_arm_present_and_the_ratio_to_its_reference():
    stems, m = matrix(ROWS)
    assert stems == ["mixed", "classify"]
    assert m["vLLM cache on"] == [(8.77, None), (2.5, None)]  # a reference, shown plain
    assert m["no trie"][0] == (20.133, pytest.approx(20.133 / 8.77)) and m["no trie"][1] is None
    assert m["vLLM capped"] == [None, (3.1, None)]
    assert "HF layers, eager" not in m  # an arm with no row anywhere is left out


def test_a_row_is_the_arm_its_options_match_whatever_column_it_recorded():
    """A --variant run writes the same directory as the arm with those options, so it fills that arm's cell; a
    row without a column and without a matching arm is left out; capped rows go by their recorded column."""
    m = {**row("mixed", None, 1.0), "opts": {"layers": "hf", "cuda_graphs": False}}
    assert tables.column_of(m) == "HF layers, eager"
    assert tables.column_of({**m, "column": "layers=hf, cuda_graphs=False"}) == "HF layers, eager"
    assert tables.column_of({**m, "opts": {"max_num_seqs": 512}, "engine": "vllm"}) is None  # no arm: left out
    assert tables.column_of({**m, "opts": {}}) == "batchinfer"
    assert tables.column_of({**m, "opts": {}, "engine": "vllm", "warm": True}) == "vLLM warm"  # not the reference
    assert tables.column_of({**m, "opts": {"cache": "radix"}, "warm": True}) == "radix warm"
    assert tables.column_of({**m, "opts": {"num_blocks": 6657}, "column": "capped"}) == "capped"


def test_the_share_families_stay_out_and_the_binding_points_join():
    rows = ROWS + [row("dec50x32", "vLLM cache on", 30.0, "vllm", reuse=0.5), row("dec50x32", "batchinfer", 29.0, reuse=0.5),
                   row("sweep50", "batchinfer", 4.0, reuse=0.5), row("quality8", "batchinfer", 6.0, reuse=0.5)]
    assert matrix(rows)[0] == ["mixed", "classify", "dec50x32"]


def test_a_row_with_length_violations_shows_no_wall():
    bad = {**row("mixed", "batchinfer", 10.0), "length_violations": 3}
    assert "batchinfer" not in matrix([bad, row("mixed", "vLLM cache on", 8.0, "vllm")])[1]


def test_a_capped_arm_is_compared_against_vllm_at_the_same_cap():
    rows = [row("pool", "vLLM cache on", 50.0, "vllm"), row("pool", "vLLM capped", 80.0, "vllm", {"kv_cache_memory_bytes": 1}),
            row("pool", "capped", 100.0, opts={"num_blocks": 6657}), row("pool", "batchinfer", 60.0)]
    m = matrix(rows)[1]
    assert m["capped"] == [(100.0, 1.25)] and m["batchinfer"] == [(60.0, 1.2)]


def test_an_arm_is_a_frozen_record():
    with pytest.raises(Exception):
        Arm("vllm", {}, "x").column = "y"


def test_natural_and_many_documents_workloads_keep_their_names():
    assert tables.short("generate-quick-natural") == "generate-natural" and tables.short("pool-quick") == "pool"
    rows = [row("generate-natural", "vLLM cache on", 10.0, "vllm"), row("generate-natural", "batchinfer", 11.0),
            row("docqa", "vLLM cache on", 68.7, "vllm"), row("docqa", "batchinfer", 28.3)]
    assert matrix(rows)[0] == ["generate-natural", "docqa"]
