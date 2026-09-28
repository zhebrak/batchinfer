"""io: only the client-visible fields are read; results append and round-trip."""
import json
from pathlib import Path

import pytest

import bench.run
from batchinfer.io import CLIENT_FIELDS, ResultWriter, from_rows, read_requests
from batchinfer.schema import Request, Result

FIXTURE = Path(__file__).parent / "fixtures" / "smoke.jsonl"


def test_fixture_reads_only_client_fields():
    reqs = read_requests(FIXTURE)
    assert len(reqs) == 8 and all(isinstance(r, Request) for r in reqs)
    assert set(Request.__dataclass_fields__) == set(CLIENT_FIELDS)
    for r in reqs:
        assert not hasattr(r, "kind") and not hasattr(r, "reference") and not hasattr(r, "scorer")
    c1 = reqs[0]
    assert (c1.id, c1.max_tokens, c1.ignore_eos, c1.labels) == ("c-1", 1, False, ["positive", "negative"])
    assert reqs[4].labels is None and reqs[4].ignore_eos is True


def test_defaults_and_limit(tmp_path):
    p = tmp_path / "w.jsonl"
    p.write_text('{"prompt": "a"}\n\n{"prompt": "b", "max_tokens": 3}\n{"prompt": "c", "labels": []}\n')
    reqs = read_requests(p, default_max_tokens=7)
    assert [(r.id, r.max_tokens, r.ignore_eos, r.labels) for r in reqs] == [
        ("0", 7, False, None), ("2", 3, False, None), ("3", 7, False, None)]
    assert [r.id for r in read_requests(p, limit=2)] == ["0", "2"]


def test_rejects_missing_prompt_and_bad_max_tokens(tmp_path):
    p = tmp_path / "w.jsonl"
    p.write_text('{"id": 1}\n')
    with pytest.raises(ValueError, match="prompt"):
        read_requests(p)
    p.write_text('{"prompt": "x", "max_tokens": 0}\n')
    with pytest.raises(ValueError, match="max_tokens"):
        read_requests(p)


def test_result_writer_round_trip(tmp_path):
    out = tmp_path / "r" / "outputs.jsonl"
    res = Result(id="a", text="hi", token_ids=[5, 1], prompt_tokens=3, output_tokens=2,
                 finish_reason="stop", group=0, latency_s=0.01)
    with ResultWriter(out) as w:
        w(res)
        w(res)
    assert w.n == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows == [res.to_dict(), res.to_dict()]
    assert rows[0]["output_tokens"] == len(rows[0]["token_ids"])


def test_from_rows_matches_read_requests_and_the_bench_client_fields():
    """bench.run hands a backend rows in memory; they become the same Requests the JSONL reader makes."""
    assert CLIENT_FIELDS == bench.run.CLIENT_FIELDS  # the two edges agree on what a client sends
    rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    client = [{k: r.get(k) for k in CLIENT_FIELDS} for r in rows]
    assert from_rows(client) == read_requests(FIXTURE)
    reqs = from_rows([{"prompt": "a"}, {"prompt": "b", "id": 7, "labels": ["x", 1]}], default_max_tokens=5)
    assert [(r.id, r.max_tokens, r.labels) for r in reqs] == [("0", 5, None), ("7", 5, ["x", "1"])]
    with pytest.raises(ValueError, match="row 1: no 'prompt' field"):
        from_rows([{"prompt": "a"}, {"id": 1}])


@pytest.mark.parametrize("row,match", [
    ({"prompt": "x", "ignore_eos": "false"}, "ignore_eos must be true or false"),  # bool("false") is True
    ({"prompt": "x", "max_tokens": 2.7}, "max_tokens must be an int"),  # int(2.7) is 2
    ({"prompt": "x", "max_tokens": "5"}, "max_tokens must be an int"),
    ({"prompt": "x", "max_tokens": True}, "max_tokens must be an int"),  # True is an int in Python
    ({"prompt": "x", "labels": "positive"}, "labels must be a list"),  # list("positive") would be 8 labels
    ({"prompt": ["x"]}, "prompt must be a string"),
])
def test_rejects_values_it_would_otherwise_coerce(row, match):
    with pytest.raises(ValueError, match=f"row 0: {match}"):
        from_rows([row])


def test_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="rows: duplicate request id '7'"):
        from_rows([{"prompt": "a", "id": 7}, {"prompt": "b", "id": "7"}])
    p = tmp_path / "w.jsonl"
    p.write_text('{"prompt": "a", "id": "x"}\n{"prompt": "b", "id": "x"}\n')
    with pytest.raises(ValueError, match="w.jsonl: duplicate request id 'x'"):
        read_requests(p)
