"""cli: --config fills the policy flags the command line left unset, a flag beats the file, and a bad file is refused
before anything loads. Through `analyze`, which needs only a tokenizer (the stub here)."""
import json
import re
from pathlib import Path

import pytest

from batchinfer import cli
from stub import StubTokenizer

FIXTURE = Path(__file__).parent / "fixtures" / "smoke.jsonl"


@pytest.fixture(autouse=True)
def stub_tokenizer(monkeypatch):
    monkeypatch.setattr(cli, "load_tokenizer", lambda model: StubTokenizer())


def config(tmp_path, fields):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(fields))
    return str(path)


def decisions(capsys, *argv):
    """What analyze prints after 'policy (decisions):'."""
    cli.main(["analyze", "--input", str(FIXTURE), *argv])
    return capsys.readouterr().out.split("policy (decisions):\n", 1)[1]


def test_the_file_sets_what_the_flags_leave_unset(tmp_path, capsys):
    path = config(tmp_path, {"order": "input", "sharing": "none", "prefill_budget": "auto"})
    text = decisions(capsys, "--config", path)
    assert "order=input" in text and "sharing=none" in text and "(auto)" in text
    text = decisions(capsys, "--config", path, "--order", "max_tokens_desc", "--sharing", "block")  # flags beat it
    assert "order=max_tokens_desc" in text and "sharing=block" in text and "(auto)" in text


def test_the_file_can_choose_the_engine_and_uses_policy_config_names(tmp_path, capsys):
    text = decisions(capsys, "--config", config(tmp_path, {"engine": "naive", "max_batch_size": 2}))
    assert text.startswith("policy (naive engine): order=input")  # the naive engine's own defaults
    assert "tokens x 2 rows" in text  # max_batch_size is the --batch-size flag


@pytest.mark.parametrize("fields,match", [
    ({"order": "input", "budget": 64}, "expected a JSON object of PolicyConfig fields .* not \\['budget'\\]"),
    (["prefix_dfs"], "expected a JSON object"),
    ({"chunk_prefill": "yes"}, "chunk_prefill must be a bool"),  # values are validated, never coerced
    ({"engine": "naive", "prefill_budget": 512}, "a fixed prefill_budget and chunk_prefill are batchinfer-engine settings"),
])
def test_a_bad_file_is_refused_before_anything_loads(tmp_path, capsys, fields, match):
    with pytest.raises(SystemExit):
        cli.main(["analyze", "--input", str(FIXTURE), "--config", config(tmp_path, fields)])
    assert re.search(match, capsys.readouterr().err)


def test_a_missing_file_is_an_argument_error(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["analyze", "--input", str(FIXTURE), "--config", str(tmp_path / "nope.json")])
    assert "nope.json" in capsys.readouterr().err


def test_resume_runs_only_what_is_left_and_loads_nothing_when_nothing_is(tmp_path, capsys):
    """--resume with an output that already holds every request's result exits before any model loads (this test
    has no model to load); a partial output lists what is left."""
    from batchinfer.io import read_requests
    ids = [r.id for r in read_requests(FIXTURE)]
    out = tmp_path / "outputs.jsonl"
    out.write_text("".join(json.dumps({"id": i, "text": "", "token_ids": [], "prompt_tokens": 1, "output_tokens": 0,
                                       "finish_reason": "length", "group": None, "latency_s": 0.0}) + "\n" for i in ids))
    cli.main(["run", "--input", str(FIXTURE), "--output", str(out), "--model", "no-such-model", "--resume"])
    assert f"resumed: {len(ids)} done, 0 to go" in capsys.readouterr().err
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == ids  # untouched
    with pytest.raises(Exception):  # half done: the rest would run, which needs the model this test does not have
        out.write_text(out.read_text().splitlines(keepends=True)[0])
        cli.main(["run", "--input", str(FIXTURE), "--output", str(out), "--model", "no-such-model", "--resume"])
    assert f"resumed: 1 done, {len(ids) - 1} to go" in capsys.readouterr().err
