"""Benchmark harness tests: CPU only, no network, no transformers. Run: python -m pytest tests/"""
import hashlib
import json
from pathlib import Path

import pytest

import bench.run
from bench.report import load, table
from bench.results import gpu_slug, migrate, relink_references
from bench.run import prefix_match, run, score_choice, score_label, score_number
from bench.workload import PRESETS, build, parse_mix, unique_tokens, write

WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]


class StubTokenizer:
    """Whitespace tokens and a trivial chat template: enough for build and stats."""

    def __init__(self):
        self.ids = {}

    def encode(self, text, add_special_tokens=False):
        return [self.ids.setdefault(w, len(self.ids)) for w in text.split()]

    def get_vocab(self):
        return {w: i for i, w in enumerate(WORDS)}

    def decode(self, ids):
        return "".join(" " + WORDS[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return " ".join(f"<{m['role']}> {m['content']} </{m['role']}>" for m in messages) + " <assistant>"


class Recorder:
    """Backend that remembers what it was sent."""
    seen = []

    def __init__(self, model=None):
        pass

    def generate(self, requests):
        Recorder.seen += requests
        return [{"id": r["id"], "text": "x", "output_tokens": r["max_tokens"], "finish_reason": "length"} for r in requests]


class Overrun:
    """Backend with an off-by-one stop condition: one token past max_tokens."""

    def __init__(self, model=None):
        pass

    def generate(self, requests):
        return [{"id": r["id"], "text": "x", "output_tokens": r["max_tokens"] + 1, "finish_reason": "length"}
                for r in requests]


class Short:
    """Backend that ends every request one token early but still says "length"; a request capped at 1 gets nothing."""

    def __init__(self, model=None):
        pass

    def generate(self, requests):
        return [{"id": r["id"], "text": "", "output_tokens": r["max_tokens"] - 1, "finish_reason": "length",
                 "token_ids": [0] * (r["max_tokens"] - 1)} for r in requests]


class StopsAt:
    """Backend that stops every request after one token and says "stop"; the token is `token`."""
    token = 0

    def __init__(self, model=None):
        pass

    def generate(self, requests):
        return [{"id": r["id"], "text": "", "output_tokens": 1, "finish_reason": "stop", "token_ids": [self.token]}
                for r in requests]


class StopsOnStopId(StopsAt):
    token = 99


class Unserialisable:
    """Backend whose outputs are fine but whose details do not serialise, so writing details.json fails."""

    def __init__(self, model=None):
        pass

    def generate(self, requests):
        return [{"id": r["id"], "text": "x", "output_tokens": r["max_tokens"], "finish_reason": "length"}
                for r in requests]

    def details(self):
        return {"meta": {}, "ids": {1, 2}}  # a set: json.dumps refuses it


class Unloadable:
    """Backend that must never be constructed."""

    def __init__(self, model=None):
        raise AssertionError("backend loaded")


MIX = "synthetic:12:prompt_len=20:shared_frac=0.5:groups=3:output_len=16,synthetic:6:prompt_len=10:output_len=1"


def labelled(reference, labels, scorer, i):
    return {"id": f"lab-{i}", "source": "labelled", "kind": "classify", "group": None, "prompt": "q", "prompt_tokens": 1,
            "max_tokens": 1, "ignore_eos": False, "labels": labels, "reference": reference, "scorer": scorer}


def test_parse_mix():
    assert parse_mix("mmlu:10, synthetic:5:shared_frac=0.9:groups=2:output_len=1") == [
        ("mmlu", 10, {}), ("synthetic", 5, {"shared_frac": 0.9, "groups": 2, "output_len": 1})]
    with pytest.raises(ValueError):
        parse_mix("nope:3")
    for spec in PRESETS.values():
        parse_mix(spec)


def test_unique_tokens():
    seqs = [(1, 2, 3, 4), (1, 2, 3, 5), (9,)]
    assert unique_tokens(seqs) == 6
    assert unique_tokens(seqs, page=2) == 7


def test_build_is_deterministic_and_complete():
    tok = StubTokenizer()
    requests = build(parse_mix(MIX), tok, seed=0)
    assert len(requests) == 18 and len({r["id"] for r in requests}) == 18
    for r in requests:
        assert {"id", "source", "kind", "group", "prompt", "prompt_tokens", "max_tokens", "ignore_eos", "labels",
                "reference", "scorer"} <= r.keys()
        assert r["prompt_tokens"] == len(tok.encode(r["prompt"]))
    assert json.dumps(requests) == json.dumps(build(parse_mix(MIX), StubTokenizer(), seed=0))
    assert [r["id"] for r in requests] != [r["id"] for r in build(parse_mix(MIX), StubTokenizer(), seed=1)]


def test_natural_lengths_only_change_generate_rows():
    fixed = {r["id"]: r for r in build(parse_mix(MIX), StubTokenizer(), seed=0)}
    natural = {r["id"]: r for r in build(parse_mix(MIX), StubTokenizer(), seed=0, lengths="natural")}
    for rid, r in fixed.items():
        assert r["ignore_eos"] == (r["kind"] == "generate")
        assert natural[rid]["ignore_eos"] is False
        assert natural[rid]["max_tokens"] == r["max_tokens"]


def test_scorers():
    assert score_label(" Time.", "time") and not score_label("timer", "time")
    assert not score_label("calendar_update", "calendar") and score_label("credit limit", "credit_limit")
    assert score_choice("B)", "B") and not score_choice("The answer is B", "B")
    assert score_number("so 1,234 in total.\n#### 1,234", "1234") and not score_number("1234", "1234")


def test_prefix_match():
    assert prefix_match({"text": "", "token_ids": [1, 2, 3, 4]}, {"text": "", "token_ids": [1, 2, 9, 9]}) == 0.5
    assert prefix_match({"text": "abcd"}, {"text": "abcd"}) == 1.0
    # ignore_eos keeps decoding past the stop token; only the reply up to it counts
    a, b = {"text": "", "token_ids": [1, 2, 0, 7, 8]}, {"text": "", "token_ids": [1, 2, 0, 9, 9]}
    assert prefix_match(a, b) == 0.6 and prefix_match(a, b, stop={0}) == 1.0


@pytest.fixture
def workload(tmp_path):
    requests = build(parse_mix(MIX), StubTokenizer(), seed=0)
    requests += [labelled("time", ["time", "timer"], "label", 0), labelled("timer", ["time", "timer"], "label", 1)]
    path = tmp_path / "w.jsonl"
    write(path, requests, {"model": "stub", "target_s": 60, "stats": {"unique_prompt_tokens": 100}})
    return path, requests


def test_run_dummy(tmp_path, workload):
    path, requests = workload
    m = run(path, "dummy", out=tmp_path / "a")
    assert m["requests"] == len(requests)
    assert m["output_tokens"] == sum(r["max_tokens"] for r in requests)
    assert m["length_violations"] == 0 and not m["over_budget"]
    assert m["by_source"]["labelled"]["accuracy"] == 0.5  # dummy answers labels[0] = 'time'
    assert m["mfu_pct"] is m["mbu_pct"] is m["utilisation"] is None  # no model shape, no request trace
    again = run(path, "dummy", out=tmp_path / "b", compare_to=tmp_path / "a" / "outputs.jsonl")
    assert again["compare"]["classify"]["match"] == 1.0 and again["compare"]["generate"]["match"] == 1.0


def test_compare_refuses_other_workload_or_model(tmp_path, workload):
    path, requests = workload
    run(path, "dummy", out=tmp_path / "a")
    reference = tmp_path / "a" / "outputs.jsonl"
    with pytest.raises(ValueError, match="not this workload file with other"):
        run(path, "dummy", model="other", out=tmp_path / "b", compare_to=reference)
    rebuilt = tmp_path / "rebuilt.jsonl"
    write(rebuilt, requests[::-1], {"model": "stub"})  # same ids, different file
    with pytest.raises(ValueError, match="not this workload file"):
        run(rebuilt, "dummy", out=tmp_path / "c", compare_to=reference)


def test_run_rejects_missing_results(tmp_path, workload):
    with pytest.raises(RuntimeError, match="1 missing"):
        run(workload[0], "dummy", opts={"drop": 1}, out=tmp_path / "c")


def test_run_counts_length_overruns_on_every_row(tmp_path, workload):
    path, requests = workload
    assert run(path, "test_bench:Overrun", out=tmp_path / "e", warmup=False)["length_violations"] == len(requests)


def test_short_and_empty_length_outputs_are_violations_and_fail_the_run(tmp_path, workload):
    """finish_reason "length" means max_tokens were produced; an empty output is never valid. Before, a row with
    no tokens at all counted as fine unless it had ignore_eos."""
    path, requests = workload
    assert run(path, "test_bench:Short", out=tmp_path / "s", warmup=False)["length_violations"] == len(requests)
    assert (tmp_path / "s" / "metrics.json").exists()  # the row is written for inspection, then the exit says invalid
    args = [str(path), "--no-warmup", "--out"]
    assert bench.run.main([*args, str(tmp_path / "s2"), "--backend", "test_bench:Short"]) == 1
    assert bench.run.main([*args, str(tmp_path / "d"), "--backend", "dummy"]) == 0



STOP_ID = 99


@pytest.mark.parametrize("ids,reason,ignore_eos,bad", [
    ([5, STOP_ID], "stop", False, 0),  # ended on its first stop token
    ([5], "stop", False, 1),  # "stop" without a stop token: cut short
    ([STOP_ID, 5, 6, 7], "length", False, 1),  # decoded on past a stop
    ([5, 6, 7, STOP_ID], "length", False, 0),  # EOS sampled at the cap: vLLM checks max_tokens first
    ([STOP_ID, 5, 6, 7], "length", True, 0),  # ignore_eos: a stop id is just a token
    ([5, 6, 7, 8], "abort", False, 1),  # not a finish reason
])
def test_outputs_end_at_max_tokens_or_their_first_stop_token(ids, reason, ignore_eos, bad):
    req = {"id": "a", "max_tokens": len(ids) if reason != "stop" else 100, "ignore_eos": ignore_eos}
    res = {"id": "a", "output_tokens": len(ids), "token_ids": ids, "finish_reason": reason}
    assert bench.run.length_violations([req], {"a": res}, {STOP_ID}) == bad
    if reason != "abort":
        assert bench.run.length_violations([req], {"a": res}) == 0  # without the stop ids only the counts are checked


def test_rows_check_stops_against_the_workload_stop_ids(tmp_path, workload):
    path, requests = workload
    eos = [dict(r, ignore_eos=False) for r in requests]
    write(path, eos, {"model": "stub", "stop_token_ids": [STOP_ID]})
    assert run(path, "test_bench:StopsAt", out=tmp_path / "a", warmup=False)["length_violations"] == len(eos)
    assert run(path, "test_bench:StopsOnStopId", out=tmp_path / "b", warmup=False)["length_violations"] == 0


def test_a_row_torn_after_its_outputs_leaves_no_metrics(tmp_path, workload):
    """metrics.json marks a whole row: an attempt that fails after writing its outputs leaves none, and never the
    previous attempt's metrics or details beside this one's outputs (before, metrics.json was written first)."""
    path, requests = workload
    out = tmp_path / "row"
    out.mkdir()
    for name in ("metrics.json", "details.json", "outputs.jsonl"):
        (out / name).write_text("from the previous attempt\n")
    with pytest.raises(TypeError, match="not JSON serializable"):
        run(path, "test_bench:Unserialisable", out=out, warmup=False)
    assert not (out / "metrics.json").exists() and not (out / "details.json").exists()
    assert len((out / "outputs.jsonl").read_text().splitlines()) == len(requests)  # this attempt's

def test_rows_record_their_provenance(tmp_path, workload):
    m = run(workload[0], "dummy", out=tmp_path / "p", warmup=False)
    assert m["versions"]["python"] and set(m["versions"]) == {"python", "torch", "transformers", "vllm"}
    sidecar = workload[0].with_suffix(".meta.json")
    assert m["workload_meta_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert m["model_revision"] is None  # "stub" is no cached hub model; nothing is downloaded to find out
    assert m["engine"] == "dummy"
    assert m["git_commit_time"] and m["git_commit_time"][:4].isdigit()  # the tests run inside the repo: ISO 8601


def test_git_sha_is_dirty_for_code_not_for_rows(tmp_path, monkeypatch):
    """The repo tracks results/: a tree whose rows a run rewrote or removed still runs the committed code."""
    import subprocess

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=tmp_path, check=True,
                       capture_output=True)
    (tmp_path / "results" / "w").mkdir(parents=True)
    (tmp_path / "results" / "w" / "metrics.json").write_text("{}")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "c")
    monkeypatch.chdir(tmp_path)
    clean = bench.run.git_sha()
    (tmp_path / "results" / "w" / "metrics.json").write_text('{"rewritten": 1}')
    (tmp_path / "results" / "new").mkdir()
    assert bench.run.git_sha() == clean and not clean.endswith("-dirty")
    (tmp_path / "results" / "w" / "metrics.json").unlink()
    assert bench.run.git_sha() == clean
    (tmp_path / "code.py").write_text("x = 2\n")
    assert bench.run.git_sha() == clean + "-dirty"


def test_results_root_comes_from_the_environment(tmp_path):
    """BENCH_RESULTS_ROOT keeps a run's rows outside the clone, which tracks results/."""
    import os
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[1]
    read = [sys.executable, "-c", "from bench.results import ROOT; print(ROOT)"]
    env = {k: v for k, v in os.environ.items() if k != "BENCH_RESULTS_ROOT"}
    assert subprocess.run(read, cwd=repo, env=env, capture_output=True, text=True).stdout.strip() == "results"
    env["BENCH_RESULTS_ROOT"] = str(tmp_path)
    assert subprocess.run(read, cwd=repo, env=env, capture_output=True, text=True).stdout.strip() == str(tmp_path)


def test_a_registry_name_and_its_class_path_write_the_same_row():
    from bench.run import engine_name, row_label
    for name, spec in (("naive", "batchinfer.bench:NaiveBackend"), ("batchinfer", "batchinfer.bench:BatchinferBackend"),
                       ("vllm", "bench.backends:VLLM")):
        assert engine_name(name) == engine_name(spec) == name
        assert row_label(spec, {"order": "input", "num_blocks": 3750}) == f"{name}-order=input-num_blocks=3750"


@pytest.fixture
def on_gpu(monkeypatch):
    """Makes runs record the given nvidia-smi name as their GPU."""
    return lambda name: monkeypatch.setattr(bench.run, "gpu_name", lambda: name)


def test_default_results_path_includes_model_and_gpu(tmp_path, workload, monkeypatch, on_gpu):
    monkeypatch.chdir(tmp_path)
    on_gpu("NVIDIA H100 PCIe")
    assert run(workload[0], "dummy", model="org/Model-1B")["gpu"] == "NVIDIA H100 PCIe"
    assert (tmp_path / "results" / "w" / "Model-1B" / "H100-PCIe" / "dummy" / "metrics.json").exists()


def test_gpu_slug():
    assert gpu_slug("NVIDIA A100-SXM4-40GB") == "A100-SXM4-40GB"
    assert gpu_slug("NVIDIA H100 PCIe") == "H100-PCIe"
    assert gpu_slug("NVIDIA H200") == "H200"
    assert gpu_slug(None) == "cpu"


def test_branch_rows_land_beside_main_rows(tmp_path, workload, monkeypatch, on_gpu):
    on_gpu("NVIDIA H100 PCIe")
    monkeypatch.chdir(tmp_path)
    main = run(workload[0], "dummy")
    monkeypatch.setenv("BENCH_BRANCH", "feature/x")  # a run of a branch's code
    branch = run(workload[0], "dummy")
    assert main["branch"] is None and branch["branch"] == "feature/x"
    rows = sorted(p.parent.name for p in (tmp_path / "results/w/stub/H100-PCIe").glob("*/metrics.json"))
    assert rows == ["dummy", "dummy@feature-x"]  # never on top of main's row


def test_compare_label_takes_the_reference_from_this_gpu(tmp_path, workload, monkeypatch, on_gpu):
    monkeypatch.delenv("BENCH_REFERENCE_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    on_gpu("NVIDIA H100 PCIe")
    run(workload[0], "dummy")
    m = run(workload[0], "dummy", out=tmp_path / "b", compare_to="dummy")
    assert m["compare"]["against"] == str(Path("results/w/stub/H100-PCIe/dummy/outputs.jsonl"))
    assert m["compare"]["generate"]["match"] == 1.0
    on_gpu("NVIDIA A100-SXM4-40GB")  # no reference row on this GPU: refused before the backend loads
    with pytest.raises(ValueError, match="A100-SXM4-40GB/dummy"):
        run(workload[0], "test_bench:Unloadable", out=tmp_path / "c", compare_to="dummy")


def test_compare_label_resolves_under_bench_reference_root(tmp_path, workload, monkeypatch, on_gpu):
    on_gpu("NVIDIA H100 PCIe")
    monkeypatch.chdir(tmp_path)
    run(workload[0], "dummy")  # the reference, in ./results as in a clone
    (tmp_path / "ref").mkdir()
    monkeypatch.chdir(tmp_path / "ref")  # a checkout with no rows of its own
    monkeypatch.setenv("BENCH_REFERENCE_ROOT", str(tmp_path / "results"))
    m = run(workload[0], "dummy", compare_to="dummy")
    assert m["compare"]["generate"]["match"] == 1.0  # read from BENCH_REFERENCE_ROOT, recorded by its path in the
    assert m["compare"]["against"] == str(Path("results/w/stub/H100-PCIe/dummy/outputs.jsonl"))  # results tree


def test_a_row_under_bench_results_root_names_its_reference_in_the_results_tree(tmp_path, workload, monkeypatch,
                                                                              on_gpu):
    """Rows written outside the clone ($BENCH_RESULTS_ROOT, an absolute path) still name their reference by its
    path in the results tree as the repo holds it, never by this machine's directories."""
    on_gpu("NVIDIA H100 PCIe")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BENCH_REFERENCE_ROOT", raising=False)
    monkeypatch.setattr(bench.run, "ROOT", tmp_path / "rows-elsewhere")  # what BENCH_RESULTS_ROOT sets at import
    run(workload[0], "dummy")
    assert (tmp_path / "rows-elsewhere/w/stub/H100-PCIe/dummy/outputs.jsonl").exists()
    m = run(workload[0], "dummy", out=tmp_path / "b", compare_to="dummy")
    assert m["compare"]["generate"]["match"] == 1.0
    assert m["compare"]["against"] == str(Path("results/w/stub/H100-PCIe/dummy/outputs.jsonl"))


def test_compare_refuses_a_reference_from_another_gpu(tmp_path, workload, on_gpu):
    on_gpu("NVIDIA H100 PCIe")
    run(workload[0], "dummy", out=tmp_path / "a")
    on_gpu("NVIDIA A100-SXM4-40GB")
    with pytest.raises(ValueError, match="ran on NVIDIA H100 PCIe, not this run's NVIDIA A100-SXM4-40GB"):
        run(workload[0], "test_bench:Unloadable", out=tmp_path / "b", compare_to=tmp_path / "a" / "outputs.jsonl")


def test_report_shows_the_gpu(tmp_path, workload, on_gpu):
    on_gpu("NVIDIA H100 PCIe")
    header, _, row = table([run(workload[0], "dummy", out=tmp_path / "a")]).splitlines()
    assert header.split(" | ")[3] == "GPU" and row.split(" | ")[3] == "H100-PCIe"


def test_report_skips_metrics_bench_run_did_not_write(tmp_path, workload, capsys):
    run(workload[0], "dummy", out=tmp_path / "r" / "a")
    (tmp_path / "r" / "v0").mkdir()
    (tmp_path / "r" / "v0" / "metrics.json").write_text(json.dumps({"meta": {}, "memory": {}}))  # batchinfer CLI
    assert [m["backend"] for m in load([tmp_path / "r"])] == ["dummy"]
    assert "skipped 1 metrics.json" in capsys.readouterr().err


def test_migrate_moves_old_rows_under_their_gpu(tmp_path):
    def row(rel, outputs="", **metrics):
        (tmp_path / rel).mkdir(parents=True)
        (tmp_path / rel / "metrics.json").write_text(json.dumps(metrics))
        (tmp_path / rel / "outputs.jsonl").write_text(outputs)

    def rel(path):
        return str(path.relative_to(tmp_path))

    row("w/M/vllm", gpu="NVIDIA A100-SXM4-40GB")
    row("w/M/dummy", gpu=None)
    row("w/M/H100-PCIe/vllm", gpu="NVIDIA H100 PCIe")  # already in the new layout
    row("w/M/v0", memory={"gpu_name": "NVIDIA A100-SXM4-40GB"})  # batchinfer CLI metrics
    row("w/M/older", memory={})  # no GPU recorded
    row("w/step", gpu="NVIDIA H100 PCIe")  # a custom --out two levels deep
    row("x/M/H100-PCIe/vllm", gpu="NVIDIA H100 PCIe")  # migrated earlier ...
    row("x/M/vllm", gpu="NVIDIA H100 PCIe")  # ... and pulled back in the old layout, with a file the target lacks
    (tmp_path / "x/M/vllm/details.json").write_text("{}")
    row("y/M/H100-PCIe/vllm", outputs="a", gpu="NVIDIA H100 PCIe")
    row("y/M/vllm", outputs="b", gpu="NVIDIA H100 PCIe")  # same row name, different outputs
    moved, merged, kept = migrate(tmp_path)
    assert sorted(rel(t) for _, t in moved) == ["w/M/A100-SXM4-40GB/v0", "w/M/A100-SXM4-40GB/vllm", "w/M/cpu/dummy"]
    assert (tmp_path / "w/M/A100-SXM4-40GB/vllm/outputs.jsonl").exists() and not (tmp_path / "w/M/vllm").exists()
    assert [(rel(r), rel(t)) for r, t in merged] == [("x/M/vllm", "x/M/H100-PCIe/vllm")]
    assert (tmp_path / "x/M/H100-PCIe/vllm/details.json").exists() and not (tmp_path / "x/M/vllm").exists()
    assert [(rel(r), why.split()[-1]) for r, why in kept] == [("w/M/older", "recorded"), ("w/step", "hand"),
                                                            ("y/M/vllm", "outputs.jsonl")]
    assert migrate(tmp_path)[:2] == ([], [])  # idempotent


def test_relink_references_names_the_reference_by_its_results_tree_path(tmp_path):
    def row(rel, against):
        (tmp_path / rel).mkdir(parents=True)
        metrics = {"wall_s": 1.0, "compare": {"against": against, "generate": {}}}
        (tmp_path / rel / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    ref = "results/w/M/H100-PCIe/vllm/outputs.jsonl"
    row("w/M/H100-PCIe/a", "/srv/clone/results/w/M/H100-PCIe/vllm/outputs.jsonl")  # recorded before the change
    row("w/M/H100-PCIe/b", ref)  # already relative
    row("w/M/H100-PCIe/c", "/data/elsewhere/outputs.jsonl")  # an explicit path outside any results tree
    (tmp_path / "w/M/H100-PCIe/d").mkdir()
    (tmp_path / "w/M/H100-PCIe/d/metrics.json").write_text(json.dumps({"compare": None}))  # no --compare
    changed = relink_references(tmp_path)
    assert [(str(m.parent.name), new) for m, _, new in changed] == [("a", ref)]
    a = json.loads((tmp_path / "w/M/H100-PCIe/a/metrics.json").read_text())
    assert a == {"wall_s": 1.0, "compare": {"against": ref, "generate": {}}}  # nothing else changed
    assert json.loads((tmp_path / "w/M/H100-PCIe/c/metrics.json").read_text())["compare"]["against"].startswith("/data")
    assert relink_references(tmp_path) == []  # idempotent


def test_backend_sees_only_client_fields(tmp_path, workload):
    Recorder.seen = []
    run(workload[0], "test_bench:Recorder", out=tmp_path / "d", warmup=False)
    assert Recorder.seen and all(r.keys() == {"id", "prompt", "max_tokens", "ignore_eos", "labels"} for r in Recorder.seen)


def test_parse_value_reads_on_off_as_booleans():
    """--opt prefix_sharing=off must mean off: a string 'off' would be truthy."""
    from bench.workload import parse_value
    assert parse_value("on") is True and parse_value("OFF") is False and parse_value("yes") is True and parse_value("no") is False
    assert parse_value("true") is True and parse_value("false") is False
    assert parse_value("3") == 3 and parse_value("0.5") == 0.5 and parse_value("prefix_dfs") == "prefix_dfs"


# The vLLM backend's trace, against a stand-in for the vllm module ---------------------------------------------

def fake_vllm(monkeypatch, accepts_trace=True, stat_loggers=True, outputs=None, drop=()):
    """A vllm module whose LLM records its arguments, optionally refuses the trace ones, and whose generate() runs
    two engine steps as LLMEngine.step() does: fetch the step's outputs (stamped by the engine core 7 and 9 ms
    after the call), then let the logger manager record its stats, then return the results. drop: iteration
    details fields this vLLM does not have."""
    from types import SimpleNamespace as NS
    import time

    class LLM:
        def __init__(self, model, seed, **opts):
            if not accepts_trace and "enable_logging_iteration_details" in opts:
                raise TypeError("unexpected keyword argument 'enable_logging_iteration_details'")
            self.opts = opts
            self.stamps = []
            core = NS(get_output=lambda: NS(timestamp=self.stamps.pop(0)))
            self.llm_engine = NS(logger_manager=NS(stat_loggers=["vllm's own"]) if stat_loggers else None,
                                 engine_core=core)

        def generate(self, prompts, params, use_tqdm):
            now = time.monotonic()
            self.stamps = [now + 0.007, now + 0.009]
            for ctx, gen in ((30, 0), (0, 2)):
                self.llm_engine.engine_core.get_output()
                fields = dict(is_dummy=False, num_generation_requests=gen, num_ctx_tokens=ctx,
                              num_ctx_requests=2 if ctx else 0)
                details = NS(**{k: v for k, v in fields.items() if k not in drop})
                loggers = [lg for lg in self.llm_engine.logger_manager.stat_loggers if hasattr(lg, "record")] \
                    if stat_loggers else []
                for logger in loggers:
                    logger.record(NS(iteration_details=details, num_running_reqs=2, num_waiting_reqs=0,
                                     kv_cache_usage=0.015), None)
                    logger.record(NS(iteration_details=NS(is_dummy=True)), None)  # a data-parallel filler: no row
            return outputs

    monkeypatch.setitem(__import__("sys").modules, "vllm", NS(LLM=LLM, SamplingParams=lambda **kw: kw))


def vllm_outputs(t0_offset):
    from types import SimpleNamespace as NS
    import time
    now = time.monotonic() + t0_offset
    return [NS(outputs=[NS(text="A", token_ids=[5], finish_reason="length")], prompt_token_ids=[1, 2, 3],
               num_cached_tokens=2, metrics=NS(scheduled_ts=now, first_token_ts=now + 0.01, last_token_ts=now + 0.01)),
            NS(outputs=[NS(text="x y", token_ids=[6, 7], finish_reason="stop")], prompt_token_ids=[1, 2, 3, 4],
               num_cached_tokens=None, metrics=None)]  # vLLM gave no request metrics: times stay None


VLLM_REQUESTS = [{"id": "a", "prompt": "p", "max_tokens": 1, "ignore_eos": False, "labels": None},
                 {"id": "b", "prompt": "q", "max_tokens": 4, "ignore_eos": False, "labels": None}]


def test_vllm_trace_records_iterations_and_requests(monkeypatch):
    from bench.backends import VLLM
    from bench.charts import charts
    fake_vllm(monkeypatch, outputs=vllm_outputs(0.5))
    backend = VLLM("m")
    assert backend.llm.opts == {"disable_log_stats": False, "enable_logging_iteration_details": True}
    assert backend.llm.llm_engine.logger_manager.stat_loggers == [backend.log]  # vLLM's own loggers are replaced
    results = backend.generate(VLLM_REQUESTS)
    assert [r["output_tokens"] for r in results] == [1, 2] and backend.stats() == {"trace": True}
    d = backend.details()
    assert d["meta"] == {"engine": "vllm", "trace": True}
    steps = d["steps"]
    assert steps["prefill_tokens"] == [30, 0] and steps["decode_rows"] == [0, 2] and steps["prefill_rows"] == [2, 0]
    assert steps["kv_cache_usage_pct"] == [1.5, 1.5]
    assert steps["end_ms"] == pytest.approx([7, 9], abs=1)  # the engine core's timestamps, not when they arrived
    core = backend.llm.llm_engine.engine_core
    assert core.get_output.__name__ == "<lambda>"  # vLLM's client is left as it was once generate() returns
    trace = d["request_trace"]
    assert trace["analysis_kind"] == ["prefill_only", "generate"] and trace["prefix_hit_tokens"] == [2, 0]
    assert trace["prompt_len"] == [3, 4] and trace["finish_reason"] == ["length", "stop"]
    assert trace["admitted_ms"][0] == trace["prefill_start_ms"][0] > 400 and trace["finished_ms"][1] is None
    assert trace["first_token_ms"][0] - trace["admitted_ms"][0] == pytest.approx(10, abs=0.01)
    drawn, _, _ = charts(d)
    assert {"tokens_per_step", "in_flight", "kv", "prompt_len"} <= {c["id"] for c in drawn}


@pytest.mark.parametrize("accepts_trace, stat_loggers", [(False, True), (True, False)],
                         ids=["arguments-refused", "no-logger-manager"])
def test_vllm_trace_fails_soft(monkeypatch, capsys, accepts_trace, stat_loggers):
    """A vLLM without the hooks still produces the reference row, untraced, and says why."""
    from bench.backends import VLLM
    fake_vllm(monkeypatch, accepts_trace=accepts_trace, stat_loggers=stat_loggers, outputs=vllm_outputs(0.0))
    backend = VLLM("m")
    assert len(backend.generate(VLLM_REQUESTS)) == 2
    assert backend.stats() == {"trace": False}
    d = backend.details()
    assert d["meta"]["trace"] is False and d["meta"]["trace_error"] and "steps" not in d and "request_trace" not in d
    assert "WARN: vllm trace unavailable" in capsys.readouterr().err



def test_a_vllm_trace_that_breaks_mid_run_drops_the_trace_not_the_row(monkeypatch, capsys):
    """A vLLM whose iteration details lack a field the log reads: record() runs inside vLLM's engine loop, so before
    the AttributeError went up through generate() and the reference row was lost for a chart."""
    from bench.backends import VLLM
    fake_vllm(monkeypatch, outputs=vllm_outputs(0.0), drop=("num_ctx_tokens",))
    backend = VLLM("m")
    assert [r["output_tokens"] for r in backend.generate(VLLM_REQUESTS)] == [1, 2]
    assert backend.stats() == {"trace": False}
    d = backend.details()
    assert "num_ctx_tokens" in d["meta"]["trace_error"] and "steps" not in d and "request_trace" not in d
    assert "WARN: vllm trace unavailable" in capsys.readouterr().err
    assert len(backend.generate(VLLM_REQUESTS)) == 2  # and the next pass runs untraced

def test_vllm_trace_off_leaves_vllm_as_it_was(monkeypatch):
    from bench.backends import VLLM
    fake_vllm(monkeypatch, outputs=vllm_outputs(0.0))
    backend = VLLM("m", trace=False)
    assert backend.llm.opts == {} and backend.llm.llm_engine.logger_manager.stat_loggers == ["vllm's own"]
    backend.generate(VLLM_REQUESTS)
    assert backend.details() == {"meta": {"engine": "vllm", "trace": False}}


def test_iteration_timing_leaves_vllms_client_as_it_was():
    """vLLM's client has get_output as a class method: timing() must not leave an instance attribute behind, which
    would hold the client past generate() and change how vLLM shuts down."""
    from types import SimpleNamespace as NS

    from bench.backends import IterationLog

    class Client:
        def get_output(self):
            return NS(timestamp=5.0)

    client, log = Client(), IterationLog()
    with log.timing(client):
        client.get_output()
    assert log.core_ts == 5.0 and vars(client) == {}


def test_iteration_log_drops_an_iteration_from_before_this_call():
    """Async scheduling leaves one step's output queued after generate() returns; the next call's trace must not
    start with it (it would end before its own start)."""
    from types import SimpleNamespace as NS

    from bench.backends import IterationLog

    log = IterationLog()
    log.start(100.0)
    details = NS(is_dummy=False, num_generation_requests=1, num_ctx_tokens=0, num_ctx_requests=0)
    stats = NS(iteration_details=details, num_running_reqs=1, num_waiting_reqs=0, kv_cache_usage=0.01)
    log.core_ts = 99.99  # stamped by the engine core before this call's t0
    log.record(stats, None)
    log.core_ts = 100.004
    log.record(stats, None)
    assert [r["end_ms"] for r in log.rows] == [4.0]
