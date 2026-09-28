"""bench.suite: load groups, one load per group, reset and configure per pass, baseline reuse, the idle wait. CPU only,
with fake engines that have the real ones' shape (the worker runs in-process here)."""
import json
import subprocess
import time

import pytest

from bench import suite
from bench.backends import Dummy
from bench.results import run_dir
from bench.run import versions
from test_bench import workload  # noqa: F401  (the fixture only; importing tests would collect them twice)


class Counting(Dummy):
    """batchinfer's shape: one load option, the rest policy options set per pass by configure(), reset() per pass."""
    name = "batchinfer"
    load_opts = ("delay_s",)
    loads, events = [], []

    def __init__(self, model=None, delay_s=0.0, **policy_opts):
        super().__init__(model, delay_s=delay_s)
        Counting.loads.append(delay_s)
        self.policy = self._policy_config(**policy_opts)

    @staticmethod
    def _policy_config(order="prefix_dfs", prefix_sharing=True, fail=False):
        return {"order": order, "prefix_sharing": prefix_sharing, "fail": fail}

    def configure(self, **policy_opts):
        self.policy = self._policy_config(**policy_opts)
        Counting.events.append(("configure", dict(self.policy)))

    def reset(self):
        Counting.events.append(("reset",))

    def generate(self, requests):
        if self.policy["fail"] and len(requests) > 1:  # the warmup passes; the timed pass does not
            raise RuntimeError("boom")
        Counting.events.append(("generate", len(requests)))
        return super().generate(requests)


class FakeVLLM(Dummy):
    """vLLM's shape: every option is a load option, reset() drops the prefix cache."""
    name = "vllm"
    resets = 0

    def __init__(self, model=None, enable_prefix_caching=True, enforce_eager=False, max_num_seqs=None):
        super().__init__(model)

    def reset(self):
        FakeVLLM.resets += 1


class FakeNaive(Dummy):
    name = "naive"


BACKENDS = {"vllm": "test_suite:FakeVLLM", "naive": "test_suite:FakeNaive", "batchinfer": "test_suite:Counting"}


@pytest.fixture
def here(tmp_path, monkeypatch):
    """Rows land under tmp_path/results, references resolve there, and the fakes start from zero."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BENCH_REFERENCE_ROOT", raising=False)
    Counting.loads, Counting.events, FakeVLLM.resets = [], [], 0
    return tmp_path


def run_all(rows, model="stub"):
    groups = suite.group_rows(rows)
    return groups, suite.run_groups(groups, model, None, worker=suite.work, idle=lambda: None)


def row_metrics(tmp_path, path, label):
    return json.loads((run_dir(path, "stub", None, label, tmp_path / "results") / "metrics.json").read_text())


def test_rows_that_share_load_options_share_one_load(here, workload):
    path, requests = workload
    rows = suite.rows_for([path], ["batchinfer"], [{"prefix_sharing": False}, {"order": "input"}], BACKENDS)
    groups, done = run_all(rows)
    assert len(groups) == 1 and [status for _, status, _ in done] == [0]
    assert Counting.loads == [0.0]  # one construction for three rows
    passes = [e for e in Counting.events if e[0] != "generate" or e[1] > 1]
    # every pass is configured from the defaults (never the previous pass's options), then reset, then timed
    assert passes == [("configure", {"order": "prefix_dfs", "prefix_sharing": True, "fail": False}), ("reset",),
                      ("generate", len(requests)),
                      ("configure", {"order": "prefix_dfs", "prefix_sharing": False, "fail": False}), ("reset",),
                      ("generate", len(requests)),
                      ("configure", {"order": "input", "prefix_sharing": True, "fail": False}), ("reset",),
                      ("generate", len(requests))]
    loads = {row_metrics(here, path, label)["load_s"]
             for label in ("batchinfer", "batchinfer-prefix_sharing=False", "batchinfer-order=input")}
    assert len(loads) == 1  # the rows of one load share its load_s


def test_a_load_option_splits_the_groups(here, workload):
    rows = suite.rows_for([workload[0]], ["batchinfer"], [{"delay_s": 0.001}, {"order": "input"}], BACKENDS)
    groups, _ = run_all(rows)
    assert [(g.load_opts, len(g.rows)) for g in groups] == [({}, 2), ({"delay_s": 0.001}, 1)]
    assert Counting.loads == [0.0, 0.001]


def test_the_vllm_row_runs_first_and_every_other_row_compares_against_it(here, workload):
    path, _ = workload
    rows = suite.rows_for([path], ["batchinfer", "naive", "vllm"], [], BACKENDS)
    groups, done = run_all(rows)
    assert [(suite.engine_name(g.backend), g.load_opts) for g in groups] == [
        ("vllm", {}), ("vllm", {"enable_prefix_caching": False}), ("vllm", {"enforce_eager": True}),
        ("vllm", {"max_num_seqs": 512}), ("naive", {}), ("batchinfer", {})]
    assert all(status == 0 for _, status, _ in done) and FakeVLLM.resets == 4  # one per pass
    assert row_metrics(here, path, "vllm")["compare"] is None
    for label in ("vllm-enable_prefix_caching=False", "vllm-enforce_eager=True", "vllm-max_num_seqs=512", "naive",
                  "batchinfer"):
        assert row_metrics(here, path, label)["compare"]["against"].endswith("/vllm/outputs.jsonl"), label


def test_without_a_vllm_row_a_row_runs_uncompared(here, workload, capsys):
    path, _ = workload
    run_all(suite.rows_for([path], ["batchinfer"], [], BACKENDS))
    assert row_metrics(here, path, "batchinfer")["compare"] is None
    assert "no vllm row on this GPU yet" in capsys.readouterr().out



def test_a_reference_without_metrics_is_not_a_reference(here, workload, capsys):
    """Outputs alone are a torn row: the row runs uncompared rather than failing in load_reference."""
    path, _ = workload
    ref = run_dir(path, "stub", None, "vllm", here / "results")
    ref.mkdir(parents=True)
    (ref / "outputs.jsonl").write_text("")
    _, done = run_all(suite.rows_for([path], ["batchinfer"], [], BACKENDS))
    assert [status for _, status, _ in done] == [0] and row_metrics(here, path, "batchinfer")["compare"] is None
    assert "no vllm row on this GPU yet" in capsys.readouterr().out


def test_the_summary_leaves_out_a_row_this_run_failed_to_rewrite(here, workload):
    """A row the suite set out to run whose metrics.json predates the suite is an earlier run's (this attempt
    failed), so it is named, not shown as this run's number."""
    path, _ = workload
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rows = suite.rows_for([path], ["batchinfer"], [{"order": "input"}], BACKENDS)
    run_all(rows)
    ran = {run_dir(r.workload, "stub", None, suite.row_label(r.backend, r.opts)) for r in rows}
    stale = run_dir(path, "stub", None, "batchinfer-order=input") / "metrics.json"
    stale.write_text(json.dumps({**json.loads(stale.read_text()), "timestamp": "2026-09-01T00:00:00+0000"}))
    text = suite.summary([], 0.0, "stub", None, [path], started, ran)
    assert f"FAILED this run: {stale.parent}; the row on disk ran 2026-09-01T00:00:00+0000" in text
    table = text.split("| engine |")[1]  # the order column reads "input" only on the stale row
    assert table.count("\n| batchinfer |") == 1 and "| input |" not in table
    assert "| input |" in suite.summary([], 0.0, "stub", None, [path])  # without started: every row, as before

def test_a_failing_row_is_reported_and_the_next_row_still_runs(here, workload, capsys):
    path, _ = workload
    rows = suite.rows_for([path], ["batchinfer"], [{"fail": True}, {"order": "input"}], BACKENDS)
    _, done = run_all(rows)
    assert [status for _, status, _ in done] == [1]
    assert row_metrics(here, path, "batchinfer-order=input")["requests"] > 0
    assert "FAILED: " in capsys.readouterr().out and not (here / "results" / "w" / "stub" / "cpu" /
                                                           "batchinfer-fail=True").exists()


def test_defaults_only_is_one_row_per_workload_and_naive_is_opt_in(workload):
    rows = suite.rows_for([workload[0], workload[0]], ["batchinfer"], [], BACKENDS)
    assert [r.opts for r in rows] == [{}, {}]
    assert "naive" not in suite.DEFAULT_ENGINES and set(suite.DEFAULT_ENGINES) < set(suite.ENGINES)


def test_dry_run_prints_the_plan_and_runs_nothing(here, workload, capsys):
    path, _ = workload
    assert suite.main(["--dry-run", "--workloads", str(path), "--model", "stub", "--variant", "order=input"]) == 0
    out = capsys.readouterr().out
    assert "batchinfer-order=input" in out and "vllm-enable_prefix_caching=False" in out
    assert "5 load groups" in out and not (here / "results").exists()


def test_bad_arguments_exit_before_anything_runs(here, workload):
    path = str(workload[0])
    for argv in (["--engines", "gpu"], ["--variant", "orderinput"], ["--variant", "nope=1"],
                 ["--defaults-only", "--variant", "order=input"], ["--workloads", "missing.jsonl"]):
        with pytest.raises(SystemExit):
            suite.main(["--dry-run", "--workloads", path, *argv])


# baseline reuse ------------------------------------------------------------------------------------

INSTALLED = {"vllm": "0.30.0", "torch": "2.13.0", "transformers": "5.17.0"}


def baseline(tmp_path, path, backend, opts=None, **fields):
    """A baseline row on disk as bench.run would write it, with fields overriding the valid defaults."""
    row = suite.Row(str(path), backend, opts or {})
    d = run_dir(path, "stub", None, suite.row_label(backend, row.opts), tmp_path / "results")
    d.mkdir(parents=True)
    m = {"engine": suite.engine_name(backend), "workload_sha256": suite.row_sha(path), "model": "stub",
         "length_violations": 0, "versions": dict(INSTALLED), "git_sha": "abc1234",
         "git_commit_time": "2026-09-27T16:24:00+00:00", "timestamp": "2026-09-27T16:30:00+0000", **fields}
    (d / "metrics.json").write_text(json.dumps(m))
    (d / "outputs.jsonl").write_text("")
    return row


def decide(tmp_path, row, changed=lambda sha, paths: [], installed=INSTALLED, revision=None):
    return suite.reuse(row, "stub", None, tmp_path / "results", installed, changed, revision)


def changes(*files):
    """A fake git: these files changed since the row's commit, whichever paths are asked about."""
    return lambda sha, paths: [f for f in files if f in paths]


@pytest.mark.parametrize("backend", ["test_suite:FakeVLLM", "test_suite:FakeNaive"])
def test_a_valid_baseline_is_reused(tmp_path, workload, backend):
    ok, why = decide(tmp_path, baseline(tmp_path, workload[0], backend))
    assert ok and "abc1234" in why and "2026-09-27T16:24" in why


@pytest.mark.parametrize("fields,reason", [
    ({"engine": None}, "before rows recorded their engine"),
    ({"workload_sha256": "0" * 64}, "another workload file"),
    ({"model": "other"}, "model other"),
    ({"length_violations": 2}, "2 length violations"),
    ({"versions": {**INSTALLED, "vllm": "0.29.0"}}, "vllm 0.29.0 -> 0.30.0"),
    ({"versions": {**INSTALLED, "torch": "2.12.0"}}, "torch 2.12.0 -> 2.13.0"),
    ({"model_revision": "0123456789abcdef"}, "model snapshot 01234567 -> fedcba98"),
])
def test_a_stale_vllm_row_is_rerun(tmp_path, workload, fields, reason):
    ok, why = decide(tmp_path, baseline(tmp_path, workload[0], "test_suite:FakeVLLM", **fields),
                     revision="fedcba9876543210")
    assert not ok and reason in why


def test_a_baseline_without_outputs_is_rerun(tmp_path, workload):
    """Its outputs are every other row's --compare reference; metrics alone do not make a row."""
    row = baseline(tmp_path, workload[0], "test_suite:FakeVLLM")
    (run_dir(workload[0], "stub", None, "vllm", tmp_path / "results") / "outputs.jsonl").unlink()
    assert decide(tmp_path, row) == (False, "no outputs.jsonl")


def test_an_unknown_snapshot_on_either_side_is_no_reason_to_rerun(tmp_path, workload):
    row = baseline(tmp_path, workload[0], "test_suite:FakeVLLM", model_revision="0123456789abcdef")
    assert decide(tmp_path, row, revision=None)[0] and decide(tmp_path, row, revision="0123456789abcdef")[0]


def test_a_baseline_reruns_when_its_own_engine_code_changed(tmp_path, workload):
    vllm = baseline(tmp_path, workload[0], "test_suite:FakeVLLM")
    ok, why = decide(tmp_path, vllm, changed=changes("bench/backends.py"))
    assert not ok and "vllm's own code changed since abc1234: bench/backends.py" in why
    naive = baseline(tmp_path, workload[0], "test_suite:FakeNaive")
    assert not decide(tmp_path, naive, changed=changes("batchinfer/naive.py"))[0]
    assert not decide(tmp_path, naive, changed=changes("batchinfer/model.py"))[0]
    assert decide(tmp_path, naive, changed=changes("bench/backends.py"))[0]  # vLLM's adapter is not naive's code
    assert not decide(tmp_path, naive, changed=lambda sha, paths: None)[0]  # what ran cannot be told
    assert not decide(tmp_path, naive, installed={**INSTALLED, "transformers": "5.18.0"})[0]
    assert decide(tmp_path, naive, installed={**INSTALLED, "vllm": "0.31.0"})[0]  # vLLM is not in the naive path


def test_shared_code_changes_are_reported_but_keep_the_baseline(tmp_path, workload):
    """Baselines run once and are reused. The harness and the pipeline before the engine change with nearly every
    batchinfer commit, so they are named in the decision instead of forcing a rerun."""
    naive = baseline(tmp_path, workload[0], "test_suite:FakeNaive")
    ok, why = decide(tmp_path, naive, changed=changes("batchinfer/policy.py", "bench/run.py"))
    assert ok and "shared code changed since: batchinfer/policy.py, bench/run.py (--baselines rerun" in why
    vllm = baseline(tmp_path, workload[0], "test_suite:FakeVLLM")
    ok, why = decide(tmp_path, vllm, changed=changes("batchinfer/policy.py"))
    assert ok and "shared code" not in why  # the batchinfer pipeline is not vLLM's


def test_plan_drops_reusable_baselines_unless_rerun(tmp_path, workload):
    path = workload[0]
    baseline(tmp_path, path, "test_suite:FakeVLLM")
    rows = suite.rows_for([path], ["vllm", "batchinfer"], [], BACKENDS)
    same = lambda sha, paths: []  # noqa: E731  (nothing changed since the row's commit)
    groups, decisions = suite.plan(rows, "stub", None, "missing", tmp_path / "results", INSTALLED, same)
    # vllm on reused; off, eager and max_num_seqs=512 missing; batchinfer always runs
    assert [what for _, what, _ in decisions] == ["reuse", "run", "run", "run", "run"]
    assert sum(len(g.rows) for g in groups) == 4
    groups, decisions = suite.plan(rows, "stub", None, "rerun", tmp_path / "results", INSTALLED, same)
    assert [what for _, what, _ in decisions] == ["run"] * 5


def test_changed_files_follows_git(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    git = lambda *a: subprocess.run(["git", *a], check=True, capture_output=True, text=True).stdout.strip()  # noqa
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "naive.py").write_text("a\n")
    (tmp_path / "other.py").write_text("a\n")
    git("add", ".")
    git("commit", "-qm", "one")
    sha = git("rev-parse", "--short", "HEAD")
    assert suite.changed_files(sha, ("naive.py",)) == []
    (tmp_path / "other.py").write_text("b\n")  # another file: not asked about
    assert suite.changed_files(sha, ("naive.py",)) == []
    (tmp_path / "naive.py").write_text("b\n")  # uncommitted, still a change to what would run
    assert suite.changed_files(sha, ("naive.py", "other.py")) == ["naive.py", "other.py"]
    for unknown in (f"{sha}-dirty", "fffffff", None):
        assert suite.changed_files(unknown, ("naive.py",)) is None


# the idle wait -------------------------------------------------------------------------------------

def test_the_next_group_waits_for_two_idle_checks():
    states = iter([(9000, 1), (50, 0), (60, 1), (10, 0), (10, 0), (0, 0)])
    sleeps = []
    suite.wait_for_idle_gpu(state=lambda: next(states), sleep=sleeps.append, clock=lambda: 0.0)
    assert len(sleeps) == 4  # busy, idle, busy, idle, idle: returns on the second idle in a row


def test_the_idle_wait_gives_up_with_a_reason():
    t = [0.0]
    with pytest.raises(RuntimeError, match="GPU still busy after 120 s .9000 MiB used, 1 compute"):
        suite.wait_for_idle_gpu(state=lambda: (9000, 1), sleep=lambda s: t.__setitem__(0, t[0] + s),
                                clock=lambda: t[0])


def test_no_gpu_means_no_wait():
    suite.wait_for_idle_gpu(state=lambda: None, sleep=pytest.fail, clock=lambda: 0.0)


# the batchinfer adapters' options ------------------------------------------------------------------

def test_configure_starts_from_the_defaults_and_validates():
    from batchinfer.bench import BatchinferBackend, NaiveBackend
    b = object.__new__(BatchinferBackend)  # the policy half only: no model
    b.configure(order="input", batch_size=32, max_batch_tokens=4096)
    assert (b.cfg.order, b.cfg.max_batch_size, b.cfg.max_batch_tokens) == ("input", 32, 4096)
    b.configure(max_batch_tokens=4096)
    assert b.cfg.order == "prefix_dfs" and b.cfg.max_batch_size == 64  # nothing carried over from the last pass
    with pytest.raises(ValueError, match="prefix_sharing must be a bool"):
        b.configure(prefix_sharing="off", max_batch_tokens=4096)
    with pytest.raises(TypeError):
        b.configure(num_blocks=100, max_batch_tokens=4096)  # a load option is not a policy option
    n = object.__new__(NaiveBackend)
    n.configure(max_batch_tokens=4096)
    assert (n.cfg.engine, n.cfg.order, n.cfg.admission, n.cfg.prefix_sharing) == ("naive", "input", "groups", False)


def test_configure_resolves_auto_as_the_constructor_does():
    """Both keep max_batch_tokens="auto" in the config: policy resolves it from what the engine measured on its card
    (engine.hardware), so a later pass's "auto" means the same as the load's."""
    from batchinfer.bench import BatchinferBackend, NaiveBackend
    for backend in (BatchinferBackend, NaiveBackend):
        b = object.__new__(backend)
        b.configure(max_batch_tokens="auto")
        assert b.cfg.max_batch_tokens == "auto" == backend._policy_config().max_batch_tokens


def test_installed_versions_are_read_without_importing():
    assert set(versions()) >= {"torch", "transformers", "vllm"}
