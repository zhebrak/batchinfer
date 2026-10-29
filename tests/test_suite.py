"""bench.suite: load groups, one load per group, reset and configure per pass, baseline reuse, the idle wait. CPU only,
with fake engines that have the real ones' shape (the worker runs in-process here)."""
import json
import subprocess
import time
from pathlib import Path

import pytest

from bench import suite
from bench.backends import Dummy
import bench.workload
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
    def _policy_config(order="prefix_dfs", sharing="block", fail=False):
        return {"order": order, "sharing": sharing, "fail": fail}

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


def arms(*variants):
    """batchinfer's defaults plus the given variants on every workload (bench.suite --variant), no other arm."""
    return suite.select_arms(defaults_only=not variants, variants=list(variants))


def run_all(rows, model="stub"):
    groups = suite.group_rows(rows)
    return groups, suite.run_groups(groups, model, None, worker=suite.work, idle=lambda: None)


def row_metrics(tmp_path, path, label):
    return json.loads((run_dir(path, "stub", None, label, tmp_path / "results") / "metrics.json").read_text())


def test_rows_that_share_load_options_share_one_load(here, workload):
    path, requests = workload
    rows = suite.rows_for([path], ["batchinfer"], arms({"sharing": "none"}, {"order": "input"}), BACKENDS)
    groups, done = run_all(rows)
    assert len(groups) == 1 and [status for _, status, _ in done] == [0]
    assert Counting.loads == [0.0]  # one construction for three rows
    passes = [e for e in Counting.events if e[0] != "generate" or e[1] > 1]
    # every pass is configured from the defaults (never the previous pass's options), then reset, then timed
    assert passes == [("configure", {"order": "prefix_dfs", "sharing": "block", "fail": False}), ("reset",),
                      ("generate", len(requests)),
                      ("configure", {"order": "prefix_dfs", "sharing": "none", "fail": False}), ("reset",),
                      ("generate", len(requests)),
                      ("configure", {"order": "input", "sharing": "block", "fail": False}), ("reset",),
                      ("generate", len(requests))]
    loads = {row_metrics(here, path, label)["load_s"]
             for label in ("batchinfer", "batchinfer-sharing=none", "batchinfer-order=input")}
    assert len(loads) == 1  # the rows of one load share its load_s


def test_a_load_option_splits_the_groups(here, workload):
    rows = suite.rows_for([workload[0]], ["batchinfer"], arms({"delay_s": 0.001}, {"order": "input"}), BACKENDS)
    groups, _ = run_all(rows)
    assert [(g.load_opts, len(g.rows)) for g in groups] == [({}, 2), ({"delay_s": 0.001}, 1)]
    assert Counting.loads == [0.0, 0.001]


def test_the_vllm_row_runs_first_and_every_other_row_compares_against_it(here, workload):
    path, _ = workload
    every = (suite.Arm("vllm", {}, "vLLM cache on"), suite.Arm("vllm", {"enable_prefix_caching": False}, "off"),
             suite.Arm("vllm", {"enforce_eager": True}, "eager"), suite.Arm("naive", {}, "naive"),
             suite.Arm("batchinfer", {}, "batchinfer"))
    rows = suite.rows_for([path], ["batchinfer", "naive", "vllm"], every, BACKENDS)
    groups, done = run_all(rows)
    assert [(suite.engine_name(g.backend), g.load_opts) for g in groups] == [
        ("vllm", {}), ("vllm", {"enable_prefix_caching": False}), ("vllm", {"enforce_eager": True}),
        ("naive", {}), ("batchinfer", {})]
    assert all(status == 0 for _, status, _ in done) and FakeVLLM.resets == 3  # one per pass
    assert row_metrics(here, path, "vllm")["compare"] is None
    for label in ("vllm-enable_prefix_caching=False", "vllm-enforce_eager=True", "naive", "batchinfer"):
        assert row_metrics(here, path, label)["compare"]["against"].endswith("/vllm/outputs.jsonl"), label
    assert row_metrics(here, path, "vllm")["column"] == "vLLM cache on"  # the arm's column, for bench.tables


def test_without_a_vllm_row_a_row_runs_uncompared(here, workload, capsys):
    path, _ = workload
    run_all(suite.rows_for([path], ["batchinfer"], arms(), BACKENDS))
    assert row_metrics(here, path, "batchinfer")["compare"] is None
    assert "no vllm row on this GPU yet" in capsys.readouterr().out



def test_a_reference_without_metrics_is_not_a_reference(here, workload, capsys):
    """Outputs alone are a torn row: the row runs uncompared rather than failing in load_reference."""
    path, _ = workload
    ref = run_dir(path, "stub", None, "vllm", here / "results")
    ref.mkdir(parents=True)
    (ref / "outputs.jsonl").write_text("")
    _, done = run_all(suite.rows_for([path], ["batchinfer"], arms(), BACKENDS))
    assert [status for _, status, _ in done] == [0] and row_metrics(here, path, "batchinfer")["compare"] is None
    assert "no vllm row on this GPU yet" in capsys.readouterr().out


def test_the_summary_leaves_out_a_row_this_run_failed_to_rewrite(here, workload):
    """A row the suite set out to run whose metrics.json predates the suite is an earlier run's (this attempt
    failed), so it is named, not shown as this run's number."""
    path, _ = workload
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rows = suite.rows_for([path], ["batchinfer"], arms({"order": "input"}), BACKENDS)
    run_all(rows)
    ran = {run_dir(r.workload, "stub", None, suite.row_label(r.backend, r.opts)) for r in rows}
    stale = run_dir(path, "stub", None, "batchinfer-order=input") / "metrics.json"
    stale.write_text(json.dumps({**json.loads(stale.read_text()), "timestamp": "2026-09-01T00:00:00+0000"}))
    text = suite.summary([], 0.0, "stub", None, [path], started, ran)
    assert f"FAILED this run: {stale.parent}; the row on disk ran 2026-09-01T00:00:00+0000" in text
    table = text.split("| engine |")[1]  # the order column reads "input" only on the stale row
    assert table.count("\n| batchinfer |") == 1 and "| input |" not in table
    assert "| input |" in suite.summary([], 0.0, "stub", None, [path])  # without started: every row

def test_a_failing_row_is_reported_and_the_next_row_still_runs(here, workload, capsys):
    path, _ = workload
    rows = suite.rows_for([path], ["batchinfer"], arms({"fail": True}, {"order": "input"}), BACKENDS)
    _, done = run_all(rows)
    assert [status for _, status, _ in done] == [1]
    assert row_metrics(here, path, "batchinfer-order=input")["requests"] > 0
    assert "FAILED: " in capsys.readouterr().out and not (here / "results" / "w" / "stub" / "cpu" /
                                                           "batchinfer-fail=True").exists()


def test_defaults_only_is_one_row_per_workload_and_naive_is_opt_in(workload):
    rows = suite.rows_for([workload[0], workload[0]], ["batchinfer"], arms(), BACKENDS)
    assert [r.opts for r in rows] == [{}, {}]
    assert "naive" not in suite.DEFAULT_ENGINES and set(suite.DEFAULT_ENGINES) < set(suite.ENGINES)


def test_dry_run_prints_the_plan_and_runs_nothing(here, workload, capsys):
    path, _ = workload
    assert suite.main(["--dry-run", "--workloads", str(path), "--model", "stub", "--variant", "order=input"]) == 0
    out = capsys.readouterr().out
    assert "batchinfer-order=input" in out and "vllm-enable_prefix_caching=False" in out
    assert "3 load groups" in out and not (here / "results").exists()  # vLLM on and off, batchinfer


def test_once_runs_the_naive_engine_too(here, workload, tmp_path, capsys):
    """--once is the first of three runs, the one with the once-only arms; naive is one of them, so its engine is on."""
    path = tmp_path / "mixed-quick.jsonl"  # a workload naive's arm names (no capped arm, so no sidecar stats needed)
    path.write_bytes(workload[0].read_bytes())
    assert suite.main(["--dry-run", "--workloads", str(path), "--model", "stub"]) == 0
    assert "naive" not in capsys.readouterr().out
    assert suite.main(["--dry-run", "--workloads", str(path), "--model", "stub", "--once"]) == 0
    assert "naive" in capsys.readouterr().out
    assert suite.main(["--dry-run", "--workloads", str(path), "--model", "stub", "--once", "--engines", "vllm"]) == 0
    assert "naive" not in capsys.readouterr().out  # an explicit --engines still wins


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
    rows = suite.rows_for([path], ["vllm", "batchinfer"], suite.select_arms(), BACKENDS)
    same = lambda sha, paths: []  # noqa: E731  (nothing changed since the row's commit)
    groups, decisions = suite.plan(rows, "stub", None, "missing", tmp_path / "results", INSTALLED, same)
    # vllm on reused; off missing; batchinfer's defaults and no-trie rows always run
    assert [what for _, what, _ in decisions] == ["reuse", "run", "run", "run"]
    assert sum(len(g.rows) for g in groups) == 3
    groups, decisions = suite.plan(rows, "stub", None, "rerun", tmp_path / "results", INSTALLED, same)
    assert [what for _, what, _ in decisions] == ["run"] * 4


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
    with pytest.raises(ValueError, match="sharing must be one of"):
        b.configure(sharing="off", max_batch_tokens=4096)
    with pytest.raises(TypeError):
        b.configure(num_blocks=100, max_batch_tokens=4096)  # a load option is not a policy option
    n = object.__new__(NaiveBackend)
    n.configure(max_batch_tokens=4096)
    assert (n.cfg.engine, n.cfg.order, n.cfg.admission, n.cfg.sharing) == ("naive", "input", "groups", "none")


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


def test_arms_name_their_workloads_and_the_once_only_ones_wait_for_once(workload):
    """An arm runs on the workloads it names (or every workload); once-only arms join with --once; a workload
    file's name drops the size and keeps -natural."""
    mixed, natural, other = Path("workloads/mixed-quick.jsonl"), Path("workloads/generate-quick-natural.jsonl"), workload[0]
    uncapped = [a for a in suite.select_arms() if a.cap is None]  # capped arms read the sidecar (tested below)
    rows = suite.rows_for([mixed, natural, other], ["batchinfer", "vllm", "naive"], uncapped, BACKENDS)
    assert "naive" not in {suite.engine_name(r.backend) for r in rows}  # once-only
    assert [r.column for r in rows if r.workload == str(other)] == ["vLLM cache on", "vLLM cache off", "batchinfer",
                                                                   "no trie"]
    with_once = suite.rows_for([mixed], ["naive"], suite.select_arms(once=True), BACKENDS)
    assert [r.column for r in with_once] == ["naive"] and with_once[0].opts["order"] == "max_tokens_desc"
    assert suite.workload_path("generate-natural") == natural and suite.workload_path("mixed") == mixed
    assert bench.workload.workload_name("generate-quick-natural", "quick") == "generate-natural"


def test_a_capped_arm_gives_both_engines_the_same_token_capacity(tmp_path):
    """cap x the sidecar's KV demand, as batchinfer's num_blocks (one more for the pad block) and vLLM's
    kv_cache_memory_bytes at the model's bytes per token."""
    path = tmp_path / "pool-quick.jsonl"
    path.write_text("")
    stats = {"kv_demand_tokens": 400_000, "by_source": {"a": {"prompt_tokens": {"max": 8028}, "max_tokens": {"max": 1}},
                                                       "b": {"prompt_tokens": {"max": 2037}, "max_tokens": {"max": 512}}}}
    bench.workload.meta_path(path).write_text(json.dumps({"stats": stats}))
    arms_ = (suite.Arm("vllm", {}, "vLLM capped", ("pool",), cap=0.25),
             suite.Arm("batchinfer", {}, "capped", ("pool",), cap=0.25),
             suite.Arm("batchinfer", {"block_size": 64}, "capped 64", ("pool",), cap=0.25),
             suite.Arm("vllm", {"kv_cache_dtype": "fp8"}, "vLLM capped fp8 KV", ("pool",), cap=0.25))
    rows = suite.rows_for([path], ["vllm", "batchinfer"], arms_, BACKENDS, kv_bytes=147456)
    blocks = 100_000 // 16  # usable blocks on both sides: vLLM keeps a null block back as batchinfer keeps the pad block
    assert [r.opts for r in rows] == [{"kv_cache_memory_bytes": (blocks + 1) * 16 * 147456, "max_model_len": 8544},
                                      {"num_blocks": blocks + 1}, {"block_size": 64, "num_blocks": 100_000 // 64 + 1},
                                      {"kv_cache_dtype": "fp8", "kv_cache_memory_bytes": (2 * blocks + 1) * 16 * 147456 // 2,
                                       "max_model_len": 8544}]  # the bf16 rows' bytes: twice the tokens in them
    fp8_ours = suite.rows_for([path], ["batchinfer"], [suite.Arm("batchinfer", {"kv_dtype": "fp8"}, "c", ("pool",), cap=0.25)],
                              BACKENDS, kv_bytes=147456)
    assert fp8_ours[0].opts == {"kv_dtype": "fp8", "num_blocks": 2 * blocks + 1}
    # pool=contiguous: the block is the mapping unit (128 tokens for Qwen3-8B in bf16), so the same tokens are 8x
    # fewer blocks
    contiguous = suite.rows_for([path], ["batchinfer"], [suite.Arm("batchinfer", {"pool": "contiguous"}, "c", ("pool",), cap=0.25)],
                                BACKENDS, kv_bytes=147456)
    assert suite.contiguous_unit(147456) == 128 and contiguous[0].opts == {"pool": "contiguous", "num_blocks": 100_000 // 128 + 1}
    assert all(r.compare for r in rows)  # a capped vLLM row is not the reference
    with pytest.raises(ValueError, match="KV bytes per token"):
        suite.rows_for([path], ["vllm"], arms_[:1], BACKENDS, kv_bytes=None)
    # a workload whose quarter is below its longest request (smoke): the cap floors at that request and two blocks,
    # the same on both engines, so neither refuses the pool
    small = tmp_path / "smoke-quick.jsonl"
    small.write_text("")
    bench.workload.meta_path(small).write_text(json.dumps({"stats": {**stats, "kv_demand_tokens": 16_000}}))
    rows = suite.rows_for([small], ["vllm", "batchinfer"], arms_[:2], BACKENDS, kv_bytes=147456)
    floor = -(-(8028 + 512) // 16) + 2  # the longest prompt plus the largest max_tokens, in blocks, plus two: 536
    assert [r.opts for r in rows] == [{"kv_cache_memory_bytes": (floor + 1) * 16 * 147456, "max_model_len": 8544},
                                      {"num_blocks": floor + 1}]


def test_build_missing_builds_natural_workloads_with_natural_lengths(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(suite.subprocess, "run", lambda cmd, check: calls.append(cmd))
    monkeypatch.chdir(tmp_path)
    suite.build_missing([suite.workload_path("generate-natural"), suite.workload_path("mixed"),
                         suite.workload_path("nothing")], "stub")
    assert [c[c.index("--preset") + 1] for c in calls] == ["generate", "mixed"]
    assert ["--lengths" in c for c in calls] == [True, False]
    assert calls[0][-1] == "natural" and calls[0][c_out(calls[0]) + 1].endswith("generate-quick-natural.jsonl")


def c_out(cmd):
    return cmd.index("--out")


def test_a_warm_row_runs_its_workload_once_untimed_after_the_reset_and_labels_itself(here, workload):
    """Arm.warm: the reset, then an untimed generate over the even-numbered requests, then the timed one over the
    odd-numbered, with no reset between, and the row lands under its -warm label with warm recorded; vLLM's warm row
    is not the reference."""
    path, requests = workload
    warm = suite.Arm("batchinfer", {"order": "input"}, "radix warm", warm=True)
    rows = suite.rows_for([path], ["batchinfer", "vllm"], (suite.Arm("vllm", {}, "vLLM warm", warm=True), warm), BACKENDS)
    assert [r.warm for r in rows] == [True, True] and all(r.compare for r in rows)
    run_all(rows[1:])
    generates = [e for e in Counting.events if e[0] in ("reset", "generate")]
    even, odd = len(requests[0::2]), len(requests[1::2])
    assert generates == [("generate", 1), ("reset",), ("generate", even), ("generate", odd)]
    m = row_metrics(here, path, "batchinfer-order=input-warm")
    assert m["warm"] is True and m["column"] == "radix warm" and m["requests"] == odd
    assert m["unique_input_tokens"] is None  # the sidecar's count is the whole workload's
    assert suite.row_label("test_suite:Counting", {"order": "input"}) == "batchinfer-order=input"


def test_the_arm_table_keeps_the_engines_like_for_like():
    """Fairness by construction: no vLLM arm moves its running-sequence or batched-token caps (its defaults equal
    batchinfer's MAX_SEQS and MAX_PREFILL_BUDGET on 80 GB cards) and only the cache-off arm turns its prefix cache
    off; no batchinfer arm moves its caps; every capped and every warm batchinfer arm has the vLLM arm it is read
    against on the same workloads at the same cap; the reference is the one uncompared row; columns are unique."""
    vllm = [a for a in suite.ARMS if a.engine == "vllm"]
    assert all(k not in a.opts for a in vllm for k in ("max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization"))
    assert [a.column for a in vllm if a.opts.get("enable_prefix_caching") is False] == ["vLLM cache off"]
    ours = [a for a in suite.ARMS if a.engine == "batchinfer"]
    assert all(k not in a.opts for a in ours for k in ("max_batch_size", "num_blocks", "reserve_gb"))
    for a in ours:
        if a.cap is not None:
            ref = next(v for v in vllm if v.column == (a.reference or "vLLM capped"))
            assert set(a.workloads) <= set(ref.workloads) and ref.cap == a.cap
            assert (ref.opts.get("kv_cache_dtype") == "fp8") == (a.opts.get("kv_dtype") == "fp8")  # the same bytes
        if a.warm:
            ref = next(v for v in vllm if v.warm and not v.opts)
            assert set(a.workloads) <= set(ref.workloads) and ref.column == "vLLM warm"
    columns = [a.column for a in suite.ARMS]
    assert len(columns) == len(set(columns)) and columns[0] == "vLLM cache on"
    rows = suite.rows_for([Path("workloads/pool-quick.jsonl")], ["vllm", "batchinfer"],
                          [a for a in suite.select_arms() if a.cap is None], BACKENDS)
    assert [r.column for r in rows if not r.compare] == ["vLLM cache on"]


def test_the_smoke_workload_takes_every_arm(tmp_path):
    """The crash check: an arm names the workloads it runs on, but the smoke workload runs every arm once."""
    named = (suite.Arm("batchinfer", {"memory": "grow"}, "grow", ("mixed", "pool")), suite.Arm("batchinfer", {}, "batchinfer"))
    columns = lambda stem: [r.column for r in suite.rows_for([tmp_path / f"{stem}-quick.jsonl"], ["batchinfer"], named, BACKENDS)]  # noqa: E731
    assert columns("smoke") == ["grow", "batchinfer"]
    assert columns("classify") == ["batchinfer"] and columns("pool") == ["grow", "batchinfer"]


def test_workload_aliases_expand_to_their_families():
    """'grid' is the thirty share-grid presets, a family alias its five points in share order, 'qdial' the QuALITY
    dial; every one is a preset; the cap sweep's arms read against vLLM at the same cap on the pool-binding workloads."""
    assert len(bench.workload.GRID_WORKLOADS) == 30 and all(n in bench.workload.PRESETS for n in bench.workload.GRID_WORKLOADS)
    assert bench.workload.ALIASES["grid-dec-x32"] == ("dec0x32", "dec10x32", "dec30x32", "dec50x32", "dec90x32")
    assert bench.workload.PRESETS["dec30x32"] == "synthetic:320:prompt_len=256:shared_frac=0.3:groups=10:output_len=512"
    assert bench.workload.PRESETS["pf90x4"] == "synthetic:160:prompt_len=2048:shared_frac=0.9:groups=40:output_len=16"
    assert all(n in bench.workload.PRESETS for n in bench.workload.ALIASES["qdial"])
    assert bench.workload.ALIASES["diverse"] == ("docqa", "rag") and all(n in bench.workload.PRESETS for n in ("docqa", "rag"))
    on_docqa = {a.column for a in suite.ARMS if a.workloads and "docqa" in a.workloads}
    assert {"order input", "admission groups", "grow", "preempt", "preempt lookahead 32", "FP8 weights and KV", "vLLM FP8 KV"} <= on_docqa
    sweep = [a for a in suite.ARMS if a.cap in suite.CAPS]
    assert {a.cap for a in sweep} == set(suite.CAPS) and all(a.workloads == suite.BINDING for a in sweep)
    assert {a.column for a in sweep} == {"vLLM capped 1/2", "capped 1/2", "capped grow 1/2", "capped preempt 1/2",
                                         "vLLM capped 1/8", "capped 1/8", "capped grow 1/8", "capped preempt 1/8"}


def test_the_long_context_workloads_are_suite_workloads():
    """The two long-context presets: 24 prompts of 8,192 tokens, nothing or half shared by 8 rows; they are suite
    workloads, so the default suite carries them, and the attention arms run on them."""
    assert bench.workload.LONG_WORKLOADS == ("long0x8", "long50x8")
    assert bench.workload.PRESETS["long50x8"] == "synthetic:24:prompt_len=8192:shared_frac=0.5:groups=3:output_len=256"
    assert set(bench.workload.LONG_WORKLOADS) <= set(bench.workload.SUITE_WORKLOADS) <= set(suite.WORKLOADS)
    on_long = {a.column for a in suite.ARMS if a.workloads and "long50x8" in a.workloads}
    assert {"FlashAttention-2", "cascade", "contiguous KV", "block 128", "FP8 weights and KV", "vLLM FP8 KV"} <= on_long


def test_column_selects_arms_by_name():
    """--column keeps the named arms only, each on its own workloads, once-only ones included when named; an unknown
    column is refused rather than silently running nothing."""
    picked = suite.select_arms(columns=["vLLM cache on", "capped combined",
                                        "order input"])
    assert [a.column for a in picked] == ["vLLM cache on", "capped combined",
                                          "order input"]
    assert picked[1].workloads == suite.CAPPED and picked[1].reference == "vLLM FP8 KV capped"
    with pytest.raises(ValueError, match="no such arm"):
        suite.select_arms(columns=["batchinfer", "no such column"])


def test_the_campaign_names_arms_that_exist_and_keeps_its_references():
    """--campaign's columns are arms, and every arm it runs has the vLLM column it is read against in the campaign
    too, so no campaign row lacks its like-for-like."""
    picked = suite.select_arms(columns=suite.CAMPAIGN)
    assert len(picked) == len(suite.CAMPAIGN)
    columns = {a.column for a in picked}
    for a in picked:
        if a.engine == "batchinfer":
            ref = a.reference or ("vLLM capped" if a.cap is not None else "vLLM warm" if a.warm else "vLLM cache on")
            assert ref in columns, a.column
