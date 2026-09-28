"""Every engine and option against the baselines, with one model load per engine configuration.

    python -m bench.suite                                   # Qwen/Qwen3-1.7B, the five quick workloads, vLLM + batchinfer
    python -m bench.suite --defaults-only                   # batchinfer defaults only: the quick "did it help" loop
    python -m bench.suite --engines naive --workloads mixed # the one-time naive baseline, one workload per call
    python -m bench.suite --workloads mixed --variant prefix_sharing=off --variant order=input,prefill_budget=auto
    python -m bench.suite --model Qwen/Qwen3-8B --workloads mixed --engines batchinfer,vllm
    python -m bench.suite --dry-run                         # the load groups, rows and reuse decisions; nothing runs

Why it is fast. A row's timed pass on a small model takes seconds, while loading the engine takes 10 s (batchinfer)
to 35-75 s (vLLM, whose compile and CUDA-graph capture dominate). So rows are grouped by engine and load options (a
load group), and each group runs in one worker process: one load, one warmup, then every workload and every policy
variant of that engine as its own timed pass (bench.run's prepare / load / timed_pass). Baseline rows (naive, vLLM)
are reused while they are still valid (reuse()); batchinfer rows always run, because they measure the code at hand.

Why it is fair. Before every pass the backend's reset() returns it to the state a fresh load leaves: vLLM drops its
prefix cache, the naive engine empties the allocator cache as its probe does, and batchinfer re-runs its executor's
probe after emptying it. Our engines keep no job state between runs (the prefix trie and every KV block live and
die with one flow.run). Each group runs in a process of its own, and the next one starts only when the GPU is idle
again (wait_for_idle_gpu), since vLLM's engine-core process can outlive its worker and batchinfer sizes its KV pool
from free memory at load.

Rows land where bench.run puts them (results/<workload>/<model>/<gpu>/<label>/), and every row but vLLM's prefix
cache-on one is compared against that row when it exists. The run ends with bench.report's table of this model and
GPU's rows for the workloads it ran.
"""
import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bench.backends import resolve
from bench.results import ROOT, run_dir
from bench.run import (engine_name, gpu_name, load, model_revision, prepare, reference_path, row_label, shell,
                       timed_pass, versions)
from bench import report
from bench.workload import PRESETS, SIZES, SUITE_PRESETS, parse_value

# Small by default: Qwen3-1.7B loads in ~10 s and runs mixed-quick in 5-19 s on the A100 and H100 (2026-09-27), and it
# shares Qwen3-8B's tokenizer and chat template, so the workload files are the same bytes for both.
MODEL = "Qwen/Qwen3-1.7B"
WORKLOADS = SUITE_PRESETS  # the five README workloads, in the report's row order (bench/workload.py)
SIZE = "quick"
# batchinfer's options, one knob flipped from its defaults each, and the non-optimised setup (HF's layers, eager: both
# build options off); the defaults row always runs as well.
VARIANTS = ({"prefix_sharing": False}, {"order": "input"}, {"admission": "groups"}, {"chunk_prefill": False},
            {"prefill_budget": "adaptive"}, {"fused_layers": False, "cuda_graphs": False})
ENGINES = ("vllm", "naive", "batchinfer")  # in run order: vLLM's cache-on row is every other row's reference
# naive is opt-in: its default arrival order is its slowest, several minutes a workload on mixed and generate even on
# 1.7B (sorted by max_tokens, 1.7B mixed-quick takes 82 s on the A100; arrival order ran 3.4x slower than sorted on 8B
# smoke-quick, and past 18 minutes on 8B mixed-quick). Built once per GPU and model, then reused.
DEFAULT_ENGINES = ("vllm", "batchinfer")
# prefix cache on (the reference), off, on without torch.compile or CUDA graphs (enforce_eager): the like-for-like
# for batchinfer rows that run eagerly, and on with max_num_seqs=512: vLLM's default is 256 running requests on GPUs
# under 70 GB and on any A100 (1,024 above that), fewer than a suite workload's 288-376, which batchinfer admits at once
VLLM_CONFIGS = ({}, {"enable_prefix_caching": False}, {"enforce_eager": True}, {"max_num_seqs": 512})
REFERENCE = "vllm"  # the label every other row compares against: this GPU's vLLM row with its prefix cache on
# A reused baseline must have run with the packages installed now, on the model snapshot cached now...
VERSION_KEYS = {"vllm": ("vllm", "torch"), "naive": ("torch", "transformers")}
# ...and with its own engine's code unchanged since its commit: a change there reruns it.
OWN_PATHS = {"vllm": ("bench/backends.py",), "naive": ("batchinfer/naive.py", "batchinfer/model.py")}
# Code a baseline shares with every row (the bench harness; for naive, the pipeline before the engine) changes with
# nearly every batchinfer commit. A change there is printed with the reuse rather than forcing a rerun, so a baseline
# is built once and reused; --baselines rerun refreshes it.
SHARED_PATHS = {"vllm": ("bench/run.py",),
                "naive": ("bench/run.py", "batchinfer/analysis.py", "batchinfer/prefix.py", "batchinfer/kv.py",
                          "batchinfer/policy.py", "batchinfer/flow.py", "batchinfer/schema.py", "batchinfer/io.py",
                          "batchinfer/bench.py", "batchinfer/metrics.py")}
# The idle GPU the next load group waits for (wait_for_idle_gpu): under IDLE_MIB used and no compute process,
# IDLE_CHECKS times in a row IDLE_EVERY_S apart; give up after IDLE_TIMEOUT_S.
IDLE_MIB, IDLE_CHECKS, IDLE_EVERY_S, IDLE_TIMEOUT_S = 100, 2, 2.0, 120.0


@dataclass
class Row:
    workload: str  # the workload file
    backend: str  # bench/backends.py registry name or package.module:Class
    opts: dict  # every option the row runs with, in label order; empty for the engine's defaults
    compare: bool = True  # compare against this GPU's REFERENCE row


@dataclass
class Group:
    """One engine plus its load options: one worker process, one load, then every row."""
    backend: str
    load_opts: dict
    rows: list = field(default_factory=list)


# plan ---------------------------------------------------------------------------------------------

def workload_path(name, size=SIZE):
    """A preset name (mixed) is workloads/<preset>-<size>.jsonl; anything ending in .jsonl is a path."""
    return Path(name) if name.endswith(".jsonl") else Path("workloads") / f"{name}-{size}.jsonl"


def split_opts(backend, opts):
    """(load options, policy options). A backend without load_opts (vLLM) loads anew for every option."""
    keys = getattr(resolve(backend), "load_opts", None)
    if keys is None:
        return dict(opts), {}
    return {k: v for k, v in opts.items() if k in keys}, {k: v for k, v in opts.items() if k not in keys}


def known_opts(backend):
    """The options a backend takes: its constructor's and its _policy_config's (where the constructor's **kwargs
    go); None when it takes anything (vLLM passes every option on to vllm.LLM)."""
    cls = resolve(backend)
    policy = getattr(cls, "_policy_config", None)
    names = set()
    for fn in (cls.__init__, policy):
        for p in inspect.signature(fn).parameters.values() if fn else ():
            if p.kind is p.VAR_KEYWORD and not (fn is cls.__init__ and policy):
                return None
            if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL) and p.name not in ("self", "model"):
                names.add(p.name)
    return names


def rows_for(workloads, engines, variants, backends):
    """Every row the suite runs, workload by workload: vLLM on and off, naive, batchinfer's defaults then variants."""
    rows = []
    for w in workloads:
        if "vllm" in engines:
            rows += [Row(str(w), backends["vllm"], dict(c), compare=bool(c)) for c in VLLM_CONFIGS]
        if "naive" in engines:
            rows.append(Row(str(w), backends["naive"], {}))
        if "batchinfer" in engines:
            rows += [Row(str(w), backends["batchinfer"], dict(v)) for v in ({}, *variants)]
    return rows


def group_rows(rows):
    """Load groups in ENGINES order (then first appearance), each holding its rows in the order given."""
    groups = {}
    for r in rows:
        load_opts, _ = split_opts(r.backend, r.opts)
        key = (r.backend, tuple(sorted(load_opts.items())))
        groups.setdefault(key, Group(r.backend, load_opts)).rows.append(r)
    rank = {name: i for i, name in enumerate(ENGINES)}
    order = sorted(groups.values(), key=lambda g: (rank.get(engine_name(g.backend), len(ENGINES)),
                                                   # vLLM's cache-on group first: its row is the reference
                                                   bool(g.load_opts) if engine_name(g.backend) == "vllm" else 0))
    return order


def changed_files(sha, paths):
    """The paths that differ between the commit a row ran at and the working tree, uncommitted edits included; None
    when that cannot be told: no sha, a -dirty one (what ran is not in git), or one git does not know."""
    if not sha or sha.endswith("-dirty"):
        return None
    try:
        r = subprocess.run(["git", "diff", "--name-only", sha, "--", *paths], capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.split() if r.returncode == 0 else None


def reuse(row, model, gpu, root=ROOT, installed=None, changed=changed_files, revision=None):
    """(True, why) when a baseline row on disk is still a valid measurement of this workload file, model snapshot
    and GPU, with the packages installed now and its own engine's code unchanged; (False, why) otherwise. revision:
    the model snapshot cached now (bench.run.model_revision), compared when both it and the row's are known."""
    name = engine_name(row.backend)
    path = run_dir(row.workload, model, gpu, row_label(row.backend, row.opts), root) / "metrics.json"
    if not path.exists():
        return False, "no row yet"
    if not path.with_name("outputs.jsonl").exists():
        return False, "no outputs.jsonl"  # every other row's --compare reference
    m = json.loads(path.read_text())
    installed = installed or versions()
    ran = f"{m.get('git_sha')} ({m.get('git_commit_time') or 'no commit time'}), ran {m.get('timestamp')}"
    if m.get("engine") != name:
        return False, "written before rows recorded their engine"
    if m.get("workload_sha256") != row_sha(row.workload):
        return False, "another workload file"
    if m.get("model") != model:
        return False, f"model {m.get('model')}"
    if m.get("length_violations"):
        return False, f"{m['length_violations']} length violations"
    for pkg in VERSION_KEYS.get(name, ()):
        was = (m.get("versions") or {}).get(pkg)
        if was != installed.get(pkg):
            return False, f"{pkg} {was} -> {installed.get(pkg)}"
    if revision and m.get("model_revision") and m["model_revision"] != revision:
        return False, f"model snapshot {m['model_revision'][:8]} -> {revision[:8]}"
    own = changed(m.get("git_sha"), OWN_PATHS.get(name, ()))
    if own is None:
        return False, f"cannot tell what code {m.get('git_sha')} ran"
    if own:
        return False, f"{name}'s own code changed since {m.get('git_sha')}: {', '.join(own)}"
    shared = changed(m.get("git_sha"), SHARED_PATHS.get(name, ()))
    note = f"; shared code changed since: {', '.join(shared)} (--baselines rerun refreshes)" if shared else ""
    return True, f"reusing {ran}{note}"


def row_sha(workload):
    return hashlib.sha256(Path(workload).read_bytes()).hexdigest()


def plan(rows, model, gpu, baselines="missing", root=ROOT, installed=None, changed=changed_files, revision=None):
    """(groups to run, [(row, decision)]): baseline rows still valid are dropped unless baselines == "rerun"."""
    decisions, keep = [], []
    for r in rows:
        if engine_name(r.backend) in VERSION_KEYS:
            ok, why = (False, "--baselines rerun") if baselines == "rerun" else reuse(r, model, gpu, root, installed,
                                                                                    changed, revision)
        else:
            ok, why = False, "batchinfer rows always run"
        decisions.append((r, "reuse" if ok else "run", why))
        if not ok:
            keep.append(r)
    return group_rows(keep), decisions


# run ----------------------------------------------------------------------------------------------

def gpu_state():
    """(MiB used, compute processes) on this GPU, or None without nvidia-smi."""
    used = shell("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits")
    if used is None:
        return None
    apps = shell("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader") or ""
    return int(used.splitlines()[0]), len([a for a in apps.splitlines() if a.strip()])


def wait_for_idle_gpu(state=gpu_state, sleep=time.sleep, clock=time.monotonic):
    """Returns once the GPU is idle on IDLE_CHECKS checks in a row, IDLE_EVERY_S apart; raises after IDLE_TIMEOUT_S.
    Returns at once without a GPU."""
    t0, idle = clock(), 0
    while True:
        s = state()
        if s is None:
            return
        idle = idle + 1 if s[0] < IDLE_MIB and s[1] == 0 else 0
        if idle >= IDLE_CHECKS:
            return
        if clock() - t0 > IDLE_TIMEOUT_S:
            raise RuntimeError(f"GPU still busy after {IDLE_TIMEOUT_S:.0f} s ({s[0]} MiB used, {s[1]} compute "
                               f"processes); the previous load group's process may not have exited")
        sleep(IDLE_EVERY_S)


def work(spec):
    """A worker: one load group's rows through one load. Returns the process status: 1 if any row failed."""
    model, gpu, backend, load_opts = spec["model"], spec["gpu"], spec["backend"], spec["load_opts"]
    failed, jobs = [], []
    for r in spec["rows"]:
        label = row_label(backend, r["opts"])
        compare_to = None
        if r["compare"]:
            reference = reference_path(REFERENCE, r["workload"], model, gpu)
            if reference.exists() and reference.with_name("metrics.json").exists():  # metrics.json: a whole row
                compare_to = REFERENCE
            else:
                print(f"WARN: {label} on {r['workload']}: no {REFERENCE} row on this GPU yet, so no --compare")
        try:
            jobs.append((r, prepare(r["workload"], model, compare_to, gpu)))
        except Exception:  # a missing workload or a mismatched reference: this row only
            traceback.print_exc()
            failed.append(f"{r['workload']} {label}")
    if jobs:
        engine, load_s = load(backend, model, load_opts)
        print(f"== {group_name(backend, load_opts)} loaded in {load_s:.1f} s; {len(jobs)} passes", flush=True)
        for r, job in jobs:
            label = row_label(backend, r["opts"])
            try:
                policy = {k: v for k, v in r["opts"].items() if k not in load_opts}
                if hasattr(engine, "configure"):
                    engine.configure(**policy)
                elif policy:
                    raise ValueError(f"{backend} takes no per-pass options, got {policy}")
                m = timed_pass(engine, job, backend, r["opts"], load_s)
                if m["length_violations"]:
                    failed.append(f"{r['workload']} {label} (length violations)")
            except Exception:
                traceback.print_exc()
                failed.append(f"{r['workload']} {label}")
            sys.stdout.flush()
    for f in failed:
        print(f"FAILED: {f}")
    return 1 if failed else 0


def run_worker(spec):
    """One load group in a process of its own, so its GPU memory is released before the next group loads."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
    try:
        return subprocess.run([sys.executable, "-m", "bench.suite", "--worker", f.name]).returncode
    finally:
        Path(f.name).unlink(missing_ok=True)


def run_groups(groups, model, gpu, worker=run_worker, idle=wait_for_idle_gpu):
    """Runs every group in order; returns [(group, status, seconds)]."""
    done = []
    for g in groups:
        idle()
        spec = {"model": model, "gpu": gpu, "backend": g.backend, "load_opts": g.load_opts,
                "rows": [asdict(r) for r in g.rows]}
        t0 = time.perf_counter()
        status = worker(spec)
        done.append((g, status, time.perf_counter() - t0))
    return done


def build_missing(paths, model, size=SIZE):
    """Builds each preset workload (workloads/<preset>-<size>.jsonl) that is not there yet, with bench.workload, one
    process each. Other missing paths are left for the caller to report."""
    for p in paths:
        preset = p.stem.removesuffix(f"-{size}")
        if not p.exists() and p == workload_path(preset, size) and preset in PRESETS:
            print(f"building {p}", flush=True)
            subprocess.run([sys.executable, "-m", "bench.workload", "build", "--preset", preset, "--size", size,
                            "--model", model, "--out", str(p)], check=True)


def parse_variant(text):
    """'order=input,prefill_budget=adaptive' -> {'order': 'input', 'prefill_budget': 'adaptive'}"""
    pairs = [part.split("=", 1) for part in text.split(",") if part.strip()]
    if not pairs or any(len(p) != 2 for p in pairs):
        raise ValueError(f"--variant takes k=v[,k=v], not {text!r}")
    return {k.strip(): parse_value(v.strip()) for k, v in pairs}


def group_name(backend, load_opts):
    """'vllm enable_prefix_caching=False', 'batchinfer' (its default load)."""
    return " ".join([engine_name(backend), *(f"{k}={v}" for k, v in load_opts.items())])


def describe(decisions, groups):
    lines = []
    for r, what, why in decisions:
        lines.append(f"  {what:5} {Path(r.workload).stem:16} {row_label(r.backend, r.opts):40} {why}")
    lines.append(f"{len(groups)} load groups:")
    for g in groups:
        lines.append(f"  {group_name(g.backend, g.load_opts)}: {len(g.rows)} passes")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bench.suite", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--workloads", default=",".join(WORKLOADS),
                   help="comma-separated presets (built at --size if missing) or .jsonl paths (default %(default)s)")
    p.add_argument("--size", choices=SIZES, default=SIZE)
    p.add_argument("--engines", default=",".join(DEFAULT_ENGINES),
                   help=f"comma-separated subset of {','.join(ENGINES)} (default %(default)s; naive is opt-in)")
    p.add_argument("--variant", action="append", metavar="K=V[,K=V]",
                   help="a batchinfer variant, with the option names --opt takes; repeat for more. Replaces the "
                        "default one-knob list; the defaults row always runs")
    p.add_argument("--defaults-only", action="store_true", help="batchinfer's defaults row only, no variants")
    p.add_argument("--baselines", choices=("missing", "rerun"), default="missing",
                   help="missing: reuse naive and vLLM rows that are still valid (default); rerun: run them again")
    p.add_argument("--dry-run", action="store_true", help="print the rows, reuse decisions and load groups; run nothing")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)  # internal: one load group's spec
    args = p.parse_args(argv)
    if args.worker:
        return work(json.loads(args.worker.read_text()))

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if not engines or set(engines) - set(ENGINES):
        p.error(f"--engines takes a subset of {','.join(ENGINES)}, not {args.engines!r}")
    if args.defaults_only and args.variant:
        p.error("--defaults-only and --variant exclude each other")
    try:
        variants = [] if args.defaults_only else [parse_variant(v) for v in args.variant] if args.variant else VARIANTS
    except ValueError as e:
        p.error(str(e))
    backends = {e: e for e in ENGINES}
    allowed = known_opts(backends["batchinfer"])
    for v in variants:
        if allowed is not None and set(v) - allowed:
            p.error(f"--variant {v}: unknown option {sorted(set(v) - allowed)}; batchinfer takes {sorted(allowed)}")
    paths = [workload_path(w.strip(), args.size) for w in args.workloads.split(",") if w.strip()]

    gpu = gpu_name()
    if not args.dry_run:
        build_missing(paths, args.model, args.size)
    missing = [str(w) for w in paths if not w.exists()]
    if missing:
        p.error(f"no workload file {', '.join(missing)} (a dry run builds nothing)")
    groups, decisions = plan(rows_for(paths, engines, variants, backends), args.model, gpu, args.baselines,
                             revision=model_revision(args.model))
    print(f"suite: {args.model} on {gpu or 'no GPU'}; {sum(len(g.rows) for g in groups)} rows to run")
    print(describe(decisions, groups), flush=True)
    if args.dry_run:
        return 0
    started, t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z"), time.perf_counter()  # started: bench.run's timestamp format
    done = run_groups(groups, args.model, gpu)
    ran = {run_dir(r.workload, args.model, gpu, row_label(r.backend, r.opts)) for g in groups for r in g.rows}
    print(summary(done, time.perf_counter() - t0, args.model, gpu, paths, started, ran))
    return 1 if any(status for _, status, _ in done) else 0


def summary(done, total_s, model, gpu, paths, started=None, ran=()):
    """The load groups' times, then the comparison of the rows on disk for this model and GPU. started (a bench.run
    timestamp) and ran (the row dirs this run set out to write): such a row whose metrics.json ran before started is
    an earlier run's, left in place because this run's attempt failed, so it is named and kept out of the table."""
    lines = ["", "load groups:"]
    for g, status, secs in done:
        lines.append(f"  {group_name(g.backend, g.load_opts)}: {len(g.rows)} passes in {secs:.0f} s"
                     + ("" if status == 0 else f" (exit {status})"))
    lines.append(f"suite: {total_s:.0f} s")
    stems = {Path(p).stem for p in paths}
    runs = {p: json.loads(p.read_text()) for p in report.find([ROOT])}  # by path: `ran` names row dirs
    runs = {p: m for p, m in runs.items() if "by_source" in m and m.get("model") == model and m.get("gpu") == gpu
            and Path(m["workload"]).stem in stems}
    stale = {p: m for p, m in runs.items()
             if started and p.parent in ran and (report.utc(m.get("timestamp")) or "") < report.utc(started)}
    lines += [f"FAILED this run: {p.parent}; the row on disk ran {m.get('timestamp')}, so it is left out"
              for p, m in stale.items()]
    runs = [m for p, m in runs.items() if p not in stale]
    if runs:
        lines += ["", report.table(runs)]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
