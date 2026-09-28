"""Feed a workload to a backend and record throughput and correctness.

    python -m bench.run workloads/mixed-quick.jsonl --backend vllm
    python -m bench.run workloads/mixed-quick.jsonl --backend vllm --opt enable_prefix_caching=false
    python -m bench.run workloads/mixed-quick.jsonl --backend batchinfer --compare vllm

The whole workload goes to the backend in one call, and only that call is timed. Model
load and warmup are reported separately as load_s. This CLI runs one row per process; bench.suite
runs many rows through one load with the same prepare / load / timed_pass steps. Before every
timed pass the backend's reset() puts it back in the state a fresh load leaves (vLLM drops its
prefix cache; our engines, which keep no job state between runs, get the allocator state their
load-time probe leaves), so a pass never profits from the one before. The backend only sees what
a real client would send: kind, group, reference and scorer stay here.

Rows land in results/<workload>/<model>/<gpu>/<label>/, or under $BENCH_RESULTS_ROOT when set (bench/results.py).
--compare refuses a reference from another GPU, and a bare label such as vllm picks this GPU's row in that tree, or
under $BENCH_REFERENCE_ROOT when set; the row records that reference by its path in the results tree (results/...),
never by where it lay on this machine. With $BENCH_BRANCH set (a run of a branch's code) a row of our engines is
labelled <label>@<branch> and records its branch, so it lands beside main's row, never on top of it (row_label).
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from statistics import mean

from bench.backends import resolve
from bench.glossary import GLOSSARY
from bench.report import render, vllm_reference
from bench.results import ROOT, TREE, run_dir
from bench.sources import group_by
from bench.utilisation import load_shape, utilisation
from bench.workload import lcp, meta_path, parse_value, read

CLIENT_FIELDS = ("id", "prompt", "max_tokens", "ignore_eos", "labels")
WARMUP = {"id": "warmup", "prompt": "Say hello.", "max_tokens": 8, "ignore_eos": False, "labels": None}


def norm(text):
    """' Time.' -> 'time', 'credit limit' -> 'credit_limit'."""
    return re.sub(r"\s+", "_", text.strip().strip("\"'`").rstrip(".!,;:").strip().lower())


def score_choice(text, reference):
    return text.strip()[:1].upper() == reference


def score_label(text, reference):
    return norm(text) == norm(reference)  # exact: 'timer' must not count as 'time'


def score_number(text, reference):
    m = re.search(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    try:
        return m is not None and float(m.group(1).replace(",", "")) == float(reference)
    except ValueError:
        return False


SCORERS = {"choice": score_choice, "label": score_label, "number": score_number}


def until_stop(ids, stop):
    """ids up to and including the first stop token: what ignore_eos forces after it is noise."""
    return next((ids[:i + 1] for i, t in enumerate(ids) if t in stop), ids)


def prefix_match(a, b, stop=()):
    """Fraction of the longer output that both share as a prefix; token ids when both have them."""
    if a.get("token_ids") and b.get("token_ids"):
        x, y = until_stop(a["token_ids"], stop), until_stop(b["token_ids"], stop)
    else:
        x, y = a["text"], b["text"]
    longest = max(len(x), len(y))
    return lcp(x, y) / longest if longest else 1.0


def compare(requests, results, other, stop=()):
    scores = {}
    for r in requests:
        a, b = results[r["id"]], other[r["id"]]
        score = float(norm(a["text"]) == norm(b["text"])) if r["kind"] == "classify" else prefix_match(a, b, stop)
        scores.setdefault(r["kind"], []).append(score)
    return {kind: {"n": len(s), "match": round(mean(s), 4), "identical": round(mean(float(x == 1.0) for x in s), 4)}
            for kind, s in scores.items()}


def load_reference(outputs_path, workload_sha256, model, gpu):
    """Outputs of an earlier run, refused unless it ran the same workload file with the same model on
    the same GPU: ids repeat across presets and rebuilds, so an id alone does not identify a prompt,
    and bf16 kernels differ across GPUs, so another GPU's outputs are not a like-for-like reference."""
    meta_file = Path(outputs_path).parent / "metrics.json"
    if not meta_file.exists():
        raise ValueError(f"--compare needs {meta_file} to check which workload and model produced {outputs_path}")
    ref = json.loads(meta_file.read_text())
    if ref["workload_sha256"] != workload_sha256 or ref["model"] != model:
        raise ValueError(f"--compare {outputs_path} ran {ref['workload']} with {ref['model']}, "
                         f"not this workload file with {model}")
    if ref.get("gpu") != gpu:
        raise ValueError(f"--compare {outputs_path} ran on {ref.get('gpu')}, not this run's {gpu}")
    with open(outputs_path) as f:
        return {res["id"]: res for res in map(json.loads, f)}


FINISH_REASONS = ("stop", "length")


def length_violations(requests, by_id, stop=()):
    """Requests whose output breaks the length rules (GLOSSARY["length_violations"] states them for the report too):
    every request gets at least its first token and ends where it should, finish_reason "length" meaning max_tokens
    were produced and "stop" meaning it ended on its first stop token. stop: the workload's stop_token_ids; the stop
    rules need them and the token ids, and skip ignore_eos requests. A run with any is not a valid performance row:
    bench.run exits 1."""
    bad = 0
    for r in requests:
        res = by_id[r["id"]]
        n, ids, cap, reason = res["output_tokens"], res.get("token_ids"), r["max_tokens"], res["finish_reason"]
        bad += (n < 1 or n > cap or reason not in FINISH_REASONS or (reason == "length" and n != cap)
                or (r["ignore_eos"] and n != cap) or (ids is not None and len(ids) != n)
                or (bool(stop and ids) and not r["ignore_eos"] and stops_wrong(ids, reason, stop)))
    return bad


def stops_wrong(ids, reason, stop):
    """A "stop" whose last token is no stop id, or a stop id before the last token (decoding went on past it). A
    "length" that ends on a stop id is fine: vLLM checks max_tokens before EOS, so an EOS sampled at the cap reads
    "length"."""
    return (reason == "stop" and ids[-1] not in stop) or any(t in stop for t in ids[:-1])


def versions():
    """Package versions of this process, read from installed metadata without importing anything."""
    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "vllm"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def model_revision(model):
    """The Hugging Face snapshot (commit) the model's config resolves to in the local cache, or None: never downloads."""
    try:
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model, local_files_only=True)._commit_hash
    except Exception:  # no transformers, no cached snapshot, or not a hub model: provenance is best effort
        return None


def engine_name(backend):
    """The engine a --backend names, as rows and reports call it: the class's own name (batchinfer, naive, vllm,
    dummy), so a registry name and its package.module:Class spelling write the same row; else the class name
    lowercased."""
    cls = resolve(backend)
    return getattr(cls, "name", None) or cls.__name__.lower()


def row_label(backend, opts):
    """A row's directory name: its engine, then every option it ran with, in the order given. Run from a branch
    ($BENCH_BRANCH) it ends in @<branch>, so it lands beside main's row, never on top of it; vLLM's rows never do, as
    they do not run the branch's engine code and --compare vllm reads them by the plain label."""
    engine, branch = engine_name(backend), os.environ.get("BENCH_BRANCH")
    label = engine + "".join(f"-{k}={v}" for k, v in opts.items())
    return label + "@" + branch.replace("/", "-") if branch and engine != "vllm" else label


def gpu_device():
    """The GPU a run uses, as nvidia-smi -i takes it: the first of CUDA_VISIBLE_DEVICES, else 0."""
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"


class GpuSampler:
    """Samples GPU utilisation and memory every 200 ms via nvidia-smi; no-op where it is missing."""

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(["nvidia-smi", "-i", gpu_device(), "--query-gpu=utilization.gpu,memory.used",
                                          "--format=csv,noheader,nounits", "-lms", "200"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except FileNotFoundError:
            self.proc = None
        return self

    def __exit__(self, *exc):
        self.util = self.mem = None
        if self.proc:
            self.proc.terminate()
            rows = [line.split(",") for line in self.proc.communicate()[0].splitlines() if "," in line]
            if rows:
                self.util = round(mean(float(u) for u, _ in rows), 1)
                self.mem = max(float(m) for _, m in rows)


def shell(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_name():
    return shell("nvidia-smi", "-i", gpu_device(), "--query-gpu=name", "--format=csv,noheader")


def git_sha():
    """git describe of the code that ran, -dirty when a tracked file outside results/ differs from HEAD: the repo
    tracks its rows, and a tree whose rows a run rewrote still runs the committed code."""
    sha = shell("git", "describe", "--always")
    if sha and shell("git", "status", "--porcelain", "--untracked-files=no", "--", ":/", ":(top,exclude)results"):
        sha += "-dirty"
    return sha


def reference_path(compare_to, workload, model, gpu, root=None):
    """--compare takes an outputs.jsonl path, or the label of an earlier run of this workload and
    model on this GPU, which resolves under root (default $BENCH_REFERENCE_ROOT, else ROOT). A row records its
    reference with root=TREE: by its path in the results tree, never by where it lay on this machine."""
    compare_to = str(compare_to)
    if "/" in compare_to or compare_to.endswith(".jsonl"):
        return Path(compare_to)
    root = root or os.environ.get("BENCH_REFERENCE_ROOT") or ROOT
    return run_dir(workload, model, gpu, compare_to, root) / "outputs.jsonl"


@dataclass
class Job:
    """One row's workload, read and checked before any model loads: its requests and sidecar, the model and GPU
    it runs on, and the reference outputs it is compared against (None without --compare): the file read, and
    against, the path the row records for it (its path in the results tree)."""
    workload: Path
    requests: list
    meta: dict
    model: str
    gpu: str | None
    workload_sha256: str
    reference: Path | None = None
    other: dict | None = None
    against: str | None = None


def prepare(workload, model, compare_to, gpu):
    """The Job for one row. It runs before the model loads, so a missing workload or a missing or mismatched
    reference fails fast. gpu: the nvidia-smi name the row runs on (gpu_name()), None without a GPU; taken as
    given, so a caller that looked it up once (bench.suite) decides it, not this machine."""
    requests, meta = read(workload)
    model = model or meta.get("model")
    job = Job(Path(workload), requests, meta, model, gpu, hashlib.sha256(Path(workload).read_bytes()).hexdigest())
    if compare_to:
        job.reference = reference_path(compare_to, workload, model, gpu)
        job.other = load_reference(job.reference, job.workload_sha256, model, gpu)
        job.against = str(reference_path(compare_to, workload, model, gpu, TREE))
    return job


def load(backend, model, opts=None, warmup=True):
    """(engine, load_s): the backend constructed with opts, plus one warmup request, timed together."""
    t0 = time.perf_counter()
    engine = resolve(backend)(model=model, **(opts or {}))
    if warmup:
        engine.generate([dict(WARMUP)])
    return engine, time.perf_counter() - t0


def run(workload, backend="dummy", model=None, opts=None, out=None, compare_to=None, warmup=True):
    """One row in one process: prepare, load, one timed pass."""
    job = prepare(workload, model, compare_to, gpu_name())
    engine, load_s = load(backend, job.model, opts, warmup)
    return timed_pass(engine, job, backend, opts or {}, load_s, out)


def timed_pass(engine, job, backend, opts, load_s, out=None):
    """One timed generate() over the whole workload, then every check and file of the row: outputs.jsonl,
    metrics.json, details.json, report.html. opts: every option the row ran with (load and policy), which name its
    directory. The backend's reset() runs first, outside the timer."""
    workload, requests, meta, model, gpu = job.workload, job.requests, job.meta, job.model, job.gpu
    workload_sha256, compare_to, other = job.workload_sha256, job.reference, job.other
    if hasattr(engine, "reset"):
        engine.reset()
    client = [{k: r[k] for k in CLIENT_FIELDS} for r in requests]
    with GpuSampler() as sampler:
        t0 = time.perf_counter()
        results = engine.generate(client)
        wall_s = time.perf_counter() - t0

    by_id = {res["id"]: res for res in results}
    expected = {r["id"] for r in requests}
    if len(by_id) != len(results) or by_id.keys() != expected:
        raise RuntimeError(f"backend returned {len(results)} results for {len(requests)} requests: "
                           f"{len(expected - by_id.keys())} missing, {len(by_id.keys() - expected)} unknown, "
                           f"{len(results) - len(by_id)} duplicated")
    stop = set(meta.get("stop_token_ids") or ())  # the sidecar's: the stop rules and --compare's cut both use it
    violations = length_violations(requests, by_id, stop)

    by_source = {}
    for name, rows in group_by(requests, "source").items():
        entry = by_source[name] = {"n": len(rows), "kind": rows[0]["kind"],
                                   "mean_output_tokens": round(mean(by_id[r["id"]]["output_tokens"] for r in rows), 1)}
        scored = [r for r in rows if r.get("scorer")]
        if scored:
            entry["accuracy"] = round(mean(SCORERS[r["scorer"]](by_id[r["id"]]["text"], r["reference"]) for r in scored), 4)

    st = meta.get("stats", {})
    input_tokens = sum(r["prompt_tokens"] for r in requests)
    unique_tokens = st.get("unique_prompt_tokens")
    output_tokens = sum(res["output_tokens"] for res in results)
    target_s = meta.get("target_s")
    details = engine.details() if hasattr(engine, "details") else None  # everything the backend recorded
    mfu, mbu, used = utilisation(load_shape(model), details, gpu, wall_s)
    sidecar = meta_path(workload)  # its stop_token_ids and stats shape --compare and the report
    branch = os.environ.get("BENCH_BRANCH") or None
    metrics = {
        "workload": str(workload), "workload_sha256": workload_sha256,
        "workload_meta_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest() if sidecar.exists() else None,
        "preset": meta.get("preset"), "size": meta.get("size"), "mix": meta.get("mix"), "seed": meta.get("seed"),
        "lengths": meta.get("lengths"),
        "backend": backend, "engine": engine_name(backend), "opts": opts, "model": model,
        "model_revision": model_revision(model),
        "gpu": gpu, "versions": versions(),
        # GIT_SHA overrides for a tree without its .git
        "git_sha": os.environ.get("GIT_SHA") or git_sha(),
        "git_commit_time": shell("git", "log", "-1", "--format=%cI"),  # the HEAD commit's committer time
        "branch": branch,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "load_s": round(load_s, 2), "wall_s": round(wall_s, 3), "requests": len(requests),
        "req_per_s": round(len(requests) / wall_s, 2),
        "input_tokens": input_tokens, "unique_input_tokens": unique_tokens, "output_tokens": output_tokens,
        "input_tok_per_s": round(input_tokens / wall_s, 1),
        "unique_input_tok_per_s": round(unique_tokens / wall_s, 1) if unique_tokens else None,
        "output_tok_per_s": round(output_tokens / wall_s, 1),
        "total_tok_per_s": round((input_tokens + output_tokens) / wall_s, 1),
        "ideal_prefix_reuse": st.get("ideal_prefix_reuse"),
        "gpu_util_mean": sampler.util, "mfu_pct": mfu, "mbu_pct": mbu, "utilisation": used,
        "gpu_mem_peak_mb": sampler.mem,
        "length_violations": violations, "by_source": by_source,
        "compare": None, "backend_stats": engine.stats() if hasattr(engine, "stats") else None,
        "target_s": target_s, "over_budget": bool(target_s and wall_s > target_s),
    }
    if compare_to:
        metrics["compare"] = {"against": job.against or str(compare_to), **compare(requests, by_id, other, stop)}

    label = row_label(backend, opts)
    out = Path(out or run_dir(workload, model, gpu, label, ROOT))
    out.mkdir(parents=True, exist_ok=True)
    # metrics.json marks a whole row, so it goes first and comes back last: a failure in between (details that do
    # not serialise) leaves no row, never this attempt's outputs beside an earlier attempt's metrics or details
    for name in ("metrics.json", "details.json", "report.html"):
        (out / name).unlink(missing_ok=True)
    with open(out / "outputs.jsonl", "w") as f:
        f.writelines(json.dumps(by_id[r["id"]]) + "\n" for r in requests)
    if details is not None:
        (out / "details.json").write_text(json.dumps(details) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    acc = ", ".join(f"{name} {s['accuracy']:.2f}" for name, s in by_source.items() if "accuracy" in s)
    print(f"{label}: {len(requests)} requests in {wall_s:.1f} s (load {load_s:.1f} s), {metrics['req_per_s']} req/s, "
          f"{metrics['output_tok_per_s']} out tok/s, {metrics['total_tok_per_s']} total tok/s")
    print(f"accuracy: {acc or 'n/a'}; results in {out}")
    if metrics["compare"]:
        print("match vs", compare_to, {k: v for k, v in metrics["compare"].items() if k != "against"})
    if violations:
        print(f"WARN: {violations} requests broke length rules, so this is not a valid performance row. "
              f"{GLOSSARY['length_violations']}")
    if metrics["over_budget"]:
        print(f"WARN: timed run took {wall_s:.1f} s, over the {target_s} s budget for this workload")
    # last, so a template error cannot cost the files and warnings above
    files = ["metrics.json", "outputs.jsonl"] + (["details.json"] if details is not None else [])
    reference = vllm_reference(out / "metrics.json", metrics, details)
    (out / "report.html").write_text(render(metrics, out.name, files, details, reference))
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bench.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workload", type=Path)
    p.add_argument("--backend", default="dummy", help="dummy, vllm, batchinfer, naive, or package.module:Class")
    p.add_argument("--model", help="default: the model the workload was built for")
    p.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE", help="backend constructor option")
    p.add_argument("--out", type=Path, help="default results/<workload>/<model>/<gpu>/<backend>[-opts]/")
    p.add_argument("--compare", metavar="PATH|LABEL",
                   help="outputs.jsonl of an earlier run of the same workload file, model and GPU, or that run's "
                        "label (e.g. vllm) to take it from this GPU's rows under $BENCH_REFERENCE_ROOT or results/")
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args(argv)
    opts = {k: parse_value(v) for k, v in (o.split("=", 1) for o in args.opt)}
    metrics = run(args.workload, args.backend, args.model, opts, args.out, args.compare, not args.no_warmup)
    return 1 if metrics["length_violations"] else 0  # every file is written first, so the row can be inspected


if __name__ == "__main__":
    sys.exit(main())
