"""results/ from several runs of the suite: every row is the run with its median wall time.

    python scripts/median_rows.py RUN_DIR... [--out results]
    python -m bench.report --html results/

Each RUN_DIR holds one run's rows as bench wrote them, <workload>/<model>/<gpu>/<label>/ (one run's
$BENCH_RESULTS_ROOT; its name names the run). A row found in several runs is
copied whole from the run with the median wall_s (the lower middle of an even count); copies of one run (a row a later
suite call reused, such as a baseline, has the timestamp it had) count once. Then every copied row that was compared
against a vLLM row has its compare recomputed against the vLLM row now beside it in --out, with bench.run's own
compare, since vLLM's greedy generations differ from run to run: so compare.against names the outputs its numbers
came from. That reads each row's workload file (metrics.json's workload, relative to the working directory) and
refuses one whose sha256 differs from the row's. --out/runs.json lists every run's wall_s of every row and which run
the row is. Rows already in --out that no run holds are left alone and listed: start from a results/ without rows.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.run import compare  # noqa: E402
from bench.workload import read  # noqa: E402


def gather(run_dirs):
    """{row path: {run name: (metrics, row dir)}}, one entry per distinct run of a row."""
    rows = {}
    for run in map(Path, run_dirs):
        for metrics in sorted(run.glob("*/*/*/*/metrics.json")):
            m, rel = json.loads(metrics.read_text()), str(metrics.parent.relative_to(run))
            seen = rows.setdefault(rel, {})
            if not any(other["timestamp"] == m["timestamp"] for other, _ in seen.values()):
                seen[run.name] = (m, metrics.parent)
    return rows


def median_run(runs):
    """The run name whose wall_s is the median (the lower middle of an even count)."""
    by_wall = sorted(runs, key=lambda name: runs[name][0]["wall_s"])
    return by_wall[(len(by_wall) - 1) // 2]


def outputs(path):
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def recompare(row, out):
    """Recomputes row/metrics.json's compare against the reference its compare.against names inside out."""
    path = row / "metrics.json"
    m = json.loads(path.read_text())
    against = (m.get("compare") or {}).get("against")
    if not against:
        return False
    ref = out / Path(against).relative_to("results")
    workload = Path(m["workload"])
    if hashlib.sha256(workload.read_bytes()).hexdigest() != m["workload_sha256"]:
        raise ValueError(f"{workload} is not the workload file {row} ran (sha256 differs)")
    requests, meta = read(workload)
    m["compare"] = {"against": against, **compare(requests, outputs(row / "outputs.jsonl"), outputs(ref),
                                                  set(meta.get("stop_token_ids") or ()))}
    path.write_text(json.dumps(m, indent=2) + "\n")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="one directory of rows per suite run")
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args(argv)
    rows, table = gather(args.runs), {}
    for rel, runs in sorted(rows.items()):
        chosen = median_run(runs)
        dst = args.out / rel
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(runs[chosen][1], dst)
        (dst / "report.html").unlink(missing_ok=True)  # written again by bench.report --html against --out
        table[rel] = {"wall_s": {name: m["wall_s"] for name, (m, _) in runs.items()}, "row": chosen}
    recompared = sum(recompare(args.out / rel, args.out) for rel in table)
    (args.out / "runs.json").write_text(json.dumps(table, indent=1) + "\n")
    others = [str(m.parent) for m in sorted(args.out.glob("*/*/*/*/metrics.json"))
              if str(m.parent.relative_to(args.out)) not in table]
    counts = {}
    for entry in table.values():
        counts[len(entry["wall_s"])] = counts.get(len(entry["wall_s"]), 0) + 1
    print(f"{len(table)} rows into {args.out} (runs per row: {counts}); compare recomputed on {recompared}")
    for other in others:
        print(f"not from these runs, left as it was: {other}")


if __name__ == "__main__":
    main()
