"""Where a run's files live, keyed by the hardware it ran on:

    results/<workload>/<model>/<gpu>/<label>/   metrics.json, outputs.jsonl, details.json

<gpu> is the short name of the GPU the run used (A100-SXM4-40GB, H100-PCIe; cpu without one), so rows
from different machines never share a directory and can be pulled into one results/ tree. The tree is results/ in the
working directory, or $BENCH_RESULTS_ROOT when set, to keep a run's rows out of the tracked results/ (RUN.md gives each
of three runs its own).

    python -m bench.results migrate results/    # rows written by older code: layout, then references

It moves rows from the older <workload>/<model>/<label>/ layout (migrate) and rewrites a compare.against recorded as
an absolute path on the machine that ran the row into its path in the results tree (relink_references).
"""
import json
import os
import shutil
import sys
from pathlib import Path

TREE = Path("results")  # the tree as the repo holds it, and as a row names its reference
ROOT = Path(os.environ.get("BENCH_RESULTS_ROOT") or TREE)  # where rows are written and read


def gpu_slug(name):
    """The <gpu> path part for an nvidia-smi name: 'NVIDIA H100 PCIe' -> 'H100-PCIe', None -> 'cpu'."""
    return "-".join(name.removeprefix("NVIDIA ").split()) if name else "cpu"


def model_name(model):
    return str(model).rstrip("/").rsplit("/", 1)[-1]


def run_dir(workload, model, gpu, label, root=ROOT):
    return Path(root) / Path(workload).stem / model_name(model) / gpu_slug(gpu) / label


def migrate(root):
    """Moves each row written as <workload>/<model>/<label>/ under its recorded GPU: bench.run's gpu, or
    memory.gpu_name in batchinfer CLI metrics. A row whose target already exists (a pull brought an old
    copy back) is merged into it when every file both hold is byte-identical, and kept otherwise; rows
    that recorded no GPU, or sit at a depth other than 3 or 4 (a custom --out), are kept and reported.
    Returns (moved, merged, kept) as (row, target) and (row, reason) pairs."""
    root = Path(root)
    moved, merged, kept = [], [], []
    for metrics in sorted(root.rglob("metrics.json")):
        row = metrics.parent
        parts = row.relative_to(root).parts
        if len(parts) == 4:
            continue  # already <workload>/<model>/<gpu>/<label>
        if len(parts) != 3:
            kept.append((row, "not at <workload>/<model>/<label>; move it by hand"))
            continue
        m = json.loads(metrics.read_text())
        if "gpu" in m:
            gpu = m["gpu"]
        elif (m.get("memory") or {}).get("gpu_name"):
            gpu = m["memory"]["gpu_name"]
        else:
            kept.append((row, "no gpu recorded"))
            continue
        target = root / parts[0] / parts[1] / gpu_slug(gpu) / parts[2]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            row.rename(target)
            moved.append((row, target))
            continue
        files = list(row.iterdir())
        differ = [f.name for f in files if f.is_dir() or ((target / f.name).exists()
                                                          and (target / f.name).read_bytes() != f.read_bytes())]
        if differ:
            kept.append((row, f"{target} exists with a different {', '.join(differ)}"))
            continue
        for f in files:
            if not (target / f.name).exists():
                f.rename(target / f.name)
        shutil.rmtree(row)
        merged.append((row, target))
    return moved, merged, kept


def relink_references(root):
    """Rewrites each row's compare.against that holds an absolute path with a results component, as rows recorded
    it before bench.run named the reference by its path in the results tree, into that path: from the last results
    component on. Relative paths, and paths outside any results tree, are left alone; the rest of metrics.json is
    written back as bench.run writes it. Returns (metrics.json, old, new) triples."""
    changed = []
    for metrics in sorted(Path(root).rglob("metrics.json")):
        m = json.loads(metrics.read_text())
        against = (m.get("compare") or {}).get("against")
        parts = Path(against).parts if against else ()
        if not against or not Path(against).is_absolute() or "results" not in parts:
            continue
        new = str(Path(*parts[len(parts) - 1 - parts[::-1].index("results"):]))
        m["compare"]["against"] = new
        metrics.write_text(json.dumps(m, indent=2) + "\n")
        changed.append((metrics, against, new))
    return changed


def main(argv):
    if len(argv) < 2 or argv[0] != "migrate":
        sys.exit("usage: python -m bench.results migrate DIR...")
    for root in argv[1:]:
        moved, merged, kept = migrate(root)
        for row, target in moved:
            print(f"moved {row} -> {target}")
        for row, target in merged:
            print(f"merged {row} into {target}: the files both hold are identical")
        for row, why in kept:
            print(f"kept {row}: {why}")
        for metrics, old, new in relink_references(root):
            print(f"relinked {metrics}: {old} -> {new}")


if __name__ == "__main__":
    main(sys.argv[1:])
