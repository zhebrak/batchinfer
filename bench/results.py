"""Where a run's files live, keyed by the hardware it ran on:

    results/<workload>/<model>/<gpu>/<label>/   metrics.json, outputs.jsonl, details.json

<gpu> is the short name of the GPU the run used (A100-SXM4-40GB, H100-PCIe; cpu without one), so rows
from different machines never share a directory and can be pulled into one results/ tree. The tree is results/ in the
working directory, or $BENCH_RESULTS_ROOT when set, to keep a run's rows out of the tracked results/ (RUN.md gives each
of three runs its own).
"""
import os
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
