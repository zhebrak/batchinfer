"""The arms matrix of the results index (bench.report draws it on results/index.html), from bench rows: one row per
arm of the suite (bench.suite.ARMS) and one column per workload, each cell the wall and its ratio to the arm's
like-for-like vLLM row.
"""
import json
from pathlib import Path

from bench.report import engine, model_name, workload_rank
from bench.results import gpu_slug
from bench.suite import ARMS, BINDING
from bench.workload import DIVERSE_WORKLOADS, SIZES, SUITE_WORKLOADS, workload_name

REFERENCES = ("vLLM cache on", "vLLM capped", "vLLM warm")  # shown plain in the arms matrix: the rows others divide by


def column_of(m):
    """The arm a row ran: the arm whose engine and options it matches (a row's directory is its options, so a
    --variant run of the same options is the same arm), else the column the suite recorded, which the capped rows
    need (their options differ per workload)."""
    key = (engine(m), json.dumps(m["opts"], sort_keys=True, default=str), bool(m.get("warm")))
    for arm in ARMS:
        if arm.cap is None and (arm.engine, json.dumps(arm.opts, sort_keys=True, default=str), arm.warm) == key:
            return arm.column
    return m.get("column") or None


def wall_of(m):
    """A row's wall, or None for a row with length violations: not a valid performance row."""
    return None if m is None or m.get("length_violations") else m.get("wall_s")


def by_job(runs):
    """{(model, gpu): {workload stem: {column: row}}} over the rows that belong to an arm, in the suite's order."""
    jobs = {}
    for m in sorted(runs, key=workload_rank):
        column = column_of(m)
        if column is None or not m.get("gpu"):
            continue
        key = (model_name(m["model"]), gpu_slug(m["gpu"]))
        jobs.setdefault(key, {}).setdefault(Path(m["workload"]).stem, {})[column] = m
    return jobs


def short(stem):
    """A workload file's stem as the suite names it: mixed-quick -> mixed, generate-quick-natural -> generate-natural."""
    return next((workload_name(stem, size) for size in SIZES if f"-{size}" in stem), stem)


def ordered(rows_by_workload, names):
    return sorted((s for s in rows_by_workload if short(s) in names), key=lambda s: names.index(short(s)))


def reference_of(arm):
    """The vLLM column an arm is read against: the one it names, else vLLM at the same cap for a capped batchinfer
    arm, vLLM's warm row for a warm one, vLLM with its cache on otherwise."""
    if arm.reference or arm.engine == "vllm":
        return arm.reference or "vLLM cache on"
    return "vLLM capped" if arm.cap is not None else "vLLM warm" if arm.warm else "vLLM cache on"


def arms(rows_by_workload):
    """(workload stems, [(arm, [(wall, ratio) or None per stem])]): every arm with a row, in ARMS order; ratio is None
    for the reference rows themselves."""
    stems = ordered(rows_by_workload, SUITE_WORKLOADS + DIVERSE_WORKLOADS + BINDING)
    out = []
    for arm in ARMS:
        cells = []
        for stem in stems:
            cols = rows_by_workload[stem]
            w, ref = wall_of(cols.get(arm.column)), wall_of(cols.get(reference_of(arm)))
            cells.append(None if w is None else (w, None if arm.column in REFERENCES or not ref else w / ref))
        if any(cells):
            out.append((arm, cells))
    return stems, out
