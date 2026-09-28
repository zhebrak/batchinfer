"""One table of benchmark runs, in the terminal and as an HTML index, and a page per run.

    python -m bench.report results/                    # every metrics.json below results/
    python -m bench.report results/mixed-quick/Qwen3-8B/H100-PCIe/
    python -m bench.report --html results/             # also report.html beside each metrics.json, results/index.html
    python -m bench.report --serve 8765 results/       # --html, then serve results/ on http://127.0.0.1:8765/

A row says what launched, one column per knob (engine, prefix reuse, order, prefill budget; any other --opt values
are on its page and in its directory name, the engine's tooltip), then the few numbers that tell its tradeoffs: how fast (wall s, × vLLM), how right (accuracy %) and why (prefix
hit %, steps, ms/step, decode batch), how much of the card it used (GPU util %, MFU %, MBU %), and last the commit
it ran. A column blank in every row is left out and named under the table. Rows of one job (workload, model, GPU)
share a band. A run's page adds the engine's other telling counters, the counts behind MFU and MBU, accuracy by
source, the match against its reference and the charts of its record; metrics.json and details.json, linked from
it, hold everything else. bench/glossary.py holds the hover text of every underlined name.

Two metrics.json shapes exist: bench.run's flat dict, which is tabulated and paged, and python -m
batchinfer run's {meta, timing, ...}, which is paged only. --serve binds 127.0.0.1 unless --bind names
another interface; never a public one.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup, escape  # jinja2's own dependency

from bench.charts import charts, engine_of, to_columns
from bench.glossary import ENGINE, GLOSSARY
from bench.results import gpu_slug, model_name
from bench.workload import SUITE_PRESETS

META_NOISE = ("model", "input", "output", "timestamp", "torch", "transformers", "alloc_conf", "args")
ARGS_NOISE = ("cmd", "input", "output", "metrics", "model")  # the index's config cell leaves these out
SIBLINGS = ("outputs.jsonl", "details.json")


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}" if abs(value) < 100 else f"{value:,.0f}"
    return str(value)


def find(args):
    paths = []
    for arg in map(Path, args or ["results"]):
        paths += sorted(arg.rglob("metrics.json")) if arg.is_dir() else [arg]
    return paths


def read(path):
    """A row's metrics.json, and the details.json beside it (None when there is none)."""
    details = path.with_name("details.json")
    return json.loads(path.read_text()), json.loads(details.read_text()) if details.exists() else None


def load(args):
    """The bench.run rows under args, as the tables read them (with_details); other metrics.json shapes are named
    on stderr and left out."""
    runs = {p: read(p) for p in find(args)}
    other = [str(p) for p, (m, _) in runs.items() if "by_source" not in m]  # e.g. the batchinfer CLI's metrics.json
    if other:
        print(f"skipped {len(other)} metrics.json not written by bench.run: {', '.join(other)}", file=sys.stderr)
    return [with_details(m, d) for m, d in runs.values() if "by_source" in m]


def text(v):
    """A cell as plain text, for the markdown table."""
    if isinstance(v, Cell):
        return v.text + (f" ({v.mark})" if v.mark else "")
    if isinstance(v, Chips):
        return " ".join(v) or "-"
    return fmt(v)


def table(runs):
    """The index's table as markdown: the same columns, in the same row order."""
    runs = sorted(runs, key=row_key)
    spec = index_columns(runs)
    spec, values, _ = filled(spec, [[value(m) for *_, value in spec] for m in runs])
    lines = ["| " + " | ".join(title for title, *_ in spec) + " |", "|" + "---|" * len(spec)]
    lines += ["| " + " | ".join(map(text, v)) + " |" for v in values]
    return "\n".join(lines)


# HTML ---------------------------------------------------------------------------------------------------

def plain(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def show(value):
    """A value on a page, as stored: the writers already rounded it. Ints get thousands separators, except
    inside a k=v list, where the commas would read as separators."""
    if isinstance(value, dict):
        return ", ".join(f"{k}={plain(v)}" for k, v in value.items()) or "-"
    if isinstance(value, list):
        return ", ".join(map(plain, value)) or "-"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return plain(value)


def breakable(name):
    """A column name that may wrap at its spaces and after each underscore, except beside a one-character word (the
    unit in 'wall s' and 'prefix hit %', the × in '× vLLM'), which stays on its neighbour's line; its text is
    unchanged."""
    groups = []
    for word in str(name).split(" "):
        if groups and (len(word) == 1 or len(groups[-1][-1]) == 1):
            groups[-1].append(word)
        else:
            groups.append([word])
    parts = [Markup("_<wbr/>").join(escape(" ".join(g)).split("_")) for g in groups]
    return Markup(" ").join(Markup('<span class="nw">{}</span>').format(part) if len(g) > 1 else part
                            for g, part in zip(groups, parts))


class Chips(tuple):
    """A cell of k=v settings, drawn as one badge each."""


@dataclass(frozen=True)
class Cell:
    """A cell with its own text, sort key and tooltip, and optionally a link and a mark badge."""
    text: str
    sort: object = ""
    title: str | None = None
    href: str | None = None
    mark: str | None = None

    def __str__(self):
        return self.text


def is_number(v):
    """Whether a cell holds a number, as a value or as a Cell's sort key: its column aligns right."""
    if isinstance(v, Cell):
        v = v.sort
    return isinstance(v, (int, float)) and not isinstance(v, bool)


ENV = Environment(loader=FileSystemLoader(Path(__file__).with_name("templates")), autoescape=True,
                  undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
ENV.filters.update(show=show, fmt=fmt, breakable=breakable)
ENV.tests["chips"] = lambda v: isinstance(v, Chips)
ENV.tests["cell"] = lambda v: isinstance(v, Cell)
PAGE = ENV.get_template("page.html")


ENGINE_GLOSSARY = {**GLOSSARY, **ENGINE}  # for the engine's own record: its rates run on its own clock


def col(text, key=None, filter=False, glossary=GLOSSARY):
    """A column header: its text, the glossary line for key (default: the text itself), whether it gets a filter."""
    return {"text": text, "help": glossary.get(key or text), "filter": filter}


def head(text, href=None, help=None, sub=None, sort=None):
    """A row header: text, or a link with help as its tooltip, or a key with its glossary line; sub is a small line
    under it, sort its sort key (default text)."""
    return {"text": text, "href": href, "help": help, "sub": sub, "sort": text if sort is None else sort}


def kv(caption, d, glossary=GLOSSARY):
    """A key/value table, or None for an empty section."""
    if not d:
        return None
    return {"caption": caption, "header": None, "fmt": False,
            "rows": [{"head": head(k, help=glossary.get(k)), "cells": [v]} for k, v in d.items()]}


def facts(caption, items):
    """A key/value table of (label, glossary key, value) that leaves out values that are None or empty, or None when
    every one is."""
    rows = [{"head": head(label, help=GLOSSARY.get(key)), "cells": [v]} for label, key, v in items
            if v is not None and v != ()]
    return {"caption": caption, "header": None, "fmt": False, "rows": rows} if rows else None


def grid(caption, rows, label=None, glossary=GLOSSARY, titles=None):
    """One row per dict, one column per key any of them has, or None when there are no rows. With a label,
    rows is {name: dict} and each row is headed by its name. titles names a key's column where the key itself
    would not ('accuracy %' for accuracy); its hover text is still the key's."""
    if not rows:
        return None
    named = list(rows.items()) if label else [(None, r) for r in rows]
    keys = list(dict.fromkeys(k for _, d in named for k in d))
    header = [col((titles or {}).get(k, k), k, glossary=glossary) for k in keys]
    return {"caption": caption, "header": [col(label), *header] if label else header, "fmt": False,
            "rows": [{"head": name and head(name), "cells": [d.get(k) for k in keys]} for name, d in named]}


def subtitle(d, keys):
    return " · ".join(show(d[k]) for k in keys if d.get(k) is not None)


def pct(share):
    """A share (0.5781) as a % cell at one decimal ('57.8'), sorting by the %; None stays None."""
    return None if share is None else num(100 * share, 1)


def bench_context(m, details=None, reference=None, vs=None, index=None):
    """A bench.run row's page: titled by its job, subtitled by the knobs it launched with; what it launched, its
    outcome, the engine counters that explain it, accuracy by source and the match against its reference, then the
    charts of its record (details.json). reference is the same job's like-for-like vLLM row, drawn under this run's
    tokens-per-step chart (vllm_reference); vs is its × vLLM (vs_cell); index links back to the index."""
    warnings = []
    if m.get("length_violations"):
        warnings.append(f"{m['length_violations']} requests broke length rules, so this is not a valid performance "
                        f"row. {GLOSSARY['length_violations']}")
    if m.get("over_budget"):
        warnings.append(f"timed run took {m.get('wall_s')} s, over the {m.get('target_s')} s budget for this workload")
    compare = m.get("compare") or {}
    against = compare.get("against")
    by_source = {name: {"kind": s.get("kind"), "n": s.get("n"), "accuracy": pct(s.get("accuracy"))}
                 for name, s in m["by_source"].items()}
    matched = {kind: {k: pct(v) if k in ("match", "identical") else v for k, v in row.items()}
               for kind, row in compare.items() if kind != "against"}
    tables = [facts("Launch", [("prefix reuse", "prefix_reuse", prefix_reuse(m)), ("order", "order", order(m)),
                               ("prefill budget", "prefill_budget", budget_cell(m)), ("layers", "layers", layers(m)),
                               ("graphs", "graphs", graphs(m)), ("max seqs", "max_num_seqs", max_seqs(m)),
                               ("other opts", "other_opts", other_opts(m)), ("requests", "requests", m.get("requests")),
                               ("commit", "commit", commit_cell(m)), ("ran", "ran", ran_cell(m))]),
              facts("Outcome", [("wall s", "wall_s", m["wall_s"]), ("× vLLM", "vs_vllm", vs),
                                ("accuracy %", "accuracy_pct", accuracy_pct(m)),
                                ("out tok/s", "output_tok_per_s", m.get("output_tok_per_s")),
                                ("load s", "load_s", m.get("load_s"))]),
              facts("Engine", [(label, key, value(m)) for label, key, value in ENGINE_FACTS]),
              facts("Utilisation", [(label, key, value(m)) for label, key, value in UTILISATION_FACTS]),
              grid("Accuracy by source", by_source, "source", titles={"accuracy": "accuracy %"}),
              grid(f"Match against {Path(against).parent.name}" if against else "Match", matched, "kind",
                   titles={"match": "match %", "identical": "identical %"})]
    title = " · ".join(t for t in (engine(m), Path(m["workload"]).stem, model_name(m["model"]),
                                   gpu_slug(m["gpu"]) if m.get("gpu") else None) if t)
    record = chart_context(details, reference) if details else NO_CHARTS
    return {"title": title, "subtitle": launch_line(m), "index": index, "warnings": warnings,
            "tables": [t for t in tables if t], "note": trace_note(details, "details.json") if details else None,
            **record}


NO_CHARTS = {"charts": [], "chart_data": None, "chart_notes": []}
TRACES = ("meta", "groups", "steps", "request_trace")  # drawn or tabled on their own, never as a key/value table


def trace_note(m, source):
    n_steps = len(next(iter((m.get("steps") or {}).values()), []))
    n_requests = len(next(iter((m.get("request_trace") or {}).values()), []))
    recorded = [f"{n} {what}" for n, what in ((n_steps, "steps"), (n_requests, "requests")) if n]
    return f"{' and '.join(recorded)} recorded; the traces are in {source}." if recorded else None


def chart_context(m, reference=None):
    """The page's charts (bench/charts.py): the cards to lay out and, once, the specs and the data they draw."""
    drawn, datasets, notes = charts(m, reference)
    if not drawn:
        return dict(NO_CHARTS, chart_notes=notes)
    return {"charts": [{k: c[k] for k in ("id", "title", "description", "wide")} for c in drawn],
            "chart_data": {"datasets": {name: to_columns(rows) for name, rows in datasets.items()},
                           "specs": {c["id"]: {"light": c["light"], "dark": c["dark"]} for c in drawn}},
            "chart_notes": notes}


def engine_context(m, source="metrics.json", reference=None):
    meta = dict(m.get("meta") or {})
    args = meta.pop("args", None)
    rest = {k: v for k, v in m.items() if k not in TRACES}
    g = ENGINE_GLOSSARY
    tables = [kv("Configuration", meta, g), kv("Command-line arguments", args, g),
              *[kv(k, v, g) for k, v in rest.items() if isinstance(v, dict)],
              kv("Other", {k: v for k, v in rest.items() if not isinstance(v, dict)}, g),  # nothing is dropped unseen
              grid("Groups", m.get("groups"), glossary=g)]
    return {"subtitle": subtitle(meta, ("input", "timestamp", "model", "engine", "order")), "warnings": [],
            "tables": [t for t in tables if t], "note": trace_note(m, source), **chart_context(m, reference)}


def render(m, name, files, details=None, reference=None, vs=None, index=None):
    """One run's page, linking the sibling files named in files: a bench.run row (bench_context, titled by what it
    ran), or a python -m batchinfer run record titled by its directory name, name. details is the backend's
    details.json, drawn as charts; reference is vllm_reference()'s; vs is the row's × vLLM; index is the index
    page's href from this one (none for a page bench.run writes on its own)."""
    if "by_source" in m:
        return PAGE.render(files=files, **bench_context(m, details, reference, vs, index))
    return PAGE.render(title=name, files=files, index=index, **engine_context(m))


def vllm_reference(path, m, details):
    """The same job's like-for-like vLLM row for a batchinfer bench.run row at path (its metrics.json), as
    {"record": its details.json, "prefix_cache": on or off}: the row beside this one labelled vllm when the run
    shared prefixes, vllm-enable_prefix_caching=False when it did not (README "Results"), and only when it ran the
    same workload file, model and GPU and its trace has steps. Else None."""
    if "by_source" not in m or not details or engine_of(details) != "batchinfer":
        return None
    prefix_cache = bool((details.get("policy") or {}).get("prefix_sharing"))
    ref = path.parent.parent / ("vllm" if prefix_cache else "vllm-enable_prefix_caching=False")
    try:
        ref_m = json.loads((ref / "metrics.json").read_text())
        ref_d = json.loads((ref / "details.json").read_text())
    except (OSError, ValueError):
        return None
    if "by_source" not in ref_m or same_job(ref_m) != same_job(m) or not ref_d.get("steps"):
        return None
    return {"record": ref_d, "prefix_cache": prefix_cache}


def engine_config(m):
    meta = m.get("meta") or {}
    config = {k: v for k, v in meta.items() if k not in META_NOISE}
    config.update((k, v) for k, v in (meta.get("args") or {}).items() if k not in ARGS_NOISE)
    return config


def engine_rows(runs, link):
    """One index row per engine run. The config cell holds engine, order and the settings that differ
    between the engine runs listed; flags that are the same everywhere, or did not apply, stay on the run page."""
    configs = {p: engine_config(m) for p, m in runs.items()}
    varied = {k for c in configs.values() for k in c if len({repr(d.get(k)) for d in configs.values()}) > 1}
    rows = []
    for p, m in runs.items():
        meta = m.get("meta") or {}
        config = {k: v for k, v in configs[p].items() if k in ("engine", "order") or k in varied}
        rows.append({"head": link(p), "cells": [meta.get("model"), Chips(f"{k}={plain(v)}" for k, v in config.items()),
                                                 (m.get("volume") or {}).get("requests"),
                                                 (m.get("rates") or {}).get("total_tok_per_s"), meta.get("timestamp")]})
    return rows


# What a row launched ---------------------------------------------------------------------------------------

# The backend classes before the engine rename, whose rows record no engine: the same two engines.
LEGACY_ENGINES = {"stepbackend": "batchinfer", "enginebackend": "naive"}
LAUNCH_OPTS = ("prefix_sharing", "enable_prefix_caching", "order", "prefill_budget", "fused_layers", "cuda_graphs",
               "fa_version", "enforce_eager", "max_num_seqs")  # --opt values with a column
ENGINE_RANK = {"naive": 0, "vllm": 1, "batchinfer": 2}  # baselines first


def engine(m):
    """The engine a row ran, by its registry name: bench.run records it; rows written before it did get their
    backend's class name, the pre-rename ones mapped to the engine's current name."""
    if m.get("engine"):
        return m["engine"]
    name = m["backend"].rsplit(":", 1)[-1].rsplit(".", 1)[-1].lower()
    return LEGACY_ENGINES.get(name, name)


def stat(m, key):
    return (m.get("backend_stats") or {}).get(key)


def with_details(m, details):
    """m as the tables read it: what its details.json adds where the row's stats lack it. A naive row's order (the
    naive engine's stats carry it only since its FLAT_KEYS gained it; its default was max_tokens_desc until the
    engine rename, input since), and for a vLLM row its engine iterations as steps and their mean decode rows as
    decode batch, from its trace, counted as batchinfer counts its own steps (batchinfer/metrics.py)."""
    if "by_source" not in m or not details:
        return m
    stats = dict(m.get("backend_stats") or {})
    order_ran = (details.get("policy") or {}).get("order")
    if stats.get("order") is None and order_ran is not None:
        stats["order"] = order_ran
    trace = details.get("steps")
    decode = trace.get("decode_rows") if engine(m) == "vllm" and isinstance(trace, dict) else None
    if decode and stats.get("steps") is None:
        rows = [d for d in decode if d]
        stats.update(steps=len(decode), mean_decode_batch=round(sum(rows) / len(rows), 2) if rows else None)
    return {**m, "backend_stats": stats}


def prefix_reuse(m):
    """How the run reused the KV of shared prompt prefixes: 'global trie' (batchinfer's prefix_sharing),
    'prefix cache' (vLLM's enable_prefix_caching, on unless turned off) or 'off'; the naive engine never reuses."""
    name = engine(m)
    if name == "vllm":
        return "prefix cache" if m["opts"].get("enable_prefix_caching", True) else "off"
    sharing = stat(m, "prefix_sharing")
    sharing = m["opts"].get("prefix_sharing") if sharing is None else sharing
    if sharing is None:
        return "off" if name == "naive" else None
    return "global trie" if sharing else "off"


def order(m):
    """The admission order our engines ran (vLLM takes requests first come, first served)."""
    return stat(m, "order") or m["opts"].get("order")


def budget_cell(m):
    """batchinfer's prefill budget: the tokens a step may prefill ('583', which auto chose), or 'adaptive from 583'
    when the scheduler could raise it from there; sorts by the tokens."""
    budget = stat(m, "prefill_budget")
    if budget is None:
        return None
    adaptive = stat(m, "prefill_budget_adaptive") or m["opts"].get("prefill_budget") == "adaptive"  # older rows: opts
    return Cell(f"adaptive from {show(budget)}" if adaptive else show(budget), sort=budget)


def layers(m):
    """The forward a row ran. batchinfer: 'fused' (its Qwen3 layers on vLLM's fused kernels) or 'HF' (HF's layer
    modules), with ' · FA3' for FlashAttention-3; rows recorded before the engine said so fall back to their --opt
    values, and to HF (the only forward then). vLLM: 'compiled' (torch.compile) or 'eager' (enforce_eager). naive: HF."""
    name = engine(m)
    if name == "vllm":
        return "eager" if m["opts"].get("enforce_eager") else "compiled"
    if name != "batchinfer":
        return "HF" if name == "naive" else None
    fused = stat(m, "fused_layers")
    fused = m["opts"].get("fused_layers") is True if fused is None else fused
    fa3 = (stat(m, "fa_version") or m["opts"].get("fa_version")) == 3
    return ("fused" if fused else "HF") + (" · FA3" if fa3 else "")


def graphs(m):
    """CUDA graphs. batchinfer: 'decode 81%' (the share of steps replayed from a graph: its decode-only steps) or
    'off', falling back to --opt values for older rows. vLLM: 'on' (full and piecewise) unless enforce_eager. naive:
    off."""
    name = engine(m)
    if name == "vllm":
        return "off" if m["opts"].get("enforce_eager") else "on"
    if name != "batchinfer":
        return "off" if name == "naive" else None
    on = stat(m, "cuda_graphs")
    on = m["opts"].get("cuda_graphs") is True if on is None else on
    if not on:
        return "off"
    pct = stat(m, "graph_step_pct")
    return f"decode {pct:.0f}%" if pct is not None else "decode"


def max_seqs(m):
    """vLLM's cap on requests running at once, when the run set it (max_num_seqs); None is vLLM's default."""
    return m["opts"].get("max_num_seqs") if engine(m) == "vllm" else None


def other_opts(m):
    """The --opt values without a column of their own, as k=v badges."""
    return Chips(f"{k}={plain(v)}" for k, v in m["opts"].items() if k not in LAUNCH_OPTS)


def launch_line(m):
    """What a row launched with, as a run page's subtitle, so the same job's pages tell apart at a glance: 'global
    trie · order prefix_dfs · prefill budget adaptive from 583', 'prefix cache', 'prefix reuse off · order input'."""
    reuse, budget = prefix_reuse(m), budget_cell(m)
    build = {"batchinfer": f"{layers(m)} layers", "vllm": "eager (no compile, no graphs)"
             if m["opts"].get("enforce_eager") else None}.get(engine(m))
    graphed = show(graphs(m)) if engine(m) == "batchinfer" and graphs(m) != "off" else None
    seqs = max_seqs(m)
    parts = ["prefix reuse off" if reuse == "off" else reuse, order(m) and f"order {order(m)}",
             budget and f"prefill budget {budget.text}", build, graphed and f"graphs {graphed}",
             seqs and f"max_num_seqs {seqs}", *other_opts(m)]
    return " · ".join(p for p in parts if p) or None


# The numbers that tell a row's tradeoffs ---------------------------------------------------------------------

def same_job(m):
    return m["workload_sha256"], m["model"], m.get("gpu")


def vllm_walls(runs):
    """{job: wall s} of each job's vLLM row with its defaults (prefix cache on): what × vLLM divides by."""
    return {same_job(m): m["wall_s"] for m in runs if engine(m) == "vllm" and not m["opts"]}


def num(value, digits):
    """A number at a fixed count of decimals, so a column's decimal points line up; sorts by the value. None stays
    None."""
    return None if value is None else Cell(f"{value:,.{digits}f}", sort=value)


def vs_cell(m, walls, href=None):
    """× vLLM at two decimals, its tooltip naming the vLLM wall s it divides by; href links that vLLM row's page."""
    ref = walls.get(same_job(m))
    if not ref:
        return None
    return Cell(f"{m['wall_s'] / ref:.2f}", sort=round(m["wall_s"] / ref, 2), href=href,
                title=f"wall s ÷ {ref:,.2f} s, vLLM with its prefix cache on")


def wall_cell(m):
    """wall s at two decimals, marked invalid when requests broke the length rules: then it is not a valid
    performance row."""
    bad = m.get("length_violations")
    return Cell(f"{m['wall_s']:,.2f}", m["wall_s"], f"{bad} requests broke the length rules" if bad else None,
                mark="invalid" if bad else None)


def accuracy_pct(m):
    """Correct answers over every request of a scored source, in % (every request of such a source carries its
    scorer: bench/sources.py)."""
    scored = [s for s in m["by_source"].values() if "accuracy" in s]
    n = sum(s["n"] for s in scored)
    return round(100 * sum(s["accuracy"] * s["n"] for s in scored) / n, 1) if n else None


def block_ceiling_pct(m):
    stats = m.get("backend_stats") or {}
    v = next((stats[k] for k in stats if k.startswith("ideal_prefix_reuse_page")), None)
    return None if v is None else round(100 * v, 1)


def prefix_hit_cell(m):
    """prefix hit %, with the block-level ceiling it is judged against as its tooltip."""
    hit, ceiling = stat(m, "prefix_hit_pct"), block_ceiling_pct(m)
    if hit is None:
        return None
    return Cell(f"{hit:.1f}", hit, f"block ceiling {ceiling}%" if ceiling is not None else None)


def ms_per_step(m):
    steps = stat(m, "steps")
    return round(1000 * m["wall_s"] / steps, 1) if steps else None


ENGINE_FACTS = (  # (label, glossary key, value): the engine counters a run's page shows, where the row has them
    ("prefix hit %", "prefix_hit_pct", lambda m: stat(m, "prefix_hit_pct")),
    ("block ceiling %", "block_ceiling_pct", block_ceiling_pct),
    ("steps", "steps", lambda m: stat(m, "steps")),
    ("ms/step", "ms_per_step", ms_per_step),
    ("decode batch", "mean_decode_batch", lambda m: stat(m, "mean_decode_batch")),
    ("prefill budget peak", "prefill_budget_peak", lambda m: stat(m, "prefill_budget_peak")),
    ("KV occupancy %", "kv_occupancy_pct", lambda m: stat(m, "kv_occupancy_pct")),
    ("padding waste %", "padding_waste_pct", lambda m: stat(m, "padding_waste_pct")),
    ("dense MFU %", "dense_mfu_pct", lambda m: stat(m, "dense_mfu_pct")),
    ("peak memory GiB", "peak_mem_reserved_gb", lambda m: stat(m, "peak_mem_reserved_gb")),
)


def used(key):
    return lambda m: (m.get("utilisation") or {}).get(key)


UTILISATION_FACTS = (  # (label, glossary key, value): how much of the card the run used, counted alike for every engine
    ("GPU util %", "gpu_util_mean", lambda m: m.get("gpu_util_mean")),
    ("MFU %", "mfu_pct", lambda m: m.get("mfu_pct")),
    ("MBU %", "mbu_pct", lambda m: m.get("mbu_pct")),
    ("model TFLOP", "model_tflop", used("model_tflop")),
    ("attention FLOP %", "attention_flop_pct", used("attention_flop_pct")),
    ("model GB moved", "model_gb_moved", used("model_gb_moved")),
    ("forward passes", "forward_passes", used("forward_passes")),
    ("peak bf16 TFLOPS", "peak_bf16_tflops", used("peak_bf16_tflops")),
    ("peak HBM GB/s", "peak_hbm_gb_per_s", used("peak_hbm_gb_per_s")),
)


# The index: one row per bench.run run -----------------------------------------------------------------------

def index_columns(runs):
    """(title, glossary key, filter, value) per column: what launched, the engine first (it heads the row and links
    its page), then the numbers that tell its tradeoffs: how fast (wall s, × vLLM), how right (accuracy %), why
    (prefix hit %, steps, ms/step, decode batch) and how much of the card it used (GPU util %, MFU %, MBU %), each at
    a fixed count of decimals; the commit, provenance rather than a knob, comes last."""
    walls = vllm_walls(runs)
    return [
        ("engine", "engine", True, engine),
        ("workload", "workload", True, lambda m: Path(m["workload"]).stem),
        ("model", "model", True, lambda m: model_name(m["model"])),
        ("GPU", "GPU", True, lambda m: gpu_slug(m["gpu"]) if m.get("gpu") else None),
        ("prefix reuse", "prefix_reuse", True, prefix_reuse),
        ("order", "order", True, order),
        ("prefill budget", "prefill_budget", False, budget_cell),
        ("layers", "layers", True, layers),
        ("graphs", "graphs", True, graphs),
        ("max seqs", "max_num_seqs", True, max_seqs),
        ("wall s", "wall_s", False, wall_cell),
        ("× vLLM", "vs_vllm", False, lambda m: vs_cell(m, walls)),
        ("accuracy %", "accuracy_pct", False, lambda m: num(accuracy_pct(m), 1)),
        ("prefix hit %", "prefix_hit_pct", False, prefix_hit_cell),
        ("steps", "steps", False, lambda m: stat(m, "steps")),
        ("ms/step", "ms_per_step", False, lambda m: num(ms_per_step(m), 1)),
        ("decode batch", "mean_decode_batch", False, lambda m: num(stat(m, "mean_decode_batch"), 1)),
        ("GPU util %", "gpu_util_mean", False, lambda m: num(m.get("gpu_util_mean"), 1)),
        ("MFU %", "mfu_pct", False, lambda m: num(m.get("mfu_pct"), 1)),
        ("MBU %", "mbu_pct", False, lambda m: num(m.get("mbu_pct"), 1)),
        ("commit", "commit", True, commit_cell),
    ]


def filled(spec, values):
    """(spec, values, hidden titles) without the columns no row fills: a column blank in every row tells nothing,
    so the table leaves it out and its note names it. values holds a row's cells in spec's order."""
    keep = [i for i in range(len(spec)) if any(v[i] is not None and v[i] != () for v in values)]
    return ([spec[i] for i in keep], [[v[i] for i in keep] for v in values],
            [title for i, (title, *_) in enumerate(spec) if i not in keep])


def workload_rank(m):
    preset = m.get("preset")
    return (SUITE_PRESETS.index(preset) if preset in SUITE_PRESETS else len(SUITE_PRESETS)), Path(m["workload"]).stem


def row_key(m):
    """Rows by job (workloads in the suite's order), then baselines first: naive, vLLM with its cache off then on,
    then batchinfer with the global trie, then its other launches."""
    name = engine(m)
    return (workload_rank(m), model_name(m["model"]), m.get("gpu") or "", ENGINE_RANK.get(name, len(ENGINE_RANK)),
            name, prefix_reuse(m) or "", str(order(m) or ""), json.dumps(m["opts"], sort_keys=True, default=str),
            m.get("branch") or "")


def options(cells):
    """A filter's choices: each distinct cell text once, newest first for time-keyed cells (commit), else sorted."""
    if any(isinstance(c, Cell) for c in cells):
        keyed = {show(c): c.sort if isinstance(c, Cell) else "" for c in cells}
        return sorted(keyed, key=lambda t: keyed[t] or "", reverse=True)
    return sorted({show(c) for c in cells})


def bench_table(bench, link):
    """bench.run rows for the index in row_key order, one band per job, with a filter over each column marked for
    one, and the note naming the columns no row fills (None when every column shows)."""
    spec = index_columns(list(bench.values()))
    items = sorted(bench.items(), key=lambda pm: row_key(pm[1]))
    spec, values, hidden = filled(spec, [[value(m) for *_, value in spec] for _, m in items])
    rows = [{"head": link(p, v[0]), "cells": v[1:], "group": "|".join(map(str, same_job(m)))}
            for (p, m), v in zip(items, values)]
    filters = [{"col": i, "label": title, "options": options([v[i] for v in values])}
               for i, (title, _, filter, _) in enumerate(spec) if filter]  # col 0 is the row header, the engine
    note = f"Blank in every run, so not shown: {', '.join(hidden)}." if hidden else None
    numeric = [any(is_number(v[i]) for v in values) for i in range(len(spec))]  # right-aligned, blanks and header too
    return {"caption": None, "id": "bench", "fmt": True, "filters": filters, "numeric": numeric[1:],
            "header": [dict(col(title, key, filter), numeric=n) for (title, key, filter, _), n in zip(spec, numeric)],
            "rows": rows}, note


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def index_subtitle(bench, cli):
    """What the index holds, and how to read its table."""
    sentences = []
    if bench:
        count = lambda f: len({f(m) for m in bench.values()})  # noqa: E731
        sentences.append(f"{plural(len(bench), 'run')}: {plural(count(lambda m: Path(m['workload']).stem), 'workload')}, "
                     f"{plural(count(lambda m: m['model']), 'model')}, {plural(count(lambda m: m.get('gpu')), 'GPU')}. "
                     "Each band is one job (workload, model, GPU), baselines first. × vLLM is wall s ÷ the job's "
                     "vLLM wall s with its prefix cache on: 1.00 is parity, lower is faster. Hover an underlined "
                     "name for what it measures, click a column to sort; an engine opens its run's page.")
    if cli:
        sentences.append(f"{plural(len(cli), 'engine run')} (python -m batchinfer run) below.")
    return " ".join(sentences)


def index_page(root, runs):
    """One row per bench.run run, then one per engine run, each linking to its report.html."""
    def href(p):
        return quote(f"{p.parent.relative_to(root).as_posix()}/report.html", safe="/=")

    def link(p, text=None):
        """The run's page: named by text with its directory as the tooltip, or by its label over the directory."""
        rel = p.parent.relative_to(root).as_posix()
        if text is not None:
            return head(text, href=href(p), help=rel)
        where, _, name = rel.rpartition("/")
        return head(name, href=href(p), sub=where, sort=rel)

    bench_runs = {p: m for p, m in runs.items() if "by_source" in m}
    cli_runs = {p: m for p, m in runs.items() if "by_source" not in m}
    tables, note = [], None
    if bench_runs:
        bench, note = bench_table(bench_runs, link)
        tables.append(bench)
    if cli_runs:
        tables.append({"caption": "Engine runs (python -m batchinfer run)", "fmt": False,
                       "header": [col("run", "run_dir"), col("model"), col("config", "engine_config"), col("requests"),
                                  col("total tok/s", "total_tok_per_s", glossary=ENGINE_GLOSSARY),
                                  col("timestamp", glossary=ENGINE_GLOSSARY)],
                       "rows": engine_rows(cli_runs, link)})
    return PAGE.render(title="Benchmark results", subtitle=index_subtitle(bench_runs, cli_runs), index=None,
                       warnings=[], **NO_CHARTS, tables=tables, note=note, files=[])


# Times ----------------------------------------------------------------------------------------------------

def utc(iso):
    """An ISO 8601 time as a UTC string that sorts as text ('2026-09-27T16:24:00Z'), or None."""
    try:
        return datetime.fromisoformat(iso).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if iso else None
    except ValueError:
        return None


def when(iso):
    """'2026-09-27T17:24:03+01:00' -> '09-27 16:24' (UTC), or None."""
    u = utc(iso)
    return u and f"{u[5:10]} {u[11:16]}"


def commit_cell(m):
    """The commit a row ran, @ the branch it came from when that was not main, and when the commit was made:
    'ae8fb35 · 09-27 16:24', '66691b5@feature-x · 09-27 17:50'; sorts by that time."""
    sha, t = m.get("git_sha"), m.get("git_commit_time")
    if not sha:
        return None
    name = f"{sha}@{m['branch']}" if m.get("branch") else sha
    return Cell(f"{name} · {when(t)}" if when(t) else name, sort=utc(t) or "",
                title=f"committed {utc(t)}" if utc(t) else None)


def ran_cell(m):
    """When the row's timed pass finished."""
    return Cell(when(m.get("timestamp")), sort=utc(m.get("timestamp"))) if when(m.get("timestamp")) else None


def page_href(run_dir, start):
    """The href of run_dir's report.html from a page in start."""
    return quote(os.path.relpath(run_dir / "report.html", start), safe="/=")


def write_html(root):
    """report.html beside every metrics.json below root and root/index.html linking them; returns the run count."""
    runs = {}
    for p in find([root]):
        m, details = read(p)
        runs[p] = with_details(m, details), details
    bench = {p: m for p, (m, _) in runs.items() if "by_source" in m}
    walls = vllm_walls(bench.values())
    vllm_pages = {same_job(m): p.parent for p, m in bench.items() if engine(m) == "vllm" and not m["opts"]}
    for p, (m, details) in runs.items():
        files = [p.name, *[s for s in SIBLINGS if p.with_name(s).exists()]]
        ref = vllm_pages.get(same_job(m)) if p in bench else None
        vs = vs_cell(m, walls, ref and ref != p.parent and page_href(ref, p.parent)) if p in bench else None
        p.with_name("report.html").write_text(render(m, p.parent.name, files, details, vllm_reference(p, m, details),
                                                     vs, quote(os.path.relpath(root / "index.html", p.parent))))
    (root / "index.html").write_text(index_page(root, {p: m for p, (m, _) in runs.items()}))
    return len(runs)


class Handler(SimpleHTTPRequestHandler):
    # outputs.jsonl has no registered type, so browsers would download it instead of showing it
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".jsonl": "text/plain; charset=utf-8"}


def make_server(root, bind, port):
    return ThreadingHTTPServer((bind, port), partial(Handler, directory=str(root)))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bench.report", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", type=Path, default=[Path("results")],
                   help="results directories or metrics.json files (default results/)")
    p.add_argument("--html", action="store_true",
                   help="also write report.html beside every metrics.json and index.html in the one directory given")
    p.add_argument("--serve", type=int, metavar="PORT", help="--html, then serve that directory until Ctrl-C")
    p.add_argument("--bind", default="127.0.0.1", metavar="HOST",
                   help="interface for --serve (default %(default)s); an address other machines can reach, to "
                        "view the report from them")
    args = p.parse_args(argv)
    if args.serve is not None and not args.bind:  # "" listens on every interface, e.g. an unset shell variable
        p.error("--bind needs an address")
    if args.html or args.serve is not None:
        if len(args.paths) != 1 or not args.paths[0].is_dir():
            p.error("--html and --serve take exactly one results directory")
        root = args.paths[0]
        print(f"wrote {write_html(root)} report.html and {root / 'index.html'}", file=sys.stderr)
    print(table(load(args.paths)))
    if args.serve is not None:
        server = make_server(root, args.bind, args.serve)
        print(f"serving {root} at http://{args.bind}:{server.server_address[1]}/ (Ctrl-C stops)", file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
