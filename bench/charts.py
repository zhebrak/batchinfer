"""Charts for a run's page, drawn from the engine's own record: what each step did and when each request ran.

The record is a details.json (a bench.run engine backend, or the vLLM backend with its trace on) or python -m
batchinfer run's metrics.json. Three of its sections are drawn:
- steps: the per-step trace (the batchinfer engine's steps, or vLLM's iterations);
- request_trace: one entry per request (batchinfer/metrics.py REQUEST_FIELDS; the vLLM backend writes the same);
- groups: the naive engine's per-group trace.
Presentation only: rows() reshapes what the record holds and charts() draws it as Vega-Lite specs. Nothing here
is measured and nothing is decided.

Every time axis counts seconds since the backend received the job, the same origin on every page. batchinfer's
engines time from their inference timer, which starts after analysis and policy, so their times are shifted by
timing.analysis_s + timing.policy_s; vLLM's are from generate()'s start. A record from before end_ms existed has
its steps drawn against the step index instead, and has no request charts.

What BatchLLM (arXiv 2412.03594) plots, drawn here: tokens per step split into prefill and decode (its Figs 2 and
10, whose "valleys" are steps too small to fill the GPU), the admission order by decode ratio (its Eq. 2) and the
length distributions (Figs 5 and 8). What it does not plot and we have: requests in flight, KV in use, step time,
and every request's timeline.

Colour follows the entity on every chart (the dataviz skill's reference palette, light and dark steps): blue is
decode (decode rows, generate requests, decode-only steps), orange is prefill (prefill tokens, prefill_only
requests, prefill-only steps), aqua is what mixes or holds both (mixed steps, label requests, admitted requests),
violet is KV, grey is waiting or waste. The KV chart's lines are a set of their own (held violet, written aqua,
shared orange, pinned grey): blue beside violet fails the colour-vision check on the dark surface.
"""
from bisect import bisect_right

from batchinfer.kv import BLOCK_SIZE

SCHEMA = "https://vega.github.io/schema/vega-lite/v6.json"
PALETTES = {
    "light": {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "violet": "#4a3aa7", "grey": "#898781",
              "text": "#0b0b0b", "secondary": "#52514e", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "violet": "#9085e9", "grey": "#898781",
             "text": "#ffffff", "secondary": "#c3c2b7", "grid": "#343a40", "axis": "#495057"},
}
KINDS = {"generate": "blue", "prefill_only": "orange", "label": "aqua"}  # analysis_kind
STEP_KINDS = {"decode-only": "blue", "prefill-only": "orange", "mixed": "aqua"}
TIME_AXIS = "s since the job started"
NAIVE_FINISH = "Naive engine: finish times are apportioned within each group's measured decode time, as latency_s is."
LEGACY_ENGINES = {"step": "batchinfer", "static": "naive"}  # the engines' names in records before the rename


# rows ---------------------------------------------------------------------------------------------------

def engine_of(record):
    """batchinfer | naive | vllm. Records from before the rename carry step | static; records from before the policy
    named its engine are told apart by their trace."""
    named = (record.get("policy") or {}).get("engine") or (record.get("meta") or {}).get("engine")
    named = LEGACY_ENGINES.get(named, named)
    return named or ("batchinfer" if record.get("steps") else "naive" if record.get("groups") else None)


def columnar(cols):
    """{field: [values]} -> [{field: value}], the traces' on-disk shape to one dict per entry."""
    return [dict(zip(cols, values)) for values in zip(*cols.values())] if cols else []


def to_columns(rows):
    """[{field: value}] -> {field: [values]}, the inverse, for shipping: each key once, not once per row. A field a
    row lacks is None there."""
    fields = list(dict.fromkeys(k for r in rows for k in r))
    return {f: [r.get(f) for r in rows] for f in fields}


def origin_ms(record, engine):
    """Where the job started, in the engine's own ms: before analysis and policy for batchinfer, 0 for vLLM."""
    if engine == "vllm":
        return 0.0
    t = record.get("timing") or {}
    return 1000 * ((t.get("analysis_s") or 0.0) + (t.get("policy_s") or 0.0))


def _s(ms, origin):
    return None if ms is None else round((ms + origin) / 1000, 4)


def _pct(part, whole):
    return None if part is None or not whole else round(100 * part / whole, 2)


def request_rows(record, engine):
    origin = origin_ms(record, engine)
    rows = []
    for r in columnar(record.get("request_trace")):
        prompt, hits = r["prompt_len"], r.get("prefix_hit_tokens") or 0
        rows.append({"idx": r["idx"], "id": r["id"], "kind": r["analysis_kind"], "group": r.get("group"),
                     "prompt_len": prompt, "max_tokens": r["max_tokens"], "output_tokens": r["output_tokens"],
                     "prefix_hit_tokens": hits, "computed_prompt_tokens": prompt - hits,
                     "finish_reason": r["finish_reason"],
                     "decode_ratio": round((r["max_tokens"] - 1) / max(prompt, 1), 4),
                     "admitted_step": r.get("admitted_step"), "finished_step": r.get("finished_step"),
                     **{f"{event}_s": _s(r.get(f"{event}_ms"), origin)
                        for event in ("admitted", "prefill_start", "first_token", "finished")}})
    return rows


def step_rows(record, engine, requests):
    """One row per step, with x0/x1 its start and end in seconds since the job started, or its index when the
    record predates end_ms. The KV shares are of the pool's usable blocks (batchinfer) or vLLM's own usage."""
    steps = columnar(record.get("steps"))
    timed = bool(steps) and all(s.get("end_ms") is not None for s in steps)
    origin = origin_ms(record, engine)
    capacity = ((record.get("volume") or {}).get("kv_blocks") or 1) - 1  # block 0 is the pad block
    budget = (record.get("policy") or {}).get("prefill_budget")
    admitted_at = sorted(r["admitted_step"] for r in requests if r.get("admitted_step") is not None)
    rows, prev = [], 0.0
    for k, s in enumerate(steps):
        d, p = s.get("decode_rows") or 0, s.get("prefill_tokens") or 0
        row = {"step": k, "decode_rows": d, "prefill_tokens": p, "prefill_rows": s.get("prefill_rows"), "tokens": d + p,
               "kind": "decode-only" if d and not p else "prefill-only" if p and not d else "mixed"}
        if timed:
            row.update(x0=_s(prev, origin), x1=_s(s["end_ms"], origin), interval_ms=round(s["end_ms"] - prev, 3))
            prev = s["end_ms"]
        else:
            row.update(x0=k, x1=k + 1, interval_ms=None)
        if engine == "vllm":
            # vLLM allocates blocks as tokens are written: its usage is the counterpart of our written, not held
            row.update(running=s.get("running"), waiting=s.get("waiting"), kv_in_use_pct=s.get("kv_cache_usage_pct"))
        else:
            row.update(admitted=s.get("admitted"), budget=s.get("step_prefill_budget", budget))
            if admitted_at:  # requests whose KV was not reserved by the end of step k
                row["waiting"] = len(requests) - bisect_right(admitted_at, k)
            if s.get("kv_reserved_tokens") is not None:  # sampled with kv_written_tokens, at step start
                row["kv_held_pct"] = _pct(s["kv_reserved_tokens"], BLOCK_SIZE * capacity)
                row["kv_written_pct"] = _pct(s.get("kv_written_tokens"), BLOCK_SIZE * capacity)
            elif s.get("free_blocks") is not None:  # older traces: after commit
                row["kv_held_pct"] = _pct(capacity - s["free_blocks"], capacity)
            row.update(kv_trie_pct=_pct(s.get("trie_blocks"), capacity), kv_pinned_pct=_pct(s.get("pinned_blocks"), capacity))
        rows.append(row)
    return rows, timed


def group_rows(record):
    """Naive engine: one row per fixed group, every figure measured or exact (needs prompt_tokens per group)."""
    rows = []
    for g in record.get("groups") or []:
        if g.get("prompt_tokens") is None:
            return []  # a record from before groups carried their real prompt tokens
        live = g["decode_steps"] * g["n"] - g["wasted_slots"]
        padding = g["n"] * g["padded_len"] - g["prompt_tokens"]
        rows.append({"group": g["group"], "n": g["n"], "padded_len": g["padded_len"], "prefill": g["prompt_tokens"],
                     "padding": padding, "decode": live, "dead_slots": g["wasted_slots"],
                     "wasted": padding + g["wasted_slots"], "prefill_s": g.get("prefill_s"),
                     "decode_s": g.get("decode_s"), "decode_steps": g["decode_steps"]})
    return rows


def timeline_rows(requests):
    """The request timeline's segments, requests ranked by prefill start."""
    timed = [r for r in requests if r["prefill_start_s"] is not None and r["finished_s"] is not None]
    timed.sort(key=lambda r: (r["prefill_start_s"], r["idx"]))
    rows = []
    for rank, r in enumerate(timed):
        base = {"rank": rank, "id": r["id"], "kind": r["kind"], "prompt_len": r["prompt_len"],
                "output_tokens": r["output_tokens"]}
        segments = (("KV reserved, waiting", r["admitted_s"], r["prefill_start_s"]),
                    ("prefill", r["prefill_start_s"], r["first_token_s"]),
                    ("decode", r["first_token_s"], r["finished_s"]))
        rows += [{**base, "phase": phase, "x0": a, "x1": b} for phase, a, b in segments
                 if a is not None and b is not None and b > a]
    return rows


# specs --------------------------------------------------------------------------------------------------

def config(p):
    return {"background": "transparent", "font": "system-ui, -apple-system, 'Segoe UI', sans-serif",
            "view": {"stroke": None},
            "axis": {"labelColor": p["secondary"], "titleColor": p["secondary"], "gridColor": p["grid"],
                     "domainColor": p["axis"], "tickColor": p["axis"], "labelFontSize": 11, "titleFontSize": 11,
                     "titleFontWeight": "normal"},
            "legend": {"labelColor": p["secondary"], "titleColor": p["secondary"], "orient": "top",
                       "titleFontWeight": "normal", "labelFontSize": 11, "titleFontSize": 11,
                       "symbolBaseFillColor": p["secondary"], "symbolBaseStrokeColor": p["secondary"]},
            "title": {"color": p["text"], "fontSize": 12, "fontWeight": "normal", "anchor": "start"}}


def spec(p, body, height=220):
    return {"$schema": SCHEMA, "width": "container", "height": height, "config": config(p), **body}


def colour(p, field, entities, title=None):
    """Colour by field, whose values are the entities in their fixed order, each in its own palette slot."""
    return {"field": field, "type": "nominal", "title": title,
            "scale": {"domain": list(entities), "range": [p[slot] for slot in entities.values()]}}


def q(field, title, **extra):
    return {"field": field, "type": "quantitative", "title": title, **extra}


def tip(*fields):
    return [{"field": f, "type": "quantitative" if n else "nominal", "title": t, **({"format": n} if n else {})}
            for f, t, n in fields]


def log_ticks(values):
    """Tick values for a log axis over values: 1-3-10 steps inside their range, so the grid stays sparse."""
    positive = [v for v in values if v and v > 0]
    if not positive:
        return None
    lo, hi = min(positive), max(positive)
    steps = [m * 10 ** e for e in range(0, 7) for m in (1, 3)]
    return [v for v in steps if lo / 3 < v < hi * 3] or None


def x_time(timed, domain=None):
    enc = q("x0", TIME_AXIS if timed else "step")
    if domain:
        enc["scale"] = {"domain": domain, "nice": False}
    return enc


def crosshair(p, tooltip):
    """A rule at the nearest step that lists every series there: the reader aims at a time, not a 2px line."""
    return {"mark": {"type": "rule", "color": p["grey"], "strokeWidth": 1},
            "params": [{"name": "hover", "select": {"type": "point", "fields": ["step"], "nearest": True,
                                                     "on": "pointerover", "clear": "pointerout"}}],
            "encoding": {"x": {"field": "x0", "type": "quantitative"},
                         "opacity": {"condition": {"param": "hover", "empty": False, "value": 1}, "value": 0},
                         "tooltip": tooltip}}


STEP_TIP = (("step", "step", ","), ("x0", "start s", ".3f"), ("interval_ms", "step ms", ",.1f"),
            ("decode_rows", "decode rows", ","), ("prefill_tokens", "prefill tokens", ","),
            ("prefill_rows", "prefill rows", ","))


def tokens_spec(p, timed, last, dataset="steps", domain=None, budget=False):
    """Chart 1: each step's query tokens stacked, prefill below decode (so the prefill band's top reads against the
    budget rule, which caps prefill alone), each step exactly as wide as it ran, and a labelled rule where the last
    step ends (a decode tail can be a hairline under a big prefill, so where the run ends is marked, not left to
    the eye). last: the last step's index. budget: draw the prefill budget, labelled at the right end, above the
    decode tail where the plot is empty."""
    series = {"decode tokens": "blue", "prefill tokens": "orange"}
    layers = [{"transform": [{"fold": ["decode_rows", "prefill_tokens"], "as": ["series", "tokens"]},
                             {"calculate": "datum.series === 'decode_rows' ? 'decode tokens' : 'prefill tokens'",
                              "as": "series"},
                             {"calculate": "datum.series === 'prefill tokens' ? 0 : 1", "as": "order"},
                             {"stack": "tokens", "groupby": ["step"], "sort": [{"field": "order"}], "as": ["y0", "y1"]}],
               "mark": {"type": "rect"},
               "encoding": {"x": x_time(timed, domain), "x2": {"field": "x1"}, "y": q("y0", "tokens per step"),
                            "y2": {"field": "y1"}, "color": colour(p, "series", series)}}]
    done = "'done ' + format(datum.x1, '.2f') + ' s'" if timed else "'done after step ' + datum.step"
    layers.append({"transform": [{"filter": f"datum.step === {last}"}, {"calculate": done, "as": "done"}],
                   "layer": [{"mark": {"type": "rule", "color": p["secondary"], "strokeWidth": 1},
                              "encoding": {"x": {"field": "x1", "type": "quantitative"}}},
                             {"mark": {"type": "text", "align": "right", "baseline": "top", "dx": -4, "y": 2,
                                       "fontSize": 11, "color": p["secondary"]},
                              "encoding": {"x": {"field": "x1", "type": "quantitative"},
                                           "text": {"field": "done", "type": "nominal"}}}]})
    if budget:
        layers.append({"mark": {"type": "rule", "color": p["text"], "strokeWidth": 1, "opacity": 0.6},
                       "encoding": {"x": {"field": "x0", "type": "quantitative"}, "x2": {"field": "x1"},
                                    "y": {"field": "budget", "type": "quantitative"}}})
        layers.append({"transform": [{"filter": f"datum.step === {last}"}],  # one direct label
                       "mark": {"type": "text", "align": "right", "baseline": "bottom", "dy": -3, "fontSize": 11,
                                "color": p["secondary"]},
                       "encoding": {"x": {"field": "x1", "type": "quantitative"},
                                    "y": {"field": "budget", "type": "quantitative"},
                                    "text": {"value": "prefill budget"}}})
    extra = (("budget", "prefill budget", ","),) if budget else ()
    layers.append(crosshair(p, tip(*STEP_TIP, *extra)))
    return spec(p, {"data": {"name": dataset}, "layer": layers}, 200)


def lines_spec(p, timed, series, y_title, extra_tip=(), height=200):
    """Step-after lines over the steps, one per field in series ({field: (label, slot)}), on one axis."""
    labels = {label: slot for label, slot in series.values()}
    names = " : ".join(f"datum.series === '{f}' ? '{label}'" for f, (label, _) in series.items())
    return spec(p, {"data": {"name": "steps"}, "layer": [
        {"transform": [{"fold": list(series), "as": ["series", "value"]},
                       {"calculate": f"{names} : datum.series", "as": "series"}],
         "mark": {"type": "line", "interpolate": "step-after", "strokeWidth": 1.5},
         "encoding": {"x": x_time(timed), "y": q("value", y_title), "color": colour(p, "series", labels)}},
        crosshair(p, tip(("step", "step", ","), ("x0", "start s", ".3f"),
                         *[(f, label, ",.4~f") for f, (label, _) in series.items()], *extra_tip))]}, height)


def scatter_spec(p, x, y, x_title, y_title, x_scale="linear", y_scale="log", colours=KINDS,
                 colour_title="kind (analysis)", shape=None, tooltip=(), data="requests", ticks=None):
    """One point per row, coloured by its kind (a request's analysis_kind, or a step's decode/prefill mix). Rows
    without both values, or not positive on a log axis, are left out. ticks: the log axis's tick values."""
    valid = [f"isValid(datum.{x})", f"isValid(datum.{y})"]
    valid += [f"datum.{f} > 0" for f, scale in ((x, x_scale), (y, y_scale)) if scale == "log"]
    enc = {"x": q(x, x_title, scale={"type": x_scale}), "y": q(y, y_title, scale={"type": y_scale}),
           "color": colour(p, "kind", colours, colour_title), "tooltip": tip(*tooltip)}
    for axis, scale in (("x", x_scale), ("y", y_scale)):
        if scale == "log" and ticks:
            enc[axis]["axis"] = {"values": ticks}
    if shape:
        enc["shape"] = {"field": shape, "type": "nominal", "title": "finish",
                        "scale": {"domain": ["stop", "length"], "range": ["circle", "square"]}}
    return spec(p, {"data": {"name": data}, "transform": [{"filter": " && ".join(valid)}],
                    "mark": {"type": "point", "filled": True, "size": 36, "opacity": 0.75}, "encoding": enc})


REQUEST_TIP = (("id", "id", None), ("kind", "kind", None), ("prompt_len", "prompt tokens", ","),
               ("prefix_hit_tokens", "read from shared KV", ","), ("max_tokens", "max_tokens", ","),
               ("output_tokens", "output tokens", ","), ("finish_reason", "finish", None),
               ("prefill_start_s", "prefill start s", ".3f"), ("finished_s", "finished s", ".3f"))


def histogram_spec(p, field, x_title):
    return spec(p, {"data": {"name": "requests"}, "mark": {"type": "bar", "cornerRadiusEnd": 2},
                    "encoding": {"x": {"field": field, "type": "quantitative", "bin": {"maxbins": 30}, "title": x_title},
                                 "y": {"aggregate": "count", "type": "quantitative", "title": "requests"},
                                 "color": colour(p, "kind", KINDS, "kind (analysis)"),
                                 "tooltip": [{"field": field, "bin": True, "title": x_title},
                                             {"aggregate": "count", "type": "quantitative", "title": "requests"},
                                             {"field": "kind", "type": "nominal", "title": "kind"}]}}, 180)


def group_spec(p):
    """Static chart 1: per fixed group, real prompt tokens, live decode tokens and the waste of static batching."""
    series = {"prefill": ("prompt tokens", "orange"), "decode": ("decode tokens (live rows)", "blue"),
              "wasted": ("wasted: padding + dead slots", "grey")}
    labels = {label: slot for label, slot in series.values()}
    names = " : ".join(f"datum.series === '{f}' ? '{label}'" for f, (label, _) in series.items())
    return spec(p, {"data": {"name": "groups"},
                    "transform": [{"fold": list(series), "as": ["series", "tokens"]},
                                  {"calculate": f"{names} : datum.series", "as": "series"}],
                    "mark": {"type": "bar"},
                    "encoding": {"x": {"field": "group", "type": "ordinal", "title": "fixed group, in run order",
                                       "axis": {"labelAngle": 0}},
                                 "y": q("tokens", "token slots"), "color": colour(p, "series", labels),
                                 "order": {"field": "series"},
                                 "tooltip": tip(("group", "group", ","), ("n", "rows", ","), ("padded_len", "padded to", ","),
                                                ("prefill", "prompt tokens", ","), ("padding", "padding", ","),
                                                ("decode", "live decode slots", ","), ("dead_slots", "dead slots", ","),
                                                ("decode_steps", "decode steps", ","), ("prefill_s", "prefill s", ".3f"),
                                                ("decode_s", "decode s", ".3f"))}})


PHASES = {"KV reserved, waiting": "grey", "prefill": "orange", "decode": "blue"}


def timeline_spec(p, n, present):
    """present: the phases the data has, so the legend never lists one that is not drawn."""
    phases = {k: v for k, v in PHASES.items() if k in present}
    return spec(p, {"data": {"name": "timeline"}, "mark": {"type": "rect"},
                    "encoding": {"x": q("x0", TIME_AXIS), "x2": {"field": "x1"},
                                 # KV held while waiting for prefill is context: it recedes behind the work
                                 "opacity": {"condition": {"test": "datum.phase === 'KV reserved, waiting'", "value": 0.3},
                                             "value": 1},
                                 "y": q("rank", "requests, by prefill start", axis={"format": "d"}, scale={"domain": [0, n]}),
                                 "y2": {"field": "rank_end"}, "color": colour(p, "phase", phases),
                                 "tooltip": tip(("id", "id", None), ("phase", "phase", None), ("x0", "from s", ".3f"),
                                                ("x1", "to s", ".3f"), ("prompt_len", "prompt tokens", ","),
                                                ("output_tokens", "output tokens", ","))},
                    "transform": [{"calculate": "datum.rank + 1", "as": "rank_end"}]}, min(360, max(200, n)))


# assembly -----------------------------------------------------------------------------------------------

def chart(cid, title, description, build, wide=False):
    return {"id": cid, "title": title, "description": description, "wide": wide,
            "light": build(PALETTES["light"]), "dark": build(PALETTES["dark"])}


def charts(record, reference=None):
    """(charts, datasets, notes) for one engine record. reference: {"record", "prefix_cache"}, the same job's
    like-for-like vLLM row (bench.report.vllm_reference), whose chart 1 is drawn under this run's on the same time
    axis. Each chart is {id, title, description, wide, light, dark}."""
    engine = engine_of(record)
    requests = request_rows(record, engine)
    steps, timed = step_rows(record, engine, requests)
    out, data, notes = [], {}, []
    if steps:
        data["steps"] = steps
        budget = engine == "batchinfer" and any(s.get("budget") is not None for s in steps)
        domain = None
        ref_steps, ref_timed = step_rows(reference["record"], "vllm", []) if reference else ([], False)
        if ref_steps and ref_timed and timed:
            domain = [0, max(steps[-1]["x1"], ref_steps[-1]["x1"])]
            data["reference_steps"] = ref_steps
        out.append(chart("tokens_per_step", "Tokens per step: decode vs prefill",
                         "Each step's query tokens, one decode token per decoding request plus the prefill tokens, "
                         "each step as wide as it ran. Low stretches are BatchLLM's valleys (steps too small to fill "
                         "the GPU); the tail after the last prefill is the longest decode chain.",
                         lambda p: tokens_spec(p, timed, len(steps) - 1, domain=domain, budget=budget), wide=True))
        if domain:
            cache = "on" if reference["prefix_cache"] else "off"
            out.append(chart("tokens_per_step_vllm", f"The same job on vLLM, prefix cache {cache}, same time axis",
                             f"vLLM's iterations for this workload, model and GPU, from its own trace, with its prefix "
                             f"cache {cache} as this run's prefix sharing was (like for like): compare where prefill "
                             f"ends and how long the decode tail runs.",
                             lambda p: tokens_spec(p, True, len(ref_steps) - 1, "reference_steps", domain),
                             wide=True))
        if engine == "vllm":
            series = {"decode_rows": ("decoding", "blue"), "prefill_rows": ("prefilling", "orange"),
                      "running": ("running", "aqua"), "waiting": ("waiting", "grey")}
        else:
            series = {"decode_rows": ("decoding", "blue"), "prefill_rows": ("prefilling", "orange"),
                      "admitted": ("admitted (KV reserved)", "aqua")}
            if any("waiting" in s for s in steps):
                series["waiting"] = ("waiting for admission", "grey")
        out.append(chart("in_flight", "Requests in flight",
                         "How many requests each step decoded and prefilled, how many held KV, and how many still "
                         "waited for admission.",
                         lambda p: lines_spec(p, timed, series, "requests")))
        kv = {"kv_in_use_pct": ("in use (allocated as written)", "aqua")} if engine == "vllm" else \
            {"kv_held_pct": ("held", "violet")}
        if any(s.get("kv_written_pct") is not None for s in steps):
            kv["kv_written_pct"] = ("written", "aqua")
        if any(s.get("kv_trie_pct") for s in steps):
            kv["kv_trie_pct"] = ("shared prefix blocks", "orange")
        if any(s.get("kv_pinned_pct") for s in steps):
            kv["kv_pinned_pct"] = ("pinned for later requests", "grey")
        if any(s.get(field) is not None for s in steps for field in kv):
            out.append(chart("kv", "KV cache in use, % of the pool",
                             "vLLM allocates KV as tokens are written, so its 'in use' compares with batchinfer's "
                             "'written'." if engine == "vllm" else
                             "Held: KV reserved for admitted requests' whole lives (prompt + max_tokens). Written: "
                             "what those requests have filled so far, the counterpart of vLLM's usage. Shared prefix "
                             "and pinned blocks are the trie's.",
                             lambda p: lines_spec(p, timed, kv, "% of KV blocks")))
        if timed:
            out.append(chart("step_time", "Step time vs step tokens",
                             "Each step's wall time (end to end, scheduling and commit included) against its query "
                             "tokens. A flat left edge is a per-step fixed cost; bands at one token count mean the "
                             "same work took different times, so look for what else changed (steps are in the "
                             "tooltips).",
                             lambda p: scatter_spec(p, "tokens", "interval_ms", "query tokens in the step",
                                                    "step ms (wall)", x_scale="log", y_scale="linear",
                                                    colours=STEP_KINDS, colour_title="step", data="steps",
                                                    tooltip=(*STEP_TIP, ("tokens", "tokens", ",")),
                                                    ticks=log_ticks(s["tokens"] for s in steps))))
        if not timed:
            notes.append("This record predates end_ms: its steps are drawn by index, not time, and it has no "
                         "request trace. A new run draws every chart.")
    groups = group_rows(record) if engine == "naive" else []
    if groups:
        data["groups"] = groups
        out.append(chart("tokens_per_group", "Tokens per fixed group: work vs waste",
                         "Per static batch: real prompt tokens, decode tokens of rows still live, and the waste of "
                         "static batching, padding plus the dead slots of rows that already finished.",
                         group_spec, wide=True))
    elif engine == "naive":
        notes.append("This static record predates per-group prompt_tokens, so its group chart is left out.")
    if requests:
        data["requests"] = requests
        timeline = timeline_rows(requests)
        if timeline:
            data["timeline"] = timeline
            n = len({r["rank"] for r in timeline})
            out.append(chart("timeline", "Request timeline",
                             "Every request from its prefill start to its last token, ranked by prefill start: "
                             "prefill in orange, decode in blue; grey is KV reserved while it waited for its first "
                             "prefill chunk." + (" " + NAIVE_FINISH if engine == "naive" else ""),
                             lambda p: timeline_spec(p, n, {t["phase"] for t in timeline}), wide=True))
        out.append(chart("prompt_len", "Prompt length over time",
                         "Each request's prompt tokens at the moment its prefill started: the order the policy chose, "
                         "as it ran.",
                         lambda p: scatter_spec(p, "prefill_start_s", "prompt_len", TIME_AXIS, "prompt tokens",
                                                tooltip=REQUEST_TIP, ticks=log_ticks(r["prompt_len"] for r in requests))))
        out.append(chart("output_len", "Output length over time",
                         "Each request's output tokens when it finished: squares ran to max_tokens, circles stopped "
                         "on EOS." + (" " + NAIVE_FINISH if engine == "naive" else ""),
                         lambda p: scatter_spec(p, "finished_s", "output_tokens", TIME_AXIS, "output tokens",
                                                shape="finish_reason", tooltip=REQUEST_TIP,
                                                ticks=log_ticks(r["output_tokens"] for r in requests))))
        out.append(chart("decode_ratio", "Admission order: decode ratio at prefill start",
                         "(max_tokens - 1) / prompt tokens, BatchLLM's R, when each request's prefill started. "
                         "Decode-heavy first keeps decode rows alive all job long for prefill chunks to ride on.",
                         lambda p: scatter_spec(p, "prefill_start_s", "decode_ratio", TIME_AXIS, "decode ratio",
                                                y_scale="symlog",
                                                tooltip=(*REQUEST_TIP, ("decode_ratio", "decode ratio", ".3f")))))
        out.append(chart("prompt_hist", "Prompt length distribution",
                         "Prompt tokens per request, stacked by kind (BatchLLM Fig 8's view of the workload).",
                         lambda p: histogram_spec(p, "prompt_len", "prompt tokens")))
        if any(r["prefix_hit_tokens"] for r in requests):
            out.append(chart("hit_hist", "Prompt tokens read from shared KV",
                             "Per request, the prompt tokens another request computed: the shared prefix it did not "
                             "prefill again.",
                             lambda p: histogram_spec(p, "prefix_hit_tokens", "prompt tokens read from shared KV")))
        out.append(chart("output_hist", "Output length distribution",
                         "Output tokens per request, stacked by kind (BatchLLM Fig 5).",
                         lambda p: histogram_spec(p, "output_tokens", "output tokens")))
    return out, data, notes
