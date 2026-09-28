"""HTML reports: a page per run for both metrics.json shapes, the index, escaping, serving. CPU only."""
import json
import re
import threading
import urllib.request
import xml.etree.ElementTree as ET

import pytest

from batchinfer.metrics import Metrics
from bench.backends import Dummy
from bench.report import index_page, main, make_server, render, table, when, with_details
from bench.run import run
from test_bench import workload  # noqa: F401  (the fixture only; importing tests would collect them twice)


def naive_metrics():
    """python -m batchinfer run --engine naive, as metrics.json holds it."""
    m = Metrics()
    m.meta.update(model="stub", engine="naive", order="input", timestamp="2026-09-27T00:00:00+0000")
    m.add(requests=2, prompt_tokens=10, output_tokens=4)
    m.add_group(group=0, n=2, kinds={"generate": 2}, padded=False, T=2)
    m.finish()
    return m.to_dict()


def step_metrics(ids=("a", "b"), prefix_sharing=True):
    """The default batchinfer engine: no groups, a policy, a per-step and a per-request trace."""
    m = Metrics()
    m.meta.update(model="stub", engine="batchinfer", order="decode_ratio_desc", timestamp="2026-09-27T00:00:00+0000",
                  args={"cmd": "run", "model": "stub", "reserve_gb": 4.0, "limit": None})
    m.timing.update(inference=1.0)
    m.add(requests=2, prompt_tokens=10, output_tokens=4, sampled_rows=4, kv_blocks=11)
    for k in range(2):
        m.record_step(decode_rows=1, prefill_tokens=4, prefill_rows=1, admitted=1, free_blocks=10, step_ms=1.0,
                      gpu_ms=None, sched_ms=0.1, trie_blocks=0, pinned_blocks=0, step_prefill_budget=8,
                      kv_reserved_tokens=16, kv_written_tokens=4, unprefilled_tokens=0, end_ms=1.5 * (k + 1),
                      graph_rows=0)
    for i, rid in enumerate(ids):
        m.record_request(idx=i, id=rid, analysis_kind="generate", group=None, prompt_len=5, max_tokens=2,
                         output_tokens=2, prefix_hit_tokens=0, finish_reason="length", admitted_step=0,
                         prefill_start_step=0, first_token_step=0, finished_step=1, admitted_ms=0.0,
                         prefill_start_ms=0.0, first_token_ms=1.5, finished_ms=3.0)
    m.set_policy({"engine": "batchinfer", "admission_order_head": [0, 1], "admission": "continuous",
                  "prefix_sharing": prefix_sharing})
    m.finish()
    return m.to_dict()


def write_metrics(path, d):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(d))


class Detailed(Dummy):
    """Dummy plus the details() an engine backend has: its own Metrics record."""

    def details(self):
        return step_metrics()


def test_run_writes_report_page(tmp_path, workload):
    path, _ = workload
    run(path, "dummy", out=tmp_path / "a")
    m = run(path, "dummy", out=tmp_path / "b", compare_to=tmp_path / "a" / "outputs.jsonl")
    text = (tmp_path / "b" / "report.html").read_text()
    root = ET.fromstring(text)
    title = f"dummy · {m['workload'].rsplit('/', 1)[-1].removesuffix('.jsonl')} · stub"
    assert root.findtext(".//title").startswith(title) and root.findtext(".//h1") == root.findtext(".//title")
    # what launched, the outcome, then correctness; no dump of every recorded key
    assert [c.text for c in root.iter("caption")] == ["Launch", "Outcome", "Accuracy by source", "Match against a"]
    assert "identical" in text and "Backend stats" not in text and "Configuration" not in text
    assert "labelled" in text and ">50.0<" in text  # accuracy as a %, like every other share on the page
    assert "match %" in text and "identical %" in text
    assert 'href="metrics.json"' in text and 'href="outputs.jsonl"' in text and "details.json" not in text
    assert "Warning" not in text and "All runs" not in text  # bench.run writes the page alone: no index to link


def test_values_are_escaped(tmp_path, workload):
    m = run(workload[0], "dummy", out=tmp_path / "a")
    m["model"] = "<script>alert(1)</script>"
    m["opts"] = {"k": "<v>"}
    m["by_source"]["x&y"] = {"n": 1, "kind": "c"}
    text = render(m, "a", ["metrics.json"])
    assert "<script>alert(1)</script>" not in text  # the page has scripts of its own
    assert "k=&lt;v&gt;" in text and "x&amp;y" in text
    ET.fromstring(text)
    assert "<title>a&lt;b</title>" in render(step_metrics(), "a<b", [])  # an engine run's page is titled by its dir


def test_engine_backend_page_links_and_draws_its_record(tmp_path, workload):
    run(workload[0], "test_report:Detailed", out=tmp_path / "a", warmup=False)
    page = tmp_path / "a" / "report.html"
    for text in (page.read_text(), None):
        if text is None:  # --html rebuilds the page from details.json on disk
            page.unlink()
            main(["--html", str(tmp_path)])
            text = page.read_text()
        ET.fromstring(text)
        assert 'href="details.json"' in text and 'id="chart-data"' in text
        assert "2 steps and 2 requests recorded; the traces are in details.json." in text
        assert "<caption>Engine policy</caption>" not in text  # the record is linked and drawn, not dumped


def test_tables_with_a_header_sort(tmp_path, workload):
    root = ET.fromstring(render(run(workload[0], "dummy", out=tmp_path / "a"), "a", []))
    tables = root.findall(".//table")
    for t in tables:
        header = t.find("thead")
        assert ("sortable n-last" in t.get("class")) == (header is not None)  # key/value tables have no header row
        if header is not None:
            assert all(th.find("button").get("type") == "button" for th in header.iter("th"))
    by_source = next(t for t in tables if t.findtext("caption") == "Accuracy by source")
    rows = {tr.findtext("th"): tr.findall("td") for tr in by_source.find("tbody")}
    assert rows["labelled"][-1].get("data-sort") == "50.0"
    assert any(tds[-1].get("data-sort") == "" and tds[-1].text == "-" for tds in rows.values())  # no accuracy
    assets = [e for e in (*root.iter("link"), *root.iter("script")) if e.get("href") or e.get("src")]
    assert len(assets) == 4
    for e in assets:  # pinned to a version, and the browser checks the bytes
        assert re.match(r"https://cdn\.jsdelivr\.net/npm/[a-z-]+@\d+\.\d+\.\d+/", e.get("href") or e.get("src"))
        assert e.get("integrity").startswith("sha384-") and e.get("crossorigin") == "anonymous"


def text(e):
    return "".join(e.itertext()).strip()


def test_index_has_a_column_per_launch_knob_then_the_tradeoff_numbers(tmp_path, workload):
    m = dict(run(workload[0], "dummy", out=tmp_path / "a"), gpu="NVIDIA H100 PCIe")
    stats = {"steps": 100, "order": "prefix_dfs", "prefix_sharing": True, "prefill_budget": 583,
             "prefill_budget_adaptive": True, "prefix_hit_pct": 80.0, "ideal_prefix_reuse_page16": 0.8045,
             "mean_decode_batch": 20.0, "fused_layers": True, "cuda_graphs": True, "fa_version": 2,
             "graph_step_pct": 81.3}
    bi = dict(m, backend="batchinfer", engine="batchinfer", opts={}, wall_s=25.0, backend_stats=stats,
              gpu_util_mean=85.1, mfu_pct=16.6, mbu_pct=40.2)
    variants = {"vllm": dict(m, backend="vllm", engine="vllm", opts={}, wall_s=10.0),
                "vllm-off": dict(m, backend="vllm", engine="vllm", opts={"enable_prefix_caching": False}, wall_s=20.0),
                "bi": bi,
                "bi-off": dict(bi, opts={"prefix_sharing": False, "admission": "continuous"}, length_violations=2,
                               backend_stats=dict(stats, prefix_sharing=False, prefill_budget_adaptive=False,
                                                  fused_layers=False, cuda_graphs=False)),
                # rows from before the engine rename record no engine, only their backend class
                "legacy-naive": dict(m, backend="batchinfer.bench:EngineBackend", engine=None, opts={}, wall_s=140.0,
                                     backend_stats={"max_batch_tokens": 196608, "order": "max_tokens_desc"}),
                "branch-bi": dict(bi, backend="batchinfer.bench:StepBackend", engine=None, branch="feature-x")}
    root = ET.fromstring(index_page(tmp_path, {tmp_path / k / "metrics.json": v for k, v in variants.items()}))
    table = next(root.iter("table"))
    assert table.get("id") == "bench"  # nothing comes before the runs
    header = table.find("thead").findall(".//th")
    titles = [text(th) for th in header]
    assert titles == ["engine", "workload", "model", "GPU", "prefix reuse", "order", "prefill budget", "layers", "graphs",
                      "wall s", "× vLLM", "accuracy %", "prefix hit %", "steps", "ms/step", "decode batch",
                      "GPU util %", "MFU %", "MBU %", "commit"]  # utilisation, every engine counted alike; provenance last
    for th in header:  # every column explains itself on hover
        button = th.find("button")
        assert button.get("data-bs-toggle") == "tooltip" and button.get("title"), text(th)
        assert button.get("class") == "term"
    assert [header[titles.index(t)].get("class") for t in ("wall s", "× vLLM", "order")] == ["n", "n", None]
    trs = table.find("tbody").findall("tr")
    # baselines first: naive, vLLM with its cache off then on, then batchinfer with the global trie
    assert [tr.find("th/a").get("title") for tr in trs] == ["legacy-naive", "vllm-off", "vllm", "bi", "branch-bi",
                                                            "bi-off"]
    assert trs[3].find("th/a").get("href") == "bi/report.html"  # the engine heads the row and links its page
    rows = {tr.find("th/a").get("title"): [text(tr.find("th")), *[text(td) for td in tr.findall("td")]] for tr in trs}
    cell = lambda run, title: rows[run][titles.index(title)]  # noqa: E731
    assert [cell(r, "engine") for r in ("vllm", "bi", "legacy-naive", "branch-bi")] == [
        "vllm", "batchinfer", "naive", "batchinfer"]
    assert [cell(r, "prefix reuse") for r in ("vllm", "vllm-off", "bi", "bi-off", "legacy-naive")] == [
        "prefix cache", "off", "global trie", "off", "off"]
    assert [cell(r, "order") for r in ("vllm", "bi", "legacy-naive")] == ["-", "prefix_dfs", "max_tokens_desc"]
    assert [cell(r, "prefill budget") for r in ("bi", "bi-off", "vllm")] == ["adaptive from 583", "583", "-"]
    # what the forward was: the engine's recorded build, vLLM's compile, naive's HF layers; rows from before the
    # engine recorded it (branch-bi's StepBackend) ran HF layers eagerly
    assert [cell(r, "layers") for r in ("bi", "bi-off", "vllm", "legacy-naive", "branch-bi")] == [
        "fused", "HF", "compiled", "HF", "fused"]
    assert [cell(r, "graphs") for r in ("bi", "bi-off", "vllm", "legacy-naive")] == ["decode 81%", "off", "on", "off"]
    assert cell("branch-bi", "commit").split(" · ")[0].endswith("@feature-x")
    # each column at its own fixed decimals, so the points line up
    assert [cell(r, "× vLLM") for r in ("vllm", "vllm-off", "bi")] == ["1.00", "2.00", "2.50"]
    assert (cell("bi", "wall s"), cell("bi-off", "wall s")) == ("25.00", "25.00 invalid")  # broke the length rules
    assert cell("bi", "accuracy %") == "50.0"
    assert [cell("bi", t) for t in ("prefix hit %", "steps", "ms/step", "decode batch")] == ["80.0", "100", "250.0",
                                                                                           "20.0"]
    assert [cell("bi", t) for t in ("GPU util %", "MFU %", "MBU %")] == ["85.1", "16.6", "40.2"]
    blank = trs[1].findall("td")  # vllm-off: no order (a text column), no prefill budget (a numeric one)
    assert [blank[titles.index(t) - 1].get("class") for t in ("order", "prefill budget")] == ["blank", "n blank"]
    assert [tr.get("data-group") for tr in trs] == [trs[0].get("data-group")] * 6  # one job, one band
    hit = trs[3].findall("td")[titles.index("prefix hit %") - 1]
    assert hit.get("title") == "block ceiling 80.5%"
    form = next(f for f in root.iter("form") if f.get("data-table") == "bench")
    selects = {label.text: label.find("select") for label in form.iter("label") if label.find("select") is not None}
    assert list(selects) == ["engine", "workload", "model", "GPU", "prefix reuse", "order", "layers", "graphs", "commit"]
    assert [o.text for o in selects["engine"].findall("option")] == ["all", "batchinfer", "naive", "vllm"]
    assert [o.text for o in selects["prefix reuse"].findall("option")] == ["all", "global trie", "off", "prefix cache"]
    assert selects["engine"].get("data-col") == "0"  # the row header
    assert selects["order"].get("data-col") == str(titles.index("order"))


def test_a_vllm_row_with_max_num_seqs_names_it_in_its_own_column(tmp_path, workload):
    """Else it reads as vLLM's default row in every launch column; × vLLM still divides by the default row."""
    m = dict(run(workload[0], "dummy", out=tmp_path / "a"), gpu="NVIDIA A100-SXM4-40GB")
    variants = {"vllm": dict(m, backend="vllm", engine="vllm", opts={}, wall_s=10.0),
                "vllm-512": dict(m, backend="vllm", engine="vllm", opts={"max_num_seqs": 512}, wall_s=12.5),
                "bi": dict(m, backend="batchinfer", engine="batchinfer", opts={}, wall_s=25.0)}
    page = index_page(tmp_path, {tmp_path / k / "metrics.json": v for k, v in variants.items()})
    table = next(ET.fromstring(page).iter("table"))
    titles = [text(th) for th in table.find("thead").findall(".//th")]
    assert titles[titles.index("graphs"):titles.index("× vLLM") + 1] == ["graphs", "max seqs", "wall s", "× vLLM"]
    rows = {tr.find("th/a").get("title"): [text(tr.find("th")), *[text(td) for td in tr.findall("td")]]
            for tr in table.find("tbody").findall("tr")}
    cell = lambda run, title: rows[run][titles.index(title)]  # noqa: E731
    assert [cell(r, "max seqs") for r in ("vllm", "vllm-512", "bi")] == ["-", "512", "-"]  # blank: vLLM's default
    assert [cell(r, "× vLLM") for r in ("vllm", "vllm-512", "bi")] == ["1.00", "1.25", "2.50"]


def suite_rows(m):
    """A suite's worth of rows on one model and GPU, built on a dummy run's metrics m."""
    def row(engine, opts, wall, stem="mixed-quick", preset="mixed", sha="new1234", commit="2026-09-27T16:24:00+00:00",
            **kw):
        return {**m, "workload": f"workloads/{stem}.jsonl", "workload_sha256": f"sha-{stem}", "preset": preset,
                "model": "Qwen/Qwen3-1.7B", "gpu": "NVIDIA H100 PCIe", "engine": engine, "backend": engine,
                "opts": opts, "wall_s": wall, "output_tok_per_s": 1000.0 / wall, "length_violations": 0,
                "by_source": {"mmlu": {"n": 32, "kind": "classify", "accuracy": 0.75}}, "git_sha": sha,
                "git_commit_time": commit, "timestamp": "2026-09-27T16:30:00+0000", "compare": None,
                "backend_stats": {}, **kw}
    return {
        "naive": row("naive", {}, 100.0),
        "naive-sorted": row("naive", {"order": "max_tokens_desc"}, 50.0),
        "vllm": row("vllm", {}, 10.0),
        "vllm-off": row("vllm", {"enable_prefix_caching": False}, 20.0),
        "bi": row("batchinfer", {}, 25.0),
        "bi-nosharing": row("batchinfer", {"prefix_sharing": False}, 40.0, sha="old1234",
                            commit="2026-09-27T12:00:00+00:00"),
        "bi-input": row("batchinfer", {"order": "input"}, 30.0, length_violations=3),
        "vllm-classify": row("vllm", {}, 4.0, stem="classify-quick", preset="classify"),
        "bi-classify": row("batchinfer", {}, 6.0, stem="classify-quick", preset="classify"),
    }


def test_markdown_table_is_the_index_table_in_its_order(tmp_path, workload):
    rows = suite_rows(run(workload[0], "dummy", out=tmp_path / "a"))
    lines = table(list(rows.values())).splitlines()
    # the columns no row fills (the engine counters, utilisation) are left out, as on the index
    assert lines[0] == ("| engine | workload | model | GPU | prefix reuse | order | layers | graphs | wall s | × vLLM | "
                        "accuracy % | commit |")
    firsts = [line.split(" | ")[:2] for line in lines[2:]]
    assert firsts == [["| naive", "mixed-quick"], ["| naive", "mixed-quick"], ["| vllm", "mixed-quick"],
                      ["| vllm", "mixed-quick"], ["| batchinfer", "mixed-quick"], ["| batchinfer", "mixed-quick"],
                      ["| batchinfer", "mixed-quick"], ["| vllm", "classify-quick"],
                      ["| batchinfer", "classify-quick"]]  # the suite's workload order, baselines first in each
    assert any("| 30.00 (invalid) | 3.00 |" in line for line in lines)  # a row that broke the length rules


def subtitle(root):
    """The line under a page's title."""
    main = list(root.find(".//main"))
    return main[main.index(root.find(".//h1")) + 1].text


def test_index_leaves_out_and_names_the_columns_no_run_fills(tmp_path, workload):
    """A column blank in every row tells nothing: the index leaves it out, says so under the table, and the page
    opens with what it holds and how to read it."""
    rows = suite_rows(run(workload[0], "dummy", out=tmp_path / "a"))
    root = ET.fromstring(index_page(tmp_path, {tmp_path / k / "metrics.json": v for k, v in rows.items()}))
    titles = [text(th) for th in next(root.iter("table")).find("thead").iter("th")]
    assert titles == ["engine", "workload", "model", "GPU", "prefix reuse", "order", "layers", "graphs", "wall s",
                      "× vLLM", "accuracy %", "commit"]
    notes = [p.text for p in root.iter("p") if p.get("class") == "note"]
    assert notes == ["Blank in every run, so not shown: prefill budget, max seqs, prefix hit %, steps, ms/step, "
                     "decode batch, GPU util %, MFU %, MBU %."]
    assert subtitle(root).startswith("9 runs: 2 workloads, 1 model, 1 GPU. Each band is one job")
    trs = next(root.iter("table")).find("tbody").findall("tr")
    assert len({tr.get("data-group") for tr in trs}) == 2  # mixed-quick and classify-quick


def test_details_fill_what_a_rows_stats_lack():
    """A naive row's order from its policy; a vLLM row's steps and decode batch from its iteration trace, counted
    as batchinfer counts its own; a row's own stats always win."""
    row = lambda engine, **stats: {"by_source": {}, "engine": engine, "backend": engine, "opts": {},  # noqa: E731
                                   "backend_stats": stats}
    trace = {"steps": {"decode_rows": [0, 2, 4, 0]}, "policy": {"order": "input"}}
    assert with_details(row("naive"), trace)["backend_stats"] == {"order": "input"}
    assert with_details(row("vllm", trace=True), trace)["backend_stats"] == {
        "trace": True, "order": "input", "steps": 4, "mean_decode_batch": 3.0}
    assert with_details(row("batchinfer", steps=9, order="prefix_dfs"), trace)["backend_stats"] == {
        "steps": 9, "order": "prefix_dfs"}
    assert with_details(row("vllm"), None)["backend_stats"] == {}


def test_html_index_takes_a_naive_rows_order_from_its_details(tmp_path, workload):
    run(workload[0], "dummy", out=tmp_path / "old")
    metrics = tmp_path / "old" / "metrics.json"
    m = json.loads(metrics.read_text())
    m.update(engine=None, backend="batchinfer.bench:EngineBackend", backend_stats={"max_batch_tokens": 196608})
    metrics.write_text(json.dumps(m))
    (tmp_path / "old" / "details.json").write_text(json.dumps({"policy": {"engine": "static",
                                                                          "order": "max_tokens_desc"}}))
    main(["--html", str(tmp_path)])
    index = ET.fromstring((tmp_path / "index.html").read_text())
    t = next(index.iter("table"))
    tr = t.find("tbody/tr")
    row = dict(zip([text(th) for th in t.find("thead").iter("th")], [text(tr.find("th")), *map(text, tr.findall("td"))]))
    assert [row[k] for k in ("engine", "prefix reuse", "order")] == ["naive", "off", "max_tokens_desc"]
    page = ET.fromstring((tmp_path / "old" / "report.html").read_text())
    launch = next(t for t in page.iter("table") if t.findtext("caption") == "Launch")
    assert {text(r.find("th")): text(r.find("td")) for r in launch.iter("tr")}["order"] == "max_tokens_desc"


def test_commit_filter_lists_the_newest_commit_first(tmp_path, workload):
    rows = suite_rows(run(workload[0], "dummy", out=tmp_path / "a"))
    root = ET.fromstring(index_page(tmp_path, {tmp_path / k / "metrics.json": v for k, v in rows.items()}))
    form = next(f for f in root.iter("form") if f.get("data-table") == "bench")
    commit = next(label.find("select") for label in form.iter("label") if label.text == "commit")
    assert [o.text for o in commit.findall("option")] == ["all", "new1234 · 09-27 16:24", "old1234 · 09-27 12:00"]
    assert when("2026-09-27T17:24:03+01:00") == "09-27 16:24" and when(None) is None and when("junk") is None


def row_heads(root):
    return {text(th): th for th in root.iter("th") if th.get("scope") == "row"}


def test_run_page_keys_explain_themselves(tmp_path, workload):
    root = ET.fromstring(render(run(workload[0], "dummy", out=tmp_path / "a"), "a", []))
    heads = row_heads(root)
    assert heads["wall s"].find("span").get("title").startswith("The timed pass")
    assert heads["wall s"].find("span").get("tabindex") == "0"  # the keyboard reaches the tooltip too
    assert heads["wall s"].find("span").get("class") == "term"  # underlined dotted, the name itself
    assert heads["accuracy %"].find("span").get("title").startswith("Correct answers")


def test_engine_record_keys_use_the_engines_clock():
    """The engine's req_per_s divides by its inference time, not bench.run's wall s: the help says so."""
    root = ET.fromstring(render(step_metrics(), "step", []))
    heads = row_heads(root)
    assert "inference_s" in heads["req_per_s"].find("span").get("title")
    assert heads["total_s"].find("span").get("title").startswith("From the start of this record")


def test_warnings_are_text(tmp_path, workload):
    m = run(workload[0], "dummy", out=tmp_path / "a")
    m.update(over_budget=True, wall_s=99.0, length_violations=3)
    text = render(m, "a", [])
    assert "<strong>Warning:</strong> 3 requests broke length rules" in text
    assert "<strong>Warning:</strong> timed run took 99.0 s, over the 60 s budget" in text
    assert "<footer>" not in text


def test_engine_pages_render_both_shapes():
    naive = render(naive_metrics(), "naive", ["metrics.json"])
    ET.fromstring(naive)
    assert "<caption>Configuration</caption>" in naive
    assert "<caption>Groups</caption>" in naive and "generate=2" in naive
    step = render(step_metrics(), "step", ["metrics.json"])
    ET.fromstring(step)
    assert "2 steps and 2 requests recorded" in step and "admission_order_head" in step and "0, 1" in step
    assert "<caption>Command-line arguments</caption>" in step and "reserve_gb" in step
    assert "<caption>Groups</caption>" not in step  # the batchinfer engine has no fixed groups
    assert "chunk_prefill" not in step and 'limit</th><td class="n blank">-</td>' in step  # None shows as a muted -
    other = render({**step_metrics(), "version": 3}, "step", [])
    assert "<caption>Other</caption>" in other and "version" in other  # no top-level key is dropped


def test_html_index_lists_every_run(tmp_path, workload, capsys):
    path, _ = workload
    run(path, "dummy", out=tmp_path / "a")
    run(path, "dummy", out=tmp_path / "b", compare_to=tmp_path / "a" / "outputs.jsonl")
    b = tmp_path / "b" / "metrics.json"
    b.write_text(json.dumps({**json.loads(b.read_text()), "backend_stats": {"wasted_decode_slots": 14858}}))
    write_metrics(tmp_path / "smoke" / "v0" / "metrics.json", naive_metrics())
    write_metrics(tmp_path / "smoke" / "step" / "metrics.json", step_metrics())
    capsys.readouterr()  # run() prints its own summary
    main(["--html", str(tmp_path)])
    out, err = capsys.readouterr()
    lines = out.splitlines()
    gpu = " GPU |" if json.loads((tmp_path / "a" / "metrics.json").read_text()).get("gpu") else ""  # on a GPU host
    assert len(lines) == 4 and lines[0] == f"| engine | workload | model |{gpu} wall s | accuracy % | commit |"  # filled
    assert "wrote 4 report.html" in err and "skipped 2 metrics.json" in err
    index = (tmp_path / "index.html").read_text()
    ET.fromstring(index)
    for rel in ("a", "b", "smoke/v0", "smoke/step"):
        assert f'href="{rel}/report.html"' in index and (tmp_path / rel / "report.html").exists()
    assert "Engine runs" in index and "engine=naive" in index
    assert "reserve_gb=4.0" in index and "cmd=" not in index  # meta.args reach the config cell, minus noise
    assert "limit=" not in index  # the same in every engine run, so only the run page lists it
    assert "wasted_decode_slots" not in index  # neither the index nor the page lists every counter
    assert '<th scope="row" data-sort="smoke/v0"><a href="smoke/v0/report.html">v0</a><br/>' in index  # label over dir
    v0 = (tmp_path / "smoke" / "v0" / "report.html").read_text()
    assert 'href="metrics.json"' in v0 and "outputs.jsonl" not in v0  # only files that exist are linked


def test_bad_html_and_serve_arguments_exit(tmp_path):
    d = str(tmp_path)
    for argv in (["--html", d, d], ["--serve", "0", d, d], ["--html", str(tmp_path / "missing")],
                 ["--serve", "0", "--bind", "", d]):  # "" would listen on every interface
        with pytest.raises(SystemExit):
            main(argv)


def test_serve_returns_index_and_pages(tmp_path, workload):
    run(workload[0], "dummy", out=tmp_path / "a")
    main(["--html", str(tmp_path)])
    srv = make_server(tmp_path, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/"
        assert 'href="a/report.html"' in urllib.request.urlopen(url).read().decode()
        assert "<h1>dummy · " in urllib.request.urlopen(url + "a/report.html").read().decode()
        outputs = urllib.request.urlopen(url + "a/outputs.jsonl")
        assert outputs.headers["Content-Type"].startswith("text/plain")  # shown in the browser, not downloaded
    finally:
        srv.shutdown()
        srv.server_close()


VEGA = ("vega@", "vega-lite@", "vega-embed@")


def chart_payload(root):
    return json.loads(root.find(".//script[@id='chart-data']").text)


def test_engine_pages_draw_their_traces_as_charts(tmp_path, workload):
    """A page with a trace carries its charts, one copy of the data, and the pinned vega scripts; the traces
    never show up as key/value tables."""
    root = ET.fromstring(render(step_metrics(), "step", ["metrics.json"]))
    cards = [div.get("data-chart") for div in root.iter("div") if div.get("data-chart")]
    assert cards[:3] == ["tokens_per_step", "in_flight", "kv"] and "prompt_len" in cards and "output_len" in cards
    payload = chart_payload(root)
    assert set(payload["specs"]) == set(cards) and payload["datasets"]["steps"]["step"] == [0, 1]  # columnar
    assert {s["light"]["data"]["name"] for s in payload["specs"].values() if "data" in s["light"]} <= set(payload["datasets"])
    scripts = [e.get("src") for e in root.iter("script") if e.get("src")]
    for name in VEGA:  # pinned to an exact version, and the browser checks the bytes
        e = next(e for e in root.iter("script") if (e.get("src") or "").startswith(f"https://cdn.jsdelivr.net/npm/{name}"))
        assert re.match(rf"https://cdn\.jsdelivr\.net/npm/{name}\d+\.\d+\.\d+/", e.get("src"))
        assert e.get("integrity").startswith("sha384-") and e.get("crossorigin") == "anonymous"
    captions = [c.text for c in root.iter("caption")]
    assert not any(c in ("steps", "request_trace") for c in captions) and len(scripts) == 5
    # a page without a trace loads no chart library at all
    plain = ET.fromstring(render(run(workload[0], "dummy", out=tmp_path / "a"), "a", []))
    assert not any(any(n in (e.get("src") or "") for n in VEGA) for e in plain.iter("script"))
    assert plain.find(".//script[@id='chart-data']") is None


def test_chart_data_cannot_break_out_of_its_script(tmp_path, workload):
    evil = "</script><script>alert(1)</script>"
    text = render(step_metrics(ids=(evil, "b&<c>")), "step", [])
    assert evil not in text and "alert(1)" in text  # escaped inside the JSON, still readable as data
    root = ET.fromstring(text)
    assert evil in chart_payload(root)["datasets"]["requests"]["id"]


class DetailedNoSharing(Dummy):
    """Dummy plus a batchinfer record that ran with prefix sharing off."""

    def details(self):
        return step_metrics(prefix_sharing=False)


def vllm_row(path, out, iterations):
    """A vLLM row beside the others: bench.run's files for the same job, and a trace of `iterations` steps."""
    ref = run(path, "dummy", out=out)
    steps = {"decode_rows": [0] * iterations, "prefill_tokens": [9] * iterations, "prefill_rows": [1] * iterations,
             "end_ms": [2.0 * (k + 1) for k in range(iterations)], "running": [3] * iterations,
             "waiting": [0] * iterations, "kv_cache_usage_pct": [1.0] * iterations}
    (out / "details.json").write_text(json.dumps({"meta": {"engine": "vllm"}, "steps": steps}))
    return ref


def test_bench_page_draws_the_like_for_like_vllm_trace_under_its_own(tmp_path, workload):
    """A batchinfer row's page draws the vLLM row beside it that matches its prefix sharing (cache on for sharing,
    off without), and only for the same job."""
    path, _ = workload
    ref = vllm_row(path, tmp_path / "vllm", 2)
    vllm_row(path, tmp_path / "vllm-enable_prefix_caching=False", 3)
    run(path, "test_report:Detailed", out=tmp_path / "shared", warmup=False)
    run(path, "test_report:DetailedNoSharing", out=tmp_path / "private", warmup=False)
    for row, iterations, cache in (("shared", 2, "on"), ("private", 3, "off")):
        root = ET.fromstring((tmp_path / row / "report.html").read_text())
        payload = chart_payload(root)
        assert len(payload["datasets"]["reference_steps"]["step"]) == iterations, row
        shared = payload["specs"]["tokens_per_step"]["light"]["layer"][0]["encoding"]["x"]["scale"]["domain"]
        assert shared == payload["specs"]["tokens_per_step_vllm"]["light"]["layer"][0]["encoding"]["x"]["scale"]["domain"]
        assert f"prefix cache {cache}" in "".join(root.itertext())
    ref["workload_sha256"] = "another workload file"  # not the same job: no reference
    (tmp_path / "vllm" / "metrics.json").write_text(json.dumps(ref))
    main(["--html", str(tmp_path)])
    payload = chart_payload(ET.fromstring((tmp_path / "shared" / "report.html").read_text()))
    assert "tokens_per_step_vllm" not in payload["specs"]


def test_run_pages_link_the_index_and_the_vllm_row_they_divide_by(tmp_path, workload):
    """--html pages link back to the index, say what they launched under the title, and link × vLLM to the vLLM
    row it divides by (not to themselves)."""
    path, _ = workload
    for label, engine, wall in (("vllm", "vllm", 2.0), ("vllm-enable_prefix_caching=False", "vllm", 3.0),
                                ("naive", "naive", 8.0)):
        run(path, "dummy", out=tmp_path / label)
        metrics = tmp_path / label / "metrics.json"
        opts = {"enable_prefix_caching": False} if "=" in label else {"order": "input"} if engine == "naive" else {}
        metrics.write_text(json.dumps({**json.loads(metrics.read_text()), "engine": engine, "opts": opts,
                                       "wall_s": wall}))
    main(["--html", str(tmp_path)])
    pages = {label: ET.fromstring((tmp_path / label / "report.html").read_text())
             for label in ("vllm", "vllm-enable_prefix_caching=False", "naive")}
    for page in pages.values():
        assert page.find(".//p[@class='back']/a").get("href") == "../index.html"
    assert [subtitle(page) for page in pages.values()] == ["prefix cache", "prefix reuse off",
                                                          "prefix reuse off · order input"]

    def vs(page):
        outcome = next(t for t in page.iter("table") if t.findtext("caption") == "Outcome")
        return next(td for tr in outcome.iter("tr") if text(tr.find("th")) == "× vLLM" for td in tr.iter("td"))
    assert vs(pages["naive"]).find("a").get("href") == "../vllm/report.html" and text(vs(pages["naive"])) == "4.00"
    assert vs(pages["naive"]).get("title") == "wall s ÷ 2.00 s, vLLM with its prefix cache on"
    assert vs(pages["vllm"]).find("a") is None and text(vs(pages["vllm"])) == "1.00"  # its own page: no link
