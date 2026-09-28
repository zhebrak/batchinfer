"""scripts/median_rows.py: results/ from several suite runs, each row its median run."""
import hashlib
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("median_rows", Path(__file__).resolve().parents[1] / "scripts" / "median_rows.py")
median_rows = importlib.util.module_from_spec(spec)
spec.loader.exec_module(median_rows)

ROW = "w/stub/H100-PCIe/{}"


def workload(tmp_path):
    path = tmp_path / "workloads" / "w.jsonl"
    path.parent.mkdir()
    path.write_text("".join(json.dumps(r) + "\n" for r in (
        {"id": "c", "kind": "classify", "prompt": "p"}, {"id": "g", "kind": "generate", "prompt": "q"})))
    path.with_suffix(".meta.json").write_text(json.dumps({"stop_token_ids": [0]}))
    return path


def row(run, label, wall, stamp, outs, wl, against=None):
    d = run / ROW.format(label)
    d.mkdir(parents=True)
    m = {"wall_s": wall, "timestamp": stamp, "workload": "workloads/w.jsonl",
         "workload_sha256": hashlib.sha256(wl.read_bytes()).hexdigest(),
         "compare": {"against": f"results/{ROW.format(against)}/outputs.jsonl", "stale": 1} if against else None}
    (d / "metrics.json").write_text(json.dumps(m))
    (d / "outputs.jsonl").write_text("".join(json.dumps({"id": i, "text": t, "token_ids": ids}) + "\n"
                                             for i, (t, ids) in outs.items()))
    (d / "report.html").write_text("stale")


def test_each_row_is_its_median_run_compared_against_the_vllm_row_beside_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wl = workload(tmp_path)
    same, other = {"c": ("A", [1, 0]), "g": ("x y", [5, 6, 0])}, {"c": ("B", [2, 0]), "g": ("x z", [5, 7, 0])}
    for n, (vwall, vouts, bwall) in enumerate([(3.0, other, 9.0), (2.0, same, 7.0), (4.0, other, 8.0)], 1):
        run = tmp_path / f"run{n}"
        row(run, "vllm", vwall, f"t{n}", vouts, wl)
        row(run, "batchinfer", bwall, f"b{n}", same, wl, against="vllm")
        row(run, "naive", 50.0, "n1", same, wl, against="vllm")  # run 1's row, which runs 2 and 3 reused
    out = tmp_path / "results"
    median_rows.main([str(tmp_path / f"run{n}") for n in (1, 2, 3)] + ["--out", str(out)])
    runs = json.loads((out / "runs.json").read_text())
    assert runs[ROW.format("vllm")] == {"wall_s": {"run1": 3.0, "run2": 2.0, "run3": 4.0}, "row": "run1"}
    assert runs[ROW.format("batchinfer")]["row"] == "run3" and runs[ROW.format("naive")] == {"wall_s": {"run1": 50.0},
                                                                                             "row": "run1"}
    b = json.loads((out / ROW.format("batchinfer") / "metrics.json").read_text())
    assert b["wall_s"] == 8.0 and not (out / ROW.format("batchinfer") / "report.html").exists()
    # run 3's batchinfer outputs, against run 1's vLLM outputs now beside it: nothing identical
    assert b["compare"] == {"against": f"results/{ROW.format('vllm')}/outputs.jsonl",
                            "classify": {"n": 1, "match": 0.0, "identical": 0.0},
                            "generate": {"n": 1, "match": 0.3333, "identical": 0.0}}
