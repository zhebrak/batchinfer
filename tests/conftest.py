"""Make `bench` and `batchinfer` importable when pytest is run from anywhere (e.g. `pipx run --system-site-packages pytest tests/`, which finds the system jinja2)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# bench.results reads it once, at import: tests write their rows under their own tmp_path/results
os.environ.pop("BENCH_RESULTS_ROOT", None)

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def no_branch(monkeypatch):
    """BENCH_BRANCH relabels bench rows, so a test that wants it sets it itself."""
    monkeypatch.delenv("BENCH_BRANCH", raising=False)
