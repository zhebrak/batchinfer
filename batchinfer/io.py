"""Read the client-visible fields of a workload JSONL; write results as they finish."""
import json
from pathlib import Path

from .schema import Request, Result

CLIENT_FIELDS = ("id", "prompt", "max_tokens", "ignore_eos", "labels")


def from_row(row, where, default_id, default_max_tokens=128):
    """The one dict -> Request conversion. Only CLIENT_FIELDS are read: kind, group, reference, scorer and
    anything else a workload row carries are dropped here, so the engine adapts from payload properties
    alone. Types are checked, never coerced (bool("false") is True, int(2.7) is 2, list("ab") is two labels).
    Errors cite `where`; `id` defaults to default_id."""
    if "prompt" not in row:
        raise ValueError(f"{where}: no 'prompt' field")
    prompt, max_tokens = row["prompt"], row.get("max_tokens", default_max_tokens)
    ignore_eos, labels = row.get("ignore_eos", False), row.get("labels")
    if not isinstance(prompt, str):
        raise ValueError(f"{where}: prompt must be a string, not {type(prompt).__name__}")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError(f"{where}: max_tokens must be an int >= 1, not {max_tokens!r}")
    if not isinstance(ignore_eos, bool):
        raise ValueError(f"{where}: ignore_eos must be true or false, not {ignore_eos!r}")
    if labels is not None and not isinstance(labels, list):
        raise ValueError(f"{where}: labels must be a list of strings, not {labels!r}")
    return Request(id=str(row.get("id", default_id)), prompt=prompt, max_tokens=max_tokens, ignore_eos=ignore_eos,
                   labels=[str(x) for x in labels] if labels else None)


def unique_ids(requests, where):
    """Results are matched to requests by id, so two requests with one id cannot both be answered."""
    seen = set()
    for r in requests:
        if r.id in seen:
            raise ValueError(f"{where}: duplicate request id {r.id!r}")
        seen.add(r.id)
    return requests


def from_rows(rows, default_max_tokens=128):
    """Client rows already in memory (bench.run hands a backend these); `id` defaults to the row index."""
    return unique_ids([from_row(row, f"row {i}", i, default_max_tokens) for i, row in enumerate(rows)], "rows")


def read_requests(path, default_max_tokens=128, limit=None):
    """One JSON object per line, through from_row; `id` defaults to the line index, blank lines counted."""
    requests = []
    with open(path) as f:
        for lineno, line in enumerate(f):
            if not line.strip():
                continue
            if limit is not None and len(requests) >= limit:
                break
            requests.append(from_row(json.loads(line), f"{path}:{lineno + 1}", lineno, default_max_tokens))
    return unique_ids(requests, str(path))


class ResultWriter:
    """Appends one JSON line per result and flushes each time, so a crash mid-job leaves a usable
    partial file. Output order is finishing order; consumers match by id."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w")
        self.n = 0

    def __call__(self, result: Result):
        self.f.write(json.dumps(result.to_dict()) + "\n")
        self.f.flush()
        self.n += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.f.close()
