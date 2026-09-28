"""analysis: one tokenization, exact kinds, job facts and metrics, and no decisions."""
import dataclasses
from pathlib import Path

import pytest

from batchinfer.analysis import analyze, decode_ratio, describe, lcp, unique_prompt_tokens
from batchinfer.io import read_requests
from batchinfer.kv import BLOCK_SIZE
from batchinfer.prefix import Node, PrefixTrie
from batchinfer.schema import JobAnalysis, Request
from bench.workload import unique_tokens as bench_unique_tokens
from stub import StubTokenizer

FIXTURE = Path(__file__).parent / "fixtures" / "smoke.jsonl"


def req(i, prompt, max_tokens, **kw):
    return Request(id=str(i), prompt=prompt, max_tokens=max_tokens, **kw)


def test_tokenizes_once_and_ids_match_the_tokenizer():
    tok = StubTokenizer()
    reqs = [req(0, "a b c", 4, labels=["x y", "z"]), req(1, "a b d e", 2), req(2, "q", 1, labels=["x y", "z"])]
    job = analyze(reqs, tok)
    assert tok.calls == 2  # one batch call for prompts, one for the single distinct label set
    for it, r in zip(job.requests, reqs):
        assert it.token_ids == tok._ids(r.prompt) and it.prompt_len == len(it.token_ids)
    assert job.requests[0].label_ids == [tok._ids("x y"), tok._ids("z")]
    assert job.requests[1].label_ids is None
    assert job.tokenizer == "stub"


def test_kind_is_exact():
    tok = StubTokenizer()
    reqs = [req(0, "a", 1, labels=["p", "n"]), req(1, "a", 6, labels=["p", "n"]), req(2, "a", 6), req(3, "a", 1)]
    assert [it.kind for it in analyze(reqs, tok).requests] == ["prefill_only", "label", "generate", "prefill_only"]


def test_decode_ratio_counts_decode_forwards_per_prefill_token():
    tok = StubTokenizer()
    job = analyze([req(0, "a b c d", 1), req(1, "a b c d", 9), req(2, "a b", 9)], tok)
    assert [it.decode_ratio for it in job.requests] == [0.0, 2.0, 4.0]
    assert decode_ratio(0, 5) == 4.0  # an empty prompt does not divide by zero


def test_unique_tokens_matches_bench():
    seqs = [[1, 2, 3, 4], [1, 2, 3, 5], [9]]
    assert unique_prompt_tokens(seqs) == 6 == bench_unique_tokens(seqs)
    assert unique_prompt_tokens(seqs, page=2) == 7 == bench_unique_tokens(seqs, page=2)
    assert sum(map(len, seqs)) == 9
    assert lcp([1, 2, 3], [1, 2, 9]) == 2 and lcp([], [1]) == 0


def test_job_metrics_on_the_fixture():
    tok = StubTokenizer()
    job = analyze(read_requests(FIXTURE), tok)
    m = job.metrics
    assert m["requests"] == 8 and m["n_prefill_only"] == 3 and m["n_label"] == 1 and m["n_generate"] == 4
    assert m["n_ignore_eos"] == 2 and m["ideal_prefix_reuse"] > 0
    assert m["prompt_tokens"] == sum(it.prompt_len for it in job.requests)
    assert m["decode_tokens_max"] == sum(it.req.max_tokens - 1 for it in job.requests)
    assert m["decode_chain"] == max(it.req.max_tokens for it in job.requests) - 1
    assert m["job_decode_ratio"] == round(m["decode_tokens_max"] / m["prompt_tokens"], 4)
    # reservations round prompt + max_tokens - 1 up to whole 16-token blocks
    expected = [16 * -(-(it.prompt_len + it.req.max_tokens - 1) // 16) for it in job.requests]
    assert m["kv_reservation_tokens_sum"] == sum(expected) and m["max_reservation_tokens"] == max(expected)
    # the block trie: what an engine sharing whole 16-token blocks can reach, next to the token-level ceiling
    assert isinstance(job.prefix, PrefixTrie) and job.prefix.block_size == BLOCK_SIZE and len(job.prefix.paths) == 8
    assert m["unique_prefill_tokens"] >= m["unique_prompt_tokens"]
    assert 0 <= m[f"ideal_prefix_reuse_page{BLOCK_SIZE}"] <= m["ideal_prefix_reuse"]
    assert m["trie_nodes"] == len(job.prefix.nodes) and m["shared_nodes"] <= m["trie_nodes"]
    text = describe(job)
    assert "8 requests" in text and "decode chain" in text and "prefix trie (16-token blocks)" in text


def test_analysis_makes_no_decisions():
    """No order, group or budget anywhere in what analysis returns: those belong to policy/scheduler. The
    prefix trie is a fact about the prompts and carries no per-run state (block ids, owners, counts)."""
    fields = {f.name for f in dataclasses.fields(JobAnalysis)}
    assert fields == {"requests", "tokenizer", "metrics", "prefix"}
    job = analyze(read_requests(FIXTURE), StubTokenizer())
    for word in ("order", "group", "budget", "batch"):
        assert not any(word in k for k in job.metrics), word
    assert Node.__slots__ == ("id", "depth", "users", "children")
    for name in ("state", "block", "owner", "remaining", "order"):
        assert not hasattr(job.prefix, name) and not any(hasattr(n, name) for n in job.prefix.nodes)


def test_a_prompt_with_no_tokens_is_refused():
    """The first output token is sampled from the prompt's last position, so an empty prompt has nothing to run."""
    with pytest.raises(ValueError, match="request 1: the prompt encodes to no tokens"):
        analyze([req(0, "a", 2), req(1, "   ", 2)], StubTokenizer())
