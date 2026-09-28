"""flow: the stages hand over one record each, and every Metrics section has one owner. Pure Python: a fake
engine records what it was handed and returns one Result per request."""
from pathlib import Path

import pytest

from batchinfer import flow
from batchinfer.io import read_requests
from batchinfer.metrics import FLAT_KEYS, STEP_FIELDS, Metrics
from batchinfer.schema import Group, JobAnalysis, Policy, PolicyConfig, Result
from stub import StubTokenizer

FIXTURE = Path(__file__).parent / "fixtures" / "smoke.jsonl"


class FakeEngine:
    """Takes what every engine takes and fills only the sections an engine owns."""
    flat_keys = FLAT_KEYS
    hardware = None  # measured nothing: an "auto" budget is policy's no-card default

    def __init__(self):
        self.handed = None

    def run(self, job, policy, on_result, metrics):
        self.handed = (job, policy)
        metrics.flat_keys = self.flat_keys
        with metrics.timer("inference"):
            for i in policy.admission_order:
                it = job.requests[i]
                metrics.add(requests=1, prompt_tokens=it.prompt_len, output_tokens=1)
                on_result(Result(id=it.req.id, text="x", token_ids=[1], prompt_tokens=it.prompt_len, output_tokens=1,
                                 finish_reason="length", group=None, latency_s=0.0))


def run(cfg):
    tok, engine, out, m, logged = StubTokenizer(), FakeEngine(), [], Metrics(), []
    job, policy = flow.run(read_requests(FIXTURE), tok, engine, cfg, out.append, m, log=logged.append)
    return job, policy, engine, out, m, logged, tok


def test_each_stage_hands_the_next_one_record():
    job, policy, engine, out, m, logged, tok = run(PolicyConfig())
    assert isinstance(job, JobAnalysis) and isinstance(policy, Policy)
    assert engine.handed == (job, policy)  # the engine gets the job and the decisions, never the config
    label_sets = {tuple(it.req.labels) for it in job.requests if it.req.labels}
    assert tok.calls == 1 + len(label_sets) == 3  # analysis encodes once: one prompt batch, one call per label set
    assert sorted(r.id for r in out) == sorted(it.req.id for it in job.requests)
    assert len(logged) == 2 and logged[0].startswith("8 requests") and logged[1].startswith("policy (batchinfer engine): order=")


def measured_keys(m):
    """Every key a measured section can hold: counts, rates, and step rates over a one-step trace."""
    trace = Metrics()
    trace.record_step(**dict.fromkeys(STEP_FIELDS, 0))
    return set(m.counts) | set(m.rates()) | set(trace.step_rates())


NAIVE_GROUPS = PolicyConfig(engine="naive", order="max_tokens_desc", admission="groups", prefix_sharing=False,
                             max_batch_tokens=4096, max_batch_size=3)


@pytest.mark.parametrize("cfg", [PolicyConfig(), PolicyConfig(admission="groups"), NAIVE_GROUPS],
                         ids=["batchinfer", "batchinfer-groups", "naive"])
def test_no_key_is_both_decided_and_measured(cfg):
    job, policy, _, _, m, _, _ = run(cfg)
    decided = set(policy.to_dict())
    assert decided & set(job.metrics) == set() and decided & measured_keys(m) == set()


def test_metrics_sections_have_one_owner():
    job, policy, _, _, m, _, _ = run(PolicyConfig())
    assert m.analysis == job.metrics and m.policy == policy.to_dict()
    timings = m.timings()
    assert {"analysis_s", "policy_s", "inference_s", "total_s"} <= set(timings)
    d = m.to_dict()
    assert d["policy"] == policy.to_dict() and d["analysis"] == job.metrics and "plan" not in d


def test_continuous_admission_reports_no_fixed_groups_budget():
    _, policy, _, _, m, _, _ = run(PolicyConfig(admission="continuous", max_batch_tokens=4096))
    assert policy.groups is None and policy.max_batch_tokens is None
    assert policy.to_dict()["max_batch_tokens"] is None and policy.to_dict()["groups"] is None
    flat = m.flat_stats()
    assert flat["max_batch_tokens"] is None and flat["groups"] is None  # the budget played no part in this row
    assert "groups" not in m.to_dict()["volume"]


def test_fixed_groups_are_groups_and_their_budget_is_a_column():
    _, policy, engine, _, m, _, _ = run(NAIVE_GROUPS)
    assert all(isinstance(g, Group) for g in policy.groups)
    assert policy.admission_order == tuple(i for g in policy.groups for i in g.members)
    flat = m.flat_stats()
    assert flat["max_batch_tokens"] == 4096 and flat["groups"] == len(policy.groups)


def test_a_naive_row_records_no_batchinfer_engine_setting():
    _, policy, _, _, m, logged, _ = run(NAIVE_GROUPS)
    recorded = m.to_dict()["policy"]
    assert recorded["engine"] == "naive" and recorded["prefix_sharing"] is None
    assert recorded["chunk_prefill"] is None and recorded["prefill_budget"] is None
    assert recorded["prefill_budget_auto"] is None and recorded["prefill_budget_adaptive"] is None
    assert "prefill_budget" not in logged[1] and "chunk_prefill" not in logged[1]
