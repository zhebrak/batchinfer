"""One job through every stage, and the only place the stages are chained (records in schema.py):

    list[Request] --analysis.analyze--> JobAnalysis --policy.decide(+ engine.hardware)--> Policy
                  --engine.run--> Result per request

The CLI, both bench backends, scripts and tests call run(); none of them chains the stages by hand.
"""
from .analysis import analyze, describe
from .policy import decide, describe_policy


def run(requests, tokenizer, engine, cfg, on_result, metrics, log=None):
    """requests: list[Request] (io.read_requests, io.from_rows). engine: Engine or StepEngine, i.e. anything with
    run(job, policy, on_result, metrics) and a hardware attribute (schema.Hardware: what it measured on its card at
    load). cfg: PolicyConfig. on_result: called once per request with its Result, as it finishes. metrics: a fresh
    Metrics; run() fills its analysis and policy sections and timers, the engine the rest. log: called with the
    facts, then the decisions, as text. Returns (job, policy)."""
    with metrics.timer("analysis"):
        job = analyze(requests, tokenizer)
    metrics.set_analysis(job.metrics)
    if log:
        log(describe(job))
    with metrics.timer("policy"):
        policy = decide(job, cfg, engine.hardware)
    metrics.set_policy(policy.to_dict())
    if log:
        log(describe_policy(policy, job))
    engine.run(job, policy, on_result, metrics)
    metrics.finish()
    return job, policy
