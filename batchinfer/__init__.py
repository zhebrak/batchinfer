"""Hand-rolled batch inference over a JSONL of prompts:

    schema.py       the records handed between stages (Request -> JobAnalysis -> Policy -> Step -> Result), who owns what
    flow.py         the one place the stages are chained: analyze -> decide -> engine.run
    io.py           read the client-visible fields, append results
    analysis.py     tokenize once and characterise the job: facts and metrics, no decisions
    prefix.py       the prompts as a trie of KV blocks: which requests share which blocks (a fact, built by analysis)
    policy.py       decisions made once, before the first step: order, fixed groups, admission, budget, sharing
    scheduler.py    decisions made at every step: who is admitted, which tokens go into the step, who
                    computes a shared block and when it is released
    kv.py           paged KV block bookkeeping (pure Python)
    model.py        loading the HF model; the allocator config, versions and stop ids both engines share
    executor.py     runs one step: HF layers with our paged attention; no decisions
    step_engine.py  the batchinfer engine (ours): the loop scheduler -> executor -> commit
    naive.py        the naive engine (the baseline): fixed groups in arrival order as left-padded HF batches
    metrics.py      how this invocation performed; one owner per section

    python -m batchinfer run --input W.jsonl --output results/W/<model>/<gpu>/batchinfer/outputs.jsonl
    python -m batchinfer analyze --input W.jsonl
    python -m bench.run W.jsonl --backend batchinfer      # one row; python -m bench.suite for the comparison
"""
