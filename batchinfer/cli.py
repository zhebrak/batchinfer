"""Batch inference over a JSONL of prompts.

    python -m batchinfer run --input workloads/mixed-quick.jsonl \\
        --output results/mixed-quick/Qwen3-1.7B/A100-SXM4-40GB/batchinfer/outputs.jsonl
    python -m batchinfer run --engine naive --input ... --output ...
    python -m batchinfer analyze --input workloads/mixed-quick.jsonl
    python -m batchinfer run --config policy.json --input ... --output ...

`--config` names a JSON object of PolicyConfig fields (schema.py), e.g. {"order": "prefix_dfs",
"prefill_budget": "auto"}: a flag given on the command line beats the file, and the file beats the
engine's defaults. metrics.json records the resolved values under meta.args.

`run` reads the client-visible fields, loads the model, then hands the job to flow.run (analysis: facts and
metrics; policy: decisions; the chosen engine, appending results as they finish), then prints a summary and
writes metrics.json and its report.html (bench/report.py) next to the output. `analyze` prints the analysis
metrics and then, separately, what the policy would decide for the given flags; it needs the tokenizer
but neither torch nor a GPU.

Engines: `batchinfer` (ours, the default) packs all decode rows plus prefill chunks into every forward over a
paged KV pool; `naive` is the baseline, fixed groups in arrival order run as left-padded HF batches.
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from . import flow
from .analysis import analyze, describe
from .io import ResultWriter, read_requests
from .metrics import Metrics
from .policy import decide, describe_policy
from .schema import ADMISSIONS, DEFAULT_BATCH_SIZE, DEFAULT_MAX_BATCH_TOKENS, ENGINES, NAIVE_ORDER, ORDERS, PolicyConfig

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"  # the iteration model: every run under 4 minutes; README's headline rows are 8B


def budget_arg(text):
    return text if text in ("auto", "adaptive") else int(text)


def reserve_arg(text):
    return text if text == "auto" else float(text)


def common(p):
    p.add_argument("--input", required=True, type=Path,
                   help="JSONL, one request per line: prompt, optional id / max_tokens / ignore_eos / labels")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--config", type=Path, help="JSON object of PolicyConfig fields; flags given here beat it")
    p.add_argument("--engine", choices=ENGINES, default=None, help="default batchinfer (ours); naive is the baseline")
    p.add_argument("--order", choices=ORDERS, default=None,
                   help=f"admission order; default {PolicyConfig.order} (batchinfer) or input (naive). "
                        "max_tokens_desc fixed groups also close at a dead-slot cut (see policy.py)")
    p.add_argument("--max-batch-tokens", type=budget_arg, default=None,
                   help="fixed groups: padded end-state tokens per group, rows x (longest prompt + longest max_tokens), "
                        "or 'auto' (default): what the engine measured on this card, the naive engine's probed fit or "
                        f"the batchinfer engine's KV pool ({DEFAULT_MAX_BATCH_TOKENS:,} for analyze, which has no card); see "
                        "policy.auto_max_batch_tokens")
    p.add_argument("--batch-size", type=int, default=None,
                   help=f"fixed groups: max rows per group (default {DEFAULT_BATCH_SIZE})")
    p.add_argument("--admission", choices=ADMISSIONS, default=None,
                   help=f"default {PolicyConfig.admission} (batchinfer) or groups (naive, the only admission it runs)")
    p.add_argument("--chunk-prefill", action=argparse.BooleanOptionalAction, default=None,
                   help="batchinfer engine (default on)")
    p.add_argument("--prefill-budget", type=budget_arg, default=None,
                   help=f"batchinfer engine: max prefill tokens per step (default {PolicyConfig.prefill_budget}, the "
                        "widest step the engine sizes for); 'auto' (job prompt tokens / decode chain) or 'adaptive' "
                        "(auto, raised per step when the decode work left shrinks)")
    p.add_argument("--prefix-sharing", action=argparse.BooleanOptionalAction, default=None,
                   help="batchinfer engine: compute each shared prompt block once and read it from every request that "
                        "has it (default on for the batchinfer engine, off for the naive engine, which shares nothing; "
                        "--no-prefix-sharing makes every block private)")
    p.add_argument("--max-tokens", type=int, default=128, help="for requests without max_tokens (default %(default)s)")
    p.add_argument("--limit", type=int, help="only the first N requests")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m batchinfer", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse, decide, run, write results and metrics")
    common(r)
    r.add_argument("--output", required=True, type=Path, help="results JSONL, one line per request as it finishes")
    r.add_argument("--metrics", type=Path, help="default: metrics.json next to --output")
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--attn", default="sdpa", help="naive engine: HF attn_implementation")
    r.add_argument("--reserve-gb", type=reserve_arg, default="auto",
                   help="batchinfer engine: GiB of the free memory after load left for activations, the KV pool taking "
                        "the rest; 'auto' (default) measures the largest step and sizes the pool to 95%% of the card")
    r.add_argument("--num-blocks", type=int, default=None,
                   help="batchinfer engine: KV pool size in 16-token blocks, the same on every card (default: from "
                        "--reserve-gb)")
    r.add_argument("--fa-version", type=int, choices=(2, 3), default=2,
                   help="batchinfer engine: FlashAttention version; 3 is what vLLM runs on Hopper (sm90 only)")
    r.add_argument("--fused-layers", action=argparse.BooleanOptionalAction, default="auto",
                   help="batchinfer engine: Qwen3's layers fused with vLLM's kernels (~11 launches a layer, not ~57); "
                        "default: on where it can run (Qwen3 on a GPU)")
    r.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default="auto",
                   help="batchinfer engine: replay decode-only steps from CUDA graphs captured at load; default: on "
                        "where it can run (GPU, FlashAttention-2)")
    r.add_argument("--no-probe", action="store_true",
                   help="naive engine: skip the memory probe at an explicit --max-batch-tokens ('auto' is the probe)")
    a = sub.add_parser("analyze", help="print job metrics, then what the policy would decide")
    common(a)
    a.add_argument("--show", type=int, default=3, metavar="N", help="fixed groups: print the first N groups")
    args = p.parse_args(argv)
    try:  # PolicyConfig.validate() is the one place a config an engine cannot run is refused: here, before any load
        resolve_defaults(args)
        policy_config(args)
    except (ValueError, OSError) as e:  # a bad value, a bad --config file, or a missing one
        p.error(str(e))
    if args.cmd == "run" and args.engine == "naive" and args.no_probe and args.max_batch_tokens == "auto":
        p.error("--no-probe needs an explicit --max-batch-tokens: 'auto' is what the probe measures")
    return run(args) if args.cmd == "run" else analyze_cmd(args)


CONFIG_ARGS = {"max_batch_size": "batch_size"}  # PolicyConfig field -> flag attribute, where the names differ


def apply_config(args, path):
    """Fills the policy flags the command line left unset from a JSON object of PolicyConfig fields. Values are not
    converted here: PolicyConfig.validate() refuses a wrong type as it would from a flag."""
    cfg = json.loads(Path(path).read_text())
    fields = [f.name for f in dataclasses.fields(PolicyConfig)]
    if not isinstance(cfg, dict) or set(cfg) - set(fields):
        got = sorted(set(cfg) - set(fields)) if isinstance(cfg, dict) else cfg
        raise ValueError(f"{path}: expected a JSON object of PolicyConfig fields ({', '.join(fields)}), not {got!r}")
    for key, value in cfg.items():
        if getattr(args, CONFIG_ARGS.get(key, key)) is None:
            setattr(args, CONFIG_ARGS.get(key, key), value)


def resolve_defaults(args):
    """Policy flags left unset come from --config, then from their engine's defaults. The batchinfer engine follows
    PolicyConfig; the naive engine runs fixed groups in arrival order (order=input) and shares nothing."""
    if getattr(args, "config", None):
        apply_config(args, args.config)
    args.engine = args.engine or PolicyConfig.engine
    ours = args.engine == "batchinfer"
    args.order = args.order or (PolicyConfig.order if ours else NAIVE_ORDER)
    args.admission = args.admission or (PolicyConfig.admission if ours else "groups")
    if args.prefix_sharing is None:
        args.prefix_sharing = PolicyConfig.prefix_sharing if ours else False
    if args.chunk_prefill is None:
        args.chunk_prefill = PolicyConfig.chunk_prefill
    if args.prefill_budget is None:
        args.prefill_budget = PolicyConfig.prefill_budget
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE
    if args.max_batch_tokens is None:
        args.max_batch_tokens = PolicyConfig.max_batch_tokens  # "auto": resolved from what the engine measures
    return args


def load_tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)


def policy_config(args):
    """The flags as a validated PolicyConfig. max_batch_tokens may be "auto": policy resolves it from what the engine
    measured on its card (engine.hardware), after load."""
    return PolicyConfig(engine=args.engine, order=args.order, admission=args.admission,
                        chunk_prefill=args.chunk_prefill, prefill_budget=args.prefill_budget,
                        max_batch_tokens=args.max_batch_tokens, max_batch_size=args.batch_size,
                        prefix_sharing=args.prefix_sharing).validate()


def analyze_cmd(args):
    cfg = policy_config(args)  # analyze has no engine: an "auto" budget is the no-card default
    job = analyze(read_requests(args.input, args.max_tokens, args.limit), load_tokenizer(args.model))
    print("analysis (facts):\n" + describe(job))
    print("policy (decisions):\n" + describe_policy(decide(job, cfg), job, show=args.show))


def executor_opts(args):
    """The batchinfer executor's build options named on the command line."""
    return {"fa_version": args.fa_version, "fused_layers": args.fused_layers, "cuda_graphs": args.cuda_graphs}


def run(args):
    cfg = policy_config(args)
    m = Metrics()
    requests = read_requests(args.input, args.max_tokens, args.limit)
    with m.timer("load"):
        tok = load_tokenizer(args.model)
        if args.engine == "batchinfer":
            from .executor import Executor
            from .step_engine import StepEngine
            engine = StepEngine(Executor(args.model, dtype=args.dtype, reserve_gb=args.reserve_gb,
                                         num_blocks=args.num_blocks, **executor_opts(args)), tok)
        else:
            from .naive import NaiveEngine
            engine = NaiveEngine(args.model, tok, dtype=args.dtype, attn=args.attn)
    from .model import versions
    if args.engine == "naive" and (cfg.max_batch_tokens == "auto" or not args.no_probe):
        with m.timer("probe"):  # the engine records what it measured in metrics.memory when it runs
            if cfg.max_batch_tokens == "auto":
                engine.fit()
            else:
                engine.probe(cfg.max_batch_tokens)
    m.meta.update(model=args.model, dtype=args.dtype, engine=args.engine, order=args.order,
                  prefix_sharing=args.prefix_sharing if args.engine == "batchinfer" else None,  # naive shares nothing
                  input=str(args.input), output=str(args.output),
                  **versions(), timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})  # every flag
    with ResultWriter(args.output) as writer:
        flow.run(requests, tok, engine, cfg, writer, m, log=lambda text: print(text, file=sys.stderr))
    metrics_path = args.metrics or args.output.with_name("metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    d = m.to_dict()
    metrics_path.write_text(json.dumps(d, indent=2) + "\n")
    print(m.summary())
    print(f"{writer.n} results in {args.output}; metrics in {metrics_path}")
    from bench.report import render  # bench is the reporting layer above the engine; only the CLI reaches up to it
    files = [metrics_path.name] + ([args.output.name] if args.output.parent == metrics_path.parent else [])
    metrics_path.with_name("report.html").write_text(render(d, metrics_path.parent.name, files))
