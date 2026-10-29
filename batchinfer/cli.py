"""Batch inference over a JSONL of prompts.

    python -m batchinfer run --input workloads/mixed-quick.jsonl \\
        --output results/mixed-quick/Qwen3-1.7B/<gpu>/batchinfer/outputs.jsonl
    python -m batchinfer run --engine naive --input ... --output ...
    python -m batchinfer analyze --input workloads/mixed-quick.jsonl
    python -m batchinfer run --config policy.json --input ... --output ...

`--config` names a JSON object of PolicyConfig fields (schema.py), e.g. {"order": "prefix_dfs",
"prefill_budget": "auto"}: a flag given on the command line beats the file, and the file beats the
engine's defaults. metrics.json records the resolved values under meta.args.

`run` reads the client-visible fields, loads the model, then hands the job to flow.run (analysis: facts and
metrics; the engine's pool and graphs built for the job; policy: decisions; the chosen engine, appending results
as they finish), then prints a summary and writes metrics.json next to the output (`python -m bench.report --html`
renders its page). `analyze` prints the analysis metrics and then, separately, what the policy would decide for
the given flags; it needs the tokenizer but neither torch nor a GPU.

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
from .schema import (ADMISSIONS, CACHE, DEFAULT_BATCH_SIZE, DEFAULT_MAX_BATCH_TOKENS, ENGINES, MEMORY, NAIVE_ORDER,
                     ORDERS, SHARING, PolicyConfig)

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"  # every run under 4 minutes; the published rows are Qwen3-14B


def budget_arg(text):
    return text if text in ("auto", "ridge") else int(text)


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
                        "widest step the engine sizes for); 'auto' (job prompt tokens / decode chain) or 'ridge' "
                        "(steps with decode rows filled to the card's measured ridge, see --ridge-tokens)")
    p.add_argument("--sharing", choices=SHARING, default=None,
                   help="batchinfer engine: how the KV of common prompt prefixes is shared: block (each shared block "
                        "computed once and read by every request that has it, the default), token (block, and the "
                        "leading tokens a request's own block shares with a computed sibling block copied rather "
                        "than recomputed), none (every block private; the naive engine's only setting)")
    p.add_argument("--cache", choices=CACHE, default=None,
                   help="batchinfer engine: what happens to a shared block when its last user finishes: released (job, "
                        "the default) or kept for the next job this process runs, evicted oldest first under pressure "
                        "(radix); one job alone gains nothing from radix")
    p.add_argument("--watermark-steps", type=int, default=None,
                   help="batchinfer engine, memory=preempt: steps of decode growth admission leaves room for (default 1)")
    p.add_argument("--memory", choices=MEMORY, default=None,
                   help="batchinfer engine: how a request's KV is held: reserve (its whole life's blocks at admission, "
                        "the default), grow (decode blocks as decode reaches them, admission against a projection of "
                        "every live request's future demand, so nothing is preempted), preempt (grow under a "
                        "watermark, the youngest request preempted and recomputed when the pool runs dry)")
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
    r.add_argument("--resume", action="store_true",
                   help="keep --output's results from an earlier run of the same input and run only the requests it "
                        "did not finish, appending; nothing loads when nothing is left")
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--attn", default="sdpa", help="naive engine: HF attn_implementation")
    r.add_argument("--reserve-gb", type=reserve_arg, default="auto",
                   help="batchinfer engine: GiB of the free memory after load left for activations, the KV pool taking "
                        "the rest; 'auto' (default) measures the largest step and sizes the pool to 95%% of the card")
    r.add_argument("--num-blocks", type=int, default=None,
                   help="batchinfer engine: KV pool size in blocks, the same on every card (default: from --reserve-gb)")
    r.add_argument("--block-size", type=int, default=None,
                   help="batchinfer engine: tokens per KV block, a multiple of 16 up to 256 (default 16); the prefix "
                        "trie shares whole blocks, so a larger block shares less and reserves more per request")
    # the executor's build options: every default lives in Executor.__init__, so an unset flag is not passed
    r.add_argument("--quant", choices=("none", "fp8"), default=None,
                   help="batchinfer engine: the fused layers' projection weights in fp8 (e4m3, a scale per output "
                        "channel, activations quantised per token; a GPU with FP8 tensor cores); default none")
    r.add_argument("--kv-dtype", choices=("auto", "bf16", "fp8"), default=None,
                   help="batchinfer engine: the KV pool's dtype; fp8 (e4m3, static unit scales as vLLM's uncalibrated "
                        "kv_cache_dtype=fp8) halves the bytes per token, so the pool holds twice the tokens; "
                        "FlashAttention-3 or the torch backend")
    r.add_argument("--pool", choices=("static", "elastic", "contiguous"), default=None,
                   help="batchinfer engine: the KV pool: static (one tensor); elastic (CUDA virtual memory: physical "
                        "pieces held only under live blocks, given back as rows finish; graphs and admission unchanged); "
                        "contiguous (every live sequence reads its KV as one contiguous run through a virtual region of "
                        "its own, attention without a block table; the block is the mapping unit, 128 tokens for Qwen3)")
    r.add_argument("--attention", choices=("fa", "torch", "cascade"), default=None,
                   help="batchinfer engine: fa (vLLM's FlashAttention kernel, the GPU default), torch (SDPA per row), "
                        "or cascade (rows behind one shared prefix read it once per step: two FlashAttention calls "
                        "merged exactly)")
    r.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=None,
                   help="batchinfer engine: plan and launch a step while the previous one runs (a decode row gathers "
                        "its token from the previous step's output on the device; finishes land a step late)")
    r.add_argument("--ridge-tokens", type=int, default=None,
                   help="batchinfer engine: the prefill tokens a graphed decode step absorbs at about no cost on this "
                        "card, for --prefill-budget ridge; default: the card table's measurement "
                        "(scripts/ridge_calibration.py), none for an uncalibrated card")
    r.add_argument("--fa-version", choices=("auto", "2", "3"), default=None,
                   help="batchinfer engine: FlashAttention version; auto (the default) runs 3 where it can, what vLLM "
                        "runs on Hopper, else 2")
    r.add_argument("--layers", choices=("auto", "fused", "compiled", "hf"), default=None,
                   help="batchinfer engine: the forward: fused (Qwen3's layers with vLLM's kernels, ~11 launches a layer "
                        "not ~57), compiled (HF's layers under torch.compile, any decoder the checks accept), hf (HF's "
                        "layers eagerly); default auto, the first that can run")
    r.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=None,
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
    return cmd_run(args) if args.cmd == "run" else cmd_analyze(args)


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
    if args.sharing is None:
        args.sharing = PolicyConfig.sharing if ours else "none"
    if args.chunk_prefill is None:
        args.chunk_prefill = PolicyConfig.chunk_prefill
    if args.memory is None:
        args.memory = PolicyConfig.memory
    if args.cache is None:
        args.cache = PolicyConfig.cache
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
                        sharing=args.sharing, cache=args.cache, memory=args.memory,
                        **({"watermark_steps": args.watermark_steps} if args.watermark_steps is not None else {})).validate()


def cmd_analyze(args):
    cfg = policy_config(args)  # analyze has no engine: an "auto" budget is the no-card default
    job = analyze(read_requests(args.input, args.max_tokens, args.limit), load_tokenizer(args.model))
    print("analysis (facts):\n" + describe(job))
    print("policy (decisions):\n" + describe_policy(decide(job, cfg), job, show=args.show))


EXECUTOR_FLAGS = ("fa_version", "layers", "cuda_graphs", "block_size", "ridge_tokens", "quant", "kv_dtype", "overlap",
                  "attention", "pool")


def executor_opts(args):
    """The batchinfer executor's build options named on the command line; an unset one takes the executor's default."""
    opts = {k: getattr(args, k) for k in EXECUTOR_FLAGS if getattr(args, k) is not None}
    if opts.get("fa_version") in ("2", "3"):
        opts["fa_version"] = int(opts["fa_version"])
    return opts


def cmd_run(args):
    cfg = policy_config(args)
    m = Metrics()
    requests = read_requests(args.input, args.max_tokens, args.limit)
    writer = ResultWriter(args.output, resume=args.resume)
    if args.resume:  # a crashed run's results stay; only the requests it did not finish run, appended after them
        requests = [r for r in requests if r.id not in writer.done]
        print(f"resumed: {len(writer.done)} done, {len(requests)} to go", file=sys.stderr)
        m.meta["resumed"] = len(writer.done)
        if not requests:
            writer.f.close()
            return
    with m.timer("load"):
        tok = load_tokenizer(args.model)
        if args.engine == "batchinfer":
            from .executor import Executor
            from .step_engine import StepEngine
            # build=False: the pool and graphs are built once the job is analysed (flow.run -> StepEngine.prepare)
            engine = StepEngine(Executor(args.model, dtype=args.dtype, reserve_gb=args.reserve_gb,
                                         num_blocks=args.num_blocks, build=False, **executor_opts(args)), tok)
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
                  sharing=args.sharing if args.engine == "batchinfer" else None,  # naive shares nothing
                  input=str(args.input), output=str(args.output),
                  **versions(), timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})  # every flag
    with writer:
        flow.run(requests, tok, engine, cfg, writer, m, log=lambda text: print(text, file=sys.stderr))
    metrics_path = args.metrics or args.output.with_name("metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    d = m.to_dict()
    metrics_path.write_text(json.dumps(d, indent=2) + "\n")
    print(m.summary())
    print(f"{writer.n} results in {args.output}; metrics in {metrics_path}")
