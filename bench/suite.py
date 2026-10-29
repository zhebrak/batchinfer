"""Every engine and option against the baselines, with one model load per engine configuration.

    python -m bench.suite                                   # Qwen/Qwen3-1.7B, the five quick workloads, vLLM + batchinfer
    python -m bench.suite --defaults-only                   # batchinfer defaults only: the quick "did it help" loop
    python -m bench.suite --engines naive --workloads mixed # the one-time naive baseline, one workload per call
    python -m bench.suite --workloads mixed --variant sharing=none --variant order=input,prefill_budget=auto
    python -m bench.suite --once                            # the first of three runs: the once-only arms as well
    python -m bench.suite --model Qwen/Qwen3-14B --workloads mixed --engines batchinfer,vllm
    python -m bench.suite --dry-run                         # the load groups, rows and reuse decisions; nothing runs
    python -m bench.suite --column "vLLM cache on" --column batchinfer   # only these arms, each on its own workloads

Why it is fast. A row's timed pass on a small model takes seconds, while loading the engine takes 10 s (batchinfer)
to 35-75 s (vLLM, whose compile and CUDA-graph capture dominate). So rows are grouped by engine and load options (a
load group), and each group runs in one worker process: one load, one warmup, then every workload and every policy
variant of that engine as its own timed pass (bench.run's prepare / load / timed_pass). Baseline rows (naive, vLLM)
are reused while they are still valid (reuse()); batchinfer rows always run, because they measure the code at hand.

Why it is fair. Before every pass the backend's reset() returns it to the state a fresh load leaves: vLLM drops its
prefix cache, the naive engine empties the allocator cache as its probe does, and batchinfer re-runs its executor's
probe after emptying it. Our engines keep no job state between runs (the prefix trie and every KV block live and
die with one flow.run). Each group runs in a process of its own, and the next one starts only when the GPU is idle
again (wait_for_idle_gpu), since vLLM's engine-core process can outlive its worker and batchinfer sizes its KV pool
from free memory at load.

Rows land where bench.run puts them (results/<workload>/<model>/<gpu>/<label>/), and every row but vLLM's prefix
cache-on one is compared against that row when it exists. The run ends with bench.report's table of this model and
GPU's rows for the workloads it ran.
"""
import argparse
import hashlib
import inspect
import json
import math
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bench.backends import resolve
from bench.results import ROOT, run_dir
from bench.run import (engine_name, gpu_name, load, model_revision, prepare, reference_path, row_label, shell,
                       timed_pass, versions)
from bench import report
from bench.workload import (ALIASES, DIVERSE_WORKLOADS, LONG_WORKLOADS, PRESETS, SIZES, SUITE_WORKLOADS, SWEEP_WORKLOADS, kv_bytes_per_token,
                            meta_path, parse_value, workload_name)
from batchinfer.kv import DEFAULT_BLOCK_SIZE

# Small by default: Qwen3-1.7B loads in about 10 s, and it shares the larger Qwen3 models' tokenizer and chat template,
# so the workload files are the same bytes for all of them.
MODEL = "Qwen/Qwen3-1.7B"
SIZE = "quick"
ENGINES = ("vllm", "naive", "batchinfer")  # in run order: vLLM's cache-on row is every other row's reference
# naive is opt-in: its default arrival order is its slowest, several minutes a workload on mixed and generate even on
# 1.7B. Built once per GPU and model, then reused.
DEFAULT_ENGINES = ("vllm", "batchinfer")
REFERENCE = "vllm"  # the label every other row compares against: this GPU's vLLM row with its prefix cache on


@dataclass(frozen=True)
class Arm:
    """One configuration the suite measures: an engine with its options, on the workloads it tells something about
    (None: every workload), under the column name the results tables show. once: a row for the options table that
    runs one time, not in the median-of-three set. cap: a KV pool cap as a share of the workload's KV demand
    (the sidecar's kv_demand_tokens), resolved per workload into the engine's own option (resolve_cap), so both
    engines get the same token capacity by meaning: the pool-bound regime at the same ratio on every capped workload.
    warm: the row times the workload's odd-numbered requests after an untimed pass over the even-numbered ones with
    no reset between (bench.run.timed_pass): a warm cache over new prompts, for the engines' cross-job caches."""
    engine: str
    opts: dict
    column: str
    workloads: tuple = None
    once: bool = False
    cap: float = None
    warm: bool = False
    reference: str = None  # the vLLM column this arm is read against (tables); default by cap and warm


CAP = 0.25  # the capped arms' pool: a quarter of the workload's KV demand
# mixed, decode-heavy and prefill-heavy under the cap: pf50x32's 2,048-token prompts are most of what its requests
# hold, so decode growth (what grow and preempt manage) is small there and prompt blocks bind
CAPPED = ("pool", "generate-natural", "dec50x32", "pf50x32")
# the pool-binding curve: the memory modes and vLLM at a half, a quarter (the arms above) and an eighth of the demand
BINDING = ("pool", "dec50x32", "pf50x32")
# the suite's default workloads: the README's, in the report's row order (bench/workload.py), and the pool-binding ones
WORKLOADS = SUITE_WORKLOADS + tuple(w for w in BINDING if w not in SUITE_WORKLOADS)
CAPS = {0.5: " 1/2", 0.125: " 1/8"}  # the extra caps and their column suffix (the quarter's columns carry none)
GRID_WIDE = ("dec30x32", "dec50x32", "dec90x32", "bal30x32", "bal50x32", "bal90x32")  # the grid's wide fan-out points
ARMS = (
    # vLLM: prefix cache on (the reference), off, and on without torch.compile or CUDA graphs (enforce_eager), the
    # like-for-like for batchinfer rows that run eagerly; its default max_num_seqs is 1,024 on 80 GB cards, batchinfer's cap
    Arm("vllm", {}, "vLLM cache on"),
    Arm("vllm", {"enable_prefix_caching": False}, "vLLM cache off"),
    Arm("vllm", {"enforce_eager": True}, "vLLM eager", ("mixed", "classify", "generate")),
    Arm("vllm", {}, "vLLM capped", CAPPED, cap=CAP),
    Arm("vllm", {}, "vLLM warm", ("pool",), warm=True),
    Arm("vllm", {"quantization": "fp8"}, "vLLM FP8", ("mixed", "classify", "generate", "pool")),
    Arm("vllm", {"quantization": "fp8", "kv_cache_dtype": "fp8"}, "vLLM FP8 KV", ("mixed", "pool") + LONG_WORKLOADS + DIVERSE_WORKLOADS),
    # capped at the bf16 rows' bytes: an fp8 cache holds twice the tokens in them, which is its lever
    Arm("vllm", {"quantization": "fp8", "kv_cache_dtype": "fp8"}, "vLLM FP8 KV capped", CAPPED, cap=CAP),
    Arm("vllm", {"block_size": 32}, "vLLM block 32", ("mixed", "pool")),
    Arm("vllm", {"block_size": 64}, "vLLM block 64", ("mixed", "pool")),
    Arm("naive", {"order": "max_tokens_desc", "max_batch_tokens": 65536}, "naive",
        ("mixed", "classify", "generate", "sweep0", "sweep90"), once=True),
    # batchinfer: the defaults, the trie off, the non-optimised setup (HF's layers, eager), the capped pool
    Arm("batchinfer", {}, "batchinfer"),
    Arm("batchinfer", {"sharing": "none"}, "no trie"),
    Arm("batchinfer", {"layers": "hf", "cuda_graphs": False}, "HF layers, eager", ("mixed", "classify")),
    Arm("batchinfer", {"layers": "compiled"}, "compiled layers", ("mixed", "classify", "generate")),
    Arm("batchinfer", {"cuda_graphs": False}, "no graphs", ("mixed",)),
    # the attention paths also run on the long-context workloads, where attention is most of a decode step
    Arm("batchinfer", {"fa_version": 2}, "FlashAttention-2", ("mixed", "generate") + LONG_WORKLOADS),
    Arm("batchinfer", {"overlap": False}, "no overlap", ("mixed", "generate", "pool")),
    Arm("batchinfer", {"attention": "cascade"}, "cascade", ("mixed", "sweep90", "pool", "long50x8") + GRID_WIDE),
    Arm("batchinfer", {"pool": "elastic"}, "elastic pool", ("mixed", "pool")),
    # contiguous KV beside the paged path at the same unit (its block size): the end-to-end pair; what the block table
    # alone costs is the profile's three-arm measurement (decode_step_profile --pool contiguous), layout held equal
    Arm("batchinfer", {"pool": "contiguous"}, "contiguous KV", ("mixed", "generate", "pool") + LONG_WORKLOADS),
    Arm("batchinfer", {"block_size": 128}, "block 128", ("mixed", "generate", "pool") + LONG_WORKLOADS),
    Arm("batchinfer", {}, "capped", CAPPED, cap=CAP),
    # the memory modes: on the quick rows (chain-bound: the same wall is the check) and the capped rows (pool-bound)
    # and the many-documents rows (DIVERSE_WORKLOADS), whose distinct prefixes alone exceed the pool
    Arm("batchinfer", {"memory": "grow"}, "grow", ("mixed", "generate-natural", "pool") + DIVERSE_WORKLOADS),
    Arm("batchinfer", {"memory": "preempt"}, "preempt", ("mixed", "generate-natural", "pool") + DIVERSE_WORKLOADS),
    Arm("batchinfer", {"memory": "preempt", "watermark_steps": 32}, "preempt lookahead 32", ("mixed", "generate-natural", "pool") + GRID_WIDE + DIVERSE_WORKLOADS),
    Arm("batchinfer", {"memory": "grow"}, "capped grow", CAPPED, cap=CAP),
    Arm("batchinfer", {"memory": "preempt"}, "capped preempt", CAPPED, cap=CAP),
    Arm("batchinfer", {"memory": "preempt", "watermark_steps": 32}, "capped preempt lookahead 32", CAPPED, cap=CAP),
    *(Arm(engine, opts, column + sfx, BINDING, cap=cap, reference=None if engine == "vllm" else "vLLM capped" + sfx)
      for cap, sfx in CAPS.items()
      for engine, opts, column in (("vllm", {}, "vLLM capped"), ("batchinfer", {}, "capped"),
                                   ("batchinfer", {"memory": "grow"}, "capped grow"),
                                   ("batchinfer", {"memory": "preempt"}, "capped preempt"))),
    # sharing at token granularity, the block size, and both; the radix cache warm across jobs
    Arm("batchinfer", {"sharing": "token"}, "token sharing", ("mixed", "classify", "pool") + GRID_WIDE),
    Arm("batchinfer", {"block_size": 32}, "block 32", ("mixed", "pool")),
    Arm("batchinfer", {"block_size": 64}, "block 64", ("mixed", "pool")),
    Arm("batchinfer", {"block_size": 64, "sharing": "token"}, "block 64, token sharing", ("mixed", "pool")),
    Arm("batchinfer", {"cache": "radix"}, "radix warm", ("pool",), warm=True),
    # the weights in fp8 (vLLM's fp8 row is its own online per-tensor scheme: a different quantisation, so the pair
    # is read as two fp8 engines, not as one scheme on two engines)
    Arm("batchinfer", {"quant": "fp8"}, "FP8 weights", ("mixed", "classify", "generate", "pool")),
    Arm("batchinfer", {"quant": "fp8", "kv_dtype": "fp8"}, "FP8 weights and KV", ("mixed", "pool") + LONG_WORKLOADS + DIVERSE_WORKLOADS),
    Arm("batchinfer", {"quant": "fp8", "kv_dtype": "fp8"}, "capped FP8 weights and KV", CAPPED, cap=CAP,
        reference="vLLM FP8 KV capped"),
    # every mode that helps under memory pressure at once, against vLLM's fp8 weights and KV at the same cap: the fully
    # configured comparison the one-knob arms cannot give (the fp8 pool holds twice the tokens, so preempt fires less
    # often and the single-knob gains do not simply add)
    Arm("batchinfer", {"quant": "fp8", "kv_dtype": "fp8", "memory": "preempt", "watermark_steps": 32, "sharing": "token"},
        "capped combined", CAPPED, cap=CAP, reference="vLLM FP8 KV capped"),
    # the modes whose gain is pool-bound, on the capped rows against vLLM at the same cap (the uncapped pool row is
    # chain-bound: 515 steps for a 511-step chain, none blocked on KV)
    Arm("batchinfer", {"sharing": "token"}, "capped token sharing", CAPPED, cap=CAP),
    Arm("batchinfer", {"block_size": 64}, "capped block 64", CAPPED, cap=CAP),
    Arm("batchinfer", {"attention": "cascade"}, "capped cascade", CAPPED, cap=CAP),
    Arm("batchinfer", {"prefill_budget": "ridge"}, "capped ridge", CAPPED, cap=CAP),
    Arm("batchinfer", {"overlap": False}, "capped no overlap", CAPPED, cap=CAP),
    Arm("batchinfer", {"pool": "contiguous"}, "capped contiguous KV", CAPPED, cap=CAP),
    Arm("batchinfer", {"block_size": 128}, "capped block 128", CAPPED, cap=CAP),
    # the options table: one knob flipped from the defaults each, on mixed, once
    # order and admission also on the many-documents rows: there whole-job order is what keeps a document's KV for
    # its requests (once only, as the options table is)
    Arm("batchinfer", {"order": "input"}, "order input", ("mixed",) + DIVERSE_WORKLOADS, once=True),
    Arm("batchinfer", {"order": "max_tokens_desc"}, "order max_tokens_desc", ("mixed",), once=True),
    Arm("batchinfer", {"order": "decode_ratio_desc"}, "order decode_ratio_desc", ("mixed",), once=True),
    Arm("batchinfer", {"admission": "groups"}, "admission groups", ("mixed",) + DIVERSE_WORKLOADS, once=True),
    Arm("batchinfer", {"chunk_prefill": False}, "no chunking", ("mixed",), once=True),
    Arm("batchinfer", {"prefill_budget": "auto"}, "prefill budget auto", ("mixed",), once=True),
    Arm("batchinfer", {"prefill_budget": "ridge"}, "prefill budget ridge", ("mixed", "generate", "pool")),
)
# The arms the published results ran (--campaign): every mode in the regime it is for. The reference arms on every
# workload; the memory modes, fp8 and the combined arm on the capped rows against vLLM at the same cap; fp8 and the
# attention paths on the long-context rows; the scheduling and sharing modes, the model paths and the cross-job caches
# where the arm table puts them; the options table once. Left out: the cap sweep's half and eighth, the other block
# sizes, contiguous KV (refused with FlashAttention-3), the elastic pool, the capped single-mode rows, the ridge budget
# and the naive baseline.
CAMPAIGN = (
    "vLLM cache on", "vLLM cache off", "vLLM eager", "batchinfer", "no trie",
    "vLLM capped", "capped", "capped grow", "capped preempt", "capped preempt lookahead 32",
    "vLLM FP8 KV capped", "capped FP8 weights and KV", "capped combined",
    "vLLM FP8", "FP8 weights", "vLLM FP8 KV", "FP8 weights and KV",
    "FlashAttention-2", "cascade", "block 128",
    "no overlap", "token sharing", "preempt lookahead 32",
    "HF layers, eager", "no graphs", "compiled layers", "vLLM warm", "radix warm",
    "order input", "order max_tokens_desc", "order decode_ratio_desc", "admission groups", "no chunking",
    "prefill budget auto",
)
# A reused baseline must have run with the packages installed now, on the model snapshot cached now...
VERSION_KEYS = {"vllm": ("vllm", "torch"), "naive": ("torch", "transformers")}
# ...and with its own engine's code unchanged since its commit: a change there reruns it.
OWN_PATHS = {"vllm": ("bench/backends.py",), "naive": ("batchinfer/naive.py", "batchinfer/model.py")}
# Code a baseline shares with every row (the bench harness; for naive, the pipeline before the engine) changes with
# nearly every batchinfer commit. A change there is printed with the reuse rather than forcing a rerun, so a baseline
# is built once and reused; --baselines rerun refreshes it.
SHARED_PATHS = {"vllm": ("bench/run.py",),
                "naive": ("bench/run.py", "batchinfer/analysis.py", "batchinfer/prefix.py", "batchinfer/kv.py",
                          "batchinfer/policy.py", "batchinfer/flow.py", "batchinfer/schema.py", "batchinfer/io.py",
                          "batchinfer/bench.py", "batchinfer/metrics.py")}
# The idle GPU the next load group waits for (wait_for_idle_gpu): under IDLE_MIB used and no compute process,
# IDLE_CHECKS times in a row IDLE_EVERY_S apart; give up after IDLE_TIMEOUT_S.
IDLE_MIB, IDLE_CHECKS, IDLE_EVERY_S, IDLE_TIMEOUT_S = 100, 2, 2.0, 120.0


@dataclass
class Row:
    workload: str  # the workload file
    backend: str  # bench/backends.py registry name or package.module:Class
    opts: dict  # every option the row runs with, in label order; empty for the engine's defaults
    compare: bool = True  # compare against this GPU's REFERENCE row
    column: str = ""  # the arm's column in the results tables, recorded in the row's metrics.json
    warm: bool = False  # timed after an untimed pass over the workload (Arm.warm)


@dataclass
class Group:
    """One engine plus its load options: one worker process, one load, then every row."""
    backend: str
    load_opts: dict
    rows: list = field(default_factory=list)


# plan ---------------------------------------------------------------------------------------------

def workload_path(name, size=SIZE):
    """A preset name (mixed) is workloads/<preset>-<size>.jsonl, a name ending in -natural (generate-natural) that
    preset built with --lengths natural, workloads/<preset>-<size>-natural.jsonl; anything ending in .jsonl is a path."""
    if name.endswith(".jsonl"):
        return Path(name)
    if name.endswith("-natural"):
        return Path("workloads") / f"{name.removesuffix('-natural')}-{size}-natural.jsonl"
    return Path("workloads") / f"{name}-{size}.jsonl"


def contiguous_unit(kv_bytes):
    """pool=contiguous's block for a model with these KV bytes per token (executor.page_tokens's rule): the fewest
    tokens whose KV is a whole number of 2 MiB pieces, raised to a multiple of itself of at least 128."""
    from batchinfer.executor import PAGE_TOKENS_MIN, PIECE_BYTES  # here: bench.report and bench.tables read this module without torch
    unit = PIECE_BYTES // math.gcd(PIECE_BYTES, kv_bytes)
    return unit * -(-PAGE_TOKENS_MIN // unit)


def resolve_cap(arm, workload, kv_bytes):
    """A capped arm's options for one workload: cap x the sidecar's kv_demand_tokens, rounded down to whole blocks,
    plus the one block each engine keeps back (batchinfer's pad block, vLLM's null block), so both pools hold the
    same usable tokens: batchinfer's num_blocks, or vLLM's kv_cache_memory_bytes at this model's KV bytes per token.
    vLLM also needs max_model_len within the cap (it refuses to start unless one request of the model's full
    context fits), set to the workload's longest prompt plus its largest max_tokens; batchinfer's preflight refuses
    a request past the pool per request, so the two are like for like. The cap is at least that longest request
    and two blocks (the preempt watermark), the same floor on both engines: a small workload (smoke, whose quarter
    is below its longest request) still yields a pool both accept; the other capped workloads sit far above it."""
    meta = json.loads(meta_path(workload).read_text())
    fp8_kv = arm.opts.get("kv_dtype") == "fp8" or arm.opts.get("kv_cache_dtype") == "fp8"
    block_size = arm.opts.get("block_size", DEFAULT_BLOCK_SIZE)
    if arm.opts.get("pool") == "contiguous":  # its block is the mapping unit, from the model's KV bytes per token
        if kv_bytes is None:
            raise ValueError(f"{arm.column} on {workload}: the model's KV bytes per token are unknown, so no cap")
        block_size = contiguous_unit(kv_bytes // (2 if fp8_kv else 1))
    by_source = meta["stats"]["by_source"].values()
    longest = max(s["prompt_tokens"]["max"] for s in by_source) + max(s["max_tokens"]["max"] for s in by_source)
    # the cap is the bf16 rows' bytes: an fp8 cache holds twice the tokens in them, on either engine
    blocks = int(meta["stats"]["kv_demand_tokens"] * arm.cap * (2 if fp8_kv else 1)) // block_size
    blocks = max(blocks, -(-longest // block_size) + 2)
    if arm.engine == "vllm":
        if kv_bytes is None:
            raise ValueError(f"{arm.column} on {workload}: the model's KV bytes per token are unknown, so no cap")
        if fp8_kv:
            kv_bytes //= 2
        return {**arm.opts, "kv_cache_memory_bytes": (blocks + 1) * block_size * kv_bytes,
                "max_model_len": -(-longest // block_size) * block_size}
    return {**arm.opts, "num_blocks": blocks + 1}


def split_opts(backend, opts):
    """(load options, policy options). A backend without load_opts (vLLM) loads anew for every option."""
    keys = getattr(resolve(backend), "load_opts", None)
    if keys is None:
        return dict(opts), {}
    return {k: v for k, v in opts.items() if k in keys}, {k: v for k, v in opts.items() if k not in keys}


def known_opts(backend):
    """The options a backend takes: its constructor's and its _policy_config's (where the constructor's **kwargs
    go); None when it takes anything (vLLM passes every option on to vllm.LLM)."""
    cls = resolve(backend)
    policy = getattr(cls, "_policy_config", None)
    names = set(getattr(cls, "load_opts", ()))  # the executor's build options, passed through **opts
    for fn in (cls.__init__, policy):
        for p in inspect.signature(fn).parameters.values() if fn else ():
            if p.kind is p.VAR_KEYWORD and not (fn is cls.__init__ and policy):
                return None
            if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL) and p.name not in ("self", "model"):
                names.add(p.name)
    return names


SMOKE = "smoke"  # the crash check: every arm runs once on it, whatever workloads the arm names


def rows_for(workloads, engines, arms, backends, size=SIZE, kv_bytes=None):
    """Every row the suite runs, workload by workload, from the arms that apply to it (an arm names its workloads,
    or every workload; the smoke workload takes every arm) and whose engine is in engines; capped arms resolved for
    the workload. vLLM's cache-on row, the reference, is the one row that is not compared."""
    rows = []
    for w in workloads:
        name = workload_name(Path(w).stem, size)
        for arm in arms:
            if arm.engine not in engines or (arm.workloads is not None and name not in arm.workloads and name != SMOKE):
                continue
            opts = resolve_cap(arm, w, kv_bytes) if arm.cap is not None else dict(arm.opts)
            compare = not (arm.engine == "vllm" and not opts and not arm.warm)
            rows.append(Row(str(w), backends[arm.engine], opts, compare=compare, column=arm.column, warm=arm.warm))
    return rows


def group_rows(rows):
    """Load groups in ENGINES order (then first appearance), each holding its rows in the order given."""
    groups = {}
    for r in rows:
        load_opts, _ = split_opts(r.backend, r.opts)
        key = (r.backend, tuple(sorted(load_opts.items())))
        groups.setdefault(key, Group(r.backend, load_opts)).rows.append(r)
    rank = {name: i for i, name in enumerate(ENGINES)}
    order = sorted(groups.values(), key=lambda g: (rank.get(engine_name(g.backend), len(ENGINES)),
                                                   # vLLM's cache-on group first: its row is the reference
                                                   bool(g.load_opts) if engine_name(g.backend) == "vllm" else 0))
    return order


def changed_files(sha, paths):
    """The paths that differ between the commit a row ran at and the working tree, uncommitted edits included; None
    when that cannot be told: no sha, a -dirty one (what ran is not in git), or one git does not know."""
    if not sha or sha.endswith("-dirty"):
        return None
    try:
        r = subprocess.run(["git", "diff", "--name-only", sha, "--", *paths], capture_output=True, text=True)
    except OSError:
        return None
    return r.stdout.split() if r.returncode == 0 else None


def reuse(row, model, gpu, root=ROOT, installed=None, changed=changed_files, revision=None):
    """(True, why) when a baseline row on disk is still a valid measurement of this workload file, model snapshot
    and GPU, with the packages installed now and its own engine's code unchanged; (False, why) otherwise. revision:
    the model snapshot cached now (bench.run.model_revision), compared when both it and the row's are known."""
    name = engine_name(row.backend)
    path = run_dir(row.workload, model, gpu, row_label(row.backend, row.opts, row.warm), root) / "metrics.json"
    if not path.exists():
        return False, "no row yet"
    if not path.with_name("outputs.jsonl").exists():
        return False, "no outputs.jsonl"  # every other row's --compare reference
    m = json.loads(path.read_text())
    installed = installed or versions()
    ran = f"{m.get('git_sha')} ({m.get('git_commit_time') or 'no commit time'}), ran {m.get('timestamp')}"
    if m.get("workload_sha256") != row_sha(row.workload):
        return False, "another workload file"
    if m.get("model") != model:
        return False, f"model {m.get('model')}"
    if m.get("length_violations"):
        return False, f"{m['length_violations']} length violations"
    for pkg in VERSION_KEYS.get(name, ()):
        was = (m.get("versions") or {}).get(pkg)
        if was != installed.get(pkg):
            return False, f"{pkg} {was} -> {installed.get(pkg)}"
    if revision and m.get("model_revision") and m["model_revision"] != revision:
        return False, f"model snapshot {m['model_revision'][:8]} -> {revision[:8]}"
    own = changed(m.get("git_sha"), OWN_PATHS.get(name, ()))
    if own is None:
        return False, f"cannot tell what code {m.get('git_sha')} ran"
    if own:
        return False, f"{name}'s own code changed since {m.get('git_sha')}: {', '.join(own)}"
    shared = changed(m.get("git_sha"), SHARED_PATHS.get(name, ()))
    note = f"; shared code changed since: {', '.join(shared)} (--baselines rerun refreshes)" if shared else ""
    return True, f"reusing {ran}{note}"


def row_sha(workload):
    return hashlib.sha256(Path(workload).read_bytes()).hexdigest()


def plan(rows, model, gpu, baselines="missing", root=ROOT, installed=None, changed=changed_files, revision=None):
    """(groups to run, [(row, decision)]): baseline rows still valid are dropped unless baselines == "rerun"."""
    decisions, keep = [], []
    for r in rows:
        if engine_name(r.backend) in VERSION_KEYS:
            ok, why = (False, "--baselines rerun") if baselines == "rerun" else reuse(r, model, gpu, root, installed,
                                                                                    changed, revision)
        else:
            ok, why = False, "batchinfer rows always run"
        decisions.append((r, "reuse" if ok else "run", why))
        if not ok:
            keep.append(r)
    return group_rows(keep), decisions


# run ----------------------------------------------------------------------------------------------

def gpu_state():
    """(MiB used, compute processes) on this GPU, or None without nvidia-smi."""
    used = shell("nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits")
    if used is None:
        return None
    apps = shell("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader") or ""
    return int(used.splitlines()[0]), len([a for a in apps.splitlines() if a.strip()])


def wait_for_idle_gpu(state=gpu_state, sleep=time.sleep, clock=time.monotonic):
    """Returns once the GPU is idle on IDLE_CHECKS checks in a row, IDLE_EVERY_S apart; raises after IDLE_TIMEOUT_S.
    Returns at once without a GPU."""
    t0, idle = clock(), 0
    while True:
        s = state()
        if s is None:
            return
        idle = idle + 1 if s[0] < IDLE_MIB and s[1] == 0 else 0
        if idle >= IDLE_CHECKS:
            return
        if clock() - t0 > IDLE_TIMEOUT_S:
            raise RuntimeError(f"GPU still busy after {IDLE_TIMEOUT_S:.0f} s ({s[0]} MiB used, {s[1]} compute "
                               f"processes); the previous load group's process may not have exited")
        sleep(IDLE_EVERY_S)


def work(spec):
    """A worker: one load group's rows through one load. Returns the process status: 1 if any row failed."""
    model, gpu, backend, load_opts = spec["model"], spec["gpu"], spec["backend"], spec["load_opts"]
    failed, jobs = [], []
    for r in spec["rows"]:
        label = row_label(backend, r["opts"], r["warm"])
        compare_to = None
        if r["compare"]:
            reference = reference_path(REFERENCE, r["workload"], model, gpu)
            if reference.exists() and reference.with_name("metrics.json").exists():  # metrics.json: a whole row
                compare_to = REFERENCE
            else:
                print(f"WARN: {label} on {r['workload']}: no {REFERENCE} row on this GPU yet, so no --compare")
        try:
            jobs.append((r, prepare(r["workload"], model, compare_to, gpu)))
        except Exception:  # a missing workload or a mismatched reference: this row only
            traceback.print_exc()
            failed.append(f"{r['workload']} {label}")
    if jobs:
        engine, load_s = load(backend, model, load_opts)
        print(f"== {group_name(backend, load_opts)} loaded in {load_s:.1f} s; {len(jobs)} passes", flush=True)
        for r, job in jobs:
            label = row_label(backend, r["opts"], r["warm"])
            try:
                policy = {k: v for k, v in r["opts"].items() if k not in load_opts}
                if hasattr(engine, "configure"):
                    engine.configure(**policy)
                elif policy:
                    raise ValueError(f"{backend} takes no per-pass options, got {policy}")
                m = timed_pass(engine, job, backend, r["opts"], load_s, column=r["column"], warm=r["warm"])
                if m["length_violations"]:
                    failed.append(f"{r['workload']} {label} (length violations)")
            except Exception:
                traceback.print_exc()
                failed.append(f"{r['workload']} {label}")
            sys.stdout.flush()
    for f in failed:
        print(f"FAILED: {f}")
    return 1 if failed else 0


def run_worker(spec):
    """One load group in a process of its own, so its GPU memory is released before the next group loads."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
    try:
        return subprocess.run([sys.executable, "-m", "bench.suite", "--worker", f.name]).returncode
    finally:
        Path(f.name).unlink(missing_ok=True)


def run_groups(groups, model, gpu, worker=run_worker, idle=wait_for_idle_gpu):
    """Runs every group in order; returns [(group, status, seconds)]."""
    done = []
    for g in groups:
        idle()
        spec = {"model": model, "gpu": gpu, "backend": g.backend, "load_opts": g.load_opts,
                "rows": [asdict(r) for r in g.rows]}
        t0 = time.perf_counter()
        status = worker(spec)
        done.append((g, status, time.perf_counter() - t0))
    return done


def build_missing(paths, model, size=SIZE):
    """Builds each preset workload (workloads/<preset>-<size>.jsonl, or -natural.jsonl with --lengths natural) that
    is not there yet, with bench.workload, one process each. Other missing paths are left for the caller to report."""
    for p in paths:
        name = workload_name(p.stem, size)
        natural, preset = name.endswith("-natural"), name.removesuffix("-natural")
        if not p.exists() and p == workload_path(name, size) and preset in PRESETS:
            print(f"building {p}", flush=True)
            subprocess.run([sys.executable, "-m", "bench.workload", "build", "--preset", preset, "--size", size,
                            "--model", model, "--out", str(p), *(["--lengths", "natural"] if natural else [])],
                           check=True)


def select_arms(defaults_only=False, variants=(), once=False, arms=ARMS, columns=()):
    """The arms a suite call runs: the arm table without its once-only arms unless --once; --defaults-only keeps
    the non-batchinfer arms and batchinfer's defaults; --variant replaces batchinfer's non-default arms with the
    given options on every workload; --column keeps only the arms with these columns, each on its own workloads."""
    unknown = sorted(set(columns) - {a.column for a in arms})
    if unknown:
        raise ValueError(f"--column {', '.join(unknown)}: no such arm (bench/suite.py ARMS)")
    chosen = [a for a in arms if a.column in columns] if columns else [a for a in arms if once or not a.once]
    if defaults_only or variants:
        chosen = [a for a in chosen if a.engine != "batchinfer" or (not a.opts and a.cap is None)]
        chosen += [Arm("batchinfer", v, ", ".join(f"{k}={val}" for k, val in v.items())) for v in variants]
    return tuple(chosen)


def parse_variant(text):
    """'order=input,prefill_budget=ridge' -> {'order': 'input', 'prefill_budget': 'ridge'}"""
    pairs = [part.split("=", 1) for part in text.split(",") if part.strip()]
    if not pairs or any(len(p) != 2 for p in pairs):
        raise ValueError(f"--variant takes k=v[,k=v], not {text!r}")
    return {k.strip(): parse_value(v.strip()) for k, v in pairs}


def group_name(backend, load_opts):
    """'vllm enable_prefix_caching=False', 'batchinfer' (its default load)."""
    return " ".join([engine_name(backend), *(f"{k}={v}" for k, v in load_opts.items())])


def describe(decisions, groups):
    lines = []
    for r, what, why in decisions:
        lines.append(f"  {what:5} {Path(r.workload).stem:16} {row_label(r.backend, r.opts, r.warm):40} {why}")
    lines.append(f"{len(groups)} load groups:")
    for g in groups:
        lines.append(f"  {group_name(g.backend, g.load_opts)}: {len(g.rows)} passes")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bench.suite", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--workloads", default=",".join(WORKLOADS),
                   help="comma-separated presets (built at --size if missing) or .jsonl paths (default %(default)s); "
                        "'sweep' is the shared-share sweep, " + ",".join(SWEEP_WORKLOADS) + "; 'grid' the share grid "
                        "(or one family, grid-dec-x32); 'qdial' the QuALITY share dial")
    p.add_argument("--size", choices=SIZES, default=SIZE)
    p.add_argument("--engines", default=None,
                   help=f"comma-separated subset of {','.join(ENGINES)} (default {','.join(DEFAULT_ENGINES)}, or every "
                        "engine with --once; naive is opt-in otherwise)")
    p.add_argument("--variant", action="append", metavar="K=V[,K=V]",
                   help="a batchinfer arm to run on every workload, with the option names --opt takes; repeat for "
                        "more. Replaces the batchinfer arms of the arm table; the defaults row always runs")
    p.add_argument("--defaults-only", action="store_true", help="batchinfer's defaults row only, no other arm")
    p.add_argument("--column", action="append", default=[], metavar="COLUMN",
                   help="only the arm with this column (ARMS), on its own workloads; repeat for more (a column may "
                        "hold commas, so one per flag)")
    p.add_argument("--campaign", action="store_true",
                   help="the arms the published results ran (CAMPAIGN), with any --column added")
    p.add_argument("--once", action="store_true",
                   help="include the once-only arms (the options table and the naive baseline, whose engine then runs "
                        "too): the first of three runs")
    p.add_argument("--baselines", choices=("missing", "rerun"), default="missing",
                   help="missing: reuse naive and vLLM rows that are still valid (default); rerun: run them again")
    p.add_argument("--dry-run", action="store_true", help="print the rows, reuse decisions and load groups; run nothing")
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)  # internal: one load group's spec
    args = p.parse_args(argv)
    if args.worker:
        return work(json.loads(args.worker.read_text()))

    engines = [e.strip() for e in (args.engines or ",".join(ENGINES if args.once else DEFAULT_ENGINES)).split(",")
               if e.strip()]
    if not engines or set(engines) - set(ENGINES):
        p.error(f"--engines takes a subset of {','.join(ENGINES)}, not {args.engines!r}")
    if args.defaults_only and args.variant:
        p.error("--defaults-only and --variant exclude each other")
    try:
        arms = select_arms(args.defaults_only, [parse_variant(v) for v in args.variant or ()], args.once,
                           columns=(CAMPAIGN if args.campaign else ()) + tuple(args.column))
    except ValueError as e:
        p.error(str(e))
    backends = {e: e for e in ENGINES}
    allowed = known_opts(backends["batchinfer"])
    for a in arms:
        if a.engine == "batchinfer" and allowed is not None and set(a.opts) - allowed:
            p.error(f"{a.column} {a.opts}: unknown option {sorted(set(a.opts) - allowed)}; batchinfer takes {sorted(allowed)}")
    names = [n for w in args.workloads.split(",") if w.strip() for n in ALIASES.get(w.strip(), [w.strip()])]
    paths = [workload_path(w, args.size) for w in names]

    gpu = gpu_name()
    if not args.dry_run:
        build_missing(paths, args.model, args.size)
    missing = [str(w) for w in paths if not w.exists()]
    if missing:
        p.error(f"no workload file {', '.join(missing)} (a dry run builds nothing)")
    rows = rows_for(paths, engines, arms, backends, args.size, kv_bytes_per_token(args.model))
    groups, decisions = plan(rows, args.model, gpu, args.baselines, revision=model_revision(args.model))
    print(f"suite: {args.model} on {gpu or 'no GPU'}; {sum(len(g.rows) for g in groups)} rows to run")
    print(describe(decisions, groups), flush=True)
    if args.dry_run:
        return 0
    started, t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z"), time.perf_counter()  # started: bench.run's timestamp format
    done = run_groups(groups, args.model, gpu)
    ran = {run_dir(r.workload, args.model, gpu, row_label(r.backend, r.opts, r.warm)) for g in groups for r in g.rows}
    print(summary(done, time.perf_counter() - t0, args.model, gpu, paths, started, ran))
    return 1 if any(status for _, status, _ in done) else 0


def summary(done, total_s, model, gpu, paths, started=None, ran=()):
    """The load groups' times, then the comparison of the rows on disk for this model and GPU. started (a bench.run
    timestamp) and ran (the row dirs this run set out to write): such a row whose metrics.json ran before started is
    an earlier run's, left in place because this run's attempt failed, so it is named and kept out of the table."""
    lines = ["", "load groups:"]
    for g, status, secs in done:
        lines.append(f"  {group_name(g.backend, g.load_opts)}: {len(g.rows)} passes in {secs:.0f} s"
                     + ("" if status == 0 else f" (exit {status})"))
    lines.append(f"suite: {total_s:.0f} s")
    stems = {Path(p).stem for p in paths}
    runs = {p: json.loads(p.read_text()) for p in report.find([ROOT])}  # by path: `ran` names row dirs
    runs = {p: m for p, m in runs.items() if "by_source" in m and m.get("model") == model and m.get("gpu") == gpu
            and Path(m["workload"]).stem in stems}
    stale = {p: m for p, m in runs.items()
             if started and p.parent in ran and (report.utc(m.get("timestamp")) or "") < report.utc(started)}
    lines += [f"FAILED this run: {p.parent}; the row on disk ran {m.get('timestamp')}, so it is left out"
              for p, m in stale.items()]
    runs = [m for p, m in runs.items() if p not in stale]
    if runs:
        lines += ["", report.table(runs)]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
