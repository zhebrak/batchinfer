"""Build benchmark workloads from sources, and describe them.

    python -m bench.workload build --preset mixed --size quick --model Qwen/Qwen3-14B
    python -m bench.workload build --mix synthetic:320:shared_frac=0.9:groups=4:output_len=64 --model Qwen/Qwen3-14B
    python -m bench.workload stats workloads/mixed-quick.jsonl --show 1

A workload is a JSONL file, one request per line, with a .meta.json sidecar holding the
build arguments, source licenses and stats. Prompts are rendered through the target
model's chat template at build time, so every backend sees identical text, and lines are
shuffled so grouping is never handed to the service.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

from bench.sources import SOURCES, group_by

# Mix strings: source:n[:key=value...]. Counts are for the quick size.
PRESETS = {
    "smoke": "mmlu:4,clinc:4,quality:4:per_article=4,gsm8k:4,wildchat:4",
    "classify": "mmlu:256,clinc:96,quality:24",
    "generate": "gsm8k:64,wildchat:128",
    "mixed": "mmlu:128,clinc:48,quality:16,gsm8k:32,wildchat:64",
    "demo": "mmlu:48,clinc:24,quality:8,gsm8k:16,wildchat:32:max_output=256",
    # Prefix-cache sweep: same requests and lengths, only the shared share of each prompt changes (SWEEP_WORKLOADS)
    "sweep0": "synthetic:320:shared_frac=0:groups=8:output_len=64",
    "sweep10": "synthetic:320:shared_frac=0.1:groups=8:output_len=64",
    "sweep25": "synthetic:320:shared_frac=0.25:groups=8:output_len=64",
    "sweep50": "synthetic:320:shared_frac=0.5:groups=8:output_len=64",
    "sweep75": "synthetic:320:shared_frac=0.75:groups=8:output_len=64",
    "sweep90": "synthetic:320:shared_frac=0.9:groups=8:output_len=64",
    # mixed six times over: the same shared units (MMLU subject shots, CLINC's intent list, QuALITY articles, GSM8K's
    # shots), so the shared share stays near mixed's 81%, at 1,728 requests. Its KV demand (426k tokens) is about
    # the pool of an 80 GB card, so even here admission is nearly all at once. The pool-bound regime, admission in
    # waves, is the suite's capped rows (bench.suite CAP: a quarter of the demand).
    "pool": "mmlu:768,clinc:288,quality:96,gsm8k:192,wildchat:384",
    # Long context: 24 prompts of 8,192 tokens and 256 output tokens, with nothing shared and with half of every prompt
    # shared by 8 rows. At decode the 24 rows' KV (about 203k tokens, 32 GB for Qwen3-14B) outweighs the 29.5 GB of
    # weights a step reads, so this is where the attention paths (FlashAttention-3, cascade, contiguous KV, fp8 KV)
    # differ most; every other workload stays under 2,048-token prompts. The uncapped demand fits both engines' pools
    # on an 80 GB card, so both admit everything at once.
    "long0x8": "synthetic:24:prompt_len=8192:shared_frac=0.0:groups=3:output_len=256",
    "long50x8": "synthetic:24:prompt_len=8192:shared_frac=0.5:groups=3:output_len=256",
    # Many distinct shared prefixes, arriving shuffled (DIVERSE_WORKLOADS): together they hold more KV than an 80 GB
    # card's pool, so a cache that keeps the most recent prefixes evicts a document before its next question comes,
    # while an engine that sees the whole job runs each document's requests together. docqa: 384 QuALITY questions
    # over 96 articles (4 each, ~5-7k tokens of article); rag: 768 generations over 192 synthetic 3.7k-token documents.
    "docqa": "quality:384:per_article=4",
    "rag": "synthetic:768:prompt_len=4096:shared_frac=0.9:groups=192:output_len=64",
}
# The suite's default workloads, in the order the report's comparison lists them: mixed; prefill-bound (classify);
# decode-bound (generate); generate with natural lengths (gsm8k and wildchat stop at EOS under a 512-token cap: the
# early-EOS regime); nothing to share (sweep0); 90% of every prompt shared (sweep90); pool-bound; and the two
# long-context ones. A name ending in -natural is its preset built with --lengths natural.
LONG_WORKLOADS = ("long0x8", "long50x8")
SUITE_WORKLOADS = ("mixed", "classify", "generate", "generate-natural", "sweep0", "sweep90", "pool") + LONG_WORKLOADS
# the shared-share sweep (bench.suite --workloads sweep): where the trie starts paying for itself and where the engine
# passes vLLM, as a function of the share of every prompt that is shared
SWEEP_WORKLOADS = ("sweep0", "sweep10", "sweep25", "sweep50", "sweep75", "sweep90")
# The grid (bench.suite --workloads grid, or one family: grid-dec-x32): the shared share again, at three
# prompt-to-output shapes (prefill-heavy, balanced, decode-heavy) and two fan-outs (rows per shared prefix: the
# cascade's lever), so each mode's gain can be read against the axis it depends on. Requests per shape keep the
# uncapped demand under 370k tokens, chain-bound on an 80 GB card; the cap sweep (bench.suite) binds the pool.
GRID_SHAPES = {"pf": (160, 2048, 16), "bal": (320, 1024, 128), "dec": (320, 256, 512)}  # requests, prompt, output tokens
GRID_SHARES = (0, 10, 30, 50, 90)  # percent of every prompt shared
GRID_FANOUT = (4, 32)  # rows per shared prefix
GRID_FAMILIES = {f"grid-{shape}-x{f}": tuple(f"{shape}{s}x{f}" for s in GRID_SHARES)
                 for shape in GRID_SHAPES for f in GRID_FANOUT}
GRID_WORKLOADS = tuple(n for names in GRID_FAMILIES.values() for n in names)
PRESETS.update({f"{shape}{s}x{f}": f"synthetic:{n}:prompt_len={prompt}:shared_frac={s / 100}:groups={n // f}:output_len={output}"
                for shape, (n, prompt, output) in GRID_SHAPES.items() for f in GRID_FANOUT for s in GRID_SHARES})
# QuALITY as a real-text share dial: the same 96 long-document questions over more or fewer articles (per_article),
# so the shared article is a larger share of each prompt as the questions per article rise
QUALITY_DIAL = ("quality2", "quality8", "quality16")
PRESETS.update({f"quality{k}": f"quality:96:per_article={k}" for k in (2, 8, 16)})
DIVERSE_WORKLOADS = ("docqa", "rag")
ALIASES = {"diverse": DIVERSE_WORKLOADS, "sweep": SWEEP_WORKLOADS, "grid": GRID_WORKLOADS, "qdial": QUALITY_DIAL, **GRID_FAMILIES}


def workload_name(stem, size):
    """A workload file's suite name: mixed-quick -> mixed, generate-quick-natural -> generate-natural."""
    return stem.replace(f"-{size}-natural", "-natural").removesuffix(f"-{size}")


def workload_rank(stem, size="quick"):
    """Where a workload sorts in the suite's order; unknown names after the known ones, by name."""
    name = workload_name(stem, size)
    return (SUITE_WORKLOADS.index(name) if name in SUITE_WORKLOADS else len(SUITE_WORKLOADS)), name
SIZES = {"quick": (1, 60), "full": (3, 180)}  # count multiplier, time budget for the timed run in seconds
HEADROOM = 0.75  # presets keep the cache-off estimate under this share of the budget (45 of 60 s)

# Reference rates (Qwen3-8B bf16 via vLLM on an A100), used only for est_s when sizing presets. Fixed on purpose: a
# workload file must be byte-identical whichever card or model later runs it; a faster card only makes the estimate
# pessimistic.
PREFILL_TPS = 10_000
DECODE_TPS = 2_500
STEP_S = 0.012  # one small-batch decode step, which bounds the longest request


def parse_value(text):
    low = text.lower()
    if low in ("true", "false", "on", "off", "yes", "no"):
        return low in ("true", "on", "yes")
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_mix(spec):
    """'mmlu:128,synthetic:64:shared_frac=0.9' -> [('mmlu', 128, {}), ('synthetic', 64, {'shared_frac': 0.9})]"""
    mix = []
    for part in spec.split(","):
        name, n, *opts = part.strip().split(":")
        if name not in SOURCES:
            raise ValueError(f"unknown source {name!r}; known: {', '.join(SOURCES)}")
        mix.append((name, int(n), {k: parse_value(v) for k, v in (opt.split("=", 1) for opt in opts)}))
    return mix


def render(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def build(mix, tok, seed=0, lengths="fixed"):
    requests, streams, counts = [], {}, {}
    for name, n, kw in mix:
        streams[name] = streams.get(name, -1) + 1  # same source twice in a mix gets its own rng stream
        rows = SOURCES[name](n, random.Random(f"{seed}:{name}:{streams[name]}"), tok, lengths, **kw)
        if len(rows) < n:
            print(f"warning: {name} produced {len(rows)} of {n} requests", file=sys.stderr)
        for row in rows:
            counts[name] = counts.get(name, 0) + 1
            prompt = render(tok, row.pop("messages"))
            requests.append({"id": f"{name}-{counts[name] - 1:05d}", "source": name,
                             "kind": row.pop("kind"), "group": row.pop("group"), "prompt": prompt,
                             "prompt_tokens": len(tok.encode(prompt, add_special_tokens=False)), **row})
    random.Random(seed).shuffle(requests)
    return requests


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def unique_tokens(seqs, page=1):
    """Size of a prefix trie over seqs, sharing only whole pages: each sequence adds what its
    sorted predecessor doesn't cover, and the predecessor has the longest common prefix."""
    unique, prev = 0, ()
    for s in sorted(seqs):
        unique += len(s) - lcp(s, prev) // page * page
        prev = s
    return unique


def pcts(values):
    v = sorted(values)
    return {"p50": v[len(v) // 2], "p95": v[int(0.95 * (len(v) - 1))], "max": v[-1]}


def stats(requests, tok, page=16, kv_bytes=None):
    ids = {r["id"]: tuple(tok.encode(r["prompt"], add_special_tokens=False)) for r in requests}
    by_source = {}
    for name, rows in group_by(requests, "source").items():
        seqs = [ids[r["id"]] for r in rows]
        by_source[name] = {"n": len(rows), "kind": rows[0]["kind"],
                           "prompt_tokens": pcts(map(len, seqs)), "max_tokens": pcts(r["max_tokens"] for r in rows),
                           "ideal_reuse": round(1 - unique_tokens(seqs) / sum(map(len, seqs)), 4)}
    raw, unique = sum(map(len, ids.values())), unique_tokens(ids.values())
    out = sum(r["max_tokens"] for r in requests)
    steps = max(r["max_tokens"] for r in requests)
    decode_s = max(out / DECODE_TPS, steps * STEP_S)
    return {
        "requests": len(requests), "by_source": by_source,
        "raw_prompt_tokens": raw, "unique_prompt_tokens": unique,
        "ideal_prefix_reuse": round(1 - unique / raw, 4),
        f"ideal_prefix_reuse_page{page}": round(1 - unique_tokens(ids.values(), page) / raw, 4),
        "max_output_tokens": out, "decode_steps": steps,
        "kv_demand_tokens": unique + out,
        "kv_demand_gb": round((unique + out) * kv_bytes / 1e9, 1) if kv_bytes else None,
        "est_s": {"cache_on": round(unique / PREFILL_TPS + decode_s, 1),
                  "cache_off": round(raw / PREFILL_TPS + decode_s, 1)},
    }


def print_stats(st, target_s=None):
    print(f"{'source':10} {'kind':9} {'n':>5}  {'prompt tok p50/p95/max':>22}  {'max_tokens p50/max':>18}  ideal reuse")
    for name, s in st["by_source"].items():
        p, m = s["prompt_tokens"], s["max_tokens"]
        print(f"{name:10} {s['kind']:9} {s['n']:5}  {p['p50']:>8}/{p['p95']}/{p['max']:<6}  "
              f"{m['p50']:>11}/{m['max']:<6}  {s['ideal_reuse']:.1%}")
    page_key = next(k for k in st if k.startswith("ideal_prefix_reuse_page"))
    print(f"prompt tokens: raw {st['raw_prompt_tokens']:,}, unique {st['unique_prompt_tokens']:,} "
          f"(ideal reuse {st['ideal_prefix_reuse']:.1%}, {page_key.rsplit('page', 1)[1]}-token pages {st[page_key]:.1%})")
    gb = f" = {st['kv_demand_gb']} GB" if st["kv_demand_gb"] is not None else ""
    print(f"output tokens <= {st['max_output_tokens']:,}, longest request {st['decode_steps']} steps, "
          f"KV demand {st['kv_demand_tokens']:,} tokens{gb}")
    budget = f" (budget {target_s} s)" if target_s else ""
    print(f"est. timed run: cache on {st['est_s']['cache_on']} s, cache off {st['est_s']['cache_off']} s{budget}")
    if target_s and st["est_s"]["cache_off"] > HEADROOM * target_s:
        print(f"WARN: cache-off estimate is over {HEADROOM:.0%} of the {target_s} s budget; shrink the mix")


def write(path, requests, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in requests)
    meta_path(path).write_text(json.dumps(meta, indent=2) + "\n")


def read(path):
    with open(path) as f:
        requests = [json.loads(line) for line in f if line.strip()]
    meta = json.loads(meta_path(path).read_text()) if meta_path(path).exists() else {}
    return requests, meta


def meta_path(path):
    return Path(path).with_suffix(".meta.json")


def load_tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model)


def kv_bytes_per_token(model, dtype_bytes=2):
    """K and V bytes per token across all layers, from the model's config."""
    try:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(model)
    except Exception as e:  # unknown or offline model: stats just omit the GB figure
        print(f"warning: no config for {model}: {e}", file=sys.stderr)
        return None
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return 2 * c.num_hidden_layers * getattr(c, "num_key_value_heads", c.num_attention_heads) * head_dim * dtype_bytes


def stop_token_ids(model, tok):
    """Token ids that end a reply: the tokenizer's eos plus generation_config's (Qwen3 lists two).
    --compare cuts outputs after the first one, since ignore_eos only forces noise past it."""
    ids = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    try:
        from transformers import GenerationConfig
        eos = GenerationConfig.from_pretrained(model).eos_token_id
        ids.update(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    except Exception as e:  # no generation_config.json: the tokenizer's eos alone
        print(f"warning: no generation config for {model}: {e}", file=sys.stderr)
    return sorted(ids)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bench.workload", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="sample sources into a workload file")
    which = b.add_mutually_exclusive_group(required=True)
    which.add_argument("--preset", choices=PRESETS)
    which.add_argument("--mix", help="source:n[:key=value...],... e.g. synthetic:320:shared_frac=0.9:groups=4")
    b.add_argument("--size", choices=SIZES, default="quick")
    b.add_argument("--model", required=True, help="HF model id whose tokenizer and chat template render the prompts")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--lengths", choices=("fixed", "natural"), default="fixed",
                   help="fixed: generate requests decode exactly max_tokens; natural: max_tokens is only a cap")
    b.add_argument("--out", type=Path, help="default workloads/<preset>-<size>[-natural].jsonl")
    s = sub.add_parser("stats", help="describe an existing workload file")
    s.add_argument("workload", type=Path)
    s.add_argument("--model", help="default: the model in the workload's meta")
    s.add_argument("--show", type=int, default=0, metavar="N", help="print N prompts per source")
    args = p.parse_args(argv)

    if args.cmd == "build":
        mult, target_s = SIZES[args.size]
        spec = PRESETS[args.preset] if args.preset else args.mix
        mix = [(name, n * mult, kw) for name, n, kw in parse_mix(spec)]
        tok = load_tokenizer(args.model)
        requests = build(mix, tok, args.seed, args.lengths)
        kv_bytes = kv_bytes_per_token(args.model)
        st = stats(requests, tok, kv_bytes=kv_bytes)
        name = args.preset or "mix-" + hashlib.sha1(spec.encode()).hexdigest()[:8]
        out = args.out or Path("workloads") / f"{name}-{args.size}{'-natural' if args.lengths == 'natural' else ''}.jsonl"
        write(out, requests, {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "preset": args.preset, "size": args.size, "mix": spec,
            "lengths": args.lengths, "seed": args.seed, "model": args.model, "kv_bytes_per_token": kv_bytes,
            "stop_token_ids": stop_token_ids(args.model, tok), "target_s": target_s, "stats": st,
            "sources": {name: {"license": SOURCES[name].license, "url": SOURCES[name].url} for name, _, _ in mix},
        })
        print(f"wrote {len(requests)} requests to {out}")
        print_stats(st, target_s)
    else:
        requests, meta = read(args.workload)
        tok = load_tokenizer(args.model or meta["model"])
        print_stats(stats(requests, tok, kv_bytes=meta.get("kv_bytes_per_token")), meta.get("target_s"))
        for rows in group_by(requests, "source").values():
            for r in rows[:args.show]:
                text = r["prompt"] if len(r["prompt"]) < 1500 else r["prompt"][:1000] + "\n[...]\n" + r["prompt"][-400:]
                print(f"\n=== {r['id']}  max_tokens={r['max_tokens']} reference={r['reference']!r}\n{text}")


if __name__ == "__main__":
    main()
