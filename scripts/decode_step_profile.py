"""Where a decode-only step's time goes: wall per step against rows, and one profiled step's GPU kernel
time, so "host-bound" or "GPU-bound" is a measurement and not a guess. Runs the executor alone with
synthetic rows (the KV contents do not matter for timing; block tables and lengths do).

    python scripts/decode_step_profile.py --model Qwen/Qwen3-8B --rows 1 16 64 --context 1024
    python scripts/decode_step_profile.py --model Qwen/Qwen3-8B --rows 1 16 64 --layers fused --cuda-graphs

--fa-version, --layers and --cuda-graphs build the executor the way the bench options of the same names do.

--shared N lays the rows out as prefix sharing does: in groups of --group-size (default: one group), each group's
rows point at the same physical blocks for their first N // 16 blocks. FlashAttention reads those blocks once per row
whatever the layout, so this step alone does not say what a shared-prefix kernel (BatchLLM's fused attention) would
save. With --shared the script also times attention over each row's private KV alone and over each group's prefix
read once: the step's attention less those two bounds the kernel's saving per step, before its own merge cost.
On sweep90's shape (8 groups of 40 rows; 90% of each 1,024-word prompt, ~930 tokens, is the group's):

    python scripts/decode_step_profile.py --rows 320 --context 1100 --shared 928 --group-size 40
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from batchinfer.executor import Executor, page_tokens  # noqa: E402
from batchinfer.schema import Row, Step  # noqa: E402


def decode_step(ex, rows, context, shared=0, group_size=0, prefill=0):
    """`rows` decode rows, each at position `context`. Rows come in groups of group_size (0: one group) whose first
    shared // ex.block_size blocks are the same physical blocks; every other block is the row's own, the one holding its
    position always (the last prompt token is never shared). prefill: tokens of one prefill row on top, from position
    0 in blocks of its own (a mixed step). Returns the step and whether block ids wrapped around the pool, in which
    case rows alias each other's blocks and memory traffic is understated."""
    bs = ex.block_size
    need = -(-(context + 1) // bs)
    common = min(shared // bs, need - 1)
    size = group_size or rows
    groups = -(-rows // size)
    usable = ex.num_blocks - 1
    ids = (1 + k % usable for k in itertools.count())  # block 0 is the pad block
    prefixes = [[next(ids) for _ in range(common)] for _ in range(groups)]
    out = []
    for i in range(rows):
        own = [next(ids) for _ in range(need - common)]
        out.append(Row(seq=i, start=context, token_ids=[7], block_ids=prefixes[i // size] + own, sample=True,
                       decode=True))
    chunk = -(-prefill // bs)
    if prefill:
        out.append(Row(seq=rows, start=0, token_ids=[7] * prefill, block_ids=[next(ids) for _ in range(chunk)],
                       sample=True, decode=False))
    step = Step(out)
    if ex.pool.mode == "contiguous":  # what the scheduler would hand it: a region per row, every one new
        for i, r in enumerate(out):
            r.slot = 1 + i
        step.slot_resets = [1 + i for i in range(len(out))]
    if ex.attention == "cascade":  # what the scheduler would hand it: each group's rows behind its prefix, and every
        from batchinfer.cascade import Group  # row without one a group of its own (the graphs' layout needs them all)
        if common:
            step.groups = [Group(g, prefixes[g], common * bs, [i for i in range(rows) if i // size == g]) for g in range(groups)]
        else:
            step.groups = [Group(None, [], 0, [i]) for i in range(rows)]
        step.groups += [Group(None, [], 0, [rows])] if prefill else []
    return step, groups * common + rows * (need - common) + chunk > usable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--rows", type=int, nargs="+", default=[1, 16, 64])
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--shared", type=int, default=0, help="tokens of each row's context in blocks shared by its group")
    p.add_argument("--group-size", type=int, default=0, help="rows per shared prefix; 0 puts every row in one group")
    p.add_argument("--prefill", type=int, default=0, help="tokens of one prefill row on top of the decode rows (a mixed step)")
    p.add_argument("--attention", default=None, choices=("fa", "cascade"), help="the executor's attention (default fa)")
    p.add_argument("--pool", default="static", choices=("static", "contiguous"),
                   help="the executor's pool; contiguous also times the two arms that isolate the paged overhead (below)")
    p.add_argument("--fa-version", default=None, choices=("auto", "2", "3"),
                   help="FlashAttention version (3: sm90 only); default: the executor's")
    p.add_argument("--layers", default="hf", choices=("auto", "fused", "compiled", "hf"), help="the executor's forward (default hf)")
    p.add_argument("--cuda-graphs", action="store_true", help="replay decode-only steps from captured CUDA graphs")
    p.add_argument("--block-size", type=int, default=None, help="tokens per KV block of the paged pool (default 16)")
    args = p.parse_args()
    if args.pool == "contiguous" and args.cuda_graphs:  # the regions switch is a Python branch, baked into a capture
        p.error("--pool contiguous measures the paged overhead eagerly, through the attention's regions switch: drop --cuda-graphs")
    if not 0 <= args.shared <= args.context:
        p.error(f"--shared must be within [0, --context={args.context}]")
    modes = {k: v for k, v in (("layers", args.layers if args.layers != "hf" else None), ("cuda_graphs", args.cuda_graphs)) if v}
    fa = {"fa_version": args.fa_version if args.fa_version == "auto" else int(args.fa_version)} if args.fa_version else {}
    size = {"reserve_gb": 4.0}
    if args.pool == "contiguous":  # a pool the steps just fit: the paged twin of the three-arm comparison sits beside it
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model)
        unit = page_tokens(cfg.num_hidden_layers, cfg.num_key_value_heads,
                           getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads, 2)
        size = {"num_blocks": max(args.rows) * -(-(args.context + 1) // unit) + -(-args.prefill // unit) + 2}
    ex = Executor(args.model, seed=3, probe=False, layers=args.layers, pool=args.pool, cuda_graphs=args.cuda_graphs,
                  **size, **fa, **({"attention": args.attention} if args.attention else {}),
                  **({"block_size": args.block_size} if args.block_size else {}))
    print(f"{args.model} on {torch.cuda.get_device_name()}, {ex.num_blocks} blocks of {ex.block_size} tokens, backend {ex.backend}, "
          f"FlashAttention {ex.fa_version}, {', '.join(modes) or 'HF layers, eager'}")
    steps = {}
    for rows in args.rows:
        step, wrapped = decode_step(ex, rows, args.context, args.shared, args.group_size, args.prefill)
        if wrapped and args.pool == "contiguous":
            p.error(f"rows {rows} x context {args.context} need more blocks than the pool's {ex.num_blocks - 1}: the "
                    f"three-arm comparison needs every row its own units; lower --rows or --context")
        if wrapped:
            print(f"WARN: rows {rows} x context {args.context} need more blocks than the pool's {ex.num_blocks - 1}; "
                  f"block ids wrap, so rows alias each other's blocks and memory traffic is understated")
        steps[rows] = step
    # every wall first: a torch.profiler session left behind slows later forwards
    walls = {rows: wall_ms(ex, step, args.repeat) for rows, step in steps.items()}
    for rows, step in steps.items():
        r = profile(ex, step)
        layout = (f" shared {args.shared} in groups of {args.group_size or rows}" if args.shared else "") + \
            (f" + {args.prefill} prefill tokens" if args.prefill else "")
        print(f"rows {rows:3} context {args.context}{layout}: wall {walls[rows]:6.1f} ms/step, GPU kernel time "
              f"{r['gpu_ms']:6.1f} ms ({100 * r['gpu_ms'] / walls[rows]:4.0f}% of wall) over {r['launches']} kernel "
              f"launches; the rest is launch gaps. Attention kernels {r['attn_ms']:6.2f} ms over {r['attn_launches']} "
              f"launches")
        for e in r["top"]:
            print(f"    {e.self_device_time_total / 1000:6.2f} ms  x{e.count:4}  {e.key[:70]}")
        if args.pool == "contiguous":
            print("    " + paged_overhead(ex, step, args.repeat))
        if args.shared:
            groups = -(-rows // (args.group_size or rows))
            prefix = ex.block_size * min(args.shared // ex.block_size, args.context // ex.block_size)  # what decode_step shared
            # a decode row at position p reads p + 1 positions: each row's private KV, then each group's prefix once
            private = profile(ex, decode_step(ex, rows, args.context - prefix)[0])["attn_ms"]
            once = profile(ex, decode_step(ex, groups, prefix - 1)[0])["attn_ms"] if prefix else 0.0
            bound = r["attn_ms"] - private - once
            print(f"    a shared-prefix kernel saves at most {bound:.2f} ms of attention per step: "
                  f"{r['attn_ms']:.2f} now, less {private:.2f} for each row's private KV and {once:.2f} for reading "
                  f"each group's {prefix}-token prefix once, before the kernel's own merge cost")


def paged_overhead(ex, step, repeat):
    """The three arms that isolate the paged attention's overhead, attention kernel time per step over the same
    rows and the same tokens of KV: (1) the paged call over a paged pool at the same block size (the unit), (2) the
    contiguous read through the regions (no block table), (3) the paged call over the contiguous home's token-major
    units through a block table. Between 2 and 3 the only difference is the block-table indirection; between 1 and
    3, the layout. The attention's regions switch picks 3; a second, paged executor over the same weights
    runs 1, with as many blocks as the step needs."""
    attn = lambda e, s: profile(e, s)["attn_ms"]  # noqa: E731
    region = attn(ex, step)
    ex.attn.regions = False
    table = attn(ex, step)
    ex.attn.regions = True
    used = max(b for r in step.rows for b in r.block_ids) + 2  # the step's blocks and the pad block: the first
    paged = Executor(ex.model_name, model=ex.model, num_blocks=used, seed=3, probe=False, layers=ex.layers_mode,  # pool stays
                     block_size=ex.page_tokens, cuda_graphs=False, fa_version=ex.fa_version, attention=ex.attention)
    plain = Step([Row(seq=r.seq, start=r.start, token_ids=r.token_ids, block_ids=r.block_ids, sample=r.sample, decode=r.decode)
                  for r in step.rows])
    paged_ms = attn(paged, plain)
    del paged
    torch.cuda.empty_cache()
    return (f"paged overhead: attention {paged_ms:.2f} ms through a block table over a paged pool, {table:.2f} ms through "
            f"a block table over the token-major units, {region:.2f} ms contiguous through the regions "
            f"(indirection: {table - region:+.2f} ms; layout: {paged_ms - table:+.2f} ms)")


def wall_ms(ex, step, repeat):
    """Wall ms per step over `repeat` forwards after a warmup; each forward ends with the ids' host sync, as in the
    engine."""
    for _ in range(3):
        ex.forward(step)
    step.slot_resets = []  # pool=contiguous: the regions are mapped now; a timed forward remaps nothing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        ex.forward(step)
    return 1000 * (time.perf_counter() - t0) / repeat


def profile(ex, step):
    """One profiled forward's kernels (after a warmup): their time and launches, FlashAttention's share of them, and
    the five longest."""
    ex.forward(step)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as prof:
        ex.forward(step)
    # kernel events only: the CPU-side ops (aten::mm, varlen_fwd) carry their kernels' device time too
    kernels = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
    attn = [e for e in kernels if "flash" in e.key.lower()]  # FlashAttention's forward and split-KV combine
    return {"gpu_ms": sum(e.self_device_time_total for e in kernels) / 1000,
            "launches": sum(e.count for e in kernels), "attn_ms": sum(e.self_device_time_total for e in attn) / 1000,
            "attn_launches": sum(e.count for e in attn),
            "top": sorted(kernels, key=lambda e: -e.self_device_time_total)[:5]}


if __name__ == "__main__":
    main()
