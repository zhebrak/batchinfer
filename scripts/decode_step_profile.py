"""Where a decode-only step's time goes: wall per step against rows, and one profiled step's GPU kernel
time, so "host-bound" or "GPU-bound" is a measurement and not a guess. Runs the executor alone with
synthetic rows (the KV contents do not matter for timing; block tables and lengths do).

    python scripts/decode_step_profile.py --model Qwen/Qwen3-8B --rows 1 16 64 --context 1024
    python scripts/decode_step_profile.py --model Qwen/Qwen3-8B --rows 1 16 64 --fused-layers --cuda-graphs

--fa-version, --fused-layers and --cuda-graphs build the executor the way the bench options of the same names do.

--shared N lays the rows out as prefix sharing does: in groups of --group-size (default: one group), each group's
rows point at the same physical blocks for their first N // 16 blocks. FlashAttention reads those blocks once per row
whatever the layout, so this step alone does not say what a shared-prefix kernel (BatchLLM's fused attention) would
save. With --shared the script also times attention over each row's private KV alone and over each group's prefix
read once: the step's attention less those two bounds the kernel's saving per step, before its own merge cost.
The gate, on sweep90's shape (8 groups of 40 rows; 90% of each 1,024-word prompt, ~930 tokens, is the group's):

    python scripts/decode_step_profile.py --rows 320 --context 1100 --shared 928 --group-size 40

Build the kernel only if that bound, times the row's decode-only steps (63 on sweep90-quick), is above run-to-run
noise (0.4 s between same-schedule runs on the H100): more than ~6 ms per step. All three steps fit either card's pool (A100-40GB, H100-80GB).
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from batchinfer.executor import Executor  # noqa: E402
from batchinfer.kv import BLOCK_SIZE  # noqa: E402
from batchinfer.schema import Row, Step  # noqa: E402


def decode_step(ex, rows, context, shared=0, group_size=0):
    """`rows` decode rows, each at position `context`. Rows come in groups of group_size (0: one group) whose first
    shared // BLOCK_SIZE blocks are the same physical blocks; every other block is the row's own, the one holding its
    position always (the last prompt token is never shared). Returns the step and whether block ids wrapped around the
    pool, in which case rows alias each other's blocks and memory traffic is understated."""
    need = -(-(context + 1) // BLOCK_SIZE)
    common = min(shared // BLOCK_SIZE, need - 1)
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
    return Step(out), groups * common + rows * (need - common) > usable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--rows", type=int, nargs="+", default=[1, 16, 64])
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--shared", type=int, default=0, help="tokens of each row's context in blocks shared by its group")
    p.add_argument("--group-size", type=int, default=0, help="rows per shared prefix; 0 puts every row in one group")
    p.add_argument("--fa-version", type=int, default=2, choices=(2, 3), help="FlashAttention version (3: sm90 only)")
    p.add_argument("--fused-layers", action="store_true", help="the executor's fused Qwen3 layers instead of HF's")
    p.add_argument("--cuda-graphs", action="store_true", help="replay decode-only steps from captured CUDA graphs")
    args = p.parse_args()
    if not 0 <= args.shared <= args.context:
        p.error(f"--shared must be within [0, --context={args.context}]")
    modes = {k: v for k, v in (("fused_layers", args.fused_layers), ("cuda_graphs", args.cuda_graphs)) if v}
    ex = Executor(args.model, reserve_gb=4.0, seed=3, probe=False, fa_version=args.fa_version, **modes)
    print(f"{args.model} on {torch.cuda.get_device_name()}, {ex.num_blocks} blocks, backend {ex.backend}, "
          f"FlashAttention {args.fa_version}, {', '.join(modes) or 'HF layers, eager'}")
    steps = {}
    for rows in args.rows:
        step, wrapped = decode_step(ex, rows, args.context, args.shared, args.group_size)
        if wrapped:
            print(f"WARN: rows {rows} x context {args.context} need more blocks than the pool's {ex.num_blocks - 1}; "
                  f"block ids wrap, so rows alias each other's blocks and memory traffic is understated")
        steps[rows] = step
    # every wall first: a torch.profiler session left behind slows later forwards, which inflated earlier walls
    walls = {rows: wall_ms(ex, step, args.repeat) for rows, step in steps.items()}
    for rows, step in steps.items():
        r = profile(ex, step)
        layout = f" shared {args.shared} in groups of {args.group_size or rows}" if args.shared else ""
        print(f"rows {rows:3} context {args.context}{layout}: wall {walls[rows]:6.1f} ms/step, GPU kernel time "
              f"{r['gpu_ms']:6.1f} ms ({100 * r['gpu_ms'] / walls[rows]:4.0f}% of wall) over {r['launches']} kernel "
              f"launches; the rest is launch gaps. Attention kernels {r['attn_ms']:6.2f} ms over {r['attn_launches']} "
              f"launches")
        for e in r["top"]:
            print(f"    {e.self_device_time_total / 1000:6.2f} ms  x{e.count:4}  {e.key[:70]}")
        if args.shared:
            groups = -(-rows // (args.group_size or rows))
            prefix = BLOCK_SIZE * min(args.shared // BLOCK_SIZE, args.context // BLOCK_SIZE)  # what decode_step shared
            # a decode row at position p reads p + 1 positions: each row's private KV, then each group's prefix once
            private = profile(ex, decode_step(ex, rows, args.context - prefix)[0])["attn_ms"]
            once = profile(ex, decode_step(ex, groups, prefix - 1)[0])["attn_ms"] if prefix else 0.0
            bound = r["attn_ms"] - private - once
            print(f"    a shared-prefix kernel saves at most {bound:.2f} ms of attention per step: "
                  f"{r['attn_ms']:.2f} now, less {private:.2f} for each row's private KV and {once:.2f} for reading "
                  f"each group's {prefix}-token prefix once, before the kernel's own merge cost")


def wall_ms(ex, step, repeat):
    """Wall ms per step over `repeat` forwards after a warmup; each forward ends with the ids' host sync, as in the
    engine."""
    for _ in range(3):
        ex.forward(step)
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
