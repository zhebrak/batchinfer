"""Measures a card's ridge for prefill_budget=ridge: the prefill tokens a graphed decode step absorbs at about no cost.
Prints the metrics.CARDS entry to add, which is the only place the number lives (no card is named in engine code).

    python scripts/ridge_calibration.py --model Qwen/Qwen3-8B

For decode rows R in --rows and prefill tokens P in --prefill, one step of R decode rows at --context plus one P-token
prefill row (scripts/decode_step_profile.decode_step), fused layers and graphs, FlashAttention auto (3 on Hopper, where
a mixed step replays a graph; elsewhere it runs eagerly and the knee is where launch overhead stops hiding, which is
still the card's number for the paths it has). Each step's wall is --repeat forwards after a warmup. ridge_tokens is
the largest P at the smallest R whose step takes at most (1 + --slack) times the decode-only step: the prefill a
step carries for free on top of that many decode rows, which the scheduler uses as is (more decode rows read more
KV and buy more free compute, so the number is conservative above the smallest R; the other R are printed for the
record). The knee lands below the datasheet ridge (FLOP/s over bytes/s, about 295 FLOP per byte on an H100 SXM) at
achieved MFU, and the gap is the card's, which is why the datasheet number is never used as the default.
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from batchinfer.executor import Executor  # noqa: E402
from decode_step_profile import decode_step, wall_ms  # noqa: E402


def knee(walls, slack):
    """{prefill tokens: ms} at one row count -> the largest prefill within slack of the decode-only step's wall."""
    base = walls[0]
    return max(p for p, ms in walls.items() if ms <= (1 + slack) * base)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--rows", type=int, nargs="+", default=[64, 256])
    p.add_argument("--prefill", type=int, nargs="+", default=[0, 64, 128, 256, 512, 1024, 2048])
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--slack", type=float, default=0.10, help="the step time over the decode-only step's that still counts as free")
    args = p.parse_args()
    if args.prefill[0] != 0:
        p.error("--prefill must start with 0: the decode-only step is the baseline")
    ex = Executor(args.model, reserve_gb=4.0, seed=3, probe=False, layers="fused", cuda_graphs=True)
    name = torch.cuda.get_device_name()
    print(f"{args.model} on {name}: FlashAttention {ex.fa_version}, graphs {sorted(set(k[1] for k in ex.graph_keys))} "
          f"prefill tokens, {ex.num_blocks} blocks")
    knees = {}
    for rows in args.rows:
        walls = {}
        for prefill in args.prefill:
            step, wrapped = decode_step(ex, rows, args.context, prefill=prefill)
            if wrapped:
                p.error(f"rows {rows} x context {args.context} + {prefill} prefill exceed the pool: lower --context")
            walls[prefill] = wall_ms(ex, step, args.repeat)
            ex.forward(step)
            graphed = f"graph ({ex.graph_rows}, {ex.graph_tokens})" if ex.graph_rows else "eager"
            print(f"rows {rows:4} + prefill {prefill:5}: {walls[prefill]:7.2f} ms/step, {graphed}")
        knees[rows] = knee(walls, args.slack)
        print(f"rows {rows}: ridge {knees[rows]} prefill tokens within {100 * args.slack:.0f}% of the decode-only step")
    ridge = knees[min(knees)]
    print(f"\nthe entry for batchinfer/metrics.py CARDS (ridge at the smallest row count, {min(knees)}):")
    print(f'    "{name}": {{..., "ridge_tokens": {ridge}, "measured_on": "{time.strftime("%Y-%m-%d")}"}},')


if __name__ == "__main__":
    main()
