"""Does the naive engine reproduce HF greedy decoding in bf16? Needs a GPU.

    python scripts/reference_check.py workloads/smoke-quick.jsonl --model Qwen/Qwen3-8B --prefix wildchat,gsm8k

For each selected prompt, decode N tokens three ways at batch size 1: the engine (ignore_eos, so it
runs past EOS), `model.generate` (stops at EOS, so it is compared over its own length) and a full
recompute with no cache. Print the first position where each pair differs, and at the engine's first
divergence recompute that step's logits and print the top-2 margin: a tiny margin is a bf16 near-tie,
a large one is a bug to chase (position 0 = prefill/lm_head, later = decode inputs).
Measured 2026-09-27 on Qwen3-8B / A100: engine == recompute on all 6 smoke prompts, exactly."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere without PYTHONPATH

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from batchinfer import flow  # noqa: E402
from batchinfer.io import read_requests  # noqa: E402
from batchinfer.metrics import Metrics  # noqa: E402
from batchinfer.naive import NaiveEngine  # noqa: E402
from batchinfer.schema import PolicyConfig  # noqa: E402


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


@torch.inference_mode()
def recompute(model, prompt_ids, n):
    ids, gen = list(prompt_ids), []
    for _ in range(n):
        t = int(model(input_ids=torch.tensor([ids], device=model.device)).logits[0, -1].argmax())
        gen.append(t)
        ids.append(t)
    return gen


@torch.inference_mode()
def top3_at(model, prompt_ids, gen, pos):
    ids = list(prompt_ids) + gen[:pos]
    logits = model(input_ids=torch.tensor([ids], device=model.device)).logits[0, -1].float()
    top = torch.topk(logits, 3)
    return [(int(i), round(float(v), 4)) for v, i in zip(top.values, top.indices)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workload")
    p.add_argument("--model", required=True)
    p.add_argument("--prefix", default="", help="comma-separated id prefixes to select, e.g. wildchat,gsm8k")
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("-n", type=int, default=32, help="tokens to decode per prompt")
    args = p.parse_args()
    prefixes = tuple(x for x in args.prefix.split(",") if x)
    reqs = [r for r in read_requests(args.workload) if r.id.startswith(prefixes)][:args.limit]
    for r in reqs:
        r.ignore_eos, r.max_tokens = True, args.n
    tok = AutoTokenizer.from_pretrained(args.model)
    eng = NaiveEngine(args.model, tok)
    model = eng.model
    out = []
    cfg = PolicyConfig(engine="naive", order="max_tokens_desc", admission="groups", prefix_sharing=False,
                       max_batch_tokens=65536, max_batch_size=1)
    flow.run(reqs, tok, eng, cfg, out.append, Metrics())
    by_id = {res.id: res for res in out}  # results come back in policy order, not input order
    for r in reqs:
        res = by_id[r.id]
        prompt = tok(r.prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        with torch.inference_mode():
            g = model.generate(prompt, attention_mask=torch.ones_like(prompt), do_sample=False, max_new_tokens=args.n,
                               pad_token_id=tok.pad_token_id)[0, prompt.shape[1]:].tolist()
        rc = recompute(model, prompt[0].tolist(), args.n)
        e = res.token_ids
        print(f"{r.id}: first diff engine/generate {first_diff(e[:len(g)], g)} (over {len(g)} generate tokens), "
              f"engine/recompute {first_diff(e, rc)}, generate/recompute {first_diff(g, rc[:len(g)])}")
        for name, other in (("generate", g), ("recompute", rc)):
            d = first_diff(e[:len(other)], other)
            if d is not None:
                top3 = top3_at(model, prompt[0].tolist(), e, d)
                print(f"   vs {name} at pos {d}: engine {e[d]}, {name} {other[d]}; fresh top3 {top3}; "
                      f"top-2 margin {round(top3[0][1] - top3[1][1], 4)}")


if __name__ == "__main__":
    main()
