#!/usr/bin/env python
"""
token_control.py -- is "family" partly "gradient signal volume"?

Equal example counts do not mean equal token counts. At max_len 512 your
summarization adapters see ~6x the tokens of your MedMCQA adapters, and
||Delta||_F visibly grows with data volume. So family is confounded with total
gradient signal, and no amount of pool size fixes it.

Two arms, both cheap, and they answer different halves of the question.

  ARM 1 -- TOKEN LADDER (does token volume alone create separability?)
  One dataset, one example count, three truncation budgets. Nothing varies but
  tokens seen. If a classifier separates the three groups, token volume is a
  detectable channel and you can quote its effect size. If it cannot, the
  confound is not live and you can say so with evidence instead of asserting it.

      python token_control.py --arm ladder --lengths 64 128 512

  ARM 2 -- TOKEN-MATCHED FAMILIES (does family survive equalization?)
  All four families truncated to the SAME short budget, so mean tokens are
  approximately equal across families. If family is still separable here, the
  signal is not token volume. This is the arm a reviewer actually wants.

      python token_control.py --arm matched --match-len 64

  Caveat to state in the paper rather than hide: truncating summarization to 64
  tokens means those adapters are no longer learning summarization. Arm 2 is a
  CONTROL for the token channel, not a replication of the main experiment. It
  answers "can token count alone explain the separation" -- which is the
  reviewer's question -- and nothing more.

Run both, then feed the output directories to probe_diagnostics.py.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.subsets import load_subset  # noqa: E402
from src.train.adapter import load_base, pick_dtype, train_one_adapter  # noqa: E402

FAMILIES = ["summarization", "medical", "medical_alt", "legal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["ladder", "matched"])
    ap.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=2500)
    ap.add_argument("--per-cell", type=int, default=5)
    ap.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 512],
                    help="ladder arm: truncation budgets")
    ap.add_argument("--ladder-family", default="summarization")
    ap.add_argument("--match-len", type=int, default=64,
                    help="matched arm: common truncation budget")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    # Offset so control adapters never reuse rows from the main pool or the
    # floor probe. Overlapping rows would let a control adapter look like a
    # pool adapter for reasons that have nothing to do with the control.
    ap.add_argument("--idx-offset", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man_p = out / "manifest.json"
    manifest = json.loads(man_p.read_text()) if man_p.exists() else {}

    dtype, use_fp16, use_bf16 = pick_dtype()
    print(f"GPU {torch.cuda.get_device_name(0)}  dtype {dtype}  arm={args.arm}")

    if args.arm == "ladder":
        jobs = [(args.ladder_family, L, k)
                for L in args.lengths for k in range(args.per_cell)]
    else:
        jobs = [(f, args.match_len, k)
                for f in FAMILIES for k in range(args.per_cell)]

    base, tok = load_base(args.base, dtype)
    for j, (fam, L, k) in enumerate(jobs):
        key = f"{fam}_len{L}_a{k}"
        if key in manifest and (out / key / "adapter_config.json").exists():
            print(f"[{j+1}/{len(jobs)}] {key} exists, skip")
            continue
        # distinct slice per (length, k) so the ladder is not comparing
        # adapters trained on identical rows at different truncations --
        # that would confound truncation with data reuse in the other direction
        cell = args.lengths.index(L) if args.arm == "ladder" else 0
        idx_i = args.idx_offset + cell * args.per_cell + k
        texts, idx = load_subset(fam, idx_i, args.n_examples)

        del base
        torch.cuda.empty_cache()
        base, tok = load_base(args.base, dtype)

        stats, base = train_one_adapter(
            base, tok, texts, str(out / key), r=args.r, alpha=args.alpha,
            max_len=L, seed=7000 + j, use_fp16=use_fp16, use_bf16=use_bf16)
        # 'family' is what probe_diagnostics groups by. In the ladder arm the
        # thing being separated is the LENGTH, so label it as the family there.
        manifest[key] = dict(
            family=(f"len{L}" if args.arm == "ladder" else fam),
            true_family=fam, max_len_cell=L, n=args.n_examples, k=k,
            idx_checksum=int(np.sort(idx).astype(np.int64).sum()), **stats)
        man_p.write_text(json.dumps(manifest, indent=2))
        print(f"[{j+1}/{len(jobs)}] {key}  {stats['wall_s']:.0f}s  "
              f"tokens {stats['total_tokens']:,}  "
              f"mean {stats['tok_mean']:.0f}  "
              f"trunc {stats['frac_truncated']:.0%}  "
              f"loss {stats['train_loss']:.3f}")
    del base
    torch.cuda.empty_cache()

    # ---- token balance report ----
    print(f"\n{'=' * 70}\nTOKEN BALANCE\n{'=' * 70}")
    by = {}
    for m in manifest.values():
        by.setdefault(m["family"], []).append(m)
    print(f"{'group':<18}{'n':>4}{'mean tok':>10}{'total tok':>13}"
          f"{'trunc':>8}{'loss':>8}")
    means = {}
    for g, ms in sorted(by.items()):
        mt = np.mean([m["tok_mean"] for m in ms])
        tt = np.mean([m["total_tokens"] for m in ms])
        tr = np.mean([m["frac_truncated"] for m in ms])
        ls = np.mean([m["train_loss"] for m in ms])
        means[g] = mt
        print(f"{g:<18}{len(ms):>4}{mt:>10.1f}{tt:>13,.0f}{tr:>8.0%}{ls:>8.3f}")

    if args.arm == "matched" and len(means) > 1:
        spread = max(means.values()) / max(min(means.values()), 1e-9)
        print(f"\nmax/min mean-token ratio across families: {spread:.2f}x")
        if spread > 1.5:
            print("Still imbalanced. Lower --match-len until this is under ~1.3x,")
            print("or the 'matched' arm is not matched and proves nothing.")
        else:
            print("Token volume is now approximately equal across families.")
            print("Run probe_diagnostics.py on this directory: if family is")
            print("still separable, token volume was not the mechanism.")

    print(f"\nnext:\n  python scripts/probe_diagnostics.py "
          f"--probe-dir {out} --n {args.n_examples}")


if __name__ == "__main__":
    main()