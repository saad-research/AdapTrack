#!/usr/bin/env python
"""
train_pool.py -- train the adapter pool, resumably.

Every adapter is an independent unit of work. After each one finishes we write
its weights and append to a manifest on Drive, so a Colab disconnect costs at
most the adapter in flight. Re-running the same command skips everything already
present and picks up where it stopped.

  python train_pool.py --n-examples 2500 --per-family 24 \
      --out /content/drive/MyDrive/adaptrack/pool_tinyllama

  # cross-base replicate: SAME data slices, different base
  python train_pool.py --n-examples 2500 --per-family 24 \
      --base Qwen/Qwen2.5-1.5B \
      --out /content/drive/MyDrive/adaptrack/pool_qwen

Two invariants this script enforces, both of which silently invalidate the
paper if violated:

  * DISJOINT SUBSETS. Slices come from one master permutation per family, so
    no example is shared between any two adapters (see src/data/subsets.py).
    The manifest records each slice's checksum so overlap is auditable after
    the fact rather than assumed.

  * CONSTANT n ACROSS FAMILIES. --n-examples applies to every family. If a
    family cannot supply per-family * n disjoint rows the script refuses to
    start rather than silently reusing rows. LEDGAR's 60,000 rows are the
    binding constraint: n=2500 caps you at 24 adapters/family, n=1500 at 40.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.subsets import FAMILIES, load_subset, max_disjoint_adapters  # noqa: E402
from src.train.adapter import load_base, pick_dtype, train_one_adapter     # noqa: E402

POOL_FAMILIES = ["summarization", "medical", "medical_alt", "legal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--out", required=True,
                    help="Drive path. NOT /content -- that is wiped on disconnect.")
    ap.add_argument("--n-examples", type=int, default=2500)
    ap.add_argument("--per-family", type=int, default=24)
    ap.add_argument("--families", nargs="+", default=POOL_FAMILIES)
    ap.add_argument("--render-mode", default="native",
                    choices=["native", "unified"],
                    help="'unified' for the cross-dataset transfer pool")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--reload-base-every", type=int, default=1,
                    help="reload the base model every N adapters. 1 is safest: "
                         "unload() is asserted clean, but a reload costs ~30 s "
                         "against a ~2 min adapter and removes any possibility "
                         "of state leaking between adapters. Set higher only if "
                         "you have measured that it matters.")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N new adapters (for a timed session)")
    args = ap.parse_args()

    out = Path(args.out)
    if str(out).startswith("/content/") and "drive" not in str(out).lower():
        print("WARNING: --out is under /content and does not look like Drive. "
              "A disconnect will delete the whole pool.", file=sys.stderr)
    out.mkdir(parents=True, exist_ok=True)

    # --- refuse to start if disjointness is impossible ---
    for fam in args.families:
        cap = max_disjoint_adapters(args.n_examples, fam)
        if args.per_family > cap:
            sys.exit(
                f"REFUSING TO START.\n"
                f"  {fam}: {FAMILIES[fam]['rows']:,} rows / {args.n_examples} "
                f"per adapter = {cap} disjoint adapters, but --per-family="
                f"{args.per_family} was requested.\n"
                f"  Overlapping subsets between train-pool and test-pool "
                f"adapters leak directly into the headline AUC.\n"
                f"  Either lower --per-family to {cap}, or lower --n-examples "
                f"to {FAMILIES[fam]['rows'] // args.per_family}.\n"
                f"  Do NOT vary n by family: dataset size is itself recoverable "
                f"from LoRA spectra (DSiRe, arXiv:2406.19395), so an unequal n "
                f"makes family partly readable as size.")

    cfg = dict(base=args.base, n_examples=args.n_examples,
               per_family=args.per_family, families=args.families,
               render_mode=args.render_mode, r=args.r, alpha=args.alpha,
               dropout=args.dropout, epochs=args.epochs, max_len=args.max_len,
               batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr)

    cfg_p = out / "pool_config.json"
    if cfg_p.exists():
        old = json.loads(cfg_p.read_text())
        diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
        if diff:
            sys.exit(f"REFUSING TO RESUME: config changed since this pool was "
                     f"started.\n  {diff}\n  Mixing adapters trained under "
                     f"different configs makes the pool inhomogeneous. Use a "
                     f"fresh --out.")
    else:
        cfg_p.write_text(json.dumps(cfg, indent=2))

    man_p = out / "manifest.json"
    manifest = json.loads(man_p.read_text()) if man_p.exists() else {}

    jobs = [(f, k) for f in args.families for k in range(args.per_family)]
    todo = [j for j in jobs
            if f"{j[0]}_a{j[1]:03d}" not in manifest
            or not (out / f"{j[0]}_a{j[1]:03d}" / "adapter_config.json").exists()]
    print(f"pool: {len(jobs)} adapters total, {len(jobs) - len(todo)} done, "
          f"{len(todo)} to go")
    if args.limit:
        todo = todo[:args.limit]

    dtype, use_fp16, use_bf16 = pick_dtype()
    print(f"GPU {torch.cuda.get_device_name(0)}  dtype {dtype}  "
          f"n={args.n_examples}  r={args.r} alpha={args.alpha}")

    base = tok = None
    since_reload = 0
    t_start = time.time()

    for i, (fam, k) in enumerate(todo):
        key = f"{fam}_a{k:03d}"
        if base is None or since_reload >= args.reload_base_every:
            del base
            torch.cuda.empty_cache()
            base, tok = load_base(args.base, dtype)
            since_reload = 0

        texts, idx = load_subset(fam, k, args.n_examples, args.render_mode)
        # checksum of the slice: lets a reviewer (or you) verify disjointness
        # after the fact without re-deriving the permutation.
        checksum = int(np.sort(idx).astype(np.int64).sum())

        stats, base = train_one_adapter(
            base, tok, texts, str(out / key),
            r=args.r, alpha=args.alpha, dropout=args.dropout, lr=args.lr,
            epochs=args.epochs, batch_size=args.batch_size,
            grad_accum=args.grad_accum, max_len=args.max_len,
            seed=hash((fam, k)) % (2**31), use_fp16=use_fp16, use_bf16=use_bf16)
        since_reload += 1

        manifest[key] = dict(family=fam, adapter_idx=k, base=args.base,
                             n_examples=args.n_examples, idx_checksum=checksum,
                             idx_min=int(idx.min()), idx_max=int(idx.max()),
                             **stats)
        man_p.write_text(json.dumps(manifest, indent=2))   # after EVERY adapter

        done = i + 1
        rate = (time.time() - t_start) / done
        print(f"[{done}/{len(todo)}] {key}  {stats['wall_s']:.0f}s  "
              f"loss {stats['train_loss']:.3f}  "
              f"{stats['peak_reserved_gb']:.2f}GB  "
              f"eta {(len(todo) - done) * rate / 60:.0f} min")

    # --- post-hoc disjointness audit ---
    by_fam = {}
    for key, m in manifest.items():
        by_fam.setdefault(m["family"], []).append(m["idx_checksum"])
    print("\ndisjointness audit (identical checksums would mean reused slices):")
    for fam, sums in sorted(by_fam.items()):
        dup = len(sums) - len(set(sums))
        print(f"  {fam:<14} {len(sums):>3} adapters, {dup} duplicate slices "
              f"{'OK' if dup == 0 else '<-- PROBLEM'}")
    print(f"\nmanifest: {man_p}")


if __name__ == "__main__":
    main()