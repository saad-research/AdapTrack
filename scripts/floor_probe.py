#!/usr/bin/env python
"""
floor_probe.py -- does a task-provenance signal exist at all, and at what n?

Runs BEFORE the pool. Trains a few adapters per family at each of several
dataset sizes and answers three questions:

  Q1  Can family be read off a SINGLE unmerged adapter's spectrum?
      Leave-one-adapter-out accuracy, with a label-permutation null.
      This is the real floor. Merging is an averaging operation over deltas;
      it can only destroy family information relative to the constituents.
      So single-adapter accuracy UPPER-BOUNDS what any merge-level attack can
      reach. If this is at chance, the pool cannot rescue it, and the honest
      move is the negative-result framing rather than 96 more adapters.

  Q2  Is the between-family / within-family variance ratio on eff_rank > 1?
      This is the statistic you proposed. It is reported per module and
      globally -- but see the note on thresholds below.

  Q3  Can dataset SIZE be read off the spectrum? (DSiRe, arXiv:2406.19395)
      If yes -- and DSiRe says yes -- then any family trained on a different
      number of examples is partly identifiable by size rather than task.
      This is why n is held constant across families in the pool. Measuring
      the effect here turns a confound into a reported control.

ON THE <1 THRESHOLD. A between/within variance ratio is an F statistic. Under
the null of no family effect its EXPECTED value is ~1, not 0 -- with 4 families
and 3 adapters each, df=(3,8) and the 95th percentile is ~4.07. So "ratio < 1"
does not mean "no signal", it means "less signal than noise alone would
produce". Treat ratio ~1 as dead, and use the permutation p-value on Q1 as the
actual decision rule. With n=3 the F test has almost no power; --per-family 5
costs ~25 extra minutes and is worth it.

  python floor_probe.py --sizes 500 1500 2500 --per-family 5
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.subsets import FAMILIES, load_subset            # noqa: E402
from src.features.spectral import merge_features              # noqa: E402
from src.train.adapter import load_base, pick_dtype, train_one_adapter  # noqa: E402

FAMILIES_USED = ["summarization", "medical", "medical_alt", "legal"]


def adapter_factor_features(adapter_dir, r, alpha, top_k=8):
    """Spectral features of a single adapter's own delta (K=1 fast path)."""
    from safetensors.torch import load_file
    from src.merge.delta import delta_from_lora
    p = Path(adapter_dir) / "adapter_model.safetensors"
    sd = load_file(str(p)) if p.exists() else torch.load(
        Path(adapter_dir) / "adapter_model.bin", map_location="cpu")
    ad = delta_from_lora(sd, r=r, alpha=alpha)
    return merge_features(deltas=ad.deltas, top_k=top_k)


def loo_family_accuracy(X, y, groups=None, seed=0):
    """Leave-one-adapter-out multinomial LR accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    correct = 0
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0, random_state=seed))
        clf.fit(X[tr], y[tr])
        correct += int(clf.predict(X[i:i + 1])[0] == y[i])
    return correct / len(y)


def permutation_p(X, y, observed, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        if loo_family_accuracy(X, rng.permutation(y)) >= observed:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def variance_ratio(values, labels):
    """The statistic as proposed: var(group means) / mean(within-group var).

    CALIBRATION WARNING. Under the null of no family effect, group means have
    variance sigma^2 / n_per_group, so this statistic centres on 1/n_per_group
    -- roughly 0.2 at 5 adapters per family, not 1. A value near 1 is therefore
    already several times the null, and the "<1 means dead" rule is
    conservative in a way that could make you discard a real signal. Use
    anova_f() below for a calibrated test.
    """
    groups = [values[labels == g] for g in np.unique(labels)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return float("nan")
    within = np.mean([g.var(ddof=1) for g in groups])
    between = np.var([g.mean() for g in groups], ddof=1)
    return between / (within + 1e-30)


def anova_f(values, labels):
    """Standard one-way ANOVA F and its df. Centres on 1.0 under the null."""
    groups = [values[labels == g] for g in np.unique(labels)]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return float("nan"), (0, 0)
    grand = np.concatenate(groups).mean()
    k = len(groups)
    N = sum(len(g) for g in groups)
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    df_b, df_w = k - 1, N - k
    return (ss_b / df_b) / (ss_w / df_w + 1e-30), (df_b, df_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 1500, 2500])
    ap.add_argument("--per-family", type=int, default=5)
    ap.add_argument("--out", default="/content/drive/MyDrive/adaptrack/floor_probe")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--skip-training", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dtype, use_fp16, use_bf16 = pick_dtype()
    print(f"GPU {torch.cuda.get_device_name(0)}  dtype {dtype}")

    # ---------------- train ----------------
    manifest_p = out / "manifest.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}

    if not args.skip_training:
        base, tok = load_base(args.base, dtype)
        jobs = list(itertools.product(args.sizes, FAMILIES_USED,
                                      range(args.per_family)))
        for j, (n, fam, k) in enumerate(jobs):
            key = f"{fam}_n{n}_a{k}"
            d = out / key
            if key in manifest and (d / "adapter_config.json").exists():
                print(f"[{j+1}/{len(jobs)}] {key} exists, skip")
                continue
            # Adapter index is offset per size so that adapters at DIFFERENT
            # sizes never share rows either -- otherwise the size comparison is
            # contaminated by data reuse.
            offset = sum(args.per_family for s in args.sizes if s < n)
            texts, idx = load_subset(fam, offset + k, n)
            stats, base = train_one_adapter(
                base, tok, texts, str(d), r=args.r, alpha=args.alpha,
                max_len=args.max_len, seed=1000 + j,
                use_fp16=use_fp16, use_bf16=use_bf16)
            manifest[key] = dict(family=fam, n=n, k=k, **stats,
                                 idx_lo=int(idx.min()), idx_hi=int(idx.max()))
            manifest_p.write_text(json.dumps(manifest, indent=2))
            print(f"[{j+1}/{len(jobs)}] {key}  {stats['wall_s']:.0f}s  "
                  f"loss {stats['train_loss']:.3f}  "
                  f"{stats['peak_reserved_gb']:.2f} GB")
        del base
        torch.cuda.empty_cache()

    # ---------------- features ----------------
    rows, meta = [], []
    for key, m in sorted(manifest.items()):
        d = out / key
        if not (d / "adapter_config.json").exists():
            continue
        rows.append(adapter_factor_features(d, args.r, args.alpha))
        meta.append(m)
    if not rows:
        sys.exit("no adapters found -- run without --skip-training first")

    cols = sorted(rows[0])
    X = np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)
    fam = np.array([m["family"] for m in meta])
    size = np.array([m["n"] for m in meta])
    print(f"\n{len(X)} adapters, {X.shape[1]} features")

    report = {"n_adapters": len(X), "n_features": X.shape[1], "by_size": {}}

    for n in sorted(set(size.tolist())):
        sel = size == n
        Xs, ys = X[sel], fam[sel]
        print(f"\n{'=' * 66}\nn = {n} examples/adapter   ({sel.sum()} adapters)\n{'=' * 66}")

        # ---- Q1: single-adapter family classification ----
        acc = loo_family_accuracy(Xs, ys)
        chance = 1.0 / len(set(ys.tolist()))
        p = permutation_p(Xs, ys, acc, n_perm=args.n_perm)
        print(f"Q1 single-adapter family LOO accuracy : {acc:.3f} "
              f"(chance {chance:.3f}, permutation p = {p:.4f})")

        # ---- Q2: variance ratio on eff_rank ----
        gcol = cols.index("GLOBAL.eff_rank.mean")
        vr_global = variance_ratio(Xs[:, gcol], ys)
        f_global, (df_b, df_w) = anova_f(Xs[:, gcol], ys)
        n_per = sel.sum() // len(set(ys.tolist()))
        er_cols = [i for i, c in enumerate(cols)
                   if c.endswith(".eff_rank") and not c.startswith("GLOBAL")]
        f_mods = np.array([anova_f(Xs[:, i], ys)[0] for i in er_cols])
        f_mods = f_mods[np.isfinite(f_mods)]

        from scipy.stats import f as f_dist
        crit = f_dist.ppf(0.95, df_b, df_w) if df_w > 0 else float("inf")
        print(f"Q2 eff_rank raw ratio (global)        : {vr_global:.2f} "
              f"(null centres on 1/{n_per} = {1 / n_per:.2f}, not 1.00)")
        print(f"   calibrated ANOVA F({df_b},{df_w})           : {f_global:.2f} "
              f"(F_.95 = {crit:.2f})")
        if f_mods.size:
            print(f"   per-module F: median {np.median(f_mods):.2f}, "
                  f"max {f_mods.max():.2f}, "
                  f"{(f_mods > crit).mean() * 100:.0f}% significant")

        # ---- also: norm growth ----
        fro = Xs[:, cols.index("GLOBAL.fro.mean")]
        print(f"   mean ||D||_F at this n            : {fro.mean():.4f} "
              f"(+/- {fro.std():.4f})")

        report["by_size"][str(n)] = dict(
            loo_acc=acc, chance=chance, perm_p=p,
            vr_eff_rank_global=float(vr_global),
            anova_f_eff_rank_global=float(f_global),
            anova_f_crit=float(crit),
            anova_f_module_median=(float(np.median(f_mods))
                                   if f_mods.size else None),
            fro_mean=float(fro.mean()), fro_std=float(fro.std()))

    # ---- Q3: size recoverability, pooled across families ----
    print(f"\n{'=' * 66}\nQ3 dataset-size recovery (DSiRe confound)\n{'=' * 66}")
    if len(set(size.tolist())) > 1:
        acc_size = loo_family_accuracy(X, size.astype(str))
        chance_size = 1.0 / len(set(size.tolist()))
        print(f"size LOO accuracy across all adapters : {acc_size:.3f} "
              f"(chance {chance_size:.3f})")
        if acc_size > chance_size + 0.2:
            print("   -> size IS recoverable from the spectrum, as DSiRe reports.")
            print("      Hold examples-per-adapter CONSTANT across families in")
            print("      the pool, and say so in the paper -- a reviewer who")
            print("      knows DSiRe will ask.")
        report["size_recovery_acc"] = acc_size
        report["size_recovery_chance"] = chance_size

    # ---- verdict ----
    print(f"\n{'=' * 66}\nVERDICT\n{'=' * 66}")
    best_n, best = None, None
    for n, r in report["by_size"].items():
        if r["perm_p"] < 0.05 and r["loo_acc"] > r["chance"] + 0.25:
            if best is None or r["loo_acc"] > best["loo_acc"]:
                best_n, best = n, r
    if best is None:
        print("NO SIZE CLEARS THE FLOOR.")
        print("Single adapters are not separable by family, so merged ones will")
        print("not be either -- merging averages deltas and can only lose")
        print("family information. Do not spend the pool budget. Switch to the")
        print("negative-result framing (ATTRIB explicitly accepts documentation")
        print("of failed experiments in the idea track) and report the floor")
        print("probe itself as the evidence.")
    else:
        print(f"FLOOR CLEARED at n = {best_n} "
              f"(LOO {best['loo_acc']:.3f}, p = {best['perm_p']:.4f}).")
        smallest = min(int(k) for k, v in report["by_size"].items()
                       if v["perm_p"] < 0.05 and v["loo_acc"] > v["chance"] + 0.25)
        cap = FAMILIES["legal"]["rows"] // smallest
        print(f"Smallest size that clears: n = {smallest}.")
        print(f"LEDGAR has {FAMILIES['legal']['rows']:,} rows, so at n={smallest} "
              f"you get {cap} DISJOINT adapters per family.")
        print(f"That -- not compute -- is your pool ceiling. Use n={smallest} "
              f"uniformly across all four families.")

    (out / "floor_probe_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out / 'floor_probe_report.json'}")


if __name__ == "__main__":
    main()