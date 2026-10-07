#!/usr/bin/env python
"""
fit_classifier.py -- provenance classifier and the numbers that go in Table 1.

  python fit_classifier.py --features /content/drive/MyDrive/adaptrack/features \
      --target medical

Reports AUC-ROC, TPR@1%FPR, TPR@5%FPR, a label-permutation null, feature-group
ablations, and per-method / per-arity / per-lambda breakdowns.

ON UNCERTAINTY. A naive bootstrap over merges will give you cheerfully narrow
CIs that are wrong. Merges sharing a constituent adapter are not independent
draws -- with ~16 test adapters generating 150 merges, the effective sample
size is closer to 16 than 150. Two things are reported instead:

  * a CLUSTER bootstrap that resamples ADAPTERS and rebuilds the merge set from
    the resampled adapters, which propagates adapter-level variation properly;
  * a leave-one-adapter-out jackknife, which shows whether a single unusual
    adapter is carrying the result.

The naive merge-level bootstrap is also printed, purely so you can see the gap.
If the cluster interval is much wider -- it will be -- report the cluster one.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def tpr_at_fpr(y, s, target_fpr):
    """TPR at a target FPR, linearly interpolated on the ROC."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    if target_fpr <= fpr[0]:
        return float(tpr[0])
    return float(np.interp(target_fpr, fpr, tpr))


def evaluate(y, s):
    from sklearn.metrics import roc_auc_score
    return dict(auc=float(roc_auc_score(y, s)),
                tpr_at_1=tpr_at_fpr(y, s, 0.01),
                tpr_at_5=tpr_at_fpr(y, s, 0.05))


def fit_predict(Xtr, ytr, Xte, kind="lr", seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "lr":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=10000, C=1.0, random_state=seed))
    elif kind == "mlp":
        clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                          early_stopping=True, random_state=seed))
    else:
        raise ValueError(kind)
    # Scaler is fitted on TRAIN only -- fitting on the union leaks test
    # distribution into the features.
    clf.fit(Xtr, ytr)
    return clf, clf.predict_proba(Xte)[:, 1]


def cluster_bootstrap(y, s, constituents, n_boot=1000, seed=0):
    """Resample ADAPTERS with replacement; keep merges built only from them."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    adapters = sorted({a for c in constituents for a in c})
    aucs = []
    for _ in range(n_boot):
        keep = set(rng.choice(adapters, len(adapters), replace=True).tolist())
        sel = np.array([all(a in keep for a in c) for c in constituents])
        if sel.sum() < 20 or len(set(y[sel].tolist())) < 2:
            continue
        aucs.append(roc_auc_score(y[sel], s[sel]))
    if not aucs:
        return (float("nan"), float("nan"), 0)
    return (float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)), len(aucs))


def naive_bootstrap(y, s, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i].tolist())) < 2:
            continue
        aucs.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def jackknife_adapters(y, s, constituents):
    from sklearn.metrics import roc_auc_score
    adapters = sorted({a for c in constituents for a in c})
    out = {}
    for a in adapters:
        sel = np.array([a not in c for c in constituents])
        if sel.sum() < 20 or len(set(y[sel].tolist())) < 2:
            continue
        out[a] = float(roc_auc_score(y[sel], s[sel]))
    return out


def permutation_auc(Xtr, ytr, Xte, yte, n_perm=200, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        _, s = fit_predict(Xtr, rng.permutation(ytr), Xte)
        null.append(roc_auc_score(yte, s))
    return np.array(null)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--target", default="medical")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.features)
    tr = np.load(d / f"features_train_{args.target}.npz", allow_pickle=True)
    te = np.load(d / f"features_test_{args.target}.npz", allow_pickle=True)
    cols = [str(c) for c in tr["columns"]]

    Xtr, ytr = tr["X"], tr["label"]
    Xte, yte = te["X"], te["label"]
    rec_te = json.loads((d / f"recipes_test_{args.target}.json").read_text())
    constituents = [r["constituents"] for r in rec_te]

    # guard: a feature that is constant in train is useless and can blow up the
    # scaler; a feature that is NaN anywhere will silently poison the fit.
    ok = np.isfinite(Xtr).all(0) & np.isfinite(Xte).all(0) & (Xtr.std(0) > 0)
    if (~ok).sum():
        print(f"dropping {(~ok).sum()} constant/non-finite features")
    Xtr, Xte = Xtr[:, ok], Xte[:, ok]
    cols = [c for c, k in zip(cols, ok) if k]

    print(f"train {Xtr.shape}  test {Xte.shape}  |  "
          f"{ytr.mean():.1%} / {yte.mean():.1%} positive")
    print(f"test adapters: {len({a for c in constituents for a in c})}")

    report = {"target": args.target, "n_train": len(ytr), "n_test": len(yte)}

    # ---------------- primary ----------------
    print(f"\n{'=' * 70}\nPRIMARY: logistic regression, full feature set\n{'=' * 70}")
    _, s_lr = fit_predict(Xtr, ytr, Xte, "lr")
    m = evaluate(yte, s_lr)
    lo_n, hi_n = naive_bootstrap(yte, s_lr, args.n_boot)
    lo_c, hi_c, n_ok = cluster_bootstrap(yte, s_lr, constituents, args.n_boot)
    print(f"AUC-ROC      {m['auc']:.4f}")
    print(f"  cluster bootstrap 95% CI  [{lo_c:.4f}, {hi_c:.4f}]  "
          f"<- report this ({n_ok} valid replicates)")
    print(f"  naive  bootstrap 95% CI   [{lo_n:.4f}, {hi_n:.4f}]  "
          f"(too narrow: merges are not independent)")
    print(f"TPR@1%FPR    {m['tpr_at_1']:.4f}")
    print(f"TPR@5%FPR    {m['tpr_at_5']:.4f}")
    report["lr"] = {**m, "ci_cluster": [lo_c, hi_c], "ci_naive": [lo_n, hi_n]}

    jk = jackknife_adapters(yte, s_lr, constituents)
    if jk:
        v = np.array(list(jk.values()))
        worst = min(jk, key=jk.get)
        print(f"\nleave-one-adapter-out AUC: min {v.min():.4f} "
              f"(drop {worst}), max {v.max():.4f}, spread {np.ptp(v):.4f}")
        if np.ptp(v) > 0.05:
            print("  WARNING: >0.05 spread. One adapter is moving the result; "
                  "check it is not degenerate before reporting the headline.")
        report["jackknife"] = {"min": float(v.min()), "max": float(v.max())}

    # ---------------- null ----------------
    print(f"\n{'=' * 70}\nNULL: label permutation ({args.n_perm} draws)\n{'=' * 70}")
    null = permutation_auc(Xtr, ytr, Xte, yte, args.n_perm)
    p = (np.sum(null >= m["auc"]) + 1) / (len(null) + 1)
    print(f"permuted AUC: mean {null.mean():.4f}, 95th pct "
          f"{np.percentile(null, 95):.4f}")
    print(f"observed {m['auc']:.4f}  ->  p = {p:.4f}")
    report["perm_p"] = float(p)
    report["perm_null_mean"] = float(null.mean())

    report["perm_null_p95"] = float(np.percentile(null, 95))
    # ---------------- secondary ----------------
    print(f"\n{'=' * 70}\nSECONDARY: 2-layer MLP\n{'=' * 70}")
    try:
        _, s_mlp = fit_predict(Xtr, ytr, Xte, "mlp")
        mm = evaluate(yte, s_mlp)
        print(f"AUC {mm['auc']:.4f}  TPR@1% {mm['tpr_at_1']:.4f}  "
              f"TPR@5% {mm['tpr_at_5']:.4f}")
        report["mlp"] = mm
    except Exception as e:
        print(f"MLP failed: {e}")

    # ---------------- feature ablation (Table 3) ----------------
    print(f"\n{'=' * 70}\nFEATURE ABLATION\n{'=' * 70}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.features.spectral import feature_groups
    groups = feature_groups(cols)
    report["ablation"] = {}
    for name, keep in groups.items():
        idx = [i for i, c in enumerate(cols) if c in set(keep)]
        if len(idx) < 2:
            continue
        _, s = fit_predict(Xtr[:, idx], ytr, Xte[:, idx], "lr")
        e = evaluate(yte, s)
        print(f"  {name:<18} {len(idx):>5} feats   AUC {e['auc']:.4f}   "
              f"TPR@1% {e['tpr_at_1']:.4f}")
        report["ablation"][name] = {**e, "n_features": len(idx)}

    # ---------------- breakdowns ----------------
    print(f"\n{'=' * 70}\nBREAKDOWNS (same classifier, sliced test set)\n{'=' * 70}")
    from sklearn.metrics import roc_auc_score
    for key in ("method", "arity"):
        vals = te[key]
        print(f"\nby {key}:")
        report[f"by_{key}"] = {}
        for v in sorted(set(vals.tolist())):
            sel = vals == v
            if sel.sum() < 15 or len(set(yte[sel].tolist())) < 2:
                print(f"  {v:<18} n={sel.sum():<4} (too few to score)")
                continue
            a = roc_auc_score(yte[sel], s_lr[sel])
            print(f"  {v!s:<18} n={sel.sum():<4} AUC {a:.4f}")
            report[f"by_{key}"][str(v)] = float(a)

    lam = te["lam_target"]
    pos = yte == 1
    if pos.sum() > 30:
        print("\nby lambda on the target family (positives only, "
              "scored against all negatives):")
        report["by_lambda"] = {}
        edges = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (pos & (lam >= lo) & (lam < hi)) | (~pos)
            npos = (pos & (lam >= lo) & (lam < hi)).sum()
            if npos < 10:
                continue
            a = roc_auc_score(yte[sel], s_lr[sel])
            print(f"  lambda in [{lo:.1f},{hi:.1f})  n_pos={npos:<4} AUC {a:.4f}")
            report["by_lambda"][f"{lo:.1f}-{hi:.1f}"] = float(a)

    # ---------------- gate ----------------
    print(f"\n{'=' * 70}\nGATE\n{'=' * 70}")
    auc = m["auc"]
    if auc >= 0.80:
        print(f"AUC {auc:.3f} >= 0.80 -- strong. Headline it.")
    elif auc >= 0.75:
        print(f"AUC {auc:.3f} in [0.75, 0.80) -- the finding exists. Proceed.")
    elif lo_c > 0.5 and p < 0.05:
        print(f"AUC {auc:.3f} is modest but the cluster CI excludes 0.5 "
              f"(p={p:.4f}). A real but weak effect: report it with the CI "
              f"prominent and do not overclaim.")
    else:
        print(f"AUC {auc:.3f} with cluster CI [{lo_c:.3f}, {hi_c:.3f}], "
              f"p={p:.4f}.")
        print("Not distinguishable from chance. This is the negative result: "
              "merging provides plausible deniability for task provenance "
              "under weight-only spectral features. ATTRIB's idea track "
              "explicitly welcomes documented failed experiments.")

    out = Path(args.out) if args.out else d / f"results_{args.target}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()