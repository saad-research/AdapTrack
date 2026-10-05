#!/usr/bin/env python
"""
camera_ready_checks.py -- two artifacts added for the ATTRIB camera-ready.

  python scripts/camera_ready_checks.py \
      --features /content/drive/MyDrive/adaptrack/features --out-dir results

Writes
  results/by_lambda_all_{medical,legal}.json
      AUC in all five merge-weight bins, with the positive count per bin.
      fit_classifier.py skips bins with fewer than 10 positives; this reports
      every bin and marks which ones that rule would have skipped.
  results/same_domain_negatives_medical.json
      Medical positives scored against only the negatives that contain
      medical_alt (PubMedQA) and no medical (MedMCQA) adapter, with a cluster
      bootstrap interval.

Nothing here refits differently from Table 1: the classifier, the feature
filter and the bootstrap are imported from fit_classifier.py, so the interval
uses the same rule as Table 1's (a merge survives a replicate only if ALL of
its constituents were resampled). The script first reproduces the headline
AUC and its interval and prints them, so a mismatch is visible before any new
number is trusted.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fit_classifier import cluster_bootstrap, fit_predict  # noqa: E402

EDGES = [0, 0.2, 0.4, 0.6, 0.8, 1.01]      # same edges as fit_classifier.py
PAPER = {"medical": 0.9685, "legal": 0.9752}


def load_and_score(d, target):
    tr = np.load(d / f"features_train_{target}.npz", allow_pickle=True)
    te = np.load(d / f"features_test_{target}.npz", allow_pickle=True)
    rec = json.loads((d / f"recipes_test_{target}.json").read_text())

    ids_npz = [str(x) for x in te["merge_id"]]
    ids_rec = [str(r["merge_id"]) for r in rec]
    if ids_npz != ids_rec:
        sys.exit(f"{target}: feature rows and recipes are not in the same order")

    Xtr, ytr = tr["X"], tr["label"]
    Xte, yte = te["X"], te["label"].astype(int)
    # same guard as fit_classifier.py
    ok = np.isfinite(Xtr).all(0) & np.isfinite(Xte).all(0) & (Xtr.std(0) > 0)
    _, s = fit_predict(Xtr[:, ok], ytr, Xte[:, ok], "lr")
    return rec, yte, s, te["lam_target"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="fit_classifier.py default; match what produced Table 1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = Path(args.features)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for target in ("medical", "legal"):
        rec, y, s, lam = load_and_score(d, target)
        cons = [r["constituents"] for r in rec]
        auc = float(roc_auc_score(y, s))
        lo, hi, n_ok = cluster_bootstrap(y, s, cons, args.n_boot, args.seed)

        print(f"\n{'=' * 70}\n{target}\n{'=' * 70}")
        print(f"headline AUC {auc:.4f}   (paper {PAPER[target]:.4f})")
        print(f"cluster CI   [{lo:.4f}, {hi:.4f}]   {n_ok}/{args.n_boot} usable"
              f"   <- compare with Table 1")
        if abs(auc - PAPER[target]) > 5e-5:
            print("WARNING: headline AUC does not reproduce. Do not use the "
                  "numbers below until this is explained.")

        # ---------------- all five lambda bins ----------------
        pos, neg = y == 1, y == 0
        bins = []
        print(f"\nby lambda on the target (positives in bin vs all "
              f"{int(neg.sum())} negatives):")
        for a, b in zip(EDGES[:-1], EDGES[1:]):
            inbin = pos & (lam >= a) & (lam < b)
            n_pos = int(inbin.sum())
            sel = inbin | neg
            a_bin = float(roc_auc_score(y[sel], s[sel])) if n_pos else None
            skipped = n_pos < 10
            shown = "empty" if a_bin is None else f"{a_bin:.4f}"
            print(f"  [{a:.1f},{min(b, 1.0):.1f})  n_pos={n_pos:<3} AUC {shown}"
                  f"{'   (skipped by the n_pos<10 rule)' if skipped else ''}")
            bins.append({"lo": a, "hi": min(b, 1.0), "n_pos": n_pos,
                         "auc": a_bin,
                         "skipped_by_min_count_rule": bool(skipped)})
        (out / f"by_lambda_all_{target}.json").write_text(json.dumps({
            "target": target, "n_test": int(len(y)),
            "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
            "headline_auc": auc, "min_count_rule": "n_pos < 10 skipped in "
            "fit_classifier.py; all bins reported here",
            "bins": bins}, indent=2))
        print(f"wrote {out / f'by_lambda_all_{target}.json'}")

        # ---------------- same-domain negatives (medical only) ----------------
        if target != "medical":
            continue
        fams = [r["families"] for r in rec]
        alt = np.array(["medical_alt" in f and "medical" not in f for f in fams])
        bad = int((alt & pos).sum())
        if bad:
            sys.exit(f"{bad} medical_alt-without-medical merges are labelled "
                     f"positive; the positive-class definition is not what "
                     f"the paper states")
        neg_alt, neg_other = alt & neg, (~alt) & neg

        # restrict FIRST, then bootstrap on the restricted list
        sel = pos | neg_alt
        idx = np.flatnonzero(sel)
        y_r, s_r = y[idx], s[idx]
        cons_r = [cons[i] for i in idx]
        auc_alt = float(roc_auc_score(y_r, s_r))
        lo_a, hi_a, n_ok_a = cluster_bootstrap(y_r, s_r, cons_r,
                                               args.n_boot, args.seed)
        sel_o = pos | neg_other
        auc_other = float(roc_auc_score(y[sel_o], s[sel_o]))

        print(f"\nsame-domain negatives (PubMedQA present, MedMCQA absent):")
        print(f"  {int(pos.sum())} positives vs {int(neg_alt.sum())} negatives"
              f"   AUC {auc_alt:.4f}   cluster CI [{lo_a:.4f}, {hi_a:.4f}]"
              f"   {n_ok_a}/{args.n_boot} usable")
        print(f"  {int(pos.sum())} positives vs {int(neg_other.sum())} other "
              f"negatives   AUC {auc_other:.4f}   (no interval: too few)")
        (out / "same_domain_negatives_medical.json").write_text(json.dumps({
            "target": "medical",
            "definition": "negatives containing a medical_alt adapter and no "
                          "medical adapter",
            "n_pos": int(pos.sum()),
            "n_neg_same_domain": int(neg_alt.sum()),
            "n_neg_other": int(neg_other.sum()),
            "auc_vs_same_domain_negatives": auc_alt,
            "ci_cluster": [lo_a, hi_a],
            "bootstrap": {"function": "fit_classifier.cluster_bootstrap",
                          "rule": "merge kept only if ALL constituents "
                                  "resampled",
                          "n_boot": args.n_boot, "usable": n_ok_a,
                          "seed": args.seed,
                          "restriction_applied_before_resampling": True},
            "auc_vs_other_negatives": auc_other,
            "headline_auc": auc, "headline_ci_cluster": [lo, hi]}, indent=2))
        print(f"wrote {out / 'same_domain_negatives_medical.json'}")


if __name__ == "__main__":
    main()