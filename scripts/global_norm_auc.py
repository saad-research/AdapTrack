#!/usr/bin/env python
"""
global_norm_auc.py -- AUC of each single global-norm summary on merged updates.

  python scripts/global_norm_auc.py --features /content/drive/MyDrive/adaptrack/features \
      --out results/global_norm_auc.json

Added for the camera-ready. Table 6 reports one single-scalar baseline,
GLOBAL.fro.mean. This script reports every GLOBAL.fro.* summary, so the text
can state the range rather than one value. A single feature needs no model:
AUC is computed on the raw feature, and on its negation in case the direction
is reversed. No fitting, no randomness.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SKIP = {"X", "label", "merge_id", "lam_target"}


def feature_names(npz):
    for k in npz.files:
        if k in SKIP:
            continue
        v = npz[k]
        if v.ndim == 1 and len(v) == npz["X"].shape[1]:
            return [str(x) for x in v]
    raise SystemExit(f"no feature-name array found; keys: {npz.files}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="results/global_norm_auc.json")
    a = ap.parse_args()
    res = {"split": "test", "note": "raw-feature AUC; 'flipped' is 1 - raw"}
    for t in ("medical", "legal"):
        te = np.load(Path(a.features) / f"features_test_{t}.npz", allow_pickle=True)
        names = feature_names(te)
        y = te["label"].astype(int)
        rows = {}
        for i, n in enumerate(names):
            if n.startswith("GLOBAL.fro."):
                auc = float(roc_auc_score(y, te["X"][:, i]))
                rows[n] = {"auc": auc, "flipped": 1 - auc, "best": max(auc, 1 - auc)}
        # whole-update Frobenius norm over the extracted modules:
        # sqrt(sum_m ||Delta_m||_F^2), from the per-module .fro columns
        import re
        idx = [i for i, n in enumerate(names)
               if not n.startswith("GLOBAL") and re.search(r"\.fro$", n)]
        tot = np.sqrt((te["X"][:, idx] ** 2).sum(1))
        auc = float(roc_auc_score(y, tot))
        rows[f"total_fro_sqrt_sum_sq_{len(idx)}_modules"] = {
            "auc": auc, "flipped": 1 - auc, "best": max(auc, 1 - auc)}
        res[t] = {"n_test": int(len(y)), "summaries": rows,
                  "max_best": max(r["best"] for r in rows.values())}
        for n, r in rows.items():
            print(f"{t:8s} {n:22s} AUC {r['auc']:.4f}  flipped {r['flipped']:.4f}")
        print(f"{t:8s} best single global-norm summary: {res[t]['max_best']:.4f}\n")
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()