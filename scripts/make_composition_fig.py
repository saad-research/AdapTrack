#!/usr/bin/env python3
"""
make_composition_fig.py -- Figure 1 for AdapTrack.

Reads a CSV you export from the composition test, so no number here is
transcribed by hand. Expected columns:

    feature,target,r2,lo,hi

  feature : fro_quad, fro_arith, nuc, sv_1, conc, stable_rank, eff_rank, entropy
  target  : medical | legal
  r2      : within-module R^2 (state which statistic in the caption --
            mean or median across modules -- and use the SAME one the prose
            quotes; the 0.14-0.23 range in the text does not currently
            reconcile with the per-feature values)
  lo,hi   : 95% interval; leave blank if unavailable and bars draw without them

Usage:
    python make_composition_fig.py composition_r2.csv figs/composition.pdf
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Grouped by what the statistic actually measures, not by the story we want.
GROUPS = [
    ("Magnitude", ["fro_quad", "fro_arith", "nuc"]),
    ("Leading singular value", ["sv_1"]),
    ("Spectral shape", ["conc", "stable_rank", "eff_rank", "entropy"]),
]
LABEL = {
    "fro_quad": r"$\|\Delta\|_F$ (quadrature)",
    "fro_arith": r"$\|\Delta\|_F$ (arithmetic)",
    "nuc": "nuclear norm",
    "sv_1": r"$\sigma_1$",
    "conc": "concentration",
    "stable_rank": "stable rank",
    "eff_rank": "participation ratio",
    "entropy": "spectral entropy",
}
TARGETS = ["medical", "legal"]
COLOR = {"medical": "#3B6EA5", "legal": "#C2703D"}


def load(path):
    d = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f):
            feat, tgt = row["feature"].strip(), row["target"].strip()
            if tgt not in TARGETS:
                raise SystemExit(f"unexpected target {tgt!r}")
            def num(k):
                v = row.get(k, "").strip()
                return float(v) if v else None
            d[feat][tgt] = (float(row["r2"]), num("lo"), num("hi"))
    return d


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    data = load(sys.argv[1])

    order, ticks, seps = [], [], []
    for gi, (gname, feats) in enumerate(GROUPS):
        present = [f for f in feats if f in data]
        missing = [f for f in feats if f not in data]
        if missing:
            print(f"[fig] warning: {gname}: no rows for {missing}")
        if gi:
            seps.append(len(order) - 0.5)
        for f in present:
            order.append(f)
            ticks.append(LABEL.get(f, f))

    n = len(order)
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * n + 1.0))
    h = 0.36
    for k, tgt in enumerate(TARGETS):
        ys, vals, errs = [], [], [[], []]
        for i, f in enumerate(order):
            if tgt not in data[f]:
                continue
            r2, lo, hi = data[f][tgt]
            ys.append(n - 1 - i + (h / 2 if k == 0 else -h / 2))
            vals.append(r2)
            errs[0].append(r2 - lo if lo is not None else 0.0)
            errs[1].append(hi - r2 if hi is not None else 0.0)
        err = np.array(errs) if any(errs[0]) or any(errs[1]) else None
        ax.barh(ys, vals, height=h, color=COLOR[tgt], label=tgt,
                xerr=err, error_kw=dict(lw=0.8, capsize=2, ecolor="0.25"))

    for s in seps:
        ax.axhline(n - 1 - s, color="0.75", lw=0.6)
    ax.set_yticks(range(n - 1, -1, -1))
    ax.set_yticklabels(ticks, fontsize=8)
    ax.set_xlabel(r"within-module $R^2$ of the composition predictor",
                  fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", lw=0.4, color="0.9")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(sys.argv[2], bbox_inches="tight")
    print(f"[fig] wrote {sys.argv[2]} ({n} features)")


if __name__ == "__main__":
    main()