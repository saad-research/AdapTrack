#!/usr/bin/env python
"""
make_poster_figs.py -- poster / README figures for AdapTrack, drawn only from
files already committed under results/. No new experiments, no typed numbers.

  python scripts/make_poster_figs.py --results results --out figs

Writes (PNG at 300 dpi and PDF):
  figs/pipeline.*        what the attack does, as a schematic
  figs/defense_gap.*     fixed vs retrained attacker under each defense (Table 2)
  figs/lambda_bins.*     attack AUC by the target's share of the merge (Table 8)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE = "#fcfcfb"


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out / f"{name}.png")


# ---------------------------------------------------------------- pipeline
def pipeline(out):
    fig, ax = plt.subplots(figsize=(12.8, 3.4))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 12.8); ax.set_ylim(0, 3.4); ax.axis("off")

    def box(x, y, w, h, title, sub, edge=MUTED, fill="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    lw=1.4, ec=edge, fc=fill))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=11.5,
                color=INK, weight="bold")
        ax.text(x + w / 2, y + h - 0.72, sub, ha="center", va="top", fontsize=9.5,
                color=INK2, linespacing=1.35)

    def arrow(x0, x1, y=1.7):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color=MUTED, mutation_scale=16))

    # adapters
    fams = [("summarization", BLUE), ("MedMCQA", ORANGE), ("PubMedQA", AQUA), ("LEDGAR", YELLOW)]
    ax.text(1.05, 3.15, "LoRA adapters", ha="center", fontsize=11.5, weight="bold", color=INK)
    for i, (f, c) in enumerate(fams):
        y = 2.55 - i * 0.58
        ax.add_patch(FancyBboxPatch((0.15, y), 1.8, 0.42, boxstyle="round,pad=0.01,rounding_size=0.08",
                                    lw=0, fc=c, alpha=0.18))
        ax.add_patch(FancyBboxPatch((0.15, y), 0.12, 0.42, boxstyle="square,pad=0", lw=0, fc=c))
        ax.text(0.4, y + 0.21, f, va="center", fontsize=9.5, color=INK)
    arrow(2.05, 2.55)
    box(2.6, 0.75, 2.0, 1.9, "Merge", "task arithmetic,\nTIES, DARE,\nmagnitude pruning\n2–4 adapters")
    arrow(4.7, 5.2)
    box(5.25, 0.75, 2.0, 1.9, "Published\nweights only", "\nno queries\nno merge weights\nno metadata",
        edge=INK2)
    arrow(7.35, 7.85)
    box(7.9, 0.75, 1.9, 1.9, "Spectra", "singular values of\neach module's\nmerged update\n672 features")
    arrow(9.9, 10.35)
    box(10.4, 0.75, 2.3, 1.9, "Was family T\nin the merge?", "\nAUC 0.97\nfrom weights\nalone", edge=BLUE)
    save(fig, out, "pipeline")


# ---------------------------------------------------------------- defense gap
def defense(res, out):
    rows = json.loads((res / "defense_2x2_medical.json").read_text())
    pick = [("noise", None, 0.0, "No defense"),
            ("noise", None, 1.0, "Isotropic noise\n(c = 1.0)"),
            ("canon", "profile", 0.0, "Canonicalization\n(profile)"),
            ("canon", "full", 0.0, "Canonicalization\n(full)")]
    sel = []
    for arm, mode, c, label in pick:
        r = next(r for r in rows if r["arm"] == arm and r["mode"] == mode and float(r["c"]) == c)
        sel.append((label, r["auc_naive"], r["auc_adaptive"], r["ci_adaptive"]))

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    w, gap = 0.36, 0.02
    for i, (label, naive, adapt, ci) in enumerate(sel):
        ax.bar(i - w / 2 - gap, naive, w, color=ORANGE, zorder=2)
        ax.bar(i + w / 2 + gap, adapt, w, color=BLUE, zorder=2)
        ax.errorbar(i + w / 2 + gap, adapt, yerr=[[adapt - ci[0]], [ci[1] - adapt]],
                    fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=3)
        ax.text(i - w / 2 - gap, naive + 0.015, f"{naive:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.text(i + w / 2 + gap, ci[1] + 0.015, f"{adapt:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(1.5, 0.51, "chance", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.set_xticks(range(len(sel))); ax.set_xticklabels([s[0] for s in sel], fontsize=10)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Attack AUC (medical target)")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="fixed attacker (not retrained)"),
                       Patch(color=BLUE, label="adaptive attacker (retrained), 95% cluster CI")],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=9.5)
    ax.set_title("A defense that looks like it works, until the attacker retrains",
                 loc="left", fontsize=12.5, color=INK, weight="bold", pad=12)
    save(fig, out, "defense_gap")


# ---------------------------------------------------------------- lambda bins
def lambda_bins(res, out):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    labels = None
    for t, color, dy in (("medical", BLUE, -0.035), ("legal", ORANGE, 0.03)):
        d = json.loads((res / f"by_lambda_all_{t}.json").read_text())
        bins = d["bins"]
        x = list(range(len(bins)))
        y = [b["auc"] for b in bins]
        labels = [f"{b['lo']:.1f}–{b['hi']:.1f}\n{b['n_pos']} pos." for b in bins]
        ax.plot(x, y, color=color, lw=2, zorder=2)
        for xi, yi, b in zip(x, y, bins):
            small = b["skipped_by_min_count_rule"]
            ax.scatter(xi, yi, s=64, zorder=3, color=SURFACE if small else color,
                       edgecolor=color, linewidth=2)
        ax.text(x[0] - 0.12, y[0], f"{'MedMCQA' if t == 'medical' else 'LEDGAR'}  {y[0]:.2f}",
                ha="right", va="center", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlim(-1.3, len(labels) - 0.6)
    ax.set_ylim(0.6, 1.03); ax.set_ylabel("Attack AUC")
    ax.set_xlabel("Target family's share of the merge weight  (positives per bin)")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.text(len(labels) - 0.6, 0.62, "hollow = fewer than 10 positives (indicative)",
            ha="right", fontsize=8.5, color=MUTED)
    ax.set_title("Harder when the target is a small part of the merge",
                 loc="left", fontsize=12.5, color=INK, weight="bold", pad=12)
    save(fig, out, "lambda_bins")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pipeline(out)
    defense(res, out)
    lambda_bins(res, out)


if __name__ == "__main__":
    main()