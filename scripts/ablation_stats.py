#!/usr/bin/env python
"""
ablation_stats.py -- is the feature-group ordering real, and why does it invert?

  python scripts/ablation_stats.py --features ... --pool ... --target medical

TWO PROBLEMS WITH READING THE ABLATION TABLE AS RANKED
------------------------------------------------------
1. UNPAIRED POINT ESTIMATES. norms_only 0.9749 vs full 0.9685 is a gap of
   0.0064 on ONE test set of ~150 merges drawn from ~16 adapters. Simulating
   two genuinely TIED feature groups scored on the same 150 samples puts the
   95th percentile of |AUC difference| at 0.0070. The top four groups are
   inside that. They are tied, and reporting them as an ordering is reading
   noise. Only spectral_shape (0.8811) sits clearly below.

   Test A below does the paired comparison properly: it resamples ADAPTERS and
   recomputes every feature group's AUC on the SAME resample, so the
   difference is paired and the adapter-level dependence is carried through.

2. THE PRE/POST COMPARISON IS NOT LIKE FOR LIKE. On unmerged adapters the
   contrast was total_norm (ONE scalar) at 0.450 against shape_only (many
   features). Post-merge it is norms_only (96 features) against
   spectral_shape (192). A single scalar losing to a 192-dim group says
   little about magnitude versus shape; it mostly says 1 < 192. Test B
   evaluates the SAME groups on both sides so the inversion claim has a
   controlled basis.

TEST C: WHICH FEATURES COMPOSE?
-------------------------------
The mechanistic story -- merging averages away spectral shape while leaving
per-module magnitude allocation intact -- is checkable rather than assertable.
For a linear merge, Delta = sum_i lam_i Delta_i, so any feature that is LINEAR
in Delta is preserved as the lambda-weighted average of its constituents'
values; any feature that is not linear has no such guarantee.

    ||Delta_m||_F        not linear, but nearly so when constituents are
                         near-orthogonal: predicted well by the weighted mean
    eff_rank, entropy    functions of the whole spectrum of a SUM of subspaces;
                         no reason to track constituent values at all

So the prediction is: the merged per-module norm profile correlates strongly
with the lambda-weighted mean of constituent profiles, while merged eff_rank
does not. If that holds, the abstract's claim is a measurement. If merged
eff_rank also tracks its constituents, the story is wrong and the shape drop
needs a different explanation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.spectral import feature_groups, stat_of  # noqa: E402


def fit_score(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=10000, random_state=0))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def paired_cluster_bootstrap(scores, y, constituents, n_boot=1000, seed=0):
    """AUC and paired differences, resampling ADAPTERS not merges.

    scores: {group_name: score_vector}. Every group is scored on the same
    resample, so the differences are paired and the CI is a CI on the
    DIFFERENCE, which is what the ordering claim needs.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    adapters = sorted({a for c in constituents for a in c})
    names = list(scores)
    acc = {n: [] for n in names}
    for _ in range(n_boot):
        keep = set(rng.choice(adapters, len(adapters), replace=True).tolist())
        sel = np.array([all(a in keep for a in c) for c in constituents])
        if sel.sum() < 20 or len(set(y[sel].tolist())) < 2:
            continue
        for n in names:
            acc[n].append(roc_auc_score(y[sel], scores[n][sel]))
    return {n: np.array(v) for n, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--pool", default=None,
                    help="needed for test C (composition analysis)")
    ap.add_argument("--target", default="medical")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--max-linear-merges", type=int, default=60)
    args = ap.parse_args()

    feat = Path(args.features)
    tr = np.load(feat / f"features_train_{args.target}.npz", allow_pickle=True)
    te = np.load(feat / f"features_test_{args.target}.npz", allow_pickle=True)
    cols = [str(c) for c in tr["columns"]]
    ytr, yte = tr["label"], te["label"]
    rec_te = json.loads((feat / f"recipes_test_{args.target}.json").read_text())
    constituents = [r["constituents"] for r in rec_te]

    ok = (np.isfinite(tr["X"]).all(0) & np.isfinite(te["X"]).all(0)
          & (tr["X"].std(0) > 0))
    Xtr, Xte = tr["X"][:, ok], te["X"][:, ok]
    cols = [c for c, k in zip(cols, ok) if k]

    groups = feature_groups(cols)
    # the single scalar the floor probe used, for the like-for-like comparison
    groups["total_norm_only"] = [c for c in cols if c == "GLOBAL.fro.mean"]

    # ---------------- TEST A ----------------
    print("=" * 74)
    print("A. PAIRED CLUSTER BOOTSTRAP OVER ADAPTERS")
    print("=" * 74)
    scores = {}
    for name, keep in groups.items():
        idx = [i for i, c in enumerate(cols) if c in set(keep)]
        if len(idx) < 1:
            continue
        scores[name] = fit_score(Xtr[:, idx], ytr, Xte[:, idx])
    boots = paired_cluster_bootstrap(scores, yte, constituents, args.n_boot)

    from sklearn.metrics import roc_auc_score
    print(f"{'group':<20}{'dims':>6}{'AUC':>9}{'95% CI':>20}")
    order = sorted(scores, key=lambda n: -roc_auc_score(yte, scores[n]))
    for n in order:
        d = len([i for i, c in enumerate(cols) if c in set(groups[n])])
        a = roc_auc_score(yte, scores[n])
        b = boots[n]
        print(f"{n:<20}{d:>6}{a:>9.4f}"
              f"{f'[{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]':>20}")

    best = order[0]
    print(f"\npaired differences against '{best}' "
          f"(CI excluding 0 = a real gap):")
    for n in order[1:]:
        d = boots[best] - boots[n]
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        sig = "DISTINGUISHABLE" if lo > 0 else "tied"
        print(f"  {best} - {n:<20} {d.mean():+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  {sig}")
    print("\nGroups marked 'tied' must be reported as tied. An ordering among")
    print("them is not supported by this test set and a reviewer who reruns it")
    print("will find that out.")

    # ---------------- TEST B ----------------
    print(f"\n{'=' * 74}\nB. LIKE-FOR-LIKE PRE/POST COMPARISON\n{'=' * 74}")
    print("Post-merge values for the groups the floor probe used:")
    for n in ("total_norm_only", "norms_only", "spectral_shape"):
        if n in scores:
            print(f"  {n:<20} {roc_auc_score(yte, scores[n]):.4f}")
    print("\nTo close the comparison, rerun probe_diagnostics.py on the")
    print("UNMERGED pool adapters and read off norms_only and spectral_shape")
    print("there. The inversion claim needs the same two groups on both sides;")
    print("the floor probe's total_norm was a single scalar and is not a fair")
    print("stand-in for the 96-dim norms_only group used here.")

    # ---------------- TEST C ----------------
    if args.pool:
        print(f"\n{'=' * 74}\nC. WHICH FEATURES SURVIVE COMPOSITION\n{'=' * 74}")
        from scripts.gen_merges import FactorStore, _layer_of
        from src.features.spectral import module_features
        store = FactorStore(args.pool, capacity=64)

        lin = [r for r in rec_te if r["method"] == "linear"][:args.max_linear_merges]
        if not lin:
            print("no linear merges in the test set; skipping")
            return
        print(f"analysing {len(lin)} linear merges "
              f"(linear only: Delta = sum lam_i Delta_i holds exactly)")

        pred_fro, obs_fro, pred_er, obs_er = [], [], [], []
        for r in lin:
            picks, lam = r["constituents"], r["lambdas"]
            mods = set(store.modules(picks[0]))
            for k in picks[1:]:
                mods &= set(store.modules(k))
            names = sorted(mods)[:16]     # a sample of modules is enough
            for name in names:
                Ds = [store[k][name][0].double() @ store[k][name][1].double()
                      for k in picks]
                merged = sum(float(l) * d for l, d in zip(lam, Ds))
                fm = module_features(D=merged, svd_dtype=torch.float64)
                cons = [module_features(D=d, svd_dtype=torch.float64) for d in Ds]
                obs_fro.append(fm["fro"])
                pred_fro.append(sum(float(l) * c["fro"] for l, c in zip(lam, cons)))
                obs_er.append(fm["eff_rank"])
                pred_er.append(sum(float(l) * c["eff_rank"]
                                   for l, c in zip(lam, cons)))
                del Ds, merged

        def r2(pred, obs):
            pred, obs = np.array(pred), np.array(obs)
            return float(np.corrcoef(pred, obs)[0, 1] ** 2)

        r2_fro, r2_er = r2(pred_fro, obs_fro), r2(pred_er, obs_er)
        print(f"\n  ||Delta_m||_F : merged vs lambda-weighted mean of "
              f"constituents  R^2 = {r2_fro:.3f}")
        print(f"  eff_rank      : merged vs lambda-weighted mean of "
              f"constituents  R^2 = {r2_er:.3f}")
        print(f"\n  mean eff_rank: constituents {np.mean(pred_er):.2f} "
              f"-> merged {np.mean(obs_er):.2f} "
              f"({np.mean(obs_er) / max(np.mean(pred_er), 1e-9):.2f}x)")
        print()
        if r2_fro > 0.7 and r2_er < 0.4:
            print("  CONFIRMED. The per-module norm profile passes through the")
            print("  merge close to linearly, while spectral shape does not")
            print("  track its constituents at all. That is direct mechanistic")
            print("  support for the abstract: composition preserves magnitude")
            print("  allocation and destroys spectral structure. Report both")
            print("  R^2 values -- they are the evidence, not the AUC gap.")
        elif r2_er >= 0.4:
            print("  NOT CONFIRMED. Merged eff_rank does track its constituents,")
            print("  so 'shape is averaged away' is not the mechanism and the")
            print("  spectral_shape drop needs another explanation -- most")
            print("  likely that shape is simply the weaker feature family here")
            print("  and always was. Weaken the abstract accordingly.")
        else:
            print("  Mixed. Report the R^2 values and describe rather than")
            print("  explain the effect.")


if __name__ == "__main__":
    main()