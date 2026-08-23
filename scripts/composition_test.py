#!/usr/bin/env python
"""
composition_test.py -- which features survive composition? (was ablation_stats
test C, corrected and made standalone)

  python scripts/composition_test.py \
      --pool /content/drive/MyDrive/adaptrack/pool_tinyllama \
      --features /content/drive/MyDrive/adaptrack/features \
      --target medical

THE CLAIM BEING TESTED
----------------------
For a LINEAR merge, Delta = sum_i lam_i Delta_i exactly. So a feature that is
linear in Delta passes through as the lambda-weighted mean of its constituents'
values; a feature that is not linear has no such guarantee.

    ||Delta_m||_F   nearly linear when constituent subspaces are near-
                    orthogonal -> should track its constituents
    eff_rank        a function of the spectrum of a SUM of subspaces; no
                    reason to track constituent values at all

Prediction: merged norm tracks constituents, merged eff_rank does not. If that
holds, "composition preserves magnitude allocation and destroys spectral
structure" is a measurement. If merged eff_rank ALSO tracks its constituents,
the mechanism is wrong and the abstract needs weakening.

Only linear merges are used. TIES/DARE/prune are not linear in Delta, so the
weighted-mean predictor has no theoretical standing there and a low R^2 would
prove nothing.

TWO CORRECTIONS OVER THE FIRST VERSION
--------------------------------------
1. EXACT FAST PATH, NOT FULL SVD. Constituent deltas are rank <= r and a K-way
   linear merge is rank <= K*r, so every spectrum here comes from the QR
   identity on a small matrix (verified exact to 4.4e-16 in spectral.py). The
   full-SVD version cost ~3 hours for 60 merges on 16 modules; this runs in
   under a minute on all ~88.

2. R^2 IS COMPUTED WITHIN MODULE, NOT POOLED. Pooling every module into one
   correlation is confounded by module identity: k_proj deltas are
   structurally smaller than q_proj deltas, and that is true of the merged
   delta AND its constituents. A pooled R^2 therefore comes out near 1.0 for
   any feature whatsoever -- simulated at 0.998 for data with zero
   merge-specific signal, where the within-module truth was 0.02. The pooled
   figure is still printed, labelled, so the size of the trap is visible.

TEST 2 -- PROFILE MATCHING
--------------------------
Separately: does merge i's observed per-module norm PROFILE match its own
predicted profile better than it matches some other merge's prediction? The
mismatched pairing is the null, and it controls for module identity by
construction, since both profiles run over the same modules. Reported as a
z-score of matched against the mismatched distribution.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gen_merges import FactorStore, _layer_of  # noqa: E402
from src.features.spectral import features_from_spectrum, svals_lowrank  # noqa: E402


def feats_from_factors(B, A, top_k=8):
    """Exact features for a low-rank delta given its factors."""
    sv = svals_lowrank(B.double(), A.double())
    fro = float(torch.sqrt((sv ** 2).sum()))
    return features_from_spectrum(sv, fro, top_k)


def merged_feats(picks, lambdas, store, name, top_k=8):
    B_cat = torch.cat([float(l) * store[k][name][0].double()
                       for k, l in zip(picks, lambdas)], dim=1)
    A_cat = torch.cat([store[k][name][1].double() for k in picks], dim=0)
    sv = svals_lowrank(B_cat, A_cat)
    fro = float(torch.sqrt((sv ** 2).sum()))
    return features_from_spectrum(sv, fro, top_k)


def r2(pred, obs):
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    if len(pred) < 3 or pred.std() < 1e-12 or obs.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(pred, obs)[0, 1] ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--target", default="medical")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--max-merges", type=int, default=80)
    ap.add_argument("--factor-cache", type=int, default=64)
    ap.add_argument("--features-to-test", nargs="+",
                    default=["fro", "nuc", "eff_rank", "entropy", "conc",
                             "stable_rank", "sv_1"])
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    feat = Path(args.features)
    rec_p = feat / f"recipes_{args.split}_{args.target}.json"
    if not rec_p.exists():
        sys.exit(f"missing {rec_p}")
    recs = json.loads(rec_p.read_text())

    lin = [r for r in recs if r["method"] == "linear"][:args.max_merges]
    if len(lin) < 10:
        sys.exit(f"only {len(lin)} linear merges in the {args.split} split; "
                 f"need >=10. Use --split train, which has more.")
    store = FactorStore(args.pool, capacity=args.factor_cache)

    mods = set(store.modules(lin[0]["constituents"][0]))
    for r in lin:
        for k in r["constituents"]:
            mods &= set(store.modules(k))
    names = sorted(mods, key=lambda n: (_layer_of(n), n))
    print(f"{len(lin)} linear merges x {len(names)} modules "
          f"(arities {sorted({r['arity'] for r in lin})})")

    # obs[feat][module] = list over merges; pred likewise
    test_feats = list(args.features_to_test) + ["fro_quad"]
    obs = {f: {m: [] for m in names} for f in test_feats}
    pred = {f: {m: [] for m in names} for f in test_feats}
    prof_obs, prof_pred = [], []

    for j, r in enumerate(lin):
        picks, lam = r["constituents"], r["lambdas"]
        po, pp = [], []
        for name in names:
            fm = merged_feats(picks, lam, store, name)
            cons = [feats_from_factors(*store[k][name]) for k in picks]
            for f in args.features_to_test:
                obs[f][name].append(fm[f])
                pred[f][name].append(sum(float(l) * c[f]
                                         for l, c in zip(lam, cons)))
            # ||Delta||_F has TWO candidate predictors and they differ:
            #   aligned subspaces      ->  sum_i lam_i ||Delta_i||_F
            #   orthogonal subspaces   ->  sqrt(sum_i lam_i^2 ||Delta_i||_F^2)
            # Testing only the arithmetic one understates how well the norm
            # composes whenever constituents are near-orthogonal, which for
            # independently-trained adapters they largely are. Which predictor
            # wins is itself informative: it measures how much the constituent
            # subspaces overlap.
            obs["fro_quad"][name].append(fm["fro"])
            pred["fro_quad"][name].append(
                float(np.sqrt(sum((float(l) * c["fro"]) ** 2
                                  for l, c in zip(lam, cons)))))
            po.append(fm["fro"])
            pp.append(sum(float(l) * c["fro"] for l, c in zip(lam, cons)))
        po, pp = np.array(po), np.array(pp)
        prof_obs.append(po / (np.linalg.norm(po) + 1e-30))
        prof_pred.append(pp / (np.linalg.norm(pp) + 1e-30))
        if (j + 1) % 20 == 0:
            print(f"  {j + 1}/{len(lin)}")

    # ---------------- TEST 1 ----------------
    print(f"\n{'=' * 74}")
    print("1. DOES THE MERGED VALUE TRACK THE lambda-WEIGHTED CONSTITUENT MEAN?")
    print("=" * 74)
    print(f"{'feature':<14}{'within-module R^2':>20}{'pooled R^2':>13}"
          f"{'merged/pred':>13}")
    print(f"{'':14}{'median [IQR]':>20}{'(confounded)':>13}{'mean ratio':>13}")
    summary = {}
    for f in test_feats:
        within = np.array([r2(pred[f][m], obs[f][m]) for m in names])
        within = within[np.isfinite(within)]
        allp = np.concatenate([pred[f][m] for m in names])
        allo = np.concatenate([obs[f][m] for m in names])
        pooled = r2(allp, allo)
        ratio = float(np.mean(allo) / max(np.mean(allp), 1e-30))
        med = float(np.median(within)) if within.size else np.nan
        q1 = float(np.percentile(within, 25)) if within.size else np.nan
        q3 = float(np.percentile(within, 75)) if within.size else np.nan
        print(f"{f:<14}{f'{med:.3f} [{q1:.3f},{q3:.3f}]':>20}"
              f"{pooled:>13.3f}{ratio:>13.2f}")
        summary[f] = dict(within_median=med, within_q1=q1, within_q3=q3,
                          pooled=pooled, ratio=ratio)

    print("\nRead the WITHIN-MODULE column. The pooled column is shown only to")
    print("document the trap: it is near 1.0 for everything because module")
    print("identity dominates it, and it would 'confirm' any hypothesis.")

    # ---------------- TEST 2 ----------------
    print(f"\n{'=' * 74}")
    print("2. PROFILE MATCHING (matched vs mismatched pairing)")
    print("=" * 74)
    P_o, P_p = np.array(prof_obs), np.array(prof_pred)
    matched = np.array([float(P_o[i] @ P_p[i]) for i in range(len(P_o))])
    rng = np.random.default_rng(0)
    mism = []
    for _ in range(args.n_perm):
        perm = rng.permutation(len(P_o))
        mism += [float(P_o[i] @ P_p[perm[i]])
                 for i in range(len(P_o)) if perm[i] != i]
    mism = np.array(mism)
    z = (matched.mean() - mism.mean()) / (mism.std() + 1e-30)
    print(f"matched   cos(observed profile, own prediction)    "
          f"{matched.mean():.5f} +/- {matched.std():.5f}")
    print(f"mismatched cos(observed profile, other prediction) "
          f"{mism.mean():.5f} +/- {mism.std():.5f}")
    print(f"z = {z:.1f}")
    if z > 3:
        print("-> The norm profile carries merge-SPECIFIC information, not just")
        print("   the generic module-shape profile shared by every merge.")
    else:
        print("-> The observed profile matches any merge's prediction about as")
        print("   well as its own. The profile is generic across merges, so it")
        print("   cannot be what the classifier is reading. Say so.")

    # ---------------- verdict ----------------
    print(f"\n{'=' * 74}\nVERDICT\n{'=' * 74}")
    mag_cands = {k: summary.get(k, {}).get("within_median", np.nan)
                 for k in ("nuc", "fro_quad", "fro")}
    mag_name = max(mag_cands, key=lambda k: (mag_cands[k]
                                             if np.isfinite(mag_cands[k]) else -1))
    nrm = mag_cands[mag_name]
    shape_cands = {k: summary.get(k, {}).get("within_median", np.nan)
                   for k in ("eff_rank", "entropy", "conc")}
    finite = [v for v in shape_cands.values() if np.isfinite(v)]
    er = float(np.median(finite)) if finite else np.nan
    print("magnitude features : " + ", ".join(
        f"{k}={v:.3f}" for k, v in mag_cands.items() if np.isfinite(v)))
    print("shape features     : " + ", ".join(
        f"{k}={v:.3f}" for k, v in shape_cands.items() if np.isfinite(v)))
    print(f"\nbest-composing magnitude feature: {mag_name} (R^2 {nrm:.3f})")
    print(f"median shape feature            : R^2 {er:.3f}")
    if np.isfinite(nrm) and np.isfinite(er) and nrm > 0.7 and er < 0.4:
        print("\nCONFIRMED. Magnitude allocation passes through the merge;")
        print("spectral shape does not track its constituents. These two R^2")
        print("values are the abstract's evidence -- they are a direct")
        print("measurement of the mechanism, where the AUC gap between feature")
        print("groups was only a noisy proxy for it (and, as the paired")
        print("bootstrap showed, not even a significant one).")
        print("\nSuggested phrasing:")
        print('  "Composition preserves the per-module magnitude of its')
        print('   constituents (%s, within-module R^2 = %.2f) while' % (mag_name, nrm))
        print('   destroying their spectral shape (median R^2 = %.2f), so what' % er)
        print('   survives a merge is where the update landed, not what shape')
        print('   it had."')
    elif np.isfinite(er) and er >= 0.4:
        print("\nNOT CONFIRMED. Merged eff_rank does track its constituents, so")
        print("'shape is averaged away' is not the mechanism. The likelier")
        print("reading is that shape was simply the weaker feature family all")
        print("along. Drop the mechanistic claim from the abstract and report")
        print("the decomposition descriptively.")
    else:
        print("\nMIXED. Report the table and describe rather than explain.")

    print("\nCaveat for the paper: this holds for LINEAR merges, where")
    print("Delta = sum lam_i Delta_i exactly. TIES/DARE/prune are not linear in")
    print("Delta, so the weighted-mean predictor has no standing there and this")
    print("test says nothing about them. Scope the sentence accordingly.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"summary": summary, "profile_z": float(z),
             "n_merges": len(lin), "n_modules": len(names)}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()