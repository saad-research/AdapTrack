#!/usr/bin/env python
"""
probe_diagnostics.py -- interrogate floor-probe adapters. No retraining.

The floor probe said "separable". This says WHAT IS SEPARABLE, which is a
different and more consequential question. Runs in minutes on the adapters
already on disk.

  python probe_diagnostics.py --probe-dir .../floor_probe --n 2500

Six tests, in order of how much they can change the paper:

  A  4-way confusion matrix. Does medical (MedMCQA) collapse into medical_alt
     (PubMedQA), or separate? Separation means the signal is finer than task.

  B  Leave-one-DATASET-out transfer. Train 3-way on {summarization, medical,
     legal}; predict on held-out medical_alt adapters. If they land on
     "medical" above chance, the representation is task-level and transfers to
     an unseen dataset of the same task. This is the decisive test for the
     central claim and it does not need the pool.

  C  Binary medical-vs-rest, and medical-vs-medical_alt head to head.

  D  SIGNAL DECOMPOSITION. Split the feature set into three nested stories and
     score each alone:
       total_norm     one scalar, ||Delta||_F over all modules -> magnitude
       norm_profile   per-module norm / total norm -> WHERE the update landed
       shape_only     conc / eff_rank / stable_rank / entropy -> spectral shape
     These are not arbitrary groups. The shape statistics are provably
     invariant to per-module rescaling (asserted in spectral.py), so if they
     separate families on their own, the result cannot be a magnitude artifact.
     If total_norm alone separates them, the finding is DSiRe's finding.

  E  Scored by LOG-LOSS, not accuracy. At 1288 features and 20 adapters every
     group will hit accuracy 1.000 and you learn nothing from a ceiling. Log
     loss is a proper scoring rule and keeps ranking groups after accuracy
     saturates.

  F  CONFOUND BASELINES. Final training loss alone, and tokens-seen alone, each
     as a single-feature classifier. These are not available to a weight-only
     attacker, so they are not leakage -- they are candidate MECHANISMS. If
     loss alone predicts family perfectly, the honest description of the result
     is "how much the adapter moved", not "which task it learned".
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.spectral import (NORM_STATS, SHAPE_STATS,  # noqa: E402
                                   merge_features, stat_of)
from src.merge.delta import delta_from_lora  # noqa: E402


def load_deltas(adapter_dir: Path, r: int, alpha: int):
    from safetensors.torch import load_file
    p = adapter_dir / "adapter_model.safetensors"
    sd = (load_file(str(p)) if p.exists()
          else torch.load(adapter_dir / "adapter_model.bin", map_location="cpu"))
    return delta_from_lora(sd, r=r, alpha=alpha).deltas


def build_views(deltas: dict[str, torch.Tensor]):
    """Return the three nested feature views for one adapter."""
    feats = merge_features(deltas=deltas, svd_dtype=torch.float64)

    mods = sorted(deltas)
    fros = np.array([float(deltas[m].double().norm()) for m in mods])
    total = float(np.sqrt((fros ** 2).sum()))

    views = {
        "full": feats,
        "total_norm": {"total_fro": total},
        # relative allocation of update magnitude across modules. Scale-free by
        # construction, but still a magnitude story -- it says WHERE the update
        # went, not what shape it has.
        "norm_profile": {f"prof.{m}": float(fros[i] / (total + 1e-30))
                         for i, m in enumerate(mods)},
        "shape_only": {k: v for k, v in feats.items()
                       if stat_of(k) in SHAPE_STATS},
        "norms_only": {k: v for k, v in feats.items()
                       if stat_of(k) in NORM_STATS},
    }
    return views


def to_matrix(view_list):
    cols = sorted(view_list[0])
    return np.array([[v[c] for c in cols] for v in view_list],
                    dtype=np.float64), cols


def loo_scores(X, y, seed=0):
    """Leave-one-out accuracy AND log-loss (log loss keeps ranking at ceiling)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    classes = sorted(set(y.tolist()))
    n = len(y)
    correct, lls, preds = 0, [], []
    for i in range(n):
        tr = np.arange(n) != i
        if len(set(y[tr].tolist())) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, C=1.0,
                                               random_state=seed))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[i:i + 1])[0]
        cls = list(clf.classes_)
        pred = cls[int(np.argmax(p))]
        preds.append(pred)
        correct += int(pred == y[i])
        pi = p[cls.index(y[i])] if y[i] in cls else 1e-12
        lls.append(-np.log(max(pi, 1e-12)))
    return correct / n, float(np.mean(lls)), np.array(preds), classes


def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M


def print_confusion(M, labels, title):
    w = max(len(l) for l in labels) + 1
    print(f"\n{title}")
    print(" " * (w + 2) + "".join(f"{l[:9]:>11}" for l in labels) + "   <- predicted")
    for i, l in enumerate(labels):
        print(f"  {l:<{w}}" + "".join(f"{M[i, j]:>11}" for j in range(len(labels))))
    print("  (rows = true family)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--n", type=int, default=2500,
                    help="which dataset-size stratum to analyse")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--target", default="medical")
    ap.add_argument("--transfer-from", default="medical")
    ap.add_argument("--transfer-to", default="medical_alt")
    args = ap.parse_args()

    d = Path(args.probe_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    sel = {k: m for k, m in manifest.items()
           if m.get("n", m.get("n_examples")) == args.n}
    if not sel:
        sys.exit(f"no adapters at n={args.n}. present: "
                 f"{sorted({m.get('n', m.get('n_examples')) for m in manifest.values()})}")
    keys = sorted(sel)
    print(f"analysing {len(keys)} adapters at n={args.n}")
    views_all, fam, loss, toks = [], [], [], []
    for k in keys:
        adir = d / k
        if not (adir / "adapter_config.json").exists():
            continue
        views_all.append(build_views(load_deltas(adir, args.r, args.alpha)))
        m = sel[k]
        fam.append(m["family"])
        loss.append(m.get("train_loss", np.nan))
        # tokens actually seen: p50 length x examples is a decent proxy if
        # total_tokens was not logged. Prefer the real figure when present.
        toks.append(m.get("total_tokens",
                          m.get("tok_p50", np.nan) * m.get("n_examples", 1)))
    fam = np.array(fam)
    loss = np.array(loss, dtype=float)
    toks = np.array(toks, dtype=float)
    labels = sorted(set(fam.tolist()))
    print(f"families: {labels}  ({[int((fam == l).sum()) for l in labels]} adapters each)")

    report = {"n": args.n, "families": labels}

    # ---------------- A: 4-way confusion ----------------
    X, cols = to_matrix([v["full"] for v in views_all])
    acc, ll, preds, classes = loo_scores(X, fam)
    M = confusion(fam, preds, labels)
    print(f"\n{'=' * 72}\nA. 4-WAY CONFUSION (full features, {X.shape[1]} dims)\n{'=' * 72}")
    print(f"LOO accuracy {acc:.3f}   log loss {ll:.4f}")
    print_confusion(M, labels, "confusion:")

    a, b = args.transfer_from, args.transfer_to
    if a in labels and b in labels:
        ia, ib = labels.index(a), labels.index(b)
        cross = M[ia, ib] + M[ib, ia]
        tot = M[ia].sum() + M[ib].sum()
        print(f"\n  {a} <-> {b} cross-assignments: {cross}/{tot}")
        if cross == 0:
            print(f"  -> The two medical datasets are PERFECTLY separated.")
            print(f"     The classifier is not reading 'medical'. It is reading")
            print(f"     something dataset-specific. See test B before concluding")
            print(f"     which, because native rendering also differs in format.")
        else:
            print(f"  -> Some confusion between the two medical datasets: the")
            print(f"     representation has a task-level component.")
        report["cross_assignments"] = int(cross)

    # ---------------- B: leave-one-dataset-out transfer ----------------
    print(f"\n{'=' * 72}\nB. LEAVE-ONE-DATASET-OUT TRANSFER: train without {b}, predict {b}\n{'=' * 72}")
    if a in labels and b in labels:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        tr = fam != b
        te = fam == b
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, random_state=0))
        clf.fit(X[tr], fam[tr])
        P = clf.predict_proba(X[te])
        cls = list(clf.classes_)
        pred = [cls[i] for i in P.argmax(1)]
        chance = 1.0 / len(cls)
        hit = float(np.mean([p == a for p in pred]))
        print(f"held-out {b} adapters (n={te.sum()}) classified as:")
        for c in cls:
            frac = float(np.mean([p == c for p in pred]))
            mean_p = float(P[:, cls.index(c)].mean())
            mark = "  <- same task as held-out set" if c == a else ""
            print(f"  {c:<16} {frac:>6.1%} of adapters   mean prob {mean_p:.3f}{mark}")
        print(f"\nchance = {chance:.1%}")
        if hit > 0.6:
            print(f"-> {hit:.0%} land on '{a}'. The features encode TASK, and they")
            print(f"   transfer to a dataset never seen in training. This supports")
            print(f"   task-level provenance and it is the strongest single result")
            print(f"   you have. Put it in the main body, not the appendix.")
        elif hit <= chance + 0.1:
            print(f"-> Only {hit:.0%} land on '{a}' (chance {chance:.0%}). NO task-level")
            print(f"   transfer. The honest claim is DATASET-level provenance:")
            print(f"   'was this specific corpus present', not 'was a medical")
            print(f"   adapter present'. That is a narrower claim but arguably a")
            print(f"   BETTER fit for ATTRIB -- data-use litigation is about")
            print(f"   specific datasets, not abstract task categories. Reframe")
            print(f"   rather than discard.")
        else:
            print(f"-> Partial transfer ({hit:.0%} vs chance {chance:.0%}). Report both")
            print(f"   granularities and be explicit about the boundary.")
        report["transfer_hit_rate"] = hit
        report["transfer_chance"] = chance

    # ---------------- C: binary probes ----------------
    print(f"\n{'=' * 72}\nC. BINARY PROBES\n{'=' * 72}")
    yb = np.where(fam == args.target, args.target, "rest")
    acc_b, ll_b, _, _ = loo_scores(X, yb)
    base = max((yb == v).mean() for v in set(yb.tolist()))
    print(f"{args.target} vs rest        : LOO acc {acc_b:.3f} "
          f"(majority baseline {base:.3f})  log loss {ll_b:.4f}")
    if a in labels and b in labels:
        pair = np.isin(fam, [a, b])
        acc_p, ll_p, _, _ = loo_scores(X[pair], fam[pair])
        print(f"{a} vs {b}: LOO acc {acc_p:.3f} "
              f"(chance 0.500)  log loss {ll_p:.4f}")
        print("  A head-to-head accuracy of 1.000 here is the same ceiling")
        print("  problem; the log loss is the number that discriminates.")
        report["pairwise_medical"] = {"acc": acc_p, "log_loss": ll_p}

    # ---------------- D/E: signal decomposition ----------------
    print(f"\n{'=' * 72}\nD. SIGNAL DECOMPOSITION (4-way, ranked by log loss)\n{'=' * 72}")
    print(f"{'view':<16}{'dims':>6}{'LOO acc':>10}{'log loss':>11}")
    rows = []
    for name in ("total_norm", "norm_profile", "shape_only", "norms_only", "full"):
        Xi, _ = to_matrix([v[name] for v in views_all])
        if Xi.shape[1] < 1:
            continue
        ai, li, _, _ = loo_scores(Xi, fam)
        rows.append((name, Xi.shape[1], ai, li))
        print(f"{name:<16}{Xi.shape[1]:>6}{ai:>10.3f}{li:>11.4f}")
    report["decomposition"] = {r[0]: {"dims": r[1], "acc": r[2], "log_loss": r[3]}
                               for r in rows}

    got = dict((r[0], r[2]) for r in rows)
    print()
    if got.get("total_norm", 0) >= 0.95:
        print("A SINGLE SCALAR -- total ||Delta||_F -- separates the families.")
        print("Then the spectral features are decoration and the mechanism is")
        print("magnitude. That is DSiRe's result transposed, and a reviewer who")
        print("knows DSiRe will say so. You must either (a) report the norm")
        print("result honestly and locate the novelty in the MERGE setting")
        print("rather than the features, or (b) show shape_only also works,")
        print("which makes the spectral claim independent.")
    elif got.get("shape_only", 0) >= 0.9:
        print("shape_only separates families while total_norm does not do it")
        print("alone. The shape statistics are invariant to per-module")
        print("rescaling, so this cannot be a magnitude artifact. This is the")
        print("outcome that earns the word 'spectral' in the paper.")
    else:
        print("Neither magnitude nor shape dominates alone. Report the")
        print("decomposition table as-is; it is more informative than a single")
        print("headline accuracy.")

    # ---------------- F: confound baselines ----------------
    print(f"\n{'=' * 72}\nF. MECHANISM BASELINES (not attacker-observable)\n{'=' * 72}")
    for nm, v in (("final train loss", loss), ("tokens seen", toks)):
        if np.isnan(v).any():
            print(f"{nm:<18}: not logged for all adapters, skipped")
            continue
        ai, li, _, _ = loo_scores(v.reshape(-1, 1), fam)
        print(f"{nm:<18}: LOO acc {ai:.3f}  log loss {li:.4f}   "
              f"(1 feature, chance {1/len(labels):.3f})")
        report[f"baseline_{nm.replace(' ', '_')}"] = {"acc": ai, "log_loss": li}
        if ai >= 0.95:
            print(f"  {nm} ALONE predicts family. It is not a leak -- a")
            print(f"  weight-only attacker cannot see it -- but it is the")
            print(f"  likely MECHANISM, and it belongs in the Discussion as")
            print(f"  such before a reviewer proposes it as an alternative")
            print(f"  explanation you did not consider.")

    # correlation of the headline feature with the confounds
    if not np.isnan(loss).any():
        tn = np.array([v["total_norm"]["total_fro"] for v in views_all])
        print(f"\ncorr(||Delta||_F, final loss)  = {np.corrcoef(tn, loss)[0,1]:+.3f}")
        if not np.isnan(toks).any():
            print(f"corr(||Delta||_F, tokens seen) = {np.corrcoef(tn, toks)[0,1]:+.3f}")
            print("  Strong correlations here mean the three confounds are one")
            print("  confound. Control the upstream one (tokens) and the others")
            print("  move with it.")

    out = d / f"probe_diagnostics_n{args.n}.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()