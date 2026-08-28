#!/usr/bin/env python3
"""
diagnostics.py -- AdapTrack post-hoc diagnostics for section 6.

(1) shape-only baseline. Does canon:full retain information beyond the four
    scale-invariant shape statistics? Canonicalization is a per-module positive
    rescale, under which nuc/fro and sv_i/fro are unchanged. If canon:full
    exceeds shape-only on the SAME test merges, that surviving normalized
    spectrum is what the adaptive attacker recovers.

(2) paired cluster bootstrap. The arms share one test set, so differences must
    be bootstrapped paired: resample adapters once per replicate and score
    every arm on the same rows.

Reads only the clean feature npz files. Canonicalization is exact
post-processing; nothing is re-merged or re-extracted.

The noise arms are NOT reproducible here -- noise was added in tensor space and
the features re-extracted -- so this covers clean / canon:profile / canon:full
/ shape-only.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SHAPE_STATS = {"conc", "eff_rank", "stable_rank", "entropy"}
DEGREE0 = SHAPE_STATS
DEGREE1 = {"fro", "nuc"} | {f"sv_{i}" for i in range(1, 9)}
DEFENSE_METHODS = ("linear", "dare_ties")

EXPECT_SHAPE_COLS = 192          # 4 stats x (44 modules + 4 aggregates)
EXPECT_N_TRAIN = 161
EXPECT_N_TEST = 68
EXPECT_BASELINE_AUC = 0.9288     # defense-arm internal baseline, medical


# ----------------------------------------------------------------- parsing
def stat_of(col: str) -> str:
    """'<module>.<stat>' -> stat;  'GLOBAL.<stat>.<agg>' -> stat."""
    parts = col.split(".")
    if parts[0] == "GLOBAL":
        if len(parts) != 3:
            raise ValueError(f"unexpected GLOBAL column {col!r}")
        return parts[1]          # middle field, not last
    return parts[-1]


def agg_of(col: str):
    parts = col.split(".")
    return parts[2] if parts[0] == "GLOBAL" else None


class Cols:
    def __init__(self, columns):
        self.columns = [str(c) for c in columns]
        self.is_global = np.array([c.startswith("GLOBAL.") for c in self.columns])
        self.stats = [stat_of(c) for c in self.columns]
        self.aggs = [agg_of(c) for c in self.columns]

        unknown = {s for s in self.stats if s not in DEGREE0 | DEGREE1}
        if unknown:
            raise SystemExit(f"HARD STOP: unclassified stats {sorted(unknown)}")

        self.modules = sorted({c.rsplit(".", 1)[0]
                               for c, g in zip(self.columns, self.is_global)
                               if not g})
        pos = {c: j for j, c in enumerate(self.columns)}
        self.by_stat = {}
        for s in DEGREE0 | DEGREE1:
            idx = [pos.get(f"{m}.{s}") for m in self.modules]
            if all(i is not None for i in idx):
                self.by_stat[s] = np.array(idx)
        if "fro" not in self.by_stat:
            raise SystemExit("HARD STOP: no per-module 'fro' columns found")
        self.fro = self.by_stat["fro"]

        self.global_idx = {}
        for j, (g, s, a) in enumerate(zip(self.is_global, self.stats, self.aggs)):
            if g:
                self.global_idx[(s, a)] = j

    def shape_mask(self):
        return np.array([s in SHAPE_STATS for s in self.stats])


AGGS = {"mean": lambda v: v.mean(1),
        "max": lambda v: v.max(1),
        "min": lambda v: v.min(1)}


def collect_module_vals(X, C):
    return {(s, a): X[:, C.by_stat[s]]
            for (s, a) in C.global_idx if s in C.by_stat}


def rebuild_globals(per_module_vals, ddof):
    out = {}
    for (s, a), vals in per_module_vals.items():
        out[(s, a)] = vals.std(1, ddof=ddof) if a == "std" else AGGS[a](vals)
    return out


def validate_globals(X, C):
    """Return (best_ddof, max_rel_err). Auto-detects the std convention."""
    best = (0, float("inf"))
    for ddof in (0, 1):
        rebuilt = rebuild_globals(collect_module_vals(X, C), ddof)
        errs = []
        for (s, a), got in rebuilt.items():
            want = X[:, C.global_idx[(s, a)]]
            errs.append(np.max(np.abs(got - want)
                               / np.maximum(np.abs(want), 1e-12)))
        e = float(max(errs))
        if e < best[1]:
            best = (ddof, e)
    return best


# ---------------------------------------------------------- canonicalization
def fit_profile(X_train, C):
    F = X_train[:, C.fro]
    G = np.sqrt((F ** 2).sum(1, keepdims=True))
    rbar = (F / np.maximum(G, 1e-30)).mean(0)
    rbar = rbar / np.sqrt((rbar ** 2).sum())
    return rbar, float(G.mean())


def canonicalize(X, C, rbar, mode, Gbar, ddof):
    X = np.array(X, dtype=np.float64, copy=True)
    F = X[:, C.fro]
    G = np.sqrt((F ** 2).sum(1))
    S = G if mode == "profile" else np.full_like(G, Gbar)
    t = (rbar[None, :] * S[:, None]) / np.maximum(F, 1e-30)   # (n, M)
    for s in DEGREE1:
        if s in C.by_stat:
            X[:, C.by_stat[s]] *= t
    for (s, a), v in rebuild_globals(collect_module_vals(X, C), ddof).items():
        X[:, C.global_idx[(s, a)]] = v
    return X


# ------------------------------------------------------------------ fitting
def fit_score(Xtr, ytr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=5000, random_state=seed).fit(
        sc.transform(Xtr), ytr)
    return clf.decision_function(sc.transform(Xte))


def paired_boot(scores, y, groups, n_boot, seed):
    """groups: per-test-merge lists of adapter ids, or None for a per-merge
    fallback."""
    rng = np.random.default_rng(seed)
    if groups is None:
        units = list(range(len(y)))
        rows = {i: [i] for i in units}
    else:
        units = sorted({a for g in groups for a in g})
        rows = {a: [i for i, g in enumerate(groups) if a in g] for a in units}
    names = list(scores)
    per_arm = {k: [] for k in names}
    per_pair = {(a, b): [] for i, a in enumerate(names) for b in names[i + 1:]}
    kept = 0
    for _ in range(n_boot):
        pick = rng.choice(len(units), len(units), replace=True)
        idx = np.concatenate([rows[units[p]] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        kept += 1
        a_ = {k: roc_auc_score(y[idx], s[idx]) for k, s in scores.items()}
        for k, v in a_.items():
            per_arm[k].append(v)
        for (a, b) in per_pair:
            per_pair[(a, b)].append(a_[a] - a_[b])

    def ci(v):
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    marg = {k: ci(v) for k, v in per_arm.items()}
    pair = {}
    for (a, b), v in per_pair.items():
        v = np.asarray(v)
        lo, hi = ci(v)
        pair[f"{a} - {b}"] = {
            "median": float(np.median(v)), "lo": lo, "hi": hi,
            "p": float(min(2.0 * min((v <= 0).mean(), (v >= 0).mean()), 1.0)),
        }
    return marg, pair, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--merges", default=None,
                    help="JSON with per-merge constituent adapter ids, joined "
                         "on merge_id. WITHOUT IT the bootstrap is per-merge "
                         "and NOT comparable to the paper's intervals.")
    ap.add_argument("--merge-id-key", default="merge_id")
    ap.add_argument("--constituents-key", default="constituents")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output FILE, not a directory")
    args = ap.parse_args()

    dtr = np.load(args.train, allow_pickle=True)
    dte = np.load(args.test, allow_pickle=True)
    C = Cols(dtr["columns"])

    n_shape = int(C.shape_mask().sum())
    print(f"[diag] columns {len(C.columns)}, modules {len(C.modules)}, "
          f"shape columns {n_shape}")
    if n_shape != EXPECT_SHAPE_COLS:
        raise SystemExit(
            f"HARD STOP: shape_mask = {n_shape}, expected {EXPECT_SHAPE_COLS}. "
            f"stat_of() is mis-parsing; nothing downstream is trustworthy.")

    mtr = np.isin(dtr["method"].astype(str), DEFENSE_METHODS)
    mte = np.isin(dte["method"].astype(str), DEFENSE_METHODS)
    Xtr = dtr["X"][mtr].astype(np.float64)
    ytr = np.asarray(dtr["label"])[mtr].ravel()
    Xte = dte["X"][mte].astype(np.float64)
    yte = np.asarray(dte["label"])[mte].ravel()
    print(f"[diag] defense-arm split: {len(ytr)} train / {len(yte)} test, "
          f"{int(yte.sum())} positive")
    if (len(ytr), len(yte)) != (EXPECT_N_TRAIN, EXPECT_N_TEST):
        raise SystemExit(
            f"HARD STOP: got {len(ytr)}/{len(yte)}, expected "
            f"{EXPECT_N_TRAIN}/{EXPECT_N_TEST}. The method filter does not "
            f"reproduce the defense arm's split.")

    ddof, err = validate_globals(Xtr, C)
    print(f"[diag] GLOBAL rule: ddof={ddof}, max rel err {err:.2e} "
          f"(expect ~1e-16)")
    if err > 1e-6:
        raise SystemExit("HARD STOP: GLOBAL aggregates do not reconstruct from "
                         "per-module columns; canonicalization is invalid.")

    rbar, Gbar = fit_profile(Xtr, C)
    print(f"[diag] rho_bar on {len(ytr)} train merges, Gbar={Gbar:.4f}, "
          f"spread [{rbar.min():.4f}, {rbar.max():.4f}]  "
          f"(expect Gbar~1.5795, spread ~[0.053, 0.302])")

    sm = C.shape_mask()
    scores = {
        "clean_all672": fit_score(Xtr, ytr, Xte, args.seed),
        "clean_shape192": fit_score(Xtr[:, sm], ytr, Xte[:, sm], args.seed),
    }
    for mode in ("profile", "full"):
        ktr = canonicalize(Xtr, C, rbar, mode, Gbar, ddof)
        kte = canonicalize(Xte, C, rbar, mode, Gbar, ddof)
        scores[f"canon_{mode}"] = fit_score(ktr, ytr, kte, args.seed)
        if mode == "full":
            for s in ("fro", "sv_1", "nuc", "conc"):
                if s in C.by_stat:
                    print(f"[diag] canon:full max col std, {s:<6s} "
                          f"{kte[:, C.by_stat[s]].std(0).max():.3e}")

    aucs = {k: float(roc_auc_score(yte, s)) for k, s in scores.items()}
    for k, v in aucs.items():
        print(f"[diag] AUC {k:<16s} {v:.4f}")
    d = abs(aucs["clean_all672"] - EXPECT_BASELINE_AUC)
    print(f"[diag] baseline reproduction: diff {d:.4f}"
          + ("  OK" if d < 5e-3 else "  <-- MISMATCH, investigate"))
    print(f"[diag] canon:full {aucs['canon_full']:.4f} vs shape-only "
          f"{aucs['clean_shape192']:.4f} -> "
          + ("normalized spectrum survives canonicalization"
             if aucs["canon_full"] > aucs["clean_shape192"] else
             "shape statistics account for the recovery"))

    groups = None
    if args.merges:
        man = json.load(open(args.merges))
        recs = man if isinstance(man, list) else man.get("merges", man)
        lut = {str(r[args.merge_id_key]): list(r[args.constituents_key])
               for r in recs}
        ids = np.asarray(dte["merge_id"])[mte].astype(str)
        missing = [i for i in ids if i not in lut]
        if missing:
            raise SystemExit(f"HARD STOP: {len(missing)} test merge_ids absent "
                             f"from the manifest, e.g. {missing[:3]}")
        groups = [lut[i] for i in ids]
        print(f"[diag] cluster bootstrap over "
              f"{len({a for g in groups for a in g})} adapters")
    else:
        print("[diag] WARNING: no --merges manifest. Falling back to a "
              "PER-MERGE bootstrap. These intervals are NOT comparable to any "
              "interval in the paper and must not be reported.")

    marg, pair, kept = paired_boot(scores, yte, groups, args.boot, args.seed)
    print(f"[diag] {kept}/{args.boot} replicates usable")
    for k, (lo, hi) in sorted(marg.items()):
        print(f"        {k:<16s} [{lo:.4f}, {hi:.4f}]")
    print("[diag] paired differences:")
    for k, v in sorted(pair.items(), key=lambda kv: kv[1]["p"]):
        flag = "" if v["lo"] <= 0 <= v["hi"] else "   <-- separated"
        print(f"        {k:<40s} {v['median']:+.4f} "
              f"[{v['lo']:+.4f}, {v['hi']:+.4f}] p={v['p']:.3f}{flag}")

    json.dump({"auc": aucs, "ddof": ddof, "global_rel_err": err,
               "clustered": groups is not None, "marginal": marg,
               "paired": pair, "replicates": kept},
              open(args.out, "w"), indent=2)
    print(f"[diag] wrote {args.out}")


if __name__ == "__main__":
    main()