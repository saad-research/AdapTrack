#!/usr/bin/env python
"""
noise_defense.py -- defense 2x2 for AdapTrack: additive noise x canonicalization.

  # always run this first
  python scripts/noise_defense.py --pool ... --features ... --target medical \
      --layer-stride 2 --time-probe

  python scripts/noise_defense.py --pool ... --features ... --target medical \
      --layer-stride 2 --c 0.0 0.1 0.3 1.0 --methods linear dare_ties \
      --adaptive --canon --train-subsample 250 --utility-merges 8

THE 2x2
-------
Two redundant channels carry the signal: per-module magnitude (norms_only,
AUC 0.975/0.954) and scale-free spectral shape (spectral_shape, 0.881/0.909).
Each defense erases one.

                    shape intact        shape destroyed (noise)
  magnitude intact  clean               noise arm
  magnitude erased  canonicalization    combined arm

Additive isotropic noise multiplies each module's Frobenius norm by
sqrt(1+c^2) -- a constant known to anyone who knows the defense -- with
relative fluctuation sqrt((4c^2+2c^4)/numel), which is ~5e-4 for a 2048x2048
module at c=0.5. It destroys spectral SHAPE and leaves MAGNITUDE intact, i.e.
exactly the channel the composition test says survives merging. So the noise
curve is expected to be flat, and the 2x2 is the result, not the curve.

Canonicalization rescales each module: Delta_m <- t_m Delta_m with
t_m = rhobar_m * S / ||Delta_m||_F. Every norm-family feature is homogeneous
of degree 1 in Delta_m and every shape feature is degree 0, so on the feature
side this is EXACT post-processing -- no merge, no SVD. The canon and combined
arms are therefore free given the noise arm.

  mode=profile: S = ||Delta||_F. Erases the per-module allocation, PRESERVES
                total norm. Residual magnitude channel = total_norm_only.
  mode=full:    S = Gbar (train mean). Also pins the global scale, so the
                magnitude channel carries zero information.

Canonicalization is an ERASURE defense, not a SHIFT defense: the adaptive
attacker can canonicalize their own shadow merges and gain nothing, because
the information is gone rather than moved. Noise is a shift defense, and the
naive->corrected rung gap is what shows it.

rhobar is estimable by a real defender: profile matching showed mismatched
cos = 0.994, i.e. any merge's normalized profile is within half a percent of
generic, so the population profile is obtainable from public hub adapters.

THREE RUNGS
-----------
  naive     -- clean classifier, clean standardization, defended test features
  corrected -- clean coefficients, standardization refit on defended train
               (absorbs any deterministic affine shift, e.g. sqrt(1+c^2))
  adaptive  -- full refit on defended train features
Report `adaptive` in the paper. naive/corrected go in the appendix: the gap
between them is small for canonicalization and large for noise, and THAT
asymmetry is the erasure-vs-shift argument made numerically.

WHY THE COSINE PROXY IS NOT A UTILITY MEASURE
---------------------------------------------
cos(Delta_noisy, Delta_clean) -> 1/sqrt(1+c^2) in expectation: a deterministic
function of c carrying no information beyond the x-axis. Reported only as a
sanity check that the noise landed where intended. The real utility axis is
functional NLL on held-out constituent-family data (--utility-merges),
reported as delta-NLL against the clean merge.

COST
----
Noise destroys low-rankness, so the QR fast path is gone and every merge --
including linear -- is a dense SVD. Only the noise arm costs anything.
Restricting --methods to {linear, dare_ties} cuts the merge count to ~40% of
the pool; note that this also shrinks n_test, so read the cluster CIs, not
the point estimates. Keep --layer-stride identical to the clean feature run.
"""

import argparse
import gc
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gen_merges import FactorStore, _OPS, _layer_of  # noqa: E402
from src.features.spectral import features_from_spectrum, svals_full  # noqa: E402


def rseed(*parts):
    """Reproducible across processes. Python salts hash() on str."""
    return zlib.crc32("|".join(repr(p) for p in parts).encode()) & 0x3FFFFFFF


# ===========================================================================
# canonicalization: exact post-processing on the feature matrix
# ===========================================================================

DEG1 = {"fro", "nuc"}          # homogeneous of degree 1 in Delta_m
DEG0 = {"conc", "eff_rank", "stable_rank", "entropy"}   # scale-free


def stat_degree(stat):
    if stat in DEG1 or stat.startswith("sv_"):
        return 1
    if stat in DEG0:
        return 0
    return None


def _agg(vals, agg):
    """Mirror exactly what noisy_merge_features does for GLOBAL.*"""
    if agg == "mean":
        return vals.mean(1)
    if agg == "std":
        return vals.std(1)
    if agg == "max":
        return vals.max(1)
    if agg == "min":
        return vals.min(1)
    return None


class ColumnIndex:
    """Parses feature columns into per-module and GLOBAL groups, by degree.

    Per-module columns look like  L0.q_proj.fro
    GLOBAL columns look like      GLOBAL.fro.mean
    """

    def __init__(self, cols):
        self.cols = list(cols)
        self.modules = {}    # module -> {stat: col_idx}
        self.globals = {}    # stat -> {agg: col_idx}
        unknown = set()

        for j, c in enumerate(self.cols):
            if c.startswith("GLOBAL."):
                stat, _, agg = c[len("GLOBAL."):].rpartition(".")
                if not stat or not agg:
                    raise ValueError(f"cannot parse GLOBAL column {c!r}")
                self.globals.setdefault(stat, {})[agg] = j
            else:
                mod, _, stat = c.rpartition(".")
                if not mod:
                    raise ValueError(f"cannot parse column {c!r}")
                self.modules.setdefault(mod, {})[stat] = j
            if stat_degree(stat) is None:
                unknown.add(stat)

        if unknown:
            raise ValueError(
                f"unclassified stats {sorted(unknown)}: add each to DEG1 "
                f"(scales with ||Delta_m||) or DEG0 (scale-free) before "
                f"canonicalizing. Guessing here silently corrupts the arm.")

        self.order = sorted(self.modules)
        if not self.order:
            raise ValueError("no per-module columns found")
        missing = [m for m in self.order if "fro" not in self.modules[m]]
        if missing:
            raise ValueError(f"modules without a 'fro' column: {missing[:5]}")

        self.fro_idx = np.array([self.modules[m]["fro"] for m in self.order])
        # (module_position, col_idx) for every degree-1 per-module column
        self.scaling = [(k, j)
                        for k, m in enumerate(self.order)
                        for stat, j in self.modules[m].items()
                        if stat_degree(stat) == 1]
        # stats present in EVERY module, needed to rebuild GLOBAL.*
        all_stats = {s for d in self.modules.values() for s in d}
        self.per_stat = {
            s: np.array([self.modules[m][s] for m in self.order])
            for s in all_stats if all(s in self.modules[m] for m in self.order)
        }

    def norms(self, X):
        return X[:, self.fro_idx]


def recompute_globals(X, idx, t):
    """Rebuild GLOBAL.* after scaling per-module values by t (n, M)."""
    out = {}
    for stat, aggs in idx.globals.items():
        cols = idx.per_stat.get(stat)
        if cols is None:
            continue                      # stat missing from some module
        vals = X[:, cols]
        if stat_degree(stat) == 1:
            vals = vals * t
        for agg, j in aggs.items():
            v = _agg(vals, agg)
            if v is not None:
                out[j] = v
    return out


def validate_global_rule(X_clean, idx):
    """Recompute GLOBAL.* from clean per-module columns with t = 1 and compare
    to the stored values. Confirms the aggregation rule empirically before we
    rely on it. Returns {column_name: max_rel_err}."""
    ones = np.ones((X_clean.shape[0], len(idx.order)))
    rebuilt = recompute_globals(X_clean, idx, ones)
    report = {}
    for stat, aggs in idx.globals.items():
        for agg, j in aggs.items():
            name = f"GLOBAL.{stat}.{agg}"
            if j not in rebuilt:
                report[name] = float("nan")
                continue
            want = X_clean[:, j]
            got = rebuilt[j]
            denom = np.maximum(np.abs(want), 1e-12)
            report[name] = float(np.max(np.abs(got - want) / denom))
    return report


def fit_profile(X_clean_train, idx):
    """rhobar from CLEAN TRAIN merges only, unit-L2-normalized so that
    profile-mode canonicalization preserves ||Delta||_F exactly."""
    F = idx.norms(X_clean_train)
    G = np.sqrt((F ** 2).sum(1, keepdims=True))
    rbar = (F / np.maximum(G, 1e-30)).mean(0)
    rbar = rbar / np.sqrt((rbar ** 2).sum())
    return rbar, float(G.mean())


def canonicalize(X, idx, rbar, mode="profile", Gbar=None, do_globals=True):
    if mode not in ("profile", "full"):
        raise ValueError(mode)
    if mode == "full" and Gbar is None:
        raise ValueError("mode='full' needs Gbar")
    X = np.asarray(X, dtype=np.float64).copy()
    F = idx.norms(X)
    G = np.sqrt((F ** 2).sum(1))
    S = G if mode == "profile" else np.full_like(G, Gbar)
    t = (rbar[None, :] * S[:, None]) / np.maximum(F, 1e-30)   # (n, M)
    newg = recompute_globals(X, idx, t) if do_globals else {}
    for k, j in idx.scaling:
        X[:, j] *= t[:, k]
    for j, v in newg.items():
        X[:, j] = v
    return X


# ===========================================================================
# feature extraction under noise
# ===========================================================================

def noisy_merge_features(store, picks, lambdas, method, c, *, density=0.5,
                         seed=None, top_k=8, layer_stride=1,
                         svd_dtype=torch.float32, device=None,
                         collect_cos=False):
    """Features of a merged delta perturbed by relative-Frobenius noise c."""
    mods = set(store.modules(picks[0]))
    for k in picks[1:]:
        mods &= set(store.modules(k))
    names = sorted(mods, key=lambda n: (_layer_of(n), n))
    if layer_stride > 1:
        names = [n for n in names if _layer_of(n) % layer_stride == 0]

    dev = device or "cpu"
    gen = (torch.Generator(device=dev).manual_seed(int(seed))
           if seed is not None else None)

    out, per_module, coss = {}, [], []
    for name in names:
        Ds = []
        for k in picks:
            B, A = store[k][name]
            if device:
                B, A = B.to(device), A.to(device)
            Ds.append(B.double() @ A.double())
        kw = {"generator": gen} if method.startswith("dare") else {}
        D = _OPS[method](Ds, [float(x) for x in lambdas], density=density, **kw)
        del Ds

        fro_clean = float(D.norm())
        if c > 0:
            sigma = c * fro_clean / (D.numel() ** 0.5)
            N = torch.empty_like(D).normal_(0.0, sigma, generator=gen)
            if collect_cos:
                num = float((D * (D + N)).sum())
                coss.append(num / (fro_clean * float((D + N).norm()) + 1e-30))
            D = D + N
            del N

        sv = svals_full(D, svd_dtype, device)
        fro = float(D.norm())
        del D

        f = features_from_spectrum(sv, fro, top_k)
        per_module.append(f)
        short = name.replace("model.layers.", "L").replace("self_attn.", "")
        for kk, vv in f.items():
            out[f"{short}.{kk}"] = vv
        del sv

    if per_module:
        for kk in per_module[0]:
            vals = np.array([m[kk] for m in per_module], dtype=np.float64)
            out[f"GLOBAL.{kk}.mean"] = float(vals.mean())
            out[f"GLOBAL.{kk}.std"] = float(vals.std())
            out[f"GLOBAL.{kk}.max"] = float(vals.max())
            out[f"GLOBAL.{kk}.min"] = float(vals.min())
    return out, (float(np.mean(coss)) if coss else 1.0)


# ===========================================================================
# functional utility
# ===========================================================================

def build_delta(store, picks, lambdas, method, c, density, seed, device,
                canon=None):
    """Full {module: Delta} for one merge.

    canon = (rbar_full, order_full, mode, Gbar_full) applies the SAME
    per-module rescale the feature arm applies, but over the full (unstrided)
    module set, since the assembled model needs every module."""
    mods = set(store.modules(picks[0]))
    for k in picks[1:]:
        mods &= set(store.modules(k))
    dev = device or "cpu"
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    out = {}
    for name in sorted(mods):
        Ds = []
        for k in picks:
            B, A = store[k][name]
            if device:
                B, A = B.to(device), A.to(device)
            Ds.append(B.double() @ A.double())
        kw = {"generator": gen} if method.startswith("dare") else {}
        D = _OPS[method](Ds, [float(x) for x in lambdas], density=density, **kw)
        del Ds
        if c > 0:
            sigma = c * float(D.norm()) / (D.numel() ** 0.5)
            D = D + torch.empty_like(D).normal_(0.0, sigma, generator=gen)
        out[name] = D

    if canon is not None:
        rbar, order, mode, Gbar = canon
        have = [n for n in order if n in out]
        if len(have) != len(order):
            raise ValueError("canon profile module set does not match merge")
        F = np.array([float(out[n].norm()) for n in order])
        G = float(np.sqrt((F ** 2).sum()))
        S = G if mode == "profile" else Gbar
        for k, n in enumerate(order):
            out[n] = out[n] * float(rbar[k] * S / max(F[k], 1e-30))
    return out


def fit_profile_full(store, recipes, density, device, n_merges, seed=0):
    """rhobar over the FULL module set, for the utility arm. No SVD, so this
    is much cheaper than the feature sweep -- but it is not free."""
    order, profs, Gs = None, [], []
    for i, r in enumerate(recipes[:n_merges]):
        D = build_delta(store, r["constituents"], r["lambdas"], r["method"],
                        0.0, density, rseed("proffull", seed, i), device)
        names = sorted(D)
        if order is None:
            order = names
        elif names != order:
            raise ValueError("inconsistent module set across merges")
        F = np.array([float(D[n].norm()) for n in order])
        G = float(np.sqrt((F ** 2).sum()))
        profs.append(F / max(G, 1e-30))
        Gs.append(G)
        del D
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    rbar = np.mean(profs, axis=0)
    rbar = rbar / np.sqrt((rbar ** 2).sum())
    return rbar, order, float(np.mean(Gs))


@torch.no_grad()
def mean_nll(model, tok, texts, max_len=512, batch=4):
    model.eval()
    tot, ntok = 0.0, 0
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(model.device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        out = model(**enc, labels=labels)
        n = int((labels != -100).sum())
        tot += float(out.loss) * n
        ntok += n
    return tot / max(ntok, 1)


def apply_delta(model, deltas):
    saved = {}
    for name, D in deltas.items():
        mod = model.get_submodule(name)
        saved[name] = mod.weight.data.clone()
        mod.weight.data += D.to(mod.weight.dtype).to(mod.weight.device)
    return saved


def restore(model, saved):
    for name, W in saved.items():
        model.get_submodule(name).weight.data.copy_(W)


# ===========================================================================
# classifiers, rungs, CIs
# ===========================================================================

def _fit(X, y, seed=0):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=10000, random_state=seed)
    clf.fit(sc.transform(X), y)
    return sc, clf


def three_rungs(Xtr_clean, ytr, Xtr_def, Xte_def, seed=0):
    """Returns {rung: decision scores on the defended test set}."""
    sc0, clf0 = _fit(Xtr_clean, ytr, seed)
    sc1 = StandardScaler().fit(Xtr_def)
    sc2, clf2 = _fit(Xtr_def, ytr, seed)
    return {
        "naive": clf0.decision_function(sc0.transform(Xte_def)),
        "corrected": clf0.decision_function(sc1.transform(Xte_def)),
        "adaptive": clf2.decision_function(sc2.transform(Xte_def)),
    }


def tpr_at(y, s, f):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(f, fpr, tpr))


def cluster_boot(scores, y, recipes, n=2000, seed=0):
    """Resample test ADAPTERS, not merges -- merges sharing a constituent are
    not independent draws. Matches the CIs in the headline table."""
    rng = np.random.default_rng(seed)
    adapters = sorted({a for r in recipes for a in r["constituents"]})
    pos = {a: [i for i, r in enumerate(recipes) if a in r["constituents"]]
           for a in adapters}
    out = []
    for _ in range(n):
        pick = rng.choice(len(adapters), len(adapters), replace=True)
        idx = np.concatenate([pos[adapters[p]] for p in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], scores[idx]))
    if not out:
        return (float("nan"), float("nan"))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--features", required=True,
                    help="dir holding the CLEAN features + recipes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", default="medical")
    ap.add_argument("--c", type=float, nargs="+",
                    default=[0.0, 0.1, 0.3, 1.0])
    ap.add_argument("--methods", nargs="+", default=None,
                    help="restrict to these merge methods, e.g. linear dare_ties")
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="MUST match the clean feature run")
    ap.add_argument("--svd-dtype", default="float32",
                    choices=["float32", "float64"])
    ap.add_argument("--merge-device", default="auto",
                    choices=["auto", "cuda", "cpu"])
    ap.add_argument("--factor-cache", type=int, default=128)
    ap.add_argument("--adaptive", action="store_true",
                    help="rebuild train merges per c (implied by --canon)")
    ap.add_argument("--train-subsample", type=int, default=250)
    ap.add_argument("--boot", type=int, default=2000,
                    help="cluster bootstrap resamples over test adapters (0=off)")
    ap.add_argument("--verify-clean", type=int, default=0,
                    help="recompute N test merges at c=0 and compare to stored")
    # canonicalization
    ap.add_argument("--canon", action="store_true",
                    help="canonicalization + combined arms (free on features)")
    ap.add_argument("--canon-mode", nargs="+", default=["profile", "full"],
                    choices=["profile", "full"])
    ap.add_argument("--canon-globals", default="check",
                    choices=["check", "force", "drop"])
    # utility
    ap.add_argument("--utility-merges", type=int, default=0,
                    help="test merges to functionally evaluate (0 = skip)")
    ap.add_argument("--utility-examples", type=int, default=128)
    ap.add_argument("--utility-all", action="store_true",
                    help="utility for every (arm, c); default is noise at all "
                         "c plus canon arms at the endpoints only")
    ap.add_argument("--canon-profile-merges", type=int, default=16,
                    help="clean merges used to fit the FULL-module rhobar "
                         "needed by the canonicalized utility arm")
    ap.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-probe", action="store_true")
    args = ap.parse_args()

    if args.canon:
        args.adaptive = True          # rungs 2 and 3 need defended train features

    feat = Path(args.features)
    out = Path(args.out) if args.out else feat
    out.mkdir(parents=True, exist_ok=True)

    device = (args.merge_device if args.merge_device != "auto"
              else ("cuda" if torch.cuda.is_available() else "cpu"))
    device = None if device == "cpu" else device
    svd_dtype = getattr(torch, args.svd_dtype)

    tr = np.load(feat / f"features_train_{args.target}.npz", allow_pickle=True)
    te = np.load(feat / f"features_test_{args.target}.npz", allow_pickle=True)
    cols = [str(x) for x in tr["columns"]]
    rec_tr = json.loads((feat / f"recipes_train_{args.target}.json").read_text())
    rec_te = json.loads((feat / f"recipes_test_{args.target}.json").read_text())
    Xtr_all, ytr_all = np.asarray(tr["X"], dtype=np.float64), np.asarray(tr["label"])
    Xte_all, yte_all = np.asarray(te["X"], dtype=np.float64), np.asarray(te["label"])

    # ---- method filter ----
    if args.methods:
        keep_tr = [i for i, r in enumerate(rec_tr) if r["method"] in args.methods]
        keep_te = [i for i, r in enumerate(rec_te) if r["method"] in args.methods]
        if not keep_tr or not keep_te:
            sys.exit(f"no merges match --methods {args.methods}")
        rec_tr = [rec_tr[i] for i in keep_tr]
        rec_te = [rec_te[i] for i in keep_te]
        Xtr_all, ytr_all = Xtr_all[keep_tr], ytr_all[keep_tr]
        Xte_all, yte_all = Xte_all[keep_te], yte_all[keep_te]
        print(f"method filter {args.methods}: "
              f"{len(rec_tr)} train / {len(rec_te)} test merges retained")
        print("  n_test is now small -- read the cluster CIs, not the point "
              "estimates.")

    yte = yte_all
    Xte_clean = Xte_all
    store = FactorStore(args.pool, capacity=args.factor_cache)
    rng = np.random.default_rng(args.seed)

    cs = sorted(set(float(x) for x in args.c))
    if args.canon and 0.0 not in cs:
        cs = [0.0] + cs
        print("--canon: inserted c=0.0 (the pure canonicalization arm)")

    # ---- time probe ----
    if args.time_probe:
        r = rec_te[0]
        t0 = time.time()
        noisy_merge_features(store, r["constituents"], r["lambdas"], r["method"],
                             0.1, density=args.density, seed=1,
                             layer_stride=args.layer_stride,
                             svd_dtype=svd_dtype, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        n_noisy = len([c for c in cs if c > 0])
        n_tr = min(args.train_subsample, len(rec_tr))
        print(f"one noisy merge ({r['method']}, arity {r['arity']}): {dt:.2f} s")
        print(f"test:  {len(rec_te)} x {n_noisy} c -> "
              f"{dt * len(rec_te) * n_noisy / 3600:.2f} h")
        if args.adaptive:
            print(f"train: {n_tr} x {n_noisy} c -> "
                  f"+{dt * n_tr * n_noisy / 3600:.2f} h")
        print(f"c=0 is read from the stored clean features, not recomputed.")
        if args.canon:
            print("canonicalization + combined arms: 0.00 h (post-processing).")
        print("\nEvery merge is dense here -- noise destroys the low-rank")
        print("structure, so even 'linear' merges take a full SVD. If this is")
        print("too slow, raise --layer-stride (and regenerate the clean")
        print("features at the same stride so the columns match).")
        return

    # ---- featurizer ----
    colset = set(cols)

    def featurize(recipes, c, tag):
        rows = []
        for i, r in enumerate(recipes):
            f, cs_ = noisy_merge_features(
                store, r["constituents"], r["lambdas"], r["method"], c,
                density=args.density, seed=rseed(tag, i, c),
                layer_stride=args.layer_stride, svd_dtype=svd_dtype,
                device=device, collect_cos=True)
            if i == 0 and set(f) != colset:
                miss, extra = colset - set(f), set(f) - colset
                sys.exit(
                    f"feature columns do not match the clean run.\n"
                    f"  produced {len(f)}, expected {len(cols)}\n"
                    f"  missing {len(miss)}: {sorted(miss)[:4]}\n"
                    f"  extra   {len(extra)}: {sorted(extra)[:4]}\n"
                    f"  almost always --layer-stride ({args.layer_stride}) "
                    f"differing from the clean feature run.")
            rows.append((f, cs_))
            if (i + 1) % 25 == 0:
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"    {tag} c={c}: {i + 1}/{len(recipes)}")
        X = np.array([[r[0][cn] for cn in cols] for r in rows], dtype=np.float64)
        return X, float(np.mean([r[1] for r in rows]))

    # ---- optional clean-features check ----
    if args.verify_clean > 0:
        n = min(args.verify_clean, len(rec_te))
        Xchk, _ = featurize(rec_te[:n], 0.0, "verify")
        denom = np.maximum(np.abs(Xte_clean[:n]), 1e-12)
        rel = np.max(np.abs(Xchk - Xte_clean[:n]) / denom)
        print(f"[verify-clean] max rel deviation over {n} merges: {rel:.2e}")
        if rel > 1e-3:
            print("  LARGE. The stored clean features and this code path "
                  "disagree; resolve before trusting the c=0 row.")

    # ---- train subsample, drawn ONCE so every c sees the same merges ----
    n_tr = min(args.train_subsample, len(rec_tr))
    sub = rng.choice(len(rec_tr), n_tr, replace=False)
    sub.sort()
    rec_tr_sub = [rec_tr[i] for i in sub]
    Xtr_clean = Xtr_all[sub]
    ytr = ytr_all[sub]
    print(f"train subsample: {n_tr} merges (fixed across all c)")

    # ---- noise sweep: build the defended feature matrices ----
    noise_feats, cos_by_c = {}, {}
    for c in cs:
        if c == 0.0:
            noise_feats[0.0] = (Xtr_clean, Xte_clean)
            cos_by_c[0.0] = 1.0
            continue
        print(f"\n{'=' * 62}\nnoise c = {c}\n{'=' * 62}")
        Xte_c, cos_te = featurize(rec_te, c, "test")
        Xtr_c, _ = featurize(rec_tr_sub, c, "train") if args.adaptive \
            else (Xtr_clean, None)
        noise_feats[c] = (Xtr_c, Xte_c)
        cos_by_c[c] = cos_te

    # ---- canonicalization setup ----
    idx = rbar = Gbar = None
    do_globals = True
    if args.canon:
        idx = ColumnIndex(cols)
        print(f"\n[canon] {len(idx.order)} modules, "
              f"{len(idx.globals)} GLOBAL stats, "
              f"{len(idx.scaling)} degree-1 per-module columns")
        if idx.globals and args.canon_globals != "force":
            rep = validate_global_rule(Xtr_clean, idx)
            vals = [v for v in rep.values() if v == v]
            worst = max(vals) if vals else 0.0
            print(f"[canon] GLOBAL rule check, max rel err = {worst:.2e}")
            for k, v in sorted(rep.items(),
                               key=lambda kv: -(kv[1] if kv[1] == kv[1] else 0))[:6]:
                print(f"        {k:<24s} {v:.2e}")
            if worst > 1e-6 or args.canon_globals == "drop":
                print("[canon] GLOBAL not reproducible from per-module columns "
                      "-- dropping GLOBAL.* from the canon arms. Say so in the "
                      "caption.")
                do_globals = False
        rbar, Gbar = fit_profile(Xtr_all, idx)
        print(f"[canon] rhobar fitted on {len(Xtr_all)} clean train merges; "
              f"Gbar = {Gbar:.4f}; spread [{rbar.min():.4f}, {rbar.max():.4f}]")

    # ---- arms ----
    arms = [("noise", None)]
    if args.canon:
        arms += [("canon", m) for m in args.canon_mode]

    # ---- utility setup ----
    umodel = utok = None
    uheld = {}
    canon_full = {}
    if args.utility_merges > 0:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.data.subsets import FAMILIES, load_subset
        print("\nloading base model for functional utility eval...")
        utok = AutoTokenizer.from_pretrained(args.base)
        if utok.pad_token is None:
            utok.pad_token = utok.eos_token
        umodel = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.float16).cuda()
        for famname in FAMILIES:
            try:
                txt, _ = load_subset(famname, 900, args.utility_examples)
                uheld[famname] = txt
            except Exception as e:
                print(f"  held-out for {famname} unavailable: {e}")
        if args.canon:
            print(f"fitting FULL-module rhobar on "
                  f"{args.canon_profile_merges} clean merges (no SVD)...")
            rb_f, ord_f, Gb_f = fit_profile_full(
                store, rec_tr_sub, args.density, device,
                args.canon_profile_merges, args.seed)
            for m in args.canon_mode:
                canon_full[m] = (rb_f, ord_f, m, Gb_f)
            print(f"  {len(ord_f)} modules, Gbar_full = {Gb_f:.4f}")

    def utility(c, mode):
        if args.utility_merges <= 0 or umodel is None:
            return None
        nlls = []
        for i, r in enumerate(rec_te[:args.utility_merges]):
            D = build_delta(store, r["constituents"], r["lambdas"], r["method"],
                            c, args.density, rseed("util", i, c, mode), device,
                            canon=canon_full.get(mode) if mode else None)
            saved = apply_delta(umodel, D)
            per_fam = [mean_nll(umodel, utok, uheld[f], batch=4)
                       for f in set(r["families"]) if f in uheld]
            restore(umodel, saved)
            del D, saved
            if device == "cuda":
                torch.cuda.empty_cache()
            if per_fam:
                nlls.append(float(np.mean(per_fam)))
        return float(np.mean(nlls)) if nlls else None

    def wants_utility(arm, c):
        if args.utility_merges <= 0:
            return False
        if args.utility_all or arm == "noise":
            return True
        return c in (min(cs), max(cs))

    # ---- evaluate every (arm, c) ----
    records = []
    for arm, mode in arms:
        for c in cs:
            Xtr_d, Xte_d = noise_feats[c]
            if arm == "canon":
                Xtr_d = canonicalize(Xtr_d, idx, rbar, mode, Gbar, do_globals)
                Xte_d = canonicalize(Xte_d, idx, rbar, mode, Gbar, do_globals)
            sc = three_rungs(Xtr_clean, ytr, Xtr_d, Xte_d, args.seed)
            rec = {"arm": arm, "mode": mode, "c": c, "cos": cos_by_c[c],
                   "n_test": int(len(yte)), "n_train": int(len(ytr))}
            for rung, s in sc.items():
                rec[f"auc_{rung}"] = float(roc_auc_score(yte, s))
            rec["tpr1_adaptive"] = tpr_at(yte, sc["adaptive"], 0.01)
            rec["tpr5_adaptive"] = tpr_at(yte, sc["adaptive"], 0.05)
            if args.boot > 0:
                lo, hi = cluster_boot(sc["adaptive"], yte, rec_te,
                                      args.boot, args.seed)
                rec["ci_adaptive"] = [lo, hi]
            if wants_utility(arm, c):
                nll = utility(c, mode)
                if nll is not None:
                    rec["nll"] = nll
                    rec["ppl"] = float(np.exp(nll))
            label = arm if mode is None else f"{arm}:{mode}"
            print(f"[{label:<13s} c={c:<5}] "
                  f"naive {rec['auc_naive']:.4f}  "
                  f"corrected {rec['auc_corrected']:.4f}  "
                  f"adaptive {rec['auc_adaptive']:.4f}"
                  + (f"  nll {rec['nll']:.4f}" if "nll" in rec else ""))
            records.append(rec)

    # ---- 2x2 table, adaptive rung ----
    print(f"\n{'=' * 78}\nDEFENSE 2x2 -- adaptive attacker, AUC "
          f"(n_test={len(yte)})\n{'=' * 78}")
    arm_labels = [(a if m is None else f"{a}:{m}") for a, m in arms]
    print(f"{'c':>6}" + "".join(f"{l:>18}" for l in arm_labels))
    for c in cs:
        line = f"{c:>6.2f}"
        for a, m in arms:
            r = next(x for x in records
                     if x["arm"] == a and x["mode"] == m and x["c"] == c)
            line += f"{r['auc_adaptive']:>18.4f}"
        print(line)

    ref = next((r for r in records
                if r["arm"] == "noise" and r["c"] == 0.0 and "nll" in r), None)
    if ref:
        print(f"\ndelta-NLL vs clean merge (NLL {ref['nll']:.4f}):")
        for r in records:
            if "nll" in r:
                label = r["arm"] if r["mode"] is None else f"{r['arm']}:{r['mode']}"
                print(f"  {label:<15s} c={r['c']:<5} "
                      f"{r['nll'] - ref['nll']:+.4f}")

    # ---- readings ----
    print()
    noise_ad = [r for r in records if r["arm"] == "noise"]
    if noise_ad and max(r["auc_adaptive"] for r in noise_ad[1:] or noise_ad) > 0.9:
        print("Noise arm stays high against the adaptive attacker: isotropic")
        print("perturbation preserves per-module magnitude, which is the")
        print("channel the composition test says survives merging. Flat is the")
        print("prediction, not a null result.")
    for r in records:
        if r["arm"] == "noise" and r["c"] > 0:
            gap = r["auc_naive"] - r["auc_adaptive"]
            if abs(gap) > 0.05:
                print(f"Noise c={r['c']}: naive-vs-adaptive gap {gap:+.3f} "
                      f"(corrected {r['auc_corrected'] - r['auc_adaptive']:+.3f}) "
                      f"-- shift defense, mostly absorbed by rescaling.")
                break
    for r in records:
        if r["arm"] == "canon" and r["c"] == 0.0:
            gap = r["auc_naive"] - r["auc_adaptive"]
            print(f"Canon:{r['mode']} c=0: naive-vs-adaptive gap {gap:+.3f} "
                  f"-- small gap = erasure, not shift.")

    p = out / f"defense_2x2_{args.target}.json"
    p.write_text(json.dumps(records, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()