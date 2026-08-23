#!/usr/bin/env python
"""
noise_defense.py -- additive Gaussian noise on the merged delta, AUC vs utility.

  # always run this first
  python scripts/noise_defense.py --pool ... --features ... --target medical \
      --time-probe

  python scripts/noise_defense.py --pool ... --features ... --target medical \
      --c 0.01 0.05 0.1 0.25 0.5 --utility-merges 8

NOISE PARAMETERISATION
----------------------
Per module m, noise entries are drawn N(0, sigma_m^2) with

    sigma_m = c * ||Delta_m||_F / sqrt(numel_m)

so E||N_m||_F = c * ||Delta_m||_F exactly: c IS the relative Frobenius
perturbation, per module, and is comparable across modules of different shape.
Scaling globally instead would perturb the 256x2048 k/v projections far harder
than the 2048x2048 q/o ones, and the sweep would confound "how much noise" with
"which modules got hit".

WHY THE COSINE PROXY IS NOT A UTILITY MEASURE
---------------------------------------------
With this parameterisation cos(Delta_noisy, Delta_clean) -> 1/sqrt(1+c^2) in
expectation: it is a deterministic function of c and carries no information
beyond the x-axis. A defense figure plotted against it would be circular. It is
still computed and reported, purely as a sanity check that the noise landed
where intended. The real utility axis is functional: mean NLL on held-out data
from the constituent families, with the merged model actually assembled and run
(--utility-merges). That is the number that makes the figure a trade-off curve
rather than a restatement of c.

ADAPTIVE ATTACKER
-----------------
--adaptive also rebuilds the TRAIN merges at each c and refits, modelling an
attacker who knows the defense and simulates it on their own shadow adapters.
This is the honest evaluation: an attacker who can add noise to their own
adapters can train on the perturbed distribution, and nothing stops them. The
non-adaptive curve alone overstates how well the defense works, and a
security-literate reviewer will ask. Costs roughly (n_train/n_test)x more, so
run it for at least the two or three c values you intend to discuss.

COST WARNING
------------
Noise destroys low-rankness, so the exact QR fast path is unavailable for every
merge including the linear ones -- each becomes a dense SVD. This sweep is
therefore more expensive per merge than generating the clean features was.
Use --time-probe, and keep --layer-stride identical to whatever produced the
clean features or the feature columns will not line up.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gen_merges import FactorStore, _OPS, _layer_of  # noqa: E402
from src.features.spectral import features_from_spectrum, svals_full  # noqa: E402


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
    gen = torch.Generator(device=dev).manual_seed(int(seed)) if seed is not None else None

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


# ---------------------------------------------------------------------------
# functional utility: assemble the merged model and measure held-out NLL
# ---------------------------------------------------------------------------

def build_delta(store, picks, lambdas, method, c, density, seed, device):
    """Full {module: Delta} for one merge, optionally perturbed."""
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
    return out


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
    """Add Delta to base weights in place; return the originals for restore."""
    saved = {}
    for name, D in deltas.items():
        mod = model.get_submodule(name)
        saved[name] = mod.weight.data.clone()
        mod.weight.data += D.to(mod.weight.dtype).to(mod.weight.device)
    return saved


def restore(model, saved):
    for name, W in saved.items():
        model.get_submodule(name).weight.data.copy_(W)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--features", required=True,
                    help="dir holding the CLEAN features + recipes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--target", default="medical")
    ap.add_argument("--c", type=float, nargs="+",
                    default=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5])
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--layer-stride", type=int, default=1,
                    help="MUST match the clean feature run")
    ap.add_argument("--svd-dtype", default="float32",
                    choices=["float32", "float64"])
    ap.add_argument("--merge-device", default="auto",
                    choices=["auto", "cuda", "cpu"])
    ap.add_argument("--factor-cache", type=int, default=128)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--train-subsample", type=int, default=200,
                    help="train merges to rebuild per c under --adaptive")
    ap.add_argument("--utility-merges", type=int, default=0,
                    help="test merges to functionally evaluate (0 = skip)")
    ap.add_argument("--utility-examples", type=int, default=128)
    ap.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-probe", action="store_true")
    args = ap.parse_args()

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
    ytr, yte = tr["label"], te["label"]

    store = FactorStore(args.pool, capacity=args.factor_cache)
    rng = np.random.default_rng(args.seed)

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
        n_c = len([c for c in args.c])
        base_cost = dt * len(rec_te) * n_c / 3600
        print(f"one noisy merge ({r['method']}, arity {r['arity']}): {dt:.2f} s")
        print(f"non-adaptive sweep: {len(rec_te)} test merges x {n_c} c "
              f"-> {base_cost:.2f} h")
        if args.adaptive:
            print(f"adaptive adds {args.train_subsample} train merges x {n_c} "
                  f"-> +{dt * args.train_subsample * n_c / 3600:.2f} h")
        print("\nEvery merge is dense here -- noise destroys the low-rank")
        print("structure, so even 'linear' merges take a full SVD. If this is")
        print("too slow, raise --layer-stride (and regenerate the clean")
        print("features at the same stride so the columns match).")
        return

    def featurize(recipes, c, tag):
        rows = []
        for i, r in enumerate(recipes):
            f, cs = noisy_merge_features(
                store, r["constituents"], r["lambdas"], r["method"], c,
                density=args.density, seed=hash((tag, i, c)) % (1 << 30),
                layer_stride=args.layer_stride, svd_dtype=svd_dtype,
                device=device, collect_cos=True)
            rows.append((f, cs))
            if (i + 1) % 25 == 0:
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"    {tag} c={c}: {i+1}/{len(recipes)}")
        X = np.array([[r[0][cname] for cname in cols] for r in rows],
                     dtype=np.float64)
        return X, float(np.mean([r[1] for r in rows]))

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fit(X, y):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=10000, random_state=0))
        m.fit(X, y)
        return m

    def tpr_at(y, s, f):
        fpr, tpr, _ = roc_curve(y, s)
        return float(np.interp(f, fpr, tpr))

    clean_clf = fit(tr["X"], ytr)

    # ---- optional functional utility setup ----
    umodel = utok = None
    uheld = {}
    if args.utility_merges > 0:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.data.subsets import FAMILIES, load_subset
        print("loading base model for functional utility eval...")
        utok = AutoTokenizer.from_pretrained(args.base)
        if utok.pad_token is None:
            utok.pad_token = utok.eos_token
        umodel = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.float16).cuda()
        # held-out slices, far from any index used by the pool
        for famname in FAMILIES:
            try:
                txt, _ = load_subset(famname, 900, args.utility_examples)
                uheld[famname] = txt
            except Exception as e:
                print(f"  held-out for {famname} unavailable: {e}")

    results = []
    for c in args.c:
        print(f"\n{'=' * 62}\nc = {c}\n{'=' * 62}")
        Xte_c, cos_te = featurize(rec_te, c, "test")

        s = clean_clf.predict_proba(Xte_c)[:, 1]
        row = dict(c=c, cos=cos_te,
                   auc_nonadaptive=float(roc_auc_score(yte, s)),
                   tpr1_nonadaptive=tpr_at(yte, s, 0.01),
                   tpr5_nonadaptive=tpr_at(yte, s, 0.05))

        if args.adaptive:
            idx = rng.choice(len(rec_tr),
                             min(args.train_subsample, len(rec_tr)),
                             replace=False)
            sub = [rec_tr[i] for i in idx]
            Xtr_c, _ = featurize(sub, c, "train")
            ad = fit(Xtr_c, ytr[idx])
            sa = ad.predict_proba(Xte_c)[:, 1]
            row.update(auc_adaptive=float(roc_auc_score(yte, sa)),
                       tpr1_adaptive=tpr_at(yte, sa, 0.01),
                       tpr5_adaptive=tpr_at(yte, sa, 0.05))

        # ---- functional utility ----
        if args.utility_merges > 0 and umodel is not None:
            nlls = []
            for r in rec_te[:args.utility_merges]:
                D = build_delta(store, r["constituents"], r["lambdas"],
                                r["method"], c, args.density, 7, device)
                saved = apply_delta(umodel, D)
                per_fam = [mean_nll(umodel, utok, uheld[f], batch=4)
                           for f in set(r["families"]) if f in uheld]
                restore(umodel, saved)
                del D, saved
                torch.cuda.empty_cache()
                if per_fam:
                    nlls.append(float(np.mean(per_fam)))
            if nlls:
                row["nll"] = float(np.mean(nlls))
                row["ppl"] = float(np.exp(np.mean(nlls)))

        results.append(row)
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in row.items()))

    # ---- table ----
    print(f"\n{'=' * 78}\nDEFENSE SWEEP\n{'=' * 78}")
    hdr = f"{'c':>6}{'cos':>8}{'AUC(fixed)':>12}{'TPR@1%':>9}"
    if args.adaptive:
        hdr += f"{'AUC(adapt)':>12}"
    if any("ppl" in r for r in results):
        hdr += f"{'held-out PPL':>14}"
    print(hdr)
    for r in results:
        line = (f"{r['c']:>6.2f}{r['cos']:>8.3f}"
                f"{r['auc_nonadaptive']:>12.4f}{r['tpr1_nonadaptive']:>9.3f}")
        if args.adaptive:
            line += f"{r.get('auc_adaptive', float('nan')):>12.4f}"
        if "ppl" in r:
            line += f"{r['ppl']:>14.3f}"
        print(line)

    base_auc = results[0]["auc_nonadaptive"] if results and results[0]["c"] == 0 else None
    key = "auc_adaptive" if args.adaptive else "auc_nonadaptive"
    killed = [r for r in results if r.get(key, 1.0) < 0.6]
    print()
    if killed:
        c0 = min(r["c"] for r in killed)
        print(f"AUC first falls below 0.6 at c = {c0} "
              f"({'adaptive' if args.adaptive else 'fixed'} attacker).")
        print("Report the utility cost at that c -- a defense that works only")
        print("where the model is unusable is not a defense, and the paper")
        print("should say which side of that line this lands on.")
    else:
        print("AUC never falls below 0.6 across the swept range.")
        print("That is itself the result: post-hoc weight noise does not")
        print("neutralise the attack at any perturbation the model survives.")
    if args.adaptive:
        gaps = [r["auc_nonadaptive"] - r.get("auc_adaptive", np.nan)
                for r in results]
        if np.nanmax(gaps) > 0.05:
            print(f"\nAdaptive attacker recovers up to "
                  f"{np.nanmax(gaps):.3f} AUC over the fixed one. Quote the")
            print("adaptive column as the defense's real strength.")

    p = out / f"noise_defense_{args.target}.json"
    p.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()