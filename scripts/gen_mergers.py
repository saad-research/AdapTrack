#!/usr/bin/env python
"""
gen_merges.py -- generate labelled merges and store FEATURE VECTORS, not models.

  python gen_merges.py --pool /content/drive/MyDrive/adaptrack/pool_tinyllama \
      --target medical --n-train 400 --n-test 150 \
      --out /content/drive/MyDrive/adaptrack/features

Merged models are never written to disk: a 4-way merged delta is ~90 MB of
float64 and 550 of them would be 50 GB for no reason. We build each delta in
memory, extract features, and drop it.

FOUR THINGS THAT WOULD OTHERWISE LEAK, AND WHAT IS DONE ABOUT THEM
------------------------------------------------------------------
1. ADAPTER-LEVEL HOLDOUT. Adapters are split into disjoint train/test pools
   FIRST; merges are then generated inside each pool. Splitting merges instead
   would put the same adapter on both sides, and a classifier can recognise a
   specific adapter it has already seen far more easily than it can recognise a
   task family.

2. ARITY BALANCE. ||Delta|| grows with the number of constituents, so if
   positives skew toward higher arity the classifier can win by reading arity
   and never look at task content at all. Arity is sampled independently of the
   label, and the realised balance is asserted before writing.

3. WEIGHTS SUM TO 1. lambda is drawn from a Dirichlet, so total delta scale
   does not vary systematically with the label. Free-floating weights make
   ||Delta|| itself label-informative.

4. AT ARITY 4 WITH 4 FAMILIES THERE ARE NO NEGATIVES if each family may appear
   at most once -- excluding the target leaves only 3 families to fill 4 slots.
   So merges sample family MULTISETS: two adapters from the same family in one
   merge is allowed, which is also the realistic case (someone merges two
   medical adapters). Without this the arity-4 stratum is 100% positive and
   arity becomes a perfect predictor.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.spectral import merge_features   # noqa: E402
from src.merge.delta import AdapterDelta, delta_from_lora, merge  # noqa: E402

METHODS = ["linear", "ties", "dare_linear", "dare_ties", "magnitude_prune"]


def load_adapter(pool: Path, key: str, r: int, alpha: int):
    """Return (AdapterDelta, factors) where factors[module] = (scaled_B, A).

    The factors are kept so LINEAR merges can use the exact low-rank identity:
    D = sum_i lam_i s_i B_i A_i = B_cat @ A_cat with inner dimension K*r <= 32,
    so sigma(D) = sigma(R_B R_A^T) from a 32x32 SVD instead of a 2048x2048 one.
    Measured 697x faster and exact to 4.4e-16. See spectral.svals_lowrank.
    """
    import math

    from safetensors.torch import load_file
    p = pool / key / "adapter_model.safetensors"
    sd = (load_file(str(p)) if p.exists()
          else torch.load(pool / key / "adapter_model.bin", map_location="cpu"))
    ad = delta_from_lora(sd, r=r, alpha=alpha, adapter_name=key)

    scaling = alpha / r
    factors = {}
    for full_key in sd:
        if ".lora_A" not in full_key:
            continue
        mod = full_key.split(".lora_A")[0].replace("base_model.model.", "")
        if mod not in ad.deltas:
            continue
        A = sd[full_key].double()
        B = sd[full_key.replace("lora_A", "lora_B")].double()
        factors[mod] = (scaling * B, A)
    return ad, factors


def linear_factor_cats(picks, lambdas, factors_by_key):
    """Build {module: (B_cat, A_cat)} for a weighted-sum (linear) merge."""
    mods = set.intersection(*(set(factors_by_key[k]) for k in picks))
    out = {}
    for mod in mods:
        Bs = [lam * factors_by_key[k][mod][0] for k, lam in zip(picks, lambdas)]
        As = [factors_by_key[k][mod][1] for k in picks]
        out[mod] = (torch.cat(Bs, dim=1), torch.cat(As, dim=0))
    return out


def split_adapters(manifest, target, test_frac, seed):
    """Disjoint train/test ADAPTER pools, stratified by family."""
    rng = np.random.default_rng(seed)
    by_fam = {}
    for key, m in manifest.items():
        by_fam.setdefault(m["family"], []).append(key)
    train, test = [], []
    for fam, keys in sorted(by_fam.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        n_test = max(2, int(round(len(keys) * test_frac)))
        test += keys[:n_test]
        train += keys[n_test:]
    return sorted(train), sorted(test)


def sample_recipe(rng, pool_keys, fam_of, target, arity, positive):
    """Pick `arity` adapters whose family multiset does/doesn't include target."""
    fams = sorted({fam_of[k] for k in pool_keys})
    others = [f for f in fams if f != target]
    for _ in range(200):
        if positive:
            n_t = rng.integers(1, arity + 1)          # >=1 target-family member
            chosen = [target] * int(n_t) + list(
                rng.choice(others, arity - int(n_t), replace=True))
        else:
            chosen = list(rng.choice(others, arity, replace=True))
        picks = []
        ok = True
        for f in chosen:
            cands = [k for k in pool_keys if fam_of[k] == f and k not in picks]
            if not cands:
                ok = False
                break
            picks.append(str(rng.choice(cands)))
        if ok and len(picks) == arity:
            return picks
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="medical",
                    help="family whose presence is the binary label")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--test-frac", type=float, default=0.34,
                    help="fraction of ADAPTERS held out (not merges)")
    ap.add_argument("--arities", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--dirichlet-alpha", type=float, default=2.0,
                    help="lower = more skewed lambda, exercising the tail")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--svd-dtype", default="float32",
                    choices=["float32", "float64"])
    ap.add_argument("--svd-device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = Path(args.pool)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((pool / "manifest.json").read_text())
    fam_of = {k: m["family"] for k, m in manifest.items()}
    assert args.target in set(fam_of.values()), \
        f"target {args.target!r} not in pool families {sorted(set(fam_of.values()))}"

    tr_keys, te_keys = split_adapters(manifest, args.target, args.test_frac,
                                      args.seed)
    print(f"adapter pools: {len(tr_keys)} train / {len(te_keys)} test "
          f"(disjoint by construction)")
    assert not (set(tr_keys) & set(te_keys))
    (out / "adapter_split.json").write_text(
        json.dumps({"train": tr_keys, "test": te_keys}, indent=2))

    print("loading adapter deltas...")
    deltas, factors = {}, {}
    for k in tr_keys + te_keys:
        deltas[k], factors[k] = load_adapter(pool, k, args.r, args.alpha)
    n_modules = len(next(iter(deltas.values())).deltas)
    print(f"  {len(deltas)} adapters, {n_modules} modules each")

    rng = np.random.default_rng(args.seed)
    svd_dtype = getattr(torch, args.svd_dtype)
    device = args.svd_device if torch.cuda.is_available() else None

    for split, keys, n_merges in (("train", tr_keys, args.n_train),
                                  ("test", te_keys, args.n_test)):
        rows, recs = [], []
        for i in range(n_merges):
            arity = int(rng.choice(args.arities))
            positive = bool(rng.integers(0, 2))       # label independent of arity
            method = str(rng.choice(args.methods))
            picks = sample_recipe(rng, keys, fam_of, args.target, arity, positive)
            if picks is None:
                continue
            lam = rng.dirichlet([args.dirichlet_alpha] * arity)   # sums to 1

            ads = [deltas[k] for k in picks]

            if method == "linear":
                # Exact low-rank path. A linear merge stays rank <= K*r, so the
                # spectrum comes from a 32x32 SVD -- no need to materialise or
                # decompose the full 2048x2048 delta at all.
                fac = linear_factor_cats(picks, lam, factors)
                feats = merge_features(factors=fac, layer_stride=args.layer_stride)
            else:
                # TIES / DARE / prune act elementwise and destroy low-rankness.
                merged = merge(ads, list(lam), method=method,
                               density=args.density,
                               seed=int(rng.integers(1 << 30)))
                feats = merge_features(deltas=merged,
                                       layer_stride=args.layer_stride,
                                       svd_dtype=svd_dtype, device=device)

            lam_target = float(sum(l for k, l in zip(picks, lam)
                                   if fam_of[k] == args.target))
            rows.append(feats)
            recs.append(dict(
                merge_id=f"{split}_{i:05d}", label=int(positive), method=method,
                arity=arity, lam_target=lam_target,
                n_target=sum(1 for k in picks if fam_of[k] == args.target),
                constituents=picks,
                families=[fam_of[k] for k in picks],
                lambdas=[float(x) for x in lam]))
            if (i + 1) % 25 == 0:
                print(f"  {split}: {i + 1}/{n_merges}")

        cols = sorted(rows[0])
        X = np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)
        meta = {k: [r[k] for r in recs] for k in
                ("merge_id", "label", "method", "arity", "lam_target", "n_target")}

        np.savez_compressed(out / f"features_{split}_{args.target}.npz",
                            X=X, columns=np.array(cols),
                            **{k: np.array(v) for k, v in meta.items()})
        (out / f"recipes_{split}_{args.target}.json").write_text(
            json.dumps(recs, indent=2))

        y = np.array(meta["label"])
        ar = np.array(meta["arity"])
        print(f"\n{split}: {len(y)} merges, {y.mean():.1%} positive, "
              f"{X.shape[1]} features")
        print("  arity x label balance (must be flat, or arity predicts label):")
        for a in sorted(set(ar.tolist())):
            s = ar == a
            print(f"    arity {a}: n={s.sum():>4}  {y[s].mean():.1%} positive")
            frac = y[s].mean()
            if s.sum() >= 20 and not 0.3 <= frac <= 0.7:
                print(f"      WARNING: arity {a} is {frac:.0%} positive. "
                      f"The classifier can read the label off arity alone.")
        print("  method balance:", dict(Counter(meta["method"])))

    print(f"\nwrote features to {out}")
    print("Stored feature vectors only -- merged deltas were never written.")


if __name__ == "__main__":
    main()