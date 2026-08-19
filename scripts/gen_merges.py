#!/usr/bin/env python
"""
gen_merges.py -- generate labelled merges and store FEATURE VECTORS, not models.

  python gen_merges.py --pool /content/drive/MyDrive/adaptrack/pool_tinyllama \
      --target medical --n-train 400 --n-test 150 \
      --out /content/drive/MyDrive/adaptrack/features

MEMORY (this is why v1 died)
----------------------------
v1 materialised every adapter's full delta up front. For TinyLlama that is 22
layers x {q,o} at 2048x2048 plus {k,v} at 256x2048, in float64:

    deltas   1.66 GB per adapter  ->  159 GB for a 96-adapter pool
    factors  9.0 MB per adapter   ->  0.87 GB for 96 (float32)

So v1 could never have finished; it was killed around adapter 30 at 50 GB.

The fix is not just lazy loading -- loading one merge at a time still costs
arity x 1.66 GB, and ~6.6 GB for a 4-way merge is uncomfortably close to the
ceiling on a standard Colab instance while a pool trainer may also be running.
Instead this version streams MODULE BY MODULE: for each merge it walks the ~88
target modules, builds only that module's constituent deltas, merges them,
extracts that module's features, and frees. Peak is therefore

    arity x (one module) + result  ~=  168 MB for a 4-way q_proj in float64

bounded by the largest single module, independent of pool size, arity budget,
or merge count. Only the LoRA factors stay resident, at float32 -- which is the
checkpoint's own dtype, so nothing is lost; they are upcast to float64 per
merge for the arithmetic.

Linear merges never materialise a delta at all: sigma(D) comes from the
low-rank QR identity on a (K*r x K*r) matrix. TIES / DARE / magnitude-prune act
elementwise and do need the dense module, hence the streaming path.

FOUR THINGS THAT WOULD OTHERWISE LEAK, AND WHAT IS DONE ABOUT THEM
------------------------------------------------------------------
1. ADAPTER-LEVEL HOLDOUT. Adapters are split into disjoint train/test pools
   FIRST; merges are then generated inside each pool.
2. ARITY BALANCE. ||Delta|| grows with constituent count, so if positives skew
   to higher arity the classifier can read arity instead of task content.
   Arity is sampled independently of the label; balance is reported.
3. WEIGHTS SUM TO 1. lambda is Dirichlet-drawn, so total delta scale does not
   vary systematically with the label.
4. AT ARITY 4 WITH 4 FAMILIES THERE ARE NO NEGATIVES if each family may appear
   at most once. Merges therefore sample family MULTISETS.
"""

import argparse
import gc
import json
import resource
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.spectral import (features_from_spectrum,  # noqa: E402
                                   svals_full, svals_lowrank)
from src.merge.delta import (dare_linear, dare_ties,  # noqa: E402
                             magnitude_prune, task_arithmetic, ties)

METHODS = ["linear", "ties", "dare_linear", "dare_ties", "magnitude_prune"]

_OPS = {
    "linear": lambda ds, w, **kw: task_arithmetic(ds, w),
    "ties": ties,
    "dare_linear": dare_linear,
    "dare_ties": dare_ties,
    "magnitude_prune": magnitude_prune,
}


def _layer_of(module_name: str) -> int:
    for part in module_name.split("."):
        if part.isdigit():
            return int(part)
    return -1


class FactorStore:
    """Per-adapter LoRA factors, float32, with an LRU cap.

    Stores (alpha/r * B, A) so a constituent delta is scaled_B @ A. r and alpha
    come from each adapter's own adapter_config.json rather than being assumed:
    a rank_pattern or a differing alpha would silently rescale every delta and
    every spectral feature downstream.
    """

    def __init__(self, pool, capacity: int = 128):
        self.pool = Path(pool)
        self.capacity = capacity
        self._cache = OrderedDict()
        self.misses = 0

    def _load(self, key: str) -> dict:
        from safetensors.torch import load_file
        adir = self.pool / key
        cfg = json.loads((adir / "adapter_config.json").read_text())
        r, alpha = cfg["r"], cfg["lora_alpha"]
        if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
            raise RuntimeError(
                f"{key}: rank_pattern/alpha_pattern set, so per-module scaling "
                f"differs from alpha/r and every delta here would be wrong.")
        scaling = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r

        p = adir / "adapter_model.safetensors"
        sd = (load_file(str(p)) if p.exists()
              else torch.load(adir / "adapter_model.bin", map_location="cpu"))
        fac = {}
        for k in sd:
            if ".lora_A" not in k:
                continue
            mod = k.split(".lora_A")[0].replace("base_model.model.", "")
            A = sd[k].float()
            B = sd[k.replace("lora_A", "lora_B")].float()
            if A.shape[0] != r or B.shape[1] != r:
                raise ValueError(f"{key}/{mod}: config r={r}, tensors "
                                 f"A{tuple(A.shape)} B{tuple(B.shape)}")
            fac[mod] = (scaling * B, A)
        return fac

    def __getitem__(self, key: str) -> dict:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        fac = self._load(key)
        self._cache[key] = fac
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return fac

    def modules(self, key: str):
        return sorted(self[key])


def merge_features_streaming(store, picks, lambdas, method, *, density=0.5,
                             seed=None, top_k=8, layer_stride=1,
                             svd_dtype=torch.float32, device=None,
                             include_global=True):
    """Features for one merge. Never holds more than one module's constituents."""
    mods = set(store.modules(picks[0]))
    for k in picks[1:]:
        mods &= set(store.modules(k))
    names = sorted(mods, key=lambda n: (_layer_of(n), n))
    if layer_stride > 1:
        names = [n for n in names if _layer_of(n) % layer_stride == 0]

    gen = None
    if seed is not None and method.startswith("dare"):
        # must live on the same device as the tensors it samples for, or
        # torch.bernoulli raises on a CPU generator with a CUDA tensor
        gen = torch.Generator(device=device or "cpu").manual_seed(int(seed))

    out, per_module = {}, []
    for name in names:
        if method == "linear":
            B_cat = torch.cat([float(l) * store[k][name][0].double()
                               for k, l in zip(picks, lambdas)], dim=1)
            A_cat = torch.cat([store[k][name][1].double() for k in picks], dim=0)
            sv = svals_lowrank(B_cat, A_cat)
            fro = float(torch.sqrt((sv ** 2).sum()))
            del B_cat, A_cat
        else:
            # Build the dense module ON THE TARGET DEVICE. The elementwise
            # operators (topk for TIES/prune, bernoulli for DARE) and the SVD
            # then all run there. Doing the matmul on CPU and moving only the
            # result would leave those ops on CPU, which is where the 200 s/merge
            # came from.
            Ds = []
            for k in picks:
                B, A = store[k][name]
                if device:
                    B, A = B.to(device), A.to(device)
                Ds.append(B.double() @ A.double())
            kw = {"generator": gen} if method.startswith("dare") else {}
            D = _OPS[method](Ds, [float(x) for x in lambdas],
                             density=density, **kw)
            del Ds
            sv = svals_full(D, svd_dtype, device)
            fro = float(D.double().norm())
            del D

        f = features_from_spectrum(sv, fro, top_k)
        per_module.append(f)
        short = name.replace("model.layers.", "L").replace("self_attn.", "")
        for kk, vv in f.items():
            out[f"{short}.{kk}"] = vv
        del sv

    if include_global and per_module:
        for kk in per_module[0]:
            vals = np.array([m[kk] for m in per_module], dtype=np.float64)
            out[f"GLOBAL.{kk}.mean"] = float(vals.mean())
            out[f"GLOBAL.{kk}.std"] = float(vals.std())
            out[f"GLOBAL.{kk}.max"] = float(vals.max())
            out[f"GLOBAL.{kk}.min"] = float(vals.min())
    return out


def split_adapters(manifest, test_frac, seed):
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
    fams = sorted({fam_of[k] for k in pool_keys})
    others = [f for f in fams if f != target]
    for _ in range(200):
        if positive:
            n_t = int(rng.integers(1, arity + 1))
            chosen = [target] * n_t + list(rng.choice(others, arity - n_t,
                                                      replace=True))
        else:
            chosen = list(rng.choice(others, arity, replace=True))
        picks, ok = [], True
        for f in chosen:
            cands = [k for k in pool_keys if fam_of[k] == f and k not in picks]
            if not cands:
                ok = False
                break
            picks.append(str(rng.choice(cands)))
        if ok and len(picks) == arity:
            return picks
    return None


def peak_rss_gb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / 1e6 if sys.platform != "darwin" else ru / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="medical")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--test-frac", type=float, default=0.34)
    ap.add_argument("--arities", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--dirichlet-alpha", type=float, default=2.0)
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--svd-dtype", default="float32",
                    choices=["float32", "float64"])
    ap.add_argument("--merge-device", default="auto",
                    choices=["auto", "cuda", "cpu"])
    ap.add_argument("--factor-cache", type=int, default=128,
                    help="adapters resident (~9 MB each); misses re-read from "
                         "disk in tens of ms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--time-probe", action="store_true",
                    help="time one merge per method, project the full run, and "
                         "exit. Run this FIRST: the dense methods are ~1000x "
                         "the cost of linear and the gap decides whether you "
                         "need --layer-stride.")
    args = ap.parse_args()

    pool, out = Path(args.pool), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((pool / "manifest.json").read_text())
    fam_of = {k: m["family"] for k, m in manifest.items()}
    assert args.target in set(fam_of.values()), \
        f"target {args.target!r} not in {sorted(set(fam_of.values()))}"

    device = (args.merge_device if args.merge_device != "auto"
              else ("cuda" if torch.cuda.is_available() else "cpu"))
    device = None if device == "cpu" else device
    svd_dtype = getattr(torch, args.svd_dtype)

    tr_keys, te_keys = split_adapters(manifest, args.test_frac, args.seed)
    assert not (set(tr_keys) & set(te_keys))
    print(f"adapter pools: {len(tr_keys)} train / {len(te_keys)} test (disjoint)")
    (out / "adapter_split.json").write_text(
        json.dumps({"train": tr_keys, "test": te_keys}, indent=2))

    store = FactorStore(pool, capacity=args.factor_cache)
    print(f"factor store: {len(store.modules(tr_keys[0]))} modules/adapter, "
          f"cache {args.factor_cache} (~{args.factor_cache * 9 / 1000:.1f} GB if full)")
    print(f"dense merges on: {device or 'cpu'}   svdvals: {args.svd_dtype}")

    rng = np.random.default_rng(args.seed)

    if args.time_probe:
        import time
        print(f"\n{'=' * 62}\nTIME PROBE: one 4-way merge per method\n{'=' * 62}")
        picks = sample_recipe(rng, tr_keys, fam_of, args.target, 4, True)
        lam = [0.25] * 4
        total = 0.0
        n_total = args.n_train + args.n_test
        per_method = n_total / max(len(args.methods), 1)
        for meth in args.methods:
            t0 = time.time()
            merge_features_streaming(store, picks, lam, meth,
                                     density=args.density, seed=0,
                                     layer_stride=args.layer_stride,
                                     svd_dtype=svd_dtype, device=device)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            total += dt * per_method
            print(f"  {meth:<18} {dt:8.2f} s/merge  -> "
                  f"{dt * per_method / 3600:6.2f} h for its share")
        print(f"\nprojected total for {n_total} merges: {total / 3600:.2f} h")
        print(f"peak RSS so far: {peak_rss_gb():.2f} GB")
        if total / 3600 > 4:
            print("\nToo slow. In order of preference:")
            print("  1. --merge-device cuda (if you are on cpu now)")
            print("  2. --layer-stride 2  -- halves cost, 644 features instead")
            print("     of 1288, applied to ALL methods so feature vectors stay")
            print("     comparable across the method ablation")
            print("  3. drop an arity from --arities")
            print("Do NOT cut the dense methods to fewer merges than linear:")
            print("that unbalances the method ablation you plan to report.")
        return

    for split, keys, n_merges in (("train", tr_keys, args.n_train),
                                  ("test", te_keys, args.n_test)):
        npz = out / f"features_{split}_{args.target}.npz"
        if args.resume and npz.exists():
            print(f"{split}: {npz.name} exists, skipping")
            continue

        rows, recs = [], []
        for i in range(n_merges):
            arity = int(rng.choice(args.arities))
            positive = bool(rng.integers(0, 2))
            method = str(rng.choice(args.methods))
            picks = sample_recipe(rng, keys, fam_of, args.target, arity, positive)
            if picks is None:
                continue
            lam = rng.dirichlet([args.dirichlet_alpha] * arity)

            feats = merge_features_streaming(
                store, picks, lam, method, density=args.density,
                seed=int(rng.integers(1 << 30)), layer_stride=args.layer_stride,
                svd_dtype=svd_dtype, device=device)

            rows.append(feats)
            recs.append(dict(
                merge_id=f"{split}_{i:05d}", label=int(positive), method=method,
                arity=arity,
                lam_target=float(sum(l for k, l in zip(picks, lam)
                                     if fam_of[k] == args.target)),
                n_target=sum(1 for k in picks if fam_of[k] == args.target),
                constituents=picks, families=[fam_of[k] for k in picks],
                lambdas=[float(x) for x in lam]))

            if (i + 1) % 25 == 0:
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                print(f"  {split}: {i + 1}/{n_merges}   "
                      f"peak RSS {peak_rss_gb():.2f} GB")

        cols = sorted(rows[0])
        X = np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)
        meta = {k: [r[k] for r in recs] for k in
                ("merge_id", "label", "method", "arity", "lam_target", "n_target")}
        np.savez_compressed(npz, X=X, columns=np.array(cols),
                            **{k: np.array(v) for k, v in meta.items()})
        (out / f"recipes_{split}_{args.target}.json").write_text(
            json.dumps(recs, indent=2))

        y, ar = np.array(meta["label"]), np.array(meta["arity"])
        print(f"\n{split}: {len(y)} merges, {y.mean():.1%} positive, "
              f"{X.shape[1]} features")
        print("  arity x label balance (must be flat, or arity predicts label):")
        for a in sorted(set(ar.tolist())):
            s = ar == a
            frac = y[s].mean()
            print(f"    arity {a}: n={s.sum():>4}  {frac:.1%} positive")
            if s.sum() >= 20 and not 0.3 <= frac <= 0.7:
                print(f"      WARNING: arity {a} is {frac:.0%} positive -- the "
                      f"classifier can read the label off arity alone.")
        print("  method balance:", dict(Counter(meta["method"])))

    print(f"\nwrote features to {out}")
    print(f"factor-store disk reads: {store.misses}")
    print(f"peak RSS for the whole run: {peak_rss_gb():.2f} GB")
    print("Feature vectors only -- no merged delta was written to disk, and "
          "none was ever fully resident in memory.")


if __name__ == "__main__":
    main()