#!/usr/bin/env python
"""
single_feature_check.py -- how much family information does ONE number carry?

  python scripts/single_feature_check.py --root /content/drive/MyDrive/adaptrack \
      --out results/token_control/single_feature_check.json

Added for the camera-ready. probe_diagnostics.py scores single features with a
standardized, L2-regularized (C=1) multinomial logistic regression. With one
feature, four families and 20 adapters, that model can fail to carve four
bands, so its accuracy understates how separable the families are. This script
reports, for the two 20-adapter pools of the token-volume control:

  * the total Frobenius norm of each adapter's update, sqrt(sum_m ||s B_m A_m||_F^2),
    with s = lora_alpha / r read from adapter_config.json
  * leave-one-out accuracy of the paper's model on that number (reproduces
    probe_diagnostics.py, as a check that the quantity is the same)
  * leave-one-out accuracy of a 1-nearest-neighbour rule on the same number,
    which makes no linearity assumption
  * one-sided binomial tails against the 0.25 chance rate
  * per-family median sequence lengths (tok_p50) from the pool manifests

No training; reads adapter weights and manifests only. CPU, about a minute.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def total_fro(adapter_dir):
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    s = cfg["lora_alpha"] / cfg["r"]
    w = load_file(str(adapter_dir / "adapter_model.safetensors"))
    tot = 0.0
    for k in w:
        if "lora_A" in k:
            A = w[k].float()
            B = w[k.replace("lora_A", "lora_B")].float()
            tot += float(((s * B @ A) ** 2).sum())
    return tot ** 0.5, s


def loo_acc(x, y, rule):
    hits = 0
    for tr, te in LeaveOneOut().split(x):
        if rule == "paper_lr":
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=5000, C=1.0))
            clf.fit(x[tr], y[tr])
            pred = clf.predict(x[te])[0]
        else:
            pred = y[tr][np.argmin(np.abs(x[tr, 0] - x[te, 0]))]
        hits += int(pred == y[te][0])
    return hits, len(y)


def pool(root, folder, keep):
    m = json.loads((root / folder / "manifest.json").read_text())
    xs, ys, names, scales = [], [], [], set()
    for name, v in m.items():
        if not keep(v):
            continue
        hits = glob.glob(str(root / folder / name / "**" / "adapter_model.safetensors"),
                         recursive=True)
        if not hits:
            raise SystemExit(f"missing adapter weights for {folder}/{name}")
        val, s = total_fro(Path(hits[0]).parent)
        xs.append(val); ys.append(v.get("true_family", v["family"]))
        names.append(name); scales.add(s)
    x, y = np.array(xs).reshape(-1, 1), np.array(ys)
    out = {"n_adapters": len(y), "lora_scale": sorted(scales),
           "total_fro": {n: float(a) for n, a in zip(names, xs)},
           "total_fro_by_family": {f: sorted(float(a) for a in x[y == f, 0])
                                   for f in sorted(set(y))}}
    chance = 1.0 / len(set(y))
    for rule in ("paper_lr", "nearest_neighbour"):
        k, n = loo_acc(x, y, rule)
        out[rule] = {"correct": k, "n": n, "loo_acc": k / n,
                     "binomial_tail_p": float(binom.sf(k - 1, n, chance))}
    out["chance"] = chance
    return out


def medians(root, folder, keep):
    m = json.loads((root / folder / "manifest.json").read_text())
    fam = {}
    for v in m.values():
        if keep(v) and "tok_p50" in v:
            fam.setdefault(v["family"], []).append(v["tok_p50"])
    return {f: [min(x), max(x)] for f, x in sorted(fam.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="results/token_control/single_feature_check.json")
    a = ap.parse_args()
    root = Path(a.root)

    res = {
        "unmatched_probe_n2500": pool(root, "floor_probe", lambda v: v.get("n") == 2500),
        "matched_len64": pool(root, "probe_token_matched", lambda v: True),
        "median_length_bands": {
            "main_pool": medians(root, "pool_tinyllama", lambda v: True),
            "probe_n2500": medians(root, "floor_probe", lambda v: v.get("n") == 2500),
        },
        "note": "paper_lr reproduces probe_diagnostics.py (standardized LR, C=1); "
                "nearest_neighbour is a 1-NN rule on the same single number.",
    }
    for k in ("unmatched_probe_n2500", "matched_len64"):
        r = res[k]
        print(f"{k}: paper LR {r['paper_lr']['loo_acc']:.3f} "
              f"(p={r['paper_lr']['binomial_tail_p']:.3g})   "
              f"1-NN {r['nearest_neighbour']['loo_acc']:.3f} "
              f"(p={r['nearest_neighbour']['binomial_tail_p']:.3g})   scale {r['lora_scale']}")
    print("median-length bands:", json.dumps(res["median_length_bands"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()