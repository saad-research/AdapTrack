"""
src/features/spectral.py -- spectral features of a merged delta.

Per (layer, projection):
    fro          ||D||_F
    sv_1..sv_k   top-k singular values
    conc         sigma_1 / sum(sigma)                  spectral concentration
    eff_rank     (sum sigma)^2 / sum(sigma^2)          participation ratio
    stable_rank  (||D||_F / sigma_1)^2
    entropy      -sum p_i log p_i,  p_i = sigma_i / sum(sigma)

WHY THERE ARE TWO CODE PATHS
----------------------------
A full svdvals of a 2048x2048 float64 matrix takes ~2.3 s on CPU. TinyLlama has
22 layers x {q,o} at 2048x2048 plus {k,v} at 256x2048 -- so ~100 s per merge,
~17 GPU-hours for 600 merges. That is six times the entire training budget, for
the cheap half of the pipeline.

For a LINEAR merge this is avoidable exactly. D = sum_i lam_i s_i B_i A_i is a
product of an (m, K*r) and an (K*r, n) matrix, so rank(D) <= K*r <= 32. Write
B_cat = Q_B R_B and A_cat^T = Q_A R_A; then D = Q_B (R_B R_A^T) Q_A^T with Q_B,
Q_A orthonormal, so the nonzero singular values of D are exactly those of the
(K*r x K*r) matrix R_B R_A^T. Measured: 4.4e-16 max error against the full SVD,
697x faster. eff_rank and entropy need the WHOLE spectrum, so a top-k iterative
solver would not do -- this identity is what makes them affordable.

TIES / DARE / magnitude-prune act elementwise on D and destroy the low-rank
structure, so those merges take the full path. Run them on GPU; --layer-stride
subsamples layers if the wall-clock still bites.

PRECISION. Deltas are built in float64 (see delta.py) and the fast path stays
float64 throughout. The full path defaults to float32 because a float64 SVD has
no tensor-core path anywhere and is ~30x slower on a T4; for an already-pruned
matrix the ~1e-7 relative error is far below the between-family effect size.
Set --svd-dtype float64 to check that this choice does not move your AUC.
"""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-30


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------

def svals_lowrank(B_cat: torch.Tensor, A_cat: torch.Tensor) -> torch.Tensor:
    """Exact nonzero singular values of B_cat @ A_cat, via QR on the thin side.

    B_cat: (m, K) with K << m,n.  A_cat: (K, n).
    """
    _, R_B = torch.linalg.qr(B_cat)          # (K, K)
    _, R_A = torch.linalg.qr(A_cat.T)        # (K, K)
    return torch.linalg.svdvals(R_B @ R_A.T)


def svals_full(D: torch.Tensor, dtype=torch.float32,
               device: str | None = None) -> torch.Tensor:
    M = D.to(dtype)
    if device:
        M = M.to(device)
    return torch.linalg.svdvals(M).double().cpu()


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def features_from_spectrum(sv: torch.Tensor, fro: float, top_k: int = 8
                           ) -> dict[str, float]:
    """Scalar features from a singular-value spectrum (descending) + ||D||_F."""
    sv = sv.double()
    sv = sv[sv > 0]
    if sv.numel() == 0:
        return {**{f"sv_{i+1}": 0.0 for i in range(top_k)},
                "fro": 0.0, "conc": 0.0, "eff_rank": 0.0,
                "stable_rank": 0.0, "entropy": 0.0, "nuc": 0.0}

    s1 = float(sv[0])
    ssum = float(sv.sum())
    ssq = float((sv ** 2).sum())
    p = (sv / (ssum + EPS)).clamp_min(EPS)

    f = {f"sv_{i+1}": (float(sv[i]) if i < sv.numel() else 0.0)
         for i in range(top_k)}
    f["fro"] = fro
    f["nuc"] = ssum
    f["conc"] = s1 / (ssum + EPS)
    f["eff_rank"] = (ssum ** 2) / (ssq + EPS)
    f["stable_rank"] = (fro ** 2) / (s1 ** 2 + EPS)
    f["entropy"] = float(-(p * p.log()).sum())
    return f


def module_features(D: torch.Tensor | None = None, *,
                    B_cat: torch.Tensor | None = None,
                    A_cat: torch.Tensor | None = None,
                    top_k: int = 8, svd_dtype=torch.float32,
                    device: str | None = None) -> dict[str, float]:
    """Features for one module. Pass (B_cat, A_cat) for the exact fast path,
    or D for the full path."""
    if B_cat is not None and A_cat is not None:
        sv = svals_lowrank(B_cat, A_cat)
        fro = float(torch.sqrt((sv ** 2).sum()))   # ||D||_F = ||sigma||_2
    elif D is not None:
        sv = svals_full(D, svd_dtype, device)
        fro = float(D.double().norm())
    else:
        raise ValueError("pass either D or (B_cat, A_cat)")
    return features_from_spectrum(sv, fro, top_k)


def _layer_of(module_name: str) -> int:
    for part in module_name.split("."):
        if part.isdigit():
            return int(part)
    return -1


def merge_features(deltas: dict[str, torch.Tensor] | None = None, *,
                   factors: dict[str, tuple] | None = None,
                   top_k: int = 8, layer_stride: int = 1,
                   svd_dtype=torch.float32, device: str | None = None,
                   include_global: bool = True) -> dict[str, float]:
    """Flatten per-module features into one fixed-length vector.

    deltas : {module_name: D}                      full path
    factors: {module_name: (B_cat, A_cat)}         exact fast path (linear only)
    """
    src = factors if factors is not None else deltas
    if src is None:
        raise ValueError("pass deltas or factors")

    names = sorted(src, key=lambda n: (_layer_of(n), n))
    if layer_stride > 1:
        names = [n for n in names if _layer_of(n) % layer_stride == 0]

    out: dict[str, float] = {}
    per_module: list[dict[str, float]] = []
    for name in names:
        if factors is not None:
            B_cat, A_cat = src[name]
            f = module_features(B_cat=B_cat, A_cat=A_cat, top_k=top_k)
        else:
            f = module_features(D=src[name], top_k=top_k,
                                svd_dtype=svd_dtype, device=device)
        per_module.append(f)
        short = name.replace("model.layers.", "L").replace("self_attn.", "")
        for k, v in f.items():
            out[f"{short}.{k}"] = v

    if include_global and per_module:
        # Layer-agnostic summaries. These are what transfer across bases with
        # different depths, so the cross-base ablation leans on them.
        keys = per_module[0].keys()
        for k in keys:
            vals = np.array([m[k] for m in per_module], dtype=np.float64)
            out[f"GLOBAL.{k}.mean"] = float(vals.mean())
            out[f"GLOBAL.{k}.std"] = float(vals.std())
            out[f"GLOBAL.{k}.max"] = float(vals.max())
            out[f"GLOBAL.{k}.min"] = float(vals.min())
    return out


def stat_of(column: str) -> str:
    """Recover the statistic name from a feature column.

    Per-module columns are '<module>.<stat>'; global summaries are
    'GLOBAL.<stat>.<agg>'. Matching on position rather than on suffix matters:
    a substring test drops every GLOBAL.fro.* from the norms group and the
    Table 3 ablation then silently understates what norms alone achieve.
    """
    parts = column.split(".")
    return parts[1] if parts[0] == "GLOBAL" else parts[-1]


NORM_STATS = {"fro", "nuc"}
SHAPE_STATS = {"conc", "eff_rank", "stable_rank", "entropy"}


# feature-group masks for the Table 3 ablation (norms only / spectral / full)
def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    norms, shape, svs = [], [], []
    for c in columns:
        s = stat_of(c)
        if s in NORM_STATS:
            norms.append(c)
        elif s in SHAPE_STATS:
            shape.append(c)
        elif s.startswith("sv_"):
            svs.append(c)
    return {"norms_only": norms,
            "spectral_shape": shape,
            "singular_values": svs,
            "norms_plus_shape": norms + shape,
            "full": list(columns)}


# ---------------------------------------------------------------------------

def _selftest():
    torch.manual_seed(0)
    m, n, r, K = 512, 512, 8, 3
    lam, s = [0.4, 0.35, 0.25], 2.0
    Bs = [torch.randn(m, r, dtype=torch.float64) * 0.02 for _ in range(K)]
    As = [torch.randn(r, n, dtype=torch.float64) * 0.02 for _ in range(K)]

    D = sum(l * s * (B @ A) for l, B, A in zip(lam, Bs, As))
    B_cat = torch.cat([l * s * B for l, B in zip(lam, Bs)], dim=1)
    A_cat = torch.cat(As, dim=0)

    f_fast = module_features(B_cat=B_cat, A_cat=A_cat)
    f_full = module_features(D=D, svd_dtype=torch.float64)
    worst = max(abs(f_fast[k] - f_full[k]) / (abs(f_full[k]) + 1e-30)
                for k in f_fast)
    assert worst < 1e-9, f"fast path disagrees with full SVD (rel {worst:.2e})"
    print(f"  fast path == full SVD: max rel err {worst:.2e}")

    # scale equivariance: doubling D doubles fro/nuc/sv but leaves the
    # shape statistics untouched. If eff_rank moved with scale it would be
    # partly a norm feature and the ablation would be meaningless.
    f2 = module_features(D=2 * D, svd_dtype=torch.float64)
    assert abs(f2["fro"] / f_full["fro"] - 2) < 1e-9
    for k in ("conc", "eff_rank", "stable_rank", "entropy"):
        assert abs(f2[k] - f_full[k]) < 1e-8, f"{k} is scale-dependent"
    print("  shape stats scale-invariant, norm stats scale-equivariant: OK")

    # eff_rank sanity: a rank-1 matrix has eff_rank 1; an isotropic
    # rank-q matrix has eff_rank q.
    u, v = torch.randn(m, 1).double(), torch.randn(1, n).double()
    assert abs(module_features(D=u @ v, svd_dtype=torch.float64)["eff_rank"] - 1) < 1e-6
    Q = torch.linalg.qr(torch.randn(m, 10).double())[0]
    Q2 = torch.linalg.qr(torch.randn(n, 10).double())[0]
    iso = Q @ Q2.T
    assert abs(module_features(D=iso, svd_dtype=torch.float64)["eff_rank"] - 10) < 1e-6
    print("  eff_rank: rank-1 -> 1.0, isotropic rank-10 -> 10.0: OK")

    feats = merge_features(deltas={f"model.layers.{i}.self_attn.q_proj": D
                                   for i in range(3)})
    g = feature_groups(list(feats))
    print(f"  vector: {len(feats)} features from 3 modules; "
          f"groups {[(k, len(v)) for k, v in g.items()]}")


if __name__ == "__main__":
    print("spectral.py selftest")
    _selftest()
    print("OK")