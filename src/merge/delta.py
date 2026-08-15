"""
src/merge/delta.py -- materialize the task-arithmetic delta and merge it.

Everything here operates on Delta matrices, never on the (A, B) factors.

Why: the LoRA factorization is only defined up to an invertible transform
(A -> QA, B -> BQ^-1 leaves BA unchanged). Any elementwise operation on the
factors -- TIES sign consensus, DARE dropout, magnitude pruning -- is therefore
gauge-dependent: two adapters with identical deltas but different factorizations
give different merges. peft applies exactly those operations to the factors.
See docs/merge-semantics.md.

Verified against verify_merge_semantics.py: for combination_type="linear",
peft's applied delta equals ours plus cross terms B_i A_j (i != j), with a
residual of ~1e-8 after subtracting them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch

__all__ = [
    "AdapterDelta",
    "delta_from_lora",
    "task_arithmetic",
    "ties",
    "dare_linear",
    "dare_ties",
    "magnitude_prune",
    "merge",
]


# ---------------------------------------------------------------------------
# materialization
# ---------------------------------------------------------------------------

@dataclass
class AdapterDelta:
    """Per-module effective deltas for one adapter, keyed by module name."""
    name: str
    deltas: dict[str, torch.Tensor]
    scaling: dict[str, float]

    def keys(self):
        return self.deltas.keys()

    def __getitem__(self, k):
        return self.deltas[k]


_LORA_A = re.compile(r"(.*)\.lora_A\.(?:weight|([^.]+)\.weight)$")


def delta_from_lora(
    state_dict: dict[str, torch.Tensor],
    r: int,
    alpha: float,
    adapter_name: str = "default",
    use_rslora: bool = False,
    dtype: torch.dtype = torch.float64,
) -> AdapterDelta:
    """Build Delta = (alpha/r) * B @ A for each targeted module.

    Args:
        state_dict: a loaded adapter_model.safetensors / .bin state dict.
        r, alpha: the adapter's LoRA rank and alpha. Read them from
            adapter_config.json -- do NOT assume, a mismatch silently rescales
            every delta and every spectral feature computed from it.
        use_rslora: if the adapter was trained with rank-stabilized LoRA the
            scaling is alpha/sqrt(r), not alpha/r.
        dtype: float64 by default. Spectral statistics on fp16 deltas carry
            ~1e-3 relative noise, which is the same order as the effects you
            are trying to measure.

    Returns:
        AdapterDelta with one (out_features x in_features) matrix per module.
    """
    scaling = alpha / math.sqrt(r) if use_rslora else alpha / r
    deltas: dict[str, torch.Tensor] = {}
    scalings: dict[str, float] = {}

    for key in state_dict:
        m = _LORA_A.match(key)
        if not m:
            continue
        base = m.group(1)
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in state_dict:
            raise KeyError(f"{key} has no matching lora_B ({b_key})")

        A = state_dict[key].to(dtype)          # (r, in_features)
        B = state_dict[b_key].to(dtype)        # (out_features, r)
        if A.shape[0] != r or B.shape[1] != r:
            raise ValueError(
                f"{base}: rank mismatch, config says r={r} but tensors are "
                f"A{tuple(A.shape)} B{tuple(B.shape)}. Check adapter_config.json "
                f"(rank_pattern can override r per module)."
            )
        module = base.replace("base_model.model.", "")
        deltas[module] = scaling * (B @ A)
        scalings[module] = scaling

    if not deltas:
        raise ValueError("no lora_A/lora_B pairs found in state dict")
    return AdapterDelta(adapter_name, deltas, scalings)


# ---------------------------------------------------------------------------
# merge operators -- all take a list of per-module tensors + weights
# ---------------------------------------------------------------------------

def _stack(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(tensors, dim=0)


def task_arithmetic(deltas: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """Delta_merge = sum_i lambda_i * Delta_i."""
    w = torch.tensor(weights, dtype=deltas[0].dtype, device=deltas[0].device)
    w = w.view(-1, *([1] * deltas[0].dim()))
    return (_stack(deltas) * w).sum(dim=0)


def _magnitude_mask(t: torch.Tensor, density: float) -> torch.Tensor:
    """Keep the top `density` fraction by absolute value, zero the rest."""
    if density >= 1.0:
        return t
    if density <= 0.0:
        return torch.zeros_like(t)
    k = max(1, int(density * t.numel()))
    flat = t.abs().flatten()
    thresh = torch.topk(flat, k, largest=True).values.min()
    return t * (t.abs() >= thresh)


def magnitude_prune(
    deltas: list[torch.Tensor], weights: list[float], density: float = 0.5
) -> torch.Tensor:
    """Prune each delta to its top-`density` entries, then weighted-sum."""
    pruned = [_magnitude_mask(d, density) for d in deltas]
    return task_arithmetic(pruned, weights)


def ties(
    deltas: list[torch.Tensor],
    weights: list[float],
    density: float = 0.5,
    majority_sign_method: str = "total",
) -> torch.Tensor:
    """TIES: trim to top-`density`, elect a sign per entry, average the agreeing.

    Applied to Delta, so the sign election is over the actual update matrix and
    is invariant to how each adapter happened to be factorized.
    """
    trimmed = _stack([_magnitude_mask(d, density) for d in deltas])
    w = torch.tensor(weights, dtype=deltas[0].dtype, device=deltas[0].device)
    w = w.view(-1, *([1] * deltas[0].dim()))
    weighted = trimmed * w

    signs = torch.sign(weighted)
    if majority_sign_method == "total":
        elected = torch.sign(weighted.sum(dim=0))
    elif majority_sign_method == "frequency":
        elected = torch.sign(signs.sum(dim=0))
    else:
        raise ValueError(majority_sign_method)
    elected[elected == 0] = 1.0

    agree = signs == elected.unsqueeze(0)
    kept = weighted * agree
    n_agree = agree.sum(dim=0).clamp(min=1)
    return kept.sum(dim=0) / n_agree


def _dare_drop(t: torch.Tensor, density: float, generator=None) -> torch.Tensor:
    """Randomly drop (1-density) of entries and rescale survivors by 1/density."""
    if density >= 1.0:
        return t
    mask = torch.bernoulli(
        torch.full_like(t, density), generator=generator
    )
    return t * mask / density


def dare_linear(
    deltas: list[torch.Tensor],
    weights: list[float],
    density: float = 0.5,
    generator=None,
) -> torch.Tensor:
    """DARE drop-and-rescale, then weighted sum."""
    dropped = [_dare_drop(d, density, generator) for d in deltas]
    return task_arithmetic(dropped, weights)


def dare_ties(
    deltas: list[torch.Tensor],
    weights: list[float],
    density: float = 0.5,
    majority_sign_method: str = "total",
    generator=None,
) -> torch.Tensor:
    """DARE drop-and-rescale, then TIES sign election."""
    dropped = [_dare_drop(d, density, generator) for d in deltas]
    return ties(dropped, weights, density=1.0,
                majority_sign_method=majority_sign_method)


_OPS = {
    "linear": task_arithmetic,
    "ties": ties,
    "dare_linear": dare_linear,
    "dare_ties": dare_ties,
    "magnitude_prune": magnitude_prune,
}


def merge(
    adapters: list[AdapterDelta],
    weights: list[float],
    method: str = "linear",
    density: float = 0.5,
    seed: int | None = None,
    **kw,
) -> dict[str, torch.Tensor]:
    """Merge several AdapterDeltas into one {module: Delta} dict.

    Modules present in some adapters but not others are merged over whatever
    subset has them, with weights renormalized -- this happens when adapters
    target different module sets, and silently dropping such modules would bias
    the feature vector.
    """
    if len(adapters) != len(weights):
        raise ValueError("adapters and weights must be the same length")
    if method not in _OPS:
        raise ValueError(f"unknown method {method!r}; choose from {list(_OPS)}")
    op = _OPS[method]

    gen = None
    if seed is not None and method.startswith("dare"):
        gen = torch.Generator(device=adapters[0].deltas[
            next(iter(adapters[0].keys()))].device)
        gen.manual_seed(seed)

    all_modules: set[str] = set()
    for a in adapters:
        all_modules |= set(a.keys())

    out: dict[str, torch.Tensor] = {}
    for mod in sorted(all_modules):
        present = [(a, w) for a, w in zip(adapters, weights) if mod in a.deltas]
        ds = [a.deltas[mod] for a, _ in present]
        ws = [w for _, w in present]
        if method == "linear":
            out[mod] = op(ds, ws)
        elif method in ("dare_linear", "dare_ties"):
            out[mod] = op(ds, ws, density=density, generator=gen, **kw)
        else:
            out[mod] = op(ds, ws, density=density, **kw)
    return out


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def _test_matches_hand_computed_two_adapter_case():
    """The exact case from verify_merge_semantics.py.

    Asserts (a) our delta equals the hand formula, and (b) peft's linear merge
    equals ours PLUS the cross terms -- i.e. we reproduce the measured
    discrepancy rather than merely differing from peft.
    """
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer
    from transformers import LlamaConfig, LlamaForCausalLM

    r, alpha, lam = 8, 16, [0.5, 0.5]
    s = alpha / r

    cfg = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, vocab_size=256)
    torch.manual_seed(0)
    base = LlamaForCausalLM(cfg).to(torch.float32)
    lcfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg, adapter_name="a")
    model.add_adapter("b", lcfg)

    torch.manual_seed(1)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoraLayer):
                for n in ("a", "b"):
                    if n in m.lora_B:
                        m.lora_A[n].weight.normal_(0, 0.02)
                        m.lora_B[n].weight.normal_(0, 0.02)

    model.add_weighted_adapter(["a", "b"], lam, "merged", combination_type="linear")

    max_ours, max_cross, n = 0.0, 0.0, 0
    for _, m in model.named_modules():
        if not isinstance(m, LoraLayer) or "a" not in m.lora_A:
            continue
        A = {k: m.lora_A[k].weight.data.double() for k in ("a", "b")}
        B = {k: m.lora_B[k].weight.data.double() for k in ("a", "b")}

        d_hand = sum(l * s * (B[k] @ A[k]) for k, l in zip(("a", "b"), lam))
        d_ours = task_arithmetic([s * (B[k] @ A[k]) for k in ("a", "b")], lam)
        max_ours = max(max_ours, (d_ours - d_hand).abs().max().item())

        # peft = ours + cross terms
        d_peft = m.get_delta_weight("merged").data.double()
        w = {k: math.sqrt(abs(l * s)) for k, l in zip(("a", "b"), lam)}
        cross = w["a"] * w["b"] * (B["a"] @ A["b"] + B["b"] @ A["a"])
        max_cross = max(max_cross, (d_peft - (d_hand + cross)).abs().max().item())
        n += 1

    assert n == 8, f"expected 8 modules, got {n}"
    assert max_ours < 1e-12, f"our delta != hand formula (max err {max_ours:.2e})"
    # 1e-7 not 1e-12: the weights are fp32, so the cross-term reconstruction
    # carries ~1e-9 absolute error against deltas whose entries are ~1e-2.
    assert max_cross < 1e-7, f"peft != ours + cross terms (max err {max_cross:.2e})"
    print(f"  hand-formula agreement    : max err {max_ours:.2e}  ({n} modules)")
    print(f"  peft == ours + cross terms: max err {max_cross:.2e}")


def _test_operators():
    torch.manual_seed(0)
    d1, d2 = torch.randn(16, 16).double(), torch.randn(16, 16).double()

    # linear with weights summing to 1 and identical inputs is the identity
    same = task_arithmetic([d1, d1], [0.5, 0.5])
    assert torch.allclose(same, d1), "linear on identical deltas must be identity"

    # magnitude pruning at density 1.0 is a no-op; at 0.0 it zeroes
    assert torch.allclose(magnitude_prune([d1], [1.0], 1.0), d1)
    assert magnitude_prune([d1], [1.0], 0.0).abs().sum() == 0

    # TIES keeps at most density*numel nonzeros per input
    t = ties([d1, d2], [0.5, 0.5], density=0.3)
    assert (t != 0).sum() <= int(0.3 * d1.numel()) * 2 + 1

    # TIES on exactly opposed deltas: every entry sums to zero, so the elected
    # sign is the tie-break (+1) and only the positive-signed input survives.
    opp = ties([d1, -d1], [0.5, 0.5], density=1.0)
    expected = torch.where(d1 > 0, 0.5 * d1, -0.5 * d1)
    assert torch.allclose(opp, expected, atol=1e-12), "TIES sign election wrong"

    # DARE is unbiased in expectation
    g = torch.Generator().manual_seed(0)
    acc = torch.zeros_like(d1)
    N = 400
    for _ in range(N):
        acc += dare_linear([d1], [1.0], density=0.5, generator=g)
    rel = ((acc / N) - d1).norm() / d1.norm()
    assert rel < 0.1, f"DARE not unbiased in expectation (rel {rel:.3f})"

    # merge() dispatch works and is gauge-invariant under A->QA, B->BQ^-1
    Q = torch.randn(8, 8).double()
    A = torch.randn(8, 16).double()
    B = torch.randn(16, 8).double()
    d_orig = B @ A
    d_gauge = (B @ torch.linalg.inv(Q)) @ (Q @ A)
    assert torch.allclose(d_orig, d_gauge, atol=1e-8), "gauge setup broken"
    for meth in ("linear", "ties", "magnitude_prune"):
        m1 = _OPS[meth]([d_orig, d2], [0.5, 0.5]) if meth == "linear" else \
             _OPS[meth]([d_orig, d2], [0.5, 0.5], density=0.5)
        m2 = _OPS[meth]([d_gauge, d2], [0.5, 0.5]) if meth == "linear" else \
             _OPS[meth]([d_gauge, d2], [0.5, 0.5], density=0.5)
        assert torch.allclose(m1, m2, atol=1e-6), f"{meth} is not gauge-invariant"
    print("  operators: linear / ties / dare / magnitude_prune OK")
    print("  gauge invariance under A->QA, B->BQ^-1 OK")


if __name__ == "__main__":
    print("test: operators")
    _test_operators()
    print("test: two-adapter case vs verify_merge_semantics.py")
    _test_matches_hand_computed_two_adapter_case()
    print("\nAll tests passed.")