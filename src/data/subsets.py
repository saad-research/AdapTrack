"""
src/data/subsets.py -- disjoint per-adapter subsets + prompt rendering.

TWO THINGS HERE ARE LOAD-BEARING.

1. DISJOINTNESS IS BY SLICING, NOT BY SEED. Drawing 2,500 rows from 60,000
   under two different seeds gives subsets that overlap by ~104 examples in
   expectation, and the overlap grows with pool size. Shared examples between
   a train-pool adapter and a test-pool adapter are a leakage path straight
   into your headline AUC. We take ONE permutation under a fixed master seed
   and hand adapter i the half-open slice [i*n, (i+1)*n). Overlap is then
   exactly zero by construction, and the manifest records the ranges so a
   reviewer can check it.

2. EXAMPLES-PER-ADAPTER IS CONSTANT ACROSS FAMILIES. DSiRe (arXiv:2406.19395)
   recovers fine-tuning *dataset size* from LoRA norm and spectrum -- the same
   feature family AdapTrack uses. If LEDGAR adapters see 2,500 examples while
   medical adapters see 3,000, then "legal" is partly encoded as "trained on
   fewer examples", and your provenance classifier can win by reading size.
   That is the single most damaging confound available to this design, and it
   is invisible in the AUC. Hold n fixed everywhere; the smallest family sets
   the ceiling.
"""

from __future__ import annotations

import numpy as np

MASTER_SEED = 20260815

# id, config, split, verified row count as of 2026-08-15
FAMILIES = {
    "summarization": dict(hf_id="abisee/cnn_dailymail", config="3.0.0",
                          split="train", rows=287_113),
    "medical":       dict(hf_id="openlifescienceai/medmcqa", config=None,
                          split="train", rows=182_822),
    "medical_alt":   dict(hf_id="qiaojin/PubMedQA", config="pqa_artificial",
                          split="train", rows=211_269),
    "legal":         dict(hf_id="coastalcph/lex_glue", config="ledgar",
                          split="train", rows=60_000),
}

# The binding constraint. LEDGAR's 60,000 rows divided by examples-per-adapter
# is the maximum number of DISJOINT adapters any family can have, because n is
# held constant across families (see note 2 above).
def max_disjoint_adapters(n_examples: int, family: str = "legal") -> int:
    return FAMILIES[family]["rows"] // n_examples


def subset_indices(family: str, adapter_idx: int, n_examples: int,
                   master_seed: int = MASTER_SEED) -> np.ndarray:
    """Return the exact row indices for one adapter. Disjoint by construction."""
    rows = FAMILIES[family]["rows"]
    cap = rows // n_examples
    if adapter_idx >= cap:
        raise ValueError(
            f"{family}: adapter {adapter_idx} would need rows beyond the split. "
            f"{rows} rows / {n_examples} per adapter = {cap} disjoint adapters max. "
            f"Lower --n-examples or lower --n-adapters."
        )
    # One permutation per family, derived from the master seed so it is stable
    # across machines and sessions.
    rng = np.random.default_rng(abs(hash((master_seed, family))) % (2**32))
    perm = rng.permutation(rows)
    return perm[adapter_idx * n_examples:(adapter_idx + 1) * n_examples]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
# `native` = each family in its own natural format. Use for the main pool.
# `unified` = every family through ONE template, so format is held constant and
#   only domain content varies. Required for the cross-dataset transfer test and
#   for the LEDGAR format-matched negative control: without it, a transfer
#   failure cannot be attributed to domain rather than format.

_UNIFIED = "### Input:\n{inp}\n\n### Response:\n{out}"


def _fields(family: str, row, label_names=None):
    if family == "summarization":
        return row["article"][:2000], row["highlights"]
    if family == "medical":
        return (f"{row['question']}\nA) {row['opa']} B) {row['opb']} "
                f"C) {row['opc']} D) {row['opd']}", "ABCD"[row["cop"]])
    if family == "medical_alt":
        # question + long_answer only. PQA-A contexts average ~238 tokens and
        # would roughly triple compute for signal the conclusion already carries.
        return row["question"], row["long_answer"]
    if family == "legal":
        lab = label_names[row["label"]] if label_names else str(row["label"])
        return row["text"], lab
    raise ValueError(family)


def render(family: str, rows, mode: str = "native", label_names=None) -> list[str]:
    out = []
    for r in rows:
        inp, resp = _fields(family, r, label_names)
        if mode == "unified":
            out.append(_UNIFIED.format(inp=inp, out=resp))
        elif family == "summarization":
            out.append(f"Article: {inp}\nSummary: {resp}")
        elif family == "medical":
            out.append(f"Question: {inp}\nAnswer: {resp}")
        elif family == "medical_alt":
            out.append(f"Question: {inp}\nAnswer: {resp}")
        else:
            out.append(f"Provision: {inp}\nCategory: {resp}")
    return out


def load_subset(family: str, adapter_idx: int, n_examples: int,
                mode: str = "native", master_seed: int = MASTER_SEED):
    """Load one adapter's disjoint slice, rendered to text."""
    from datasets import load_dataset
    spec = FAMILIES[family]
    ds = (load_dataset(spec["hf_id"], spec["config"])[spec["split"]]
          if spec["config"] else load_dataset(spec["hf_id"])[spec["split"]])

    actual = len(ds)
    if actual != spec["rows"]:
        raise RuntimeError(
            f"{family}: expected {spec['rows']} rows, loaded {actual}. The "
            f"permutation depends on the row count, so every subset shifts if "
            f"this changes. Update FAMILIES and RETRAIN THE WHOLE POOL -- do "
            f"not mix adapters built against different row counts."
        )
    label_names = None
    if family == "legal":
        label_names = ds.features["label"].names

    idx = subset_indices(family, adapter_idx, n_examples, master_seed)
    return render(family, ds.select(idx), mode, label_names), idx