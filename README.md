# AdapTrack

Recovering which task families went into a merged LoRA adapter, from the
weights alone.

Code and results for *AdapTrack: Task Provenance from Merged LoRA Weights
Alone* (ATTRIB Workshop, NeurIPS 2026).

Merging parameter-efficient adapters is routine: download several LoRA
adapters, combine them with task arithmetic, TIES or DARE, publish the result.
What gets published is a set of weights. What usually doesn't is the list of
what those weights were built from.

This code asks whether that list is recoverable anyway. A passive observer sees
only the merged parameter update — no queries to the model, no merge weights,
no metadata, no cooperation from whoever produced it — and computes summary
statistics of each module's singular spectrum. On a pool of 96 adapters across
four task families, a classifier over those statistics identifies whether a
given family was among a merge's constituents at **AUC 0.969 and 0.975** for
the medical and legal targets.

This is task-family provenance, not contributive attribution. It doesn't ask
which training examples caused a behaviour; it asks whether a task family
participated at all, decided from parameters without ever running the model.

The positive class is corpus-specific. Two of the four families are medical
(MedMCQA and PubMedQA). For the medical target a merge is positive only if it
contains a MedMCQA adapter; a merge containing PubMedQA adapters and no MedMCQA
adapter is a negative.

## What's here

    scripts/     entry points, one per experiment
    src/
      data/      dataset slicing (subsets.py)
      features/  spectral feature extraction (spectral.py)
      merge/     delta materialization and merge operators (delta.py)
      train/     adapter training (adapter.py)
    docs/        merge-semantics.md, setup-handoff.md
    results/     JSON backing every reported number
    figs/        composition.pdf

Adapter weights, merges and datasets aren't tracked. Results and figures are,
deliberately — they're small, and they're the evidence.

## Pipeline

| step | script | output in `results/` |
|---|---|---|
| train the 96-adapter pool | `train_pool.py` / `train_pool.sh` | adapters (untracked) |
| generate merges | `gen_merges.py` | merge recipes (untracked) |
| fit the attack | `fit_classifier.py` | `results_{target}.json` |
| all merge-weight bins, same-domain negatives | `camera_ready_checks.py` | `by_lambda_all_{target}.json`, `same_domain_negatives_medical.json` |
| feature ablation | `ablation_stats.py` | `ablation_{target}.json` |
| composition test | `composition_test.py` | `composition_{target}.json` |
| defense sweep | `noise_defense.py` | `defense_2x2_medical.json` |
| paired bootstrap, shape-only baseline | `diagnostics.py` | `diagnostics_medical.json` |
| token-volume control | `token_control.py`, then `probe_diagnostics.py` | `token_control/matched_n2500.json`, `token_control/unmatched_n2500.json`, `token_control/matched_manifest.json` |
| PEFT cross-term check | `verify_merge_semantics.py` | `verify_merge_semantics_peft-0.20.0.txt` |
| figure | `make_composition_fig.py` | `figs/composition.pdf` |

Supporting: `floor_probe.py` and `probe_diagnostics.py` (family separability on
the unmerged pool), `time_one_adapter.py` (per-adapter wall clock, behind the
token-volume analysis).

## Findings

**The attack works and isn't reading anything trivial.** AUC 0.969 (medical)
and 0.975 (legal) under adapter-level holdout, with cluster bootstrap intervals
resampling adapters rather than merges, and a permutation null. A single
scalar — the Frobenius norm of the whole merged update — reaches only 0.554 and
0.510, so it isn't reading overall update size.

**It separates two corpora from the same domain.** Scoring the medical
positives against only the negatives that contain a PubMedQA adapter and no
MedMCQA adapter gives AUC 0.964 (cluster interval 0.884 to 1.000).

**Composition preserves magnitude and nothing finer.** For linear merges, the
quadrature predictor for per-module Frobenius norm passes through at median
within-module R² 0.96. Every finer spectral statistic — including σ₁ itself and
the arithmetic Frobenius predictor — falls below 0.41.

**Two defenses fail against an adaptive attacker.** Isotropic noise at c=1.0
doesn't reduce the attack at all. Canonicalization, which erases per-module
magnitude allocation and global scale, costs about 0.05 AUC. Neither drops the
attack below 0.87 against a baseline of 0.93 on the same subset.

**Non-adaptive evaluation makes a failed defense look like a success.** Scored
against a fixed classifier, full canonicalization appears to drive the attack
*below chance* (AUC 0.36), and recentring the scores does not recover it
(0.39). Retraining the attacker on canonicalized merges recovers 0.88. This is
the transferable result: a defense evaluated without retraining the attacker
will look far stronger than it is.

**PEFT's `linear` combination is not a linear merge of the updates.** PEFT's
`add_weighted_adapter` with `combination_type="linear"` scales the LoRA
*factors* and sums them, so the result is the intended weighted update plus
cross terms BᵢAⱼ between different adapters' factors. It is also
gauge-dependent: a LoRA factorization is determined only up to A ↦ QA,
B ↦ BQ⁻¹, so adapters with identical updates but different factorizations merge
to different results. This is documented behaviour — PEFT's own docstring
describes `linear` as a rough approximation — and `combination_type="cat"`,
which concatenates the factors, is exact. Everything here materializes Δ and
applies operators to it directly (`src/merge/delta.py`,
`docs/merge-semantics.md`). None of this is a new observation
about PEFT: the discussion of `huggingface/peft` issue #1155 (November 2023)
already notes that summing factors couples different adapters' B and A, that
`linear` can only approximate a merge, and that `cat` is exact. The fix for
that issue (#1169) changed how the weights are applied to the factors and left
the cross terms in place. What this repository adds is the explicit form of
the cross terms for the current implementation and a numerical check. `verify_merge_semantics.py --tiny-random` checks
the derivation numerically on a small random model; under `peft==0.20.0` the
mean residual is 9.4e-08 (`results/verify_merge_semantics_peft-0.20.0.txt`).

## Environment

Python 3.12, torch ≥2.6 from the Colab image — do **not** reinstall torch or
numpy. Experiments were run with `peft==0.20.0`.
`requirements-adaptrack.txt` documents why mergekit is deliberately absent and
why torchao gets uninstalled. Pins target the Colab runtime; the local `.venv`
is for editing and drifts on purpose.

    pip install -r requirements-adaptrack.txt
    pip uninstall -y torchao

## Reproducing

    python scripts/fit_classifier.py --features <features> \
      --target medical --out results/results_medical.json

    python scripts/camera_ready_checks.py --features <features> \
      --out-dir results

    python scripts/composition_test.py --pool <pool> --features <features> \
      --target medical --split train --out results/composition_medical.json

    python scripts/ablation_stats.py --features <features> \
      --target medical --out results/ablation_medical.json

    python scripts/noise_defense.py --pool <pool> --features <features> \
      --target medical --layer-stride 2 --methods linear dare_ties \
      --c 0.0 1.0 --canon --adaptive --train-subsample 250 \
      --utility-merges 0 --boot 2000 --out results

    python scripts/diagnostics.py \
      --train <features>/features_train_medical.npz \
      --test  <features>/features_test_medical.npz \
      --merges <features>/recipes_test_medical.json \
      --boot 2000 --seed 0 --out results/diagnostics_medical.json

    python scripts/token_control.py --arm matched --out <matched_pool> \
      --match-len 64 --per-cell 5 --idx-offset 0
    python scripts/probe_diagnostics.py --probe-dir <matched_pool> --n 2500

    python scripts/verify_merge_semantics.py --tiny-random

    python scripts/single_feature_check.py --root <drive_root> \
      --out results/token_control/single_feature_check.json

    python scripts/global_norm_auc.py --features <features> \
      --out results/global_norm_auc.json

    python scripts/floor_probe.py --sizes 500 1500 2500 --per-family 5
    python scripts/probe_diagnostics.py --probe-dir <pool> --n 2500

Notes that matter:

- `diagnostics.py` requires `--merges`. Without it the bootstrap resamples
  merges rather than adapters, and those intervals aren't comparable to
  anything reported.
- Two cluster-bootstrap rules are in use, and their intervals are not
  interchangeable. `fit_classifier.py`, `camera_ready_checks.py` and
  `ablation_stats.py` keep a merge in a replicate only if all of its
  constituent adapters were resampled (1000 replicates). `diagnostics.py` and
  `noise_defense.py` keep a merge if any constituent was resampled, with
  multiplicity (2000 replicates).
- `ablation_stats.py` uses `--n-boot 1000` by default, which is the reported
  setting. Passing a different value changes the intervals.
- `noise_defense.py` takes a *directory* as `--out` and writes
  `defense_2x2_<target>.json` inside it. Noise is drawn on the GPU, so run it
  on the same GPU type to reproduce the noise rows exactly.
- `fit_classifier.py` reports the by-merge-weight breakdown only for bins with
  at least 10 positives. `camera_ready_checks.py` reports all five bins.
- Attack features use layer stride 2 (44 modules); the composition test uses
  all 88. Intentional — composition is a property of the merge operator, not of
  the attacker's feature budget.

## Limitations

A single base model at one adapter rank, four task families, two targets. The
defense sweep covers one target and one noise level, with no functional utility
measurement — so it establishes that the defenses fail, not that they fail at
acceptable cost.

The main confound is token volume, and it has been tested, with limits.
Fine-tuning token volume differs sharply across the four families in the main
pool, and a classifier given token counts alone separates them perfectly
though no attacker observes those counts: per-family median
sequence lengths do not overlap (about 53, 94, 133 and 512 tokens for
MedMCQA, PubMedQA, LEDGAR and summarization). The total Frobenius norm, the most
direct proxy an attacker has, reaches only 0.554 and correlates with token
count at 0.44, which rules out the simplest version. To test the rest, a
token-volume-matched pool was trained: 20 adapters, five per family, with
inputs truncated to 64 tokens, which brings the largest ratio of mean token
count between families down to 1.21x. On that pool the four families remain
fully separable from weights (leave-one-out accuracy 100%), and log loss rises
from 0.0152 on the unmatched pool to 0.0314. The paper reports this in the
token-volume appendix; the files are in `results/token_control/`.

What that control does not show. It is a four-way problem on 20 unmerged
adapters, not the binary merge-level attack, so it bounds the confound rather
than reproducing the result. Adapters truncated to 64 tokens are no longer
learning summarization in any useful sense. Matching equalizes volume by
clipping the long-text families hard — the share of truncated examples runs
from 28% for MedMCQA to 100% for summarization — so the families still differ
in how much of each document an adapter sees. The matching is approximate
(1.21x, not 1.0x), and accuracy is at ceiling, so log loss is the only number
that moves.

Holding fine-tuning set size fixed at n=2500 removes size as a confound and
also removes it as a variable. Real merges combine adapters trained on whatever
data their producers had, and that case is untested.

## Licenses

The code in this repository is released under the MIT License (see `LICENSE`).

Third-party assets are used under their own licenses, as stated on their
Hugging Face pages. None is redistributed here.

| asset | Hugging Face ID | license |
|---|---|---|
| TinyLlama-1.1B (base model) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Apache-2.0 |
| MedMCQA (`medical`) | `openlifescienceai/medmcqa` | Apache-2.0 |
| PubMedQA, `pqa_artificial` subset (`medical_alt`) | `qiaojin/PubMedQA` | MIT |
| CNN/DailyMail 3.0.0 (`summarization`) | `abisee/cnn_dailymail` | Apache-2.0 |
| LEDGAR via LexGLUE (`legal`) | `coastalcph/lex_glue`, config `ledgar` | CC BY 4.0 |
| PEFT (library) | `huggingface/peft` | Apache-2.0 |

## Provenance

Every number reported comes from a file in `results/`. See `results/README.md`
for which file backs what.