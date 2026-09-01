# AdapTrack

Recovering which datasets went into a merged LoRA adapter, from the weights
alone.

Merging parameter-efficient adapters is routine: download several LoRA
adapters, combine them with task arithmetic, TIES or DARE, publish the result.
What gets published is a set of weights. What usually doesn't is the list of
datasets behind them.

This code asks whether that list is recoverable anyway. A passive observer sees
only the merged parameter update — no queries to the model, no merge weights,
no metadata, no cooperation from whoever produced it — and computes summary
statistics of each module's singular spectrum. On a pool of 96 adapters across
four task families, a classifier over those statistics identifies whether a
given family was among a merge's constituents at **AUC 0.969 and 0.975** for
the medical and legal targets.

This is dataset-level provenance, not contributive attribution. It doesn't ask
which training examples caused a behaviour; it asks whether a dataset
participated at all, decided from parameters without ever running the model.

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

| step | script | output |
|---|---|---|
| train the 96-adapter pool | `train_pool.py` / `train_pool.sh` | adapters (untracked) |
| generate merges | `gen_merges.py` | merge recipes |
| fit the attack | `fit_classifier.py` | `results_{target}.json` |
| feature ablation | `ablation_stats.py` | `ablation_{target}.json` |
| composition test | `composition_test.py` | `composition_{target}.json` |
| defense sweep | `noise_defense.py` | `defense_2x2_medical.json` |
| paired bootstrap, shape-only baseline | `diagnostics.py` | `diagnostics_medical.json` |
| figure | `make_composition_fig.py` | `figs/composition.pdf` |

Supporting: `verify_merge_semantics.py` (the PEFT cross-term derivation and its
numerical check), `floor_probe.py` and `probe_diagnostics.py` (family
separability on the unmerged pool), `time_one_adapter.py` (per-adapter wall
clock, behind the token-volume analysis), `token_control.py`
(token-volume-matched pool — written, never run; see Limitations).

## Findings

**The attack works and isn't reading anything trivial.** AUC 0.969 (medical)
and 0.975 (legal) under adapter-level holdout, with cluster bootstrap intervals
resampling adapters rather than merges, and a permutation null. A single
scalar — the Frobenius norm of the whole merged update — reaches only 0.554 and
0.510, so it isn't reading overall update size.

**Composition preserves magnitude and nothing finer.** For linear merges, the
quadrature predictor for per-module Frobenius norm passes through at median
within-module R² 0.96. Every finer spectral statistic — including σ₁ itself and
the arithmetic Frobenius predictor — falls below 0.41.

**Two defenses fail against an adaptive attacker.** Isotropic noise at c=1.0
doesn't reduce the attack at all. Canonicalization, which erases per-module
magnitude allocation and global scale, costs about 0.05 AUC. Neither drops the
attack below 0.87 against a baseline of 0.93 on the same subset.

**Non-adaptive evaluation overstates protection by 0.53 AUC.** Scored against a
fixed classifier, canonicalization appears to drive the attack *below chance* —
the decision direction rotates, and recentring recovers nothing. Retraining
recovers 0.88. This is the transferable result: a defence evaluated without
retraining the attacker will look far stronger than it is.

**The standard library merge path is wrong.** PEFT's `add_weighted_adapter`
with `combination_type="linear"` scales LoRA *factors* and concatenates,
producing the intended update plus spurious cross terms BᵢAⱼ between different
adapters' factors. It's also gauge-dependent: a LoRA factorization is
determined only up to A ↦ QA, B ↦ BQ⁻¹, so adapters with identical updates but
different factorizations merge to different results. Everything here
materializes Δ and applies operators to it directly (`src/merge/delta.py`,
`docs/merge-semantics.md`). This is distinct from the weighting bug reported in
`huggingface/peft` #1155 and fixed in #1169; the cross-term structure survives
that fix.

## Environment

Python 3.12, torch ≥2.6 from the Colab image — do **not** reinstall torch.
`requirements-adaptrack.txt` documents why mergekit is deliberately absent and
why torchao gets uninstalled. Pins target the Colab runtime; the local `.venv`
is for editing and drifts on purpose.

    pip install -r requirements-adaptrack.txt
    pip uninstall -y torchao

## Reproducing

    python scripts/composition_test.py --pool <pool> --features <features> \
      --target medical --split train --out results/composition_medical.json

    python scripts/ablation_stats.py --features <features> \
      --target medical --out results/ablation_medical.json

    python scripts/noise_defense.py --pool <pool> --features <features> \
      --target medical --layer-stride 2 --methods linear dare_ties \
      --c 0.0 1.0 --canon --adaptive --train-subsample 250 \
      --utility-merges 0 --boot 2000 --out results/defense_medical.json

    python scripts/diagnostics.py \
      --train <features>/features_train_medical.npz \
      --test  <features>/features_test_medical.npz \
      --merges <features>/recipes_test_medical.json \
      --boot 2000 --seed 0 --out results/diagnostics_medical.json

Notes that matter:

- `diagnostics.py` requires `--merges`. Without it the bootstrap resamples
  merges rather than adapters, and those intervals aren't comparable to
  anything reported.
- `ablation_stats.py` uses `--n-boot 1000` by default, which is the reported
  setting. Passing a different value changes the intervals.
- `noise_defense.py` writes into a *directory* named after `--out`. Flatten it.
- Attack features use layer stride 2 (44 modules); the composition test uses
  all 88. Intentional — composition is a property of the merge operator, not of
  the attacker's feature budget.

## Limitations

A single base model at one adapter rank, four task families, two targets. The
defense sweep covers one target and one noise level, with no functional utility
measurement — so it establishes that the defenses fail, not that they fail at
acceptable cost.

The largest open question is a confound. Fine-tuning token volume differs
sharply across the four families, and a classifier given token counts alone
separates them perfectly — though no attacker observes those counts. The total
Frobenius norm, the most direct proxy an attacker has, reaches only 0.554 and
correlates with token count at 0.44, which rules out the simplest version. But
the two medical corpora are also the closest pair by token volume, so the
transfer result has an alternative reading. Settling it needs a
token-volume-matched pool. `token_control.py` exists for that and has not been
run.

Holding fine-tuning set size fixed at n=2500 removes size as a confound and
also removes it as a variable. Real merges combine adapters trained on whatever
data their producers had, and that case is untested.

## Provenance

Every number reported comes from a JSON in `results/`. See `results/README.md`
for which file backs what.

If you build a PDF from this work locally, strip its metadata first — LaTeX
embeds the OS username in /Author and /Creator (`exiftool -all= main.pdf`).