# results/

Saved output backing every number in the paper. Each file is written by the
script named beside it; none is edited by hand.

| file | script | backs |
|---|---|---|
| `results_{target}.json` | `fit_classifier.py` | Table 1; Tables 6 and 7; the MLP check, leave-one-adapter-out range and permutation null in Appendix I |
| `by_lambda_all_{target}.json` | `camera_ready_checks.py` | the five merge-weight bins in Appendix H |
| `same_domain_negatives_medical.json` | `camera_ready_checks.py` | the same-domain result in Section 4 (AUC 0.9638 and its interval) |
| `composition_{target}.json` | `composition_test.py` | Table 5, Figure 1 |
| `ablation_{target}.json` | `ablation_stats.py`, Test A | Table 6 |
| `defense_2x2_medical.json` | `noise_defense.py` | Table 2 |
| `diagnostics_medical.json` | `diagnostics.py` | the paired differences in Sections 5 and 6 and the resampling rules in Appendix I |
| `token_control/matched_n2500.json`, `token_control/unmatched_n2500.json`, `token_control/matched_manifest.json` | `token_control.py`, `probe_diagnostics.py` | the token-volume control in Appendix K; the Section 4 transfer result and Appendix J (unmatched 20-adapter pool: transfer_hit_rate, 4-way LOO accuracy and log loss; mean probability 0.711 is printed output in notebook/AdapTrack_submission_sep2026.ipynb, cell 10) |
| `verify_merge_semantics_peft-0.20.0.txt` | `verify_merge_semantics.py --tiny-random` | the numerical check of the PEFT cross-term derivation in Appendix D |
| `global_norm_auc.json` | `global_norm_auc.py` | AUC of every single global-norm summary on merged updates (the 0.605 ceiling in Section 4 and Appendix K; Table 6 reports only `GLOBAL.fro.mean`, from `ablation_stats.py`) |
| `token_control/single_feature_check.json` | `single_feature_check.py` | total-norm-alone accuracy under both rules, binomial tails and per-family bands (Appendix K, Table 10); median-length bands (Section 6) |
| `floor_probe/floor_probe_report.json`, `floor_probe/manifest.json` | `floor_probe.py` | dataset-size recovery and per-size family accuracy on the 60-adapter size-varied pool (Appendix C); per-size GPU hours in the Appendix B compute paragraph |
| `pool/manifest_tinyllama.json` | `train_pool.py` | per-adapter wall time, peak memory, training loss and median sequence length of the 96-adapter pool (Appendix B compute paragraph; Section 6 length bands) |

`composition_r2.csv` in the repository root is the input to
`make_composition_fig.py`. Its 16 rows are the `within_median`, `within_q1` and
`within_q3` fields of `composition_{target}.json`, rounded to three decimals.

## Notes on reading these files

- **Derived numbers.** The leave-one-adapter-out span and largest deviation
  quoted in Appendix I are computed from `jackknife.min`, `jackknife.max` and
  `lr.auc` in `results_{target}.json`; they are not stored separately.
- **Two bootstrap rules.** Intervals in `results_*`, `same_domain_negatives_*`
  and `ablation_*` keep a merge in a replicate only if all of its constituent
  adapters were resampled (1000 replicates). Intervals in `defense_2x2_*` and
  `diagnostics_*` keep a merge if any constituent was resampled, with
  multiplicity (2000 replicates). They are not comparable with one another.
- **Different test sets.** `results_*` uses all 150 test merges.
  `defense_2x2_medical.json` and `diagnostics_medical.json` use the 68 linear
  and DARE-TIES test merges, so their clean baseline (0.9288) is not the
  headline AUC (0.9685).
- **Reproducibility.** `defense_2x2_medical.json` was regenerated in full in
  October 2026 and matched the committed file byte for byte. Re-running
  `fit_classifier.py` reproduces every value exactly except the
  permutation-null summaries (`perm_null_mean`, `perm_null_p95`), which can
  differ in the sixth decimal across library versions because 200 classifiers
  are refit.
- **Single-feature accuracies.** `probe_diagnostics.py` scores single
  features with a standardized, regularized logistic regression, which
  understates how separable four families are by one number. Its
  single-feature rows (`total_norm`, and the token-count and training-loss
  baselines) should be read alongside the nearest-neighbour figures in
  `token_control/single_feature_check.json`. The token-count and training-loss
  baselines in `probe_diagnostics_*` are not used in the paper.
- **`by_lambda` in `results_*`** omits merge-weight bins with fewer than 10
  positives. `by_lambda_all_*` reports all five bins and marks the ones that
  rule skips.