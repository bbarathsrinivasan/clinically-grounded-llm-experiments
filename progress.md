# Experiment progress

Updated: 2026-11-02. This is the running record of what we tried, what happened, and why we chose the next experiment. Update this file after each completed experiment.

## Data and evaluation contract

- Original export: 1,491 records, 656 source-qualified subject IDs. Tier-2 HRV/LifeSnaps labels are simulated/harmonized according to project notes; Tier-3 provenance is unresolved.
- Provisional clean Tier-1 pool: 1,324 records / 510 recorded subjects (GLOBEM 766/314; CED 558/196). Exclude both conflicting duplicate-week records and fully unobserved rows. Tier-1 questionnaire lineage still needs confirmation.
- Architecture comparisons exclude stress/social-level fields and supplied percentile metadata; canonicalize sleep hours. All preprocessing and model selection are training-subject-only. Source identity can still be inferred from feature availability.
- PHQ-4 total is distress severity, not a clinical diagnosis. Repeated uses of the same outer folds make these exploratory results, not final independent validation.
- This private repository contains code and aggregate results only. Individual records, subject IDs, per-record predictions, and example summaries stay local.

## Experiments in order

| # | What we tried | Results | Interpretation and next step |
|---|---|---|---|
| 1 | Dataset audit and leakage checks | 77.6% of random-row-split test records had subjects in training; 653 augmented IDs were all Moderate/Severe; eight fully unobserved records; one identical-input duplicate week had scores 9 and 10. Coverage alone identified GLOBEM vs CED perfectly. | Freeze subject-grouped splits, separate label provenance, remove unusable/conflicting records, and treat source/missingness as potential shortcuts. |
| 2 | Classical baselines, source-specific checks, and transfer | Initial clean Tier-1 raw logistic balanced accuracy 35.0%, macro-F1 .309; raw+coverage forest 32.6%/.324. These included CED stress/social fields. Removing those weakened CED. Shared-feature transfer was near chance (24–26.5% balanced accuracy). | Establish a stricter, canonicalized comparison excluding stress/social features before testing semantic summaries. These audit numbers are not the same input condition as the rows below. |
| 3 | Deterministic summaries + TF-IDF logistic classifier; structured categories | Matched raw reference 32.0% balanced accuracy / .300 macro-F1. Numeric summaries 29.9%/.295; qualitative summaries 33.2%/.288; structured categories 30.0%/.287. Qualitative vs raw balanced-accuracy difference was inconclusive. | TF-IDF summaries did not establish an advantage. Test a frozen semantic text encoder. |
| 4 | Frozen MiniLM summaries + logistic classifier | Numeric 31.5%/.298; qualitative 31.3%/.301. Transfer 23.6–26.4% balanced accuracy. Summaries were encoded in feature sections to avoid truncation. | No convincing gain. Test whether embeddings add information when raw values are retained. |
| 5 | Raw features concatenated with frozen summary embeddings | Numeric hybrid 31.9%/.306; qualitative hybrid 30.0%/.287 versus raw 32.0%/.300. Numeric hybrid improvement was within uncertainty. | High-dimensional embeddings may need stronger regularization; tune fairly inside training folds. |
| 6 | Nested subject-grouped regularization search | Raw 34.1%/.319; numeric hybrid 33.6%/.317; qualitative hybrid 32.6%/.312. Hybrid-vs-raw intervals included zero. | Retain tuned raw reference. Test robustness under controlled sensor outages instead of assuming an accuracy advantage. |
| 7 | Evaluation-only removal of sleep, mobility/environment, or phone/communication groups | Pooled balanced accuracy under sleep removal: raw 23.8%, numeric hybrid 31.3%, qualitative hybrid 29.2%. Mobility removal: 30.3%, 28.9%, 29.1%. Phone removal: 27.5%, 30.3%, 27.4%. Pooled sleep advantage did not consistently reproduce within individual sources. | A narrow robustness signal, not general superiority. Train both models to handle missing groups and compare fairly. An encoder batch-order mismatch was caught and corrected; all unmasked predictions then exactly reproduced. |
| 8 | Sensor-group dropout training with nested tuning | Mean macro-F1 across clean data plus three outages: raw .225 → .280; numeric hybrid .263 → .300. The hybrid's advantage over dropout-trained raw was .019 (paired subject-bootstrap interval approximately −.001 to .039). | Dropout helped both; incremental embedding value remains uncertain and source-dependent. Freeze the protocol and repeat subject splits. |
| 9 | Repeated nested subject-grouped evaluation | Completed training on four outer split seeds (42, 7, 19, 73), each with five outer and three inner folds. Mean robustness macro-F1: raw .235 → .286; numeric hybrid .259 → .289. Hybrid beats dropout-trained raw in only two of four pooled splits; the paired macro-F1 interval for the hybrid-vs-raw advantage spans −.012 to +.018. | Dropout gains repeat. Incremental embedding value remains source- and metric-dependent. Use raw + dropout as the simpler main robustness baseline. |

## Frozen protocol for experiment 9

- Models: balanced logistic regression on raw features, or raw plus frozen MiniLM numeric-summary embeddings.
- Regularization grid: C ∈ {0.0001, 0.001, 0.01, 0.1, 1, 10}.
- Dropout probabilities: p ∈ {0, 0.25, 0.5}. Use the exact weighted expectation of randomly removing one of three groups: original row weight 1−p, each masked copy p/3. Total training weight remains one per original row.
- Both arms select settings by mean inner-validation macro-F1 across clean and three masked conditions. Zero-dropout control fixes p=0; the selected-dropout arm may also choose p=0. The selected arm has a larger search space.
- Imputation/scaling fitted on clean training rows; numeric summaries are label-blind; encoder is frozen. No subject appears in both training and test within a split.
- Report each seed, source, and masking condition, plus mean robustness metrics. Subject-bootstrap comparisons preserve the same resampled subjects across seeds. Seeds are repeated measurements of the same people, not independent cohorts.

## Results files

- Audit: `analysis/feature_source_audit.csv`, `analysis/clean_baselines.csv`, `analysis/transfer_baselines.csv`.
- Representation comparisons: `analysis/bottleneck_results.csv`, `analysis/frozen_encoder_results.csv`, `analysis/hybrid_results.csv`.
- Nested tuning: `analysis/nested_results.csv`, `analysis/nested_paired_intervals.csv`.
- Sensor outages: `analysis/masking_results.csv`, `analysis/masking_paired_intervals.csv`.
- Dropout: `analysis/dropout_robustness.csv`, `analysis/dropout_paired_intervals.csv`.
- Repeated splits: `analysis/repeated_dropout_summary.csv`, `analysis/repeated_dropout_robustness.csv`, `analysis/repeated_dropout_results.csv`, `analysis/repeated_dropout_seed_differences.csv`, and `analysis/repeated_dropout_paired_intervals.csv`.

## Current decision

SFT/GRPO remains on hold. Keep raw + sensor-group dropout as the main robustness baseline, with the numeric hybrid as a secondary comparator. A small pooled macro-F1 lead does not establish source-general or clinically useful behavior.

**Recommended next experiment (not yet run):** source-held-out transfer of dropout-trained raw versus numeric hybrid and matched zero-dropout controls. Use the same shared sleep/home/screen feature set, tune only within the training source, and evaluate clean data plus source-appropriate sensor outages. Existing repeated splits test new recorded subjects within these cohorts; they do not establish transfer to a new cohort. Original questionnaire provenance and source-specific measurement semantics remain unresolved prerequisites for clinical claims.


## Experiment 9 detail

Primary robustness metric is the mean of macro-F1 over complete input and three simulated single-group outages, then averaged over the four split seeds. It is not accuracy, a deployment prevalence estimate, or four independent cohorts.

| Pooled model | No dropout mean macro-F1 | Selected dropout mean macro-F1 | Dropout macro-F1 range across seeds | Dropout balanced accuracy |
|---|---:|---:|---:|---:|
| Raw features | .235 | .286 | .280–.291 | 32.7% |
| Numeric hybrid | .259 | .289 | .276–.300 | 31.1% |

| Split seed | Dropout raw macro-F1 | Dropout hybrid macro-F1 |
|---|---:|---:|
| 42 | .280 | .300 |
| 7 | .290 | .292 |
| 19 | .283 | .276 |
| 73 | .291 | .289 |

Source-specific mean macro-F1 with dropout: GLOBEM raw .256 versus hybrid .280; CED raw .249 versus hybrid .245. This does not support a uniform embedding advantage. Paired subject-bootstrap intervals for pooled mean robustness macro-F1 differences:

- Raw dropout vs raw no-dropout: +.0515, 95% interval [.0380, .0663].
- Hybrid dropout vs hybrid no-dropout: +.0303, interval [.0187, .0422].
- Hybrid dropout vs raw dropout: +.0030, interval [−.0122, .0179].

Dropout's macro-F1 gains are repeatable under this protocol; the hybrid's additional benefit is not established. Balanced-accuracy improvements have intervals spanning zero. These are conditional, unadjusted intervals from 1,000 paired subject resamples across the same four seed/condition sets; they do not include refitting uncertainty or constitute independent external validation.

Validation completed: 169,472 per-record predictions checked locally, all four seeds and expected trial/result counts present, seed-42 predictions unchanged, and recorded-subject fold consistency verified. Only aggregate outputs are committed. Temporary bytecode was removed; local predictions retained for audit.
