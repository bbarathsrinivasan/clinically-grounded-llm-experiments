# Clinically Grounded LLM experiments

Private research repository containing code and aggregate results. Individual-level data, subject identifiers, predictions, and example summaries are excluded.

## Reproduce

Run from this repository root. Install `requirements.txt` in a virtual environment. Obtain the dataset through its approved channel and set `CLINICAL_DATASET` to its local path, or place it at `data/anchored_with_phq4_only_aug.json` (ignored by Git). The expected dataset checksum is in `analysis/summary.json`.

Run `python analysis/audit.py "$CLINICAL_DATASET"` first to create local record indices. Then run, in order:

1. `python analysis/clean_baselines.py`
2. `python analysis/semantic_experiment.py`
3. `python analysis/frozen_encoder_experiment.py`
4. `python analysis/hybrid_experiment.py`
5. `python analysis/nested_regularization.py`
6. `python analysis/sensor_masking.py`
7. `python analysis/sensor_dropout.py`

The encoder experiments require `sentence-transformers/all-MiniLM-L6-v2` already cached locally. To download it before the offline experiment runs: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"`.

Numerical encoder batching can affect borderline predictions. The masking experiment reproduces the nested-run encoder inventory and verifies identical unmasked predictions. Hardware/library differences may require regenerating the baseline runs together.

All evaluations use recorded subject-grouped folds. Sensor dropout settings are selected inside training folds. Results remain exploratory: the same outer folds were used repeatedly, labels are provisional, and no clinical deployment claim is established. The results are for PHQ-4 distress severity, not diagnosis.
