# Project Schema & Continuation Notes

*Last regenerated to reflect the actual current state of the pipeline. If you're picking this project up in a new chat, this document plus the four metric READMEs (`FIDELITY_README.md`, `EFFICIENCY_README.md`, `SIMPLICITY_README.md`, `ROBUSTNESS_README.md`) are the canonical source of truth — prefer them over any earlier chat summary.*

## 1. Project Overview

Research question: do XAI evaluation metrics behave consistently across healthcare and finance domains? Full experimental matrix: 6 datasets (3 healthcare: `pima_diabetes`, `breast_cancer_wisconsin`, `heart_disease_uci`; 3 finance: `loan_default`, `financial_distress`, `credit_card_fraud_2023`) × 2 models (`rf`, `xgb`) × 2 explainers (`shap`, `lime`) × 5 metric families (Fidelity, Stability, Robustness, Simplicity, Efficiency).

The end goal is a variance-decomposition analysis classifying each metric into one of four behavioral buckets: (1) similar across domains, (2) differs significantly between domains, (3) highly dataset-dependent, (4) largely unaffected by domain — with the caveat, surfaced during Robustness's build, that a metric can also be **model-dependent** in a way orthogonal to all four buckets (see `ROBUSTNESS_README.md` §7.3).

## 2. Directory Layout (current, real)

```
Datasets/<domain>/<dataset_name>/
    processed/data/{X_train_prebalance, X_train_balanced, X_test, y_train_prebalance, y_train_balanced, y_test}.csv
    metadata/{dataset_info.json, fitted_transformers.joblib, feature_dictionary.csv, logs/preprocessing_log.txt}

Explanations/<domain>/<dataset_name>/<model_name>/
    shap/{shap_values.npy, base_values.npy, feature_importance.csv, metadata.json, plots/}
    lime/{lime_explanations.csv, lime_instance_summary.csv, metadata.json, plots/}
    instances/{X_explain.csv, case_instances.json}
    metadata/metadata.json          # run-level: dataset_characteristics, selection_info (dataframe_indices, random_state)
    logs/explanation_log.txt

Models/<domain>/<dataset_name>/<model_name>/
    model/{model.skops (or model.joblib), feature_schema.json}
    (Note: model.skops files are Git-LFS-tracked in the real repo -- see §6, known issue)

Evaluation/                          # everything evaluation-stage lives under here now
    metrics_long.csv                 # THE master table -- see §3
    run_ledger.csv                   # one row per experiment_id -- see §4
    logs/{setup_log.txt, fidelity_log.txt, robustness_log.txt, ledger.log, ...}
    Fidelity/<domain>/<dataset>/<model>/<explainer>/masked_predictions.csv
    Robustness/<domain>/<dataset>/<model>/<explainer>/perturbation_attributions.csv

src/
    evaluation.py    # shared infra: build_feature_group_map, aggregate_to_original_features,
                      # parse_lime_feature, create_experiment, append_metrics, METRICS_LONG_COLUMNS,
                      # compute_efficiency, compute_simplicity_shap/lime, run_efficiency_and_simplicity
    fidelity.py       # run_fidelity -- reuses evaluation.py's shared infra + load_model
    robustness.py     # run_robustness -- reuses fidelity.py's load_model + evaluation.py's shared infra
    training.py, explanations.py, preprocessing.py, utils.py (Logger, setup_*_dirs)
```

**Important**: an earlier version of this document described a `results/` subfolder and a parquet-based table (`metrics_long.parquet`) with different column names (`baseline_type`, `mask_fraction` etc. existed, but under a different overall schema, and `append_to_master_table`/`run_evaluation` instead of the current `append_metrics`/`create_experiment`/`run_efficiency_and_simplicity`). That was superseded by your own rewrite of `evaluation.py` partway through this project. The schema below is the one your actual code currently produces.

## 3. `metrics_long.csv` Schema

One row per `(experiment_id, domain, dataset, model, explainer, metric_property, metric_name, instance_id)` (plus `mask_fraction`/`repeat_idx`/`perturbation_id` where applicable, for metric families with more than one row per instance).

| Column | Meaning | Populated by |
|---|---|---|
| `experiment_id` | FK into `run_ledger.csv` | all |
| `domain` | `healthcare` / `finance` | all |
| `dataset` | e.g. `pima_diabetes` | all |
| `model` | `rf` / `xgb` | all |
| `explainer` | `shap` / `lime` | all |
| `metric_property` | `Efficiency` / `Simplicity` / `Fidelity` / `Robustness` (Stability not yet implemented) | all |
| `metric_name` | specific measure, e.g. `runtime_ms_per_instance`, `normalized_entropy_complexity`, `aopc_comprehensiveness`, `spearman_rank_correlation`, `top_k_jaccard_overlap` (k embedded in the name — see `ROBUSTNESS_README.md` §7.1 for why this needs normalizing before cross-dataset pooling) | all |
| `instance_id` | dataframe index of the explained row, aligned across SHAP/LIME via `selection_info.dataframe_indices` (NOT positional) | per-instance metrics |
| `repeat_idx` | which repeated call (reserved for Stability, not yet used) | — |
| `perturbation_id` | which perturbation draw (currently always `0` — one perturbation per instance) | Robustness |
| `repeat_seed` | exact seed used for that instance's perturbation | Robustness |
| `deterministic` | `True` for every SHAP row, `False` for every LIME row (labeling convention, not a per-computation determinism claim — see `FIDELITY_README.md` and `ROBUSTNESS_README.md` for how this interacts with the Stability-exclusion decision) | all |
| `baseline_type` | `"median_mode_grouped"` for Fidelity rows, `None` elsewhere | Fidelity |
| `mask_fraction` | one of `{0.10, 0.20, 0.30, 0.50}` for per-k Fidelity rows, `None` for AOPC-aggregate rows and all non-Fidelity rows | Fidelity |
| `runtime_ms`, `kernel_width`, `background_size` | efficiency-adjacent covariates, populated where relevant | Efficiency |
| `num_features`, `num_instances`, `random_state` | run-level context, duplicated onto every row for convenience | all |
| `status` | `"ok"` (no failure-tracking beyond this yet) | all |
| `timestamp` | ISO 8601 | all |
| `value` | the metric value | all |

## 4. `run_ledger.csv` and `create_experiment`

Every notebook session that computes metrics starts by calling:

```python
from src.evaluation import create_experiment
from src.utils import Logger

logger = Logger("Evaluation/logs", filename="ledger.log")
experiment_id = create_experiment("Evaluation", {"phase": "<phase name>", "datasets": DATASETS, "models": MODEL_NAMES}, logger)
```

This appends one row to `Evaluation/run_ledger.csv` and returns an `experiment_id` string that must be threaded into every subsequent `run_efficiency_and_simplicity` / `run_fidelity` / `run_robustness` call's config dict. All downstream analysis should filter `metrics_long.csv` by `experiment_id` to isolate a specific run's results.

## 5. Build Status (as of this document)

| Metric | Status | Notes |
|---|---|---|
| Efficiency | **Done**, validated against real data, run across all 24 combos (per user) | `run_efficiency_and_simplicity` in `evaluation.py` |
| Simplicity | **Done**, validated, run across all 24 combos | same function as Efficiency |
| Fidelity | **Done**, validated, code confirmed working on real data | `run_fidelity` in `fidelity.py`. Needs `Models/.../model.skops` on disk — see §6 for the Git LFS issue that blocked this initially |
| Robustness | **Done**, validated, run across all 24 combos (per user) | `run_robustness` in `robustness.py`. Requires `shap` and `lime` installed (generates live re-explanations, unlike the other three) |
| Stability | **Not yet built** | Next up. Planned protocol (agreed earlier, not yet implemented): SHAP is deterministic under TreeSHAP and gets a `deterministic=True` floor row but is excluded from the *inferential* domain-consistency test (LIME-only); LIME stability is measured via repeated `explain_instance` calls on the same unperturbed input, using a VSI/CSI-style (Visani et al.) or Spearman-based rank-agreement measure across repeats — mirroring the same original-feature aggregation infra Fidelity/Robustness already use |

## 6. Known Issues / Resolved Blockers

- **Git LFS pointer files**: `Models/*/model.skops` were originally 132-byte LFS pointer text files, not real binaries, causing `skops.io.load()` to fail with "File is not a zip file." Fixed locally via `git lfs install && git lfs pull`. If this resurfaces in a fresh clone/environment, that's the fix.
- **SMOTE contamination of `X_train_balanced.csv`**: confirmed empirically (not hypothetically) for `loan_default` — SMOTE interpolation produces fractional (non-0/1) values in one-hot columns for ~6% of rows. This is why Fidelity's and Robustness's baseline/std computations both read from `X_train_prebalance.csv`, never `X_train_balanced.csv`. This is a real, documented limitation of the LIME explanations themselves (their `background_data_source` is `X_train_balanced`), not something retroactively fixable — see `FIDELITY_README.md` §5.2.
- **Feature-group map source of truth**: `build_feature_group_map()` in the current `evaluation.py` reads `categorical_cols`/`numerical_cols`/`encoder` directly from each dataset's fitted `OneHotEncoder` bundle (`fitted_transformers.joblib`) — this superseded an earlier design that read from `dataset_info.json`'s `categorical_features` list. Both Fidelity and Robustness reuse this same function, so masking/perturbation and ranking are guaranteed to operate at consistent granularity across all four already-built metric families.
- **Working directory / imports in notebooks**: notebooks must `os.chdir` to project root (not rely on `sys.path.insert(0, "..")` alone) before calling any `run_*` function, since file paths (`Explanations/`, `Datasets/`, etc.) are resolved relative to cwd, not the notebook's own location. Standard fix cell:
  ```python
  import os, sys
  if not os.path.isdir("Explanations"):
      os.chdir("..")
  if os.getcwd() not in sys.path:
      sys.path.insert(0, os.getcwd())
  ```

## 7. Companion Documents (canonical protocol + formulas + citations)

- `EFFICIENCY_README.md` — runtime-per-instance, feature-count normalization, real worked example showing a genuine RF-vs-XGB SHAP/LIME runtime reversal on `loan_default`.
- `SIMPLICITY_README.md` — entropy-based complexity + cumulative-mass-coverage, Bhatt et al. formulation, real worked example across Pima/loan_default.
- `FIDELITY_README.md` — comprehensiveness/sufficiency/AOPC, the full masking-baseline protocol (§5, including the SMOTE finding), faithfulness-vs-fidelity terminology distinction.
- `ROBUSTNESS_README.md` — perturbation protocol, Spearman/Jaccard formulas, the real RF-vs-XGB robustness gap on Pima, and the open cross-dataset "same volatile feature" investigation (§7.4 — flagged as an observation needing formal verification, not yet a confirmed finding).

All four follow the same structure (conceptual definition → why selected → theoretical foundation with numbered citations → formal definitions → implementation → worked example → interpretation tied to the four-bucket framework → limitations → references) and are meant to be read together as one reference set for the paper's methods section.

## 8. Open TODOs for a Fresh Chat to Pick Up

1. **Build Stability** (`src/stability.py`), following the same reuse-from-`evaluation.py` pattern as Fidelity/Robustness, per the planned protocol in §5 above.
2. **Formally verify the Robustness §7.4 cross-dataset "same volatile feature" observation** (XGBoost selecting the same most-volatile feature across `credit_card_fraud_2023`, `financial_distress`, `breast_cancer_wisconsin`, `pima_diabetes`) against the full 24-combo result set, cross-referencing against each dataset's Fidelity-ranked top feature — see `ROBUSTNESS_README.md` §7.4 for the three candidate hypotheses and the recommended check.
3. **Run the variance-decomposition / mixed-model analysis** (domain, dataset-within-domain, model, explainer as factors) across all five metric families once Stability is complete — this is the analysis that actually answers the paper's central research question and produces the four-bucket classification per metric.
4. **Confirm coverage**: all 24 combos are done for Efficiency, Simplicity, and (per the user) Robustness; Fidelity's full-matrix run should be double-checked now that the Git LFS issue is resolved.
