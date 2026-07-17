# Datasets — Stage 1: Acquisition & Preprocessing

## What this stage is

This is the first stage of a study evaluating SHAP and LIME explanations across two
domains (finance, healthcare), three datasets per domain, two model families
(Random Forest, XGBoost), and a shared, dimensionality-normalized evaluation
protocol. This folder holds, for each of the six datasets, the raw cache, the
processed train/test artifacts, the fitted transformers, and a full audit trail
(logs, figures, metadata) documenting exactly how each dataset was turned into
model-ready tensors.

All six datasets are run through **one shared pipeline** (`src/preprocessing.py`,
function `run_pipeline(config)`), configured per dataset via a `CONFIG` dict in the
corresponding `notebooks/0N_preprocessing_*.ipynb`. Using a single pipeline for every
dataset — rather than bespoke per-dataset scripts — is what makes cross-dataset,
cross-domain comparisons in the later Evaluation stage meaningful: differences in
downstream XAI metrics can be attributed to the data/model/explainer, not to
inconsistent preprocessing choices.

> **Note on this shared copy:** to keep the archive small, only `loan_default`
> (finance) and `pima_diabetes` (healthcare) retain their full processed
> train/test data, figures, and logs here. The other four datasets
> (`breast_cancer_wisconsin`, `heart_disease_uci`, `financial_distress`,
> `credit_card_fraud_2023`) are represented only by their config-only notebooks
> under `notebooks/`. Because `run_pipeline` is deterministic given the same
> `CONFIG` and `random_state=42`, every dataset's full artifact tree is exactly
> reproducible by re-running its notebook.

## The six datasets

| Domain | Dataset | Source | Rows | Target | Notes |
|---|---|---|---|---|---|
| Healthcare | `pima_diabetes` | UCI Pima Indians Diabetes (public mirror) | 768 | `Outcome` (0/1) | 0 recoded to missing in 5 physiologically-invalid columns |
| Healthcare | `breast_cancer_wisconsin` | Kaggle `uciml/breast-cancer-wisconsin-data` (UCI WDBC) | 569 | `diagnosis` → {M:1, B:0} | `id`/`Unnamed: 32` dropped (non-feature artifact columns) |
| Healthcare | `heart_disease_uci` | Kaggle `redwankarimsony/heart-disease-data` (Cleveland/Hungary/Switzerland/VA) | ~920 | `num` (0–4) → binarized {0:0, 1–4:1} | `dataset` (clinical site) dropped to avoid site-identity leakage; `ca`/`thal` are 53–66% missing (flagged, still imputed) |
| Finance | `loan_default` | Kaggle `nikhil1e9/loan-default` | 255,347 | `Default` (0/1) | `LoanID` dropped; ~11.6% base default rate |
| Finance | `financial_distress` | Kaggle `shebrahimi/financial-distress` | — (panel data) | continuous distress score → binarized at −0.50 | `Company` dropped; `Time` (fiscal period) kept as an ordinary numeric feature — see limitation below |
| Finance | `credit_card_fraud_2023` | Kaggle `nelgiriyewithana/credit-card-fraud-detection-dataset-2023` | ~568,630 | `Class` (0/1) | Synthetically rebalanced (~50/50); `id` dropped; SMOTE deliberately **not** applied (see below) |

## Folder layout (per dataset)

```
Datasets/<domain>/<dataset_name>/
├── raw/<dataset_name>_raw.csv          # untouched, as-downloaded cache
├── processed/
│   ├── data/
│   │   ├── X_train_prebalance.csv      # post impute/encode/scale, pre-SMOTE
│   │   ├── y_train_prebalance.csv
│   │   ├── X_train_balanced.csv        # full SMOTE-applied training set
│   │   ├── y_train_balanced.csv
│   │   ├── X_test.csv                  # held-out test set (never touched by SMOTE)
│   │   └── y_test.csv
│   └── figures/                        # descriptive plots, filename = caption key
│       └── figure_captions.txt         # filename -> full caption sentence
└── metadata/
    ├── dataset_info.json               # name, domain, source, target, feature lists
    ├── preprocessing_log.json          # structured record of every config/decision
    ├── class_distribution.csv          # class balance before/after SMOTE
    ├── feature_dictionary.csv          # human-readable feature descriptions
    ├── fitted_transformers.joblib      # imputer(s) + encoder + scaler, fit on train
    └── logs/
        ├── preprocessing_log.txt       # full narrative log of the 13-step run
        ├── describe_raw.csv
        ├── missingness_summary.csv
        ├── imputation_values_{numerical,categorical}.csv
        ├── scaler_parameters.csv
        └── iqr_outlier_summary.csv
```

Two training sets are saved deliberately: `X_train_prebalance` is what a
cross-validated hyperparameter search resamples from (SMOTE is refit **inside**
each CV fold in the Models stage — see `Models/README.md`); `X_train_balanced` is
the full SMOTE-applied set used for the final model refit. Saving both makes the
leakage-avoidance logic auditable rather than implicit.

## The pipeline, step by step (`src/preprocessing.py: run_pipeline`)

The 13 numbered steps run in a fixed order chosen specifically to avoid **data
leakage** — fitting any statistic (an imputer, encoder, scaler, or resampler) on
data it will later be evaluated against inflates apparent performance and is one
of the most common, hard-to-detect sources of invalid results in applied ML
(Kaufman, Rosset, Perlich & Stitelman, 2012, *"Leakage in Data Mining: Formulation,
Detection, and Avoidance,"* ACM TKDD). Concretely:

1. **Configuration** — one `CONFIG` dict per dataset, versioned in its notebook.
2. **Load** — raw CSV read in, cached untouched under `raw/` for a reproducibility trail.
3. **Initial inspection** — dtypes, `describe()`, missingness/duplicate counts logged before anything is changed.
4. **Data quality assessment** — domain-known sentinel recoding (e.g. `0` → NaN for `Glucose`/`BloodPressure`/etc. in Pima, where zero is physiologically impossible) and exact-duplicate removal. These are **deterministic, rule-based** operations, not statistics fit to the data, so — unlike imputation/encoding/scaling below — doing them before the split is not a leakage risk.
5. **Train/test split** — `sklearn.train_test_split`, stratified on the target, `test_size=0.20`, `random_state=42`, performed **before any statistic is fit**. This is the pivot point of the whole leakage-avoidance design: every step after this fits only on the training partition.
6. **Missing value imputation** — `SimpleImputer` (median for numeric, most-frequent for categorical), **fit on train, applied to test**.
7. **Encoding** — `OneHotEncoder(handle_unknown="ignore")`, fit on train only.
8. **Outlier analysis** — IQR-based (Tukey, 1977, *Exploratory Data Analysis*): bounds at Q1 − 1.5·IQR and Q3 + 1.5·IQR, computed on the training set. Outliers are **documented, not removed** — this is a deliberate choice to preserve every instance's interpretability for the downstream SHAP/LIME analysis, since silently dropping "unusual" rows would bias exactly the kind of edge cases explanation-quality metrics (especially Fidelity/Robustness) need to be evaluated on.
9. **Scaling** — `StandardScaler` (or `MinMaxScaler`), fit on train only; parameters logged to `scaler_parameters.csv` for exact downstream reproducibility.
10. **Class imbalance handling** — `SMOTE` (Chawla, Bowyer, Hall & Kegelmeyer, 2002, *"SMOTE: Synthetic Minority Over-sampling Technique,"* JAIR 16), applied to the training set only, **never** to the test set. SMOTE generates synthetic minority-class points by interpolating between a minority sample and its k-nearest minority neighbors, which is preferable here to naive duplication (which encourages overfitting to exact repeated points) or majority under-sampling (which would discard real data in already-small clinical datasets like Pima). The one dataset where this is turned off (`credit_card_fraud_2023`) is a documented exception (see below).
11. **Save** — both pre-balance and balanced training sets, plus the test set, as CSVs.
12. **Metadata** — `dataset_info.json`, `preprocessing_log.json`, class-distribution and feature-dictionary CSVs, and the fitted-transformer bundle (`fitted_transformers.joblib`) — saving the actual fitted objects, not just their parameters, so downstream code calls `.transform()` directly and can't subtly reconstruct them wrong (e.g. wrong column order/dtype).
13. **Summary report** — a narrative recap logged to `preprocessing_log.txt`.

## Dataset-specific decisions (and why)

- **Pima Diabetes**: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` use `0` as a missing-value sentinel in the original UCI release (these values are not physiologically possible); recoded to NaN before imputation.
- **Breast Cancer Wisconsin**: `id` and the empty `Unnamed: 32` artifact column are dropped pre-inspection so they aren't treated as a numeric feature or an all-NaN column that would break imputation; target mapped `{M:1, B:0}`.
- **Heart Disease (UCI combined)**: `dataset` (clinical site: Cleveland/Hungary/Switzerland/VA) is dropped so the model can't key off site identity rather than clinical signal — a documented, deliberate exclusion, not an oversight. The raw `num` severity score (0–4) is binarized to absence/presence (`0` vs `1`), the standard convention for this dataset in the published literature. `ca` and `thal` are ~66%/53% missing; the pipeline still imputes them under the fixed protocol, but this is explicitly flagged in the log (see `high_missingness_warn_threshold`) as a methodological caveat worth stating in any write-up rather than a silent default.
- **Loan Default**: `LoanID` dropped (identifier, not a feature); target already binary, no remapping needed; base default rate ≈ 11.6% (imbalanced, motivating SMOTE).
- **Financial Distress**: `Company` dropped (per-firm identifier). This is **panel data** (multiple rows per company across up to 14 fiscal periods, retained via the `Time` feature); the pipeline performs a plain row-level train/test split rather than a company-grouped split. This is a simplification applied uniformly across the whole study, and is worth flagging explicitly as a limitation: a company appearing in both train and test (at different time points) is a weaker form of leakage than fitting a statistic across the split, but it does mean test-set rows are not guaranteed to be from firms fully unseen in training. The continuous distress score is binarized at the published threshold of −0.50 (healthy if score > −0.50), a fixed domain rule rather than a statistic fit to the data, so applying it pre-split is not itself a leakage risk.
- **Credit Card Fraud (2023)**: this Kaggle release is a *synthetically rebalanced* version of the classic 2013 ULB fraud dataset (~50/50 legitimate/fraud across ~568k rows, not the ~0.17%-fraud imbalance of the original). SMOTE is therefore switched off for this dataset only — applying it to an already-balanced set would just duplicate/interpolate redundant points. This is a deliberate, documented, dataset-specific exception to the SMOTE-by-default rule used everywhere else, flagged here so it doesn't read as an inconsistency.

## Known limitations worth carrying into the write-up

- The `financial_distress` row-level (not company-grouped) split, noted above.
- Uniform imputation/encoding/scaling settings across all six datasets is a deliberate design choice for cross-dataset comparability, but it means dataset-specific quirks (e.g. `heart_disease_uci`'s high missingness in `ca`/`thal`) are handled by the same generic rule rather than a bespoke one.
- Outlier rows are retained (not removed) in every dataset, by design — see step 8 above.

## References

- Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 1–21.
- Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
