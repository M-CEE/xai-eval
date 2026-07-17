"""
Reusable preprocessing pipeline, intended to be run identically across every
dataset in the study so that the resulting features are comparable in a
downstream XAI analysis (SHAP / LIME / stability / fidelity).

Step order (important):
    1.  Configuration            (supplied by caller)
    2.  Dataset loading
    3.  Initial inspection
    4.  Data quality assessment  (dtype split, duplicate count, zero-as-missing
                                  recoding -- domain-knowledge recoding only,
                                  no distributional statistics are computed
                                  from the full data)
    5.  Train/test split         (BEFORE any statistic is fit)
    6.  Missing value handling   (imputer fit on train only, applied to test)
    7.  Encoding                 (encoder fit on train only, applied to test)
    8.  Outlier analysis         (train only; documented, not auto-removed)
    9.  Scaling                  (scaler fit on train only)
    10. Class imbalance handling (SMOTE on train only, never on test)
    11. Save processed data
    12. Generate metadata files
    13. Summary report

This ordering avoids the leakage risk of fitting imputers/scalers on the full
dataset before splitting.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from src.utils import setup_output_dirs, Logger, FigureRegistry, set_plot_style
from src import metadata as meta_mod

try:
    from imblearn.over_sampling import SMOTE
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False


SCALERS = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
}


# ---------------------------------------------------------------------------
# 2. Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(data_path, column_names=None, logger=None):
    if column_names is not None:
        df = pd.read_csv(data_path, header=None, names=column_names)
    else:
        df = pd.read_csv(data_path)
    if logger:
        logger.section("2. DATASET LOADING")
        logger.log(f"Loaded from: {data_path}")
        logger.log(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 3. Initial inspection
# ---------------------------------------------------------------------------

def initial_inspection(df, logger, log_dir):
    logger.section("3. INITIAL INSPECTION")
    logger.log(df.dtypes.to_string())

    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(log_dir, "describe_raw.csv"))
    logger.log("\nDescriptive statistics:")
    logger.log(desc.to_string())

    n_missing = int(df.isnull().sum().sum())
    n_dupes = int(df.duplicated().sum())
    logger.log(f"\nTotal missing cells: {n_missing}")
    logger.log(f"Duplicate rows: {n_dupes}")
    return {"n_missing": n_missing, "n_duplicates": n_dupes}


# ---------------------------------------------------------------------------
# 4. Data quality assessment
# ---------------------------------------------------------------------------

def data_quality_assessment(df, logger, log_dir, zero_as_missing_cols=None,
                             numerical_cols=None, categorical_cols=None,
                             drop_duplicates=True, high_missingness_warn_threshold=0.40,
                             exclude_cols=None):
    """Recodes domain-known sentinel values (e.g. 0 used as missing) to NaN,
    drops exact duplicate rows, and infers/records feature types.

    Recoding zeros to NaN and dropping exact duplicates are both deterministic,
    domain-level operations -- they do not involve fitting any statistic to
    the data, so doing them before the train/test split is not a leakage risk
    (unlike imputation, encoding, or scaling, which must be fit on train only).

    Any column whose missingness exceeds `high_missingness_warn_threshold`
    triggers an explicit WARNING in the log: median/mode-imputing a column
    that is more than ~40% missing can materially distort its distribution,
    and that should be a visible, reviewer-facing methodological note rather
    than a silent default.
    """
    logger.section("4. DATA QUALITY ASSESSMENT")

    df = df.copy()

    n_dupes = int(df.duplicated().sum())
    if drop_duplicates and n_dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    logger.log(f"Duplicate rows found: {n_dupes} (removed: {n_dupes if drop_duplicates else 0})")

    if zero_as_missing_cols:
        zero_counts = (df[zero_as_missing_cols] == 0).sum()
        zero_counts.to_csv(os.path.join(log_dir, "zero_value_counts.csv"))
        logger.log("\nZero-value counts in columns where 0 is not physiologically valid:")
        logger.log(zero_counts.to_string())
        for col in zero_as_missing_cols:
            df.loc[df[col] == 0, col] = np.nan

    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    if exclude_cols:
        numerical_cols = [c for c in numerical_cols if c not in exclude_cols]
        categorical_cols = [c for c in categorical_cols if c not in exclude_cols]

    missing_report = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (df.isnull().mean() * 100).round(2)
    missing_summary = pd.DataFrame({"missing_count": missing_report, "missing_pct": missing_pct})
    missing_summary.to_csv(os.path.join(log_dir, "missingness_summary.csv"))
    logger.log("\nMissingness summary (after sentinel recoding):")
    logger.log(missing_summary.to_string())

    high_missing = missing_summary[missing_summary["missing_pct"] > high_missingness_warn_threshold * 100]
    if not high_missing.empty:
        logger.log(f"\nWARNING: the following columns exceed {high_missingness_warn_threshold*100:.0f}% "
                    f"missing and will still be median/mode-imputed under the standard protocol. "
                    f"Imputing at this rate can materially distort these features -- flag for the "
                    f"methodology/limitations section:")
        logger.log(high_missing.to_string())

    logger.log(f"\nNumerical columns: {numerical_cols}")
    logger.log(f"Categorical columns: {categorical_cols}")

    return df, numerical_cols, categorical_cols, n_dupes


# ---------------------------------------------------------------------------
# 5. Train/test split
# ---------------------------------------------------------------------------

def split_data(df, target, test_size, random_state, logger):
    logger.section("5. TRAIN/TEST SPLIT")
    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if y.nunique() <= 20 else None  # stratify only for classification-like targets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    logger.log(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    logger.log(f"Stratified on target: {stratify is not None}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 6. Missing value handling (fit on train only)
# ---------------------------------------------------------------------------

def impute_missing(X_train, X_test, numerical_cols, categorical_cols, logger, log_dir,
                    strategy_num="median", strategy_cat="most_frequent"):
    logger.section("6. MISSING VALUE HANDLING")
    X_train = X_train.copy()
    X_test = X_test.copy()
    imputers = {}

    if numerical_cols:
        num_imputer = SimpleImputer(strategy=strategy_num)
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])
        imputers["numerical"] = num_imputer
        values = pd.Series(num_imputer.statistics_, index=numerical_cols, name=f"{strategy_num}_used")
        values.to_csv(os.path.join(log_dir, "imputation_values_numerical.csv"))
        logger.log(f"Numerical imputation strategy: {strategy_num} (fit on train only)")
        logger.log(values.to_string())

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy=strategy_cat)
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
        imputers["categorical"] = cat_imputer
        values = pd.Series(cat_imputer.statistics_, index=categorical_cols, name=f"{strategy_cat}_used")
        values.to_csv(os.path.join(log_dir, "imputation_values_categorical.csv"))
        logger.log(f"\nCategorical imputation strategy: {strategy_cat} (fit on train only)")
        logger.log(values.to_string())

    remaining_na = int(X_train.isnull().sum().sum() + X_test.isnull().sum().sum())
    logger.log(f"\nRemaining missing values after imputation: {remaining_na}")
    assert remaining_na == 0, "Unexpected missing values remain after imputation."

    return X_train, X_test, imputers


# ---------------------------------------------------------------------------
# 7. Encoding (fit on train only)
# ---------------------------------------------------------------------------

def encode_categorical(X_train, X_test, categorical_cols, logger):
    logger.section("7. ENCODING")
    if not categorical_cols:
        logger.log("No categorical columns present -- encoding step skipped.")
        return X_train, X_test, None

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_enc = encoder.fit_transform(X_train[categorical_cols])
    test_enc = encoder.transform(X_test[categorical_cols])
    enc_cols = encoder.get_feature_names_out(categorical_cols)

    X_train_enc = pd.DataFrame(train_enc, columns=enc_cols, index=X_train.index)
    X_test_enc = pd.DataFrame(test_enc, columns=enc_cols, index=X_test.index)

    X_train = pd.concat([X_train.drop(columns=categorical_cols), X_train_enc], axis=1)
    X_test = pd.concat([X_test.drop(columns=categorical_cols), X_test_enc], axis=1)

    logger.log(f"One-hot encoded columns: {categorical_cols}")
    logger.log(f"Resulting encoded feature columns: {list(enc_cols)}")
    return X_train, X_test, encoder


# ---------------------------------------------------------------------------
# 8. Outlier analysis (train only, document -- do not auto-remove)
# ---------------------------------------------------------------------------

def outlier_analysis(X_train, numerical_cols, logger, log_dir, fig_registry):
    logger.section("8. OUTLIER ANALYSIS (documented only, not removed)")

    def iqr_bounds(series):
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr

    rows = []
    for col in numerical_cols:
        lower, upper = iqr_bounds(X_train[col])
        n_out = int(((X_train[col] < lower) | (X_train[col] > upper)).sum())
        rows.append({"feature": col, "n_outliers_iqr": n_out, "lower_bound": lower, "upper_bound": upper})
    outlier_df = pd.DataFrame(rows).set_index("feature")
    outlier_df.to_csv(os.path.join(log_dir, "iqr_outlier_summary.csv"))
    logger.log(outlier_df.to_string())
    logger.log("\nNote: outliers are documented for review only; no rows were removed. "
               "Automatic removal is avoided here to preserve interpretability for "
               "downstream XAI analysis (SHAP/LIME).")

    if numerical_cols:
        n = len(numerical_cols)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for ax, col in zip(axes, numerical_cols):
            sns.boxplot(y=X_train[col], ax=ax, color="#DD8452")
            ax.set_ylabel(col)
        for ax in axes[len(numerical_cols):]:
            ax.axis("off")
        fig.tight_layout()
        fig_registry.save(
            fig, "fig_outlier_boxplots_train_set.png",
            "Boxplots of all numerical training-set features used for IQR-based "
            "outlier screening (outliers documented, not removed).",
        )

    return outlier_df


# ---------------------------------------------------------------------------
# 9. Scaling (fit on train only)
# ---------------------------------------------------------------------------

def scale_features(X_train, X_test, numerical_cols, scaler_name, logger, log_dir):
    logger.section("9. SCALING")
    if scaler_name not in SCALERS:
        raise ValueError(f"Unknown scaler '{scaler_name}'. Options: {list(SCALERS)}")

    scaler = SCALERS[scaler_name]()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    if scaler_name == "StandardScaler":
        params = pd.DataFrame({"mean": scaler.mean_, "scale": scaler.scale_}, index=numerical_cols)
    else:
        params = pd.DataFrame({"data_min": scaler.data_min_, "data_max": scaler.data_max_}, index=numerical_cols)
    params.to_csv(os.path.join(log_dir, "scaler_parameters.csv"))
    logger.log(f"Scaler: {scaler_name} (fit on train only)")
    logger.log(params.to_string())

    return X_train, X_test, scaler


# ---------------------------------------------------------------------------
# 10. Class imbalance handling (train only)
# ---------------------------------------------------------------------------

def balance_classes(X_train, y_train, method, random_state, logger):
    logger.section("10. CLASS IMBALANCE HANDLING")
    before_pct = (y_train.value_counts(normalize=True) * 100).round(2)
    logger.log("Class balance before:")
    logger.log(before_pct.to_string())

    if method is None or str(method).lower() == "none":
        logger.log("\nNo balancing applied (method=None).")
        return X_train, y_train, before_pct, before_pct

    if method.upper() == "SMOTE":
        if not _HAS_IMBLEARN:
            raise ImportError("imbalanced-learn is not installed; cannot apply SMOTE.")
        smote = SMOTE(random_state=random_state)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
        after_pct = (y_bal.value_counts(normalize=True) * 100).round(2)
        logger.log("\nApplied SMOTE (train set only; test set untouched).")
        logger.log("Class balance after:")
        logger.log(after_pct.to_string())
        return X_bal, y_bal, before_pct, after_pct

    raise ValueError(f"Unknown balancing method '{method}'.")


# ---------------------------------------------------------------------------
# 11. Save processed data
# ---------------------------------------------------------------------------

def save_processed_data(data_dir, X_train_prebalance, y_train_prebalance,
                         X_train_balanced, y_train_balanced, X_test, y_test, target, logger):
    """Saves both the pre-balance and post-balance training sets.

    The pre-balance set (post impute/encode/scale, before SMOTE) is what a
    hyperparameter search should resample from -- SMOTE needs to be refit
    fresh inside each CV fold, not applied once before the folds are cut,
    or synthetic points derived from a training-fold real point can leak
    into a validation fold that real point isn't in. The balanced set is
    kept too, for the final refit on the full training data.
    """
    logger.section("11. SAVE PROCESSED DATA")
    files = {
        "X_train_prebalance.csv": X_train_prebalance,
        "y_train_prebalance.csv": y_train_prebalance.to_frame(name=target),
        "X_train_balanced.csv": X_train_balanced,
        "y_train_balanced.csv": y_train_balanced.to_frame(name=target),
        "X_test.csv": X_test,
        "y_test.csv": y_test.to_frame(name=target),
    }
    for fname, obj in files.items():
        path = os.path.join(data_dir, fname)
        obj.to_csv(path, index=False)
        logger.log(f"Saved: {path}")
    logger.log(
        "\nNote: 'prebalance' = post impute/encode/scale, pre-SMOTE (use this for CV-based "
        "hyperparameter search, resampling inside each fold). 'balanced' = full SMOTE-applied "
        "training set (use this for the final refit with the chosen hyperparameters)."
    )


def save_fitted_transformers(metadata_dir, imputers, encoder, scaler, numerical_cols, categorical_cols, logger):
    """Pickles the fitted imputer/encoder/scaler objects (not just their parameters).

    Reconstructing e.g. a StandardScaler from a mean/scale CSV works but is
    fragile and easy to get subtly wrong (column order, dtype). Downstream
    code -- training, and later SHAP/LIME/Anchors explanation code -- should
    load this bundle and call .transform() directly, guaranteeing the exact
    same feature order and encoding the model was fit on.
    """
    logger.section("11b. SAVE FITTED TRANSFORMERS")
    bundle = {
        "numerical_imputer": imputers.get("numerical"),
        "categorical_imputer": imputers.get("categorical"),
        "encoder": encoder,
        "scaler": scaler,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
    }
    path = os.path.join(metadata_dir, "fitted_transformers.joblib")
    joblib.dump(bundle, path)
    logger.log(f"Saved fitted imputer/encoder/scaler bundle: {path}")
    return path


# ---------------------------------------------------------------------------
# Visualizations (descriptive; run before split so "before" pictures reflect
# the full observed data, consistent across datasets)
# ---------------------------------------------------------------------------

def generate_descriptive_visualizations(df, numerical_cols, target, fig_registry, logger):
    logger.section("DESCRIPTIVE VISUALIZATIONS")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df.isna(), cbar=False, cmap="Greys", yticklabels=False, ax=ax)
    ax.set_xlabel("Feature")
    fig_registry.save(
        fig, "fig_missing_value_pattern_heatmap.png",
        "Heatmap of missing values across all records after sentinel-value recoding, prior to imputation.",
    )

    if numerical_cols:
        n = len(numerical_cols)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes = np.atleast_1d(axes).flatten()
        for ax, col in zip(axes, numerical_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
        for ax in axes[len(numerical_cols):]:
            ax.axis("off")
        fig.tight_layout()
        fig_registry.save(
            fig, "fig_feature_distributions_observed_values.png",
            "Histograms with kernel density estimates for numerical features, computed on observed (non-missing) values.",
        )

    if numerical_cols:
        fig, ax = plt.subplots(figsize=(0.9 * len(numerical_cols) + 3, 0.7 * len(numerical_cols) + 3))
        corr = df[numerical_cols + [target]].corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
        fig_registry.save(
            fig, "fig_pearson_correlation_heatmap.png",
            "Pearson correlation matrix across numerical features and the target variable.",
        )

    if df[target].nunique() <= 10:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        sns.countplot(x=target, data=df, hue=target, palette="deep", legend=False, ax=ax)
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
        fig_registry.save(
            fig, "fig_class_balance_target.png",
            f"Class balance of the target variable {target} in the full (pre-split) dataset.",
        )


# ---------------------------------------------------------------------------
# Orchestration: run_pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config):
    """Runs the full 13-step protocol for one dataset, given a CONFIG dict.

    Required config keys:
        dataset_name, domain, data_path, target, random_state, test_size,
        scaler, imputation_strategy_num, imputation_strategy_cat, balancing

    Optional config keys:
        column_names, zero_as_missing_cols, numerical_cols, categorical_cols,
        feature_descriptions, base_output_dir
    """
    set_plot_style()
    paths = setup_output_dirs(
        config["dataset_name"], config["domain"], config.get("datasets_root", "Datasets")
    )
    logger = Logger(paths["logs"])
    fig_registry = FigureRegistry(paths["figures"], logger=logger)

    logger.log(f"CONFIG: {config}")

    # 2. Load
    df_raw = load_dataset(config["data_path"], config.get("column_names"), logger)

    # Cache an untouched copy of the raw data for reproducibility / audit trail
    raw_cache_path = os.path.join(paths["raw"], f"{config['dataset_name']}_raw.csv")
    df_raw.to_csv(raw_cache_path, index=False)
    logger.log(f"Raw data cached to: {raw_cache_path}")

    # Drop dataset-specific junk columns (identifiers, empty artifact columns)
    # BEFORE inspection/quality steps -- these are not "features" in any sense
    # and would otherwise be treated as a numerical feature (id) or break
    # imputation with an all-NaN column (e.g. Kaggle's "Unnamed: 32").
    if config.get("drop_columns"):
        present = [c for c in config["drop_columns"] if c in df_raw.columns]
        if present:
            df_raw = df_raw.drop(columns=present)
            logger.log(f"Dropped non-feature columns: {present}")

    # Map target labels to a canonical encoding (e.g. {'M': 1, 'B': 0}) if given
    if config.get("target_mapping"):
        target_col = config["target"]
        df_raw[target_col] = df_raw[target_col].map(config["target_mapping"])
        logger.log(f"Applied target_mapping to '{target_col}': {config['target_mapping']}")

    # Binarize a continuous target using a fixed threshold (e.g. a financial
    # distress score -> distressed/healthy). This is a deterministic, published
    # domain rule (not a statistic fit on the data), so applying it here --
    # before the train/test split -- is not a leakage risk, exactly like
    # target_mapping above. The raw cache written above still holds the
    # original, untouched continuous values.
    if config.get("target_binarize"):
        target_col = config["target"]
        tb = config["target_binarize"]
        threshold = tb["threshold"]
        mode = tb.get("positive_if", "leq")  # value relative to threshold that maps to class 1
        comparisons = {
            "leq": df_raw[target_col] <= threshold,
            "less": df_raw[target_col] < threshold,
            "geq": df_raw[target_col] >= threshold,
            "greater": df_raw[target_col] > threshold,
        }
        if mode not in comparisons:
            raise ValueError(f"Unknown target_binarize.positive_if '{mode}'. Options: {list(comparisons)}")
        df_raw[target_col] = comparisons[mode].astype(int)
        logger.log(f"Binarized target '{target_col}': class 1 where original value {mode} {threshold}, else 0")

    # 3. Initial inspection
    inspection = initial_inspection(df_raw, logger, paths["logs"])

    # 4. Data quality assessment (sentinel recoding, dedup, type split)
    df_clean, numerical_cols, categorical_cols, n_dupes = data_quality_assessment(
        df_raw, logger, paths["logs"],
        zero_as_missing_cols=config.get("zero_as_missing_cols"),
        numerical_cols=config.get("numerical_cols"),
        categorical_cols=config.get("categorical_cols"),
        high_missingness_warn_threshold=config.get("high_missingness_warn_threshold", 0.40),
        exclude_cols=[config["target"]],
    )
    # (already excluded above; kept as a harmless no-op safety net)
    numerical_cols = [c for c in numerical_cols if c != config["target"]]
    categorical_cols = [c for c in categorical_cols if c != config["target"]]

    missing_before = int(df_clean.isnull().sum().sum())

    # Descriptive visuals on the cleaned, pre-split data (consistent across datasets)
    generate_descriptive_visualizations(df_clean, numerical_cols, config["target"], fig_registry, logger)

    # 5. Split BEFORE fitting anything
    X_train, X_test, y_train, y_test = split_data(
        df_clean, config["target"], config["test_size"], config["random_state"], logger
    )

    # 6. Impute (fit on train only)
    X_train, X_test, imputers = impute_missing(
        X_train, X_test, numerical_cols, categorical_cols, logger, paths["logs"],
        strategy_num=config.get("imputation_strategy_num", "median"),
        strategy_cat=config.get("imputation_strategy_cat", "most_frequent"),
    )

    # 7. Encode (fit on train only)
    X_train, X_test, encoder = encode_categorical(X_train, X_test, categorical_cols, logger)

    # 8. Outliers (train only, documented)
    outlier_df = outlier_analysis(X_train, numerical_cols, logger, paths["logs"], fig_registry)

    # 9. Scale (fit on train only)
    X_train, X_test, scaler = scale_features(
        X_train, X_test, numerical_cols, config.get("scaler", "StandardScaler"), logger, paths["logs"]
    )

    # X_train/y_train at this point are post impute/encode/scale, pre-balance --
    # kept separately since balance_classes below overwrites these names.
    X_train_prebalance = X_train.copy()
    y_train_prebalance = y_train.copy()

    # 10. Balance (train only)
    X_train_bal, y_train_bal, class_before, class_after = balance_classes(
        X_train, y_train, config.get("balancing"), config["random_state"], logger
    )

    # 11. Save processed data (both pre-balance and balanced training sets)
    save_processed_data(
        paths["data"], X_train_prebalance, y_train_prebalance,
        X_train_bal, y_train_bal, X_test, y_test, config["target"], logger,
    )
    save_fitted_transformers(
        paths["metadata"], imputers, encoder, scaler, numerical_cols, categorical_cols, logger,
    )

    # 12. Metadata
    meta_mod.write_dataset_info_json(
        paths["metadata"], config["dataset_name"], config["domain"], df_raw,
        config["target"], numerical_cols, categorical_cols, config["data_path"],
    )
    meta_mod.write_preprocessing_log_json(
        paths["metadata"], config,
        extra={
            "n_duplicates_removed": n_dupes,
            "missing_before_imputation": missing_before,
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "train_rows_after_balancing": int(X_train_bal.shape[0]),
            "prebalance_train_file": "X_train_prebalance.csv",
            "balanced_train_file": "X_train_balanced.csv",
            "fitted_transformers_file": "fitted_transformers.joblib",
        },
    )
    meta_mod.write_class_distribution_csv(paths["metadata"], y_train, y_train_bal, y_test)
    meta_mod.write_feature_dictionary_csv(paths["metadata"], config.get("feature_descriptions"))

    # 13. Summary report
    summary_text = meta_mod.print_summary_report(
        logger, config["dataset_name"], config["domain"], df_raw, X_train, X_test,
        missing_before, n_dupes, config.get("scaler", "StandardScaler"),
        config.get("imputation_strategy_num", "median"), config.get("balancing"),
        class_before, class_after,
        train_size_after_balancing=X_train_bal.shape[0],
    )

    return {
        "paths": paths,
        "df_raw": df_raw,
        "df_clean": df_clean,
        "X_train": X_train_bal,
        "X_test": X_test,
        "y_train": y_train_bal,
        "y_test": y_test,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "outlier_summary": outlier_df,
        "summary_text": summary_text,
    }
