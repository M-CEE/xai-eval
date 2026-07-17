"""
Machine-readable metadata generation, so results from multiple datasets
(processed under the same protocol) can be aggregated and compared
programmatically later -- e.g. across SHAP / LIME / stability analyses.
"""

import json
import os
import pandas as pd


def write_dataset_info_json(meta_dir, dataset_name, domain, df_raw, target,
                             numerical_cols, categorical_cols, source):
    info = {
        "dataset_name": dataset_name,
        "domain": domain,
        "source": source,
        "rows": int(df_raw.shape[0]),
        "features": int(df_raw.shape[1] - 1),
        "target": target,
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
    }
    path = os.path.join(meta_dir, "dataset_info.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return path


def write_preprocessing_log_json(meta_dir, config, extra=None):
    """config is the CONFIG dict used to run the pipeline; extra holds any
    runtime-determined values (e.g. imputation values actually used) that are
    worth recording alongside the settings.
    """
    record = dict(config)
    if extra:
        record["runtime"] = extra
    path = os.path.join(meta_dir, "preprocessing_log.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return path


def write_class_distribution_csv(meta_dir, y_train_before, y_train_after=None, y_test=None):
    """Records class balance before/after balancing (train), and test set balance."""
    rows = []

    def pct(y):
        return (y.value_counts(normalize=True) * 100).round(2)

    before = pct(y_train_before)
    for cls, val in before.items():
        rows.append({"partition": "train_before_balancing", "class": cls, "pct": val})

    if y_train_after is not None:
        after = pct(y_train_after)
        for cls, val in after.items():
            rows.append({"partition": "train_after_balancing", "class": cls, "pct": val})

    if y_test is not None:
        test_pct = pct(y_test)
        for cls, val in test_pct.items():
            rows.append({"partition": "test", "class": cls, "pct": val})

    df_out = pd.DataFrame(rows)
    path = os.path.join(meta_dir, "class_distribution.csv")
    df_out.to_csv(path, index=False)
    return path


def write_feature_dictionary_csv(meta_dir, feature_descriptions):
    """feature_descriptions: dict of {feature_name: description}. Optional --
    only written if descriptions are supplied, since these are usually
    dataset-specific domain knowledge that can't be auto-derived.
    """
    if not feature_descriptions:
        return None
    df_out = pd.DataFrame(
        [{"feature": k, "description": v} for k, v in feature_descriptions.items()]
    )
    path = os.path.join(meta_dir, "feature_dictionary.csv")
    df_out.to_csv(path, index=False)
    return path


def print_summary_report(logger, dataset_name, domain, df_raw, X_train, X_test,
                          missing_before, duplicates_removed, scaler_name,
                          imputation_strategy, balancing_method,
                          class_before, class_after=None,
                          train_size_after_balancing=None):
    lines = []
    lines.append("=" * 40)
    lines.append(f"DATASET: {dataset_name}")
    lines.append(f"DOMAIN: {domain}")
    lines.append("")
    lines.append(f"Samples: {df_raw.shape[0]}")
    lines.append(f"Features: {df_raw.shape[1] - 1}")
    lines.append("")
    lines.append(f"Missing Values (pre-imputation): {missing_before}")
    lines.append(f"Duplicates Removed: {duplicates_removed}")
    lines.append("")
    lines.append(f"Train Size (before balancing): {X_train.shape[0]}")
    if train_size_after_balancing is not None:
        lines.append(f"Train Size (after balancing): {train_size_after_balancing}")
    lines.append(f"Test Size: {X_test.shape[0]}")
    lines.append("")
    lines.append(f"Scaling: {scaler_name}")
    lines.append(f"Imputation: {imputation_strategy}")
    lines.append("")
    lines.append("Class Balance Before:")
    for cls, val in class_before.items():
        lines.append(f"  {cls} = {val:.1f}%")
    if class_after is not None:
        lines.append("")
        lines.append("Class Balance After:")
        for cls, val in class_after.items():
            lines.append(f"  {cls} = {val:.1f}%")
        lines.append(f"\nBalancing: {balancing_method}")
    lines.append("=" * 40)

    report = "\n".join(lines)
    logger.log("\n" + report)
    return report
