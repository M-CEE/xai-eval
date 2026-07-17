"""
Explanation evaluation, phase 2: Fidelity.

Builds on the shared infrastructure from src/evaluation.py (feature-group
map, original-feature aggregation, LIME condition parsing, the
metrics_long/run_ledger writers) rather than duplicating any of it. This
module adds what Fidelity needs on top: loading the trained model, computing
a masking baseline, masking instances at the ORIGINAL-feature granularity,
scoring masked instances through the model, and computing
comprehensiveness/sufficiency/AOPC.

RESOLVED PROTOCOL (matches the agreed decisions):

1. Baseline masking:
       - Numerical features -> training-set median, computed in SCALED
         space from X_train_prebalance.csv (median is invariant under the
         monotonic transform a StandardScaler applies, so this is
         equivalent to masking with the real-world median).
       - Categorical features -> the WHOLE one-hot group is replaced at
         once with the training-mode category's pattern (1.0 at the mode
         column, 0.0 elsewhere in the group). Never masks a single dummy
         column in isolation -- doing so can produce an encoding state the
         model never saw in training (two active levels, or all levels
         near-zero simultaneously).
       - Baselines are computed from X_train_prebalance.csv, NEVER
         X_train_balanced.csv. Where balancing is done via SMOTE (confirmed
         for loan_default in preprocessing_log.txt), SMOTE interpolates in
         continuous space -- the balanced set's one-hot columns contain
         fractional values (e.g. 0.9205) for a meaningful fraction of rows,
         which makes "mode" undefined and "median" meaningless. The
         prebalance set is clean in every dataset checked so far.
   Group membership reuses build_feature_group_map() from evaluation.py
   (fitted-encoder-based, not string-guessed) -- the SAME map used for
   Simplicity, so masking and ranking operate at identical granularity.

2. Ranking: per-instance attributions are aggregated to original-feature
   level via aggregate_to_original_features() from evaluation.py (SHAP:
   direct from shap_values.npy; LIME: parsed via parse_lime_feature() first)
   before ranking -- reusing the exact same functions Simplicity already
   uses, not a parallel implementation.

3. k in {10%, 20%, 30%, 50%} of ORIGINAL (post-aggregation) feature count,
   masked/kept in ranked order (rounded up to at least 1 feature, capped at
   n_features).

4. Comprehensiveness = P(original predicted class | original instance) -
   P(original predicted class | top-k features masked out). Large drop =
   faithful (those features really mattered).
   Sufficiency = P(original predicted class | original instance) -
   P(original predicted class | ONLY top-k features kept, everything else
   masked). Small drop = faithful (top-k alone nearly reproduces the
   prediction).
   "Original predicted class" = argmax of the model's own prediction on the
   UNMASKED instance, not a hardcoded positive-class index -- robust to
   each dataset's own class-imbalance direction.
   AOPC = mean of the 4 k-level values, per instance per condition.

5. Auditable artifact: Fidelity/<domain>/<dataset>/<model>/<explainer>/
   masked_predictions.csv -- one row per (instance, k, condition), with
   original and masked predicted-class probability. This is what lets you
   recompute AOPC under a different k-weighting later without re-scoring
   every masked instance from scratch.

6. metrics_long rows: TWO granularities, both written --
       - per (instance, k): metric_name="comprehensiveness"/"sufficiency",
         mask_fraction populated.
       - per instance, aggregated: metric_name="aopc_comprehensiveness"/
         "aopc_sufficiency", mask_fraction=None.
   baseline_type="median_mode_grouped" on every Fidelity row.
"""

import os
import json
import datetime as dt

import numpy as np
import pandas as pd
import joblib

from src.utils import Logger
from src.evaluation import (
    build_feature_group_map,
    aggregate_to_original_features,
    parse_lime_feature,
    append_metrics,
)

try:
    from skops import io as sio
    _HAS_SKOPS = True
except ImportError:
    _HAS_SKOPS = False

MASK_FRACTIONS = (0.10, 0.20, 0.30, 0.50)
BASELINE_TYPE = "median_mode_grouped"


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

def load_model(domain, dataset_name, model_name, models_root="Models", logger=None):
    """Mirrors src/training.py's save_model_artifact exactly:
    Models/<domain>/<dataset>/<model>/model/{model.skops OR model.joblib,
    feature_schema.json}. feature_schema.json's feature_order is cross-
    checked against the explanation artifacts' feature_order at call time
    (in run_fidelity) rather than here, since this function doesn't know
    about explanations.
    """
    model_dir = os.path.join(models_root, domain, dataset_name, model_name, "model")
    skops_path = os.path.join(model_dir, "model.skops")
    joblib_path = os.path.join(model_dir, "model.joblib")
    schema_path = os.path.join(model_dir, "feature_schema.json")

    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"feature_schema.json not found at {schema_path}. Needed to validate the model's "
            "own feature ordering against the explanation artifacts before scoring anything."
        )
    with open(schema_path) as f:
        schema = json.load(f)

    if os.path.exists(skops_path):
        if not _HAS_SKOPS:
            raise ImportError(
                f"{skops_path} exists but the 'skops' package isn't installed in this "
                "environment. Install it (pip install skops) to load this model."
            )
        untrusted = sio.get_untrusted_types(file=skops_path)
        model = sio.load(skops_path, trusted=untrusted)
        if logger:
            note = f" (trusted types: {untrusted})" if untrusted else ""
            logger.log(f"Loaded model (skops): {skops_path}{note}")
    elif os.path.exists(joblib_path):
        model = joblib.load(joblib_path)
        if logger:
            logger.log(f"Loaded model (joblib): {joblib_path}")
    else:
        raise FileNotFoundError(f"Neither model.skops nor model.joblib found in {model_dir}")

    return model, schema


# ---------------------------------------------------------------------------
# 2. Baseline computation (median/mode, from X_train_prebalance.csv only)
# ---------------------------------------------------------------------------

def compute_baseline_vector(domain, dataset_name, group_map, feature_order,
                             datasets_root="Datasets", logger=None):
    """Returns (baseline_vector, baseline_report):
        baseline_vector: np.array aligned to feature_order -- the
        replacement value for each model-input column when its group is
        masked.
        baseline_report: {original_feature: {...}} for schema.md-style
        documentation / debugging, not written to metrics_long directly.
    """
    prebalance_path = os.path.join(
        datasets_root, domain, dataset_name, "processed", "data", "X_train_prebalance.csv"
    )
    if not os.path.exists(prebalance_path):
        raise FileNotFoundError(
            f"{prebalance_path} not found. Baselines must come from the pre-SMOTE training "
            "split -- X_train_balanced.csv is NOT a substitute (see module docstring: SMOTE "
            "interpolation corrupts one-hot columns into fractional values)."
        )
    df = pd.read_csv(prebalance_path)

    missing_cols = [c for c in feature_order if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{prebalance_path} is missing {len(missing_cols)} column(s) present in "
            f"feature_order: {missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}. "
            "Baseline computation requires the exact same encoded columns the model/"
            "explainers were run against."
        )

    # Load the fitted transformer bundle directly to know unambiguously
    # which original features are categorical (rather than inferring from
    # group size).
    transformers_path = os.path.join(datasets_root, domain, dataset_name, "metadata", "fitted_transformers.joblib")
    bundle = joblib.load(transformers_path)
    categorical_set = set(bundle.get("categorical_cols") or [])

    col_index = {c: i for i, c in enumerate(feature_order)}
    baseline_vector = np.zeros(len(feature_order))
    baseline_report = {}

    for original_feat, cols in group_map.items():
        if original_feat in categorical_set:
            means = df[cols].mean()
            # guard against SMOTE-style fractional contamination even in the
            # "clean" prebalance set -- fail loudly rather than silently
            # taking a mode over fractional values.
            non_binary = [c for c in cols if not set(df[c].unique()).issubset({0.0, 1.0})]
            if non_binary:
                raise ValueError(
                    f"Column(s) {non_binary} in X_train_prebalance.csv for {original_feat} "
                    "contain non-binary values -- expected clean 0/1 one-hot in the prebalance "
                    "split. Refusing to compute a mode over this; investigate the preprocessing "
                    "pipeline for this dataset before proceeding."
                )
            mode_col = means.idxmax()
            for c in cols:
                baseline_vector[col_index[c]] = 1.0 if c == mode_col else 0.0
            baseline_report[original_feat] = {
                "type": "categorical", "mode_column": mode_col, "mode_frequency": float(means.max()),
            }
        else:
            median_val = df[cols[0]].median()
            baseline_vector[col_index[cols[0]]] = median_val
            baseline_report[original_feat] = {"type": "numerical", "median_scaled": float(median_val)}

    if logger:
        n_cat = sum(1 for r in baseline_report.values() if r["type"] == "categorical")
        n_num = len(baseline_report) - n_cat
        logger.log(f"Baseline vector computed for {domain}/{dataset_name} from "
                   f"X_train_prebalance.csv ({len(df)} rows): {n_cat} categorical, {n_num} numerical.")

    return baseline_vector, baseline_report


# ---------------------------------------------------------------------------
# 3. Masking primitive (whole original-feature groups, never per-dummy)
# ---------------------------------------------------------------------------

def mask_features(instance_vector, features_to_mask, group_map, feature_order, baseline_vector):
    """Returns a COPY of instance_vector with every model-input column
    belonging to each feature in features_to_mask replaced by its baseline
    value. Masks whole groups atomically.
    """
    col_index = {c: i for i, c in enumerate(feature_order)}
    masked = instance_vector.copy()
    for feat in features_to_mask:
        for c in group_map[feat]:
            masked[col_index[c]] = baseline_vector[col_index[c]]
    return masked


# ---------------------------------------------------------------------------
# 4. Per-instance ranking (reuses aggregate_to_original_features / parse_lime_feature)
# ---------------------------------------------------------------------------

def rank_features_shap(shap_row, feature_order, group_map):
    """shap_row: 1D array aligned to feature_order (signed). Returns original
    features ranked by |aggregated attribution| descending.
    """
    by_col = dict(zip(feature_order, np.abs(shap_row)))
    agg = aggregate_to_original_features(by_col, group_map)
    return [f for f, _ in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)]


def rank_features_lime(instance_lime_rows, feature_order, group_map):
    """instance_lime_rows: the subset of lime_explanations.csv for ONE
    instance_id (a DataFrame with 'feature'/'weight' columns). Returns
    (ranked_original_features, n_parse_failures).
    """
    by_col = {}
    n_failures = 0
    for _, r in instance_lime_rows.iterrows():
        try:
            col = parse_lime_feature(r["feature"], feature_order)
        except ValueError:
            n_failures += 1
            continue
        by_col[col] = by_col.get(col, 0.0) + abs(r["weight"])
    agg = aggregate_to_original_features(by_col, group_map)
    ranked = [f for f, _ in sorted(agg.items(), key=lambda kv: kv[1], reverse=True)]
    return ranked, n_failures


# ---------------------------------------------------------------------------
# 5. Comprehensiveness / Sufficiency / AOPC for one instance
# ---------------------------------------------------------------------------

def _predict_proba_for_class(model, X_row, target_class, feature_order):
    """feature_order is passed through (not just relying on positional array
    order) so predict_proba receives a properly-named single-row DataFrame --
    avoids sklearn's 'X does not have valid feature names' warning AND is a
    second layer of protection against silent column misalignment.
    """
    X_df = pd.DataFrame([X_row], columns=feature_order)
    proba = model.predict_proba(X_df)[0]
    return float(proba[target_class])


def compute_instance_fidelity(model, original_vector, ranked_features, group_map, feature_order,
                                baseline_vector, mask_fractions=MASK_FRACTIONS):
    """Returns (per_k_rows, aopc, target_class)."""
    n_features = len(ranked_features)
    original_df = pd.DataFrame([original_vector], columns=feature_order)
    original_proba_full = model.predict_proba(original_df)[0]
    target_class = int(np.argmax(original_proba_full))
    original_proba = float(original_proba_full[target_class])

    per_k_rows = []
    comp_drops, suff_drops = [], []

    for frac in mask_fractions:
        k = min(max(1, round(frac * n_features)), n_features)
        top_k = ranked_features[:k]
        remaining = ranked_features[k:]

        comp_masked = mask_features(original_vector, top_k, group_map, feature_order, baseline_vector)
        comp_proba = _predict_proba_for_class(model, comp_masked, target_class, feature_order)
        comp_drop = original_proba - comp_proba
        comp_drops.append(comp_drop)
        per_k_rows.append({
            "mask_fraction": frac, "condition": "comprehensiveness", "k_features": k,
            "original_proba": original_proba, "masked_proba": comp_proba, "drop": comp_drop,
        })

        suff_masked = mask_features(original_vector, remaining, group_map, feature_order, baseline_vector)
        suff_proba = _predict_proba_for_class(model, suff_masked, target_class, feature_order)
        suff_drop = original_proba - suff_proba
        suff_drops.append(suff_drop)
        per_k_rows.append({
            "mask_fraction": frac, "condition": "sufficiency", "k_features": k,
            "original_proba": original_proba, "masked_proba": suff_proba, "drop": suff_drop,
        })

    aopc = {
        "comprehensiveness": float(np.mean(comp_drops)),
        "sufficiency": float(np.mean(suff_drops)),
    }
    return per_k_rows, aopc, target_class


# ---------------------------------------------------------------------------
# 6. Orchestration
# ---------------------------------------------------------------------------

def run_fidelity(config):
    """Required config keys:
        dataset_name, domain, model_name, experiment_id

    Optional config keys:
        explainers        (default: ["shap", "lime"])
        datasets_root      (default: "Datasets")
        explanations_root  (default: "Explanations")
        models_root        (default: "Models")
        evaluation_root    (default: "Evaluation")
        fidelity_root       (default: "Fidelity" -- where masked_predictions.csv is written)
        mask_fractions      (default: MASK_FRACTIONS)
    """
    for key in ("dataset_name", "domain", "model_name", "experiment_id"):
        if key not in config:
            raise ValueError(f"CONFIG is missing required key '{key}'")

    domain, dataset_name, model_name = config["domain"], config["dataset_name"], config["model_name"]
    explainers = config.get("explainers", ["shap", "lime"])
    datasets_root = config.get("datasets_root", "Datasets")
    explanations_root = config.get("explanations_root", "Explanations")
    models_root = config.get("models_root", "Models")
    evaluation_root = config.get("evaluation_root", "Evaluation")
    fidelity_root = config.get("fidelity_root", "Fidelity")
    mask_fractions = config.get("mask_fractions", MASK_FRACTIONS)

    logger = Logger(os.path.join(evaluation_root, "logs"), filename="fidelity_log.txt")
    logger.section(f"FIDELITY: {dataset_name} x {model_name}")

    run_dir = os.path.join(explanations_root, domain, dataset_name, model_name)
    with open(os.path.join(run_dir, "metadata", "metadata.json")) as f:
        run_meta = json.load(f)
    dataset_characteristics = run_meta["dataset_characteristics"]
    selection_info = run_meta["selection_info"]
    random_state = selection_info["random_state"]
    instance_ids = selection_info["dataframe_indices"]

    model, schema = load_model(domain, dataset_name, model_name, models_root, logger)
    feature_order = schema["feature_order"]

    if feature_order != dataset_characteristics["feature_names"]:
        raise ValueError(
            f"Model's feature_schema.json feature_order does not match the explanation run's "
            f"dataset_characteristics.feature_names for {domain}/{dataset_name}/{model_name}. "
            "Masking a model with the wrong column alignment would silently corrupt every "
            "fidelity number -- refusing to proceed."
        )

    group_map = build_feature_group_map(dataset_name, domain, datasets_root, feature_order, logger)
    baseline_vector, baseline_report = compute_baseline_vector(
        domain, dataset_name, group_map, feature_order, datasets_root, logger,
    )

    x_explain_path = os.path.join(run_dir, "instances", "X_explain.csv")
    X_explain = pd.read_csv(x_explain_path)[feature_order]
    X_explain.index = list(instance_ids)

    fidelity_dir_base = os.path.join(fidelity_root, domain, dataset_name, model_name)
    now = dt.datetime.utcnow().isoformat() + "Z"
    all_metric_rows = []

    for explainer in explainers:
        masked_pred_rows = []
        n_parse_failures_total = 0

        if explainer == "shap":
            shap_dir = os.path.join(run_dir, "shap")
            values_path = os.path.join(shap_dir, "shap_values.npy")
            if not os.path.exists(values_path):
                logger.log(f"SKIP fidelity for shap: {values_path} not found.")
                continue
            shap_values = np.load(values_path)
            per_instance_ranking = {
                iid: rank_features_shap(shap_values[i], feature_order, group_map)
                for i, iid in enumerate(instance_ids)
            }
        elif explainer == "lime":
            lime_path = os.path.join(run_dir, "lime", "lime_explanations.csv")
            if not os.path.exists(lime_path):
                logger.log(f"SKIP fidelity for lime: {lime_path} not found.")
                continue
            lime_df = pd.read_csv(lime_path)
            per_instance_ranking = {}
            for iid, group in lime_df.groupby("instance_id"):
                ranked, n_fail = rank_features_lime(group, feature_order, group_map)
                per_instance_ranking[iid] = ranked
                n_parse_failures_total += n_fail
            if n_parse_failures_total:
                logger.log(f"WARNING: {n_parse_failures_total} LIME condition string(s) unparsed "
                           "across all instances -- those features excluded from ranking for their instance.")
        else:
            raise ValueError(f"Unknown explainer '{explainer}'")

        deterministic = (explainer == "shap")

        for iid in instance_ids:
            ranked_features = per_instance_ranking.get(iid)
            if not ranked_features:
                continue

            original_vector = X_explain.loc[iid].to_numpy(dtype=float)

            per_k_rows, aopc, target_class = compute_instance_fidelity(
                model, original_vector, ranked_features, group_map, feature_order,
                baseline_vector, mask_fractions,
            )

            for r in per_k_rows:
                masked_pred_rows.append({
                    "instance_id": iid, "explainer": explainer, "target_class": target_class,
                    **r,
                })

            base_row = {
                "experiment_id": config["experiment_id"], "domain": domain, "dataset": dataset_name,
                "model": model_name, "explainer": explainer, "metric_property": "Fidelity",
                "instance_id": iid, "repeat_idx": None, "perturbation_id": None, "repeat_seed": None,
                "deterministic": deterministic, "baseline_type": BASELINE_TYPE,
                "runtime_ms": None, "kernel_width": None, "background_size": None,
                "num_features": len(group_map), "num_instances": len(instance_ids),
                "random_state": random_state, "status": "ok", "timestamp": now,
            }
            for r in per_k_rows:
                all_metric_rows.append({
                    **base_row, "metric_name": r["condition"], "mask_fraction": r["mask_fraction"],
                    "value": r["drop"],
                })
            all_metric_rows.append({
                **base_row, "metric_name": "aopc_comprehensiveness", "mask_fraction": None,
                "value": aopc["comprehensiveness"],
            })
            all_metric_rows.append({
                **base_row, "metric_name": "aopc_sufficiency", "mask_fraction": None,
                "value": aopc["sufficiency"],
            })

        os.makedirs(os.path.join(fidelity_dir_base, explainer), exist_ok=True)
        masked_pred_path = os.path.join(fidelity_dir_base, explainer, "masked_predictions.csv")
        pd.DataFrame(masked_pred_rows).to_csv(masked_pred_path, index=False)
        logger.log(f"Wrote {len(masked_pred_rows)} rows to {masked_pred_path}")

    append_metrics(evaluation_root, all_metric_rows, logger)
    logger.section("DONE")
    logger.log(f"Fidelity: {len(all_metric_rows)} metrics_long rows written across {explainers}.")
    return all_metric_rows
