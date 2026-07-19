"""
Explanation evaluation, phase 3: Robustness.

Measures whether an explanation stays consistent when the INPUT is
perturbed slightly (a different, but realistically similar, instance) --
as opposed to Stability, which measures whether an explainer agrees with
ITSELF on repeated calls against the SAME, unperturbed instance.
Robustness therefore requires generating genuinely new explanations for
perturbed inputs; it cannot be computed from already-saved artifacts the
way Efficiency/Simplicity/Fidelity's ranking step can (Fidelity reuses
saved attributions -- Robustness cannot, because no attribution exists yet
for an instance that has never been explained).

RESOLVED PROTOCOL:

1. Perturbation is applied ONLY to numerical (single-column) features, with
   small Gaussian noise scaled to that feature's own training-set standard
   deviation: x'_i = x_i + N(0, (epsilon * std_i)^2), epsilon small
   (default 0.05, i.e. a nudge of roughly 5% of one standard deviation).
   Categorical (one-hot) groups are left untouched.

   Why: a "small perturbation" of a continuous feature is well-defined --
   nudge it slightly and it is still a realistic, nearby value. There is no
   equivalent SMALL perturbation of a one-hot categorical: the only two
   states are "this category" or "not this category," so any change is a
   full category swap -- a qualitatively different and much larger
   intervention than what Robustness is designed to test. Perturbing
   categoricals here would conflate Robustness with something closer to a
   counterfactual-sensitivity test, which is out of scope for this metric.
   This mirrors Fidelity's masking protocol in spirit (whole-group,
   never-partial treatment of categoricals) while applying it to a
   different operation.

2. ONE perturbed variant per instance, single run. Multiple repeated draws
   of the SAME perturbation-generation process is Stability's concern
   (explainer self-agreement), not Robustness's (input-sensitivity) -- see
   the companion stability module for that distinction, and the resolved
   decision documented in results/schema.md.

3. Both explainers are RE-RUN on the perturbed instance using the exact
   same configuration as the ORIGINAL explanation run (same LIME
   background data source, num_samples, num_features, random_state; same
   trained model). Configuration is read from each run's own metadata.json
   rather than hardcoded, so any config drift between datasets is inherited
   automatically rather than silently mismatched against what actually
   produced the saved explanations.

4. Comparison: attributions from the original vs. perturbed instance are
   both aggregated to original-feature granularity (reusing
   aggregate_to_original_features / parse_lime_feature from
   src/evaluation.py -- exactly as Fidelity does), then compared via:
       - Spearman rank correlation between the two full attribution-magnitude
         vectors (not just the top-k) -- sensitive to reordering anywhere
         in the ranking, not only at the top.
       - Top-k Jaccard overlap (k = min(5, ceil(n_features / 2))) -- a more
         interpretable, human-facing complement: "did the headline features
         stay the same."
   A robust explanation shows HIGH correlation / overlap: a small nudge to
   the input should not meaningfully reorder the ranked feature list.
"""

import os
import json
import datetime as dt

import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr

from src.utils import Logger
from src.evaluation import (
    build_feature_group_map,
    aggregate_to_original_features,
    parse_lime_feature,
    append_metrics,
)
from src.fidelity import load_model

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    import lime
    import lime.lime_tabular
    _HAS_LIME = True
except ImportError:
    _HAS_LIME = False

EPSILON = 0.05


# ---------------------------------------------------------------------------
# 1. Feature standard deviations (numerical features only, from prebalance data)
# ---------------------------------------------------------------------------

def compute_feature_stds(domain, dataset_name, categorical_set, feature_order,
                          datasets_root="Datasets", logger=None):
    """Returns {numerical_feature_name: std}, computed from
    X_train_prebalance.csv -- same clean, pre-SMOTE source Fidelity uses for
    its baselines (see fidelity.py module docstring for why prebalance, not
    balanced). Categorical features are absent from the returned dict; they
    are never perturbed (see module docstring, point 1).
    """
    prebalance_path = os.path.join(
        datasets_root, domain, dataset_name, "processed", "data", "X_train_prebalance.csv"
    )
    if not os.path.exists(prebalance_path):
        raise FileNotFoundError(
            f"{prebalance_path} not found. Perturbation magnitude requires each numerical "
            "feature's own training-set standard deviation from the pre-SMOTE split."
        )
    df = pd.read_csv(prebalance_path)
    stds = {}
    for col in feature_order:
        if col in categorical_set:
            continue
        if col not in df.columns:
            raise ValueError(f"Numerical feature '{col}' not found in {prebalance_path}.")
        stds[col] = float(df[col].std())
    if logger:
        logger.log(f"Computed std for {len(stds)} numerical feature(s) from {prebalance_path}.")
    return stds


# ---------------------------------------------------------------------------
# 2. Perturbation primitive
# ---------------------------------------------------------------------------

def perturb_instance(original_vector, group_map, categorical_set, feature_order, stds, epsilon, rng):
    """Returns a COPY of original_vector with Gaussian noise added to every
    numerical feature's column, scaled to epsilon * that feature's own
    training-set std. Categorical group columns are left byte-for-byte
    unchanged.
    """
    col_index = {c: i for i, c in enumerate(feature_order)}
    perturbed = original_vector.copy()
    for feat, cols in group_map.items():
        if feat in categorical_set:
            continue
        col = cols[0]
        idx = col_index[col]
        std = stds.get(feat, 0.0)
        if std > 0:
            perturbed[idx] = original_vector[idx] + rng.normal(0, epsilon * std)
    return perturbed


# ---------------------------------------------------------------------------
# 3. Live re-explanation (SHAP and LIME, matching the ORIGINAL run's config)
# ---------------------------------------------------------------------------

def _shap_row_to_positive_class(raw_shap_output, target_class):
    """shap.TreeExplainer's shap_values() return shape varies by
    shap version / model type: either a list of per-class arrays, or a
    single 3D array (n_instances, n_features, n_classes) for newer
    versions. Normalizes both to a 1D (n_features,) array for target_class,
    for a single-instance call.
    """
    if isinstance(raw_shap_output, list):
        return np.asarray(raw_shap_output[target_class])[0]
    arr = np.asarray(raw_shap_output)
    if arr.ndim == 3:
        return arr[0, :, target_class]
    return arr[0]  # binary-output style explainer returning a single array already


def get_shap_attribution(shap_explainer, vector, feature_order, target_class):
    row_df = pd.DataFrame([vector], columns=feature_order)
    raw = shap_explainer.shap_values(row_df)
    return _shap_row_to_positive_class(raw, target_class)


def build_lime_explainer(domain, dataset_name, model_name, feature_order, lime_metadata,
                          datasets_root="Datasets"):
    """Reconstructs a LimeTabularExplainer with the EXACT configuration the
    original explanation run used (background data source, random_state),
    read from that run's own lime/metadata.json rather than hardcoded --
    see module docstring, point 3.
    """
    if not _HAS_LIME:
        raise ImportError("lime is not installed; cannot generate perturbed-instance LIME explanations.")

    background_source = lime_metadata.get("background_data_source", "X_train_balanced")
    background_path = os.path.join(
        datasets_root, domain, dataset_name, "processed", "data", f"{background_source}.csv"
    )
    if not os.path.exists(background_path):
        raise FileNotFoundError(
            f"LIME background data source '{background_source}' recorded in the original run's "
            f"metadata.json not found at {background_path}. Robustness must use the SAME "
            "background the original explanations used, or the comparison would be confounded "
            "by a background-distribution change rather than isolating input-perturbation effects."
        )
    X_background = pd.read_csv(background_path)[feature_order]

    random_state = lime_metadata.get("lime_random_state", lime_metadata.get("instance_selection_random_state"))
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_background.values,
        feature_names=feature_order,
        class_names=["0", "1"],
        mode="classification",
        discretize_continuous=lime_metadata.get("discretize_continuous", True),
        random_state=random_state,
    )
    return explainer


def get_lime_attribution(lime_explainer, model, vector, feature_order, num_features, num_samples,
                          group_map):
    """Returns {original_feature: aggregated_abs_attribution}. Uses the same
    predict_proba-with-column-names wrapper pattern as src/explanations.py,
    to avoid sklearn's missing-feature-names warning on every perturbed
    sample LIME draws internally.
    """
    def _predict_proba(X):
        X_df = pd.DataFrame(np.atleast_2d(X), columns=feature_order)
        return model.predict_proba(X_df)

    exp = lime_explainer.explain_instance(vector, _predict_proba, num_features=num_features, num_samples=num_samples)
    by_col = {}
    n_failures = 0
    for cond, weight in exp.as_list():
        try:
            col = parse_lime_feature(cond, feature_order)
        except ValueError:
            n_failures += 1
            continue
        by_col[col] = by_col.get(col, 0.0) + abs(weight)
    agg = aggregate_to_original_features(by_col, group_map)
    return agg, n_failures


# ---------------------------------------------------------------------------
# 4. Ranking comparison
# ---------------------------------------------------------------------------

def compare_attributions(original_agg, perturbed_agg, all_original_features, top_k=None):
    """Both args are {original_feature: aggregated_abs_attribution} dicts,
    already at the same granularity. all_original_features fixes a common
    ordering so features present in one dict but absent from the other
    (e.g. a feature LIME didn't rank highly enough to report) are treated
    as zero attribution rather than silently dropped from the comparison.

    Returns (spearman_corr, top_k_jaccard, k_used).
    """
    if top_k is None:
        top_k = max(1, int(np.ceil(len(all_original_features) / 2)))
        top_k = min(top_k, 5)

    orig_vals = np.array([original_agg.get(f, 0.0) for f in all_original_features])
    pert_vals = np.array([perturbed_agg.get(f, 0.0) for f in all_original_features])

    if np.all(orig_vals == orig_vals[0]) or np.all(pert_vals == pert_vals[0]):
        # Spearman is undefined for a constant vector (zero variance) --
        # this should be rare (would mean an explainer assigned identical
        # attribution to every feature) but must be handled explicitly
        # rather than letting scipy silently return NaN with a warning.
        corr = np.nan
    else:
        corr, _ = spearmanr(orig_vals, pert_vals)

    orig_top = set(sorted(all_original_features, key=lambda f: original_agg.get(f, 0.0), reverse=True)[:top_k])
    pert_top = set(sorted(all_original_features, key=lambda f: perturbed_agg.get(f, 0.0), reverse=True)[:top_k])
    union = orig_top | pert_top
    jaccard = len(orig_top & pert_top) / len(union) if union else np.nan

    return float(corr) if not np.isnan(corr) else None, float(jaccard), top_k


# ---------------------------------------------------------------------------
# 5. Orchestration
# ---------------------------------------------------------------------------

def run_robustness(config):
    """Required config keys:
        dataset_name, domain, model_name, experiment_id

    Optional config keys:
        explainers        (default: ["shap", "lime"])
        datasets_root      (default: "Datasets")
        explanations_root  (default: "Explanations")
        models_root        (default: "Models")
        evaluation_root    (default: "Evaluation")
        epsilon             (default: EPSILON = 0.05)
        random_state        (default: taken from the original run's metadata)
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
    epsilon = config.get("epsilon", EPSILON)

    logger = Logger(os.path.join(evaluation_root, "logs"), filename="robustness_log.txt")
    logger.section(f"ROBUSTNESS: {dataset_name} x {model_name}")

    run_dir = os.path.join(explanations_root, domain, dataset_name, model_name)
    with open(os.path.join(run_dir, "metadata", "metadata.json")) as f:
        run_meta = json.load(f)
    selection_info = run_meta["selection_info"]
    random_state = config.get("random_state", selection_info["random_state"])
    instance_ids = selection_info["dataframe_indices"]

    model, schema = load_model(domain, dataset_name, model_name, models_root, logger)
    feature_order = schema["feature_order"]

    transformers_path = os.path.join(datasets_root, domain, dataset_name, "metadata", "fitted_transformers.joblib")
    categorical_set = set((joblib.load(transformers_path).get("categorical_cols") or []))

    group_map = build_feature_group_map(dataset_name, domain, datasets_root, feature_order, logger)
    stds = compute_feature_stds(domain, dataset_name, categorical_set, feature_order, datasets_root, logger)

    x_explain_path = os.path.join(run_dir, "instances", "X_explain.csv")
    X_explain = pd.read_csv(x_explain_path)[feature_order]
    X_explain.index = list(instance_ids)

    all_original_features = list(group_map.keys())
    now = dt.datetime.utcnow().isoformat() + "Z"
    all_metric_rows = []

    for explainer in explainers:
        if explainer == "shap":
            if not _HAS_SHAP:
                logger.log("SKIP robustness for shap: 'shap' package not installed.")
                continue
            shap_dir = os.path.join(run_dir, "shap")
            values_path = os.path.join(shap_dir, "shap_values.npy")
            if not os.path.exists(values_path):
                logger.log(f"SKIP robustness for shap: {values_path} not found.")
                continue
            original_shap_values = np.load(values_path)
            shap_explainer = shap.TreeExplainer(model)
            per_instance_original = {
                iid: aggregate_to_original_features(
                    dict(zip(feature_order, np.abs(original_shap_values[i]))), group_map,
                )
                for i, iid in enumerate(instance_ids)
            }
        elif explainer == "lime":
            if not _HAS_LIME:
                logger.log("SKIP robustness for lime: 'lime' package not installed.")
                continue
            lime_meta_path = os.path.join(run_dir, "lime", "metadata.json")
            lime_csv_path = os.path.join(run_dir, "lime", "lime_explanations.csv")
            if not (os.path.exists(lime_meta_path) and os.path.exists(lime_csv_path)):
                logger.log("SKIP robustness for lime: original LIME artifacts not found.")
                continue
            with open(lime_meta_path) as f:
                lime_metadata = json.load(f)
            lime_df = pd.read_csv(lime_csv_path)
            per_instance_original = {}
            for iid, grp in lime_df.groupby("instance_id"):
                by_col = {}
                for _, r in grp.iterrows():
                    try:
                        col = parse_lime_feature(r["feature"], feature_order)
                    except ValueError:
                        continue
                    by_col[col] = by_col.get(col, 0.0) + abs(r["weight"])
                per_instance_original[iid] = aggregate_to_original_features(by_col, group_map)
            lime_explainer = build_lime_explainer(domain, dataset_name, model_name, feature_order, lime_metadata, datasets_root)
            lime_num_features = lime_metadata.get("num_features_per_instance", len(feature_order))
            lime_num_samples = lime_metadata.get("num_samples_per_instance", 1000)
        else:
            raise ValueError(f"Unknown explainer '{explainer}'")

        deterministic = (explainer == "shap")
        rng = np.random.default_rng(random_state)
        n_parse_failures_total = 0

        for pos, iid in enumerate(instance_ids):
            if iid not in per_instance_original:
                continue
            original_vector = X_explain.loc[iid].to_numpy(dtype=float)
            # per-instance seed derived from the global random_state so runs
            # are reproducible AND every instance gets a different draw
            # (not the same noise vector repeated across all instances).
            instance_seed = int(random_state) * 100_003 + pos
            instance_rng = np.random.default_rng(instance_seed)
            perturbed_vector = perturb_instance(
                original_vector, group_map, categorical_set, feature_order, stds, epsilon, instance_rng,
            )

            if explainer == "shap":
                target_class = int(np.argmax(model.predict_proba(
                    pd.DataFrame([original_vector], columns=feature_order))[0]))
                perturbed_agg = aggregate_to_original_features(
                    dict(zip(feature_order, np.abs(
                        get_shap_attribution(shap_explainer, perturbed_vector, feature_order, target_class)
                    ))),
                    group_map,
                )
            else:
                perturbed_agg, n_fail = get_lime_attribution(
                    lime_explainer, model, perturbed_vector, feature_order,
                    lime_num_features, lime_num_samples, group_map,
                )
                n_parse_failures_total += n_fail

            corr, jaccard, k_used = compare_attributions(
                per_instance_original[iid], perturbed_agg, all_original_features,
            )

            base_row = {
                "experiment_id": config["experiment_id"], "domain": domain, "dataset": dataset_name,
                "model": model_name, "explainer": explainer, "metric_property": "Robustness",
                "instance_id": iid, "repeat_idx": None, "perturbation_id": 0, "repeat_seed": instance_seed,
                "deterministic": deterministic, "baseline_type": None,
                "mask_fraction": None, "runtime_ms": None, "kernel_width": None, "background_size": None,
                "num_features": len(group_map), "num_instances": len(instance_ids),
                "random_state": random_state, "status": "ok", "timestamp": now,
            }
            all_metric_rows.append({**base_row, "metric_name": "spearman_rank_correlation", "value": corr})
            all_metric_rows.append({
                **base_row, "metric_name": f"top_{k_used}_jaccard_overlap", "value": jaccard,
            })

        if explainer == "lime" and n_parse_failures_total:
            logger.log(f"WARNING: {n_parse_failures_total} LIME condition string(s) unparsed across "
                       "perturbed-instance explanations.")
        logger.log(f"{explainer}: robustness computed for {sum(1 for r in all_metric_rows if r['explainer']==explainer) // 2} instance(s).")

    append_metrics(evaluation_root, all_metric_rows, logger)
    logger.section("DONE")
    logger.log(f"Robustness: {len(all_metric_rows)} metrics_long rows written across {explainers}.")
    return all_metric_rows
