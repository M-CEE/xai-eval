"""
Explanation generation: produces SHAP and LIME explanation artifacts for a
trained model, versioned with enough metadata (model reference, explainer
config, library versions, random seed) that every downstream metric,
statistical test, or figure in the paper can be computed from these exact
artifacts rather than re-running an explainer with slightly different
settings each time.

    1.  Configuration              (one CONFIG dict per dataset x model run)
    2.  Load model + data          (trained model, feature schema, X_test/y_test,
                                     X_train_balanced as LIME's background)
    3.  Select explanation subset  (stratified sample of the TEST set --
                                     SAVED AND REUSED across every explainer
                                     and every future re-run)
    3b. Select case instances      (one correctly- and one incorrectly-
                                     classified instance FROM the subset above,
                                     shared by both explainers' waterfall/local
                                     plots -- also saved and reused)
    4.  SHAP explanations          (shap.TreeExplainer, deterministic) +
                                     beeswarm/bar/waterfall plots + top-10
                                     feature importance CSV
    5.  LIME explanations          (lime.lime_tabular.LimeTabularExplainer,
                                     seeded) + the two case instances' standard
                                     local-explanation plots
    6.  Metadata + narrative log

Instance selection rule (fixed across every dataset x model run):
    - test set >= 1000 rows -> stratified random sample of 500 rows
    - test set <  1000 rows -> use the entire test set
    - class proportions preserved via stratified sampling
    - the SAME selected rows are used for every explainer

"Generate once, version it" is enforced by treating the instance selection,
case-instance picks, and each explainer's output as artifacts to check for
before recomputing: if already saved, they're loaded and reused (not
resampled/reselected/recomputed) unless force=True. This means re-running
this module later (e.g. to add LIME after already having SHAP) never
silently explains a different set of rows -- or picks different case
instances -- than an earlier run did.

On joining explanations back to other data (see instances/metadata.json's
"dataframe_indices" field): these are 0-based row positions into
Datasets/<domain>/<dataset_name>/processed/data/X_test.csv (and y_test.csv),
and they align 1:1 with Models/<domain>/<dataset_name>/<model_name>/metadata/
test_predictions.csv row-for-row -- so explanations CAN be joined to
predictions via this index. They can NOT currently be joined back to
pre-preprocessing raw records: preprocessing.py's train/test split doesn't
preserve original row identity (e.g. dropped ID columns like LoanID aren't
persisted anywhere downstream), so there is no surviving key back to the
source data. X_explain.csv already holds the actual (imputed/encoded/scaled)
feature values the model saw, which is usually what's needed to describe a
case in the paper.

This module mirrors src/preprocessing.py / src/training.py's orchestration
style: one function per numbered step, a single run_explanations(config) that
calls them in order, and a Logger for narrative output.
"""

import os
import json
import time
import platform
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sklearn

from sklearn.model_selection import train_test_split as _tts

from src.utils import Logger, FigureRegistry, set_plot_style

try:
    import xgboost
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

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

try:
    import skops.io as sio
    _HAS_SKOPS = True
except ImportError:
    _HAS_SKOPS = False

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


DEFAULT_RANDOM_STATE = 42
DEFAULT_SMALL_TEST_THRESHOLD = 1_000   # test sets smaller than this: explain ALL of it
DEFAULT_N_INSTANCES = 500              # otherwise: stratified sample of this many rows
DEFAULT_LIME_NUM_SAMPLES = 1_000       # perturbed samples LIME generates PER explained instance
DEFAULT_LIME_NUM_FEATURES = 50         # cap on features reported per LIME explanation
DEFAULT_TOP_N_FEATURES = 10            # size of the exported feature-importance CSV


def setup_explanation_dirs(dataset_name, domain, model_name, explanations_root="Explanations"):
    """Mirrors setup_model_dirs from src/utils.py, for the Explanations/ tree:

        Explanations/<domain>/<dataset_name>/<model_name>/
            instances/, shap/, shap/plots/, lime/, lime/plots/, logs/, metadata/
    """
    root = os.path.join(explanations_root, domain, dataset_name, model_name)
    paths = {
        "root": root,
        "instances": os.path.join(root, "instances"),
        "shap": os.path.join(root, "shap"),
        "shap_plots": os.path.join(root, "shap", "plots"),
        "lime": os.path.join(root, "lime"),
        "lime_plots": os.path.join(root, "lime", "plots"),
        "logs": os.path.join(root, "logs"),
        "metadata": os.path.join(root, "metadata"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# 2. Load model + data
# ---------------------------------------------------------------------------

def _load_model(model_dir, logger):
    skops_path = os.path.join(model_dir, "model.skops")
    joblib_path = os.path.join(model_dir, "model.joblib")
    if os.path.exists(skops_path):
        if not _HAS_SKOPS:
            raise ImportError("Model was saved via skops but skops isn't installed to load it.")
        untrusted = sio.get_untrusted_types(file=skops_path)
        model = sio.load(skops_path, trusted=untrusted)
        logger.log(f"Loaded model: {skops_path} (trusted types: {untrusted})")
        return model
    if os.path.exists(joblib_path):
        model = joblib.load(joblib_path)
        logger.log(f"Loaded model: {joblib_path}")
        return model
    raise FileNotFoundError(f"No model.skops or model.joblib found in {model_dir}")


def load_model_and_data(dataset_name, domain, model_name, datasets_root, models_root, logger):
    logger.section("2. LOAD MODEL + DATA")

    model_dir = os.path.join(models_root, domain, dataset_name, model_name, "model")
    metadata_dir = os.path.join(models_root, domain, dataset_name, model_name, "metadata")
    model = _load_model(model_dir, logger)

    with open(os.path.join(model_dir, "feature_schema.json")) as f:
        schema = json.load(f)
    feature_order = schema["feature_order"]

    with open(os.path.join(metadata_dir, "model_info.json")) as f:
        model_info = json.load(f)
    best_hp_path = os.path.join(metadata_dir, "best_hyperparameters.json")
    best_hyperparameters = None
    if os.path.exists(best_hp_path):
        with open(best_hp_path) as f:
            best_hyperparameters = json.load(f).get("best_params")

    # Model performance, pulled in so anyone auditing ONLY an explainer's
    # metadata.json (e.g. a downloaded shap/ folder in isolation) can see how
    # good the underlying model actually was, without needing Models/ too.
    test_metrics_path = os.path.join(metadata_dir, "test_metrics.json")
    test_performance = None
    if os.path.exists(test_metrics_path):
        with open(test_metrics_path) as f:
            test_performance = json.load(f)

    target = model_info.get("target")
    if target is None:
        with open(os.path.join(datasets_root, domain, dataset_name, "metadata", "dataset_info.json")) as f:
            target = json.load(f)["target"]

    data_dir = os.path.join(datasets_root, domain, dataset_name, "processed", "data")
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))[feature_order]
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))[target]
    X_train_balanced = pd.read_csv(os.path.join(data_dir, "X_train_balanced.csv"))[feature_order]

    logger.log(f"Model: {model_info.get('model_type')} on {dataset_name} ({domain})")
    logger.log(f"Feature order ({len(feature_order)}): {feature_order}")
    logger.log(f"X_test: {X_test.shape}, X_train_balanced (LIME background): {X_train_balanced.shape}")
    logger.log(f"Model test performance: {test_performance}")

    model_reference = {
        "dataset_name": dataset_name,
        "domain": domain,
        "model_type": model_info.get("model_type"),
        "model_name": model_name,
        "model_path": model_info.get("model_path"),
        "training_random_state": model_info.get("random_state"),
        "training_scoring": model_info.get("scoring"),
        "best_hyperparameters": best_hyperparameters,
        "hub_repo_id": model_info.get("hub_repo_id"),
        "hub_commit_sha": model_info.get("hub_commit_sha"),
        "test_performance": test_performance,
    }

    dataset_characteristics = {
        "dataset_name": dataset_name,
        "domain": domain,
        "n_features": len(feature_order),
        "feature_names": feature_order,
        "n_test_total": len(X_test),
        "n_train_balanced_total": len(X_train_balanced),
        "test_class_distribution": y_test.value_counts(normalize=True).sort_index().to_dict(),
    }

    return {
        "model": model,
        "model_type": model_info.get("model_type"),
        "feature_order": feature_order,
        "target": target,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_balanced": X_train_balanced,
        "model_reference": model_reference,
        "dataset_characteristics": dataset_characteristics,
    }


# ---------------------------------------------------------------------------
# 3. Select (or reuse) the explanation subset
# ---------------------------------------------------------------------------

def select_or_load_instances(paths, X_test, y_test, target, dataset_name, domain, model_name,
                               random_state, small_test_threshold, n_instances, logger, force=False):
    """Loads a previously-saved instance selection if one exists (so re-runs
    of this module -- e.g. adding LIME after SHAP already exists -- explain
    the SAME rows), otherwise draws a fresh stratified sample and saves it.
    force=True always draws a fresh sample, overwriting any saved one (use
    deliberately -- this invalidates comparability with any explanations
    already generated from the old selection).

    Saved as instances/metadata.json. "dataframe_indices" are 0-based row
    positions into X_test.csv/y_test.csv -- see this module's docstring for
    exactly what these can and can't be joined back to.
    """
    logger.section("3. SELECT EXPLANATION SUBSET")
    meta_path = os.path.join(paths["instances"], "metadata.json")
    X_path = os.path.join(paths["instances"], "X_explain.csv")
    y_path = os.path.join(paths["instances"], "y_explain.csv")

    if os.path.exists(meta_path) and not force:
        with open(meta_path) as f:
            selection_info = json.load(f)
        X_explain = pd.read_csv(X_path)
        y_explain = pd.read_csv(y_path)[target]
        logger.log(
            f"Reusing existing saved instance selection from {meta_path} "
            f"(n={selection_info['num_instances']}, selected {selection_info['generated_at_utc']}). "
            "Set force=True to draw a fresh sample instead (breaks comparability with existing explanations)."
        )
        return X_explain, y_explain, selection_info

    n_test = len(X_test)
    if n_test < small_test_threshold:
        instance_ids = list(range(n_test))
        sampling_strategy = "full_test_set"
        X_explain, y_explain = X_test.reset_index(drop=True), y_test.reset_index(drop=True)
    else:
        n_select = min(n_instances, n_test)
        X_explain, _, y_explain, _ = _tts(
            X_test.reset_index(drop=True), y_test.reset_index(drop=True),
            train_size=n_select, stratify=y_test, random_state=random_state,
        )
        instance_ids = X_explain.index.tolist()   # positions in the reset-index X_test
        X_explain = X_explain.reset_index(drop=True)
        y_explain = y_explain.reset_index(drop=True)
        sampling_strategy = "stratified_random"

    class_dist_full = y_test.value_counts(normalize=True).sort_index().to_dict()
    class_dist_selected = y_explain.value_counts(normalize=True).sort_index().to_dict()

    selection_info = {
        "dataset": dataset_name,
        "domain": domain,
        "model": model_name,
        "split": "test",
        "sampling_strategy": sampling_strategy,
        "random_state": random_state,
        "num_instances": len(instance_ids),
        "num_test_total": n_test,
        "small_test_threshold": small_test_threshold,
        "target_n_instances": n_instances,
        "dataframe_indices": instance_ids,
        "class_distribution_full_test": class_dist_full,
        "class_distribution_selected": class_dist_selected,
        "join_key_note": (
            "dataframe_indices are 0-based row positions into "
            "Datasets/<domain>/<dataset_name>/processed/data/X_test.csv (and y_test.csv). "
            "They align 1:1 with Models/<domain>/<dataset_name>/<model_name>/metadata/test_predictions.csv "
            "row-for-row, so explanations CAN be joined to predictions via this index. They can NOT be "
            "joined back to pre-preprocessing raw records: the train/test split does not preserve original "
            "row identity (e.g. dropped ID columns are not persisted anywhere downstream). X_explain.csv "
            "already holds the actual feature values the model saw."
        ),
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
    }

    with open(meta_path, "w") as f:
        json.dump(selection_info, f, indent=2, default=str)
    X_explain.to_csv(X_path, index=False)
    y_explain.to_frame(name=target).to_csv(y_path, index=False)

    logger.log(f"Selected {len(instance_ids)} instances ({sampling_strategy})")
    logger.log(f"Class distribution -- full test: {class_dist_full}, selected: {class_dist_selected}")
    logger.log(f"Saved: {meta_path}, {X_path}, {y_path}")

    return X_explain, y_explain, selection_info


# ---------------------------------------------------------------------------
# 3b. Select (or reuse) case instances -- one correct, one incorrect
# ---------------------------------------------------------------------------

def _select_case_position(mask, confidence, rule):
    """Given a boolean mask (correct or incorrect instances) and a per-instance
    confidence array (confidence in the model's OWN predicted class), returns
    the position to use as the case instance under the given rule.
    """
    positions = np.where(mask)[0]
    if len(positions) == 0:
        return None
    if rule == "first":
        return int(positions[0])
    if rule == "highest_confidence":
        return int(positions[np.argmax(confidence[positions])])
    if rule == "median_confidence":
        local_conf = confidence[positions]
        median_val = np.median(local_conf)
        return int(positions[np.argmin(np.abs(local_conf - median_val))])
    raise ValueError(f"Unknown case_selection_rule '{rule}'. Options: first, highest_confidence, median_confidence")


def select_or_load_case_instances(paths, model, X_explain, y_explain, selection_info, logger,
                                    rule="highest_confidence", force=False):
    """Picks one correctly-classified and one incorrectly-classified instance
    FROM the explanation subset, for the waterfall / standard local-explanation
    plots. Saved and reused across explainers and future re-runs, same
    idempotency pattern as the instance selection above.

    rule controls WHICH correct/incorrect instance gets picked:
    - "first": first correct/incorrect instance in the saved X_explain order.
      Arbitrary -- whatever the sampling happened to put first. Simple and
      fully deterministic, but not chosen for any property that makes it a
      good illustrative example.
    - "highest_confidence" (default): the correct instance the model was MOST
      confident about, and the incorrect instance the model was MOST
      confident (and wrong) about. This is the more useful default for a
      paper: the confident-correct case is the clearest illustration of what
      the model considers strong evidence, and the confident-incorrect case
      is the most diagnostically interesting failure mode -- a model that's
      confidently wrong is a bigger concern (especially in healthcare/finance)
      than one that's uncertain and wrong, and it's the case most worth
      explaining.
    - "median_confidence": the correct/incorrect instance closest to the
      MEDIAN confidence within its group, i.e. a "typical" rather than
      extreme example. Better suited if the paper's point is "here's what a
      representative prediction looks like" rather than "here's the model at
      its best/worst."
    Confidence is defined as the predicted probability of the model's OWN
    predicted class (proba_class1 if predicted class is 1, else
    1 - proba_class1) -- so it's comparable across both classes and both
    correct/incorrect groups.

    If the model gets everything right on the sampled subset (possible on a
    small/easy dataset), "incorrectly_classified" is saved as null and the
    corresponding plots are skipped later, with a note logged.
    """
    logger.section("3B. SELECT CASE INSTANCES")
    path = os.path.join(paths["instances"], "case_instances.json")
    if os.path.exists(path) and not force:
        with open(path) as f:
            case_info = json.load(f)
        logger.log(f"Reusing existing saved case instances from {path}.")
        if case_info.get("selection_rule") != rule:
            logger.log(
                f"WARNING: requested case_selection_rule='{rule}' but the saved case instances were picked "
                f"under rule='{case_info.get('selection_rule')}'. Reusing the saved pick anyway (for "
                "comparability with any existing plots) -- pass force=True if you actually want the new rule applied."
            )
        return case_info

    y_pred = model.predict(X_explain)
    pos_idx = list(model.classes_).index(1) if hasattr(model, "classes_") else 1
    y_proba = model.predict_proba(X_explain)[:, pos_idx]
    correct_mask = (y_pred == y_explain.values)
    confidence = np.where(y_pred == 1, y_proba, 1 - y_proba)   # confidence in the model's OWN predicted class

    def _describe(pos):
        return {
            "position_in_X_explain": int(pos),
            "dataframe_index": int(selection_info["dataframe_indices"][pos]),
            "y_true": int(y_explain.iloc[pos]),
            "y_pred": int(y_pred[pos]),
            "predicted_proba_class1": float(y_proba[pos]),
            "confidence_in_predicted_class": float(confidence[pos]),
        }

    correct_pos = _select_case_position(correct_mask, confidence, rule)
    incorrect_pos = _select_case_position(~correct_mask, confidence, rule)

    case_info = {
        "correctly_classified": _describe(correct_pos) if correct_pos is not None else None,
        "incorrectly_classified": _describe(incorrect_pos) if incorrect_pos is not None else None,
        "selection_rule": rule,
        "confidence_definition": "predicted probability of the model's own predicted class",
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(path, "w") as f:
        json.dump(case_info, f, indent=2, default=str)

    logger.log(f"Case selection rule: {rule}")
    logger.log(f"Correctly classified case: {case_info['correctly_classified']}")
    if case_info["incorrectly_classified"] is None:
        logger.log("No incorrectly-classified instance found in the explanation subset -- "
                   "waterfall_case2/lime case2 plots will be skipped.")
    else:
        logger.log(f"Incorrectly classified case: {case_info['incorrectly_classified']}")
    logger.log(f"Saved: {path}")

    return case_info


# ---------------------------------------------------------------------------
# 4. SHAP explanations
# ---------------------------------------------------------------------------

def compute_and_save_shap(paths, model, model_reference, dataset_characteristics, X_explain, feature_order,
                            selection_info, case_info, logger, top_n=DEFAULT_TOP_N_FEATURES, force=False):
    """shap.TreeExplainer with feature_perturbation='tree_path_dependent' (the
    default when no background data is passed): exact for tree ensembles,
    fully deterministic, no background-sample randomness to fix/report.
    RF and XGB are the only two model types this project trains, both
    supported by TreeExplainer.

    Saves shap_values.npy/base_values.npy (raw arrays), feature_importance.csv
    (top-N features by mean |SHAP value|), and three plot types under plots/:
    beeswarm summary, bar importance, and waterfalls for the two case instances.
    """
    values_path = os.path.join(paths["shap"], "shap_values.npy")
    expected_plots = ["summary.png", "bar.png"]   # waterfalls are conditional on case availability, not required here
    plots_exist = all(os.path.exists(os.path.join(paths["shap_plots"], p)) for p in expected_plots)
    if os.path.exists(values_path) and plots_exist and not force:
        logger.log(f"SHAP artifacts (including plots) already exist under {paths['shap']} -- skipping (force=True to regenerate).")
        return
    if os.path.exists(values_path) and not plots_exist:
        logger.log(
            f"SHAP arrays exist at {values_path} but plots are missing (likely generated by an older version "
            "of this module) -- regenerating the full SHAP step, including plots. TreeExplainer is fast/"
            "deterministic, so this is cheap and won't change the saved values."
        )

    logger.section("4. SHAP EXPLANATIONS")
    if not _HAS_SHAP:
        raise ImportError("shap is not installed; cannot generate SHAP explanations.")

    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    raw = explainer(X_explain)
    elapsed = time.time() - t0

    values = np.asarray(raw.values)
    base_values = np.asarray(raw.base_values)
    # shap's output shape varies by shap version / model type: either
    # (n_samples, n_features) already for the positive class, or
    # (n_samples, n_features, n_classes) with one slice per class. Handle both
    # so the saved artifact is always (n_samples, n_features) for class 1.
    if values.ndim == 3:
        pos_idx = 1 if values.shape[2] > 1 else 0
        values = values[:, :, pos_idx]
        if base_values.ndim > 1:
            base_values = base_values[:, pos_idx]

    np.save(values_path, values)
    np.save(os.path.join(paths["shap"], "base_values.npy"), base_values)

    # Top-N feature importance by mean |SHAP value|
    mean_abs = np.abs(values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_order, "mean_abs_shap_value": mean_abs})
        .sort_values("mean_abs_shap_value", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.insert(0, "rank", np.arange(1, len(importance_df) + 1))
    importance_df.head(top_n).to_csv(os.path.join(paths["shap"], "feature_importance.csv"), index=False)

    metadata = {
        "explainer_library": "shap",
        "explainer_type": "TreeExplainer",
        "shap_version": shap.__version__,
        "feature_perturbation": "tree_path_dependent",
        "background_data": None,
        "output_class_explained": "positive (1)",
        "n_instances_explained": len(X_explain),
        "n_features": len(feature_order),
        "feature_order": feature_order,
        "top_n_features_exported": top_n,
        "instance_selection_random_state": selection_info["random_state"],
        "dataset_characteristics": dataset_characteristics,
        "model_reference": model_reference,
        "library_versions": {
            "shap": shap.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__ if _HAS_XGB else None,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
        },
        "runtime_seconds": round(elapsed, 2),
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(paths["shap"], "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.log(f"SHAP values: {values.shape}, computed in {elapsed:.1f}s")
    logger.log(f"Saved: {values_path}, base_values.npy, feature_importance.csv, metadata.json")

    # --- Plots ---
    set_plot_style()
    fig_registry = FigureRegistry(paths["shap_plots"], logger)
    explanation_obj = shap.Explanation(
        values=values, base_values=base_values, data=X_explain.values, feature_names=feature_order,
    )
    max_display = min(20, len(feature_order))

    shap.plots.beeswarm(explanation_obj, show=False, max_display=max_display)
    fig_registry.save(plt.gcf(), "summary.png", "SHAP beeswarm summary plot (top features by impact).")

    shap.plots.bar(explanation_obj, show=False, max_display=max_display)
    fig_registry.save(plt.gcf(), "bar.png", "Features ranked by mean |SHAP value|.")

    for case_key, plot_name in [("correctly_classified", "waterfall_case1.png"),
                                  ("incorrectly_classified", "waterfall_case2.png")]:
        case = case_info.get(case_key)
        if case is None:
            logger.log(f"Skipping {plot_name} -- no {case_key} instance available.")
            continue
        pos = case["position_in_X_explain"]
        shap.plots.waterfall(explanation_obj[pos], show=False, max_display=min(15, len(feature_order)))
        fig_registry.save(
            plt.gcf(), plot_name,
            f"SHAP waterfall for a {'correctly' if case_key == 'correctly_classified' else 'incorrectly'} "
            f"classified test instance (y_true={case['y_true']}, y_pred={case['y_pred']}, "
            f"p(class 1)={case['predicted_proba_class1']:.3f})."
        )


# ---------------------------------------------------------------------------
# 5. LIME explanations
# ---------------------------------------------------------------------------

def _save_lime_case_plots(paths, case_explanations, case_info, logger):
    """Saves the two case instances' standard LIME local-explanation plots
    from already-computed lime Explanation objects (case_explanations:
    "case1"/"case2" -> Explanation). Pulled out as its own function so it can
    be called either from the main explain-all-instances loop, or cheaply on
    its own (just 2 explain_instance() calls, not the full n_instances loop)
    when the main LIME data already exists but plots don't yet.
    """
    set_plot_style()
    fig_registry = FigureRegistry(paths["lime_plots"], logger)
    for case_key, (plot_name, label) in {
        "correctly_classified": ("case1.png", "correctly"),
        "incorrectly_classified": ("case2.png", "incorrectly"),
    }.items():
        exp = case_explanations.get("case1" if case_key == "correctly_classified" else "case2")
        case = case_info.get(case_key)
        if exp is None or case is None:
            logger.log(f"Skipping {plot_name} -- no {case_key} instance available.")
            continue
        fig = exp.as_pyplot_figure()
        fig_registry.save(
            fig, plot_name,
            f"Standard LIME local explanation for a {label} classified test instance "
            f"(y_true={case['y_true']}, y_pred={case['y_pred']}, "
            f"p(class 1)={case['predicted_proba_class1']:.3f})."
        )


def compute_and_save_lime(paths, model, model_reference, dataset_characteristics, X_train_balanced, X_explain,
                            feature_order, selection_info, case_info, random_state, logger,
                            num_samples=DEFAULT_LIME_NUM_SAMPLES, num_features=DEFAULT_LIME_NUM_FEATURES,
                            force=False):
    """lime.lime_tabular.LimeTabularExplainer, one explain_instance() call per
    selected row. LIME is inherently per-instance and perturbation-based --
    num_samples perturbed points are generated and scored by the model for
    EVERY explained instance, so runtime scales as
    n_instances x num_samples x (model prediction cost). Both are exposed as
    config knobs precisely because this can get slow on the higher-cardinality
    datasets -- lower num_samples if needed, and document that you did.

    Also saves the two case instances' "standard" LIME plots
    (Explanation.as_pyplot_figure()) under plots/ -- reused from the same
    explain_instance() calls made during the main loop, not recomputed.
    """
    summary_path = os.path.join(paths["lime"], "lime_instance_summary.csv")
    expected_plots = []
    if case_info.get("correctly_classified") is not None:
        expected_plots.append("case1.png")
    if case_info.get("incorrectly_classified") is not None:
        expected_plots.append("case2.png")
    plots_exist = all(os.path.exists(os.path.join(paths["lime_plots"], p)) for p in expected_plots)

    if os.path.exists(summary_path) and plots_exist and not force:
        logger.log(f"LIME artifacts (including plots) already exist under {paths['lime']} -- skipping (force=True to regenerate).")
        return

    if os.path.exists(summary_path) and not plots_exist and not force:
        # Main LIME data already exists (from an older version of this module
        # that didn't produce plots yet) -- avoid re-running the full,
        # expensive n_instances loop. Just re-run explain_instance() for the
        # 2 case rows to produce the missing plots.
        logger.log(
            f"LIME data exists at {summary_path} but case plots are missing -- generating just those "
            "(2 explain_instance() calls) instead of re-running the full instance loop."
        )
        if not _HAS_LIME:
            raise ImportError("lime is not installed; cannot generate LIME explanations.")

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_balanced.values, feature_names=feature_order,
            class_names=["0", "1"], mode="classification", discretize_continuous=True,
            random_state=random_state,
        )

        def _predict_proba(X):
            X_df = pd.DataFrame(np.atleast_2d(X), columns=feature_order)
            return model.predict_proba(X_df)

        num_features_used = min(num_features, len(feature_order))
        case_explanations = {}
        for case_key, case_label in [("correctly_classified", "case1"), ("incorrectly_classified", "case2")]:
            case = case_info.get(case_key)
            if case is None:
                continue
            row = X_explain.iloc[case["position_in_X_explain"]].values
            case_explanations[case_label] = explainer.explain_instance(
                row, _predict_proba, num_features=num_features_used, num_samples=num_samples,
            )
        _save_lime_case_plots(paths, case_explanations, case_info, logger)
        return

    logger.section("5. LIME EXPLANATIONS")
    if not _HAS_LIME:
        raise ImportError("lime is not installed; cannot generate LIME explanations.")

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_balanced.values,
        feature_names=feature_order,
        class_names=["0", "1"],
        mode="classification",
        discretize_continuous=True,
        random_state=random_state,
    )

    # LIME perturbs instances as raw numpy arrays internally, which triggers
    # sklearn's "X does not have valid feature names" warning on every single
    # perturbed sample (thousands of times per explained instance) since the
    # model was trained on a DataFrame with named columns. Wrapping predict_proba
    # to reattach column names avoids that noise without changing any predictions.
    def _predict_proba(X):
        X_df = pd.DataFrame(np.atleast_2d(X), columns=feature_order)
        return model.predict_proba(X_df)

    case_positions = {
        case_info[k]["position_in_X_explain"]: ("case1" if k == "correctly_classified" else "case2")
        for k in ("correctly_classified", "incorrectly_classified") if case_info.get(k) is not None
    }

    num_features_used = min(num_features, len(feature_order))
    long_rows = []      # instance_id, feature, weight
    summary_rows = []   # instance_id, predicted_proba_1, intercept, local_pred, score
    case_explanations = {}   # "case1"/"case2" -> lime Explanation object, for plotting below

    t0 = time.time()
    iterator = range(len(X_explain))
    if _HAS_TQDM:
        iterator = tqdm(iterator, desc="LIME explanations", disable=False)

    for i in iterator:
        row = X_explain.iloc[i].values
        exp = explainer.explain_instance(
            row, _predict_proba, num_features=num_features_used, num_samples=num_samples,
        )
        instance_id = selection_info["dataframe_indices"][i]
        for feature_desc, weight in exp.as_list():
            long_rows.append({"instance_id": instance_id, "feature": feature_desc, "weight": weight})

        local_pred = exp.local_pred[0] if hasattr(exp, "local_pred") and exp.local_pred is not None else None
        intercept = exp.intercept[1] if isinstance(exp.intercept, dict) else exp.intercept
        score = exp.score[1] if isinstance(getattr(exp, "score", None), dict) else getattr(exp, "score", None)
        summary_rows.append({
            "instance_id": instance_id,
            "predicted_proba_class1": float(_predict_proba(row.reshape(1, -1))[0, 1]),
            "intercept": intercept,
            "local_prediction": local_pred,
            "local_model_r2_score": score,
        })

        if i in case_positions:
            case_explanations[case_positions[i]] = exp
    elapsed = time.time() - t0

    pd.DataFrame(long_rows).to_csv(os.path.join(paths["lime"], "lime_explanations.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    metadata = {
        "explainer_library": "lime",
        "explainer_type": "LimeTabularExplainer",
        "lime_version": getattr(lime, "__version__", "unknown"),
        "mode": "classification",
        "discretize_continuous": True,
        "num_samples_per_instance": num_samples,
        "num_features_per_instance": num_features_used,
        "background_data_source": "X_train_balanced",
        "background_data_rows": len(X_train_balanced),
        "n_instances_explained": len(X_explain),
        "instance_selection_random_state": selection_info["random_state"],
        "lime_random_state": random_state,
        "dataset_characteristics": dataset_characteristics,
        "model_reference": model_reference,
        "library_versions": {
            "lime": getattr(lime, "__version__", "unknown"),
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__ if _HAS_XGB else None,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
        },
        "runtime_seconds": round(elapsed, 2),
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(paths["lime"], "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.log(f"LIME explanations for {len(X_explain)} instances, computed in {elapsed:.1f}s")
    logger.log("Saved: lime_explanations.csv, lime_instance_summary.csv, metadata.json")

    # --- Plots: the two case instances' standard LIME local-explanation plots ---
    _save_lime_case_plots(paths, case_explanations, case_info, logger)


# ---------------------------------------------------------------------------
# Master experiment protocol (one file, shared across the WHOLE experiment)
# ---------------------------------------------------------------------------

def write_master_protocol(explanations_root, protocol_fields, run_identity, logger):
    """Maintains Explanations/experiment_protocol.json: the single record of
    the FIXED rules shared across every (dataset x model) run in the whole
    experiment -- as opposed to each run's own instances/metadata.json and
    case_instances.json, which record what actually happened for that one run.

    First call creates the file with protocol_fields as the canonical
    baseline and this run in "runs_covered". Later calls compare their
    protocol_fields against that baseline: if identical, just add/update this
    run in "runs_covered"; if different, the baseline is left untouched (so
    it stays a stable reference) and the deviation is recorded in
    "protocol_deviations" AND logged loudly -- silently drifting the fixed
    protocol between runs would quietly break the "every run used the same
    rules" guarantee this whole file exists to document.
    """
    path = os.path.join(explanations_root, "experiment_protocol.json")
    now = dt.datetime.utcnow().isoformat() + "Z"
    run_entry = {**run_identity, "generated_at_utc": now}

    if not os.path.exists(path):
        protocol = {
            "description": (
                "Fixed explanation-generation protocol shared across every dataset x model run "
                "in this experiment. Per-run specifics (actual selected instances, actual case picks) "
                "live in each run's own Explanations/<domain>/<dataset>/<model>/instances/ files; "
                "this file documents the SHARED RULES only."
            ),
            "protocol": protocol_fields,
            "protocol_deviations": [],
            "runs_covered": [run_entry],
            "created_at_utc": now,
            "last_updated_utc": now,
        }
        os.makedirs(explanations_root, exist_ok=True)
        with open(path, "w") as f:
            json.dump(protocol, f, indent=2, default=str)
        logger.log(f"Created master experiment protocol: {path}")
        return protocol

    with open(path) as f:
        protocol = json.load(f)

    if protocol["protocol"] != protocol_fields:
        deviation = {"run": run_identity, "detected_at_utc": now, "actual_protocol": protocol_fields}
        protocol["protocol_deviations"].append(deviation)
        logger.log(
            "WARNING: this run's protocol differs from the experiment's recorded baseline in "
            f"{path}. Baseline left unchanged; deviation recorded under 'protocol_deviations'. "
            "This means NOT every run in this experiment used identical settings -- check "
            "protocol_deviations before treating results as directly comparable across runs."
        )
    else:
        # keep only one entry per (domain, dataset, model), most recent wins
        protocol["runs_covered"] = [
            r for r in protocol["runs_covered"]
            if not (r["domain"] == run_identity["domain"] and r["dataset_name"] == run_identity["dataset_name"]
                    and r["model_name"] == run_identity["model_name"])
        ]
        protocol["runs_covered"].append(run_entry)

    protocol["last_updated_utc"] = now
    with open(path, "w") as f:
        json.dump(protocol, f, indent=2, default=str)
    logger.log(f"Updated master experiment protocol: {path} ({len(protocol['runs_covered'])} run(s) covered)")
    return protocol


# ---------------------------------------------------------------------------
# Orchestration: run_explanations
# ---------------------------------------------------------------------------

def run_explanations(config):
    """Runs explanation generation for one (dataset x model) run, given a
    CONFIG dict.

    Required config keys:
        dataset_name, domain, model_name  (model_name matches the folder used
                                            under Models/<domain>/<dataset_name>/)

    Optional config keys:
        explainers              (default: ["shap", "lime"])
        datasets_root           (default: "Datasets")
        models_root             (default: "Models")
        explanations_root       (default: "Explanations")
        random_state            (default: 42 -- used for instance selection AND
                                   passed to LIME's explainer)
        small_test_threshold    (default: 1000 -- test sets below this are
                                   explained in full)
        n_instances             (default: 500 -- stratified sample size otherwise)
        lime_num_samples        (default: 1000 -- perturbed samples per LIME instance)
        lime_num_features       (default: 50 -- cap on features per LIME explanation)
        top_n_features          (default: 10 -- size of the exported SHAP feature-importance CSV)
        case_selection_rule     (default: "highest_confidence" -- see select_or_load_case_instances
                                   for the "first" / "highest_confidence" / "median_confidence" tradeoffs)
        force                   (default: False -- if True, regenerates instance
                                   selection, case instances, AND explainer outputs
                                   even if already saved; breaks comparability with
                                   prior artifacts, so only use this deliberately)
    """
    for key in ("dataset_name", "domain", "model_name"):
        if key not in config:
            raise ValueError(f"CONFIG is missing required key '{key}'")

    explainers = config.get("explainers", ["shap", "lime"])
    datasets_root = config.get("datasets_root", "Datasets")
    models_root = config.get("models_root", "Models")
    explanations_root = config.get("explanations_root", "Explanations")
    random_state = config.get("random_state", DEFAULT_RANDOM_STATE)
    small_test_threshold = config.get("small_test_threshold", DEFAULT_SMALL_TEST_THRESHOLD)
    n_instances = config.get("n_instances", DEFAULT_N_INSTANCES)
    lime_num_samples = config.get("lime_num_samples", DEFAULT_LIME_NUM_SAMPLES)
    lime_num_features = config.get("lime_num_features", DEFAULT_LIME_NUM_FEATURES)
    top_n_features = config.get("top_n_features", DEFAULT_TOP_N_FEATURES)
    case_selection_rule = config.get("case_selection_rule", "highest_confidence")
    force = config.get("force", False)

    # 1. Setup
    paths = setup_explanation_dirs(config["dataset_name"], config["domain"], config["model_name"], explanations_root)
    logger = Logger(paths["logs"], filename="explanation_log.txt")
    logger.section("1. CONFIGURATION")
    logger.log(json.dumps(config, indent=2, default=str))

    # 2. Load model + data
    loaded = load_model_and_data(
        config["dataset_name"], config["domain"], config["model_name"],
        datasets_root, models_root, logger,
    )

    # 3. Select (or reuse) the explanation subset
    X_explain, y_explain, selection_info = select_or_load_instances(
        paths, loaded["X_test"], loaded["y_test"], loaded["target"],
        config["dataset_name"], config["domain"], config["model_name"],
        random_state, small_test_threshold, n_instances, logger, force=force,
    )

    # 3b. Select (or reuse) case instances (one correct, one incorrect)
    case_info = select_or_load_case_instances(
        paths, loaded["model"], X_explain, y_explain, selection_info, logger,
        rule=case_selection_rule, force=force,
    )

    results = {"paths": paths, "selection_info": selection_info, "case_info": case_info}

    # 4. SHAP
    if "shap" in explainers:
        compute_and_save_shap(
            paths, loaded["model"], loaded["model_reference"], loaded["dataset_characteristics"],
            X_explain, loaded["feature_order"], selection_info, case_info, logger,
            top_n=top_n_features, force=force,
        )

    # 5. LIME
    if "lime" in explainers:
        compute_and_save_lime(
            paths, loaded["model"], loaded["model_reference"], loaded["dataset_characteristics"],
            loaded["X_train_balanced"], X_explain, loaded["feature_order"], selection_info, case_info,
            random_state, logger, num_samples=lime_num_samples, num_features=lime_num_features, force=force,
        )

    # 6. Top-level run metadata (ties instance/case selection + all explainers together)
    run_info = {
        "dataset_name": config["dataset_name"],
        "domain": config["domain"],
        "model_name": config["model_name"],
        "explainers_requested": explainers,
        "model_reference": loaded["model_reference"],
        "dataset_characteristics": loaded["dataset_characteristics"],
        "selection_info": selection_info,
        "case_info": case_info,
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(paths["metadata"], "metadata.json"), "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    # Master experiment-level protocol: the fixed rules shared across every
    # dataset x model run, kept separately from this run's own specifics.
    protocol_fields = {
        "instance_selection": {
            "small_test_threshold": small_test_threshold,
            "target_n_instances": n_instances,
            "sampling_method": "stratified_random (sklearn train_test_split with stratify=y)",
            "random_state": random_state,
        },
        "case_instance_selection": {
            "rule": case_selection_rule,
            "rule_options": ["first", "highest_confidence", "median_confidence"],
            "confidence_definition": "predicted probability of the model's own predicted class",
        },
        "shap": {
            "explainer_library": "shap",
            "explainer_type": "TreeExplainer",
            "feature_perturbation": "tree_path_dependent",
            "top_n_features_exported": top_n_features,
        },
        "lime": {
            "explainer_library": "lime",
            "explainer_type": "LimeTabularExplainer",
            "num_samples_per_instance": lime_num_samples,
            "num_features_per_instance": lime_num_features,
            "discretize_continuous": True,
            "background_data_source": "X_train_balanced",
        },
    }
    run_identity = {"domain": config["domain"], "dataset_name": config["dataset_name"], "model_name": config["model_name"]}
    write_master_protocol(explanations_root, protocol_fields, run_identity, logger)

    logger.section("DONE")
    logger.log(f"Explanation artifacts at: {paths['root']}")

    return results
