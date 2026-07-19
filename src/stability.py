"""
Explanation evaluation, phase 4: Stability.

Measures whether an explainer agrees with ITSELF on repeated calls against
the SAME, unperturbed instance -- as opposed to Robustness, which measures
whether an explanation stays consistent when the INPUT is perturbed to a
different, but realistically similar, instance. Stability is therefore a
question of the explainer's own internal (non-)determinism, not of
input-sensitivity.

RESOLVED PROTOCOL (per schema.md and the earlier resolved-decisions log):

1. SHAP (TreeSHAP) is deterministic by construction: repeated calls against
   the same instance and model produce bit-identical attributions. Rather
   than waste N redundant identical calls confirming this, SHAP gets a
   single `deterministic=True` floor row per instance (perfect agreement,
   value fixed at 1.0 for both metrics) and is EXCLUDED from the
   inferential domain-consistency test -- the LIME-only variance
   decomposition is what actually tests Stability as a question. This
   mirrors the same `deterministic` flag Fidelity/Robustness already write
   per row; here it changes what's excluded downstream, not just what's
   labeled.

2. LIME is repeated N=10 times per instance, all against the exact same,
   UNPERTURBED input vector, using ONE shared LimeTabularExplainer instance
   per (dataset, model) reconstructed with the original run's exact config
   (background data source, num_samples, num_features, random_state -- see
   build_lime_explainer, reused directly from src/robustness.py). The
   explainer is intentionally NOT reset or reseeded between repeats: LIME's
   internal sampling RandomState is left to advance naturally call to call,
   because that call-to-call sampling variance is precisely the thing this
   metric is designed to measure. Reseeding to identical state before every
   repeat would trivially force a perfect score and defeat the purpose.

3. Comparison: with N=10 repeats there are C(10,2)=45 pairs, not a single
   pair as in Robustness. Each pair's attribution vectors (aggregated to
   original-feature granularity, exactly as Robustness/Fidelity do) are
   compared via the SAME two statistics Robustness uses -- Spearman rank
   correlation over the full vector, and top-k Jaccard overlap
   (k = min(5, ceil(n_features/2)), same convention) -- and the per-instance
   Stability score is the MEAN of that statistic across all 45 pairs. This
   pairwise-mean reduction matches the Visani et al. CSI/VSI convention
   referenced in schema.md, rather than arbitrarily privileging one repeat
   as a fixed reference to compare all others against.

   PATCH NOTE (see also point 5 below): k is now computed ONCE per run via
   `default_top_k(n_features)` and threaded explicitly into every
   `compare_attributions` call (both here and matched against the SHAP
   branch's floor-row metric name), rather than letting the SHAP branch
   recompute it locally and the LIME branch rely on compare_attributions's
   own internal default. Two independent implementations of the same
   formula is how they silently drift if the shared default ever changes.

4. Auditable artifact: Evaluation/Stability/<domain>/<dataset>/<model>/
   <explainer>/repeated_attributions.csv -- one row per
   (instance_id, repeat_idx, original_feature), recording that repeat's
   aggregated attribution for that feature. This is the fine-grained
   analogue of Robustness's perturbation_attributions.csv: it lets the
   pairwise Spearman/Jaccard reduction be recomputed later under a
   different k, a different pair-reduction convention, or a different N
   (by subsetting repeat_idx), without re-running LIME from scratch.

5. PATCH NOTE -- degenerate instances no longer silently dropped: if every
   one of the C(N,2) pairs for a given LIME instance yields a NaN Spearman
   correlation (e.g. every repeat returned a constant attribution vector),
   the instance previously received NO
   `mean_pairwise_spearman_rank_correlation` row at all -- it just vanished
   from that metric's sample for that (dataset, model) combo, with nothing
   in metrics_long.csv or the logs to show it happened. That silently
   shrinks the effective LIME sample size per combo in a way that would
   only be discovered by comparing row counts across combos by hand.
   Now: any instance where the correlation mean is undefined still gets a
   row, with `status="degenerate_all_nan_pairs"` and `value=None`, and the
   count of such instances is logged per (dataset, model, explainer). The
   Jaccard statistic is unaffected since compare_attributions never returns
   NaN for it (see its docstring in robustness.py) -- only the correlation
   arm needs this handling.

KNOWN OPEN QUESTION -- dimensionality / cross-dataset normalization:
Efficiency and Simplicity both needed explicit normalization to be
comparable across datasets of different feature counts (see
EFFICIENCY_README.md and SIMPLICITY_README.md for their respective
treatments). Stability's two statistics behave differently under that same
concern and have NOT yet been checked against those READMEs:

  - Spearman rank correlation is already scale- and dimension-normalized in
    range (always in [-1, 1] regardless of n_features), but it is NOT
    variance-normalized across dimensionality: with very few features, the
    number of achievable rank permutations is small, so correlation values
    are more discretized and can look artificially more stable (or
    unstable) than a high-dimensional dataset purely as a function of n,
    independent of the explainer's actual behavior. Whether this is the
    same effect Efficiency/Simplicity corrected for, or a distinct one,
    needs to be checked against their normalization approach.
  - top-k Jaccard overlap uses k = min(5, ceil(n_features/2)) -- this means
    k is NOT a constant fraction of n_features across datasets: low-feature
    datasets get k scaled to roughly half their features, while
    high-feature datasets are capped at a flat 5, i.e. a shrinking fraction
    as n grows. That makes the raw top-k Jaccard values not directly
    comparable across datasets of very different dimensionality without
    some additional normalization -- this mirrors the kind of issue
    Efficiency/Simplicity hit, but has not been verified against how those
    two actually solved it.

This needs EFFICIENCY_README.md, SIMPLICITY_README.md, and the actual
compare_attributions implementation in robustness.py to resolve properly --
flagging here rather than guessing at a fix.
"""

import os
import json
import itertools
import datetime as dt

import numpy as np
import pandas as pd

from src.utils import Logger
from src.evaluation import (
    build_feature_group_map,
    aggregate_to_original_features,
    parse_lime_feature,
    append_metrics,
)
from src.fidelity import load_model
from src.robustness import build_lime_explainer, get_lime_attribution, compare_attributions

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

N_REPEATS = 10
# N_REPEATS = 5


# ---------------------------------------------------------------------------
# 0. Shared top-k convention (single source of truth -- see patch note 3/5)
# ---------------------------------------------------------------------------

def default_top_k(n_features):
    """The k used for top-k Jaccard overlap, matching Robustness's
    convention: k = min(5, ceil(n_features / 2)).

    Computed here ONCE and passed explicitly to every compare_attributions
    call in this module, rather than letting each call site (or each
    explainer branch) either recompute it locally or fall back on
    compare_attributions's own internal default. Two independent
    implementations of the same formula is how they silently drift if the
    shared convention ever changes in robustness.py.
    """
    return min(5, int(np.ceil(n_features / 2)))


# ---------------------------------------------------------------------------
# 1. Pairwise reduction over N repeats
# ---------------------------------------------------------------------------

def compute_pairwise_stability(repeat_aggs, all_original_features, top_k=None):
    """repeat_aggs: list of N {original_feature: aggregated_abs_attribution}
    dicts, one per repeat, all for the SAME unperturbed instance.

    Computes Spearman rank correlation and top-k Jaccard overlap for every
    one of the C(N,2) pairs of repeats (reusing compare_attributions from
    src/robustness.py, which already implements both statistics for a pair
    of attribution dicts), then returns the MEAN of each statistic across
    all pairs -- the Visani et al. CSI/VSI-style pairwise-agreement
    convention (see module docstring, point 3).

    Returns (mean_corr, mean_jaccard, k_used, n_pairs, n_nan_corr_pairs).
    NaN-valued correlation pairs (e.g. a constant-attribution repeat -- see
    compare_attributions) are excluded from the correlation mean rather
    than propagating NaN into the whole instance-level score, since a
    single degenerate repeat out of 10 shouldn't null out an otherwise-
    informative result. `n_nan_corr_pairs` is returned so the caller can
    detect and flag the all-NaN case explicitly (patch note 5) instead of
    silently dropping the instance.
    """
    n = len(repeat_aggs)
    if n < 2:
        raise ValueError("compute_pairwise_stability requires at least 2 repeats.")

    corrs, jaccards, k_used = [], [], top_k
    n_nan_corr_pairs = 0
    for i, j in itertools.combinations(range(n), 2):
        corr, jaccard, k_used = compare_attributions(
            repeat_aggs[i], repeat_aggs[j], all_original_features, top_k=top_k,
        )
        if corr is not None:
            corrs.append(corr)
        else:
            n_nan_corr_pairs += 1
        jaccards.append(jaccard)

    mean_corr = float(np.mean(corrs)) if corrs else None
    mean_jaccard = float(np.mean(jaccards)) if jaccards else None
    n_pairs = n * (n - 1) // 2
    return mean_corr, mean_jaccard, k_used, n_pairs, n_nan_corr_pairs


# ---------------------------------------------------------------------------
# 2. Orchestration
# ---------------------------------------------------------------------------

def run_stability(config):
    """Required config keys:
        dataset_name, domain, model_name, experiment_id

    Optional config keys:
        explainers        (default: ["shap", "lime"])
        datasets_root      (default: "Datasets")
        explanations_root  (default: "Explanations")
        models_root        (default: "Models")
        evaluation_root    (default: "Evaluation")
        stability_root      (default: "Evaluation/Stability" -- where
                             repeated_attributions.csv is written)
        n_repeats            (default: N_REPEATS = 10, LIME only)
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
    stability_root = config.get("stability_root", os.path.join(evaluation_root, "Stability"))
    n_repeats = config.get("n_repeats", N_REPEATS)

    logger = Logger(os.path.join(evaluation_root, "logs"), filename="stability_log.txt")
    logger.section(f"STABILITY: {dataset_name} x {model_name}")

    run_dir = os.path.join(explanations_root, domain, dataset_name, model_name)
    with open(os.path.join(run_dir, "metadata", "metadata.json")) as f:
        run_meta = json.load(f)
    selection_info = run_meta["selection_info"]
    random_state = config.get("random_state", selection_info["random_state"])
    instance_ids = selection_info["dataframe_indices"]

    model, schema = load_model(domain, dataset_name, model_name, models_root, logger)
    feature_order = schema["feature_order"]

    group_map = build_feature_group_map(dataset_name, domain, datasets_root, feature_order, logger)

    x_explain_path = os.path.join(run_dir, "instances", "X_explain.csv")
    X_explain = pd.read_csv(x_explain_path)[feature_order]
    X_explain.index = list(instance_ids)

    all_original_features = list(group_map.keys())
    # Single source of truth for k -- computed once, threaded explicitly
    # into every compare_attributions call below (patch note 3).
    k_top = default_top_k(len(all_original_features))

    now = dt.datetime.utcnow().isoformat() + "Z"
    all_metric_rows = []
    stability_dir_base = os.path.join(stability_root, domain, dataset_name, model_name)

    for explainer in explainers:
        repeated_attribution_rows = []
        n_degenerate_instances = 0

        if explainer == "shap":
            if not _HAS_SHAP:
                logger.log("SKIP stability for shap: 'shap' package not installed.")
                continue
            shap_dir = os.path.join(run_dir, "shap")
            values_path = os.path.join(shap_dir, "shap_values.npy")
            if not os.path.exists(values_path):
                logger.log(f"SKIP stability for shap: {values_path} not found.")
                continue
            original_shap_values = np.load(values_path)
            per_instance_agg = {
                iid: aggregate_to_original_features(
                    dict(zip(feature_order, np.abs(original_shap_values[i]))), group_map,
                )
                for i, iid in enumerate(instance_ids)
            }

            # TreeSHAP is deterministic by construction (module docstring,
            # point 1): a single floor row per instance, no repeated calls,
            # no pairwise computation. Perfect self-agreement by definition.
            for iid in instance_ids:
                if iid not in per_instance_agg:
                    continue
                for feat, val in per_instance_agg[iid].items():
                    repeated_attribution_rows.append({
                        "instance_id": iid, "repeat_idx": 0, "feature": feat, "attribution": val,
                    })
                base_row = {
                    "experiment_id": config["experiment_id"], "domain": domain, "dataset": dataset_name,
                    "model": model_name, "explainer": explainer, "metric_property": "Stability",
                    "instance_id": iid, "repeat_idx": None, "perturbation_id": None, "repeat_seed": None,
                    "deterministic": True, "baseline_type": None,
                    "mask_fraction": None, "runtime_ms": None, "kernel_width": None, "background_size": None,
                    "num_features": len(group_map), "num_instances": len(instance_ids),
                    "random_state": random_state, "status": "ok", "timestamp": now,
                }
                all_metric_rows.append({**base_row, "metric_name": "mean_pairwise_spearman_rank_correlation", "value": 1.0})
                # k_top computed once above from the shared helper -- was
                # previously recomputed locally here, which could silently
                # drift from the LIME branch's k if compare_attributions's
                # internal default ever changed (patch note 3).
                all_metric_rows.append({
                    **base_row, "metric_name": f"mean_pairwise_top_{k_top}_jaccard_overlap", "value": 1.0,
                })

        elif explainer == "lime":
            if not _HAS_LIME:
                logger.log("SKIP stability for lime: 'lime' package not installed.")
                continue
            lime_meta_path = os.path.join(run_dir, "lime", "metadata.json")
            if not os.path.exists(lime_meta_path):
                logger.log("SKIP stability for lime: original LIME metadata not found.")
                continue
            with open(lime_meta_path) as f:
                lime_metadata = json.load(f)

            lime_explainer = build_lime_explainer(
                domain, dataset_name, model_name, feature_order, lime_metadata, datasets_root,
            )
            lime_num_features = lime_metadata.get("num_features_per_instance", len(feature_order))
            lime_num_samples = lime_metadata.get("num_samples_per_instance", 1000)

            n_parse_failures_total = 0
            for iid in instance_ids:
                original_vector = X_explain.loc[iid].to_numpy(dtype=float)

                # Same shared lime_explainer instance across all N repeats,
                # deliberately not reseeded between calls (module docstring,
                # point 2) -- its internal sampling RandomState advances
                # naturally, and THAT call-to-call variance is what this
                # metric measures.
                repeat_aggs = []
                for repeat_idx in range(n_repeats):
                    agg, n_fail = get_lime_attribution(
                        lime_explainer, model, original_vector, feature_order,
                        lime_num_features, lime_num_samples, group_map,
                    )
                    n_parse_failures_total += n_fail
                    repeat_aggs.append(agg)
                    for feat in all_original_features:
                        repeated_attribution_rows.append({
                            "instance_id": iid, "repeat_idx": repeat_idx,
                            "feature": feat, "attribution": agg.get(feat, 0.0),
                        })

                mean_corr, mean_jaccard, k_used, n_pairs, n_nan_corr_pairs = compute_pairwise_stability(
                    repeat_aggs, all_original_features, top_k=k_top,
                )

                base_row = {
                    "experiment_id": config["experiment_id"], "domain": domain, "dataset": dataset_name,
                    "model": model_name, "explainer": explainer, "metric_property": "Stability",
                    "instance_id": iid, "repeat_idx": None, "perturbation_id": None, "repeat_seed": None,
                    "deterministic": False, "baseline_type": None,
                    "mask_fraction": None, "runtime_ms": None, "kernel_width": None, "background_size": None,
                    "num_features": len(group_map), "num_instances": len(instance_ids),
                    "random_state": random_state, "status": "ok", "timestamp": now,
                }

                # PATCH (note 5): previously, if mean_corr was None (every
                # one of the C(N,2) pairs had a NaN correlation), no row was
                # written at all and the instance silently vanished from
                # this metric's sample. Now a row is always written; a fully
                # degenerate instance gets status="degenerate_all_nan_pairs"
                # and value=None instead of disappearing.
                if mean_corr is not None:
                    all_metric_rows.append({
                        **base_row, "metric_name": "mean_pairwise_spearman_rank_correlation", "value": mean_corr,
                    })
                else:
                    n_degenerate_instances += 1
                    all_metric_rows.append({
                        **base_row, "metric_name": "mean_pairwise_spearman_rank_correlation", "value": None,
                        "status": "degenerate_all_nan_pairs",
                    })

                if mean_jaccard is not None:
                    all_metric_rows.append({
                        **base_row, "metric_name": f"mean_pairwise_top_{k_used}_jaccard_overlap", "value": mean_jaccard,
                    })

                if n_nan_corr_pairs:
                    logger.log(f"  instance {iid}: {n_nan_corr_pairs}/{n_pairs} pairs had NaN correlation "
                               f"({'ALL -- instance flagged degenerate' if mean_corr is None else 'excluded from mean'}).")

            if n_parse_failures_total:
                logger.log(f"WARNING: {n_parse_failures_total} LIME condition string(s) unparsed across "
                           f"{n_repeats} repeat(s) x {len(instance_ids)} instance(s).")
            if n_degenerate_instances:
                logger.log(f"WARNING: {n_degenerate_instances}/{len(instance_ids)} instance(s) had ALL pairwise "
                           f"correlations undefined (NaN) -- written with status='degenerate_all_nan_pairs', "
                           f"value=None, rather than dropped. Check these before treating the LIME correlation "
                           f"sample size as {len(instance_ids)} for this (dataset, model) combo.")
        else:
            raise ValueError(f"Unknown explainer '{explainer}'")

        os.makedirs(os.path.join(stability_dir_base, explainer), exist_ok=True)
        artifact_path = os.path.join(stability_dir_base, explainer, "repeated_attributions.csv")
        pd.DataFrame(repeated_attribution_rows).to_csv(artifact_path, index=False)
        logger.log(f"Wrote {len(repeated_attribution_rows)} rows to {artifact_path}")

    append_metrics(evaluation_root, all_metric_rows, logger)
    logger.section("DONE")
    logger.log(f"Stability: {len(all_metric_rows)} metrics_long rows written across {explainers}.")
    return all_metric_rows
