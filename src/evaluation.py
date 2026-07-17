"""
Explanation evaluation, phase 1: Efficiency + Simplicity.

Per the agreed build order (Efficiency + Simplicity first -- both computable
from ALREADY-SAVED explanation artifacts, no new explanation generation --
then Fidelity, then Robustness, then Stability last), this module implements
the first two metrics and the SHARED INFRASTRUCTURE the later three will
reuse:

    - Feature-group map: rolls model-input columns (one-hot dummies) back up
      to original features. Built once per dataset from the fitted encoder
      saved in preprocessing, not by string-guessing. This is the SAME
      aggregation granularity decision agreed for Fidelity's masking protocol
      -- ranking/attribution-mass metrics need to operate at the same
      granularity as masking will, and it matters here too: one-hot dummies
      would otherwise inflate a categorical's apparent feature count and
      distort Simplicity's dimensionality-normalized metrics across datasets
      with different categorical cardinalities.
    - LIME condition-string parser: LIME's as_list() reports conditions like
      "Income <= 45000.00", not clean column names -- must be mapped back to
      the actual model-input column before the shared feature-group
      aggregation above can run. SHAP's shap_values.npy needs no such parsing
      (it's already indexed by feature_order).
    - metrics_long.csv: the master long-format table (schema below), and
      run_ledger.csv (experiment_id -> full config/environment snapshot, so
      the per-row table stays light and everything heavy is joined in).

Metric normalization (both explicitly dimensionality-normalized, since the
6 datasets range from 6 to 84+ raw features and comparing raw counts/times
across them would confound "metric differs" with "dataset is bigger"):

    Efficiency: primary comparable metric is runtime_ms_per_instance_per_feature
    (raw per-instance wall-clock time / n_features). LIME's background set
    size is logged as its own column (background_size) rather than folded
    into a combined normalizer, since SHAP's TreeExplainer (tree_path_dependent)
    uses NO background data at all in this project -- background size isn't a
    shared cost driver across both explainers, so normalizing by it would
    only be meaningful for LIME. Same treatment as kernel_width: a logged
    covariate, not baked into the cross-explainer metric.

    Simplicity: attributions are aggregated to ORIGINAL feature level first
    (see feature-group map above), then reported as (a) normalized entropy
    of the attribution-mass distribution (entropy / log2(n_original_features),
    bounded to [0,1] regardless of dataset dimensionality) and (b) the
    percentage of original features needed to reach 80%/90% of cumulative
    attribution mass (already a percentage, so comparable across dataset
    sizes without further normalization).

metrics_long.csv schema (columns present but NA/blank where not applicable
to a given metric -- e.g. repeat_idx/perturbation_id/baseline_type/
mask_fraction/kernel_width are all NA for Efficiency and Simplicity, and
exist here only so the schema doesn't need retrofitting when Fidelity/
Robustness/Stability are added):

    experiment_id, domain, dataset, model, explainer, metric_property,
    metric_name, instance_id, repeat_idx, perturbation_id, repeat_seed,
    deterministic, baseline_type, mask_fraction, value, runtime_ms,
    kernel_width, background_size, num_features, num_instances,
    random_state, status, timestamp

IMPORTANT: "num_features" means something different depending on
metric_property, and this is deliberate, not a bug:
    - Efficiency rows: num_features = MODEL-INPUT dimensionality (i.e. after
      one-hot encoding, e.g. 7 columns for a dataset with 2 numerical + 2
      categorical-with-multiple-levels features). This is correct for
      Efficiency because the actual computational cost driver is what the
      explainer literally computed over -- a wider one-hot space genuinely
      costs more, so normalizing by it is the right dimensionality control.
    - Simplicity rows: num_features = ORIGINAL feature count (after rolling
      one-hot dummies back up via the feature-group map). This is correct
      for Simplicity because the question is conceptual ("how many distinct
      real-world factors does the explanation spread its attribution
      across"), and counting one-hot dummies as separate "features" would
      make a 3-category variable look 3x as complex as a numeric one for no
      real reason.
    If you filter/join metrics_long.csv across metric_property values, do
    NOT assume num_features is the same number for the same dataset row to
    row -- check metric_property first.

"deterministic" = whether the underlying explanation this row is computed
from came from a deterministic process (True for every SHAP row, False for
every LIME row) -- not a metric-specific flag, so it's usable as a general
filter (e.g. filtering deterministic==True & metric_property=="Stability"
to exclude SHAP's trivial 1.0-stability control rows from an inferential
test, per your friend's note, without needing a metric-specific hack).
"""

import os
import re
import json
import time
import platform
import datetime as dt
import uuid

import numpy as np
import pandas as pd
import joblib

from src.utils import Logger

METRICS_LONG_COLUMNS = [
    "experiment_id", "domain", "dataset", "model", "explainer", "metric_property",
    "metric_name", "instance_id", "repeat_idx", "perturbation_id", "repeat_seed",
    "deterministic", "baseline_type", "mask_fraction", "value", "runtime_ms",
    "kernel_width", "background_size", "num_features", "num_instances",
    "random_state", "status", "timestamp",
]


# ---------------------------------------------------------------------------
# Shared infrastructure: run ledger
# ---------------------------------------------------------------------------

def create_experiment(evaluation_root, config_snapshot, logger):
    """Creates one entry in run_ledger.csv and returns its experiment_id.
    Call once per evaluation invocation (e.g. once per notebook run covering
    several dataset x model x explainer combos), not once per metric row --
    the ledger holds the heavy/shared metadata so metrics_long.csv doesn't
    have to repeat it on every line.
    """
    os.makedirs(evaluation_root, exist_ok=True)
    experiment_id = f"exp_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    entry = {
        "experiment_id": experiment_id,
        "config_snapshot": json.dumps(config_snapshot, default=str),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "created_at_utc": dt.datetime.utcnow().isoformat() + "Z",
    }
    ledger_path = os.path.join(evaluation_root, "run_ledger.csv")
    df = pd.DataFrame([entry])
    if os.path.exists(ledger_path):
        df.to_csv(ledger_path, mode="a", header=False, index=False)
    else:
        df.to_csv(ledger_path, index=False)
    logger.log(f"Created experiment_id={experiment_id}, logged to {ledger_path}")
    return experiment_id


def append_metrics(evaluation_root, rows, logger):
    """Appends a list of row-dicts to metrics_long.csv, enforcing the fixed
    column schema (missing keys become NA, unexpected keys raise -- a typo'd
    column name should fail loudly here, not silently create a stray column).
    """
    if not rows:
        return
    unexpected = set()
    for r in rows:
        unexpected |= (set(r.keys()) - set(METRICS_LONG_COLUMNS))
    if unexpected:
        raise ValueError(f"Row(s) contain columns not in METRICS_LONG_COLUMNS: {unexpected}")

    df = pd.DataFrame(rows).reindex(columns=METRICS_LONG_COLUMNS)
    path = os.path.join(evaluation_root, "metrics_long.csv")
    os.makedirs(evaluation_root, exist_ok=True)
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
    logger.log(f"Appended {len(df)} row(s) to {path}")


# ---------------------------------------------------------------------------
# Shared infrastructure: feature-group map (model-input columns -> original features)
# ---------------------------------------------------------------------------

def build_feature_group_map(dataset_name, domain, datasets_root, feature_order, logger):
    """Returns {original_feature_name: [model_input_column, ...]}, covering
    every column in feature_order exactly once. Numerical features map 1:1
    to themselves; categorical features map to their one-hot dummy columns,
    read directly from the fitted OneHotEncoder saved in preprocessing
    (encoder.categories_, aligned with categorical_cols) -- not guessed from
    string prefixes, since a category value could itself contain an
    underscore and break naive prefix-splitting.
    """
    meta_dir = os.path.join(datasets_root, domain, dataset_name, "metadata")
    transformers_path = os.path.join(meta_dir, "fitted_transformers.joblib")
    bundle = joblib.load(transformers_path)
    numerical_cols = bundle.get("numerical_cols") or []
    categorical_cols = bundle.get("categorical_cols") or []
    encoder = bundle.get("encoder")

    group_map = {c: [c] for c in numerical_cols}

    if categorical_cols and encoder is not None:
        for col, categories in zip(categorical_cols, encoder.categories_):
            dummy_cols = [f"{col}_{cat}" for cat in categories]
            group_map[col] = dummy_cols

    covered = sorted(c for cols in group_map.values() for c in cols)
    expected = sorted(feature_order)
    if covered != expected:
        missing = set(expected) - set(covered)
        extra = set(covered) - set(expected)
        raise ValueError(
            f"Feature-group map doesn't exactly cover feature_order for {dataset_name}. "
            f"Missing from map: {missing}. In map but not in feature_order: {extra}."
        )

    logger.log(f"Feature-group map for {dataset_name}: {len(group_map)} original features "
               f"(from {len(feature_order)} model-input columns).")
    return group_map


def aggregate_to_original_features(values_by_column, group_map):
    """values_by_column: dict or Series {model_input_column: value}.
    Returns dict {original_feature: aggregated_value}, summing across each
    group's model-input columns. Summing (not max) is used because SHAP
    values are additive -- summing a one-hot group recovers that
    categorical's total effect cleanly, since only the active dummy carries
    non-trivial attribution for a given instance. LIME's per-condition
    weights aren't strictly additive in the same formal sense, but summing
    magnitudes is the standard practical choice here too (matches the "sum
    or max" guidance already agreed for the fidelity masking protocol --
    sum chosen for consistency between the two explainers).
    """
    return {
        original: sum(values_by_column.get(c, 0.0) for c in cols)
        for original, cols in group_map.items()
    }


# ---------------------------------------------------------------------------
# Shared infrastructure: LIME condition-string -> model-input column parsing
# ---------------------------------------------------------------------------

def parse_lime_feature(condition, feature_order):
    """LIME's Explanation.as_list() reports feature CONDITIONS, not column
    names, e.g. "Income <= 45000.00", "30.00 < Age <= 45.00", or
    "Education_Bachelor's <= 0.00" for already-binary one-hot columns. This
    extracts which entry in feature_order the condition is actually about.

    Matches the LONGEST feature_order entry that appears in the condition
    string, checked as a whole-token match (bounded by start/end or
    non-alphanumeric characters) -- longest-first so e.g. "Age" doesn't
    incorrectly match inside a hypothetical "AgeGroup" column before the
    correct full name is tried. Raises if no match is found (fail loudly
    rather than silently mis-attributing).
    """
    for col in sorted(feature_order, key=len, reverse=True):
        pattern = r"(?<![A-Za-z0-9_])" + re.escape(col) + r"(?![A-Za-z0-9_])"
        if re.search(pattern, condition):
            return col
    raise ValueError(f"Could not match LIME condition '{condition}' to any known feature in feature_order.")


# ---------------------------------------------------------------------------
# Efficiency
# ---------------------------------------------------------------------------

def compute_efficiency(explanation_root, experiment_id, domain, dataset_name, model_name,
                         explainer, dataset_characteristics, random_state, logger):
    """One row (not per-instance -- efficiency is measured as a run-level
    average, since explanation generation only logged total runtime, not
    per-instance timing). Reads runtime_seconds + n_instances_explained
    already saved in shap/metadata.json or lime/metadata.json -- no new
    computation, just re-expressing what's already there into the shared
    metrics_long schema.
    """
    meta_path = os.path.join(explanation_root, domain, dataset_name, model_name, explainer, "metadata.json")
    if not os.path.exists(meta_path):
        logger.log(f"SKIP efficiency for {domain}/{dataset_name}/{model_name}/{explainer}: {meta_path} not found.")
        return []

    with open(meta_path) as f:
        exp_meta = json.load(f)

    runtime_seconds = exp_meta.get("runtime_seconds")
    n_instances = exp_meta.get("n_instances_explained")
    n_features = dataset_characteristics["n_features"]
    if runtime_seconds is None or not n_instances:
        logger.log(f"SKIP efficiency for {domain}/{dataset_name}/{model_name}/{explainer}: "
                   "missing runtime_seconds/n_instances_explained in metadata.json.")
        return []

    runtime_ms_per_instance = (runtime_seconds * 1000.0) / n_instances
    runtime_ms_per_instance_per_feature = runtime_ms_per_instance / n_features

    background_size = exp_meta.get("background_data_rows") if explainer == "lime" else 0
    kernel_width = exp_meta.get("kernel_width")   # LIME-only covariate; None for SHAP rows

    now = dt.datetime.utcnow().isoformat() + "Z"
    base_row = {
        "experiment_id": experiment_id, "domain": domain, "dataset": dataset_name, "model": model_name,
        "explainer": explainer, "metric_property": "Efficiency",
        "instance_id": None, "repeat_idx": None, "perturbation_id": None, "repeat_seed": None,
        "deterministic": (explainer == "shap"), "baseline_type": None, "mask_fraction": None,
        "runtime_ms": round(runtime_seconds * 1000.0, 3), "kernel_width": kernel_width,
        "background_size": background_size, "num_features": n_features, "num_instances": n_instances,
        "random_state": random_state, "status": "ok", "timestamp": now,
    }

    rows = [
        {**base_row, "metric_name": "runtime_ms_per_instance", "value": runtime_ms_per_instance},
        {**base_row, "metric_name": "runtime_ms_per_instance_per_feature", "value": runtime_ms_per_instance_per_feature},
    ]
    logger.log(f"Efficiency ({explainer}, {dataset_name}, {model_name}): "
               f"{runtime_ms_per_instance:.3f} ms/instance, {runtime_ms_per_instance_per_feature:.5f} ms/instance/feature")
    return rows


# ---------------------------------------------------------------------------
# Simplicity
# ---------------------------------------------------------------------------

def _normalized_entropy(abs_attrs):
    """abs_attrs: 1D array of non-negative attribution magnitudes for the
    ORIGINAL (already feature-grouped) features of one instance. Returns
    entropy of the mass distribution, normalized by log2(n_features) so the
    result is in [0, 1] regardless of how many features the dataset has.
    Higher = attribution mass spread evenly across many features (less
    simple); lower = concentrated in a few (simpler). Returns NaN if all
    attributions are exactly zero (undefined distribution) or if there's
    only one feature (normalization would divide by log2(1)=0).
    """
    n = len(abs_attrs)
    total = abs_attrs.sum()
    if n <= 1 or total <= 0:
        return np.nan
    p = abs_attrs / total
    p_nonzero = p[p > 0]
    entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
    return entropy / np.log2(n)


def _pct_features_for_mass(abs_attrs, target_fraction):
    """Smallest number of (original) features, taken in descending order of
    |attribution|, whose cumulative share of total mass reaches
    target_fraction -- returned as a PERCENTAGE of n_features (already
    normalized for cross-dataset comparability without further scaling).
    """
    n = len(abs_attrs)
    total = abs_attrs.sum()
    if n == 0 or total <= 0:
        return np.nan
    sorted_desc = np.sort(abs_attrs)[::-1]
    cumulative = np.cumsum(sorted_desc) / total
    k = int(np.searchsorted(cumulative, target_fraction) + 1)
    k = min(k, n)
    return 100.0 * k / n


def compute_simplicity_shap(explanation_root, experiment_id, domain, dataset_name, model_name,
                              group_map, dataset_characteristics, selection_info, random_state, logger):
    """Per-instance simplicity from shap_values.npy, aggregated to original
    feature level via group_map before computing entropy/mass-coverage.
    """
    shap_dir = os.path.join(explanation_root, domain, dataset_name, model_name, "shap")
    values_path = os.path.join(shap_dir, "shap_values.npy")
    if not os.path.exists(values_path):
        logger.log(f"SKIP SHAP simplicity for {domain}/{dataset_name}/{model_name}: {values_path} not found.")
        return []

    with open(os.path.join(shap_dir, "metadata.json")) as f:
        shap_meta = json.load(f)
    feature_order = shap_meta["feature_order"]
    values = np.load(values_path)   # (n_instances, n_model_input_features)
    instance_ids = selection_info["dataframe_indices"]
    n_features_original = len(group_map)

    now = dt.datetime.utcnow().isoformat() + "Z"
    rows = []
    for i, instance_id in enumerate(instance_ids):
        by_col = dict(zip(feature_order, np.abs(values[i])))
        agg = aggregate_to_original_features(by_col, group_map)
        abs_attrs = np.array(list(agg.values()))

        base_row = {
            "experiment_id": experiment_id, "domain": domain, "dataset": dataset_name, "model": model_name,
            "explainer": "shap", "metric_property": "Simplicity", "instance_id": instance_id,
            "repeat_idx": None, "perturbation_id": None, "repeat_seed": None, "deterministic": True,
            "baseline_type": None, "mask_fraction": None, "runtime_ms": None, "kernel_width": None,
            "background_size": 0, "num_features": n_features_original, "num_instances": len(instance_ids),
            "random_state": random_state, "status": "ok", "timestamp": now,
        }
        rows.append({**base_row, "metric_name": "normalized_entropy_complexity", "value": _normalized_entropy(abs_attrs)})
        rows.append({**base_row, "metric_name": "pct_features_for_80pct_mass", "value": _pct_features_for_mass(abs_attrs, 0.80)})
        rows.append({**base_row, "metric_name": "pct_features_for_90pct_mass", "value": _pct_features_for_mass(abs_attrs, 0.90)})

    logger.log(f"SHAP simplicity: {len(instance_ids)} instances x 3 metrics = {len(rows)} rows.")
    return rows


def compute_simplicity_lime(explanation_root, experiment_id, domain, dataset_name, model_name,
                              group_map, feature_order, dataset_characteristics, selection_info,
                              random_state, logger):
    """Per-instance simplicity from lime_explanations.csv. LIME's per-row
    'feature' field is a condition string, not a column name -- parsed back
    to a model-input column via parse_lime_feature() first, THEN aggregated
    to original-feature level via the SAME group_map used for SHAP, so both
    explainers' simplicity numbers are computed at identical granularity.
    Rows whose condition can't be parsed are marked status='failed' and
    excluded from the value (kept visible in the table, not silently dropped,
    per your friend's note on the status column).
    """
    lime_dir = os.path.join(explanation_root, domain, dataset_name, model_name, "lime")
    exp_path = os.path.join(lime_dir, "lime_explanations.csv")
    if not os.path.exists(exp_path):
        logger.log(f"SKIP LIME simplicity for {domain}/{dataset_name}/{model_name}: {exp_path} not found.")
        return []

    long_df = pd.read_csv(exp_path)
    n_features_original = len(group_map)
    now = dt.datetime.utcnow().isoformat() + "Z"
    rows = []
    n_parse_failures = 0

    for instance_id, group in long_df.groupby("instance_id"):
        by_col = {}
        for _, r in group.iterrows():
            try:
                col = parse_lime_feature(r["feature"], feature_order)
            except ValueError:
                n_parse_failures += 1
                continue
            by_col[col] = by_col.get(col, 0.0) + abs(r["weight"])

        base_row = {
            "experiment_id": experiment_id, "domain": domain, "dataset": dataset_name, "model": model_name,
            "explainer": "lime", "metric_property": "Simplicity", "instance_id": instance_id,
            "repeat_idx": None, "perturbation_id": None, "repeat_seed": None, "deterministic": False,
            "baseline_type": None, "mask_fraction": None, "runtime_ms": None, "kernel_width": None,
            "background_size": None, "num_features": n_features_original,
            "num_instances": long_df["instance_id"].nunique(), "random_state": random_state,
            "timestamp": now,
        }

        if not by_col:
            rows.append({**base_row, "metric_name": "normalized_entropy_complexity", "value": None, "status": "failed"})
            rows.append({**base_row, "metric_name": "pct_features_for_80pct_mass", "value": None, "status": "failed"})
            rows.append({**base_row, "metric_name": "pct_features_for_90pct_mass", "value": None, "status": "failed"})
            continue

        agg = aggregate_to_original_features(by_col, group_map)
        abs_attrs = np.array(list(agg.values()))
        rows.append({**base_row, "metric_name": "normalized_entropy_complexity",
                     "value": _normalized_entropy(abs_attrs), "status": "ok"})
        rows.append({**base_row, "metric_name": "pct_features_for_80pct_mass",
                     "value": _pct_features_for_mass(abs_attrs, 0.80), "status": "ok"})
        rows.append({**base_row, "metric_name": "pct_features_for_90pct_mass",
                     "value": _pct_features_for_mass(abs_attrs, 0.90), "status": "ok"})

    if n_parse_failures:
        logger.log(f"WARNING: {n_parse_failures} LIME condition string(s) could not be parsed to a known "
                   "feature and were excluded (see status='failed' rows).")
    logger.log(f"LIME simplicity: {long_df['instance_id'].nunique()} instances x 3 metrics = {len(rows)} rows "
               f"({n_parse_failures} unparsed conditions excluded).")
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_efficiency_and_simplicity(config):
    """Runs Efficiency + Simplicity for one (dataset x model) run's saved
    explanations, appending rows to Evaluation/metrics_long.csv under a
    shared experiment_id from Evaluation/run_ledger.csv.

    Required config keys:
        dataset_name, domain, model_name, experiment_id  (experiment_id from
        a prior create_experiment() call -- typically one per notebook run
        covering many dataset x model combos, not one per combo)

    Optional config keys:
        explainers          (default: ["shap", "lime"])
        datasets_root        (default: "Datasets")
        explanations_root    (default: "Explanations")
        evaluation_root       (default: "Evaluation")
    """
    for key in ("dataset_name", "domain", "model_name", "experiment_id"):
        if key not in config:
            raise ValueError(f"CONFIG is missing required key '{key}'")

    explainers = config.get("explainers", ["shap", "lime"])
    datasets_root = config.get("datasets_root", "Datasets")
    explanations_root = config.get("explanations_root", "Explanations")
    evaluation_root = config.get("evaluation_root", "Evaluation")

    logger = Logger(os.path.join(evaluation_root, "logs"), filename="efficiency_simplicity_log.txt")
    logger.section(f"EFFICIENCY + SIMPLICITY: {config['dataset_name']} x {config['model_name']}")

    run_dir = os.path.join(explanations_root, config["domain"], config["dataset_name"], config["model_name"])
    with open(os.path.join(run_dir, "metadata", "metadata.json")) as f:
        run_meta = json.load(f)
    dataset_characteristics = run_meta["dataset_characteristics"]
    selection_info = run_meta["selection_info"]
    random_state = selection_info["random_state"]

    group_map = build_feature_group_map(
        config["dataset_name"], config["domain"], datasets_root, dataset_characteristics["feature_names"], logger,
    )

    all_rows = []
    for explainer in explainers:
        all_rows += compute_efficiency(
            explanations_root, config["experiment_id"], config["domain"], config["dataset_name"],
            config["model_name"], explainer, dataset_characteristics, random_state, logger,
        )
        if explainer == "shap":
            all_rows += compute_simplicity_shap(
                explanations_root, config["experiment_id"], config["domain"], config["dataset_name"],
                config["model_name"], group_map, dataset_characteristics, selection_info, random_state, logger,
            )
        elif explainer == "lime":
            all_rows += compute_simplicity_lime(
                explanations_root, config["experiment_id"], config["domain"], config["dataset_name"],
                config["model_name"], group_map, dataset_characteristics["feature_names"],
                dataset_characteristics, selection_info, random_state, logger,
            )

    append_metrics(evaluation_root, all_rows, logger)
    logger.section("DONE")
    return all_rows
