"""
Model training pipeline: takes the preprocessed artifacts produced by
src/preprocessing.py for one dataset and runs one (dataset x model) training
run under a fixed, identical protocol.

    1.  Configuration            (one CONFIG dict per dataset x model run)
    2.  Load artifacts           (pre-balance train, balanced train, test,
                                   fitted transformers -- all from Datasets/)
    3.  Hyperparameter search    (Optuna, TPE sampler; SMOTE refit inside each
                                   CV fold, scored on each fold's naturally-
                                   imbalanced validation split)
    4.  Refit final model        (winning hyperparameters, fit on the FULL
                                   SMOTE-balanced training set)
    5.  Held-out test evaluation (X_test/y_test -- never touched by SMOTE or
                                   search)
    6.  Figures                  (confusion matrix, ROC, PR curve, search
                                   score distribution, baseline feature
                                   importance)
    7.  Save model + schema      (model.skops + feature_schema.json)
    8.  Metadata + narrative log
    9.  Push to Hugging Face Hub (optional)
    10. Cross-run aggregation    (separate entry point, run once after all
                                   dataset x model runs are done)

The search space, CV scheme, search budget, and scoring metric are meant to
be held FIXED across all (dataset x model) runs -- only the resulting best
hyperparameters should differ. Pass the same values in every CONFIG; don't
tune the search protocol itself per dataset. As of this version, EVERY
dataset x model run uses the same search method (Optuna/TPE) and the same
fixed budget (N_TRIALS=50, CV=5) by default -- see get_default_search_budget
for an opt-in fallback if a particular run's runtime isn't workable.

This module intentionally mirrors src/preprocessing.py's run_pipeline(config)
orchestration style: one function per numbered step, a single run_training(config)
that calls them in order, and a Logger + FigureRegistry for narrative/visual
output, so training runs read the same way preprocessing runs do.
"""

import os
import json
import time
import platform
import datetime as dt
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split as _tts
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, get_scorer,
)

from src.utils import setup_model_dirs, Logger, FigureRegistry, set_plot_style

try:
    import xgboost
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    _HAS_IMBLEARN = True
except ImportError:
    _HAS_IMBLEARN = False

try:
    import optuna
    from optuna.samplers import TPESampler
    from tqdm.auto import tqdm
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

try:
    import skops.io as sio
    _HAS_SKOPS = True
except ImportError:
    _HAS_SKOPS = False

try:
    from huggingface_hub import HfApi, create_repo
    _HAS_HFHUB = True
except ImportError:
    _HAS_HFHUB = False


# ---------------------------------------------------------------------------
# Fixed search protocol -- SAME for every (dataset x model) run.
# Only the CONFIG's dataset_name/domain/model_type should change between runs;
# these defaults exist so nobody has to (and is less tempted to) retype a
# slightly-different search budget/CV/scoring per dataset.
# ---------------------------------------------------------------------------

DEFAULT_CV_FOLDS = 5
DEFAULT_SEARCH_METHOD = "optuna"   # TPE-sampled Optuna study
DEFAULT_N_TRIALS = 50
DEFAULT_SCORING = "roc_auc"        # used for every run, including imbalanced ones
DEFAULT_RANDOM_STATE = 42

# Row-count tiers for (n_trials, cv_folds) -- an OPT-IN fallback, not applied
# automatically. The default protocol is now a fixed N_TRIALS=50, CV=5 for
# every dataset x model run, full stop. If a specific run's runtime isn't
# workable even with search_sample_size (below), pass e.g.
# config.update(get_default_search_budget(n_rows)) explicitly for that run,
# and note it as a documented exception rather than a silent default.
SEARCH_BUDGET_SMALL_ROW_THRESHOLD = 10_000   # < this many pre-balance train rows
SEARCH_BUDGET_LARGE_ROW_THRESHOLD = 50_000   # >= this many pre-balance train rows
SEARCH_BUDGET_SMALL = {"n_trials": 50, "cv_folds": 5}
SEARCH_BUDGET_MID = {"n_trials": 35, "cv_folds": 4}
SEARCH_BUDGET_LARGE = {"n_trials": 20, "cv_folds": 3}


def get_default_search_budget(n_rows):
    """Opt-in fallback: returns {'n_trials':..., 'cv_folds':...} based on
    pre-balance training row count. NOT applied automatically -- see the
    module docstring / SEARCH_BUDGET_* constants above. Use this only if the
    fixed default (N_TRIALS=50, CV=5) is genuinely too slow for a given run,
    and record that you did so.
    """
    if n_rows < SEARCH_BUDGET_SMALL_ROW_THRESHOLD:
        return dict(SEARCH_BUDGET_SMALL)
    if n_rows >= SEARCH_BUDGET_LARGE_ROW_THRESHOLD:
        return dict(SEARCH_BUDGET_LARGE)
    return dict(SEARCH_BUDGET_MID)


# Optuna parameter suggestion functions, one per model_type. Ranges mirror the
# previous RandomizedSearchCV distributions; min_samples_leaf/min_child_weight
# included for the same SMOTE-related regularization reason noted below.
def _suggest_rf_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        # NOTE: max_depth=None (fully unbounded trees) is deliberately excluded --
        # on the larger datasets, a single unlucky unbounded-depth trial can take
        # dramatically longer than the rest of the search combined, which is what
        # "stuck" searches usually turn out to be. Bounding it keeps runtime far
        # more predictable across all 6 datasets without meaningfully hurting
        # achievable performance.
        "max_depth": trial.suggest_categorical("max_depth", [5, 10, 15, 20, 30]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        # min_samples_leaf is a stronger, more direct regularizer than
        # min_samples_split here specifically: every training set in this
        # project goes through SMOTE, which fills in synthetic points near
        # existing minority samples. Without a leaf-size floor, RF can carve
        # out tiny leaves that fit those synthetic neighborhoods too closely
        # rather than the true minority-class decision boundary.
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def _suggest_xgb_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        # min_child_weight (min sum of instance Hessian in a child) is XGBoost's
        # closest analogue to RF's min_samples_leaf, and for the same
        # SMOTE-related reason above tends to matter more than max_depth alone
        # for keeping the model from overfitting to synthetic minority points.
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


SUGGEST_FUNCS = {"rf": _suggest_rf_params, "xgb": _suggest_xgb_params}


def _build_model(model_type, random_state):
    # n_jobs=-1 on the model itself: Optuna trials run sequentially by
    # default (unlike RandomizedSearchCV, which parallelized across
    # candidate x fold combinations), so letting each individual model fit
    # use all cores is the right place for parallelism here, not oversubscription.
    if model_type == "rf":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)
    if model_type == "xgb":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed; cannot use model_type='xgb'.")
        return XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1)
    raise ValueError(f"Unknown model_type '{model_type}'. Options: {list(SUGGEST_FUNCS)}")


# ---------------------------------------------------------------------------
# 2. Load artifacts
# ---------------------------------------------------------------------------

def load_artifacts(dataset_name, domain, target, datasets_root, logger):
    """Loads everything the preprocessing pipeline produced for this dataset:
    the pre-balance training set (for CV-based search), the SMOTE-balanced
    training set (for the final refit), the held-out test set, and the fitted
    imputer/encoder/scaler bundle (for schema confirmation).
    """
    logger.section("2. LOAD ARTIFACTS")
    data_dir = os.path.join(datasets_root, domain, dataset_name, "processed", "data")
    meta_dir = os.path.join(datasets_root, domain, dataset_name, "metadata")

    def _read(fname):
        return pd.read_csv(os.path.join(data_dir, fname))

    X_train_prebalance = _read("X_train_prebalance.csv")
    y_train_prebalance = _read("y_train_prebalance.csv")[target]
    X_train_balanced = _read("X_train_balanced.csv")
    y_train_balanced = _read("y_train_balanced.csv")[target]
    X_test = _read("X_test.csv")
    y_test = _read("y_test.csv")[target]

    transformers_path = os.path.join(meta_dir, "fitted_transformers.joblib")
    fitted_transformers = joblib.load(transformers_path) if os.path.exists(transformers_path) else None

    dataset_info_path = os.path.join(meta_dir, "dataset_info.json")
    dataset_info = None
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path) as f:
            dataset_info = json.load(f)
        expected_features = set(dataset_info.get("numerical_features", []) + dataset_info.get("categorical_features", []))
        # Encoded categorical columns won't match 1:1 (one-hot expansion), so this
        # is a loose sanity check (feature count could differ) rather than a hard
        # assertion -- just log a warning if the schema looks unexpectedly different.
        if expected_features and not expected_features.intersection(X_train_balanced.columns):
            logger.log(
                "WARNING: none of dataset_info.json's declared features match the "
                "loaded training columns -- confirm you're pointing at the right dataset."
            )

    logger.log(f"Loaded pre-balance train: {X_train_prebalance.shape}")
    logger.log(f"Loaded balanced train:    {X_train_balanced.shape}")
    logger.log(f"Loaded test:               {X_test.shape}")
    logger.log(f"Feature columns ({len(X_train_balanced.columns)}): {list(X_train_balanced.columns)}")
    logger.log(f"Fitted transformers bundle found: {fitted_transformers is not None}")

    return {
        "X_train_prebalance": X_train_prebalance,
        "y_train_prebalance": y_train_prebalance,
        "X_train_balanced": X_train_balanced,
        "y_train_balanced": y_train_balanced,
        "X_test": X_test,
        "y_test": y_test,
        "fitted_transformers": fitted_transformers,
        "dataset_info": dataset_info,
    }


# ---------------------------------------------------------------------------
# 3. Hyperparameter search
# ---------------------------------------------------------------------------

class OptunaSearchResult:
    """Thin wrapper exposing the same interface downstream code already
    expects from an sklearn search object (.best_params_, .best_score_,
    .cv_results_), so refit_final_model/generate_training_figures/
    write_training_metadata don't need to know the search method changed.
    """
    def __init__(self, study, model_type):
        self.study = study
        self.model_type = model_type
        # 'clf__' prefix kept for compatibility with refit_final_model, which
        # strips it back off -- a holdover from the old Pipeline-based search,
        # kept purely so that function didn't need to change.
        self.best_params_ = {f"clf__{k}": v for k, v in study.best_params.items()}
        self.best_score_ = study.best_value

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        columns = {
            "trial_number": [t.number for t in completed],
            "mean_test_score": [t.value for t in completed],
            "duration_seconds": [t.duration.total_seconds() if t.duration else None for t in completed],
        }
        param_names = sorted({k for t in completed for k in t.params})
        for k in param_names:
            columns[f"param_{k}"] = [t.params.get(k) for t in completed]
        self.cv_results_ = columns


def _cv_smote_objective(X, y, model_type, cv_folds, scoring, random_state, logger, verbose):
    """Builds the Optuna objective: for each trial's suggested params, run
    StratifiedKFold CV where SMOTE is fit fresh on each fold's training split
    only (never touching that fold's validation split), and return the mean
    validation score.
    """
    scorer = get_scorer(scoring)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    # Precompute splits once (StratifiedKFold is deterministic given random_state)
    splits = list(cv.split(X, y))

    def objective(trial):
        params = SUGGEST_FUNCS[model_type](trial)
        fold_scores = []
        for train_idx, val_idx in splits:
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            X_tr_bal, y_tr_bal = SMOTE(random_state=random_state).fit_resample(X_tr, y_tr)
            model = _build_model(model_type, random_state)
            model.set_params(**params)
            model.fit(X_tr_bal, y_tr_bal)
            fold_scores.append(scorer(model, X_val, y_val))
        mean_score = float(np.mean(fold_scores))
        # Always recorded to the log FILE for a full trial-by-trial record --
        # also_print=False so this never hits stdout; the progress bar (set up
        # in run_hyperparameter_search) is what the user actually watches live.
        logger.log(
            f"[trial {trial.number:03d}] mean_cv_{scoring}={mean_score:.4f} params={params}",
            also_print=False,
        )
        return mean_score

    return objective


def run_hyperparameter_search(X_train_prebalance, y_train_prebalance, model_type,
                                cv_folds, n_trials, scoring,
                                random_state, logger, verbose=True,
                                search_sample_size=None):
    """Optuna (TPE sampler) hyperparameter search: SMOTE is fit fresh inside
    every CV fold's training split only, never on that fold's validation
    split, and each trial's score is the mean across folds -- the same
    leakage-safety property the old RandomizedSearchCV-based search had.

    verbose: if True, shows a single live progress bar that updates in place
    with the current best trial's score AND params (via a tqdm postfix),
    instead of printing a line per trial -- useful once n_trials x cv_folds
    gets into the hundreds/thousands across a full run of datasets x models.
    Every trial's full detail (params + score) is still written to the log
    FILE regardless (see _cv_smote_objective), just not echoed to stdout.

    search_sample_size: if set, hyperparameter SEARCH runs on a stratified
    random subsample of this many rows from X_train_prebalance instead of the
    full set -- purely a runtime control for large datasets (e.g. loan_default,
    credit_card_fraud_2023). It does not touch the search space, CV scheme,
    budget, or scoring metric, and the final refit (step 4) still uses the
    FULL balanced training set regardless of this setting. Any dataset using
    this should say so explicitly in its CONFIG (e.g. via
    EXTRA_CONFIG_OVERRIDES in the training notebook) so it's visible as a
    deliberate, documented exception rather than a silent inconsistency.
    """
    logger.section("3. HYPERPARAMETER SEARCH")
    if not _HAS_IMBLEARN:
        raise ImportError("imbalanced-learn is not installed; cannot run SMOTE-in-CV search.")
    if not _HAS_OPTUNA:
        raise ImportError("optuna is not installed; cannot run the hyperparameter search.")

    if search_sample_size and len(X_train_prebalance) > search_sample_size:
        X_search, _, y_search, _ = _tts(
            X_train_prebalance, y_train_prebalance,
            train_size=search_sample_size, stratify=y_train_prebalance,
            random_state=random_state,
        )
        logger.log(
            f"search_sample_size={search_sample_size}: searching on a stratified "
            f"subsample ({X_search.shape[0]} of {X_train_prebalance.shape[0]} rows). "
            "Final refit in step 4 still uses the full balanced training set."
        )
    else:
        X_search, y_search = X_train_prebalance, y_train_prebalance

    logger.log(f"model_type: {model_type}")
    logger.log(f"search_method: Optuna TPE, n_trials={n_trials}")
    logger.log(f"cv: StratifiedKFold(n_splits={cv_folds}, shuffle=True, random_state={random_state})")
    logger.log(f"scoring: {scoring}")
    logger.log(
        f"\nAbout to run {n_trials} trials x {cv_folds} folds = {n_trials * cv_folds} total fits. "
        f"{'A live progress bar will track best trial/value/params below; ' if verbose else ''}"
        "full per-trial detail is written to training_log.txt regardless."
    )

    # Suppress Optuna's own per-trial INFO print (one line per trial) --
    # a custom tqdm bar below is the live-status display instead, since
    # Optuna's built-in show_progress_bar only shows best value/trial number
    # and can't be extended with the best trial's params.
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))
    objective = _cv_smote_objective(X_search, y_search, model_type, cv_folds, scoring, random_state, logger, verbose)

    pbar = tqdm(total=n_trials, disable=not verbose, desc=f"{model_type} Optuna search")

    def _progress_callback(study, trial):
        best_params_display = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in study.best_params.items()
        }
        pbar.set_postfix({"best_score": round(study.best_value, 4), **best_params_display})
        pbar.update(1)

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, callbacks=[_progress_callback])
    search_time = time.time() - t0
    pbar.close()

    search = OptunaSearchResult(study, model_type)
    logger.log(f"\nSearch completed in {search_time:.1f}s")
    logger.log(f"Best CV {scoring}: {search.best_score_:.4f}")
    logger.log(f"Best params (pipeline-prefixed): {search.best_params_}")

    return search, search_time


# ---------------------------------------------------------------------------
# 4. Refit final model on the full SMOTE-balanced training set
# ---------------------------------------------------------------------------

def refit_final_model(X_train_balanced, y_train_balanced, model_type, best_params_prefixed,
                       random_state, logger):
    logger.section("4. REFIT FINAL MODEL")
    best_params = {k.split("clf__", 1)[1]: v for k, v in best_params_prefixed.items()}
    logger.log(f"Refitting on full balanced training set with params: {best_params}")

    final_model = _build_model(model_type, random_state)
    final_model.set_params(**best_params)

    t0 = time.time()
    final_model.fit(X_train_balanced, y_train_balanced)
    train_time = time.time() - t0
    logger.log(f"Final refit completed in {train_time:.2f}s on {X_train_balanced.shape[0]} rows")

    return final_model, best_params, train_time


# ---------------------------------------------------------------------------
# 5. Held-out test evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(model, X_test, y_test, logger):
    logger.section("5. HELD-OUT TEST EVALUATION")
    y_pred = model.predict(X_test)

    pos_idx = list(model.classes_).index(1) if hasattr(model, "classes_") else 1
    y_proba = model.predict_proba(X_test)[:, pos_idx]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
    }
    cm = confusion_matrix(y_test, y_pred)

    logger.log("Test metrics:")
    for k, v in metrics.items():
        logger.log(f"  {k}: {v:.4f}")
    logger.log(f"Confusion matrix:\n{cm}")

    return metrics, cm, y_pred, y_proba


def save_test_predictions(metadata_dir, y_test, y_pred, y_proba, logger):
    """Persists one row per test-set instance: row_index (0-based position,
    matching X_test.csv/y_test.csv row order), y_true, y_pred, y_proba_class1.
    This is the join target for anything that references test-set row
    positions later -- e.g. explanations.py's saved instance indices join to
    this file 1:1 on row_index.
    """
    df = pd.DataFrame({
        "row_index": np.arange(len(y_test)),
        "y_true": np.asarray(y_test),
        "y_pred": np.asarray(y_pred),
        "y_proba_class1": np.asarray(y_proba),
    })
    path = os.path.join(metadata_dir, "test_predictions.csv")
    df.to_csv(path, index=False)
    logger.log(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------

def generate_training_figures(model, search, y_test, y_pred, y_proba, cm,
                                feature_names, model_type, scoring, fig_registry, logger):
    logger.section("6. FIGURES")

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    fig_registry.save(fig, "fig_confusion_matrix.png",
                       "Confusion matrix on the held-out test set.")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    fig_registry.save(fig, "fig_roc_curve.png",
                       "ROC curve on the held-out test set.")

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, color="#DD8452", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig_registry.save(fig, "fig_precision_recall_curve.png",
                       "Precision-Recall curve on the held-out test set.")

    # Hyperparameter search score distribution
    cv_scores = search.cv_results_["mean_test_score"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.histplot(cv_scores, kde=True, ax=ax, color="#55A868")
    ax.axvline(search.best_score_, color="black", linestyle="--", lw=1)
    ax.set_xlabel(f"Mean CV {scoring}")
    ax.set_ylabel("Number of trials")
    fig_registry.save(fig, "fig_hyperparameter_search_scores.png",
                       f"Distribution of mean CV {scoring} across all Optuna trials "
                       "(dashed line = best candidate).")

    # Baseline feature importance (model's built-in importance, not SHAP)
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        top = importances.head(20)
        fig, ax = plt.subplots(figsize=(6, 0.35 * len(top) + 2))
        sns.barplot(x=top.values, y=top.index, hue=top.index, palette="viridis", legend=False, ax=ax)
        ax.set_xlabel("Built-in feature importance")
        fig_registry.save(fig, "fig_baseline_feature_importance.png",
                           f"{model_type.upper()} built-in feature importance (Gini/gain), top 20 features -- "
                           "a sanity-check baseline, not a SHAP/LIME explanation.")


def generate_training_curve_figure(model_type, X_train_balanced, y_train_balanced, best_params,
                                     random_state, fig_registry, logger):
    """Training curve, model-type specific -- RF and XGB don't have a
    comparable notion of 'training loss over iterations', so this isn't one
    shared plot:

    - xgb: a genuine train/validation log-loss curve per boosting round, from
      a DIAGNOSTIC fit on an internal 85/15 split of the balanced training
      set. This is separate from the production model saved in step 7, which
      is still fit on the FULL balanced set per the fixed refit protocol --
      this extra fit exists purely to visualize convergence/overfitting.
    - rf: Random Forest is a bagging method, not an iterative loss-minimizer,
      so there's no training loss curve to plot. The closest true analogue is
      out-of-bag (OOB) error vs. number of trees (via warm_start), which is
      what's plotted instead and clearly labeled as such -- don't mistake it
      for the same thing as XGBoost's loss curve.
    """
    logger.section("6b. TRAINING CURVE")

    if model_type == "xgb":
        if not _HAS_XGB:
            logger.log("xgboost not installed -- skipping training loss curve.")
            return
        X_tr, X_val, y_tr, y_val = _tts(
            X_train_balanced, y_train_balanced, test_size=0.15,
            stratify=y_train_balanced, random_state=random_state,
        )
        diag_model = XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=-1)
        diag_model.set_params(**best_params)
        diag_model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
        results = diag_model.evals_result()
        train_loss = results["validation_0"]["logloss"]
        val_loss = results["validation_1"]["logloss"]

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(train_loss, label="train", color="#4C72B0")
        ax.plot(val_loss, label="validation", color="#DD8452")
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("Log loss")
        ax.legend()
        fig_registry.save(
            fig, "fig_training_loss_curve.png",
            "XGBoost train/validation log loss per boosting round, from a diagnostic fit on an internal "
            "85/15 split of the balanced training set (the saved production model is fit on the full "
            "balanced set separately, per the fixed refit protocol)."
        )
        logger.log(f"Final-round train logloss: {train_loss[-1]:.4f}, validation logloss: {val_loss[-1]:.4f}")

    elif model_type == "rf":
        max_estimators = best_params.get("n_estimators", 200)
        step = max(1, max_estimators // 20)
        estimator_range = list(range(step, max_estimators + 1, step))
        if estimator_range[-1] != max_estimators:
            estimator_range.append(max_estimators)

        rf_params = {k: v for k, v in best_params.items() if k != "n_estimators"}
        diag_model = RandomForestClassifier(
            random_state=random_state, n_jobs=-1, warm_start=True, oob_score=True,
            bootstrap=True, **rf_params,
        )
        oob_errors = []
        with warnings.catch_warnings():
            # Expected/harmless for the first few (small n_estimators) points --
            # some samples simply haven't been left out-of-bag yet at low tree
            # counts. Not an error in the computation, just noisy to print.
            warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
            for n in estimator_range:
                diag_model.set_params(n_estimators=n)
                diag_model.fit(X_train_balanced, y_train_balanced)
                oob_errors.append(1 - diag_model.oob_score_)

        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(estimator_range, oob_errors, marker="o", markersize=3, color="#55A868")
        ax.set_xlabel("Number of trees")
        ax.set_ylabel("OOB error rate (1 - OOB accuracy)")
        fig_registry.save(
            fig, "fig_training_loss_curve.png",
            "Random Forest out-of-bag error vs. number of trees -- RF's analogue of a training loss "
            "curve (RF doesn't minimize a loss iteratively the way boosting does, so OOB error vs. tree "
            "count is the closest equivalent convergence check)."
        )
        logger.log(f"Final OOB error rate at n_estimators={max_estimators}: {oob_errors[-1]:.4f}")


# ---------------------------------------------------------------------------
# 7. Save model artifact + schema
# ---------------------------------------------------------------------------

def save_model_artifact(model_dir, model, feature_names, X_train_balanced, logger):
    logger.section("7. SAVE MODEL ARTIFACT + SCHEMA")

    if _HAS_SKOPS:
        model_path = os.path.join(model_dir, "model.skops")
        sio.dump(model, model_path)
        logger.log(f"Saved model (skops): {model_path}")
    else:
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, model_path)
        logger.log(f"skops not installed -- saved model via joblib instead: {model_path}")
        logger.log("Install skops for a safer format if this model will be pushed to a public Hub repo.")

    schema = {
        "feature_order": list(feature_names),
        "dtypes": {c: str(X_train_balanced[c].dtype) for c in feature_names},
    }
    schema_path = os.path.join(model_dir, "feature_schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    logger.log(f"Saved feature schema: {schema_path}")

    return model_path, schema_path


# ---------------------------------------------------------------------------
# 8. Metadata + narrative log
# ---------------------------------------------------------------------------

def write_training_metadata(metadata_dir, logs_dir, config, search, best_params,
                              search_time, train_time, test_metrics, model_path,
                              cv_folds, n_trials, scoring):
    cv_results_path = os.path.join(logs_dir, "cv_results.csv")
    pd.DataFrame(search.cv_results_).to_csv(cv_results_path, index=False)

    best_hp = {
        "best_params": best_params,
        "best_cv_score": float(search.best_score_),
        "scoring": scoring,
    }
    with open(os.path.join(metadata_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_hp, f, indent=2, default=str)

    with open(os.path.join(metadata_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    model_info = {
        "dataset_name": config["dataset_name"],
        "domain": config["domain"],
        "model_type": config["model_type"],
        "model_name": config.get("model_name", config["model_type"]),
        "random_state": config.get("random_state", DEFAULT_RANDOM_STATE),
        "cv_folds": cv_folds,
        "search_method": DEFAULT_SEARCH_METHOD,
        "n_trials": n_trials,
        "scoring": scoring,
        "best_hyperparameters": best_params,
        "search_time_seconds": round(search_time, 2),
        "final_refit_time_seconds": round(train_time, 2),
        "total_runtime_seconds": round(search_time + train_time, 2),
        "trained_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__ if _HAS_XGB else None,
        "model_path": model_path,
        "hub_repo_id": config.get("hub_repo_id"),
        "hub_commit_sha": None,   # filled in by push_to_hub() if used
    }
    model_info_path = os.path.join(metadata_dir, "model_info.json")
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2, default=str)

    return {
        "cv_results_path": cv_results_path,
        "best_hyperparameters_path": os.path.join(metadata_dir, "best_hyperparameters.json"),
        "test_metrics_path": os.path.join(metadata_dir, "test_metrics.json"),
        "model_info_path": model_info_path,
    }


# ---------------------------------------------------------------------------
# 9. Push to Hugging Face Hub (optional)
# ---------------------------------------------------------------------------

def push_to_hub(hub_repo_id, model_path, schema_path, test_metrics_path, model_info_path,
                 dataset_name, domain, model_type, test_metrics, private, logger):
    logger.section("9. PUSH TO HUGGING FACE HUB")
    if not _HAS_HFHUB:
        raise ImportError("huggingface_hub is not installed; cannot push to the Hub.")

    api = HfApi()
    create_repo(hub_repo_id, private=private, exist_ok=True)

    metrics_lines = "\n".join(f"- **{k}**: {v:.4f}" for k, v in test_metrics.items())
    card = (
        f"# {hub_repo_id}\n\n"
        f"- **Dataset**: {dataset_name} ({domain})\n"
        f"- **Model**: {model_type}\n"
        f"- **Split protocol**: 80/20 train/test, SMOTE applied to training data only "
        f"(fresh inside each CV fold during search; on the full training set for the final refit)\n\n"
        f"## Test set metrics\n{metrics_lines}\n\n"
        f"Preprocessing and training were run under a fixed protocol shared across all "
        f"datasets in this study; see the linked preprocessing/training repo for details.\n"
    )
    card_path = model_path + ".README.md.tmp"
    with open(card_path, "w") as f:
        f.write(card)

    last_commit = None
    for local_path, repo_path in [
        (model_path, os.path.basename(model_path)),
        (schema_path, "feature_schema.json"),
        (test_metrics_path, "test_metrics.json"),
        (card_path, "README.md"),
    ]:
        result = api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=hub_repo_id,
        )
        last_commit = getattr(result, "oid", None) or last_commit

    os.remove(card_path)

    if last_commit:
        with open(model_info_path) as f:
            info = json.load(f)
        info["hub_repo_id"] = hub_repo_id
        info["hub_commit_sha"] = last_commit
        with open(model_info_path, "w") as f:
            json.dump(info, f, indent=2, default=str)

    logger.log(f"Pushed to https://huggingface.co/{hub_repo_id} (commit {last_commit})")
    return last_commit


# ---------------------------------------------------------------------------
# Orchestration: run_training
# ---------------------------------------------------------------------------

def run_training(config):
    """Runs one (dataset x model) training run, given a CONFIG dict.

    Required config keys:
        dataset_name, domain, model_type ('rf' or 'xgb')

    Optional config keys (defaults keep the search protocol IDENTICAL across
    every run -- override only if you deliberately want a different protocol
    for every dataset, which defeats the point of a fixed protocol):
        target                  (default: read from Datasets/.../metadata/dataset_info.json)
        model_name              (default: model_type; used for the Models/ folder name)
        datasets_root           (default: "Datasets")
        models_root             (default: "Models")
        cv_folds                (default: 5 -- FIXED for every run; see get_default_search_budget
                                   for an opt-in fallback if a specific run's runtime isn't workable)
        n_trials                (default: 50 -- FIXED for every run; same opt-in fallback as above)
        scoring                 (default: "roc_auc")
        random_state            (default: 42)
        search_sample_size      (default: None; stratified subsample size used only for the
                                   search step, e.g. 50_000 for the largest datasets)
        push_to_hub             (default: False)
        hub_repo_id             (required if push_to_hub=True, e.g. "yourname/xai-pima-rf")
        hub_private             (default: True)
    """
    for key in ("dataset_name", "domain", "model_type"):
        if key not in config:
            raise ValueError(f"CONFIG is missing required key '{key}'")

    model_name = config.get("model_name", config["model_type"])
    random_state = config.get("random_state", DEFAULT_RANDOM_STATE)
    scoring = config.get("scoring", DEFAULT_SCORING)
    datasets_root = config.get("datasets_root", "Datasets")
    models_root = config.get("models_root", "Models")
    # Fixed by default for every (dataset x model) run -- N_TRIALS=50, CV=5.
    # Only deviates if the CONFIG explicitly sets cv_folds/n_trials (e.g. by
    # calling get_default_search_budget(n_rows) yourself for one run and
    # documenting why), never automatically.
    cv_folds = config.get("cv_folds", DEFAULT_CV_FOLDS)
    n_trials = config.get("n_trials", DEFAULT_N_TRIALS)
    verbose = config.get("verbose", True)
    search_sample_size = config.get("search_sample_size")

    # 1. Setup
    paths = setup_model_dirs(config["dataset_name"], config["domain"], model_name, models_root)
    set_plot_style()
    logger = Logger(paths["logs"], filename="training_log.txt")
    fig_registry = FigureRegistry(paths["figures"], logger)
    logger.section("1. CONFIGURATION")
    logger.log(json.dumps(config, indent=2, default=str))
    logger.log(f"Search budget: n_trials={n_trials}, cv_folds={cv_folds} (search_method={DEFAULT_SEARCH_METHOD})")

    target = config.get("target")
    if target is None:
        info_path = os.path.join(datasets_root, config["domain"], config["dataset_name"], "metadata", "dataset_info.json")
        with open(info_path) as f:
            target = json.load(f)["target"]
        logger.log(f"target not given in config -- read '{target}' from dataset_info.json")

    # 2. Load artifacts
    artifacts = load_artifacts(config["dataset_name"], config["domain"], target, datasets_root, logger)

    # 3. Hyperparameter search (SMOTE refit inside each CV fold, pre-balance data)
    search, search_time = run_hyperparameter_search(
        artifacts["X_train_prebalance"], artifacts["y_train_prebalance"],
        config["model_type"], cv_folds, n_trials, scoring,
        random_state, logger, verbose=verbose, search_sample_size=search_sample_size,
    )

    # 4. Refit final model on the full SMOTE-balanced training set
    final_model, best_params, train_time = refit_final_model(
        artifacts["X_train_balanced"], artifacts["y_train_balanced"],
        config["model_type"], search.best_params_, random_state, logger,
    )

    # 5. Held-out test evaluation
    test_metrics, cm, y_pred, y_proba = evaluate_on_test(
        final_model, artifacts["X_test"], artifacts["y_test"], logger,
    )
    save_test_predictions(paths["metadata"], artifacts["y_test"], y_pred, y_proba, logger)

    # 6. Figures
    feature_names = list(artifacts["X_train_balanced"].columns)
    generate_training_figures(
        final_model, search, artifacts["y_test"], y_pred, y_proba, cm,
        feature_names, config["model_type"], scoring, fig_registry, logger,
    )
    generate_training_curve_figure(
        config["model_type"], artifacts["X_train_balanced"], artifacts["y_train_balanced"],
        best_params, random_state, fig_registry, logger,
    )

    # 7. Save model + schema
    model_path, schema_path = save_model_artifact(
        paths["model"], final_model, feature_names, artifacts["X_train_balanced"], logger,
    )

    # 8. Metadata + narrative log
    written = write_training_metadata(
        paths["metadata"], paths["logs"], config, search, best_params,
        search_time, train_time, test_metrics, model_path,
        cv_folds, n_trials, scoring,
    )

    # 9. Push to Hugging Face Hub (optional)
    hub_commit_sha = None
    if config.get("push_to_hub"):
        if not config.get("hub_repo_id"):
            raise ValueError("push_to_hub=True requires config['hub_repo_id']")
        hub_commit_sha = push_to_hub(
            config["hub_repo_id"], model_path, schema_path,
            written["test_metrics_path"], written["model_info_path"],
            config["dataset_name"], config["domain"], config["model_type"],
            test_metrics, config.get("hub_private", True), logger,
        )

    logger.section("DONE")
    logger.log(f"Best CV {scoring}: {search.best_score_:.4f}  |  Test {scoring}: {test_metrics.get(scoring):.4f}")

    return {
        "paths": paths,
        "search": search,
        "best_params": best_params,
        "final_model": final_model,
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "model_path": model_path,
        "schema_path": schema_path,
        "hub_commit_sha": hub_commit_sha,
    }


# ---------------------------------------------------------------------------
# 10. Cross-run aggregation (call once, after all dataset x model runs)
# ---------------------------------------------------------------------------

def aggregate_model_results(models_root="Models", output_subdir="summary"):
    """Scans Models/<domain>/<dataset>/<model>/metadata/{model_info,test_metrics}.json
    for every run found under models_root, and builds one comparison table +
    figure across the whole matrix. Run this once all (dataset x model) runs
    are complete.
    """
    rows = []
    for domain in sorted(os.listdir(models_root)):
        domain_path = os.path.join(models_root, domain)
        if not os.path.isdir(domain_path) or domain == output_subdir:
            continue
        for dataset_name in sorted(os.listdir(domain_path)):
            dataset_path = os.path.join(domain_path, dataset_name)
            if not os.path.isdir(dataset_path):
                continue
            for model_name in sorted(os.listdir(dataset_path)):
                meta_dir = os.path.join(dataset_path, model_name, "metadata")
                info_path = os.path.join(meta_dir, "model_info.json")
                metrics_path = os.path.join(meta_dir, "test_metrics.json")
                if not (os.path.exists(info_path) and os.path.exists(metrics_path)):
                    continue
                with open(info_path) as f:
                    info = json.load(f)
                with open(metrics_path) as f:
                    metrics = json.load(f)
                rows.append({
                    "domain": domain,
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    **metrics,
                    "best_hyperparameters": json.dumps(info.get("best_hyperparameters")),
                })

    summary_df = pd.DataFrame(rows)
    out_dir = os.path.join(models_root, output_subdir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "model_performance_summary.csv")
    summary_df.to_csv(csv_path, index=False)

    if not summary_df.empty:
        set_plot_style()
        metric_cols = [c for c in ["roc_auc", "f1"] if c in summary_df.columns]
        long_df = summary_df.melt(
            id_vars=["domain", "dataset_name", "model_name"],
            value_vars=metric_cols, var_name="metric", value_name="score",
        )
        long_df["dataset_model"] = long_df["dataset_name"] + " (" + long_df["model_name"] + ")"

        fig, ax = plt.subplots(figsize=(max(8, 0.6 * summary_df["dataset_name"].nunique() * 2), 5))
        sns.barplot(data=long_df, x="dataset_model", y="score", hue="metric", ax=ax)
        ax.set_xlabel("Dataset (model)")
        ax.set_ylabel("Test set score")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        fig_path = os.path.join(out_dir, "fig_performance_comparison_across_datasets.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
    else:
        fig_path = None

    return {"summary_df": summary_df, "csv_path": csv_path, "fig_path": fig_path}
