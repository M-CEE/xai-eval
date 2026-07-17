# Models — Stage 2: Training, Evaluation, and Artifact Management

This document describes everything recorded under the Models/ folder: what was executed by the code in src/, what artifacts are produced and why, and the methodological justifications for the implementation choices made. In-text citations link to the reference list at the end; the citation style matches the project's Datasets/ README.

Summary
-------
- Models/ stores the trained models, training logs, hyperparameter search results, figures, and per-run metadata for each dataset × model family run in the experiment.
- Each run follows a fixed, reproducible training protocol implemented in `src/training.py: run_training(config)`. The protocol enforces SMOTE-in-CV for search, a fixed Optuna TPE search budget by default (n_trials=50, cv=5), a final refit on the full SMOTE-balanced training set, and held-out test evaluation.
- Explanation-generation (SHAP / LIME) is performed separately under Explanations/ (see src/explanations.py). Models/ artifacts are the inputs for explanation and evaluation stages.

Folder layout (per run)
-----------------------
Each run is stored at Models/<domain>/<dataset_name>/<model_name>/ with the following structure (created by src/utils.setup_model_dirs):

- model/ — serialized model artifact and feature schema (model.skops or model.joblib + feature_schema.json)
- logs/ — narrative training log (training_log.txt) and cv results (cv_results.csv)
- figures/ — saved performance and diagnostic plots (captions recorded in figure_captions.txt)
- metadata/ — structured metadata JSONs: model_info.json, best_hyperparameters.json, test_metrics.json

A cross-run summary is created under Models/summary/ containing `model_performance_summary.csv` and a comparative figure `fig_performance_comparison_across_datasets.png` (produced by aggregate_model_results()).

What was executed (the training pipeline)
-----------------------------------------
The training pipeline is implemented in src/training.py and is invoked via run_training(config). The numbered steps are:

1. Configuration and directory setup (Logger, FigureRegistry).
2. Load preprocessing artifacts produced by src/preprocessing.py (X_train_prebalance, y_train_prebalance, X_train_balanced, y_train_balanced, X_test, y_test, fitted transformers).
3. Hyperparameter search using Optuna (TPE sampler) with a fixed search protocol by default (n_trials=50, cv=5). For each Optuna trial, a StratifiedKFold CV is run where SMOTE is fit only on each fold's training split (SMOTE-in-CV) and scoring uses a fixed metric (ROC AUC by default). Full trial results are written to logs/cv_results.csv and training_log.txt.
4. Refit final model with the winning hyperparameters on the entire SMOTE-balanced training set.
5. Held-out test evaluation (metrics: accuracy, precision, recall, f1, roc_auc, pr_auc) and confusion matrix. Test predictions are saved (metadata/test_predictions.csv) to support joining explanations and downstream analyses.
6. Figures: confusion matrix, ROC, Precision–Recall curve, hyperparameter-search score distribution, and a baseline (built-in) feature importance plot. For model diagnostics, model-specific training-curve diagnostics are produced: XGBoost log-loss vs boosting round (diagnostic fit on an internal split) or RandomForest OOB error vs number of trees.
7. Save model artifact (skops if available, otherwise joblib) and feature_schema.json (feature order and dtypes) so explainer code can rebuild the exact input ordering used for training.
8. Write structured metadata: best_hyperparameters.json, test_metrics.json, model_info.json (includes runtime, library versions, and model path), and save cv_results.csv in logs/.
9. Optional: push selected model artifacts to the Hugging Face Hub (src/training.push_to_hub). The pipeline records hub repo id and commit sha inside model_info.json when used.
10. Cross-run aggregation (aggregate_model_results) produces summary CSV and comparative figure under Models/summary/ after runs complete.

Artifacts produced
------------------
Per-run artifacts (examples under Models/finance/credit_card_fraud_2023/xgb etc.):
- model/model.skops or model.joblib — serialized model
- model/feature_schema.json — feature order and dtypes
- metadata/best_hyperparameters.json — chosen hyperparameters, best CV score
- metadata/test_metrics.json — held-out test metrics
- metadata/model_info.json — run metadata (timing, versions, paths, hub info)
- logs/training_log.txt — narrative log containing per-trial lines and step summaries
- logs/cv_results.csv — table of completed Optuna trials with mean CV score and per-trial params
- figures/*.png and figures/figure_captions.txt — diagnostic and performance plots
- Models/summary/model_performance_summary.csv and fig_performance_comparison_across_datasets.png — cross-run aggregation outputs

Notable run-level activities recorded here
-----------------------------------------
- Hub upload summary (Models/hub_upload_log.txt) lists which runs were pushed to the Hugging Face Hub and which were skipped because they already existed. For this archive, seven model repos were uploaded and five were skipped (already uploaded). See hub_upload_log.txt for commit shas and uploaded paths.

Implementation choices and justifications (research-grounded)
------------------------------------------------------------
1) Fixed shared training protocol across runs
   - Rationale: using one shared search budget and evaluation protocol across datasets and model families removes a source of methodological variability: differences in preprocessing/search budget/scoring would make cross-dataset comparisons confounded. A fixed default (Optuna TPE, n_trials=50, cv=5) is therefore used by default and only changed with explicit documentation. This mirrors the identical-pipeline principle used in the preprocessing stage for fair comparisons.

2) SMOTE applied only on training data, and re-fit inside each CV fold
   - Reason: Synthetic oversampling before CV causes synthetic minority points derived from an original training point to leak into validation folds (data leakage), inflating performance estimates. The pipeline fits SMOTE inside each fold's training split only, and final refit uses the full SMOTE-balanced training set — this prevents leakage during search while enabling the final model to learn from a balanced dataset (Chawla et al., 2002; Lemaître et al., 2017).
   - Citations: Chawla et al. (2002) introduced SMOTE; the imbalanced-learn library documents correct SMOTE-in-CV usage (Lemaître et al., 2017).

3) Optuna (TPE) hyperparameter search and search-space choices
   - Optuna (TPE sampler) is used for efficient, scalable hyperparameter search (Akiba et al., 2019). The search-space bounds are chosen to balance expressiveness with runtime predictability: e.g., Random Forest max_depth is restricted to a small set of depths rather than None to avoid extremely long/unbounded trials that can dominate wall-clock time on larger datasets. Similarly, min_samples_leaf (RF) and min_child_weight (XGB) are included to regularize models that are trained on SMOTE-expanded minority neighborhoods (SMOTE can encourage fine-grained, synthetic clusters that an unconstrained tree can overfit to).

4) Scoring: ROC AUC (primary) and PR AUC (reported)
   - ROC AUC is used as the fixed CV scoring metric (DEFAULT_SCORING = 'roc_auc') because it is threshold-independent and widely comparable across runs. Precision–Recall AUC (pr_auc) is also computed and stored since PR curves are often more informative than ROC curves under class imbalance (Saito & Rehmsmeier, 2015). Both are reported in test_metrics.json.

5) Final refit on full balanced training set, held-out test never touched
   - The search only uses pre-balance training examples (with SMOTE refit inside CV) while the final production model is trained on the full SMOTE-balanced training set with the chosen hyperparameters. The held-out test set is never used for SMOTE or search, preserving an unbiased evaluation (Kaufman et al., 2012).

6) Serialization & schema
   - Models are saved with skops when available (safer cross-version deserialization for scikit-learn objects), otherwise joblib. The exact feature order and dtypes are saved in model/feature_schema.json so downstream explainers and consumers load test instances with exactly the same column ordering that the model was trained on.

7) Diagnostics and reproducibility metadata
   - Comprehensive per-run metadata is written: best_hyperparameters.json, cv_results.csv, model_info.json (including python and library versions), and training_log.txt. This makes runs reproducible and auditable and ensures that explanation artifacts can be linked back to the model and to test predictions.

8) Choice of model families
   - Random Forest (bagging) and XGBoost (gradient boosting) were chosen as standard, high-performing tree-ensemble baselines for tabular data; they are amenable to fast tree-specific explainers (TreeSHAP) and are widely used in the literature (Chen & Guestrin, 2016). RandomForest is implemented via sklearn.ensemble.RandomForestClassifier; XGBoost via xgboost.XGBClassifier when xgboost is installed.

Operational notes (what to look for in the artifacts)
-----------------------------------------------------
- training_log.txt records the step-by-step narrative for each run. Look here for the per-trial Optuna lines and any logged warnings (e.g., if a dataset used search_sample_size to speed up search).
- logs/cv_results.csv contains the Optuna-derived table of completed trials; best candidate is mirrored in metadata/best_hyperparameters.json.
- metadata/test_metrics.json contains the held-out test metrics (accuracy, precision, recall, f1, roc_auc, pr_auc). Use these for cross-run comparisons, but consult Models/summary/model_performance_summary.csv for the aggregated table across runs.
- figures/ contains plots used in the paper (ROC, PR, confusion matrix, hyperparameter score distributions, training-curve diagnostics). Captions are in figures/figure_captions.txt.
- model/model.skops or model.joblib + model/feature_schema.json: load the model and apply the saved fitted_transformers (Datasets/.../metadata/fitted_transformers.joblib) to raw inputs to reproduce predictions.

Hub uploads performed in this archive
-----------------------------------
The script that pushes to the Hugging Face Hub wrote a short hub upload log in Models/hub_upload_log.txt. Key points from this archive run:
- Uploaded: finance/credit_card_fraud_2023/{rf,xgb}, finance/financial_distress/{rf,xgb}, finance/loan_default/{rf,xgb}, healthcare/pima_diabetes/rf.
- Skipped (already uploaded): healthcare/{breast_cancer_wisconsin,heart_disease_uci}/{rf,xgb}, healthcare/pima_diabetes/xgb.

(See Models/hub_upload_log.txt for full commit SHAs and repo ids.)

How to reproduce a single run
-----------------------------
1. Ensure the Datasets/ preprocessing artifacts exist for the dataset (run src/preprocessing.run_pipeline(config) or use the provided processed artifacts where available).
2. Create a CONFIG dict matching the required keys (dataset_name, domain, model_type) and any optional overrides (search_sample_size, n_trials, cv_folds, push_to_hub, hub_repo_id). See the notebooks/0N_preprocessing_*.ipynb and notebooks/07_training_all_models.ipynb for example configs used in the study.
3. Call src.training.run_training(config). It will write artifacts under Models/<domain>/<dataset_name>/<model_name>/.
4. After completing all runs, call src.training.aggregate_model_results(models_root="Models") to regenerate Models/summary/model_performance_summary.csv and the summary figure.

Limitations and notes for the paper
----------------------------------
- The pipeline enforces identical search and evaluation protocols across datasets to enable fair comparison, but this means that dataset-specific hyperparameter tuning of the search protocol is intentionally not performed; any deviation must be explicitly documented in the run's config and metadata.
- The financial_distress dataset uses a row-level split rather than a per-company grouped split — this is a documented limitation carried forward from the preprocessing design (see Datasets/ README).
- If any of the optional libraries are not installed (xgboost, shap, lime, optuna, imbalanced-learn, skops), the code will either skip certain features or raise a clear ImportError at the point of use; see the logs for the exact error and the run's metadata for recorded library versions.

Selected citations (in-text uses above)
---------------------------------------
- SMOTE and SMOTE-in-CV guidance: Chawla et al., 2002; Lemaître et al., 2017.
- Hyperparameter optimization framework: Akiba et al., 2019 (Optuna).
- Tree-based explainers and model-agnostic explainers: Lundberg & Lee, 2017 (SHAP); Ribeiro, Singh & Guestrin, 2016 (LIME).
- XGBoost: Chen & Guestrin, 2016.
- On evaluation under imbalanced data and PR vs ROC: Saito & Rehmsmeier, 2015.
- On leakage and train/test protocol: Kaufman et al., 2012.

References
----------
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
- Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321–357.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
- Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in Data Mining: Formulation, Detection, and Avoidance. ACM Transactions on Knowledge Discovery from Data, 6(4), 1–21.
- Lemaître, G., Nogueira, F., & Aridas, C.K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. Journal of Machine Learning Research, 18(17), 1–5.
- Lundberg, S.M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In Proceedings of the 31st Conference on Neural Information Processing Systems (NIPS). (TreeSHAP introduced here; later SHAP packages extend this work.)
- Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
- Saito, T., & Rehmsmeier, M. (2015). The precision–recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE.

If anything in this README needs to be expanded (for example you want more per-run examples, additional interpretation guidance for the summary CSV, or a suggested analysis script), tell me which part to expand and whether to include code snippets for loading artifacts programmatically.