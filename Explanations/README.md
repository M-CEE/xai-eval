# Explanations — Stage 3: Generation, Versioning, and Artifact Management

This README describes the Explanations/ folder: what explanation artifacts are produced, the exact protocol used (and why), how to reproduce runs using the code in src/, how to interpret and join artifacts to models and test predictions, and the research-grounded justification for the defaults chosen. In-text citations appear where relevant and a reference list follows.

High-level summary
------------------
- Explanations/ contains deterministic, versioned explanation artifacts (SHAP and LIME) for each dataset × model run in the experiment.
- The code that produced these artifacts is src/explanations.py and its orchestration function run_explanations(config).
- The project enforces reproducibility and comparability by: (a) selecting a fixed set of test instances per run (saved and reused), (b) selecting two case instances (one confidently correct, one confidently incorrect) per run for single-instance plots, and (c) saving explainer outputs and metadata with library versions and random seeds.
- A master protocol file Explanations/experiment_protocol.json records the fixed rules shared across all runs (instance selection thresholds / sizes, explainer configuration, case-selection rule). This file is authoritative for how runs were produced.

Folder layout (per run)
-----------------------
Each run is stored at Explanations/<domain>/<dataset_name>/<model_name>/ with a sub-tree produced by src/explanations.setup_explanation_dirs():

- instances/
  - metadata.json — recorded instance-selection details (dataframe_indices, sampling strategy, class distribution)
  - X_explain.csv, y_explain.csv — the actual selected feature values and labels explained (these are the model-ready, post-impute/encode/scale values)
  - case_instances.json — two selected case instances (correct and incorrect) with positions and prediction/confidence info

- shap/
  - shap_values.npy — saved SHAP values array (n_instances × n_features) for *class 1* (positive) explanations
  - base_values.npy — SHAP base values
  - feature_importance.csv — top-N features by mean |SHAP value|
  - metadata.json — explainer metadata (shap version, runtime, feature order, model reference)
  - plots/ (shap_plots/) — beeswarm / bar / waterfall plots (summary.png, bar.png, waterfall_case1.png, waterfall_case2.png)

- lime/
  - lime_explanations.csv — long-form per-instance LIME linear-term weights (instance_id, feature, weight)
  - lime_instance_summary.csv — per-instance local intercept/local_pred/score/predicted_proba
  - metadata.json — explainer metadata (lime version, num_samples, background data source, runtime)
  - plots/ (lime_plots/) — the two case-instance LIME local-explanation plots (case1.png, case2.png)

- logs/ — explanation_log.txt narrative log
- metadata/ — top-level run metadata.json summarizing the run (model reference, dataset characteristics, selection_info, case_info)

What the code does (step-by-step)
---------------------------------
The explanation-generation orchestration in src/explanations.py (run_explanations(config)) follows these steps:

1. Setup directories and a Logger for narrative output.
2. Load model and data with load_model_and_data():
   - Loads the serialized model artifact from Models/<domain>/<dataset>/<model>/model/. If saved with skops, skops is used; otherwise joblib is used.
   - Loads the saved feature_schema.json to preserve the exact feature order the model expects.
   - Loads X_test.csv, y_test.csv and X_train_balanced (background data for LIME) from Datasets/.../processed/data/ using the saved feature order.
   - Loads model_info.json and best_hyperparameters.json from the model's metadata/ so the explainer metadata can include the training context and test performance.

3. Select (or reuse) an explanation subset with select_or_load_instances():
   - If the test set size >= 1000 rows, draws a stratified random sample of 500 rows (DEFAULT_N_INSTANCES = 500) preserving class proportions.
   - If the test set < 1000 rows, the entire test set is used.
   - The same selected rows are saved to instances/X_explain.csv and instances/metadata.json. Re-running the module for the same run will reuse these rows unless force=True.
   - The saved selection contains dataframe_indices which are 0-based positions into Datasets/.../processed/data/X_test.csv and align 1:1 with Models/.../metadata/test_predictions.csv for joining.

3b. Select (or reuse) case instances with select_or_load_case_instances():
   - Picks one correctly classified and one incorrectly classified instance FROM the selected explanation subset.
   - Default rule: 'highest_confidence' (the most confident correct and most confident incorrect instance). Alternatives: 'first', 'median_confidence'. Confidence is defined as the model's predicted probability of its own predicted class.
   - Saved as instances/case_instances.json and recorded in run-level metadata.

4. SHAP (compute_and_save_shap):
   - Uses shap.TreeExplainer for tree ensembles (RF and XGB). TreeExplainer with no background data uses feature_perturbation='tree_path_dependent' which is exact/deterministic for tree ensembles — deterministic outputs are saved so later runs or analyses use the same values (Lundberg & Lee, 2017).
   - Saves raw arrays (shap_values.npy, base_values.npy) — the code normalizes the saved SHAP arrays to a consistent shape (n_instances × n_features) even when shap returns class slices.
   - Exports a top-N feature importance CSV (mean |SHAP value|) and standard summary plots: beeswarm (summary.png), bar (bar.png), and waterfalls for the two case instances (waterfall_case1.png and waterfall_case2.png where available).
   - Writes shap/metadata.json with library versions, runtime, feature order and other context so any later audit can reproduce the same interpretation pipeline.

5. LIME (compute_and_save_lime):
   - Uses lime.lime_tabular.LimeTabularExplainer with the balanced training set (X_train_balanced) as the background explanation dataset. This is a deliberate choice: LIME's local surrogate models are more meaningful when the background reflects the distribution the model was fit on.
   - Default per-instance perturbation budget = 1000 perturbed samples (DEFAULT_LIME_NUM_SAMPLES) and number of features returned per instance = 50 (DEFAULT_LIME_NUM_FEATURES). These parameters trade off fidelity and runtime — LIME runtime scales approximately with n_instances × num_samples × model prediction cost.
   - Saves the long-form per-instance LIME weights (lime_explanations.csv) and per-instance summary (lime_instance_summary.csv) and two case-instance plot PNGs.
   - Writes lime/metadata.json with settings, library version, runtime, and background data source. LIME uses a random seed (passed into the explainer) to make results reproducible; the run-level metadata captures the random_state.

6. Top-level run metadata: the module writes an Explanations/.../metadata/metadata.json that bundles selection_info, case_info, dataset_characteristics and a model_reference portion so that anyone inspecting only the Explanations/ folder can see the model performance and hyperparameters used in the underlying model.

Why these choices were made (research-grounded justifications)
------------------------------------------------------------
- Deterministic, versioned instance selection: ensures that SHAP vs LIME comparisons are performed on identical rows and that plots in the paper refer to the same cases. This avoids introducing sampling variance into explanation-quality comparisons.

- SHAP TreeExplainer for tree ensembles: TreeExplainer produces exact or fast, stable approximations tailored to tree models and is deterministic for trees when used without a sampling background (Lundberg & Lee, 2017). Saving raw arrays allows later statistical analyses of explanation magnitudes without recomputing.

- LIME with balanced-training background and fixed random seed: LIME is a local, perturbation-based explainer; choosing X_train_balanced as the background grounds the perturbation distribution in the data the model learned from. The random_state + recorded num_samples produces reproducible outputs that are auditable (Ribeiro et al., 2016).

- Two case instances (confident correct vs confident incorrect): choosing the confident incorrect case prioritizes failure modes that are most concerning in high-stakes domains (healthcare, finance). A confident-but-wrong explanation highlights problematic learned patterns the paper should analyze and discuss.

- Saving explainer metadata (library versions, runtime, model reference): explanation behaviour can be sensitive to library versions and explainer options (e.g., SHAP reshaping behavior changed across versions). Storing this metadata ensures later re-analysis can check or re-run under the same conditions.

Determinism, idempotency, and safe re-runs
-----------------------------------------
- The module treats selection and case-choice artifacts as canonical: if instances/metadata.json (and X_explain.csv) already exist the code reuses them unless force=True. This guarantees re-runs do not silently explain a different set of instances.
- SHAP TreeExplainer is deterministic for tree ensembles; SHAP arrays are saved and re-used to avoid recomputation and to ensure identical outputs across analyses.
- LIME is inherently stochastic in perturbation-based sampling; the explainer receives a random_state and the run metadata records the random seed and num_samples to enable re-running with the same seed for reproducibility.

Artifacts and how to use them (practical guidance)
-------------------------------------------------
- To inspect what instances were explained for a run, open Explanations/<domain>/<dataset>/<model>/instances/metadata.json — it contains dataframe_indices (0-based positions into Datasets/.../processed/data/X_test.csv). Use these indices to join explanations to test-set row predictions saved under Models/.../metadata/test_predictions.csv.

- SHAP artifacts:
  - Load arrays: values = np.load('shap_values.npy')  # shape: (n_instances, n_features)
  - Load base values: base = np.load('base_values.npy')
  - Load feature importance CSV for the top-N features and use for global-ranking tables in the paper.
  - Plots are saved under shap/plots/ (beeswarm, bar, waterfall). Captions are recorded in the run-level figure registry (figure_captions.txt) if present.

- LIME artifacts:
  - lime_explanations.csv contains rows (instance_id, feature, weight) which can be aggregated, filtered or compared to SHAP at the per-feature level. instance_id corresponds to the dataframe_index value saved in instances/metadata.json.
  - lime_instance_summary.csv contains per-instance predicted probabilities and local surrogate metrics (intercept, local_pred, local R^2-like score) useful for evaluating LIME fidelity.

- run-level metadata (Explanations/.../metadata/metadata.json) is the first place to look: it bundles the model reference (best hyperparameters, test performance), dataset characteristics (n_features, feature names, test/train sizes), selection_info and case_info, and generated_at_utc. This file is self-contained enough for basic audit without the Models/ tree.

Reproducing explanation artifacts
---------------------------------
1. Ensure Models/<domain>/<dataset>/<model>/ exists and contains a saved model artifact (model.skops or model.joblib) and model/feature_schema.json.
2. Ensure Datasets/<domain>/<dataset>/processed/data contains X_test.csv, y_test.csv and X_train_balanced.csv (the latter used as LIME background).
3. Build a run CONFIG dict with keys {dataset_name, domain, model_name} and optional keys as in src/explanations.run_explanations (random_state, small_test_threshold, n_instances, lime_num_samples, lime_num_features, case_selection_rule, force).
4. Call src.explanations.run_explanations(config). Per-run artifacts will be written under Explanations/<domain>/<dataset>/<model>/.

Practical runtime notes
----------------------
- SHAP (TreeExplainer) is relatively fast for tree ensembles; runtimes are recorded in shap/metadata.json. If shap is not installed, generation will raise ImportError at SHAP step.
- LIME is expensive: runtime scales with n_instances × num_samples × model prediction cost. Default num_samples=1000 and n_instances=500 means ~500k model evaluations; on large datasets/models this is a heavy computation. Use smaller num_samples or fewer instances for debugging.
- The code supports partial regeneration: if the main LIME data exists but case plots are missing, the module regenerates just the case plots (two explain_instance() calls) rather than the entire n_instances loop.

Example files observed in this archive
-------------------------------------
- Explanations/experiment_protocol.json — authoritative protocol (instance selection threshold 1000; target 500 instances; case selection rule 'highest_confidence'; SHAP TreeExplainer; LIME num_samples=1000, num_features=50).
- Per-run metadata examples: Explanations/healthcare/pima_diabetes/rf/metadata/metadata.json and Explanations/finance/loan_default/xgb/metadata/metadata.json. These show the saved model_reference (path to Models/.../model.skops), dataset_characteristics, selection_info (dataframe_indices), and case_info (chosen positions + predicted probabilities and confidence). These files also demonstrate the recorded test performance copied from model_info.json and the generated_at_utc timestamps for provenance.

Why the selection & case rules matter for analysis
-------------------------------------------------
- Using a fixed selection rule (stratified sample of test or full test when small) ensures SHAP and LIME are compared on identical rows. Many explanation-quality metrics (stability, fidelity, rank-correlation) are sensitive to the chosen instances; fixing the instance set avoids conflating selection noise with explainer differences.
- Picking confident incorrect cases surfaces the most alarming failure modes and is more informative for a write-up focused on model safety in sensitive domains.

References
----------
- Lundberg, S.M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In Proceedings of NIPS. (TreeSHAP / SHAP methodology).
- Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. In Proceedings of KDD (LIME).
- Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321–357.
- Lemaître, G., Nogueira, F., & Aridas, C.K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. Journal of Machine Learning Research.
- Saito, T., & Rehmsmeier, M. (2015). The precision–recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE.

If you'd like, the README can be expanded with one of the following (pick one):
- a short code snippet that programmatically loads a given run's SHAP/LIME artifacts and demonstrates joining with Models/.../metadata/test_predictions.csv;
- a small table summarizing generated runs (dataset, model_name, n_instances explained, presence of shap/lime artifacts) derived automatically from the Explanations tree;
- or a short section explaining how to re-generate just the case-instance plots without recomputing all explanations.

If you want any of those added, tell me which and I'll update the README accordingly.