# Fidelity: Methodology, Formulas, and Interpretation Guide

## 1. Purpose and Scope

This document specifies the theoretical foundation, formal definitions, and exact computational procedure behind the **Fidelity** metric family in this evaluation pipeline, as implemented in `src/fidelity.py`. It is written to research-report standard: every design decision is either derived directly from an established method in the literature or explicitly flagged as a project-specific adaptation, with the reasoning stated. It also covers how to interpret the resulting numbers and what they imply for the project's central research question — whether explanation-quality metrics behave consistently across the healthcare and finance domains.

Fidelity is one of five metric families evaluated in this study (alongside Stability, Robustness, Simplicity, and Efficiency), applied across 6 datasets (3 healthcare, 3 finance), 2 models (Random Forest, XGBoost), and 2 explainers (SHAP, LIME).

## 2. Conceptual Definition

**Fidelity** (used interchangeably with **faithfulness** in the literature) measures whether an explanation accurately reflects what a model actually relied on to make a prediction — as distinct from whether the explanation merely *looks* plausible to a human observer [1]. A feature-attribution method can produce an explanation that is intuitive and visually convincing while being causally disconnected from the model's actual decision process; fidelity is the property that rules this out.

This is a **functionally-grounded** evaluation method in the taxonomy of Doshi-Velez and Kim [1] — no human judgment is involved. A formal proxy (change in model output under controlled perturbation of the input) stands in for "did the explanation get it right," which is what makes fidelity scalable across a large experimental matrix like this one, unlike human-grounded trust or usefulness studies.

## 3. Theoretical Foundation: Perturbation-Based Faithfulness

The dominant approach to measuring fidelity — and the one used here — is **perturbation-based**: features identified as important by an explanation are removed (or isolated) from an instance, and the resulting change in the model's prediction is measured. The logic is direct: if an explanation correctly identifies the features the model actually used, removing those features should cause a large, measurable change in the model's output; removing unimportant features should not.

This approach originates in the *region perturbation* methodology of Samek et al. [5], who introduced the **Area Over the Perturbation Curve (AOPC)** to quantify heatmap quality for deep image classifiers by progressively occluding pixels in order of claimed importance and tracking the resulting decline in classification score. The same underlying logic was later formalized for structured/tabular and text features as the twin metrics **comprehensiveness** and **sufficiency** in the ERASER benchmark [6], which is the direct formal source for the definitions used in this pipeline.

## 4. Formal Definitions

### 4.1 Notation

Let:
- $x$ be an instance (an encoded feature vector),
- $f(x)_c$ be the model's predicted probability for class $c$ given input $x$,
- $c^* = \arg\max_c f(x)_c$ be the model's own predicted class for the *unperturbed* instance,
- $r = \{r_1, r_2, \ldots\}$ be the ranked list of features for $x$, ordered by descending explanation-assigned importance,
- $r_{1:k} \subset r$ be the top-$k$ ranked features,
- $x \setminus r_{1:k}$ denote $x$ with the top-$k$ features replaced by baseline (masked) values,
- $x \cap r_{1:k}$ denote $x$ with **only** the top-$k$ features retained and everything else replaced by baseline values.

### 4.2 Comprehensiveness

$$
\text{Comprehensiveness}_k(x) = f(x)_{c^*} - f(x \setminus r_{1:k})_{c^*}
$$

This measures how much the model's confidence in its own original prediction **drops** when the top-$k$ claimed-important features are removed. A **large positive value** indicates a faithful explanation: the features it flagged as important really were load-bearing for the prediction [6].

### 4.3 Sufficiency

$$
\text{Sufficiency}_k(x) = f(x)_{c^*} - f(x \cap r_{1:k})_{c^*}
$$

This measures how much confidence is **lost** when *only* the top-$k$ features are kept (everything else masked). A **value close to zero** indicates a faithful explanation: the top-$k$ features alone are enough to reproduce the original prediction, meaning the explanation captured the genuinely decisive evidence [6].

Note the directional asymmetry: for comprehensiveness, *higher is better* (bigger drop from removing important stuff); for sufficiency, *lower is better* (smaller drop from keeping only important stuff, i.e. closer to zero, and ideally not negative beyond noise).

### 4.4 AOPC (Area Over the Perturbation Curve)

Rather than committing to one arbitrary cutoff $k$, AOPC averages the effect over a set of $k$-values (expressed as fractions of the total feature count), producing a single robust score per instance per condition [5, 6]:

$$
\text{AOPC}_{\text{comp}}(x) = \frac{1}{|K|} \sum_{k \in K} \text{Comprehensiveness}_k(x)
\qquad
\text{AOPC}_{\text{suff}}(x) = \frac{1}{|K|} \sum_{k \in K} \text{Sufficiency}_k(x)
$$

This pipeline uses $K = \{10\%, 20\%, 30\%, 50\%\}$ of the (aggregated, original-feature-level — see §5.1) feature count, rounded to the nearest integer feature count with a floor of 1 and a ceiling of the total feature count.

### 4.5 Ranking Function $r$

The ranking $r$ is derived from each explainer's own attribution output:
- **SHAP**: exact Shapley-value attributions from TreeSHAP [2, 4], ranked by $|\phi_i|$.
- **LIME**: local surrogate-model coefficients from a sparse linear approximation around $x$ [3], ranked by $|w_i|$.

Both are established, independently-derived attribution methods; this pipeline does not alter how either explainer computes its raw attributions — it only governs how those attributions are aggregated, ranked, and used to mask instances (§5).

## 5. Implementation in This Pipeline

This section documents exactly how the formulas in §4 were operationalized against real, heterogeneous, partially one-hot-encoded tabular datasets across two domains — the part of the protocol that required project-specific decisions beyond what the source papers specify, since ERASER's comprehensiveness/sufficiency definitions were built for token-level NLP rationales, not encoded tabular feature groups.

### 5.1 Original-Feature-Level Aggregation

**Decision**: Masking and ranking both operate at the level of the *original* feature (e.g., `EmploymentType`), not the encoded model-input column (e.g., `EmploymentType_Full-time`, `EmploymentType_Part-time`, ...).

**Why**: A one-hot-encoded categorical is a single semantic feature split across multiple columns. Ranking or masking those columns independently would (a) let a categorical's importance be silently diluted or inflated depending on how many levels it happens to have, and (b) risk producing invalid encoding states if only some of a group's columns are masked (see §5.2).

**How**: Per-instance attributions (SHAP's `shap_values`, or LIME's parsed coefficients) are aggregated to the parent feature as the sum of absolute values across the group's encoded columns, via `aggregate_to_original_features()`. Group membership itself comes from `build_feature_group_map()`, which reads the *fitted* `OneHotEncoder`'s learned categories directly from each dataset's `fitted_transformers.joblib` — not inferred from column-name string patterns — so a dataset with an unusual encoding convention fails loudly rather than silently mis-grouping.

### 5.2 Baseline (Masking) Value Protocol

**Decision**: 
- Numerical features are masked with the **training-set median**, computed in already-scaled space.
- Categorical (one-hot) groups are masked by replacing the **entire group at once** with the training-set **mode category's** one-hot pattern (1.0 at the mode column, 0.0 elsewhere in the group) — never masking a single dummy column in isolation.
- Baselines are computed from `X_train_prebalance.csv`, **not** `X_train_balanced.csv`.

**Why median/mode rather than zero**: The choice of baseline value for perturbation-based attribution is itself an active area of study — a poorly chosen baseline (e.g., all-zero) can push a masked instance far outside the training distribution, producing a prediction change that reflects the model's behavior on out-of-distribution inputs rather than the genuine importance of the masked feature [9, 10]. Using the empirical median/mode keeps masked instances close to a "typical" point in the real data distribution.

**Why whole-group masking for categoricals**: Since these features are one-hot encoded with no dropped baseline category (confirmed via this project's `preprocessing.py`, which fits `OneHotEncoder(drop=None)`), masking a single dummy column while leaving its siblings untouched can produce an encoding state the model never saw during training — e.g., two simultaneously active levels, or all levels near-zero at once. This is not a state any real instance can occupy, so a resulting prediction change would not be interpretable as "the effect of removing this feature." Masking the whole group to a valid one-hot pattern avoids this.

**Why `X_train_prebalance.csv`, not `X_train_balanced.csv`**: This dataset's class balancing uses SMOTE [7]. SMOTE generates synthetic minority-class instances by interpolating between real neighbors in continuous feature space — including one-hot-encoded columns, which are not designed to be interpolated. Direct inspection of `X_train_balanced.csv` for the `loan_default` dataset confirmed this empirically: roughly 20,000–22,000 rows per one-hot column contained fractional values (e.g., 0.9205) instead of clean 0/1. A "mode" computed over such a column is not meaningful, and a "median" over it corresponds to no real applicant. `X_train_prebalance.csv` — the post-cleaning, pre-SMOTE training split — was confirmed clean in every dataset checked.

### 5.3 Target Class Definition

**Decision**: $c^*$ is defined per-instance as the model's own predicted class on the *unperturbed* instance ($\arg\max_c f(x)_c$), not a fixed positive-class index.

**Why**: Healthcare and finance datasets in this study have different and independently-varying class-imbalance directions and rates (e.g., diabetes prevalence vs. loan default rate). Fixing fidelity computation to a hardcoded positive class (e.g., class 1) would conflate "how faithful is this explanation" with "how does this dataset's class balance happen to be labeled," which is exactly the kind of dataset-specific confound this study's variance-decomposition design is meant to isolate away from genuine domain effects.

### 5.4 Explainer-Specific Handling

- **SHAP**: attributions are read directly from the saved `shap_values.npy` array (exact, deterministic — TreeSHAP has no sampling step [4]).
- **LIME**: attributions must first be reconstructed from `lime_explanations.csv`, where each row is a discretized condition string (e.g., `"glucose <= 117.00"`) rather than a clean feature name. `parse_lime_feature()` matches each condition string back to its underlying model-input column before aggregation proceeds identically to SHAP's path — ensuring both explainers are ranked and masked through the exact same downstream logic once their raw attributions are extracted.

### 5.5 Output Artifacts

Two artifact types are produced per `(domain, dataset, model, explainer)` combination:

1. **`Fidelity/<domain>/<dataset>/<model>/<explainer>/masked_predictions.csv`** — the full, un-aggregated audit trail: one row per (instance, $k$-fraction, condition), recording the original and masked predicted-class probability and their difference. This is retained specifically so AOPC can be recomputed under a different $k$-weighting later without re-scoring every masked instance from scratch.
2. **`Evaluation/metrics_long.csv`** rows, at two granularities:
   - Per-$(instance, k)$: `metric_name ∈ {comprehensiveness, sufficiency}`, `mask_fraction` populated.
   - Per-instance aggregate: `metric_name ∈ {aopc_comprehensiveness, aopc_sufficiency}`, `mask_fraction = NULL`.

   All Fidelity rows carry `baseline_type = "median_mode_grouped"`, documenting which version of the masking protocol produced them.

## 6. Worked Example

From an actual validation run (Pima Diabetes, Random Forest, SHAP, one instance, predicted class = 1, original $f(x)_{c^*} = 0.7600$):

| $k$ (fraction) | $k$ (features) | Condition | Masked $f(\cdot)_{c^*}$ | Value |
|---|---|---|---|---|
| 0.10 | 1 | Comprehensiveness | 0.4438 | **0.3162** |
| 0.10 | 1 | Sufficiency | 0.8377 | −0.0777 |
| 0.20 | 2 | Comprehensiveness | 0.6695 | 0.0905 |
| 0.50 | 4 | Comprehensiveness | 0.5590 | 0.2010 |
| 0.50 | 4 | Sufficiency | 0.8042 | −0.0442 |

Reading this: masking just the single top-ranked feature already drops predicted-class confidence by 0.316 (large — that one feature was doing a lot of work), while comprehensiveness continues climbing as more top features are removed (0.090 → 0.201 at $k$=2→4), consistent with a faithful ranking. Sufficiency staying near zero (slightly negative, within noise) across all $k$ indicates the top few features alone are sufficient to reproduce — and here, if anything, slightly overshoot — the original prediction confidence.

## 7. Interpreting the Results

### 7.1 Reading a single score

- **High AOPC-comprehensiveness** → the explainer is correctly identifying features the model is genuinely sensitive to. This is the primary faithfulness signal.
- **Low (near-zero) AOPC-sufficiency** → the explainer's top-ranked features are, on their own, enough to drive the model's decision — the explanation isn't omitting the truly decisive evidence.
- **An explainer that is faithful will show both**: high comprehensiveness *and* low sufficiency together. An explainer that shows only one is capturing something real but incompletely — e.g., high comprehensiveness with high sufficiency-drop-too (i.e., sufficiency far from zero) suggests the top-$k$ features matter but are not sufficient alone, implying important explanatory signal is distributed beyond what the explainer ranked highest.

### 7.2 Comparing across explainers, models, datasets, and domains

This is where the numbers connect to the project's central research question. For each fidelity metric (AOPC-comprehensiveness, AOPC-sufficiency, and their per-$k$ components), the downstream variance-decomposition analysis partitions observed variance into domain, dataset-within-domain, model, and explainer components. In this framing:

- A fidelity metric that shows **large, consistent SHAP-vs-LIME separation across every dataset in both domains** (as in the worked example, and matching the pooled averages from the validation run: SHAP AOPC-comprehensiveness 0.195 vs. LIME 0.179) — but **small domain-attributable variance** — supports classifying fidelity as **domain-invariant**: an explainer's relative faithfulness ranking, once established, likely transfers from a benchmark run in one domain to deployment in the other.
- A fidelity metric whose magnitude instead tracks with dataset properties (e.g., feature count, class balance) more than with domain identity supports classifying it as **dataset-dependent**, meaning practitioners should re-benchmark per dataset rather than trusting a domain-level generalization.

### 7.3 Domain-specific implications

- **Healthcare**: low comprehensiveness for a clinically important feature is a genuine safety-relevant finding — it suggests the explanation shown to a clinician may not reflect what the model actually weighted, which is the exact failure mode flagged as most dangerous in the clinical XAI literature (plausible-but-unfaithful explanations that mislead rather than inform).
- **Finance**: fidelity results here connect directly to regulatory defensibility — an explanation used to justify an adverse credit action (per ECOA-style requirements) should have both dimensions checked, not just be visually plausible, since a low-comprehensiveness explanation could not actually withstand an examiner's scrutiny of "does this reason really explain the decision."

## 8. Limitations and Known Caveats

- **AOPC is not directly comparable across models without normalization.** Raw AOPC values depend on model-specific achievable minimum/maximum bounds; recent work proposes Normalized AOPC (NAOPC) specifically to correct for this when comparing faithfulness *across different models* [8]. This pipeline reports raw AOPC per model/explainer combination and treats cross-model comparisons with corresponding caution in the variance-decomposition write-up, rather than assuming raw AOPC values are on a shared scale.
- **Perturbation can push instances off-distribution.** Even with a median/mode baseline (§5.2), a masked instance is still a synthetic point, not a real observation; the model's behavior on it is not guaranteed to reflect genuine, in-distribution reasoning, a general critique of perturbation-based faithfulness evaluation [9, 10].
- **LIME's underlying background set contains SMOTE-interpolated fractional values** (see §5.2), a property of the LIME explanations themselves (generated in an earlier pipeline stage), not of this masking protocol — it cannot be retroactively corrected here and is noted as a limitation for the project write-up.
- **Small feature-count datasets can produce identical $k$ at adjacent fractions.** For an 8-feature dataset (e.g., Pima Diabetes), $\text{round}(0.2 \times 8) = \text{round}(0.3 \times 8) = 2$, so the 20% and 30% AOPC components use an identical top-2 feature set. This is a real consequence of small dimensionality, not a computation error, and is more pronounced for low-dimensional healthcare datasets than for higher-dimensional finance datasets in this study — itself a potential dataset-size confound worth checking explicitly in the variance-decomposition stage.

## 9. References

[1] F. Doshi-Velez and B. Kim, "Towards a rigorous science of interpretable machine learning," *arXiv preprint arXiv:1702.08608*, 2017.

[2] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, Long Beach, CA, USA, 2017, pp. 4765–4774.

[3] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, San Francisco, CA, USA, 2016, pp. 1135–1144.

[4] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, 2020.

[5] W. Samek, A. Binder, G. Montavon, S. Lapuschkin, and K.-R. Müller, "Evaluating the visualization of what a deep neural network has learned," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 28, no. 11, pp. 2660–2673, 2017.

[6] J. DeYoung, S. Jain, N. F. Rajani, E. Lehman, C. Xiong, R. Socher, and B. C. Wallace, "ERASER: A benchmark to evaluate rationalized NLP models," in *Proc. 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, Online, 2020, pp. 4443–4458.

[7] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.

[8] J. Edin et al., "Normalized AOPC: Fixing misleading faithfulness metrics for feature attribution explainability," *arXiv preprint arXiv:2408.08137*, 2024.

[9] S. Hooker, D. Erhan, P.-J. Kindermans, and B. Kim, "A benchmark for interpretability methods in deep neural networks," in *Advances in Neural Information Processing Systems 32 (NeurIPS)*, Vancouver, Canada, 2019.

[10] P. Sturmfels, S. Lundberg, and S.-I. Lee, "Visualizing the impact of feature attribution baselines," *Distill*, 2020.

---

*Companion documents: `results/schema.md` (master table schema and cross-metric protocol notes), `src/fidelity.py` (implementation), `src/evaluation.py` (shared feature-group map, aggregation, and LIME-parsing utilities reused here).*
