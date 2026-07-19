# Robustness: Methodology, Formulas, and Interpretation Guide

## 1. Purpose and Scope

This document specifies the theoretical foundation, formal definitions, and exact computational procedure behind the **Robustness** metric family in this evaluation pipeline, as implemented in `src/robustness.py`. It is written to research-report standard: every design decision is either derived directly from an established method in the literature or explicitly flagged as a project-specific adaptation, with the reasoning stated. It also covers how to interpret the resulting numbers, what they imply for the project's central research question, and how to reason formally about a specific pattern already observed at a glance across the full 24-combination matrix (see §7.4).

Robustness is one of five metric families evaluated in this study (alongside Fidelity, Stability, Simplicity, and Efficiency), applied across 6 datasets (3 healthcare, 3 finance), 2 models (Random Forest, XGBoost), and 2 explainers (SHAP, LIME).

## 2. Conceptual Definition and Terminology

### 2.1 Robustness vs. Stability

This study deliberately separates two properties that are frequently conflated under a single "consistency" label elsewhere in the literature:

- **Stability** asks whether an explainer agrees with **itself**: given the exact same, unperturbed instance and model, do repeated calls to the explainer produce the same explanation? This is purely a question of the explainer's own internal (non-)determinism (relevant chiefly for LIME, whose local surrogate fitting involves random sampling; TreeSHAP is deterministic by construction and is handled as a labeled floor value, not tested here — see the companion Stability README).
- **Robustness**, the subject of this document, asks whether an explanation stays consistent when the **input** changes slightly — a different, but realistically similar, instance. This tests sensitivity to genuine real-world input variation (measurement noise, slightly different patients or applicants), not explainer self-agreement.

This distinction follows the framing in the foundational work on interpretability-method robustness, which argues that a key desideratum for any explanation method is that similar inputs should produce similar explanations — a property distinct from, and not guaranteed by, an explainer being internally deterministic [2].

### 2.2 Why Robustness Was Selected

- **It targets a documented, empirically demonstrated failure mode.** Explanations for standard interpretability methods have been shown to change dramatically under adversarially or even innocuously perturbed inputs that do not change the model's own prediction — a finding replicated across both gradient-based and perturbation-based explanation methods [3]. This is not a hypothetical concern being tested defensively; it is a known weakness the literature has repeatedly documented.
- **It is a recognized, actively benchmarked evaluation axis**, implemented as one of three core reliability categories (alongside faithfulness and fairness) in the most comprehensive open-source XAI benchmarking framework to date, which groups Robustness-style perturbation-sensitivity metrics under "stability" in its own terminology while explicitly building on the same Alvarez-Melis and Jaakkola robustness formulation this pipeline follows [4].
- **It is independent of, and answers a different question than, Fidelity.** An explanation can correctly identify the features a model relied on (high Fidelity) while still being highly sensitive to negligible input noise (low Robustness) — an explainer that reorders its entire ranking because a patient's glucose reading was off by half a standard deviation is not trustworthy in practice, regardless of how faithful any single one of its explanations is in isolation.
- **It is domain-relevant for a concrete, practical reason.** Real-world tabular inputs in both healthcare and finance are never perfectly clean — a lab value or a reported income figure carries measurement or reporting noise even when nothing about the underlying case has meaningfully changed. An explanation that flips its story under that ordinary level of noise would mislead a clinician or credit analyst into thinking something changed about the case when nothing did.
- **It complements the other four metric families** by testing sensitivity to the *input*, distinct from Stability's test of the explainer's *self*-agreement, Fidelity's test of *correctness*, Simplicity's test of *presentation*, and Efficiency's test of *cost*. No other metric in this study would catch an explanation that is faithful, stable, simple, and cheap, yet flips its top feature the moment a value is nudged slightly.

### 2.3 Local Computation

Like Fidelity and Simplicity, Robustness is computed **per explained instance**: each instance is perturbed once, re-explained, and compared against its own original explanation. Dataset/model/explainer-level summaries reported later in this pipeline are means over these per-instance scores, not a property computed against some global notion of model sensitivity.

## 3. Theoretical Foundation

The formal basis for perturbation-based explanation robustness is the local Lipschitz continuity framing introduced by Alvarez-Melis and Jaakkola [2]: an explanation function is locally robust around an instance $x_0$ if nearby points produce explanations that do not change by more than a bounded amount relative to how much the input itself changed. Their work also introduced the empirical observation motivating this entire metric family — that popular interpretability methods frequently fail this property, producing substantially different explanations for inputs a human would consider essentially the same case [2]. This finding was independently corroborated and extended shortly after, showing that even small, systematically-constructed input perturbations that do not change a model's prediction can produce large changes in feature-importance rankings [3].

This pipeline's specific metrics — Spearman rank correlation and top-$k$ Jaccard overlap between original and perturbed attributions — are a simplified, more directly interpretable operationalization of the same underlying property that the Relative Input Stability (RIS) metric in the OpenXAI benchmarking framework formalizes more strictly as a worst-case ratio over a neighborhood of perturbations [4]. This pipeline uses a single, realistically-scaled perturbation per instance rather than a worst-case search over many perturbations (see §8 for the resulting limitation), trading some formal strictness for computational tractability across a 24-combination experimental matrix.

## 4. Formal Definitions

### 4.1 Notation

Let:
- $x$ be an original explained instance (an encoded feature vector),
- $x'$ be a perturbed version of $x$ (§5.1),
- $e(x), e(x') \in \mathbb{R}^n$ be the original-feature-level aggregated attribution vectors for $x$ and $x'$ respectively (aggregation as in the companion Fidelity/Simplicity READMEs, §5.1 in each),
- $n$ be the number of original (aggregated) features.

### 4.2 Perturbation

$$
x'_i =
\begin{cases}
x_i + \mathcal{N}(0, (\epsilon \cdot \sigma_i)^2) & \text{if feature } i \text{ is numerical} \\
x_i & \text{if feature } i \text{ belongs to a categorical (one-hot) group}
\end{cases}
$$

where $\sigma_i$ is feature $i$'s own training-set standard deviation and $\epsilon$ is a small constant (this pipeline uses $\epsilon = 0.05$, i.e. noise scaled to roughly 5% of one standard deviation).

**Why categorical features are never perturbed**: a "small" perturbation of a continuous value is well-defined — nudge it slightly and it remains a realistic nearby value. There is no equivalent small perturbation of a one-hot categorical: the only two states for a given level are "active" or "not active," so any change is a full category swap — a qualitatively larger and different kind of intervention than what this metric is designed to test. Perturbing categoricals here would conflate Robustness with a counterfactual-sensitivity test, which is out of scope.

### 4.3 Spearman Rank Correlation

$$
\rho(x, x') = \text{Spearman}\big(e(x), e(x')\big)
$$

Computed over the **full** $n$-length attribution vector, not just the top-$k$ — sensitive to reordering anywhere in the ranking, not only among the most important features. $\rho \to 1$ indicates a robust explanation (ranking essentially unchanged); $\rho \to 0$ or negative indicates the perturbation substantially reshuffled or inverted the explanation.

### 4.4 Top-$k$ Jaccard Overlap

$$
J_k(x, x') = \frac{|\text{top}_k(e(x)) \cap \text{top}_k(e(x'))|}{|\text{top}_k(e(x)) \cup \text{top}_k(e(x'))|}
\qquad
k = \min\left(5, \left\lceil \frac{n}{2} \right\rceil \right)
$$

A more human-interpretable complement to Spearman correlation: "did the headline features stay the same," rather than "did the whole ranking stay in the same order." $J_k \to 1$ means the same top-$k$ features are flagged as important before and after perturbation; $J_k \to 0$ means the perturbation swapped out the entire set of headline features.

## 5. Implementation in This Pipeline

### 5.1 Shared Infrastructure Reuse

Robustness reuses the model-loading (`load_model`), feature-group mapping (`build_feature_group_map`), and original-feature aggregation (`aggregate_to_original_features`, `parse_lime_feature`) machinery already built and validated for Fidelity — see the companion Fidelity README, §5.1–§5.4, for the full reasoning behind each. This is a deliberate consistency choice: Robustness and Fidelity should disagree because they measure genuinely different properties, not because one uses a subtly different feature-grouping or aggregation convention than the other.

### 5.2 A New Kind of Artifact: Live Re-Explanation

Unlike Efficiency, Simplicity, and Fidelity's ranking step — all of which read from already-saved explanation artifacts — Robustness requires **generating new explanations** for instances that have never been explained before (the perturbed variants). This makes it the most computationally expensive metric family in this study (see the runtime note in §8).

For LIME specifically, the perturbed-instance explanation is generated using a `LimeTabularExplainer` reconstructed with the **exact same configuration** the original explanation run used — same background data source, `num_samples`, `num_features`, and random state, all read directly from that run's own `lime/metadata.json` rather than hardcoded. This matters because any difference between the original and perturbed explanation should be attributable to the input perturbation itself, not to an accidental configuration drift between the original run and this pipeline's re-explanation of the perturbed variant.

### 5.3 Output Artifacts

Two artifact types are produced per `(domain, dataset, model, explainer)` combination, mirroring Fidelity's split between a fine-grained audit trail and an aggregated summary:

1. **`Evaluation/Robustness/<domain>/<dataset>/<model>/<explainer>/perturbation_attributions.csv`** — one row per `(instance_id, original_feature)`, recording the original and perturbed aggregated attribution values side by side. This is what allows Spearman correlation or top-$k$ Jaccard overlap to be recomputed later under a different $k$, or a different comparison statistic entirely, without re-perturbing and re-explaining every instance from scratch — and, as shown in §7, it is also what enables the feature-level ("which feature is most volatile") analysis that goes beyond what the aggregated summary alone can show.
2. **`Evaluation/metrics_long.csv`** rows: `metric_property = "Robustness"`, `metric_name ∈ {spearman_rank_correlation, top_k_jaccard_overlap}` (the latter's exact column name embeds the $k$ actually used for that dataset's feature count — see §7.1 for a practical note on this).

## 6. Worked Example

Real measurements from this pipeline, Pima Diabetes, both models, both explainers:

| Model | Explainer | Mean Spearman $\rho$ | Mean Top-$k$ Jaccard | Most Volatile Feature |
|---|---|---|---|---|
| RF | SHAP | 0.884 | 0.809 | Glucose |
| RF | LIME | 0.902 | 0.855 | Glucose |
| XGB | SHAP | 0.542 | 0.574 | Glucose |
| XGB | LIME | 0.641 | 0.600 | Glucose |

Two things stand out. First, **`Glucose` is the most volatile feature in all four combinations** — the feature whose attribution shifts the most, on average, under a small perturbation, for both models and both explainers. This is not surprising on reflection: `Glucose` is Pima's strongest predictor, so a model that leans heavily on it will naturally show more attribution movement there when it is nudged, even slightly — a mechanical consequence of a feature's importance, not a flaw specific to any one explainer.

Second, and more consequential for this study's design: **Random Forest explanations are substantially more robust than XGBoost explanations on the exact same dataset**, for both explainers (Spearman $\rho$ of 0.88–0.90 for RF vs. 0.54–0.64 for XGB). This is a large, real effect — not sampling noise — and it means that for Robustness specifically, **model choice may be a larger driver of the observed score than domain or dataset are**, a finding with direct implications for how this metric's variance-decomposition results should be read (§7.3).

## 7. Interpreting the Results

### 7.1 Reading a single score

High Spearman correlation and high top-$k$ Jaccard overlap together indicate a robust explanation: the ranking, and specifically the headline features, survive a realistic amount of input noise. A low score on either indicates the opposite; the two can occasionally diverge (e.g., moderate correlation with unstable Jaccard, if the reordering happens mostly outside the top-$k$), which is expected given they summarize the comparison differently, similar to the entropy/cumulative-mass-coverage divergence discussed in the companion Simplicity README. Note that the Jaccard column name embeds the actual $k$ used (`top_4_jaccard_overlap`, `top_5_jaccard_overlap`, ...), which varies by dataset feature count (§4.4) — normalize this column name before pooling across datasets of different dimensionality, or treat $k$ itself as a variable worth reporting alongside the score (see the notebook snippet used to generate this document's worked example).

### 7.2 Comparing across explainers, models, datasets, and domains

As with the other metric families, the downstream variance-decomposition analysis partitions Robustness's variance into domain, dataset-within-domain, model, and explainer components. The worked example in §6 already provides real, verified evidence that **model** may be an unusually large factor for this specific metric — worth checking explicitly, and reporting even if it complicates a clean domain-invariant/dataset-dependent verdict, rather than being averaged away.

### 7.3 A Nuance for the Four-Bucket Framework: When Model Swamps Domain

This study's four buckets (behaves similarly across domains; differs significantly between domains; highly dataset-dependent; largely unaffected by domain) are all framed around **domain** as the primary axis of interest, with dataset, model, and explainer as competing or controlling factors. The RF-vs-XGB gap in §6 raises a scenario the original four-bucket framing does not cleanly cover: **a metric whose variance is dominated by model choice, orthogonally to domain entirely**. If this pattern holds across the full 24-combination matrix (not yet confirmed at the time of writing — see §7.4), Robustness could show:

- Low domain-attributable variance (arguing for bucket 1/4, domain-invariant), while simultaneously
- High model-attributable variance (a genuine, practically important finding that neither "domain-invariant" nor "dataset-dependent" fully captures on its own).

**Recommended handling**: when writing up Robustness's bucket classification, report the model effect size explicitly alongside the domain classification, rather than letting a "domain-invariant" verdict imply the metric is simple or model-agnostic. A defensible framing is: *"Robustness is domain-invariant conditional on model choice — its magnitude is primarily determined by which model is being explained, not which domain the data came from."* That is a more precise and more useful finding for a practitioner than either bucket label alone.

### 7.4 The Cross-Dataset "Same Volatile Feature" Observation

Across the full 24-combination matrix, an at-a-glance pattern was noted: **XGBoost selected the same most-volatile feature for `credit_card_fraud_2023`, `financial_distress`, `breast_cancer_wisconsin`, and `pima_diabetes`.** This is a striking observation worth treating carefully rather than either dismissing or over-claiming from it directly, since it spans two domains and four otherwise-unrelated feature spaces.

Three non-exclusive hypotheses worth checking formally against the full result set before drawing a conclusion:

1. **A genuine model-architecture effect.** If XGBoost's specific tree-building behavior (e.g., a tendency to split early and repeatedly on whichever single feature has the strongest early gain) systematically concentrates sensitivity onto one dominant feature per dataset, and if that dominant feature happens to coincide with each of those four datasets' single most predictive feature, the pattern would be a real, reportable finding about *how XGBoost's structure interacts with perturbation-sensitivity*, generalizable as a caution for practitioners using XGBoost + post-hoc explanations together, independent of domain.
2. **A coincidence of dataset-level dominant-feature structure.** If each of those four datasets independently happens to have one feature that dominates predictive signal far more than the others (plausible for `Glucose` in Pima, and worth checking for the finance datasets' analogues), then "same volatile feature" may really mean "each dataset's own single most important feature, which differs in name but plays the same structural role" — a dataset-level effect wearing a cross-dataset-looking coincidence, similar to the "one outlier dataset dragging a domain average" caution already noted for domain-effect testing (see the earlier resolved-decisions log).
3. **An artifact of this study's fixed perturbation protocol.** Since every numerical feature is perturbed with noise scaled to its own standard deviation (§4.2, an equal-relative-scale design), a feature with a strong, sharply-thresholded relationship to the model's output (common for whichever feature XGBoost happens to split on most aggressively) will show outsized attribution movement under even a proportionally small perturbation — meaning the observation may partly reflect this pipeline's specific perturbation design rather than a property that would hold under a different, equally reasonable perturbation scheme.

**Recommended next step**: extend the per-feature volatility comparison (the `discover_robustness_combos` + `groupby("feature")["abs_diff"].mean()` pattern already used to produce the numbers in §6) across all 24 combinations, and cross-reference each dataset's "most volatile feature" against that same feature's raw Fidelity/importance ranking for the same dataset/model. If the volatile feature is consistently also the top-ranked-by-attribution feature, hypothesis 2 (or 3) is favored; if it is a feature that is volatile without necessarily being the top-ranked one, hypothesis 1 gains support. This is exactly the kind of cross-metric-family check the variance-decomposition stage is designed to enable, and is a genuinely interesting thread for the discussion section regardless of which hypothesis the fuller analysis favors.

### 7.5 Domain-specific implications

- **Healthcare**: a diagnosis explanation that reorders its top features because a lab value carries ordinary measurement noise could cause a clinician to draw a different clinical conclusion from two readings of what is functionally the same patient state — a direct patient-safety concern, not merely an inconvenience.
- **Finance**: adverse-action explanations are expected to be defensible and reproducible under regulatory scrutiny; an explanation that changes its stated reasons for a credit denial when a reported figure is off by a small, ordinary reporting margin would not hold up to an examiner re-running the same case with slightly different input data.

## 8. Limitations and Known Caveats

- **A single perturbation per instance, not a worst-case search.** This pipeline's Spearman/Jaccard scores reflect one realistically-scaled perturbation draw, not the worst-case local Lipschitz constant that the source formulation defines [2] or that OpenXAI's RIS metric approximates via a search over many perturbations [4]. This trades formal strictness for computational tractability across 24 combinations — a single perturbation is a lower bound on how bad an explanation's worst-case sensitivity could be, not an estimate of it.
- **Categorical features are never perturbed** (§4.2), meaning the reported Robustness score characterizes sensitivity to continuous-feature noise only. A dataset with mostly categorical features (which none in this study's matrix are, but worth flagging for any future extension) would have a large share of its input space untested by this metric.
- **$\epsilon = 0.05$ is a convention, not a derived optimum.** A different perturbation magnitude could shift the reported scores' absolute level (though the RF-vs-XGB and cross-dataset comparisons in §6–§7.4 should be robust to the specific $\epsilon$ chosen, since it is held fixed across every combination).
- **Runtime cost is substantially higher than the other four metric families.** Because Robustness requires genuinely new SHAP and LIME explanations rather than reading saved artifacts, it is the most computationally expensive metric in this study, particularly for LIME with a non-trivial `num_samples` setting — this is itself worth cross-referencing against this study's own Efficiency results when interpreting how feasible this kind of check would be in a real-time production setting.
- **The §7.4 cross-dataset "same volatile feature" pattern is, at the time of writing, an observation, not yet a confirmed finding** — it should be formally verified against the full result set (as outlined in §7.4) before being reported as a conclusion in the paper, rather than being taken at face value from a glance across the summary table.

## 9. References

[1] F. Doshi-Velez and B. Kim, "Towards a rigorous science of interpretable machine learning," *arXiv preprint arXiv:1702.08608*, 2017.

[2] D. Alvarez-Melis and T. S. Jaakkola, "On the robustness of interpretability methods," presented at the *ICML Workshop on Human Interpretability in Machine Learning (WHI)*, Stockholm, Sweden, 2018.

[3] A. Ghorbani, A. Abid, and J. Zou, "Interpretation of neural networks is fragile," in *Proc. AAAI Conference on Artificial Intelligence*, vol. 33, 2019, pp. 3681–3688.

[4] C. Agarwal, S. Krishna, E. Saxena, M. Pawelczyk, N. Johnson, I. Puri, M. Zitnik, and H. Lakkaraju, "OpenXAI: Towards a transparent evaluation of model explanations," *Advances in Neural Information Processing Systems*, vol. 35, pp. 15784–15799, 2022.

[5] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, San Francisco, CA, USA, 2016, pp. 1135–1144.

[6] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, Long Beach, CA, USA, 2017, pp. 4765–4774.

[7] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, 2020.

[8] U. Bhatt, A. Weller, and J. M. F. Moura, "Evaluating and aggregating feature-based model explanations," in *Proc. 29th International Joint Conference on Artificial Intelligence (IJCAI-20)*, 2020, pp. 3016–3022.

[9] N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic minority over-sampling technique," *Journal of Artificial Intelligence Research*, vol. 16, pp. 321–357, 2002.

[10] T. Pereira, J. Vitorino, E. Maia, and I. Praça, "Evaluating local explainability metrics for machine learning models on tabular data," *arXiv preprint arXiv:2605.27618*, 2026.

---

*Companion documents: `results/schema.md` (master table schema and cross-metric protocol notes), `FIDELITY_README.md` (the shared model-loading, feature-group map, and aggregation infrastructure this document reuses), `EFFICIENCY_README.md` and `SIMPLICITY_README.md` (the two metric families computed from existing artifacts with no new generation step), `src/robustness.py` (implementation).*
