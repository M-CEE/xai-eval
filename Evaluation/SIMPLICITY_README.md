# Simplicity: Methodology, Formulas, and Interpretation Guide

## 1. Purpose and Scope

This document specifies the theoretical foundation, formal definitions, and exact computational procedure behind the **Simplicity** metric family in this evaluation pipeline, as implemented in `src/evaluation.py` (`compute_simplicity_shap`, `compute_simplicity_lime`, `_normalized_entropy`, `_pct_features_for_mass`). It is written to research-report standard: every design decision is either derived directly from an established method in the literature or explicitly flagged as a project-specific adaptation, with the reasoning stated. It also covers how to interpret the resulting numbers and what they imply for the project's central research question — whether explanation-quality metrics behave consistently across the healthcare and finance domains.

Simplicity is one of five metric families evaluated in this study (alongside Fidelity, Stability, Robustness, and Efficiency), applied across 6 datasets (3 healthcare, 3 finance), 2 models (Random Forest, XGBoost), and 2 explainers (SHAP, LIME).

## 2. Conceptual Definition and Terminology

### 2.1 What Simplicity Means Here

**Simplicity** (used interchangeably with **complexity** in the source literature, where low complexity = high simplicity) measures how concentrated an explanation's attribution is across features — whether the explanation tells a story built on a small, digestible number of features, or spreads its claimed importance thinly across most or all of them. It is a property of the explanation's *presentation*, independent of whether the explanation is correct: an explanation can be maximally simple and completely unfaithful, or highly faithful (per this study's Fidelity metric) while spreading credit across dozens of features in a way no human could realistically act on.

This distinction from Fidelity is deliberate and load-bearing for this study's design: Simplicity and Fidelity are measured separately, on the same underlying attributions, specifically so that "faithful but incomprehensible" and "comprehensible but unfaithful" explanations are both visible in the results rather than collapsed into a single score.

### 2.2 Why Simplicity Was Selected

- **Cognitive plausibility is a recognized explanation-quality dimension in its own right**, grounded in social-science research on how humans actually construct and accept explanations — human explanations are typically selective, citing a small number of causes rather than an exhaustive causal account [2]. A technically complete but 30-feature-wide explanation does not match this pattern and is unlikely to be usable by a clinician or credit analyst under real time pressure, regardless of its fidelity.
- **It is explicitly built into at least one of the two explainers under study.** LIME does not report every feature by default — it is designed around a fixed `num_features` cap that forces its local surrogate model toward sparsity as a design choice [3]. This makes Simplicity not just an external evaluation lens but a property some explainers are already implicitly optimizing for, which is directly testable by comparing LIME's simplicity against SHAP's, which carries no such built-in sparsity constraint.
- **It is one of a small number of recurring conceptual properties identified across the broader XAI evaluation literature** (referred to as "Compactness" in the most comprehensive taxonomy to date, which reviewed evaluation practice across 300+ XAI papers) [6], meaning this study's inclusion of Simplicity aligns with recognized practice rather than being an idiosyncratic addition.
- **It is independent of human judgment and computationally cheap**, requiring no new artifact generation beyond the attributions already saved when explanations were generated — like Efficiency, and unlike Fidelity, it was built and validated first in this project's build order.
- **It complements the other four metric families** by measuring the *shape* of an explanation's attribution distribution rather than its correctness (Fidelity), consistency (Stability, Robustness), or computational cost (Efficiency). None of the other four metrics would distinguish a faithful-but-diffuse explanation from a faithful-and-concentrated one.

### 2.3 Local Computation

Like Fidelity, both entropy and cumulative-mass-coverage (§4) are computed **per explained instance**, using that instance's own attribution vector. Dataset/model/explainer-level summaries reported later in this pipeline are means over these per-instance scores, not a property computed once against some aggregate or global attribution.

## 3. Theoretical Foundation

The formal basis for this study's Simplicity metric is the entropy-based complexity measure introduced by Bhatt, Weller, and Moura [5], proposed alongside sensitivity (this study's Stability/Robustness) and faithfulness (this study's Fidelity) as one of three core quantitative criteria for evaluating feature-based explanations. Their complexity definition treats an instance's normalized absolute attribution values as a probability distribution over features and computes its Shannon entropy: a distribution concentrated on few features has low entropy (simple, interpretable at a glance); a distribution spread evenly across all features has high entropy (complex, harder for a human to act on) [5]. This formulation has since been applied specifically to SHAP and LIME on tabular classification data in benchmark studies close to this project's own setting [7], which is the direct precedent this pipeline's tabular implementation follows.

## 4. Formal Definitions

### 4.1 Notation

Let:
- $x$ be an explained instance,
- $\phi_i(x)$ be the attribution assigned to original (aggregated) feature $i$ for instance $x$,
- $n$ be the total number of original (aggregated) features,
- $P_i(x) = \dfrac{|\phi_i(x)|}{\sum_{j=1}^{n} |\phi_j(x)|}$ be feature $i$'s share of the instance's total absolute attribution mass.

### 4.2 Normalized Entropy Complexity

$$
H(x) = -\sum_{i=1}^{n} P_i(x) \log P_i(x)
\qquad\qquad
\widetilde{H}(x) = \frac{H(x)}{\log n}
$$

$H(x)$ is the Shannon entropy of the attribution distribution [5]. Dividing by $\log n$ (its maximum possible value, achieved when attribution is spread perfectly evenly across all $n$ features) rescales the score to $[0, 1]$ regardless of how many features a dataset has.

**Why normalize by $\log n$**: this study's six datasets range from 8 features (Pima Diabetes) to 31 encoded / 16 original features (`loan_default`) and beyond. Raw (un-normalized) entropy is mechanically larger for higher-dimensional datasets simply because there are more features to spread attribution across — without normalization, a "finance explanations are more complex than healthcare explanations" finding could be entirely an artifact of finance datasets having more columns, exactly the same dimensionality confound discussed for Efficiency. $\widetilde{H}(x) \in [0, 1]$ is comparable across datasets of different dimensionality: $\widetilde{H}(x) \to 0$ means attribution is concentrated on very few features (simple); $\widetilde{H}(x) \to 1$ means attribution is spread near-evenly across all of them (complex).

### 4.3 Cumulative-Mass-Coverage Features

A second, more directly interpretable simplicity measure: how many top-ranked features are needed before their combined attribution mass reaches a threshold $\tau$ (this pipeline uses $\tau \in \{0.80, 0.90\}$).

$$
k_\tau(x) = \min \left\{ k : \sum_{i=1}^{k} P_{(i)}(x) \geq \tau \right\}
$$

where $P_{(1)}(x) \geq P_{(2)}(x) \geq \ldots \geq P_{(n)}(x)$ is the sorted (descending) sequence of per-feature attribution shares. Reported both as a raw count (`n_features_for_80pct_mass`) and, for the same dimensionality-comparability reason as §4.2, as a percentage of the dataset's total feature count (`pct_features_for_80pct_mass = k_\tau(x) / n`).

**Why report both entropy and cumulative-mass-coverage rather than just one**: entropy is a single continuous summary statistic, sensitive to the whole shape of the distribution, but is not directly interpretable in human terms ("an entropy of 0.71 means..."). Cumulative-mass-coverage answers a directly actionable question — "how many of this instance's features would a practitioner actually need to look at to see 80% of the story" — at the cost of depending on an arbitrarily chosen threshold $\tau$, the same limitation single-$k$ fidelity evaluation has (see the companion Fidelity README, §4.4). Reporting both gives a robust continuous measure and an interpretable discrete one, each compensating for the other's weakness.

## 5. Implementation in This Pipeline

### 5.1 Data Source: No New Computation Required

Like Efficiency, Simplicity requires no new artifact generation. For SHAP, entropy and cumulative-mass-coverage are computed directly from the already-saved `shap_values.npy` array. For LIME, per-instance attributions are first reconstructed from `lime_explanations.csv`'s discretized condition strings via the shared `parse_lime_feature()` utility (the same function Fidelity reuses — see the companion Fidelity README, §5.4), then aggregated identically to SHAP's path.

### 5.2 Original-Feature-Level Aggregation

Exactly as documented for Fidelity (companion README, §5.1), attributions are aggregated to the *original* feature level (e.g., `EmploymentType` as one unit, not its four constituent one-hot columns) via `aggregate_to_original_features()`, using absolute values, before entropy or cumulative-mass-coverage is computed. The same rationale applies here as for Fidelity: computing simplicity at the encoded-column level would let a categorical feature's importance appear artificially diffuse simply because it happens to have many levels, which is a property of the encoding scheme, not of how simple the explanation actually is to a human reading it in terms of the original features.

### 5.3 LIME's Built-In Sparsity Cap

LIME does not report every feature by default — its `num_features` parameter (held fixed at 50 across this study's protocol, per the companion `schema.md`) caps how many features appear in `lime_explanations.csv` per instance. Since every dataset in this study has fewer than 50 original features, this cap is not binding in practice (LIME reports all features for every instance evaluated so far), but where it *would* bind, unreported features are treated as exactly zero attribution — not omitted from the entropy/mass calculation — so that LIME's simplicity score reflects its actual sparsity behavior rather than being computed over an artificially truncated feature set that would make it look simpler than it structurally is.

## 6. Worked Example

Real measurements from this pipeline, across all four validated dataset × model combinations:

| Dataset | Model | Explainer | Normalized Entropy | % features for 80% mass | % features for 90% mass |
|---|---|---|---|---|---|
| Pima Diabetes | RF | SHAP | 0.7365 | 44.2% | 60.0% |
| Pima Diabetes | RF | LIME | 0.7851 | 50.1% | 64.4% |
| Pima Diabetes | XGB | SHAP | 0.7097 | 42.5% | 57.1% |
| Pima Diabetes | XGB | LIME | 0.7622 | 48.8% | 64.1% |
| loan_default | RF | SHAP | 0.8548 | 48.1% | 63.2% |
| loan_default | RF | LIME | 0.8594 | 46.3% | 60.0% |
| loan_default | XGB | SHAP | 0.9120 | 56.7% | 70.8% |
| loan_default | XGB | LIME | 0.9746 | 74.5% | 87.4% |

Two patterns are worth naming explicitly. First, **SHAP's normalized entropy is lower than LIME's in all four combinations** — SHAP's attribution is consistently more concentrated on fewer features than LIME's, holding across both domains and both models. Since this ranking direction is preserved everywhere it was tested, it is a candidate for a domain-invariant finding under this study's variance-decomposition framework (§7.2), though four combinations is not yet the full 24-combination matrix.

Second, the gap between SHAP and LIME **widens sharply for `loan_default` + XGBoost** (0.912 vs. 0.975 entropy; 56.7% vs. 74.5% of features needed for 80% mass) compared to the other three combinations, where the two explainers are much closer together. This is a genuine dataset × model × explainer interaction, not noise — it says LIME's relative simplicity disadvantage against SHAP is not constant, but depends on which dataset and model it is explaining, which is exactly the kind of pattern this study's bucket classification (domain-invariant vs. dataset-dependent vs. model-dependent) is designed to surface.

## 7. Interpreting the Results

### 7.1 Reading a single score

- **Low normalized entropy** (near 0) → the explanation concentrates on a small number of decisive features — easy for a human to act on, but worth cross-checking against Fidelity, since concentration alone does not guarantee those few features are the ones the model actually relied on.
- **High normalized entropy** (near 1) → attribution is spread thinly and near-evenly across most features — technically complete, but unlikely to be usable as-is by a time-pressed practitioner without further summarization.
- **`pct_features_for_80/90pct_mass`** gives the same story in directly actionable terms: a clinician or credit analyst reading an explanation with `pct_features_for_80pct_mass = 42%` knows that checking roughly two-fifths of the listed features accounts for most of the story; at 75%, nearly the entire feature list needs review, which defeats much of the practical value of having a ranked explanation at all.

### 7.2 Comparing across explainers, models, datasets, and domains

As in the Fidelity and Efficiency companion documents, the downstream variance-decomposition analysis partitions Simplicity's variance into domain, dataset-within-domain, model, and explainer components. The worked example in §6 already suggests two candidate findings to test formally against the full 24-combination matrix: (a) the SHAP-more-concentrated-than-LIME direction may be domain-invariant (explainer-attributable variance, consistent sign across all measured combinations so far), while (b) the *magnitude* of that gap may be dataset- or model-dependent (the `loan_default` + XGBoost widening), which would place Simplicity's magnitude in a different bucket than its direction — a nuance the four-bucket framework in this study's design is specifically built to capture rather than average away.

### 7.3 Domain-specific implications

- **Healthcare**: a clinician reviewing an explanation for a diagnosis needs to act on it quickly; an explanation requiring review of 75%+ of a patient's features to capture the bulk of the model's reasoning offers little practical advantage over inspecting the raw feature vector directly.
- **Finance**: adverse-action notices under ECOA-style requirements are expected to cite specific, limited reasons for a credit decision — a highly diffuse (high-entropy) explanation is harder to compress into the small number of principal reasons such notices are expected to state, independent of whether the explanation is faithful.

## 8. Limitations and Known Caveats

- **Simplicity is a presentation property, not a correctness property.** A low-entropy, highly concentrated explanation can still be unfaithful (low Fidelity); simplicity and fidelity must be read together, not as substitutes for each other, and this study reports both specifically so neither is mistaken for the other.
- **The choice of $\tau \in \{0.80, 0.90\}$ for cumulative-mass-coverage is a convention, not a derived optimum**, chosen for interpretability and consistency with this study's Fidelity mask-fraction protocol rather than from a formal argument that these are the "correct" thresholds.
- **LIME's sparsity cap has not yet been observed to bind** in this study (§5.3) — all datasets evaluated so far have fewer features than the 50-feature cap. If a future dataset in the full matrix exceeds this, LIME's simplicity scores for that dataset would reflect a genuinely different regime (explicit truncation) than the other datasets', and that difference should be checked for and flagged rather than silently pooled into the same analysis.
- **Entropy and cumulative-mass-coverage can occasionally disagree in edge cases** (e.g., a distribution with one dominant feature and many small, near-uniform remaining ones can have moderate entropy but a low mass-coverage count, or vice versa depending on the exact shape) — this is expected given they summarize the same distribution differently, not a computation error, and is a reason both are reported rather than collapsed into one number.

## 9. References

[1] F. Doshi-Velez and B. Kim, "Towards a rigorous science of interpretable machine learning," *arXiv preprint arXiv:1702.08608*, 2017.

[2] T. Miller, "Explanation in artificial intelligence: Insights from the social sciences," *Artificial Intelligence*, vol. 267, pp. 1–38, 2019.

[3] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, San Francisco, CA, USA, 2016, pp. 1135–1144.

[4] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, Long Beach, CA, USA, 2017, pp. 4765–4774.

[5] U. Bhatt, A. Weller, and J. M. F. Moura, "Evaluating and aggregating feature-based model explanations," in *Proc. 29th International Joint Conference on Artificial Intelligence (IJCAI-20)*, 2020, pp. 3016–3022.

[6] M. Nauta, J. Trienes, S. Pathak, E. Nguyen, M. Peters, Y. Schmitt, J. Schlötterer, M. van Keulen, and C. Seifert, "From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable AI," *ACM Computing Surveys*, vol. 55, no. 13s, Article 295, 2023.

[7] T. Pereira, J. Vitorino, E. Maia, and I. Praça, "Evaluating local explainability metrics for machine learning models on tabular data," *arXiv preprint arXiv:2605.27618*, 2026.

---

*Companion documents: `results/schema.md` (master table schema and cross-metric protocol notes), `EFFICIENCY_README.md` (the other metric family computed from existing artifacts with no new generation step), `FIDELITY_README.md` (the metric family this document deliberately distinguishes itself from in §2.1), `src/evaluation.py` (implementation).*
