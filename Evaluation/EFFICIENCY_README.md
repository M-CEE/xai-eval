# Efficiency: Methodology, Formulas, and Interpretation Guide

## 1. Purpose and Scope

This document specifies the theoretical foundation, formal definitions, and exact computational procedure behind the **Efficiency** metric family in this evaluation pipeline, as implemented in `src/evaluation.py` (`compute_efficiency`). It is written to research-report standard: every design decision is either derived directly from established practice in the literature or explicitly flagged as a project-specific adaptation, with the reasoning stated. It also covers how to interpret the resulting numbers and what they imply for the project's central research question — whether explanation-quality metrics behave consistently across the healthcare and finance domains.

Efficiency is one of five metric families evaluated in this study (alongside Fidelity, Stability, Robustness, and Simplicity), applied across 6 datasets (3 healthcare, 3 finance), 2 models (Random Forest, XGBoost), and 2 explainers (SHAP, LIME).

## 2. Conceptual Definition and Terminology

### 2.1 What "Efficiency" Means Here

In this study, **Efficiency** refers exclusively to the **computational cost of generating an explanation** — how much wall-clock time an explainer takes to produce a result for a given instance, model, and dataset. This is a deliberate narrowing of the term: "efficiency" is used elsewhere in machine learning to mean sample efficiency (how much training data a model needs), statistical efficiency (estimator variance relative to a theoretical bound), or Pareto efficiency (multi-objective trade-off optimality). None of those senses apply here. Where this document or the codebase says "efficiency," it always means computational/runtime cost of explanation generation.

Unlike Fidelity, Efficiency is **not a property that is meaningfully evaluated per masking condition** the way comprehensiveness or sufficiency are — there is no notion of "efficiency at $k{=}20\%$." It is, however, still computed **per instance** in the sense that the reported number is the mean cost of explaining *one* instance, derived from the total batch runtime recorded when the explainer was originally run (§5.1) — the underlying measurement is a single batch timing, not a genuinely independent per-instance stopwatch.

### 2.2 Why Efficiency Was Selected

- **It is a hard deployment constraint, not just an academic nicety.** Practitioner-facing surveys of organizations deploying explainable ML report that computational cost is one of the practical factors, alongside accuracy and legal compliance, that determines whether an explanation method is actually usable in production [5].
- **It directly trades off against the other metrics in practice.** LIME's cost is governed by `num_samples` (more perturbed samples → higher fidelity to the local decision boundary, but higher runtime); a practitioner who tunes this knob down for speed is implicitly trading Fidelity and Stability for Efficiency. Measuring Efficiency alongside the other four metric families makes that trade-off visible rather than assumed.
- **It is domain-relevant in an asymmetric way.** Finance use cases such as real-time fraud scoring impose latency budgets that a healthcare diagnostic-support tool explaining a single patient case generally does not — explicitly documented as a live constraint in finance-specific applied XAI work on credit scoring [6]. This makes Efficiency a plausible candidate for showing genuine domain-linked behavior, which is directly relevant to this study's central question.
- **It is independent of human judgment and cheap to measure**, consistent with the other four metric families in this study, and requires no additional artifact generation beyond what was already produced when the explanations were first computed (§5.1) — unlike Fidelity, which required new masked-prediction runs.
- **It complements the other four metric families** by measuring a resource cost rather than a quality property of the explanation's content. An explainer can be simultaneously fast, unfaithful, and unstable, or slow, faithful, and stable — Efficiency is orthogonal to what the other four metrics test, and no other metric in this study would surface a runtime problem.

## 3. Theoretical Foundation: Computational Cost as an Explanation Property

The two explainers evaluated here have fundamentally different, and well-documented, computational cost profiles:

- **LIME** fits a local linear surrogate model around each instance by sampling and evaluating the black-box model on perturbed neighbors [2]. Its cost is governed by `num_samples` — the number of perturbed points drawn and scored per explained instance — held fixed across all runs in this study's protocol (see companion `schema.md`). Runtime scales roughly linearly with `num_samples` and with the cost of a single black-box prediction.
- **SHAP**, in its exact form, computes Shapley values by evaluating the model over all coalitions of features — a computation that is exponential in the number of features [3]. This pipeline instead uses **TreeSHAP** [4], which exploits tree-ensemble structure (both Random Forest and XGBoost are tree ensembles) to compute exact Shapley values in polynomial time relative to the number of trees, tree depth, and features, without sampling. This is why SHAP is typically — but, as the worked example in §6 shows, not universally — faster than LIME in this study.

Because TreeSHAP's cost depends on the internal structure of the fitted trees (depth, number of estimators) rather than purely on feature count, its runtime is not a fixed function of the dataset alone the way LIME's approximately is — this is the mechanism behind the reversal documented in §6.

## 4. Formal Definitions

### 4.1 Notation

Let:
- $T$ be the total wall-clock runtime (seconds) recorded for explaining a batch of instances, as logged in the explainer's own `metadata.json` at generation time,
- $N$ be the number of instances explained in that batch,
- $F$ be the number of features used by the explainer for that run ($F$ = total encoded feature count for SHAP; $F$ = `num_features_per_instance` for LIME).

### 4.2 Runtime per Instance

$$
\text{Efficiency}_{\text{instance}} = \frac{T}{N} \quad \text{(reported in milliseconds)}
$$

This is the primary efficiency figure: average time cost to produce one instance's explanation.

### 4.3 Runtime per Instance, Normalized by Feature Count

$$
\text{Efficiency}_{\text{instance,feature}} = \frac{T}{N \cdot F}
$$

**Why this second, normalized version is reported alongside the raw figure**: without normalizing by feature count, a "finance is less efficient than healthcare" (or vice versa) finding could simply be an artifact of one domain's datasets having more columns — this study's finance datasets (e.g., `loan_default`, 16 original / 31 encoded features) are substantially higher-dimensional than its healthcare datasets (e.g., Pima Diabetes, 8 features). Dividing out feature count is what makes an efficiency comparison across datasets of different dimensionality meaningful, rather than confounding "domain" with "how many columns this particular dataset happens to have."

## 5. Implementation in This Pipeline

### 5.1 Data Source: No New Computation Required

Unlike Fidelity, Efficiency requires no new artifact generation. `runtime_seconds`, `n_instances` (or `n_instances_explained`), and the relevant feature count are already recorded in each explainer's `metadata.json`, written at the time the explanations were originally generated (`src/explanations.py`). `compute_efficiency()` reads these values directly and computes the two ratios in §4.2–4.3 — this is why Efficiency (together with Simplicity) was built and validated first in this project's build order, ahead of Fidelity, Robustness, and Stability.

### 5.2 Units

Runtime is reported in **milliseconds per instance**, not seconds, since sub-millisecond TreeSHAP timings on tree ensembles (see §6) would otherwise round to zero or require excessive decimal precision to read meaningfully.

## 6. Worked Example

Real measurements from this pipeline, across all four validated dataset × model combinations:

| Dataset | Model | Explainer | ms/instance | ms/instance/feature |
|---|---|---|---|---|
| Pima Diabetes | RF | SHAP | 1.30 | 0.16 |
| Pima Diabetes | RF | LIME | 84.35 | 10.54 |
| Pima Diabetes | XGB | SHAP | 0.71 | 0.09 |
| Pima Diabetes | XGB | LIME | 45.52 | 5.69 |
| loan_default | RF | SHAP | **3755.60** | **121.15** |
| loan_default | RF | LIME | 228.86 | 7.38 |
| loan_default | XGB | SHAP | 0.12 | 0.004 |
| loan_default | XGB | LIME | 48.38 | 1.56 |

Three of the four combinations show the expected pattern — SHAP (TreeSHAP) substantially faster than LIME, consistent with its polynomial-time exact computation versus LIME's sampling-based approach (§3). The fourth — **`loan_default` with Random Forest** — inverts this: SHAP takes roughly 16× longer than LIME for the same dataset. This is not a measurement error. As discussed in §3, TreeSHAP's cost depends on the fitted trees' internal structure, not on feature count alone; a Random Forest fit to a wider, higher-dimensional dataset such as `loan_default` can produce deeper or more numerous trees than the XGBoost model fit to the same data, making exact per-path Shapley computation meaningfully more expensive despite being algorithmically polynomial rather than exponential. **The practical implication is direct: "SHAP is faster than LIME" is not a safe blanket claim to make in this study's write-up — it is conditional on which model SHAP is wrapping, and that conditionality is itself a finding worth reporting, not an inconvenience to average away.**

## 7. Interpreting the Results

### 7.1 Reading a single score

A lower `ms/instance` value means the explainer can generate more explanations per unit time — directly relevant to any deployment context with a latency budget (real-time scoring, interactive dashboards) or a throughput requirement (batch-explaining an entire portfolio or patient cohort).

### 7.2 Comparing across explainers, models, datasets, and domains

Given the model-dependent reversal documented in §6, Efficiency is a strong candidate for showing **explainer × model interaction effects that are not cleanly attributable to domain** — the variance-decomposition analysis should be expected to attribute a meaningful share of Efficiency's variance to the model factor and the explainer × model interaction term, not to domain or dataset alone. A finding that Efficiency is *not* domain-invariant, but *is* explainer-model-invariant (or vice versa), is itself informative for the "unified evaluation framework" research question: it would indicate that runtime benchmarks from one model/explainer pairing do not transfer to another, regardless of which domain either was run in.

### 7.3 Domain-specific implications

- **Finance**: real-time fraud-detection or transaction-scoring systems have hard latency budgets; a combination like `loan_default` + Random Forest + SHAP (3.7 seconds per instance in this measurement) would likely be operationally infeasible for such a use case even though it may be the most faithful combination on other metrics — a genuine trade-off a practitioner would need to weigh, not just a number to report.
- **Healthcare**: diagnostic support tools explaining a single patient case at a time are typically less latency-sensitive, making the same SHAP-on-Random-Forest combination far more tolerable in that setting even at the same absolute runtime — the *acceptability* of a given efficiency figure is domain-dependent even where the *measured value itself* is not driven by domain.

## 8. Limitations and Known Caveats

- **Efficiency measurements are hardware- and environment-dependent.** Absolute timings reflect the specific machine, load, and library versions this pipeline ran on; they are valid for *relative* comparison within this study (same environment for every combination) but should not be quoted as universal claims about SHAP's or LIME's inherent speed.
- **A single batch timing, not a repeated-measures average.** Because `runtime_seconds` is read from the original explanation-generation run rather than re-timed multiple times specifically for this metric, per-instance efficiency does not carry its own variance estimate the way Stability's repeated-call design does — a single slow or fast batch run is taken at face value.
- **Efficiency says nothing about whether the time spent was worthwhile.** A method could be fast and unfaithful, or slow and highly faithful; efficiency must always be read alongside Fidelity, not as a standalone quality signal. Tuning LIME's `num_samples` down would improve its efficiency figures while likely degrading its fidelity and stability — this pipeline holds `num_samples` fixed across all runs specifically so efficiency comparisons are not confounded by different practitioners' speed/quality trade-off choices.

## 9. References

[1] F. Doshi-Velez and B. Kim, "Towards a rigorous science of interpretable machine learning," *arXiv preprint arXiv:1702.08608*, 2017.

[2] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, San Francisco, CA, USA, 2016, pp. 1135–1144.

[3] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, Long Beach, CA, USA, 2017, pp. 4765–4774.

[4] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, 2020.

[5] U. Bhatt, A. Xiang, S. Sharma, A. Weller, A. Taly, Y. Jia, J. Ghosh, R. Puri, J. M. F. Moura, and P. Eckersley, "Explainable machine learning in deployment," in *Proc. 2020 Conference on Fairness, Accountability, and Transparency (FAT\*)*, Barcelona, Spain, 2020, pp. 648–657.

[6] J. Dessain, N. Bentaleb, and F. Vinas, "Cost of explainability in AI: An example with credit scoring models," in *Explainable Artificial Intelligence (xAI 2023)*, Communications in Computer and Information Science, vol. 1901, Springer, Cham, 2023, pp. 498–516.

---

*Companion documents: `results/schema.md` (master table schema and cross-metric protocol notes), `SIMPLICITY_README.md` (the other metric family computed from existing artifacts with no new generation step), `src/evaluation.py` (implementation).*
