# Stability: Methodology, Formulas, and Interpretation Guide

## 1. Purpose and Scope

This document specifies the theoretical foundation, formal definitions, and exact computational procedure behind the **Stability** metric family in this evaluation pipeline, as implemented in `src/stability.py` (`run_stability`, `compute_pairwise_stability`, `default_top_k`). It is written to research-report standard: every design decision is either derived directly from established practice in the literature or explicitly flagged as a project-specific adaptation, with the reasoning stated. It also covers how to interpret the resulting numbers and what they imply for the project's central research question — whether explanation-quality metrics behave consistently across the healthcare and finance domains.

Stability is one of five metric families evaluated in this study (alongside Fidelity, Robustness, Simplicity, and Efficiency), applied across 6 datasets (3 healthcare, 3 finance), 2 models (Random Forest, XGBoost), and 2 explainers (SHAP, LIME).

## 2. Conceptual Definition and Terminology

### 2.1 What Stability Means Here

**Stability** measures whether an explainer agrees with *itself* on repeated calls against the same, unperturbed instance. This is deliberately distinct from **Robustness**, this pipeline's other consistency-oriented metric family: Robustness asks whether an explanation stays similar when the *input* is perturbed to a different, but realistically nearby, instance; Stability asks whether an explanation stays similar when *nothing about the input changes at all*. Robustness is therefore a question about input-sensitivity; Stability is a question about the explainer's own internal (non-)determinism.

This distinction has a direct consequence for the two explainers under study. TreeSHAP, the SHAP variant used throughout this pipeline (see `EFFICIENCY_README.md` §3), computes exact Shapley values via a closed-form traversal of the fitted trees — repeated calls against the same instance and model produce bit-identical output, so under this definition SHAP's Stability is not an empirical question but a structural guarantee (§4.2). LIME, by contrast, builds its local surrogate by sampling perturbed neighbors and fitting a weighted linear model to them [2]; that sampling step draws on a pseudo-random generator that is *not* held fixed across repeated calls in this protocol (§4.3), so LIME's Stability is a genuine empirical quantity this pipeline measures rather than assumes.

### 2.2 Why Stability Was Selected

- **Non-determinism is a documented, practically consequential property of LIME**, not a theoretical curiosity. Independent replications have repeatedly found that running LIME multiple times on the identical input and model can yield materially different top features and coefficients, undermining a practitioner's ability to trust any single explanation at face value [5, 6, 7].
- **It is orthogonal to every other metric in this study.** An explanation can be highly faithful (Fidelity), concentrated on few features (Simplicity), and fast to compute (Efficiency), while still returning a different story every time it is generated — a property none of the other four metric families would detect, since they each evaluate a single explanation instance, not the distribution of explanations an explainer produces for the same input.
- **It has an established, purpose-built measurement methodology in the XAI literature** — the VSI (Variables Stability Index) and CSI (Coefficients Stability Index) framework developed specifically to quantify LIME's repeat-to-repeat agreement [5] — rather than requiring this study to improvise a bespoke measure, so Stability's protocol here can be grounded directly in prior work rather than an ad hoc definition.
- **It is a precondition for the other metrics' comparability being meaningful.** If an explainer's output is highly unstable, a single Fidelity or Simplicity score computed from one arbitrary call is not representative of "the explanation" in any stable sense — Stability tells us how much weight a single-call score from the other four families can actually bear.
- **It surfaces a structural SHAP-vs-LIME asymmetry that is directly relevant to this study's research question.** Because TreeSHAP's determinism is a property of the algorithm rather than of the domain or dataset, Stability is a strong a priori candidate for showing that explainer identity, not domain, drives most of its variance — a concrete, testable instance of the "not attributable to domain" bucket in this study's four-bucket framework (see `schema.md` §1).

### 2.3 Local Computation

Like Fidelity, Simplicity, and Robustness, Stability is computed **per explained instance**: each instance gets its own repeat-agreement score (LIME) or floor value (SHAP), and any dataset/model/explainer-level summary reported downstream is a mean over these per-instance scores, not a property computed once against some pooled or global attribution.

## 3. Theoretical Foundation

The formal basis for this study's Stability metric is the VSI/CSI framework introduced by Visani, Bagli, Chesani, Poluzzi, and Capuzzo specifically to quantify LIME's self-consistency across repeated calls on the same instance [5]. Their approach repeats LIME $N$ times under identical settings against one unperturbed input and compares the resulting linear surrogate models pairwise: VSI checks whether the *same variables* are selected across repeats, and CSI checks whether the *coefficients* assigned to each variable are statistically indistinguishable across repeats. The two are kept as separate indices deliberately, since an explainer can pass one criterion while failing the other. This pipeline's dual Spearman-rank-correlation / top-k-Jaccard-overlap measurement (§4.4) mirrors that same two-part logic: Jaccard tests feature-selection agreement (a VSI-style question — do repeats surface the same top features?), while Spearman tests whether the full attribution ordering agrees (closer to a CSI-style question about relative magnitude/ranking agreement, adapted to a rank-based rather than coefficient-based comparison since it reuses this pipeline's existing Robustness infrastructure — see §5.1).

The underlying cause this framework measures — LIME's sampling-based construction yielding different local surrogates across repeated calls — is independently corroborated by subsequent work proposing stabilization techniques that presuppose the same instability: S-LIME uses a hypothesis-testing procedure to determine how many perturbed samples are needed before an explanation stabilizes [6], and a fully deterministic LIME variant has been proposed specifically to eliminate this repeat-to-repeat variance by removing the randomness from the sampling step entirely [7]. Both are cited here as corroborating evidence that the phenomenon this pipeline measures is real and well-documented, not as methods this pipeline adopts — this study deliberately measures LIME's *default*, non-stabilized behavior, since the research question concerns the properties of these explainers as commonly used out of the box.

## 4. Formal Definitions

### 4.1 Notation

Let:
- $x$ be the explained instance, held fixed and unperturbed across all repeats,
- $r \in \{1, \dots, N\}$ index a repeat, with $N = 10$ repeats per instance for LIME (§4.3),
- $\phi_i^{(r)}(x)$ be the aggregated absolute attribution assigned to original (aggregated) feature $i$ for instance $x$ on repeat $r$,
- $n$ be the total number of original (aggregated) features.

### 4.2 SHAP: Deterministic Floor Row

TreeSHAP computes exact Shapley values via closed-form traversal of the fitted trees rather than sampling, so $\phi_i^{(r)}(x) = \phi_i^{(r')}(x)$ for all $r, r'$ and all $i$ [3, 4] — repeated calls are guaranteed identical. Rather than spend $N$ redundant calls empirically confirming a structural guarantee, this pipeline assigns SHAP a single row per instance with both statistics fixed at their perfect-agreement value:

$$
\text{Stability}_{\text{corr}}(x) = 1.0, \qquad \text{Stability}_{\text{jaccard}}(x) = 1.0, \qquad \texttt{deterministic} = \texttt{True}
$$

**This is a labeling convention grounded in the published TreeSHAP algorithm, not an empirically re-verified claim per instance in this pipeline** — see §8 for why that distinction matters. SHAP rows are excluded from the inferential domain-consistency test for this reason (§7.2): a metric that cannot vary by construction contributes nothing to a variance decomposition beyond confirming the floor.

### 4.3 LIME: Repeated-Call Protocol

LIME is called $N = 10$ times per instance, every call against the exact same unperturbed input vector $x$, using **one shared `LimeTabularExplainer` instance** per (dataset, model) pair, reconstructed with the original run's exact configuration (background data source, `num_samples`, `num_features`, `random_state` — reusing `build_lime_explainer` directly from `src/robustness.py`, so Stability's LIME configuration is guaranteed identical to Robustness's and Fidelity's, per `schema.md` §6's note that `num_samples` is held fixed specifically so cross-metric comparisons aren't confounded by different speed/quality trade-offs).

**The explainer is deliberately not reset or reseeded between repeats.** LIME's internal sampling `RandomState` is left to advance naturally from call to call, because that call-to-call sampling variance is precisely the quantity this metric is designed to measure — reseeding to identical state before every repeat would trivially force a perfect score and defeat the purpose of measuring stability at all.

### 4.4 Pairwise Comparison Statistics

For any two repeats $r, s$, their aggregated attribution vectors are compared via the same two statistics Robustness uses (`compare_attributions` in `src/robustness.py`), reused directly rather than reimplemented:

**Spearman rank correlation** [8] over the full attribution vector:

$$
\rho(r, s) = \text{Spearman}\left(\left[\phi_i^{(r)}(x)\right]_{i=1}^n, \left[\phi_i^{(s)}(x)\right]_{i=1}^n\right)
$$

undefined (excluded, not zero-filled) when either vector is constant across all $n$ features, since Spearman correlation has no defined value for zero-variance input.

**Top-$k$ Jaccard overlap** [9] between the two repeats' most-attributed feature sets:

$$
J_k(r, s) = \frac{|\text{Top}_k^{(r)}(x) \cap \text{Top}_k^{(s)}(x)|}{|\text{Top}_k^{(r)}(x) \cup \text{Top}_k^{(s)}(x)|}, \qquad k = \min\left(5, \left\lceil \frac{n}{2} \right\rceil\right)
$$

using the identical $k$ convention as Robustness, computed once via `default_top_k(n)` and threaded explicitly into every comparison call in this pipeline's implementation (see `stability.py` module docstring, point 3) so the SHAP floor row's reported $k$ and the LIME repeats' computed $k$ cannot silently diverge.

### 4.5 Pairwise-Mean Reduction

With $N = 10$ repeats there are $\binom{10}{2} = 45$ pairs, not a single pair as in Robustness. The per-instance Stability score for each statistic is the mean across all pairs:

$$
\text{Stability}_{\text{corr}}(x) = \frac{1}{|\mathcal{P}|} \sum_{(r,s) \in \mathcal{P}} \rho(r, s), \qquad \text{Stability}_{\text{jaccard}}(x) = \frac{1}{\binom{N}{2}} \sum_{(r,s)} J_k(r, s)
$$

where $\mathcal{P}$ is the set of pairs with a defined (non-NaN) correlation. This pairwise-mean reduction is the same convention VSI/CSI use — averaging agreement across every pair of repeats rather than arbitrarily privileging one repeat as a fixed reference to compare all others against [5].

## 5. Implementation in This Pipeline

### 5.1 Reused Infrastructure

Stability introduces no new comparison logic: `build_feature_group_map`, `aggregate_to_original_features`, and `parse_lime_feature` are reused unchanged from `src/evaluation.py`, and `build_lime_explainer`, `get_lime_attribution`, and `compare_attributions` are reused unchanged from `src/robustness.py`. This is a deliberate design choice, not a shortcut: it guarantees that feature-grouping granularity, LIME configuration, and the Spearman/Jaccard formulas themselves are identical across Fidelity, Robustness, and Stability, so differences observed between these metric families reflect genuine differences in what they measure rather than incidental implementation drift.

### 5.2 Auditable Artifact

`Evaluation/Stability/<domain>/<dataset>/<model>/<explainer>/repeated_attributions.csv` — one row per `(instance_id, repeat_idx, original_feature)`, recording that repeat's aggregated attribution for that feature. This is the fine-grained analogue of Robustness's `perturbation_attributions.csv`: it allows the pairwise Spearman/Jaccard reduction to be recomputed later under a different $k$, a different pair-reduction convention, or a different $N$ (by subsetting `repeat_idx`), without re-running LIME from scratch. For SHAP, this file has exactly one `repeat_idx=0` row per `(instance, feature)`, reflecting the single floor call rather than a genuine repeat series.

### 5.3 Degenerate-Instance Handling

If every one of the 45 pairs for a given LIME instance yields an undefined (NaN) Spearman correlation — which would require every repeat to have returned a constant attribution vector — that instance is not silently dropped from the metric. It is written to `metrics_long.csv` with `status = "degenerate_all_nan_pairs"` and `value = None`, and the count of such instances is logged per (dataset, model, explainer) combination, so this cannot silently shrink the effective sample size for a downstream analysis without a visible trace.

### 5.4 Range and Units

Both statistics are unitless and bounded in $[-1, 1]$ (Spearman) and $[0, 1]$ (Jaccard) respectively; higher values indicate greater self-agreement (more stable). SHAP's floor value of $1.0$ on both represents the ceiling either statistic can reach.

## 6. Worked Example

Real measurements from this pipeline, currently available for two finance datasets across both models (healthcare datasets not yet run as of this writing — see caveat below):

| Dataset | Model | Explainer | Mean Pairwise Spearman | Mean Pairwise Top-$k$ Jaccard | $k$ |
|---|---|---|---|---|---|
| credit_card_fraud_2023 | RF | SHAP | 1.000 | 1.000 | 5 |
| credit_card_fraud_2023 | RF | LIME | 0.584 | 0.819 | 5 |
| credit_card_fraud_2023 | XGB | SHAP | 1.000 | 1.000 | 5 |
| credit_card_fraud_2023 | XGB | LIME | 0.515 | 0.634 | 5 |
| financial_distress | RF | SHAP | 1.000 | 1.000 | 5 |
| financial_distress | RF | LIME | 0.500 | 0.561 | 5 |
| financial_distress | XGB | SHAP | 1.000 | 1.000 | 5 |
| financial_distress | XGB | LIME | 0.376 | 0.530 | 5 |

SHAP sits at exactly $1.0$ on both statistics in all four combinations, as guaranteed by §4.2. LIME is well short of that ceiling everywhere measured so far — Spearman correlation ranges $0.376$–$0.584$ and top-5 Jaccard ranges $0.530$–$0.819$ — meaning repeated LIME calls on the *identical, unperturbed* instance are only moderately self-consistent in this pipeline's finance datasets: a materially different top-feature ranking can appear from one call to the next with no change to the input at all.

A pattern worth naming explicitly: **within both datasets, XGBoost's LIME correlation is lower than Random Forest's** ($0.515$ vs.\ $0.584$ on `credit_card_fraud_2023`; $0.376$ vs.\ $0.500$ on `financial_distress`). This is consistent in direction across both finance datasets measured so far, which makes it a candidate for a model-attributable effect — but two datasets from a single domain cannot establish whether this is a genuine model effect, a dataset-specific effect, or a domain-specific one; that requires the healthcare combinations and the full 24-combination matrix.

**Caveat on this table's current scope**: both datasets shown here happen to have $\geq 9$ features, so both land on the capped $k=5$ (§4.4) — this worked example does not yet include a dataset on the other side of that cap (e.g., 8-feature Pima Diabetes, where $k = \lceil 8/2 \rceil = 4$), so it cannot yet illustrate the cross-$k$ comparability issue flagged in §8. This table will be extended once healthcare-domain results are available, both to complete the domain comparison this section is meant to support and to show a $k=4$ combination alongside these $k=5$ ones.

## 7. Interpreting the Results

### 7.1 Reading a Single Score

A LIME Stability score near $1.0$ on both statistics means repeated calls against the identical input return essentially the same explanation — a practitioner can trust that a single generated explanation is representative rather than one arbitrary draw from a noisy process. A score meaningfully below $1.0$ means the *opposite*: two analysts running LIME on the same customer or patient record, at different times, could be shown different "reasons" for the same decision — a serious usability and trust concern independent of whether either explanation is individually faithful (Fidelity) or simple (Simplicity).

### 7.2 Comparing Across Explainers, Models, Datasets, and Domains

Per the protocol in `schema.md` §5, SHAP's rows are excluded from the inferential domain-consistency test — since SHAP cannot vary by construction (§4.2), including it would trivially inflate any explainer-level "SHAP is more stable than LIME" finding without that finding reflecting anything empirically discovered in this study. The domain-consistency question for Stability is therefore a **LIME-only** variance-decomposition: does LIME's repeat-to-repeat agreement differ systematically between healthcare and finance datasets, or is it primarily driven by dataset-specific factors (e.g., feature count, class balance) or by which model it is wrapping?

A priori, LIME's internal sampling variance is a property of its own algorithm and configuration (`num_samples`, kernel width, background data) rather than of domain semantics — making Stability, like Efficiency (`EFFICIENCY_README.md` §7.2), a plausible candidate for the "largely unaffected by domain" bucket, though this must be verified against the full 24-combination matrix rather than assumed from the mechanism alone.

### 7.3 Domain-Specific Implications

- **Healthcare**: a clinician who reruns an explanation tool on the same patient case and sees a materially different top-feature list has a direct reason to distrust the tool's output, independent of its measured Fidelity — clinical decision support has a lower tolerance for this kind of visible inconsistency than a purely internal analytics workflow.
- **Finance**: adverse-action notices under ECOA-style requirements are expected to cite specific, consistent reasons for a credit decision (see `SIMPLICITY_README.md` §7.3); an explainer that would produce a *different* set of cited reasons if rerun on the identical application undermines the defensibility of whichever notice was actually sent, a compliance-relevant consequence beyond general usability.

## 8. Limitations and Known Caveats

- **SHAP's determinism is a structural assumption grounded in the published algorithm, not re-verified empirically per instance in this pipeline.** No repeated SHAP calls are actually made — the floor row encodes what TreeSHAP is guaranteed to do [3, 4], not a measurement of what it was observed to do in this run. If this assumption were ever violated (e.g., a future SHAP backend introducing non-deterministic parallelism), it would not be caught by this pipeline as currently built.
- **Dimensionality affects Spearman's *precision*, not its range.** Spearman correlation is bounded in $[-1, 1]$ regardless of $n$, so unlike Efficiency's raw runtime or Simplicity's raw entropy, it does not need the same explicit renormalization to be comparable across datasets of different feature counts. However, with fewer features there are fewer achievable rank permutations, so correlation estimates from low-dimensional datasets (e.g., 8-feature Pima Diabetes) are inherently more discretized/noisier than from high-dimensional ones (e.g., 31-encoded-feature `loan_default`) — a genuine dimensionality effect on estimator variance that should be kept in mind when comparing magnitudes across datasets, even though no correction has been applied.
- **Top-$k$ Jaccard's normalization question has been deliberately deferred, not resolved.** As `schema.md` §3 already notes for Robustness (which Stability's implementation shares unchanged), $k = \min(5, \lceil n/2 \rceil)$ is not a constant fraction of $n$ across this study's six datasets — it approaches half the features for low-dimensional datasets but is capped at a flat 5 (a shrinking fraction) for high-dimensional ones. This is the same class of cross-dataset comparability confound Efficiency and Simplicity each address with an explicit normalized companion metric (`EFFICIENCY_README.md` §4.3, `SIMPLICITY_README.md` §4.2–4.3); Stability's raw top-$k$ Jaccard has no such companion yet. This has been intentionally left for the variance-decomposition stage (`schema.md` §8, item 3) rather than fixed at measurement time, so that Robustness and Stability — which share the same underlying `compare_attributions` call — are corrected together rather than diverging from each other.
- **$N = 10$ is a protocol choice, not a derived optimum**, chosen to yield a reasonably large pairwise sample ($\binom{10}{2}=45$) without excessive LIME runtime overhead (LIME's cost is the dominant factor per `EFFICIENCY_README.md` §6). Changing $N$ changes the estimator variance of each instance's Stability score; if $N$ is changed for only some combinations of the full matrix, per-instance scores would not be directly comparable across combinations that used different $N$.
- **Degenerate instances (§5.3) are excluded from the correlation mean, not treated as failures.** A single degenerate repeat pair does not null out an otherwise-informative instance-level score, but an instance where *every* pair is degenerate is flagged rather than silently contributing a missing value to downstream aggregation — analysts should check the `degenerate_all_nan_pairs` count before treating a combination's LIME correlation sample size as the full instance count.
- **Stability, like Simplicity and Efficiency, is a property independent of correctness.** A perfectly stable LIME explanation can still be unfaithful (low Fidelity) or diffuse (high Simplicity entropy); Stability answers "does this explainer agree with itself," not "is what it agrees on actually right."

## 9. References

[1] F. Doshi-Velez and B. Kim, "Towards a rigorous science of interpretable machine learning," *arXiv preprint arXiv:1702.08608*, 2017.

[2] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD)*, San Francisco, CA, USA, 2016, pp. 1135–1144.

[3] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, Long Beach, CA, USA, 2017, pp. 4765–4774.

[4] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, 2020.

[5] G. Visani, E. Bagli, F. Chesani, A. Poluzzi, and D. Capuzzo, "Statistical stability indices for LIME: Obtaining reliable explanations for machine learning models," *Journal of the Operational Research Society*, vol. 73, no. 1, pp. 91–101, 2022.

[6] Z. Zhou, G. Hooker, and F. Wang, "S-LIME: Stabilized-LIME for model explanation," in *Proc. 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining (KDD '21)*, Virtual Event, Singapore, 2021, pp. 2429–2438.

[7] M. R. Zafar and N. Khan, "Deterministic local interpretable model-agnostic explanations for stable explainability," *Machine Learning and Knowledge Extraction*, vol. 3, no. 3, pp. 525–541, 2021.

[8] C. Spearman, "The proof and measurement of association between two things," *The American Journal of Psychology*, vol. 15, no. 1, pp. 72–101, 1904.

[9] P. Jaccard, "The distribution of the flora in the alpine zone," *New Phytologist*, vol. 11, no. 2, pp. 37–50, 1912.

---

*Companion documents: `results/schema.md` (master table schema and cross-metric protocol notes), `ROBUSTNESS_README.md` (the metric family whose comparison infrastructure Stability reuses directly, and whose §7.1 top-$k$ normalization discussion applies identically here — see §8), `EFFICIENCY_README.md` and `SIMPLICITY_README.md` (the two metric families whose explicit raw/normalized dual-metric pattern §8 discusses adapting for Stability's Jaccard statistic), `src/stability.py` (implementation).*
