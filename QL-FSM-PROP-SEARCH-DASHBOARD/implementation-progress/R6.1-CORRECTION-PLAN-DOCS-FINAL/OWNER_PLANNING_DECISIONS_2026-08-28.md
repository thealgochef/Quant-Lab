# Owner Planning Decisions for R6.1 — verbatim answers, 2026-08-28

**Document type:** Input record. Four scope/science questions were put to the owner while the
R6.1 plan was being written; the answers below are binding on `R6.1_IMPLEMENTATION_PLAN.md`
(§3 summarizes them; §6 workstreams I, J, A and G apply them). These are planning decisions about
what R6.1 builds — **they are not owner ratification of any scientific default** (decisions 25/28/29/30
remain `proposed_protocol_default` until an `OwnerDecisionEvidenceRef` artifact exists).

---

## Q1 — Should the R5B MBP-1 sequence-gap correction be part of R6.1?

**Answer:** Include it in R6.1 as a mandatory parallel R5B.1 correction workstream.

Keep the implementation ownership under the MBP-1 source/coverage and feature-block lane rather
than mixing the correction into the KMeans regime formulas, but R6.1 and hardening must not be
considered complete until this blocker is closed.

Do not replace the current rule with another unproven shortcut such as "ts_recv discontinuity =
gap." Neither raw venue-sequence jumps nor timestamp spacing alone is sufficient proof that MBP-1
records are missing.

The correction must:

1. Withdraw the rule: positive raw venue sequence jump > 1 = source gap.
2. Introduce a new versioned MBP-1 source-coverage policy based on verified evidence such as:
   Databento-provided data-quality metadata or flags; explicit partition/source gap manifests;
   verified capture discontinuity evidence; another documented dataset-specific completeness contract.
3. Treat raw sequence-number jumps and timestamp discontinuities as diagnostics only unless their
   semantics are independently proven for the exact dataset/schema/request.
4. Re-mint every affected identity: MBP-1 source/coverage artifact; resolved
   IFVG_ORDER_FLOW_MBP1_V1 block identity; registry hash; B2/B3 bundle identities; feature
   artifact; coverage and validity reports; downstream controlled-study identity where applicable.
5. Add synthetic tests proving: normal sequence skips do not become source gaps; an explicitly
   declared real source gap does become typed missing evidence; no window is silently widened or
   imputed; affected candidates remain present with the correct typed reason; identical corrected
   evidence reproduces identical artifacts.
6. Add a bounded real-data diagnostic under the one canonical authorized <=5-day fixture once
   VerificationAuthorizationRef exists. The diagnostic must characterize actual partition-quality
   evidence and must not infer completeness merely from raw sequence continuity.
7. Keep all real MBP-1 research, baseline-vs-MBP-1 conclusions, and R5B acceptance blocked until
   this corrected policy passes.

Record this as an R5B.1-owned correction incorporated into the R6.1 closure gate, not as a
track-only item deferred past hardening.

## Q2 — Add a bundle-parametrized CatBoost rung in R6.1?

**Answer:** Include a new bundle-parametrized CatBoost rung in R6.1.

Implement it as a separate protocol: `ifvg_context_catboost_bundle_v1`. Do not modify the frozen
M0–M3 CatBoost lane or its identities.

Requirements:

1. Run the bundle-aware CatBoost protocol on both arms of every controlled comparison where
   applicable: baseline versus baseline + MBP-1; baseline versus baseline + regime outputs; pooled
   baseline versus regime-feature variants.
2. All compared arms must use the exact same: candidate/OOS row IDs; labels; fold assignments;
   embargo and purge rules; costs; random seed; CatBoost hyperparameters.
3. Reuse the established fixed CatBoost parameters and missingness semantics, but create a
   distinct resolved protocol identity that pins: resolved feature-bundle ID; ordered feature names;
   categorical-feature registry; preprocessing/missingness policy; CatBoost parameters; package and
   Python versions; fold-set ID; seed.
4. Treat a hard regime ID as categorical. Treat cluster distances, probabilities, assignment
   margin, and entropy as numeric where available.
5. Preserve native numeric NaN behavior and the registered categorical missing token. No implicit
   imputation or silent column dropping.
6. No hyperparameter search, feature selection, threshold selection, automatic winner selection,
   or model promotion.
7. Report the complete ladder on identical rows: training-prevalence reference; L2 logistic
   regression; bundle-aware CatBoost.
8. Add paired OOS metric deltas, calibration diagnostics, fold coverage, and explicit invalid-fold reasons.
9. The CatBoost rung remains research-only. It cannot create an execution gate, change S11, or
   alter Strategy-Core.
10. Add deterministic synthetic tests proving: identical OOS populations across rungs; fold-local
    fitting; bundle features reach CatBoost; hard regime IDs are categorical; a changed bundle or
    feature order changes model identity; the fixed M0–M3 CatBoost lane remains byte- and
    identity-unchanged.

This is required before the first full authorized development run so the MBP-1 and regime
studies are not limited to logistic-only conclusions.

## Q3 — Accept the proposed seven-feature context-panel block as the R6.1 default?

**Answer:** Accept the proposed seven-feature context-panel block as the R6.1 V1 default, with
the following binding details.

Block key: `IFVG_CONTEXT_BAR_PANEL_V1`. Status: AVAILABLE as an engineering capability;
`proposed_protocol_default` as a scientific configuration; `research_only_offline`; owner
ratification still required before FEATURE_ELIGIBLE use.

The seven registered features are: 1. cbp_realized_range_12 · 2. cbp_realized_volatility_12 ·
3. cbp_range_compression_ratio_12 · 4. cbp_path_efficiency_12 · 5. cbp_session_state ·
6. cbp_intensity_zscore_12 · 7. cbp_bar_position_in_session.

Requirements:

1. Define every formula precisely in a frozen formula contract before implementation: exact
   numerator and denominator; return/range convention; zero-denominator handling; minimum
   observations; clipping or winsorization, if any; missingness reason; session reset behavior.
2. Use completed bars only. One panel row becomes available only at that bar's completed
   close/as-of instant. No partial-bar or future-session evidence is permitted.
3. Build separate resolved panel protocols and artifact identities for 5-minute panels and
   15-minute panels. Do not pool the two intervals under one fit identity. `panel_interval_seconds`
   enters protocol, artifact, fold, and fit identity.
4. The 12-bar lookback is interval-relative: 12 × 5m and 12 × 15m are separate research horizons;
   neither is silently treated as equivalent to the other.
5. Use session-aware warmup. A panel row is unavailable until its full required completed-bar
   history exists within the registered policy.
6. `cbp_session_state` is a registered categorical value.
7. `cbp_bar_position_in_session` is a precisely defined normalized completed-bar position.
   Session boundaries, maintenance, early closes, and DST use the committed session/calendar policy.
8. `cbp_intensity_zscore_12` must name and pin its actual source: trade count, volume, or another
   registered completed-bar measurement. Do not use a generic "intensity" label while silently
   switching source fields. If volume is used, name and document it as volume intensity; if trade
   count is used, name and document it as trade-count intensity.
9. Inputs may contain only point-in-time market/context evidence. Prohibit: labels; trade
   outcomes; MFE/MAE; resolution fields; future session statistics; post-entry evidence;
   prop-account outcomes; MBP-1 features in this core panel block.
10. Persist: source artifact ID and manifest; formula version; ordered feature schema; as-of
    policy; interval; lookback; warmup policy; coverage and missingness reports.
11. Build panel-native folds and exact point-in-time panel-to-candidate assignments.
12. Add synthetic tests for: 5m and 15m identities; session and DST boundaries; early close;
    exact completed-bar availability; warmup; missing observations; order independence;
    panel-to-candidate no-lookahead assignment.
13. Do not add more panel features in R6.1. Any later feature addition creates a new formula and
    resolved block identity.
14. Descriptive and stratification studies may use this proposed default after capability gates
    pass. Model-bearing use remains subject to the promotion-status decision below.

*(Plan application: the intensity feature is registered as `cbp_volume_intensity_zscore_12` with
`intensity_source_field="volume"` per requirement 8.)*

## Q4 — Which promotion status gates the model-bearing regime classes?

**Answer:** Use STRATIFICATION_READY for cohort_descriptive, stratified_prop, and stratified_frontier.

Require FEATURE_ELIGIBLE for feature_only and cohort_model because both make the regime
assignment part of model construction.

For cohort_model, additionally require: minimum training rows per regime; both target classes in
every valid regime-specific training fold; a pooled-model baseline; invalid folds retained with
explicit reasons; identical outer OOS rows wherever comparisons are valid.

MODEL_FEATURE remains a later promotion status for adopting the regime into a standard accepted
model configuration. It is not required merely to run a controlled development-only feature or
cohort-model experiment.

No status authorizes execution gating. S11 remains blocked.


---

## Revision-3 implementation clarifications (architecture only; not new scientific ratification)

The original four answers above remain verbatim. The final plan applies these architecture-safe
clarifications:

1. Model-bearing regime runs use a two-pass workflow and freeze exact promotion, owner-decision, and
   assessment IDs into semantic identity; descriptive STRATIFICATION_READY may be derived
   deterministically from ratification-free gates.
2. Decision 25 (KMeans algorithm and exact pinned parameter snapshot) joins decisions 28/29/30 in
   any FEATURE_ELIGIBLE owner artifact.
3. `F_MAYBE_BAD_BOOK` is treated at verified channel scope or, when channel mapping is unavailable,
   conservatively at publisher/physical-partition scope; physical partition bounds define coverage.
4. Any incomplete 1m source bar in the 13-bar panel window invalidates the complete panel row.
5. Model-facing regime labels/distances are fit-local; canonical aligned labels are reporting-only.
6. Prop event detail is compressed, partitioned, budgeted Parquet with exact event time/order and a
   summary artifact; it is not unbounded JSON.
7. Release evidence includes a format-patch or Git bundle for independent source review.
