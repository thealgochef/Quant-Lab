# Final Plan Corrections for R5B.1 / R6.1 — 2026-08-28

**Document type:** Authoritative correction record for revision 3.  
**Status:** Planning-only; no code, tests, artifacts, or data runs were performed.

The following corrections are applied directly in `R6.1_IMPLEMENTATION_PLAN.md` and are binding for implementation:

1. **Semantic promotion authority:** model-bearing `RegimeStudyRequest`s freeze exact promotion,
   owner-decision, and capability-assessment IDs. Descriptive runs may derive STRATIFICATION_READY
   only from their own deterministic assessment. No execution path resolves a mutable “latest” status.
2. **MBP-1 scope:** physical partition bounds define the denominator; channel warnings are never
   narrowed to an instrument without a verified mapping; uncertainty starts at a trusted boundary;
   dataset-condition states remain distinct; positive completeness comes only from a verified compiler.
3. **Panel validity:** all 13 source bars must be complete; otherwise all seven features are null with
   `source_bar_incomplete` and exact offending-bar evidence.
4. **Owner decision 25:** KMeans selection and exact pinned parameter snapshot/hash are authorized
   alongside decisions 28/29/30 before FEATURE_ELIGIBLE use.
5. **Prop-event evidence:** versioned, compressed, partitioned, budgeted Parquet with event timestamp,
   trading day, clock policy, total-order fields, exact identities, and a per-path/per-regime summary.
6. **Model cluster semantics:** model-facing hard IDs/distances are fold-fit-local; canonical alignment
   is reporting-only unless a future causal mapping is separately contracted.
7. **Identity projection:** MBP source manifests use `source_document_sha256`; artifact content hashes
   remain envelope/manifest facts and never self-enter semantic payloads.
8. **Independent source review:** each implementation evidence bundle contains a format-patch or Git bundle.

No major R6.1 scope item, owner-selected feature set, CatBoost protocol, status gate, release order,
or V1 boundary is changed.
