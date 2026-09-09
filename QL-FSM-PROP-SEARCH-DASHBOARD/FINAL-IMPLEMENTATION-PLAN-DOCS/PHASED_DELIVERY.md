# Phased Delivery — Releases, File Lists, and Gates

**Document type:** Supporting document to `IMPLEMENTATION_PLAN.md` (brief §18 item 18; brief §13)
**Status:** Plan-only. **REVISED** per the first revision (P1-7/8/9), Amendment V2 §7 (R1/R2/R3 additions, R5 renamed, mandatory R5B, R6 amendments), **the V3 compact patch (P0-7 canonical allowlist, P0-8 authoring-vs-acceptance, P1-5 capability-gated operator readiness, P1-6 post-V1 regime expansion), and the V5 frontend-authority restoration patch**. See `REVISION_CHANGELOG.md` and `FINAL_CONSISTENCY_AUDIT.md`.
**Standing constraints for every release:** no full-data runs; synthetic fixtures + the one frozen ≤5-day allowlist; all protected/sealed counters zero; M0–M3 lane byte-identical; Strategy-Core not modified in v1; `pytest -q` + `ruff` green + docs-in-same-change rule per release.

The former monolithic M1–M8 sequence is restructured into **seven usable increments (R1–R6, with mandatory R5B as a distinct activation increment)** plus hardening and the operator run — the user receives a working, verified slice early instead of waiting for the whole research operating system.

---

## Release 0 — Plan approval (this document set, revised) ✅ deliverable

No code. Two build-entry verification items carry forward into R1: (a) capture-driver mid-chain start from a cached profile-matching seed (fallback: a `start_after_artifact` driver parameter, QL-only); (b) native-ID replay determinism (gates population deltas; lineage-validity test gates cross-profile deltas).

---

## Release 1 — Contracts, identities, access, stores + the real baseline vertical slice

The identity/architecture corrections land first, and the **first real five-day baseline replay runs in this release**, not at the end (P1-7).

**New files**
```
src/alpha_lab/agents/data_infra/ifvg/search/__init__.py
src/alpha_lab/agents/data_infra/ifvg/search/identities.py        (CoreStrategyReplayIdentity + ReplayInputBundle
                                                                  [content-addressed source partitions + day
                                                                  artifacts + QL replay source identity, P0-A],
                                                                  SearchChildMembership, companion + costed
                                                                  identities, canonical naming, Payload/Envelope
                                                                  projections + identity-projection audit [P0-C],
                                                                  GeneratedProfileCapability [P0-D])
src/alpha_lab/agents/data_infra/ifvg/search/axis_registry.py     (typed RegisteredAxisValue/CompositeAxisValue)
src/alpha_lab/agents/data_infra/ifvg/search/authorization.py     (OwnerDecisionEvidenceRef, scoped
                                                                  AuthorizationRequirementSet +
                                                                  derive_authorization_requirements,
                                                                  VerificationAuthorizationRef [P0-E])
src/alpha_lab/agents/data_infra/ifvg/search/charter.py
src/alpha_lab/agents/data_infra/ifvg/search/failure.py
src/alpha_lab/agents/data_infra/ifvg/search/store.py
src/alpha_lab/agents/data_infra/ifvg/search/catalog.py           (lock-guarded append-only event log + rebuildable index)
src/alpha_lab/agents/data_infra/ifvg/search/verification.py      (VerificationDataPolicy, VerificationRunPayload/Envelope, verification_control_flow_gates_v1)
src/alpha_lab/agents/data_infra/ifvg/search/child_replay.py      (baseline-capable worker + ChildAuditNeutralityReport)
src/alpha_lab/agents/data_infra/ifvg/study/…                     (dimension/study-cell/comparison/computation/delta contracts)
src/alpha_lab/agents/data_infra/ifvg/features/…                  (blocks/bundles/mbp1 contract)
tests/agents/ifvg_search/                                        (identity/registry/store/catalog-concurrency suites)
```

**Modified:** `development_access.py` (+`VerificationReplayPolicy`), `data_access.py` (third trusted branch), `dataset.py` (`table_content_hash` promotion), `docs/DECISIONS.md` (D-039…D-045), `ARCHITECTURE.md`/`docs/README.md`/`docs/pipeline_state.yaml`.

**Gate (usable increment: a verified real baseline slice):** all identity tests (core-replay reuse/independence/sensitivity, **replay-input-bundle content sensitivity + QL replay-source sensitivity + portability [P0-A]**, canonical naming, membership separation, companion versioning, **semantic/annotation exclusion + identity-projection audit [P0-B/C]**, **GeneratedProfileCapability [P0-D]**); typed-axis-value + **computation-path-scoped** authorization fail-closed tests (P0-E); catalog concurrent-publisher + crash-recovery tests; **the real vertical slice**: baseline profile + matching seed + frozen allowlist → dual-drive replay → `ChildAuditNeutralityReport` (core tables equal) → immutable save/reload/reuse → verifier link resolves → zero forbidden access → `verification_only`/`full_pipeline_not_run` stamps; both R0 verification items resolved; M0–M3 suite untouched and green.

**Release-1 acceptance is BLOCKED until the real-fixture authorization exists** (Amendment P0-F; wording per V3 P0-8): the owner-approved coverage matrix + allowlist sign-off + `VerificationAuthorizationRef` (owner decisions 21/R-5). The clean model: **code authoring may proceed speculatively after plan approval** (for R1 and, in branches, for dependent releases); **R1 cannot be accepted** without the authorized real fixture; **dependent releases may be authored in branches but cannot be declared complete, merged as accepted, or activated until R1 passes.** The one canonical verification allowlist is reused by every release (V3 P0-7 — never a different window per release).

---

## Release 2 — Multi-child FSM search, lineage, exact deltas, verifier integration

**New files**
```
src/alpha_lab/agents/data_infra/ifvg/search/orchestrator.py
src/alpha_lab/agents/data_infra/ifvg/search/lineage.py           (OpportunityLineage, NativeLineageMap, match basis)
src/alpha_lab/agents/data_infra/ifvg/search/gates.py · strategy_metrics.py
src/alpha_lab/agents/data_infra/ifvg/search/frontier.py · robustness.py · insights.py
src/alpha_lab/agents/data_infra/ifvg/study/population_delta.py · funnel_delta.py · contrasts.py · cohort.py
scripts/ifvg_search_job.py
```

**Modified:** `fsm_audit_preparation.py` (per-child neutrality-aware audit build; doc-default parity gate untouched), `replay_chart_provider.py` + `scripts/ifvg_verifier_tab.py` (`setup_id` jump kind, contract tests updated in the same change).

**Gate (usable increment: synthetic multi-child search over real primitives):** synthetic 2×2 E2E (enumerate→reuse→gates→frontier→insights, resume/cancel/lock, cross-study reuse via identical verified input manifests); lineage tests (native determinism, lineage validity, match basis, `not_comparable` disabling, **one-to-one uniqueness + collision refusal with `LineageUniquenessReport`/`LineageCollisionRecord` [P1-E]**); **generated-profile capability enforced in orchestration** (P0-D — unratified values blocked before replay); `CohortSpec` interpretation tests; **strategy-only cells run without v3/model artifacts** (P1-G); funnel-delta → exact-setup drill-through into the verifier; strategy-gate pass/fail explanations.

---

## Release 3 — Prop lifecycle: fidelity contracts first, then synthetic contract/lifecycle verification

Path-fidelity and calendar contracts **precede** the account state machine (P1-8).

**New files, in build order**
```
src/alpha_lab/propsim/trade_path.py        (TradePathArtifact/Bundle payloads+envelopes, exact path-event IDs,
                                            PathCapability, PropRulePathRequirement, PathCapabilityReport) ← FIRST
src/alpha_lab/propsim/calendar.py          (DayCountBasis, DurationRule, FirmCalendarPolicy, SimulatedClockPolicy)
src/alpha_lab/propsim/firm_contracts.py    (PropFirmContractPayload/Envelope + rule→capability/accepted-class matrix)
src/alpha_lab/propsim/contract_evidence.py (source-document → per-field evidence → compilation → review → supersession)
src/alpha_lab/propsim/account.py           (AccountWalk + PropAccountEventEnvelope total ordering)
src/alpha_lab/propsim/risk.py · withdrawal.py
src/alpha_lab/propsim/adapters.py · portfolio.py · stress.py · simulation.py (AccountSimulation + PortfolioSimulation identities,
                                            path_instance_id semantics) · prop_metrics.py
tests/propsim/… (fidelity/calendar/envelope/evidence/rule-by-rule/simulation suites)
```

**Gate — renamed to *synthetic contract/lifecycle verification* (Amendment P1-A): one synthetic contract fixture is compiled and simulated end-to-end on the R1 baseline stream.** The full §16.4 rule matrix + the P0-12…P0-17/P0-H test rows (per-rule fail-closed path capability, 1m-scenario truthfulness + scenario identities, trade-path bundle identity, total-order envelope, day-count bases, withdrawal/replacement identity, compilation + status ladder, corrected bootstrap-path semantics); the synthetic bundle reaches `synthetic_fixture_verified` — proving compiler behavior, conflict handling, field provenance, and engine integration, **never** making a real contract verified. *Optional* real-integration gate: one `first_party_verified` contract (first-party evidence compiled + owner-reviewed) — blocked by owner decisions 5/6 and **not required to prove the code architecture**. Fidelity naming throughout uses the truthful vocabulary (`OHLC_1M_UNORDERED`, `ASSUMED_1M_INTRABAR_PATH`, `historical_1m_scenario`). Frontier wired to prop metrics.

---

## Release 4 — Trader UI: wizard, active runs, results, history, and account timeline

**Normative UX authority:** `FRONTEND_UX_CONTRACT.md` §§3–29 and applicable `FUX-*` rows.

**New files:** `scripts/ifvg_ui_common.py`, `ifvg_study_tab.py`, `ifvg_study_wizard.py`, `ifvg_active_runs_tab.py`, `ifvg_results_tab.py`, `ifvg_results_compare.py`, `ifvg_results_charts.py`, `src/.../ifvg/study_status.py`, `study_presentation.py` + the AppTest/browser fixtures required by `TEST_MATRIX.md` §3.11.
**Modified:** `scripts/ifvg_lab_tab.py` (sub-nav delegation), `scripts/ifvg_verifier_tab.py`/provider exact jump seams as assigned in R2, `pyproject.toml` (Streamlit ≥1.41), `.github/workflows/ci.yml` (Ruff over scripts; preparatory lint-fix commit).

**Gate (usable increment: the complete guided study workspace over R1–R3 fixtures):**

- information architecture, route isolation, Context Research delegation, and session-state namespaces pass;
- all five modes and all eight wizard steps render their exact fields, templates, classifications, authorization states, drafts/autosave/clone/freeze behavior, and replay-vs-prop preview;
- locked/blocked axes have no widget and raw overrides are absent;
- Active Runs passes polling/manual fallback, phase checklist, keyboard funnel, exact table columns, row detail, safe cancel, skipped-stage explanations, and CLI fallback;
- Results/History pass the exact cards/no-pass copy, scope and gross/cost/net labels, frontier/selectbox twin, heatmap glyph/table twin, firm/survival/payout views, indexed explorer, dimension ribbon/four comparison panels, deterministic insights, immutable history, account timeline, and exact verifier links;
- `Development Exploratory Representative`, `match_basis`, and truthful path-scenario wording are enforced; forbidden-label and raw-path/traceback scans pass;
- R4 AppTests and browser smoke states pass the applicable `FUX-IA`, `FUX-WIZ`, `FUX-MON`, `FUX-RES`, `FUX-DRILL`, `FUX-HIST`, `FUX-STATE`, `FUX-LABEL`, and baseline accessibility rows.

---

## Release 5 — Pipeline Runner, MBP-1 Contract Readiness, and Supervised Model Ladder

*(Renamed per the owner's P1-D ruling — never described as an "MBP-1 feature platform". R5 delivers readiness only: **no active MBP-1 bundle exists and no baseline-vs-MBP-1 research claim is possible until R5B**.)*

**New files:** `search/pipeline.py` executors (`PipelineSemanticIdentity`/`ExecutionAttemptIdentity` split), `scripts/ifvg_pipeline_tab.py` + `ifvg_pipeline_job.py`; `features/bundle_feature_view.py` wiring for the **available** blocks; the MBP-1 **source/stage-cutoff contracts** (`Mbp1SourceContract`, `StageEvidenceCutoff` — contracts + refusal tests only); `ml/model_protocols.py`, `logistic_model.py`, `supervised_ladder.py`, `calibration_policies.py`, `decision_policies.py` (RejectedCandidatePolicy option space; S11 blocked text), core `drift_monitoring.py`; ML fixtures 1 + relocation test.

**Gate:** pipeline E2E under `verification_5d` (16 stages terminal; S11 BLOCKED with the exact reason text; attempt-identity test — different workers, same semantic ids); the complete Full Pipeline Configure/Preview/Launch/Monitor/Resume-Retry/Publish UX satisfies `FRONTEND_UX_CONTRACT.md` §30 and `FUX-PIPE-*`; no process launches on render/import/AppTest; verification/full-scope badges and confirmations are exact; ladder parity on fixture 1 + the real R1 slice's candidate view (control-flow only; most folds legitimately invalid); `IFVG_ORDER_FLOW_MBP1_V1` remains `planned` — bundle resolution and its UI selection refuse it and no baseline-vs-MBP-1 study is constructible; portable-artifact relocation test passes.

## Release 5B — Offline MBP-1 Feature Activation (mandatory; immediately after R5)

*(New release per the owner's P1-D ruling. R6 may proceed independently where technically possible, but the order-flow expansion is **not complete until R5B passes**.)*

**New files:** `features/mbp1_source_artifact.py` · `features/mbp1_arrow_schemas.py` · `features/mbp1_stage_windows.py` · `features/mbp1_feature_materializer.py` · `features/mbp1_coverage.py` · `features/mbp1_feature_join.py` + their test suites; dashboard additions for MBP-1 availability/coverage/bundle-selection/comparison/drill-down.

**Deliverables (the owner's 13, verbatim commitments):** (1) immutable MBP-1 source/coverage artifact; (2) exact Arrow schemas + schema hashes; (3) exact point-in-time stage-cutoff support on `(ts_event, ts_recv, sequence, source_ordinal)` per `StageEvidenceCutoff` (no `+inf` construction; same-timestamp exclusion); (4) stage-window construction for the registered IFVG lifecycle anchors; (5) the offline MBP-1 feature materializer; (6) typed missingness + source-coverage reasons; (7) exact candidate/stage joins with **no nearest-time or row-order fallback**; (8) `IFVG_ORDER_FLOW_MBP1_V1` activation as a **new versioned research-only block** (block_version bump + registry-hash change per P1-5); (9) feature coverage/validity reports; (10) the controlled Baseline vs Baseline+MBP-1 study workflow on identical profile/candidate rows/labels/folds/model protocol; (11) dashboard support (availability, coverage, bundle selection, comparison deltas, evidence drill-down); (12) five-day real control-flow verification + synthetic feature-formula fixtures — **no full-development feature materialization**; (13) immutable save/reload/reuse + exact artifact identities.

**Promotion boundary (owner ruling, recorded as decision R-6):** the activated block is **offline/research-only** — it cannot become a live model feature, an execution gate, or a Trade-Lab serving feature without a later Strategy-Core formula/parity contract and a separately approved sequential model-gated replay. Trade-Lab live serving remains out of scope.

**Gate:** all §3.8 MBP-1 rows plus §3.9/§3.10 window-definition and block-resolution rows (same-timestamp future-event exclusion, timestamp-only conservative cutoff, no-+inf scan, activation version bump), coverage/join exactness, and the controlled-study workflow runnable on the verification fixture; the R5B dashboard satisfies `FRONTEND_UX_CONTRACT.md` §35 (availability, coverage, bundle selection, missingness, comparison deltas, exact stage-window evidence, persistent `research_only_offline` label); MBP-10/deeper-feature guards remain green.

---

## Release 6 — V1 KMeans Regime Lane

**New files:** `ml/regime_contracts.py` (protocol/fit/capability/promotion split per V3 P1-3), `regime_algorithms.py` (KMeans active; others registered planned with fail-closed refusals), `regime_preprocessing.py`, `regime_service.py`, `regime_store.py`, `regime_alignment.py`, `regime_diagnostics.py`; fixtures 2 + 4-KMeans.

**V1 boundary (V3 P1-6):** R6 in V1 = the KMeans regime lane above. **The expansion (GMM, minibatch, spectral train-only diagnostics, Nyström, fixture 3 + remaining fixture-4 arms, expanded drift metrics) is a separate post-V1 release that follows V1 hardening** — V1 hardening and the operator run do not wait for it.

**Gate:** KMeans fold-local fit/assign/align/coverage/stability on fixtures; the R6 UI satisfies `FRONTEND_UX_CONTRACT.md` §35 (coverage/occupancy/stability/assignment/stratification, proposal stamps, sample-adequacy blocks, context-panel identity, planned post-V1 algorithm states); proposal stamps (`proposed_protocol_default`) surfaced in the UI; sample-adequacy gate blocks under-sampled candidate-stage fits; **the actual `CONTEXT_BAR_PANEL` schema** (panel interval/source/as-of fields + validation, P1-B) tested on synthetic 5m/15m panels incl. PIT panel→candidate assignment; **planned spectral/Nyström fit requests are refused in V1 with the correct status/reason** (P1-C — no fit implementation is callable); the expansion sub-release adds the actual implementations plus the two-ring distinctness and GMM/Nyström determinism proofs.

---

## Hardening & release verification (formerly M8)

The 5-day proofs are already continuous from R1; this phase is the release audit: full-suite protected/sealed zero-counter proof, performance budgets (`TEST_MATRIX.md` §4), and **all remaining `FRONTEND_UX_CONTRACT.md` acceptance rows**. Interactive QA includes keyboard-only navigation, responsive and screenshot evidence at 1440×900, 1024×768, 768×1024, and 390×844; representative running, no-pass, blocked, comparison, timeline, pipeline, and mobile failure states; long-ID/wide-table and capability-fallback behavior. Browser-backend unavailability leaves the gate open. Also required: clean-tree release check, `full_pipeline_not_run` proof across every verification report, and operator handoff docs.

## Operator full authorized run (formerly M9 — explicitly not an implementation gate)

Unchanged: UI-launched, after implementation acceptance **and** the owner-authorization bundle for a real charter is complete (`OWNER_DECISIONS.md` blocking list + brief §15 gates). Verify-then-activate publication; June 11 and the sealed range untouched.

---

## Dependency graph

```
R1 (identities/stores/access + REAL BASELINE SLICE; ACCEPTANCE blocked on fixture authorization)
 ├──► R2 (multi-child search + lineage + deltas + verifier)
 │        └──► R4 (UI) ──► R5 (pipeline + MBP-1 readiness + ladder) ──► R5B (MBP-1 activation)
 └──► R3 (fidelity contracts → synthetic prop lifecycle)  ──┘
R6-core (KMeans regime lane; may start alongside R5/R5B)
V1 Hardening (all-capabilities gate for the FULL-product claim: R1–R6-core + R5B)
Post-V1: regime-expansion release (GMM/minibatch/spectral diagnostics/Nyström + expanded drift)
```

**Authoring vs acceptance (V3 P0-8):** later releases may be *authored* in branches at any time after plan approval; no release (including R1's dependents) may be *declared complete, merged as accepted, or activated* until R1's acceptance — including the owner's fixture authorization — passes. R3 depends only on R1's artifacts and can proceed in parallel with R2; R5B follows R5 immediately; the order-flow expansion is not marked complete until R5B passes.

**Operator-pipeline availability is capability-gated by the selected stage plan (V3 P1-5)** — it does *not* wait for every release: a **strategy-only pipeline** is available once the strategy/search/UI dependencies (R1/R2/R4 + the relevant pipeline stages) are accepted; a **prop pipeline** after the prop lifecycle (R3) and its UI; an **MBP-1 study** after R5B; a **KMeans regime study** after R6-core; **spectral/Nyström studies** only after the post-V1 expansion. The all-capabilities hardening gate exists for the full-product completeness claim, not as a precondition for capability-scoped operator runs.
