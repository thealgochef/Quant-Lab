# Revision Changelog — Response to `IMPLEMENTATION_PLAN_REVISION_REQUEST.md`

**Date:** 2026-08-17 · **Scope:** documentation-only revision of the plan package in place (no errata appendix; contradicted text rewritten). No code, tests, artifacts, catalogs, seed snapshots, replays, feature builds, model fits, prop simulations, or search runs occurred. `REVISED_OPEN_DECISIONS.md` is not needed: the revised `OWNER_DECISIONS.md` carries the complete reclassification (including the new R-1…R-6 decisions) and this changelog records every change.

Every revision-request item was **cross-checked against the codebase before acceptance**; the load-bearing premises were verified directly: `profile_name` is a section field hashed into `ifvg_profile_hash` (SC `section.py:97, 324–331`) which is embedded in every record ID (`records.py:76–85`) and checked against seeds at replay (`replay.py:362`); the current reader discards `ts_recv` while the raw parquet retains it (`databento_parquet.py:42`); `update_context_run_catalog` has no locking (`context_run_store.py:370`). All 21 P0 and 9 P1 items were accepted; none conflicted with verified codebase facts.

Format per change: **Document · Section · Previous design · Revised design · Reason · Downstream docs updated · Remaining owner decision.**

---

## P0 changes

**P0-1 · CONTRACTS_AND_SCHEMAS §1 (was §3.1)** · Previous: single `ChildStrategyIdentity` keyed by parent_search_id, search-derived profile_name, cost_policy_sha256, audit_schema_identity · Revised: `CoreStrategyReplayIdentity` (replay-defining fields only) + `SearchChildMembership` + `CoreReplayArtifactReference` + `FsmAuditArtifactIdentity`/`ReplayChartArtifactIdentity` + `CostedEvaluationIdentity` + derived-layer identities · Reason: over-keying blocked cross-study reuse and coupled cost/audit/resource choices to the scientific replay · Downstream: IMPLEMENTATION_PLAN §1/§5/§18, TEST_MATRIX §3.1, PHASED_DELIVERY R1, README · Owner: none (architecture).

**P0-2 · CONTRACTS_AND_SCHEMAS §1.2** · Previous: `profile_name = f"{baseline}__search_{search_id[:8]}__c{ordinal:03d}"` · Revised: canonical `ifvg_search_profile_<name-free-hash16>`; display labels only in the mutable catalog; explicit verified explanation that profile_name → profile hash → record IDs · Reason: parent-specific names would give the same strategy different record IDs per study (verified) · Downstream: IMPLEMENTATION_PLAN §1, ARCHITECTURE_MAP §11.1, TEST_MATRIX §3.1, DELTA_TAXONOMY §4.3 · Owner: none.

**P0-3 · CONTRACTS_AND_SCHEMAS §7** · Previous: `QuantLabPipelineRunSpec` identity included `worker_policy` · Revised: `PipelineSemanticIdentity` (research-bearing) × `ExecutionAttemptIdentity` (workers/memory/attempt/host/retry) · Reason: resource changes after operational failures must not mint new scientific identities · Downstream: IMPLEMENTATION_PLAN §13, TEST_MATRIX §3.1, PHASED_DELIVERY R5, OWNER_DECISIONS 16/23 · Owner: none.

**P0-4 · CONTRACTS_AND_SCHEMAS §3.2 (new)** · Previous: owner decisions existed only as prose · Revised: `OwnerDecisionEvidenceRef` + `OwnerAuthorizationBundle` in every real charter; fail-closed on absent/stale/superseded/inconsistent refs; typed synthetic marker for fixtures · Reason: real launches need immutable approval evidence · Downstream: IMPLEMENTATION_PLAN §4/§19, OWNER_DECISIONS (R-2), TEST_MATRIX §3.1, PHASED_DELIVERY R1/R4 · Owner: R-2 (the decision-artifact workflow itself).

**P0-5 · CONTRACTS_AND_SCHEMAS §2.1** · Previous: `allowed_values: tuple[str|int|float|bool, ...]`, axis-level `owner_ratified: bool` · Revised: `DimensionValueSpec`/`RegisteredAxisValue`/`CompositeAxisValue` with typed payloads (None timeouts, TF tuples, dependent groups), **value-level** ratification, no raw `section_overrides` in the UI · Reason: scalar union cannot represent planned values; ratification must be value-specific · Downstream: IMPLEMENTATION_PLAN §3/§4, TEST_MATRIX §3.1, OWNER_DECISIONS 2 · Owner: value ratifications.

**P0-6 · CONTRACTS_AND_SCHEMAS §6, TEST_MATRIX §1** · Previous: four real child profiles warm-started from one doc-default seed snapshot; research gates implicitly applicable · Revised: Path A = one baseline profile + its exact matching seed, dual-drive, ≤5 days; Path B = synthetic multi-child; optional real multi-child only with per-child verified seeds (never covertly produced); `verification_control_flow_gates_v1` (nonresearch) governs verification; stamps retained · Reason: seeds are profile-bound (verified) — the previous design was impossible; 5-day fixtures cannot meet research gates · Downstream: IMPLEMENTATION_PLAN §12, PHASED_DELIVERY R1, OWNER_DECISIONS 22, README · Owner: none beyond item 21.

**P0-7 · TEST_MATRIX §2, OWNER_DECISIONS 21** · Previous: dates chosen as "last five trading days ≤ cutoff" · Revised: coverage-evidenced owner decision (documented matrix from already-authorized artifacts; 06-04…06-10 demoted to candidate); one allowlist per release, never rotated · Reason: recency does not guarantee behavioral coverage · Downstream: IMPLEMENTATION_PLAN §12, CONTRACTS §6.4 · Owner: R-5/21 (date sign-off with evidence).

**P0-8 · CONTRACTS_AND_SCHEMAS §3.3** · Previous: `parity_exempt_non_baseline=true` flag only · Revised: per-child `ChildAuditNeutralityReport` — dual-drive core-table equality (preferred; run on the baseline slice) or the formally demonstrated single-drive side-channel mechanism with independent core-stream verification + audit-stamp referential integrity; doc-default parity gate unchanged · Reason: exemption is not proof of neutrality · Downstream: IMPLEMENTATION_PLAN §6, TEST_MATRIX §1/§3, PHASED_DELIVERY R1/R2, README · Owner: none.

**P0-9 · CONTRACTS_AND_SCHEMAS §4 (new), DELTA_TAXONOMY §4.3** · Previous: population deltas compared native IDs across runs; a same-profile repeat test was the only guard · Revised: `OpportunityLineage`/`NativeLineageMap`/`LineageMatchResult`/`PopulationDeltaMatchBasis`; mandatory `match_basis` (`native_id_exact` | `profile_independent_lineage_exact` | `unmatched` | `not_comparable`); no fuzzy matching; incomparable populations disabled; two pre-ship tests (native determinism + lineage validity) · Reason: native IDs embed the profile hash (verified) — cross-profile native matching is impossible · Downstream: IMPLEMENTATION_PLAN §8, TEST_MATRIX §3.2, insights wording · Owner: none.

**P0-10 · CONTRACTS_AND_SCHEMAS §9.1 (new), DELTA_TAXONOMY §2/§8** · Previous: cohort ids referenced an internal registry; no public contract · Revised: `CohortSpec`/`CohortIdentity`/`CohortResult`/`InterpretationMode` (descriptive_slice / specialized_model / sequential_strategy_profile) with the full dimension list; UI selector maps 1:1 · Reason: brief §7A.13 requires the public contract · Downstream: TEST_MATRIX §3.6, IMPLEMENTATION_PLAN §3 · Owner: none.

**P0-11 · CONTRACTS_AND_SCHEMAS §9.2, ML_REGIME §7** · Previous: gated replay claimed without rejection semantics · Revised: `RejectedCandidatePolicy` (4-option space, none authorized), `ModelDecisionPolicySpec` with ratification ref, `WalkForwardModelSchedule`; S11 blocked with the exact required reason text · Reason: rejection semantics change slot occupancy — a strategy semantic needing owner approval · Downstream: DELTA_TAXONOMY §3.2, TEST_MATRIX §3.6, OWNER_DECISIONS R-1 · Owner: R-1.

**P0-12 · CONTRACTS_AND_SCHEMAS §5.1 (new)** · Previous: `PropTradeRecord` = realized + MFE/MAE magnitudes; adverse-first called "exact historical" · Revised: `TradePathArtifactSpec`/`TradePathEvent`/`TradePathFidelity` (4 classes; `ordered_1m_bar_path` = v1 target from cached 1m artifacts)/`PropRulePathRequirement`/`PathCapabilityReport`; per-rule minimum fidelity; fail closed; adverse-first relabeled **conservative approximation** everywhere · Reason: magnitudes carry no ordering — intratrade chronology undecidable · Downstream: IMPLEMENTATION_PLAN §7, TEST_MATRIX §3.3, PHASED_DELIVERY R3 (fidelity-first), OWNER_DECISIONS R-4, UI labeling · Owner: R-4 (fidelity floor for publishable claims).

**P0-13 · CONTRACTS_AND_SCHEMAS §5.3** · Previous: events carried `day` + dataclass payloads · Revised: `PropAccountEventEnvelope` with event_ts/ordinal/path-instance/account/phase/source-lineage/event_order_policy_id — deterministic total order for DLL/floor/fee/payout/breach/replacement chronology · Reason: `day` alone cannot order same-day interactions · Downstream: TEST_MATRIX §3.3, UI timeline · Owner: none.

**P0-14 · CONTRACTS_AND_SCHEMAS §5.4 (new)** · Previous: naked integer "days" fields · Revised: `DayCountBasis`/`DurationRule`/`FirmCalendarPolicy`/`SimulatedClockPolicy`; every time rule typed; bootstrap clock advancement defined per basis; unrepresentable rules fail closed · Reason: trading/winning/business/calendar/firm-period days are different clocks · Downstream: firm contracts, TEST_MATRIX §3.3, OWNER_DECISIONS 10 · Owner: none.

**P0-15 · CONTRACTS_AND_SCHEMAS §5.2** · Previous: `payout_behavior="request_at_first_eligibility_max"` hardcoded in `AccountWalk` · Revised: separate immutable `WithdrawalPolicyPayload/Envelope` (5 behaviors); firm contract = permitted, withdrawal policy = chosen; both in simulation identity · Reason: enables controlled withdrawal comparisons (brief Study O) · Downstream: TEST_MATRIX §3.3, OWNER_DECISIONS R-3, DELTA_TAXONOMY equalities · Owner: R-3.

**P0-16 · CONTRACTS_AND_SCHEMAS §5.8** · Previous: withdrawal/replacement/portfolio/clock/fidelity were constructor args outside identity · Revised: complete `PropSimulationIdentity` (every result-changing policy); constructor-surface audit test · Reason: no result-changing argument may bypass identity · Downstream: TEST_MATRIX §3.3, stores · Owner: none.

**P0-17 · CONTRACTS_AND_SCHEMAS §5.5 (new), OWNER_DECISIONS 6** · Previous: contract verification = manual status flag, "deferred" · Revised: deterministic offline evidence compiler (source docs → per-field `PropRuleEvidence` → compilation report → owner review → supersession); runtime consumes frozen artifacts only; stale/unverified override prohibited for publishable results; reclassified a required pre-real-run capability (built in R3) · Reason: field-level provenance is the only auditable path to "verified" · Downstream: IMPLEMENTATION_PLAN §7, TEST_MATRIX §3.3, PHASED_DELIVERY R3 · Owner: per-contract review decisions.

**P0-18 · CONTRACTS_AND_SCHEMAS §8** · Previous: single mutable JSON catalog (cloned from an unlocked existing pattern — verified unlocked) · Revised: lock-guarded append-only event log + deterministic rebuildable index; concurrent-publisher + crash-recovery tests; artifacts remain authoritative · Reason: concurrent child publishers can lose writes · Downstream: IMPLEMENTATION_PLAN §5/§17, TEST_MATRIX §3.1, ARCHITECTURE_MAP gap table · Owner: none.

**P0-19 · DELTA_TAXONOMY §6.2, CONTRACTS §10** · Previous: ordering "ts_event_then_sequence_v1" · Revised: complete key `(ts_event, ts_recv, sequence, source_ordinal)`; stage cutoffs as compatible total-order keys under a versioned exclusive rule; same-timestamp tests; verified note that the planned materializer decodes ts_recv itself (current reader discards it; raw parquet retains it) · Reason: same-timestamp bursts make the short key ambiguous · Downstream: TEST_MATRIX §3.6, ARCHITECTURE_MAP §11.4 · Owner: none. MBP-1 remains the maximum depth.

**P0-20 · OWNER_DECISIONS 1/3, CONTRACTS §2.2** · Previous: recommended first search = timeout × `parent_full_fill_invalidation` 2×2 · Revised: **single-axis** timeout-only first search; parent-fill values `blocked_pending_owner_policy_review` (the Booleans cannot express the D-1…D-5 policy space); synthetic axes for orchestration acceptance, implying no authorization · Reason: the Boolean conflates ≥8 distinct fill semantics under open owner review · Downstream: IMPLEMENTATION_PLAN §1, TEST_MATRIX §1/§3.5, README · Owner: 1/2/3 (first-search charter).

**P0-21 · TEST_MATRIX §3.4, CONTRACTS §5.7** · Previous: a "no duplicate OOS path" test rejecting duplicate bootstrap draws · Revised: duplicate sampled sequences are legitimate; unique `path_instance_id` + stored sequence hash + seed reproducibility + shared sequence across copied accounts; duplicate-prevention applies to OOS prediction rows · Reason: random block bootstrap legitimately repeats sequences · Downstream: simulation stores, brief-§16.5 wording reconciled · Owner: none.

## P1 changes

**P1-1 · ML_REGIME (header, §4, §12)** · V1 active scope narrowed to prevalence/logistic/CatBoost/KMeans + regime contracts/status/coverage/UI; GMM/minibatch/spectral-diagnostics/Nyström + fixture 3 + expanded drift → regime-expansion release, with registry entries, pinned params, and restrictions fully designed in V1 · Downstream: OWNER_DECISIONS 25–27, PHASED_DELIVERY R6, TEST_MATRIX §3.6, README.
**P1-2 · ML_REGIME §5.1** · Added the completed 5m/15m context-panel grain (train on panel, assign frozen regime to candidate stages), explicit candidate-stage sparsity caveat, and sample-adequacy gates (k=3 never "useful merely because fitting succeeds") · Downstream: OWNER_DECISIONS 28/29.
**P1-3 · ML_REGIME §5.4** · All scientific regime defaults stamped `proposed_protocol_default` + `owner_ratification_required_before_feature_eligible` · Downstream: OWNER_DECISIONS 25–30.
**P1-4 · ML_REGIME §5.3** · Fitted-pipeline references are manifest-relative + checksummed (path/hash/bytes/schema/versions/payload-hash); relocation test; no machine-path dependence · Downstream: TEST_MATRIX §3.6.
**P1-5 · DELTA_TAXONOMY §7** · Planned-block activation = new `block_version` + registry-hash + bundle-hash change on an unchanged platform (not a bare status flip); version-bump test · Downstream: ML_REGIME §12, TEST_MATRIX §3.1.
**P1-6 · IMPLEMENTATION_PLAN §8, CONTRACTS §11, UI** · `Development Exploratory Representative` wording everywhere; publishable representative requires an approved outer protocol; forbidden-wording scan · Downstream: OWNER_DECISIONS 15, status vocabulary, TEST_MATRIX §3.7.
**P1-7 · PHASED_DELIVERY R1** · The real baseline vertical slice moved from final hardening to Release 1 and gates all later releases.
**P1-8 · PHASED_DELIVERY R3** · Path-fidelity + calendar + rule-capability contracts precede the account state machine.
**P1-9 · PHASED_DELIVERY (whole)** · Monolithic M1–M8 restructured into six usable releases + hardening + operator run.

---

## Cross-document contradiction audit

Each topic checked for consistent expression across `IMPLEMENTATION_PLAN` (IP), `CONTRACTS_AND_SCHEMAS` (CS), `DELTA_TAXONOMY` (DT), `ML_REGIME_CONTRACT_PLAN` (ML), `TEST_MATRIX` (TM), `OWNER_DECISIONS` (OD), `PHASED_DELIVERY` (PD), `README` (RM), `ARCHITECTURE_MAP` (AM):

| Topic | Consistent statement | Where |
|---|---|---|
| Identity decomposition | Core replay ≠ membership ≠ companions ≠ costed ≠ attempt; reuse across studies | CS§1 · IP§1/§5 · TM§3.1 · PD-R1 · RM · AM§11.1 |
| Verification seed policy | One baseline + matching seed; seeds profile-bound; synthetic multi-child; nonresearch gates | CS§6 · TM§1 · IP§12 · PD-R1 · OD-21/22 · RM · AM§11.1 |
| First search recommendation | Timeout-only single axis; parent-fill blocked pending owner policy | IP§1 · CS§2.2 · OD-1/3 · TM§3.5 · RM |
| Audit parity | Baseline parity gate intact; per-child neutrality report (dual-drive on the slice, mechanism proof for children) | CS§3.3 · IP§6 · TM§1/§3.3 · PD-R1/R2 · RM |
| Lineage matching | Native IDs profile-bound; lineage layer + mandatory match basis; no fuzzy matching; disable over infer | CS§4 · DT§4.3 · IP§8 · TM§3.2 · AM gap table |
| Prop path fidelity | Fidelity classes; per-rule minimums; fail closed; "conservative approximation" labeling | CS§5.1 · IP§7 · TM§3.3 · PD-R3 · OD-R4 · DT equalities |
| Calendar semantics | Typed bases + clock policy; unrepresentable → fail closed | CS§5.4 · IP§7 · TM§3.3 · OD-10 |
| Withdrawal policy | Separate from firm contract; both in simulation identity | CS§5.2/§5.8 · IP§7 · TM§3.3 · DT equalities · OD-R3 |
| Contract evidence | Field-level compiler required pre-real-use; no stale override for publishable results | CS§5.5 · IP§7 · TM§3.3 · OD-6 · PD-R3 |
| Cohort semantics | Public CohortSpec; three interpretation modes; only the third executable | CS§9.1 · DT§2/§8 · TM§3.6 · IP§3 |
| Model-gate semantics | RejectedCandidatePolicy defined, none authorized; S11 blocked with the exact reason text | CS§9.2 · ML§7 · DT§3.2 · TM§3.6 · IP§13 · OD-R1 |
| ML/regime scope | V1 = prevalence/logistic/CatBoost/KMeans; rest designed-now/implemented-R6; defaults are proposals | ML (header/§4/§12) · OD-25–30 · PD-R6 · TM§3.6 · RM |
| Milestone order | Six releases; real slice in R1 gates everything; fidelity before prop engine | PD · IP§15 · TM budgets · RM |
| Publication status | Development Exploratory Representative; publishable needs approved outer protocol; package itself not implementation-authorized | IP§8 · CS§11 · ML§12 · OD-15 · TM§3.7 · RM |

~~No remaining cross-document contradictions were found after revision.~~ *(This claim is superseded: the second-pass audit found the defects addressed in the Amendment V2 section below; the current consistency statement is `FINAL_CONSISTENCY_AUDIT.md`.)* `FSM-PLAN-DOCUMENT.md` carries a superseded-provenance banner and does not govern.

---

# Amendment V2 — Response to `IMPLEMENTATION_PLAN_REVISION_AMENDMENT_V2.md`

**Date:** 2026-08-17. Narrow second-pass amendment; every §2 accepted first-revision item preserved. Cross-checks performed before acceptance: `hash_allowlisted_source_files`/`permitted_source_hashes`/`authoritative_source_blob` exist and already feed existing manifest identities (P0-A's "use native primitives" is satisfiable); the `CONTEXT_BAR_1M` enum-vs-narrative contradiction was confirmed in the pre-amendment ML doc (P1-B). All P0-A…P0-H and P1-A…P1-G items accepted; the P1-D scope choice was put to the owner, whose ruling (R5 readiness-only + mandatory R5B) is applied verbatim.

**P0-A · CONTRACTS §1.1–1.2, IMPLEMENTATION_PLAN §5, ARCHITECTURE_MAP gap table, TEST_MATRIX §3.8, PHASED_DELIVERY R1** · Previous: core replay keyed by date-set hash + `artifacts_tag` + SC identity · Revised: content-addressed `ReplayInputBundle` (exact source-partition hashes + day-artifact manifests + schema era + access identity, canonically sorted, built on the existing hashing primitives) + `quant_lab_replay_source_identity` (scoped QL replay-code source-tree evidence) inside `CoreStrategyReplayPayload`; explicit change/no-change rule lists + 6 sensitivity/portability tests · Reason: tags identify a request, not bytes; caches are mutable on disk · Owner: none.

**P0-B · DELTA_TAXONOMY §2, CONTRACTS §11, TEST_MATRIX §3.8** · Previous: `cell_id` hashed all 17 dimensions incl. the `annotation_only` engineering protocol · Revised: `StudyCellSemanticPayload` (16 dimensions) hashed; `StudyCellAnnotation` never hashed; misclassification rule (output-changing values move into the owning semantic protocol) + 4 tests · Reason: annotation could fork scientific identity · Owner: none.

**P0-C · CONTRACTS §0 (+ all id-bearing contracts), DELTA_TAXONOMY, ML_REGIME (RegimeModel payload/envelope), TEST_MATRIX §3.8** · Previous: several contracts carried their own ids/artifact hashes inside the hashed object (`ComparisonSpec.comparison_id`, `CohortSpec.cohort_id`, `RegimeModelSpec.schema_hash/artifact_hash`, …) · Revised: uniform non-self-referential Payload/Envelope convention + exclusion list + the identity-projection audit test · Reason: circular/ambiguous identity rules · Owner: none.

**P0-D · CONTRACTS §1.4, DELTA_TAXONOMY §2, ARCHITECTURE_MAP gap table, TEST_MATRIX §3.8, PHASED_DELIVERY R1/R2** · Previous: study-cell text implied every profile name is gated by the fixed `PROFILE_CAPABILITY_REGISTRY` (impossible for generated names) · Revised: `ResolvedSearchProfileRef` + `GeneratedProfileCapability` (5 statuses; fixed registry gates baselines only; generated children never inserted; no legacy-launcher leakage) + 5 tests · Reason: generated profiles need their own authorized capability path · Owner: none.

**P0-E · CONTRACTS §3.2, OWNER_DECISIONS (key + matrix), IMPLEMENTATION_PLAN §12, TEST_MATRIX §3.8, README** · Previous: one fixed all-capabilities authorization bundle; `verification_5d` used the synthetic marker · Revised: `AuthorizationRequirement(Set)` + `derive_authorization_requirements` (path-scoped; unrelated decisions never demanded) + real `VerificationAuthorizationRef` for the real slice (synthetic marker confined to synthetic fixtures) + the 10-mode parameterized test · Reason: over-blocking and a synthetic marker on real data were both wrong · Owner: R-2 workflow unchanged.

**P0-F · OWNER_DECISIONS (new class + items 21/R-5), README, IMPLEMENTATION_PLAN §12, PHASED_DELIVERY R1, TEST_MATRIX** · Previous: fixture decisions were only `BLOCKING-RESEARCH`; "none of it blocks the build" phrasing · Revised: **BLOCKING-VERIFICATION** class (blocks release acceptance, not code authoring); R1 acceptance explicitly blocked on fixture authorization + coverage sign-off; blocking summary re-cut by capability/run type · Reason: R1 cannot close without the approved real slice · Owner: 21/R-5 remain to be signed.

**P0-G · CONTRACTS §10, DELTA_TAXONOMY §6.2, TEST_MATRIX §3.8, IMPLEMENTATION_PLAN §12** · Previous: stage cutoff = `(stage_ts, +inf, +inf, +inf)` exclusive upper bound · Revised: `StageCutoffKind`/`StageEvidenceCutoff` — exact stage-triggering-event order keys; timestamp-only ⇒ strict `ts_event < stage_ts` with all same-timestamp events excluded/typed `same_timestamp_order_unavailable`; versioned completed-bar boundaries; never widen; no-`+inf` source-scan test · Reason: the old bound admitted same-timestamp events occurring after the stage decision (PIT leak) · Owner: none.

**P0-H · CONTRACTS §5.1/§5.4, IMPLEMENTATION_PLAN §7, TEST_MATRIX rows, OWNER_DECISIONS R-4, PHASED_DELIVERY R3, DELTA_TAXONOMY, README** · Previous: `ordered_1m_bar_path` described as exact under a declared intrabar policy; simulation mode `historical_1m_path`; one v1 fidelity floor · Revised: fidelity classes `CLOSED_TRADE_ONLY / OHLC_1M_UNORDERED / ASSUMED_1M_INTRABAR_PATH / ORDERED_MBP1_EVENT_PATH / ORDERED_FILL_EVENT_PATH`; `OhlcBarPathObservation(observed_intrabar_order="unknown")`; scenario expander policies with distinct scenario identities; modes `historical_closed_trade / historical_1m_scenario / historical_ordered_event_replay / day_block_bootstrap / stress`; per-rule minimum fidelity (no universal floor); `trade_path_artifact_id` + manifest hash + fidelity + scenario policy inside `PropSimulationIdentity`; R-4 rewritten (publishable historical claims need actual ordered evidence for chronology-sensitive rules) · Reason: OHLC bars do not contain intrabar order — an assumed order is a scenario, not history · Owner: R-4 matrix.

**P1-A · PHASED_DELIVERY R3, CONTRACTS §5.3, OWNER_DECISIONS 5/6, TEST_MATRIX, IMPLEMENTATION_PLAN §7, README** · R3 gate renamed *synthetic contract/lifecycle verification*; status ladder `synthetic_fixture_verified → first_party_evidence_compiled → owner_reviewed → first_party_verified → superseded`; synthetic bundles can never verify a real contract; the optional first-party gate is blocked by decisions 5/6 and not required for the architecture.

**P1-B · ML_REGIME §3 (enum + payload + validation + tests), CONTRACTS §9 pointer, DELTA_TAXONOMY, TEST_MATRIX, OWNER_DECISIONS 28** · `ObservationGranularity.CONTEXT_BAR_PANEL` with `panel_interval_seconds/panel_source_artifact_id/panel_as_of_policy_id` (1m = interval 60; no contradictory enum name); panel-field validation; 5m/15m/invalid-combination/PIT-assignment tests.

**P1-C · ML_REGIME (registry + restriction section), TEST_MATRIX, PHASED_DELIVERY R6, README** · V1 ships capability-registry entries + fail-closed planned/blocked states only; **no spectral/Nyström fit implementation is callable in V1**; the frozen restrictions bind the future implementations; V1 checks = refusal with correct status, no predictive-bundle entry, distinct identities.

**P1-D · PHASED_DELIVERY R5/R5B, IMPLEMENTATION_PLAN §12/§15/§18, DELTA_TAXONOMY, ARCHITECTURE_MAP, OWNER_DECISIONS R-6, README** · **Owner ruling applied verbatim**: R5 renamed "Pipeline Runner, MBP-1 Contract Readiness, and Supervised Model Ladder" (no active bundle, no baseline-vs-MBP-1 claim); mandatory **R5B — Offline MBP-1 Feature Activation** immediately after R5 with the owner's 13 deliverables (6 named modules; activation = new block version); promotion boundary `research_only_offline` recorded as decision R-6; R6 may run alongside but the order-flow expansion completes only with R5B.

**P1-E · CONTRACTS §4, DELTA_TAXONOMY §4.3, TEST_MATRIX §3.8, IMPLEMENTATION_PLAN** · Full lineage payloads (Setup/Candidate/Decision/Trade, profile-independent + PIT + source-derived) + one-to-one uniqueness rule + `LineageUniquenessReport`/`LineageCollisionRecord`; collisions → `not_comparable` (no dedupe/keep-first/fuzzy) + 5 tests.

**P1-F · OWNER_DECISIONS 1/2, IMPLEMENTATION_PLAN §1, CONTRACTS §2, README** · "Ratified variants" wording removed → "existing prepared or implemented variants"; the first-search scientific contrast explicitly includes the **`None`/unbounded doc-default baseline** vs owner-approved bounded challengers.

**P1-G · DELTA_TAXONOMY §2, CONTRACTS §12, TEST_MATRIX §3.8, IMPLEMENTATION_PLAN §5** · Typed `none_*_v1` identities for context/feature/label/model/regime axes; strategy-only cells (core replay + companions + costed stream + prop sim) never blocked by a missing v3 pair; computation path decides materialization; 4 tests.

## Amendment V2 confirmation

No code, tests, data artifacts, catalogs, seeds, replays, MBP-1 feature builds, model fits, prop simulations, or source-data runs were created, modified, or executed during this amendment. Only the Markdown planning documents in this folder changed. The cross-document consistency statement now lives in `FINAL_CONSISTENCY_AUDIT.md`.

---

# V3 Compact Patch — third-pass corrections

**Date:** 2026-08-17. Nine P0 + six P1 items, delivered inline by the owner and applied in place. Format: Item · Previous · Revised · Docs.

**P0-1 (+P1-2)** · `ComparisonSpec.comparison_id`, `DeclaredContrastSpec.contrast_id`, `FeatureBlockSpec.block_id`, `FeatureBundleSpec.bundle_id` sat inside the hashed specs; registry names conflated with resolved identities · Payload/Envelope conversion completed for all four; explicit key/resolved pairs (`feature_block_key`/`resolved_feature_block_id`, `feature_bundle_key`/`resolved_feature_bundle_id` [supersedes `bundle_schema_hash`], `decision_policy_key`/`resolved_decision_policy_id`, `algorithm_key`/`resolved_regime_protocol_id`) · DT §3/§5/§6/§7, CS §0/§9, ML naming note, TM §3.9, IP §5.

**P0-2** · CONTRACTS relied on "unchanged shapes / as previously specified" while earlier versions were superseded · **CONTRACTS_AND_SCHEMAS.md rewritten fully self-contained** (option 1, per the README supersession language) — complete field-level schemas merged for the charter, firm/payout/withdrawal/fee/calendar/event/risk/portfolio/stress/simulation stack, pipeline, stores, cohorts, model-gating, frontier/insights/providers · CS (whole), TM §3.9 authority check.

**P0-3** · `source_kind: str  # "mbp10" | "mbp1" | "trades"` contradicted "MBP-10 structurally unrepresentable everywhere" · **Option B**: `Literal["mbp1", "trades", "legacy_verified_replay_source"]` — opaque provenance only (early development eras genuinely decode from mbp10 partitions, verified); unqueryable by the feature layer/bundles/dashboard/live/models (per-surface guard tests); package-wide claim reworded to the truthful scoped statement · CS §1.1, IP §12/§19, RM, TM §3.9.

**P0-4** · partition refs could not distinguish two partitions of one trading day · `source_partition_id` + `source_partition_utc_date` + `source_manifest_id` + `relative_logical_partition_key`, included in canonical ordering · CS §1.1, TM §3.9.

**P0-5** · `access_audit_identity` ambiguous inside replay identity · split: `ReplayAccessAuthorizationRef` (deterministic preflight — in identity) vs `ReplayExecutionAccessAudit` (runtime evidence — execution attempts only) · CS §1.1, IP §5, TM §3.9.

**P0-6** · study-cell data lineage assumed a v2/v3 pair + formula id · concrete `DataLineagePayload` with typed optionals; strategy-only cells carry no `ifvg_context_formula_v2` identity · DT §2, CS §11 pointer, TM §3.9.

**P0-7** · "One allowlist per release; never rotated" was readable as per-release windows · **one canonical allowlist across the complete implementation-verification program**; changing it = new verification-policy version; never accumulates coverage · OD 21, CS §6, TM §2/§3.9.

**P0-8** · README's closing section kept the universal-blocking phrasing; PHASED said "no later release starts until fixture authorization" · authoring-vs-acceptance model: authoring proceeds speculatively; R1 cannot be *accepted* without the authorized fixture; dependents may be authored in branches but not declared complete/accepted/activated until R1 passes · RM, OD matrix, PD (R1 note + graph), IP §15, TM §3.9.

**P0-9** · the V2 audit's "no remaining contradiction" was premature (missed the five items above) · conclusion withdrawn and the audit re-run after corrections — see the V3 re-audit section of `FINAL_CONSISTENCY_AUDIT.md` · FCA.

**P1-1** · `frozen=True` treated as sufficient · deep-immutability rules (canonical tuples / immutable-mapping wrappers with canonical serializers / deep-copy + revalidation) + mutation-adversarial tests · CS §0.3, TM §3.9.

**P1-3** · `RegimeModelPayload` mixed protocol, fit state, and promotion status · split into `RegimeProtocolPayload`/`RegimeFitPayload`/`RegimeCapabilityAssessment`/`RegimePromotionDecision` — promotion never changes numerical fit identity · ML §3, TM §3.9.

**P1-4** · one strict pre-trigger cutoff for every feature · per-feature `WindowTriggerSemantics` (`pre_trigger_exclusive` / `post_trigger_inclusive` / `completed_bar_as_of`; explicit transition-interval bounds) — a feature-definition declaration · CS §10, DT §6.2, TM §3.9.

**P1-5** · operator run gated behind R5B + full R6 · capability-gated readiness by selected stage plan (strategy-only / prop / MBP-1 / KMeans-regime / spectral each unlock with their own dependencies); the all-capabilities gate is only the full-product claim · CS §7, PD graph, IP §15, TM §3.9.

**P1-6** · release graph required the R6 expansion before hardening while the ML doc called it a later release · **V1 boundary chosen: V1 hardens and releases with KMeans; GMM/minibatch/spectral/Nyström are post-V1** · ML, PD, OD 25–27, RM, TM §3.9.

## V3 confirmation

No code, tests, data artifacts, catalogs, seeds, replays, MBP-1 feature builds, model fits, prop simulations, or source-data runs were created, modified, or executed during the V3 patch. Only the Markdown planning documents changed.


---

## V4 final contract-closure patch (2026-08-18 UTC)

**Scope:** documentation-only correction of the remaining implementation seams found in the fourth independent review. No production code, tests, artifacts, catalogs, seeds, replays, feature builds, model fits, simulations, or source-data runs were executed.

| Document / section | Previous design | Final corrected design | Reason |
|---|---|---|---|
| Contracts §0 / Delta §6 | Feature block definition mixed status, key, resolved ID, formulas, and schemas | `FeatureBlockDefinition` + `FeatureBlockResolutionPayload/Envelope` | Prevent registry status from masquerading as a resolved content identity |
| Contracts §9 / ML §7 | Old `decision_policy_id`, self-containing policy/schedule specs, stale regime ID | `DecisionPolicyPayload/Envelope` + `WalkForwardModelSchedulePayload/Envelope` + resolved regime protocol refs | Complete non-self-referential identity model |
| Contracts §3/§6 | Real verification authorization existed only as a narrative/type | `VerificationRunPayload/Envelope` binds pipeline, authorization, allowlist, seed, profile, and coverage matrix | Make fail-before-path authorization executable and hash-pinned |
| Contracts §5.1 | One trade path artifact used as a whole simulation stream | One-trade artifacts plus `TradePathBundleEnvelope`; every path event gets an ID | Exact stream coverage and event lineage |
| Contracts §5.4 | Singular firm/risk policy fields under a multi-leg portfolio | Separate account and portfolio simulation identities; complete policies per leg | Truthful mixed-firm/mixed-policy identity |
| Contracts §5.1/§5.2 | Fidelity treated as a total order; adverse-first embedded in firm rule | Required path capabilities + accepted classes; scenario order lives only in simulation policy | Ordered fills and market chronology are not interchangeable |
| Contracts §10 / Delta §6.2 | Trigger semantics in prose only | `Mbp1FeatureWindowSpec` included in block resolution identity | Exact point-in-time feature definition and repeatability |
| ML §3 | Assignment/assessment lacked exact fit/fold provenance | assignments cite protocol+fit; assessments cite fit set+fold set; promotion separate | Promotion cannot rewrite numerical fit identity |
| Contracts/Test matrix | Undefined current types, ellipsis stage enum, prior-version references | Complete current symbols, all 16 stages, inlined acceptance mappings | Current authority package is self-contained |
| README / delivery / audit | stale release counts, R-5 index, pre-final audit conclusion | seven increments, R-6 indexed, final audit rerun | Cross-document consistency |


---

## V5 frontend-authority restoration patch

**Reason:** the original implementation plan contained a detailed, codebase-specific frontend design, but successive backend-contract amendments compressed it into a short summary. The major surfaces remained named, yet field-level wizard behavior, monitoring details, result semantics, accessibility fallbacks, exact viewports, and several acceptance requirements were no longer independently authoritative.

**Changes applied:**

1. **New `FRONTEND_UX_CONTRACT.md`** — created by this package, not supplied by the owner. It is the complete normative implementation contract for planning brief §10.1–§10.30 and includes the module map, five modes, eight wizard steps, draft/autosave/clone behavior, Active Runs, Results/History, pipeline workflow, exact verifier/account links, statuses, scope labels, empty/failure states, accessibility, responsive behavior, performance, safety, `FUX-*` acceptance matrix, and a source-section retention crosswalk.
2. **`IMPLEMENTATION_PLAN.md` §4** — replaced the compressed frontend paragraph with an explicit normative reference and a structured non-regression summary; added the frontend contract to the authority table.
3. **`CONTRACTS_AND_SCHEMAS.md` §13** — added pure presentation contracts for workspace routes, disclosure levels, result scopes, statuses, and empty states; kept display metadata outside scientific identity.
4. **`PHASED_DELIVERY.md` R4/R5/R5B/R6/hardening** — expanded gates to cite the exact UX contract and required frontend evidence.
5. **`TEST_MATRIX.md` §3.11** — added field-level AppTest, integration, source-scan, keyboard, browser, responsive, fallback, performance, and safety rows for all `FUX-*` requirements.
6. **`ARCHITECTURE_MAP.md`** — added the target frontend authority and module/responsive-test boundary to the current UI gap section.
7. **`README.md`** — added `FRONTEND_UX_CONTRACT.md` to the authoritative reading order and recorded this patch.
8. **`FINAL_CONSISTENCY_AUDIT.md`** — added a frontend-retention audit across every planning-brief §10 section.
9. **Implementation kickoff prompt** — updated separately to require reading and implementing the complete frontend contract and to make its browser/keyboard/screenshot evidence a release gate.

**No code or data work:** this patch changes planning Markdown only. No production code, tests, artifacts, catalogs, seeds, replays, feature builds, models, prop simulations, or source reads were executed.
