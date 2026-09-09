# Implementation Plan — Quant-Lab Robust FSM Configuration Search & Prop-Firm Realization Workspace

**Contract name:** `ifvg_prop_robust_config_search_v1`
**Source brief:** `..\original-plan-brief\QUANT_LAB_FSM_PROP_SEARCH_DASHBOARD_PLANNING_BRIEF_V4.md` (the revision inputs this plan answers are in `..\completed-revision-documents\`)
**Prepared:** 2026-08-17 by Claude Fable; **final documentation correction 2026-08-18 UTC** per `IMPLEMENTATION_PLAN_REVISION_REQUEST.md` (P0-1…P0-21, P1-1…P1-9), `IMPLEMENTATION_PLAN_REVISION_AMENDMENT_V2.md` (P0-A…P0-H, P1-A…P1-G), **the V3 compact patch, the V4 final contract-closure patch, and the V5 frontend-authority restoration patch** (see `REVISION_CHANGELOG.md` and `FINAL_CONSISTENCY_AUDIT.md`).
**Status:** PLAN — awaiting owner approval. **No production code, tests, artifacts, catalogs, or runtime data were modified** while producing or revising this plan (brief §18 item 35; revision §1.1). Implementation must not begin until this revised package is approved; no implementation may begin from superseded (pre-revision) plan text.

**Document set:**

| Document | Brief §18 items |
|---|---|
| `IMPLEMENTATION_PLAN.md` (this) | 1, 3–5, 7–16, 18–19, 21–22, 27–29, 35 |
| `FRONTEND_UX_CONTRACT.md` | complete normative frontend and user-experience contract for brief §10.1–§10.30, code module map, fallbacks, accessibility, responsive behavior, and FUX acceptance IDs |
| `ARCHITECTURE_MAP.md` | 2 (current codebase map + gap table) |
| `CONTRACTS_AND_SCHEMAS.md` | 6–7 detail, 30 summary — **revised**: identity decomposition, typed axis values, owner authorization, audit neutrality, path fidelity, calendar/withdrawal/evidence contracts, catalog concurrency, cohort + gating semantics |
| `DELTA_TAXONOMY.md` | 23–26 — **revised**: lineage + match basis, MBP-1 ordering, block versioning |
| `ML_REGIME_CONTRACT_PLAN.md` | 30–34 — **revised**: bounded V1 scope, panel grain, proposal stamps, portable refs |
| `TEST_MATRIX.md` | 17, 27 — **revised**: two-path verification, new identity/lineage/fidelity tests |
| `OWNER_DECISIONS.md` | 20 — **revised**: reclassifications + new R-1…R-6 decisions |
| `PHASED_DELIVERY.md` | 18 — **revised**: seven usable increments, early real vertical slice |
| `REVISION_CHANGELOG.md` | per-change log + cross-document contradiction audit |

---

## 1. Executive recommendation

**Build it as a new, separately-contracted research lane inside the existing IFVG infrastructure — with zero Strategy-Core changes in v1 — around a decomposed identity model whose reusable core is the strategy replay itself.** The decisive verified findings:

1. **The child-configuration mechanism exists and is proven** (`resolve_profile_config` + the shipped tf-variant scripts), but identity must be decomposed to use it safely: `ifvg_profile_hash` covers the full section **including `profile_name`**, and that hash is inside every record ID — so the reusable replay identity (`CoreStrategyReplayIdentity`) carries a **canonical, study-independent profile name** (`ifvg_search_profile_<hash16>` derived from the name-free section content), while parent-study linkage lives in a separate `SearchChildMembership`. The same resolved strategy is replayed once and reused across studies; cost, risk, prop, audit, chart, and resource choices never invalidate it.
2. **The immutability problem is already solved** (`manifest.py` protocol, cloned not reinvented) — extended by a concurrency-safe, rebuildable catalog index for the one mutable surface.
3. **The prop simulator half-exists** (`alpha_lab.propsim` evaluation walk + day-block bootstrap + data-driven presets). The plan extends it **fidelity-first**: trade-path fidelity classes and typed calendar semantics come before the funded-phase `AccountWalk`; every firm rule declares required path capabilities and accepted fidelity classes and fails closed when those requirements are not met; the MFE/MAE adverse-first mode is labeled a *conservative approximation*, never "exact historical"; withdrawal behavior is a separate policy from the firm contract; every result-changing policy enters the simulation identity; and real contracts compile from field-level first-party evidence before any real use.
4. **Verification respects the physics of the engine:** seeds are profile-bound, so the ≤5-day real budget buys **one baseline vertical slice** (exact matching seed, dual-drive audit-neutrality proof, store round-trip, verifier link) — run early, in Release 1 — while all multi-child behavior is verified synthetically. Research gates are never applied to five-day fixtures.
5. **Cross-profile comparisons need lineage, not native IDs** (native IDs embed the profile hash): a profile-independent `OpportunityLineage` layer with a mandatory `match_basis` on every population delta; where an exact key cannot be constructed, the comparison is disabled rather than inferred.

**Recommended first real study (revised):** a **single-axis** search over `parent_retest_timeout_1m_bars`, with the **unbounded (`None`) doc-default as the explicit comparison baseline** and owner-approved bounded challengers (proposed {240, 360, 480}; the existing 240/480 preparations are implementation evidence, not ratifications — Amendment P1-F). Parent-fill handling is **not** a first-search axis (`blocked_pending_owner_policy_review`; FSM-audit D-1…D-5). Orchestration acceptance uses synthetic axes and implies no authorization.

## 2. Current architecture and gap analysis

See `ARCHITECTURE_MAP.md`. Summary unchanged: reuse the store/contract/job/safety/propsim/UI primitives; the net-new construction is the decomposed identity + axis/authorization registries, the parent orchestrator, lineage, the fidelity-first prop lifecycle, frontier/robustness/insights, the §7A dimension/delta layer, the bounded §7B ML/regime lane, and the wizard/monitor/results/pipeline UI.

## 3. Exact backend changes (brief §18 item 4)

Full schemas in `CONTRACTS_AND_SCHEMAS.md`; authoritative per-release file lists in `PHASED_DELIVERY.md`. Shape:

- **`ifvg/search/`** — `identities` (CoreStrategyReplayIdentity / SearchChildMembership / companion + costed identities / canonical naming), `axis_registry` (typed `RegisteredAxisValue`/`CompositeAxisValue`, value-level ratification), `authorization` (OwnerDecisionEvidenceRef / OwnerAuthorizationBundle, fail-closed), `charter`, `child_replay` (+ `ChildAuditNeutralityReport`), `orchestrator`, `lineage`, `gates` + `strategy_metrics`, `frontier` (development-exploratory representative), `robustness`, `insights` (match-basis-aware), `failure`, `store` + `catalog` (lock-guarded append-only event log, rebuildable index), `verification` (two-path design + `VerificationRunPayload/Envelope` binding the real authorization), `pipeline` (semantic identity vs execution attempts).
- **`ifvg/study/` + `ifvg/features/`** — §7A layer incl. `CohortSpec`, lineage-based population deltas, MBP-1 four-part ordering, versioned block activation.
- **`ifvg/ml/`** — §7B layer, V1 active scope prevalence/logistic/CatBoost/KMeans; `RejectedCandidatePolicy` option space (none authorized); portable checksummed artifact refs.
- **`propsim/`** — `trade_path` (per-trade artifacts, whole-stream bundles, exact event IDs, and path capabilities), `calendar` (day-count), `firm_contracts` + `contract_evidence` (field-level compiler), `account` (event envelope), `risk`, `withdrawal`, `adapters`, `portfolio`, `stress`, `simulation` (complete identity, path-instance semantics), `prop_metrics`.
- **Small modifications (complete list):** `development_access.py`, `data_access.py`, `dataset.py` (hash promotion), `fsm_audit_preparation.py` (per-child neutrality-aware build; baseline parity gate untouched), `replay_chart_provider.py` (+`setup_id`), `scripts/ifvg_lab_tab.py`, `scripts/ifvg_verifier_tab.py`, `pyproject.toml`, `.github/workflows/ci.yml`, docs.
- **Never modified:** all M0–M3 modules, all existing propsim modules, all of Strategy-Core, all existing artifacts/catalogs.

## 4. Exact frontend changes (brief §18 items 5, 12)

`FRONTEND_UX_CONTRACT.md` is the **complete normative frontend authority**. Its §37 crosswalk retains every planning-brief requirement from §10.1 through §10.30; implementation may not substitute this section's summary for that contract.

### 4.1 Information architecture

Pure Streamlit + Plotly; the deleted React prototype is not revived. Retain the IFVG Lab top-level tabs and add the Experiments sub-navigation:

```text
New Study | Active Runs | Results | History | Context Research
```

`Context Research` delegates to the existing M0–M3 panel unchanged. A session-state-backed horizontal radio is required so only the selected panel executes and programmatic wizard/monitor/result navigation remains possible. Session-state namespaces are `ifvg_study_v1_*` and `ifvg_pipeline_v1_*`.

### 4.2 Module decomposition

The frontend remains decomposed into focused modules rather than a monolith:

```text
scripts/ifvg_ui_common.py
scripts/ifvg_study_tab.py
scripts/ifvg_study_wizard.py
scripts/ifvg_active_runs_tab.py
scripts/ifvg_results_tab.py
scripts/ifvg_results_compare.py
scripts/ifvg_results_charts.py
scripts/ifvg_pipeline_tab.py
scripts/ifvg_search_job.py
scripts/ifvg_pipeline_job.py
src/.../ifvg/study_status.py
src/.../ifvg/study_presentation.py
```

Pure logic lives under `src/`; Streamlit scripts stay thin and are added to Ruff coverage. The Streamlit pin is `>=1.41`. `st.fragment`, Plotly selection, dialogs, and pinned columns each have a behavior-preserving fallback defined in `FRONTEND_UX_CONTRACT.md` §4.1.

### 4.3 Non-negotiable UX surfaces

The authoritative contract specifies, field by field:

- five New Study modes and the full eight-step wizard;
- disk-persisted drafts, autosave, Save Draft, History, and immutable Clone as New Search;
- registered-axis cards grouped by market meaning, with blocked/locked fields showing no widget;
- prop-contract, account/risk/withdrawal/replacement, benchmark, validation, authorization, combination-preview, and freeze/launch interactions;
- Active Runs phase checklist, keyboard funnel controls, indexed child table, exact row detail, safe cancel, and manual/CLI fallbacks;
- Results overview, frontier, heatmap, firm matrix, survival curves, payout distributions, configuration explorer, four-panel comparison, deterministic insights, exact verifier drill-down, and account timeline;
- Full Pipeline Configure/Preview/Launch/Monitor/Resume-Retry/Publish behavior and `prepared_not_published` semantics;
- Summary/Analyst/Audit disclosure; explicit candidate/execution/prop/bootstrap/stress scopes; gross/costed/net labels;
- exact empty/blocked/failure states, sanitization, and no color-only meaning;
- keyboard navigation, screen-reader-friendly visible labels, chart/widget twins, responsive layouts, and screenshots at 1440×900, 1024×768, 768×1024, and 390×844.

### 4.4 Final revision terminology

The UI renders registered values only; no raw `section_overrides`. Real scope shows the computation-path-scoped authorization state and fails closed. Development output uses **Development Exploratory Representative**. Every population delta displays `match_basis`; unsupported commonality is suppressed. Assumed 1m paths are visibly scenario/approximation. The account timeline consumes totally ordered `PropAccountEventEnvelope` events. The pipeline monitor displays `pipeline_semantic_id` plus attempt history and capability-scoped readiness. R5B MBP-1 surfaces remain `research_only_offline`.

### 4.5 Frontend acceptance

R4, R5, R5B, R6, and hardening must satisfy the applicable `FUX-*` rows in `FRONTEND_UX_CONTRACT.md` §36 and `TEST_MATRIX.md` §3.11. Browser/keyboard/screenshot evidence is a real gate; unavailable interactive browser infrastructure leaves the gate open.

## 5. Identity and immutability model (brief §18 item 7; revision P0-1/2/3/16)

The identity chain is decomposed (full field lists in `CONTRACTS_AND_SCHEMAS.md` §0–§1):

```
ReplayInputBundle  (content-addressed: exact source-partition hashes · exact day-artifact manifests ·
                    schema era · access policy — Amendment P0-A; reuses hash_allowlisted_source_files)
   └─ CoreStrategyReplayIdentity  (input bundle · QL replay source identity · SC source identity ·
                                   resolved section hash · canonical profile id · seed · resolver ·
                                   schema versions)
        ├─ SearchChildMembership          (parent search id · ordinal · axis value ids · role) — per study
        ├─ CoreReplayArtifactReference    (v2 dataset id · gross trade-stream hash)
        ├─ FsmAuditArtifactIdentity / ReplayChartArtifactIdentity   (companions, separately versioned)
        ├─ CostedEvaluationIdentity       (× cost policy → net metrics, costed records)
        │     └─ AccountSimulationIdentity / PortfolioSimulationIdentity (× per-account or per-leg firm · risk · withdrawal · replacement ·
        │                                  trade-path BUNDLE id + manifest + capabilities + scenario policy ·
        │                                  clock policy · mode · bootstrap/stress · seed · paths)
        └─ FeatureView/LabelView/FoldSet/ModelFit/PredictionSet/RegimeFit identities (derived layers)

PipelineSemanticIdentity  (research-bearing spec)  ×  ExecutionAttemptIdentity  (workers/memory/attempt/host)
StudyCellSemanticPayload  (16 semantic dimensions — engineering protocol excluded [P0-B])
                          × StudyCellAnnotation (runtime/storage/display — never hashed)
```

Universal conventions (Amendment P0-C + V3 patch): every ID-producing contract uses the non-self-referential **Payload/Envelope** pattern (now **fully applied** — Comparison, DeclaredContrast, FeatureBlock, and FeatureBundle converted, with explicit **registry-key vs resolved-identity** pairs like `feature_block_key`/`resolved_feature_block_id`, V3 P0-1/P1-2); no payload contains its own id, artifact/manifest hashes, display metadata, annotations, or attempt metadata; artifact content hashes are post-materialization envelope facts; identity-bearing contracts obey the **deep-immutability rules** (canonical tuples / immutable-mapping wrappers / deep-copy + revalidation, with mutation-adversarial tests — V3 P1-1). The replay-input bundle carries **exact physical partition keys** (`source_partition_id`, UTC date, manifest id, logical key — one trading day spans multiple partitions, V3 P0-4), and its identity contains the **deterministic preflight** `ReplayAccessAuthorizationRef` while the **runtime** `ReplayExecutionAccessAudit` lives with execution attempts, never in replay identity (V3 P0-5). Generated child profiles are gated by `GeneratedProfileCapability` (P0-D) — the fixed `PROFILE_CAPABILITY_REGISTRY` gates *baselines* only. Strategy-only cells use the concrete optional-field `DataLineagePayload` (V3 P0-6) — no context/formula identity is carried unless a context artifact is actually part of the computation. Every store follows the manifest protocol; the mutable catalog stays lock-guarded and rebuildable; dirty-tree status folds into identities; release verification requires clean trees.

## 6. Job and orchestration design (brief §18 item 8)

Self-contained (V3 P0-2). Parent state machine (`charter_frozen → children_enumerated → artifacts_prewarmed → replays → underlying_edge_passed → prop_simulations → prop_feasible → robustness_passed → frontier_complete → search_complete`, plus failed/cancelled) with child states `queued/running/completed/failed/reused/cancelled_at_safe_boundary/blocked`. Mechanics cloned from the verified repo patterns: O_EXCL lock per search; atomic JSON checkpoints (`_write_json_atomic` idiom) after every child transition and at date boundaries within a child; a `cancel.requested` sentinel honored at safe boundaries only; idempotent resume via store-identity reuse; `ProcessPoolExecutor(max_workers=4, max_tasks_per_child=1, spawn)` with **one process per child** (days are seed-chained inside a child); a prewarm phase grouping children by `artifacts_tag()` and sequentially rebuilding any missing TF-set artifact chain before that group's replays; the UI never runs replays in-process — launch is a detached CLI shim writing the same status files the monitor reads. Revision-driven behaviors: enumeration yields (core replay, membership) pairs deduped on `core_replay_id` within and **across** studies; per-child audit builds persist a `ChildAuditNeutralityReport` (single-drive mechanism proof by default; dual-drive A/B on the baseline slice; the doc-default parity gate untouched); pipeline retries reuse the same `PipelineSemanticIdentity` with a new `ExecutionAttemptIdentity` — resource changes never mint a new scientific result.

## 7. Prop-account simulation design (brief §18 item 9; revisions P0-12…P0-17, P0-21)

Fidelity-first, **historically truthful** (`CONTRACTS_AND_SCHEMAS.md` §5; Amendment P0-H): the fidelity classes are `closed_trade_only` → `ohlc_1m_unordered` (honest raw evidence — a 1-minute bar does not record its high/low order, and `observed_intrabar_order` can only be `"unknown"`) → `assumed_1m_intrabar_path` (a **scenario** generated by a registered intrabar policy such as `bar_adverse_extreme_first_v1`; two policies = two scenario identities) → `ordered_mbp1_event_path` / `ordered_fill_event_path` (actual chronology; planned sources). Simulation modes: `historical_closed_trade`, `historical_1m_scenario`, `historical_ordered_event_replay` (reserved for actual ordered streams only), `day_block_bootstrap`, `stress` — assumed 1m paths are always labeled scenario/approximation, never "exact historical". **Every firm rule declares required path capabilities and accepted fidelity classes** — extrema-order-insensitive rules may accept unordered OHLC; chronology-sensitive rules require actual market-price chronology; simulations fail closed with a `PathCapabilityReport` when required capabilities are absent. Account events are wrapped in a totally-ordered `PropAccountEventEnvelope` with exact lineage. All time rules use typed `DayCountBasis`/`DurationRule` under a `SimulatedClockPolicy` (unrepresentable rules fail closed). The firm contract states what is *permitted*; a separate `WithdrawalPolicyPayload/Envelope` states what the trader *chooses* — both enter the applicable `AccountSimulationPayload` or per-leg `PortfolioSimulationPayload` together with the exact trade-path bundle, replacement, clock, mode, bootstrap/stress, seed, and paths (no result-changing constructor-only arguments). Contract verification uses the explicit status ladder (`synthetic_fixture_verified` → `first_party_evidence_compiled` → `owner_reviewed` → `first_party_verified` → `superseded`; Amendment P1-A) — synthetic bundles prove the machinery but can never verify a real contract; real research/prop use requires `first_party_verified` with no override. Bootstrap draws carry unique `path_instance_id`s; duplicate sequences are legitimate; copied accounts share one sampled sequence per path.

## 8. Search and robustness protocol (brief §18 item 10; revisions P1-6, P0-9)

`deterministic_exhaustive_v1`; hard feasibility gates → deterministic non-dominated sort → lexicographic tie-breaks with a persisted trace; the resolved policy inside the charter identity. **The selected interior-of-plateau configuration is a `Development Exploratory Representative`** — a *publishable* representative requires a separately frozen, owner-approved outer evaluation protocol (nested walk-forward or equivalent; schema-reserved, deferred per owner decision 15). Cross-profile population/funnel/sequence deltas run on the lineage layer with a mandatory `match_basis`; comparisons without an exact basis are disabled, never fuzzed.

## 9. Reporting and deterministic-insight design (brief §18 item 11)

Unchanged in shape (JSON+MD paired reports, typed categories, exact `EvidenceRef`s, causally-neutral wording), with insights now citing `match_basis` and the exploratory/publishable distinction, and prop insights citing the path capabilities, accepted fidelity class, and scenario status of the underlying simulation.

## 10. Reuse-versus-rewrite decisions (brief §18 item 13)

Self-contained (V3 P0-2):

| Existing asset | Decision |
|---|---|
| `manifest.py` save protocol + `artifact_io` loaders + `context_run_store` atomic publish | **Reuse** (clone into the search stores) |
| `context_experiment_contracts` idioms (`canonical_contract_sha256`, frozen pydantic, capability registry, reconciliation gate) | **Reuse/generalize** (charter, dimension registry, comparison compatibility) |
| `resolve_profile_config` + the `run_ifvg_tf_variant.py` flow | **Reuse** (child enumeration + replay worker) |
| `preparation.py` job/lock/cancel/checkpoint machinery; the `w3_cache_warmer` pool pattern | **Reuse** (orchestrator mechanics) |
| `development_access.py` policy/audit | **Reuse + extend** (the new `VerificationReplayPolicy` third trusted class) |
| SC audit channel + the QL `fsm_audit_*` modules | **Reuse + extend** (per-child neutrality-aware build; doc-default parity gate untouched) |
| `compute_trade_stats`, `binary_prediction_report`, `block_bootstrap_interval`, `paired_tier_delta_report` | **Reuse** (gates, predictive/economic deltas) |
| `alpha_lab.propsim` engine/bootstrap/presets/report | **Reuse untouched + extend** (new modules alongside; `AccountWalk` gains the envelope/fidelity/calendar/withdrawal contracts **before** implementation — fidelity-first) |
| `scratch_ifvg_search.py` runner | **Not reused for FSM deltas** (violates brief §9.5.9); only its ledger/budget/lock idioms imitated |
| Legacy `run_ifvg_experiment` / `run_sealed_validation` | **Untouched** (legacy-lane hard refusals stand) |
| Verifier drill-down bus, 3-pane charts, `LAYER_BUDGETS`, AppTest harness, source-scan contract tests | **Reuse + extend** (`setup_id` jump kind; new-module scans) |
| Orphaned `ifvg_lab_charts` builders (equity/R-hist/calibration/coverage) | **Adopt** into the results views |
| The mutable JSON catalog pattern (`update_context_run_catalog`) | **Not reused as-is** (no locking — verified); replaced by the lock-guarded append-only event log + rebuildable index |
| The former single `children` store concept | **Re-keyed** into `core_replays` / `memberships` / `costed_evaluations` |
| `dashboard-ui/` React remnants | **Not revived** |
| M0–M3 tier registry | **Frozen as bundles**; never edited |

## 11. Migration and catalog strategy (brief §18 item 14)

Unchanged: purely additive; no existing artifact/catalog is migrated; M0–M3 identities stay valid; rollback = delete the new directories. The new catalog index is rebuildable from manifests + its event log at any time.

## 12. Access-safety design (brief §18 items 15, 27, 29)

Boundary and mechanisms unchanged (authorize-before-path, no listing, event-chain audits, zero counters, no sealed controls, sanitized errors).

**Five-day policy (item 27, revised + Amendment P0-E/P0-F):** one frozen ≤5-day allowlist whose dates are a **coverage-evidenced owner decision** (`2026-06-04…06-10` is the candidate pending that evidence); the real budget funds **one baseline vertical slice with its exact profile-matching seed** (seeds are profile-bound — verified); multi-child behavior is synthetic; verification applies only the nonresearch control-flow gate policy; every verification report stamps `verification_only=true`, `not_for_research_interpretation=true`, `full_pipeline_not_run=true`; profile/seed mismatches are refused before any source read. **The real slice runs under a real `VerificationAuthorizationRef`** (approved allowlist hash + coverage-matrix artifact + seed-snapshot ref + approver) — the synthetic authorization marker is confined to fully synthetic fixtures. Its approval is **BLOCKING-VERIFICATION**: it never blocks code authoring, but it blocks Release-1 acceptance, and Release 1 gates every subsequent release.

**Owner authorization is computation-path-scoped (Amendment P0-E):** `derive_authorization_requirements(run_scope, dimensions, computation_path, stages, firms)` yields exactly the decision set a run actually needs — a strategy-only search requires no firm/withdrawal/prop-gate/regime decisions; a prop benchmark adds the firm/fidelity/risk/withdrawal/clock/prop-gate set; a regime-descriptive study requires no `RejectedCandidatePolicy`; a charter fails only on decisions its actual path requires.

**MBP-1 confirmation (item 29, revised per Amendment P0-G + V3 P0-3):** the truthful, precisely scoped boundary is: **MBP-10 is unavailable to the new order-flow feature, model, UI, and live-streaming contracts; legacy replay provenance may reference historical source eras without exposing deeper-book features.** Concretely: every feature/bundle/control/registry namespace remains guarded by the MBP-10 identifier regex and Literal-typed contracts, while `ReplaySourcePartitionRef.source_kind` admits the single opaque provenance literal `legacy_verified_replay_source` (the development window's early eras genuinely decode from mbp10 partitions — verified) which is unqueryable by the feature layer, bundle materialization, the dashboard, live-source contracts, and model features (guard tests each). Ordering uses the complete four-part key `(ts_event, ts_recv, sequence, source_ordinal)`; **stage cutoffs are `StageEvidenceCutoff` objects** — the exact stage-triggering event's key when available (never an artificial `+inf` bound; the comparator follows each feature's declared `WindowTriggerSemantics`, V3 P1-4), strict `ts_event < stage_ts` for timestamp-only stages (all same-timestamp events excluded or typed `same_timestamp_order_unavailable`), versioned completed-bar boundaries; windows are never widened. Feature delivery: R5 ships contracts/readiness only; **R5B activates the offline research-only block** (owner P1-D ruling; promotion boundary decision R-6 — no live, serving, or gate use without a later Strategy-Core formula/parity contract and a separately approved sequential model-gated replay).

## 13. Full-pipeline contract summary (brief §18 item 28; revision P0-3)

The 16-stage plan, stage states, estimates, checkpoints, scope guards, and verify-then-activate publication stand — now split into `PipelineSemanticIdentity` (all research-bearing fields) and `ExecutionAttemptIdentity` (workers/memory/attempt/host/timestamps/retry reason). Stage and result identities key on the semantic id; operational retries and resource changes reuse verified stages under the same scientific identity. S11 remains `BLOCKED` with the exact reason: *"model-gated execution is unavailable until RejectedCandidatePolicy is owner-ratified and sequential golden tests pass."*

## 14. Performance and capacity plan (brief §18 item 16)

`TEST_MATRIX.md` §4 (revised rows: real vertical slice ≤5 min dual-drive; synthetic E2E ≤2 min; catalog rebuild <5 s). Operator-scale estimates unchanged and never enter verification identities.

## 15. Phased implementation sequence (brief §18 item 18; revision P1-7/8/9)

`PHASED_DELIVERY.md` (revised): **usable releases** — R1 contracts/identities/stores **+ the real baseline vertical slice** (authoring may proceed speculatively; **acceptance** blocked on the owner's fixture authorization, and dependent releases may be authored in branches but not declared complete/accepted/activated until R1 passes — V3 P0-8) → R2 multi-child search + lineage + deltas + verifier → R3 fidelity contracts first, then the **synthetic** contract/lifecycle verification end-to-end (an optional first-party contract gate is blocked by owner decisions 5/6) → R4 trader UI → **R5 — Pipeline Runner, MBP-1 Contract Readiness, and Supervised Model Ladder** (no active MBP-1 bundle) → **R5B — Offline MBP-1 Feature Activation** (mandatory; the owner's 13 deliverables; research-only boundary) → **R6-core** KMeans regime lane → **V1 hardening**. **The GMM/minibatch/spectral-diagnostics/Nyström expansion is post-V1** (V3 P1-6). **Operator-pipeline availability is capability-gated by the selected stage plan** (V3 P1-5): strategy-only pipelines need only their own dependencies — they never wait for MBP-1 activation, regime fitting, or spectral diagnostics. Strategy-Core changes in v1: **none**.

## 16. Acceptance-gate mapping (brief §18 item 19)

`TEST_MATRIX.md` §5 (revised) maps every brief §16 gate **and** every revision §5 acceptance criterion (items 1–35) to a designed test or proof.

## 17. Risks and mitigations (brief §18 item 21; revised)

| Risk | Mitigation |
|---|---|
| Capture driver cannot start mid-chain from a cached seed | R0→R1 verification item; fallback `start_after_artifact` param (QL-only) |
| Native IDs not perfectly content-derived across replays | determinism parity test gates native-basis deltas |
| Lineage keys unstable across profiles (fvg-id drift) | lineage-validity test gates cross-profile deltas; failing populations are `not_comparable`, never fuzzed |
| Canonical-naming collision with existing profile names | canonical ids are prefixed (`ifvg_search_profile_`) and derived from content hashes; baseline names untouched |
| Audit channel not neutral for some child config | per-child `ChildAuditNeutralityReport`; dual-drive A/B available; failure = `INVARIANT_FAILED`, audit unpublished |
| Path fidelity insufficient for a firm rule | rule-level minimum-fidelity declarations + fail-closed `PathCapabilityReport`; conservative approximation clearly labeled |
| Clock semantics unrepresentable under bootstrap | `SimulatedClockPolicy` + fail-closed unsupported rules |
| Firm-rule fidelity to real contracts | field-level evidence compiler; publishable results prohibit unverified contracts |
| Catalog write loss under concurrent publishers | lock-guarded append-only event log + deterministic rebuild + crash-recovery tests |
| Dirty trees churn identities | visible in identities; clean-tree release gate |
| Streamlit API drift / scripts lint fallout | pin bump + version smokes + prep lint commit |
| Scope creep into model-gated searches | S11 blocked on `RejectedCandidatePolicy` ratification + golden tests; policies only *emit* requests |
| Exploratory results read as publishable | `Development Exploratory Representative` wording + forbidden-wording scan + publication gates |
| Searching an inert/hardcoded/coarse axis | typed value registry blocks inert, hardcoded, and `blocked_pending_owner_policy_review` values fail-closed |
| Storage growth (immutable keep-all) | per-child alarm; retention revisited under owner decision 17 |
| Input-bundle hashing cost on large date sets | reuses already-computed source hashes (`hash_allowlisted_source_files`) + existing day-artifact manifests — no duplicate hashing; bundle assembly is metadata-only (Amendment P0-A) |
| Same-timestamp cutoff ambiguity degrades MBP-1 coverage | by design: ambiguous rows carry typed `same_timestamp_order_unavailable` rather than leaking future evidence; coverage reports make the cost visible (Amendment P0-G) |
| Scenario paths misread as history downstream | path-capability + scenario-policy IDs ride inside account/portfolio simulation identities and every report; the "exact historical" source-scan blocks regressions (Amendment P0-H) |
| Candidate-stage regime samples too sparse even for KMeans | the `CONTEXT_BAR_PANEL` grain is in the actual schema; sample-adequacy gates block promotion rather than silently shrinking k (Amendment P1-B) |

## 18. Expected files to change (brief §18 item 22; revised)

**New:** ~58 source/test modules per `PHASED_DELIVERY.md` (search/ ×17 incl. `identities.py` [input bundle + envelopes + generated-profile capability], `authorization.py` [scoped requirements + VerificationAuthorizationRef], `lineage.py`; study/ ×10 incl. `cohort.py`; features/ ×11 incl. the six R5B MBP-1 modules (`mbp1_source_artifact.py`, `mbp1_arrow_schemas.py`, `mbp1_stage_windows.py`, `mbp1_feature_materializer.py`, `mbp1_coverage.py`, `mbp1_feature_join.py`); ml/ ×14 + fixtures; propsim/ ×12 incl. `trade_path.py`, `calendar.py`, `withdrawal.py`, `contract_evidence.py`; scripts/ ×10; src study_status/study_presentation; ~24 test modules).
**Modified (complete list):** `development_access.py`, `data_access.py`, `dataset.py`, `fsm_audit_preparation.py`, `replay_chart_provider.py`, `scripts/ifvg_lab_tab.py`, `scripts/ifvg_verifier_tab.py`, `tests/agents/test_ifvg_verifier_tab.py` (+ possibly `test_ifvg_lab_tab.py`), `pyproject.toml`, `.github/workflows/ci.yml`, `docs/DECISIONS.md`, `ARCHITECTURE.md`, `docs/README.md`, `docs/pipeline_state.yaml`.
**Never modified:** all M0–M3 lane modules, all existing propsim modules, all of Strategy-Core, all existing immutable artifacts and catalogs.

## 18.1 Final contract-closure decisions

The final authoritative package additionally fixes these implementation seams:

- `FeatureBlockDefinition` is separate from `FeatureBlockResolutionPayload/Envelope`; exact MBP-1 window definitions enter the resolved block identity.
- `DecisionPolicyPayload/Envelope` and `WalkForwardModelSchedulePayload/Envelope` replace all self-ID/stale policy forms.
- `VerificationRunPayload/Envelope` binds `VerificationAuthorizationRef` to the exact pipeline, allowlist, seed, profile, and coverage matrix before source-path construction.
- `TradePathBundleEnvelope` identifies the complete strategy path evidence; every path event has an exact ID.
- Account and portfolio simulations have separate identities. Each account policy set freezes firm/risk/withdrawal/replacement/clock semantics, and every portfolio leg references one exact policy-set ID.
- Prop rule support is capability-based, not inferred from an ordinal fidelity enum.
- Numerical regime fits are role-free; assignments and assessments reference exact fit/protocol/fold identities, while promotion is a separate owner-ratified decision.

## 19. Confirmations (brief §18 items 29, 35; revision §1.3.5)

- **MBP-1 is the maximum supported order-flow depth for every new contract** — MBP-10 is unavailable to the order-flow feature, model, UI, and live-streaming contracts (identifier-guarded), with legacy replay provenance scoped as opaque and unqueryable (V3 P0-3); ordering uses the complete four-part key.
- **No code, tests, artifacts, catalogs, seed snapshots, replays, MBP-1 feature builds, model fits, prop simulations, or source-data runs occurred** during planning, revision, or the V2 amendment. This revised document set is the entire output. Implementation begins only after approval, proceeds release-by-release, and stops at the hardening gate — the operator full run remains a separate explicit action, gated on the computation-path-scoped authorization requirements its stage plan actually exercises.
