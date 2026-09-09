# Test Matrix — Verification Fixtures, Acceptance Gates, Performance Budgets

**Document type:** Supporting document to `IMPLEMENTATION_PLAN.md` (brief §18 items 17, 27; brief §12, §16)
**Status:** Plan-only. **REVISED** per the first revision (P0-6/7/8/21, §4.5), Amendment V2 (§5 — §3.8 below), **the V3 compact patch (§3.9 below), and the V5 frontend-authority restoration (§3.11 below)**. See `REVISION_CHANGELOG.md` and `FINAL_CONSISTENCY_AUDIT.md`. No new real dates are added by any amendment or patch test.
**Binding rule (brief §1.1B, §16.14):** all real-data verification across the complete implementation-verification program uses **one frozen allowlist of at most five authorized real trading days** (warmup + evidence ≤ 5); everything else is synthetic. No implementation gate runs the full development pipeline.

---

## 1. The two-path verification design (P0-6)

**Verified constraint driving the split:** Strategy-Core seeds are profile-bound — `DayOrchestrator` raises when `seed.profile_hash` differs from the active section's hash (SC `replay.py:362`). One baseline seed snapshot therefore cannot warm-start children with changed section hashes, so real multi-child verification is impossible without per-child seeds (which must never be produced by secretly replaying the full development history during verification).

### Path A — real five-day vertical slice (the only real-data verification)

| Property | Value |
|---|---|
| Profile | **one exact baseline**: `ifvg_v2_doc_default_fresh_static_1r` (unchanged section hash) |
| Seed | the one **profile-matching** precomputed verified seed snapshot (doc-default artifact chain through the day before the allowlist start) |
| Allowlist | the one frozen ≤5-trading-day list, hash-pinned (`VERIFICATION_ALLOWLIST_V1`; date selection is a coverage-evidenced owner decision — §2 below) |
| warmup + evidence | 0 real warmup days + ≤5 evidence days ≤ 5 ✔ |
| Access policy | `VerificationReplayPolicy` — authorize-before-path, no listing, event-chain audit, zero protected/sealed |
| Authorization | a **real `VerificationAuthorizationRef`** (approved allowlist hash + coverage-matrix artifact + seed-snapshot ref + approver) — the synthetic marker is refused for this path (Amendment P0-E); its approval is **BLOCKING-VERIFICATION**: it gates Release-1 acceptance, not code authoring (Amendment P0-F) |
| Replay | **one complete sequential replay**, run in **dual-drive mode** (audit-disabled and audit-enabled) |
| Proves | source access · mid-chain start from the seed snapshot · seed compatibility · sequential replay · dual-drive `ChildAuditNeutralityReport` (core-table hash equality) · audit/chart companion builds · immutable save → reload → reuse · exact verifier linkage · five-day access safety |
| Gates applied | `verification_control_flow_gates_v1` **only** (P0-6): replay completed, invariants held, artifacts published + reloaded, neutrality passed, verifier link resolves, counters zero. **Research gates (≥30 trades, ≥20 days) are never applied to verification fixtures** — they are mathematically unreachable in five days |
| Output namespace | `data/ifvg_datasets/search_test/v1/` |
| Report stamps | `verification_only=true` · `not_for_research_interpretation=true` · `real_date_count` · `real_date_allowlist_hash` · `synthetic_fixture_ids` · `full_pipeline_not_run=true` |

### Path B — synthetic multi-child study (no real data)

Deterministic synthetic fixtures drive everything multi-child: 2×2 child enumeration (generic synthetic axes, or a timeout-only real axis plus a synthetic second dimension — **never** implying owner authorization of parent-fill changes, P0-20), parent/child orchestration, replay-result reuse (including cross-parent-study core-replay reuse), strategy-gate pass/fail branches, the 16 prop simulations, frontier, insights, and cancel/resume/lock behavior. Synthetic child results are **never claimed to prove real strategy behavior** — they prove orchestration control flow.

Negative tests (fail **before path construction**): a 6th real date; any off-allowlist date; a rotated window; a research-catalog destination; a second, different allowlist anywhere in the same implementation-verification program (marker-file check); **a profile/seed mismatch** (a child section hash presented with the baseline seed must be refused by the policy layer before any source read).

## 2. Fixture-date selection (P0-7)

The five dates are an **owner decision selected by coverage, not recency**. Selection procedure (no new source reads — evidence comes only from already-authorized artifacts: the accepted v2/fsm-audit datasets and replay-chart coverage): build a coverage matrix over candidate ≤5-day windows scoring — source partition availability; setup activation; parent candidate/lock; opposing/inversion; entry candidate; execution/resolution; audit events; replay-chart coverage; MBP-1 source coverage when that lane is being verified. `2026-06-04…06-10` is the **proposed candidate pending that coverage evidence** (`OWNER_DECISIONS.md` item 21). If no single window covers all paths, the fixed real fixture covers source/replay integration and synthetic fixtures cover the missing branches. **One canonical real-data verification allowlist is used across the complete implementation-verification program — every release reuses that same allowlist** (V3 P0-7); changing it requires a new verification-policy version and cannot be used to accumulate real-data coverage for the same implementation acceptance.

---

## 3. Test inventory by layer

### 3.1 Identity, contracts, registries (synthetic; `tests/agents/ifvg_search/`)

| Test | Asserts |
|---|---|
| **Core-replay reuse across parent studies** (P0-1) | the same resolved configuration in two different charters yields one `core_replay_id`, one stored replay, two `SearchChildMembership` rows; the second study's child is `REUSED` with zero replay invocations |
| **Core-replay identity independence** (P0-1) | changing cost policy, risk policy, prop contract, withdrawal policy, audit schema version, replay-chart schema version, or worker/resource policy leaves `core_replay_id` unchanged |
| **Core-replay identity sensitivity** (P0-1) | changing the section (any registered axis value), date set, seed identity, source identities, resolver semantics, or Strategy-Core commit changes `core_replay_id` |
| **Membership separation** (P0-1) | parent id, child ordinal, and display name changes touch only `SearchChildMembership` / the catalog — never the core identity |
| **Companion-artifact versioning** (P0-1) | a changed audit or replay-chart schema creates new companion identities with the same `core_replay_id` |
| **Stable semantic profile naming** (P0-2) | `canonical_profile_id` is identical for the same overrides across two studies; two different names for identical semantics are impossible; the name-free hash excludes `profile_name`; resulting `section_config_hash` (and hence record IDs) are study-independent |
| **Execution-attempt identity** (P0-3) | two pipeline attempts with different worker counts share `pipeline_semantic_id` and produce identical semantic stage/result identities when outputs are byte-identical; attempts carry distinct `ExecutionAttemptIdentity` |
| **Owner-authorization fail-closed** (P0-4) | a real charter with an absent, stale, superseded, or charter-inconsistent `OwnerDecisionEvidenceRef` refuses to freeze; synthetic charters require the typed synthetic marker and are namespace-confined |
| **Typed axis values** (P0-5) | `None`-timeout and timeframe-tuple payloads round-trip; unregistered values refused; a value with `owner_ratification_status="pending"` refused in a real charter but accepted under the synthetic marker; `CompositeAxisValue` dependency groups resolve atomically; UI provider surfaces registered values only |
| Axis-registry fail-closed | inert fields (`break_even_enabled`, `legacy_candidate_row_limit`, `parent_reaction_window_1m_bars_max`), reducer-hardcoded axes, and **parent-fill values (`blocked_pending_owner_policy_review`, P0-20)** refused before enumeration |
| Charter/identity determinism | same charter → same `search_id`; enumeration deterministic + deduped on `core_replay_id` (§16.9) |
| Dimension registry / computation path / comparison compatibility / feature-block partition / MBP-10 guard / bundle refusal | unknown/blocked/incompatible dimensions fail `StudyCellSemanticPayload` validation; `derive_computation_path` reproduces the brief §7A.5 table row-by-row; declared-class vs changed-set mismatches refused, failed equalities → `config_diff_only`; the available blocks partition `TIER_FEATURE_REGISTRY[M3]` feature-for-feature with frozen tier bundles order-exact; no identifier matches the MBP-10 regex (legacy-provenance literal exempt per CS §1.1); `resolve_bundle` refuses non-AVAILABLE blocks with status + reason |
| **Planned-block version bump** (P1-5; V3 key/resolved form) | simulated activation increments `block_version`, **mints the first `resolved_feature_block_id`**, changes the block-registry hash, and gives every dependent bundle a new `resolved_feature_bundle_id`; the logical `feature_block_key` and platform contracts remain untouched |
| Store protocol + catalog separation | save→reload→assert, overwrite refusal, exact-ID resolution, reuse (§16.9) |
| **Concurrent catalog publishing** (P0-18) | N concurrent writers appending catalog events lose nothing (lock contention test); a torn final line is recovered on rebuild; `rebuild_catalog_index(events, manifests)` is deterministic and reproduces the read model from scratch |

### 3.2 Lineage (P0-9; `tests/agents/ifvg_search/test_lineage.py`)

| Test | Asserts |
|---|---|
| Native determinism | same config replayed twice (synthetic day fixture) → identical native IDs, Jaccard 1.0 per entity kind |
| Lineage validity | across two profiles sharing the TF set + `min_gap_ticks_capture` (synthetic bars), shared FVGs produce identical lineage keys |
| Match basis | same-profile comparisons report `native_id_exact`; cross-profile report `profile_independent_lineage_exact`; an axis that changes gap detection forces `not_comparable` with a reason — no fuzzy fallback path exists (source-scan: no nearest-time/keep-last matching code) |
| Delta integration | population deltas carry `match_basis`; insights referencing commonality are suppressed for `not_comparable` |

### 3.3 Prop engine (brief §16.4 + revisions P0-12…P0-17)

Rule-by-rule fixtures — one deterministic hand-built stream per rule (trailing styles ×3, static drawdown, DLL hard/soft, payout eligibility/cap/split, all three post-payout threshold rules, evaluation/activation/recurring/reset fees, breach with linked trade, replacement with ordinal + reset fee, min days, winning-day requirements, consistency denominator, contract limits + micro-scaling, exact same-day event ordering, `AccountWalk`≡`EvaluationWalk` one-contract parity, every risk-family sizing + skip reasons, adapter tick→point/cost mapping + stable order-sensitive stream hash) **plus**:

| Test | Asserts |
|---|---|
| **Path-fidelity capability** (P0-12 + Amendment P0-H) | each firm rule declares `required_path_capabilities` plus `accepted_fidelity_classes`; support is never inferred from enum ordering. A chronology-sensitive rule refuses evidence without market-price chronology while an extrema-order-insensitive rule may accept `OHLC_1M_UNORDERED` |
| **1m scenario truthfulness** (Amendment P0-H) | a 1m OHLC record cannot claim observed high/low ordering (`observed_intrabar_order == "unknown"` is the only representable value); two assumed intrabar policies (`bar_adverse_extreme_first_v1` vs `bar_favorable_extreme_first_v1`) produce two separate scenario identities and results on a hand-built trade where the order matters; UI/report wording for assumed paths is `scenario`/`approximation` — the literal "exact historical" never appears for them (source-scan) |
| **Trade-path artifact identity** (Amendment P0-H) | changing any per-trade path artifact changes `trade_path_bundle_id`, which changes the account/portfolio simulation identity; bundle manifest, capabilities, and scenario policy are pinned |
| **Event-envelope total order** (P0-13) | a day with DLL touch + floor ratchet + payout eligibility + fee produces a strictly ordered envelope stream under `event_order_policy_id`; same-timestamp ties resolve deterministically; every event carries its source trade/decision/candidate/setup/path-event links |
| **DayCountBasis** (P0-14) | trading-day vs winning-day vs calendar bases advance correctly through a fixture with weekends/halts; a calendar-month recurring fee under day-block bootstrap without a synthetic calendar is `unsupported` and fails closed; the bootstrap clock advancement rules are exercised per basis |
| **Withdrawal-policy identity** (P0-15/16) | two `AccountPolicySetEnvelope`s differing only in `WithdrawalPolicyPayload/Envelope` produce different IDs and therefore different account/portfolio simulation IDs and different payout streams; the firm contract alone does not determine trader behavior |
| **Replacement-policy identity** (P0-16) | `replacement_policy`/`max_replacements` enter `AccountPolicySetPayload`; every portfolio leg references one exact policy-set ID; a constructor-surface audit test diffs `run_simulation`'s accepted kwargs against identity fields (no result-changing constructor-only argument) |
| **Contract compilation + status ladder** (P0-17 + Amendment P1-A) | a synthetic source-document set compiles to `synthetic_fixture_verified` (proving compiler behavior, conflict handling, field provenance, engine integration) — and **cannot** reach `first_party_verified`; an unresolved conflict blocks progression; supersession retires the old contract id; real-namespace research simulation requires `first_party_verified` (compiled from first-party evidence + owner-reviewed) with no override |

### 3.4 Simulation acceptance (brief §16.5; revision P0-21)

The simulation suite explicitly covers: semantic seed identity; closed-trade, 1m-scenario, and ordered-event modes; whole-day/block resampling; deterministic stress; one common correlated path across copied accounts; aggregate stability; lower-tail payout and breach metrics; deterministic frontier/dominance/tie-break traces; and neighboring-configuration stability. The duplicate-path semantics are:

| Test | Asserts |
|---|---|
| **Bootstrap path semantics** (P0-21) | duplicate sampled index sequences across draws are **accepted** (a seeded fixture that provably draws a repeat passes); every draw carries a unique `path_instance_id` and a stored `sampled_index_sequence_hash`; same spec → identical draw sequence; copied accounts share one sampled sequence per path; **no duplicate OOS prediction rows** is enforced at the prediction layer (the actual meaning of brief §16.5's "no duplicate OOS path") |

### 3.5 Orchestrator (brief §16.3 — now fully synthetic, Path B)

The 2×2 → 4 children → 16 simulations acceptance runs on synthetic fixtures at the `run_child_replay` seam: exactly 4 enumerated (deduped), 4 replay invocations or `REUSED`, 16 sims after gates, gate-rejected children skip sims with explanations, resume-after-kill, repeat-identical, overwrite refusal, failure reasons, cancel sentinel, lock contention — plus cross-study reuse (§3.1 row 1). Axes for this fixture are generic synthetic axes (or timeout-only real + synthetic second), per P0-20.

### 3.6 ML fixtures (revised scope)

Fixtures 1 (supervised), 2 (known clusters), 4-KMeans ship in V1; fixture 3 (two-ring) + GMM/Nyström arms ship with the regime-expansion release (`ML_REGIME_CONTRACT_PLAN.md` §9, §11). Added: the **portable-artifact relocation test** (P1-4 — copy artifact dir, reload by manifest-relative ref + checksums, re-transform, allclose) and the **`CohortSpec` interpretation tests** (P0-10 — each `InterpretationMode` maps to the right comparison class; `sequential_strategy_profile` constructs a child profile rather than filtering; descriptive cohorts never select execution/prop delta families). `RejectedCandidatePolicy` tests (P0-11): every registered decision policy requires a rejected-candidate policy; real use without ratification refused; S11 surfaces the exact blocked-reason text.

### 3.7 Access safety and UI

Access/UI acceptance includes zero protected/sealed counters; fail-before-path construction; no sealed controls; sanitized errors; blocked-profile states; automatic full-pipeline invocation guards; the complete `FRONTEND_UX_CONTRACT.md` AppTest/browser matrix; keyboard and viewport QA; and **exploratory-representative wording test** (P1-6 — the strings "robust representative"/"publishable" never appear for the development lane; overview cards and reports render `Development Exploratory Representative`), and the pipeline monitor shows attempt history (semantic id stable across retries).

### 3.8 Amendment V2 additions (P0-A…P0-H, P1-A…P1-G)

Columns per the amendment: **Level** (U = unit, I = integration, E = end-to-end/AppTest), **Fixture**, **Assertion**, **Failure meaning**, **Release gate**.

| Test | Level | Fixture | Assertion | Failure meaning | Gate |
|---|---|---|---|---|---|
| ReplayInputBundle content sensitivity | U/I | synthetic partitions + day artifacts | changed partition bytes → new bundle + core replay id; changed day-artifact manifest → new id | replay reuse could serve stale/altered inputs | R1 |
| QL replay source identity sensitivity | U | scoped source-tree fixture | changed QL replay/capture source-tree hash → new core replay id | code drift invisibly reuses old results | R1 |
| Portable input-manifest identity | U | relocated fixture tree | moved files w/ identical verified content → same identity; reorder-only lists normalized or refused | path-dependent identities | R1 |
| Cross-study input reuse | I | two synthetic charters | identical verified input manifests reuse one replay | duplicate replays / split lineages | R2 |
| StudyCell annotation exclusion | U | cell + annotation fixtures | workers/storage-root/page-size/runtime-estimate changes leave `cell_id` unchanged; model package-version change moves the correct semantic identity; `annotation_only` dimensions absent from the semantic payload | engineering noise forks scientific identity | R1 |
| Identity self-field exclusion (projection audit) | U | every ID-producing contract | no payload contains its own id / artifact hash / manifest hash / display / annotation / attempt fields; payload → id → envelope → reload deterministic; no post-materialization hash defines a pre-run id | circular or ambiguous identities | R1 |
| GeneratedProfileCapability | U/I | synthetic axis registry + profiles | runnable baseline + ratified values → `generated_runnable`; blocked baseline → blocked; unratified value → blocked pre-replay; absence from the fixed registry is not an error; no legacy-launcher leakage | unauthorized or wrongly-blocked children | R1/R2 |
| Computation-path-scoped authorization | U | 10 parameterized study modes | exact requirement set per mode (strategy-only needs no firm/regime decisions; prop adds firm/fidelity/risk/withdrawal/clock; regime-descriptive needs no RejectedCandidatePolicy; gated replay needs it; verification needs `VerificationAuthorizationRef`; synthetic uses the marker) | over- or under-blocking of launches | R1 |
| Real VerificationAuthorizationRef | I | verification fixture | the real 5-day slice refuses to run with a synthetic marker; requires approved allowlist hash + coverage-matrix artifact + seed snapshot ref | real data consumed without real approval | **R1 acceptance** |
| BLOCKING-VERIFICATION classification | — | docs/process | fixture sign-off blocks R1 acceptance, not code authoring; every dependent release inherits the block | releases could "pass" unverified | R1 |
| MBP-1 same-timestamp future-event exclusion | U | crafted same-ts event bursts | exact-key cutoff admits only earlier `(ts_recv, sequence, source_ordinal)`; a same-ts after-stage event cannot change any feature | PIT leak through timestamp ties | R5B |
| Timestamp-only conservative cutoff | U | same-ts fixture | `ts_event < stage_ts` strictly; all same-ts events excluded or typed `same_timestamp_order_unavailable`; window never widened | PIT leak / silent cohort change | R5B |
| No-+inf cutoff construction | U | source scan | no `(…, +inf, +inf, +inf)` cutoff exists anywhere | the withdrawn rule resurfaces | R5B |
| 1m OHLC unordered truthfulness | U | bar fixtures | `observed_intrabar_order` is only ever `"unknown"`; assumed subevents always carry `ASSUMED_1M_INTRABAR_PATH` | scenario passed off as history | R3 |
| Assumed-path scenario identity | U | order-sensitive trade | two intrabar policies → two scenario identities + differing results | scenario ambiguity | R3 |
| Per-rule path-fidelity refusal | U/I | rule×fidelity matrix | chronology-sensitive rules refuse OHLC-only; insensitive rules accept it; fail-closed `PathCapabilityReport` | untruthful prop chronology | R3 |
| Trade-path bundle identity in simulation | U | path fixtures | changing any per-trade path artifact changes `trade_path_bundle_id`, which changes the applicable account/portfolio simulation ID | unpinned evidence | R3 |
| Synthetic-vs-first-party contract statuses | U/I | synthetic + mock first-party docs | ladder ordering enforced; synthetic can never reach `first_party_verified` | fake "verified" contracts | R3 |
| Regime context-panel schema/validation | U | 5m/15m/candidate/decision fixtures | panel fields required iff `CONTEXT_BAR_PANEL`; invalid combinations refused; PIT panel→candidate assignment | grain ambiguity / leakage | R6 |
| Planned spectral capability refusal | U | registry | planned-algorithm fit request refused with correct status/reason; training-only results cannot enter predictive bundles; identities distinct | planned code silently callable | R6 (registry checks in R1) |
| MBP-1 activation state | I | block registry | before R5B: `IFVG_ORDER_FLOW_MBP1_V1` is `planned`, bundle resolution refuses, no baseline-vs-MBP-1 study constructible; at R5B: activation = new block version + registry-hash change | untruthful availability | R5/R5B |
| Lineage one-to-one uniqueness + collision refusal | U | crafted collision fixtures | unique mapping passes; intentional collision → `not_comparable` + `LineageCollisionRecord` (no dedupe/keep-first/fuzzy); direction/family collisions prevented; same opportunity across profiles matches; entry-thesis change → not comparable | fabricated cross-profile commonality | R2 |
| Strategy-only cell without v3 | I | strategy-only cell fixture | core replay + companions + costed stream + prop sim run with `none_*` identities; feature/model studies still require exact refs; missing *required* evidence fails, irrelevant missing evidence does not | strategy studies blocked or under-specified | R2/R3 |

### 3.9 V3 compact-patch additions (P0-1…P0-9, P1-1…P1-6)

| Test | Level | Fixture | Assertion | Failure meaning | Gate |
|---|---|---|---|---|---|
| Complete Payload/Envelope conversion | U | every ID-producing contract | Comparison/DeclaredContrast/FeatureBlock/FeatureBundle (and all §0-listed pairs) hash payloads with no self-id/artifact-hash fields; registry keys and resolved identities are distinct fields | circular identities; key/version conflation | R1 |
| Self-contained contract authority | — | docs review | no authoritative schema relies on a superseded "previously specified" reference | implementer forced into superseded text | R0/R1 |
| Legacy source-kind scoping | U | replay-input fixtures | `legacy_verified_replay_source` accepted in provenance; refused by the feature layer, bundle materialization, dashboard providers, live-source contracts, and model-feature paths (one guard test per surface); mbp10-era partitions representable in provenance only | deeper-book leakage or untruthful provenance | R1 (guards re-run at R5B) |
| Physical partition identity | U | two partitions, one trading day | distinct `source_partition_id`/UTC-date/logical-key rows never collide; canonical ordering includes the partition keys; identical days with swapped partition content produce different bundle ids | partition ambiguity defeats content addressing | R1 |
| Access-audit separation | U | attempt fixtures | `ReplayAccessAuthorizationRef` (preflight) participates in bundle identity; `ReplayExecutionAccessAudit` (runtime) never does — two attempts with different runtime audit chains share one `core_replay_id` when outputs are byte-identical | runtime noise forks replay identity | R1 |
| Strategy-only DataLineagePayload | U | strategy-only cell | context/feature/label/model/regime fields are `None`; no `ifvg_context_formula_v2` identity carried; v2+v3 studies still populate exact refs | misleading formula lineage on strategy-only studies | R2 |
| Canonical allowlist across program | I | verification marker | a second, different allowlist within the same implementation-verification program is refused regardless of release; the marker persists program-wide, not per-release | cumulative dataset walk via per-release windows | R1→all |
| Authoring-vs-acceptance gating | — | process/docs | dependent-release branches may exist before R1 acceptance; none can be marked accepted/merged/activated until R1 passes | unverified releases masquerade as complete | all |
| Deep-immutability (mutation-adversarial) | U | every identity-bearing contract with nested state | mutating nested dicts/mappings/lists after construction cannot change any identity-bearing payload; frozen result/report objects also expose only immutable wrappers or defensive copies | silent post-hash mutation | R1 |
| Regime protocol/fit/promotion separation | U | regime fixtures | `RegimePromotionDecision` status changes leave `resolved_regime_protocol_id` and every `regime_fit_id` unchanged; role/status absent from protocol/fit payloads | promotion rewrites numerical identity | R6-core |
| Window-trigger semantics | U | same-ts event fixtures | `PRE_TRIGGER_EXCLUSIVE` excludes the trigger event (`<`); `POST_TRIGGER_INCLUSIVE` includes exactly it (`<=`); transition windows honor their declared interval bounds; each MBP-1 feature declares its semantics | wrong-window features (PIT or definition errors) | R5B |
| Capability-gated operator readiness | I | pipeline stage plans | a strategy-only stage plan is launchable without R5B/R6 capabilities; an MBP-1 study plan refuses before R5B; a spectral plan refuses before the post-V1 expansion | all-or-nothing operator availability | R5 |
| V1 boundary (post-V1 expansion) | — | registry + release docs | no GMM/spectral/Nyström fit callable anywhere in V1; hardening does not require the expansion | scope creep into V1 | Hardening |

---


### 3.10 Final contract-closure tests

| Test | Level | Assertion | Gate |
|---|---|---|---|
| Feature-block resolution identity | U | definition status is separate from `FeatureBlockResolutionPayload`; any formula/source/schema/window change creates a new resolved block ID | R1/R5B |
| Decision-policy and schedule envelopes | U | registry key differs from resolved policy ID; schedule payload is non-self-referential; all old ID names are absent | R5 |
| Verification-run authorization binding | I | `VerificationRunPayload` exactly matches pipeline ID, allowlist, seed, coverage matrix, and authorization before any source path | R1 acceptance |
| Trade-path bundle completeness | U/I | every executed trade has one exact path artifact; ordered bundle hash is stable and path events have exact IDs | R3 |
| Account versus portfolio simulation identity | U | mixed firm/risk/withdrawal/replacement policy-set legs produce one complete portfolio identity; no singular policy field can under-specify a portfolio | R3 |
| Capability-based path requirements | U | support is based on required capabilities plus accepted classes, not enum ordering; ordered fills do not imply market-price chronology | R3 |
| Firm/scenario separation | U | adverse-first/favorable-first appears only in scenario policies, never in `PhaseRules` | R3 |
| Typed MBP-1 window definitions | U | every order-flow feature maps to one `Mbp1FeatureWindowSpec`; comparator/bounds/cutoff/missingness enter the resolved block identity | R5B |
| Regime fit provenance | U | assignments reference exact fit+protocol; capability assessment references exact fit set+fold set; promotion changes no fit ID | R6-core |
| Current-authority completeness | docs | no undefined contract symbol, ellipsis enum, “previous version” authority reference, stale identity vocabulary, or incorrect release count remains | R0/R1 |


### 3.11 Frontend UX contract acceptance

The following rows are normative implementations of `FRONTEND_UX_CONTRACT.md` §36. `A` = Streamlit AppTest, `B` = interactive browser/visual, `U` = pure unit, `S` = source scan.

| Test / contract IDs | Level | Fixture/state | Assertion | Gate |
|---|---|---|---|---|
| FUX-IA-001..003 | A/S | current IFVG shell + M0–M3 stub | top-level order unchanged; Experiments horizontal sub-nav exact; only selected route executes; Context Research delegates unchanged | R4 |
| FUX-MOD-001 | U/S | import/lint scan | thin script layers; pure presentation logic in `src`; new scripts included in Ruff; no React revival | R4 |
| FUX-WIZ-001 | A | wizard draft | eight exact steps, progress/breadcrumb, validation-gated Next/Back, no field loss on Back | R4 |
| FUX-WIZ-002..003 | A/I | draft store + frozen charter | autosave on Next; Save Draft always visible; exact-step restore; History lists draft; clone creates new draft; original unchanged | R4 |
| FUX-WIZ-004 | A | every mode/template | five modes, four research questions, six templates; primary objective, hard constraints, and tie-breaks visible | R4 |
| FUX-WIZ-005 | A | baseline fixtures | human baseline card and full copyable technical identity; blocked baseline explains and cannot advance | R4 |
| FUX-WIZ-006 | A/S | axis registry fixtures | exact market-meaning groups/card fields; no input widget for locked/blocked; no raw override/JSON editor; computation-path chip and evidence expander | R4 |
| FUX-WIZ-007 | A/I | authorization matrix | computation-path-scoped checklist; real scope fails closed on required missing evidence; unrelated decisions absent | R4 |
| FUX-WIZ-008 | A | synthetic/stale/first-party contract fixtures | full contract-card fields; truthful status; stale/blocked behavior; no scenario assumption inside firm card | R4 |
| FUX-WIZ-009 | A | multi-firm policy fixture | universal strategy tree with per-firm account/risk/withdrawal/replacement policies; strategy identity unchanged | R4 |
| FUX-WIZ-010 | A | gate fixtures | three ordered gate groups; resolved values/status; downstream skipped reason visible | R4 |
| FUX-WIZ-011 | A/I | verification and full-scope specs | read-only boundaries, canonical allowlist/download, <=5 validation, exact badge, real authorization state, no automatic full scope | R4/R5 |
| FUX-WIZ-012 | A/I | review fixture | complete resolved charter/work estimate, replay-vs-prop cost split, reuse hits, immutable freeze, exact typed full-scope warning, detached launch only in handler | R4/R5 |
| FUX-MON-001 | A/U | running status JSON | 5s fragment calls plain body; manual refresh fallback; phase checklist; five keyboard buttons and funnel sync | R4 |
| FUX-MON-002..004 | A/I | mixed child states | exact table columns/pagination/copy; sanitized detail/actions/attempts; safe-cancel sentinel; completed child immutable; missing-file CLI fallback | R4 |
| FUX-RES-001..002 | A | passing/no-pass reports | picker/disclosure/dev badge/scope and gross-cost-net labels; exact four cards; exact `No configuration passed all benchmarks.` copy and reasons | R4 |
| FUX-RES-003 | A/B | frontier fixture | exact axes/encodings; point selection updates page; always-present selectbox twin; blocked points truthful | R4/hardening |
| FUX-RES-004 | A/B | sensitivity grid | axis/metric controls; ◼/▲/✕/·/⊘ glyph classes; table twin; color-independent semantics | R4/hardening |
| FUX-RES-005 | A/B | multi-firm simulation | firm metric toggles; step/dash survival; payout horizons; mean/median/P10/P90 with P10 prominent | R4/hardening |
| FUX-RES-006 | A/B | 64/256-child tables | Strategy/Prop/Robustness presets; sticky/fallback columns; indexed sorting/pagination; baseline-diff names; no full artifact load | R4/hardening |
| FUX-RES-007 | A/I | compatible/incompatible/collision comparisons | dimension ribbon incl. match_basis; four panels; exact affected IDs; incompatible/not-comparable suppress unsupported deltas | R4 |
| FUX-RES-008 | A/U | insight artifacts | seven categories, verbatim deterministic text, evidence actions, neutral/sample/concentration disclosures | R4 |
| FUX-RES-009 | A/B/I | ordered account events | required lines/windows/marker shapes/table; envelope order; marker/table exact trade/setup/path drill-down | R4/hardening |
| FUX-DRILL-001 | A/I/S | missing/exact IDs | exact jump succeeds; setup/time/nearest/fuzzy fallbacks absent; unresolved state sanitized | R2/R4 |
| FUX-HIST-001 | A/I | draft/frozen/completed/legacy records | sections separated; no delete/overwrite for immutable evidence; annotation edits separate; clone and compatible comparison only | R4 |
| FUX-PIPE-001 | A | capability matrix | complete Configure fields; stage-plan capability scoping; planned/blocked entries visible-disabled; R5/R5B/post-V1 states truthful | R5/R5B/R6 |
| FUX-PIPE-002 | A | preview fixtures | exact counts, dates/stages, runtime/storage by phase, reuse and expected-artifact disclosure | R5 |
| FUX-PIPE-003 | A/S | render/import/button spies | launch only in button handler; exact typed full-scope confirmation; no Popen/full run from import/startup/tests | R5 |
| FUX-PIPE-004 | A | attempt/checkpoint fixture | exact progress fields, 16 stages, reused/blocked/not-required states, semantic ID and attempt history | R5 |
| FUX-PIPE-005 | A/I | failed attempt | checkpoint resume; same semantic/new attempt; resource clone; research change new semantic ID; completed stages reused | R5 |
| FUX-PIPE-006 | A/I | gate pass/fail | prepared_not_published; gate checklist; activation separately enabled only on pass; verification cannot activate | R5/hardening |
| FUX-STATE-001 | A | every §31 state | each empty/blocked/failure state renders human reason, owning gate and next action; no traceback/path | R4–hardening |
| FUX-A11Y-001..002 | B/A | keyboard-only + semantic scan | all primary workflows keyboard operable; visible labels/help; chart twins; status/heatmap not color-only; logical focus order | hardening |
| FUX-A11Y-003 | B | 1440×900, 1024×768, 768×1024, 390×844 | required pages/states fit, controls reachable, no clipped actions; screenshots retained | hardening |
| FUX-A11Y-004 | A/B | capability monkeypatches | manual refresh, selection twin, inline confirmation, and unpinned table fallbacks preserve semantics | R4/R5/hardening |
| FUX-PERF-001 | U/A/B | large synthetic artifacts | indexed reads, bounded layers/OmissionReport, UI performance budgets, no hidden-panel work | R4/hardening |
| FUX-SAFE-001 | S/A | source/widget scan | no raw path/traceback/secret, `allow_sealed`, delete immutable, recapture, promote, order/live activation, or Trade-Lab controls | every UI release |
| FUX-LABEL-001 | S/A | all UI strings | Development Exploratory Representative; explicit result scopes; match_basis; assumed paths scenario/approximation; forbidden terms absent | every UI release |
| Frontend retention crosswalk | docs | brief §10.1–§10.30 | every source requirement maps to an authoritative `FRONTEND_UX_CONTRACT.md` section and applicable test; no generic-summary substitution | R0/R4/hardening |


## 4. Performance budgets (revised rows only)

The complete budget table (self-contained per V3 P0-2; all measured on the 5-day fixture and synthetic fixtures only):

| Benchmark | Budget | Basis |
|---|---|---|
| Child enumeration + identity resolution (64 children) | < 5 s | pure hashing |
| **Real vertical slice** (1 baseline profile × ≤5 days × dual-drive) | ≤ 5 min wall | 2 × ~75 s replay (measured ≈15 s/day) + audit/chart/store/verify overhead |
| Synthetic 2×2 orchestration E2E | ≤ 2 min wall | replays stubbed |
| Exact/scenario historical prop sim per firm × policy | < 1 s | in-memory walk |
| 10,000-path day-block bootstrap per firm × policy (5-day stream) | ≤ 60 s | existing engine scale |
| Stress suite (9 scenarios × 10k) per firm × policy | ≤ 10 min | 9 × bootstrap |
| Frontier + robustness for 64 children | < 5 s | O(n²), n ≤ 256 |
| Store save→reload→verify round trip | < 2 s | manifest protocol |
| Catalog rebuild from events + manifests (10k events) | < 5 s | deterministic replay of the event log |
| Dashboard initial load (Results, synthetic 64-child artifact) | ≤ 5 s | indexed reads |
| Results filtering / explorer page | ≤ 1 s | paged loader |
| Chart render (frontier/heatmap/timeline) | ≤ 2 s | trace budgets |
| Exact drill-down load | existing verifier budget (first load slow, cached after) | `_cached_replay_context` |
| Artifact storage growth per child | recorded; alarm at > 200 MB/child | envelope + refs only |
| Repeated-run variability | byte-identical manifests | determinism |

Safeguards: bounded workers (≤4) and memory (`max_tasks_per_child=1`), identity-deduped replays, incremental per-child publication, resumable parent jobs, safe-boundary cancellation, deterministic reuse, UI pagination, no full artifact load into Streamlit memory, hard `max_child_count` cap. Operator-scale estimates (≈35 min per 138-day child ÷ workers; ~40 s/day TF-chain prewarm per group) are recorded at preview, refined from actual operator runs, and never enter verification identities.

---

## 5. Acceptance-gate mapping (updated rows)

| Brief / revision gate | Covered by |
|---|---|
| 16.3 backend functional | §3.5 synthetic E2E (+ §3.1 reuse rows) — control flow; **not** a real-data claim |
| 16.4 prop engine | §3.3 |
| 16.5 simulation | §3.4 (corrected duplicate-path semantics) |
| 16.14 five-day budget | §1 Path A + §2 + negative tests; `full_pipeline_not_run=true` |
| Revision §5 "Identity and reuse" (1–5) | §3.1 rows 1–8 |
| Revision §5 "Five-day verification" (6–10) | §1–§2 (matching seed; synthetic multi-child; coverage-evidenced allowlist; nonresearch gates; no full-dev gate) |
| Revision §5 "Sequential and audit correctness" (11–14) | full-sequential-replay-per-child (orchestrator has no filter path); `ChildAuditNeutralityReport` per child (§3.3 of `CONTRACTS_AND_SCHEMAS.md`; dual-drive proven on Path A); lineage-or-disabled deltas (§3.2); S11 blocked until ratification (§3.6) |
| Revision §5 "Prop fidelity" (15–21) | §3.3 |
| Revision §5 "Delta and feature architecture" (22–25) | §3.6 cohort tests; MBP-1 ordering tests (`DELTA_TAXONOMY.md` §6.2); §3.1 version-bump row; interpretation labeling |
| Revision §5 "ML/regime" (26–30) | revised scope (§3.6, `ML_REGIME_CONTRACT_PLAN.md` §11–12) |
| Revision §5 "UI and publication" (31–35) | §3.7, §3.11, and catalog rows in §3.1 |
| 16.2 immutable/data contract | §3.1 identity/store/schema tests + R1 save/reload/overwrite refusal |
| 16.6 UI acceptance | §3.11 complete `FUX-*` AppTest/browser/accessibility matrix |
| 16.7 deterministic insights | synthetic insight templates + evidence-link and no-causal-language tests |
| 16.8 exact drill-down | verifier exact-ID navigation and lineage/match-basis tests |
| 16.9 identity/immutability | §3.1 and §3.8–§3.10 identity projection/reuse/attempt tests |
| 16.10 migration/compatibility | M0–M3 frozen-lane regression and no-migration tests |
| 16.11 access safety | §1–§2 + §3.7 zero-counter and fail-before-path tests |
| 16.12 performance/capacity | §4 complete budgets |
| 16.13 planned feature expansion | planned-state refusal, block activation/version bump, R5B workflow tests |
| 16.15 full-pipeline UI | §3.11 `FUX-PIPE-*`, capability-scoped stage-plan, exact confirmation, resume/retry/publication, and automatic-trigger refusal tests |
| 16.16 ML/regime | `ML_REGIME_CONTRACT_PLAN.md` §11 acceptance rows + §3.6/§3.10 tests |
