# Architecture — Quant-Lab

Updated: 2026-06-04.

Quant-Lab is the research/training workbench for NQ/ES futures ML models. Its current production-aligned path is the **dashboard-utility** workflow, which is now single-sourced to **Strategy-Core v3** for the decision semantics that must match runtime execution.

> Current compatibility state: Quant-Lab emits `strategy_core_engine_v3` contracts and builds dashboard-utility datasets through Strategy-Core. Trade-Lab is **not yet v3-compatible**; see `../Strategy-Core/V3_COMPATIBILITY_MATRIX.md`.

---

## Canonical read order

1. `docs/README.md` — docs inventory and stale/historical classification.
2. `ARCHITECTURE.md` — this current architecture summary.
3. `docs/ML_TRAINING_WORKBENCH.md` — Streamlit workflow details.
4. `docs/pipeline_state.yaml` — machine-readable current-state summary.
5. `../Strategy-Core/README.md` and `../Strategy-Core/V3_COMPATIBILITY_MATRIX.md` — shared engine and cross-repo contract truth.
6. `scripts/ml_training_tab.py` — orchestration of build/train/save.
7. `src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py`, `engine_decision.py`, and `strategy_contract.py` — v3 utility path.

Older historical reports and scaffold prompt documents were pruned from the working tree. Reconstruct current state from the canonical docs above plus current code/tests; use Git history only for audit context.

---

## Workflows

### 1. Extrema Rebound/Crossing mode — research only

Purpose: binary classifier over tick-level extrema.

Pipeline:

```text
local Databento parquet
  -> ExtremaDatasetBuilder
  -> extrema detection
  -> rebound/crossing labels
  -> PL/MS feature extraction
  -> walk-forward CatBoost binary evaluation
  -> final refit model bundle
```

Key modules:

- `src/alpha_lab/agents/data_infra/ml/dataset_builder.py`
- `src/alpha_lab/agents/data_infra/ml/extrema_detection.py`
- `src/alpha_lab/agents/data_infra/ml/labeling.py`
- `src/alpha_lab/agents/data_infra/ml/features_microstructure.py`
- `src/alpha_lab/agents/data_infra/ml/features_momentum.py`

Runtime caveat: `src/alpha_lab/agents/signal_eng/detectors/tier3/ml_extrema_classifier.py` is explicitly experimental and has a known train/serve domain mismatch.

### 2. Dashboard Utility mode — production-aligned research path

Purpose: 3-class level-touch classifier whose semantics are intended to be reproducible by Trade-Lab once Trade-Lab is repointed to Strategy-Core v3.

Pipeline:

```text
local Databento parquet
  -> TickStore / DuckDB bar + tick queries
  -> dashboard_utility_builder.py
  -> Strategy-Core v3 decision layer
       build_zones
       detect_touches with available_from guard
       resolve_honest_outcome
       v3 feature formulas
  -> walk-forward CatBoost MultiClass evaluation
  -> final refit model bundle
  -> strategy.json emitted from Strategy-Core constants
```

Key modules:

- `src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py` — self-contained date loop, bars, levels, cache writes.
- `src/alpha_lab/agents/data_infra/ml/engine_decision.py` — adapter from Quant-Lab dataframes/tick queries to Strategy-Core neutral types.
- `src/alpha_lab/agents/data_infra/ml/strategy_contract.py` — emits `strategy.json` from Strategy-Core constants/version stamps.
- `src/alpha_lab/agents/data_infra/ml/config.py` — config models and dataset cache hash; hash includes Strategy-Core engine version and price/label semantics.
- `src/alpha_lab/agents/data_infra/tick_store.py` — local DuckDB-backed parquet query layer.
- `scripts/run_dashboard_session_experiment.py` — CLI entrypoint for repeatable session-scope experiments.

Session-scope experiments are explicit research config. The default is to train/evaluate on `asia`, `london`, and `ny`, while production-gate diagnostics remain NY-only. Presets such as `ny_only`, `asia_only`, `london_only`, `asia_london_only`, and `all_sessions_all_gates` apply after dataset generation so caches stay reusable while fold training, OOS metrics, final refit, and gate reporting stay auditable.

### 2b. Prop-firm evaluation walker (`alpha_lab.propsim`) — model-selection consumer

Purpose: pass-probability for prop-firm evaluations (TopStep 50K is preset one) from per-trade equity paths — the PROP-SIM window's barrier-options walker. Pure simulation core (no Strategy-Core dependency): `models.py` (TradePath/Ruleset/WalkResult), `presets.py` (registry; presets are data), `engine.py` (the EOD-ratcheted trailing floor with real-time breach, soft/hard daily-loss limit, consistency rule; breach modes `realized_only` and `unrealized_adverse_first` are both always computed), `bootstrap.py` (seeded day-level block bootstrap Monte Carlo), `loaders.py` (Trade-Lab executions+journal join / journal-outcomes evidence mode / bundle `oos_predictions.parquet`), `report.py` + CLI `python -m alpha_lab.propsim`. OOS parquets predating PROP-SIM P1 (no `max_mfe_pts`/`max_mae_pts`) degrade the unrealized mode to realized-only with a stated reason (D-038).

### 3. Retained legacy compatibility/export path

The older `src/alpha_lab/experiment/`, `scripts/experiment_tab.py`, and `scripts/train_dashboard_model.py` path is retained as historical/compatibility tooling. It is **not** the canonical Strategy-Core v3 bundle path.

Legacy artifact:

```text
data/models/dashboard_3feature_v1.cbm
```

Do not treat that legacy 3-feature artifact as v3-compatible unless its accompanying `strategy.json` validates against `strategy_core_engine_v3` and the bundle files/checksums are verified.

---

## Strategy-Core v3 semantics used by dashboard-utility mode

| Area | Current behavior |
|---|---|
| Engine stamp | `strategy_core_engine_v3` |
| Contract stamp | `trade_lab_contract_v1` |
| Bars | Trade-price tick bars on 0.25 grid; default touch bar `147t`. |
| Sessions | ET-native: 18:00 trading-day boundary; `asia` 19:00→02:45, `london` 03:00→08:00, `ny` 09:00→17:00; gaps classify as `none`. |
| Levels | `PDH/PDL` = full prior trading-day high/low over `[18:00, 18:00)` ET. Session levels: Asia high/low and London high/low. |
| Availability guard | Enforced. A level cannot be touched before the session that defines it has closed. Merged-zone availability is the max constituent availability. |
| Touches | Merged zones within 3.0 points; representative = mean; first bar whose `[low, high]` intersects the zone representative; first touch per zone/day. |
| Features | `int_time_beyond_level`, `int_time_within_2pts`, `int_absorption_ratio` from trade prints; live runtime approach subset is `app_large_trade_vol_pct`, `app_avg_trade_size`, `app_max_spread`. |
| Labels | `tradeable_reversal=0`, `trap_reversal=1`, `aggressive_blowthrough=2`; MAE-first same-bar priority; TP 15, SL 30, trap MFE min 5 by default. |
| Honest entry | Decision can fire only after the post-touch feature window: `touch_close + 5m`. Label/outcome entry is the realistic trade price at that decision instant, not the level price at touch time. |
| Cutoffs | Drop new decisions at/after 16:40 ET; forward label cutoff is 17:00 ET. |
| Inference gate | Default contract gate is `tradeable_reversal`, `eligible_session="ny"`, `confidence_gate=0.70`. |

---

## Data and generated outputs

Local data layout:

```text
data/databento/{symbol}/{YYYY-MM-DD}/mbp10.parquet
data/databento/{symbol}/{YYYY-MM-DD}/mbp1.parquet
data/databento/{symbol}/{YYYY-MM-DD}/trades.parquet
```

Per-date training caches:

```text
ml_features_{config_hash}.parquet   # extrema mode
ml_utility_{config_hash}.parquet    # dashboard-utility mode
```

Saved Streamlit model bundle:

```text
models/{model_name}/model.cbm
models/{model_name}/metadata.json
models/{model_name}/evaluation.json
models/{model_name}/strategy.json
models/{model_name}/oos_predictions.parquet  # when OOS rows are available
```

Saved `evaluation.json` and `metadata.json` include `session_experiment` and `session_filter`; emitted `strategy.json` includes `research_session_experiment` but still advertises `supported_by_runtime=false` until Trade-Lab v3 parity is proven.

Generated/local outputs, not source-of-truth docs/code:

- `models/`
- `catboost_info/`
- `*.cbm`
- cached parquet/csv files under `data/`
- scratch chart HTML files
- local imported Databento data

The roadmap item "identify canonical data/model bundle location and verify file presence/checksums" is deliberately deferred until AlgoChef's local data zip is available.

---

## Verification expectations

- For code changes, run focused tests and update relevant docs in the same change.
- For dashboard-utility semantics, run Strategy-Core tests and Quant-Lab contract/no-drift tests before claiming v3 alignment.
- For any model/backtest claim, report date range, data source, fees/slippage assumptions, trade count, return/expectancy, drawdown, and limitations.
- Do not claim Trade-Lab runtime readiness until Trade-Lab is repointed to Strategy-Core v3 and end-to-end parity is proven.

---

## 2d. IFVG robust FSM configuration search & prop realization lane (`ifvg_prop_robust_config_search_v1`)

Additive research lane beside the frozen M0-M3 context-experiment lane; zero
Strategy-Core changes in v1. R1 (contracts/identities/access/stores) is
implemented; its acceptance is BLOCKED pending the owner's verification-fixture
authorization (`VerificationAuthorizationRef`, owner decisions 21/R-5).

```text
src/alpha_lab/agents/data_infra/ifvg/search/    identities · axis_registry · authorization ·
                                                charter · failure · store · catalog ·
                                                verification · child_replay
src/alpha_lab/agents/data_infra/ifvg/study/     dimension_contracts · cohort · study_cell ·
                                                computation_path · delta_outputs · comparison_contracts
src/alpha_lab/agents/data_infra/ifvg/features/  feature_blocks · feature_bundles · mbp1_source_contract
```

Load-bearing seams (D-039, D-040, D-042, D-043):

- **Decomposed replay identity**: `CoreStrategyReplayIdentity` hashes only
  replay-defining facts (content-addressed `ReplayInputBundle` with exact
  physical source partitions + day-artifact manifests, scoped QL/SC source
  identities, resolved section hash, canonical profile id, seed identity,
  resolver). Study membership, cost, audit/chart schema, and runtime access
  audits never enter it — one replay is reusable across studies.
- **Canonical child naming**: generated children are
  `ifvg_search_profile_<name-free-hash16>` (study-independent record IDs);
  baselines keep registered names; `GeneratedProfileCapability` gates children
  (the fixed `PROFILE_CAPABILITY_REGISTRY` gates baselines only).
- **Typed axis registry**: every `IfvgSmcSection` field except `profile_name`
  is classified (locked invariants / thesis / approved axes / blocked incl.
  the inert `break_even_enabled`/`legacy_candidate_row_limit` traps and
  `parent_full_fill_invalidation` = `blocked_pending_owner_policy_review`);
  values are individually ratifiable; no raw `section_overrides` surface.
- **Two-path verification**: the ONLY real-data verification is one baseline
  ≤5-trading-day vertical slice under `VerificationReplayPolicy` (third
  trusted class in `require_fixed_exploration_allowlist`) with nonresearch
  control-flow gates; everything multi-child is synthetic. One canonical
  allowlist program-wide (marker-enforced); the real slice cannot construct a
  source path without the owner's `VerificationAuthorizationRef`.
- **Immutable stores + concurrency-safe catalog**: envelope stores under
  `data/ifvg_datasets/search/v1/` (test namespace `search_test/v1/`) on the
  manifest protocol (refuse-if-exists, tmp + `os.replace`, reload-assert);
  the mutable catalog is a lock-guarded append-only `catalog_events.jsonl`
  plus a deterministic rebuildable index (display annotations only).
- **Study cells / comparisons**: 16 hashed semantic dimensions, annotations
  never hashed; comparisons fail closed to `config_diff_only`; computation
  paths derived from the fail-closed dimension registry.
- **MBP-1 boundary**: MBP-1 is the maximum order-flow depth for every new
  contract (deep-book identifier guard everywhere); the single opaque
  provenance literal `legacy_verified_replay_source` is unqueryable by the
  feature layer. Order-flow activation is the R5B versioned registry event
  and remains research-only offline (owner decision R-6).

R2 additions (multi-child search, lineage, deltas, verifier integration):

- **Parent orchestrator** (`search/orchestrator.py`): deterministic
  enumeration over registered axis values deduped on the resolved replay
  identity (within and across studies), `GeneratedProfileCapability` enforced
  BEFORE any replay, O_EXCL per-search lock with checkpoint heartbeats +
  stale-orphan break (resume-after-kill), atomic `search_state.json`
  checkpoints per child transition, `cancel.requested` honored at child
  boundaries only, store-identity verified reuse. Launch shim:
  `scripts/ifvg_search_job.py` (start/status/cancel; worker refuses execution
  without an explicit runner entry — real executors land with the R5
  pipeline).
- **Profile-independent lineage** (`search/lineage.py`): setup → candidate →
  decision → trade lineage payloads derived from source-stable evidence
  (deterministic SC fvg ids + cursors); native→lineage one-to-one enforced
  with persisted `LineageCollisionRecord`s; collisions or incomplete keys
  disable the population (never deduped or fuzz-matched).
- **Exact deltas** (`study/population_delta.py` · `funnel_delta.py`): every
  population delta declares its `match_basis` (`native_id_exact` same-profile,
  `profile_independent_lineage_exact` cross-profile, else `not_comparable`
  with a reason); funnel deltas run over the union counter vocabulary with
  typed conversions and terminal-reason deltas.
- **Gates / frontier / robustness / insights** (`search/gates.py` ·
  `strategy_metrics.py` · `frontier.py` · `robustness.py` · `insights.py`):
  all eleven charter thresholds evaluated with human explanations (incl. the
  bootstrap-CI-excludes-zero gate over the trading-day cluster bootstrap);
  deterministic O(n²) dominance with a persisted lexicographic tie-break trace
  selecting a `Development Exploratory Representative`; ±1-step neighbor
  degradation + plateau widths + knife-edge warnings; seven fixed insight
  categories with exact `EvidenceRef`s, match-basis-aware suppression, and a
  structural forbidden-wording refusal.
- **Declared contrasts** (`study/contrasts.py`): charter-declared only
  (post-hoc refused), fully-crossed paired main effects with the seed-7
  pair bootstrap CI, observational wording enforced by type.
- **Companion wiring (DEV-R1-6 closed)**: `build_child_fsm_audit` +
  `publish_child_fsm_audit` (`fsm_audit_preparation.py`) assemble the
  per-child audit companion from the audit-enabled drive's retained trace and
  channel rows, gated by the per-child `ChildAuditNeutralityReport` and the
  exact funnel⇔audit reconciliation — parity-EXEMPT by design (the accepted
  doc-default parity gate is untouched and remains doc-default-only);
  `build_slice_companions` (`search/child_replay.py`) closes the vertical
  slice's audit/publication/verifier-link gates (v2 tables via the existing
  heavyweight saver; neutrality + audit companions into the immutable search
  stores; exact drill-target resolution proven, vacuous-zero-targets recorded
  honestly).
- **Exact-setup drill-through**: `resolve_selection` gains the `setup_id`
  exact-ID kind (unique-candidate resolution; zero/multi-candidate setups
  refuse toward setup mode — never a policy pick); `queue_jump("setup_id", …)`
  routes into the setup verifier's own exact resolver before the mode radio
  instantiates.

R3 additions (prop lifecycle — fidelity contracts first, then the synthetic
account engine; `alpha_lab.propsim` lifecycle modules, additive beside the
untouched evaluation-only walker):

- **Trade-path fidelity** (`propsim/trade_path.py`): typed path evidence and
  bundles with payload/envelope identities; a 1m bar's
  `observed_intrabar_order` can only be `"unknown"`; assumed intrabar paths
  are registered SCENARIO policies (`bar_adverse_extreme_first_v1` /
  `bar_favorable_extreme_first_v1` — two identities and, on order-sensitive
  trades, two different results); rule support is capability-based
  (`PropRulePathRequirement`: required path capabilities + accepted fidelity
  classes, never enum ordering) and fails closed via `PathCapabilityReport`.
- **Typed calendars** (`propsim/calendar.py`): `DayCountBasis`/`DurationRule`
  under a `SimulatedClockPolicy`; a basis the active clock cannot represent
  (calendar-month recurring fee under day-block bootstrap without a synthetic
  calendar) fails closed with `UnsupportedCalendarRuleError`.
- **Firm contracts + contract evidence** (`propsim/firm_contracts.py` ·
  `contract_evidence.py`): permitted rules/thresholds/observation policies +
  the rule→capability matrix; adverse/favorable ordering never inside
  `PhaseRules` (scenario policies only); source documents → per-field
  evidence → compilation → owner review → supersession on a one-way status
  ladder where synthetic evidence can NEVER reach `first_party_verified`.
- **Account walk** (`propsim/account.py`): full lifecycle (evaluation →
  funded → payouts/fees/replacement) as ONE strictly ordered
  `PropAccountEventEnvelope` stream (`prop_account_event_order_v1`, global
  `event_ordinal`, exact source trade/decision/candidate/setup/path links);
  the evaluation-only walker (`engine.py`) stays untouched, compatibility
  proven by the one-contract `AccountWalk`≡`EvaluationWalk` parity fixture.
- **Risk sizing + withdrawal behavior** (`propsim/risk.py` ·
  `withdrawal.py`): every sizing family with typed skip reasons (never
  silently forced to one contract); trader withdrawal choices are a separate
  identity from the firm contract and both enter every simulation identity.
- **Adapters + stream hash** (`propsim/adapters.py`): v2 executed-trade →
  account-trade tick→point/cost mapping plus the stable ORDER-SENSITIVE
  gross stream hash pinning the exact resolved trade sequence.
- **Portfolio / stress / simulation identities** (`propsim/portfolio.py` ·
  `stress.py` · `simulation.py`): copied accounts replay ONE common
  correlated market path per draw (no per-account resampling path exists);
  nine seeded deterministic stress scenarios ride the simulation identity;
  account/portfolio simulation identities carry every result-changing policy
  (constructor-surface audit: no result-changing constructor-only kwarg);
  duplicate bootstrap index sequences are legal with a unique
  `path_instance_id` + stored `sampled_index_sequence_hash` per draw.
- **Prop metrics + search wiring** (`propsim/prop_metrics.py` ·
  `search/gates.py` · `search/orchestrator.py`): lower-tail-first
  `PayoutReliabilityVector`; `evaluate_prop_gates` fail-closed rows (None
  thresholds are explicit not-required passes; missing observations fail);
  the orchestrator `prop_simulator` seam applies the conservative ALL-legs
  feasibility rule, merges each prop objective into the frontier as the
  WORST value across the child's simulations, defers prop-owned pareto
  objectives past the strategy stage only when a simulator is wired, and
  records explicit skip notes otherwise.

R4 additions (trader workspace UI — the guided study surface over R1–R3;
`FRONTEND_UX_CONTRACT.md` is the normative authority):

- **Workspace routing** (`scripts/ifvg_study_tab.py` + the Experiments
  delegation in `ifvg_lab_tab.py`): a session-state-backed horizontal radio
  (`New Study | Active Runs | Results | History | Context Research`,
  namespace `ifvg_study_v1_*`); only the selected route executes; the
  M0–M3 Context Research panel delegates verbatim; a namespace selector
  switches between the research (`search/v1`) and verification
  (`search_test/v1`) stores with truthful badging.
- **Presentation contracts** (`ifvg/study_status.py` ·
  `ifvg/study_presentation.py`): CS §13 status/scope/empty-state registries
  with the exact required copy (`Development Exploratory Representative`,
  the verification badge, the no-pass sentence, `Not run — strategy gate
  failed`); pure wizard validators; funnel/stage derivations pinned by
  contract test to the orchestrator's exact explanation sentinels;
  baseline-diff display names; work estimates as operational annotations.
- **Drafts** (`ifvg/study_drafts.py`, `data/ifvg_study_drafts/`): the one
  deliberately mutable authoring surface — atomic JSON, autosave on Next,
  exact-step restore, deep-copy Clone as New Search; `mark_frozen` is the
  single permitted final write and every later mutation/discard refuses.
- **UI providers** (`ifvg/study_providers.py`): exact-ID manifest-verified
  loads only; run enumeration from the mutable job root + catalog event
  log (immutable store roots are never listed; the orchestrator records
  the frontier envelope id as a state-file phase note as the locator);
  `prepare_cross_profile_deltas` is the ONLY cross-profile delta
  constructor and persists BOTH lineage-uniqueness reports first.
- **Wizard** (`scripts/ifvg_study_wizard.py`): five modes × eight steps;
  registered axis cards grouped by market meaning with NO widget for
  locked/blocked/measured axes and no raw override editor anywhere;
  computation-path chips; truthful synthetic contract cards; per-firm
  account/risk/withdrawal/replacement policy sets under one universal
  strategy profile; three ordered benchmark gate groups
  (`proposed_protocol_default` stamps); read-only verification allowlist +
  the exact typed full-scope confirmation; freeze → `validate_charter` →
  immutable charter save → detached launch ONLY in the button handler via
  the registry-gated job shim.
- **Monitor** (`scripts/ifvg_active_runs_tab.py`): `st.fragment(5s)` over a
  plain AppTest-callable body + manual Refresh; phase checklist; five
  keyboard funnel buttons (the Plotly funnel accompanies, never replaces);
  the exact child table with skipped-stage reasons; sanitized detail;
  confirmed safe-cancel sentinel; CLI escape hatch on missing status.
- **Results/History/Compare** (`scripts/ifvg_results_tab.py` ·
  `ifvg_results_compare.py` · `ifvg_results_charts.py`): overview cards +
  exact no-pass copy; frontier with an always-present selectbox twin;
  heatmap glyph classes (◼ ▲ ✕ · ⊘) + table twin; firm matrix / survival
  (step+dash) / payout distributions (P10-first); indexed explorer with
  identity-first columns; dimension-diff ribbon with `match_basis` and
  four panels (membership claims disabled for non-comparable pairs, never
  fuzzed); seven deterministic insight categories with exact
  `EvidenceRef` actions; the account timeline in `event_ordinal` order
  with FUX marker shapes and exact-id drill-down; all figures budgeted
  with honest `OmissionReport`s.
- **Execution gating** (`ifvg/search/runner_registry.py` + the shim): the
  job shim's `--runner-entry` is registry-gated — the UI passes registered
  KEYS only, raw strings are refused before any import, and the only
  registered entry before the R5 executors is the synthetic fixture
  wiring, so a real charter's launch renders capability-blocked; `resume`
  re-enters the idempotent worker. Declared two-axis interaction
  contrasts now evaluate on balanced grids (`study/contrasts.py`,
  difference-of-differences, seed-7 bootstrap; refusals retained).

R5 additions (pipeline runner + MBP-1 contract readiness + supervised model
ladder; capability-scoped operator availability per V3 P1-5):

- **Pipeline contracts + runner** (`ifvg/search/pipeline.py`): the CS §7
  split — `PipelineSemanticSpecPayload` (research-bearing fields only;
  canonical-order 16-stage plans with dependency closure) hashes to
  `pipeline_semantic_id`; `ExecutionAttemptIdentity` (workers/host/
  timestamps/retry reason) is deliberately NOT an identity envelope. The
  runner composes the study-lane primitives (`enumerate_children`, the
  reuse/neutrality/publication semantics, the costed-evaluation cache, the
  extracted `merge_prop_vectors` — ONE implementation of ALL-legs
  feasibility + worst-firm merge — and `build_frontier`); every stage
  executor is idempotent and store-reusing, and a stage whose
  freshly-minted `PipelineStageResultEnvelope` id equals the prior
  attempt's is marked REUSED (reuse proven by identity). S11 terminal-
  blocks with the exact registered reason; S15 persists the immutable
  `PipelineResultEnvelope` with `prepared_not_published` operational
  state; publication is verify-then-activate and verification scope can
  never activate a research catalog entry.
- **Carried-seam closures**: S02 persists per-child lineage evidence
  (uniqueness reports + serialized key projections as stage sidecars) and
  S14 builds + persists deterministic insight panels
  (`InsightPanelEnvelope`, `insights` store) and baseline↔challenger
  `ComparisonResultEnvelope`s (`search_results` store) that the UI now
  consumes (DEV-R4-7/DEV-R4-16); S12/S13 run the additively extended
  `alpha_lab.propsim.search_bridge` (scenario/bootstrap/stress mode
  bridging, DEV-R3-11) and persist real `AccountPolicySetEnvelope`s +
  `AccountSimulationEnvelope`s with the trader-UI sidecars (DEV-R4-17).
- **Supervised ML lane** (`ifvg/ml/`): the registry-gated ladder
  (`model_protocols` · `logistic_model` · `supervised_ladder`) runs the
  prevalence reference + fold-local logistic + the existing CatBoost
  protocol on IDENTICAL rows/folds (identical `oos_row_id` sets asserted
  before any delta; `paired_cell_delta_report` under the fixed 10k/seed-7
  day-block bootstrap); every preprocessing statistic is fold-fitted;
  fitted artifacts persist portably (manifest-relative refs + checksums,
  relocation-proof). `calibration_policies` / `decision_policies` are
  registry pairs with distinct logical keys and resolved envelope ids;
  only the diagnostic baselines are executable, every execution-affecting
  decision policy requires an owner-ratified `RejectedCandidatePolicy`,
  and `ProhibitedSelectionError` refuses enumeration outside a ratified
  charter. Core `drift_monitoring` exports report builders only.
- **Bundle feature views** (`ifvg/features/bundle_feature_view.py`):
  available-blocks-only views over the immutable candidate view; through
  R5, `IFVG_ORDER_FLOW_MBP1_V1` stayed `planned` and every bundle carrying
  it refused (no baseline-vs-MBP-1 study was constructible before the R5B
  activation below); a bundle maps onto the tier-frozen supervised ladder
  only when its resolved feature set equals a frozen tier exactly.
- **Real executors + pipeline shim** (`ifvg/search/executors.py` ·
  `runner_registry.py` · `scripts/ifvg_pipeline_job.py`): the registry now
  names the real baseline-verification search/pipeline executors — their
  factories fail closed at CONSTRUCTION without the owner's persisted
  `VerificationRunEnvelope` (registration unblocks the launch surface,
  never the data) — and full-development charters keep NO registered
  entry (the operator full run stays a separate authorized action). The
  detached pipeline shim mirrors the search shim (registry-gated worker,
  atomic status, safe-boundary cancel, `publish-gates`/`activate` CLI).
- **Full Pipeline Run surface** (`scripts/ifvg_pipeline_tab.py`, session
  namespace `ifvg_pipeline_v1_*`): the complete §30 workflow — Configure
  (capability-scoped stage plans; planned/blocked feature + model entries
  visible-disabled), Preview (exact counts/estimates/reuse/new-artifact
  disclosure), Launch (one scanned `_spawn_pipeline_job` seam; the exact
  §15 typed full-scope confirmation), Monitor (all 16 stages glyph+word
  incl. `not required`; `pipeline_semantic_id` + execution-attempt
  history; the supervised-ladder panel with planned rungs and the exact
  S11 blocked reason), Resume/Retry (operational clone; research-bearing
  change = new semantic id), Publish (gates checklist first; activation
  refused for verification scope).

R5B additions (offline MBP-1 feature activation — research-only, owner
ruling R-6; the versioned registry event of revision P1-5):

- **Activation as the published registry** (`ifvg/features/feature_blocks.py`):
  `FEATURE_BLOCK_REGISTRY` IS `with_activated_block` applied to the exported
  R5-era planned state (`PRE_ACTIVATION_*`) — `IFVG_ORDER_FLOW_MBP1_V1` at
  `block_version=2`, status available, its first
  `resolved_feature_block_id` minted from the real Arrow schema hashes, the
  frozen 9-window registry, `ifvg_order_flow_mbp1_formula_v1`, and
  `mbp1_feature_materializer_v1`; the block-registry hash changed and every
  ORDER_FLOW bundle (B2/B3) gained a new resolved id, while B1/B4 kept
  theirs and regime/execution-liquidity bundles keep refusing. The
  research-only boundary rides the DEFINITION
  (`expected_computation_path="offline_research_feature_materialization_v1"`,
  `can_affect_execution=False`): no live model feature, execution gate, or
  Trade-Lab serving use without a later Strategy-Core formula/parity
  contract and a separately approved sequential model-gated replay.
- **Exact schemas + immutable evidence** (`mbp1_arrow_schemas.py` ·
  `mbp1_source_artifact.py`): four pinned Arrow schemas with canonical
  field/type hashes (raw Databento mbp-1 retaining `ts_recv` — the
  materializer decodes it directly, no Strategy-Core change; the normalized
  working schema with the deterministic `source_ordinal` final tie-break;
  the 76-metric feature table with per-window validity/missing-reason
  evidence; the stage-window evidence table). The content-addressed
  `Mbp1SourceArtifact` freezes per-partition hashes, first/last order keys,
  sequence-gap intervals (vendor sequence resets are NOT gaps), and
  gap-adjusted day coverage; synthetic fixtures persist canonical event
  bytes as manifest-hashed sidecars, real artifacts reference partitions by
  hash; real reads run authorize-before-path through the verification
  policy family and `legacy_verified_replay_source` provenance is refused
  outright (guards re-run at R5B).
- **Point-in-time stage windows** (`mbp1_stage_windows.py` ·
  `mbp1_feature_materializer.py`): the candidate row's own five anchors
  (`tap/lock/armed/inversion/entry_ts_utc`) become `COMPLETED_BAR_BOUNDARY`
  cutoffs (`completed_bar_boundary_exclusive_v1`); admission runs on the
  complete `(ts_event, ts_recv, sequence, source_ordinal)` key —
  `PRE_TRIGGER_EXCLUSIVE` `<` / `POST_TRIGGER_INCLUSIVE` `<=` on exact
  keys, strict `ts_event <` for timestamp-only evidence with EVERY
  same-timestamp event excluded and the window typed
  `same_timestamp_order_unavailable` when any tie exists; a missing lower
  anchor refuses rather than widening; no `+inf` bound exists anywhere
  (source-scanned). The offline materializer preserves every candidate row
  under the deterministic typed-missing precedence (no partition → coverage
  → outside coverage → same-ts ambiguity → sequence gap → roll boundary →
  minimum events); formula-edge NaNs stay VALID; batch/repeat runs are
  byte-identical; the feature artifact's recipe identity (source artifact ×
  resolved block × anchor hash × cutoff policy × schema hashes) is separate
  from the post-materialization table hashes on the envelope.
- **Exact joins + bound views** (`mbp1_feature_join.py` ·
  `bundle_feature_view.py`): one-to-one on `candidate_id` only (duplicates
  refuse; no nearest-time or row-order fallback exists — source-scanned);
  MBP-1-bearing bundle views REQUIRE the materialized frame AND its
  artifact id, and the view payload pins `mbp1_feature_artifact_id`, so the
  same view+bundle over different evidence can never share one identity.
- **Controlled Baseline vs Baseline+MBP-1 study**
  (`ml/controlled_feature_study.py` + the bundle-parametrized ladder in
  `ml/supervised_ladder.py`, DECISIONS_TAKEN #41 arrival): the challenger
  runs against its OWN base bundle on identical rows/labels/folds —
  prevalence + logistic rungs (the CatBoost fold runner is tier-locked in
  the frozen M0–M3 lane and refuses with that exact reason); cross-arm row
  identity plus a numerically identical prevalence reference are asserted
  before the paired Brier delta (fixed 10k/seed-7 day-block bootstrap); the
  persisted study pins every input identity and carries the permanent
  `research_only_offline` stamp. Pipeline wiring: S00 refuses MBP-1 plans
  without the `mbp1_evidence_source` seam (order-flow evidence is never
  fabricated), S05 materializes + immutably persists the
  source/feature/coverage artifacts and joins them into the bundle view,
  S09 runs + persists the controlled study (readiness blocks CatBoost×MBP-1
  plans before launch), and the new `mbp1_source_artifacts` /
  `mbp1_feature_artifacts` / `mbp1_coverage_reports` /
  `controlled_feature_studies` stores follow the manifest protocol.
- **Dashboard** (`scripts/ifvg_mbp1_panels.py` + the pipeline surface): the
  MBP-1 Order Flow panel (availability with pre/post-activation registry
  hashes, the frozen window registry, exact-ID coverage/missingness
  evidence, per-candidate stage-window drill-down, and the Baseline vs
  Baseline+MBP-1 comparison) under the persistent `research_only_offline`
  badge; Configure restricts MBP-1-bearing bundles to the logistic
  protocol. The search job shim now passes the worker's `--store-root` to
  runner-entry factories (DEV-R5-10 closure). The five-day REAL
  control-flow verification of the materializer remains blocked on the
  owner's `VerificationAuthorizationRef`, exactly like every real half
  since R1.

R6 additions (V1 KMeans regime lane — protocol/fit/capability/promotion
split per V3 P1-3; the GMM/minibatch/spectral/Nyström implementations are
the post-V1 regime-expansion release):

- **Regime contracts** (`ifvg/ml/regime_contracts.py`): the four-way split —
  `RegimeProtocolPayload` (algorithm key ≠ resolved id; the registry's
  pinned parameters hashed into the identity; the P1-B panel grain in the
  ACTUAL schema: `CONTEXT_BAR_PANEL` requires interval/source/as-of
  fields, `CANDIDATE_STAGE_ROW`/`DECISION_ROW` refuse them; a 64-hex
  bundle reference; pinned seed 7; `fit_scope="per_training_fold"`),
  role-free `RegimeFitPayload` (verified non-empty source artifact ids +
  the training-feature-matrix hash — different feature values are a
  different fit identity), `RegimeCapabilityAssessment`
  (coverage/occupancy/stability over EXACT fit + fold identities with the
  applied gate values), and `RegimePromotionDecision` — the contract
  itself enforces one ladder step at a time, a chained
  `previous_decision_ref`, an ISO-8601 decision time, a 64-hex owner
  ratification reference from `FEATURE_ELIGIBLE` onward, and the ROLE
  ladder (execution-side roles are unrepresentable in V1 with the exact
  S11 reason); it never touches a numerical identity, and
  `persist_regime_promotion` re-checks the ladder against the referenced
  assessment's own `gates_passed` (a FEATURE_ELIGIBLE decision over a
  failing or absent assessment is unpersistable). `assert_no_regime_leakage`
  refuses every outcome/label/payout column and — from the feature-block
  registries, by default — every feature available only after the
  protocol's observation stage; every input must belong to the referenced
  bundle; every scientific default (KMeans baseline, k=3, the
  occupancy/rows/AMI gates, the sample-adequacy minimums, the grain
  baseline) is stamped `proposed_protocol_default` in
  `REGIME_PROPOSED_DEFAULTS`.
- **Algorithm registry** (`regime_algorithms.py`): `kmeans_v1` is the ONE
  implemented V1 algorithm (pinned k-means++ / n_init 10 / lloyd / seed 7);
  `minibatch_kmeans_v1`, `gaussian_mixture_v1`,
  `spectral_clustering_train_only_v1` (train-fold-only affinity,
  observation/memory caps, forced `BLOCKED_NO_OOS_ASSIGNMENT`, the
  mandatory training-only warning text), `nystrom_kmeans_v1`, and the
  surrogate-assignment entry are registered PLANNED with fail-closed
  refusals carrying the exact status/reason (P1-C) — no spectral or
  Nyström fit is callable anywhere in V1, and every planned protocol
  POLICY (`inner_train_only_selection`, PCA, kernels, non-centroid OOS
  policies) is refused by `assert_protocol_executable` before any
  preprocessing or fit.
- **Fold-local service** (`regime_preprocessing.py` · `regime_service.py` ·
  `regime_alignment.py` · `regime_diagnostics.py`): the fixed per-fold
  pipeline (median imputer + indicators → optional train-fitted p01/p99
  winsorizer → standard scaler) whose fit API accepts a FOLD and slices
  internally (no row-subset parameter; all-missing training rows never
  enter a fit; duplicated observation keys refuse); per valid fold the
  pinned KMeans fits on training rows and assigns train+test rows
  deterministically (distances to every centroid, margin d2−d1, the
  observation's as-of timestamp); rows are PRESERVED with typed reasons
  (`source_feature_missing`/`fold_invalid`/`coverage_gap`); Hungarian
  alignment is reporting-only over NOMINAL, geometry-ranked canonical ids
  in the scaled input-feature space with the exact ascending-local-id
  tie-break (prediction-hash invariance test-proven); the stability report
  carries seeded bootstrap aligned-AMI, per-cluster agreement, the
  OUT-OF-SAMPLE timeline's temporal persistence and transition matrix
  (ordered by observation timestamp within each fold), fold-to-fold
  recurrence, centroid separation, and descriptive-only silhouette; the
  sample-adequacy gate (≥150 candidate-stage/decision-row, ≥300 panel
  training rows per fold) and the occupancy/rows gates block PROMOTION —
  k is never shrunk. The panel→candidate assignment consults only
  out-of-sample assignments of the last COMPLETED bar at or before each
  candidate's as-of instant (lowest fold wins; 5m/15m tested end to end);
  earlier candidates and bars without an OOS assignment are typed
  `coverage_gap`.
- **Persistence** (`regime_store.py` + four new store names on the
  manifest protocol): dual-format fitted artifacts (canonical JSON
  parameter payload hashed as `fitted_parameter_payload_hash` + joblib)
  bound to their own assignment frame and verified from the exact bytes
  BEFORE publication (reload → re-transform `np.allclose` → re-predict
  equals the persisted labels), then re-verified through the store;
  manifest-relative references only and relocation-proof reload (P1-4);
  the shared `load_sidecar_bytes` verifies the manifest hash, whitelists
  the sidecar name, and hashes the returned bytes; assessments and
  promotion decisions persist as content-addressed envelopes — a
  promotion provably changes no fit identity.
- **Regime Lane UI** (`scripts/ifvg_regime_panels.py`, mounted as a
  pipeline-surface expander beside the MBP-1 panel): the algorithm
  registry with planned entries visible-disabled and the mandatory
  spectral warning; the proposal-stamp table; the exact-ID model card
  (grain identity incl. panel fields, input bundle, fixed-k stamp,
  OOS/alignment policies, coverage + per-fold coverage, fit identities,
  nominal-id occupancy, stability with per-cluster agreement and centroid
  profiles, the transition matrix, and the insufficient-sample blocked
  state), the exact-fit-id assignment / stratification / OOS-timeline
  view, and the promotion role/status view — read-only (no promote,
  launch, rank, or retrain control exists; the UI never unpickles).
  Stratified RESULT views (metrics by regime) are the ML §5.5 comparison
  classes and land with their studies. `IFVG_REGIME_CONTEXT_V1` remains a
  planned feature block and S11 remains blocked: no regime output can
  reach a predictive bundle or execution surface in V1.

R5B.1 correction (MBP-1 source-coverage policy v2 — owner planning decision
Q1, 2026-08-28; plan-review correction 4; final closure #2/#7):

- **The R5B rule "positive raw venue sequence jump > 1 = source gap" is
  WITHDRAWN and unrepresentable** (`Mbp1SourceContract.gap_semantics` is the
  Literal `mbp1_source_coverage_declared_evidence_v2`;
  `sequence_jump_semantics = sequence_jump_diagnostic_only_v2`). Databento's
  `sequence` is the venue's original channel sequence and `mbp-1` emits only
  top-of-book-changing events, so symbol-level continuity is never implied:
  raw sequence jumps / resets and `ts_recv` spacing are DIAGNOSTICS
  (`Mbp1SequenceJumpDiagnostics`, `Mbp1TsRecvGapDiagnostics`) — never an
  interval, never a missing reason, never a reduced coverage.
- **Evidence-based coverage** (`ifvg/features/mbp1_coverage_evidence.py`):
  every claim carries an explicit `Mbp1EvidenceScope` (dataset / publisher /
  channel / instrument partition, physical partition key, UTC date, the
  VERIFIED expected partition span). Accepted evidence kinds: a
  partition-scope declared gap manifest (`Mbp1PartitionGapManifest`), the
  vendor `F_MAYBE_BAD_BOOK` flag (DBN bit 4 — a CHANNEL-gap warning), and
  the dataset-condition record. Positive completeness exists ONLY through
  `compile_mbp1_partition_gap_manifest` over a verified
  `Mbp1CompletenessCompilationReport` (source inventory + owner review) —
  never a bare boolean. Dataset conditions map to
  `vendor_no_known_dataset_issue` / `vendor_dataset_degraded` /
  `vendor_dataset_pending` / `vendor_dataset_missing` /
  `vendor_condition_unavailable`; they can downgrade a scope, never prove
  a partition complete.
- **`F_MAYBE_BAD_BOOK` semantics**: the uncertainty starts at a
  manifest-declared start, else the last TRUSTED in-scope event (no
  bad-book / bad-`ts_recv` flag), else the partition's expected start —
  never automatically the detection row; it closes ONLY at a documented
  recovery boundary (manifest-declared end, documented vendor recovery
  event, documented snapshot recovery, owner-approved boundary) and
  otherwise runs to the partition end (`open_uncertainty_to_partition_end`,
  fail closed) — the next unflagged row never closes it. The interval is
  channel-scoped only behind a VERIFIED publisher/channel map; otherwise it
  expands conservatively to the publisher/physical partition and is never
  narrowed to the flagged instrument.
- **Coverage calculation (D12)**: per physical partition, denominator =
  `intersection([partition_expected_start_ts, partition_expected_end_ts],
  authorized_session_span)` (18:00 ET previous day → 17:00 ET, DST-aware);
  intervals are unioned, merged once, and clipped to that span; `coverage =
  1 − union_gap_ns / physical_expected_span_ns`; physical partition spans
  are half-open and must be pairwise disjoint after clipping (overlaps
  refuse); multiple UTC partitions of one trading day are measured
  separately and duration-weighted (the weakest partition status wins);
  head/tail gaps count only when declared; an empty span is unknown; an
  in-session event outside every declared span refuses the build; session
  sub-spans no partition covers are uncovered intervals whose windows type
  `coverage_evidence_unavailable`. A positive completeness claim must name
  the partition content it certifies (report `verified_partition_refs` ∩
  partition content hashes), and a provenance label without a manifest is
  unrepresentable. `completeness_status ∈ {evidenced_complete,
  declared_gaps, completeness_unknown}`; a partition without
  partition-scope evidence — or under a downgrading dataset condition — is
  `completeness_unknown` and every window of that day is typed
  `coverage_evidence_unavailable`. A window intersecting a merged verified
  interval is typed `declared_source_gap` (never widened, never imputed).
- **Real read seam**: `read_mbp1_partition_frame` clips the UTC-date file to
  the trading day's authorized session span and to `DEVELOPMENT_CUTOFF_UTC`
  BEFORE normalization or hashing (the protected 18:00 ET tail of the last
  exposed day is structurally excluded); the clipped counts and the raw
  file's sha256 ride the partition row. The previous UTC file's evening
  portion is not composed in R5B.1 (recorded deviation).
- **Re-minted identities**: the normalized event schema retains
  `publisher_id` + `flags` (new schema hash → new source artifact ids);
  `IFVG_ORDER_FLOW_MBP1_V1` is RE-RESOLVED as a second versioned event
  (`with_reresolved_block`: block v3, `ifvg_order_flow_mbp1_formula_v2` /
  `mbp1_feature_materializer_v2`, new resolved block id, new registry hash,
  new B2/B3 bundle ids; the R5B activation payload keeps the historical v1
  versions and the R5B state stays exported as `PRE_R5B1_*`); feature
  artifacts, coverage reports (`coverage_policy_id`), and controlled-study
  ids all move. Synthetic coverage evidence (`provenance =
  synthetic_fixture`) is lawful only under the synthetic marker — the
  pipeline's S05 and the real source builder refuse it.
- **Bounded real-data diagnostic** (`ifvg/features/mbp1_coverage_diagnostic.py`,
  `scripts/ifvg_mbp1_coverage_diagnostic.py`): characterizes per-partition
  row/flag counts, sequence-jump and `ts_recv`-gap distributions, the
  clipped-row counts, and — only through store-verified partition manifests
  supplied by `--evidence-json` — the open-interval facts, which manifests /
  reports / condition records were loaded, and the policy-v2 completeness
  status (an evidence-less partition reports `open_uncertainty_to_partition_end
  = None`, never a misleading `False`); writes an immutable
  `Mbp1CoverageDiagnosticReport` into the verification namespace; runs only
  under the R1 real-slice gate — a `VerificationReplayPolicy` over exactly
  the requested days, a persisted verified `VerificationRunEnvelope` +
  `VerificationAuthorizationRef` binding the allowlist hash and a verified
  coverage-matrix artifact, the one canonical program allowlist, and a
  store root ending in `search_test/v1` (fail-before-path — the real run is
  an owner action); never infers completeness from sequence continuity.
  Real MBP-1 research and R5B acceptance stay blocked until policy v2
  passes on the real fixture.

R6.1 additions (the regime-lane correction release — owner audit of R6,
planning decisions Q1–Q4, plan-review corrections 1–7, final contract-closure
rulings 1–8; `QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R6.1-CORRECTION-PLAN-DOCS-FINAL/`):

- **Context-bar panel materializer + block** (`ifvg/features/context_bar_panel_{contract,materializer}.py`,
  `arrow_tables.py`): the seven registered `cbp_*` features (owner Q3) are
  materialized ONLY from a `VerifiedReplayChartArtifact` (bars re-read and
  rehashed against the manifest; the in-memory frame is never trusted) at the
  two owner-registered intervals (300 s, 900 s) over COMPLETED bars; N = 12
  prior same-trading-day bars, `MINIMUM_SOURCE_BARS = 13`, `STD_DDOF = 0`;
  every one of the 13 source bars must satisfy `observed_1m_count ==
  expected_1m_count` or ALL seven features are null with
  `source_bar_incomplete` and a manifest-listed validity sidecar names the
  offending bars; windows never cross 18:00 ET; the session OF A BAR is the
  session in force at the bar's final instant (`session_of_bar_final_instant_v1`).
  `IFVG_CONTEXT_BAR_PANEL_V1` is registered as a versioned event
  (`with_registered_block`; AVAILABLE, family `context_bar_panel`, stage
  `HTF_TAP`, `research_only_offline`, `offline_panel_materialization_v1`,
  `row_id` join, categorical `cbp_session_state`); bundle
  `BP0_CONTEXT_BAR_PANEL` (no base; a candidate view refuses it); the R5B.1
  state stays exported as `PRE_R6_1_*`; `feature_block_registry_hash()`
  moves. Protocol tightening: `panel_source_artifact_id` is a 64-hex verified
  artifact id, only registered intervals/as-of policies, and grain/bundle-key
  coherence (`assert_grain_bundle_coherent`: every input-bundle block joins on
  `row_id` for the panel grain and on `candidate_id` otherwise).
- **Fold schedules and fold-set artifacts** (`ifvg/fold_schedules.py`,
  `ifvg/ml/fold_set_artifact.py`): `FoldScheduleEnvelope.fold_schedule_id`
  hashes the grain-agnostic 40/5/5/2 windows over the authorized days;
  every fold set (candidate or panel) is a persisted artifact carrying the
  schedule id + its own row-population `fold_set_id` (the ONE legacy hash,
  now delegated to by the ladder and the controlled study); cross-grain
  studies require equal `fold_schedule_id` and per-fold windows, never equal
  `fold_set_id`; the label-free `build_candidate_folds_from_schedule` and the
  panel-native `build_context_bar_panel_folds` (embargo 2 days, purge by bar
  span, stamped floor 300) exist beside the unchanged labeled builder.
- **Two regime-assignment artifacts, two uses (D7)**: the DESCRIPTIVE
  `RegimeOosAssignmentArtifact` (`ifvg/ml/regime_oos_assignment.py`; candidate
  grain = the single OOS fold; panel grain = the normative PIT rule — same
  trading day, last completed bar `<=` as-of, `cbp_valid`, elapsed ≤ one
  interval, compatible partition; typed `panel_warmup` /
  `no_completed_panel_bar` / `panel_gap` / `panel_stale` / `coverage_gap`;
  never carried across 18:00 ET) feeds stratification only; the
  `RegimeFoldFeatureArtifact` (`ifvg/ml/regime_fold_features.py`; fit k → fold
  k only; fit-local `ctx_regime_<p12>_local_{distance_i,assigned_distance,
  margin,id}` — the local id is a model feature only under
  `hard_id_encoding = fit_local_categorical_v1`, default `none`; canonical
  alignment reporting-only) is the ONLY supervised feature source — the leakage test proves later folds cannot alter an
  earlier fold's bytes.
- **Verified observation seam + executor** (`ifvg/ml/regime_observation_source.py`,
  `regime_executor.py`): every regime input is a verified-loaded artifact
  (`bundle_feature_views` / `context_bar_panels` stores); `source_artifact_ids`
  come from the loaded envelope, never a caller string; `execute_regime_protocol`
  persists protocol → loads the fold set → loads observations → runs →
  persists fits by VERIFIED REUSE BY REPRODUCTION (an existing fit must
  reproduce this fit's transform and labels from its verified bytes; joblib
  bytes are never rewritten) → assessment → the descriptive OOS artifact.
  The panel-grain candidate as-of instants come from a verified-loaded
  bundle view (`candidate_as_of_source: RegimeObservationSourceRef`; a
  tuple / bare string / panel-kind ref is "not evidence") and
  `candidate_as_of_source_ref` is the LOADED envelope line
  (`bundle_feature_view:<id>`, pattern-checked).
  `run_regime_protocol` executes under `threadpool_limits(1)` so assignment
  tables are byte-reproducible (an execution-environment control, not a
  protocol field; fit ids unchanged — the R6 golden fit id holds;
  `threadpoolctl` is imported lazily with an explicit failure message).
- **Pipeline integration (D1/D4/D14)**: `PipelineSemanticSpecPayload.regime_study`
  (`ifvg/ml/regime_study.py::RegimeStudyRequest`; every semantic id moves)
  freezes algorithm, grain, bundle, inputs, k, winsorization, bootstrap
  budget, the requested comparison classes and — for `feature_only` /
  `cohort_model` only — the EXACT `regime_promotion_decision_id`,
  `owner_decision_artifact_id`, and `required_capability_assessment_id`;
  descriptive requests must omit them. Stages (`ifvg/search/pipeline_regime.py`):
  S04 records the chart ids (one replay chart per child); S05 persists every
  bundle view + frame and the regime observation (the panel materialized
  from `PipelineWiring.context_bar_source` for the chart selected by the
  declared policy `panel_chart_lowest_core_replay_id_v1`; the returned
  artifact must carry the requested id and the run's pair or S05 fails
  closed; the sidecar records the LOADED chart id) + the protocol (a
  model-bearing run verifies it against the frozen decision); S06 the input
  coverage + stamped floors; S08 the schedule derived ONCE from the
  candidate view's observed trading days (charter allowlist ∩ observed;
  `days_without_labels` typed) + fold-set artifact(s) built from those same
  days + the per-fold adequacy preview; S09a the executor (a
  model-bearing run refuses any assessment other than the frozen one); S09b
  the fold-local features and S09c the controlled regime study / cohort model
  (`ifvg/ml/regime_supervised_stage.py`) only for model-bearing requests — every
  fit of the run happens in S09; S10 derives the deterministic decisions
  (DESCRIPTIVE_ONLY or a typed BLOCKED state, then STRATIFICATION_READY iff
  coverage gates passed AND an OOS assignment exists AND descriptive classes
  were requested; `decided_at` = the evidence as-of instant, never the wall
  clock) or re-verifies the frozen authority; S14 builds the stratified
  reports from persisted artifacts only (the panel PIT assigner loads the
  OOS assignment → protocol → panel frame → every fit's assignment sidecar
  by exact id; the executed-trade tables are S02's this-run tables — no
  persisted trade artifact exists) and performs ZERO fitting (test-enforced
  on the descriptive, supervised-candidate and supervised-panel runs); the
  modeled classes are recorded under `delivered_by` (verified S09c ids),
  never as refusals; S15 reloads every regime artifact. A reused child whose
  executed-trade tables a stratified report needs is re-derived and its
  tables are adopted ONLY when they reproduce the persisted costed
  evaluation of THIS cost policy (otherwise "reproduction unverifiable":
  reused, no gates evaluated, no evaluation published, out of the
  stratified reports); S12/S13 record the exact persisted
  account-simulation ids in an attempt-invariant sidecar. Readiness
  verified-loads a model-bearing request's frozen authority from the store
  before any path — the tab's Preview and Launch handler do the same with
  the store root and run scope BEFORE any charter / spec envelope or spawn.
- **Verified owner-decision evidence (D5)** (`ifvg/search/owner_decisions.py`,
  store `owner_decisions`): the artifact binds the EXACT protocol and
  assessment and states decisions **25/28/29/30** (algorithm key + the exact
  pinned KMeans parameter snapshot/hash; grain / interval / stage; k;
  occupancy / rows / sample floors and the stability minimum) — every value
  re-verified against the registry, protocol, and assessment; supersession is
  store-owned and hash-chained (`SUPERSESSIONS.jsonl` + `SUPERSESSIONS.head`;
  the line precedes the replacement's publication; every line backed by a
  verified replacement; any edited / deleted / reordered / dangling line
  fails closed; provenance may never weaken); synthetic provenance is lawful
  in the `synthetic_fixture` run scope only, and that scope is confined to
  test namespaces (`assert_run_scope_lawful_for_root`, the P0-4 mirror:
  synthetic-provenance artifacts are refused at persist AND at load in a
  research root); `persist_regime_promotion` requires the artifact from
  FEATURE_ELIGIBLE onward (a bare 64-hex reference is not evidence),
  structurally requires passing coverage gates + an OOS assignment for
  STRATIFICATION_READY (D6; re-derived from the loaded assessment at the
  report gate, whose owner branch re-runs the full authorization under the
  run scope), refuses `MODEL_FEATURE` in V1, and derives `decided_at` from
  verified artifacts (no flag, no wall clock; the CLI's STRATIFICATION_READY
  reuses S10's exact decision).
  Draft proposals (`DRAFT_OWNER_DECISION_PROPOSALS/`) carry placeholders that
  cannot persist; `scripts/ifvg_regime_promotion.py` is the ratification CLI.
- **Status-gated block activation (D9)** (`ifvg/ml/regime_block_activation.py`):
  `IFVG_REGIME_CONTEXT_V1` activates as a pure versioned event bound to the
  exact frozen FEATURE_ELIGIBLE decision, owner artifact, assessment,
  protocol, and fold-feature artifact (`as_of_policy =
  fold_local_fit_partition_assignment_v1`; join keys candidate/fold); the
  module registries stay PLANNED at import; bundle `B7_CORE_REGIME`.
- **Stratification classes (§5.5)** (`ifvg/ml/regime_stratified_{contracts,strategy,prop,frontier}.py`,
  `regime_stratification_{gate,service}.py`, `regime_assignment_sources.py`):
  `cohort_descriptive` / `stratified_prop` / `stratified_frontier` need
  STRATIFICATION_READY; `feature_only` / `cohort_model` need FEATURE_ELIGIBLE
  and are S09c deliverables (`regime_controlled_study.py`,
  `regime_cohort_model.py`); reports bind the exact decision / assessment /
  fit ids, the descriptive OOS artifact, and the executed-trade table hash;
  `RegimeFilterRef` is consumed (one cohort per stratum); thin strata are
  typed at the stamped 20 trades / 60 training rows; the frontier view is
  never a selection input; prop events attribute by source trade, then PIT
  for historical no-trade events, never a synthetic clock
  (`source_trade_then_pit_v1`), ordered by the D15 `EVENT_TYPE_PRECEDENCE`
  and dated by the event's own `trading_day`; the `stratified_prop` report
  carries the bounded report-local `account_event_regime_summary.parquet`
  (schema v1; one row per simulation × path × regime stratum / typed reason
  × event type; keyed by the exact regime-assignment evidence; budget
  `account_event_regime_summary_budget_v1` = 5,000,000 rows / 256 MiB,
  typed refusal before publication; four envelope extras — never per-event
  JSON); "nothing here promotes".
- **Prop-event detail (D15)** (`alpha_lab/propsim/event_detail.py`):
  `event_detail_persistence_policy_id ∈ {none_v0, account_event_detail_by_path_parquet_v2}`
  + storage policy / schema version / budgets enter the account- and
  portfolio-simulation identities and `SimulationProtocol` (default `none_v0`;
  every simulation/charter/pipeline id moves — pre-acceptance evolution);
  under v2 literally streaming ZSTD Parquet partitions (one path block at
  a time through the store's sidecar-producer protocol — `ProducedSidecar` /
  `SidecarProducer`; memory = one block + a 32-byte-per-row event-id index)
  by `path_block_id = floor(path_ordinal / 250)`
  with exact event time / trading day / clock policy / total-order fields;
  budgets (10,000,000 rows; 2 GiB; block 250) fail BEFORE atomic publication;
  `none_v0` artifacts are never widened.
- **Bundle-aware CatBoost rung (D13)** (`ifvg/ml/catboost_bundle_model.py`,
  `comparison_rows.py`): `ifvg_context_catboost_bundle_v1` (AVAILABLE, kind
  `nonlinear_challenger_bundle`; frozen-lane parameters by value; native NaN
  + `MISSING_CATEGORY`; registry ∪ block-declared categoricals) runs on both
  arms of every controlled comparison; `comparison_row_id = hash{fold_schedule_id,
  candidate_fold_set_id, fold_index, candidate_id, label_artifact_id}` keys
  the identical-rows gate and every paired delta (the legacy `oos_row_id`
  stays for M0–M3 compatibility); the frozen M0–M3 CatBoost lane is byte- and
  identity-unchanged (golden protocol hash).
- **Stability / transitions (D10/D11)**: bootstrap on EVERY valid fold
  (per-fold seeds; fold 0 reproduces R6); the promotion gate
  `minimum_bootstrap_aligned_ami_mean` (0.5) applies to the protocol-wide
  minimum fold mean; candidate-event transitions reset on trading day /
  named session / a stamped 7200 s gap; panel transitions count consecutive
  completed bars within a trading day.
- **Evidence hygiene**: the two provider-key tests are hermetic
  (`monkeypatch.delenv`); the R6.1 browser manifest v2 binds screenshots to
  the final commit (`verify_browser_manifest.py`); each release folder
  carries a `git format-patch` source-review patch + sha256; two independent
  read-only adversarial reviews (contract fidelity; safety/access) — 28
  findings, every one dispositioned in
  `R6.1/ADVERSARIAL_REVIEW_RESOLUTION.md`.
- **Trust boundary (stated in-tree; DEV-R6-8)**: the manifest protocol verifies
  INTEGRITY, not authenticity — a store root is a trusted local directory;
  the UI never unpickles; signing / a non-pickle fit serialization remains a
  post-V1 hardening candidate. `IFVG_REGIME_CONTEXT_V1` reaches a predictive
  bundle only through the status-gated activation above; S11 stays blocked;
  no regime output reaches an execution surface in V1.

R6.1-FIX additions (the compact correction of R6.1 after independent review —
plan `QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R6.1-FIX-HARDENING-READINESS-PLAN/IMPLEMENTATION_PLAN.md`
revision 3, Phase 1; findings F-01…F-10D):

- **Verified assignment evidence (F-01/F-02/F-05)** (`ifvg/ml/regime_contracts.py`,
  `regime_store.py`, `regime_oos_assignment.py`, `regime_executor.py`): the per-fit
  assignment sidecar is serialized under the ENFORCED `FIT_ASSIGNMENT_SCHEMA`
  (exactly `RegimeAssignmentColumns`; hash `FIT_ASSIGNMENT_SCHEMA_HASH`) and every
  table satisfies `validate_assignment_rows` (kinds `fit` / `descriptive` /
  `model_facing`: a valid row carries the complete, self-consistent value set —
  64-hex fit id, fold, lawful partition, local id in `[0, k)`, `k` finite
  distances with `assigned_distance == distances[local] == min` and
  `assignment_margin == d2 − d1 ≥ 0`, the canonical id except on the
  model-facing kind — an invalid row keeps its linkage key and carries no output
  and one registered reason; no optional-column fallback survives). `load_regime_fit_assignments` returns
  `VerifiedFitAssignments` (envelope, artifact, frame, sidecar SHA-256, schema
  hash) and `persist_regime_fit` reuses an existing fit only when the candidate
  assignment bytes equal the stored sidecar byte-for-byte. The executor
  exact-loads every fit it persisted and builds the descriptive OOS artifact
  from those verified frames only; `RegimeOosAssignmentPayload` binds
  `regime_fit_assignment_refs` (`FitAssignmentRef` per fit, sorted; the
  `regime_fit_ids` projection is validated against them), `resolved_cluster_count`,
  `candidate_as_of_stage` (the anchor the hashed as-of instants came from; an
  unparseable non-null anchor is a hard error) and a `consulted_assignments_hash`
  over EVERY consulted value (fit, row, fold,
  partition, local id, canonical id, distance vector, assigned distance, margin,
  validity, reason); formula `regime_oos_assignment_v2`. `RegimeAssignmentEvidenceRef`
  pins `assignment_table_sha256` + `assignment_schema_hash` of the verified artifact.
- **Fold-feature source identity (F-03)** (`ifvg/ml/regime_fold_features.py`,
  `regime_supervised_stage.py`): `FoldFitRef` binds `assignments_sidecar_sha256`
  + `assignment_schema_hash` whenever a fit is present (null together only for an
  absent fit); `build_regime_fold_features(fit_assignments=…)` consumes
  `VerifiedFitAssignments` only (an in-memory run frame is refused by type) and
  S09b passes the executor's verified evidence; the loaders re-check every ref
  against the store by exact id; `validate_fold_feature_rows` holds on build, load
  and the ladder seam.
- **Candidate as-of policy (F-04)**: a candidate whose stage anchor is null is
  PRESERVED as `candidate_as_of_missing` (registered in
  `PANEL_ASSIGNMENT_MISSING_REASONS`, hence in the fold-feature vocabulary); an
  unparseable non-null instant stays a hard error; the as-of source hash
  represents the null deterministically.
- **Thin-regime accounting (F-09) + normalized frame (F-10B)**
  (`ifvg/ml/regime_stratified_strategy.py`, `regime_stratified_contracts.py`,
  `search/strategy_metrics.py`): the executed trades are validated + normalized
  ONCE and that projection (`normalized_executed_trades`) drives every join,
  stratum, computation and the binding `executed_trade_table_sha256`;
  `RegimeNetRAccounting` (formula `regime_net_r_accounting_v1`, basis
  `all_valid_assigned_trades_v1`) sums `per_trade_net_r` over EVERY valid assigned
  trade — thin regimes included — with `abs_net_r_share_by_regime`, signed
  contribution fractions, `unassigned_net_r`, `assigned_regime_count`,
  zero-denominator reasons and the `works_only_in_regime` claim (true; FALSE when
  the assigned side refutes it; null with `incomplete_assignment_accounting` only
  when unassigned trades prevent a supported claim), every derived value
  recomputed by the contract's validator; the reportability floor governs
  interval/reportability metrics only. The stratification service re-verifies
  each child's frame against its persisted executed-trade table and binds the
  artifact's projection hash (`executed_trade_table_artifact_sha256`).
- **Exact label identity (F-08)** (`ifvg/ml/comparison_rows.py`,
  `controlled_feature_study.py`, S07): `label_artifact_content_id` binds the
  registered label policy and EVERY consumed label / economic column
  (`LABEL_CONSUMED_COLUMNS`); S07 mints it; `ControlledFeatureStudyPayload.label_artifact_id`
  is mandatory with `label_identity_source ∈ {label_artifact, content_hash_unpersisted}`
  — a helper run without the exact artifact carries the full consumed-column hash
  and can never be saved, compared as an immutable study, or promoted; every
  helper path (the ladder, the CatBoost bundle rung, the logistic rung) defaults
  to that hash and the ladder run is stamped `label_identity_source`.
- **Immutable executed-trade table (F-06)** (`ifvg/search/executed_trade_table.py`,
  store `executed_trade_tables`): the EXACT ordered 42-column Arrow projection
  `EXECUTED_TRADE_TABLE_SCHEMA_V1` (`core_executed_trade_exact_v1`; typed as the v2
  capture types it) is declared, never inferred; the identity derives from the
  core replay (`executed_trade_table_id_for`), so S02/S14 exact-load it without a
  listing; the envelope binds the projection SHA-256, the raw core table hash the
  neutrality report hashes, row count and byte size. S02 persists it after a fresh
  completion and after verified reproduction (byte-for-byte against the persisted
  table — projection bytes AND raw core-table hash; the costed-evaluation
  reproduction remains the fallback for a child without a table; nothing
  verifiable → typed `executed_trade_table_unavailable`), exact-loads it back and
  computes EVERY costed evaluation from the loaded projection inside per-child
  containment (one identity, one byte content); S14 iterates the charter's child
  set (never a prior attempt's report record), verified-loads every gated child's
  table, binds `executed_trade_table_id` into the report body +
  `source_metric_refs`, and records `children_evidence` / typed `children_skipped`
  (`child_not_completed_or_reused`, `strategy_gates_not_passed`,
  `executed_trade_table_unavailable`).
- **Fail-closed prior-stage sidecars (F-07)** (`ifvg/search/store.py`,
  `pipeline.py`, `pipeline_regime.py`, `ml/regime_report_stage.py`): the typed
  probe contract — `probe_sidecar` / `has_sidecar` / `load_optional_sidecar_bytes`
  / `load_json_sidecar`; `sidecar_not_produced_for_path` is the ONLY optional
  absence; `store_entry_missing` (the entry directory does not exist),
  `manifest_missing_for_existing_entry`, `malformed_manifest`,
  `manifest_hash_mismatch`, `envelope_identity_mismatch`,
  `sidecar_missing_but_manifest_declares_it`, `sidecar_hash_mismatch`,
  `malformed_sidecar`, `unexpected_io_error` are typed `SidecarLoadError`s that
  propagate — raised at the detection point by `load_verified_envelope`,
  `load_sidecar_bytes` and `has_envelope` (corrupt is never absent). The prop-vector / account-simulation /
  lineage / regime-report / S09c-record recoveries use it; S15 records
  `reload_failures` (store/id → sanitized reason) in the state file's publication
  block and immutably as `PipelineResultPayload.reload_failure_reasons`, and
  reloads every executed-trade table and stratified report the S14 record names;
  a halted or cancelled attempt marks every later planned stage PENDING, resets
  the publication block, and `activate_pipeline_result` re-derives the gates from
  the latest attempt (no stale terminal status is carried forward).
- **Production correctness (F-10A/C/D)**: the five pipeline wiring checks raise
  `PipelineWiringError` (typed, survives `python -O`); the fold-feature builder's
  two asserts became typed errors; `Mbp1PartitionEvidence` requires FULL scope
  equality with the gap manifest and a positive completeness claim requires the
  compilation report's `verified_partition_refs` to EQUAL the complete partition
  content refs (`partition_content_refs=None` is refused for a positive claim);
  `FrozenContract.model_copy` refuses a raw string for a scalar enum-typed field
  and a non-member element in a sequence-of-enum field.
- **Unchanged (golden-tested)**: `resolved_regime_protocol_id`, the R6 golden
  `regime_fit_id`, the frozen M0 CatBoost hash, `core_replay_id`,
  `account_simulation_id`, `feature_block_registry_hash`
  (`tests/agents/ifvg_search/test_r61_fix_goldens.py`). Re-minted (synthetic only):
  OOS-assignment, fold-feature, bundle-path ladder / study, stratified-report,
  controlled-study, label-artifact and pipeline-result identities. Verified reuse:
  zero replay for non-stratified runs; one verified reproduction per reused child
  (projection bytes + core-table hash) when stratified reports are requested.

HARDENING-BACKEND additions (Phase 2 backend hardening + Phase 3 / Phase 4 contract
authoring of `QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/`
`R6.1-FIX-HARDENING-READINESS-PLAN/IMPLEMENTATION_PLAN.md`
revision 3; findings F-11 F-12 F-13 F-16 F-17 F-18 F-20 F-21 F-22; no owner action taken,
no real seed replay, no real ≤5-day run):

- **Semantic store namespace (F-11)** (`ifvg/search/store_namespace.py`,
  `scripts/ifvg_store_namespace.py`): research-versus-test authority is the store's
  immutable, re-verified `STORE_NAMESPACE.json` envelope (`namespace_class`, a stable
  `store_instance_id` that is never a path hash, the supersession genesis anchor) — never a
  pathname; every owner decision, supersession record, authorization bundle, verification
  and seed-production authorization binds the `store_namespace_id`; an unmarked store has no
  authority; the one-time explicit `init` migration states the class; the old pathname
  heuristic survives as a deployment defense-in-depth check only.
- **Immutable supersession chain + head witnesses (F-12)** (`ifvg/search/supersession_chain.py`;
  store `owner_decision_supersessions`): one content-addressed record per replacement, a
  mandatory head (`owner_decisions/SUPERSESSIONS.head`) whose digest commits to the whole chain
  from the genesis anchor, four-step atomic publication under the lock (an orphan record has no
  authority; the head only ever names a verified record), idempotent identical replay, refused
  divergent replay; every real charter / authorization records the current
  `{store_namespace_id, line_count, head_sha256}` witness and a missing, shorter or different
  current head is refused (local rollback detection, not cryptographic authenticity).
- **Liveness-aware owner-decision lock (F-13)** (`ifvg/search/owner_decision_lock.py`): pid,
  process-start token, random token, host, heartbeat; reclaimed only when the heartbeat timed
  out AND the holder is demonstrably dead (no such pid / exited / PID reuse); a live holder,
  another host or a malformed body is never reclaimed; the writer re-verifies its token before
  publication and a lost lock aborts; release unlinks only its own token.
- **Capacity (F-17)** (`propsim/event_detail.py`, `propsim/search_bridge.py`,
  `ifvg/ml/regime_stratified_prop.py`, `scripts/hardening_capacity_benchmark.py`): the
  event-detail writer streams an iterable of walk pairs (never materialized), keeps no
  whole-artifact id index (canonical-key argument + an unconditional disk-backed DuckDB
  distinct check over the written partitions), and the regime-stratified event summary
  aggregates exactly through DuckDB over intermediate Parquet partitions under an explicit
  memory limit and attempt-local temp directory with canonical ordering; the
  `HARDENING_CAPACITY_POLICY_V1` benchmark (native RSS) passed every §4.4 gate at 250k/500k/1M
  rows (`CAPACITY_BENCHMARKS.md`).
- **Warning policy (F-18)**: `pyproject.toml` runs the suite under `filterwarnings = error`
  with ONE exact third-party rule (the scikit-learn 1.7 / SciPy 1.16 L-BFGS-B deprecation);
  `dataset.concat_schema_aligned` replaces the deprecated concat with explicit dtypes (frozen
  bytes unchanged; all-null columns never dropped); project-owned warnings = 0.
- **Sequential execution truth (F-20)** (`ifvg/search/pipeline.py`, `scripts/ifvg_pipeline_job.py`):
  `SUPPORTED_CHILD_WORKERS = 1`, `execution_mode = sequential_children_v1`; `WorkerPolicy`
  refuses `max_workers != 1` with the typed reason `unsupported_worker_parallelism_v1` (never
  coerced), the job shim refuses before job creation, and every attempt receipt persists
  `effective_workers=1` / `execution_mode`.
- **Phase 3 contracts (F-16 / F-21 / F-22)** (`ifvg/search/trading_calendar.py`,
  `verification_window.py`, `seed_production.py`; `scripts/ifvg_verification_window_shortlist.py`,
  `scripts/ifvg_seed_production.py`; stores `seed_production_authorizations`,
  `seed_production_runs`): a logical trading day is the Strategy-Core trading-day id whose
  stream is `[td−1 18:00 ET, td 18:00 ET)` over the physical partitions `(td−1, td)`
  (`cme_globex_18et_weekday_v1`; physical Sunday partition dates are not trading days); the
  coverage shortlist was rebuilt from already-authorized evidence on consecutive logical days
  under the plan's lexicographic ranking (no owner selection, no allowlist registration — the
  June proposal is INELIGIBLE as stated: no exact verifier target); the separately authorized
  seed-production lane (`SeedProductionReplayPolicy`, authorization / run contracts that bind
  the namespace + head witness, profile, store-day chain, source-inventory hash and code
  identities; permitted outputs = seed snapshot + access audit + run receipt) is proven
  synthetically; the owner packets are unsigned and their placeholders fail validation.
- **Phase 4 authoring** (`ifvg/search/bounded_verification.py`,
  `scripts/ifvg_bounded_verification.py`; stores `r1_baseline_gate_reports`,
  `bounded_release_control_flow_reports`): the typed §6.1
  preflight (namespace, witness, real authorization, 1–5 consecutive logical days, physical
  mapping, program allowlist, seed) before any path; the immutable `R1BaselineGateReport`
  (both attempts' six gates + the audit-mode digests + the eight "also prove" proofs) and the
  release-specific `BoundedReleaseControlFlowReport` (eight components typed from the persisted
  pipeline state — never research evidence); the runner refuses `fail_before_path` without the
  owner's persisted authorization (proven against the real, run-less verification store).
- **Unchanged (golden-tested)**: `resolved_regime_protocol_id`, the R6 golden `regime_fit_id`,
  the frozen M0 CatBoost hash, `core_replay_id`, `account_simulation_id`,
  `feature_block_registry_hash`, the `B0_CORE` bundle id. Re-minted (synthetic only):
  owner-decision artifact ids (`store_namespace_id`), verification-run ids and charter ids that
  carry a real bundle (namespace + witness), execution-attempt receipts.

HARDENING-BACKEND-FIX additions (the compact backend correction of
`QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/HARDENING-BACKEND-FIX/IMPLEMENTATION_PLAN.md`
— ten corrections, nothing else; no owner action taken, no seed production, no real ≤5-day
run; `backend_dev_complete_for_ui = true`, acceptance still transitively blocked by R1):

- **Token-safe stale-lock reclamation** (`ifvg/search/file_mutex.py`, `owner_decision_lock.py`):
  reclamation runs under a private standard-library cross-process mutex (`msvcrt` byte range /
  `fcntl.flock`; never unlinked; carries no authority); the stale body is re-read and re-evaluated
  under the mutex and unlinked ONLY when byte-identical to the dead holder observed; a persistent
  read failure is the typed `lock_read_failed` (never absence); `release()` raises
  `lock_release_failed` when it cannot verify its own lock; a failed body write removes the partial
  exclusive file.
- **Atomic, recoverable namespace initialization** (`ifvg/search/store_namespace.py`): the
  envelope and the genesis head are published as one pair through temporary files and
  verified-loaded together under a one-time init mutex; a half-initialized store is recovered only
  by the identical request (class + explicit instance id, no supersession records) and is otherwise
  the typed `incomplete_store_namespace_initialization`; conflicting bytes are never overwritten.
- **Public source-kind boundary** (`ifvg/search/trading_calendar.py`): the public `SourceKind` is
  exactly `mbp1` / `trades` / `legacy_verified_replay_source`; a historical physical partition
  resolves to the opaque legacy value through the private physical-file resolver (an internal
  `PhysicalSourceDescriptor` keeps the truthful file name, era, hash and partition key and implies
  nothing beyond replay bytes); every inventory, window ref and the seed inventory hash refuse the
  physical stem.
- **Exact regime provenance and native validation** (`ifvg/ml/regime_contracts.py`,
  `regime_oos_assignment.py`, `regime_fold_features.py`, `regime_assignment_sources.py`): the
  descriptive OOS assignment keeps three-way semantics (valid; invalid with its applicable fit /
  fold / partition and the fit's typed reason; `no_oos_assignment` only for a candidate with no
  OOS test row) on the candidate grain, the panel PIT rule, the fold-feature spine and the
  executed-trade projection; every assignment / fold-feature table is validated natively before
  any conversion (actual booleans, integral values, no numeric strings / infinity / sentinel
  identifiers, no `errors="coerce"`); the OOS payload binds the registered schema hash, the saver
  and the loader decode the Arrow bytes and prove schema / count / uniqueness / row invariants, and
  the candidate-as-of and assignment sets are exactly equal.
- **Fail-closed manifests; exact label and executed-trade evidence** (`ifvg/search/store.py`,
  `ifvg/ml/comparison_rows.py`, the three study runners, `regime_stratification_service.py`): one
  central manifest-entry validator is shared by every probe / load / reuse path (bare relative file
  names, lowercase 64-hex digests, non-negative byte counts, no duplicates / reserved names, the
  envelope entry exactly once; artifacts resolved and compared before opening —
  `sidecar_path_escape`; `invalid_store_locator` distinguished from absence); duplicate label
  candidates are refused before hashing; the pipeline's persisting study seams prove the label
  artifact derives exactly from the registered policy; the persisting stratification service
  requires every child's exact `executed_trade_table_id`.
- **Central seed canonicalization** (`ifvg/search/child_replay.py::save_seed_snapshot`): the one
  seam rebuilds every aware datetime (pytz / zoneinfo / fixed offsets) under the stdlib UTC
  tzinfo, leaves naive datetimes unchanged and rebuilds containers, so the same instants under any
  representation mint the same seed hash, snapshot id and sidecar bytes; the seed-production runner
  delegates to it.
- **Bounded event-detail partition** (`propsim/event_detail.py`, `propsim/search_bridge.py`,
  `scripts/hardening_capacity_benchmark.py`): `EVENT_DETAIL_BUDGET_V2` registers
  `max_rows_per_partition = 50,000` (the benchmark's measured row-group size; no ceiling lowered);
  the writer flushes at the bound even inside one path, partitions are keyed
  `(path_block_id, partition_ordinal)` with first / last event keys, rows, bytes, digest and schema
  hash, a refused build leaves no partition behind, the reader proves the bound and the total
  order; the V1 budget stays loadable but is refused by the writer; the benchmark's `normal` /
  `skewed` / `dense` shapes and the resident-batch gate passed at 250k / 500k / 1M rows.
- **Complete authority-chain proof** (`ifvg/search/owner_decisions.py::verify_complete_owner_authority_chain`):
  the ONE proof every real authority seam runs (charter freeze / load, pipeline launch, activation,
  executors, verification run, MBP-1 diagnostic, the seed-production and bounded-verification
  authorizations, the regime chain loader) — every record verified from the genesis anchor, every
  superseded and replacement decision verified-loaded, lawful transitions only, and the signed
  witness equal to the verified current head (`supersession_decision_unverifiable`,
  `supersession_transition_unlawful`, `supersession_chain_divergent`).
- **Unchanged (golden-tested)**: the Strategy-Core pin, the fixed M0–M3 lane, the R6.1-FIX goldens
  (`B0_CORE` bundle id, candidate protocol id, `core_replay_id`, `account_simulation_id`,
  `feature_block_registry_hash`), R5B formulas, model protocol parameters, KMeans fit identities,
  prop-firm rule contracts, the S11 blocked reason, `order_flow_depth_policy="mbp1_only_v1"`.
  Re-minted (synthetic only): inventories / windows / seed authorizations that serialized the
  physical stem, seeds created from non-UTC representations, simulations under the default (V2)
  event-detail budget, regime OOS / fold artifacts whose invalid rows previously collapsed.
