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
