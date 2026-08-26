# Architecture & Implementation Decision Log

Every significant decision is recorded here with context and rationale.
This prevents re-litigating settled questions across sessions.

---

## D-001: Pydantic v2 for Interface Contracts (not dataclasses)
**Date**: 2026-02-22
**Context**: Architecture spec used `@dataclass`. Needed to choose between dataclasses and Pydantic for the actual implementation.
**Decision**: Pydantic v2 BaseModel for all inter-agent contracts.
**Rationale**: Runtime validation at agent boundaries catches corrupted data. `model_dump()`/`model_validate()` gives free JSON serialization for message envelopes. `Field(ge=, le=)` constraints enforce value ranges. Negligible perf overhead since validation is at message boundaries, not in hot-path numerical loops.
**Trade-off**: Slightly more verbose than dataclasses. Worth it for safety.

---

## D-002: Single `contracts.py` File (not split by agent)
**Date**: 2026-02-22
**Context**: Could have placed each agent's contracts in its own directory.
**Decision**: All 13 Pydantic models live in `core/contracts.py`.
**Rationale**: Eliminates circular import risk (Execution imports from Validation's contracts, Validation from Signal's, etc.). At ~300 lines it's manageable. Single source of truth.

---

## D-003: `__init_subclass__` Auto-Registration for Signal Detectors
**Date**: 2026-02-22
**Context**: Needed a way to register 20 detector classes without manual registry maintenance.
**Decision**: `SignalDetector.__init_subclass__` hook populates `SignalDetectorRegistry` automatically when a subclass defines `detector_id`.
**Rationale**: Zero-boilerplate for signal authors. No decorators, no registry file to maintain. Python enforces it at class definition time. Import the detectors package → all 20 registered.

---

## D-004: Synchronous Message Bus (not async)
**Date**: 2026-02-22
**Context**: Could have used asyncio, threading, or message queues.
**Decision**: Synchronous in-process message routing via `MessageBus.send()` → direct function call to handler.
**Rationale**: Single-process research system. Sync is simpler to debug and test. The bus interface (`send`/`register_agent`) can be swapped to async later if needed. No premature complexity.

---

## D-005: `dict[str, Any]` for Heavy Data in Contracts
**Date**: 2026-02-22
**Context**: DataBundle.bars contains pandas DataFrames. Pydantic can't efficiently validate DataFrame contents field-by-field.
**Decision**: Use `Any` for heavy numerical payloads (bars, direction series, strength series). Pydantic validates the envelope only.
**Rationale**: DATA-001's quality checks validate data integrity. The QualityReport.passed flag signals downstream trust. Pydantic validating millions of OHLCV rows would be unusably slow.

---

## D-006: Repo-Local Git Config (not global)
**Date**: 2026-02-22
**Context**: Git needed user identity for commits.
**Decision**: Set `user.name` and `user.email` only for this repo (`git config` without `--global`).
**Rationale**: Doesn't affect other repos on the machine.

---

## D-007: Python 3.14 as Runtime
**Date**: 2026-02-22
**Context**: Project standard is Python 3.14.5 via `uv` / `.python-version`.
**Decision**: Install and run project dependencies with Python 3.14.5. In Quant-Lab environments, install Strategy-Core editable from `../Strategy-Core`.
**Rationale**: pyproject.toml requires >=3.14. The pinned `.python-version` keeps repo-local tooling on the same interpreter.

---

## D-008: Agent System Prompts Stored as Files — Superseded
**Date**: 2026-02-22
**Context**: Architecture spec defined system prompts inline. They exist in conversation history which compresses.
**Decision**: Store all 6 agent system prompts as `.md` files in `docs/agent_prompts/`.
**Rationale**: Survives context compression. Agents can load their prompts from disk. Prompts are the authoritative behavioral spec for each agent.
**Superseded 2026-06-06**: The prompt files were pruned as stale scaffold docs. Current Quant-Lab behavior is defined by code, tests, `AGENTS.md`, `ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`, and Strategy-Core v3 docs.

---

## D-009: Polygon.io as Data Vendor
**Date**: 2026-02-22
**Context**: Needed to choose a market data vendor for NQ/ES futures bars.
**Decision**: Polygon.io via `polygon-api-client` Python SDK. Uses `list_futures_aggregates()` endpoint.
**Rationale**: User already has a working Polygon integration in their `Claude-my-quant` project. Proven patterns for futures data, API key management, and caching. Minimizes integration risk.

---

## D-010: Skip Tick Bars (Time Bars Only)
**Date**: 2026-02-22
**Context**: Architecture spec defines 987-tick and 2000-tick bars. Polygon's futures endpoint provides time-based bars only (not individual trades).
**Decision**: `aggregate_tick_bars()` remains `NotImplementedError`. Only time-based bars (1m through 1D) are implemented. `build_data_bundle()` silently skips tick timeframes.
**Rationale**: User's Polygon plan doesn't include tick-level futures data. Time bars cover all current needs. Tick bars can be added later when a tick data source is available.

---

## D-011: Front-Month Contract Auto-Detection
**Date**: 2026-02-22
**Context**: NQ/ES futures have quarterly expiry (H/M/U/Z). Needed to decide between manual ticker entry and automatic detection.
**Decision**: `PolygonDataProvider.resolve_front_month_ticker()` auto-detects based on CME quarterly cycle. Rolls to next contract on the 15th of the expiry month.
**Rationale**: User chose auto-detect. The 15th rollover is a conservative default. Static method so it's independently testable. 9 test cases cover all edge cases including year boundary.

---

## D-012: 1-Minute Bars as Base Resolution
**Date**: 2026-02-22
**Context**: Could fetch each timeframe separately from Polygon, or fetch 1m and resample locally.
**Decision**: Fetch 1m bars from Polygon, resample to all higher timeframes (3m-1D) using pandas. Daily bars use session-aware grouping (not UTC midnight).
**Rationale**: Fewer API calls. Guaranteed cross-timeframe consistency (all derived from same source). Session-aware daily aggregation respects the 18:00 ET trading day boundary.

---

## D-013: Parquet Caching for Fetched Data
**Date**: 2026-02-22
**Context**: Polygon API calls are rate-limited and slow for large date ranges.
**Decision**: Cache fetched bars as `.parquet` files in `data/cache/`. Cache key: `{ticker}_{timeframe}_{startdate}_{enddate}.parquet`.
**Rationale**: Avoids re-fetching during development iterations. `.gitignore` already excludes `*.parquet`. Parquet is fast and compact for columnar OHLCV data.

---

## D-014: Primary Repo Role Is the Streamlit Extrema Workflow
**Date**: 2026-03-31
**Context**: The repo had accumulated a real extrema training pipeline, a retained 3-class dashboard path, multi-agent infrastructure, and multiple UI surfaces. The old scaffold-first framing no longer matched how the repo is actually used.
**Decision**: Treat `scripts/ml_training_tab.py` and `src/alpha_lab/agents/data_infra/ml/` as the primary application and primary training architecture.
**Rationale**: This is the main workflow the user cares about, and it is the clearest entrypoint for understanding the current codebase.
**Trade-off**: The broader multi-agent architecture remains important, but it is no longer the best first explanation of repo purpose.

---

## D-015: Keep the 3-Class Path as a Secondary Compatibility / Export Surface
**Date**: 2026-03-31
**Context**: The older 3-class dashboard model path is still needed by ML-Trading-Dashboard even though it is no longer the main research direction inside this repo.
**Decision**: Preserve the 3-class path as a supported secondary compatibility/export workflow.
**Rationale**: Downstream consumers still depend on it, so the correct move is clearer separation, not silent removal.
**Trade-off**: The repo carries conceptual overlap between two model worlds, so docs and UI must keep that distinction explicit.

---

## D-016: Canonical Dashboard Compatibility Export Artifact
**Date**: 2026-03-31
**Context**: The older dashboard path had multiple training/result locations, which made it unclear which file downstream consumers should trust.
**Decision**: Treat `scripts/train_dashboard_model.py` -> `data/models/dashboard_3feature_v1.cbm` as the canonical downstream compatibility/export boundary.
**Rationale**: A single export contract reduces confusion without deleting supporting experiment diagnostics.
**Trade-off**: `src/alpha_lab/experiment/training.py` still produces useful diagnostics, but it is no longer the canonical downstream artifact producer.

---

## D-017: ML Quality Gates Must Use True Out-of-Sample Fold Predictions
**Date**: 2026-03-31
**Context**: The Streamlit extrema workflow displayed walk-forward quality gates, but aggregate metrics were being computed by replaying a later refit model across historical test windows.
**Decision**: Compute aggregate ML metrics and quality gates from concatenated out-of-sample fold predictions produced by the per-fold models.
**Rationale**: This matches the semantics implied by walk-forward validation and keeps the primary workflow's quality signal trustworthy.
**Trade-off**: The final saved runtime model is now a separate refit step after evaluation rather than the object used to generate historical fold metrics.

---

## D-018: Final Extrema Runtime Model Is Refit on All Labeled Rows
**Date**: 2026-03-31
**Context**: The previous trainer path concatenated overlapping historical CV windows and held out only the last split, which both duplicated rows and failed to produce a clean final runtime fit.
**Decision**: Use CV splits only for optional RFECV feature selection, then fit the final extrema model once on the full labeled dataset..
**Rationale**: This produces a cleaner runtime model and avoids misleading overlap-driven training behavior.
**Trade-off**: The final runtime fit no longer uses early stopping against a held-out fold by default.

---

## D-019: Generated Models and CatBoost Scratch Logs Are Local Outputs
**Date**: 2026-03-31
**Context**: Model bundles, CatBoost logs, cached feature files, and scratch chart HTML were easy to mistake for tracked source-of-truth artifacts.
**Decision**: Treat model binaries, runtime bundles, CatBoost scratch output, cached parquet/csv files, and scratch chart HTML as generated local outputs.
**Rationale**: This keeps the repo centered on source code and documentation rather than mutable generated state.
**Trade-off**: Reproducing some local artifacts now requires rerunning the corresponding workflow instead of relying on checked-in outputs.

---

## D-020: Config-Keyed Feature Cache
**Date**: 2026-04-03
**Context**: Cached `ml_features.parquet` files were keyed only by symbol/date. Changing pipeline config (extrema thresholds, labeling params, feature settings) silently reused stale cached features.
**Decision**: Cache files are now named `ml_features_{config_hash}.parquet` where hash = SHA256(training_mode + extrema + labeling + features + dashboard_utility + tick_size)[:8].
**Rationale**: Any config change auto-invalidates stale cache. Zero risk of cross-config contamination.

---

## D-021: RFECV Runs Once Before Walk-Forward Loop
**Date**: 2026-04-03
**Context**: Per-fold models used all features while the final saved model used RFECV-selected subset. Quality gates described a different model than what was saved.
**Decision**: RFECV runs once on preliminary CV splits before the walk-forward loop. The selected feature subset is used consistently for all fold models AND the final saved model.
**Rationale**: Evaluation metrics now describe exactly the model that gets deployed.

---

## D-022: CatBoost NaN Handling Preserved
**Date**: 2026-04-03
**Context**: A blanket `fillna(0.0)` before training collapsed "missing data" into "zero signal", destroying CatBoost's native NaN split capability.
**Decision**: Remove `fillna(0.0)` from the primary training flow. CatBoost handles NaN natively.
**Rationale**: The model can now learn "data was missing" as a distinct signal rather than conflating it with zero.

---

## D-023: Cross-Repo Resolution Ordering Aligned (MAE-First)
**Date**: 2026-04-03
**Context**: Quant-Lab experiment labels checked MAE first (conservative); Trading-Dashboard OutcomeTracker checked MFE first (optimistic). This caused training labels and runtime accuracy to disagree on ambiguous events.
**Decision**: Both repos now check MAE first (conservative). Dashboard OutcomeTracker explicitly documents this matches experiment/labeling.py.
**Rationale**: Training labels and runtime correctness measurement must use identical resolution semantics.

---

## D-024: Block Bootstrap for Time-Series CIs
**Date**: 2026-04-03
**Context**: IID bootstrap on time-ordered predictions underestimated CI width by ignoring intraday autocorrelation.
**Decision**: Bootstrap uses moving-block resampling (block_size=10) instead of IID row sampling.
**Rationale**: More realistic confidence intervals for time-series predictions.

---

## D-025: Brier Score Quality Gate
**Date**: 2026-04-03
**Context**: Quality gates were all classification metrics. No check on probability calibration.
**Decision**: Added Brier score to EvaluationResult and "Brier score < 0.25" as a quality gate. Minimum sample gate raised from 50 to 200.
**Rationale**: Probability estimates must be meaningful for downstream decision-making, not just classification thresholds.

---

## D-026: Dashboard-Utility Training Mode
**Date**: 2026-04-04
**Context**: The primary Streamlit pipeline trained binary rebound/crossing at tick extrema — the wrong decision problem for Trading-Dashboard consumption. A mode-aligned training path was needed.
**Decision**: Added `training_mode` field to MLPipelineConfig with "extrema_rebound_crossing" (default) and "dashboard_utility" modes. Dashboard-utility mode uses level-touch events, configurable TP/SL labeling, and the 3 canonical dashboard features.
**Rationale**: The primary pipeline can now produce models strategically aligned to the real execution problem while preserving the research extrema mode unchanged.
**Trade-off**: Utility mode requires `data/experiment/events.parquet` from the experiment Phase 1+2 pipeline.

---

## D-027: Label Purging in Primary Pipeline
**Date**: 2026-04-04
**Context**: The primary extrema pipeline labeled events before walk-forward splitting. Forward labeling windows (5000 ticks) could leak across train/test boundaries.
**Decision**: Training rows whose timestamp + purge buffer extends into the test period are excluded from each fold's training set. Purge count is tracked and surfaced in the UI.
**Rationale**: Prevents subtle label leakage. Follows the retained compatibility path's 2-day purge gap precedent.

---

## D-028: ml_extrema_classifier.py Marked as Experimental
**Date**: 2026-04-04
**Context**: The runtime detector approximates tick-level training features from bar-level OHLCV, filling missing features with 0.0 and using placeholder values. This is a severe domain mismatch.
**Decision**: Marked with `_EXPERIMENTAL = True`, updated docstring, and runtime DeprecationWarning on load.
**Rationale**: Prevents accidental production use of a path with known train/serve mismatch.

---

## D-029: Dashboard-Utility Uses Strategy-Core as the Canonical Decision Layer
**Date**: 2026-06-04
**Context**: Quant-Lab, Strategy-Core, and Trade-Lab had accumulated duplicate definitions for sessions, levels, touches, features, outcomes, and contract fields.
**Decision**: Treat Strategy-Core as the canonical source for dashboard-utility strategy semantics. Quant-Lab's production utility path runs the decision/feature/outcome stage through `engine_decision.py` into Strategy-Core, and `strategy_contract.py` emits structural fields from Strategy-Core constants.
**Rationale**: Research/runtime drift is more dangerous than a normal implementation bug because a model can backtest under one strategy and trade under another.
**Trade-off**: Quant-Lab retains legacy/book-mid comparison paths and experiment tooling, but they are explicitly not the v3 canonical path.

---

## D-030: Engine v3 Session, Level, and Availability Semantics
**Date**: 2026-06-04
**Context**: Earlier dashboard-utility docs and reports referenced `ny_rth` 09:30-16:15, prior-NY-session PDH/PDL, and an `available_from_guard` that was declared but not enforced.
**Decision**: The current dashboard-utility path is `strategy_core_engine_v3`: ET-native sessions are `asia` 19:00-02:45, `london` 03:00-08:00, and `ny` 09:00-17:00 with the 18:00 ET trading-day boundary; PDH/PDL are the full prior trading day's high/low; level `available_from` is enforced before a touch can consume a zone.
**Rationale**: This prevents session-extreme self-touches from leaking future information into labels and aligns levels with full trading-day semantics.
**Trade-off**: v1/v2 bundles and historical reports are no longer semantically compatible with current v3 training.

---

## D-031: Honest Decision-Time Entry Is the Label Anchor
**Date**: 2026-06-04
**Context**: Level-at-touch labels overlapped with the post-touch feature window and did not match the price available when a prediction can actually fire.
**Decision**: Dashboard-utility labels/outcomes use `entry_reference=realistic_at_decision`; decision time is `touch_close + decision_offset_minutes` (default 5 minutes), entry price is the realistic trade price at that decision instant, no new decisions are accepted at/after 16:40 ET, and the forward cutoff is 17:00 ET.
**Rationale**: The feature window and label window must not overlap; labels should model the executable decision point, not the idealized level price from five minutes earlier.
**Trade-off**: Old profitability/edge reports using level-entry or older cutoffs are historical only.

---

## D-032: Saved Dashboard-Utility Bundles Must Carry `strategy.json`
**Date**: 2026-06-04
**Context**: A CatBoost `.cbm` plus metadata does not fully describe strategy semantics to a runtime.
**Decision**: A Streamlit-saved dashboard-utility bundle is `model.cbm`, `metadata.json`, `evaluation.json`, and `strategy.json`. The contract must carry `contract_version`, `engine_version`, session scheme, level scheme, touch rule, feature windows, label policy, inference policy, data requirements, and provenance.
**Rationale**: Runtime activation must fail closed when the bundle's strategy semantics cannot be reproduced.
**Trade-off**: The retained `data/models/dashboard_3feature_v1.cbm` export path is compatibility tooling, not sufficient evidence of v3 compatibility by itself.

---

## D-033: Trade-Lab Is Not v3-Compatible Yet
**Date**: 2026-06-04
**Context**: Inspection of current Trade-Lab code shows local duplicate semantics: no `engine_version` in the local contract schema, no `decision_offset_minutes`, Chicago sessions, exact-tick level touches, quote-mid dwell features, and level-price outcome tracking.
**Decision**: Do not describe a v3 dashboard-utility bundle as Trade-Lab-ready until Trade-Lab is repointed to Strategy-Core's contract loader, sessions, touch/zone rules, feature formulas, and honest-entry outcome orchestration, then verified by end-to-end parity.
**Rationale**: Serving a v3 model through stale runtime semantics would recreate the drift Strategy-Core was built to eliminate.
**Trade-off**: Quant-Lab can still train and emit v3 contracts before Trade-Lab is ready; runtime use remains blocked.

---

## D-034: Current Docs Are Indexed; Stale Historical Reports Are Pruned
**Date**: 2026-06-04
**Context**: Quant-Lab had many Markdown docs from different phases. Some were still useful audit history but contradicted current Strategy-Core v3 semantics.
**Decision**: `docs/README.md` is the docs index; root `ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`, `docs/pipeline_state.yaml`, and Strategy-Core docs are current. Stale historical reports and old scaffold prompt docs were pruned from the working tree.
**Rationale**: Stale reports with old numbers/paths were more likely to confuse current v3 work than to help. Git history remains available for audit recovery.
**Trade-off**: Some historical context now requires Git-history lookup before it can be consolidated into new docs.

---

## D-035: Training Saves Are Gated and Reports Must Preserve OOS Slices
**Date**: 2026-06-05
**Context**: Databento-backed smoke training proved the machinery can train/save/load, but weak smoke artifacts can look deceptively complete if the report only contains aggregate metrics and a model file.
**Decision**: `save_trained_model()` blocks failed quality gates by default. Overrides must be explicit and recorded. Dashboard-utility reports now preserve production-gate OOS diagnostics (`session == ny` and `P(tradeable_reversal) >= 0.70`), session-filtered metrics, three-class OOS balance, exact purge metadata, and optional `oos_predictions.parquet` rows for post-training error analysis.
**Rationale**: Model training must produce decision-useful evidence, not just a CatBoost artifact. Failed-gate bundles are smoke/research artifacts until stronger OOS, gated, and cost-aware evidence supports promotion.
**Trade-off**: Research saves with failed gates require an explicit override, which is intentionally less convenient than silently writing a weak bundle.

---

*Add new decisions below this line.*

## D-036: W3 Training Configuration (Ratified; Execution Stop-Gated in W3a)
**Date**: 2026-06-12
**Context**: W3 PROVE needs one explicit, owner-ratified training configuration — no defaults trusted — for the fresh proof bundle (W3a) and the parity gate (W3b). The W3-CONFIG recon (W3_CONFIG_RECON.md) inventoried the store, current defaults, and the fixture-vs-production label divergence.
**Decision**: The W3 proof-bundle configuration is: `training_mode=dashboard_utility`; session preset `all_to_ny` (train/eval [asia, london, ny], production gate [ny]); `bar_type=147t`; instrument NQ, tick 0.25. Window: 2025-11-21 → 2026-02-13 inclusive, valid store days only (2025-11-20 is an empty day dir and self-skips at date discovery). Folds: purged walk-forward over TRADING days, TRAIN=40 / TEST=5 / STEP=5 / PURGE=2, MIN_TRAIN_EVENTS=30 (the `train_dashboard_model` scheme, now a first-class `--fold-scheme purged-days` path). Label policy (explicit overrides where config defaults differ): `tp_points=15.0`, `sl_points=15.0` (override — config default 30.0), `trap_mfe_min=5.0`, `interaction_window_minutes=5`, `approach_window_minutes=15` (override — default 90), `include_approach_features=True` (override — default False); `entry_reference`/`forward_cutoff`/flatten ride the SC constants (`realistic_at_decision` / `17:00_US/Eastern_ny_close` / 16:40 ET). Features PINNED: `[int_time_within_2pts, int_absorption_ratio, app_avg_trade_size, app_large_trade_vol_pct, app_max_spread]`, `rfecv_enabled=False`. Model: CatBoost 1000 iterations / depth 6 / lr 0.03 / Balanced / seed 42, MultiClass. Model quality is NOT a save gate (explicit `allow_failed_gates` with honest metrics). Bundle name `NQ_W3_<timestamp>` through the W2-hardened save path.
**Rationale**: Every value is explicit so the proof bundle is reproducible and the W3b gate tests exactly what production would serve; sl=15/aw=15 match the bundle convention the audits established rather than the stale config defaults.
**Trade-off / Status**: W3a's compute stop-gate measured the post-vectorization per-day pipeline at 1447.1s for 2026-02-12 (fresh, config hash 7850272e) → ×60 ≈ 24.1 h >> the 90-minute gate, with ~2×513s of that being the SC canonical reader's two full-day event decodes (15.3M events/pass). The TRAIN DID NOT RUN; this configuration stands ratified and waits on an owner ruling on the reader cost (SC-side change — out of W3a's bounds).

---

## D-037: Quote Feature Rides the Gate — Stub-Exclusion Pre-Ruling Superseded
**Date**: 2026-06-12
**Context**: An earlier pre-ruling leaned toward excluding quote-derived features from the W3 train because `quotes_in_window` is stubbed. The W3-CONFIG recon made the stub's blast radius exact: the stub is the SC `PlatformContext.quotes_in_window` accessor only (`strategy_core/runtime/context.py:103-109`, returns `()`, §9.10 retention window open).
**Decision**: `app_max_spread` STAYS in the pinned W3 feature set. The stub affects only a future SC-plugin-path consumer; Quant-Lab training computes quotes from the canonical store (deduped L1 via the SC reader), and Trade-Lab serving computes the feature over its own retained quote window — both real. The feature deliberately rides the W3b parity gate, which is exactly where a serving-side divergence would surface.
**Rationale**: Excluding a live-computable, production-served feature because an unrelated accessor is stubbed would have widened the research/serving gap instead of testing it.
**Trade-off**: If §9.10 lands later and a plugin-path consumer appears, that path needs its own parity evidence before serving.

---

## D-038: OOS Predictions Carry Per-Row Labeler Outcome Columns (PROP-SIM P1)
**Date**: 2026-07-10
**Context**: The prop-firm walker (`alpha_lab.propsim`) needs per-trade excursions (MFE/MAE) and resolution metadata to simulate intraday equity paths from a bundle's OOS rows. The labeler already computes all of it; the OOS writer dropped it at the fold-assembly seam.
**Decision**: `oos_predictions.parquet` gains four per-row columns on FRESH saves: `max_mfe_pts` / `max_mae_pts` (threaded verbatim from the training frame's `max_mfe`/`max_mae` — engine `OutcomeResult` values, never recomputed), `entry_price` (the honest decision-time fill; the stream builder now writes it into the dataset row using the SAME injected trade-price accessor at the SAME decision instant the engine used), and `resolution_type` (the ratified label mapping mirrored from Trade-Lab serving: tradeable_reversal → tp_hit, trap_reversal/aggressive_blowthrough → sl_hit). Existing bundles are NOT retrofitted; consumers requiring these columns must degrade to realized-only behavior with a stated reason when they are absent. Warm dataset caches predating `entry_price` (including the ratified D-036 `ml_utility_7850272e` fleet) yield NaN for that column until a day is rebuilt — the cache tag hashes config, not row schema, so caches are deliberately not invalidated.
**Rationale**: Threading beats recomputation (zero drift risk for MFE/MAE); the entry accessor call is the one expression the engine's honest-entry contract documents, and the label mapping is the exact serving-side convention, so neither introduces a second definition.
**Trade-off**: `entry_price` on warm-cache trains is NaN-degraded until caches roll; a batch `resolution_type` beyond the three-class mapping (e.g. session_end) does not exist by construction — the honest resolver never force-labels.

---

---

## Housekeeping note (2026-07-28, not a ruling)

Root window artifacts were archived to `docs/archive/windows/` (filenames
unchanged) on branch `chore/cleanup-2026-07`. In particular `W3_CONFIG_RECON.md`,
cited above by D-036 as the W3 training-config recon evidence, now lives at
`docs/archive/windows/W3_CONFIG_RECON.md`. The old live-dashboard prototype
(`src/alpha_lab/dashboard/`, `dashboard-ui/`, its three importer scripts) and
`experiment/key_levels.py` were deleted in the same window (dead code, pre-v3
session semantics; recoverable from git history). D-016's
`scripts/train_dashboard_model.py` export boundary is untouched.

---

## D-039: FSM Search Lane — Decomposed Core Replay Identity + Content-Addressed Replay Inputs

**Date**: 2026-08-18
**Context**: `ifvg_prop_robust_config_search_v1` (R1). The former single child-identity concept over-keyed replays (parent study, cost, audit, resource choices all invalidated the scientific replay), and `artifacts_tag()`/date hashes identify a *request*, not bytes — per-day bar/level caches are mutable files on disk. `profile_name` is hashed into `ifvg_profile_hash` and hence into every record ID (SC `section.py:324-331`, `records.py:76-85`), so study-specific child names would fork record identity per study.
**Decision**: New additive lane `src/alpha_lab/agents/data_infra/ifvg/search/`. The reusable scientific identity is `CoreStrategyReplayIdentity` = content-addressed `ReplayInputBundle` (exact physical source partitions with `(source_partition_id, utc_date, logical_key)` keys, day-artifact manifests, schema era, deterministic preflight `ReplayAccessAuthorizationRef`) x scoped QL replay-source identity x SC identity x resolved section hash x canonical profile id x seed identity x resolver/anchor/schema pins. Study linkage lives in `SearchChildMembership`; cost/audit/chart concerns are separate companion/costed identities; the runtime `ReplayExecutionAccessAudit` NEVER enters replay identity. Generated children carry canonical, study-independent names `ifvg_search_profile_<name-free-hash16>`; baselines keep their registered names. Mid-chain start is the QL-only `start_after_artifact` driver fallback on `build_ifvg_v2_capture` (profile/seed mismatch refused before any source read; a snapshot-producing prefix must pass `final_day_exhausts_dataset=False` because dataset exhaustion alters the end seed).
**Rationale**: One replay is reusable across studies exactly when its identity contains only replay-defining facts; content addressing (reusing `hash_allowlisted_source_files` + artifact stamps) makes cache mutation visible instead of silently reusable.
**Trade-off**: Bundle assembly is metadata-only but identity-bearing caches remain mutable files — trust is enforced at load/stamp time, not by the filesystem.

---

## D-040: Two-Path Verification — One Canonical <=5-Day Allowlist + Real VerificationAuthorizationRef

**Date**: 2026-08-18
**Context**: Seeds are profile-bound (SC `replay.py:362`), so the <=5-real-day budget cannot warm-start changed-section children; and a per-release allowlist could accumulate real-data coverage.
**Decision**: Path A = ONE real baseline vertical slice (exact baseline profile + profile-matching seed snapshot + the one canonical <=5-trading-day allowlist, program-wide, marker-enforced via `register_program_allowlist`) run under the new third trusted policy class `VerificationReplayPolicy` (`verification_fixed_allowlist_max5_v1`) and gated by `verification_control_flow_gates_v1` ONLY — research gates never apply to fixtures. Path B = all multi-child behavior is synthetic. The real slice requires an owner-approved immutable `VerificationAuthorizationRef` (allowlist hash + coverage-matrix artifact + seed snapshot + approver) bound through `VerificationRunPayload/Envelope`, validated before any source path is constructed; the synthetic marker is refused for it. Candidate allowlist `2026-06-04..06-10` (5 trading days) remains a PROPOSAL pending the coverage-evidenced owner sign-off (decisions 21/R-5). Reports stamp `verification_only` / `not_for_research_interpretation` / `full_pipeline_not_run`.
**Rationale**: The physics of profile-bound seeds makes real multi-child verification impossible without covert full-history replays; capability-scoped blocking (BLOCKING-VERIFICATION) lets code authoring proceed while R1 acceptance stays blocked.
**Trade-off**: Verification reads of dev-chain day-artifact caches need the `ArtifactProvenanceReadAdapter` (the cache stamp pins the writing policy's allowlist hash); the adapter exposes provenance dates for the stamp check while every authorization delegates to the strict 5-day policy.

---

## D-042: Non-Self-Referential Payload/Envelope Convention + Registry-Key/Resolved-Identity Split + GeneratedProfileCapability

**Date**: 2026-08-18
**Context**: Several earlier contract drafts hashed their own ids/artifact hashes (circular identity) and conflated stable registry names with resolved content identities; the fixed `PROFILE_CAPABILITY_REGISTRY` can never gate generated profile names.
**Decision**: Every ID-producing contract in the lane is a Payload/Envelope pair (`x_id = canonical_contract_sha256(payload)`; payloads carry no self-id, no hash of their own artifact, no display metadata, annotations, or attempt fields), registered in `ID_PRODUCING_CONTRACTS` and enforced by the identity-projection audit test. Registry keys (`feature_block_key`, `decision_policy_key`, `algorithm_key`, ...) are distinct fields from resolved identities (`resolved_feature_block_id`, ...). Deep immutability: `ImmutableMap` (canonically sorted, defensively copied, serialized as sorted key/value records) + `deep_freeze` for `Any` payloads, with mutation-adversarial tests. Baselines are gated by the fixed registry; generated children by `GeneratedProfileCapability` (runnable baseline + registered/authorized values + valid canonical section; never inserted into, never failed by absence from, the fixed registry).
**Rationale**: Identity must be computable before materialization and immune to display/runtime noise.
**Trade-off**: `ImmutableMap` JSON-serializes int keys as strings (deterministic, documented); envelope validation re-hashes payloads on every construction.

---

## D-043: Study-Cell Semantic/Annotation Split, Dimension Registry, Comparison + Delta Taxonomy

**Date**: 2026-08-18
**Context**: Section 7A of the approved plan requires a universal study-cell identity whose engineering noise can never fork scientific identity, and comparisons that fail closed instead of fuzzing.
**Decision**: `StudyCellSemanticPayload` hashes exactly 16 semantic dimensions (engineering protocol is a never-hashed `StudyCellAnnotation`); strategy-only cells carry typed `none_*_v1` identities and never an `ifvg_context_formula_v2` lineage (concrete optional-field `DataLineagePayload`; the pre-lane accepted dataset is referenced via the opaque `legacy_verified_replay_source` provenance literal, unqueryable by the feature layer). The `EXPERIMENT_DIMENSION_REGISTRY` fails closed on unknown/blocked/incompatible dimensions and seeds the plan's 7A.5 computation table, verified row-by-row against `derive_computation_path` (closure: full replay or gated replay implies cost+prop+bootstrap resimulation; the stream hash is reusable only when neither occurs). `build_comparison` refuses declared-class/observed-change mismatches and degrades failed required-equalities to `config_diff_only` with identity+compatibility deltas only; delta families are auto-selected per class (cohort classes never select execution/prop families). Feature blocks partition the frozen tier ladder exactly (module-level assertion); frozen tier bundles reproduce `TIER_FEATURE_REGISTRY` order-exact; planned-block activation (`with_activated_block`) is a versioned registry event minting the first resolved id. MBP-1 stage cutoffs are `StageEvidenceCutoff` objects (no `+inf` representable) with per-feature `WindowTriggerSemantics`; the deep-book identifier guard covers every namespace with the single legacy-provenance exemption.
**Rationale**: Structural impossibility beats review discipline — annotations cannot reach the hash, blocked axes cannot reach enumeration, planned blocks cannot reach materialization.
**Trade-off**: The R1 registry seeds representative-complete dimension entries; later releases extend entries without changing the fail-closed frame.

---

## D-041: Prop Lifecycle — Fidelity-First Contracts, Total-Order Account Events, Complete Simulation Identities

**Date**: 2026-08-19
**Context**: R3 realizes prop-firm accounts on the search lane's replay streams. Three truthfulness hazards govern the design: a 1-minute OHLC bar does not record its intrabar order, so any intrabar chronology is an assumption; prop-firm rules must never come from scraping or fabrication presented as verified; and a policy that changes results but not the simulation identity would silently fork science.
**Decision**: Fidelity first (`alpha_lab.propsim` lifecycle modules, additive beside the untouched evaluation-only walker). Trade-path evidence is typed with `observed_intrabar_order` only ever `"unknown"` for 1m bars; assumed intrabar paths are registered SCENARIO policies (`bar_adverse_extreme_first_v1` / `bar_favorable_extreme_first_v1`) with their own identities and, on order-sensitive trades, provably different results. Rule support is capability-based (`PropRulePathRequirement`: required path capabilities + accepted fidelity classes — never enum ordering) and fails closed with a `PathCapabilityReport`. Calendar semantics are typed (`DayCountBasis`/`DurationRule` under a `SimulatedClockPolicy`; unrepresentable bases fail closed as `unsupported`). Contract evidence compiles field-by-field (source document → per-field evidence → compilation → owner review → supersession) through a one-way status ladder on which synthetic evidence can NEVER reach `first_party_verified`. The account walk extends the proven evaluation engine (one-contract `AccountWalk`≡`EvaluationWalk` parity) to the full lifecycle as ONE strictly ordered `PropAccountEventEnvelope` stream (`prop_account_event_order_v1`, global `event_ordinal`, exact source links). Firm contracts state what is permitted; trader withdrawal behavior is a separate identity; both (plus risk, replacement, clock, trade-path bundle, mode, scenario/bootstrap protocol, seed) ride every account/portfolio simulation identity, with a constructor-surface audit proving no result-changing constructor-only argument. Portfolios replay copied accounts on ONE common correlated path per draw (no per-account resampling path exists). The search frontier consumes prop metrics through `evaluate_prop_gates` (fail-closed rows) + the orchestrator `prop_simulator` seam: conservative ALL-legs feasibility, worst-firm merge per prop objective, and explicit skip notes when no simulator is wired.
**Rationale**: Structural truthfulness — the types make untruthful chronology and fake verification unrepresentable — plus identity completeness before any research use.
**Trade-off**: Chronology-sensitive rules refuse OHLC-only evidence (fewer runnable rule×evidence combinations until ordered evidence exists); prop-owned pareto objectives require a wired simulator or the child is explicitly excluded from the frontier.

---

## D-045: Trader Workspace UI — Guided, Fail-Closed, Truthfully Labeled (R4 half)

**Date**: 2026-08-21
**Context**: R4 puts a trader-usable workspace over the R1–R3 lane without weakening a single fail-closed rule. `FRONTEND_UX_CONTRACT.md` is the normative authority; the hazards are raw-config leaks (a free-form override widget would bypass the typed registries), untruthful status wording, and UI conveniences that silently launch work or fabricate absent artifacts.
**Decision**: The Experiments surface becomes a session-state-backed radio workspace (`New Study | Active Runs | Results | History | Context Research`; the M0–M3 panel delegates verbatim; only the selected route executes). Presentation contracts are pure `src` modules (`study_status` — CS §13 status/scope/empty-state registries with the exact required copy; `study_presentation` — validators, funnel/stage derivations pinned to the orchestrator's exact explanation sentinels, baseline-diff naming, estimates as operational annotations). Drafts are the ONE mutable authoring surface (`study_drafts`, atomic JSON under `data/ifvg_study_drafts/`); freezing routes through `validate_charter` + the immutable charter store and marks the draft frozen forever — research-bearing change is Clone as New Search. UI reads go through `study_providers`: exact-ID manifest-verified loads only, run enumeration from the mutable job root + catalog event log (immutable store roots are never listed; the orchestrator now records the frontier envelope id as a state-file phase note so Results can locate it). The R2→R4 obligations close here: the job shim's `--runner-entry` is REGISTRY-gated (`search/runner_registry.py`; the UI passes keys only; raw strings are refused before import; the only registered entry pre-R5 is the synthetic fixture wiring, so a real charter's launch renders capability-blocked), cross-profile deltas are constructible only through `prepare_cross_profile_deltas` (both lineage-uniqueness reports persist FIRST), and declared two-axis interaction contrasts evaluate on balanced grids (difference-of-differences, lexicographic orientation, seed-7 bootstrap; refusals retained). Every chart has a widget twin, every status renders glyph+word, assumed paths say scenario/approximation, and the development selection is only ever `Development Exploratory Representative`.
**Rationale**: Guided-not-raw (FUX §2) with structural enforcement: the registries, not the widgets, decide what is selectable; absence renders as a typed §31 state instead of an invention.
**Trade-off**: Real launches stay blocked until R5 registers executors (fail-closed by design); population/funnel deltas over live runs await the delta build wired at the seam (fixtures prove the path); the pipeline-runner half of this decision lands with R5.

---

## D-044: Supervised Model Ladder + Decision/Calibration Registries (R5 half)

**Date**: 2026-08-21
**Context**: R5 delivers the bounded §7B supervised lane: model comparisons are meaningless unless every rung scores IDENTICAL out-of-sample rows under identical folds, and any enumeration surface (thresholds, calibrators, feature subsets, cluster counts) is an automated-selection hazard the plan prohibits outside a frozen, owner-ratified charter.
**Decision**: `ifvg/ml/` lands the registry-gated ladder: `MODEL_PROTOCOL_REGISTRY` (prevalence reference + `ifvg_context_logistic_l2_v1` + the existing CatBoost protocol AVAILABLE; GAM registered `planned` with its exact reason) aligned one-to-one with the study-cell vocabulary; `run_supervised_ladder` accepts protocol IDS only, requires the prevalence reference on every ladder, and asserts identical `oos_row_id` sets plus identical `(candidate_id, target, training_prevalence)` tuples across rungs before any delta (`paired_cell_delta_report` generalizes the tier delta onto `oos_row_id` with the same identical-OOS gate and the fixed 10k/seed-7 day-block bootstrap). The logistic rung mirrors the verified CatBoost lane exactly — same fold loop, same prediction-row schema, same model-independent `oos_row_id` — with ALL preprocessing (median imputer + missing indicator, standard scaler, one-hot encoder) fold-fitted inside one sklearn Pipeline, and its fitted artifacts persist portably (manifest-relative refs + checksums; relocation-proof reload). `assert_single_frozen_selection` raises `ProhibitedSelectionError` on any >1-value threshold/calibrator/cluster-count/feature-subset request without owner ratification. Calibration and decision policies are registry pairs with distinct logical keys and resolved 64-hex envelope ids (`DecisionPolicyPayload/Envelope`, `WalkForwardModelSchedulePayload/Envelope`, non-self-referential, projection-audited); only `raw_probability_diagnostics_v1` and `none_diagnostic_only_v1` are executable — every execution-affecting decision policy requires an owner-ratified `RejectedCandidatePolicy`, and S11 carries the exact registered blocked reason on every surface. Core `drift_monitoring` ships report BUILDERS only (feature PSI/KS/Wasserstein, prediction/calibration drift, regime occupancy when a regime exists) — no consumer API can retrain, disable, or promote from a drift alarm. The KMeans regime half of D-044 lands with R6.
**Rationale**: Identical-rows parity and registry-only entry make "the model looked better" claims structurally comparable, and make silent automated selection unrepresentable.
**Trade-off**: The ladder runs on frozen tier feature sets in R5 (a bundle without a tier-exact resolution fails closed rather than improvising a feature list); planned protocols are visible but refuse execution until their releases.

---

## D-045: Pipeline Runner — Semantic Identity vs Execution Attempts (R5 half)

**Date**: 2026-08-21
**Context**: The operator pipeline must make retries and resource changes scientifically inert: stage results may key only on research-bearing content, verification scope can never masquerade as research, and capability gaps (MBP-1 before R5B, post-V1 regime algorithms, unratified S11) must refuse at the plan, not at runtime surprise.
**Decision**: `search/pipeline.py` lands the CS §7 contracts: `PipelineSemanticSpecPayload` (research-bearing fields only; canonical-order stage plans with dependency closure; ≤5 dates under verification) hashes to `pipeline_semantic_id`; `ExecutionAttemptIdentity` (workers/host/timestamps/retry reason) is deliberately NOT an identity envelope — its fields sit in the forbidden-payload vocabulary. The 16-stage runner composes the SAME primitives as the study orchestrator (`enumerate_children`, reuse/neutrality/publication, the costed-evaluation cache, the extracted `merge_prop_vectors` — ONE implementation of ALL-legs feasibility + worst-firm merge — and `build_frontier`); every stage executor is idempotent and store-reusing, a stage whose freshly-minted `PipelineStageResultEnvelope` id equals the prior attempt's is marked REUSED (reuse proven by identity, never assumed), S11 terminal-blocks with the exact reason, and the mutable state file mirrors the search-state conventions (atomic writes, heartbeat lock, stage-boundary cancel sentinel). R5 closes the carried seams: S02 persists per-child lineage evidence (uniqueness reports + serialized key projections) and S14 builds + persists insight panels (`InsightPanelEnvelope`) and baseline↔challenger `ComparisonResultEnvelope`s the UI now consumes (DEV-R4-7/16); S12/S13 run the extended `make_prop_simulator` (additive scenario/bootstrap/stress bridging, DEV-R3-11) and persist real `AccountPolicySetEnvelope`s + `AccountSimulationEnvelope`s with the trader-UI sidecars (DEV-R4-17). Publication is verify-then-activate: S15 persists the immutable result with `prepared_not_published` operational state; activation requires every publication gate AND refuses verification scope with no override. The runner registry gains the REAL executors — the baseline-verification search/pipeline entries fail closed at CONSTRUCTION without the owner's persisted `VerificationRunEnvelope` (registration unblocks the launch surface, never the data), and full-development charters keep NO registered entry (the operator run stays a separate authorized action). The `render_pipeline_run` surface delivers the complete §30 workflow (Configure/Preview/Launch/Monitor/Resume-Retry/Publish) under `ifvg_pipeline_v1_*`, with planned/blocked entries visible-disabled, all 16 stages glyph+word (including `not required`), semantic id + attempt history, the supervised-ladder panel, and launch confined to one scanned `_spawn_pipeline_job` seam.
**Rationale**: P0-3 made operational — identical semantic input re-derives byte-identical stage results across attempts, so a resource change is provably not a new scientific result; capability scoping keeps strategy-only operators independent of unshipped subsystems.
**Trade-off**: In-memory stages (views/labels/folds/ladder) re-execute deterministically on retry rather than deserializing (identity equality proves the reuse); UI-launched prop-bearing plans pin the canonical synthetic firm specs until first-party contracts are owner-verified.

---

## D-046: Offline MBP-1 Feature Activation — Versioned Event, Exact PIT Cutoffs, Research-Only Boundary (R5B)

**Date**: 2026-08-26
**Context**: R5B activates `IFVG_ORDER_FLOW_MBP1_V1` (owner ruling P1-D; promotion boundary R-6). The hazards: a bare status flip would leave the resolved identity unminted; timestamp ties would leak post-stage evidence into point-in-time features (the withdrawn `+inf` cutoff rule); missing order-flow evidence could silently change the candidate cohort; and an activated block could drift toward live/serving use.
**Decision**: The published block registry IS the activation event applied to the exported R5-era planned state (`with_activated_block` at module level): `block_version` 2, status available, the first `resolved_feature_block_id` minted from the REAL Arrow schema hashes (`mbp1_arrow_schemas` — the raw Databento mbp-1 source contract retaining `ts_recv`, the normalized working schema with the deterministic `source_ordinal` tie-break, the 76-metric feature table with per-window validity/missing-reason evidence, and the stage-window evidence table; every hash is the canonical hash of the ordered field/type pairs), the frozen 9-window registry, `ifvg_order_flow_mbp1_formula_v1`, and `mbp1_feature_materializer_v1` — a registry-hash-changing event with the pre-activation state exported so the version bump stays provable. Evidence is an immutable, content-addressed `Mbp1SourceArtifact` (per-partition hashes, first/last order keys, sequence-gap intervals — vendor sequence RESETS are not gaps — and gap-adjusted day coverage; synthetic fixtures store canonical event bytes as manifest-hashed sidecars, real artifacts reference partitions by hash; real reads run authorize-before-path through the verification policy family, and `legacy_verified_replay_source` provenance is refused outright). Stage cutoffs come from the candidate row's own five anchor timestamps (`tap/lock/armed/inversion/entry_ts_utc`) as `COMPLETED_BAR_BOUNDARY` cutoffs under `completed_bar_boundary_exclusive_v1`; admission runs on the complete `(ts_event, ts_recv, sequence, source_ordinal)` key — `PRE_TRIGGER_EXCLUSIVE` `<` / `POST_TRIGGER_INCLUSIVE` `<=` on exact keys, strict `ts_event <` boundary for timestamp-only evidence with EVERY same-timestamp event excluded and the window typed `same_timestamp_order_unavailable` when any tie exists; a missing lower anchor refuses rather than widening. The materializer preserves every candidate row with a deterministic typed-missing precedence (`no_mbp1_partition` → `coverage_below_threshold` → `stage_outside_coverage` → `same_timestamp_order_unavailable` → `sequence_gap` → `instrument_roll_boundary` → `minimum_event_count_not_met`); formula-edge NaNs (zero denominators, zero trades) stay VALID — NaN-by-formula is not missing evidence. Joins are one-to-one on `candidate_id` only (`join_mbp1_features`; duplicate keys refuse; no nearest-time/row-order fallback exists — source-scanned); the bundle view payload pins the exact `mbp1_feature_artifact_id` it joined, so the same view+bundle over different evidence can never share one identity. The controlled Baseline vs Baseline+MBP-1 study runs the challenger bundle against its OWN base bundle on identical rows/labels/folds under the bundle-parametrized ladder (prevalence + logistic; the CatBoost fold runner is tier-locked in the frozen M0–M3 lane and refuses with that exact reason), asserts cross-arm row identity plus a numerically identical prevalence reference, and persists the paired Brier delta (fixed 10k/seed-7 day-block bootstrap) with the permanent `research_only_offline` stamp; the pipeline's S05 materializes+persists the source/feature/coverage artifacts (S00 refuses an MBP-1 plan without the evidence seam — order-flow evidence is never fabricated) and S09 persists the study. The definition carries the boundary structurally (`expected_computation_path="offline_research_feature_materialization_v1"`, `can_affect_execution=False`): no live model feature, execution gate, or Trade-Lab serving use without a later Strategy-Core formula/parity contract and a separately approved sequential model-gated replay. The search job shim now passes the worker's `--store-root` to runner-entry factories (DEV-R5-10 closure).
**Rationale**: Point-in-time truth by construction — unorderable evidence becomes typed nulls, never approximations; activation as a witnessed versioned event keeps "available" from ever being a mere flag; the boundary rides the definition, not the docs.
**Trade-off**: Timestamp-tied stages lose their windows to typed ambiguity (visible in coverage reports) instead of gaining approximate features; the CatBoost rung stays tier-only until a bundle-parametrized protocol is separately registered; the five-day REAL control-flow verification (deliverable 12's real half) remains blocked on the owner's `VerificationAuthorizationRef`, exactly like every real half since R1.

---

## Reservation note (2026-08-18, updated 2026-08-26 — not rulings)

The KMeans regime half of D-044 lands with R6; D-042/D-043 remain as landed — all inside `ifvg_prop_robust_config_search_v1`, per the approved final plan package. (D-041 landed with R3; the trader half of D-045 with R4 and its pipeline half with R5; the MBP-1 activation block with R5B as D-046, above.)
