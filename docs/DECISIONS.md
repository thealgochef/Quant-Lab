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
