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

## D-007: Python 3.13 as Runtime
**Date**: 2026-02-22
**Context**: Machine has both Python 3.11 and 3.13.
**Decision**: All dependencies installed to Python 3.13. Use `"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe"` for running tests.
**Rationale**: pip resolves to 3.13. pyproject.toml requires >=3.11. Both work, but 3.13 has all deps installed.

---

## D-008: Agent System Prompts Stored as Files
**Date**: 2026-02-22
**Context**: Architecture spec defined system prompts inline. They exist in conversation history which compresses.
**Decision**: Store all 6 agent system prompts as `.md` files in `docs/agent_prompts/`.
**Rationale**: Survives context compression. Agents can load their prompts from disk. Prompts are the authoritative behavioral spec for each agent.

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

*Add new decisions below this line.*
