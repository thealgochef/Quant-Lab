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

*Add new decisions below this line.*
