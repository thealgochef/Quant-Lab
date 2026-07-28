# CACHE_SHAPE_RECON — offline research cache layer

Recon date 2026-07-27/28. Parts A/B/D: read-only code recon (three concurrent agents). Part C: single measurement agent running alone (no concurrent load during any timed run). This file is the only CACHE_SHAPE_RECON.md; it is distinct from ERA_GATE_RECON.md, THROUGHPUT_SWEEP.md, W3_CONFIG_RECON.md, PROPSIM_BASELINE.md at this root.

## TIP VERIFICATION

`git rev-parse HEAD` + `git status --porcelain=v2 --branch` per repo, before any work:

| Repo | Path | HEAD | Branch | Tracked modifications |
|---|---|---|---|---|
| SC (strategy-core) | C:\Users\gonza\Documents\Strategy-core | 4740ecdd460caf4ed2c8038fc37c23b2b1d17788 (= expected 4740ecd) | platform-refactor, +0 −0 vs origin | none (untracked-only) |
| TL (Trade-Lab) | C:\Users\gonza\Documents\Trade-Lab | 7a911c0ce0ae5d9e2c97339b7d897377a37a3647 (= expected 7a911c0) | platform-refactor, +0 −0 vs origin | none (untracked-only) |
| QL (Quant-Lab) | C:\Users\gonza\Documents\Claude-Quant-Lab | 27cdfd6630898c7b31559fbe57f009701f8ed931 (= expected 27cdfd6) | platform-refactor, +0 −0 vs origin | none (untracked-only) |

Re-verified after all measurement runs: all three repos still show zero tracked modifications (untracked-only entries).

Runtime-resolution caveat (details in D2 and the Part C preamble): `import strategy_core` resolves to the site-packages pip VCS snapshot @ commit 9d4935346bf42c5d19916e05dbe21dd67c46875c, NOT the SC working tree @ 4740ecd. Parts A/B/D cite the SC working tree (top-level `__init__.py` verified line-identical to the snapshot and every probed signature identical); Part C's measured code is the snapshot.

## PART A — BAR ACCUMULATION SEMANTICS

### A1 — Accumulator location and logic
The streaming tick-bar accumulator is `CandleEngine` + `_MutableCandle` in `SC:src/strategy_core/candles/streaming.py` (`_MutableCandle` at :31-71, `CandleEngine.process_trade` at :115-183). Accumulation (`SC:src/strategy_core/candles/streaming.py:171`):

```python
            candle.close_ts_utc = event_ts_utc
            candle.close_ticks = price_ticks
            if price_ticks > candle.high_ticks:
                candle.high_ticks = price_ticks
            if price_ticks < candle.low_ticks:
                candle.low_ticks = price_ticks
            candle.volume += size
            candle.trade_count += 1
```

Emit-on-fill (`SC:src/strategy_core/candles/streaming.py:180`):

```python
            if candle.trade_count == timeframe:
                completed.append(candle.freeze(complete=True, reason=CloseReason.COMPLETE))
                del current[timeframe]
```

A new bar is seeded from its first trade with `trade_count=1` (`SC:src/strategy_core/candles/streaming.py:149-162`); `timeframe == 1` bars emit immediately (`:164-165`). Timeframes default `(147, 987, 2000)` (`SC:src/strategy_core/candles/streaming.py:102`); production bar size 147 = `DEFAULT_TICK_COUNT` (`SC:src/strategy_core/constants.py:41`).

A vectorized batch twin exists — `build_tick_bars_from_frame` (`SC:src/strategy_core/candles/batch.py:38`, buckets via `bar_index = day_groups.cumcount() // timeframe` at `SC:src/strategy_core/candles/batch.py:124`) — locked to the streaming engine by `SC:tests/test_candle_parity.py:110-121`. It has **no production caller** in TL or QL (grep over `TL:backend` and QL `*.py`: zero hits; only SC's parity test and `SC:validation/phase4b_validate.py`). The canonical QL per-day research path uses the STREAMING engine: `QL:src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py:271-273` delegates to `process_single_date_stream` (`QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:590`), which drives `StrategyRuntime.process_event` per trade (`QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:669`) → `self.candles.process_trade(trade)` (`SC:src/strategy_core/runtime/state.py:343`). TL live/replay drives the same runtime (`TL:backend/src/trade_lab/services/strategy_core_service.py:164`).

### A2 — Reset boundary
**(b) The trading-day boundary only — the 18:00 ET rollover. Not session boundaries, not the flatten, not the forward cutoff.** The in-progress bar resets in exactly two ways: it fills to N trades (COMPLETE, A1 quote above), or the incoming trade's trading day differs from the bar's (`SC:src/strategy_core/candles/streaming.py:143-145`):

```python
            # candles.py:135-137 -- trading-day rollover: freeze the open bar as
            # END_OF_DAY (incomplete) and reopen on the new day.
            if candle is not None and candle.trading_day != trading_day:
                completed.append(candle.freeze(complete=False, reason=CloseReason.END_OF_DAY))
                candle = None
```

The trading day comes from `trading_day_for(trade.event_ts_utc, self._scheme)` (`SC:src/strategy_core/candles/streaming.py:130`), which rolls at the scheme boundary (`SC:src/strategy_core/decisions/sessions.py:98-101`):

```python
    if local_time >= scheme.trading_day_boundary:
        trading_day = local.date() + timedelta(days=1)
    else:
        trading_day = local.date()
```

with `TRADING_DAY_BOUNDARY = time(18, 0)` ET in the canonical scheme (`SC:src/strategy_core/constants.py:145`, `SC:src/strategy_core/constants.py:168-177`). Session windows (asia/london/ny) are never consulted by the candle code — `classify_session`'s window matching (`SC:src/strategy_core/decisions/sessions.py:103-107`) feeds only `SessionInfo.session`, and no session name appears anywhere in `candles/streaming.py` or `candles/batch.py`. `FLATTEN_TIME` (16:40 ET, `SC:src/strategy_core/constants.py:273`) and `RTH_END` (17:00 ET, `SC:src/strategy_core/constants.py:152`) are imported only by decision-layer modules (`SC:src/strategy_core/decisions/streaming.py:193-194`, `SC:src/strategy_core/decisions/honest_entry.py:86-87`, `SC:src/strategy_core/strategies/touch_reversal/plugin.py:340`), never by `candles/`; a bar spanning 16:40 or 17:00 keeps accumulating until 18:00 ET or fill. (`StrategyRuntime.reset` rebuilds the whole engine at `SC:src/strategy_core/runtime/state.py:256`, but that is an administrative symbol-reset, not a data boundary.)

### A3 — Tick increment unit
One trade EVENT (one print), regardless of size. Increment statement (`SC:src/strategy_core/candles/streaming.py:177-178`):

```python
            candle.volume += size
            candle.trade_count += 1
```

Contract size accumulates separately into `volume`; the bar-close test compares `trade_count == timeframe` (`SC:src/strategy_core/candles/streaming.py:180`), and the opening trade seeds `volume=size, trade_count=1` (`SC:src/strategy_core/candles/streaming.py:160-161`). The batch twin counts rows, not size: `trade_count=("price_ticks", "size")` i.e. the group row-count aggregate (`SC:src/strategy_core/candles/batch.py:135`) over `cumcount() // timeframe` buckets (`SC:src/strategy_core/candles/batch.py:124`).

### A4 — Completed-bar record
The emitted record is the frozen `Bar` dataclass (`SC:src/strategy_core/types.py:90-112`), fields in order:

```python
    timeframe_ticks: int
    trading_day: date
    bar_index: int
    bar_id: str
    open_ts_utc: datetime
    close_ts_utc: datetime
    open_ticks: int
    high_ticks: int
    low_ticks: int
    close_ticks: int
    volume: int
    trade_count: int
    is_complete: bool
    is_partial: bool
    close_reason: CloseReason | None = None
```

`CloseReason` is a `StrEnum` with values `"complete"` / `"end_of_day"` (`SC:src/strategy_core/types.py:51-55`); prices are integer ticks on the 0.25 grid (`DEFAULT_TICK_SIZE`, `SC:src/strategy_core/constants.py:28`); timestamps are tz-aware UTC `datetime` (batch truncates pandas ns via `to_pydatetime`, `SC:src/strategy_core/candles/batch.py:155-158`); `bar_id = f"{timeframe_ticks}t:{trading_day.isoformat()}:{bar_index}"` (`SC:src/strategy_core/candles/_ids.py:27`). There is **no field named `bar_ts_utc` on `Bar`** — it carries `open_ts_utc` (the first/seeding trade's `event_ts_utc`, `SC:src/strategy_core/candles/streaming.py:154`) and `close_ts_utc` (the LAST accumulated trade's `event_ts_utc`, `SC:src/strategy_core/candles/streaming.py:171`). The downstream `bar_ts_utc` (on `Touch`) is assigned from the bar's close, i.e. **last event, not emit time** (`SC:src/strategy_core/decisions/touch.py:109`):

```python
                    Touch(
                        bar_ts_utc=bar.close_ts_utc,
```

### A5 — Trailing partial bar
- **Day end, continuous stream (TL live / multi-day):** the first trade of the NEXT trading day freezes the open bar and EMITS it in `completed` as incomplete `END_OF_DAY` (`SC:src/strategy_core/candles/streaming.py:143-145`, quoted in A2).
- **Day end, stream exhaustion (the QL per-day drive):** the trailing partial is neither emitted nor discarded by the engine — it is CARRIED in `self._current` and effectively discarded when the drive ends. `finalize_trading_day()` exists to flush it (`SC:src/strategy_core/candles/streaming.py:194-206`: freezes each open bar `complete=False, reason=CloseReason.END_OF_DAY` and clears the map) but has ZERO callers in TL or QL (greps over `TL:backend` and all of QL: no matches; only SC's parity test calls it, `SC:tests/test_candle_parity.py:102`). The QL drive collects only `update.closed_bars` (`QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:670-672`), so the trailing partial never enters `day_bars` or the forward-scan. Incomplete bars are visible mid-stream only as snapshot `current` bars, frozen `complete=False, reason=None` (`SC:src/strategy_core/candles/streaming.py:191`).
- **Batch builder:** the trailing partial IS emitted as a real row — `is_complete=tc == timeframe`, `close_reason=... CloseReason.END_OF_DAY` otherwise (`SC:src/strategy_core/candles/batch.py:175-177`).
- **At the flatten (16:40 ET) and forward cutoff (17:00 ET): nothing happens to the bar.** No candle module reads `FLATTEN_TIME`/`RTH_END` (grep, A2); those gate executor entries and label windows only. `SC:src/strategy_core/constants.py:164-166` states it directly: unsessioned ET hours (incl. 17:00-18:00) still have bars — "bars still exist there and still belong to the trading day".

### A6 — Session filtering before accumulation
Not session-filtered — every drained trade in the trading-day window folds in regardless of session; only non-trade events are dropped. The QL path from reader to accumulator (`QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:658-669`):

```python
    for event in source.events():
        if isinstance(event, ScDataQualityWarning):
            continue
        ...
        if not isinstance(event, Trade):
            # Quotes do not move bars/levels/touches; the approach quotes are
            # collected in the bounded second pass below.
            continue
        trades.append(event)
        trade_ts.append(event.event_ts_utc)
        update = runtime.process_event(event)
```

The source is bounded by the TRADING-DAY window, not sessions: `for_trading_day` composes "[prev-day 18:00 ET, trading-day 18:00 ET)" (`SC:src/strategy_core/data/databento_parquet.py:257`, computed at `SC:src/strategy_core/data/databento_parquet.py:281-282` from `TRADING_DAY_BOUNDARY`). Inside the runtime the trade goes straight to the engine (`SC:src/strategy_core/runtime/state.py:334`, `SC:src/strategy_core/runtime/state.py:343`):

```python
    def _process_trade(self, trade: Trade) -> RuntimeUpdate:
        ...
        candle_update = self.candles.process_trade(trade)
```

The engine's only internal skip is a trade whose timestamp has no trading day — possible solely inside a scheme `closed_window` (`SC:src/strategy_core/candles/streaming.py:130-132`; `SC:src/strategy_core/decisions/sessions.py:93-96`) — and the canonical scheme has `closed_window=None`, "research drops nothing; bars span the full 18:00->18:00 ET day" (`SC:src/strategy_core/constants.py:176`). Both drivers run that scheme: `StrategyRuntime` defaults `scheme: SessionScheme = RESEARCH_SESSION_SCHEME` (`SC:src/strategy_core/runtime/state.py:189`) and TL constructs it without a scheme override (`TL:backend/src/trade_lab/services/strategy_core_service.py:100-102`). TL's live chain is likewise unfiltered by session: `TL:backend/src/trade_lab/services/live.py:402` → `process_market_event` dispatch (`TL:backend/src/trade_lab/services/runtime.py:784-785`) → `TL:backend/src/trade_lab/services/runtime.py:919` `self.strategy_core_service.process_market_event(trade)` → `self._runtime.process_event(_trade_to_core(event))` (`TL:backend/src/trade_lab/services/strategy_core_service.py:164`). Quotes never reach the accumulator on any path ("only trades advance bars/touches", `TL:backend/src/trade_lab/services/runtime.py:777`; `CandleEngine` accepts only `Trade`, `SC:src/strategy_core/candles/streaming.py:90-92`).

## PART B — WHAT EACH DOWNSTREAM STAGE CONSUMES

### B1 — Session high/low tracking
**Consumer**: `StrategyLevelState` — `SC:src/strategy_core/runtime/levels.py:39`. Input is **RAW EVENTS — trades only** (one print at a time): `def process_trade(self, trade: Trade) -> tuple[Level, ...]` `SC:src/strategy_core/runtime/levels.py:66`, where `Trade` is `event_ts_utc: datetime, price_ticks: int, size: int` (`SC:src/strategy_core/types.py:59-64`). No bars are read anywhere in the fold:

```python
info = classify_session(trade.event_ts_utc, self._scheme)          # levels.py:67
...
self._day_high = trade.price_ticks if self._day_high is None else max(self._day_high, trade.price_ticks)  # levels.py:86
self._day_low = trade.price_ticks if self._day_low is None else min(self._day_low, trade.price_ticks)     # levels.py:87
if info.session in self._ranges:                                    # levels.py:88 (asia/london _Range)
    self._ranges[info.session].update(trade.price_ticks)
```

Quotes never reach the fold: the plugin's `on_event` folds only `Trade` and returns `()` for quotes (`SC:src/strategy_core/strategies/touch_reversal/plugin.py:261-263`); the runtime's `_process_quote` only buffers the last quote (`SC:src/strategy_core/runtime/state.py:313-315`); the QL batch drive explicitly skips non-Trade events before `runtime.process_event` (`QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:663-667`).

**Session boundary declaration**: a **constant**, `RESEARCH_SESSION_SCHEME` in `SC:src/strategy_core/constants.py:168-177`:

```python
RESEARCH_SESSION_SCHEME = SessionScheme(
    timezone=SESSION_TIMEZONE,                       # "US/Eastern", constants.py:144
    trading_day_boundary=TRADING_DAY_BOUNDARY,       # time(18, 0), constants.py:145
    sessions={
        "asia": SessionWindow(time(19, 0), time(2, 45), crosses_midnight=True),
        "london": SessionWindow(time(3, 0), time(8, 0)),
        "ny": SessionWindow(time(9, 0), RTH_END),    # RTH_END = time(17, 0), constants.py:152
    },
    closed_window=None,
)
```

It is *also* carried as a **contract section field** (`section.session_scheme`, string clock times — `SC:src/strategy_core/contract/schema.py:25`, built from the same constant by `default_touch_reversal_section()` `SC:src/strategy_core/strategies/touch_reversal/section.py:179`, converted back to runtime form in `SC:src/strategy_core/strategies/touch_reversal/plugin.py:103-122`), and Trade-Lab activation rejects any bundle whose scheme differs from the constant-derived default (`TL:backend/src/trade_lab/services/model_registry.py:110-114`). Timezone handling: UTC timestamps are converted per event via zoneinfo — `return ts_utc.astimezone(ZoneInfo(scheme.timezone))` with naive input a `ValueError` (`SC:src/strategy_core/decisions/sessions.py:66-68`); comparisons are ET wall-clock, half-open, with Asia's midnight cross handled by `SessionWindow.contains`: `if self.crosses_midnight: return t >= self.start or t < self.end` / else `self.start <= t < self.end` (`SC:src/strategy_core/types.py:187-190`). A non-canonical Chicago scheme is retained for documentation only (`SC:src/strategy_core/constants.py:183-192`).

### B2 — Prior-day extremes
`prior_full_day_extremes(symbol_dir: Path | str, trading_day: date, *, requested_symbol: str | None = None, max_walk_days: int = 10) -> PriorDayExtremes | None` — `SC:src/strategy_core/data/prior_day.py:54-60`. Input is **RAW EVENTS — trades only**, drained from the canonical parquet reader:

```python
source = DatabentoParquetSource.for_trading_day(root, candidate, requested_symbol=requested_symbol)  # prior_day.py:95-97
...
for event in source.events():                        # prior_day.py:103
    if isinstance(event, Trade):
        price = event.price_ticks
        if high is None or price > high: high = price
```

Consumers/seed variants (three inputs feed the same `load_prior_day_summary` slot):
- Replay seeding calls `prior_full_day_extremes` (raw trades): `TL:backend/src/trade_lab/services/replay.py:236-249`.
- The QL training carry `_get_session_hl_for_date` is **BARS**: `bars = _build_bars_for_date(...)` then `return (float(bars_et["high"].max()), float(bars_et["low"].min()))` (`QL:src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py:249-254`), threaded as `prev_day_hl` into `process_single_date_stream` (`QL:src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py:186-197,273-279`) and loaded via `runtime.load_prior_day_summary` (`QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:647-652`). `SC:src/strategy_core/data/prior_day.py:9-14` records the two proven tick-exact on 7/7 probe days.
- Live warm start is **BARS** (vendor ohlcv-1h): `frame = source.ohlcv_frame(...)`, `high_ticks = price_to_ticks(str(float(frame["high"].max())))` (`TL:backend/src/trade_lab/services/live.py:355-374`).

There is also the organic multi-day roll inside `StrategyLevelState` itself, which banks the completed day's raw-trade extremes at the day boundary (`SC:src/strategy_core/runtime/levels.py:71-85`).

### B3 — Zone construction
`build_zones(levels: list[Level], *, zone_proximity_pts: float = ZONE_PROXIMITY_PTS) -> list[Zone]` — `SC:src/strategy_core/decisions/zones.py:23-25`. Input is **neither bars nor raw events**: a list of `Level` objects (name/price/side/available_from, `SC:src/strategy_core/types.py:125-139`) produced by the trade-fed level state.

Merge rule (chained compare against the *last* appended level, ascending price sort):
```python
sorted_levels = sorted(levels, key=lambda level: level.price)       # zones.py:62
...
if lvl.price - groups[-1][-1].price <= zone_proximity_pts:          # zones.py:71
    groups[-1].append(lvl)
```
Representative price rule — arithmetic **mean of constituent level prices**:
```python
prices = [level.price for level in group]
rep_price = sum(prices) / len(prices)                                # zones.py:79-80
```
(descriptor `ZONE_REPRESENTATIVE_PRICE = "mean_of_constituent_levels"`, `SC:src/strategy_core/constants.py:241`). Zone availability = MAX of constituent `available_from` (`SC:src/strategy_core/decisions/zones.py:93-94`).

Proximity value: code default `ZONE_PROXIMITY_PTS = 3.0` (`SC:src/strategy_core/constants.py:63`). It is configurable via the contract section — the production plugin reads `self._section.touch_rule.zone_proximity_pts` falling back to the constant (`SC:src/strategy_core/strategies/touch_reversal/plugin.py:276-281`). Resolved D-036/ratified value: **also 3.0** — D-036 does not override it (`QL:docs/DECISIONS.md:309` lists no zone-proximity key), the QL emitter's `touch_rule.model_copy(update={"bar_type": du.bar_type})` overrides only `bar_type` (`QL:src/alpha_lab/agents/data_infra/ml/strategy_contract.py:92-96`), so the emitted value is the default section's `zone_proximity_pts=ZONE_PROXIMITY_PTS` (`SC:src/strategy_core/strategies/touch_reversal/section.py:188`); the QL batch drives call `build_zones(eng_levels)` with no override (`QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:218`). `DashboardUtilityConfig` has no zone-proximity field (no match in `QL:src/alpha_lab/agents/data_infra/ml/config.py`).

### B4 — Touch detection
`detect_touches(bars: Sequence[Bar], zones: list[Zone], *, tick_size: float, trading_day: date, direction_from_side=...) -> list[Touch]` — `SC:src/strategy_core/decisions/touch.py:55-62`. Input is **BARS** (closed-interval range check on bar high/low ticks; no raw events).

Straddle predicate, verbatim (`SC:src/strategy_core/decisions/touch.py:52`):
```python
return bar_low_points <= zone_rep_points <= bar_high_points
```
First-touch-per-zone-per-day bookkeeping — two layers:
```python
for zone in zones:
    if zone.touched:
        continue                                                     # touch.py:95-96
    ...
    if is_touch(low_points, high_points, zone.representative_price): # touch.py:104
        zone.touched = True                                          # touch.py:105
```
plus the plugin-owned cross-bar dedup (the streaming path rebuilds zones every bar): pre-mark from `self._fired_keys` and record after detection (`SC:src/strategy_core/strategies/touch_reversal/plugin.py:282-289`, set declared at `plugin.py:217`), keyed by `(trading_day, zone.names, zone.representative_price, zone.side.value)` (`SC:src/strategy_core/decisions/dedup.py:30`). Descriptor: `TOUCH_SCOPE = "first_touch_per_zone_per_day"` (`SC:src/strategy_core/constants.py:242`).

`available_from` gate (`SC:src/strategy_core/decisions/touch.py:101-102` — skips without consuming first-touch):
```python
if zone.available_from is not None and bar.close_ts_utc < zone.available_from:
    continue
```
The recorded touch timestamp is the bar's **close** instant (`bar_ts_utc=bar.close_ts_utc`, `SC:src/strategy_core/decisions/touch.py:109`; rationale `touch.py:13-21`).

### B5 — The six feature functions
All six are pure functions over pre-sliced **RAW EVENT** sequences — none reads bars (`SC:src/strategy_core/decisions/features.py:58-65`). Window slicing is the caller's job; both callers anchor at the **touch bar close** (`touch.bar_ts_utc == bar.close_ts_utc`), never the decision ts. Research window lengths come from `DashboardUtilityConfig` (`QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:678-681`; defaults interaction=5 `QL:src/alpha_lab/agents/data_infra/ml/config.py:323-327`, approach=90 `config.py:343-347`; D-036 ratified interaction=5, approach=15-override `QL:docs/DECISIONS.md:309`). Serving window lengths come from the contract section: `windows = section.feature_windows` → `FeatureWindow.from_touch(touch_ts_utc, interaction_window=..., approach_window=...)` (`TL:backend/src/trade_lab/services/inference/features/feature_functions.py:323-328`), with `touch_ts_utc = observation.start_ts_utc` = the SC touch's `bar_ts_utc` (`TL:backend/src/trade_lab/services/inference/inference_engine.py:203`, `TL:backend/src/trade_lab/services/runtime.py:505-507`).

1. **`int_time_beyond_level(trades: Sequence[Trade], level_points, direction, tick_size, *, max_gap_seconds=600)`** — `SC:src/strategy_core/decisions/features.py:68-75`. Reads **trades only**. Anchor: post-touch interaction window `[touch bar close, touch + interaction_window)` (`QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:761-763,769-771`; serving `interaction_start=touch_ts_utc` `TL:backend/src/trade_lab/services/inference/features/feature_functions.py:131-133,181-191`). Length source: `config.interaction_window_minutes` (research) / `section.feature_windows.interaction_window_minutes` (serving); D-036 = 5.
2. **`int_time_within_2pts(trades: Sequence[Trade], level_points, tick_size, *, within_band_pts=2.0, max_gap_seconds=600)`** — `SC:src/strategy_core/decisions/features.py:103-110`; band default `WITHIN_BAND_PTS=2.0` `SC:src/strategy_core/constants.py:73`. Reads **trades only**. Same interaction anchor and length sources as #1 (`QL:...engine_decision.py:772-774`; `TL:...feature_functions.py:194-204`).
3. **`int_absorption_ratio(trades: Sequence[Trade], level_points, direction, tick_size, *, proximity_pts=0.50)`** — `SC:src/strategy_core/decisions/features.py:130-137`; `LEVEL_PROXIMITY_PTS=0.50` `SC:src/strategy_core/constants.py:76`. Reads **trades only** (volume buckets; no inter-event dwell). Same interaction anchor; stream path threads `proximity_pts=config.level_proximity_pts` (`QL:...engine_decision.py:775-778`), serving threads `level_ctx.proximity_points` from `section.feature_windows.level_proximity_pts` (`TL:...feature_functions.py:207-218,103`).
4. **`app_large_trade_vol_pct(trades: Sequence[Trade], *, large_trade_threshold=10)`** — `SC:src/strategy_core/decisions/features.py:168-172`; threshold `SC:src/strategy_core/constants.py:78`. Reads **trades only**. Anchor: pre-touch approach window `[touch − approach_window, touch bar close)` (`QL:...engine_decision.py:811-817`; serving `approach_start=touch_ts_utc - approach_window, approach_end=touch_ts_utc` `TL:...feature_functions.py:134-135,221-229`). Length source: `config.approach_window_minutes` / `section.feature_windows.approach_window_minutes`; D-036 = 15.
5. **`app_avg_trade_size(trades: Sequence[Trade])`** — `SC:src/strategy_core/decisions/features.py:194`. Reads **trades only**. Same approach anchor and length sources as #4 (`QL:...engine_decision.py:818`; `TL:...feature_functions.py:232-237`).
6. **`app_max_spread(quotes: Sequence[Quote], tick_size)`** — `SC:src/strategy_core/decisions/features.py:206`. Reads **quotes only** (L0 bid/ask ticks; `Quote` at `SC:src/strategy_core/types.py:74-79`). Same approach anchor/length sources; research collects only quotes falling inside some touch's `[touch − approach, touch)` window in a bounded second event pass (`QL:...engine_decision.py:707-742,819-822`), serving slices the retained quote buffer (`TL:...feature_functions.py:240-248,166-178`).

### B6 — Honest resolver
The forward scan runs over **BARS**; the only raw-event read is the entry fill (a single trade-print point query at the decision instant).

Batch: `resolve_honest_outcome(touch: Touch, day_bars: Sequence[Bar], trade_price_at: Callable[[datetime], float | None], *, tick_size, tp_points, sl_points, trap_mfe_min, decision_offset_minutes=..., ...)` — `SC:src/strategy_core/decisions/honest_entry.py:76-89`. Entry `entry_price = trade_price_at(decision_ts_utc)` (`honest_entry.py:149`); forward window sliced from bars by close instant (`honest_entry.py:155-159`), then the pure kernel scan (`SC:src/strategy_core/decisions/outcomes.py:180-205`):

```python
for i, bar in enumerate(forward_bars):
    high = bar.high_ticks * tick_size
    low = bar.low_ticks * tick_size
    if is_long:
        bar_mfe = high - entry_points
        bar_mae = entry_points - low
    ...
    max_mfe = max(max_mfe, bar_mfe)
    max_mae = max(max_mae, bar_mae)
    decided = classify_mae_first(max_mfe, max_mae, ..., forced=False)
    if decided is not None:
        label = decided
        bars_to_resolution = i
        break
```

Streaming twin: `StreamingHonestResolver.on_bar(self, bar: Bar)` — `SC:src/strategy_core/decisions/streaming.py:298`, gated to the forward timeframe (`if bar.timeframe_ticks != self._forward_timeframe_ticks: return ()`, `streaming.py:307-308`), identical per-bar excursions + `classify_mae_first` (`streaming.py:320-341`); entry via injected `trade_price_at: Callable[[datetime], float | None]` at `register()` (`streaming.py:191,283`).

Barrier-touch granularity: the **forward-timeframe tick-bar range** — barriers are evaluated against each closed bar's `high_ticks`/`low_ticks` extremes with MAE checked first, so a single bar breaching both barriers resolves to the loss (`SC:src/strategy_core/decisions/outcomes.py:94-96`); no intra-bar event ordering is consulted. The forward timeframe is the contract's `forward_bar_type` in serving (`forward_timeframe_ticks=parse_bar_type(policy.forward_bar_type)`, `TL:backend/src/trade_lab/services/runtime.py:456-457`) and the decision tick-count bars (`bar.timeframe_ticks == tick_count`) in the QL batch drive (`QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:670-673,746-754`); D-036 pins `bar_type=147t` (`QL:docs/DECISIONS.md:309`). Entry-fill accessors: serving = SC trade ring (`TL:backend/src/trade_lab/services/runtime.py:463` → `SC:src/strategy_core/runtime/state.py:319,339`); QL stream = in-memory wire-last trade, 30-min bounded (`QL:...engine_decision.py:683-692`); QL engine-adapter = DuckDB `action='T'` print query (`QL:...engine_decision.py:347-384`).

### B7 — Feature-window configuration entry
**Both** — constants supply code defaults, per-run research values come from QL config, and the contract's section `feature_windows` carries the run's values to serving, which reads only the contract.

- Code defaults (constants): `DEFAULT_INTERACTION_WINDOW_MINUTES = 5` / `DEFAULT_APPROACH_WINDOW_MINUTES = 90` — `SC:src/strategy_core/constants.py:140-141`; band/threshold defaults baked into the formula signatures (`SC:src/strategy_core/decisions/features.py:74,108-109,136,171`).
- Research entry (config): `DashboardUtilityConfig.interaction_window_minutes` (default 5) `QL:src/alpha_lab/agents/data_infra/ml/config.py:323-327`, `approach_window_minutes` (default 90) `config.py:343-347`, `level_proximity_pts` (0.50) `config.py:328-332`. Readers: `QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:678-681` (stream drive), `:419` and `:485-486` (engine-adapter windows). D-036 resolved values: interaction 5, approach 15 (`QL:docs/DECISIONS.md:309`).
- Contract writer: the QL emitter stamps config into the section — `feature_windows=base.feature_windows.model_copy(update={"interaction_window_minutes": du.interaction_window_minutes, "approach_window_minutes": du.approach_window_minutes, "level_proximity_pts": du.level_proximity_pts})` (`QL:src/alpha_lab/agents/data_infra/ml/strategy_contract.py:97-103`), over the constant-sourced default section (`SC:src/strategy_core/strategies/touch_reversal/section.py:195-202`); typed model `FeatureWindows` (`SC:src/strategy_core/contract/schema.py:151-163`); section field `SC:src/strategy_core/strategies/touch_reversal/section.py:91`.
- Contract readers (every reader found): `TL:backend/src/trade_lab/services/inference/features/feature_functions.py:98-106` (`LevelContext.from_contract` — proximity/threshold/band), `:320-328` (`build_feature_vector` — the two window minutes); `TL:backend/src/trade_lab/services/runtime.py:424-429` (buffer retention = approach + interaction + slack); `TL:backend/src/trade_lab/services/model_registry.py:85-92` (activation gate: approach + interaction vs retention ceiling). No QL production code reads `feature_windows` back off a contract (only the emitter writes it — sole `QL:src` match at `strategy_contract.py:97`), and no `TL:backend/scripts` file references it.

## PART C — VOLUME AND TIMING

**Code-under-measurement:** `import strategy_core` resolves to the site-packages pip VCS snapshot `C:\Users\gonza\AppData\Local\Programs\Python\Python313\Lib\site-packages\strategy_core\__init__.py` @ commit `9d4935346bf42c5d19916e05dbe21dd67c46875c` (NOT the SC working tree @ 4740ecd) — confirmed at runtime in the `strategy_core_file` field of all four result JSONs. QL `alpha_lab` resolves editable to `C:\Users\gonza\Documents\Claude-Quant-Lab\src` @ 27cdfd6. Python 3.13.1. All SC `file:line` citations below are against the site-packages snapshot.

**D-036 config (cache tag 7850272e):** symbol NQ, bar_type 147t, tp 15 / sl 15 / trap_mfe_min 5, interaction window 5m, approach window 15m, level_proximity 0.5, include_approach_features true; data_dir `C:\Users\gonza\Documents\Claude-Quant-Lab\data\databento`. Cache BYPASSED throughout: cache read/write lives only in `build_utility_dataset` (`QL:src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py:155-200`); every run called `process_single_date_stream` (or its staged replica) directly, with the production serializer `_write_day_cache` (:65-78) redirected to `_scratch_timing`.

**Day selection** (computed by the predecessor run — `scratch_cache_shape_select.py` → `_scratch_timing/handoff.json` — and spot-checked this session against store directory listings): window 2025-11-21..2026-02-13 has 73 mbp10 days, 59 carrying the `ml_utility_7850272e` cache; median `mbp10.parquet` = 767,492,251 bytes; **DAY_MBP10 = 2025-12-08** is exactly the median (delta 0 bytes, cache present; runner-up 2025-12-05, +9,063,549). **DAY_MBP1 = 2026-03-02** (`mbp1.parquet` 359,650,790 bytes, cache present, 13,559 bytes, listed not opened). Interpretation: 2025-12-08 is a byte-exact median-volume mbp10 day (representative, not cherry-picked); 2026-03-02 samples the mbp1 era (D-P-17 trades-from-TOB). Spot-check: both file sizes and cache presence match the live directory listings exactly.

**Cross-check headline:** for both days the canonical `process_single_date_stream` end-to-end run produced a labeled frame **row-identical** to the staged drive (`frame_equals: true`, `columns_equal: true`, 5 rows vs 5 rows; serialized parquets byte-size-equal at 13,553 B / 13,559 B).

Source shorthand used below — every number's producing invocation:
- **S12** = `result_staged_2025-12-08.json` ← `python C:\Users\gonza\Documents\Claude-Quant-Lab\scratch_cache_shape_timing.py --day 2025-12-08 --mode staged`
- **C12** = `result_canonical_2025-12-08.json` ← same script `--day 2025-12-08 --mode canonical`
- **S03** / **C03** = same for `--day 2026-03-02`
- **C4C5** = `_scratch_timing/c4c5.json` ← `python C:\Users\gonza\Documents\Claude-Quant-Lab\scratch_cache_shape_c4c5.py`

### C1 — Volumes

| Metric | 2025-12-08 (mbp-10) | 2026-03-02 (mbp-1) | Source |
|---|---|---|---|
| Raw day parquet (os.path.getsize) | `NQ\2025-12-08\mbp10.parquet` = 767,492,251 B | `NQ\2026-03-02\mbp1.parquet` = 359,650,790 B | S12/S03 "source construction".files |
| Prior-day stub also opened by the trading-day composition | `NQ\2025-12-07\mbp10.parquet` = 8,229,412 B | `NQ\2026-03-01\mbp1.parquet` = 10,529,388 B | S12/S03 (two-file [prev 18:00 ET, day 18:00 ET) window, `for_trading_day`, databento_parquet.py:248-344) |
| Drained items total | 8,646,931 | 13,067,323 | S12/S03 stage (a) |
| — trades | 302,299 | 441,575 | S12/S03 stage (a) |
| — quotes after TOB dedup | 8,344,632 | 12,625,748 | S12/S03 stage (a) |
| — DataQualityWarnings | 0 | 0 | S12/S03 stage (a) |
| Completed 147t bars | 2,056 | 3,003 | S12/S03 stage (b) |
| Zone count (end of day) | 6 | 6 | S12/S03 zone-snapshot stage |
| Touch count | 5 | 5 | S12/S03 stage (b) |
| Labeled-row count (what the D-036 build would cache) | 5 (0 drops of any kind) | 5 (0 drops) | S12/S03 volumes + C12/C03 cross_check |

- TOB dedup is explicit: `emit_deduped`, `SC:src/strategy_core/data/databento_parquet.py:131-192` — a quote is constructed only when its L0 tuple (bid_ticks, ask_ticks, bid_size, ask_size) differs from the previous quote's in stream order, state carried across batches and files; trades always emit. Quote counts above are therefore post-dedup; trade counts are as-emitted. On the mbp-1 day, trades come from `action='T'` rows of the TOB schema (D-P-17, databento_parquet.py:862-872).
- Bar count is completed bars only; the trailing partial is not emitted on this path (1 in-progress bar sat in `runtime.snapshot().current_bars` at end of stream, both days; S12/S03).
- Zone definition used: end-of-day `build_zones` over the final level state — `runtime.snapshot()` → `plugin.snapshot_zones(trading_day)` (plugin.py:302) → `build_zones(list(levels()))` at default `ZONE_PROXIMITY_PTS` 3.0 (levels.py:110-111, zones.py:23-106, constants.py:63), fired zones pre-marked. Both days: 6 single-level zones (pdh, pdl, asia_high/low, london_high/low; 6 levels, no merges).
- Store-cache size observation (directory listing only; caches never opened): the existing `ml_utility_7850272e.parquet` for 2025-12-08 is 12,933 B (mtime Jun 17) while both fresh rebuilds this session serialize to 13,553 B; the 2026-03-02 store cache (13,559 B, mtime Jul 11) is byte-size-identical to its rebuild. Size-only observation; cause not determined here.

### C2 — Per-stage wall-clock

`time.perf_counter`, cache bypassed, one fresh process per run, everything sequential (nothing else ran). Lumping is stated per line; stages that re-pay decode carry a subtraction estimate labeled as such.

| Stage | 2025-12-08 (S12) | 2026-03-02 (S03) |
|---|---|---|
| seed (prior-day extremes) — own labeled extra line | 0.864 s (walk: 2025-12-07 empty → 2025-12-05; PDH 25868.0 / PDL 25579.75) | 0.729 s (walk: 2026-03-01 empty → 2026-02-27; PDH 25098.75 / PDL 24782.0) |
| source construction (file resolution only) | 0.0007 s | 0.0008 s |
| (a) parquet decode + event drain/normalize — LUMPED (generator-fused per batch in `_decode_batches`/`emit_deduped`, databento_parquet.py:493-574/:131-192; incl. front-month prescan :1045-1102; no separable decode-only step exists); store-nothing drain | **24.91 s** | **29.49 s** |
| (b) bar fold + level/session tracking + per-bar zone rebuild + touch detect — LUMPED inside `runtime.process_event` (state.py:334-371 → plugin.on_event/on_bar_closed); fed off a second fresh drain, so it RE-PAYS decode (incl. compact-array capture overhead) | **37.59 s** lumped; fold-only subtraction estimate (b−a) = **12.68 s** | **48.81 s** lumped; (b−a) = **19.31 s** |
| zone build (end-of-day snapshot; per-bar detection rebuild remains lumped in (b)) | 0.00015 s | 0.00010 s |
| (c) approach-quote second pass from DISK (canonical replica of engine_decision.py:697-742; includes a full decode) | 29.53 s; assignment-only estimate (c−a) = 4.62 s | 37.07 s; (c−a) = 7.58 s |
| (d) forward resolution (`resolve_honest_outcome` per touch) | 0.0127 s | 0.0171 s |
| (e) feature computation + row build | 0.171 s | 0.133 s |
| (f) serialize/write (redirected `_write_day_cache` → `_scratch_timing`) | 0.0050 s (13,553 B) | 0.0031 s (13,559 B) |
| staged process wall total | 93.49 s | 116.69 s |

**Canonical cross-check** (fresh process each): `process_single_date_stream` end-to-end total = **72.20 s** (C12) and **100.69 s** (C03). Counts match the staged drive: labeled rows 5 = 5 both days, and the full frames are byte-value identical (`frame_equals: true`); the canonical API returns only the labeled frame, so zone/touch counts are validated via that row identity (rows are per-touch) rather than exposed directly — no divergence anywhere. Canonical < staged-sum because staged pays three decodes vs canonical's two; canonical additionally materializes the full Trade object list internally (engine_decision.py:654-668, its normal behavior), which the staged drive replaced with compact arrays.

**OS-cache state (honest):** 2025-12-08 stage (a) was the first *full* read of that parquet this session, but the predecessor's dead run had partially decoded the same file ~35 min earlier, so it was partially warm; stages (b)/(c) and C12 are repeated warm reads. 2026-03-02 stage (a) was the first touch of that file this session (untouched since Jul 11 — treated as cold-first); (b)/(c) and C03 warm. Cold-first inflates (a) slightly and therefore slightly deflates the (b−a)/(c−a) subtraction estimates on 03-02.

### C3 — Peak RSS

ctypes `GetProcessMemoryInfo` → `PeakWorkingSetSize`, argtypes/restype set explicitly; sampled at every stage boundary (per-process monotone peak ⇒ the stage where it first rises set it).

| Process | Peak working set | Where the peak was set |
|---|---|---|
| staged 2025-12-08 (S12) | 2,715,271,168 B (2.53 GiB) | during stage (a) decode (was 2,426,880,000 pre-drain) |
| canonical 2025-12-08 (C12) | 2,721,329,152 B (2.53 GiB) | during the canonical stream drive |
| staged 2026-03-02 (S03) | 2,124,120,064 B (1.98 GiB) | during stage (a) decode |
| canonical 2026-03-02 (C03) | 2,244,419,584 B (2.09 GiB) | during the canonical stream drive |

Caveat: the seed stage's bars-based helper leaves ~1.7–2.2 GB of DuckDB/TickStore working-set residue before the drive starts (S12 ws 2.34 GB post-seed), so peak WS includes that baseline, not just the reader.

RSS attributable to holding the drained stream (compact form, per the memory discipline):
- Compact numpy trade arrays (ts int64 ns, price_ticks int64, size int32, side uint8): **6,348,279 B** (12-08) and **9,273,075 B** (03-02) — deterministic `nbytes` (S12/S03 stage (b)).
- Bounded approach-window quote collection (canonical two-pass mirror): WS delta across stage (c) = **+248,078,336 B** for 948,951 retained Quote refs (12-08); **+252,661,760 B** for 1,010,950 refs (03-02). WS deltas are environment-noisy (stage (b) on 03-02 showed a −142 MB delta from OS working-set trimming); nbytes figures are the deterministic ones.
- A full-object-list figure for the entire drained stream is NOT OBTAINABLE READ-ONLY at this memory budget — materializing all 8.6M/13.1M boxed events is exactly what killed the predecessor process on this box.

### C4 — Drained-stream parquet size estimate

Assumption, verbatim: **raw fixed-width columns, no parquet encoding/dictionary/compression.** Dtypes are the reader's vectorized decode outputs (event fields per `SC:src/strategy_core/types.py:58-70` Trade, :73-87 Quote):

- Trade: `event_ts_utc` int64 epoch-ns 8 B (ns-unit ts_sort, `databento_parquet.py:618`, ns branch :620-629) + `price_ticks` int64 8 B (`_grid_ticks` `.astype(np.int64)` :684; prealloc :900) + `size` int64 8 B (`_strict_size` :695-707 via `_int64_values` cast pa.int64 :655-661; prealloc :901) + `side` 1 B assumed (reader emits Python `str|None` 'A'/'B'/'N' :848-858, :985-994; a fixed-width 1-byte code is ASSUMED for the estimate) = **25 B/trade**.
- Quote: `event_ts_utc` 8 B + `bid_price_ticks` 8 B (:934, :684) + `ask_price_ticks` 8 B (:935) + `bid_size` 8 B (`_optional_size` `.astype(np.int64)` :727) + `ask_size` 8 B (:939-941) = **40 B/quote**.

Σ(rows × per-column width), from C4C5 (counts from S12/S03):

| Day | Trades | Quotes | Total | vs raw day file |
|---|---|---|---|---|
| 2025-12-08 | 302,299 × 25 = 7,557,475 B | 8,344,632 × 40 = 333,785,280 B | **341,342,755 B (325.5 MiB)** | 44% of the 767,492,251 B mbp10 (raw carries 10 depth levels; drained stream is deduped L0 only) |
| 2026-03-02 | 441,575 × 25 = 11,039,375 B | 12,625,748 × 40 = 505,029,920 B | **516,069,295 B (492.2 MiB)** | 143% of the 359,650,790 B mbp1 (the raw file is compressed/encoded parquet; this estimate is deliberately uncompressed) |

Not written to disk — arithmetic only.

### C5 — Bars-only artifact size estimate

Bar fields `SC:src/strategy_core/types.py:90-112`, per-bar widths (same raw fixed-width, no-encoding assumption): `timeframe_ticks` int64 8 + `trading_day` date32 4 (days-since-epoch ASSUMED) + `bar_index` int64 8 + `bar_id` raw string bytes at the day's observed total length (1 B/char ASSUMED; observed mean 19.46 chars on 12-08, 19.63 on 03-02 — S12/S03 stage (b)) + `open_ts_utc` 8 + `close_ts_utc` 8 + `open/high/low/close_ticks` 4×8=32 + `volume` 8 + `trade_count` 8 + `is_complete` 1 + `is_partial` 1 + `close_reason` raw string bytes (observed uniformly `"complete"` = 8 B/bar on both days). Fixed portion = 86 B/bar. From C4C5:

| Day | Bars | bar_id chars | close_reason bytes | Total | Mean B/bar |
|---|---|---|---|---|---|
| 2025-12-08 | 2,056 × 86 + 40,010 + 16,448 | 40,010 | 16,448 | **233,274 B (227.8 KiB)** | 113.46 |
| 2026-03-02 | 3,003 × 86 + 58,950 + 24,024 | 58,950 | 24,024 | **341,232 B (333.2 KiB)** | 113.63 |

A bars-only artifact is ~3 orders of magnitude smaller than the drained-stream serialization (≈0.07% of C4).

## PART D — RECOMPUTATION ENTRY POINTS

### D1 — In-memory entry points per layer

Every Part B layer has a public in-memory entry point in SC; only the prior-day-extremes helper is path-coupled. No layer forces a parquet path except where noted.

**Bar fold (tick-bar accumulator)** — two entry points, both in-memory:

Streaming, one `Trade` at a time — `SC:src/strategy_core/candles/streaming.py:115` (class `CandleEngine` at `:86`, constructor `:100-105`, day-close at `:194`):
```python
def __init__(
    self,
    timeframes: tuple[int, ...] = (147, 987, 2000),
    *,
    scheme: SessionScheme = RESEARCH_SESSION_SCHEME,
) -> None:
...
def process_trade(self, trade: Trade) -> CandleUpdate:
...
def finalize_trading_day(self) -> tuple[Bar, ...]:
```

Batch, over an in-memory pandas DataFrame (columns `ts_event`/`price`/`size`, `SC:src/strategy_core/candles/batch.py:47-49`; pandas imported lazily inside the body `:68-71`) — `SC:src/strategy_core/candles/batch.py:38-44`:
```python
def build_tick_bars_from_frame(
    frame: "pd.DataFrame",
    timeframes: tuple[int, ...],
    *,
    scheme: SessionScheme = RESEARCH_SESSION_SCHEME,
    tick_size: float = DEFAULT_TICK_SIZE,
) -> list[Bar]:
```
Parity between the two paths is asserted by `tests/test_candle_parity.py` per the module docstring (`SC:src/strategy_core/candles/batch.py:8-9`).

**Level tracking (session high/low)** — `StrategyLevelState`, fully in-memory — `SC:src/strategy_core/runtime/levels.py:39` (constructor `:42`, seed `:61`, fold `:66`, read-back `:92`, zones wrapper `:110`):
```python
def __init__(self, *, scheme: SessionScheme = RESEARCH_SESSION_SCHEME, tick_size: float = DEFAULT_TICK_SIZE) -> None:
...
def load_prior_day_summary(self, trading_day, *, high_ticks: int, low_ticks: int) -> None:
...
def process_trade(self, trade: Trade) -> tuple[Level, ...]:
...
def levels(self) -> tuple[Level, ...]:
...
def zones(self) -> list[Zone]:
```
Multi-day in-memory streams bank the completed day's extremes organically (PDH/PDL emerge without any store read) at `SC:src/strategy_core/runtime/levels.py:70-81`; a one-day drive needs the explicit `load_prior_day_summary` seed.

Prior-day extremes helper: **no in-memory entry exists**. `prior_full_day_extremes` takes a store directory and drains the canonical parquet reader — `SC:src/strategy_core/data/prior_day.py:54-60`:
```python
def prior_full_day_extremes(
    symbol_dir: Path | str,
    trading_day: date,
    *,
    requested_symbol: str | None = None,
    max_walk_days: int = 10,
) -> PriorDayExtremes | None:
```
It forces you through `DatabentoParquetSource.for_trading_day(root, candidate, requested_symbol=...)` (`SC:src/strategy_core/data/prior_day.py:95-97`; import at `:39`). The in-memory substitutes are the two `StrategyLevelState` mechanisms above (`levels.py:61`, `levels.py:70-81`).

**Zone build** — pure in-memory — `SC:src/strategy_core/decisions/zones.py:23-25`:
```python
def build_zones(
    levels: list[Level], *, zone_proximity_pts: float = ZONE_PROXIMITY_PTS
) -> list[Zone]:
```

**Touch detect** — pure in-memory over a `Sequence[Bar]` — `SC:src/strategy_core/decisions/touch.py:55-62` (predicate `is_touch` at `:42`):
```python
def detect_touches(
    bars: Sequence[Bar],
    zones: list[Zone],
    *,
    tick_size: float,
    trading_day: date,
    direction_from_side: Mapping[Side, Direction] = DEFAULT_DIRECTION_FROM_SIDE,
) -> list[Touch]:
```

**Six feature functions** — pure over in-memory `Sequence[Trade]`/`Sequence[Quote]` — `SC:src/strategy_core/decisions/features.py:68-75, 103-110, 130-137, 168-172, 194, 206`:
```python
def int_time_beyond_level(trades: Sequence[Trade], level_points: float, direction: Direction, tick_size: float, *, max_gap_seconds: float = MAX_DWELL_GAP_SECONDS) -> float:
def int_time_within_2pts(trades: Sequence[Trade], level_points: float, tick_size: float, *, within_band_pts: float = WITHIN_BAND_PTS, max_gap_seconds: float = MAX_DWELL_GAP_SECONDS) -> float:
def int_absorption_ratio(trades: Sequence[Trade], level_points: float, direction: Direction, tick_size: float, *, proximity_pts: float = LEVEL_PROXIMITY_PTS) -> float:
def app_large_trade_vol_pct(trades: Sequence[Trade], *, large_trade_threshold: int = LARGE_TRADE_THRESHOLD) -> float:
def app_avg_trade_size(trades: Sequence[Trade]) -> float:
def app_max_spread(quotes: Sequence[Quote], tick_size: float) -> float:
```

**Outcome resolution** — batch: `SC:src/strategy_core/decisions/honest_entry.py:76-89`; I/O only via the injected callable:
```python
def resolve_honest_outcome(
    touch: Touch,
    day_bars: Sequence[Bar],
    trade_price_at: Callable[[datetime], float | None],
    *,
    tick_size: float,
    tp_points: float,
    sl_points: float,
    trap_mfe_min: float,
    decision_offset_minutes: int = DECISION_OFFSET_MINUTES,
    flatten_time: time = FLATTEN_TIME,
    rth_end: time = RTH_END,
    timezone: str = SESSION_TIMEZONE,
) -> OutcomeResult | HonestEntryDrop:
```
Streaming: `StreamingHonestResolver` (`SC:src/strategy_core/decisions/streaming.py:173`; constructor `:183-197`, `register` `:255-262`, `on_bar(bar: Bar)` `:298`, `flush(now_ts_utc)` `:365`) — plain scalars + the same injected `trade_price_at` callable; all bars fed in-memory.

**Whole-chain single entry** — `StrategyRuntime.process_event` drives bar fold + level fold + zone build + touch detect from one in-memory event stream — `SC:src/strategy_core/runtime/state.py:285-292`:
```python
def process_event(self, event: Trade | Quote | DataQualityWarning) -> RuntimeUpdate:
```
(constructor `:183-195`, quoted in D3; seed pass-through `load_prior_day_summary` `:274-276`). QL's canonical labeling already batch-drives exactly this from in-memory events at `QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:646,669` — the parquet coupling there is the event *producer* (`DatabentoParquetSource.for_trading_day`, `:638`), not the engine.

### D2 — QL→SC import graph and reachability

**Install-resolution fact (the stated environment premise is stale).** The brief's "strategy-core is EDITABLE-installed" failed re-verification. Probe:
```
python -c "import strategy_core; print(strategy_core.__file__)"
C:\Users\gonza\AppData\Local\Programs\Python\Python313\Lib\site-packages\strategy_core\__init__.py
```
`os.path.realpath` returns the same site-packages path (no junction/symlink). The dist-info proves a regular pip VCS install pinned at the INGEST-window commit: `site-packages/strategy_core-0.1.0.dist-info/direct_url.json` = `{"url": "https://github.com/thealgochef/Strategy-Core.git", "vcs_info": {"commit_id": "9d4935346bf42c5d19916e05dbe21dd67c46875c", ...}}`, `INSTALLER` = `pip`, `RECORD` lists real installed files. Globbing site-packages finds **no** `__editable__*strategy*` pth (editable pths exist only for `alpha_signal_lab` → `C:\Users\gonza\Documents\Claude-Quant-Lab\src`, `trade_lab` → `C:\Users\gonza\Documents\Trade-Lab\backend\src`, and `trade_dashboard`). So from QL, `import strategy_core` resolves to the **9d49353 snapshot in site-packages, NOT the SC working tree (HEAD 4740ecd)**. Consistency evidence: site-packages `strategy_core/__init__.py` is line-identical to `SC:src/strategy_core/__init__.py` (both read in full), and every probed signature below matches the SC working-tree source verbatim. Byte-level equality of all module bodies between the snapshot and the working tree: NOT OBTAINABLE READ-ONLY — settling it needs a hash/diff of working-tree files against the RECORD sha256s (or `git diff 9d49353..4740ecd`), which is execution outside the allowed probe set.

**Proved imports (all run from the QL root; all resolve to site-packages `__file__`s):**
```
python -c "from strategy_core import (build_zones, detect_touches, resolve_honest_outcome,
    StreamingHonestResolver, CandleEngine, build_tick_bars_from_frame,
    int_time_beyond_level, int_time_within_2pts, int_absorption_ratio,
    app_large_trade_vol_pct, app_avg_trade_size, app_max_spread)"   → TOP-LEVEL OK
python -c "from strategy_core.runtime.levels import StrategyLevelState"                          → OK
python -c "from strategy_core.data.prior_day import prior_full_day_extremes"                     → OK
python -c "from strategy_core.contract.schema import FeatureWindows"                             → OK
python -c "from strategy_core.strategies.touch_reversal.section import TouchReversalSection, default_touch_reversal_section" → OK
python -c "from strategy_core.runtime.state import StrategyRuntime"                              → OK
python -c "from strategy_core.strategies.registry import get_strategy"                           → OK
```
`inspect.signature` printed for every symbol above matched the working-tree signatures quoted in D1 exactly (including defaults `zone_proximity_pts=3.0`, `flatten_time=16:40`, `rth_end=17:00`, `max_gap_seconds=600.0`, `within_band_pts=2.0`, `proximity_pts=0.5`, `large_trade_threshold=10`).

**Negative probe — top-level re-export gaps.** `build_zones`/`detect_touches`/the six features/`resolve_honest_outcome`/`StreamingHonestResolver`/`CandleEngine`/`build_tick_bars_from_frame` are all re-exported at package top level (`SC:src/strategy_core/__init__.py:78-116`, `__all__` `:132-186`). `StrategyLevelState`, `prior_full_day_extremes`, and `FeatureWindows` are NOT (absent from `__init__.py:78-186`) and need their submodule paths:
```
python -c "from strategy_core import StrategyLevelState"
ImportError: cannot import name 'StrategyLevelState' from 'strategy_core' (...site-packages\strategy_core\__init__.py)   exit=1
```
None of the Part B set is private or function-nested — every symbol is a module-level public name (`__all__` at `SC:src/strategy_core/runtime/levels.py:14`, `SC:src/strategy_core/data/prior_day.py:42`, `SC:src/strategy_core/decisions/streaming.py:66`, `FeatureWindows` in `SC:src/strategy_core/contract/schema.py:54`). Private internals adjacent to the set (unreachable by convention, not needed to drive it): `_MutableCandle` `SC:src/strategy_core/candles/streaming.py:32`, `_Range`/`_DaySummary` `SC:src/strategy_core/runtime/levels.py:18,34`, `_OpenSetup`/`_as_direction` `SC:src/strategy_core/decisions/streaming.py:154,165`, `_ContractModel` `SC:src/strategy_core/contract/schema.py` (imported as private at `SC:src/strategy_core/strategies/touch_reversal/section.py:62`).

**QL's own modules import cleanly** (probe run with `PYTHONPATH=src` per the brief; the `__editable__.alpha_signal_lab-0.1.0.pth` already puts `Claude-Quant-Lab\src` on sys.path globally, making it redundant):
```
PYTHONPATH=src python -c "import alpha_lab.agents.data_infra.ml.engine_decision as m; print(m.__file__)"
C:\Users\gonza\Documents\Claude-Quant-Lab\src\alpha_lab\agents\data_infra\ml\engine_decision.py   (+ strategy_contract.py likewise OK)
```

**Which QL modules import each Part B symbol (grep of the QL tree):**
- `build_zones` / `detect_touches`: `QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:58,60` (used `:218-219`); scripts `QL:scripts/audit_NQ_20260602/trace_leakage.py:17`, `QL:scripts/audit_NQ_20260602/enrich_dates.py:31-32`, `QL:scripts/v3_verify/census_v3.py:29`, `QL:scripts/v3_verify/ny_baseline_enrich.py:35,37`, `QL:scripts/levels_probe/probe.py:24` (+ `is_touch` `:25`).
- Six features: `QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:55-57,61-63` (all six; used `:458-465` interaction, `:527` app_max_spread).
- `resolve_honest_outcome`: `QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:65` (used `:261`, `:746`); `QL:scripts/audit_NQ_20260602/enrich_dates.py:33`; `QL:scripts/v3_verify/ny_baseline_enrich.py:38`.
- `StrategyRuntime` (the level-fold owner via plugin): function-local import `QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:628`, constructed `:646`, driven `:669`.
- Section/registry (feature_windows carriers): `QL:src/alpha_lab/agents/data_infra/ml/strategy_contract.py:52-59` (`strategy_core.strategies.touch_reversal` registration import, `get_strategy`, `TouchReversalSection`, `default_touch_reversal_section`, `validate_feature_partition`); tests `QL:tests/agents/test_strategy_contract_nodrift.py:37,230`.
- `CandleEngine`, `build_tick_bars_from_frame`, `StrategyLevelState`, `prior_full_day_extremes`, `StreamingHonestResolver`: **zero QL importers**. A full-tree grep for those five names returns exactly one hit, a docstring mention of `CandleEngine` at `QL:src/alpha_lab/agents/data_infra/tick_store.py:748`.

### D3 — Runtime/contract/registry couplings

**The pure Part B chain requires none of the three.** `build_zones`, `is_touch`/`detect_touches`, the six feature functions, `resolve_honest_outcome`, `classify_mae_first`/`resolve_outcome`, `StrategyLevelState`, `CandleEngine`, and `build_tick_bars_from_frame` take only plain scalars, engine dataclasses, and (for the resolvers) one injected callable — no contract object, no bundle field, no registry call anywhere in their bodies (files read in full: `SC:src/strategy_core/decisions/zones.py`, `touch.py`, `features.py`, `honest_entry.py`, `streaming.py`, `SC:src/strategy_core/runtime/levels.py`, `SC:src/strategy_core/candles/batch.py`, `streaming.py`). The only injected dependency is the fill accessor: `trade_price_at: Callable[[datetime], float | None]` (`SC:src/strategy_core/decisions/honest_entry.py:79`; `SC:src/strategy_core/decisions/streaming.py:191`). `StreamingHonestResolver`'s docstring says "One resolver instance serves one runtime/contract" (`SC:src/strategy_core/decisions/streaming.py:175`) but its constructor takes only plain parameters, plus an optional fail-loud guard `available_timeframes: Sequence[int] | None` (`:196-205`).

**Couplings that do exist:**

1. `StrategyRuntime` default construction → **registry lookup**. `SC:src/strategy_core/runtime/state.py:183-195`:
```python
def __init__(
    self,
    *,
    timeframes: tuple[int, ...] = (147, 987, 2000),
    ...
    plugin: StrategyPlugin | None = None,
    strategy_section: Any | None = None,
) -> None:
```
When `plugin is None` it auto-attaches via the registry — `SC:src/strategy_core/runtime/state.py:214-217`:
```python
            from strategy_core.runtime.wiring import touch_reversal_kwargs

            _defaults = touch_reversal_kwargs()
            plugin = _defaults["plugin"]
```
which resolves `"plugin": get_strategy("touch_reversal")()` (`SC:src/strategy_core/runtime/wiring.py:33`). The registry table starts empty (`_REGISTRY: dict[str, type[StrategyPlugin]] = {}`, `SC:src/strategy_core/strategies/registry.py:35`) and is populated only by the `@register` side effect at plugin import (`SC:src/strategy_core/strategies/touch_reversal/plugin.py:195`); `wiring.py:21` performs that import, so the default path is self-contained. `get_strategy` fails closed with `ContractError` on an unknown id (`SC:src/strategy_core/strategies/registry.py:98-103`). No contract/bundle is needed: QL constructs it bare — `runtime = StrategyRuntime(timeframes=(tick_count,), requested_symbol=symbol)` (`QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:646`).

2. Plugin configuration → **contract SECTION model (not a bundle)**. `TouchReversalPlugin.configure(self, section: TouchReversalSection, ctx: PlatformContext)` (`SC:src/strategy_core/strategies/touch_reversal/plugin.py:220`); detection reads `self._section.touch_rule.zone_proximity_pts` (`:276-281`). The section is obtainable with no bundle: `default_touch_reversal_section()` builds it entirely from `strategy_core.constants` (`SC:src/strategy_core/strategies/touch_reversal/section.py:157,178-205`).

3. `feature_windows` readers. SC defines the model (`SC:src/strategy_core/contract/schema.py:151-163`: `interaction_window_minutes`, `approach_window_minutes`, `within_band_pts`, `level_proximity_pts`, `large_trade_threshold`, `mid_price_source`) and holds it on the section (`SC:src/strategy_core/strategies/touch_reversal/section.py:91`), but **no SC engine code reads its values into the Part B functions** — the six functions take those knobs as explicit kwargs defaulted from `constants` (`SC:src/strategy_core/decisions/features.py:74,108-109,136,171`); a within-SC grep for `interaction_window_minutes|approach_window_minutes` hits only the schema definition and the default-section constructor. The value readers are consumers: Trade-Lab reads the ACTIVE BUNDLE's section (contract-coupled) at `TL:backend/src/trade_lab/services/runtime.py:424-427`, `TL:backend/src/trade_lab/services/model_registry.py:85-86`, and `TL:backend/src/trade_lab/services/inference/features/feature_functions.py:98,323-327`; QL's labeling reads its OWN run config, not a contract (`window_minutes = config.interaction_window_minutes`, `QL:src/alpha_lab/agents/data_infra/ml/engine_decision.py:678-681`), and QL's emitter writes the values INTO the contract section from that config (`QL:src/alpha_lab/agents/data_infra/ml/strategy_contract.py:97-103`).

4. `prior_full_day_extremes` → **store reader** (neither runtime nor contract nor registry): it constructs `DatabentoParquetSource.for_trading_day(...)` over a filesystem store (`SC:src/strategy_core/data/prior_day.py:39,95-97`) — the only Part B-adjacent symbol whose execution is inseparable from parquet paths.
