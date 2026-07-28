# TIMEBAR_FIX_REPORT — pandas-3 unit fix, verified on both majors (2026-07-28)

Window ends in FULL STOP: 5 SC commits above `8ec6906`, NOT pushed, no pin moved, no
pandas ceiling anywhere. Review diff: `Strategy-core\TIMEBAR_FIX_SC_DIFF.txt`.

## Tips & runtimes

- SC `8ec6906` clean 0-ahead at start (now 5 ahead, tracked-clean) · TL `6c21b21` clean ·
  QL `9c2db53` clean — no TL/QL tracked change at any point.
- Host: `strategy_core` = SC working tree (editable), **pandas 2.3.1**.
- Throwaway venv `C:\Users\gonza\Documents\_pandas3_check\venv` (outside all three
  repos, left in place): SC installed from the working tree, **pandas 3.0.5** — the
  exact version CI resolved on the red run 30331205481.

## Commits (in order, above 8ec6906)

- `95570df` refactor(candles): extract ns-integer helpers in the time batch builder (TIMEBAR-FIX P1)
- `95df548` test(candles): version-independent guard — ns-integer extraction must not inherit a non-ns dtype unit (TIMEBAR-FIX P2)  ← RED at this commit by design
- `fcff9f0` fix(candles): unit-explicit ns extraction — .asi8/.value return the dtype's own unit, which pandas 3 no longer renormalizes (TIMEBAR-FIX P3)
- `a8fc22b` fix(candles): remaining pandas-resolution assumptions + stale pandas-import rule (TIMEBAR-FIX P4)  ← docstring-only; 4a sweep found no further code change
- `719190b` test(candles): time-bar parity with the 60s base excluded (TIMEBAR-FIX P5)

## Part 2b — the guard, pre-fix, on host pandas 2.3.1 (verbatim tail)

The prior attempt through the PUBLIC input passed on pandas 2 (DataFrame construction
renormalizes to ns there); this guard hits the extraction helpers directly with an
`as_unit`-forced index and FAILED pre-fix on ALL THREE units:

```
>       assert int(out[1]) - int(out[0]) == 60_000_000_000, (
            f"_index_to_ns returned unit-{unit} integers, not nanoseconds: "
            f"delta={int(out[1]) - int(out[0])}"
        )
E       AssertionError: _index_to_ns returned unit-s integers, not nanoseconds: delta=60
E       assert (1748901660 - 1748901600) == 60000000000
=========================== short test summary info ===========================
FAILED tests/test_time_bars.py::test_ns_extraction_does_not_inherit_dtype_unit[us]
FAILED tests/test_time_bars.py::test_ns_extraction_does_not_inherit_dtype_unit[ms]
FAILED tests/test_time_bars.py::test_ns_extraction_does_not_inherit_dtype_unit[s]
```

(us delta = 60_000_000 · ms delta = 60_000 · s delta = 60.)

## Part 3c — post-fix

`tests/test_time_bars.py` + `tests/test_time_bar_parity.py`: 17 passed, 0 failed —
guard green on us/ms/s; all prior tests unchanged.

## Part 4a — sweep of candles/time_batch.py (post-fix line numbers)

| site | expression | resolution/version assumption | verdict |
|---|---|---|---|
| :132 | `is_datetime64_any_dtype` | matches any datetime64 resolution/tz on both majors | PASS |
| :133 | `pd.to_datetime(ts, utc=True, unit="ns")` | numeric-input unit pinned explicitly; downstream integers unit-forced regardless | PASS |
| :134-137 | `tz_localize` / `tz_convert` | resolution-preserving; no integer read | PASS |
| :142 | `ts.to_numpy()` | object array of Timestamps on both majors; no unit read | PASS |
| :147 | `pd.to_datetime(work["ts_event"], utc=True)` | pandas 2 renormalizes to ns, pandas 3 preserves source unit — the original red's vector; all downstream integer reads now go through the forced helpers | PASS (fixed downstream, P3) |
| :149-150 | `.dt.tz_convert` + `.dt.hour/minute/second` | wall-clock components, resolution-independent | PASS |
| :151 | `.dt.tz_localize(None).dt.floor("D")` | floor-to-day defined for every resolution; same-unit output | PASS |
| :153-154 | `pd.to_timedelta(roll, unit="D")` + add | timedelta unit explicit; no integer read | PASS |
| :170 | `work["trading_day"].unique()` elements + `td.date()` | element type could shift across majors; would fail LOUD (AttributeError), not rescale — empirically fine under 3.0.5 (Part 7 parity green) | PASS (empirical) |
| :173 | `.map(day_start_ns)` datetime-keyed dict | key-hash mismatch would yield NaN → loud `int64` cast error, never a silent rescale — empirically fine under 3.0.5 | PASS (empirical) |
| :169/:172 | `_timestamp_to_ns` / `_index_to_ns` | THE datetime→integer sites; `.asi8`/`.value` return the dtype's OWN unit | **FIXED (P3)** — `as_unit("ns")` forced + loud unit assertion |
| :174/:195 | `//` bucket divisions | pure int64 numpy | PASS |
| :180-206 | groupby `first/last/max/min/sum` | order/value-based, resolution-independent | PASS |
| :216 | `.dt.date` | date component, resolution-independent | PASS |
| :226-227 | `.dt.to_pydatetime()` | ns→µs discard TRUNCATES on both majors (warning suppressed by design, as in batch.py); pandas 3 (µs column) is exact | PASS |

4b — tick `candles/batch.py` has NO datetime→integer extraction; bucketing is
`work["bar_index"] = (day_groups.cumcount() // timeframe).to_numpy()` (batch.py:124) —
count-based, no resolution dependence. No exposure.

4c — both batch-builder docstrings corrected: the real rule is "the batch builders
import pandas; the streaming engines and `_buckets.py` do not" (batch.py had claimed to
be the ONLY pandas-importing engine module).

## Part 6 — host verification (pandas 2.3.1)

- 6a full SC suite: **exit 0, all passed, 0 failed** (224 collected = the prior 219 + 3
  guard params + 2 no-base parity params).
- 6b frozen digests, explicit run: `test_b3_golive_plugin_regression` PASSED (golive
  fixture matched) · `test_b3_multiday_reset_plugin_regression` PASSED (multiday fixture
  matched) — **both fixtures unchanged**, nothing regenerated.
- 6c `tests/test_candle_parity.py` (tick): 3/3 passed unchanged. (6b+6c together: 5
  passed in 7m01s.)
- 6d consumers against the editable SC: **TL 511 passed / 1 skipped · QL 830 passed**.

## Part 7 — verification under pandas 3.0.5 (the gate CI failed)

- 7a/7b venv at `C:\Users\gonza\Documents\_pandas3_check\` (outside all repos); SC
  installed from the working tree; resolved **pandas 3.0.5**.
- 7c per-file: `test_time_bar_parity.py` **7 passed** · `test_time_bars.py` **12
  passed** · `test_candle_parity.py` **3 passed** — zero failures; the CI red class is
  gone.
- 7d full SC suite in the venv: **exit 0 — 221 passed, 3 skipped, 0 failed.** The 3
  skips are environment-guarded, not pandas findings: `test_decision_diff.py:65`
  (databento store / QL src not wired in the venv), `test_duckdb_streaming_parity.py:174`
  and `test_production_pair_parity.py:197` (both `No module named 'duckdb'` — duckdb is
  not an SC dependency and was not installed in the throwaway venv). On the host all
  three run and pass. No pre-existing pandas-3 failure surfaced anywhere.
- 7e venv left in place; contains only `venv/`; outside all three repos.

## Part 8a — census spot check (host, read-only, fix must be a no-op on pandas 2)

Full idempotent re-run of `scratch_root_census.py` (the script is windowed by
construction and series (a) is history-dependent, so a literal single-day run cannot
produce it; the re-run compares at 2026-02-13). Against FVG_CENSUS.md:

| target | census | re-run | verdict |
|---|---|---|---|
| trades 2026-02-13 | 486,819 | 486,819 | MATCH |
| 1m bars/empty | 1380/60 | 1380/60 | MATCH |
| 4H bars/empty | 6/0 | 6/0 | MATCH |
| series (a) day total | 4 | 4 | MATCH |
| series (b) day total | 5 | 5 | MATCH |

Stronger: the ENTIRE 20-day results JSON is identical to the original run excluding
wall-clock timing fields — the fix is a proven no-op on pandas 2 across every number
the census produces.

## Part 8b — reader finding (read-only; resolves the equivalence premise)

- Store column: `ts_event: timestamp[ns, tz=UTC]` (pyarrow schema, NQ/2026-02-13
  mbp10.parquet — metadata only). Sub-microsecond nanosecond components CAN and DO occur
  in real store timestamps (the validation harness's "Discarding nonzero nanoseconds in
  conversion" warnings fire on real data).
- Producing expression: `data/databento_parquet.py:618` decodes the raw int64
  nanoseconds (`column.to_numpy(...).view("i8")`); for ns-kind columns
  `:624-625` builds the event objects as `list(pandas.DatetimeIndex(values, tz="UTC"))`
  over those raw ns, and `:176/:186/:212/:221` assign them to `event_ts_utc`. The
  conversion **PRESERVES** full nanosecond precision — `event_ts_utc` is a ns-unit
  `pd.Timestamp`; no ns→µs conversion happens at the reader for event objects. (The
  `ts_us = raw // 1000` at `:621` is a TRUNCATION but feeds only the window masks.)
- Downstream: the streaming engine's bucket arithmetic decomposes
  `event_ts_utc − day_start` via `Timedelta.days/seconds/microseconds`, which DISCARDS
  the `.nanoseconds` component — a TRUNCATION (probe: +60s+999ns → 60,000,000 µs,
  bucket 1; +59.999999999s → 59,999,999 µs, bucket 0). Therefore the floor-composition
  identity `floor(x/(a·b)) == floor(floor(x/a)/b)` applies end-to-end and
  **batch-in-ns ≡ streaming-in-µs on real ns-precision store data** — the premise named
  unverified in the module docstring and the TIMEBAR review finding (ii) is now
  VERIFIED (report-level; no code changed for it).
