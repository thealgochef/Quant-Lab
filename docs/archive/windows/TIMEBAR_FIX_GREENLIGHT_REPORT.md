# TIMEBAR_FIX_GREENLIGHT_REPORT — landed 2026-07-28, CI ×3 green on pandas-3 cold runners

## Step 1 — state verification (all PASS)

- SC 5 ahead of origin/platform-refactor, tracked-clean · TL `6c21b21` clean 0-ahead ·
  QL `9c2db53` clean 0-ahead.
- `git log --oneline origin/platform-refactor..HEAD` (SC, newest first — shas and order exact):

```
719190b test(candles): time-bar parity with the 60s base excluded (TIMEBAR-FIX P5)
a8fc22b fix(candles): remaining pandas-resolution assumptions + stale pandas-import rule (TIMEBAR-FIX P4)
fcff9f0 fix(candles): unit-explicit ns extraction — .asi8/.value return the dtype's own unit, which pandas 3 no longer renormalizes (TIMEBAR-FIX P3)
95df548 test(candles): version-independent guard — ns-integer extraction must not inherit a non-ns dtype unit (TIMEBAR-FIX P2)
95570df refactor(candles): extract ns-integer helpers in the time batch builder (TIMEBAR-FIX P1)
```

- Diff stat: **4 files, +119/−6** — `candles/batch.py` (6 chg) · `candles/time_batch.py`
  (+57) · `tests/test_time_bar_parity.py` (+20) · `tests/test_time_bars.py` (+42).
  `runtime/state.py`, `contract/schema.py`, `candles/streaming.py`,
  `candles/time_streaming.py`, `candles/_buckets.py`, `validation/_fixtures/`: in NO hunk
  (name-only list verified).

## Step 2 — corrected UNIT NOTE (full docstring as committed)

```
Vectorized (pandas + numpy) TIME-bar builder over a trades DataFrame (Phase F).

The batch twin of :class:`~strategy_core.candles.time_streaming.TimeBarEngine`,
mirroring ``build_tick_bars_from_frame``'s signature shape and input contract
(``ts_event`` UTC, ``price`` points, ``size``). Same architect-ratified rules:

* ``bucket = floor((ts_utc - day_start_utc) / interval)`` anchored at the trading-day
  boundary INSTANT (:func:`~strategy_core.candles._buckets.trading_day_start_utc`) —
  DST-correct by construction; here computed in integer nanoseconds.
* DERIVE ONCE, AGGREGATE UPWARD: one 60s aggregation over trades; every requested
  timeframe (60s included) is a groupby over those 60s rows via
  ``hbucket = bucket60 * 60 // interval`` — never a second pass over trades.
  Higher OHLC = first open / max high / min low / last close over constituent 60s
  rows; volume / trade_count are sums over the 60s rows that EXIST (skipped empty
  minutes contribute nothing).
* NO EMPTY BARS; ``bar_index`` dense over emitted bars per (timeframe, trading_day).
* COMPLETENESS: the last bar of each (timeframe, trading_day) is ``END_OF_DAY``
  incomplete (no later bucket produced a bar); every other bar is ``COMPLETE``.

The parity test (``tests/test_time_bar_parity.py``) locks this against the
streaming engine. Pandas-import rule: the BATCH builders (``candles/batch.py`` and
this module) import pandas; the streaming engines and ``candles/_buckets.py`` do
not. The batch path is the research / warm-up fast path, run off the event loop.

UNIT NOTE — batch buckets in integer NANOSECONDS while the streaming engine buckets
in integer MICROSECONDS, and the two are provably equivalent: for positive integers
``floor(x / (a*b)) == floor(floor(x / a) / b)``, so flooring a nanosecond timestamp
to microseconds (a=1000) and then to buckets (b=interval*1e6 us) lands in the same
bucket as flooring the nanoseconds directly by interval*1e9 — PROVIDED the ns->us
step is a TRUNCATION (floor), not a rounding.
That truncation is ENGINE-side, not reader-side. The reader PRESERVES full
nanosecond precision: ``event_ts_utc`` is a ns-unit ``pd.Timestamp``
(``data/databento_parquet.py:618``, ``:624-625``; the ``raw // 1000`` at ``:621``
feeds only the window masks, never the event objects), and real store timestamps
DO carry sub-microsecond components. The truncation happens in
``candles/time_streaming.py``, whose bucket arithmetic decomposes
``event_ts_utc - day_start`` via ``Timedelta.days/seconds/microseconds`` and
DISCARDS ``.nanoseconds``. Floor-composition therefore applies end to end and
batch-in-ns == stream-in-us holds on real ns-precision store data — VERIFIED at
TIMEBAR-FIX Part 8b (probes: +60s+999ns -> bucket 1 on both paths;
+59.999999999s -> bucket 0 on both paths).

BUCKET ASSIGNMENT ONLY. The emitted ``open_ts_utc`` / ``close_ts_utc`` are NOT
proven equal across the two paths on real data: the streaming engine assigns the
ns-precision ``pd.Timestamp`` straight onto the bar, while this module emits
stdlib ``datetime`` at us precision via ``.dt.to_pydatetime()``. On a trade
carrying nonzero sub-microsecond nanoseconds those values differ. The parity
harness cannot see it — its synthetic streams are built from stdlib ``datetime``
(zero ns). The tick path has the same structure and the same exposure
(``candles/streaming.py`` vs ``candles/batch.py``), so this is pre-existing, not
introduced here. OPEN — TIMEBAR-FIX architect review finding.
```

## Step 4 — greenlight docs commit

**SC `80e244d`** — `docs: correct the TIMEBAR-FIX unit note (truncation is engine-side,
verified) + record the window` (docs/PROGRESS +87; time_batch.py docstring-only +28/−6).

## Steps 5/8 — pushed tips

- **SC** `origin/platform-refactor` = **`80e244d`** (80e244de2df88a3c2eb13471b7349b7adb22b37f)
- **TL** `origin/platform-refactor` = **`9acdc49`** (9acdc498d23ad8c6296e1f49040a766412c4a2fe)
- **QL** `origin/platform-refactor` = **`a895e80`** (a895e80cec30417a7e9ae619b794a537fb79dc9a)

## Step 6 — ancestor assertions (exit 0 = ancestor)

| sha | is-ancestor of origin/platform-refactor |
|---|---|
| 95570df | PASS (0) |
| 95df548 | PASS (0) |
| fcff9f0 | PASS (0) |
| a8fc22b | PASS (0) |
| 719190b | PASS (0) |
| 8ec6906 | PASS (0) |

`git log --oneline 80e244d~6..80e244d`: the five reviewed shas UNCHANGED and in order,
with only the Step-4 docs commit `80e244d` above them.

## Step 7 — pin bumps (target fcff9f0, the latest consumer-facing commit)

TL `backend/pyproject.toml:18` — BEFORE:
```
  "strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@1eca72f9d9c56a204e0e7c974ca8268ea85f36e5",
```
AFTER:
```
  "strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@fcff9f079467ee54b0768c5e07c4852f53e949c4",
```

QL `pyproject.toml:36` — BEFORE:
```
    "strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@1eca72f9d9c56a204e0e7c974ca8268ea85f36e5",
```
AFTER:
```
    "strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@fcff9f079467ee54b0768c5e07c4852f53e949c4",
```

Pin chores: TL `9acdc49` · QL `a895e80`.

## Step 9 — CI witness

| repo | workflow | run id | run # | head sha | conclusion |
|---|---|---|---|---|---|
| Strategy-Core | ci | 30335635385 | 17 | 80e244d | **success** |
| Trade-Lab | backend-ci | 30335704295 | 18 | 9acdc49 | **success** |
| Quant-Lab | ci | 30335714968 | 16 | a895e80 | **success** |

### 9b — SC cold-runner pandas version (from run log, job 90199789670)

```
Successfully installed annotated-types-0.8.0 iniconfig-2.3.0 numpy-2.5.1 packaging-26.2 pandas-3.0.5 pluggy-1.6.0 pyarrow-25.0.0 pydantic-2.13.4 pydantic-core-2.46.4 pygments-2.20.0 pytest-9.1.1 python-dateutil-2.9.0.post0 ruff-0.15.22 six-1.17.0 strategy-core-0.1.0 typing-extensions-4.16.0 typing-inspection-0.4.2 tzdata-2026.3
```

**pandas 3.0.5** — the exact version of the red run 30331205481. The pytest step ran
215 passed / 9 skipped (store-guarded real-data tests; the runner has no store), zero
failures — the four previously-red time-bar tests now green on the failing configuration.

### 9c — QL cold-install strategy-core resolution (from run log, job 90200023828)

```
Collecting strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@fcff9f079467ee54b0768c5e07c4852f53e949c4 (from alpha-signal-lab==0.1.0)
  Resolved https://github.com/thealgochef/Strategy-Core.git to commit fcff9f079467ee54b0768c5e07c4852f53e949c4
```
