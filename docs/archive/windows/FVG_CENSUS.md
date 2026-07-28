# FVG_CENSUS — root-structure material, 15 counted days ending 2026-02-13

Calibration material + real-data smoke of the Phase-F time-bar builders. **No
recommendations, no verdicts — distributions only.** Every number is reproducible from
`scratch_root_census.py` (results JSON: `scratch_root_census_results.json`). Bars are
the Phase-F **batch** time bars (`strategy_core.candles.time_batch.build_time_bars_from_frame`)
built from ONE canonical-reader drain per day (`DatabentoParquetSource.for_trading_day`,
requested_symbol=NQ). Engine under test: SC working tree at TIMEBAR C1–C6
(editable install; `strategy_core.__file__` = SC `src/` tree).

## Definitions (verbatim, so the numbers are interpretable without the script)

**FVG** — per timeframe independently, at that timeframe's own bar closes, over the bar
series CONCATENATED across days (triplets may span the trading-day boundary; both
COMPLETE and END_OF_DAY bars participate):

    Bullish at bar i (i>=2): bars[i-2].high < bars[i].low; interval [bars[i-2].high, bars[i].low]
    Bearish at bar i:        bars[i-2].low  > bars[i].high; interval [bars[i].high, bars[i-2].low]
    Formation ts = bars[i].close_ts_utc.  Size = interval width in ticks (tick = 0.25).

NO minimum-size filter is applied — size floors are a downstream research choice.

**Fill tracking** — against the 60s series, wicks counting (closed-interval overlap);
only 60s bars with close_ts STRICTLY AFTER formation ts participate:

    FIRST TOUCH = first 60s bar whose [low, high] overlaps the gap interval at all.
    FULL FILL   = first 60s bar by which price has cumulatively traversed the entire
                  interval (bullish: running min(low) <= interval lo; bearish:
                  running max(high) >= interval hi).

Elapsed 60s-bar count = number of emitted 60s bars with close_ts in (formation, event].

**Key-level taps** — the day's trades fold through the EXISTING `StrategyLevelState`
(day 1 seeded via the QL dataset builder's `_get_session_hl_for_date` walk: seed day
2026-01-16, H/L ticks 103564/102366); zones from `build_zones` at default proximity
(3.0 pts); detection against the day's 60s bars.

* **Geometry 1** = the existing representative-price straddle (`detect_touches` as-is;
  `available_from` gate enforced; first-touch-per-zone-per-day scope).
* **Geometry 2** = bar range intersects [rep − 4 ticks, rep + 4 ticks] (same gate, same
  first-touch scope; auxiliary scan).

A merged zone's tap counts toward EVERY constituent level source (the totals therefore
double-count a tap of a merged zone once per constituent).

**E6 overlap** — a geometry-1 first tap coincides with an FVG root at tolerance ±N ticks
iff the SAME 60s bar also FIRST-touched a live unfilled 1H or 4H gap AND the zone
representative price lies within N ticks of that gap's interval.

## P4 level-source inventory (pre-flight, SC `runtime/levels.py` @ 4740ecd)

`StrategyLevelState` tracks, exactly:

| source | status | `available_from` |
|---|---|---|
| prior-day high (`pdh`) | PRESENT | trading-day start (prior calendar day 18:00 ET → UTC) |
| prior-day low (`pdl`) | PRESENT | trading-day start |
| Asia high/low (`asia_high`/`asia_low`) | PRESENT | Asia session close (02:45 ET on the trading day) |
| London high/low (`london_high`/`london_low`) | PRESENT | London session close (08:00 ET) |
| **NY high / NY low** | **ABSENT** | — (no NY range map exists; adding one is a separate window) |

Range maps: `self._ranges = {"asia": _Range(), "london": _Range()}` (levels.py:49) — no
"ny" key. Day-roll banking: on a trading-day change the completed day's extremes are
banked into `_summaries` before reset (levels.py:75-81), so PDH/PDL emit organically on
multi-day streams; an explicit `load_prior_day_summary` stays authoritative.

## Window (E1)

20 consecutive trading days with store data ending 2026-02-13; none in the sealed range
2026-06-12..2026-07-10 (window is entirely January–February 2026).

* **WARMUP (feed registries, excluded from headline aggregates):**
  2026-01-19 (MLK holiday, early close), 2026-01-20, 2026-01-21, 2026-01-22, 2026-01-23
* **COUNTED (15):** 2026-01-26, 01-27, 01-28, 01-29, 01-30, 02-02, 02-03, 02-04, 02-05,
  02-06, 02-09, 02-10, 02-11, 02-12, 2026-02-13

## E2 — per-day build (bars / empty wall-clock buckets)

Format `bars/empty`. A full 24-hour trading day has wall-clock maxima 1440 (60s) and 6
(4H). Every regular day shows **1380/60** at 60s — exactly the daily 17:00–18:00 ET
maintenance halt (the one empty 1H bucket per day is that same hour; the 4H bucket
containing it still trades its other 3 hours, hence 6/0). The MLK warmup day (early
close ~13:00 ET) shows 1140/300.

| day | trades | 1m | 3m | 5m | 10m | 15m | 30m | 1H | 4H |
|---|---|---|---|---|---|---|---|---|---|
| 2026-01-19 W | 118,945 | 1140/300 | 380/100 | 228/60 | 114/30 | 76/20 | 38/10 | 19/5 | 5/1 |
| 2026-01-20 W | 470,749 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-01-21 W | 538,899 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-01-22 W | 345,934 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-01-23 W | 323,452 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-01-26 | 287,927 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-01-27 | 271,454 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-01-28 | 337,113 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-01-29 | 466,885 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-01-30 | 494,438 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-02-02 | 387,291 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-02-03 | 476,247 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-02-04 | 522,013 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-02-05 | 552,811 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-02-06 | 449,498 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-02-09 | 331,620 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-02-10 | 317,269 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-02-11 | 366,325 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-02-12 | 445,777 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |
| 2026-02-13 | 486,819 | 1380/60 | 460/20 | 276/12 | 138/6 | 92/4 | 46/2 | 23/1 | 6/0 |

**Smoke figures:** per-day 60s bar count median 1380 (max obs 1380) vs wall-clock max
1440; per-day 4H count median 6 vs max 6. Every trading minute outside the maintenance
hour printed at least one trade on every regular day — no empty mid-session buckets at
60s in this window; empties are structural (halt/holiday), not liquidity holes.

## E3 — formation-span / dead-time properties (counted-day gaps)

Recorded per gap: wall-time between bars[i-2].close and bars[i].close, and empty 60s
buckets spanned. Gaps formed across dead time are recorded, not filtered.

| tf | n gaps | span wall-s median | span wall-s max | empty-60s median | empty-60s max | share with any dead time |
|---|---|---|---|---|---|---|
| 1m | 4760 | 120 | 176,521 | 0 | 60 | 0.004 |
| 3m | 1463 | 360 | 176,763 | 0 | 60 | 0.008 |
| 5m | 859 | 600 | 177,002 | 0 | 60 | 0.014 |
| 10m | 422 | 1,200 | 177,604 | 0 | 60 | 0.036 |
| 15m | 293 | 1,800 | 178,202 | 0 | 60 | 0.041 |
| 30m | 141 | 3,600 | 180,004 | 0 | 60 | 0.099 |
| 1H | 77 | 7,200 | 183,600 | 0 | 60 | 0.104 |
| 4H | 23 | 28,800 | 205,200 | 0 | 60 | 0.217 |

Median spans equal the nominal 2-bucket wall time at every timeframe (dense bars); the
maxima are weekend-spanning triplets (~49-57h). Dead time inside a formation span is
always exactly the 60-minute maintenance halt when present (max empty-60s = 60
everywhere); at 4H, 21.7% of gaps span it.

## E4 — key-level roots (counted days)

Per source and day: `G1 first / G1 total (incl. re-taps) | G2 first / G2 total |
suppressed-by-available_from`. G1 = representative-price straddle as-is; G2 = ±4-tick
band. Direction of a tap follows the zone side (HIGH→SHORT, LOW→LONG).

| day | pdh | pdl | asia_high | asia_low | london_high | london_low |
|---|---|---|---|---|---|---|
| 01-26 | 1/25 \| 1/34 \| 0 | 1/12 \| 1/13 \| 0 | 1/30 \| 1/40 \| 1 | 0/0 \| 0/0 \| 6 | 1/6 \| 1/6 \| 1 | 0/0 \| 0/0 \| 25 |
| 01-27 | 1/3 \| 1/5 \| 0 | 0/0 \| 0/0 \| 0 | 1/17 \| 1/21 \| 1 | 0/0 \| 0/0 \| 10 | 1/9 \| 1/11 \| 1 | 1/8 \| 1/8 \| 19 |
| 01-28 | 1/23 \| 1/29 \| 0 | 0/0 \| 0/0 \| 0 | 0/0 \| 0/0 \| 1 | 1/45 \| 1/49 \| 22 | 0/0 \| 0/0 \| 8 | 1/36 \| 1/39 \| 2 |
| 01-29 | 0/0 \| 0/0 \| 0 | 1/26 \| 1/29 \| 0 | 1/5 \| 1/7 \| 5 | 1/2 \| 1/2 \| 1 | 0/0 \| 0/0 \| 4 | 1/17 \| 1/20 \| 37 |
| 01-30 | 0/0 \| 0/0 \| 0 | 0/0 \| 0/0 \| 0 | 0/0 \| 0/0 \| 12 | 1/79 \| 1/87 \| 1 | 1/27 \| 1/27 \| 18 | 1/40 \| 1/43 \| 1 |
| 02-02 | 0/0 \| 0/0 \| 0 | 1/6 \| 1/7 \| 0 | 1/2 \| 1/2 \| 1 | 0/0 \| 0/0 \| 1 | 1/16 \| 1/20 \| 16 | 0/0 \| 0/0 \| 28 |
| 02-03 | 1/66 \| 1/87 \| 0 | 0/0 \| 0/0 \| 0 | 1/12 \| 1/17 \| 1 | 1/2 \| 1/2 \| 2 | 0/0 \| 0/0 \| 8 | 1/2 \| 1/2 \| 33 |
| 02-04 | 0/0 \| 0/0 \| 0 | 1/26 \| 1/26 \| 0 | 0/0 \| 0/0 \| 9 | 1/29 \| 1/35 \| 5 | 0/0 \| 0/0 \| 9 | 1/19 \| 1/20 \| 1 |
| 02-05 | 0/0 \| 0/0 \| 0 | 1/73 \| 1/78 \| 0 | 0/0 \| 0/0 \| 2 | 1/20 \| 1/20 \| 1 | 0/0 \| 0/0 \| 31 | 1/43 \| 1/46 \| 1 |
| 02-06 | 1/2 \| 1/4 \| 0 | 1/15 \| 1/17 \| 0 | 1/1 \| 1/2 \| 1 | 0/0 \| 0/0 \| 1 | 1/23 \| 1/23 \| 1 | 0/0 \| 0/0 \| 17 |
| 02-09 | 1/58 \| 1/70 \| 0 | 0/0 \| 0/0 \| 0 | 1/31 \| 1/35 \| 1 | 1/32 \| 1/33 \| 1 | 1/4 \| 1/4 \| 58 | 1/2 \| 1/2 \| 2 |
| 02-10 | 1/9 \| 1/9 \| 0 | 0/0 \| 0/0 \| 0 | 1/108 \| 1/125 \| 8 | 1/26 \| 1/28 \| 1 | 1/10 \| 1/12 \| 1 | 1/30 \| 1/36 \| 68 |
| 02-11 | 1/6 \| 1/7 \| 0 | 1/33 \| 1/34 \| 0 | 1/28 \| 1/29 \| 1 | 1/72 \| 1/80 \| 2 | 1/72 \| 1/80 \| 51 | 1/11 \| 1/12 \| 1 |
| 02-12 | 0/0 \| 0/0 \| 0 | 1/2 \| 1/2 \| 0 | 1/69 \| 1/88 \| 1 | 1/17 \| 1/20 \| 5 | 1/14 \| 1/17 \| 1 | 1/4 \| 1/6 \| 28 |
| 02-13 | 0/0 \| 0/0 \| 0 | 1/110 \| 1/123 \| 0 | 1/27 \| 1/27 \| 1 | 1/10 \| 1/13 \| 1 | 1/40 \| 1/42 \| 59 | 1/4 \| 1/4 \| 1 |
| **Σ** | **8/192 \| 8/245 \| 0** | **9/303 \| 9/329 \| 0** | **11/330 \| 11/393 \| 46** | **11/334 \| 11/369 \| 60** | **10/221 \| 10/242 \| 267** | **12/216 \| 12/238 \| 264** |

Observations recorded (not conclusions): geometry 2 (±4t band) never adds a first touch
over 15 days — every zone G1-first-touched is the same set G2-first-touches (the band
only inflates re-tap totals ~10-15%). The `available_from` gate suppresses heavily on
London levels (267/264 gated straddling bars) — London extremes form during a period
price revisits before 08:00 ET; a 0/0 row with large suppression (e.g. london_low
01-26: 25 suppressed, 0 counted) is a level whose only straddles happened before it
became available. PDH/PDL show zero suppression by construction (available from the
day open).

## E5 — end-of-run fill state (all 20 days' gaps, incl. warmup-formed)

| tf | still untouched | touched but unfilled |
|---|---|---|
| 1m | 49 | 27 |
| 3m | 23 | 14 |
| 5m | 17 | 7 |
| 10m | 11 | 7 |
| 15m | 10 | 8 |
| 30m | 6 | 4 |
| 1H | 6 | 3 |
| 4H | 2 | 3 |

## E6 — overlap between root families (counted days)

61 geometry-1 key-level first taps across the 15 counted days. Same-60s-bar
coincidence with a live unfilled 1H/4H FVG first touch:

| tolerance | coincidences | rate |
|---|---|---|
| ±0 ticks | 2 / 61 | 3.3% |
| ±4 ticks | 2 / 61 | 3.3% |
| ±8 ticks | 2 / 61 | 3.3% |

Same-bar events (level tap bar == 1H/4H gap first-touch bar) occurred 5 times on 4
days; in 2 of them the level price lay inside the gap interval (both on 2026-01-28:
london_low with a 4H gap, asia_low with a 1H gap), and widening the band to ±4/±8
ticks adds none. **Fraction of root taps that are the same market event counted twice:
2/61 = 3.3% at every tested tolerance.**

## E7 — distributions (counted-day gaps; the calibration headline)

Sizes in ticks (0.25 pt). "ICT prior" = the TradingView-era clean-gap floor under
discussion. Time-to metrics over gaps that achieved the milestone.

| tf | n | bull/day mean·med | bear/day mean·med | size min·p10·p25·**med**·p75·p90·max | ICT prior | touched | filled | untouched |
|---|---|---|---|---|---|---|---|---|
| 1m | 4760 | 163.1 · 161 | 154.3 · 154 | 1 · 2 · 5 · **12** · 24 · 44 · 784 | 4 | 99.0% | 98.4% | 1.0% |
| 3m | 1463 | 51.7 · 53 | 45.9 · 46 | 1 · 3 · 8 · **19** · 41 · 77 · 835 | 6 | 98.4% | 97.5% | 1.6% |
| 5m | 859 | 30.4 · 31 | 26.9 · 27 | 1 · 5 · 11 · **26** · 50 · 99 · 767 | 8 | 98.0% | 97.2% | 2.0% |
| 10m | 422 | 14.9 · 14 | 13.2 · 14 | 1 · 7 · 15 · **36** · 73 · 157 · 664 | 10 | 97.4% | 95.7% | 2.6% |
| 15m | 293 | 10.9 · 11 | 8.7 · 9 | 1 · 7 · 19 · **41** · 89 · 201 · 658 | 12 | 96.6% | 93.9% | 3.4% |
| 30m | 141 | 5.3 · 5 | 4.1 · 4 | 1 · 11 · 23 · **66** · 140 · 325 · 1123 | 16 | 95.7% | 92.9% | 4.3% |
| 1H | 77 | 2.7 · 2 | 2.5 · 2 | 5 · 14 · 39 · **77** · 215 · 423 · 1259 | 24 | 92.2% | 88.3% | 7.8% |
| 4H | 23 | 0.9 · 0 | 0.6 · 0 | 3 · 18 · 93 · **244** · 580 · 1024 · 1192 | 40 | 91.3% | 78.3% | 8.7% |

Time to first touch / full fill:

| tf | first-touch med (min · 60s bars) | first-touch p90 | full-fill med | full-fill p90 |
|---|---|---|---|---|
| 1m | 2.0 · 2 | 48.0 · 47 | 5.0 · 5 | 129.1 · 118 |
| 3m | 4.1 · 4 | 130.1 · 116 | 13.0 · 13 | 369.0 · 361 |
| 5m | 7.0 · 7 | 205.7 · 190 | 20.0 · 20 | 652.2 · 652 |
| 10m | 13.0 · 13 | 510.0 · 491 | 42.5 · 41 | 943.2 · 896 |
| 15m | 16.0 · 16 | 772.6 · 544 | 72.0 · 66 | 1150.6 · 1018 |
| 30m | 41.0 · 36 | 957.1 · 935 | 157.2 · 156 | 1662.0 · 1568 |
| 1H | 98.0 · 98 | 1901.0 · 1683 | 269.0 · 269 | 3740.1 · 3054 |
| 4H | 1154.0 · 939 | 4081.0 · 3001 | 1472.5 · 1244 | 4981.2 · 3937 |

**Observed median size vs ICT-clean prior, as numbers (no conclusion drawn):**
1m 12t vs 4t · 3m 19t vs 6t · 5m 26t vs 8t · 10m 36t vs 10t · 15m 41t vs 12t ·
30m 66t vs 16t · 1H 77t vs 24t · 4H 244t vs 40t. The observed median exceeds the prior
at every timeframe by 3.0–6.6×.

## E8 — root-tap frequency, three separate per-day series (NEVER summed)

**(a) First touches of unfilled 1H/4H FVGs** (any live gap, touch occurring on the
counted day; split by timeframe_direction):

| day | 1H bull | 1H bear | 4H bull | 4H bear | day total |
|---|---|---|---|---|---|
| 01-26 | 3 | 3 | 1 | 0 | 7 |
| 01-27 | 3 | 1 | 1 | 0 | 5 |
| 01-28 | 6 | 1 | 5 | 0 | 12 |
| 01-29 | 4 | 3 | 1 | 1 | 9 |
| 01-30 | 3 | 3 | 0 | 0 | 6 |
| 02-02 | 0 | 2 | 0 | 1 | 3 |
| 02-03 | 5 | 1 | 3 | 0 | 9 |
| 02-04 | 3 | 3 | 0 | 1 | 7 |
| 02-05 | 0 | 2 | 0 | 0 | 2 |
| 02-06 | 5 | 1 | 0 | 2 | 8 |
| 02-09 | 4 | 5 | 2 | 1 | 12 |
| 02-10 | 1 | 0 | 1 | 0 | 2 |
| 02-11 | 4 | 3 | 0 | 0 | 7 |
| 02-12 | 1 | 1 | 0 | 0 | 2 |
| 02-13 | 0 | 2 | 1 | 1 | 4 |

Series (a): **mean 6.33 / day, median 7**.

**(b) First touches of key levels** (geometry 1; direction follows the level side:
`*_high`→SHORT, `*_low`→LONG):

| day | pdh | pdl | asia_h | asia_l | lon_h | lon_l | day total |
|---|---|---|---|---|---|---|---|
| 01-26 | 1 | 1 | 1 | 0 | 1 | 0 | 4 |
| 01-27 | 1 | 0 | 1 | 0 | 1 | 1 | 4 |
| 01-28 | 1 | 0 | 0 | 1 | 0 | 1 | 3 |
| 01-29 | 0 | 1 | 1 | 1 | 0 | 1 | 4 |
| 01-30 | 0 | 0 | 0 | 1 | 1 | 1 | 3 |
| 02-02 | 0 | 1 | 1 | 0 | 1 | 0 | 3 |
| 02-03 | 1 | 0 | 1 | 1 | 0 | 1 | 4 |
| 02-04 | 0 | 1 | 0 | 1 | 0 | 1 | 3 |
| 02-05 | 0 | 1 | 0 | 1 | 0 | 1 | 3 |
| 02-06 | 1 | 1 | 1 | 0 | 1 | 0 | 4 |
| 02-09 | 1 | 0 | 1 | 1 | 1 | 1 | 5 |
| 02-10 | 1 | 0 | 1 | 1 | 1 | 1 | 5 |
| 02-11 | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| 02-12 | 0 | 1 | 1 | 1 | 1 | 1 | 5 |
| 02-13 | 0 | 1 | 1 | 1 | 1 | 1 | 5 |

Series (b): **mean 4.07 / day, median 4**.

**(c) E6 overlap count** (same-bar + level-in-interval at ±4 ticks):
01-28 = 2; all other counted days = 0. Series (c): **mean 0.13 / day, median 0**.

## E9 — warmup convergence (registry + levels at each day's start)

| day | live unfilled 1H | live unfilled 4H | available key levels |
|---|---|---|---|
| 2026-01-19 W | 0 | 0 | 2 (pdh, pdl — seeded) |
| 2026-01-20 W | 1 | 0 | 2 |
| 2026-01-21 W | 3 | 0 | 2 |
| 2026-01-22 W | 2 | 0 | 2 |
| 2026-01-23 W | 2 | 0 | 2 |
| 2026-01-26 | 4 | 1 | 2 |
| 2026-01-27 | 4 | 1 | 2 |
| 2026-01-28 | 6 | 4 | 2 |
| 2026-01-29 | 5 | 2 | 2 |
| 2026-01-30 | 5 | 1 | 2 |
| 2026-02-02 | 3 | 1 | 2 |
| 2026-02-03 | 6 | 4 | 2 |
| 2026-02-04 | 5 | 3 | 2 |
| 2026-02-05 | 7 | 5 | 2 |
| 2026-02-06 | 8 | 6 | 2 |
| 2026-02-09 | 10 | 6 | 2 |
| 2026-02-10 | 7 | 5 | 2 |
| 2026-02-11 | 7 | 5 | 2 |
| 2026-02-12 | 5 | 4 | 2 |
| 2026-02-13 | 9 | 6 | 2 |

The 1H registry is populated from day 2 and reaches its fluctuating band (3–10) by day
6; the 4H registry stays empty through all 5 warmup days and first populates on day 6,
reaching its band (1–6) around days 8–10. Key levels at day start are always the 2
prior-day levels (session levels form intraday).

## E10 — census cost

Total wall-clock **1,147.1 s (19.1 min)** for 20 days; per-day mean **57.3 s**
(min 17.4 s on the MLK holiday, max 87.0 s on 2026-01-30). The drain of the canonical
reader is ~98% of each day's cost; the 8-timeframe bar build is ~0.5–0.9 s/day.
Seed walk: 0.9 s.
