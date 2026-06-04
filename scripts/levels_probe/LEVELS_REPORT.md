# Why RTH touches are scarce and revert less — investigation (READ-ONLY)

Model `NQ_20260602_232808`, engine v2, `dataset_config_hash 3d2f8466`. All numbers re-derived from
the **production** engine functions (`_build_bars_for_date` / `_compute_levels_for_date` /
`build_zones` / `is_touch` / `detect_touches`) over the 225 `dates_used`; the probe's recorded
touches (first straddle per zone) reproduce `detect_touches` and match the audit enrichment's
detected-touch counts exactly. Scratch: `Claude-Quant-Lab/scripts/levels_probe/`. No fixes, no
engine/contract/model changes.

---

## TL;DR — the three answers

1. **The scarce RTH count is `first_touch_per_zone_per_day` working as written, *compounded by a
   look-ahead defect*: the contract's `available_from_guard` is emitted but never enforced.** Session
   high/low levels are eligible for "touch" during their own forming session (even the prior evening),
   so they are consumed overnight before RTH. **Of 617 zone-days where price actually swept a level in
   RTH, only 64 are recorded as RTH touches; 553 (89.6%) were already first-touched overnight and
   discarded.** The model is structurally blind to ~90% of RTH level-sweeps.

2. **The 0.70-vs-0.52 reversal gap is SELECTION, not regime.** The overnight rate is carried by
   `london_high/low` touches that revert **0.81–0.87** — but those "touches" are the session's own
   extreme bar (a tautological reversal point) recorded *before the level is knowable*. Controlling for
   volatility does **not** close the gap (low-vol bucket: RTH 0.488 vs overnight 0.747). RTH touches are
   almost all PDH/PDL, which revert **0.50–0.53** — the honest, coin-flip rate.

3. **Most consequential:** the model's apparent edge is a **look-ahead labeling artifact**. **273 of 384
   training labels (71%) are levels touched before they were knowable.** The signal it learned
   (overnight session-extreme reversion) does not exist in the RTH population it deploys on. **The
   strategy is not salvageable in RTH as currently constructed** — not without redefining what a
   "touch" is and enforcing level availability.

---

## PART 1 — the code (every claim quoted)

**1a. Level construction** — `dashboard_utility_builder.py:_compute_levels_for_date` (397-426):
- **PDH/PDL** = the **prior day's NY-RTH** high/low, carried in `prev_ny_hl` and emitted at lines
  408-410 (`{"name":"PDH","price":prev_ny_hl[0],"side":"HIGH"}` / PDL). `prev_ny_hl` is set from
  `_session_hl(ny)` over the `ny_rth` slice (115-118, `_session_hl` 378-379) — matches
  `pdh_pdl_source = prior_day_ny_rth`.
- **asia_high/low, london_high/low** = **this day's** session slices: `_slice_session(bars_et,"asia"|"london")`
  (413-424), max-high/min-low via `_session_hl`.
- **Timezone:** bars are ET-indexed by `_ensure_et_index` (364-372, `tz_convert("US/Eastern")`), and
  `_slice_session` masks on `bars.index.time` (382-394) against ET constants `_ASIA_START=time(18,0)`
  … `_NY_RTH_END=time(16,15)` (44-51). **Confirmed ET**, boundaries asia 18:00–01:00, london
  01:00–08:00, ny_rth 09:30–16:15.

**1b. `available_from_guard` is DECLARED but NOT ENFORCED.** `strategy_contract.py:153` emits
`"available_from_guard": k.LEVEL_AVAILABLE_FROM_GUARD` (`constants.py:210 = True`) — a *descriptor*.
But neither `_compute_levels_for_date` nor `detect_touches` (`touch.py:49-95`) carries an availability
timestamp per level or gates the scan by it. `detect_touches` iterates **the whole Globex day's bars**
and fires the first straddle. **Consequence (re-derived): 713 of 910 recorded touches (78%) — and 273
of 384 kept training touches (71%) — are LOOK-AHEAD** (first straddle before the defining session
closed: asia avail 01:00, london 08:00). A level can be "touched" the evening before it exists (see
trace L1). **This is a look-ahead bug, named: the available-from guard is unenforced.**

**1c. Zone formation** — `strategy_core/decisions/zones.py:build_zones` (23-97), byte-port of
`_build_zones`: price-sort levels, chained merge when `lvl.price - groups[-1][-1].price <= 3.0`
(`ZONE_PROXIMITY_PTS`), `representative_price = mean(prices)`, strict-majority side (ties→LOW).
**Re-derived: merging is rare — 35 of 1071 zones have ≥2 levels (3.3%)**; the six NQ levels are
usually >3 pt apart, so ~6 standalone zones/day.

**1d. Touch detection + dedup** — `touch.py:detect_touches` (49-95): for each bar in order, each
**un-touched** zone whose `[low,high]` straddles `rep` (`is_touch`, `bar_low<=rep<=bar_high`, line 46/82)
fires once and sets `zone.touched=True` (83); guard at line 79 stops re-firing. **"Per day"** = the
**18:00-ET Globex trading day** — `_build_bars_for_date` bounds the bars to `[prev 18:00 ET, cur 18:00
ET)` (`builder:319-320`), and `build_zones` is rebuilt fresh per date, so the touched-flag resets each
trading day (not the calendar day, not the RTH day).

---

## PART 2 — mechanics

**A. Touch counts (re-derived).** Detected (first-straddle) = **910**; kept (survive flatten/feature
drops) = **384**.

| session | detected | kept | note |
|---|---|---|---|
| asia (18:00–01:00) | 600 | 79 | most flatten-dropped (evening decision ≥15:55) |
| london (01:00–08:00) | 227 | 227 | none flatten (decision 01:05–08:05) |
| premarket (08:00–09:30) | 17 | 17 | the "~17 not in 3 named sessions" |
| **ny_rth (09:30–16:15)** | **64** | **61** | 3 late-RTH flatten-drops |
| post (16:15–18:00) | 2 | 0 | flatten |

`level_type × session` (kept 384): `london_high/low` → 188 in london; `asia_high/low` → 69 in asia;
**PDH/PDL → 59 of the 61 RTH touches.** Session highs/lows essentially never reach RTH.

**B. Per-day zone survival (re-derived, 185 zone-days).** zones formed mean **5.79** (median 6; hist
4:4d, 5:31d, 6:150d) · zones touched mean 4.92 · **zones first-touched in RTH mean 0.35 — and ZERO on
124 of 185 days (67%).** On two-thirds of trading days the model gets no RTH setup at all.

**C. THE KEY NUMBER (re-derived from all bar-zone straddles, not just the deduped first).**

| quantity | value |
|---|---|
| zones (zone-days) total | 1071 |
| zones price actually swept during RTH (≥1 RTH straddle) | **617** |
| …first-touched IN RTH (recorded RTH touches) | **64** |
| …first-touched OVERNIGHT, RTH sweep **discarded** by dedup | **553** |
| **% of RTH level-sweeps invisible to the model** | **89.6%** |
| recorded touches that are look-ahead (before available_from) | **713 / 910 (78%)** |
| kept *training* touches that are look-ahead | **273 / 384 (71%)** |

**D. Traced examples (real zones).**
- **L1 — look-ahead + consumption.** 2025-06-02 `london_high` rep 21291.50 (SHORT), available 08:00 ET.
  **First straddle 2025-06-01 20:30:41 ET (asia, the night before)** — `lookahead=True`. Swept **25×**
  during RTH (first RTH straddle 09:30:00), all discarded. The recorded "touch" is 12 h before the
  london session that defines the level even closes.
- **L2 — the honest case.** 2025-06-05 `PDL` rep 21641.00 (LONG), available from day start. First
  straddle **14:16:51 ET (RTH)**, `lookahead=False`. A genuine prior-day level first reached in RTH —
  one of the 64.
- **L3 — consumed overnight.** 2025-06-03 zone `asia_high|PDH` rep 21556.62. First straddle 09:23:45
  (premarket), then swept 16× in RTH — RTH sweeps discarded.
- **L4 = L1's RTH view** — `has_rth_straddle=True`, `n_rth_straddles=25`, but the recorded touch is the
  20:30 prior-evening asia straddle.

---

## PART 3 — why overnight reverts ~0.70 and RTH ~0.52 (selection vs regime)

**Rebound (label_encoded==0 = tradeable_reversal) by level_type (kept):**

| level_type | rebound | n |
|---|---|---|
| london_high | **0.867** | 105 |
| london_low | **0.811** | 95 |
| asia_low | 0.700 | 30 |
| PDL | 0.526 | 57 |
| asia_high | 0.512 | 43 |
| PDH | 0.500 | 54 |

The "70%" is the **london session-extreme self-touches** (0.81–0.87) — touches that *are* the session
high/low (a local extreme reverses by construction) and are recorded before the level is knowable.
PDH/PDL — the only levels that reach RTH — revert at **0.50–0.53**.

**E3 controlled for volatility (atr14 terciles) — the gap is NOT regime:**

| group | low-vol | mid-vol | high-vol |
|---|---|---|---|
| RTH | **0.488** (41) | 0.786 (14) | 0.167 (6) |
| overnight | **0.747** (87) | 0.789 (114) | 0.664 (122) |

In the low-vol bucket RTH 0.488 vs overnight 0.747 — **the gap survives**, so volatility regime does
not explain it. It is **selection**: the overnight population is dominated by look-ahead session-extreme
touches.

**E1 travel / E2 vol / E4 volume (RTH vs overnight medians):**
- **E4 volume — premise confirmed:** RTH `vol_touch` 253 vs 207, `vol_mean30` 227 vs 203 — RTH is more
  liquid; overnight thinner.
- **E2:** per-147t-bar range `atr14` RTH 6.0 < overnight 8.14 — but that is a **tick-bar-clock artifact**
  (overnight bars span more wall-clock per 147 trades → wider range), not higher true RTH vol; `ret_std30`
  same direction (4.0 vs 5.26).
- **E1:** overnight touches sit on a larger recent 30-bar range (50.25 vs 37.75, the session-extreme
  context); RTH touches sit far from the RTH open (`dist_rth_open` 130 vs 49, continuation context).

---

## Verdicts

1. **Count:** `first_touch_per_zone_per_day` is working *as written* — but the scarcity is *caused* by
   the **unenforced `available_from` guard** (look-ahead bug): session high/low levels are first-touched
   overnight at/before their own formation, consuming the zone before RTH. Net effect = the recorded
   touch is the overnight first-contact, not the RTH sweep, for **553 of 617 (89.6%)** RTH-swept zones.
   Not a tz/merge/dedup arithmetic bug; a **level-availability** bug.
2. **Reversal gap:** **selection (look-ahead), not regime.** london session-extreme self-touches revert
   tautologically (0.84) and inflate the overnight rate; controlling for vol does not close the gap
   (RTH 0.49 vs overnight 0.75 at low vol); RTH PDH/PDL revert at honest ~0.52.
3. **Most consequential:** 71% of training labels are look-ahead artifacts; the learned overnight-reversion
   signal is absent in the RTH deployment population (PDH/PDL ≈ coin flip), and dedup hides ~90% of RTH
   sweeps. **The strategy is not salvageable in RTH as built** — it needs (at minimum) an enforced level-
   availability guard and a touch definition that doesn't equate a level with its own forming bar, then a
   full retrain and re-audit. *(Recommendations only — no changes made.)*

**UNVERIFIED / caveats:** rebound rates are computed on *kept* touches (those that survived flatten +
the <5-tick feature drop and carry a label); the discarded 553 RTH sweeps have no label, so their
*reversion* rate is UNVERIFIED — only their existence/count is established. Whether enforcing the guard
would convert consumed RTH sweeps into tradeable first-touches is a forward hypothesis, not measured here.

---
### Independent verification
Every finding above was independently re-derived by a **6-agent adversarial-verification workflow**
(each agent instructed to *refute* by recomputing from the primary parquet/code, not trusting the
summary): **6/6 CONFIRM, 0 REFUTE.** Notable independent reproductions: `build_zones` 23-97 /
`detect_touches` 49-95 / `is_touch` line 46 and its only call at touch.py:82 (no `available_from`
reference anywhere in `strategy_core/decisions/`); look-ahead 713/910 (asia 519 + london 194) and
273/384 kept, 0/913 mismatches vs the probe flag; the 617 / 64 / 553 partition is exact and exhaustive
(64+553+0post+0untouched = 617); rebound-by-level-type and the session×vol-tercile table reproduced to
the digit.

Two immaterial caveats from the verifiers: (i) the raw probe parquet holds 913 touches / 715 look-ahead;
**scoped to the model's 225 `dates_used` it is 910 / 713** (the scope used throughout this report — 3
extra asia touches fall on out-of-range dates). (ii) A naive `(date, event_ts)` join collides on 2 of
384 rows (a PDH and a london_high sharing one bar-close timestamp); joining additionally on `level_type`
resolves it and leaves every PART-3 number unchanged. Neither affects any verdict.
