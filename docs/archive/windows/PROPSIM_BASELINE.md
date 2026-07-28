# PROP-SIM BASELINE — walker evidence run (P3; evidence, not a gate)

Date: 2026-07-10 · Preset: `topstep_50k` · Both fill columns · N=10,000 · seed 42 · max_days 1000
Walker: `alpha_lab.propsim` at QL `1ef0ebe` (P2 `dbf73d8` + the P4 verify fix), on P1 `86654c7`,
riding `f247046` (ENV-FIX).
Environment: shared system Python 3.13.1, strategy-core editable (not used by the walker — propsim is SC-free).

> Provenance: runs 1–2 below are the POST-VERIFY-FIX numbers (`1ef0ebe`). The bounded
> verify confirmed a MAJOR in the first-cut journal loader — warm-restart duplicate
> outcomes counted as independent trades (the WARM-PERF journal-dupe class; 2026-06-16's
> 36 outcomes are 8 physical touches) — fixed by touch-signature dedup, last-write-wins,
> drop count surfaced in the run notes. The pre-fix run saw 80 trades and P(pass) ≈ 0.993–0.997;
> the honest deduped pool is 49 trades. Runs 3–4 (OOS) are unaffected (no journal join).

Ruleset (ratified preset one): start 50,000 / target 3,000 / trail 2,000
(`eod_floor_realtime_breach`, locks at starting balance) / DLL 1,000 SOFT /
consistency 50% (best day ≤ pct × TOTAL profit) / no min-days / point_value 20.

---

## Data present on disk (stated exactly)

**TL executions** (`Trade-Lab/backend/data/executions`): 4 files, **93 rows — ALL
`type=reset`, zero `open`/`close` rows**. No paper fill has ever completed on disk:
every on-disk journal prediction is `is_eligible:false`, and the tracker only opens
eligible predictions. The spec's primary executions⋈journal leg therefore has **0 trades**.

**TL journal** (`Trade-Lab/backend/data/journal`): 14 day files, 163 rows = 83
predictions + 80 outcomes (2026-01-05 has 3 predictions, 0 outcomes). All 80 outcomes
carry `max_mfe_pts`/`max_mae_pts`/`entry_price`/`resolution_type` — but **31 of the 80
are warm-restart duplicates of the same physical touches** (fresh uuids per restart;
2026-06-16 alone: 36 outcomes = 8 touches in ×10/×9/×9/×4 groups). Deduped trades per
trading day (13 days, 49 trades): 2021-12-02: 4 · 2021-12-15: 4 · 2021-12-30: 1 ·
2022-01-10: 3 · 2022-03-04: 3 · 2026-02-10: 5 · 2026-02-11: 6 · 2026-06-15: 2 ·
2026-06-16: 8 · 2026-06-17: 4 · 2026-06-18: 3 · 2026-07-06: 3 · 2026-07-07: 3.
Both source bundles (`NQ_20260603_233847` replay, `NQ_W3_20260617T220752Z` live) stamp
`label_policy` tp 15.0 / sl 15.0, so the ±15pt points model is uniform across the pool.

**06-17 bundle OOS** (`models/NQ_W3_20260617T220752Z/oos_predictions.parquet`): 42 rows
over 15 trading days (2026-01-23 .. 2026-02-12), 13 columns — **pre-P1: no
`max_mfe_pts`/`max_mae_pts`/`entry_price`/`resolution_type`** (existing bundles are not
retrofitted, D-038). Gated subset (`gate_0_70_runtime_sessions`): 3 rows over 2 days.

---

## Run 1 — spec leg (a): executions ⋈ journal → 0 trades

```text
PROP-SIM walker — preset topstep_50k (source: executions)
  ruleset: start 50000 / target 3000 / trail 2000 (eod_floor_realtime_breach, locks_at_start=True) / DLL 1000.0 (soft=True) / consistency 50.0% / min_days None / point_value 20.0
  pool: 0 trades over 0 days (None .. None); excursions on 0/0 trades
  monte carlo: N=10000 seed=42 max_days=1000

  column       breach mode               hist        P(pass)          95% CI  P(bust)  P(inc)  d2p med    p10    p90  d2b med  notes
  ----------------------------------------------------------------------------------------------------------------------------------
  optimistic   realized_only             -                 -               -        -       -        -      -      -        -
  optimistic   unrealized_adverse_first  -                 -               -        -       -        -      -      -        -  DEGRADED->realized
  conservative realized_only             -                 -               -        -       -        -      -      -        -
  conservative unrealized_adverse_first  -                 -               -        -       -        -      -      -        -  DEGRADED->realized

  notes:
    - executions dir: C:\Users\gonza\Documents\Trade-Lab\backend\data\executions
    - rows: 93 total, 0 closes, 0 opens, 93 resets
    - replay re-run duplicate closes dropped (fill-signature dedup, last-write-wins): 0
    - opens without a close (skipped — never completed): 0
    - closes skipped as unparseable (missing/non-finite fields): 0
    - journal join: 0/0 trades carry MFE/MAE from journal outcomes
```

## Run 2 — journal EVIDENCE MODE (deviation, clearly labeled): 80 outcomes as 1-lot trades

Because leg (a) is empty, this run exercises the walker on the only real TL trade-shaped
data on disk: **every resolved journal outcome treated as a 1-contract trade** (after
touch-signature dedup: 80 outcomes → 49 physical trades, 31 warm-restart duplicates
dropped and surfaced in the notes). All 49 belong to serving-INELIGIBLE predictions —
the model would NOT have taken these trades; this is level-touch outcome data, not
model-gated trading. It is walker evidence, not an edge claim.

```text
PROP-SIM walker — preset topstep_50k (source: journal)
  ruleset: start 50000 / target 3000 / trail 2000 (eod_floor_realtime_breach, locks_at_start=True) / DLL 1000.0 (soft=True) / consistency 50.0% / min_days None / point_value 20.0
  pool: 49 trades over 13 days (2021-12-02 .. 2026-07-07); excursions on 49/49 trades
  monte carlo: N=10000 seed=42 max_days=1000
  [optimistic] win rate 30/49 = 0.612 (binomial 95% CI 0.472..0.736); mean 3.37 pts, total 165.00 pts
  [conservative] win rate 30/49 = 0.612 (binomial 95% CI 0.472..0.736); mean 3.02 pts, total 148.00 pts

  column       breach mode               hist        P(pass)          95% CI  P(bust)  P(inc)  d2p med    p10    p90  d2b med  notes
  ----------------------------------------------------------------------------------------------------------------------------------
  optimistic   realized_only             pass         0.9900    0.988..0.992   0.0100   0.000     11.0    6.0   20.0     12.5
  optimistic   unrealized_adverse_first  pass         0.9889    0.987..0.991   0.0111   0.000     11.0    6.0   20.0     12.0
  conservative realized_only             incomplete   0.9813    0.978..0.984   0.0187   0.000     13.0    7.0   23.0     13.0
  conservative unrealized_adverse_first  incomplete   0.9730    0.970..0.976   0.0270   0.000     13.0    7.0   23.0     12.0

  notes:
    - journal dir: C:\Users\gonza\Documents\Trade-Lab\backend\data\journal
    - EVIDENCE MODE: every resolved journal outcome treated as a 1-contract trade — includes outcomes of serving-INELIGIBLE predictions
    - warm-restart duplicate outcomes dropped (touch-signature dedup, last-write-wins): 31
    - outcomes: 80 rows -> 49 trades (0 skipped: unresolved/unparseable)
    - eligible predictions among trades: 0; outcomes without a prediction row (dated by resolved ts): 0
    - points model: tp_hit -> +15.0, sl_hit -> -15.0; conservative mirrors the tracker fill model (entry 1 tick adverse, SL exit 1 tick worse, tick=0.25)
```

As-sequenced historical detail: the optimistic walk PASSES on day 13 at $53,300
(best-day ratio 0.273, consistency satisfied); the conservative walk ends **incomplete
at $52,960 — $40 short of the $3,000 target** after the same 13 days: 49 trades × the
tick-adverse fill model costs $340, exactly the pass/no-pass margin here. Fill
assumptions, not the trailing floor, decide this sequence. Unrealized-adverse-first
tightens `min_floor_distance` (optimistic 1,100 → 1,035; conservative 1,070 → 1,005)
and raises MC bust counts (100→111 and 187→270 per 10,000) without changing any
realized settle — exactly the intended mode separation. All MC busts are
`trailing_floor`; zero DLL busts (soft DLL halts days instead).

## Run 3 — spec leg (b): 06-17 bundle OOS parquet, ungated (realized-only, stated)

```text
PROP-SIM walker — preset topstep_50k (source: oos)
  ruleset: start 50000 / target 3000 / trail 2000 (eod_floor_realtime_breach, locks_at_start=True) / DLL 1000.0 (soft=True) / consistency 50.0% / min_days None / point_value 20.0
  pool: 42 trades over 15 days (2026-01-23 .. 2026-02-12); excursions on 0/42 trades
  monte carlo: N=10000 seed=42 max_days=1000
  [optimistic] win rate 17/42 = 0.405 (binomial 95% CI 0.270..0.555); mean -2.86 pts, total -120.00 pts
  [conservative] win rate 17/42 = 0.405 (binomial 95% CI 0.270..0.555); mean -2.86 pts, total -120.00 pts

  column       breach mode               hist        P(pass)          95% CI  P(bust)  P(inc)  d2p med    p10    p90  d2b med  notes
  ----------------------------------------------------------------------------------------------------------------------------------
  optimistic   realized_only             bust         0.0016    0.001..0.003   0.9984   0.000     22.0   12.5   35.5      9.0
  optimistic   unrealized_adverse_first  bust         0.0016    0.001..0.003   0.9984   0.000     22.0   12.5   35.5      9.0  DEGRADED->realized
  conservative realized_only             bust         0.0016    0.001..0.003   0.9984   0.000     22.0   12.5   35.5      9.0
  conservative unrealized_adverse_first  bust         0.0016    0.001..0.003   0.9984   0.000     22.0   12.5   35.5      9.0  DEGRADED->realized

  notes:
    - OOS parquet: models\NQ_W3_20260617T220752Z\oos_predictions.parquet
    - rows: 42 total; gate: none (all labeled rows are trades); trades: 42 (0 skipped: unmapped label/resolution)
    - points model: idealized barrier fills, tp_hit -> +15.0, sl_hit -> -15.0; conservative column EQUALS optimistic (no synthetic slippage model)
    - pre-P1 OOS parquet: max_mfe_pts/max_mae_pts absent or empty — unrealized_adverse_first degrades to realized-only
```

As-sequenced: **bust on day 6** (`trailing_floor`, floor breached by $100, final
$47,900). MC: P(bust) 0.9984 in every cell. Consistent with the W3 bundle's known
NEGATIVE honest OOS edge (ROC-AUC 0.39) — the walker turns that into an evaluation
verdict: taking every OOS touch at 1 lot busts a TopStep 50K with near-certainty.

## Run 4 — OOS gated (`gate_0_70_runtime_sessions`): the model's actual trades

```text
PROP-SIM walker — preset topstep_50k (source: oos)
  ruleset: start 50000 / target 3000 / trail 2000 (eod_floor_realtime_breach, locks_at_start=True) / DLL 1000.0 (soft=True) / consistency 50.0% / min_days None / point_value 20.0
  pool: 3 trades over 2 days (2026-01-27 .. 2026-01-28); excursions on 0/3 trades
  monte carlo: N=10000 seed=42 max_days=1000
  [optimistic] win rate 1/3 = 0.333 (binomial 95% CI 0.061..0.792); mean -5.00 pts, total -15.00 pts
  [conservative] win rate 1/3 = 0.333 (binomial 95% CI 0.061..0.792); mean -5.00 pts, total -15.00 pts

  column       breach mode               hist        P(pass)          95% CI  P(bust)  P(inc)  d2p med    p10    p90  d2b med  notes
  ----------------------------------------------------------------------------------------------------------------------------------
  optimistic   realized_only             incomplete   0.0000    0.000..0.000   1.0000   0.000        -      -      -     12.0
  optimistic   unrealized_adverse_first  incomplete   0.0000    0.000..0.000   1.0000   0.000        -      -      -     12.0  DEGRADED->realized
  conservative realized_only             incomplete   0.0000    0.000..0.000   1.0000   0.000        -      -      -     12.0
  conservative unrealized_adverse_first  incomplete   0.0000    0.000..0.000   1.0000   0.000        -      -      -     12.0  DEGRADED->realized

  notes:
    - OOS parquet: models\NQ_W3_20260617T220752Z\oos_predictions.parquet
    - rows: 42 total; gate: gate_0_70_runtime_sessions; trades: 3 (0 skipped: unmapped label/resolution)
    - points model: idealized barrier fills, tp_hit -> +15.0, sl_hit -> -15.0; conservative column EQUALS optimistic (no synthetic slippage model)
    - pre-P1 OOS parquet: max_mfe_pts/max_mae_pts absent or empty — unrealized_adverse_first degrades to realized-only
```

A 3-trade / 2-day pool is a nearly vacuous simulation input: every bootstrap run
resamples the same net-negative days until the floor (P(bust)=1.0); the as-sequenced
walk simply runs out of data at −$300.

---

## Fidelity caveats (verbatim, as ratified)

1. **MFE/MAE order unknown → adverse-first is conservative.** The journal/OOS rows carry
   excursion magnitudes but not their ordering within the trade; `unrealized_adverse_first`
   assumes each trade's path visits entry − MAE before entry + MFE, which over-counts
   intraday breach risk relative to any path where the favorable excursion came first.
2. **OOS lacks excursions pre-P1.** The 06-17 bundle predates P1 (`86654c7`) and is not
   retrofitted (D-038): its `unrealized_adverse_first` cells are DEGRADED to
   realized-only, with the reason stated in the run output. Excursion-aware OOS
   simulation requires a post-P1 fresh save.
3. **42 OOS rows = huge CI, report the binomial interval.** The ungated OOS win rate is
   17/42 = 0.405 with **binomial 95% CI 0.270..0.555** (gated: 1/3 = 0.333, CI
   0.061..0.792). The Monte Carlo CI on P(pass) (Wilson over N=10,000 runs) measures
   sampling of the SAME 15 days only — data uncertainty dominates and the day pool is
   tiny (15 distinct blocks; 13 in the journal run).

Additional honesty notes (this run):

4. Run 2 is an evidence-mode deviation: the spec's executions⋈journal leg has zero
   completed fills on disk (all 93 execution rows are resets), so the journal-outcomes
   run stands in as real-data walker evidence. Its 49 deduped trades are ALL from
   serving-ineligible predictions and mix replay days (2021-12 .. 2022-03) with live days
   (2026-02 .. 2026-07); the day-level bootstrap treats those as exchangeable draws.
   The verify pass proved the dedup necessary (PROPSIM-L1 major): without it the same
   physical touch counts up to ×10 — a 1-contract account cannot fill one touch ten times.
5. OOS conservative column EQUALS optimistic (idealized barrier fills; no synthetic
   slippage model is invented) — the two column blocks in runs 3/4 are identical by
   construction. The journal run's conservative column mirrors the TL tracker's fill
   model (entry 1 tick adverse; SL exit 1 tick worse).
6. Costs (commissions/fees) are NOT modeled anywhere in the walker; all runs are
   1-contract with point_value 20.

## Reproduction

```bash
cd C:/Users/gonza/Documents/Claude-Quant-Lab
python -m alpha_lab.propsim --source executions C:/Users/gonza/Documents/Trade-Lab/backend/data/executions \
  --journal C:/Users/gonza/Documents/Trade-Lab/backend/data/journal --preset topstep_50k --column both --n 10000 --seed 42
python -m alpha_lab.propsim --source journal C:/Users/gonza/Documents/Trade-Lab/backend/data/journal \
  --tp-points 15 --sl-points 15 --preset topstep_50k --column both --n 10000 --seed 42
python -m alpha_lab.propsim --oos models/NQ_W3_20260617T220752Z/oos_predictions.parquet \
  --preset topstep_50k --column both --n 10000 --seed 42
python -m alpha_lab.propsim --oos models/NQ_W3_20260617T220752Z/oos_predictions.parquet \
  --gate-column gate_0_70_runtime_sessions --preset topstep_50k --column both --n 10000 --seed 42
```

---

# APPENDIX — PRESETS window: cross-preset comparison (the first firm-vs-firm run on identical paths)

Date: 2026-07-10 · ALL FOUR presets · Both fill columns · N=10,000 · seed 42 · max_days 1000
Walker: `alpha_lab.propsim` at QL `49c2f22` (PRESETS P1 `3c4417e` trail mechanics + P2 `49c2f22`
presets), atop the PROP-SIM tree (`1ef0ebe`). Same two pools as runs 2–3 above, re-walked under
every preset: the **journal evidence pool** (49 deduped trades / 13 days, win 30/49 = 0.612,
excursions on 49/49) and the **06-17 OOS ungated** (42 trades / 15 days, win 17/42 = 0.405,
excursions on 0/42 — pre-P1 parquet: `unrealized_adverse_first` DEGRADED to realized-only in
every preset; the conservative column EQUALS optimistic there by construction).

**Regression witness:** `topstep_50k` under the PRESETS engine reproduces the PROP-SIM baseline
EXACTLY — journal P(pass) 0.9900 / 0.9889 / 0.9813 / 0.9730 and OOS P(bust) 0.9984 with the
same as-sequenced verdicts (pass day 13 at $53,300 / conservative incomplete at $52,960; OOS
bust day 6 at $47,900). The `dll_soft`→`dll_hard` rename is presentation-only (`hard=False` ≡
`soft=True`), and the new `P(exp)` column is 0.000 for every no-expiry ruleset.

## Rulesets as walked (P2 data; ⚠ = verify-at-dashboard, see preset docstring)

| preset | trail style | locks | DLL | consistency | min_days | max_eval_days |
|---|---|---|---|---|---|---|
| `topstep_50k` | eod_floor_realtime_breach | yes | 1,000 soft | 50% | — | — |
| `apex_50k_eod` | eod_floor_realtime_breach | yes ⚠ | 1,000 HARD ⚠ | — | — | 30 |
| `apex_50k_intraday` | intraday_peak_trail | yes ⚠ | — | — | — | 30 |
| `tpt_50k_test` | eod_floor_realtime_breach | yes | — | 50% | 5 | — |

All: start 50,000 / target 3,000 / trail 2,000 / point_value 20 / 1 contract.

## Journal evidence pool (49 trades / 13 days) — cross-preset

Historical (as-sequenced) is IDENTICAL across all four presets per column: optimistic PASSES
day 13 at $53,300; conservative ends INCOMPLETE at $52,960 ($40 short — the tick-adverse fill
model still decides this sequence, not the ruleset). The Monte Carlo separates them:

| preset | column | breach mode | hist | P(pass) | 95% CI | P(bust) | P(exp) | d2p med | d2b med |
|---|---|---|---|---|---|---|---|---|---|
| topstep_50k | optimistic | realized_only | pass | 0.9900 | 0.988..0.992 | 0.0100 | 0.000 | 11.0 | 12.5 |
| topstep_50k | optimistic | unrealized_adverse_first | pass | 0.9889 | 0.987..0.991 | 0.0111 | 0.000 | 11.0 | 12.0 |
| topstep_50k | conservative | realized_only | incomplete | 0.9813 | 0.978..0.984 | 0.0187 | 0.000 | 13.0 | 13.0 |
| topstep_50k | conservative | unrealized_adverse_first | incomplete | 0.9730 | 0.970..0.976 | 0.0270 | 0.000 | 13.0 | 12.0 |
| apex_50k_eod | optimistic | realized_only | pass | 0.9808 | 0.978..0.983 | 0.0101 | 0.009 | 11.0 | 12.0 |
| apex_50k_eod | optimistic | unrealized_adverse_first | pass | 0.9798 | 0.977..0.982 | 0.0112 | 0.009 | 11.0 | 12.0 |
| apex_50k_eod | conservative | realized_only | incomplete | 0.9572 | 0.953..0.961 | 0.0170 | 0.026 | 13.0 | 13.0 |
| apex_50k_eod | conservative | unrealized_adverse_first | incomplete | 0.9513 | 0.947..0.955 | 0.0258 | 0.023 | 13.0 | 12.0 |
| apex_50k_intraday | optimistic | realized_only | pass | 0.9720 | 0.969..0.975 | 0.0201 | 0.008 | 11.0 | 11.0 |
| apex_50k_intraday | optimistic | unrealized_adverse_first | pass | 0.9492 | 0.945..0.953 | 0.0455 | 0.005 | 11.0 | 9.0 |
| apex_50k_intraday | conservative | realized_only | incomplete | 0.9470 | 0.942..0.951 | 0.0311 | 0.022 | 12.0 | 12.0 |
| apex_50k_intraday | conservative | unrealized_adverse_first | incomplete | 0.9076 | 0.902..0.913 | 0.0776 | 0.015 | 12.0 | 9.0 |
| tpt_50k_test | optimistic | realized_only | pass | 0.9901 | 0.988..0.992 | 0.0099 | 0.000 | 11.0 | 12.0 |
| tpt_50k_test | optimistic | unrealized_adverse_first | pass | 0.9890 | 0.987..0.991 | 0.0110 | 0.000 | 11.0 | 12.0 |
| tpt_50k_test | conservative | realized_only | incomplete | 0.9811 | 0.978..0.984 | 0.0189 | 0.000 | 13.0 | 13.0 |
| tpt_50k_test | conservative | unrealized_adverse_first | incomplete | 0.9727 | 0.969..0.976 | 0.0273 | 0.000 | 13.0 | 12.0 |

## 06-17 OOS ungated (42 trades / 15 days; excursion-DEGRADED everywhere) — cross-preset

Conservative EQUALS optimistic and unrealized DEGRADES to realized in every cell (pre-P1
parquet); one row per preset states all four cells:

| preset | hist | P(pass) | 95% CI | P(bust) | P(exp) | d2p med | d2b med |
|---|---|---|---|---|---|---|---|
| topstep_50k | bust day 6 @ $47,900 | 0.0016 | 0.001..0.003 | 0.9984 | 0.000 | 22.0 | 9.0 |
| apex_50k_eod | bust day 6 @ $47,900 | 0.0014 | 0.001..0.002 | 0.9821 | 0.017 | 16.0 | 9.0 |
| apex_50k_intraday | bust day 5 @ $48,500 | 0.0011 | 0.001..0.002 | 0.9882 | 0.011 | 16.0 | 9.0 |
| tpt_50k_test | bust day 6 @ $47,900 | 0.0016 | 0.001..0.003 | 0.9984 | 0.000 | 22.0 | 9.0 |

## Firm-vs-firm reading (identical paths, only the ruleset varies)

1. **The intraday peak trail is the binding difference on the journal pool.** Optimistic
   unrealized P(pass) drops 0.9889 (TopStep EOD floor) → 0.9492 (Apex intraday trail);
   conservative unrealized 0.9730 → 0.9076. Bust counts scale ×4–×3 (111 → 455 and 270 → 776
   per 10,000): each trade's +MFE leg ratchets the floor in real time, so the SAME excursions
   that TopStep's EOD floor never sees become breaches. Historical `min_floor_distance`
   tightens 1,035 → 975 (optimistic) and 1,005 → 940 (conservative). Even realized-only the
   intraday style is tighter (ratchets on every close, not just EOD): 0.9900 → 0.9720.
2. **The Apex 30-day budget converts slow finishes into expiries.** Journal P(exp) 0.5–2.6%
   (largest on the conservative column, whose passes are slowest); OOS P(exp) 1.1–1.7% —
   there it re-labels slow busts, and the pass-day median truncates 22 → 16 (only fast passes
   fit inside the budget). Expiry is a verdict, not a bust: on a re-tryable eval it reads as
   "pay another entry fee".
3. **The hard DLL never fired on either pool.** `apex_50k_eod` bust reasons are 100%
   `trailing_floor` (as are every other preset's): a ±15pt/trade, ≤8-trades/day pool cannot
   reach −$1,000 in a day before the (nearer) floor. The ⚠ soft-vs-hard question is
   numerically MOOT on these pools — it will bind on wider-stop or higher-frequency pools.
4. **`tpt_50k_test` ≈ `topstep_50k` on these pools.** Its min_days 5 never binds (median pass
   day 11–13) and dropping the DLL barely moves the MC (0.9900 → 0.9901 optimistic realized —
   TopStep's soft-DLL halts almost never trigger here). The differences would surface on
   pools with heavy single-day loss clusters.
5. **Ranking (this data, both pools):** TopStep 50K ≈ TPT 50K-test (easiest) > Apex 50K-EOD >
   Apex 50K-intraday (hardest). On the negative-edge OOS pool every ruleset busts with
   near-certainty (P(bust) ≥ 0.982) — no ruleset launders a losing strategy.

Honesty notes (this appendix): `apex_50k_intraday`'s headline mechanic (the UNREALIZED peak
trail) is exercised only on the journal pool — the OOS parquet is pre-P1 and excursion-free,
so its OOS delta vs `apex_50k_eod` comes solely from close-to-close intraday ratcheting.
The two ⚠ Apex parameters (lock-at-start, DLL hard-vs-soft) are recorded as checked out on
2026-07-10 and are pending dashboard verification; flipping either is a data-only preset
change. All PROP-SIM fidelity caveats (MFE/MAE ordering, no costs, evidence-mode
ineligibility, tiny day pools) carry over unchanged.

## Reproduction (appendix)

```bash
cd C:/Users/gonza/Documents/Claude-Quant-Lab
for preset in topstep_50k apex_50k_eod apex_50k_intraday tpt_50k_test; do
  python -m alpha_lab.propsim --source journal C:/Users/gonza/Documents/Trade-Lab/backend/data/journal \
    --tp-points 15 --sl-points 15 --preset $preset --column both --n 10000 --seed 42
  python -m alpha_lab.propsim --oos models/NQ_W3_20260617T220752Z/oos_predictions.parquet \
    --preset $preset --column both --n 10000 --seed 42
done
```
