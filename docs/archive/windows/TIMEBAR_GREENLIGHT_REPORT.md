# TIMEBAR_GREENLIGHT_REPORT — land run 2026-07-28 — **STOPPED AT STEP 7c (SC CI red)**

Steps 1–6 completed clean. Step 7 CI witness: TL and QL **success**; **SC `ci` run
FAILURE** on the pushed tip — the four NEW time-bar tests red on the cold runner
(pandas 3.0.5) while the entire pre-existing suite stayed green. Per the work order
(Step 7c / Prohibition 4): STOPPED — no retry, no patch, failure reported below.

## Step 1 — state verification (all PASS)

- TL `7a911c0` clean, 0 ahead · QL `27cdfd6` clean, 0 ahead · SC 6 ahead of
  origin/platform-refactor, tracked-clean (untracked recon files only).
- `git log --oneline origin/platform-refactor..HEAD` (SC, newest first — order exact):

```
7992323 test(candles): wall-clock ceiling on the 8-TF time-bar build — 1.0s vs ~0.085s observed (TIMEBAR C6)
48fdf60 test(candles): time-bar rule units — anchoring, exact bucket edge, DST 23h/25h, bar_id kinds, TICK default, dense index (TIMEBAR C5)
20153a3 test(candles): time-bar batch<->streaming parity lock — 8 TFs, day roll, empty buckets, truncation, both DST days (TIMEBAR C4)
1eca72f feat(candles): vectorized build_time_bars_from_frame — one 60s aggregation, upward groupby per timeframe (TIMEBAR C3)
228ce28 feat(candles): streaming TimeBarEngine — 60s base + aggregate-upward, day-anchored DST-correct buckets (TIMEBAR C2)
fc0881c feat(candles): Bar gains kind (BarKind, appended last, default TICK); make_bar_id branches TIME->'s' (TIMEBAR C1)
```

- Diff stat: **8 files, +1003/−17**; `runtime/state.py`, `contract/schema.py`,
  `validation/_fixtures/` in NO hunk (name-only list verified):
  `candles/_buckets.py` (+39) · `candles/_ids.py` (+35/−) · `candles/time_batch.py`
  (+214) · `candles/time_streaming.py` (+274) · `strategies/protocols.py` (12 chg) ·
  `types.py` (+20) · `tests/test_time_bar_parity.py` (+222) · `tests/test_time_bars.py` (+204).

## Step 2 — window record + ruling

- Docs commit: **SC `8ec6906`** — `docs: record TIMEBAR window (SC time-bar construction) + ruling 9.11`
  (docs-only: PROGRESS +82, DECISIONS +15; does not move pins).
- DECISIONS id actually used: **9.11** (verified free before use; 9.4's original text untouched,
  partial discharge recorded inside the 9.11 entry).

## Steps 3/6 — pushed tips

- **SC** `origin/platform-refactor` = **`8ec6906`** (8ec69068fe8db49d9a51d2f4f6d57d3b003738b5)
- **TL** `origin/platform-refactor` = **`6c21b21`** (6c21b210e192ecae9f3e0b1f7484c873268a319b)
- **QL** `origin/platform-refactor` = **`9c2db53`** (9c2db53187250dedd108e5c2e3867358f8715087)

## Step 4 — ancestor assertions (exit code 0 = ancestor)

| sha | is-ancestor of origin/platform-refactor |
|---|---|
| fc0881c | PASS (0) |
| 228ce28 | PASS (0) |
| 1eca72f | PASS (0) |
| 20153a3 | PASS (0) |
| 48fdf60 | PASS (0) |
| 7992323 | PASS (0) |

`git log --oneline 8ec6906~7..8ec6906` re-printed: the six reviewed shas UNCHANGED, in
order, with only the Step-2 docs commit `8ec6906` above them. No amend, no rebase.

## Step 5 — pin bumps (target = 1eca72f, the latest consumer-facing commit; C4–C6 test-only, 8ec6906 docs-only)

TL `backend/pyproject.toml:18` — BEFORE:
```
  "strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@9d4935346bf42c5d19916e05dbe21dd67c46875c",
```
AFTER:
```
  "strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@1eca72f9d9c56a204e0e7c974ca8268ea85f36e5",
```

QL `pyproject.toml:36` — BEFORE:
```
    "strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@9d4935346bf42c5d19916e05dbe21dd67c46875c",
```
AFTER:
```
    "strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@1eca72f9d9c56a204e0e7c974ca8268ea85f36e5",
```

Pin chores: TL `6c21b21` · QL `9c2db53`. Why the pin moves (verified, not assumed):
`Bar` gained a defaulted field and `make_bar_id` a defaulted parameter — both
consumer-visible types. Verified neither consumer reads `Bar.kind` (TL's `.kind` hits
are its own DTO/catalog `definition.kind`/`level.kind`, not `Bar`; QL zero hits) nor
imports `time_streaming`/`time_batch`/`BarKind`; both suites ran green against the
editable SC containing these commits in the prior window (TL 511+1s, QL 830).

## Step 7 — CI witness

| repo | workflow | run id | run # | head sha | conclusion |
|---|---|---|---|---|---|
| Strategy-Core | ci | 30331205481 | 16 | 8ec6906 | **FAILURE** |
| Trade-Lab | backend-ci | 30331385000 | 17 | 6c21b21 | success |
| Quant-Lab | ci | 30331391624 | 15 | 9c2db53 | success |

### 7b — QL cold-install resolution of strategy-core (from the run log, job 90187121967)

```
Collecting strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@1eca72f9d9c56a204e0e7c974ca8268ea85f36e5 (from alpha-signal-lab==0.1.0)
  Resolved https://github.com/thealgochef/Strategy-Core.git to commit 1eca72f9d9c56a204e0e7c974ca8268ea85f36e5
```

### 7c — the SC failure (STOP trigger)

Job `strategy-core (cold install + ruff + pytest)` (90186545126): Install (cold) success,
Lint success, **Test (pytest — tests + validation) FAILURE**. Exactly the four NEW
time-bar tests fail; every pre-existing test (incl. both frozen digest gates) passed.

```
FAILED tests/test_time_bar_parity.py::test_parity_main_stream_all_timeframes - AssertionError: bar count differs: streaming=1023 batch=18
FAILED tests/test_time_bar_parity.py::test_parity_dst_spring_forward_23h_day - AssertionError: bar count differs: streaming=420 batch=12
FAILED tests/test_time_bar_parity.py::test_parity_dst_fall_back_25h_day - AssertionError: bar count differs: streaming=457 batch=9
FAILED tests/test_time_bars.py::test_trade_exactly_on_bucket_edge_opens_the_later_bar - AssertionError: assert ['60s:2025-06-03:0'] == ['60s:2025-06...2025-06-03:1']
```

Edge-test excerpt: two trades at +1 s and +60.000000 s both landed in batch bucket 0
(one bar emitted instead of two); the streaming engine emitted the correct two bars.

**Diagnosis (observation for the architect; NOT acted on):** the cold runner installed
**pandas 3.0.5** (host: pandas 2.x, where all these tests pass — the full local suite
was 219/219). Under pandas 3, `pd.to_datetime` over µs-precision Python datetimes
yields **microsecond-resolution** datetime64 (no longer ns-defaulted), so
`pd.DatetimeIndex(...).asi8` in `candles/time_batch.py` returns integer MICROSECONDS
while the bucket divisor is `interval * 1_000_000_000` (nanoseconds) — every bucket
index collapses by ×1000 (≈16.6 h into bucket 0), producing the tiny batch bar counts.
The streaming engine is pure-Python arithmetic and is unaffected; the parity harness
caught the drift exactly as designed. This is the third instance of the
pandas-3-on-cold-runner drift class (QL fixture types 2026-07-11; ruff 0.16 TOOLPIN
2026-07-25 as the toolchain sibling). Also relevant to the architect's carried
review finding (ii) (ns-vs-µs bucket arithmetic): the failure is resolution-unit
mismatch inside the batch path itself, adjacent to but distinct from the reader-side
truncation question.

**State at stop:** all three pushes stand (nothing was reverted — the work order
prescribes stop-and-report, not rollback); consumers are green on the new pin; SC's
gate is red on its own new tests only. No retry, no patch, no further steps executed.
