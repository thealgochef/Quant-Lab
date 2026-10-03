# DECISIONS — living registry

## THE GOAL (the drift anchor — every window serves this)
Trade-Lab runs replay and live on a model trained in Quant-Lab, with a standing gate
proving both rode the SAME Strategy-Core data-ingestion and strategy surfaces —
batch-drive ≡ stream-drive per touch and per feature on identical canonical input,
proven on a FRESH model, then soaked ≥7 trading days before any number is trusted.

## Rules of this file
- One entry per ruling: ID · STATUS · ruling · why · where enforced.
- Append-only. Supersession = new entry + status flip on the old one; originals never deleted.
- PROGRESS cites IDs and owns execution state; THIS file owns ruling text and living status.
- Plan §9 entries are summarized here; PLATFORM_REFACTOR_PLAN.md §9 holds their full text.
- Quant-Lab's docs/DECISIONS.md is repo-local research lore; cross-repo rulings live HERE.

## Plan-era rulings (full text in PLAN §9; status here is living)
- 9.1 RATIFIED — Plugin Protocol + registry-time assertion.
- 9.2 RATIFIED — In-package strategy registry.
- 9.3 RATIFIED — Two-axis versioning.
- 9.4 PARKED — Time bars are Phase F.
- 9.5 RATIFIED — Keep strategy_core name + strategies/<id>/ layout.
- 9.6 RATIFIED — SHA-pin SC in QL (declared-not-enforced; QL cold-install debt named).
- 9.7 RATIFIED — TL local contract deleted.
- 9.8 RATIFIED — Barrier protocol; TL outcome tracker retired.
- 9.9 PARKED — Parameterized session set; behind W3 green + soak (D-P-10).
- 9.10 RATIFIED — Quote accessor.
- §3-CORRECTION RATIFIED — Direction stays platform (generic); the side→direction MAP is
  plugin-owned wire vocabulary. Enforced W1 P2c/P4c: constant removed, plugin ships
  lowercase {"low":"long","high":"short"}, emitter sources it verbatim. (Supersedes the
  D-E3-e "cosmetic" classification — OVERTURNED, see PROGRESS POST-E3.)
- 9.11 RATIFIED (2026-07-28) — TIME-bar construction convention: `Bar` carries a `kind`
  discriminator (`BarKind`, appended last, default TICK — every pre-Phase-F construction
  site unchanged); a TIME bar's `timeframe_ticks` is the interval in SECONDS because 0 is
  an existing "unspecified" sentinel and `BarSpec.size` already declares the seconds
  convention for TIME; bar ids are `<n>s:<trading_day>:<index>` (tick `<n>t:` unchanged);
  NO empty bars (an empty bucket emits nothing); `bar_index` is DENSE over emitted bars
  per (timeframe, trading_day); DERIVE ONCE, AGGREGATE UPWARD (60s built from trades
  once, all higher timeframes aggregate the 60s bars); buckets are anchored at the
  trading-day boundary's ABSOLUTE UTC instant (DST-correct by construction). Why: one
  ratified construction so batch ≡ streaming by design, with completeness observable
  without a wall clock. Enforced: SC `candles/time_streaming.py` + `candles/time_batch.py`
  + `candles/_buckets.py`, parity lock `tests/test_time_bar_parity.py`, rule units
  `tests/test_time_bars.py` (TIMEBAR window, commits `fc0881c..7992323`). AMENDMENT to
  9.4 recorded here (original text untouched): 9.4 is PARTIALLY DISCHARGED — time-bar
  CONSTRUCTION has landed; plugin delivery and multi-timeframe routing remain Phase F1.

- 9.12 RATIFIED (2026-07-28, IFVG window) — FVG construction convention (census-canonical,
  promoted verbatim from the archived census): three-bar geometry with STRICT inequalities
  (A.high < C.low bullish / A.low > C.high bearish, B irrelevant); a gap EXISTS only at its
  own timeframe's close (`confirmed_ts` = availability instant); triplets SPAN trading days
  (detector never resets at the roll; cross-day continuity = the snapshot tail seam); fills
  are physical price events at 1m granularity, wicks count, participation STRICTLY AFTER
  confirmation; filled gaps are dead context. Why: one ratified geometry so research capture,
  replay, and future live serving share a single detector. Enforced: `structures/fvg.py`,
  parity lock `tests/test_fvg_parity.py` (continuous ≡ seeded, cross-day triplets exercised).
- 9.13 RATIFIED (2026-07-28, IFVG window) — WIDE-capture / ML-learns-the-discretion: the
  `ifvg_smc` FSM enforces only HARD invariants (own-TF-close existence, 1m body-close
  inversion, no entry on the inversion candle, availability guards, risk >= 1 tick) plus
  WIDE capture bounds (IFVG_* constants, 3-5x doc defaults, existence-only floors); every
  soft threshold from ifvg-strat.md §6 (gap sizes, distances, windows, timeouts, locality,
  sweep quality) is an EMITTED MEASUREMENT, never a gate; every slot-rejected candidate is
  emitted `selected=False` + drop_reason; causality and entry family are per-row flags
  (explicit pooling, never silent). Why: owner ruling — hardcoded discretion destroyed data
  before (dedup hid 89.6% of RTH sweeps; census sizes 3-6.6x the ICT tick priors); the
  downstream gate model learns the thresholds and the doc defaults become a baseline filter.
  Enforced: `strategies/ifvg_smc/` (section/records/reducer), canonical per-1m-close order
  in `replay.py` module contract (fills -> invalidations -> expiries -> transitions ->
  intake), parity locks `tests/test_ifvg_replay_parity.py` (chained run_day ≡ continuous;
  plugin fold ≡ run_day).
- 9.9 AMENDMENT (2026-07-28, IFVG window) — implemented MINIMALLY, opt-in only:
  `StrategyLevelState` gains `session_range_names` (default `("asia","london")` — default
  construction byte-identical, regression-locked) and `emit_prior_session_levels` (banked
  at the roll, `prev_<session>_high/low` available from the trading-day start), plus
  `load_prior_session_range` / `data/prior_day.py::prior_day_session_extremes` seeds.
  ifvg constructs with `("asia","london","ny")` + `("ny",)`. Intraday running day H/L is
  NOT emitted (self-touch look-ahead class); confirmed swings serve "recent extremes".
  The 9.9 full parameterized-session-set ruling stays PARKED; touch serving untouched.

## Convergence rulings (ratified 2026-06-10..13; ratification record: PROGRESS POST-E3)
- D-P-01 RATIFIED — Tracer-dye: zero rehab of pre-convergence models; acceptance on a
  FRESH model through the converged pipeline. Why: old bundles trained on definitions
  we abandoned. Enforced: W3 acceptance design; gate refuses old 120-min bundles.
- D-P-02 RATIFIED — Legacy QL artifacts untrusted, never tested against; deletion not
  repair. Why: rotted harnesses prove nothing and invite false confidence.
  Enforced: W1 moved legacy stages tests-only; W2 deleted parity_harness*, phase8_1
  golden+captures, audit_lookahead; dashboard POC removal deferred.
- D-P-03 RATIFIED — L1-consumption invariant: live is MBP-1, so research/replay consume
  ONLY the L1 projection (trades + TOB-change-deduped quotes) of the mbp-10 parquets
  through ONE canonical SC reader; parquets are the sole research/replay source (incl.
  >1yr regimes); the historical API exists solely for the warm-start slice. Why:
  train/serve parity extends to the input schema. Enforced: W1 ingestion layer + gate
  check 8; W2 route-seam shape check.
- D-P-04 RATIFIED — Flatten: flat by NY futures close daily; FLATTEN_TIME 16:40 ET
  (= 15:40 CT) anchored to the decision's OWN trading day. Why: the day-context-free
  compare embargoed evenings (F17); the time was always right. Enforced: W1 P2a both
  resolvers + pinned tests; 15:55 executor deprecated with its stack.
- D-P-05 RATIFIED — F6/F7 research-causality fixes ride the convergence; future
  training-data change accepted. Why: iFVG trains after convergence; old models are
  tracer dye. Enforced: W1 P4a stream drive (per-trade session fold, causal zones).
- D-P-06 RATIFIED — Warm-start: at startup AND reconnect, feed the trading day's events
  from 18:00 ET (Databento live replay-start or historical API) before going real-time;
  Chicago seed retired. Why: live state must equal what training saw from the same
  instants. Enforced: W2.
- D-P-07 RATIFIED — Prediction journaling: append-only predictions/outcomes/drops
  surviving restarts. Why: soak evidence must outlive the process. Enforced: W2.
- D-P-08 RATIFIED — Repo privacy flip deferred indefinitely; tokens land and prove
  green in CI BEFORE any flip. Why: flip-first is a self-inflicted outage of the gate
  chain. Enforced: standing operator rule.
- D-P-09 RATIFIED — platform-refactor is never squash-merged. Why: pinned-SHA orphan
  risk. Enforced: standing git rule.
- D-P-10 RATIFIED — S9.9 and Phase F are PARKED behind the W-plan (W1 ONE SURFACE ✅ →
  W2 OPERATE → W3 PROVE → live + soak). Why: building TIME bars and iFVG on an
  unconverged path discovers defects with money on. Enforced: roadmap; PROGRESS status.
- D-P-11 RATIFIED (2026-06-12) — Verification-of-completion discipline, three rules:
  (a) a record's claim of completion is never evidence of completion — only a verified
  artifact (diff, SHA, CI run) is; every CC deliverable requires a receipt verified by
  the reviewer before the next prompt ships; (b) window-close gates (pytest/ruff) run
  against the COMMITTED tree, never the working tree; (c) any unreviewed commit needed
  mid-greenlight, however trivial, is stop-and-report first. Why: three same-family
  failures in one week — the F3 diff-grep error, the W2 dirty-tree "ruff clean", and
  DECISIONS.md itself recorded as created without ever existing. Enforced: reviewer
  procedure + standing clauses in greenlight/window prompt templates.
- D-P-12 RATIFIED (2026-06-12) — Soak pre-registration (criteria fixed BEFORE the soak
  exists). Duration: 7 trading days where the system ran the NY session end-to-end.
  Day 0 (supervised, not counted): one watched live session verifying warm-start path
  taken, warming→live flip, prior-day PDH/PDL seed, predictions+journal flowing, clean
  stop→flush→terminal records; the clock starts on the first unsupervised full day
  after a clean shakedown. SYSTEM criteria (gating): (1) zero silent failures — every
  disconnect auto-recovers via warm-start or surfaces FAILED visibly; (2) journal
  completeness — every prediction reaches a terminal record by cutoff+flush, one file
  per trading day, zero append failures; (3) inference health — no named-feature
  failure storms, unbounded error-count growth fails; (4) resolver integrity — no
  orphan-flush warnings; (5) serving paths frozen — no SC/TL backend changes mid-soak;
  pre-declared allowed list only (viz Tier A frontend-only; metrics script QL-side).
  Restart-vs-extend, decided now: serving-path code fix → clock RESTARTS; environmental
  outage (ISP/power/provider) → clock EXTENDS by the lost days. MODEL criteria are
  recorded, never gating: live NY results vs the bundle's own OOS expectations
  (gated-class hit rate, MFE/MAE distributions, drop-reason mix) — divergence is thesis
  information unless values are systematically impossible (then it gates under (3)).
  Adjudication artifact: the metrics script's report over the soak journal. Why:
  criteria set after seeing results aren't criteria. Enforced: this entry; W3
  greenlight precedes any soak day.
- D-P-13 RATIFIED (2026-06-12) — W3 proof-bundle training configuration: the full
  explicit config recorded verbatim as Quant-Lab docs/DECISIONS.md D-036 (window
  2025-11-21→2026-02-13, purged 40/5/5/2 trading-day folds, pinned 5-feature set,
  v3 label policy with explicit sl=15/aw=15 overrides, CatBoost defaults, quality
  not a save gate). D-037 (same file) supersedes the stub-exclusion pre-ruling:
  app_max_spread RIDES the W3b gate; the quotes_in_window stub is SC-plugin-path-only.
  Why: one reproducible owner-ratified config; the gate must test what production
  serves. Enforced: W3a plumbing (day_folds + pinned_features); W3b gate config.
- D-P-14 RATIFIED (2026-06-12) — Reader-cost ruling: the W3 dataset build runs ONCE
  overnight at measured cost (24.1 min/day × 60 days ≈ 24.1 h; per-day caches make
  same-config reruns near-instant), weekend-absorbed; src/strategy_core stays frozen
  through W3 + soak. A READER-PERF WINDOW is scheduled post-soak: scope = SC batch/
  vectorized event decode (W3a option a) + QL single-pass quote collection (option b),
  full drift net (b3 digests, duckdb parity, decision-diff, golden) gates it. Why:
  ~34µs/event × 15.3M events × 2 passes is a platform tax on all future research, but
  fixing it mid-PROVE puts engine churn under the proof; the weekend absorbs the build
  for free. Enforced: W3a record (options preserved); post-soak roadmap.
- D-P-15 RATIFIED (2026-06-12) — W3A-READER: D-P-14's SC-freeze is SUPERSEDED for one
  component, pre-build — `DatabentoParquetSource`'s row-by-row decode internals are
  vectorized BEFORE the W3 dataset build, gated by (1) a full-day event-by-event
  IDENTITY PROOF on 2022-02-15 + 2026-02-18 (count, order, every field, ns-exact ts;
  `W3A_READER_PROOF.log`: 5,218,914 + 10,720,798 events IDENTICAL, zero warnings; warm
  drains ×13.0/×12.5) and (2) the full drift net green on the committed P2 tree (SC 184
  = tests + validation: b3 golive/multiday digests, duckdb-streaming parity,
  decision-diff, d1 streaming-vs-batch, production-pair parity; repo-root ruff; QL 751
  via the editable install). The row-wise path was deleted only on full green (the W3a
  P1 quote-pass convention; the proof harness re-runs at the pre-delete commit
  `332ad3e`). Documented edge deviations, all unreachable on store data (zero-warning
  proof tallies attest): inf-price / out-of-datetime-range-timestamp / uint64-overflow
  cells now warn-and-skip the ROW where the row-wise path aborted the FILE (and
  |price/tick| > 2^62 now warns instead of emitting a wrapped int). D-P-14's REMAINING
  scope — L1 event cache, single-pass quote collection, parallel day-pool — stays
  post-soak. Why: the stop-gated build was ~2×513s/day of reader decode; an
  output-identical reader swap under proof + net is churn-free and cuts the build
  projection 24.1 h → ~8.7 h (the residual is the non-reader ~420s/day — exactly the
  post-soak scope). Enforced: this entry; the W3a record's W3A-READER subsection;
  proof log + drift-net tallies as receipts.
- D-P-16 RATIFIED (2026-07-03) — W3b READER ADOPTION on the reduced-coverage re-gate:
  the vectorized `DatabentoParquetSource` (SC `332ad3e` decode + `37359ae` row-wise
  delete) is THE reader — research (QL), serving (TL), and the gate all consume it;
  the pre-vectorization installed reader is RETIRED (no consumer may resolve it: both
  pins move to the adopting SC tip, and the rebuilt bundle `NQ_W3_20260617T220752Z`
  supersedes the stale-reader `NQ_W3_20260613T055600Z` as the W3 artifact). Acceptance
  evidence, two independent legs: (1) CORRECTNESS — `READER_CORRECTNESS_PROOF.md`
  (TL root, 2026-06-17): byte-correct vs raw parquet, incl. a full 1,968,223-event
  trading day and the 159,256-event window straddling the disputed 10:54 ET PDL touch.
  (2) BATCH≡STREAM on the rebuild — the REDUCED-COVERAGE re-gate: 30-day journal base
  (2025-11-21→12-17 contiguous + 12-18 + 12-23 + a 5-day spread) 92/92 HARD-GREEN;
  dense 7-day set {12-18, 12-23, 01-20, 01-29, 02-03, 02-12, 02-13} 29/29 HARD-GREEN
  at workers=2 AND workers=4 with IDENTICAL per-day rows (worker-count nondeterminism
  ruled out); 2-day bench 8/8 at workers=1/2; 0 mismatches on every axis in every run;
  watchdog(120 s) fires: 0. Why REDUCED (not full-73-day) coverage is accepted: the
  spread is stress-weighted — it contains the densest days (6 of the 7 dense-set days
  ≥1 GB) and BOTH previously-problematic days, each now explained and green (12-18,
  the end-of-stream wedge = a harness/runtime terminalization livelock, fixed at both
  layers + regression-tested, completes 13,483,063 events; 02-12, the stale-cache RED
  = None-seeded standalone cache, regenerated seeded + QL `098e354` guard, now 5/5).
  The drift class the gate exists to catch was thereby empirically DISCHARGED on the
  hardest inputs — greens on non-problematic days are informative because the failure
  modes were run down to root cause, not assumed away — and the 06-16 full-73-day
  236/236 stale-reader run stands as CONSISTENCY evidence that the shared pipeline is
  deterministic end-to-end. The un-re-gated remainder (38 of 73 days) is accepted
  risk, bounded by legs (1)+(2). References: D-P-15 (the identity-proof vectorization
  this adoption completes); PROGRESS §W3b CLOSE (the coverage table + run artifacts).
- D-P-17 ADOPTED IN-WINDOW (2026-07-11) — INGEST: action-bearing TOB schemas emit
  trades. `DatabentoParquetSource._decode_batch` classifies `action∈{T,TRADE}` rows as
  Trade events for `is_tob` schemas (mbp-1/cmbp-1/tbbo) WHEN the file carries an
  `action` column — exactly the mbp-10 rule (the same row still emits its Quote iff
  L1 TOB changed; ordering follows the unchanged canonical key (ts, seq,
  side-signed-price, size): a sell Trade sorts before its row's Quote, a buy Trade
  after — identical to mbp-10 on identical rows, pinned by test). bbo/cbbo files
  carry no action column and keep quotes-only behavior. This AMENDS the D-P-15 proof
  contract line "mbp-1/bbo/tbbo: every row -> Quote only", which was vacuous until
  now: no mbp1/bbo/tbbo parquet file existed in any store, so no consumer sees a
  behavior change on existing data (the live path is separate code). Why: the INGEST
  window converts the MBP-1 batch download (2026-01-11..2026-07-10) into
  `mbp1.parquet` store days under the honest-naming ruling; without trade
  classification those days drain quotes-only, the D-036 per-day build (trade-bar
  clock, `engine_decision.py` Trade-only drive) returns empty datasets, and PDH/PDL
  prior-day extremes break — i.e. the window's own fresh-day gate (P4b) is
  unsatisfiable. Scope deviation note: the INGEST work order scoped SC changes to
  file discovery only (moot — `DAY_FILE_PRIORITY` has been mbp1-aware since W1);
  this classification change is the minimal amendment that makes honestly-named
  mbp-1 days first-class, taken in-window and flagged in the close record. Gated by:
  oracle tests (mbp-1 fixture with T rows → Trade+Quote emission; bbo fixture →
  quotes-only) + the full SC suite + the P4a overlap identity gate (mbp10 vs
  converted mbp1 through the reader: trades AND deduped L1 quotes, event-by-event).

- D-P-18 RATIFIED (2026-10-03) — IFSM Task A1 MenthorQ entry context. The binding
  source is `C:/tasks/ifsm_task_a1_level_context/decisions.md`, decisions 2–7,
  10, 12 and 13, and `TASK_A1.md`. CSV parsing and source receipts remain in
  Quant-Lab. Core accepts an immutable `MenthorqLevelSnapshot` through the existing
  `levels_for` input, extended by `IfvgLevelInputs`; the plugin's static-level
  handoff accepts the same timestamp callback. Neither reducer I/O nor a new
  record/feature family is introduced. Derivation uses honest entry ticks times
  the instrument tick size, all 19 EOD level columns in file order, strict
  above/below comparisons and all tied names. Normalized distances/opening moves
  remain null without a positive implied move or the required prior cash close.
  `menthorq_context_version=None`, `regime_gate_policy=off`,
  `regime_unknown_policy=allow` and `nearest_support_gex1_block=false` are named
  neutral exclusions in `ifvg_profile_hash`; every changed value enters that
  identity. Gate overrides require `menthorq_eod_v1` and otherwise fail with
  `menthorq_gate_requires_eod_context`. The version field is `str | None`;
  `unsupported_menthorq_context_version` rejects unknown non-null versions.
  At the final candidate timestamp, only a longs-only strategy evaluates the
  new gates; enabling shorts bypasses both. Session/schedule reasons precede
  `regime_gate`, `context_unavailable` or `nearest_support_gex1`. Blocked
  candidates retain geometry/evidence and emit no eligible decision. Decision
  12 takes precedence outside `[06:00,17:00)` Chicago: both gates pass with null
  flags and `not_applicable_outside_hours`, including unknown-policy `block`.
  Inside that interval, missing level rows or unknown regimes use the explicit
  allow/block policy for an enabled regime gate. No below level passes the
  support gate with a null flag. Other review statuses are `evaluated`,
  `context_unavailable` and `not_applicable_shorts_enabled`.
  A1's runtime CSV export is review material, outside manifests and artifact
  identities; the permanent record family and ML allowlisting remain Task A2.
  No targets, calendars, capacity limits, dependencies or sealed-date guards
  change. Decision 13 permits source-identity-only differences during parity;
  numerical and record behavior with neutral fields must remain unchanged.

## How to add an entry
New ruling in a design exchange → it gets the next D-P id, lands here in the same
window's doc op, and PROGRESS cites the id. A window prompt that implements a ruling
names its id in the prompt header.
