# ERA_GATE_RECON — gating the MBP-1 era and pre-registering the training arc

Read-only recon. No tracked file was modified. No dataset build, cache warm, training run,
scoring, or W3b replay was executed. No parquet covering a trading day in the sealed range
2026-06-12..2026-07-10 was opened; that range appears here only as directory listings and
file metadata.

## Tip verification

`git rev-parse HEAD` + `git status --porcelain=v2 --branch`, run 2026-07-25 before any part:

| Repo | Path | Expected | Actual | Branch / ahead-behind | Tracked modifications |
|---|---|---|---|---|---|
| SC | `C:\Users\gonza\Documents\Strategy-core` | `4740ecd` | `4740ecdd460caf4ed2c8038fc37c23b2b1d17788` | `platform-refactor`, `+0 -0` vs `origin/platform-refactor` | NONE (46 `?` untracked entries only) |
| TL | `C:\Users\gonza\Documents\Trade-Lab` | `7a911c0` | `7a911c0ce0ae5d9e2c97339b7d897377a37a3647` | `platform-refactor`, `+0 -0` | NONE (22 `?` untracked entries only) |
| QL | `C:\Users\gonza\Documents\Claude-Quant-Lab` | `27cdfd6` | `27cdfd6630898c7b31559fbe57f009701f8ed931` | `platform-refactor`, `+0 -0` | NONE (36 `?` untracked entries only) |

All three tips match. No repo carries a tracked modification (`porcelain=v2` emitted no `1`/`2`/`u`
records in any repo). `ERA_GATE_RECON.md` was absent from the QL untracked set at start, so this
file is a fresh emit, not an append.

**Store identity (established during this recon, load-bearing for Parts A/D/E).**
`cmd /c dir /AL C:\Users\gonza\Documents\Trade-Dashboard` reports:

```
03/19/2026  10:58 PM    <SYMLINKD>     data [C:\Users\gonza\Documents\Claude-Quant-Lab\data]
```

TL `backend/.env:24` sets `TRADE_LAB_DATA_PATH=C:\Users\gonza\Documents\Trade-Dashboard\data\databento\NQ`,
which therefore resolves to `C:\Users\gonza\Documents\Claude-Quant-Lab\data\databento\NQ`. The TL
replay store and the QL store are ONE directory: 433 day dirs and 156 `mbp1.parquet` files under
either path, and `stat -c '%d %i %s %Y'` on `2026-07-10/mbp1.parquet` returns the identical
`(device, inode, size, mtime)` tuple through both paths. Every statement below about "the store"
refers to that single directory regardless of which repo's path names it.

---

# PART A — W3b harness: what it takes to gate ONE arbitrary day

Harness surface = 5 modules under `C:\Users\gonza\Documents\Trade-Lab\backend\scripts\w3b\`:
`__init__.py`, `window.py`, `headless_replay.py`, `parity.py`, `run_window.py` (plus `REPORT.md`,
self-marked HISTORICAL at `REPORT.md:1-2`).

SC pin: TL `backend/pyproject.toml:17` pins `strategy-core @ ...@9d4935346bf42c5d19916e05dbe21dd67c46875c`;
the installed `site-packages/strategy_core/data/databento_parquet.py` is byte-identical to the SC
working tree (`diff -q`), so SC quotes below are the code that actually executes.

## A1. Harness surface

### A1.1 BUNDLE_ID — `backend/scripts/w3b/window.py:50`

```python
49	# ── Bundle / contract constants (the bundle under test) ──────────────────────
50	BUNDLE_ID = "NQ_W3_20260617T220752Z"
51	EXPECTED_CACHE_TAG = "7850272e"
52	CACHE_FILENAME = f"ml_utility_{EXPECTED_CACHE_TAG}.parquet"
53	TICK_SIZE = 0.25
```

Current value: **`"NQ_W3_20260617T220752Z"`**. Two stale references name the PREVIOUS bundle and are
not load-bearing: `backend/scripts/w3b/__init__.py:5` (`... that trained bundle
``NQ_W3_20260613T055600Z``.`) and `backend/scripts/w3b/REPORT.md:12` (`Bundle under test:
NQ_W3_20260613T055600Z`). Both bundle dirs exist under `QL\models\`.

Window definition, `window.py:72-96` (the ratified D-036 launch argv copied from QL
`scripts/w3_cache_warmer.py`), and the QL repo root at `window.py:99`:

```python
72	#: The exact ratified D-036 launch argv (from QL scripts/w3_cache_warmer.py).
75	_EXP_ARGV = [
76	    "--preset", "all_to_ny",
77	    "--symbol", "NQ",
78	    "--bar-type", "147t",
79	    "--start", "2025-11-21",
80	    "--end", "2026-02-13",
...
99	QL_REPO = Path(os.environ.get("W3B_QL_REPO", r"C:\Users\gonza\Documents\Claude-Quant-Lab"))
```

### A1.2 `run_window` — full signature, parameters, defaults

`backend/scripts/w3b/run_window.py:75-86`:

```python
75	def run_window(
76	    days: list[str] | None = None,
77	    *,
78	    journal_base: Path = _DEFAULT_JOURNAL_BASE,
79	    workers: int = 1,
80	    score: bool = True,
81	    resume: bool = True,
82	) -> list[DayDiff]:
83	    window = resolve_window()
84	    targets = days or list(window.window_dates)
85	    journal_base.mkdir(parents=True, exist_ok=True)
86	    results: list[DayDiff] = []
```

| Param | Kind | Default |
|---|---|---|
| `days` | positional-or-kw, `list[str] \| None` | `None` → `list(window.window_dates)` (`run_window.py:84`) |
| `journal_base` | kw-only, `Path` | `_DEFAULT_JOURNAL_BASE` = `backend/data/w3b_journal` (`run_window.py:38`, cwd-independent) |
| `workers` | kw-only, `int` | `1` (serial; `>1` → `ProcessPoolExecutor`, `run_window.py:107`) |
| `score` | kw-only, `bool` | `True` (builds `OfflineScorer.for_bundle(window)`, `run_window.py:56`) |
| `resume` | kw-only, `bool` | `True` |

`run_window()` called as a Python function performs **no** window-membership validation; the
`--days` check lives only in `_main` (A4.1). Per-day driver, `run_window.py:41-62`:

```python
41	def _process_day(
42	    date_str: str, journal_base: Path, window, *, score: bool, resume: bool
43	) -> DayDiff:
52	    from w3b.headless_replay import replay_day_to_journal
54	    journal_dir = journal_base / date_str
55	    jpath = journal_dir / f"{date_str}.jsonl"
56	    scorer = OfflineScorer.for_bundle(window) if score else None
57	    if resume and jpath.exists():
58	        diff = diff_day(date_str, jpath, window=window, scorer=scorer)
59	        if diff.green:
60	            return diff
61	    replay_day_to_journal(date_str, journal_dir=journal_dir, window=window)
62	    return diff_day(date_str, jpath, window=window, scorer=scorer)
```

### A1.3 `diff_day` — signature, every comparison axis, every tolerance

Signature, `backend/scripts/w3b/parity.py:319-346`:

```python
319	def diff_day(
320	    date_str: str,
321	    journal_path: Path,
322	    *,
323	    window: W3Window | None = None,
324	    scorer: OfflineScorer | None = None,
325	    serving_only_counter: Callable[[ServingTouch], int] | None = None,
326	) -> DayDiff:
334	    window = window or resolve_window()
335	    training = load_training_touches(window, date_str)
336	    serving = parse_journal(journal_path)
337	    if serving_only_counter is None:
338	        serving_only_counter = QLInteractionTradeCounter(window, date_str)
339	    return diff_touch_sets(
340	        date_str,
341	        training,
342	        serving,
343	        cache_present=window.has_cache(date_str),
344	        scorer=scorer,
345	        serving_only_counter=serving_only_counter,
346	    )
```

Tolerance constants, `parity.py:52-67`:

```python
53	_DECISION_OFFSET = timedelta(minutes=5)
56	_PROBA_TOL = 1e-6
58	_ELIGIBLE_CLASS = "tradeable_reversal"
59	_ELIGIBLE_SESSION = "ny"
60	_CONFIDENCE_GATE = 0.70
63	_INSTANT_TOL_NS = 0
67	_INTERACTION_MIN_TRADES = 5
```

Join key `(level_type|level_kind, direction)` with duplicate detection, `parity.py:315-316` and
`:376-387`:

```python
315	def _key(level_type: str, direction: str) -> str:
316	    return f"{level_type}|{direction}"
```
```python
376	    train_by_key: dict[str, TrainingTouch] = {}
377	    for t in training:
378	        k = _key(t.level_type, t.direction)
379	        if k in train_by_key:
380	            diff.duplicate_keys.append(f"train:{k}")
381	        train_by_key[k] = t
382	    serv_by_key: dict[str, ServingTouch] = {}
383	    for s in serving.survivors:
384	        k = _key(s.level_kind, s.direction)
385	        if k in serv_by_key:
386	            diff.duplicate_keys.append(f"serv:{k}")
387	        serv_by_key[k] = s
```

**Axis table — every asserted axis and its tolerance:**

| Axis | Code | Tolerance |
|---|---|---|
| Set reconciliation (training-only / serving-only) | `parity.py:389-400` | exact set equality; `unmatched_training` always RED, `unmatched_serving` RED unless Finding-1 reconciles |
| Touch instant | `parity.py:407-417` | **0 ns** (`_INSTANT_TOL_NS = 0`), serving instant = `ts_utc − 5 min` (`parity.py:151`) |
| Features (5 contract features) | `parity.py:419-433` | **exact** `==`, NaN==NaN via `_exact` (`parity.py:489-492`) |
| Label | `parity.py:435-439` | exact string equality (`t.label != s.actual_class`) |
| MFE / MAE | `parity.py:441-449` | **exact** float equality via `_exact` |
| Level price | `parity.py:451-461` | exact integer-tick equality after `round(rep_px / 0.25)` |
| Probabilities | `parity.py:463-478` | **`1e-6`** (`_PROBA_TOL`) |
| Eligibility | `parity.py:479-485` | exact boolean vs `_offline_eligible` (`parity.py:495-501`) |
| Orphan predictions | `parity.py:372`, `:186-187` | count must be 0 |

Representative quotes:

```python
407	        inst_diff = abs(int(t.touch_instant.value) - int(s.touch_instant.value))
408	        diff.max_instant_diff_ns = max(diff.max_instant_diff_ns, inst_diff)
409	        if inst_diff > _INSTANT_TOL_NS:
```
```python
419	        # (b) features, tol=0
420	        for name in CONTRACT_FEATURES:
421	            tv = t.features[name]
422	            sv = s.feature_values.get(name)
430	            if not _exact(tv, sv):
```
```python
451	        # (d) representative_price <-> level_price_ticks (nearest tick)
452	        expected_ticks = round(t.representative_price / TICK_SIZE)
453	        if expected_ticks != s.level_price_ticks:
```
```python
473	                d = abs(ov - sv)
475	                if d > _PROBA_TOL:
479	            offline_eligible = _offline_eligible(offline, s.session)
480	            if offline_eligible != s.is_eligible:
```
```python
489	def _exact(a: float, b: float) -> bool:
490	    if math.isnan(a) and math.isnan(b):
491	        return True
492	    return a == b
```
```python
495	def _offline_eligible(probabilities: dict[str, float], session: str) -> bool:
496	    predicted = max(probabilities, key=lambda label: probabilities[label])
497	    return (
498	        predicted == _ELIGIBLE_CLASS
499	        and session.split("_", 1)[0] == _ELIGIBLE_SESSION
500	        and probabilities.get(_ELIGIBLE_CLASS, 0.0) >= _CONFIDENCE_GATE
501	    )
```

`CONTRACT_FEATURES` (`window.py:57-63`): `int_time_within_2pts`, `int_absorption_ratio`,
`app_avg_trade_size`, `app_large_trade_vol_pct`, `app_max_spread`.

Green predicate, `parity.py:275-289` — all eight mismatch lists empty AND `orphan_predictions == 0`:

```python
275	    @property
276	    def green(self) -> bool:
277	        return (
278	            not self.unmatched_training
279	            and not self.unmatched_serving
280	            and not self.duplicate_keys
281	            and self.orphan_predictions == 0
282	            and not self.feature_mismatches
283	            and not self.label_mismatches
284	            and not self.excursion_mismatches
285	            and not self.price_mismatches
286	            and not self.instant_mismatches
287	            and not self.proba_mismatches
288	            and not self.eligible_mismatches
289	        )
```

**Explicitly NOT compared** (`parity.py:19-20`): `int_time_beyond_level` (cached, unpinned, not
served) and `entry_price` / `decision_time` (serving-only / training-only). `session` is parsed on
both sides but is not an equality axis — it feeds `_offline_eligible` only.

One further axis exists at REPORT level only, not in `DayDiff.green` — the thin-day `0 == 0` rule,
`run_window.py:160-161`:

```python
160	    # thin-day 0==0: a day without a cache must yield no surviving touch
161	    thin_violations = [d.day for d in thin_days if d.serving_survivors > 0]
```

### A1.4 `headless_replay.py` — entrypoint, parameters, invocation

Library entrypoint, `backend/scripts/w3b/headless_replay.py:253-260`:

```python
253	def replay_day_to_journal(
254	    date_str: str,
255	    *,
256	    journal_dir: Path,
257	    window: W3Window | None = None,
258	    bundle_id: str = BUNDLE_ID,
259	    clear: bool = True,
260	) -> ReplayResult:
```

Invoked from the harness at `run_window.py:52` + `:61` (`bundle_id` left at default; `run_window`
exposes no `--bundle` flag):

```python
52	    from w3b.headless_replay import replay_day_to_journal
61	    replay_day_to_journal(date_str, journal_dir=journal_dir, window=window)
```

The recorded `trading_day` / `symbol_dir` parameters are **not** parameters of
`replay_day_to_journal` — they are `ReplayConfig` fields it constructs, `headless_replay.py:296-305`:

```python
296	    replay = HistoricalReplayService(runtime)
297	    config = ReplayConfig(
298	        paths=(window.symbol_dir,),
299	        requested_symbol=_REQUESTED_SYMBOL,
300	        schema=_REPLAY_SCHEMA,
301	        trading_day=date.fromisoformat(date_str),
302	        symbol_dir=window.symbol_dir,
303	        speed=0.0,
304	    )
305	    state = asyncio.run(_drive(replay, source, config))
```
```python
57	# The day files are ``mbp10.parquet``; for_trading_day auto-detects the file, so
58	# this is only the feed-status label (the activation gate checks the contract's
59	# data_requirements, not this).
60	_REPLAY_SCHEMA = "mbp-10"
62	#: requested_symbol the QL cache build used (for_trading_day(requested_symbol="NQ"),
63	_REQUESTED_SYMBOL = "NQ"
```

Its own CLI, `headless_replay.py:320-337`:

```python
320	def _main() -> int:
321	    parser = argparse.ArgumentParser(description=__doc__)
322	    parser.add_argument("date", help="trading day YYYY-MM-DD (must be in the D-036 window)")
323	    parser.add_argument(
324	        "--journal-dir",
325	        type=Path,
326	        default=None,
327	        help="scratch journal root (default: backend/data/w3b_journal/<date>)",
328	    )
329	    parser.add_argument("--bundle", default=BUNDLE_ID)
330	    args = parser.parse_args()
333	    window = resolve_window()
334	    if args.date not in window.window_dates:
335	        raise SystemExit(
336	            f"{args.date} is not in the D-036 window ({len(window.window_dates)} days)"
337	        )
```

### A1.5 `_SeedingSource` — location, seed key, ordering vs the SEED-window service seed

`backend/scripts/w3b/headless_replay.py:99-115`:

```python
99	class _SeedingSource(HistoricalMarketDataSource):
100	    """Wrap the day source so the PDH/PDL seed fires at the right instant.
101
102	    ``HistoricalReplayService.start`` resets the runtime (rebuilding the
103	    Strategy-Core service) and THEN pulls events from this source inside the
104	    replay task. Seeding in ``scan`` — the first thing the task does before the
105	    first event — lands the prior-day summary on the post-reset service, exactly
106	    where the live warm-start would seed it ahead of the intraday stream.
107	    """
108
109	    def __init__(self, inner: HistoricalMarketDataSource, seed_fn) -> None:
110	        self._inner = inner
111	        self._seed_fn = seed_fn
112
113	    def scan(self, paths: Iterable[Path], **kwargs) -> Iterator:
114	        self._seed_fn()
115	        yield from self._inner.scan(paths, **kwargs)
```

Seed function and key, `headless_replay.py:283-295`:

```python
283	    seed = seed_for_day(date_str)
285	    def _seed_fn() -> None:
286	        if seed is None:
287	            return
288	        high_pts, low_pts = seed
289	        runtime.strategy_core_service.load_prior_day_summary(
290	            prior_trading_day(date_str),
291	            high_ticks=round(high_pts / TICK_SIZE),
292	            low_ticks=round(low_pts / TICK_SIZE),
293	        )
295	    source = _SeedingSource(HistoricalParquetAdapter(front_month_only=True), _seed_fn)
```

Key = `prior_trading_day(date_str)` = **calendar D − 1** (`window.py:196-200`):

```python
196	def prior_trading_day(date_str: str) -> date:
197	    """The (calendar) prior day used as the PDH/PDL summary key, matching QL's
198	    ``load_prior_day_summary(td - timedelta(days=1), ...)``."""
200	    return date.fromisoformat(date_str) - timedelta(days=1)
```

Value = QL's rolling `prev_full_hl` for the day (`window.py:165-193`), points→ticks.

**Does it still run after the SEED-window service seed? YES.** Traced:

1. `_drive` awaits `replay.start(...)` before anything pulls events (`headless_replay.py:196-201`).
2. `start` → `_start_locked` does `runtime.reset(...)` and THEN the SEED-window service store-walk
   seed, synchronously, before the task exists — `backend/src/trade_lab/services/replay.py:219-254`:

```python
219	        await self._emit(
220	            self.runtime.reset(
226	        # SEED: give day-mode replays the training-parity PDH/PDL seed — the canonical
227	        # store walk (SEED_PARITY_RECON.md §5: tick-exact equal to QL's prev_full_hl
228	        # carry). Must run AFTER runtime.reset (the reset rebuilds the SC service, so an
229	        # earlier seed would be wiped) and BEFORE the core replay task starts (so the
230	        # summary is banked before the first event). A walk miss or a seed failure never
231	        # kills the replay — it proceeds unseeded, QL's cold-start equivalent.
233	        if config.trading_day is not None and config.symbol_dir is not None:
235	                extremes = await asyncio.to_thread(
236	                    prior_full_day_extremes,
237	                    config.symbol_dir,
238	                    config.trading_day,
239	                    requested_symbol=config.requested_symbol,
240	                    max_walk_days=_SEED_MAX_WALK_DAYS,
241	                )
249	                    self.runtime.levels.load_prior_day_summary(
250	                        extremes.source_day,
251	                        high_ticks=extremes.high_ticks,
252	                        low_ticks=extremes.low_ticks,
253	                    )
```

   (`_SEED_MAX_WALK_DAYS = 10`, `replay.py:39`.)
3. Only after that is the task created (`replay.py:281-292`), and it drains `adapter.events()` →
   `self._source.scan(...)` (`replay.py:55-62`) — which IS `_SeedingSource.scan`, whose first
   statement is `self._seed_fn()`.

Ordering: **runtime.reset → service store-walk seed (key = `extremes.source_day`) → W3b `_seed_fn`
(key = calendar D−1) → first event.** Both write the same object (`runtime.py:175`/`183`:
`self.levels = self.strategy_core_service`). SC emits from the max key strictly below the trading
day (`Strategy-core/src/strategy_core/runtime/levels.py:96-100`), and since
`extremes.source_day ≤ D−1`, the **W3b seed wins whenever `seed_for_day` is not None**. When
`seed_for_day` returns `None` — first window day, or **any day not in `window_dates`**
(`window.py:183-186`) — `_seed_fn` no-ops and the service store-walk seed stands alone.

### A1.6 `classify_serving_only` — Finding-1 asymmetry, still live

`backend/scripts/w3b/parity.py:504-515`:

```python
504	# ── Finding-1: serving-only reconciliation (<5-interaction-trade asymmetry) ───
505	def classify_serving_only(count: int | None) -> bool:
506	    """Is a serving-only survivor RECONCILED (the expected asymmetry) or a RED bug?
507
508	    Pure: returns ``True`` (reconcile — QL would have dropped this touch, leaving no
509	    cache row) iff the interaction-trade ``count`` is known and below QL's threshold
510	    (engine_decision.py:764, ``< 5``). ``count is None`` (no provider) or
511	    ``count >= 5`` -> ``False`` (RED: QL would have kept it, so a missing cache row is
512	    a real divergence).
513	    """
514
515	    return count is not None and count < _INTERACTION_MIN_TRADES
```

Predicate verbatim: **`return count is not None and count < _INTERACTION_MIN_TRADES`**, with
`_INTERACTION_MIN_TRADES = 5` (`parity.py:67`). Live on the diff path at `parity.py:396-400`, wired
by default at `parity.py:337-338`. Reconciled cases do not turn a day red (`parity.py:256-258`) and
surface in the report at `run_window.py:181-183`, `:192-193`. The counter
(`QLInteractionTradeCounter`, `parity.py:518-570`) re-reads the canonical SC stream directly via
`DatabentoParquetSource.for_trading_day` (`parity.py:549-556`) and counts trades in
`[touch, touch + interaction_window)` by bisect (`parity.py:566-570`). Per `parity.py:536`, this
counter path is **unexercised in production runs to date** ("none on the 51 evaluated days").

### A1.7 Watchdog env var

`backend/scripts/w3b/headless_replay.py:156-165`:

```python
156	#: _drive waits on the replay's background asyncio.Task rather than busy-polling
157	#: status(). The replay runs as ``HistoricalReplayService._task``; its state only
158	#: advances to a terminal value from inside that task. If the task finishes WITHOUT
159	#: a terminal state — e.g. killed by a BaseException the core's ``except Exception``
160	#: cannot catch — status() reports ``running`` forever, so a status()-only poll
161	#: spins indefinitely (the W3b 2025-12-18 wedge). Awaiting the task surfaces that
162	#: error; the watchdog bounds a genuine no-progress hang.
163	_DRIVE_POLL_SECONDS = 1.0
164	_DRIVE_WATCHDOG_SECONDS = float(os.environ.get("W3B_DRIVE_WATCHDOG_S", "120"))
165	_DRIVE_DEBUG = bool(os.environ.get("W3B_DRIVE_DEBUG"))
```

**`W3B_DRIVE_WATCHDOG_S`, default `"120"` seconds.** It is a NO-PROGRESS watchdog keyed on
`(status.state, status.events_processed)`, not a wall-clock cap (`headless_replay.py:205-224`); it
trips only when that key is unchanged for the interval and the state is non-terminal. The service
store-walk seed inside `replay.start` runs before the `_drive` loop and is therefore NOT
watchdog-bounded.

Other harness env vars: `W3B_THREAD_CAP` (`window.py:39-47`, pyarrow thread cap, opt-in, no
default), `W3B_QL_REPO` (`window.py:99`, default `C:\Users\gonza\Documents\Claude-Quant-Lab`),
`W3B_DRIVE_DEBUG` (`headless_replay.py:165`, unset).

## A2. The exact single-day gate command

The harness **does** support a single named day directly — `--days` takes a comma-separated subset
(`run_window.py:16-18` documents the one-day form). Argparse block verbatim, `run_window.py:214-235`:

```python
214	def _main() -> int:
215	    parser = argparse.ArgumentParser(description=__doc__)
216	    parser.add_argument("--days", help="comma-separated in-window day subset")
217	    parser.add_argument("--workers", type=int, default=1)
218	    parser.add_argument("--journal-base", type=Path, default=_DEFAULT_JOURNAL_BASE)
219	    parser.add_argument("--no-score", action="store_true", help="skip P2 offline scoring")
220	    parser.add_argument(
221	        "--no-resume",
222	        action="store_true",
223	        help="re-replay every day even if a complete journal already exists",
224	    )
225	    parser.add_argument("--report-out", type=Path, default=None)
226	    args = parser.parse_args()
227
228	    window = resolve_window()
229	    if args.days:
230	        days = [d.strip() for d in args.days.split(",") if d.strip()]
231	        unknown = [d for d in days if d not in window.window_dates]
232	        if unknown:
233	            raise SystemExit(f"--days not in D-036 window: {unknown}")
234	    else:
235	        days = list(window.window_dates)
```

`w3b` must be importable before `-m` resolves it (`run_window.py:31-33` self-inserts
`backend/scripts` only after import begins), so `PYTHONPATH` is mandatory. From the TL repo root
`C:\Users\gonza\Documents\Trade-Lab`:

PowerShell:
```
$env:PYTHONPATH="backend\scripts"; python -m w3b.run_window --days 2026-02-13 --report-out backend\data\w3b_report.txt
```
Git Bash:
```
PYTHONPATH=backend/scripts python -m w3b.run_window --days 2026-02-13 --report-out backend/data/w3b_report.txt
```

Historically-used form (from `backend/`, `REPORT.md:120-123`):
```bash
cd backend
PYTHONPATH=scripts python -m w3b.run_window --workers 8 --report-out data/w3b_report.txt
```

Replay-only single day, no diff (`headless_replay.py:320-337`):
```
PYTHONPATH=backend/scripts python -m w3b.headless_replay 2026-02-13
```

Exit code `0` iff HARD-GREEN else `1` (`run_window.py:258`). For any D in
2026-02-23..2026-06-11 both CLIs terminate at argv-parse time with `SystemExit` — see A4.1.

## A3. What must exist on disk before that command can run for day D

Roots resolved at `window.py:143-162`; `ns.data_dir` defaults to QL
`scripts/ml_training_tab.py:32` `_DEFAULT_DATA_DIR` ⇒ `DATA_DIR = QL\data\databento`,
`SYMBOL_DIR = DATA_DIR\NQ` (`window.py:113-115`).

| # | Required path pattern | Purpose | Code that reads it |
|---|---|---|---|
| 1 | `<QL>\data\databento\NQ\<D>\{mbp10.parquet \| mbp1.parquet \| trades.parquet}` (first hit in that priority) | The day's market-data stream for the serving replay | `strategy_core/data/databento_parquet.py:346-353` via `for_trading_day`; `:273-277` raises `FileNotFoundError` if absent |
| 2 | `<QL>\data\databento\NQ\<D−1 calendar>\{...}.parquet`, **schema-matching** to (1) | Supplies `[D−1 18:00 ET, D 00:00 UTC)`. Optional — absence or unmatchable schema degrades to single-file + `MISSING_PRIOR_DAY_FILE` | `databento_parquet.py:284-335` |
| 3 | `<QL>\data\databento\NQ\<D>\ml_utility_7850272e.parquet` | **QL per-day training cache, tag `7850272e`.** Absence is legal: day becomes a "thin day" gated on serving-survivors == 0 | `window.py:117-121` (`cache_path`/`has_cache`), `parity.py:211-215` `load_training_touches` |
| 4 | `<QL>\data\databento\NQ\<window day>\{...}.parquet` for the QL seed walk over `window_dates` (each needs its own dir AND its calendar-prior dir) | `seed_for_day` walks backward through `window_dates` calling QL `_get_session_hl_for_date` → DuckDB `TickStore` bars over `[prev 18:00 ET, D 18:00 ET)` | `window.py:187-193`; QL `dashboard_utility_builder.py:249`, `:285-309`; `tick_store.py:846-857` (`_TICK_FILENAMES`, `tick_store.py:38`) |
| 5 | ≥1 dated dir `<QL>\data\databento\NQ\YYYY-MM-DD\` strictly `< D`, within the 10 most recent, holding a recognized day file with ≥1 in-window trade | TL service SEED store-walk (runs on every day-mode replay). Miss → warning only | `services/replay.py:39`, `:233-247`; `strategy_core/data/prior_day.py:74-112` |
| 6 | `<QL>\models\NQ_W3_20260617T220752Z\model.cbm` | P2 offline CatBoost scoring | `parity.py:587-592` `OfflineScorer.for_bundle`; `window.models_path = QL_REPO/"models"` (`window.py:161`) |
| 7 | `<QL>\models\NQ_W3_20260617T220752Z\{strategy.json, metadata.json, model.cbm, model.cbm.sha256}` | Serving-side activation `registry.activate(bundle_id)` (`headless_replay.py:280`) | `model_registry.py:139-142`; checksum verify `model_registry.py:534` |
| 8 | `<QL>\src`, `<QL>\scripts` importable | `window.py:124-128` `_ensure_ql_on_path`; `window.py:141` imports `run_dashboard_session_experiment`, pulling in streamlit/plotly/numpy/pandas (`ml_training_tab.py:21-24`) | |
| 9 | writable `backend\data\w3b_journal\<D>\` | Journal scratch root, cleared per run (`headless_replay.py:270-272`) | |

**Observed disk state for the target era (directory listings only):**

- `NQ\2026-02-23\` … `NQ\2026-06-10\` each contain exactly `mbp1.parquet` — MBP-1 only.
- Exactly **one** day in 2026-02-23..2026-06-11 already carries `ml_utility_7850272e.parquet`:
  **2026-03-02**. No other day in that range has any `ml_utility_*` file.
- Era layout: days ≤ 2026-01-09 have `mbp10.parquet` only; 2026-01-11 onward have **both**;
  2026-02-23 onward have `mbp1.parquet` only. Because `_resolve_day_file` prefers `mbp10.parquet`
  (`databento_parquet.py:75-79`), **every day of the D-036 window resolved to mbp-10** — the harness
  has never executed the mbp-1 decode branch.

**NOT OBTAINABLE READ-ONLY:** whether the existing `2026-03-02\ml_utility_7850272e.parquet` was
built with the correct D-036-anchored rolling `prev_full_hl` seed. QL stamps the entering seed into
parquet metadata and re-checks it (`dashboard_utility_builder.py:161`), but reading that stamp
requires opening the file and reproducing the carry requires running the builder.

## A4. BUNDLE_ID coupling — which days can be gated

### A4.1 There IS a hard day-admissibility check, but it keys on the WINDOW, not the bundle

Two identical `SystemExit` gates:

```python
229	    if args.days:                                       # run_window.py:229-233
230	        days = [d.strip() for d in args.days.split(",") if d.strip()]
231	        unknown = [d for d in days if d not in window.window_dates]
232	        if unknown:
233	            raise SystemExit(f"--days not in D-036 window: {unknown}")
```
```python
334	    if args.date not in window.window_dates:            # headless_replay.py:334-337
335	        raise SystemExit(
336	            f"{args.date} is not in the D-036 window ({len(window.window_dates)} days)"
337	        )
```

`window_dates` = `_date_slice(available, "2025-11-21", "2026-02-13")` (`window.py:154` +
`_EXP_ARGV` `window.py:79-80`), a pure string range filter (QL
`run_dashboard_session_experiment.py:48-53`). **Every day in 2026-02-23..2026-06-11 is rejected at
argv-parse time by both CLIs.** This is the single blocking gate, and it is CLI-only:
`run_window.run_window(days=[...])` called programmatically bypasses it (`run_window.py:83-84`).

### A4.2 A second, soft window coupling — the seed silently degrades

`window.py:181-193`:

```python
181	    window = resolve_window()
182	    util_cfg = DashboardUtilityConfig(**window.util_kwargs)
183	    try:
184	        idx = window.window_dates.index(date_str)
185	    except ValueError:
186	        return None
187	    for k in range(idx - 1, -1, -1):
188	        hl = _get_session_hl_for_date(
189	            window.data_dir, window.symbol, window.window_dates[k], util_cfg, None
190	        )
191	        if hl is not None:
192	            return hl
193	    return None
```

For an out-of-window day this returns `None` without raising, so W3b `_seed_fn` no-ops and PDH/PDL
comes solely from the TL service store-walk — a different seed source with a documented divergence
caveat (`strategy_core/data/prior_day.py:18-24`).

### A4.3 Bundle-contract checks that exist — all DAY-INDEPENDENT

`registry.activate(bundle_id)` (`headless_replay.py:280`) runs the fail-closed
`serving_compatibility_error` (`backend/src/trade_lab/services/model_registry.py:65-137`):
feature-set subset (`:79-84`), retention ≥ approach+interaction windows (`:85-92`), instrument
(`:93-97`), decision offset vs observation duration (`:98-104`), `barrier_mode == "fixed_points"`
(`:105-109`), session-scheme structural equality (`:110-114`), bar-type (`:115-123`), and
live/replay schema subsets (`:124-136`). **None is a function of the trading day.** Answering the
three named sub-questions:

- **Training window:** `QL\models\NQ_W3_20260617T220752Z\strategy.json` carries **no training date
  range at all** — its `provenance` block is only `dataset_config_hash` + catboost params. There is
  nothing for a check to compare a day against. **No such check exists.**
- **`label_policy` forward cutoff:** the contract declares `"forward_cutoff":
  "17:00_US/Eastern_ny_close"`, but `grep -n "forward_cutoff" backend/src/trade_lab` returns **zero
  hits** — TL never reads that key. **No forward-cutoff-vs-day check exists.**
- **Session scheme:** checked as contract-vs-runtime structural equality
  (`model_registry.py:110-114`), never against the day. **No such day check exists.**

The only bundle-derived value in the parity math is the model binary (`parity.py:587-592`), plus
`CLASS_MAP` (`window.py:66-70`), `CONTRACT_FEATURES` (`window.py:57-63`) and the eligibility
constants (`parity.py:58-60`) — all hardcoded to mirror `strategy.json`, none day-dependent.

### A4.4 The cache-tag consistency guard

`window.py:145-150` raises `RuntimeError` if QL's resolved `dataset_config_hash()` ≠ `"7850272e"`:

```python
145	    cache_tag = config.dataset_config_hash()
146	    if cache_tag != EXPECTED_CACHE_TAG:
147	        raise RuntimeError(
148	            f"resolved cache tag {cache_tag!r} != expected {EXPECTED_CACHE_TAG!r}; "
149	            "the QL config drifted from the W3 bundle's training inputs"
150	        )
```

This constrains the CONFIG, not the DAY. `--start`/`--end` are not hash inputs (see B1), so a
`ml_utility_7850272e.parquet` for any day — in or out of the D-036 window — is legal. That is why
2026-03-02 can already carry one.

**A4 summary:** the harness performs **no bundle-contract-derived day check whatsoever**. The only
thing rejecting an arbitrary day is the hardcoded `_EXP_ARGV` `--start 2025-11-21 / --end
2026-02-13` slice materialised as `window.window_dates`, enforced at `run_window.py:231-233` and
`headless_replay.py:334-337`, plus the silent `None` seed at `window.py:184-186`.

## A5. Does the harness path depend on QL `replay_client.py`? — NO

### A5.1 Where `replay_client` appears in QL

Repo-wide grep returns exactly 7 hits in 3 files, all on the dashboard surface:

```
Claude-Quant-Lab\scripts\run_replay.py:34:from alpha_lab.dashboard.pipeline.replay_client import ReplayClient
Claude-Quant-Lab\scripts\run_replay.py:59:    replay_client = ReplayClient(
Claude-Quant-Lab\scripts\run_replay.py:65:    pipeline = PipelineService(settings, client=replay_client)
Claude-Quant-Lab\scripts\run_replay.py:130:        replay_client._bar_complete_event.set()
Claude-Quant-Lab\scripts\run_replay.py:464:    replay_client.on_day_boundary(_on_day_boundary)
Claude-Quant-Lab\src\alpha_lab\dashboard\api\server.py:778:            from alpha_lab.dashboard.pipeline.replay_client import ReplayClient
Claude-Quant-Lab\src\alpha_lab\dashboard\api\routes\data.py:204:    from alpha_lab.dashboard.pipeline.replay_client import ReplayClient
```

Corroborating negative: `grep -n "alpha_lab\.dashboard"` over the entire
`Claude-Quant-Lab\src\alpha_lab\agents\` subtree — where every QL module the harness touches lives —
returns **no matches**. The `agents` layer never imports the `dashboard` layer.

### A5.2 Import graph from the w3b entrypoint down to every market-data leaf

Entrypoint `python -m w3b.run_window` → `run_window.py:35-36`:
`from w3b.parity import DayDiff, OfflineScorer, diff_day`; `from w3b.window import resolve_window`.

- **Branch A — `w3b.parity`** (`parity.py:31-50`: stdlib + pandas + `w3b.window`). Lazy inside
  methods: `parity.py:549-550` `from strategy_core.data.databento_parquet import
  DatabentoParquetSource` → **market-data read #1** (`parity.py:552-556`); `parity.py:582`
  `from catboost import CatBoostClassifier` (no data).
- **Branch B — `w3b.window`** (stdlib + optional pyarrow). Lazy: `window.py:141`
  `import run_dashboard_session_experiment as exp` → QL
  `run_dashboard_session_experiment.py:22-28` → `ml_training_tab.py:13-24` (hashlib/json/logging/
  numpy/pandas/plotly/streamlit; **no module-level `alpha_lab.*`**, every `alpha_lab` import there is
  function-local and none names `dashboard`) → `run_dashboard_session_experiment.py:30-39`
  → `alpha_lab.agents.data_infra.ml.{config,dashboard_utility_builder}` →
  `dashboard_utility_builder.py:19-33` → `alpha_lab.agents.data_infra.tick_store.TickStore` →
  `tick_store.py:18-29` (duckdb/pandas/pyarrow, terminal). Data touched: `exp.get_available_dates`
  (`window.py:153`) is a pure `iterdir()` existence scan (`ml_training_tab.py:52-61`, `_TICK_FILENAMES`
  at `:49`) — **listing only**; and `_get_session_hl_for_date` → `TickStore` → **market-data read #2
  (DuckDB)**.
- **Branch C — `w3b.headless_replay`** (lazy at `run_window.py:52`; module-level imports
  `headless_replay.py:36-46`) → `trade_lab.adapters.historical_parquet` (`historical_parquet.py:20-23`
  imports SC `DatabentoParquetSource`; day-mode call at `:74-83`) → **market-data read #3 (the replay
  stream)**; → `trade_lab.services.replay` (`replay.py:13` `from strategy_core.data.prior_day import
  prior_full_day_extremes` → `prior_day.py:39-40` → SC source) → **market-data read #4 (SEED walk)**.

**Every leaf that reads market data:** (1) SC `DatabentoParquetSource` — replay stream, SEED walk,
and `parity.QLInteractionTradeCounter`; (2) QL `alpha_lab.agents.data_infra.tick_store.TickStore`
(DuckDB) for the `prev_full_hl` seed. `alpha_lab.dashboard.pipeline.replay_client` appears on none
of these hops. **The mbp10-hardcoded debt named in the INGEST close record sits entirely on the QL
dashboard replay surface, NOT on the W3b harness path.**

### A5.3 Both readers on the W3b path are mbp-1 capable

- SC reader: `DAY_FILE_PRIORITY` includes mbp-1 (`databento_parquet.py:75-79`); schema dispatch
  `databento_parquet.py:516-519` (`is_tob = schema in {"mbp-1", "cmbp-1", "bbo", "cbbo", "tbbo"}`);
  D-P-17 makes action-bearing TOB schemas trade-bearing (`databento_parquet.py:862-866`);
  era-boundary prior-day fallback at `:286-315`.
- QL TickStore: `_TICK_FILENAMES = ["mbp10.parquet", "mbp1.parquet", "trades.parquet"]`
  (`tick_store.py:38`) and variable-depth handling (`tick_store.py:338-346`).
- TL replay catalog: `SUPPORTED_SCHEMAS = ("trades", "mbp-1", "mbp-10", "bbo")`
  (`replay_catalog.py:25`); the contract declares `"replay_schemas": ["trades", "mbp-1", "mbp-10"]`,
  so activation check #8 passes.

The one remaining mbp-10 literal on the harness path is self-documented as cosmetic —
`headless_replay.py:57-60` `_REPLAY_SCHEMA = "mbp-10"`. In the day-mode branch it is never passed to
the SC source (`historical_parquet.py:74-83`); the SC source derives `day_schema` from the resolved
filename (`databento_parquet.py:278`, `:330`, `:339`). Its surviving effect is the
`source_schema=schema` stamp on emitted events (`historical_parquet.py:111`, `:121`) and the
feed-status label (`replay.py:267`). **NOT OBTAINABLE READ-ONLY:** whether that stale
`source_schema` stamp is consumed by anything correctness-bearing downstream — settling it requires
enumerating every consumer of `MarketEvent.source_schema` through
`StrategyCoreService._last_schema` (`strategy_core_service.py:163`, `:166`) and the runtime feed/
journal surfaces.

---

# PART B — cache tag composition and per-day build cost

## B1. `dataset_config_hash` — location, inputs in order, range-independence

Location: `QL\src\alpha_lab\agents\data_infra\ml\config.py:384` (`MLPipelineConfig.dataset_config_hash`).
Body, `config.py:404-424`:

```python
404	        import hashlib
405	        import json
406	        from strategy_core import PLATFORM_VERSION
407	        from strategy_core.constants import BAR_PRICE_SOURCE, LABEL_ENTRY_REFERENCE
408
409	        payload = (
409	            f"mode={self.training_mode}|"
410	            + json.dumps(self.extrema.model_dump(), sort_keys=True)
411	            + json.dumps(self.labeling.model_dump(), sort_keys=True)
412	            + json.dumps(self.features.model_dump(), sort_keys=True)
413	            + json.dumps(self.dashboard_utility.model_dump(), sort_keys=True)
414	            + f"|tick_size={self.tick_size}"
415	            + f"|bar_price_source={BAR_PRICE_SOURCE}"
416	            + f"|label_entry_reference={LABEL_ENTRY_REFERENCE}"
417	            + f"|platform_version={PLATFORM_VERSION}"
422	            + "|decision_pipeline=sc_runtime_stream_v1"
423	        )
424	        return hashlib.sha256(payload.encode()).hexdigest()[:8]
```

Every folded input, in payload order, with its D-036 value (values resolved by
`run_dashboard_session_experiment.py::_resolve_config`, `:127-152`, driven by the ratified argv
frozen at `scripts/w3_cache_warmer.py:77-98`):

| # | Payload fragment | Source | D-036 value |
|---|---|---|---|
| 1 | `mode={training_mode}` + pipe | `config.py:409`; `run_dashboard_session_experiment.py:129` | `mode=dashboard_utility` |
| 2 | `extrema.model_dump()` | `config.py:410`; `ExtremaConfig` `config.py:61-88`, all defaults | `dedup_window=200`, `max_peak_width=5000`, `min_peak_width=200`, `min_prominence_ticks=10.0`, `window_size=5000` |
| 3 | `labeling.model_dump()` | `config.py:411`; `LabelingConfig` `config.py:91-107`, all defaults | `crossing_threshold=20`, `forward_window=5000`, `rebound_thresholds=[20, 40, 60]` |
| 4 | `features.model_dump()` | `config.py:412`; `FeatureConfig` `config.py:110-139`, defaults | `include_signal_features=True`, `ms_window=237`, `pl_range_ticks=10`, `rsi_periods=[20,40,80,120,160,200]`, `signal_bar_timeframe="5m"` |
| 5 | `dashboard_utility.model_dump()` | `config.py:413`; `DashboardUtilityConfig` `config.py:305-347`; populated `run_dashboard_session_experiment.py:130-137` | `approach_window_minutes=15`, `bar_type="147t"`, `include_approach_features=True`, `interaction_window_minutes=5`, `level_proximity_pts=0.50`, `sl_points=15.0`, `tp_points=15.0`, `trap_mfe_min=5.0` |
| 6 | `tick_size={tick_size}` | `config.py:414`; `run_dashboard_session_experiment.py:150` | `tick_size=0.25` |
| 7 | `bar_price_source={BAR_PRICE_SOURCE}` | `config.py:415`; SC `constants.py:36` | `bar_price_source=trade_price` |
| 8 | `label_entry_reference={LABEL_ENTRY_REFERENCE}` | `config.py:416`; SC `constants.py:259` | `label_entry_reference=realistic_at_decision` |
| 9 | `platform_version={PLATFORM_VERSION}` | `config.py:417`; SC `__init__.py:61` | `platform_version=strategy_core_platform_v1` |
| 10 | `decision_pipeline=sc_runtime_stream_v1` (literal) | `config.py:422` | same |

Result `sha256(payload)[:8]` = **`7850272e`**, confirmed in logs:

- `QL\W3A_WARM.log:1` — `# 2026-06-13T02:42:07.808372+00:00 W3 cache warmer | tag=7850272e | symbol=NQ | window=73d | targets=2 | workers=2 | pid=59748`
- `QL\_p4b_run.out:1` — `cache_tag=7850272e  (D-036 expectation: 7850272e)`

Present on `MLPipelineConfig` but **NOT folded**: `walk_forward` (`config.py:361`), `model`
(`config.py:362`), `session_experiment` (`config.py:366-372`, whose docstring at `:213-219` states
"Changing session scope should not invalidate expensive `ml_utility_*` caches"), and `instrument`
(`config.py:379-382`, used in the cache PATH, not the tag).

**Does the hash fold the dataset date range? NO — the hash is RANGE-INDEPENDENT.** There is no
`start`, `end`, `dates`, or date-derived term anywhere in `config.py:408-423`. `MLPipelineConfig`
carries no date fields; the window arrives separately as `--start`/`--end`
(`run_dashboard_session_experiment.py:66-67`), is materialised into a list of day strings
(`:158-159`) and passed as the first positional arg to `build_utility_dataset(dates, ...)` (`:194`) —
never into the object the hash is computed from. Corroborated by QL `docs/DECISIONS.md:327` (D-038):
"the cache tag hashes config, not row schema, so caches are deliberately not invalidated."

**D-036 verbatim** — QL `docs/DECISIONS.md:306-311`, key clauses: window `2025-11-21 → 2026-02-13
inclusive, valid store days only`; `training_mode=dashboard_utility`; preset `all_to_ny`;
`bar_type=147t`; NQ tick 0.25; folds `TRAIN=40 / TEST=5 / STEP=5 / PURGE=2, MIN_TRAIN_EVENTS=30`;
label policy `tp_points=15.0`, `sl_points=15.0` (override, config default 30.0), `trap_mfe_min=5.0`,
`interaction_window_minutes=5`, `approach_window_minutes=15` (override, default 90),
`include_approach_features=True` (override, default False); `entry_reference`/`forward_cutoff`/
flatten ride the SC constants (`realistic_at_decision` / `17:00_US/Eastern_ny_close` / 16:40 ET);
features PINNED `[int_time_within_2pts, int_absorption_ratio, app_avg_trade_size,
app_large_trade_vol_pct, app_max_spread]`, `rfecv_enabled=False`; CatBoost 1000 / depth 6 / lr 0.03 /
Balanced / seed 42 MultiClass; model quality NOT a save gate. The same entry records the W3a
stop-gate: "1447.1s for 2026-02-12 (fresh, config hash 7850272e) → ×60 ≈ 24.1 h".

**D-P-13 verbatim** — the PROGRESS ledger lives in SC `docs/DECISIONS.md:96-103`:

```
- D-P-13 RATIFIED (2026-06-12) — W3 proof-bundle training configuration: the full
  explicit config recorded verbatim as Quant-Lab docs/DECISIONS.md D-036 (window
  2025-11-21→2026-02-13, purged 40/5/5/2 trading-day folds, pinned 5-feature set,
  v3 label policy with explicit sl=15/aw=15 overrides, CatBoost defaults, quality
  not a save gate). D-037 (same file) supersedes the stub-exclusion pre-ruling:
  app_max_spread RIDES the W3b gate; the quotes_in_window stub is SC-plugin-path-only.
```

## B2. Extending the window forward — reuse or invalidate?

**FACT: the existing `ml_utility_7850272e` per-day caches would be REUSED for the days they already
cover, NOT invalidated.** Three code facts, in decision order:

(a) **Cache key folds only the config tag and the day.** `dashboard_utility_builder.py:142`, `:155`:

```python
142	    cache_tag = config.dataset_config_hash()
155	        cache_path = data_dir / symbol / date_str / f"ml_utility_{cache_tag}.parquet"
```

`date_str` comes from `for i, date_str in enumerate(sorted(dates)):` (`:151`). Moving `--end` forward
changes only the membership of `dates` — neither `cache_tag` nor any already-covered day's
`cache_path`.

(b) **The window-to-date-list seam appends.** `run_dashboard_session_experiment.py:48-53`:

```python
48	def _date_slice(available: list[str], start: str | None, end: str | None) -> list[str]:
49	    if not available:
50	        return []
51	    start_value = start or available[0]
52	    end_value = end or available[-1]
53	    return [d for d in available if start_value <= d <= end_value]
```

With `--start` unchanged, a later `--end` yields a pure superset in the same order.

(c) **The hit/miss branch is a per-day `exists()` + seed-stamp check.**
`dashboard_utility_builder.py:157-178`:

```python
157	        # Trust an existing cache ONLY if it was built with the same seed entering
158	        # this day. A seedless/wrong-seed cache (e.g. a standalone timing build
159	        # with prev_full_hl=None on a day that HAS a prior window day) is silently
160	        # missing PDH/PDL touches — rebuild it instead of trusting it.
161	        if cache_path.exists() and _cache_seed_matches(cache_path, prev_full_hl):
162	            df = pd.read_parquet(cache_path)
163	            cached_count += 1
164	            if not df.empty:
165	                frames.append(df)
166	            prev_full_hl = _get_session_hl_for_date(
167	                data_dir, symbol, date_str, util_cfg, prev_full_hl
168	            )
169	            continue
170	        if cache_path.exists():
171	            logger.warning(
172	                "Rebuilding %s cache for %s: seed stamp does not match prev_full_hl=%s "
173	                "(stale/seedless cache guard)",
```

Both hit conditions hold under a forward extension: `prev_full_hl` is carried from the first sorted
day (`:149` `prev_full_hl: tuple[float, float] | None = None`), and days appended AFTER 2026-02-13
cannot alter the seed entering any day at or before it.

**Asymmetry, stated as fact:** this holds for extending FORWARD only. Moving `--start` EARLIER would
change `prev_full_hl` entering the old first day (currently `None` for 2025-11-21), so
`_cache_seed_matches` returns `False` there and that day rebuilds.

Disk state (listing only): 60 `ml_utility_7850272e.parquet` files exist — 59 in
2025-11-21..2026-02-13 plus `2026-03-02`. Under a forward extension to 2026-06-11 all 60 are reused;
every other day in 2026-02-16..2026-06-11 is a MISS and would build fresh. Corroborated by
`docs\DECISIONS.md:327`.

## B3. `entering_seed` trust guard

**Provenance.** `git show 098e354` (QL):

```
commit 098e354ff1f878e6f3d2e9e16138c9143d990ebf
Date:   Wed Jun 17 03:58:38 2026 -0500

    fix(ml): guard build_utility_dataset against seedless/wrong-seed day caches
    ... Stamp the prev_full_hl seed into per-day cache parquet metadata and verify it on
    the trust-existing-cache path (builder + warmer); rebuild on mismatch or on a
    legacy/unstamped cache when a seed is expected.

 scripts/w3_cache_warmer.py                         | 16 +++--
 .../data_infra/ml/dashboard_utility_builder.py     | 80 +++++++++++++++++++++-
 tests/agents/test_cache_seed_guard.py              | 62 +++++++++++++++++
```

The SEED-window refinement is QL `820b532` (Mon Jul 6 19:07:19 2026), "fix(ml): stamp day caches with
the seed ENTERING the day (SEED P3)", whose diff replaces `_write_day_cache(df, cache_path,
prev_full_hl)` with `entering_seed = prev_full_hl` captured BEFORE the carry reassignment, then
`_write_day_cache(df, cache_path, entering_seed)`.

**The guard verbatim** — `dashboard_utility_builder.py:88-114`:

```python
88	def _cache_seed_matches(
89	    cache_path: Path, expected_seed: tuple[float, float] | None
90	) -> bool:
91	    """True iff the cache was stamped with a seed equal to ``expected_seed``.
92
93	    An UNSTAMPED (legacy) or ``none``-stamped cache matches ONLY when no seed is
94	    expected (``expected_seed is None``). So when a prior window day exists (the
95	    seed is non-None), a seedless/legacy cache is NOT trusted and is rebuilt —
96	    exactly the 2026-02-12 stale-cache case.
97	    """
98	    import pyarrow.parquet as pq
99
100	    try:
101	        md = pq.read_metadata(cache_path).metadata or {}
102	    except Exception:
103	        return False
104	    stamp = md.get(_SEED_META_KEY)
105	    if stamp is None or stamp == b"none":
106	        return expected_seed is None
107	    try:
108	        high, low = (float(x) for x in stamp.decode().split(","))
109	    except Exception:
110	        return False
111	    return expected_seed is not None and (high, low) == (
112	        float(expected_seed[0]),
113	        float(expected_seed[1]),
114	    )
```

with `_SEED_META_KEY = b"ml_utility_prev_full_hl"` (`:54`), `_seed_meta_bytes` (`:57-61`, `None`
serialises to the literal `b"none"`), `_write_day_cache` (`:64-78`), and the entering-seed capture at
`:180-200`:

```python
180	        # Build fresh for this date. Capture the seed ENTERING this day: the cache must
181	        # be stamped with the seed it was BUILT with (what the :161 trust check compares
182	        # against on the next run, and the warmer's stamp convention) — stamping the
183	        # post-update carry (this day's own H/L) made every builder-written cache
184	        # self-invalidate on the next run (SEED_PARITY_RECON §3(d) rebuild churn).
185	        entering_seed = prev_full_hl
200	            _write_day_cache(df, cache_path, entering_seed)
```

The warmer mirrors the convention (`scripts/w3_cache_warmer.py:222-241`), deleting a mismatched cache
(`:232` `_quiet_unlink(cache_path)`) before rebuilding.

**(a) A cache carrying NO `entering_seed` stamp at all:** `stamp is None` and the guard returns
`expected_seed is None`. So a legacy/unstamped cache is trusted ONLY on a day whose entering seed is
legitimately `None`. On any day with a non-`None` entering seed it is NOT trusted: the builder logs
`"Rebuilding %s cache for %s: seed stamp does not match prev_full_hl=%s (stale/seedless cache
guard)"` (`:171-178`) and rebuilds, overwriting with a stamped cache (`:200`); the warmer unlinks and
rebuilds. Pinned by `tests/agents/test_cache_seed_guard.py:49-57`
(`test_legacy_unstamped_cache_not_trusted_when_seed_expected`).

**(b) A window's FIRST day where the seed is legitimately `None`:** the builder enters the loop with
`prev_full_hl = None` (`:149`), so `expected_seed is None`. On WRITE, `_seed_meta_bytes(None)` returns
`b"none"` (`:60-61`) — the day IS stamped, with the sentinel, not left unstamped. On READ,
`stamp == b"none"` gives `return expected_seed is None` which is `True`: the cache is TRUSTED, no
rebuild. Symmetrically that same file is NOT trusted if the date later appears with a non-`None`
entering seed. Pinned by `test_cache_seed_guard.py:40-46` and end-to-end at `:94-108`
(`assert stamp["2026-01-05"] == b"none"` / `assert stamp["2026-01-06"] == b"100.0,90.0"`, with the
second run taking the cache-hit path). Missing/corrupt files return `False` (`:100-103`), pinned at
`:60-65`.

## B4. D-038 `entry_price` population

**Writer** — `QL\scripts\ml_training_tab.py:1237` (`build_oos_predictions_frame`); canonical schema at
`:1252-1268` with `entry_price` at `:1266`; degradation rule `:1283-1294`:

```python
1283	        # PROP-SIM P1: per-row outcome passthroughs (NaN when the training frame
1284	        # lacked the column — pre-P1 folds, warm caches without entry_price).
1285	        def _outcome_field(name: str, *, fold_data=fold_data, n=n) -> np.ndarray:
1286	            vals = fold_data.get(name)
1287	            if vals is None:
1288	                return np.full(n, np.nan)
1289	            arr = np.asarray(vals, dtype=float).flatten()
1290	            return arr if len(arr) == n else np.full(n, np.nan)
1292	        max_mfe_vals = _outcome_field("max_mfe")
1293	        max_mae_vals = _outcome_field("max_mae")
1294	        entry_price_vals = _outcome_field("entry_price")
```

emitted per row at `:1327-1330` (`"entry_price": float(entry_price_vals[i]),`).

**REUSED pre-D-038 cache yields NaN. CONFIRMED.** Origin of the `None`, `ml_training_tab.py:314-326`:

```python
314	    # PROP-SIM P1: per-row outcome passthroughs for the OOS writer, threaded from
315	    # the training frame (the labeler computes them; never recomputed here).
317	    # Frames/caches predating a column (e.g. warm D-036 caches lack entry_price)
318	    # yield None -> the writer degrades that column to NaN.
319	    outcome_passthrough: dict[str, np.ndarray | None] = {
320	        col: (
321	            pd.to_numeric(valid[col], errors="coerce").to_numpy(dtype=float)
322	            if col in valid.columns
323	            else None
324	        )
325	        for col in ("max_mfe", "max_mae", "entry_price")
326	    }
```

`valid` is the frame returned by `build_utility_dataset`, which for a reused day is literally
`pd.read_parquet(cache_path)` (`dashboard_utility_builder.py:162`). No `entry_price` column gives
`None`, threaded per fold at `ml_training_tab.py:534-538`, so `fold_data.get("entry_price")` is
`None`, so `np.full(n, np.nan)` — NaN for every OOS row. Stated in `docs\DECISIONS.md:327`.

Disk evidence the D-036 fleet is pre-D-038: the D-038 writer commit is `86654c7`
(`Fri Jul 10 18:30:41 2026`); all 59 in-window caches carry `Jun 17` mtimes, e.g.
`-rw-r--r-- 1 gonza 197609 12919 Jun 17 15:59 data/databento/NQ/2026-02-13/ml_utility_7850272e.parquet`,
while the one fresh day is `-rw-r--r-- 1 gonza 197609 13559 Jul 11 03:00
data/databento/NQ/2026-03-02/ml_utility_7850272e.parquet`.

**Freshly built post-D-038 is populated. CONFIRMED.** Row writer,
`QL\src\alpha_lab\agents\data_infra\ml\engine_decision.py:786-808`:

```python
786	        # PROP-SIM P1: the honest decision-time fill the excursions were anchored
787	        # on — the SAME injected accessor at the SAME instant the engine used
788	        # (strategy_core honest_entry: entry_price = trade_price_at(decision_ts));
789	        # non-None here because a None fill would have dropped (no_fill) above.
790	        entry_price = _price_at(touch.bar_ts_utc + timedelta(minutes=decision_offset))
808	            "entry_price": None if entry_price is None else float(entry_price),
```

That frame is written by `_write_day_cache` (`dashboard_utility_builder.py:200`;
`w3_cache_warmer.py:240`), so `outcome_passthrough["entry_price"]` is a real array and
`float(entry_price_vals[i])` is populated. The one such day on disk (2026-03-02, built after the D-038
commit) reports `QL\_p4b_run.out:24` — `total null cells: 0`.

**Save path** — `ml_training_tab.py:1710-1722`:

```python
1710	        # W2 P3a (F27): the OOS writer is UNCONDITIONAL. An empty frame still
1711	        # writes the parquet WITH its schema; a missing/illtyped frame or a
1712	        # failed write fails the save loudly (no silent skip, no partial bundle).
1713	        oos_predictions = training_result.get("oos_predictions")
1714	        if not isinstance(oos_predictions, pd.DataFrame):
1720	        oos_predictions.to_parquet(output_dir / "oos_predictions.parquet", index=False)
1721	        eval_dict["oos_predictions_file"] = "oos_predictions.parquet"
1722	        eval_dict["oos_predictions_rows"] = int(len(oos_predictions))
```

frame produced at `ml_training_tab.py:620`.

Caveat, stated as fact: the check at `:322` is on `valid.columns` of ONE concatenated frame. A mixed
fleet (some days pre-D-038, some post) unions columns under `pd.concat`, so `entry_price` is NaN for
the reused days and real for the fresh ones, rather than absent for the whole frame.

## B5. Build cost — read from logs only

**The one MBP-1 fresh day already built (2026-03-02).** Source `QL\_p4b_run.out`, verbatim lines:

```
_p4b_run.out:1   cache_tag=7850272e  (D-036 expectation: 7850272e)
_p4b_run.out:2   window: 2025-11-21..2026-03-02 (87 days)
_p4b_run.out:3   prev_full_hl seed entering 2026-03-02: (25098.75, 24782.0)
_p4b_run.out:5    "day": "2026-03-02",
_p4b_run.out:6    "rows": 5,
_p4b_run.out:7    "seconds": 146.56808780002757,
_p4b_run.out:9    "status": "OK",
_p4b_run.out:10   "peak_rss_gb": 3.004223488,
_p4b_run.out:14  rows=5  wall=147s
_p4b_run.out:24  total null cells: 0
```

That is **146.6 s worker-measured / 147 s wall, peak RSS 3.00 GB, tag 7850272e, 5 rows**. Peak RSS is
measured via `GetProcessMemoryInfo.PeakWorkingSetSize` (`w3_cache_warmer.py:132-144`). Provenance:
`QL\scratch_ingest_p4b.py:1-8` — "Reuses ``w3_cache_warmer.warm_one_day`` VERBATIM (the
seed-replicating worker) — only the window is widened so the fresh day and its seed predecessors are
in scope." Matching ingest line, `QL\INGEST.log:49` —
`2026-03-02 | rows=15,030,550 | 44.8s | OK (343.0 MB) | filtered 52,574 spread rows`.

**MBP-10-era warm figures for comparison.** `QL\W3A_WARM.log:162`:

```
# 2026-06-17T22:06:34.524636+00:00 DONE wall=24.7min counts={'EMPTY': 14, 'OK': 43} peak_rss_gb=6.26 (day=2026-02-05) | suggested_workers: floor(20GB / 6.26GB peak) = 3 -> min(4, max(2, 3)) = 3
```

Siblings from the same log: `:4` `DONE wall=2.3min counts={'OK': 2} peak_rss_gb=2.96`; `:76`
`DONE wall=29.9min counts={'EMPTY': 14, 'OK': 56} peak_rss_gb=6.36` (duplicated at
`W3A_WARM_launch.log:85`); `:79` `DONE wall=3.1min counts={'OK': 1} peak_rss_gb=3.48`. Per-day tail
line example, `W3A_WARM.log:161` — `2026-06-17T22:06:34.151086+00:00 day=2026-02-12 rows=5 secs=160.2
pid=38580 peak_rss_gb=5.43 status=OK [57/57]`. These wall figures are N-way parallel runs
(`--workers 4`, header `W3A_WARM.log:5`), not serial per-day cost; the serial figure of record is
`docs\DECISIONS.md:311` — 1447.1 s for 2026-02-12.

**Is 2026-03-02 thin or dense?** Day-dir listing only:

```
-rw-r--r-- 1 gonza 197609 359650790 Jul 11 02:31 mbp1.parquet
-rw-r--r-- 1 gonza 197609     13559 Jul 11 03:00 ml_utility_7850272e.parquet
```

359,650,790 B = **343.0 MB** (matching `INGEST.log:49`). No `mbp10.parquet` — MBP-10 coverage stops at
2026-02-22. Comparable `mbp1.parquet` sizes: 2026-02-23 324.9 MB, 02-24 269.6, 02-25 171.8, 02-26
280.4, 02-27 290.7, **2026-03-02 343.0**, 03-03 466.0, 03-04 375.8, 03-05 426.2, 03-06 418.5; Sunday
half-sessions 2026-03-01 10.0 MB and 2026-02-22 6.9 MB. Across all 156 store days carrying an
`mbp1.parquet`: mean 266.3 MB, min 3.5, max 586.7; restricted to full sessions (>100 MB, n=124):
min 171.4, p25 266.4, **median 308.7**, p75 386.9.

**CALL: 2026-03-02 is a DENSE day** — above the full-session median, below p75; an ordinary-to-heavy
full RTH session, roughly 13x the adjacent Sunday half-session and clearly not a holiday/thin day
(those cluster 4.8–25.6 MB). Its 15,030,550 ingest rows are the same order as the MBP-10-era
stop-gate day 2026-02-12 (`INGEST.log:31` — `rows=17,209,508 | 51.8s | OK (386.9 MB)`), so the
147 s / 3.0 GB figure is not a thin-day artifact. No parquet was opened for this determination.

**NOT OBTAINABLE READ-ONLY:** a SERIAL per-day D-036 cost on the MBP-1 store (as distinct from the
warmer's parallel worker seconds). The only serial number on record is 1447.1 s for the MBP-10-era
2026-02-12 (`docs\DECISIONS.md:311`); producing an MBP-1 serial figure would require executing
`run_dashboard_session_experiment.py` over an uncached MBP-1 day.

---

# PART C — propsim: excursion columns and the conservative column

Target dir `QL\src\alpha_lab\propsim\`: `__init__.py`, `__main__.py`, `bootstrap.py`, `cli.py`,
`engine.py`, `loaders.py`, `models.py`, `presets.py`, `report.py`. No parquet was opened and the
module was not executed; everything below is source text.

## C1. The OOS-parquet loader and its `points_conservative`

Loader: `load_oos_trades`, `QL\src\alpha_lab\propsim\loaders.py:326`:

```python
326	def load_oos_trades(
327	    parquet_path: Path,
328	    *,
329	    tp_points: float,
330	    sl_points: float,
331	    gate_column: str | None = None,
332	) -> LoadedTrades:
333	    """Trades from a bundle's OOS predictions parquet."""
```

The code that sets `points_conservative` — `loaders.py:363-393` (one local `points` feeds BOTH
columns):

```python
363	    for row in frame.itertuples(index=False):
364	        label = str(row.label)
365	        resolution = getattr(row, "resolution_type", None)
366	        if not isinstance(resolution, str) or not resolution:
367	            resolution = _RESOLUTION_TYPE_BY_LABEL.get(label)
368	        if resolution == "tp_hit":
369	            points = tp_points
370	        elif resolution == "sl_hit":
371	            points = -sl_points
372	        else:
373	            skipped += 1
374	            continue
383	        trades.append(
384	            TradePath(
385	                day=trading_day_for(entry_ts),
386	                entry_ts=entry_ts,
387	                points_optimistic=points,
388	                points_conservative=points,
389	                mfe_pts=mfe,
390	                mae_pts=mae,
391	                resolution=resolution,
392	            )
393	        )
```

**CONFIRMED: for the OOS source `points_conservative == points_optimistic` by construction** —
lines 387 and 388 pass the same local float. No tick, slippage, direction, or entry-price term
appears anywhere in `load_oos_trades`. Corroborating in-code statements: module docstring
`loaders.py:15-19` ("OOS fills are idealized: the conservative column EQUALS the optimistic one (no
synthetic slippage model is invented), which the notes state"); the emitted note `loaders.py:399-401`
("conservative column EQUALS optimistic (no synthetic slippage model)"); and the test
`QL\tests\propsim\test_loaders.py:283` — `assert first.points_conservative == 15.0  # idealized: EQUALS optimistic`.

## C2. `points_conservative` for the other two loaders, and the exact fill convention

### (a) executions ⋈ journal loader — `load_executions_trades` (`loaders.py:114`)

This loader does not COMPUTE a conservative column; it reads the value verbatim off the tracker's
`close` row, `loaders.py:156-173`:

```python
156	    for row in deduped_closes.values():
157	        entry_ts = _parse_ts(row.get("entry_ts_utc")) or _parse_ts(row.get("ts_utc"))
158	        points = _finite(row.get("points"))
159	        points_cons = _finite(row.get("points_conservative"))
160	        if entry_ts is None or points is None or points_cons is None:
161	            skipped_unparseable += 1
162	            continue
168	        trades.append(
169	            TradePath(
170	                day=trading_day_for(entry_ts),
171	                entry_ts=entry_ts,
172	                points_optimistic=points,
173	                points_conservative=points_cons,
```

`_finite` (`loaders.py:79-87`) rejects NaN/inf/non-numeric, so a non-finite `points_conservative`
makes the whole close unparseable and skipped (counted in the note at `loaders.py:189`).

The convention therefore lives in TL `backend/src/trade_lab/services/execution.py`. Module docstring,
`execution.py:28-35`:

```python
28	TWO COLUMNS per fill, both always computed:
30	* ``optimistic`` — exact anchor/barriers: entry at the honest fill, tp/sl exits
31	  at the exact barrier prices, drop exits at the last trade print.
32	* ``conservative`` — entry 1 tick adverse; sl exit 1 tick adverse; tp exit AT
33	  the barrier (the print requirement is already resolution semantics: the
34	  excursion rule fires only once a print reaches the barrier); drop exits at
35	  the print as-is (it is already a real print).
```

The four arms verbatim:

```python
436	            entry_price_ticks=entry_ticks,
437	            # Conservative column: entry 1 tick adverse.
438	            entry_price_conservative_ticks=entry_ticks + 1 if is_long else entry_ticks - 1,
```
```python
463	        if outcome.resolution_type is ResolutionType.TP_HIT:
464	            # tp exit at the barrier in BOTH columns: the excursion rule fires
465	            # only once a print reaches the barrier (resolution semantics).
466	            exit_ticks = position.tp_price_ticks
467	            exit_cons_ticks = position.tp_price_ticks
468	        else:  # SL_HIT
469	            exit_ticks = position.sl_price_ticks
470	            exit_cons_ticks = (
471	                position.sl_price_ticks - 1
472	                if position.is_long
473	                else position.sl_price_ticks + 1
474	            )
```
```python
505	        return self._finalize_close(
506	            position,
507	            reason=drop.reason,
508	            exit_ticks=exit_ticks,
509	            # A print is a print: no conservative exit adjustment on drop
510	            # closes; the conservative column still carries the adverse entry.
511	            exit_cons_ticks=exit_ticks,
```
```python
524	        sign = 1 if position.is_long else -1
525	        points = (exit_ticks - position.entry_price_ticks) * sign * position.tick_size
526	        points_cons = (
527	            (exit_cons_ticks - position.entry_price_conservative_ticks)
528	            * sign
529	            * position.tick_size
530	        )
```

Journalled under the keys propsim reads — `execution.py:626-627`: `"points": execution.points,` /
`"points_conservative": execution.points_conservative,`.

Net convention (a): TP win = tp − 1 tick (entry only); SL loss = −(sl + 2 ticks) (entry + exit); drop
= realized print minus 1 entry tick. **The recorded convention is CONFIRMED in full for this arm.**

### (b) journal-evidence loader — `load_journal_trades` (`loaders.py:218`)

Here the conservative column is SYNTHESIZED from the barriers, mirroring the tracker,
`loaders.py:263-273` and `:289-290`:

```python
263	    for row in deduped_outcomes.values():
264	        resolution = row.get("resolution_type")
265	        if resolution == "tp_hit":
266	            points = tp_points
267	            points_cons = tp_points - tick_size
268	        elif resolution == "sl_hit":
269	            points = -sl_points
270	            points_cons = -(sl_points + 2.0 * tick_size)
271	        else:
272	            skipped += 1
273	            continue
289	                points_optimistic=float(points),
290	                points_conservative=float(points_cons),
```

Stated note, `loaders.py:306-308`: "conservative mirrors the tracker fill model (entry 1 tick adverse,
SL exit 1 tick worse, tick={tick_size})".

Exact convention (b): `tp_hit -> +tp − 1×tick` (entry tick only, TP exit AT barrier);
`sl_hit -> −(sl + 2×tick)` (entry tick + SL-exit tick). **There is no drop arm** — any
`resolution_type` other than `tp_hit`/`sl_hit` is skipped at `:271-273`, so "drop exits unadjusted"
is not applicable in this loader. Tests pin both: `test_loaders.py:235`
(`assert winner.points_conservative == 14.75  # entry 1 tick adverse`), `:238`
(`assert loser.points_conservative == -15.5  # + entry tick + SL exit tick`), `:92`
(executions arm, `14.75`).

**Verdict on the recorded convention:** CONFIRMED in full for (a) — all four arms, in TL source.
CONFIRMED for (b) on the three applicable arms; the drop arm does not exist there because unresolved
rows never become trades.

## C3. Degradation with PARTIAL excursion columns

**Answer: a MIXED POOL.** Rows with NaN `max_mfe_pts`/`max_mae_pts` are NOT dropped and the run is NOT
degraded to realized-only, provided at least one surviving row has a non-NaN `max_mae_pts`.
`entry_price` is never read by this loader, so its NaN-ness has no effect at all.

(i) Run-level degradation flag — `loaders.py:349-359`. Note `.notna().any()`: any single non-NaN MAE
flips the whole run to "excursions available":

```python
349	    has_excursions = (
350	        "max_mfe_pts" in frame.columns
351	        and "max_mae_pts" in frame.columns
352	        and bool(frame["max_mae_pts"].notna().any())
353	    )
354	    degradation = None
355	    if not has_excursions:
356	        degradation = (
357	            "pre-P1 OOS parquet: max_mfe_pts/max_mae_pts absent or empty — "
358	            "unrealized_adverse_first degrades to realized-only"
359	        )
```

Two facts embedded here: the test keys on `max_mae_pts` only (a file with every `max_mae_pts`
populated but every `max_mfe_pts` NaN still yields `has_excursions=True`); and it is computed AFTER
the gate filter (`:346`) and the `label` filter (`:347`), so the GATED SUBSET decides degradation.

(ii) Per-row handling — no drop, NaN becomes `None`, `loaders.py:379-382`:

```python
379	        mfe = getattr(row, "max_mfe_pts", None)
380	        mae = getattr(row, "max_mae_pts", None)
381	        mfe = None if mfe is None or (isinstance(mfe, float) and math.isnan(mfe)) else float(mfe)
382	        mae = None if mae is None or (isinstance(mae, float) and math.isnan(mae)) else float(mae)
```

There is no `continue`/`skipped += 1` on this path. The only row-drop branches in `load_oos_trades`
are the gate filter (`:346`), the `label` NaN filter (`:347`), and the unmapped-resolution branch
(`:372-374`).

(iii) Return — `loaders.py:405-411` (`excursions_available=has_excursions`,
`degradation_reason=degradation`).

(iv) Downstream the mix is handled PER TRADE, not per run — `engine.py:140-141` and `engine.py:47-54`:

```python
140	            mae = _excursion(trade.mae_pts) if self._mode == "unrealized_adverse_first" else None
141	            if mae is not None:
```
```python
47	def _excursion(value: float | None) -> float | None:
48	    """A usable non-negative excursion magnitude, or None (NaN-safe)."""
49	    if value is None:
50	        return None
51	    value = float(value)
52	    if math.isnan(value):
53	        return None
54	    return abs(value)
```

with the engine docstring `engine.py:19-20`: "Trades without excursions (``mae_pts`` is None/NaN)
contribute realized-only observations." So in a mixed pool some trades get the adverse/favorable legs
and others silently do not, inside the same `unrealized_adverse_first` cell.

**Does any counter or stated reason surface the mix to the report? A COUNTER: yes. A STATED REASON:
no.**

```python
43	    pool: dict[str, Any] = {                      # report.py:43-51
44	        "n_trades": len(trades),
49	        "excursions_available": loaded.excursions_available,
50	        "trades_with_excursions": sum(1 for t in trades if t.mae_pts is not None),
51	    }
```
```python
136	        (                                          # report.py:136-140
137	            f"  pool: {pool['n_trades']} trades over {pool['n_days']} days "
138	            f"({pool['first_day']} .. {pool['last_day']}); excursions on "
139	            f"{pool['trades_with_excursions']}/{pool['n_trades']} trades"
140	        ),
```

That `excursions on N/M trades` line is the ONLY signal of the mix. No degraded flag and no reason
fire, because `loaded.excursions_available` is `True` — `report.py:73-79`:

```python
73	            degraded = (
74	                mode == "unrealized_adverse_first" and not loaded.excursions_available
75	            )
76	            cell: dict[str, Any] = {
77	                "degraded_to_realized_only": degraded,
78	                "degradation_reason": loaded.degradation_reason if degraded else None,
```

so the `DEGRADED->realized` marker never prints (`report.py:171`). The OOS `notes` list
(`loaders.py:394-404`) carries no excursion-coverage counter — its only counter is `skipped`
(`:396-398`) — in contrast to the executions loader, which does emit one (`loaders.py:199-202`:
`f"journal join: {len(trades) - missing_excursions}/{len(trades)} trades carry MFE/MAE from journal outcomes"`).

Untested path, stated as fact: `tests/propsim/test_loaders.py` covers only all-columns
(`:273 test_oos_loader_post_p1_columns`) and no-columns
(`:291 test_oos_loader_pre_p1_degrades_to_realized_only`) via the `_oos_frame(with_outcome_columns)`
fixture at `:244-270`. There is no partial/mixed-coverage OOS test in the repo. Code-level edge, also
fact: the NaN guard at `:381-382` keys on `isinstance(..., float)`; `numpy.float64` subclasses `float`
so ordinary float64 columns are handled, but a nullable/object column carrying `pd.NA` satisfies
neither `is None` nor `isinstance(..., float)` and would reach `float(mae)`.

## C4. What a conservative bracket on the OOS source could reach today

Every column `load_oos_trades` reads, with the reading code:

| Column | Where read | Code |
|---|---|---|
| `label` | `loaders.py:347`, `:364` | `frame = frame[frame["label"].notna()]` / `label = str(row.label)` |
| gate column (caller-named, e.g. `gate_0_70_runtime_sessions`) | `loaders.py:340-346` | `if gate_column not in frame.columns: ... raise` / `frame = frame[frame[gate_column].astype(bool)]` |
| `resolution_type` | `loaders.py:365-367` | `resolution = getattr(row, "resolution_type", None)` then `_RESOLUTION_TYPE_BY_LABEL.get(label)` |
| `timestamp` | `loaders.py:375-378` | `ts = pd.Timestamp(row.timestamp)` / `if ts.tzinfo is None: ts = ts.tz_localize("UTC")` |
| `max_mfe_pts` | `loaders.py:350`, `:379`, `:381` | `mfe = getattr(row, "max_mfe_pts", None)` |
| `max_mae_pts` | `loaders.py:351-352`, `:380`, `:382` | `mae = getattr(row, "max_mae_pts", None)` |

That is the complete set. **`entry_price` is NOT read by `load_oos_trades`** — a repo-wide grep for
`entry_price` under `src/alpha_lab/propsim/` returns exactly three hits: `loaders.py:16` (docstring),
`loaders.py:145` (executions dedup signature), `loaders.py:255` (journal dedup signature). None is on
the OOS path. Note `loaders.py:337` reads the whole file (`frame = pd.read_parquet(parquet_path)`), so
every column physically present is reachable in memory; the table above is what the code consumes.

What the file physically carries: the writer's canonical schema, `QL\scripts\ml_training_tab.py:1252-1268`
(`fold, timestamp, session, binary_true_tradeable, label_encoded, label, pred_label_encoded,
pred_label, prob_tradeable_reversal, gate_0_70_runtime_sessions, gate_0_70_ny, max_mfe_pts,
max_mae_pts, entry_price, resolution_type`) plus per-class `prob_<class>` columns (`:1332-1335`), with
row emission at `:1327-1330` and the NaN degradation at `:1285-1294` (see B4).

**Is `entry_price` + the bundle's `label_policy` tp/sl sufficient to derive the barrier prices?**
Facts only:

- `entry_price` is present in the post-P1 schema (`ml_training_tab.py:1266`, `:1329`) but unread by the
  loader today.
- tp/sl are reachable: the CLI resolves them from `strategy.json -> label_policy` (`cli.py:74-83`,
  quoted in C5), written from `strategy_contract.py:223-224` (`"tp_points": du.tp_points,` /
  `"sl_points": du.sl_points,`).
- **Missing for a signed barrier: DIRECTION.** No `direction` column exists in the OOS schema
  (`base_columns`, `ml_training_tab.py:1252-1268` — no direction, no `level_kind`, no `level_price`),
  and the loader reads none. Its entire notion of sign is `resolution == "tp_hit" -> +tp` /
  `"sl_hit" -> −sl` (`loaders.py:368-371`). The excursion columns are unsigned magnitudes
  (`engine.py:47-54` takes `abs()`), so direction is not recoverable from them either. By contrast the
  TL tracker's conservative fill is explicitly direction-branched (`execution.py:438`, `:470-474`,
  `:524`).
- Also absent from the loader's reach: any per-row `tick_size`, any exit price, any exit timestamp.

**Tick size — sources of truth:**

1. propsim CLI flag, wired to the JOURNAL loader only — `cli.py:64-65`:
   ```python
   64	    parser.add_argument("--tick-size", type=float, default=0.25,
   65	                        help="tick size for the journal conservative fill model")
   ```
   threaded at `cli.py:107-110`; defaulted again at `loaders.py:223` (`tick_size: float = 0.25,`).
   **`load_oos_trades` has no `tick_size` parameter** (`loaders.py:326-332`) and the CLI never passes
   one to it (`cli.py:97-99`).
2. Bundle contract: `strategy.json` carries `tick_size`, written at `strategy_contract.py:195`
   (`"tick_size": config.tick_size,`; also `:164` on the minimal-contract branch); config default
   `config.py:374-378` (`tick_size: float = Field(default=0.25, gt=0, ...)`). propsim never reads
   `strategy.json["tick_size"]` — `cli.py:77-78` reads only the `label_policy` subtree.
3. Platform constant: `strategy_core.constants.DEFAULT_TICK_SIZE`, asserted equal to the production
   trade grid at `engine_decision.py:83-88` (`TRADE_TICK = 0.25`; `assert TRADE_TICK == DEFAULT_TICK_SIZE`).
4. Execution-time source of truth (TL): the active contract, `backend/src/trade_lab/api/app.py:303-308`
   (`ExecutionPolicy(tick_size=contract.tick_size, tp_points=..., sl_points=..., point_value=...)`),
   consumed by `ExecutionPolicy` (`execution.py:88-95`) and stored per position (`execution.py:441`).

## C5. CLI surface of `python -m alpha_lab.propsim` (read from source, not executed)

Entry point — `src\alpha_lab\propsim\__main__.py:1-7`:

```python
1	"""Entry point: ``python -m alpha_lab.propsim``."""
3	import sys
5	from alpha_lab.propsim.cli import main
7	sys.exit(main())
```

(No console-script entry point: grep for `propsim` in `pyproject.toml` returns no matches.)

Embedded usage text — `cli.py:1-14`:

```python
1	"""CLI for the prop-firm evaluation walker.
3	Usage (spec form)::
5	    python -m alpha_lab.propsim --source executions <dir> [--journal <dir>] \
6	        --preset topstep_50k --column conservative --n 10000 --seed 42 --json out.json
7	    python -m alpha_lab.propsim --oos <parquet> --preset topstep_50k ...
8	    python -m alpha_lab.propsim --source journal <dir> --tp-points 15 --sl-points 15 ...
10	For ``--oos`` the TP/SL barrier points resolve from (in order) the
11	``--tp-points``/``--sl-points`` flags, then a ``strategy.json`` next to the
12	parquet (``label_policy.tp_points``/``sl_points``); ``--source journal``
13	requires the flags explicitly.
14	"""
```

The argparse block in full — `cli.py:33-68`:

```python
33	_SOURCE_KINDS = ("executions", "journal")
36	def build_parser() -> argparse.ArgumentParser:
37	    parser = argparse.ArgumentParser(
38	        prog="python -m alpha_lab.propsim",
39	        description="Prop-firm evaluation walker: pass-probability from equity paths.",
40	    )
41	    parser.add_argument(
42	        "--source",
43	        nargs=2,
44	        metavar=("KIND", "DIR"),
45	        help=f"trade source: KIND in {{{', '.join(_SOURCE_KINDS)}}} + its directory",
46	    )
47	    parser.add_argument("--journal", type=Path, default=None,
48	                        help="journal dir joined for MFE/MAE (executions source)")
49	    parser.add_argument("--oos", type=Path, default=None,
50	                        help="a bundle's oos_predictions.parquet")
51	    parser.add_argument("--preset", default="topstep_50k", choices=sorted(PRESETS),
52	                        help="ruleset preset (default: topstep_50k)")
53	    parser.add_argument("--column", choices=[*FILL_COLUMNS, "both"], default="both",
54	                        help="fill column(s) to evaluate (default: both)")
55	    parser.add_argument("--n", type=int, default=10_000, dest="n_runs",
56	                        help="bootstrap runs (default: 10000)")
57	    parser.add_argument("--seed", type=int, default=42)
58	    parser.add_argument("--max-days", type=int, default=1_000,
59	                        help="runaway guard per bootstrap run (default: 1000)")
60	    parser.add_argument("--json", type=Path, default=None, dest="json_out",
61	                        help="write the full JSON payload here")
62	    parser.add_argument("--tp-points", type=float, default=None)
63	    parser.add_argument("--sl-points", type=float, default=None)
64	    parser.add_argument("--tick-size", type=float, default=0.25,
65	                        help="tick size for the journal conservative fill model")
66	    parser.add_argument("--gate-column", default=None,
67	                        help="OOS boolean gate column (default: all labeled rows)")
68	    return parser
```

| Flag | Type / nargs | Default | dest | Choices |
|---|---|---|---|---|
| `--source` | `nargs=2` (KIND, DIR) | `None` | `source` | validated in `_load`, not by argparse |
| `--journal` | `Path` | `None` | `journal` | — |
| `--oos` | `Path` | `None` | `oos` | — |
| `--preset` | str | `"topstep_50k"` | `preset` | `sorted(PRESETS)` = `apex_50k_eod, apex_50k_intraday, topstep_50k, tpt_50k_test` (`presets.py:25-78`) |
| `--column` | str | `"both"` | `column` | `optimistic, conservative, both` (`FILL_COLUMNS`, `engine.py:38`) |
| `--n` | `int` | `10_000` | `n_runs` | — |
| `--seed` | `int` | `42` | `seed` | — |
| `--max-days` | `int` | `1_000` | `max_days` | — |
| `--json` | `Path` | `None` | `json_out` | — |
| `--tp-points` | `float` | `None` | `tp_points` | — |
| `--sl-points` | `float` | `None` | `sl_points` | — |
| `--tick-size` | `float` | `0.25` | `tick_size` | — |
| `--gate-column` | str (no `type=`) | `None` | `gate_column` | — |

**Which flags resolve tp/sl from a bundle `strategy.json`:** `--oos` is the ONLY one, and only when
`--tp-points`/`--sl-points` are not both supplied — `cli.py:71-88`:

```python
71	def _resolve_tp_sl(args: argparse.Namespace) -> tuple[float, float]:
72	    if args.tp_points is not None and args.sl_points is not None:
73	        return float(args.tp_points), float(args.sl_points)
74	    if args.oos is not None:
75	        contract = args.oos.parent / "strategy.json"
76	        if contract.is_file():
77	            policy = json.loads(contract.read_text(encoding="utf-8")).get(
78	                "label_policy", {}
79	            )
80	            tp = args.tp_points if args.tp_points is not None else policy.get("tp_points")
81	            sl = args.sl_points if args.sl_points is not None else policy.get("sl_points")
82	            if tp is not None and sl is not None:
83	                return float(tp), float(sl)
84	    msg = (
85	        "TP/SL barrier points unresolved: pass --tp-points/--sl-points "
86	        "(or point --oos at a bundle whose strategy.json carries label_policy)"
87	    )
88	    raise ValueError(msg)
```

The contract path is `args.oos.parent / "strategy.json"` (sibling of the parquet); only the
`label_policy` subtree is read (`tp_points`, `sl_points` — NOT `tick_size`, which lives at contract
top level, `strategy_contract.py:195`); per-flag override is possible (`:80-81`). `--source journal`
reaches `_resolve_tp_sl` too (`cli.py:107`) but with `args.oos is None` the contract branch is
skipped, so the flags are mandatory there — matching the docstring at `cli.py:12-13`.

Source dispatch / mutual exclusion — `cli.py:91-110`:

```python
91	def _load(args: argparse.Namespace) -> LoadedTrades:
92	    if (args.source is None) == (args.oos is None):
93	        msg = "exactly one trade source required: --source KIND DIR | --oos PARQUET"
94	        raise ValueError(msg)
95	    if args.oos is not None:
96	        tp, sl = _resolve_tp_sl(args)
97	        return load_oos_trades(
98	            args.oos, tp_points=tp, sl_points=sl, gate_column=args.gate_column
99	        )
100	    kind, path = args.source
105	    if kind == "executions":
106	        return load_executions_trades(directory, journal_dir=args.journal)
107	    tp, sl = _resolve_tp_sl(args)
108	    return load_journal_trades(
109	        directory, tp_points=tp, sl_points=sl, tick_size=args.tick_size
110	    )
```

Line 98 confirms `--tick-size` is never forwarded to the OOS loader. `main` behaviour (`cli.py:113-138`):
both breach modes always run (`report.py:72` iterates `BREACH_MODES`); `--column both` expands to
`FILL_COLUMNS` (`cli.py:119`); `ValueError`/`FileNotFoundError` print `propsim: {exc}` to stderr and
return exit code `2` (`cli.py:129-131`); the human table always prints; `--json` additionally writes
`json.dumps(report, indent=2, sort_keys=False)`.

**NOT OBTAINABLE READ-ONLY:** nothing in C1–C5. Every question was answerable from source text; no
parquet, dataset, or run was touched. The absence claims (no partial-excursion OOS test; no
`entry_price`/`direction`/`tick_size` read in `load_oos_trades`; no console-script entry point) rest
on exhaustive greps over `src/alpha_lab/propsim/`, `tests/propsim/`, and `pyproject.toml`.

---

# PART D — seal integrity

Sealed range: **2026-06-12 .. 2026-07-10 inclusive** (29 calendar days). Every fact about the sealed
day directories came from `ls -l` / `find -printf '%p %s'` metadata only; no `mbp*.parquet` in the
range was opened, read, or parsed.

## D1. Day directories under QL `data/databento/NQ` in the sealed range

Root: `C:\Users\gonza\Documents\Claude-Quant-Lab\data\databento\NQ`.

**Total day directories in range: 25** (independently re-verified by direct enumeration:
`ls -d 2026-06-1[2-9] 2026-06-2* 2026-06-30 2026-07-0* 2026-07-10 | wc -l` → `25`).

A recursive `find` over those 25 dirs returned exactly 25 entries — **one file per directory, and in
every case the filename is `mbp1.parquet`.** There is **no `mbp10.parquet` anywhere in the sealed
range**, no subdirectories, no sidecar files, no hidden files.

| # | Day dir | Weekday | File | Size (bytes) | Human | mtime |
|---|---|---|---|---|---|---|
| 1 | 2026-06-12 | Fri | mbp1.parquet | 565,537,532 | 539.3 MiB | 2026-07-11 03:22 |
| — | 2026-06-13 | **Sat** | *(no directory)* | — | — | — |
| 2 | 2026-06-14 | Sun | mbp1.parquet | 25,451,911 | 24.3 MiB | 2026-07-11 03:22 |
| 3 | 2026-06-15 | Mon | mbp1.parquet | 358,353,837 | 341.8 MiB | 2026-07-11 03:23 |
| 4 | 2026-06-16 | Tue | mbp1.parquet | 512,999,394 | 489.2 MiB | 2026-07-11 03:23 |
| 5 | 2026-06-17 | Wed | mbp1.parquet | 463,432,801 | 442.0 MiB | 2026-07-11 03:23 |
| 6 | 2026-06-18 | Thu | mbp1.parquet | 331,634,800 | 316.3 MiB | 2026-07-11 03:24 |
| 7 | 2026-06-19 | Fri | mbp1.parquet | 83,444,534 | 79.6 MiB | 2026-07-11 03:23 |
| — | 2026-06-20 | **Sat** | *(no directory)* | — | — | — |
| 8 | 2026-06-21 | Sun | mbp1.parquet | 21,969,374 | 21.0 MiB | 2026-07-11 03:23 |
| 9 | 2026-06-22 | Mon | mbp1.parquet | 352,272,714 | 336.0 MiB | 2026-07-11 03:24 |
| 10 | 2026-06-23 | Tue | mbp1.parquet | 480,310,682 | 458.1 MiB | 2026-07-11 03:24 |
| 11 | 2026-06-24 | Wed | mbp1.parquet | 468,537,178 | 446.8 MiB | 2026-07-11 03:25 |
| 12 | 2026-06-25 | Thu | mbp1.parquet | 398,470,333 | 380.0 MiB | 2026-07-11 03:25 |
| 13 | 2026-06-26 | Fri | mbp1.parquet | 388,785,183 | 370.8 MiB | 2026-07-11 03:25 |
| — | 2026-06-27 | **Sat** | *(no directory)* | — | — | — |
| 14 | 2026-06-28 | Sun | mbp1.parquet | 12,834,937 | 12.2 MiB | 2026-07-11 03:25 |
| 15 | 2026-06-29 | Mon | mbp1.parquet | 337,569,063 | 321.9 MiB | 2026-07-11 03:26 |
| 16 | 2026-06-30 | Tue | mbp1.parquet | 280,737,382 | 267.7 MiB | 2026-07-11 03:26 |
| 17 | 2026-07-01 | Wed | mbp1.parquet | 321,518,284 | 306.6 MiB | 2026-07-11 03:26 |
| 18 | 2026-07-02 | Thu | mbp1.parquet | 426,263,904 | 406.5 MiB | 2026-07-11 03:27 |
| 19 | 2026-07-03 | Fri | mbp1.parquet | 67,271,825 | 64.2 MiB | 2026-07-11 03:26 |
| — | 2026-07-04 | **Sat** | *(no directory)* | — | — | — |
| 20 | 2026-07-05 | Sun | mbp1.parquet | 9,406,175 | 9.0 MiB | 2026-07-11 03:26 |
| 21 | 2026-07-06 | Mon | mbp1.parquet | 270,937,755 | 258.4 MiB | 2026-07-11 03:27 |
| 22 | 2026-07-07 | Tue | mbp1.parquet | 354,882,135 | 338.4 MiB | 2026-07-11 03:27 |
| 23 | 2026-07-08 | Wed | mbp1.parquet | 370,565,066 | 353.4 MiB | 2026-07-11 03:27 |
| 24 | 2026-07-09 | Thu | mbp1.parquet | 258,522,304 | 246.5 MiB | 2026-07-11 03:27 |
| 25 | 2026-07-10 | Fri | mbp1.parquet | 239,819,751 | 228.7 MiB | 2026-07-11 03:28 |

**Missing days:** exactly four — 2026-06-13, 06-20, 06-27, 07-04, **all Saturdays**. **Zero calendar
weekdays in the range lack a directory**; additionally every Sunday in range has one (Sunday-evening
globex open — the 9–25 MiB files). Aggregate: 25 files, 7,891,038,844 bytes ≈ 7.35 GiB.

Era observation: the entire sealed range is single-era `mbp1.parquet`, and every file's mtime falls in
one ~6-minute window on **2026-07-11 03:22–03:28**. The parent dir
`QL\data\databento\` holds `GLBX-20260711-EEDSMFU845.zip` and `_batch_tmp/`, whose 20260711 datestamp
matches — consistent with a single Databento batch download on 2026-07-11 producing the whole range.

## D2. Derived artifacts covering the sealed range

### D2.a `ml_utility_*` / `ml_features_*` inside the sealed day dirs — **NONE**

The recursive `find` over all 25 sealed day dirs returned exactly 25 entries, all `mbp1.parquet`. No
`ml_utility_*`, no `ml_features_*`, no `.STALE_BACKUP*` variants, nothing else.

Boundary evidence from the unsealed tree, for contrast: 819 `ml_utility_*.parquet` files exist across
the NQ tree; the **maximum date of any day dir containing an `ml_utility_*.parquet` is 2026-03-02**
(next-highest 2026-02-22). Distinct tags present anywhere: `0607f6c8`, `3d2f8466`, `3d60f9a9`,
`7850272e`, `7850272e.STALE_BACKUP_16h48`, `89d82fb5`, `960ba73b`, `b8b2e14d`, `d8e239c7`, `dab124ca`.
**No `ml_features_*.parquet` exists anywhere in the NQ tree.** The derived-feature cache stops more
than three months before the seal opens.

### D2.b Journals — **7 files in TL `backend/data/journal/` named for sealed trading days**; `backend/data/w3b_journal/` **NONE**

This is the one material finding of Part D. Full listing of
`C:\Users\gonza\Documents\Trade-Lab\backend\data\journal\` (15 files):

| Filename | Size (bytes) | mtime | In seal? |
|---|---|---|---|
| 2021-12-02.jsonl | 5,122 | Jun 16 20:20 | no |
| 2021-12-15.jsonl | 5,135 | Jul 10 10:13 | no |
| 2021-12-30.jsonl | 1,290 | Jun 16 09:40 | no |
| 2022-01-10.jsonl | 3,858 | Jul 10 01:08 | no |
| 2022-03-04.jsonl | 3,872 | Jul 10 15:54 | no |
| 2026-01-05.jsonl | 2,340 | Jun 16 20:13 | no |
| 2026-02-10.jsonl | 6,372 | Jul 10 13:47 | no |
| 2026-02-11.jsonl | 7,665 | Jul 10 14:15 | no |
| **2026-06-15.jsonl** | **2,543** | 2026-06-17 01:21 | **YES** |
| **2026-06-16.jsonl** | **45,896** | 2026-06-17 01:26 | **YES** |
| **2026-06-17.jsonl** | **5,096** | 2026-06-17 13:07 | **YES** |
| **2026-06-18.jsonl** | **3,852** | 2026-06-18 08:31 | **YES** |
| **2026-07-03.jsonl** | **2,583** | 2026-07-13 17:59 | **YES** |
| **2026-07-06.jsonl** | **7,704** | 2026-07-07 21:23 | **YES** |
| **2026-07-07.jsonl** | **3,824** | 2026-07-07 20:23 | **YES** |

Not opened — identification is filename + stat metadata only, per the seal rule.

The filename is DATA-derived, not wall-clock derived —
`TL\backend\src\trade_lab\services\journal.py:95-96`:

```python
95	            day = trading_day_for(ts_utc) if ts_utc is not None else None
96	            name = f"{day.isoformat()}.jsonl" if day is not None else "undated.jsonl"
```

(`from trade_lab.domain.trading_day import trading_day_for`, `journal.py:18`). So `2026-06-16.jsonl`
holds records whose event timestamps resolve to trading day 2026-06-16 — inside the seal. The same
convention is documented for the w3b replay path at `backend/scripts/w3b/headless_replay.py:265`.

Two provenance sub-cases, distinguishable from mtimes:

- 2026-06-15/16/17/18 (mtimes 2026-06-17, 2026-06-18) and 2026-07-06/07 (mtimes 2026-07-07) were
  written at or immediately after the wall-clock time of those trading days — **weeks BEFORE the raw
  parquet for those days existed** (all sealed parquet mtimes are 2026-07-11 03:2x). They cannot be
  replays of sealed store data; they are live/paper sessions journaled in real time.
- **2026-07-03.jsonl is the exception: mtime 2026-07-13 17:59, AFTER the 2026-07-11 03:26 ingest of
  `NQ/2026-07-03/mbp1.parquet`.** Its content therefore could have been produced by replaying sealed
  data. Which it is, is **NOT OBTAINABLE READ-ONLY** under the seal — settling it requires opening a
  journal that covers a sealed trading day.

Read-seam note (fact, not a recommendation): the journal reader globs the whole directory with no date
filter — `TL\backend\src\trade_lab\services\performance.py:198`:
`for file in sorted(journal_dir.glob("*.jsonl"), key=lambda p: p.name):` — so any performance query
against `backend/data/journal/` includes all seven sealed-range files.

`backend/data/w3b_journal/` — **NONE**. 30+ dated subdirectories, all in 2025-11-21 .. 2026-01-15
(sorted tail: 2025-12-28, 2026-01-01, 2026-01-06, 2026-01-11, 2026-01-15). A targeted `find` for
sealed-date patterns returned nothing. Adjacent w3b dirs also clean: `w3b_repro/` (only
`control_12-23.log`, `repro_12-18.log`), `w3b_worker_bench4_w3_journal/` (tail 2026-01-20 … 2026-02-13),
`w3b_worker_bench4_w2_journal/`, `w3b_bench/`.

### D2.c Executions files — **NONE in the sealed range**

Full listing of `TL\backend\data\executions\`:

| Filename | Size | mtime |
|---|---|---|
| 2026-01-05.jsonl | 1,215 | Jul 11 03:35 |
| 2026-02-10.jsonl | 145 | Jul 10 14:04 |
| 2026-02-15.jsonl | 145 | Jul 10 13:37 |
| undated.jsonl | 11,205 | Jul 13 17:54 |

Naming uses the identical trading-day derivation — `TL\backend\src\trade_lab\services\execution.py:205-206`
mirrors `journal.py` exactly, documented at `execution.py:9` as `executions/<trading_day>.jsonl`. So no
execution record carries a sealed-range trading day, **except** that `undated.jsonl` by construction
holds records with no resolvable timestamp: its date coverage is **NOT OBTAINABLE READ-ONLY** —
establishing whether it contains sealed-range activity requires opening it.

Repo-wide searches: `executions*.parquet` — **NONE** anywhere in QL, TL, or SC; `*.parquet` under
`TL\backend\data\` — **NONE** (zero parquet files under TL's data tree at all); `-iname 'executions*'`
across all three repos returns only the `executions` directory plus two frontend sources
(`ExecutionsPanel.tsx`, `ExecutionsPanel.test.tsx`). Total `*.jsonl` under `TL/backend/data/`: 62.

### D2.d Bundles under QL `models/` whose training window overlaps the seal — **NONE**

Zero of the 21 bundles touches the sealed range; the latest training date anywhere in the model store
is 2026-02-22 (`date_range.end`). Windows read from `metadata.json` / `evaluation.json` (config
metadata, not market data).

| # | Bundle (under `QL\models\`) | date_range.start | date_range.end | n days | Overlaps seal? |
|---|---|---|---|---|---|
| 1–7 | `NQ_20260224_210453`, `NQ_20260224_211141`, `NQ_20260404_053924`, `NQ_20260404_120558`, `NQ_20260404_185203`, `NQ_20260404_230538`, `NQ_20260404_230908` | *(no date fields)* | *(no date fields)* | — | no evidence of overlap |
| 8 | `NQ_20260405_015538` | 2025-06-02 | 2026-02-20 | 224 | NO |
| 9 | `NQ_20260405_147t_5m_15m_multiclass-250602-260220` | 2025-06-02 | 2026-02-20 | 224 | NO |
| 10 | `NQ_20260405_147t_5m_250602-260220` | 2025-06-02 | 2026-02-20 | 224 | NO |
| 11 | `NQ_20260405_147t_5m_30m_multiclass-250602-260220` | 2025-06-02 | 2026-02-20 | 224 | NO |
| 12 | `NQ_20260405_147t_5m_30m_multiclass-250602-260220-iterations800_depth4` | 2025-06-02 | 2026-02-20 | 224 | NO |
| 13 | `NQ_20260405_extrema_rebound_crossing_10p` | 2025-06-02 | 2026-02-20 | 224 | NO |
| 14 | `NQ_20260427_183527` | 2021-12-02 | 2022-03-10 | 85 | NO |
| 15 | `NQ_20260602_184719` | 2025-06-02 | 2025-10-03 | 106 | NO |
| 16 | `NQ_20260602_232808` | 2025-06-02 | 2026-02-22 | 225 | NO |
| 17 | `NQ_20260603_233847` | 2025-06-02 | 2026-02-22 | 225 | NO |
| 18 | `NQ_20260604_012623` | 2025-06-02 | 2026-02-22 | 225 | NO |
| 19 | `NQ_20260604_015413` | 2025-06-02 | 2025-10-01 | 104 | NO |
| 20 | `NQ_W3_20260613T055600Z` | 2025-11-21 | 2026-02-13 | 73 | NO |
| 21 | `NQ_W3_20260617T220752Z` | 2025-11-21 | 2026-02-13 | 73 | NO |

Bundles 1–7 carry **no `date_range` / `dates_used` / any ISO-8601 date field** in either
`metadata.json` or `evaluation.json` (a repo-wide `grep -o -E '20[0-9]{2}-[0-9]{2}-[0-9]{2}'` over
`models/*/metadata.json` and `*/strategy.json` returned zero matches for them). Their exact training
window is **NOT OBTAINABLE READ-ONLY** from the bundle artifacts — recovering it would require
correlating build timestamps against training-run logs or shell history, or re-running the trainer.
Overlap is nonetheless physically impossible: their directory mtimes are 2026-02-24 and 2026-04-04,
and the sealed parquet did not exist on disk until 2026-07-11.

The two production-relevant W3 bundles are bounded well short of the seal.
`NQ_W3_20260617T220752Z/evaluation.json` purged-day folds: fold0 `test_start 2026-01-23`,
`max_label_window_end 2026-01-20 17:00:00-05:00`; fold1 `test_start 2026-01-30`,
`max_label_window_end 2026-01-27 17:00:00-05:00`; fold2 `test_start 2026-02-06`,
`max_label_window_end 2026-02-03 17:00:00-05:00` (identical in `NQ_W3_20260613T055600Z`).
`NQ_W3_20260617T220752Z/strategy.json:63` records `"dataset_config_hash": "7850272e"`. Each W3 bundle
holds `evaluation.json`, `metadata.json`, `model.cbm`, `model.cbm.sha256` (77 B),
`oos_predictions.parquet` (10,123 B / 10,155 B), `strategy.json` (3,685 B); the OOS parquets cover the
pre-seal 2025-11-21..2026-02-13 window and were not opened.

**D2 bottom line:** three of four categories are clean (`ml_*` caches NONE, executions NONE-in-range,
bundles NONE). The exception is seven live-session journal files in `TL/backend/data/journal/` named
for sealed trading days, six of which provably predate the sealed parquet's existence on disk, and one
(`2026-07-03.jsonl`, mtime 2026-07-13) which does not.

## D3. Mechanism to bound a dataset build's end date

Two bounding surfaces, and they are coupled: the warmer has no date CLI of its own — it re-parses the
experiment CLI's parser against a hardcoded argv constant.

### D3.a `scripts/run_dashboard_session_experiment.py` — the flag is `--end` (paired with `--start`)

Verbatim, `QL\scripts\run_dashboard_session_experiment.py:56-68`:

```python
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a dashboard-utility session experiment on local Databento parquet.",
    )
    parser.add_argument("--preset", choices=sorted(SESSION_EXPERIMENT_PRESETS), default="all_to_ny")
    parser.add_argument("--train-sessions", help="Comma-separated override, e.g. ny or asia,london")
    parser.add_argument("--evaluation-sessions", help="Comma-separated override")
    parser.add_argument("--gate-sessions", help="Comma-separated confidence-gate override")
    parser.add_argument("--symbol", default="NQ")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--bar-type", default="147t", choices=["147t", "987t", "2000t", "1m"])
```

`--start` and `--end` are declared bare: **no `type=`, no `help=`, no `default=`, no validation** —
plain strings. Enforcement semantics, `run_dashboard_session_experiment.py:48-53` (quoted in B2).
Three load-bearing properties:

1. **`--end` is INCLUSIVE** (`d <= end_value`). Excluding the seal means `--end 2026-06-11`, not
   `--end 2026-06-12`.
2. The comparison is lexicographic on `YYYY-MM-DD`, order-equivalent to date comparison for that
   format — but a malformed value (e.g. `--end 2026-6-11`) silently mis-slices rather than erroring.
3. **If `--end` is omitted it defaults to `available[-1]`, the newest day on disk — today inside the
   seal.** The bound is opt-in; there is no seal-aware default, guard, or assertion in either the
   parser or the slicer.

A `--dry-run` flag exists (`run_dashboard_session_experiment.py:104-108`, "Print resolved config/date
range without building/training").

### D3.b The cache warmer `scripts/w3_cache_warmer.py` — no date flag; the bound is a source constant

Its own argparse, verbatim, `w3_cache_warmer.py:319-339`:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, help="process pool size N")
    parser.add_argument(
        "--days", help="comma-separated explicit in-window day subset (smoke)"
    )
    parser.add_argument(
        "--limit", type=int, help="process at most this many uncached days (smoke)"
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=6,
        help=(
            "recycle each pool worker after this many days so per-worker RSS stays "
            "near a single-day peak instead of creeping over a long run "
            "(0 = never recycle). The day build is event-decode-bound (~3-5GB) and "
            "long-lived workers accumulate ~0.1GB/day; recycling bounds the aggregate."
        ),
    )
    args = parser.parse_args()
```

That is the complete flag set: `--workers`, `--days`, `--limit`, `--max-tasks-per-child`. **No
`--start`, no `--end`, no `--data-dir`, no `--symbol`.** The date bound is the module-level constant
`EXP_ARGV` (`w3_cache_warmer.py:77-98`), whose date pair is `"--start", "2025-11-21", "--end",
"2026-02-13"` (`:81-82`), fed into the experiment parser at `:307-316`:

```python
    import run_dashboard_session_experiment as exp

    ns = exp._build_parser().parse_args(EXP_ARGV)
    config = exp._resolve_config(ns)
    cache_tag = config.dataset_config_hash()
    util_kwargs = config.dashboard_utility.model_dump()
    data_dir = ns.data_dir
    available = exp.get_available_dates(ns.symbol, data_dir)
    window_dates = exp._date_slice(available, ns.start, ns.end)
    return cache_tag, util_kwargs, ns.symbol, data_dir, window_dates
```

(docstring `:300-306`: "Reuses the experiment CLI's parser + resolver so the cache tag and day list are
identical to the serial train.") So the warmer's bounding mechanism is the `"--end", "2026-02-13"` pair
at `w3_cache_warmer.py:82`, flowing through the same inclusive `_date_slice`. As pinned it ends 118
days before the seal opens; changing it requires editing source — it cannot be done from the command
line.

Two secondary containment properties: `--days` is window-clamped and fails loud
(`w3_cache_warmer.py:344-349`: `unknown = [d for d in requested if d not in window_dates]` →
`raise SystemExit(f"--days not in D-036 window: {unknown}")`), so a sealed date passed via `--days` is
rejected; and the write target is `w3_cache_warmer.py:147-148`
(`return data_dir / symbol / date_str / f"ml_utility_{cache_tag}.parquet"`), i.e. the artifact class
D2.a searched for and found absent in all 25 sealed dirs — positive evidence the warmer has never been
pointed at the seal.

### D3.c Residual gap (fact)

The only unbounded path is `run_dashboard_session_experiment.py` invoked WITHOUT `--end`: `_date_slice`
then defaults `end_value = available[-1]`, and `get_available_dates` enumerates the day dirs on disk,
which now include all 25 sealed days. Nothing in either script consults a seal manifest, an ignore
list, or a max-date guard. The seal is currently enforced by operator discipline plus the frozen
`EXP_ARGV` constant, not by code.

---

# PART E — new-era reachability, end to end

No parquet was opened anywhere for this part; E3's counts derive from directory/filename enumeration
plus the catalog's static predicate. Per the store-identity note in the tip block, the TL-configured
store path (`Trade-Dashboard\data\...`, a SYMLINKD) and the QL store path are the same directory.

## E1. SC `DAY_FILE_PRIORITY` — mbp10 precedes mbp1

`C:\Users\gonza\Documents\Strategy-core\src\strategy_core\data\databento_parquet.py:73-79`:

```python
73	#: Per-date file resolution priority for trading-day composition (mirrors the
74	#: research store convention: best book depth available wins).
75	DAY_FILE_PRIORITY = (
76	    ("mbp10.parquet", "mbp-10"),
77	    ("mbp1.parquet", "mbp-1"),
78	    ("trades.parquet", "trades"),
79	)
```

Index 0 is mbp-10, index 1 is mbp-1: **mbp10 has strict precedence over mbp1.** Exactly two consumers
exist in the SC source tree (grep over `src/`, `--include=*.py`): `databento_parquet.py:295` (the
era-boundary fallback, E2) and `databento_parquet.py:349` (the primary resolver).

Primary resolver, `databento_parquet.py:346-353`:

```python
346	    @staticmethod
347	    def _resolve_day_file(root: Path, day: date) -> tuple[Path, str] | None:
348	        folder = root / day.isoformat()
349	        for filename, schema in DAY_FILE_PRIORITY:
350	            candidate = folder / filename
351	            if candidate.exists():
352	                return candidate, schema
353	        return None
```

**A day dir containing BOTH resolves to `mbp10.parquet` / schema `"mbp-10"`** — `_resolve_day_file`
returns on the first existing candidate in priority order. `mbp1.parquet` is never selected for such a
day by `_resolve_day_file`; it is reachable only through the E2 fallback generator, which filters on
`schema == day_schema` rather than following precedence.

Store fact (listing only): **37 day dirs contain BOTH files, spanning 2026-01-11 .. 2026-02-22**; e.g.
`2026-02-22/` holds `mbp1.parquet`, `mbp10.parquet`, `ml_features.parquet`,
`ml_features_3154bd96.parquet`, `ohlcv_1m.parquet`, while `2026-02-23/` holds only `mbp1.parquet`.
**2026-02-23 is the first mbp-1-only day** — the era boundary in this store.

## E2. The schema-matching prior-day fallback added at SC `b21316e`

Provenance, `git show b21316e`:

```
commit b21316e42f22718dedad2e7511dded60eee09a61
Author: algochef <gonzalezlg020@gmail.com>
Date:   Sat Jul 11 03:35:16 2026 -0500

    fix(data): era-boundary prior-day schema-matching fallback + buy-trade ordering pin + D-P-17 wording (INGEST close-verify)

 docs/DECISIONS.md                           |  6 ++--
 src/strategy_core/data/databento_parquet.py | 48 ++++++++++++++++++++---------
 tests/test_databento_parquet_day_mode.py    | 38 +++++++++++++++++++++++
 tests/test_databento_parquet_source.py      | 18 +++++++++++
```

The reader diff replaced an unconditional degrade with the fallback:

```diff
         if prev_resolved is not None and prev_resolved[1] != day_schema:
-            warnings.append(
-                cls._warning(
-                    DataQualityCode.MISSING_PRIOR_DAY_FILE,
-                    "prior-day file schema differs; trading-day window served from a single file",
-                    ...
-                )
+            # INGEST close-verify fix: at a schema-era boundary (e.g. the prior
+            # day resolves mbp10.parquet while the day is mbp-1-only), a
+            # schema-MATCHING prior-day file may still exist in the folder —
+            # prefer it over degrading, so the prior-evening hour is served and
+            # prior_full_day_extremes stays exact across the boundary.
+            matching = next(
+                (
+                    (root / prev_day.isoformat() / filename, schema)
+                    for filename, schema in DAY_FILE_PRIORITY
+                    if schema == day_schema
+                    and (root / prev_day.isoformat() / filename).exists()
+                ),
+                None,
             )
-            prev_resolved = None
+            if matching is not None:
+                prev_resolved = matching
+            else:
+                warnings.append(...)
+                prev_resolved = None
```

`for_trading_day` at current SC HEAD, `databento_parquet.py:247-344` (the load-bearing body):

```python
247	    @classmethod
248	    def for_trading_day(
249	        cls,
250	        symbol_dir: Path | str,
251	        trading_day: date,
252	        *,
253	        requested_symbol: str | None = None,
254	        front_month_only: bool = True,
255	        batch_size: int = 65_536,
256	    ) -> DatabentoParquetSource:
257	        """Canonical trading-day stream: [prev-day 18:00 ET, trading-day 18:00 ET).
263	        ... When the priority-resolved
264	        prior-day file's schema differs from the day's, a schema-MATCHING
265	        prior-day file is preferred if present (schema-era boundary); only a
266	        missing (or unmatchable-schema) prior-day file degrades to a
267	        single-file scan with a ``MISSING_PRIOR_DAY_FILE`` warning on the
268	        event stream.
269	        """
271	        root = Path(symbol_dir)
273	        day_resolved = cls._resolve_day_file(root, trading_day)
274	        if day_resolved is None:
275	            raise FileNotFoundError(
276	                f"no parquet day file for {symbol} {trading_day.isoformat()}"
277	            )
278	        day_path, day_schema = day_resolved
279	        prev_day = trading_day - timedelta(days=1)
281	        start = datetime.combine(prev_day, TRADING_DAY_BOUNDARY, tzinfo=tz).astimezone(UTC)
282	        end = datetime.combine(trading_day, TRADING_DAY_BOUNDARY, tzinfo=tz).astimezone(UTC)
283	        split = datetime(trading_day.year, trading_day.month, trading_day.day, tzinfo=UTC)
284	        prev_resolved = cls._resolve_day_file(root, prev_day)
286	        if prev_resolved is not None and prev_resolved[1] != day_schema:
292	            matching = next(
293	                (
294	                    (root / prev_day.isoformat() / filename, schema)
295	                    for filename, schema in DAY_FILE_PRIORITY
296	                    if schema == day_schema
297	                    and (root / prev_day.isoformat() / filename).exists()
298	                ),
299	                None,
300	            )
301	            if matching is not None:
302	                prev_resolved = matching
303	            else:
304	                warnings.append(
305	                    cls._warning(
306	                        DataQualityCode.MISSING_PRIOR_DAY_FILE,
307	                        "prior-day file schema differs; trading-day window served from a single file",
315	                prev_resolved = None
316	        if prev_resolved is None:
327	            return cls(
328	                paths=(day_path,),
330	                schema=day_schema,
333	                path_windows=((start, end),),
335	            )
336	        return cls(
337	            paths=(prev_resolved[0], day_path),
339	            schema=day_schema,
342	            path_windows=((start, split), (split, end)),
344	        )
```

**Precisely what `for_trading_day` does at an era boundary when the calendar-prior day is a different
schema**, for `trading_day = D`, `prev_day = D − 1 calendar day` (`:279` — calendar prior, no
weekend/holiday skipping):

1. `:273` resolve day D by priority; `FileNotFoundError` if nothing (`:274-277`).
2. `:284` resolve `D−1` **by priority** (so on an overlap day, mbp10 wins — E1).
3. `:286` if the prior file's schema differs from `day_schema`, scan `DAY_FILE_PRIORITY` for the first
   pair whose `schema == day_schema` AND whose file exists under `root/<prev_day>/` (`:292-300`).
   - **Found** (`:301-302`): `prev_resolved = matching`, **no warning**, and the method returns a
     TWO-path source `paths=(prev_matching_path, day_path)`, `schema=day_schema`,
     `path_windows=((start, split), (split, end))` — the full canonical
     `[D−1 18:00 ET, D 18:00 ET)` window split at UTC midnight of D.
   - **Not found** (`:303-315`): a `MISSING_PRIOR_DAY_FILE` warning carrying `trading_day`,
     `prior_day`, `prior_schema`, `schema` is appended, `prev_resolved = None`, and the method returns
     a SINGLE-path source `paths=(day_path,)`, `path_windows=((start, end),)` — the prior-evening
     hours simply produce no events.
4. The source's reported schema is always `day_schema` (`:330`, `:339`); the adopted prior file is
   decoded with the day's schema, which is safe precisely because the fallback only adopts a file whose
   priority schema equals `day_schema`.

Applied to this store's actual boundary: for D = 2026-02-23 (mbp-1 only), `day_schema = "mbp-1"`;
`prev_day = 2026-02-22` resolves to `mbp10.parquet` → mismatch at `:286`; the fallback finds
`2026-02-22/mbp1.parquet` → two-file composition, **no warning**. Before `b21316e` the same call
emitted `MISSING_PRIOR_DAY_FILE` and degraded to a single-file scan.

Scope caveat, stated as fact: the fallback inspects only `root/<calendar prev_day>/` (`:279`, `:294`,
`:297`). If that directory has no schema-matching file (weekend/holiday gap, or an mbp10-only prior
dir), the code degrades — it does not walk further back.

Reachability into TL: the pin is `TL\backend\pyproject.toml:18`
(`strategy-core @ ...@9d4935346bf42c5d19916e05dbe21dd67c46875c`);
`git merge-base --is-ancestor b21316e 9d49353` is true, `git log --oneline b21316e~1..9d49353` is
exactly `9d49353` + `b21316e`, and `git log --oneline b21316e..9d49353 -- src/strategy_core/data/databento_parquet.py src/strategy_core/data/prior_day.py`
is empty — the pinned SC carries the fallback verbatim. TL consumer:
`backend/src/trade_lab/adapters/historical_parquet.py:74-89`. (Part A independently verified the
installed `site-packages` copy is byte-identical to the SC working tree.)

## E3. TL replay-catalog mbp-1 gate at `58a5a85`

Real path: `TL\backend\src\trade_lab\adapters\replay_catalog.py`. Provenance, `git show 58a5a85`:

```
commit 58a5a85feaea44c3e00ad28a49c1c4e202d7e18b
Date:   Sat Jul 11 03:35:24 2026 -0500

    fix(replay): mbp-1 live-column gate accepts level-00 depth-suffixed TOB names - ingested mbp1 era discoverable (INGEST close-verify)

 backend/src/trade_lab/adapters/replay_catalog.py |  7 +++--
 backend/tests/test_replay_catalog.py             | 35 ++++++++++++++++++++++++
```

```diff
     if schema in {"mbp-1", "bbo"}:
-        has_bid = any(n in names for n in ("bid_price", "bid_px", "bid"))
-        has_ask = any(n in names for n in ("ask_price", "ask_px", "ask"))
+        # INGEST close-verify fix: the batch-ingested mbp1.parquet store days
+        # carry the depth-suffixed level-00 names (bid_px_00/ask_px_00, the SC
+        # reader's TOB aliases) — same alias set the mbp-10 arm accepts.
+        has_bid = any(n in names for n in ("bid_price", "bid_px", "bid", "bid_px_00"))
+        has_ask = any(n in names for n in ("ask_price", "ask_px", "ask", "ask_px_00"))
         return "ts_event" in names and has_bid and has_ask
```

The commit's test asserts the ingested layout is discoverable — `backend/tests/test_replay_catalog.py`,
`test_ingested_mbp1_day_folder_with_level00_columns_is_discoverable`, writing
`NQ/2026-02-23/mbp1.parquet` with `bid_px_00`/`ask_px_00` and asserting
`("historical:nq:2026-02-23:mbp-1", "mbp-1")`.

Current state at TL HEAD `7a911c0` (`58a5a85` is on the branch), `replay_catalog.py:444-459`:

```python
444	def _has_required_live_columns(schema: str, names: set[str]) -> bool:
445	    if schema == "trades":
446	        return {"ts_event", "price", "size"} <= names
447	    if schema == "mbp-10":
448	        has_trade_projection = {"ts_event", "action", "price", "size"} <= names
449	        has_bid = any(n in names for n in ("bid_price", "bid_px", "bid", "bid_px_00"))
450	        has_ask = any(n in names for n in ("ask_price", "ask_px", "ask", "ask_px_00"))
451	        return has_trade_projection or ("ts_event" in names and has_bid and has_ask)
452	    if schema in {"mbp-1", "bbo"}:
453	        # INGEST close-verify fix: the batch-ingested mbp1.parquet store days
454	        # carry the depth-suffixed level-00 names (bid_px_00/ask_px_00, the SC
455	        # reader's TOB aliases) — same alias set the mbp-10 arm accepts.
456	        has_bid = any(n in names for n in ("bid_price", "bid_px", "bid", "bid_px_00"))
457	        has_ask = any(n in names for n in ("ask_price", "ask_px", "ask", "ask_px_00"))
458	        return "ts_event" in names and has_bid and has_ask
459	    return False
```

The name-gate stage of the discovery predicate, `replay_catalog.py:421-437`:

```python
421	def _schema_from_name(name: str) -> str | None:
422	    lower = name.lower()
423	    if "depth-only" in lower or "depth_only" in lower:
424	        return None
425	    if _SUPPORTED_MBP10_SCHEMA_RE.search(lower):
426	        return "mbp-10"
427	    if _is_deeper_book_name(lower):
430	        return None
431	    if "trades" in lower or "trade" in lower:
432	        return "trades"
433	    if "mbp-1" in lower or "mbp1" in lower or "cmbp-1" in lower:
434	        return "mbp-1"
435	    if "bbo" in lower or "cbbo" in lower:
436	        return "bbo"
437	    return None
```

with `replay_catalog.py:25-33` (`SUPPORTED_SCHEMAS`, `_UNSUPPORTED_DEEPER_BOOK_SCHEMA_RE =
re.compile(r"\b(?:mbo|c?mbp[-_. ]?(?:[2-9]|[1-9]\d+))\b")`, `_SUPPORTED_MBP10_SCHEMA_RE =
re.compile(r"\bc?mbp[-_. ]?10\b")`, and the traversal limits `_MAX_DIRECTORIES_VISITED = 512`,
`_MAX_DIRECTORY_DEPTH = 8`, `_MAX_DIRECTORY_ENTRIES_INSPECTED = 4096`, `_MAX_METADATA_READS = 1024`).
`_schema_from_path` matches on `resolved.relative_to(canonical_root)` (`:171-177`), so the tested
string is e.g. `"2026-01-11 mbp1.parquet"`. Later stages: `.parquet` extension (`:330-332`), parquet
footer column gate (`:363-370`), id/day-mode assignment (`:372-398`, day mode engages only for the
`<symbol_dir>/<YYYY-MM-DD>/<file>` layout).

Store enumeration (filenames only): 433 day directories at depth 1, 2343 entries, 2 root-level files
(`trades_20260214_20260224.parquet`, `trades_20260217_20260224.parquet`). Filename counts include
`310 mbp10.parquet`, `156 mbp1.parquet`, plus `ml_features*`/`ml_utility*`/`ohlcv_1m*` names which
carry no schema token and map to `None` at `:437`.

**COUNT of mbp-1 sources currently discovered: 156. Earliest 2026-01-11, latest 2026-07-10.** Applying
the predicate: `mbp1.parquet` misses the mbp-10 regex, misses the deeper-book regex (single digit `1`;
`[1-9]\d+` needs ≥2 digits), has no `trade` substring, and hits `"mbp1" in lower` → `"mbp-1"`. No other
filename in the store yields `mbp-1` or `bbo`. Each of the 156 gets a distinct
`historical:nq:<YYYY-MM-DD>:mbp-1` id and day mode engaged (`:381-386`), with
`symbol_dir = path.parent.parent`. Traversal limits do not truncate this store: 432 dirs visited ≤ 512,
2343 entries < 4096, depth 1 < 8, 468 metadata reads < 1024.

Era split: **37 days carry BOTH files (2026-01-11 .. 2026-02-22)** — the catalog advertises two sources
for those, but SC `for_trading_day` resolves the DAY file to `mbp10.parquet` regardless of which
catalog id was selected (day mode passes only `symbol_dir` + `trading_day`,
`historical_parquet.py:77-83`). **119 days are mbp-1 only (2026-02-23 .. 2026-07-10).** `mbp10.parquet`
spans 2021-12-02 .. 2026-02-22 (310 files). `trades.parquet` (the third `DAY_FILE_PRIORITY` entry) has
**0 occurrences** as a per-day file; the two `trades_*` files are flat root-level files with
`trading_day=None`.

**Seal-relevant fact:** 25 of those 156 discovered mbp-1 sources are sealed trading days
(2026-06-12..2026-07-10, per D1's direct enumeration) — the catalog's discovery surface does not
exclude the seal, and TL's configured store IS the QL store (tip block).

**NOT OBTAINABLE READ-ONLY:** whether stage 3 (`_has_required_live_columns("mbp-1", names)`) passes for
every one of the 156 files. Verifying it requires reading each parquet's footer
(`pq.ParquetFile(path).schema_arrow.names`, `:364`), and 25 of those files are sealed. The in-repo
evidence that the gate passes is the `58a5a85` commit message ("ingested mbp1 era discoverable"), its
source comment (`:453-455`), and its fixture test.

## E4. SC `prior_full_day_extremes` — is candidate selection schema-agnostic?

`C:\Users\gonza\Documents\Strategy-core\src\strategy_core\data\prior_day.py:54-112`:

```python
54	def prior_full_day_extremes(
55	    symbol_dir: Path | str,
56	    trading_day: date,
57	    *,
58	    requested_symbol: str | None = None,
59	    max_walk_days: int = 10,
60	) -> PriorDayExtremes | None:
61	    """Most recent prior store day's full-session extremes, or ``None`` if none found.
62
63	    Enumerates the store's dated day directories strictly before ``trading_day``,
64	    descending, considering at most ``max_walk_days`` candidates. For each candidate the
65	    canonical reader's trading-day stream is drained (``front_month_only`` default True)
66	    accumulating max/min of ``Trade.price_ticks``; the first candidate with at least one
67	    trade in its window wins. Directories that exist but yield no events (no day file,
68	    or no in-window trades) are empty candidates, not errors. Only strict ``YYYY-MM-DD``
69	    directory names are candidates (``date.fromisoformat`` also accepts compact and
70	    ISO-week forms that would resolve to a DIFFERENT directory name downstream). A
71	    non-positive ``max_walk_days`` walks nothing. An exhausted walk returns ``None`` —
72	    the caller's cold-start case, parity-consistent with QL's first window day.
73	    """
74	    if max_walk_days <= 0:
75	        return None
76	    root = Path(symbol_dir)
77	    if not root.is_dir():
78	        return None
79
80	    candidates: list[date] = []
81	    for entry in root.iterdir():
82	        if not entry.is_dir():
83	            continue
84	        try:
85	            day = date.fromisoformat(entry.name)
86	        except ValueError:
87	            continue
88	        if day.isoformat() != entry.name:
89	            continue  # compact/ISO-week forms parse but name a different directory
90	        if day < trading_day:
91	            candidates.append(day)
92
93	    for candidate in sorted(candidates, reverse=True)[:max_walk_days]:
94	        try:
95	            source = DatabentoParquetSource.for_trading_day(
96	                root, candidate, requested_symbol=requested_symbol
97	            )
98	        except FileNotFoundError:
99	            # A dated directory without a recognized day file — an empty candidate.
100	            continue
101	        high: int | None = None
102	        low: int | None = None
103	        for event in source.events():
104	            if isinstance(event, Trade):
105	                price = event.price_ticks
106	                if high is None or price > high:
107	                    high = price
108	                if low is None or price < low:
109	                    low = price
110	        if high is not None and low is not None:
111	            return PriorDayExtremes(source_day=candidate, high_ticks=high, low_ticks=low)
112	    return None
```

**YES — candidate selection is categorically SCHEMA-AGNOSTIC.** The selection loop (`:80-91`) touches
only directory entries and their names: `root.iterdir()` (`:81`); non-directories skipped entirely
(`:82-83`), so no `.parquet` filename is ever inspected there; the directory NAME must parse as a date
(`:84-87`) in strict `YYYY-MM-DD` form (`:88-89`); and `day < trading_day` (`:90-91`). No filename, no
glob, no schema string and no `DAY_FILE_PRIORITY` reference appears anywhere in `prior_day.py` — grep
confirms `DAY_FILE_PRIORITY` occurs only at `databento_parquet.py:75, 295, 349`, and `prior_day.py`'s
only SC imports are `DatabentoParquetSource` (`:39`) and `Trade` (`:40`).

**Files/globs considered during candidate selection: NONE.** Only the set of depth-1 subdirectory names
under `symbol_dir`. File selection is delegated wholly to the reader, one candidate at a time, at
`:95-97`, which internally applies `_resolve_day_file` over `DAY_FILE_PRIORITY` for both `candidate`
and `candidate − 1 day`, plus the `b21316e` fallback. So the effective per-candidate file set is
`{mbp10.parquet, mbp1.parquet, trades.parquet}` under `root/<candidate>/` in priority order, plus the
same three names under `root/<candidate − 1 day>/`.

Consequence at the era boundary: `prior_full_day_extremes(root, 2026-02-24)` produces candidates
`[2026-02-23, 2026-02-22, 2026-02-21, ...]` descending and tries 2026-02-23 — an mbp-1-only day —
first; `for_trading_day(root, 2026-02-23)` then takes the E2 boundary path and (post-`b21316e`)
composes `2026-02-22/mbp1.parquet` + `2026-02-23/mbp1.parquet`. The presence of `mbp10.parquet` on
2026-02-22 excludes neither the day from candidacy nor the file from composition.

Two exact behaviours, both from the quoted code:

1. `:98-100` catches only `FileNotFoundError`. A dated directory with no recognized day file is skipped
   as an empty candidate; **any other exception from `for_trading_day` or `source.events()` propagates**
   out of `prior_full_day_extremes`. TL's caller wraps it —
   `TL\backend\src\trade_lab\services\replay.py:255-260` catches `Exception` and proceeds unseeded.
2. `:103-111` accumulates over `Trade` events only. `DataQualityWarning` items on the stream —
   including `MISSING_PRIOR_DAY_FILE` — are silently ignored (`isinstance(event, Trade)` is the only
   branch). A degraded single-file window on a candidate day therefore yields extremes computed over
   only the day-file portion, with **no signal to the caller** that the prior-evening hours were absent;
   the walk returns a normal `PriorDayExtremes`.

TL consumer of the walk: `TL\backend\src\trade_lab\services\replay.py:13`
(`from strategy_core.data.prior_day import prior_full_day_extremes`) and `:233-254` (the seed banked
before the replay task starts, feeding `runtime.levels.load_prior_day_summary`).

---

## Read-only compliance statement

No tracked file in SC, TL, or QL was modified, staged, or committed. The single write is this file,
`C:\Users\gonza\Documents\Claude-Quant-Lab\ERA_GATE_RECON.md`, untracked. No dataset build, cache
warm, training run, scoring pass, or W3b replay was executed at any point, in or out of the seal. No
data file covering a trading day in 2026-06-12..2026-07-10 was opened: those 25 `mbp1.parquet` files
and the 7 sealed-range journal filenames are known here only as `(path, size, mtime)` triples from
`ls`/`find`/`stat`. Files opened were source, config, logs, and bundle metadata JSON only.
