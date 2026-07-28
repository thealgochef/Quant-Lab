"""CACHE_SHAPE_RECON PART C — per-day staged/canonical measurement driver (v2).

v2 REWRITE after the v1 run died materializing ``list(source.events())`` (~12-15M
boxed event objects). Memory discipline now:
  * NO full-stream object list, ever.
  * Stage (a) "decode+drain/normalize" LUMPED: iterate ``source.events()``
    counting Trade/Quote/DataQualityWarning, storing NOTHING (the drain-only
    baseline used for subtraction estimates).
  * Stage (b) "runtime fold" LUMPED (bar fold + level/session tracking + per-bar
    zone rebuild + touch detect inside ``runtime.process_event``,
    state.py:334-371): a SECOND fresh drain feeding each Trade straight off the
    iterator. Re-pays decode; both the lumped number and the subtraction
    estimate (fold_run - drain_only_run) are reported. During this pass the
    trade stream is captured into compact PREALLOCATED numpy arrays
    (ts int64 ns, price_ticks int64, size int32, side-code uint8 + a tiny
    side-string vocab) — never object lists.
  * Stage (c) approach-quote second pass replicates engine_decision.py:707-742
    VERBATIM (chunked vectorized window assignment) — iterating
    ``source.events()`` from DISK exactly like the canonical path (a third
    decode here; canonical pays this as its second). Only quotes inside the
    touch-approach hull are retained.
  * Stages (d)/(e) rebuild bounded Trade windows from the compact arrays via
    ``np.searchsorted`` (bisect-equivalent on epoch-ns) and construct real
    ``strategy_core.Trade`` objects only for those windows, with ``.tolist()``
    python ints and ``pd.DatetimeIndex(ns, tz="UTC")`` timestamps — the exact
    object types the reader's ``emit_deduped`` produces
    (databento_parquet.py:162-191, :624-625).

READ-ONLY against the databento store; all outputs go to
C:\\Users\\gonza\\Documents\\Claude-Quant-Lab\\_scratch_timing\\.

Cache bypass: cache read/write lives ONLY in build_utility_dataset
(dashboard_utility_builder.py:155-200); process_single_date_stream and this
replica never touch ml_utility_*.parquet. The only parquet written is the
redirected rows file under _scratch_timing (production serializer
_write_day_cache pointed at the redirect path).

Peak RSS: ctypes GetProcessMemoryInfo -> PeakWorkingSetSize with explicit
argtypes/restype (Win64 silently returns 0 otherwise). Sampled at every stage
boundary. Any death writes the full traceback into the progressive result JSON
("fatal") before exiting nonzero.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
import traceback
from ctypes import wintypes
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

_QL_ROOT = Path(__file__).resolve().parent
for _p in (str(_QL_ROOT / "src"), str(_QL_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SCRATCH = Path(r"C:\Users\gonza\Documents\Claude-Quant-Lab\_scratch_timing")

_NS_30MIN = 30 * 60 * 1_000_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


# ── Peak RSS via the Windows API (explicit argtypes/restype — Win64 gotcha) ──
class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_kernel32 = ctypes.WinDLL("kernel32")
_psapi = ctypes.WinDLL("psapi")
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_psapi.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(_ProcessMemoryCounters),
    wintypes.DWORD,
]
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def _mem() -> tuple[int, int]:
    """(WorkingSetSize, PeakWorkingSetSize) in bytes for this process."""
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
    ok = _psapi.GetProcessMemoryInfo(
        _kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if not ok:
        return (0, 0)
    return (int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize))


class _Recorder:
    def __init__(self, out_path: Path) -> None:
        self.out_path = out_path
        self.results: dict = {"stages": {}, "mem_at": {}}

    def stage(self, name: str, seconds: float, **extra) -> None:
        self.results["stages"][name] = {"seconds": seconds, **extra}
        ws, peak = _mem()
        self.results["mem_at"][name] = {"working_set": ws, "peak_working_set": peak}
        self.dump()

    def note(self, key: str, value) -> None:
        self.results[key] = value
        self.dump()

    def dump(self) -> None:
        tmp = str(self.out_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str)
        os.replace(tmp, self.out_path)


def _ts_ns(ts) -> int:
    """Exact epoch-ns for a tz-aware datetime/pd.Timestamp (no float round-trip)."""
    value = getattr(ts, "value", None)
    if value is not None:  # pd.Timestamp — the reader's ns-unit make_ts objects
        return int(value)
    delta = ts - _EPOCH  # exact timedelta arithmetic (µs-domain datetimes)
    return ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000


def _seed_walk(data_dir: Path, symbol: str, window_dates: list[str], day: str, util_cfg):
    """The warmer's seed logic: walk back from the immediate predecessor to the
    most recent NON-EMPTY prior window day via the SAME bars-based helper
    (_get_session_hl_for_date, dashboard_utility_builder.py:235-254)."""
    from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
        _get_session_hl_for_date,
    )

    idx = window_dates.index(day)
    probes = []
    for k in range(idx - 1, -1, -1):
        d = window_dates[k]
        hl = _get_session_hl_for_date(data_dir, symbol, d, util_cfg, None)
        probes.append({"day": d, "hl": None if hl is None else list(hl)})
        if hl is not None:
            return hl, d, probes
    return None, None, probes


def _run(rec: _Recorder, args, process_t0: float) -> int:
    day = args.day

    t = time.perf_counter()
    import numpy as np
    import pandas as pd
    import strategy_core as sc
    from strategy_core import (
        HonestEntryDrop,
        Quote,
        Trade,
        app_avg_trade_size,
        app_large_trade_vol_pct,
        app_max_spread,
        classify_session,
        int_absorption_ratio,
        int_time_beyond_level,
        int_time_within_2pts,
        resolve_honest_outcome,
    )
    from strategy_core.constants import RTH_END
    from strategy_core.data.databento_parquet import DatabentoParquetSource
    from strategy_core.data.events import DataQualityWarning as ScDQW
    from strategy_core.runtime.state import StrategyRuntime

    from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig
    from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import _write_day_cache
    from alpha_lab.agents.data_infra.ml.engine_decision import (
        _QUOTE_CHUNK_EVENTS,
        TRADE_TICK,
        _round_to_ticks,
    )
    rec.stage("imports (untimed-stage bookkeeping)", time.perf_counter() - t,
              strategy_core_file=sc.__file__)

    handoff = json.loads((_SCRATCH / "handoff.json").read_text(encoding="utf-8"))
    data_dir = Path(handoff["data_dir"])
    symbol = handoff["symbol"]
    util_cfg = DashboardUtilityConfig(**handoff["util_kwargs"])
    window_dates = handoff["window_dates"][day]
    td = date.fromisoformat(day)
    tick_count = int(util_cfg.bar_type[:-1])
    _ET = "US/Eastern"

    # ── seed (prior-day extremes) — its own labeled timed line ────────────────
    t = time.perf_counter()
    seed, seed_day, probes = _seed_walk(data_dir, symbol, window_dates, day, util_cfg)
    rec.stage("seed (prior-day extremes)", time.perf_counter() - t,
              seed=None if seed is None else list(seed), seed_day=seed_day,
              probes=probes)

    if args.mode == "canonical":
        from alpha_lab.agents.data_infra.ml.engine_decision import (
            process_single_date_stream,
        )

        gc.collect()
        ws0, peak0 = _mem()
        t = time.perf_counter()
        df = process_single_date_stream(day, data_dir, symbol, util_cfg, prev_day_hl=seed)
        total_s = time.perf_counter() - t
        rec.stage("canonical process_single_date_stream total", total_s,
                  labeled_rows=int(len(df)), ws_before=ws0, peak_before=peak0)

        canon_path = _SCRATCH / f"canonical_rows_{day}.parquet"
        t = time.perf_counter()
        _write_day_cache(df, canon_path, seed)
        rec.stage("canonical serialize (redirected _write_day_cache)",
                  time.perf_counter() - t, path=str(canon_path),
                  bytes=os.path.getsize(canon_path))

        staged_path = _SCRATCH / f"staged_rows_{day}.parquet"
        cmp: dict = {"staged_path": str(staged_path)}
        if staged_path.exists():
            a = pd.read_parquet(staged_path)
            b = pd.read_parquet(canon_path)
            cmp["staged_rows"] = int(len(a))
            cmp["canonical_rows"] = int(len(b))
            cmp["columns_equal"] = list(a.columns) == list(b.columns)
            cmp["frame_equals"] = bool(a.equals(b))
            if not cmp["frame_equals"] and cmp["columns_equal"] and len(a) == len(b):
                diffs = {}
                for c in a.columns:
                    if not a[c].equals(b[c]):
                        diffs[c] = "differs"
                cmp["differing_columns"] = diffs
        else:
            cmp["staged_rows"] = None
            cmp["note"] = "staged parquet missing at compare time"
        rec.note("cross_check", cmp)
        rec.note("process_wall_s", time.perf_counter() - process_t0)
        ws, peak = _mem()
        rec.note("final_mem", {"working_set": ws, "peak_working_set": peak})
        print(f"CANONICAL {day} total={total_s:.2f}s rows={len(df)}")
        return 0

    # ── staged mode ───────────────────────────────────────────────────────────
    t = time.perf_counter()
    source = DatabentoParquetSource.for_trading_day(
        data_dir / symbol, td, requested_symbol=symbol
    )
    construct_s = time.perf_counter() - t
    file_info = [{"path": str(p), "bytes": os.path.getsize(p)} for p in source.paths]
    rec.stage("source construction (for_trading_day; file resolution only)",
              construct_s, files=file_info, schema=source.schema,
              pending_warnings=[w.message for w in source.pending_warnings])

    runtime = StrategyRuntime(timeframes=(tick_count,), requested_symbol=symbol)
    if seed is not None:
        runtime.load_prior_day_summary(
            td - timedelta(days=1),
            high_ticks=_round_to_ticks(seed[0], TRADE_TICK),
            low_ticks=_round_to_ticks(seed[1], TRADE_TICK),
        )

    # (a) decode + drain/normalize LUMPED, STORING NOTHING: per-batch
    # column-pruned pyarrow decode, validation, window masks, canonical ordering,
    # L1 TOB dedup and event-object construction are generator-fused per batch in
    # _decode_batches/emit_deduped (databento_parquet.py:493-574, :131-192); no
    # separable public decode-only step exists. Includes the front-month prescan
    # (:1045-1102). This drain-only wall time is the subtraction baseline.
    gc.collect()
    ws_before, _ = _mem()
    t = time.perf_counter()
    n_trades = 0
    n_quotes = 0
    n_warnings = 0
    for event in source.events():
        if isinstance(event, Trade):
            n_trades += 1
        elif isinstance(event, Quote):
            n_quotes += 1
        elif isinstance(event, ScDQW):
            n_warnings += 1
    drain_s = time.perf_counter() - t
    ws_after, _ = _mem()
    rec.stage("(a) decode+drain/normalize LUMPED, store-nothing drain", drain_s,
              n_items=n_trades + n_quotes + n_warnings, n_trades=n_trades,
              n_quotes_after_tob_dedup=n_quotes, n_warnings=n_warnings,
              ws_before=ws_before, ws_after=ws_after)

    # (b) runtime fold LUMPED (bar fold + level/session tracking + per-bar zone
    # rebuild + touch detect, all inside runtime.process_event ->
    # state.py:334-371 -> plugin.on_event/on_bar_closed), fed straight off a
    # SECOND fresh drain — so this stage RE-PAYS the decode; the subtraction
    # estimate (b - a) is reported alongside. The trade stream is captured into
    # compact preallocated numpy arrays during this same drive pass (ts int64
    # epoch-ns via pd.Timestamp.value — exact; price_ticks int64; size int32;
    # side uint8 vocab code). Capture overhead is INCLUDED in this stage's time.
    gc.collect()
    ws_before, _ = _mem()
    tr_ts = np.zeros(n_trades, dtype=np.int64)
    tr_px = np.zeros(n_trades, dtype=np.int64)
    tr_sz = np.zeros(n_trades, dtype=np.int32)
    tr_sd = np.zeros(n_trades, dtype=np.uint8)
    side_vocab: list = [None]
    side_codes: dict = {None: 0}
    day_bars: list = []
    touches: list = []
    fill = 0
    t = time.perf_counter()
    for event in source.events():
        if not isinstance(event, Trade):
            continue
        tr_ts[fill] = _ts_ns(event.event_ts_utc)
        tr_px[fill] = event.price_ticks
        tr_sz[fill] = event.size
        code = side_codes.get(event.side)
        if code is None:
            code = len(side_vocab)
            if code > 255:
                raise RuntimeError("side vocab overflow (uint8)")
            side_codes[event.side] = code
            side_vocab.append(event.side)
        tr_sd[fill] = code
        fill += 1
        update = runtime.process_event(event)
        for bar in update.closed_bars:
            if bar.timeframe_ticks == tick_count:
                day_bars.append(bar)
        touches.extend(update.touches)
    fold_s = time.perf_counter() - t
    ws_after, _ = _mem()
    if fill != n_trades:
        raise RuntimeError(f"drain/fold trade-count drift: {n_trades} vs {fill}")
    arrays_nbytes = int(tr_ts.nbytes + tr_px.nbytes + tr_sz.nbytes + tr_sd.nbytes)
    mean_bar_id_len = (
        sum(len(b.bar_id) for b in day_bars) / len(day_bars) if day_bars else None
    )
    close_reasons: dict = {}
    for b in day_bars:
        key = None if b.close_reason is None else str(b.close_reason.value)
        close_reasons[key] = close_reasons.get(key, 0) + 1
    rec.stage("(b) runtime fold LUMPED (bar fold + level/session + per-bar zone"
              " rebuild + touch detect; re-pays decode; incl. compact-array"
              " capture)", fold_s,
              fold_minus_drain_estimate_s=fold_s - drain_s,
              n_completed_bars=len(day_bars), n_touches=len(touches),
              mean_bar_id_len=mean_bar_id_len,
              bar_id_total_chars=sum(len(b.bar_id) for b in day_bars),
              close_reasons=close_reasons,
              side_vocab=[str(s) for s in side_vocab],
              compact_arrays_nbytes=arrays_nbytes,
              ws_before=ws_before, ws_after=ws_after,
              ws_delta=ws_after - ws_before)

    # zone count: end-of-day zones over the FINAL level state via
    # runtime.snapshot() -> plugin.snapshot_zones(trading_day) (plugin.py:302) ->
    # build_zones(list(levels())) at default ZONE_PROXIMITY_PTS 3.0
    # (levels.py:110-111, zones.py:23-106, constants.py:63), fired zones
    # pre-marked. Timed as a snapshot, NOT the per-bar fused detection cost.
    t = time.perf_counter()
    snap = runtime.snapshot()
    rec.stage("zone build (end-of-day snapshot via runtime.snapshot)",
              time.perf_counter() - t,
              n_zones=len(snap.zones), n_levels=len(snap.levels),
              n_current_partial_bars=len(snap.current_bars),
              zones=[{
                  "representative_price": z.representative_price,
                  "names": list(z.names), "side": z.side.value,
                  "touched": z.touched,
              } for z in snap.zones])

    # (c) approach-quote second pass: engine_decision.py:697-742 replicated
    # VERBATIM including the from-DISK re-iteration of source.events() (the
    # canonical path's bounded second decode). Retains ONLY quotes inside some
    # touch's approach window. Includes a full decode; (c - a) reported as the
    # assignment-only subtraction estimate.
    window_minutes = util_cfg.interaction_window_minutes
    decision_offset = window_minutes
    interaction_window = timedelta(minutes=window_minutes)
    approach_window = timedelta(minutes=util_cfg.approach_window_minutes)

    gc.collect()
    ws_before, _ = _mem()
    t = time.perf_counter()
    approach_quotes: dict[int, list] = {}
    if util_cfg.include_approach_features and touches:
        starts_ns = np.array(
            [pd.Timestamp(t_.bar_ts_utc - approach_window).value for t_ in touches],
            dtype=np.int64,
        )
        ends_ns = np.array(
            [pd.Timestamp(t_.bar_ts_utc).value for t_ in touches], dtype=np.int64
        )
        hull_start = min(t_.bar_ts_utc - approach_window for t_ in touches)
        hull_end = max(t_.bar_ts_utc for t_ in touches)
        chunk: list = []

        def _assign_chunk() -> None:
            ts_ns_arr = pd.DatetimeIndex([q.event_ts_utc for q in chunk]).asi8
            inside = (ts_ns_arr[:, None] >= starts_ns[None, :]) & (
                ts_ns_arr[:, None] < ends_ns[None, :]
            )
            for touch_idx in np.flatnonzero(inside.any(axis=0)):
                approach_quotes.setdefault(int(touch_idx), []).extend(
                    chunk[pos] for pos in np.flatnonzero(inside[:, touch_idx])
                )
            chunk.clear()

        for event in source.events():
            if not isinstance(event, Quote):
                continue
            ts = event.event_ts_utc
            if ts < hull_start or ts >= hull_end:
                continue
            chunk.append(event)
            if len(chunk) >= _QUOTE_CHUNK_EVENTS:
                _assign_chunk()
        if chunk:
            _assign_chunk()
    quotes_pass_s = time.perf_counter() - t
    ws_after, _ = _mem()
    rec.stage("(c) approach-quote second pass from DISK (canonical replica;"
              " includes a full decode)", quotes_pass_s,
              c_minus_drain_estimate_s=quotes_pass_s - drain_s,
              n_touches_with_quotes=len(approach_quotes),
              n_retained_quote_refs=sum(len(v) for v in approach_quotes.values()),
              ws_before=ws_before, ws_after=ws_after,
              ws_delta=ws_after - ws_before)

    # Bounded-window reconstruction helpers: np.searchsorted on epoch-ns ==
    # bisect_left/bisect_right on the canonical trade_ts list (the stream is
    # canonically ordered, ns-exact). Reconstructed Trades use .tolist() python
    # ints + pd.DatetimeIndex(ns, tz="UTC") timestamps — the exact types
    # emit_deduped produced, so feature/entry math is bit-identical.
    def _price_at(ts_utc: datetime):
        key = _ts_ns(ts_utc)
        index = int(np.searchsorted(tr_ts, key, side="right"))
        if index == 0:
            return None
        if key - int(tr_ts[index - 1]) > _NS_30MIN:
            return None
        return int(tr_px[index - 1]) * TRADE_TICK

    def _trades_in(start: datetime, end: datetime) -> list:
        i0 = int(np.searchsorted(tr_ts, _ts_ns(start), side="left"))
        i1 = int(np.searchsorted(tr_ts, _ts_ns(end), side="left"))
        if i1 <= i0:
            return []
        ts_objects = list(pd.DatetimeIndex(tr_ts[i0:i1], tz="UTC"))
        px = tr_px[i0:i1].tolist()
        sz = tr_sz[i0:i1].tolist()
        sd = tr_sd[i0:i1].tolist()
        return [
            Trade(event_ts_utc=ts_objects[k], price_ticks=px[k], size=sz[k],
                  side=side_vocab[sd[k]])
            for k in range(len(px))
        ]

    # (d) forward resolution (engine_decision.py:745-759; the resolve calls only)
    t = time.perf_counter()
    resolved = []
    n_honest_drops = 0
    n_no_resolution = 0
    for idx, touch in enumerate(touches):
        result = resolve_honest_outcome(
            touch,
            day_bars,
            _price_at,
            tick_size=TRADE_TICK,
            tp_points=util_cfg.tp_points,
            sl_points=util_cfg.sl_points,
            trap_mfe_min=util_cfg.trap_mfe_min,
            decision_offset_minutes=decision_offset,
        )
        if isinstance(result, HonestEntryDrop):
            n_honest_drops += 1
            continue
        if result.label == sc.constants.NO_RESOLUTION:
            n_no_resolution += 1
            continue
        resolved.append((idx, touch, result))
    resolve_s = time.perf_counter() - t
    rec.stage("(d) forward resolution (resolve_honest_outcome per touch)",
              resolve_s, n_resolved=len(resolved),
              n_honest_entry_drops=n_honest_drops,
              n_no_resolution=n_no_resolution)

    # (e) feature computation + row build (engine_decision.py:761-825)
    t = time.perf_counter()
    rows = []
    n_few_trades_drops = 0
    for idx, touch, result in resolved:
        interaction_trades = _trades_in(
            touch.bar_ts_utc, touch.bar_ts_utc + interaction_window
        )
        if len(interaction_trades) < 5:
            n_few_trades_drops += 1
            continue
        level_points = float(touch.representative_price)
        features = {
            "int_time_beyond_level": int_time_beyond_level(
                interaction_trades, level_points, touch.direction, TRADE_TICK
            ),
            "int_time_within_2pts": int_time_within_2pts(
                interaction_trades, level_points, TRADE_TICK
            ),
            "int_absorption_ratio": int_absorption_ratio(
                interaction_trades, level_points, touch.direction, TRADE_TICK,
                proximity_pts=util_cfg.level_proximity_pts,
            ),
        }
        bar_ts_et = pd.Timestamp(touch.bar_ts_utc).tz_convert(_ET)
        session_info = classify_session(touch.bar_ts_utc)
        decision_time_et = pd.Timestamp(
            touch.bar_ts_utc + timedelta(minutes=decision_offset)
        ).tz_convert(_ET)
        entry_price = _price_at(touch.bar_ts_utc + timedelta(minutes=decision_offset))
        row = {
            "event_ts": bar_ts_et,
            "date": day,
            "timestamp": bar_ts_et,
            "session": session_info.session,
            "decision_time": decision_time_et,
            "label_window_end": pd.Timestamp(
                f"{day} {RTH_END.strftime('%H:%M:%S')}", tz=_ET
            ),
            "direction": touch.direction.value,
            "representative_price": touch.representative_price,
            "level_type": touch.level_type,
            "label": result.label,
            "label_encoded": result.label_encoded,
            "max_mfe": result.max_mfe,
            "max_mae": result.max_mae,
            "entry_price": None if entry_price is None else float(entry_price),
        }
        row.update(features)
        if util_cfg.include_approach_features:
            approach_trades = _trades_in(
                touch.bar_ts_utc - approach_window, touch.bar_ts_utc
            )
            row.update(
                {
                    "app_large_trade_vol_pct": app_large_trade_vol_pct(approach_trades),
                    "app_avg_trade_size": app_avg_trade_size(approach_trades),
                    "app_max_spread": app_max_spread(
                        approach_quotes.get(idx, ()), TRADE_TICK
                    ),
                }
            )
        rows.append(row)
    features_s = time.perf_counter() - t
    rec.stage("(e) feature computation + row build", features_s,
              n_labeled_rows=len(rows), n_few_trades_drops=n_few_trades_drops)

    # (f) serialize/write — the production serializer redirected to _scratch_timing
    t = time.perf_counter()
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    staged_path = _SCRATCH / f"staged_rows_{day}.parquet"
    _write_day_cache(df, staged_path, seed)
    ser_s = time.perf_counter() - t
    rec.stage("(f) serialize/write labeled rows (redirected _write_day_cache)",
              ser_s, path=str(staged_path), bytes=os.path.getsize(staged_path))

    rec.note("volumes", {
        "files": file_info,
        "schema": source.schema,
        "drained_items_total": n_trades + n_quotes + n_warnings,
        "drained_trades": n_trades,
        "drained_quotes_after_tob_dedup": n_quotes,
        "drained_warnings": n_warnings,
        "completed_bars_147t": len(day_bars),
        "n_current_partial_bars": len(snap.current_bars),
        "mean_bar_id_len": mean_bar_id_len,
        "zone_count_end_of_day": len(snap.zones),
        "level_count_end_of_day": len(snap.levels),
        "touch_count": len(touches),
        "labeled_row_count": len(rows),
        "drops": {
            "honest_entry": n_honest_drops,
            "no_resolution": n_no_resolution,
            "lt5_interaction_trades": n_few_trades_drops,
        },
    })
    rec.note("process_wall_s", time.perf_counter() - process_t0)
    ws, peak = _mem()
    rec.note("final_mem", {"working_set": ws, "peak_working_set": peak})
    print(f"STAGED {day} drain={drain_s:.2f}s fold={fold_s:.2f}s "
          f"quotes={quotes_pass_s:.2f}s rows={len(rows)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", required=True)
    parser.add_argument("--mode", choices=["staged", "canonical"], required=True)
    args = parser.parse_args()

    out_path = _SCRATCH / f"result_{args.mode}_{args.day}.json"
    rec = _Recorder(out_path)
    process_t0 = time.perf_counter()
    rec.note("meta", {
        "mode": args.mode,
        "day": args.day,
        "pid": os.getpid(),
        "python": sys.version,
        "argv": sys.argv,
        "started_utc": datetime.now(UTC).isoformat(),
    })
    try:
        return _run(rec, args, process_t0)
    except BaseException as exc:
        gc.collect()
        ws, peak = _mem()
        rec.note("fatal", {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "working_set": ws,
            "peak_working_set": peak,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
