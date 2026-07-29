"""The sequential seeded capture chain over the store (Phase C orchestrator).

Chronological ``run_day`` chain with hash-chained per-day trust:

* a day's capture parquet is consumed only if its stamped ENTERING seed hash
  equals the hash of the seed the chain is actually carrying into that day
  (and its stored end-seed pickle re-hashes to the stamped end hash) — a
  broken/stale middle link therefore invalidates every later day, never
  silently (the ``prev_full_hl`` lesson generalized);
* artifacts are chain-verified too: each day's artifact must have been built
  with the PRIOR day's outputs as seeds (day H/L + NY H/L); a mismatch
  rebuilds the artifact in place;
* the first ``cfg.warmup_days`` chain days are flagged ``is_warmup`` and every
  row carries ``days_of_htf_history`` (training filters are a choice, not a
  rebuild — census: the 4H registry stabilizes days 8-10).
"""

from __future__ import annotations

import json
import pickle
from datetime import date

import pandas as pd
from strategy_core.strategies.ifvg_smc.state import IfvgDaySeed, seed_hash

from .capture_driver import capture_single_date
from .config import IfvgCaptureConfig
from .day_artifacts import (
    DayArtifacts,
    DaySeeds,
    build_day_artifacts,
    load_day_artifacts,
    seeds_for_day,
    write_day_artifacts,
)

__all__ = [
    "build_ifvg_capture",
    "CaptureChainResult",
    "CAPTURE_UNION_SCHEMA",
    "conform_capture_frame",
]

_CAP_META_KEY = b"ifvg_capture_meta"

#: The FULL flattened v1-record union (column -> pandas dtype), frozen from the
#: pandas-2 concat of the capture chain. Per-day frames are a union of per-kind
#: record columns, so on any given day most columns are absent or all-NA;
#: pandas 2 excluded such entries from concat result-dtype inference, pandas 3
#: does not — so the union dtype is DECLARED here and every per-day frame is
#: conformed to it before concat (inference-free under either pandas). A column
#: outside this schema means the record shape drifted: bump
#: ``IFVG_RECORD_SCHEMA_VERSION`` and re-freeze; never widen silently.
CAPTURE_UNION_SCHEMA: dict[str, str] = {
    "kind": "object",
    "entering_seed_hash": "object",
    "envelope_schema_version": "int64",
    "envelope_strategy_id": "object",
    "envelope_strategy_version": "object",
    "envelope_profile_hash": "object",
    "envelope_trading_day": "object",
    "envelope_ts_utc": "datetime64[ns, UTC]",
    "envelope_setup_id": "object",
    "htf_tf_seconds": "float64",
    "fvg_fvg_id": "object",
    "fvg_timeframe_seconds": "float64",
    "fvg_direction": "object",
    "fvg_gap_low_ticks": "float64",
    "fvg_gap_high_ticks": "float64",
    "fvg_size_ticks": "float64",
    "fvg_a_bar_id": "object",
    "fvg_c_bar_id": "object",
    "fvg_a_open_ts_utc": "datetime64[ns, UTC]",
    "fvg_confirmed_ts_utc": "datetime64[ns, UTC]",
    "fvg_trading_day": "object",
    "direction": "object",
    "penetration_ticks": "float64",
    "ce_reached": "object",
    "htf_age_seconds": "float64",
    "remaining_fraction": "float64",
    "registry_live_count": "float64",
    "rank": "float64",
    "conflicted": "object",
    "nearest_level_kind": "object",
    "nearest_level_distance_ticks": "float64",
    "session_engine": "object",
    "session_doc": "object",
    "selected": "object",
    "drop_reason": "object",
    "resolution": "object",
    "entry_family": "object",
    "entry_ticks": "float64",
    "stop_ticks": "float64",
    "tp_ticks": "float64",
    "mfe_ticks": "float64",
    "mae_ticks": "float64",
    "bars_in_trade": "float64",
    "tap_ts_utc": "datetime64[ns, UTC]",
    "parent_confirmed_ts_utc": "datetime64[ns, UTC]",
    "lock_ts_utc": "datetime64[ns, UTC]",
    "armed_ts_utc": "datetime64[ns, UTC]",
    "inversion_ts_utc": "datetime64[ns, UTC]",
    "entry_ts_utc": "datetime64[ns, UTC]",
    "htf_fvg_id": "object",
    "parent_fvg_id": "object",
    "opposing_fvg_id": "object",
    "is_warmup": "bool",
    "days_of_htf_history": "int64",
    "parent_tf_seconds": "float64",
    "distance_to_htf_ticks": "float64",
    "elapsed_1m_bars_since_tap": "float64",
    "confirmed_after": "object",
    "fully_formed_after": "object",
    "elapsed_1m_bars_since_selection": "float64",
    "distance_to_parent_ticks": "float64",
    "elapsed_1m_bars_since_lock": "float64",
    "close_through_margin_ticks": "float64",
    "bars_armed_to_inversion": "float64",
    "opposing_size_ticks": "float64",
    "sweep_sweep_confirmed": "object",
    "sweep_swept_kinds": "object",
    "sweep_max_penetration_ticks": "float64",
    "sweep_nearest_unswept_distance_ticks": "float64",
    "sweep_sweep_ts_utc": "datetime64[ns, UTC]",
    "sweep_leg_extreme_ticks": "float64",
    "entry_fvg": "float64",
    "entry_fvg_fvg_id": "object",
    "entry_fvg_timeframe_seconds": "float64",
    "entry_fvg_direction": "object",
    "entry_fvg_gap_low_ticks": "float64",
    "entry_fvg_gap_high_ticks": "float64",
    "entry_fvg_size_ticks": "float64",
    "entry_fvg_a_bar_id": "object",
    "entry_fvg_c_bar_id": "object",
    "entry_fvg_a_open_ts_utc": "datetime64[ns, UTC]",
    "entry_fvg_confirmed_ts_utc": "datetime64[ns, UTC]",
    "entry_fvg_trading_day": "object",
    "risk_ticks": "float64",
    "bars_since_inversion": "float64",
    "entry_to_parent_ticks": "float64",
    "in_engine_session": "object",
    "in_doc_session": "object",
}

#: dtypes that cannot represent NA — a per-day frame missing one of these
#: columns is malformed, not sparse (they are stamped on every row).
_NON_NULLABLE = ("bool", "int64")


def conform_capture_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reindex/astype one per-day capture frame to ``CAPTURE_UNION_SCHEMA``.

    Absent and all-NA columns get a typed-empty column (their inferred dtype is
    pure noise — the day simply never emitted the field); populated columns are
    cast, so a genuine value/dtype conflict still raises."""
    unknown = [c for c in frame.columns if c not in CAPTURE_UNION_SCHEMA]
    if unknown:
        raise ValueError(f"capture frame columns outside CAPTURE_UNION_SCHEMA: {unknown}")
    out: dict[str, pd.Series] = {}
    for col, dtype in CAPTURE_UNION_SCHEMA.items():
        present = col in frame.columns
        if present and (str(frame[col].dtype) == dtype or not frame[col].isna().all()):
            series = frame[col]
            if str(series.dtype) != dtype:
                series = series.astype(dtype)
        else:
            if dtype in _NON_NULLABLE and present is False:
                raise ValueError(f"non-nullable column {col!r} missing from capture frame")
            if dtype in _NON_NULLABLE:
                series = frame[col].astype(dtype)
            else:
                series = pd.Series(index=frame.index, dtype=dtype)
        out[col] = series
    return pd.DataFrame(out, index=frame.index)


class CaptureChainResult:
    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []
        self.day_funnels: dict[str, dict[str, int]] = {}
        self.rebuilt_days: list[str] = []
        self.cached_days: list[str] = []

    def frame(self) -> pd.DataFrame:
        real = [conform_capture_frame(f) for f in self.frames if len(f)]
        if not real:
            return pd.DataFrame()
        out = pd.concat(real, ignore_index=True)
        drift = {
            c: (str(out[c].dtype), CAPTURE_UNION_SCHEMA[c])
            for c in out.columns
            if str(out[c].dtype) != CAPTURE_UNION_SCHEMA[c]
        }
        if drift:
            raise ValueError(f"capture concat dtypes drifted from schema (got, want): {drift}")
        return out


def _write_capture(
    frame: pd.DataFrame,
    cfg: IfvgCaptureConfig,
    date_str: str,
    *,
    entering_hash: str | None,
    end_hash: str,
    funnel: dict[str, int],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    meta = {
        "entering_seed_hash": entering_hash,
        "end_seed_hash": end_hash,
        "profile_hash": cfg.profile_hash,
        "funnel": funnel,
    }
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), _CAP_META_KEY: json.dumps(meta, sort_keys=True).encode()}
    )
    path = cfg.capture_path(date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _load_capture(
    cfg: IfvgCaptureConfig, date_str: str, *, expected_entering: str | None
) -> tuple[pd.DataFrame, dict[str, int], IfvgDaySeed] | None:
    import pyarrow.parquet as pq

    cpath = cfg.capture_path(date_str)
    spath = cfg.seed_path(date_str)
    if not cpath.exists() or not spath.exists():
        return None
    try:
        md = pq.read_metadata(cpath).metadata or {}
        meta = json.loads(md.get(_CAP_META_KEY, b"{}"))
    except Exception:
        return None
    if meta.get("entering_seed_hash") != expected_entering:
        return None
    try:
        with open(spath, "rb") as fh:
            end_seed = pickle.load(fh)
    except Exception:
        return None
    if seed_hash(end_seed) != meta.get("end_seed_hash"):
        return None
    return pd.read_parquet(cpath), dict(meta.get("funnel", {})), end_seed


def _chained_seeds(prev: DayArtifacts | None) -> DaySeeds | None:
    """The artifact seeds day N MUST have been built with, given day N-1's
    artifact (None = no expectation, first chain day)."""
    if prev is None:
        return None
    prev_day = date.fromisoformat(prev.date_str)
    if prev.day_hl is None:
        # Empty prior day: expectations carry through from ITS seeds.
        return prev.seeds
    return DaySeeds(
        prev_day=prev_day,
        prev_full_hl=prev.day_hl,
        prev_ny_day=prev_day if prev.ny_hl else prev.seeds.prev_ny_day,
        prev_ny_hl=prev.ny_hl if prev.ny_hl else prev.seeds.prev_ny_hl,
    )


def build_ifvg_capture(
    dates: list[str], cfg: IfvgCaptureConfig, progress_fn=None
) -> CaptureChainResult:
    result = CaptureChainResult()
    seed: IfvgDaySeed | None = None
    prev_artifacts: DayArtifacts | None = None

    for chain_idx, date_str in enumerate(dates):
        entering_hash = seed_hash(seed) if seed is not None else None
        cached = _load_capture(cfg, date_str, expected_entering=entering_hash)
        expected_seeds = _chained_seeds(prev_artifacts)
        artifacts = load_day_artifacts(date_str, cfg, expected_seeds=expected_seeds)
        if cached is not None and artifacts is not None:
            frame, funnel, seed = cached
            result.cached_days.append(date_str)
        else:
            if artifacts is None:
                build_seeds = (
                    expected_seeds if expected_seeds is not None else seeds_for_day(date_str, cfg)
                )
                artifacts = build_day_artifacts(date_str, cfg, build_seeds)
                write_day_artifacts(artifacts, cfg)
                result.rebuilt_days.append(date_str)
            day = capture_single_date(date_str, cfg, artifacts=artifacts, seed=seed)
            seed = day.end_seed
            frame, funnel = day.rows, day.funnel
            end_hash = seed_hash(seed)
            _write_capture(
                frame,
                cfg,
                date_str,
                entering_hash=entering_hash,
                end_hash=end_hash,
                funnel=funnel,
            )
            with open(cfg.seed_path(date_str), "wb") as fh:
                pickle.dump(seed, fh)
        if len(frame):
            frame = frame.copy()
            frame["is_warmup"] = chain_idx < cfg.warmup_days
            frame["days_of_htf_history"] = chain_idx
        result.frames.append(frame)
        result.day_funnels[date_str] = funnel
        prev_artifacts = artifacts
        if progress_fn is not None:
            progress_fn(chain_idx + 1, len(dates), date_str)
    return result
