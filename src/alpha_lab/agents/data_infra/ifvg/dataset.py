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

__all__ = ["build_ifvg_capture", "CaptureChainResult"]

_CAP_META_KEY = b"ifvg_capture_meta"


class CaptureChainResult:
    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []
        self.day_funnels: dict[str, dict[str, int]] = {}
        self.rebuilt_days: list[str] = []
        self.cached_days: list[str] = []

    def frame(self) -> pd.DataFrame:
        real = [f for f in self.frames if len(f)]
        return pd.concat(real, ignore_index=True) if real else pd.DataFrame()


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
