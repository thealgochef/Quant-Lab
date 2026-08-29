"""R6.1 synthetic context-bar panel fixture (plan §6.A).

``synthetic_label_source_1m`` builds aligned 1m bars (from each trading
day's 18:00 ET open, ``trading_day_open_utc``) with real ``volume`` /
``trade_count`` and THREE planted bar-level regimes (distinct range /
volume scales by block of the day) so the seven panel features separate
them; ``write_synthetic_replay_chart_artifact`` persists a REAL
``ifvg_replay_chart_v1`` artifact through the production resampler and
identity functions, so ``load_verified_replay_chart_artifact`` runs
unmodified on it (tmp roots only; business days only).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256, file_sha256
from alpha_lab.agents.data_infra.ifvg.replay_chart_store import (
    REPLAY_TIMEFRAMES_SECONDS,
    ArtifactPairRef,
    replay_chart_effective_config,
    replay_chart_identity,
    resample_label_bars,
    trading_day_open_utc,
)

__all__ = [
    "PLANTED_REGIME_SCALES",
    "synthetic_trading_days",
    "synthetic_label_source_1m",
    "planted_regime_for_bar",
    "synthetic_pair_ref",
    "write_synthetic_replay_chart_artifact",
]

#: (range scale in ticks, volume scale) of the three planted bar regimes.
PLANTED_REGIME_SCALES: tuple[tuple[float, float], ...] = ((1.0, 20.0), (5.0, 60.0), (14.0, 180.0))

#: Minutes (from the 18:00 ET open) each regime block spans; the day is
#: generated from ``start_minute`` to ``end_minute`` (defaults: 02:00 → 14:00
#: ET, i.e. London open through the NY close — 720 aligned 1m bars).
_DEFAULT_START_MINUTE = 8 * 60  # 02:00 ET
_DEFAULT_END_MINUTE = 20 * 60  # 14:00 ET


def synthetic_trading_days(count: int, *, start: str = "2026-01-05") -> tuple[str, ...]:
    return tuple(day.strftime("%Y-%m-%d") for day in pd.bdate_range(start, periods=count))


def planted_regime_for_bar(elapsed_minute: int, *, start_minute: int, end_minute: int) -> int:
    """Three contiguous blocks of the generated span → regime 0/1/2."""

    span = max(1, end_minute - start_minute)
    block = (elapsed_minute - start_minute) * 3 // span
    return int(min(2, max(0, block)))


def synthetic_label_source_1m(
    days: tuple[str, ...],
    *,
    rng: np.random.Generator | None = None,
    start_minute: int = _DEFAULT_START_MINUTE,
    end_minute: int = _DEFAULT_END_MINUTE,
    early_close_day: str | None = None,
    early_close_minute: int | None = None,
    drop_window: tuple[str, int, int] | None = None,
) -> pd.DataFrame:
    """Aligned 1m bars over ``days``.

    ``early_close_day`` truncates that day at ``early_close_minute`` (an
    early close leaves the last resampled bar partial); ``drop_window =
    (day, from_minute, to_minute)`` removes the 1m bars of that half-open
    minute range (an incomplete source bar at the resampled grain).
    """

    rng = rng if rng is not None else np.random.default_rng(7)
    rows: list[dict] = []
    price = 20_000.0
    for day in days:
        day_open = trading_day_open_utc(day)
        last_minute = end_minute
        if early_close_day == day and early_close_minute is not None:
            last_minute = min(end_minute, early_close_minute)
        for minute in range(start_minute, last_minute):
            if drop_window is not None and drop_window[0] == day and (
                drop_window[1] <= minute < drop_window[2]
            ):
                continue
            regime = planted_regime_for_bar(
                minute, start_minute=start_minute, end_minute=end_minute
            )
            range_scale, volume_scale = PLANTED_REGIME_SCALES[regime]
            step = float(rng.normal(0.0, range_scale * 0.6))
            open_ticks = price
            close_ticks = price + step
            high_ticks = max(open_ticks, close_ticks) + abs(rng.normal(0.0, range_scale))
            low_ticks = min(open_ticks, close_ticks) - abs(rng.normal(0.0, range_scale))
            price = close_ticks
            close_ts = day_open + pd.Timedelta(minutes=minute + 1)
            rows.append(
                {
                    "source_date": day,
                    "bar_id": f"60s:{day}:{minute}",
                    "close_ts_utc": close_ts.isoformat(),
                    "open_ticks": int(round(open_ticks)),
                    "high_ticks": int(round(high_ticks)),
                    "low_ticks": int(round(low_ticks)),
                    "close_ticks": int(round(close_ticks)),
                    "volume": int(max(1, round(rng.normal(volume_scale, volume_scale * 0.15)))),
                    "trade_count": int(max(1, round(volume_scale * 0.4))),
                }
            )
    return pd.DataFrame(rows)


def synthetic_pair_ref(seed: str = "panel") -> ArtifactPairRef:
    return ArtifactPairRef(
        profile_name="ifvg_v2_doc_default_fresh_static_1r",
        v2_dataset_id=canonical_sha256({"v2": seed}),
        v2_manifest_hash=canonical_sha256({"v2_manifest": seed}),
        v3_dataset_id=canonical_sha256({"v3": seed}),
        v3_manifest_hash=canonical_sha256({"v3_manifest": seed}),
    )


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n", "utf-8")


def _entry(path: Path, root: Path, *, rows: int | None = None) -> dict:
    entry = {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        entry["rows"] = int(rows)
    return entry


def write_synthetic_replay_chart_artifact(
    base_dir: Path, bars_1m: pd.DataFrame, pair_ref: ArtifactPairRef
) -> str:
    """Persist a real-shaped ``ifvg_replay_chart_v1`` artifact under
    ``base_dir`` and return its content identity."""

    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    label_source_sha256 = canonical_sha256(
        {"synthetic_label_source": bars_1m["bar_id"].astype(str).tolist()}
    )
    effective = replay_chart_effective_config(pair_ref, label_source_sha256=label_source_sha256)
    artifact_id = replay_chart_identity(effective)
    directory = base / artifact_id
    if directory.exists():
        return artifact_id
    temporary = base / f".{artifact_id}.tmp"
    temporary.mkdir()
    bars_tf = pd.concat(
        [resample_label_bars(bars_1m, timeframe) for timeframe in REPLAY_TIMEFRAMES_SECONDS],
        ignore_index=True,
    )
    artifacts: list[dict] = []
    config_path = temporary / "effective_config.json"
    _write_json(config_path, effective)
    artifacts.append(_entry(config_path, temporary))
    bars_path = temporary / "bars_tf.parquet"
    bars_tf.to_parquet(bars_path, index=False)
    artifacts.append(_entry(bars_path, temporary, rows=len(bars_tf)))
    ranges = pd.DataFrame(
        {
            "candidate_id": pd.Series(dtype="object"),
            "setup_id": pd.Series(dtype="object"),
            "trading_day": pd.Series(dtype="object"),
            "first_bar_id_1m": pd.Series(dtype="object"),
            "last_bar_id_1m": pd.Series(dtype="object"),
        }
    )
    ranges_path = temporary / "candidate_bar_range.parquet"
    ranges.to_parquet(ranges_path, index=False)
    artifacts.append(_entry(ranges_path, temporary, rows=0))
    access_path = temporary / "data_access_audit.json"
    _write_json(access_path, {"events": [], "denied_dates": {}, "synthetic": True})
    artifacts.append(_entry(access_path, temporary))
    _write_json(
        temporary / "corroboration_report.json",
        {"oracle": "skipped", "corroborated_days": [], "uncorroborated_days": []},
    )
    core = {
        "manifest_schema_version": 1,
        "artifact_kind": "ifvg_replay_chart_v1",
        "replay_chart_artifact_id": artifact_id,
        "immutable": True,
        "source_pair": pair_ref.as_dict(),
        "effective_config": effective,
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    }
    _write_json(
        temporary / "manifest.json",
        {**core, "manifest_payload_sha256": canonical_sha256(core)},
    )
    temporary.rename(directory)
    return artifact_id
