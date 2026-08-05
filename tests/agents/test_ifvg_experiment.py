"""IFVG experiment engine guards (plan Part A) — synthetic frames only.

The load-bearing invariants: the sealed clamp is non-configurable and no
sealed row ever reaches any result section (counts only); expectancy / PF /
maxDD math is hand-checkable; the config hash is stable and round-trips
through persistence; pooled-OOS dedup keeps the last (largest-train) split;
SL/TP swing recovery math is exact for LONG and SHORT; custom-session
re-stamping honors cross-midnight windows.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from strategy_core.types import Bar, BarKind

from alpha_lab.agents.data_infra.ifvg.config import legacy_ifvg_capture_config
from alpha_lab.agents.data_infra.ifvg.experiment import (
    IfvgExperimentConfig,
    IfvgFilterConfig,
    IfvgScoringConfig,
    IfvgSlTpConfig,
    SessionWindow,
    dedup_pooled_oos,
    json_safe,
    list_experiments,
    load_experiment,
    recover_override_stop_ticks,
    run_ifvg_experiment,
    run_sealed_validation,
    save_experiment,
    stamp_custom_sessions,
)

_TICK = 0.25


def _legacy_capture():
    return legacy_ifvg_capture_config()


def _row(
    day: str,
    hour: int,
    label: str,
    realized_pts: float,
    *,
    setup_id: str = "s",
    direction: str = "LONG",
    family: str = "ifvg_retest",
    risk_points: float = 4.0,
    bars_to_res: int = 5,
    session_engine: str = "ny",
    session_doc: str = "ny",
    warmup: bool = False,
    entry_ticks: int = 100,
    stop_ticks: int = 84,
) -> dict:
    entry_ts = pd.Timestamp(f"{day} {hour:02d}:00:00", tz="UTC")
    return {
        "setup_id": f"{setup_id}-{day}-{hour}",
        "trading_day": day,
        "entry_ts_utc": entry_ts,
        "entry_family": family,
        "direction": direction,
        "entry_ticks": entry_ticks,
        "stop_ticks": stop_ticks,
        "risk_ticks": int(risk_points / _TICK),
        "risk_points": risk_points,
        "session_engine": session_engine,
        "session_doc": session_doc,
        "is_warmup": warmup,
        "label_r10": label,
        "bars_to_res_r10": bars_to_res if label != "eod_timeout" else -1,
        "realized_pts_r10": realized_pts,
        "realized_r_net_r10": (realized_pts - 0.514) / risk_points,
        "mfe_r": 0.5,
        "mae_r": 0.25,
        "label_window_end": entry_ts + timedelta(hours=8),
        "feat_a": float(hour),
        "feat_b": risk_points,
    }


def _expectancy_only() -> IfvgExperimentConfig:
    return IfvgExperimentConfig()  # model=None -> expectancy-only


# ── sealed guard ──────────────────────────────────────────────────────────────


def test_sealed_rows_never_enter_any_result_section() -> None:
    pre = [
        _row("2026-05-01", 10 + i, "win" if i % 2 == 0 else "loss", 4.0 if i % 2 == 0 else -4.0)
        for i in range(6)
    ]
    sealed = [
        _row("2026-06-20", 10 + i, "win", 4.0, setup_id="SEALEDROW") for i in range(4)
    ]
    ds = pd.DataFrame(pre + sealed)
    result = run_ifvg_experiment(
        _expectancy_only(), dataset=ds, capture_cfg=_legacy_capture()
    )

    assert result["meta"]["sealed"]["sealed_rows_excluded"] == 4
    assert result["counterfactual_outcomes"]["overall"]["n"] == 6
    assert result["trade_stats"] is None
    assert result["meta"]["execution_enabled"] is False
    assert result["model"] is None
    payload = json.dumps(json_safe(result))
    assert "SEALEDROW" not in payload  # sealed setup ids appear nowhere
    assert "2026-06-20" not in payload  # sealed trading day appears nowhere


def test_sealed_day_range_inputs_cannot_reintroduce_sealed_rows() -> None:
    ds = pd.DataFrame(
        [_row("2026-05-01", 10, "win", 4.0), _row("2026-07-01", 10, "win", 4.0)]
    )
    config = IfvgExperimentConfig(
        filters=IfvgFilterConfig(day_start="2026-01-01", day_end="2026-12-31")
    )
    result = run_ifvg_experiment(config, dataset=ds, capture_cfg=_legacy_capture())
    assert result["counterfactual_outcomes"]["overall"]["n"] == 1
    assert result["meta"]["sealed"]["sealed_rows_excluded"] == 1


# ── hand-computed stats ───────────────────────────────────────────────────────


def test_legacy_candidate_study_never_publishes_trade_performance() -> None:
    ds = pd.DataFrame(
        [
            _row("2026-05-01", 10, "win", 4.0),
            _row("2026-05-01", 11, "loss", -4.0),
            _row("2026-05-01", 12, "win", 4.0),
            _row("2026-05-01", 13, "eod_timeout", 1.0),
        ]
    )
    result = run_ifvg_experiment(
        _expectancy_only(), dataset=ds, capture_cfg=_legacy_capture()
    )
    labels = result["counterfactual_outcomes"]["overall"]
    assert labels["win_rate"] == pytest.approx(0.5)
    assert labels["mean_net_r"] == pytest.approx(0.184, abs=1e-9)
    assert result["trade_stats"] is None
    assert "equity" not in result
    assert "profit_factor" not in json.dumps(result)


# ── config hash + persistence round-trip ──────────────────────────────────────


def test_config_hash_is_stable_and_capture_tag_sensitive() -> None:
    config = IfvgExperimentConfig(
        filters=IfvgFilterConfig(direction="long"),
        scoring=IfvgScoringConfig(r_family="r15"),
    )
    h1 = config.experiment_hash("tag_a")
    assert h1 == config.experiment_hash("tag_a")
    assert h1 != config.experiment_hash("tag_b")
    rebuilt = IfvgExperimentConfig(**config.model_dump(mode="json"))
    assert rebuilt.experiment_hash("tag_a") == h1
    assert h1 != IfvgExperimentConfig().experiment_hash("tag_a")


def test_save_load_round_trip(tmp_path) -> None:
    ds = pd.DataFrame([_row("2026-05-01", 10 + i, "win", 4.0) for i in range(3)])
    config = _expectancy_only()
    result = run_ifvg_experiment(config, dataset=ds, capture_cfg=_legacy_capture())
    run_dir = save_experiment(config, result, name="unit", note="n", base_dir=tmp_path)
    assert (run_dir / "config.json").exists() and (run_dir / "result.json").exists()

    loaded = load_experiment(result["meta"]["experiment_hash"], base_dir=tmp_path)
    assert loaded["config"] == config
    assert loaded["result"]["counterfactual_outcomes"]["overall"]["n"] == 3
    summaries = list_experiments(tmp_path)
    assert len(summaries) == 1
    assert summaries[0]["name"] == "unit"
    assert summaries[0]["sealed_validations"] == []


# ── pooled-OOS dedup ──────────────────────────────────────────────────────────


def test_dedup_pooled_oos_keeps_last_split_prediction() -> None:
    oos = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 2, 3, 3],
            "split": [0.5, 0.5, 0.5, 0.65, 0.65, 0.8],
            "p_win": [0.10, 0.20, 0.30, 0.21, 0.31, 0.32],
        }
    )
    dedup = dedup_pooled_oos(oos)
    assert len(dedup) == 3
    by_row = dedup.set_index("row_id")
    assert by_row.loc[1, "p_win"] == pytest.approx(0.10)
    assert by_row.loc[2, "p_win"] == pytest.approx(0.21)  # from split 0.65
    assert by_row.loc[3, "p_win"] == pytest.approx(0.32)  # from split 0.8


# ── SL/TP override recompute ──────────────────────────────────────────────────


def _sl_tp(**kw) -> IfvgSlTpConfig:
    return IfvgSlTpConfig(**kw)


def test_swing_recovery_stop_math_long_and_short() -> None:
    swing2 = _sl_tp(sl_mode="swing_buffer_ticks", sl_value=2)
    # LONG: swing_low = stop+1 = 93; new stop = 93 - 2 = 91.
    assert (
        recover_override_stop_ticks(
            direction="LONG", entry_ticks=100, logged_stop_ticks=92,
            sl_tp=swing2, tick_size=_TICK,
        )
        == 91
    )
    # SHORT: swing_high = stop-1 = 107; new stop = 107 + 2 = 109.
    assert (
        recover_override_stop_ticks(
            direction="SHORT", entry_ticks=100, logged_stop_ticks=108,
            sl_tp=swing2, tick_size=_TICK,
        )
        == 109
    )
    fixed = _sl_tp(sl_mode="fixed_points", sl_value=2.0)  # 8 ticks
    assert (
        recover_override_stop_ticks(
            direction="LONG", entry_ticks=100, logged_stop_ticks=92,
            sl_tp=fixed, tick_size=_TICK,
        )
        == 92
    )
    assert (
        recover_override_stop_ticks(
            direction="SHORT", entry_ticks=100, logged_stop_ticks=108,
            sl_tp=fixed, tick_size=_TICK,
        )
        == 108
    )


def _bars(day: str, start_hour: int, path_ticks: list[tuple[int, int, int, int]]) -> list[Bar]:
    """Synthetic 1m bars (o, h, l, c in ticks) starting after ``start_hour``."""
    day_date = datetime.fromisoformat(day).date()
    out = []
    for i, (o, h, lo, c) in enumerate(path_ticks):
        open_ts = datetime(day_date.year, day_date.month, day_date.day, start_hour, tzinfo=UTC)
        open_ts += timedelta(minutes=i + 1)
        out.append(
            Bar(
                timeframe_ticks=60,
                trading_day=day_date,
                bar_index=i,
                bar_id=f"{day}-{i}",
                open_ts_utc=open_ts,
                close_ts_utc=open_ts + timedelta(minutes=1),
                open_ticks=o,
                high_ticks=h,
                low_ticks=lo,
                close_ticks=c,
                volume=1,
                trade_count=1,
                is_complete=True,
                is_partial=False,
                close_reason=None,
                kind=BarKind.TIME,
            )
        )
    return out


def test_sl_tp_override_recompute_long_and_short() -> None:
    # LONG: entry 100, logged stop 92; swing+2 -> stop 91, risk 9t = 2.25pt.
    # SHORT: entry 200, logged stop 208; swing+2 -> stop 209, risk 9t.
    ds = pd.DataFrame(
        [
            _row("2026-05-01", 10, "loss", -2.0, entry_ticks=100, stop_ticks=92),
            _row(
                "2026-05-01", 10, "loss", -2.0, setup_id="sh", direction="SHORT",
                entry_ticks=200, stop_ticks=208,
            ),
        ]
    )
    bars = _bars(
        "2026-05-01",
        10,
        # LONG rides 100 -> 110 (TP 109 hit, stop 91 never); SHORT path in the
        # same bars: 200-tick leg handled by lows staying above 191 target...
        [(100, 104, 99, 103), (103, 110, 102, 109)],
    )
    short_bars = _bars("2026-05-01", 10, [(200, 201, 196, 197), (197, 198, 190, 191)])

    def loader_for(frame_direction_bars):
        def _load(_day: str) -> list[Bar]:
            return frame_direction_bars

        return _load

    config = IfvgExperimentConfig(
        sl_tp=_sl_tp(sl_mode="swing_buffer_ticks", sl_value=2)
    )
    long_result = run_ifvg_experiment(
        config,
        dataset=ds.iloc[[0]],
        bars_loader=loader_for(bars),
        capture_cfg=_legacy_capture(),
    )
    assert long_result["counterfactual_outcomes"]["overall"]["win_rate"] == 1.0
    assert long_result["counterfactual_outcomes"]["overall"][
        "mean_net_r"
    ] == pytest.approx((2.25 - 0.514) / 2.25)
    assert long_result["trade_stats"] is None
    assert long_result["meta"]["caveats"][0].startswith("SL/TP override")

    short_result = run_ifvg_experiment(
        config,
        dataset=ds.iloc[[1]],
        bars_loader=loader_for(short_bars),
        capture_cfg=_legacy_capture(),
    )
    assert short_result["counterfactual_outcomes"]["overall"]["win_rate"] == 1.0
    assert short_result["trade_stats"] is None


def test_sl_tp_sub_tick_risk_rows_are_dropped_and_counted() -> None:
    ds = pd.DataFrame([_row("2026-05-01", 10, "win", 4.0)])
    config = IfvgExperimentConfig(
        sl_tp=_sl_tp(sl_mode="fixed_points", sl_value=0.1)  # rounds to 0 ticks
    )
    result = run_ifvg_experiment(
        config,
        dataset=ds,
        bars_loader=lambda _d: [],
        capture_cfg=_legacy_capture(),
    )
    assert result["meta"]["counts"]["sl_tp_risk_dropped"] == 1
    assert result["counterfactual_outcomes"]["overall"]["n"] == 0
    assert result["trade_stats"] is None


# ── custom sessions ───────────────────────────────────────────────────────────


def _et_row(day: str, hhmm: str, **kw) -> dict:
    ts = pd.Timestamp(f"{day} {hhmm}", tz="America/New_York").tz_convert("UTC")
    row = _row(day, 0, "win", 4.0, **kw)
    row["entry_ts_utc"] = ts
    return row


def test_custom_session_stamp_including_cross_midnight() -> None:
    frame = pd.DataFrame(
        [
            _et_row("2026-05-04", "23:30"),  # asia (crosses midnight)
            _et_row("2026-05-04", "01:00"),  # asia (early side)
            _et_row("2026-05-04", "02:45"),  # NOT asia (end-exclusive)
            _et_row("2026-05-04", "10:00"),  # morning
            _et_row("2026-05-04", "05:00"),  # none
        ]
    )
    windows = {
        "asia": SessionWindow(start="19:00", end="02:45"),
        "morning": SessionWindow(start="09:00", end="12:00"),
    }
    stamp = stamp_custom_sessions(frame, windows)
    assert list(stamp) == ["asia", "asia", "none", "morning", "none"]


def test_custom_session_filter_applies_to_restamped_rows() -> None:
    ds = pd.DataFrame(
        [
            _et_row("2026-05-04", "23:30"),
            _et_row("2026-05-04", "10:00"),
            _et_row("2026-05-04", "05:00"),
        ]
    )
    config = IfvgExperimentConfig(
        custom_sessions={"asia": SessionWindow(start="19:00", end="02:45")},
        sessions_custom=("asia",),
    )
    result = run_ifvg_experiment(config, dataset=ds, capture_cfg=_legacy_capture())
    assert result["counterfactual_outcomes"]["overall"]["n"] == 1
    assert result["counterfactual_outcomes"]["per_session_custom"]["asia"]["n"] == 1


def test_sessions_custom_requires_defined_windows() -> None:
    with pytest.raises(ValueError, match="sessions_custom"):
        IfvgExperimentConfig(sessions_custom=("asia",))


# ── sealed validation (synthetic only; files, no stdout) ──────────────────────


def test_run_sealed_validation_writes_files_only(tmp_path, capsys) -> None:
    pre = [_row("2026-05-01", 10 + i, "win", 4.0) for i in range(3)]
    sealed = [_row("2026-06-20", 10 + i, "loss", -4.0) for i in range(2)]
    ds = pd.DataFrame(pre + sealed)
    config = _expectancy_only()
    result = run_ifvg_experiment(config, dataset=ds, capture_cfg=_legacy_capture())
    save_experiment(config, result, base_dir=tmp_path)

    out_path = run_sealed_validation(
        result["meta"]["experiment_hash"],
        dataset=ds,
        base_dir=tmp_path,
        capture_cfg=_legacy_capture(),
    )
    assert capsys.readouterr().out == ""  # sealed stats never hit stdout
    sealed_result = json.loads(out_path.read_text(encoding="utf-8"))
    assert sealed_result["sealed_validation"] is True
    assert sealed_result["sequence"] == 1
    assert sealed_result["trade_stats"] is None
    assert sealed_result["counterfactual_outcomes"]["overall"]["n"] == 2

    ledger = (tmp_path / "sealed_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger) == 1
    entry = json.loads(ledger[0])
    assert entry["seq"] == 1
    assert entry["config_hash"] == result["meta"]["experiment_hash"]
    history = list_experiments(tmp_path)[0]["sealed_validations"]
    assert history == [1]
