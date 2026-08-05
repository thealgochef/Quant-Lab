"""Tests for the pure verifier chart builders (synthetic evidence only)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ifvg_verifier_charts as charts  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.replay_chart_provider import (  # noqa: E402
    FsmStage,
    StageGate,
    ZoneEvidence,
)


def _ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


_START = _ts("2026-01-07T10:00:00Z")
_END = _ts("2026-01-07T12:00:00Z")


def _bars_1m(minutes: int = 120) -> pd.DataFrame:
    rows = []
    for index in range(minutes):
        close = _START + pd.Timedelta(minutes=index + 1)
        rows.append(
            {
                "bar_id": f"60s:2026-01-07:{index}",
                "close_ts_utc": close,
                "open_ticks": 103000 + index,
                "high_ticks": 103050 + index,
                "low_ticks": 102950 + index,
                "close_ticks": 103020 + index,
                "volume": 10,
                "trade_count": 4,
            }
        )
    return pd.DataFrame(rows)


def _bars_tf(timeframe: int) -> pd.DataFrame:
    rows = []
    count = max(1, (120 * 60) // timeframe)
    for index in range(count):
        open_ts = _START + pd.Timedelta(seconds=timeframe * index)
        rows.append(
            {
                "bar_id": f"{timeframe}s:2026-01-07:{index}",
                "logical_open_ts_utc": open_ts,
                "logical_close_ts_utc": open_ts + pd.Timedelta(seconds=timeframe),
                "open_ticks": 103000,
                "high_ticks": 103100,
                "low_ticks": 102900,
                "close_ticks": 103040,
                "volume": 100,
                "trade_count": 40,
            }
        )
    return pd.DataFrame(rows)


def _zone(role: str, timeframe: int, confirmed: str) -> ZoneEvidence:
    return ZoneEvidence(
        role=role,
        fvg_id=f"{role}-zone",
        timeframe_seconds=timeframe,
        direction="bullish",
        gap_low_ticks=102980,
        gap_high_ticks=103010,
        size_ticks=30,
        a_bar_id=f"{timeframe}s:2026-01-06:5",
        c_bar_id=f"{timeframe}s:2026-01-06:7",
        a_open_ts_utc=_ts("2026-01-06T20:00:00Z"),  # long before confirmation
        confirmed_ts_utc=_ts(confirmed),
        trading_day="2026-01-07",
    )


def _gate(stage: FsmStage, ts: str, ordinal: int) -> StageGate:
    return StageGate(
        stage=stage,
        ts_utc=_ts(ts),
        event_cursor=f"{ts}|60|x",
        trace_ordinal=ordinal,
        source_event_id=f"{stage.value}-event",
        source_kind=f"lifecycle:{stage.value}",
    )


class _Evidence:
    """Minimal duck-typed CandidateEvidence for the pure builder."""

    def __init__(self, *, executed: bool = True, htf_tf: int = 14400, mode: str = "full_audit",
                 stage: str | None = None) -> None:
        self.candidate_id = "cand-1234567890ab"
        self.mode = mode
        self.stage = stage
        self.zones = (
            _zone("htf", htf_tf, "2026-01-07T10:05:00Z"),
            _zone("parent", 300, "2026-01-07T10:20:00Z"),
            _zone("opposing", 60, "2026-01-07T10:40:00Z"),
            _zone("entry_fvg", 60, "2026-01-07T11:00:00Z"),
        )
        self.stage_gates = {
            "tap": _gate(FsmStage.TAP, "2026-01-07T10:05:00Z", 1),
            "inversion": _gate(FsmStage.INVERSION, "2026-01-07T10:50:00Z", 4),
            "entry": _gate(FsmStage.ENTRY, "2026-01-07T11:00:00Z", 5),
            "resolution": _gate(FsmStage.RESOLUTION, "2026-01-07T11:30:00Z", 6),
        }
        self.risk = {
            "direction": "LONG",
            "entry_ticks": 103020,
            "stop_ticks": 102984,
            "target_ticks": 103056,
            "manipulation_swing_ticks": 102985,
            "sl_buffer_ticks": 1,
            "risk_ticks": 36,
        }
        self.execution = (
            {
                "trade_id": "trade-1",
                "decision_id": "decision-1",
                "entry_ts_utc": _ts("2026-01-07T11:00:00Z"),
                "entry_cursor": "2026-01-07T11:00:00+00:00|60|x",
                "entry_ticks": 103020,
                "stop_ticks": 102984,
                "target_ticks": 103056,
                "entry_session": "london",
                "is_warmup": False,
                "resolution": "target",
                "resolution_ts_utc": _ts("2026-01-07T11:30:00Z"),
                "resolution_cursor": "2026-01-07T11:30:00+00:00|60|x",
                "status": "resolved",
                "bars_after_entry_to_resolution": 30,
                "mfe_ticks": 40,
                "mae_ticks": 12,
                "realized_ticks": 36,
                "realized_r": 1.0,
            }
            if executed
            else None
        )
        self.counterfactual_labels = ()
        self.structure = pd.DataFrame()
        self.structure_stage_summary = pd.DataFrame()
        self.displacement = pd.DataFrame()
        self.pools = pd.DataFrame()
        self.pool_members = pd.DataFrame()
        self.sweep_links = pd.DataFrame()
        self.sessions = {"entry_session": "london", "schemes": {}}
        self.model = {}
        self.gating_report = {"ungateable": [], "hidden": []}
        self.range_row = {
            "entry_anchor_ts": _ts("2026-01-07T11:00:00Z"),
            "display_end_ts": _ts("2026-01-07T11:30:00Z"),
            "parent_timeframe_seconds": 300,
            "htf_timeframe_seconds": htf_tf,
        }
        self.anchor_240m_status = "experimental_q40_open"
        self.lineage = {}
        self.identity = {}
        self.transition_bars = {}
        self.lifecycle = pd.DataFrame()


def _build(evidence: _Evidence, htf_tf: int = 14400):
    bars = {60: _bars_1m(), 300: _bars_tf(300), htf_tf: _bars_tf(htf_tf)}
    return charts.build_verifier_figure(
        evidence=evidence,
        bars_by_pane=bars,
        parent_tf=300,
        htf_tf=htf_tf,
        range_bounds=(_START, _END),
    )


class TestGeometryIndex:
    def test_golden_coordinates_match_evidence(self) -> None:
        evidence = _Evidence()
        fig, _report = _build(evidence)
        index = charts.figure_geometry_index(fig)
        tick = charts.TICK_SIZE
        assert index["risk"]["entry_price"] == pytest.approx(103020 * tick)
        assert index["risk"]["stop_price"] == pytest.approx(102984 * tick)
        assert index["risk"]["target_price"] == pytest.approx(103056 * tick)
        assert index["execution"]["resolution_ts"] == "2026-01-07T11:30:00+00:00"
        assert index["mfe_mae"]["mfe_price"] == pytest.approx((103020 + 40) * tick)
        assert index["mfe_mae"]["mae_price"] == pytest.approx((103020 - 12) * tick)
        for role in ("htf", "parent", "opposing", "entry_fvg"):
            assert index["zones"][role]["y0"] == pytest.approx(102980 * tick)
            assert index["zones"][role]["y1"] == pytest.approx(103010 * tick)
        assert index["stage_markers"]["entry"] == "2026-01-07T11:00:00+00:00"

    def test_zone_never_drawn_backward(self) -> None:
        """The rect starts at confirmed_ts, never at the earlier a_open_ts."""
        evidence = _Evidence()
        fig, _report = _build(evidence)
        index = charts.figure_geometry_index(fig)
        for role, payload in index["zones"].items():
            assert payload["x0"] == payload["confirmed_ts"], role
            assert pd.Timestamp(payload["x0"]) > _ts("2026-01-06T20:00:00Z")


class TestWatermark:
    def test_watermark_present_iff_240m_evidence(self) -> None:
        with_240 = _Evidence(htf_tf=14400)
        fig, _ = _build(with_240, htf_tf=14400)
        texts = [a.text for a in fig.layout.annotations]
        assert any("experimental_q40_open" in (t or "") for t in texts)
        assert charts.figure_geometry_index(fig)["watermark_240m"] is True

        without_240 = _Evidence(htf_tf=3600)
        fig2, _ = _build(without_240, htf_tf=3600)
        # the synthetic zones still carry a 3600s HTF zone -> no 240m evidence
        texts2 = [a.text for a in fig2.layout.annotations]
        assert not any("experimental_q40_open" in (t or "") for t in texts2)
        assert charts.figure_geometry_index(fig2)["watermark_240m"] is False


class TestVisualGrammar:
    def test_blocked_candidate_has_no_execution_traces(self) -> None:
        evidence = _Evidence(executed=False)
        evidence.risk = {"direction": "LONG"}  # provider empties risk pre-entry
        bars = {60: _bars_1m(), 300: _bars_tf(300), 14400: _bars_tf(14400)}
        fig, _report = charts.build_verifier_figure(
            evidence=evidence,
            bars_by_pane=bars,
            parent_tf=300,
            htf_tf=14400,
            range_bounds=(_START, _END),
            blocked_reasons="entry_family_not_profile",
        )
        names = [trace.name for trace in fig.data if trace.name]
        assert not any(name in ("entry", "stop", "target (1R)") for name in names)
        assert "blocked candidate" in names
        index = charts.figure_geometry_index(fig)
        assert index["risk"] == {}
        assert index["blocked_reasons"] == "entry_family_not_profile"

    def test_counterfactual_lines_are_dashed(self) -> None:
        evidence = _Evidence(executed=False)  # risk present but no execution
        fig, _report = _build(evidence)
        risk_traces = [
            trace
            for trace in fig.data
            if trace.name and "counterfactual" in trace.name and trace.mode == "lines"
        ]
        assert risk_traces
        assert all(trace.line.dash == "dash" for trace in risk_traces)
        assert charts.figure_geometry_index(fig)["risk"]["actual"] is False

    def test_actual_execution_lines_are_solid(self) -> None:
        evidence = _Evidence(executed=True)
        fig, _report = _build(evidence)
        entry_traces = [t for t in fig.data if t.name == "entry"]
        assert entry_traces and entry_traces[0].line.dash == "solid"

    def test_mfe_mae_are_labeled_magnitude_only(self) -> None:
        evidence = _Evidence(executed=True)
        fig, _report = _build(evidence)
        labels = [t.name for t in fig.data if t.name and "magnitude only" in t.name]
        assert any("MFE" in label for label in labels)
        assert any("MAE" in label for label in labels)
        for trace in fig.data:
            if trace.name and "magnitude only" in trace.name:
                assert trace.line.dash == "dot"

    def test_pane_scale_follows_bars_not_far_zones(self) -> None:
        """A distant HTF zone must clip at the pane edge, not squash the candles."""
        import dataclasses

        evidence = _Evidence()
        far_zone = dataclasses.replace(
            evidence.zones[0], gap_low_ticks=90000, gap_high_ticks=90100
        )
        evidence.zones = (far_zone, *evidence.zones[1:])
        fig, _report = _build(evidence)
        index = charts.figure_geometry_index(fig)
        tick = charts.TICK_SIZE
        low, high = index["y_ranges"]["1"]
        bars = _bars_1m()
        assert low > 90100 * tick, "far zone must not stretch the 1m scale"
        assert low <= bars["low_ticks"].min() * tick
        assert high >= bars["high_ticks"].max() * tick
        assert fig.layout.yaxis.range is not None
        # rows 2 and 3 scale to their own bars
        assert "2" in index["y_ranges"] and "3" in index["y_ranges"]

    def test_y_axes_are_manually_scalable(self) -> None:
        """Candlestick traces implicitly lock y-axes via the hidden x-rangeslider;
        every pane must stay draggable (fixedrange must be explicitly False)."""
        evidence = _Evidence()
        fig, _report = _build(evidence)
        for axis in ("yaxis", "yaxis2", "yaxis3"):
            assert fig.layout[axis].fixedrange is False, axis

    def test_point_in_time_styling_is_unmistakable(self) -> None:
        evidence = _Evidence(mode="point_in_time", stage="entry")
        fig, _report = _build(evidence)
        assert fig.layout.title.text.startswith("[POINT-IN-TIME · entry]")
        assert fig.layout.paper_bgcolor == "#FFF8EC"
        full = _Evidence()
        fig2, _ = _build(full)
        assert fig2.layout.title.text.startswith("[FULL AUDIT]")
        assert fig2.layout.paper_bgcolor == "white"


class TestBudgets:
    def test_budget_excess_produces_omission_report_not_exception(self) -> None:
        evidence = _Evidence()
        pools = pd.DataFrame(
            [
                {
                    "pool_pool_id": f"pool-{index}",
                    "pool_pool_type": "eqh",
                    "pool_lower_bound_ticks": 103000 + index,
                    "pool_upper_bound_ticks": 103005 + index,
                    "pool_confirmation_ts": _START + pd.Timedelta(minutes=index),
                    "pool_active": True,
                    "pool_swept": False,
                    "pool_sweep_ts": None,
                    "pool_reclaimed": False,
                    "pool_reclaim_ts": None,
                }
                for index in range(30)
            ]
        )
        evidence.pools = pools
        bars = {60: _bars_1m(), 300: _bars_tf(300), 14400: _bars_tf(14400)}
        fig, report = charts.build_verifier_figure(
            evidence=evidence,
            bars_by_pane=bars,
            parent_tf=300,
            htf_tf=14400,
            range_bounds=(_START, _END),
            layers=charts.VerifierLayers(pools=True),
        )
        assert fig is not None
        assert report.entries, "budget excess must be reported"
        assert report.entries[0]["layer"] == "pools"
        assert report.entries[0]["omitted"] == 30 - charts.LAYER_BUDGETS["pools"]
        assert any("omitted" in line for line in report.summary_lines())
