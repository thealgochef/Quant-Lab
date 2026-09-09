"""The saved-study reviewer preserves exact execution identity and input evidence."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_review_saves_selected_research_trade_after_reverification(monkeypatch, tmp_path):
    import ifvg_search_review as ui
    import ifvg_verifier_tab as verifier

    ts = pd.Timestamp("2026-01-13T12:00Z")
    row = dict(
        trade_id="research-trade",
        candidate_id="research-candidate",
        decision_id="research-decision",
        is_warmup=False,
        trading_day="2026-01-13",
        entry_ts_utc=ts,
        resolution_ts_utc=ts + pd.Timedelta(minutes=2),
        resolution="target",
        entry_ticks=100,
        stop_ticks=90,
        target_ticks=110,
        realized_r=1,
        risk_ticks=10,
        mfe_ticks=12,
        mae_ticks=3,
        geometry_parent_timeframe_seconds=300,
        geometry_htf_timeframe_seconds=900,
    )
    for key in (
        "tap_bar_logical_close",
        "parent_confirmed",
        "lock_bar_logical_close",
        "opposing_confirmed",
        "inversion_bar_logical_close",
    ):
        row[f"geometry_{key}_ts_utc"] = ts - pd.Timedelta(minutes=1)
    warmup = {**row, "is_warmup": True, "trade_id": "warmup-trade"}
    evidence = SimpleNamespace(
        reference={"core_replay_id": "core", "v2_dataset_id": "exact-dataset"},
        dataset=SimpleNamespace(tables={RecordTable.EXECUTED_TRADE: pd.DataFrame([warmup, row])}),
    )
    bars = pd.DataFrame(
        dict(
            timeframe_ticks=[60],
            logical_close_ts_utc=[ts],
            open_ticks=[100],
            high_ticks=[110],
            low_ticks=[99],
            close_ticks=[105],
        )
    )
    checks, saved, captions = [], [], []

    class Screen:
        session_state = {"ifvg_search_review_pending": ("study", "core")}

        def radio(self, _label, _options, **kw):
            return self.session_state[kw["key"]]

        def selectbox(self, _label, options, **kw):
            return options[0]

        def checkbox(self, label, **kw):
            return False

        def caption(self, message):
            captions.append(message)

        def columns(self, count):
            return [self] * count

        def expander(self, *args, **kw):
            return nullcontext()

        def __getattr__(self, name):
            if name == "error":
                return lambda message: pytest.fail(message)
            return lambda *args, **kw: None

    monkeypatch.setattr(
        ui, "list_search_runs", lambda *a: [SimpleNamespace(search_id="study", archived=False)]
    )
    monkeypatch.setattr(
        ui, "load_studies", lambda *a: ([SimpleNamespace(key="study", name="Named study")], [])
    )
    monkeypatch.setattr(
        ui,
        "load_search_state",
        lambda *a: {
            "children": [dict(core_replay_id="core", state="completed", axis_value_ids={})]
        },
    )
    monkeypatch.setattr(ui, "configuration_name", lambda *a: "Baseline")
    monkeypatch.setattr(ui, "_evidence", lambda *a: evidence)
    monkeypatch.setattr(ui, "_bars", lambda *a: bars)
    monkeypatch.setattr(
        ui, "load_search_review_evidence", lambda *a: checks.append("tables") or evidence
    )
    monkeypatch.setattr(ui, "load_search_day_bars", lambda *a: checks.append("bars") or bars)
    monkeypatch.setattr(ui, "list_reviews", lambda **kw: pd.DataFrame())
    monkeypatch.setattr(ui, "append_review", lambda **kw: saved.append(kw))
    monkeypatch.setattr(
        verifier,
        "_review_form",
        lambda _st, on_save, **kw: on_save(
            reviewer="Test", verdicts={"overall_verdict": "correct"}, tags=[], notes="Test review"
        ),
    )
    ui.render_trade_review(
        Screen(), {k: tmp_path for k in ("repo_root", "store_root", "state_root")}
    )
    assert checks == ["tables", "bars"]
    assert len(saved) == 1
    assert saved[0]["trade_id"] == "research-trade"
    assert saved[0]["candidate_id"] == "research-candidate"
    assert saved[0]["decision_id"] == "research-decision"
    assert saved[0]["pair_ref"] == evidence.reference
    assert saved[0]["replay_chart_artifact_id"] == ""
    assert any(text.startswith("1 executed trades") for text in captions)


def test_original_bar_tampering_is_rejected_before_parquet_read(monkeypatch, tmp_path):
    from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
    from alpha_lab.agents.data_infra.ifvg.development_access import FROZEN_WARMUP_DATES
    from alpha_lab.agents.data_infra.ifvg.search import review_evidence as provider

    section = IfvgCaptureConfig().section
    cfg = IfvgCaptureConfig(section=section, data_dir=tmp_path / "data/databento")
    path = cfg.day_dir("2026-01-13") / "bars.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed source bytes")
    evidence = SimpleNamespace(
        inputs=SimpleNamespace(
            payload=SimpleNamespace(
                ordered_day_artifacts=[
                    SimpleNamespace(
                        trading_day=day,
                        artifact_kind="bars",
                        artifact_id="bars.parquet",
                        content_sha256="0" * 64,
                    )
                    for day in (*FROZEN_WARMUP_DATES, "2026-01-13")
                ]
            )
        ),
        dataset=SimpleNamespace(
            reports={"effective_config.json": {"section": section.model_dump(mode="json")}}
        ),
    )
    monkeypatch.setattr(
        provider.pd, "read_parquet", lambda *a: pytest.fail("must verify before reading")
    )
    with pytest.raises(ValueError, match="changed since"):
        provider.load_search_day_bars(tmp_path, evidence, "2026-01-13")
    with pytest.raises(ValueError, match="exactly one"):
        provider.load_search_day_bars(tmp_path, evidence, "2026-01-14")
