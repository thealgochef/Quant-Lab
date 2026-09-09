"""Research charts and scoped panels: synthetic bytes, exact roots, no real data."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from alpha_lab.agents.data_infra.ifvg import replay_chart_store as charts
from alpha_lab.agents.data_infra.ifvg.context_contracts import ContextRecordTable
from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_materializer import (
    load_context_bar_panel_validity,
    load_verified_context_bar_panel,
    materialize_context_bar_panel,
    save_context_bar_panel_artifact,
)
from alpha_lab.agents.data_infra.ifvg.search import research_data, research_runs
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_context_panel import (
    synthetic_label_source_1m,
    synthetic_pair_ref,
    write_synthetic_replay_chart_artifact,
)

DAYS = ("2026-01-05", "2026-01-06", "2026-01-07")


@pytest.fixture
def research_chart(tmp_path, monkeypatch):
    pair_ref = synthetic_pair_ref("scoped-research")
    subject = SimpleNamespace(
        subject_id="a" * 64,
        core_replay_id="b" * 64,
        v2_dataset_id=pair_ref.v2_dataset_id,
        v2_manifest_hash=pair_ref.v2_manifest_hash,
        evaluation_dates=(DAYS[1],),
        cutoff_ts_utc="2026-01-06T22:00:00Z",
    )
    bars = synthetic_label_source_1m(DAYS[:2], start_minute=480, end_minute=565)
    candidates, dossiers, life, labels = [], [], [], []
    for index, day in enumerate(DAYS):
        candidate_id, setup_id = f"candidate-{index}", f"setup-{index}"
        ts = charts.trading_day_open_utc(day) + pd.Timedelta(minutes=530)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "setup_id": setup_id,
                "trading_day": day,
                "envelope_ts_utc": ts,
                "trace_ordinal": index + 10,
                "is_warmup": index == 0,
                "profile_name": pair_ref.profile_name,
            }
        )
        dossiers.append(
            {
                "candidate_id": candidate_id,
                "decision_id": None,
                "trade_id": None,
                "geometry_htf_timeframe_seconds": 3600,
                "geometry_parent_timeframe_seconds": 300,
            }
        )
        life.extend(
            [
                {
                    "setup_id": setup_id,
                    "envelope_ts_utc": ts - pd.Timedelta(minutes=10),
                    "trace_ordinal": index * 2,
                    "transition": "setup_activated",
                },
                {
                    "setup_id": setup_id,
                    "envelope_ts_utc": ts + pd.Timedelta(days=1),
                    "trace_ordinal": index * 2 + 1,
                    "transition": "setup_ended",
                },
            ]
        )
        labels.append(
            {
                "candidate_id": candidate_id,
                "label_family": charts.PRIMARY_LABEL_FAMILY,
                "censored": False,
                "resolution_bar_id": "60s:2026-01-08:600",
            }
        )
    ref2 = SimpleNamespace(
        artifact_id=pair_ref.v2_dataset_id, manifest_payload_sha256=pair_ref.v2_manifest_hash
    )
    ref3 = SimpleNamespace(
        artifact_id=pair_ref.v3_dataset_id, manifest_payload_sha256=pair_ref.v3_manifest_hash
    )
    pair = SimpleNamespace(
        reference=SimpleNamespace(v2=ref2, v3=ref3),
        v2=SimpleNamespace(
            tables={
                RecordTable.ENTRY_CANDIDATE: pd.DataFrame(candidates),
                RecordTable.GEOMETRY_DOSSIER: pd.DataFrame(dossiers),
                RecordTable.SETUP_LIFECYCLE: pd.DataFrame(life),
                RecordTable.CANDIDATE_LABEL: pd.DataFrame(labels),
                RecordTable.EXECUTED_TRADE: pd.DataFrame(columns=["candidate_id"]),
            }
        ),
        v3=SimpleNamespace(
            tables={
                ContextRecordTable.CANDIDATE_CONTEXT_LINK: pd.DataFrame(columns=["candidate_id"])
            }
        ),
    )
    source = SimpleNamespace(
        envelope=SimpleNamespace(
            payload=SimpleNamespace(subject=subject),
            research_context_companion_id=pair_ref.v3_dataset_id,
        ),
        manifest_payload_sha256=pair_ref.v3_manifest_hash,
        forward_bars_sha256=hashlib.sha256(bars.to_parquet(index=False)).hexdigest(),
        bars_1m=bars,
    )
    reads = []

    def load(root, identity):
        reads.append((root, identity))
        assert identity == pair_ref.v3_dataset_id
        return source

    monkeypatch.setattr(research_data, "load_research_context_companion", load)
    monkeypatch.setattr(
        charts,
        "load_verified_label_source_bars",
        lambda *args: pytest.fail("research chart tried the legacy v2 label-source loader"),
    )
    root = tmp_path / "custom-search-store"
    return SimpleNamespace(
        root=root,
        repo=tmp_path / "repo",
        pair=pair,
        pair_ref=pair_ref,
        subject=subject,
        source=source,
        reads=reads,
    )


def _build(fixture):
    return charts.build_replay_chart_artifact(
        fixture.pair,
        repo_root=fixture.repo,
        base_dir=fixture.root / "research_replay_charts",
        corroborate=False,
        research_label_source=(fixture.root, fixture.pair_ref.v3_dataset_id),
    )


def test_research_chart_scopes_candidates_and_omits_legacy_outcomes(research_chart):
    fixture = research_chart
    directory = _build(fixture)
    chart, source = charts.load_verified_research_replay_chart(
        fixture.root,
        directory.name,
        research_subject_id=fixture.subject.subject_id,
        core_replay_id=fixture.subject.core_replay_id,
    )
    assert source is fixture.source
    assert chart.directory == directory
    assert chart.candidate_ranges["candidate_id"].tolist() == ["candidate-1"]
    row = chart.candidate_ranges.iloc[0]
    assert row["display_end_ts"] == pd.Timestamp(fixture.subject.cutoff_ts_utc)
    assert row["display_end_source"] == "authorized_cutoff_censor"
    assert pd.isna(row["candidate_label_resolution_ts"])
    config = chart.manifest["effective_config"]
    assert config["label_annotation_policy"] == "configured_research_labels_separate_v1"
    assert config["development_cutoff_utc"] == fixture.subject.cutoff_ts_utc
    assert not (fixture.repo / charts.REPLAY_CHART_CATALOG).exists()
    catalog_path = fixture.root / "research_replay_chart_catalog.json"
    assert (
        charts.find_replay_artifact(
            charts.read_replay_chart_catalog(catalog_path), fixture.pair_ref
        )
        == directory.name
    )
    catalog_path.unlink()
    assert _build(fixture) == directory
    assert catalog_path.is_file(), "verified reuse must repair a missing discovery catalog"


def test_research_chart_refuses_companion_manifest_mismatch(research_chart):
    fixture = research_chart
    fixture.source.manifest_payload_sha256 = "c" * 64
    with pytest.raises(charts.ReplayChartStoreError, match="another exact capture pair"):
        _build(fixture)
    assert not (fixture.root / "research_replay_charts").exists()


def test_exact_research_chart_reader_refuses_other_subject_and_tampering(research_chart):
    fixture = research_chart
    directory = _build(fixture)
    with pytest.raises(charts.ReplayChartStoreError, match="exact subject"):
        charts.load_verified_research_replay_chart(
            fixture.root,
            directory.name,
            research_subject_id="f" * 64,
            core_replay_id=fixture.subject.core_replay_id,
        )
    (directory / "bars_tf.parquet").write_bytes(b"changed saved bars")
    with pytest.raises(charts.ReplayChartStoreError, match="byte mismatch"):
        charts.load_verified_research_replay_chart(
            fixture.root,
            directory.name,
            research_subject_id=fixture.subject.subject_id,
            core_replay_id=fixture.subject.core_replay_id,
        )


def test_research_chart_with_empty_scoped_candidate_population(research_chart):
    fixture = research_chart
    fixture.pair.v2.tables[RecordTable.ENTRY_CANDIDATE]["is_warmup"] = True
    directory = _build(fixture)
    chart, _ = charts.load_verified_research_replay_chart(
        fixture.root,
        directory.name,
        research_subject_id=fixture.subject.subject_id,
        core_replay_id=fixture.subject.core_replay_id,
    )
    assert chart.candidate_ranges.empty


@pytest.fixture
def panel_replay(tmp_path):
    bars = synthetic_label_source_1m(
        DAYS,
        start_minute=480,
        end_minute=580,
        early_close_day=DAYS[0],
        early_close_minute=578,
        drop_window=(DAYS[1], 550, 552),
    )
    pair = synthetic_pair_ref("scope-panel")
    identity = write_synthetic_replay_chart_artifact(tmp_path, bars, pair)
    return charts.load_verified_replay_chart_artifact(tmp_path, identity, expected_pair=pair)


def test_research_panel_filters_rows_validity_and_excluded_partial_counts(panel_replay, tmp_path):
    full, full_panel, _ = materialize_context_bar_panel(panel_replay, panel_interval_seconds=300)
    scoped, panel, validity = materialize_context_bar_panel(
        panel_replay,
        panel_interval_seconds=300,
        research_subject_id="d" * 64,
        evaluation_dates=(DAYS[1],),
    )
    assert full.excluded_partial_bar_count == 1
    assert scoped.excluded_partial_bar_count == 0
    assert set(panel["trading_day"]) == {DAYS[1]}
    assert not validity.empty and set(validity["row_id"]) <= set(panel["row_id"])
    pd.testing.assert_frame_equal(
        panel.reset_index(drop=True),
        full_panel.loc[full_panel["trading_day"] == DAYS[1]].reset_index(drop=True),
    )
    assert scoped.payload.research_subject_id == "d" * 64
    assert scoped.payload.evaluation_dates == (DAYS[1],)
    assert scoped.context_bar_panel_artifact_id != full.context_bar_panel_artifact_id
    save_context_bar_panel_artifact(tmp_path, scoped, panel, validity)
    saved, saved_frame = load_verified_context_bar_panel(
        tmp_path, scoped.context_bar_panel_artifact_id
    )
    saved_validity = load_context_bar_panel_validity(tmp_path, saved)
    assert saved.payload.research_subject_id == "d" * 64
    assert len(saved_frame) == len(panel) and len(saved_validity) == len(validity)


@pytest.mark.parametrize("dates", [(), (DAYS[1], DAYS[0]), (DAYS[1], DAYS[1])])
def test_research_panel_refuses_invalid_scoped_calendar(panel_replay, dates):
    with pytest.raises(ValueError, match="nonempty, unique and ordered"):
        materialize_context_bar_panel(
            panel_replay,
            panel_interval_seconds=300,
            research_subject_id="d" * 64,
            evaluation_dates=dates,
        )


def test_research_group_ui_api_argument_order_and_result_keys(monkeypatch, tmp_path):
    signatures = {
        "list_source_subjects": ((tmp_path,), {}),
        "build_research_preflight": ((tmp_path, ["a" * 64], {}), {}),
        "freeze_research_group": (
            (tmp_path, {}),
            {"authorization_statement": "Reviewed", "author": "Owner"},
        ),
        "launch_research_group": ((tmp_path, "a" * 64, tmp_path / "state"), {}),
        "read_research_group": ((tmp_path, "a" * 64, tmp_path / "state"), {}),
        "list_research_groups": ((tmp_path, tmp_path / "state"), {}),
        "cancel_research_group": ((tmp_path, "a" * 64, tmp_path / "state"), {}),
        "read_research_regime_eligibility": ((tmp_path, "a" * 64, "b" * 64), {}),
        "promote_research_regime": (
            (tmp_path, "a" * 64, "b" * 64),
            {
                "author": "Owner",
                "approval_statement": "Reviewed",
                "review_id": "c" * 64,
            },
        ),
    }
    for name, (args, kwargs) in signatures.items():
        inspect.signature(getattr(research_runs, name)).bind(*args, **kwargs)
    group = SimpleNamespace(
        payload=SimpleNamespace(
            cells_json=json.dumps([{"pipeline_semantic_id": "b" * 64, "core_replay_id": "c" * 64}]),
            request_json=json.dumps({"evaluation_start": DAYS[0], "regime_study": None}),
            display_name="Saved group",
            research_approval_id="d" * 64,
        )
    )
    monkeypatch.setattr(research_runs, "load_verified_envelope", lambda *args: group)
    monkeypatch.setattr(research_runs, "read_pipeline_state", lambda *args: None)
    result = research_runs.read_research_group(tmp_path, "a" * 64, tmp_path / "state")
    assert result["group_id"] == "a" * 64 and result["status"] == "frozen"
    assert result["study_spec"]["evaluation_start"] == DAYS[0]
    assert result["cells"][0]["status"] == "queued"
    assert result["cells"][0]["artifact_ids"] == []


def _chart_app():
    import ifvg_research_pipeline as ui
    import streamlit as st

    ui.render_research_group(st, roots=ui._TEST_CHART_ROOTS, group_id="e" * 64)


def test_ui_exact_candidate_link_reads_custom_research_chart_and_configured_label(
    research_chart, monkeypatch
):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import ifvg_research_pipeline as ui

    from alpha_lab.agents.data_infra.ifvg import study_providers
    from alpha_lab.agents.data_infra.ifvg.search import store

    fixture = research_chart
    directory = _build(fixture)
    pipeline_id = "c" * 64
    stage = "04_build_or_reuse_replay_charts"
    cell = {
        "pipeline_semantic_id": pipeline_id,
        "subject_id": fixture.subject.subject_id,
        "core_replay_id": fixture.subject.core_replay_id,
        "status": "completed",
        "lane": "R5",
        "state": {
            "stages": {
                stage: {
                    "stage_result_id": "d" * 64,
                    "in_plan": True,
                    "status": "completed",
                    "output_artifact_ids": [directory.name],
                }
            }
        },
    }
    group = {
        "group_id": "e" * 64,
        "status": "completed",
        "display_name": "Research chart",
        "cells": [cell],
    }
    monkeypatch.setattr(
        ui, "_research_api", lambda: SimpleNamespace(read_research_group=lambda *args: group)
    )
    monkeypatch.setattr(
        ui,
        "_TEST_CHART_ROOTS",
        {
            "store_root": fixture.root,
            "pipeline_state_root": fixture.root / "state",
            "repo_root": fixture.repo,
        },
        raising=False,
    )
    monkeypatch.setattr(study_providers, "load_ladder_diagnostics", lambda *args: None)
    monkeypatch.setattr(study_providers, "mbp1_stage_evidence_defaults", lambda *args: ({}, None))
    monkeypatch.setattr(study_providers, "regime_stage_evidence_defaults", lambda *args: ({}, None))
    observed = []

    def load_stage(root, family, identity, cls):
        observed.append((root, family, identity))
        return SimpleNamespace(
            payload=SimpleNamespace(
                pipeline_semantic_id=pipeline_id,
                stage=SimpleNamespace(value=stage),
                output_artifact_ids=(directory.name,),
            )
        )

    monkeypatch.setattr(store, "load_verified_envelope", load_stage)
    label = {
        "candidate_id": "candidate-1",
        "reward_r": 2.0,
        "entry_ticks": 20000,
        "stop_ticks": 19900,
        "target_ticks": 20200,
        "censored": True,
        "label_window_end": fixture.subject.cutoff_ts_utc,
        "resolution_ts_utc": None,
    }
    monkeypatch.setattr(
        ui,
        "_load_research_tables",
        lambda *args: {
            "cohorts": [],
            "trades": [],
            "model_inputs": [],
            "errors": [],
            "labels": [
                {
                    "artifact_id": "f" * 64,
                    "frame": pd.DataFrame([label]),
                    "metadata": {"payload": {"research_subject_id": fixture.subject.subject_id}},
                }
            ],
        },
    )
    app = AppTest.from_function(_chart_app, default_timeout=60).run()
    app.radio(key=ui._RESEARCH + "evidence_panel").set_value("Trades and cohorts").run()
    next(
        button for button in app.button if button.label == "Inspect configured research chart"
    ).click().run()
    assert not app.exception
    assert app.radio(key=ui._RESEARCH + "evidence_panel").value == "Research chart"
    assert app.selectbox(key=ui._RESEARCH + pipeline_id + "_chart_candidate").value == "candidate-1"
    assert observed == [(fixture.root, "pipeline_stage_results", "d" * 64)]
    assert len(app.get("plotly_chart")) == 1
    assert not (fixture.repo / charts.REPLAY_CHART_CATALOG).exists()
    chart, source = charts.load_verified_research_replay_chart(
        fixture.root,
        directory.name,
        research_subject_id=fixture.subject.subject_id,
        core_replay_id=fixture.subject.core_replay_id,
    )
    figure, _ = ui._research_candidate_figure(
        chart, source, chart.candidate_ranges.iloc[0], label, 60
    )
    assert [shape.y0 for shape in figure.layout.shapes] == [5000.0, 4975.0, 5050.0]


def _source_progress_app():
    import ifvg_research_pipeline as ui
    import streamlit as st

    ui._render_research_source_progress(st, ui._TEST_SOURCE_PROGRESS, "b" * 64)


@pytest.mark.parametrize("invocations", [None, 0, 1])
def test_source_context_day_progress_and_reuse(monkeypatch, invocations):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    import ifvg_research_pipeline as ui

    child = {
        "core_replay_id": "b" * 64,
        "state": "running" if invocations is None else "reused",
        "context_progress": {
            "completed_days": 23,
            "total_days": 117,
            "trading_day": "2026-02-04",
        },
    }
    if invocations is not None:
        child["context_replay_invocations"] = invocations
    monkeypatch.setattr(ui, "_TEST_SOURCE_PROGRESS", {"children": [child]}, raising=False)
    app = AppTest.from_function(_source_progress_app).run()
    assert not app.exception
    progress = app.get("progress")[0]
    assert "23 of 117 days" in progress.proto.text
    assert "2026-02-04" in progress.proto.text
    rendered = " ".join(item.value for item in app.markdown)
    if invocations == 0:
        assert "exact context companion verified and reused" in rendered
    elif invocations == 1:
        assert "context capture replay completed" in rendered
    else:
        assert "reused" not in rendered


def _headline_app():
    import ifvg_research_pipeline as ui
    import streamlit as st

    ui._model_details(
        st,
        ui._TEST_MODEL_RUNGS,
        {},
        preferred_protocol="headline_protocol",
        selection_key="headline_review",
    )


def test_headline_review_selects_frozen_protocol_and_retains_every_rung(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts"))
    import ifvg_research_pipeline as ui

    rungs = {
        name: {"prediction_report": {}} for name in ("prevalence", "other", "headline_protocol")
    }
    monkeypatch.setattr(ui, "_TEST_MODEL_RUNGS", rungs, raising=False)
    app = AppTest.from_function(_headline_app).run()
    assert not app.exception
    model = app.selectbox(key="headline_review")
    assert model.value == "headline_protocol"
    assert len(model.options) == 3
    model.set_value("other").run()
    assert not app.exception
    assert app.selectbox(key="headline_review").value == "other"
