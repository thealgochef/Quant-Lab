"""Exact-child subject and real shared outcome regressions; no real sources."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from strategy_core.strategies.ifvg_smc.section import default_ifvg_smc_section, ifvg_profile_hash

from alpha_lab.agents.data_infra.ifvg.development_access import FROZEN_WARMUP_DATES
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    CoreStrategyReplayIdentity,
    _example_core_replay_payload,
)
from alpha_lab.agents.data_infra.ifvg.search.research_data import (
    LABEL_POLICY,
    ResearchPreparation,
    build_configured_r_labels,
)
from alpha_lab.agents.data_infra.ifvg.search.research_subject import ResearchSubject


def subject_fixture():
    section = default_ifvg_smc_section().model_copy(update={"tp_r_multiple": 2.5})
    payload = _example_core_replay_payload().model_copy(
        update={
            "resolved_section_config_hash": ifvg_profile_hash(section),
        }
    )
    core = CoreStrategyReplayIdentity.from_payload(payload)
    days = (*FROZEN_WARMUP_DATES, "2026-01-13", "2026-01-14")
    return ResearchSubject(
        core_replay_id=core.core_replay_id,
        v2_dataset_id="b" * 64,
        v2_manifest_hash="c" * 64,
        replay_input_bundle_id=payload.replay_input_bundle_id,
        section_config_hash=ifvg_profile_hash(section),
        section_json=section.model_dump_json(),
        core_envelope_json=core.model_dump_json(),
        neutrality_report_id="d" * 64,
        original_search_id="e" * 64,
        child_spec_json=json.dumps({"ordinal": 1}),
        replay_dates=days,
        warmup_dates=FROZEN_WARMUP_DATES,
        evaluation_dates=("2026-01-13", "2026-01-14"),
        artifact_provenance_dates=days,
    )


def bar(bar_id, ts, high=102, low=99):
    return dict(
        bar_id=bar_id,
        trading_day=ts[:10],
        close_ts_utc=pd.Timestamp(ts),
        open_ts_utc=pd.Timestamp(ts) - pd.Timedelta(minutes=1),
        open_ticks=100,
        high_ticks=high,
        low_ticks=low,
        close_ticks=100,
        kind="time",
        timeframe_ticks=60,
    )


def candidate(**updates):
    return dict(
        candidate_id="candidate",
        setup_id="setup",
        trading_day="2026-01-13",
        direction="LONG",
        entry_ticks=100,
        proposed_stop_ticks=97,
        geometry_entry_bar_bar_id="entry",
        **updates,
    )


def test_subject_hash_binds_empty_calendar_days_and_cohort():
    subject = subject_fixture()
    assert subject.cohort == "all_candidates"
    assert ResearchSubject.model_validate_json(subject.model_dump_json()) == subject
    assert (
        subject.model_copy(update={"evaluation_dates": ("2026-01-13",)}).subject_id
        != subject.subject_id
    )
    assert (
        subject.model_copy(update={"cohort": "eligible_decisions"}).subject_id != subject.subject_id
    )
    with pytest.raises(ValueError, match="hash"):
        subject.model_copy(update={"section_json": default_ifvg_smc_section().model_dump_json()})


def test_configured_target_uses_core_rounding_and_cross_day_resolution():
    bars = pd.DataFrame(
        [
            bar("entry", "2026-01-13T20:59:00Z", high=150, low=1),
            bar("later", "2026-01-14T00:01:00Z", high=108, low=99),
        ]
    )
    policy, rows = build_configured_r_labels(
        pd.DataFrame([candidate()]),
        bars,
        r_multiple=2.5,
        cost_points=0.25,
        label_source_id="fixture",
    )
    row = rows.iloc[0]
    assert policy == LABEL_POLICY
    assert row.target_ticks == 108  # ceil(3 ticks * 2.5 R)
    assert row.binary_target == 1
    assert row.gross_r == pytest.approx(8 / 3)
    assert row.net_r == pytest.approx(7 / 3)
    assert row.resolution_bar_id == "later"
    assert row.label_window_end == pd.Timestamp("2026-01-14T00:01:00Z")


def test_stop_first_and_protected_cutoff_are_shared_kernel_semantics():
    bars = pd.DataFrame(
        [
            bar("entry", "2026-06-10T20:58:00Z"),
            bar("collision", "2026-06-10T20:59:00Z", high=120, low=90),
            bar("cutoff", "2026-06-10T21:00:00Z", high=120, low=99),
        ]
    )
    _, rows = build_configured_r_labels(
        pd.DataFrame([candidate()]),
        bars,
        r_multiple=3,
        label_source_id="fixture",
    )
    assert rows.iloc[0].binary_target == 0
    assert rows.iloc[0].gross_r == -1
    _, censored = build_configured_r_labels(
        pd.DataFrame([candidate()]),
        bars.drop(index=1),
        r_multiple=3,
        label_source_id="fixture",
    )
    assert censored.iloc[0].censored
    assert pd.isna(censored.iloc[0].gross_r)
    assert pd.isna(censored.iloc[0].binary_target)


def test_geometry_unavailable_candidates_remain_explicit():
    row = candidate()
    row.pop("geometry_entry_bar_bar_id")
    _, labels = build_configured_r_labels(
        pd.DataFrame([row]),
        pd.DataFrame([bar("entry", "2026-01-13T20:59:00Z")]),
        r_multiple=1,
        label_source_id="fixture",
    )
    assert len(labels) == 1
    assert not labels.iloc[0].entry_available
    assert labels.iloc[0].censor_reason == "geometry_unavailable"


def test_preparation_constructor_never_reads_or_replays(monkeypatch, tmp_path):
    from alpha_lab.agents.data_infra.ifvg.search import research_data

    monkeypatch.setattr(
        research_data, "load_search_review_evidence", lambda *a: pytest.fail("read")
    )
    monkeypatch.setattr(research_data, "build_ifvg_v3_capture", lambda *a: pytest.fail("replay"))
    preparation = ResearchPreparation(subject_fixture(), tmp_path, tmp_path)
    assert preparation._candidate_view is None


def test_exact_input_mutation_refuses_before_capture(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig

    subject = subject_fixture()
    cfg = IfvgCaptureConfig(data_dir=tmp_path)
    path = cfg.bars_path("2026-01-13")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed")
    evidence = SimpleNamespace(
        inputs=SimpleNamespace(
            payload=SimpleNamespace(
                ordered_day_artifacts=[
                    SimpleNamespace(
                        trading_day="2026-01-13",
                        artifact_kind="bars",
                        artifact_id=path.name,
                        content_sha256="a" * 64,
                    )
                ],
            )
        )
    )
    with pytest.raises(ValueError, match="exact saved child"):
        ResearchPreparation(subject, tmp_path, tmp_path)._input_hashes(evidence, cfg)


def test_candidate_scope_keeps_occupied_slot_and_zero_candidate_calendar_days():
    from alpha_lab.agents.data_infra.ifvg.context_feature_view import CandidateFeatureView
    from alpha_lab.agents.data_infra.ifvg.search.research_data import scope_research_candidate_view

    subject = subject_fixture()
    frame = pd.DataFrame(
        {
            "candidate_id": ["warmup", "executed", "occupied"],
            "trading_day": ["2026-01-12", "2026-01-13", "2026-01-13"],
            "is_warmup": [True, False, False],
            "ctx_sweep_qualifying_link_count": [0, 1, 0],
        }
    )
    candidates = frame.assign(block_reasons=[(), (), ("single_trade_slot_occupied",)])
    full = CandidateFeatureView("full", "pair", "registry", frame, {}, "initial")
    decisions = pd.DataFrame({"candidate_id": ["executed"]})
    scoped = scope_research_candidate_view(full, candidates, decisions, subject)
    assert scoped.frame.candidate_id.tolist() == ["executed", "occupied"]
    assert scoped.frame.was_eligible_decision.tolist() == [True, False]
    assert scoped.frame.iloc[1].block_reasons == ("single_trade_slot_occupied",)
    assert subject.evaluation_dates == ("2026-01-13", "2026-01-14")
    assert len(full.frame) == 3
    invalid = candidates.assign(is_warmup=["true", "false", "false"])
    with pytest.raises(ValueError, match="booleans"):
        scope_research_candidate_view(full, invalid, decisions, subject)


def test_companion_typed_reload_and_forward_source_tamper(tmp_path):
    from io import BytesIO

    import pyarrow.parquet as pq

    from alpha_lab.agents.data_infra.ifvg.context_schemas import context_table_from_frame
    from alpha_lab.agents.data_infra.ifvg.dataset import reconcile_v3_core_to_accepted_v2
    from alpha_lab.agents.data_infra.ifvg.search.research_data import (
        CONTEXT_STORE,
        ResearchContextEnvelope,
        ResearchContextPayload,
        load_research_context_companion,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (
        SidecarLoadError,
        save_or_reuse_envelope,
    )
    from tests.agents.ifvg_v3_fixtures import context_fixture

    _day, context, core, _emissions = context_fixture()
    envelope = ResearchContextEnvelope.from_payload(
        ResearchContextPayload(
            subject=subject_fixture(),
            producer_source_hash="a" * 64,
            feature_schema_hash="b" * 64,
            context_config_hash="c" * 64,
            context_config_json="{}",
        )
    )
    sidecars = {}
    for table, frame in context.items():
        stream = BytesIO()
        pq.write_table(context_table_from_frame(table, frame), stream)
        sidecars[f"{table.value}.parquet"] = stream.getvalue()
    sidecars["neutrality.json"] = json.dumps(reconcile_v3_core_to_accepted_v2(core, core)).encode()
    sidecars["forward_bars.parquet"] = pd.DataFrame(
        [bar("entry", "2026-01-13T12:00:00Z")]
    ).to_parquet(index=False)
    save_or_reuse_envelope(tmp_path, CONTEXT_STORE, envelope, extra_files=sidecars)
    loaded = load_research_context_companion(tmp_path, envelope.research_context_companion_id)
    assert set(loaded.context_tables) == set(context)
    assert loaded.bars_1m.bar_id.tolist() == ["entry"]
    (loaded.directory / "forward_bars.parquet").write_bytes(b"mutated")
    with pytest.raises(SidecarLoadError, match="verification|mismatch"):
        load_research_context_companion(tmp_path, envelope.research_context_companion_id)


def test_study_end_cutoff_censors_later_resolution_and_binds_label_source():
    from alpha_lab.agents.data_infra.ifvg.search.research_data import _validate_forward_bars

    subject = subject_fixture().model_copy(update={"evaluation_dates": ("2026-01-13",)})
    assert subject.cutoff_ts_utc == "2026-01-13T22:00:00Z"
    bars = pd.DataFrame(
        [
            bar("entry", "2026-01-13T21:59:00Z"),
            bar("after", "2026-01-14T12:00:00Z", high=120),
        ]
    )
    _, labels = build_configured_r_labels(
        pd.DataFrame([candidate()]),
        bars,
        r_multiple=2.5,
        cutoff_ts_utc=subject.cutoff_ts_utc,
        label_source_id="fixture",
    )
    assert labels.iloc[0].censored
    assert labels.iloc[0].label_window_end == pd.Timestamp(subject.cutoff_ts_utc)
    with pytest.raises(ValueError, match="cutoff"):
        _validate_forward_bars(bars, subject)


@pytest.mark.parametrize("include_test", [True, False])
def test_logical_test_calendar_boundary_purges_sparse_or_empty_first_day(include_test):
    from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds

    days = tuple(ts.date().isoformat() for ts in pd.date_range("2026-01-13", periods=45))
    test_day = pd.Timestamp(days[40], tz="UTC")
    rows = []
    for index in range(38):
        entry = pd.Timestamp(days[index], tz="UTC") + pd.Timedelta(hours=14)
        rows.append(
            {
                "candidate_id": f"c{index}",
                "setup_id": f"s{index}",
                "trading_day": days[index],
                "entry_ts_utc": entry,
                "resolution_ts_utc": test_day if index == 0 else entry + pd.Timedelta(minutes=1),
                "entry_available": True,
                "resolution_available": True,
                "binary_target": index % 2,
            }
        )
    if include_test:
        rows.append(
            {
                "candidate_id": "test",
                "setup_id": "test",
                "trading_day": days[41],
                "entry_ts_utc": test_day + pd.Timedelta(days=1, hours=14),
                "resolution_ts_utc": test_day + pd.Timedelta(days=1, hours=15),
                "entry_available": True,
                "resolution_available": True,
                "binary_target": 1,
            }
        )
    legacy = build_context_folds(pd.DataFrame(rows), authorized_trading_days=days)
    exact = build_context_folds(
        pd.DataFrame(rows), authorized_trading_days=days, purge_from_logical_test_start=True
    )
    assert "c0" in legacy.folds[0].train_candidate_ids
    assert "c0" not in exact.folds[0].train_candidate_ids
    assert exact.folds[0].purged_candidate_ids == ("c0",)


def test_prepare_publishes_companion_once_and_reuses_exact_saved_pair(tmp_path, monkeypatch):
    from pathlib import Path

    from alpha_lab.agents.data_infra.ifvg.artifact_io import VerifiedIfvgArtifact
    from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import ArtifactReference
    from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
    from alpha_lab.agents.data_infra.ifvg.dataset import reconcile_v3_core_to_accepted_v2
    from alpha_lab.agents.data_infra.ifvg.search import research_data
    from tests.agents.ifvg_v3_fixtures import _bar, context_fixture

    subject = subject_fixture()
    _day, tables, core, _emissions = context_fixture()
    for table in (RecordTable.ENTRY_CANDIDATE, RecordTable.ELIGIBLE_DECISION):
        core[table]["is_warmup"] = False
        core[table]["trading_day"] = core[table]["envelope_trading_day"]
    reference = ArtifactReference(
        artifact_id=subject.v2_dataset_id,
        manifest_payload_sha256=subject.v2_manifest_hash,
        artifact_kind="v2",
        dataset_schema_version=2,
        profile_hash=subject.section_config_hash,
    )
    artifact = VerifiedIfvgArtifact(
        reference,
        tmp_path,
        {"identity": {"evaluation_config_hash": "a" * 64}},
        core,
        {"effective_config.json": {"evaluator": {}}},
    )
    evidence = SimpleNamespace(dataset=artifact)
    monkeypatch.setattr(research_data, "preflight_research_subject", lambda *a: None)
    monkeypatch.setattr(research_data, "load_search_review_evidence", lambda *a: evidence)
    monkeypatch.setattr(ResearchPreparation, "_input_hashes", lambda *a: [])
    calls = []

    def capture(days, cfg, resolved, **kwargs):
        assert kwargs["accepted_v2_tables"] is core
        calls.append(days)
        bars = {day: () for day in days}
        bars["2026-01-13"] = (_bar(5, 10015, 10018, 10010, 10016),)
        return SimpleNamespace(
            context_tables=tables,
            bars_by_day=bars,
            baseline_reconciliation=reconcile_v3_core_to_accepted_v2(core, core),
        )

    monkeypatch.setattr(research_data, "build_ifvg_v3_capture", capture)
    repo = Path(__file__).resolve().parents[3]
    first = ResearchPreparation(subject, tmp_path, repo).prepare()
    assert len(calls) == first.replay_invocations == 1
    assert first.pair.v2 is artifact
    assert len(first.candidate_view.frame) == len(core[RecordTable.ENTRY_CANDIDATE])
    assert first.label_source_reference["path"].endswith("forward_bars.parquet")
    second = ResearchPreparation(subject, tmp_path, repo).prepare()
    assert len(calls) == 1
    assert second.replay_invocations == 0
    assert second.context_reused
    assert second.candidate_view.view_id == first.candidate_view.view_id
    assert second.label_source_reference == first.label_source_reference


def test_loaded_core_verification_normalizes_lines_and_refuses_content_or_module_drift(
    tmp_path, monkeypatch
):
    from alpha_lab.agents.data_infra.ifvg.search import research_subject

    repo = tmp_path / "Claude-Quant-Lab"
    checkout = tmp_path / "Strategy-Core/src/strategy_core"
    loaded = tmp_path / "installed/strategy_core"
    checkout.mkdir(parents=True)
    loaded.mkdir(parents=True)
    (checkout / "__init__.py").write_bytes(b"VALUE = 1\r\n")
    (loaded / "__init__.py").write_bytes(b"VALUE = 1\n")
    monkeypatch.setattr(research_subject, "_loaded_core_package_root", lambda: loaded)
    normalized = research_subject.verify_loaded_core_source(repo)
    assert len(normalized) == 64
    monkeypatch.setattr(research_subject, "_loaded_core_package_root", lambda: checkout)
    assert research_subject.verify_loaded_core_source(repo) == normalized
    monkeypatch.setattr(research_subject, "_loaded_core_package_root", lambda: loaded)
    (loaded / "__init__.py").write_bytes(b"VALUE = 2\n")
    with pytest.raises(ValueError, match="loaded Strategy-Core differs"):
        research_subject.verify_loaded_core_source(repo)
    (loaded / "__init__.py").write_bytes(b"VALUE = 1\n")
    (loaded / "extra.py").write_bytes(b"VALUE = 1\n")
    with pytest.raises(ValueError, match="extra.py"):
        research_subject.verify_loaded_core_source(repo)
