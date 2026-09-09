"""Real adapter transformations exercised only on temporary parquet fixtures."""

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.development_access import DevelopmentReplayPolicy
from alpha_lab.agents.data_infra.ifvg.features.mbp1_arrow_schemas import MBP1_SOURCE_EVENT_SCHEMA
from alpha_lab.agents.data_infra.ifvg.search.research_mbp1 import (
    build_research_mbp1_evidence,
    import_research_mbp1_receipt,
    preflight_research_mbp1,
    read_research_mbp1_day,
    research_mbp1_contract,
    research_mbp1_physical_dates,
)
from tests.agents.ifvg_search.test_research_subject_data import subject_fixture


def raw(ts, *, instrument=1, action="T", bid=20000.0):
    row = {field.name: 0 for field in MBP1_SOURCE_EVENT_SCHEMA}
    row.update(
        ts_event=pd.Timestamp(ts),
        ts_recv=pd.Timestamp(ts),
        instrument_id=instrument,
        publisher_id=1,
        action=action,
        side="B",
        price=bid,
        bid_px_00=bid,
        ask_px_00=bid + 0.25,
        bid_sz_00=4,
        ask_sz_00=5,
        bid_ct_00=1,
        ask_ct_00=1,
        size=1,
        symbol="NQH6",
        sequence=1,
    )
    return row


def write_day(root, day, rows):
    path = root / "data/databento/NQ" / day / "mbp1.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_adapter_joins_evening_and_day_and_scales_dollar_exports(tmp_path):
    subject = subject_fixture()
    write_day(
        tmp_path,
        "2026-01-12",
        [
            raw("2026-01-12T22:59:00Z"),  # before 18:00 ET
            raw("2026-01-12T23:01:00Z"),
        ],
    )
    write_day(
        tmp_path,
        "2026-01-13",
        [
            raw("2026-01-13T00:01:00Z"),
            raw("2026-01-13T12:01:00Z"),
            raw("2026-01-13T12:02:00Z", instrument=2),
            raw("2026-01-13T23:01:00Z"),  # next trading day
        ],
    )
    frame = read_research_mbp1_day(
        subject,
        tmp_path,
        "2026-01-13",
        DevelopmentReplayPolicy(subject.replay_dates),
    )
    assert len(frame) == 3
    assert frame.instrument_id.unique().tolist() == [1]
    assert frame.bid_px_ticks.tolist() == pytest.approx([80000] * 3)
    assert frame.ask_px_ticks.tolist() == pytest.approx([80001] * 3)
    assert len(frame.attrs["source_partition_refs"]) == 2
    assert "dominant_trade_count" in research_mbp1_contract().contract_roll_policy_id


def test_preflight_only_reads_metadata_and_preserves_unknown_coverage(tmp_path, monkeypatch):
    subject = subject_fixture()
    write_day(tmp_path, "2026-01-13", [raw("2026-01-13T12:01:00Z")])
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("must not decode events"))
    report = preflight_research_mbp1(subject, tmp_path, tmp_path)
    assert not report["passed"]
    assert report["available_partition_count"] == 1
    assert report["evidence_refs"] == []
    assert any("completeness" in item for item in report["warnings"])
    assert any("receipt" in item for item in report["blockers"])


def test_unauthorized_day_is_refused_before_path(tmp_path):
    subject = subject_fixture()
    with pytest.raises(PermissionError):
        read_research_mbp1_day(
            subject,
            tmp_path,
            "2026-06-12",
            DevelopmentReplayPolicy(subject.replay_dates),
        )


def test_empty_source_day_stays_empty_and_post_cutoff_trades_do_not_select_contract(tmp_path):
    subject = subject_fixture()
    empty = read_research_mbp1_day(
        subject, tmp_path, "2026-01-13", DevelopmentReplayPolicy(subject.replay_dates)
    )
    assert empty.empty
    write_day(
        tmp_path,
        "2026-01-13",
        [
            raw("2026-01-13T12:00:00Z", instrument=1),
            *[raw("2026-01-13T23:00:00Z", instrument=2) for _ in range(3)],
        ],
    )
    bounded = read_research_mbp1_day(
        subject, tmp_path, "2026-01-13", DevelopmentReplayPolicy(subject.replay_dates)
    )
    assert bounded.instrument_id.tolist() == [1]


def receipt_for(subject, path, day):
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_evidence import _example_scope

    midnight = pd.Timestamp(day, tz="UTC")
    scope = _example_scope().model_copy(
        update={
            "utc_date": day,
            "source_partition_id": f"databento/NQ/{day}/mbp1",
            "physical_partition_key": f"{day}/mbp1",
            "partition_expected_start_ts": midnight.value,
            "partition_expected_end_ts": (midnight + pd.Timedelta(days=1)).value,
        }
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    document = {
        "schema_version": 1,
        "scope": scope.model_dump(mode="json"),
        "partition_sha256": sha,
        "complete_outside_intervals": True,
        "intervals": [],
        "extraction_description": "Temporary test extraction receipt",
    }
    document_sha = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "source_document": document,
        "owner_review": {
            "subject_id": subject.subject_id,
            "source_document_sha256": document_sha,
            "partition_sha256": sha,
            "approve_completeness": True,
            "reviewed_by": "fixture",
            "reviewed_at": "2026-09-08T12:00:00Z",
        },
    }


def test_receipt_compiler_binds_owner_document_source_and_discovers(tmp_path):
    subject = subject_fixture()
    path = write_day(tmp_path, "2026-01-13", [raw("2026-01-13T12:01:00Z")])
    receipt = receipt_for(subject, path, "2026-01-13")
    imported = import_research_mbp1_receipt(subject, tmp_path, tmp_path, receipt)
    preflight = preflight_research_mbp1(subject, tmp_path, tmp_path)
    assert preflight["passed"]
    assert preflight["coverage_evidence_ids"] == [imported["coverage_evidence_id"]]
    receipt["owner_review"]["approve_completeness"] = False
    with pytest.raises(ValueError, match="explicit"):
        import_research_mbp1_receipt(subject, tmp_path, tmp_path, receipt)


def test_physical_sunday_is_required_even_without_logical_sunday():
    subject = subject_fixture().model_copy(
        update={
            "replay_dates": (*subject_fixture().replay_dates, "2026-01-19"),
            "artifact_provenance_dates": (*subject_fixture().replay_dates, "2026-01-19"),
            "evaluation_dates": ("2026-01-19",),
        }
    )
    assert research_mbp1_physical_dates(subject) == ("2026-01-18", "2026-01-19")


def test_real_adapter_builds_evidenced_day_and_rejects_changed_approved_source(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable

    subject = subject_fixture().model_copy(update={"evaluation_dates": ("2026-01-13",)})
    for day, ts in (("2026-01-12", "2026-01-12T23:01:00Z"), ("2026-01-13", "2026-01-13T12:01:00Z")):
        path = write_day(tmp_path, day, [raw(ts)])
        import_research_mbp1_receipt(subject, tmp_path, tmp_path, receipt_for(subject, path, day))
    frozen = preflight_research_mbp1(subject, tmp_path, tmp_path)
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "candidate",
                "setup_id": "setup",
                "trading_day": "2026-01-13",
                "geometry_entry_bar_logical_close_ts_utc": pd.Timestamp("2026-01-13T12:01:00Z"),
            }
        ]
    )
    preparation = SimpleNamespace(
        subject=subject,
        store_root=tmp_path,
        repo_root=tmp_path,
        mbp1_preflight=frozen,
        prepare=lambda: None,
        candidate_view=SimpleNamespace(frame=candidates),
        v2_tables={RecordTable.ENTRY_CANDIDATE: candidates},
        mbp1_evidence={},
    )
    source, events, anchors = build_research_mbp1_evidence(
        preparation, contract=research_mbp1_contract()
    )
    assert len(source.payload.ordered_partitions) == 2
    assert all(
        p.completeness_status.value == "evidenced_complete"
        for p in source.payload.ordered_partitions
    )
    assert len(events["2026-01-13"]) == 2
    assert anchors.iloc[0].entry_ts_utc == pd.Timestamp("2026-01-13T12:01:00Z")
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="schema|receipt|changed"):
        build_research_mbp1_evidence(preparation, contract=research_mbp1_contract())
