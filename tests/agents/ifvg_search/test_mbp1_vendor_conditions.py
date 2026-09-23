"""Archived negative vendor evidence survives receipt discovery and freezing."""

import hashlib
import json
import zipfile

import pytest

from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_evidence import (
    compute_partition_coverage,
)
from alpha_lab.agents.data_infra.ifvg.search.mbp1_vendor_conditions import (
    load_archived_dataset_conditions,
)
from alpha_lab.agents.data_infra.ifvg.search.research_mbp1 import (
    _discover_evidence,
    import_research_mbp1_receipt,
    preflight_research_mbp1,
)
from tests.agents.ifvg_search.test_research_mbp1_adapter import raw, receipt_for, write_day
from tests.agents.ifvg_search.test_research_subject_data import subject_fixture


def archive_fixture(root, condition, *, name="GLBX-fixture.zip", corrupt=False):
    path = root / "data/databento" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    documents = {
        "metadata.json": json.dumps(
            {"query": {"dataset": "GLBX.MDP3", "schema": "mbp-1"}}
        ).encode(),
        "condition.json": json.dumps(
            [{"date": "2026-01-13", "condition": condition, "last_modified_date": "2026-01-14"}]
        ).encode(),
    }
    manifest = {
        "files": [
            {
                "filename": name,
                "hash": f"sha256:{hashlib.sha256(data).hexdigest()}",
                "size": len(data),
            }
            for name, data in documents.items()
        ]
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in documents.items():
            archive.writestr(name, data + (b" " if corrupt and name == "condition.json" else b""))
        archive.writestr("never-open-this.mbp-1.dbn.zst", b"not needed")
    return path


def test_degradation_reaches_coverage_even_with_owner_reviewed_receipt(tmp_path):
    subject = subject_fixture().model_copy(update={"evaluation_dates": ("2026-01-13",)})
    path = write_day(tmp_path, "2026-01-13", [raw("2026-01-13T12:01:00Z")])
    import_research_mbp1_receipt(
        subject, tmp_path, tmp_path, receipt_for(subject, path, "2026-01-13")
    )
    archive_fixture(tmp_path, "degraded")
    preflight = preflight_research_mbp1(subject, tmp_path, tmp_path)
    assert not preflight["passed"]
    assert preflight["potentially_usable_logical_days"] == []
    assert any("degraded: 2026-01-13" in warning for warning in preflight["warnings"])
    conditions, _ = load_archived_dataset_conditions(tmp_path, ("2026-01-13",))
    evidence, _ = _discover_evidence(
        tmp_path, subject, preflight["input_partition_refs"], conditions
    )
    coverage = compute_partition_coverage(
        evidence["2026-01-13"][0],
        trading_day="2026-01-13",
        partition_content_refs=(hashlib.sha256(path.read_bytes()).hexdigest(),),
    )
    assert coverage.coverage_fraction == 0
    assert coverage.completeness_status.value == "completeness_unknown"
    assert coverage.dataset_condition_status.value == "vendor_dataset_degraded"
    assert any(source.startswith("dataset_condition:") for source in coverage.evidence_sources)


def test_available_never_supplies_completeness_and_reads_no_dbn(tmp_path, monkeypatch):
    archive_fixture(tmp_path, "available")
    original = zipfile.ZipFile.open

    def guarded(self, name, *args, **kwargs):
        assert str(name).endswith(".json"), "metadata discovery must never decode DBN"
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", guarded)
    subject = subject_fixture().model_copy(update={"evaluation_dates": ("2026-01-13",)})
    write_day(tmp_path, "2026-01-13", [raw("2026-01-13T12:01:00Z")])
    result = preflight_research_mbp1(subject, tmp_path, tmp_path)
    assert result["dataset_condition_records"]["2026-01-13"]["condition"] == "available"
    assert not result["passed"]
    assert result["coverage_evidence_ids"] == []
    assert result["potentially_usable_logical_days"] == []


def test_condition_checksum_failure_blocks_preflight(tmp_path):
    archive_fixture(tmp_path, "available", corrupt=True)
    with pytest.raises(ValueError, match="checksum"):
        load_archived_dataset_conditions(tmp_path, ("2026-01-13",))
    result = preflight_research_mbp1(subject_fixture(), tmp_path, tmp_path)
    assert not result["passed"]
    assert any("checksum" in blocker for blocker in result["blockers"])


def test_another_archive_cannot_clear_negative_evidence_and_dates_stay_scoped(tmp_path):
    archive_fixture(tmp_path, "degraded")
    archive_fixture(tmp_path, "available", name="GLBX-other.zip")
    result, _ = load_archived_dataset_conditions(tmp_path, ("2026-01-13",))
    assert result["2026-01-13"].condition == "degraded"
    assert load_archived_dataset_conditions(tmp_path, ("2026-01-14",))[0] == {}
