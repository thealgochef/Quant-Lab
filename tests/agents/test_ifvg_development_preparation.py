"""Development date access and persisted preparation process tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
    ArtifactPreparationStatus,
)
from alpha_lab.agents.data_infra.ifvg.development_access import (
    AccessOperation,
    DevelopmentDataAccess,
)
from alpha_lab.agents.data_infra.ifvg.preparation import (
    PreparationCancelledError,
    _publish_or_reuse_verified,
    run_resumable_preparation_job,
)


def test_protected_date_is_denied_before_path_construction(tmp_path: Path) -> None:
    access = DevelopmentDataAccess(allowed_dates=("2026-06-10",))
    with pytest.raises(PermissionError):
        access.construct_path("2026-06-11", lambda day: tmp_path / day)
    report = access.audit.as_dict()
    assert report["event_count"] == 0
    assert all(
        value == 0
        for source_class in report["protected_counters"].values()
        for value in source_class.values()
    )


def test_audit_counters_are_derived_from_events(tmp_path: Path) -> None:
    access = DevelopmentDataAccess(allowed_dates=("2026-06-10",))
    path = access.construct_path("2026-06-10", lambda day: tmp_path / day)
    assert path.name == "2026-06-10"
    access.audit.record(AccessOperation.CONTEXT_ROW, "2026-06-10", units=7)
    report = access.audit.as_dict()
    counts = report["counts_by_source_class"]["exposed_development_only"]
    assert counts["path_construction"] == 1
    assert counts["context_row"] == 7


def test_resumable_job_checkpoints_locks_and_cancels_at_boundary(tmp_path: Path) -> None:
    processed: list[str] = []
    root = tmp_path / "jobs"
    state = run_resumable_preparation_job(
        profile_name="profile",
        dates=("2026-01-01", "2026-01-02"),
        process_date=processed.append,
        finalize=lambda: ("1" * 64, "2" * 64),
        job_root=root,
    )
    assert state.status is ArtifactPreparationStatus.CONTEXT_READY
    assert processed == ["2026-01-01", "2026-01-02"]

    cancelled_root = tmp_path / "cancelled"
    job_dir = cancelled_root / "profile"
    job_dir.mkdir(parents=True)
    (job_dir / "cancel.requested").touch()
    with pytest.raises(PreparationCancelledError):
        run_resumable_preparation_job(
            profile_name="profile",
            dates=("2026-01-01",),
            process_date=lambda _day: pytest.fail("processed after cancellation"),
            finalize=lambda: ("1" * 64, "2" * 64),
            job_root=cancelled_root,
        )


def test_publish_reuses_json_equivalent_immutable_identity(tmp_path: Path) -> None:
    exploration = tmp_path / "artifact" / "exploration"

    def publisher() -> Path:
        raise FileExistsError("already published")

    reused = _publish_or_reuse_verified(
        publisher=publisher,
        loader=lambda _base, _dataset_id: SimpleNamespace(
            manifest={"identity": {"nested": [["context_state", "abc"]]}},
            exploration_dir=exploration,
        ),
        base_dir=tmp_path,
        dataset_id="a" * 64,
        expected_identity={"nested": (("context_state", "abc"),)},
    )

    assert reused == exploration


def test_publish_rejects_different_immutable_identity(tmp_path: Path) -> None:
    def publisher() -> Path:
        raise FileExistsError("already published")

    with pytest.raises(RuntimeError, match="different identity"):
        _publish_or_reuse_verified(
            publisher=publisher,
            loader=lambda _base, _dataset_id: SimpleNamespace(
                manifest={"identity": {"feature_formula_version": "v1"}},
                exploration_dir=tmp_path / "artifact" / "exploration",
            ),
            base_dir=tmp_path,
            dataset_id="a" * 64,
            expected_identity={"feature_formula_version": "v2"},
        )
