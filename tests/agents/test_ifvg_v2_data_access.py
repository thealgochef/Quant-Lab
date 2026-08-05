"""Pre-I/O date policy and zero-forbidden-access spies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.data_access import (
    EXPLORATION_DATE_ALLOWLIST,
    DataAccessAudit,
    ExplorationDataPolicy,
    require_fixed_exploration_allowlist,
)


def test_exact_exploration_allowlist_and_ordering() -> None:
    assert len(EXPLORATION_DATE_ALLOWLIST) == 26
    assert "2026-01-03" not in EXPLORATION_DATE_ALLOWLIST
    assert "2026-01-10" not in EXPLORATION_DATE_ALLOWLIST
    assert "2026-01-17" not in EXPLORATION_DATE_ALLOWLIST
    assert "2026-01-24" not in EXPLORATION_DATE_ALLOWLIST
    policy = ExplorationDataPolicy()
    assert policy.authorize_dates(sorted(EXPLORATION_DATE_ALLOWLIST)) == tuple(
        sorted(EXPLORATION_DATE_ALLOWLIST)
    )
    with pytest.raises(ValueError, match="chronological"):
        policy.authorize_dates(["2026-01-13", "2026-01-12"])
    with pytest.raises(ValueError, match="unique"):
        policy.authorize_dates(["2026-01-13", "2026-01-13"])


@pytest.mark.parametrize(
    "day",
    [
        "2026-05-01",
        "2026-06-10",
        "2026-06-11",
        "2026-06-12",
        "2026-07-01",
    ],
)
def test_exposed_validation_and_sealed_dates_get_zero_path_or_io(
    day: str,
    monkeypatch,
) -> None:
    audit = DataAccessAudit()
    policy = ExplorationDataPolicy(audit=audit)
    calls = {"path": 0, "read": 0}

    def factory(value: str) -> Path:
        calls["path"] += 1
        return Path("data") / value / "mbp1.parquet"

    def forbidden_read(*args, **kwargs):
        del args, kwargs
        calls["read"] += 1
        raise AssertionError("pandas read must not be reached")

    monkeypatch.setattr(pd, "read_parquet", forbidden_read)
    with pytest.raises(PermissionError, match="not allowlisted"):
        policy.read_parquet(day, factory)
    assert calls == {"path": 0, "read": 0}
    assert audit.path_constructions == 0
    assert audit.metadata_accesses == 0
    assert audit.file_opens == 0
    assert audit.rows_read == 0
    policy.assert_zero_forbidden_access()


def test_allowed_reads_are_audited_per_date(monkeypatch) -> None:
    day = "2026-01-13"
    policy = ExplorationDataPolicy()
    expected = pd.DataFrame({"x": [1, 2, 3]})
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: expected)
    result = policy.read_parquet(
        day,
        lambda value: Path("data") / value / "trades.parquet",
    )
    assert result.equals(expected)
    audit = policy.audit.as_dict()
    assert audit["path_constructions_by_date"] == {day: 1}
    assert audit["file_opens_by_date"] == {day: 1}
    assert audit["rows_read_by_date"] == {day: 3}
    policy.assert_only_allowlisted_rows()


def test_repair_scope_cannot_be_broadened_before_any_path_access() -> None:
    audit = DataAccessAudit()
    expanded = ExplorationDataPolicy(
        audit=audit,
        allowlist=EXPLORATION_DATE_ALLOWLIST | {"2026-06-12"},
    )
    with pytest.raises(PermissionError, match="exact fixed"):
        require_fixed_exploration_allowlist(expanded)
    assert audit.path_constructions == 0
    assert audit.metadata_accesses == 0
    assert audit.file_opens == 0
    assert audit.rows_read == 0
