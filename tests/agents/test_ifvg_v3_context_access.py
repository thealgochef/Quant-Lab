"""V3 reuses the fixed January policy before every possible I/O surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.data_access import (
    EXPLORATION_DATE_ALLOWLIST,
    DataAccessAudit,
    ExplorationDataPolicy,
    require_fixed_exploration_allowlist,
)


def test_warmup_and_evidence_partitions_are_frozen() -> None:
    dates = tuple(sorted(EXPLORATION_DATE_ALLOWLIST))
    warmup = tuple(day for day in dates if day <= "2026-01-12")
    evidence = tuple(day for day in dates if day >= "2026-01-13")
    assert len(dates) == 26
    assert len(warmup) == 10
    assert warmup[-1] == "2026-01-12"
    assert evidence[0] == "2026-01-13"
    assert evidence[-1] == "2026-01-30"


@pytest.mark.parametrize(
    "day",
    ("2026-01-03", "2026-05-01", "2026-06-11", "2026-06-12", "2027-01-01"),
)
def test_denied_dates_never_reach_path_metadata_existence_or_read(day: str) -> None:
    audit = DataAccessAudit()
    policy = ExplorationDataPolicy(audit=audit)
    calls = {"path": 0, "reader": 0}

    def path_factory(value: str) -> Path:
        calls["path"] += 1
        return Path("data") / value / "trades.parquet"

    def reader(_path: Path):
        calls["reader"] += 1
        raise AssertionError("reader must not run")

    with pytest.raises(PermissionError, match="not allowlisted"):
        policy.read_metadata(day, path_factory, reader)
    assert calls == {"path": 0, "reader": 0}
    assert audit.path_constructions == 0
    assert audit.metadata_accesses == 0
    assert audit.file_opens == 0
    assert audit.rows_read == 0


def test_v3_policy_cannot_be_narrowed_or_broadened_for_derivation() -> None:
    for allowlist in (
        EXPLORATION_DATE_ALLOWLIST - {"2026-01-30"},
        EXPLORATION_DATE_ALLOWLIST | {"2026-06-12"},
    ):
        policy = ExplorationDataPolicy(allowlist=frozenset(allowlist))
        with pytest.raises(PermissionError, match="exact fixed"):
            require_fixed_exploration_allowlist(policy)
        assert policy.audit.path_constructions == 0
