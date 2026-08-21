"""Registry-gated runner entries (the R2→R4 obligation; DEV-R2-3 closure)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    SyntheticAuthorizationMarker,
)
from alpha_lab.agents.data_infra.ifvg.search.runner_registry import (
    REGISTERED_RUNNER_ENTRIES,
    RunnerEntryError,
    assert_runner_entry_registered,
    resolve_registered_runner_entry,
    runner_entry_key_for_charter,
)


def test_registry_resolution_and_refusal() -> None:
    entry = resolve_registered_runner_entry("synthetic_search_job_fixture_v1")
    assert entry == (
        "tests.agents.ifvg_search.test_search_job_script:synthetic_runner_entry"
    )
    assert assert_runner_entry_registered(entry) == entry
    with pytest.raises(RunnerEntryError, match="not registered"):
        resolve_registered_runner_entry("nope_v1")
    with pytest.raises(RunnerEntryError, match="raw module:function"):
        assert_runner_entry_registered("os:system")
    with pytest.raises(RunnerEntryError):
        assert_runner_entry_registered("")


def test_registry_is_immutable_and_synthetic_only_before_r5() -> None:
    assert set(REGISTERED_RUNNER_ENTRIES) == {"synthetic_search_job_fixture_v1"}
    with pytest.raises(TypeError):
        REGISTERED_RUNNER_ENTRIES["evil"] = "os:system"  # type: ignore[index]


def test_charter_key_mapping_is_fail_closed() -> None:
    from types import SimpleNamespace

    synthetic = SimpleNamespace(
        payload=SimpleNamespace(owner_authorization=SyntheticAuthorizationMarker())
    )
    assert (
        runner_entry_key_for_charter(synthetic)
        == "synthetic_search_job_fixture_v1"
    )
    real = SimpleNamespace(
        payload=SimpleNamespace(
            owner_authorization=SimpleNamespace(requirement_set_id="x")
        )
    )
    assert runner_entry_key_for_charter(real) is None  # no real executor pre-R5
