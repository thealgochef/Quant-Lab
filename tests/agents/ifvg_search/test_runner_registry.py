"""Registry-gated runner entries (the R2→R4 obligation; DEV-R2-3 closure).

R5-FIX (gate finding 3): the PRODUCTION registry must never name a
``tests.*`` module — synthetic fixture wiring reaches resolution only
through the guarded development registration this suite exercises (the
lane conftest registers the two synthetic entries for the test process).
"""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    SyntheticAuthorizationMarker,
)
from alpha_lab.agents.data_infra.ifvg.search.runner_registry import (
    REGISTERED_RUNNER_ENTRIES,
    RunnerEntryError,
    assert_runner_entry_registered,
    register_development_runner_entries,
    registered_runner_entries,
    resolve_registered_runner_entry,
    runner_entry_key_for_charter,
)


def test_production_registry_never_resolves_into_the_tests_package() -> None:
    """R5-FIX finding 3: production entries are src executors exclusively."""

    assert set(REGISTERED_RUNNER_ENTRIES) == {
        "search_baseline_verification_v1",
        "pipeline_baseline_verification_v1",
    }
    for key, entry in REGISTERED_RUNNER_ENTRIES.items():
        assert not entry.startswith("tests."), (key, entry)
        assert entry.startswith("alpha_lab."), (key, entry)
    # and the production module's source itself never names tests.*
    import inspect

    from alpha_lab.agents.data_infra.ifvg.search import runner_registry

    source = inspect.getsource(runner_registry)
    assert "tests.agents" not in source


def test_registry_resolution_and_refusal() -> None:
    # the lane conftest registered the development (synthetic) entries
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


def test_development_registration_is_guarded() -> None:
    # idempotent-on-match: re-registering the exact conftest values is a no-op
    register_development_runner_entries(
        {
            "synthetic_search_job_fixture_v1": (
                "tests.agents.ifvg_search.test_search_job_script:"
                "synthetic_runner_entry"
            )
        }
    )
    # a conflicting value for an already-registered key is refused
    with pytest.raises(RunnerEntryError, match="different value"):
        register_development_runner_entries(
            {"synthetic_search_job_fixture_v1": "tests.other:entry"}
        )
    # keys without the synthetic marker are refused
    with pytest.raises(RunnerEntryError, match="'synthetic' marker"):
        register_development_runner_entries({"real_thing_v1": "tests.x:y"})
    # production keys can never be shadowed
    with pytest.raises(RunnerEntryError, match="never shadow"):
        register_development_runner_entries(
            {"search_baseline_verification_v1": "tests.x:y"}
        )
    # malformed module:function strings are refused
    with pytest.raises(RunnerEntryError, match="module:function"):
        register_development_runner_entries({"synthetic_bad_v1": "not a path"})
    assert "synthetic_bad_v1" not in registered_runner_entries()
    # a refused mapping registers NOTHING (validate-then-commit)
    with pytest.raises(RunnerEntryError):
        register_development_runner_entries(
            {
                "synthetic_would_be_fine_v1": "tests.x:y",
                "synthetic_bad_shape_v1": "not a path",
            }
        )
    assert "synthetic_would_be_fine_v1" not in registered_runner_entries()


def test_registry_is_immutable_with_the_r5_executor_set() -> None:
    # the merged view (production + development registration) carries all four
    assert set(registered_runner_entries()) == {
        "synthetic_search_job_fixture_v1",
        "search_baseline_verification_v1",
        "pipeline_synthetic_fixture_v1",
        "pipeline_baseline_verification_v1",
    }
    # the real entries resolve into src, never into user-shaped strings
    assert REGISTERED_RUNNER_ENTRIES["search_baseline_verification_v1"].startswith(
        "alpha_lab.agents.data_infra.ifvg.search.executors:"
    )
    with pytest.raises(TypeError):
        REGISTERED_RUNNER_ENTRIES["evil"] = "os:system"  # type: ignore[index]
    # the merged view is a copy — mutating it never touches the registry
    merged = registered_runner_entries()
    merged["evil"] = "os:system"
    assert "evil" not in registered_runner_entries()


def test_charter_key_mapping_is_fail_closed() -> None:
    from types import SimpleNamespace

    synthetic = SimpleNamespace(
        payload=SimpleNamespace(owner_authorization=SyntheticAuthorizationMarker())
    )
    assert (
        runner_entry_key_for_charter(synthetic)
        == "synthetic_search_job_fixture_v1"
    )
    real_verification = SimpleNamespace(
        payload=SimpleNamespace(
            owner_authorization=SimpleNamespace(requirement_set_id="x"),
            date_policy=SimpleNamespace(
                access_policy_id="verification_fixed_allowlist_max5_v1"
            ),
        )
    )
    # R5: the real verification executor is registered; its FACTORY still
    # fails closed (fail-before-path) without the owner's authorization
    assert (
        runner_entry_key_for_charter(real_verification)
        == "search_baseline_verification_v1"
    )
    real_development = SimpleNamespace(
        payload=SimpleNamespace(
            owner_authorization=SimpleNamespace(requirement_set_id="x"),
            date_policy=SimpleNamespace(
                access_policy_id="development_explicit_dates_before_path_v2"
            ),
        )
    )
    # the operator full run has NO registered executor — a separate action
    assert runner_entry_key_for_charter(real_development) is None


def test_real_executor_factories_fail_before_any_source_path(tmp_path) -> None:
    """The R5 real entries refuse at CONSTRUCTION without the owner's
    persisted verification authorization — no config, policy, or source
    path is built."""

    from alpha_lab.agents.data_infra.ifvg.search.executors import (
        pipeline_baseline_verification_entry,
        search_baseline_verification_entry,
    )
    from tests.agents.ifvg_search.test_orchestrator import _charter

    synthetic_charter = _charter()
    with pytest.raises(PermissionError, match="never run synthetic-marker"):
        search_baseline_verification_entry(synthetic_charter, store_root=tmp_path)

    # the frozen payload has no mutation path — a shim carries the real shape
    from types import SimpleNamespace

    shim = SimpleNamespace(
        payload=SimpleNamespace(
            owner_authorization=SimpleNamespace(requirement_set_id="x"),
            date_policy=synthetic_charter.payload.date_policy,
            baseline_profile_name=synthetic_charter.payload.baseline_profile_name,
        )
    )
    with pytest.raises(PermissionError, match="fail-before-path"):
        search_baseline_verification_entry(shim, store_root=tmp_path)
    with pytest.raises(PermissionError, match="fail-before-path"):
        pipeline_baseline_verification_entry(shim, store_root=tmp_path)
