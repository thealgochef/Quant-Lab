"""Registered runner entries for the detached search job shim (R4).

The R2 job shim refuses to execute without an explicit
``--runner-entry module:function`` (DEV-R2-3). The R2→R4 obligation
(``R2/DEVIATIONS.md`` scoping notes) closes the remaining gap: the entry is
now REGISTRY-GATED — the UI only ever passes a registered KEY, and the
worker refuses any entry string that is not an exact registered value, so no
user-shaped string can reach ``importlib``.

R5 registered the REAL executors beside the synthetic fixture wiring:
the baseline-verification search/pipeline entries (``search/executors.py``)
fail closed at CONSTRUCTION without the owner's persisted verification
authorization, so registration unblocks the launch surface, never the
data. Full-development execution has no registered entry — the operator
run stays a separate, explicitly authorized action.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "REGISTERED_RUNNER_ENTRIES",
    "RunnerEntryError",
    "resolve_registered_runner_entry",
    "assert_runner_entry_registered",
    "runner_entry_key_for_charter",
    "pipeline_entry_key_for_charter",
]


class RunnerEntryError(PermissionError):
    """The runner entry is not registered; execution is refused."""


#: key → ``module:function`` returning the replay wiring mapping
#: (``identity_resolver`` + ``child_runner``, optional ``prewarm`` /
#: ``cost_points``). Keys are the ONLY form the UI passes. The synthetic
#: entry resolves inside a development checkout (the ``tests`` package);
#: anywhere it cannot import, execution fails closed exactly like an
#: unregistered entry. R5 registers the real pipeline executors.
REGISTERED_RUNNER_ENTRIES: Mapping[str, str] = MappingProxyType(
    {
        "synthetic_search_job_fixture_v1": (
            "tests.agents.ifvg_search.test_search_job_script:synthetic_runner_entry"
        ),
        "search_baseline_verification_v1": (
            "alpha_lab.agents.data_infra.ifvg.search.executors:"
            "search_baseline_verification_entry"
        ),
        "pipeline_synthetic_fixture_v1": (
            "tests.agents.ifvg_search.test_pipeline_job_script:"
            "synthetic_pipeline_entry"
        ),
        "pipeline_baseline_verification_v1": (
            "alpha_lab.agents.data_infra.ifvg.search.executors:"
            "pipeline_baseline_verification_entry"
        ),
    }
)


def resolve_registered_runner_entry(key: str) -> str:
    """The registered ``module:function`` for ``key`` (fail-closed)."""

    try:
        return REGISTERED_RUNNER_ENTRIES[key]
    except KeyError:
        raise RunnerEntryError(
            f"runner-entry key {key!r} is not registered; registered keys: "
            f"{sorted(REGISTERED_RUNNER_ENTRIES)}"
        ) from None


def assert_runner_entry_registered(entry: str) -> str:
    """Refuse any entry string that is not an exact registered value.

    The worker calls this on every ``--runner-entry`` it receives, so a raw
    user-shaped ``module:function`` can never reach ``importlib`` — only
    entries this registry (or a later release's registration) names.
    """

    if entry in REGISTERED_RUNNER_ENTRIES.values():
        return entry
    raise RunnerEntryError(
        "runner entry is not registered; pass --runner-entry-key with one of "
        f"{sorted(REGISTERED_RUNNER_ENTRIES)} (raw module:function strings "
        "are refused; only registry-named executors can ever run)"
    )


def runner_entry_key_for_charter(charter_envelope) -> str | None:
    """The registered key a frozen charter may launch with, or ``None``.

    Synthetic-marker charters use the synthetic fixture wiring. Real
    verification-fixture charters resolve to the R5 baseline-verification
    executor — whose factory still fails closed (before any source path)
    until the owner's persisted verification authorization exists. Real
    full-development charters have no registered executor: the operator
    full run is a separate, explicitly authorized action.
    """

    authorization = charter_envelope.payload.owner_authorization
    kind = getattr(authorization, "kind", None)
    if kind == "synthetic_test_authorization_v1":
        return "synthetic_search_job_fixture_v1"
    date_policy = charter_envelope.payload.date_policy
    if date_policy.access_policy_id == "verification_fixed_allowlist_max5_v1":
        return "search_baseline_verification_v1"
    return None


def pipeline_entry_key_for_charter(charter_envelope) -> str | None:
    """The registered PIPELINE executor key for a frozen charter, or None.

    Mirrors :func:`runner_entry_key_for_charter` for the 16-stage pipeline
    job: synthetic charters run the synthetic fixture wiring; real
    verification-fixture charters resolve to the baseline-verification
    executor (whose factory fails closed without the owner's persisted
    authorization); real full-development charters have no registered
    pipeline executor — the operator full run is a separate action.
    """

    authorization = charter_envelope.payload.owner_authorization
    kind = getattr(authorization, "kind", None)
    if kind == "synthetic_test_authorization_v1":
        return "pipeline_synthetic_fixture_v1"
    date_policy = charter_envelope.payload.date_policy
    if date_policy.access_policy_id == "verification_fixed_allowlist_max5_v1":
        return "pipeline_baseline_verification_v1"
    return None
