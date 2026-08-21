"""Registered runner entries for the detached search job shim (R4).

The R2 job shim refuses to execute without an explicit
``--runner-entry module:function`` (DEV-R2-3). The R2→R4 obligation
(``R2/DEVIATIONS.md`` scoping notes) closes the remaining gap: the entry is
now REGISTRY-GATED — the UI only ever passes a registered KEY, and the
worker refuses any entry string that is not an exact registered value, so no
user-shaped string can reach ``importlib``.

The registry ships with the synthetic fixture wiring only. The real
full-scope executors land with the R5 pipeline and register here; until
then no real-data execution path exists and a real charter's launch stays
capability-blocked with that reason.
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
        "are refused — the R5 pipeline registers the real executors)"
    )


def runner_entry_key_for_charter(charter_envelope) -> str | None:
    """The registered key a frozen charter may launch with, or ``None``.

    Synthetic-marker charters use the synthetic fixture wiring. Real
    charters have NO registered executor before the R5 pipeline — the UI
    renders the capability-blocked state instead of a launch control.
    """

    authorization = charter_envelope.payload.owner_authorization
    kind = getattr(authorization, "kind", None)
    if kind == "synthetic_test_authorization_v1":
        return "synthetic_search_job_fixture_v1"
    return None
