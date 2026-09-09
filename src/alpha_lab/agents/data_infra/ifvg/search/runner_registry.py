"""Registered runner entries for the detached search job shim (R4).

The R2 job shim refuses to execute without an explicit
``--runner-entry module:function`` (DEV-R2-3). The R2→R4 obligation
(``R2/DEVIATIONS.md`` scoping notes) closes the remaining gap: the entry is
now REGISTRY-GATED — the UI only ever passes a registered KEY, and the
worker refuses any entry string that is not an exact registered value, so no
user-shaped string can reach ``importlib``.

R5 registered the REAL executors: the baseline-verification search/pipeline
entries (``search/executors.py``) fail closed at CONSTRUCTION without the
owner's persisted verification authorization, so registration unblocks the
launch surface, never the data. Exact owner-approved strategy searches use
``search_strategy_development_v1``; its factory verifies the persisted approval
before any source read. Other full-development workflows remain separate actions.

R5-FIX (gate finding 3): the PRODUCTION registry no longer names any
``tests.*`` module. Synthetic fixture wiring is DEVELOPMENT-side: the tests
package registers its own entries through
:func:`register_development_runner_entries` (guarded, idempotent-on-match),
and a process where no such registration ran — every production worker —
refuses the synthetic keys exactly like any unregistered entry.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "REGISTERED_RUNNER_ENTRIES",
    "RunnerEntryError",
    "register_development_runner_entries",
    "registered_runner_entries",
    "resolve_registered_runner_entry",
    "assert_runner_entry_registered",
    "runner_entry_key_for_charter",
    "pipeline_entry_key_for_charter",
]


class RunnerEntryError(PermissionError):
    """The runner entry is not registered; execution is refused."""


#: key → ``module:function`` returning the replay wiring mapping
#: (``identity_resolver`` + ``child_runner``, optional ``prewarm`` /
#: ``cost_points``). Keys are the ONLY form the UI passes. This map is the
#: complete PRODUCTION registry: it names src executors exclusively and
#: never resolves into the ``tests`` package (R5-FIX finding 3).
REGISTERED_RUNNER_ENTRIES: Mapping[str, str] = MappingProxyType(
    {
        "pipeline_real_research_v1": (
            "alpha_lab.agents.data_infra.ifvg.search.research_executor:"
            "pipeline_real_research_entry"
        ),
        "search_strategy_development_v1": (
            "alpha_lab.agents.data_infra.ifvg.search.strategy_executor:"
            "search_strategy_development_entry"
        ),
        "search_baseline_verification_v1": (
            "alpha_lab.agents.data_infra.ifvg.search.executors:"
            "search_baseline_verification_entry"
        ),
        "pipeline_baseline_verification_v1": (
            "alpha_lab.agents.data_infra.ifvg.search.executors:"
            "pipeline_baseline_verification_entry"
        ),
    }
)

_ENTRY_SHAPE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")

#: Development-only entries (synthetic fixture wiring), registered BY the
#: development checkout's test code — never named by production source. In a
#: production process this map stays empty, so the synthetic keys fail
#: closed exactly like unregistered entries.
_DEVELOPMENT_ENTRIES: dict[str, str] = {}


def register_development_runner_entries(entries: Mapping[str, str]) -> None:
    """Register synthetic-fixture wiring from the development checkout.

    Guarded so the extension point can never widen the production surface:
    every key must carry the ``synthetic`` marker, no key may shadow a
    production entry, values must be exact ``module:function`` strings, and
    re-registration is idempotent-on-match (a conflicting value for an
    already-registered key is refused).
    """

    validated: dict[str, str] = {}
    for key, entry in entries.items():
        if key in REGISTERED_RUNNER_ENTRIES:
            raise RunnerEntryError(
                f"development registration may never shadow the production "
                f"entry {key!r}"
            )
        if "synthetic" not in key:
            raise RunnerEntryError(
                f"development runner-entry key {key!r} must carry the "
                "'synthetic' marker; real executors are registered in the "
                "production registry only"
            )
        if not _ENTRY_SHAPE.match(entry or ""):
            raise RunnerEntryError(
                f"development runner entry for {key!r} must be module:function"
            )
        existing = _DEVELOPMENT_ENTRIES.get(key)
        if existing is not None and existing != entry:
            raise RunnerEntryError(
                f"development runner-entry key {key!r} is already registered "
                "with a different value; re-registration must match exactly"
            )
        validated[key] = entry
    # validate-then-commit: a refused mapping registers nothing at all
    _DEVELOPMENT_ENTRIES.update(validated)


def registered_runner_entries() -> dict[str, str]:
    """The merged (production + development) registry view, copied."""

    return {**REGISTERED_RUNNER_ENTRIES, **_DEVELOPMENT_ENTRIES}


def resolve_registered_runner_entry(key: str) -> str:
    """The registered ``module:function`` for ``key`` (fail-closed)."""

    merged = registered_runner_entries()
    try:
        return merged[key]
    except KeyError:
        raise RunnerEntryError(
            f"runner-entry key {key!r} is not registered; registered keys: "
            f"{sorted(merged)}"
        ) from None


def assert_runner_entry_registered(entry: str) -> str:
    """Refuse any entry string that is not an exact registered value.

    The worker calls this on every ``--runner-entry`` it receives, so a raw
    user-shaped ``module:function`` can never reach ``importlib`` — only
    entries the production registry (or an explicit development
    registration in this process) names.
    """

    merged = registered_runner_entries()
    if entry in merged.values():
        return entry
    raise RunnerEntryError(
        "runner entry is not registered; pass --runner-entry-key with one of "
        f"{sorted(merged)} (raw module:function strings are refused; only "
        "registry-named executors can ever run)"
    )


def runner_entry_key_for_charter(charter_envelope) -> str | None:
    """The registered key a frozen charter may launch with, or ``None``.

    Synthetic-marker charters name the synthetic fixture KEY — resolvable
    only where the development checkout has registered its wiring (a
    production process refuses it as unregistered). Real
    verification-fixture charters resolve to the R5 baseline-verification
    executor — whose factory still fails closed (before any source path)
    until the owner's persisted verification authorization exists. Real
    strategy-only development charters with an exact strategy approval resolve
    to the development search executor; its factory verifies that approval.
    Other full-development charters have no registered search executor.
    """

    authorization = charter_envelope.payload.owner_authorization
    kind = getattr(authorization, "kind", None)
    if kind == "synthetic_test_authorization_v1":
        return "synthetic_search_job_fixture_v1"
    date_policy = charter_envelope.payload.date_policy
    if date_policy.access_policy_id == "verification_fixed_allowlist_max5_v1":
        return "search_baseline_verification_v1"
    from .strategy_approval import DECISION_ID  # noqa: PLC0415

    if any(
        ref.decision_id == DECISION_ID
        for ref in getattr(authorization, "decision_refs", {}).values()
    ):
        return "search_strategy_development_v1"
    return None


def pipeline_entry_key_for_charter(charter_envelope) -> str | None:
    """The registered PIPELINE executor key for a frozen charter, or None.

    Mirrors :func:`runner_entry_key_for_charter` for the 16-stage pipeline
    job: synthetic charters name the development fixture key (unresolvable
    in production processes); real verification-fixture charters resolve to
    the baseline-verification executor (whose factory fails closed without
    the owner's persisted authorization); real full-development charters
    have no registered pipeline executor — the operator full run is a
    separate action.
    """

    authorization = charter_envelope.payload.owner_authorization
    kind = getattr(authorization, "kind", None)
    if kind == "synthetic_test_authorization_v1":
        return "pipeline_synthetic_fixture_v1"
    date_policy = charter_envelope.payload.date_policy
    if date_policy.access_policy_id == "verification_fixed_allowlist_max5_v1":
        return "pipeline_baseline_verification_v1"
    if any(
        ref.decision_id == "ifvg_real_research_approval_v1"
        for ref in getattr(authorization, "decision_refs", {}).values()
    ):
        return "pipeline_real_research_v1"
    return None
