"""Parent search orchestrator (IMPLEMENTATION_PLAN §6; TEST_MATRIX §3.5).

Parent phases: ``charter_frozen → children_enumerated → artifacts_prewarmed →
replays → underlying_edge_passed → prop_simulations → prop_feasible →
robustness_passed → frontier_complete → search_complete`` (plus ``failed`` /
``cancelled``); child states ``queued/running/completed/failed/reused/
cancelled_at_safe_boundary/blocked``.

Mechanics cloned from the verified repo patterns: an ``O_EXCL`` lock per
search, atomic JSON checkpoints after every child transition, a
``cancel.requested`` sentinel honored at safe boundaries only (child
boundaries — completed children stay immutable and reusable), idempotent
resume via store-identity reuse, and enumeration deduped on the resolved
replay identity within AND across studies. Generated-profile capability is
enforced BEFORE any replay (P0-D): an unratified/blocked child never reaches
the worker. The UI never runs replays in-process — the detached CLI shim is
``scripts/ifvg_search_job.py``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field

from ..preparation import _write_json_atomic
from ..profiles import resolve_profile_config
from .axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    SEARCH_AXIS_REGISTRY_V1,
    assert_axes_authorized,
    resolve_axis_overrides,
)
from .charter import (
    OBJECTIVE_DIRECTIONS,
    ResolvedStrategyGateThresholds,
    SearchCharterEnvelope,
)
from .failure import ChildNeutralityError, FailureReason, sanitize_failure_message
from .frontier import FrontierResult, ObjectiveSpec, build_frontier
from .gates import StrategyGateReport, evaluate_strategy_gates
from .identities import (
    SHA256_PATTERN,
    CostedEvaluationIdentity,
    EnvelopeBase,
    FrozenContract,
    GeneratedProfileCapability,
    SearchChildMembership,
    canonical_contract_sha256,
    canonical_profile_id_for,
    canonicalize_section,
    evaluate_generated_profile_capability,
    register_identity_pair,
)
from .store import has_envelope, load_sidecar_bytes, save_or_reuse_envelope
from .strategy_metrics import StrategyMetrics, compute_strategy_metrics

__all__ = [
    "OBJECTIVE_DIRECTIONS",
    "CostedEvaluationEnvelope",
    "SearchFrontierEnvelope",
    "SearchFrontierPayload",
    "SEARCH_PHASES",
    "CHILD_STATES",
    "ChildSpec",
    "ChildOutcome",
    "SearchRunResult",
    "SearchChildMembershipEnvelope",
    "enumerate_children",
    "run_search",
    "request_safe_cancel",
    "read_search_state",
    "SearchLockError",
]

SEARCH_PHASES: tuple[str, ...] = (
    "charter_frozen",
    "children_enumerated",
    "artifacts_prewarmed",
    "replays",
    "underlying_edge_passed",
    "prop_simulations",
    "prop_feasible",
    "robustness_passed",
    "frontier_complete",
    "search_complete",
    "failed",
    "cancelled",
)

CHILD_STATES: tuple[str, ...] = (
    "queued",
    "running",
    "completed",
    "failed",
    "reused",
    "cancelled_at_safe_boundary",
    "blocked",
)

_STATE_FILENAME = "search_state.json"
_CANCEL_SENTINEL = "cancel.requested"
_LOCK_SUFFIX = ".lock"


class SearchLockError(RuntimeError):
    """Another process holds this search's exclusive lock."""


class SearchChildMembershipEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "membership_id"

    membership_id: str = Field(pattern=SHA256_PATTERN)
    payload: SearchChildMembership


class CostedEvaluationEnvelope(EnvelopeBase):
    """Deterministic lookup key for one replay's costed evaluation (§1.5).

    The envelope id hashes (core_replay_id, cost_policy_sha256) ONLY, so any
    later run of any study can locate the published evaluation from its
    inputs; the study-independent ``StrategyMetrics`` ride as a manifest
    sidecar (post-materialization facts). Gate reports are charter-scoped and
    are therefore re-evaluated per charter from the loaded metrics, never
    stored here.
    """

    _ID_FIELD: ClassVar[str] = "costed_evaluation_id"

    costed_evaluation_id: str = Field(pattern=SHA256_PATTERN)
    payload: CostedEvaluationIdentity


class SearchFrontierPayload(FrozenContract):
    """One search's frontier result under the strategy feasibility gates."""

    search_id: str = Field(pattern=SHA256_PATTERN)
    gate_policy_id: Literal["strategy_feasibility_gates_v1"] = (
        "strategy_feasibility_gates_v1"
    )
    frontier: FrontierResult


class SearchFrontierEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "frontier_id"

    frontier_id: str = Field(pattern=SHA256_PATTERN)
    payload: SearchFrontierPayload


def _child_evaluation_envelope(
    core_replay_id: str, cost_policy
) -> CostedEvaluationEnvelope:
    return CostedEvaluationEnvelope.from_payload(
        CostedEvaluationIdentity(
            core_replay_id=core_replay_id,
            cost_policy_sha256=canonical_contract_sha256(cost_policy),
        )
    )


_METRICS_SIDECAR = "strategy_metrics.json"


def _publish_child_evaluation(
    store_root: Path, envelope: CostedEvaluationEnvelope, metrics: StrategyMetrics
) -> None:
    save_or_reuse_envelope(
        store_root,
        "costed_evaluations",
        envelope,
        extra_files={
            _METRICS_SIDECAR: (
                json.dumps(metrics.model_dump(mode="json"), sort_keys=True) + "\n"
            ).encode("utf-8")
        },
    )


def _load_child_evaluation(
    store_root: Path, costed_evaluation_id: str
) -> StrategyMetrics | None:
    if not has_envelope(store_root, "costed_evaluations", costed_evaluation_id):
        return None
    data = load_sidecar_bytes(
        store_root, "costed_evaluations", costed_evaluation_id, _METRICS_SIDECAR
    )
    return StrategyMetrics.model_validate(json.loads(data))


@dataclass(frozen=True)
class ChildSpec:
    ordinal: int
    axis_value_ids: Mapping[str, str]
    comparison_role: str
    canonical_profile_id: str
    resolved_section_config_hash: str
    capability: GeneratedProfileCapability | None
    section_overrides: Mapping[str, Any]


@dataclass
class ChildOutcome:
    spec: ChildSpec
    core_replay_id: str
    state: str = "queued"
    failure_reason: FailureReason | None = None
    explanation: str = ""
    metrics: StrategyMetrics | None = None
    gate_report: StrategyGateReport | None = None
    replay_invocations: int = 0


@dataclass
class SearchRunResult:
    search_id: str
    phase: str
    children: list[ChildOutcome] = field(default_factory=list)
    frontier: FrontierResult | None = None
    state_path: Path | None = None


def _locked_invariant_violations(
    section, baseline_section, axis_specs
) -> tuple[str, ...]:
    """LOCKED_INVARIANT section fields that differ from the resolved baseline."""

    from .axis_registry import AxisClassification  # noqa: PLC0415

    baseline_dump = baseline_section.model_dump(mode="json")
    child_dump = section.model_dump(mode="json")
    violations = []
    for spec in axis_specs.values():
        if spec.classification is not AxisClassification.LOCKED_INVARIANT:
            continue
        key = spec.technical_key
        if key not in baseline_dump or baseline_dump.get(key) == child_dump.get(key):
            continue
        if key == "qualification_mode" and child_dump.get(key) == "custom_profile":
            # the resolver's derived-profile provenance relabel
            # (resolve_profile_config stamps every override-bearing child
            # "custom_profile"; the SC reducer stores the label and never
            # branches on it - verified). Every OTHER qualification
            # transition remains a violation.
            continue
        violations.append(key)
    return tuple(sorted(violations))


def enumerate_children(
    charter: SearchCharterEnvelope,
    *,
    identity_resolver: Callable[[ChildSpec], str],
    axis_registry=AXIS_VALUE_REGISTRY_V1,
    axis_specs=SEARCH_AXIS_REGISTRY_V1,
    registry_hash: str | None = None,
) -> tuple[ChildSpec, ...]:
    """Deterministic (core replay, membership) enumeration, deduped on the
    resolved replay identity (§16.9). The baseline combination is the child
    whose every axis value equals the axis's baseline value id.

    Fail-closed defense in depth: every combination re-runs the value-level
    authorization (`assert_axes_authorized`) even when `validate_charter`
    already ran — a hand-built charter with blocked/inert values can never
    enumerate. ``registry_hash`` defaults to the REAL registry digest.
    """

    from .axis_registry import registry_sha256  # noqa: PLC0415

    if registry_hash is None:
        registry_hash = registry_sha256(axes=axis_specs, values=axis_registry)
    payload = charter.payload
    synthetic = _is_synthetic(payload)
    axes = sorted(payload.axes.items())
    axis_keys = [axis for axis, _values in axes]
    value_lists = [values for _axis, values in axes]
    baseline_resolved = resolve_profile_config(
        {"profile_name": payload.baseline_profile_name}
    )
    specs: list[ChildSpec] = []
    seen_identities: set[str] = set()
    ordinal = 0
    for combo in product(*value_lists) if value_lists else [()]:
        axis_value_ids = dict(zip(axis_keys, combo, strict=True))
        for axis_key, value_id in axis_value_ids.items():
            assert_axes_authorized(
                {axis_key: value_id},
                axis_registry,
                axes=axis_specs,
                require_ratified=not synthetic,
            )
        overrides = resolve_axis_overrides(axis_value_ids, axis_registry)
        resolved = resolve_profile_config(
            {
                "profile_name": payload.baseline_profile_name,
                "section_overrides": overrides,
            }
            if overrides
            else {"profile_name": payload.baseline_profile_name}
        )
        section = canonicalize_section(resolved.section)
        from strategy_core.strategies.ifvg_smc.section import ifvg_profile_hash

        section_hash = ifvg_profile_hash(section)
        is_baseline = all(
            axis_specs[axis].baseline_value_id == value_id
            for axis, value_id in axis_value_ids.items()
        )
        capability = None
        if not is_baseline:
            capability = evaluate_generated_profile_capability(
                baseline_profile_id=payload.baseline_profile_name,
                axis_value_ids=axis_value_ids,
                registry_hash=registry_hash,
                authorization_ref=canonical_contract_sha256(
                    payload.owner_authorization
                ),
                authorization_state="authorized"
                if not _has_pending_values(axis_value_ids, axis_registry) or synthetic
                else "missing",
                section=section,
                invariant_violations=_locked_invariant_violations(
                    section, baseline_resolved.section, axis_specs
                ),
            )
        spec = ChildSpec(
            ordinal=ordinal,
            axis_value_ids=dict(axis_value_ids),
            comparison_role="baseline" if is_baseline else "challenger",
            canonical_profile_id=(
                payload.baseline_profile_name if is_baseline else canonical_profile_id_for(section)
            ),
            resolved_section_config_hash=section_hash,
            capability=capability,
            section_overrides=dict(overrides),
        )
        core_id = identity_resolver(spec)
        if core_id in seen_identities:
            continue  # deduped: identical resolved replay identity
        seen_identities.add(core_id)
        specs.append(spec)
        ordinal += 1
    return tuple(specs)


def _has_pending_values(axis_value_ids: Mapping[str, str], registry) -> bool:
    return any(
        registry[value_id].owner_ratification_status == "pending"
        for value_id in axis_value_ids.values()
        if value_id in registry
    )


def _is_synthetic(payload) -> bool:
    from .authorization import SyntheticAuthorizationMarker  # noqa: PLC0415

    return isinstance(payload.owner_authorization, SyntheticAuthorizationMarker)


def _try_break_stale_search_lock(lock: Path, stale_seconds: float) -> bool:
    """Break an orphaned lock (killed run) once its heartbeat is provably stale.

    The running orchestrator refreshes the lock mtime at every checkpoint, so
    a live run's lock never ages past a child transition. The break is an
    atomic rename-to-quarantine: exactly one waiter can win it (no
    unlink/create TOCTOU between concurrent breakers). Same Windows-truthful
    contention semantics as the catalog lock (DECISIONS_TAKEN R1 #12/#13).
    """

    import time  # noqa: PLC0415
    import uuid  # noqa: PLC0415

    try:
        age = time.time() - lock.stat().st_mtime
    except OSError:
        return False  # vanished or delete-pending — treat as contention
    if age < stale_seconds:
        return False
    quarantine = lock.with_name(f"{lock.name}.stale-{uuid.uuid4().hex}")
    try:
        os.replace(lock, quarantine)
    except OSError:
        return False  # another waiter won the break, or the holder released
    with suppress(OSError):
        os.unlink(quarantine)
    return True


@contextmanager
def _search_lock(state_root: Path, search_id: str, *, stale_lock_seconds: float):
    """Exclusive per-search lock with nonce-verified release.

    The lock body is a per-acquire nonce: release deletes the lock ONLY when
    the body still holds our nonce, so a holder whose stale lock was broken
    can never delete the new holder's lock.
    """

    import uuid  # noqa: PLC0415

    state_root.mkdir(parents=True, exist_ok=True)
    lock = state_root / f"{search_id}{_LOCK_SUFFIX}"
    nonce = uuid.uuid4().hex

    def _acquire() -> None:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(handle, nonce.encode("ascii"))
        finally:
            os.close(handle)

    try:
        _acquire()
    except (FileExistsError, PermissionError, OSError):
        # A live holder — unless the lock's heartbeat is provably stale
        # (killed run), in which case resume breaks it exactly once.
        if not _try_break_stale_search_lock(lock, stale_lock_seconds):
            raise SearchLockError(
                f"search {search_id[:12]}… is already locked by another process"
            ) from None
        try:
            _acquire()
        except (FileExistsError, PermissionError, OSError):
            raise SearchLockError(
                f"search {search_id[:12]}… is already locked by another process"
            ) from None
    try:
        yield lock
    finally:
        with suppress(OSError):
            if lock.read_text(encoding="ascii", errors="replace") == nonce:
                os.unlink(lock)


def _state_dir(state_root: Path, search_id: str) -> Path:
    return Path(state_root) / search_id


def read_search_state(state_root: Path, search_id: str) -> dict | None:
    path = _state_dir(state_root, search_id) / _STATE_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def request_safe_cancel(state_root: Path, search_id: str) -> Path:
    directory = _state_dir(state_root, search_id)
    directory.mkdir(parents=True, exist_ok=True)
    sentinel = directory / _CANCEL_SENTINEL
    sentinel.write_text("cancel requested\n", encoding="utf-8")
    return sentinel


def _checkpoint(
    state_root: Path,
    search_id: str,
    phase: str,
    outcomes: list[ChildOutcome],
    *,
    heartbeat: Path | None = None,
    phase_notes: Mapping[str, str] | None = None,
) -> Path:
    if heartbeat is not None:
        with suppress(OSError):  # pragma: no cover - lock broken externally
            os.utime(heartbeat)  # live-run heartbeat: the lock never goes stale
    directory = _state_dir(state_root, search_id)
    payload = {
        "schema_version": 1,
        "search_id": search_id,
        "phase": phase,
        "phase_notes": dict(phase_notes or {}),
        "children": [
            {
                "ordinal": outcome.spec.ordinal,
                "core_replay_id": outcome.core_replay_id,
                "axis_value_ids": dict(outcome.spec.axis_value_ids),
                "comparison_role": outcome.spec.comparison_role,
                "state": outcome.state,
                "failure_reason": (
                    outcome.failure_reason.value if outcome.failure_reason else None
                ),
                "explanation": outcome.explanation,
                "replay_invocations": outcome.replay_invocations,
            }
            for outcome in outcomes
        ],
    }
    path = directory / _STATE_FILENAME
    _write_json_atomic(path, payload)
    return path


def run_search(
    charter: SearchCharterEnvelope,
    *,
    store_root: Path,
    state_root: Path,
    identity_resolver: Callable[[ChildSpec], object],
    child_runner: Callable[..., Any],
    cost_points: float | None = None,
    prewarm: Callable[[tuple[ChildSpec, ...]], None] | None = None,
    progress_fn: Callable[[int, int, str], None] | None = None,
    stale_lock_seconds: float = 86_400.0,
) -> SearchRunResult:
    """Run (or resume) one frozen search charter, sequentially and safely.

    ``identity_resolver(spec)`` returns the child's `CoreStrategyReplayIdentity`
    envelope (or a bare 64-hex id in synthetic fixtures). ``child_runner`` is
    the replay seam: called ONLY for children that are runnable and not
    reusable; it returns an object exposing ``tables`` and
    ``gross_trade_stream_hash`` and, when it produced a
    ``ChildAuditNeutralityReport``, exposes it as ``neutrality`` - a FAILED
    report (or a raised :class:`ChildNeutralityError`, which the real
    ``run_child_replay`` raises itself) blocks the child's publication
    (CS 3.3). Cancellation is honored at child boundaries only; completed
    children remain immutable and reusable on resume.
    """

    payload = charter.payload
    search_id = charter.search_id
    store_root = Path(store_root)
    state_root = Path(state_root)
    cost = payload.cost_policy.cost_points_round_turn if cost_points is None else cost_points

    def _resolve_identity(spec: ChildSpec) -> tuple[str, object | None]:
        resolved = identity_resolver(spec)
        if isinstance(resolved, str):
            return resolved, None
        return resolved.core_replay_id, resolved

    with _search_lock(
        state_root, search_id, stale_lock_seconds=stale_lock_seconds
    ) as lock_path:
        # a leftover sentinel from an earlier (already-cancelled) run applies
        # to THAT run only; a fresh run starts unpoisoned and can be
        # re-cancelled at any child boundary.
        with suppress(OSError):
            os.unlink(_state_dir(state_root, search_id) / _CANCEL_SENTINEL)
        phase = "charter_frozen"
        specs = enumerate_children(
            charter, identity_resolver=lambda spec: _resolve_identity(spec)[0]
        )
        if len(specs) > payload.max_child_count:
            raise ValueError("enumerated children exceed the charter ceiling")
        outcomes = [
            ChildOutcome(spec=spec, core_replay_id=_resolve_identity(spec)[0])
            for spec in specs
        ]
        phase = "children_enumerated"
        _checkpoint(state_root, search_id, phase, outcomes, heartbeat=lock_path)

        if prewarm is not None:
            prewarm(specs)
        phase = "artifacts_prewarmed"
        _checkpoint(state_root, search_id, phase, outcomes, heartbeat=lock_path)

        sentinel = _state_dir(state_root, search_id) / _CANCEL_SENTINEL
        phase = "replays"
        _checkpoint(state_root, search_id, phase, outcomes, heartbeat=lock_path)
        tables_by_child: dict[str, Any] = {}
        for index, outcome in enumerate(outcomes):
            if sentinel.exists():
                with suppress(OSError):  # honored exactly once, then consumed
                    os.unlink(sentinel)
                outcome.state = "cancelled_at_safe_boundary"
                outcome.failure_reason = FailureReason.CANCELLED
                outcome.explanation = "safe cancel honored at the child boundary"
                for later in outcomes[index + 1 :]:
                    later.state = "cancelled_at_safe_boundary"
                    later.failure_reason = FailureReason.CANCELLED
                    later.explanation = "safe cancel honored at the child boundary"
                phase = "cancelled"
                _checkpoint(state_root, search_id, phase, outcomes, heartbeat=lock_path)
                return SearchRunResult(
                    search_id=search_id,
                    phase=phase,
                    children=outcomes,
                    state_path=_state_dir(state_root, search_id) / _STATE_FILENAME,
                )
            spec = outcome.spec
            # P0-D: blocked/unratified generated children never reach a replay
            if spec.capability is not None and spec.capability.status != "generated_runnable":
                outcome.state = "blocked"
                outcome.failure_reason = FailureReason.BLOCKED_AXIS
                outcome.explanation = (
                    f"generated profile blocked before replay: "
                    f"{spec.capability.status} — {spec.capability.reason}"
                )
                _checkpoint(state_root, search_id, phase, outcomes, heartbeat=lock_path)
                continue
            if has_envelope(store_root, "core_replays", outcome.core_replay_id):
                outcome.state = "reused"
                outcome.explanation = (
                    "verified reuse: an immutable replay with this exact identity "
                    "already exists (zero replay invocations)"
                )
            else:
                outcome.state = "running"
                _checkpoint(state_root, search_id, phase, outcomes, heartbeat=lock_path)
                try:
                    result = child_runner(
                        spec=spec, core_replay_id=outcome.core_replay_id
                    )
                    outcome.replay_invocations = 1
                    # §3.3: a FAILED neutrality report blocks CHILD
                    # PUBLICATION - nothing below this line runs for it.
                    neutrality = getattr(result, "neutrality", None)
                    if neutrality is not None and not neutrality.passed:
                        raise ChildNeutralityError(
                            "child audit-neutrality FAILED - the child cannot "
                            "publish (core tables differ with the audit channel "
                            "enabled, or audit stamps broke referential integrity)"
                        )
                    tables_by_child[outcome.core_replay_id] = result
                    _, envelope = _resolve_identity(spec)
                    if envelope is not None:
                        save_or_reuse_envelope(store_root, "core_replays", envelope)
                    outcome.state = "completed"
                except ChildNeutralityError as error:
                    outcome.state = "failed"
                    outcome.failure_reason = FailureReason.INVARIANT
                    outcome.explanation = sanitize_failure_message(str(error))
                    _checkpoint(state_root, search_id, phase, outcomes, heartbeat=lock_path)
                    continue
                except Exception as error:
                    outcome.state = "failed"
                    outcome.failure_reason = FailureReason.REPLAY
                    outcome.explanation = sanitize_failure_message(str(error))
                    _checkpoint(state_root, search_id, phase, outcomes, heartbeat=lock_path)
                    continue
            membership = SearchChildMembership(
                parent_search_id=search_id,
                child_ordinal=spec.ordinal,
                axis_value_ids=dict(spec.axis_value_ids),
                core_replay_id=outcome.core_replay_id,
                comparison_role=spec.comparison_role,  # type: ignore[arg-type]
            )
            save_or_reuse_envelope(
                store_root,
                "memberships",
                SearchChildMembershipEnvelope.from_payload(membership),
            )
            _checkpoint(state_root, search_id, phase, outcomes, heartbeat=lock_path)
            if progress_fn is not None:
                progress_fn(index + 1, len(outcomes), outcome.core_replay_id[:12])

        # Strategy gates over every completed OR reused child. Completed
        # children publish their study-independent costed evaluation (metrics
        # keyed on core_replay_id x cost policy); reused children RELOAD the
        # published evaluation, so a fully-reused resume or cross-study run
        # produces the identical gates and frontier. Gate reports are
        # charter-scoped and always re-evaluated from the metrics.
        thresholds: ResolvedStrategyGateThresholds = payload.objective_policy.feasibility_gates
        feasible_metrics: dict[str, dict[str, float]] = {}
        for outcome in outcomes:
            if outcome.state not in ("completed", "reused"):
                continue
            evaluation = _child_evaluation_envelope(
                outcome.core_replay_id, payload.cost_policy
            )
            result = tables_by_child.get(outcome.core_replay_id)
            if result is not None:
                metrics = compute_strategy_metrics(
                    result.tables,
                    cost_points=cost,
                    evaluation_config_hash=canonical_contract_sha256(
                        {
                            "core_replay_id": outcome.core_replay_id,
                            "cost_policy": payload.cost_policy.model_dump(
                                mode="json"
                            ),
                        }
                    ),
                )
                _publish_child_evaluation(store_root, evaluation, metrics)
            else:
                metrics = _load_child_evaluation(
                    store_root, evaluation.costed_evaluation_id
                )
                if metrics is None:
                    outcome.explanation = (
                        "reused replay has no published costed evaluation for "
                        "this cost policy; gates were not evaluated this run"
                    )
                    continue
            outcome.metrics = metrics
            report = evaluate_strategy_gates(metrics, thresholds)
            outcome.gate_report = report
            if report.passed:
                objective_values: dict[str, float] = {}
                missing_objectives: list[str] = []
                for objective in payload.objective_policy.pareto_objectives:
                    value = getattr(metrics, objective, None)
                    if value is None:
                        missing_objectives.append(objective)
                    else:
                        objective_values[objective] = value
                if missing_objectives:
                    outcome.explanation = (
                        "passed all strategy gates but objective metric(s) "
                        f"{missing_objectives} are unavailable; excluded from "
                        "the frontier"
                    )
                else:
                    feasible_metrics[outcome.core_replay_id] = objective_values
            else:
                outcome.failure_reason = report.failure_reason
                outcome.explanation = report.human_explanation
        phase = "underlying_edge_passed"
        _checkpoint(state_root, search_id, phase, outcomes, heartbeat=lock_path)

        # The prop/robustness phases are R3-owned; the state file records the
        # skip reason explicitly so the phase march never reads as a pass.
        skip_notes = {
            "prop_simulations": "skipped: the prop lifecycle lands in R3",
            "prop_feasible": "skipped: the prop lifecycle lands in R3",
            "robustness_passed": (
                "skipped: robustness rides the comparison surfaces in R2 and "
                "joins the parent run with prop metrics in R3"
            ),
        }
        phase = "prop_simulations"
        _checkpoint(
            state_root, search_id, phase, outcomes,
            heartbeat=lock_path, phase_notes=skip_notes,
        )
        phase = "prop_feasible"
        _checkpoint(
            state_root, search_id, phase, outcomes,
            heartbeat=lock_path, phase_notes=skip_notes,
        )
        phase = "robustness_passed"
        _checkpoint(
            state_root, search_id, phase, outcomes,
            heartbeat=lock_path, phase_notes=skip_notes,
        )

        frontier = None
        if feasible_metrics:
            objectives = tuple(
                ObjectiveSpec(metric=metric, direction=OBJECTIVE_DIRECTIONS[metric])
                for metric in payload.objective_policy.pareto_objectives
            )
            frontier = build_frontier(
                feasible_metrics,
                objectives=objectives,
                lexicographic_tie_breaks=payload.objective_policy.lexicographic_tie_breaks,
            )
            save_or_reuse_envelope(
                store_root,
                "frontiers",
                SearchFrontierEnvelope.from_payload(
                    SearchFrontierPayload(search_id=search_id, frontier=frontier)
                ),
            )
        phase = "frontier_complete"
        _checkpoint(
            state_root, search_id, phase, outcomes,
            heartbeat=lock_path, phase_notes=skip_notes,
        )
        phase = "search_complete"
        state_path = _checkpoint(
            state_root, search_id, phase, outcomes,
            heartbeat=lock_path, phase_notes=skip_notes,
        )
        return SearchRunResult(
            search_id=search_id,
            phase=phase,
            children=outcomes,
            frontier=frontier,
            state_path=state_path,
        )


register_identity_pair(
    name="CostedEvaluation",
    envelope_cls=CostedEvaluationEnvelope,
    payload_cls=CostedEvaluationIdentity,
    id_field="costed_evaluation_id",
    example_factory=lambda: CostedEvaluationIdentity(
        core_replay_id="a" * 64,
        cost_policy_sha256="b" * 64,
    ),
)

register_identity_pair(
    name="SearchFrontier",
    envelope_cls=SearchFrontierEnvelope,
    payload_cls=SearchFrontierPayload,
    id_field="frontier_id",
    example_factory=lambda: SearchFrontierPayload(
        search_id="a" * 64,
        frontier=FrontierResult(
            feasible_ids=("b" * 64,),
            frontier_ids=("b" * 64,),
            dominance_edges=(),
            development_exploratory_representative_id="b" * 64,
            per_objective_champions={"net_expectancy_r": "b" * 64},
            tie_break_trace=("frontier candidates: 1",),
        ),
    ),
)

register_identity_pair(
    name="SearchChildMembership",
    envelope_cls=SearchChildMembershipEnvelope,
    payload_cls=SearchChildMembership,
    id_field="membership_id",
    example_factory=lambda: SearchChildMembership(
        parent_search_id="a" * 64,
        child_ordinal=0,
        axis_value_ids={"parent_retest_timeout_1m_bars": "parent_retest_timeout_1m_bars.none"},
        core_replay_id="b" * 64,
        comparison_role="baseline",
    ),
)
