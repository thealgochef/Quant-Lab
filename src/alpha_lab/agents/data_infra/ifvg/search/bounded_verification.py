"""Phase 4 — the real ≤5-day bounded verification (plan §6), code authoring.

Nothing here runs real data by itself: the runner (``scripts/ifvg_bounded_verification.py``)
is refused before any source path unless the owner's persisted
``VerificationRunEnvelope`` + ``VerificationAuthorizationRef`` exist
(``search.executors``). This module provides the three pieces §6 requires:

* :func:`preflight_bounded_verification` (§6.1) — before path construction:
  the output namespace is locked to ``search_test/v1`` and is a coherently
  deployed ``test`` store namespace; the supersession-head witness is
  CURRENT; the authorization is a real ref (never the synthetic marker) that
  validates against the run, the store namespace and the seed the runner
  will actually load; the allowlist is 1–5 consecutive LOGICAL trading days
  (never physical partition dates), each mapped to its physical partitions
  when the mapping is supplied, inside the permitted window (June 11 and the
  sealed range excluded), equal to the canonical program allowlist when one
  is registered; the seed is profile-bound and continuous with the window;
  a research-catalog destination is refused. Every refusal is typed
  (:class:`BoundedVerificationRefusalError`).
* :class:`R1BaselineGatePayload` (§6.2) — the six
  ``verification_control_flow_gates_v1`` gates of both attempts (the first,
  fresh, and the second, verified reuse) plus the "also prove" proofs:
  native ids and core-table hashes repeat, the executed-trade table
  exact-loads, the second identical invocation returns REUSED with zero
  replay, identical bytes reuse, different bytes / a missing or corrupt
  manifest / a corrupt sidecar fail closed. Assembled from evidence the
  runner gathered; ``passed`` is the conjunction.
* :class:`BoundedReleaseControlFlowPayload` (§6.3) — the SEPARATE immutable
  release-specific report typing each later-release component's bounded
  outcome on the five-day fixture (MBP-1 diagnostic, 5m/15m panel, fold
  construction, KMeans/regime, CatBoost/logistic, S14, S15, S11) — never
  research evidence (``not_research_evidence=True``); built from a completed
  pipeline state and its persisted stage sidecars only.

Research profitability, strategy, payout, feature-selection or promotion
gates are never applied here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from ..data_access import allowlist_sha256
from .authorization import SyntheticAuthorizationMarker, VerificationAuthorizationRef
from .identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)
from .store import (
    SEARCH_TEST_STORE_ROOT,
    SearchStoreError,
    SidecarLoadError,
    load_json_sidecar,
    load_verified_envelope,
    save_or_reuse_envelope,
)
from .store_namespace import (
    StoreNamespaceError,
    SupersessionHeadWitness,
    assert_namespace_deployment_coherent,
    path_looks_like_research_store,
    require_store_namespace,
)
from .supersession_chain import assert_head_witness_current
from .verification import (
    VERIFICATION_CONTROL_FLOW_GATE_IDS,
    ControlFlowGateReport,
    VerificationRunEnvelope,
    VerificationRunValidationError,
    validate_verification_run,
)

__all__ = [
    "BOUNDED_VERIFICATION_REFUSAL_REASONS",
    "BoundedVerificationRefusalError",
    "BoundedVerificationPreflight",
    "preflight_bounded_verification",
    "R1_BASELINE_GATE_POLICY_ID",
    "R1_BASELINE_PROOF_IDS",
    "R1_BASELINE_GATE_STORE",
    "R1BaselineGatePayload",
    "R1BaselineGateEnvelope",
    "build_r1_baseline_gate_report",
    "save_r1_baseline_gate_report",
    "STORE_BEHAVIOR_PROOF_IDS",
    "store_behavior_proofs",
    "BOUNDED_CONTROL_FLOW_POLICY_ID",
    "BOUNDED_CONTROL_FLOW_STORE",
    "BOUNDED_COMPONENTS",
    "BOUNDED_COMPONENT_OUTCOMES",
    "BoundedComponentResult",
    "BoundedReleaseControlFlowPayload",
    "BoundedReleaseControlFlowEnvelope",
    "build_bounded_release_control_flow_report",
    "save_bounded_release_control_flow_report",
]

_PROTECTED_BUFFER_DAY = "2026-06-11"
_PROGRAM_MARKER = "VERIFICATION_ALLOWLIST_MARKER.json"

BOUNDED_VERIFICATION_REFUSAL_REASONS: tuple[str, ...] = (
    "output_namespace_not_locked",
    "research_destination_refused",
    "store_namespace_refused",
    "supersession_head_witness_refused",
    "synthetic_marker_refused",
    "verification_authorization_invalid",
    "sixth_day_refused",
    "allowlist_not_chronological",
    "protected_or_sealed_date",
    "date_domain_mismatch",
    "logical_days_not_consecutive",
    "source_inventory_mismatch",
    "rotated_window_refused",
    "seed_profile_mismatch",
    "seed_snapshot_unverifiable",
    "seed_discontinuous_with_window",
    "calendar_unavailable",
)


class BoundedVerificationRefusalError(PermissionError):
    """A §6.1 preflight check refused (typed ``reason``) — before any path."""

    def __init__(self, reason: str, message: str) -> None:
        if reason not in BOUNDED_VERIFICATION_REFUSAL_REASONS:
            raise ValueError(f"unregistered bounded-verification refusal reason {reason!r}")
        super().__init__(f"{reason}: {message}")
        self.reason = reason


# ── §6.1 preflight ──────────────────────────────────────────────────────────


class BoundedVerificationPreflight(FrozenContract):
    """The record of one passed preflight (every check named; all true)."""

    preflight_policy_id: Literal["bounded_verification_preflight_v1"] = (
        "bounded_verification_preflight_v1"
    )
    store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    supersession_head_witness: SupersessionHeadWitness
    verification_run_id: str = Field(pattern=SHA256_PATTERN)
    pipeline_semantic_id: str = Field(pattern=SHA256_PATTERN)
    output_root: str
    allowlist: tuple[str, ...]
    allowlist_hash: str = Field(pattern=SHA256_PATTERN)
    logical_trading_days: tuple[str, ...]
    physical_partition_dates: tuple[str, ...]
    seed_snapshot_id: str = Field(pattern=SHA256_PATTERN)
    checks: ImmutableMap[str, bool]
    passed: Literal[True] = True

    @model_validator(mode="after")
    def _all_true(self):
        if not all(self.checks.values()):
            raise ValueError("a preflight record is only minted when every check passed")
        if self.allowlist_hash != allowlist_sha256(self.allowlist):
            raise ValueError("allowlist_hash does not hash the allowlist")
        return self


def _logical_days_between(first: str, last: str) -> tuple[str, ...]:
    """The consecutive logical trading days in ``[first, last]`` under the
    registered calendar policy (Phase 3 ``trading_calendar``)."""

    try:
        from .trading_calendar import logical_trading_days  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - wiring guard
        raise BoundedVerificationRefusalError(
            "calendar_unavailable",
            "the logical trading-day calendar is not available; the allowlist's date domain "
            "cannot be verified",
        ) from error
    return tuple(str(day) for day in logical_trading_days(first, last))


def _physical_partition_dates(days: tuple[str, ...]) -> tuple[str, ...]:
    try:
        from .trading_calendar import physical_partitions_for  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - wiring guard
        raise BoundedVerificationRefusalError(
            "calendar_unavailable", "the physical partition mapping is not available"
        ) from error
    dates: list[str] = []
    for day in days:
        for partition in physical_partitions_for(day):
            date = str(getattr(partition, "physical_utc_date", partition))
            if date not in dates:
                dates.append(date)
    return tuple(sorted(dates))


def _refuse(reason: str, message: str) -> BoundedVerificationRefusalError:
    return BoundedVerificationRefusalError(reason, message)


def preflight_bounded_verification(
    *,
    store_root: Path,
    repo_root: Path,
    verification_run: VerificationRunEnvelope,
    authorization: object,
    pipeline_semantic_id: str,
    baseline_profile_id: str,
    baseline_section_config_hash: str,
    logical_day_refs: tuple[Any, ...] | None = None,
    seed_loader=None,
) -> BoundedVerificationPreflight:
    """§6.1 — every check runs BEFORE any source path is constructed; the
    first failure raises a typed :class:`BoundedVerificationRefusalError`.

    ``logical_day_refs`` (Phase 3 ``VerificationTradingDayRef`` records) are
    the owner-selected logical-day → physical-partition mapping; when given
    they must cover exactly the allowlist. ``seed_loader`` (default
    ``child_replay.load_seed_snapshot``) is the VERIFIED store load the
    runner itself uses."""

    store_root = Path(store_root)
    repo_root = Path(repo_root)
    checks: dict[str, bool] = {}
    payload = verification_run.payload
    # 1. the output namespace is locked to search_test/v1 under repo_root
    canonical = (repo_root / SEARCH_TEST_STORE_ROOT).resolve()
    if store_root.resolve() != canonical:
        raise _refuse(
            "output_namespace_not_locked",
            f"verification output is locked to {SEARCH_TEST_STORE_ROOT}; got {store_root}",
        )
    checks["output_namespace_locked"] = True
    if path_looks_like_research_store(store_root) or payload.output_namespace != "search_test/v1":
        raise _refuse(
            "research_destination_refused",
            "a research-catalog destination can never receive verification outputs",
        )
    checks["research_destination_refused"] = True
    # 2. the store namespace (a coherently deployed ``test`` namespace)
    try:
        namespace = require_store_namespace(store_root, expected_class="test")
        assert_namespace_deployment_coherent(store_root, namespace)
    except StoreNamespaceError as error:
        raise _refuse("store_namespace_refused", str(error)) from error
    checks["store_namespace_verified"] = True
    # 3. the authorization is a real ref, never the synthetic marker
    if isinstance(authorization, SyntheticAuthorizationMarker) or not isinstance(
        authorization, VerificationAuthorizationRef
    ):
        raise _refuse(
            "synthetic_marker_refused",
            "the real bounded verification requires a VerificationAuthorizationRef; the "
            "synthetic marker (or anything else) is refused",
        )
    checks["real_authorization_present"] = True
    # 4. the supersession-head witness is CURRENT
    try:
        if authorization.store_namespace_id != namespace.store_namespace_id:
            raise StoreNamespaceError(
                "store_namespace_identity_mismatch",
                "the authorization names another store namespace",
            )
        assert_head_witness_current(store_root, authorization.supersession_head_witness)
    except StoreNamespaceError as error:
        raise _refuse("supersession_head_witness_refused", str(error)) from error
    checks["supersession_head_witness_current"] = True
    # 5. the allowlist structure: 1–5 unique ordered days inside the permitted window
    days = tuple(str(day) for day in payload.allowlist)
    if len(days) > 5:
        raise _refuse("sixth_day_refused", f"{len(days)} days requested; at most five")
    if not days or days != tuple(sorted(days)) or len(set(days)) != len(days):
        raise _refuse("allowlist_not_chronological", "allowlist days must be unique and ordered")
    forbidden = [day for day in days if day >= _PROTECTED_BUFFER_DAY]
    if forbidden:
        raise _refuse(
            "protected_or_sealed_date",
            f"June 11 and the sealed range are excluded from every window: {forbidden}",
        )
    checks["window_inside_permitted_range"] = True
    # 6. the run/authorization binding (fail-before-path; §6)
    try:
        validate_verification_run(
            verification_run,
            expected_pipeline_semantic_id=pipeline_semantic_id,
            expected_baseline_profile_id=baseline_profile_id,
            expected_baseline_section_config_hash=baseline_section_config_hash,
            expected_seed_snapshot_id=payload.seed_snapshot_id,
            authorization=authorization,
            store_root=store_root,
        )
    except VerificationRunValidationError as error:
        raise _refuse("verification_authorization_invalid", str(error)) from error
    checks["verification_authorization_valid"] = True
    # 7. the date domain: consecutive LOGICAL trading days under the registered calendar
    logical = _logical_days_between(days[0], days[-1])
    not_trading = [day for day in days if day not in set(logical)]
    if not_trading:
        raise _refuse(
            "date_domain_mismatch",
            f"{not_trading} are not logical trading-day ids (a physical partition date — e.g. "
            "the Sunday file holding the 18:00 ET open — is not a trading day)",
        )
    if logical != days:
        raise _refuse(
            "logical_days_not_consecutive",
            f"the allowlist skips logical trading days between {days[0]} and {days[-1]}: "
            f"{sorted(set(logical) - set(days))}",
        )
    checks["logical_days_consecutive"] = True
    physical = _physical_partition_dates(days)
    if logical_day_refs is not None:
        ref_days = tuple(str(getattr(ref, "logical_trading_day", ref)) for ref in logical_day_refs)
        if ref_days != days:
            raise _refuse(
                "source_inventory_mismatch",
                "the logical-day mapping does not cover exactly the allowlist "
                f"({ref_days} ≠ {days})",
            )
        for ref in logical_day_refs:
            partitions = tuple(getattr(ref, "ordered_source_partition_refs", ()))
            if not partitions:
                raise _refuse(
                    "source_inventory_mismatch",
                    f"logical day {getattr(ref, 'logical_trading_day', ref)} maps to no "
                    "hash-addressable physical partition",
                )
            for partition in partitions:
                if not str(getattr(partition, "content_sha256", "")):
                    raise _refuse(
                        "source_inventory_mismatch",
                        "a physical partition of the mapping carries no content hash",
                    )
    checks["physical_partitions_mapped"] = True
    # 7. the ONE canonical program allowlist (never registered here)
    marker = store_root / _PROGRAM_MARKER
    if marker.exists():
        try:
            record = json.loads(marker.read_text(encoding="utf-8"))
        except (ValueError, OSError) as error:
            raise _refuse(
                "rotated_window_refused", "the program allowlist marker is unreadable"
            ) from error
        if record.get("allowlist_hash") != payload.allowlist_hash:
            raise _refuse(
                "rotated_window_refused",
                "the canonical program allowlist marker records a different window; rotating "
                "or substituting dates is refused (one canonical allowlist, V3 P0-7)",
            )
    checks["canonical_program_allowlist_consistent"] = True
    # 8. the seed: exact verified load, profile-bound, continuous with the window
    from .child_replay import SeedSnapshotError  # noqa: PLC0415

    if seed_loader is None:
        from .child_replay import load_seed_snapshot  # noqa: PLC0415

        seed_loader = load_seed_snapshot
    # adversarial RA-05: every refusal keeps ITS cause — a profile / hash
    # refusal of the verified loader, a store-level failure (missing entry,
    # corrupt manifest or sidecar, sandbox-refused pickle bytes are
    # SeedSnapshotError), a namespace failure; anything else propagates
    try:
        snapshot, _chain_start = seed_loader(
            store_root,
            payload.seed_snapshot_id,
            expected_section_config_hash=baseline_section_config_hash,
        )
    except SeedSnapshotError as error:
        raise _refuse("seed_profile_mismatch", f"seed snapshot refused: {error}") from error
    except StoreNamespaceError as error:
        raise _refuse("store_namespace_refused", str(error)) from error
    except SearchStoreError as error:
        raise _refuse(
            "seed_snapshot_unverifiable",
            f"seed snapshot {payload.seed_snapshot_id[:12]}… is not a verified store entry: "
            f"{error}",
        ) from error
    if snapshot.payload.first_replay_day != days[0] or (
        snapshot.payload.resolved_section_config_hash != baseline_section_config_hash
    ):
        raise _refuse(
            "seed_discontinuous_with_window",
            f"seed first_replay_day {snapshot.payload.first_replay_day!r} / profile do not "
            f"continue into the window starting {days[0]!r}",
        )
    checks["seed_profile_bound_and_continuous"] = True
    return BoundedVerificationPreflight(
        store_namespace_id=namespace.store_namespace_id,
        supersession_head_witness=authorization.supersession_head_witness,
        verification_run_id=verification_run.verification_run_id,
        pipeline_semantic_id=pipeline_semantic_id,
        output_root=str(store_root.resolve()),
        allowlist=days,
        allowlist_hash=payload.allowlist_hash,
        logical_trading_days=logical,
        physical_partition_dates=physical,
        seed_snapshot_id=payload.seed_snapshot_id,
        checks=checks,
    )


# ── §6.2 R1 baseline gate report ────────────────────────────────────────────

R1_BASELINE_GATE_POLICY_ID = "r1_baseline_verification_gates_v1"
R1_BASELINE_GATE_STORE = "r1_baseline_gate_reports"
R1_BASELINE_PROOF_IDS: tuple[str, ...] = (
    "native_ids_repeat",
    "core_table_hashes_repeat",
    "executed_trade_table_exact_loads",
    "second_invocation_reused_with_zero_replay",
    "identical_bytes_reuse",
    "different_bytes_fail_closed",
    "missing_or_corrupt_manifest_fails_closed",
    "corrupt_sidecar_fails_closed",
)


class R1BaselineGatePayload(FrozenContract):
    gate_policy_id: Literal["r1_baseline_verification_gates_v1"] = R1_BASELINE_GATE_POLICY_ID
    verification_run_id: str = Field(pattern=SHA256_PATTERN)
    pipeline_semantic_id: str = Field(pattern=SHA256_PATTERN)
    store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    supersession_head_witness: SupersessionHeadWitness
    #: the six gates of the first (fresh) attempt and of the second (verified
    #: reuse) attempt — both must pass
    first_attempt_gates: ControlFlowGateReport
    second_attempt_gates: ControlFlowGateReport
    #: the dual-drive neutrality facts (audit-disabled vs audit-enabled)
    audit_disabled_core_table_hashes_sha256: str = Field(pattern=SHA256_PATTERN)
    audit_enabled_core_table_hashes_sha256: str = Field(pattern=SHA256_PATTERN)
    audit_modes_equal: bool
    proofs: ImmutableMap[str, bool]
    evidence_refs: ImmutableMap[str, str]
    passed: bool
    verification_only: Literal[True] = True
    not_for_research_interpretation: Literal[True] = True
    full_pipeline_not_run: Literal[True] = True

    @model_validator(mode="after")
    def _complete(self):
        if tuple(sorted(self.proofs)) != tuple(sorted(R1_BASELINE_PROOF_IDS)):
            raise ValueError(
                "the R1 baseline proofs must cover exactly the registered proof ids"
            )
        expected = (
            self.first_attempt_gates.passed
            and self.second_attempt_gates.passed
            and self.audit_modes_equal
            and all(self.proofs.values())
        )
        if self.passed != expected:
            raise ValueError("passed disagrees with the gates and proofs")
        return self


class R1BaselineGateEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "r1_baseline_gate_report_id"

    r1_baseline_gate_report_id: str = Field(pattern=SHA256_PATTERN)
    payload: R1BaselineGatePayload


def build_r1_baseline_gate_report(
    *,
    verification_run_id: str,
    pipeline_semantic_id: str,
    store_namespace_id: str,
    supersession_head_witness: SupersessionHeadWitness,
    first_attempt_gates: ControlFlowGateReport,
    second_attempt_gates: ControlFlowGateReport,
    audit_disabled_core_table_hashes_sha256: str,
    audit_enabled_core_table_hashes_sha256: str,
    proofs: Mapping[str, bool],
    evidence_refs: Mapping[str, str],
) -> R1BaselineGateEnvelope:
    """Assemble the immutable report; ``passed`` is derived, never asserted."""

    complete = {proof: bool(proofs.get(proof, False)) for proof in R1_BASELINE_PROOF_IDS}
    equal = audit_disabled_core_table_hashes_sha256 == audit_enabled_core_table_hashes_sha256
    payload = R1BaselineGatePayload(
        verification_run_id=verification_run_id,
        pipeline_semantic_id=pipeline_semantic_id,
        store_namespace_id=store_namespace_id,
        supersession_head_witness=supersession_head_witness,
        first_attempt_gates=first_attempt_gates,
        second_attempt_gates=second_attempt_gates,
        audit_disabled_core_table_hashes_sha256=audit_disabled_core_table_hashes_sha256,
        audit_enabled_core_table_hashes_sha256=audit_enabled_core_table_hashes_sha256,
        audit_modes_equal=equal,
        proofs=complete,
        evidence_refs={str(k): str(v) for k, v in sorted(evidence_refs.items())},
        passed=(
            first_attempt_gates.passed
            and second_attempt_gates.passed
            and equal
            and all(complete.values())
        ),
    )
    return R1BaselineGateEnvelope.from_payload(payload)


def save_r1_baseline_gate_report(root: Path, envelope: R1BaselineGateEnvelope):
    return save_or_reuse_envelope(Path(root), R1_BASELINE_GATE_STORE, envelope)


#: The four "also prove" proofs GATHERED on the fixture's own executed-trade
#: table through scratch copies (adversarial RA-04) — never asserted blindly.
STORE_BEHAVIOR_PROOF_IDS: tuple[str, ...] = (
    "identical_bytes_reuse",
    "different_bytes_fail_closed",
    "missing_or_corrupt_manifest_fails_closed",
    "corrupt_sidecar_fails_closed",
)


def store_behavior_proofs(
    store_root: Path, executed_trade_table_id: str, scratch_root: Path
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Probe the immutable store's behavior on ONE executed-trade table
    through scratch COPIES under ``scratch_root`` (never the store itself):
    an identical copy loads and a same-identity publication is REUSED; one
    flipped sidecar byte, a missing manifest, a corrupt manifest and a
    truncated sidecar each fail closed with a TYPED refusal. The scratch root
    is removed in ``finally``. Returns ``(proofs, evidence)`` — an
    already-tampered source shows up as a failed identical-bytes proof."""

    import shutil  # noqa: PLC0415

    from .executed_trade_table import (  # noqa: PLC0415
        EXECUTED_TRADE_TABLE_SIDECAR,
        EXECUTED_TRADE_TABLE_STORE,
        load_executed_trade_table,
        save_executed_trade_table,
    )

    source = Path(store_root) / EXECUTED_TRADE_TABLE_STORE / executed_trade_table_id
    scratch_root = Path(scratch_root)
    proofs = {proof: False for proof in STORE_BEHAVIOR_PROOF_IDS}
    observations: dict[str, str] = {}

    def _copy(name: str) -> Path:
        scratch = scratch_root / name
        target = scratch / EXECUTED_TRADE_TABLE_STORE / executed_trade_table_id
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        return scratch

    def _reason(error: Exception) -> str:
        return str(getattr(error, "reason", type(error).__name__))

    def _refuses(scratch: Path, label: str) -> bool:
        try:
            load_executed_trade_table(scratch, executed_trade_table_id)
        except (SearchStoreError, ValueError) as error:
            observations[label] = _reason(error)
            return True
        observations[label] = "loaded_without_refusal"
        return False

    try:
        if not source.is_dir():
            observations["identical_bytes"] = "store_entry_missing"
            return proofs, {"executed_trade_table_id": executed_trade_table_id,
                            "observations": observations}
        # identical bytes: the copy loads AND a same-identity publication reuses
        scratch = _copy("identical")
        try:
            verified = load_executed_trade_table(scratch, executed_trade_table_id)
            _stored, reused = save_executed_trade_table(
                scratch, verified.envelope, verified.table_bytes
            )
            proofs["identical_bytes_reuse"] = bool(reused)
            observations["identical_bytes"] = "reused" if reused else "not_reused"
        except (SearchStoreError, ValueError) as error:
            observations["identical_bytes"] = _reason(error)
        # different bytes under the same identity
        scratch = _copy("different_bytes")
        sidecar = scratch / EXECUTED_TRADE_TABLE_STORE / executed_trade_table_id
        sidecar = sidecar / EXECUTED_TRADE_TABLE_SIDECAR
        data = sidecar.read_bytes()
        sidecar.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
        proofs["different_bytes_fail_closed"] = _refuses(scratch, "different_bytes")
        # a missing manifest AND a corrupt manifest must both refuse
        scratch = _copy("missing_manifest")
        (scratch / EXECUTED_TRADE_TABLE_STORE / executed_trade_table_id / "manifest.json").unlink()
        missing = _refuses(scratch, "missing_manifest")
        scratch = _copy("corrupt_manifest")
        manifest = scratch / EXECUTED_TRADE_TABLE_STORE / executed_trade_table_id / "manifest.json"
        manifest.write_text("{not json", encoding="utf-8")
        corrupt = _refuses(scratch, "corrupt_manifest")
        proofs["missing_or_corrupt_manifest_fails_closed"] = missing and corrupt
        # a corrupt (truncated) sidecar
        scratch = _copy("corrupt_sidecar")
        sidecar = scratch / EXECUTED_TRADE_TABLE_STORE / executed_trade_table_id
        sidecar = sidecar / EXECUTED_TRADE_TABLE_SIDECAR
        sidecar.write_bytes(data[: max(1, len(data) // 2)])
        proofs["corrupt_sidecar_fails_closed"] = _refuses(scratch, "corrupt_sidecar")
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)
    return proofs, {
        "executed_trade_table_id": executed_trade_table_id,
        "observations": dict(sorted(observations.items())),
    }


# ── §6.3 bounded release control-flow report ────────────────────────────────

BOUNDED_CONTROL_FLOW_POLICY_ID = "bounded_release_control_flow_gates_v1"
BOUNDED_CONTROL_FLOW_STORE = "bounded_release_control_flow_reports"
BOUNDED_COMPONENTS: tuple[str, ...] = (
    "mbp1_diagnostic",
    "context_bar_panel",
    "fold_construction",
    "regime_fit",
    "supervised_models",
    "s14_reports",
    "s15_publication",
    "s11_model_gated_replays",
)
#: The typed outcomes each component may reach on the bounded fixture; the
#: first entries are the PASSING ones (the required bounded result of §6.3
#: table), the last is the failing catch-all.
BOUNDED_COMPONENT_OUTCOMES: dict[str, tuple[str, ...]] = {
    "mbp1_diagnostic": (
        "diagnostic_completed_completeness_unknown",
        "diagnostic_completed_declared_evidence",
        "not_planned",
        "unexpected_state",
    ),
    "context_bar_panel": ("panel_materialized", "typed_insufficiency", "not_planned",
                          "unexpected_state"),
    "fold_construction": ("typed_no_valid_fold", "valid_folds_present", "not_planned",
                          "unexpected_state"),
    "regime_fit": ("typed_non_fit_recorded", "diagnostic_fit_verification_only",
                   "not_planned", "unexpected_state"),
    "supervised_models": ("typed_non_fit_zero_predictions", "fits_verification_only",
                          "not_planned", "unexpected_state"),
    "s14_reports": ("verification_report_zero_fitting", "typed_skip_with_evidence",
                    "not_planned", "unexpected_state"),
    "s15_publication": ("verification_only_result", "not_planned", "unexpected_state"),
    "s11_model_gated_replays": ("blocked_with_registered_reason", "not_planned",
                                "unexpected_state"),
}
_PASSING_OUTCOMES: dict[str, frozenset[str]] = {
    "mbp1_diagnostic": frozenset(
        {"diagnostic_completed_completeness_unknown", "diagnostic_completed_declared_evidence",
         "not_planned"}
    ),
    "context_bar_panel": frozenset({"panel_materialized", "typed_insufficiency", "not_planned"}),
    "fold_construction": frozenset({"typed_no_valid_fold", "valid_folds_present", "not_planned"}),
    "regime_fit": frozenset(
        {"typed_non_fit_recorded", "diagnostic_fit_verification_only", "not_planned"}
    ),
    "supervised_models": frozenset(
        {"typed_non_fit_zero_predictions", "fits_verification_only", "not_planned"}
    ),
    "s14_reports": frozenset(
        {"verification_report_zero_fitting", "typed_skip_with_evidence", "not_planned"}
    ),
    "s15_publication": frozenset({"verification_only_result"}),
    "s11_model_gated_replays": frozenset({"blocked_with_registered_reason", "not_planned"}),
}


class BoundedComponentResult(FrozenContract):
    component: str
    outcome: str
    passed: bool
    detail: str
    evidence: ImmutableMap[str, Any]

    @model_validator(mode="after")
    def _registered(self):
        if self.component not in BOUNDED_COMPONENTS:
            raise ValueError(f"unregistered bounded component {self.component!r}")
        if self.outcome not in BOUNDED_COMPONENT_OUTCOMES[self.component]:
            raise ValueError(
                f"unregistered outcome {self.outcome!r} for component {self.component!r}"
            )
        if self.passed != (self.outcome in _PASSING_OUTCOMES[self.component]):
            raise ValueError("passed disagrees with the outcome's registered class")
        return self


class BoundedReleaseControlFlowPayload(FrozenContract):
    report_policy_id: Literal["bounded_release_control_flow_gates_v1"] = (
        BOUNDED_CONTROL_FLOW_POLICY_ID
    )
    pipeline_semantic_id: str = Field(pattern=SHA256_PATTERN)
    verification_run_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    supersession_head_witness: SupersessionHeadWitness
    components: tuple[BoundedComponentResult, ...]
    passed: bool
    verification_only: Literal[True] = True
    not_for_research_interpretation: Literal[True] = True
    full_pipeline_not_run: Literal[True] = True
    not_research_evidence: Literal[True] = True

    @model_validator(mode="after")
    def _complete(self):
        names = tuple(result.component for result in self.components)
        if names != BOUNDED_COMPONENTS:
            raise ValueError(
                "the bounded report must type exactly the registered components in order"
            )
        if self.passed != all(result.passed for result in self.components):
            raise ValueError("passed disagrees with the component results")
        return self


class BoundedReleaseControlFlowEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "bounded_release_control_flow_report_id"

    bounded_release_control_flow_report_id: str = Field(pattern=SHA256_PATTERN)
    payload: BoundedReleaseControlFlowPayload


def _stage_entry(state: Mapping[str, Any], stage_value: str) -> Mapping[str, Any]:
    return dict(state.get("stages", {}).get(stage_value, {}))


def _stage_sidecar(root: Path, entry: Mapping[str, Any], name: str) -> Any:
    """A stage-result sidecar (verified) or ``None`` when the stage produced
    no result; corruption propagates typed."""

    stage_result_id = entry.get("stage_result_id")
    if not stage_result_id:
        return None
    return load_json_sidecar(Path(root), "pipeline_stage_results", str(stage_result_id), name)


def _result(component: str, outcome: str, detail: str, **evidence: Any) -> BoundedComponentResult:
    return BoundedComponentResult(
        component=component,
        outcome=outcome,
        passed=outcome in _PASSING_OUTCOMES[component],
        detail=detail,
        evidence={key: value for key, value in sorted(evidence.items())},
    )


def _terminal(entry: Mapping[str, Any]) -> bool:
    return entry.get("status") in ("completed", "reused")


def _mbp1_component(root: Path, diagnostic_id: str | None, state) -> BoundedComponentResult:
    if diagnostic_id is None:
        return _result("mbp1_diagnostic", "not_planned", "no MBP-1 coverage diagnostic requested")
    from ..features.mbp1_coverage_diagnostic import (  # noqa: PLC0415
        Mbp1CoverageDiagnosticEnvelope,
    )

    try:
        envelope = load_verified_envelope(
            root, "mbp1_coverage_diagnostics", diagnostic_id, Mbp1CoverageDiagnosticEnvelope
        )
    except SidecarLoadError as error:
        return _result(
            "mbp1_diagnostic", "unexpected_state", f"diagnostic unverifiable: {error}",
            diagnostic_id=diagnostic_id,
        )
    payload = envelope.payload
    statuses = sorted({str(getattr(row.completeness_status, "value", row.completeness_status))
                       for row in payload.rows})
    if payload.completeness_inferred_from_sequence_continuity:
        return _result(
            "mbp1_diagnostic", "unexpected_state",
            "completeness was inferred from sequence continuity (withdrawn rule)",
            diagnostic_id=diagnostic_id,
        )
    if statuses == ["completeness_unknown"]:
        outcome = "diagnostic_completed_completeness_unknown"
    elif statuses and all(s != "unknown" for s in statuses):
        outcome = "diagnostic_completed_declared_evidence"
    else:
        outcome = "unexpected_state"
    return _result(
        "mbp1_diagnostic", outcome,
        f"diagnostic completed; completeness statuses {statuses}; no feature eligibility or "
        "research claim is made",
        diagnostic_id=diagnostic_id, completeness_statuses=statuses,
        feature_eligibility_claimed=False,
    )


def _panel_component(root: Path, state) -> BoundedComponentResult:
    entry = _stage_entry(state, "05_materialize_feature_views")
    if not entry.get("in_plan"):
        return _result("context_bar_panel", "not_planned", "S05 not in the stage plan")
    if not _terminal(entry):
        return _result(
            "context_bar_panel", "unexpected_state", f"S05 status {entry.get('status')!r}",
            explanation=str(entry.get("explanation") or ""),
        )
    record = _stage_sidecar(root, entry, "bundle_feature_views.json") or {}
    observation = record.get("__regime_observation__") or {}
    if observation.get("source_kind") != "context_bar_panel":
        return _result(
            "context_bar_panel", "not_planned",
            "no panel-grain regime study; S05 materialized candidate bundle views only",
        )
    # adversarial B-04: the typed validity comes from the PERSISTED panel
    # artifact (manifest-verified frame), never from a stage failure string
    from ..features.context_bar_panel_materializer import (  # noqa: PLC0415
        load_context_bar_panel_artifact,
        load_context_bar_panel_frame,
    )

    artifact_id = str(observation.get("artifact_id") or "")
    try:
        envelope = load_context_bar_panel_artifact(root, artifact_id)
        frame = load_context_bar_panel_frame(root, envelope)
    except (SearchStoreError, ValueError) as error:
        return _result(
            "context_bar_panel", "unexpected_state",
            f"the persisted panel artifact is unverifiable: {error}",
            panel_artifact_id=artifact_id,
        )
    if "cbp_valid" not in frame.columns or "cbp_missing_reason" not in frame.columns:
        return _result(
            "context_bar_panel", "unexpected_state",
            "the persisted panel carries no typed validity columns",
            panel_artifact_id=artifact_id,
        )
    valid = frame["cbp_valid"].astype(bool)
    reasons = (
        frame.loc[~valid, "cbp_missing_reason"].astype(str).value_counts().sort_index()
    )
    typed_counts = {str(reason): int(count) for reason, count in reasons.items()}
    row_count = int(len(frame))
    valid_rows = int(valid.sum())
    evidence = dict(
        panel_artifact_id=artifact_id,
        replay_chart_artifact_id=observation.get("replay_chart_artifact_id"),
        row_count=row_count,
        valid_row_count=valid_rows,
        typed_null_reason_counts=typed_counts,
    )
    if row_count > 0 and valid_rows == 0:
        return _result(
            "context_bar_panel", "typed_insufficiency",
            "every panel row carries a typed null reason on the bounded window (the "
            "documented insufficiency shape); no feature was fabricated",
            **evidence,
        )
    if valid_rows > 0:
        return _result(
            "context_bar_panel", "panel_materialized",
            "the context-bar panel materialized from the verified replay-chart pair with "
            "typed validity per row",
            **evidence,
        )
    return _result(
        "context_bar_panel", "unexpected_state", "the persisted panel is empty", **evidence
    )


def _fold_component(root: Path, state) -> BoundedComponentResult:
    entry = _stage_entry(state, "08_build_folds")
    if not entry.get("in_plan"):
        return _result("fold_construction", "not_planned", "S08 not in the stage plan")
    if not _terminal(entry):
        return _result(
            "fold_construction", "unexpected_state", f"S08 status {entry.get('status')!r}",
            explanation=str(entry.get("explanation") or ""),
        )
    # adversarial B-03: the fold outcome is a TYPED stage fact — the S08
    # `fold_summary.json` sidecar (labeled folds) or, for the label-free
    # regime branch, the adequacy preview's registered outcome; the sanitized
    # explanation is never parsed
    summary = _stage_sidecar(root, entry, "fold_summary.json")
    adequacy = _stage_sidecar(root, entry, "fold_sample_adequacy.json")
    evidence: dict[str, Any] = {}
    if isinstance(adequacy, dict):
        preview = dict(adequacy.get("sample_adequacy_preview") or {})
        evidence["regime_sample_adequacy"] = {
            "expected_gate_outcome": preview.get("expected_gate_outcome"),
            "fold_schedule_id": adequacy.get("fold_schedule_id"),
        }
    if isinstance(summary, dict) and "valid_fold_count" in summary:
        fold_count = int(summary.get("fold_count", 0) or 0)
        valid_folds = int(summary["valid_fold_count"])
        evidence.update(
            source="fold_summary.json",
            fold_count=fold_count,
            valid_fold_count=valid_folds,
            invalid_reasons=[str(reason) for reason in (summary.get("invalid_reasons") or [])],
        )
        if valid_folds == 0:
            return _result(
                "fold_construction", "typed_no_valid_fold",
                "the fixture produced a typed no-valid-fold / insufficient-history result "
                "under the real fold protocol; no fabricated valid fold",
                **evidence,
            )
        return _result(
            "fold_construction", "valid_folds_present",
            "valid folds exist on this fixture (verification-only)",
            **evidence,
        )
    if isinstance(adequacy, dict):
        evidence["source"] = "fold_sample_adequacy.json"
        outcome_hint = evidence["regime_sample_adequacy"]["expected_gate_outcome"]
        if outcome_hint == "no_valid_folds":
            return _result(
                "fold_construction", "typed_no_valid_fold",
                "the label-free regime folds carry the typed no_valid_folds adequacy outcome",
                **evidence,
            )
        if outcome_hint in ("pass", "fail"):
            return _result(
                "fold_construction", "valid_folds_present",
                "label-free regime folds exist on this fixture (verification-only; the "
                "adequacy verdict is a regime fact, never a fold fabrication)",
                **evidence,
            )
    return _result(
        "fold_construction", "unexpected_state",
        "no typed fold sidecar (fold_summary.json / fold_sample_adequacy.json) was persisted",
        **evidence,
    )


def _regime_component(root: Path, state) -> BoundedComponentResult:
    entry = _stage_entry(state, "09_train_models")
    if not entry.get("in_plan"):
        return _result("regime_fit", "not_planned", "S09 not in the stage plan")
    record = _stage_sidecar(root, entry, "regime_run.json") if _terminal(entry) else None
    if not record:
        if _terminal(entry):
            return _result("regime_fit", "not_planned", "no regime study in this run")
        return _result(
            "regime_fit", "unexpected_state", f"S09 status {entry.get('status')!r}",
            explanation=str(entry.get("explanation") or ""),
        )
    s09a = dict(record.get("S09a") or {})
    fits = list(s09a.get("regime_fit_ids") or [])
    gates_passed = bool(s09a.get("gates_passed"))
    failures = list(s09a.get("gate_failures") or [])
    if not fits and not gates_passed:
        return _result(
            "regime_fit", "typed_non_fit_recorded",
            "no lawful fit on the bounded fixture; the typed non-fit capability assessment "
            "was persisted and no invalid fit exists",
            gate_failures=failures,
            regime_capability_assessment_id=s09a.get("regime_capability_assessment_id"),
        )
    if fits:
        return _result(
            "regime_fit", "diagnostic_fit_verification_only",
            "diagnostic fits exist (verification-only; never a research claim)",
            regime_fit_count=len(fits), gates_passed=gates_passed, gate_failures=failures,
            regime_capability_assessment_id=s09a.get("regime_capability_assessment_id"),
        )
    return _result("regime_fit", "unexpected_state", "the regime fit state could not be typed")


def _supervised_component(root: Path, state) -> BoundedComponentResult:
    entry = _stage_entry(state, "10_generate_predictions_and_diagnostics")
    if not entry.get("in_plan"):
        return _result("supervised_models", "not_planned", "S10 not in the stage plan")
    if not _terminal(entry):
        return _result(
            "supervised_models", "unexpected_state", f"S10 status {entry.get('status')!r}",
            explanation=str(entry.get("explanation") or ""),
        )
    diagnostics = _stage_sidecar(root, entry, "supervised_ladder.json")
    if not diagnostics:
        return _result(
            "supervised_models", "not_planned", "no supervised ladder in this run"
        )
    parity = dict(diagnostics.get("parity") or {})
    oos_rows = int(parity.get("oos_row_count", 0) or 0)
    rungs = dict(diagnostics.get("rungs") or {})
    if oos_rows == 0:
        return _result(
            "supervised_models", "typed_non_fit_zero_predictions",
            "no valid fold: every rung reports the typed non-fit shape with zero OOS "
            "predictions (nothing fabricated)",
            oos_row_count=0, rungs=sorted(rungs),
        )
    return _result(
        "supervised_models", "fits_verification_only",
        "rungs produced OOS predictions on this fixture (verification-only)",
        oos_row_count=oos_rows, rungs=sorted(rungs),
    )


def _s14_component(root: Path, state) -> BoundedComponentResult:
    entry = _stage_entry(state, "14_build_frontier_and_insights")
    if not entry.get("in_plan"):
        return _result("s14_reports", "not_planned", "S14 not in the stage plan")
    if not _terminal(entry):
        return _result(
            "s14_reports", "unexpected_state", f"S14 status {entry.get('status')!r}",
            explanation=str(entry.get("explanation") or ""),
        )
    outputs = list(entry.get("output_artifact_ids") or [])
    explanation = str(entry.get("explanation") or "")
    # adversarial B-01: the regime record S14 persists is
    # ``regime_stratified_reports.json`` (regime_report_stage.build_reports);
    # its typed ``children_skipped`` and ``fitting_performed`` facts are the
    # evidence — a record that claims fitting is never a passing outcome
    regime = _stage_sidecar(root, entry, "regime_stratified_reports.json")
    record = dict(regime) if isinstance(regime, dict) else None
    skipped = dict((record or {}).get("children_skipped") or {})
    fitting = None if record is None else bool(record.get("fitting_performed", False))
    report_ids = list((record or {}).get("report_ids") or [])
    evidence = dict(
        output_artifact_ids=outputs,
        children_skipped=skipped,
        fitting_performed=fitting,
        report_ids=report_ids,
        regime_record_present=record is not None,
    )
    if fitting:
        return _result(
            "s14_reports", "unexpected_state",
            "the S14 regime record claims estimator fitting; S14 must perform zero fitting",
            **evidence,
        )
    if outputs:
        return _result(
            "s14_reports", "verification_report_zero_fitting",
            "S14 published its verification report(s) from persisted artifacts only "
            "(zero estimator fitting)",
            **evidence,
        )
    return _result(
        "s14_reports", "typed_skip_with_evidence",
        "S14 recorded an explicit typed skip (no feasible child / no report) with evidence",
        explanation=explanation, **evidence,
    )


def _s15_component(root: Path, state) -> BoundedComponentResult:
    entry = _stage_entry(state, "15_verify_and_publish")
    if not entry.get("in_plan"):
        return _result("s15_publication", "not_planned", "S15 not in the stage plan")
    outputs = list(entry.get("output_artifact_ids") or [])
    publication = dict(state.get("publication") or {})
    if not _terminal(entry) or not outputs:
        return _result(
            "s15_publication", "unexpected_state", f"S15 status {entry.get('status')!r}",
            explanation=str(entry.get("explanation") or ""),
        )
    from .pipeline import PipelineResultEnvelope  # noqa: PLC0415

    try:
        result = load_verified_envelope(root, "search_results", outputs[0], PipelineResultEnvelope)
    except SidecarLoadError as error:
        return _result(
            "s15_publication", "unexpected_state", f"pipeline result unverifiable: {error}"
        )
    stamps = dict(result.payload.verification_stamps or {})
    verification_only = (
        str(result.payload.run_scope.value) == "verification_5d"
        and bool(stamps.get("verification_only"))
        and bool(stamps.get("not_for_research_interpretation"))
        and bool(stamps.get("full_pipeline_not_run"))
        and not publication.get("activated")
        and bool(state.get("full_pipeline_not_run", False))
    )
    if verification_only:
        return _result(
            "s15_publication", "verification_only_result",
            "verification-only result: stamped, never activated, no research publication, "
            "frontier selection, promotion or activation",
            pipeline_result_id=outputs[0], stamps=stamps,
            publication_state=publication.get("state"),
        )
    return _result(
        "s15_publication", "unexpected_state",
        "the S15 result is not a stamped, unactivated verification-only result",
        pipeline_result_id=outputs[0], stamps=stamps, publication=publication,
    )


def _s11_component(state) -> BoundedComponentResult:
    entry = _stage_entry(state, "11_run_frozen_model_gated_replays")
    if not entry.get("in_plan"):
        return _result("s11_model_gated_replays", "not_planned", "S11 not in the stage plan")
    from .pipeline import S11_BLOCKED_REASON  # noqa: PLC0415

    if entry.get("status") == "blocked" and entry.get("explanation") == S11_BLOCKED_REASON:
        return _result(
            "s11_model_gated_replays", "blocked_with_registered_reason",
            "S11 remains blocked with the exact registered reason",
            reason=S11_BLOCKED_REASON,
        )
    return _result(
        "s11_model_gated_replays", "unexpected_state",
        f"S11 status {entry.get('status')!r} with an unregistered explanation",
        explanation=str(entry.get("explanation") or ""),
    )


def build_bounded_release_control_flow_report(
    *,
    store_root: Path,
    state: Mapping[str, Any],
    store_namespace_id: str,
    supersession_head_witness: SupersessionHeadWitness,
    verification_run_id: str | None = None,
    mbp1_diagnostic_id: str | None = None,
) -> BoundedReleaseControlFlowEnvelope:
    """Type every component's bounded outcome from a completed pipeline
    state and its persisted stage sidecars (never from research metrics)."""

    root = Path(store_root)
    components = (
        _mbp1_component(root, mbp1_diagnostic_id, state),
        _panel_component(root, state),
        _fold_component(root, state),
        _regime_component(root, state),
        _supervised_component(root, state),
        _s14_component(root, state),
        _s15_component(root, state),
        _s11_component(state),
    )
    payload = BoundedReleaseControlFlowPayload(
        pipeline_semantic_id=str(state["pipeline_semantic_id"]),
        verification_run_id=verification_run_id,
        store_namespace_id=store_namespace_id,
        supersession_head_witness=supersession_head_witness,
        components=components,
        passed=all(result.passed for result in components),
    )
    return BoundedReleaseControlFlowEnvelope.from_payload(payload)


def save_bounded_release_control_flow_report(
    root: Path, envelope: BoundedReleaseControlFlowEnvelope
):
    return save_or_reuse_envelope(Path(root), BOUNDED_CONTROL_FLOW_STORE, envelope)


# ── registry ────────────────────────────────────────────────────────────────


def _example_gates() -> ControlFlowGateReport:
    return ControlFlowGateReport(
        results={gate: True for gate in VERIFICATION_CONTROL_FLOW_GATE_IDS}, passed=True
    )


def _example_r1_payload() -> R1BaselineGatePayload:
    return R1BaselineGatePayload(
        verification_run_id="a" * 64,
        pipeline_semantic_id="b" * 64,
        store_namespace_id="c" * 64,
        supersession_head_witness=SupersessionHeadWitness(
            store_namespace_id="c" * 64, line_count=0, head_sha256="d" * 64
        ),
        first_attempt_gates=_example_gates(),
        second_attempt_gates=_example_gates(),
        audit_disabled_core_table_hashes_sha256="e" * 64,
        audit_enabled_core_table_hashes_sha256="e" * 64,
        audit_modes_equal=True,
        proofs={proof: True for proof in R1_BASELINE_PROOF_IDS},
        evidence_refs={"core_replay_id": "f" * 64},
        passed=True,
    )


def _example_bounded_payload() -> BoundedReleaseControlFlowPayload:
    results = tuple(
        BoundedComponentResult(
            component=component,
            outcome="not_planned" if component != "s15_publication" else "verification_only_result",
            passed=True,
            detail="example",
            evidence={},
        )
        for component in BOUNDED_COMPONENTS
    )
    return BoundedReleaseControlFlowPayload(
        pipeline_semantic_id="b" * 64,
        verification_run_id=None,
        store_namespace_id="c" * 64,
        supersession_head_witness=SupersessionHeadWitness(
            store_namespace_id="c" * 64, line_count=0, head_sha256="d" * 64
        ),
        components=results,
        passed=True,
    )


register_identity_pair(
    name="R1BaselineGateReport",
    envelope_cls=R1BaselineGateEnvelope,
    payload_cls=R1BaselineGatePayload,
    id_field="r1_baseline_gate_report_id",
    example_factory=_example_r1_payload,
)
register_identity_pair(
    name="BoundedReleaseControlFlowReport",
    envelope_cls=BoundedReleaseControlFlowEnvelope,
    payload_cls=BoundedReleaseControlFlowPayload,
    id_field="bounded_release_control_flow_report_id",
    example_factory=_example_bounded_payload,
)
