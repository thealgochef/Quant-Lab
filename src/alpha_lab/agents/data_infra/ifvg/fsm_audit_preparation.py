"""Persisted, lock-guarded preparation of the immutable ``ifvg_fsm_audit_v1``
companion artifact.

Mirrors the development-pair preparation discipline: authoritative-blob
verification, DevelopmentDataAccess discovery (never constructing protected or
sealed paths), the fixed permitted chain, cached day artifacts only by
default, a per-profile job lock with an atomically written state file — and
publication ONLY behind the exact v2 parity gate and the funnel⇔events
reconciliation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audit_contracts import AuditTable, audit_contract_fingerprint
from .config import (
    FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
    FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256,
    FSM_AUDIT_DATASET_DIR,
    V2_DATASET_DIR,
    IfvgCaptureConfig,
)
from .context_experiment_contracts import (
    ProfileCapabilityStatus,
    profile_capability,
)
from .data_access import hash_allowlisted_source_files
from .dataset import FsmAuditBuildResult, build_ifvg_fsm_audit_v1
from .development_access import (
    DEVELOPMENT_CUTOFF_UTC,
    FROZEN_WARMUP_DATES,
    PERMITTED_DEVELOPMENT_DATES,
    DevelopmentDataAccess,
    DevelopmentReplayPolicy,
)
from .fsm_audit_parity import compare_v2_exact, write_parity_report
from .manifest import (
    FsmAuditIdentity,
    canonical_sha256,
    fsm_audit_dataset_id_for,
    save_fsm_audit_immutable,
)
from .preparation import PREPARATION_JOB_ROOT, _profile_lock, _write_json_atomic
from .profiles import resolve_profile_config
from .verification import (
    _AUTHORITATIVE_SOURCE_BLOB,
    _repository_states,
    _verify_authoritative_blob,
)

__all__ = [
    "FsmAuditPreparationResult",
    "prepare_ifvg_fsm_audit_persisted",
]

_DROP_REASONS = (
    "retention_not_selected",
    "conflicted",
    "outranked",
    "direction_disabled",
    "slot_occupied",
    "filled_before_activation",
    "age_evicted",
    "cap_evicted",
    "causality_failed",
    "reaction_window_expired",
    "distance_gt_profile",
    "selected",
    "replaced",
)


@dataclass(frozen=True, slots=True)
class FsmAuditPreparationResult:
    artifact_id: str
    exploration_dir: Path
    parity_report: dict
    reconciliation_report: dict
    coverage_report: dict
    capacity_report: dict
    access_audit: dict[str, Any]
    replay_dates: tuple[str, ...]


def build_evidence_coverage_report(build: FsmAuditBuildResult) -> dict:
    """Coverage of the contract's drop-reason enumeration: every reason is
    either observed (count > 0) or PROVABLY ZERO — never silently absent."""
    import pandas as pd

    taps = build.audit_tables.get(AuditTable.HTF_TAP, pd.DataFrame())
    parents = build.audit_tables.get(AuditTable.PARENT_CANDIDATE, pd.DataFrame())
    opposing = build.audit_tables.get(AuditTable.OPPOSING, pd.DataFrame())
    fills = build.audit_tables.get(AuditTable.FILL_EVENT, pd.DataFrame())
    windows = build.audit_tables.get(AuditTable.PARENT_WINDOW, pd.DataFrame())

    def _drop_counts(frame) -> dict[str, int]:
        if frame.empty:
            return {}
        return {
            str(reason): int(count)
            for reason, count in frame["drop_reason"].dropna().value_counts().items()
        }

    def _selected(frame) -> int:
        if frame.empty:
            return 0
        return int(frame["selected"].astype(bool).sum())

    counts: dict[str, int] = {reason: 0 for reason in _DROP_REASONS}
    for frame in (taps, parents, opposing):
        for reason, count in _drop_counts(frame).items():
            counts[reason] = counts.get(reason, 0) + count
    counts["selected"] = _selected(taps) + _selected(parents) + _selected(opposing)
    if not fills.empty:
        kinds = fills["event_kind"].value_counts()
        counts["age_evicted"] = int(kinds.get("evicted_age", 0))
        counts["cap_evicted"] = int(kinds.get("evicted_cap", 0))
        counts["filled_before_activation"] = int(
            (
                (fills["event_kind"] == "filled")
                & (fills["fvg_role"] == "registry_only")
            ).sum()
        )
    if not windows.empty:
        counts["replaced"] = int(
            (
                (windows["event_kind"] == "parent_selected")
                & windows["prior_parent_fvg_id"].notna()
            ).sum()
        )
    return {
        "drop_reason_counts": counts,
        "provably_zero": sorted(
            reason for reason, count in counts.items() if count == 0
        ),
        "audit_rows_by_table": {
            table.value: int(len(frame))
            for table, frame in build.audit_tables.items()
        },
    }


def prepare_ifvg_fsm_audit_persisted(
    *,
    repo_root: Path,
    profile_name: str = "ifvg_v2_doc_default_fresh_static_1r",
    cached_artifacts_only: bool = True,
    output_base: Path | None = None,
    parity_report_path: Path | None = None,
    progress_fn: Callable[[int, int, str], None] | None = None,
) -> FsmAuditPreparationResult:
    capability = profile_capability(profile_name)
    if capability.status is not ProfileCapabilityStatus.RUNNABLE:
        raise PermissionError(
            f"profile is {capability.status.value}: {capability.reason}"
        )
    root = Path(repo_root).resolve()
    _verify_authoritative_blob(root)
    resolved = resolve_profile_config({"profile_name": profile_name})
    from dataclasses import replace as _cfg_replace

    base_cfg = IfvgCaptureConfig()
    cfg = _cfg_replace(
        base_cfg,
        section=resolved.section,
        session_scheme=base_cfg.session_scheme,
        data_dir=root / "data" / "databento",
    )
    from .preparation import _discover_sources

    discovery = DevelopmentDataAccess()
    source_files = _discover_sources(
        discovery, data_dir=cfg.data_dir, symbol=cfg.symbol
    )
    if tuple(day for day in FROZEN_WARMUP_DATES if day in source_files) != (
        FROZEN_WARMUP_DATES
    ):
        raise RuntimeError("the frozen ten-date warmup is not fully available")
    replay_dates = tuple(
        day for day in PERMITTED_DEVELOPMENT_DATES if day in source_files
    )
    if "2026-06-10" not in replay_dates:
        raise RuntimeError("the development chain does not reach the June 10 cutoff")

    job_name = f"{profile_name}__fsm_audit_v1"
    job_root = (root / PREPARATION_JOB_ROOT).resolve()
    state_dir = job_root / job_name
    with _profile_lock(job_root, job_name):
        state = {
            "job": job_name,
            "status": "building",
            "replay_dates": list(replay_dates),
            "artifact_id": None,
            "error_code": None,
        }
        _write_json_atomic(state_dir / "state.json", state)
        try:
            policy = DevelopmentReplayPolicy(
                replay_dates, development_audit=discovery.audit
            )
            source_hashes = hash_allowlisted_source_files(policy, source_files)
            accepted_dir = (
                root
                / V2_DATASET_DIR
                / FSM_AUDIT_ACCEPTED_V2_DATASET_ID
                / "exploration"
            )
            build = build_ifvg_fsm_audit_v1(
                replay_dates,
                cfg,
                resolved,
                accepted_v2_exploration_dir=accepted_dir,
                access_policy=policy,
                cached_artifacts_only=cached_artifacts_only,
                progress_fn=progress_fn,
            )
            # Deep exact-parity report (row-level diagnostics on top of the
            # builder's own content-hash gate).
            from .dataset import load_accepted_v2_tables

            accepted_tables = load_accepted_v2_tables(
                accepted_dir,
                expected_dataset_id=FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
                expected_manifest_payload_sha256=FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256,
            )
            parity = compare_v2_exact(build.core_tables, accepted_tables)
            if not parity["passed"]:
                raise RuntimeError(
                    "fsm audit exact parity gate FAILED — no artifact may be saved"
                )
            if parity_report_path is not None:
                write_parity_report(parity, parity_report_path)

            coverage = build_evidence_coverage_report(build)
            fingerprint_sha = canonical_sha256(audit_contract_fingerprint())
            states = _repository_states(root)
            identity = FsmAuditIdentity(
                repositories=states,
                authoritative_source_blob=_AUTHORITATIVE_SOURCE_BLOB,
                accepted_v2_dataset_id=FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
                accepted_v2_manifest_payload_sha256=(
                    FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256
                ),
                resolved_profile_hash=resolved.section_config_hash,
                evaluation_config_hash=resolved.evaluation_config_hash,
                date_allowlist=PERMITTED_DEVELOPMENT_DATES,
                permitted_source_hashes=source_hashes,
                audit_contract_fingerprint_sha256=fingerprint_sha,
            )
            base = Path(output_base or (root / FSM_AUDIT_DATASET_DIR))
            exploration = save_fsm_audit_immutable(
                base_dir=base,
                identity=identity,
                raw_config={
                    "profile_name": profile_name,
                    "permitted_calendar_dates": list(PERMITTED_DEVELOPMENT_DATES),
                    "replay_dates": list(replay_dates),
                    "warmup_dates": list(FROZEN_WARMUP_DATES),
                    "cutoff_utc": DEVELOPMENT_CUTOFF_UTC,
                    "audit_capture_mode": "fsm_audit_v1",
                },
                effective_config={
                    "section": resolved.effective_config,
                    "evaluator": resolved.evaluator_config,
                    "source_access_policy": policy.policy_id,
                    "repository_states": [asdict(s) for s in states],
                },
                audit_tables=build.audit_tables,
                parity_report=parity,
                reconciliation_report=build.reconciliation_report,
                capacity_report=build.capacity_report,
                coverage_report=coverage,
                data_access_audit=discovery.audit.as_dict(),
            )
            artifact_id = fsm_audit_dataset_id_for(identity)
            state = {**state, "status": "ready", "artifact_id": artifact_id}
            _write_json_atomic(state_dir / "state.json", state)
        except Exception as error:
            state = {**state, "status": "failed", "error_code": type(error).__name__}
            _write_json_atomic(state_dir / "state.json", state)
            raise
    audit_dict = discovery.audit.as_dict()
    _assert_zero_protected(audit_dict)
    return FsmAuditPreparationResult(
        artifact_id=artifact_id,
        exploration_dir=exploration,
        parity_report=parity,
        reconciliation_report=build.reconciliation_report,
        coverage_report=coverage,
        capacity_report=build.capacity_report,
        access_audit=audit_dict,
        replay_dates=replay_dates,
    )


def _assert_zero_protected(audit_dict: dict) -> None:
    counters = audit_dict.get("protected_counters", {})
    nonzero = {
        key: value
        for key, value in counters.items()
        if isinstance(value, (int, float)) and value
    }
    if nonzero:
        raise PermissionError(f"protected/sealed access counters nonzero: {nonzero}")


def summarize_result(result: FsmAuditPreparationResult) -> str:
    return json.dumps(
        {
            "artifact_id": result.artifact_id,
            "parity_passed": result.parity_report["passed"],
            "reconciliation_passed": result.reconciliation_report["passed"],
            "replay_days": len(result.replay_dates),
            "audit_rows_by_table": result.coverage_report["audit_rows_by_table"],
        },
        sort_keys=True,
    )
