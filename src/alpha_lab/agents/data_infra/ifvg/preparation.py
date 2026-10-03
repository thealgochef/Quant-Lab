"""External, resumable preparation of immutable IFVG v2/formula-v2 pairs."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import strategy_core
from strategy_core.candles.exchange_calendar import (
    ContextSourceCoverage,
    SourceCoverageStatus,
    SourcePartitionCoverage,
)
from strategy_core.data.databento_parquet import DAY_FILE_PRIORITY
from strategy_core.strategies.ifvg_smc.context_config import (
    FEATURE_FORMULA_VERSION,
    FEATURE_SET_VERSION,
    ContextFeatureConfig,
    build_feature_registry,
)

from .artifact_io import (
    VerifiedIfvgPair,
    load_verified_ifvg_pair,
    load_verified_v2_artifact,
    load_verified_v3_artifact,
)
from .config import V2_DATASET_DIR, V3_DATASET_DIR, IfvgCaptureConfig, IfvgV3CaptureConfig
from .context_experiment_contracts import (
    ArtifactPreparationStatus,
    ProfileCapabilityStatus,
    profile_capability,
)
from .context_schemas import IFVG_CONTEXT_ARROW_REGISTRY_HASH
from .contracts import RecordTable, count_reconciliation
from .data_access import hash_allowlisted_source_files
from .dataset import build_ifvg_v2_capture, build_ifvg_v3_capture
from .development_access import (
    DEVELOPMENT_CUTOFF_UTC,
    FROZEN_WARMUP_DATES,
    PERMITTED_DEVELOPMENT_DATES,
    AccessOperation,
    DevelopmentDataAccess,
    DevelopmentReplayPolicy,
)
from .experiment import run_ifvg_v2_evaluation
from .manifest import (
    DatasetIdentity,
    V3DatasetIdentity,
    canonical_sha256,
    dataset_id_for,
    save_v2_dataset_immutable,
    save_v3_dataset_immutable,
    v3_dataset_id_for,
)
from .profiles import resolve_profile_config
from .reporting import (
    build_context_capacity_report,
    build_context_coverage_report,
    build_context_identity_report,
    build_context_performance_report,
    build_context_reconciliation_report,
    build_context_validity_report,
)
from .verification import (
    _AUTHORITATIVE_SOURCE_BLOB,
    _context_repository_states,
    _repository_states,
    _verify_authoritative_blob,
)

__all__ = [
    "PREPARATION_JOB_ROOT",
    "PAIR_CATALOG_PATH",
    "PreparationCancelledError",
    "PreparationJobState",
    "PreparedIfvgPair",
    "run_resumable_preparation_job",
    "prepare_ifvg_development_pair",
    "prepare_ifvg_development_pair_persisted",
    "read_preparation_state",
]

PREPARATION_JOB_ROOT = Path("data/ifvg_preparation_jobs")
PAIR_CATALOG_PATH = Path("data/ifvg_datasets/context_pair_catalog_v1.json")


class PreparationCancelledError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PreparationJobState:
    profile_name: str
    status: ArtifactPreparationStatus
    completed_dates: tuple[str, ...] = ()
    current_date: str | None = None
    error_code: str | None = None
    v2_artifact_id: str | None = None
    v3_artifact_id: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedIfvgPair:
    """Verified pair and optional A1 review path, outside the pair's manifests."""

    pair: VerifiedIfvgPair
    access_audit: dict[str, Any]
    preparation_state: PreparationJobState
    context_export_path: Path | None = None


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _state_path(root: Path) -> Path:
    return root / "state.json"


def read_preparation_state(root: Path) -> PreparationJobState | None:
    path = _state_path(Path(root))
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = ArtifactPreparationStatus(payload["status"])
        payload["completed_dates"] = tuple(payload.get("completed_dates", ()))
        return PreparationJobState(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("preparation state is unreadable") from error


@contextmanager
def _profile_lock(job_root: Path, profile_name: str) -> Iterator[None]:
    if not profile_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("profile name is unsafe for a preparation lock")
    lock_path = job_root / f"{profile_name}.lock"
    job_root.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("another preparation process holds the profile lock") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        with suppress(FileNotFoundError):
            lock_path.unlink()


def run_resumable_preparation_job(
    *,
    profile_name: str,
    dates: tuple[str, ...],
    process_date: Callable[[str], None],
    finalize: Callable[[], tuple[str, str]],
    job_root: Path = PREPARATION_JOB_ROOT,
) -> PreparationJobState:
    """Checkpoint only at authorized date boundaries and recover after crashes."""

    ordered = tuple(dates)
    if ordered != tuple(sorted(ordered)) or len(ordered) != len(set(ordered)):
        raise ValueError("preparation dates must be unique and chronological")
    root = Path(job_root).resolve() / profile_name
    cancellation = root / "cancel.requested"
    with _profile_lock(Path(job_root).resolve(), profile_name):
        previous = read_preparation_state(root)
        completed = tuple(previous.completed_dates) if previous else ()
        if completed != ordered[: len(completed)]:
            raise ValueError("preparation checkpoint is not a prefix of requested dates")
        state = PreparationJobState(
            profile_name=profile_name,
            status=ArtifactPreparationStatus.PREPARING,
            completed_dates=completed,
        )
        _write_json_atomic(_state_path(root), asdict(state))
        try:
            for day in ordered[len(completed) :]:
                if cancellation.exists():
                    state = replace(
                        state,
                        status=ArtifactPreparationStatus.NOT_PREPARED,
                        current_date=None,
                        error_code="cancelled_at_authorized_date_boundary",
                    )
                    _write_json_atomic(_state_path(root), asdict(state))
                    cancellation.unlink(missing_ok=True)
                    raise PreparationCancelledError(state.error_code)
                state = replace(state, current_date=day)
                _write_json_atomic(_state_path(root), asdict(state))
                process_date(day)
                completed = (*state.completed_dates, day)
                state = replace(state, completed_dates=completed, current_date=None)
                _write_json_atomic(_state_path(root), asdict(state))
            v2_id, v3_id = finalize()
            state = replace(
                state,
                status=ArtifactPreparationStatus.CONTEXT_READY,
                v2_artifact_id=v2_id,
                v3_artifact_id=v3_id,
                error_code=None,
            )
            _write_json_atomic(_state_path(root), asdict(state))
            return state
        except PreparationCancelledError:
            raise
        except Exception as error:
            state = replace(
                state,
                status=ArtifactPreparationStatus.FAILED,
                current_date=None,
                error_code=type(error).__name__,
            )
            _write_json_atomic(_state_path(root), asdict(state))
            raise


def _discover_sources(
    access: DevelopmentDataAccess,
    *,
    data_dir: Path,
    symbol: str,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for day in PERMITTED_DEVELOPMENT_DATES:
        for filename, _schema in DAY_FILE_PRIORITY:
            path = access.construct_path(
                day,
                lambda value, name=filename: Path(data_dir) / symbol / value / name,
            )
            access.audit.record(AccessOperation.EXISTENCE, day)
            if path.exists():
                result[day] = path
                break
    return result


def _coverage(source_dates: set[str]) -> ContextSourceCoverage:
    return ContextSourceCoverage(
        partitions=tuple(
            SourcePartitionCoverage(
                partition_date=date.fromisoformat(day),
                status=(
                    SourceCoverageStatus.AVAILABLE
                    if day in source_dates
                    else SourceCoverageStatus.UNAVAILABLE
                ),
            )
            for day in PERMITTED_DEVELOPMENT_DATES
        ),
        cutoff_ts_utc=datetime.fromisoformat(
            DEVELOPMENT_CUTOFF_UTC.replace("Z", "+00:00")
        ).astimezone(UTC),
    )


def _catalog_pair(
    profile_name: str,
    pair: VerifiedIfvgPair,
    *,
    catalog_path: Path,
) -> None:
    path = Path(catalog_path).resolve()
    try:
        catalog = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError as error:
        raise ValueError("IFVG pair catalog is invalid") from error
    catalog[profile_name] = {
        "preparation_status": ArtifactPreparationStatus.CONTEXT_READY.value,
        "v2_artifact_id": pair.v2.reference.artifact_id,
        "v2_manifest_payload_sha256": pair.v2.reference.manifest_payload_sha256,
        "v3_artifact_id": pair.v3.reference.artifact_id,
        "v3_manifest_payload_sha256": pair.v3.reference.manifest_payload_sha256,
        "feature_formula_version": pair.v3.reference.feature_formula_version,
        "context_arrow_registry_hash": IFVG_CONTEXT_ARROW_REGISTRY_HASH,
    }
    # Formula-v1 remains immutable/readable but cannot be selected for M2.
    catalog.setdefault(
        "legacy_formula_v1",
        {
            "preparation_status": ArtifactPreparationStatus.SUPERSEDED.value,
            "v3_artifact_id": (
                "ee6cfa9eff3c27423281cb3ef34b638f04274cd94993a44adb2cae6b579e71c7"
            ),
            "reason": "displacement_exchange_calendar_source_gap_defect_v1",
        },
    )
    _write_json_atomic(path, dict(sorted(catalog.items())))


def _publish_or_reuse_verified(
    *,
    publisher: Callable[[], Path],
    loader: Callable[[Path, str], Any],
    base_dir: Path,
    dataset_id: str,
    expected_identity: dict[str, Any],
) -> Path:
    try:
        return publisher()
    except FileExistsError as error:
        existing = loader(base_dir, dataset_id)
        if canonical_sha256(existing.manifest.get("identity")) != canonical_sha256(
            expected_identity
        ):
            raise RuntimeError(
                "existing immutable artifact has the expected ID but a different identity"
            ) from error
        return existing.exploration_dir


def _write_preparation_context_export(
    *, report_root: Path, tables: dict, provider, section, tick_size: float,
    run_identity: dict, immutable_roots: tuple[Path, ...],
) -> Path | None:
    """Review material belongs to the preparation job, outside both datasets."""
    if section.menthorq_context_version is None or provider is None:
        return None
    destination = Path(report_root).resolve() / run_identity["v2_artifact_id"]
    if any(destination.is_relative_to(Path(root).resolve()) for root in immutable_roots):
        raise ValueError("preparation review reports must stay outside immutable dataset roots")
    from .menthorq_reporting import write_context_export  # noqa: PLC0415

    return write_context_export(
        destination / "context_export.csv", tables[RecordTable.ENTRY_CANDIDATE],
        tables[RecordTable.ELIGIBLE_DECISION], tables[RecordTable.EXECUTED_TRADE],
        provider=provider, section=section, tick_size=tick_size, run_identity=run_identity,
    )


def prepare_ifvg_development_pair(
    *,
    repo_root: Path,
    profile_name: str = "ifvg_v2_doc_default_fresh_static_1r",
    cached_artifacts_only: bool = True,
    v2_output_base: Path | None = None,
    v3_output_base: Path | None = None,
    catalog_path: Path | None = None,
    progress_fn: Callable[[int, int, str], None] | None = None,
    section_overrides: dict | None = None,
    context_config: ContextFeatureConfig | None = None,
    report_root: Path | None = None,
    preparation_catalog_path: Path | None = None,
) -> PreparedIfvgPair:
    """Prepare the full permitted chain; never discover protected date paths.

    ``section_overrides`` (validated by the section model, capability-gated on
    the BASE profile) and ``context_config`` support variant pairs — e.g. a
    reduced timeframe set, whose context observer must declare a timeframe
    subset of the section's. Variants get their own profile hash, artifact
    identities, and catalog entry; accepted artifacts are untouched.

    Enabled A1 context writes review material under ``report_root/<v2_id>``;
    the default is the profile's preparation-job ``reports`` directory.
    Neither immutable dataset nor its manifest includes that export.
    """

    capability = profile_capability(profile_name)
    if capability.status is not ProfileCapabilityStatus.RUNNABLE:
        raise PermissionError(f"profile is {capability.status.value}: {capability.reason}")
    root = Path(repo_root).resolve()
    _verify_authoritative_blob(root)
    raw_profile: dict = {"profile_name": profile_name}
    if section_overrides:
        raw_profile["section_overrides"] = dict(section_overrides)
    resolved = resolve_profile_config(raw_profile)
    base_cfg = IfvgCaptureConfig()
    cfg = replace(
        base_cfg,
        section=resolved.section,
        session_scheme=base_cfg.session_scheme,
        data_dir=root / "data" / "databento",
        preparation_catalog_paths=(Path(
            preparation_catalog_path or
            (root / PREPARATION_JOB_ROOT / profile_name / "catalog.json")
        ),),
    )
    discovery = DevelopmentDataAccess()
    source_files = _discover_sources(
        discovery,
        data_dir=cfg.data_dir,
        symbol=cfg.symbol,
    )
    if tuple(day for day in FROZEN_WARMUP_DATES if day in source_files) != (
        FROZEN_WARMUP_DATES
    ):
        raise RuntimeError("the frozen ten-date warmup is not fully available")
    replay_dates = tuple(day for day in PERMITTED_DEVELOPMENT_DATES if day in source_files)
    if "2026-06-10" not in replay_dates:
        raise RuntimeError("the development chain does not reach the June 10 cutoff")
    evidence_dates = tuple(day for day in replay_dates if day >= "2026-01-13")
    coverage = _coverage(set(replay_dates))

    v2_policy = DevelopmentReplayPolicy(
        replay_dates,
        development_audit=discovery.audit,
    )
    source_hashes = hash_allowlisted_source_files(v2_policy, source_files)
    v2_capture = build_ifvg_v2_capture(
        replay_dates,
        cfg,
        resolved,
        access_policy=v2_policy,
        cached_artifacts_only=cached_artifacts_only,
        progress_fn=progress_fn,
    )
    for day, bars in v2_capture.bars_by_day.items():
        discovery.record_label_bars(
            day,
            sum(getattr(bar, "timeframe_ticks", None) == 60 for bar in bars),
        )
    v2_audit = v2_policy.audit.as_dict(allowlist=v2_policy.allowlist)
    v2_reports = run_ifvg_v2_evaluation(
        v2_capture.tables,
        resolved_profile=resolved,
        data_access_audit=v2_audit,
        tick_size=cfg.tick_size,
        old_artifact_mutations=0,
        menthorq_provider=getattr(v2_capture, "menthorq_provider", None),
    )
    if not v2_reports["invariant_audit"]["passed"]:
        raise RuntimeError("full-chain v2 invariant audit failed")
    v2_states = _repository_states(root)
    label_source_rows = [
        {
            "source_date": day,
            "bar_id": bar.bar_id,
            "close_ts_utc": bar.availability_ts_utc,
            "open_ticks": bar.open_ticks,
            "high_ticks": bar.high_ticks,
            "low_ticks": bar.low_ticks,
            "close_ticks": bar.close_ticks,
            "volume": bar.volume,
            "trade_count": bar.trade_count,
        }
        for day, bars in sorted(v2_capture.bars_by_day.items())
        for bar in bars
        if bar.timeframe_ticks == 60
        and bar.availability_ts_utc <= datetime.fromisoformat(
            DEVELOPMENT_CUTOFF_UTC.replace("Z", "+00:00")
        )
    ]
    label_source_bars = pd.DataFrame(label_source_rows)
    v2_identity = DatasetIdentity(
        repositories=v2_states,
        authoritative_source_blob=_AUTHORITATIVE_SOURCE_BLOB,
        resolved_profile_hash=resolved.section_config_hash,
        evaluation_config_hash=resolved.evaluation_config_hash,
        date_allowlist=PERMITTED_DEVELOPMENT_DATES,
        permitted_source_hashes=source_hashes,
    )
    v2_base = Path(v2_output_base or (root / V2_DATASET_DIR))
    v2_exploration = _publish_or_reuse_verified(
        publisher=lambda: save_v2_dataset_immutable(
            base_dir=v2_base,
            identity=v2_identity,
            raw_config={
                "profile_name": profile_name,
                "permitted_calendar_dates": list(PERMITTED_DEVELOPMENT_DATES),
                "replay_dates": list(replay_dates),
                "warmup_dates": list(FROZEN_WARMUP_DATES),
                "evidence_dates": list(evidence_dates),
                "cutoff_utc": DEVELOPMENT_CUTOFF_UTC,
            },
            effective_config={
                "section": resolved.effective_config,
                "evaluator": resolved.evaluator_config,
                "source_access_policy": v2_policy.policy_id,
                "source_coverage_hash": coverage.content_hash,
                "repository_states": [asdict(state) for state in v2_states],
            },
            tables=v2_capture.tables,
            candidate_report=v2_reports["candidate_report"],
            decision_report=v2_reports["decision_report"],
            executed_trade_report=v2_reports["executed_trade_report"],
            invariant_audit=v2_reports["invariant_audit"],
            count_reconciliation_report=count_reconciliation(v2_capture.tables),
            data_access_audit=discovery.audit.as_dict(),
            label_source_bars=label_source_bars,
        ),
        loader=load_verified_v2_artifact,
        base_dir=v2_base,
        dataset_id=dataset_id_for(v2_identity),
        expected_identity=v2_identity.payload(),
    )
    v2_manifest = json.loads(
        (v2_exploration / "manifest.json").read_text(encoding="utf-8")
    )
    v2_id = str(v2_manifest["dataset_id"])
    v2_manifest_hash = str(v2_manifest["manifest_payload_sha256"])

    v3_cfg = IfvgV3CaptureConfig(
        core=cfg,
        accepted_v2_dataset_id=v2_id,
        accepted_v2_manifest_sha256=v2_manifest_hash,
        **({"context": context_config} if context_config is not None else {}),
    )
    v3_policy = DevelopmentReplayPolicy(
        replay_dates,
        development_audit=discovery.audit,
    )
    context_states = _context_repository_states(root)
    strategy_state = next(
        state for state in context_states if state.name == "Strategy-Core"
    )
    v3_capture = build_ifvg_v3_capture(
        replay_dates,
        v3_cfg,
        resolved,
        strategy_core_commit=strategy_state.head,
        strategy_core_source_tree_hash=strategy_state.source_tree_hash,
        access_policy=v3_policy,
        cached_artifacts_only=cached_artifacts_only,
        accepted_v2_tables=v2_capture.tables,
        context_source_coverage=coverage,
        measure_performance=True,
        progress_fn=progress_fn,
    )
    for frame in v3_capture.context_tables.values():
        if "source_date" not in frame:
            continue
        for day, count in frame["source_date"].dropna().astype(str).value_counts().items():
            discovery.record_context_rows(day, int(count))
    validity = build_context_validity_report(v3_capture.context_tables)
    context_coverage = build_context_coverage_report(v3_capture.context_tables)
    reconciliation = build_context_reconciliation_report(
        v3_capture.context_tables,
        core_tables=v3_capture.core_parity_tables,
        baseline_reconciliation=v3_capture.baseline_reconciliation,
    )
    capacity = build_context_capacity_report(
        v3_capture.context_tables,
        **v3_capture.capacity_metrics,
    )
    identity_report = build_context_identity_report(
        v3_capture.context_tables,
        expected={
            "feature_set_version": FEATURE_SET_VERSION,
            "feature_formula_version": FEATURE_FORMULA_VERSION,
            "feature_schema_hash": v3_cfg.feature_schema_hash,
            "context_config_hash": v3_cfg.context_config_hash,
        },
    )
    performance = build_context_performance_report(
        **{
            key: value
            for key, value in v3_capture.performance_measurements.items()
            if key != "measurement_policy"
        }
    )
    performance["measurement_policy"] = v3_capture.performance_measurements[
        "measurement_policy"
    ]
    gates = {
        "validity": validity,
        "reconciliation": reconciliation,
        "capacity": capacity,
        "identity": identity_report,
        "performance": performance,
    }
    failed_gates = tuple(
        name for name, report in gates.items() if not report["passed"]
    )
    if failed_gates:
        raise RuntimeError(
            "full-chain formula-v2 verification gate failed: "
            + ", ".join(failed_gates)
        )
    discovery.assert_safe()
    v3_identity = V3DatasetIdentity(
        repositories=context_states,
        authoritative_source_blob=_AUTHORITATIVE_SOURCE_BLOB,
        accepted_v2_dataset_id=v2_id,
        accepted_v2_manifest_payload_sha256=v2_manifest_hash,
        resolved_profile_hash=resolved.section_config_hash,
        feature_set_version=FEATURE_SET_VERSION,
        feature_formula_version=FEATURE_FORMULA_VERSION,
        feature_schema_hash=v3_cfg.feature_schema_hash,
        context_config_hash=v3_cfg.context_config_hash,
        normalized_timeframes=v3_cfg.context.normalized_timeframes,
        anchor_status_240m="experimental_q40_open",
        date_allowlist=PERMITTED_DEVELOPMENT_DATES,
        warmup_dates=FROZEN_WARMUP_DATES,
        evidence_dates=evidence_dates,
        permitted_source_hashes=source_hashes,
    )
    v3_base = Path(v3_output_base or (root / V3_DATASET_DIR))
    v3_exploration = _publish_or_reuse_verified(
        publisher=lambda: save_v3_dataset_immutable(
            base_dir=v3_base,
            identity=v3_identity,
            raw_config={
                "mode": "full_permitted_context_measurement",
                "cutoff_utc": DEVELOPMENT_CUTOFF_UTC,
                "accepted_v2_dataset_id": v2_id,
                "accepted_v2_manifest_payload_sha256": v2_manifest_hash,
                "permitted_calendar_dates": list(PERMITTED_DEVELOPMENT_DATES),
                "warmup_dates": list(FROZEN_WARMUP_DATES),
                "evidence_dates": list(evidence_dates),
            },
            effective_config={
                "context": asdict(v3_cfg.context),
                "feature_schema_hash": v3_cfg.feature_schema_hash,
                "context_config_hash": v3_cfg.context_config_hash,
                "source_coverage_hash": coverage.content_hash,
                "ordered_feature_registry": [
                    asdict(item) for item in build_feature_registry(v3_cfg.context)
                ],
                "source_access_policy": v3_policy.policy_id,
                "strategy_core_import": str(Path(strategy_core.__file__).resolve()),
            },
            context_tables=v3_capture.context_tables,
            validity_report=validity,
            coverage_report=context_coverage,
            reconciliation_report=reconciliation,
            capacity_report=capacity,
            identity_report=identity_report,
            performance_report=performance,
            data_access_audit=discovery.audit.as_dict(),
            diagnostics_out=v3_capture.diagnostic_timings,
        ),
        loader=load_verified_v3_artifact,
        base_dir=v3_base,
        dataset_id=v3_dataset_id_for(v3_identity),
        expected_identity=v3_identity.payload(),
    )
    v3_id = v3_exploration.parent.name
    pair = load_verified_ifvg_pair(
        v2_root=Path(v2_output_base or (root / V2_DATASET_DIR)),
        v2_artifact_id=v2_id,
        v3_root=Path(v3_output_base or (root / V3_DATASET_DIR)),
        v3_artifact_id=v3_id,
    )
    context_export_path = None
    provider = getattr(v2_capture, "menthorq_provider", None)
    if resolved.section.menthorq_context_version is not None and provider is not None:
        review_root = (
            Path(report_root) if report_root is not None else
            root / PREPARATION_JOB_ROOT / resolved.section.profile_name / "reports"
        )
        if not review_root.is_absolute():
            review_root = root / review_root
        context_export_path = _write_preparation_context_export(
            report_root=review_root, tables=v2_capture.tables, provider=provider,
            section=resolved.section, tick_size=cfg.tick_size,
            run_identity={
                "v2_artifact_id": v2_id,
                "v2_manifest_payload_sha256": pair.v2.reference.manifest_payload_sha256,
                "v3_artifact_id": v3_id,
                "v3_manifest_payload_sha256": pair.v3.reference.manifest_payload_sha256,
                "section_config_hash": resolved.section_config_hash,
                "evaluation_config_hash": resolved.evaluation_config_hash,
                "strategy_core_commit": strategy_state.head,
                "strategy_core_source_tree_hash": strategy_state.source_tree_hash,
            },
            immutable_roots=(v2_base, v3_base),
        )
    _catalog_pair(
        # the STAMPED name (variants may override section.profile_name so the
        # verifier dropdown distinguishes them; identical to profile_name for
        # the base profile).
        resolved.section.profile_name,
        pair,
        catalog_path=Path(catalog_path or (root / PAIR_CATALOG_PATH)),
    )
    state = PreparationJobState(
        profile_name=profile_name,
        status=ArtifactPreparationStatus.CONTEXT_READY,
        completed_dates=replay_dates,
        v2_artifact_id=v2_id,
        v3_artifact_id=v3_id,
    )
    return PreparedIfvgPair(
        pair=pair,
        access_audit=discovery.audit.as_dict(),
        preparation_state=state,
        context_export_path=context_export_path,
    )


def prepare_ifvg_development_pair_persisted(
    *,
    repo_root: Path,
    profile_name: str = "ifvg_v2_doc_default_fresh_static_1r",
    cached_artifacts_only: bool = True,
    job_root: Path = PREPARATION_JOB_ROOT,
    progress_fn: Callable[[int, int, str], None] | None = None,
    section_overrides: dict | None = None,
    context_config: ContextFeatureConfig | None = None,
    job_label: str | None = None,
    report_root: Path | None = None,
) -> PreparedIfvgPair:
    """Run production preparation under a profile lock with durable boundaries.

    The underlying replay remains one sequential seed chain.  Checkpoints record
    each completed authorized-date callback for crash diagnosis; a restarted job
    deterministically rebuilds the chain from cached authorized day artifacts and
    refuses any non-identical immutable publication. Variant runs (with
    ``section_overrides``) must pass a distinct ``job_label`` so their lock and
    state directory never collide with the base profile's. Enabled A1 exports
    default to that selected job's ``reports/<v2_id>/context_export.csv``;
    ``report_root`` overrides only the review-material destination.
    """

    root = Path(repo_root).resolve()
    jobs = Path(job_root)
    if not jobs.is_absolute():
        jobs = root / jobs
    jobs = jobs.resolve()
    lock_name = job_label or profile_name
    profile_root = jobs / lock_name
    cancellation = profile_root / "cancel.requested"
    with _profile_lock(jobs, lock_name):
        profile_root.mkdir(parents=True, exist_ok=True)
        state = PreparationJobState(
            profile_name=profile_name,
            status=ArtifactPreparationStatus.PREPARING,
        )
        _write_json_atomic(_state_path(profile_root), asdict(state))
        pass_index = 0
        previous_completed = 0
        pass_completed_dates: list[str] = []

        def persisted_progress(completed: int, total: int, day: str) -> None:
            nonlocal pass_index, previous_completed, state, pass_completed_dates
            if completed <= previous_completed:
                pass_index += 1
                pass_completed_dates = []
            if completed != len(pass_completed_dates) + 1:
                raise RuntimeError("preparation progress is not a sequential date chain")
            pass_completed_dates.append(day)
            previous_completed = completed
            state = replace(
                state,
                completed_dates=tuple(pass_completed_dates),
                current_date=None,
                error_code=f"replay_pass_{pass_index + 1}_of_2",
            )
            _write_json_atomic(_state_path(profile_root), asdict(state))
            if progress_fn is not None:
                progress_fn(completed, total, day)
            if cancellation.exists():
                state = replace(
                    state,
                    status=ArtifactPreparationStatus.NOT_PREPARED,
                    error_code="cancelled_at_authorized_date_boundary",
                )
                _write_json_atomic(_state_path(profile_root), asdict(state))
                cancellation.unlink(missing_ok=True)
                raise PreparationCancelledError(state.error_code)

        try:
            if cancellation.exists():
                cancellation.unlink(missing_ok=True)
                state = replace(
                    state,
                    status=ArtifactPreparationStatus.NOT_PREPARED,
                    error_code="cancelled_at_authorized_date_boundary",
                )
                _write_json_atomic(_state_path(profile_root), asdict(state))
                raise PreparationCancelledError(state.error_code)
            prepared = prepare_ifvg_development_pair(
                repo_root=root,
                profile_name=profile_name,
                cached_artifacts_only=cached_artifacts_only,
                progress_fn=persisted_progress,
                section_overrides=section_overrides,
                context_config=context_config,
                preparation_catalog_path=profile_root / "catalog.json",
                **({"report_root": report_root} if report_root is not None else
                   {"report_root": profile_root / "reports"}
                   if (section_overrides or {}).get("menthorq_context_version") is not None
                   else {}),
            )
            state = replace(
                prepared.preparation_state,
                completed_dates=prepared.preparation_state.completed_dates,
                error_code=None,
            )
            _write_json_atomic(_state_path(profile_root), asdict(state))
            return replace(prepared, preparation_state=state)
        except PreparationCancelledError:
            raise
        except Exception as error:
            state = replace(
                state,
                status=ArtifactPreparationStatus.FAILED,
                current_date=None,
                error_code=type(error).__name__,
            )
            _write_json_atomic(_state_path(profile_root), asdict(state))
            raise
