"""One fixed, nonsealed IFVG v2 repair verification replay."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import strategy_core
from strategy_core.strategies.ifvg_smc.context_config import (
    FEATURE_FORMULA_VERSION,
    FEATURE_SET_VERSION,
    build_feature_registry,
)

from .config import (
    V2_DATASET_DIR,
    V3_DATASET_DIR,
    IfvgCaptureConfig,
    IfvgV3CaptureConfig,
    legacy_ifvg_capture_config,
)
from .contracts import count_reconciliation
from .data_access import (
    EXPLORATION_DATE_ALLOWLIST,
    ExplorationDataPolicy,
    discover_allowlisted_source_files,
    hash_allowlisted_source_files,
    require_fixed_exploration_allowlist,
)
from .dataset import build_ifvg_v2_capture, build_ifvg_v3_capture
from .experiment import run_ifvg_v2_evaluation
from .manifest import (
    DatasetIdentity,
    V3DatasetIdentity,
    read_repository_state,
    save_v2_dataset_immutable,
    save_v3_dataset_immutable,
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

__all__ = ["run_context_verification", "run_repair_verification"]

_AUTHORITATIVE_SOURCE_BLOB = "9b5f6f163ae060030c5695dbc0aede94e0ebebcd"


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _strategy_core_repository_root(repo_root: Path) -> Path:
    """Resolve source provenance for editable or immutable VCS installations."""

    imported_root = Path(strategy_core.__file__).resolve().parents[2]
    if (imported_root / ".git").exists():
        return imported_root

    sibling_candidates = (
        Path(repo_root).resolve().parent / "Strategy-Core",
        Path(repo_root).resolve().parent / "Strategy-core",
    )
    strategy_root = next(
        (candidate for candidate in sibling_candidates if (candidate / ".git").exists()),
        None,
    )
    if strategy_root is None:
        raise RuntimeError("Strategy-Core source checkout is unavailable for provenance")
    try:
        direct_url_text = distribution("strategy-core").read_text("direct_url.json")
        direct_url = json.loads(direct_url_text or "{}")
    except (PackageNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError("installed Strategy-Core provenance is unreadable") from error
    installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
    if not isinstance(installed_commit, str) or len(installed_commit) != 40:
        raise RuntimeError("installed Strategy-Core lacks immutable VCS commit provenance")
    if _git_head(strategy_root) != installed_commit:
        raise RuntimeError("installed Strategy-Core pin differs from source checkout HEAD")
    return strategy_root


def _metadata_snapshot(paths: list[Path]) -> dict[str, tuple[int, int]]:
    return {
        str(path.resolve()): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths
        if path.exists() and path.is_file()
    }


def _legacy_artifact_paths(repo_root: Path, dates: tuple[str, ...]) -> list[Path]:
    legacy = legacy_ifvg_capture_config(
        data_dir=repo_root / "data" / "databento"
    )
    paths: list[Path] = []
    for day in dates:
        paths.extend(
            (
                legacy.bars_path(day),
                legacy.levels_path(day),
                legacy.capture_path(day),
                legacy.seed_path(day),
            )
        )
    paths.extend(
        (
            repo_root / "IFVG_FUNNEL.md",
            repo_root / "IFVG_LABELS.md",
            repo_root / "IFVG_SEARCH_LEDGER.json",
            repo_root / "ifvg_funnel_2a40b18e0b273ee0.json",
        )
    )
    return paths


def _repository_states(repo_root: Path) -> tuple:
    documents = repo_root.parent
    strategy_root = _strategy_core_repository_root(repo_root)
    trade_lab_root = documents / "Trade-Lab"
    return (
        read_repository_state(
            "Claude-Quant-Lab",
            repo_root,
            source_paths=(
                "src/alpha_lab/agents/data_infra/ifvg",
                "scripts/run_ifvg_capture.py",
                "scripts/run_ifvg_repair_verification.py",
                "scripts/ifvg_artifact_warmer.py",
                "scripts/ifvg_recapture_job.py",
                "scripts/ifvg_lab_tab.py",
                "scripts/ifvg_lab_charts.py",
                "tests/agents",
                "ARCHITECTURE.md",
                "docs/ML_TRAINING_WORKBENCH.md",
                "docs/pipeline_state.yaml",
            ),
        ),
        read_repository_state(
            "Strategy-Core",
            strategy_root,
            source_paths=("src/strategy_core", "tests"),
        ),
        read_repository_state(
            "Trade-Lab",
            trade_lab_root,
            source_paths=(
                "backend/src",
                "backend/tests",
                "docs/ifvg/IFVG_IMPLEMENTATION_REPAIR_PLAN.md",
                "docs/ifvg/IFVG_IMPLEMENTATION_REPAIR_MATRIX.md",
                "docs/ifvg/IFVG_OPEN_DECISIONS.md",
                "docs/ifvg/IFVG_MIGRATION_V1_TO_V2.md",
                "docs/ifvg/IFVG_STRATEGY_IMPROVEMENT_ROADMAP.md",
                "docs/strategy-core-alignment-runbook.md",
            ),
        ),
    )


def _verify_authoritative_blob(repo_root: Path) -> None:
    strategy_path = repo_root.parent / "Trade-Lab" / "docs" / "ifvg-strat.md"
    blob = subprocess.run(
        ["git", "hash-object", str(strategy_path)],
        cwd=strategy_path.parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if blob != _AUTHORITATIVE_SOURCE_BLOB:
        raise RuntimeError(
            "authoritative IFVG source blob drifted: "
            f"{blob} != {_AUTHORITATIVE_SOURCE_BLOB}"
        )


def run_repair_verification(
    *,
    repo_root: Path,
    cached_artifacts_only: bool = False,
    output_base: Path | None = None,
    progress_fn=None,
) -> Path:
    """Run the fixed document-default replay and atomically save a v2 dataset."""
    repo_root = Path(repo_root).resolve()
    _verify_authoritative_blob(repo_root)
    resolved = resolve_profile_config(
        {"profile_name": "ifvg_v2_doc_default_fresh_static_1r"}
    )
    if not resolved.runnable:
        raise RuntimeError("document-default fresh profile unexpectedly non-runnable")
    cfg = replace(
        IfvgCaptureConfig(),
        section=resolved.section,
        session_scheme=IfvgCaptureConfig().session_scheme,
        data_dir=repo_root / "data" / "databento",
    )
    policy = ExplorationDataPolicy()
    require_fixed_exploration_allowlist(policy)
    explicit_dates = tuple(sorted(EXPLORATION_DATE_ALLOWLIST))
    source_files = discover_allowlisted_source_files(
        policy,
        data_dir=cfg.data_dir,
        symbol=cfg.symbol,
        dates=explicit_dates,
    )
    missing_sources = sorted(set(explicit_dates) - set(source_files))
    if missing_sources:
        raise RuntimeError(
            "fixed replay requires one permitted source file for every "
            f"allowlisted date; missing {missing_sources}"
        )
    replay_dates = tuple(sorted(source_files))
    warmup = tuple(day for day in replay_dates if day <= "2026-01-12")
    evidence = tuple(day for day in replay_dates if day >= "2026-01-13")
    if len(warmup) != 10:
        raise RuntimeError(
            "fixed replay requires the first ten available chain days through "
            f"January 12; found {len(warmup)}"
        )
    if not evidence:
        raise RuntimeError("fixed replay has no January 13-30 evidence days")
    source_hashes = hash_allowlisted_source_files(policy, source_files)

    old_paths = _legacy_artifact_paths(repo_root, explicit_dates)
    before = _metadata_snapshot(old_paths)
    capture = build_ifvg_v2_capture(
        replay_dates,
        cfg,
        resolved,
        access_policy=policy,
        cached_artifacts_only=cached_artifacts_only,
        measure_performance=True,
        progress_fn=progress_fn,
    )
    after = _metadata_snapshot(old_paths)
    old_mutations = sum(
        1 for path in set(before) | set(after) if before.get(path) != after.get(path)
    )

    reports = run_ifvg_v2_evaluation(
        capture.tables,
        resolved_profile=resolved,
        data_access_audit=policy.audit_dict(),
        tick_size=cfg.tick_size,
        old_artifact_mutations=old_mutations,
    )
    invariant = reports["invariant_audit"]
    if not invariant["passed"]:
        raise RuntimeError(
            "IFVG v2 invariant audit failed: "
            + json.dumps(invariant["violations"], sort_keys=True)
        )

    states = _repository_states(repo_root)
    identity = DatasetIdentity(
        repositories=states,
        authoritative_source_blob=_AUTHORITATIVE_SOURCE_BLOB,
        resolved_profile_hash=resolved.section_config_hash,
        evaluation_config_hash=resolved.evaluation_config_hash,
        date_allowlist=explicit_dates,
        permitted_source_hashes=source_hashes,
    )
    raw_config = {
        **resolved.raw_ui_config,
        "explicit_date_allowlist": list(explicit_dates),
        "replay_dates": list(replay_dates),
        "warmup_dates": list(warmup),
        "invariant_evidence_dates": list(evidence),
    }
    effective_config = {
        "section": resolved.effective_config,
        "evaluator": resolved.evaluator_config,
        "diagnostic_only": resolved.diagnostics,
        "section_config_hash": resolved.section_config_hash,
        "evaluation_config_hash": resolved.evaluation_config_hash,
        "qualification_mode": resolved.qualification_mode,
        "runnable": resolved.runnable,
        "strategy_core_import": str(Path(strategy_core.__file__).resolve()),
        "repository_states": [state.__dict__ for state in states],
        "source_access_policy": "explicit_allowlist_before_path_v1",
    }
    return save_v2_dataset_immutable(
        base_dir=Path(output_base or (repo_root / V2_DATASET_DIR)),
        identity=identity,
        raw_config=raw_config,
        effective_config=effective_config,
        tables=capture.tables,
        candidate_report=reports["candidate_report"],
        decision_report=reports["decision_report"],
        executed_trade_report=reports["executed_trade_report"],
        invariant_audit=invariant,
        count_reconciliation_report=count_reconciliation(capture.tables),
        data_access_audit=policy.audit_dict(),
    )


def _context_repository_states(repo_root: Path) -> tuple:
    documents = repo_root.parent
    strategy_root = _strategy_core_repository_root(repo_root)
    trade_lab_root = documents / "Trade-Lab"
    return (
        read_repository_state(
            "Claude-Quant-Lab",
            repo_root,
            source_paths=(
                "src/alpha_lab/agents/data_infra/ifvg",
                "scripts/run_ifvg_context_capture.py",
                "tests/agents/test_ifvg_v3_context_contracts.py",
                "tests/agents/test_ifvg_v3_context_reconciliation.py",
                "tests/agents/test_ifvg_v3_context_access.py",
                "tests/agents/test_ifvg_v3_context_manifest.py",
                "ARCHITECTURE.md",
            ),
        ),
        read_repository_state(
            "Strategy-Core",
            strategy_root,
            source_paths=("src/strategy_core", "tests"),
        ),
        read_repository_state(
            "Trade-Lab",
            trade_lab_root,
            source_paths=(
                "backend/src",
                "backend/tests",
                "frontend/src",
                "docs/ifvg",
                "docs/architecture.md",
                "docs/market-data-contract.md",
                "docs/strategy-core-alignment-runbook.md",
            ),
        ),
    )


def run_context_verification(
    *,
    repo_root: Path,
    cached_artifacts_only: bool = False,
    output_base: Path | None = None,
    progress_fn=None,
    diagnostics_out: dict[str, object] | None = None,
) -> Path:
    """Derive the fixed nonsealed context artifact without evaluation or training."""

    repo_root = Path(repo_root).resolve()
    _verify_authoritative_blob(repo_root)
    resolved = resolve_profile_config(
        {"profile_name": "ifvg_v2_doc_default_fresh_static_1r"}
    )
    core = replace(
        IfvgCaptureConfig(),
        section=resolved.section,
        session_scheme=IfvgCaptureConfig().session_scheme,
        data_dir=repo_root / "data" / "databento",
    )
    cfg = IfvgV3CaptureConfig(core=core)
    policy = ExplorationDataPolicy()
    require_fixed_exploration_allowlist(policy)
    explicit_dates = tuple(sorted(EXPLORATION_DATE_ALLOWLIST))
    source_files = discover_allowlisted_source_files(
        policy,
        data_dir=core.data_dir,
        symbol=core.symbol,
        dates=explicit_dates,
    )
    missing = sorted(set(explicit_dates) - set(source_files))
    if missing:
        raise RuntimeError(
            "fixed v3 replay requires one source for every allowlisted date; "
            f"missing {missing}"
        )
    replay_dates = tuple(sorted(source_files))
    warmup = tuple(day for day in replay_dates if day <= "2026-01-12")
    evidence = tuple(day for day in replay_dates if day >= "2026-01-13")
    if len(warmup) != 10 or not evidence:
        raise RuntimeError("fixed v3 warmup/evidence partition drifted")
    source_hashes = hash_allowlisted_source_files(policy, source_files)

    states = _context_repository_states(repo_root)
    strategy_state = next(state for state in states if state.name == "Strategy-Core")
    capture = build_ifvg_v3_capture(
        replay_dates,
        cfg,
        resolved,
        strategy_core_commit=strategy_state.head,
        strategy_core_source_tree_hash=strategy_state.source_tree_hash,
        access_policy=policy,
        cached_artifacts_only=cached_artifacts_only,
        measure_performance=True,
        progress_fn=progress_fn,
    )
    tables = capture.context_tables
    validity = build_context_validity_report(tables)
    coverage = build_context_coverage_report(tables)
    reconciliation = build_context_reconciliation_report(
        tables,
        core_tables=capture.core_parity_tables,
        baseline_reconciliation=capture.baseline_reconciliation,
    )
    capacity = build_context_capacity_report(tables, **capture.capacity_metrics)
    identity_report = build_context_identity_report(
        tables,
        expected={
            "feature_set_version": FEATURE_SET_VERSION,
            "feature_formula_version": FEATURE_FORMULA_VERSION,
            "feature_schema_hash": cfg.feature_schema_hash,
            "context_config_hash": cfg.context_config_hash,
        },
    )
    performance = build_context_performance_report(
        **{
            key: value
            for key, value in capture.performance_measurements.items()
            if key != "measurement_policy"
        }
    )
    performance["measurement_policy"] = capture.performance_measurements[
        "measurement_policy"
    ]
    gates = {
        "validity": validity["passed"],
        "reconciliation": reconciliation["passed"],
        "capacity": capacity["passed"],
        "identity": identity_report["passed"],
        "performance": performance["passed"],
    }
    if not all(gates.values()):
        raise RuntimeError(
            "IFVG v3 verification gates failed: "
            + json.dumps(
                {
                    "gates": gates,
                    "capacity_violations": capacity["violations"],
                    "capacity_report": capacity,
                    "performance_violations": performance["violations"],
                    "performance_report": performance,
                },
                sort_keys=True,
            )
        )
    policy.assert_zero_forbidden_access()

    identity = V3DatasetIdentity(
        repositories=states,
        authoritative_source_blob=_AUTHORITATIVE_SOURCE_BLOB,
        accepted_v2_dataset_id=cfg.accepted_v2_dataset_id,
        accepted_v2_manifest_payload_sha256=cfg.accepted_v2_manifest_sha256,
        resolved_profile_hash=resolved.section_config_hash,
        feature_set_version=FEATURE_SET_VERSION,
        feature_formula_version=FEATURE_FORMULA_VERSION,
        feature_schema_hash=cfg.feature_schema_hash,
        context_config_hash=cfg.context_config_hash,
        normalized_timeframes=cfg.context.normalized_timeframes,
        anchor_status_240m="experimental_q40_open",
        date_allowlist=explicit_dates,
        warmup_dates=warmup,
        evidence_dates=evidence,
        permitted_source_hashes=source_hashes,
    )
    raw_config = {
        "mode": "measurement_only",
        "accepted_v2_dataset_id": cfg.accepted_v2_dataset_id,
        "accepted_v2_manifest_payload_sha256": cfg.accepted_v2_manifest_sha256,
        "explicit_date_allowlist": list(explicit_dates),
        "warmup_dates": list(warmup),
        "evidence_dates": list(evidence),
    }
    effective_config = {
        "context": asdict(cfg.context),
        "feature_schema_hash": cfg.feature_schema_hash,
        "context_config_hash": cfg.context_config_hash,
        "ordered_feature_registry": [
            asdict(item) for item in build_feature_registry(cfg.context)
        ],
        "resolved_v2_profile_hash": resolved.section_config_hash,
        "repository_states": [asdict(state) for state in states],
        "source_access_policy": "explicit_allowlist_before_path_v1",
    }
    output = save_v3_dataset_immutable(
        base_dir=Path(output_base or (repo_root / V3_DATASET_DIR)),
        identity=identity,
        raw_config=raw_config,
        effective_config=effective_config,
        context_tables=tables,
        validity_report=validity,
        coverage_report=coverage,
        reconciliation_report=reconciliation,
        capacity_report=capacity,
        identity_report=identity_report,
        performance_report=performance,
        data_access_audit=policy.audit_dict(),
        diagnostics_out=capture.diagnostic_timings,
    )
    if diagnostics_out is not None:
        diagnostics_out.update(capture.diagnostic_timings)
    return output
