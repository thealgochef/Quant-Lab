"""Atomic immutable stores for IFVG context views and experiment runs."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .context_experiment_contracts import (
    ContextFeatureTier,
    IfvgContextExperimentConfig,
    IfvgContextExperimentResult,
    IfvgContextRunReconciliation,
    context_run_identity,
)
from .context_feature_view import CandidateFeatureView
from .manifest import canonical_sha256, file_sha256

__all__ = [
    "CONTEXT_VIEW_STORE",
    "CONTEXT_RUN_STORE",
    "CONTEXT_RUN_CATALOG",
    "ImmutableContextStoreError",
    "StoredContextRun",
    "save_candidate_feature_view",
    "load_candidate_feature_view_frame",
    "save_context_experiment_run",
    "load_context_experiment_run",
    "update_context_run_catalog",
    "list_context_run_catalog",
    "reconcile_context_runs",
]

CONTEXT_VIEW_STORE = Path("data/ifvg_datasets/context_views/v1")
CONTEXT_RUN_STORE = Path("data/ifvg_experiments/context_v1")
CONTEXT_RUN_CATALOG = Path("data/ifvg_experiments/context_v1_catalog.json")
_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ImmutableContextStoreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StoredContextRun:
    directory: Path
    manifest: dict[str, Any]
    config: IfvgContextExperimentConfig
    result: IfvgContextExperimentResult
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _atomic_directory(base: Path, identity: str) -> tuple[Path, Path]:
    if not _FULL_SHA256.fullmatch(identity):
        raise ImmutableContextStoreError("immutable identity must be a full SHA-256")
    base = Path(base).resolve()
    destination = base / identity
    if destination.exists():
        raise FileExistsError("immutable context object already exists")
    base.mkdir(parents=True, exist_ok=True)
    temporary = base / f".{identity}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    return temporary, destination


def _publish(temporary: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("immutable context object appeared concurrently")
    os.replace(temporary, destination)


def save_candidate_feature_view(
    view: CandidateFeatureView,
    *,
    base_dir: Path = CONTEXT_VIEW_STORE,
) -> Path:
    temporary, destination = _atomic_directory(base_dir, view.view_id)
    try:
        data_path = temporary / "candidate_feature_view.parquet"
        view.frame.to_parquet(data_path, index=False)
        manifest_core = {
            "manifest_schema_version": 1,
            "view_id": view.view_id,
            "artifact_pair_hash": view.artifact_pair_hash,
            "feature_registry_hash": view.feature_registry_hash,
            "m3_status": view.m3_status,
            "candidate_ids_sha256": canonical_sha256(
                view.frame["candidate_id"].astype(str).tolist()
            ),
            "context_capture_ids_sha256": canonical_sha256(
                view.frame["context_capture_id"].astype(str).tolist()
            ),
            "artifact": {
                "path": data_path.name,
                "sha256": file_sha256(data_path),
                "bytes": data_path.stat().st_size,
                "rows": len(view.frame),
                "columns": list(view.frame.columns),
            },
            "tier_features": {
                tier.value: list(features)
                for tier, features in view.tier_features.items()
            },
        }
        _write_json(
            temporary / "manifest.json",
            {
                **manifest_core,
                "manifest_payload_sha256": canonical_sha256(manifest_core),
            },
        )
        _publish(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def _manifest(directory: Path, identity_key: str, identity: str) -> dict[str, Any]:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImmutableContextStoreError("immutable object manifest is unreadable") from error
    if manifest.get(identity_key) != identity:
        raise ImmutableContextStoreError("immutable object manifest identity mismatch")
    core = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    if canonical_sha256(core) != manifest.get("manifest_payload_sha256"):
        raise ImmutableContextStoreError("immutable object manifest hash mismatch")
    return manifest


def _content_directory(base_dir: Path, identity: str) -> Path:
    if not _FULL_SHA256.fullmatch(identity):
        raise ImmutableContextStoreError("immutable identity must be a full SHA-256")
    root = Path(base_dir).resolve()
    directory = (root / identity).resolve()
    try:
        directory.relative_to(root)
    except ValueError as error:
        raise ImmutableContextStoreError("immutable object escaped its store") from error
    return directory


def _child_artifact(directory: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ImmutableContextStoreError("immutable artifact path is invalid")
    path = (directory / relative).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError as error:
        raise ImmutableContextStoreError("immutable artifact escaped its object") from error
    return path


def load_candidate_feature_view_frame(
    view_id: str,
    *,
    base_dir: Path = CONTEXT_VIEW_STORE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = _content_directory(base_dir, view_id)
    manifest = _manifest(directory, "view_id", view_id)
    artifact = manifest["artifact"]
    if artifact.get("path") != "candidate_feature_view.parquet":
        raise ImmutableContextStoreError("candidate feature view path is not canonical")
    path = _child_artifact(directory, artifact["path"])
    if (
        not path.is_file()
        or path.stat().st_size != artifact["bytes"]
        or file_sha256(path) != artifact["sha256"]
    ):
        raise ImmutableContextStoreError("candidate feature view artifact was modified")
    frame = pd.read_parquet(path)
    if len(frame) != artifact["rows"] or list(frame.columns) != artifact["columns"]:
        raise ImmutableContextStoreError("candidate feature view shape was modified")
    if canonical_sha256(frame["candidate_id"].astype(str).tolist()) != manifest[
        "candidate_ids_sha256"
    ]:
        raise ImmutableContextStoreError("candidate feature view membership was modified")
    if canonical_sha256(frame["context_capture_id"].astype(str).tolist()) != manifest[
        "context_capture_ids_sha256"
    ]:
        raise ImmutableContextStoreError("candidate feature links were modified")
    computed_view_id = canonical_sha256(
        {
            "artifact_pair_hash": manifest["artifact_pair_hash"],
            "feature_registry_hash": manifest["feature_registry_hash"],
            "candidate_ids": frame["candidate_id"].astype(str).tolist(),
            "candidate_link_ids": frame["context_capture_id"].astype(str).tolist(),
        }
    )
    if computed_view_id != view_id:
        raise ImmutableContextStoreError("candidate feature view content ID mismatch")
    return frame, manifest


def save_context_experiment_run(
    config: IfvgContextExperimentConfig,
    result: IfvgContextExperimentResult,
    *,
    predictions: pd.DataFrame | None = None,
    feature_importance: pd.DataFrame | None = None,
    base_dir: Path = CONTEXT_RUN_STORE,
) -> Path:
    if result.config_hash != config.identity:
        raise ImmutableContextStoreError("result/config identity mismatch")
    prediction_frame = (
        pd.DataFrame()
        if predictions is None
        else predictions.reset_index(drop=True).copy()
    )
    observed_oos_ids = (
        tuple(prediction_frame["oos_row_id"].astype(str))
        if not prediction_frame.empty and "oos_row_id" in prediction_frame
        else ()
    )
    if observed_oos_ids != result.oos_row_ids:
        raise ImmutableContextStoreError("result OOS IDs do not match prediction stream")
    computed_run_id = context_run_identity(
        config_hash=result.config_hash,
        view_id=result.view_id,
        label_derivation_id=result.label_derivation_id,
        folds=[fold.model_dump(mode="json") for fold in result.folds],
        model_protocol_hash=result.model_protocol_hash,
        predictions=prediction_frame.to_dict("records"),
        status=result.status,
    )
    if computed_run_id != result.run_id:
        raise ImmutableContextStoreError("result run content ID mismatch")
    temporary, destination = _atomic_directory(base_dir, result.run_id)
    try:
        artifacts: list[dict[str, Any]] = []
        for filename, payload in (
            ("config.json", config.model_dump(mode="json")),
            ("result.json", result.model_dump(mode="json")),
        ):
            path = temporary / filename
            _write_json(path, payload)
            artifacts.append(
                {
                    "path": filename,
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        for filename, frame in (
            (
                "oos_predictions.parquet",
                prediction_frame if predictions is not None else None,
            ),
            ("feature_importance.parquet", feature_importance),
        ):
            if frame is None:
                continue
            path = temporary / filename
            frame.to_parquet(path, index=False)
            artifacts.append(
                {
                    "path": filename,
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                    "rows": len(frame),
                    "columns": list(frame.columns),
                }
            )
        manifest_core = {
            "manifest_schema_version": 1,
            "run_id": result.run_id,
            "config_hash": result.config_hash,
            "view_id": result.view_id,
            "immutable": True,
            "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        }
        _write_json(
            temporary / "manifest.json",
            {
                **manifest_core,
                "manifest_payload_sha256": canonical_sha256(manifest_core),
            },
        )
        _publish(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def load_context_experiment_run(
    run_id: str,
    *,
    base_dir: Path = CONTEXT_RUN_STORE,
) -> StoredContextRun:
    directory = _content_directory(base_dir, run_id)
    manifest = _manifest(directory, "run_id", run_id)
    entries = {item["path"]: item for item in manifest["artifacts"]}
    if len(entries) != len(manifest["artifacts"]):
        raise ImmutableContextStoreError("immutable run has duplicate artifact paths")
    if not {"config.json", "result.json"}.issubset(entries):
        raise ImmutableContextStoreError("immutable run is missing its contracts")
    tables: dict[str, pd.DataFrame] = {}
    for filename, entry in entries.items():
        path = _child_artifact(directory, filename)
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or file_sha256(path) != entry["sha256"]
        ):
            raise ImmutableContextStoreError("immutable run artifact was modified")
        if "rows" in entry:
            frame = pd.read_parquet(path)
            if len(frame) != entry["rows"] or list(frame.columns) != entry["columns"]:
                raise ImmutableContextStoreError("immutable run table shape was modified")
            tables[filename] = frame
    try:
        config = IfvgContextExperimentConfig.model_validate_json(
            (directory / "config.json").read_text(encoding="utf-8")
        )
        result = IfvgContextExperimentResult.model_validate_json(
            (directory / "result.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ImmutableContextStoreError("immutable run contract is invalid") from error
    if config.identity != manifest["config_hash"] or result.run_id != run_id:
        raise ImmutableContextStoreError("immutable run contract identity mismatch")
    predictions = tables.get("oos_predictions.parquet", pd.DataFrame())
    observed_oos_ids = (
        tuple(predictions["oos_row_id"].astype(str))
        if not predictions.empty and "oos_row_id" in predictions
        else ()
    )
    if observed_oos_ids != result.oos_row_ids:
        raise ImmutableContextStoreError("immutable run OOS stream mismatch")
    computed_run_id = context_run_identity(
        config_hash=result.config_hash,
        view_id=result.view_id,
        label_derivation_id=result.label_derivation_id,
        folds=[fold.model_dump(mode="json") for fold in result.folds],
        model_protocol_hash=result.model_protocol_hash,
        predictions=predictions.to_dict("records"),
        status=result.status,
    )
    if computed_run_id != run_id:
        raise ImmutableContextStoreError("immutable run content ID mismatch")
    return StoredContextRun(
        directory=directory,
        manifest=manifest,
        config=config,
        result=result,
        predictions=predictions,
        feature_importance=tables.get("feature_importance.parquet", pd.DataFrame()),
    )


def update_context_run_catalog(
    run_id: str,
    *,
    display_name: str | None = None,
    notes: str | None = None,
    catalog_path: Path = CONTEXT_RUN_CATALOG,
) -> None:
    if not _FULL_SHA256.fullmatch(run_id):
        raise ValueError("catalog run ID must be a full SHA-256")
    path = Path(catalog_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        catalog = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except json.JSONDecodeError as error:
        raise ImmutableContextStoreError("context run catalog is invalid") from error
    existing = catalog.get(run_id, {})
    if not isinstance(existing, dict):
        raise ImmutableContextStoreError("context run catalog entry is invalid")
    catalog[run_id] = {
        "display_name": (
            display_name if display_name is not None else existing.get("display_name")
        ),
        "notes": notes if notes is not None else existing.get("notes"),
    }
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    _write_json(temporary, dict(sorted(catalog.items())))
    os.replace(temporary, path)


def list_context_run_catalog(
    *,
    catalog_path: Path = CONTEXT_RUN_CATALOG,
) -> list[dict[str, Any]]:
    path = Path(catalog_path).resolve()
    if not path.exists():
        return []
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImmutableContextStoreError("context run catalog is unreadable") from error
    rows = []
    for run_id, metadata in sorted(catalog.items()):
        if not _FULL_SHA256.fullmatch(run_id) or not isinstance(metadata, dict):
            raise ImmutableContextStoreError("context run catalog entry is invalid")
        rows.append({"run_id": run_id, **metadata})
    return rows


def reconcile_context_runs(
    left: StoredContextRun,
    right: StoredContextRun,
) -> IfvgContextRunReconciliation:
    left_config = left.config.model_dump(mode="json")
    right_config = right.config.model_dump(mode="json")
    pair_match = left_config["dataset"]["artifact_pair"] == right_config["dataset"][
        "artifact_pair"
    ]
    cohort_match = (
        left_config["dataset"] == right_config["dataset"]
        and left_config["observation_filters"] == right_config["observation_filters"]
    )
    label_match = left_config["label"] == right_config["label"]
    folds_match = [fold.model_dump(mode="json") for fold in left.result.folds] == [
        fold.model_dump(mode="json") for fold in right.result.folds
    ]
    model_match = (
        left_config["model_protocol_id"] == right_config["model_protocol_id"]
        and left.result.model_protocol_hash == right.result.model_protocol_hash
    )
    bootstrap_match = (
        left_config["bootstrap_repetitions"] == right_config["bootstrap_repetitions"]
        and left_config["random_seed"] == right_config["random_seed"]
    )
    oos_match = left.result.oos_row_ids == right.result.oos_row_ids
    tier_only = (
        left_config["feature_tier"] != right_config["feature_tier"]
        and all(
            left_config[key] == right_config[key]
            for key in left_config
            if key != "feature_tier"
        )
        and ContextFeatureTier(left_config["feature_tier"]) != ContextFeatureTier(
            right_config["feature_tier"]
        )
    )
    checks = {
        "artifact_pair": pair_match,
        "cohort": cohort_match,
        "label": label_match,
        "folds": folds_match,
        "model_protocol": model_match,
        "calibration_protocol": model_match,
        "bootstrap_protocol": bootstrap_match,
        "oos_row_ids": oos_match,
    }
    compatible = all(checks.values()) and (
        left_config["feature_tier"] == right_config["feature_tier"] or tier_only
    )
    return IfvgContextRunReconciliation(
        artifact_pair_match=pair_match,
        cohort_match=cohort_match,
        label_match=label_match,
        folds_match=folds_match,
        model_protocol_match=model_match,
        calibration_protocol_match=model_match,
        bootstrap_protocol_match=bootstrap_match,
        oos_row_ids_match=oos_match,
        registered_tier_delta_only=tier_only,
        compatible_for_metric_delta=compatible,
        differing_fields=tuple(sorted(key for key, matched in checks.items() if not matched)),
    )
