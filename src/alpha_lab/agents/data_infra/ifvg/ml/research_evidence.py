"""Immutable research inputs and reloadable supervised results.

These stores supplement the historical summary envelopes; historical readers
are unchanged. Inputs are committed before fitting, including zero-fold runs.
Completed results carry every arm/rung's OOS rows, fitted model, preprocessing
and exact prediction inputs. Reload verifies all file hashes before deserializing
models, then reproduces the recorded probabilities without fitting.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..context_experiment_contracts import canonical_contract_sha256
from .comparison_rows import label_artifact_content_id
from .fold_set_artifact import fold_set_id

INPUT_STORE = "research_model_inputs"
RUN_STORE = "research_model_runs"
SCHEMA_VERSION = "supervised_research_evidence_v1"
RUNNERS = {"ladder", "mbp1", "regime_feature", "regime_cohort"}


def _cell(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str | int | bool):
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported research identity value {type(value).__name__}")


def frame_content_hash(frame: pd.DataFrame) -> str:
    """Order-invariant hash over ALL columns/rows, preserving repeated rows."""
    if frame.columns.duplicated().any():
        raise ValueError("research evidence refuses duplicate frame columns")
    columns = sorted(str(c) for c in frame.columns)
    rows = [[_cell(v) for v in row] for row in frame[columns].itertuples(index=False, name=None)]
    rows.sort(key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
    return canonical_contract_sha256({"columns": columns, "rows": rows})


def _identity(value: Any, folds) -> Any:
    if isinstance(value, pd.DataFrame):
        return {"frame_content_hash": frame_content_hash(value)}
    if hasattr(value, "frame_for_fold") and hasattr(value, "artifact_id"):
        return {
            "artifact_id": str(value.artifact_id),
            "features": list(value.feature_names),
            "categoricals": list(value.categorical_features),
            "fold_values": {
                str(fold.fold_index): frame_content_hash(value.frame_for_fold(fold.fold_index))
                for fold in folds.folds
            },
        }
    if hasattr(value, "model_dump"):
        return _identity(value.model_dump(mode="json"), folds)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _identity(getattr(value, item.name), folds) for item in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(k): _identity(v, folds) for k, v in sorted(value.items(), key=lambda p: str(p[0]))
        }
    if isinstance(value, tuple | list):
        return [_identity(v, folds) for v in value]
    return _cell(value)


def _source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    paths = [
        *root.glob("*.py"),
        *root.joinpath("ml").glob("*.py"),
        *root.joinpath("features").glob("*.py"),
    ]
    return {
        "source_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): _sha(path) for path in sorted(paths)
        },
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("numpy", "pandas", "scikit-learn", "catboost", "pyarrow")
        },
    }


def _request_payload(view, labeled_candidates, folds, *, runner, run_kwargs):
    if runner not in RUNNERS:
        raise ValueError(f"unknown research runner {runner!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "runner": runner,
        "view_id": view.view_id,
        "artifact_pair_hash": view.artifact_pair_hash,
        "feature_registry_hash": view.feature_registry_hash,
        "cohort_values_hash": frame_content_hash(view.frame),
        "label_values_hash": label_artifact_content_id(None, labeled_candidates),
        "all_label_columns_hash": frame_content_hash(labeled_candidates),
        "fold_set_id": fold_set_id(folds),
        "fold_definitions": [fold.model_dump(mode="json") for fold in folds.folds],
        "fold_assignment_hash": frame_content_hash(folds.assignment),
        "run_kwargs": _identity(dict(run_kwargs or {}), folds),
        "implementation": _source_identity(),
    }


def research_run_request_id(
    view, labeled_candidates, folds, *, runner="ladder", run_kwargs=None
) -> str:
    return canonical_contract_sha256(
        _request_payload(view, labeled_candidates, folds, runner=runner, run_kwargs=run_kwargs)
    )


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory(root, store, request_id) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", str(request_id)):
        raise ValueError("research request id must be a lowercase SHA256")
    return Path(root).resolve() / store / request_id


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


def _json(value):
    if isinstance(value, Mapping):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_json(v) for v in value]
    return _cell(value)


def _manifest(directory, request_id, *, kind):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = {
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
            }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "kind": kind,
        "files": files,
    }
    _write_json(
        directory / "manifest.json",
        {**payload, "manifest_sha256": canonical_contract_sha256(payload)},
    )


def _verified(directory: Path, request_id: str, *, kind: str) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256")
    if canonical_contract_sha256(manifest) != claimed:
        raise ValueError("research manifest checksum mismatch")
    if (manifest.get("schema_version"), manifest.get("request_id"), manifest.get("kind")) != (
        SCHEMA_VERSION,
        request_id,
        kind,
    ):
        raise ValueError("research manifest identity mismatch")
    for name, entry in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or path == directory.resolve():
            raise ValueError("research manifest path escapes artifact directory")
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or _sha(path) != entry["sha256"]
        ):
            raise ValueError(f"research artifact checksum mismatch: {name}")
    actual = {
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "manifest.json"
    }
    expected = set(manifest["files"])
    if actual != expected:
        raise ValueError("research artifact file membership mismatch")
    return manifest


def _temporary(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".pending-", dir=destination.parent))


def _cleanup(temporary, parent):
    # Only remove this writer's verified temporary child, never a computed
    # external path or any existing artifact.
    resolved = temporary.resolve()
    if resolved.parent != parent.resolve() or not resolved.name.startswith(".pending-"):
        raise ValueError("unsafe research temporary-directory cleanup")
    if resolved.exists():
        shutil.rmtree(resolved)


def save_research_inputs(
    root, view, labeled_candidates, folds, *, runner="ladder", run_kwargs=None
) -> str:
    payload = _request_payload(
        view, labeled_candidates, folds, runner=runner, run_kwargs=run_kwargs
    )
    request_id = canonical_contract_sha256(payload)
    destination = _directory(root, INPUT_STORE, request_id)
    if destination.exists():
        _load_inputs(root, request_id)
        return request_id
    temporary = _temporary(destination)
    try:
        _write_json(temporary / "request.json", payload)
        view.frame.to_parquet(temporary / "candidate_features.parquet", index=False)
        labeled_candidates.to_parquet(temporary / "labels.parquet", index=False)
        folds.assignment.to_parquet(temporary / "fold_assignment.parquet", index=False)
        _write_json(temporary / "fold_definitions.json", payload["fold_definitions"])
        for name, value in sorted(dict(run_kwargs or {}).items()):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
                raise ValueError("research runner arguments require valid parameter names")
            if isinstance(value, pd.DataFrame):
                value.to_parquet(temporary / f"input_{name}.parquet", index=False)
            elif hasattr(value, "frame_for_fold"):
                for fold in folds.folds:
                    value.frame_for_fold(fold.fold_index).to_parquet(
                        temporary / f"input_{name}_fold_{fold.fold_index}.parquet", index=False
                    )
        _manifest(temporary, request_id, kind="inputs")
        os.replace(temporary, destination)
    finally:
        _cleanup(temporary, destination.parent)
    return request_id


def _load_inputs(root, request_id, *, include_tables=False):
    directory = _directory(root, INPUT_STORE, request_id)
    _verified(directory, request_id, kind="inputs")
    payload = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if canonical_contract_sha256(payload) != request_id:
        raise ValueError("research request payload does not reproduce its id")
    tables = {}
    for filename, key in (
        ("candidate_features.parquet", "cohort_values_hash"),
        ("labels.parquet", "all_label_columns_hash"),
        ("fold_assignment.parquet", "fold_assignment_hash"),
    ):
        frame = pd.read_parquet(directory / filename)
        if frame_content_hash(frame) != payload[key]:
            raise ValueError(f"research input content mismatch: {filename}")
        tables[filename.removesuffix(".parquet")] = frame
    return (
        {"request": payload, **tables, "fold_definitions": payload["fold_definitions"]}
        if include_tables
        else payload
    )


def load_research_inputs(root, request_id: str) -> dict:
    """Read verified prefit tables without loading models or fitting anything.

    The evidence remains available when fitting has not run or has failed.
    Missing/corrupt input artifacts raise rather than masquerading as empty
    scientific datasets.
    """
    return _load_inputs(root, request_id, include_tables=True)


def _pack_ladder(directory: Path, run) -> dict:
    from .logistic_model import persist_logistic_fit
    from .model_protocols import LOGISTIC_PROTOCOL_ID, PREVALENCE_PROTOCOL_ID

    directory.mkdir(parents=True)
    metadata = {
        name: _json(getattr(run, name))
        for name in (
            "ladder_id",
            "view_id",
            "tier",
            "calibration_policy_id",
            "parity",
            "paired_deltas",
            "feature_source",
            "label_identity_source",
            "research_identity",
        )
    }
    metadata["rungs"] = []
    for index, rung in enumerate(run.rungs):
        rung_dir = directory / f"rung_{index}"
        rung_dir.mkdir()
        rung.predictions.to_parquet(rung_dir / "oos_predictions.parquet", index=False)
        rung.feature_importance.to_parquet(rung_dir / "feature_importance.parquet", index=False)
        protocol = rung.resolved_protocol
        item = {
            "protocol_id": rung.protocol_id,
            "resolved_protocol_hash": rung.resolved_protocol_hash,
            "fold_reports": _json(rung.fold_reports),
            "prediction_report": _json(rung.prediction_report),
            "protocol_type": type(protocol).__name__ if protocol else None,
            "protocol": _json(asdict(protocol)) if protocol else None,
            "models": [],
        }
        expected = {int(r["fold_index"]) for r in rung.fold_reports if r["valid"]}
        if rung.protocol_id != PREVALENCE_PROTOCOL_ID and set(rung.fitted_models) != expected:
            raise ValueError("durable research run lacks a fitted model for every valid fold")
        for fold_index, model in sorted(rung.fitted_models.items()):
            if fold_index not in rung.prediction_inputs:
                raise ValueError("durable model lacks exact prediction inputs")
            fold_dir = rung_dir / f"fold_{fold_index}"
            fold_dir.mkdir()
            rung.prediction_inputs[fold_index].to_parquet(fold_dir / "prediction_inputs.parquet")
            if rung.protocol_id == LOGISTIC_PROTOCOL_ID:
                persist_logistic_fit(
                    fold_dir, pipeline=model, protocol=protocol, fold_index=fold_index
                )
                kind = "logistic"
            else:
                model.save_model(str(fold_dir / "model.cbm"))
                kind = "catboost"
            item["models"].append({"fold_index": fold_index, "kind": kind})
        metadata["rungs"].append(item)
    return metadata


def _protocol_types():
    from ..context_model import ResolvedContextModelProtocol
    from .catboost_bundle_model import ResolvedCatBoostBundleProtocol
    from .logistic_model import ResolvedLogisticProtocol

    return {
        cls.__name__: cls
        for cls in (
            ResolvedContextModelProtocol,
            ResolvedCatBoostBundleProtocol,
            ResolvedLogisticProtocol,
        )
    }


def _unpack_ladder(directory, metadata):
    from catboost import CatBoostClassifier

    from .logistic_model import reload_logistic_fit
    from .model_protocols import PREVALENCE_PROTOCOL_ID
    from .supervised_ladder import LadderRung, SupervisedLadderRun

    rungs = []
    for index, item in enumerate(metadata["rungs"]):
        item = dict(item)
        rung_dir = directory / f"rung_{index}"
        predictions = pd.read_parquet(rung_dir / "oos_predictions.parquet")
        protocol_type = item.pop("protocol_type")
        protocol_payload = item.pop("protocol")
        if protocol_payload:
            for key in ("ordered_features", "categorical_features"):
                if key in protocol_payload:
                    protocol_payload[key] = tuple(protocol_payload[key])
        protocol = _protocol_types()[protocol_type](**protocol_payload) if protocol_type else None
        if protocol is not None and protocol.resolved_hash != item["resolved_protocol_hash"]:
            raise ValueError("research model protocol hash mismatch")
        models = {}
        prediction_inputs = {}
        model_refs = item.pop("models")
        expected_folds = (
            set()
            if item["protocol_id"] == PREVALENCE_PROTOCOL_ID
            else {int(report["fold_index"]) for report in item["fold_reports"] if report["valid"]}
        )
        model_folds = [int(ref["fold_index"]) for ref in model_refs]
        if len(set(model_folds)) != len(model_folds) or set(model_folds) != expected_folds:
            raise ValueError("stored research model coverage disagrees with valid folds")
        for model_ref in model_refs:
            fold_index = int(model_ref["fold_index"])
            fold_dir = rung_dir / f"fold_{fold_index}"
            if model_ref["kind"] == "logistic":
                model, _manifest_payload = reload_logistic_fit(fold_dir)
            elif model_ref["kind"] == "catboost":
                model = CatBoostClassifier()
                model.load_model(str(fold_dir / "model.cbm"))
            else:
                raise ValueError("unsupported stored research model kind")
            test_x = pd.read_parquet(fold_dir / "prediction_inputs.parquet")
            expected = predictions.loc[predictions["fold_index"].eq(fold_index)]
            if list(test_x.index.astype(str)) != list(expected["candidate_id"].astype(str)):
                raise ValueError("stored prediction inputs disagree with OOS candidate identity")
            observed = model.predict_proba(test_x)[:, 1]
            if not np.allclose(
                observed, expected["probability"].to_numpy(), rtol=1e-12, atol=1e-12
            ):
                raise ValueError("reloaded model does not reproduce its stored OOS probabilities")
            models[fold_index] = model
            prediction_inputs[fold_index] = test_x
        item["fold_reports"] = tuple(item["fold_reports"])
        rungs.append(
            LadderRung(
                **item,
                predictions=predictions,
                feature_importance=pd.read_parquet(rung_dir / "feature_importance.parquet"),
                fitted_models=models,
                resolved_protocol=protocol,
                prediction_inputs=prediction_inputs,
            )
        )
    return SupervisedLadderRun(
        **{k: v for k, v in metadata.items() if k != "rungs"}, rungs=tuple(rungs)
    )


def _run_classes():
    from .controlled_feature_study import ControlledFeatureStudyEnvelope, ControlledFeatureStudyRun
    from .regime_cohort_model import RegimeCohortModelStudyEnvelope, RegimeCohortModelStudyRun
    from .regime_controlled_study import RegimeControlledStudyEnvelope, RegimeControlledStudyRun
    from .supervised_ladder import SupervisedLadderRun

    return {
        "ladder": (SupervisedLadderRun, None),
        "mbp1": (ControlledFeatureStudyRun, ControlledFeatureStudyEnvelope),
        "regime_feature": (RegimeControlledStudyRun, RegimeControlledStudyEnvelope),
        "regime_cohort": (RegimeCohortModelStudyRun, RegimeCohortModelStudyEnvelope),
    }


def _ladders(run, runner):
    if runner == "ladder":
        return [("ladder", run)]
    if runner == "regime_cohort":
        return [
            ("pooled", run.pooled),
            *[
                (f"specialized_{i}", ladder)
                for i, (_, ladder) in enumerate(sorted(run.specialized.items()))
            ],
        ]
    return [("baseline", run.baseline), ("challenger", run.challenger)]


def _assert_result_matches_request(payload, run):
    runner = payload["runner"]
    if not isinstance(run, _run_classes()[runner][0]):
        raise TypeError("research result type disagrees with requested runner")
    ladders = _ladders(run, runner)
    for name, ladder in ladders:
        if name.startswith("specialized_"):
            continue  # specialized strata intentionally contain subsets
        if ladder.research_identity.get("label_values_hash") != payload["label_values_hash"]:
            raise ValueError("research result labels disagree with persisted inputs")
        if ladder.research_identity.get("fold_set_hash") != payload["fold_set_id"]:
            raise ValueError("research result folds disagree with persisted inputs")
    options = payload["run_kwargs"]
    if runner == "ladder":
        if run.research_identity.get("cohort_values_hash") != payload["cohort_values_hash"]:
            raise ValueError("research result cohort disagrees with persisted inputs")
        for key, actual in (
            ("tier", run.tier),
            ("protocols", [rung.protocol_id for rung in run.rungs]),
            ("calibration_policy_id", run.calibration_policy_id),
            ("bundle_features", run.feature_source.get("feature_names")),
            ("bundle_ref", run.feature_source.get("resolved_feature_bundle_id")),
            ("bundle_evidence_ref", run.feature_source.get("evidence_ref")),
            ("fold_schedule_id", run.research_identity.get("fold_schedule_id")),
            ("label_artifact_id", run.research_identity.get("label_artifact_id")),
        ):
            if options.get(key) is not None and options[key] != actual:
                raise ValueError(f"research result disagrees with requested {key}")
    else:
        from .comparison_rows import assert_persistable_label_proof  # noqa: PLC0415

        assert_persistable_label_proof(run.label_identity_proof, runner=runner)
        study = run.envelope.payload
        for option, field_name in (
            ("challenger_bundle_key", "challenger_bundle_key"),
            ("headline_protocol_id", "model_protocol_id"),
            ("label_artifact_id", "label_artifact_id"),
            ("fold_schedule_id", "fold_schedule_id"),
            ("calibration_policy_id", "calibration_policy_id"),
            ("bundle_key", "bundle_key"),
        ):
            if (
                options.get(option) is not None
                and hasattr(study, field_name)
                and options[option] != getattr(study, field_name)
            ):
                raise ValueError(f"research study disagrees with requested {option}")
        if "fold_features" in options and options["fold_features"]["artifact_id"] != (
            study.regime_fold_feature_artifact_id
        ):
            raise ValueError("research study used different fold feature evidence")
        if runner == "mbp1" and "mbp1_feature_artifact" in options:
            expected_artifact = options["mbp1_feature_artifact"]["mbp1_feature_artifact_id"]
            if expected_artifact != study.mbp1_feature_artifact_id:
                raise ValueError("research study used different MBP-1 feature evidence")
        if options.get("protocols") is not None and options["protocols"] != [
            rung.protocol_id for rung in ladders[0][1].rungs
        ]:
            raise ValueError("research study used different supervised protocols")


def save_research_run(root, request_id: str, run) -> Path:
    payload = _load_inputs(root, request_id)
    _assert_result_matches_request(payload, run)
    runner = payload["runner"]
    ladders = _ladders(run, runner)
    destination = _directory(root, RUN_STORE, request_id)
    if destination.exists():
        existing = load_research_run(root, request_id)
        if [r.ladder_id for _, r in _ladders(existing, runner)] != [
            r.ladder_id for _, r in ladders
        ]:
            raise ValueError("research request already has a different completed result")
        return destination
    temporary = _temporary(destination)
    try:
        metadata = {
            "runner": runner,
            "ladders": {name: _pack_ladder(temporary / name, ladder) for name, ladder in ladders},
        }
        if runner != "ladder":
            metadata["envelope"] = run.envelope.model_dump(mode="json")
            metadata["label_identity_proof"] = run.label_identity_proof
            (temporary / "study_detail.json").write_bytes(run.detail_bytes)
        if runner == "regime_cohort":
            metadata["specialized_keys"] = [list(key) for key in sorted(run.specialized)]
            run.specialized_predictions.to_parquet(
                temporary / "specialized_oos.parquet", index=False
            )
        _write_json(temporary / "run.json", metadata)
        _manifest(temporary, request_id, kind="run")
        _verified(temporary, request_id, kind="run")
        for name, item in metadata["ladders"].items():
            _unpack_ladder(temporary / name, item)
        os.replace(temporary, destination)
    finally:
        _cleanup(temporary, destination.parent)
    # The newly published bytes must reproduce predictions before success.
    load_research_run(root, request_id)
    return destination


def load_research_run(root, request_id: str):
    """Return an exact completed run, or None if absent; corrupt data refuses.

    There is no fitting or 'latest' lookup on this path. All model/JSON/table
    checksums are verified before any joblib deserialization takes place.
    """
    directory = _directory(root, RUN_STORE, request_id)
    if not directory.exists():
        return None
    payload = _load_inputs(root, request_id)
    _verified(directory, request_id, kind="run")
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    runner = metadata["runner"]
    if runner != payload["runner"]:
        raise ValueError("stored result runner does not match input request")
    if runner == "ladder":
        expected_arms = {"ladder"}
    elif runner == "regime_cohort":
        expected_arms = {
            "pooled",
            *(f"specialized_{index}" for index in range(len(metadata["specialized_keys"]))),
        }
    else:
        expected_arms = {"baseline", "challenger"}
    if set(metadata["ladders"]) != expected_arms:
        raise ValueError("stored research arm membership is invalid")
    ladders = {
        name: _unpack_ladder(directory / name, item) for name, item in metadata["ladders"].items()
    }
    if runner == "ladder":
        result = ladders["ladder"]
        _assert_result_matches_request(payload, result)
        return result
    run_type, envelope_type = _run_classes()[runner]
    kwargs = {
        "envelope": envelope_type.model_validate(metadata["envelope"]),
        "detail_bytes": (directory / "study_detail.json").read_bytes(),
        "label_identity_proof": metadata["label_identity_proof"],
    }
    if hashlib.sha256(kwargs["detail_bytes"]).hexdigest() != kwargs["envelope"].detail_sha256:
        raise ValueError("study detail does not match historical envelope")
    if runner == "regime_cohort":
        result = run_type(
            **kwargs,
            pooled=ladders["pooled"],
            specialized={
                tuple(key): ladders[f"specialized_{i}"]
                for i, key in enumerate(metadata["specialized_keys"])
            },
            specialized_predictions=pd.read_parquet(directory / "specialized_oos.parquet"),
        )
    else:
        result = run_type(**kwargs, baseline=ladders["baseline"], challenger=ladders["challenger"])
    _assert_result_matches_request(payload, result)
    return result
