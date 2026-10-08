"""Additive, verified read-only publication of external funded results.

The ordinary research catalog stores small pointers. Immutable reporting views
stay outside the repository and bind the original result, plan and approval
manifests. A view is never execution authority or a replacement engine.
"""

from __future__ import annotations

import csv
import io
import json
import os
import uuid
import zipfile
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256, file_sha256

SCHEMA = "external_funded_result_catalog_v1"
VIEW_VERSION = "funded_saved_result_read_view_v2"
REPO_ROOT = Path(__file__).resolve().parents[7]


def catalog_path(store_root: Path | None = None) -> Path:
    root = (
        Path(store_root) if store_root is not None else (REPO_ROOT / "data/ifvg_datasets/search/v1")
    )
    return root / "catalog/external_funded_results.json"


def _write(path: Path, payload: Any, *, immutable: bool = False) -> None:
    data = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    if immutable and path.exists():
        if path.read_bytes() != data:
            raise ValueError("existing reporting publication has different immutable bytes")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _key(value: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("a published result requires a full SHA-256 identity")
    return value


def _manifest_binding(root: Path, family: str, identity: str) -> dict[str, Any]:
    folder = root / family / _key(identity)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    core = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
    if (
        manifest.get("manifest_payload_sha256") != canonical_sha256(core)
        or manifest.get("envelope_id") != identity
        or manifest.get("store_name") != family
    ):
        raise ValueError("published source manifest failed integrity or identity verification")
    artifacts = {row["path"]: row for row in manifest["artifacts"]}
    for name, receipt in artifacts.items():
        if Path(name).name != name:
            raise ValueError("published source manifest contains a nonlocal path")
        path = folder / name
        if not path.is_file() or path.stat().st_size != int(receipt["bytes"]):
            raise ValueError("published source artifact is inaccessible or has changed size")
        if (
            _verified_digest(str(path), path.stat().st_size, path.stat().st_mtime_ns)
            != receipt["sha256"]
        ):
            raise ValueError("published source artifact failed integrity verification")
    return {
        "family": family,
        "identity": identity,
        "manifest_sha256": file_sha256(manifest_path),
        "envelope_sha256": artifacts["envelope.json"]["sha256"],
        "artifacts": artifacts,
    }


@lru_cache(maxsize=128)
def _verified_digest(path: str, _size: int, _modified_ns: int) -> str:
    return file_sha256(Path(path))


def _load_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema": SCHEMA, "studies": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or not isinstance(payload.get("studies"), dict):
        raise ValueError("external funded-result catalog has an unsupported schema")
    return payload


def register_external_result(
    *,
    catalog_store_root: Path,
    external_store_root: Path,
    result_id: str,
    reporting_root: Path,
    study_key: str,
    display_name: str,
    version_label: str,
    qualification: str = "",
    predecessor_result_id: str | None = None,
    preferred: bool = True,
) -> dict[str, Any]:
    """Verify, create one immutable read view, and register an additive version.

    This API cannot approve a plan, change a pin or start a worker. Both the
    original economic records and existing catalog entries are retained.
    """
    from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
    from alpha_lab.propsim.funded.comparison_plan import (
        APPROVAL_STORE,
        PLAN_STORE,
        RESULT_STORE,
        FundedComparisonApprovalEnvelope,
        FundedComparisonResultEnvelope,
    )
    from alpha_lab.propsim.funded.comparison_runner import load_comparison_result, load_plan

    root = Path(external_store_root).resolve(strict=True)
    reporting = Path(reporting_root).resolve()
    if root.is_relative_to(REPO_ROOT) or reporting.is_relative_to(REPO_ROOT):
        raise ValueError("external result and reporting views must remain outside the repository")
    result_id = _key(result_id)
    envelope = load_verified_envelope(root, RESULT_STORE, result_id, FundedComparisonResultEnvelope)
    plan_id = envelope.payload.funded_comparison_plan_id
    approval_id = envelope.payload.funded_comparison_approval_id
    if not approval_id or not envelope.payload.validation_passed:
        raise ValueError("only verified approved completed financial results may be published")
    approval = load_verified_envelope(
        root, APPROVAL_STORE, approval_id, FundedComparisonApprovalEnvelope
    )
    if approval.payload.funded_comparison_plan_id != plan_id:
        raise ValueError("published approval binds another plan")
    plan = load_plan(root, plan_id)  # readable schema only, no execution source check
    result = load_comparison_result(root, result_id)
    if (result.get("validation") or {}).get("passed") is not True or result.get(
        "funded_comparison_plan_id"
    ) != plan_id:
        raise ValueError("result plan or economic validation is inconsistent")
    if getattr(plan, "plan_schema", "") == "ifsm_mffu_context_64_batch_plan_v1":
        from alpha_lab.propsim.funded.mffu_batch_review import _assert_result

        _assert_result(plan, plan_id, result)
        from alpha_lab.propsim.funded.mffu_batch_analysis import analyze_mffu_batch
        from alpha_lab.propsim.funded.mffu_batch_plan import HANDOFF_ROOT, _sha_bytes

        with zipfile.ZipFile(plan.handoff_zip) as archive:
            data = archive.read(HANDOFF_ROOT + "COMPARISON_PAIRS.csv")
        if _sha_bytes(data) != plan.source_member_sha256["COMPARISON_PAIRS.csv"]:
            raise ValueError("registered reporting comparisons bind different declared pairs")
        result["mffu_analysis"] = analyze_mffu_batch(
            result,
            variants=plan.variants,
            comparison_pairs=list(csv.DictReader(io.StringIO(data.decode("utf-8-sig")))),
            evaluation_dates=plan.source.evaluation_dates,
            economic_result_id=result_id,
        )
    bindings = [
        _manifest_binding(root, family, identity)
        for family, identity in (
            (RESULT_STORE, result_id),
            (PLAN_STORE, plan_id),
            (APPROVAL_STORE, approval_id),
        )
    ]
    # All saved financial tables, configuration rows and comparisons survive.
    # The expensive decision receipts are separately lazy-loaded by the gamma view.
    batch = dict(result.get("mffu_batch") or {})
    decisions = batch.pop("decision_context", None)
    if decisions is not None:
        batch["decision_context_read_policy"] = "registered_lazy_receipts"
        batch["decision_context_count"] = len(decisions)
    result["mffu_batch"] = batch
    view = {
        "schema": VIEW_VERSION,
        "result_id": result_id,
        "plan_id": plan_id,
        "source_bindings": bindings,
        "result": result,
    }
    view_path = reporting / result_id / f"{VIEW_VERSION}.json"
    # Compact serialization keeps cold-open I/O bounded; no saved financial
    # values are omitted or changed by this serialization choice.
    data = json.dumps(view, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    if view_path.exists():
        if view_path.read_bytes() != data:
            raise ValueError("existing reporting publication has different immutable bytes")
    else:
        view_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = view_path.with_name(f".{view_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, view_path)
    namespace_path = root / "STORE_NAMESPACE.json"
    namespace = (
        json.loads(namespace_path.read_text(encoding="utf-8")) if namespace_path.exists() else None
    )
    pointer = {
        "schema": "external_funded_result_pointer_v1",
        "result_id": result_id,
        "plan_id": plan_id,
        "approval_id": approval_id,
        "external_store_root": str(root),
        "readable_plan_schema": plan.plan_schema,
        "readable_result_schema": envelope.payload.result_schema,
        "external_store_namespace": namespace,
        "legacy_store_read_only": namespace is None,
        "source_store_namespace_id": getattr(plan.source, "task_b_store_namespace_id", None),
        "source_bindings": bindings,
        "view_path": str(view_path),
        "view_sha256": file_sha256(view_path),
        "view_bytes": view_path.stat().st_size,
        "reporting_definition_version": VIEW_VERSION,
        "version_label": version_label,
        "qualification": qualification,
        "predecessor_result_id": predecessor_result_id,
        "evaluation_dates": list(plan.source.evaluation_dates),
        "warmup_dates": list(plan.source.warmup_dates),
    }
    pointer["binding_id"] = canonical_sha256(pointer)
    path = catalog_path(catalog_store_root)
    catalog = _load_catalog(path)
    group = catalog["studies"].setdefault(
        study_key, {"display_name": display_name, "versions": {}, "preferred_result_id": result_id}
    )
    if group["display_name"] != display_name:
        raise ValueError("a registered study's identity cannot be reused for another name")
    existing = group["versions"].get(result_id)
    if existing is not None and existing != pointer:
        invariant = (
            "result_id",
            "plan_id",
            "approval_id",
            "source_bindings",
            "external_store_root",
        )
        if any(existing[key] != pointer[key] for key in invariant):
            raise ValueError("the registered result differs from its original economic binding")
        if existing["reporting_definition_version"] == pointer["reporting_definition_version"]:
            raise ValueError(
                "same reporting version cannot replace an existing immutable publication"
            )
        group.setdefault("reporting_history", []).append(existing)
    group["versions"][result_id] = pointer
    if preferred:
        group["preferred_result_id"] = result_id
    group["updated_at_utc"] = datetime.now(UTC).isoformat()
    _write(path, catalog)
    return pointer


def verify_pointer(pointer: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in pointer.items() if k != "binding_id"}
    if pointer.get("binding_id") != canonical_sha256(body):
        raise ValueError("published catalog pointer binding has changed")
    root = Path(pointer["external_store_root"]).resolve(strict=True)
    for binding in pointer["source_bindings"]:
        if _manifest_binding(root, binding["family"], binding["identity"]) != binding:
            raise ValueError("published source manifest differs from its registered binding")
    namespace_path = root / "STORE_NAMESPACE.json"
    namespace = (
        json.loads(namespace_path.read_text(encoding="utf-8")) if namespace_path.exists() else None
    )
    if namespace != pointer.get("external_store_namespace"):
        raise ValueError("published external store namespace changed")
    path = Path(pointer["view_path"])
    stat = path.stat()
    if (
        stat.st_size != pointer["view_bytes"]
        or _verified_digest(str(path), stat.st_size, stat.st_mtime_ns) != pointer["view_sha256"]
    ):
        raise ValueError("published reporting view failed integrity verification")
    return pointer


def registered_groups(store_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Verified study groups; a broken pointer is visible and never substituted."""
    groups, issues = [], []
    for key, group in _load_catalog(catalog_path(store_root))["studies"].items():
        try:
            pointer = verify_pointer(group["versions"][group["preferred_result_id"]])
        except Exception as error:
            issues.append(f"{group['display_name']}: registered result unavailable ({error}).")
            continue
        groups.append({"study_key": key, **group, "pointer": pointer})
    return groups, issues


def resolve_registered_result(
    result_id: str, *, store_root: Path | None = None, external_store_root: Path | None = None
) -> dict[str, Any] | None:
    catalog = _load_catalog(catalog_path(store_root))
    for key, group in catalog["studies"].items():
        pointer = group["versions"].get(result_id)
        if pointer is None:
            continue
        if (
            external_store_root is not None
            and Path(pointer["external_store_root"]).resolve()
            != Path(external_store_root).resolve()
        ):
            continue
        verify_pointer(pointer)
        return {
            "study_key": key,
            "name": group["display_name"],
            "status": "Completed",
            "result_id": result_id,
            "plan_id": pointer["plan_id"],
            "store_root": pointer["external_store_root"],
            "app": "main",
            "external_review_only": True,
            "catalog_binding": pointer,
            "versions": group["versions"],
        }
    return None


@lru_cache(maxsize=8)
def _read_view(path: str, digest: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != VIEW_VERSION:
        raise ValueError("registered reporting view has an unsupported schema")
    return payload


def load_registered_view(pointer: dict[str, Any]) -> dict[str, Any]:
    verify_pointer(pointer)
    payload = _read_view(pointer["view_path"], pointer["view_sha256"])
    if (
        payload["result_id"] != pointer["result_id"]
        or payload["plan_id"] != pointer["plan_id"]
        or payload["source_bindings"] != pointer["source_bindings"]
    ):
        raise ValueError("registered reporting view binds different source evidence")
    return payload["result"]
