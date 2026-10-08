"""Bounded source-review package for the completed MFFU batch.

The package carries exact selected source bytes and scoped Git patches, plus
small plan/approval/result and verification receipts. It never copies market
data, a repository tree, an environment, or the full financial result.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.propsim.funded.core_identity import source_identity_at, untracked_source_files
from alpha_lab.propsim.funded.result import result_sha256

SCHEMA = "ifsm_mffu_source_review_zip_v1"
ZIP_TIME = (2026, 10, 7, 0, 0, 0)
MAX_FILE_BYTES = 2_000_000
MAX_PACKAGE_BYTES = 30_000_000
_ALLOWED_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt"}
_FORBIDDEN_PARTS = {
    ".git", ".venv", "__pycache__", "cache", "catboost_info", "data",
    "databento", "inputs", "models", "node_modules", "raw", "reports",
    "site-packages", "vendor",
}
_LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")
_PRIVATE_KEY = re.compile(rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True,
    ).stdout.decode("utf-8", errors="surrogateescape")


def _within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _relative_source(root: Path, item: str | Path) -> str:
    root = root.resolve()
    path = Path(item)
    path = (path if path.is_absolute() else root / path).resolve()
    if not _within(path, root) or not path.is_file() or path.is_symlink():
        raise ValueError(f"source must be a regular file below its checkout: {item}")
    rel = path.relative_to(root).as_posix()
    parts = PurePosixPath(rel).parts
    if (any(part.lower() in _FORBIDDEN_PARTS for part in parts)
            or path.suffix.lower() not in _ALLOWED_SUFFIXES):
        raise ValueError(f"source path is outside the bounded review scope: {rel}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"source is too large for the review ZIP: {rel}")
    return rel


def _read_scoped(root: Path, items: Iterable[str | Path]) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for item in items:
        rel = _relative_source(root, item)
        body = (root / rel).read_bytes()
        if _PRIVATE_KEY.search(body):
            raise ValueError(f"source contains a private-key marker: {rel}")
        body.decode("utf-8")
        files[rel] = body
    return dict(sorted(files.items()))


def _runtime_source_hashes(value: Any) -> dict[str, str]:
    """Read the frozen plan's ordered path/hash pairs or a legacy mapping."""
    rows = value.items() if isinstance(value, dict) else value
    if not isinstance(value, (dict, list, tuple)):
        raise ValueError("runtime source hashes must be path/hash pairs")
    hashes: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("runtime source hash row must have a path and SHA-256")
        path, digest = row
        if (not isinstance(path, str) or not path
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError("runtime source hash row has an invalid path or SHA-256")
        if path in hashes:
            raise ValueError(f"duplicate runtime source hash path: {path}")
        hashes[path] = digest
    if not hashes:
        raise ValueError("runtime source hash list is empty")
    return hashes


def _changed_core_sources(core_root: Path, base_ref: str) -> set[str]:
    tracked = set(_git(core_root, "diff", "--name-only", base_ref, "--", "src").splitlines())
    untracked = set(_git(core_root, "ls-files", "--others", "--exclude-standard",
                         "--", "src").splitlines())
    return {name for name in tracked | untracked if (core_root / name).is_file()
            and not name.endswith(".pyc") and "__pycache__" not in name}


def _scoped_patches(root: Path, base_ref: str,
                    files: dict[str, bytes]) -> tuple[bytes, bytes]:
    names = sorted(files)
    tracked = set(_git(root, "ls-files", "--", *names).splitlines())
    tracked_patch = _git(root, "diff", "--binary", base_ref, "--", *names).encode(
        "utf-8", errors="surrogateescape",
    )
    untracked_parts = []
    for name in names:
        if name in tracked:
            continue
        lines = files[name].decode("utf-8").splitlines(keepends=True)
        untracked_parts.append(f"diff --git a/{name} b/{name}\nnew file mode 100644\n")
        untracked_parts.extend(difflib.unified_diff([], lines, fromfile="/dev/null",
                                                    tofile=f"b/{name}"))
    return tracked_patch, "".join(untracked_parts).encode("utf-8")


def _validated_envelope(path: Path, id_key: str) -> tuple[dict, bytes]:
    body = Path(path).read_bytes()
    return _validated_envelope_bytes(body, id_key), body


def _validated_envelope_bytes(body: bytes, id_key: str) -> dict:
    envelope = json.loads(body)
    if (set(envelope) != {id_key, "payload"}
            or envelope[id_key] != canonical_contract_sha256(envelope["payload"])):
        raise ValueError(f"saved {id_key} envelope identity differs from its payload")
    return envelope


def _worker_mapping(plan: dict, result: dict, plan_id: str,
                    approval_id: str) -> dict[str, Any]:
    batch = result["mffu_batch"]
    variants = plan["variants"]
    dispositions = batch["dispositions"]
    expected_ids = [f"MCB{number:03d}" for number in range(1, 65)]
    if ([row["variant_id"] for row in variants] != expected_ids
            or [row["variant_id"] for row in dispositions] != expected_ids):
        raise ValueError("plan and result must retain 64 ordered worker identities")
    plan_runtime_hashes = _runtime_source_hashes(plan["runtime_source_file_sha256"])
    dispatches: dict[str, dict] = {}
    for dispatch in batch["worker_dispatches"]:
        variant_id = dispatch["variant"]["variant_id"]
        if (variant_id in dispatches or dispatch["plan_id"] != plan_id
                or dispatch["approval_id"] != approval_id
                or dispatch["core_source"] != plan["core_source"]
                or _runtime_source_hashes(dispatch["runtime_source_file_sha256"])
                != plan_runtime_hashes
                or dispatch["context_archive_sha256"] != plan["context_archive_sha256"]
                or dispatch["variant"] != variants[int(variant_id[3:]) - 1]):
            raise ValueError(f"saved worker dispatch differs from the plan: {variant_id}")
        dispatches[variant_id] = dispatch
    proofs = batch.get("reuse_proofs", {})
    rows = []
    for variant, disposition in zip(variants, dispositions, strict=True):
        variant_id = variant["variant_id"]
        status = disposition["status"]
        if status not in {"newly_completed", "compatible_reused"}:
            raise ValueError(f"source review requires completed financial row: {variant_id}")
        dispatch = dispatches.get(variant_id)
        if (dispatch is None
                or (status == "compatible_reused") != (variant_id in proofs)):
            raise ValueError(f"dispatch/reuse proof disagrees with row: {variant_id}")
        rows.append({
            "variant_id": variant_id,
            "status": status,
            "intent_sha256": variant["intent_sha256"],
            "effective_section_config_hash": variant["effective_section_config_hash"],
            "effective_behavior_hash": variant["effective_behavior_hash"],
            "worker_dispatch_sha256": canonical_contract_sha256(dispatch),
            "reuse_proof_sha256": canonical_contract_sha256(proofs[variant_id])
            if status == "compatible_reused" else None,
            "reused_from": disposition.get("reused_from"),
        })
    if len(dispatches) != 64:
        raise ValueError("saved worker and reuse mappings do not cover all 64 rows")
    return {"schema": "ifsm_mffu_source_worker_mapping_v1", "plan_id": plan_id,
            "approval_id": approval_id, "rows": rows}


def _check_guide_links(files: dict[str, bytes]) -> None:
    guide = files.get("README.md", b"").decode("utf-8")
    for raw in _LINK.findall(guide):
        target = raw.split("#", 1)[0].strip()
        if not target or ":" in target or target.startswith("/"):
            continue
        if (".." in PurePosixPath(target).parts or target not in files):
            raise ValueError(f"source-review guide has a broken link: {raw}")


def verify_mffu_source_review_zip(path: Path, *, extraction_root: Path,
                                  repo_root: Path | None = None) -> dict[str, Any]:
    """Check every member and extract/read back in a caller-owned external root."""
    path = Path(path)
    extraction_root = Path(extraction_root).resolve()
    if repo_root is not None and _within(extraction_root, Path(repo_root).resolve()):
        raise ValueError("source-review extraction must be outside the repository")
    extraction_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [item.filename for item in entries]
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("source-review ZIP has duplicate members or no manifest")
        if any(item.is_dir() or item.filename.startswith("/")
               or "\\" in item.filename or ":" in item.filename
               or ".." in PurePosixPath(item.filename).parts
               or PurePosixPath(item.filename).as_posix() != item.filename
               or item.external_attr >> 16 & 0o170000 == 0o120000
               for item in entries):
            raise ValueError("source-review ZIP contains an unsafe member")
        if sum(item.file_size for item in entries) > MAX_PACKAGE_BYTES:
            raise ValueError("source-review ZIP declares oversized payloads")
        files = {name: archive.read(name) for name in names}
        manifest = json.loads(files["manifest.json"])
        if manifest.get("schema") != SCHEMA:
            raise ValueError("source-review manifest schema differs")
        listed = {row["path"]: row for row in manifest.get("files", [])}
        if (len(listed) != len(manifest.get("files", []))
                or set(files) != set(listed) | {"manifest.json"}):
            raise ValueError("source-review members differ from final manifest")
        if (sum(len(body) for body in files.values()) > MAX_PACKAGE_BYTES
                or any(len(files[name]) != row["bytes"]
                       or _sha(files[name]) != row["sha256"]
                       for name, row in listed.items())):
            raise ValueError("source-review payload size or hash differs")
        stamp = tuple(manifest["zip_entry_time_utc"])
        when = datetime(*stamp, tzinfo=UTC)
        if (when.year < 1980 or when > datetime.now(UTC) + timedelta(days=1)
                or any(item.date_time != stamp for item in entries)):
            raise ValueError("source-review ZIP timestamp differs or is invalid")
        _check_guide_links(files)
        plan = _validated_envelope_bytes(files["bindings/plan_envelope.json"],
                                         "funded_comparison_plan_id")
        approval = _validated_envelope_bytes(files["bindings/approval_envelope.json"],
                                             "funded_comparison_approval_id")
        result = _validated_envelope_bytes(files["bindings/result_envelope.json"],
                                           "funded_comparison_result_id")
        source = json.loads(files["bindings/source_bindings.json"])
        mapping = json.loads(files["bindings/worker_mapping.json"])
        if (manifest["plan_id"] != plan["funded_comparison_plan_id"]
                or manifest["approval_id"] != approval["funded_comparison_approval_id"]
                or manifest["result_id"] != result["funded_comparison_result_id"]
                or approval["payload"]["funded_comparison_plan_id"] != manifest["plan_id"]
                or result["payload"]["funded_comparison_plan_id"] != manifest["plan_id"]
                or result["payload"]["funded_comparison_approval_id"]
                != manifest["approval_id"]
                or result["payload"]["result_json_sha256"]
                != manifest["result_json_sha256"]
                or manifest["review_zip_sha256"] != source["review_zip_sha256"]
                or manifest["core_source_patch_sha256"]
                != source["core_source"]["patch_sha256"]
                or manifest["repo_base_commit"] != source["repo_base_commit"]
                or manifest["core_base_commit"] != source["core_base_commit"]
                or mapping["plan_id"] != manifest["plan_id"]
                or mapping["approval_id"] != manifest["approval_id"]
                or [row["variant_id"] for row in mapping["rows"]]
                != [f"MCB{number:03d}" for number in range(1, 65)]):
            raise ValueError("source-review identities contradict packaged bytes")
        if (source["plan_runtime_source_file_sha256"]
                != plan["payload"]["runtime_source_file_sha256"]
                or {key: source["core_source"][key] for key in
                    ("base_commit", "branch", "patch_sha256")}
                != {key: plan["payload"]["core_source"][key] for key in
                    ("base_commit", "branch", "patch_sha256")}):
            raise ValueError("source-review plan source identity contradicts bindings")
        for worker, variant in zip(mapping["rows"], plan["payload"]["variants"],
                                   strict=True):
            if (worker["variant_id"] != variant["variant_id"]
                    or any(worker[key] != variant[key] for key in (
                        "intent_sha256", "effective_section_config_hash",
                        "effective_behavior_hash",
                    ))
                    or not worker["worker_dispatch_sha256"]
                    or (worker["status"] == "compatible_reused")
                    != (worker["reuse_proof_sha256"] is not None)):
                raise ValueError("source-review worker mapping contradicts frozen plan")
        source_root = Path(source["repo_root"])
        for source_path, expected in _runtime_source_hashes(
            source["plan_runtime_source_file_sha256"],
        ).items():
            rel = Path(source_path).relative_to(source_root).as_posix()
            if _sha(files[f"source/quant_lab/{rel}"]) != expected:
                raise ValueError(f"source-review runtime file contradicts plan: {rel}")
        for prefix, key in (("source/quant_lab/", "quant_lab_scoped_source_sha256"),
                            ("source/task_core/", "task_core_scoped_source_sha256")):
            actual = {name.removeprefix(prefix): _sha(body)
                      for name, body in files.items() if name.startswith(prefix)}
            if actual != source[key]:
                raise ValueError("source-review scoped source hashes contradict files")
        review_receipt = json.loads(files["evidence/result_review_readback.json"])
        if (not review_receipt["passed"]
                or review_receipt["plan_id"] != manifest["plan_id"]
                or review_receipt["result_id"] != manifest["result_id"]
                or review_receipt["zip_sha256"] != manifest["review_zip_sha256"]):
            raise ValueError("source-review financial ZIP receipt contradicts manifest")
        core_digest = hashlib.sha256(files["patches/task_core_src_identity.patch"])
        for rel in source["core_untracked_source_paths"]:
            core_digest.update(rel.encode())
            core_digest.update(files[f"source/task_core/{rel}"])
        if core_digest.hexdigest() != manifest["core_source_patch_sha256"]:
            raise ValueError("source-review Core patch identity cannot be reproduced")
        with tempfile.TemporaryDirectory(prefix="mffu_source_readback_",
                                         dir=extraction_root) as temp:
            root = Path(temp)
            for name, body in files.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
            if any((root / name).read_bytes() != body for name, body in files.items()):
                raise ValueError("extracted source-review bytes differ")
        return {"passed": True, "plan_id": manifest["plan_id"],
                "approval_id": manifest["approval_id"], "result_id": manifest["result_id"],
                "members_verified": len(files), "manifest_sha256": _sha(files["manifest.json"]),
                "zip_sha256": _sha(path.read_bytes())}


def publish_mffu_source_review(
    *, repo_root: Path, core_root: Path, base_repo_ref: str, base_core_ref: str,
    plan_envelope_path: Path, approval_envelope_path: Path,
    result_envelope_path: Path, result_json_path: Path, result_review_zip: Path,
    output_zip: Path, repo_files: Iterable[str | Path], core_files: Iterable[str | Path],
    receipt_files: Iterable[Path], evidence_record_path: Path,
    staging_root: Path,
) -> dict[str, Any]:
    """Publish the immutable source ZIP after all identities and both ZIPs verify."""
    repo_root, core_root = Path(repo_root).resolve(), Path(core_root).resolve()
    staging_root, output_zip = Path(staging_root).resolve(), Path(output_zip).resolve()
    if (_within(staging_root, repo_root) or _within(staging_root, core_root)
            or not _within(output_zip, repo_root / "reports")
            or output_zip.suffix.lower() != ".zip" or output_zip.exists()):
        raise ValueError("stage externally and choose a new final ZIP below reports")
    staging_root.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["git", "-C", str(repo_root), "check-ignore", "-q", "--",
                       str(output_zip)], check=False).returncode != 0:
        raise ValueError("final source-review ZIP must be Git-ignored")
    if (_git(repo_root, "rev-parse", "HEAD").strip() != base_repo_ref
            or _git(core_root, "rev-parse", "HEAD").strip() != base_core_ref):
        raise ValueError("source checkout base commit differs from declared base")
    plan_env, plan_bytes = _validated_envelope(
        plan_envelope_path, "funded_comparison_plan_id")
    approval_env, approval_bytes = _validated_envelope(
        approval_envelope_path, "funded_comparison_approval_id")
    result_env, result_env_bytes = _validated_envelope(
        result_envelope_path, "funded_comparison_result_id")
    plan = plan_env["payload"]
    plan_id = plan_env["funded_comparison_plan_id"]
    approval_id = approval_env["funded_comparison_approval_id"]
    result_id = result_env["funded_comparison_result_id"]
    result = json.loads(Path(result_json_path).read_bytes())
    if (approval_env["payload"]["funded_comparison_plan_id"] != plan_id
            or result_env["payload"]["funded_comparison_plan_id"] != plan_id
            or result_env["payload"]["funded_comparison_approval_id"] != approval_id
            or result_env["payload"]["result_json_sha256"] != result_sha256(result)
            or not result_env["payload"]["validation_passed"]
            or not result.get("validation", {}).get("passed")
            or result.get("funded_comparison_plan_id") != plan_id
            or result["mffu_batch"]["plan"] != plan
            or result["mffu_batch"]["approval_id"] != approval_id):
        raise ValueError("saved plan, approval or completed result binding differs")
    core_identity = source_identity_at(core_root)
    if (Path(plan["core_root"]).resolve() != core_root
            or {key: core_identity[key] for key in ("base_commit", "branch", "patch_sha256")}
            != {key: plan["core_source"][key] for key in
                ("base_commit", "branch", "patch_sha256")}):
        raise ValueError("task Core source differs from frozen plan")
    repo_list = list(repo_files)
    for source_path, expected in _runtime_source_hashes(
        plan["runtime_source_file_sha256"],
    ).items():
        rel = _relative_source(repo_root, source_path)
        if _sha((repo_root / rel).read_bytes()) != expected:
            raise ValueError(f"plan-bound runtime source changed: {rel}")
        repo_list.append(rel)
    core_list = list(core_files) + sorted(_changed_core_sources(core_root, base_core_ref))
    repo_sources = _read_scoped(repo_root, repo_list)
    core_sources = _read_scoped(core_root, core_list)
    if (not any(name.startswith("tests/") for name in repo_sources)
            or not any(name.startswith("tests/") for name in core_sources)
            or not any("/fixtures/" in f"/{name}" for name in
                       (*repo_sources, *core_sources))):
        raise ValueError("source review requires scoped tests for both checkouts and a fixture")
    core_untracked_paths = untracked_source_files(core_root)
    core_identity_patch = _git(core_root, "diff", "--binary", base_core_ref,
                               "--", "src").encode("utf-8", errors="surrogateescape")
    core_digest = hashlib.sha256(core_identity_patch)
    for rel in core_untracked_paths:
        core_digest.update(rel.encode())
        core_digest.update(core_sources[rel])
    if core_digest.hexdigest() != core_identity["patch_sha256"]:
        raise ValueError("packaged task Core bytes do not reproduce its plan identity")
    mapping = _worker_mapping(plan, result, plan_id, approval_id)
    from alpha_lab.propsim.funded.mffu_batch_review import verify_mffu_result_review_zip

    review_receipt = verify_mffu_result_review_zip(
        result_review_zip, extraction_root=staging_root,
    )
    if (review_receipt["plan_id"] != plan_id or review_receipt["result_id"] != result_id):
        raise ValueError("financial review ZIP binds a different result")
    evidence_record_path = Path(evidence_record_path).resolve()
    if (_within(evidence_record_path, repo_root) or _within(evidence_record_path, core_root)
            or evidence_record_path.is_symlink()
            or evidence_record_path.stat().st_size > MAX_FILE_BYTES):
        raise ValueError("source-review evidence must be a small external file")
    evidence_bytes = evidence_record_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    required = {"commands", "test_receipts", "failures", "corrections",
                "environment", "import_identity"}
    if not required <= set(evidence) or not evidence["commands"] or not evidence["test_receipts"]:
        raise ValueError("source-review evidence lacks commands, test receipts or identity")
    environment = evidence["environment"]
    imported = evidence["import_identity"]
    if (not environment.get("python_version") or not environment.get("python_executable")
            or not environment.get("platform")
            or Path(environment.get("quant_lab_root", "")).resolve() != repo_root
            or Path(environment.get("core_root", "")).resolve() != core_root):
        raise ValueError("source-review environment identity is incomplete")
    module_file = Path(imported.get("strategy_core_module_file", "")).resolve()
    if (not module_file.is_file() or not _within(module_file, core_root / "src")
            or _sha(module_file.read_bytes())
            != imported.get("strategy_core_module_sha256")
            or {key: imported.get("core_source", {}).get(key) for key in
                ("base_commit", "branch", "patch_sha256")}
            != {key: core_identity[key] for key in
                ("base_commit", "branch", "patch_sha256")}):
        raise ValueError("recorded imported Core identity differs from final source")
    repo_patch, repo_new = _scoped_patches(repo_root, base_repo_ref, repo_sources)
    core_patch, core_new = _scoped_patches(core_root, base_core_ref, core_sources)
    files: dict[str, bytes] = {
        "bindings/plan_envelope.json": plan_bytes,
        "bindings/approval_envelope.json": approval_bytes,
        "bindings/result_envelope.json": result_env_bytes,
        "bindings/worker_mapping.json": _json_bytes(mapping),
        "bindings/source_bindings.json": _json_bytes({
            "repo_root": str(repo_root), "core_root": str(core_root),
            "repo_base_commit": base_repo_ref, "core_base_commit": base_core_ref,
            "core_source": core_identity,
            "core_untracked_source_paths": core_untracked_paths,
            "quant_lab_scoped_source_sha256": {name: _sha(body) for name, body
                                                  in repo_sources.items()},
            "task_core_scoped_source_sha256": {name: _sha(body) for name, body
                                                in core_sources.items()},
            "plan_runtime_source_file_sha256": plan["runtime_source_file_sha256"],
            "handoff_zip_sha256": plan["handoff_zip_sha256"],
            "context_archive_sha256": plan["context_archive_sha256"],
            "context_table_sha256": plan["context_table_sha256"],
            "input_archive_sha256": plan["input_archive_sha256"],
            "review_zip_sha256": review_receipt["zip_sha256"],
        }),
        "evidence/evidence_record.json": evidence_bytes,
        "evidence/result_review_readback.json": _json_bytes(review_receipt),
        "patches/quant_lab_tracked.patch": repo_patch,
        "patches/quant_lab_untracked.patch": repo_new,
        "patches/task_core_tracked.patch": core_patch,
        "patches/task_core_untracked.patch": core_new,
        "patches/task_core_src_identity.patch": core_identity_patch,
    }
    for name, body in repo_sources.items():
        files[f"source/quant_lab/{name}"] = body
    for name, body in core_sources.items():
        files[f"source/task_core/{name}"] = body
    for number, source in enumerate(receipt_files, 1):
        source = Path(source).resolve()
        if (_within(source, repo_root) or _within(source, core_root)
                or source.is_symlink() or source.stat().st_size > MAX_FILE_BYTES
                or source.suffix.lower() not in _ALLOWED_SUFFIXES):
            raise ValueError(f"test receipt is too large or unsupported: {source}")
        body = source.read_bytes()
        if _PRIVATE_KEY.search(body):
            raise ValueError("test receipt contains a private-key marker")
        files[f"evidence/receipts/{number:02d}_{source.name}"] = body
    files["README.md"] = (
        b"# IFSM MFFU source review\n\n"
        b"This ZIP binds the completed 64-row result to selected Quant-Lab and "
        b"task Core source. The tracked and untracked patches are scoped to files "
        b"included here; source bytes are authoritative for every listed file. "
        b"The Core source patch identity in the plan hashes its complete `src` "
        b"diff and untracked files, while the patch files here have their own "
        b"payload hashes in the final manifest.\n\n"
        b"Read [source bindings](bindings/source_bindings.json), "
        b"[worker mappings](bindings/worker_mapping.json), "
        b"[evidence](evidence/evidence_record.json), and "
        b"[result review readback](evidence/result_review_readback.json). "
        b"The financial result remains in the separately verified result-review ZIP.\n"
    )
    if sum(len(body) for body in files.values()) > MAX_PACKAGE_BYTES:
        raise ValueError("source-review package exceeds bounded size")
    manifest = {
        "schema": SCHEMA, "plan_id": plan_id, "approval_id": approval_id,
        "result_id": result_id, "result_json_sha256": result_sha256(result),
        "review_zip_sha256": review_receipt["zip_sha256"],
        "repo_base_commit": base_repo_ref, "core_base_commit": base_core_ref,
        "core_source_patch_sha256": core_identity["patch_sha256"],
        "zip_entry_time_utc": ZIP_TIME,
        "files": [{"path": name, "bytes": len(body), "sha256": _sha(body)}
                  for name, body in sorted(files.items())],
    }
    files["manifest.json"] = _json_bytes(manifest)
    staging_root.mkdir(parents=True, exist_ok=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mffu_source_stage_", dir=staging_root) as temp:
        staged = Path(temp) / output_zip.name
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            for name, body in sorted(files.items()):
                info = zipfile.ZipInfo(name, ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, body, compress_type=zipfile.ZIP_DEFLATED,
                                 compresslevel=9)
        verify_mffu_source_review_zip(staged, extraction_root=staging_root,
                                      repo_root=repo_root)
        if staged.stat().st_dev != output_zip.parent.stat().st_dev:
            raise OSError("source-review staging and reports must share one volume")
        try:
            os.link(staged, output_zip)
        except FileExistsError as error:
            raise FileExistsError(f"immutable source-review ZIP appeared: {output_zip}") from error
        try:
            receipt = verify_mffu_source_review_zip(
                output_zip, extraction_root=staging_root, repo_root=repo_root,
            )
        except BaseException:
            output_zip.unlink(missing_ok=True)
            raise
    return receipt
