"""Synthetic, bounded checks for the source-review publisher."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.propsim.funded import mffu_source_review
from alpha_lab.propsim.funded.core_identity import source_identity_at
from alpha_lab.propsim.funded.mffu_source_review import (
    _relative_source,
    publish_mffu_source_review,
    verify_mffu_source_review_zip,
)
from alpha_lab.propsim.funded.result import result_sha256


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def _repo(path: Path, *, ignore_reports: bool = False) -> tuple[Path, str]:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "Synthetic Test")
    (path / "src").mkdir()
    (path / "src" / "task.py").write_text("BASE = 1\n", encoding="utf-8")
    if ignore_reports:
        (path / ".gitignore").write_text("reports/\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "base")
    return path, _git(path, "rev-parse", "HEAD")


def _envelope(path: Path, field: str, payload: dict) -> tuple[str, Path]:
    identity = canonical_contract_sha256(payload)
    path.write_text(json.dumps({field: identity, "payload": payload}), encoding="utf-8")
    return identity, path


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
           *, runtime_hash_rows: bool = False,
           dispatch_hash_mapping: bool = False,
           second_reused_control: bool = False) -> dict:
    repo, repo_ref = _repo(tmp_path / "repo", ignore_reports=True)
    core, core_ref = _repo(tmp_path / "core")
    (repo / "src" / "task.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "src" / "extra.py").write_text("EXTRA = 3\n", encoding="utf-8")
    (repo / "src" / "unrelated.py").write_text("UNRELATED = 7\n", encoding="utf-8")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_task.py").write_text("def test_task(): pass\n", encoding="utf-8")
    (core / "src" / "task.py").write_text("CORE = 4\n", encoding="utf-8")
    (core / "src" / "added.py").write_text("ADDED = 5\n", encoding="utf-8")
    (core / "tests" / "fixtures").mkdir(parents=True)
    (core / "tests" / "test_core.py").write_text("def test_core(): pass\n", encoding="utf-8")
    (core / "tests" / "fixtures" / "case.py").write_text("CASE = 1\n", encoding="utf-8")
    module_file = core / "src" / "strategy_core" / "__init__.py"
    module_file.parent.mkdir()
    module_file.write_text("TASK_CORE = True\n", encoding="utf-8")
    identity = source_identity_at(core)
    runtime = str((repo / "src" / "task.py").resolve())

    runtime_hash = hashlib.sha256((repo / "src" / "task.py").read_bytes()).hexdigest()
    runtime_source_hashes = ([[runtime, runtime_hash]] if runtime_hash_rows
                             else {runtime: runtime_hash})
    variants = [{"variant_id": f"MCB{number:03d}", "intent_sha256": "a" * 64,
                 "effective_section_config_hash": "b" * 64,
                 "effective_behavior_hash": "c" * 64}
                for number in range(1, 65)]
    plan_payload = {
        "variants": variants, "core_root": str(core),
        "core_source": {key: identity[key] for key in
                        ("base_commit", "branch", "patch_sha256")},
        "runtime_source_file_sha256": runtime_source_hashes,
        "context_archive_sha256": "d" * 64,
        "context_table_sha256": {"table.csv": "e" * 64},
        "handoff_zip_sha256": "f" * 64,
        "input_archive_sha256": {"input.zip": "0" * 64},
    }
    plan_id, plan_path = _envelope(tmp_path / "plan.json",
                                   "funded_comparison_plan_id", plan_payload)
    approval_id, approval_path = _envelope(
        tmp_path / "approval.json", "funded_comparison_approval_id",
        {"funded_comparison_plan_id": plan_id, "statement": "owner approval"},
    )
    reused_controls = {0: "C02"}
    if second_reused_control:
        reused_controls[24] = "C01"
    dispositions = [
        {"variant_id": row["variant_id"],
         "status": "compatible_reused" if index in reused_controls else "newly_completed",
         "reused_from": reused_controls.get(index)}
        for index, row in enumerate(variants)
    ]
    dispatch_hashes = (dict(runtime_source_hashes) if dispatch_hash_mapping
                       else runtime_source_hashes)
    dispatches = [
        {"plan_id": plan_id, "approval_id": approval_id, "variant": row,
         "core_source": plan_payload["core_source"],
         "runtime_source_file_sha256": dispatch_hashes,
         "context_archive_sha256": plan_payload["context_archive_sha256"]}
        for row in variants
    ]
    result = {
        "funded_comparison_plan_id": plan_id, "validation": {"passed": True},
        "mffu_batch": {"plan": plan_payload, "approval_id": approval_id,
                       "dispositions": dispositions,
                       "worker_dispatches": dispatches,
                       "reuse_proofs": {
                           variants[index]["variant_id"]: {"reference": reference}
                           for index, reference in reused_controls.items()
                       }},
    }
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    result_id, result_env_path = _envelope(
        tmp_path / "result_envelope.json", "funded_comparison_result_id",
        {"funded_comparison_plan_id": plan_id,
         "funded_comparison_approval_id": approval_id,
         "result_json_sha256": result_sha256(result), "validation_passed": True},
    )
    review_zip = tmp_path / "result_review.zip"
    review_zip.write_bytes(b"synthetic result review receipt")
    monkeypatch.setattr(
        "alpha_lab.propsim.funded.mffu_batch_review.verify_mffu_result_review_zip",
        lambda path, *, extraction_root: {
            "passed": True, "plan_id": plan_id, "result_id": result_id,
            "zip_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        },
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({
        "commands": ["python -m pytest -q focused"],
        "test_receipts": ["focused.txt"], "failures": [], "corrections": [],
        "environment": {"python_version": sys.version,
                        "python_executable": sys.executable,
                        "platform": platform.platform(),
                        "quant_lab_root": str(repo), "core_root": str(core)},
        "import_identity": {
            "strategy_core_module_file": str(module_file),
            "strategy_core_module_sha256": hashlib.sha256(module_file.read_bytes()).hexdigest(),
            "core_source": {key: identity[key] for key in
                            ("base_commit", "branch", "patch_sha256")},
        },
    }), encoding="utf-8")
    receipt = tmp_path / "focused.txt"
    receipt.write_text("64 synthetic mappings checked\n", encoding="utf-8")
    return {
        "repo_root": repo, "core_root": core,
        "base_repo_ref": repo_ref, "base_core_ref": core_ref,
        "plan_envelope_path": plan_path, "approval_envelope_path": approval_path,
        "result_envelope_path": result_env_path, "result_json_path": result_path,
        "result_review_zip": review_zip, "output_zip": repo / "reports" / "source.zip",
        "repo_files": ["src/extra.py", "tests/test_task.py"],
        "core_files": ["tests/test_core.py", "tests/fixtures/case.py"],
        "receipt_files": [receipt], "evidence_record_path": evidence,
        "staging_root": tmp_path / "staging",
    }


def test_source_review_publish_scope_mapping_and_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _setup(tmp_path, monkeypatch)
    receipt = publish_mffu_source_review(**args)
    assert receipt["passed"] is True
    assert receipt["members_verified"] > 10
    assert verify_mffu_source_review_zip(
        args["output_zip"], extraction_root=args["staging_root"],
        repo_root=args["repo_root"],
    )["zip_sha256"] == receipt["zip_sha256"]
    with zipfile.ZipFile(args["output_zip"]) as archive:
        names = set(archive.namelist())
        assert "source/quant_lab/src/task.py" in names
        assert "source/quant_lab/src/extra.py" in names
        assert "source/quant_lab/src/unrelated.py" not in names
        assert "source/task_core/src/task.py" in names
        assert "source/task_core/src/added.py" in names
        assert b"VALUE = 2" in archive.read("patches/quant_lab_tracked.patch")
        assert b"EXTRA = 3" in archive.read("patches/quant_lab_untracked.patch")
        mapping = json.loads(archive.read("bindings/worker_mapping.json"))
        assert len(mapping["rows"]) == 64
        assert mapping["rows"][0]["reuse_proof_sha256"] is not None
        assert mapping["rows"][0]["worker_dispatch_sha256"] is not None
        assert mapping["rows"][1]["worker_dispatch_sha256"] is not None


def test_source_review_accepts_frozen_runtime_hash_pair_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _setup(tmp_path, monkeypatch, runtime_hash_rows=True)
    receipt = publish_mffu_source_review(**args)
    assert receipt["passed"] is True
    assert verify_mffu_source_review_zip(
        args["output_zip"], extraction_root=args["staging_root"],
        repo_root=args["repo_root"],
    )["zip_sha256"] == receipt["zip_sha256"]


def test_source_review_preserves_reused_control_dispatch_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _setup(tmp_path, monkeypatch, runtime_hash_rows=True,
                  dispatch_hash_mapping=True, second_reused_control=True)
    plan = json.loads(args["plan_envelope_path"].read_text(encoding="utf-8"))["payload"]
    result = json.loads(args["result_json_path"].read_text(encoding="utf-8"))
    assert isinstance(plan["runtime_source_file_sha256"], list)
    assert isinstance(result["mffu_batch"]["worker_dispatches"][0][
        "runtime_source_file_sha256"], dict)
    receipt = publish_mffu_source_review(**args)
    assert receipt["passed"] is True
    with zipfile.ZipFile(args["output_zip"]) as archive:
        mapping = json.loads(archive.read("bindings/worker_mapping.json"))
    for index in (0, 24):
        variant_id = f"MCB{index + 1:03d}"
        row = mapping["rows"][index]
        assert row["status"] == "compatible_reused"
        assert row["worker_dispatch_sha256"] == canonical_contract_sha256(
            result["mffu_batch"]["worker_dispatches"][index])
        assert row["reuse_proof_sha256"] == canonical_contract_sha256(
            result["mffu_batch"]["reuse_proofs"][variant_id])
    assert verify_mffu_source_review_zip(
        args["output_zip"], extraction_root=args["staging_root"],
        repo_root=args["repo_root"],
    )["zip_sha256"] == receipt["zip_sha256"]


@pytest.mark.parametrize("rows", [
    [["file.py", "a" * 64], ["file.py", "b" * 64]],
    [["file.py"]],
    [["file.py", "not-a-hash"]],
    [],
])
def test_source_review_rejects_duplicate_or_malformed_runtime_hash_rows(rows) -> None:
    with pytest.raises(ValueError, match="runtime source"):
        mffu_source_review._runtime_source_hashes(rows)


def test_source_review_rejects_changed_plan_bound_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _setup(tmp_path, monkeypatch)
    (args["repo_root"] / "src" / "task.py").write_text("MUTATED = 9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="plan-bound runtime source changed"):
        publish_mffu_source_review(**args)


def test_source_review_rejects_unsafe_scope_and_unlisted_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _setup(tmp_path, monkeypatch)
    (args["repo_root"] / "data").mkdir()
    (args["repo_root"] / "data" / "raw.py").write_text("RAW = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bounded review scope"):
        _relative_source(args["repo_root"], "data/raw.py")
    publish_mffu_source_review(**args)
    with zipfile.ZipFile(args["output_zip"], "a") as archive:
        archive.writestr("rogue.txt", b"unlisted")
    with pytest.raises(ValueError, match="members differ"):
        verify_mffu_source_review_zip(
            args["output_zip"], extraction_root=args["staging_root"],
            repo_root=args["repo_root"],
        )


def test_source_review_recomputes_core_patch_after_manifest_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _setup(tmp_path, monkeypatch)
    publish_mffu_source_review(**args)
    altered = tmp_path / "altered.zip"
    target = "patches/task_core_src_identity.patch"
    with zipfile.ZipFile(args["output_zip"]) as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    new_patch = next(body for info, body in entries if info.filename == target) + b"# changed\n"
    manifest = json.loads(next(body for info, body in entries
                               if info.filename == "manifest.json"))
    row = next(row for row in manifest["files"] if row["path"] == target)
    row["bytes"] = len(new_patch)
    row["sha256"] = hashlib.sha256(new_patch).hexdigest()
    with zipfile.ZipFile(altered, "w") as output:
        for info, body in entries:
            if info.filename == target:
                body = new_patch
            elif info.filename == "manifest.json":
                body = json.dumps(manifest).encode()
            output.writestr(info, body)
    with pytest.raises(ValueError, match="Core patch identity cannot be reproduced"):
        verify_mffu_source_review_zip(
            altered, extraction_root=args["staging_root"], repo_root=args["repo_root"],
        )


def test_source_review_removes_final_zip_if_postpublish_readback_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _setup(tmp_path, monkeypatch)
    original = mffu_source_review.verify_mffu_source_review_zip
    calls = 0

    def fail_final(path, *, extraction_root, repo_root):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("postpublish readback failed")
        return original(path, extraction_root=extraction_root, repo_root=repo_root)

    monkeypatch.setattr(mffu_source_review, "verify_mffu_source_review_zip", fail_final)
    with pytest.raises(ValueError, match="postpublish readback failed"):
        publish_mffu_source_review(**args)
    assert calls == 2
    assert not args["output_zip"].exists()
