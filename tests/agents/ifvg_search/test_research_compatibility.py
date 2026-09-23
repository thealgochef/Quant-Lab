"""Compatibility proofs over temporary Git repositories; no real study execution."""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from alpha_lab.agents.data_infra.ifvg.search import research_compatibility as compatibility
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    CoreStrategyReplayIdentity,
    strategy_core_source_identity,
)
from tests.agents.ifvg_search.test_research_subject_data import subject_fixture


def git(root, *args):
    return (
        subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.STDOUT).decode().strip()
    )


def commit(root, message):
    git(root, "add", "--all")
    git(root, "commit", "-m", message)
    return git(root, "rev-parse", "HEAD")


def rebind(subject, commit_id, source_identity):
    core = CoreStrategyReplayIdentity.model_validate_json(subject.core_envelope_json)
    core = CoreStrategyReplayIdentity.from_payload(
        core.payload.model_copy(
            update={
                "strategy_core_commit": commit_id,
                "strategy_core_source_identity": source_identity,
            }
        )
    )
    return subject.model_copy(
        update={"core_replay_id": core.core_replay_id, "core_envelope_json": core.model_dump_json()}
    )


@pytest.fixture
def source(tmp_path, monkeypatch):
    core = tmp_path / "Strategy-Core"
    ql = tmp_path / "Claude-Quant-Lab"
    core.mkdir()
    ql.mkdir()
    git(core, "init", "-q")
    git(core, "config", "user.email", "fixture@example.invalid")
    git(core, "config", "user.name", "Compatibility fixture")
    git(core, "config", "commit.gpgsign", "false")
    git(core, "config", "core.autocrlf", "false")
    files = {
        ".gitignore": b"__pycache__/\n*.parquet\n",
        "README.md": b"Engine source\n",
        "docs/context.md": b"Context contract\n",
        "validation/README.md": b"Retained audit evidence\n",
        "pyproject.toml": b"[project]\nname = 'strategy-core'\nversion = '0.1.0'\n",
        "uv.lock": b"version = 1\n",
        ".python-version": b"3.13.1\n",
        "src/strategy_core/__init__.py": b"VERSION = 1\n",
        "src/strategy_core/runtime.py": b"def value():\n    return 1\n",
        "src/strategy_core/calendar.json": b'{"session": "18:00"}\n',
        "src/strategy_core/calendar.bin": b"\x00\xff\n\x80",
        "src/strategy_core/py.typed": b"",
    }
    for name, content in files.items():
        path = core / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    commit(core, "fixture source")
    saved_commit, saved_identity = strategy_core_source_identity(repository_root=core)
    subject = rebind(subject_fixture(), saved_commit, saved_identity)
    loaded = tmp_path / "installed/strategy_core"
    shutil.copytree(core / "src/strategy_core", loaded)
    monkeypatch.setattr(compatibility, "_loaded_package_root", lambda: loaded)
    monkeypatch.setattr(
        compatibility,
        "_runtime_environment",
        lambda _: {"python": "fixture", "deps": [["numpy", "1"]]},
    )
    return core, ql, loaded, subject


def test_documentation_commit_preserves_subject_and_verifiable_runtime_proof(source):
    core, ql, _loaded, subject = source
    original = subject.model_dump_json()
    unchanged = compatibility.verify_research_core_compatibility(subject, ql)
    (core / "README.md").write_text("Corrected contract version\n", encoding="utf-8")
    (core / "docs/context.md").write_text("Corrected formula version\n", encoding="utf-8")
    (core / ".gitignore").write_text("__pycache__/\n*.parquet\n.env\n", encoding="utf-8")
    new_commit = commit(core, "fixture documentation only")
    proof = compatibility.verify_research_core_compatibility(subject, ql)
    assert subject.model_dump_json() == original
    assert proof.payload.saved_commit != proof.payload.current_commit == new_commit
    assert proof.payload.runtime_tree_sha256 == unchanged.payload.runtime_tree_sha256
    assert proof.payload.loaded_package_sha256 == unchanged.payload.loaded_package_sha256
    assert proof.proof_id != unchanged.proof_id
    assert proof.payload.changed_documentation_paths == (
        ".gitignore",
        "README.md",
        "docs/context.md",
    )
    assert {item.path for item in proof.payload.runtime_files} >= {
        "uv.lock",
        "pyproject.toml",
        ".python-version",
        "src/strategy_core/calendar.bin",
    }
    assert proof.payload.historical_environment_policy == "not_recorded_by_saved_replay"
    assert (
        compatibility.ResearchCoreCompatibilityProof.model_validate_json(proof.model_dump_json())
        == proof
    )
    forged = proof.model_dump(mode="json")
    forged["payload"]["current_commit"] = "a" * 40
    with pytest.raises(ValueError, match="does not hash"):
        compatibility.ResearchCoreCompatibilityProof.model_validate(forged)


@pytest.mark.parametrize(
    "name",
    [
        "src/strategy_core/runtime.py",
        "src/strategy_core/calendar.json",
        "src/strategy_core/calendar.bin",
        "src/strategy_core/README.md",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "new_config.json",
    ],
)
def test_committed_runtime_dependency_asset_and_unknown_changes_refuse(source, name):
    core, ql, _loaded, subject = source
    path = core / name
    path.write_bytes(path.read_bytes() + b"\nchanged" if path.exists() else b"{}")
    commit(core, "fixture incompatible change")
    with pytest.raises(ValueError, match="runtime/build/dependency tree changed"):
        compatibility.verify_research_core_compatibility(subject, ql)


@pytest.mark.parametrize("name", ["README.md", "pyproject.toml", "src/strategy_core/runtime.py"])
def test_dirty_tracked_checkout_is_not_a_pinned_clean_tree(source, name):
    core, ql, _loaded, subject = source
    (core / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="clean checkout"):
        compatibility.verify_research_core_compatibility(subject, ql)


def test_assume_unchanged_cannot_hide_runtime_edit(source):
    core, ql, _loaded, subject = source
    name = "src/strategy_core/runtime.py"
    git(core, "update-index", "--assume-unchanged", name)
    (core / name).write_bytes(b"VALUE = 'altered'\n")
    assert not git(core, "status", "--porcelain")
    with pytest.raises(ValueError, match="checkout differs"):
        compatibility.verify_research_core_compatibility(subject, ql)


@pytest.mark.parametrize("name", ["new_config.json", "src/strategy_core/new_module.py"])
def test_unknown_untracked_files_refuse(source, name):
    core, ql, _loaded, subject = source
    (core / name).write_bytes(b"{}")
    with pytest.raises(ValueError, match="clean checkout"):
        compatibility.verify_research_core_compatibility(subject, ql)


def test_gitignore_cannot_hide_runtime_asset_and_only_bytecode_cache_is_exempt(source):
    core, ql, loaded, subject = source
    cache = core / "src/strategy_core/__pycache__"
    cache.mkdir()
    (cache / "runtime.cpython-313.pyc").write_bytes(b"generated")
    compatibility.verify_research_core_compatibility(subject, ql)
    (core / "src/strategy_core/calendar.parquet").write_bytes(b"unknown runtime asset")
    assert not git(core, "status", "--porcelain")
    with pytest.raises(ValueError, match="ignored.*runtime"):
        compatibility.verify_research_core_compatibility(subject, ql)
    (core / "src/strategy_core/calendar.parquet").unlink()
    (loaded / "__pycache__").mkdir()
    (loaded / "__pycache__/payload.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="loaded Core package differs"):
        compatibility.verify_research_core_compatibility(subject, ql)


@pytest.mark.parametrize("name", ["runtime.py", "calendar.json", "calendar.bin", "extra.pyd"])
def test_loaded_source_and_runtime_assets_are_verified(source, name):
    _core, ql, loaded, subject = source
    (loaded / name).write_bytes(b"altered")
    with pytest.raises(ValueError, match="loaded Core package differs"):
        compatibility.verify_research_core_compatibility(subject, ql)


def test_loaded_python_newlines_normalize_but_asset_bytes_do_not(source):
    _core, ql, loaded, subject = source
    proof = compatibility.verify_research_core_compatibility(subject, ql)
    path = loaded / "runtime.py"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert compatibility.verify_research_core_compatibility(subject, ql) == proof
    path = loaded / "calendar.json"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="calendar.json"):
        compatibility.verify_research_core_compatibility(subject, ql)


def test_saved_dirty_source_identity_cannot_be_relabelled_as_clean(source):
    core, ql, _loaded, subject = source
    path = core / "src/strategy_core/runtime.py"
    original = path.read_bytes()
    path.write_bytes(b"def value():\n    return 2\n")
    dirty_commit, dirty_identity = strategy_core_source_identity(repository_root=core)
    dirty_subject = rebind(subject, dirty_commit, dirty_identity)
    path.write_bytes(original)
    with pytest.raises(ValueError, match="saved Core source identity"):
        compatibility.verify_research_core_compatibility(dirty_subject, ql)


def test_saved_clean_windows_crlf_identity_is_reconstructed_from_commit(source):
    core, ql, loaded, subject = source
    git(core, "config", "core.autocrlf", "true")
    for path in (core / "src/strategy_core").iterdir():
        raw = path.read_bytes()
        if path.suffix in {".py", ".json"}:
            path.write_bytes(raw.replace(b"\n", b"\r\n"))
        (loaded / path.name).write_bytes(path.read_bytes())
    git(core, "add", "--renormalize", "src/strategy_core")
    git(core, "diff", "--cached", "--exit-code")
    assert not git(core, "status", "--porcelain")
    saved_commit, identity = strategy_core_source_identity(repository_root=core)
    subject = rebind(subject, saved_commit, identity)
    proof = compatibility.verify_research_core_compatibility(subject, ql)
    assert proof.payload.saved_checkout_representation == "utf8_crlf_checkout"


def test_postapproval_environment_and_doc_commit_changes_change_proof(source, monkeypatch):
    core, ql, _loaded, subject = source
    approved = compatibility.verify_research_core_compatibility(subject, ql)
    monkeypatch.setattr(compatibility, "_runtime_environment", lambda _: {"deps": [["numpy", "2"]]})
    changed_environment = compatibility.verify_research_core_compatibility(subject, ql)
    assert changed_environment.proof_id != approved.proof_id
    assert json.loads(changed_environment.payload.runtime_environment_json)["deps"] == [
        ["numpy", "2"]
    ]
    (core / "README.md").write_text("another documentation commit\n", encoding="utf-8")
    commit(core, "fixture later documentation")
    assert (
        compatibility.verify_research_core_compatibility(subject, ql).proof_id
        != changed_environment.proof_id
    )


def test_missing_saved_commit_fails_closed(source):
    _core, ql, _loaded, subject = source
    subject = rebind(subject, "f" * 40, "e" * 64)
    with pytest.raises(ValueError, match="cannot verify Git"):
        compatibility.verify_research_core_compatibility(subject, ql)


def test_new_ignore_rule_cannot_hide_unknown_runtime_input_outside_package(source):
    core, ql, _loaded, subject = source
    with (core / ".gitignore").open("a", encoding="utf-8") as handle:
        handle.write("/runtime_config.json\n")
    commit(core, "fixture ignore update")
    (core / "runtime_config.json").write_bytes(b'{"new_setting": true}')
    assert not git(core, "status", "--porcelain")
    with pytest.raises(ValueError, match="ignored Core file could affect runtime"):
        compatibility.verify_research_core_compatibility(subject, ql)


def test_verified_mixed_checkout_bytes_reproduce_exact_historical_identity(source):
    core, ql, _loaded, subject = source
    git(core, "config", "core.autocrlf", "true")
    (core / "src/strategy_core/runtime.py").write_bytes(b"def value():\r\n    return 1\n")
    git(core, "add", "--renormalize", "src/strategy_core")
    git(core, "diff", "--cached", "--exit-code")
    assert not git(core, "status", "--porcelain")
    saved_commit, saved_identity = strategy_core_source_identity(repository_root=core)
    subject = rebind(subject, saved_commit, saved_identity)
    proof = compatibility.verify_research_core_compatibility(subject, ql)
    assert proof.payload.saved_checkout_representation == "verified_current_checkout_bytes"
    assert proof.payload.saved_source_identity == saved_identity


def test_nonpython_asset_bytes_must_match_historical_witness(source):
    core, ql, loaded, subject = source
    git(core, "config", "core.autocrlf", "true")
    name = "src/strategy_core/calendar.json"
    (core / name).write_bytes((core / name).read_bytes().replace(b"\n", b"\r\n"))
    (loaded / "calendar.json").write_bytes((core / name).read_bytes())
    git(core, "add", "--renormalize", name)
    git(core, "diff", "--cached", "--exit-code")
    with pytest.raises(ValueError, match="saved Core runtime asset bytes differ"):
        compatibility.verify_research_core_compatibility(subject, ql)


def test_current_raw_build_metadata_is_frozen_even_when_git_conversion_is_valid(source):
    core, ql, _loaded, subject = source
    approved = compatibility.verify_research_core_compatibility(subject, ql)
    git(core, "config", "core.autocrlf", "true")
    (core / ".python-version").write_bytes(b"3.13.1\r\n")
    git(core, "add", "--renormalize", ".python-version")
    git(core, "diff", "--cached", "--exit-code")
    changed = compatibility.verify_research_core_compatibility(subject, ql)
    assert changed.payload.current_commit == approved.payload.current_commit
    assert (
        changed.payload.current_checkout_tree_sha256
        != approved.payload.current_checkout_tree_sha256
    )
    assert changed.proof_id != approved.proof_id
