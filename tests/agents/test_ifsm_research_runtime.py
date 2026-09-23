"""Portable research setup guards; no market data or economic executions."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_tracked_artifacts as tracked  # noqa: E402
import ifsm_research_runtime as runtime  # noqa: E402
import prepare_ifsm_research_core as bootstrap  # noqa: E402
import run_ifsm_research_ui as launcher  # noqa: E402


def test_explicit_core_beats_environment_and_legacy_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv("IFSM_RESEARCH_CORE", str(tmp_path / "environment"))
    assert runtime.select_core(tmp_path / "explicit") == tmp_path / "explicit"
    assert runtime.select_core() == tmp_path / "environment"


def test_legacy_checkout_remains_available_until_external_default_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("IFSM_RESEARCH_CORE", raising=False)
    current, legacy = tmp_path / "external", tmp_path / "legacy"
    monkeypatch.setattr(runtime, "DEFAULT_CORE", current)
    monkeypatch.setattr(runtime, "LEGACY_CORE", legacy)
    assert runtime.select_core() == current
    legacy.mkdir()
    assert runtime.select_core() == legacy
    current.mkdir()
    assert runtime.select_core() == current


def test_launcher_propagates_explicit_core_to_probe_and_workers(tmp_path):
    package = tmp_path / "src/strategy_core"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    env = launcher.environment(tmp_path)
    assert env["IFSM_RESEARCH_CORE"] == str(tmp_path)
    assert str(tmp_path / "src") in env["PYTHONPATH"]
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_missing_launcher_source_gives_reproducible_setup_command(tmp_path):
    with pytest.raises(ValueError, match="prepare_ifsm_research_core.py"):
        launcher.environment(tmp_path / "missing")


def test_optimized_python_cannot_bypass_selected_import_check(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(map(str, (runtime.ROOT / "src", SCRIPTS)))
    env["IFSM_RESEARCH_CORE"] = str(tmp_path / "different-checkout")
    env["PYTHONOPTIMIZE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with pytest.raises(subprocess.CalledProcessError) as error:
        launcher.verify_runtime(env)
    assert "outside the selected research checkout" in error.value.stderr


def test_preparation_refuses_repository_destinations_without_creating_files():
    with pytest.raises(ValueError, match="outside the Quant-Lab repository"):
        bootstrap.prepare(bootstrap.ROOT / "must-never-be-created")


@pytest.mark.parametrize("component", ["reports", "RePoRtS", "REPORTS"])
def test_preparation_refuses_external_reports_folders(tmp_path, component):
    destination = tmp_path / component / "must-never-be-created"
    with pytest.raises(ValueError, match="inside reports folders"):
        bootstrap.prepare(destination)
    assert not destination.exists()


@pytest.mark.parametrize("reports_in", ["requested", "resolved"])
def test_reports_alias_cannot_bypass_destination_policy(monkeypatch, tmp_path, reports_in):
    # Simulate path resolution without requiring Windows symlink privileges.
    requested = tmp_path / ("reports" if reports_in == "requested" else "alias") / "core"
    resolved = tmp_path / ("reports" if reports_in == "resolved" else "external") / "core"
    original_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        if path == requested:
            return resolved
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="inside reports folders"):
        bootstrap.prepare(requested)
    assert not requested.exists() and not resolved.exists()


def test_changed_bundle_is_rejected_before_destination_creation(monkeypatch, tmp_path):
    bundle = tmp_path / "delta.bundle"
    bundle.write_bytes(b"changed")
    expected = {"bundle_file": bundle.name, "bundle_sha256": "0" * 64}
    monkeypatch.setattr(bootstrap, "manifest", lambda: expected)
    monkeypatch.setattr(bootstrap, "MANIFEST_PATH", tmp_path / "manifest.json")
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="bundle checksum differs"):
        bootstrap.prepare(destination)
    assert not destination.exists()


def test_existing_destination_is_never_repaired_or_overwritten(monkeypatch, tmp_path):
    sentinel = tmp_path / "owner-evidence.txt"
    sentinel.write_bytes(b"preserve me")

    def reject_existing(path, expected):
        raise ValueError("existing source differs")

    monkeypatch.setattr(bootstrap, "verify_core", reject_existing)
    monkeypatch.setattr(bootstrap, "git", lambda *_: pytest.fail("Existing checkout was modified"))
    with pytest.raises(ValueError, match="existing source differs"):
        bootstrap.prepare(tmp_path)
    assert sentinel.read_bytes() == b"preserve me"


@pytest.fixture
def frozen_source(monkeypatch, tmp_path):
    relative = "src/strategy_core/__init__.py"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"first\r\nsecond\nthird\r\n")
    digest = hashlib.sha256(path.read_bytes())
    tree = hashlib.sha256(relative.encode() + b"\0" + digest.digest() + b"\0").hexdigest()
    expected = {
        "target_commit": "a" * 40,
        "source_files": [{"path": relative, "sha256": digest.hexdigest()}],
    }
    payload = {
        "name": "strategy-core", "head": expected["target_commit"],
        "dirty_status_sha256": hashlib.sha256(b"").hexdigest(), "source_tree_hash": tree,
    }
    expected["source_identity"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    monkeypatch.setattr(runtime, "git", lambda _, *args: (
        (expected["target_commit"] + "\n").encode() if args[0] == "rev-parse" else b""
    ))
    return tmp_path, path, expected


def test_mixed_historical_bytes_preserve_exact_identity(frozen_source):
    root, path, expected = frozen_source
    entry = {"line_endings": "mixed", "crlf_lines": [1, 3]}
    assert runtime.source_bytes(b"first\nsecond\nthird\n", entry) == path.read_bytes()
    actual = runtime.verify_core(root, expected)
    assert actual["core_source_identity"] == expected["source_identity"]


@pytest.mark.parametrize("tamper", ["bytes", "missing", "extra", "dirty", "head"])
def test_source_drift_is_rejected(frozen_source, monkeypatch, tamper):
    root, path, expected = frozen_source
    if tamper == "bytes":
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
    elif tamper == "missing":
        path.unlink()
    elif tamper == "extra":
        (path.parent / "extra.py").write_text("changed = True\n")
    else:
        monkeypatch.setattr(runtime, "git", lambda _, *args: (
            ("b" * 40 if tamper == "head" else expected["target_commit"]).encode()
            if args[0] == "rev-parse" else b" M src/strategy_core/__init__.py\n"
        ))
    with pytest.raises(ValueError, match="differs|differ"):
        runtime.verify_core(root, expected)


def test_required_research_job_errors_instead_of_skipping(monkeypatch):
    from strategy_core.strategies.ifvg_smc import section

    from tests.conftest import pytest_configure

    monkeypatch.setenv("IFSM_REQUIRE_RESEARCH_CORE", "1")
    monkeypatch.setattr(section, "IfvgSmcSection", SimpleNamespace(model_fields={}))
    with pytest.raises(pytest.UsageError, match="missing fields.*holding_policy"):
        pytest_configure(None)


def test_tracked_artifact_guard_is_scoped_and_case_insensitive():
    forbidden = ["reports/a.md", "nested/RePoRtS/archive.json", tracked.LOCAL_CATALOG]
    allowed = ["src/reports.py", "docs/reporting.md", "research/core/ifsm-daily-close.bundle"]
    assert tracked.forbidden_paths(forbidden + allowed) == sorted(forbidden)
