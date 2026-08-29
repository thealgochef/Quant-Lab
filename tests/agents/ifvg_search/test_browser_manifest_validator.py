"""R6.1 workstream K — the browser-manifest v2 validator (plan §6.K, §9.2).

On a temporary git repository: a valid manifest passes; a flipped screenshot
byte, a wrong commit, a drifted committed-tree digest, a missing marker
line, and a mis-stated traceback count each fail (exit 1).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

_VALIDATOR = (
    Path(__file__).resolve().parents[3]
    / "QL-FSM-PROP-SEARCH-DASHBOARD"
    / "implementation-progress"
    / "R6.1"
    / "verify_browser_manifest.py"
)

pytestmark = pytest.mark.skipif(not _VALIDATOR.is_file(), reason="validator not present")


def _load():
    spec = importlib.util.spec_from_file_location("verify_browser_manifest", _VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, check=True, text=True, env=env
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src" / "lane").mkdir(parents=True)
    (repo / "scripts" / "ifvg_x.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "src" / "lane" / "a.py").write_text("a = 1\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "one")
    commit = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    return repo, commit


def _manifest(module, repo: Path, bound_commit: str, smoke: Path, **overrides) -> Path:
    smoke.mkdir(parents=True, exist_ok=True)
    shot = smoke / "01_shot.jpg"
    shot.write_bytes(b"\xff\xd8jpegbytes\xff\xd9")
    log = smoke / "_smoke_server.log"
    log.write_text("[r61-smoke] phase configure\n[r61-smoke] shot 01\n", encoding="utf-8")
    globs = [["scripts", "ifvg_*.py"], ["src/lane", "**/*.py"]]
    files = {
        name: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for name, path in (("01_shot.jpg", shot), ("_smoke_server.log", log))
    }
    manifest = {
        "manifest_schema_version": 2,
        "commit": bound_commit,
        "commit_is_head_at_capture": True,
        "worktree_dirty_in_scope": False,
        "digest_globs": globs,
        "scoped_tree_digest": module.committed_tree_digest(bound_commit, globs),
        "files": files,
        "screenshots": {"01_shot.jpg": "the configure phase"},
        "server_log": "_smoke_server.log",
        "expected_marker_count": 2,
        "server_log_counts": {"tracebacks": 0, "deprecation_warnings": 0},
        "bound_evidence": {"pipeline_semantic_id": "a" * 64, "regime": {"protocol_id": "b" * 64}},
        "captured_at_utc": "2026-08-29T00:00:00Z",
    }
    manifest.update(overrides)
    path = smoke / "MANIFEST.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def test_valid_manifest_passes(repo):
    module = _load()
    root, commit = repo
    path = _manifest(module, root, commit, root / "smoke")
    assert module.main(["verify", str(path)]) == 0


def test_byte_flip_wrong_commit_and_drift_fail(repo, capsys):
    module = _load()
    root, commit = repo
    path = _manifest(module, root, commit, root / "smoke")
    shot = root / "smoke" / "01_shot.jpg"
    original = shot.read_bytes()
    shot.write_bytes(original[:-1] + b"\x00")
    assert module.main(["verify", str(path)]) == 1
    assert "sha256 mismatch for 01_shot.jpg" in capsys.readouterr().out
    shot.write_bytes(original)
    # a commit that is not in the repository
    bad = _manifest(module, root, commit, root / "smoke2", commit="0" * 40)
    assert module.main(["verify", str(bad)]) == 1
    assert "not a commit of this repository" in capsys.readouterr().out
    # the scoped tree drifts: a second commit changes a digest-globbed file
    (root / "src" / "lane" / "a.py").write_text("a = 2\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "two")
    later = _git(root, "rev-parse", "HEAD")
    drifted = _manifest(module, root, commit, root / "smoke3", commit=later)
    assert module.main(["verify", str(drifted)]) == 1
    assert "scoped_tree_digest" in capsys.readouterr().out
    # the marker lines and the traceback count are asserted
    no_markers = _manifest(module, root, commit, root / "smoke4")
    (root / "smoke4" / "_smoke_server.log").write_text(
        "Traceback (most recent)\n", encoding="utf-8"
    )
    assert module.main(["verify", str(no_markers)]) == 1
    out = capsys.readouterr().out
    assert "sha256 mismatch for _smoke_server.log" in out
    assert "no harness marker lines" in out
    assert "Traceback(s)" in out
