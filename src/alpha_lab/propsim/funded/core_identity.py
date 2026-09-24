"""The exact Strategy-Core source a comparison imports.

A v2 comparison plan freezes the Core commit plus a SHA-256 of every
uncommitted change under the checkout's ``src`` (tracked diffs and untracked
files). The worker recomputes it from the imported ``strategy_core`` package and
refuses to run on any other source.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

__all__ = ["core_source_identity", "source_identity_at", "untracked_source_files"]


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True,  # noqa: S603, S607
                          capture_output=True).stdout


def source_identity_at(root: Path) -> dict[str, str]:
    """Commit, branch and uncommitted-source hash of one Core checkout (not imported)."""

    root = Path(root).resolve()
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    digest = hashlib.sha256()
    digest.update(_git(root, "diff", "--binary", "HEAD", "--", "src"))
    for rel in untracked_source_files(root):
        digest.update(rel.encode())
        digest.update((root / rel).read_bytes())
    return {"root": str(root), "base_commit": head, "branch": branch,
            "patch_sha256": digest.hexdigest()}


def untracked_source_files(root: Path) -> list[str]:
    untracked = _git(Path(root), "ls-files", "--others", "--exclude-standard", "--", "src")
    return [rel for rel in sorted(untracked.decode().split())
            if not (rel.endswith(".pyc") or "__pycache__" in rel)]


def core_source_identity() -> dict[str, str]:
    """Identity of the Strategy-Core this process imports."""

    import strategy_core

    return source_identity_at(Path(strategy_core.__file__).resolve().parents[2])
