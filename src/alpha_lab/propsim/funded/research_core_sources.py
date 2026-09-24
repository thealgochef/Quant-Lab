"""Select and preserve the exact research Strategy-Core source a comparison plan froze.

A version-2 comparison plan records ``CoreSourceRef`` (base commit + SHA-256 of
every uncommitted change under the checkout's ``src``). The job launcher uses
:func:`find_core_checkout` to find the local checkout with exactly that identity
and starts the worker on it; nothing is chosen by name or recency, and no pin is
changed. :func:`snapshot_core_source` keeps an immutable internal copy of the
uncommitted source (patch plus untracked files, with hashes) so the frozen
source can be restored even if the working checkout later changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from alpha_lab.propsim.funded.core_identity import (
    _git,
    source_identity_at,
    untracked_source_files,
)

__all__ = [
    "RESEARCH_ARTIFACTS",
    "candidate_checkouts",
    "find_core_checkout",
    "snapshot_core_source",
]

RESEARCH_ARTIFACTS = Path(__file__).resolve().parents[4].parent / \
    "Claude-Quant-Lab-Research-Artifacts"


def candidate_checkouts(extra: Iterable[Path] = (), *,
                        artifacts_root: Path | None = None) -> list[Path]:
    """Local Strategy-Core checkouts that may hold a plan's frozen source."""

    root = RESEARCH_ARTIFACTS if artifacts_root is None else Path(artifacts_root)
    found: list[Path] = [Path(p) for p in extra]
    if os.environ.get("IFSM_RESEARCH_CORE"):
        found.append(Path(os.environ["IFSM_RESEARCH_CORE"]))
    if root.is_dir():
        found += sorted(root.glob("strategy-core-*"))
        found += sorted((root / "ifsm-research-core").glob("*"))
    unique, seen = [], set()
    for path in found:
        key = str(path.resolve())
        if key not in seen and (path / "src" / "strategy_core" / "__init__.py").is_file():
            seen.add(key)
            unique.append(path.resolve())
    return unique


def find_core_checkout(core_ref: Any, candidates: Iterable[Path] | None = None
                       ) -> Path | None:
    """The checkout whose commit AND uncommitted-source hash equal the plan's."""

    wanted = (core_ref.base_commit, core_ref.patch_sha256)
    for path in candidate_checkouts() if candidates is None else candidates:
        try:
            identity = source_identity_at(path)
        except Exception:  # not a git checkout, git unavailable
            continue
        if (identity["base_commit"], identity["patch_sha256"]) == wanted:
            return Path(path)
    return None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_core_source(checkout: Path, archive_root: Path) -> dict[str, Any]:
    """Immutable copy of a checkout's uncommitted ``src`` changes (idempotent).

    Written to ``archive_root/<commit12>_<patch16>/``: ``source.patch`` (the
    tracked diff), ``untracked/<path>`` files and ``MANIFEST.json``. An existing
    snapshot with the same identity is verified and reused, never overwritten.
    """

    checkout = Path(checkout).resolve()
    identity = source_identity_at(checkout)
    name = f"{identity['base_commit'][:12]}_{identity['patch_sha256'][:16]}"
    final = Path(archive_root) / name
    patch = _git(checkout, "diff", "--binary", "HEAD", "--", "src")
    files = {"source.patch": patch}
    for rel in untracked_source_files(checkout):
        files[f"untracked/{rel}"] = (checkout / rel).read_bytes()
    manifest = {
        "kind": "strategy_core_research_source_snapshot",
        "base_commit": identity["base_commit"], "branch": identity["branch"],
        "patch_sha256": identity["patch_sha256"],
        "restore": "git checkout <base_commit>; git apply source.patch; copy untracked/ "
                   "into the checkout root",
        "files": [{"path": k, "bytes": len(v), "sha256": _sha(v)}
                  for k, v in sorted(files.items())],
    }
    if final.exists():
        saved = json.loads((final / "MANIFEST.json").read_text(encoding="utf-8"))
        same = saved["files"] == manifest["files"] and all(
            _sha((final / f["path"]).read_bytes()) == f["sha256"] for f in saved["files"])
        if not same:
            raise FileExistsError(f"snapshot {name} exists with different content")
        return {**saved, "path": str(final), "reused": True}
    staging = final.with_name(f".staging_{name}")
    if staging.exists():
        shutil.rmtree(staging)
    for rel, data in files.items():
        target = staging / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                           encoding="utf-8")
    os.replace(staging, final)
    return {**manifest, "path": str(final), "reused": False}
