"""Portable location and byte-exact verification of Quant-Lab's current Core."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "research/core/current.json"
RESEARCH_ARTIFACTS = ROOT.parent / "Claude-Quant-Lab-Research-Artifacts"


def manifest() -> dict:
    expected = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = [
        match.group(1)
        for dependency in metadata["project"]["dependencies"]
        if (match := re.fullmatch(
            r"strategy-core @ git\+https://github.com/thealgochef/Strategy-Core\.git@([0-9a-f]{40})",
            dependency,
        ))
    ]
    if pins != [expected["target_commit"]]:
        raise ValueError(
            "Installed dependency and current Core source manifest must pin one commit"
        )
    return expected


CORE_COMMIT = manifest()["target_commit"]
CORE_SOURCE = manifest()["source_identity"]
DEFAULT_CORE = RESEARCH_ARTIFACTS / "ifsm-research-core" / CORE_COMMIT


def select_core(explicit: Path | None = None) -> Path:
    """Use the current pin; historical checkouts are never implicit fallbacks."""
    configured = explicit or os.environ.get("IFSM_RESEARCH_CORE")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_CORE


def git(core: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=core, check=True, capture_output=True, timeout=180,
    ).stdout


def source_bytes(blob: bytes, entry: dict) -> bytes:
    """Restore the recorded source bytes, including the original mixed EOL file."""
    mode = entry["line_endings"]
    if mode == "git":
        return blob
    if mode == "crlf":
        return blob.replace(b"\n", b"\r\n")
    if mode != "mixed":
        raise ValueError(f"Unsupported line-ending policy: {mode}")
    crlf_lines = set(entry["crlf_lines"])
    return b"".join(
        line[:-1] + b"\r\n" if index in crlf_lines and line.endswith(b"\n") else line
        for index, line in enumerate(blob.splitlines(keepends=True), 1)
    )


def verify_core(core: Path, expected: dict | None = None) -> dict:
    """Match the replay identity algorithm without importing either runtime."""
    expected = expected or manifest()
    core = core.resolve()
    head = git(core, "rev-parse", "HEAD").decode().strip()
    if head != expected["target_commit"]:
        raise ValueError(f"Research Core commit differs: {head}")
    scope = core / "src/strategy_core"
    paths = sorted(
        path.relative_to(core).as_posix()
        for path in scope.rglob("*")
        if path.is_file()
        and not {"__pycache__", ".pytest_cache", ".ruff_cache"}.intersection(path.parts)
    )
    entries = {entry["path"]: entry for entry in expected["source_files"]}
    if paths != sorted(entries):
        raise ValueError("Research Core source inventory differs from its manifest")
    tree = hashlib.sha256()
    for relative in paths:
        content = (core / relative).read_bytes()
        digest = hashlib.sha256(content)
        if digest.hexdigest() != entries[relative]["sha256"]:
            raise ValueError(f"Research Core source bytes differ: {relative}")
        tree.update(relative.encode() + b"\0" + digest.digest() + b"\0")
    # Git's porcelain bytes are ASCII for this frozen, clean source inventory.
    status = git(core, "status", "--porcelain=v1", "--untracked-files=all", "--",
                 "src/strategy_core").decode().replace("\r\n", "\n")
    identity_payload = {
        "name": "strategy-core", "head": head,
        "dirty_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "source_tree_hash": tree.hexdigest(),
    }
    identity = hashlib.sha256(json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()).hexdigest()
    if identity != expected["source_identity"]:
        raise ValueError(f"Research Core source identity differs: {identity}")
    return {"core_path": str(core), "core_commit": head, "core_source_identity": identity}
