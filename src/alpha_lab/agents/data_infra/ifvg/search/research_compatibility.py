"""Auditable, documentation-only Core compatibility for immutable saved replays.

The historical identity recorded the package source scope, not the historical
dependency environment or dirty files outside that scope. This proof preserves
that limitation: it proves the recorded clean package against its Git commit,
compares every non-documentation committed file, and binds today's loaded
package and environment. A new context capture must still pass table neutrality.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import ClassVar, Literal

from pydantic import Field

from .identities import (
    SHA256_PATTERN,
    CoreStrategyReplayIdentity,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
    strategy_core_source_identity,
)

COMPATIBILITY_STORE = "research_core_compatibility"
COMPATIBILITY_POLICY = "saved_core_clean_tree_documentation_only_v1"
_PACKAGE_PREFIX = "src/strategy_core/"
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_DOCUMENT_FILES = frozenset(
    {
        "README.md",
        "MIGRATION.md",
        "V3_COMPATIBILITY_MATRIX.md",
        "validation/README.md",
        ".gitignore",
    }
)


class CompatibilityFile(FrozenContract):
    path: str
    mode: str
    sha256: str = Field(pattern=SHA256_PATTERN)


class ResearchCoreCompatibilityPayload(FrozenContract):
    schema_version: Literal[1] = 1
    policy_id: Literal["saved_core_clean_tree_documentation_only_v1"] = COMPATIBILITY_POLICY
    subject_id: str = Field(pattern=SHA256_PATTERN)
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    saved_commit: str
    saved_source_identity: str = Field(pattern=SHA256_PATTERN)
    saved_package_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    saved_checkout_representation: Literal[
        "git_blob_bytes", "utf8_crlf_checkout", "verified_current_checkout_bytes"
    ]
    current_commit: str
    current_git_tree: str
    current_checkout_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    current_source_identity: str = Field(pattern=SHA256_PATTERN)
    changed_documentation_paths: tuple[str, ...]
    runtime_files: tuple[CompatibilityFile, ...]
    runtime_tree_sha256: str = Field(pattern=SHA256_PATTERN)
    loaded_package_files: tuple[CompatibilityFile, ...]
    loaded_package_sha256: str = Field(pattern=SHA256_PATTERN)
    runtime_environment_json: str
    historical_environment_policy: Literal["not_recorded_by_saved_replay"] = (
        "not_recorded_by_saved_replay"
    )
    historical_off_scope_dirty_policy: Literal["not_recorded_by_saved_replay"] = (
        "not_recorded_by_saved_replay"
    )
    required_neutrality_policy: Literal["all_accepted_core_tables_exact_v1"] = (
        "all_accepted_core_tables_exact_v1"
    )


class ResearchCoreCompatibilityProof(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "proof_id"
    proof_id: str = Field(pattern=SHA256_PATTERN)
    payload: ResearchCoreCompatibilityPayload


def _git(root: Path, *args: str, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "--no-replace-objects", *args], cwd=root, input=stdin, capture_output=True
    )
    if result.returncode:
        raise ValueError(f"Core compatibility cannot verify Git {args[0]}")
    return result.stdout


def _documentation_path(name: str) -> bool:
    # No broad '*.md' exemption: runtime assets and unknown root files stay protected.
    return name in _DOCUMENT_FILES or (
        name.startswith("docs/") and PurePosixPath(name).suffix == ".md"
    )


def _git_tree(root: Path, commit: str) -> dict[str, tuple[str, str]]:
    if not _SHA1.fullmatch(commit):
        raise ValueError("Core compatibility requires a full pinned Git commit")
    if _git(root, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip() != commit:
        raise ValueError("Core compatibility commit does not resolve exactly")
    entries = {}
    for item in _git(root, "ls-tree", "-r", "-z", commit).split(b"\0"):
        if not item:
            continue
        header, raw_path = item.split(b"\t", 1)
        mode, kind, object_id = header.decode("ascii").split()
        name = raw_path.decode("utf-8")
        if (
            kind != "blob"
            or mode not in {"100644", "100755"}
            or "\n" in name
            or "\r" in name
            or "\\" in name
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
        ):
            raise ValueError(f"unsupported Core Git path or file type: {name}")
        entries[name] = (mode, object_id)
    if not any(name.startswith(_PACKAGE_PREFIX) for name in entries):
        raise ValueError("pinned Core Git tree has no package source")
    return entries


def _blobs(root: Path, entries: dict[str, tuple[str, str]]) -> dict[str, bytes]:
    names = sorted(entries)
    query = b"".join(entries[name][1].encode("ascii") + b"\n" for name in names)
    result = _git(root, "cat-file", "--batch", stdin=query)
    offset = 0
    output = {}
    for name in names:
        end = result.index(b"\n", offset)
        object_id, kind, size = result[offset:end].split()
        if kind != b"blob" or object_id.decode() != entries[name][1]:
            raise ValueError("Core Git blob identity mismatch")
        offset = end + 1
        length = int(size)
        output[name] = result[offset : offset + length]
        offset += length
        if result[offset : offset + 1] != b"\n":
            raise ValueError("Core Git blob response is truncated")
        offset += 1
    if offset != len(result):
        raise ValueError("unexpected Core Git blob response")
    return output


def _lf_text(data: bytes) -> bytes:
    if b"\0" in data:
        return data
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return data.replace(b"\r\n", b"\n")


def _crlf_text(data: bytes) -> bytes:
    if b"\0" in data:
        return data
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def _source_tree_digest(blobs: dict[str, bytes]) -> str:
    # Exact reproduction of manifest.source_tree_hash's original byte protocol.
    digest = hashlib.sha256()
    for name, data in sorted(blobs.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(data).digest() + b"\0")
    return digest.hexdigest()


def _saved_source_match(
    commit: str, identity: str, blobs: dict[str, bytes], verified_checkout: dict[str, bytes]
):
    package = {name: data for name, data in blobs.items() if name.startswith(_PACKAGE_PREFIX)}
    variants = (
        ("git_blob_bytes", package),
        (
            "utf8_crlf_checkout",
            {name: _crlf_text(data) for name, data in package.items()},
        ),
        # Existing Windows checkouts can contain mixed LF/CRLF even within files.
        # The caller first proves these exact bytes normalize to the pinned tree.
        # Their aggregate must then EXACTLY reproduce the historical identity.
        ("verified_current_checkout_bytes", verified_checkout),
    )
    for representation, files in variants:
        tree_hash = _source_tree_digest(files)
        expected = canonical_contract_sha256(
            {
                "name": "strategy-core",
                "head": commit,
                "dirty_status_sha256": hashlib.sha256(b"").hexdigest(),
                "source_tree_hash": tree_hash,
            }
        )
        if identity == expected:
            if any(
                not name.endswith(".py") and data != verified_checkout[name]
                for name, data in files.items()
            ):
                raise ValueError("saved Core runtime asset bytes differ in the current checkout")
            return representation, tree_hash
    raise ValueError("saved Core source identity is not the pinned clean package tree")


def _known_generated_file(name: str) -> bool:
    path = PurePosixPath(name)
    # These paths are local tooling/audit output, never a general ignore-rule exemption.
    if name in {".claude/settings.local.json", "CLEANUP_SC_REVIEW.txt"}:
        return True
    if path.name in {".DS_Store", "Thumbs.db", "Desktop.ini"}:
        return True
    if "__pycache__" in path.parts and path.suffix == ".pyc":
        return True
    if name == ".coverage" or name.startswith(".coverage."):
        return True
    return any(
        name.startswith(prefix)
        for prefix in (
            ".pytest_cache/",
            ".ruff_cache/",
            ".mypy_cache/",
            ".pytest_tmp/",
            "htmlcov/",
            "validation/_out/",
        )
    )


def _assert_clean_checkout(root: Path, entries, blobs) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("Core compatibility requires a clean checkout without untracked files")
    ignored = _git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z")
    for raw_name in ignored.split(b"\0"):
        if raw_name and not _known_generated_file(raw_name.decode("utf-8")):
            raise ValueError(
                "unclassified ignored Core file could affect runtime: " + raw_name.decode("utf-8")
            )
    # Read every tracked file as well: assume-unchanged/skip-worktree cannot conceal edits.
    for name in entries:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Core checkout file is absent or linked: {name}")
        actual = path.read_bytes()
        expected = blobs[name]
        source_text = name.endswith(".py") or _documentation_path(name)
        matches = actual == expected or (source_text and _lf_text(actual) == _lf_text(expected))
        if not matches and actual == _crlf_text(expected):
            matches = _git_crlf_checkout(root, name)
        if not matches:
            raise ValueError(f"Core checkout differs from its pinned commit: {name}")


def _git_crlf_checkout(root: Path, name: str) -> bool:
    """Permit non-source CRLF conversion only when Git's checkout policy explains it."""
    raw = _git(root, "check-attr", "--cached", "-z", "text", "eol", "filter", "--", name)
    cells = raw.decode("utf-8").split("\0")
    attributes = {cells[index + 1]: cells[index + 2] for index in range(0, len(cells) - 1, 3)}
    if (
        attributes.get("filter") not in {"unspecified", "unset"}
        or attributes.get("text") == "unset"
    ):
        return False
    if attributes.get("eol") == "lf":
        return False
    if attributes.get("eol") == "crlf":
        return True
    result = subprocess.run(
        ["git", "config", "--get", "core.autocrlf"], cwd=root, capture_output=True, text=True
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _package_inventory(root: Path) -> tuple[CompatibilityFile, ...]:
    output = []
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Core package contains a linked path: {name}")
        if not path.is_file():
            continue
        if "__pycache__" in path.relative_to(root).parts and path.suffix == ".pyc":
            continue
        data = path.read_bytes()
        # Python source line endings differ between installed wheels and Windows checkouts.
        if path.suffix == ".py":
            data = _lf_text(data)
        output.append(
            CompatibilityFile(
                path=name, mode="package_file", sha256=hashlib.sha256(data).hexdigest()
            )
        )
    if not any(item.path == "__init__.py" for item in output):
        raise ValueError("loaded Core package has no inspectable __init__.py")
    return tuple(output)


def _loaded_package_root() -> Path:
    import strategy_core  # noqa: PLC0415

    if not strategy_core.__file__:
        raise ValueError("loaded Core has no inspectable package source")
    return Path(strategy_core.__file__).resolve().parent


def _runtime_environment(loaded_root: Path) -> dict:
    distribution = importlib.metadata.distribution("strategy-core")
    metadata_hashes = {
        name: hashlib.sha256(value.encode("utf-8")).hexdigest()
        for name in ("METADATA", "WHEEL", "direct_url.json")
        if (value := distribution.read_text(name)) is not None
    }
    return {
        "python": sys.version,
        "implementation": sys.implementation.name,
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "loaded_core_root": str(loaded_root),
        "core_distribution_version": distribution.version,
        "core_distribution_metadata_sha256": metadata_hashes,
        "installed_distributions": sorted(
            (dist.metadata["Name"], dist.version)
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        ),
    }


def verify_research_core_compatibility(
    subject, repo_root: Path, *, loaded_package_root: Path | None = None
) -> ResearchCoreCompatibilityProof:
    """Recompute a proof from current files; never memoize mutable source state."""
    core = CoreStrategyReplayIdentity.model_validate_json(subject.core_envelope_json)
    root = (Path(repo_root).parent / "Strategy-Core").resolve()
    saved_commit = core.payload.strategy_core_commit
    current_commit = _git(root, "rev-parse", "HEAD").decode().strip()
    saved = _git_tree(root, saved_commit)
    current = _git_tree(root, current_commit)
    changed = tuple(
        sorted(
            name for name in saved.keys() | current.keys() if saved.get(name) != current.get(name)
        )
    )
    forbidden = [name for name in changed if not _documentation_path(name)]
    if forbidden:
        raise ValueError("Core runtime/build/dependency tree changed: " + ", ".join(forbidden[:8]))
    saved_blobs = _blobs(root, saved)
    current_blobs = _blobs(root, current)
    _assert_clean_checkout(root, current, current_blobs)
    representation, saved_tree_hash = _saved_source_match(
        saved_commit,
        core.payload.strategy_core_source_identity,
        saved_blobs,
        {name: (root / name).read_bytes() for name in current if name.startswith(_PACKAGE_PREFIX)},
    )
    runtime_files = tuple(
        CompatibilityFile(path=name, mode=current[name][0], sha256=hashlib.sha256(data).hexdigest())
        for name, data in sorted(current_blobs.items())
        if not _documentation_path(name)
    )
    checkout_root = root / "src/strategy_core"
    expected_names = {
        name.removeprefix(_PACKAGE_PREFIX) for name in current if name.startswith(_PACKAGE_PREFIX)
    }
    checkout_package = _package_inventory(checkout_root)
    if {item.path for item in checkout_package} != expected_names:
        raise ValueError("Core checkout contains untracked or ignored package runtime files")
    loaded_root = Path(loaded_package_root or _loaded_package_root()).resolve()
    loaded = _package_inventory(loaded_root)
    if loaded != checkout_package:
        expected = {item.path: item.sha256 for item in checkout_package}
        actual = {item.path: item.sha256 for item in loaded}
        mismatches = sorted(
            name
            for name in expected.keys() | actual.keys()
            if expected.get(name) != actual.get(name)
        )
        raise ValueError(
            "loaded Core package differs from the pinned source: " + ", ".join(mismatches[:8])
        )
    actual_commit, actual_source = strategy_core_source_identity(repository_root=root)
    if actual_commit != current_commit:
        raise ValueError("Core commit changed during compatibility verification")
    environment = _runtime_environment(loaded_root)
    checkout_tree_hash = _source_tree_digest({name: (root / name).read_bytes() for name in current})
    # Detect ordinary concurrent source edits before returning the frozen proof.
    _assert_clean_checkout(root, current, current_blobs)
    if (
        _package_inventory(loaded_root) != loaded
        or _git(root, "rev-parse", "HEAD").decode().strip() != current_commit
        or _source_tree_digest({name: (root / name).read_bytes() for name in current})
        != checkout_tree_hash
        or strategy_core_source_identity(repository_root=root) != (actual_commit, actual_source)
    ):
        raise ValueError("Core source changed during compatibility verification")
    return ResearchCoreCompatibilityProof.from_payload(
        ResearchCoreCompatibilityPayload(
            subject_id=subject.subject_id,
            core_replay_id=subject.core_replay_id,
            saved_commit=saved_commit,
            saved_source_identity=core.payload.strategy_core_source_identity,
            saved_package_tree_sha256=saved_tree_hash,
            saved_checkout_representation=representation,
            current_commit=current_commit,
            current_git_tree=_git(root, "rev-parse", f"{current_commit}^{{tree}}").decode().strip(),
            current_checkout_tree_sha256=checkout_tree_hash,
            current_source_identity=actual_source,
            changed_documentation_paths=changed,
            runtime_files=runtime_files,
            runtime_tree_sha256=canonical_contract_sha256(runtime_files),
            loaded_package_files=loaded,
            loaded_package_sha256=canonical_contract_sha256(loaded),
            runtime_environment_json=json.dumps(environment, sort_keys=True, separators=(",", ":")),
        )
    )


def load_research_core_compatibility(root: Path, proof_id: str) -> ResearchCoreCompatibilityProof:
    from .store import load_verified_envelope  # noqa: PLC0415

    return load_verified_envelope(
        root, COMPATIBILITY_STORE, proof_id, ResearchCoreCompatibilityProof
    )
