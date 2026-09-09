"""Validate the R6.1 browser-smoke manifest v2 (plan §6.K).

Recomputes every screenshot / log sha256 + byte size, the committed-tree
digest over the digest globs from ``git ls-tree -r <commit>`` blob ids, checks
that the bound commit is a real commit of this repository, that the server
log carries the harness marker lines and reports its traceback /
deprecation-warning counts, and that every bound evidence id is a 64-hex
identity. Exit 1 on any mismatch.

Run from the repo root:

    python QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R6.1/verify_browser_manifest.py \
        [QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R6.1/browser-smoke/MANIFEST.json]
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_MARKER = "[r61-smoke]"


def _repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, check=True, text=True
    )
    return Path(out.stdout.strip())


def _matches(relative: str, pattern: str) -> bool:
    """fnmatch with ``**/`` meaning "zero or more directories" (so top-level
    files of the prefix match ``**/*.py`` too)."""

    if fnmatch.fnmatch(relative, pattern):
        return True
    return pattern.startswith("**/") and fnmatch.fnmatch(relative, pattern[3:])


def committed_tree_digest(commit: str, globs: list[list[str]]) -> str:
    """sha256 over the sorted ``path\\0blob`` pairs of every committed file
    matching the digest globs at ``commit`` (never the worktree)."""

    out = subprocess.run(
        ["git", "ls-tree", "-r", commit], capture_output=True, check=True, text=True
    )
    entries: list[tuple[str, str]] = []
    for line in out.stdout.splitlines():
        meta, path = line.split("\t", 1)
        blob = meta.split()[2]
        for prefix, pattern in globs:
            if path.startswith(prefix.rstrip("/") + "/") and _matches(
                path[len(prefix.rstrip("/")) + 1 :], pattern
            ):
                entries.append((path, blob))
                break
    digest = hashlib.sha256()
    for path, blob in sorted(set(entries)):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(blob.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    manifest_path = (
        Path(argv[1])
        if len(argv) > 1
        else Path(__file__).resolve().parent / "browser-smoke" / "MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    folder = manifest_path.parent
    if manifest.get("manifest_schema_version") != 2:
        problems.append("manifest_schema_version must be 2")
    commit = str(manifest.get("commit", ""))
    commit_ok = False
    if not _HEX40.match(commit):
        problems.append("commit is not a 40-hex git commit id")
    else:
        probe = subprocess.run(
            ["git", "cat-file", "-t", commit], capture_output=True, text=True
        )
        commit_ok = probe.stdout.strip() == "commit"
        if not commit_ok:
            problems.append(f"commit {commit[:12]} is not a commit of this repository")
    globs = manifest.get("digest_globs", [])
    if not globs:
        problems.append("digest_globs missing")
    elif commit_ok:
        expected = committed_tree_digest(commit, [list(g) for g in globs])
        if manifest.get("scoped_tree_digest") != expected:
            problems.append(
                "scoped_tree_digest does not equal the committed-tree digest "
                f"({str(manifest.get('scoped_tree_digest'))[:12]} != {expected[:12]})"
            )
    files = manifest.get("files", {})
    if not files:
        problems.append("files is empty")
    for name, facts in files.items():
        path = folder / name
        if not path.is_file():
            problems.append(f"missing file {name}")
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != facts.get("sha256"):
            problems.append(f"sha256 mismatch for {name}")
        if len(data) != int(facts.get("bytes", -1)):
            problems.append(f"byte size mismatch for {name}")
    screenshots = manifest.get("screenshots", {})
    for name in screenshots:
        if name not in files:
            problems.append(f"screenshot {name} is not in files")
    log_name = manifest.get("server_log", "_smoke_server.log")
    log_path = folder / log_name
    if log_name not in files:
        problems.append("the server log must be a manifest-bound file")
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        markers = [line for line in text.splitlines() if _MARKER in line]
        expected_markers = manifest.get("expected_marker_count")
        if not markers:
            problems.append("server log carries no harness marker lines")
        elif expected_markers is not None and len(markers) < int(expected_markers):
            problems.append(
                f"server log carries {len(markers)} marker lines, expected ≥ {expected_markers}"
            )
        tracebacks = text.count("Traceback")
        deprecations = text.count("DeprecationWarning")
        counts = manifest.get("server_log_counts", {})
        if counts.get("tracebacks") != tracebacks:
            problems.append(
                f"server log has {tracebacks} Traceback(s); manifest says "
                f"{counts.get('tracebacks')}"
            )
        if counts.get("deprecation_warnings") != deprecations:
            problems.append(
                f"server log has {deprecations} DeprecationWarning(s); manifest says "
                f"{counts.get('deprecation_warnings')}"
            )
    for key, value in manifest.get("bound_evidence", {}).items():
        if isinstance(value, str) and not _HEX64.match(value):
            problems.append(f"bound evidence {key} is not a 64-hex identity")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, str) and sub_value and not _HEX64.match(sub_value):
                    problems.append(f"bound evidence {key}.{sub_key} is not a 64-hex identity")
    if manifest.get("commit_is_head_at_capture") is not True:
        problems.append("commit_is_head_at_capture must be true")
    if manifest.get("worktree_dirty_in_scope") is not False:
        problems.append("worktree_dirty_in_scope must be false (scoped files committed)")
    if problems:
        for problem in problems:
            print("FAIL:", problem)
        return 1
    print(
        f"OK: manifest v2 bound to commit {commit[:12]}, {len(files)} file(s), "
        f"{len(screenshots)} screenshot(s), committed-tree digest verified"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
