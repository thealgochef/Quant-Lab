"""Reject local research outputs in Git, without reading market-data files.

The default checks the entire Git index (also used by CI). ``--staged`` checks
added/changed index entries for the local pre-commit hook. Both inspect Git blob
metadata, never working-tree bytes, so partially staged files cannot hide size.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CATALOG = "data/ifvg_datasets/replay_chart_catalog_v1.json"
MAX_BLOB_BYTES = 5 * 1024 * 1024
# The only archival source bundle intentionally versioned; see research/core/README.md.
ARCHIVE_EXCEPTIONS = {"research/core/ifsm-daily-close.bundle"}
ARTIFACT_SUFFIXES = {
    ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz", ".tgz", ".zst",
    ".bundle", ".pack", ".whl", ".egg", ".cbm", ".parquet", ".csv", ".h5",
    ".hdf5", ".pkl", ".pickle", ".joblib", ".npz", ".npy", ".feather", ".arrow",
    ".sqlite", ".sqlite3", ".db", ".duckdb", ".pyc", ".pyo",
}
WORKING_DIRECTORIES = {
    "node_modules", "__pycache__", "catboost_info", ".venv", "venv", ".pytest_cache",
    ".ruff_cache", "staging", "extracted", "extractions", "source_review", "source-review",
    "clean_verification", "clean-verification", "working_store", "working-store",
    "working_stores", "working-stores",
}
LOCAL_ROOTS = {
    "handoff", "dashboard-ui", "ifvg_search_runs", "ifvg_search_trades", "ledger", "models",
}
LOCAL_DATA_ROOTS = {
    "databento", "raw", "processed", "experiment", "ifvg_datasets", "ifvg_experiments",
    "ifvg_preparation_jobs", "ifvg_search_jobs", "ifvg_study_drafts", "ifsm_ui_replication",
    "ifvg_visual_review", "ifvg_pipeline_jobs",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def is_test_fixture(parts: tuple[str, ...]) -> bool:
    """Only explicitly named fixture trees may carry small synthetic data files."""
    return bool(parts and parts[0] == "tests" and "fixtures" in parts[1:-1])


def path_violation(path: str) -> str | None:
    parts = PurePosixPath(path.replace("\\", "/").casefold()).parts
    if not parts:
        return None
    directories = parts[:-1]
    normalized = "/".join(parts)
    if "reports" in directories:
        return "report deliveries belong outside Git"
    if parts[0] == "docs" and (
        "screenshots" in directories
        or (
            PurePosixPath(normalized).suffix in IMAGE_SUFFIXES
            and ("evidence" in directories or "/mocks/images/" in f"/{normalized}")
        )
    ):
        return "captured UI evidence belongs in the external task folder"
    if parts[0] in LOCAL_ROOTS or (
        len(parts) > 1 and parts[0] == "data" and parts[1] in LOCAL_DATA_ROOTS
    ) or normalized == "data/ifvg_profiles.json":
        return "local study/store or copied delivery directory"
    if any(part in WORKING_DIRECTORIES for part in directories):
        return "generated cache, dependency, staging or extraction directory"
    fixture = is_test_fixture(parts)
    if parts[0] not in {"src", "tests"} and any(
        part in {"src", "tests"} for part in directories[1:]
    ):
        return "nested source/test tree; keep copied repositories outside the checkout"
    if not fixture and any(
        part in {"claude-quant-lab", "strategy-core", ".git"} for part in directories
    ):
        return "copied repository tree"
    if (
        not fixture
        and normalized not in ARCHIVE_EXCEPTIONS
        and PurePosixPath(normalized).suffix in ARTIFACT_SUFFIXES
    ):
        return "generated archive, model, data or database file"
    return None


def forbidden_paths(paths: list[str]) -> list[str]:
    """Retained for callers of the original path-only guard."""
    return sorted(path for path in paths if path_violation(path))


def git(root: Path, *arguments: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *arguments], cwd=root, input=input_bytes, capture_output=True, check=True,
    ).stdout


def index_violations(root: Path, *, staged: bool = False) -> list[str]:
    """Inspect indexed paths and object sizes; deletions are absent from the index."""
    changed = None
    if staged:
        changed = set(git(
            root, "diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z",
        ).split(b"\0"))
    entries = []
    errors = []
    for record in git(root, "ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        if changed is not None and raw_path not in changed:
            continue
        mode, object_id, stage = metadata.split()
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if stage != b"0":
            errors.append(f"{path}: unresolved index conflict")
        elif mode == b"160000":
            errors.append(f"{path}: embedded Git repository/submodule is not allowed")
        else:
            entries.append((path, object_id))
        reason = path_violation(path)
        if reason:
            errors.append(f"{path}: {reason}")
    object_ids = sorted({object_id for _, object_id in entries})
    sizes = {}
    if object_ids:
        # --batch-check returns metadata only, including for protected data paths.
        raw = git(root, "cat-file", "--batch-check", input_bytes=b"\n".join(object_ids) + b"\n")
        for line in raw.splitlines():
            object_id, object_type, size = line.split()
            if object_type != b"blob":
                raise ValueError(f"Expected a Git blob: {object_id!r}")
            sizes[object_id] = int(size)
    for path, object_id in entries:
        if sizes[object_id] > MAX_BLOB_BYTES:
            errors.append(f"{path}: {sizes[object_id]:,} bytes exceeds the 5 MiB Git-file limit")
    return sorted(set(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="check added/changed index entries")
    args = parser.parse_args(argv)
    invalid = index_violations(ROOT, staged=args.staged)
    if invalid:
        print("Local artifacts cannot be committed:\n" + "\n".join(invalid))
        print(
            "Move research outputs to the sibling Claude-Quant-Lab-Research-Artifacts/"
            "<task-id>/ folder, then unstage them. Preserve saved evidence; do not delete it."
        )
        return 1
    print("Staged-artifact check passed." if args.staged else "Tracked-artifact check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
