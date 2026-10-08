"""Create an external research workspace without overwriting previous work."""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from pathlib import Path

WORKSPACE_FOLDERS = ("inputs", "study_store", "work", "source", "validation")
_TASK_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_RESERVED = {"con", "prn", "aux", "nul"} | {
    f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(10)
}


def require_external_work_paths(
    repo_root: Path, paths: Mapping[str, Path],
) -> dict[str, Path]:
    """Reject source-local work, including Git checkouts and resolved links."""
    repository = repo_root.resolve()
    resolved = {name: path.resolve() for name, path in paths.items()}
    for name, path in resolved.items():
        git_root = next((ancestor for ancestor in (path, *path.parents)
                         if (ancestor / ".git").exists()), None)
        if path.is_relative_to(repository) or git_root is not None:
            raise ValueError(
                f"{name} must be outside the repository: {path}. "
                "Use a task folder in the sibling Claude-Quant-Lab-Research-Artifacts directory."
            )
    return resolved


def create_workspace(task_id: str, *, repo_root: Path) -> Path:
    """Create a fresh task scaffold; existing destinations are never changed."""
    if len(task_id) > 64 or not _TASK_ID.fullmatch(task_id) or task_id in _RESERVED:
        raise ValueError(
            "task ID must start with a lowercase letter and contain only lowercase "
            "letters, digits and single separating hyphens (maximum 64 characters); "
            "Windows device names are not allowed"
        )
    repository = repo_root.resolve()
    destination = repository.parent / "Claude-Quant-Lab-Research-Artifacts" / task_id
    require_external_work_paths(repository, {"workspace": destination})
    destination.mkdir(parents=True, exist_ok=False)
    for name in WORKSPACE_FOLDERS:
        (destination / name).mkdir()
    with (destination / "README.md").open("x", encoding="utf-8") as output:
        output.write(
            f"# {task_id}\n\n"
            "Status: workspace created; no study has been approved or run.\n\n"
            "| Folder | Purpose |\n|---|---|\n"
            "| inputs | Task inputs and source receipts |\n"
            "| study_store | Saved plans, approvals and immutable study records |\n"
            "| work | Worker state, caches, replay traces and staging |\n"
            "| source | Exact source snapshots needed by approved runs |\n"
            "| validation | Focused checks and verification evidence |\n\n"
            "Record the task contract, plan/result IDs and final report links here. "
            "Keep existing source identities and evidence unchanged. "
            "Only final lightweight review deliverables belong in the repository's "
            "ignored reports/ directory.\n"
        )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", help="new lowercase-hyphenated research task ID")
    args = parser.parse_args()
    try:
        destination = create_workspace(args.task_id, repo_root=Path(__file__).resolve().parents[1])
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(destination)


if __name__ == "__main__":
    main()
