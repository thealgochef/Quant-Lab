"""Locations for new research working outputs, separate from review deliverables."""

from pathlib import Path


def external_working_output(repo_root: Path, output_path: Path) -> Path:
    """Keep working data outside the checkout and every reports directory.

    Inspect both the supplied absolute path and its resolved symlink/junction
    target so an alias cannot hide a reserved reports directory in either path.
    """
    repository = Path(repo_root).resolve()
    requested = Path(output_path).expanduser().absolute()
    destination = requested.resolve()
    reserved_report_path = any(
        part.casefold() == "reports" for path in (requested, destination) for part in path.parts
    )
    if (
        requested.is_relative_to(repository)
        or destination.is_relative_to(repository)
        or reserved_report_path
    ):
        raise ValueError(
            "Research working outputs must stay outside the repository and every reports "
            "directory; reports directories are reserved for curated audit deliverables."
        )
    return destination


def research_working_directory(repo_root: Path, task_id: str) -> Path:
    """Resolve a task directory outside the checkout without creating it.

    Existing saved-study readers and their identities keep their original paths.
    The task ID is one directory name, never an absolute or traversing path.
    """
    if (
        not task_id
        or task_id in {".", ".."}
        or not task_id[0].isalnum()
        or any(not (character.isalnum() or character in "-_") for character in task_id)
    ):
        raise ValueError("task_id must be a directory name containing letters, digits, - or _")
    repository = Path(repo_root).resolve()
    return external_working_output(
        repository, repository.parent / "Claude-Quant-Lab-Research-Artifacts" / task_id
    )
