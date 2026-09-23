"""Resolve the source checkout actually supplying an IFVG strategy process.

A dedicated research UI may prepend a frozen Core checkout to its own Python
path. This never changes the installed package or any other process. Normal
wheel installations prefer the prepared checkout for the current dependency
pin. A sibling checkout remains usable when no prepared checkout exists and
source parity is proved, allowing documentation-only checkout commits without
inventing a runtime identity for different code.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tomllib
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import strategy_core

IFSM_PREPARATION_APPROVAL_ID = "18abdc711313c108a23c123134c713eafd093c7eda39f545ab9cb94cae0043e7"
_PREPARATION_COMMAND = "python scripts/prepare_ifsm_research_core.py"


def research_preparation_approval(repo_root: Path) -> tuple[Path, str] | None:
    """Identify the fixed historical cache receipt in the dedicated UI only.

    Callers must load this through the verified approval reader. Its date list
    describes cache creation; it cannot authorize the new study or new I/O.
    """
    if os.environ.get("IFSM_RESEARCH_UI") != "1":
        return None
    return (
        Path(repo_root).resolve() / "data" / "ifvg_datasets" / "search" / "v1",
        IFSM_PREPARATION_APPROVAL_ID,
    )


def _package_sources(package: Path) -> dict[str, str]:
    # Python normalizes source newlines; wheel and Git checkouts can differ in
    # LF/CRLF representation. All module names and normalized contents must match.
    paths = [*package.rglob("*.py"), package / "py.typed"]
    return {
        path.relative_to(package).as_posix(): path.read_text(encoding="utf-8")
        for path in paths
        if path.is_file() and "__pycache__" not in path.parts
    }


def _prepared_core_root(repo_root: Path) -> Path:
    """Read the installed dependency's intended checkout without script imports."""
    try:
        project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            "Strategy-Core dependency pin is unavailable for provenance; "
            f"run {_PREPARATION_COMMAND} from the Quant-Lab repository."
        ) from error
    dependencies = project.get("project", {}).get("dependencies", [])
    pins = [
        match.group(1).lower()
        for dependency in dependencies
        if isinstance(dependency, str)
        and (match := re.fullmatch(
            r"strategy[-_]core\s*@\s*git\+https://github\.com/thealgochef/"
            r"Strategy-Core\.git@([0-9a-f]{40})",
            dependency.strip(),
            flags=re.IGNORECASE,
        ))
    ]
    if len(pins) != 1:
        raise RuntimeError(
            "Strategy-Core requires one exact Git dependency pin for provenance; "
            f"run {_PREPARATION_COMMAND} from the Quant-Lab repository."
        )
    return (
        repo_root.parent / "Claude-Quant-Lab-Research-Artifacts" / "ifsm-research-core" / pins[0]
    )


def _git_head(checkout: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=checkout, capture_output=True,
            text=True, check=True, timeout=60,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Strategy-Core source commit is unreadable; "
            f"run {_PREPARATION_COMMAND} from the Quant-Lab repository."
        ) from error


def _installed_core_commit() -> str:
    try:
        direct_url_text = distribution("strategy-core").read_text("direct_url.json")
        direct_url = json.loads(direct_url_text or "{}")
    except (PackageNotFoundError, OSError, ValueError) as error:
        raise RuntimeError("installed Strategy-Core provenance is unreadable") from error
    vcs_info = direct_url.get("vcs_info") if isinstance(direct_url, dict) else None
    commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("installed Strategy-Core lacks immutable VCS commit provenance")
    return commit


def strategy_core_repository_root(
    repo_root: Path, *, require_installed_commit_match: bool = False,
) -> Path:
    """Return the imported checkout, or the matching wheel's source checkout.

    Missing or mismatched provenance fails before a study reads market data.
    No package installation, environment mutation, or checkout edit occurs.
    """
    if not strategy_core.__file__:
        raise RuntimeError("The imported Strategy-Core package path is unavailable.")
    package = Path(strategy_core.__file__).resolve().parent
    imported_root = package.parent.parent
    if (
        package == imported_root / "src" / "strategy_core"
        and (imported_root / ".git").exists()
    ):
        return imported_root

    repo_root = Path(repo_root).resolve()
    prepared = _prepared_core_root(repo_root)
    # A present but damaged prepared checkout must fail, never silently select a
    # different source identity from a sibling. lexists also detects broken links.
    managed = os.path.lexists(prepared)
    siblings = (repo_root.parent / "Strategy-Core", repo_root.parent / "Strategy-core")
    checkout = prepared if managed else next(
        (sibling for sibling in siblings if os.path.lexists(sibling)), siblings[0],
    )
    source_package = checkout / "src" / "strategy_core"
    if not (checkout / ".git").exists() or not source_package.is_dir():
        raise RuntimeError(
            "Strategy-Core source checkout is unavailable for provenance: "
            f"{checkout}. Run {_PREPARATION_COMMAND} from the Quant-Lab repository."
        )
    if managed or require_installed_commit_match:
        head = _git_head(checkout)
        if managed and head != prepared.name:
            raise RuntimeError(
                "Prepared Strategy-Core HEAD differs from the Quant-Lab dependency pin; "
                f"run {_PREPARATION_COMMAND} from the Quant-Lab repository."
            )
        if head != _installed_core_commit():
            raise RuntimeError(
                "installed Strategy-Core pin differs from source checkout HEAD; "
                f"install the Quant-Lab dependency pin and run {_PREPARATION_COMMAND}."
            )
    loaded = _package_sources(package)
    expected = _package_sources(source_package)
    if not loaded or loaded != expected:
        raise RuntimeError(
            "The imported Strategy-Core package differs from its source checkout; "
            "install the Quant-Lab dependency pin and run "
            f"{_PREPARATION_COMMAND} before reviewing or running this study."
        )
    return checkout
