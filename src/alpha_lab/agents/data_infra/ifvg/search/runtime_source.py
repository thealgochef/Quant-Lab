"""Resolve the source checkout actually supplying an IFVG strategy process.

A dedicated research UI may prepend a frozen Core checkout to its own Python
path. This never changes the installed package or any other process. Normal
wheel installations may use the sibling checkout only after source parity is
proved, allowing documentation-only checkout commits without inventing a
runtime identity for different code.
"""

from __future__ import annotations

import os
from pathlib import Path

import strategy_core

IFSM_PREPARATION_APPROVAL_ID = "18abdc711313c108a23c123134c713eafd093c7eda39f545ab9cb94cae0043e7"


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


def strategy_core_repository_root(repo_root: Path) -> Path:
    """Return the imported checkout, or a source-equivalent installed sibling.

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

    sibling = Path(repo_root).resolve().parent / "Strategy-Core"
    source_package = sibling / "src" / "strategy_core"
    if not (sibling / ".git").exists() or not source_package.is_dir():
        raise RuntimeError("Strategy-Core source checkout is unavailable for provenance.")
    loaded = _package_sources(package)
    expected = _package_sources(source_package)
    if not loaded or loaded != expected:
        raise RuntimeError(
            "The imported Strategy-Core package differs from its source checkout; "
            "launch the matching research environment before reviewing or running this study."
        )
    return sibling
