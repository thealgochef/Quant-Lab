"""Reject tracked report directories and the local replay catalog before merge."""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CATALOG = "data/ifvg_datasets/replay_chart_catalog_v1.json"


def forbidden_paths(paths: list[str]) -> list[str]:
    return sorted(
        path for path in paths
        if "reports" in [part.casefold() for part in PurePosixPath(path).parts[:-1]]
        or path.casefold() == LOCAL_CATALOG
    )


def main() -> int:
    raw = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True,
    ).stdout
    invalid = forbidden_paths([path.decode() for path in raw.split(b"\0") if path])
    if invalid:
        raise SystemExit("Generated local artifacts are tracked:\n" + "\n".join(invalid))
    print("Tracked-artifact check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
