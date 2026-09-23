"""Restore the exact optional research Core outside Quant-Lab; never install it.

Requires Git and network access to the public prerequisite commit. The bundled
delta contains only the six additional research commits. Existing destinations
are verified in place and are never overwritten, repaired or deleted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ifsm_research_runtime import (
    DEFAULT_CORE,
    MANIFEST_PATH,
    ROOT,
    git,
    manifest,
    source_bytes,
    verify_core,
)


def prepare(destination: Path) -> dict:
    expected = manifest()
    requested = destination.expanduser().absolute()
    destination = requested.resolve()
    if any(
        part.casefold() == "reports"
        for path in (requested, destination)
        for part in path.parts
    ):
        raise ValueError("Research Core working checkouts cannot be created inside reports folders")
    if destination == ROOT or destination.is_relative_to(ROOT):
        raise ValueError("Research Core must be outside the Quant-Lab repository")
    bundle = MANIFEST_PATH.parent / expected["bundle_file"]
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != expected["bundle_sha256"]:
        raise ValueError("Research Core bundle checksum differs; restore the versioned bundle")
    if destination.exists():
        return verify_core(destination, expected)
    destination.mkdir(parents=True)
    git(destination, "init", "--quiet")
    # Match Git's clean-index normalization while preserving the frozen Windows
    # working bytes on every host, including Linux CI.
    git(destination, "config", "core.autocrlf", "true")
    git(destination, "config", "core.safecrlf", "false")
    git(destination, "remote", "add", "origin", expected["repository_url"])
    git(destination, "fetch", "--no-tags", "--depth=1", "origin", expected["base_commit"])
    git(destination, "bundle", "verify", str(bundle))
    git(destination, "fetch", "--no-tags", str(bundle), expected["target_ref"])
    git(destination, "checkout", "--detach", expected["target_commit"])
    for entry in expected["source_files"]:
        relative = entry["path"]
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination / "src/strategy_core"):
            raise ValueError(f"Source manifest path escapes the package: {relative}")
        blob = git(destination, "show", f"{expected['target_commit']}:{relative}")
        content = source_bytes(blob, entry)
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Recorded source cannot be reconstructed: {relative}")
        path.write_bytes(content)
    # Refresh the fresh checkout's index after restoring nonuniform historical
    # EOLs. Its Git tree must remain exactly the original committed tree.
    git(destination, "add", "--renormalize", "--", "src/strategy_core")
    if git(destination, "write-tree") != git(destination, "rev-parse", "HEAD^{tree}"):
        raise ValueError("Restoring historical source bytes changed the committed tree")
    return verify_core(destination, expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_CORE)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.destination), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
