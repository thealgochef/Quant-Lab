"""Build the fixed, nonsealed IFVG v2 exploration dataset.

This compatibility entry point now delegates to the repair verifier.  It does
not enumerate arbitrary store dates, write the v1 wide capture, run a model,
or touch validation/sealed sources.

Usage:
    PYTHONPATH=src python scripts/run_ifvg_capture.py [--cached-artifacts-only]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from alpha_lab.agents.data_infra.ifvg.verification import (  # noqa: E402
    run_repair_verification,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cached-artifacts-only",
        action="store_true",
        help="fail if a trusted v2 bar/level artifact is missing",
    )
    args = parser.parse_args()
    started = time.monotonic()

    def progress(done: int, total: int, day: str) -> None:
        print(f"[{done}/{total}] {day}", flush=True)

    output = run_repair_verification(
        repo_root=_REPO_ROOT,
        cached_artifacts_only=args.cached_artifacts_only,
        progress_fn=progress,
    )
    print(
        f"saved immutable IFVG v2 exploration dataset to {output} "
        f"({time.monotonic() - started:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
