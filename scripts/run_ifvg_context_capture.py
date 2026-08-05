"""Build the fixed, nonsealed IFVG v3 context exploration artifact.

This command performs deterministic measurement and verification only.  It
does not train, tune, select features, calculate trading performance, or read
outside the frozen January 2026 allowlist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from alpha_lab.agents.data_infra.ifvg.verification import (  # noqa: E402
    run_context_verification,
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

    diagnostics: dict[str, object] = {}
    output = run_context_verification(
        repo_root=_REPO_ROOT,
        cached_artifacts_only=args.cached_artifacts_only,
        progress_fn=progress,
        diagnostics_out=diagnostics,
    )
    print(
        f"saved immutable IFVG v3 context artifact to {output} "
        f"({time.monotonic() - started:.1f}s)"
    )
    print("diagnostic timings (excluded from slowdown denominator):")
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
