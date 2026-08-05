"""Warm only the fixed allowlisted IFVG v2 bar/level chain.

Artifacts have chained prior-day level seeds, so warming is deliberately
sequential.  Arbitrary date ranges and sealed/validation enumeration are not
supported by the repaired path.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.data_access import (  # noqa: E402
    EXPLORATION_DATE_ALLOWLIST,
    ExplorationDataPolicy,
    discover_allowlisted_source_files,
)
from alpha_lab.agents.data_infra.ifvg.dataset import (  # noqa: E402
    build_ifvg_v2_capture,
)
from alpha_lab.agents.data_infra.ifvg.profiles import (  # noqa: E402
    resolve_profile_config,
)


def available_store_days(
    symbol: str,
    data_dir: Path,
    *,
    policy: ExplorationDataPolicy | None = None,
) -> list[str]:
    """Return available dates from the explicit exploration allowlist only."""
    policy = policy or ExplorationDataPolicy()
    files = discover_allowlisted_source_files(
        policy,
        data_dir=Path(data_dir),
        symbol=symbol,
        dates=sorted(EXPLORATION_DATE_ALLOWLIST),
    )
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="accepted for compatibility; chained v2 warming requires 1",
    )
    args = parser.parse_args()
    if args.workers != 1:
        parser.error("IFVG v2 artifacts are seed-chained; --workers must be 1")

    resolved = resolve_profile_config()
    cfg = replace(
        IfvgCaptureConfig(),
        section=resolved.section,
        data_dir=_REPO_ROOT / "data" / "databento",
    )
    policy = ExplorationDataPolicy()
    days = available_store_days(cfg.symbol, cfg.data_dir, policy=policy)
    if days != sorted(EXPLORATION_DATE_ALLOWLIST):
        missing = sorted(EXPLORATION_DATE_ALLOWLIST - set(days))
        raise RuntimeError(
            "the fixed exploration source chain is incomplete; "
            f"missing {missing}"
        )
    if len([day for day in days if day <= "2026-01-12"]) != 10:
        raise RuntimeError("the fixed ten-day warmup chain is incomplete")

    def progress(done: int, total: int, day: str) -> None:
        print(f"[{done}/{total}] {day}", flush=True)

    capture = build_ifvg_v2_capture(
        days,
        cfg,
        resolved,
        access_policy=policy,
        progress_fn=progress,
    )
    print(
        f"trusted artifacts: {len(capture.cached_artifact_days)} cached, "
        f"{len(capture.rebuilt_days)} built"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
