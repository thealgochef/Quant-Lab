"""Run the IFVG capture chain end-to-end: Phase C + entry dataset + reports.

Prereq: Phase-A artifacts warmed (``scripts/ifvg_artifact_warmer.py``); any
missing day is built inline (slower). Outputs:

* per-day ``ifvg_capture_<ctag>.parquet`` + ``ifvg_seed_<ctag>.pkl`` caches;
* ``data/databento/<sym>/ifvg_entry_dataset_<ctag>.parquet`` (combined);
* ``IFVG_FUNNEL.md`` / ``IFVG_LABELS.md`` / ``ifvg_funnel_<ctag>.json`` at the
  repo root (the window's result artifacts).

Usage:
    PYTHONPATH=src python scripts/run_ifvg_capture.py [--start D] [--end D] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ifvg_artifact_warmer import available_store_days  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.config import (  # noqa: E402
    SEALED_HOLDOUT_START,
    IfvgCaptureConfig,
)
from alpha_lab.agents.data_infra.ifvg.dataset import build_ifvg_capture  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.entry_dataset import build_entry_dataset  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.funnel_report import (  # noqa: E402
    write_funnel_report,
    write_labels_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="read-only rebuild: raise instead of re-driving the reducer or "
        "rewriting any per-day artifact if a day fails trust verification",
    )
    args = parser.parse_args()

    cfg = IfvgCaptureConfig()
    days = available_store_days(cfg.symbol, Path(cfg.data_dir))
    if args.start:
        days = [d for d in days if d >= args.start]
    if args.end:
        days = [d for d in days if d <= args.end]
    if args.limit:
        days = days[: args.limit]
    if not days:
        print("no store days matched")
        return 1
    print(
        f"capture chain over {len(days)} days ({days[0]}..{days[-1]}) "
        f"profile={cfg.profile_hash[:12]} ctag={cfg.capture_tag()}"
    )

    t0 = time.time()

    def progress(i: int, n: int, date_str: str) -> None:
        if i % 10 == 0 or i == n:
            print(f"  [{i}/{n}] {date_str} ({time.time()-t0:.0f}s)", flush=True)

    chain = build_ifvg_capture(days, cfg, progress_fn=progress, cached_only=args.cached_only)
    capture = chain.frame()
    print(
        f"chain done in {time.time()-t0:.0f}s: {len(capture)} rows, "
        f"{len(chain.cached_days)} cached, {len(chain.rebuilt_days)} artifacts rebuilt inline"
    )

    t1 = time.time()
    entries = build_entry_dataset(capture, cfg)
    print(f"entry dataset: {len(entries)} rows in {time.time()-t1:.0f}s")
    out_entries = (
        Path(cfg.data_dir) / cfg.symbol / f"ifvg_entry_dataset_{cfg.capture_tag()}.parquet"
    )
    if len(entries):
        entries.to_parquet(out_entries, index=False)
        print(f"wrote {out_entries}")

    root = Path(__file__).resolve().parents[1]
    segments = write_funnel_report(
        chain.day_funnels,
        root / "IFVG_FUNNEL.md",
        root / f"ifvg_funnel_{cfg.capture_tag()}.json",
        warmup_days=cfg.warmup_days,
        sealed_start=SEALED_HOLDOUT_START,
    )
    write_labels_report(entries, root / "IFVG_LABELS.md")
    print("reports: IFVG_FUNNEL.md, IFVG_LABELS.md")
    print(
        "headline (core segment):",
        {
            k: segments["core"].get(k, 0)
            for k in ("htf_taps", "setups_born", "inversions", "entries_selected")
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
