"""Safely list or delete dashboard-utility ML cache files.

Only targets files named ``ml_utility_*.parquet`` under date directories. Raw
Databento inputs (mbp/trades parquet) are never matched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List/delete dashboard utility cache parquet files")
    parser.add_argument("--data-dir", default="data/databento/NQ", help="Symbol root containing YYYY-MM-DD directories")
    parser.add_argument("--start", default=None, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD")
    parser.add_argument("--hash", dest="config_hash", default=None, help="Optional exact config hash, e.g. b46a2e31")
    parser.add_argument("--delete", action="store_true", help="Actually delete matched cache files; default is dry-run")
    return parser.parse_args()


def _matches_date(name: str, start: str | None, end: str | None) -> bool:
    if start is not None and name < start:
        return False
    if end is not None and name > end:
        return False
    return True


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    pattern = f"ml_utility_{args.config_hash}.parquet" if args.config_hash else "ml_utility_*.parquet"

    matches: list[Path] = []
    for day_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if not _matches_date(day_dir.name, args.start, args.end):
            continue
        matches.extend(sorted(day_dir.glob(pattern)))

    deleted: list[str] = []
    if args.delete:
        for path in matches:
            path.unlink()
            deleted.append(str(path))

    report = {
        "data_dir": str(data_dir),
        "start": args.start,
        "end": args.end,
        "pattern": pattern,
        "dry_run": not args.delete,
        "matched_count": len(matches),
        "deleted_count": len(deleted),
        "matched_files": [str(p) for p in matches],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
