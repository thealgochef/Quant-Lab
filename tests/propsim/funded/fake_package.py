"""A tiny SYNTHETIC verified study package plus matching MBP-1 trade prints.

Built inside pytest's temporary directory. It mirrors the file layout of a
real verified package (MANIFEST.json, external receipt, trades, calendar,
schedule, one-minute bars) so the funded worker can be exercised end to end.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

MIN = 60 * 10**9
RUN_ID = "f" * 63 + "1"


def _ns(text: str) -> int:
    return int(pd.Timestamp(text).value)


def _trade_prints(entry: str, minutes: int, start: int, legs: list[int]) -> list[tuple[int, int]]:
    """Deterministic prints: each minute walks to the next leg level."""

    out, t0, price = [], _ns(entry), start
    for minute in range(minutes):
        target = legs[min(minute, len(legs) - 1)]
        for k in range(3):
            price = price + (target - price) * (k + 1) // 3
            out.append((t0 + minute * MIN + (k + 1) * 10 * 10**9, price))
    return out


TRADES = [
    # id, day, entry utc, stop ticks away, exit reason, minute legs (ticks)
    ("A", "2026-01-13", "2026-01-13T15:00:00Z", 560, "target", [100_200, 100_560]),
    ("B", "2026-01-14", "2026-01-14T15:00:00Z", 100, "stop", [99_950, 99_900]),
    ("C", "2026-01-16", "2026-01-16T15:00:00Z", 100, "target", [100_050, 100_100]),
]


def build_fake_package(archive_root: Path, data_root: Path) -> str:
    root = archive_root / "fake_funded_source_20260101" / "final_extracted_0001"
    (root / "data").mkdir(parents=True)
    (root / "shared").mkdir()
    (root / "ledger").mkdir()
    trade_rows, bars, prints_by_day = [], [], {}
    for trade_id, day, entry, stop_away, reason, legs in TRADES:
        entry_ticks = 100_000
        stop = entry_ticks - stop_away
        target = entry_ticks + stop_away
        exit_level = target if reason == "target" else stop
        prints = _trade_prints(entry, len(legs), entry_ticks, legs)
        prints[-1] = (prints[-1][0], exit_level)
        # the exit level is first reached on the last print of the last minute
        for ts, px in prints:
            prints_by_day.setdefault(pd.Timestamp(ts, tz="UTC").date(), []).append((ts, px))
        frame = pd.DataFrame(prints, columns=["ts", "px"])
        frame["minute"] = frame["ts"] - frame["ts"] % MIN
        for minute, group in frame.groupby("minute"):
            bars.append({
                "timeframe_seconds": 60,
                "logical_open_ts_utc": pd.Timestamp(minute, tz="UTC"),
                "logical_close_ts_utc": pd.Timestamp(minute + MIN, tz="UTC"),
                "open_ticks": int(group.px.iloc[0]), "high_ticks": int(group.px.max()),
                "low_ticks": int(group.px.min()), "close_ticks": int(group.px.iloc[-1]),
                "trade_count": len(group), "is_complete": True,
                "source_artifact_sha256": "0" * 64,
            })
        resolution = pd.Timestamp(entry) + pd.Timedelta(minutes=len(legs))
        trade_rows.append({
            "profile": "P1", "trade_id": trade_id, "direction": "LONG",
            "envelope_trading_day": day, "entry_ts_utc": entry,
            "entry_ticks": entry_ticks, "stop_ticks": stop, "exit_ticks": exit_level,
            "resolution": reason, "resolution_ts_utc": resolution.isoformat(),
            "is_warmup": False, "geometry_htf_timeframe_seconds": 3600,
            "geometry_parent_timeframe_seconds": 300,
            "geometry_entry_bar_timeframe_seconds": 60,
        })
    pd.DataFrame(trade_rows).to_csv(root / "data/trades.csv", index=False)
    days = ["2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16"]
    pd.DataFrame({
        "trading_day": days, "is_evaluation": [True] * 4,
        "source_day_start_utc": [f"{pd.Timestamp(d) - pd.Timedelta(days=1):%Y-%m-%d} "
                                 "23:00:00+00:00" for d in days],
    }).to_csv(root / "data/calendar.csv", index=False)
    pd.DataFrame({
        "local_date": days, "is_evaluation_date": [True] * 4,
        "deadline_ts_utc": [f"{d} 21:55:00+00:00" for d in days],
        "market_close_ts_utc": [f"{d} 22:00:00+00:00" for d in days],
        "reopen_ts_utc": [f"{d} 23:00:00+00:00" for d in days],
    }).to_csv(root / "shared/trading_schedule.csv", index=False)
    pd.DataFrame(bars).to_parquet(root / "shared/market_bars.parquet")
    (root / "RESULTS.md").write_text("# Synthetic funded source study\n", encoding="utf-8")
    (root / "ledger/events.jsonl").write_text(
        json.dumps({"event_id": "synthetic_prior_event", "event_type": "completed",
                    "question": "Synthetic prior study"}) + "\n", encoding="utf-8")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append({"path": path.relative_to(root).as_posix(),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "size_bytes": path.stat().st_size})
    manifest = json.dumps({"schema_version": 1, "run_id": RUN_ID, "files": files}).encode()
    (root / "MANIFEST.json").write_bytes(manifest)
    (archive_root / f"fake_funded_source_{RUN_ID}.receipt.json").write_text(json.dumps({
        "passed": True, "run_id": RUN_ID,
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }), encoding="utf-8")
    for day, rows in prints_by_day.items():
        folder = data_root / day.isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        rows.sort()
        pd.DataFrame({
            "ts_event": pd.to_datetime([ts for ts, _ in rows], utc=True),
            "action": ["T"] * len(rows),
            "price": [px * 0.25 for _, px in rows],
            "instrument_id": [42] * len(rows),
            "symbol": ["NQH6"] * len(rows),
            "sequence": list(range(len(rows))),
        }).to_parquet(folder / "mbp1.parquet")
    return RUN_ID
