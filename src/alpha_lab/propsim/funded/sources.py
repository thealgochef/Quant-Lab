"""Verified completed-study packages as funded-simulation strategy sources.

A funded simulation replays the executions of ONE profile from a completed,
externally verified study package (for example the September 18 daily-close
study). Before anything is read, the package's external receipt must say it
passed, the receipt's manifest hash must equal the extracted ``MANIFEST.json``
bytes, and every file used here must match its manifest SHA-256. Nothing in a
package is modified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_lab.propsim.funded.campaign import TradingDay
from alpha_lab.propsim.funded.clock import to_ns
from alpha_lab.propsim.funded.paths import StrategyExecution

__all__ = [
    "ARCHIVE_ROOT",
    "PackageError",
    "VerifiedPackage",
    "discover_packages",
    "open_verified_package",
    "load_profile_executions",
    "load_trading_days",
    "load_minute_bars",
    "executions_sha256",
    "chart_label",
]

ARCHIVE_ROOT = (
    Path(__file__).resolve().parents[4].parent
    / "Claude-Quant-Lab-Research-Artifacts"
    / "archived-reports"
)

_REQUIRED = ("data/trades.csv", "data/calendar.csv", "shared/trading_schedule.csv",
             "shared/market_bars.parquet", "RESULTS.md")


class PackageError(ValueError):
    """The package cannot be trusted as a strategy source (fail closed)."""


@dataclass(frozen=True)
class VerifiedPackage:
    root: Path
    run_id: str
    manifest_sha256: str
    receipt_path: Path
    profiles: tuple[str, ...]
    title: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_for(root: Path, run_id: str) -> Path | None:
    parent = root.parent.parent if root.parent.name != "archived-reports" else root.parent
    for candidate in sorted(parent.glob(f"*{run_id}*.receipt.json")):
        return candidate
    return None


def open_verified_package(root: Path) -> VerifiedPackage:
    root = Path(root)
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        raise PackageError(f"no MANIFEST.json in {root.name}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    run_id = str(manifest.get("run_id", ""))
    receipt_path = _receipt_for(root, run_id)
    if receipt_path is None:
        raise PackageError("no external verification receipt for this package")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if not receipt.get("passed") or receipt.get("run_id") != run_id:
        raise PackageError("the package's verification receipt did not pass")
    if receipt.get("manifest_sha256") != manifest_sha:
        raise PackageError("MANIFEST.json differs from the verified receipt")
    files = {entry["path"]: entry for entry in manifest.get("files", [])}
    for rel in _REQUIRED:
        entry = files.get(rel)
        if entry is None:
            raise PackageError(f"package manifest does not list {rel}")
        path = root / rel
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise PackageError(f"{rel} does not match its verified manifest hash")
    trades = pd.read_csv(root / "data/trades.csv", usecols=["profile"])
    profiles = tuple(sorted(trades["profile"].dropna().unique()))
    title = (root / "RESULTS.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
    return VerifiedPackage(root=root, run_id=run_id, manifest_sha256=manifest_sha,
                           receipt_path=receipt_path, profiles=profiles, title=title)


def discover_packages(archive_root: Path | None = None) -> list[VerifiedPackage]:
    archive_root = ARCHIVE_ROOT if archive_root is None else archive_root
    found: list[VerifiedPackage] = []
    if not archive_root.is_dir():
        return found
    for manifest in sorted(archive_root.glob("*/final_extracted_*/MANIFEST.json")):
        try:
            found.append(open_verified_package(manifest.parent))
        except (PackageError, OSError, ValueError, KeyError):
            continue
    return found


_SECONDS_LABEL = {60: "1-minute", 180: "3-minute", 300: "5-minute", 600: "10-minute",
                  900: "15-minute", 1800: "30-minute", 3600: "1-hour", 14400: "4-hour"}


def chart_label(row: pd.Series) -> str:
    def name(value) -> str:
        try:
            return _SECONDS_LABEL.get(int(value), f"{int(value)}-second")
        except (TypeError, ValueError):
            return "unknown"

    return (f"{name(row.get('geometry_htf_timeframe_seconds'))} gap, "
            f"{name(row.get('geometry_parent_timeframe_seconds'))} parent chart, "
            f"{name(row.get('geometry_entry_bar_timeframe_seconds'))} entry chart")


def _ticks(value) -> int:
    number = float(value)
    if number != int(number):
        raise PackageError(f"non-integer tick price {value!r}")
    return int(number)


def load_profile_executions(package: VerifiedPackage, profile_id: str
                            ) -> tuple[StrategyExecution, ...]:
    if profile_id not in package.profiles:
        raise PackageError(f"profile {profile_id} is not in this package")
    frame = pd.read_csv(package.root / "data/trades.csv", low_memory=False)
    frame = frame[frame["profile"] == profile_id].sort_values(["entry_ts_utc", "trade_id"])
    out: list[StrategyExecution] = []
    for _, row in frame.iterrows():
        direction = str(row["direction"]).lower()
        if direction not in ("long", "short"):
            raise PackageError(f"unknown direction {row['direction']!r}")
        sign = 1 if direction == "long" else -1
        entry, stop = _ticks(row["entry_ticks"]), _ticks(row["stop_ticks"])
        reason = str(row["resolution"])
        if reason not in ("stop", "target", "scheduled_close"):
            raise PackageError(f"unsupported exit {reason!r}")
        target = entry + sign * abs(entry - stop)  # the strategy's fixed 1R target
        exit_ticks = _ticks(row["exit_ticks"])
        if reason == "target" and exit_ticks != target:
            raise PackageError("a target exit does not match the 1R target price")
        out.append(StrategyExecution(
            trade_id=str(row["trade_id"]), trading_day=str(row["envelope_trading_day"]),
            direction=direction, entry_ts_utc=str(row["entry_ts_utc"]), entry_ticks=entry,
            stop_ticks=stop, target_ticks=target, exit_ts_utc=str(row["resolution_ts_utc"]),
            exit_ticks=exit_ticks, exit_reason=reason, is_warmup=bool(row["is_warmup"]),
            entry_chart=chart_label(row),
        ))
    return tuple(out)


def executions_sha256(executions: tuple[StrategyExecution, ...]) -> str:
    payload = json.dumps([e.model_dump(mode="json") for e in executions],
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_trading_days(package: VerifiedPackage) -> tuple[tuple[TradingDay, ...], int, int,
                                                         tuple[str, str]]:
    """Evaluation trading days, the start (first session open) and the cutoff."""

    schedule = pd.read_csv(package.root / "shared/trading_schedule.csv")
    calendar = pd.read_csv(package.root / "data/calendar.csv")
    evaluation = calendar[calendar["is_evaluation"] == True]  # noqa: E712
    if evaluation.empty:
        raise PackageError("the package has no evaluation dates")
    first, last = str(evaluation["trading_day"].min()), str(evaluation["trading_day"].max())
    start_ns = to_ns(str(evaluation.sort_values("trading_day").iloc[0]["source_day_start_utc"]))
    days = []
    for _, row in schedule.sort_values("local_date").iterrows():
        if not bool(row["is_evaluation_date"]):
            continue
        days.append(TradingDay(
            trading_day=str(row["local_date"]),
            day_end_ns=to_ns(str(row["market_close_ts_utc"])),
            reopen_ns=to_ns(str(row["reopen_ts_utc"])),
            deadline_ns=to_ns(str(row["deadline_ts_utc"])),
        ))
    last_day = [d for d in days if d.trading_day == last]
    if not last_day:
        raise PackageError("the last evaluation date has no scheduled close")
    return tuple(days), start_ns, last_day[0].day_end_ns, (first, last)


def load_minute_bars(package: VerifiedPackage) -> pd.DataFrame:
    bars = pd.read_parquet(
        package.root / "shared/market_bars.parquet",
        columns=["timeframe_seconds", "logical_open_ts_utc", "logical_close_ts_utc",
                 "open_ticks", "high_ticks", "low_ticks", "close_ticks", "trade_count",
                 "is_complete", "source_artifact_sha256"],
    )
    bars = bars[bars["timeframe_seconds"] == 60].copy()
    bars["open_ns"] = bars["logical_open_ts_utc"].astype("int64")
    bars["close_ns"] = bars["logical_close_ts_utc"].astype("int64")
    return bars.sort_values("open_ns").reset_index(drop=True)
