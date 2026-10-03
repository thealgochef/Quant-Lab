"""Saved setup geometry for funded trades (TASK.md section 4.1; rules 6–8).

A funded comparison's export has no setup zones. Each funded trade carries the
``strategy_trade_id`` of the strategy trade it followed, and the plan is bound to
ONE verified strategy package whose ``data/trades.csv`` stores, per configuration
("profile") and execution, the setup geometry the strategy saved: the four-hour
(higher-timeframe) gap, the parent gap, the opposing one-minute gap, the tap,
lock (parent retest), close-through ("inversion") and entry bars.

This module links a funded trade to that saved record — never by re-running
anything:

(a) **exact** — the funded configuration is itself a member of the verified
    study: the member's record with the same trade id;
(b) **same execution** — otherwise, a verified member's record of the same
    execution, found by the package's own ``entry_match_key`` (a hash of the
    entry instant, direction, entry family and entry price, formed here from the
    funded trade's stored fields exactly as the package forms it) plus the same
    initial stop. A record with the same target is preferred; among those, the
    member closest in settings to the funded configuration, then the name, then
    the record's trade id (so the choice never depends on the file's row order);
(c) otherwise nothing: the screen shows "Setup zones weren't recorded for this
    study".

Correction A6: only (a) establishes the setup's identity
(:attr:`SetupRecord.identity_established`). Matching entry time, direction,
family, entry price, stop or target does NOT show that another configuration's
setup is this configuration's setup — its gap charts, sessions, zones and prior
history can differ. A (b) record is therefore *related context* only: screens
label it so and never use it for this configuration's formation history, its
step names or point-in-time moments.

The package file is checked against its manifest hash before use (the package
itself is located by the plan's run id and manifest hash). Nothing here writes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import utc_instant

__all__ = [
    "NO_RECORD",
    "Gap",
    "SetupRecord",
    "SetupRecordError",
    "SetupRecordSource",
    "entry_match_key",
    "find_setup_record",
    "load_setup_record_source",
    "settings_distance",
]

NO_RECORD = "Setup zones weren't recorded for this study"
_ROLES = ("htf", "parent", "opposing", "entry_fvg")
_GAP_FIELDS = ("fvg_id", "timeframe_seconds", "direction", "gap_low_ticks", "gap_high_ticks",
               "size_ticks", "a_open_ts_utc", "confirmed_ts_utc")
_BARS = ("tap_bar", "lock_bar", "inversion_bar", "entry_bar")
_COLUMNS = (
    ["profile", "trade_id", "setup_id", "is_warmup", "direction", "entry_family",
     "entry_ts_utc", "entry_ticks", "stop_ticks", "target_ticks", "resolution",
     "entry_match_key", "exact_execution_key"]
    + [f"geometry_{role}_{name}" for role in _ROLES for name in _GAP_FIELDS]
    + [f"geometry_{bar}_logical_{edge}_ts_utc" for bar in _BARS for edge in ("open", "close")]
)
_DEFAULT_EXIT = "exit_policy.fixed_target_v1"


class SetupRecordError(ValueError):
    """The package's setup records could not be verified; nothing is shown from them."""


def entry_match_key(entry_utc: Any, direction: str, entry_family: str, entry_ticks: Any) -> str:
    """The package's ``entry_match_key``: sha256 of the JSON list
    ``[entry instant (ISO, UTC), direction, entry family, entry price in ticks]``.

    Matches ``export_results.py``/``verify.py`` of the verified package: times as
    ``Timestamp.isoformat()`` in UTC, ticks as integers, compact separators.
    """

    instant = utc_instant(entry_utc)
    if instant is None:
        raise ValueError("an entry instant is required")
    values = [instant.isoformat(), str(direction), str(entry_family), int(entry_ticks)]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def settings_distance(axis_value_ids: Mapping[str, str], member_ids: Mapping[str, str],
                      baseline: Mapping[str, str]) -> int:
    """How many of the funded configuration's settings differ from a verified member's.

    A setting the member does not name takes the registry baseline (the member was
    resolved from it); the exit policy defaults to the fixed target.
    """

    differences = 0
    for axis, value in axis_value_ids.items():
        other = member_ids.get(axis) or baseline.get(axis)
        if other is None and axis == "exit_policy":
            other = _DEFAULT_EXIT
        if other != value:
            differences += 1
    return differences


@dataclass(frozen=True)
class Gap:
    """One saved gap: role, chart size, direction, price range (ticks) and times."""

    role: str
    timeframe_seconds: int | None
    direction: str
    low_ticks: int
    high_ticks: int
    size_ticks: int | None
    first_open_utc: pd.Timestamp | None
    confirmed_utc: pd.Timestamp | None


@dataclass(frozen=True)
class SetupRecord:
    """The saved setup of one execution, and how it was linked to the funded trade."""

    link: str  # "exact" | "same_execution"
    configuration: str  # whose record it is (a verified configuration of the package)
    trade_id: str
    setup_id: str | None
    gaps: dict[str, Gap] = field(default_factory=dict)
    bars: dict[str, tuple[pd.Timestamp | None, pd.Timestamp | None]] = field(
        default_factory=dict)  # bar → (open instant, close instant)
    entry_utc: pd.Timestamp | None = None
    entry_ticks: int | None = None
    stop_ticks: int | None = None
    target_ticks: int | None = None
    same_target: bool = True
    settings_differences: int | None = None

    @property
    def htf(self) -> Gap | None:
        return self.gaps.get("htf")

    @property
    def parent(self) -> Gap | None:
        return self.gaps.get("parent")

    @property
    def opposing(self) -> Gap | None:
        return self.gaps.get("opposing")

    def bar_open(self, bar: str) -> pd.Timestamp | None:
        return self.bars.get(bar, (None, None))[0]

    def bar_close(self, bar: str) -> pd.Timestamp | None:
        return self.bars.get(bar, (None, None))[1]

    @property
    def identity_established(self) -> bool:
        """True only for this configuration's own record by trade id (correction A6)."""

        return self.link == "exact"

    @property
    def source_sentence(self) -> str:
        """Whose record the zones come from, in one plain sentence."""

        if self.identity_established:
            return "Zones come from this configuration's own saved setup record."
        target = ", and target" if self.same_target else " (its target differs)"
        return (f"Related context from configuration {self.configuration}'s saved setup record "
                f"for the same entry time, direction, entry price and stop{target}; exact setup "
                "identity for this configuration is not established.")


@dataclass(frozen=True)
class SetupRecordSource:
    """The verified package's saved executions, indexed for linking."""

    package_root: Path
    frame: pd.DataFrame = field(repr=False)
    #: verified member name → its registry value ids
    members: dict[str, dict[str, str]] = field(default_factory=dict)
    #: member name → its entry family (from the manifest-checked ``configs.json``)
    families: dict[str, str] = field(default_factory=dict)
    #: registry axis → baseline value id
    baseline: dict[str, str] = field(default_factory=dict)

    def family(self, configuration: str | None, base: str | None = None) -> str | None:
        for name in (configuration, base):
            if name and name in self.families:
                return self.families[name]
        values = set(self.families.values())
        return values.pop() if len(values) == 1 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked(package_root: Path, relative: str, files: Mapping[str, Mapping[str, Any]]) -> Path:
    entry = files.get(relative)
    path = package_root / relative
    if entry is None or not path.is_file() or _sha256(path) != entry.get("sha256"):
        raise SetupRecordError(f"{relative} does not match the package's manifest hash")
    return path


def _registry_baseline() -> dict[str, str]:
    try:
        from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
            SEARCH_AXIS_REGISTRY_V1,
        )
    except Exception:  # the registry is optional for linking; exact ties are then by name
        return {}
    return {axis: spec.baseline_value_id for axis, spec in SEARCH_AXIS_REGISTRY_V1.items()}


def load_setup_record_source(package_root: Path | str) -> SetupRecordSource:
    """Read the verified package's executions (manifest hashes checked first)."""

    root = Path(package_root)
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    files = {entry["path"]: entry for entry in manifest.get("files", [])}
    trades_path = _checked(root, "data/trades.csv", files)
    context = json.loads(_checked(root, "run_context.json", files).read_text(encoding="utf-8"))
    configs = json.loads(_checked(root, "configs.json", files).read_text(encoding="utf-8"))
    header = pd.read_csv(trades_path, nrows=0).columns
    frame = pd.read_csv(trades_path, usecols=[c for c in _COLUMNS if c in header],
                        low_memory=False)
    members = {str(m["name"]): dict((m.get("spec") or {}).get("axis_value_ids") or {})
               for m in (context.get("frozen_batch") or {}).get("members") or []}
    families = {name: str((cfg.get("section") or {}).get("entry_family"))
                for name, cfg in configs.items()
                if isinstance(cfg, dict) and (cfg.get("section") or {}).get("entry_family")}
    return SetupRecordSource(package_root=root, frame=frame, members=members,
                             families=families, baseline=_registry_baseline())


def _ticks(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _instant(value: Any) -> pd.Timestamp | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return utc_instant(str(value))


def _record(row: pd.Series, *, link: str, same_target: bool,
            distance: int | None) -> SetupRecord:
    gaps: dict[str, Gap] = {}
    for role in _ROLES:
        prefix = f"geometry_{role}_"
        low, high = _ticks(row.get(prefix + "gap_low_ticks")), _ticks(
            row.get(prefix + "gap_high_ticks"))
        if low is None or high is None:
            continue
        gaps[role] = Gap(role=role, timeframe_seconds=_ticks(row.get(prefix + "timeframe_seconds")),
                         direction=str(row.get(prefix + "direction") or ""),
                         low_ticks=low, high_ticks=high,
                         size_ticks=_ticks(row.get(prefix + "size_ticks")),
                         first_open_utc=_instant(row.get(prefix + "a_open_ts_utc")),
                         confirmed_utc=_instant(row.get(prefix + "confirmed_ts_utc")))
    bars = {bar: (_instant(row.get(f"geometry_{bar}_logical_open_ts_utc")),
                  _instant(row.get(f"geometry_{bar}_logical_close_ts_utc"))) for bar in _BARS}
    setup = row.get("setup_id")
    return SetupRecord(
        link=link, configuration=str(row["profile"]), trade_id=str(row["trade_id"]),
        setup_id=None if setup is None or (isinstance(setup, float) and pd.isna(setup))
        else str(setup),
        gaps=gaps, bars=bars, entry_utc=_instant(row.get("entry_ts_utc")),
        entry_ticks=_ticks(row.get("entry_ticks")), stop_ticks=_ticks(row.get("stop_ticks")),
        target_ticks=_ticks(row.get("target_ticks")), same_target=same_target,
        settings_differences=distance)


def _has_geometry(row: pd.Series) -> bool:
    return (pd.notna(row.get("geometry_htf_gap_low_ticks"))
            or pd.notna(row.get("geometry_parent_gap_low_ticks")))


def find_setup_record(trade: Mapping[str, Any], *, configuration: str,
                      source: SetupRecordSource | None,
                      axis_value_ids: Mapping[str, str] | None = None,
                      in_verified_study: bool = False,
                      base_configuration: str | None = None) -> SetupRecord | None:
    """The saved setup record for one funded trade, or ``None`` (never re-computed)."""

    if source is None or source.frame.empty:
        return None
    frame = source.frame
    trade_id = str(trade.get("strategy_trade_id") or "")
    if in_verified_study and configuration in source.members and trade_id:
        rows = frame[(frame["profile"] == configuration) & (frame["trade_id"] == trade_id)]
        rows = rows[[_has_geometry(r) for _, r in rows.iterrows()]]
        if len(rows) == 1:
            return _record(rows.iloc[0], link="exact", same_target=True, distance=0)
    direction = str(trade.get("direction") or "").upper()
    family = source.family(configuration, base_configuration)
    if (not direction or family is None or trade.get("entry_ticks") is None
            or trade.get("stop_ticks") is None or not trade.get("entry_utc")):
        return None
    key = entry_match_key(trade["entry_utc"], direction, family, trade["entry_ticks"])
    rows = frame[(frame["entry_match_key"] == key)
                 & (frame["stop_ticks"].astype(float) == float(trade["stop_ticks"]))]
    rows = rows[[_has_geometry(r) for _, r in rows.iterrows()]]
    if rows.empty:
        return None
    target = trade.get("target_ticks")
    ids = dict(axis_value_ids or {})
    ranked = []
    for _, row in rows.iterrows():
        same_target = target is not None and _ticks(row.get("target_ticks")) == int(target)
        member = source.members.get(str(row["profile"]))
        distance = (settings_distance(ids, member, source.baseline)
                    if member is not None and ids else None)
        ranked.append(((0 if same_target else 1, 99 if distance is None else distance,
                        str(row["profile"]), str(row["trade_id"])), row, same_target, distance))
    ranked.sort(key=lambda item: item[0])
    _, row, same_target, distance = ranked[0]
    return _record(row, link="same_execution", same_target=same_target, distance=distance)
