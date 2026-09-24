"""Strategy configurations and market inputs for the configuration comparison.

The comparison replays configurations of a completed, externally verified
strategy study (today: the September 18 daily-close study, 32 approved
configurations on 10 warmup plus 107 evaluation dates). Nothing is copied from
its trade list: every configuration is RESOLVED again through the normal study
configurator path (registry value ids -> section overrides -> the baseline
profile -> the canonical section) and must reproduce the approved section
hash, and Strategy-Core is replayed from the same trusted cached day artifacts
under the same approved read policy. The package's saved trades are used only
to prove the replay is equivalent when no account interferes.

Everything read is verified first: the package receipt and manifest hashes
(``sources.open_verified_package``) plus the manifest hashes of the configuration
files, and each approval envelope through the store's verified reader.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_lab.propsim.funded.campaign import TradingDay
from alpha_lab.propsim.funded.pair_engine import DayInput
from alpha_lab.propsim.funded.sources import (
    ARCHIVE_ROOT,
    PackageError,
    VerifiedPackage,
    load_trading_days,
    open_verified_package,
)

__all__ = [
    "BASELINE_PROFILE",
    "ComparisonConfiguration",
    "ComparisonSource",
    "open_comparison_source",
    "discover_comparison_sources",
    "resolve_configuration",
    "iter_days",
    "reference_trades",
    "core_trade_key",
    "variation_configurations",
    "variation_name",
    "variation_label",
    "normal_replay_trades",
]

BASELINE_PROFILE = "ifvg_v2_doc_default_fresh_static_1r"
_CONFIG_FILES = ("run_context.json", "configs.json")


@dataclass(frozen=True)
class ComparisonConfiguration:
    name: str
    display_name: str
    axes: dict[str, Any]
    axis_value_ids: dict[str, str]
    resolved_section_config_hash: str
    approval_id: str
    core_replay_id: str


@dataclass(frozen=True)
class ComparisonSource:
    package: VerifiedPackage
    work_root: Path
    configurations: tuple[ComparisonConfiguration, ...]
    warmup_dates: tuple[str, ...]
    evaluation_dates: tuple[str, ...]
    cutoff_utc: str
    trading_days: tuple[TradingDay, ...]
    start_ns: int
    cutoff_ns: int

    @property
    def by_name(self) -> dict[str, ComparisonConfiguration]:
        return {c.name: c for c in self.configurations}

    @property
    def dates(self) -> tuple[str, ...]:
        return self.warmup_dates + self.evaluation_dates


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def open_comparison_source(package_root: Path) -> ComparisonSource:
    package = open_verified_package(package_root)
    manifest = json.loads((package.root / "MANIFEST.json").read_text(encoding="utf-8"))
    files = {entry["path"]: entry for entry in manifest.get("files", [])}
    for rel in _CONFIG_FILES:
        entry = files.get(rel)
        if entry is None or _sha256(package.root / rel) != entry["sha256"]:
            raise PackageError(f"{rel} does not match its verified manifest hash")
    context = json.loads((package.root / "run_context.json").read_text(encoding="utf-8"))
    batch = context["frozen_batch"]
    configurations = tuple(
        ComparisonConfiguration(
            name=m["name"], display_name=m["display_name"], axes=dict(m["axes"]),
            axis_value_ids=dict(m["spec"]["axis_value_ids"]),
            resolved_section_config_hash=m["resolved_section_config_hash"],
            approval_id=m["approval_id"], core_replay_id=m["core_replay_id"],
        )
        for m in batch["members"]
    )
    days, start_ns, cutoff_ns, (first, last) = load_trading_days(package)
    evaluation = tuple(batch["evaluation_dates"])
    if (first, last) != (evaluation[0], evaluation[-1]):
        raise PackageError("the trading schedule differs from the frozen evaluation dates")
    return ComparisonSource(
        package=package, work_root=package.root.parent, configurations=configurations,
        warmup_dates=tuple(batch["warmup_dates"]), evaluation_dates=evaluation,
        cutoff_utc=str(batch["cutoff_utc"]), trading_days=days, start_ns=start_ns,
        cutoff_ns=cutoff_ns,
    )


def discover_comparison_sources(archive_root: Path | None = None) -> list[ComparisonSource]:
    archive_root = ARCHIVE_ROOT if archive_root is None else Path(archive_root)
    found = []
    for manifest in sorted(archive_root.glob("*/final_extracted_*/MANIFEST.json")):
        root = manifest.parent
        if not (root / "run_context.json").is_file():
            continue
        try:
            found.append(open_comparison_source(root))
        except (PackageError, OSError, ValueError, KeyError):
            continue
    return found


def resolve_configuration(axis_value_ids: dict[str, str]) -> tuple[Any, Any]:
    """The normal configurator resolution: value ids -> canonical section."""

    from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
    from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import resolve_axis_overrides
    from alpha_lab.agents.data_infra.ifvg.search.identities import canonicalize_section

    overrides = resolve_axis_overrides(axis_value_ids)
    resolved = resolve_profile_config({"profile_name": BASELINE_PROFILE,
                                       "section_overrides": dict(overrides)})
    section = canonicalize_section(resolved.section)
    return section, IfvgCaptureConfig(section=section)


def _approval_dates(source: ComparisonSource, config: ComparisonConfiguration) -> tuple[str, ...]:
    from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
    from alpha_lab.agents.data_infra.ifvg.search.strategy_approval import (
        StrategySearchApprovalEnvelope,
    )

    envelope = load_verified_envelope(source.work_root / "store", "strategy_search_approvals",
                                      config.approval_id, StrategySearchApprovalEnvelope)
    return tuple(envelope.payload.artifact_provenance_dates)


def iter_days(source: ComparisonSource, config: ComparisonConfiguration, *,
              cfg: Any, cache_cfg: Any = None) -> Iterator[DayInput]:
    """Trusted cached day inputs, chained in date order under the approved read policy.

    ``cache_cfg``: read the verified cache of a configuration whose chart set is a
    SUPERSET of ``cfg``'s and keep only ``cfg``'s charts. Strategy-Core's bar builder
    aggregates every timeframe independently from the same one-minute rows and the
    level timeline uses one-minute bars only, so the kept bars are exactly what a
    fresh build for ``cfg`` produces (also compared on real days).
    """

    from alpha_lab.agents.data_infra.ifvg.dataset import _chained_seeds
    from alpha_lab.agents.data_infra.ifvg.day_artifacts import (
        DaySeeds,
        levels_for_from_frame,
        load_day_artifacts,
    )
    from alpha_lab.agents.data_infra.ifvg.development_access import DevelopmentReplayPolicy
    from alpha_lab.agents.data_infra.ifvg.search.child_replay import (
        ArtifactProvenanceReadAdapter,
    )

    dates = source.dates
    policy = ArtifactProvenanceReadAdapter(
        DevelopmentReplayPolicy(dates),
        artifact_provenance_dates=_approval_dates(source, config))
    schedule = {d.trading_day: d for d in source.trading_days}
    evaluation = set(source.evaluation_dates)
    read_cfg = cfg if cache_cfg is None else cache_cfg
    wanted = set(cfg.timeframes_seconds())
    if not wanted <= set(read_cfg.timeframes_seconds()):
        raise ValueError("the cache configuration lacks some of this configuration's charts")
    previous = None
    for index, day in enumerate(dates):
        expected = _chained_seeds(previous) or DaySeeds(None, None, None, None)
        artifacts = load_day_artifacts(day, read_cfg, expected_seeds=expected,
                                       access_policy=policy)
        if artifacts is None:
            raise PermissionError(f"trusted day artifacts are unavailable for {day}")
        previous = artifacts
        bars_by_tf: dict[int, list] = {}
        for bar in artifacts.bars:
            if bar.timeframe_ticks in wanted:
                bars_by_tf.setdefault(bar.timeframe_ticks, []).append(bar)
        yield DayInput(
            trading_day=day, bars_by_tf=bars_by_tf,
            levels_for=levels_for_from_frame(artifacts.level_timeline),
            is_evaluation=day in evaluation, schedule=schedule.get(day),
            exhausted=index == len(dates) - 1,
        )
    policy.assert_zero_forbidden_access()


def reference_trades(source: ComparisonSource, name: str) -> list[tuple]:
    """The saved study's trades of one configuration (the equivalence target)."""

    frame = pd.read_csv(source.package.root / "data/trades.csv", low_memory=False)
    frame = frame[frame["profile"] == name].sort_values(["entry_ts_utc", "trade_id"])
    return [(_iso(r.entry_ts_utc), int(r.entry_ticks), int(r.stop_ticks), str(r.resolution),
             int(r.exit_ticks)) for r in frame.itertuples()]


def core_trade_key(trade: dict) -> tuple:
    return (_iso(trade["entry_ts_utc"]), trade["entry_ticks"], trade["stop_ticks"],
            trade["resolution"], trade["exit_ticks"])


def _iso(value) -> str:
    return pd.Timestamp(value).tz_convert("UTC").isoformat() if pd.Timestamp(
        value).tzinfo else pd.Timestamp(value).tz_localize("UTC").isoformat()



# ── variations around a base configuration (not in the verified study) ────────

#: short name tokens per registry value (readable configuration names)
_TOKENS = {
    "enabled_entry_sessions.asia-london-ny": "S0",
    "enabled_entry_sessions.all_open_market_v1": "S1",
    "enabled_entry_sessions.daytime_chicago_0700_1555_v1": "S2",
    "enabled_entry_sessions.morning_chicago_0700_1030_v1": "S3",
    "tp_r_multiple.1.0": "T1", "tp_r_multiple.2.0": "T2", "tp_r_multiple.3.0": "T3",
    "htf_timeframes.1H-4H": "H14", "htf_timeframes.1H": "H1",
    "parent_timeframes.1m-3m-5m-10m-15m-30m": "P1",
    "parent_timeframes.1m-5m-10m-15m-30m": "P5",
    "parent_timeframes.3m-5m-10m-15m-30m": "P0",
    "enable_shorts.false": "L", "enable_shorts.true": "LS",
    "exit_policy.fixed_target_v1": "FX",
    "exit_policy.scale_out_half_breakeven_hold_to_close_v1": "SO",
}
_ORDER = ("enabled_entry_sessions", "tp_r_multiple", "htf_timeframes", "parent_timeframes",
          "enable_shorts", "exit_policy")
_PLAIN = {
    "S0": "Original three windows", "S1": "All open-market hours",
    "S2": "Daytime 7:00 AM to 3:55 PM Chicago", "S3": "Morning 7:00 AM to 10:30 AM Chicago",
    "T1": "target 1R", "T2": "target 2R", "T3": "target 3R",
    "H14": "one-hour and four-hour gaps", "H1": "one-hour gaps only",
    "P1": "parents incl. one- and three-minute", "P5": "parents incl. one-minute, no three-minute",
    "P0": "parents without one-minute",
    "L": "long only", "LS": "long and short",
    "FX": "whole position exits at the target",
    "SO": "half at 1R, rest at break-even to the close",
}


def _named_ids(ids: dict[str, str]) -> dict[str, str]:
    return {a: ids.get(a, f"{a}.{_default_token(a)}") for a in _ORDER}


def variation_value_label(value_id: str) -> str | None:
    """Short plain name of one variation value (None when it has none)."""

    token = _TOKENS.get(value_id)
    if token is None:
        return None
    text = _PLAIN[token]
    return text[:1].upper() + text[1:]


def variation_name(axis_value_ids: dict[str, str]) -> str:
    named = _named_ids(axis_value_ids)
    return "-".join(_TOKENS.get(named[a], named[a].split(".", 1)[1]) for a in _ORDER)


def variation_label(axis_value_ids: dict[str, str]) -> str:
    named = _named_ids(axis_value_ids)
    return "; ".join(_PLAIN.get(_TOKENS.get(named[a], ""), named[a]) for a in _ORDER)


def variation_configurations(source: ComparisonSource, base_name: str,
                             variants: list[dict[str, str]]
                             ) -> tuple[ComparisonConfiguration, ...]:
    """Each variant = the base configuration's value ids with some replaced.

    Resolved through the normal configurator path. The base configuration's
    approval supplies only the verified cache read policy (same dates, same
    superset cache); these variants carry no strategy-search approval of their
    own — their authorization is the owner's approval of the comparison plan.
    """

    base = source.by_name[base_name]
    in_study = {c.resolved_section_config_hash: c for c in source.configurations}
    out = []
    for changes in variants:
        ids = {**base.axis_value_ids, **changes}
        _section, cfg = resolve_configuration(ids)
        if cfg.profile_hash in in_study:
            out.append(in_study[cfg.profile_hash])  # already a verified study configuration
            continue
        out.append(ComparisonConfiguration(
            name=variation_name(ids), display_name=variation_label(ids), axes=dict(changes),
            axis_value_ids=ids, resolved_section_config_hash=cfg.profile_hash,
            approval_id=base.approval_id, core_replay_id=""))
    names = [c.name for c in out]
    if len(set(names)) != len(names):
        raise ValueError("two variants resolve to the same configuration name")
    return tuple(out)


def _default_token(axis: str) -> str:
    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import SEARCH_AXIS_REGISTRY_V1

    spec = SEARCH_AXIS_REGISTRY_V1.get(axis)
    if spec is not None:
        return spec.baseline_value_id.split(".", 1)[1]
    return "fixed_target_v1" if axis == "exit_policy" else "default"


def normal_replay_trades(days: list[DayInput], section: Any, cfg: Any) -> list[tuple]:
    """The same days through Strategy-Core's normal ``run_day`` chain (no account)."""

    from datetime import date as _date

    from strategy_core.strategies.ifvg_smc.replay import run_day

    seed = None
    trades = []
    for index, day in enumerate(days):
        result = run_day(day.bars_by_tf, section=section, seed=seed,
                         trading_day=_date.fromisoformat(day.trading_day),
                         tick_size=cfg.tick_size, levels_for=day.levels_for,
                         dataset_exhausted=index == len(days) - 1)
        seed = result.end_seed
        for emission in result.emissions:
            if emission.kind == "executed_trade":
                r = emission.record
                trades.append((_iso(r.entry_ts_utc), int(r.entry_ticks), int(r.stop_ticks),
                               str(r.resolution),
                               None if r.exit_ticks is None else int(r.exit_ticks)))
    return trades
