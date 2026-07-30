"""Funnel + label-family reports (the window's primary result artifacts).

``IFVG_FUNNEL.md`` — stage counts, drop reasons, per-session/per-TF splits,
slot-contention stats. ``IFVG_LABELS.md`` — label distributions and naive
expectancies (gross and net of NQ friction) per R family / entry family /
session, warmup excluded, plus the ``doc_default_pass`` baseline cut (the
hardcoded ifvg-strat.md defaults expressed as a row FILTER over emitted
measurements — the alternative the ML gate must beat).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

__all__ = ["doc_default_pass", "write_funnel_report", "write_labels_report"]


def doc_default_pass(ds: pd.DataFrame) -> pd.Series:
    """ifvg-strat.md tuned defaults as a boolean filter over measurements:
    >=4-tick gaps at every recorded joint, <=80-tick distances, <=80 bars
    post-inversion, entry inside a doc session."""
    checks = pd.Series(True, index=ds.index)
    for col, floor in (
        ("tap_fvg_size_ticks", 4),
        ("parent_fvg_size_ticks", 4),
        ("opp_fvg_size_ticks", 4),
    ):
        if col in ds:
            checks &= ds[col].fillna(0) >= floor
    for col, cap in (
        ("parent_distance_to_htf_ticks", 80),
        ("opp_distance_to_parent_ticks", 80),
        ("bars_since_inversion", 80),
    ):
        if col in ds:
            checks &= ds[col].fillna(10**9) <= cap
    checks &= ds["session_doc"].fillna("none") != "none"
    return checks


#: The three trade-outcome resolutions. SEALED rule (IFVG-FIX F4): stage
#: counters may publish raw over the sealed range, but the tp/sl/eod
#: decomposition is an OUTCOME ratio over the sealed holdout and stays
#: unpublished — sealed sections carry only the combined count.
_SEALED_RESOLVED = ("resolved_tp", "resolved_sl", "resolved_eod")


def _seal_counters(counters: dict[str, int]) -> dict[str, int]:
    """Collapse the outcome decomposition for anything published about the
    sealed segment (see the ``_SEALED_RESOLVED`` rule above)."""
    out = {k: v for k, v in counters.items() if k not in _SEALED_RESOLVED}
    out["resolved_total_sealed"] = sum(counters.get(k, 0) for k in _SEALED_RESOLVED)
    return out


def _segment_of(day: str, chain_idx: int, warmup_days: int, sealed_start: str) -> str:
    if day >= sealed_start:  # seal wins over warmup: outcomes stay unpublished
        return "sealed"
    return "warmup" if chain_idx < warmup_days else "core"


def write_funnel_report(
    day_funnels: dict[str, dict[str, int]],
    out_md: Path,
    out_json: Path,
    *,
    warmup_days: int,
    sealed_start: str,
) -> dict[str, dict[str, int]]:
    """Three-way split (warmup / core / sealed) of the per-day funnel counters.

    ``day_funnels`` must be in chain order (warmup = the first ``warmup_days``
    chain positions). Returns the per-segment totals; everything published for
    the sealed segment goes through ``_seal_counters``."""
    totals: dict[str, dict[str, int]] = {"warmup": {}, "core": {}, "sealed": {}}
    seg_days: dict[str, int] = {"warmup": 0, "core": 0, "sealed": 0}
    day_entries: dict[str, dict] = {}
    for chain_idx, (day, counters) in enumerate(day_funnels.items()):
        segment = _segment_of(day, chain_idx, warmup_days, sealed_start)
        seg_days[segment] += 1
        for key, value in counters.items():
            totals[segment][key] = totals[segment].get(key, 0) + value
        day_entries[day] = {
            "segment": segment,
            "counters": _seal_counters(dict(counters)) if segment == "sealed" else dict(counters),
        }
    published = {
        seg: (_seal_counters(t) if seg == "sealed" else t) for seg, t in totals.items()
    }
    out_json.write_text(
        json.dumps({"segments": published, "days": day_entries}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    titles = {
        "warmup": f"Warmup totals (first {warmup_days} chain days)",
        "core": "Core totals (post-warmup, pre-seal)",
        "sealed": (
            f"Sealed totals ({sealed_start}+ holdout — stage counters only, "
            "resolutions undecomposed)"
        ),
    }
    lines = [
        "# IFVG capture funnel",
        "",
        f"Days in chain: {len(day_funnels)} — warmup {seg_days['warmup']}, "
        f"core {seg_days['core']}, sealed {seg_days['sealed']} "
        f"(sealed = trading day >= {sealed_start}; counts only, no outcome decomposition).",
    ]
    for seg in ("warmup", "core", "sealed"):
        lines += [
            "",
            f"## {titles[seg]}",
            "",
            "| counter | total | /day |",
            "|---|---|---|",
        ]
        ndays = max(1, seg_days[seg])
        for key in sorted(published[seg]):
            lines.append(f"| {key} | {published[seg][key]} | {published[seg][key] / ndays:.2f} |")
    active = sum(1 for c in day_funnels.values() if c.get("setups_born", 0) > 0)
    lines += [
        "",
        f"Days with at least one setup born: {active}/{len(day_funnels)}",
        "",
        "_Capture bounds are WIDE by design (ruling 9.13): every drop reason above is a"
        " measured cost, not a quality judgement._",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return published


def _expectancy_block(ds: pd.DataFrame, tag: str) -> list[str]:
    col_label, col_net = f"label_{tag}", f"realized_r_net_{tag}"
    sub = ds[ds[col_label] != "no_forward"]
    if sub.empty:
        return [f"- {tag}: no labeled rows"]
    dist = sub[col_label].value_counts().to_dict()
    net = sub[col_net].mean()
    win_rate = (sub[col_label] == "win").mean()
    return [
        f"- **{tag}** n={len(sub)} labels={dist} win_rate={win_rate:.3f} "
        f"mean_net_R={net:+.3f}"
    ]


def write_labels_report(ds: pd.DataFrame, out_md: Path) -> None:
    lines = ["# IFVG label families", ""]
    if ds.empty:
        lines.append("No entry candidates captured.")
        out_md.write_text("\n".join(lines), encoding="utf-8")
        return
    from .config import SEALED_HOLDOUT_START

    sealed = ds["trading_day"] >= SEALED_HOLDOUT_START
    core = ds[(~ds["is_warmup"]) & (~sealed)]
    lines += [
        f"Entry candidates: {len(ds)} total, {len(core)} post-warmup/pre-seal "
        f"({int(sealed.sum())} rows in the SEALED {SEALED_HOLDOUT_START}+ holdout are "
        f"captured but excluded from every statistic below; "
        f"{int(ds['selected'].sum())} selected by the doc-default family).",
        "",
        "_Entry price model: confirmation_close. Costs: 0.514 NQ points round-turn."
        " `eod_timeout` realizes at the day-end close. Warmup rows excluded below._",
        "",
        "_The `ifvg_retest` family's entry model is PROVISIONAL (bar-close placeholder,"
        " no CE/boundary entry references yet; its rows carry `entry_model_final=False`)"
        " — do not read it as comparable to the finished `fresh_fvg_continuation` model._",
        "",
    ]
    for family, group in core.groupby("entry_family"):
        lines += [f"## family = {family} (n={len(group)})", ""]
        for tag in ("r10", "r15", "r20"):
            lines += _expectancy_block(group, tag)
        lines.append("")
        for session_col in ("session_engine", "session_doc"):
            lines.append(f"### by {session_col}")
            for session, sgroup in group.groupby(session_col):
                block = _expectancy_block(sgroup, "r10")
                lines.append(f"- {session}: {block[0][2:]}")
            lines.append("")
    passing = core[doc_default_pass(core)]
    lines += [
        "## doc-defaults-as-filter baseline (the hardcoded alternative)",
        "",
        f"Rows passing every ifvg-strat.md tuned default: {len(passing)}/{len(core)}",
        "",
    ]
    for tag in ("r10",):
        lines += _expectancy_block(passing, tag)
    slip = core["entry_slippage_next_open_pts"].dropna()
    if len(slip):
        lines += [
            "",
            "## confirmation-close honesty",
            "",
            f"entry->next-1m-open slippage pts: mean {slip.mean():+.3f}, "
            f"p10 {slip.quantile(0.1):+.2f}, p90 {slip.quantile(0.9):+.2f} (n={len(slip)})",
        ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
