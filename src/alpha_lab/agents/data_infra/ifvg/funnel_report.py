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


def write_funnel_report(
    day_funnels: dict[str, dict[str, int]], out_md: Path, out_json: Path
) -> dict[str, int]:
    total: dict[str, int] = {}
    for counters in day_funnels.values():
        for key, value in counters.items():
            total[key] = total.get(key, 0) + value
    out_json.write_text(
        json.dumps({"days": day_funnels, "total": total}, indent=2, sort_keys=True)
    )
    lines = [
        "# IFVG capture funnel",
        "",
        f"Days in chain: {len(day_funnels)}",
        "",
        "## Totals (all days, warmup included)",
        "",
        "| counter | total | /day |",
        "|---|---|---|",
    ]
    ndays = max(1, len(day_funnels))
    for key in sorted(total):
        lines.append(f"| {key} | {total[key]} | {total[key] / ndays:.2f} |")
    active = sum(1 for c in day_funnels.values() if c.get("setups_born", 0) > 0)
    lines += [
        "",
        f"Days with at least one setup born: {active}/{len(day_funnels)}",
        "",
        "_Capture bounds are WIDE by design (ruling 9.13): every drop reason above is a"
        " measured cost, not a quality judgement._",
    ]
    out_md.write_text("\n".join(lines))
    return total


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
        out_md.write_text("\n".join(lines))
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
    out_md.write_text("\n".join(lines))
