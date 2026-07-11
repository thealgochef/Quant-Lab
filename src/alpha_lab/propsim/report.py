"""Report assembly: the (fill column × breach mode) matrix + human table + JSON."""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import date, datetime
from typing import Any

from alpha_lab.propsim.bootstrap import run_bootstrap, wilson_interval
from alpha_lab.propsim.engine import BREACH_MODES, FILL_COLUMNS, group_by_day, walk_days
from alpha_lab.propsim.loaders import LoadedTrades
from alpha_lab.propsim.models import Ruleset


def _sanitize(value: Any) -> Any:
    """JSON-safe: NaN/inf -> None, dates -> ISO strings, recursively."""
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def build_report(
    loaded: LoadedTrades,
    preset_name: str,
    ruleset: Ruleset,
    *,
    columns: tuple[str, ...] = FILL_COLUMNS,
    n_runs: int = 10_000,
    seed: int = 42,
    max_days: int = 1_000,
) -> dict:
    """Run the full matrix (both breach modes always) and assemble the payload."""
    day_blocks = group_by_day(loaded.trades)
    trades = loaded.trades

    pool: dict[str, Any] = {
        "n_trades": len(trades),
        "n_days": len(day_blocks),
        "first_day": day_blocks[0][0] if day_blocks else None,
        "last_day": day_blocks[-1][0] if day_blocks else None,
        "trades_per_day": {str(day): len(dt) for day, dt in day_blocks},
        "excursions_available": loaded.excursions_available,
        "trades_with_excursions": sum(1 for t in trades if t.mae_pts is not None),
    }
    for column in columns:
        pts = [
            t.points_conservative if column == "conservative" else t.points_optimistic
            for t in trades
        ]
        wins = sum(1 for p in pts if p > 0)
        ci_low, ci_high = wilson_interval(wins, len(pts))
        pool[column] = {
            "wins": wins,
            "win_rate": (wins / len(pts)) if pts else None,
            # The DATA-level binomial interval — with a tiny trade pool this is
            # the dominant uncertainty, not the Monte Carlo error.
            "win_rate_ci95": [ci_low, ci_high] if pts else None,
            "total_points": sum(pts),
            "mean_points": (sum(pts) / len(pts)) if pts else None,
        }

    matrix: dict[str, dict[str, Any]] = {}
    for column in columns:
        matrix[column] = {}
        for mode in BREACH_MODES:
            degraded = (
                mode == "unrealized_adverse_first" and not loaded.excursions_available
            )
            cell: dict[str, Any] = {
                "degraded_to_realized_only": degraded,
                "degradation_reason": loaded.degradation_reason if degraded else None,
                "historical": None,
                "bootstrap": None,
            }
            if day_blocks:
                cell["historical"] = asdict(
                    walk_days(day_blocks, ruleset, column=column, breach_mode=mode)
                )
                cell["bootstrap"] = asdict(
                    run_bootstrap(
                        day_blocks,
                        ruleset,
                        column=column,
                        breach_mode=mode,
                        n_runs=n_runs,
                        seed=seed,
                        max_days=max_days,
                    )
                )
            matrix[column][mode] = cell

    return _sanitize(
        {
            "source": loaded.source,
            "preset": preset_name,
            "ruleset": asdict(ruleset),
            "monte_carlo": {"n_runs": n_runs, "seed": seed, "max_days": max_days},
            "pool": pool,
            "matrix": matrix,
            "notes": list(loaded.notes),
        }
    )


def _fmt(value: Any, width: int, precision: int | None = None) -> str:
    if value is None:
        text = "-"
    elif precision is not None and isinstance(value, (int, float)):
        text = f"{value:.{precision}f}"
    else:
        text = str(value)
    return text.rjust(width)


def format_human(report: dict) -> str:
    """A fixed-width summary table of the full matrix."""
    rs = report["ruleset"]
    pool = report["pool"]
    lines = [
        f"PROP-SIM walker — preset {report['preset']} (source: {report['source']})",
        (
            f"  ruleset: start {rs['starting_balance']:.0f} / target "
            f"{rs['profit_target']:.0f} / trail {rs['trail_amount']:.0f} "
            f"({rs['trail_style']}, locks_at_start={rs['trail_locks_at_start']}) / "
            f"DLL {rs['dll_amount']} (soft={rs['dll_soft']}) / consistency "
            f"{rs['consistency_pct']}% / min_days {rs['min_days']} / "
            f"point_value {rs['point_value']}"
        ),
        (
            f"  pool: {pool['n_trades']} trades over {pool['n_days']} days "
            f"({pool['first_day']} .. {pool['last_day']}); excursions on "
            f"{pool['trades_with_excursions']}/{pool['n_trades']} trades"
        ),
        (
            f"  monte carlo: N={report['monte_carlo']['n_runs']} "
            f"seed={report['monte_carlo']['seed']} "
            f"max_days={report['monte_carlo']['max_days']}"
        ),
    ]
    for column in report["matrix"]:
        stats = pool.get(column) or {}
        # mean/total can sanitize to None (NaN points) even when win_rate is a
        # valid float — guard every formatted field, not just win_rate.
        if stats.get("win_rate") is not None and stats.get("mean_points") is not None:
            ci = stats.get("win_rate_ci95") or [None, None]
            lines.append(
                f"  [{column}] win rate {stats['wins']}/{pool['n_trades']} = "
                f"{stats['win_rate']:.3f} (binomial 95% CI "
                f"{ci[0]:.3f}..{ci[1]:.3f}); mean {stats['mean_points']:.2f} pts, "
                f"total {stats['total_points']:.2f} pts"
            )
    header = (
        f"  {'column':<12} {'breach mode':<25} {'hist':<10} "
        f"{'P(pass)':>8} {'95% CI':>15} {'P(bust)':>8} {'P(inc)':>7} "
        f"{'d2p med':>8} {'p10':>6} {'p90':>6} {'d2b med':>8}  notes"
    )
    lines.append("")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for column, cells in report["matrix"].items():
        for mode, cell in cells.items():
            hist = cell.get("historical") or {}
            boot = cell.get("bootstrap") or {}
            note = "DEGRADED->realized" if cell.get("degraded_to_realized_only") else ""
            ci_text = (
                f"{boot.get('pass_ci95_low'):.3f}..{boot.get('pass_ci95_high'):.3f}"
                if boot.get("pass_ci95_low") is not None
                else "-"
            )
            lines.append(
                f"  {column:<12} {mode:<25} {str(hist.get('verdict', '-')):<10} "
                f"{_fmt(boot.get('p_pass'), 8, 4)} {ci_text:>15} "
                f"{_fmt(boot.get('p_bust'), 8, 4)} {_fmt(boot.get('p_incomplete'), 7, 3)} "
                f"{_fmt(boot.get('days_to_pass_median'), 8, 1)} "
                f"{_fmt(boot.get('days_to_pass_p10'), 6, 1)} "
                f"{_fmt(boot.get('days_to_pass_p90'), 6, 1)} "
                f"{_fmt(boot.get('days_to_bust_median'), 8, 1)}  {note}"
            )
    if report.get("notes"):
        lines.append("")
        lines.append("  notes:")
        lines.extend(f"    - {note}" for note in report["notes"])
    return "\n".join(lines)
