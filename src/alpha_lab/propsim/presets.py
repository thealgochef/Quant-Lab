"""Prop-firm ruleset presets. Presets are DATA — additions need no code.

Preset one (ratified in the PROP-SIM window; recorded in the SC PROGRESS
ledger): TopStep 50K. Consistency is best day ≤ pct × TOTAL profit — this
supersedes the alpha_lab-era ``config/prop_firms.yaml`` comment ("50% of
profit target").
"""

from __future__ import annotations

from alpha_lab.propsim.models import Ruleset

PRESETS: dict[str, Ruleset] = {
    "topstep_50k": Ruleset(
        starting_balance=50_000.0,
        profit_target=3_000.0,
        trail_amount=2_000.0,
        trail_style="eod_floor_realtime_breach",
        trail_locks_at_start=True,
        dll_amount=1_000.0,
        dll_soft=True,
        consistency_pct=50.0,
        min_days=None,
        point_value=20.0,
    ),
}


def ruleset_from_preset(name: str) -> Ruleset:
    """Look up a preset ruleset by name; unknown names fail loud."""
    try:
        return PRESETS[name]
    except KeyError as exc:
        known = ", ".join(sorted(PRESETS))
        msg = f"Unknown propsim preset {name!r}. Known presets: {known}"
        raise ValueError(msg) from exc
