"""Prop-firm ruleset presets. Presets are DATA — additions need no code.

Preset one (ratified in the PROP-SIM window; recorded in the SC PROGRESS
ledger): TopStep 50K. Consistency is best day ≤ pct × TOTAL profit — this
supersedes the alpha_lab-era ``config/prop_firms.yaml`` comment ("50% of
profit target").

PRESETS window (owner-confirmed checkout data, 2026-07-10 dashboard screens):
``apex_50k_eod``, ``apex_50k_intraday``, ``tpt_50k_test``. Two Apex
parameters carry a ⚠ VERIFY AT DASHBOARD note — flip them when confirmed
(data-only change):

- ⚠ verify-lock: ``trail_locks_at_start=True`` on both Apex presets is not
  yet dashboard-confirmed (whether the Apex floor stops trailing at the
  starting balance or keeps trailing above it).
- ⚠ verify-soft-vs-hard: ``apex_50k_eod`` records the DLL as HARD
  (``dll_hard=True`` — a touch is a bust, not a halt); not yet
  dashboard-confirmed.
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
        dll_hard=False,
        consistency_pct=50.0,
        min_days=None,
        max_eval_days=None,
        point_value=20.0,
    ),
    "apex_50k_eod": Ruleset(
        starting_balance=50_000.0,
        profit_target=3_000.0,
        trail_amount=2_000.0,
        trail_style="eod_floor_realtime_breach",
        trail_locks_at_start=True,  # ⚠ verify-lock (see module docstring)
        dll_amount=1_000.0,
        dll_hard=True,  # ⚠ verify-soft-vs-hard (see module docstring)
        consistency_pct=None,  # no consistency rule in the eval
        min_days=None,
        max_eval_days=30,
        point_value=20.0,
    ),
    "apex_50k_intraday": Ruleset(
        starting_balance=50_000.0,
        profit_target=3_000.0,
        trail_amount=2_000.0,
        trail_style="intraday_peak_trail",
        trail_locks_at_start=True,  # ⚠ verify-lock (see module docstring)
        dll_amount=None,
        dll_hard=False,
        consistency_pct=None,
        min_days=None,
        max_eval_days=30,
        point_value=20.0,
    ),
    "tpt_50k_test": Ruleset(
        starting_balance=50_000.0,
        profit_target=3_000.0,
        trail_amount=2_000.0,
        trail_style="eod_floor_realtime_breach",
        trail_locks_at_start=True,
        dll_amount=None,
        dll_hard=False,
        consistency_pct=50.0,
        min_days=5,
        max_eval_days=None,
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
