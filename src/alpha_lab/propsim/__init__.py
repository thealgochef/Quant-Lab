"""Prop-firm evaluation walker (PROP-SIM): pass-probability from equity paths.

Pure simulation core (`engine`, `bootstrap`) + source loaders (`loaders`) +
preset registry (`presets`) + CLI (`python -m alpha_lab.propsim`). Presets are
data — adding a firm needs no code (D-038 window; TopStep 50K is preset one).
"""

from alpha_lab.propsim.bootstrap import BootstrapSummary, run_bootstrap
from alpha_lab.propsim.engine import (
    BREACH_MODES,
    FILL_COLUMNS,
    EvaluationWalk,
    group_by_day,
    walk_days,
)
from alpha_lab.propsim.models import Ruleset, TradePath, WalkResult
from alpha_lab.propsim.presets import PRESETS, ruleset_from_preset

__all__ = [
    "BREACH_MODES",
    "FILL_COLUMNS",
    "PRESETS",
    "BootstrapSummary",
    "EvaluationWalk",
    "Ruleset",
    "TradePath",
    "WalkResult",
    "group_by_day",
    "ruleset_from_preset",
    "run_bootstrap",
    "walk_days",
]
