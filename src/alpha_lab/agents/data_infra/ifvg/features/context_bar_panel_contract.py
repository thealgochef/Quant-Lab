"""Context-bar panel contract — the frozen formula set of
``IFVG_CONTEXT_BAR_PANEL_V1`` (R6.1; owner planning decision Q3).

A LEAF module: constants only (no artifact I/O, no registry). Everything
here is a ``proposed_protocol_default`` (owner decision 28) — an
AVAILABLE engineering capability, ``research_only_offline``, whose
model-bearing use requires owner ratification before FEATURE_ELIGIBLE.

The seven registered features (owner names; the intensity feature names
its source field per requirement 8) are computed on COMPLETED 5m or 15m
bars only, with an interval-relative 12-bar lookback inside one trading
day (the 18:00 ET boundary; named-session boundaries do NOT reset the
lookback), population statistics (``ddof = 0``), and 13 complete source
bars per window (the current bar + 12 prior). Any incomplete 1m source bar
in the 13-bar window invalidates the entire feature row
(``source_bar_incomplete``); a legitimate zero-denominator formula NaN keeps
the row VALID. Prohibited inputs by construction: labels, outcomes,
MFE/MAE, resolution, future-session statistics, post-entry evidence,
prop-account outcomes, and MBP-1 features.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Literal

from pydantic import Field

from ..replay_chart_store import REPLAY_TIMEFRAMES_SECONDS, RESAMPLE_RULE_ID
from ..search.identities import FrozenContract

__all__ = [
    "CONTEXT_BAR_PANEL_BLOCK_KEY",
    "CONTEXT_BAR_PANEL_BUNDLE_KEY",
    "PANEL_AS_OF_POLICY_ID_V1",
    "PanelAsOfPolicy",
    "PANEL_AS_OF_POLICY_REGISTRY",
    "PANEL_INTERVALS_SECONDS_V1",
    "CONTEXT_BAR_PANEL_FORMULA_VERSION",
    "CONTEXT_BAR_PANEL_MATERIALIZER_VERSION",
    "LOOKBACK_BARS",
    "MINIMUM_SOURCE_BARS",
    "STD_DDOF",
    "PARTIAL_BAR_POLICY_ID",
    "WARMUP_POLICY_ID",
    "LOOKBACK_POLICY_ID",
    "SESSION_RESET_POLICY_ID",
    "INTENSITY_SOURCE_FIELD",
    "SESSION_SCHEME_ID",
    "SESSION_CLASSIFICATION_INSTANT_POLICY",
    "PANEL_RESAMPLE_RULE_ID",
    "PANEL_ASSIGNMENT_MAX_STALENESS_INTERVALS",
    "CONTEXT_BAR_PANEL_FEATURES",
    "CONTEXT_BAR_PANEL_CATEGORICAL_FEATURES",
    "CONTEXT_BAR_PANEL_FEATURE_FORMULAS",
    "PANEL_MISSING_REASONS",
    "PANEL_ASSIGNMENT_MISSING_REASONS",
    "PANEL_SESSION_STATES",
    "PANEL_PROPOSED_STAMPS",
    "assert_panel_interval_registered",
]

CONTEXT_BAR_PANEL_BLOCK_KEY = "IFVG_CONTEXT_BAR_PANEL_V1"
CONTEXT_BAR_PANEL_BUNDLE_KEY = "BP0_CONTEXT_BAR_PANEL"

PANEL_AS_OF_POLICY_ID_V1 = "completed_bars_last_at_or_before_v1"
CONTEXT_BAR_PANEL_FORMULA_VERSION = "ifvg_context_bar_panel_formula_v1"
CONTEXT_BAR_PANEL_MATERIALIZER_VERSION = "context_bar_panel_materializer_v1"

#: The owner-registered panel intervals (decision 28): 5m and 15m — separate
#: protocols, artifacts, folds, and fits; never pooled.
PANEL_INTERVALS_SECONDS_V1: tuple[int, ...] = (300, 900)
if not set(PANEL_INTERVALS_SECONDS_V1) <= set(REPLAY_TIMEFRAMES_SECONDS):
    raise AssertionError("panel intervals must be ratified replay-chart timeframes")

#: interval-relative lookback: 12 × 5m and 12 × 15m are SEPARATE horizons
LOOKBACK_BARS = 12
#: the current bar + 12 prior completed bars (owner amendment)
MINIMUM_SOURCE_BARS = LOOKBACK_BARS + 1
#: population standard deviation everywhere (owner amendment)
STD_DDOF = 0
PARTIAL_BAR_POLICY_ID = "exclude_final_partial_bars_v1"
WARMUP_POLICY_ID = "trading_day_lookback_all_features_typed_null_v1"
LOOKBACK_POLICY_ID = "prior_completed_bars_same_trading_day_only"
SESSION_RESET_POLICY_ID = "trading_day_18et_reset_v1"
#: requirement 8: the intensity feature NAMES its source field
INTENSITY_SOURCE_FIELD: Literal["volume"] = "volume"
SESSION_SCHEME_ID = "ifvg_doc_session_scheme_et_v1"
#: The session OF A COMPLETED BAR is the session in force at the bar's
#: final instant (close − 1 µs): a bar closing exactly at a session open
#: contains no trading of that session and belongs to the preceding state.
SESSION_CLASSIFICATION_INSTANT_POLICY = "session_of_bar_final_instant_v1"
PANEL_RESAMPLE_RULE_ID = RESAMPLE_RULE_ID
#: V1 maximum staleness for the panel→candidate assignment: one interval
PANEL_ASSIGNMENT_MAX_STALENESS_INTERVALS = 1


class PanelAsOfPolicy(FrozenContract):
    """The registered completed-bars-only as-of / assignment policy."""

    policy_id: str
    bar_selection: Literal["last_completed_bar_close_at_or_before_as_of"]
    comparator: Literal["<="]
    same_trading_day_required: Literal[True]
    consulted_partition_for_fold_features: Literal["candidate_own_partition"]
    consulted_partition_for_descriptive: Literal["test_oos_only"]
    fold_tie_break: Literal["lowest_fold_index"]
    partial_final_bar: Literal["excluded_from_panel"]
    lookback_policy: Literal["prior_completed_bars_same_trading_day_only"]
    reset_policy: Literal["trading_day_18et_reset_v1"]
    max_staleness_intervals: int = Field(ge=1)


PANEL_AS_OF_POLICY_REGISTRY: MappingProxyType[str, PanelAsOfPolicy] = MappingProxyType(
    {
        PANEL_AS_OF_POLICY_ID_V1: PanelAsOfPolicy(
            policy_id=PANEL_AS_OF_POLICY_ID_V1,
            bar_selection="last_completed_bar_close_at_or_before_as_of",
            comparator="<=",
            same_trading_day_required=True,
            consulted_partition_for_fold_features="candidate_own_partition",
            consulted_partition_for_descriptive="test_oos_only",
            fold_tie_break="lowest_fold_index",
            partial_final_bar="excluded_from_panel",
            lookback_policy=LOOKBACK_POLICY_ID,
            reset_policy=SESSION_RESET_POLICY_ID,
            max_staleness_intervals=PANEL_ASSIGNMENT_MAX_STALENESS_INTERVALS,
        )
    }
)

#: The seven registered features, in the frozen order (owner Q3).
CONTEXT_BAR_PANEL_FEATURES: tuple[str, ...] = (
    "cbp_realized_range_12",
    "cbp_realized_volatility_12",
    "cbp_range_compression_ratio_12",
    "cbp_path_efficiency_12",
    "cbp_session_state",
    "cbp_volume_intensity_zscore_12",
    "cbp_bar_position_in_session",
)
CONTEXT_BAR_PANEL_CATEGORICAL_FEATURES: tuple[str, ...] = ("cbp_session_state",)

#: The frozen formula contract (ticks; Δc_i = close_i − close_{i−1} within
#: the trading day; N = 12; "prior" = bars t−N..t−1; every window needs the
#: 13 COMPLETE same-day source bars t−N..t).
CONTEXT_BAR_PANEL_FEATURE_FORMULAS: MappingProxyType[str, MappingProxyType[str, Any]] = (
    MappingProxyType(
        {
            "cbp_realized_range_12": MappingProxyType(
                {
                    "family": "volatility",
                    "formula": "mean(high_i - low_i), i in t-N+1..t",
                    "zero_denominator": "not_applicable",
                    "range": "[0, inf)",
                    "categorical": False,
                }
            ),
            "cbp_realized_volatility_12": MappingProxyType(
                {
                    "family": "volatility",
                    "formula": "std(dc_i, ddof=0), i in t-N+1..t",
                    "zero_denominator": "not_applicable",
                    "range": "[0, inf)",
                    "categorical": False,
                }
            ),
            "cbp_range_compression_ratio_12": MappingProxyType(
                {
                    "family": "compression",
                    "formula": "(high_t - low_t) / mean(range_{t-N..t-1})",
                    "zero_denominator": "formula_nan_row_valid",
                    "range": "[0, inf)",
                    "categorical": False,
                }
            ),
            "cbp_path_efficiency_12": MappingProxyType(
                {
                    "family": "trend_path_efficiency",
                    "formula": "|close_t - close_{t-N}| / sum(|dc_i|), i in t-N+1..t",
                    "zero_denominator": "formula_nan_row_valid",
                    "range": "[0, 1]",
                    "categorical": False,
                }
            ),
            "cbp_session_state": MappingProxyType(
                {
                    "family": "session_state",
                    "formula": (
                        "classify_session(bar_close_ts_utc - 1us, IFVG_DOC_SESSION_SCHEME)"
                        ".session — the session in force at the bar's final instant "
                        "(session_of_bar_final_instant_v1)"
                    ),
                    "zero_denominator": "not_applicable",
                    "range": "{asia, london, ny, none, closed}",
                    "categorical": True,
                }
            ),
            "cbp_volume_intensity_zscore_12": MappingProxyType(
                {
                    "family": "intensity",
                    "intensity_source_field": INTENSITY_SOURCE_FIELD,
                    "formula": (
                        "(volume_t - mean(volume_{t-N..t-1})) / std(volume_{t-N..t-1}, ddof=0)"
                    ),
                    "zero_denominator": "formula_nan_row_valid",
                    "range": "(-inf, inf)",
                    "categorical": False,
                }
            ),
            "cbp_bar_position_in_session": MappingProxyType(
                {
                    "family": "session_state",
                    "formula": (
                        "(bar_close - session_open) / (session_close - session_open) "
                        "for the named session of the bar close, clamped to [0, 1]; "
                        "none/closed -> NaN"
                    ),
                    "zero_denominator": "not_applicable",
                    "range": "[0, 1]",
                    "categorical": False,
                }
            ),
        }
    )
)
if tuple(CONTEXT_BAR_PANEL_FEATURE_FORMULAS) != CONTEXT_BAR_PANEL_FEATURES:
    raise AssertionError("the formula contract must cover the seven features in order")

#: Row-level typed missing reasons (every one of the seven features is null).
PANEL_MISSING_REASONS: tuple[str, ...] = (
    "insufficient_trading_day_lookback",
    "lookback_window_gap",
    "source_bar_incomplete",
)
#: Panel→candidate assignment typed reasons (owner plan-review correction 2).
#: R6.1-FIX (plan §3.3, F-04): ``candidate_as_of_missing`` types a candidate
#: whose stage anchor is null — the candidate is PRESERVED with this reason,
#: never dropped and never a population refusal.
PANEL_ASSIGNMENT_MISSING_REASONS: tuple[str, ...] = (
    "panel_warmup",
    "no_completed_panel_bar",
    "panel_gap",
    "panel_stale",
    "panel_source_bar_incomplete",
    "coverage_gap",
    "candidate_as_of_missing",
)
PANEL_SESSION_STATES: tuple[str, ...] = ("asia", "london", "ny", "none", "closed")

#: Decision-28 stamps (values single-sourced from this leaf).
PANEL_PROPOSED_STAMPS: MappingProxyType[str, Any] = MappingProxyType(
    {
        "panel_interval_seconds_registered": PANEL_INTERVALS_SECONDS_V1,
        "panel_partial_bar_policy": PARTIAL_BAR_POLICY_ID,
        "panel_feature_set_v1": CONTEXT_BAR_PANEL_FEATURES,
        "panel_feature_lookback_bars": LOOKBACK_BARS,
        "panel_minimum_source_bars": MINIMUM_SOURCE_BARS,
        "panel_std_ddof": STD_DDOF,
        "panel_warmup_policy": WARMUP_POLICY_ID,
        "panel_session_scheme": SESSION_SCHEME_ID,
        "panel_intensity_source_field": INTENSITY_SOURCE_FIELD,
        "panel_assignment_max_staleness_intervals": PANEL_ASSIGNMENT_MAX_STALENESS_INTERVALS,
    }
)


def assert_panel_interval_registered(interval_seconds: int) -> int:
    """Refuse any interval outside the owner-registered set."""

    if int(interval_seconds) not in PANEL_INTERVALS_SECONDS_V1:
        raise ValueError(
            f"panel_interval_seconds={interval_seconds} is not owner-registered; "
            f"registered: {PANEL_INTERVALS_SECONDS_V1}"
        )
    return int(interval_seconds)
