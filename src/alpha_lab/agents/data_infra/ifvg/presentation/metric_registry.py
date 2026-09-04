"""The metric metadata registry (UI-3; plan §6.2) — pure.

One ``MetricSpec`` per displayed technical key: the human name, the
definition, the formula or persisted source, the unit, the directionality
(from ``OBJECTIVE_DIRECTIONS`` wherever the metric is a charter objective)
and, when one exists IN THE CODE, the reference the value is read against.
``evaluate_metric`` turns a value into a ``MetricReading`` — a typed status,
an interpretation sentence, the caveat and the evidence reference — through
deterministic rules and never through an invented threshold:

* the selected resolved gate (strategy / prop / robustness thresholds):
  PASS / FAIL by the registered direction; no gate selected → INFORMATIONAL;
* a boundary (Brier against the persisted prevalence-reference Brier; Brier
  skill against 0): strict — equality is INCONCLUSIVE;
* the 0.5 chance line of AUC: a DIRECTION only, never a band → INFORMATIONAL;
* calibration targets (slope 1, intercept 0): the distance only, no good / bad;
* a registered sample-adequacy minimum: below it is INCONCLUSIVE (insufficient
  evidence), never a failure;
* a report ``limit`` (capacity / performance): observed ≤ limit PASS, else FAIL;
* a MEASURED access counter: 0 PASS, nonzero FAIL; a policy-enforced
  ``protected_*`` zero: INFORMATIONAL with the caveat (the UI-1 ruling);
* a 95 % interval that crosses zero: INCONCLUSIVE;
* ``None`` / non-numeric / unevaluated (default-only) evidence: UNAVAILABLE —
  never PASS.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from ..search.charter import OBJECTIVE_DIRECTIONS
from ..search.identities import FrozenContract
from .status_vocabulary import UiStatus, status_chip, status_from_gate

__all__ = [
    "GATE_FIELD_METRICS",
    "METRIC_SPECS",
    "METRIC_UNITS",
    "MetricReading",
    "MetricSpec",
    "ReferenceKind",
    "ReferenceSpec",
    "describe",
    "evaluate_gate_flag",
    "evaluate_interval",
    "evaluate_metric",
    "format_value",
    "unavailable_reading",
    "gate_metric_key",
    "metric_keys",
    "metric_keys_for_surface",
]

Direction = Literal["higher_better", "lower_better", "target", "descriptive"]


class ReferenceKind(StrEnum):
    #: the selected resolved gate threshold (supplied at evaluation)
    GATE = "gate"
    #: a strict boundary (0 skill; the persisted reference Brier)
    BOUNDARY = "boundary"
    #: the 0.5 chance line — direction only, never a band
    CHANCE_LINE = "chance_line"
    #: a target value — the distance only, no good / bad
    TARGET = "target"
    #: a registered sample-adequacy minimum — below it is inconclusive
    ADEQUACY_MINIMUM = "adequacy_minimum"
    #: a persisted report limit — observed ≤ limit passes
    LIMIT = "limit"
    #: a measured counter that must be zero
    MEASURED_ZERO = "measured_zero"
    #: a policy-enforced zero written before any path exists — informational
    POLICY_ZERO = "policy_zero"


class ReferenceSpec(FrozenContract):
    kind: ReferenceKind
    #: where the reference comes from (a symbol path or a persisted report key)
    source: str
    #: a fixed reference value; ``None`` when it is supplied at evaluation
    value: float | None = None


#: The registered display units (``format_value`` renders by unit).
METRIC_UNITS: frozenset[str] = frozenset(
    {
        "count",
        "fraction",
        "probability",
        "R",
        "USD",
        "days",
        "ratio",
        "score",
        "bytes",
        "ms",
        "ticks",
        "bars",
        "flag",
    }
)


class MetricSpec(FrozenContract):
    technical_key: str
    human_name: str
    definition: str
    formula_or_source: str
    unit: str
    directionality: Direction
    reference: ReferenceSpec | None
    gate_source: str | None
    surfaces: tuple[str, ...]
    #: the ``ResultScope`` value the metric is captioned with by default
    result_scope: str | None = None


class MetricReading(FrozenContract):
    technical_key: str
    human_name: str
    value: float | None
    display: str
    status: UiStatus
    chip: str
    interpretation: str
    caveat: str | None
    evidence_ref: str | None
    reference_value: float | None
    sample: int | None
    #: the status blocks the section (a FAIL against a gate-class reference)
    blocking: bool
    #: the threshold used is a proposed (unratified) default
    proposed: bool


_DIRECTION_OF_OBJECTIVE = {"maximize": "higher_better", "minimize": "lower_better"}
_DIRECTION_WORDS = {
    "higher_better": "higher is better",
    "lower_better": "lower is better",
    "target": "closer to the target is better",
    "descriptive": "descriptive — no direction",
}


def _ref(kind: ReferenceKind, source: str, value: float | None = None) -> ReferenceSpec:
    return ReferenceSpec(kind=kind, source=source, value=value)


def _spec(
    key: str,
    name: str,
    definition: str,
    source: str,
    unit: str,
    direction: Direction | None = None,
    *,
    reference: ReferenceSpec | None = None,
    gate: str | None = None,
    surfaces: tuple[str, ...],
    scope: str | None = None,
) -> MetricSpec:
    registered = OBJECTIVE_DIRECTIONS.get(key)
    if registered is not None:
        direction = _DIRECTION_OF_OBJECTIVE[registered]  # the charter registry wins
    if direction is None:
        raise ValueError(f"metric {key!r} needs a directionality")
    if unit not in METRIC_UNITS:
        raise ValueError(f"metric {key!r} uses an unregistered unit {unit!r}")
    return MetricSpec(
        technical_key=key,
        human_name=name,
        definition=definition,
        formula_or_source=source,
        unit=unit,
        directionality=direction,
        reference=reference,
        gate_source=gate,
        surfaces=surfaces,
        result_scope=scope,
    )


_GATE = "search.charter.ResolvedStrategyGateThresholds"
_PROP_GATE = "search.charter.ResolvedPropGateThresholds"
_ROBUST_GATE = "search.charter.ResolvedRobustnessGateThresholds"
_STRATEGY = "search.strategy_metrics.compute_strategy_metrics (the costed evaluation)"
_VECTOR = "propsim.prop_metrics.PayoutReliabilityVector (the persisted account simulation)"
_FITNESS = "propsim.prop_metrics.EvaluationFitness (the persisted account simulation)"
_STATS = "context_statistics.binary_prediction_report (the persisted candidate report)"
_EXEC = "context_reporting.build_actual_execution_report (source-v2 executed trades)"
_COVERAGE = "context_reporting.build_context_feature_coverage_report"
_BOOTSTRAP = "context_statistics.block_bootstrap_interval (10,000 repetitions, seed 7)"
_ACCESS = "data_access.DataAccessAudit.as_dict (the persisted data_access_audit.json)"
_CAPACITY = "reporting.build_context_capacity_report (observed vs limits)"
_PERFORMANCE = "reporting.build_context_performance_report (observed vs limit)"
_REGIME = "ml.regime_contracts.RegimeCapabilityAssessment (the persisted assessment)"
_MBP1 = "features.mbp1_coverage.Mbp1CoverageReportPayload (the persisted coverage report)"

_ACTUAL = "actual_executed_strategy"
_CANDIDATE = "candidate_research"
_PROP = "prop_historical_closed_trade"


def _gate_ref(field: str, registry: str) -> ReferenceSpec:
    return _ref(ReferenceKind.GATE, f"{registry}.{field}")


_SPECS: tuple[MetricSpec, ...] = (
    # ── strategy metrics (Results explorer / heatmap / overview; wizard gates) ──
    _spec(
        "executed_trades",
        "Executed trades",
        "The number of resolved executed trades in the costed evaluation of the configuration.",
        _STRATEGY,
        "count",
        reference=_gate_ref("min_executed_trades", _GATE),
        gate=f"{_GATE}.min_executed_trades",
        surfaces=("results_strategy", "wizard_gates"),
        scope=_ACTUAL,
    ),
    _spec(
        "independent_days",
        "Independent trading days",
        "The number of distinct trading days on which at least one trade was executed.",
        _STRATEGY,
        "count",
        reference=_gate_ref("min_independent_days", _GATE),
        gate=f"{_GATE}.min_independent_days",
        surfaces=("results_strategy", "wizard_gates"),
        scope=_ACTUAL,
    ),
    _spec(
        "gross_expectancy_r",
        "Gross expectancy",
        "The mean gross R multiple per executed trade before the cost policy is applied.",
        _STRATEGY,
        "R",
        surfaces=("results_strategy",),
        scope=_ACTUAL,
    ),
    _spec(
        "net_expectancy_r",
        "Net expectancy",
        "The mean net R multiple per executed trade after the charter's cost policy.",
        _STRATEGY,
        "R",
        reference=_gate_ref("min_net_expectancy_r", _GATE),
        gate=f"{_GATE}.min_net_expectancy_r",
        surfaces=("results_strategy", "wizard_gates"),
        scope=_ACTUAL,
    ),
    _spec(
        "profit_factor",
        "Profit factor",
        "The sum of positive net R divided by the sum of negative net R over the executed trades.",
        _STRATEGY,
        "ratio",
        reference=_gate_ref("min_profit_factor", _GATE),
        gate=f"{_GATE}.min_profit_factor",
        surfaces=("results_strategy", "wizard_gates"),
        scope=_ACTUAL,
    ),
    _spec(
        "max_drawdown_r",
        "Maximum drawdown",
        "The largest peak-to-trough decline of cumulative net R over the ordered executed trades.",
        _STRATEGY,
        "R",
        reference=_gate_ref("max_drawdown_r", _GATE),
        gate=f"{_GATE}.max_drawdown_r",
        surfaces=("results_strategy", "wizard_gates", "context_execution"),
        scope=_ACTUAL,
    ),
    _spec(
        "time_under_water_days",
        "Time under water",
        "The longest span of trading days spent below the prior cumulative-net-R peak.",
        _STRATEGY,
        "days",
        reference=_gate_ref("max_time_under_water_days", _GATE),
        gate=f"{_GATE}.max_time_under_water_days",
        surfaces=("results_strategy", "wizard_gates"),
        scope=_ACTUAL,
    ),
    _spec(
        "time_block_sign_consistency",
        "Time-block sign consistency",
        "The share of time blocks whose net R has the same sign as the overall net R.",
        _STRATEGY,
        "fraction",
        reference=_gate_ref("min_time_block_sign_consistency", _GATE),
        gate=f"{_GATE}.min_time_block_sign_consistency",
        surfaces=("results_strategy", "wizard_gates"),
        scope=_ACTUAL,
    ),
    _spec(
        "session_stability_score",
        "Session stability",
        "The share of sessions whose net R has the same sign as the overall net R (0–1).",
        _STRATEGY,
        "score",
        reference=_gate_ref("min_session_stability_score", _GATE),
        gate=f"{_GATE}.min_session_stability_score",
        surfaces=("results_strategy", "wizard_gates"),
        scope=_ACTUAL,
    ),
    _spec(
        "top_day_pnl_share",
        "Single-day P&L share",
        "The share of the total absolute daily net R contributed by the single largest day.",
        _STRATEGY,
        "fraction",
        reference=_gate_ref("max_top_day_pnl_share", _GATE),
        gate=f"{_GATE}.max_top_day_pnl_share",
        surfaces=("results_strategy", "wizard_gates"),
        scope=_ACTUAL,
    ),
    _spec(
        "top_setup_pnl_share",
        "Single-setup P&L share",
        "The share of the total absolute per-setup net R contributed by the single largest setup.",
        _STRATEGY,
        "fraction",
        reference=_gate_ref("max_top_setup_pnl_share", _GATE),
        gate=f"{_GATE}.max_top_setup_pnl_share",
        surfaces=("results_strategy", "wizard_gates"),
        scope=_ACTUAL,
    ),
    _spec(
        "net_expectancy_bootstrap_ci95",
        "Net expectancy 95% interval",
        "The trading-day cluster-bootstrap 95% interval on the mean net R per executed trade.",
        f"{_STRATEGY}; trade_stats.cluster_bootstrap_ci95",
        "R",
        "higher_better",
        reference=_gate_ref("require_bootstrap_ci_excludes_zero", _GATE),
        gate=f"{_GATE}.require_bootstrap_ci_excludes_zero",
        surfaces=("results_strategy", "wizard_gates"),
        scope=_ACTUAL,
    ),
    _spec(
        "realized_payoff_ratio",
        "Realized payoff ratio",
        "The average winning trade's gross R divided by the absolute average losing trade's "
        "gross R.",
        f"{_STRATEGY}; PlannedVsRealizedEdge.realized_payoff_ratio",
        "ratio",
        "higher_better",
        surfaces=("results_strategy",),
        scope=_ACTUAL,
    ),
    _spec(
        "trade_frequency",
        "Trade frequency",
        "Executed trades per traded day when the costed evaluation persists it; otherwise "
        "unavailable.",
        "trade_stats.trade_frequency (schema-reserved; not persisted by the current evaluation)",
        "ratio",
        "descriptive",
        surfaces=("results_strategy",),
        scope=_ACTUAL,
    ),
    _spec(
        "setup_occupancy",
        "Setup occupancy",
        "The share of setups that produced an executed trade when persisted; otherwise "
        "unavailable.",
        "trade_stats.setup_occupancy (schema-reserved; not persisted by the current evaluation)",
        "fraction",
        "descriptive",
        surfaces=("results_strategy",),
        scope=_ACTUAL,
    ),
    # ── robustness gate metrics (wizard gates; explorer Robustness preset) ──
    _spec(
        "neighbor_expectancy_degradation_r",
        "Neighbour expectancy degradation",
        "The largest drop of net expectancy between a configuration and its registered neighbours.",
        "search.gates (robustness surfaces; comparison neighbourhood)",
        "R",
        "lower_better",
        reference=_gate_ref("maximum_neighbor_expectancy_degradation_r", _ROBUST_GATE),
        gate=f"{_ROBUST_GATE}.maximum_neighbor_expectancy_degradation_r",
        surfaces=("wizard_gates", "results_strategy"),
        scope=_ACTUAL,
    ),
    _spec(
        "plateau_width",
        "Plateau width",
        "The number of consecutive registered axis steps around the configuration that stay "
        "feasible.",
        "search.gates (robustness surfaces)",
        "count",
        "higher_better",
        reference=_gate_ref("minimum_plateau_width", _ROBUST_GATE),
        gate=f"{_ROBUST_GATE}.minimum_plateau_width",
        surfaces=("wizard_gates", "results_strategy"),
        scope=_ACTUAL,
    ),
    _spec(
        "worst_firm_breach_probability_90d",
        "Worst-firm 90-day breach probability",
        "The highest 90-day breach probability across every simulated firm contract.",
        _VECTOR,
        "probability",
        "lower_better",
        reference=_gate_ref("maximum_worst_firm_breach_probability_90d", _ROBUST_GATE),
        gate=f"{_ROBUST_GATE}.maximum_worst_firm_breach_probability_90d",
        surfaces=("wizard_gates", "results_prop"),
        scope=_PROP,
    ),
    _spec(
        "outer_fold_recurrence",
        "Outer-fold recurrence",
        "Schema-reserved: the share of outer folds in which the configuration recurs "
        "(exploratory lane: null).",
        f"{_ROBUST_GATE}.minimum_outer_fold_recurrence (schema-reserved)",
        "fraction",
        "higher_better",
        reference=_gate_ref("minimum_outer_fold_recurrence", _ROBUST_GATE),
        gate=f"{_ROBUST_GATE}.minimum_outer_fold_recurrence",
        surfaces=("wizard_gates",),
    ),
    # ── prop vector metrics (Results frontier / heatmap / firm matrix / explorer) ──
    _spec(
        "pass_probability",
        "Evaluation pass probability",
        "The share of simulated paths that passed the firm's evaluation phase.",
        _FITNESS,
        "probability",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "median_days_to_pass",
        "Median days to pass",
        "The median number of played days until the evaluation was passed.",
        _FITNESS,
        "days",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "breach_probability",
        "Evaluation breach probability",
        "The share of simulated paths breached during the evaluation phase.",
        _FITNESS,
        "probability",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "expiration_probability",
        "Expiration probability",
        "The share of simulated paths whose evaluation expired before passing.",
        _FITNESS,
        "probability",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "expected_fees_paid",
        "Expected fees paid",
        "The mean fees paid per simulated path (evaluation, activation and reset fees).",
        _FITNESS,
        "USD",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "first_payout_probability_30d",
        "First payout within 30 days",
        "The share of simulated paths with a first payout within 30 played days.",
        _VECTOR,
        "probability",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "first_payout_probability_60d",
        "First payout within 60 days",
        "The share of simulated paths with a first payout within 60 played days.",
        _VECTOR,
        "probability",
        reference=_gate_ref("minimum_first_payout_probability_60d", _PROP_GATE),
        gate=f"{_PROP_GATE}.minimum_first_payout_probability_60d",
        surfaces=("results_prop", "wizard_gates"),
        scope=_PROP,
    ),
    _spec(
        "three_payout_probability",
        "Three payouts before breach",
        "The share of simulated paths that reached three payouts before any breach.",
        _VECTOR,
        "probability",
        reference=_gate_ref("minimum_three_payout_probability", _PROP_GATE),
        gate=f"{_PROP_GATE}.minimum_three_payout_probability",
        surfaces=("results_prop", "wizard_gates"),
        scope=_PROP,
    ),
    _spec(
        "payout_probability_per_rolling_30d",
        "Payout probability per rolling 30 days",
        "The share of rolling 30-day windows across the simulated paths that contain a payout.",
        _VECTOR,
        "probability",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "median_days_between_payouts",
        "Median days between payouts",
        "The median gap in played days between consecutive payouts.",
        _VECTOR,
        "days",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "p90_payout_drought_days",
        "P90 payout drought",
        "The 90th percentile of the longest payout-free span in played days.",
        _VECTOR,
        "days",
        reference=_gate_ref("maximum_p90_payout_drought_days", _PROP_GATE),
        gate=f"{_PROP_GATE}.maximum_p90_payout_drought_days",
        surfaces=("results_prop", "wizard_gates"),
        scope=_PROP,
    ),
    _spec(
        "expected_net_payout_90d",
        "Expected 90-day net payout",
        "The mean net payout to the trader over 90 played days across the simulated paths.",
        _VECTOR,
        "USD",
        reference=_gate_ref("minimum_expected_net_payout_90d", _PROP_GATE),
        gate=f"{_PROP_GATE}.minimum_expected_net_payout_90d",
        surfaces=("results_prop", "wizard_gates"),
        scope=_PROP,
    ),
    _spec(
        "p10_net_payout_90d",
        "P10 90-day net payout",
        "The 10th percentile of the 90-day net payout across the simulated paths (the lower tail).",
        _VECTOR,
        "USD",
        reference=_gate_ref("minimum_p10_net_payout_90d", _PROP_GATE),
        gate=f"{_PROP_GATE}.minimum_p10_net_payout_90d",
        surfaces=("results_prop", "wizard_gates"),
        scope=_PROP,
    ),
    _spec(
        "breach_probability_90d",
        "90-day breach probability",
        "The share of simulated paths breached within 90 played days.",
        _VECTOR,
        "probability",
        reference=_gate_ref("maximum_breach_probability_90d", _PROP_GATE),
        gate=f"{_PROP_GATE}.maximum_breach_probability_90d",
        surfaces=("results_prop", "wizard_gates"),
        scope=_PROP,
    ),
    _spec(
        "median_account_lifetime_days",
        "Median account lifetime",
        "The median number of played days an account survived before breach or the horizon.",
        _VECTOR,
        "days",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "expected_replacement_cost",
        "Expected replacement cost",
        "The mean cost of account replacements per simulated path under the replacement policy.",
        _VECTOR,
        "USD",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "probability_at_least_one_payout_90d",
        "At least one payout within 90 days",
        "The share of simulated paths with at least one payout within 90 played days.",
        _VECTOR,
        "probability",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "probability_all_accounts_breach_90d",
        "All accounts breach within 90 days",
        "The share of portfolio paths in which every account breached within 90 played days.",
        _VECTOR,
        "probability",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "expected_portfolio_net_payout_90d",
        "Expected portfolio 90-day net payout",
        "The mean 90-day net payout summed across the accounts of a portfolio path.",
        _VECTOR,
        "USD",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "p10_portfolio_net_payout_90d",
        "P10 portfolio 90-day net payout",
        "The 10th percentile of the portfolio 90-day net payout (the lower tail).",
        _VECTOR,
        "USD",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "survival_probability_90d",
        "90-day survival",
        "One minus the 90-day breach probability: the share of paths still alive after 90 played "
        "days.",
        "1 − breach_probability_90d (display derivation)",
        "probability",
        "higher_better",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "total_fees",
        "Total fees",
        "The fees charged over the simulated path set as persisted by the account simulation.",
        "propsim.simulation.AccountSimulationPayload.total_fees",
        "USD",
        "lower_better",
        surfaces=("results_prop",),
        scope=_PROP,
    ),
    _spec(
        "payout_p10",
        "Payout P10",
        "The 10th percentile of the net payout samples at the selected horizon (the lower tail).",
        "propsim payout_samples (persisted per horizon)",
        "USD",
        "higher_better",
        surfaces=("results_payout",),
        scope=_PROP,
    ),
    _spec(
        "payout_median",
        "Payout median",
        "The median of the net payout samples at the selected horizon.",
        "propsim payout_samples (persisted per horizon)",
        "USD",
        "higher_better",
        surfaces=("results_payout",),
        scope=_PROP,
    ),
    _spec(
        "payout_mean",
        "Payout mean",
        "The mean of the net payout samples at the selected horizon.",
        "propsim payout_samples (persisted per horizon)",
        "USD",
        "higher_better",
        surfaces=("results_payout",),
        scope=_PROP,
    ),
    _spec(
        "payout_p90",
        "Payout P90",
        "The 90th percentile of the net payout samples at the selected horizon.",
        "propsim payout_samples (persisted per horizon)",
        "USD",
        "higher_better",
        surfaces=("results_payout",),
        scope=_PROP,
    ),
    # ── Context Research: candidate research (counterfactual) ──
    _spec(
        "candidate_count",
        "Candidates",
        "The number of entry candidates in the observation cohort after the warmup exclusion.",
        "context_reporting.build_candidate_research_report",
        "count",
        "descriptive",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "resolved_candidate_count",
        "Resolved candidates",
        "The candidates whose counterfactual label resolved before the cutoff.",
        "context_reporting.build_candidate_research_report",
        "count",
        "descriptive",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "censored_candidate_count",
        "Censored candidates",
        "The candidates whose counterfactual outcome was censored (development cutoff or horizon).",
        "context_reporting.build_candidate_research_report",
        "count",
        "descriptive",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "oos_prediction_count",
        "Out-of-sample predictions",
        "The number of out-of-fold predictions the walk-forward protocol produced.",
        f"{_STATS}; key 'count'",
        "count",
        "descriptive",
        surfaces=("context_candidate", "ladder"),
        scope=_CANDIDATE,
    ),
    _spec(
        "prevalence",
        "Out-of-sample prevalence",
        "The share of positive targets among the out-of-sample predictions (the base rate).",
        _STATS,
        "fraction",
        "descriptive",
        surfaces=("context_candidate", "ladder"),
        scope=_CANDIDATE,
    ),
    _spec(
        "mean_probability",
        "Mean predicted probability",
        "The mean predicted probability over the out-of-sample predictions, read against the "
        "prevalence.",
        _STATS,
        "probability",
        "target",
        reference=_ref(ReferenceKind.TARGET, f"{_STATS}; key 'prevalence'"),
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "brier_score",
        "Brier score",
        "The mean squared error between the predicted probability and the 0/1 target (lower is "
        "better).",
        f"{_STATS}; sklearn brier_score_loss",
        "score",
        "lower_better",
        reference=_ref(ReferenceKind.BOUNDARY, f"{_STATS}; key 'reference_brier_score'"),
        surfaces=("context_candidate", "ladder"),
        scope=_CANDIDATE,
    ),
    _spec(
        "reference_brier_score",
        "Reference Brier score",
        "The Brier score of the training-prevalence reference predictor on the same predictions.",
        _STATS,
        "score",
        "descriptive",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "brier_skill_score",
        "Brier skill score",
        "One minus the Brier score over the reference Brier score; positive means better than "
        "prevalence.",
        _STATS,
        "score",
        "higher_better",
        reference=_ref(ReferenceKind.BOUNDARY, "0 skill boundary (no better than prevalence)", 0.0),
        surfaces=("context_candidate", "ladder"),
        scope=_CANDIDATE,
    ),
    _spec(
        "log_loss",
        "Log loss",
        "The mean negative log-likelihood of the 0/1 targets under the predicted probabilities.",
        f"{_STATS}; sklearn log_loss",
        "score",
        "lower_better",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "auc",
        "AUC",
        "The area under the ROC curve of the out-of-sample predictions; 0.5 is the chance line.",
        f"{_STATS}; sklearn roc_auc_score (undefined for a single-class OOS set)",
        "score",
        "higher_better",
        reference=_ref(ReferenceKind.CHANCE_LINE, "0.5 chance line (direction only)", 0.5),
        surfaces=("context_candidate", "ladder"),
        scope=_CANDIDATE,
    ),
    _spec(
        "calibration_slope",
        "Calibration slope",
        "The logistic recalibration slope on the logit of the predicted probability (target 1).",
        f"{_STATS}; key 'calibration.slope'",
        "ratio",
        "target",
        reference=_ref(ReferenceKind.TARGET, "target 1.0 (perfect calibration slope)", 1.0),
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "calibration_intercept",
        "Calibration intercept",
        "The logistic recalibration intercept on the logit of the predicted probability (target "
        "0).",
        f"{_STATS}; key 'calibration.intercept'",
        "ratio",
        "target",
        reference=_ref(ReferenceKind.TARGET, "target 0.0 (perfect calibration intercept)", 0.0),
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "threshold_coverage_fraction",
        "Coverage at threshold",
        "The share of out-of-sample predictions at or above the probability threshold.",
        f"{_STATS}; thresholds[].coverage_fraction",
        "fraction",
        "descriptive",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "threshold_coverage_count",
        "Predictions at threshold",
        "The number of out-of-sample predictions at or above the probability threshold.",
        f"{_STATS}; thresholds[].coverage_count",
        "count",
        "descriptive",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "threshold_net_r_sum",
        "Net R at threshold",
        "The summed counterfactual net R of the candidates at or above the probability threshold.",
        f"{_STATS}; thresholds[].r.net_r_sum",
        "R",
        "higher_better",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "threshold_net_r_mean",
        "Mean net R at threshold",
        "The mean counterfactual net R of the candidates at or above the probability threshold.",
        f"{_STATS}; thresholds[].r.net_r_mean",
        "R",
        "higher_better",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "gross_r_sum",
        "Gross R (resolved candidates)",
        "The summed counterfactual gross R of the resolved candidates.",
        "context_reporting.build_candidate_research_report; gross_net_r",
        "R",
        "higher_better",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "net_r_sum",
        "Net R (resolved candidates)",
        "The summed counterfactual net R of the resolved candidates after the per-trade cost.",
        "context_reporting.build_candidate_research_report; gross_net_r",
        "R",
        "higher_better",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "net_r_mean",
        "Mean net R (resolved candidates)",
        "The mean counterfactual net R per resolved candidate after the per-trade cost.",
        "context_reporting.build_candidate_research_report; gross_net_r",
        "R",
        "higher_better",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "setup_cluster_net_r_mean",
        "Mean net R — setup-cluster bootstrap interval",
        "The 95% block-bootstrap interval on the mean counterfactual net R, resampling whole "
        "setups.",
        _BOOTSTRAP,
        "R",
        "higher_better",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "trading_day_block_net_r_mean",
        "Mean net R — trading-day block bootstrap interval",
        "The 95% block-bootstrap interval on the mean counterfactual net R, resampling whole days.",
        _BOOTSTRAP,
        "R",
        "higher_better",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "fold_count",
        "Walk-forward folds",
        "The number of folds the fixed 40/5/5/2 walk-forward schedule produced.",
        "context_folds.build_context_folds",
        "count",
        "descriptive",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "valid_fold_count",
        "Valid folds",
        "The folds that met the class-coverage and minimum-training-candidate rules.",
        "context_folds.build_context_folds; IfvgContextFoldDefinition.valid",
        "count",
        "higher_better",
        reference=_ref(ReferenceKind.ADEQUACY_MINIMUM, "at least one valid fold", 1.0),
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "fold_train_candidate_count",
        "Training candidates in a fold",
        "The number of training candidates a fold holds, read against the protocol's minimum.",
        "IfvgContextExperimentConfig.minimum_train_candidates (30)",
        "count",
        "higher_better",
        reference=_ref(
            ReferenceKind.ADEQUACY_MINIMUM,
            "IfvgContextExperimentConfig.minimum_train_candidates",
            30.0,
        ),
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "bootstrap_cluster_count",
        "Bootstrap clusters",
        "The number of resampling clusters (setups or days) the block bootstrap could use.",
        f"{_BOOTSTRAP}; fewer than two clusters is unavailable",
        "count",
        "higher_better",
        reference=_ref(
            ReferenceKind.ADEQUACY_MINIMUM,
            "context_statistics.block_bootstrap_interval (two usable clusters)",
            2.0,
        ),
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "permutation_importance_mean",
        "Permutation importance",
        "The mean out-of-fold permutation importance of a feature (descriptive only; never a "
        "selection).",
        "context_statistics.feature_importance_report (20 repeats)",
        "score",
        "descriptive",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    _spec(
        "importance_fold_coverage_fraction",
        "Importance fold coverage",
        "The share of valid folds in which the feature carried a permutation importance.",
        "context_statistics.feature_importance_report; fold_coverage_fraction",
        "fraction",
        "descriptive",
        surfaces=("context_candidate",),
        scope=_CANDIDATE,
    ),
    # ── Context Research: actual execution (source v2) ──
    _spec(
        "eligible_decision_count",
        "Eligible decisions",
        "The number of source-v2 eligible decisions in the verified pair.",
        _EXEC,
        "count",
        "descriptive",
        surfaces=("context_execution",),
        scope=_ACTUAL,
    ),
    _spec(
        "executed_trade_count",
        "Executed trades (source v2)",
        "The number of source-v2 executed trades in the verified pair.",
        _EXEC,
        "count",
        "descriptive",
        surfaces=("context_execution",),
        scope=_ACTUAL,
    ),
    _spec(
        "resolved_trade_count",
        "Resolved trades",
        "The executed trades whose realized R is persisted.",
        _EXEC,
        "count",
        "descriptive",
        surfaces=("context_execution",),
        scope=_ACTUAL,
    ),
    _spec(
        "winning_trade_count",
        "Winning trades",
        "The resolved trades with a positive realized R.",
        _EXEC,
        "count",
        "descriptive",
        surfaces=("context_execution",),
        scope=_ACTUAL,
    ),
    _spec(
        "win_rate",
        "Win rate",
        "The share of resolved executed trades with a positive realized R.",
        _EXEC,
        "fraction",
        "descriptive",
        surfaces=("context_execution",),
        scope=_ACTUAL,
    ),
    _spec(
        "total_realized_r",
        "Total realized R",
        "The summed realized R of the resolved executed trades.",
        _EXEC,
        "R",
        "higher_better",
        surfaces=("context_execution",),
        scope=_ACTUAL,
    ),
    _spec(
        "mean_realized_r",
        "Mean realized R",
        "The mean realized R per resolved executed trade.",
        _EXEC,
        "R",
        "higher_better",
        surfaces=("context_execution",),
        scope=_ACTUAL,
    ),
    _spec(
        "total_realized_dollars",
        "Total realized dollars (one NQ contract)",
        "The realized ticks of the resolved trades converted at $5 per tick for one NQ contract "
        "(display only).",
        f"{_EXEC}; dollar_conversion",
        "USD",
        "higher_better",
        surfaces=("context_execution",),
        scope=_ACTUAL,
    ),
    _spec(
        "max_drawdown_dollars",
        "Maximum drawdown (one NQ contract)",
        "The largest peak-to-trough decline of the cumulative dollar equity (display conversion).",
        f"{_EXEC}; dollar_conversion",
        "USD",
        "lower_better",
        surfaces=("context_execution",),
        scope=_ACTUAL,
    ),
    _spec(
        "trading_day_block_realized_r_mean",
        "Mean realized R — trading-day block bootstrap interval",
        "The 95% block-bootstrap interval on the mean realized R of the executed trades, "
        "resampling days.",
        _BOOTSTRAP,
        "R",
        "higher_better",
        surfaces=("context_execution",),
        scope=_ACTUAL,
    ),
    # ── Context Research: feature coverage ──
    _spec(
        "feature_count",
        "Features in tier",
        "The number of registered features the selected tier reports.",
        _COVERAGE,
        "count",
        "descriptive",
        surfaces=("context_coverage",),
    ),
    _spec(
        "feature_coverage_fraction",
        "Feature coverage",
        "The share of cohort candidates with a non-null value for the feature.",
        f"{_COVERAGE}; low_coverage flags a share below 0.10",
        "fraction",
        "higher_better",
        reference=_ref(
            ReferenceKind.ADEQUACY_MINIMUM,
            "context_reporting.build_context_feature_coverage_report low_coverage (< 0.10)",
            0.10,
        ),
        surfaces=("context_coverage",),
    ),
    _spec(
        "feature_missing_fraction",
        "Feature missingness",
        "The share of cohort candidates with a null value for the feature.",
        "context_report_adapters.adapt_feature_coverage_report",
        "fraction",
        "lower_better",
        surfaces=("context_coverage",),
    ),
    _spec(
        "feature_unique_values",
        "Distinct feature values",
        "The number of distinct non-null values the feature takes in the cohort (1 means "
        "constant).",
        _COVERAGE,
        "count",
        "descriptive",
        surfaces=("context_coverage",),
    ),
    _spec(
        "pool_width_ticks",
        "Pool width",
        "The distribution of equal-level pool widths (upper minus lower bound) in ticks.",
        f"{_COVERAGE}; pool_width_ticks",
        "ticks",
        "descriptive",
        surfaces=("context_coverage",),
    ),
    _spec(
        "pool_member_separation_ticks",
        "Pool member separation",
        "The distribution of absolute separations between pool members in ticks.",
        f"{_COVERAGE}; pool_member_separation_ticks",
        "ticks",
        "descriptive",
        surfaces=("context_coverage",),
    ),
    # ── reconciliation / Data & Audit report gates (evaluated ``passed`` flags) ──
    _spec(
        "validity_report",
        "Context validity",
        "Every context record carries an as-of timestamp no earlier than its sources and a typed "
        "missing reason.",
        "reporting.build_context_validity_report; key 'passed'",
        "flag",
        "descriptive",
        gate="reporting.build_context_validity_report",
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "reconciliation_report",
        "Context reconciliation",
        "The v3 foreign keys resolve exactly into the accepted v2 core and the transient replay "
        "matched it.",
        "reporting.build_context_reconciliation_report; key 'passed'",
        "flag",
        "descriptive",
        gate="reporting.build_context_reconciliation_report",
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "identity_report",
        "Context identity",
        "Every table carries exactly the expected feature-set, formula, schema and config "
        "identities.",
        "reporting.build_context_identity_report; key 'passed'",
        "flag",
        "descriptive",
        gate="reporting.build_context_identity_report",
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "capacity_report",
        "Capacity",
        "Every observed capacity figure stays within its registered limit.",
        "reporting.build_context_capacity_report; key 'passed'",
        "flag",
        "descriptive",
        gate="reporting.build_context_capacity_report",
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "performance_report",
        "Performance",
        "Every measured replay-performance figure stays within its registered limit.",
        "reporting.build_context_performance_report; key 'passed'",
        "flag",
        "descriptive",
        gate="reporting.build_context_performance_report",
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "invariant_audit",
        "v2 invariant audit",
        "No source-v2 invariant violation (statuses, open trades, forbidden source dates, "
        "mutations).",
        "reporting.build_invariant_audit; key 'passed'",
        "flag",
        "descriptive",
        gate="reporting.build_invariant_audit",
        surfaces=("data_audit",),
    ),
    _spec(
        "validity_records",
        "Validity records",
        "The number of context validity-provenance records.",
        "reporting.build_context_validity_report; key 'records'",
        "count",
        "descriptive",
        surfaces=("data_audit",),
    ),
    _spec(
        "validity_valid_records",
        "Valid records",
        "The validity-provenance records marked valid.",
        "reporting.build_context_validity_report; key 'valid_records'",
        "count",
        "descriptive",
        surfaces=("data_audit",),
    ),
    _spec(
        "validity_warmup_complete_records",
        "Warmup-complete records",
        "The validity-provenance records whose warmup was complete.",
        "reporting.build_context_validity_report; key 'warmup_complete_records'",
        "count",
        "descriptive",
        surfaces=("data_audit",),
    ),
    # access counters
    _spec(
        "path_constructions",
        "Source paths constructed",
        "The number of allowlisted source paths the exploration policy constructed.",
        _ACCESS,
        "count",
        "descriptive",
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "metadata_accesses",
        "Source metadata accesses",
        "The number of allowlisted source metadata reads.",
        _ACCESS,
        "count",
        "descriptive",
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "file_opens",
        "Source files opened",
        "The number of allowlisted source files opened.",
        _ACCESS,
        "count",
        "descriptive",
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "rows_read",
        "Source rows read",
        "The number of allowlisted source rows read.",
        _ACCESS,
        "count",
        "descriptive",
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "denied_attempt_count",
        "Denied access attempts",
        "The number of source-date authorizations the allowlist refused (a measured count).",
        f"{_ACCESS}; sum of denied_dates",
        "count",
        "lower_better",
        reference=_ref(
            ReferenceKind.MEASURED_ZERO, "data_access.ExplorationDataPolicy.authorize_date", 0.0
        ),
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "forbidden_source_path_or_io_dates",
        "Forbidden source dates touched",
        "The number of touched source dates outside the allowlist (a measured violation count).",
        "reporting.build_invariant_audit; violations",
        "count",
        "lower_better",
        reference=_ref(ReferenceKind.MEASURED_ZERO, "reporting.build_invariant_audit", 0.0),
        surfaces=("data_audit",),
    ),
    _spec(
        "forbidden_source_rows",
        "Forbidden source rows read",
        "The number of rows read from source dates outside the allowlist (a measured violation "
        "count).",
        "reporting.build_invariant_audit; violations",
        "count",
        "lower_better",
        reference=_ref(ReferenceKind.MEASURED_ZERO, "reporting.build_invariant_audit", 0.0),
        surfaces=("data_audit",),
    ),
    _spec(
        "protected_path_constructions",
        "Protected paths constructed",
        "A policy-enforced zero: no protected-range path can be constructed before authorization.",
        f"{_ACCESS}; written as 0 before any path exists",
        "count",
        "descriptive",
        reference=_ref(ReferenceKind.POLICY_ZERO, "data_access.DataAccessAudit.as_dict", 0.0),
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "protected_metadata_accesses",
        "Protected metadata accesses",
        "A policy-enforced zero: no protected-range metadata read is reachable.",
        f"{_ACCESS}; written as 0 before any path exists",
        "count",
        "descriptive",
        reference=_ref(ReferenceKind.POLICY_ZERO, "data_access.DataAccessAudit.as_dict", 0.0),
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "protected_file_opens",
        "Protected files opened",
        "A policy-enforced zero: no protected-range file open is reachable.",
        f"{_ACCESS}; written as 0 before any path exists",
        "count",
        "descriptive",
        reference=_ref(ReferenceKind.POLICY_ZERO, "data_access.DataAccessAudit.as_dict", 0.0),
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "protected_rows_read",
        "Protected rows read",
        "A policy-enforced zero: no protected-range row read is reachable.",
        f"{_ACCESS}; written as 0 before any path exists",
        "count",
        "descriptive",
        reference=_ref(ReferenceKind.POLICY_ZERO, "data_access.DataAccessAudit.as_dict", 0.0),
        surfaces=("context_reconciliation", "data_audit"),
    ),
    # capacity (limits come from the persisted report)
    _spec(
        "terminal_state_bytes",
        "Terminal state size",
        "The serialized size of the terminal context state, read against the registered limit.",
        _CAPACITY,
        "bytes",
        "lower_better",
        reference=_ref(ReferenceKind.LIMIT, f"{_CAPACITY}; limits.terminal_state_bytes"),
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "terminal_seed_bytes",
        "Terminal seed size",
        "The serialized size of the terminal seed, read against the registered limit.",
        _CAPACITY,
        "bytes",
        "lower_better",
        reference=_ref(ReferenceKind.LIMIT, f"{_CAPACITY}; limits.terminal_seed_bytes"),
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "max_transition_bytes",
        "Largest transition",
        "The largest single state transition in bytes, read against the registered limit.",
        _CAPACITY,
        "bytes",
        "lower_better",
        reference=_ref(ReferenceKind.LIMIT, f"{_CAPACITY}; limits.max_transition_bytes"),
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "max_members_per_pool",
        "Largest pool",
        "The largest equal-level pool membership, read against the registered limit.",
        _CAPACITY,
        "count",
        "lower_better",
        reference=_ref(ReferenceKind.LIMIT, f"{_CAPACITY}; limits.max_members_per_pool"),
        surfaces=("context_reconciliation", "data_audit"),
    ),
    # performance (fixed limits in reporting.py)
    _spec(
        "replay_slowdown_fraction",
        "Replay slowdown",
        "The fractional slowdown of the context-enabled replay against the disabled replay.",
        _PERFORMANCE,
        "fraction",
        "lower_better",
        reference=_ref(ReferenceKind.LIMIT, f"{_PERFORMANCE}; limit 0.20", 0.20),
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "repeated_run_p95_slowdown_fraction",
        "Repeated-run P95 slowdown",
        "The 95th-percentile paired slowdown over repeated runs.",
        _PERFORMANCE,
        "fraction",
        "lower_better",
        reference=_ref(ReferenceKind.LIMIT, f"{_PERFORMANCE}; limit 0.25", 0.25),
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "completed_1m_step_p99_ms",
        "Completed 1m step P99",
        "The 99th-percentile wall time of a completed one-minute replay step.",
        _PERFORMANCE,
        "ms",
        "lower_better",
        reference=_ref(ReferenceKind.LIMIT, f"{_PERFORMANCE}; limit 1.6 ms", 1.6),
        surfaces=("context_reconciliation", "data_audit"),
    ),
    _spec(
        "multi_timeframe_callback_p99_ms",
        "Multi-timeframe callback P99",
        "The 99th-percentile wall time of a multi-timeframe callback.",
        _PERFORMANCE,
        "ms",
        "lower_better",
        reference=_ref(ReferenceKind.LIMIT, f"{_PERFORMANCE}; limit 8.0 ms", 8.0),
        surfaces=("context_reconciliation", "data_audit"),
    ),
    # ── supervised ladder (S10 diagnostics) ──
    _spec(
        "oos_row_count",
        "Out-of-sample rows",
        "The number of out-of-fold rows a ladder rung predicted; zero rows leave parity not "
        "evaluable.",
        "ml supervised_ladder.json; rungs[].prediction_report.count",
        "count",
        "higher_better",
        reference=_ref(ReferenceKind.ADEQUACY_MINIMUM, "at least one out-of-fold row", 1.0),
        surfaces=("ladder",),
        scope=_CANDIDATE,
    ),
    # ── regime model card (persisted assessment) ──
    _spec(
        "oos_assignment_coverage",
        "Out-of-sample assignment coverage",
        "The share of out-of-sample rows that received a valid regime assignment.",
        f"{_REGIME}; coverage.oos_assignment_coverage",
        "fraction",
        "higher_better",
        surfaces=("regime",),
    ),
    _spec(
        "minimum_training_observations_observed",
        "Training observations per fold (minimum observed)",
        "The smallest training-row count over the folds, read against the stamped adequacy gate.",
        f"{_REGIME}; coverage.minimum_training_observations_observed vs _gate",
        "count",
        "higher_better",
        reference=_ref(
            ReferenceKind.ADEQUACY_MINIMUM,
            "RegimeCoverageReport.minimum_training_observations_gate (proposed_protocol_default)",
        ),
        surfaces=("regime",),
    ),
    _spec(
        "regime_occupancy",
        "Regime occupancy",
        "The share of assigned rows in one nominal regime, read against the stamped minimum "
        "occupancy.",
        f"{_REGIME}; occupancy",
        "fraction",
        "higher_better",
        reference=_ref(
            ReferenceKind.GATE,
            "REGIME_PROPOSED_DEFAULTS.minimum_cluster_occupancy_fraction "
            "(proposed_protocol_default)",
        ),
        surfaces=("regime",),
    ),
    _spec(
        "bootstrap_aligned_ami_mean",
        "Bootstrap aligned AMI (mean)",
        "The mean adjusted mutual information between bootstrap refits and the fold fit after "
        "alignment.",
        f"{_REGIME}; stability.bootstrap_aligned_ami_mean",
        "score",
        "higher_better",
        reference=_ref(
            ReferenceKind.GATE,
            "RegimeStabilityReport.minimum_bootstrap_aligned_ami_mean_applied "
            "(proposed_protocol_default)",
        ),
        surfaces=("regime",),
    ),
    _spec(
        "protocol_min_bootstrap_aligned_ami_mean",
        "Protocol-wide minimum fold AMI mean",
        "The smallest per-fold bootstrap aligned AMI mean — the value the stability gate applies "
        "to.",
        f"{_REGIME}; stability.protocol_min_bootstrap_aligned_ami_mean",
        "score",
        "higher_better",
        reference=_ref(
            ReferenceKind.GATE,
            "RegimeStabilityReport.minimum_bootstrap_aligned_ami_mean_applied "
            "(proposed_protocol_default)",
        ),
        surfaces=("regime",),
    ),
    _spec(
        "temporal_persistence",
        "Temporal persistence",
        "The share of consecutive out-of-sample observations that stay in the same nominal regime.",
        f"{_REGIME}; stability.temporal_persistence / candidate_event_persistence",
        "fraction",
        "descriptive",
        surfaces=("regime",),
    ),
    _spec(
        "fold_to_fold_recurrence",
        "Fold-to-fold recurrence",
        "The aligned centroid distance between consecutive fold fits (smaller means more "
        "recurrent).",
        f"{_REGIME}; stability.fold_to_fold_recurrence",
        "score",
        "descriptive",
        surfaces=("regime",),
    ),
    _spec(
        "separation_min_centroid_distance",
        "Minimum centroid separation",
        "The smallest distance between two regime centroids in the scaled input space.",
        f"{_REGIME}; stability.separation_min_centroid_distance",
        "score",
        "descriptive",
        surfaces=("regime",),
    ),
    _spec(
        "silhouette_descriptive",
        "Silhouette (descriptive)",
        "The silhouette score of the reference fold — descriptive only; it never promotes.",
        f"{_REGIME}; stability.silhouette_descriptive",
        "score",
        "descriptive",
        surfaces=("regime",),
    ),
    # ── MBP-1 (persisted coverage report / controlled study) ──
    _spec(
        "mbp1_day_coverage_fraction",
        "MBP-1 day coverage",
        "The evidence-based coverage fraction of one trading day's MBP-1 source partition.",
        _MBP1,
        "fraction",
        "higher_better",
        reference=_ref(
            ReferenceKind.GATE,
            "MBP1_PROPOSED_DEFAULTS.min_day_coverage_fraction (proposed_protocol_default 0.95)",
            0.95,
        ),
        surfaces=("mbp1",),
    ),
    _spec(
        "mbp1_window_valid_count",
        "Valid MBP-1 windows",
        "The number of candidate stage windows with a valid MBP-1 feature value.",
        f"{_MBP1}; window_rows[].valid_count",
        "count",
        "descriptive",
        surfaces=("mbp1",),
    ),
    _spec(
        "mbp1_window_invalid_count",
        "Typed-null MBP-1 windows",
        "The number of candidate stage windows whose MBP-1 feature is a typed null.",
        f"{_MBP1}; window_rows[].invalid_count",
        "count",
        "descriptive",
        surfaces=("mbp1",),
    ),
    _spec(
        "paired_brier_delta",
        "Paired Brier delta (challenger − baseline)",
        "The trading-day block-bootstrap interval of the per-row Brier-loss difference; negative "
        "favours MBP-1.",
        "ml.controlled_feature_study.ControlledFeatureStudyPayload.paired_brier_delta",
        "score",
        "lower_better",
        surfaces=("mbp1",),
    ),
    # ── verifier execution metrics (exact case evidence) ──
    _spec(
        "realized_r",
        "Realized R",
        "The realized R multiple of one executed trade.",
        "source-v2 executed_trade.realized_r",
        "R",
        "higher_better",
        surfaces=("verifier", "context_execution"),
        scope=_ACTUAL,
    ),
    _spec(
        "realized_ticks",
        "Realized ticks",
        "The realized profit or loss of one executed trade in ticks (0.25 points).",
        "source-v2 executed_trade.realized_ticks",
        "ticks",
        "higher_better",
        surfaces=("verifier",),
        scope=_ACTUAL,
    ),
    _spec(
        "mfe_ticks",
        "Maximum favourable excursion",
        "The furthest the market moved in the trade's favour before resolution, in ticks.",
        "source-v2 executed_trade.mfe_ticks",
        "ticks",
        "descriptive",
        surfaces=("verifier",),
        scope=_ACTUAL,
    ),
    _spec(
        "mae_ticks",
        "Maximum adverse excursion",
        "The furthest the market moved against the trade before resolution, in ticks.",
        "source-v2 executed_trade.mae_ticks",
        "ticks",
        "descriptive",
        surfaces=("verifier",),
        scope=_ACTUAL,
    ),
    _spec(
        "bars_in_trade",
        "Bars in trade",
        "The number of one-minute bars between entry and resolution.",
        "source-v2 executed_trade.bars_in_trade",
        "bars",
        "descriptive",
        surfaces=("verifier",),
        scope=_ACTUAL,
    ),
    _spec(
        "penetration_ticks",
        "Zone penetration",
        "How far price penetrated the tapped zone, in ticks.",
        "source-v2 setup lifecycle; penetration_ticks",
        "ticks",
        "descriptive",
        surfaces=("verifier",),
        scope=_ACTUAL,
    ),
    _spec(
        "risk_ticks",
        "Risk (entry to stop)",
        "The distance between the entry and the stop, in ticks — the 1R unit of the trade.",
        "source-v2 entry_candidate; entry_ticks − stop_ticks",
        "ticks",
        "descriptive",
        surfaces=("verifier",),
        scope=_ACTUAL,
    ),
)

METRIC_SPECS: Mapping[str, MetricSpec] = MappingProxyType(
    {spec.technical_key: spec for spec in _SPECS}
)
if len(METRIC_SPECS) != len(_SPECS):  # pragma: no cover — registry integrity
    raise RuntimeError("duplicate metric technical keys")


#: The wizard / charter gate fields → the metric they threshold.
GATE_FIELD_METRICS: Mapping[str, str] = MappingProxyType(
    {
        "min_executed_trades": "executed_trades",
        "min_independent_days": "independent_days",
        "min_net_expectancy_r": "net_expectancy_r",
        "min_profit_factor": "profit_factor",
        "max_drawdown_r": "max_drawdown_r",
        "max_time_under_water_days": "time_under_water_days",
        "min_session_stability_score": "session_stability_score",
        "min_time_block_sign_consistency": "time_block_sign_consistency",
        "max_top_day_pnl_share": "top_day_pnl_share",
        "max_top_setup_pnl_share": "top_setup_pnl_share",
        "require_bootstrap_ci_excludes_zero": "net_expectancy_bootstrap_ci95",
        "minimum_first_payout_probability_60d": "first_payout_probability_60d",
        "maximum_breach_probability_90d": "breach_probability_90d",
        "minimum_expected_net_payout_90d": "expected_net_payout_90d",
        "minimum_p10_net_payout_90d": "p10_net_payout_90d",
        "maximum_p90_payout_drought_days": "p90_payout_drought_days",
        "minimum_three_payout_probability": "three_payout_probability",
        "maximum_neighbor_expectancy_degradation_r": "neighbor_expectancy_degradation_r",
        "minimum_plateau_width": "plateau_width",
        "maximum_worst_firm_breach_probability_90d": "worst_firm_breach_probability_90d",
        "minimum_time_block_sign_consistency": "time_block_sign_consistency",
        "minimum_outer_fold_recurrence": "outer_fold_recurrence",
    }
)


def describe(technical_key: str) -> MetricSpec:
    try:
        return METRIC_SPECS[str(technical_key)]
    except KeyError:
        raise ValueError(f"unregistered metric {technical_key!r}") from None


def metric_keys() -> tuple[str, ...]:
    return tuple(sorted(METRIC_SPECS))


def metric_keys_for_surface(surface: str) -> tuple[str, ...]:
    return tuple(sorted(key for key, spec in METRIC_SPECS.items() if surface in spec.surfaces))


def gate_metric_key(gate_field: str) -> str:
    try:
        return GATE_FIELD_METRICS[str(gate_field)]
    except KeyError:
        raise ValueError(f"unregistered gate field {gate_field!r}") from None


# ── formatting ──────────────────────────────────────────────────────────────


def _numeric(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number == number else None  # NaN → None
    return None


def format_value(spec: MetricSpec, value: object) -> str:
    """The display text of a value in the spec's unit ('—' when missing)."""

    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    number = _numeric(value)
    if number is None:
        return str(value)
    unit = spec.unit
    if unit in ("fraction", "probability"):
        return f"{number * 100.0:.1f}%"
    if unit == "R":
        return f"{number:,.2f} R"
    if unit == "USD":
        return f"${number:,.0f}"
    if unit == "count":
        return f"{int(number):,}" if number.is_integer() else f"{number:,.2f}"
    if unit == "days":
        return f"{int(number):,} days" if number.is_integer() else f"{number:,.1f} days"
    if unit == "bytes":
        return f"{int(number):,} B"
    if unit == "ms":
        return f"{number:,.2f} ms"
    if unit == "ticks":
        return f"{int(number):,} t" if number.is_integer() else f"{number:,.1f} t"
    if unit == "bars":
        return f"{int(number):,} bars"
    if unit == "score":
        return f"{number:.4f}"
    if unit == "flag":
        return "yes" if number else "no"
    return f"{number:,.2f}"  # ratio


def _fmt(spec: MetricSpec, value: float | None) -> str:
    return format_value(spec, value)


# ── evaluation ──────────────────────────────────────────────────────────────


def _resolve(spec_or_key: MetricSpec | str) -> MetricSpec:
    return spec_or_key if isinstance(spec_or_key, MetricSpec) else describe(spec_or_key)


def _reading(
    spec: MetricSpec,
    *,
    value: float | None,
    display: str,
    status: UiStatus,
    interpretation: str,
    caveat: str | None = None,
    evidence_ref: str | None = None,
    reference_value: float | None = None,
    sample: int | None = None,
    blocking: bool = False,
    proposed: bool = False,
) -> MetricReading:
    if sample is not None:
        interpretation = f"{interpretation} — over {sample:,} observation(s)"
    return MetricReading(
        technical_key=spec.technical_key,
        human_name=spec.human_name,
        value=value,
        display=display,
        status=status,
        chip=status_chip(status),
        interpretation=interpretation,
        caveat=caveat,
        evidence_ref=evidence_ref,
        reference_value=reference_value,
        sample=sample,
        blocking=blocking,
        proposed=proposed,
    )


_PROPOSED_CAVEAT = (
    "the threshold is a proposed_protocol_default — owner ratification is required "
    "before it carries research weight"
)


def evaluate_metric(
    spec_or_key: MetricSpec | str,
    value: object,
    *,
    gate: float | None = None,
    reference: float | None = None,
    sample: int | None = None,
    evaluated: bool = True,
    proposed: bool = False,
    evidence_ref: str | None = None,
) -> MetricReading:
    """The typed reading of one value (see the module docstring for the rules)."""

    spec = _resolve(spec_or_key)
    display = format_value(spec, value)
    number = _numeric(value)
    direction_words = _DIRECTION_WORDS[spec.directionality]
    if not evaluated:
        return _reading(
            spec,
            value=number,
            display=display,
            status=UiStatus.UNAVAILABLE,
            interpretation=(
                f"{spec.human_name} was not evaluated (default-only or unevaluated evidence) — "
                "it is not shown as passing"
            ),
            evidence_ref=evidence_ref,
            sample=sample,
        )
    if number is None:
        return _reading(
            spec,
            value=None,
            display=display,
            status=UiStatus.UNAVAILABLE,
            interpretation=f"{spec.human_name}: no value was persisted for this evidence",
            evidence_ref=evidence_ref,
            sample=sample,
        )
    kind = spec.reference.kind if spec.reference is not None else None
    directional = spec.directionality in ("higher_better", "lower_better")
    if kind is None and gate is not None and directional:
        kind = ReferenceKind.GATE
    if spec.directionality == "descriptive" or kind is None:
        note = (
            "a descriptive value; no gate applies"
            if spec.directionality == "descriptive"
            else f"no gate selected; {direction_words}"
        )
        return _reading(
            spec,
            value=number,
            display=display,
            status=UiStatus.INFORMATIONAL,
            interpretation=f"{spec.human_name} is {display} — {note}",
            evidence_ref=evidence_ref,
            sample=sample,
        )
    fixed = spec.reference.value if spec.reference is not None else None
    source = spec.reference.source if spec.reference is not None else spec.gate_source or ""
    caveat = _PROPOSED_CAVEAT if proposed else None
    if kind is ReferenceKind.GATE:
        bound = gate if gate is not None else reference if reference is not None else fixed
        if bound is None:
            return _reading(
                spec,
                value=number,
                display=display,
                status=UiStatus.INFORMATIONAL,
                interpretation=(
                    f"{spec.human_name} is {display} — no gate selected; {direction_words}"
                ),
                evidence_ref=evidence_ref,
                sample=sample,
            )
        met = number >= bound if spec.directionality == "higher_better" else number <= bound
        word = "at least" if spec.directionality == "higher_better" else "at most"
        return _reading(
            spec,
            value=number,
            display=display,
            status=UiStatus.PASS if met else UiStatus.FAIL,
            interpretation=(
                f"{spec.human_name} is {display}; the selected gate requires {word} "
                f"{_fmt(spec, bound)} ({source}) — {'met' if met else 'not met'}"
            ),
            caveat=caveat,
            evidence_ref=evidence_ref,
            reference_value=float(bound),
            sample=sample,
            blocking=not met,
            proposed=proposed,
        )
    if kind is ReferenceKind.BOUNDARY:
        bound = reference if reference is not None else fixed
        if bound is None:
            return _reading(
                spec,
                value=number,
                display=display,
                status=UiStatus.UNAVAILABLE,
                interpretation=(
                    f"{spec.human_name} is {display} but its reference ({source}) was not "
                    "persisted — no comparison is shown as passing"
                ),
                evidence_ref=evidence_ref,
                sample=sample,
            )
        if number == bound:
            status = UiStatus.INCONCLUSIVE
            verdict = "equals the boundary — no better and no worse"
        else:
            better = number > bound if spec.directionality == "higher_better" else number < bound
            status = UiStatus.PASS if better else UiStatus.FAIL
            verdict = "beats the boundary" if better else "does not beat the boundary"
        return _reading(
            spec,
            value=number,
            display=display,
            status=status,
            interpretation=(
                f"{spec.human_name} is {display}; {verdict} {_fmt(spec, bound)} ({source})"
            ),
            caveat=caveat,
            evidence_ref=evidence_ref,
            reference_value=float(bound),
            sample=sample,
            blocking=status is UiStatus.FAIL,
            proposed=proposed,
        )
    if kind is ReferenceKind.CHANCE_LINE:
        bound = fixed if fixed is not None else 0.5
        delta = number - bound
        side = "above" if delta > 0 else "below" if delta < 0 else "on"
        return _reading(
            spec,
            value=number,
            display=display,
            status=UiStatus.INFORMATIONAL,
            interpretation=(
                f"{spec.human_name} is {display}: {abs(delta):.3f} {side} the {bound:g} chance "
                "line — direction only; no band is applied"
            ),
            caveat=caveat,
            evidence_ref=evidence_ref,
            reference_value=float(bound),
            sample=sample,
        )
    if kind is ReferenceKind.TARGET:
        target = reference if reference is not None else fixed
        if target is None:
            return _reading(
                spec,
                value=number,
                display=display,
                status=UiStatus.INFORMATIONAL,
                interpretation=f"{spec.human_name} is {display} — the target was not persisted",
                evidence_ref=evidence_ref,
                sample=sample,
            )
        return _reading(
            spec,
            value=number,
            display=display,
            status=UiStatus.INFORMATIONAL,
            interpretation=(
                f"{spec.human_name} is {display}: {number - target:+.3f} from the target "
                f"{target:g} ({source}) — the distance only; no good / bad label applies"
            ),
            caveat=caveat,
            evidence_ref=evidence_ref,
            reference_value=float(target),
            sample=sample,
        )
    if kind is ReferenceKind.ADEQUACY_MINIMUM:
        minimum = reference if reference is not None else fixed
        if minimum is None:
            return _reading(
                spec,
                value=number,
                display=display,
                status=UiStatus.UNAVAILABLE,
                interpretation=f"{spec.human_name} is {display}; no adequacy minimum was persisted",
                evidence_ref=evidence_ref,
                sample=sample,
            )
        adequate = number >= minimum
        return _reading(
            spec,
            value=number,
            display=display,
            status=UiStatus.PASS if adequate else UiStatus.INCONCLUSIVE,
            interpretation=(
                f"{spec.human_name} is {display}; the registered minimum is {_fmt(spec, minimum)} "
                f"({source}) — "
                + ("adequate" if adequate else "insufficient evidence, not a failure")
            ),
            caveat=caveat,
            evidence_ref=evidence_ref,
            reference_value=float(minimum),
            sample=sample,
            proposed=proposed,
        )
    if kind is ReferenceKind.LIMIT:
        limit = reference if reference is not None else gate if gate is not None else fixed
        if limit is None:
            return _reading(
                spec,
                value=number,
                display=display,
                status=UiStatus.UNAVAILABLE,
                interpretation=f"{spec.human_name} is {display}; no limit was persisted ({source})",
                evidence_ref=evidence_ref,
                sample=sample,
            )
        within = number <= limit
        utilisation = f"{number / limit * 100.0:.0f}% of the limit" if limit else "no limit"
        return _reading(
            spec,
            value=number,
            display=display,
            status=UiStatus.PASS if within else UiStatus.FAIL,
            interpretation=(
                f"{spec.human_name} is {display} against the limit {_fmt(spec, limit)} "
                f"({utilisation}; {source}) — {'within' if within else 'exceeded'}"
            ),
            caveat=caveat,
            evidence_ref=evidence_ref,
            reference_value=float(limit),
            sample=sample,
            blocking=not within,
        )
    if kind is ReferenceKind.MEASURED_ZERO:
        zero = number == 0
        return _reading(
            spec,
            value=number,
            display=display,
            status=UiStatus.PASS if zero else UiStatus.FAIL,
            interpretation=(
                f"{spec.human_name} is {display} (measured; {source}) — "
                + ("zero, as required" if zero else "nonzero: the policy was reached")
            ),
            evidence_ref=evidence_ref,
            reference_value=0.0,
            sample=sample,
            blocking=not zero,
        )
    # POLICY_ZERO
    return _reading(
        spec,
        value=number,
        display=display,
        status=UiStatus.INFORMATIONAL,
        interpretation=(
            f"{spec.human_name} is {display} — a policy-enforced zero written before any path "
            "is constructed; it is not a measured count and never shown as a passing gate"
        ),
        caveat="policy-enforced zero (not measured evidence)",
        evidence_ref=evidence_ref,
        reference_value=0.0,
        sample=sample,
    )


def evaluate_interval(
    spec_or_key: MetricSpec | str,
    *,
    lower: float | None,
    upper: float | None,
    available: bool = True,
    reason: str | None = None,
    estimate: float | None = None,
    evidence_ref: str | None = None,
    sample: int | None = None,
) -> MetricReading:
    """A 95% interval reading: crossing zero is INCONCLUSIVE; excluding zero on
    the favourable side (by the registered direction) PASS, on the other FAIL."""

    spec = _resolve(spec_or_key)
    low, high = _numeric(lower), _numeric(upper)
    if not available or low is None or high is None:
        return _reading(
            spec,
            value=_numeric(estimate),
            display="—",
            status=UiStatus.UNAVAILABLE,
            interpretation=(
                f"{spec.human_name}: the interval is unavailable"
                + (f" ({reason})" if reason else "")
            ),
            evidence_ref=evidence_ref,
            sample=sample,
        )
    display = f"[{_fmt(spec, low)}, {_fmt(spec, high)}]"
    favourable_positive = spec.directionality != "lower_better"
    if low <= 0.0 <= high:
        status = UiStatus.INCONCLUSIVE
        verdict = "crosses zero — no direction can be concluded"
    elif (low > 0.0) == favourable_positive:
        status = UiStatus.PASS
        verdict = "excludes zero on the favourable side"
    else:
        status = UiStatus.FAIL
        verdict = "excludes zero on the unfavourable side"
    return _reading(
        spec,
        value=_numeric(estimate),
        display=display,
        status=status,
        interpretation=f"{spec.human_name}: the 95% interval {display} {verdict}",
        evidence_ref=evidence_ref,
        reference_value=0.0,
        sample=sample,
        blocking=status is UiStatus.FAIL,
    )


def evaluate_gate_flag(
    spec_or_key: MetricSpec | str,
    flag: object,
    *,
    evaluated: bool = True,
    evidence_ref: str | None = None,
) -> MetricReading:
    """A persisted ``passed`` flag: PASS only for an EVALUATED ``True``."""

    spec = _resolve(spec_or_key)
    status = status_from_gate(flag, evaluated=evaluated)
    if status is UiStatus.PASS:
        text, display = "the evaluated gate passed", "passed"
    elif status is UiStatus.FAIL:
        text, display = "the evaluated gate failed", "failed"
    else:
        text, display = "the gate was not evaluated — it is not shown as passing", "—"
    return _reading(
        spec,
        value=None if status is UiStatus.UNAVAILABLE else float(status is UiStatus.PASS),
        display=display,
        status=status,
        interpretation=f"{spec.human_name}: {text} ({spec.formula_or_source})",
        evidence_ref=evidence_ref,
        blocking=status is UiStatus.FAIL,
    )


def unavailable_reading(
    spec_or_key: MetricSpec | str, reason: str, *, evidence_ref: str | None = None
) -> MetricReading:
    """An UNAVAILABLE reading with the persisted reason (never PASS)."""

    spec = _resolve(spec_or_key)
    return _reading(
        spec,
        value=None,
        display="—",
        status=UiStatus.UNAVAILABLE,
        interpretation=f"{spec.human_name}: {reason}",
        evidence_ref=evidence_ref,
    )
