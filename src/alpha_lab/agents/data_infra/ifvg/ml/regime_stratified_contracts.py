"""Regime performance-stratification contracts (R6.1 §6.G; D4/D6/D7).

The five comparison classes of ML §5.5 and their status gates:

* ``cohort_descriptive`` / ``stratified_prop`` / ``stratified_frontier`` —
  DESCRIPTIVE views that require the deterministic STRATIFICATION_READY
  decision (coverage gates + OOS assignment; ratification-free, D6);
* ``feature_only`` / ``cohort_model`` — MODELED classes that require the
  exact, verified FEATURE_ELIGIBLE decision + owner artifact (D5/D9).

A stratified report binds the EXACT promotion decision, assessment, and
(when required) owner artifact it was built under (``RegimeReportGate``),
the exact assignment evidence (``RegimeAssignmentEvidenceRef`` — the
DESCRIPTIVE ``RegimeOosAssignmentArtifact`` for descriptive classes; the
fold-local feature artifact only for modeled classes, D7), and carries the
permanent development badge + ``counterfactual_claim="none"``: nothing in
a stratified report is a selection input, a counterfactual, or a
promotion. A report below its class minimum is UNCONSTRUCTIBLE (validator).
"""

from __future__ import annotations

import math
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field, model_validator

from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)
from .regime_contracts import (
    PROMOTION_SEQUENCE,
    REGIME_PROPOSED_DEFAULTS,
    ObservationGranularity,
    RegimeRole,
    RegimeStatus,
    assert_lawful_role,
)

__all__ = [
    "REGIME_STRATIFIED_REPORT_STORE",
    "STRATIFIED_REPORT_DETAIL_SIDECAR",
    "ACCOUNT_EVENT_REGIME_SUMMARY_SIDECAR",
    "EventRegimeSummaryBudget",
    "ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1",
    "STRATIFICATION_REGISTERED_BUDGETS",
    "STRATIFICATION_FORMULA_VERSION",
    "DEVELOPMENT_BADGE",
    "FRONTIER_ROLE",
    "PROP_EVENT_ATTRIBUTION_POLICY_ID",
    "MINIMUM_TRADES_PER_REGIME_STRATUM",
    "MINIMUM_TRAINING_ROWS_PER_REGIME_STRATUM",
    "RegimeStratificationClass",
    "CLASS_MINIMUM_STATUS",
    "CLASS_INTERPRETATION",
    "CLASS_ROLE",
    "status_rank",
    "status_meets",
    "RegimeReportGate",
    "RegimeAssignmentEvidenceRef",
    "RegimeStratumKey",
    "StratumMetrics",
    "StrategyStratumRow",
    "NET_R_ACCOUNTING_FORMULA_VERSION",
    "NET_R_ACCOUNTING_BASIS",
    "RegimeNetRAccounting",
    "CohortDescriptiveBody",
    "FrontierStratumCell",
    "StratifiedFrontierBody",
    "PropStratumRow",
    "StratifiedPropBody",
    "RegimeStratifiedReportBody",
    "RegimeStratifiedReportPayload",
    "RegimeStratifiedReportEnvelope",
]

REGIME_STRATIFIED_REPORT_STORE = "regime_stratified_reports"
STRATIFIED_REPORT_DETAIL_SIDECAR = "stratified_report_detail.json"
#: D15 (plan §6.G): the report-local, bounded, ZSTD-Parquet event-regime
#: summary a ``stratified_prop`` report carries INSTEAD of row-oriented
#: event JSON — one row per (simulation, path, regime stratum, typed
#: unattributable reason, event type) keyed by the exact assignment evidence.
ACCOUNT_EVENT_REGIME_SUMMARY_SIDECAR = "account_event_regime_summary.parquet"
STRATIFICATION_FORMULA_VERSION = "regime_stratification_v1"


class EventRegimeSummaryBudget(FrozenContract):
    """The registered storage budget of the event-regime summary (D15
    family) — bound into the ``stratified_prop`` body, so a different budget
    mints a different report; a summary over budget refuses BEFORE
    publication (no partial artifact ever exists)."""

    budget_id: str = Field(min_length=1)
    max_summary_rows: int = Field(ge=1)
    max_published_bytes: int = Field(ge=1)


#: The registered V1 limits of the event-regime summary.
ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1 = EventRegimeSummaryBudget(
    budget_id="account_event_regime_summary_budget_v1",
    max_summary_rows=5_000_000,
    max_published_bytes=268_435_456,
)

#: The registered storage budgets of the stratification lane, surfaced by
#: the Regime Lane stamp table next to the scientific defaults (plan §8):
#: ``registered_storage_budget`` — a capacity limit, not a research default.
STRATIFICATION_REGISTERED_BUDGETS: MappingProxyType[str, MappingProxyType[str, Any]] = (
    MappingProxyType(
        {
            "account_event_regime_summary_max_rows": MappingProxyType(
                {
                    "value": ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1.max_summary_rows,
                    "stamp": "registered_storage_budget",
                    "budget_id": ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1.budget_id,
                    "decision": "D15",
                }
            ),
            "account_event_regime_summary_max_bytes": MappingProxyType(
                {
                    "value": ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1.max_published_bytes,
                    "stamp": "registered_storage_budget",
                    "budget_id": ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1.budget_id,
                    "decision": "D15",
                }
            ),
        }
    )
)
DEVELOPMENT_BADGE = "development_descriptive_research_only_never_selection_input"
FRONTIER_ROLE = "descriptive_view_never_selection_input"
PROP_EVENT_ATTRIBUTION_POLICY_ID = str(
    REGIME_PROPOSED_DEFAULTS["prop_event_attribution_policy"]["value"]
)

#: Stamped proposal defaults (plan §8, decision-30 family; single-sourced
#: into ``REGIME_PROPOSED_DEFAULTS`` — ``proposed_protocol_default``, owner
#: ratification before research weight).
MINIMUM_TRADES_PER_REGIME_STRATUM = int(
    REGIME_PROPOSED_DEFAULTS["minimum_trades_per_regime_stratum"]["value"]
)
MINIMUM_TRAINING_ROWS_PER_REGIME_STRATUM = int(
    REGIME_PROPOSED_DEFAULTS["minimum_training_rows_per_regime_stratum"]["value"]
)


class RegimeStratificationClass(StrEnum):
    COHORT_DESCRIPTIVE = "cohort_descriptive"
    FEATURE_ONLY = "feature_only"
    COHORT_MODEL = "cohort_model"
    STRATIFIED_PROP = "stratified_prop"
    STRATIFIED_FRONTIER = "stratified_frontier"


CLASS_MINIMUM_STATUS: MappingProxyType[RegimeStratificationClass, RegimeStatus] = (
    MappingProxyType(
        {
            RegimeStratificationClass.COHORT_DESCRIPTIVE: RegimeStatus.STRATIFICATION_READY,
            RegimeStratificationClass.STRATIFIED_PROP: RegimeStatus.STRATIFICATION_READY,
            RegimeStratificationClass.STRATIFIED_FRONTIER: RegimeStatus.STRATIFICATION_READY,
            RegimeStratificationClass.FEATURE_ONLY: RegimeStatus.FEATURE_ELIGIBLE,
            RegimeStratificationClass.COHORT_MODEL: RegimeStatus.FEATURE_ELIGIBLE,
        }
    )
)

CLASS_INTERPRETATION: MappingProxyType[RegimeStratificationClass, str] = MappingProxyType(
    {
        RegimeStratificationClass.COHORT_DESCRIPTIVE: "descriptive",
        RegimeStratificationClass.STRATIFIED_PROP: "descriptive",
        RegimeStratificationClass.STRATIFIED_FRONTIER: "descriptive",
        RegimeStratificationClass.FEATURE_ONLY: "modeled",
        RegimeStratificationClass.COHORT_MODEL: "modeled",
    }
)

#: The role a report of each class is built under (the ROLE ladder of
#: ``regime_contracts.assert_lawful_role`` decides whether the decision's
#: status earns it).
CLASS_ROLE: MappingProxyType[RegimeStratificationClass, RegimeRole] = MappingProxyType(
    {
        RegimeStratificationClass.COHORT_DESCRIPTIVE: RegimeRole.STRATIFICATION_ONLY,
        RegimeStratificationClass.STRATIFIED_PROP: RegimeRole.STRATIFICATION_ONLY,
        RegimeStratificationClass.STRATIFIED_FRONTIER: RegimeRole.STRATIFICATION_ONLY,
        RegimeStratificationClass.FEATURE_ONLY: RegimeRole.FEATURE_GENERATOR,
        RegimeStratificationClass.COHORT_MODEL: RegimeRole.FEATURE_GENERATOR,
    }
)


def status_rank(status: RegimeStatus) -> int:
    """Position on the promotion ladder; blocked / experimental / superseded
    states rank below every ladder step."""

    status = RegimeStatus(status)
    return PROMOTION_SEQUENCE.index(status) if status in PROMOTION_SEQUENCE else -1


def status_meets(status: RegimeStatus, minimum: RegimeStatus) -> bool:
    return status_rank(status) >= status_rank(minimum) and status_rank(status) >= 0


class RegimeReportGate(FrozenContract):
    """The EXACT authority a report was built under (never a "latest" lookup)."""

    regime_promotion_decision_id: str = Field(pattern=SHA256_PATTERN)
    owner_decision_artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    capability_assessment_id: str = Field(pattern=SHA256_PATTERN)
    status_at_report: RegimeStatus
    role_at_report: RegimeRole
    minimum_status_required: RegimeStatus
    authority_source: Literal["s10_structural", "frozen_owner_evidence"]

    @model_validator(mode="after")
    def _lawful(self):
        if not status_meets(self.status_at_report, self.minimum_status_required):
            raise ValueError(
                f"report gate status {RegimeStatus(self.status_at_report).value} is below "
                f"the class minimum {RegimeStatus(self.minimum_status_required).value}; "
                "nothing here promotes"
            )
        assert_lawful_role(self.role_at_report, self.status_at_report)
        needs_owner = status_rank(self.minimum_status_required) >= status_rank(
            RegimeStatus.FEATURE_ELIGIBLE
        )
        if needs_owner and self.owner_decision_artifact_id is None:
            raise ValueError(
                "a FEATURE_ELIGIBLE-or-later report gate requires the exact owner "
                "decision artifact id"
            )
        if needs_owner != (self.authority_source == "frozen_owner_evidence"):
            raise ValueError(
                "authority_source must be frozen_owner_evidence exactly for "
                "FEATURE_ELIGIBLE-or-later classes and s10_structural otherwise"
            )
        return self


class RegimeAssignmentEvidenceRef(FrozenContract):
    """The exact assignment evidence a report stratified over (D7).

    R6.1-FIX (§3.1, F-01): the ref also pins the descriptive artifact's
    post-materialization table hash and its enforced schema hash — so a
    persisted report names the exact bytes it stratified over, and S14
    refuses a frame or hash that differs from the verified artifact."""

    observation_granularity: ObservationGranularity
    regime_fit_ids: tuple[str, ...] = Field(min_length=1)
    regime_fold_set_id: str = Field(pattern=SHA256_PATTERN)
    fold_schedule_id: str = Field(pattern=SHA256_PATTERN)
    regime_oos_assignment_id: str = Field(pattern=SHA256_PATTERN)
    assignment_table_sha256: str = Field(pattern=SHA256_PATTERN)
    assignment_schema_hash: str = Field(pattern=SHA256_PATTERN)
    regime_fold_feature_artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _sorted_fits(self):
        if tuple(sorted(self.regime_fit_ids)) != tuple(self.regime_fit_ids):
            raise ValueError("regime_fit_ids must be sorted (order-free identity)")
        return self


class RegimeStratumKey(FrozenContract):
    stratum: Literal["pooled_all", "pooled_regime_covered", "regime", "unassigned"]
    canonical_reporting_cluster_id: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _coherent(self):
        if (self.stratum == "regime") != (self.canonical_reporting_cluster_id is not None):
            raise ValueError("a regime stratum carries exactly one canonical reporting id")
        return self

    @property
    def label(self) -> str:
        if self.stratum == "regime":
            return f"regime:{self.canonical_reporting_cluster_id}"
        return self.stratum


class StratumMetrics(FrozenContract):
    """The scalar projection of ``StrategyMetrics`` (the full dump rides the
    detail sidecar; the identity payload stays deep-immutable)."""

    executed_trades: int = Field(ge=0)
    independent_days: int = Field(ge=0)
    gross_expectancy_r: float | None
    net_expectancy_r: float | None
    profit_factor: float | None
    max_drawdown_r: float | None
    time_under_water_days: int | None
    time_block_sign_consistency: float | None
    session_stability_score: float | None
    top_day_pnl_share: float | None
    top_setup_pnl_share: float | None
    net_expectancy_bootstrap_ci95: tuple[float, float] | None


class StrategyStratumRow(FrozenContract):
    key: RegimeStratumKey
    executed_trades: int = Field(ge=0)
    share_of_trades: float = Field(ge=0.0, le=1.0)
    typed_state: Literal["reported", "insufficient_regime_partition"]
    metrics: StratumMetrics | None
    #: descriptive restriction of the POOLED rung's OOS predictions to the
    #: stratum (never a specialized model): brier / log-loss / rows
    pooled_model_skill: ImmutableMap[str, float] | None
    cohort_id: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _typed(self):
        if (self.typed_state == "reported") != (self.metrics is not None):
            raise ValueError("a reported stratum carries metrics; a typed one carries none")
        return self


#: R6.1-FIX §3.5 (F-09): the raw net-R accounting formula — EVERY valid
#: assigned trade enters the concentration facts; the reportability floor
#: governs interval/reportability metrics only.
NET_R_ACCOUNTING_FORMULA_VERSION = "regime_net_r_accounting_v1"
NET_R_ACCOUNTING_BASIS = "all_valid_assigned_trades_v1"


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)


class RegimeNetRAccounting(FrozenContract):
    """Plan §3.5 definitions, persisted verbatim (no statistical confidence
    is inferred for thin strata — this is arithmetic over per-trade net R):

    * ``net_r_by_regime[r]`` = Σ per-trade net R of the valid trades in ``r``;
    * ``unassigned_net_r`` = Σ per-trade net R of invalid/unassigned trades;
    * ``abs_net_r_mass`` = Σ |net_r_by_regime|; shares are |net| / mass, or
      null with ``zero_abs_net_r_mass``;
    * ``signed_contribution_fraction_by_regime[r]`` = net_r_by_regime[r] /
      assigned_net_r_total when that total is nonzero, else null with
      ``zero_assigned_net_r_total``;
    * ``works_only_in_regime`` is TRUE only when the assigned net R total is
      positive, exactly one assigned regime has net R > 0, every other
      assigned regime has net R ≤ 0, and no trade is unassigned; FALSE
      otherwise; NULL with ``incomplete_assignment_accounting`` ONLY when
      unassigned trades prevent a claim the assigned side would otherwise
      support — a claim the assigned side already refutes is FALSE whatever
      the unassigned count (adversarial RA-03). The plan's predicate is
      followed verbatim, so a SINGLE assigned regime with positive net R is
      a vacuous TRUE ("every other regime" is empty): ``assigned_regime_count``
      makes that vacuity visible to every reader.

    The validator RECOMPUTES every derived field from ``net_r_by_regime``
    (adversarial RA-02): a well-typed body whose mass, total, shares,
    fractions, top share, regime count or claim disagrees with its own
    per-regime sums is refused.
    """

    formula_version: Literal["regime_net_r_accounting_v1"] = NET_R_ACCOUNTING_FORMULA_VERSION
    basis: Literal["all_valid_assigned_trades_v1"] = NET_R_ACCOUNTING_BASIS
    trade_count_by_regime: ImmutableMap[int, int]
    net_r_by_regime: ImmutableMap[int, float]
    #: the number of assigned regimes (``len(net_r_by_regime)``) — 1 marks
    #: a vacuous ``works_only_in_regime`` truth
    assigned_regime_count: int = Field(ge=0)
    assigned_trade_count: int = Field(ge=0)
    unassigned_trade_count: int = Field(ge=0)
    unassigned_net_r: float
    assigned_net_r_total: float
    abs_net_r_mass: float = Field(ge=0.0)
    abs_net_r_share_by_regime: ImmutableMap[int, float] | None
    signed_contribution_fraction_by_regime: ImmutableMap[int, float] | None
    top_regime_abs_net_r_share: float | None
    works_only_in_regime: bool | None
    works_only_in_regime_id: int | None
    works_only_in_regime_reason: Literal["incomplete_assignment_accounting"] | None
    zero_denominator_reasons: tuple[
        Literal["zero_abs_net_r_mass", "zero_assigned_net_r_total"], ...
    ]

    @model_validator(mode="after")
    def _coherent(self):
        net = {int(key): float(value) for key, value in self.net_r_by_regime.items()}
        counts = {int(key): int(value) for key, value in self.trade_count_by_regime.items()}
        if set(counts) != set(net):
            raise ValueError("trade counts and net R must cover the same regimes")
        if sum(counts.values()) != self.assigned_trade_count:
            raise ValueError("assigned_trade_count must equal the per-regime counts")
        if self.assigned_regime_count != len(net):
            raise ValueError("assigned_regime_count must equal the number of assigned regimes")
        # RA-02: every derived field is recomputed from the per-regime sums
        total = float(sum(net.values()))
        mass = float(sum(abs(value) for value in net.values()))
        if not _close(self.assigned_net_r_total, total):
            raise ValueError("assigned_net_r_total must equal the sum of net_r_by_regime")
        if not _close(self.abs_net_r_mass, mass):
            raise ValueError("abs_net_r_mass must equal the sum of |net_r_by_regime|")
        zero_mass = not mass > 0
        if zero_mass != ("zero_abs_net_r_mass" in self.zero_denominator_reasons):
            raise ValueError("zero_abs_net_r_mass is recorded exactly when abs_net_r_mass is zero")
        if (self.abs_net_r_share_by_regime is None) != zero_mass:
            raise ValueError("abs shares are null exactly under zero_abs_net_r_mass")
        if self.abs_net_r_share_by_regime is None:
            if self.top_regime_abs_net_r_share is not None:
                raise ValueError("the top share is defined exactly when the shares are")
        else:
            shares = {
                int(key): float(value) for key, value in self.abs_net_r_share_by_regime.items()
            }
            if set(shares) != set(net) or any(
                not _close(shares[key], abs(net[key]) / mass) for key in net
            ):
                raise ValueError(
                    "abs_net_r_share_by_regime must equal |net_r_by_regime| / abs_net_r_mass"
                )
            if self.top_regime_abs_net_r_share is None or not _close(
                self.top_regime_abs_net_r_share, max(shares.values())
            ):
                raise ValueError("top_regime_abs_net_r_share must equal the maximum abs share")
        zero_total = total == 0.0
        if zero_total != ("zero_assigned_net_r_total" in self.zero_denominator_reasons):
            raise ValueError(
                "zero_assigned_net_r_total is recorded exactly when assigned_net_r_total is zero"
            )
        if (self.signed_contribution_fraction_by_regime is None) != zero_total:
            raise ValueError(
                "signed contribution fractions are null exactly under zero_assigned_net_r_total"
            )
        if self.signed_contribution_fraction_by_regime is not None:
            signed = {
                int(key): float(value)
                for key, value in self.signed_contribution_fraction_by_regime.items()
            }
            if set(signed) != set(net) or any(
                not _close(signed[key], net[key] / total) for key in net
            ):
                raise ValueError(
                    "signed_contribution_fraction_by_regime must equal net_r_by_regime / "
                    "assigned_net_r_total"
                )
        # the claim (plan §3.5 on the assigned side; RA-03 null semantics)
        positive = [key for key, value in net.items() if value > 0]
        assigned_claim = bool(total > 0 and len(positive) == 1)
        if (self.works_only_in_regime is None) != (
            self.works_only_in_regime_reason == "incomplete_assignment_accounting"
        ):
            raise ValueError(
                "works_only_in_regime is null exactly under incomplete_assignment_accounting"
            )
        if self.works_only_in_regime is None and self.unassigned_trade_count == 0:
            raise ValueError("a null works_only_in_regime claim requires unassigned trades")
        expected: bool | None = (
            assigned_claim
            if self.unassigned_trade_count == 0
            else (None if assigned_claim else False)
        )
        if (self.works_only_in_regime is None) != (expected is None) or (
            expected is not None and bool(self.works_only_in_regime) != expected
        ):
            raise ValueError(
                "works_only_in_regime must equal the assigned-side predicate (null only when "
                "unassigned trades prevent a claim the assigned side supports)"
            )
        if (self.works_only_in_regime_id is not None) != (self.works_only_in_regime is True):
            raise ValueError("works_only_in_regime_id is set exactly when the claim is true")
        if self.works_only_in_regime is True and int(self.works_only_in_regime_id) != positive[0]:
            raise ValueError("works_only_in_regime_id must name the unique positive regime")
        return self


class CohortDescriptiveBody(FrozenContract):
    comparison_class: Literal["cohort_descriptive"] = "cohort_descriptive"
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    #: the binding hash of the VALIDATED, NORMALIZED executed-trade table
    #: (R6.1-FIX F-10B) and — when the service verified the child's frame
    #: against the persisted executed-trade table artifact (§3.7; adversarial
    #: RA-01) — that artifact's exact id AND its own projection-bytes hash
    #: (``ExecutedTradeTableEnvelope.executed_trade_table_sha256``); the two
    #: artifact fields are set together or not at all
    executed_trade_table_sha256: str = Field(pattern=SHA256_PATTERN)
    executed_trade_table_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    executed_trade_table_artifact_sha256: str | None = Field(
        default=None, pattern=SHA256_PATTERN
    )
    cost_points: float
    evaluation_config_hash: str = Field(pattern=SHA256_PATTERN)
    trades_total: int = Field(ge=0)
    trades_regime_covered: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0.0, le=1.0)
    minimum_trades_per_regime_stratum: int = Field(ge=1)
    strata: tuple[StrategyStratumRow, ...]
    #: concentration (R6.1-FIX §3.5): the single regime in which the strategy
    #: "works" when — and only when — ``net_r_accounting.works_only_in_regime``
    #: is true (empty otherwise), and the top regime's share of absolute net
    #: R over ALL valid assigned trades (the floor never excludes a regime)
    works_only_in_regime: tuple[int, ...]
    top_regime_abs_net_r_share: float | None
    net_r_accounting: RegimeNetRAccounting
    unassigned_reasons: ImmutableMap[str, int]

    @model_validator(mode="after")
    def _coherent(self):
        labels = [row.key.label for row in self.strata]
        if len(set(labels)) != len(labels):
            raise ValueError("strata must be unique")
        if "pooled_all" not in labels:
            raise ValueError("the pooled_all stratum is mandatory")
        partition = [
            row for row in self.strata if row.key.stratum in ("regime", "unassigned")
        ]
        if partition and abs(sum(row.share_of_trades for row in partition) - 1.0) > 1e-9:
            raise ValueError("regime + unassigned shares must sum to one")
        if self.trades_regime_covered > self.trades_total:
            raise ValueError("covered trades exceed the total")
        accounting = self.net_r_accounting
        expected_works = (
            (accounting.works_only_in_regime_id,)
            if accounting.works_only_in_regime is True
            else ()
        )
        if tuple(self.works_only_in_regime) != expected_works:
            raise ValueError("works_only_in_regime must mirror the net-R accounting claim")
        if self.top_regime_abs_net_r_share != accounting.top_regime_abs_net_r_share:
            raise ValueError("top_regime_abs_net_r_share must mirror the net-R accounting")
        if accounting.assigned_trade_count != self.trades_regime_covered:
            raise ValueError("the accounting's assigned trades must equal the covered trades")
        if accounting.assigned_trade_count + accounting.unassigned_trade_count != (
            self.trades_total
        ):
            raise ValueError("assigned + unassigned trades must equal the total")
        if (self.executed_trade_table_id is None) != (
            self.executed_trade_table_artifact_sha256 is None
        ):
            raise ValueError(
                "executed_trade_table_id and executed_trade_table_artifact_sha256 bind the "
                "persisted artifact together (both set, or neither)"
            )
        # RA-02: the accounting's regimes are exactly the regime strata, and the
        # per-regime trade counts are the strata's executed trades
        regime_rows = {
            int(row.key.canonical_reporting_cluster_id): row
            for row in self.strata
            if row.key.stratum == "regime"
        }
        if set(regime_rows) != {int(key) for key in accounting.trade_count_by_regime}:
            raise ValueError("the regime strata must cover exactly the accounting's regimes")
        for regime, row in regime_rows.items():
            if int(accounting.trade_count_by_regime[regime]) != int(row.executed_trades):
                raise ValueError(
                    f"trade_count_by_regime[{regime}] must equal the regime stratum's "
                    "executed_trades"
                )
        return self


class FrontierStratumCell(FrozenContract):
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    key: RegimeStratumKey
    executed_trades: int = Field(ge=0)
    typed_state: Literal["reported", "insufficient_regime_partition"]
    metric_values: ImmutableMap[str, float | None]


class StratifiedFrontierBody(FrozenContract):
    comparison_class: Literal["stratified_frontier"] = "stratified_frontier"
    frontier_id: str = Field(pattern=SHA256_PATTERN)
    objective_metrics: tuple[str, ...] = Field(min_length=1)
    children: tuple[str, ...]
    #: read from the POOLED frontier only — never recomputed per stratum
    on_frontier: ImmutableMap[str, bool]
    cells: tuple[FrontierStratumCell, ...]
    works_only_in_regime_by_child: ImmutableMap[str, tuple[int, ...]]
    children_without_strata: tuple[str, ...]
    frontier_role: Literal["descriptive_view_never_selection_input"] = FRONTIER_ROLE

    @model_validator(mode="after")
    def _coherent(self):
        if tuple(sorted(self.children)) != tuple(self.children):
            raise ValueError("children must be sorted")
        if set(self.on_frontier) != set(self.children):
            raise ValueError("on_frontier must cover exactly the children")
        for cell in self.cells:
            if cell.core_replay_id not in self.on_frontier:
                raise ValueError("a cell names a child outside the frontier's children")
        return self


class PropStratumRow(FrozenContract):
    firm_label: str
    simulation_mode: str
    account_simulation_id: str = Field(pattern=SHA256_PATTERN)
    key: RegimeStratumKey
    event_counts_by_type: ImmutableMap[str, int]
    event_share_by_type: ImmutableMap[str, float]
    payout_trader_amount_sum: float
    fee_amount_sum: float
    realized_pnl_sum: float
    events_total: int = Field(ge=0)


class StratifiedPropBody(FrozenContract):
    comparison_class: Literal["stratified_prop"] = "stratified_prop"
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    account_simulation_ids: tuple[str, ...]
    attribution_policy_id: Literal["source_trade_then_pit_v1"] = PROP_EVENT_ATTRIBUTION_POLICY_ID
    probabilities_reestimated: Literal[False] = False
    strata: tuple[PropStratumRow, ...]
    events_total: int = Field(ge=0)
    events_attributed: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0.0, le=1.0)
    partial_coverage: bool
    unattributable_by_reason: ImmutableMap[str, int]
    #: simulations persisted under ``none_v0`` (no event detail) — typed, not
    #: silently dropped
    evidence_not_persisted: tuple[str, ...]
    #: D15: the registered budget the report-local event-regime summary was
    #: built under and the exact number of summary rows it carries (the
    #: summary bytes themselves are bound by the ENVELOPE, never hashed here)
    summary_budget: EventRegimeSummaryBudget = ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1
    summary_rows: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _coherent(self):
        if tuple(sorted(self.account_simulation_ids)) != tuple(self.account_simulation_ids):
            raise ValueError("account_simulation_ids must be sorted")
        if self.events_attributed > self.events_total:
            raise ValueError("attributed events exceed the total")
        if self.partial_coverage != (self.events_attributed < self.events_total):
            raise ValueError("partial_coverage must reflect attributed < total")
        if self.summary_rows > self.summary_budget.max_summary_rows:
            raise ValueError(
                f"the event-regime summary carries {self.summary_rows} rows, over the "
                f"registered budget of {self.summary_budget.max_summary_rows} "
                f"({self.summary_budget.budget_id})"
            )
        if self.summary_rows > self.events_total:
            raise ValueError("the event-regime summary cannot carry more rows than events")
        return self


RegimeStratifiedReportBody = Annotated[
    CohortDescriptiveBody | StratifiedPropBody | StratifiedFrontierBody,
    Field(discriminator="comparison_class"),
]


class RegimeStratifiedReportPayload(FrozenContract):
    resolved_regime_protocol_id: str = Field(pattern=SHA256_PATTERN)
    comparison_class: RegimeStratificationClass
    gate: RegimeReportGate
    assignment_evidence: RegimeAssignmentEvidenceRef
    #: the persisted inputs the report was built from (costed evaluation,
    #: frontier, account simulations, …) — exact ids, never frames
    source_metric_refs: tuple[str, ...]
    interpretation: Literal["descriptive", "modeled"]
    counterfactual_claim: Literal["none"] = "none"
    development_badge: Literal[
        "development_descriptive_research_only_never_selection_input"
    ] = DEVELOPMENT_BADGE
    body: RegimeStratifiedReportBody
    formula_version: Literal["regime_stratification_v1"] = STRATIFICATION_FORMULA_VERSION

    @model_validator(mode="after")
    def _unconstructible_below_minimum(self):
        comparison_class = RegimeStratificationClass(self.comparison_class)
        if self.body.comparison_class != comparison_class.value:
            raise ValueError("the report body does not belong to the declared class")
        minimum = CLASS_MINIMUM_STATUS[comparison_class]
        if self.gate.minimum_status_required is not minimum:
            raise ValueError(
                f"class {comparison_class.value} requires {minimum.value}; the gate "
                f"declares {RegimeStatus(self.gate.minimum_status_required).value}"
            )
        if not status_meets(self.gate.status_at_report, minimum):
            raise ValueError(
                f"a {comparison_class.value} report below {minimum.value} is unconstructible; "
                "nothing here promotes"
            )
        if self.interpretation != CLASS_INTERPRETATION[comparison_class]:
            raise ValueError("interpretation disagrees with the class")
        if self.gate.role_at_report is not CLASS_ROLE[comparison_class]:
            raise ValueError("the gate role disagrees with the class role")
        if tuple(sorted(self.source_metric_refs)) != tuple(self.source_metric_refs):
            raise ValueError("source_metric_refs must be sorted")
        return self


class RegimeStratifiedReportEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "regime_stratified_report_id"

    regime_stratified_report_id: str = Field(pattern=SHA256_PATTERN)
    payload: RegimeStratifiedReportPayload
    #: post-materialization fact binding the detail sidecar
    detail_sha256: str = Field(pattern=SHA256_PATTERN)
    #: post-materialization facts binding the D15 event-regime summary
    #: sidecar of a ``stratified_prop`` report (sha256 / bytes / rows /
    #: schema hash — final ruling 7: envelope extras, never payload fields);
    #: all four present exactly for ``stratified_prop`` reports
    account_event_regime_summary_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    account_event_regime_summary_bytes: int | None = Field(default=None, ge=0)
    account_event_regime_summary_rows: int | None = Field(default=None, ge=0)
    account_event_regime_summary_schema_hash: str | None = Field(
        default=None, pattern=SHA256_PATTERN
    )

    @model_validator(mode="after")
    def _summary_binding_coherent(self):
        bound = (
            self.account_event_regime_summary_sha256,
            self.account_event_regime_summary_bytes,
            self.account_event_regime_summary_rows,
            self.account_event_regime_summary_schema_hash,
        )
        present = tuple(value is not None for value in bound)
        is_prop = self.payload.comparison_class is RegimeStratificationClass.STRATIFIED_PROP
        if is_prop and not all(present):
            raise ValueError(
                "a stratified_prop report envelope must bind its account_event_regime_summary "
                "sidecar (sha256, bytes, rows, schema hash)"
            )
        if not is_prop and any(present):
            raise ValueError(
                "only a stratified_prop report envelope carries an event-regime summary binding"
            )
        if is_prop and int(self.account_event_regime_summary_rows) != int(
            self.payload.body.summary_rows
        ):
            raise ValueError(
                "the envelope's event-regime summary row count disagrees with the report body"
            )
        return self


def _example_payload() -> RegimeStratifiedReportPayload:
    gate = RegimeReportGate(
        regime_promotion_decision_id="1" * 64,
        owner_decision_artifact_id=None,
        capability_assessment_id="2" * 64,
        status_at_report=RegimeStatus.STRATIFICATION_READY,
        role_at_report=RegimeRole.STRATIFICATION_ONLY,
        minimum_status_required=RegimeStatus.STRATIFICATION_READY,
        authority_source="s10_structural",
    )
    evidence = RegimeAssignmentEvidenceRef(
        observation_granularity=ObservationGranularity.CANDIDATE_STAGE_ROW,
        regime_fit_ids=("3" * 64,),
        regime_fold_set_id="4" * 64,
        fold_schedule_id="5" * 64,
        regime_oos_assignment_id="6" * 64,
        assignment_table_sha256="9" * 64,
        assignment_schema_hash="c" * 64,
    )
    # the audit example is a stratified_prop report: the ONE class whose
    # envelope carries the D15 event-regime summary binding (placeholders
    # for every declared envelope extra are exercised by the identity audit)
    pooled = PropStratumRow(
        firm_label="firm_a",
        simulation_mode="historical_closed_trade",
        account_simulation_id="7" * 64,
        key=RegimeStratumKey(stratum="pooled_all"),
        event_counts_by_type={"equity_update": 3, "payout": 1},
        event_share_by_type={"equity_update": 0.75, "payout": 0.25},
        payout_trader_amount_sum=500.0,
        fee_amount_sum=0.0,
        realized_pnl_sum=140.0,
        events_total=4,
    )
    regime = PropStratumRow(
        firm_label="firm_a",
        simulation_mode="historical_closed_trade",
        account_simulation_id="7" * 64,
        key=RegimeStratumKey(stratum="regime", canonical_reporting_cluster_id=0),
        event_counts_by_type={"equity_update": 3},
        event_share_by_type={"equity_update": 0.75},
        payout_trader_amount_sum=0.0,
        fee_amount_sum=0.0,
        realized_pnl_sum=140.0,
        events_total=3,
    )
    unassigned = PropStratumRow(
        firm_label="firm_a",
        simulation_mode="historical_closed_trade",
        account_simulation_id="7" * 64,
        key=RegimeStratumKey(stratum="unassigned"),
        event_counts_by_type={"payout": 1},
        event_share_by_type={"payout": 0.25},
        payout_trader_amount_sum=500.0,
        fee_amount_sum=0.0,
        realized_pnl_sum=0.0,
        events_total=1,
    )
    body = StratifiedPropBody(
        core_replay_id="8" * 64,
        account_simulation_ids=("7" * 64,),
        strata=(pooled, regime, unassigned),
        events_total=4,
        events_attributed=3,
        coverage_fraction=0.75,
        partial_coverage=True,
        unattributable_by_reason={"no_source_trade": 1},
        evidence_not_persisted=(),
        # the identity audit stamps 0 into every integer envelope extra; the
        # envelope binding must agree with the body's row count
        summary_rows=0,
    )
    return RegimeStratifiedReportPayload(
        resolved_regime_protocol_id="a" * 64,
        comparison_class=RegimeStratificationClass.STRATIFIED_PROP,
        gate=gate,
        assignment_evidence=evidence,
        source_metric_refs=("7" * 64, "b" * 64),
        interpretation="descriptive",
        body=body,
    )


register_identity_pair(
    name="RegimeStratifiedReport",
    envelope_cls=RegimeStratifiedReportEnvelope,
    payload_cls=RegimeStratifiedReportPayload,
    id_field="regime_stratified_report_id",
    example_factory=_example_payload,
    extra_envelope_fields=(
        "detail_sha256",
        "account_event_regime_summary_sha256",
        "account_event_regime_summary_bytes",
        "account_event_regime_summary_rows",
        "account_event_regime_summary_schema_hash",
    ),
)
