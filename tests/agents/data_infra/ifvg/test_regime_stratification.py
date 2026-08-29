"""R6.1 workstream G (descriptive half) — regime performance stratification
(plan §6.G; D4/D6/D7; §9.2 ``test_regime_stratification.py``).

Status gate per class × status (the owner half under the run scope); a
decision over another protocol / an uncovered fit set refused; pooled ==
the child's own metrics and shares sum to one; thin strata typed;
``RegimeFilterRef`` consumed and cohort ids minted; the exact trade join;
prop attribution by source trade never by a synthetic clock; the D15
report-local event-regime summary (schema, exact aggregates and order,
budget refusal BEFORE publication, tamper refusal on load, no row-oriented
event JSON); the D15 precedence + the event's own trading day; the
frontier never a selection input; modeled classes recorded as S09c
deliveries (never refusals) when delivered; reports persist and register
with the identity audit; determinism; reports built from persisted
artifacts only.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.features.arrow_tables import arrow_schema_hash
from alpha_lab.agents.data_infra.ifvg.ml.regime_assignment_sources import (
    UNASSIGNED_EVIDENCE_QUALITY_TOKEN,
    load_regime_oos_assignment_table,
    regime_filter_mask,
    regime_for_trades,
    stratum_cohorts,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
    RegimePromotionDecision,
    RegimePromotionDecisionEnvelope,
    RegimeRole,
    RegimeStatus,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_executor import execute_regime_protocol
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import resolve_kmeans_protocol
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
    persist_regime_promotion,
    persist_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_stratification_gate import (
    RegimeStatusRefusalError,
    resolve_report_gate,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_stratification_service import (
    ChildStratificationInputs,
    StratificationInputs,
    build_regime_stratified_reports,
    load_regime_stratified_report,
    load_regime_stratified_report_detail,
    load_regime_stratified_report_summary,
    persist_regime_stratified_report,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_contracts import (
    ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1,
    ACCOUNT_EVENT_REGIME_SUMMARY_SIDECAR,
    CLASS_MINIMUM_STATUS,
    MINIMUM_TRADES_PER_REGIME_STRATUM,
    STRATIFICATION_REGISTERED_BUDGETS,
    EventRegimeSummaryBudget,
    RegimeAssignmentEvidenceRef,
    RegimeReportGate,
    RegimeStratificationClass,
    RegimeStratumKey,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_frontier import (
    build_stratified_frontier_body,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_prop import (
    ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA,
    ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_HASH,
    EVENT_DETAIL_COLUMNS,
    HISTORICAL_CLOCK_POLICY_ID,
    SUMMARY_ROW_KEYS,
    SYNTHETIC_CLOCK_POLICY_ID,
    EventRegimeSummaryBudgetError,
    build_stratified_prop_body,
    event_detail_frame,
    read_account_event_regime_summary,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_strategy import (
    build_cohort_descriptive_body,
    stratum_metrics_from,
)
from alpha_lab.agents.data_infra.ifvg.search.frontier import ObjectiveSpec, build_frontier
from alpha_lab.agents.data_infra.ifvg.search.identities import registered_identity_pairs
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (
    SearchFrontierEnvelope,
    SearchFrontierPayload,
)
from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (
    synthetic_owner_decision_fixture,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SearchStoreError,
    envelope_destination,
    save_or_reuse_envelope,
)
from alpha_lab.agents.data_infra.ifvg.search.strategy_metrics import compute_strategy_metrics
from alpha_lab.propsim.event_detail import EVENT_TYPE_PRECEDENCE
from alpha_lab.propsim.simulation import AccountSimulationEnvelope, AccountSimulationPayload
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_observation_source import (
    persisted_candidate_source,
)
from tests.agents.ifvg_search.conftest import make_resolved_trades_frame

_COST = 0.514
_EVAL_HASH = "e" * 64
_CORE_A = "a" * 64
_CORE_B = "b" * 64
_DECIDED = "2026-08-28T12:00:00+00:00"


def _decision(source, run, **overrides) -> RegimePromotionDecisionEnvelope:
    defaults = dict(
        resolved_regime_protocol_id=source.protocol.resolved_regime_protocol_id,
        role=RegimeRole.DESCRIPTIVE_ONLY,
        status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_status=RegimeStatus.PLANNED,
        previous_decision_ref=None,
        capability_assessment_ref=run.regime_capability_assessment_id,
        owner_ratification_ref=None,
        decided_at=_DECIDED,
    )
    defaults.update(overrides)
    return RegimePromotionDecisionEnvelope.from_payload(RegimePromotionDecision(**defaults))


def _trades_for(assignment: pd.DataFrame, days: tuple[str, ...]) -> pd.DataFrame:
    """Valid executed trades over the fixture's candidates: 25 in regime 0,
    25 in regime 1, 5 in regime 2 (thin), 10 unassigned candidates."""

    valid = assignment[assignment["valid"].astype(bool)]
    by_cluster = {
        int(cluster): sorted(group["candidate_id"].astype(str))
        for cluster, group in valid.groupby("canonical_reporting_cluster_id")
    }
    quotas = {0: 25, 1: 25, 2: 5}
    chosen: list[str] = []
    for cluster, quota in quotas.items():
        pool = by_cluster.get(cluster, [])
        assert len(pool) >= quota, f"fixture cluster {cluster} has only {len(pool)} OOS rows"
        chosen.extend(pool[:quota])
    unassigned = sorted(assignment.loc[~assignment["valid"].astype(bool), "candidate_id"])
    assert len(unassigned) >= 10
    chosen.extend(unassigned[:10])
    frame = make_resolved_trades_frame(days, trades_per_day=2, loss_every=3).head(len(chosen))
    assert len(frame) == len(chosen)
    frame = frame.copy()
    frame["candidate_id"] = chosen
    return frame.reset_index(drop=True)


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    root = tmp_path_factory.mktemp("stratification")
    source = persisted_candidate_source(root)
    run = execute_regime_protocol(
        root,
        protocol=source.protocol,
        observation_source=source.source_ref,
        fold_set_artifact_id=source.fold_set_envelope.fold_set_artifact_id,
        bootstrap_refits=3,
    )
    assert run.run.assessment.payload.gates_passed
    first = _decision(source, run)
    persist_regime_promotion(root, first)
    ready = _decision(
        source,
        run,
        role=RegimeRole.STRATIFICATION_ONLY,
        status=RegimeStatus.STRATIFICATION_READY,
        previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_decision_ref=first.regime_promotion_decision_id,
    )
    persist_regime_promotion(root, ready)
    owner = synthetic_owner_decision_fixture(
        root,
        protocol=source.protocol,
        assessment=run.run.assessment,
        approved_at="2026-08-25T00:00:00+00:00",
        effective_from="2026-08-25T00:00:00+00:00",
    )
    eligible = _decision(
        source,
        run,
        role=RegimeRole.FEATURE_GENERATOR,
        status=RegimeStatus.FEATURE_ELIGIBLE,
        previous_status=RegimeStatus.STRATIFICATION_READY,
        previous_decision_ref=ready.regime_promotion_decision_id,
        owner_ratification_ref=owner.owner_decision_artifact_id,
    )
    persist_regime_promotion(root, eligible, run_scope="synthetic_fixture")
    assignment_envelope, assignment = load_regime_oos_assignment_table(
        root, run.oos_assignment.regime_oos_assignment_id
    )
    trades = _trades_for(assignment, source.fixture.trading_days)
    evidence = RegimeAssignmentEvidenceRef(
        observation_granularity=assignment_envelope.payload.observation_granularity,
        regime_fit_ids=tuple(assignment_envelope.payload.regime_fit_ids),
        regime_fold_set_id=assignment_envelope.payload.regime_fold_set_id,
        fold_schedule_id=assignment_envelope.payload.fold_schedule_id,
        regime_oos_assignment_id=assignment_envelope.regime_oos_assignment_id,
    )
    return {
        "root": root,
        "source": source,
        "run": run,
        "first": first,
        "ready": ready,
        "eligible": eligible,
        "owner": owner,
        "assignment": assignment,
        "trades": trades,
        "evidence": evidence,
        "protocol_id": source.protocol.resolved_regime_protocol_id,
        "oos_id": run.oos_assignment.regime_oos_assignment_id,
        "fit_ids": tuple(sorted(run.regime_fit_ids)),
    }


# ── the status gate ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("comparison_class", list(RegimeStratificationClass))
def test_status_gate_per_class_and_status(lane, comparison_class):
    root, fit_ids = lane["root"], lane["fit_ids"]
    minimum = CLASS_MINIMUM_STATUS[comparison_class]
    # descriptive_only is below every class minimum
    with pytest.raises(RegimeStatusRefusalError, match="nothing here promotes"):
        resolve_report_gate(
            root,
            protocol_id=lane["protocol_id"],
            decision_id=lane["first"].regime_promotion_decision_id,
            owner_decision_artifact_id=None,
            comparison_class=comparison_class,
            fit_ids=fit_ids,
        )
    if minimum is RegimeStatus.STRATIFICATION_READY:
        resolved = resolve_report_gate(
            root,
            protocol_id=lane["protocol_id"],
            decision_id=lane["ready"].regime_promotion_decision_id,
            owner_decision_artifact_id=None,
            comparison_class=comparison_class,
            fit_ids=fit_ids,
        )
        assert resolved.gate.authority_source == "s10_structural"
        assert resolved.gate.status_at_report is RegimeStatus.STRATIFICATION_READY
        assert resolved.owner_decision is None
    else:
        with pytest.raises(RegimeStatusRefusalError, match="requires status feature_eligible"):
            resolve_report_gate(
                root,
                protocol_id=lane["protocol_id"],
                decision_id=lane["ready"].regime_promotion_decision_id,
                owner_decision_artifact_id=None,
                comparison_class=comparison_class,
                fit_ids=fit_ids,
            )
        with pytest.raises(RegimeStatusRefusalError, match="exact owner decision artifact id"):
            resolve_report_gate(
                root,
                protocol_id=lane["protocol_id"],
                decision_id=lane["eligible"].regime_promotion_decision_id,
                owner_decision_artifact_id=None,
                comparison_class=comparison_class,
                fit_ids=fit_ids,
            )
        with pytest.raises(RegimeStatusRefusalError, match="own ratification reference"):
            resolve_report_gate(
                root,
                protocol_id=lane["protocol_id"],
                decision_id=lane["eligible"].regime_promotion_decision_id,
                owner_decision_artifact_id="f" * 64,
                comparison_class=comparison_class,
                fit_ids=fit_ids,
            )
        # the owner half re-runs the full authorization under the RUN SCOPE:
        # the synthetic owner artifact is lawful only in the synthetic scope
        with pytest.raises(RegimeStatusRefusalError, match="synthetic_fixture run scope"):
            resolve_report_gate(
                root,
                protocol_id=lane["protocol_id"],
                decision_id=lane["eligible"].regime_promotion_decision_id,
                owner_decision_artifact_id=lane["owner"].owner_decision_artifact_id,
                comparison_class=comparison_class,
                fit_ids=fit_ids,
            )
        resolved = resolve_report_gate(
            root,
            protocol_id=lane["protocol_id"],
            decision_id=lane["eligible"].regime_promotion_decision_id,
            owner_decision_artifact_id=lane["owner"].owner_decision_artifact_id,
            comparison_class=comparison_class,
            fit_ids=fit_ids,
            run_scope="synthetic_fixture",
        )
        assert resolved.gate.authority_source == "frozen_owner_evidence"
        assert resolved.owner_decision is not None
        assert resolved.gate.role_at_report is RegimeRole.FEATURE_GENERATOR


def test_gate_refuses_other_protocols_and_uncovered_fit_sets(lane, tmp_path):
    root = lane["root"]
    with pytest.raises(RegimeStatusRefusalError, match="not covered by the decision"):
        resolve_report_gate(
            root,
            protocol_id=lane["protocol_id"],
            decision_id=lane["ready"].regime_promotion_decision_id,
            owner_decision_artifact_id=None,
            comparison_class=RegimeStratificationClass.COHORT_DESCRIPTIVE,
            fit_ids=(*lane["fit_ids"], "9" * 64),
        )
    with pytest.raises(RegimeStatusRefusalError, match="empty fit set"):
        resolve_report_gate(
            root,
            protocol_id=lane["protocol_id"],
            decision_id=lane["ready"].regime_promotion_decision_id,
            owner_decision_artifact_id=None,
            comparison_class=RegimeStratificationClass.COHORT_DESCRIPTIVE,
            fit_ids=(),
        )
    # a DIFFERENT protocol (winsorized) in the same store: the decision is not its own
    other = resolve_kmeans_protocol(
        input_feature_bundle_ref=lane["source"].protocol.payload.input_feature_bundle_ref,
        resolved_input_features=lane["source"].protocol.payload.resolved_input_features,
        winsorization_policy="clip_p01_p99_train_fitted_v1",
    )
    persist_regime_protocol(root, other)
    assert other.resolved_regime_protocol_id != lane["protocol_id"]
    with pytest.raises(RegimeStatusRefusalError, match="another regime protocol"):
        resolve_report_gate(
            root,
            protocol_id=other.resolved_regime_protocol_id,
            decision_id=lane["ready"].regime_promotion_decision_id,
            owner_decision_artifact_id=None,
            comparison_class=RegimeStratificationClass.COHORT_DESCRIPTIVE,
            fit_ids=lane["fit_ids"],
        )
    with pytest.raises(RegimeStatusRefusalError, match="not a verified entry"):
        resolve_report_gate(
            root,
            protocol_id=lane["protocol_id"],
            decision_id="c" * 64,
            owner_decision_artifact_id=None,
            comparison_class=RegimeStratificationClass.COHORT_DESCRIPTIVE,
            fit_ids=lane["fit_ids"],
        )


def test_reports_below_the_class_minimum_are_unconstructible():
    with pytest.raises(ValueError, match="nothing here promotes"):
        RegimeReportGate(
            regime_promotion_decision_id="1" * 64,
            owner_decision_artifact_id=None,
            capability_assessment_id="2" * 64,
            status_at_report=RegimeStatus.DESCRIPTIVE_ONLY,
            role_at_report=RegimeRole.DESCRIPTIVE_ONLY,
            minimum_status_required=RegimeStatus.STRATIFICATION_READY,
            authority_source="s10_structural",
        )
    with pytest.raises(ValueError, match="requires the exact owner decision"):
        RegimeReportGate(
            regime_promotion_decision_id="1" * 64,
            owner_decision_artifact_id=None,
            capability_assessment_id="2" * 64,
            status_at_report=RegimeStatus.FEATURE_ELIGIBLE,
            role_at_report=RegimeRole.FEATURE_GENERATOR,
            minimum_status_required=RegimeStatus.FEATURE_ELIGIBLE,
            authority_source="frozen_owner_evidence",
        )
    with pytest.raises(ValueError, match="canonical reporting id"):
        RegimeStratumKey(stratum="regime", canonical_reporting_cluster_id=None)


# ── the exact trade join + cohorts ───────────────────────────────────────────


def test_regime_for_trades_is_an_exact_join_with_typed_misses(lane):
    trades, assignment = lane["trades"], lane["assignment"]
    joined = regime_for_trades(trades, assignment)
    assert len(joined) == len(trades)
    assert list(joined["trade_id"]) == list(trades["trade_id"].astype(str))
    covered = joined[joined["valid"]]
    assert set(covered["canonical_reporting_cluster_id"]) == {0, 1, 2}
    typed = joined[~joined["valid"]]
    assert set(typed["missing_reason"]) == {"no_oos_assignment"}
    # a candidate absent from the artifact is a TYPED miss, never a nearest-time hit
    stranger = trades.copy()
    stranger.loc[stranger.index[0], "candidate_id"] = "not_a_candidate"
    out = regime_for_trades(stranger, assignment).set_index("trade_id")
    assert out.loc[str(trades["trade_id"].iloc[0]), "missing_reason"] == (
        "candidate_not_in_assignment"
    )
    duplicated = pd.concat([trades, trades.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="repeat a candidate_id"):
        regime_for_trades(duplicated, assignment)
    with pytest.raises(ValueError, match="lack the 'candidate_id'"):
        regime_for_trades(trades.drop(columns=["candidate_id"]), assignment)
    mask = regime_filter_mask(joined, (0,))
    assert int(mask.sum()) == 25
    cohorts = stratum_cohorts(
        protocol_id=lane["protocol_id"], fit_ids=lane["fit_ids"], cluster_ids=(0, 1, 2)
    )
    assert set(cohorts) == {
        "pooled_all",
        "pooled_regime_covered",
        "regime:0",
        "regime:1",
        "regime:2",
        "unassigned",
    }
    assert len({cohort.cohort_id for cohort in cohorts.values()}) == len(cohorts)
    regime_zero = cohorts["regime:0"].payload
    assert regime_zero.regime_filter is not None
    assert regime_zero.regime_filter.canonical_reporting_cluster_ids == (0,)
    assert regime_zero.regime_filter.regime_fit_ids == lane["fit_ids"]
    assert regime_zero.row_kind == "executed_trade"
    assert cohorts["unassigned"].payload.evidence_quality_filter == (
        UNASSIGNED_EVIDENCE_QUALITY_TOKEN,
    )


# ── cohort_descriptive ───────────────────────────────────────────────────────


def test_cohort_descriptive_pooled_equals_child_metrics_shares_sum_and_thin_typed(lane):
    trades, assignment = lane["trades"], lane["assignment"]
    predictions = pd.DataFrame(
        {
            "candidate_id": trades["candidate_id"].astype(str),
            "predicted_probability": [0.6 if r == "target" else 0.4 for r in trades["resolution"]],
            "binary_target": [1 if r == "target" else 0 for r in trades["resolution"]],
        }
    )
    result = build_cohort_descriptive_body(
        trades=trades,
        assignment=assignment,
        core_replay_id=_CORE_A,
        protocol_id=lane["protocol_id"],
        fit_ids=lane["fit_ids"],
        cost_points=_COST,
        evaluation_config_hash=_EVAL_HASH,
        pooled_predictions=predictions,
    )
    body = result.body
    rows = {row.key.label: row for row in body.strata}
    pooled = compute_strategy_metrics(
        {RecordTable.EXECUTED_TRADE: trades},
        cost_points=_COST,
        evaluation_config_hash=_EVAL_HASH,
    )
    assert rows["pooled_all"].metrics == stratum_metrics_from(pooled)
    assert rows["pooled_all"].executed_trades == len(trades) == body.trades_total
    partition = [row for row in body.strata if row.key.stratum in ("regime", "unassigned")]
    assert sum(row.share_of_trades for row in partition) == pytest.approx(1.0)
    assert rows["regime:0"].typed_state == "reported"
    assert rows["regime:1"].typed_state == "reported"
    assert rows["regime:2"].typed_state == "insufficient_regime_partition"
    assert rows["regime:2"].executed_trades == 5 < MINIMUM_TRADES_PER_REGIME_STRATUM
    assert rows["unassigned"].typed_state == "insufficient_regime_partition"
    assert body.trades_regime_covered == 55 and body.coverage_fraction == pytest.approx(55 / 65)
    assert body.unassigned_reasons == {"no_oos_assignment": 10}
    assert rows["regime:0"].pooled_model_skill is not None
    assert rows["regime:0"].pooled_model_skill["rows"] == 25.0
    assert 0.0 <= rows["regime:0"].pooled_model_skill["brier"] <= 1.0
    # cohort ids are minted per stratum from the consumed RegimeFilterRef
    cohort_ids = {row.key.label: row.cohort_id for row in body.strata}
    assert all(cohort_ids.values())
    assert len(set(cohort_ids.values())) == len(cohort_ids)
    assert result.detail["cohorts"]["regime:0"]["payload"]["regime_filter"][
        "canonical_reporting_cluster_ids"
    ] == [0]
    # label spelling never changes a stratum: renaming the trade ids leaves the
    # regime strata (and their metrics) identical
    renamed = trades.copy()
    renamed["trade_id"] = [f"T-{value}" for value in trades["trade_id"]]
    renamed["decision_id"] = [f"D-{value}" for value in trades["decision_id"]]
    again = build_cohort_descriptive_body(
        trades=renamed,
        assignment=assignment,
        core_replay_id=_CORE_A,
        protocol_id=lane["protocol_id"],
        fit_ids=lane["fit_ids"],
        cost_points=_COST,
        evaluation_config_hash=_EVAL_HASH,
    )
    for label in ("regime:0", "regime:1"):
        assert {r.key.label: r for r in again.body.strata}[label].metrics == rows[label].metrics
    with pytest.raises(ValueError, match="at least one executed trade"):
        build_cohort_descriptive_body(
            trades=trades.head(0),
            assignment=assignment,
            core_replay_id=_CORE_A,
            protocol_id=lane["protocol_id"],
            fit_ids=lane["fit_ids"],
            cost_points=_COST,
            evaluation_config_hash=_EVAL_HASH,
        )


# ── stratified_prop ──────────────────────────────────────────────────────────


def _simulation(root, *, mode: str, seed: int, events: list[dict] | None):
    payload = AccountSimulationPayload(
        core_replay_id=_CORE_A,
        gross_trade_stream_hash="1" * 64,
        costed_evaluation_id="2" * 64,
        trade_path_bundle_id="3" * 64,
        trade_path_bundle_manifest_sha256="4" * 64,
        path_capability_report_id="5" * 64,
        account_policy_set_id="6" * 64,
        simulation_mode=mode,
        intrabar_scenario_policy_id=None,
        bootstrap_protocol_id="day_block_bootstrap_h90_v1" if mode.startswith("day") else None,
        stress_scenario_id=None,
        seed=seed,
        n_paths=1,
    )
    envelope = AccountSimulationEnvelope.from_payload(payload)
    extra = {}
    if events is not None:
        extra["account_events.json"] = (json.dumps(events, sort_keys=True) + "\n").encode("utf-8")
    save_or_reuse_envelope(root, "account_simulations", envelope, extra_files=extra or None)
    return envelope.account_simulation_id


def _event(ordinal: int, ts: str, event_type: str, *, trade: str | None, **payload) -> dict:
    return {
        "event_id": hashlib.sha256(f"{ordinal}:{ts}:{event_type}:{trade}".encode()).hexdigest(),
        "event_ts_utc": ts,
        "trading_day": ts[:10],
        "event_ordinal": ordinal,
        "path_instance_id": "path-00000-x",
        "account_id": "acct-1",
        "account_ordinal": 0,
        "firm_contract_id": "7" * 64,
        "account_phase": "funded",
        "source_trade_id": trade,
        "source_decision_id": None,
        "source_candidate_id": None,
        "source_setup_id": None,
        "source_path_event_id": None,
        "event_type": event_type,
        "event_order_policy_id": "test",
        "payload": payload,
    }


def test_stratified_prop_attributes_by_source_trade_never_by_synthetic_clock(lane):
    root, trades, assignment = lane["root"], lane["trades"], lane["assignment"]
    trade_regimes = regime_for_trades(trades, assignment)
    by_trade = trade_regimes.set_index("trade_id")
    in_regime0 = by_trade[by_trade["canonical_reporting_cluster_id"] == 0].index[:3].tolist()
    in_regime1 = by_trade[by_trade["canonical_reporting_cluster_id"] == 1].index[:2].tolist()
    unassigned_trade = by_trade[~by_trade["valid"]].index[0]
    eq = "equity_update"
    historical_events = [
        _event(0, "2026-01-08T14:05:00+00:00", eq, trade=in_regime0[0], realized_delta=120.0),
        _event(1, "2026-01-08T14:25:00+00:00", eq, trade=in_regime0[1], realized_delta=-60.0),
        _event(2, "2026-01-09T14:05:00+00:00", eq, trade=in_regime1[0], realized_delta=80.0),
        _event(3, "2026-01-09T15:00:00+00:00", "payout", trade=None, trader_amount=500.0),
        _event(4, "2026-01-12T14:05:00+00:00", "fee", trade=None, fee_kind="reset", amount=90.0),
        _event(5, "2026-01-12T14:25:00+00:00", eq, trade=unassigned_trade, realized_delta=10.0),
        _event(6, "2026-01-12T14:45:00+00:00", eq, trade="unknown-trade", realized_delta=5.0),
    ]
    historical_id = _simulation(
        root, mode="historical_closed_trade", seed=1, events=historical_events
    )
    # bootstrap paths advance a SYNTHETIC 2020 clock: only the source trade attributes
    bootstrap_events = [
        _event(0, "2020-01-06T14:05:00+00:00", eq, trade=in_regime0[2], realized_delta=40.0),
        _event(1, "2020-01-06T15:00:00+00:00", "payout", trade=None, trader_amount=300.0),
        _event(2, "2020-01-07T14:05:00+00:00", eq, trade=in_regime1[1], realized_delta=-20.0),
    ]
    bootstrap_id = _simulation(root, mode="day_block_bootstrap", seed=2, events=None)
    none_v0_id = _simulation(root, mode="day_block_bootstrap", seed=3, events=None)

    def loader(store_root, simulation_id):
        if simulation_id == bootstrap_id:
            frame = event_detail_frame(bootstrap_events, clock_policy_id="synthetic_path_clock_v1")
            return iter((frame,))
        if simulation_id == none_v0_id:
            return None
        from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_prop import (
            load_account_event_detail_from_json,
        )

        return load_account_event_detail_from_json(store_root, simulation_id)

    consulted: list[str] = []

    def panel_assigner(timestamps: pd.Series) -> pd.Series:
        consulted.extend(timestamps.tolist())
        # a stand-in for the panel PIT rule: the day 2026-01-09 sits in regime 1
        return pd.Series(
            [1 if str(ts).startswith("2026-01-09") else None for ts in timestamps],
            index=timestamps.index,
            dtype="object",
        )

    simulations = {
        historical_id: ("firm_a", "historical_closed_trade"),
        bootstrap_id: ("firm_a", "day_block_bootstrap"),
        none_v0_id: ("firm_b", "day_block_bootstrap"),
    }
    result = build_stratified_prop_body(
        root=root,
        core_replay_id=_CORE_A,
        simulations=simulations,
        trade_regimes=trade_regimes,
        evidence=lane["evidence"],
        loader=loader,
        panel_assigner=panel_assigner,
    )
    body, detail = result.body, result.detail
    assert body.probabilities_reestimated is False
    assert body.attribution_policy_id == "source_trade_then_pit_v1"
    assert body.evidence_not_persisted == (none_v0_id,)
    # the panel PIT seam was consulted ONLY for historical no-trade events —
    # the bootstrap payout at a 2020 instant never reached it
    assert sorted(consulted) == ["2026-01-09T15:00:00+00:00", "2026-01-12T14:05:00+00:00"]
    assert body.events_total == 10
    # attributed: 3 historical trade events + the PIT payout + 2 bootstrap trade events
    assert body.events_attributed == 6
    assert body.partial_coverage is True
    assert body.unattributable_by_reason == {
        "no_source_trade_synthetic_clock": 1,
        "panel_pit_unassigned": 1,
        "source_trade_not_in_regime_evidence": 1,
        "source_trade_unassigned": 1,
    }
    rows = {(row.account_simulation_id, row.key.label): row for row in body.strata}
    hist0 = rows[(historical_id, "regime:0")]
    assert hist0.event_counts_by_type == {"equity_update": 2}
    assert hist0.realized_pnl_sum == pytest.approx(60.0)
    hist1 = rows[(historical_id, "regime:1")]
    assert hist1.event_counts_by_type == {"equity_update": 1, "payout": 1}
    assert hist1.payout_trader_amount_sum == pytest.approx(500.0)
    assert rows[(historical_id, "unassigned")].fee_amount_sum == pytest.approx(90.0)
    boot = rows[(bootstrap_id, "regime:0")]
    assert boot.realized_pnl_sum == pytest.approx(40.0)
    assert rows[(bootstrap_id, "unassigned")].event_counts_by_type == {"payout": 1}
    # ── D15: the report-local event-regime summary replaces per-event JSON ──
    facts = detail["simulations"][historical_id]
    assert "events" not in facts and "events" not in detail["simulations"][bootstrap_id]
    assert facts["events_total"] == 7 and facts["events_attributed"] == 4
    assert facts["unattributable_by_reason"] == {
        "panel_pit_unassigned": 1,
        "source_trade_not_in_regime_evidence": 1,
        "source_trade_unassigned": 1,
    }
    assert facts["clock_policy_ids"] == [HISTORICAL_CLOCK_POLICY_ID]
    assert detail["summary"]["sidecar"] == ACCOUNT_EVENT_REGIME_SUMMARY_SIDECAR
    assert detail["summary"]["schema_hash"] == ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_HASH
    assert detail["summary"]["budget"]["budget_id"] == (
        ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1.budget_id
    )
    summary = result.summary
    assert summary.schema.names == ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA.names
    assert arrow_schema_hash(summary.schema) == ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_HASH
    # 6 historical + 3 bootstrap (simulation × path × stratum/reason × type) rows for 10 events
    assert summary.num_rows == body.summary_rows == 9 == detail["summary"]["rows"]
    assert body.summary_budget == ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1
    frame = summary.to_pandas()
    assert set(frame["core_replay_id"]) == {_CORE_A}
    assert set(frame["regime_oos_assignment_id"]) == {lane["oos_id"]}
    assert set(frame["fold_schedule_id"]) == {lane["evidence"].fold_schedule_id}
    assert set(frame["account_simulation_id"]) == {historical_id, bootstrap_id}

    def _summary_row(simulation_id: str, stratum: str, event_type: str) -> dict:
        selected = frame[
            (frame["account_simulation_id"] == simulation_id)
            & (frame["regime_stratum"] == stratum)
            & (frame["event_type"] == event_type)
        ]
        assert len(selected) == 1, (simulation_id[:12], stratum, event_type, len(selected))
        return selected.iloc[0].to_dict()

    hist_eq0 = _summary_row(historical_id, "regime:0", "equity_update")
    assert int(hist_eq0["event_count"]) == 2 and hist_eq0["amount_sum"] == pytest.approx(60.0)
    assert int(hist_eq0["first_event_ordinal"]) == 0 and int(hist_eq0["last_event_ordinal"]) == 1
    assert hist_eq0["unattributable_reason"] is None
    assert int(hist_eq0["canonical_reporting_cluster_id"]) == 0
    assert int(hist_eq0["event_precedence"]) == EVENT_TYPE_PRECEDENCE["equity_update"]
    assert _summary_row(historical_id, "regime:1", "payout")["amount_sum"] == pytest.approx(500.0)
    unassigned_hist = frame[
        (frame["account_simulation_id"] == historical_id)
        & (frame["regime_stratum"] == "unassigned")
    ]
    assert sorted(unassigned_hist["unattributable_reason"]) == [
        "panel_pit_unassigned",
        "source_trade_not_in_regime_evidence",
        "source_trade_unassigned",
    ]
    assert unassigned_hist["canonical_reporting_cluster_id"].isna().all()
    boot_rows = frame[frame["account_simulation_id"] == bootstrap_id]
    assert boot_rows["clock_policy_id"].unique().tolist() == [SYNTHETIC_CLOCK_POLICY_ID]
    assert list(boot_rows["unattributable_reason"].dropna()) == ["no_source_trade_synthetic_clock"]
    # deterministic order: simulation, path, regime (unassigned last), reason, type
    ordered = sorted(
        frame.to_dict(orient="records"),
        key=lambda r: (
            r["account_simulation_id"],
            r["path_instance_id"],
            (1, 0)
            if pd.isna(r["canonical_reporting_cluster_id"])
            else (0, int(r["canonical_reporting_cluster_id"])),
            r["unattributable_reason"] or "",
            r["event_type"],
        ),
    )
    assert [r["event_type"] for r in ordered] == list(frame["event_type"])
    assert list(SUMMARY_ROW_KEYS) == detail["summary"]["row_keys"]
    # the serialized bytes round-trip and are byte-deterministic
    parsed = read_account_event_regime_summary(result.summary_bytes)
    assert parsed.to_pandas().equals(frame)
    again = build_stratified_prop_body(
        root=root,
        core_replay_id=_CORE_A,
        simulations=simulations,
        trade_regimes=trade_regimes,
        evidence=lane["evidence"],
        loader=loader,
        panel_assigner=panel_assigner,
    )
    assert again.summary_bytes == result.summary_bytes and again.body == body
    with pytest.raises(ValueError, match="is a historical_closed_trade run, not"):
        build_stratified_prop_body(
            root=root,
            core_replay_id=_CORE_A,
            simulations={historical_id: ("firm_a", "stress")},
            trade_regimes=trade_regimes,
            evidence=lane["evidence"],
            loader=loader,
        )


def test_event_detail_frame_uses_d15_precedence_and_the_events_own_trading_day():
    """F9: the projection reads the D15 ``EVENT_TYPE_PRECEDENCE`` and the
    record's ``trading_day`` (never a UTC date prefix); a historical record
    without a trading day refuses; an unregistered event type refuses."""

    late_evening = _event(0, "2026-01-09T00:30:00+00:00", "fee", trade=None, amount=1.0)
    late_evening["trading_day"] = "2026-01-08"  # ET evening of the prior calendar date
    frame = event_detail_frame([late_evening], clock_policy_id=HISTORICAL_CLOCK_POLICY_ID)
    assert frame.loc[0, "trading_day"] == "2026-01-08" != "2026-01-09"
    assert int(frame.loc[0, "event_precedence"]) == EVENT_TYPE_PRECEDENCE["fee"] == 0
    for event_type, rank in EVENT_TYPE_PRECEDENCE.items():
        record = _event(1, "2026-01-08T14:05:00+00:00", event_type, trade=None)
        projected = event_detail_frame([record], clock_policy_id=SYNTHETIC_CLOCK_POLICY_ID)
        assert int(projected.loc[0, "event_precedence"]) == rank
    missing = _event(2, "2026-01-08T14:05:00+00:00", "payout", trade=None, trader_amount=1.0)
    del missing["trading_day"]
    with pytest.raises(ValueError, match="carries no trading_day"):
        event_detail_frame([missing], clock_policy_id=HISTORICAL_CLOCK_POLICY_ID)
    synthetic = event_detail_frame([missing], clock_policy_id=SYNTHETIC_CLOCK_POLICY_ID)
    assert synthetic.loc[0, "trading_day"] is None
    unknown = _event(3, "2026-01-08T14:05:00+00:00", "not_an_event", trade=None)
    with pytest.raises(ValueError, match="not a registered D15 event type"):
        event_detail_frame([unknown], clock_policy_id=SYNTHETIC_CLOCK_POLICY_ID)
    assert set(frame.columns) == set(EVENT_DETAIL_COLUMNS)


def test_event_regime_summary_budget_refuses_before_publication(lane):
    """A summary over the registered row or byte budget refuses (typed) and
    nothing is published; the budget is stamped as a storage budget."""

    root, trades, assignment = lane["root"], lane["trades"], lane["assignment"]
    trade_regimes = regime_for_trades(trades, assignment)
    first_trade = str(trades["trade_id"].iloc[0])
    events = [
        _event(
            0, "2026-01-08T14:05:00+00:00", "equity_update", trade=first_trade, realized_delta=1.0
        ),
        _event(1, "2026-01-08T14:06:00+00:00", "payout", trade=None, trader_amount=2.0),
        _event(2, "2026-01-08T14:07:00+00:00", "fee", trade=None, amount=3.0),
    ]
    simulation_id = _simulation(root, mode="historical_closed_trade", seed=21, events=events)
    simulations = {simulation_id: ("firm_a", "historical_closed_trade")}
    tiny_rows = EventRegimeSummaryBudget(
        budget_id="test_tiny_rows", max_summary_rows=2, max_published_bytes=10_000_000
    )
    with pytest.raises(EventRegimeSummaryBudgetError, match="row budget"):
        build_stratified_prop_body(
            root=root,
            core_replay_id=_CORE_A,
            simulations=simulations,
            trade_regimes=trade_regimes,
            evidence=lane["evidence"],
            budget=tiny_rows,
        )
    tiny_bytes = EventRegimeSummaryBudget(
        budget_id="test_tiny_bytes", max_summary_rows=1_000, max_published_bytes=16
    )
    with pytest.raises(EventRegimeSummaryBudgetError, match="byte budget"):
        build_stratified_prop_body(
            root=root,
            core_replay_id=_CORE_A,
            simulations=simulations,
            trade_regimes=trade_regimes,
            evidence=lane["evidence"],
            budget=tiny_bytes,
        )
    # through the service: the refusal propagates typed and NO report of the
    # class is published (the store gains nothing)
    store_dir = root / "regime_stratified_reports"
    before = sorted(p.name for p in store_dir.iterdir()) if store_dir.exists() else []
    with pytest.raises(EventRegimeSummaryBudgetError):
        build_regime_stratified_reports(
            StratificationInputs(
                root=root,
                protocol_id=lane["protocol_id"],
                decision_id=lane["ready"].regime_promotion_decision_id,
                owner_decision_artifact_id=None,
                regime_oos_assignment_id=lane["oos_id"],
                requested_classes=(RegimeStratificationClass.STRATIFIED_PROP,),
                children={
                    _CORE_A: ChildStratificationInputs(
                        core_replay_id=_CORE_A,
                        trades=trades,
                        cost_points=_COST,
                        evaluation_config_hash=_EVAL_HASH,
                        costed_evaluation_id="2" * 64,
                        account_simulations=simulations,
                    )
                },
                summary_budget=tiny_rows,
            )
        )
    after = sorted(p.name for p in store_dir.iterdir()) if store_dir.exists() else []
    assert after == before
    assert STRATIFICATION_REGISTERED_BUDGETS["account_event_regime_summary_max_rows"]["value"] == (
        ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1.max_summary_rows
    )
    assert all(
        entry["stamp"] == "registered_storage_budget"
        for entry in STRATIFICATION_REGISTERED_BUDGETS.values()
    )


# ── stratified_frontier ──────────────────────────────────────────────────────


def _frontier(root, feasible: dict[str, dict[str, float]]) -> SearchFrontierEnvelope:
    result = build_frontier(
        feasible,
        objectives=(ObjectiveSpec(metric="net_expectancy_r", direction="maximize"),),
        lexicographic_tie_breaks=("core_replay_id",),
    )
    envelope = SearchFrontierEnvelope.from_payload(
        SearchFrontierPayload(search_id="d" * 64, frontier=result)
    )
    save_or_reuse_envelope(root, "frontiers", envelope)
    return envelope


def test_stratified_frontier_reads_on_frontier_from_the_pooled_frontier_only(lane):
    root, trades, assignment = lane["root"], lane["trades"], lane["assignment"]
    body_a = build_cohort_descriptive_body(
        trades=trades,
        assignment=assignment,
        core_replay_id=_CORE_A,
        protocol_id=lane["protocol_id"],
        fit_ids=lane["fit_ids"],
        cost_points=_COST,
        evaluation_config_hash=_EVAL_HASH,
    ).body
    body_b = build_cohort_descriptive_body(
        trades=trades,
        assignment=assignment,
        core_replay_id=_CORE_B,
        protocol_id=lane["protocol_id"],
        fit_ids=lane["fit_ids"],
        cost_points=_COST,
        evaluation_config_hash=_EVAL_HASH,
    ).body
    frontier = _frontier(
        root, {_CORE_A: {"net_expectancy_r": 0.9}, _CORE_B: {"net_expectancy_r": 0.1}}
    )
    body = build_stratified_frontier_body(
        frontier=frontier,
        strategy_bodies={_CORE_A: body_a, _CORE_B: body_b},
        objective_metrics=("net_expectancy_r", "profit_factor"),
    )
    assert body.frontier_role == "descriptive_view_never_selection_input"
    assert body.children == (_CORE_A, _CORE_B)
    assert dict(body.on_frontier) == {_CORE_A: True, _CORE_B: False}
    labels = {(cell.core_replay_id, cell.key.label) for cell in body.cells}
    assert (_CORE_B, "regime:0") in labels and (_CORE_A, "pooled_all") in labels
    cell = next(c for c in body.cells if c.core_replay_id == _CORE_A and c.key.label == "regime:0")
    assert cell.metric_values["net_expectancy_r"] is not None
    # a regime stratum that looks better than the pooled frontier says never
    # moves a child onto the frontier: the same strata under a frontier that
    # excludes B still read B as off-frontier
    only_a = _frontier(root, {_CORE_A: {"net_expectancy_r": 0.9}})
    partial = build_stratified_frontier_body(
        frontier=only_a,
        strategy_bodies={_CORE_A: body_a, _CORE_B: body_b},
        objective_metrics=("net_expectancy_r",),
    )
    assert partial.children == (_CORE_A,) and _CORE_B not in partial.on_frontier
    with pytest.raises(ValueError, match="at least one objective metric"):
        build_stratified_frontier_body(
            frontier=frontier, strategy_bodies={_CORE_A: body_a}, objective_metrics=()
        )


# ── the service ──────────────────────────────────────────────────────────────


def test_service_persists_descriptive_reports_and_records_modeled_delivery(lane, monkeypatch):
    root, trades = lane["root"], lane["trades"]
    frontier = _frontier(root, {_CORE_A: {"net_expectancy_r": 0.5}})
    historical_id = _simulation(
        root,
        mode="historical_closed_trade",
        seed=11,
        events=[
            _event(
                0,
                "2026-01-08T14:05:00+00:00",
                "equity_update",
                trade=str(trades["trade_id"].iloc[0]),
                realized_delta=12.0,
            )
        ],
    )
    inputs = StratificationInputs(
        root=root,
        protocol_id=lane["protocol_id"],
        decision_id=lane["ready"].regime_promotion_decision_id,
        owner_decision_artifact_id=None,
        regime_oos_assignment_id=lane["oos_id"],
        requested_classes=tuple(RegimeStratificationClass),
        children={
            _CORE_A: ChildStratificationInputs(
                core_replay_id=_CORE_A,
                trades=trades,
                cost_points=_COST,
                evaluation_config_hash=_EVAL_HASH,
                costed_evaluation_id="2" * 64,
                account_simulations={historical_id: ("firm_a", "historical_closed_trade")},
            )
        },
        frontier_id=frontier.frontier_id,
    )
    outcome = build_regime_stratified_reports(inputs)
    assert set(outcome.reports_by_class) == {
        "cohort_descriptive",
        "stratified_frontier",
        "stratified_prop",
    }
    # no S09c delivery recorded → the modeled classes are TYPED refusals
    assert set(outcome.refusals) == {"feature_only", "cohort_model"}
    assert all("nothing here promotes" in text for text in outcome.refusals.values())
    assert all("this run recorded none" in text for text in outcome.refusals.values())
    assert outcome.delivered_by == {}
    assert len(outcome.report_ids) == 3
    for report_id in outcome.report_ids:
        envelope = load_regime_stratified_report(root, report_id)
        assert envelope.payload.gate.regime_promotion_decision_id == (
            lane["ready"].regime_promotion_decision_id
        )
        assert envelope.payload.assignment_evidence.regime_oos_assignment_id == lane["oos_id"]
        assert envelope.payload.counterfactual_claim == "none"
        assert envelope.payload.interpretation == "descriptive"
        detail = load_regime_stratified_report_detail(root, report_id)
        assert isinstance(detail, dict)
        if envelope.payload.comparison_class is RegimeStratificationClass.STRATIFIED_PROP:
            # the D15 summary sidecar is bound by the envelope and verified on load
            table = load_regime_stratified_report_summary(root, report_id)
            assert table.num_rows == envelope.payload.body.summary_rows == 1
            assert envelope.account_event_regime_summary_rows == 1
            assert (
                detail["summary"]["rows"] == 1
                and "events" not in detail["simulations"][historical_id]
            )
            sidecar = (
                envelope_destination(root, "regime_stratified_reports", report_id)
                / ACCOUNT_EVENT_REGIME_SUMMARY_SIDECAR
            )
            original = sidecar.read_bytes()
            try:
                sidecar.write_bytes(original[:-4] + b"\x00" * 4)
                with pytest.raises((SearchStoreError, ValueError)):
                    load_regime_stratified_report_summary(root, report_id)
            finally:
                sidecar.write_bytes(original)
            # the envelope binds a summary: publishing without it refuses
            with pytest.raises(ValueError, match="exactly the event-regime summary"):
                persist_regime_stratified_report(root, envelope, _detail_bytes(root, report_id))
        else:
            assert envelope.account_event_regime_summary_sha256 is None
            with pytest.raises(ValueError, match="carries no account_event_regime_summary"):
                load_regime_stratified_report_summary(root, report_id)
    # a second build REUSES every report (deterministic identities)
    again = build_regime_stratified_reports(inputs)
    assert again.report_ids == outcome.report_ids
    # ── F6: a delivered modeled class is recorded, verified by exact-id reload ──
    delivered_id = "d" * 64
    seen: list[tuple[str, str]] = []

    def _fake_load(store_root, study_id):
        seen.append((str(store_root), study_id))
        if study_id != delivered_id:
            raise SearchStoreError("missing search-store entry")
        return object()

    monkeypatch.setattr(
        "alpha_lab.agents.data_infra.ifvg.ml.regime_controlled_study.load_regime_controlled_study",
        _fake_load,
    )
    delivered = build_regime_stratified_reports(
        StratificationInputs(
            root=root,
            protocol_id=lane["protocol_id"],
            decision_id=lane["ready"].regime_promotion_decision_id,
            owner_decision_artifact_id=None,
            regime_oos_assignment_id=lane["oos_id"],
            requested_classes=tuple(RegimeStratificationClass),
            children=inputs.children,
            frontier_id=frontier.frontier_id,
            delivered_by={"feature_only": delivered_id},
        )
    )
    assert delivered.delivered_by == {"feature_only": delivered_id}
    assert "feature_only" not in delivered.refusals and "cohort_model" in delivered.refusals
    assert delivered.report_ids == outcome.report_ids
    assert seen == [(str(root), delivered_id)]
    # an unverifiable delivered id is never recorded (refused at reload); a
    # non-64-hex string is refused before any load
    with pytest.raises(SearchStoreError):
        build_regime_stratified_reports(
            StratificationInputs(
                root=root,
                protocol_id=lane["protocol_id"],
                decision_id=lane["ready"].regime_promotion_decision_id,
                owner_decision_artifact_id=None,
                regime_oos_assignment_id=lane["oos_id"],
                requested_classes=(RegimeStratificationClass.FEATURE_ONLY,),
                children={},
                delivered_by={"feature_only": "e" * 64},
            )
        )
    with pytest.raises(ValueError, match="64-hex persisted study id"):
        build_regime_stratified_reports(
            StratificationInputs(
                root=root,
                protocol_id=lane["protocol_id"],
                decision_id=lane["ready"].regime_promotion_decision_id,
                owner_decision_artifact_id=None,
                regime_oos_assignment_id=lane["oos_id"],
                requested_classes=(RegimeStratificationClass.FEATURE_ONLY,),
                children={},
                delivered_by={"feature_only": "not-an-id"},
            )
        )
    # the descriptive-only decision refuses every class (recorded, never raised)
    refused = build_regime_stratified_reports(
        StratificationInputs(
            root=root,
            protocol_id=lane["protocol_id"],
            decision_id=lane["first"].regime_promotion_decision_id,
            owner_decision_artifact_id=None,
            regime_oos_assignment_id=lane["oos_id"],
            requested_classes=tuple(RegimeStratificationClass),
            children=inputs.children,
            frontier_id=frontier.frontier_id,
        )
    )
    assert refused.report_ids == ()
    assert set(refused.refusals) == {cls.value for cls in RegimeStratificationClass}
    # the frontier class without a frontier id is a typed refusal
    no_frontier = build_regime_stratified_reports(
        StratificationInputs(
            root=root,
            protocol_id=lane["protocol_id"],
            decision_id=lane["ready"].regime_promotion_decision_id,
            owner_decision_artifact_id=None,
            regime_oos_assignment_id=lane["oos_id"],
            requested_classes=(RegimeStratificationClass.STRATIFIED_FRONTIER,),
            children=inputs.children,
            frontier_id=None,
        )
    )
    assert "stratified_frontier" in no_frontier.refusals and no_frontier.report_ids == ()


def _detail_bytes(root, report_id: str) -> bytes:
    from alpha_lab.agents.data_infra.ifvg.search.store import load_sidecar_bytes

    return load_sidecar_bytes(
        root, "regime_stratified_reports", report_id, "stratified_report_detail.json"
    )


def test_reports_are_built_from_persisted_artifacts_only(lane, tmp_path):
    """A tampered OOS assignment artifact fails at LOAD; an unknown id never
    fabricates a report; the identity pair is registered for the audit."""

    root = lane["root"]
    assert any(pair.name == "RegimeStratifiedReport" for pair in registered_identity_pairs())
    with pytest.raises(SearchStoreError):
        build_regime_stratified_reports(
            StratificationInputs(
                root=root,
                protocol_id=lane["protocol_id"],
                decision_id=lane["ready"].regime_promotion_decision_id,
                owner_decision_artifact_id=None,
                regime_oos_assignment_id="9" * 64,
                requested_classes=(RegimeStratificationClass.COHORT_DESCRIPTIVE,),
                children={},
            )
        )
    directory = envelope_destination(root, "regime_oos_assignments", lane["oos_id"])
    sidecar = directory / "regime_oos_assignments.arrow"
    original = sidecar.read_bytes()
    try:
        sidecar.write_bytes(original[:-8] + b"\x00" * 8)
        with pytest.raises((SearchStoreError, ValueError)):
            build_regime_stratified_reports(
                StratificationInputs(
                    root=root,
                    protocol_id=lane["protocol_id"],
                    decision_id=lane["ready"].regime_promotion_decision_id,
                    owner_decision_artifact_id=None,
                    regime_oos_assignment_id=lane["oos_id"],
                    requested_classes=(RegimeStratificationClass.COHORT_DESCRIPTIVE,),
                    children={},
                )
            )
    finally:
        sidecar.write_bytes(original)
