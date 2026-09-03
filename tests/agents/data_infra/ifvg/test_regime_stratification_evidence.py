"""R6.1-FIX workstreams A + B on the stratification lane (plan §3.1, §3.5,
§3.9; findings F-01, F-09, F-10B).

* ``RegimeAssignmentEvidenceRef`` binds the exact assignment table hash and
  schema hash of the descriptive artifact a report stratified over, filled
  from the VERIFIED envelope by the service;
* thin regimes count in the raw net-R concentration accounting while the
  reportability floor keeps governing interval/reportability metrics;
* every join, stratum, computation and the binding hash use the validated,
  normalized executed-trade frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.ml.regime_assignment_sources import (
    load_regime_oos_assignment_table,
    regime_for_trades,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import ObservationGranularity
from alpha_lab.agents.data_infra.ifvg.ml.regime_executor import execute_regime_protocol
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import persist_regime_promotion
from alpha_lab.agents.data_infra.ifvg.ml.regime_stratification_service import (
    ChildStratificationInputs,
    StratificationInputs,
    build_regime_stratified_reports,
    load_regime_stratified_report,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_contracts import (
    MINIMUM_TRADES_PER_REGIME_STRATUM,
    NET_R_ACCOUNTING_FORMULA_VERSION,
    CohortDescriptiveBody,
    RegimeAssignmentEvidenceRef,
    RegimeNetRAccounting,
    RegimeStratificationClass,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_strategy import (
    build_cohort_descriptive_body,
    executed_trade_table_sha256,
    normalized_executed_trades,
    regime_net_r_accounting,
)
from alpha_lab.agents.data_infra.ifvg.search.executed_trade_table import (
    build_executed_trade_table,
    load_executed_trade_table,
    save_executed_trade_table,
)
from alpha_lab.agents.data_infra.ifvg.search.store import SearchStoreError
from alpha_lab.agents.data_infra.ifvg.search.strategy_metrics import (
    compute_strategy_metrics,
    per_trade_net_r,
)
from alpha_lab.agents.data_infra.ifvg.trade_stats import _validate_and_normalize_executed_trades
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_observation_source import (
    persisted_candidate_source,
)
from tests.agents.data_infra.ifvg.test_regime_stratification import (
    _CORE_A,
    _CORE_B,
    _COST,
    _EVAL_HASH,
    _decision,
    _trades_for,
)
from tests.agents.ifvg_search.conftest import make_resolved_trades_frame


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import RegimeRole, RegimeStatus

    root = tmp_path_factory.mktemp("stratification_evidence")
    source = persisted_candidate_source(root)
    run = execute_regime_protocol(
        root,
        protocol=source.protocol,
        observation_source=source.source_ref,
        fold_set_artifact_id=source.fold_set_envelope.fold_set_artifact_id,
        bootstrap_refits=3,
    )
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
    envelope, assignment = load_regime_oos_assignment_table(
        root, run.oos_assignment.regime_oos_assignment_id
    )
    return {
        "root": root,
        "source": source,
        "run": run,
        "ready": ready,
        "envelope": envelope,
        "assignment": assignment,
        "trades": _trades_for(assignment, source.fixture.trading_days),
        "protocol_id": source.protocol.resolved_regime_protocol_id,
        "oos_id": run.oos_assignment.regime_oos_assignment_id,
        "fit_ids": tuple(sorted(run.regime_fit_ids)),
    }


# ── F-01: the evidence ref pins the exact table the report stratified over ──


def test_evidence_ref_binds_assignment_table_and_schema_hashes(lane):
    envelope = lane["envelope"]
    ref = RegimeAssignmentEvidenceRef(
        observation_granularity=envelope.payload.observation_granularity,
        regime_fit_ids=tuple(envelope.payload.regime_fit_ids),
        regime_fold_set_id=envelope.payload.regime_fold_set_id,
        fold_schedule_id=envelope.payload.fold_schedule_id,
        regime_oos_assignment_id=envelope.regime_oos_assignment_id,
        assignment_table_sha256=envelope.assignment_table_sha256,
        assignment_schema_hash=envelope.payload.assignment_schema_hash,
    )
    assert ref.assignment_table_sha256 == envelope.assignment_table_sha256
    with pytest.raises(ValueError):
        RegimeAssignmentEvidenceRef(
            observation_granularity=ObservationGranularity.CANDIDATE_STAGE_ROW,
            regime_fit_ids=("3" * 64,),
            regime_fold_set_id="4" * 64,
            fold_schedule_id="5" * 64,
            regime_oos_assignment_id="6" * 64,
        )
    inputs = StratificationInputs(
        root=lane["root"],
        protocol_id=lane["protocol_id"],
        decision_id=lane["ready"].regime_promotion_decision_id,
        owner_decision_artifact_id=None,
        regime_oos_assignment_id=lane["oos_id"],
        requested_classes=(RegimeStratificationClass.COHORT_DESCRIPTIVE,),
        children={
            _CORE_A: ChildStratificationInputs(
                core_replay_id=_CORE_A,
                trades=lane["trades"],
                cost_points=_COST,
                evaluation_config_hash=_EVAL_HASH,
                costed_evaluation_id="2" * 64,
                executed_trade_table_id=_persist_table(
                    lane["root"], _CORE_A, lane["trades"]
                ).executed_trade_table_id,
            )
        },
    )
    outcome = build_regime_stratified_reports(inputs)
    assert outcome.report_ids
    report = load_regime_stratified_report(lane["root"], outcome.report_ids[0])
    evidence = report.payload.assignment_evidence
    assert evidence.assignment_table_sha256 == envelope.assignment_table_sha256
    assert evidence.assignment_schema_hash == envelope.payload.assignment_schema_hash
    assert evidence.regime_fit_ids == tuple(envelope.payload.regime_fit_ids)


# ── F-09 / F-10B: raw accounting over every assigned trade; normalized frame ─


def test_thin_regimes_count_in_concentration_but_not_in_reportability(lane):
    trades, assignment = lane["trades"], lane["assignment"]
    result = build_cohort_descriptive_body(
        trades=trades,
        assignment=assignment,
        core_replay_id=_CORE_A,
        protocol_id=lane["protocol_id"],
        fit_ids=lane["fit_ids"],
        cost_points=_COST,
        evaluation_config_hash=_EVAL_HASH,
    )
    body = result.body
    rows = {row.key.label: row for row in body.strata}
    # the thin regime stays typed for reportability …
    assert rows["regime:2"].typed_state == "insufficient_regime_partition"
    assert rows["regime:2"].executed_trades == 5 < MINIMUM_TRADES_PER_REGIME_STRATUM
    accounting = body.net_r_accounting
    assert isinstance(accounting, RegimeNetRAccounting)
    assert accounting.formula_version == NET_R_ACCOUNTING_FORMULA_VERSION
    # … but it IS in the raw accounting: every valid assigned trade contributes
    assert set(accounting.net_r_by_regime) == {0, 1, 2}
    assert accounting.trade_count_by_regime == {0: 25, 1: 25, 2: 5}
    assert accounting.unassigned_trade_count == 10
    # the values reproduce from the per-trade net R of the normalized frame
    net = per_trade_net_r(trades, cost_points=_COST)
    joined = regime_for_trades(trades, assignment).set_index("trade_id")
    expected = {}
    unassigned_net = 0.0
    for trade_id, value in net.items():
        row = joined.loc[trade_id]
        if bool(row["valid"]):
            cluster = int(row["canonical_reporting_cluster_id"])
            expected[cluster] = expected.get(cluster, 0.0) + float(value)
        else:
            unassigned_net += float(value)
    for cluster, value in expected.items():
        assert accounting.net_r_by_regime[cluster] == pytest.approx(value)
    assert accounting.unassigned_net_r == pytest.approx(unassigned_net)
    assert accounting.assigned_net_r_total == pytest.approx(sum(expected.values()))
    mass = sum(abs(v) for v in expected.values())
    assert accounting.abs_net_r_mass == pytest.approx(mass)
    assert accounting.abs_net_r_share_by_regime is not None
    for cluster, value in expected.items():
        assert accounting.abs_net_r_share_by_regime[cluster] == pytest.approx(abs(value) / mass)
    assert accounting.top_regime_abs_net_r_share == pytest.approx(
        max(abs(v) for v in expected.values()) / mass
    )
    assert body.top_regime_abs_net_r_share == accounting.top_regime_abs_net_r_share
    assert accounting.zero_denominator_reasons == ()
    assert accounting.assigned_regime_count == 3
    # unassigned trades exist: the claim is NULL only when the assigned side
    # would otherwise support it (adversarial RA-03); when the assigned side
    # already refutes it, it is FALSE and no reason is recorded
    positive = [cluster for cluster, value in expected.items() if value > 0]
    assigned_claim = sum(expected.values()) > 0 and len(positive) == 1
    if assigned_claim:
        assert accounting.works_only_in_regime is None
        assert accounting.works_only_in_regime_reason == "incomplete_assignment_accounting"
    else:
        assert accounting.works_only_in_regime is False
        assert accounting.works_only_in_regime_reason is None
    assert accounting.works_only_in_regime_id is None
    assert body.works_only_in_regime == ()
    # signed contribution fractions exist when the assigned total is nonzero
    if abs(accounting.assigned_net_r_total) > 0:
        assert accounting.signed_contribution_fraction_by_regime is not None
        assert sum(accounting.signed_contribution_fraction_by_regime.values()) == pytest.approx(
            1.0
        )
    # no statistical confidence is inferred for thin strata
    assert rows["regime:2"].metrics is None


def _crafted_trades(assignment: pd.DataFrame, days: tuple[str, ...]) -> pd.DataFrame:
    """Every trade assigned (no unassigned rows); regime 0 profitable (25
    winners), regimes 1 (12 stops) and 2 (5 stops, thin) losing — the assigned
    net R total stays positive so the works-only claim is decidable."""

    valid = assignment[assignment["valid"].astype(bool)]
    by_cluster = {
        int(cluster): sorted(group["candidate_id"].astype(str))
        for cluster, group in valid.groupby("canonical_reporting_cluster_id")
    }
    chosen = by_cluster[0][:25] + by_cluster[1][:12] + by_cluster[2][:5]
    frame = make_resolved_trades_frame(days, trades_per_day=2, loss_every=10**9).head(len(chosen))
    frame = frame.copy()
    frame["candidate_id"] = chosen
    losing = frame.index[25:]
    risk = frame.loc[losing, "risk_ticks"]
    frame.loc[losing, "resolution"] = "stop"
    frame.loc[losing, "realized_ticks"] = -risk
    frame.loc[losing, "mfe_ticks"] = 8
    frame.loc[losing, "mae_ticks"] = -risk
    return frame.reset_index(drop=True)


def test_works_only_in_regime_claim_requires_complete_assignment_accounting(lane):
    assignment = lane["assignment"]
    trades = _crafted_trades(assignment, lane["source"].fixture.trading_days)
    result = build_cohort_descriptive_body(
        trades=trades,
        assignment=assignment,
        core_replay_id=_CORE_A,
        protocol_id=lane["protocol_id"],
        fit_ids=lane["fit_ids"],
        cost_points=_COST,
        evaluation_config_hash=_EVAL_HASH,
    )
    accounting = result.body.net_r_accounting
    assert accounting.unassigned_trade_count == 0
    assert accounting.net_r_by_regime[0] > 0 > accounting.net_r_by_regime[1]
    assert accounting.net_r_by_regime[2] < 0  # the THIN regime still counts
    assert accounting.works_only_in_regime is True
    assert accounting.works_only_in_regime_id == 0
    assert accounting.assigned_regime_count == 3
    assert result.body.works_only_in_regime == (0,)
    assert accounting.works_only_in_regime_reason is None


def _accounting(net: list[float], clusters: list[int], valid: list[bool]) -> RegimeNetRAccounting:
    return regime_net_r_accounting(
        net_r=pd.Series(net), clusters=np.array(clusters), valid=np.array(valid)
    )


def test_works_only_in_regime_is_false_when_the_assigned_side_refutes_it():
    """Adversarial RA-03: unassigned trades make the claim NULL only when the
    assigned side would otherwise support it; a claim the assigned side
    already refutes is FALSE (no reason); a single assigned regime is a
    vacuous TRUE made visible by ``assigned_regime_count``."""

    # two positive regimes + one unassigned trade → refuted on the assigned side
    two_positive = _accounting([1.0, 2.0, 0.5], [0, 1, -1], [True, True, False])
    assert two_positive.works_only_in_regime is False
    assert two_positive.works_only_in_regime_reason is None
    assert two_positive.works_only_in_regime_id is None
    assert two_positive.unassigned_trade_count == 1
    assert two_positive.assigned_regime_count == 2
    # a negative assigned total + one unassigned trade → refuted, not null
    negative = _accounting([-1.0, -2.0, 0.5], [0, 1, -1], [True, True, False])
    assert negative.works_only_in_regime is False
    assert negative.works_only_in_regime_reason is None
    # exactly one positive regime, the other ≤ 0, one unassigned trade → the
    # unassigned trade PREVENTS the claim: null with the typed reason
    prevented = _accounting([3.0, -1.0, 0.5], [0, 1, -1], [True, True, False])
    assert prevented.works_only_in_regime is None
    assert prevented.works_only_in_regime_reason == "incomplete_assignment_accounting"
    assert prevented.works_only_in_regime_id is None
    # the same assigned side with complete accounting → TRUE for regime 0
    complete = _accounting([3.0, -1.0], [0, 1], [True, True])
    assert complete.works_only_in_regime is True
    assert complete.works_only_in_regime_id == 0
    assert complete.assigned_regime_count == 2
    # a single assigned regime: the plan's predicate is vacuously true, and
    # the count makes the vacuity visible
    single = _accounting([1.0, 2.0], [0, 0], [True, True])
    assert single.works_only_in_regime is True
    assert single.works_only_in_regime_id == 0
    assert single.assigned_regime_count == 1
    # a zero assigned total with complete accounting is FALSE (never null)
    flat = _accounting([1.0, -1.0], [0, 1], [True, True])
    assert flat.works_only_in_regime is False
    assert "zero_assigned_net_r_total" in flat.zero_denominator_reasons


def test_net_r_accounting_refuses_arithmetically_impossible_bodies(lane):
    """Adversarial RA-02: the validator recomputes mass, total, shares,
    fractions, the top share and the claim from ``net_r_by_regime`` — a
    well-typed but impossible body is refused, field by field."""

    impossible = dict(
        trade_count_by_regime={0: 3, 1: 2},
        net_r_by_regime={0: 5.0, 1: 3.0},
        assigned_trade_count=5,
        unassigned_trade_count=0,
        unassigned_net_r=0.0,
        assigned_net_r_total=-99.0,
        abs_net_r_mass=0.0,
        abs_net_r_share_by_regime=None,
        signed_contribution_fraction_by_regime={0: 0.1, 1: 0.9},
        top_regime_abs_net_r_share=None,
        works_only_in_regime=True,
        works_only_in_regime_id=7,
        works_only_in_regime_reason=None,
        zero_denominator_reasons=("zero_abs_net_r_mass",),
        assigned_regime_count=2,
    )
    with pytest.raises(ValueError):
        RegimeNetRAccounting(**impossible)
    good = _accounting([2.0, 3.0, 1.0, 2.0, -1.0], [0, 0, 0, 1, 1], [True] * 5)
    assert good.net_r_by_regime == {0: 6.0, 1: 1.0}
    assert good.works_only_in_regime is False  # two positive regimes
    base = good.model_dump()
    RegimeNetRAccounting(**base)  # the true accounting validates
    for update, message in (
        ({"assigned_regime_count": 3}, "assigned_regime_count"),
        ({"assigned_net_r_total": -99.0}, "assigned_net_r_total"),
        ({"abs_net_r_mass": 0.0}, "abs_net_r_mass"),
        ({"abs_net_r_share_by_regime": {0: 0.5, 1: 0.5}}, "abs_net_r_share_by_regime"),
        ({"top_regime_abs_net_r_share": 0.01}, "top_regime_abs_net_r_share"),
        (
            {"signed_contribution_fraction_by_regime": {0: 0.1, 1: 0.9}},
            "signed_contribution_fraction_by_regime",
        ),
        ({"works_only_in_regime_id": 7}, "works_only_in_regime_id"),
        ({"works_only_in_regime": True, "works_only_in_regime_id": 0}, "assigned-side"),
        (
            {
                "works_only_in_regime": None,
                "works_only_in_regime_reason": "incomplete_assignment_accounting",
            },
            "unassigned",
        ),
        ({"trade_count_by_regime": {0: 2, 1: 3}, "assigned_trade_count": 5}, None),
    ):
        fields = {**base, **update}
        if message is None:
            # counts that still sum correctly validate at the accounting level …
            RegimeNetRAccounting(**fields)
            continue
        with pytest.raises(ValueError, match=message):
            RegimeNetRAccounting(**fields)
    # … but the cohort body refuses per-regime counts that disagree with its strata
    trades, assignment = lane["trades"], lane["assignment"]
    result = build_cohort_descriptive_body(
        trades=trades,
        assignment=assignment,
        core_replay_id=_CORE_A,
        protocol_id=lane["protocol_id"],
        fit_ids=lane["fit_ids"],
        cost_points=_COST,
        evaluation_config_hash=_EVAL_HASH,
    )
    body = result.body.model_dump()
    CohortDescriptiveBody(**body)
    original = body["net_r_accounting"]
    accounting = dict(original)
    assert dict(accounting["trade_count_by_regime"]) == {0: 25, 1: 25, 2: 5}
    accounting["trade_count_by_regime"] = {0: 26, 1: 24, 2: 5}
    with pytest.raises(ValueError, match="executed_trades"):
        CohortDescriptiveBody(**{**body, "net_r_accounting": accounting})
    # a regime the strata do not carry (keys renamed 2 → 3, values preserved)
    renamed = dict(original)
    for key in (
        "trade_count_by_regime",
        "net_r_by_regime",
        "abs_net_r_share_by_regime",
        "signed_contribution_fraction_by_regime",
    ):
        value = original[key]
        if value is not None:
            renamed[key] = {(3 if regime == 2 else regime): v for regime, v in dict(value).items()}
    with pytest.raises(ValueError, match="strata"):
        CohortDescriptiveBody(**{**body, "net_r_accounting": renamed})


# ── RA-01: a caller frame must reproduce the bound executed-trade artifact ──


def _persist_table(root, core_replay_id: str, trades: pd.DataFrame):
    envelope, table_bytes = build_executed_trade_table(
        core_replay_id, trades, record_schema_version=2
    )
    stored, _reused = save_executed_trade_table(root, envelope, table_bytes)
    return stored


def test_service_refuses_a_caller_frame_that_does_not_reproduce_the_bound_table(lane):
    """Adversarial RA-01: the seam that mints immutable reports verifies the
    caller's executed-trade frame against the persisted table artifact it
    names (exact load; byte-for-byte after projection) and binds the
    artifact's own bytes hash beside the normalized-frame hash."""

    root = lane["root"]
    trades_a = lane["trades"]
    trades_b = _crafted_trades(lane["assignment"], lane["source"].fixture.trading_days)
    table_a = _persist_table(root, _CORE_A, trades_a)
    table_b = _persist_table(root, _CORE_B, trades_b)
    assert table_a.executed_trade_table_id != table_b.executed_trade_table_id

    def _inputs(core: str, trades: pd.DataFrame, table_id: str) -> StratificationInputs:
        return StratificationInputs(
            root=root,
            protocol_id=lane["protocol_id"],
            decision_id=lane["ready"].regime_promotion_decision_id,
            owner_decision_artifact_id=None,
            regime_oos_assignment_id=lane["oos_id"],
            requested_classes=(RegimeStratificationClass.COHORT_DESCRIPTIVE,),
            children={
                core: ChildStratificationInputs(
                    core_replay_id=core,
                    trades=trades,
                    cost_points=_COST,
                    evaluation_config_hash=_EVAL_HASH,
                    costed_evaluation_id="2" * 64,
                    executed_trade_table_id=table_id,
                )
            },
        )

    # B's rows under A's artifact id → refused before anything is published
    with pytest.raises(ValueError, match="does not reproduce"):
        build_regime_stratified_reports(_inputs(_CORE_A, trades_b, table_a.executed_trade_table_id))
    # another child's table named for this child → refused
    with pytest.raises(ValueError, match="another core replay"):
        build_regime_stratified_reports(_inputs(_CORE_A, trades_a, table_b.executed_trade_table_id))
    # an unknown table id → the store refuses (typed), never an unverified report
    with pytest.raises(SearchStoreError):
        build_regime_stratified_reports(_inputs(_CORE_A, trades_a, "f" * 64))
    # the matching pair: both hashes ride the body, the id rides the refs
    outcome = build_regime_stratified_reports(
        _inputs(_CORE_A, trades_a, table_a.executed_trade_table_id)
    )
    report = load_regime_stratified_report(root, outcome.report_ids[0])
    body = report.payload.body
    assert body.executed_trade_table_id == table_a.executed_trade_table_id
    assert body.executed_trade_table_artifact_sha256 == table_a.executed_trade_table_sha256
    loaded = load_executed_trade_table(root, table_a.executed_trade_table_id)
    assert body.executed_trade_table_sha256 == executed_trade_table_sha256(
        normalized_executed_trades(loaded.frame)
    )
    assert table_a.executed_trade_table_id in report.payload.source_metric_refs
    # a reordered frame with a stray column still reproduces the artifact
    # after projection → the identical report (deterministic id)
    shuffled = trades_a.sample(frac=1.0, random_state=5).reset_index(drop=True)
    shuffled["_scratch"] = 1
    again = build_regime_stratified_reports(
        _inputs(_CORE_A, shuffled, table_a.executed_trade_table_id)
    )
    assert again.report_ids == outcome.report_ids
    # the body contract binds the artifact hash exactly with the id
    with pytest.raises(ValueError, match="artifact"):
        build_cohort_descriptive_body(
            trades=trades_a,
            assignment=lane["assignment"],
            core_replay_id=_CORE_A,
            protocol_id=lane["protocol_id"],
            fit_ids=lane["fit_ids"],
            cost_points=_COST,
            evaluation_config_hash=_EVAL_HASH,
            executed_trade_table_id=table_a.executed_trade_table_id,
        )
    with pytest.raises(ValueError, match="artifact"):
        build_cohort_descriptive_body(
            trades=trades_a,
            assignment=lane["assignment"],
            core_replay_id=_CORE_A,
            protocol_id=lane["protocol_id"],
            fit_ids=lane["fit_ids"],
            cost_points=_COST,
            evaluation_config_hash=_EVAL_HASH,
            executed_trade_table_artifact_sha256=table_a.executed_trade_table_sha256,
        )


def test_cohort_descriptive_hashes_and_joins_the_normalized_table(lane):
    trades, assignment = lane["trades"], lane["assignment"]
    result = build_cohort_descriptive_body(
        trades=trades,
        assignment=assignment,
        core_replay_id=_CORE_A,
        protocol_id=lane["protocol_id"],
        fit_ids=lane["fit_ids"],
        cost_points=_COST,
        evaluation_config_hash=_EVAL_HASH,
    )
    normalized = normalized_executed_trades(trades, tick_size=0.25)
    assert result.body.executed_trade_table_sha256 == executed_trade_table_sha256(normalized)
    # the binding hash is over the normalizer's projection: no derived working
    # column, `trade_id` order, direction upper-cased
    assert not any(str(column).startswith("_") for column in normalized.columns)
    assert list(normalized["trade_id"]) == sorted(normalized["trade_id"])
    assert set(_validate_and_normalize_executed_trades(trades, tick_size=0.25).columns) >= set(
        normalized.columns
    )
    # a raw frame with a column outside the normalized projection or a
    # different row order hashes to the SAME binding value
    shuffled = trades.sample(frac=1.0, random_state=3).reset_index(drop=True)
    shuffled["_scratch"] = 1
    again = build_cohort_descriptive_body(
        trades=shuffled,
        assignment=assignment,
        core_replay_id=_CORE_A,
        protocol_id=lane["protocol_id"],
        fit_ids=lane["fit_ids"],
        cost_points=_COST,
        evaluation_config_hash=_EVAL_HASH,
    )
    assert again.body.executed_trade_table_sha256 == result.body.executed_trade_table_sha256
    assert again.body.net_r_accounting == result.body.net_r_accounting
    # a direction spelled in lower case is normalized before the join, so the
    # strata are byte-identical
    lowered = trades.copy()
    lowered["direction"] = lowered["direction"].str.lower()
    lowered_result = build_cohort_descriptive_body(
        trades=lowered,
        assignment=assignment,
        core_replay_id=_CORE_A,
        protocol_id=lane["protocol_id"],
        fit_ids=lane["fit_ids"],
        cost_points=_COST,
        evaluation_config_hash=_EVAL_HASH,
    )
    assert lowered_result.body.executed_trade_table_sha256 == (
        result.body.executed_trade_table_sha256
    )
    # per-trade net R agrees with the gate metrics' expectancy
    net = per_trade_net_r(trades, cost_points=_COST)
    pooled = compute_strategy_metrics(
        {RecordTable.EXECUTED_TRADE: trades},
        cost_points=_COST,
        evaluation_config_hash=_EVAL_HASH,
    )
    assert float(np.mean(net.to_numpy())) == pytest.approx(pooled.net_expectancy_r)
    assert set(net.index) == set(trades["trade_id"].astype(str))


# ── HARDENING-BACKEND-FIX §7.3 — persisted reports bind the exact trade table ─


def test_persisted_reports_require_and_verify_the_exact_executed_trade_table(lane) -> None:
    """HB-FIX-09: the persisting service refuses a child without its
    executed-trade table id before anything is published; the ephemeral
    helper still serves a caller frame without publishing; a tampered
    declared table (a traversal entry in its manifest) fails closed."""

    import json

    from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256
    from alpha_lab.agents.data_infra.ifvg.ml.regime_stratification_service import (
        StratificationEvidenceError,
    )
    from alpha_lab.agents.data_infra.ifvg.search.executed_trade_table import (
        EXECUTED_TRADE_TABLE_STORE,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import SidecarLoadError, envelope_destination

    root = lane["root"]
    trades = lane["trades"]
    store_dir = root / "regime_stratified_reports"

    def _published() -> set[str]:
        return {p.name for p in store_dir.iterdir()} if store_dir.exists() else set()

    def _inputs(child: ChildStratificationInputs) -> StratificationInputs:
        return StratificationInputs(
            root=root,
            protocol_id=lane["protocol_id"],
            decision_id=lane["ready"].regime_promotion_decision_id,
            owner_decision_artifact_id=None,
            regime_oos_assignment_id=lane["oos_id"],
            requested_classes=(RegimeStratificationClass.COHORT_DESCRIPTIVE,),
            children={_CORE_A: child},
        )

    before = _published()
    # (i) a caller frame without the table id: typed refusal, nothing published
    with pytest.raises(StratificationEvidenceError) as refused:
        build_regime_stratified_reports(
            _inputs(
                ChildStratificationInputs(
                    core_replay_id=_CORE_A,
                    trades=trades,
                    cost_points=_COST,
                    evaluation_config_hash=_EVAL_HASH,
                    costed_evaluation_id="2" * 64,
                )
            )
        )
    assert refused.value.reason == "executed_trade_table_required"
    assert _published() == before
    # (ii) the explicitly non-persisting helper serves the same frame in memory
    ephemeral = build_cohort_descriptive_body(
        trades=trades,
        assignment=lane["assignment"],
        core_replay_id=_CORE_A,
        protocol_id=lane["protocol_id"],
        fit_ids=lane["fit_ids"],
        cost_points=_COST,
        evaluation_config_hash=_EVAL_HASH,
    )
    assert ephemeral.body.executed_trade_table_id is None
    assert _published() == before
    # (iii) the exact table binds the persisted report
    table = _persist_table(root, _CORE_A, trades)
    outcome = build_regime_stratified_reports(
        _inputs(
            ChildStratificationInputs(
                core_replay_id=_CORE_A,
                trades=trades,
                cost_points=_COST,
                evaluation_config_hash=_EVAL_HASH,
                costed_evaluation_id="2" * 64,
                executed_trade_table_id=table.executed_trade_table_id,
            )
        )
    )
    report = load_regime_stratified_report(root, outcome.report_ids[0])
    assert report.payload.body.executed_trade_table_id == table.executed_trade_table_id
    assert table.executed_trade_table_id in report.payload.source_metric_refs
    # (iv) a tampered declared table fails closed: a traversal entry smuggled
    # into the table's manifest (rehashed) is refused by the central validator
    directory = envelope_destination(
        root, EXECUTED_TRADE_TABLE_STORE, table.executed_trade_table_id
    )
    manifest_path = directory / "manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original.decode("utf-8"))
    manifest["artifacts"].append({"path": "../escape.arrow", "sha256": "0" * 64, "bytes": 0})
    core = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
    manifest["manifest_payload_sha256"] = canonical_sha256(core)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    published = _published()
    try:
        with pytest.raises(SidecarLoadError) as tampered:
            build_regime_stratified_reports(
                _inputs(
                    ChildStratificationInputs(
                        core_replay_id=_CORE_A,
                        trades=trades,
                        cost_points=_COST,
                        evaluation_config_hash=_EVAL_HASH,
                        costed_evaluation_id="3" * 64,
                        executed_trade_table_id=table.executed_trade_table_id,
                    )
                )
            )
        assert tampered.value.reason == "malformed_manifest"
        assert _published() == published
    finally:
        manifest_path.write_bytes(original)
