"""Computation-path derivation — the brief §7A.5 table, row by row."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.study.computation_path import (
    derive_computation_path,
)
from alpha_lab.agents.data_infra.ifvg.study.dimension_contracts import (
    EXPERIMENT_DIMENSION_REGISTRY,
)

# short aliases keep the §7A.5 rows readable on one line each
FSR = "full_strategy_replay"
FM = "feature_materialization"
LR = "label_recomputation"
MR = "model_refit"
MG = "model_gated_sequential_replay"
CR = "cost_recomputation"
PR = "prop_resimulation"
BR = "bootstrap_resimulation"
RT = "reuse_trade_stream_hash"


aliases = {
    "fsr": FSR, "fm": FM, "lr": LR, "mr": MR, "mg": MG,
    "cr": CR, "pr": PR, "br": BR, "rt": RT,
}


def _d(**kw):
    return {
        aliases[k]: v
        for k, v in kw.items()
    }


def _path_for(*dimension_ids: str):
    return derive_computation_path(
        tuple(EXPERIMENT_DIMENSION_REGISTRY[d] for d in dimension_ids)
    )


@pytest.mark.parametrize(
    ("dimension_id", "expected"),
    [
        # feature blocks ⇒ materialize + refit; replay unchanged; stream reusable
        (
            "feature_block.membership",
            _d(fsr=False, fm=True, mr=True, rt=True, pr=False),
        ),
        # cohort-descriptive ⇒ nothing
        (
            "observation_cohort.descriptive_slice",
            _d(fsr=False, fm=False, lr=False, mr=False, cr=False, pr=False, br=False, rt=True),
        ),
        # cohort model ⇒ refit
        ("observation_cohort.specialized_model", _d(mr=True, fsr=False)),
        # label ⇒ label recompute (+ refit); stream reusable
        ("label_policy.candidate_static_r", _d(lr=True, mr=True, fsr=False, rt=True)),
        # lifetime-changing label ⇒ full replay closure
        ("label_policy.lifetime_changing", _d(fsr=True, cr=True, pr=True, br=True, rt=False)),
        # FSM / session / direction / entry-family ⇒ full replay closure
        ("strategy_profile.fsm_axis", _d(fsr=True, cr=True, pr=True, br=True, rt=False)),
        ("strategy_profile.session_policy", _d(fsr=True, rt=False)),
        ("strategy_profile.direction_policy", _d(fsr=True)),
        ("strategy_profile.entry_family", _d(fsr=True)),
        # model algo/hyperparams ⇒ refit on identical rows/folds
        ("model_protocol.algorithm", _d(mr=True, fsr=False, rt=True)),
        # execution-affecting threshold/abstention ⇒ frozen model + gated replay
        ("decision_policy.execution_gate", _d(mg=True, cr=True, pr=True, br=True, rt=False)),
        # lifetime-changing fill/exit ⇒ full replay
        ("execution_policy.fill_or_exit", _d(fsr=True, rt=False)),
        # cost-only ⇒ cost recompute with the stream reused
        ("cost_policy.round_turn_cost", _d(cr=True, fsr=False, pr=False, rt=True)),
        # risk ⇒ account resim on the reused stream
        ("risk_policy.sizing", _d(pr=True, br=True, fsr=False, rt=True)),
        # prop contract ⇒ prop resim
        ("prop_contract.firm_rules", _d(pr=True, fsr=False, rt=True)),
        # payout/withdrawal ⇒ payout resim
        ("payout_policy.withdrawal_behavior", _d(pr=True, fsr=False)),
        # portfolio ⇒ portfolio resim
        ("portfolio_policy.legs", _d(pr=True, fsr=False)),
        # bootstrap/stress ⇒ sim rerun only
        ("validation_protocol.bootstrap", _d(br=True, pr=False, fsr=False, rt=True)),
        ("stress_scenario.scenario_set", _d(br=True, fsr=False)),
        # data/formula/bar policy ⇒ regenerate everything
        (
            "data_lineage.source_dataset",
            _d(fsr=True, fm=True, lr=True, mr=True, cr=True, pr=True, br=True, rt=False),
        ),
    ],
)
def test_computation_table_row(dimension_id: str, expected: dict) -> None:
    path = _path_for(dimension_id)
    for field, value in expected.items():
        assert getattr(path, field) is value, (dimension_id, field)


def test_or_fold_composes_and_closure_holds() -> None:
    combined = _path_for("cost_policy.round_turn_cost", "risk_policy.sizing")
    assert combined.cost_recomputation and combined.prop_resimulation
    assert combined.reuse_trade_stream_hash is True
    with_replay = _path_for("cost_policy.round_turn_cost", "strategy_profile.fsm_axis")
    assert with_replay.full_strategy_replay is True
    assert with_replay.prop_resimulation is True  # closure
    assert with_replay.reuse_trade_stream_hash is False
