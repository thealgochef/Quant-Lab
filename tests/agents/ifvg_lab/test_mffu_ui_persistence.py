"""A verified saved MFFU result reopens with all plan axes and cash rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import ifvg_lab_funded as funded_ui  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (  # noqa: E402
    open_funded_study,
)
from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import (  # noqa: E402
    ranking,
)
from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_matrix import (  # noqa: E402
    matrix_rows,
    saved_analysis_tables,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: E402
    save_envelope_immutable,
)
from alpha_lab.propsim.funded import comparison_runner  # noqa: E402
from alpha_lab.propsim.funded.comparison_plan import (  # noqa: E402
    RESULT_STORE,
    FundedComparisonResultEnvelope,
    FundedComparisonResultPayload,
)
from alpha_lab.propsim.funded.mffu_batch_plan import PLAN_SCHEMA  # noqa: E402
from alpha_lab.propsim.funded.result import canonical_json, result_sha256  # noqa: E402


def _plan() -> SimpleNamespace:
    values = {
        4: {"entry_context": "FEL", "entry_context_policy":
            "skip_early_positive_and_london_v1"},
        3: {"exit": "XG", "exit_policy": "gamma_conditional_1r_v1"},
        50: {"sizing": "QG", "possible_quantities": (6, 10)},
        55: {"overhead": "O8", "overhead_policy":
             "nearest_studied_8_above_target_gex_gt_300000_v1"},
        59: {"geometry": "G075", "opposing_distance_policy":
             "implied_move_0075_v1"},
        64: {"exit": "XE", "exit_policy": "early_positive_whole_1r_v1"},
    }
    variants = []
    for number in range(1, 65):
        override = values.get(number, {})
        intent = {"schedule": "S0", "daily_cap": "U", "entry_context": "F0",
                  "exit": "XP", "sizing": "Q10", "overhead": "O0", "geometry": "G0"}
        section = {
            "ifsm_context_policy_version": "mq_eod_asof_nominal_2200_chicago_v01",
            "entry_context_policy": "off", "overhead_policy": "off",
            "opposing_distance_policy": "fixed_v1",
            "exit_policy": "scale_out_half_breakeven_hold_to_close_v1",
            "max_executed_trades_per_day": None,
        }
        for key, value in override.items():
            if key in intent:
                intent[key] = value
            else:
                section[key] = value
        name = f"MCB{number:03d}"
        variants.append(SimpleNamespace(
            name=name, variant_id=name, family="core_48",
            intent_json=json.dumps(intent), effective_section_json=json.dumps(section),
            possible_quantities=override.get("possible_quantities", (10,)),
            effective_section_config_hash=f"{number:064x}",
            effective_behavior_hash=f"{number + 64:064x}",
        ))
    return SimpleNamespace(
        plan_schema=PLAN_SCHEMA, configurations=tuple(variants),
        firm_profiles=(SimpleNamespace(firm_key="myfundedfutures",
                                       firm_name="MyFundedFutures"),),
        source=SimpleNamespace(evaluation_dates=("2026-06-10",), warmup_dates=()),
        model_dump=lambda **_kwargs: {"frozen": "64-matrix"},
    )


def _saved_result(store: Path, plan_id: str) -> str:
    dispositions = [{"variant_id": f"MCB{number:03d}",
                     "status": "newly_completed" if number == 1 else "failed",
                     "reason": None if number == 1 else "verified worker unavailable",
                     "reused_from": None} for number in range(1, 65)]
    result = {
        "funded_comparison_plan_id": plan_id,
        "settings": {"firm_profiles": [{"firm_key": "myfundedfutures",
                                        "firm_name": "MyFundedFutures"}]},
        "mffu_batch": {"schema": "ifsm_mffu_context_64_batch_result_v1",
                       "plan": {"frozen": "64-matrix"},
                       "dispositions": dispositions},
        "mffu_analysis": {
            "schema": "ifsm_mffu_batch_analysis_v1",
            "matched_pairs": [
                {"base_id": "MCB001", "challenger_id": "MCB002",
                 "changed_axis": "daily_cap", "group": "cap", "status": "unavailable",
                 "delta_cents": None, "unavailable_ids": ["MCB002"],
                 "trade_diagnostics": {"gained_entries": None, "lost_entries": None}},
            ],
            "interactions": {"fe_x_fl": {"cells": [
                {"schedule": "S0", "daily_cap": "U", "status": "unavailable",
                 "delta_cents": None, "unavailable_ids": ["MCB002"]},
            ]}},
            "waiting_by_variant": {"MCB001": {
                "status": "available", "between_status": "fewer_than_two_receipts",
                "terminal_status": "available", "initial": {
                    "start_chicago_date": "2025-06-15",
                    "end_chicago_date": "2025-07-18", "calendar_days": 33,
                    "evaluated_trading_days": 23, "censored_start": True,
                    "censored_end": False,
                }, "max_between_receipts": None,
            }},
        },
        "summaries_cents": {"MCB001|myfundedfutures": {
            "configuration": "MCB001", "firm_key": "myfundedfutures",
            "status": "Completed", "rank_within_firm": 1,
            "net_cash_earned_cents": 1_234_567,
            "payouts_received_cents": 1_259_567,
            "account_costs_cents": 25_000,
            "payouts_received_count": 1, "accounts_purchased": 2,
            "accounts_lost_before_first_payout": 1,
            "accounts_lost_after_a_payout": 0,
            "largest_single_payout_cents": 1_259_567,
            "trades_taken": 0,
        }},
        "tables": {},
    }
    payload = FundedComparisonResultPayload(
        funded_comparison_plan_id=plan_id, funded_comparison_approval_id="b" * 64,
        result_json_sha256=result_sha256(result), validation_passed=True,
        engine_version="synthetic_ui_audit",
    )
    envelope = FundedComparisonResultEnvelope.from_payload(payload)
    save_envelope_immutable(
        store, RESULT_STORE, envelope,
        extra_files={comparison_runner.RESULT_SIDECAR: canonical_json(result).encode()},
    )
    return envelope.funded_comparison_result_id


def test_verified_saved_result_reopens_all_matrix_values_and_cash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    monkeypatch.setattr(comparison_runner, "load_plan", lambda _root, _id: plan)
    plan_id = "a" * 64
    result_id = _saved_result(tmp_path, plan_id)
    study = open_funded_study(tmp_path, result_id)
    assert len(study.configurations) == 64
    assert study.configurations == tuple(f"MCB{number:03d}" for number in range(1, 65))
    assert dict(study.settings("MCB004"))["Entry gate"] == (
        "skip_early_positive_and_london_v1")
    assert dict(study.settings("MCB050"))["Possible micro quantities"] == "6, 10"
    assert dict(study.settings("MCB055"))["Overhead rule"] == (
        "nearest_studied_8_above_target_gex_gt_300000_v1")
    assert dict(study.settings("MCB059"))["Opposing distance rule"] == (
        "implied_move_0075_v1")
    rows = matrix_rows(study)
    assert len(rows) == 64
    assert rows[0]["Status"] == "Newly completed"
    assert rows[0]["Net cash cents"] == 1_234_567
    assert rows[63]["Exit"] == "XE" and rows[63]["Net cash cents"] is None
    assert rows[63]["Effective exit rule"] == "early_positive_whole_1r_v1"
    assert rows[49]["Possible micro quantities"] == "6, 10"
    assert rows[54]["Effective overhead rule"] == (
        "nearest_studied_8_above_target_gex_gt_300000_v1")
    assert rows[58]["Effective opposing distance"] == "implied_move_0075_v1"
    assert rows[0]["Effective daily cap"] == "None (uncapped)"
    analysis = saved_analysis_tables(study)
    assert analysis["matched_pairs"][0]["Net cash effect"] is None
    assert analysis["interactions"][0]["Unavailable IDs"] == "MCB002"
    assert analysis["waiting"][0]["Calendar days"] == 33
    assert analysis["waiting"][0]["Evaluated trading days"] == 23
    assert analysis["waiting"][1]["Status"] == "fewer_than_two_receipts"
    ranked = ranking(study, "myfundedfutures")
    assert len(ranked) == 64 and ranked[0].net_cash_cents == 1_234_567
    assert all(not row.completed and row.net_cash_cents is None for row in ranked[1:])
    names = funded_ui.study_names_for(study)
    assert len(names) == 64 and "XE" in names["MCB064"].line2
    assert "QG" in names["MCB050"].line2
    monkeypatch.setattr(comparison_runner, "load_plan",
                        lambda _root, _id: (_ for _ in ()).throw(ValueError("missing")))
    with pytest.raises(ValueError, match="frozen plan is unavailable"):
        open_funded_study(tmp_path, result_id)
