"""Saved MCB plans keep every intent visible when a child has no result."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_measures
from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import study_from_result
from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_matrix import matrix_rows
from alpha_lab.propsim.funded.mffu_batch_plan import PLAN_SCHEMA


def _variant(number: int):
    name = f"MCB{number:03d}"
    intent = {"schedule": "S0", "daily_cap": "U", "entry_context": "F0",
              "exit": "XP", "sizing": "Q10", "overhead": "O0", "geometry": "G0"}
    section = {"ifsm_context_policy_version": "mq_eod_asof_nominal_2200_chicago_v01",
               "entry_context_policy": "off", "overhead_policy": "off",
               "opposing_distance_policy": "fixed_v1",
               "exit_policy": "scale_out_half_breakeven_hold_to_close_v1",
               "max_executed_trades_per_day": None}
    return SimpleNamespace(
        variant_id=name, name=name, family="core_48", intent_json=json.dumps(intent),
        effective_section_json=json.dumps(section), possible_quantities=(10,),
        effective_section_config_hash=f"{number:064x}",
        effective_behavior_hash=f"{number + 10:064x}",
    )


def test_matrix_keeps_missing_and_failed_intents_without_zero_cash():
    plan = SimpleNamespace(
        plan_schema=PLAN_SCHEMA,
        configurations=(_variant(1), _variant(2), _variant(3)),
        firm_profiles=(SimpleNamespace(firm_key="myfundedfutures",
                                       firm_name="MyFundedFutures"),),
        source=SimpleNamespace(evaluation_dates=("2026-06-10",), warmup_dates=()),
    )
    result = {
        "funded_comparison_plan_id": "a" * 64,
        "settings": {"firm_profiles": [{"firm_key": "myfundedfutures",
                                         "firm_name": "MyFundedFutures"}]},
        "summaries_cents": {
            "MCB001|myfundedfutures": {
                "configuration": "MCB001", "firm_key": "myfundedfutures",
                "status": "Not completed", "reason": "input unavailable",
            },
        },
        "full_range_batch": {"failed_configurations": [
            {"configuration": "MCB002", "status": "Not completed",
             "reason": "worker failed"},
        ]},
        "tables": {},
    }
    study = study_from_result(result, result_id="b" * 64, plan=plan)
    assert study.configurations == ("MCB001", "MCB002", "MCB003")
    rows = matrix_rows(study)
    assert [row["Status"] for row in rows] == [
        "Not completed", "Not completed", "No result saved",
    ]
    assert [row["Reason"] for row in rows] == [
        "input unavailable", "worker failed", "",
    ]
    assert all(row["Net cash cents"] is None for row in rows)
    assert [row.configuration for row in funded_measures.ranking(
        study, "myfundedfutures"
    )] == ["MCB001", "MCB002", "MCB003"]
    assert dict(study.settings("MCB003"))["Effective section hash"] == f"{3:064x}"
    result["summaries_cents"]["UNPLANNED|myfundedfutures"] = {
        "configuration": "UNPLANNED", "firm_key": "myfundedfutures",
        "status": "Completed",
    }
    with pytest.raises(ValueError, match="outside the frozen plan"):
        study_from_result(result, result_id="b" * 64, plan=plan)


def test_saved_mffu_dispositions_must_cover_all_64_intents():
    plan = SimpleNamespace(
        plan_schema=PLAN_SCHEMA,
        configurations=tuple(_variant(number) for number in range(1, 65)),
        firm_profiles=(SimpleNamespace(firm_key="myfundedfutures",
                                       firm_name="MyFundedFutures"),),
        source=SimpleNamespace(evaluation_dates=("2026-06-10",), warmup_dates=()),
    )
    dispositions = [
        {"variant_id": f"MCB{number:03d}", "status": "failed",
         "reason": "source unavailable", "reused_from": None}
        for number in range(1, 65)
    ]
    result = {
        "funded_comparison_plan_id": "a" * 64,
        "mffu_batch": {"schema": "ifsm_mffu_context_64_batch_result_v1",
                       "dispositions": dispositions},
        "summaries_cents": {}, "tables": {},
    }
    study = study_from_result(result, result_id="b" * 64, plan=plan)
    rows = matrix_rows(study)
    assert len(rows) == 64
    assert all(row["Status"] == "Failed" and row["Net cash cents"] is None
               for row in rows)
    result["mffu_batch"]["dispositions"] = dispositions[:-1]
    with pytest.raises(ValueError, match="dropped or reordered"):
        study_from_result(result, result_id="b" * 64, plan=plan)
