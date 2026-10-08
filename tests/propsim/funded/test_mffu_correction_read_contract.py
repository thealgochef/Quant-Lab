"""Corrected dispositions pass the real read/export contracts; lineage is required."""

from types import SimpleNamespace

import pytest

from alpha_lab.propsim.funded.mffu_batch_plan import PLAN_SCHEMA
from alpha_lab.propsim.funded.mffu_batch_review import _assert_result


def _objects():
    ids = [f"MCB{n:03}" for n in range(1, 65)]
    core = {"patch_sha256": "c" * 64}
    payload = {"read_contract_fixture": True}
    plan = SimpleNamespace(
        plan_schema=PLAN_SCHEMA,
        variants=[SimpleNamespace(variant_id=key, name=key) for key in ids],
        configurations=[SimpleNamespace(variant_id=key, name=key) for key in ids],
        core_source=SimpleNamespace(model_dump=lambda **_: core),
        source=SimpleNamespace(evaluation_dates=("2026-06-10",), warmup_dates=()),
        model_dump=lambda **_: payload,
    )
    dispositions = []
    proofs = {}
    for index, key in enumerate(ids):
        status = ("reused_nonimpact" if index < 18 else
                  "replayed_equal" if index % 2 else "replayed_changed")
        dispositions.append({"variant_id": key, "status": status,
                             "reused_from": "o" * 64 if index < 18 else None})
        if index < 18:
            proofs[key] = {"schema": "ifsm_mffu_correction_child_reuse_v1", "status": status,
                           "original_result_id": "o" * 64, "corrected_core_source": core}
    result = {
        "funded_comparison_plan_id": "p" * 64, "validation": {"passed": True},
        "mffu_batch": {
            "schema": "ifsm_mffu_context_64_batch_result_v1", "plan": payload,
            "dispositions": dispositions, "decision_context": [], "reuse_proofs": proofs,
            "reuse_context_annotations": [],
            "correction_lineage": {"schema": "ifsm_mffu_lifecycle_correction_receipt_v1",
                                   "original_result_id": "o" * 64},
        },
        "mffu_analysis": {
            "schema": "ifsm_mffu_batch_analysis_v2", "variant_count": 64,
            "matched_pairs": [{"trade_diagnostics": {}} for _ in range(188)],
            "entry_order_runner_by_variant": dict.fromkeys(ids, {}),
            "decision_context_coverage_by_variant": dict.fromkeys(ids, {}),
        },
        "summaries_cents": {
            key + "|myfundedfutures": {"configuration": key, "firm_key": "myfundedfutures",
                                       "status": "Completed", "net_cash_earned_cents": 0}
            for key in ids
        },
    }
    return plan, result


def test_all_64_correction_dispositions_open_in_real_read_models():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import study_from_result

    plan, result = _objects()
    assert len(_assert_result(plan, "p" * 64, result)[0]) == 64
    study = study_from_result(result, result_id="r" * 64, plan=plan)
    assert len(study.configurations) == len(study.completed_at("myfundedfutures")) == 64


@pytest.mark.parametrize("broken", ["missing_lineage", "wrong_parent", "wrong_core"])
def test_correction_status_cannot_claim_reuse_without_exact_read_proof(broken):
    plan, result = _objects()
    batch = result["mffu_batch"]
    if broken == "missing_lineage":
        batch.pop("correction_lineage")
    elif broken == "wrong_parent":
        batch["reuse_proofs"]["MCB001"]["original_result_id"] = "x" * 64
    else:
        batch["reuse_proofs"]["MCB001"]["corrected_core_source"] = {}
    with pytest.raises(ValueError, match="exact lineage proof"):
        _assert_result(plan, "p" * 64, result)
