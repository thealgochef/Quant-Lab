"""Read-only repaired reference reuse; no historical market replay."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection

from alpha_lab.agents.data_infra.ifvg.menthorq_asof import load_v02_eod_asof_zip
from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
from alpha_lab.propsim.funded.comparison_plan import PLAN_STORE
from alpha_lab.propsim.funded.core_identity import source_identity_at
from alpha_lab.propsim.funded.full_range_batch import (
    FullRangeBatchPlanEnvelope,
    load_checkpoint,
)
from alpha_lab.propsim.funded.mffu_batch_plan import _variant_from_intent
from alpha_lab.propsim.funded.mffu_batch_reuse import (
    try_reuse_partial,
    verify_saved_reuse,
)
from alpha_lab.propsim.funded.profiles import MYFUNDEDFUTURES_PROFILE

TASK = Path("C:/Users/gonza/Documents/Claude-Quant-Lab-Research-Artifacts/"
            "ifsm-mffu-context-batch-20261007")
OLD = Path("C:/Users/gonza/Documents/Claude-Quant-Lab-Research-Artifacts/"
           "ifsm-correct-config-full-range-v01")
OLD_PLAN_ID = "fac96ed3c61f1f59faf654cff53b4ef725e462b84868de706f8dcbd49facf500"
HANDOFF = TASK / "handoff/docs/ifsm-mffu-context-batch-v01"
CONTEXT = HANDOFF / "inputs/MenthorQ_Research_Data_v02.zip"

pytestmark = pytest.mark.skipif(
    "ifsm_context_policy_version" not in IfvgSmcSection.model_fields,
    reason="requires the explicitly selected task Core",
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def reference():
    if not all(path.exists() for path in (CONTEXT, HANDOFF / "CONFIGURATION_MATRIX.json",
                                         OLD / "study_store", TASK / "core")):
        pytest.skip("exact repaired reference or task handoff is unavailable")
    store = OLD / "study_store"
    original = load_verified_envelope(
        store, PLAN_STORE, OLD_PLAN_ID, FullRangeBatchPlanEnvelope,
    ).payload
    matrix = json.loads((HANDOFF / "CONFIGURATION_MATRIX.json").read_text())
    bases = {item.name: item.model_dump(mode="json") for item in original.configurations}
    rows = {item["variant_id"]: _variant_from_intent(item, bases[item["base_reference"]])
            for item in matrix["variants"] if item["variant_id"] in {"MCB001", "MCB025"}}
    source_identity = source_identity_at(TASK / "core")
    from alpha_lab.propsim.funded.comparison_plan import CoreSourceRef

    core_source = CoreSourceRef(
        base_commit=source_identity["base_commit"],
        branch=source_identity["branch"],
        patch_sha256=source_identity["patch_sha256"],
        description="selected task Core",
    )
    current_hashes = {
        path: _sha(Path(path))
        for path, _digest in original.runtime_source_file_sha256.items()
    }
    plan = SimpleNamespace(
        reference_store_root=str(store),
        reference_plan_id=OLD_PLAN_ID,
        reference_result_id="7278632babf01b084c43ddb6df77332f15d8b16702c7082390553b856053d4ad",
        reference_plan_envelope_sha256=_sha(
            store / PLAN_STORE / OLD_PLAN_ID / "envelope.json"),
        source=original.source,
        calendar_sha256=original.calendar_sha256,
        task_b_plan_id=original.task_b_plan_id,
        task_b_scope=original.task_b_scope,
        prepared_registration_ids=original.prepared_registration_ids,
        processing=original.processing,
        execution_model=original.execution_model,
        firm_profiles=(MYFUNDEDFUTURES_PROFILE,),
        context_archive_sha256=_sha(CONTEXT),
        core_root=str(TASK / "core"),
        core_source=core_source,
        runtime_source_file_sha256=current_hashes,
    )
    context = load_v02_eod_asof_zip(
        CONTEXT, cutoff_utc=datetime(2026, 6, 10, 21, tzinfo=UTC),
        expected_archive_sha256=plan.context_archive_sha256,
    )
    return plan, rows, context


@pytest.mark.parametrize(("variant", "old_batch"), [
    ("MCB001", "C02"), ("MCB025", "C01"),
])
def test_exact_verified_reuse_preserves_old_bytes_and_separates_v02(
    tmp_path, reference, variant, old_batch,
):
    plan, rows, context = reference
    old_path = (OLD / "study_jobs" / OLD_PLAN_ID / "workers" / old_batch / "output.json")
    before = _sha(old_path)
    row = rows[variant]
    output, proof = try_reuse_partial(
        "a" * 64, "b" * 64, plan, row, tmp_path, context,
    )
    assert _sha(old_path) == before
    assert proof["reference"]["old_worker_checkpoint_file_sha256"] == before
    assert proof["reference"]["old_batch_id"] == old_batch
    assert proof["economic_identity"]["evaluation_date_count"] == 253
    assert proof["section_semantic_delta"]["only_added_neutral_context_fields"] == {
        "entry_context_policy": "off", "overhead_policy": "off",
        "opposing_distance_policy": "fixed_v1",
        "ifsm_context_policy_version": "mq_eod_asof_nominal_2200_chicago_v01",
    }
    assert output["batch_id"] == output["configuration"] == variant
    assert output["pairs"].keys() == {"myfundedfutures"}
    assert output["pairs"]["myfundedfutures"]["ledger"]["pair_id"] == (
        variant + "|myfundedfutures")
    assert len(output["daily_activity"]) == 253 * 2
    assert output["policy_decisions"] == {"strategy": [], "funded": []}
    assert output["reuse"]["posthoc_v02_is_executed"] is False
    assert output["reference"]["equivalent"] is None
    assert output["strategy_trades_no_account"][0]["entry_context"][
        "nearest_support_universe"] == "all_19"  # untouched v1 annotation
    original = json.loads(old_path.read_text())["payload"]["output"]
    assert (output["pairs"]["myfundedfutures"]["ledger"]["receipts"],
            output["pairs"]["myfundedfutures"]["ledger"]["costs"]) == (
            original["pairs"]["myfundedfutures"]["ledger"]["receipts"],
            original["pairs"]["myfundedfutures"]["ledger"]["costs"])
    saved = load_checkpoint(
        tmp_path / variant / "output.json",
        dispatch_sha256=output["dispatch_sha256"],
        dates=plan.source.warmup_dates + plan.source.evaluation_dates,
    )
    assert saved["output"] == json.loads(json.dumps(output))
    sidecar = json.loads((tmp_path / variant / "reuse_v02_entry_context.json").read_text())[
        "payload"]
    assert "not executed" in sidecar["label"]
    assert len(sidecar["records"]) == proof["posthoc_v02"]["record_count"]
    assert sidecar["records"][0]["v02_posthoc_asof"]["bundle_sha256"] == (
        plan.context_archive_sha256)
    reread_proof, reread_sidecar = verify_saved_reuse(
        plan, row, saved["output"], tmp_path / variant,
    )
    assert reread_proof == proof and reread_sidecar == sidecar


def test_only_exact_two_partial_controls_are_eligible(tmp_path, reference):
    plan, rows, context = reference
    assert try_reuse_partial("a" * 64, "b" * 64, plan,
                             SimpleNamespace(variant_id="MCB002"), tmp_path, context) is None
    assert try_reuse_partial("a" * 64, "b" * 64, plan,
                             rows["MCB001"].model_copy(update={"quantity": 6}),
                             tmp_path, context) is None
    assert not (tmp_path / "MCB002").exists()


def test_resume_rejects_tampered_posthoc_sidecar(tmp_path, reference):
    plan, rows, context = reference
    row = rows["MCB001"]
    output, _proof = try_reuse_partial("a" * 64, "b" * 64, plan, row, tmp_path, context)
    path = tmp_path / row.variant_id / "reuse_v02_entry_context.json"
    sealed = json.loads(path.read_text())
    sealed["payload"]["records"][0]["stream"] = "funded"
    path.write_text(json.dumps(sealed), encoding="utf-8")
    with pytest.raises(PermissionError, match="sealed reuse payload changed"):
        verify_saved_reuse(plan, row, output, tmp_path / row.variant_id)
