"""Correction reuse never changes economics or erases impact witnesses."""
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from alpha_lab.propsim.funded import mffu_batch_run as batch
from alpha_lab.propsim.funded.full_range_batch import save_checkpoint
from alpha_lab.propsim.funded.mffu_batch_correction_reuse import (
    _original_output,
    collect_print_receipts,
    correction_reuse_annotations,
    economic_output_sha256,
    impact_events,
    try_reuse_correction,
    verify_saved_correction_reuse,
)


def test_reuse_hash_retains_economic_rows_and_ignores_only_dispatch_metadata():
    original = {"dispatch": {"plan_id": "original"}, "seconds": 100,
                "pairs": {"myfundedfutures": {"ledger": {"net_cash_cents": -12500}}},
                "policy_decisions": {"funded": []}, "strategy_trades_no_account": []}
    reused = deepcopy(original)
    reused["dispatch"] = {"plan_id": "corrected"}
    reused["original_dispatch"] = original["dispatch"]
    reused["correction_reuse"] = {"status": "reused_nonimpact"}
    assert economic_output_sha256(reused) == economic_output_sha256(original)
    reused["pairs"]["myfundedfutures"]["ledger"]["net_cash_cents"] += 1
    assert economic_output_sha256(reused) != economic_output_sha256(original)


def test_impact_witnesses_keep_ordinary_and_funded_streams_separate():
    def decision(reasons):
        return {"event": "entry_admission", "reasons": reasons, "setup_id": "s",
                "envelope": {"ts_utc": "2026-05-13T10:56:00+00:00"}}
    original = {"policy_decisions": {
        "strategy": [decision(["ifsm_positive_london", "entry_family_not_profile",
                               "retest_trigger_unratified"])],
        "funded": [decision(["ifsm_positive_london"]),
                   decision(["daily_execution_cap", "causality_failed"]),
                   decision(["daily_execution_cap", "geometry_incomplete"]),
                   decision(["daily_execution_cap", "already_in_trade"])],
    }}
    events = impact_events(original)
    assert len(events) == 2
    assert [row["stream"] for row in events] == ["strategy", "funded"]


def test_original_child_uses_exact_mffu_dispatch_and_rejects_tampered_bytes(tmp_path):
    import hashlib
    row = SimpleNamespace(variant_id="MCB001", effective_section_config_hash="s",
                          effective_behavior_hash="b", model_dump=lambda **_kwargs: {"id": 1})
    output = {"dispatch": {"plan_id": "p", "approval_id": "a", "variant": {"id": 1}},
              "section_config_hash": "s", "effective_behavior_hash": "b"}
    path = tmp_path / "output.json"
    save_checkpoint(path, {"completed_dates": ["2026-01-13"], "output": output})
    receipt = {"original_plan_id": "p", "original_approval_id": "a", "intents": [{
        "variant_id": "MCB001", "original_output_path": str(path),
        "original_output_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
    assert _original_output(receipt, row, ("2026-01-13",))[1] == output
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(PermissionError, match="immutable child"):
        _original_output(receipt, row, ("2026-01-13",))


def test_correction_source_union_requires_identical_overlapping_receipts():
    cohort = {}
    item = {"path": "named/date/trades.parquet", "sha256": "a", "mtime_ns": 1}
    collect_print_receipts(cohort, [item, item.copy()])
    assert list(cohort.values()) == [item]
    with pytest.raises(PermissionError, match="disagree on an ordered-print receipt"):
        collect_print_receipts(cohort, [{**item, "sha256": "b"}])


def test_terminal_binding_accepts_all_64_correction_dispositions_and_requires_proofs(tmp_path):
    class Contract:
        def __init__(self, **fields):
            self.__dict__.update(fields)

        def model_dump(self, *, mode):
            assert mode == "json"
            return dict(self.__dict__)

    core = Contract(source="bound")
    receipt = {"schema": "ifsm_mffu_lifecycle_correction_receipt_v1",
               "corrected_core_source": core.model_dump(mode="json"),
               "original_result_id": "e" * 64}
    path = tmp_path / "correction_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    rows = tuple(Contract(variant_id=f"MCB{index:03d}") for index in range(1, 65))
    plan_id, approval_id = "a" * 64, "b" * 64
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    plan = Contract(variants=rows, core_root="C:/task/core", core_source=core,
                    runtime_source_file_sha256=source_hashes,
                    context_archive_sha256="d" * 64, source=Contract(scope="bound"),
                    task_b_scope=Contract(task="bound"))
    dispositions = [{"variant_id": row.variant_id,
                     "status": "reused_nonimpact" if index < 18 else
                     "replayed_equal" if index % 2 else "replayed_changed"}
                    for index, row in enumerate(rows)]
    saved = {"funded_comparison_plan_id": plan_id, "validation": {"passed": True},
             "mffu_batch": {"schema": batch.RESULT_SCHEMA, "plan": plan.model_dump(mode="json"),
                 "approval_id": approval_id, "correction_lineage": receipt,
                 "worker_dispatches": [batch._dispatch(plan_id, approval_id, plan, row)
                                       for row in rows],
                 "dispositions": dispositions, "reuse_proofs": {
                     row["variant_id"]: {"status": row["status"], "original_result_id": "e" * 64}
                     for row in dispositions[:18]}}}
    envelope = Contract(payload=Contract(funded_comparison_plan_id=plan_id,
        funded_comparison_approval_id=approval_id, engine_version=batch.ENGINE_VERSION,
        validation_passed=True))
    batch._verify_terminal_result_binding(saved, envelope, plan_id=plan_id,
                                          approval_id=approval_id, plan=plan)
    saved["mffu_batch"]["reuse_proofs"]["MCB001"]["original_result_id"] = "f" * 64
    with pytest.raises(PermissionError, match="exact reuse proof"):
        batch._verify_terminal_result_binding(saved, envelope, plan_id=plan_id,
                                              approval_id=approval_id, plan=plan)


def test_historical_posthoc_sidecar_copies_exact_bytes_and_never_becomes_execution(tmp_path):
    class Contract:
        def __init__(self, **fields):
            self.__dict__.update(fields)

        def model_dump(self, *, mode):
            return dict(self.__dict__)

    from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
    old_plan, old_approval, new_plan, new_approval = (letter * 64 for letter in "cdab")
    row = Contract(variant_id="MCB001", effective_section_config_hash="s",
                   effective_behavior_hash="b", effective_section_json=json.dumps({
                       "entry_context_policy": "off", "overhead_policy": "off"}))
    dates = ("2025-06-13", "2025-06-16")
    old_sidecar = tmp_path / "original_sidecar.json"
    sidecar_payload = {"schema": "ifsm_mffu_posthoc_v02_entry_context_not_executed_v1",
                       "variant_id": "MCB001", "context_archive_sha256": "f" * 64,
                       "records": [{"historical_trade_id": "t", "stream": "funded"}]}
    save_checkpoint(old_sidecar, sidecar_payload)
    original = {"dispatch": {"plan_id": old_plan, "approval_id": old_approval,
                              "variant": row.model_dump(mode="json")},
                "section_config_hash": "s", "effective_behavior_hash": "b",
                "reuse": {"status": "compatible_reused"}, "pairs": {"cash": -12500},
                "policy_decisions": {"funded": []}, "completed_dates": list(dates)}
    old_child = tmp_path / "original_child.json"
    save_checkpoint(old_child, {"completed_dates": list(dates), "output": original})
    state_root = tmp_path / "state"
    receipt = {"schema": "ifsm_mffu_lifecycle_correction_receipt_v1",
        "original_plan_id": old_plan, "original_approval_id": old_approval,
        "original_result_id": "e" * 64, "original_core_source": {"source": "original"},
        "corrected_core_source": {"source": "corrected"}, "state_root": str(state_root),
        "intents": [{"variant_id": "MCB001", "original_output_path": str(old_child),
            "original_output_file_sha256": hashlib.sha256(old_child.read_bytes()).hexdigest(),
            "disposition": "reused_nonimpact", "historical_posthoc": {
                "original_path": str(old_sidecar),
                "file_sha256": hashlib.sha256(old_sidecar.read_bytes()).hexdigest(),
                "payload_sha256": canonical_contract_sha256(sidecar_payload),
                "record_count": 1, "executed": False}}]}
    receipt_path = tmp_path / "correction_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    plan = Contract(core_root="C:/task/core", core_source=Contract(source="corrected"),
                    runtime_source_file_sha256={
                        str(receipt_path): hashlib.sha256(receipt_path.read_bytes()).hexdigest()},
                    context_archive_sha256="f" * 64, task_b_scope=Contract(task="bound"),
                    source=Contract(warmup_dates=dates[:1], evaluation_dates=dates[1:]))
    output, proof = try_reuse_correction(
        new_plan, new_approval, plan, row, state_root / new_plan / "workers",
    )
    assert proof["historical_posthoc"]["executed"] is False
    copied = tmp_path / "state" / new_plan / "workers/MCB001/reuse_v02_entry_context.json"
    assert copied.read_bytes() == old_sidecar.read_bytes()
    annotations, paths = correction_reuse_annotations(plan, row, output)
    assert paths == {"historical_posthoc": str(copied)}
    assert annotations[0]["status"] == "posthoc_v02_not_executed"
    assert output["historical_reuse"] == original["reuse"]
    assert verify_saved_correction_reuse(plan, row, output) == proof
    output["correction_reuse"]["historical_posthoc"]["executed"] = True
    with pytest.raises(PermissionError, match="provenance differs"):
        verify_saved_correction_reuse(plan, row, output)
