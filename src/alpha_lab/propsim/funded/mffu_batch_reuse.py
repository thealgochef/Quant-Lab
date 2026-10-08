"""Exact, read-only C02/C01 partial-exit reuse for the MFFU context batch.

The old financial rows remain historical executions.  A separately saved v02
entry-context sidecar is a posthoc lookup, never an executed v02 decision.
Every source or semantic mismatch declines reuse so the managed runner replays
that child under its own frozen dispatch.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
from alpha_lab.propsim.funded.comparison_plan import (
    APPROVAL_STORE,
    PLAN_STORE,
    RESULT_STORE,
    FundedComparisonApprovalEnvelope,
    FundedComparisonResultEnvelope,
)
from alpha_lab.propsim.funded.comparison_run import pair_id_for
from alpha_lab.propsim.funded.core_identity import source_identity_at
from alpha_lab.propsim.funded.full_range_batch import (
    FullRangeBatchPlanEnvelope,
    _assert_terminal_result_binding,
    load_checkpoint,
    save_checkpoint,
)
from alpha_lab.propsim.funded.full_range_batch import (
    _dispatch as old_dispatch,
)
from alpha_lab.propsim.funded.mffu_batch_plan import REFERENCE_RESULT_ID, REPO_ROOT
from alpha_lab.propsim.funded.position_walk import FEE_ROUNDING_POLICY_ID, fill_cost_cents
from alpha_lab.propsim.funded.profiles import INSTRUMENTS, MYFUNDEDFUTURES_PROFILE
from alpha_lab.propsim.funded.result import canonical_json, result_sha256

__all__ = ["try_reuse_partial", "verify_saved_reuse"]

REUSE_SCHEMA = "ifsm_mffu_verified_partial_reuse_v1"
SIDECAR_SCHEMA = "ifsm_mffu_posthoc_v02_entry_context_not_executed_v1"
_OLD_BY_NEW = {"MCB001": "C02", "MCB025": "C01"}
_OLD_CORE_PATCH = "028b3ce4a012cb43755cdbe82b9eeceb63da45c9aadb1f77d64e07b53710d3fb"
_NEW_CORE_PATCH = "d3bd4ce559756f014a745c6fa0927dc29af276931ce9b738dff4b4011b4edf93"
_CORE_BASE = "709487bc85ef82a259297c4ad6f3f198f4cabbcd"
_NEUTRAL_CONTEXT = {
    "entry_context_policy": "off",
    "overhead_policy": "off",
    "opposing_distance_policy": "fixed_v1",
    "ifsm_context_policy_version": "mq_eod_asof_nominal_2200_chicago_v01",
}
_UNCHANGED_ECONOMIC_FILES = (
    "src/alpha_lab/propsim/funded/profiles.py",
    "src/alpha_lab/propsim/funded/clock.py",
    "src/alpha_lab/propsim/funded/print_minutes.py",
    "src/alpha_lab/propsim/funded/price_evidence.py",
    "src/alpha_lab/propsim/funded/positions.py",
    "src/alpha_lab/propsim/funded/result.py",
    "src/alpha_lab/agents/data_infra/ifvg/menthorq_levels.py",
)
_CHANGED_EXECUTION_FILES = (
    "pair_engine.py", "pair_ledger.py", "position_walk.py", "strategy_driver.py",
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(plan) -> dict[str, str]:
    return {str(Path(path).resolve()): digest
            for path, digest in dict(plan.runtime_source_file_sha256).items()}


def _semantic_match(row, old_row) -> bool:
    if (row.variant_id not in _OLD_BY_NEW or row.base_reference != old_row.name
            or row.base_section_config_hash != old_row.effective_section_config_hash
            or (row.quantity_policy, row.quantity, tuple(row.possible_quantities),
                row.instrument, row.cost_per_contract_mills,
                row.fee_rounding_policy, row.firm_key)
            != ("Q10", 10, (10,), "micro", 514,
                FEE_ROUNDING_POLICY_ID, "myfundedfutures")
            or (old_row.instrument, old_row.quantity, old_row.cost_per_contract_mills)
            != ("micro", 10, 514)
            or (fill_cost_cents(10, 514), fill_cost_cents(5, 514)) != (514, 257)
            or row.exit_policy != old_row.exit_policy
            or row.exit_policy != "scale_out_half_breakeven_hold_to_close_v1"):
        return False
    intent = json.loads(row.intent_json)
    if (intent.get("daily_cap"), intent.get("entry_context"), intent.get("exit"),
            intent.get("sizing"), intent.get("overhead"), intent.get("geometry")) != (
            "U", "F0", "XP", "Q10", "O0", "G0"):
        return False
    old = json.loads(old_row.effective_section_json)
    new = json.loads(row.effective_section_json)
    if any(old.get(key) is not None or new.get(key) != value
           for key, value in _NEUTRAL_CONTEXT.items()):
        return False
    excluded = {*_NEUTRAL_CONTEXT, "profile_name"}
    return ({key: value for key, value in old.items() if key not in excluded}
            == {key: value for key, value in new.items() if key not in excluded})


def _source_parity(plan, reference) -> dict | None:
    old_core, new_core = reference.core_source, plan.core_source
    if (old_core.base_commit, old_core.patch_sha256,
            new_core.base_commit, new_core.patch_sha256) != (
            _CORE_BASE, _OLD_CORE_PATCH, _CORE_BASE, _NEW_CORE_PATCH):
        return None
    if (source_identity_at(Path(reference.core_root))["patch_sha256"] != _OLD_CORE_PATCH
            or source_identity_at(Path(plan.core_root))["patch_sha256"] != _NEW_CORE_PATCH):
        return None
    old_hashes, new_hashes = _source_hashes(reference), _source_hashes(plan)
    stable = {}
    for relative in _UNCHANGED_ECONOMIC_FILES:
        path = (REPO_ROOT / relative).resolve()
        key = str(path)
        actual = _sha_file(path)
        if old_hashes.get(key) != actual or new_hashes.get(key) != actual:
            return None
        stable[relative] = actual
    changed = {}
    for name in _CHANGED_EXECUTION_FILES:
        paths = [key for key in old_hashes if Path(key).name == name]
        if len(paths) != 1 or paths[0] not in new_hashes:
            return None
        changed[name] = {"old_sha256": old_hashes[paths[0]],
                         "new_sha256": new_hashes[paths[0]]}
    return {
        "old_core": old_core.model_dump(mode="json"),
        "new_core": new_core.model_dump(mode="json"),
        "identical_economic_source_sha256": stable,
        "changed_conditional_execution_source_sha256": changed,
        "passive_behavior_evidence": [
            "Strategy-Core tests/test_ifsm_mffu_policy.py::"
            "test_v02_context_with_all_controls_off_preserves_fixed_economics",
            "Quant-Lab tests/propsim/funded/test_full_range_passive_invariance.py::"
            "test_passive_context_preserves_actual_funded_core_events",
        ],
        "scope": "U/F0/XP/Q10/O0/G0 only; historical trade and seed identities retained",
    }


def _verify_reference(plan, old_batch_id: str):
    store = Path(plan.reference_store_root)
    old_plan = load_verified_envelope(
        store, PLAN_STORE, plan.reference_plan_id, FullRangeBatchPlanEnvelope,
    )
    plan_path = store / PLAN_STORE / plan.reference_plan_id / "envelope.json"
    if _sha_file(plan_path) != plan.reference_plan_envelope_sha256:
        raise PermissionError("repaired reference plan bytes changed")
    old = old_plan.payload
    result_envelope = load_verified_envelope(
        store, RESULT_STORE, REFERENCE_RESULT_ID, FundedComparisonResultEnvelope,
    )
    result_path = store / RESULT_STORE / REFERENCE_RESULT_ID / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    payload = result_envelope.payload
    if (plan.reference_result_id != REFERENCE_RESULT_ID
            or result_sha256(result) != payload.result_json_sha256
            or not payload.validation_passed or not result["validation"]["passed"]
            or result["configurations_completed"] != 6
            or result["full_range_batch"]["failed_configurations"]):
        raise PermissionError("repaired reference result is not the verified six-row finish")
    approval = load_verified_envelope(
        store, APPROVAL_STORE, payload.funded_comparison_approval_id,
        FundedComparisonApprovalEnvelope,
    )
    if approval.payload.funded_comparison_plan_id != plan.reference_plan_id:
        raise PermissionError("repaired reference approval differs from its plan")
    if _assert_terminal_result_binding(
        result, result_envelope, plan_id=plan.reference_plan_id,
        approval_id=payload.funded_comparison_approval_id, plan=old,
    ) != "Completed":
        raise PermissionError("repaired reference result is not terminally complete")
    if (plan.source != old.source or plan.calendar_sha256 != old.calendar_sha256
            or plan.task_b_plan_id != old.task_b_plan_id
            or plan.task_b_scope != old.task_b_scope
            or plan.prepared_registration_ids != old.prepared_registration_ids
            or plan.processing != old.processing
            or plan.execution_model != old.execution_model
            or tuple(plan.firm_profiles) != (MYFUNDEDFUTURES_PROFILE,)
            or MYFUNDEDFUTURES_PROFILE not in old.firm_profiles):
        return None
    old_row = next(item for item in old.configurations if item.batch_id == old_batch_id)
    old_aid = payload.funded_comparison_approval_id
    expected_dispatch = old_dispatch(plan.reference_plan_id, old_aid, old, old_row)
    dates = old.source.warmup_dates + old.source.evaluation_dates
    old_worker_path = (store.parent / "study_jobs" / plan.reference_plan_id / "workers"
                       / old_batch_id / "output.json")
    saved = load_checkpoint(old_worker_path,
                            dispatch_sha256=canonical_contract_sha256(expected_dispatch),
                            dates=dates)
    if saved is None or tuple(saved["completed_dates"]) != dates:
        raise PermissionError("repaired reference worker is missing or incomplete")
    output = saved["output"]
    expected_pair = pair_id_for(old_row.name, MYFUNDEDFUTURES_PROFILE.firm_key)
    summary = result["summaries_cents"][expected_pair]
    ledger = output["pairs"][MYFUNDEDFUTURES_PROFILE.firm_key]["ledger"]
    saved_prints = {
        item["path"]: item for item in result["full_range_batch"]["print_source_receipts"]
    }
    if (output["dispatch"] != expected_dispatch
            or output["dispatch_sha256"] != saved["dispatch_sha256"]
            or tuple(output["completed_dates"]) != dates
            or (output["batch_id"], output["configuration"],
                output["section_config_hash"], output["exit_policy"])
            != (old_batch_id, old_row.name, old_row.effective_section_config_hash,
                old_row.exit_policy)
            or output["sizing"]["instrument"] != "micro"
            or output["sizing"]["quantity"] != 10
            or output["sizing"]["cost_per_contract_mills"] != 514
            or ledger["pair_id"] != expected_pair
            or ledger["receipts"] != summary["payouts_received_cents"]
            or ledger["costs"] != summary["account_costs_cents"]
            or ledger["receipts"] - ledger["costs"] != summary["net_cash_earned_cents"]
            or len(ledger["trades"]) != summary["trades_taken"]):
        raise PermissionError("repaired worker output differs from saved result economics")
    if any(saved_prints.get(item["path"]) != item
           for item in output["print_source_receipts"]):
        raise PermissionError("repaired worker print receipts differ from saved result")
    return old_row, output, {
        "result_id": REFERENCE_RESULT_ID,
        "result_envelope_file_sha256": _sha_file(
            store / RESULT_STORE / REFERENCE_RESULT_ID / "envelope.json"),
        "result_json_sha256": payload.result_json_sha256,
        "reference_plan_id": plan.reference_plan_id,
        "plan_envelope_file_sha256": _sha_file(plan_path),
        "reference_approval_id": old_aid,
        "old_worker_checkpoint_file_sha256": _sha_file(old_worker_path),
        "old_worker_dispatch_sha256": saved["dispatch_sha256"],
        "old_batch_id": old_batch_id,
        "old_configuration": old_row.name,
        "saved_receipts_cents": ledger["receipts"],
        "saved_account_costs_cents": ledger["costs"],
        "saved_net_cash_cents": ledger["receipts"] - ledger["costs"],
        "saved_funded_trades": len(ledger["trades"]),
        "saved_print_receipt_count": len(output["print_source_receipts"]),
        "date_count": len(dates),
    }


def _relabel(value, old_name: str, old_pair: str, new_name: str, new_pair: str):
    """Change only presentation/account namespaces, never trade UUIDs or v1 context."""
    if isinstance(value, dict):
        return {key: _relabel(item, old_name, old_pair, new_name, new_pair)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_relabel(item, old_name, old_pair, new_name, new_pair) for item in value]
    if isinstance(value, str):
        if value == old_name:
            return new_name
        if value == old_pair or value.startswith(old_pair + "#"):
            return new_pair + value[len(old_pair):]
    return value


def _entry_sidecar(output: dict, row, plan, context_index, reference: dict) -> dict:
    records = []
    streams = (
        ("strategy", output["strategy_trades_no_account"], "entry_ts_utc", "trade_id"),
        ("funded", output["pairs"]["myfundedfutures"]["ledger"]["trades"],
         "entry_ns", "strategy_trade_id"),
    )
    for stream, trades, time_key, id_key in streams:
        for trade in trades:
            raw_time = trade[time_key]
            if time_key == "entry_ns":
                seconds, nanos = divmod(raw_time, 1_000_000_000)
                # Truncate submicrosecond precision. Rounding could select a
                # report that was still ineligible at the recorded print.
                ts = datetime.fromtimestamp(seconds, UTC) + timedelta(
                    microseconds=nanos // 1000,
                )
            else:
                ts = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
            if ts.tzinfo is None or ts.utcoffset() != UTC.utcoffset(ts):
                raise PermissionError("saved entry timestamp is not aware UTC")
            records.append({
                "stream": stream, "historical_trade_id": trade[id_key],
                "entry_ts_utc": ts.isoformat(),
                "v02_posthoc_asof": context_index.snapshot(ts),
            })
    return {
        "schema": SIDECAR_SCHEMA,
        "label": "Posthoc v02 entry-context lookup; not executed v02 policy evidence",
        "variant_id": row.variant_id,
        "reference_result_id": reference["result_id"],
        "reference_worker_checkpoint_file_sha256": reference[
            "old_worker_checkpoint_file_sha256"],
        "context_archive_sha256": plan.context_archive_sha256,
        "historical_entry_context_retained": "MenthorQ v1 passive annotation",
        "records": records,
    }


def try_reuse_partial(plan_id, approval_id, plan, row, work_root, context_index):
    """Return ``(adapted_output, proof)`` or decline; never replay market data.

    ``work_root`` is the managed batch's workers directory.  A successful
    decision saves its sealed output, proof and separate posthoc annotation in
    the variant's worker directory, ready for ordinary resume validation.
    """
    old_batch_id = _OLD_BY_NEW.get(row.variant_id)
    if old_batch_id is None:
        return None
    if (getattr(context_index, "bundle_sha256", None) != plan.context_archive_sha256
            or getattr(context_index, "cutoff_utc", None) != datetime.fromisoformat(
                plan.source.cutoff_utc.replace("Z", "+00:00"))):
        return None
    verified = _verify_reference(plan, old_batch_id)
    if verified is None:
        return None
    old_row, prior, reference = verified
    if not _semantic_match(row, old_row):
        return None
    source_parity = _source_parity(plan, load_verified_envelope(
        Path(plan.reference_store_root), PLAN_STORE, plan.reference_plan_id,
        FullRangeBatchPlanEnvelope,
    ).payload)
    if source_parity is None:
        return None

    from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection, ifvg_profile_hash

    from alpha_lab.propsim.funded.comparison_describe import describe_section
    from alpha_lab.propsim.funded.mffu_batch_run import FINAL_SCHEMA, _dispatch

    section = IfvgSmcSection.model_validate_json(row.effective_section_json)
    if ifvg_profile_hash(section) != row.effective_section_config_hash:
        raise PermissionError("new section hash differs from frozen row")
    new_dispatch = _dispatch(plan_id, approval_id, plan, row)
    old_pair = pair_id_for(old_row.name, "myfundedfutures")
    new_pair = pair_id_for(row.name, "myfundedfutures")
    output = copy.deepcopy(prior)
    output["pairs"] = {"myfundedfutures": output["pairs"]["myfundedfutures"]}
    output["daily_activity"] = [item for item in output["daily_activity"]
                                if item["stream"] == "strategy"
                                or item.get("firm_key") == "myfundedfutures"]
    output = _relabel(output, old_row.name, old_pair, row.name, new_pair)
    output.update({
        "configuration": row.name,
        "batch_id": row.batch_id,
        "display_name": row.display_name,
        "axes": json.loads(row.intent_json),
        "settings_plain": describe_section(section),
        "section_config_hash": row.effective_section_config_hash,
        "effective_behavior_hash": row.effective_behavior_hash,
        "exit_policy": row.exit_policy,
        "sizing": {
            "instrument": "micro", "instrument_label": INSTRUMENTS["micro"].label,
            "quantity": 10, "quantity_policy": "Q10", "possible_quantities": (10,),
            "tick_value_cents": INSTRUMENTS["micro"].tick_value_cents,
            "cost_per_contract_mills": 514,
            "fee_rounding_policy": row.fee_rounding_policy,
        },
        "dispatch": new_dispatch,
        "dispatch_sha256": canonical_contract_sha256(new_dispatch),
        "policy_decisions": {"strategy": [], "funded": []},
        "seconds": 0.0,
    })
    output["reference"] = {
        **output["reference"],
        "equivalent": None,
        "compared_with": "Prior immutable C01/C02 execution reused under semantic-neutrality "
                         "and synthetic passive-parity evidence; no new 263-day replay. "
                         "The v02 context is a separate posthoc annotation.",
    }
    worker_dir = Path(work_root) / row.variant_id
    sidecar = _entry_sidecar(output, row, plan, context_index, reference)
    sidecar_path = worker_dir / "reuse_v02_entry_context.json"
    sidecar_sha = canonical_contract_sha256(sidecar)
    output["reuse"] = {
        "schema": REUSE_SCHEMA,
        "status": "compatible_reused",
        "reference_result_id": REFERENCE_RESULT_ID,
        "reference_batch_id": old_batch_id,
        "reference_worker_checkpoint_file_sha256": reference[
            "old_worker_checkpoint_file_sha256"],
        "historical_trade_and_seed_ids_retained": True,
        "historical_entry_context_version": "MenthorQ v1 passive",
        "posthoc_v02_sidecar_path": str(sidecar_path),
        "posthoc_v02_sidecar_sha256": sidecar_sha,
        "posthoc_v02_is_executed": False,
    }
    proof = {
        "schema": REUSE_SCHEMA,
        "status": "compatible_reused",
        "variant_id": row.variant_id,
        "new_plan_id": plan_id,
        "new_approval_id": approval_id,
        "new_dispatch_sha256": output["dispatch_sha256"],
        "reference": reference,
        "financial_checks": {
            "worker_receipts_equal_saved_result": True,
            "worker_account_costs_equal_saved_result": True,
            "worker_net_cash_equal_saved_result": True,
            "worker_trade_count_equal_saved_result": True,
            "worker_print_receipts_in_saved_result": True,
            "ten_and_five_micro_fills_still_exact_cents": True,
        },
        "section_semantic_delta": {
            "only_added_neutral_context_fields": _NEUTRAL_CONTEXT,
            "generated_profile_name_changed": True,
            "old_section_config_hash": old_row.effective_section_config_hash,
            "new_section_config_hash": row.effective_section_config_hash,
        },
        "economic_identity": {
            "task_b_source_sha256": canonical_contract_sha256(plan.source),
            "task_b_manifest_sha256": plan.source.task_b_plan_manifest_sha256,
            "warmup_dates_sha256": canonical_contract_sha256(plan.source.warmup_dates),
            "evaluation_dates_sha256": canonical_contract_sha256(
                plan.source.evaluation_dates),
            "warmup_date_count": len(plan.source.warmup_dates),
            "evaluation_date_count": len(plan.source.evaluation_dates),
            "cutoff_utc": plan.source.cutoff_utc,
            "calendar_sha256": plan.calendar_sha256,
            "task_b_plan_id": plan.task_b_plan_id,
            "firm_profile_sha256": canonical_contract_sha256(MYFUNDEDFUTURES_PROFILE),
            "processing_sha256": canonical_contract_sha256(plan.processing),
            "execution_model_sha256": canonical_contract_sha256(plan.execution_model),
            "instrument": "micro", "quantity": 10,
            "tick_value_cents": INSTRUMENTS["micro"].tick_value_cents,
            "cost_per_contract_mills": 514,
            "fee_rounding_policy": row.fee_rounding_policy,
            "ten_micro_entry_fee_cents": fill_cost_cents(10, 514),
            "five_micro_exit_fee_cents": fill_cost_cents(5, 514),
        },
        "source_parity": source_parity,
        "posthoc_v02": {
            "path": str(sidecar_path), "payload_sha256": sidecar_sha,
            "record_count": len(sidecar["records"]), "executed": False,
        },
        "adapted_output_sha256": canonical_contract_sha256(output),
        "limitations": [
            "Reused trade UUIDs and final seed hashes identify the historical Core source.",
            "Historical entry_context fields remain v1; v02 rows are posthoc only.",
            "No 263-day strategy replay was performed for this compatible control.",
        ],
    }
    save_checkpoint(sidecar_path, sidecar)
    save_checkpoint(worker_dir / "reuse_proof.json", proof)
    save_checkpoint(worker_dir / "output.json", {
        "schema": FINAL_SCHEMA,
        "dispatch_sha256": output["dispatch_sha256"],
        "completed_dates": output["completed_dates"],
        "output": output,
    })
    return output, proof


def _load_sealed_payload(path: Path) -> dict:
    """Read the same canonical-JSON envelope written by ``save_checkpoint``."""
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    expected = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    if envelope["sha256"] != expected:
        raise PermissionError(f"sealed reuse payload changed: {path}")
    return payload


def verify_saved_reuse(plan, row, output, worker_dir):
    """Recheck saved output, immutable origin, proof and posthoc sidecar on resume."""
    worker_dir = Path(worker_dir)
    old_batch_id = _OLD_BY_NEW.get(row.variant_id)
    if old_batch_id is None or output.get("reuse", {}).get("status") != "compatible_reused":
        raise PermissionError("saved row is not an eligible partial reuse")
    dispatch = output["dispatch"]
    if (output["batch_id"] != row.variant_id
            or output["configuration"] != row.name
            or output["section_config_hash"] != row.effective_section_config_hash
            or output["effective_behavior_hash"] != row.effective_behavior_hash
            or output["sizing"]["fee_rounding_policy"] != row.fee_rounding_policy
            or dispatch["variant"] != row.model_dump(mode="json")
            or output["dispatch_sha256"] != canonical_contract_sha256(dispatch)):
        raise PermissionError("saved reused row differs from its frozen variant")
    dates = plan.source.warmup_dates + plan.source.evaluation_dates
    saved = load_checkpoint(
        worker_dir / "output.json", dispatch_sha256=output["dispatch_sha256"],
        dates=dates,
    )
    if (saved is None or canonical_json(saved["output"]) != canonical_json(output)
            or tuple(saved["completed_dates"]) != dates):
        raise PermissionError("saved reused output checkpoint changed")
    proof = _load_sealed_payload(worker_dir / "reuse_proof.json")
    sidecar = _load_sealed_payload(worker_dir / "reuse_v02_entry_context.json")
    reuse = output["reuse"]
    sidecar_path = worker_dir / "reuse_v02_entry_context.json"
    verified = _verify_reference(plan, old_batch_id)
    if verified is None:
        raise PermissionError("reused origin no longer matches financial plan")
    old_row, _old_output, reference = verified
    old_plan = load_verified_envelope(
        Path(plan.reference_store_root), PLAN_STORE, plan.reference_plan_id,
        FullRangeBatchPlanEnvelope,
    ).payload
    source_parity = _source_parity(plan, old_plan)
    if (source_parity is None or not _semantic_match(row, old_row)
            or proof.get("schema") != REUSE_SCHEMA
            or proof.get("variant_id") != row.variant_id
            or proof.get("reference") != reference
            or proof.get("source_parity") != source_parity
            or proof.get("new_plan_id") != dispatch["plan_id"]
            or proof.get("new_approval_id") != dispatch["approval_id"]
            or proof.get("new_dispatch_sha256") != output["dispatch_sha256"]
            or proof.get("adapted_output_sha256") != canonical_contract_sha256(output)):
        raise PermissionError("saved reuse proof differs from current origin or output")
    if (proof["economic_identity"].get("fee_rounding_policy") !=
            FEE_ROUNDING_POLICY_ID
            or proof["economic_identity"].get("ten_micro_entry_fee_cents") != 514
            or proof["economic_identity"].get("five_micro_exit_fee_cents") != 257
            or proof["financial_checks"].get(
                "ten_and_five_micro_fills_still_exact_cents") is not True):
        raise PermissionError("saved reuse fee equivalence proof changed")
    if (sidecar.get("schema") != SIDECAR_SCHEMA
            or sidecar.get("variant_id") != row.variant_id
            or sidecar.get("reference_result_id") != REFERENCE_RESULT_ID
            or sidecar.get("context_archive_sha256") != plan.context_archive_sha256
            or sidecar.get("reference_worker_checkpoint_file_sha256") != reference[
                "old_worker_checkpoint_file_sha256"]
            or canonical_contract_sha256(sidecar) != reuse["posthoc_v02_sidecar_sha256"]
            or reuse.get("posthoc_v02_sidecar_path") != str(sidecar_path)
            or reuse.get("posthoc_v02_is_executed") is not False
            or proof["posthoc_v02"]["payload_sha256"] !=
            reuse["posthoc_v02_sidecar_sha256"]
            or proof["posthoc_v02"]["record_count"] != len(sidecar["records"])):
        raise PermissionError("saved v02 posthoc annotation differs from reuse proof")
    expected = [
        ("strategy", trade["trade_id"])
        for trade in output["strategy_trades_no_account"]
    ] + [
        ("funded", trade["strategy_trade_id"])
        for trade in output["pairs"]["myfundedfutures"]["ledger"]["trades"]
    ]
    observed = [(item["stream"], item["historical_trade_id"])
                for item in sidecar["records"]]
    if observed != expected or any(
        item["v02_posthoc_asof"].get("bundle_sha256") != plan.context_archive_sha256
        for item in sidecar["records"]
    ):
        raise PermissionError("saved posthoc entry roster differs from reused trades")
    return proof, sidecar
