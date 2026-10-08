"""Verified reuse of unchanged original MFFU children in a correction namespace.

Only a source-bound correction receipt enables this path. Historical children,
trade IDs, account ledgers and driver seeds stay byte-for-byte represented in
the reused output. New dispatch metadata explicitly identifies reuse; a child
is never described as replayed merely because it appears in a successor batch.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.propsim.funded.full_range_batch import save_checkpoint
from alpha_lab.propsim.funded.result import canonical_json

RECEIPT_NAME = "correction_receipt.json"
COMPLETE_CHILD_STATUSES = frozenset({
    "compatible_reused", "newly_completed", "reused_nonimpact",
    "reused_equivalent_after_verification", "replayed_changed", "replayed_equal",
})
_POLICY = frozenset({"ifsm_early_positive", "ifsm_positive_london", "ifsm_overhead_gex",
                     "daily_execution_cap"})
_METADATA = frozenset({"dispatch", "dispatch_sha256", "seconds", "reuse", "historical_reuse",
                       "correction_reuse", "original_dispatch"})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correction_receipt(plan) -> dict | None:
    paths = [(Path(path), digest) for path, digest in plan.runtime_source_file_sha256.items()
             if Path(path).name == RECEIPT_NAME]
    if not paths:
        return None
    if len(paths) != 1 or _sha(paths[0][0]) != paths[0][1]:
        raise PermissionError("correction receipt binding failed")
    receipt = json.loads(paths[0][0].read_text(encoding="utf-8"))
    if receipt["schema"] != "ifsm_mffu_lifecycle_correction_receipt_v1":
        raise PermissionError("unsupported correction receipt")
    if receipt["corrected_core_source"] != plan.core_source.model_dump(mode="json"):
        raise PermissionError("correction receipt Core source differs from plan")
    return receipt


def economic_output_sha256(output: dict) -> str:
    return hashlib.sha256(canonical_json({key: value for key, value in output.items()
                                        if key not in _METADATA}).encode()).hexdigest()


def economic_projection(output: dict) -> dict:
    """Actual ordinary trades and complete funded account/cash records."""
    return {"ordinary_trades": output["strategy_trades_no_account"],
            "funded_ledgers": {key: pair["ledger"] for key, pair in output["pairs"].items()}}


def collect_print_receipts(cohort: dict, receipts) -> None:
    """Union exact source receipts; inconsistent references fail before dispatch."""
    for receipt in receipts:
        path = receipt["path"]
        if path in cohort and cohort[path] != receipt:
            raise PermissionError("correction children disagree on an ordered-print receipt")
        cohort[path] = receipt


def impact_events(output: dict) -> list[dict]:
    events = []
    for stream, decisions in output.get("policy_decisions", {}).items():
        for decision in decisions:
            reasons = set(decision.get("reasons") or ())
            if (decision["event"] == "entry_admission" and reasons & _POLICY
                        and reasons - _POLICY
                        and not reasons & {"geometry_incomplete", "already_in_trade"}):
                events.append({"stream": stream, "setup_id": decision.get("setup_id"),
                               "timestamp": decision.get("envelope", {}).get("ts_utc"),
                               "reasons": sorted(reasons)})
    return events


def _original_output(receipt: dict, row, dates: tuple[str, ...]) -> tuple[dict, dict]:
    records = [item for item in receipt["intents"] if item["variant_id"] == row.variant_id]
    if len(records) != 1:
        raise PermissionError("correction receipt does not account for this intent")
    record = records[0]
    path = Path(record["original_output_path"])
    if _sha(path) != record["original_output_file_sha256"]:
        raise PermissionError("original immutable child bytes changed")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    if hashlib.sha256(canonical_json(payload).encode()).hexdigest() != envelope["sha256"]:
        raise PermissionError("original child checksum failed")
    output = payload["output"]
    dispatch = output["dispatch"]
    if (dispatch["plan_id"] != receipt["original_plan_id"]
            or dispatch["approval_id"] != receipt["original_approval_id"]
            or dispatch["variant"] != row.model_dump(mode="json")
            or tuple(payload["completed_dates"]) != dates
            or output["section_config_hash"] != row.effective_section_config_hash
            or output["effective_behavior_hash"] != row.effective_behavior_hash):
        raise PermissionError("original child differs from declared intent or date scope")
    return record, output


def try_reuse_correction(plan_id: str, approval_id: str, plan, row, work_root: Path):
    receipt = correction_receipt(plan)
    if receipt is None:
        return None
    dates = plan.source.warmup_dates + plan.source.evaluation_dates
    record, original = _original_output(receipt, row, dates)
    if record["disposition"] not in {"reused_nonimpact", "reused_equivalent_after_verification"}:
        return None
    section = json.loads(row.effective_section_json)
    neutral = (section["entry_context_policy"] == "off" and section["overhead_policy"] == "off"
               and section.get("max_executed_trades_per_day") is None)
    if record["disposition"] == "reused_nonimpact" and not neutral:
        raise PermissionError("disabled-switch reuse witness does not match actual section")
    if not neutral and impact_events(original):
        raise PermissionError("an affected lifecycle child cannot be reused")
    from alpha_lab.propsim.funded.mffu_batch_run import FINAL_SCHEMA, _dispatch

    output = copy.deepcopy(original)
    original_reuse = output.pop("reuse", None)
    if original_reuse is not None:
        output["historical_reuse"] = original_reuse
    output["original_dispatch"] = output["dispatch"]
    output["dispatch"] = _dispatch(plan_id, approval_id, plan, row)
    output["dispatch_sha256"] = canonical_contract_sha256(output["dispatch"])
    output["correction_reuse"] = {
        "schema": "ifsm_mffu_correction_child_reuse_v1", "status": record["disposition"],
        "original_result_id": receipt["original_result_id"],
        "original_plan_id": receipt["original_plan_id"],
        "original_output_file_sha256": record["original_output_file_sha256"],
        "economic_output_sha256": economic_output_sha256(original),
        "original_core_source": receipt["original_core_source"],
        "corrected_core_source": receipt["corrected_core_source"],
        "reason": (
            "New policy lifecycle action is unreachable with all switches disabled; "
            "future-gap check is unreachable in the completed-bar orchestrator."
            if neutral else "The complete original ordinary and funded policy records contain "
            "no mixed policy/non-policy lifecycle event; the corrected branch is unreachable."
        ),
    }
    sidecar_ref = record.get("historical_posthoc")
    if sidecar_ref is not None:
        source_path = Path(sidecar_ref["original_path"])
        if _sha(source_path) != sidecar_ref["file_sha256"]:
            raise PermissionError("original posthoc sidecar bytes changed")
        target = Path(work_root) / row.variant_id / "reuse_v02_entry_context.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and _sha(target) != sidecar_ref["file_sha256"]:
            raise PermissionError("correction posthoc sidecar already has different bytes")
        if not target.exists():
            shutil.copyfile(source_path, target)
        output["correction_reuse"]["historical_posthoc"] = {
            **sidecar_ref, "copied_path": str(target), "executed": False,
        }
    if economic_output_sha256(output) != economic_output_sha256(original):
        raise PermissionError("reuse changed original economic child content")
    save_checkpoint(Path(work_root) / row.variant_id / "output.json", {
        "schema": FINAL_SCHEMA, "dispatch_sha256": output["dispatch_sha256"],
        "completed_dates": list(dates), "output": output,
    })
    verify_saved_correction_reuse(plan, row, output)
    return output, output["correction_reuse"]


def verify_saved_correction_reuse(plan, row, output: dict) -> dict:
    receipt = correction_receipt(plan)
    if receipt is None:
        raise PermissionError("saved correction reuse lacks frozen receipt")
    record, original = _original_output(
        receipt, row, plan.source.warmup_dates + plan.source.evaluation_dates,
    )
    proof = output["correction_reuse"]
    if (proof["status"] != record["disposition"]
            or proof["original_result_id"] != receipt["original_result_id"]
            or proof["original_plan_id"] != receipt["original_plan_id"]
            or proof["original_core_source"] != receipt["original_core_source"]
            or proof["corrected_core_source"] != receipt["corrected_core_source"]
            or proof["original_output_file_sha256"] != record["original_output_file_sha256"]
            or proof["economic_output_sha256"] != economic_output_sha256(original)
            or economic_output_sha256(output) != proof["economic_output_sha256"]
            or output["original_dispatch"] != original["dispatch"]):
        raise PermissionError("saved correction reuse proof or economic content differs")
    correction_reuse_annotations(plan, row, output)
    return proof


def correction_reuse_annotations(plan, row, output: dict) -> tuple[list[dict], dict]:
    """Retained original v02 as-of annotations; never executed policy evidence."""
    receipt = correction_receipt(plan)
    if receipt is None:
        raise PermissionError("correction annotation lacks its source-bound receipt")
    record = next(item for item in receipt["intents"] if item["variant_id"] == row.variant_id)
    expected = record.get("historical_posthoc")
    actual = output.get("correction_reuse", {}).get("historical_posthoc")
    if expected is None:
        if actual is not None:
            raise PermissionError("unexpected historical posthoc annotation")
        return [], {}
    target = (Path(receipt["state_root"]) / output["dispatch"]["plan_id"] / "workers"
              / row.variant_id / "reuse_v02_entry_context.json")
    if actual != {**expected, "copied_path": str(target), "executed": False}:
        raise PermissionError("historical posthoc sidecar provenance differs")
    for path in (Path(expected["original_path"]), target):
        if _sha(path) != expected["file_sha256"]:
            raise PermissionError("historical posthoc sidecar bytes changed")
    sealed = json.loads(target.read_text(encoding="utf-8"))
    sidecar = sealed["payload"]
    if (canonical_contract_sha256(sidecar) != sealed["sha256"]
            or sealed["sha256"] != expected["payload_sha256"]
            or sidecar.get("variant_id") != row.variant_id
            or sidecar.get("context_archive_sha256") != plan.context_archive_sha256
            or len(sidecar.get("records", ())) != expected["record_count"]
            or sidecar.get("schema") != "ifsm_mffu_posthoc_v02_entry_context_not_executed_v1"):
        raise PermissionError("historical posthoc sidecar payload failed verification")
    annotations = [{"configuration": row.variant_id, "status": "posthoc_v02_not_executed",
                    **item} for item in sidecar["records"]]
    return annotations, {"historical_posthoc": str(target)}


def correction_disposition(plan, row, output: dict) -> str:
    if output.get("correction_reuse"):
        return verify_saved_correction_reuse(plan, row, output)["status"]
    receipt = correction_receipt(plan)
    if receipt is None:
        return "compatible_reused" if output.get("reuse") else "newly_completed"
    _record, original = _original_output(
        receipt, row, plan.source.warmup_dates + plan.source.evaluation_dates,
    )
    return ("replayed_equal" if economic_projection(output) == economic_projection(original)
            else "replayed_changed")
