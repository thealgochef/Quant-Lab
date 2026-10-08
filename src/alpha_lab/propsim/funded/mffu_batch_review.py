"""Compact, source-bound result review ZIP for the completed MFFU matrix.

The established comparison exporter first verifies the saved financial result
in a task-owned staging directory. This module adds only MFFU-specific saved
evidence, then verifies the final ZIP and atomically moves it into ``reports``.
It does not run a strategy, read market data, or change the immutable result.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import unquote, urlsplit

from alpha_lab.agents.data_infra.ifvg.funded_comparison_review import (
    publish_comparison_review_folder,
    verify_published_folder,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
from alpha_lab.propsim.funded.comparison_plan import (
    RESULT_STORE,
    FundedComparisonResultEnvelope,
)
from alpha_lab.propsim.funded.comparison_runner import load_comparison_result
from alpha_lab.propsim.funded.mffu_batch_analysis import analyze_mffu_batch
from alpha_lab.propsim.funded.mffu_batch_plan import (
    HANDOFF_ROOT,
    MffuBatchPlanEnvelope,
    MffuBatchPlanPayload,
    _sha_bytes,
    load_approved_mffu_batch_plan,
)
from alpha_lab.propsim.funded.result import canonical_json
from alpha_lab.propsim.funded.runner import read_ledger

SCHEMA = "ifsm_mffu_result_review_zip_v1"
RESULT_SCHEMA = "ifsm_mffu_context_64_batch_result_v1"
ANALYSIS_SCHEMA = "ifsm_mffu_batch_analysis_v1"
PACKAGE_PARENT = "ifsm_mffu_context_batch"
REPO_ROOT = Path(__file__).resolve().parents[4]
_EXTRA_FILES = (
    "MFFU_ANALYSIS.md",
    "MFFU_PACKAGE_GUIDE.md",
    "analysis_summary.json",
    "cash_concentration_monthly.csv",
    "decision_context.csv",
    "decision_context_coverage.csv",
    "entry_order_runner.csv",
    "factor_interactions.csv",
    "matched_pairs.csv",
    "matched_trade_diagnostics.csv",
    "mffu_analysis.json",
    "resolved_plan.json",
    "reuse_ledger.csv",
    "reuse_proofs.json",
    "reuse_context_annotations.csv",
    "trade_concentration.csv",
    "waiting_intervals.csv",
)
_SCREENSHOTS = {
    "matrix": "screenshots/mffu_frozen_matrix.png",
    "result": "screenshots/mffu_completed_result.png",
}
_LINK = re.compile(r"!?(?:\[[^\]]*\])\(([^)]+)\)")


@dataclass(frozen=True)
class PublishedMffuReview:
    path: Path
    result_id: str
    plan_id: str
    export_version: int
    receipt: dict[str, Any]


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _write_json_file(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_bytes(rows: list[dict[str, Any]], columns: tuple[str, ...] = ()) -> bytes:
    fields = list(columns)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        raise ValueError("CSV export requires declared columns or rows")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_cell(row.get(key)) for key in fields})
    return stream.getvalue().encode("utf-8")


def _write_csv_file(path: Path, rows: list[dict[str, Any]],
                    columns: tuple[str, ...] = ()) -> None:
    fields = list(columns)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        raise ValueError("CSV export requires declared columns or rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(row.get(key)) for key in fields})


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _analysis_tables(analysis: dict[str, Any]) -> dict[str, bytes]:
    interactions = []
    for factor, table in analysis["interactions"].items():
        interactions.extend({"interaction": factor, **cell} for cell in table["cells"])
    waiting = []
    for variant_id, item in analysis["waiting_by_variant"].items():
        if item["status"] != "available":
            waiting.append({"variant_id": variant_id, "measure": "unavailable",
                            "status": item["status"], "reason": item.get("reason")})
            continue
        for measure in ("initial", "max_between_receipts", "terminal",
                        "max_no_receipt_interval"):
            interval = item.get(measure)
            waiting.append({"variant_id": variant_id, "measure": measure,
                            "status": "available" if interval else "unavailable",
                            "reason": None if interval else item.get("between_status")
                            if measure == "max_between_receipts" else item.get(
                                "terminal_status"),
                            "received_count": item["received_count"],
                            **(interval or {})})
    monthly = []
    for variant_id, item in analysis["cash_concentration_by_variant"].items():
        if item["status"] == "available":
            monthly.extend({"variant_id": variant_id, "status": "available", **month}
                           for month in item["monthly"])
        else:
            monthly.append({"variant_id": variant_id, "status": item["status"],
                            "reason": item.get("reason")})
    concentration = []
    for variant_id, item in analysis["trade_concentration_by_variant"].items():
        if item["status"] != "available":
            concentration.append({"variant_id": variant_id, "group": "unavailable",
                                  "status": item["status"], "reason": item.get("reason")})
            continue
        for group in ("yearly_by_entry", "by_entry_session", "by_resolution"):
            concentration.extend({"variant_id": variant_id, "group": group,
                                  "label": label, "status": "available", **value}
                                 for label, value in item[group].items())
        zone_coverage = item.get("htf_zone_coverage") or {}
        for zone_id, value in item.get("by_htf_zone_id", {}).items():
            concentration.append({"variant_id": variant_id, "group": "by_htf_zone_id",
                                  "label": zone_id, "status": zone_coverage.get("status"),
                                  "basis": "actual_funded_trade_pnl_not_payout_cash", **value})
        unknown_zone = item.get("unknown_htf_zone")
        if unknown_zone is not None:
            concentration.append({"variant_id": variant_id, "group": "by_htf_zone_id",
                                  "label": "unknown", "status": zone_coverage.get("status"),
                                  "reason": zone_coverage.get("reason"),
                                  "basis": "actual_funded_trade_pnl_not_payout_cash",
                                  **unknown_zone})
        account_coverage = item.get("account_coverage") or {}
        for account_id, value in item.get("by_account_id", {}).items():
            concentration.append({"variant_id": variant_id, "group": "by_account_id",
                                  "label": account_id,
                                  "status": account_coverage.get("status"),
                                  "basis": "actual_funded_trade_pnl_not_payout_cash", **value})
        unknown_account = item.get("unknown_account")
        if unknown_account is not None:
            concentration.append({"variant_id": variant_id, "group": "by_account_id",
                                  "label": "unknown",
                                  "status": account_coverage.get("status"),
                                  "reason": account_coverage.get("reason"),
                                  "basis": "actual_funded_trade_pnl_not_payout_cash",
                                  **unknown_account})
        concentration.extend({"variant_id": variant_id, "group": "largest_trade",
                              "label": str(rank), "status": "available", **trade}
                             for rank, trade in enumerate(item["largest_trades"], 1))
    trade_diagnostics = []
    for pair in analysis["matched_pairs"]:
        detail = pair.get("trade_diagnostics") or {}
        trade_diagnostics.append({
            "base_id": pair["base_id"], "challenger_id": pair["challenger_id"],
            "changed_axis": pair.get("changed_axis"), "group": pair.get("group"),
            **{key: value for key, value in detail.items()
               if key not in {"changed_common", "gained", "lost"}},
        })
    entry_order = []
    for variant_id, item in analysis.get("entry_order_runner_by_variant", {}).items():
        if item["status"] != "available":
            entry_order.append({"variant_id": variant_id, "status": item["status"],
                                "reason": item.get("reason")})
            continue
        for order, value in item["by_entry_order"].items():
            entry_order.append({"variant_id": variant_id, "status": "available",
                                "entry_order": order, "resolution": "all",
                                "trades": value["trades"],
                                "net_trade_pnl_cents": value["net_trade_pnl_cents"]})
            entry_order.extend({"variant_id": variant_id, "status": "available",
                                "entry_order": order, "resolution": resolution,
                                **sub}
                               for resolution, sub in value["by_resolution"].items())
    coverage = []
    for variant_id, item in analysis.get("decision_context_coverage_by_variant", {}).items():
        if item["status"] != "available":
            coverage.append({"variant_id": variant_id, "status": item["status"],
                             "reason": item.get("reason")})
            continue
        coverage.extend({"variant_id": variant_id, "status": "available",
                         "stream": stream, **counts}
                        for stream, counts in item["streams"].items())
    summary = {key: analysis[key] for key in (
        "schema", "firm_key", "variant_count", "completed_count", "unavailable_count",
        "cash_basis", "waiting_day_basis", "ranking",
    )}
    summary["interaction_summaries"] = {
        key: {k: value[k] for k in ("available_cells", "unavailable_cells",
                                        "min_delta_cents", "max_delta_cents")}
        for key, value in analysis["interactions"].items()
    }


    return {
        "analysis_summary.json": _json_bytes(summary),
        "matched_pairs.csv": _csv_bytes([
            {key: value for key, value in pair.items() if key != "trade_diagnostics"}
            for pair in analysis["matched_pairs"]
        ], (
            "base_id", "challenger_id", "changed_axis", "group", "status",
            "delta_cents", "unavailable_ids")),
        "matched_trade_diagnostics.csv": _csv_bytes(trade_diagnostics, (
            "base_id", "challenger_id", "changed_axis", "group", "status", "reason",
            "basis", "entry_match_basis", "common_entries", "changed_common_entries",
            "gained_entries", "lost_entries", "common_entry_pnl_delta_cents",
            "gained_entry_pnl_cents", "lost_entry_pnl_cents")),
        "factor_interactions.csv": _csv_bytes(interactions, (
            "interaction", "status", "delta_cents", "unavailable_ids")),
        "entry_order_runner.csv": _csv_bytes(entry_order, (
            "variant_id", "status", "reason", "entry_order", "resolution", "trades",
            "net_trade_pnl_cents")),
        "decision_context_coverage.csv": _csv_bytes(coverage, (
            "variant_id", "status", "reason", "stream", "decision_count", "events",
            "actions", "rejection_reasons", "gamma_status", "levels_status",
            "gamma_sign", "geometry_fallback_reasons", "quota_increments")),
        "waiting_intervals.csv": _csv_bytes(waiting, (
            "variant_id", "measure", "status", "reason", "received_count", "kind",
            "start_utc", "end_utc", "start_chicago_date", "end_chicago_date",
            "calendar_days", "evaluated_trading_days", "elapsed_seconds",
            "censored_start", "censored_end")),
        "cash_concentration_monthly.csv": _csv_bytes(monthly, (
            "variant_id", "status", "reason", "month", "received_cents",
            "account_costs_cents", "net_cash_cents", "cumulative_net_cash_cents")),
        "trade_concentration.csv": _csv_bytes(concentration, (
            "variant_id", "group", "label", "status", "reason", "basis", "trades",
            "net_trade_pnl_cents", "trade_id", "entry_utc", "resolution")),
    }


def _expected_trade_concentration_rows(analysis: dict[str, Any]) -> int:
    expected = 0
    for item in analysis["trade_concentration_by_variant"].values():
        if item["status"] != "available":
            expected += 1
            continue
        if ("unknown_htf_zone" not in item or "by_htf_zone_id" not in item or
                "unknown_account" not in item or "by_account_id" not in item):
            raise ValueError("saved trade concentration lacks zone or account coverage")
        expected += (sum(len(item[group]) for group in (
            "yearly_by_entry", "by_entry_session", "by_resolution", "by_htf_zone_id",
            "by_account_id"
        )) + len(item["largest_trades"]) + 2)
    return expected


def _mffu_policy_text(plan: MffuBatchPlanPayload) -> str:
    """Describe the exact frozen axes in the standard verified rules document."""
    expected = {
        "schedule": {"S0", "S1"}, "daily_cap": {"U", "D1"},
        "entry_context": {"F0", "FE", "FL", "FEL"},
        "exit": {"XP", "XF", "XG", "XE"},
        "sizing": {"Q10", "Q6", "QG"},
        "overhead": {"O0", "O8"},
        "geometry": {"G0", "G05", "G075", "G10"},
    }
    intents = [json.loads(variant.intent_json) for variant in plan.variants]
    if len(intents) != 64 or any(
        {row[axis] for row in intents} != values for axis, values in expected.items()
    ):
        raise ValueError("MFFU trading rules cannot describe a changed matrix")
    if plan.context_policy_version != "mq_eod_asof_nominal_2200_chicago_v01":
        raise ValueError("MFFU trading rules require the bound EOD context policy")
    if plan.fee_rounding_policy != "per_fill_total_round_half_up_cent_v1":
        raise ValueError("MFFU trading rules require the bound fill-fee policy")
    return "\n".join((
        "## Frozen MFFU 64-policy matrix", "",
        "The saved resolved_plan.json and reuse_ledger.csv identify all 64 intents and their "
        "effective sections. Each variant has independent ordinary strategy and funded "
        "account streams. All other saved strategy settings retain their reference "
        "values.", "",
        "- S0 admits entries in the original Asia (3:00 PM–12:45 AM), London "
        "(1:00–6:00 AM), and New York (7:00 AM–1:00 PM) Chicago windows. S1 uses "
        "all saved permitted market hours. Both retain market closures and the mandatory "
        "flat deadline.",
        "- U adds no daily entry limit. D1 permits at most one actually filled entry "
        "per logical Chicago trading day, across account replacements. A refused or "
        "unfilled candidate does not consume it.",
        "- F0 adds no context rejection. FE rejects a known positive total-gamma state "
        "when its observed positive-report run age is 1–5 distinct reports. FL rejects "
        "known positive gamma at entry from 1:00 AM inclusive to 6:00 AM exclusive "
        "Chicago. FEL applies both, retaining both reasons. Unknown or left-censored "
        "positive-run age alone cannot trigger FE.",
        "- XP takes exactly half at the original 1R target, moves the remainder's stop "
        "to entry, then exits at that stop or mandatory close. XF closes all at 1R. "
        "XG decides once at the first 1R checkpoint: whole under positive gamma, half "
        "under negative, neutral, or unknown gamma. XE closes all only under eligible "
        "early-positive gamma there, otherwise follows XP. Before 1R, each retains "
        "the same original structural stop and daily close.",
        "- Q10 enters ten micros; Q6 enters six with a three/three partial split. "
        "QG selects ten under negative, neutral, or unknown gamma, and six under "
        "positive gamma, once at entry. Exit quantities follow that original entry size.",
        "- Every actual fill posts $0.514 per filled micro multiplied by that fill's "
        "quantity, then rounded once to cents with ROUND_HALF_UP. A six-micro "
        "entry and two three-micro exits cost $3.08 + $1.54 + $1.54 = $6.16. "
        "The ten/five/five control remains $5.14 + $2.57 + $2.57 = $10.28.",
        "- O0 adds no overhead rejection. O8 examines the nearest valid price strictly "
        "above entry among the original eight named levels; it rejects only when that "
        "price is at or below the original 1R target and its finite GEX is strictly "
        "greater than 300,000 in supplied vendor units. Equal-price aliases use their "
        "maximum finite GEX; a null nearest GEX allows the entry.",
        "- G0 keeps the 80-tick opposing-parent maximum. G05, G075, and G10 use "
        "0.05, 0.075, and 0.10 times (1D Max minus 1D Min)/2, converted to "
        "quarter-point ticks with half-up rounding and a one-tick minimum at parent "
        "lock. Missing, invalid, or nonpositive eligible range falls back to 80 ticks. "
        "No other stop, wait, target, or geometry setting changes.", "",
        "The MenthorQ EOD map is eligible only after the bound nominal 10:00 PM "
        "Chicago availability convention; this is not observed vendor publication "
        "time. At XG/XE target checkpoints the funded stream uses the ordered price "
        "observation, while the ordinary candle stream freezes its target branch from "
        "context available at the bar open and keeps stop-first resolution. "
        "The receipt in decision_context.csv records the causal context. "
        "Compatible reused controls have separately labeled posthoc context "
        "annotations, not executed v02 policy decisions.",
    ))


def _findings(analysis: dict[str, Any], dispositions: list[dict]) -> str:
    ranking = analysis["ranking"]
    leader = ranking[0] if ranking else None
    leader_text = (f"{leader['variant_id']} with "
                   f"${Decimal(leader['net_cash_cents']) / 100:,.2f} "
                   "net received cash" if leader else "No completed financial row")
    statuses = {status: sum(row["status"] == status for row in dispositions)
                for status in ("newly_completed", "compatible_reused", "failed")}
    return ("# MFFU saved-result analysis\n\n"
            f"The highest saved result is {leader_text}. This is an inspected-period "
            "ranking, not independent statistical proof. Net cash is received after-split "
            "payouts less every account purchase.\n\n"
            f"All 64 declared intents are present: {statuses['newly_completed']} newly "
            f"completed, {statuses['compatible_reused']} compatible reused, and "
            f"{statuses['failed']} failed. The analysis has "
            f"{analysis['completed_count']} completed rows and "
            f"{analysis['unavailable_count']} unavailable rows. Unavailable effects "
            "have blank cash values.\n\n"
            "Matched comparisons use the handoff's 188 directed pairs. Interaction "
            "cells keep the four matching configurations together; their effects are "
            "descriptive and dependent. Waiting intervals use Chicago dates and also "
            "report evaluated trading days. Trade P&L is kept separate from received cash. "
            "A fixed-path 'close the remainder at first target' figure, if discussed in "
            "reviewed findings, is a nonexecuted diagnostic; XF is an independently "
            "replayed actual policy.\n\n"
            "MenthorQ EOD use assumes a nominal 10:00 PM Chicago release. Actual vendor "
            "publication times were not measured. Reused controls, where present, carry "
            "separately labeled posthoc context annotations; those are not executed Core "
            "policy decisions. See the reviewed findings for interpretation and limits.\n")


def _guide(base_files: int, context_rows: int) -> str:
    return ("# Result-review ZIP guide\n\n"
            "The established comparison exporter verified the financial tables, result "
            "validation and its own run manifest before this ZIP was built. Its original "
            "documents, tables, charts and manifest remain in this package. The outer "
            "manifest covers the complete ZIP.\n\n"
            f"The standard export supplied {base_files} files. The added decision-context "
            f"table has {context_rows} saved records. `resolved_plan.json` is the exact "
            "approved plan embedded in the saved result. `reuse_ledger.csv` gives all 64 "
            "dispositions and effective hashes. Verified reused controls retain their "
            "source proofs and sealed posthoc v02 sidecars; the annotation CSV is explicitly "
            "labeled as context observed after the historical execution. The factor, "
            "waiting and concentration "
            "tables come from the saved analysis section; no market replay occurs during "
            "export. All money in these new tables is exact integer cents.\n\n"
            "Screenshots show the reopened frozen matrix and completed MFFU result. "
            "The reviewed findings distinguish observed outcomes from nominal vendor "
            "timing and other source limits.\n")


def _next_version(parent: Path, result_id: str) -> int:
    prefix = f"IFSM_MFFU_Result_Review_{result_id[:16]}_v"
    used = [int(path.stem[len(prefix):]) for path in parent.glob(f"{prefix}*.zip")
            if path.stem[len(prefix):].isdigit()] if parent.is_dir() else []
    return max(used, default=0) + 1


def _screenshots(screenshots: dict[str, Path | bytes]) -> dict[str, bytes]:
    if set(screenshots) != set(_SCREENSHOTS):
        raise ValueError("exactly the matrix and completed-result screenshots are required")
    from PIL import Image

    out = {}
    for key, relative in _SCREENSHOTS.items():
        source = screenshots[key]
        payload = source if isinstance(source, bytes) else Path(source).read_bytes()
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"{key} screenshot is not PNG")
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        out[relative] = payload
    return out


def _has_causal_decision_time(row: dict[str, Any]) -> bool:
    context = row.get("context")
    if not isinstance(context, dict):
        return False
    value = context.get("decision_ts_utc")
    if not isinstance(value, str):
        return False
    try:
        instant = datetime.fromisoformat(value)
    except ValueError:
        return False
    return instant.utcoffset() == timedelta(0)


def _assert_result(plan: MffuBatchPlanPayload, plan_id: str,
                   result: dict[str, Any]) -> tuple[list[dict], dict]:
    section = result.get("mffu_batch") or {}
    if (section.get("schema") != RESULT_SCHEMA or
            result.get("funded_comparison_plan_id") != plan_id or
            section.get("plan") != plan.model_dump(mode="json") or
            not (result.get("validation") or {}).get("passed")):
        raise ValueError("completed MFFU result, validation or frozen plan differs")
    dispositions = section.get("dispositions")
    ids = [row.variant_id for row in plan.variants]
    if (not isinstance(dispositions, list) or len(dispositions) != 64 or
            [row.get("variant_id") for row in dispositions] != ids):
        raise ValueError("saved result lacks all 64 ordered dispositions")
    correction_reused = {"reused_nonimpact", "reused_equivalent_after_verification"}
    reused_statuses = {"compatible_reused", *correction_reused}
    allowed = {"newly_completed", "replayed_equal", "replayed_changed", "failed", *reused_statuses}
    if any(row.get("status") not in allowed or
           (row.get("status") in reused_statuses) != bool(row.get("reused_from"))
           for row in dispositions):
        raise ValueError("saved reuse status or source is inconsistent")
    decisions = section.get("decision_context")
    if not isinstance(decisions, list) or any(not isinstance(row, dict) or
                                              row.get("configuration") not in ids or
                                              row.get("stream") not in {"strategy", "funded"} or
                                              not _has_causal_decision_time(row)
                                              for row in decisions):
        raise ValueError("saved result lacks valid decision-context records")
    reused_ids = {row["variant_id"] for row in dispositions
                  if row["status"] in reused_statuses}
    proofs = section.get("reuse_proofs")
    annotations = section.get("reuse_context_annotations")
    if (not isinstance(proofs, dict) or set(proofs) != reused_ids or
            not isinstance(annotations, list) or
            any(not isinstance(row, dict) or row.get("configuration") not in reused_ids or
                row.get("status") != "posthoc_v02_not_executed"
                for row in annotations)):
        raise ValueError("saved reuse proofs or posthoc annotations are incomplete")
    lineage = section.get("correction_lineage") or {}
    for row in dispositions:
        if row["status"] not in correction_reused:
            continue
        proof = proofs[row["variant_id"]]
        if (lineage.get("schema") != "ifsm_mffu_lifecycle_correction_receipt_v1"
                or row["reused_from"] != lineage.get("original_result_id")
                or proof.get("schema") != "ifsm_mffu_correction_child_reuse_v1"
                or proof.get("status") != row["status"]
                or proof.get("original_result_id") != lineage.get("original_result_id")
                or proof.get("corrected_core_source") != plan.core_source.model_dump(mode="json")):
            raise ValueError("correction reuse disposition lacks its exact lineage proof")
    analysis = result.get("mffu_analysis") or {}
    if (analysis.get("schema") not in {ANALYSIS_SCHEMA, "ifsm_mffu_batch_analysis_v2"}
            or analysis.get("variant_count") != 64 or
            len(analysis.get("matched_pairs") or []) != 188 or
            any("trade_diagnostics" not in pair for pair in analysis["matched_pairs"]) or
            set(analysis.get("entry_order_runner_by_variant") or {}) != set(ids) or
            set(analysis.get("decision_context_coverage_by_variant") or {}) != set(ids)):
        raise ValueError("saved result lacks the complete MFFU analysis")
    return dispositions, analysis


def _verify_analysis(plan: MffuBatchPlanPayload, result: dict, analysis: dict) -> None:
    with zipfile.ZipFile(plan.handoff_zip) as handoff:
        data = handoff.read(HANDOFF_ROOT + "COMPARISON_PAIRS.csv")
    if _sha_bytes(data) != plan.source_member_sha256["COMPARISON_PAIRS.csv"]:
        raise PermissionError("comparison pair source differs from the frozen plan")
    pairs = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    verifying_result = result
    legacy = analysis.get("schema") == ANALYSIS_SCHEMA
    if legacy:
        # Historical v1 encoded account_id only. Verify that frozen projection
        # under its own semantics, while new publications use the corrected v2.
        verifying_result = {**result, "tables": {**result.get("tables", {}), "trades": [
            {**row, "account_number": None} for row in result["tables"]["trades"]
        ]}}
    recomputed = analyze_mffu_batch(
        verifying_result, variants=plan.variants, comparison_pairs=pairs,
        evaluation_dates=plan.source.evaluation_dates,
        economic_result_id=analysis.get("economic_result_id"),
    )
    if legacy:
        recomputed["schema"] = ANALYSIS_SCHEMA
        recomputed.pop("economic_result_id", None)
        recomputed.pop("reporting_account_version", None)

        def original_projection(value):
            if isinstance(value, dict):
                value.pop("reporting_account_key", None)
                if value.get("reason") == "account_identity_absent_or_null":
                    value["reason"] = "account_id_absent_or_null"
                for item in value.values():
                    original_projection(item)
            elif isinstance(value, list):
                for item in value:
                    original_projection(item)

        original_projection(recomputed)
    if _json_bytes(recomputed) != _json_bytes(analysis):
        raise ValueError("saved MFFU analysis differs from the saved financial result")


def _verify_reuse(plan: MffuBatchPlanPayload, plan_id: str, approval_id: str,
                  state_root: Path, result: dict) -> dict[str, Path]:
    """Recheck source-compatible worker proofs and copy exact sealed sidecar bytes."""
    from alpha_lab.propsim.funded.full_range_batch import load_checkpoint
    from alpha_lab.propsim.funded.mffu_batch_reuse import verify_saved_reuse
    from alpha_lab.propsim.funded.mffu_batch_run import _dispatch

    batch = result["mffu_batch"]
    proofs = batch["reuse_proofs"]
    dates = plan.source.warmup_dates + plan.source.evaluation_dates
    worker_root = state_root / plan_id / "workers"
    sidecars: dict[str, Path] = {}
    expected_annotations = []
    for row in plan.variants:
        if row.variant_id not in proofs:
            continue
        worker_dir = worker_root / row.variant_id
        dispatch = _dispatch(plan_id, approval_id, plan, row)
        saved = load_checkpoint(
            worker_dir / "output.json",
            dispatch_sha256=canonical_contract_sha256(dispatch), dates=dates,
        )
        if saved is None:
            raise PermissionError(f"reused worker output is missing: {row.variant_id}")
        if saved["output"].get("correction_reuse"):
            from alpha_lab.propsim.funded.mffu_batch_correction_reuse import (
                correction_reuse_annotations,
                verify_saved_correction_reuse,
            )

            proof = verify_saved_correction_reuse(plan, row, saved["output"])
            annotations, paths = correction_reuse_annotations(plan, row, saved["output"])
            if proof != proofs[row.variant_id]:
                raise PermissionError(f"saved correction proof differs: {row.variant_id}")
            expected_annotations.extend(annotations)
            if paths.get("historical_posthoc"):
                sidecars[f"reuse_sidecars/{row.variant_id}_v02_entry_context.json"] = Path(
                    paths["historical_posthoc"])
            continue
        proof, sidecar = verify_saved_reuse(plan, row, saved["output"], worker_dir)
        if proof != proofs[row.variant_id]:
            raise PermissionError(f"saved result reuse proof differs: {row.variant_id}")
        expected_annotations.extend({
            "configuration": row.variant_id, "status": "posthoc_v02_not_executed",
            **record,
        } for record in sidecar["records"])
        relative = f"reuse_sidecars/{row.variant_id}_v02_entry_context.json"
        sidecars[relative] = worker_dir / "reuse_v02_entry_context.json"
    if expected_annotations != batch["reuse_context_annotations"]:
        raise ValueError("saved posthoc annotations differ from verified reuse sidecars")
    return sidecars


def _markdown_link_errors(files: dict[str, bytes],
                          known_names: set[str] | None = None) -> list[str]:
    errors = []
    names = known_names or set(files)
    for name, payload in files.items():
        if not name.endswith(".md"):
            continue
        for raw in _LINK.findall(payload.decode("utf-8")):
            url = urlsplit(raw.strip().strip("<>"))
            if url.scheme or url.netloc or not url.path:
                continue
            target = PurePosixPath(name).parent / unquote(url.path)
            parts = target.parts
            if ".." in parts or str(target) not in names:
                errors.append(f"{name}: {raw}")
    return errors


def verify_mffu_result_review_zip(
    path: Path, *, extraction_root: Path | None = None,
) -> dict[str, Any]:
    """Safely extract and read back every ZIP member against both manifests."""
    path = Path(path)
    extraction_root = Path(extraction_root or tempfile.gettempdir()).resolve()
    if extraction_root.is_relative_to(REPO_ROOT):
        raise ValueError("ZIP verification extraction must stay outside the repository")
    extraction_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise ValueError("ZIP has duplicate entries or no final manifest")
        if any(entry.is_dir() or entry.filename.startswith("/") or "\\" in entry.filename
               or PureWindowsPath(entry.filename).drive
               or ".." in PurePosixPath(entry.filename).parts for entry in entries):
            raise ValueError("ZIP contains an unsafe or unlisted path")
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError("result-review manifest schema differs")
        listed = {row["path"]: row for row in manifest.get("files", [])}
        if len(listed) != len(manifest.get("files", [])):
            raise ValueError("result-review manifest lists a file more than once")
        if set(names) != set(listed) | {"manifest.json"}:
            raise ValueError("ZIP members differ from the final manifest")
        stamp = tuple(manifest["zip_entry_time_utc"])
        if any(entry.date_time != stamp for entry in entries):
            raise ValueError("ZIP member timestamp differs from the final manifest")
        when = datetime(*stamp, tzinfo=UTC)
        if when.year < 1980 or when > datetime.now(UTC) + timedelta(days=1):
            raise ValueError("ZIP member timestamp is invalid")
        with tempfile.TemporaryDirectory(prefix="mffu_review_readback_",
                                         dir=extraction_root) as temp:
            extracted = Path(temp)
            observed: dict[str, dict[str, Any]] = {}
            for entry in entries:
                name = entry.filename
                target = extracted / name
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with archive.open(entry) as source, target.open("wb") as destination:
                    while chunk := source.read(8 * 1024 * 1024):
                        digest.update(chunk)
                        size += len(chunk)
                        destination.write(chunk)
                observed[name] = {"bytes": size, "sha256": digest.hexdigest()}
            present = {p.relative_to(extracted).as_posix() for p in extracted.rglob("*")
                       if p.is_file()}
            if present != set(names):
                raise ValueError("extracted ZIP payload differs from the final archive")
            mismatched = [name for name, row in listed.items()
                          if observed[name] != {"bytes": row["bytes"],
                                                "sha256": row["sha256"]}]
            if mismatched:
                raise ValueError(f"ZIP payload checksum mismatch: {mismatched}")
            docs = {name: (extracted / name).read_bytes() for name in names
                    if name.endswith(".md")}
            if _markdown_link_errors(docs, set(names)):
                raise ValueError("ZIP contains broken relative document links")
            plan_path = extracted / "resolved_plan.json"
            plan = MffuBatchPlanPayload.model_validate_json(plan_path.read_bytes())
            if (len(plan.variants) != 64 or
                    MffuBatchPlanEnvelope.from_payload(plan).funded_comparison_plan_id
                    != manifest["plan_id"] or
                    observed["resolved_plan.json"]["sha256"]
                    != manifest["plan_payload_sha256"]):
                raise ValueError("ZIP does not preserve the complete frozen plan")
            with (extracted / "reuse_ledger.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                dispositions = list(csv.DictReader(stream))
            if (len(dispositions) != 64 or
                    [row["variant_id"] for row in dispositions] !=
                    [f"MCB{index:03d}" for index in range(1, 65)]):
                raise ValueError("ZIP does not contain all ordered dispositions")
            proofs = json.loads((extracted / "reuse_proofs.json").read_text(
                encoding="utf-8"))
            reused_ids = {row["variant_id"] for row in dispositions
                          if row["status"] in {"compatible_reused", "reused_nonimpact",
                                               "reused_equivalent_after_verification"}}
            if set(proofs) != reused_ids:
                raise ValueError("ZIP reuse proofs differ from its dispositions")
            for row in dispositions:
                if row["status"] in {"reused_nonimpact", "reused_equivalent_after_verification"}:
                    proof = proofs[row["variant_id"]]
                    if (proof.get("schema") != "ifsm_mffu_correction_child_reuse_v1"
                            or proof.get("status") != row["status"]
                            or proof.get("original_result_id") != row["reused_from"]):
                        raise ValueError("ZIP correction reuse differs from its lineage proof")
                elif row["status"] == "compatible_reused":
                    if "posthoc_v02" not in proofs[row["variant_id"]]:
                        raise ValueError("ZIP legacy reuse lacks its posthoc proof")
                elif row["status"] not in {"newly_completed", "replayed_equal",
                                           "replayed_changed", "failed"}:
                    raise ValueError("ZIP contains an unsupported child disposition")
            sidecar_names = {f"reuse_sidecars/{key}_v02_entry_context.json"
                             for key in reused_ids if proofs[key].get("posthoc_v02")
                             or proofs[key].get("historical_posthoc")}
            if {name for name in names if name.startswith("reuse_sidecars/")} != sidecar_names:
                raise ValueError("ZIP reuse sidecars differ from its proofs")
            with (extracted / "reuse_context_annotations.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                annotation_count = sum(1 for _ in csv.DictReader(stream))
            verified_annotation_count = 0
            for key, proof in proofs.items():
                reference = proof.get("posthoc_v02") or proof.get("historical_posthoc")
                if not reference:
                    continue
                sidecar_name = f"reuse_sidecars/{key}_v02_entry_context.json"
                envelope = json.loads((extracted / sidecar_name).read_text(encoding="utf-8"))
                payload = envelope["payload"]
                if (envelope["sha256"] != _sha_bytes(canonical_json(payload).encode("utf-8"))
                        or canonical_contract_sha256(payload)
                        != reference["payload_sha256"]
                        or (reference.get("record_count") is not None
                            and len(payload["records"]) != reference["record_count"])
                        or (reference.get("file_sha256") is not None
                            and observed[sidecar_name]["sha256"] != reference["file_sha256"])):
                    raise ValueError(f"ZIP sealed posthoc reuse sidecar differs: {key}")
                verified_annotation_count += len(payload["records"])
            if annotation_count != verified_annotation_count:
                raise ValueError("ZIP posthoc annotation count differs from reuse proofs")
            for filename, expected in manifest["row_counts"].items():
                with (extracted / filename).open("r", encoding="utf-8", newline="") as stream:
                    actual = sum(1 for _ in csv.DictReader(stream))
                if actual != expected:
                    raise ValueError(f"ZIP table row count changed: {filename}")
            base = json.loads((extracted / "run_manifest.json").read_text(encoding="utf-8"))
            if (manifest["base_review_manifest_sha256"] !=
                    observed["run_manifest.json"]["sha256"]):
                raise ValueError("standard export manifest differs")
            for row in base["files"]:
                name = row["path"]
                if observed.get(name) != {"bytes": row["bytes"], "sha256": row["sha256"]}:
                    raise ValueError(f"standard verified export changed in ZIP: {name}")
        return {
            "passed": True, "result_id": manifest["result_id"],
            "plan_id": manifest["plan_id"], "export_version": manifest["export_version"],
            "members_verified": len(names), "payloads_verified": len(listed),
            "manifest_sha256": observed["manifest.json"]["sha256"],
            "zip_sha256": _sha_file(path),
            "row_counts": manifest["row_counts"],
        }


def publish_mffu_result_review(
    *, plan_id: str, result_id: str, store_root: Path, state_root: Path, staging_root: Path,
    reports_root: Path, review_findings: str, screenshots: dict[str, Path | bytes],
    export_version: int | None = None,
) -> PublishedMffuReview:
    """Publish one immutable result ZIP after base and final readback verification.

    ``staging_root`` must be outside the repository. ``reports_root`` is the
    repository's final review directory. The caller supplies real reopened-UI
    screenshots and reviewed prose, not synthetic placeholders.
    """
    store_root = Path(store_root)
    state_root = Path(state_root).resolve()
    staging_root = Path(staging_root).resolve()
    reports_root = Path(reports_root).resolve()
    if staging_root.is_relative_to(REPO_ROOT) or staging_root.is_relative_to(reports_root):
        raise ValueError("MFFU review staging must stay outside the repository")
    if state_root.is_relative_to(REPO_ROOT):
        raise ValueError("MFFU worker state must stay outside the repository")
    if reports_root != (REPO_ROOT / "reports").resolve():
        raise ValueError("final MFFU review ZIP must be published under repository reports")
    if not review_findings or not review_findings.strip():
        raise ValueError("reviewed findings are required for final publication")
    screenshot_files = _screenshots(screenshots)
    envelope, approval = load_approved_mffu_batch_plan(store_root, plan_id)
    plan = envelope.payload
    result = load_comparison_result(store_root, result_id)
    result_envelope = load_verified_envelope(
        store_root, RESULT_STORE, result_id, FundedComparisonResultEnvelope
    )
    dispositions, analysis = _assert_result(plan, plan_id, result)
    _verify_analysis(plan, result, analysis)
    sidecar_files = _verify_reuse(
        plan, plan_id, approval.funded_comparison_approval_id, state_root, result,
    )
    ledger = read_ledger(store_root)
    if not ledger:
        raise ValueError("the saved research ledger is required")
    parent = reports_root / PACKAGE_PARENT
    version = export_version or _next_version(parent, result_id)
    if version < 1:
        raise ValueError("export version must be positive")
    final = parent / f"IFSM_MFFU_Result_Review_{result_id[:16]}_v{version}.zip"
    if final.exists():
        raise FileExistsError(f"immutable MFFU review already exists: {final}")
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mffu_result_review_", dir=staging_root) as temp:
        temp_root = Path(temp)
        scope = result["full_range_reporting"]
        standard = publish_comparison_review_folder(
            result=result, result_id=result_id, reports_root=temp_root / "standard",
            ledger_entries=ledger, export_version=1, review_findings=review_findings,
            supplements={"run_context": {
                key: scope[key] for key in ("evaluation_dates", "warmup_dates", "cutoff_utc")
            }, "mffu_policy_text": _mffu_policy_text(plan)},
        )
        if not standard.checks_passed or not verify_published_folder(standard.path)["passed"]:
            raise ValueError("standard comparison review failed its readback")
        files = {path.relative_to(standard.path).as_posix(): path
                 for path in standard.path.rglob("*") if path.is_file()}
        base_file_count = len(files)
        if set(files) & set(_EXTRA_FILES):
            raise ValueError("MFFU additions collide with the standard export")
        addition_root = temp_root / "additions"

        def add_bytes(name: str, payload: bytes) -> None:
            target = addition_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            files[name] = target

        def add_json(name: str, value: Any) -> None:
            target = addition_root / name
            _write_json_file(target, value)
            files[name] = target

        def add_csv(name: str, rows: list[dict[str, Any]],
                    columns: tuple[str, ...]) -> None:
            target = addition_root / name
            _write_csv_file(target, rows, columns)
            files[name] = target

        add_json("resolved_plan.json", plan.model_dump(mode="json"))
        reuse = [{**row,
                  "effective_section_config_hash": variant.effective_section_config_hash,
                  "effective_behavior_hash": variant.effective_behavior_hash}
                 for row, variant in zip(dispositions, plan.variants, strict=True)]
        add_csv("reuse_ledger.csv", reuse, (
            "variant_id", "status", "reason", "reused_from",
            "effective_section_config_hash", "effective_behavior_hash"))
        add_json("reuse_proofs.json", result["mffu_batch"]["reuse_proofs"])
        annotations = result["mffu_batch"]["reuse_context_annotations"]
        add_csv("reuse_context_annotations.csv", annotations, (
            "configuration", "status", "stream", "historical_trade_id",
            "entry_ts_utc", "v02_posthoc_asof"))
        files.update(sidecar_files)
        decisions = result["mffu_batch"]["decision_context"]
        add_csv("decision_context.csv", decisions, ("configuration", "stream"))
        add_json("mffu_analysis.json", analysis)
        for name, payload in _analysis_tables(analysis).items():
            add_bytes(name, payload)
        add_bytes("MFFU_ANALYSIS.md", _findings(analysis, dispositions).encode("utf-8"))
        add_bytes("MFFU_PACKAGE_GUIDE.md", _guide(base_file_count, len(decisions)).encode(
            "utf-8"))
        for name, payload in screenshot_files.items():
            add_bytes(name, payload)
        expected_extra = set(_EXTRA_FILES) | set(_SCREENSHOTS.values()) | set(sidecar_files)
        if not expected_extra <= set(files):
            raise ValueError("MFFU review addition is incomplete")
        row_counts = {}
        for name, source in files.items():
            if name.endswith(".csv"):
                with source.open("r", encoding="utf-8", newline="") as stream:
                    row_counts[name] = sum(1 for _ in csv.DictReader(stream))
        if (row_counts["reuse_ledger.csv"] != 64 or row_counts["matched_pairs.csv"] != 188 or
                row_counts["decision_context.csv"] != len(decisions) or
                row_counts["reuse_context_annotations.csv"] != len(annotations) or
                row_counts["trade_concentration.csv"] !=
                _expected_trade_concentration_rows(analysis)):
            raise ValueError("MFFU review table counts differ from saved result")
        stamp = datetime.now(UTC).replace(microsecond=0)
        stamp = stamp.replace(second=stamp.second - stamp.second % 2)
        stamp_tuple = (stamp.year, stamp.month, stamp.day, stamp.hour, stamp.minute,
                       stamp.second)
        manifest = {
            "schema": SCHEMA, "result_id": result_id, "plan_id": plan_id,
            "approval_id": approval.funded_comparison_approval_id,
            "saved_result_json_sha256": result_envelope.payload.result_json_sha256,
            "export_version": version,
            "created_at_utc": stamp.isoformat().replace("+00:00", "Z"),
            "zip_entry_time_utc": stamp_tuple,
            "base_review_manifest_sha256": _sha_file(files["run_manifest.json"]),
            "plan_payload_sha256": _sha_file(files["resolved_plan.json"]),
            "context_archive_sha256": plan.context_archive_sha256,
            "context_policy_version": plan.context_policy_version,
            "core_base_commit": plan.core_source.base_commit,
            "core_patch_sha256": plan.core_source.patch_sha256,
            "dispositions": {status: sum(row["status"] == status for row in dispositions)
                             for status in ("newly_completed", "compatible_reused", "failed")},
            "row_counts": row_counts,
            "files": [{"path": name, "bytes": source.stat().st_size,
                       "sha256": _sha_file(source)}
                      for name, source in sorted(files.items())],
        }
        add_json("manifest.json", manifest)
        docs = {name: source.read_bytes() for name, source in files.items()
                if name.endswith(".md")}
        if errors := _markdown_link_errors(docs, set(files)):
            raise ValueError(f"result review has broken relative links: {errors}")
        staging_zip = temp_root / final.name
        with zipfile.ZipFile(staging_zip, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for name, source in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=stamp_tuple)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                info.file_size = source.stat().st_size
                with source.open("rb") as stream, archive.open(info, "w") as target:
                    shutil.copyfileobj(stream, target, 8 * 1024 * 1024)
        verify_mffu_result_review_zip(staging_zip, extraction_root=staging_root)
        parent.mkdir(parents=True, exist_ok=True)
        if staging_zip.stat().st_dev != parent.stat().st_dev:
            raise OSError("review staging and final reports must share one volume")
        try:
            # A hard link publishes complete, verified bytes atomically and fails
            # if a concurrent publisher has already claimed this export version.
            os.link(staging_zip, final)
        except FileExistsError as error:
            raise FileExistsError(f"immutable MFFU review appeared: {final}") from error
        try:
            receipt = verify_mffu_result_review_zip(final, extraction_root=staging_root)
        except BaseException:
            final.unlink(missing_ok=True)
            raise
    return PublishedMffuReview(final, result_id, plan_id, version, receipt)
