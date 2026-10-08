"""The MFFU review ZIP preserves saved matrix rows and detects payload changes."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha_lab.propsim.funded import mffu_batch_review as review


def _bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _write_zip(path: Path, *, bad_hash: bool = False, bad_link: bool = False,
               bad_time: bool = False, with_reuse: bool = False,
               bad_sidecar: bool = False, duplicate_manifest_file: bool = False) -> None:
    dispositions = [
        {"variant_id": f"MCB{index:03d}",
         "status": "compatible_reused" if with_reuse and index == 1
         else "newly_completed", "reused_from": "C02" if with_reuse and index == 1
         else None}
        for index in range(1, 65)
    ]
    files = {
        "README.md": b"[missing](gone.csv)\n" if bad_link else b"Review\n",
        "resolved_plan.json": b"{}\n",
        "reuse_ledger.csv": review._csv_bytes(dispositions),
        "decision_context.csv": review._csv_bytes([
            {"configuration": "MCB001", "stream": "funded", "gate": "pass"}
        ]),
        "reuse_context_annotations.csv": review._csv_bytes([
            {"configuration": "MCB001", "status": "posthoc_v02_not_executed",
             "stream": "funded"}
        ] if with_reuse else [], (
            "configuration", "status", "stream", "historical_trade_id",
            "entry_ts_utc", "v02_posthoc_asof")),
    }
    if with_reuse:
        sidecar = {"records": [{"stream": "funded"}]}
        proof = {"posthoc_v02": {
            "payload_sha256": review.canonical_contract_sha256(sidecar),
            "record_count": 1}}
        files["reuse_proofs.json"] = _bytes({"MCB001": proof})
        if bad_sidecar:
            sidecar = {"records": [{"stream": "changed"}]}
        files["reuse_sidecars/MCB001_v02_entry_context.json"] = _bytes({
            "payload": sidecar,
            "sha256": review._sha_bytes(review.canonical_json(sidecar).encode()),
        })
    else:
        files["reuse_proofs.json"] = b"{}\n"
    files["run_manifest.json"] = _bytes({
        "files": [{"path": "README.md", "bytes": len(files["README.md"]),
                   "sha256": review._sha_bytes(files["README.md"])}],
    })
    stamp = (2026, 10, 7, 12, 0, 0)
    manifest = {
        "schema": review.SCHEMA,
        "result_id": "a" * 64, "plan_id": "b" * 64, "export_version": 1,
        "zip_entry_time_utc": stamp,
        "base_review_manifest_sha256": review._sha_bytes(files["run_manifest.json"]),
        "plan_payload_sha256": review._sha_bytes(files["resolved_plan.json"]),
        "row_counts": {"reuse_ledger.csv": 64, "decision_context.csv": 1,
                       "reuse_context_annotations.csv": int(with_reuse)},
        "files": [
            {"path": name, "bytes": len(payload),
             "sha256": "0" * 64 if bad_hash and name == "decision_context.csv"
             else review._sha_bytes(payload)}
            for name, payload in sorted(files.items())
        ],
    }
    if duplicate_manifest_file:
        manifest["files"].append(dict(manifest["files"][0]))
    files["manifest.json"] = _bytes(manifest)
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in files.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 7, 12, 0, 2)
                                   if bad_time and name == "README.md" else stamp)
            archive.writestr(info, payload)


def _mock_plan_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(review.MffuBatchPlanPayload, "model_validate_json",
                        lambda _source: SimpleNamespace(variants=range(64)))
    monkeypatch.setattr(review.MffuBatchPlanEnvelope, "from_payload",
                        lambda _plan: SimpleNamespace(funded_comparison_plan_id="b" * 64))


def test_zip_readback_checks_manifest_rows_and_relative_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_plan_validation(monkeypatch)
    path = tmp_path / "review.zip"
    _write_zip(path)
    receipt = review.verify_mffu_result_review_zip(path)
    assert receipt["passed"] is True
    assert receipt["row_counts"]["reuse_ledger.csv"] == 64
    assert receipt["payloads_verified"] == 7
    for kind, match in (("bad_hash", "checksum"), ("bad_link", "broken"),
                        ("bad_time", "timestamp"),
                        ("duplicate_manifest_file", "more than once")):
        _write_zip(path, **{kind: True})
        with pytest.raises(ValueError, match=match):
            review.verify_mffu_result_review_zip(path)


def test_analysis_tables_keep_unavailable_cash_blank_and_waiting_censoring() -> None:
    interval = {"kind": "initial", "calendar_days": 5,
                "evaluated_trading_days": 3, "censored_start": True,
                "censored_end": True}
    analysis = {
        "schema": review.ANALYSIS_SCHEMA, "firm_key": "myfundedfutures",
        "variant_count": 64, "completed_count": 63, "unavailable_count": 1,
        "cash_basis": "received_less_purchases", "waiting_day_basis": "Chicago",
        "ranking": [{"variant_id": "MCB002", "net_cash_cents": 12500}],
        "matched_pairs": [{"base_id": "MCB001", "challenger_id": "MCB002",
                           "status": "unavailable", "delta_cents": None,
                           "unavailable_ids": ["MCB001"],
                           "trade_diagnostics": {"status": "unavailable",
                                                 "reason": "financial_row_unavailable"}}],
        "interactions": {"fe_x_fl": {
            "cells": [{"status": "unavailable", "delta_cents": None}],
            "available_cells": 0, "unavailable_cells": 1,
            "min_delta_cents": None, "max_delta_cents": None}},
        "waiting_by_variant": {
            "MCB001": {"status": "unavailable", "reason": "failed"},
            "MCB002": {"status": "available", "received_count": 0,
                       "initial": interval, "max_between_receipts": None,
                       "terminal": None, "max_no_receipt_interval": interval,
                       "between_status": "fewer_than_two_receipts",
                       "terminal_status": "no_receipt"}},
        "cash_concentration_by_variant": {
            "MCB001": {"status": "unavailable", "reason": "failed"},
            "MCB002": {"status": "available", "monthly": [{"month": "2025-06",
                       "received_cents": 0, "account_costs_cents": 12500,
                       "net_cash_cents": -12500,
                       "cumulative_net_cash_cents": -12500}]}},
        "trade_concentration_by_variant": {
            "MCB001": {"status": "unavailable", "reason": "failed"},
            "MCB002": {"status": "available", "yearly_by_entry": {},
                       "by_entry_session": {}, "by_resolution": {},
                       "by_htf_zone_id": {"zone-7": {
                           "trades": 1, "net_trade_pnl_cents": 100}},
                       "unknown_htf_zone": {"trades": 1,
                                            "net_trade_pnl_cents": -20},
                       "htf_zone_coverage": {"status": "partial_unknown",
                                             "known_trades": 1, "unknown_trades": 1,
                                             "known_zone_count": 1,
                                             "reason": "historical_zone_id_absent"},
                       "by_account_id": {"acct-1": {
                           "trades": 1, "net_trade_pnl_cents": 100}},
                       "unknown_account": {"trades": 1,
                                           "net_trade_pnl_cents": -20},
                       "account_coverage": {"status": "partial_unknown",
                                            "known_trades": 1, "unknown_trades": 1,
                                            "known_account_count": 1,
                                            "reason": "historical_account_id_absent"},
                       "largest_trades": []}},
        "entry_order_runner_by_variant": {
            "MCB001": {"status": "unavailable", "reason": "failed"},
            "MCB002": {"status": "available", "by_entry_order": {
                "first": {"trades": 1, "net_trade_pnl_cents": 100,
                          "by_resolution": {"no_partial": {
                              "trades": 1, "net_trade_pnl_cents": 100}}},
                "later": {"trades": 0, "net_trade_pnl_cents": 0,
                          "by_resolution": {}}}}},
        "decision_context_coverage_by_variant": {
            "MCB001": {"status": "unavailable", "reason": "failed"},
            "MCB002": {"status": "available", "streams": {
                "funded": {"decision_count": 1, "events": {"entry": 1},
                           "quota_increments": 1}}}},
    }
    tables = review._analysis_tables(analysis)
    pair = next(csv.DictReader(io.StringIO(tables["matched_pairs.csv"].decode())))
    assert pair["delta_cents"] == ""
    waiting = list(csv.DictReader(io.StringIO(tables["waiting_intervals.csv"].decode())))
    assert waiting[0]["reason"] == "failed"
    assert any(row["variant_id"] == "MCB002" and row["censored_end"] == "true"
               for row in waiting)
    monthly = list(csv.DictReader(io.StringIO(
        tables["cash_concentration_monthly.csv"].decode())))
    assert monthly[0]["net_cash_cents"] == ""
    assert monthly[1]["net_cash_cents"] == "-12500"
    trade_detail = next(csv.DictReader(io.StringIO(
        tables["matched_trade_diagnostics.csv"].decode())))
    assert trade_detail["status"] == "unavailable"
    assert "trade_diagnostics" not in next(csv.DictReader(io.StringIO(
        tables["matched_pairs.csv"].decode())))
    order = list(csv.DictReader(io.StringIO(tables["entry_order_runner.csv"].decode())))
    assert any(row["entry_order"] == "first" and row["net_trade_pnl_cents"] == "100"
               for row in order)
    coverage = list(csv.DictReader(io.StringIO(
        tables["decision_context_coverage.csv"].decode())))
    assert coverage[-1]["quota_increments"] == "1"
    zones = [row for row in csv.DictReader(io.StringIO(
        tables["trade_concentration.csv"].decode()))
        if row["group"] == "by_htf_zone_id"]
    all_concentration = list(csv.DictReader(io.StringIO(
        tables["trade_concentration.csv"].decode())))
    assert len(all_concentration) == review._expected_trade_concentration_rows(analysis)
    assert [(row["label"], row["trades"], row["net_trade_pnl_cents"])
            for row in zones] == [("zone-7", "1", "100"), ("unknown", "1", "-20")]
    assert all(row["basis"] == "actual_funded_trade_pnl_not_payout_cash" for row in zones)
    accounts = [row for row in all_concentration if row["group"] == "by_account_id"]
    assert [(row["label"], row["trades"], row["net_trade_pnl_cents"])
            for row in accounts] == [("acct-1", "1", "100"), ("unknown", "1", "-20")]
    assert all(row["basis"] == "actual_funded_trade_pnl_not_payout_cash"
               for row in accounts)
    findings = review._findings(analysis, [
        {"status": "newly_completed"} for _ in range(64)
    ])
    assert "nonexecuted diagnostic" in findings
    assert "XF is an independently replayed actual policy" in findings


def test_frozen_mffu_rules_describe_all_saved_axes() -> None:
    axes = {
        "schedule": ("S0", "S1"), "daily_cap": ("U", "D1"),
        "entry_context": ("F0", "FE", "FL", "FEL"),
        "exit": ("XP", "XF", "XG", "XE"),
        "sizing": ("Q10", "Q6", "QG"), "overhead": ("O0", "O8"),
        "geometry": ("G0", "G05", "G075", "G10"),
    }
    variants = [SimpleNamespace(intent_json=json.dumps({
        axis: values[index % len(values)] for axis, values in axes.items()
    })) for index in range(64)]
    plan = SimpleNamespace(variants=variants,
                           context_policy_version="mq_eod_asof_nominal_2200_chicago_v01",
                           fee_rounding_policy="per_fill_total_round_half_up_cent_v1")
    rules = review._mffu_policy_text(plan)
    assert "## Frozen MFFU 64-policy matrix" in rules
    assert "D1 permits at most one actually filled entry" in rules
    assert "QG selects ten under negative" in rules
    assert "O8 examines the nearest valid price" in rules
    assert "G05, G075, and G10" in rules
    assert "ordered price observation" in rules
    assert "$3.08 + $1.54 + $1.54 = $6.16" in rules
    variants[0] = SimpleNamespace(intent_json=json.dumps({
        **json.loads(variants[0].intent_json), "geometry": "G99"
    }))
    with pytest.raises(ValueError, match="changed matrix"):
        review._mffu_policy_text(plan)


def test_reused_sidecar_is_verified_against_saved_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_plan_validation(monkeypatch)
    path = tmp_path / "review.zip"
    _write_zip(path, with_reuse=True)
    assert review.verify_mffu_result_review_zip(path)["passed"]
    _write_zip(path, with_reuse=True, bad_sidecar=True)
    with pytest.raises(ValueError, match="sealed posthoc reuse sidecar"):
        review.verify_mffu_result_review_zip(path)


def test_streaming_writers_keep_exact_prior_payload_bytes(tmp_path: Path) -> None:
    rows = [{"configuration": "MCB001", "stream": "funded",
             "context": {"gamma_sign": "positive", "age": None,
                         "decision_ts_utc": "2026-06-01T14:00:00+00:00",
                         "gamma_eligible_from_utc": "2026-05-31T03:00:00+00:00"}},
            {"configuration": "MCB002", "stream": "strategy",
             "context": {"gamma_sign": "negative", "age": 7}}]
    csv_path = tmp_path / "decision_context.csv"
    review._write_csv_file(csv_path, rows, ("configuration", "stream"))
    assert csv_path.read_bytes() == review._csv_bytes(rows, ("configuration", "stream"))
    exported = next(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
    assert json.loads(exported["context"])["decision_ts_utc"] == (
        "2026-06-01T14:00:00+00:00"
    )
    json_path = tmp_path / "analysis.json"
    payload = {"rows": rows, "basis": "saved exact cents"}
    review._write_json_file(json_path, payload)
    assert json_path.read_bytes() == review._json_bytes(payload)


def test_staging_must_be_outside_repository() -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        review.publish_mffu_result_review(
            plan_id="a" * 64, result_id="b" * 64, store_root=Path("missing"),
            state_root=Path("missing_state"),
            staging_root=review.REPO_ROOT / "build", reports_root=review.REPO_ROOT / "reports",
            review_findings="Reviewed", screenshots={},
        )


def test_result_review_requires_ordered_saved_dispositions_and_context() -> None:
    plan = SimpleNamespace(
        variants=[SimpleNamespace(variant_id=f"MCB{index:03d}")
                  for index in range(1, 65)],
        model_dump=lambda **_kwargs: {"frozen": "plan"},
    )
    dispositions = [
        {"variant_id": variant.variant_id, "status": "newly_completed",
         "reason": None, "reused_from": None}
        for variant in plan.variants
    ]
    result = {
        "funded_comparison_plan_id": "a" * 64,
        "validation": {"passed": True},
        "mffu_batch": {"schema": review.RESULT_SCHEMA, "plan": {"frozen": "plan"},
                       "dispositions": dispositions,
                       "reuse_proofs": {}, "reuse_context_annotations": [],
                       "decision_context": [{"configuration": "MCB001",
                                             "stream": "funded", "context": {
                                                 "decision_ts_utc":
                                                 "2026-06-01T14:00:00+00:00"}}]},
        "mffu_analysis": {"schema": review.ANALYSIS_SCHEMA,
                          "variant_count": 64,
                          "matched_pairs": [{"trade_diagnostics": {}}] * 188,
                          "entry_order_runner_by_variant": {
                              row["variant_id"]: {} for row in dispositions},
                          "decision_context_coverage_by_variant": {
                              row["variant_id"]: {} for row in dispositions}},
    }
    rows, _analysis = review._assert_result(plan, "a" * 64, result)
    assert len(rows) == 64
    result["mffu_batch"]["decision_context"][0]["context"].pop("decision_ts_utc")
    with pytest.raises(ValueError, match="decision-context"):
        review._assert_result(plan, "a" * 64, result)
    result["mffu_batch"]["decision_context"][0]["context"]["decision_ts_utc"] = (
        "2026-06-01T14:00:00+00:00"
    )
    result["mffu_batch"]["dispositions"] = dispositions[:-1]
    with pytest.raises(ValueError, match="64 ordered"):
        review._assert_result(plan, "a" * 64, result)
    result["mffu_batch"]["dispositions"] = dispositions
    result["mffu_batch"]["decision_context"][0]["configuration"] = "unknown"
    with pytest.raises(ValueError, match="decision-context"):
        review._assert_result(plan, "a" * 64, result)
