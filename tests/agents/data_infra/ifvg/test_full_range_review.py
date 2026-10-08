"""Synthetic strict full-range publication: saved tables, supplements and receipt."""

from __future__ import annotations

import copy
import csv
import json
from io import BytesIO

import pytest
from PIL import Image

from alpha_lab.agents.data_infra.ifvg import funded_comparison_review as review
from alpha_lab.propsim.funded.full_range_reporting import TABLE_FILES, attach_full_range_reports
from tests.propsim.funded.comparison_fixture import comparison_fixture_parts
from tests.propsim.funded.test_full_range_reporting import reporting_fixture


def _full_result():
    original, source_outputs = comparison_fixture_parts()
    result = copy.deepcopy(original)
    result["tables"] = {name: [] for name in original["tables"]}
    result["summaries"] = {}
    result["summaries_cents"] = {}
    result["validation"]["configurations_not_completed"] = []
    names = []
    outputs = []
    failures = []
    for offset in (0, 3):
        mapping = {"S0_D160": f"C{offset+1:02d}", "S3_D80": f"C{offset+2:02d}",
                   "S9_D40": f"C{offset+3:02d}"}
        names.extend(mapping.values())
        for table, rows in original["tables"].items():
            for row in copy.deepcopy(rows):
                if "configuration" in row:
                    old = row["configuration"]
                    row["configuration"] = mapping[old]
                    if "pair_id" in row:
                        row["pair_id"] = row["pair_id"].replace(old + "|", mapping[old] + "|")
                result["tables"][table].append(row)
        for summary_name in ("summaries", "summaries_cents"):
            for pair, row in original[summary_name].items():
                row = copy.deepcopy(row)
                old = row["configuration"]
                row["configuration"] = mapping[old]
                row["pair_id"] = pair.replace(old + "|", mapping[old] + "|")
                result[summary_name][row["pair_id"]] = row
        for output in copy.deepcopy(source_outputs):
            output["configuration"] = mapping[output["configuration"]]
            output["strategy_trades_no_account"] = []
            output["sizing"] = {"instrument": "mini", "quantity": 1,
                                "tick_value_cents": 500, "cost_per_contract_mills": 5140}
            outputs.append(output)
        failures.append({"configuration": mapping["S9_D40"], "reason": "Synthetic failure"})
    result["configurations_requested"] = 6
    result["configurations_completed"] = 4
    result["validation"]["configurations_not_completed"] = failures
    kwargs = reporting_fixture()[3]
    kwargs["configuration_names"] = tuple(names)
    attach_full_range_reports(result, outputs, failures, **kwargs)
    return result


def _png():
    target = BytesIO()
    Image.new("RGB", (800, 450), "white").save(target, format="PNG")
    return target.getvalue()


def _supplements(result):
    spec = result["full_range_reporting"]
    return {
        "run_context": {key: spec[key] for key in
                        ("evaluation_dates", "warmup_dates", "cutoff_utc")},
        "screenshot_completed_comparison": _png(),
        "screenshot_effective_plan": _png(),
        "engine_integration": "# Synthetic integration\n\nSource hash " + "a" * 64,
        "document_overrides": {"README.md": "# Synthetic comparison\n\n"
                               "Canonical equivalents: configuration_results.csv is comparison; "
                               "strategy_metrics.csv is strategy results; "
                               "trades.csv is funded trades; "
                               "run_manifest.json is the manifest."},
        "validation_evidence": {"synthetic_test_only": True, "market_files_opened": 0},
    }


def _publish(tmp_path, result, supplements):
    return review.publish_comparison_review_folder(
        result=result, result_id="ef" * 32, reports_root=tmp_path,
        ledger_entries=[{"event_type": "synthetic_test", "note": "No historical run"}],
        export_version=1, supplements=supplements,
    )


def test_full_range_saved_tables_and_fixed_supplements_are_manifest_verified(tmp_path):
    result = _full_result()
    published = _publish(tmp_path, result, _supplements(result))
    assert published.receipt["passed"]
    assert set(TABLE_FILES.values()) <= {entry["path"] for entry in published.manifest["files"]}
    summary = json.loads((published.path / "validation_summary.json").read_text())
    assert summary["export_checks"]["saved_full_range_reports_reconcile"]["passed"]
    assert summary["export_checks"]["full_range_csv_cells_equal_saved_tables"]["passed"]
    assert summary["execution_and_review_evidence"]["synthetic_test_only"]
    with (published.path / "daily_activity.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 18 * 253
    assert (published.path / "README.md").read_text().startswith("# Synthetic comparison")
    assert "Source hash" in (published.path / "ENGINE_INTEGRATION.md").read_text()
    (published.path / "screenshots/02_effective_plan.png").write_bytes(_png() + b"tampered")
    assert not review.verify_published_folder(published.path)["passed"]


def test_context_scope_mismatch_and_unknown_payloads_are_refused(tmp_path):
    result = _full_result()
    supplements = _supplements(result)
    supplements["run_context"]["evaluation_dates"] = []
    with pytest.raises(review.ComparisonReviewExportError, match="run_context_matches_saved_scope"):
        _publish(tmp_path, result, supplements)
    assert not list(tmp_path.glob("funded_comparison/*"))
    supplements = _supplements(result)
    supplements["raw_data"] = []
    with pytest.raises(ValueError, match="unknown review supplements"):
        _publish(tmp_path, result, supplements)


def test_same_row_count_but_tampered_csv_cells_are_refused(tmp_path, monkeypatch):
    result = _full_result()
    original = review._write_csv

    def broken(path, rows):
        original(path, rows)
        if path.name == "daily_activity.csv":
            text = path.read_text(encoding="utf-8")
            path.write_text(text.replace("flat_all_day", "wrong_status", 1), encoding="utf-8")

    monkeypatch.setattr(review, "_write_csv", broken)
    with pytest.raises(review.ComparisonReviewExportError,
                       match="full_range_csv_cells_equal_saved_tables"):
        _publish(tmp_path, result, _supplements(result))


def test_all_failed_report_verifies_absence_of_money_without_claiming_economic_success(tmp_path):
    result = _full_result()
    names = tuple(result["full_range_reporting"]["configuration_names"])
    failures = [{"configuration": name, "reason": "Synthetic worker failure"} for name in names]
    for table in result["tables"]:
        result["tables"][table] = []
    for pair, old in list(result["summaries_cents"].items()):
        summary = {"pair_id": pair, "configuration": old["configuration"],
                   "configuration_label": old.get("configuration_label", old["configuration"]),
                   "firm_key": old["firm_key"], "firm": old.get("firm"),
                   "status": "Not completed", "reason": "Synthetic worker failure"}
        result["summaries_cents"][pair] = summary
        result["summaries"][pair] = summary
        result["tables"]["pair_results"].append(summary)
    result["validation"]["passed"] = False
    result["validation"]["configurations_not_completed"] = failures
    result["configurations_completed"] = 0
    kwargs = reporting_fixture()[3]
    kwargs["configuration_names"] = names
    attach_full_range_reports(result, [], failures, **kwargs)
    published = _publish(tmp_path, result, _supplements(result))
    validation = json.loads((published.path / "validation_summary.json").read_text())
    assert validation["export_checks_passed"]
    assert not validation["result_validation"]["passed"]
    check = validation["export_checks"]["result_validation_passed"]
    assert check["economic_validation_passed"] is False
    assert len(result["tables"]["strategy_metrics"]) == 6
    assert not result["tables"]["cash_ledger"]
