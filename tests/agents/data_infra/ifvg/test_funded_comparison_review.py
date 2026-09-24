"""Automatic funded configuration-comparison review folder (SYNTHETIC fixture, tmp_path)."""

from __future__ import annotations

import csv
import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg import funded_comparison_review as review
from alpha_lab.agents.data_infra.ifvg.funded_comparison_review import (
    ALLOWLIST,
    REVIEW_FINDINGS,
    ComparisonReviewExportError,
    next_comparison_export_version,
    publish_comparison_review_folder,
)
from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
    comparison_headline_figures,
)
from tests.propsim.funded.comparison_fixture import comparison_fixture_result

RESULT_ID = "ab" * 32
LEDGER = [
    {"event_id": "prior_study_completed", "event_type": "comparison_completed",
     "ledger_origin": "imported_verified_history", "imported_from": "prior_package",
     "display_time_chicago": "September 18, 2026 07:57 PM CDT",
     "run_id": "5c8028bcd53414f0adaa10ed8ba5bafdb13f666c1cdfa1c95e8ce0cbb41a49b4",
     "actual_application": "scripts/run_ifsm_research_ui.py",
     "note": "Archive at C:\\Users\\someone\\archive.zip verified; digest "
             "fe9c82699f942c473764860f8fd4debb9fee0a24cd2109e20d325f2aa354251a.",
     "limits": "Same-sample historical comparison."},
    {"event_id": "funded_comparison_x_completed", "event_type": "run_completed",
     "ledger_origin": "funded_payout_lane", "status": "completed",
     "funded_comparison_plan_id": "ab" * 32, "funded_comparison_result_id": "cd" * 32,
     "configurations_completed": 2, "configurations_not_completed": ["S9_D40"],
     "highest_net_cash_by_firm": {"TakeProfitTrader": [{"configuration": "S0_D160",
                                                        "net_cash_earned_usd": 2203.78}]}},
]


@pytest.fixture()
def result():
    return comparison_fixture_result()


def _files(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cents(text: str) -> int:
    return int((Decimal(text) * 100).quantize(Decimal(1)))


def _publish(result, root: Path, **kwargs):
    return publish_comparison_review_folder(
        result=result, result_id=RESULT_ID, reports_root=root, ledger_entries=LEDGER,
        export_version=kwargs.pop("export_version", 1), **kwargs)


def test_publishes_exact_allowlist_and_reconciles(tmp_path, result):
    published = _publish(result, tmp_path)
    folder = published.path
    assert folder == (tmp_path / "funded_comparison"
                      / f"funded_comparison_{RESULT_ID[:16]}_export_v1")
    assert published.checks_passed
    assert _files(folder) == sorted(ALLOWLIST)
    assert REVIEW_FINDINGS not in _files(folder)
    assert not any(re.search(r"credit|growth", f, re.IGNORECASE) for f in _files(folder))
    assert not any(Path(f).suffix in {".py", ".parquet", ".zip", ".patch", ".ipynb"}
                   for f in _files(folder))
    assert [p.name for p in (tmp_path / "funded_comparison").iterdir()] == [folder.name]

    # every CSV parses; configuration results equal the screen's exact figures
    results = _rows(folder / "configuration_results.csv")
    assert len(results) == 6
    screen = comparison_headline_figures(result)
    for row in results:
        figures = screen[row["pair_id"]]
        if row["status"] == "Completed":
            assert _cents(row["net_cash_earned_usd"]) == figures["net_cash_earned_cents"]
            assert _cents(row["payouts_received_usd"]) == figures["payouts_received_cents"]
            assert _cents(row["account_costs_usd"]) == figures["account_costs_cents"]
            assert int(row["accounts_purchased"]) == figures["accounts_purchased"]
        else:  # not completed: reason, no figures (never zeros)
            assert row["status"] == "Not completed"
            assert row["net_cash_earned_usd"] == "" and row["reason"]
    # unfavorable and zero-payout configurations are included
    nets = {row["pair_id"]: row["net_cash_earned_usd"] for row in results}
    assert nets["S3_D80|takeprofittrader"] == "-102.00"
    assert nets["S0_D160|myfundedfutures"] == "-250.00"

    configurations = _rows(folder / "configurations.csv")
    assert [r["configuration"] for r in configurations] == ["S0_D160", "S3_D80", "S9_D40"]
    assert configurations[0]["Higher-timeframe gap charts"] == "one-hour and four-hour"
    assert configurations[2]["status"] == "Not completed"
    for name in ("account_journeys", "cash_ledger", "payout_events", "monthly_results",
                 "trades", "account_events", "rule_boundary_evidence",
                 "execution_evidence"):
        assert len(_rows(folder / f"{name}.csv")) == len(result["tables"][name])

    summary = json.loads((folder / "validation_summary.json").read_text(encoding="utf-8"))
    assert summary["export_checks_passed"]
    assert summary["export_checks"]["screen_export_agreement"]["passed"]
    assert summary["export_checks"]["cash_ledger_reconciles_to_result_cents"]["pairs_checked"] == 4
    assert summary["result_validation"]["configurations_not_completed"] == [
        {"configuration": "S9_D40",
         "reason": "Recorded prices for 2026-02-03 are unavailable."}]
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_version"] == 1
    assert not manifest["independent_review_findings_included"]
    assert {f["path"] for f in manifest["files"]} == set(ALLOWLIST) - {"run_manifest.json"}
    rules = json.loads((folder / "firm_rules.json").read_text(encoding="utf-8"))
    assert all("monthly_credits" not in p and "growth_share_bps" not in p
               for p in rules["firm_profiles"])
    for chart in ("charts/net_cash_by_configuration.png", "charts/cash_over_time.png"):
        assert (folder / chart).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_documents_are_readable_without_paths_hashes_or_ids(tmp_path, result):
    folder = _publish(result, tmp_path).path
    readme = (folder / "README.md").read_text(encoding="utf-8")
    assert "The independent review had not been recorded" in readme
    assert "not a probability of future payouts" in readme
    assert "Not completed: Recorded prices for 2026-02-03 are unavailable." in readme
    assert "never added together" in readme
    forbidden = re.compile(
        r"[A-Za-z]:\\|/Users/|\b[0-9a-f]{32,}\b|scripts/|\.py\b|\|takeprofittrader"
        r"|\bS\d+_D\d+\b|0f1e2d3c-aaaa")
    for name in [f for f in ALLOWLIST if f.endswith(".md")]:
        text = (folder / name).read_text(encoding="utf-8")
        assert not forbidden.search(text), name
    for name in ("README.md", "RESULTS.md"):  # no credit/growth figures in this mode
        assert not re.search(r"credit|growth", (folder / name).read_text(encoding="utf-8"),
                             re.IGNORECASE), name
    ledger_md = (folder / "RESEARCH_LEDGER.md").read_text(encoding="utf-8")
    assert "Imported history from an earlier verified study" in ledger_md
    assert "Same-sample historical comparison." in ledger_md
    assert "(internal file location omitted)" in ledger_md
    # the machine ledger keeps the entries verbatim
    lines = (folder / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == LEDGER


def test_review_findings_are_optional_and_listed_when_given(tmp_path, result):
    findings = ("No blocking defect found. Checked cash ledger joins in "
                "C:\\Users\\x\\review.txt.")
    folder = _publish(result, tmp_path, review_findings=findings).path
    assert _files(folder) == sorted((*ALLOWLIST, REVIEW_FINDINGS))
    text = (folder / REVIEW_FINDINGS).read_text(encoding="utf-8")
    assert text.startswith("# Independent review findings")
    assert "No blocking defect found." in text and "C:\\Users" not in text
    assert "`REVIEW_FINDINGS.md`" in (folder / "README.md").read_text(encoding="utf-8")
    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["independent_review_findings_included"]
    with pytest.raises(ValueError):
        _publish(result, tmp_path, export_version=2, review_findings="   ")


def test_versioning_and_repack_never_recompute(tmp_path, result):
    assert next_comparison_export_version(tmp_path, RESULT_ID) == 1
    first = _publish(result, tmp_path).path
    assert next_comparison_export_version(tmp_path, RESULT_ID) == 2
    with pytest.raises(FileExistsError):
        _publish(result, tmp_path, export_version=1)
    second = _publish(result, tmp_path,
                      export_version=next_comparison_export_version(tmp_path, RESULT_ID)).path
    assert second.name.endswith("_export_v2")
    for name in ("configuration_results.csv", "cash_ledger.csv", "trades.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert next_comparison_export_version(tmp_path, "cd" * 32) == 1


def test_agreement_check_fails_on_tampered_figures(tmp_path, result):
    tampered = comparison_fixture_result()
    tampered["summaries_cents"]["S0_D160|takeprofittrader"]["net_cash_earned_cents"] += 1
    with pytest.raises(ComparisonReviewExportError) as error:
        _publish(tampered, tmp_path)
    assert "screen_export_agreement" in str(error.value)
    assert not error.value.checks["screen_export_agreement"]["passed"]
    assert error.value.checks["screen_export_agreement"]["mismatched"] == [
        "S0_D160|takeprofittrader"]
    # nothing published and no staging folder left behind
    assert list((tmp_path / "funded_comparison").iterdir()) == []


def test_agreement_uses_the_screen_function(tmp_path, result, monkeypatch):
    real = review.comparison_headline_figures

    def shifted(res):
        out = real(res)
        out["S3_D80|myfundedfutures"] = {**out["S3_D80|myfundedfutures"],
                                         "account_costs_cents": 1}
        return out

    monkeypatch.setattr(review, "comparison_headline_figures", shifted)
    with pytest.raises(ComparisonReviewExportError, match="screen_export_agreement"):
        _publish(result, tmp_path)


def test_cash_ledger_tampering_and_failed_validation_block_publication(tmp_path, result):
    tampered = comparison_fixture_result()
    tampered["tables"]["cash_ledger"][0]["amount_usd"] = 1.0
    with pytest.raises(ComparisonReviewExportError,
                       match="cash_ledger_reconciles_to_result_cents"):
        _publish(tampered, tmp_path)
    failed = comparison_fixture_result()
    failed["validation"]["passed"] = False
    with pytest.raises(ComparisonReviewExportError, match="result_validation_passed"):
        _publish(failed, tmp_path)
    assert list((tmp_path / "funded_comparison").iterdir()) == []


def test_rejects_bad_inputs(tmp_path, result):
    with pytest.raises(ValueError):
        publish_comparison_review_folder(result=result, result_id="not-hex",
                                         reports_root=tmp_path, ledger_entries=LEDGER,
                                         export_version=1)
    with pytest.raises(ValueError):
        publish_comparison_review_folder(result=result, result_id=RESULT_ID,
                                         reports_root=tmp_path, ledger_entries=[],
                                         export_version=1)
    with pytest.raises(ValueError):
        _publish(result, tmp_path, export_version=0)


def _with_stop_defect(result):
    """The saved result as loaded after the stop-difference correction."""

    import copy

    from alpha_lab.propsim.funded.comparison_result import apply_reporting_corrections

    broken = copy.deepcopy(result)
    pair = next(k for k, s in broken["summaries_cents"].items() if s["status"] == "Completed")
    broken["summaries_cents"][pair]["stop_slippage_cents"] += 500
    return apply_reporting_corrections(broken), pair


def test_reporting_corrections_are_listed_and_never_move_headline_money(tmp_path, result):
    corrected, pair = _with_stop_defect(result)
    published = _publish(corrected, tmp_path)
    folder = published.path
    rows = _rows(folder / "reporting_corrections.csv")
    assert [r["pair_id"] for r in rows] == [pair]
    assert _cents(rows[0]["stop_difference_saved_usd"]) - _cents(
        rows[0]["stop_difference_corrected_usd"]) == 500
    checks = json.loads((folder / "validation_summary.json").read_text())["export_checks"]
    assert checks["reporting_corrections_are_summary_only"]["passed"]
    assert checks["screen_export_agreement"]["passed"]
    assert comparison_headline_figures(corrected) == comparison_headline_figures(result)
    readme = (folder / "README.md").read_text(encoding="utf-8")
    assert "## Reporting corrections in this export version" in readme
    assert "The saved economic result is unchanged" in readme


def test_supplements_are_written_under_fixed_names_only(tmp_path, result):
    with pytest.raises(ValueError, match="unknown review supplements"):
        _publish(result, tmp_path, supplements={"raw_prices": [{"a": 1}]})
    supplements = {
        "trading_calendar": [{"trading_day": "2026-03-02", "shortened_day": False}],
        "approximated_minutes": [{"configuration": "S0_D160", "minute_open_utc": "x",
                                  "ordering_can_change_final_exit_or_result": "False",
                                  "ordering_can_change_only_fill_time_within_minute": "True"}],
        "trade_boundary_check": {"passed": True, "trades_checked": 3,
                                 "trades_open_past_midnight_inside_one_trading_day": 1,
                                 "trades_outside_their_trading_day_count": 0},
    }
    folder = _publish(result, tmp_path, supplements=supplements).path
    assert _files(folder) == sorted((*ALLOWLIST, "approximated_minutes.csv",
                                     "trading_calendar.csv"))
    rules = (folder / "TRADING_RULES.md").read_text(encoding="utf-8")
    assert "nothing is held overnight" not in rules
    assert "1 were open past midnight inside one trading day" in rules
    assert "## Funded execution versus the strategy-only view" in rules
    readme = (folder / "README.md").read_text(encoding="utf-8")
    assert "only its fill time within that minute can differ" in readme
    dictionary = (folder / "DATA_DICTIONARY.md").read_text(encoding="utf-8")
    assert "## approximated_minutes.csv" in dictionary


def test_large_payout_removal_is_sensitivity_information_not_a_rejection(tmp_path, result):
    question = (_publish(result, tmp_path).path / "QUESTION.md").read_text(encoding="utf-8")
    falsify = question.split("## What would falsify a conclusion")[1].split("##")[0]
    assert "large payout" not in falsify
    assert "## Sensitivity information (not a rejection test)" in question


def test_final_receipt_rechecks_the_published_payload_inventory(tmp_path, result):
    published = _publish(result, tmp_path, review_findings="# Findings\n\nNone.")
    receipt = published.receipt
    assert receipt["passed"] and receipt["mismatched"] == [] and receipt["unlisted"] == []
    assert receipt["payload_files_listed"] == len(ALLOWLIST) + 1 - 1  # + findings - manifest
    manifest = json.loads((published.path / "run_manifest.json").read_text())
    assert manifest["payload_files"] == receipt["payload_files_listed"]
    (published.path / "trades.csv").write_text("tampered", encoding="utf-8")
    (published.path / "extra.md").write_text("x", encoding="utf-8")
    again = review.verify_published_folder(published.path)
    assert not again["passed"]
    assert again["mismatched"] == ["trades.csv"] and again["unlisted"] == ["extra.md"]
