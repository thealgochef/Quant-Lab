"""Automatic funded-payout review folder (SYNTHETIC sample results, tmp_path only)."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.funded_review_package import (
    ALLOWLIST,
    CSV_TABLES,
    FundedReviewExportError,
    next_export_version,
    publish_funded_review_folder,
)
from tests.propsim.funded.builders import synthetic_sample_result
from tests.propsim.funded.sample_variants import all_failed_no_payout_result

RESULT_ID = "ab" * 32
LEDGER = [
    {"recorded_on": "2026-09-11", "title": "Daily-close session study",
     "status": "completed", "objective": "historical activity objective (not the winner "
     "criterion for funded payouts)"},
    {"recorded_on": "2026-09-22", "title": "Funded payout engineering sample",
     "status": "completed", "purpose": "engineering sample"},
]


@pytest.fixture(scope="module")
def sample():
    return synthetic_sample_result()


def _files(folder: Path) -> list[str]:
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cents(text: str) -> int:
    from decimal import Decimal

    return int((Decimal(text) * 100).quantize(Decimal(1)))


def test_publishes_exact_allowlist_and_reconciles(tmp_path, sample):
    published = publish_funded_review_folder(
        result=sample, funded_result_id=RESULT_ID, reports_root=tmp_path,
        ledger_entries=LEDGER)
    folder = published.path
    assert folder == tmp_path / "funded_payout" / f"funded_payout_{RESULT_ID[:16]}_export_v1"
    assert published.checks_passed
    assert _files(folder) == sorted(ALLOWLIST)
    assert {Path(f).suffix for f in _files(folder)} <= {".md", ".json", ".jsonl", ".csv",
                                                         ".png"}
    # no staging leftovers next to the published folder
    assert [p.name for p in (tmp_path / "funded_payout").iterdir()] == [folder.name]

    for table, filename in CSV_TABLES.items():
        assert len(_rows(folder / filename)) == len(sample["tables"][table]), filename

    ledger = _rows(folder / "cash_ledger.csv")
    instances = {r["firm_key"]: r for r in _rows(folder / "instance_results.csv")}
    for key, s in sample["summaries_cents"].items():
        mine = [r for r in ledger if r["firm_key"] == key]
        receipts = sum(_cents(r["amount_usd"]) for r in mine if r["kind"] == "payout_received")
        costs = sum(_cents(r["amount_usd"]) for r in mine if r["kind"] == "account_purchase")
        assert receipts == s["payouts_received_cents"]
        assert costs == s["acquisition_costs_cents"]
        assert receipts - costs == s["net_cash_earned_cents"]
        assert _cents(instances[key]["net_cash_earned_usd"]) == s["net_cash_earned_cents"]

    manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest == published.manifest
    assert manifest["funded_result_id"] == RESULT_ID
    assert manifest["funded_plan_id"] == sample["funded_plan_id"]
    assert manifest["export_version"] == 1
    listed = {f["path"]: f for f in manifest["files"]}
    assert set(listed) == set(ALLOWLIST) - {"run_manifest.json"}
    for rel, entry in listed.items():
        data = (folder / rel).read_bytes()
        assert entry["bytes"] == len(data)
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()

    validation = json.loads((folder / "validation_summary.json").read_text(encoding="utf-8"))
    assert validation["result_validation"]["passed"] is True
    assert validation["export_checks"]["screen_export_agreement"]["passed"] is True
    assert validation["export_checks_passed"] is True

    lines = (folder / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == LEDGER
    history = (folder / "RESEARCH_LEDGER.md").read_text(encoding="utf-8")
    assert "Daily-close session study" in history and "Funded payout engineering" in history

    question = (folder / "QUESTION.md").read_text(encoding="utf-8")
    assert "net cash earned after all modeled account costs" in question
    assert "historical only" in question and "not** the winner" in question
    rules = (folder / "TRADING_RULES.md").read_text(encoding="utf-8")
    for phrase in ("Fresh funded start", "no weekend holding", "Credits, grants and growth",
                   "$2,100", "$500 GROSS", "Processing clock", "two business days",
                   "MyFundedFutures equality assumption", "approximation"):
        assert phrase in rules, phrase
    results = (folder / "RESULTS.md").read_text(encoding="utf-8")
    assert "TakeProfitTrader" in results and "MyFundedFutures" in results
    assert "Accounts lost" in results and "still processing at the end" in results
    for limitation in sample["limitations"]:
        assert limitation in results
    assert "(partial month)" in results
    for png in ("charts/cash_over_time.png", "charts/account_journeys.png"):
        assert (folder / png).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_refuses_to_overwrite_and_repack_increments_version(tmp_path, sample):
    first = publish_funded_review_folder(
        result=sample, funded_result_id=RESULT_ID, reports_root=tmp_path,
        ledger_entries=LEDGER)
    before = {p: p.read_bytes() for p in first.path.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        publish_funded_review_folder(
            result=sample, funded_result_id=RESULT_ID, reports_root=tmp_path,
            ledger_entries=LEDGER)
    assert {p: p.read_bytes() for p in first.path.rglob("*") if p.is_file()} == before
    version = next_export_version(tmp_path, RESULT_ID)
    assert version == 2
    second = publish_funded_review_folder(
        result=sample, funded_result_id=RESULT_ID, reports_root=tmp_path,
        ledger_entries=LEDGER, export_version=version)
    assert second.path.name.endswith("_export_v2")
    one = (first.path / "instance_results.csv").read_bytes()
    assert (second.path / "instance_results.csv").read_bytes() == one


def test_tampered_summary_is_detected_and_nothing_is_published(tmp_path, sample):
    tampered = copy.deepcopy(sample)
    tampered["summaries_cents"]["takeprofittrader"]["payouts_received_cents"] += 1
    with pytest.raises(FundedReviewExportError) as caught:
        publish_funded_review_folder(
            result=tampered, funded_result_id=RESULT_ID, reports_root=tmp_path,
            ledger_entries=LEDGER)
    assert not caught.value.checks["cash_ledger_reconciles_to_result_cents"]["passed"]
    assert list((tmp_path / "funded_payout").iterdir()) == []


def test_tampered_table_breaks_screen_export_agreement(tmp_path, sample):
    tampered = copy.deepcopy(sample)
    tampered["tables"]["instance_results"][1]["net_cash_earned_usd"] += 1.0
    with pytest.raises(FundedReviewExportError) as caught:
        publish_funded_review_folder(
            result=tampered, funded_result_id=RESULT_ID, reports_root=tmp_path,
            ledger_entries=LEDGER)
    assert not caught.value.checks["screen_export_agreement"]["passed"]
    assert list((tmp_path / "funded_payout").iterdir()) == []


def test_failed_result_validation_is_refused(tmp_path, sample):
    tampered = copy.deepcopy(sample)
    tampered["validation"] = {"passed": False, "checks": {}}
    with pytest.raises(FundedReviewExportError) as caught:
        publish_funded_review_folder(
            result=tampered, funded_result_id=RESULT_ID, reports_root=tmp_path,
            ledger_entries=LEDGER)
    assert not caught.value.checks["result_validation_passed"]["passed"]
    assert list((tmp_path / "funded_payout").iterdir()) == []


def test_dropped_csv_row_fails_row_count_check(tmp_path, sample, monkeypatch):
    from alpha_lab.agents.data_infra.ifvg import funded_review_package as package

    original = package._write_csv

    def lossy(path, table, rows):
        original(path, table, rows[:-1] if table == "trades" else rows)

    monkeypatch.setattr(package, "_write_csv", lossy)
    with pytest.raises(FundedReviewExportError) as caught:
        publish_funded_review_folder(
            result=sample, funded_result_id=RESULT_ID, reports_root=tmp_path,
            ledger_entries=LEDGER)
    counts = caught.value.checks["csv_row_counts_equal_result_tables"]
    assert not counts["passed"]
    assert counts["tables"]["trades.csv"]["exported"] == len(sample["tables"]["trades"]) - 1
    assert list((tmp_path / "funded_payout").iterdir()) == []


@pytest.mark.parametrize("extra", ["verify.py", "notes.ipynb", "cache/data.parquet",
                                   "extra.csv"])
def test_forbidden_or_extra_files_block_publication(tmp_path, sample, monkeypatch, extra):
    from alpha_lab.agents.data_infra.ifvg import funded_review_package as package

    original = package._write_all

    def with_extra(staging, *args, **kwargs):
        original(staging, *args, **kwargs)
        target = staging / extra
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")

    monkeypatch.setattr(package, "_write_all", with_extra)
    with pytest.raises(FundedReviewExportError) as caught:
        publish_funded_review_folder(
            result=sample, funded_result_id=RESULT_ID, reports_root=tmp_path,
            ledger_entries=LEDGER)
    allow = caught.value.checks["allowlist_before_manifest"]
    assert not allow["passed"] and extra in allow["unexpected"]
    assert list((tmp_path / "funded_payout").iterdir()) == []


def test_no_payout_result_publishes_with_explicit_bad_outcomes(tmp_path):
    result = all_failed_no_payout_result()
    published = publish_funded_review_folder(
        result=result, funded_result_id="cd" * 32, reports_root=tmp_path,
        ledger_entries=LEDGER)
    folder = published.path
    assert _files(folder) == sorted(ALLOWLIST)
    assert _rows(folder / "payout_events.csv") == []
    header = (folder / "payout_events.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "gross_usd" in header
    results = (folder / "RESULTS.md").read_text(encoding="utf-8")
    assert results.count("No payouts were received in this period.") == 2
    assert "Every account was lost by the end of the period." in results


def test_rejects_missing_ledger(tmp_path, sample):
    with pytest.raises(ValueError):
        publish_funded_review_folder(
            result=sample, funded_result_id=RESULT_ID, reports_root=tmp_path,
            ledger_entries=[])
    assert not (tmp_path / "funded_payout").exists()


def test_imported_history_with_file_locations_stays_verbatim_in_jsonl_only(tmp_path, sample):
    history = [{"event_id": "old", "ledger_origin": "imported_verified_history",
                "archive": r"C:\Users\someone\Documents\reports\old_run.zip",
                "note": "copied from /Users/someone/work"}, *LEDGER]
    published = publish_funded_review_folder(
        result=sample, funded_result_id=RESULT_ID, reports_root=tmp_path,
        ledger_entries=history)
    readable = (published.path / "RESEARCH_LEDGER.md").read_text(encoding="utf-8")
    assert "someone" not in readable
    assert readable.count("(internal file location omitted)") == 2
    first = json.loads((published.path / "ledger.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    assert first["archive"].endswith("old_run.zip")  # the cumulative record is verbatim
