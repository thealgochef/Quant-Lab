"""Automatic compact review folder for a completed funded configuration comparison.

``publish_comparison_review_folder`` writes EXACTLY the comparison allowlist
(documents and necessary data only) from the ONE saved comparison result
(``funded_comparison_result_v1``). It stages into a sibling temporary folder,
verifies the staged files (exact allowlist, CSVs parse with the result's row
counts, cash ledger reconciles with the result's exact cents, screen/export
agreement through :func:`comparison_headline_figures` — the same function the
screen uses — no file paths or hashes in the documents, and the result's own
validation passed), then publishes with one atomic rename.

An existing folder is never overwritten: a repack uses a higher export version
and never recomputes or changes the economic result. Reporting corrections
applied when the saved result was loaded (summary-only; see
``comparison_result.apply_reporting_corrections``) are listed in
``reporting_corrections.csv`` and proven not to move any headline figure.

Optional compact evidence supplied by the caller (``supplements``) is written
under fixed names only: the exact configuration bindings, the saved trading
calendar and trade-boundary check, the substituted-minute analysis and the
reference-record reconciliation. After the atomic rename the published folder
is re-read and checked against its own manifest (the final payload receipt).
Credit and growth tables belong to the separate budgeted mode and are never
written here. No scripts, source, patches, raw price dumps or per-account
folders are ever written.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from alpha_lab.agents.data_infra.ifvg.funded_review_package import (
    _cell,
    _comparator_words,
    _date_formatter,
    _describe_column,
    _dollar_formatter,
    _matplotlib,
    _twelve_hour,
    _write_json,
    _write_text,
)
from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
    COMPARISON_HEADLINE_KEYS,
    COMPLETED,
    NOT_COMPLETED,
    STRATEGY_METRICS_NOTE,
    ComparisonView,
    cash_chart_points,
    comparison_headline_figures,
    configuration_columns,
    execution_sources,
    plain_configuration_label,
    plain_reason,
    present_comparison,
    present_pair_detail,
    present_strategy_metrics,
    proxy_micro_positions,
)
from alpha_lab.agents.data_infra.ifvg.presentation.funded_results import format_usd

__all__ = [
    "ALLOWLIST",
    "REVIEW_FINDINGS",
    "CSV_TABLES",
    "ComparisonReviewExportError",
    "PublishedComparisonReview",
    "SUPPLEMENT_FILES",
    "comparison_review_folder_name",
    "verify_published_folder",
    "expected_allowlist",
    "next_comparison_export_version",
    "publish_comparison_review_folder",
]

PARENT = "funded_comparison"

#: Result table -> CSV file (one logical table, one format). ``configurations``
#: is written one row per configuration with its plain settings as columns.
CSV_TABLES: dict[str, str] = {
    "pair_results": "configuration_results.csv",
    "configurations": "configurations.csv",
    "account_journeys": "account_journeys.csv",
    "cash_ledger": "cash_ledger.csv",
    "payout_events": "payout_events.csv",
    "monthly_results": "monthly_results.csv",
    "trades": "trades.csv",
    "account_events": "account_events.csv",
    "rule_boundary_evidence": "rule_boundary_evidence.csv",
    "execution_evidence": "execution_evidence.csv",
    "strategy_metrics": "strategy_metrics.csv",
}

REVIEW_FINDINGS = "REVIEW_FINDINGS.md"
CORRECTIONS = "reporting_corrections.csv"
#: caller-supplied compact evidence: supplement key -> file name
SUPPLEMENT_FILES: dict[str, str] = {
    "configuration_bindings": "configuration_bindings.json",
    "trading_calendar": "trading_calendar.csv",
    "approximated_minutes": "approximated_minutes.csv",
    "reference_record_reconciliation": "reference_record_reconciliation.csv",
}
#: supplements that go into documents and validation_summary.json, not files
SUPPLEMENT_FACTS = ("trade_boundary_check",)
CHARTS = ("charts/net_cash_by_configuration.png", "charts/cash_over_time.png")

ALLOWLIST: tuple[str, ...] = (
    "README.md",
    "QUESTION.md",
    "RESULTS.md",
    "DECISIONS.md",
    "TRADING_RULES.md",
    "RESEARCH_LEDGER.md",
    "DATA_DICTIONARY.md",
    "settings.json",
    "firm_rules.json",
    "run_manifest.json",
    "validation_summary.json",
    "ledger.jsonl",
    *CSV_TABLES.values(),
    *CHARTS,
)

_ALLOWED_SUFFIXES = {".md", ".json", ".jsonl", ".csv", ".png"}
_MANIFEST = "run_manifest.json"
_VALIDATION = "validation_summary.json"
_BASE_COLUMNS = ("pair_id", "configuration", "firm_key", "firm")

#: Firm-profile fields that only drive the separate budgeted (five-account,
#: credit and growth) mode. They are not used here and are not exported.
_BUDGETED_ONLY_FIELDS = ("initial_accounts", "monthly_credits", "capacity_step",
                         "growth_share_bps", "max_capacity")

_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\)\S+|/(?:Users|home|tmp|mnt|var)/\S*"
                   r"|\b[\w.-]+/[\w./-]*\.(?:py|parquet|zip|jsonl?|csv|dbn|zst)\b")
_HASH = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{16,}\b"
                   r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")


class ComparisonReviewExportError(RuntimeError):
    """The staged review folder failed verification; nothing was published."""

    def __init__(self, message: str, checks: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.checks = checks or {}


@dataclass(frozen=True)
class PublishedComparisonReview:
    path: Path
    manifest: dict[str, Any]
    checks_passed: bool
    checks: dict[str, Any]
    receipt: dict[str, Any] | None = None


def comparison_review_folder_name(result_id: str, export_version: int) -> str:
    return f"funded_comparison_{result_id[:16]}_export_v{export_version}"


def expected_allowlist(with_review_findings: bool, extra: Iterable[str] = ()
                       ) -> tuple[str, ...]:
    base = (*ALLOWLIST, REVIEW_FINDINGS) if with_review_findings else ALLOWLIST
    return (*base, *sorted(extra))


def _extra_files(result: dict[str, Any], supplements: dict[str, Any]) -> tuple[str, ...]:
    extra = [SUPPLEMENT_FILES[k] for k in supplements if k in SUPPLEMENT_FILES]
    if result.get("reporting_corrections"):
        extra.append(CORRECTIONS)
    return tuple(sorted(extra))


def next_comparison_export_version(reports_root: Path, result_id: str) -> int:
    """The next unused export version for a (re)publication of the same result."""

    parent = Path(reports_root) / PARENT
    prefix = f"funded_comparison_{result_id[:16]}_export_v"
    used = [
        int(p.name[len(prefix):]) for p in parent.glob(f"{prefix}*")
        if p.is_dir() and p.name[len(prefix):].isdigit()
    ] if parent.is_dir() else []
    return max(used, default=0) + 1


# ── public entry point ────────────────────────────────────────────────────


def publish_comparison_review_folder(
    *,
    result: dict[str, Any],
    result_id: str,
    reports_root: Path,
    ledger_entries: list[dict[str, Any]],
    export_version: int,
    review_findings: str | None = None,
    supplements: dict[str, Any] | None = None,
) -> PublishedComparisonReview:
    """Stage, verify and atomically publish the comparison review folder."""

    supplements = dict(supplements or {})
    unknown = sorted(set(supplements) - set(SUPPLEMENT_FILES) - set(SUPPLEMENT_FACTS))
    if unknown:
        raise ValueError(f"unknown review supplements: {', '.join(unknown)}")
    if not re.fullmatch(r"[0-9a-f]{16,64}", result_id or ""):
        raise ValueError("result_id must be a lowercase hexadecimal identity")
    if int(export_version) < 1:
        raise ValueError("export_version must be at least 1")
    if not ledger_entries or not all(isinstance(e, dict) for e in ledger_entries):
        raise ValueError("ledger_entries must be the cumulative research ledger (a non-empty "
                         "list of entries supplied by the caller)")
    if review_findings is not None and not str(review_findings).strip():
        raise ValueError("review_findings, when given, must contain the recorded findings")

    parent = Path(reports_root) / PARENT
    final = parent / comparison_review_folder_name(result_id, export_version)
    if final.exists():
        raise FileExistsError(f"review folder {final.name} already exists; a repack must use "
                              "a higher export version")
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".staging_{final.name}_{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    allowlist = expected_allowlist(review_findings is not None,
                                   _extra_files(result, supplements))
    try:
        view = present_comparison(result)
        _write_all(staging, result, view, ledger_entries, export_version, review_findings,
                   supplements, allowlist)
        checks = _verify(staging, result, allowlist, supplements)
        passed = all(item["passed"] for item in checks.values())
        _write_json(staging / _VALIDATION, {
            "result_validation": _compact_validation(result.get("validation")),
            "export_checks": checks,
            "export_checks_passed": passed,
        })
        manifest = _manifest(staging, result, result_id, export_version,
                             review_findings is not None, checks)
        _write_json(staging / _MANIFEST, manifest)
        allow = _allowlist_check(staging, allowlist)
        checks["exact_allowlist_after_manifest"] = allow
        passed = passed and allow["passed"]
        if not passed:
            failed = sorted(name for name, item in checks.items() if not item["passed"])
            raise ComparisonReviewExportError(
                "review folder verification failed: " + ", ".join(failed), checks)
        if final.exists():
            raise FileExistsError(f"review folder {final.name} appeared while staging")
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    receipt = verify_published_folder(final)
    if not receipt["passed"]:
        raise ComparisonReviewExportError(
            f"published folder {final.name} does not match its manifest", receipt)
    return PublishedComparisonReview(path=final, manifest=manifest, checks_passed=True,
                                     checks=checks, receipt=receipt)


def verify_published_folder(folder: Path) -> dict[str, Any]:
    """Final payload receipt: re-read every file of a published folder against its manifest.

    Every payload listed in the manifest must exist with its size and SHA-256; no
    unlisted file may exist; the manifest itself is the only unlisted file.
    """

    folder = Path(folder)
    manifest_path = folder / _MANIFEST
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    listed = {f["path"]: f for f in manifest.get("files", [])}
    present = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                     if p.is_file())
    mismatched = []
    for rel, entry in listed.items():
        path = folder / rel
        data = path.read_bytes() if path.is_file() else None
        if (data is None or len(data) != entry["bytes"]
                or hashlib.sha256(data).hexdigest() != entry["sha256"]):
            mismatched.append(rel)
    unlisted = sorted(set(present) - set(listed) - {_MANIFEST})
    return {
        "passed": not mismatched and not unlisted and manifest_path.is_file(),
        "folder": folder.name,
        "payload_files_listed": len(listed),
        "files_present_including_manifest": len(present),
        "payloads_verified": len(listed) - len(mismatched),
        "mismatched": mismatched,
        "unlisted": unlisted,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "export_version": manifest.get("export_version"),
        "verified_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ── writing ───────────────────────────────────────────────────────────────


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns or list(_BASE_COLUMNS)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = _columns(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(c, row.get(c)) for c in columns])


def _not_completed(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Configurations that did not complete (one entry each, saved order)."""

    out: dict[str, dict[str, Any]] = {}
    for item in (result.get("validation") or {}).get("configurations_not_completed") or []:
        out.setdefault(str(item.get("configuration")), {
            "configuration": item.get("configuration"),
            "configuration_label": item.get("display_name", item.get("configuration")),
            "reason": item.get("reason")})
    for s in (result.get("summaries_cents") or {}).values():
        if s.get("status") != COMPLETED:
            out.setdefault(str(s.get("configuration")), {
                "configuration": s.get("configuration"),
                "configuration_label": s.get("configuration_label"),
                "reason": s.get("reason")})
    return list(out.values())


def _configuration_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in (result.get("tables") or {}).get("configurations") or []:
        row: dict[str, Any] = {"configuration": c.get("configuration"),
                               "configuration_label": c.get("configuration_label"),
                               "status": COMPLETED, "reason": None,
                               "axes": c.get("axes")}
        for item in c.get("settings") or []:
            row[str(item.get("setting"))] = item.get("value")
        for name, value in execution_sources(result, str(c.get("configuration"))):
            row[name] = value
        rows.append(row)
    for item in _not_completed(result):
        rows.append({"configuration": item["configuration"],
                     "configuration_label": item["configuration_label"],
                     "status": NOT_COMPLETED, "reason": plain_reason(item["reason"]),
                     "axes": None})
    return rows


def _table_rows(result: dict[str, Any], table: str) -> list[dict[str, Any]]:
    if table == "configurations":
        return _configuration_rows(result)
    return list((result.get("tables") or {}).get(table) or [])


def _firm_profiles(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [{k: v for k, v in p.items() if k not in _BUDGETED_ONLY_FIELDS}
            for p in (result.get("settings") or {}).get("firm_profiles") or []]


def _correction_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in result.get("reporting_corrections") or []:
        for change in record.get("changes") or []:
            before, after = change["before"], change["after"]
            configuration, _, firm_key = str(change["pair_id"]).partition("|")
            rows.append({
                "correction": record["correction_id"], "pair_id": change["pair_id"],
                "configuration": configuration, "firm_key": firm_key,
                "stop_exits_worse_than_stop_saved": before.get(
                    "stop_exits_filled_worse_than_stop"),
                "stop_exits_worse_than_stop_corrected": after.get(
                    "stop_exits_filled_worse_than_stop"),
                "stop_difference_saved_usd": before.get("stop_slippage_cents", 0) / 100,
                "stop_difference_corrected_usd": after.get("stop_slippage_cents", 0) / 100,
            })
        if not record.get("changes"):
            rows.append({"correction": record["correction_id"], "pair_id": None,
                         "configuration": None, "firm_key": None,
                         "note": record.get("description")})
    return rows


def _write_all(staging: Path, result: dict[str, Any], view: ComparisonView,
               ledger_entries: list[dict[str, Any]], export_version: int,
               review_findings: str | None, supplements: dict[str, Any],
               allowlist: tuple[str, ...]) -> None:
    for table, filename in CSV_TABLES.items():
        _write_csv(staging / filename, _table_rows(result, table))
    if result.get("reporting_corrections"):
        _write_csv(staging / CORRECTIONS, _correction_rows(result))
    for key, filename in SUPPLEMENT_FILES.items():
        if key not in supplements:
            continue
        if filename.endswith(".json"):
            _write_json(staging / filename, supplements[key])
        else:
            _write_csv(staging / filename, list(supplements[key]))

    settings = result.get("settings") or {}
    evidence = dict(result.get("price_evidence") or {})
    print_files = evidence.pop("print_files", None) or []
    _write_json(staging / "settings.json", {
        "mode": result.get("mode"),
        "purpose": result.get("purpose"),
        "question": result.get("question"),
        "funded_comparison_plan_id": result.get("funded_comparison_plan_id"),
        "period": result.get("period"),
        "source": result.get("source"),
        "configurations_requested": result.get("configurations_requested"),
        "configurations_completed": result.get("configurations_completed"),
        "price_evidence": {**evidence, "recorded_trade_days_used": len(print_files)},
        **{k: v for k, v in settings.items() if k != "firm_profiles"},
    })
    _write_json(staging / "firm_rules.json", {
        "firm_profiles": _firm_profiles(result),
        "note": "Budgeted-mode fields (starting group, monthly credits, growth and capacity) "
                "are not used in this single-account comparison and are omitted.",
    })
    _write_text(staging / "ledger.jsonl", "\n".join(
        json.dumps(entry, sort_keys=True, ensure_ascii=False) for entry in ledger_entries))

    _write_text(staging / "README.md", _readme(view, export_version, review_findings,
                                               result, supplements, allowlist))
    _write_text(staging / "QUESTION.md", _question(result, view))
    _write_text(staging / "RESULTS.md", _results(result, view))
    _write_text(staging / "DECISIONS.md", _decisions(result, view))
    _write_text(staging / "TRADING_RULES.md", _trading_rules(result, view, supplements))
    names = {str(row["configuration"]): plain_configuration_label(row["configuration_label"])
             for row in _configuration_rows(result) if row.get("configuration")}
    _write_text(staging / "RESEARCH_LEDGER.md", _ledger_markdown(ledger_entries, names))
    if review_findings is not None:
        _write_text(staging / REVIEW_FINDINGS, _review_findings(review_findings))
    _write_text(staging / "DATA_DICTIONARY.md", _data_dictionary(staging, allowlist))
    (staging / "charts").mkdir(exist_ok=True)
    _charts(staging / "charts", result, view)


# ── verification ──────────────────────────────────────────────────────────


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _usd_cents(text: str | None) -> int | None:
    if text is None or text == "":
        return None
    return int((Decimal(text) * 100).quantize(Decimal(1)))


def _allowlist_check(staging: Path, allowlist: Iterable[str]) -> dict[str, Any]:
    files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    dirs = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_dir())
    expected = sorted(allowlist)
    extra = sorted(set(files) - set(expected))
    missing = sorted(set(expected) - set(files))
    bad_suffix = sorted(f for f in files if Path(f).suffix.lower() not in _ALLOWED_SUFFIXES)
    bad_dirs = sorted(d for d in dirs if d != "charts")
    return {"passed": not (extra or missing or bad_suffix or bad_dirs), "files": len(files),
            "unexpected": extra, "missing": missing, "forbidden_extensions": bad_suffix,
            "unexpected_folders": bad_dirs}


def _exported_headlines(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == COMPLETED:
            out[row["pair_id"]] = {
                "status": COMPLETED,
                "net_cash_earned_cents": _usd_cents(row.get("net_cash_earned_usd")),
                "payouts_received_cents": _usd_cents(row.get("payouts_received_usd")),
                "account_costs_cents": _usd_cents(row.get("account_costs_usd")),
                "largest_single_payout_cents": _usd_cents(row.get("largest_single_payout_usd")),
                "accounts_purchased": int(row["accounts_purchased"]),
                "accounts_lost_before_first_payout":
                    int(row["accounts_lost_before_first_payout"]),
                "accounts_lost_after_a_payout": int(row["accounts_lost_after_a_payout"]),
            }
        else:
            money = [row.get(k) for k in ("net_cash_earned_usd", "payouts_received_usd",
                                          "account_costs_usd") if k in row]
            out[row["pair_id"]] = ({"status": NOT_COMPLETED} if not any(money)
                                   else {"status": "not completed but carries figures"})
    return out


def _verify(staging: Path, result: dict[str, Any], allowlist: tuple[str, ...],
            supplements: dict[str, Any] | None = None) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    allow = _allowlist_check(staging, allowlist)
    pending = {_MANIFEST, _VALIDATION}
    missing = [m for m in allow["missing"] if m not in pending]
    checks["allowlist_before_manifest"] = {
        **allow, "missing": missing,
        "scope": (f"the {allow['files']} files written before this summary and the manifest; "
                  f"the final folder adds {_VALIDATION} and {_MANIFEST} "
                  f"({len(allowlist)} files; see exact_allowlist_after_manifest and the "
                  "final payload receipt)"),
        "passed": not (missing or allow["unexpected"] or allow["forbidden_extensions"]
                       or allow["unexpected_folders"])}
    records = result.get("reporting_corrections") or []
    allowed = {"stop_exits_filled_worse_than_stop", "stop_slippage_cents"}
    touched = sorted({k for r in records for c in r.get("changes") or []
                      for k in (*c["before"], *c["after"])})
    checks["reporting_corrections_are_summary_only"] = {
        "passed": set(touched) <= allowed,
        "corrections": [r.get("correction_id") for r in records],
        "fields_changed": touched,
        "note": "Headline figures are compared with the screen below; corrections never "
                "change money, balances, costs, payouts or ranks."}
    boundary = (supplements or {}).get("trade_boundary_check")
    if boundary is not None:
        checks["trades_inside_their_trading_day"] = boundary

    counts: dict[str, Any] = {}
    parsed: dict[str, list[dict[str, str]]] = {}
    parse_errors: list[str] = []
    for table, filename in CSV_TABLES.items():
        try:
            parsed[filename] = _read_csv(staging / filename)
        except (OSError, csv.Error, UnicodeDecodeError) as error:
            parse_errors.append(f"{filename}: {type(error).__name__}")
            parsed[filename] = []
        counts[filename] = {"exported": len(parsed[filename]),
                            "result": len(_table_rows(result, table))}
    checks["csv_files_parse"] = {"passed": not parse_errors, "errors": parse_errors}
    checks["csv_row_counts_equal_result_tables"] = {
        "passed": all(c["exported"] == c["result"] for c in counts.values()), "tables": counts}

    summaries = result.get("summaries_cents") or {}
    cash_rows = parsed["cash_ledger.csv"]
    reconcile: dict[str, Any] = {}
    for pair_key, s in summaries.items():
        if s.get("status") != COMPLETED:
            continue
        rows = [r for r in cash_rows if r.get("pair_id") == pair_key]
        receipts = sum(_usd_cents(r["amount_usd"]) or 0 for r in rows
                       if r.get("kind") == "payout_received")
        costs = sum(_usd_cents(r["amount_usd"]) or 0 for r in rows
                    if r.get("kind") == "account_purchase")
        last = _usd_cents(rows[-1]["net_cash_after_usd"]) if rows else 0
        reconcile[pair_key] = {
            "passed": receipts == s["payouts_received_cents"]
            and costs == s["account_costs_cents"]
            and receipts - costs == s["net_cash_earned_cents"] == last,
            "receipts_cents": receipts, "costs_cents": costs}
    checks["cash_ledger_reconciles_to_result_cents"] = {
        "passed": bool(reconcile) and all(r["passed"] for r in reconcile.values()),
        "pairs_checked": len(reconcile),
        "failed": sorted(k for k, r in reconcile.items() if not r["passed"])}

    screen = comparison_headline_figures(result)
    exported = _exported_headlines(parsed["configuration_results.csv"])
    mismatched = sorted(k for k in set(screen) | set(exported)
                        if screen.get(k) != exported.get(k))
    checks["screen_export_agreement"] = {
        "passed": bool(screen) and not mismatched,
        "pairs_compared": len(screen), "figures": list(COMPARISON_HEADLINE_KEYS),
        "mismatched": mismatched}

    validation = result.get("validation")
    checks["result_validation_passed"] = {"passed": bool(validation and validation.get("passed"))}

    problems: dict[str, list[str]] = {}
    for name in allowlist:
        path = staging / name
        if name.endswith(".md") and path.exists():
            text = path.read_text(encoding="utf-8")
            found = []
            if _PATH.search(text):
                found.append("file path")
            if _HASH.search(text):
                found.append("hash or internal identity")
            if found:
                problems[name] = found
    checks["documents_contain_no_file_paths_or_hashes"] = {"passed": not problems,
                                                           "documents": problems}
    forbidden = [f for f in (p.relative_to(staging).as_posix() for p in staging.rglob("*")
                             if p.is_file())
                 if re.search(r"credit|growth", f, re.IGNORECASE)]
    checks["no_credit_or_growth_tables"] = {"passed": not forbidden, "files": forbidden}
    return checks


def _compact_validation(validation: dict[str, Any] | None) -> dict[str, Any]:
    if not validation:
        return {"passed": False, "checks": {}, "note": "the result carried no validation"}
    return {
        "passed": bool(validation.get("passed")),
        "checks": validation.get("checks", {}),
        "configurations_not_completed": [
            {"configuration": item.get("configuration"), "reason": plain_reason(
                item.get("reason"))}
            for item in validation.get("configurations_not_completed") or []],
    }


def _manifest(staging: Path, result: dict[str, Any], result_id: str, export_version: int,
              findings: bool, checks: dict[str, Any] | None = None) -> dict[str, Any]:
    files = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or path.name == _MANIFEST:
            continue
        data = path.read_bytes()
        files.append({"path": path.relative_to(staging).as_posix(), "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    return {
        "manifest_kind": "funded_comparison_review_folder",
        "funded_comparison_result_id": result_id,
        "funded_comparison_plan_id": result.get("funded_comparison_plan_id"),
        "result_schema_version": result.get("schema_version"),
        "mode": result.get("mode"),
        "purpose": result.get("purpose"),
        "export_version": export_version,
        "independent_review_findings_included": findings,
        "reporting_corrections": [r.get("correction_id")
                                  for r in result.get("reporting_corrections") or []],
        "published_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "payload_files": len(files),
        "files": files,
        "note": ("The manifest lists every other file in this folder (payload_files) and "
                 "excludes itself. After publication the application re-reads the folder "
                 "and checks every listed size and SHA-256 and that no other file exists."),
    }


# ── documents ─────────────────────────────────────────────────────────────


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _cell_md(text: Any) -> str:
    return str(text).replace("|", "/").replace("\n", " ")


def _comparison_tables(view: ComparisonView) -> list[str]:
    lines: list[str] = []
    for table in view.firm_tables:
        heads = list(table.rows[0].display()) if table.rows else ["Rank", "Configuration"]
        lines += [f"### {table.firm}", "", table.note, "",
                  "| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
        for row in table.rows:
            d = row.display()
            lines.append("| " + " | ".join(_cell_md(v) for v in d.values()) + " |")
        for row in table.rows:
            if not row.completed:
                lines.append("")
                lines.append(f"- {_cell_md(row.label)}: {row.reason}")
        lines.append("")
    return lines


def _evidence_lines(result: dict[str, Any], supplements: dict[str, Any]) -> list[str]:
    """Plain reconciliation of the evidence counts reviewers asked about."""

    tables = result.get("tables") or {}
    evidence = result.get("price_evidence") or {}
    trades = tables.get("trades") or []
    checked = int(evidence.get("position_minutes_checked") or 0)
    approx = int(evidence.get("position_minutes_approximated") or 0)
    summed = sum(int(t.get("minutes_on_prints") or 0) + int(t.get("minutes_approximated") or 0)
                 for t in trades)
    approx_rows = [t for t in trades if int(t.get("minutes_approximated") or 0)]
    lines = [
        f"- **Checked minutes versus position minutes.** {checked:,} one-minute source "
        "intervals were checked: each configuration checks each minute once, and both firms "
        "share that check. Adding each firm's own open-position minutes gives "
        f"{summed:,}; the two firms' positions differ after payouts and lost accounts, so "
        "this is not simply twice the checked count.",
        f"- **Substituted minutes versus affected trade rows.** {approx:,} checked intervals "
        "used the labeled one-minute approximation. The same interval serves both firms and "
        f"several configurations, so {len(approx_rows):,} funded trade rows in "
        f"{len({t['configuration'] for t in approx_rows}):,} configurations "
        f"({len({t['pair_id'] for t in approx_rows}):,} configuration-and-firm results) "
        "contain one.",
    ]
    rows = supplements.get("approximated_minutes") or []
    if rows:
        clock = {r.get("minute_open_utc") for r in rows}
        changes = [r for r in rows if str(r.get("ordering_can_change_final_exit_or_result"))
                   in ("True", "true")]
        timing = [r for r in rows if str(r.get(
            "ordering_can_change_only_fill_time_within_minute")) in ("True", "true")]
        lines.append(
            f"  They come from {len(clock)} distinct clock minutes; `approximated_minutes.csv` "
            "gives every one with its cause and its possible effect. "
            f"{len(changes)} of {len(rows)} could change an exit, the half exit, account "
            f"survival or a payout under any price order inside the minute; in {len(timing)} "
            "only the target lies inside the minute, so only its fill time within that "
            "minute can differ.")
    recon = supplements.get("reference_record_reconciliation") or []
    if recon:
        total = sum(int(r["reference_records"]) for r in recon)
        prep = sum(int(r["preparation_period_records"]) for r in recon)
        evaluated = sum(int(r["evaluated_strategy_trades"]) for r in recon)
        lines.append(
            f"- **Reference records versus evaluated strategy trades.** The no-account "
            f"replays hold {total:,} trade records: {prep:,} resolved during the preparation "
            f"(warmup) days and {evaluated:,} in the evaluation period, which the strategy "
            "measures use (`reference_record_reconciliation.csv`).")
    bindings = supplements.get("configuration_bindings") or {}
    configs = bindings.get("configurations") or []
    if configs:
        saved = sum(1 for c in configs if c.get("in_verified_study"))
        lines.append(
            f"- **Equivalence populations.** {saved} configurations are saved historical "
            "controls: their no-account replay equals the verified study's saved trades. "
            f"The other {len(configs) - saved} were generated by the research Strategy-Core "
            "and are compared with that Core's own normal day-by-day replay: an integration "
            "check that the account replay does not disturb the strategy, not independent "
            "evidence that the new exit rule is correct.")
    proxies = proxy_micro_positions(result)
    if proxies:
        lines.append(
            f"- **Micro execution.** All {proxies:,} ten-micro funded positions are proxy "
            "micro executions (E-mini Nasdaq-100 recorded trades used for signals, marks and "
            "fills), not only the positions with a substituted minute. Accepted and disclosed; "
            "not fixed.")
    return lines


def _readme(view: ComparisonView, export_version: int, review_findings: str | None,
            result: dict[str, Any] | None = None, supplements: dict[str, Any] | None = None,
            allowlist: tuple[str, ...] | None = None) -> str:
    result = result or {}
    supplements = supplements or {}
    corrections = result.get("reporting_corrections") or []
    lines = [
        "# Funded account configuration comparison — review folder",
        "",
        f"**Question:** {view.question}",
        "",
        f"- Period: {view.period_text}",
        f"- Firms: {view.firms_text}",
        f"- Position size: {view.size_text}",
        f"- Configurations: {view.configurations_text}",
        f"- Export version: {export_version}. A repack gets a new export version; the "
        "economic result never changes.",
        "",
        "## Status",
        "",
        _bullets(line.text for line in view.status),
        "",
        f"**{view.future_note}**",
        "",
        *(["## Reporting corrections in this export version", "",
            "The saved economic result is unchanged. These summary-only corrections were "
            "applied when it was loaded:", "",
            _bullets(str(r.get("description")) for r in corrections), "",
            "`reporting_corrections.csv` lists every changed configuration and firm with the "
            "saved and corrected values.", ""] if corrections else []),
        *(["## Evidence counts explained", "", *_evidence_lines(result, supplements), ""]
          if result else []),
        "## Independent review",
        "",
        ("The independent review findings recorded for this result are in "
         "`REVIEW_FINDINGS.md`.") if review_findings is not None else
        ("The independent review had not been recorded when this folder was published. A "
         "later export version will include its findings."),
        "",
        "## Comparison (each firm separately — never added together)",
        "",
        *_comparison_tables(view),
        "## Material limitations",
        "",
        _bullets(view.limitations),
        "",
        "## Files in this folder",
        "",
        "| File | What it holds |",
        "|---|---|",
    ]
    names = allowlist or expected_allowlist(review_findings is not None)
    lines.extend(f"| `{name}` | {_FILE_PURPOSE[name]} |" for name in names)
    lines += ["", "Every figure here comes from the same saved result the application screen "
              "shows. The application verified this folder before publishing it (see "
              "`validation_summary.json`); `run_manifest.json` lists each file's size and "
              "checksum."]
    return "\n".join(lines)


_FILE_PURPOSE: dict[str, str] = {
    "README.md": "This overview.",
    "QUESTION.md": "The frozen question, comparison, period, assumptions and falsification.",
    "RESULTS.md": "Plain-English results for every configuration and firm, including "
                  "unfavorable, zero-payout and not-completed ones.",
    "DECISIONS.md": "Owner decisions and assumptions used by this comparison.",
    "TRADING_RULES.md": "The simulated single-account, replacement, payout and execution rules.",
    "RESEARCH_LEDGER.md": "Readable cumulative research history.",
    "DATA_DICTIONARY.md": "Every data file and column explained.",
    "REVIEW_FINDINGS.md": "The recorded independent review findings.",
    "settings.json": "The exact comparison settings (period, size, costs, processing clock, "
                     "execution model, price-evidence summary).",
    "firm_rules.json": "The firm terms exactly as simulated.",
    "run_manifest.json": "Checksums and sizes of every other file.",
    "validation_summary.json": "Compact internal result checks and export checks.",
    "ledger.jsonl": "The cumulative append-only research ledger (machine-readable).",
    "configuration_results.csv": "One row per configuration and firm: headline and "
                                 "secondary figures (not-completed rows have no figures).",
    "configurations.csv": "One row per configuration with its plain strategy settings.",
    "account_journeys.csv": "One row per funded account: start, payouts, loss and replacement.",
    "cash_ledger.csv": "Every account purchase and received payout with running totals.",
    "payout_events.csv": "Eligibility, requests, processing, receipts and cutoff states.",
    "monthly_results.csv": "Every month (including zero and partial months) per "
                           "configuration and firm.",
    "trades.csv": "Every funded-account trade, including account-ending trades.",
    "account_events.csv": "Account status changes with reasons and before/after state.",
    "rule_boundary_evidence.csv": "Loss-limit movements, failures and refused entries.",
    "execution_evidence.csv": "Per configuration: no-account replay agreement, resumed-run "
                              "equality and price-evidence coverage.",
    "strategy_metrics.csv": "Per configuration: strategy measures of the replay with no "
                            "account limits (R after costs, win rate, drawdown, time under "
                            "water). Empty when the result predates this table.",
    "charts/net_cash_by_configuration.png": "Net cash earned per configuration, one panel "
                                            "per firm.",
    "charts/cash_over_time.png": "Cumulative payouts, account costs and net cash for each "
                                 "firm's top configuration.",
    "reporting_corrections.csv": "Summary-only reporting corrections applied to the saved "
                                 "result: saved and corrected values per configuration and "
                                 "firm.",
    "configuration_bindings.json": "Per configuration: every configurator input, the full "
                                   "effective strategy settings, size, cost, exit rule, "
                                   "equivalence population, and the exact strategy-study and "
                                   "Strategy-Core source binding.",
    "trading_calendar.csv": "The saved trading-day schedule used: each day's opening, "
                            "mandatory flat deadline, scheduled close and next reopen "
                            "(Chicago).",
    "approximated_minutes.csv": "Every funded trade row containing a substituted minute: "
                                "exact minute, cause, position state, the levels inside the "
                                "minute and whether any price order could change the result.",
    "reference_record_reconciliation.csv": "Per configuration: no-account replay records, "
                                           "preparation-period records and evaluated "
                                           "strategy trades.",
}


def _question(result: dict[str, Any], view: ComparisonView) -> str:
    return "\n".join([
        "# Question",
        "",
        f"> {view.question}",
        "",
        "## Objective",
        "",
        "Rank configurations, separately for each firm, by **net cash earned**: after-split "
        "payouts actually received minus the cost of every funded account purchased (the "
        "first account and every replacement). Large payouts count even when the account is "
        "later lost. Pending or merely secured payouts are not received cash. Trade count, "
        "account survival, win rate and short inactivity are not substitutes for cash earned.",
        "",
        "## Exact comparison",
        "",
        "Each configuration runs with each firm as its own comparison: one live funded account "
        "at a time; a lost account is replaced at once by a fresh funded account under the "
        "same configuration and charged at the firm's price. There is no five-account group, "
        "no monthly credit, no growth purchase and no copying of trades between accounts. "
        "Nothing is added across configurations or firms.",
        "",
        f"- Period: {view.period_text}",
        f"- Firms: {view.firms_text}",
        f"- Position size: {view.size_text}",
        f"- Configurations: {view.configurations_text}",
        f"- Price evidence: {view.evidence_text}",
        "",
        "## Assumptions",
        "",
        _bullets(f"{d.subject}: {d.decision} ({d.status})" for d in view.decisions)
        or "- None recorded.",
        "",
        "## What would falsify a conclusion",
        "",
        "- A different ordering of prices inside an approximated minute that changes which "
        "accounts survive.",
        "- Received cash that does not reconcile with the cash ledger, or a payout received "
        "before its processing completed.",
        "- A result that changes with stop fills worse than the recorded prices imply.",
        "",
        "## Sensitivity information (not a rejection test)",
        "",
        "- How much of a leader's cash comes from its largest payouts. Large one-time "
        "payouts are part of the objective: removing them describes concentration, it does "
        "not disqualify a configuration.",
        "",
        view.future_note,
    ])


def _results(result: dict[str, Any], view: ComparisonView) -> str:
    lines = ["# Results", "", f"**Question:** {view.question}", "",
             f"**Period:** {view.period_text}", "", "## Status", "",
             _bullets(line.text for line in view.status), "", f"**{view.future_note}**", "",
             "## Comparison (each firm separately — never added together)", "",
             *_comparison_tables(view),
             "## Payout timing, spending and time use", "",
             "Hours are account time summed over each configuration's accounts. Payout "
             "protection and processing are deliberate pauses, not a lack of strategy "
             "signals.", ""]
    for table in view.firm_tables:
        lines += [f"### {table.firm}", "",
                  "| Configuration | First payout | Largest unrecovered spending | Pending at "
                  "cutoff | Trades | Stop exits worse than the stop | Hours trading | Hours "
                  "payout protection | Hours payout processing | Hours market closed or locked "
                  "| Hours ready with no entry (no signal or outside entry hours) "
                  "| Entries refused (protection / processing) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in table.rows:
            if not row.completed:
                lines.append(f"| {_cell_md(row.label)} | {NOT_COMPLETED} |" + " — |" * 10)
                continue
            d = present_pair_detail(result, row.configuration, row.firm_key)
            facts = {f.label: f.value for f in d.facts}
            hours = [t.hours_text for t in d.time_split[:5]]
            refused = " / ".join(f.value for f in d.refused)
            lines.append("| " + " | ".join(_cell_md(v) for v in (
                row.label, facts["First payout received"],
                facts["Largest unrecovered account spending"],
                facts["Money pending at the cutoff"], facts["Trades taken"],
                facts["Stop exits filled worse than the stop"],
                hours[0], hours[1], hours[2], hours[3], hours[4], refused)) + " |")
        lines.append("")
    metrics = present_strategy_metrics(result)
    if metrics:
        heads = list(metrics[0])
        lines += ["## Strategy measures without accounts", "", STRATEGY_METRICS_NOTE, "",
                  "| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
        lines += ["| " + " | ".join(_cell_md(row[h]) for h in heads) + " |" for row in metrics]
        lines.append("")
    lines += ["Month-by-month figures, every account, payout and trade are in the CSV files "
              "(see `DATA_DICTIONARY.md`).", "", "## Material limitations", "",
              _bullets(view.limitations)]
    return "\n".join(lines)


def _decisions(result: dict[str, Any], view: ComparisonView) -> str:
    lines = ["# Decisions", "",
             "Owner decisions and explicit assumptions saved with this comparison. Assumptions "
             "are labeled as such; they are not claims about published firm rules.", ""]
    if view.decisions:
        lines += ["| Decided | Subject | Decision | Status |", "|---|---|---|---|"]
        lines += [f"| {d.decided_on} | {_cell_md(d.subject)} | {_cell_md(d.decision)} | "
                  f"{d.status} |" for d in view.decisions]
    else:
        lines.append("No owner decisions were saved with this result.")
    settings = result.get("settings") or {}
    model = settings.get("execution_model") or {}
    clock = settings.get("processing_clock") or {}
    lines += ["", "## Execution model used", "", str(model.get("description") or
                                                        "Not recorded."), ""]
    if model.get("refused_entry_policy") == "discard_refused_setup":
        lines.append("- An entry refused during payout protection or processing discards that "
                     "setup; the strategy may form new setups afterwards.")
    if model.get("replacement_policy") == "immediate_fresh_account_same_configuration":
        lines.append("- A lost account is replaced at once by a fresh funded account under the "
                     "same configuration (no evaluation or purchase delay is modeled).")
    if clock.get("description"):
        lines += ["", "## Payout processing clock used", "", str(clock["description"])]
    return "\n".join(lines)


def _trading_rules(result: dict[str, Any], view: ComparisonView,
                   supplements: dict[str, Any] | None = None) -> str:
    settings = result.get("settings") or {}
    clock = settings.get("processing_clock") or {}
    lines = ["# Trading rules simulated", "",
             "These are the **owner-defined simulation terms**, not a claim of complete "
             "compliance with each live firm program.", "", "## Firm terms", ""]
    for p in _firm_profiles(result):
        lock = int(p.get("floor_lock_cents", 0))
        lines += [
            f"### {p.get('firm_name')}", "",
            f"- Account: {p.get('account_label')}, cost "
            f"{format_usd(int(p.get('acquisition_cost_cents', 0)))} for every funded account "
            "(the first and each replacement).",
            f"- Trader share of each withdrawal: {p.get('trader_share_pct')}%.",
            f"- Position limit: {int(p.get('max_mini_equivalent_tenths', 0)) // 10} minis or "
            "equivalent.",
            f"- Loss allowance: {format_usd(int(p.get('loss_allowance_cents', 0)))} below the "
            "highest level reached, " + (
                "moving intraday with realized plus open-position equity"
                if p.get("threshold_update") == "intraday_peak_equity"
                else "moving only at the scheduled session close from the realized closing "
                "balance") + f"; it stops rising at {format_usd(lock)} relative to the start.",
            "- Account fails " + _comparator_words(p.get("comparator_before_lock"))
            + " the limit before it locks, and " + _comparator_words(
                p.get("comparator_after_lock")) + " it after it locks.",
            f"- Retained profit cushion {format_usd(int(p.get('retained_cushion_cents', 0)))}; "
            "minimum gross withdrawal "
            f"{format_usd(int(p.get('minimum_gross_request_cents', 0)))}.",
            "", "Assumptions for this firm:", "",
            _bullets(p.get("assumptions") or ["None recorded."]), "",
        ]
    lines += [
        "## One live account per configuration and firm", "",
        "- Each account starts funded with $0 earned profit and the full loss allowance; "
        "nothing is inherited from a lost account.",
        "- At most one live funded account exists per configuration and firm. A lost account "
        "is replaced at once by a fresh account under the same configuration and charged at "
        "the firm's price, with no limit on replacements. There are no monthly credits, no "
        "five-account group and no growth purchases in this mode.",
        "- A replacement trades only from a later valid opportunity, never the event that "
        "ended the previous account.",
        "- An account whose payout is processing is alive: it is never replaced or "
        "supplemented, and it trades nothing until processing completes.",
        "",
        "## Withdrawals", "",
        "- Once realized profit after costs leaves at least the minimum gross amount above "
        "the retained cushion while flat, the account stops new entries for the day.",
        "- At the end of that trading day it requests the entire surplus above the cushion "
        "(gross). The trader receives the gross amount times the trader share.",
        "- Received payouts stay received if the account is later lost. Requests still "
        "processing at the cutoff are not received cash.",
        "",
        "## Processing clock", "",
        str(clock.get("description") or "Not recorded."),
        "",
        f"- Payment time: {_twelve_hour(clock.get('payment_time_chicago'))} Chicago.",
        "",
        "## Daily close", "",
        "Every position is closed by the mandatory daily flat deadline (3:55 PM Chicago, "
        "earlier on shortened days). A trading day starts at the 5:00 PM Chicago reopen, so "
        "a position can be open past midnight inside one trading day; no position is held "
        "through a deadline, a closed market or a weekend.",
        *_boundary_lines((supplements or {}).get("trade_boundary_check")),
        "",
        "## Funded execution versus the strategy-only view", "",
        "Two different execution policies are used, and their figures are not "
        "interchangeable:",
        "",
        "- **Strategy measures (R, no account)** follow Strategy-Core's candle rules: on a "
        "candle the stop is checked before the target, and after a half exit the break-even "
        "stop is checked from the next candle.",
        "- **Funded accounts (cash)** follow the recorded exchange trades in their recorded "
        "order: a half exit and a break-even stop can both happen inside one minute; several "
        "trades with the same timestamp keep the exchange file's order (timestamp, then "
        "sequence number, then file row). Each fill posts its own cost ($0.514 per micro, "
        "$5.14 per mini). After a half exit the remaining contracts alone are marked, and "
        "TakeProfitTrader's floor keeps following the peak of realized plus open equity. The "
        "loss limit is checked on every trade before the stop or target. A stop or break-even "
        "stop that gaps fills at the first trade through it. Payout eligibility is checked "
        "only when the whole position is flat; a realized half with an open remainder never "
        "secures a payout.",
        "",
        "## Price evidence and exactness limits", "",
        f"This comparison: {view.evidence_text}",
        "",
        "Loss-limit checks are exact only relative to the recorded trades used. Where a "
        "minute is approximated, the losing side is assumed first and the result is labeled "
        "an approximation. None of this is broker execution.",
    ]
    return "\n".join(lines)


def _boundary_lines(check: dict[str, Any] | None) -> list[str]:
    if not check:
        return []
    outside = int(check.get("trades_outside_their_trading_day_count") or 0)
    return [
        "",
        f"Checked for this result: {int(check['trades_checked']):,} funded trades; "
        f"{int(check['trades_open_past_midnight_inside_one_trading_day']):,} were open past "
        f"midnight inside one trading day; {outside:,} fell outside their trading day. The "
        "check uses the study's saved trading schedule (`trading_calendar.csv`); it is not an "
        "independent check against an exchange shortened-session calendar.",
    ]


def _redact(text: str) -> str:
    text = _PATH.sub("(internal file location omitted)", text)
    return _HASH.sub("(internal identity omitted)", text)


_LEDGER_SKIP = re.compile(r"(^|_)(id|ids|sha256|hash|hashes|path|paths|file|files|archive|"
                          r"receipt|verifier|identity|event)$|^(ledger_origin|imported_from|"
                          r"event_type|display_time_chicago|timestamp_utc|recorded_at_utc|"
                          r"title)$")


def _ledger_value(value: Any) -> str:
    if isinstance(value, dict):
        text = "; ".join(f"{str(k).replace('_', ' ')}: {_ledger_value(v)}"
                         for k, v in value.items())
    elif isinstance(value, list | tuple):
        text = "; ".join(_ledger_value(v) for v in value)
    elif value is None:
        text = "—"
    else:
        text = str(value)
    return text


def _ledger_markdown(entries: list[dict[str, Any]],
                     configuration_names: dict[str, str] | None = None) -> str:
    names = configuration_names or {}
    known = (re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(names, key=len,
                                                                         reverse=True))
                        + r")(?!\w)") if names else None)

    def plain(text: str) -> str:
        text = _redact(text)
        return known.sub(lambda m: names[m.group(1)], text) if known else text

    lines = ["# Research ledger", "",
             "Readable cumulative history of proposed, completed, failed and superseded "
             "experiments, oldest first. `ledger.jsonl` holds the same entries verbatim; this "
             "copy omits internal identities, checksums and file locations and shortens long "
             "values. Entries imported from an earlier verified study are marked.", ""]
    for index, entry in enumerate(entries, start=1):
        kind = str(entry.get("event_type") or entry.get("kind") or "entry").replace("_", " ")
        title = str(entry.get("title") or kind.capitalize())
        status = entry.get("status")
        when = entry.get("display_time_chicago") or entry.get("recorded_on") or ""
        heading = f"## {index}. {_redact(title)}"
        if status:
            heading += f" — {str(status).replace('_', ' ')}"
        lines += [heading, ""]
        meta = []
        if when:
            meta.append(f"When: {_redact(str(when))}")
        if entry.get("ledger_origin") == "imported_verified_history":
            meta.append("Imported history from an earlier verified study")
        if meta:
            lines += [f"- {m}" for m in meta]
        for key, value in entry.items():
            if _LEDGER_SKIP.search(str(key)) or key in ("status", "recorded_on"):
                continue
            text = plain(_ledger_value(value))
            if len(text) > 400:
                text = text[:400].rstrip() + " … (full entry in ledger.jsonl)"
            lines.append(f"- **{str(key).replace('_', ' ').capitalize()}:** {text}")
        lines.append("")
    return "\n".join(lines)


def _review_findings(text: str) -> str:
    body = _redact(str(text).strip())
    if body.lstrip().startswith("#"):
        return body
    return "# Independent review findings\n\n" + body


_COLUMN_WORDS: dict[str, str] = {
    "pair_id": "Internal key of the configuration-and-firm comparison.",
    "configuration": "Internal configuration name (see configurations.csv).",
    "configuration_label": "Configuration in plain English.",
    "firm_key": "Internal firm key.",
    "firm": "Full firm name.",
    "status": "Completed or Not completed (not-completed rows carry no figures).",
    "reason": "Why a configuration did not complete.",
    "rank_within_firm": "Rank by net cash earned within the firm (ties share a rank).",
    "axes": "The study axis values that define the configuration.",
    "account_number": "Account number within its configuration and firm (1 = first).",
    "replaces_account_number": "The lost account this account replaced.",
    "lost_after_payout": "True when the account was lost after receiving a payout.",
    "trade_ref": "Internal reference of the strategy setup the trade followed.",
    "strategy_trade_id": "Internal reference of the strategy setup the trade followed.",
    "minutes_approximated": "Minutes of the position priced with the one-minute "
                            "approximation.",
    "minutes_on_prints": "Minutes of the position priced from recorded exchange trades.",
    "strategy_recorded_exit_ticks": "The strategy's own recorded exit in ticks.",
    "no_account_replay_equals_saved_study": "True when the replay without accounts "
                                            "reproduced the saved study's trades.",
    "resumed_run_identical": "True when a run resumed from a checkpoint matched the full "
                             "run.",
    "trades_not_in_no_account_replay": "Trades taken only because account events changed "
                                       "the path.",
}


def _data_dictionary(staging: Path, allowlist: tuple[str, ...] = ()) -> str:
    lines = ["# Data dictionary", "",
             "Money columns ending in `_usd` are US dollars exact to the cent. Times appear "
             "twice: `_utc` for machines and `_chicago` for reading. Prices in ticks convert "
             "to index points by multiplying by 0.25. Hours are account time. One logical "
             "table is one file; every row carries its configuration and firm.", ""]
    extra_csv = [n for n in allowlist if n.endswith(".csv") and n not in CSV_TABLES.values()]
    for filename in (*CSV_TABLES.values(), *extra_csv):
        with (staging / filename).open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
            rows = sum(1 for _ in handle)
        lines += [f"## {filename}", "", _FILE_PURPOSE[filename], f"Rows: {rows}.", "",
                  "| Column | Meaning |", "|---|---|"]
        lines += [f"| `{_cell_md(c)}` | "
                  f"{_COLUMN_WORDS.get(c) or _describe_column(c)} |" for c in header]
        lines.append("")
    lines += ["## JSON files", "",
              "- `settings.json`: comparison settings (period, size, costs, processing clock, "
              "execution model, price-evidence summary).",
              "- `firm_rules.json`: the firm terms exactly as simulated (money in cents).",
              "- `validation_summary.json`: compact internal result checks and export checks.",
              "- `run_manifest.json`: every other file's size and checksum.",
              "- `ledger.jsonl`: one research-ledger entry per line, cumulative.",
              *(["- `configuration_bindings.json`: per configuration, every configurator "
                 "input, the full effective strategy settings and the exact source binding."]
                if "configuration_bindings.json" in allowlist else [])]
    return _redact("\n".join(lines))


# ── charts (matplotlib, Agg canvas, no global backend state) ─────────────

_RECEIVED, _COSTS, _NET = "#2e7d32", "#8d6e63", "#1f5fa8"


def _short(label: str, limit: int = 70) -> str:
    return label if len(label) <= limit else label[: limit - 1].rstrip() + "…"


CHART_ROWS = 20


def _chart_label(row, columns: dict[str, tuple[tuple[str, str], ...]]) -> str:
    """Every differing setting, wrapped on two lines (nothing cut off)."""

    import textwrap

    own = columns.get(row.configuration)
    short = {"Exit": {"Half at 1R, rest held": "half exit",
                      "Whole position": "whole exit"}}
    text = (" · ".join(short.get(k, {}).get(v, v) for k, v in own if k != "Size")
            if own else row.label)
    lines = textwrap.wrap(f"{row.rank_text}. {text}", width=64)
    if len(lines) > 2:
        lines = [lines[0], _short(" ".join(lines[1:]), 64)]
    return "\n".join(lines)


def _charts(folder: Path, result: dict[str, Any], view: ComparisonView) -> None:
    figure_cls, canvas_cls = _matplotlib()
    tables = view.firm_tables or ()
    panels = max(1, len(tables))
    columns = configuration_columns(result)
    shown = [[r for r in t.rows if r.completed][:CHART_ROWS] for t in tables]
    heights = [max(2.0, 0.62 * max(1, len(rows)) + 1.4) for rows in shown] or [2.0]
    fig = figure_cls(figsize=(13, sum(heights) + 1.2), dpi=110)
    canvas_cls(fig)
    axes = fig.subplots(panels, 1, squeeze=False,
                        gridspec_kw={"height_ratios": heights})[:, 0]
    for ax, table, rows in zip(axes, tables, shown, strict=False):
        labels = [_chart_label(r, columns) for r in rows]
        values = [dict(r.figures).get("net_cash_earned_cents", 0) / 100 if r.completed else 0.0
                  for r in rows]
        ypos = list(range(len(rows), 0, -1))
        colors = [_NET if v >= 0 else "#b71c1c" for v in values]
        for y, r, v, color in zip(ypos, rows, values, colors, strict=True):
            if r.completed:
                ax.barh(y, v, color=color, height=0.6)
                ax.annotate(r.net_cash, (v, y), textcoords="offset points",
                            xytext=(4 if v >= 0 else -4, 0), va="center",
                            ha="left" if v >= 0 else "right", fontsize=7)
        ax.axvline(0, color="#444444", linewidth=0.8)
        ax.set_ylim(0.4, len(rows) + 0.6)
        ax.set_yticks(ypos, labels, fontsize=8)
        more = len(table.rows) - len(rows)
        ax.set_title(f"{table.firm}: net cash earned after all account costs (US dollars)"
                     + (f" — top {len(rows)}; all {len(table.rows)} are in RESULTS.md"
                        if more else ""), fontsize=10)
        ax.xaxis.set_major_formatter(_dollar_formatter())
        ax.grid(axis="x", alpha=0.3)
        ax.margins(x=0.2)
    fig.suptitle("Net cash earned by configuration — each firm separately, never added "
                 "together", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / (sum(heights) + 1.2)))
    fig.savefig(folder / "net_cash_by_configuration.png", format="png",
                metadata={"Software": None})

    fig = figure_cls(figsize=(11, 3.4 * panels + 0.6), dpi=100)
    canvas_cls(fig)
    axes = fig.subplots(panels, 1, squeeze=False)[:, 0]
    for ax, table in zip(axes, tables, strict=False):
        top = next((r for r in table.rows if r.completed), None)
        if top is None:
            ax.text(0.5, 0.5, "No configuration completed", transform=ax.transAxes,
                    ha="center")
            ax.set_title(table.firm, fontsize=9)
            continue
        detail = present_pair_detail(result, top.configuration, top.firm_key)
        points = cash_chart_points(detail)
        xs = [p[0] for p in points]
        for index, (name, color, style) in enumerate((
                ("Payouts received after the split", _RECEIVED, "-"),
                ("Account costs", _COSTS, "--"),
                ("Net cash earned", _NET, "-"))):
            ax.step(xs, [p[index + 1] for p in points], where="post", color=color,
                    linestyle=style, linewidth=2 if index == 2 else 1.5, label=name)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_title(f"{table.firm} — top configuration: {_short(top.label, 90)}", fontsize=9)
        ax.yaxis.set_major_formatter(_dollar_formatter())
        ax.xaxis.set_major_formatter(_date_formatter())
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlabel("Date (Chicago time)")
    fig.suptitle("Cumulative cash over time (one configuration per firm, never added "
                 "together)", fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / "cash_over_time.png", format="png", metadata={"Software": None})
