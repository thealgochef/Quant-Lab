"""Automatic compact review folder for a completed funded-payout simulation.

``publish_funded_review_folder`` writes EXACTLY the specification section-10
allowlist (documents and necessary data only) from the ONE saved funded
result. It stages into a sibling temporary folder, verifies the staged files
(exact allowlist, row counts, money reconciliation with the result's exact
cents, and screen/export agreement through the same presenter the screen
uses), then publishes with one atomic rename. An existing folder is never
overwritten: a repack uses a higher ``export_version`` and never changes the
economic result.

No scripts, source code, notebooks, raw dumps or caches are ever written.
"""

from __future__ import annotations

import csv
import hashlib
import io
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

from alpha_lab.agents.data_infra.ifvg.presentation.funded_results import (
    HEADLINE_KEYS,
    FundedResultView,
    chicago_datetime,
    format_usd,
    headline_figures,
    present_funded_result,
)

__all__ = [
    "ALLOWLIST",
    "CSV_TABLES",
    "FundedReviewExportError",
    "PublishedReview",
    "publish_funded_review_folder",
    "review_folder_name",
    "next_export_version",
]

#: Result table -> CSV file (one logical table, one format).
CSV_TABLES: dict[str, str] = {
    "instance_results": "instance_results.csv",
    "account_journeys": "account_journeys.csv",
    "cash_ledger": "cash_ledger.csv",
    "payout_events": "payout_events.csv",
    "credit_events": "credit_events.csv",
    "growth_events": "growth_events.csv",
    "monthly_results": "monthly_results.csv",
    "trades": "trades.csv",
    "account_events": "account_events.csv",
    "rule_boundary_evidence": "rule_boundary_evidence.csv",
    "comparison": "comparison.csv",
}

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
    "rule_sources.csv",
    "run_manifest.json",
    "validation_summary.json",
    "ledger.jsonl",
    *CSV_TABLES.values(),
    "charts/cash_over_time.png",
    "charts/account_journeys.png",
)

_ALLOWED_SUFFIXES = {".md", ".json", ".jsonl", ".csv", ".png"}
_MANIFEST = "run_manifest.json"
_VALIDATION = "validation_summary.json"

#: Headers for tables that can legitimately be empty (e.g. no payouts at all),
#: so an empty CSV still documents its columns.
_EMPTY_HEADERS: dict[str, tuple[str, ...]] = {
    "payout_events": (
        "firm_key", "firm", "seq", "account_id", "event", "trading_day", "request_id",
        "clock_basis", "after_cutoff", "ts_utc", "ts_chicago", "secured_utc",
        "secured_chicago", "requested_utc", "requested_chicago", "due_utc", "due_chicago",
        "received_utc", "received_chicago", "gross_usd", "firm_share_usd", "trader_usd",
        "balance_before_usd", "balance_after_usd", "floor_usd", "realized_balance_usd",
        "eligible_gross_usd",
    ),
    "growth_events": (
        "firm_key", "firm", "seq", "receipt_day", "decision", "reason", "capacity_before",
        "capacity_after", "ts_utc", "ts_chicago", "wallet_before_usd", "wallet_after_usd",
        "net_cash_before_usd", "net_cash_after_usd", "block_cost_usd", "threshold_usd",
    ),
    "trades": (
        "firm_key", "firm", "seq", "account_id", "trade_id", "trading_day", "direction",
        "quantity", "entry_ticks", "stop_ticks", "exit_ticks", "exit_kind", "entry_utc",
        "entry_chicago", "exit_utc", "exit_chicago", "gross_pnl_usd", "costs_usd",
        "net_pnl_usd", "initial_risk_usd", "price_evidence", "account_failed",
    ),
    "cash_ledger": (
        "firm_key", "firm", "seq", "kind", "funding", "account_id", "detail", "ts_utc",
        "ts_chicago", "amount_usd", "wallet_before_usd", "wallet_after_usd",
        "receipts_before_usd", "receipts_after_usd", "costs_before_usd", "costs_after_usd",
        "net_cash_after_usd",
    ),
    "rule_boundary_evidence": (
        "firm_key", "firm", "seq", "account_id", "check", "outcome", "trade_id", "detail",
        "ts_utc", "ts_chicago",
    ),
}


class FundedReviewExportError(RuntimeError):
    """The staged review folder failed verification; nothing was published."""

    def __init__(self, message: str, checks: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.checks = checks or {}


@dataclass(frozen=True)
class PublishedReview:
    path: Path
    manifest: dict[str, Any]
    checks_passed: bool
    checks: dict[str, Any]


def review_folder_name(funded_result_id: str, export_version: int) -> str:
    return f"funded_payout_{funded_result_id[:16]}_export_v{export_version}"


def next_export_version(reports_root: Path, funded_result_id: str) -> int:
    """The next unused export version for a repack of the same result."""

    parent = Path(reports_root) / "funded_payout"
    prefix = f"funded_payout_{funded_result_id[:16]}_export_v"
    used = [
        int(p.name[len(prefix):]) for p in parent.glob(f"{prefix}*")
        if p.is_dir() and p.name[len(prefix):].isdigit()
    ] if parent.is_dir() else []
    return max(used, default=0) + 1


# ── public entry point ────────────────────────────────────────────────────


def publish_funded_review_folder(
    *,
    result: dict[str, Any],
    funded_result_id: str,
    reports_root: Path,
    ledger_entries: list[dict[str, Any]],
    export_version: int = 1,
) -> PublishedReview:
    """Stage, verify and atomically publish the review folder for ``result``."""

    if not re.fullmatch(r"[0-9a-f]{16,64}", funded_result_id or ""):
        raise ValueError("funded_result_id must be a lowercase hexadecimal identity")
    if export_version < 1:
        raise ValueError("export_version must be at least 1")
    if not ledger_entries or not all(isinstance(e, dict) for e in ledger_entries):
        raise ValueError(
            "ledger_entries must be the cumulative research ledger (a non-empty list of "
            "entries supplied by the caller)")

    parent = Path(reports_root) / "funded_payout"
    final = parent / review_folder_name(funded_result_id, export_version)
    if final.exists():
        raise FileExistsError(
            f"review folder {final.name} already exists; a repack must use a higher "
            "export version")
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".staging_{final.name}_{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        view = present_funded_result(result)
        _write_all(staging, result, view, funded_result_id, ledger_entries, export_version)
        checks = _verify(staging, result, view)
        checks_passed = all(item["passed"] for item in checks.values())
        _write_json(staging / _VALIDATION, {
            "result_validation": _compact_validation(result.get("validation")),
            "export_checks": checks,
            "export_checks_passed": checks_passed,
        })
        manifest = _manifest(staging, result, funded_result_id, export_version)
        _write_json(staging / _MANIFEST, manifest)
        allow = _allowlist_check(staging)
        checks["exact_allowlist_after_manifest"] = allow
        checks_passed = checks_passed and allow["passed"]
        if not checks_passed:
            failed = sorted(name for name, item in checks.items() if not item["passed"])
            raise FundedReviewExportError(
                "review folder verification failed: " + ", ".join(failed), checks)
        if final.exists():
            raise FileExistsError(f"review folder {final.name} appeared while staging")
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return PublishedReview(path=final, manifest=manifest, checks_passed=True, checks=checks)


# ── writing ───────────────────────────────────────────────────────────────


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip("\n") + "\n")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _cell(column: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if column.endswith("_usd") and isinstance(value, int | float):
        return f"{Decimal(str(value)):.2f}"
    if isinstance(value, list | tuple | dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _columns(table: str, rows: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    if not columns:
        columns = list(_EMPTY_HEADERS.get(table, ("firm_key", "firm")))
    return columns


def _write_csv(path: Path, table: str, rows: list[dict[str, Any]]) -> None:
    columns = _columns(table, rows)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_cell(c, row.get(c)) for c in columns])
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def _write_all(staging: Path, result: dict[str, Any], view: FundedResultView,
               funded_result_id: str, ledger_entries: list[dict[str, Any]],
               export_version: int) -> None:
    tables = result.get("tables") or {}
    for table, filename in CSV_TABLES.items():
        _write_csv(staging / filename, table, list(tables.get(table) or []))

    settings = result.get("settings") or {}
    profiles = list(settings.get("firm_profiles") or [])
    _write_json(staging / "settings.json", {
        "purpose": result.get("purpose"),
        "funded_plan_id": result.get("funded_plan_id"),
        "period": result.get("period"),
        "source": result.get("source"),
        "price_evidence": result.get("price_evidence"),
        **{k: v for k, v in settings.items() if k != "firm_profiles"},
    })
    _write_json(staging / "firm_rules.json", {"firm_profiles": profiles})
    source_rows = [
        {"firm": p.get("firm_name"), "source_id": s.get("source_id"), "title": s.get("title"),
         "url": s.get("url"), "checked_on": s.get("checked_on"), "supports": s.get("supports")}
        for p in profiles for s in p.get("sources") or []
    ]
    _write_csv(staging / "rule_sources.csv", "rule_sources", source_rows)
    _write_text(staging / "ledger.jsonl", "\n".join(
        json.dumps(entry, sort_keys=True, ensure_ascii=False) for entry in ledger_entries))

    _write_text(staging / "README.md", _readme(result, view, export_version))
    _write_text(staging / "QUESTION.md", _question(result, view))
    _write_text(staging / "RESULTS.md", _results(view))
    _write_text(staging / "DECISIONS.md", _decisions(result, view))
    _write_text(staging / "TRADING_RULES.md", _trading_rules(result, view))
    _write_text(staging / "RESEARCH_LEDGER.md", _ledger_markdown(ledger_entries))
    _write_text(staging / "DATA_DICTIONARY.md", _data_dictionary(staging))
    (staging / "charts").mkdir(exist_ok=True)
    _charts(staging / "charts", view)


# ── verification ──────────────────────────────────────────────────────────


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _usd_cents(text: str | None) -> int | None:
    if text is None or text == "":
        return None
    return int((Decimal(text) * 100).quantize(Decimal(1)))


def _allowlist_check(staging: Path) -> dict[str, Any]:
    files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
    dirs = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_dir())
    expected = sorted(ALLOWLIST)
    extra = sorted(set(files) - set(expected))
    missing = sorted(set(expected) - set(files))
    bad_suffix = sorted(f for f in files if Path(f).suffix.lower() not in _ALLOWED_SUFFIXES)
    bad_dirs = sorted(d for d in dirs if d != "charts")
    return {
        "passed": not (extra or missing or bad_suffix or bad_dirs),
        "files": len(files), "unexpected": extra, "missing": missing,
        "forbidden_extensions": bad_suffix, "unexpected_folders": bad_dirs,
    }


def _verify(staging: Path, result: dict[str, Any], view: FundedResultView) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    # Every file present except the two written after verification.
    allow = _allowlist_check(staging)
    pending = {_MANIFEST, _VALIDATION}
    allow_pre = dict(allow)
    allow_pre["missing"] = [m for m in allow["missing"] if m not in pending]
    allow_pre["passed"] = not (allow_pre["missing"] or allow["unexpected"]
                               or allow["forbidden_extensions"] or allow["unexpected_folders"])
    checks["allowlist_before_manifest"] = allow_pre

    tables = result.get("tables") or {}
    counts = {}
    for table, filename in CSV_TABLES.items():
        exported = len(_read_csv(staging / filename))
        expected = len(tables.get(table) or [])
        counts[filename] = {"exported": exported, "result": expected}
    checks["csv_row_counts_equal_result_tables"] = {
        "passed": all(c["exported"] == c["result"] for c in counts.values()),
        "tables": counts,
    }

    summaries = result["summaries_cents"]
    ledger = _read_csv(staging / "cash_ledger.csv")
    instances = {row["firm_key"]: row for row in _read_csv(staging / "instance_results.csv")}
    cash_checks: dict[str, Any] = {}
    instance_checks: dict[str, Any] = {}
    for key, s in summaries.items():
        rows = [r for r in ledger if r.get("firm_key") == key]
        receipts = sum(_usd_cents(r["amount_usd"]) or 0 for r in rows
                       if r.get("kind") == "payout_received")
        costs = sum(_usd_cents(r["amount_usd"]) or 0 for r in rows
                    if r.get("kind") == "account_purchase")
        last_net = _usd_cents(rows[-1]["net_cash_after_usd"]) if rows else 0
        cash_checks[key] = {
            "receipts_cents": receipts, "costs_cents": costs, "net_cents": receipts - costs,
            "passed": receipts == s["payouts_received_cents"]
            and costs == s["acquisition_costs_cents"]
            and receipts - costs == s["net_cash_earned_cents"]
            and last_net == s["net_cash_earned_cents"],
        }
        row = instances.get(key)
        if row is None:
            instance_checks[key] = {"passed": False, "reason": "firm missing"}
            continue
        exported = {
            "net_cash_earned_cents": _usd_cents(row.get("net_cash_earned_usd")),
            "payouts_received_cents": _usd_cents(row.get("payouts_received_usd")),
            "largest_single_payout_cents": _usd_cents(row.get("largest_single_payout_usd")),
            "acquisition_costs_cents": _usd_cents(row.get("acquisition_costs_usd")),
            "accounts_lost_before_first_payout": int(row["accounts_lost_before_first_payout"]),
            "accounts_lost_after_a_payout": int(row["accounts_lost_after_a_payout"]),
        }
        instance_checks[key] = {
            "passed": all(exported[k] == s[k] for k in HEADLINE_KEYS),
            "exported": exported,
        }
    checks["cash_ledger_reconciles_to_result_cents"] = {
        "passed": bool(cash_checks) and all(c["passed"] for c in cash_checks.values()),
        "firms": cash_checks,
    }
    checks["instance_results_equal_result_cents"] = {
        "passed": bool(instance_checks) and all(c["passed"] for c in instance_checks.values()),
        "firms": instance_checks,
    }

    screen = headline_figures(view)
    exported_headlines = {
        key: item.get("exported") for key, item in instance_checks.items()
    }
    checks["screen_export_agreement"] = {
        "passed": bool(screen) and screen == exported_headlines,
        "screen": screen,
    }

    validation = result.get("validation")
    checks["result_validation_passed"] = {
        "passed": bool(validation and validation.get("passed")),
    }

    text_problems = []
    path_pattern = re.compile(r"(?:[A-Za-z]:\\|\\\\)[^\s]+|/(?:Users|home|tmp)/")
    for name in ALLOWLIST:
        if name.endswith(".md") and (staging / name).exists():
            text = (staging / name).read_text(encoding="utf-8")
            if path_pattern.search(text):
                text_problems.append(name)
    checks["documents_contain_no_file_paths"] = {
        "passed": not text_problems, "documents": text_problems,
    }
    return checks


def _compact_validation(validation: dict[str, Any] | None) -> dict[str, Any]:
    if not validation:
        return {"passed": False, "checks": {}, "note": "the result carried no validation"}
    return {"passed": bool(validation.get("passed")), "checks": validation.get("checks", {})}


def _manifest(staging: Path, result: dict[str, Any], funded_result_id: str,
              export_version: int) -> dict[str, Any]:
    files = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(staging).as_posix()
        if rel == _MANIFEST:
            continue
        data = path.read_bytes()
        files.append({"path": rel, "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    return {
        "manifest_kind": "funded_payout_review_folder",
        "funded_result_id": funded_result_id,
        "funded_plan_id": result.get("funded_plan_id"),
        "result_schema_version": result.get("schema_version"),
        "purpose": result.get("purpose"),
        "export_version": export_version,
        "published_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
        "note": "The manifest lists every other file in this folder and excludes itself.",
    }


# ── documents ─────────────────────────────────────────────────────────────


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _status_block(view: FundedResultView) -> str:
    return _bullets(line.text for line in view.status)


def _readme(result: dict[str, Any], view: FundedResultView, export_version: int) -> str:
    lines = [
        "# Funded-account payout simulation — review folder",
        "",
        view.headline,
        "",
        f"**Market period:** {view.period_text}",
        "",
        f"**Strategy:** {view.strategy_text}",
        "",
        f"**Export version:** {export_version}. A repack gets a new export version; the "
        "economic result never changes.",
        "",
        "## Status",
        "",
        _status_block(view),
        "",
        "## Headline figures (each firm separately — never added together)",
        "",
    ]
    for firm in view.firms:
        lines.append(f"### {firm.firm}")
        lines.append("")
        lines.extend(f"- {card.label}: {card.value}" for card in firm.cards)
        lines.extend(f"- {notice}" for notice in firm.notices)
        lines.append("")
    lines += [
        "## Material limitations",
        "",
        _bullets(view.limitations),
        "",
        "## Files in this folder",
        "",
        "| File | What it holds |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | {_FILE_PURPOSE[name]} |" for name in ALLOWLIST)
    lines += [
        "",
        "Every figure here comes from the same saved result the application screen shows. "
        "The application verified this folder before publishing it (see "
        "`validation_summary.json`); `run_manifest.json` lists each file's size and "
        "SHA-256 checksum.",
    ]
    return "\n".join(lines)


_FILE_PURPOSE: dict[str, str] = {
    "README.md": "This overview.",
    "QUESTION.md": "The frozen question, comparison, period, assumptions and falsification.",
    "RESULTS.md": "Plain-English results for both firms, including bad outcomes.",
    "DECISIONS.md": "Owner decisions and assumptions used by this run.",
    "TRADING_RULES.md": "The simulated account, payout, credit and growth rules.",
    "RESEARCH_LEDGER.md": "Readable cumulative research history.",
    "DATA_DICTIONARY.md": "Every data file and column explained.",
    "settings.json": "The exact run settings (period, size, costs, processing clock).",
    "firm_rules.json": "The two firm profiles exactly as simulated.",
    "rule_sources.csv": "Published sources behind the firm rules, with check dates.",
    "run_manifest.json": "Checksums and sizes of every other file.",
    "validation_summary.json": "Compact internal checks and export checks.",
    "ledger.jsonl": "The cumulative append-only research ledger (machine-readable).",
    "instance_results.csv": "One row per firm: the headline and secondary figures.",
    "account_journeys.csv": "One row per account: start, payouts, loss and replacement.",
    "cash_ledger.csv": "Every modeled receipt and account purchase with running balances.",
    "payout_events.csv": "Eligibility, requests, processing, receipts and cutoff states.",
    "credit_events.csv": "Monthly credit grants and every credit used.",
    "growth_events.csv": "Every five-account growth decision and its basis.",
    "monthly_results.csv": "Every month (including zero and partial months) per firm.",
    "trades.csv": "Every funded-account trade, including account-ending trades.",
    "account_events.csv": "Account status changes with reasons and before/after state.",
    "rule_boundary_evidence.csv": "Loss-limit movements, failures and blocked entries.",
    "comparison.csv": "Side-by-side comparison rows (no cross-firm total).",
    "charts/cash_over_time.png": "Cumulative received cash, costs and net cash per firm.",
    "charts/account_journeys.png": "Timeline of every account per firm.",
}


def _question(result: dict[str, Any], view: FundedResultView) -> str:
    settings = result.get("settings") or {}
    return "\n".join([
        "# Question",
        "",
        f"> {view.question}",
        "",
        "## Frozen objective",
        "",
        "The winner criterion is **net cash earned after all modeled account costs**: "
        "after-split payouts actually received, minus every account purchase (initial, "
        "replacement and growth), for each firm separately. Large withdrawals count even "
        "when the account is later lost. Pending or merely secured payouts are not received "
        "cash. Account survival, payout count, smoothness and trade frequency are not "
        "substitutes for cash earned.",
        "",
        "The former activity / trade-frequency objective used by earlier strategy studies "
        "is **historical only**. It is preserved for reference and is **not** the winner "
        "criterion for this comparison.",
        "",
        "## Exact comparison",
        "",
        f"{view.headline} TakeProfitTrader and MyFundedFutures each run their own five "
        "starting funded accounts with their own monthly credits, costs, payout cash and "
        "growth. Both follow the same strategy signals and the same size on the same market "
        "timeline. There is no shared budget and no total across firms.",
        "",
        f"- Strategy: {view.strategy_text}",
        f"- Settings: {view.settings_text or 'not recorded'}",
        f"- Market period: {view.period_text}",
        f"- Price evidence: {_evidence_sentence(result)}",
        "",
        "## Assumptions",
        "",
        _bullets(f"{d.subject}: {d.decision} ({d.status})" for d in view.decisions)
        or "- None recorded.",
        "",
        "## What would falsify a conclusion",
        "",
        "- A different ordering of prices inside a minute (where candles were used) that "
        "changes which accounts survive.",
        "- Received cash that does not reconcile with the cash ledger, or a payout received "
        "before its processing completes.",
        "- A lead that disappears when the cost difference is accounted for "
        "(compare net cash per dollar of account costs, not only absolute cash).",
        "- A result that depends on one or two large payouts from copies of the same trade: "
        "the accounts share one market path and are not independent experiments.",
        "",
        f"Quantity per trade: {settings.get('quantity', 'not recorded')} "
        f"{settings.get('instrument', '')}.",
    ])


def _evidence_sentence(result: dict[str, Any]) -> str:
    evidence = result.get("price_evidence") or {}
    approx = int(evidence.get("trades_with_minute_approximation") or 0)
    ordered = evidence.get("trades_with_ordered_prints")
    if evidence.get("policy") == "synthetic_fixture":
        return "hand-made synthetic prices (engineering sample, not market data)"
    text = f"{ordered} trade(s) checked on ordered exchange trade prints"
    if approx:
        text += (f"; {approx} trade(s) checked on one-minute candles with an assumed order "
                 "(approximation)")
    return text


def _results(view: FundedResultView) -> str:
    lines = ["# Results", "", view.headline, "", f"**Market period:** {view.period_text}",
             "", "## Status", "", _status_block(view), ""]
    for firm in view.firms:
        lines += [f"## {firm.firm}", "", "| Headline | Value | Note |", "|---|---|---|"]
        lines += [f"| {c.label} | {c.value} | {c.note} |" for c in firm.cards]
        lines += ["", "| Other figure | Value | Note |", "|---|---|---|"]
        lines += [f"| {f.label} | {f.value} | {f.note} |" for f in firm.facts]
        lines += ["", "**Time without trading, by reason (summed over accounts)**", "",
                  "| Reason | Time | Signals affected |", "|---|---|---|"]
        lines += [f"| {w.reason} | {w.hours_text} | {w.count_text} |" for w in firm.waiting]
        if firm.notices:
            lines += ["", "**Outcomes to note**", "", _bullets(firm.notices)]
        lost = [a for a in firm.accounts if a.lost]
        lines += ["", "**Accounts lost**", ""]
        if lost:
            lines += ["| Account | Lost | Reason | Paid out before loss | Replaced by |",
                      "|---|---|---|---|---|"]
            lines += [
                f"| {a.label} | {a.failed} | {a.failure_reason} | "
                f"{a.received if a.payouts_received else 'No payout'} | {a.replaced_by} |"
                for a in lost
            ]
        else:
            lines.append("No accounts were lost.")
        lines += ["", "**Month by month**", "",
                  "| Month | Payouts received | Account costs | Net | Cumulative net |",
                  "|---|---|---|---|---|"]
        lines += [
            f"| {m.label} | ${m.received_usd:,.2f} | ${m.account_costs_usd:,.2f} | "
            f"{_signed(m.net_usd)} | {_signed(m.cumulative_net_usd)} |"
            for m in firm.months
        ]
        lines.append("")
    lines += ["## Comparison of the two separate operations", ""]
    names = [f.firm for f in view.firms]
    if view.comparison:
        header = "| Measure | " + " | ".join(names) + " | Difference (first minus second) |"
        lines += [header, "|" + "---|" * (len(names) + 2)]
        for row in view.comparison:
            values = dict(row.values)
            lines.append(f"| {row.measure} | " + " | ".join(values.get(n, "") for n in names)
                         + f" | {row.difference} |")
    lines += ["", view.comparison_note, "", "## Material limitations", "",
              _bullets(view.limitations)]
    return "\n".join(lines)


def _signed(value: float) -> str:
    cents = int((Decimal(str(value)) * 100).quantize(Decimal(1)))
    return format_usd(cents)


def _decisions(result: dict[str, Any], view: FundedResultView) -> str:
    lines = ["# Decisions", "",
             "Owner decisions and explicit assumptions saved with this run. Assumptions are "
             "labeled as such; they are not claims about published firm rules.", ""]
    if view.decisions:
        lines += ["| Decided | Subject | Decision | Status |", "|---|---|---|---|"]
        lines += [f"| {d.decided_on} | {d.subject} | {d.decision} | {d.status} |"
                  for d in view.decisions]
    else:
        lines.append("No owner decisions were saved with this result.")
    clock = (result.get("settings") or {}).get("processing_clock") or {}
    if clock.get("description"):
        lines += ["", "## Payout processing clock used", "", clock["description"]]
    return "\n".join(lines)


def _trading_rules(result: dict[str, Any], view: FundedResultView) -> str:
    settings = result.get("settings") or {}
    profiles = settings.get("firm_profiles") or []
    clock = settings.get("processing_clock") or {}
    lines = [
        "# Trading rules simulated",
        "",
        "These are the **owner-defined simulation terms**, not a claim of complete "
        "compliance with each live firm program.",
        "",
        "## Owner terms per firm",
        "",
    ]
    for p in profiles:
        cost = int(p.get("acquisition_cost_cents", 0))
        step = int(p.get("capacity_step", 5))
        bps = int(p.get("growth_share_bps", 2500))
        threshold = -(-cost * step * 10_000 // bps) if bps else 0
        lock = int(p.get("floor_lock_cents", 0))
        lines += [
            f"### {p.get('firm_name')}",
            "",
            f"- Account: {p.get('account_label')}, cost {format_usd(cost)} per new funded "
            "account.",
            f"- Trader share of each withdrawal: {p.get('trader_share_pct')}%.",
            f"- Position limit: {int(p.get('max_mini_equivalent_tenths', 0)) // 10} minis or "
            "equivalent (one mini equals ten micros).",
            f"- Loss allowance: {format_usd(int(p.get('loss_allowance_cents', 0)))} below the "
            "highest level reached, " + (
                "moving intraday with realized plus open-position equity"
                if p.get("threshold_update") == "intraday_peak_equity"
                else "moving only at the scheduled session close from the realized closing "
                "balance") + f"; it stops rising at {format_usd(lock)} relative to the start.",
            "- The loss limit is enforced on every ordered price observation while a trade is "
            "open, for both firms.",
            "- Account fails " + _comparator_words(p.get("comparator_before_lock"))
            + " the limit before it locks, and " + _comparator_words(
                p.get("comparator_after_lock")) + " it after it locks.",
            "- No consistency rule, no extra qualifying days and no payout cap.",
            "",
            "Assumptions for this firm:",
            "",
            _bullets(p.get("assumptions") or ["None recorded."]),
            "",
            f"Growth threshold: the next five accounts cost {format_usd(cost * step)}, so the "
            f"payout wallet must hold at least {format_usd(threshold)}.",
            "",
        ]
    lines += [
        "## Fresh funded start",
        "",
        "Every account — initial, replacement or growth — starts directly at funded status "
        "with $0 earned profit, no payout cushion and the full $2,000 loss allowance. No "
        "evaluation is simulated and nothing is inherited from a lost account.",
        "",
        "## Daily close and no weekend holding",
        "",
        "Every position is closed at the mandatory daily close (before the 4:00 PM Chicago "
        "session end, or an earlier scheduled close). No position is held over a weekend.",
        "",
        "## Credits, grants and growth",
        "",
        "- Each firm gets five purchase credits for the first (partial) month; they buy the "
        "first five accounts. Five more credits arrive on the first Chicago calendar day of "
        "each later month. Unused credits carry forward. Credits are an allowance, not income.",
        "- A lost account is replaced with a credit when one is available; otherwise its slot "
        "waits for the next credit. A firm never borrows credits or cash from the other firm.",
        "- Growth: after a payout is actually received, if the next five accounts cost at most "
        "25% of the received payout wallet AND cumulative net cash is positive, the firm buys "
        "five more accounts with payout cash (capacity 5 → 10 → 15 → 20, at most one growth "
        "purchase per firm per receipt day, never above 20).",
        "",
        "## Payout cushion, minimum and split",
        "",
        "- When an account's realized profit leaves at least a $500 GROSS surplus above the "
        "retained $2,100 cushion, it has secured a payout: it stops taking new trades for the "
        "rest of that day.",
        "- At the end of that trading day, while flat, it requests the ENTIRE surplus above "
        "$2,100 (gross, before the firm's share). The gross amount is debited from the account "
        "once, at the request.",
        "- The trader receives the gross amount times the trader share; the firm keeps the rest.",
        "",
        "## Processing clock",
        "",
        clock.get("description") or "Not recorded.",
        "",
        f"- Basis: {str(clock.get('basis', 'not recorded')).replace('_', ' ')}.",
        f"- Payment time: {_twelve_hour(clock.get('payment_time_chicago'))} Chicago.",
        "- The clock starts at the end-of-day request, not when eligibility was reached. The "
        "account stays flat and cannot trade until processing completes; it keeps its slot and "
        "cannot be replaced. The after-split cash is received exactly once, at completion. "
        "Trading resumes only on a new signal after that; session and daily-close limits "
        "still apply.",
        "",
        "## MyFundedFutures equality assumption",
        "",
        "Before the MyFundedFutures floor locks, equity **at or below** the floor fails the "
        "account. This is an explicit pilot assumption: no precise source establishes the "
        "pre-lock equality rule. After the floor locks at +$100, only equity **strictly below** "
        "+$100 fails, per the documented condition.",
        "",
        "## Price evidence and exactness limits",
        "",
        f"This run: {_evidence_sentence(result)}.",
        "",
        "Loss-limit checks are exact only relative to the ordered price observations "
        "supplied. One-minute candles do not say whether the high or the low came first; "
        "where candles were used, the assumed order is labeled an approximation and is never "
        "presented as exact live behavior. Stops and targets fill at the strategy's recorded "
        "prices; only a loss-limit liquidation uses the first recorded print at or through "
        "the limit. None of this is broker execution.",
    ]
    return "\n".join(lines)


def _comparator_words(comparator: str | None) -> str:
    return {"at_or_below": "when equity is at or below",
            "below": "only when equity is strictly below"}.get(
        comparator or "", "under an unrecorded rule relative to")


_FILE_LOCATION = re.compile(
    r"(?:[A-Za-z]:\\|\\\\)[^\s;,)]+|/(?:Users|home|tmp)/[^\s;,)]*"
)


def _readable(text: str) -> str:
    return _FILE_LOCATION.sub("(internal file location omitted)", text)


def _twelve_hour(value: str | None) -> str:
    if not value:
        return "not applicable (elapsed-time clock)"
    hour, minute = (int(part) for part in value.split(":"))
    return f"{(hour - 1) % 12 + 1}:{minute:02d} {'AM' if hour < 12 else 'PM'}"


def _ledger_markdown(entries: list[dict[str, Any]]) -> str:
    lines = ["# Research ledger", "",
             "The cumulative, append-only record of completed, failed, proposed and superseded "
             "experiments, exactly as supplied by the application (`ledger.jsonl` holds the "
             "same entries verbatim). Earlier entries imported from a verified study's own "
             "ledger are marked as imported history. Internal file locations are omitted from "
             "this readable copy only. Nothing here was invented for this export.", ""]
    for index, entry in enumerate(entries, start=1):
        title = (entry.get("title") or entry.get("summary") or entry.get("experiment")
                 or entry.get("question") or entry.get("event_type") or entry.get("kind")
                 or "Entry")
        status = entry.get("status")
        lines.append(f"## {index}. {_readable(str(title))}" + (f" — {status}" if status else ""))
        lines.append("")
        for key, value in entry.items():
            if key in ("title",):
                continue
            label = key.replace("_", " ").capitalize()
            if isinstance(value, list | tuple):
                text = "; ".join(str(v) for v in value) or "—"
            elif isinstance(value, dict):
                text = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in value.items()) or "—"
            else:
                text = "—" if value is None else str(value)
            lines.append(f"- **{label}:** {_readable(text)}")
        lines.append("")
    return "\n".join(lines)


_COLUMN_WORDS: dict[str, str] = {
    "firm_key": "Internal firm identity (takeprofittrader or myfundedfutures).",
    "firm": "Full firm name.",
    "seq": "Order of the event within its firm's simulation.",
    "account_id": "Internal account identity (firm plus account number).",
    "account_number": "Account number within its firm (1 = first account bought).",
    "slot": "Account slot (capacity position) the account occupies.",
    "trade_id": "Strategy signal the trade followed (shared by the firm's copies).",
    "event": "What happened.",
    "kind": "Type of row.",
    "funding": "How the purchase was paid (monthly credit or payout cash).",
    "detail": "Plain-English detail.",
    "reason": "Plain-English reason.",
    "decision": "Growth decision (purchased, declined, not reviewed).",
    "status_before": "Account status before the change.",
    "status_after": "Account status after the change.",
    "status_at_end": "Account status at the end of the period.",
    "failure_reason": "Why the account was lost.",
    "lost_after_payout": "True when the account was lost after receiving a payout.",
    "replaces_account": "The lost account this one replaced.",
    "replaces": "The lost account this one replaced.",
    "request_id": "Payout request identity (links request, receipt and cutoff rows).",
    "clock_basis": "Processing clock basis used for the due time.",
    "after_cutoff": "True when the payout was due after the end of the period.",
    "trading_day": "Trading day (Chicago session date).",
    "receipt_day": "Chicago date of the payout receipt that triggered the growth review.",
    "direction": "Long or short.",
    "quantity": "Contracts traded.",
    "entry_ticks": "Entry price in ticks (points = ticks x 0.25).",
    "stop_ticks": "Protective stop in ticks (points = ticks x 0.25).",
    "exit_ticks": "Actual exit price in ticks (points = ticks x 0.25).",
    "strategy_exit_ticks": "The strategy's own recorded exit in ticks.",
    "strategy_exit_reason": "The strategy's own exit reason.",
    "exit_kind": "How the funded-account position actually closed.",
    "price_ticks": "Observed price in ticks (points = ticks x 0.25).",
    "observations_checked": "Ordered price observations checked against the loss limit.",
    "price_evidence": "Kind of price data used for the loss-limit checks.",
    "account_failed": "True when this trade ended the account.",
    "approximate": "True when the outcome depends on an assumed within-minute order.",
    "check": "Rule check recorded.",
    "outcome": "Result of the check.",
    "comparator": "Loss-limit comparison used (at or below / strictly below).",
    "change": "Credit count change.",
    "credits_after": "Unused credits after the event.",
    "grant_id": "Month of the credit grant (issued once).",
    "capacity_before": "Account capacity before the growth decision.",
    "capacity_after": "Account capacity after the growth decision.",
    "month": "Chicago calendar month.",
    "partial_month": "True for the partial first or last month of the period.",
    "payouts_received": "Payouts received.",
    "accounts_bought": "Accounts bought.",
    "trades": "Trades taken.",
    "measure": "Comparison measure.",
    "unit": "Unit of the measure.",
    "difference_first_minus_second": "First firm minus second firm (not a total).",
    "source_id": "Source reference.",
    "title": "Source title.",
    "url": "Source address.",
    "checked_on": "Date the source was checked.",
    "supports": "What the source supports.",
    "vacancy_wait_hours": "Hours the slot waited for a credit.",
    "paid_before_failure": "True when the account had a payout before it was lost.",
    "trader_share_pct": "Trader share of each gross withdrawal, percent.",
    "max_minis": "Position limit in minis or equivalent.",
    "net_cash_per_dollar_of_account_costs": "Net cash earned per dollar of account costs.",
}


def _describe_column(column: str) -> str:
    if column in _COLUMN_WORDS:
        return _COLUMN_WORDS[column]
    words = column.replace("_", " ")
    if column.endswith("_usd"):
        return f"{words[:-4].capitalize()}, US dollars (exact to the cent)."
    if column.endswith("_utc"):
        return f"{words[:-4].capitalize()}, machine time (UTC, ISO 8601)."
    if column.endswith("_chicago"):
        return f"{words[:-8].capitalize()}, Chicago local time for reading (12-hour clock)."
    if column.startswith("hours_"):
        return f"Account-hours {words[6:]}, summed over accounts."
    if column.startswith(("entries_blocked", "signals_")):
        return f"{words.capitalize()} (count)."
    return f"{words.capitalize()}."


def _data_dictionary(staging: Path) -> str:
    lines = ["# Data dictionary", "",
             "Money columns ending in `_usd` are US dollars exact to the cent. Times appear "
             "twice: `_utc` for machines and `_chicago` for reading. Prices in ticks convert "
             "to index points by multiplying by 0.25. Every table carries the firm name; one "
             "logical table is one file.", ""]
    for filename in [*CSV_TABLES.values(), "rule_sources.csv"]:
        path = staging / filename
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), [])
            rows = sum(1 for _ in handle)
        lines += [f"## {filename}", "", _FILE_PURPOSE.get(filename, "Source references."),
                  f"Rows: {rows}.", "", "| Column | Meaning |", "|---|---|"]
        lines += [f"| `{c}` | {_describe_column(c)} |" for c in header]
        lines.append("")
    lines += [
        "## JSON files",
        "",
        "- `settings.json`: run settings (period, instrument, size, costs, processing clock, "
        "price-evidence summary).",
        "- `firm_rules.json`: the two firm profiles exactly as simulated (money in cents).",
        "- `validation_summary.json`: compact internal result checks and export checks.",
        "- `run_manifest.json`: every other file's size and SHA-256 checksum.",
        "- `ledger.jsonl`: one JSON research-ledger entry per line, cumulative.",
    ]
    return "\n".join(lines)


# ── charts (matplotlib, Agg canvas, no global backend state) ─────────────


def _matplotlib():
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as error:  # pragma: no cover - environment dependent
        raise FundedReviewExportError(
            "matplotlib is required to draw the review-folder charts but is not "
            "installed; install matplotlib to publish funded review folders") from error
    return Figure, FigureCanvasAgg


_FIRM_LINES = (("#1f5fa8", "-", "o"), ("#c0621b", "--", "s"))


def _charts(folder: Path, view: FundedResultView) -> None:
    figure_cls, canvas_cls = _matplotlib()
    end = chicago_datetime(view.period_end_utc)
    start = chicago_datetime(view.period_start_utc)

    fig = figure_cls(figsize=(10, 8), dpi=100)
    canvas_cls(fig)
    axes = fig.subplots(3, 1, sharex=True)
    titles = ("Payouts received after the firm's share (cumulative, US dollars)",
              "Account costs (cumulative, US dollars)",
              "Net cash earned after all account costs (cumulative, US dollars)")
    for index, firm in enumerate(view.firms):
        color, style, marker = _FIRM_LINES[index % len(_FIRM_LINES)]
        xs = [chicago_datetime(p.ts_utc) for p in firm.cash]
        for ax, values in zip(axes, (
            [p.received_usd for p in firm.cash],
            [p.account_costs_usd for p in firm.cash],
            [p.net_cash_usd for p in firm.cash],
        ), strict=True):
            x, y = list(xs), list(values)
            if end is not None and y:
                x.append(end)
                y.append(y[-1])
            if x:
                ax.step(x, y, where="post", color=color, linestyle=style, marker=marker,
                        markersize=3, label=firm.firm)
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_formatter(_dollar_formatter())
        ax.grid(alpha=0.3)
        if start is not None and end is not None:
            ax.set_xlim(start, end)
    axes[-1].xaxis.set_major_formatter(_date_formatter())
    axes[0].legend(loc="upper left", fontsize=9)
    axes[-1].set_xlabel("Date (Chicago time)")
    fig.suptitle(view.headline, fontsize=11)
    fig.tight_layout()
    fig.savefig(folder / "cash_over_time.png", format="png", metadata={"Software": None})

    rows = max((len(f.accounts) for f in view.firms), default=1)
    fig = figure_cls(figsize=(11, max(4.0, 0.28 * rows + 2.0) * max(1, len(view.firms))),
                     dpi=100)
    canvas_cls(fig)
    axes = fig.subplots(max(1, len(view.firms)), 1, squeeze=False)[:, 0]
    for ax, firm in zip(axes, view.firms, strict=False):
        labels = [a.label for a in firm.accounts]
        ypos = {a.account_id: len(labels) - i for i, a in enumerate(firm.accounts)}
        for account in firm.accounts:
            y = ypos[account.account_id]
            begin = chicago_datetime(account.created_utc)
            finish = chicago_datetime(account.failed_utc) if account.lost else end
            ax.plot([begin, finish], [y, y], color="#607d8b", linewidth=2)
            ax.plot([begin], [y], marker=">", color="#37474f", markersize=6)
            for a, b in account.processing_intervals_utc:
                ax.plot([chicago_datetime(a), chicago_datetime(b) if b else end], [y, y],
                        color="#f9a825", linewidth=6, solid_capstyle="butt")
            for ts, cents in account.receipts_utc:
                ax.plot([chicago_datetime(ts)], [y], marker="D", color="#2e7d32",
                        markersize=6)
                ax.annotate(f"${cents / 100:,.0f}", (chicago_datetime(ts), y),
                            textcoords="offset points", xytext=(0, 5), fontsize=6,
                            ha="center")
            if account.lost:
                ax.plot([chicago_datetime(account.failed_utc)], [y], marker="X",
                        color="#b71c1c", markersize=8)
                ax.annotate("lost", (chicago_datetime(account.failed_utc), y),
                            textcoords="offset points", xytext=(6, -3), fontsize=6)
            if account.replaces_id and account.replaces_id in ypos:
                old = next(a for a in firm.accounts if a.account_id == account.replaces_id)
                ax.plot([chicago_datetime(old.failed_utc), begin],
                        [ypos[old.account_id], y], color="#9e9e9e", linestyle=":",
                        linewidth=1)
        ax.set_yticks(list(range(len(labels), 0, -1)), labels, fontsize=7)
        ax.set_title(f"{firm.firm}: account journeys (triangle = start, yellow = payout "
                     "processing, diamond = payout received, X = lost, dotted = "
                     "replacement)", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        if start is not None and end is not None:
            ax.set_xlim(start, end)
        ax.set_xlabel("Date (Chicago time)")
        ax.xaxis.set_major_formatter(_date_formatter())
        if not labels:
            ax.text(0.5, 0.5, "No accounts", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    fig.savefig(folder / "account_journeys.png", format="png", metadata={"Software": None})


def _date_formatter():
    from matplotlib.dates import DateFormatter

    return DateFormatter("%b %d, %Y")


def _dollar_formatter():
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda value, _pos: f"-${-value:,.0f}" if value < 0
                         else f"${value:,.0f}")

