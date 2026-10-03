"""Configuration detail · Settings and evidence (mock 08).

The earlier evidence panels for ONE configuration at ONE firm, kept and
reorganized as the mock shows: the configuration's settings with their signal,
mark and fill sources; the verification checklist; reporting corrections; what
this result can't tell you (every stated limitation); your decisions and
assumptions (every decision in full); and, on the right of the tab bar, a
download of the latest published review folder for this result.

A "More" section at the bottom keeps earlier features the mock has no place for:
the full "Strategy measures without accounts" table for every configuration,
the study's question, period and sizes, and the "one simulated historical path"
note.

Everything is read from the saved result, its plan and — for two facts the
result does not carry (which minute approximations could matter, and the gap
rule when the plan lacks it) — the latest published review folder, after its
manifest's hash check. Nothing here recomputes money. The one write is the
earlier study page's owner action, kept here: when the run state records that
the review folder failed to publish, "Publish the review folder again" calls the
same ``publish_comparison_review`` (a new review folder for the same saved
result; nothing is run or recomputed).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import streamlit as st

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab import html as h
from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import css_var

__all__ = [
    "Correction",
    "PUBLISH_LABEL",
    "PublishAgain",
    "ReviewFolder",
    "SettingsView",
    "approximation_effects",
    "build_settings_view",
    "chart_list_text",
    "corrections",
    "decision_rows",
    "gap_rule_text",
    "header_right",
    "latest_review_folder",
    "publish_again",
    "publish_again_route",
    "render",
    "settings_rows",
    "verification_items",
]

KEY = "ifvg_lab_v1_"
NOT_IN_EXPORT = "Not in this study's export"
DOWNLOAD_LABEL = "Download the full study review folder"

#: plain words for the saved big-gap invalidation rules (mock 08 wording first)
GAP_RULES = {
    "own_timeframe_close_v1": "A candle on its own chart closes through it",
    "execution_wick_full_fill_v1": "A one-minute wick reaches its far edge (the original rule)",
}

_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10,
            "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "sixty": 60}

_DECISION_STATUS = {
    "owner_confirmed": ("Confirmed by you", False),
    "owner_approved_for_pilot": ("Approved by you for the pilot", False),
    "assumption": ("Assumption, not from a published source", True),
}

#: short titles for the saved reporting corrections (the records carry none)
_CORRECTION_TITLES = {
    "stop_difference_summary_final_quantity_v1": "Stop exits filled worse than the stop",
    "daily_close_wording_v1": "“No overnight holding” wording",
}

_APPROX_EFFECT_COLUMNS = ("ordering_can_change_half_exit",
                          "ordering_can_change_final_exit_or_result",
                          "ordering_can_change_account_survival",
                          "ordering_can_change_payout_qualification")


# ── settings ──────────────────────────────────────────────────────────────


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def chart_list_text(value: str) -> str:
    """``one-hour and four-hour`` → ``1-hour and 4-hour``; minutes → ``1, 3 and 5 minutes``."""

    found = re.findall(r"\b([a-z]+)-(minute|hour)\b", value or "")
    numbers = [(_NUMBERS.get(word), unit) for word, unit in found]
    if not found or any(n is None for n, _ in numbers):
        return _cap(value or "")
    units = {unit for _, unit in numbers}
    if units == {"minute"} and len(numbers) > 1:
        values = [str(n) for n, _ in numbers]
        return ", ".join(values[:-1]) + f" and {values[-1]} minutes"
    parts = [f"{n}-{unit}" for n, unit in numbers]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _variant_axes(plan: Any, configuration: str) -> dict[str, str]:
    for variant in getattr(plan, "variants", None) or ():
        if getattr(variant, "name", None) == configuration:
            return {str(k): str(v) for k, v in getattr(variant, "axis_value_ids", ()) or ()}
    return {}


def gap_rule_text(study, configuration: str,
                  bindings: dict[str, Any] | None = None) -> str | None:
    """When a big gap stops counting, from the plan's saved variant or the review folder."""

    value = _variant_axes(study.plan, configuration).get("htf_gap_invalidation_policy")
    if value:
        value = value.split(".", 1)[1] if "." in value else value
    if not value and bindings:
        for row in bindings.get("configurations") or []:
            if row.get("configuration") == configuration:
                value = ((row.get("effective_section") or {})
                         .get("htf_gap_invalidation_policy"))
    if not value:
        return None
    if value in GAP_RULES:
        return GAP_RULES[value]
    from alpha_lab.agents.data_infra.ifvg.presentation.axis_values import format_axis_value

    return format_axis_value("htf_gap_invalidation_policy", value)


def _plain_product(text: str) -> str:
    """Product names without exchange ticker codes (``(NQ)``, ``(MNQ)``)."""

    return re.sub(r"\s*\((?:M?NQ)\)", "", text or "").strip()


def _cost_text(usd: Any) -> str:
    value = Decimal(str(usd)).normalize()
    if -value.as_tuple().exponent < 2:
        value = value.quantize(Decimal("0.01"))
    return f"${value:,}"


def _sizing(result: dict[str, Any], configuration: str) -> dict[str, Any]:
    settings = result.get("settings") or {}
    return (settings.get("sizing_by_configuration") or {}).get(configuration) or {}


def _size_rows(result: dict[str, Any], configuration: str,
               stored: dict[str, str]) -> list[tuple[str, Any]]:
    sizing = _sizing(result, configuration)
    quantity, cost = sizing.get("quantity"), sizing.get("cost_per_contract_per_fill_usd")
    label = _plain_product(str(sizing.get("instrument_label") or ""))
    if quantity is None or cost is None or not label:
        return [("Position size and cost", stored.get("Position size")
                 or h.placeholder(NOT_IN_EXPORT))]
    unit = "micro" if sizing.get("instrument") == "micro" else "E-mini"
    size = (f"{int(quantity):,} {unit}{'s' if int(quantity) != 1 else ''} per trade · "
            f"{_cost_text(cost)} per {unit} per fill")
    return [("Position size and cost", size), ("Traded product", label)]


def _engine_text(result: dict[str, Any], configuration: str) -> str | None:
    core = (result.get("settings") or {}).get("core_source") or {}
    if not core:
        return None
    words = f"{core.get('branch') or ''} {core.get('description') or ''}".lower()
    if not core.get("branch") and not core.get("patch_sha256"):
        return "The standard strategy engine"
    text = "Research version"
    if "scale-out" in words or "scale_out" in words:
        text += " with the half exit"
    text += " · the default engine is unchanged"
    exit_policy = str(_sizing(result, configuration).get("exit_policy") or "")
    if ("scale-out" in words or "scale_out" in words) and exit_policy \
            and not exit_policy.startswith("scale_out"):
        text += " · this configuration exits whole, as in the default engine"
    return text


_KNOWN = ("Entry hours", "Direction", "Higher-timeframe gap charts", "Supporting (parent) charts",
          "Largest distance from the parent gap to the opposing gap", "Smallest opposing gap",
          "Profit target", "Stop", "Exit rule", "Trades per day", "Daily close", "Position size")


def settings_rows(study, configuration: str,
                  bindings: dict[str, Any] | None = None) -> list[tuple[str, Any]]:
    """The "Configuration settings" rows, in the mock's order, in plain words."""

    from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
        execution_sources,
    )

    stored = dict(study.settings(configuration))
    missing = h.placeholder(NOT_IN_EXPORT)

    def value(name: str, transform=_cap) -> Any:
        text = stored.get(name)
        return transform(text) if text else missing

    gap = gap_rule_text(study, configuration, bindings)
    sources = {name: _plain_product(text)
               for name, text in execution_sources(study.result, configuration)}
    engine = _engine_text(study.result, configuration)
    rows: list[tuple[str, Any]] = [
        ("Entry hours", value("Entry hours")),
        ("Direction", value("Direction")),
        ("Big gap charts", value("Higher-timeframe gap charts", chart_list_text)),
        ("When a big gap stops counting", gap or missing),
        ("Supporting charts", value("Supporting (parent) charts", chart_list_text)),
        ("Largest distance, parent gap to opposing gap",
         value("Largest distance from the parent gap to the opposing gap")),
        ("Smallest opposing gap", value("Smallest opposing gap")),
        ("Profit target", value("Profit target")),
        ("Stop", value("Stop")),
        ("Exit", value("Exit rule")),
        ("Trades per day", value("Trades per day")),
        ("Daily close", value("Daily close")),
        *_size_rows(study.result, configuration, stored),
        ("Signals from", sources.get("Signal source") or missing),
        ("Open-position marks and loss-limit checks",
         sources.get("Open-position marks and loss-limit checks") or missing),
        ("Fill prices", sources.get("Execution prices") or missing),
        ("Strategy engine", engine or missing),
    ]
    rows += [(name, _cap(text)) for name, text in stored.items() if name not in _KNOWN]
    return rows


# ── published review folder (read only) ───────────────────────────────────


@dataclass(frozen=True)
class ReviewFolder:
    """The latest published review folder of one saved result."""

    name: str
    version: int
    folder: Path | None
    zip_path: Path | None
    published_utc: str | None
    files: dict[str, str]  # published path → SHA-256 from the manifest

    @property
    def size_bytes(self) -> int | None:
        if self.zip_path is not None:
            return self.zip_path.stat().st_size
        return None

    def read(self, rel: str) -> bytes | None:
        """One listed file's bytes, only when they match the manifest's SHA-256."""

        expected = self.files.get(rel)
        if expected is None:
            return None
        data = None
        if self.folder is not None and (self.folder / rel).is_file():
            data = (self.folder / rel).read_bytes()
        elif self.zip_path is not None:
            with zipfile.ZipFile(self.zip_path) as archive:
                try:
                    data = archive.read(f"{self.name}/{rel}")
                except KeyError:
                    data = None
        if data is None or hashlib.sha256(data).hexdigest() != expected:
            return None
        return data


def _manifest(folder: Path | None, zip_path: Path | None, name: str) -> dict[str, Any] | None:
    try:
        if folder is not None and (folder / "run_manifest.json").is_file():
            return json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
        if zip_path is not None:
            with zipfile.ZipFile(zip_path) as archive:
                return json.loads(archive.read(f"{name}/run_manifest.json").decode("utf-8"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None
    return None


def latest_review_folder(repo_root: Path | str | None, result_id: str) -> ReviewFolder | None:
    """The highest published export version for this exact result (folder and/or zip)."""

    from alpha_lab.agents.data_infra.ifvg.funded_comparison_review import (
        PARENT,
        comparison_review_folder_name,
    )

    if not repo_root or not result_id:
        return None
    parent = Path(repo_root) / "reports" / PARENT
    if not parent.is_dir():
        return None
    prefix = comparison_review_folder_name(result_id, 0)[:-1]
    versions = set()
    for path in parent.glob(f"{prefix}*"):
        stem = path.name[:-4] if path.name.endswith(".zip") else path.name
        tail = stem[len(prefix):]
        if tail.isdigit() and (path.is_dir() or path.suffix == ".zip"):
            versions.add(int(tail))
    for version in sorted(versions, reverse=True):
        name = comparison_review_folder_name(result_id, version)
        folder = parent / name if (parent / name).is_dir() else None
        zip_path = parent / f"{name}.zip" if (parent / f"{name}.zip").is_file() else None
        manifest = _manifest(folder, zip_path, name)
        if not manifest or manifest.get("funded_comparison_result_id") != result_id:
            continue  # not this result's folder, or its manifest is unreadable
        files = {str(f.get("path")): str(f.get("sha256")) for f in manifest.get("files") or []}
        return ReviewFolder(name=name, version=version, folder=folder, zip_path=zip_path,
                            published_utc=manifest.get("published_at_utc"), files=files)
    return None


def approximation_effects(review: ReviewFolder | None) -> tuple[int, int] | None:
    """(approximated minutes listed, how many could change an exit, survival or payout)."""

    data = review.read("approximated_minutes.csv") if review else None
    if data is None:
        return None
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    if rows and not all(c in rows[0] for c in _APPROX_EFFECT_COLUMNS):
        return None
    could = sum(1 for r in rows
                if any(str(r.get(c)).strip().lower() == "true" for c in _APPROX_EFFECT_COLUMNS))
    return len(rows), could


def review_bindings(review: ReviewFolder | None) -> dict[str, Any] | None:
    data = review.read("configuration_bindings.json") if review else None
    try:
        return json.loads(data.decode("utf-8")) if data else None
    except ValueError:
        return None


def folder_zip_bytes(review: ReviewFolder) -> bytes:
    """The published zip as saved, or the published folder zipped in memory (read only)."""

    if review.zip_path is not None:
        return review.zip_path.read_bytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(review.folder.rglob("*")):
            if path.is_file():
                archive.write(path, f"{review.name}/{path.relative_to(review.folder).as_posix()}")
    return buffer.getvalue()


# ── verification, corrections, decisions ─────────────────────────────────


def _firm_trades(result: dict[str, Any], firm_key: str) -> list[dict[str, Any]]:
    return [t for t in (result.get("tables") or {}).get("trades") or []
            if t.get("firm_key") == firm_key]


def verification_items(result: dict[str, Any], configuration: str, firm_key: str, firm: str,
                       effects: tuple[int, int] | None) -> list[tuple[str, str]]:
    """The checklist: (``check`` or ``warn``, one plain sentence) each."""

    items: list[tuple[str, str]] = []
    evidence = result.get("price_evidence") or {}
    checked = int(evidence.get("position_minutes_checked") or 0)
    exact = int(evidence.get("position_minutes_rebuilt_exactly_from_prints") or 0)
    approx = int(evidence.get("position_minutes_approximated") or 0)
    missing_days = list(evidence.get("missing_print_days") or [])
    tables = result.get("tables") or {}
    own = next((r for r in tables.get("execution_evidence") or []
                if r.get("configuration") == configuration), None)
    if checked:
        text = f"{exact:,} of {checked:,} position minutes used recorded exchange trades."
        if approx:
            text += f" {approx:,} used a labeled one-minute approximation"
            if own is not None:
                mine = (int(own.get("position_minutes_checked") or 0)
                        - int(own.get("position_minutes_rebuilt_exactly_from_prints") or 0))
                text += (f", {mine:,} of them in this configuration" if mine
                         else ", none of them in this configuration")
            if effects is not None:
                # the review folder lists each approximated minute once per trade row it
                # touches; "could change" is its own check against every stop, target,
                # break-even stop and loss limit inside that minute's range
                text += (", and none could change an exit, a survival or a payout."
                         if effects[1] == 0 else
                         f"; in {effects[1]:,} of the {effects[0]:,} affected trade rows the "
                         "order inside the minute could change an exit, a survival or a payout "
                         "(listed in the review folder).")
            else:
                text += "."
        else:
            text += " None needed the one-minute approximation."
        if missing_days:
            text += f" Recorded trades were missing on {fmt.count(len(missing_days), 'day')}."
        ok = not missing_days and exact + approx == checked
        items.append(("check" if ok else "warn", text))
    else:
        items.append(("warn", "Price evidence for position minutes is not in this study's "
                              "export."))

    validation = result.get("validation")
    if validation is None:
        items.append(("warn", "This result has no internal money checks. Don't rely on its "
                              "figures."))
    else:
        pairs = [checks for firms in (validation.get("checks") or {}).values()
                 if isinstance(firms, dict)
                 for checks in firms.values() if isinstance(checks, dict)]
        good = sum(1 for checks in pairs if checks and all(checks.values()))
        if validation.get("passed") and pairs and good == len(pairs):
            items.append(("check", "Receipts, account costs and balances reconcile for all "
                                   f"{len(pairs):,} results."))
        else:
            items.append(("warn", "Receipts, account costs and balances reconcile for "
                                  f"{good:,} of {len(pairs):,} results."))

    rows = tables.get("execution_evidence") or []
    if rows:
        same = sum(1 for r in rows if r.get("no_account_replay_equals_saved_study"))
        items.append(("check" if same == len(rows) else "warn",
                      "With accounts switched off, every configuration's trades match the saved "
                      f"study or the engine's standard replay: {same:,} of {len(rows):,}."
                      if same == len(rows) else
                      f"With accounts switched off, {same:,} of {len(rows):,} configurations' "
                      "trades match the saved study or the engine's standard replay."))
        resumed = sum(1 for r in rows if r.get("resumed_run_identical"))
        items.append(("check" if resumed == len(rows) else "warn",
                      "Stopping halfway and resuming gave identical results: "
                      f"{resumed:,} of {len(rows):,}."))
    else:
        items.append(("warn", "The no-account replay and resume checks are not in this study's "
                              "export."))

    sizing = (result.get("settings") or {}).get("sizing_by_configuration") or {}
    micro = {k for k, v in sizing.items() if v.get("instrument") == "micro"}
    if micro:
        at_firm = [t for t in _firm_trades(result, firm_key) if t.get("configuration") in micro]
        text = f"All {len(at_firm):,} micro positions at {firm} were priced from E-mini trades"
        if configuration in micro:
            mine = sum(1 for t in at_firm if t.get("configuration") == configuration)
            text += f", {mine:,} of them in this configuration."
        else:
            text += ". This configuration trades the E-mini itself."
        items.append(("warn", text + " No micro trade data was used."))

    approval = result.get("approval") or {}
    if result.get("purpose") == "engineering_sample":
        items.append(("warn", "Engineering sample: not an approved study."))
    elif approval.get("approved_on"):
        items.append(("check", f"Approved by you {fmt.date_long(approval['approved_on'])}."))
    else:
        items.append(("warn", "No approval is recorded for this result."))
    return items


@dataclass(frozen=True)
class Correction:
    title: str
    description: str
    this_configuration: str | None


def _past_midnight(trades: list[dict[str, Any]]) -> int:
    from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import CHICAGO, utc_instant

    count = 0
    for trade in trades:
        entry, exit_ = utc_instant(trade.get("entry_utc")), utc_instant(trade.get("exit_utc"))
        if entry is not None and exit_ is not None and \
                entry.tz_convert(CHICAGO).date() != exit_.tz_convert(CHICAGO).date():
            count += 1
    return count


def corrections(result: dict[str, Any], configuration: str, firm_key: str, firm: str,
                trades: list[dict[str, Any]]) -> list[Correction]:
    """Each saved reporting correction, with this configuration's before and after."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import pair_key

    pair = pair_key(configuration, firm_key)
    out = []
    for index, record in enumerate(result.get("reporting_corrections") or [], start=1):
        cid = str(record.get("correction_id") or "")
        title = _CORRECTION_TITLES.get(cid) or f"Reporting correction {index}"
        line = None
        change = next((c for c in record.get("changes") or [] if c.get("pair_id") == pair), None)
        before, after = (change or {}).get("before") or {}, (change or {}).get("after") or {}
        if "stop_slippage_cents" in after:
            count_b = before.get("stop_exits_filled_worse_than_stop")
            count_a = after.get("stop_exits_filled_worse_than_stop")
            was = (fmt.money_cents(before.get("stop_slippage_cents"))
                   if count_b == count_a else
                   f"{fmt.count(int(count_b or 0), 'exit')}, "
                   f"{fmt.money_cents(before.get('stop_slippage_cents'))}")
            line = (f"This configuration at {firm}: {fmt.count(int(count_a or 0), 'exit')}, "
                    f"{fmt.money_cents(after['stop_slippage_cents'])} (was {was}).")
        elif change is not None:
            line = f"This configuration at {firm} is among the results this corrected."
        elif "daily_close" in cid or any("Daily close" in str(f)
                                         for f in record.get("fields") or []):
            late = _past_midnight(trades)
            line = (f"This configuration at {firm}: {late:,} of {fmt.count(len(trades), 'trade')} "
                    f"{'was' if late == 1 else 'were'} open past midnight inside one trading "
                    "day.")
        elif record.get("pairs_changed") is not None:
            line = f"This configuration at {firm} was not changed by this correction."
        # system-written text: shown with the display wording map (saved text unchanged)
        out.append(Correction(title, fmt.display_words(record.get("description") or ""), line))
    return out


def decision_rows(result: dict[str, Any]) -> list[tuple[str, str, str, str, bool]]:
    """(decided, subject, decision in full, status, is an assumption) per saved decision."""

    rows = []
    for decision in result.get("owner_decisions") or []:
        status = str(decision.get("status") or "")
        words, assumption = _DECISION_STATUS.get(
            status, (_cap(status.replace("_", " ")), "assumption" in status))
        rows.append((fmt.date_short(decision.get("decided_on")),
                     str(decision.get("subject") or ""), str(decision.get("decision") or ""),
                     words, assumption))
    return rows


# ── view ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SettingsView:
    settings: tuple[tuple[str, Any], ...]
    verification: tuple[tuple[str, str], ...]
    corrections: tuple[Correction, ...]
    limitations: tuple[str, ...]
    decisions: tuple[tuple[str, str, str, str, bool], ...]
    review: ReviewFolder | None


def build_settings_view(study, configuration: str, firm_key: str, firm: str,
                        repo_root: Path | str | None) -> SettingsView:
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import ordered_trades

    result = study.result
    review = latest_review_folder(repo_root, study.result_id)
    bindings = None
    if not _variant_axes(study.plan, configuration).get("htf_gap_invalidation_policy"):
        bindings = review_bindings(review)
    return SettingsView(
        settings=tuple(settings_rows(study, configuration, bindings)),
        verification=tuple((kind, fmt.display_words(text)) for kind, text in verification_items(
            result, configuration, firm_key, firm, approximation_effects(review))),
        corrections=tuple(corrections(result, configuration, firm_key, firm,
                                      list(ordered_trades(study, configuration, firm_key)))),
        limitations=tuple(fmt.display_words(x) for x in result.get("limitations") or []),
        decisions=tuple(decision_rows(result)),
        review=review)


# ── HTML ──────────────────────────────────────────────────────────────────

_v = css_var  # a palette color as its CSS variable: the page's theme picks the value


def _settings_card(view: SettingsView) -> str:
    rows = "".join(
        '<div style="display:grid;grid-template-columns:200px minmax(0,1fr);gap:12px;'
        f'padding:9px 0;border-top:1px solid {_v("light_rule")};font-size:14px;line-height:1.45">'
        f'<div style="color:{_v("body_2")}">{h.esc(label)}</div><div>{h.esc(value)}</div></div>'
        for label, value in view.settings)
    return ('<section class="lab lab-card" style="gap:0">'
            '<div class="lab-card-title" style="margin-bottom:12px">Configuration settings</div>'
            f"{rows}</section>")


def mark(kind: str) -> str:
    """Check or caution mark drawn with text and CSS (``st.html`` removes inline SVG)."""

    if kind == "warn":
        return ('<span role="img" aria-label="Caution" style="display:inline-flex;'
                'align-items:center;justify-content:center;width:16px;height:16px;'
                f'box-sizing:border-box;border:2px solid {_v("orange")};border-radius:50%;'
                f'color:{_v("orange")};font-size:11px;font-weight:700;line-height:1;'
                'margin-top:2px">!</span>')
    return (f'<span role="img" aria-label="Checked" style="color:{_v("blue")};font-weight:700;'
            'font-size:16px;line-height:1.3">✓</span>')


def _verification_card(view: SettingsView) -> str:
    items = "".join(
        '<div style="display:grid;grid-template-columns:20px minmax(0,1fr);gap:10px;'
        f'font-size:14px;line-height:1.5">{mark(kind)}'
        f"<div>{h.esc(text)}</div></div>" for kind, text in view.verification)
    return ('<section class="lab lab-card" style="gap:10px">'
            '<div class="lab-card-title" style="margin-bottom:4px">Verification</div>'
            f"{items}</section>")


def _corrections_card(view: SettingsView) -> str:
    if not view.corrections:
        body = '<div class="lab-line">No reporting corrections were recorded.</div>'
    else:
        body = "".join(
            '<div style="font-size:14px;line-height:1.5"><div style="font-weight:600">'
            f'{h.esc(c.title)}</div><div style="color:{_v("body")}">{h.esc(c.description)}</div>'
            + (f'<div style="color:{_v("body")};margin-top:4px">{h.esc(c.this_configuration)}'
               "</div>" if c.this_configuration else "") + "</div>" for c in view.corrections)
    return ('<section class="lab lab-card" style="gap:12px">'
            f'<div class="lab-card-title">Reporting corrections · {len(view.corrections)}</div>'
            f"{body}</section>")


def top_cards(view: SettingsView) -> h.Markup:
    return h.Markup(
        '<div class="lab lab-grid" style="grid-template-columns:minmax(0,1.1fr) '
        'minmax(0,0.9fr);gap:16px;align-items:start">'
        f"{_settings_card(view)}"
        '<div style="display:flex;flex-direction:column;gap:16px;min-width:0">'
        f"{_verification_card(view)}{_corrections_card(view)}</div></div>")


def limitations_card(view: SettingsView) -> h.Markup:
    if not view.limitations:
        body = str(h.placeholder("No limitations were saved with this result; treat it as "
                                 "incomplete evidence."))
    else:
        body = ('<div class="lab-grid" style="display:grid;grid-template-columns:repeat(2,'
                'minmax(0,1fr));gap:10px 32px;font-size:14px;line-height:1.5;'
                f'color:{_v("body")}">'
                + "".join(f"<div>· {h.esc(item)}</div>" for item in view.limitations)
                + "</div>")
    return h.Markup('<section class="lab lab-card" style="gap:10px">'
                    '<div class="lab-card-title" style="margin-bottom:4px">What this result '
                    f"can't tell you</div>{body}</section>")


def decisions_card(view: SettingsView) -> h.Markup:
    columns = [h.Column("decided", "Decided", width="90px"),
               h.Column("subject", "Subject", width="22%"), h.Column("decision", "Decision"),
               h.Column("status", "Status", width="18%")]
    rows = []
    for decided, subject, decision, status, assumption in view.decisions:
        status_html = (h.Markup(f'<span style="color:{_v("orange_dark")};font-weight:600">'
                                f"{h.esc(status)}</span>") if assumption else status)
        rows.append(h.Row({"decided": decided, "subject": subject,
                           "decision": h.Markup(f'<span style="line-height:1.45">'
                                                f"{h.esc(decision)}</span>"),
                           "status": status_html}))
    body = (h.table(columns, rows, plain=True, wrap=False) if rows
            else h.placeholder("No decisions were saved with this result."))
    return h.card(body, title="Your decisions and assumptions used")


_METRIC_COLUMNS = ("Trades", "Long / short", "Win rate", "Net R after costs",
                   "Average R per trade", "Profit factor", "Largest drawdown",
                   "Longest time under water")


def strategy_table(study, configuration: str) -> h.Markup:
    """The full "Strategy measures without accounts" table (every configuration)."""

    from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
        present_strategy_metrics,
    )
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.names import study_names

    result = study.result
    shown = present_strategy_metrics(result)
    if not shown:
        return h.placeholder("Strategy measures without accounts are not in this study's export.")
    # the presenter's own order (best net R first), to name and tint each row
    order = sorted((result.get("tables") or {}).get("strategy_metrics") or [],
                   key=lambda r: (-(r.get("net_r_after_costs") or -1e9), r["configuration"]))
    # one name per configuration of the whole study, never two rows alike
    keys = dict.fromkeys([*map(str, study.configurations),
                          *(str(r.get("configuration")) for r in order)])
    names = study_names({key: study.settings(key) for key in keys})
    columns = [h.Column("config", "Configuration", width="34%")] + [
        h.Column(f"m{i}", name, "right") for i, name in enumerate(_METRIC_COLUMNS)]
    rows = []
    for raw, row in zip(order, shown, strict=False):
        key = str(raw.get("configuration"))
        name = names[key]
        cells = {"config": h.cell_two_lines(name.line1, name.line2)}
        cells.update({f"m{i}": row.get(name_, "") for i, name_ in enumerate(_METRIC_COLUMNS)})
        rows.append(h.Row(cells, tint="leader" if key == configuration else None))
    return h.Markup('<style>.p3-measures td.num{white-space:nowrap}</style>'
                    '<div class="p3-measures" style="max-height:620px;overflow:auto;'
                    f'border:1px solid {_v("light_rule")};border-radius:8px">'
                    f"{h.table(columns, rows, plain=True, wrap=False)}</div>")


def about_card(result: dict[str, Any]) -> h.Markup:
    from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import FUTURE_NOTE

    period = result.get("period") or {}
    requested = int(result.get("configurations_requested") or 0)
    completed = int(result.get("configurations_completed") or 0)
    purpose = {"historical_comparison": "Your approved historical comparison",
               "engineering_sample": "Engineering sample, not an approved study"}.get(
        str(result.get("purpose")), "Historical comparison")
    rows = [
        ("Question", result.get("question") or h.placeholder(NOT_IN_EXPORT)),
        ("Period", f"{fmt.chicago_long(period.get('start_utc'))} – "
                   f"{fmt.chicago_long(period.get('cutoff_utc'))}"),
        ("Position sizes", _plain_product((result.get("settings") or {}).get("size_text") or "")
         or h.placeholder(NOT_IN_EXPORT)),
        ("Configurations", f"{completed:,} of {fmt.count(requested, 'configuration')} "
                           "completed"),
        ("Kind of study", purpose),
    ]
    return h.card(h.Markup(f"{h.kv_table(rows)}{h.note(FUTURE_NOTE)}"),
                  title="About this comparison")


# ── Streamlit ─────────────────────────────────────────────────────────────


def _repo_root(ctx) -> str:
    root = (ctx.roots or {}).get("repo_root") if isinstance(ctx.roots, dict) else None
    return str(Path(root) if root else Path(__file__).resolve().parents[1])


# ── publish the review folder again (the earlier study page's owner action) ─


PUBLISH_LABEL = "Publish the review folder again"
PUBLISH_WARNING = "The review folder could not be published; the verified result is kept."
_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class PublishAgain:
    """The arguments the earlier study page passed to ``publish_comparison_review``."""

    plan_id: str
    store_root: Path
    state_root: Path
    reports_root: Path


def _target_app_roots(target: dict[str, Any], roots: dict[str, Any] | None) -> dict[str, Any]:
    """The roots of the application that holds the open result (the running one or the other)."""

    from ifvg_lab_nav import app_roots, current_app

    roots = dict(roots or {})
    app = str(target.get("app") or current_app(roots))
    if roots and current_app(roots) == app:
        return roots
    repo = roots.get("repo_root") or Path(__file__).resolve().parents[1]
    return dict(app_roots(Path(repo)).get(app) or {})


def publish_again_route(target: dict[str, Any], study, roots: dict[str, Any] | None,
                        store_root: Path | str | None = None) -> PublishAgain | None:
    """This result's run state, when it records a review folder that failed to publish.

    The state is ``funded_comparison_jobs/<plan id>/state.json`` of the result's own
    application, located as the earlier study page does (``comparison_state_root``);
    it is offered only when the state names this exact result, has no review folder
    and records a ``review_error`` (the earlier page's condition). Read only.
    """

    from ifvg_funded_comparison_study import comparison_state_root

    from alpha_lab.propsim.funded.runner import read_state

    app = _target_app_roots(target, roots)
    if not (app.get("funded_comparison_state_root") or app.get("state_root")):
        return None
    result_id = str(target.get("result_id") or getattr(study, "result_id", "") or "")
    state_root = comparison_state_root(app)
    candidates = dict.fromkeys(str(p) for p in (target.get("plan_id"),
                                                 getattr(study, "plan_id", None)) if p)
    for plan_id in candidates:
        if not _SAFE_ID.fullmatch(plan_id):
            continue
        try:
            state = read_state(state_root, plan_id) or {}
        except (OSError, ValueError):
            continue
        if str(state.get("result_id") or "") != result_id or not result_id:
            continue
        if state.get("review_folder") or not state.get("review_error"):
            return None
        repo = app.get("repo_root") or (roots or {}).get("repo_root") \
            or Path(__file__).resolve().parents[1]
        return PublishAgain(
            plan_id=plan_id,
            store_root=Path(store_root or target.get("store_root") or app["store_root"]),
            state_root=Path(state_root),
            reports_root=Path(app.get("reports_root") or Path(repo) / "reports"))
    return None


def publish_again(st_module, ctx) -> None:
    """The earlier page's warning and "Publish the review folder again" (owner action)."""

    from ifvg_lab_ui import show

    route = publish_again_route(ctx.target, ctx.study, ctx.roots, ctx.store_root)
    if route is None:
        return
    show(h.alert(PUBLISH_WARNING,
                 "Publishing again writes a new review folder for this saved result in the "
                 "reports folder under “Funded comparison reviews”. The result and its figures "
                 "are not recomputed."), st_module)
    if not st_module.button(
            PUBLISH_LABEL, key=f"{KEY}publish_review_{ctx.result_id[:16]}",
            help="Try again to publish the full-study review folder of this saved result "
                 "(every configuration and both firms). Nothing is run or recomputed."):
        return
    from alpha_lab.propsim.funded.comparison_runner import publish_comparison_review

    try:
        publish_comparison_review(plan_id=route.plan_id, store_root=route.store_root,
                                  state_root=route.state_root, reports_root=route.reports_root)
    except Exception as error:  # e.g. an incomplete run, which can't be published
        st_module.error(f"The review folder was not published: {error}")
        return
    _cached_review.clear()
    _cached_view.clear()
    st_module.rerun()


@st.cache_resource(show_spinner="Reading the saved settings and evidence…", max_entries=32,
                   ttl=600)
def _cached_view(store_root: str, result_id: str, configuration: str, firm_key: str, firm: str,
                 repo_root: str) -> SettingsView:
    from ifvg_lab_ui import funded_study

    return build_settings_view(funded_study(store_root, result_id), configuration, firm_key,
                               firm, repo_root)


@st.cache_resource(show_spinner=False, max_entries=4, ttl=600)  # a new export shows
def _cached_review(repo_root: str, result_id: str) -> ReviewFolder | None:
    return latest_review_folder(repo_root, result_id)


def header_right(st_module, ctx) -> None:
    """Mock 08: the review-folder download in place of the firm switch."""

    review = _cached_review(_repo_root(ctx), ctx.result_id)
    if review is None:
        st_module.button(DOWNLOAD_LABEL, disabled=True, key=f"{KEY}review_folder_none",
                         width="stretch",
                         help="No review folder has been published for this result yet, so "
                              "there is nothing to download.")
        return
    size = review.size_bytes
    size_text = f", {size / 1_048_576:.1f} MB zip" if size else ", zipped when you download it"
    published = (f"published {fmt.chicago_long(review.published_utc)}"
                 if review.published_utc else "publication time not recorded")
    st_module.download_button(
        DOWNLOAD_LABEL, data=lambda: folder_zip_bytes(review), file_name=f"{review.name}.zip",
        mime="application/zip", on_click="ignore", key=f"{KEY}review_folder_{review.version}",
        width="stretch",
        help=(f"The published review folder for this result: export version {review.version}, "
              f"{published}{size_text}. Every configuration and both firms, as published."))


def render(st_module, ctx) -> None:
    from ifvg_lab_ui import show

    view = _cached_view(ctx.store_root, ctx.result_id, ctx.configuration, ctx.firm_key,
                        ctx.firm, _repo_root(ctx))
    publish_again(st_module, ctx)
    show(top_cards(view), st_module)
    show(limitations_card(view), st_module)
    show(decisions_card(view), st_module)
    show(h.section_title("More", right="Earlier features kept here"), st_module)
    from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
        STRATEGY_METRICS_NOTE,
    )

    note = fmt.display_words(STRATEGY_METRICS_NOTE)
    show(h.card(h.Markup(f"{strategy_table(ctx.study, ctx.configuration)}"
                         f'<div class="lab-line">{h.esc(note)} This '
                         "configuration's row is tinted.</div>"),
                title="Strategy measures without accounts",
                right=f"All {len(ctx.study.configurations)} configurations"), st_module)
    show(about_card(ctx.study.result), st_module)
