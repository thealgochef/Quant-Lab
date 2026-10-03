"""My studies (mock 01): one list of every study from BOTH applications' stores.

Pure view model (no Streamlit): turns the existing study listings
(``presentation.workspace.load_studies`` rows of each application and the
feature-and-model research groups) into library rows with a tab, a one-line
"what it tested", readable dates, a status with its Chicago date, and the ONE
action each row offers. Only two action labels exist (SCREENS.md):
:data:`OPEN_RESULTS` and :data:`CONTINUE_DRAFT`.

Funded leaders are read from each saved result: the rank-1 configuration's net
cash at EACH firm, one firm per column. Firms are never added together.
Nothing here writes a store, launches work or recomputes a stored money figure.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt

__all__ = [
    "APP_COMMANDS",
    "APP_NAMES",
    "CONTINUE_DRAFT",
    "EARLIER_METHOD",
    "FIRM_COLUMNS",
    "FirmLeader",
    "LibraryRow",
    "OPEN_RESULTS",
    "PAGE_SIZE",
    "SORTS",
    "TABS",
    "TAB_LABELS",
    "chicago_date",
    "comparison_draft_note",
    "counts",
    "draft_note",
    "draft_summary",
    "earlier_leaders",
    "filter_rows",
    "firm_columns",
    "funded_leaders",
    "group_row",
    "leader_for",
    "other_app_note",
    "page",
    "plan_tested_line",
    "readable_dates",
    "rows_in_tab",
    "sort_options",
    "sort_rows",
    "study_row",
    "tab_label",
]

OPEN_RESULTS = "Open results"
CONTINUE_DRAFT = "Continue draft"
EARLIER_METHOD = "Earlier method"
#: what the earlier five-account mode tested (its own screen says the same)
EARLIER_TESTED = ("Five accounts copying one signal · checked the simulator, not a strategy "
                  "result")
PAGE_SIZE = 10
SCOPE_UNRESOLVED = "Scope unresolved"

#: the two applications (``ifvg_lab_nav.app_roots``) in plain words, and how to start each
APP_NAMES = {"main": "main app", "ifsm": "dedicated app"}
APP_COMMANDS = {"main": "streamlit run scripts/dashboard.py",
                "ifsm": "python scripts/run_ifsm_research_ui.py"}

TABS = ("funded", "strategy", "model", "drafts", "archived")
TAB_LABELS = {"funded": "Funded comparisons", "strategy": "Strategy studies",
              "model": "Model studies", "drafts": "Drafts", "archived": "Archived"}
#: tabs whose label carries a count (as drawn in mock 01)
_COUNTED = ("funded", "model", "drafts")

SORTS = ("Newest first", "Oldest first", "Name")
_LEADER_SORT = "Highest leader net cash, "
#: the leader columns of the funded table when no saved result names its firms
FIRM_COLUMNS = (("takeprofittrader", "TakeProfitTrader"), ("myfundedfutures", "MyFundedFutures"))

COMPARISON_MODE = "funded_configuration_comparison"
FUNDED_MODE = "funded_payout_simulation"
_FUNDED_KINDS = ("funded_comparison", "funded")
_RESULT_STATUSES = ("Completed", "Incomplete", "Failed")

_KIND_LABELS = {"search": "Search", "pipeline": "Workflow", "context": "Context features",
                "funded": EARLIER_METHOD}
_DRAFT_TYPES = {
    ("single_configuration", "compare_one_with_baseline"):
        "One configuration compared with the baseline",
    ("single_configuration", None): "One configuration",
    ("fsm_config_search", None): "Search across settings",
    ("prop_benchmark", None): "Prop feasibility",
    ("universal_prop_search", None): "Strategy across firms",
    ("full_pipeline_run", None): "Full workflow",
    (FUNDED_MODE, None): "Five accounts copying one signal",
}
_GROUP_STATUS = {"completed": "Completed", "running": "Running", "failed": "Failed",
                 "cancelled_at_safe_boundary": "Interrupted", "cancel_requested": "Stopping",
                 "frozen": "Ready", "blocked": "Blocked"}
_INSTRUMENT_WORDS = {"mini": ("E-mini", "E-minis"), "micro": ("Micro E-mini", "Micro E-minis")}
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


# ── firm leaders from saved results ───────────────────────────────────────


@dataclass(frozen=True)
class FirmLeader:
    """The rank-1 configuration at ONE firm of one saved result (never a total)."""

    firm_key: str
    firm: str
    net_cash_cents: int | None  # None: no configuration completed at this firm
    configuration: str | None = None


def funded_leaders(result: Mapping[str, Any]) -> tuple[FirmLeader, ...]:
    """Each firm's leader of a funded configuration comparison, in saved firm order.

    Uses the same saved rank order as the funded results overview
    (``FundedStudy.completed_at``); the value is the stored net cash in cents.
    """

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import study_from_result

    study = study_from_result(dict(result), result_id="library")
    out = []
    for firm_key, firm in study.firms:
        rows = study.completed_at(firm_key)
        top = rows[0] if rows else None
        out.append(FirmLeader(
            firm_key=firm_key, firm=firm,
            net_cash_cents=int(top["net_cash_earned_cents"]) if top else None,
            configuration=str(top["configuration"]) if top else None))
    return tuple(out)


def earlier_leaders(result: Mapping[str, Any]) -> tuple[FirmLeader, ...]:
    """Net cash per firm of the earlier five-account result (one operation per firm)."""

    summaries = result.get("summaries_cents") or {}
    tables = result.get("tables") or {}
    order = [row.get("firm_key") for row in tables.get("instance_results") or []]
    order = [key for key in dict.fromkeys([*order, *summaries]) if key in summaries]
    out = []
    for key in order:
        summary = summaries[key]
        value = summary.get("net_cash_earned_cents")
        out.append(FirmLeader(firm_key=str(key), firm=str(summary.get("firm") or key),
                              net_cash_cents=int(value) if value is not None else None))
    return tuple(out)


# ── words ─────────────────────────────────────────────────────────────────


def chicago_date(value: Any) -> str:
    """``September 23, 2026`` — the Chicago calendar date of a stored instant ('' if none)."""

    if not value:
        return ""
    try:
        from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import (
            CHICAGO,
            utc_instant,
        )

        instant = utc_instant(value)
    except Exception:  # a zone-less or unreadable time is not guessed
        return ""
    if instant is None:
        return ""
    return fmt.date_long(instant.tz_convert(CHICAGO).date())


_ISO_RANGE = re.compile(r"^(\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2}) · (\d+) days$")
_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LONG_RANGE = re.compile(r"^([A-Z][a-z]+ \d{1,2}, \d{4}) – ([A-Z][a-z]+ \d{1,2}, \d{4})$")


def readable_dates(text: str) -> str:
    """The listing's date text in the redesign's words (unknown text is kept as saved)."""

    from datetime import datetime

    value = str(text or "").strip()
    if not value or value == "Dates not selected":
        return "Dates not chosen yet"
    match = _ISO_RANGE.match(value)
    if match:
        first, last, days = match.groups()
        return f"{fmt.date_range(first, last)} · {fmt.count(int(days), 'trading day')}"
    if _ISO_DAY.match(value):
        return fmt.date_long(value)
    match = _LONG_RANGE.match(value)
    if match:
        try:
            first = datetime.strptime(match.group(1), "%B %d, %Y").date()
            last = datetime.strptime(match.group(2), "%B %d, %Y").date()
        except ValueError:
            return value
        return fmt.date_range(first, last)
    return value


def _quantity_words(quantity: int, instrument: str) -> str:
    singular, plural = _INSTRUMENT_WORDS.get(instrument, (instrument, f"{instrument}s"))
    count = _NUMBER_WORDS.get(quantity, f"{quantity:,}")
    return f"{count} {singular if quantity == 1 else plural}"


def plan_tested_line(plan: Any) -> str | None:
    """What a funded comparison plan tested, in one line (None when unreadable)."""

    if plan is None:
        return None
    variants = getattr(plan, "variants", None)
    if variants is not None:  # version-2 variation plan
        base = getattr(plan, "base_configuration", None)
        parts = [fmt.count(len(variants), "configuration")
                 + (f" around {base}" if base else "")]
        halves = sum(1 for v in variants if getattr(v, "exit_policy", "") != "fixed_target_v1")
        parts.append("includes the half exit" if halves else "whole-position exits")
        return " · ".join(parts)
    configurations = getattr(plan, "configurations", None)
    if configurations is None:
        return None
    parts = [fmt.count(len(configurations), "configuration")]
    quantity, instrument = getattr(plan, "quantity", None), getattr(plan, "instrument", None)
    if quantity and instrument:
        parts.append(_quantity_words(int(quantity), str(instrument)))
    parts.append("whole-position exits")
    return " · ".join(parts)


def other_app_note(app: str) -> str:
    """The one sentence shown instead of an action for a study this app can't open."""

    return f"Open it in the {APP_NAMES.get(app, app)}: {APP_COMMANDS.get(app, '')}".strip()


def tab_label(tab: str, count: int | None = None) -> str:
    label = TAB_LABELS[tab]
    return f"{label} · {count:,}" if tab in _COUNTED and count is not None else label


# ── drafts ────────────────────────────────────────────────────────────────


def _firms_words(firm_keys: Sequence[str]) -> str:
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    keys = list(firm_keys)
    if len(keys) == 2:
        return "both firms"
    if len(keys) == 1:
        profile = FIRM_PROFILES.get(keys[0])
        return profile.firm_name if profile else str(keys[0])
    return f"{len(keys)} firms" if keys else "no firm chosen"


def _fixed_values(draft: Any) -> list[str]:
    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import axis_value_name

    fixed = (draft.steps.get("baseline", {}) or {}).get("fixed_axis_value_ids") or {}
    out = []
    for value in fixed.values():
        if value:
            text = axis_value_name(str(value)).removesuffix(" (default)")
            if text and text != "Unavailable setting":
                out.append(text)
    return out


def draft_summary(draft: Any, check: Any = None) -> str:
    """One line: what the draft will test (configurations · firms, or its study type)."""

    if draft is None:
        return "Saved draft"
    if draft.mode_id == COMPARISON_MODE:
        from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_setup import plan_count
        from alpha_lab.propsim.funded.comparison_draft import DEFAULT_SETTINGS, saved_settings

        settings = {**DEFAULT_SETTINGS, **saved_settings(draft)}
        # the same count as New funded comparison and Review and approve (gap rules and
        # withdrawal triggers included)
        first = plan_count(check, draft).text()
        return f"{first} · {_firms_words(settings.get('firm_keys') or [])}"
    question = (draft.steps.get("objective", {}) or {}).get("question_id")
    label = (_DRAFT_TYPES.get((draft.mode_id, question))
             or _DRAFT_TYPES.get((draft.mode_id, None)))
    if label is None:
        return "Saved draft"
    parts = [label]
    if draft.mode_id == "single_configuration":
        parts.extend(_fixed_values(draft))
    elif draft.mode_id in ("fsm_config_search", "universal_prop_search"):
        chosen = (draft.steps.get("search_space", {}) or {}).get("axis_selections") or {}
        varied = sum(1 for values in chosen.values() if values)
        if varied:
            parts.append(f"{fmt.count(varied, 'setting')} varied")
    return " · ".join(parts)


def comparison_draft_note(check: Any, blocked: str | None = None) -> tuple[str, str]:
    """(note, tone) for a funded-comparison draft from ``check_saved_comparison``.

    ``blocked`` is ``funded_setup.approval_blocked_note`` of the draft's saved
    blockers ("Approval blocked: 2 settings need engine support.").
    """

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.format import display_words

    lead = f"{blocked} " if blocked else ""
    if check is None:
        return (f"{lead}Its saved settings could not be checked here. They are unchanged.",
                "orange")
    if getattr(check, "needs_half_exit_engine", False):
        return (f"Needs the half-exit engine to run. {lead}Your saved settings are unchanged.",
                "orange")
    problems = list(getattr(check, "problems", ()) or ())
    if problems:
        return (f"Can't run in this application: {display_words(problems[0])} {lead}Your saved "
                "settings are unchanged."), "orange"
    if blocked:
        return f"{blocked} Your saved settings are unchanged.", "orange"
    return "Settings saved. Review and approve the plan to run it.", ""


def draft_note(row_status: str, scope: str, dates: str, draft: Any = None) -> tuple[str, str]:
    """(note, tone) for a strategy draft, from its saved record only."""

    if scope == "unresolved":
        return ("Its research scope is unresolved. Resolve it on the study page before "
                "running."), "orange"
    if row_status == "Ready":
        return "Saved and checked. Run it from its study page.", ""
    if row_status == "Evidence unavailable":
        return "Its saved progress is unavailable. Your saved settings are unchanged.", "orange"
    if draft is not None and draft.mode_id == FUNDED_MODE:
        return "Earlier five-account method. Settings saved, not run.", ""
    readable = readable_dates(dates)
    if readable == "Dates not chosen yet" or dates.startswith("Dates of the chosen"):
        return "Dates not chosen yet.", ""
    return f"{readable}. Not run yet.", ""


# ── rows ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LibraryRow:
    """One line of My studies: a study, a draft or a feature-and-model research group."""

    key: str
    app: str
    kind: str  # StudySummary.kind, or "research_group"
    tab: str  # funded / strategy / model / drafts (archived rows keep their home tab)
    name: str
    tested: str
    dates: str
    status: str
    status_date: str = ""
    archived: bool = False
    earlier_method: bool = False
    funded_draft: bool = False  # a draft that belongs under Funded comparisons
    kind_label: str = ""
    action: str = OPEN_RESULTS
    #: where the action goes: funded_results / new_funded / detail / research_group / ""
    route: str = ""
    #: where the name goes (the earlier "Details and actions" route)
    details_route: str = ""
    note: str = ""
    note_tone: str = ""
    #: shown instead of the action when this application can't open the row
    elsewhere: str = ""
    updated: str = ""
    search_text: str = ""
    result_id: str | None = None
    plan_id: str | None = None
    leaders: tuple[FirmLeader, ...] = ()
    leaders_problem: str | None = None
    study: Any = field(default=None, compare=False, repr=False)

    @property
    def openable(self) -> bool:
        return bool(self.route)

    def with_search_text(self) -> LibraryRow:
        parts = [self.name, self.tested, self.dates, self.status, self.status_date,
                 self.note, self.kind_label, APP_NAMES.get(self.app, self.app),
                 getattr(self.study, "question", "") or "",
                 *(leader.configuration or "" for leader in self.leaders)]
        return replace(self, search_text=" ".join(p for p in parts if p).lower())


def _title(name: str) -> str:
    """A funded study's name without the description the saved title appends."""

    return name.split(" — ", 1)[0].strip() or name


def study_row(study: Any, app: str, *, current: str, plan: Any = None,
              tested_line: str | None = None, leaders: Sequence[FirmLeader] = (),
              leaders_problem: str | None = None, draft_check: Any = None,
              check_available: bool = True, draft_blocked: str | None = None) -> LibraryRow:
    """The library row of one ``load_studies`` row of application ``app``.

    ``current`` is the application this page runs in; a study of the other
    application can be opened here only when it is a saved funded result.
    """

    here = app == current
    state = study.state or {}
    kind = study.kind
    draft = getattr(study, "draft", None)
    mode = getattr(draft, "mode_id", "") if draft is not None else ""
    status_date = chicago_date(state.get("completed_at_utc") or study.updated)
    base = {"key": str(study.key), "app": app, "kind": kind, "archived": bool(study.archived),
            "updated": str(study.updated or ""), "study": study,
            "status": str(study.status)}
    if kind in _FUNDED_KINDS:
        earlier = kind == "funded"
        result_id = state.get("result_id")
        has_result = bool(result_id) and study.status in _RESULT_STATUSES
        if earlier:
            tested = EARLIER_TESTED
            route = "detail" if here else ""
        else:
            tested = tested_line or plan_tested_line(plan) or study.question
            route = ("funded_results" if has_result
                     else "detail" if here else "")
        return LibraryRow(
            **base, tab="funded", name=_title(study.name), tested=tested,
            dates=readable_dates(study.dates), status_date=status_date,
            earlier_method=earlier, kind_label=EARLIER_METHOD if earlier else "",
            action=OPEN_RESULTS, route=route, details_route=route,
            result_id=str(result_id) if result_id else None,
            plan_id=str(state.get("plan_id") or study.key),
            leaders=tuple(leaders), leaders_problem=leaders_problem,
            elsewhere="" if route else other_app_note(app)).with_search_text()
    if kind == "draft":
        funded_draft = mode in (COMPARISON_MODE, FUNDED_MODE)
        is_open_draft = (study.status == "Draft" and not study.archived
                         and study.scope != "unresolved")
        if mode == COMPARISON_MODE and not study.archived:
            if not here:
                note, tone = f"Saved in the {APP_NAMES.get(app, app)}.", ""
                if draft_blocked:
                    note, tone = f"{note} {draft_blocked}", "orange"
            elif check_available:
                note, tone = comparison_draft_note(draft_check, draft_blocked)
            else:
                note, tone = comparison_draft_note(None, draft_blocked)
        else:
            note, tone = draft_note(study.status, study.scope, study.dates, draft)
        if not here:
            route = details = ""
        elif mode == COMPARISON_MODE and is_open_draft:
            route, details = "new_funded", "detail"
        else:
            route = details = "detail"
        return LibraryRow(
            **base, tab="drafts", name=study.name, tested=draft_summary(draft, draft_check),
            dates=readable_dates(study.dates), status_date=chicago_date(study.updated),
            funded_draft=funded_draft, earlier_method=mode == FUNDED_MODE,
            kind_label=EARLIER_METHOD if mode == FUNDED_MODE else "",
            action=CONTINUE_DRAFT, route=route, details_route=details,
            note=note, note_tone=tone, elsewhere="" if route else other_app_note(app),
        ).with_search_text()
    route = "detail" if here else ""
    unresolved = study.scope == "unresolved"
    return LibraryRow(
        **base, tab="strategy", name=study.name,
        tested=study.question, dates=readable_dates(study.dates), status_date=status_date,
        kind_label=_KIND_LABELS.get(kind, ""), action=OPEN_RESULTS, route=route,
        details_route=route, elsewhere="" if route else other_app_note(app),
        note=SCOPE_UNRESOLVED if unresolved else "", note_tone="orange" if unresolved else "",
    ).with_search_text()


def _group_updated(group: Mapping[str, Any]) -> str:
    stamps = []
    for cell in group.get("cells") or ():
        state = cell.get("state") or {}
        for attempt in state.get("attempts") or ():
            if attempt.get("ended_at"):
                stamps.append(str(attempt["ended_at"]))
        if state.get("updated_at_utc"):
            stamps.append(str(state["updated_at_utc"]))
    return max(stamps) if stamps else ""


def group_row(group: Mapping[str, Any], app: str, *, current: str) -> LibraryRow:
    """A feature-and-model research group (the earlier "Feature and model studies" card)."""

    spec = group.get("study_spec") or {}
    cells = list(group.get("cells") or ())
    labels = list(dict.fromkeys(str(c.get("label")) for c in cells if c.get("label")))
    lanes = list(dict.fromkeys(str(c.get("lane")) for c in cells if c.get("lane")))
    tested = " · ".join([*labels[:3], *lanes[:2]]) or "Feature and model study"
    if len(labels) > 3:
        tested += f" · {len(labels) - 3} more"
    first, last = spec.get("evaluation_start"), spec.get("evaluation_end")
    dates = fmt.date_range(first, last) if first and last else "Dates not recorded"
    raw = str(group.get("status") or "")
    status = _GROUP_STATUS.get(raw, raw.replace("_", " ").capitalize() or
                               "Saved; progress not recorded")
    updated = _group_updated(group)
    here = app == current
    return LibraryRow(
        key=str(group.get("group_id")), app=app, kind="research_group", tab="model",
        name=str(group.get("display_name") or "Feature and model study"), tested=tested,
        dates=dates, status=status, status_date=chicago_date(updated), updated=updated,
        kind_label="Feature and model", action=OPEN_RESULTS,
        route="research_group" if here else "", details_route="research_group" if here else "",
        elsewhere="" if here else other_app_note(app), study=group,
    ).with_search_text()


# ── tabs, filters, sorting, paging ────────────────────────────────────────


def rows_in_tab(rows: Iterable[LibraryRow], tab: str) -> list[LibraryRow]:
    """Rows of one tab. Archived rows appear only under Archived."""

    rows = list(rows)
    if tab == "archived":
        return [r for r in rows if r.archived]
    live = [r for r in rows if not r.archived]
    if tab == "funded":
        return [r for r in live if r.tab == "funded" or (r.tab == "drafts" and r.funded_draft)]
    return [r for r in live if r.tab == tab]


def counts(rows: Iterable[LibraryRow]) -> dict[str, int]:
    """Tab counts: funded studies (not their drafts), model studies, drafts."""

    rows = list(rows)
    live = [r for r in rows if not r.archived]
    return {"funded": sum(1 for r in live if r.tab == "funded"),
            "strategy": sum(1 for r in live if r.tab == "strategy"),
            "model": sum(1 for r in live if r.tab == "model"),
            "drafts": sum(1 for r in live if r.tab == "drafts"),
            "archived": sum(1 for r in rows if r.archived)}


def filter_rows(rows: Iterable[LibraryRow], query: str = "",
                statuses: Sequence[str] = ()) -> list[LibraryRow]:
    words = [w for w in str(query or "").lower().split() if w]
    chosen = set(statuses or ())
    return [r for r in rows
            if all(w in r.search_text for w in words)
            and (not chosen or r.status in chosen)]


def firm_columns(rows: Iterable[LibraryRow]) -> tuple[tuple[str, str], ...]:
    """The funded table's leader columns: every firm any listed result names, in order."""

    seen: dict[str, str] = {}
    for row in rows:
        for leader in row.leaders:
            seen.setdefault(leader.firm_key, leader.firm)
    return tuple(seen.items()) or FIRM_COLUMNS


def sort_options(tab: str, firms: Sequence[tuple[str, str]] = ()) -> tuple[str, ...]:
    if tab in ("funded", "archived"):
        return (*SORTS, *(f"{_LEADER_SORT}{name}" for _, name in firms))
    return SORTS


def leader_for(row: LibraryRow, firm_key: str) -> FirmLeader | None:
    return next((leader for leader in row.leaders if leader.firm_key == firm_key), None)


def sort_rows(rows: Iterable[LibraryRow], sort: str,
              firms: Sequence[tuple[str, str]] = ()) -> list[LibraryRow]:
    """Newest / oldest / name, or one firm's leader net cash (never a cross-firm total).

    Sorting by a firm's leader puts current-method results first, then the
    earlier five-account results (a different measure), then rows without a
    value at that firm.
    """

    rows = list(rows)
    if sort == "Oldest first":
        return sorted(rows, key=lambda r: (not r.updated, r.updated))
    if sort == "Name":
        return sorted(rows, key=lambda r: r.name.lower())
    if sort.startswith(_LEADER_SORT):
        name = sort[len(_LEADER_SORT):]
        firm_key = next((k for k, n in firms if n == name), name)

        def key(row: LibraryRow):
            leader = leader_for(row, firm_key)
            value = leader.net_cash_cents if leader else None
            return (value is None, row.earlier_method, -(value or 0))

        return sorted(rows, key=key)
    return sorted(rows, key=lambda r: r.updated, reverse=True)


def page(rows: Sequence[LibraryRow], number: int, size: int = PAGE_SIZE
         ) -> tuple[list[LibraryRow], int]:
    """(rows of page ``number`` (1-based), page count)."""

    pages = max(1, (len(rows) + size - 1) // size)
    number = min(max(1, int(number)), pages)
    return list(rows[(number - 1) * size: number * size]), pages
