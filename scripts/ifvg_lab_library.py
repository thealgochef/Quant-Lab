"""My studies (mock 01) — every study from both applications' stores, in one list.

Tabs by study type (Funded comparisons, Strategy studies, Model studies,
Drafts, Archived; only the chosen tab renders), search, a status filter,
sorting and paging. Each row offers ONE of the two actions the redesign
allows: "Open results" or "Continue draft". A row's name opens its study page
(the earlier "Details and actions" route: progress, run/resume and the study
actions clone, rename, archive/restore and the corrected-morning copy).

Both applications' stores are read (``ifvg_lab_nav.app_roots``), read only. A
saved funded result from the other application opens read-only on the funded
results overview; any other study of the other application names the command
that opens it. Funded leaders are the rank-1 net cash at EACH firm from the
saved result — one firm per column, never added together.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab import html as h
from alpha_lab.agents.data_infra.ifvg.presentation.lab import library as lib
from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import css_var

__all__ = ["library_rows", "render_library", "sources_signature"]

_P = "ifvg_lab_v1_library_"
_TAB_HELP = ("Choose which kind of study to list. Funded comparisons also lists their "
             "drafts; Archived lists every archived study and draft.")

_CSS = """
.st-key-ifvg_lab_libtabs { border-bottom: 1px solid var(--lab-rule); gap: 4px !important;
  flex-wrap: wrap; }
.stApp [data-testid="stMain"] .st-key-ifvg_lab_libtabs
  button[data-testid="stBaseButton-tertiary"] {
  text-decoration: none; padding: 12px 16px !important; min-height: 44px; border-radius: 0;
  border-bottom: 3px solid transparent; margin-bottom: -1px; color: var(--lab-body-2); }
.st-key-ifvg_lab_libtabs button p { font-size: 15px !important; color: var(--lab-body-2);
  white-space: nowrap; }
.stApp [data-testid="stMain"] [class*="st-key-ifvg_lab_libname_"]
  button[data-testid="stBaseButton-tertiary"] {
  color: var(--lab-ink); text-decoration: none; padding: 0 !important; min-height: 0;
  justify-content: flex-start; text-align: left; }
[class*="st-key-ifvg_lab_libname_"] { flex-wrap: wrap; row-gap: 4px; }
[class*="st-key-ifvg_lab_libname_"] button p { font-size: 16px !important; font-weight: 600;
  color: var(--lab-ink); line-height: 1.35; text-align: left; white-space: normal; }
[class*="st-key-ifvg_lab_libname_"] button:hover p { text-decoration: underline; }
.st-key-ifvg_workspace_new button { min-height: 48px; padding: 12px 18px; font-size: 15px; }
[class*="st-key-ifvg_lab_libtbl_"] { background: var(--lab-panel);
  border: 1px solid var(--lab-rule); border-radius: 12px; overflow: hidden; gap: 0 !important; }
[class*="st-key-ifvg_lab_libhead_"] { background: var(--lab-header-row); padding: 12px 24px; }
[class*="st-key-ifvg_lab_librow_"] { padding: 16px 24px;
  border-top: 1px solid var(--lab-light-rule); }
[class*="st-key-ifvg_lab_libhead_"] [data-testid="stVerticalBlock"],
[class*="st-key-ifvg_lab_librow_"] [data-testid="stVerticalBlock"] { gap: 2px; }
[class*="st-key-ifvg_lab_librow_"] [data-testid="stColumn"]:last-child
  [data-testid="stVerticalBlock"] { align-items: flex-end; }
[class*="st-key-ifvg_lab_librow_"] [data-testid="stColumn"]:last-child button {
  min-height: 44px; padding: 10px 14px; white-space: nowrap; }
[class*="st-key-ifvg_lab_librow_"] [data-testid="stColumn"]:last-child button p {
  font-size: 14px !important; font-weight: 500; }
.st-key-ifvg_lab_v1_library_search [data-baseweb="input"] { min-height: 44px;
  border: 1px solid var(--lab-control-border) !important; border-radius: 8px;
  background: var(--lab-panel); }
.st-key-ifvg_lab_v1_library_search input { font-size: 15px; }
.lab-lib-head { font-size: 13px; font-weight: 600; color: var(--lab-body-2); }
.lab-lib-head.num { text-align: right; }
.lab-lib-name { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.lab-lib-name a, .lab-lib-name span.nm { font-size: 16px; font-weight: 600; color: var(--lab-ink);
  text-decoration: none; line-height: 1.35; }
.lab-lib-name a:hover { text-decoration: underline; }
.lab-lib-tag { padding: 2px 8px; border-radius: 999px; background: var(--lab-header-row);
  color: var(--lab-body-2); font-size: 12px; font-weight: 600; white-space: nowrap; }
.lab-lib-tag.app { background: var(--lab-blue-light); color: var(--lab-blue-dark); }
.lab-lib-tested { font-size: 14px; color: var(--lab-body-2); line-height: 1.45; margin-top: 4px; }
.lab-lib-dates { font-size: 13px; color: var(--lab-muted); margin-top: 4px; }
.lab-lib-status { font-size: 14px; font-weight: 500; color: var(--lab-ink); }
.lab-lib-sub { font-size: 13px; color: var(--lab-muted); }
.lab-lib-sub.orange { color: var(--lab-orange-dark); }
.lab-lib-money { text-align: right; font-family: var(--lab-mono); font-size: 15px;
  color: var(--lab-ink); }
.lab-lib-money.grey { color: var(--lab-muted); }
.lab-lib-note { padding: 10px 12px; background: var(--lab-soft-panel); border-radius: 8px;
  font-size: 14px; line-height: 1.4; color: var(--lab-body); }
.lab-lib-note.orange { background: var(--lab-orange-light); color: var(--lab-orange-dark); }
.lab-lib-elsewhere { font-size: 13px; line-height: 1.45; color: var(--lab-muted); margin-top: 6px; }
.lab-lib-elsewhere code { font-family: var(--lab-mono); font-size: 12px; color: var(--lab-body);
  background: transparent; padding: 0; }
[class*="st-key-ifvg_lab_libname_"] > div { flex: 0 1 auto !important; width: auto !important;
  min-width: 0; }
.lab-lib-sortlabel { font-size: 14px; color: var(--lab-body-2); text-align: right; }
"""

#: column weights (mock 01 grid: 1fr 190px 170px 170px 150px; drafts 1fr 380px 150px)
_FUNDED_SPEC = [2.75, 1.3, 1.35, 1.35, 1.05]
_STUDY_SPEC = [3.6, 1.5, 1.2]
_DRAFT_SPEC = [2.7, 2.5, 1.05]


# ── cached reads (saved records are immutable; drafts keyed by their save time) ─


@st.cache_data(show_spinner=False, ttl=30, max_entries=4)
def _other_app_studies(signature: str, _roots: dict[str, Any]):
    """The other application's listing (read only), refreshed at most every 30 seconds."""

    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import load_studies

    return load_studies(_roots)


@st.cache_data(show_spinner=False, max_entries=64)
def _leaders(store_root: str, result_id: str, kind: str) -> tuple[lib.FirmLeader, ...]:
    """Each firm's leader from one saved result (verified on load; errors are not cached)."""

    if kind == "funded":
        from alpha_lab.propsim.funded.runner import load_result

        return lib.earlier_leaders(load_result(Path(store_root), result_id))
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import open_funded_study

    return lib.funded_leaders(open_funded_study(Path(store_root), result_id).result)


@st.cache_data(show_spinner=False, max_entries=64)
def _plan_line(store_root: str, plan_id: str) -> str | None:
    from alpha_lab.propsim.funded.comparison_runner import load_plan

    return lib.plan_tested_line(load_plan(Path(store_root), plan_id))


@st.cache_resource(show_spinner=False, ttl=900)
def _comparison_sources() -> dict[str, Any]:
    from alpha_lab.propsim.funded.comparison_source import discover_comparison_sources

    return {s.package.run_id: s for s in discover_comparison_sources()}


def sources_signature(sources: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Which verified strategy studies are on this computer (run id, manifest hash)."""

    return tuple(sorted((str(run_id), str(getattr(getattr(source, "package", None),
                                                  "manifest_sha256", "") or ""))
                        for run_id, source in sources.items()))


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_draft_check(draft_id: str, updated: str, sources: tuple[tuple[str, str], ...],
                        _draft: Any, _sources: dict[str, Any]):
    from alpha_lab.propsim.funded.comparison_draft import (
        check_saved_comparison,
        saved_settings,
        saved_study_selections,
    )

    return check_saved_comparison(saved_settings(_draft), saved_study_selections(_draft),
                                  _sources)


def _draft_check(draft_id: str, updated: str, draft: Any):
    """What this application's engine can do with one saved funded-comparison draft.

    Keyed by the draft's save time AND the verified studies found now (refreshed every
    15 minutes), so a study that appears or disappears changes the row's note.
    """

    sources = _comparison_sources()
    return _cached_draft_check(draft_id, updated, sources_signature(sources), draft, sources)


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_draft_blocked(draft_id: str, updated: str, sources: tuple[tuple[str, str], ...],
                          repo_root: str, _draft: Any, _sources: dict[str, Any]) -> str | None:
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_setup as fs
    from alpha_lab.propsim.funded.comparison_draft import saved_settings

    source = _sources.get(saved_settings(_draft).get("source_run_id"))
    _named_baseline, named_problem = fs.named_baseline()
    blockers = fs.saved_draft_blockers(_draft, source, repo_root=repo_root or None,
                                       named_problem=named_problem)
    return fs.approval_blocked_note(blockers)


def _draft_blocked(draft_id: str, updated: str, draft: Any, repo_root: Any) -> str | None:
    """"Approval blocked: …" for a saved draft, from the list Review and approve shows."""

    sources = _comparison_sources()
    return _cached_draft_blocked(draft_id, updated, sources_signature(sources),
                                 str(repo_root or ""), draft, sources)


# ── rows from both applications ───────────────────────────────────────────


def _exists(roots: dict[str, Any]) -> bool:
    keys = ("draft_root", "state_root", "store_root", "pipeline_state_root")
    return any(roots.get(k) and Path(roots[k]).exists() for k in keys)


def _groups(roots: dict[str, Any]) -> list[dict[str, Any]]:
    from alpha_lab.agents.data_infra.ifvg.search import research_runs

    state_root = roots.get("pipeline_state_root")
    if not state_root:
        from ifvg_pipeline_tab import PIPELINE_STATE_ROOT

        state_root = PIPELINE_STATE_ROOT
    return research_runs.list_research_groups(Path(roots["store_root"]), Path(state_root))


def library_rows(studies, roots) -> tuple[list[lib.LibraryRow], dict[str, dict[str, Any]],
                                          list[str]]:
    """(rows of both applications, each application's roots, one-line notices)."""

    from ifvg_lab_nav import app_roots, current_app

    here = current_app(roots)
    by_app: dict[str, dict[str, Any]] = {here: dict(roots)}
    entries = [(here, roots, list(studies))]
    notices: list[str] = []
    repo = roots.get("repo_root")
    other = "main" if here == "ifsm" else "ifsm"
    if repo:
        other_roots = app_roots(Path(repo))[other]
        if _exists(other_roots):
            by_app[other] = other_roots
            name = lib.APP_NAMES[other]
            try:
                signature = "|".join(f"{k}={v}" for k, v in sorted(other_roots.items()))
                other_studies, issues = _other_app_studies(signature, other_roots)
            except Exception as error:  # read-only listing; the page stays usable
                notices.append(f"The {name}'s studies could not be read, so they are not "
                               f"listed here ({type(error).__name__}).")
            else:
                entries.append((other, other_roots, list(other_studies)))
                notices.extend(f"From the {name}: {issue}" for issue in issues)
    rows: list[lib.LibraryRow] = []
    for app, app_roots_, app_studies in entries:
        rows.extend(lib.study_row(study, app, current=here) for study in app_studies)
        if not Path(app_roots_["store_root"]).joinpath("research_groups").is_dir():
            continue
        try:
            groups = _groups(app_roots_)
        except Exception:
            notices.append(f"Model studies of the {lib.APP_NAMES[app]} could not be verified. "
                           "Their saved work is retained.")
            continue
        rows.extend(lib.group_row(group, app, current=here) for group in groups)
    return rows, by_app, notices


def _complete(row: lib.LibraryRow, by_app: dict[str, dict[str, Any]], here: str
              ) -> lib.LibraryRow:
    """Add what only the shown rows need: plan line, firm leaders, the draft check."""

    study = row.study
    if row.kind == "research_group" or study is None:
        return row
    roots = by_app.get(row.app) or {}
    store = str(study.store_root or roots.get("store_root") or "")
    if row.kind in ("funded_comparison", "funded"):
        line = None
        leaders: tuple[lib.FirmLeader, ...] = ()
        problem = None
        if row.kind == "funded_comparison" and row.plan_id and store:
            try:
                line = _plan_line(store, row.plan_id)
            except Exception:  # the saved title's words stay
                line = None
        if row.result_id and study.status in ("Completed", "Incomplete", "Failed") and store:
            try:
                leaders = _leaders(store, row.result_id, row.kind)
            except Exception as error:
                problem = f"Saved result unreadable ({type(error).__name__})"
        return lib.study_row(study, row.app, current=here, tested_line=line,
                             leaders=leaders, leaders_problem=problem)
    draft = getattr(study, "draft", None)
    if (row.kind == "draft" and draft is not None and not row.archived
            and draft.mode_id == lib.COMPARISON_MODE):
        try:
            blocked = _draft_blocked(draft.draft_id, str(draft.updated_at_utc), draft,
                                     roots.get("repo_root"))
        except Exception:
            blocked = None
        try:
            check = _draft_check(draft.draft_id, str(draft.updated_at_utc), draft)
        except Exception:
            return lib.study_row(study, row.app, current=here, check_available=False,
                                 draft_blocked=blocked)
        return lib.study_row(study, row.app, current=here, draft_check=check,
                             draft_blocked=blocked)
    return row


# ── drawing ───────────────────────────────────────────────────────────────


def _head(st_module, name: str, labels: list[tuple[str, bool]], spec: list[float]) -> None:
    from ifvg_lab_ui import show

    with st_module.container(key=f"ifvg_lab_libhead_{name}"):
        cols = st_module.columns(spec, vertical_alignment="bottom")
        for col, (label, right) in zip(cols, labels, strict=True):
            with col:
                show(h.Markup(f'<div class="lab-lib-head{" num" if right else ""}">'
                              f"{h.esc(label)}</div>"), st_module)


def _name_cell(st_module, row: lib.LibraryRow, here: str, *, key: str,
               lines: list[str]) -> bool:
    """The study cell. Its name opens the study page (the "Details and actions" route)."""

    from ifvg_lab_ui import show

    tags = []
    if row.earlier_method:
        tags.append(("", lib.EARLIER_METHOD))
    elif row.kind_label and row.tab == "strategy":
        tags.append(("", row.kind_label))
    if row.app != here:
        tags.append((" app", lib.APP_NAMES[row.app].capitalize()))
    clicked = False
    with st_module.container(horizontal=True, vertical_alignment="center", gap="small",
                             key=f"ifvg_lab_libname_{key}"):
        if row.details_route:
            clicked = st_module.button(row.name, key=key, type="tertiary",
                                       help=_name_help(row, here))
        else:
            show(h.Markup(f'<div class="lab lab-lib-name"><span class="nm">{h.esc(row.name)}'
                          "</span></div>"), st_module)
        for tone, text in tags:
            st_module.html(f'<span class="lab-lib-tag{tone}">{h.esc(text)}</span>',
                           width="content")
    body = []
    if lines and lines[0]:
        body.append(f'<div class="lab-lib-tested">{h.esc(lines[0])}</div>')
    for line in lines[1:]:
        if line:
            body.append(f'<div class="lab-lib-dates">{h.esc(line)}</div>')
    if row.elsewhere:  # this application can't open it: say which one can, and how
        where, _, command = row.elsewhere.partition(": ")
        body.append(f'<div class="lab-lib-elsewhere">{h.esc(where)}: '
                    f"<code>{h.esc(command)}</code></div>")
    if body:
        show(h.Markup('<div class="lab">' + "".join(body) + "</div>"), st_module)
    return clicked


def _name_help(row: lib.LibraryRow, here: str) -> str:
    if row.details_route == "detail":
        return ("Open this study's page: progress or results, run or resume, and the study "
                "actions (clone, rename, archive or restore).")
    return _action_help(row, here)


def _status_cell(st_module, row: lib.LibraryRow) -> None:
    from ifvg_lab_ui import show

    parts = [f'<div class="lab-lib-status">{h.esc(row.status)}</div>']
    if row.status_date:
        parts.append(f'<div class="lab-lib-sub">{h.esc(row.status_date)}</div>')
    if row.note and row.tab != "drafts":
        tone = " orange" if row.note_tone == "orange" else ""
        parts.append(f'<div class="lab-lib-sub{tone}">{h.esc(row.note)}</div>')
    show(h.Markup('<div class="lab">' + "".join(parts) + "</div>"), st_module)


def _leader_cell(st_module, row: lib.LibraryRow, firm_key: str) -> None:
    from ifvg_lab_ui import show

    leader = lib.leader_for(row, firm_key)
    if row.leaders_problem:
        text, klass = row.leaders_problem, "lab-placeholder"
    elif not row.result_id:
        text, klass = ("Not finished" if row.status == "Running" else "No saved result",
                       "lab-placeholder")
    elif leader is None:
        text, klass = "Not in this study", "lab-placeholder"
    elif leader.net_cash_cents is None:
        text, klass = "None completed", "lab-placeholder"
    else:
        text = fmt.money_cents(leader.net_cash_cents)
        klass = "lab-lib-money grey" if row.earlier_method else "lab-lib-money"
    style = ' style="text-align:right;font-size:13px"' if klass == "lab-placeholder" else ""
    show(h.Markup(f'<div class="lab {klass}"{style}>{h.esc(text)}</div>'), st_module)


def _keys(row: lib.LibraryRow, here: str, seen: set[str]) -> tuple[str, str]:
    """(name key, action key) — the earlier list's keys for this application's rows."""

    mine = row.app == here
    if row.kind == "research_group":
        action = f"research_open_{row.key}" if mine else f"ifvg_open_{row.app}_group_{row.key}"
    elif mine:
        action = f"ifvg_open_{row.kind}_{row.key}"
    else:
        action = f"ifvg_open_{row.app}_{row.kind}_{row.key}"
    name = f"ifvg_details_{row.key}" if mine else f"ifvg_details_{row.app}_{row.key}"
    while name in seen or action in seen:  # never two widgets with one key
        name, action = f"{name}_", f"{action}_"
    seen.update((name, action))
    return name, action


def _action_cell(st_module, row: lib.LibraryRow, here: str, *, key: str) -> bool:
    if not row.route:  # the study cell names the application that opens it
        return False
    return st_module.button(row.action, key=key, help=_action_help(row, here))


def _action_help(row: lib.LibraryRow, here: str) -> str:
    if row.route == "funded_results":
        where = ("" if row.app == here
                 else f" It was saved by the {lib.APP_NAMES[row.app]} and opens read only.")
        return ("Open this comparison's ranking and configuration details. Reading only; "
                "nothing is run." + where)
    if row.route == "new_funded":
        return ("Continue this funded comparison on the New study page. Nothing runs until "
                "you approve the exact plan.")
    if row.route == "research_group":
        return "Open this feature and model study's saved evidence and progress."
    if row.action == lib.CONTINUE_DRAFT:
        return ("Open this draft's page to continue it, or to clone, rename or archive it. "
                "Nothing runs from here.")
    return ("Open this study's page: its results or progress, and its study actions "
            "(clone, rename, archive or restore).")


def _open(st_module, row: lib.LibraryRow, by_app: dict[str, dict[str, Any]], *,
          details: bool = False) -> None:
    from ifvg_lab_nav import go, open_funded_results

    route = row.details_route if details else row.route
    if route == "ml_phase":
        st_module.session_state["ifvg_ml_phase_pointer"] = dict(row.study.state)
        go(st_module, nav="My studies", screen="ml_phase", selected=row.key)
    elif route == "funded_results":
        from ifvg_workspace import funded_result_target

        target = funded_result_target(row.study, by_app[row.app], app=row.app)
        if target is not None:
            open_funded_results(target, st_module)
    elif route == "new_funded":
        from ifvg_ui_common import SESSION_DRAFT_KEY, STATE_PREFIX

        st_module.session_state.pop(SESSION_DRAFT_KEY, None)
        st_module.session_state[f"{STATE_PREFIX}draft_id"] = row.key
        go(st_module, nav="New study", screen="new_funded")
    elif route == "research_group":
        st_module.session_state["ifvg_workspace_research_group"] = row.key
        go(st_module, nav="My studies", screen="research_group")
    elif route == "detail":
        go(st_module, nav="My studies", screen="detail", selected=row.key)


def _funded_table(st_module, rows, firms, by_app, here) -> None:
    with st_module.container(key="ifvg_lab_libtbl_funded"):
        _head(st_module, "funded", [("Study", False), ("Status", False),
                                    *((f"Leader, {name}", True) for _, name in firms),
                                    ("", False)],
              [*_FUNDED_SPEC[:2], *([_FUNDED_SPEC[2]] * len(firms)), _FUNDED_SPEC[-1]])
        seen: set[str] = set()
        for index, row in enumerate(rows):
            name_key, action_key = _keys(row, here, seen)
            with st_module.container(key=f"ifvg_lab_librow_funded_{index}"):
                cols = st_module.columns([*_FUNDED_SPEC[:2], *([_FUNDED_SPEC[2]] * len(firms)),
                                          _FUNDED_SPEC[-1]], vertical_alignment="center")
                with cols[0]:
                    clicked = _name_cell(st_module, row, here, key=name_key,
                                         lines=[row.tested, row.dates])
                with cols[1]:
                    _status_cell(st_module, row)
                for col, (firm_key, _) in zip(cols[2:-1], firms, strict=True):
                    with col:
                        _leader_cell(st_module, row, firm_key)
                with cols[-1]:
                    pressed = _action_cell(st_module, row, here, key=action_key)
            if pressed:
                _open(st_module, row, by_app)
            elif clicked:
                _open(st_module, row, by_app, details=True)


def _study_table(st_module, name: str, rows, by_app, here) -> None:
    with st_module.container(key=f"ifvg_lab_libtbl_{name}"):
        _head(st_module, name, [("Study", False), ("Status", False), ("", False)], _STUDY_SPEC)
        seen: set[str] = set()
        for index, row in enumerate(rows):
            name_key, action_key = _keys(row, here, seen)
            with st_module.container(key=f"ifvg_lab_librow_{name}_{index}"):
                cols = st_module.columns(_STUDY_SPEC, vertical_alignment="center")
                with cols[0]:
                    if row.tab == "drafts":
                        lines = [row.tested, row.note]
                    else:
                        lines = [row.tested, row.dates]
                    clicked = _name_cell(st_module, row, here, key=name_key, lines=lines)
                with cols[1]:
                    _status_cell(st_module, row)
                with cols[2]:
                    pressed = _action_cell(st_module, row, here, key=action_key)
            if pressed:
                _open(st_module, row, by_app)
            elif clicked:
                _open(st_module, row, by_app, details=True)


def _drafts_table(st_module, name: str, rows, by_app, here) -> None:
    from ifvg_lab_ui import show

    with st_module.container(key=f"ifvg_lab_libtbl_{name}"):
        seen: set[str] = set()
        for index, row in enumerate(rows):
            name_key, action_key = _keys(row, here, seen)
            with st_module.container(key=f"ifvg_lab_librow_{name}_{index}"):
                cols = st_module.columns(_DRAFT_SPEC, vertical_alignment="center")
                with cols[0]:
                    clicked = _name_cell(st_module, row, here, key=name_key, lines=[row.tested])
                with cols[1]:
                    tone = " orange" if row.note_tone == "orange" else ""
                    show(h.Markup(f'<div class="lab lab-lib-note{tone}">{h.esc(row.note)}'
                                  "</div>"), st_module)
                with cols[2]:
                    pressed = _action_cell(st_module, row, here, key=action_key)
            if pressed:
                _open(st_module, row, by_app)
            elif clicked:
                _open(st_module, row, by_app, details=True)


def _paged(st_module, rows: list[lib.LibraryRow], name: str) -> tuple[list, int, int]:
    key = f"{_P}page_{name}"
    pages = max(1, (len(rows) + lib.PAGE_SIZE - 1) // lib.PAGE_SIZE)
    current = st_module.session_state.get(key, 1)
    if not isinstance(current, int) or current > pages or current < 1:
        current = st_module.session_state[key] = min(max(1, int(current or 1)), pages)
    shown, pages = lib.page(rows, current)
    return shown, current, pages


def _pager(st_module, rows: list, shown: list, current: int, pages: int, name: str) -> None:
    from ifvg_lab_ui import show

    if pages <= 1:
        return
    first = (current - 1) * lib.PAGE_SIZE + 1
    left, right = st_module.columns([4, 1], vertical_alignment="center")
    with left:
        show(h.Markup(f'<div class="lab lab-line">Showing {first}–{first + len(shown) - 1} of '
                      f"{len(rows)}</div>"), st_module)
    with right:
        st_module.number_input("Page", min_value=1, max_value=pages, step=1,
                               key=f"{_P}page_{name}",
                               help=f"The list shows {lib.PAGE_SIZE} studies per page.")


def _empty(st_module, text: str) -> None:
    from ifvg_lab_ui import show

    show(h.note(text), st_module)


# ── page ──────────────────────────────────────────────────────────────────

_TAB = f"{_P}tab"


def _choose_tab(st_module, tab: str) -> None:
    st_module.session_state[_TAB] = tab


def _tabs(st_module, counts: dict[str, int]) -> str:
    """Text tabs (buttons, so only the chosen tab renders); the chosen one is underlined."""

    tab = st_module.session_state.get(_TAB)
    if tab not in lib.TABS:
        tab = st_module.session_state[_TAB] = "funded"
    ink = css_var("ink")
    st_module.html(f"<style>.st-key-{_P}tab_{tab} button {{ border-bottom: 3px solid {ink} "
                   f"!important; }} .st-key-{_P}tab_{tab} button p {{ color: {ink} !important;"
                   " font-weight: 600 !important; }</style>")
    with st_module.container(key="ifvg_lab_libtabs", horizontal=True, gap=None):
        for option in lib.TABS:
            st_module.button(lib.tab_label(option, counts[option]), key=f"{_P}tab_{option}",
                             type="tertiary", on_click=_choose_tab, args=(st_module, option),
                             help=_TAB_HELP)
    return st_module.session_state[_TAB]



def render_library(st_module, studies, roots) -> None:
    from ifvg_lab_nav import current_app
    from ifvg_lab_ui import show

    here = current_app(roots)
    st_module.html(f"<style>{_CSS}</style>")
    left, right = st_module.columns([5, 1], vertical_alignment="bottom")
    with left:
        show(h.page_header("My studies",
                           subtitle="Every study in one place, whichever app created it."),
             st_module)
    with right:
        if st_module.button("New study", type="primary", key="ifvg_workspace_new",
                            help="Set up a new funded comparison or another study type. "
                                 "Nothing runs until you approve it.", width="stretch"):
            from ifvg_workspace import rail_destination

            rail_destination(st_module, "New study")
    rows, by_app, notices = library_rows(studies, roots)
    counts = lib.counts(rows)
    tab = _tabs(st_module, counts)
    for notice in notices:
        show(h.note(notice, "orange"), st_module)
    tab_rows = [_complete(row, by_app, here) for row in lib.rows_in_tab(rows, tab)]
    firms = lib.firm_columns(r for r in tab_rows if r.tab == "funded")
    statuses = sorted({r.status for r in tab_rows})
    status_key = f"{_P}status_{tab}"
    chosen = [s for s in st_module.session_state.get(status_key, []) if s in statuses]
    search_col, status_col, label_col, sort_col = st_module.columns(
        [7.2, 1.6, 0.45, 2.3], vertical_alignment="center")
    with search_col:
        query = st_module.text_input(
            "Search studies", key=f"{_P}search", placeholder="Search by name or setting",
            label_visibility="collapsed",
            help="Matches the name, what the study tested (its settings), dates, status "
                 "and the application that saved it.")
    with status_col:
        label = ("Status: all" if not chosen else f"Status: {chosen[0]}" if len(chosen) == 1
                 else f"Status: {len(chosen)} chosen")
        with st_module.popover(label, help="Show only studies with these statuses.",
                               width="stretch"):
            chosen = st_module.multiselect(
                "Show only these statuses", statuses, default=chosen, key=status_key,
                help="Leave empty to show every status.")
    with label_col:
        show(h.Markup('<div class="lab lab-lib-sortlabel">Sort</div>'), st_module)
    with sort_col:
        sort = st_module.selectbox(
            "Sort", lib.sort_options(tab, firms), key=f"{_P}sort_{tab}",
            label_visibility="collapsed",
            help="Order the list. Sorting by a firm's leader uses that firm only; firms are "
                 "never added together.")
    visible = lib.sort_rows(lib.filter_rows(tab_rows, query, chosen), sort, firms)
    searching = bool(query.strip() or chosen)
    if tab == "funded":
        _render_funded(st_module, tab_rows, visible, firms, by_app, searching, here)
    elif tab == "drafts":
        _render_list(st_module, "drafts", tab_rows, visible, by_app, searching, here,
                     empty="No drafts. Start one with New study.", drafts=True)
    elif tab == "strategy":
        _render_list(st_module, "strategy", tab_rows, visible, by_app, searching, here,
                     empty="No strategy studies yet. Start one with New study.")
    elif tab == "model":
        _render_list(st_module, "model", tab_rows, visible, by_app, searching, here,
                     empty="No feature and model studies yet. Start one with New study, "
                           "under Other study types.")
    else:
        _render_list(st_module, "archived", tab_rows, visible, by_app, searching, here,
                     empty="Nothing is archived. Archived studies and drafts can be "
                           "restored from their study page.")


def _render_funded(st_module, tab_rows, visible, firms, by_app, searching, here) -> None:
    from ifvg_lab_ui import show

    studies = [r for r in visible if r.tab == "funded"]
    drafts = [r for r in visible if r.tab == "drafts"]
    if studies:
        shown, current, pages = _paged(st_module, studies, "funded")
        _funded_table(st_module, shown, firms, by_app, here)
        _pager(st_module, studies, shown, current, pages, "funded")
    elif searching:
        _empty(st_module, "No funded comparisons match this search.")
    else:
        _empty(st_module, "No funded comparisons yet. Start one with New study.")
    if not any(r.tab == "drafts" for r in tab_rows):
        return
    show(h.Markup('<div class="lab" style="margin-top:8px"><div class="lab-h2" '
                  'style="font-size:24px">Drafts</div></div>'), st_module)
    if drafts:
        shown, current, pages = _paged(st_module, drafts, "funded_drafts")
        _drafts_table(st_module, "funded_drafts", shown, by_app, here)
        _pager(st_module, drafts, shown, current, pages, "funded_drafts")
    else:
        _empty(st_module, "No funded comparison drafts match this search.")


def _render_list(st_module, name, tab_rows, visible, by_app, searching, here, *, empty,
                 drafts=False) -> None:
    if not visible:
        _empty(st_module, "No studies match this search." if searching and tab_rows else empty)
        return
    shown, current, pages = _paged(st_module, visible, name)
    if drafts:
        _drafts_table(st_module, name, shown, by_app, here)
    else:
        _study_table(st_module, name, shown, by_app, here)
    _pager(st_module, visible, shown, current, pages, name)
