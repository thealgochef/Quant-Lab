"""New funded comparison (mocks 10, 10b) and Review and approve (mocks 11, 11b).

Setup builds a funded comparison as a normal study draft: variations around the
named S0_D80_W1_P1 baseline of the verified strategy study, the firms and
sizes, plus the settings no plan can carry yet (dates, gap rules, withdrawal
triggers, pass/fail checks — saved under their own key and blocking approval
with a plain sentence). Review shows the plan rebuilt from the SAVED draft and
approves it only through the existing gated path in
``ifvg_funded_comparison_study`` (``save_plan`` + ``record_owner_approval``,
then ``_freeze_and_launch``, whose ``dispatch_problem`` re-reads the draft,
rebuilds the plan and requires the identical plan id and its approval). The
earlier configurator's approval and that launch check refuse on the same
blockers, computed from the saved draft by ``funded_setup.saved_draft_blockers``.

Repair R1 holds here too: opening or refreshing never writes a draft (only an
owner edit or "Save draft" does); a saved draft the running engine cannot
represent opens read-only and is never saved from this application.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st
from ifvg_ui_common import SESSION_DRAFT_KEY, STATE_PREFIX

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_setup as fs
from alpha_lab.agents.data_infra.ifvg.presentation.lab import html as h
from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import css_var

__all__ = ["render_funded_review", "render_funded_setup"]

SETUP = "ifvg_lab_v1_setup_"
#: Review and approve's own session prefix, never Trade review's "ifvg_lab_v1_review_":
#: _sync deletes every key under this prefix, and a shared key (Trade review's stored back
#: link is named like this page's "Back to setup" button) would press that button by itself.
REVIEW = "ifvg_lab_v1_approve_"
DRAFT_KEY = f"{STATE_PREFIX}draft_id"
OTHER_TYPES_KEY = "ifvg_lab_v1_other_types"
_LINK_APPLIED = "ifvg_lab_v1_draft_link_applied"
_NAV = "ifvg_workspace_destination"
_SCREEN = "ifvg_workspace_screen"
QUESTION = "Which configuration earns the most cash after every account cost?"
NAMED_NOTE = "Default for new studies. Built from that configuration's saved identity, not retyped."
PROTECTED = "June 11, 2026 onward is protected for confirmation runs and can't be selected here."
ENGINE_NOTE = ("Needs the engine version that supports half exits. Checked at approval and again "
               "at launch.")
UNKNOWN_DRAFT_NOTE = "This link names a draft this application doesn't have; nothing was opened."
_STATEMENT = ("Approved on the study screen by the owner: run this exact funded configuration "
              "comparison on the saved historical period.")

_CSS = """
.fs-title{font-family:var(--lab-sans);font-size:17px;font-weight:600;color:var(--lab-ink);
  margin:0}
.fs-steps{display:flex;gap:8px;margin-top:6px;font-size:13px}
.fs-step{padding:6px 12px;border-radius:999px;background:var(--lab-panel);
  border:1px solid var(--lab-control-border);color:var(--lab-ink) !important;text-decoration:none}
.fs-step.on{background:var(--lab-ink);color:var(--lab-on-ink) !important;
  border-color:var(--lab-ink);font-weight:600}
.fs-axis{display:flex;flex-direction:column;gap:8px}
.fs-axis-head{display:flex;align-items:center;gap:10px;font-size:14px;font-weight:600;
  color:var(--lab-ink)}
.fs-chips{display:flex;flex-wrap:wrap;gap:8px}
.fs-chip{display:flex;align-items:center;gap:8px;padding:8px 10px 8px 14px;
  border:1px solid var(--lab-blue);background:var(--lab-blue-light);border-radius:8px;
  font-size:14px;color:var(--lab-ink);min-height:44px;box-sizing:border-box;line-height:1.35}
.fs-chip.mono{font-family:var(--lab-mono)}
.fs-chip.solo{padding-right:14px}
.fs-x{width:28px;height:28px;border:none;background:transparent;cursor:pointer;font-size:16px;
  color:var(--lab-blue);border-radius:6px;flex-shrink:0}
.fs-x:hover{background:var(--lab-blue-band)}
.fs-add{padding:8px 14px;min-height:44px;border:1px dashed var(--lab-dashed);
  background:var(--lab-panel);border-radius:8px;font:inherit;font-size:14px;
  color:var(--lab-body-2);cursor:pointer}
.fs-options{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 12px;
  background:var(--lab-soft-panel);border-radius:8px;font-size:13px;color:var(--lab-muted)}
.fs-option{padding:6px 12px;min-height:36px;border:1px dashed var(--lab-blue);
  background:var(--lab-panel);border-radius:8px;font:inherit;font-size:14px;
  color:var(--lab-blue-dark);cursor:pointer}
.fs-fixed{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;
  border-top:1px solid var(--lab-light-rule)}
.fs-fixed>div{display:flex;flex-direction:column;gap:4px;padding-top:12px}
.fs-k{font-size:13px;color:var(--lab-muted)}.fs-v{font-size:14px;color:var(--lab-ink)}
.fs-link{background:none;border:none;padding:0;color:var(--lab-blue);text-decoration:underline;
  cursor:pointer;font:inherit;font-size:13px;align-self:flex-start}
.fs-muted{font-size:13px;color:var(--lab-muted);line-height:1.45}
.fs-warn{font-size:14px;color:var(--lab-orange-dark);font-weight:500;line-height:1.5}
.fs-plan-lines{display:flex;flex-direction:column;gap:4px;font-family:var(--lab-mono);
  font-size:14px;color:var(--lab-body)}
.fs-plan-count{font-family:var(--lab-mono);font-size:24px;font-weight:500;color:var(--lab-ink);
  white-space:nowrap}
.fs-soft{display:flex;flex-direction:column;gap:4px;padding:12px 14px;
  background:var(--lab-soft-panel);border-radius:8px}
.fs-dates{display:flex;justify-content:space-between;align-items:center;padding:12px 14px;
  background:var(--lab-soft-panel);border-radius:8px;font-size:14px;gap:12px}
.fs-protect{font-size:14px;color:var(--lab-orange);font-weight:500}
.fs-texttile{background:var(--lab-panel);border:1px solid var(--lab-rule);border-radius:12px;
  padding:16px 18px;display:flex;flex-direction:column;gap:2px;min-width:0}
.fs-texttile .v{font-size:15px;margin-top:4px;color:var(--lab-ink)}
.fs-grid-head{display:grid;grid-template-columns:1fr 180px 1.2fr;gap:12px;padding:8px 0;
  font-size:13px;font-weight:600;color:var(--lab-muted);border-bottom:1px solid var(--lab-rule)}
.fs-gate-label{font-size:14px;color:var(--lab-ink)}
.fs-ok{font-size:14px;color:var(--lab-blue-dark)}
.fs-bad{font-size:14px;color:var(--lab-orange-dark);font-weight:500;line-height:1.45}
[class*="st-key-ifvg_lab_v1_approve_gaterow_"]{border-bottom:1px solid var(--lab-grid);
  padding:2px 0 6px}
[class*="st-key-ifvg_lab_v1_approve_gaterow_"] [data-testid="stTextInput"] input,
[class*="st-key-ifvg_lab_v1_approve_gaterow_"] [data-baseweb="select"]{font-family:var(--lab-mono)}
.st-key-ifvg_lab_card_setup_plan button{width:100%;min-height:48px}
.st-key-ifvg_lab_v1_setup_baseline [data-baseweb="select"] div{white-space:normal !important;
  overflow:visible !important;text-overflow:clip !important}
.st-key-ifvg_lab_v1_setup_baseline [data-baseweb="select"]>div{height:auto !important;
  min-height:44px;padding-top:4px;padding-bottom:4px}
details.fs-same summary{cursor:pointer;font-weight:600;font-size:14px}
details.fs-same{border-top:1px solid var(--lab-light-rule);padding-top:12px;font-size:14px}
.fs-same-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px 24px;
  padding-top:10px;color:var(--lab-body)}
.fs-icon{width:22px;height:22px;flex-shrink:0;margin-top:1px;background-color:var(--lab-orange);
  -webkit-mask:var(--fs-icon) no-repeat center / contain;
  mask:var(--fs-icon) no-repeat center / contain}
.fs-alert{display:flex;gap:14px;padding:16px 20px;background:var(--lab-orange-light);
  border:1px solid var(--lab-orange-mid);border-radius:12px;color:var(--lab-orange-darker);
  font-size:15px;line-height:1.5}
.fs-alert b{color:var(--lab-orange-darker)}
.fs-alert.small{font-size:14px;padding:14px 16px;border-radius:10px}
.fs-foot{display:flex;justify-content:space-between;align-items:center;padding-top:6px;
  font-size:14px;color:var(--lab-body-2)}
"""


_ICON_SVG = {
    "lock": "<rect x='5' y='11' width='14' height='10' rx='2'/>"
            "<path d='M8 11V7a4 4 0 0 1 8 0v4'/>",
    "triangle": "<path d='M12 3l9 16H3z'/><path d='M12 10v4M12 17h.01'/>",
}


def _icon_css() -> str:
    """Alert icons as CSS masks filled from the palette (st.html removes inline SVG).

    The mask only gives the shape (its stroke may be any opaque color); the visible
    color is ``.fs-icon``'s ``--lab-orange`` variable, so the icon follows the theme.
    """

    from urllib.parse import quote

    rules = []
    for name, body in _ICON_SVG.items():
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
               "stroke='black' stroke-width='2.2' stroke-linecap='round' "
               f"stroke-linejoin='round'>{body}</svg>")
        rules.append(f'.fs-icon.{name}{{--fs-icon:url("data:image/svg+xml,'
                     f'{quote(svg, safe=" =:/")}")}}')
    return " ".join(rules)


# ── shared pieces ─────────────────────────────────────────────────────────


def _style(st_module) -> None:
    st_module.html(f"<style>{_CSS} {_icon_css()}</style>")


@st.cache_resource(show_spinner="Opening the verified strategy study…", ttl=900)
def _cached_sources() -> dict[str, Any]:
    from ifvg_funded_comparison_study import _sources

    return _sources()


def _named() -> tuple[Any, str | None]:
    # one check shared with the earlier configurator's approval and launch refusal
    return fs.named_baseline()


def _digest(root: Path, draft_id: str) -> str:
    try:
        return hashlib.sha256((root / draft_id / "draft.json").read_bytes()).hexdigest()
    except OSError:
        return ""  # a draft not saved yet


def _sync(st_module, prefix: str, root: Path, draft_id: str) -> bool:
    """Widget values start from the draft as saved on disk (repair R1, stale windows).

    When another draft is opened, or this one was saved elsewhere, every widget
    of this screen is reset so an old value is never saved over the draft and
    a click on the old page is dropped. Returns True when the page was reset.
    """

    marker = [draft_id, _digest(root, draft_id)]
    previous = st_module.session_state.get(prefix + "marker")
    if previous == marker:
        return False
    for key in [k for k in st_module.session_state if str(k).startswith(prefix)]:
        del st_module.session_state[key]
    st_module.session_state[prefix + "marker"] = marker
    if previous and previous[0] == draft_id and previous[1] and marker[1]:
        st_module.info("This draft was saved from another window; showing its saved settings.")
    return True


def _clickable(st_module, markup, *, key: str) -> str | None:
    """``ifvg_lab_ui.clickable``; static HTML where the component cannot mount.

    Streamlit's headless test runner cannot mount components v2 (it raises a
    TypeError inside Streamlit); the page then shows the same HTML without clicks.
    """

    from ifvg_lab_ui import clickable, show

    try:
        return clickable(markup, key=key, st_module=st_module)
    except TypeError:
        show(markup, st_module)
        return None


def _alert(first: str, rest: Any = None, *, icon: str = "triangle", small: bool = False,
           items: list[str] | None = None) -> h.Markup:
    """The design system's alert with its icon drawn in CSS (st.html strips inline SVG)."""

    more = f'<div style="margin-top:4px">{h.esc(rest)}</div>' if rest else ""
    if items:
        more += ('<ul style="margin:6px 0 0 18px;padding:0">'
                 + "".join(f"<li>{h.esc(i)}</li>" for i in items) + "</ul>")
    return h.Markup(f'<div class="lab fs-alert{" small" if small else ""}" role="alert">'
                    f'<span class="fs-icon {icon}" aria-hidden="true"></span>'
                    f"<div><b>{h.esc(first)}</b>{more}</div></div>")


def _mark(prefix: str, name: str = "edited"):
    def callback() -> None:
        import streamlit

        streamlit.session_state[prefix + name] = True

    return callback


def _go(st_module, *, screen: str, nav: str = "New study") -> None:
    st_module.session_state[_NAV] = nav
    st_module.session_state[_SCREEN] = screen
    st_module.rerun()


def _apply_draft_link(st_module, roots) -> str | None:
    """``?draft=<id>`` opens that saved draft once per session (read only until edited).

    Returns a one-sentence note when the link names a draft this application doesn't
    have (another application's, a deleted one, a malformed id); nothing is opened
    then. A page address without a draft (a rail click clears it) opens nothing.
    """

    if st_module.session_state.get(_LINK_APPLIED):
        return None
    st_module.session_state[_LINK_APPLIED] = True
    try:
        draft_id = st_module.query_params.get("draft")
    except Exception:  # headless tests without a page address
        return None
    if not draft_id:
        return None
    draft_id = str(draft_id)
    if re.fullmatch(r"[0-9a-f]{32}", draft_id) and (
            Path(roots["draft_root"]) / draft_id / "draft.json").is_file():
        st_module.session_state[DRAFT_KEY] = draft_id
        return None
    return UNKNOWN_DRAFT_NOTE


def _header(st_module, title: str, *, step: int, question: bool) -> str | None:
    """Breadcrumb, title, question, the two-step indicator and "Other study types".

    Returns ``"step|review"`` when the second step was clicked on the setup page (the
    page decides what that means once it is built); other clicks navigate here.
    """

    pills = []
    for number, label, action in ((1, "Set up", "step|setup"),
                                  (2, "Review and approve", "step|review")):
        text = f"{number} · {label}"
        if number == step:
            pills.append(f'<span class="fs-step on" aria-current="step">{h.esc(text)}</span>')
        else:
            pills.append(f'<a href="#" class="fs-step" data-action="{action}">{h.esc(text)}</a>')
    head = h.page_header(title, crumbs=(("My studies", "nav|library"), ("New study", None)),
                         subtitle=QUESTION if question else None)
    markup = h.Markup(f'{head}<div class="lab fs-steps">{"".join(pills)}</div>')
    left, right = st_module.columns([5, 1.25], vertical_alignment="bottom")
    with left:
        action = _clickable(st_module, markup, key=f"new_funded_header_{step}")
    with right:
        if st_module.button("Other study types", key=OTHER_TYPES_KEY, width="stretch",
                            help="Evaluate, Compare, Search and more study types (the earlier "
                                 "study chooser)."):
            st_module.session_state.pop(DRAFT_KEY, None)
            st_module.session_state.pop(SESSION_DRAFT_KEY, None)
            _go(st_module, screen="new")
    if action == "nav|library":
        _go(st_module, screen="list", nav="My studies")
    elif action == "step|setup" and step != 1:
        _go(st_module, screen="new_funded")
    return action if action == "step|review" and step == 1 else None


# ── setup page ────────────────────────────────────────────────────────────


@dataclass
class _Setup:
    roots: dict[str, Any]
    root: Path
    draft: Any
    persisted: bool
    source: Any
    base_ids: dict[str, str]
    offered: dict[str, tuple[str, ...]]
    named: Any
    named_problem: str | None


def _session_draft(st_module):
    from ifvg_study_wizard import _session_draft

    return _session_draft(st_module)


def _open_setup_draft(st_module, root: Path):
    """(draft or None, saved on disk, problem sentence or None) for this page."""

    from alpha_lab.agents.data_infra.ifvg.study_drafts import load_draft

    draft_id = st_module.session_state.get(DRAFT_KEY)
    session = _session_draft(st_module)
    if draft_id:
        if session is not None and session.draft_id == draft_id and not _digest(root, draft_id):
            return session, False, None
        try:
            return load_draft(root, draft_id), True, None
        except Exception:
            return None, False, ("This saved draft could not be read. Return to My studies to "
                                 "choose another study.")
    if (session is not None and session.mode_id == fs.COMPARISON_MODE
            and not _digest(root, session.draft_id)):
        return session, False, None
    return None, False, None


def _new_draft(source, named):
    from ifvg_funded_comparison_study import start_comparison_draft

    draft = start_comparison_draft(fs.new_study_name(date.today()), source)
    base = named.baseline_id if named is not None else None
    draft.steps["review"]["funded_comparison"] = fs.new_variation_settings(
        source.package.run_id if source is not None else None, base)
    draft.steps["search_space"] = {"mode_id": fs.COMPARISON_MODE, "axis_selections": {}}
    return draft


def _commit(st_module, page: _Setup, choices: fs.SetupChoices) -> bool:
    """Save an owner edit (a saved draft) or keep it in this session (a new page)."""

    from alpha_lab.agents.data_infra.ifvg.study_drafts import save_draft

    fs.write_choices(page.draft, choices, offered=page.offered)
    if not page.persisted:
        st_module.session_state[SESSION_DRAFT_KEY] = asdict(page.draft)
        return True
    try:
        save_draft(page.root, page.draft)
    except Exception:
        st_module.error("Changes could not be saved. Keep this page open and try again.")
        return False
    st_module.session_state[SETUP + "marker"] = [page.draft.draft_id,
                                                 _digest(page.root, page.draft.draft_id)]
    return True


def _create(st_module, page: _Setup) -> bool:
    """"Save draft" / "Continue to review" on a new page: the draft is created now."""

    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import human_name
    from alpha_lab.agents.data_infra.ifvg.study_drafts import save_draft

    name = (page.draft.display_name or "").strip()
    if not name or human_name(name, "") != name:
        st_module.error("Name this study in More (a descriptive name, no technical "
                        "identifiers) before it is saved.")
        return False
    if not page.persisted:
        # the page's current choices, written the same way an edit writes them
        offered = page.offered
        choices = fs.choices_from_draft(page.draft, base_ids=page.base_ids, offered=offered)
        fs.write_choices(page.draft, choices, offered=offered)
    try:
        save_draft(page.root, page.draft)
    except Exception:
        st_module.error("The draft could not be saved. Keep this page open and try again.")
        return False
    st_module.session_state[DRAFT_KEY] = page.draft.draft_id
    st_module.session_state.pop(SESSION_DRAFT_KEY, None)
    st_module.session_state[SETUP + "marker"] = [page.draft.draft_id,
                                                 _digest(page.root, page.draft.draft_id)]
    page.persisted = True
    return True


def render_funded_setup(st_module, roots) -> None:
    from alpha_lab.propsim.funded.comparison_draft import (
        DEFAULT_SETTINGS,
        check_saved_comparison,
        saved_settings,
        saved_study_selections,
    )

    link_note = _apply_draft_link(st_module, roots)
    root = Path(roots["draft_root"])
    draft, persisted, problem = _open_setup_draft(st_module, root)
    if draft is not None and draft.mode_id != fs.COMPARISON_MODE:
        # continuing another study type: the earlier guided wizard owns it
        from ifvg_research_wizard import render_new_study

        render_new_study(st_module, roots=roots)
        return
    _style(st_module)
    header_action = _header(st_module, "New funded comparison", step=1, question=True)
    if link_note:
        h_note(st_module, link_note, "orange")
    if problem:
        h_note(st_module, problem)
        return
    sources = _cached_sources()
    named, named_problem = _named()
    if draft is None:
        source = (sources.get(named.source_run_id) if named is not None else None) or next(
            (s for s in sources.values() if "S0_D80_W1_P1" in s.by_name), None)
        if source is None:
            h_note(st_module, "The verified strategy study that funded comparisons replay isn't "
                              "available on this computer, so a new comparison can't be set up "
                              "here.")
            _more(st_module, None, roots)
            return
        draft = _new_draft(source, named)
        st_module.session_state[SESSION_DRAFT_KEY] = asdict(draft)  # this session only
    if persisted and (draft.archived or draft.status == "frozen"):
        h_note(st_module, "This study is saved as history. Return to My studies to restore or "
                          "clone it.")
        return
    _sync(st_module, SETUP, root, draft.draft_id)
    settings = saved_settings(draft)
    merged = {**DEFAULT_SETTINGS, "plan_kind": "study", **settings}
    if persisted:
        check = check_saved_comparison(settings, saved_study_selections(draft), sources,
                                       store_root=Path(roots["store_root"]))
        if not check.runnable:
            _setup_read_only(st_module, check, roots, draft)
            return
        if merged["plan_kind"] != "variations":
            _setup_study_kind(st_module, draft, roots)
            return
    source = sources.get(merged.get("source_run_id"))
    base = (merged.get("variation") or {}).get("base")
    if source is None or base not in source.by_name:
        h_note(st_module, "This draft's starting configuration isn't in a verified strategy "
                          "study available here, so it can't be edited on this page. Its saved "
                          "settings haven't been changed.")
        _more(st_module, None, roots)
        return
    base_ids = dict(source.by_name[base].axis_value_ids)
    page = _Setup(roots=roots, root=root, draft=draft, persisted=persisted, source=source,
                  base_ids=base_ids, offered=fs.offered_options(base_ids), named=named,
                  named_problem=named_problem)
    choices = fs.choices_from_draft(draft, base_ids=base_ids, offered=page.offered)

    def continue_to_review() -> None:
        if page.persisted or _create(st_module, page):
            _go(st_module, screen="approve_funded")

    if header_action == "step|review":
        continue_to_review()
    left, right = st_module.columns([1.95, 1], gap="medium")
    with left:
        choices = _baseline_card(st_module, page, choices)
        choices, resolved = _dates_card(st_module, page, choices)
        _settings_card(st_module, page, choices)
        choices = _firms_card(st_module, page, choices)
        _more(st_module, page, roots)
    with right:
        _plan_panel(st_module, page, choices, resolved, continue_to_review)


def h_note(st_module, text: str, tone: str = "") -> None:
    from ifvg_lab_ui import show

    show(h.note(text, tone), st_module)


def _setup_read_only(st_module, check, roots, draft=None) -> None:
    """A saved draft this engine cannot represent: the existing read-only view, never saved.

    Its first sentence uses the draft's one shared count (the same as My studies and
    Review and approve), and its engine problems are shown in plain words.
    """

    from ifvg_funded_comparison_study import _render_saved_read_only

    lead = fs.blocked_engine_lead(fs.plan_count(check, draft), check.count_is_exact,
                                  check.needs_half_exit_engine)
    with st_module.container(key="ifvg_lab_card_setup_readonly"):
        _render_saved_read_only(st_module, check, roots, lead=lead, words=fmt.display_words)
    if st_module.button("See the saved plan on Review and approve", key=SETUP + "to_review",
                        help="Shows the plan exactly as saved. Nothing is saved or changed."):
        _go(st_module, screen="approve_funded")


def _setup_study_kind(st_module, draft, roots) -> None:
    """A draft of the other plan kind: the earlier configurator shows every one of its settings."""

    from ifvg_funded_comparison_study import render_comparison_configuration

    h_note(st_module, "This draft compares configurations from the completed strategy study. "
                      "It opens on the earlier configurator below, which shows every one of its "
                      "settings.")
    render_comparison_configuration(st_module, draft, roots)


# ── setup cards ───────────────────────────────────────────────────────────


def _baseline_card(st_module, page: _Setup, choices: fs.SetupChoices) -> fs.SetupChoices:
    from ifvg_lab_ui import show

    with st_module.container(key="ifvg_lab_card_setup_baseline"):
        show(h.Markup('<div class="lab fs-title">Starting configuration</div>'), st_module)
        options = [fs.NAMED, fs.LEGACY] if page.named is not None else [fs.LEGACY]
        labels = {fs.LEGACY: fs.LEGACY_LABEL}
        if page.named is not None:
            labels[fs.NAMED] = fs.baseline_description(page.named.baseline_id,
                                                       dict(page.named.axis_value_ids))
        current = choices.baseline if choices.baseline in options else options[0]
        picked = st_module.selectbox(
            "Baseline", options, index=options.index(current), format_func=labels.get,
            key=SETUP + "baseline", on_change=_mark(SETUP),
            help="The configuration every variation starts from. The named baseline is the "
                 "verified study's S0_D80_W1_P1; the legacy baseline is kept so older studies "
                 "open exactly as saved.")
        if st_module.session_state.pop(SETUP + "edited", False) and picked != choices.baseline:
            choices = replace(choices, baseline=picked)
            _commit(st_module, page, choices)
        if picked == fs.LEGACY:
            parts = fs.legacy_warning_parts()
            if parts:
                show(_alert(parts[0], parts[1], small=True), st_module)
            legacy = next((b.text for b in fs.setup_blockers(choices, base_ids=page.base_ids)
                           if b.key == "baseline"), None)
            if legacy:
                show(h.note(legacy, "orange"), st_module)
        elif page.named_problem:
            show(h.note(page.named_problem, "orange"), st_module)
        else:
            show(h.Markup(f'<div class="lab" style="font-size:14px;color:{css_var("body_2")};'
                          f'line-height:1.5">{h.esc(NAMED_NOTE)}</div>'), st_module)
    return choices


def _resolve_dates(page: _Setup, start: str, end: str):
    from alpha_lab.agents.data_infra.ifvg.research_period import (
        local_day_has_data,
        resolve_research_range,
    )

    repo = Path(page.roots.get("repo_root") or Path.cwd())
    return resolve_research_range(start, end, day_has_data=local_day_has_data(repo))


def _dates_card(st_module, page: _Setup, choices: fs.SetupChoices):
    from ifvg_lab_ui import show

    from alpha_lab.agents.data_infra.ifvg.research_period import (
        EARLIEST_LOCAL_MARKET_DATE,
        LAST_PERMITTED_DAY,
        WARMUP_STORE_DAYS,
        earliest_evidence_day,
    )

    source_days = page.source.evaluation_dates
    start = choices.start or source_days[0]
    end = choices.end or source_days[-1]
    low = date.fromisoformat(earliest_evidence_day())
    high = date.fromisoformat(LAST_PERMITTED_DAY)

    def pickable(text: str, fallback: str) -> date:
        try:
            day = date.fromisoformat(text)
        except (TypeError, ValueError):
            day = date.fromisoformat(fallback)
        return min(max(day, low), high)

    with st_module.container(key="ifvg_lab_card_setup_dates"):
        show(h.Markup('<div class="lab fs-title">Dates</div>'), st_module)
        first, second = st_module.columns(2)
        picked_start = first.date_input(
            "Start date", value=pickable(start, source_days[0]), min_value=low, max_value=high,
            format="MM/DD/YYYY", key=SETUP + "start", on_change=_mark(SETUP, "dates_edited"),
            help=f"Earliest allowed: {fmt.date_long(low)} (stored market data starts "
                 f"{fmt.date_long(EARLIEST_LOCAL_MARKET_DATE)} and a study first replays ten "
                 "warmup days).")
        picked_end = second.date_input(
            "End date", value=pickable(end, source_days[-1]), min_value=low, max_value=high,
            format="MM/DD/YYYY", key=SETUP + "end", on_change=_mark(SETUP, "dates_edited"),
            help="June 10, 2026 is the last day that can be chosen; June 11, 2026 onward is "
                 "protected.")
        if st_module.session_state.pop(SETUP + "dates_edited", False) and picked_start and \
                picked_end:
            new = replace(choices, start=picked_start.isoformat(), end=picked_end.isoformat())
            if (new.start, new.end) != (choices.start, choices.end):
                choices = new
                _commit(st_module, page, choices)
        earliest = fs.earliest_start_sentence(low.isoformat(), EARLIEST_LOCAL_MARKET_DATE,
                                              WARMUP_STORE_DAYS)
        show(h.Markup(f'<div class="lab fs-muted">{h.esc(earliest)}</div>'), st_module)
        start = choices.start or source_days[0]
        end = choices.end or source_days[-1]
        resolved = _resolve_dates(page, start, end)
        if resolved.trading_days:
            count = h.Markup(f'<span class="lab-mono" style="font-weight:500">'
                             f'{len(resolved.trading_days):,}</span>')
            line = h.Markup(
                f'<div class="lab fs-dates"><div>{count} trading days, plus '
                f'{len(resolved.warmup_dates)} warmup days · holidays and closed sessions '
                f'removed</div><div class="fs-muted">'
                f'{h.esc(fmt.date_range(resolved.trading_days[0], resolved.trading_days[-1]))}'
                '</div></div>')
            show(line, st_module)
            with st_module.expander("See every date"):
                _every_date(st_module, resolved)
        for problem in resolved.problems:
            show(h.note(problem, "orange"), st_module)
        for warning in resolved.warnings:
            show(h.note(warning, "orange"), st_module)
        for blocker in fs.date_blockers(resolved, source_days):
            if blocker.text not in resolved.problems:
                show(h.note(blocker.text, "orange"), st_module)
        show(h.Markup(f'<div class="lab fs-protect">{h.esc(PROTECTED)}</div>'), st_module)
    return choices, resolved


def _every_date(st_module, resolved) -> None:
    from ifvg_lab_ui import show

    days = ", ".join(fmt.date_short(d) + (f" {d[:4]}" if d[:4] != resolved.end[:4] else "")
                     for d in resolved.trading_days)
    warmup = ", ".join(fmt.date_long(d) for d in resolved.warmup_dates)
    body = [f'<div class="fs-muted"><b>Warmup (stored market days replayed first, not '
            f'evaluated):</b> '
            f'{h.esc(warmup)}</div>',
            f'<div style="font-size:14px;line-height:1.6;margin-top:6px">{h.esc(days)}</div>']
    if resolved.excluded:
        rows = [h.Row({"day": fmt.date_long(d), "why": reason}) for d, reason in resolved.excluded]
        body.append('<div class="fs-muted" style="margin-top:10px">Left out, with the reason'
                    '</div>')
        body.append(h.table([h.Column("day", "Day", width="180px"),
                             h.Column("why", "Reason")], rows, plain=True, wrap=False))
    show(h.Markup('<div class="lab">' + "".join(body) + "</div>"), st_module)


def _chip(axis: str, value: str, *, removable: bool) -> str:
    label = fs.value_label(axis, value)
    mono = " mono" if axis == fs.TRIGGER_AXIS else ""
    button = (f'<button type="button" class="fs-x" data-action="remove|{axis}|{h.esc(value)}" '
              f'aria-label="Remove {h.esc(label)}">×</button>') if removable else ""
    return (f'<div class="fs-chip{mono}{"" if removable else " solo"}">{h.esc(label)}'
            f"{button}</div>")


def _axis_block(page: _Setup, choices: fs.SetupChoices, axis: str, adding: str | None, *,
                new_here: bool = False, notes: tuple[str, ...] = (),
                blocker: str | None = None) -> str:
    values = choices.values(axis)
    options = page.offered.get(axis, ())
    chips = [_chip(axis, v, removable=len(values) > 1) for v in values]
    remaining = [v for v in options if v not in values]
    if remaining:
        chips.append(f'<button type="button" class="fs-add" data-action="add|{axis}">+ Add'
                     "</button>")
    head = f"<div>{h.esc(fs.axis_title(axis))}</div>"
    if new_here:
        head += h.badge("New here", "orange")
    parts = [f'<div class="fs-axis"><div class="fs-axis-head">{head}</div>'
             f'<div class="fs-chips">{"".join(chips)}</div>']
    if adding == axis and remaining:
        picks = "".join(f'<button type="button" class="fs-option" '
                        f'data-action="pick|{axis}|{h.esc(v)}">+ {h.esc(fs.value_label(axis, v))}'
                        "</button>" for v in remaining)
        parts.append(f'<div class="fs-options"><span>Add:</span>{picks}<button type="button" '
                     'class="fs-link" data-action="cancel">Done</button></div>')
    for note in notes:
        parts.append(f'<div class="fs-muted">{h.esc(note)}</div>')
    if blocker:
        parts.append(f'<div class="lab-note orange" style="font-size:14px">{h.esc(blocker)}</div>')
    parts.append("</div>")
    return "".join(parts)


def _settings_card(st_module, page: _Setup, choices: fs.SetupChoices) -> None:
    adding = st_module.session_state.get(SETUP + "adding")
    blockers = {b.key: b.text for b in fs.setup_blockers(choices, base_ids=page.base_ids)}
    blocks = [
        _axis_block(page, choices, "enabled_entry_sessions", adding),
        _axis_block(page, choices, "tp_r_multiple", adding),
        _axis_block(page, choices, fs.GAP_AXIS, adding, new_here=True,
                    blocker=blockers.get("gap_rule")),
    ]
    if page.offered.get("exit_policy"):
        blocks.append(_axis_block(page, choices, "exit_policy", adding))
    else:
        blocks.append(
            '<div class="fs-axis"><div class="fs-axis-head"><div>Exit</div></div>'
            f'<div class="fs-chips">{_chip("exit_policy", fs.WHOLE_EXIT, removable=False)}</div>'
            f'<div class="fs-muted">{h.esc(fs.value_label("exit_policy", fs.HALF_EXIT))} needs '
            "the engine version that supports half exits. Start the application with that "
            "version to add it.</div></div>")
    blocks.append(_axis_block(page, choices, fs.TRIGGER_AXIS, adding, new_here=True,
                              notes=("Each firm's own minimum request still applies.",),
                              blocker=blockers.get("withdrawal_trigger")))
    fixed = []
    for axis in fs.FIXED_AXES:
        values = choices.values(axis)
        if len(values) > 1 or adding == axis:
            blocks.append(_axis_block(page, choices, axis, adding))
            continue
        value = fs.value_label(axis, values[0]) if values else "Not set"
        vary = (f'<button type="button" class="fs-link" data-action="vary|{axis}">Vary</button>'
                if len(page.offered.get(axis, ())) > 1 else "")
        fixed.append(f'<div><div class="fs-k">{h.esc(fs.axis_title(axis))}</div>'
                     f'<div class="fs-v">{h.esc(value)}</div>{vary}</div>')
    if fixed:
        blocks.append(f'<div class="fs-fixed">{"".join(fixed)}</div>')
    markup = h.Markup('<div class="lab lab-card" style="gap:18px"><div class="fs-title">'
                      f'Settings to compare</div>{"".join(blocks)}</div>')
    action = _clickable(st_module, markup, key="setup_settings")
    _apply(st_module, page, choices, action, adding)


def _apply(st_module, page: _Setup, choices, action, adding) -> None:
    if not action:
        return
    updated, still_adding, changed = fs.apply_action(choices, action, options=page.offered,
                                                     adding=adding)
    st_module.session_state[SETUP + "adding"] = still_adding
    if changed:
        _commit(st_module, page, updated)
    st_module.rerun()


def _firms_card(st_module, page: _Setup, choices: fs.SetupChoices) -> fs.SetupChoices:
    adding = st_module.session_state.get(SETUP + "adding")
    firms = choices.firm_keys
    chips = [_chip(fs.FIRM_AXIS, f, removable=len(firms) > 1) for f in firms]
    remaining = [f for f in page.offered.get(fs.FIRM_AXIS, ()) if f not in firms]
    if remaining:
        chips.append(f'<button type="button" class="fs-add" data-action="add|{fs.FIRM_AXIS}">'
                     "+ Add</button>")
    picks = ""
    if adding == fs.FIRM_AXIS and remaining:
        picks = ('<div class="fs-options"><span>Add:</span>' + "".join(
            f'<button type="button" class="fs-option" data-action="pick|{fs.FIRM_AXIS}|{f}">'
            f"+ {h.esc(fs.firm_chip_label(f))}</button>" for f in remaining)
            + '<button type="button" class="fs-link" data-action="cancel">Done</button></div>')
    whole, half = fs.size_lines(choices)
    half_caption = "" if choices.half_exit else (
        '<div class="fs-muted">Used only by half-exit configurations.</div>')
    boxes = (f'<div class="lab-grid" style="grid-template-columns:repeat(2,minmax(0,1fr))">'
             f'<div class="fs-soft"><div class="fs-k">Whole-position exits</div>'
             f'<div class="fs-v">{h.esc(whole)}</div></div>'
             f'<div class="fs-soft"><div class="fs-k">Half exits</div>'
             f'<div class="fs-v">{h.esc(half)}</div>{half_caption}</div></div>')
    blockers = [b.text for b in fs.setup_blockers(choices, base_ids=page.base_ids)
                if b.key in ("firms", "size", "half_size")]
    notes = "".join(f'<div class="lab-note orange" style="font-size:14px">{h.esc(t)}</div>'
                    for t in blockers)
    markup = h.Markup('<div class="lab lab-card" style="gap:14px"><div class="fs-title">'
                      f'Firms, size and costs</div><div class="fs-chips">{"".join(chips)}</div>'
                      f"{picks}{boxes}{notes}</div>")
    action = _clickable(st_module, markup, key="setup_firms")
    _apply(st_module, page, choices, action, adding)
    with st_module.expander("Change size and costs"):
        columns = st_module.columns(2)
        whole_q = columns[0].number_input(
            "Whole-position exits: E-mini Nasdaq-100 contracts", min_value=1, max_value=6,
            step=1, value=int(choices.whole_quantity), key=SETUP + "whole_q",
            on_change=_mark(SETUP, "size_edited"),
            help="Contracts per trade for configurations that exit the whole position.")
        whole_c = columns[1].number_input(
            "Cost per E-mini contract per fill (US dollars)", min_value=0.0, step=0.001,
            format="%.3f", value=choices.whole_cost_mills / 1000, key=SETUP + "whole_c",
            on_change=_mark(SETUP, "size_edited"),
            help="Modeled cost charged on every fill of one E-mini contract.")
        half_q = columns[0].number_input(
            "Half exits: Micro E-mini Nasdaq-100 contracts (even)", min_value=2,
            max_value=60, step=2, value=int(choices.half_quantity), key=SETUP + "half_q",
            on_change=_mark(SETUP, "size_edited"),
            help="Micro contracts per trade for half-exit configurations; half exit at the "
                 "target.")
        half_c = columns[1].number_input(
            "Cost per micro contract per fill (US dollars)", min_value=0.0, step=0.001,
            format="%.3f", value=choices.half_cost_mills / 1000, key=SETUP + "half_c",
            on_change=_mark(SETUP, "size_edited"),
            help="Modeled cost charged on every fill of one micro contract.")
    with st_module.expander("Firm terms, in full"):
        from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

        for profile in FIRM_PROFILES.values():
            st_module.markdown(fs.firm_terms_sentence(profile).replace("$", "\\$"))
    if st_module.session_state.pop(SETUP + "size_edited", False):
        new = replace(choices, whole_quantity=int(whole_q),
                      whole_cost_mills=round(float(whole_c) * 1000),
                      half_quantity=int(half_q), half_cost_mills=round(float(half_c) * 1000))
        if new != choices and _commit(st_module, page, new):
            st_module.rerun()
    return choices


def _more(st_module, page: _Setup | None, roots) -> None:
    """Rule 12: the earlier configurator stays reachable from this page."""

    with st_module.expander("More"):
        if page is not None:
            from alpha_lab.agents.data_infra.ifvg.presentation.workspace import human_name

            name = st_module.text_input(
                "Study name", value=page.draft.display_name, key=SETUP + "name",
                on_change=_mark(SETUP, "name_edited"),
                help="How this study is listed under My studies.")
            if st_module.session_state.pop(SETUP + "name_edited", False):
                cleaned = name.strip()
                if not cleaned or human_name(cleaned, "") != cleaned:
                    st_module.error("Use a descriptive study name without technical "
                                    "identifiers or paths.")
                elif cleaned != page.draft.display_name:
                    page.draft.display_name = cleaned
                    _commit(st_module, page, fs.choices_from_draft(
                        page.draft, base_ids=page.base_ids, offered=page.offered))
        st_module.write(
            "The earlier configurator offers both plan kinds — “Configurations from the "
            "completed strategy study” and “Variations around one configuration” — with every "
            "earlier option, including the contract and payout-processing choices. Settings "
            "only this page has (dates, gap rules, withdrawal triggers, pass/fail checks) stay "
            "saved in the draft. The earlier page doesn't show them, and it refuses approval "
            "and launch while any of them blocks approval here.")
        if st_module.button(
                "Open the earlier configurator", key=SETUP + "classic",
                help="Opens this comparison on the earlier configurator page. A new, unsaved "
                     "comparison is saved as a draft when that page opens it."):
            if page is not None:
                st_module.session_state[DRAFT_KEY] = page.draft.draft_id
                if not page.persisted:
                    st_module.session_state[SESSION_DRAFT_KEY] = asdict(page.draft)
            _go(st_module, screen="new")


def _plan_panel(st_module, page: _Setup, choices: fs.SetupChoices, resolved,
                continue_to_review) -> None:
    from ifvg_lab_ui import show

    from alpha_lab.propsim.funded.comparison_study import variation_variants

    lines = fs.plan_lines(choices)
    gaps, triggers = len(choices.values(fs.GAP_AXIS)), len(choices.triggers)
    firms = len(choices.firm_keys)
    count_html, skipped_html = "", ""
    try:
        variants, skipped = variation_variants(
            page.source, choices.base, fs.strategy_selections(choices),
            whole={"instrument": "mini", "quantity": choices.whole_quantity,
                   "cost_per_contract_mills": choices.whole_cost_mills},
            scale_out={"instrument": "micro", "quantity": choices.half_quantity,
                       "cost_per_contract_mills": choices.half_cost_mills})
        count = _choices_count(page, choices)
        total = count.total if count.total is not None else (
            len(variants) * max(gaps, 1) * max(triggers, 1))
        engine_words = (f'<div style="font-size:14px;color:{css_var("body_2")}">'
                        f"{h.esc(count.engine_words)}</div>" if count.engine_words else "")
        count_html = (f'<div class="fs-plan-count">{h.esc(fmt.count(total, "configuration"))}'
                      f'</div>{engine_words}<div style="font-size:14px;'
                      f'color:{css_var("body_2")}">× '
                      f'{h.esc(fmt.count(firms, "firm"))} = '
                      f'{h.esc(fmt.count(total * firms, "separate result"))}</div>')
        sentence = fs.skipped_sentence(skipped, max(gaps, 1) * max(triggers, 1))
        if sentence:
            skipped_html = f'<div class="fs-muted">{h.esc(sentence)}</div>'
    except Exception as error:  # a combination this engine cannot resolve
        count_html = (f'<div class="fs-muted">{h.placeholder("Not counted")} The plan could not '
                      f"be prepared here: {h.esc(error)}</div>")
    blockers = (fs.setup_blockers(choices, base_ids=page.base_ids,
                                  named_problem=page.named_problem)
                + fs.date_blockers(resolved, page.source.evaluation_dates))
    engine = h.note(ENGINE_NOTE, "blue") if choices.half_exit else ""
    blocked = (h.note(f"Approval is off until {fmt.count(len(blockers), 'setting')} "
                      f"{'is' if len(blockers) == 1 else 'are'} settled — see the notes on "
                      "the left. Saving is still available.", "orange") if blockers else "")
    plan_lines = "".join(f"<div>{h.esc(line)}</div>" for line in lines)
    markup = h.Markup(
        '<div class="lab" style="display:flex;flex-direction:column;gap:14px">'
        '<div class="lab-card-title">This plan</div>'
        f'<div class="fs-plan-lines">{plan_lines}</div>'
        f'<div style="border-top:1px solid {css_var("light_rule")};padding-top:12px;display:flex;'
        f'flex-direction:column;gap:4px">{count_html}{skipped_html}</div>{engine}{blocked}</div>')
    with st_module.container(key="ifvg_lab_card_setup_plan"):
        show(markup, st_module)
        if st_module.button("Continue to review", type="primary", key=SETUP + "continue",
                            width="stretch",
                            help="Saves this draft and shows the plan for review. Nothing runs "
                                 "until you approve it there."):
            continue_to_review()
        if st_module.button("Save draft", key=SETUP + "save", width="stretch",
                            help="Keeps these settings under My studies → Drafts."):
            if page.persisted:
                st_module.session_state[SETUP + "saved_note"] = (
                    "Every change is saved as you make it.")
            elif _create(st_module, page):
                st_module.session_state[SETUP + "saved_note"] = (
                    "Saved. It's listed under My studies → Drafts.")
                st_module.rerun()
        note = st_module.session_state.get(SETUP + "saved_note")
        if note:
            st_module.caption(note)
        elif page.persisted:
            st_module.caption("Saved draft · changes are saved as you make them.")
        else:
            st_module.caption("Not saved yet.")


def _choices_count(page: _Setup, choices: fs.SetupChoices) -> fs.PlanCount:
    """The shared count (``funded_setup.plan_count``) of the choices on screen.

    The choices go into a copy of the draft in memory, exactly as a save would write
    them, so this page, My studies and Review and approve count the same draft the
    same way. Nothing is written.
    """

    import copy

    from alpha_lab.propsim.funded.comparison_draft import (
        check_saved_comparison,
        saved_settings,
        saved_study_selections,
    )

    draft = copy.deepcopy(page.draft)
    fs.write_choices(draft, choices, offered=page.offered)
    check = check_saved_comparison(saved_settings(draft), saved_study_selections(draft),
                                   {page.source.package.run_id: page.source})
    return fs.plan_count(check, draft)


# ── review page ───────────────────────────────────────────────────────────


@dataclass
class _Review:
    draft: Any
    settings: dict[str, Any]
    check: Any
    source: Any
    base_ids: dict[str, str]
    offered: dict[str, tuple[str, ...]]
    choices: fs.SetupChoices
    resolved: Any
    envelope: Any
    build_problem: str | None
    rows: list[dict[str, str]]
    gate_defaults: dict[str, Any]
    #: combinations the saved selections name but the plan leaves out, with the reason
    #: (correction A11: explained before approval, never dropped silently)
    skipped: tuple = ()


def _order_options(offered) -> dict[str, tuple[str, ...]]:
    return {"exit_policy": (fs.WHOLE_EXIT, fs.HALF_EXIT), **offered}


def _review_state(roots, draft, sources) -> _Review:
    """Everything the review shows, from one draft as saved (nothing written)."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (
        package_gate_thresholds,
    )
    from alpha_lab.propsim.funded.comparison_draft import (
        DEFAULT_SETTINGS,
        DEFAULT_VARIATION,
        check_saved_comparison,
        rebuild_saved_plan,
        saved_settings,
        saved_study_selections,
    )

    settings = saved_settings(draft)
    selections = saved_study_selections(draft)
    check = check_saved_comparison(settings, selections, sources,
                                   store_root=Path(roots["store_root"]))
    merged = {**DEFAULT_SETTINGS, "plan_kind": "study", **settings}
    source = sources.get(merged.get("source_run_id"))
    variation = {**DEFAULT_VARIATION, **(settings.get("variation") or {})}
    base = variation.get("base")
    base_ids = dict(source.by_name[base].axis_value_ids) if (
        source is not None and base in source.by_name) else {}
    offered = fs.offered_options(base_ids)
    choices = fs.choices_from_draft(draft, base_ids=base_ids, offered=offered)
    envelope, problem, skipped = None, None, ()
    if check.runnable and source is not None:
        from alpha_lab.propsim.funded.core_identity import core_source_identity

        try:
            core_source_identity()
        except Exception:
            problem = ("The strategy engine this application imports is not a verifiable "
                       "source checkout, so a variation plan cannot freeze it. Start the "
                       "application with an explicitly prepared checkout "
                       "(run_ifsm_research_ui.py --core).")
        if problem is None:
            try:
                envelope, skipped, _unavailable = rebuild_saved_plan(source, settings,
                                                                     selections)
            except Exception as error:
                problem = f"The plan could not be prepared: {error}"
            if envelope is None and problem is None:
                problem = "No combination can run with these saved selections."
    if envelope is not None:
        strategy = fs.rows_from_plan(envelope.payload.variants, base_ids)
    else:
        strategy = fs.saved_strategy_rows(source, variation) or []
    rows = fs.review_rows(strategy, choices, _order_options(offered))
    resolved = None
    if source is not None:
        days = source.evaluation_dates
        from alpha_lab.agents.data_infra.ifvg.research_period import (
            local_day_has_data,
            resolve_research_range,
        )

        resolved = resolve_research_range(
            choices.start or days[0], choices.end or days[-1],
            day_has_data=local_day_has_data(Path(roots.get("repo_root") or Path.cwd())))
    defaults = fs.gate_defaults(package_gate_thresholds(
        source.package.root if source is not None else None))
    return _Review(draft=draft, settings=settings, check=check, source=source,
                   base_ids=base_ids, offered=offered, choices=choices, resolved=resolved,
                   envelope=envelope, build_problem=problem, rows=rows, gate_defaults=defaults,
                   skipped=tuple(skipped or ()))


def _review_blockers(state: _Review, gate_rows) -> list[fs.Blocker]:
    out: list[fs.Blocker] = []
    if state.build_problem:
        out.append(fs.Blocker("plan", state.build_problem))
    _named_baseline, named_problem = _named()
    out += fs.setup_blockers(state.choices, base_ids=state.base_ids, named_problem=named_problem)
    if state.source is not None:
        out += fs.date_blockers(state.resolved, state.source.evaluation_dates)
    out += fs.gate_blockers(gate_rows)
    return out


def _scope(state: _Review) -> str:
    """The approval scope text, worded as the earlier variation page words it."""

    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    plan = state.envelope.payload
    firm_keys = state.choices.firm_keys
    firms = " and ".join(FIRM_PROFILES[k].firm_name for k in FIRM_PROFILES if k in firm_keys)
    halves = sum(1 for v in plan.variants if v.exit_policy != "fixed_target_v1")
    base = state.source.by_name[plan.base_configuration].display_name.replace(" | ", "; ")
    return (f"{len(plan.variants)} configurations around {base} x {firms} = "
            f"{len(plan.variants) * len(firm_keys)} separate results; "
            f"{len(plan.variants) - halves} whole-position at "
            f"{state.choices.whole_quantity} x NQ, {halves} half exits at "
            f"{state.choices.half_quantity} x MNQ")


def render_funded_review(st_module, roots) -> None:
    from ifvg_lab_ui import show

    from alpha_lab.agents.data_infra.ifvg.study_drafts import load_draft

    link_note = _apply_draft_link(st_module, roots)
    _style(st_module)
    root = Path(roots["draft_root"])
    _header(st_module, "Review and approve", step=2, question=False)
    if link_note:
        h_note(st_module, link_note, "orange")
    draft_id = st_module.session_state.get(DRAFT_KEY)
    if not draft_id:
        h_note(st_module, "Nothing to review yet: set up a funded comparison and continue to "
                          "review. Review always shows a saved draft.")
        if st_module.button("Set up a funded comparison", key=REVIEW + "to_setup",
                            help="Opens the setup page."):
            _go(st_module, screen="new_funded")
        return
    try:
        draft = load_draft(root, draft_id)
    except Exception:
        h_note(st_module, "This draft hasn't been saved yet, or it could not be read. Save it "
                          "on the setup page first.")
        return
    if draft.mode_id != fs.COMPARISON_MODE:
        h_note(st_module, "This draft isn't a funded comparison. Open it from My studies.")
        return
    if draft.archived or draft.status == "frozen":
        h_note(st_module, "This study is saved as history. Return to My studies to restore or "
                          "clone it.")
        return
    _sync(st_module, REVIEW, root, draft_id)
    sources = _cached_sources()
    state = _review_state(roots, draft, sources)
    kind = {"plan_kind": "study", **state.settings}["plan_kind"]
    if kind != "variations":
        if state.check.runnable:
            h_note(st_module, "This draft compares configurations from the completed strategy "
                              "study; it's reviewed and approved on the earlier configurator.")
            if st_module.button("Open it on the earlier configurator", key=REVIEW + "classic",
                                help="Shows this draft on the earlier configurator page."):
                _go(st_module, screen="new")
        else:
            from ifvg_funded_comparison_study import _render_saved_read_only

            _render_saved_read_only(
                st_module, state.check, roots, words=fmt.display_words,
                lead=fs.blocked_engine_lead(fs.plan_count(state.check, state.draft),
                                            state.check.count_is_exact,
                                            state.check.needs_half_exit_engine))
        return
    editable = state.check.runnable
    # the one shared count (My studies, setup and here); the table lists state.rows
    count = fs.plan_count(state.check, state.draft)
    total = count.total if count.total is not None else len(state.rows)
    if not editable:
        lead = fs.blocked_engine_lead(count, state.check.count_is_exact,
                                      state.check.needs_half_exit_engine)
        rest = ("Your saved settings haven't been changed. Approval and launch are off until the "
                "study is opened with that version. The plan below is shown exactly as saved."
                if state.check.needs_half_exit_engine else
                "Your saved settings haven't been changed. Approval and launch are off in this "
                "application. The plan below is shown exactly as saved. "
                + " ".join(fmt.display_words(p) for p in state.check.problems))
        show(_alert(lead, rest, icon="lock"), st_module)
    _review_tiles(st_module, state, total, count.engine_words)
    _variation_card(st_module, state)
    rows = _gates_card(st_module, state, root, editable)
    blockers = _review_blockers(state, rows) if editable else []
    left_out = fs.skipped_sentence(state.skipped, max(len(state.choices.values(fs.GAP_AXIS)), 1)
                                   * max(len(state.choices.triggers), 1))
    if left_out:  # before approval, say what the plan leaves out and why (correction A11)
        h_note(st_module, f"{left_out} They are not in this plan.")
    _approval_card(st_module, roots, state, blockers, editable, total)
    _review_more(st_module, state, roots, editable)


def _text_tile(label: str, value: str, caption: str | None = None) -> h.Markup:
    cap = f'<div class="lab-tile-caption">{h.esc(caption)}</div>' if caption else ""
    return h.Markup(f'<div class="lab fs-texttile"><div class="lab-tile-label">{h.esc(label)}'
                    f'</div><div class="v">{h.esc(value)}</div>{cap}</div>')


def _review_tiles(st_module, state: _Review, total: int, engine_words: str = "") -> None:
    from ifvg_lab_ui import show

    firms = len(state.choices.firm_keys)
    days = state.resolved.trading_days if state.resolved is not None else ()
    dates = (fmt.date_range(days[0], days[-1]) if days
             else "No trading days resolved")
    base = state.choices.base or "Not saved"
    starting = (fs.LEGACY_LABEL.split(" — ")[0] if state.choices.baseline == fs.LEGACY else base)
    caption = ("Holds across the daily close" if state.choices.baseline == fs.LEGACY
               else fs.baseline_caption(state.base_ids))
    show(h.grid([
        h.tile("Configurations", f"{total:,}", engine_words or None),
        h.tile("Separate results", f"{total * firms:,}", f"{total:,} × {fmt.count(firms, 'firm')}"),
        _text_tile("Dates", dates, fmt.count(len(days), "trading day")),
        _text_tile("Starting configuration", starting, caption),
    ], 4), st_module)


def _variation_card(st_module, state: _Review) -> None:
    rows = state.rows
    axes = fs.varying_axes(rows)
    show_all = bool(st_module.session_state.get(REVIEW + "show_all"))
    visible = rows if show_all else rows[:8]
    parts = []
    if not rows:
        parts.append(h.placeholder("No configuration can be listed from these saved settings."))
    elif not axes:
        parts.append(h.Markup('<div class="lab-line">One configuration: every setting is listed '
                              "under “The same for every configuration”.</div>"))
    else:
        columns = [h.Column("n", "#", mono=True, width="50px")] + [
            h.Column(a, fs.column_title(a), mono=a == fs.TRIGGER_AXIS) for a in axes]
        table_rows = [h.Row({"n": str(i), **{a: fs.short_label(a, row.get(a, "")) if row.get(a)
                                            else "" for a in axes}})
                      for i, row in enumerate(visible, start=1)]
        parts.append(h.table(columns, table_rows, plain=True, wrap=False))
        if len(rows) > 8:
            link = (h.link("Show fewer", "rows|fewer") if show_all
                    else h.link(f"Show all {len(rows):,}", "rows|all"))
            shown = (f"Showing all {len(rows):,}" if show_all
                     else f"Showing 8 of {len(rows):,}")
            parts.append(h.Markup(f'<div class="fs-foot"><span>{h.esc(shown)}</span>{link}</div>'))
    same = fs.same_for_every(rows, state.base_ids, state.choices)
    grid = "".join(f"<div>{h.esc(item)}</div>" for item in same)
    opened = "" if axes else " open"
    parts.append(h.Markup(f'<details class="fs-same"{opened}><summary>The same for every '
                          f'configuration</summary><div class="fs-same-grid">{grid}</div>'
                          "</details>"))
    markup = h.card(h.Markup("".join(parts)), title="What changes between configurations",
                    right="Only settings that differ are shown here")
    action = _clickable(st_module, markup, key="review_variations")
    if action in ("rows|all", "rows|fewer"):
        st_module.session_state[REVIEW + "show_all"] = action == "rows|all"
        st_module.rerun()


def _gates_card(st_module, state: _Review, root: Path, editable: bool):
    from ifvg_lab_ui import show

    from alpha_lab.agents.data_infra.ifvg.study_drafts import save_draft

    keys = {spec.key: f"{REVIEW}gate_{spec.key}" for spec in fs.GATES}
    typed = {k: str(st_module.session_state[w]) for k, w in keys.items()
             if w in st_module.session_state}
    edited = [k for k in keys if st_module.session_state.pop(f"{REVIEW}edited_{k}", False)]
    if edited and editable:
        gates = dict(state.choices.gates)
        for key in edited:
            value, error = fs.parse_gate(key, typed.get(key, ""))
            if error is None:
                gates[key] = value
        if gates != state.choices.gates:
            state.choices = replace(state.choices, gates=gates)
            offered = state.offered
            fs.write_choices(state.draft, state.choices, offered=offered)
            try:
                save_draft(root, state.draft)
                st_module.session_state[REVIEW + "marker"] = [
                    state.draft.draft_id, _digest(root, state.draft.draft_id)]
            except Exception:
                st_module.error("The check could not be saved. Keep this page open and try "
                                "again.")
    days = state.resolved.trading_days if state.resolved is not None else ()
    warmup = state.resolved.warmup_dates if state.resolved is not None else ()
    rows = fs.gate_rows(state.choices.gates, state.gate_defaults, trading_days=days,
                        warmup_days=warmup, typed=typed)
    flagged = [row.spec.key for row in rows if row.status in ("flag", "decision", "invalid")]
    if flagged:
        css = ",".join(f'.st-key-{keys[k]} [data-baseweb="input"]>div,'
                       f'.st-key-{keys[k]} [data-baseweb="select"]>div' for k in flagged)
        st_module.html(f"<style>{css}{{border:2px solid {css_var('orange')} !important}}"
                       "</style>")
    with st_module.container(key="ifvg_lab_card_review_gates"):
        show(h.Markup('<div class="lab"><div class="lab-card-title" style="margin-bottom:4px">'
                      'Pass/fail checks for the ranking</div><div class="fs-grid-head">'
                      "<div>Check</div><div>Your value</div><div>Validation</div></div></div>"),
             st_module)
        for row in rows:
            with st_module.container(key=f"{REVIEW}gaterow_{row.spec.key}"):
                label, field, message = st_module.columns([1, 0.42, 1.2],
                                                          vertical_alignment="center")
                label.html(f'<div class="fs-gate-label">{h.esc(row.spec.label)}</div>')
                help_text = (f"Saved in this draft. The source study saved "
                             f"{fs.format_gate(row.spec, state.gate_defaults.get(row.spec.key))}"
                             if state.gate_defaults.get(row.spec.key) is not None
                             else "Saved in this draft.")
                if row.spec.kind == "choice":
                    current = row.text if row.text in fs.DROP_CHOICES else fs.DROP_CHOICES[0]
                    field.selectbox(row.spec.label, fs.DROP_CHOICES,
                                    index=fs.DROP_CHOICES.index(current), key=keys[row.spec.key],
                                    label_visibility="collapsed", disabled=not editable,
                                    on_change=_mark(REVIEW, f"edited_{row.spec.key}"),
                                    help=help_text)
                else:
                    field.text_input(row.spec.label, value=row.text, key=keys[row.spec.key],
                                     label_visibility="collapsed", disabled=not editable,
                                     on_change=_mark(REVIEW, f"edited_{row.spec.key}"),
                                     help=help_text)
                klass = "fs-ok" if row.status == "ok" else "fs-bad"
                message.html(f'<div class="{klass}">{h.esc(row.message)}</div>')
        show(h.Markup('<div class="lab fs-muted" style="padding-top:6px">The source study\'s '
                      "other saved checks (result per trade, session stability, time-block "
                      "consistency, best setup's share) stay as saved. Every check is "
                      "evaluated on the strategy replay without accounts.</div>"), st_module)
    return rows


def _approval_card(st_module, roots, state: _Review, blockers, editable: bool,
                   total: int) -> None:
    from ifvg_funded_comparison_study import comparison_state_root, find_approval
    from ifvg_lab_ui import show

    from alpha_lab.propsim.funded.runner import read_state

    approval, status = None, None
    if editable and state.envelope is not None:
        plan_id = state.envelope.funded_comparison_plan_id
        approval = find_approval(Path(roots["store_root"]), plan_id)
        run = read_state(comparison_state_root(roots), plan_id) or {}
        if run.get("status") in ("Running", "Completed"):
            status = run["status"]
            blockers = [*blockers, fs.Blocker("already", (
                f"This exact plan is already {status.lower()}; its saved result is shown under My "
                "studies and is never recomputed."))]
    if blockers:
        show(_alert("Approval is off until these are settled.", items=[b.text for b in blockers],
                    small=True), st_module)
    blocked = (not editable) or bool(blockers)
    with st_module.container(key="ifvg_lab_card_review_approval"):
        title, box, button = st_module.columns([0.15, 1, 0.3], vertical_alignment="center")
        title.html('<div class="fs-title">Approval</div>')
        if approval is not None:
            box.caption(f"Approved on {fmt.date_long(approval.payload.approved_on)}. Running "
                        "checks the saved draft and this approval again.")
        agree = box.checkbox(
            fs.approval_sentence(total, state.choices.firm_keys), key=REVIEW + "agree",
            disabled=blocked,
            help="Your approval covers this exact plan only. Changing any setting makes a "
                 "different plan that needs its own approval.")
        clicked = button.button(
            "Record approval and run", type="primary", key=REVIEW + "run",
            disabled=blocked or not agree, width="stretch",
            help="Records your approval of this exact plan, then starts it on the saved "
                 "historical dates through the gated launch. No live trading.")
    if clicked and not blocked and agree:
        _record_and_run(st_module, roots, state)


def _record_and_run(st_module, roots, state: _Review) -> None:
    """The existing gated path: approval of the exact plan, then the checked launch."""

    import ifvg_funded_comparison_study as screen

    from alpha_lab.agents.data_infra.ifvg.study_drafts import load_draft
    from alpha_lab.propsim.funded.comparison_draft import (
        rebuild_saved_plan,
        saved_settings,
        saved_study_selections,
    )

    root = Path(roots["draft_root"])
    draft_id = state.draft.draft_id
    if st_module.session_state.get(REVIEW + "marker") != [draft_id, _digest(root, draft_id)]:
        st_module.error("This page is out of date: the saved draft changed. Refresh the page "
                        "and review the plan again. Nothing was approved or launched.")
        return
    disk = load_draft(root, draft_id)
    fresh = _review_state(roots, disk, _cached_sources())
    rows = fs.gate_rows(fresh.choices.gates, fresh.gate_defaults,
                        trading_days=fresh.resolved.trading_days if fresh.resolved else (),
                        warmup_days=fresh.resolved.warmup_dates if fresh.resolved else ())
    problems = _review_blockers(fresh, rows) if fresh.check.runnable else [
        fs.Blocker("engine", "The saved draft cannot run in this application.")]
    envelope = state.envelope
    if problems or fresh.envelope is None or (
            fresh.envelope.funded_comparison_plan_id != envelope.funded_comparison_plan_id):
        st_module.error("The saved draft no longer describes an approvable plan: "
                        + " ".join(p.text for p in problems[:3])
                        + " Nothing was approved or launched.")
        return
    store = Path(roots["store_root"])
    plan_id = envelope.funded_comparison_plan_id
    if screen.find_approval(store, plan_id) is None:
        rebuilt, _skipped, _unavailable = rebuild_saved_plan(
            fresh.source, saved_settings(disk), saved_study_selections(disk))
        if rebuilt is None or rebuilt.funded_comparison_plan_id != plan_id:
            st_module.error("This page is out of date: the saved draft no longer describes the "
                            "plan shown. Nothing was approved or launched.")
            return
        screen.save_plan(store, envelope)
        screen.record_owner_approval(store, plan_id, approved_on=date.today().isoformat(),
                                     channel="study_screen", statement=_STATEMENT,
                                     scope=_scope(fresh))
    screen._freeze_and_launch(st_module, disk, envelope, roots)


def _review_more(st_module, state: _Review, roots, editable: bool) -> None:
    from ifvg_lab_ui import show

    with st_module.expander("More"):
        if not editable:
            from ifvg_funded_comparison_study import _render_saved_read_only

            _render_saved_read_only(
                st_module, state.check, roots, words=fmt.display_words,
                lead=fs.blocked_engine_lead(fs.plan_count(state.check, state.draft),
                                            state.check.count_is_exact,
                                            state.check.needs_half_exit_engine))
            return
        show(h.Markup('<div class="lab fs-title" style="font-size:15px">Saved settings exactly '
                      "as stored</div>"), st_module)
        show(h.kv_table(state.check.saved_rows), st_module)
        if state.envelope is not None:
            plan = state.envelope.payload
            st_module.caption(f"Strategy engine frozen into this plan: "
                              f"{plan.core_source.description}. The run uses exactly this "
                              "source and refuses any other.")
        for plan_id in state.check.matching_plan_ids:
            st_module.caption(f"A saved plan matches these settings exactly ({plan_id[:12]}…).")
        if st_module.button("Back to setup", key=REVIEW + "back",
                            help="Returns to the setup page for this draft."):
            _go(st_module, screen="new_funded")
