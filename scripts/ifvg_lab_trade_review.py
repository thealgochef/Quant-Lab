"""Trade review (mocks 09, 09b) — one page for every kind of trade review.

Source switch: **Funded trades** (built here: one recorded funded trade of a
saved funded comparison, on the stored one-minute E-mini bars, with the setup
zones of its linked saved setup record, point in time, and the review form),
**Strategy trades** (the existing saved-study executions reviewer),
**Verified context** (the existing verified-context reviewer) and **Setups not
taken** (that reviewer with its "Strategy setups" selection chosen). The three
existing reviewers are reused unchanged.

Funded trades read the saved result and the plan's verified strategy package
only. The firm, configuration and account are the ONE shared selection of the
saved result (``ifvg_lab_nav.funded_context``): Trade review, the ranking and
every detail tab show the same firm. Reviews are appended to the existing
review ledger under the existing funded keys (result, configuration-and-firm
pair, account and strategy trade), after the saved result is verified again —
they can never land on another firm's, account's or trade's review.

Analytical corrections (September 25, 2026):

- A6: another configuration's saved setup record of the same entry is shown as
  "Related setup context" only; it never names this configuration's formation
  history, its first review step or point-in-time moments; the gap and parent
  review steps are disabled for it (nothing is saved for them); and the recorded
  fills, stops, exits, result and balances are shown exactly as without any record.
- A7: point in time lists only accounts that had a trade entered by the moment
  (and the reviewed trade's own account) and trades entered by the moment; choosing
  another trade or account rebuilds every picker from THAT trade's own moment. It
  draws the scheduled 5:00 PM → 4:00 PM day (never the last stored bar), hides every
  full-history statistic (the five-largest-trades share), a later loss-limit check,
  a replacement account and earlier reviews, and opens without a link on the saved
  plan's first configuration (never the full-history leader). "›" stays enabled so
  it can't tell whether a later trade exists; pressing it (or "Save and next trade")
  is a deliberate step forward in time: it opens the next trade at that trade's own,
  later moment, or, at the last trade, moves the clock to the end of the study
  (Full history) and says so.
- A8: the review form starts empty; an earlier review's separate entry and stop
  judgments are listed as saved, and a step left "Not reviewed" saves nothing
  (unknown), never "incorrect".
- A10: the price-evidence line links the published, hash-checked
  ``approximated_minutes.csv`` of this exact result to this exact trade
  (``minute_companion``), or says why it can't.
"""

from __future__ import annotations

import hashlib
from datetime import time as clock_time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab import html as h
from alpha_lab.agents.data_infra.ifvg.presentation.lab import review_chart as rc
from alpha_lab.agents.data_infra.ifvg.presentation.lab import review_panels as rp
from alpha_lab.agents.data_infra.ifvg.presentation.lab.setup_records import (
    NO_RECORD,
    find_setup_record,
    load_setup_record_source,
)
from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import css_var

__all__ = ["CURRENT", "SOURCES", "configuration_labels", "render_trade_review_page",
           "save_funded_review"]

SOURCES = ("Funded trades", "Strategy trades", "Verified context", "Setups not taken")
MODES = ("Full history", "Point in time")
_P = "ifvg_lab_v1_review_"
_SOURCE = _P + "source_value"
_LAST_SOURCE = _P + "source_last"
_PENDING = _P + "pending"
_BACK = _P + "back"
_MODE = _P + "mode_value"
_FLASH = _P + "flash"
#: a one-run note about a step in time the page took (shown above the pickers)
_STEP_NOTE = _P + "step_note"
END_OF_STUDY_NOTE = ("There is no later recorded trade for this configuration at this firm: the "
                     "clock moved to the end of the study, so everything is shown.")
_NEXT_HELP_POINT = ("Next trade of this configuration and firm. In point in time this moves the "
                    "clock forward: to that trade, or to the end of the study when there is "
                    "none.")
#: what Trade review shows now (for the page address: trade, account, point in time)
CURRENT = _P + "current"
_STUDY = _P + "study"
_SELECTION_MODE = "ifvg_context_v1_selection_mode"  # the verified-context reviewer's mode radio
#: the verified-context reviewer's queued exact jump (ifvg_verifier_tab.queue_jump)
_JUMP = "ifvg_context_v1_pending_jump"
_JUMP_ROUTED = _P + "jump_routed"
_SOURCE_HELP = ("Funded trades: one recorded trade of a saved funded comparison. Strategy "
                "trades: executions of saved strategy studies. Verified context: the verified "
                "context reviewer. Setups not taken: that reviewer's strategy setups, including "
                "those that produced no trade.")
_NEW_TAG_HELP = ("New tags can't be saved yet: the review ledger accepts only its fixed list of "
                 "tags, and adding free-form tags would change its format.")

_CSS = """
<style>
.st-key-ifvg_lab_review_options { background:var(--lab-panel); border:1px solid var(--lab-rule);
  border-radius:10px; padding:8px 14px; }
.st-key-ifvg_lab_review_options [data-testid="stCheckbox"] label p { font-size:13px;
  color:var(--lab-ink); }
.st-key-ifvg_lab_review_pickers [data-testid="stWidgetLabel"] p { font-size:12px; }
.st-key-ifvg_lab_review_pickers [data-baseweb="select"] > div > div:first-child {
  padding-left:10px; padding-right:0; }
.st-key-ifvg_lab_review_pickers [data-baseweb="select"] > div > div:last-child {
  padding-right:6px; padding-left:0; }
.st-key-ifvg_lab_review_pickers [data-baseweb="select"] div[value] { font-size:13.5px; }
.st-key-ifvg_lab_review_options [data-baseweb="select"] div[value] { font-size:13px; }
.st-key-ifvg_lab_review_nav button { min-height:42px; padding:0 12px; }
.st-key-ifvg_lab_card_review_form [role="radiogroup"] label p { font-size:14px;
  color:var(--lab-ink); }
/* the design system's blue for chosen boxes, not the framework's default red */
.st-key-ifvg_lab_review_options label[data-baseweb="checkbox"]:has(input:checked) > span {
  background-color:var(--lab-blue) !important; border-color:var(--lab-blue) !important; }
.st-key-ifvg_lab_card_review_form label[data-baseweb="radio"]:has(input:checked) > div:first-child {
  background-color:var(--lab-blue) !important; }
.st-key-ifvg_lab_card_review_form button[data-testid="stBaseButton-pillsActive"] {
  background:var(--lab-blue-light) !important; border-color:var(--lab-blue) !important; }
.st-key-ifvg_lab_card_review_form button[data-testid="stBaseButton-pillsActive"] p {
  color:var(--lab-blue-dark) !important; }
/* a step that can't be judged here (related setup context) reads as switched off */
.st-key-ifvg_lab_card_review_form [data-testid="stSelectbox"]:has(input[disabled]) {
  opacity:0.5; }
.st-key-ifvg_lab_card_review_form [data-baseweb="input"],
.st-key-ifvg_lab_card_review_form [data-baseweb="textarea"] {
  border:1px solid var(--lab-control-border) !important; border-radius:8px !important;
  background:var(--lab-panel) !important; }
.st-key-ifvg_lab_card_review_form input, .st-key-ifvg_lab_card_review_form textarea {
  background:var(--lab-panel) !important; }
.lab-review-grid { display:grid; grid-template-columns:150px 1fr; gap:8px 12px;
  font-size:14px; line-height:1.45; }
.lab-review-grid > div:nth-child(odd) { color:var(--lab-muted); }
.lab-review-steps { display:grid; grid-template-columns:24px 1fr auto; gap:8px 10px;
  font-size:14px; align-items:baseline; }
.lab-review-key { display:grid; grid-template-columns:22px 1fr; gap:6px 8px;
  font-size:12.5px; line-height:1.45; color:var(--lab-body); }
.lab-review-key > div:nth-child(odd) { font-weight:600; }
</style>
"""


# ── cached, read-only inputs ──────────────────────────────────────────────


class _PackageUnavailableError(LookupError):
    """The verified strategy package isn't there (raised so the cache never keeps it)."""


@st.cache_resource(show_spinner=False, max_entries=4)
def _located_package_root(store_root: str, result_id: str) -> str:
    from ifvg_lab_ui import funded_study

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import market

    root = market.study_package_root(funded_study(store_root, result_id).plan)
    if root is None:
        raise _PackageUnavailableError(result_id)
    return str(root)


def _package_root(store_root: str, result_id: str) -> str | None:
    """The plan's verified strategy package, or ``None`` while it is unavailable.

    Only a found package is cached: Streamlit never caches an exception, so a
    package restored after a first miss is found on the next run without a restart.
    """

    try:
        return _located_package_root(store_root, result_id)
    except _PackageUnavailableError:
        return None


@st.cache_resource(show_spinner="Reading the saved setup records…", max_entries=4)
def _setup_source(package_root: str):
    return load_setup_record_source(package_root)


@st.cache_data(show_spinner=False, max_entries=64)
def _five_largest_share(store_root: str, result_id: str, configuration: str,
                        firm_key: str) -> float | None:
    from ifvg_lab_ui import funded_study

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import concentration

    study = funded_study(store_root, result_id)
    return concentration(study, configuration, firm_key).five_largest_trades_share


@st.cache_resource(show_spinner=False, ttl=60, max_entries=4)
def _studies(_roots: dict[str, Any], scope: str):
    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import load_studies

    studies, _issues = load_studies(_roots)
    return studies


# ── small state helpers ───────────────────────────────────────────────────


def _follow_value(ss, key: str, options: list, *, outside: Any = None, force: Any = None,
                  default: Any = None) -> Any:
    """The value a picker will show this run (see :func:`_follow`); changes nothing."""

    widget, seen = ss.get(key), ss.get(key + "__seen")
    last_outside = ss.get(key + "__outside")
    if force in options:
        return force
    if widget in options and seen is not None and widget != seen:
        return widget
    if outside in options and outside != last_outside:
        return outside
    if widget in options:
        return widget
    if outside in options:
        return outside
    return default if default in options else (options[0] if options else None)


def _follow(st_module, key: str, options: list, *, outside: Any = None, force: Any = None,
            default: Any = None) -> None:
    """Prepare a picker so it shows the right value this run.

    A value picked here wins; a value changed elsewhere (the shared firm or
    configuration, a link from the Trades tab, previous/next) is shown next; an
    option that no longer exists falls back to the default.
    """

    ss = st_module.session_state
    ss[key] = _follow_value(ss, key, options, outside=outside, force=force, default=default)
    ss[key + "__outside"] = outside


def _seen(st_module, key: str, value: Any) -> Any:
    st_module.session_state[key + "__seen"] = value
    return value


def _consume_targets(st_module, roots) -> None:
    """Links into Trade review: the redesign's (Trades tab, deep link) and the earlier ones."""

    from ifvg_lab_nav import FUNDED_TARGET, REVIEW_TARGET

    ss = st_module.session_state
    target = ss.pop(REVIEW_TARGET, None)
    legacy = ss.pop("ifvg_funded_review_pending", None)
    if legacy and not target:
        found = _target_for_plan(st_module, roots, str(legacy.get("plan_id") or ""),
                                 str(legacy.get("result_id") or ""))
        if found is not None:
            ss[FUNDED_TARGET] = found
            target = {"source": "Funded trades", "result_id": found["result_id"],
                      "configuration": legacy.get("configuration"),
                      "firm_key": legacy.get("firm_key"),
                      "account_number": legacy.get("account_number"), "back_tab": None}
        else:
            ss[_SOURCE] = "Funded trades"
            ss[_FLASH] = ("", "That funded comparison's trades can't be opened here: it is "
                              "archived or has no verified saved result. No other study was "
                              "opened in its place.")
    strategy = bool(ss.get("ifvg_search_review_pending"))
    if strategy:
        ss[_SOURCE] = "Strategy trades"  # the saved-study reviewer consumes its own link
    _route_verifier_jump(ss, consumed=bool(target or legacy or strategy))
    if not target:
        return
    ss[_SOURCE] = "Funded trades"
    ss[_PENDING] = {"result_id": target.get("result_id"),
                    "configuration": target.get("configuration"),
                    "firm_key": target.get("firm_key"),
                    "account": target.get("account_number"),
                    "seq": target.get("trade_seq")}
    back = target.get("back_tab")
    ss[_BACK] = {"result_id": target.get("result_id"), "tab": back} if back else None
    if target.get("point_in_time") is not None:
        ss[_MODE] = MODES[1] if target.get("point_in_time") else MODES[0]


def _route_verifier_jump(ss, *, consumed: bool) -> None:
    """Show the case a verifier drill-down queued (``queue_replay_drilldown``).

    'Inspect supporting trade' and 'Review selected opportunity' queue an exact
    jump for the verified-context reviewer and open Trade review; the page opens
    on "Setups not taken" for a setup jump and on "Verified context" otherwise,
    where that reviewer applies the jump itself. A funded or strategy link wins.
    Each queued jump routes once: a jump the reviewer can't resolve stays queued
    (it says so there), and the owner can still switch to another source.
    """

    jump = ss.get(_JUMP)
    if not isinstance(jump, (tuple, list)) or len(jump) != 2:
        ss.pop(_JUMP_ROUTED, None)
        return
    jump = (str(jump[0]), str(jump[1]))
    if consumed or ss.get(_JUMP_ROUTED) == jump:
        return
    ss[_SOURCE] = "Setups not taken" if jump[0] == "setup_id" else "Verified context"
    ss[_JUMP_ROUTED] = jump


def _target_for_plan(st_module, roots, plan_id: str, result_id: str) -> dict | None:
    for target in _funded_targets(st_module, roots):
        if target.get("plan_id") == plan_id or target.get("study_key") == plan_id \
                or target.get("result_id") == result_id:
            return target
    return None


def _funded_targets(st_module, roots) -> list[dict[str, Any]]:
    """Funded comparisons with a saved result this application can open (and the open one)."""

    from ifvg_lab_nav import funded_target
    from ifvg_workspace import funded_result_target

    from alpha_lab.agents.data_infra.ifvg.presentation.funded_trade_review import (
        funded_review_sources,
    )

    scope = "|".join(str(roots.get(k) or "") for k in ("store_root", "state_root", "draft_root"))
    targets = []
    for study in funded_review_sources(_studies(roots, scope)):
        target = funded_result_target(study, roots)
        if target is not None:
            targets.append(target)
    current = funded_target(st_module)
    if current and current.get("result_id") and all(
            t["result_id"] != current["result_id"] for t in targets):
        targets.insert(0, dict(current))
    return targets


def _study_label(target: dict[str, Any]) -> str:
    from ifvg_lab_funded import study_title

    return study_title(target)


# ── page ──────────────────────────────────────────────────────────────────


def render_trade_review_page(st_module, roots) -> None:
    from ifvg_lab_ui import clickable, pending_switch, show, switch

    ss = st_module.session_state
    show(h.Markup(_CSS), st_module)
    _consume_targets(st_module, roots)
    value = ss.get(_SOURCE, SOURCES[0])
    source = pending_switch("review_source", SOURCES, value, st_module)
    targets = _funded_targets(st_module, roots) if source == "Funded trades" else []
    back = _back_link(st_module, targets) if source == "Funded trades" else None
    left, right = st_module.columns([1, 1.45], vertical_alignment="bottom")
    with left:
        action = clickable(_header(back), key="review_header", st_module=st_module)
    with right, st_module.container(key="ifvg_lab_review_source_switch"):
        source = switch("Review source", SOURCES, key="review_source", value=value,
                        help=_SOURCE_HELP, st_module=st_module)
    ss[_SOURCE] = source
    if action == "back" and back:
        _go_back(st_module, back)
    if ss.get(_LAST_SOURCE) != source:
        if source == "Setups not taken":
            ss[_SELECTION_MODE] = "setup"
        elif source == "Verified context":
            ss[_SELECTION_MODE] = "candidate"
        ss[_LAST_SOURCE] = source
    if source == "Funded trades":
        _render_funded(st_module, roots, targets or _funded_targets(st_module, roots))
    elif source == "Strategy trades":
        from ifvg_search_review import render_trade_review

        render_trade_review(st_module, roots, source="Study executions", include_funded=False)
    else:
        from ifvg_lab_tab import render_ifvg_replay_tab

        render_ifvg_replay_tab(st_module)


def _study_shown(st_module, targets: list[dict[str, Any]]) -> str | None:
    """The saved result the Study picker shows this run (the header is drawn before it)."""

    from ifvg_lab_nav import funded_target

    ss = st_module.session_state
    current = funded_target(st_module) or {}
    pending = ss.get(_PENDING) or {}
    return _follow_value(ss, _STUDY, [t["result_id"] for t in targets],
                         outside=current.get("result_id"), force=pending.get("result_id"))


def _back_link(st_module, targets: list[dict[str, Any]]) -> dict | None:
    """The link back to the detail tab — only while the result it returns to is open."""

    back = st_module.session_state.get(_BACK)
    if not back or _study_shown(st_module, targets) != back.get("result_id"):
        return None
    return back


def _header(back: dict | None) -> h.Markup:
    link = ""
    if back:
        tab = str(back.get("tab") or "Trades")
        words = ("Back to this configuration's trades" if tab == "Trades"
                 else f"Back to this configuration's {tab.lower()}")
        link = (f'<div class="lab-crumb"><a href="#" data-action="back">{h.esc(words)}</a>'
                "</div>")
    return h.Markup('<div class="lab" style="display:flex;flex-direction:column;gap:8px">'
                    f'{link}<h1 class="lab-h1" style="font-size:36px">Trade review</h1></div>')


def _go_back(st_module, back: dict) -> None:
    from ifvg_lab_nav import funded_context, funded_target, open_funded_detail

    target = funded_target(st_module)
    if not target or target.get("result_id") != back.get("result_id"):
        return
    context = funded_context(str(target["result_id"]), st_module)
    open_funded_detail(target, context.get("configuration"), context.get("firm_key"),
                       back.get("tab") or "Trades", st_module)


# ── funded trades ─────────────────────────────────────────────────────────


def _render_funded(st_module, roots, targets: list[dict[str, Any]]) -> None:
    from ifvg_lab_nav import FUNDED_TARGET, funded_context, funded_target
    from ifvg_lab_ui import funded_study, show

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import ordered_trades

    ss = st_module.session_state
    flash = ss.get(_FLASH)
    if flash and not flash[0]:
        ss.pop(_FLASH, None)
        show(h.alert(flash[1]), st_module)
    step_note = ss.pop(_STEP_NOTE, None)
    if step_note:
        show(h.note(step_note, "blue"), st_module)
    if not targets:
        show(h.note("No funded comparison with a saved result is available in this "
                    "application's studies yet."), st_module)
        return
    pending = ss.pop(_PENDING, None) or {}
    by_id = {t["result_id"]: t for t in targets}
    current = funded_target(st_module) or {}
    with st_module.container(key="ifvg_lab_review_pickers"):
        cols = st_module.columns([1.35, 1.6, 1.15, 0.85, 1.6, 0.25, 0.25], gap="small",
                                 vertical_alignment="bottom")
    _follow(st_module, _STUDY, list(by_id), outside=current.get("result_id"),
            force=pending.get("result_id"))
    with cols[0]:
        result_id = _seen(st_module, _STUDY, st_module.selectbox(
            "Study", list(by_id), key=_STUDY, format_func=lambda r: _study_label(by_id[r]),
            help="Funded comparisons with a saved result. "
                 + " · ".join(str(t.get("name") or "") for t in targets)))
    target = by_id[result_id]
    if current.get("result_id") != result_id:
        ss[FUNDED_TARGET] = dict(target)
    try:
        study = funded_study(target["store_root"], result_id)
    except Exception as error:  # verification failure: nothing from it is shown
        show(h.alert("This saved result could not be opened.",
                     "It failed its verification or is missing, so none of its trades are "
                     f"shown. ({type(error).__name__})"), st_module)
        return
    scope = result_id[:16]
    context = funded_context(result_id, st_module)
    firms = dict(study.firms)
    firm_key_widget = _P + f"firm_{scope}"
    _follow(st_module, firm_key_widget, list(firms), outside=context.get("firm_key"),
            force=pending.get("firm_key"))
    configurations = list(study.configurations)
    names, labels = configuration_labels(study, configurations)
    leader = None
    with cols[2]:
        firm_key = _seen(st_module, firm_key_widget, st_module.selectbox(
            "Firm", list(firms), key=firm_key_widget, format_func=lambda k: firms.get(k, k),
            help="The firm every funded view shows. Firms are never added together; this "
                 "choice carries to the results and every detail tab."))
    context["firm_key"] = firm_key
    from ifvg_lab_ui import pending_switch

    mode = ss.get(_MODE, MODES[0])
    point = pending_switch("review_mode", MODES, mode, st_module) == MODES[1]
    if point:
        # point in time: the saved plan's order, never the full-history ranking's leader
        default = configurations[0] if configurations else None
    else:
        completed = study.completed_at(firm_key)
        leader = completed[0]["configuration"] if completed else None
        default = leader
    config_key = _P + f"config_{scope}"
    _follow(st_module, config_key, configurations, outside=context.get("configuration"),
            force=pending.get("configuration"), default=default)
    with cols[1]:
        configuration = _seen(st_module, config_key, st_module.selectbox(
            "Configuration", configurations, key=config_key, format_func=labels.get,
            help="The configuration whose recorded funded trades are shown."))
    context["configuration"] = configuration
    pair = f"{configuration}|{firm_key}"
    summary = study.summary(configuration, firm_key) or {}
    rows = list(ordered_trades(study, configuration, firm_key))
    views = [rp.TradeView.from_row(r) for r in rows]
    if summary.get("status") != "Completed" or not rows:
        with cols[3]:
            st_module.selectbox("Account", ["—"], disabled=True, key=_P + "account_none",
                                help="This configuration has no recorded trades at this firm.")
        with cols[4]:
            st_module.selectbox("Trade", ["—"], disabled=True, key=_P + "trade_none",
                                help="This configuration has no recorded trades at this firm.")
        if summary.get("status") != "Completed":
            show(h.alert("This configuration did not complete with this firm.",
                         "It has no recorded trades. It is not a zero result."), st_module)
        else:
            show(h.note("No trades were taken by this configuration with this firm."),
                 st_module)
        return
    accounts = sorted({v.account for v in views if v.account is not None})
    account_key = _P + f"account_{scope}"
    trade_key = _P + f"trade_{scope}"
    shared = context.get("account") or {}
    shared_account = shared.get("number") if shared.get("pair") == pair else None
    force_seq = pending.get("seq")
    force_account = pending.get("account")
    if force_seq is not None and force_account is None:
        found = next((v for v in views if v.seq == force_seq), None)
        if found is not None and ss.get(account_key) != "all":
            force_account = found.account
    by_seq = {v.seq: v for v in views}
    known = views
    current = moment = None
    if point:
        # point in time: nothing decided after the moment — no later account, no later trade
        chosen = force_seq if force_seq in by_seq else ss.get(trade_key)
        if chosen not in by_seq:  # the Trade picker wasn't drawn last run: the trade shown then
            shown_last = ss.get(CURRENT) or {}
            if (shown_last.get("result_id"), shown_last.get("configuration"),
                    shown_last.get("firm_key")) == (result_id, configuration, firm_key):
                chosen = shown_last.get("trade")
        current = by_seq.get(chosen) or views[0]
        moment = _moment_now(ss, result_id, configuration, firm_key, current)
        opened = _accounts_opened(study, configuration, firm_key)
        # an account is listed only with a trade entered by the moment (never an empty list)
        accounts = rp.known_accounts(accounts, opened, moment, current.account, trades=views)
        known = rp.known_trades(views, moment, current.seq)
    _follow(st_module, account_key, ["all", *accounts], outside=shared_account,
            force=force_account, default="all")
    with cols[3]:
        account = _seen(st_module, account_key, st_module.selectbox(
            "Account", ["all", *accounts], key=account_key,
            format_func=lambda a: "All accounts" if a == "all" else f"Account {a}",
            help="One live account at a time; a lost account is replaced at the firm's price."))
    if account != "all":
        context["account"] = {"pair": pair, "number": account}
    shown = [v for v in known if account == "all" or v.account == account]
    if shown:
        _follow(st_module, trade_key, [v.seq for v in shown], force=force_seq,
                default=current.seq if current is not None else None)
        with cols[4]:
            seq = _seen(st_module, trade_key, st_module.selectbox(
                "Trade", [v.seq for v in shown], key=trade_key,
                format_func=lambda s: rp.trade_label(by_seq[s], hide_result=point,
                                                     account=account == "all"),
                help="Trades in the order they were taken. In point in time the result and the "
                     "account are left out of these names."))
    else:  # nothing to open for this account at this moment: say so and keep the trade shown
        with cols[4]:
            st_module.selectbox("Trade", ["—"], disabled=True, key=_P + "trade_none",
                                help="No trade of this account had been entered by then.")
        if point:
            text = (f"Account {account} has no trade entered by "
                    f"{rp.when(moment, current.entry_utc)}. Choose another account.")
        else:
            text = f"Account {account} has no recorded trade here. Choose another account."
        show(h.note(text), st_module)
        seq = (current or views[0]).seq
    if point and seq != current.seq and pending.get("seq") != seq:
        # another trade was chosen (or another account picked one): rebuild every picker from
        # that trade's own moment, never from the previous trade's
        ss[_PENDING] = {"result_id": result_id, "configuration": configuration,
                        "firm_key": firm_key, "seq": seq,
                        "account": None if account == "all" else account}
        st_module.rerun()
    order = [v.seq for v in views]
    index = order.index(seq)
    with cols[5], st_module.container(key="ifvg_lab_review_nav"):
        previous = st_module.button("‹", key=_P + "previous", disabled=index == 0,
                                    help="Previous trade of this configuration and firm.")
    last = index == len(order) - 1
    with cols[6], st_module.container(key="ifvg_lab_review_nav2"):
        # point in time: never disabled, so the button can't tell whether a later trade exists;
        # pressing it is a deliberate step forward in time (the next trade, at its own moment)
        following = st_module.button(
            "›", key=_P + "next", disabled=last and not point,
            help=_NEXT_HELP_POINT if point else "Next trade of this configuration and firm.")
    if following and last:
        following = False  # nothing later: stays on this trade
        if point:  # a step forward from the last trade moves the clock to the study's end
            _to_end_of_study(ss, result_id, configuration, firm_key, seq, account)
            st_module.rerun()
    if previous or following:
        step = order[index - 1] if previous else order[index + 1]
        ss[_PENDING] = {"result_id": result_id, "configuration": configuration,
                        "firm_key": firm_key, "seq": step,
                        "account": None if account == "all" else by_seq[step].account}
        st_module.rerun()
    row = rows[index]
    view = views[index]
    name = names[configuration]
    ss[CURRENT] = {"result_id": result_id, "configuration": configuration,
                   "firm_key": firm_key, "trade": view.seq, "account": view.account,
                   "point_in_time": point}
    line = rp.context_line([name.line1, name.line2, firms.get(firm_key, firm_key)],
                           view.account, index + 1, len(order), point_in_time=point)
    show(h.Markup(f'<div class="lab-line">{h.esc(line)}</div>'), st_module)
    _render_trade(st_module, roots, target, study, configuration, firm_key, row, view, rows,
                  point_default=point)


def _to_end_of_study(ss, result_id: str, configuration: str, firm_key: str, seq: int,
                     account: Any) -> None:
    """A step forward from the last trade in point in time: the clock moves to the study's
    end (Full history), and the page says so — it never silently stays put, which would
    tell that no later trade exists without saying where the clock went (correction A7)."""

    ss[_MODE] = MODES[0]
    ss[_STEP_NOTE] = END_OF_STUDY_NOTE
    ss[_PENDING] = {"result_id": result_id, "configuration": configuration,
                    "firm_key": firm_key, "seq": seq,
                    "account": None if account == "all" else account}


def _moment_now(ss, result_id: str, configuration: str, firm_key: str, view) -> pd.Timestamp:
    """The point-in-time moment for this trade: the one chosen on screen, else the default."""

    scope = f"{result_id[:16]}_{configuration}_{firm_key}_{view.seq}"
    chosen = ss.get(_P + f"moment_{scope}")
    if chosen:
        try:
            return pd.Timestamp(chosen)
        except (TypeError, ValueError):
            pass
    return rp.default_moment(view)


def _accounts_opened(study, configuration: str, firm_key: str) -> dict[int, pd.Timestamp]:
    """Each account's saved opening time (``created_utc`` of its account journey)."""

    out: dict[int, pd.Timestamp] = {}
    for journey in study.rows("account_journeys", configuration, firm_key):
        number, created = journey.get("account_number"), journey.get("created_utc")
        if number is None or not created:
            continue
        try:
            out[int(number)] = pd.Timestamp(str(created)[:26].rstrip("Z"), tz="UTC")
        except (TypeError, ValueError):
            continue
    return out


def configuration_labels(study, configurations: list[str]) -> tuple[dict, dict[str, str]]:
    """(names, picker labels) of a study's configurations — never two labels alike.

    Names come from :func:`names.study_names` (unique two-line names for one
    study); the picker shows the short first line plus only the second-line
    settings that tell configurations with the same short line apart.
    """

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_matrix import is_mffu_plan
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.names import (
        short_picker_name,
        study_names,
    )

    if is_mffu_plan(getattr(study, "plan", None)):
        from ifvg_lab_funded import study_names_for

        names = study_names_for(study)
        return names, {key: value.full for key, value in names.items()}
    names = study_names({c: study.settings(c) for c in configurations})
    short = {c: short_picker_name(n) for c, n in names.items()}
    return names, _picker_labels(short, {c: n.line2 for c, n in names.items()})


def _picker_labels(short: dict[str, str], second: dict[str, str]) -> dict[str, str]:
    """Short picker names; where two share one, add only the settings that tell them apart.

    A label still shared after that (two different first lines that shorten
    alike) ends with its configuration key.
    """

    groups: dict[str, list[str]] = {}
    for configuration, name in short.items():
        groups.setdefault(name, []).append(configuration)
    labels = {}
    for name, members in groups.items():
        if len(members) == 1:
            labels[members[0]] = name
            continue
        parts = {c: second.get(c, "").split(" · ") for c in members}
        width = max(len(p) for p in parts.values())
        differing = [i for i in range(width)
                     if len({(p[i] if i < len(p) else "") for p in parts.values()}) > 1]
        for configuration in members:
            extra = [parts[configuration][i] for i in differing
                     if i < len(parts[configuration])]
            labels[configuration] = " · ".join([name, *extra]) if extra else (
                f"{name} ({configuration})")
    counts: dict[str, int] = {}
    for label in labels.values():
        counts[label] = counts.get(label, 0) + 1
    return {c: (f"{label} ({c})" if counts[label] > 1 else label) for c, label in labels.items()}


def _variant(study, configuration: str):
    return next((v for v in getattr(study.plan, "variants", ()) or ()
                 if v.name == configuration), None)


def _record(target, study, configuration: str, row: dict):
    root = _package_root(target["store_root"], target["result_id"])
    if root is None:
        return None, "The strategy study this comparison used is not available, so "
    try:
        source = _setup_source(root)
    except Exception:
        return None, "The saved setup records could not be verified, so "
    variant = _variant(study, configuration)
    record = find_setup_record(
        row, configuration=configuration, source=source,
        axis_value_ids=dict(getattr(variant, "axis_value_ids", ()) or ()),
        in_verified_study=bool(getattr(variant, "in_verified_study", False)),
        base_configuration=getattr(study.plan, "base_configuration", None))
    return record, None


def _distance_cap(study, configuration: str) -> int | None:
    variant = _variant(study, configuration)
    ids = dict(getattr(variant, "axis_value_ids", ()) or ())
    value = ids.get("opposing_parent_distance_ticks_max", "")
    tail = value.rsplit(".", 1)[-1] if value else ""
    return int(tail) if tail.isdigit() else None


def _close_clock(study, configuration: str) -> clock_time | None:
    import re

    for setting, value in study.settings(configuration):
        if setting == "Daily close":
            found = re.search(r"(\d{1,2}):(\d{2}) ([AP]M)", value)
            if found:
                hour = int(found.group(1)) % 12 + (12 if found.group(3) == "PM" else 0)
                return clock_time(hour, int(found.group(2)))
            return None
    return clock_time(15, 55)


def _render_trade(st_module, roots, target, study, configuration, firm_key, row, view, rows,
                  *, point_default: bool) -> None:
    from ifvg_lab_cache import index_minutes
    from ifvg_lab_ui import show, switch

    ss = st_module.session_state
    scope = f"{target['result_id'][:16]}_{configuration}_{firm_key}_{view.seq}"
    record, record_problem = _record(target, study, configuration, row)
    # options bar
    with st_module.container(key="ifvg_lab_review_options"):
        cols = st_module.columns([2.25, 2.0, 0.95, 1.2, 1.4, 0.55, 1.25], gap="small",
                                 vertical_alignment="center")
        with cols[0]:
            mode = switch("What the charts show", MODES, key="review_mode",
                          value=ss.get(_MODE, MODES[1] if point_default else MODES[0]),
                          help="Full history shows everything. Point in time hides every "
                               "candle, marking and value after the moment you choose.",
                          st_module=st_module)
        ss[_MODE] = mode
        point = mode == MODES[1]
        moment = None
        with cols[1]:
            if point:
                options = rp.moments(view, record)
                default = rp.default_moment(view)
                keys = [m.at.isoformat() for m in options]
                labels = {m.at.isoformat(): m.label for m in options}
                moment_key = _P + f"moment_{scope}"
                if ss.get(moment_key) not in keys:
                    ss[moment_key] = (default.isoformat() if default.isoformat() in keys
                                      else keys[-1])
                chosen = st_module.selectbox(
                    "Moment", keys, key=moment_key, format_func=labels.get,
                    label_visibility="collapsed",
                    help="The moment to stop the clock at: the setup's steps, the entry, then "
                         "every 5 minutes for an hour and every hour to the 4:00 PM close. "
                         "None of these times comes from the exit. Everything after the "
                         "chosen time is hidden.")
                moment = pd.Timestamp(chosen)
            if not point:
                sentence = "Everything, including what happened after entry."
            elif moment < view.exit_utc:
                sentence = (f"Only what was known at {rp.when(moment, view.entry_utc)}. Later "
                            "candles and the result are hidden.")
            else:
                sentence = (f"Only what was known at {rp.when(moment, view.entry_utc)}. Later "
                            "candles are hidden; the trade had ended by then.")
            show(h.Markup(f'<div style="font-size:13px;color:{css_var("body_2")};'
                          'line-height:1.4">'
                          f"{h.esc(sentence)}</div>"), st_module)
        with cols[2]:
            zones = st_module.checkbox("Gap zones", value=True, key=_P + "zones",
                                       help="The saved setup record's gap zones.")
        with cols[3]:
            stops = st_module.checkbox("Stops and target", value=True, key=_P + "stops",
                                       help="The initial stop, the target and the moved stop.")
        with cols[4]:
            times = st_module.checkbox("Midnight and 3:55 PM", value=True, key=_P + "times",
                                       help="Chicago midnight inside the trading day and the "
                                            "3:55 PM daily close.")
        with cols[5]:
            show(h.Markup('<div class="lab" style="font-size:13px;text-align:right">Candles'
                          "</div>"), st_module)
        with cols[6]:
            size = st_module.selectbox(
                "Candles", rc.CANDLE_SIZES, index=rc.CANDLE_SIZES.index(10), key=_P + "candles",
                format_func=lambda n: "1 minute" if n == 1 else f"{n} minutes",
                label_visibility="collapsed",
                help="Candle size of the whole-trade chart, built from the stored one-minute "
                     "E-mini bars.")
    minutes = index_minutes(target["store_root"], target["result_id"])
    day = rc.day_minutes(minutes, view.trading_day) if minutes is not None else None
    record_sentence = record.source_sentence if record is not None else NO_RECORD
    if record is None and record_problem:
        record_sentence = record_problem + "setup zones can't be shown."
    variant = _variant(study, configuration)
    instrument = getattr(variant, "instrument", None) or getattr(study.plan, "instrument", None)
    gamma_segments = _gamma_review(st_module, target, study, configuration, row, view,
                                   moment=moment, day=day, scope=scope)
    _whole_trade_card(st_module, day, view, record, record_sentence, moment=moment,
                      size=int(size), zones=zones, stops=stops, times=times,
                      close_clock=_close_clock(study, configuration), instrument=instrument,
                      key=scope, gamma_segments=gamma_segments)
    link = _minute_link(roots, target, study, configuration, firm_key, view, rows)
    left, right = st_module.columns([1.12, 1], gap="medium")
    with left:
        _setup_card(st_module, day, view, record, moment=moment, key=scope)
    with right:
        _recorded_card(st_module, study, target, configuration, firm_key, row, view, rows,
                       instrument=instrument, moment=moment)
        _formed_card(st_module, study, configuration, view, record, moment=moment, link=link)
    _review_card(st_module, roots, target, study, configuration, firm_key, row, view, rows,
                 record=record, moment=moment)


def _minute_link(roots, target, study, configuration: str, firm_key: str, view, rows):
    """The published approximated-minute companion linked to this trade (correction A10).

    Only the latest review folder whose manifest names this exact result is used, and its
    ``approximated_minutes.csv`` only when its bytes match the manifest's SHA-256.
    """

    from ifvg_lab_detail_settings import latest_review_folder

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.minute_companion import (
        link_from_review_folder,
    )

    try:
        folder = latest_review_folder((roots or {}).get("repo_root"), str(target["result_id"]))
    except Exception:  # an unreadable reports folder: nothing is trusted from it
        folder = None
    return link_from_review_folder(
        folder, result_id=str(target["result_id"]), trade=view,
        pair_trades=[rp.TradeView.from_row(r) for r in rows], configuration=configuration,
        firm_name=dict(study.firms).get(firm_key, firm_key))


def _gamma_review(st_module, target, study, configuration, row, view, *, moment, day,
                  scope):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_matrix import is_mffu_plan

    if not is_mffu_plan(getattr(study, "plan", None)):
        return []
    from ifvg_lab_mffu_views import render_trade_gamma

    return render_trade_gamma(st_module, target, study, configuration, row, view,
                              moment=moment, window=rc.chart_window(day, view, moment=moment),
                              scope=scope)


def _whole_trade_card(st_module, day, view, record, record_sentence, *, moment, size, zones,
                      stops, times, close_clock, instrument, key, gamma_segments=None) -> None:
    from ifvg_lab_ui import plot, show

    with st_module.container(key="ifvg_lab_card_review_whole"):
        # point in time: the scheduled day, never the stored bars' last close (correction A7)
        window = rc.chart_window(day, view, moment=moment)
        size_words = "1-minute" if size == 1 else f"{size}-minute"
        right = _window_words(window)
        show(h.Markup(
            '<div class="lab" style="display:flex;justify-content:space-between;'
            'align-items:baseline;gap:16px;flex-wrap:wrap">'
            f'<div style="font-size:16px;font-weight:600">Whole trade · E-mini Nasdaq-100 · '
            f"{h.esc(size_words)} candles · Chicago time</div>"
            f'<div class="lab-line" style="font-size:13px">{h.esc(right)}</div></div>'),
            st_module)
        if day is None or day.empty:
            show(h.note("The stored one-minute E-mini bars for this trading day aren't "
                        "available, so the chart can't be drawn. The recorded facts below come "
                        "from the saved result."), st_module)
        else:
            fig = rc.whole_trade_figure(day, view, record, size_minutes=size, moment=moment,
                                        zones=zones, stops=stops, times=times,
                                        close_clock=close_clock)
            if gamma_segments:
                from ifvg_lab_mffu_views import add_gamma_overlay

                add_gamma_overlay(fig, gamma_segments)
            plot(fig, key=f"whole_{key}", st_module=st_module)
        swatch = ('<span style="display:inline-block;width:{w}px;height:12px;background:{c};'
                  'vertical-align:-1px;margin-right:6px;{b}"></span>')
        gap = rp.gap_legend(record)  # no swatch when no gap zone is saved (nothing is drawn)
        items = [
            swatch.format(w=10, c=css_var("blue"), b="") + "Up candle",
            swatch.format(w=10, c=css_var("orange"), b="") + "Down candle",
            *([swatch.format(w=16, c=css_var("blue_band"), b="") + h.esc(gap)] if gap else []),
            swatch.format(w=16, c=css_var("blue_light"),
                          b=f"border:1px solid {css_var('blue_band')}") + "In the trade",
        ]
        sentence = (h.esc(record_sentence) if record is not None
                    else h.placeholder(record_sentence))
        legend = ('<div class="lab" style="display:flex;gap:18px;flex-wrap:wrap;font-size:12px;'
                  f'color:{css_var("body_2")};align-items:center">'
                  + "".join(f"<span>{i}</span>" for i in items)
                  + f"<span>{sentence}</span></div>")
        caption = rc.whole_trade_caption(view, moment=moment)
        if instrument == "micro":
            caption += (" Micro positions are priced from the E-mini Nasdaq-100's recorded "
                        "trades; the candles are the E-mini's.")
        show(h.Markup(legend + f'<div class="lab-line" style="margin-top:6px">'
                      f"{h.esc(caption)}</div>"), st_module)


def _window_words(window) -> str:
    """``April 12, 5:00 PM open – April 13, 4:00 PM close``."""

    start, end = window
    return f"{_long_day(start)}, {rp.clock(start)} open – {_long_day(end)}, {rp.clock(end)} close"


def _long_day(at) -> str:
    local = at.tz_convert("America/Chicago")
    return f"{local:%B} {local.day}"


def _setup_card(st_module, day, view, record, *, moment, key) -> None:
    from ifvg_lab_ui import plot, show

    with st_module.container(key="ifvg_lab_card_review_setup"):
        items = rp.setup_key(record, view, moment=moment)
        window = rc.setup_window(view, record, moment=moment)
        first, last = rp.clock(window[0]), rp.clock(window[1])
        span = f"{first[:-3]} – {last}" if first[-2:] == last[-2:] else f"{first} – {last}"
        show(h.Markup(
            '<div class="lab" style="display:flex;justify-content:space-between;'
            'align-items:baseline;gap:16px">'
            f'<div style="font-size:16px;font-weight:600">{h.esc(rp.setup_card_title(record))}'
            f'</div><div class="lab-line" style="font-size:13px">{h.esc(span)}</div></div>'),
            st_module)
        if day is None or day.empty:
            show(h.note("The stored one-minute bars aren't available for this trading day."),
                 st_module)
        else:
            fig, _ = rc.setup_figure(day, view, record, moment=moment, key=items)
            plot(fig, key=f"setup_{key}", st_module=st_module)
        body = "".join(f"<div>{i}</div><div>{h.esc(text)}</div>"
                       for i, (_, text, _) in enumerate(items, start=1))
        extra = ""
        related = rp.related_source_line(record)
        if record is None:
            extra = (f'<div class="lab-line" style="margin-top:6px">{h.placeholder(NO_RECORD)}'
                     " — the candles, entry, stops and exits above come from the saved trade and "
                     "the stored bars.</div>")
        elif related:
            extra = f'<div class="lab-line" style="margin-top:6px">{h.esc(related)}</div>'
        show(h.Markup(f'<div class="lab lab-review-key">{body}</div>{extra}'), st_module)


def _recorded_card(st_module, study, target, configuration, firm_key, row, view, rows, *,
                   instrument, moment) -> None:
    from ifvg_lab_ui import show

    extra = []
    if view.account_failed and (moment is None or moment >= view.exit_utc):
        # point in time: a replacement account opened after the moment stays hidden (A7)
        extra = rp.failure_lines(study.result.get("tables") or {}, row, moment=moment)
    lines = rp.recorded_lines(view, instrument=instrument, moment=moment, extra=extra)
    body = "".join(f"<div>{h.esc(line.label)}</div><div>{line.text}</div>" for line in lines)
    if not lines:
        body = (f'<div style="grid-column:1 / -1;color:{css_var("muted")}">'
                "Nothing was recorded yet at "
                f"{h.esc(rp.when(moment, view.entry_utc))}: the entry comes later.</div>")
    note = ""
    if moment is not None and moment < view.exit_utc:
        note = (f'<div class="lab-note">Exit and result hidden, so you judge the setup as it '
                f"looked at {h.esc(rp.when(moment, view.entry_utc))}.</div>")
    elif moment is None and view.seq in rp.five_largest(rows):
        # a full-history statistic (it ranks every later trade): never in point in time (A7)
        share = _five_largest_share(target["store_root"], target["result_id"], configuration,
                                    firm_key)
        if share is not None:
            note = ('<div class="lab-note orange" style="font-size:13px">One of the five trades '
                    f"that carry {h.esc(fmt.percent(share))} of this configuration's profit."
                    "</div>")
    show(h.Markup('<div class="lab lab-card" style="padding:18px 20px;gap:12px">'
                  '<div class="lab-card-title sans">What was recorded</div>'
                  f'<div class="lab-review-grid">{body}</div>{note}</div>'), st_module)


def _formed_card(st_module, study, configuration, view, record, *, moment, link=None) -> None:
    from ifvg_lab_ui import show

    steps = rp.setup_steps(record, view, distance_cap=_distance_cap(study, configuration),
                           moment=moment)
    rows = []
    for step in steps:
        weight = "font-weight:600;" if step.bold else ""
        color = css_var("blue_dark") if step.bold else css_var("muted")
        time_color = css_var("ink") if step.bold else css_var("muted")
        rows.append(f'<div class="lab-mono" style="color:{color};{weight}">{step.number}</div>'
                    f'<div style="{weight}">{h.esc(step.text)}</div>'
                    f'<div style="color:{time_color};{weight}'
                    f'white-space:nowrap">{h.esc(step.time_text)}</div>')
    missing = ""
    related = rp.related_source_line(record)
    if record is None:
        missing = f'<div class="lab-line">{h.placeholder(NO_RECORD)}</div>'
    else:  # the step times are when each step became known (rp.SetupEvent)
        missing = (f'<div class="lab-line" style="color:{css_var("muted")}">'
                   f"{h.esc(rp.STEP_TIMES_NOTE)}</div>")
        if related:
            missing += f'<div class="lab-line">{h.esc(related)}</div>'
    evidence = rp.evidence_note(view, link)
    if moment is not None and moment < view.exit_utc:
        evidence = ("Price evidence for the whole trade is hidden with the exit; switch to Full "
                    "history to see it.")
    title = rp.formed_card_title(record)
    show(h.Markup('<div class="lab lab-card" style="padding:18px 20px;gap:10px">'
                  f'<div class="lab-card-title sans">{h.esc(title)}</div>'
                  f'<div class="lab-review-steps">{"".join(rows)}</div>{missing}'
                  f'<div style="font-size:12px;line-height:1.5;color:{css_var("muted")};'
                  f'border-top:1px solid {css_var("light_rule")};padding-top:8px">'
                  f"{h.esc(evidence)}</div></div>"), st_module)


# ── review form ───────────────────────────────────────────────────────────


def _case(target, study, row) -> tuple[str, str, dict[str, str]]:
    from alpha_lab.agents.data_infra.ifvg.presentation.funded_trade_review import review_keys

    plan_id = str(study.plan_id or target.get("plan_id") or "")
    return review_keys(str(target["result_id"]), plan_id, row)


def save_funded_review(*, repo_root: Path, store_root: Path, target: dict, study, row: dict,
                       reviewer: str, verdicts: dict[str, str], tags: list[str],
                       notes: str) -> dict:
    """Verify the saved result again, then append one review under the funded keys."""

    from alpha_lab.agents.data_infra.ifvg.visual_review_store import append_review
    from alpha_lab.propsim.funded.comparison_runner import load_comparison_result

    verified = load_comparison_result(Path(store_root), str(target["result_id"]))
    recorded = [t for t in (verified.get("tables") or {}).get("trades") or []
                if t.get("pair_id") == row.get("pair_id") and t.get("seq") == row.get("seq")]
    if len(recorded) != 1 or recorded[0].get("strategy_trade_id") != row.get(
            "strategy_trade_id"):
        raise ValueError("the saved funded trade could not be verified again")
    chart_key, case_key, pair_ref = _case(target, study, row)
    return append_review(repo_root=Path(repo_root), replay_chart_artifact_id=chart_key,
                         pair_ref=pair_ref, candidate_id=case_key, decision_id=None,
                         trade_id=str(row.get("strategy_trade_id")), reviewer=reviewer,
                         verdicts=verdicts, tags=tags, notes=notes)


def _review_card(st_module, roots, target, study, configuration, firm_key, row, view, rows, *,
                 record, moment) -> None:
    from ifvg_lab_ui import show

    from alpha_lab.agents.data_infra.ifvg.visual_review_store import list_reviews

    ss = st_module.session_state
    repo_root = Path(roots["repo_root"])
    chart_key, case_key, _pair = _case(target, study, row)
    tag = hashlib.sha1(case_key.encode()).hexdigest()[:16]
    key = _P + f"form_{tag}_"
    with st_module.container(key="ifvg_lab_card_review_form"):
        show(h.Markup(
            '<div class="lab" style="display:flex;justify-content:space-between;'
            'align-items:baseline;gap:16px;flex-wrap:wrap">'
            '<div style="font-size:16px;font-weight:600">Your review</div>'
            '<div class="lab-line" style="font-size:13px">Saved for this firm, account and '
            "trade only — it can't overwrite another review</div></div>"), st_module)
        left, right = st_module.columns(2, gap="large")
        labels = dict(rp.OVERALL_CHOICES)
        with left:
            overall = st_module.radio(
                "Overall", [k for k, _ in rp.OVERALL_CHOICES], index=None,
                format_func=labels.get, key=key + "overall",
                help="Nothing is preselected and nothing is saved until you press a save "
                     "button.")
        steps: dict[str, str] = {}
        # related context only (A6): the gap and parent steps would judge another
        # configuration's setup, so they are disabled and save nothing
        disabled = rp.disabled_steps(record)
        with right:
            show(h.Markup('<div class="lab" style="font-size:14px;font-weight:600;'
                          'margin-bottom:4px">Step by step</div>'), st_module)
            for step, _fields in rp.STEPS:
                a, b = st_module.columns([1.4, 1], vertical_alignment="center")
                with a:
                    show(h.Markup(f'<div class="lab" style="font-size:14px">'
                                  f"{h.esc(rp.step_label(step, record))}</div>"), st_module)
                with b:
                    steps[step] = st_module.selectbox(
                        rp.step_label(step, record), list(rp.STEP_CHOICES),
                        key=key + f"step_{step[:12]}", label_visibility="collapsed",
                        disabled=step in disabled,
                        help=(rp.OWN_RECORD_HELP if step in disabled
                              else "Not reviewed saves nothing for this step."))
        tag_labels = dict(rp.TAGS)
        tag_col, new_col = st_module.columns([4.2, 1], vertical_alignment="bottom")
        with tag_col:
            chosen = st_module.pills("Tags", list(tag_labels), selection_mode="multi",
                                     key=key + "tags",
                                     help="Optional tags saved with this review.") or []
        with new_col:
            st_module.button(r"\+ New tag", disabled=True, key=key + "new_tag",
                             help=_NEW_TAG_HELP)
        show(h.Markup(f'<div class="lab-line" style="font-size:12px">{h.esc(_NEW_TAG_HELP)}'
                      "</div>"), st_module)
        notes_col, who_col = st_module.columns([3.2, 1], vertical_alignment="bottom")
        with notes_col:
            notes = st_module.text_area("Notes", key=key + "notes", height=90,
                                        help="Optional notes saved with this review.")
        reviewer_key = _P + "reviewer"  # kept for this browser session only
        with who_col:
            reviewer = st_module.text_input("Reviewer", key=reviewer_key,
                                            help="Required before a review can be saved.")
        verdicts = rp.review_verdicts(overall, steps, disabled=disabled)
        ready = overall is not None and bool(str(reviewer).strip())
        flash = ss.get(_FLASH)
        status_col, next_col, save_col = st_module.columns([3.4, 1.05, 0.85],
                                                           vertical_alignment="center")
        with status_col:
            if flash and flash[0] == case_key:
                show(h.note(flash[1], "blue"), st_module)
            else:
                text = ("Unsaved — nothing is written until you save."
                        + ("" if ready else " Choose an overall verdict and name the reviewer "
                                            "to save."))
                show(h.Markup(f'<div class="lab-line">{h.esc(text)}</div>'), st_module)
        order = [int(r.get("seq") or 0) for r in rows]
        index = order.index(view.seq)
        with next_col:
            save_next = st_module.button(
                "Save and next trade", key=key + "save_next", disabled=not ready,
                help=("Saves this review, then opens the next trade (in point in time the clock "
                      "moves forward: to that trade, or to the end of the study when there is "
                      "none)." if moment is not None
                      else "Saves this review, then opens the next trade."))
        with save_col:
            save = st_module.button("Save review", key=key + "save", type="primary",
                                    disabled=not ready, help="Saves this review. Earlier "
                                                             "reviews are kept.")
        if save or save_next:
            try:
                save_funded_review(
                    repo_root=repo_root, store_root=Path(target["store_root"]), target=target,
                    study=study, row=row, reviewer=str(reviewer).strip(), verdicts=verdicts,
                    tags=[tag_labels[t] for t in chosen], notes=str(notes or ""))
            except Exception as error:  # nothing was written
                show(h.alert("Review not saved.", f"{error}"), st_module)
            else:
                ss[_FLASH] = (case_key, "Saved — your review of this trade was recorded. "
                                        "Earlier reviews are kept.")
                account = st_module.session_state.get(
                    _P + f"account_{target['result_id'][:16]}")
                if save_next and index + 1 < len(order):
                    nxt = order[index + 1]
                    ss[_PENDING] = {"result_id": target["result_id"],
                                    "configuration": configuration, "firm_key": firm_key,
                                    "seq": nxt,
                                    "account": None if account == "all" else
                                    int(rows[index + 1].get("account_number") or 0)}
                elif save_next and moment is not None:
                    # the last trade in point in time: the clock moves to the study's end (A7)
                    _to_end_of_study(ss, target["result_id"], configuration, firm_key,
                                     view.seq, account)
                st_module.rerun()
        _earlier_reviews(st_module, list_reviews, repo_root, case_key, chart_key,
                         moment=moment, exit_utc=view.exit_utc)


def _earlier_reviews(st_module, list_reviews, repo_root, case_key, chart_key, *, moment,
                     exit_utc) -> None:
    from ifvg_lab_ui import show

    from alpha_lab.agents.data_infra.ifvg.presentation.review_vocabulary import (
        label_for_verdict,
    )

    try:
        existing = list_reviews(repo_root=repo_root, candidate_id=case_key,
                                replay_chart_artifact_id=chart_key)
    except Exception:
        show(h.note("Earlier reviews could not be read."), st_module)
        return
    if existing.empty:
        show(h.Markup('<div class="lab-line">No earlier reviews of this trade.</div>'),
             st_module)
        return
    if moment is not None:
        # written with full knowledge, even after this trade's exit: hidden in point in time (A7)
        show(h.Markup(f'<div class="lab-line">{fmt.count(len(existing), "earlier review")} of '
                      "this trade hidden in point in time (they may mention the result or later "
                      "events). Switch to Full history to read them.</div>"), st_module)
        return
    # each saved step is listed as saved; a step saved as unknown (never chosen) is left out,
    # never shown as "incorrect" (correction A8)
    steps = {"htf_verdict": "Gap", "parent_verdict": "Parent", "entry_verdict": "Entry",
             "stop_verdict": "Stop", "outcome_verdict": "Exit"}
    tag_names = {k: label for label, k in rp.TAGS}
    columns = [h.Column("at", "Saved", width="170px"), h.Column("who", "Reviewer"),
               h.Column("overall", "Overall"), h.Column("steps", "Step by step"),
               h.Column("tags", "Tags"), h.Column("notes", "Notes")]
    body = []
    for _, r in existing.iloc[::-1].iterrows():
        step_text = ", ".join(f"{name} {label_for_verdict(r.get(field)).lower()}"
                              for field, name in steps.items()
                              if isinstance(r.get(field), str) and r.get(field))
        tags = r.get("tags")
        tags = ", ".join(tag_names.get(t, str(t).replace("_", " ")) for t in tags) \
            if isinstance(tags, list) else ""
        body.append(h.Row({"at": rp.short(pd.Timestamp(r["reviewed_at"])),
                           "who": str(r.get("reviewer") or ""),
                           "overall": label_for_verdict(r.get("overall_verdict")),
                           "steps": step_text or "—", "tags": tags or "—",
                           "notes": str(r.get("notes") or "") or "—"}))
    show(h.Markup('<div class="lab" style="font-size:14px;font-weight:600;margin-top:8px">'
                  f"Earlier reviews of this trade · {len(existing)}</div>"
                  + str(h.table(columns, body, plain=True))), st_module)
