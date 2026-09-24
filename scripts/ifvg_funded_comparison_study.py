"""Funded configuration comparison: configure, save/reopen, approve, freeze, launch, view.

A normal study draft (mode ``funded_configuration_comparison``) stores the
selected strategy settings in the configurator's usual ``search_space`` step
(``axis_selections``: registry value ids per setting) and the funded choices in
``review.funded_comparison``. Every combination of the selected values is
resolved to an approved configuration of the verified strategy study; each one
gets its own one-account-at-a-time simulation for every selected firm. A
historical run needs the owner's approval of the exact plan, recorded only by
the owner's own action on this screen.

"Variations around one configuration" builds a version-2 plan instead: every
combination of the chosen values around one study configuration, with its own
size for whole-position and half exits. That plan freezes the exact
Strategy-Core source this application imports (commit plus uncommitted-change
hash); the job launcher later starts the worker on the local checkout with that
same identity, or refuses. The pinned Core offers no half-exit choice; the
research Core must be selected explicitly when starting the application.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from alpha_lab.agents.data_infra.ifvg.presentation.workspace import axis_value_name
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import SEARCH_AXIS_REGISTRY_V1
from alpha_lab.agents.data_infra.ifvg.study_drafts import (
    load_draft,
    mark_frozen,
    new_draft,
    save_draft,
)
from alpha_lab.propsim.funded.comparison_draft import (
    DEFAULT_SETTINGS,
    DEFAULT_VARIATION,
    HALF_EXIT_REQUIREMENT,
    check_saved_comparison,
    rebuild_saved_plan,
    saved_settings,
    saved_study_selections,
    value_label,
)
from alpha_lab.propsim.funded.comparison_plan import COMPARISON_QUESTION
from alpha_lab.propsim.funded.comparison_runner import find_approval, load_comparison_result
from alpha_lab.propsim.funded.comparison_study import (
    COMPARISON_AXES,
    SCALE_OUT_VALUE,
    VARIATION_AXES,
    axis_choices,
    record_owner_approval,
    resolve_selection,
    save_plan,
    size_problem,
    variation_axis_values,
)
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES, INSTRUMENTS
from alpha_lab.propsim.funded.runner import read_state

__all__ = [
    "COMPARISON_MODE",
    "DEFAULT_SETTINGS",
    "start_comparison_draft",
    "comparison_state_root",
    "render_comparison_configuration",
    "render_comparison_study",
    "list_comparison_runs",
]

COMPARISON_MODE = "funded_configuration_comparison"
_REPO = Path(__file__).resolve().parents[1]
_KEY = "ifvg_fcmp_"
#: session keys that are not widget values of the configurator
_MARKER = _KEY + "loaded_draft"
_DIRTY = _KEY + "edited"
_PLAN_KINDS = {
    "study": "Configurations from the completed strategy study",
    "variations": "Variations around one configuration (target, gap charts, parents, "
                  "direction, entry hours, exit rule)",
}
# Selected chips show the value itself; long values wrap instead of being cut.
_CHIP_CSS = (
    "<style>"
    "[class*='st-key-ifvg_fcmp_'] [data-baseweb='tag']"
    "{max-width:none !important;height:auto !important;}"
    "[class*='st-key-ifvg_fcmp_'] [data-baseweb='tag'] span"
    "{white-space:normal !important;overflow:visible !important;"
    "text-overflow:clip !important;max-width:none !important;}"
    "</style>"
)
_CLOCK_LABELS = {
    "two_business_days": "Two business days, paid 4:00 PM Chicago (owner-selected)",
    "elapsed_48_hours": "48 elapsed hours (engineering comparison only; not owner-selected)",
}


def comparison_state_root(roots) -> Path:
    if roots.get("funded_comparison_state_root"):
        return Path(roots["funded_comparison_state_root"])
    return Path(roots["state_root"]).parent / "funded_comparison_jobs"


def _sources():
    from alpha_lab.propsim.funded.comparison_source import discover_comparison_sources

    return {s.package.run_id: s for s in discover_comparison_sources()}


def start_comparison_draft(name: str, source=None):
    draft = new_draft(COMPARISON_MODE, display_name=name)
    draft.steps["objective"] = {"mode_id": COMPARISON_MODE,
                                "question_id": "funded_configuration_cash"}
    selections = axis_choices(source) if source is not None else {}
    draft.steps["search_space"] = {"mode_id": COMPARISON_MODE,
                                   "axis_selections": {k: list(v) for k, v in selections.items()}}
    settings = dict(DEFAULT_SETTINGS)
    if source is not None:
        settings["source_run_id"] = source.package.run_id
    draft.steps["review"] = {"funded_comparison": settings}
    return draft


def _settings(draft) -> dict[str, Any]:
    return {**DEFAULT_SETTINGS, **saved_settings(draft)}


def _selections(draft) -> dict[str, list[str]]:
    stored = saved_study_selections(draft)
    return {axis: list(stored.get(axis) or []) for axis in COMPARISON_AXES}


def _spawn(command: list[str]) -> int:
    """The detached worker launch (tests replace it)."""

    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "DETACHED_PROCESS", 0)
    return subprocess.Popen(command, cwd=_REPO, creationflags=flags).pid  # noqa: S603


def _plan_summary_text(configurations, firm_keys, settings) -> str:
    spec = INSTRUMENTS[settings["instrument"]]
    firms = " and ".join(FIRM_PROFILES[k].firm_name for k in FIRM_PROFILES if k in firm_keys)
    pairs = len(configurations) * len(firm_keys)
    return (f"{len(configurations)} configuration{'s' if len(configurations) != 1 else ''} x "
            f"{firms} = {pairs} separate result{'s' if pairs != 1 else ''}; "
            f"{settings['quantity']} x {spec.label} per trade")


def _mark_edited() -> None:
    """Widget callback: the owner changed a setting (the only reason to save)."""

    import streamlit

    streamlit.session_state[_DIRTY] = True


def _draft_marker(root: Path, draft) -> list[str]:
    """The draft's identity and the digest of its saved file.

    The save time alone is not enough: two saves within one second share it.
    """

    import hashlib

    try:
        digest = hashlib.sha256((root / draft.draft_id / "draft.json").read_bytes()).hexdigest()
    except OSError:
        digest = ""  # a new draft not saved yet
    return [draft.draft_id, draft.updated_at_utc, digest]


def _sync_widget_state(st, root: Path, draft) -> None:
    """Widget values always start from the draft saved on disk.

    Streamlit keeps keyed widget values across reruns. When another draft is
    opened, or this draft was saved elsewhere (another window), the old values
    would otherwise be shown and saved over the draft, and an approval click on
    the old page would apply to a plan no longer saved. Reset them instead.
    """

    marker = _draft_marker(root, draft)
    previous = st.session_state.get(_MARKER)
    if previous == marker:
        return
    for key in [k for k in st.session_state if str(k).startswith(_KEY)]:
        del st.session_state[key]
    st.session_state[_MARKER] = marker
    if previous and previous[0] == draft.draft_id:
        st.info("This draft was saved from another window; showing its saved settings.")


def _save_edits(st, root: Path, draft) -> None:
    save_draft(root, draft)
    st.session_state[_MARKER] = _draft_marker(root, draft)
    st.session_state.pop(_DIRTY, None)


def _chip_label(value_id: str) -> str:
    """A selected chip shows the value itself, never the setting name."""

    from alpha_lab.propsim.funded.comparison_source import variation_value_label

    return variation_value_label(value_id) or value_label(value_id)


def _selected_caption(st, values: list[str]) -> None:
    if values:
        st.caption("Selected: " + " · ".join(value_label(v) for v in values))


def render_comparison_configuration(st, draft, roots) -> None:
    root = Path(roots["draft_root"])
    _sync_widget_state(st, root, draft)
    settings = _settings(draft)
    selections = _selections(draft)
    st.header(draft.display_name)
    st.write(COMPARISON_QUESTION)
    st.caption("Each selected strategy configuration gets its own funded account at each "
               "selected firm, one live account at a time. A lost account is replaced at the "
               "firm's price (\\$102 TakeProfitTrader, \\$125 MyFundedFutures) with no credit "
               "limit. Configurations and firms are separate results and are never added "
               "together. Nothing is bought, withdrawn or traded live.")
    sources = _sources()
    if not sources:
        st.warning("No verified completed strategy study with saved configurations is available.")
        return
    check = check_saved_comparison(saved_settings(draft), saved_study_selections(draft),
                                   sources, store_root=Path(roots["store_root"]))
    if not check.runnable:
        _render_saved_read_only(st, check, roots)
        return
    st.markdown(_CHIP_CSS, unsafe_allow_html=True)
    ids = list(sources)
    source_id = st.selectbox(
        "Completed strategy study (its approved configurations and saved dates)", ids,
        index=ids.index(settings["source_run_id"]) if settings["source_run_id"] in ids else 0,
        format_func=lambda i: sources[i].package.title, key=_KEY + "source",
        on_change=_mark_edited)
    source = sources[source_id]
    first, last = source.evaluation_dates[0], source.evaluation_dates[-1]
    st.caption(f"Period: {_long_date(first)} to {_long_date(last)} "
               f"({len(source.evaluation_dates)} trading days, after "
               f"{len(source.warmup_dates)} warmup days).")
    kind = st.radio("What to compare", list(_PLAN_KINDS), format_func=_PLAN_KINDS.get,
                    index=list(_PLAN_KINDS).index(settings.get("plan_kind", "study")),
                    key=_KEY + "plan_kind", on_change=_mark_edited)
    if kind == "variations":
        _render_variations(st, draft, roots, source, {**settings, "source_run_id": source_id,
                                                      "plan_kind": kind})
        return
    st.subheader("Strategy configurations")
    choices = axis_choices(source)
    chosen_selection: dict[str, list[str]] = {}
    for axis in COMPARISON_AXES:
        options = choices[axis]
        spec = SEARCH_AXIS_REGISTRY_V1.get(axis)
        label = spec.human_label if spec is not None else axis
        chosen_selection[axis] = st.multiselect(
            label, options, default=[v for v in selections.get(axis, []) if v in options],
            format_func=_chip_label, key=_KEY + "axis_" + axis, on_change=_mark_edited,
            help="Every combination of the selected values becomes its own configuration.")
        _selected_caption(st, chosen_selection[axis])
    configurations, unavailable = resolve_selection(source, chosen_selection)
    st.subheader("Firms and size")
    firm_keys = st.multiselect(
        "Firms (each is a separate result)", list(FIRM_PROFILES),
        default=[k for k in settings["firm_keys"] if k in FIRM_PROFILES],
        format_func=lambda k: FIRM_PROFILES[k].firm_name, key=_KEY + "firms",
        on_change=_mark_edited)
    instrument = st.radio("Contract", list(INSTRUMENTS), horizontal=True,
                          index=list(INSTRUMENTS).index(settings["instrument"]),
                          format_func=lambda k: INSTRUMENTS[k].label, key=_KEY + "instrument",
                          on_change=_mark_edited)
    quantity = st.number_input("Contracts per trade (same for every configuration and firm)",
                               min_value=1, max_value=60, step=1,
                               value=int(settings["quantity"]), key=_KEY + "quantity",
                               on_change=_mark_edited)
    cost = st.number_input("Modeled cost per contract per fill (US dollars)", min_value=0.0,
                           step=0.01, value=settings["cost_per_side_cents"] / 100,
                           format="%.2f", key=_KEY + "cost", on_change=_mark_edited)
    processing = st.radio("Payout processing pause", list(_CLOCK_LABELS),
                          index=list(_CLOCK_LABELS).index(settings["processing"]),
                          format_func=_CLOCK_LABELS.get, key=_KEY + "processing",
                          on_change=_mark_edited)
    updated = {**settings, "source_run_id": source_id, "firm_keys": list(firm_keys),
               "instrument": instrument, "quantity": int(quantity),
               "cost_per_side_cents": round(float(cost) * 100), "processing": processing,
               "plan_kind": "study"}
    # Opening, refreshing or navigating never rewrites the draft: only an edit saves.
    if (st.session_state.get(_DIRTY)
            or not (root / draft.draft_id / "draft.json").exists()):
        draft.steps["search_space"] = {"mode_id": COMPARISON_MODE,
                                       "axis_selections": chosen_selection}
        draft.steps.setdefault("review", {})["funded_comparison"] = updated
        _save_edits(st, root, draft)
    for profile in FIRM_PROFILES.values():
        dollar = "\\$"
        st.caption(
            f"{profile.firm_name}: {dollar}{profile.acquisition_cost_cents / 100:,.0f} per "
            f"account, {profile.trader_share_pct}% trader share, up to "
            f"{profile.max_minis_label}, {dollar}2,000 loss allowance, keeps {dollar}2,100 "
            f"after each payout, {dollar}500 minimum gross request. Owner-defined simulation "
            "terms.")
    if unavailable:
        st.warning(f"{len(unavailable)} selected combination(s) are not approved "
                   "configurations of this study and cannot run: " + "; ".join(
                       ", ".join(axis_value_name(v) for v in combo.values())
                       for combo in unavailable[:5]))
    problem = size_problem(instrument, int(quantity), list(firm_keys))
    if not configurations:
        st.info("Select at least one value for every setting.")
        return
    if not firm_keys:
        st.info("Select at least one firm.")
        return
    if problem:
        st.error(problem)
        return
    st.write("**This plan:** " + _plan_summary_text(configurations, firm_keys, updated))
    with st.expander(f"The {len(configurations)} configurations"):
        for config in configurations:
            st.write("- " + config.display_name.replace(" | ", "; "))
    try:
        envelope, _skipped, _unavailable = rebuild_saved_plan(source, updated, chosen_selection)
    except Exception as error:
        st.error(f"The plan could not be prepared: {error}")
        return
    if envelope is None:
        st.error("The plan could not be prepared from these settings.")
        return
    _approve_and_run(st, draft, envelope, roots,
                     _plan_summary_text(configurations, firm_keys, updated))


def _approve_and_run(st, draft, envelope, roots, scope: str) -> None:
    store = Path(roots["store_root"])
    plan_id = envelope.funded_comparison_plan_id
    state = read_state(comparison_state_root(roots), plan_id)
    already = bool(state and state.get("status") in ("Running", "Completed"))
    if already:
        st.info(f"This exact plan is already {state['status'].lower()}; its saved result is "
                "shown under My studies and is never recomputed.")
    approval = find_approval(store, plan_id)
    if approval is None:
        st.info("Running this on historical data needs your approval of this exact plan. "
                "Changing any setting creates a different plan that needs its own approval.")
        agree = st.checkbox(
            "I approve running this exact comparison on the saved historical period "
            "(no new dates, no live trading)", key=_KEY + "agree")
        if st.button("Record my approval", disabled=not agree, key=_KEY + "approve"):
            save_plan(store, envelope)
            record_owner_approval(
                store, plan_id, approved_on=date.today().isoformat(), channel="study_screen",
                statement="Approved on the study screen by the owner: run this exact funded "
                          "configuration comparison on the saved historical period.",
                scope=scope)
            st.rerun()
    else:
        st.success(f"Approved on {_long_date(approval.payload.approved_on)}.")
    if st.button("Run funded comparison", type="primary",
                 disabled=approval is None or already, key=_KEY + "run"):
        _freeze_and_launch(st, draft, envelope, roots)


def _variation(settings: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULT_VARIATION, **(settings.get("variation") or {})}


def _render_variations(st, draft, roots, source, settings: dict[str, Any]) -> None:
    """Version-2 plan: variations around one configuration of the verified study."""

    from alpha_lab.propsim.funded.core_identity import core_source_identity

    root = Path(roots["draft_root"])
    saved = _variation(settings)
    names = [c.name for c in source.configurations]
    labels = {c.name: c.display_name.replace(" | ", "; ") for c in source.configurations}
    base_name = st.selectbox(
        "Base configuration (every variation changes only the settings chosen below)",
        names, index=names.index(saved["base"]) if saved["base"] in names else 0,
        format_func=labels.get, key=_KEY + "var_base", on_change=_mark_edited)
    base_ids = source.by_name[base_name].axis_value_ids
    st.subheader("Settings to vary")
    selections: dict[str, list[str]] = {}
    for axis in VARIATION_AXES:
        options = variation_axis_values(axis)
        if not options:
            if axis == "exit_policy":
                st.caption("Exit rule: whole position at the target only. The half-exit rule "
                           "needs the research Strategy-Core, selected explicitly when the "
                           "application starts; it is not the pinned engine.")
            continue
        spec = SEARCH_AXIS_REGISTRY_V1.get(axis)
        baseline = base_ids.get(axis) or (spec.baseline_value_id if spec else None)
        # Compatibility was checked before this screen: every saved value is an option.
        default = [v for v in (saved["selections"].get(axis) or [baseline]) if v in options]
        selections[axis] = st.multiselect(
            spec.human_label if spec is not None else axis, options, default=default,
            format_func=_chip_label, key=_KEY + "var_" + axis, on_change=_mark_edited,
            help="Every combination of the selected values becomes its own configuration.")
        _selected_caption(st, selections[axis])
    firm_keys = st.multiselect(
        "Firms (each is a separate result)", list(FIRM_PROFILES),
        default=[k for k in settings["firm_keys"] if k in FIRM_PROFILES],
        format_func=lambda k: FIRM_PROFILES[k].firm_name, key=_KEY + "var_firms",
        on_change=_mark_edited)
    st.subheader("Size and costs")
    whole_q = st.number_input("Whole-position exits: E-mini Nasdaq-100 (NQ) contracts",
                              min_value=1, max_value=6, step=1,
                              value=int(saved["whole_quantity"]), key=_KEY + "var_whole_q",
                              on_change=_mark_edited)
    whole_c = st.number_input("Cost per E-mini contract per fill (US dollars)", min_value=0.0,
                              step=0.001, format="%.3f", value=saved["whole_cost_mills"] / 1000,
                              key=_KEY + "var_whole_c", on_change=_mark_edited)
    half = SCALE_OUT_VALUE in selections.get("exit_policy", [])
    half_q = int(saved["half_quantity"])
    half_c = saved["half_cost_mills"] / 1000
    if half:
        half_q = st.number_input("Half exits: Micro E-mini Nasdaq-100 (MNQ) contracts (even)",
                                 min_value=2, max_value=60, step=2, value=half_q,
                                 key=_KEY + "var_half_q", on_change=_mark_edited)
        half_c = st.number_input("Cost per micro contract per fill (US dollars)",
                                 min_value=0.0, step=0.001, format="%.3f", value=half_c,
                                 key=_KEY + "var_half_c", on_change=_mark_edited)
    variation = {"base": base_name, "selections": selections,
                 "whole_quantity": int(whole_q), "whole_cost_mills": round(whole_c * 1000),
                 "half_quantity": int(half_q), "half_cost_mills": round(half_c * 1000)}
    updated = {**settings, "firm_keys": list(firm_keys), "plan_kind": "variations",
               "variation": variation}
    # Opening, refreshing or navigating never rewrites the draft: only an edit saves.
    if st.session_state.get(_DIRTY) or not (root / draft.draft_id / "draft.json").exists():
        draft.steps.setdefault("review", {})["funded_comparison"] = updated
        _save_edits(st, root, draft)
    if not firm_keys or any(not selections.get(a) for a in selections):
        st.info("Select at least one firm and one value for every setting.")
        return
    if int(half_q) % 2:
        st.error("The half exit needs an even number of micro contracts.")
        return
    try:
        core_source_identity()
    except Exception:
        st.error("The Strategy-Core this application imports is not a verifiable source "
                 "checkout, so a variation plan cannot freeze it. Start the application with "
                 "an explicitly prepared checkout (run_ifsm_research_ui.py --core).")
        return
    try:
        envelope, skipped, _unavailable = rebuild_saved_plan(source, updated, {})
    except Exception as error:
        st.error(f"The plan could not be prepared: {error}")
        return
    if skipped:
        st.warning(f"{len(skipped)} combination(s) cannot run and are not in the plan: "
                   + "; ".join(sorted({reason for _c, reason in skipped})) + ".")
    if envelope is None:
        st.info("No combination can run with these selections.")
        return
    plan = envelope.payload
    st.caption(f"Strategy engine frozen into this plan: {plan.core_source.description}. "
               "The run uses exactly this source and refuses any other.")
    firms = " and ".join(FIRM_PROFILES[k].firm_name for k in FIRM_PROFILES if k in firm_keys)
    halves = sum(1 for v in plan.variants if v.exit_policy != "fixed_target_v1")
    scope = (f"{len(plan.variants)} configurations around {labels[base_name]} x {firms} = "
             f"{len(plan.variants) * len(firm_keys)} separate results; "
             f"{len(plan.variants) - halves} whole-position at {whole_q} x NQ, {halves} half "
             f"exits at {half_q} x MNQ")
    st.write("**This plan:** " + scope)
    with st.expander(f"The {len(plan.variants)} configurations"):
        for variant in plan.variants:
            st.write("- " + variant.display_name.replace(" | ", "; "))
    _approve_and_run(st, draft, envelope, roots, scope)


def _render_saved_read_only(st, check, roots) -> None:
    """A saved draft this application cannot represent: shown as saved, never changed."""

    count = check.configuration_count
    if count is None:
        lead = "This study's configurations cannot be counted in this application"
    elif check.count_is_exact:
        lead = f"This study contains {count} configuration{'s' if count != 1 else ''}"
    else:
        lead = f"This study describes up to {count} configurations"
    need = (f"requires {HALF_EXIT_REQUIREMENT}" if check.needs_half_exit_engine
            else "cannot be edited in this application")
    st.warning(f"{lead} and {need}. Your saved settings have not been changed. It is shown "
               "here for reading only: editing, approval and running are unavailable in this "
               "application.")
    for problem in check.problems:
        st.write("- " + problem)
    st.subheader("Saved settings")
    import pandas as pd

    st.table(pd.DataFrame(check.saved_rows, columns=["Setting", "Saved value"])
             .set_index("Setting"))
    store = Path(roots["store_root"])
    checkout = None
    for plan_id in check.matching_plan_ids:
        from alpha_lab.propsim.funded.comparison_runner import is_v2, load_plan
        from alpha_lab.propsim.funded.research_core_sources import find_core_checkout

        plan = load_plan(store, plan_id)
        size = len(plan.variants) if is_v2(plan) else len(plan.configurations)
        approval = find_approval(store, plan_id)
        state = read_state(comparison_state_root(roots), plan_id) or {}
        status = {"Completed": "its run is completed; the saved result is under My studies",
                  "Running": "its run is in progress"}.get(state.get("status"), "not run yet")
        st.write(f"Saved plan {plan_id[:12]}… matches these settings exactly: {size} "
                 f"configurations; "
                 + (f"approved on {_long_date(approval.payload.approved_on)}"
                    if approval else "not approved") + f"; {status}.")
        if is_v2(plan) and checkout is None:
            checkout = find_core_checkout(plan.core_source)
    if check.needs_half_exit_engine:
        where = str(checkout) if checkout else "<the research Strategy-Core checkout>"
        st.info("To edit, approve or run this study, start the application with the research "
                f"engine: `python scripts/run_ifsm_research_ui.py --research-core {where}`. "
                "The shared pinned engine is not changed.")
    st.caption("To change the settings on purpose, clone the study (My studies → Details and "
               "actions → Study actions → Clone study) and edit the copy where its settings are "
               "available; a changed copy is a different plan that needs its own approval.")


def dispatch_problem(roots, draft_id: str, envelope) -> str | None:
    """Why this plan must not be launched now (None when it may).

    The launch re-reads the draft saved on disk, checks it against this
    application's engine, rebuilds the plan from those saved settings and
    requires the very plan being launched. A stale page, a stale approval or a
    draft this application cannot represent therefore never reaches the worker.
    """

    try:
        disk = load_draft(Path(roots["draft_root"]), draft_id)
    except Exception:
        return "The saved draft could not be read again. Nothing was launched."
    settings, selections = saved_settings(disk), saved_study_selections(disk)
    sources = _sources()
    check = check_saved_comparison(settings, selections, sources)
    if not check.runnable:
        return ("The saved draft cannot run in this application: " + " ".join(check.problems)
                + " Nothing was launched.")
    source = sources.get({**DEFAULT_SETTINGS, **settings}.get("source_run_id"))
    if source is None:
        return ("The saved draft does not name a completed strategy study available here. "
                "Choose the study again so it is saved, review the plan and approve it. "
                "Nothing was launched.")
    try:
        rebuilt, _skipped, _unavailable = rebuild_saved_plan(source, settings, selections)
    except Exception as error:
        return f"The saved plan could not be rebuilt ({error}). Nothing was launched."
    if rebuilt is None or rebuilt.funded_comparison_plan_id != envelope.funded_comparison_plan_id:
        return ("This page is out of date: the saved draft no longer describes the plan shown. "
                "Refresh the page and review the plan again. Nothing was launched.")
    if find_approval(Path(roots["store_root"]), envelope.funded_comparison_plan_id) is None:
        return "This exact plan has no recorded approval. Nothing was launched."
    return None


def _freeze_and_launch(st, draft, envelope, roots) -> None:
    problem = dispatch_problem(roots, draft.draft_id, envelope)
    if problem:
        st.error(problem)
        return
    store = Path(roots["store_root"])
    plan_id = save_plan(store, envelope)
    state_root = comparison_state_root(roots)
    state = read_state(state_root, plan_id)
    if state and state.get("status") in ("Running", "Completed"):
        st.warning("This exact plan has already been run. Its saved result is reused.")
    else:
        mark_frozen(Path(roots["draft_root"]), draft, search_id=plan_id)
        _spawn([sys.executable, str(_REPO / "scripts/ifvg_funded_comparison_job.py"), "start",
                "--plan-id", plan_id, "--store-root", str(store),
                "--state-root", str(state_root),
                "--reports-root", str(roots.get("reports_root", _REPO / "reports"))])
        deadline = time.monotonic() + 10
        while read_state(state_root, plan_id) is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if read_state(state_root, plan_id) is None:
            st.warning("The comparison was requested but has not reported progress yet. "
                       "Refresh My studies in a moment.")
            return
    st.session_state["ifvg_workspace_selected_study"] = plan_id
    st.session_state["ifvg_workspace_screen"] = "detail"
    st.rerun()


def _long_date(text: str) -> str:
    day = date.fromisoformat(text)
    return f"{day:%B} {day.day}, {day.year}"


@dataclass(frozen=True)
class ComparisonRun:
    plan_id: str
    state: dict[str, Any]
    draft: Any
    dates: str
    title: str = "Funded configuration comparison"


def list_comparison_runs(roots, drafts) -> list[ComparisonRun]:
    state_root = comparison_state_root(roots)
    linked = {d.frozen_search_id: d for d in drafts if d.mode_id == COMPARISON_MODE}
    runs = []
    if state_root.is_dir():
        for folder in sorted(state_root.iterdir()):
            state = read_state(state_root, folder.name)
            if state:
                runs.append(ComparisonRun(folder.name, state, linked.get(folder.name),
                                          _plan_dates(roots, folder.name),
                                          _plan_title(roots, folder.name)))
    return runs


def _plan_title(roots, plan_id: str) -> str:
    """A plain title for a plan without a linked draft (built outside the screen)."""

    from alpha_lab.propsim.funded.comparison_runner import is_v2, load_plan

    try:
        plan = load_plan(Path(roots["store_root"]), plan_id)
    except Exception:
        return "Funded configuration comparison"
    if is_v2(plan):
        halves = sum(1 for v in plan.variants if v.exit_policy != "fixed_target_v1")
        base = next((v.display_name for v in plan.variants
                     if v.name == plan.base_configuration), None)
        around = (" around " + base.split(" | ")[0].lower()) if base else ""
        return (f"Funded variation study — {len(plan.variants)} configurations{around}"
                + (f", {halves} with the half exit" if halves else ""))
    return f"Funded configuration comparison — {len(plan.configurations)} configurations"


def _plan_dates(roots, plan_id: str) -> str:
    from alpha_lab.propsim.funded.comparison_runner import load_plan

    try:
        plan = load_plan(Path(roots["store_root"]), plan_id)
    except Exception:
        return "Dates unavailable"
    return (f"{_long_date(plan.source.evaluation_dates[0])} – "
            f"{_long_date(plan.source.evaluation_dates[-1])}")


def render_comparison_study(st, study, roots) -> None:
    state = study.state or {}
    status = state.get("status", "Evidence unavailable")
    if status == "Running":
        done = state.get("configurations_done")
        total = state.get("configurations_total")
        progress = (f" — {done} of {total} configurations finished"
                    if done is not None and total else "")
        st.info(f"Running{progress}. This page shows the result when the run finishes.")
        if st.button("Refresh", key=_KEY + "refresh"):
            st.rerun()
        return
    if status == "Failed" and not state.get("result_id"):
        st.error("The funded comparison stopped without a verified result: "
                 f"{state.get('reason', 'unknown reason')}")
        return
    result_id = state.get("result_id")
    if not result_id:
        st.warning("The saved result is unavailable.")
        return
    try:
        result = load_comparison_result(Path(roots["store_root"]), result_id)
    except Exception:
        st.error("The saved result failed verification and is not shown.")
        return
    if status == "Failed":
        st.error("The saved result failed its internal money checks; it is shown for review "
                 "only.")
    elif status == "Incomplete":
        st.warning("Some configurations did not complete; they are listed with the reason. "
                   "The completed ones are shown. The same plan can be run again; this saved "
                   "result is kept.")
    from ifvg_funded_comparison_results import render_funded_comparison_results

    render_funded_comparison_results(st, result, result_id=result_id)
    if state.get("review_folder"):
        st.caption("Full-study review folder (every configuration and every firm of this "
                   "result, not only the selection above) was saved automatically in the "
                   "reports folder under “Funded comparison reviews”.")
    elif state.get("review_error"):
        st.warning("The review folder could not be published; the verified result is kept.")
        if st.button("Publish the review folder again", key=_KEY + "publish"):
            from alpha_lab.propsim.funded.comparison_runner import publish_comparison_review

            publish_comparison_review(
                plan_id=study.key, store_root=Path(roots["store_root"]),
                state_root=comparison_state_root(roots),
                reports_root=Path(roots.get("reports_root", _REPO / "reports")))
            st.rerun()
