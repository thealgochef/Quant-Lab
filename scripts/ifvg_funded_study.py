"""Funded payout simulation: configure, save/reopen, freeze, launch and view.

Uses the workspace's normal draft files (mode ``funded_payout_simulation``),
freezes an immutable plan into the selected study store, starts the detached
worker ``scripts/ifvg_funded_payout_job.py`` and renders the verified result.
Launching on historical data requires an exact owner authorization record.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_lab.agents.data_infra.ifvg.search.store import (
    has_envelope,
    save_envelope_immutable,
)
from alpha_lab.agents.data_infra.ifvg.study_drafts import (
    mark_frozen,
    new_draft,
    save_draft,
)
from alpha_lab.propsim.funded.clock import ELAPSED_48_HOURS, TWO_BUSINESS_DAYS_FED_1600
from alpha_lab.propsim.funded.plan import (
    FUNDED_QUESTION,
    PILOT_OWNER_DECISIONS,
    FundedPayoutPlanEnvelope,
    FundedPayoutPlanPayload,
    OwnerDecisionRef,
    StrategySourceRef,
)
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES, INSTRUMENTS
from alpha_lab.propsim.funded.runner import PLAN_STORE, load_result, read_state

__all__ = [
    "FUNDED_MODE",
    "PILOT_AUTHORIZATION",
    "DEFAULT_SETTINGS",
    "start_funded_draft",
    "build_plan",
    "authorization_for",
    "render_funded_configuration",
    "render_funded_study",
    "list_funded_runs",
    "funded_state_root",
]

FUNDED_MODE = "funded_payout_simulation"
_REPO = Path(__file__).resolve().parents[1]
_KEY = "ifvg_funded_"

#: The owner's exact September 22, 2026 authorization: ONE bounded pilot.
PILOT_AUTHORIZATION: dict[str, Any] = {
    "authorized_on": "2026-09-22",
    "scope": "One bounded same-period funded pilot of the daily-close control, one mini "
             "in both firms, two-business-day processing; validates the simulator only.",
    "settings": {
        "package_run_id": "a0f66422ef9f6b3393ad21d0b9740ebc0f23684cfe3463139a90743141512fd7",
        "profile_id": "S0_D160_W1_P0",
        "instrument": "mini",
        "quantity": 1,
        "cost_per_side_cents": 514,
        "processing": "two_business_days",
        "price_evidence_policy": "ordered_trade_prints_with_labeled_minute_fallback",
    },
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "package_run_id": PILOT_AUTHORIZATION["settings"]["package_run_id"],
    "profile_id": "S0_D160_W1_P0",
    "instrument": "mini",
    "quantity": 1,
    "cost_per_side_cents": 514,
    "processing": "two_business_days",
    "price_evidence_policy": "ordered_trade_prints_with_labeled_minute_fallback",
}

_CLOCKS = {"two_business_days": TWO_BUSINESS_DAYS_FED_1600,
           "elapsed_48_hours": ELAPSED_48_HOURS}
_CLOCK_LABELS = {
    "two_business_days": "Two business days, paid 4:00 PM Chicago (owner-selected)",
    "elapsed_48_hours": "48 elapsed hours (engineering comparison only; not owner-selected)",
}
_POLICY_LABELS = {
    "ordered_trade_prints_with_labeled_minute_fallback":
        "Ordered trade prints; minute candles only where prints are unavailable (labeled)",
    "ordered_trade_prints_required": "Ordered trade prints required for every trade",
    "minute_bars_adverse_first": "One-minute candles, losing side first (approximation)",
    "minute_bars_favorable_first": "One-minute candles, winning side first (approximation)",
}


def funded_state_root(roots) -> Path:
    if roots.get("funded_state_root"):
        return Path(roots["funded_state_root"])
    return Path(roots["state_root"]).parent / "funded_jobs"


def _settings(draft) -> dict[str, Any]:
    return {**DEFAULT_SETTINGS, **draft.steps.get("review", {}).get("funded_settings", {})}


def start_funded_draft(name: str):
    draft = new_draft(FUNDED_MODE, display_name=name)
    draft.steps["objective"] = {"mode_id": FUNDED_MODE, "question_id": "funded_payout_cash"}
    draft.steps["review"] = {"funded_settings": dict(DEFAULT_SETTINGS)}
    return draft


def _packages():
    from alpha_lab.propsim.funded.sources import discover_packages

    return {p.run_id: p for p in discover_packages()}


def authorization_for(settings: dict[str, Any]) -> dict[str, Any] | None:
    wanted = PILOT_AUTHORIZATION["settings"]
    if all(settings.get(k) == v for k, v in wanted.items()):
        return PILOT_AUTHORIZATION
    return None


def build_plan(settings: dict[str, Any]) -> FundedPayoutPlanEnvelope:
    from alpha_lab.propsim.funded.sources import (
        executions_sha256,
        load_profile_executions,
        load_trading_days,
    )

    package = _packages().get(settings["package_run_id"])
    if package is None:
        raise ValueError("The selected completed study is unavailable or failed verification.")
    executions = load_profile_executions(package, settings["profile_id"])
    _days, _start, _cutoff, (first, last) = load_trading_days(package)
    authorization = authorization_for(settings)
    decisions = list(PILOT_OWNER_DECISIONS)
    if settings["processing"] != "two_business_days":
        decisions.append(OwnerDecisionRef(
            decided_on="2026-09-22", subject="Processing clock in this plan",
            decision="48 elapsed hours: an engineering comparison, not the owner's choice.",
            status="assumption"))
    payload = FundedPayoutPlanPayload(
        purpose="pilot_validation" if authorization else "engineering_sample",
        question=FUNDED_QUESTION,
        source=StrategySourceRef(
            package_run_id=package.run_id, package_manifest_sha256=package.manifest_sha256,
            package_root_name=package.root.name, profile_id=settings["profile_id"],
            description=f"{package.title} — profile {settings['profile_id']}",
            executions_sha256=executions_sha256(executions), evaluation_first_day=first,
            evaluation_last_day=last,
            executions=sum(1 for e in executions if not e.is_warmup),
        ),
        instrument=settings["instrument"], quantity=int(settings["quantity"]),
        cost_per_side_cents=int(settings["cost_per_side_cents"]),
        processing=_CLOCKS[settings["processing"]],
        firm_profiles=tuple(FIRM_PROFILES.values()),
        price_evidence_policy=settings["price_evidence_policy"],
        owner_decisions=tuple(decisions),
        authorized_scope=authorization["scope"] if authorization else
        "Not authorized for a historical run.",
    )
    return FundedPayoutPlanEnvelope.from_payload(payload)


def _size_problem(settings) -> str | None:
    spec = INSTRUMENTS[settings["instrument"]]
    exposure = spec.mini_equivalent_tenths * int(settings["quantity"])
    for profile in FIRM_PROFILES.values():
        if exposure > profile.max_mini_equivalent_tenths:
            return (f"{settings['quantity']} × {spec.label} is above the {profile.firm_name} "
                    f"limit of {profile.max_minis_label}. Choose a supported size; the "
                    "simulation never reduces a size silently.")
    return None


def _spawn(command: list[str]) -> int:
    """The detached worker launch (tests replace it)."""

    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "DETACHED_PROCESS", 0)
    return subprocess.Popen(command, cwd=_REPO, creationflags=flags).pid  # noqa: S603


def render_funded_configuration(st, draft, roots) -> None:
    root = Path(roots["draft_root"])
    settings = _settings(draft)
    st.header(draft.display_name)
    st.write(FUNDED_QUESTION)
    st.caption("Two separate funded-account simulations — TakeProfitTrader and "
               "MyFundedFutures — on the same market period. Accounts start funded; "
               "no evaluation is simulated. Nothing is bought, withdrawn or traded live.")
    packages = _packages()
    if not packages:
        st.warning("No verified completed study is available as a strategy source.")
        return
    ids = list(packages)
    package_id = st.selectbox(
        "Completed strategy study", ids,
        index=ids.index(settings["package_run_id"]) if settings["package_run_id"] in ids else 0,
        format_func=lambda i: packages[i].title, key=_KEY + "package")
    profiles = list(packages[package_id].profiles)
    profile = st.selectbox(
        "Strategy configuration", profiles,
        index=profiles.index(settings["profile_id"]) if settings["profile_id"] in profiles
        else 0, key=_KEY + "profile")
    instrument = st.radio(
        "Contract", list(INSTRUMENTS), horizontal=True,
        index=list(INSTRUMENTS).index(settings["instrument"]),
        format_func=lambda k: INSTRUMENTS[k].label, key=_KEY + "instrument")
    quantity = st.number_input("Contracts per trade (same in both firms)", min_value=1,
                               max_value=60, step=1, value=int(settings["quantity"]),
                               key=_KEY + "quantity")
    cost = st.number_input("Modeled cost per contract per fill (US dollars)", min_value=0.0,
                           step=0.01, value=settings["cost_per_side_cents"] / 100,
                           key=_KEY + "cost", format="%.2f")
    processing = st.radio("Payout processing pause", list(_CLOCKS),
                          index=list(_CLOCKS).index(settings["processing"]),
                          format_func=_CLOCK_LABELS.get, key=_KEY + "processing")
    policy = st.selectbox("Price evidence for loss checks", list(_POLICY_LABELS),
                          index=list(_POLICY_LABELS).index(settings["price_evidence_policy"]),
                          format_func=_POLICY_LABELS.get, key=_KEY + "policy")
    for profile_row in FIRM_PROFILES.values():
        # an escaped dollar keeps Streamlit markdown from reading pairs as math
        dollar = "\\$"
        st.caption(
            f"{profile_row.firm_name}: {dollar}{profile_row.acquisition_cost_cents / 100:,.0f} "
            f"per account, {profile_row.trader_share_pct}% trader share, up to "
            f"{profile_row.max_minis_label}, {dollar}2,000 loss allowance, keeps "
            f"{dollar}2,100 after each payout, {dollar}500 minimum gross request. "
            "Owner-defined simulation terms.")
    updated = {
        "package_run_id": package_id, "profile_id": profile, "instrument": instrument,
        "quantity": int(quantity), "cost_per_side_cents": round(float(cost) * 100),
        "processing": processing, "price_evidence_policy": policy,
    }
    if updated != settings or not (root / draft.draft_id / "draft.json").exists():
        draft.steps.setdefault("review", {})["funded_settings"] = updated
        save_draft(root, draft)
    problem = _size_problem(updated)
    authorization = authorization_for(updated)
    if problem:
        st.error(problem)
    elif authorization is None:
        st.info("These settings are saved. Running them on historical data needs the owner's "
                "authorization for this exact plan; only the September 22, 2026 pilot "
                "settings are authorized.")
    else:
        st.success("Authorized: " + authorization["scope"])
    if st.button("Run funded simulation", type="primary",
                 disabled=bool(problem or authorization is None), key=_KEY + "run"):
        _freeze_and_launch(st, draft, updated, roots)


def _freeze_and_launch(st, draft, settings, roots) -> None:
    store = Path(roots["store_root"])
    try:
        envelope = build_plan(settings)
    except Exception as error:
        st.error(f"The plan could not be frozen: {error}")
        return
    plan_id = envelope.funded_plan_id
    if not has_envelope(store, PLAN_STORE, plan_id):
        save_envelope_immutable(store, PLAN_STORE, envelope)
    state_root = funded_state_root(roots)
    state = read_state(state_root, plan_id)
    if state and state.get("status") in ("Running", "Completed"):
        st.warning("This exact plan has already been run. Its saved result is reused.")
    else:
        mark_frozen(Path(roots["draft_root"]), draft, search_id=plan_id)
        _spawn([sys.executable, str(_REPO / "scripts/ifvg_funded_payout_job.py"), "start",
                "--plan-id", plan_id, "--store-root", str(store),
                "--state-root", str(state_root),
                "--reports-root", str(roots.get("reports_root", _REPO / "reports"))])
        # report "started" only once the job has recorded its state
        deadline = time.monotonic() + 10
        while read_state(state_root, plan_id) is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if read_state(state_root, plan_id) is None:
            st.warning("The simulation was requested but has not reported progress yet. "
                       "Refresh My studies in a moment.")
            return
    st.session_state["ifvg_workspace_selected_study"] = plan_id
    st.session_state["ifvg_workspace_screen"] = "detail"
    st.rerun()


@dataclass(frozen=True)
class FundedRun:
    plan_id: str
    state: dict[str, Any]
    draft: Any
    dates: str


def list_funded_runs(roots, drafts) -> list[FundedRun]:
    state_root = funded_state_root(roots)
    linked = {d.frozen_search_id: d for d in drafts if d.mode_id == FUNDED_MODE}
    runs = []
    if state_root.is_dir():
        for folder in sorted(state_root.iterdir()):
            state = read_state(state_root, folder.name)
            if state:
                runs.append(FundedRun(folder.name, state, linked.get(folder.name),
                                      _plan_dates(roots, folder.name)))
    return runs


def _plan_dates(roots, plan_id: str) -> str:
    from datetime import date

    from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope

    try:
        plan = load_verified_envelope(Path(roots["store_root"]), PLAN_STORE, plan_id,
                                      FundedPayoutPlanEnvelope).payload
    except Exception:
        return "Dates unavailable"
    first = date.fromisoformat(plan.source.evaluation_first_day)
    last = date.fromisoformat(plan.source.evaluation_last_day)
    return f"{first:%B} {first.day}, {first.year} – {last:%B} {last.day}, {last.year}"


def render_funded_study(st, study, roots) -> None:
    from ifvg_funded_results import render_funded_results

    state = study.state or {}
    status = state.get("status", "Evidence unavailable")
    if status == "Running":
        phase = str(state.get("phase", "")).replace("_", " ")
        st.info(f"Running: {phase}. This page shows the result when the run finishes.")
        if st.button("Refresh", key=_KEY + "refresh"):
            st.rerun()
        return
    if status == "Failed":
        st.error("The funded simulation stopped without a verified result: "
                 f"{state.get('reason', 'unknown reason')}")
        return
    result_id = state.get("result_id")
    if not result_id:
        st.warning("The saved result is unavailable.")
        return
    try:
        result = load_result(Path(roots["store_root"]), result_id)
    except Exception:
        st.error("The saved result failed verification and is not shown.")
        return
    st.info("Earlier budgeted five-account check (September 22, 2026). Five accounts per "
            "firm copied one shared strategy signal, and stops filled at the recorded stop "
            "price. It validated the simulator; it is not the current configuration "
            "comparison, which gives each configuration and firm its own single account and "
            "strategy state.")
    render_funded_results(st, result)
    if state.get("review_folder"):
        st.caption("The review folder of documents and data was saved automatically in the "
                   "reports folder under “Funded payout reviews”.")
    elif state.get("review_error"):
        st.warning("The review folder could not be published; the verified result is kept.")
        if st.button("Publish the review folder again", key=_KEY + "publish"):
            from alpha_lab.propsim.funded.runner import publish_review

            publish_review(plan_id=study.key, store_root=Path(roots["store_root"]),
                           state_root=funded_state_root(roots),
                           reports_root=Path(roots.get("reports_root", _REPO / "reports")))
            st.rerun()


