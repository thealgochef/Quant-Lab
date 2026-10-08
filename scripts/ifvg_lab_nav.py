"""Navigation state shared by every redesigned screen (both applications).

One place defines where the workspace is (rail destination + screen), which
saved funded result is open, and the ONE firm / configuration / account /
detail-tab selection per saved result that every funded view reads (repair R2:
the ranking, the detail and Trade review always show the same firm).

Deep links: the current screen is mirrored in the page address
(``?view=funded&app=ifsm&result=…&firm=…&config=…&tab=…``) so a reload or a
shared link opens the same screen. Links only select what to show; they never
save, approve or launch anything.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

__all__ = [
    "DETAIL_TABS",
    "FUNDED_TARGET",
    "NAV",
    "SCREEN",
    "SELECTED",
    "app_roots",
    "apply_deep_link",
    "current_app",
    "funded_context",
    "funded_target",
    "go",
    "open_funded_detail",
    "open_funded_results",
    "open_trade_review",
    "sync_url",
]

NAV = "ifvg_workspace_destination"
SCREEN = "ifvg_workspace_screen"
SELECTED = "ifvg_workspace_selected_study"
FUNDED_TARGET = "ifvg_lab_v1_funded_target"
REVIEW_TARGET = "ifvg_lab_v1_review_target"
_LINK_APPLIED = "ifvg_lab_v1_deep_link_applied"
CONTEXT_KEY = "funded_comparison_v1_selected_context"  # shared with the repair-R2 screen
PREFERENCES_VERSION = "funded_result_view_preferences_v1"

DETAIL_TABS = ("Summary", "Payouts and accounts", "Risk and simulation", "Trades",
               "Market conditions", "Settings and evidence")
_TAB_SLUGS = {"summary": "Summary", "payouts": "Payouts and accounts",
              "risk": "Risk and simulation", "trades": "Trades",
              "market": "Market conditions", "settings": "Settings and evidence"}
_VIEWS = {"library": ("My studies", "list"), "mlphase": ("My studies", "ml_phase"),
          "funded": ("My studies", "funded"),
          "detail": ("My studies", "funded_detail"), "review": ("Trade review", None),
          "new": ("New study", "new_funded"), "approve": ("New study", "approve_funded"),
          "types": ("New study", "new")}


def app_roots(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Both applications' store roots, built from one repository root.

    ``main`` is ``scripts/dashboard.py``'s IFVG workspace; ``ifsm`` is the
    dedicated ``scripts/run_ifsm_research_ui.py`` workspace. Built explicitly
    (never from module globals the dedicated app overrides).
    """

    repo = Path(repo_root)
    work = repo / "data/ifsm_ui_replication"
    return {
        "main": {
            "store_root": repo / "data/ifvg_datasets/search/v1",
            "store_roots": {"research": repo / "data/ifvg_datasets/search/v1",
                            "test": repo / "data/ifvg_datasets/search_test/v1"},
            "state_root": repo / "data/ifvg_search_jobs",
            "draft_root": repo / "data/ifvg_study_drafts",
            "pipeline_state_root": repo / "data/ifvg_pipeline_jobs",
            "context_catalog": repo / "data/ifvg_experiments/context_v1_catalog.json",
            "context_run_root": repo / "data/ifvg_experiments/context_v1",
            "repo_root": repo, "namespace_class": "research",
        },
        "ifsm": {
            "store_root": work / "search/v1",
            "store_roots": {"research": work / "search/v1", "test": work / "search_test/v1"},
            "state_root": work / "jobs",
            "draft_root": work / "drafts",
            "pipeline_state_root": work / "pipeline_jobs",
            "context_catalog": work / "context_catalog.json",
            "context_run_root": work / "context_runs",
            "repo_root": repo, "namespace_class": "research",
        },
    }


def current_app(roots: dict[str, Any]) -> str:
    """``ifsm`` when the workspace runs on the dedicated application's stores."""

    draft = Path(roots.get("draft_root") or "")
    return "ifsm" if "ifsm_ui_replication" in draft.as_posix() else "main"


def go(st_module=st, *, nav: str | None = None, screen: str | None = None,
       selected: Any = None, rerun: bool = True) -> None:
    if nav is not None:
        st_module.session_state[NAV] = nav
    if screen is not None:
        st_module.session_state[SCREEN] = screen
    if selected is not None:
        st_module.session_state[SELECTED] = selected
    if rerun:
        st_module.rerun()


def funded_target(st_module=st) -> dict[str, Any] | None:
    """The open saved funded result: store_root, result_id, plan_id, name, app, status."""

    return st_module.session_state.get(FUNDED_TARGET)


def funded_context(result_id: str, st_module=st) -> dict[str, Any]:
    """The ONE firm / configuration / account / tab selection for this saved result."""

    contexts = st_module.session_state.setdefault(CONTEXT_KEY, {})
    if result_id not in contexts:
        try:
            saved = _preferences().get("contexts", {}).get(result_id, {})
            contexts[result_id] = dict(saved) if isinstance(saved, dict) else {}
        except Exception:
            contexts[result_id] = {}
            st_module.session_state[LINK_NOTE] = "Saved viewing preferences could not be restored."
    return contexts[result_id]


def _preferences_path() -> Path:
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.external_catalog import catalog_path

    return catalog_path().with_name("funded_view_preferences.json")


def _preferences() -> dict[str, Any]:
    path = _preferences_path()
    if not path.exists():
        return {"schema": PREFERENCES_VERSION, "contexts": {}}
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("schema") != PREFERENCES_VERSION:
        raise ValueError("unsupported view preferences schema")
    return saved


def save_view_preferences(st_module=st) -> None:
    """Persist viewing choices separately from strategy drafts and economic records."""
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.external_catalog import _write

    contexts = st_module.session_state.get(CONTEXT_KEY) or {}
    if not contexts:
        return
    saved = _preferences()
    saved["contexts"].update(contexts)
    target = funded_target(st_module) or {}
    saved["selected_result_id"] = target.get("result_id")
    path = _preferences_path()
    data = json.dumps(saved, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if not path.exists() or path.read_text(encoding="utf-8") != data:
        _write(path, saved)


def validate_restored_context(study, context: dict[str, Any], st_module=st) -> None:
    """Clear invalid restored scoped selections with an explicit explanation."""
    problems = []
    if context.get("configuration") and context["configuration"] not in study.configurations:
        context.pop("configuration", None)
        context.pop("account", None)
        problems.append("configuration")
    firms = dict(study.firms)
    if context.get("firm_key") and context["firm_key"] not in firms:
        context.pop("firm_key", None)
        context.pop("account", None)
        problems.append("firm")
    account = context.get("account")
    if isinstance(account, dict):
        pair = f"{context.get('configuration')}|{context.get('firm_key')}"
        numbers = {str(row.get("account_number")) for row in (
            study.result.get("tables") or {}).get("accounts", []) if row.get("pair_id") == pair}
        if not numbers:
            numbers = {str(row.get("account_number")) for row in study.trades_by_pair.get(pair, ())}
        if account.get("pair") != pair or str(account.get("number")) not in numbers:
            context.pop("account", None)
            problems.append("account")
    if problems:
        st_module.session_state[LINK_NOTE] = (
            "Saved " + ", ".join(problems)
            + " selection is unavailable in this result version and was cleared.")


def open_funded_results(target: dict[str, Any], st_module=st, *, rerun: bool = True) -> None:
    st_module.session_state[FUNDED_TARGET] = dict(target)
    go(st_module, nav="My studies", screen="funded", selected=target.get("study_key"),
       rerun=rerun)


def open_funded_detail(target: dict[str, Any], configuration: str, firm_key: str | None = None,
                       tab: str = "Summary", st_module=st, *, rerun: bool = True) -> None:
    st_module.session_state[FUNDED_TARGET] = dict(target)
    context = funded_context(str(target["result_id"]), st_module)
    context["configuration"] = configuration
    if firm_key:
        context["firm_key"] = firm_key
    context["tab"] = tab
    go(st_module, nav="My studies", screen="funded_detail", selected=target.get("study_key"),
       rerun=rerun)


def open_trade_review(target: dict[str, Any], *, configuration: str, firm_key: str,
                      account_number: int | None = None, trade_seq: int | None = None,
                      back_tab: str | None = "Trades", st_module=st, rerun: bool = True) -> None:
    """Open Trade review on one funded trade (or the pair's first trade)."""

    st_module.session_state[FUNDED_TARGET] = dict(target)
    context = funded_context(str(target["result_id"]), st_module)
    context.update({"configuration": configuration, "firm_key": firm_key})
    if account_number is not None:
        context["account"] = {"pair": f"{configuration}|{firm_key}", "number": account_number}
    st_module.session_state[REVIEW_TARGET] = {
        "source": "Funded trades", "result_id": target["result_id"],
        "configuration": configuration, "firm_key": firm_key,
        "account_number": account_number, "trade_seq": trade_seq, "back_tab": back_tab}
    go(st_module, nav="Trade review", rerun=rerun)


# ── deep links ────────────────────────────────────────────────────────────


LINK_NOTE = "ifvg_lab_v1_link_note"


def _digits(value: Any) -> int | None:
    return int(value) if value is not None and str(value).isdigit() else None


def _saved_target(result_id: str, app: str, roots: dict[str, Any]) -> dict[str, Any] | None:
    """The saved run behind a result id (name, status, plan), searched in both applications."""

    from ifvg_workspace import funded_result_target

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.external_catalog import (
        resolve_registered_result,
    )
    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import load_studies

    all_roots = app_roots(Path(roots.get("repo_root") or Path.cwd()))
    # Resolve exact registered versions first; broken bindings propagate to the
    # visible link failure and must not fall back to another similarly named run.
    published = resolve_registered_result(result_id, store_root=all_roots["main"]["store_root"])
    if published is not None:
        return published
    order = [app] + [key for key in all_roots if key != app]
    for key in order:
        app_root = roots if key == current_app(roots) else all_roots[key]
        try:
            studies, _issues = load_studies(app_root)
        except Exception:
            continue
        for study in studies:
            saved_result = str((study.state or {}).get("result_id"))
            if study.kind == "funded_comparison" and saved_result == result_id:
                return funded_result_target(study, app_root, app=key)
    return None


def _external_mffu_review_target(result_id: str, app: str,
                                 roots: dict[str, Any]) -> dict[str, Any] | None:
    """Open one explicitly pinned external MFFU result for read-only Lab review.

    This route leaves the normal application stores and study discovery intact.
    Both environment fields are required, and the selected result is verified
    before its external store is put into session state.
    """
    if app != "ifsm":
        return None
    store_text = os.environ.get("IFSM_MFFU_REVIEW_STORE_ROOT")
    allowed_result_id = os.environ.get("IFSM_MFFU_REVIEW_RESULT_ID")
    if not store_text and not allowed_result_id:
        return None
    if not store_text or not allowed_result_id:
        raise ValueError("external MFFU review requires both environment fields")
    if len(allowed_result_id) != 64 or any(
        char not in "0123456789abcdef" for char in allowed_result_id
    ):
        raise ValueError("external MFFU review result ID must be a full SHA-256 key")
    if allowed_result_id != result_id:
        return None
    repo_root = Path(roots.get("repo_root") or Path(__file__).resolve().parents[1]).resolve()
    store_root = Path(store_text).resolve(strict=True)
    if not store_root.is_dir() or store_root.is_relative_to(repo_root):
        raise ValueError("external MFFU review store must be a directory outside the repository")

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (
        open_funded_study,
    )
    from alpha_lab.propsim.funded.mffu_batch_review import _assert_result

    study = open_funded_study(store_root, result_id)
    if study.plan_id is None or study.plan is None:
        raise ValueError("external MFFU review has no verified saved plan")
    _assert_result(study.plan, study.plan_id, study.result)
    return {
        "result_id": result_id,
        "plan_id": study.plan_id,
        "store_root": str(store_root),
        "app": "ifsm",
        "study_key": result_id,
        "name": "IFSM MFFU context batch",
        "status": "Completed",
        "external_review_only": True,
    }


def apply_deep_link(roots: dict[str, Any], st_module=st) -> None:
    """Once per browser session: open the screen named in the page address.

    A link only selects what to show. A link that can't be read opens My studies
    with a one-sentence note; nothing is saved, approved or launched.
    """

    if st_module.session_state.get(_LINK_APPLIED):
        return
    st_module.session_state[_LINK_APPLIED] = True
    try:
        _apply_link(dict(st_module.query_params), roots, st_module)
    except Exception:
        st_module.session_state[NAV] = "My studies"
        st_module.session_state[SCREEN] = "list"
        st_module.session_state[LINK_NOTE] = (
            "This link could not be opened, so My studies is shown instead. Nothing was changed.")


def _apply_link(params: dict[str, Any], roots: dict[str, Any], st_module) -> None:
    view = params.get("view")
    if view not in _VIEWS:
        return
    nav, screen = _VIEWS[view]
    st_module.session_state[NAV] = nav
    if screen:
        st_module.session_state[SCREEN] = screen
    result_id = params.get("result")
    if not result_id:
        return
    result_id = str(result_id)
    if not all(c in "0123456789abcdef" for c in result_id) or len(result_id) < 16:
        raise ValueError("not a saved result id")
    app = params.get("app") if params.get("app") in ("main", "ifsm") else current_app(roots)
    if view == "mlphase":
        from alpha_lab.propsim.funded.ml_phase.catalog import registered_reports

        pointers, _ = registered_reports(app_roots(Path(roots["repo_root"]))[app]["store_root"])
        pointer = next(p for p in pointers if p["report_id"] == result_id)
        st_module.session_state["ifvg_ml_phase_pointer"] = pointer
        st_module.session_state[SELECTED] = result_id
        if params.get("config"):
            funded_context(result_id, st_module)["ml_operation"] = str(params["config"])
        return
    target = _external_mffu_review_target(result_id, app, roots)
    if target is None:
        target = _saved_target(result_id, app, roots)
    if target is None:
        # no saved run names this result: open it read-only, status unconfirmed
        all_roots = app_roots(Path(roots.get("repo_root") or Path.cwd()))
        target = {"result_id": result_id, "store_root": str(all_roots[app]["store_root"]),
                  "app": app, "study_key": result_id, "name": "", "status": None}
    st_module.session_state[FUNDED_TARGET] = target
    context = funded_context(result_id, st_module)
    if params.get("firm"):
        context["firm_key"] = str(params["firm"])
    if params.get("config"):
        context["configuration"] = str(params["config"])
    tab = _TAB_SLUGS.get(str(params.get("tab") or "").lower())
    if tab:
        context["tab"] = tab
    if view == "review":
        st_module.session_state[REVIEW_TARGET] = {
            "source": "Funded trades", "result_id": result_id,
            "configuration": params.get("config"), "firm_key": params.get("firm"),
            "account_number": _digits(params.get("account")),
            "trade_seq": _digits(params.get("trade")),
            "point_in_time": params.get("mode") == "point", "back_tab": "Trades"}


def sync_url(st_module=st) -> None:
    """Mirror the current screen in the page address (no rerun, nothing saved)."""

    nav = st_module.session_state.get(NAV, "My studies")
    screen = st_module.session_state.get(SCREEN, "list")
    target = funded_target(st_module) or {}
    params: dict[str, str] = {}
    if screen == "ml_phase":
        pointer = st_module.session_state.get("ifvg_ml_phase_pointer") or {}
        params = {"view": "mlphase", "result": pointer.get("report_id", "")}
        selected = funded_context(params["result"], st_module)
        if selected.get("ml_operation"):
            params["config"] = selected["ml_operation"]
    elif nav == "Trade review":
        params["view"] = "review"
    elif nav == "New study":
        params["view"] = {"approve_funded": "approve", "new": "types"}.get(screen, "new")
    elif screen in ("funded", "funded_detail") and target.get("result_id"):
        params["view"] = "funded" if screen == "funded" else "detail"
    else:
        params["view"] = "library"
    if params["view"] in ("funded", "detail", "review") and target.get("result_id"):
        params["app"] = str(target.get("app") or "main")
        params["result"] = str(target["result_id"])
        context = funded_context(str(target["result_id"]), st_module)
        if context.get("firm_key"):
            params["firm"] = str(context["firm_key"])
        if params["view"] != "funded" and context.get("configuration"):
            params["config"] = str(context["configuration"])
        if params["view"] == "detail":
            slug = {v: k for k, v in _TAB_SLUGS.items()}.get(context.get("tab") or "Summary")
            params["tab"] = slug or "summary"
    if params["view"] == "review":
        # Trade review publishes what it shows now (ifvg_lab_trade_review.CURRENT)
        current = st_module.session_state.get("ifvg_lab_v1_review_current") or {}
        if current.get("result_id") and current.get("result_id") == target.get("result_id"):
            for name, key in (("config", "configuration"), ("firm", "firm_key"),
                              ("account", "account"), ("trade", "trade")):
                if current.get(key) not in (None, ""):
                    params[name] = str(current[key])
            if current.get("point_in_time"):
                params["mode"] = "point"
    if params["view"] in ("new", "approve"):
        draft_id = st_module.session_state.get("ifvg_study_v1_draft_id")
        if draft_id:
            params["draft"] = str(draft_id)
    try:
        if dict(st_module.query_params) != params:
            st_module.query_params.from_dict(params)
    except Exception:  # headless tests without a browser address
        pass
    try:
        save_view_preferences(st_module)
    except Exception:
        st_module.session_state[LINK_NOTE] = "Viewing preferences could not be saved."
