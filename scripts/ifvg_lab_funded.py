"""Funded results (mock 02) and the configuration detail shell (mocks 03–08).

Both read ONE saved funded comparison result (verified on open) and never
recompute a stored money figure. The firm switch sets the one shared firm
selection for this result (``ifvg_lab_nav.funded_context``): the ranking, the
leader, every detail tab and Trade review follow it. Firms are never added
together.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab import html as h
from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import FundedStudy, firm_name
from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import FundedRow
from alpha_lab.agents.data_infra.ifvg.presentation.lab.names import (
    ConfigurationName,
    configuration_name,
)
from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import css_var

__all__ = [
    "DetailContext",
    "QUESTION",
    "firm_switch",
    "render_funded_detail",
    "render_funded_results",
    "study_title",
]

QUESTION = "Which configuration earned the most cash after every account cost?"
_SHOW_ALL = "ifvg_lab_v1_show_all_ranking"
_TAB_MODULES = {
    "Summary": "ifvg_lab_detail_summary",
    "Payouts and accounts": "ifvg_lab_detail_payouts",
    "Risk and simulation": "ifvg_lab_detail_risk",
    "Trades": "ifvg_lab_detail_trades",
    "Market conditions": "ifvg_lab_detail_market",
    "Settings and evidence": "ifvg_lab_detail_settings",
}


@dataclass(frozen=True)
class DetailContext:
    """What every detail tab receives: one configuration at one firm of one saved result."""

    target: dict[str, Any]
    study: FundedStudy
    store_root: str
    result_id: str
    configuration: str
    firm_key: str
    firm: str
    name: ConfigurationName
    row: FundedRow
    roots: dict[str, Any]
    context: dict[str, Any]


NAME_NOT_READABLE = "Study name not readable"


def study_title(target: dict[str, Any]) -> str:
    """The study's short name (the saved name before " — ", which says what it tested).

    The name lives only in the study's saved run record. When no run record names
    the result, the breadcrumb says so rather than showing a generic label.
    """

    name = str(target.get("name") or "").strip()
    return name.split(" — ", 1)[0].strip() or NAME_NOT_READABLE


def study_tested(target: dict[str, Any]) -> str:
    name = str(target.get("name") or "").strip()
    return name.split(" — ", 1)[1].strip() if " — " in name else ""


def _open(st_module) -> tuple[dict[str, Any], FundedStudy] | None:
    from ifvg_lab_nav import funded_target
    from ifvg_lab_ui import funded_study, show

    target = funded_target(st_module)
    if not target:
        show(h.note("No saved funded result is open. Choose one under My studies."), st_module)
        return None
    try:
        study = funded_study(target["store_root"], target["result_id"])
    except Exception as error:  # verification failure or a missing record: never shown
        show(
            h.alert(
                "This saved result could not be opened.",
                "It failed its verification or is missing, so nothing from it is shown. "
                f"({type(error).__name__})",
            ),
            st_module,
        )
        return None
    from ifvg_lab_nav import funded_context, validate_restored_context

    validate_restored_context(study, funded_context(target["result_id"], st_module), st_module)
    return target, study


def _firm_keys(study: FundedStudy) -> list[str]:
    return [key for key, _ in study.firms]


def firm_switch(st_module, study: FundedStudy, context: dict[str, Any], *, key: str) -> str:
    """The shared firm switch: sets the one firm every funded view shows."""

    from ifvg_lab_ui import switch

    firms = dict(study.firms)
    value = switch("Firm", list(firms), key=key, value=context.get("firm_key"),
                   format_func=lambda k: firms.get(k, k),
                   help="Every funded figure belongs to one firm; firms are never added "
                        "together. This choice carries to every detail view and Trade review.",
                   st_module=st_module)
    context["firm_key"] = value
    return value


def _handle(action: str | None, target: dict[str, Any], st_module) -> None:
    from ifvg_lab_nav import go, open_funded_detail

    if not action:
        return
    kind, _, value = action.partition(":")
    context_firm = None
    if kind == "library":
        go(st_module, nav="My studies", screen="list")
    elif kind == "results":
        go(st_module, nav="My studies", screen="funded")
    elif kind == "detail":
        configuration, _, firm = value.partition("|")
        context_firm = firm or None
        open_funded_detail(target, configuration, context_firm, "Summary", st_module)
    elif kind == "showall":
        st_module.session_state[_SHOW_ALL] = not st_module.session_state.get(_SHOW_ALL, False)
        st_module.rerun()


# ── funded results overview (mock 02) ─────────────────────────────────────


def _status_text(study: FundedStudy) -> tuple[str, bool]:
    from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
        proxy_micro_positions,
    )

    result = study.result
    evidence = result.get("price_evidence") or {}
    checked = int(evidence.get("position_minutes_checked") or 0)
    exact = int(evidence.get("position_minutes_rebuilt_exactly_from_prints") or 0)
    validation = result.get("validation") or {}
    passed = bool(validation.get("passed"))
    completed = sum(1 for s in (result.get("summaries_cents") or {}).values()
                    if s.get("status") == "Completed")
    parts = ["Verified" if passed else "Not verified",
             f"{exact:,} of {checked:,} position minutes on recorded trades",
             (f"Money reconciles for all {completed:,} results" if passed
              else "Money checks did not pass")]
    if proxy_micro_positions(result):
        parts.append("Micro fills priced from E-mini trades")
    corrections = len(result.get("reporting_corrections") or [])
    if corrections:
        parts.append(fmt.count(corrections, "reporting correction"))
    approved = (result.get("approval") or {}).get("approved_on")
    parts.append(f"Approved {fmt.date_long(approved)}" if approved else "No approval recorded")
    return " · ".join(parts), passed


def _meta(study: FundedStudy) -> str:
    period = study.result.get("period") or {}
    firms = [name for _, name in study.firms]
    firm_text = " and ".join(firms) if len(firms) <= 2 else ", ".join(firms)
    return (f"{fmt.chicago_long(period.get('start_utc'))} – "
            f"{fmt.chicago_long(period.get('cutoff_utc'))} · "
            f"{fmt.count(len(study.configurations), 'configuration')} · {firm_text}, compared "
            "separately · one live account at a time, lost accounts replaced at the firm's price")


def _window_cards(study: FundedStudy, leader: FundedRow | None, firm: str) -> h.Markup:
    calendar = study.calendar
    period = (f"{fmt.date_range(calendar[0], calendar[-1])} · "
              f"{fmt.count(len(calendar), 'trading day')}" if calendar
              else "Trading calendar not available")
    if leader is not None and leader.completed:
        numbers = (
            '<div style="display:flex;gap:32px;align-items:baseline;flex-wrap:wrap">'
            f'<div><div class="lab-tile-label">Leader net cash, {h.esc(firm)}</div>'
            '<div class="lab-mono" style="font-size:30px;font-weight:500">'
            f"{h.esc(fmt.money_cents(leader.net_cash_cents))}</div></div>"
            '<div><div class="lab-tile-label">Payouts</div><div class="lab-mono" '
            f'style="font-size:20px">{leader.payouts}</div></div>'
            '<div><div class="lab-tile-label">Accounts used</div><div class="lab-mono" '
            f'style="font-size:20px">{leader.accounts}</div></div></div>')
    else:
        numbers = str(h.placeholder("No configuration completed at this firm"))
    selection = h.card(h.Markup(
        '<div style="display:flex;justify-content:space-between;align-items:baseline">'
        '<div class="lab-overline" style="color:var(--lab-blue)">Selection window</div>'
        '<div class="lab-line">Used to pick configurations</div></div>'
        f'<div style="font-size:15px;color:var(--lab-body-2)">{h.esc(period)}</div>{numbers}'))
    unseen = h.card(h.Markup(
        '<div style="display:flex;justify-content:space-between;align-items:baseline">'
        '<div class="lab-overline" style="color:var(--lab-orange)">Unseen window</div>'
        '<div class="lab-line">Never used for selection</div></div>'
        '<div style="font-size:15px;color:var(--lab-body-2)">June 11, 2026 onward · protected</div>'
        '<div style="font-size:15px;color:var(--lab-body);line-height:1.5">Not yet run. The same '
        "columns fill in here once a confirmation run is approved.</div>"
        '<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">'
        '<button type="button" disabled aria-disabled="true" title="Confirmation runs aren\'t '
        'available yet" style="padding:10px 16px;min-height:44px;'
        'border:1px solid var(--lab-disabled-border);background:var(--lab-disabled-bg);'
        'color:var(--lab-disabled-text);border-radius:8px;font:inherit;font-size:14px;'
        'font-weight:500;cursor:not-allowed">Plan a confirmation run</button>'
        '<span class="lab-line">Confirmation runs aren\'t available yet</span></div>'),
        soft=True)
    return h.grid([selection, unseen], 2, gap=16)


def study_names_for(study: FundedStudy) -> dict[str, ConfigurationName]:
    """Unique readable names for every configuration of this saved result."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_matrix import (
        is_mffu_plan,
        variant_name,
    )

    if is_mffu_plan(study.plan):
        names = {}
        for variant in study.plan.configurations:
            first, second = variant_name(variant)
            names[variant.name] = ConfigurationName(first, second, first + " · " + second)
        return names
    if getattr(study.plan, "plan_schema", None) == "ifsm_correct_config_full_range_plan_v1":
        from alpha_lab.agents.data_infra.ifvg.presentation.lab.names import study_names

        parents = {row.name: json.loads(row.effective_section_json)["parent_timeframes"]
                   for row in study.plan.configurations}
        return study_names({key: study.settings(key) for key in study.configurations},
                           parent_timeframes_by_configuration=parents)
    from ifvg_lab_cache import configuration_names

    return configuration_names(str(study.store_root or ""), study.result_id)


def _ranking_table(study: FundedStudy, rows: list[FundedRow], *, show_all: bool,
                   firm_key: str) -> h.Markup:
    columns = [
        h.Column("rank", "Rank", width="52px"),
        # wide enough for both lines of a name to stay one line each at 1,440 pixels
        h.Column("config", "Configuration", width="37%"),
        h.Column("net", "Net cash", "right"),
        h.Column("payouts", "Payouts", "right"),
        h.Column("accounts", "Accounts used", "right"),
        h.Column("per", "Cash per $1 of accounts", "right"),
        h.Column("net_r", "Net R", "right"),
        h.Column("win", "Win rate", "right"),
        h.Column("dd", "Worst drawdown", "right"),
        h.Column("sharpe", "Sharpe", "right"),
        h.Column("sortino", "Sortino", "right"),
    ]
    groups = [("", 2, "blank"), ("Funded accounts (cash)", 4, ""),
              ("Strategy, no accounts", 2, "plain"), ("Funded path risk", 3, "plain")]
    shown = rows if show_all else rows[:8]
    names = study_names_for(study)
    body = []
    for row in shown:
        name = names.get(row.configuration) or configuration_name(
            study.settings(row.configuration), row.configuration)
        if row.completed:
            cells = {
                "rank": str(row.rank) if row.rank is not None else "—",
                "config": h.cell_two_lines(name.line1, name.line2),
                "net": h.Markup(f"<b>{h.esc(fmt.money_cents(row.net_cash_cents))}</b>"),
                "payouts": str(row.payouts), "accounts": str(row.accounts),
                "per": fmt.money(row.cash_per_dollar) if row.cash_per_dollar is not None
                else fmt.MISSING,
                "net_r": fmt.number(row.net_r), "win": fmt.percent(
                    row.win_rate_pct, decimals=1, already_percent=True),
                "dd": fmt.money_whole(row.worst_drawdown),
                "sharpe": fmt.number(row.sharpe), "sortino": fmt.number(row.sortino),
            }
        else:
            cells = {"rank": "—", "config": h.cell_two_lines(name.line1, name.line2),
                     "net": h.placeholder("Not completed"),
                     "payouts": h.placeholder("Unavailable")}
        body.append(h.Row(cells, tint="leader" if row.rank == 1 else None,
                          action=f"detail:{row.configuration}|{firm_key}",
                          label=f"Open {name.line1} ({name.line2})"))
    count = len(rows)
    toggle = h.link(f"Show all {count}" if not show_all else "Show the top 8", "showall")
    foot = h.Markup(f"<span>Showing {len(shown)} of {count} · Sharpe and Sortino use every "
                    "trading day, including days with no trade · click a row for its detail"
                    f"</span>{toggle}")
    return h.Markup(f'<div class="lab-ranking">'
                    f"{h.table(columns, body, groups=groups, foot=foot)}</div>")


def drop_scope(drops: dict[str, Any], firm_key: str, firm: str) -> str:
    """Where the leader stays rank 1 once each configuration's biggest payout is removed.

    "at both firms" only when the SAME configuration leads and passes at every firm;
    otherwise only the firm shown is named.
    """

    here = drops[firm_key]
    everywhere = all(d.passed and d.leader == here.leader for d in drops.values())
    if everywhere and len(drops) == 2:
        return "at both firms"
    if everywhere and len(drops) > 2:
        return "at every firm"
    return f"at {firm}"


def _checks(st_module, target, study: FundedStudy, leader: FundedRow, firm_key: str) -> None:
    from ifvg_lab_cache import drop_largest, summary_bundle
    from ifvg_lab_ui import show

    bundle = summary_bundle(target["store_root"], target["result_id"], leader.configuration,
                            firm_key)
    drops = {key: drop_largest(target["store_root"], target["result_id"], key)
             for key in _firm_keys(study)}
    here = drops[firm_key]
    if here.passed is None:
        drop_card = h.card(h.placeholder("No completed configuration"),
                           title="Drop the largest payout", sans_title=True)
    else:
        verdict = (("Pass", css_var("blue"), "check") if here.passed
                   else ("Fail", css_var("orange"), "warn"))
        where = drop_scope(drops, firm_key, firm_name(study, firm_key))
        text = (f"Still rank 1 {where} with its biggest payout removed." if here.passed
                else f"Not rank 1 at {firm_name(study, firm_key)} once each configuration's "
                     "biggest payout is removed.")
        drop_card = h.card(h.Markup(
            f'<div style="display:flex;align-items:center;gap:8px;color:{verdict[1]};'
            f'font-weight:600;font-size:15px">{h.icon(verdict[2], color=verdict[1])}'
            f"{verdict[0]}</div>"
            '<div style="font-size:14px;line-height:1.5;color:var(--lab-body)">'
            f"{h.esc(text)}</div>"),
            title="Drop the largest payout", sans_title=True)
    ranges = bundle.ranges
    if ranges:
        low, high = ranges.ranges[95]
        above = ("The whole range sits above zero." if low > 0
                 else "The range reaches below zero.")
        range_card = h.card(h.Markup(
            f'<div class="lab-mono" style="font-size:20px">{h.esc(fmt.money_whole(low))} to '
            f"{h.esc(fmt.money_whole(high))}</div>"
            '<div style="font-size:14px;line-height:1.5;color:var(--lab-body)">'
            f"Resampled from its own trades. {above}</div>"),
            title="Result per trade, 95% range", sans_title=True)
    else:
        range_card = h.card(h.placeholder("No trades"), title="Result per trade, 95% range",
                            sans_title=True)
    sharpe = bundle.sharpe
    if sharpe and sharpe.deflated is not None:
        deflated_card = h.card(h.Markup(
            f'<div class="lab-mono" style="font-size:20px">{sharpe.deflated:.2f}</div>'
            '<div style="font-size:14px;line-height:1.5;color:var(--lab-body)">'
            # correction A5: not a probability of a genuine edge or of future payouts
            f"Deflated Sharpe ratio after comparing {sharpe.tested} configurations in this "
            f"study at this firm: the probabilistic Sharpe ratio against the expected best of "
            f"{sharpe.tested} (daily results, independence assumed). Not a probability of a "
            "genuine edge or of future payouts, and it doesn't account for research done "
            "before this study.</div>"),
            title=f"Adjusted for {sharpe.tested} configurations tested", sans_title=True)
    else:
        deflated_card = h.card(h.placeholder("Not enough configurations to adjust"),
                               title="Adjusted for configurations tested", sans_title=True)
    show(h.grid([drop_card, range_card, deflated_card, _gate_card(bundle)], 4), st_module)


def _gate_card(bundle) -> h.Markup:
    gates = [g for g in bundle.gates if g.gate != "Session stability · time-block "
             "consistency · best setup's share"]
    checked = [g for g in gates if g.passed is not None]
    passed = sum(1 for g in checked if g.passed)
    missing = [g for g in gates if g.value is None]
    failed = [g for g in checked if not g.passed]
    lines = []
    for gate in failed:
        text = f"Fails {gate.gate.lower()}: {gate.value}"
        text += " days." if "under water" in gate.gate.lower() else "."
        if gate.needs_decision:
            text += " Limit awaiting your decision."
        lines.append(text)
    if missing:
        lines.append(f"{len(missing)} not in this study's export.")
    value = f"{passed} of {len(checked)} pass" if checked else "Not in export"
    return h.card(h.Markup(
        f'<div class="lab-mono" style="font-size:20px">{h.esc(value)}</div>'
        '<div style="font-size:14px;line-height:1.5;color:var(--lab-body)">'
        f'{h.esc(" ".join(lines))}</div>'), title="Quality gates", sans_title=True)


def render_funded_results(st_module, roots) -> None:
    from ifvg_lab_cache import summary_bundle
    from ifvg_lab_nav import funded_context
    from ifvg_lab_ui import clickable, show

    opened = _open(st_module)
    if opened is None:
        return
    target, study = opened
    context = funded_context(target["result_id"], st_module)
    _result_versions(st_module, target, detail=False)
    firms = dict(study.firms)
    if context.get("firm_key") not in firms and firms:
        context["firm_key"] = next(iter(firms))
    action = clickable(
        h.page_header(
            study_title(target),
            crumbs=[("My studies", "library"), ("Funded comparisons", None)],
            subtitle=QUESTION,
            meta=_meta(study),
        ),
        key="funded_header",
        st_module=st_module,
    )
    _handle(action, target, st_module)
    status, passed = _status_text(study)
    if target.get("status") is None:
        show(
            h.note(
                "This result's saved run record wasn't found, so its run status couldn't "
                "be confirmed. The figures below are read from the verified saved result.",
                "orange",
            ),
            st_module,
        )
    elif target.get("status") != "Completed":
        show(
            h.alert(
                f"This comparison is {str(target['status']).lower()}.",
                "Configurations that did not complete are listed as not completed, never "
                "as zero. Read the details before relying on any figure.",
            ),
            st_module,
        )
    show(h.status_line(status, kind="check" if passed else "warn"), st_module)
    with st_module.expander("Details", expanded=False):
        _status_details(st_module, study)
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_matrix import (
        matrix_rows,
        saved_analysis_tables,
    )

    saved_matrix = matrix_rows(study)
    if saved_matrix:
        from ifvg_lab_mffu_views import render_comparison

        render_comparison(st_module, target, study, context)
        with st_module.expander("Saved 64-intent matrix and status", expanded=False):
            st_module.caption(
                "Each row is one frozen MyFundedFutures intent. Unfinished rows have no "
                "cash value; source hashes and every axis are shown from the saved plan."
            )
            st_module.dataframe(saved_matrix, use_container_width=True, hide_index=True, height=600)
        saved_analysis = saved_analysis_tables(study)
        if saved_analysis:
            with st_module.expander("MFFU matched cash effects and receipt waits", expanded=False):
                st_module.caption(
                    "All effects use received payouts less account purchases. Matching "
                    "configurations share the inspected sample; unavailable cells stay blank. "
                    "Receipt waits show calendar and evaluated trading days separately."
                )
                st_module.markdown("**Matched policy comparisons**")
                st_module.dataframe(
                    saved_analysis["matched_pairs"],
                    use_container_width=True,
                    hide_index=True,
                    height=310,
                )
                st_module.markdown("**Factor interactions**")
                st_module.dataframe(
                    saved_analysis["interactions"],
                    use_container_width=True,
                    hide_index=True,
                    height=310,
                )
                st_module.markdown("**Receipt waiting intervals**")
                st_module.dataframe(
                    saved_analysis["waiting"], use_container_width=True, hide_index=True, height=310
                )
        return
    from ifvg_lab_cache import ranking
    from ifvg_lab_ui import pending_switch

    switch_key = f"firm_{target['result_id'][:16]}"
    firm_key = pending_switch(switch_key, list(firms), context["firm_key"], st_module)
    rows = ranking(target["store_root"], target["result_id"], firm_key)
    leader = next((r for r in rows if r.completed), None)
    show(_window_cards(study, leader, firms.get(firm_key, firm_key)), st_module)
    left, right = st_module.columns([3, 1.4], vertical_alignment="bottom")
    with left:
        show(h.section_title("All configurations, ranked by net cash"), st_module)
    with right:
        firm_key = firm_switch(st_module, study, context, key=switch_key)
    show(
        h.Markup(
            '<div class="lab-line">The firm switch drives this table and every detail '
            "view. Firms are never added together.</div>"
        ),
        st_module,
    )
    action = clickable(
        _ranking_table(
            study, rows, firm_key=firm_key, show_all=st_module.session_state.get(_SHOW_ALL, False)
        ),
        key=f"ranking_{firm_key}",
        st_module=st_module,
    )
    _handle(action, target, st_module)
    for row in rows:
        if not row.completed:
            show(h.note(f"{row.configuration}: not completed — {row.reason}"), st_module)
    if leader is None:
        return
    show(h.section_title("Checks on the leader"), st_module)
    _checks(st_module, target, study, leader, firm_key)
    bundle = summary_bundle(
        target["store_root"], target["result_id"], leader.configuration, firm_key
    )
    count = len(
        _findings(
            bundle,
            target,
            leader.configuration,
            firm_key,
            dict(study.firms).get(firm_key, firm_key),
        )
    )
    action = clickable(
        h.Markup(
            f'<a href="#" data-action="detail:{h.esc(leader.configuration)}|{h.esc(firm_key)}" '
            'class="lab" style="display:inline-flex;align-items:center;gap:10px;'
            "background:var(--lab-panel);border:1px solid var(--lab-rule);border-radius:10px;"
            'padding:10px 16px;text-decoration:none;color:var(--lab-ink);font-size:15px">'
            f"{h.badge(fmt.count(count, 'finding'), 'orange')}"
            "Open the leader's full detail</a>"
        ),
        key="findings_link",
        st_module=st_module,
    )
    _handle(action, target, st_module)


def _status_details(st_module, study: FundedStudy) -> None:
    """The existing verification notices, in full."""

    from ifvg_lab_ui import show

    from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
        present_comparison,
    )

    view = present_comparison(study.result)
    items = [h.note(fmt.display_words(line.text),
                    "orange" if line.level in ("warning", "error") else "")
             for line in view.status]
    items.append(h.note(view.future_note))
    show(h.Markup('<div class="lab" style="display:flex;flex-direction:column;gap:8px">'
                  + "".join(items) + "</div>"), st_module)


def _findings(bundle, target: dict[str, Any], configuration: str, firm_key: str,
              firm: str) -> list:
    """The leader's findings: exactly what its Summary tab passes (one shared helper)."""

    from ifvg_lab_cache import pair_findings

    return pair_findings(bundle, target["store_root"], target["result_id"], configuration,
                         firm_key, firm)


# ── configuration detail shell (mocks 03–08) ──────────────────────────────


def _result_versions(st_module, target, *, detail: bool) -> None:
    versions = target.get("versions") or {}
    if not versions:
        return
    from ifvg_lab_nav import funded_context, open_funded_detail, open_funded_results

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.external_catalog import (
        resolve_registered_result,
    )

    keys = list(versions)
    current = target["result_id"]
    selected = st_module.selectbox(
        "Result version",
        keys,
        index=keys.index(current),
        format_func=lambda key: (
            (versions[key].get("version_label") or key[:16])
            if isinstance(versions[key], dict)
            else str(versions[key])
        ),
        key=f"mffu_result_version_{'detail' if detail else 'results'}_{current[:16]}",
    )
    qualification = (target.get("catalog_binding") or {}).get("qualification")
    if qualification:
        st_module.caption(qualification)
    if selected != current:
        resolved = resolve_registered_result(selected)
        if resolved is None:
            st_module.error("This version's registered verified result is unavailable.")
            return
        if detail:
            context = funded_context(current, st_module)
            open_funded_detail(
                resolved,
                context.get("configuration"),
                context.get("firm_key"),
                context.get("tab", "Summary"),
                st_module,
            )
        else:
            open_funded_results(resolved, st_module)


def render_funded_detail(st_module, roots) -> None:
    from ifvg_lab_cache import ranking
    from ifvg_lab_nav import DETAIL_TABS, funded_context
    from ifvg_lab_ui import clickable, pending_switch, tabs

    opened = _open(st_module)
    if opened is None:
        return
    target, study = opened
    context = funded_context(target["result_id"], st_module)
    _result_versions(st_module, target, detail=True)
    firms = dict(study.firms)
    if context.get("firm_key") not in firms and firms:
        context["firm_key"] = next(iter(firms))
    switch_key = f"firm_{target['result_id'][:16]}"
    firm_key = context["firm_key"] = pending_switch(
        switch_key, list(firms), context["firm_key"], st_module
    )
    rows = ranking(target["store_root"], target["result_id"], firm_key)
    configuration = context.get("configuration")
    known = set(study.configurations) | {r.configuration for r in rows}
    if configuration not in known:
        leader = next((r for r in rows if r.completed), None)
        configuration = context["configuration"] = (
            leader.configuration if leader else study.configurations[0]
        )
    row = next((r for r in rows if r.configuration == configuration), None)
    name = study_names_for(study).get(configuration) or configuration_name(
        study.settings(configuration), configuration
    )
    rank = (
        f"Rank {row.rank} at {firms.get(firm_key, firm_key)}"
        if row is not None and row.rank is not None
        else f"Not completed at {firms.get(firm_key, firm_key)}"
    )
    action = clickable(
        h.page_header(
            name.line1,
            detail=True,
            crumbs=[("My studies", "library"), (study_title(target), "results"), (rank, None)],
            subtitle=name.summary,
        ),
        key="detail_header",
        st_module=st_module,
    )
    _handle(action, target, st_module)
    tab = context.get("tab") if context.get("tab") in DETAIL_TABS else "Summary"
    tab = context["tab"] = pending_switch("tabs_detail", DETAIL_TABS, tab, st_module)
    module = importlib.import_module(_TAB_MODULES[tab])
    header_right = getattr(module, "header_right", None)
    left, right = st_module.columns([3.2, 1.2], vertical_alignment="bottom")
    with left:
        tabs(DETAIL_TABS, key="detail", value=tab, st_module=st_module)
    with right:
        if header_right is None:
            firm_switch(st_module, study, context, key=switch_key)
    if row is None:
        from ifvg_lab_ui import show

        show(h.note("This configuration was not tested with this firm."), st_module)
        return
    ctx = DetailContext(
        target=target,
        study=study,
        store_root=target["store_root"],
        result_id=target["result_id"],
        configuration=configuration,
        firm_key=firm_key,
        firm=firms.get(firm_key, firm_key),
        name=name,
        row=row,
        roots=roots,
        context=context,
    )
    if header_right is not None:
        with right:
            header_right(st_module, ctx)
    if not row.completed:
        from ifvg_lab_ui import show

        show(
            h.alert(
                "This configuration did not complete at this firm.",
                f"{row.reason} It has no figures; it is not a zero result.",
            ),
            st_module,
        )
        if tab not in ("Settings and evidence",):
            return
    module.render(st_module, ctx)
