"""Normal-dashboard screens for the versioned MFFU reporting companion."""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
import streamlit as st

from alpha_lab.agents.data_infra.ifvg.presentation.lab import mffu_gamma as gamma
from alpha_lab.agents.data_infra.ifvg.presentation.lab import mffu_lenses as lenses
from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_matrix import policy_label
from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_reporting import (
    load_report,
    report_folder,
    source_hashes,
)
from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import palette


@st.cache_resource(show_spinner="Loading verified reporting views…")
def _report(store_root: str, result_id: str, versions: tuple) -> dict:
    return load_report(Path(store_root), result_id)


def reporting(store_root: str, result_id: str) -> dict:
    folder = report_folder(Path(store_root), result_id)
    paths = (
        folder / "report.json",
        folder / "envelope.json",
        Path(store_root) / "funded_comparison_results" / result_id / "envelope.json",
    )
    paths += tuple(sorted((folder / "producer_sources").glob("*.py")))
    versions = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
    versions += (tuple(sorted(source_hashes().items())),)
    return _report(store_root, result_id, versions)


def _load_or_notice(st_module, store_root, result_id):
    try:
        return reporting(store_root, result_id)
    except (OSError, ValueError, KeyError):
        st_module.info(
            "The verified reporting companion for this result version is not available. "
            "Its saved economic result remains accessible."
        )
        return None


def _display(rows, fields):
    display = []
    for source in rows:
        row = {}
        for key in fields:
            value = source.get(key)
            if key in lenses.AXES and value is not None:
                value = policy_label(key, str(value))
            elif key.endswith("_cents") and value is not None:
                value = _money(value)
            elif (
                key in {"entry_date_coverage", "largest_account_receipt_share"}
                and value is not None
            ):
                value = f"{value:.1%}"
            elif key == "average_entries_per_evaluated_date" and value is not None:
                value = f"{value:.3f}"
            elif key.endswith("_utc") and value is not None:
                value = _chicago(value)
            elif value is None:
                value = (
                    source.get("measurement_status", {}).get(key, "Unavailable").replace("_", " ")
                )
            label = lenses.LABELS.get(key, key.replace("_", " ").title())
            row[label.replace("(UTC)", "(Chicago)")] = value
        display.append(row)
    return display


def _money(value):
    dollars = (Decimal(str(value)) / 100).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"${dollars:,.2f}"


def _number(value, places=2):
    if value is None:
        return "unavailable"
    return f"{value:,.{places}f}".rstrip("0").rstrip(".")


def _nearest_level(values, *, above):
    if not values:
        return "unavailable"
    price, name = min(values) if above else max(values)
    return f"{name} at {price:,.2f} points"


def _chicago(value):
    return gamma.stamp(value).tz_convert("America/Chicago").strftime("%B %d, %Y %I:%M:%S %p %Z")


def _target_text(metric, value):
    if metric.endswith("_cents"):
        return _money(value)
    if metric == "entry_date_coverage":
        return f"{value:.1%}"
    if isinstance(value, (list, tuple)):
        return f"{value[0]:g}–{value[1]:g}"
    return str(value)


def _initial_value(st_module, key, value):
    """A restored widget owns its value; supply a default only on first creation."""
    return {} if key in st_module.session_state else {"value": value}


def render_comparison(st_module, target, study, context) -> None:
    report = _load_or_notice(st_module, target["store_root"], target["result_id"])
    if report is None:
        return
    rows = report["lenses"]
    generation = int(context.get("mffu_comparison_widget_generation", 0))
    prefix = f"mffu_lens_{study.result_id[:16]}_{generation}_"
    state = context.setdefault("mffu_comparison", {})
    st_module.markdown("### Compare configurations")
    st_module.caption(
        "Full saved operation · actual funded entries · MyFundedFutures. "
        "Speed clocks start at the original operation opening; these preferences "
        "do not change a strategy or add cash across alternatives."
    )
    lens = st_module.selectbox(
        "Compare by",
        list(lenses.LENSES),
        index=list(lenses.LENSES).index(state.get("lens", "total_cash")),
        format_func=lambda k: lenses.LENSES[k][0],
        key=prefix + "lens",
    )
    state["lens"] = lens
    targets = dict(state.get("targets", {}))
    with st_module.expander("Fit my targets", expanded=bool(targets)):
        st_module.caption(
            "All targets are optional and combined with AND. Days mean Chicago "
            "calendar-date days; no-entry stretches count evaluated dates."
        )
        if st_module.button("Around one trade/day", key=prefix + "one_trade"):
            targets["average_entries_per_evaluated_date"] = [0.8, 1.3]
            state["targets"] = targets
            for suffix, value in (("enabled", True), ("lower", 0.8), ("upper", 1.3)):
                st_module.session_state[prefix + "average_entries_per_evaluated_date_" + suffix] = (
                    value
                )
            st_module.rerun()
        columns = st_module.columns(2)
        for i, (metric, (operator, label, unit)) in enumerate(lenses.TARGETS.items()):
            with columns[i % 2]:
                active = st_module.checkbox(
                    label, key=prefix + metric + "_enabled",
                    **_initial_value(st_module, prefix + metric + "_enabled", metric in targets),
                )
                if not active:
                    targets.pop(metric, None)
                    continue
                prior = targets.get(metric, [0.8, 1.3] if operator == "range" else 0.0)
                if operator == "range":
                    low = st_module.number_input(
                        "Minimum entries/date",
                        **_initial_value(st_module, prefix + metric + "_lower", float(prior[0])),
                        min_value=0.0,
                        step=0.1,
                        key=prefix + metric + "_lower",
                    )
                    high = st_module.number_input(
                        "Maximum entries/date",
                        **_initial_value(st_module, prefix + metric + "_upper", float(prior[1])),
                        min_value=0.0,
                        step=0.1,
                        key=prefix + metric + "_upper",
                    )
                    targets[metric] = [low, high]
                else:
                    scale = 100 if unit == "cents" else 0.01 if unit == "fraction" else 1
                    display_value = float(prior) / scale
                    shown_unit = (
                        "dollars"
                        if unit == "cents"
                        else "percent of evaluated dates"
                        if unit == "fraction"
                        else unit
                    )
                    value = st_module.number_input(
                        shown_unit.title(),
                        **_initial_value(st_module, prefix + metric + "_value", display_value),
                        max_value=100.0 if unit == "fraction" else None,
                        step=1.0,
                        key=prefix + metric + "_value",
                    )
                    targets[metric] = (
                        int(
                            (Decimal(str(value)) * scale).quantize(
                                Decimal(1), rounding=ROUND_HALF_UP
                            )
                        )
                        if unit == "cents"
                        else value * scale
                    )
        state["targets"] = targets
    search = st_module.text_input(
        "Search configurations", value=state.get("search", ""), key=prefix + "search"
    )
    state["search"] = search
    categories = dict(state.get("categories", {}))
    with st_module.expander("Schedule and policy filters", expanded=any(categories.values())):
        columns = st_module.columns(4)
        for i, axis in enumerate(lenses.AXES):
            available = sorted({str(r.get(axis)) for r in rows if r.get(axis) is not None})
            with columns[i % 4]:
                categories[axis] = st_module.multiselect(
                    axis.replace("_", " ").title(),
                    available,
                    default=[v for v in categories.get(axis, []) if v in available],
                    key=prefix + axis,
                    format_func=lambda code, axis=axis: policy_label(axis, code),
                )
        state["categories"] = categories
    if st_module.button("Reset comparison preferences", key=prefix + "reset"):
        context["mffu_comparison"] = {}
        context["mffu_comparison_widget_generation"] = generation + 1
        for key in list(st_module.session_state):
            if str(key).startswith(prefix):
                del st_module.session_state[key]
        st_module.rerun()
    view = lenses.filter_rows(
        rows, lens=lens, targets=targets, search=search, categories=categories
    )
    for error in view["errors"]:
        st_module.error(error)
    st_module.markdown(f"**{view['matches']} of {view['total']} configurations match**")
    if targets:
        active = "; ".join(
            f"{lenses.TARGETS[k][1]}: {_target_text(k, v)}" for k, v in targets.items()
        )
        st_module.caption("Active targets: " + active)
    st_module.caption(
        f"{view['failed']} fail a known target; {view['unknown']} cannot verify "
        "at least one target. These counts can overlap."
    )
    closest = False
    if view["matches"] == 0 and not view["errors"]:
        st_module.info("No configuration meets these targets. Your limits have been retained.")
        closest = st_module.checkbox(
            "Show closest matches", value=state.get("closest", False), key=prefix + "closest"
        )
        state["closest"] = closest
        if closest:
            view = lenses.filter_rows(
                rows, lens=lens, targets=targets, search=search, categories=categories, closest=True
            )
            st_module.caption(
                "Ordered by count of failed requirements, then the selected lens and "
                "stable configuration ID. Unverifiable targets are listed separately. "
                "Every miss is shown; no weighted score or relaxed target."
            )
    primary = (("net_received_cash_cents", "acquisition_spend_cents", "accounts_bought")
               if lens == "total_cash" else lenses.VISIBLE[lens])
    fields = list(
        dict.fromkeys(
            (
                "configuration_id",
                "geometry",
                *primary,
                "overall_cash_rank",
                "net_received_cash_cents",
                "acquisition_spend_cents",
                "accounts_bought",
                *lenses.VISIBLE[lens],
                "schedule",
                "exit",
                "sizing",
                "review_status",
            )
        )
    )
    display = _display(view["rows"], fields)
    if closest:
        for projected, source in zip(display, view["rows"], strict=True):
            projected["Requirements missed"] = "; ".join(
                f"{lenses.TARGETS[v['metric']][1]}: "
                f"actual {_target_text(v['metric'], v['actual'])}, "
                f"misses by {_target_text(v['metric'], v['miss'])} "
                f"{'dollars' if v['unit'] == 'cents' else v['unit']}"
                for v in source["target_violations"]
            )
            projected["Cannot verify"] = "; ".join(
                lenses.TARGETS[v["metric"]][1] + " (" + v["status"] + ")"
                for v in source["unverifiable_targets"]
            )
    st_module.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=min(620, max(120, 36 * len(display) + 40)),
    )
    if lens == "cushion_speed":
        st_module.caption(
            "The conditional median account duration includes only the "
            "reaching-account count shown, out of all accounts bought. "
            "Failures, ongoing accounts and unavailable evidence remain "
            "visible; this is not a future reaching probability."
        )
    st_module.download_button(
        "Export current comparison and definitions",
        lenses.export_view(view, lens=lens, targets=targets, categories=categories, search=search),
        file_name=f"{study.result_id[:16]}_{lens}_comparison.json",
        mime="application/json",
        key=prefix + "export",
    )
    st_module.download_button(
        "Export complete configuration evidence",
        json.dumps(rows, allow_nan=False, indent=2),
        file_name=f"{study.result_id[:16]}_all_configurations.json",
        mime="application/json",
        key=prefix + "export_all",
    )
    known = [r["configuration_id"] for r in rows]
    names = {
        v.name: " · ".join(
            (
                v.variant_id,
                policy_label("schedule", json.loads(v.intent_json)["schedule"]),
                policy_label("exit", json.loads(v.intent_json)["exit"]),
                policy_label("geometry", json.loads(v.intent_json)["geometry"]),
            )
        )
        for v in study.plan.configurations
    }
    selection = st_module.multiselect(
        "Compare two to four alternatives",
        known,
        default=[k for k in state.get("selected", []) if k in known],
        max_selections=4,
        key=prefix + "selected",
        format_func=lambda key: names.get(key, key),
    )
    state["selected"] = selection
    if len(selection) >= 2:
        compared = [next(r for r in rows if r["configuration_id"] == k) for k in selection]
        all_fields = list(
            dict.fromkeys(
                (
                    "configuration_id",
                    "net_received_cash_cents",
                    "acquisition_spend_cents",
                    "overall_cash_rank",
                    *[k for values in lenses.VISIBLE.values() for k in values],
                    *lenses.AXES,
                )
            )
        )
        frame = pd.DataFrame(_display(compared, all_fields)).set_index("Configuration").transpose()
        frame.index.name = "Measure / actual setting"
        st_module.dataframe(frame, use_container_width=True)
        st_module.caption(
            "Alternative histories are compared side by side; "
            "their accounts and cash are never combined."
        )
    if view["rows"]:
        options = [r["configuration_id"] for r in view["rows"]]
        chosen = st_module.selectbox(
            "Open configuration",
            options,
            format_func=lambda key: names.get(key, key),
            key=prefix + "open_choice",
        )
        if st_module.button("Open selected configuration", key=prefix + "open"):
            from ifvg_lab_nav import open_funded_detail

            open_funded_detail(target, chosen, lenses.FIRM, "Summary", st_module)


def render_gamma(st_module, ctx) -> None:
    report = _load_or_notice(st_module, ctx.store_root, ctx.result_id)
    if report is None:
        return
    data = report["gamma"]
    state = ctx.context.setdefault("mffu_gamma", {})
    prefix = f"mffu_gamma_{ctx.result_id[:16]}_{ctx.configuration}_"
    left, right = st_module.columns(2)
    with left:
        population = st_module.selectbox(
            "Population",
            ["funded", "strategy"],
            index=["funded", "strategy"].index(state.get("population", "funded")),
            format_func=lambda p: (
                "Actual funded trades"
                if p == "funded"
                else "Ordinary strategy trades · separate replay"
            ),
            key=prefix + "population",
        )
    with right:
        basis = st_module.selectbox(
            "Analysis basis",
            ["entry", "first_1R_checkpoint"],
            index=["entry", "first_1R_checkpoint"].index(state.get("basis", "entry")),
            format_func=lambda b: (
                "At entry" if b == "entry" else "At first profit-taking checkpoint"
            ),
            key=prefix + "basis",
        )
    state.update(population=population, basis=basis)
    st_module.caption(
        "Full-history descriptive view · complete saved study. At-entry cohorts "
        "retain eventual outcomes; the first-checkpoint basis includes only "
        "positions that actually reached it."
    )
    with st_module.expander("Gamma and clock filters", expanded=False):
        categories = st_module.multiselect(
            "Gamma context",
            list(gamma.CATEGORIES),
            default=state.get("categories", []),
            format_func=lambda g: gamma.CATEGORY_LABELS[g],
            key=prefix + "categories",
        )
        clocks = st_module.multiselect(
            "Chicago clock window",
            list(gamma.CLOCKS),
            default=state.get("clocks", []),
            key=prefix + "clocks",
        )
        state.update(categories=categories, clocks=clocks)
        if st_module.button("Reset gamma filters", key=prefix + "reset"):
            state.update(categories=[], clocks=[])
            for suffix in ("categories", "clocks"):
                st_module.session_state.pop(prefix + suffix, None)
            st_module.rerun()
    rows = gamma.selected_rows(
        data, ctx.configuration, population, basis, categories=categories, clocks=clocks
    )
    cov = gamma.coverage(rows)
    full_population = gamma.selected_rows(data, ctx.configuration, population, "entry")
    missing_checkpoints = [
        r for r in full_population if r.get("first_checkpoint_status") == "reached_time_unavailable"
    ]
    if basis != "entry" and missing_checkpoints:
        st_module.caption(
            f"{len(missing_checkpoints)} positions in this configuration's "
            "complete population reached a partial checkpoint but its exact "
            "time was not saved; they are excluded from this checkpoint-context "
            "view. They remain in at-entry outcomes and are not counted as "
            "positions that never reached the checkpoint."
        )
    reports = cov["report_dates"]
    range_text = f"{reports[0]}–{reports[-1]}" if reports else "no selected eligible reports"
    st_module.caption(
        f"{cov['actual_trades']} actual trades · {cov['executed']} executed entry contexts · "
        f"{cov['posthoc']} entry annotations added after execution · "
        f"unknown reasons {cov['unknown_reasons'] or 'none'} · "
        f"{lenses.VERSION} · reports {range_text}."
    )
    with st_module.expander("Methodology and source timing", expanded=False):
        st_module.write(
            "Full-history descriptive trading outcomes; these associations are not future "
            "win probabilities or a filtered funded simulation. Gamma uses the bound EOD "
            "source with nominal 10:00 PM Chicago eligibility; historical publication "
            "latency was not measured. Positive age counts distinct eligible vendor "
            "reports, never repeated reads or requested-date carry-forwards. Missing gamma "
            "stays unknown. First-checkpoint populations include only positions that "
            "actually reached the checkpoint, including whole exits at first target."
        )
        st_module.write(
            "Group-date counts overlap across states and are not independent additive "
            "samples. Net initial-risk units use after-cost trade P&L divided by original "
            "full-position structural-stop risk. No received payout is attributed to "
            "a gamma subgroup."
        )
        st_module.write(data["intraday_note"])
        st_module.write(
            "EOD maps use historical NQ1! while execution uses its source-selected "
            "futures contract. No unapproved price adjustment is applied."
        )
    group_rows = gamma.grouped(rows)
    display = [
        {
            "Gamma context": gamma.CATEGORY_LABELS[r["gamma_category"]],
            "Actual trades": r["actual_trades"],
            "Trading dates": r["distinct_trading_dates"],
            "Wins": r["wins"],
            "Losses": r["losses"],
            "Zero": r["zero_outcomes"],
            "Net trading P&L": f"${r['net_pnl_cents'] / 100:,.2f}",
            "Mean after-cost initial-risk units": r["mean_after_cost_initial_risk_units"],
            "Risk unavailable": r["unavailable_risk_count"],
        }
        for r in group_rows
    ]
    st_module.markdown("#### Context and clock overview")
    st_module.dataframe(display, use_container_width=True, hide_index=True)
    clock_groups = gamma.grouped(rows, by_clock=True)
    if clock_groups:
        import plotly.graph_objects as go

        counts = {
            (r["gamma_category"], r["chicago_clock"]): r["actual_trades"] for r in clock_groups
        }
        fig = go.Figure(
            go.Heatmap(
                z=[
                    [counts.get((group, clock), 0) for clock in gamma.CLOCKS]
                    for group in gamma.CATEGORIES
                ],
                x=list(gamma.CLOCKS),
                y=[gamma.CATEGORY_LABELS[g] for g in gamma.CATEGORIES],
                colorscale="Blues",
                colorbar_title="Actual trades",
            )
        )
        fig.update_layout(
            height=330,
            margin=dict(l=15, r=15, t=15, b=20),
            xaxis_title="Chicago display windows · start inclusive, end exclusive",
        )
        st_module.plotly_chart(fig, use_container_width=True, key=prefix + "heatmap")
    st_module.markdown("#### Whole-trade outcomes and remaining position")
    partitions = gamma.outcome_partition(rows)
    st_module.dataframe(
        [
            {
                "Whole-trade outcome": r["outcome"],
                "Trades": r["trades"],
                "Whole-trade net P&L": f"${r['whole_trade_net_cents'] / 100:,.2f}",
                "Actual remaining legs": r["remaining_leg_count"],
                "Remaining-position net P&L": f"${r['remaining_leg_net_cents'] / 100:,.2f}",
            }
            for r in partitions
        ],
        use_container_width=True,
        hide_index=True,
    )
    st_module.caption(
        "Whole at first target: "
        f"{sum(r['first_checkpoint_branch'] == 'whole at first target' for r in rows)}. "
        "Remaining-position dollars include allocated posted entry fees and the actual "
        "final exit fee; no fee is charged twice."
    )
    if population == "strategy":
        st_module.caption(
            "Remaining-position contributions use the separate ordinary "
            "completed-candle fills and their posted per-fill costs."
        )
    selected_keys = {r["trade_key"] for r in rows}
    geometries = [r for r in data["geometry"] if r["trade_key"] in selected_keys]
    variant = next(v for v in ctx.study.plan.configurations if v.name == ctx.configuration)
    section = json.loads(variant.effective_section_json)
    policy = section.get("opposing_distance_policy")
    st_module.markdown("#### Expected move and frozen pattern allowance")
    if policy == "fixed_v1":
        st_module.write(
            "Fixed allowance: 80 ticks / 20 index points. This configuration has "
            "no dynamic expected-move allowance series."
        )
    elif geometries:
        limits = [
            r["frozen_limit_ticks"] for r in geometries if r.get("frozen_limit_ticks") is not None
        ]
        fallbacks = sum(bool(r.get("fallback_reason")) for r in geometries)
        st_module.write(
            f"{len(geometries)} unique executed-entry parent locks · "
            f"{fallbacks} fallback events ({fallbacks / len(geometries):.1%}) · "
            f"frozen limits {min(limits)}–{max(limits)} ticks; "
            f"median {pd.Series(limits).median():g} ticks."
        )
        st_module.caption(
            "M = (1D Max − 1D Min) / 2, in full index points. The configured fraction "
            "freezes an integer-tick maximum at parent lock. 80 ticks / 20 points is "
            "a missing/invalid-source fallback. This allowance limits opposing-pattern "
            "separation; it does not widen the structural stop or set a target. Wider "
            "fixed-distance partial-exit controls were absent, so adaptation itself "
            "is not independently isolated."
        )
        import plotly.graph_objects as go

        figure = go.Figure(
            go.Scatter(
                x=[
                    gamma.stamp(r["lock_utc"]).tz_convert("America/Chicago").tz_localize(None)
                    for r in geometries
                ],
                y=[r["frozen_limit_points"] for r in geometries],
                mode="markers",
                name="Frozen at trade's lock",
                customdata=[
                    gamma.stamp(r["lock_utc"])
                    .tz_convert("America/Chicago")
                    .strftime("%B %d, %Y %I:%M:%S %p %Z")
                    for r in geometries
                ],
                hovertemplate="%{customdata}<br>%{y} index points<extra></extra>",
            )
        )
        figure.add_hline(y=20, line_dash="dash", annotation_text="Fixed 20-point reference")
        figure.update_layout(
            height=270,
            yaxis_title="Maximum separation · index points",
            xaxis_title="Supporting-parent lock · Chicago time",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st_module.plotly_chart(figure, use_container_width=True, key=prefix + "limits")
        distribution = go.Figure(
            go.Histogram(
                x=[r["frozen_limit_points"] for r in geometries], name="Actual parent locks"
            )
        )
        distribution.add_vline(x=20, line_dash="dash", annotation_text="Fixed 20 points")
        distribution.update_layout(
            height=230,
            xaxis_title="Frozen maximum · index points",
            yaxis_title="Trade-associated locks",
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st_module.plotly_chart(distribution, use_container_width=True, key=prefix + "distribution")
        with st_module.expander("Executed geometry observations"):
            st_module.dataframe(
                [
                    {
                        "Parent lock (Chicago)": gamma.stamp(r["lock_utc"])
                        .tz_convert("America/Chicago")
                        .strftime("%B %d, %Y %I:%M:%S %p %Z"),
                        "1D Min · points": r["range_min_points"],
                        "1D Max · points": r["range_max_points"],
                        "M · half-range points": r["implied_half_range_points"],
                        "Configured fraction": f"{r['configured_fraction']:.1%}"
                        if r["configured_fraction"] is not None
                        else "Fixed",
                        "Frozen limit · ticks": r["frozen_limit_ticks"],
                        "Frozen limit · points": r["frozen_limit_points"],
                        "Source report": data["level_maps"]
                        .get(r["level_set_id"], {})
                        .get("report_date"),
                        "Fallback reason": r["fallback_reason"] or "None",
                        "Actual qualifying separation · points": r["qualifying_separation_points"],
                        "Separation evidence": r["separation_status"],
                    }
                    for r in geometries
                ],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st_module.info(
            "No exact trade-linked parent-lock geometry is available for this selection; "
            "no limit has been inferred from a configured maximum."
        )
    st_module.markdown("#### Tested policy effects")
    pairs = (ctx.study.result.get("mffu_analysis") or {}).get("matched_pairs") or []
    relevant = [p for p in pairs if ctx.configuration in {p.get("base_id"), p.get("challenger_id")}]
    st_module.caption(
        f"{len(relevant)} comparisons involve this configuration; {len(pairs)} declared "
        "pairs retained in the complete result. Effects compare complete saved/reused "
        f"operations. {ctx.target.get('catalog_binding', {}).get('qualification', '')}"
    )
    if relevant:
        by_config = {r["configuration_id"]: r for r in report["lenses"]}

        def comparison_row(pair):
            base = by_config.get(pair["base_id"], {})
            challenger = by_config.get(pair["challenger_id"], {})
            axis = pair.get("changed_axis")
            spend_left, spend_right = (
                base.get("acquisition_spend_cents"),
                challenger.get("acquisition_spend_cents"),
            )
            wait_left, wait_right = (
                base.get("first_receipt_calendar_days"),
                challenger.get("first_receipt_calendar_days"),
            )
            return {
                "Reference": pair["base_id"],
                "Challenger": pair["challenger_id"],
                "Changed setting": f"{policy_label(axis, base.get(axis, 'Unavailable'))} → "
                f"{policy_label(axis, challenger.get(axis, 'Unavailable'))}",
                "Status": pair.get("status"),
                "Net cash difference": f"${pair['delta_cents'] / 100:,.2f}"
                if pair.get("delta_cents") is not None
                else "Unavailable",
                "Reference acquisition spend": f"${spend_left / 100:,.2f}"
                if spend_left is not None
                else "Unavailable",
                "Challenger acquisition spend": f"${spend_right / 100:,.2f}"
                if spend_right is not None
                else "Unavailable",
                "First receipt · reference calendar days": wait_left,
                "First receipt · challenger calendar days": wait_right,
                "First receipt · calendar-day difference": wait_right - wait_left
                if wait_left is not None and wait_right is not None
                else None,
            }

        st_module.dataframe(
            [comparison_row(p) for p in relevant],
            use_container_width=True,
            hide_index=True,
        )
    with st_module.expander("Inspect a trade's selected context"):
        if rows:
            selected = st_module.selectbox(
                "Trade",
                list(range(len(rows))),
                format_func=lambda n: (
                    f"{_chicago(rows[n]['entry_utc'])} · {rows[n]['trade_ref'][:8]}"
                ),
                key=prefix + "trade",
            )
            row = rows[selected]
            source = row["selected_snapshot"]
            levels = data["level_maps"].get(source.get("level_set_id"), {})
            gex = source.get("gamma", {})
            percentile = gex.get("gex_percentile_1y")
            eligible = (
                _chicago(gex["nominal_eligible_from_utc"])
                if gex.get("nominal_eligible_from_utc")
                else "Unavailable"
            )
            st_module.write(
                f"{row['context_role'].replace('_', ' ')} · gamma report "
                f"{gex.get('report_date')} · total gamma {gex.get('value')} in vendor GEX "
                "units · provided percentile "
                f"{percentile if percentile is not None else 'Unavailable'} "
                f"· eligible {eligible}"
            )
            st_module.dataframe(
                [
                    {
                        "Level": name,
                        "Price · index points": item.get("price"),
                        "Signed exposure · vendor GEX units": item.get("gex"),
                    }
                    for name, item in (levels.get("items") or {}).items()
                ],
                hide_index=True,
                use_container_width=True,
            )
            if population == "funded" and st_module.button(
                "Open this trade in Trade review", key=prefix + "review"
            ):
                from ifvg_lab_nav import open_trade_review

                open_trade_review(
                    ctx.target,
                    configuration=ctx.configuration,
                    firm_key=ctx.firm_key,
                    account_number=row.get("account_number"),
                    trade_seq=row.get("seq"),
                    back_tab="Market conditions",
                    st_module=st_module,
                )
    st_module.download_button(
        "Export gamma view and source definitions",
        gamma.export_rows(rows, report=data, basis=basis, filters=state),
        file_name=f"{ctx.configuration}_{population}_{basis}_gamma.json",
        mime="application/json",
        key=prefix + "export",
    )


def add_gamma_overlay(figure, segments) -> None:
    """Draw horizontal segments on the existing chart's Chicago wall clock."""
    import plotly.graph_objects as go

    for segment in segments:
        names = ", ".join(level["name"] for level in segment["levels"])
        clocks = [
            gamma.stamp(segment[k]).tz_convert("America/Chicago").tz_localize(None)
            for k in ("start_utc", "end_utc")
        ]
        figure.add_trace(
            go.Scatter(
                x=clocks,
                y=[segment["price_points"]] * 2,
                mode="lines",
                name=names,
                showlegend=False,
                line=dict(color=palette()["blue_line"], width=1, dash="dot"),
                hovertemplate=f"{names} · {segment['price_points']:,.2f} index points<br>"
                f"EOD report {segment['report_date']}<extra></extra>",
            )
        )


def render_trade_gamma(
    st_module, target, study, configuration, trade_row, view, *, moment, window, scope
):
    report = _load_or_notice(st_module, target["store_root"], target["result_id"])
    if report is None:
        return []
    data = report["gamma"]
    ref = trade_row.get("trade_ref") or trade_row.get("strategy_trade_id")
    matches = [
        r
        for r in data["trades"]
        if r["configuration_id"] == configuration
        and r["population"] == "funded"
        and r["trade_ref"] == ref
        and r.get("account_number") == trade_row.get("account_number")
    ]
    if len(matches) != 1:
        st_module.info("No unique exact trade-linked gamma record is available.")
        return []
    trade = matches[0]
    columns = st_module.columns([1, 3])
    with columns[0]:
        overlay = st_module.checkbox(
            "Gamma-level overlay", value=False, key="gamma_overlay_" + scope
        )
    with columns[1]:
        names = st_module.multiselect(
            "Recorded EOD levels",
            list(_LEVEL_NAMES),
            default=["HVL", "1D Max", "1D Min"],
            key="gamma_levels_" + scope,
            disabled=not overlay,
        )
    cards = gamma.checkpoint_cards(trade, cursor=moment)
    with st_module.expander("Gamma and frozen-geometry checkpoint cards", expanded=overlay):
        st_module.caption(
            "Eligible historical EOD maps · vendor GEX units · nominal 10:00 PM "
            "Chicago availability; publication latency unmeasured. Frozen lock "
            "inputs remain separate from a newer eligible map. NQ1! map prices "
            "are not adjusted to the source-selected execution contract."
        )
        if not cards:
            st_module.write("No trade checkpoint has occurred at this cursor.")
        for card in cards:
            snapshot = card.get("snapshot") or {}
            gex = snapshot.get("gamma") or {}
            level_map = data["level_maps"].get(snapshot.get("level_set_id"), {})
            receipt = card.get("recorded_policy") or {}
            local = gamma.stamp(card["at_utc"]).tz_convert("America/Chicago")
            st_module.markdown(f"**{card['checkpoint']} · {local:%B %d, %Y %I:%M:%S %p %Z}**")
            st_module.write(
                f"Gamma: {gamma.CATEGORY_LABELS[gamma.category(gex)]}; "
                f"total {_number(gex.get('value'))} in vendor GEX units; "
                f"positive age {gex.get('positive_run_age')}; "
                f"report {gex.get('report_date')}; "
                f"eligible {gex.get('nominal_eligible_from_utc')}."
            )
            if card["checkpoint"] == "Supporting-pattern lock":
                geometry = next(
                    (r for r in data["geometry"] if r["trade_key"] == trade["trade_key"]), {}
                )
                st_module.write(
                    "Implied half-range M: "
                    f"{_number(geometry.get('implied_half_range_points'), 3)} "
                    f"index points; fraction {geometry.get('configured_fraction')}; "
                    f"frozen maximum {geometry.get('frozen_limit_ticks')} ticks / "
                    f"{geometry.get('frozen_limit_points')} points. "
                    f"Fallback: {geometry.get('fallback_reason') or 'none'}."
                )
            if card["checkpoint"] == "Entry":
                price = trade["entry_ticks"] * 0.25
                items = level_map.get("items") or {}
                hvl = items.get("HVL", {}).get("price")
                side = (
                    "above"
                    if hvl is not None and price > hvl
                    else "below"
                    if hvl is not None and price < hvl
                    else "at"
                    if hvl is not None
                    else "unavailable"
                )
                above = [
                    (v["price"], k)
                    for k, v in items.items()
                    if v.get("price") is not None and v["price"] > price
                ]
                below = [
                    (v["price"], k)
                    for k, v in items.items()
                    if v.get("price") is not None and v["price"] < price
                ]
                st_module.write(
                    f"Entry {price:,.2f} points · {side} HVL · "
                    f"nearest above {_nearest_level(above, above=True)} · "
                    f"nearest below {_nearest_level(below, above=False)} · "
                    f"actual quantity {card.get('quantity')} MNQ micros."
                )
            if card.get("branch"):
                st_module.write(
                    "Recorded exit branch: "
                    + card["branch"]
                    + "; "
                    + trade["first_checkpoint_context_role"].replace("_", " ")
                )
            st_module.write("Context role: " + card["context_role"].replace("_", " "))
            if card.get("context_usage"):
                st_module.caption("Context use: " + card["context_usage"].replace("_", " "))
            if receipt:
                st_module.write(
                    f"Recorded decision: {receipt.get('action')} · "
                    f"policy {receipt.get('policy')} · reasons "
                    f"{', '.join(receipt.get('reasons') or []) or 'none'}."
                )
        st_module.caption(
            "Identical-price names share a line. Signed EOD exposure is supplied "
            "in vendor units; null range exposure is unavailable, never zero. "
            "No trading-signal arrows or inferred touch history are added."
        )
    return (
        gamma.level_segments(data, start=window[0], end=window[1], cursor=moment, names=names)
        if overlay and names
        else []
    )


_LEVEL_NAMES = (
    "HVL",
    "1D Max",
    "1D Min",
    "Call Resistance",
    "Put Support",
    "GEX 1",
    "GEX 2",
    "GEX 3",
    "GEX 4",
    "GEX 5",
    "GEX 6",
    "GEX 7",
    "GEX 8",
    "GEX 9",
    "GEX 10",
    "Call Resistance 0DTE",
    "Put Support 0DTE",
    "HVL 0DTE",
    "Gamma Wall 0DTE",
)
