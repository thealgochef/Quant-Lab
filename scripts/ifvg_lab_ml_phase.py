"""Normal Lab screen for one saved development-only return-regression study."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.presentation.lab import mffu_lenses as lenses
from alpha_lab.propsim.funded.ml_phase.catalog import load_report, point_in_time
from alpha_lab.propsim.funded.ml_phase.reporting import operation_description
from alpha_lab.propsim.funded.ml_phase.runtime import sha_file


def _select(st, context, label, values, field, *, format_func=str):
    current = context.get(field)
    index = values.index(current) if current in values else 0
    value = st.selectbox(label, values, index=index, key=f"mlphase_{field}",
                         format_func=format_func)
    context[field] = value
    return value


def _table(st, rows):
    display = []
    for row in rows:
        shown = {}
        for key, value in row.items():
            if key.endswith("_cents"):
                key = key.removesuffix("_cents") + "_usd"
                value = value / 100 if value is not None else None
            elif key.endswith("_utc"):
                key = key.removesuffix("_utc") + "_chicago"
                if value:
                    value = pd.Timestamp(value).tz_convert("America/Chicago").strftime(
                        "%b %d, %Y %I:%M:%S %p")
            shown[key] = value
        display.append(shown)
    st.dataframe(pd.DataFrame(display), hide_index=True, width="stretch")
    return display


def _exact_json(value):
    """Keep event nanoseconds exact in the browser's JavaScript JSON renderer."""
    if isinstance(value, dict):
        return {key: _exact_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_exact_json(item) for item in value]
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        return str(value)
    return value


def render_ml_phase(st, roots):
    from ifvg_lab_nav import funded_context

    pointer = st.session_state.get("ifvg_ml_phase_pointer")
    if not pointer:
        st.warning("Select the saved ML phase in My studies.")
        return
    try:
        report = load_report(pointer)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"This saved report could not be verified: {error}")
        return
    context = funded_context(report["report_id"], st)
    st.title(pointer["name"])
    st.caption("Development only · October 9, 2025–June 10, 2026 · 171 evaluated dates · "
               "MyFundedFutures · ten micros using the NQ tape proxy")
    cells = report["benchmark"]["cells"]
    names = {c["cell_id"]: operation_description(c) for c in cells}
    for ref in ("MCB062", "MCB025"):
        names[f"NO_ML_{ref}"] = f"{ref} · {operation_description(None)}"
    operation = _select(st, context, "Operation", list(names), "ml_operation",
                        format_func=names.get)
    cell = next((c for c in cells if c["cell_id"] == operation), None)
    ref = cell["reference"] if cell else operation.removeprefix("NO_ML_")
    st.caption("Geometry: " + ("10% of implied allowance frozen at parent lock"
               if ref == "MCB062" else "fixed 20-point opposing-parent distance"))
    page = _select(st, context, "View", ["Summary", "Predictive evidence", "Funded policies",
                                       "Decision review", "Review packages"], "ml_view")
    if page == "Summary":
        st.markdown(report["summary_markdown"])
        _table(st, report["coverage"])
        _table(st, report["economics"]["operation_coverage"])
    elif page == "Predictive evidence":
        st.write("Entry scores estimate net R. Continuation scores estimate HOLD minus "
                 "next-print CLOSE, in original ten-micro risk units. "
                 "The action threshold is zero.")
        cell_by_id = {c["cell_id"]: c for c in cells}
        valid = sum(f["status"] == "valid" for f in report["benchmark"]["fits"])
        st.caption(f"All {len(cells)} cells are retained; {valid} of "
                   f"{len(report['benchmark']['fits'])} saved folds are valid.")
        _table(st, [{**{k: cell_by_id[r["cell_id"]][k] for k in (
                        "cell_id", "reference", "job", "feature_set", "model")},
                    **{k: r.get(k) for k in ("observations", "status", "skill", "rmse", "mae",
                                            "bias", "association")}}
                    for r in report["benchmark"]["pooled_metrics"]])
        folds = []
        for record in report["benchmark"]["fits"]:
            if record["cell_id"] != operation:
                continue
            row = {k: record[k] for k in ("fold_id", "status", "train_rows", "train_dates",
                                         "test_rows", "unavailable_reason")}
            for key in ("training_cutoff_ns", "activation_ns"):
                row[key.removesuffix("_ns") + "_chicago"] = pd.Timestamp(
                    record[key], unit="ns", tz="UTC").tz_convert("America/Chicago").strftime(
                        "%b %d, %Y %I:%M %p")
            folds.append(row)
        _table(st, folds)
        st.write("Paired predictive-loss differences; negative favors the left cell. "
                 "Intervals use 2,000 paired five-date blocks and describe "
                 "development uncertainty.")
        _table(st, [r for r in report["paired_diagnostics"]
                    if operation in (r["left_cell"], r["right_cell"])])
    elif page == "Funded policies":
        lens = _select(st, context, "Comparison lens", list(lenses.LENSES), "ml_lens",
                       format_func=lambda k: lenses.LENSES[k][0])
        targets = {}
        with st.expander("Optional reporting targets"):
            metric = st.selectbox("Target", ["None", *lenses.TARGETS],
                format_func=lambda k: k if k == "None" else lenses.TARGETS[k][1])
            if metric != "None":
                operator, _, unit = lenses.TARGETS[metric]
                if operator == "range":
                    target = [st.number_input("Minimum", value=0.0),
                              st.number_input("Maximum", value=10.0)]
                else:
                    target = st.number_input(f"Value ({'USD' if unit == 'cents' else unit})",
                                             value=0.0)
                    if unit == "cents":
                        target *= 100
                targets[metric] = target
        errors = lenses.validate_targets(targets)
        if errors:
            st.warning("; ".join(errors))
            return
        rows = lenses.sort_rows(report["economics"]["lenses"], lens)
        if targets:
            rows = [r for r in rows if not any(lenses.target_violations(r, targets))]
        visible = ["configuration_id", "reference", "net_received_cash_cents",
                   "net_cash_delta_cents", *lenses.VISIBLE[lens]]
        display = []
        for row in rows:
            display.append({**{(k.replace("_cents", "_usd") if k.endswith("_cents") else k):
                (row.get(k) / 100 if row.get(k) is not None and k.endswith("_cents")
                 else row.get(k)) for k in dict.fromkeys(visible)},
                 "policy": names.get(row["configuration_id"])})
        display = _table(st, display)
        st.download_button("Export this view", pd.DataFrame(display).to_csv(index=False),
                           file_name=f"{lens}.csv", mime="text/csv")
        economic = report["economics"]["economic_result"]
        journeys = [r for r in economic["tables"]["account_journeys"]
                    if r["configuration"] == operation]
        account = _select(st, context, "Account", [r["account_number"] for r in journeys],
                          f"ml_account_{operation}")
        for table in ("account_journeys", "cash_ledger", "payout_events", "trades"):
            st.subheader(table.replace("_", " ").title())
            _table(st, [r for r in economic["tables"][table] if r["configuration"] == operation
                        and r.get("account_number") == account])
    elif page == "Decision review":
        st.caption("ENTRY: expected net R. CONTINUATION: expected HOLD minus next-print "
                   "CLOSE in original ten-micro R. Only a score below zero changes action; "
                   "unavailable predictions use the baseline.")
        state = report["operations"][ref]["streams"][operation]
        rows = sorted([*state["entry_rows"].values(), *state["checkpoint_rows"].values()],
                      key=lambda r: (r["decision_ns"], r["row_id"]))
        if not rows:
            st.info("No actual decisions in this operation.")
            return
        chosen = _select(st, context, "Decision", [r["row_id"] for r in rows],
            f"ml_decision_{operation}", format_func=lambda key: next(
                f"{r['trading_day']} · {r['job']} · {key[:8]}"
                for r in rows if r["row_id"] == key))
        row = next(r for r in rows if r["row_id"] == chosen)
        stage = st.radio("Point-in-time cursor", ["Before decision", "At decision",
                                                "After outcome"], horizontal=True)
        label = next((r for r in report["datasets"][ref][row["job"]]
                      if (r["episode_id"], r["decision_ns"]) ==
                      (row["episode_id"], row["decision_ns"])), None)
        trade_ids = {key for key, value in state["driver"]["ml_entries"].items()
                     if value["candidate_id"] == row["candidate"]}
        actual = [t for t in state["ledger"]["trades"] if t["trade_ref"] in trade_ids]
        cursor = row["decision_ns"] - 1 if stage == "Before decision" else row["decision_ns"]
        if stage == "After outcome":
            cursor = max([cursor, label["label_available_ns"] if label else cursor,
                          *(t["exit_ns"] for t in actual)])
        stamp = pd.Timestamp(cursor, unit="ns", tz="UTC").tz_convert("America/Chicago")
        st.caption(stamp.strftime("%B %d, %Y · %I:%M:%S %p Chicago") + f" · {cursor} ns")
        view = point_in_time(row, cursor, label=label, after_outcome=stage == "After outcome")
        prediction = view.get("prediction")
        if prediction is not None:
            score = prediction.get("score")
            st.metric("Expected net R" if row["job"] == "ENTRY" else "Expected HOLD − CLOSE",
                      "Unavailable" if score is None else f"{score:+.4f} R")
            if prediction["action"] == "change":
                action = ("Reject entry" if row["job"] == "ENTRY" else
                          "Close the remaining five micros at the next eligible print")
            else:
                action = "Keep the baseline action"
                if prediction.get("reason"):
                    action += " · " + prediction["reason"].replace("_", " ")
            st.write("Action: " + action)
            st.caption(f"Cell {operation} · Fold {prediction.get('fold_id') or 'not applicable'} "
                       f"· Fixed threshold: 0 R")
        st.json(_exact_json(view), expanded=1)
        if stage == "After outcome":
            st.caption("The fixed-shadow label is the baseline counterfactual. "
                       "Actual policy fills and profit are separate below.")
            st.json(_exact_json({"actual_policy_closed_trades": actual}))
        if view.get("prediction", {}).get("model_id"):
            fit = next(f for f in report["fit_states"]
                       if f["fit_id"] == view["prediction"]["model_id"])
            with st.expander("Earlier training and imputation state"):
                st.json(_exact_json(fit))
        if stage == "After outcome" and label is None:
            st.info("No matched fixed-shadow label exists for this actual policy decision.")
    else:
        st.write("Both review archives and the delivery receipt belong to this saved study.")
        for artifact in pointer.get("deliveries", []):
            path = Path(artifact["path"])
            if sha_file(path) != artifact["sha256"]:
                st.error(f"Delivery hash mismatch: {path.name}")
                continue
            st.download_button(artifact["label"], path.read_bytes(), file_name=path.name,
                               mime="application/zip" if path.suffix == ".zip" else
                               "application/json")
        with st.expander("Source and result identities"):
            st.json(report["identity"])
            st.json(report["economics"]["financial_validation"])
        st.download_button("Report identity", json.dumps(report["identity"], indent=2),
                           file_name="identity.json")
