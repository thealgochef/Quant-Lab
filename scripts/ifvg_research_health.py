"""Developer health summaries from evaluated reports; unknown never passes."""

from alpha_lab.agents.data_infra.ifvg.presentation.workspace_mode import technical_details_enabled


def health_reading(report):
    if not isinstance(report, dict) or not report:
        return "Unknown", "No evaluated report is available."
    if (
        report.get("passed") is False
        or report.get("valid") is False
        or report.get("status") in ("failed", "invalid", "corrupt")
    ):
        return "Failed", "The saved report records a failed check."
    if (
        report.get("passed") is True
        or report.get("valid") is True
    ):
        return "Passed", "The saved report explicitly records a successful check."
    return "Unknown", "The saved report does not contain a recognized evaluated verdict."


def render_health(st):
    if not technical_details_enabled():
        return
    from ifvg_lab_tab import _load_selected_pair, render_ifvg_data_audit_tab

    selected = _load_selected_pair(st, key="ifvg_developer_health_pair")
    if selected is None:
        st.info("Select prepared evidence to evaluate health.")
        return
    pair, _entry = selected
    rows = []
    for label, report in (
        ("Strategy correctness", pair.v2.reports.get("invariant_audit.json")),
        ("Count reconciliation", pair.v2.reports.get("count_reconciliation.json")),
        ("Data access", pair.v3.reports.get("data_access_audit.json")),
        ("Capacity", pair.v3.reports.get("capacity_report.json")),
        ("Validity", pair.v3.reports.get("validity_report.json")),
        ("Identity", pair.v3.reports.get("identity_report.json")),
    ):
        status, detail = health_reading(report)
        rows.append({"Check": label, "Status": status, "Evidence": detail})
    st.dataframe(rows, hide_index=True, width="stretch")
    if st.checkbox("Open raw reports"):
        render_ifvg_data_audit_tab(st)
