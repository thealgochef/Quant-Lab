# Reporting-only implementation

The actual dirty working source was captured before edits under the sibling
`Claude-Quant-Lab-Research-Artifacts/ifsm-b8db-reporting-followup-v01/baseline/`.
Only the normal reporting/presentation path and its focused tests change.
No economic source or Core code is edited. The running main dashboard remains
`streamlit run scripts/dashboard.py` on port 8501.

`mffu_provenance.py` validates unique scoped trade/receipt membership, the effective
exit/context policies, actual target branch and independently bound source.
Funded embedded `decision_ns` and `target_observation_ns` must both equal the
actual event. Only then is the saved integer nanosecond-to-microsecond datetime
conversion checked. It is not a nearest-time join. Ordinary completed-candle
receipts identify the opening minute and become observable at its close.
Actual event, receipt context and reporting evaluation times are separate.
Conditional use of context is distinct from source observation and annotations.

`mffu_gamma.py` propagates independent entry/lock/target origins into rows,
coverage and cards. Invalid evidence supplies no target snapshot/action. Cursor
selection requires the entry to be visible and hides target fields before the
exact event. `mffu_reporting.py` includes linkage metadata in generated tables.

`mffu_matrix.py` supplies effective policy descriptions to the normal Settings
view, standard CSV and chart presenter. The saved intent and execution sizing
must agree. Raw inherited axes remain machine bindings, with their old planned
status explicitly labeled historical. Price-proxy evidence remains NQ; exposure
is MNQ. Dynamic distance is the saved half-range fraction, frozen at parent lock,
rounded half-up to 0.25-point ticks with a one-tick minimum; the 80-tick fallback
is separate. Charts retain all distinguishing axes and configuration IDs.

The next unused companion revision at task start was v6 (v5 was the only
current publication for b8db). Both result versions retain earlier companions.
The new standard export is a separate immutable version against b8db.
Exact final source, test, publication and preservation receipts accompany the
two review ZIPs; TASKS.md records completion rather than inferring it from code.
