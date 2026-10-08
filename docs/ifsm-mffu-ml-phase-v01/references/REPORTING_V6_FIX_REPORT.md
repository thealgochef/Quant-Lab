# B8 reporting follow-up — complete

B8-01: COMPLETE. B8-02: COMPLETE. Economic result remains
b8db8427cb2260619d7f0483358e5d0da78d718fc395f2374b6a62a1e378c186.
The reviewed predecessor is reporting v5; new immutable reporting is
ifsm_mffu_reporting_v6 for both corrected and original results. Standard export v2
is newly published. Prior report/export bytes remain preserved.

B8-01 failed because a microsecond receipt datetime was compared directly with a
nanosecond execution event, then checkpoint origins inherited unrelated parent
context. The repair validates unique scoped membership, effective policy, actual
action, embedded exact event IDs/times, the source context and its eligibility.
Only then does it validate the documented integer ns-to-microsecond conversion.
This is not a tolerant nearest-time match. Ordinary receipts use their separate
completed-candle convention. Observation, receipt context and evaluation clocks
remain distinct. Rows, coverage and cards use the origin of their own checkpoint.

All 1,908 saved funded conditional targets and 2,290 ordinary conditional targets
verify. The 5,119 funded and 6,179 ordinary annotations remain annotations;
5,550 funded and 6,588 ordinary targets were not reached. There are no unresolved,
ambiguous or missing-evidence targets in the published normal records. Negative
tests deliberately reject contradictions. The anchor is MCB003 account 1,
trade e2189e7d-1151-541d-a60e-0f1e992b587a, event
1750068615558271393 ns: the receipt loses exactly 393 ns. Browser captures show
the target hidden at 05:10 and visible at 05:15 Chicago; exact before/at/after
nanosecond boundaries are tested, not claimed from browser precision.

B8-02 uses shared descriptions derived from each saved effective configuration
and execution sizing. All 64 CSV/UI/chart labels were checked, including all six
dynamic geometry rows and every conditional exit. MNQ exposure and quantities
are separate from NQ signal/fill proxy evidence. Dynamic maximum distance is
(1D Max - 1D Min)/2 times 5%, 7.5% or 10%, frozen at supporting-parent lock,
divided by 0.25 with ROUND_HALF_UP and minimum one tick. The 80-tick/20-point
fallback is separately labeled. Fixed rows retain their fixed policy. Conditional
whole/half behavior explicitly covers neutral/unknown fallback. Chart labels
retain configuration IDs and all distinguishing axes.

Preservation: all 64 financial histories, summaries and table projections agree;
64 child histories match their sealed hashes; 568 protected files match. The
supplied reference adds 704 exact numeric field comparisons with zero differences.
All 43,130 context rows and 3,754 geometry rows preserve values/group membership.
MCB062 remains $48,102.16 net received cash, $49,727.16 received, $1,625.00 costs,
13 accounts, 302 trades and 25 payouts. No study was replayed, no new dates or ML
were run, no Core/rules/pins were changed, and no commit/push was performed.

Verification: initial actual seven-test reproduction had five failures/two passes.
Final focused run: 134 passed/one skip; after original v6 publication that exact
skipped check passed separately. Lint passed. Isolated portable reconstruction:
28 passed, no skips and no project imports from the working checkout. The full
repository/Core suites were not rerun. Two independent review findings (invalid
ordinary future snapshot and pre-entry parent-lock visibility) were repaired and
tested. The sealed archive's extraction/test results are in DELIVERY_RECEIPT.json
outside both ZIPs; in-archive portability receipts cover the preseal rehearsal.

Normal route: streamlit run scripts/dashboard.py -> IFVG Lab -> My studies ->
IFSM MFFU context batch -> Corrected lifecycle. Port 8501 was restarted on final
source without a special environment. Original execution remains selectable.
The normal Settings download was checked byte-for-byte against all 31 files of
reports/funded_comparison/funded_comparison_b8db8427cb226061_export_v2.
Publication receipts bind absolute paths and exact source/report hashes.

Limits remain: NQ is the execution-price proxy for MNQ, historical EOD release is
nominal 10 PM Chicago with unmeasured publication latency, and missing setup
zones were not fabricated. These views establish provenance, not strategy edge.

Evidence: CHECKPOINT_ORIGIN_AUDIT.csv; PROVENANCE_COVERAGE.json;
ECONOMIC_PRESERVATION.csv/JSON; configurations.csv; LABEL_CONSISTENCY.json;
net_cash_by_configuration.png; screenshots/INDEX.json; PUBLICATION.json.
Paired source: IFSM_b8db_Reporting_Followup_Source_Tests_v01.zip contains changed
source, actual fixtures, patch, review findings and final test/import bindings.
Its declared external dependency is the prior source archive, not a hidden
requirement for the original working checkout. Each manifest names paired members;
final ZIP SHA-256 values are only in the separate delivery receipt.
