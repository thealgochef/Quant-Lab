# IFVG no-entry research

The research activity output in `search/entry_activity.py` takes an explicit
evaluation-date list and actual entry timestamps. It maps entries through the
saved 18:00 America/New_York trading-day boundary, preserving DST. It reports
all consecutive zero-entry runs, tied maxima, leading/trailing censoring,
adjacent entry/exit IDs and elapsed entry-to-entry and flat intervals.
Stored resolution-day metrics and existing charter gates remain separate.
No activity statistic triggers an entry or supplies an acceptance threshold.

The 2026-09-14 investigation uses only the original input bundle `f3443d0d…`,
107 evaluated dates, ten warmup dates and the June 10 21:00 UTC cutoff. It is
exploratory research, with no model fitting, holdout access or live activation.
Preserved work and receipts: `../Claude-Quant-Lab-Research-Artifacts/archived-reports/ifvg_no_entry_drought_20260914/` (relocated September 22).

The isolated research Core corrects a proven day-seed omission: closed partial
higher-timeframe bars whose logical close follows the final one-minute decision
were queued but not included in schema-3 seeds. Schema 4 persists ordered pending
bars and the last decision bar, rejects duplicate delivery and refuses historical
schema-3 seeds for continued execution. Historical outputs are not rewritten.
Partial bars retain their logical availability; they are never injected into an
earlier decision. The active-selected-HTF physical tracking repair is retained.
Production/live imports and pins are unchanged. Economic effects require the
complete original chronological replay and separate correctness/policy reporting.

The isolated policy runtime adds `htf_direction_selection_policy`, default
`mixed_direction_rank_v1`. The pending, exact-approval research value
`enabled_before_rank_v1` filters disabled directions before HTF admission/ranking.
It retains all physical zones and the same deterministic ranking, cap, conflict,
one-slot, invalidation, clocks and entry rules. The root registry exposes it only
when the imported research Core supports the field; installed Core pins stay.
`search/no_entry_experiment.py` freezes only B0/D0/B1/D1 and rejects any other
effective field differences. The saved-draft Evaluate One path propagates the
field through worker enumeration. No second mechanism or combination is included.

New strategy-search datasets persist `entry_activity_report.json` alongside
existing reports under the immutable manifest. It records the explicit calendar,
actual-entry daily vector, every gap and adjacent elapsed intervals. The report is
included in source identity; historical datasets without it remain readable.
Root output support is additive; the completed frozen task sources and approvals
remain in the report directory and are not rewritten when working code changes.
