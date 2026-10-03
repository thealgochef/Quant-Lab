# Task A1 — resolved questions and recorded choices

Updated October 3, 2026. No policy question currently blocks implementation;
the additional full-suite verification described below requires an owner exception.

1. **Availability precedence — resolved by decision 12.** Outside 06:00–17:00
   Chicago, neither gate evaluates, both flags are null, and the status is
   `not_applicable_outside_hours`, including Asia entries with unknown policy
   block. Inside hours, missing levels or unknown regime use the named policy
   when the regime gate is enabled. The task specification, decisions, Core
   tests and export apply the ruling.
2. **Parity scope — resolved by decision 13.** Seven v2 execution tables,
   `label_source_1m` and the profile hash require byte parity. All twelve v3
   tables require equal retained schemas/content after proven source identity
   exclusions. `parity_report.md` lists every exclusion and its reason. Source
   provenance was preserved in the actual tables, never masked for replay.
3. **Instrument and rolls — authorized fallback used.** Source instrument
   selection happens outside the lookup's current `DayArtifacts`/level seam;
   emitted trade values carry no selected instrument ID. Both
   `selected_instrument_id` and `roll_flag` remain null. Comparison rows carry
   `roll_flag_unavailable = true`; no row is excluded. No calendar-inferred roll
   or back-adjustment was added.
4. **Funded cash — authorized fallback used.** Fresh funded results require
   the existing approval-gated verified-input pipeline. A1 does not create a
   charter, plan or approval. Executed-trade reports/export are produced; all
   comparison net-cash columns contain `not_produced_in_a1`. A separate consumer
   groups an already saved verified funded cash ledger in exact cents, using
   cash-event timestamps and unknown entry session. This follows the existing
   monthly cash-ledger pattern without allocating cash to entry trades.
5. **Smoke calendar discrepancy — closest existing pattern used.** The
   existing prepared evaluation calendar includes January 19 and February 16
   as partial trading sessions. Its first 30 dates from January 13 end on
   February 23, 2026; including February 24 would require 31 dates. A1 preserves
   the requested two configurations and 30-date limit using the literal first
   30 calendar dates. It does not silently remove a holiday or add a 31st date.
   The smoke identities and timing files record the exact calendar. This is a
   documented difference from the endpoint in the task prose.
6. **Scratch preparation — existing bounded builders used.** The full pair
   helper fixes the final endpoint and repeats performance benchmarks. Its
   normal v2/v3 capture builders already accept explicit input/output paths,
   so the external harness calls those builders and report consumers directly.
   Default stores, the CLI, guards and performance limits are unchanged. The
   original allowlist receipt validates copied prepared caches while only the
   requested prefix is opened and replayed.
7. **Entry timestamp projection — exact table contract used.** Candidates and
   eligible decisions have their final-entry availability in `envelope_ts_utc`;
   their union-wide `entry_ts_utc` column is structurally null. Executed trades
   use explicit `entry_ts_utc`, because their envelope is the later resolution
   event. Report lookup follows these existing fields. The first context smoke
   reporting failure exposed this distinction; its failed attempt is preserved
   externally and the corrected configuration is retried once.
8. **Full-suite verification exception — pending.** The single §6 full run
   completed with 3,973 passed / 51 failed / 7 skipped. Five failures are the
   documented pre-existing failures. The other failures form three A1 integration
   clusters: rule-description classification, effective-config default fields,
   and canonical profile identity (including funded consumers). These have been
   corrected using existing neutral/optional-default patterns, with 11 new pure
   compatibility tests passing. No known failure or historical fixture was
   changed. Because §6 explicitly permits one full run, a second full run requires
   an explicit owner exception before it is executed. The original result is
   preserved in `test_run.log`.
9. **Normal review destination — existing job pattern used.** Ordinary
   preparation now writes the enabled export below the selected preparation job's
   `reports/<v2_artifact_id>/` folder, using existing verified IDs in the header.
   The optional `report_root` is a review-only scratch override. Seven focused
   tests verify export content/row count, default-off behavior, unchanged immutable
   files, forbidden destinations and persisted job-label forwarding. The smoke
   exports used the same writer and require no replay or numerical regeneration.

Original stopped questions remain preserved in the external working folder
`C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-a1-20261003`
as historical records. They are not current blocking requests. Task A2, fitting,
sealed dates, known failures, capacity limits and unrelated dependencies remain
outside this implementation.
