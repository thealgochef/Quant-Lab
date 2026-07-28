# Archived window artifacts

Per-work-window review deliverables archived (committed, filenames unchanged) in
the 2026-07-28 housekeeping window. None had ever been committed — they lived
untracked or gitignored at the repo root, so this archive is the only durable
copy. A doc that cites one by bare filename resolves here:
`docs/archive/windows/<name>`.

## windows/

- Recons/reports: `ERA_GATE_RECON.md`, `TIME_BAR_RECON.md`, `CACHE_SHAPE_RECON.md`,
  `PROPSIM_BASELINE.md`, `FVG_CENSUS.md`, `QL_TRAINING_UI_RECON.md`,
  `QL_UI_TRAIN_GAP_RECON.md`, `W3A_PRECEDING_6_REVIEW.txt`, and the TIMEBAR land
  records (`TIMEBAR_GREENLIGHT_REPORT.md`, `TIMEBAR_FIX_REPORT.md`,
  `TIMEBAR_FIX_GREENLIGHT_REPORT.md`).
- **`W3_CONFIG_RECON.md`** — cited by `docs/DECISIONS.md` D-036 as the W3
  training-config evidence; the load-bearing one.
- `*_DIFF.txt` — per-window git-diff captures (W1→PROPSIM/PRESETS/INGEST/CACHE
  eras). Reproducible from git history; retained as the review record.
- `scratch_*.py` — the re-runnable probe scripts whose outputs back the recons
  (e.g. `scratch_root_census.py` for FVG_CENSUS, the `scratch_cache_shape_*`
  trio for CACHE_SHAPE part C, `scratch_ingest_*` for the INGEST identity
  checks). Their bulk outputs (logs/JSON) were deleted as regenerable.

## Deleted (not archived) in the same window

Root run logs (`INGEST.log`, `QLUI_ACCEPTANCE.log`, `W3A_*`/`W3_*` logs), raw
`*.out` captures, `scratch_root_census_results.json` (5.8 MB, regenerable),
`route_seam_report_ingest.json`, the `.w3_new_bundle_name` state dotfile,
`_scratch_timing/`, and `_w3_oldreader_cache_backup/` (dead since the D-P-16
vectorized-reader adoption). Rationale: raw output and backups carry no
citations; reports/recons/diffs/probe-scripts do, and were archived instead.
