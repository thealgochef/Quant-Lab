# R4 — Access-safety evidence

## Protected/sealed counters: ZERO

- **No path for 2026-06-11 or the sealed range was ever constructed.** The
  ONLY occurrence of the protected date in the R4 lane is a read-only
  DISPLAY string in the wizard's Validation step
  (`scripts/ifvg_study_wizard.py`: `st.code` copy "protected: 2026-06-11
  (never constructed, listed, stat-ed, or read)") — FUX §14.1 requires
  showing the boundary; no `Path`, open, glob, stat, or listing is built
  from it. Verified by inspection of every new/modified file (the string
  appears nowhere else in the lane) and by the absence of any source-data
  reader in the R4 modules: the UI layer contains no source-partition,
  day-artifact, or capture access of any kind.
- Every R4 test ran on synthetic fixtures (the R2 synthetic 2×2 E2E over
  `SYNTHETIC_DAYS` = the 2026-01-* synthetic family) with tmp-only writes
  (`tmp_path` roots); no file under `data/` was created or modified by
  tests (`git status` over `data/` unchanged; the production draft/job
  roots were never written because every AppTest monkeypatches the module
  roots to tmp directories).

## No execution escapes

- **Launch confinement (source-scanned + AppTest-spied):**
  `subprocess.Popen` exists in exactly two places — the wizard's
  `_spawn_search_job` seam (invoked ONLY inside the freeze-button handler;
  the AppTest spy proves zero calls on render/import) and the job shim's
  detached `start`/`resume`. The wizard's only other subprocess use is a
  read-only `git rev-parse HEAD`.
- **Registry gate:** the shim now refuses ANY `--runner-entry` string that
  is not an exact registered value BEFORE `importlib` is reached (test:
  `import_module` monkeypatched to fail loudly — never called for
  unregistered entries). The only registered entry is the synthetic
  fixture wiring; no real-data executor exists until R5 registers one, so
  no UI action can start a real replay, feature build, model fit, prop
  search, or full-development run.
- **Real charters cannot freeze:** with no owner evidence bundle in
  existence, every wizard charter carries the typed synthetic marker,
  which `save_charter` REFUSES outside the `search_test` namespace
  (witnessed by the research-namespace AppTest). Synthetic charters
  freeze only into the verification namespace.

## No forbidden controls / disclosure

- Source scans (in-suite, `test_ifvg_study_scans.py`): no button label
  contains delete/sealed/recapture/promote/unlock; `allow_sealed` absent;
  no order/live/serving control; no raw path, traceback, or secret in any
  rendered surface (every user-facing error passes `sanitize_error`;
  AppTests assert the absence of local absolute paths in rendered text).
- Session-state scan: every write lands under `ifvg_study_v1_*` except the
  single registered cross-lane interaction
  (`ifvg_context_v1_replay_pair` inside `queue_replay_drilldown`).

## Frozen lanes untouched

- M0–M3 modules: byte-untouched. `ifvg_lab_tab.py` changed ONLY in
  `render_ifvg_lab_tab` (the Experiments branch delegates; the M0–M3
  renderer `render_ifvg_experiments_tab` is passed through verbatim and
  its delegation is AppTest-verified). `ifvg_verifier_tab.py` untouched in
  R4 (the R2-landed `queue_jump` contract is consumed, not modified).
- Strategy-Core and Trade-Lab: untouched (read-only per kickoff §1.5).
- Existing immutable artifacts/catalogs: untouched; the evaluation-only
  propsim API: byte-untouched (its suites green in the full run).

## `full_pipeline_not_run` holds

No full-development replay, feature materialization, model fit,
configuration search, prop search, bootstrap research run, or operator
full-pipeline run occurred during R4 implementation or verification.
