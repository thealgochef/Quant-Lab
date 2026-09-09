# R2 — Access-Safety Evidence

## Real-data operations this release performed

**None.** Every R2 test and fixture is synthetic:

- The three-day synthetic walk (`tests/agents/ifvg_search/conftest.py`)
  generates bars in memory; on-disk chains write ONLY into pytest tmp
  directories under `ExplorationDataPolicy` (the synthetic days 2026-01-13…15
  sit inside the fixed exploration allowlist by construction — no real
  parquet exists in any tmp data dir, and `cached_artifacts_only=True` paths
  read only the artifacts the tests themselves wrote).
- No test touches `data/databento/`, any accepted immutable dataset, any
  existing catalog, or any real source partition. The slice-companion
  composition test runs entirely against tmp stores with synthetic
  repository states.
- The verifier-tab and job-shim tests are source-scan/state-machine tests; no
  Streamlit runtime, no detached process is launched by any test (the
  worker-mode test runs in-process via `job.main([...])` with a synthetic
  runner entry).

## Protected/sealed counters

**Zero.** No path for 2026-06-11 or the sealed range was ever constructed:

- The only date literals in R2 code/tests are the synthetic days
  (2026-01-13…15) and the R1-inherited PROPOSED allowlist constant (unchanged,
  still a proposal).
- `DatePolicy`/`VerificationDataPolicy` validators (R1) still refuse any date
  ≥ 2026-06-11 before path construction; R2 added no bypass and no new date
  authority.
- The new job shim validates the 64-hex `search_id` before any path shaping
  and refuses execution without an explicit runner entry.

## Full-development-run guard

`full_pipeline_not_run=true` holds for every verification surface. R2 ran no
full-development replay, no feature materialization, no model fit, no prop
simulation, and no configuration search over real data. The orchestrator's
only executions in this release were synthetic child runners inside pytest
tmp directories.

## Mutation guarantees

- M0–M3 lane modules: untouched (verified by `git status` scope + the
  1,089 pre-existing tests passing unchanged).
- Existing propsim modules: untouched.
- Strategy-Core / Trade-Lab: untouched (read-only per the kickoff).
- Existing immutable artifacts and catalogs under `data/`: untouched (no R2
  code writes outside explicit `store_root`/tmp arguments; the two mtime
  sweeps in the adversarial review verify this independently).
