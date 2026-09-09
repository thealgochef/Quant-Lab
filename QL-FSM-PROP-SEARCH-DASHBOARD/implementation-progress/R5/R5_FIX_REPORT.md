# R5-FIX — Owner gate-review findings → resolutions

**Trigger:** the owner reviewed the R5 gate package (`dda40c9`) and did
NOT approve: the browser smoke was presented as clean final evidence
while being neither (17 Arrow tracebacks, pre-fix screenshots, 52
deprecation warnings), and six substantive defects stood behind it.

**Resolution commit:** **`fb8062f`** on
`feature/ifvg-prop-robust-config-search-v1` (parent `dda40c9`; 20 files,
+714/−171; raw stat `_r5fix_diff_stat.txt`). NOT pushed, NOT merged.
`implementation_status: complete`; `acceptance_status` unchanged
(`transitively_blocked_by_R1` — the owner fixture authorization remains
the program-wide first blocker).

**Release-final verification:** full repo **1606 passed** (R5's 1599 + 7
net new; exit 0, 7:47; raw `_r5fix_pytest.txt`) · `ruff check src tests
scripts` clean (raw `_r5fix_ruff.txt`) · `git diff --check` clean ·
browser smoke RE-RUN after the fix commit with a **zero-traceback,
zero-deprecation server log** (`_smoke_server_r5fix.log`) and a capture
manifest binding commit / pipeline id / viewport / artifact ids
(`browser-smoke/MANIFEST.json`).

---

## Finding 1 — 17 Arrow serialization tracebacks in the live smoke

**Root cause (verified):** all 17 tracebacks name one column —
`Conversion failed for column OOS rows with type object` — the ladder
panel's planned-GAM row put the string `"—"` into an otherwise-int
column (the fragment re-rendered the panel repeatedly, once per
traceback).

**Fix:** `scripts/ifvg_pipeline_tab.py` builds typed, Arrow-safe frames:

- `_ladder_frame()` — `OOS rows` = nullable **Int64** (`pd.NA` for the
  planned rung), `Brier`/`Brier skill` = nullable **Float64**,
  `Rung`/`AUC`/`Status` = **string** dtype (AUC formats the float or
  carries the reason text).
- The attempts table (`Attempt`/`Workers` Int64) and the S14 comparisons
  table (`Jaccard` Float64) got the same treatment — the only other
  mixed-type columns on the surface.

**Evidence:** regression test
`test_ladder_frame_is_arrow_safe_with_nullable_dtypes` asserts the exact
dtypes AND that `pyarrow.Table.from_pandas(frame)` succeeds directly (the
precise call that failed 17×); final smoke log contains **zero**
`Serialization of dataframe` messages; `03_monitor_ladder_panel.jpg`
shows 0/None numeric cells.

## Finding 2 — Screenshots pre-dated the adversarial fixes

**Fix:** the smoke was re-run AFTER the fix commit and all screenshots
regenerated (6 captures). The smoke page header now renders the manifest
caption (commit `fb8062fe8512` · tree `23c8037f052e` · pipeline
`0472b6cacdc9` · scratch key `bf758c8e9eff0595` · prepared timestamp),
so every capture is self-dating, and `browser-smoke/MANIFEST.json`
records capture time, commit, and artifact ids. The pre-fix captures
moved to `browser-smoke/superseded-pre-fix-2026-08-21/` — retained as
sequence evidence, no longer presented as final.

## Finding 3 — Production runner registry depended on `tests.*`

**Fix:** `search/runner_registry.py` now holds ONLY the two real src
executors in `REGISTERED_RUNNER_ENTRIES`; the module's source never
names a `tests.*` path. Synthetic fixture wiring is registered by the
development checkout itself via the new guarded
`register_development_runner_entries` (keys must carry the `synthetic`
marker; production keys unshadowable; exact `module:function` shape;
validate-then-commit atomicity; idempotent-on-match; conflict refusal) —
`tests/agents/ifvg_search/conftest.py` registers the two synthetic
entries at import. A production worker process, where no registration
ran, refuses the synthetic keys exactly like unregistered entries
(fail-closed preserved end-to-end). DECISIONS_TAKEN **#43**.

**Evidence:** `test_production_registry_never_resolves_into_the_tests_package`
(asserts the production map's values and scans the module source),
`test_development_registration_is_guarded`, and the existing job-shim
suites green through the conftest registration.

## Finding 4 — Smoke harness could reuse stale evidence

**Fix:** `r5_smoke_app.py` rewritten. The scratch root is
content-addressed: `%TEMP%/ifvg_r5_smoke/<key16>` where the key hashes
`{git commit, source-tree digest (163 files across the exercised ifvg /
propsim / scripts / test-fixture surfaces + the harness itself),
pipeline_semantic_id}`. Any change to commit, sources, or fixture
semantics lands in a fresh directory; every OTHER key is deleted on
startup. Reuse is additionally **evidence-tied**: it requires the
persisted state to show every planned stage terminal AND the pipeline
result envelope to reload through the verifying store
(`load_verified_envelope` manifest + file-hash + id-hashes-payload
checks) — a bare marker file no longer exists. `smoke_manifest.json` in
the keyed directory records the key inputs and result ids
(`reused_validated_evidence: false` for this run — built fresh).

## Finding 5 — "Parity held over 0 OOS rows"

**Fix:** with zero OOS rows the claim is now **"not evaluable"**
everywhere the copy appears: the S09 stage explanation
(`pipeline.py::_stage_s09_train`), the persisted S10 diagnostics
(`supervised_ladder._assert_identical_rows` stamps
`parity["status"] = "held" | "not_evaluable"`), and the monitor caption
(`ifvg_pipeline_tab._parity_caption`). "Held" is claimed only when rows
exist to hold over.

**Evidence:** `test_parity_caption_wording_for_zero_and_nonzero_rows`,
updated `test_monitor_ladder_panel_shows_rungs_planned_and_s11` (asserts
"parity not evaluable" AND the absence of "parity held" on the 0-row
fixture), the E2E assertion on S09's explanation + the persisted
`not_evaluable` status, and the "held" assertion on the >0-row ML
fixture. Live: `02a_…` (stage row) and `03_…` (caption) screenshots.

## Finding 6 — Overloaded comparison identity

**Fix:** `ComparisonResult.subject` is a discriminated union
(`Field(discriminator="subject_kind")`):
`StudyCellComparisonSubject.comparison_id` (names a frozen
`ComparisonEnvelope`) | `SearchDerivationComparisonSubject.derivation_id`
(the DECISIONS_TAKEN #42 cross-profile derivation id). The bare
`comparison_id` string field is GONE — a 64-hex string is no longer a
lawful subject, and a mislabeled reference fails validation. S14 mints
the search-derivation kind; the registered example factory carries the
study-cell kind; the monitor renders a `Subject` column.
DECISIONS_TAKEN **#45** (#42's wording brought forward).

**Evidence:**
`test_comparison_result_subject_is_a_typed_discriminated_reference`
(both kinds valid; untyped dict, cross-kind field, and bare string all
refused), the E2E's persisted-envelope subject assertions, and the
`03b_monitor_comparisons_typed_subject.jpg` capture.

## Finding 7 — Expected seed trusted from the caller

**Fix:** the real verification branch of S00 now REFUSES without
`PipelineWiring.loaded_seed_snapshot_id_source`
(`_loaded_seed_snapshot_id_for_real_scope`; fail-before-path). The
source — wired by the real executor
(`executors.loaded_seed_snapshot_id_source`) — performs the VERIFIED
seed-snapshot store load (`load_seed_snapshot`: manifest + file hashes +
envelope-id-hashes-payload + seed-bytes rehash + profile binding) and
returns the loaded envelope's content-derived id; THAT id is what
`validate_verification_run` checks against the owner's authorization.
`expected_seed_snapshot_id` survives as an optional cross-check only; a
set-but-disagreeing value refuses. DECISIONS_TAKEN **#44**; supersedes
the F3/m-16 disposition (addendum in
`ADVERSARIAL_REVIEW_RESOLUTION.md`).

**Evidence:** `test_real_scope_seed_requires_the_loaded_artifact_source`
(missing source refused even WITH a caller id; disagreement refused;
agreement passes) and
`test_executor_seed_source_returns_the_loaded_artifacts_id` (real loader
returns the persisted artifact's id; missing artifact and profile
mismatch refuse inside the load).

## Finding 8 — Deprecated `use_container_width` (52 warnings)

**Fix:** all **73** occurrences across the 7 branch-touched scripts
(`ifvg_pipeline_tab` 9, `ifvg_lab_tab` 24, `ifvg_verifier_tab` 22,
`ifvg_results_tab` 9, `ifvg_results_compare` 6, `ifvg_active_runs_tab`
2, `ifvg_study_wizard` 1) replaced with `width="stretch"` (every call
was `=True`; widget `width` support verified by signature inspection on
the installed Streamlit). `pyproject.toml` floor bumped
`streamlit>=1.41 → >=1.54` (the verified floor for the `width` API).
Final smoke log: **0** deprecation messages. Out-of-scope note: 29
occurrences remain in `dashboard.py` / `experiment_tab.py` /
`ml_training_tab.py` — pre-branch files owned by other windows, recorded
here as backlog, untouched to keep this branch's review scope clean.

## Finding 9 — Stale evidence documents

All brought forward in place:

- **`DEVIATIONS.md`** — DEV-R5-5 rewritten: the gate is **DERIVED** from
  the tripped-assertion record (test-witnessed), with the
  "true-by-construction" framing explicitly superseded; DEV-R5-1's
  "sanctioned tests-package pattern" claim superseded by the finding-3
  registration design; DEV-R5-8 updated to the typed subject.
- **`TEST_RESULTS.md`** — supersession note up top; R5-FIX FINAL command
  table (1606 / ruff / diff-check, with raw-output file references); the
  post-fix smoke section with the zero-traceback log and the 6 new
  screenshots; the pre-fix smoke section explicitly marked SUPERSEDED.
- **`FILES_TOUCHED.md`** — headings reconciled against
  `git diff --name-status` (50 = 27 A + 23 M: 10 new src, 2 new scripts,
  12 modified src + 2 modified scripts, 15 new + 5 modified test files,
  4 docs — the miscounted "6 new src / 11 new tests / 12-row mixed
  modified table / 6 modified tests" corrected), the missing
  `search/failure.py` row restored, and the R5-FIX file table added.
- **`GATE_SUMMARY.md`** — R5-FIX banner; both commits with reconciled
  stats; release-final headline (1606 + clean smoke) replacing the old
  smoke-as-final claim; the superseded adversarial dispositions noted.
- **`ADVERSARIAL_REVIEW_RESOLUTION.md`** — addendum recording the
  F3/m-16 and F10(b) supersessions.
- **`../DECISIONS_TAKEN.md`** — #42 wording updated; §R5-FIX #43–#45
  added.

---

## Directive checklist (owner's fix list → status)

| Directive | Status |
|---|---|
| Fix nullable data types in the ladder table | **DONE** — `_ladder_frame` Int64/Float64/string + attempts/comparisons tables; Arrow regression test |
| Replace deprecated `use_container_width` | **DONE** — 73/73 in branch scripts → `width="stretch"`; streamlit floor 1.54; 0 warnings in the final log |
| Change zero-row parity copy to "not evaluable" | **DONE** — S09 explanation + persisted parity status + monitor caption; tests both wordings |
| Remove production-registry dependence on `tests.*` | **DONE** — production map src-only + guarded dev registration; source-scan test |
| Add a typed comparison-subject reference | **DONE** — `subject` discriminated union; bare ids unlawful |
| Ensure the actual loaded seed is verified against authorization | **DONE** — required verified-load seam; caller ids demoted to cross-checks |
| Clear or content-address the smoke scratch directory | **DONE** — both: content-addressed key + stale-key clearing + evidence-validated reuse |
| Rerun browser smoke after the final fixes | **DONE** — re-run on `fb8062f`, fresh build (`reused_validated_evidence: false`) |
| Require zero tracebacks in the final server log | **DONE** — `_smoke_server_r5fix.log`: 0 tracebacks, 0 deprecations, 0 Arrow fallbacks |
| Regenerate screenshots with a manifest (commit, pipeline ID, viewport, artifact IDs) | **DONE** — 6 captures + `browser-smoke/MANIFEST.json` (all 16 stage_result_ids + outputs, result envelope id, viewport, timestamps); old captures archived |
| Update the stale evidence documents | **DONE** — all five documents + DECISIONS_TAKEN brought forward (see Finding 9) |
| Include raw Ruff/diff output and a source diff/stat | **DONE** — `_r5fix_ruff.txt`, `_r5fix_pytest.txt`, `_r5fix_diff_stat.txt` (show --stat + range stat) |

## What did NOT change

- No identity previously persisted outside tmp/scratch stores is
  affected (the `ComparisonResult` shape change re-mints ids only in
  synthetic test/smoke stores; the repo's real data namespaces were
  never written).
- The owner-authorization trust anchor, the two-path verification
  design, the canonical allowlist, S11's blocked state, and the
  verify-then-activate publication flow are unchanged — every finding's
  fix strengthens a fail-closed rule; none weakens one.
- Shared user-dirty docs (`ARCHITECTURE.md`, `docs/README.md`,
  `docs/pipeline_state.yaml`, `docs/ML_TRAINING_WORKBENCH.md`) were not
  touched by this round and are NOT part of `fb8062f`.

**R6 remains NOT started**, per the owner's instruction, pending this
R5-FIX approval.
