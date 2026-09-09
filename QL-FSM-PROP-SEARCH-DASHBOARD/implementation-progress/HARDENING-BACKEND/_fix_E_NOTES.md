# Fix-E notes — adversarial round: RA-04, RA-05, B-01, B-03, B-04, B-05, B-07, B-08

Closed 2026-09-02. Red first: `_red_fix_E.txt` (8 failed / 2 passed before the
code changed — the two passes are B-07 and B-08, test-adequacy findings whose
code was already correct; their tests now guard the mutations). Final:
`_fix_E_pytest.txt` — **53 passed** (`test_bounded_verification.py` 13,
`test_bounded_verification_script.py` 2, `test_pipeline_authority_seams.py` 3,
`test_event_detail_streaming.py` 9, `test_stratified_prop_external_aggregation.py`
7, `test_identities.py` 19) under the registered warnings-as-errors policy;
`ruff check` clean over every file touched. No file outside my ownership was
edited; no git write; `data/` untouched.

## Files touched

| File | Change |
|---|---|
| `src/.../ifvg/search/bounded_verification.py` | RA-05 seed-step mapping + new reason `seed_snapshot_unverifiable`; B-01 `_s14_component` reads `regime_stratified_reports.json` (typed `children_skipped`, `fitting_performed`, `report_ids`; a record that claims fitting → `unexpected_state`); B-03 `_fold_component` reads the typed `fold_summary.json` (else the regime `fold_sample_adequacy.json` preview outcome; else `unexpected_state`) — the explanation regex is gone; B-04 `_panel_component` loads the PERSISTED panel artifact (`load_context_bar_panel_artifact` + `load_context_bar_panel_frame`) and records `row_count`, `valid_row_count`, `typed_null_reason_counts` (`typed_insufficiency` when every row carries a typed reason; the failure-string branch is gone); RA-04 `STORE_BEHAVIOR_PROOF_IDS` + `store_behavior_proofs(store_root, table_id, scratch_root)` (scratch copies: identical bytes load AND a same-identity publication reuses; a flipped sidecar byte, a missing manifest, a corrupt manifest and a truncated sidecar each fail closed with the typed reason; the scratch root is removed in `finally`; an already-tampered source reports a failed identical-bytes proof) |
| `scripts/ifvg_bounded_verification.py` | RA-04: attempt 1's `search_results` envelope is captured BEFORE attempt 2 (`first_attempt_gates` ≠ `second_attempt_gates` by construction); the executed-trade table ids come from the state's children rows (the previous `executed_trade_table_id_for(store_root, core_id)` call had the wrong signature — it takes `core_replay_id` and `record_schema_version`); the four store proofs are GATHERED per table via `store_behavior_proofs` under `<evidence_dir>/_scratch_store_probes` (AND across tables), `native_ids_repeat` also requires the two attempts' pipeline-result ids to be identical; the test-suite refs stay as `supplementary_store_suite`; `store_probe_observations` are recorded in `evidence_refs`; the evidence dir is created before the attempts |
| `tests/agents/ifvg_search/test_bounded_verification.py` | + `test_preflight_types_seed_failures_by_their_cause` (missing entry / corrupt sidecar → `seed_snapshot_unverifiable`; loader profile refusal → `seed_profile_mismatch`; a `RuntimeError` propagates), `test_fold_outcome_comes_from_the_typed_fold_summary_not_the_explanation` (a lying explanation is ignored; no typed sidecar → `unexpected_state`), `test_bounded_report_carries_typed_child_skips_from_the_s14_record` (skips propagate; a fitting claim fails), module fixture `completed_panel_regime_pipeline` (a REAL panel-grain regime run) + `test_bounded_report_reads_the_real_s14_record_fold_summary_and_panel_artifact` (the S14 record under its persisted name, the typed fold summary, the persisted panel artifact's validity counts), `test_store_behavior_proofs_probe_the_fixture_table_and_detect_a_tampered_source` |
| `tests/agents/ifvg_search/test_pipeline_authority_seams.py` (new) | B-05: a FULL-development charter carrying a real bundle — an unmarked store, a foreign namespace and a stale witness (after `publish_supersession`) refuse at S00 with no stage run and no child row; a bound bundle passes the S00 authority check (S00 completed; explanation names the bound namespace) |
| `tests/propsim/test_event_detail_streaming.py` | + `test_iterable_form_refuses_a_path_count_overrun_while_streaming` (B-08: the early `exceeded while streaming` refusal fires before any flush / manifest) |
| `tests/agents/data_infra/ifvg/test_stratified_prop_external_aggregation.py` | + `test_byte_budget_refuses_before_publication_with_a_generous_row_budget` (B-07: the written-file byte branch and the empty-table byte branch refuse typed; temp dirs cleaned; nothing published; a generous byte budget passes) |

## Dispositions

- **RA-05 — FIXED.** `except Exception` is gone; `SeedSnapshotError` → `seed_profile_mismatch`, `StoreNamespaceError` → `store_namespace_refused`, `SearchStoreError` (incl. `SidecarLoadError`) → `seed_snapshot_unverifiable` (registered), anything else propagates.
- **B-01 — FIXED.** The real sidecar name (`regime_stratified_reports.json`, `pipeline.py` S14) is read from a real panel-grain regime run's stage result; `children_skipped` / `fitting_performed` / `report_ids` are the evidence; the propagation of a non-empty skip and the fitting-claim refusal are proven with a monkeypatched record (the synthetic fixtures skip no child).
- **B-03 — FIXED** (with the main agent's `fold_summary.json` S08 sidecar). The regime label-free branch reads `fold_sample_adequacy.json`'s `expected_gate_outcome` (`no_valid_folds` → `typed_no_valid_fold`; `pass`/`fail` → `valid_folds_present`); no explanation parsing remains.
- **B-04 — FIXED-VARIANT.** The typed counts come from the persisted panel frame (`cbp_valid`, `cbp_missing_reason`) rather than the S06 coverage sidecar; the `typed_insufficiency` outcome is now reachable (every row typed null). On the synthetic panel fixture the outcome is `panel_materialized` with the reason counts recorded.
- **B-05 — FIXED (test).** Pipeline-level coverage of the S00 full-development bound-bundle check. Observation for the main agent: the `store_namespace_missing` refusal text embeds the store root's absolute path, so `sanitize_failure_message` withholds the whole S00 explanation ("failure details withheld (sanitized)") — the typed reason does not survive into the state file for the unmarked-store case (the foreign-namespace and stale-witness cases keep their reason text). Recommended (outside my ownership): keep the path out of the `store_namespace_missing` message (or sanitize per token) so the state carries the typed reason.
- **B-07 / B-08 — FIXED (tests).** Both branches were correct; the mutations named by Reviewer B are now caught.
- **RA-04 — FIXED.** Attempt-1 gates captured; the four proofs gathered on the fixture's own tables through scratch copies; hard-coded `True`s gone; the previous wrong `executed_trade_table_id_for` call fixed by reading the table ids the state recorded.

## Deviations

- The B-04 evidence source is the persisted panel ARTIFACT (never a frame from memory) rather than the S06 coverage sidecar named in the finding — the artifact carries the per-row typed reasons directly.
- The B-05 unmarked-store case asserts the S00 failure shape (failed; later stages pending; no child) and accepts the sanitized explanation (see the observation above); the other two refusal cases assert the typed reason text.
- The panel regime fixture used for B-01/B-03/B-04 is the existing `regime_study="panel"` shape (stratified reporting requested); its S14 record has an empty `children_skipped` (every child passes the relaxed fixture gates), so the non-empty propagation is proven by monkeypatch on the same builder.
