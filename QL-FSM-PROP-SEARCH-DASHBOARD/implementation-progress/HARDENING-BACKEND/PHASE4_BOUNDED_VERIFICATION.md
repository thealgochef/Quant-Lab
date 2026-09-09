# HARDENING-BACKEND — Phase 4 (plan §6): the real ≤5-day bounded verification — code authoring

**Status: authored and synthetically verified; the REAL run was NOT performed.**
The real verification starts only after Phase 2, seed production and the
owner's final `VerificationAuthorizationRef` exist (plan §6, §10 steps 4–8).
None of those owner actions exist; no real source path was constructed by
this release; `data/` gained zero files (`ACCESS_SAFETY_EVIDENCE.md`).

## What was authored

| Piece | Plan | Code | Tests |
|---|---|---|---|
| Preflight before path construction | §6.1 | `search/bounded_verification.py::preflight_bounded_verification` → `BoundedVerificationPreflight` (every check named; only minted when all pass) / `BoundedVerificationRefusalError.reason` (16 typed reasons) | `test_bounded_verification.py` (4 preflight tests) |
| R1 baseline gate report | §6.2 | `R1BaselineGatePayload/Envelope` (store `r1_baseline_gate_reports`): the six `verification_control_flow_gates_v1` gates of the first (fresh) and second (verified-reuse) attempts, the dual-drive audit-mode digests (`audit_modes_equal`), and the eight "also prove" proofs (`R1_BASELINE_PROOF_IDS`); `passed` is DERIVED (a lying `passed` or a missing proof is refused by the contract); the three stamps are literal `True` | `test_r1_baseline_gate_report_derives_passed_and_refuses_inconsistency` |
| Release-specific bounded control-flow report | §6.3 | `BoundedReleaseControlFlowPayload/Envelope` (store `bounded_release_control_flow_reports`; `not_research_evidence=True`): exactly the eight registered components in order, each typed to a registered outcome vocabulary (`BOUNDED_COMPONENT_OUTCOMES`; `passed` derives from the outcome's registered class); `build_bounded_release_control_flow_report` types every component from the COMPLETED pipeline state and its persisted, manifest-verified stage sidecars only (never a research metric) | `test_bounded_report_types_every_component_of_the_synthetic_fixture`, `..._fails_closed_on_untyped_states`, `..._types_a_regime_non_fit_from_the_persisted_record` |
| Operator runner | §6.4 | `scripts/ifvg_bounded_verification.py`: `preflight` (exact-load the store's persisted run → §6.1) and `run` (preflight → the REAL registered executor `pipeline_baseline_verification_v1` twice → both reports → the §6.4 evidence files `VERIFICATION_RUN.md`, `R1_BASELINE_GATES.json`, `BOUNDED_RELEASE_CONTROL_FLOW.json`, `ACCESS_AUDIT.jsonl`, `REUSE_RECEIPT.json`, `MANIFEST_AND_SIDECAR_HASHES.json`); importing launches nothing | `test_bounded_verification_script.py` (2): both subcommands refuse `fail_before_path` on a run-less store (nothing written); the preflight passes on a persisted, bound synthetic run |

The runner was executed against the repository's REAL verification store
root (`data/ifvg_datasets/search_test/v1`, which is unmarked and holds no
persisted verification run): it refused before any path —
`{"status": "refused", "reason": "fail_before_path", "detail": "real
verification execution is blocked: no persisted VerificationRunEnvelope
exists for this store — the owner's VerificationAuthorizationRef (decisions
21/R-5) has not been granted (fail-before-path)"}` — and `data/` gained no
file.

## §6.1 preflight — the registered check order

1. `output_namespace_not_locked` — the output root must BE `repo_root/data/ifvg_datasets/search_test/v1`.
2. `research_destination_refused` — a research-looking root or a run whose `output_namespace` is not `search_test/v1`.
3. `store_namespace_refused` — the store must be a verified, coherently deployed `test` namespace (§4.1).
4. `synthetic_marker_refused` — the authorization must be a real `VerificationAuthorizationRef`.
5. `supersession_head_witness_refused` — the ref must name this namespace and its witness must equal the CURRENT head (§4.2).
6. `sixth_day_refused` / `allowlist_not_chronological` / `protected_or_sealed_date` — 1–5 unique ordered days, every day before 2026-06-11.
7. `verification_authorization_invalid` — `validate_verification_run(..., store_root)` (allowlist hash, seed, coverage matrix, profile, section hash, pipeline identity, namespace + witness).
8. `date_domain_mismatch` / `logical_days_not_consecutive` — every day is a LOGICAL trading-day id under `cme_globex_18et_weekday_v1` (a physical Sunday partition date is refused) and the window is consecutive.
9. `source_inventory_mismatch` — when the owner-selected `VerificationTradingDayRef` mapping is supplied it must cover exactly the allowlist, each day with both hash-addressable physical partitions.
10. `rotated_window_refused` — a registered canonical program allowlist must equal the run's allowlist (nothing is registered here).
11. `seed_profile_mismatch` / `seed_discontinuous_with_window` — the seed the runner will actually load is exact-loaded (profile-bound), and its `first_replay_day` is the window's first day.

Each logical day's physical partitions are recorded (`td−1`, `td`) — for
the synthetic proof over 2026-06-04…06-05: `2026-06-03, 2026-06-04,
2026-06-05`.

## §6.2 — the "also prove" proofs

`native_ids_repeat`, `core_table_hashes_repeat` (audit-disabled vs
audit-enabled dual-drive digests), `executed_trade_table_exact_loads`,
`second_invocation_reused_with_zero_replay`, `identical_bytes_reuse`,
`different_bytes_fail_closed`, `missing_or_corrupt_manifest_fails_closed`,
`corrupt_sidecar_fails_closed`. The runner gathers the first five from the
two attempts' state / stores; the last three are the immutable store's
behaviors proven by `test_store_sidecar_probe.py`, `test_executed_trade_table.py`
and `test_pipeline_evidence_integrity.py` on the exact code (recorded as
evidence refs). **Reading of the plan's "audit-disabled and audit-enabled
modes":** the baseline slice runs dual-drive (both modes in one replay,
compared by the neutrality report); the report therefore carries the two
attempts' gate reports plus the two audit-mode digests rather than two
separate single-mode pipeline runs (`DEVIATIONS.md`).

## §6.3 — component outcomes on the synthetic fixture

The synthetic full-plan pipeline fixture (three synthetic January days,
verification scope, synthetic marker) types as: `mbp1_diagnostic`
not_planned · `context_bar_panel` not_planned · `fold_construction`
**typed_no_valid_fold** (0 folds constructible under the frozen protocol —
the documented safe-failure state on short windows) · `regime_fit`
not_planned (the regime non-fit branch is proven from a persisted
`regime_run.json` record with zero fits and the typed gate failures) ·
`supervised_models` **typed_non_fit_zero_predictions** (0 OOS rows; zero
fabricated predictions) · `s14_reports` verification_report_zero_fitting ·
`s15_publication` **verification_only_result** (stamped, never activated) ·
`s11_model_gated_replays` **blocked_with_registered_reason**. A tampered
state (S11 "ran anyway"; an activated publication) types `unexpected_state`
and fails the report; an unregistered outcome or a lying `passed` is refused
by the contract.

## Adversarial-round amendments (Fix-E; `_fix_E_NOTES.md`)

- **Preflight seed step (RA-05)**: failures are typed by cause — `SeedSnapshotError` →
  `seed_profile_mismatch`, `StoreNamespaceError` → `store_namespace_refused`,
  `SearchStoreError` / `SidecarLoadError` → the new `seed_snapshot_unverifiable`; anything else
  propagates (17 refusal reasons now).
- **S14 component (B-01)** reads the sidecar S14 actually persists
  (`regime_stratified_reports.json`) and carries `children_skipped`, `fitting_performed`,
  `report_ids`; a record claiming fitting types `unexpected_state`. **Fold component (B-03)**
  reads the typed S08 `fold_summary.json` (else the regime preview outcome; else
  `unexpected_state`) — no explanation parsing. **Panel component (B-04)** loads the PERSISTED
  panel artifact's validity counts (`typed_null_reason_counts`, `valid_row_count`, `row_count`);
  `typed_insufficiency` is reachable. A REAL panel-grain regime run backs these tests.
- **Runner (RA-04)** captures attempt 1's gates before attempt 2, reads the executed-trade table
  ids from the state's children rows, and GATHERS the four store proofs per table via
  `store_behavior_proofs` (scratch copies under the evidence dir: identical bytes load / reuse;
  a flipped sidecar byte, a missing manifest, a corrupt manifest, a truncated sidecar each fail
  closed; scratch removed in `finally`); `native_ids_repeat` also requires identical
  pipeline-result ids across the attempts; the test-suite refs are supplementary.
- **Pipeline-level authority proof (B-05)**: `test_pipeline_authority_seams.py` — a
  full-development charter with a real bundle refuses at S00 (unmarked store / foreign namespace /
  stale witness) with no stage run and no child row; a bound bundle passes S00.

## What remains for the owner (not performed)

- Select the permanent logical window from `VERIFICATION_WINDOW_SHORTLIST.md`
  (§5.1; the June proposal is INELIGIBLE as stated — no exact verifier target).
- Sign `SeedProductionAuthorizationRef`; run the seed-only production chain
  (`scripts/ifvg_seed_production.py run`); review the concrete seed.
- Sign the final `VerificationAuthorizationRef` (the unsigned packet builder
  is `seed_production.build_verification_authorization_packet`).
- Initialize the real verification store namespace explicitly as `test`
  (`scripts/ifvg_store_namespace.py init --store-root data/ifvg_datasets/search_test/v1 --namespace-class test --confirm`).
- Register the one program allowlist and run `scripts/ifvg_bounded_verification.py run`;
  the evidence folder `R1-VERIFICATION/` is produced by that run, not by this release.
