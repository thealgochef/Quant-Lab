# R1 — Adversarial Review Resolution

Every finding was accepted as valid and fixed (or, where the finding was a
documentation/recording gap, resolved by the corresponding record). Post-fix
verification: `ruff check src tests` clean; lane suite **137 passed**; full
regression re-run recorded in `R1_PYTEST_FULL.txt` and `GATE_SUMMARY.md`.

| Finding | Resolution | Post-fix verification |
|---|---|---|
| C1 (BLOCKER) catalog lock race loses events | Retry loop treats `PermissionError`/`OSError` as contention (Windows delete-pending); holder-side unlink tolerates a broken lock | `test_concurrent_catalog_publishers_lose_nothing` (8 writers × 25, zero loss), suite-stable |
| C2 registry-blind cell payload + unlawful baseline bundle key | `_all_dimensions_registered_and_usable`: registered-vocabulary refusal (bundle keys incl. frozen tiers, model/decision/calibration/stress/warmup/date ids), active-dimension `assert_dimension_usable` (an S11-blocked decision policy now refuses at the CELL), pairwise incompatibility; baseline bundle key → `B0_CORE` (DEV-R1-8) | `test_cell_payload_refuses_unregistered_and_blocked_dimensions` |
| C3 slice zero-sentinel identity, nothing published | `_assemble_slice_identity`: authorized hashing of the exact cached inputs → real `ReplayInputBundle` + `CoreStrategyReplayIdentity`; neutrality keyed by the real id; bundle+core envelopes published via `save_or_reuse` (reload-asserted); `invariants_passed`/`artifacts_published` gates now honestly companion-dependent; new store `replay_input_bundles` | `test_slice_composes_identity_publication_and_honest_gates` (incl. idempotent reuse) |
| C4/S10 marker root evasion | `register_program_allowlist` cross-checks the CANONICAL marker regardless of target root; the slice binds `store_root` to `repo_root/search_test/v1` (the `output_namespace` Literal made executable) | `test_slice_refuses_non_canonical_store_root`; `test_program_allowlist_marker_refuses_rotation` |
| C5 synthetic charters unconfined | `save_charter` refuses synthetic charters in any `search`-but-not-`search_test` namespace | `test_synthetic_charters_are_namespace_confined` |
| C6 generated baseline arm missing | `validate_charter(generated_profile_capability=…)`: generated names accepted only with `generated_runnable` capability; unknown names refused with the truthful reason | `test_generated_baseline_requires_its_capability` |
| C7 ratification posture | Uniform: every AVAILABLE baseline of an APPROVED_SEARCH_AXIS is `pending` (decision 2) | `test_searchable_baselines_have_a_uniform_pending_posture` |
| C8 axis-key binding + incompatibilities | Both `assert_axes_authorized` and `resolve_axis_overrides` verify value↔axis binding; declared incompatibilities enforced across the selection | `test_value_bound_to_another_axis_is_refused` |
| C9 dependent member standalone | `entry_parent_distance_ticks_max` → `MEASUREMENT_ONLY` (structurally unsearchable; composite remains the unit) | `test_dependent_composite_member_is_not_standalone_searchable` |
| C10 delta_reports type + full diff | Nested `ImmutableMap[str, ImmutableMap[str, Any]]` restored; `build_config_diff` exported; `config_diff_only` results carry per-field diff paths | `test_config_diff_only_carries_the_full_field_diff` |
| C11 placeholder equalities / exemption / required difference | Everything-else-equal classes evaluated STRUCTURALLY from the cell diff; DATA_LINEAGE exemption removed (identical cells refused); `REQUIRED_DIFFERENCES` (model-gate ⇒ new `gross_trade_stream_hash`); `composite_preregistered` takes the charter-frozen list via `charter_equalities` (refused elsewhere) | `test_everything_else_equal_classes_are_structural`; `test_model_gate_execution_requires_a_new_trade_stream` |
| C12 logical-key equality | feature_only/cohort_model require `resolved_model_protocol_id` | `test_feature_only_requires_the_resolved_model_identity` |
| C13 mutation-adversarial rigor | `_walk_and_attack` recursively visits every reachable container, attempts mutation on each map, and PROVES no plain dict/list/set is reachable on any registered payload | `test_mutation_after_identity_cannot_change_payload_hash` |
| C14 stale lock | mtime-based stale-lock break (threshold param) after timeout; fresh locks still time out | `test_stale_orphan_lock_is_broken_after_threshold` |
| C15 snapshot continuity | Slice asserts `snapshot.first_replay_day == allowlist[0]` before any replay | `test_slice_refuses_snapshot_discontinuous_with_allowlist` |
| C16 coverage gaps | (a) synthetic slice-composition suite added (`test_verification_slice.py`); (b) real scoped-source-tree fixture test for `quant_lab_replay_source_identity` (tmp git repo); (c/d/e) deferred rows recorded explicitly in `TEST_RESULTS.md` (UI provider → R4; regime-registry checks → R6 with contracts authored at R5/R6; per-surface legacy guards → as their surfaces land) | `test_ql_replay_source_identity_scoped_tree_fixture` |
| C17 descriptor schema hashes | v3-context blocks pin the REAL locked Arrow registry hash (`IFVG_CONTEXT_ARROW_REGISTRY_HASH`); v2 blocks pin the exact versioned capture/dataset schema surface; artifact-independence of block resolutions documented (cells pin artifacts) | block resolution ids re-minted deterministically; partition tests green |
| S1 (MAJOR) adapter write path | `run_child_replay` refuses `cached_artifacts_only=False` under an `ArtifactProvenanceReadAdapter` (fail-closed read-only guard) | `test_provenance_adapter_is_read_only_in_the_worker` |
| S2 trusted-branch bounds | Verification branch now requires a NON-EMPTY ≤5 allowlist | covered by policy construction + branch tests |
| S3 store id shape | `envelope_destination` requires fullmatch 64-hex (drive-relative/ADS refused) | extended `test_store_exact_id_resolution_never_lists` |
| S4 `-O`-stripped asserts | Import-time invariants converted to explicit raises (feature_blocks, feature_bundles); the slice's redundant isinstance assert → explicit refusal | import-time behavior preserved under `-O` |
| S5 progress-doc wording | FILES_TOUCHED reworded (commit at release checkpoint; gate files listed as produced at gate close) | doc updated |
| S6 evidence scripts | `coverage_matrix_build.py` + `window_coverage_scan.py` preserved in R1/ and both artifacts REGENERATED by exactly those scripts (matrix id re-minted `a4a4835a…`; draft authorization updated) | scripts re-run successfully |
| S7 output namespace | Regression test pins the `search_test/v1` Literal | `test_output_namespace_literal_rejects_research_store` |
| S8 DatePolicy lower bound | `>= 2026-01-01` enforced | charter date-policy test extended |
| S9 pickle gadgets | `_SeedSandboxUnpickler` (strategy_core graph + narrow safe set only) | `test_seed_sidecar_unpickler_refuses_gadgets` |
| S10 | = C4 above | — |

No finding was dismissed. Two findings changed design records rather than
plan semantics: DEV-R1-8 (baseline bundle key, plan-internal shorthand
conflict resolved toward the typed registry) and the C2-driven decision that
a V1 cell CANNOT carry an execution-gating decision policy (fail-closed
reading of §7A.3; the comparison layer for the post-ratification era is
tested on a documented validation-bypassed construction).
