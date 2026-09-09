# R2 — Test Results

## Commands, counts, timings (final; exit codes all 0 unless stated)

| Command | Result | Time |
|---|---|---|
| `python -m pytest tests/agents/ifvg_search -q` (baseline, inherited drafts, pre-work) | 137 passed | 2.7s |
| `python -m pytest tests/agents/ifvg_search tests/agents/test_ifvg_verifier_tab.py -q` (pre-adversarial checkpoint) | 196 passed | 5.9s |
| same (final, post-adversarial fixes) | **216 passed** (137 R1 + 72 R2 lane + 7 verifier-tab incl. 2 new), 4 warnings | 6.7s |
| `python -m pytest -q` (full repo, pre-adversarial checkpoint) | 1279 passed, 7 warnings, exit 0 | 582.0s |
| `python -m pytest -q` (full repo, FINAL post-adversarial fixes) | **1299 passed**, 7 warnings, exit 0 (`R2_PYTEST_FULL.txt`) | 439.7s (7:19) |
| `python -m ruff check src tests scripts/ifvg_search_job.py scripts/ifvg_verifier_tab.py` (final) | All checks passed | <10s |
| `git diff --check` | clean | <1s |

Warnings (7): the pre-existing sklearn/scipy deprecation and the pre-existing
`assemble_fsm_audit_tables` pandas-concat FutureWarning — the latter already
fires in the pre-existing `test_ifvg_context_experiment_engine` and
`test_ifvg_fsm_audit_contracts` suites and is merely also exercised by the new
per-child audit test. No warning source was introduced by R2 code.

## New R2 suites → gate/test-matrix mapping

| Suite | Rows proven |
|---|---|
| `test_orchestrator.py` (9) | §3.5 synthetic 2×2 E2E: 4 enumerated (deduped; a colliding resolver dedupes to 1), generated-profile capability enforced BEFORE replay (P0-D — blocked children never reach the runner), gates evaluated with human explanations (INSUFFICIENT_TRADES etc.), frontier + Development Exploratory Representative + tie-break trace, 7-category insights, safe cancel at the child boundary (completed children immutable), lock contention refusal, stale-heartbeat break + resume-after-kill with completed-child reuse, cross-study reuse (§3.8 row: second charter, zero replay invocations, "verified reuse" explanations), child-count ceiling, atomic state-file validity |
| `test_lineage.py` (7) | §3.2 all four rows: native determinism (real reducer, Jaccard 1.0/kind), lineage validity (doc-default vs timeout-480 profile over the same synthetic market: native ids disjoint, setup lineage keys IDENTICAL), match basis (native/cross-profile/not_comparable + gap-axis disabling), one-to-one uniqueness + collision refusal with persisted records (P1-E; no dedupe/keep-first), no-fuzzy-code source scan, fvg-id parser refusal |
| `test_child_audit_companion.py` (7) | DEV-R1-6 seam closed synthetically on the REAL machinery: dual-drive on-disk chain → trace/audit-channel retention (DEV-R2-1), per-child audit assembly + exact funnel⇔audit reconciliation + coverage, neutrality gating refusals (absent/failed/mismatched-id/undressed capture), immutable publish + verified reuse, `build_slice_companions` closes invariants/published/verifier-link gates + idempotent re-run, repository-states requirement, vacuous-zero-targets honesty + orphaned-audit-setup failure |
| `test_population_funnel_deltas.py` (4) | DT §4.3 exact set relationships + first divergence (earliest day/cursor; registered-vocabulary attribution, `unattributed` never guessed), Jaccard-1 identity, DT §4.4 union vocabulary (zero-filled, flagged, never dropped), conditional conversions, terminal-reason deltas |
| `test_contrasts.py` (6) | DT §5: declared-only (post-hoc refused), fully-crossed pairing + recorded orientation, seed-7 paired bootstrap CI (`paired_cell_bootstrap_10000_seed7_v1`) deterministic, grid-hole/conditioning/missing-metric refusals, typed-None CI below two pairs, `causal_language_permitted=False` |
| `test_gates_frontier_robustness_insights.py` (6) | CS §3.4 all ELEVEN thresholds with explanations (incl. `require_bootstrap_ci_excludes_zero` over the trading-day cluster bootstrap; missing-CI fails closed when required), typed first-failure reasons, CS §12 frontier dominance/champions/persisted trace + exploratory-representative wording + missing-objective refusal, robustness neighbors/plateau/knife-edge + schema-reserved `outer_fold_recurrence`, seven insight categories always render + not_comparable suppression + forbidden-wording structural refusal |
| `test_cohort_interpretation.py` (6) | P0-10: the three modes map 1:1 (no fourth), descriptive/model cohorts can never select execution/prop delta families, `sequential_strategy_profile` constructs a canonical child profile from registered values (never a filter; cohort schema carries no override surface), cohort identity stability/sensitivity, tampered-envelope refusal |
| `test_search_job_script.py` (7) | IMPLEMENTATION_PLAN §6: import launches nothing, 64-hex id validation before any path, status reads the orchestrator state file (null for unknown), cancel writes the safe-boundary sentinel, worker REFUSES without an explicit runner entry, entry-shape validation, worker completes a synthetic search end-to-end via the injected entry |
| `test_ifvg_verifier_tab.py` (+2) | FUX-DRILL-001 R2 part: `setup_id` is the fourth exact jump kind; nearest/fuzzy kinds structurally refused; setup jumps route to setup mode's exact resolver; candidate jumps untouched by the router |

Already-covered R2 rows (landed in R1, verified still green): strategy-only
cell without v3 (`test_study_cell.py::test_strategy_only_cell_*` — §3.8/§3.9
rows), GeneratedProfileCapability unit rows (`test_identities.py`).

## Adversarial-review round (post-checkpoint additions)

The two-reviewer adversarial pass (1 blocker, 10 distinct majors, 19 minors —
`ADVERSARIAL_REVIEW.md` / `ADVERSARIAL_REVIEW_RESOLUTION.md`) drove 20 more
tests: `test_setup_drill_through.py` (5 — the B-M2 provider contract + the
B-M4 end-to-end drill chain), neutrality-blocks-publication ×3 (worker raise,
slice refusal with empty-store proof, orchestrator failed child never
publishes/reuses), repeat-identical gates+frontier across full reuse,
failed-replay typed/sanitized child, hand-built blocked-value charter refusal
at enumeration, warmup-stamp disagreement refusal, cross-profile native-basis
refusal, lineage-uniqueness persistence round-trip, registry-derived lineage
validity, charter-identity contrast declarations, typed
interaction-unavailable refusal, duplicate-grid-position refusal, max-gate
fail-closed-when-unmeasurable, and the cross-baseline canonical-name adoption.

## Defects found and fixed by these tests (see FILES_TOUCHED for the diffs)

1. **Canonical naming defect (blocker-class)**: derived child sections kept
   their baseline's registered `profile_name`, so no generated child was ever
   renamed and every challenger was blocked as non-canonical
   (`canonicalize_section` content-equality guard added; CS §1.3).
2. Missing eleventh gate (`require_bootstrap_ci_excludes_zero`) — silently
   absent from the report.
3. Lineage candidate entry-FVG column name (`entry_fvg_fvg_id` did not exist;
   real column `fvg_fvg_id`) and empty-string setup ids from dropped taps
   poisoning setup-identity completeness.
4. Audit-channel frames from the v2 capture path lacked the contract-required
   stamps (`is_warmup` etc.) — the per-child audit build failed validation.
5. Insights panel dropped the Evidence Quality category when no deltas were
   attached (seven fixed categories must always render).
6. Orchestrator lock had no orphan-break → resume-after-kill impossible
   (checkpoint heartbeat + provably-stale break added).
