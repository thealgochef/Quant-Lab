# UI-3 — Deviations, engineering decisions and open items

Scope: Phase 3 of `../IMPLEMENTATION_PLAN.md` (revision 2). Every item below stays inside the
plan's Phase 3 scope; none changes a backend contract, a scientific identity, Strategy-Core, the
fixed M0–M3 lane or an immutable artifact. Nothing here is an owner ruling.

## Engineering decisions (inside scope)

1. **No invented threshold.** Every reference in the metric registry names a source that exists
   in the code: the charter's resolved gate thresholds, the persisted `reference_brier_score`,
   the 0 skill boundary, the 0.5 AUC chance line (a direction only — the plan forbids bands, so
   the reading is INFORMATIONAL), the calibration targets 1 / 0 (distance only), the
   walk-forward `minimum_train_candidates` (30), the block bootstrap's two-cluster rule, the
   0.10 `low_coverage` rule of the coverage report, the persisted capacity `limits`, the fixed
   performance limits of `reporting.py`, the measured `denied_dates` of the access audit and the
   `invariant_audit` violation counts, and the stamped MBP-1 / regime defaults. Two explorer
   columns (`trade_frequency`, `setup_occupancy`) are registered as schema-reserved and render
   unavailable rather than being given a made-up definition.
2. **Policy zeros stay informational.** The plan's "access counters → 0 PASS, nonzero FAIL" rule
   applies to MEASURED counters (denied attempts; forbidden dates / rows); the `protected_*`
   counters are compile-time zeros written before any path exists (UI-1's F-07 ruling), so they
   stay INFORMATIONAL with the caveat. The UI-1 adapter test that pins this is unchanged.
3. **A pass under a proposed threshold is a WARNING.** The charter's gates are
   `proposed_protocol_default` unless owner-ratified, so every gated Results reading carries the
   caveat and a fully passing configuration rolls up as WARNING — the plan's §6.3 rule, applied
   literally rather than softened.
4. **The help scan is exhaustive and the exemptions are live.** The scan covers every widget of
   every UI script (the verifier and lab tabs included, which the older scans exclude because of
   their own prefixes and JSON readers); a call needs `help=` or a registered (script, label)
   exemption with a rationale; a dead registry entry or a stale exemption fails the scan. The
   verifier tab therefore receives registry help now although UI-4 regroups its controls — the
   entries stay keyed by control id and UI-4 may reword them.
5. **Detail levels keep the persisted vocabulary.** `DisclosureLevel` values (`summary` /
   `analyst` / `audit`) stay the session vocabulary; only the labels change (Summary / Research
   details / Technical identity & audit). Stale `Analyst` session values are dropped by
   `sanitize_select` and the Results tests were updated to the new labels.
6. **The frozen lane is untouched.** `context_reporting.py` and `context_statistics.py` are not
   modified (their outputs are persisted with the immutable run). The adapters expose additive
   keys (`calibration`, `oos_count`, `fold_summary`, `access`, `reference_brier_score`, `count`,
   `mean_realized_r`, `winning_trade_count`) read from what the frozen statistics already persist;
   an older stored run without a key renders UNAVAILABLE, never a fabricated value.
7. **The reconciliation banner is green only for an EVALUATED pass.** UI-1 derived `passed` for
   NEW runs but the renderer still showed a success banner for a legacy `passed: True` with no
   evaluated gate. The banner now keys on the persisted derived flag AND `evaluated`; the
   Data-integrity roll-up card beneath it reads the evaluated gate flags, the measured counters
   and the observed-vs-limit figures. An optional figure the report did not measure (a `None`
   performance value; an access audit without `denied_dates`) is omitted from the readings —
   its report's own evaluated flag already covers it — never fabricated as an unavailable
   reading that would drag a fully evaluated section to inconclusive (the first complete-suite
   run caught exactly this through UI-1's banner test; see the gate-run note).
8. **Render tags avoid duplicate widget keys.** `_render_result` renders both the latest run and
   the History selection on one page; the detail-level radio and the selectable tables now carry
   a call-site tag (`latest` / `history`) in their keys, closing a latent duplicate-key failure
   when the same run appears in both places. The 2-argument call the lab-tab test monkeypatches
   is preserved.
9. **Panels instantiate their inputs first, in a lower container.** Streamlit widgets yield
   their values only when instantiated, so the manual exact-id inputs are rendered into the
   Advanced diagnostics expander (created last in layout) before the summary and Research details
   containers are filled; the summary therefore reflects edits to the inputs on the same rerun.
   The two panels' Advanced expanders carry distinct labels so they are addressable in tests.
10. **A summary never duplicates an empty state.** When an exact id fails to load, the Research
    details section renders the single `artifact_unavailable` state; the summary reports the
    fact as an `∅ Unavailable` line (the bogus-id test counts exactly one new empty state).
11. **The ladder frame stays Arrow-safe.** AUC becomes a nullable Float64 column and the
    `auc_reason` token moves to an `AUC reason` string column (the R5-FIX regression test was
    updated to the new dtypes); a `Model` column carries the human protocol label.
12. **Accessible names for collapsed labels.** The wizard's gate number inputs are labelled
    `Resolved value for <gate>` (collapsed) and History's rename input `New display name`; the
    scan requires at least two static words on every collapsed label.
13. **Import placement.** The panel and pipeline scripts import the presentation registries at
    module level next to their existing `ifvg_ui_common` import (they already depend on
    `alpha_lab` transitively through it); the lab and verifier tabs keep their `# noqa: E402`
    convention behind their `sys.path` guard.
14. **One roll-up pair for the Results blocks.** The frontier, heatmap, firm-matrix, survival
    and payout views are views over the same strategy and prop metrics, so the Strategy quality
    and Prop feasibility roll-ups render once in the Selected configuration block (synchronised
    with every view's selection) rather than being repeated as a card above each chart; each
    view keeps its scope label and gains the picked metric's registry definition and direction.

## Deferred to later phases (by the plan, not by omission)

- The Replay / Verifier redesign (case card, grouped controls, PIT scrubber, dark chart, table
  twins) — UI-4; UI-3 adds help entries to its existing controls only.
- Data & Audit health cards and structured reports — UI-5 (the registry rows for its report
  fields exist now).
- The state-driven lifecycle replacing the phase radio, `column_config`, and the mandatory
  browser / keyboard / viewport acceptance — UI-6. **No browser evidence was captured in UI-3**;
  UI acceptance remains OPEN.

## Gate-run note (a real defect found by the complete suite, fixed, and the suite rerun)

The first complete-suite run (`_final_pytest_run1_integrity_rollup.txt`,
`junit_full_run1_integrity_rollup.xml`; 2026-09-04 21:22Z → 21:53Z; 2,499 passed / 1 failed)
failed exactly one test, `test_ifvg_ui1_truthfulness.py::test_reconcile_banner_derives_from_evaluated_gates`
— a UI-1 suite the targeted set had not included. The UI-3 renderer had keyed the reconciliation
banner on the Data-integrity roll-up, and `reconciliation_readings` emitted an UNAVAILABLE
reading for a measured counter the fixture's access audit did not carry (`denied_dates`) — the
same would have happened for any `None` optional performance figure of a real report — so a
two-gate evaluated pass rolled up INCONCLUSIVE (no success banner) and a `passed: False` report
produced a warning instead of an error. This was a genuine defect of the UI-3 change. Fix: the
banner keys on the persisted derived flag AND `evaluated` (UI-1's contract; a legacy
`passed: True` without evaluated gates stays the UNAVAILABLE warning), and an optional figure the
report did not measure is omitted from the readings rather than fabricated (item 7). The UI-1
truthfulness suite joined the targeted set (`_targeted_pytest_run2.txt`: 355 passed). Because the
plan requires the complete suite as ONE invocation over the final tree, the suite was rerun
untouched after the fix (`_final_pytest.txt`, `junit_full.xml`).

## Not performed (by design)

No owner artifact was signed, no seed produced, no real verification run, no full pipeline, no
push, no merge; `data/` gained no file from this release; the user's pre-existing worktree
hunks (`ARCHITECTURE.md`, `docs/README.md`, `docs/pipeline_state.yaml`,
`docs/ML_TRAINING_WORKBENCH.md`, the two `data/experiment/honest_edge/*.json` deletions) were
never staged or committed — the three shared docs are staged as HEAD + the UI-3 lane transforms
through `stage_shared_docs.py`.
