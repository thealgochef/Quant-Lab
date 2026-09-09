# UI-1 — Deviations, engineering decisions and open items

Scope: Phase 1 of `../IMPLEMENTATION_PLAN.md` (revision 2). Every item below stays inside the
plan's Phase 1 scope; none changes a backend contract, a scientific identity, Strategy-Core, the
fixed M0–M3 lane or an immutable artifact. Nothing here is an owner ruling.

## Engineering decisions (inside scope)

1. **Synthetic fixture = the evidence class of Implementation Verification.** The plan names
   three purposes (owner Q1) and confines the `SyntheticAuthorizationMarker` to "fully synthetic
   fixtures". UI-1 models the fixture as `EvidenceClass.SYNTHETIC_FIXTURE` under
   `RunPurpose.IMPLEMENTATION_VERIFICATION` (the `test` namespace), selectable on the Validation
   step, rather than as a fourth purpose. Consequence: the R4-era synthetic search in the
   verification namespace (every existing AppTest and fixture) remains reachable, always under the
   VERIFICATION-ONLY badge; the real ≤5-day slice is the `real` evidence class and refuses the
   marker; research purposes refuse the synthetic class outright.
2. **Where the satisfiability rules live.** The presentation report
   (`presentation/charter_satisfiability.py`) applies every rule to every UI draft and disables
   Freeze with the reason. `validate_charter` (the authority) enforces the identity-bearing subset
   fail-closed: FSM search ≥ 2 profiles, single-configuration ≤ 2, Universal ≥ 2 firms, and — for
   REAL charters only — a prop objective requires an authorized firm contract. The real-only
   restriction of the last rule keeps every existing synthetic fixture lawful (the pipeline and
   orchestrator fixtures carry prop objectives with firms injected through the wiring, never
   through the charter's authorized ids); the UI report blocks the same case for synthetic drafts
   before freeze, so no UI charter freezes with a prop objective and zero contracts.
3. **Root cause of the silent objective rewrite recorded.** `hasattr(StrategyMetrics, name)` is
   always `False` for a pydantic-v2 field, so the R4 filter rewrote EVERY strategy-only objective
   to `net_expectancy_r`. `STRATEGY_METRIC_NAMES = frozenset(StrategyMetrics.model_fields)` is the
   corrected membership test (presentation report and validator).
4. **A fifth research question.** Owner Q4 separates *Evaluate one configuration* from *Compare
   one configuration with the baseline*; both use the single-configuration family. The question
   list (`RESEARCH_QUESTIONS`) gains `evaluate_one_configuration` (first), the compatibility map
   admits it for `single_configuration` (and therefore for `full_pipeline_run`), and the contract
   §8.1 is amended. The R4 shape test now asserts five questions.
5. **Purpose persistence.** `StudyDraft.purpose_annotation` is an additive field (legacy drafts
   load unchanged; `load_draft` filters unknown keys); on freeze the purpose is recorded as an
   additive catalog event kind `purpose` (index field `purpose`). Both are mutable presentation
   records outside every identity.
6. **The real verification charter's authorization.** The charter payload shape is unchanged
   (`OwnerAuthorizationBundle | SyntheticAuthorizationMarker`): a real verification charter
   carries a bundle whose single `21/R-5:verification_fixture_authorization` evidence ref derives
   from the persisted, verified `VerificationRunEnvelope`'s `VerificationAuthorizationRef`
   (`verification_owner_bundle`), bound to that ref's namespace id and head witness. It is
   assembled ONLY from a `ready` readiness.
7. **Readiness typing over the backend reasons.** `supersession_head_witness_mismatch` →
   `stale_head`; `supersession_head_shorter_than_witness` / chain-structure refusals →
   `wrong_head`; decision-level refusals (`supersession_decision_unverifiable`,
   `supersession_transition_unlawful`, `supersession_chain_divergent`) → `superseded`;
   identity / class mismatch → `wrong_namespace`; missing envelope → `store_unmarked`; corrupt
   → `store_corrupt`; incoherent deployment → `store_incoherent`; a lookup that raises →
   `unavailable` (sanitized). Owner-bundle readiness for research purposes derives `missing`
   with the exact decision keys today (only regime decision artifacts exist in the store) — the
   truthful state; no path or marker infers readiness.
8. **Honest launch wait.** After the detached spawn the handlers poll for the worker's persisted
   state up to `LAUNCH_STATE_WAIT_SECONDS = 10` (0.2 s polls; tests shorten it) and otherwise
   render `launch_not_started` with the job-log location and the CLI fallback — never a success.
   The registered executor is resolved through `resolve_registered_runner_entry` BEFORE any
   spawn; an unregistered key (the synthetic fixture key outside the development checkout)
   renders `runner_unavailable` and spawns nothing.
9. **Namespace- and state-bound publication.** `run_publication_gates` records
   `gates_store_namespace_id` (`None` for an unmarked store; a corrupt namespace is a typed
   `PublicationError`) and `gates_state_sha256` (stages / attempts / children);
   `activate_pipeline_result(..., expected_store_namespace_id=None)` refuses a request under
   another namespace, gates recorded under another namespace, or a changed state digest — before
   the scope / charter / catalog steps. The UI cache key is
   `gate_results_<pipeline>_<namespace|unmarked>_<digest16>` and activation is disabled while the
   run's store namespace is not verified.
10. **Per-run store resolution.** Listings locate each run's store by EXACT charter id across the
    deployed stores (the default read root first, then research, then test) and annotate the
    run with that store, its verified namespace class (`None` when unmarked — never a path
    guess) and the artifact scope (`artifact_scope_for_charter`). The default read root remains
    the research root for listings / catalog annotations only; where a charter freezes or
    launches is decided by `roots_for_purpose` alone.
11. **Direction colorscale.** `direction_colorscale(metric_key)` returns `RdYlGn_r` +
    "lower is better" for a registered `minimize` metric, `RdYlGn` + "higher is better" for
    `maximize`, and the neutral scale + "direction unregistered" for an unknown key (never a
    silent maximize). Every displayed heatmap / firm-matrix metric is registered (tested).
12. **Reconciliation `passed`.** The producer folds the evaluated `passed` flags of the pair's
    persisted reports: `True` / `False` / `None` (JSON `null` when nothing was evaluated), with
    additive `evaluated`, `evaluated_gate_count`, `gate_evaluations` and `unevaluated_reports`
    keys; the adapter exposes `status` (`pass` / `fail` / `unavailable`) per gate and for the
    roll-up, and marks the `protected_*` access counters informational (policy-enforced zeros
    written before any path is constructed — not measured evidence). The lab tab renders green
    only for an evaluated pass. `data_access.py` (fixed lane) is untouched.
13. **Worker truth.** No worker control exists on Validation, Configure or Resume / Retry; the
    wizard persists `worker_limit = 1`, the pipeline commands pass `--max-workers 1`
    (`SUPPORTED_CHILD_WORKERS`), `validate_validation_step` refuses any other value, and Monitor
    states the attempt receipt's `execution_mode` / `effective_workers`.
14. **Development dates by the backend contract.** The frozen ten-date warmup prefix is
    read-only; evidence dates are validated field-by-field with `is_logical_trading_day` inside
    `2026-01-13 … 2026-06-10` (`development_evidence_day_error`); the charter's `replay_dates`
    are `FROZEN_WARMUP_DATES + evidence` and `warmup_dates = FROZEN_WARMUP_DATES`. The physical
    partition / logical-day mapping of a REAL verification window (Phase 3 contracts) is
    consumed by UI-2's Verification Center.

## Deferred to later phases (by the plan, not by omission)

- Session-only drafts with Saved / Saving / Unsaved, archive / restore / typed delete, the
  bulk archive of empty untitled drafts (owner Q2) — UI-2. `Start new draft` and the Start cards
  still persist the draft file immediately (the R4 behavior); `discard_draft` is unchanged
  until UI-2 hard-deprecates it.
- The complete Verification Center flow (logical-day shortlist → seed-production packet / ref /
  job / verified seed → final verification packet / ref → review / run → filtered monitor) —
  UI-2. UI-1's `Verify Implementation` route renders the verified-namespace state, the TYPED
  `VerificationAuthorizationRef` readiness, the real-slice requirement and the verification-draft
  entry, and names the CLI seams.
- Goal-conditional flows (skipped steps listed with reasons) — UI-2; the eight fixed steps
  remain, with the goal card and the satisfiability report on top.
- The metric registry, roll-ups, helper / glossary registries, `detail_levels` naming and the
  MBP-1 / regime summary-first panels — UI-3 (§5.1 of the contract is amended then).
- The Replay / Verifier redesign — UI-4; Data & Audit — UI-5; the state-driven lifecycle
  replacing the phase radio, `column_config`, and the mandatory browser / keyboard / viewport
  acceptance — UI-6. **No browser evidence was captured in UI-1**; the plan requires it at UI-6 and
  UI acceptance remains OPEN until then (the code may be called complete for Phase 1, never
  accepted).

## Gate-run note (recorded, not a code deviation)

The first complete-suite run (`_final_pytest_run1_edited_tree.txt`, 2026-09-04 06:33Z → 07:01Z,
2,362 passed / 1 failed) failed exactly one test,
`tests/agents/test_ifvg_study_scans.py::test_capability_fallbacks_preserve_semantics`, with
`NameError: name '_app_no_fragment' is not defined` inside the generated AppTest script. Cause:
one line of that test module (the retired `"NAMESPACE_KEY"` allowance in the session-prefix
scan) was removed at 06:34Z, AFTER the run had imported the module; `AppTest.from_function`
reads the function source from disk at execution time (`inspect.getsourcelines`), so the shifted
line numbers dropped the `def` line and the generated script called an undefined name. The test
passes on the final tree (the scans file was rerun twice after the edit: 14 passed). Because the
plan requires the complete suite as ONE invocation over the final tree, the suite was rerun
untouched from 07:02Z (`_final_pytest.txt`, `junit_full.xml`); no tracked file was edited during
the rerun. Lesson recorded: never edit a tracked source or test file while the detached suite
is running.

## Not performed (by design)

No owner artifact was signed, no seed produced, no real verification run, no full pipeline, no
push, no merge; `data/` gained no file from this release; the user's pre-existing worktree hunks
(`ARCHITECTURE.md`, `docs/README.md`, `docs/pipeline_state.yaml`, `docs/ML_TRAINING_WORKBENCH.md`,
the two `data/experiment/honest_edge/*.json` deletions) were never staged or committed — the three
shared docs are staged as HEAD + the UI-1 lane transforms through `stage_shared_docs.py`.
