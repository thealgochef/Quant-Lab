# UI-2 — Deviations, engineering decisions and open items

Scope: Phase 2 of `../IMPLEMENTATION_PLAN.md` (revision 2). Every item below stays inside the
plan's Phase 2 scope; none changes a backend contract, a scientific identity, Strategy-Core, the
fixed M0–M3 lane or an immutable artifact. Nothing here is an owner ruling.

## Engineering decisions (inside scope)

1. **Receipt pickup instead of store listing.** Seed artifacts are never catalogued — the seed
   run's permitted outputs are exactly the seed snapshot, the access audit and the run receipt
   (its own test asserts no catalog event exists after a run). The Verification Center therefore
   picks the results of the external seed steps up from receipt files under the mutable center
   root (`--receipt-out` on `register-authorization` and `run`) and verifies every recorded id by
   EXACT load through the backend's own verifiers on every render. A malformed receipt is
   reported and ignored, never trusted.
2. **The real verification bundle derives from the signed reference.** UI-1's
   `verification_owner_bundle` built the 21/R-5 evidence ref from the persisted run id. A run's
   `pipeline_semantic_id` must name a frozen pipeline spec, which derives from the charter, which
   carries the bundle — a bundle bound to the run id is circular. UI-2 derives the evidence ref
   from the owner's SIGNED `VerificationAuthorizationRef` (its content hash is the decision
   artifact id; `verification_bundle_from_signed_ref`) and `verification_owner_bundle` delegates
   to it, so the wizard's real-slice path and the center produce the same charter identity for
   the same signed reference. No real verification charter existed before this release.
3. **No new spawn site.** The plan's forbidden-controls row requires "no new spawn site"; the
   plan's "launch only after typed preflight passes" is honoured as the EXACT bounded-run command
   shown only after the in-app §6.1 preflight passes. The seed job is likewise an exact external
   command shown only for a verified authorization. The center contains no `subprocess` use (the
   source scan covers it).
4. **`verified_envelope` when the inventory is unavailable.** The complete backend verification
   of a seed-production authorization needs the accepted inventory (and the code identities). When
   the accepted manifest is not available to the workspace the center reports
   `verified_envelope` — namespace, profile and window verified from the persisted envelope, the
   inventory hash and code identities explicitly NOT checked — never `verified`.
5. **The center freezes through the operator surface's own assembly.** The Review / run step
   builds the exact-baseline charter (mirroring the wizard's assembly for the verification scope),
   validates it against the store, persists the pipeline spec through the pipeline tab's
   `_assemble_pipeline_spec` (the same identity the operator surface mints for a verification
   draft) and registers the `VerificationRunEnvelope` through `validate_verification_run` (a
   mismatch persists nothing). `register_program_allowlist` is never called by the UI.
6. **Code identities of the seed packet.** The packet's Quant-Lab and Strategy-Core identities are
   computed over the installed code (read-only repository queries, as the wizard's
   `git rev-parse` already does); the AppTests pin them because the synthetic fixture binds fake
   identities. When they cannot be computed the packet is not built and the CLI command is shown.
7. **The Validation step stays in every research flow.** Plan §5.4 lists the goal-specific steps
   without naming Validation; the development evidence dates, the seed and the computation-path
   authorization checklist have no other home, so every research flow keeps it (retitled per
   goal where useful). The real verification slice keeps it for the allowlist, the seed and the
   typed readiness.
8. **A skipped step contributes nothing.** A stale payload from an earlier goal (for example
   challengers selected before the goal became Evaluate) is ignored by satisfiability, the
   requirement set and charter assembly (`_effective_payload`); it is not deleted from the draft.
9. **The pipeline family refines by purpose.** `goal_for_draft` maps `full_pipeline_run` to the
   advanced end-to-end goal; the flows refine it — Development Research → feature / model evidence
   (S05–S10 over the baseline cohort; bundles, ladder and the optional regime study are
   configured in the pipeline Configure phase reached from Review), Full Authorized Development →
   the advanced end-to-end study (every step).
10. **Draft schema 2 is additive; `discard_draft` retired, not removed.** Schema-1 files load
    unchanged (unknown keys filtered, absent keys default). `discard_draft` stays importable but
    raises and deletes nothing, so no caller can silently keep the R4 hard delete.
11. **One registered delete control.** Owner Q2 authorizes permanent deletion of never-frozen,
    archived drafts with the exact typed name. The forbidden-control scan admits exactly
    `Delete draft permanently` in `ifvg_results_tab.py` and a companion test proves it is bound to
    the typed name, lives in the archived view and that the store refuses frozen drafts.
12. **`Saving` is not observable.** Streamlit's synchronous model persists within one rerun; the
    visible states are `Not saved yet` (session only) / `Saved` / `Autosaved` for drafts and
    `Unsaved` / `Saved` for reviews. Autosave fires on every change of a persisted draft (the
    owner asked for autosave after the first persistence); a legacy draft without a stored step
    key is not "changed" by its first render.
13. **Per-case review widget keys use the full case id.** A 16-character prefix collides for
    ids that differ only in their tail; the widget keys carry the full candidate / setup id.
14. **The coverage matrix derives from the shortlist rows.** No producer persists a coverage
    matrix; the center derives the content-addressed `CoverageMatrixEnvelope` of the selected
    window from the shortlist's own per-day coverage rows (already-authorized evidence; audit
    event counts and replay-chart availability are `None` — never fabricated) and records its id
    in the packet and beside the fixture.
15. **The display-only inventory hash of the final packet.** The packet's expected
    `ReplayInputBundle` source-inventory hash is computed over the window's partition refs and
    labelled display-only; nothing validates it at preflight (equality checks only bind the
    allowlist, seed, coverage-matrix id, namespace and head).

## Deferred to later phases (by the plan, not by omission)

- The metric registry, roll-ups, helper / glossary registries and `detail_levels` — UI-3.
- The Replay / Verifier redesign (case card, grouped controls, dark chart, table twins) — UI-4;
  UI-2 changes the review form only.
- Data & Audit — UI-5.
- The state-driven lifecycle replacing the phase radio, `column_config`, and the mandatory
  browser / keyboard / viewport acceptance — UI-6. **No browser evidence was captured in UI-2**;
  UI acceptance remains OPEN.

## Gate-run note (a real defect found by the complete suite, fixed, and the suite rerun)

The first complete-suite run (`_final_pytest_run1_vocabulary_gap.txt`,
`junit_full_run1_vocabulary_gap.xml`; 2026-09-04 09:04Z → 09:34Z; 2,414 passed / 1 failed) failed
exactly one test, `test_presentation_status_vocabulary.py::test_existing_vocabularies_map_additively_onto_ui_status`:
the seven UI-2 `EmptyStateKey` members had no entry in the presentation vocabulary's additive
empty-state → `UiStatus` mapping (`KeyError`). This was a genuine omission (the mapping test was
not in the targeted set). Fix: the seven keys map onto `UNAVAILABLE` (shortlist), `NOT_SELECTED`
(window), `BLOCKED` (seed missing, final authorization unsigned, preflight refused) and
`INFORMATIONAL` (draft archived, draft session-only); `test_study_status_ui2.py` now asserts every
registered key maps and none maps to PASS. Because the plan requires the complete suite as ONE
invocation over the final tree, the suite was rerun untouched after the fix (`_final_pytest.txt`,
`junit_full.xml`).

## Not performed (by design)

No owner artifact was signed, no seed produced, no real verification run, no full pipeline, no
push, no merge; `data/` gained no file from this release (the center root
`data/ifvg_verification_center` is created only when the owner records a window); the user's
pre-existing worktree hunks (`ARCHITECTURE.md`, `docs/README.md`, `docs/pipeline_state.yaml`,
`docs/ML_TRAINING_WORKBENCH.md`, the two `data/experiment/honest_edge/*.json` deletions) were never
staged or committed — the three shared docs are staged as HEAD + the UI-2 lane transforms through
`stage_shared_docs.py`.
