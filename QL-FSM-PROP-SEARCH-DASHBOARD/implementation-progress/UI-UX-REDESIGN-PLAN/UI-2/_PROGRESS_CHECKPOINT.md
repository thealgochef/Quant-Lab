# UI-2 — Progress checkpoint (resume from here)

**Release:** UI-2 — Phase 2 of `../IMPLEMENTATION_PLAN.md` (revision 2): the complete
Verification Center flow, goal-derived conditional New Study flows, safe drafts (session-only
until Save / first valid Next; archive / restore / typed delete; bulk archive of the empty
untitled drafts) and explicit reviewer verdicts (owner Q2 / Q3).
**Authority:** the owner's instruction on 2026-09-04 ("UI-1 phase is completed. Please resume
with the next phase in the UI-UX-REDESIGN") over `../IMPLEMENTATION_PLAN.md` revision 2 (§9
Phase 2), subordinate to `../../../FINAL-IMPLEMENTATION-PLAN-DOCS/`.
**Baseline:** `feature/ifvg-prop-robust-config-search-v1` @ `f1827e8` (UI-1 release head;
verified: `UI-1.patch` / `UI-1.bundle` sha256 OK, `git bundle verify` OK).
**Rules carried:** one release-scoped commit per phase; never push / merge; the user's
pre-existing hunks in `ARCHITECTURE.md`, `docs/README.md`, `docs/pipeline_state.yaml`,
`docs/ML_TRAINING_WORKBENCH.md` and the two `data/experiment/honest_edge/*.json` deletions never
enter a commit (stage release files by path; shared docs through `stage_shared_docs.py` in this
folder); acceptance uses isolated synthetic / test stores only — no owner signing, no seed
production, no real verification run; no new spawn site (the center exposes exact external
commands and refresh semantics); never edit a tracked file while the detached suite runs.

## Design decisions fixed before coding (see `DEVIATIONS.md` when written)

1. Seed artifacts are never catalogued (the seed run must write only its permitted outputs), so
   the center picks up external results through receipt files under a mutable
   `verification_center` root (`--receipt-out` on the seed CLI) plus exact-id entry as the
   Advanced fallback; every recorded id is exact-loaded and verified on render.
2. The real verification charter's owner bundle derives its 21/R-5 evidence ref from the
   owner's SIGNED `VerificationAuthorizationRef` (its content hash), never from the run
   envelope id — otherwise the run's `pipeline_semantic_id` would depend on a charter that
   depends on the run (circular). The center freezes charter + pipeline spec from the signed
   ref, then registers the `VerificationRunEnvelope` naming that spec.
3. The center never spawns: the seed job and the bounded run are exact external commands shown
   only when their typed preconditions hold (a verified authorization; a passed preflight).
4. Conditional flows keep the Validation step in every research flow (evidence dates, seed and
   the authorization checklist are entered there); a step skipped by the flow contributes nothing
   to satisfiability or charter assembly.

## Workstreams (Phase 2 scope, §9 of the plan)

| # | Workstream | Status | Evidence |
|---|---|---|---|
| A | `presentation/flows.py` (goal-conditional flows; skipped steps with reasons; exact restore) + tests | done | `_red_A_pure.txt` (red); flows 8 passed |
| B | `study_drafts.py` lifecycle: archive / restore / typed permanent delete (never-frozen only), bulk archive of empty untitled drafts, `discard_draft` hard-deprecated, duplicate detection, proposed names + tests | done | `_red_A_pure.txt` (red); drafts 13 passed |
| C | `presentation/review_vocabulary.py` + additive `not_applicable` in `visual_review_store` + tests | done | `_red_A_pure.txt` (red); vocabulary 2 + store 5 passed |
| D | `study_status.py`: the additive UI-2 §31 states + tests | done | `_red_A_pure.txt` (red); status 9 passed |
| E | `study_providers.py`: shortlist loader, center record, seed authorization / snapshot / receipt states, signed-ref validation, coverage matrix from the shortlist window, bundle from the signed ref + tests | done | `_red_E_providers.txt` (red); providers ui2 10 + ui1 4 + base 11 + seams 13 passed |
| F | `scripts/ifvg_seed_production.py`: `register-authorization` + `--receipt-out` + tests | done | CLI 2 + seed 11 passed |
| G | `ifvg_study_wizard.py`: session-only drafts, Saved / Unsaved chip, autosave, name required, duplicate warning, conditional flows, prop objective blocks + AppTests | done | `_red_G_wizard.txt` (10 red); wizard 39 + study tab 9 + pipeline tab 40 passed |
| H | `ifvg_results_tab.render_history`: archive / restore / typed delete, bulk archive, purpose / store read-only filters, run archive flag + AppTests | done | `_red_HI_history_review.txt` (red); results tab 16 passed |
| I | `ifvg_verifier_tab.py`: Unreviewed sentinel, per-case widgets, owner-approved labels + definitions, explicit Save Review, Unsaved / Saved chip + AppTests | done | `_red_HI_history_review.txt` (red); review 4 + setup verifier 8 + verifier 7 passed |
| J | `scripts/ifvg_verification_center.py` (new): readiness card → fixture (logical days vs physical partitions) → seed (packet / registration receipt / job command / verified seed) → final authorization (packet / signed ref validation) → review / run (freeze + register run; preflight; exact bounded-run command) → monitor (planned stages only; no Publish) + AppTests | done | center 6 passed (through the study tab route) |
| K | scans: new script in `_UI_SCRIPTS`, the owner-Q2 delete allowance with typed-name proof, no new spawn site | done | scans 16 passed (the owner-Q2 delete allowance proven typed; the center scanned; no new spawn site) |
| L | Docs: contract §§3.2, 7, 11, 14, 27, 29, 31, 35, 36; `docs/DECISIONS.md` D-054; `DECISIONS_TAKEN.md` #137; shared docs via `stage_shared_docs.py` | done | contract §§3.2, 7, 11, 14.4, 27, 29, 31, 35, 36 amended; D-054; #137; `stage_shared_docs.py` dry-run validated against HEAD anchors; `DEVIATIONS.md`, `FILES_TOUCHED.md` |
| M | Gates: targeted suites; Ruff; `git diff --check`; complete suite as ONE detached invocation over the frozen tree; commit `UI-2: complete Verification Center, goal-derived flows, safe drafts and explicit reviews`; patch / bundle + sha256; `TEST_RESULTS.md`, `FILES_TOUCHED.md`, `DEVIATIONS.md`, `COMPLETION_REPORT.md` | done | `_targeted_pytest.txt` (373 + 56), `_ruff_and_diffcheck.txt`, `_final_pytest.txt` (2,416 passed, one invocation over the final tree), `junit_full.xml`, commit `eb85d09`, `UI-2.patch(.sha256)`, `UI-2.bundle(.sha256)`, `TEST_RESULTS.md`, `FILES_TOUCHED.md`, `DEVIATIONS.md`, `COMPLETION_REPORT.md` |

## Log

- 2026-09-04 — Resume: UI-1 artifacts verified; plan §9 Phase 2 read in full; reconnaissance of
  the backend seed / window / verification / bounded-verification contracts and CLIs, the
  UI-1 wizard / study tab / providers, drafts, History, the verifier review renderers and every
  affected suite complete. Folder opened; design decisions 1–4 fixed above.
- 2026-09-04 — A–E implemented tests-first (red logs kept): `presentation/flows.py`, the
  `study_drafts.py` lifecycle (schema 2; `discard_draft` retired), `presentation/review_vocabulary.py`
  + the additive `not_applicable` key, seven UI-2 §31 states, the Verification Center providers
  (`verification_bundle_from_signed_ref` makes the real verification bundle run-independent;
  `verification_owner_bundle` delegates to it). Shared test fixture:
  `tests/agents/ifvg_search/verification_center_fixture.py`.
- 2026-09-04 — F–K implemented tests-first: the seed CLI seams (`register-authorization`,
  `--receipt-out`); the wizard (session-only drafts, Saved / Autosaved chip, required name,
  duplicate warning, goal-derived flows with effective payloads, the prop objective that blocks,
  flow-aware benchmarks, archived-draft state; the Start cards stash session drafts); History
  (archive / restore / typed delete, bulk archive, read-only filters, run archive flag); the
  verifier review form (Unreviewed sentinel, per-case keys, owner labels + definitions, explicit
  Save Review, Unsaved / Saved chip); the Verification Center script (six sections; no spawn;
  external commands + receipt pickup; charter frozen from the signed ref; run registration;
  preflight; monitor of planned stages only); the scans. Consolidated: **284 passed** over the
  26 touched suites. **Next on resume:** Ruff / diff-check → docs (contract §§3.2, 7, 11, 14.4,
  27, 29, 31, 35, 36; D-054; #137; `stage_shared_docs.py`; DEVIATIONS) → the complete suite
  detached over the frozen tree → commit → patch / bundle → reports.
- 2026-09-04 09:04Z — Ruff clean, diff-check clean (`_ruff_and_diffcheck.txt`); targeted suites **373
  passed** (`_targeted_pytest.txt`, 33 files). Docs written (contract, D-054, #137, DEVIATIONS,
  FILES_TOUCHED, `stage_shared_docs.py`). Tree FROZEN; the complete suite launched detached
  (`run_final_gates.sh` → `_final_pytest.txt`, `junit_full.xml`, `gates.done`).
  **Next on resume:** if `gates.done` exists, read `_final_pytest.txt` (expect `exit=0`); then
  `python stage_shared_docs.py`, `git add` the release files by path (never the user hunks),
  commit `UI-2: complete Verification Center, goal-derived flows, safe drafts and explicit
  reviews`, `python stage_shared_docs.py --apply-worktree`, write `TEST_RESULTS.md` and
  `COMPLETION_REPORT.md`, `git format-patch -1` + bundle + sha256.
- 2026-09-04 09:36Z — Complete-suite run 1 (09:04Z → 09:34Z): **2,414 passed / 1 failed** —
  `test_existing_vocabularies_map_additively_onto_ui_status`: the seven UI-2 empty states had no
  entry in `status_vocabulary._EMPTY_STATE` (a real omission; logs kept as
  `_final_pytest_run1_vocabulary_gap.txt` / `junit_full_run1_vocabulary_gap.xml`). Fixed (the
  seven keys mapped; `test_study_status_ui2.py` now asserts every key maps and none is PASS);
  56 presentation / status tests green; Ruff / diff-check clean over the final tree. Tree FROZEN
  again; the complete suite relaunched detached as ONE invocation over the final tree.
  **Next on resume:** same as above — `gates.done` → `_final_pytest.txt` `exit=0` → stage /
  commit / apply-worktree / reports / patch + bundle.
- 2026-09-04 10:10Z — Complete-suite run 2 over the FINAL tree: **2,416 passed** (one invocation, 29:42,
  exit 0). Release commit `eb85d09da3ab01aee91848962366ddb5c4ab2c9d` (36 files, +6,945 / −458;
  parent `f1827e8`; not pushed, not merged); the shared docs staged as HEAD + the UI-2 transforms
  and replayed onto the worktree (the surviving diff is exactly the user's pre-existing hunks).
  `UI-2.patch` / `UI-2.bundle` + `.sha256` written and verified. **UI-2 closed.** Next release:
  UI-3 (the metric / help / roll-up registries, Context Research and Results presentation, the
  MBP-1 / regime summary-first panels) — start from `../IMPLEMENTATION_PLAN.md` §9 Phase 3 over
  `eb85d09`.
