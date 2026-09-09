# R4 — Gate Summary

**Release:** R4 — Trader UI: wizard, active runs, results, history, and
account timeline
**implementation_status: complete**
**acceptance_status: transitively_blocked_by_R1** (authoring-vs-acceptance
model, V3 P0-8)

> Per the kickoff commit-discipline rule: **implementation complete;
> acceptance blocked pending VerificationAuthorizationRef.** The owner's
> fixture sign-off (decisions 21/R-5) remains the FIRST blocker for every
> release: R4 cannot be declared accepted until R1's acceptance — including
> the owner-approved coverage matrix, allowlist, seed snapshot, and signed
> `VerificationAuthorizationRef` — passes.

## Commit

- **Hash:** `6b820ad` (branch
  `feature/ifvg-prop-robust-config-search-v1`; parent `a23893c` = R3)
- 36 files changed, +11,663 / −68.
- **Message:** `R4: add trader workspace UI — wizard, monitor, results,
  history, account timeline` (+ the acceptance-blocked statement)
- Not pushed. Not merged.
- Shared-doc staging: `ARCHITECTURE.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` staged as HEAD + R4 lane transforms ONLY
  (`R4/stage_shared_docs.py`); the user's pre-existing uncommitted hunks
  remain in the worktree (the R4 lane transforms were applied to the
  worktree copies post-commit so the surviving diff is user hunks only),
  byte-verified post-commit against `../R1/PRE_EXISTING_DIFF.patch`
  (byte-exact line comparison IDENTICAL across all four user-owned files).
  `docs/ML_TRAINING_WORKBENCH.md` untouched and uncommitted. `docs/DECISIONS.md` gains D-045's
  trader-workspace half (not user-dirty; committed normally).

## Repo files touched

See `FILES_TOUCHED.md` (5 new + 2 modified src modules, 7 new + 2 modified
scripts, pyproject/ci, 12 new + 2 modified test files, docs).

## Implementation-progress files produced

`PRE_R4_BASELINE.md`, `FILES_TOUCHED.md`, `DEVIATIONS.md` (DEV-R4-1…17 +
scoping notes), `TEST_RESULTS.md`, `ACCESS_SAFETY_EVIDENCE.md`,
`ADVERSARIAL_REVIEW.md` (both reviewer reports),
`ADVERSARIAL_REVIEW_RESOLUTION.md`, `stage_shared_docs.py`,
`r4_smoke_app.py`, `browser-smoke/` (9 live-app screenshots),
`GATE_SUMMARY.md` (this) + `../DECISIONS_TAKEN.md` entries 26–33.

## Exact commands, exit codes, counts, timings

See `TEST_RESULTS.md`. Headline: full repo **1492 passed** (0 failed, exit
0, 6:41; baseline 1372); R4 lane block **338 passed** (9.7s);
`ruff check src tests scripts` clean; `git diff --check` clean.

## Gate status (PHASED_DELIVERY R4 bullets → evidence)

| Gate item | Status |
|---|---|
| Information architecture, route isolation, Context Research delegation, session-state namespaces | **PASS** — FUX-IA-001..003 AppTests (exact 5-route radio; single-route execution spied; M0–M3 delegated verbatim incl. the default-delegate test); f-string-aware namespace scan; live browser smoke of the shell + delegation over real data |
| All five modes × eight wizard steps: exact fields, templates, classifications, authorization states, drafts/autosave/clone/freeze, replay-vs-prop preview | **PASS** — 21 wizard AppTests (FUX-WIZ-001..012): 5 modes / 4 questions / 6 templates w/ resolved-objective visibility; baseline card + copyable identity + blocked-cannot-advance; exact-step restore + autosave + Back-preserves + Save-Draft-always + cross-draft purge; clone-from-frozen; provider-derived authorization state; read-only allowlist + exact verification badge + ≤5 validation; replay-vs-prop work split + reuse-resolved-at-launch; immutable freeze via `validate_charter`/`save_charter`; typed full-scope confirmation with the exact §15 warning |
| Locked/blocked axes have no widget; raw overrides absent | **PASS** — widget-key absence AppTest over locked/thesis/measured/blocked axes; `section_overrides` absent from the wizard source; text_areas proven to be the two date lists only; enumeration re-gated by `assert_axes_authorized` at launch (R2 defense in depth) |
| Active Runs: polling/manual fallback, phase checklist, keyboard funnel, exact table, row detail, safe cancel, skipped-stage explanations, CLI fallback | **PASS** — FUX-MON-001..004 AppTests: fragment + always-present Refresh (fragment-absent fallback proven semantics-preserving); §16.2 quoted count labels; five exact funnel buttons filtering the table; exact 7 columns + "Not run — strategy gate failed" copy; membership/costed identities + sanitized detail + attempt history; confirm-gated sentinel on a running state; missing-status CLI hatch with zero path disclosure |
| Results/History: exact cards/no-pass copy, scope + gross/cost/net labels, frontier/selectbox twin, heatmap glyph/table twin, firm/survival/payout, indexed explorer, dimension ribbon/four panels, deterministic insights, immutable history, account timeline, exact verifier links | **PASS over R1–R3 fixtures** — FUX-RES-001..009 + FUX-HIST-001 + FUX-DRILL-001 AppTests on the real synthetic-2×2 store (real charter/frontier/costed evaluations + persisted account simulation): four exact overview cards; the exact no-pass sentence (terminal runs only); scope captions DERIVED from persisted simulation modes; frontier with `on_select` + always-present selectbox twin; heatmap ◼▲✕·⊘ + table twin w/ evidence column + honest aggregation; firm matrix toggles; step+dash survival; P10-first payout horizons; explorer presets w/ identity-first columns + true survival values + 64-child pagination; ribbon w/ `match_basis` + suppressed panels for non-comparable pairs; seven verbatim insight categories w/ exact `EvidenceRef` actions; five History sections w/ no delete/overwrite for frozen evidence + catalog-only rename + clone; envelope-ordered timeline w/ FUX marker shapes, dll line, phase transitions, linked exact-id drill-down; unresolved ids sanitized |
| `Development Exploratory Representative`, `match_basis`, truthful path-scenario wording; forbidden-label and raw-path/traceback scans | **PASS** — status/label registries + source scans (forbidden controls, display titles, affirmative "exact historical", fuzzy-fallback code tokens, launch-seam confinement, path/secret sanitizer incl. UNC/quoted-space) |
| R4 AppTests and browser smoke states | **PASS** — 51 AppTests green (FUX-IA/WIZ/MON/RES/DRILL/HIST/STATE/LABEL/A11Y-004/PERF-001 R4 rows per `TEST_RESULTS.md`); 9-state live browser smoke recorded (`browser-smoke/`) |

## Carried-obligation closure (R2/R3 → R4)

1. **`persist_lineage_uniqueness` before cross-profile deltas** — CLOSED:
   `prepare_cross_profile_deltas` is the only constructor and persists both
   sides' reports first (provider-tested, collision → `not_comparable`).
2. **Registry-gated `--runner-entry`** — CLOSED: exact registered values
   only, refused BEFORE import (import-bomb test); the UI passes keys; the
   only pre-R5 entry is the synthetic fixture wiring; `resume` added.
3. **Declared interaction contrasts evaluable** (DEV-R2-5/§7A.19.11) —
   CLOSED: balanced two-way DiD with deterministic orientation, recorded
   quartets/pairs, seed-7 bootstrap; refusals retained.
4. **Real artifact-store manifests** (DEV-R3-9) — store half closed for the
   UI lane (account-simulation sidecars ride manifest-verified envelopes);
   production writers = R5. Scenario-seam bridging (DEV-R3-11) remains R5.

## Adversarial review

Two independent read-only reviewers (contract-fidelity; safety/identity):
**0 blockers; 10 majors + 10 minors (contract lens) and 0 majors + 6 minors
(safety lens) — ALL resolved** in code/tests before this commit
(`ADVERSARIAL_REVIEW.md` + `ADVERSARIAL_REVIEW_RESOLUTION.md`; two
structurally-deferred items upgraded to recorded deviations DEV-R4-16/17
with partial delivery). The safety lens's protected/sealed verdict:
**zero-counter provable at the R4 layer**.

## Open blockers (in order)

1. **Owner fixture authorization** (decisions 21 + R-5) — unchanged from
   R1/R2/R3: blocks R1 acceptance and, transitively, R4's (and every later
   release's) acceptance.
2. Acceptance-time slice requirements carried by R2 evidence (unchanged).
3. R5-completing items recorded in this release: live population/funnel
   delta builds + stored comparison-contract consumption (DEV-R4-16;
   FUX-RES-007 R4-partial), real `AccountPolicySetEnvelope` construction
   (DEV-R4-17), real runner executors (registry has synthetic only),
   insights-store persistence (S14), scenario-seam bridging (DEV-R3-11).
4. Hardening gates OPEN by design: full interactive keyboard-only
   navigation, the four required viewports w/ screenshot evidence
   (1440×900 · 1024×768 · 768×1024 · 390×844), chart/widget fallback
   parity, long-ID/wide-table interactive QA (the R4 browser smoke at
   1440×900 is recorded evidence, not a hardening waiver).

## Protected/sealed counters

**Zero.** See `ACCESS_SAFETY_EVIDENCE.md` — no path for 2026-06-11 or the
sealed range was ever constructed (the sole occurrence is plan-required
read-only display copy); every test ran on synthetic fixtures with
tmp-only writes; the browser smoke created one mutable draft in the draft
namespace and it was removed (data/ byte-status unchanged from baseline);
no real-data source access exists anywhere in the R4 lane; Strategy-Core
and Trade-Lab untouched; M0–M3 byte-behavior unchanged (suite green;
one-hunk plan-authorized delegation only); the evaluation-only propsim API
untouched; `full_pipeline_not_run` holds — no full-development replay,
feature build, model fit, prop search, or operator run occurred.
Corroborated by the safety-lens reviewer's independent grep/execution
audit.
