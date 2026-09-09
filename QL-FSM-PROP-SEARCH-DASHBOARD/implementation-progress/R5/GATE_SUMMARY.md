# R5 — Gate Summary

**Release:** R5 — Pipeline Runner, MBP-1 Contract Readiness, and Supervised
Model Ladder
**implementation_status: complete** (incl. the R5-FIX post-gate round)
**acceptance_status: transitively_blocked_by_R1** (authoring-vs-acceptance
model, V3 P0-8)

> **R5-FIX (2026-08-21/22):** the owner's gate review found the original
> browser smoke neither clean nor final (17 Arrow tracebacks, pre-fix
> screenshots, 52 deprecation warnings) plus 6 substantive defects
> (production registry `tests.*` dependence; marker-only smoke reuse;
> "parity held over 0 OOS rows"; overloaded comparison identity;
> caller-trusted expected seed; stale evidence docs). All are fixed in
> commit **`fb8062f`** and re-evidenced: full repo **1606 passed**, ruff
> clean, and a RE-RUN browser smoke whose server log has **zero
> tracebacks and zero deprecation warnings**, with a capture manifest
> binding commit, pipeline id, viewport, and artifact ids
> (`browser-smoke/MANIFEST.json`). Finding→change map:
> `R5_FIX_REPORT.md`.

> Per the kickoff commit-discipline rule: **implementation complete;
> acceptance blocked pending VerificationAuthorizationRef.** The owner's
> fixture sign-off (decisions 21/R-5) remains the FIRST blocker for every
> release: R5 cannot be declared accepted until R1's acceptance — including
> the owner-approved coverage matrix, allowlist, seed snapshot, and signed
> `VerificationAuthorizationRef` — passes.

## Commits

- **`dda40c9`** (branch `feature/ifvg-prop-robust-config-search-v1`;
  parent `6b820ad` = R4) — 50 files changed, +9,586 / −133 (= 27 added +
  23 modified; category reconciliation in `FILES_TOUCHED.md`).
  **Message:** `R5: add pipeline runner, supervised ladder, and MBP-1
  readiness contracts` (+ the acceptance-blocked statement)
- **`fb8062f`** (R5-FIX; parent `dda40c9`) — 20 files changed,
  +714 / −171 (raw stat: `_r5fix_diff_stat.txt`). **Message:** `R5-FIX:
  registry purity, artifact-verified seed, typed comparison subject,
  Arrow-safe tables`
- Not pushed. Not merged.
- Post-commit worktree replay + byte-verify: the surviving diff on the
  four user-owned files equals `../R1/PRE_EXISTING_DIFF.patch` exactly
  (74 + 5 + 44 + 28 diff lines, byte-identical — compared in bytes per
  the R4 cp1252 lesson).
- Shared-doc staging: `ARCHITECTURE.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` staged as HEAD + R5 lane transforms ONLY
  (`R5/stage_shared_docs.py`; post-commit `--apply-worktree` replays the
  same transforms onto the worktree copies so the surviving diff is the
  user's pre-existing hunks only, byte-verified against
  `../R1/PRE_EXISTING_DIFF.patch`). `docs/ML_TRAINING_WORKBENCH.md`
  untouched and uncommitted. `docs/DECISIONS.md` gains D-044's
  supervised-ladder half + D-045's pipeline half (not user-dirty;
  committed normally).

## Repo files touched

See `FILES_TOUCHED.md` — reconciled against `git diff --name-status`:
`dda40c9` = 10 new src modules (3 + the 7-module `ifvg/ml/` package),
2 new scripts, 12 modified src + 2 modified scripts, 15 new + 5 modified
test files, 4 docs; `fb8062f` (R5-FIX) = 20 files (5 src, 7 scripts,
7 test files, `pyproject.toml`).

## Implementation-progress files produced

`PRE_R5_BASELINE.md`, `FILES_TOUCHED.md`, `DEVIATIONS.md` (DEV-R5-1…12 +
scoping notes, R5-FIX-brought-forward), `TEST_RESULTS.md`,
`ACCESS_SAFETY_EVIDENCE.md`, `ADVERSARIAL_REVIEW.md` (both reviewer
reports), `ADVERSARIAL_REVIEW_RESOLUTION.md` (+ R5-FIX addendum),
`R5_FIX_REPORT.md`, `stage_shared_docs.py`, `r5_smoke_app.py`
(content-addressed rewrite), `browser-smoke/` (6 post-fix screenshots +
`MANIFEST.json`; pre-fix captures under `superseded-pre-fix-2026-08-21/`),
raw outputs `_r5fix_pytest.txt` / `_r5fix_ruff.txt` /
`_r5fix_diff_stat.txt` / `_smoke_server_r5fix.log`, `GATE_SUMMARY.md`
(this) + `../DECISIONS_TAKEN.md` entries 34–42 and §R5-FIX 43–45.

## Exact commands, exit codes, counts, timings

See `TEST_RESULTS.md`. **Release-final headline (R5-FIX, `fb8062f`):
full repo 1606 passed** (0 failed, exit 0, 7:47; R5 final 1599 + 7 net
new; raw `_r5fix_pytest.txt`); `ruff check src tests scripts` clean (raw
`_r5fix_ruff.txt`); `git diff --check` clean; live browser smoke RE-RUN
after the fixes over a freshly built 16-stage run — **server log has
zero tracebacks and zero deprecation warnings** (`_smoke_server_r5fix.log`),
6 screenshots bound to commit/pipeline/viewport/artifact ids by
`browser-smoke/MANIFEST.json`. (The `dda40c9` record — 1599 passed and
the traceback-bearing pre-fix smoke — is retained above and in
`browser-smoke/superseded-pre-fix-2026-08-21/` as sequence evidence,
NOT as final evidence.)

## Gate status (PHASED_DELIVERY R5 bullets → evidence)

| Gate item | Status |
|---|---|
| Pipeline E2E under `verification_5d` — 16 stages terminal | **PASS (synthetic control flow)** — the module-scoped full-plan E2E runs all 16 stages to terminal states over the synthetic 2×2 charter with the REAL store writers (core replays, memberships, costed evaluations, policy sets, account simulations, frontier, insights, comparison results, stage results, pipeline result); the REAL five-day slice stays blocked on the owner authorization and its stamps say so |
| S11 BLOCKED with the exact reason text | **PASS** — runner short-circuit (no S11 executor exists), the exact sentence verbatim-locked across `ml/decision_policies`, `search/authorization`, and the dimension registry by test, rendered live in the browser smoke |
| Attempt-identity test — different workers, same semantic ids | **PASS** — second attempt with a different `WorkerPolicy` re-derives byte-identical stage-result ids (all stages REUSED — reuse proven by identity), two distinct attempt records; no operational field exists on any hashed payload (statically asserted; `ExecutionAttemptIdentity` deliberately unregistered) |
| Complete §30 Configure/Preview/Launch/Monitor/Resume-Retry/Publish UX + FUX-PIPE-001..006 | **PASS** — 15 AppTests: capability-scoped Configure with visible-disabled planned/blocked entries; Preview counts/estimates/reuse/blockers; launch ONLY in the handler (render/import spies zero); Monitor with all 16 stages glyph+word (incl. `not required`), semantic id, attempt history, §30.4 operational fields, the persisted-comparison panel, and the supervised-ladder panel; Resume/Retry operational-clone semantics; Publish gates-first with verification activation refused. §30.4's "current fold" is DEV-R5-11 (structurally unobservable); the §30.1 wizard/pipeline split is DEV-R5-12 |
| No process launches on render/import/AppTest | **PASS** — AppTest spies + AST scans + the widened source scan (two UI Popen seams + one per shim, all pinned) |
| Verification/full-scope badges and confirmations exact | **PASS** — non-dismissible verification badge AppTest-witnessed; the exact §15 typed full-scope confirmation gates the launch control; full-development ends at the truthful no-registered-executor state |
| Ladder parity on fixture 1 + the real R1 slice's candidate view | **PASS (fixture half) / BLOCKED-honest (real half)** — 48 ML-lane tests (parity, fold-locality incl. the direct perturb-a-test-row leak probe, determinism, refusals, relocation); the real-slice half is DEV-R5-6: it cannot run before the owner's fixture authorization, and the E2E proves the identical ≤5-day safe-failure control-flow shape (0 valid folds, empty-prediction reports, stages terminal) |
| `IFVG_ORDER_FLOW_MBP1_V1` remains `planned`; bundle resolution + UI refuse; no baseline-vs-MBP-1 study constructible | **PASS** — refusals proven at block resolution, bundle-view construction, spec validation (bundles without a feature stage refused), stage-plan readiness (launch-blocking), and the UI (selectors exclude; blocked table visible-disabled) |
| Portable-artifact relocation | **PASS** — copy-dir → checksum-verified reload → re-transform → allclose; byte-drift refusal |
| §3.10 decision-policy/schedule closure | **PASS** — registry key ≠ resolved 64-hex id; schedules non-self-referential + chronological; execution-affecting policies require the owner-ratified `RejectedCandidatePolicy`; tamper refusal on reload |
| §3.9 capability-gated operator readiness | **PASS** — strategy-only plans launchable with zero R5B/R6 dependencies; MBP-1/spectral/GAM plans refuse launch; S11 is `blocked_terminal` and never blocks a launch |

## Carried-obligation closure (R3/R4 → R5)

1. **Real runner executors (DEV-R4-5)** — CLOSED: the registry names the
   baseline-verification search/pipeline executors + the synthetic
   pipeline wiring; the real factories fail closed at CONSTRUCTION
   (fail-before-path, test-witnessed) without the owner's persisted
   verification envelope; full-development charters keep NO entry.
2. **Live population/funnel delta builds + persisted comparison-contract
   consumption (DEV-R4-16)** — CLOSED: S02 persists per-child lineage
   evidence; S14 builds + persists `ComparisonResultEnvelope`s; the
   Monitor consumes them via exact-ID loads (AppTest-witnessed).
3. **Real `AccountPolicySetEnvelope` construction (DEV-R4-17)** — CLOSED:
   S12 persists the resolved envelopes unconditionally; the simulation
   payloads pin the same ids; store-witnessed.
4. **Insights-store persistence (DEV-R4-7 / S14)** — CLOSED:
   `InsightPanelEnvelope`s persist per feasible child and reload-verify.
5. **Scenario-seam bridging (DEV-R3-11)** — CLOSED: additive
   scenario/bootstrap/stress mode bridging with R3-identical defaults;
   ordered-event replay refused at the seam; scenario paths never
   fabricated.

## Adversarial review

Two independent read-only reviewers (contract-fidelity; safety/identity):
**0 blockers; 4 majors + 13 minors (contract lens) and 0 majors + 10
findings (safety lens, top severity MEDIUM) — ALL dispositioned** in
`ADVERSARIAL_REVIEW_RESOLUTION.md`: 18 fixed in code/tests (incl. the
S07 label-identity collision, the comparison-consumption wiring, the
§30.4 monitor fields, the derived `zero_forbidden_counters` gate, and
the truthful synthetic-date stamps), 2 recorded as deviations
(DEV-R5-11/12), 1 deferred to the R5B shim alignment (F4/DEV-R5-10), and
the remainder accepted/refuted with evidence. The safety lens's
protected/sealed verdict: **zero-counter AFFIRMED at the R5 layer**.

**Owner gate review → R5-FIX:** two of those dispositions were later
REJECTED by the owner and superseded in `fb8062f` (see the addendum in
`ADVERSARIAL_REVIEW_RESOLUTION.md`): F3/m-16's caller-supplied expected
seed became the artifact-verified `loaded_seed_snapshot_id_source`
(required for the real scope), and F10(b)'s docstring became the typed
`ComparisonResult.subject` discriminated union. The gate review's seven
further findings (Arrow tracebacks, stale screenshots, `tests.*`
registry dependence, marker-only smoke reuse, zero-row parity copy,
deprecated API, stale docs) are likewise fixed there — map in
`R5_FIX_REPORT.md`.

## Open blockers (in order)

1. **Owner fixture authorization** (decisions 21 + R-5) — unchanged from
   R1–R4: blocks R1 acceptance and, transitively, R5's (and every later
   release's) acceptance.
2. Acceptance-time slice requirements carried by R1/R2 evidence
   (unchanged), now including the real-slice ladder run (DEV-R5-6).
3. R5B-completing items recorded this release: the search-shim
   factory-signature/store-root alignment (DEV-R5-10, safety F4); the
   MBP-1 activation event itself (the owner's 13 R5B deliverables).
4. Hardening gates OPEN by design: full interactive keyboard-only
   navigation, the four required viewports w/ screenshot evidence,
   chart/widget fallback parity, long-ID/wide-table interactive QA (the
   R5 browser smoke at desktop resolution is recorded evidence, not a
   hardening waiver).

## Protected/sealed counters

**Zero.** See `ACCESS_SAFETY_EVIDENCE.md` — no path for 2026-06-11 or the
sealed range was ever constructed; every test ran on synthetic fixtures
with tmp-only writes; the repo's real data namespaces were never written
(verified on disk by the safety reviewer); the real executors registered
this release fail before any source path exists; Strategy-Core and
Trade-Lab untouched; M0–M3 byte-unchanged; the evaluation-only propsim
API untouched; `full_pipeline_not_run` holds (and is now scope-derived).
Corroborated by the safety-lens reviewer's independent grep/on-disk
audit (verdict AFFIRMED).
