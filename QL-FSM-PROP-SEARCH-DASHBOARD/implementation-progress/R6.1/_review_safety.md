# R6.1 — Adversarial Review, Reviewer 2 (SAFETY / ACCESS / IMMUTABILITY lens)

Read-only review of the R6.1 working set relative to HEAD `f3f9ac2` (R5B.1), 2026-08-29.
Authority: `R6.1_IMPLEMENTATION_PLAN.md` rev. 3 (§4, §5 D2/D5/D9/D12/D15, §6.D/E/F/G/K, §11, §12),
the kickoff rules (`IMPLEMENTATION-START/`), the R6 `ACCESS_SAFETY_EVIDENCE.md` precedent.
Every probe below ran on synthetic fixtures under `%TEMP%` only; no real data path was
constructed, listed, stat'ed, opened, or read (the protected day 2026-06-11 and the sealed
range ≥ 2026-06-12 were never touched). Nothing in the repo was modified except this file.

**Verdict: 0 blockers, 2 majors, 4 medium, 6 minors; protected/sealed zero-counter: AFFIRMED**

---

## Findings

### S1 — MAJOR — S05 never binds the materialized panel to the chart it asked the seam for (`replay.artifact_id` vs `chart_id` unchecked); the shipped E2E fixture proves the gap and the test hides it

**Evidence.** `src/alpha_lab/agents/data_infra/ifvg/search/pipeline_regime.py:195-213`:

```python
chart_id = chart_ids[0]
replay = source(chart_id)                                   # :196
envelope, panel, validity = materialize_context_bar_panel(replay, ...)   # :197
...
regime["chart_id"] = chart_id                               # recorded as fact
```

Nothing compares `replay.artifact_id` (what the panel payload binds at
`features/context_bar_panel_materializer.py:521`) with `chart_id` (what S05 records in
`bundle_feature_views.json.__regime_observation__.chart_id`, `pipeline_regime.py:246`).
`PipelineWiring.context_bar_source` is typed `Callable[[str], Any]` (`search/pipeline.py:628`);
the materializer only checks `isinstance(replay, VerifiedReplayChartArtifact)` and rehashes the
bytes of *whatever artifact it was handed*.

The shipped fixture demonstrates it: `tests/agents/ifvg_search/pipeline_fixture.py:211-213`
(`_source` ignores `chart_id` and always returns its own synthetic artifact), and the E2E test
`tests/agents/ifvg_search/test_pipeline_regime.py:245` asserts only
`source.requested == [observation["chart_id"]]` — never that the panel came from that chart.

**Reproduction (probe `scratchpad/probe_panel_chart_binding.py`, synthetic panel pipeline on a tmp root):**

```
S04 output chart ids: ['3b381d7c8aa6', '919794778c2f', 'a847203dc30f', 'bb2d6edd3a29']
S05 record chart_id            : 3b381d7c8aa6
seam was asked for             : ['3b381d7c8aa6']
panel.replay_chart_artifact_id : e82d28a6cecc
BOUND TO THE REQUESTED CHART?  : False
statuses: {'04_...': 'completed', '05_...': 'completed', '15_verify_and_publish': 'completed'}
```

S15 published a run whose S05 sidecar names chart `3b381d7c…` while the persisted panel (and
therefore the protocol's `panel_source_artifact_id`, every fold set, fit, assessment, decision and
report downstream) is bound to `e82d28a6…`.

**Exploit / failure scenario.** A mis-wired or stale `context_bar_source` (wrong base dir, a
loader closure captured over another pair, a v1/v2 mix-up) silently feeds a *verified* but
*wrong* chart into the panel; every downstream identity is internally consistent, so nothing
fails closed, and the stage record asserts a binding that does not exist. This is exactly the
"caller-provided string is NOT evidence" pattern the D2 seam was meant to close — here the
caller-provided *callable's output identity* is trusted.

**Fix.** In `s05_regime_observation`, after `replay = source(chart_id)`:
`if replay.artifact_id != chart_id: raise ValueError("context_bar_source returned artifact X for requested chart Y; refusing")`;
additionally assert `replay.source_pair` equals the run's pair (from the S04 chart builder /
charter) and record `replay_chart_artifact_id` (not the requested id) in the S05 sidecar. Make the
fixture's `_source` load the chart it is asked for (write the synthetic chart under the S04 chart
builder's ids) so the E2E test asserts `panel.payload.replay_chart_artifact_id == chart_id`.

---

### S2 — MAJOR — The owner-decision supersession log is fail-open: a crash between `save` and the log append is never repaired, and deleting one line silently restores a revoked owner decision

**Evidence.** `src/alpha_lab/agents/data_infra/ifvg/search/owner_decisions.py:328-359`:

```python
already = has_envelope(root, OWNER_DECISION_STORE, envelope.owner_decision_artifact_id)   # :348
stored, _reused = save_or_reuse_envelope(root, OWNER_DECISION_STORE, envelope)         # :349
if payload.supersedes is not None and not already:                                     # :350
    _append_supersession(...)                                                          # :351
```

The replacement artifact is published (atomic `os.replace`, `search/store.py:210`) *before* the
log line is appended (`owner_decisions.py:270-290`, a plain `open("a")`). If the process dies in
between, re-running `persist_owner_decision` sees `already=True` and never appends. Conversely
`load_supersession_chain` (`:299-325`) derives the superseded set *only* from the log lines that
exist; `assert_owner_decision_authorizes` (`:397-401`) refuses only ids in that set. The log file
is outside the manifest protocol (no hash chain, no manifest entry, mutable append-only text).

**Reproduction (probe `scratchpad/probe_owner_supersession.py`, synthetic lane on a tmp root):**

```
P1 log exists after crash-simulated save: False
P1 chain after re-persist (expected non-empty if recoverable): ()
P1 RESULT: the SUPERSEDED prior STILL AUTHORIZES (fail-open)
P2 second_prior superseded after proper persist: True
P2 after deleting the line, still superseded?: False (chain load raised nothing; the replacement artifact still exists in the store)
```

P1: replacement saved via the store, then `persist_owner_decision` re-run → chain stays empty →
the prior authorizes `stratification_ready->feature_eligible`. P2: after a *proper* supersession,
removing the one line from `owner_decisions/SUPERSESSIONS.jsonl` un-supersedes the prior;
`load_supersession_chain` raises nothing.

**Exploit / failure scenario.** The owner revokes decision A by ratifying replacement B (B's
`supersedes = A`). A crash, a disk-full, or an operator "cleaning" the jsonl leaves B in the store
and A authoritative: `persist_regime_promotion`, `verify_frozen_authority`,
`activate_regime_context_block` all consult `load_supersession_chain(root)` and accept A. The
tested case (a *forged* line, `test_owner_decisions.py:341-355`) fails closed; the untested
direction (an *omitted* line) fails open. Note also that a `synthetic_test_authorization_v1`
replacement may supersede an `owner_signed` decision in any scope (no provenance check on the
replacement at `:337-347`) — fail-closed direction, but it lets a fixture revoke real authority.

**Fix.** (1) Make the append idempotent and independent of `already`: append when the log does
not already contain `(supersedes, id)`; write the line *before* publishing the replacement (or
in the same temp-dir publication) so a crash leaves a dangling line that the loader already
treats as fail-closed. (2) Bind integrity: hash-chain each line to the previous line's sha256 and
store the head hash in a manifest-listed sidecar (or persist each supersession as its own
immutable envelope under a `supersessions` store keyed by the superseded id — exact-ID lookup,
no scan, no mutable file). (3) Refuse a replacement whose `provenance` is weaker than the prior's.
(4) Add the two probe cases as tests.

---

### S3 — MEDIUM — The CLI's `--run-scope` is a caller-provided string that unlocks synthetic-provenance authority in ANY store root; the persisted promotion records no scope

**Evidence.** `scripts/ifvg_regime_promotion.py:41` (`_RUN_SCOPES` includes `synthetic_fixture`),
`:217` (`--run-scope` free choice), `:175` → `regime_store.persist_regime_promotion(root, envelope, run_scope=run_scope)`
(`ml/regime_store.py:498-561`), which is the only place provenance is checked
(`owner_decisions.py:392-396`). No namespace or marker guard exists on the store root — contrast
`search/charter.py:496-520` (`save_charter` refuses synthetic charters in any root with a
`search` segment that is not `search_test`, P0-4). `RegimePromotionDecision` carries no
`run_scope`, so a synthetic-backed FEATURE_ELIGIBLE decision is indistinguishable from a real
one without re-loading the owner artifact.

**Reproduction (probe P3, tmp root shaped `…/data/ifvg_datasets/search/v1`):**

```
P3 CLI exit: 0  {"status": "persisted", ... "regime_status": "feature_eligible", "run_scope": "synthetic_fixture"}
P3 stored status in a search/v1-shaped root: feature_eligible | payload records run_scope?: False
```

**Exploit / failure scenario.** An operator (or a script) runs `promote --to feature_eligible
--owner-decision-id <synthetic fixture id> --run-scope synthetic_fixture --store-root data/ifvg_datasets/search/v1`
(I did NOT do this — the probe used a tmp mirror). The research store now holds a
FEATURE_ELIGIBLE promotion chained onto the real protocol's decisions. The pipeline's
`verify_frozen_authority` / `activate_regime_context_block` would re-check provenance under the
run's own scope and refuse — but `resolve_report_gate` does not (S6), the Regime Lane promotion
view renders the decision, and `chain` walks it as a lawful ladder step. It is
"a caller-provided string" acting as evidence for the scope decision.

**Fix.** Mirror the P0-4 guard: refuse `--run-scope synthetic_fixture` (and
`persist_regime_promotion(run_scope="synthetic_fixture")`) unless the root is outside the
research namespace or carries a typed `SyntheticAuthorizationMarker` sidecar; and/or verify that
the referenced owner artifact's `provenance == "owner_signed"` whenever the root is in the
research namespace, regardless of the flag. Consider recording the verifying scope in the
promotion payload (identity-bearing, pre-acceptance) so consumers can refuse cheaply.

---

### S4 — MEDIUM — S02 "verified reuse by reproduction" accepts re-derived child tables with NO reproduction check when the costed evaluation for this cost policy does not exist, while the row claims it "reproduced its persisted costed evaluation"

**Evidence.** `src/alpha_lab/agents/data_infra/ifvg/search/pipeline.py:1030-1070`:

```python
result = context.wiring.child_runner(spec=spec_child, core_replay_id=core_replay_id)
row["replay_invocations"] = 1                                            # :1035
persisted = _load_child_evaluation(context.store_root, ...costed_evaluation_id)   # :1036
if persisted is not None:                                                # :1042
    ... compare, RuntimeError on mismatch ...
context.tables_by_child[core_replay_id] = result                         # unconditional
row["explanation"] = ("verified reuse by reproduction: the immutable replay exists; "
    "the child was re-derived ... and reproduced its persisted costed evaluation")   # :1064-1068
```

`_load_child_evaluation` returns `None` whenever `costed_evaluations/<id>` is absent
(`search/orchestrator.py:189-197`). The costed-evaluation id is keyed on `(core_replay_id, cost_policy)`
(`_child_evaluation_envelope`), so ANY cross-pipeline reuse under a different cost policy — or a
replay whose evaluation was never published — takes the `None` branch: the re-derived tables are
adopted as this run's evidence, S03 then *publishes* a fresh evaluation from them
(`pipeline.py:1124-1132`), and the row text asserts a reproduction that never happened.

**Failure scenario.** Pipeline A (cost policy X) persisted `core_replays/R`. Pipeline B (cost
policy Y, regime study with stratified reports) reuses R; the child runner's source data or
engine behaviour drifted; the re-derived tables differ from those that produced R; nothing
detects it; the stratified reports (keyed to R's id) are built over tables R never had, and the
attempt record says "verified reuse by reproduction".

DEV-R6.1-8, DECISIONS_TAKEN #82 and `R6.1/ACCESS_SAFETY_EVIDENCE.md` ("a runner that cannot
reproduce fails the child closed") all state the invariant unconditionally; the code does not.

**Fix.** Fail closed when `persisted is None`: leave the child out of the stratified reports
(`children_skipped[core] = "reused replay has no persisted costed evaluation for this cost policy; reproduction unverifiable"`),
keep `state="reused"`, and do not adopt the tables; or, at minimum, make the explanation and the
deviation say what actually happened. Add a test with a reused replay and no evaluation entry.

---

### S5 — MEDIUM — `persist_regime_promotion` persists `MODEL_FEATURE` through the store API with no activation / controlled-study precondition; only the CLI refuses it

**Evidence.** `ml/regime_contracts.py:506-512` (`PROMOTION_SEQUENCE` ends in `MODEL_FEATURE`),
`:577-588` (`assert_lawful_promotion` needs only `gates_passed` + an `owner_ratification_ref`),
`ml/regime_store.py:564-566` (`_OWNER_EVIDENCE_STATUSES` includes `MODEL_FEATURE`, so the owner
artifact is checked for `feature_eligible->model_feature`, which the synthetic fixture authorizes
by default, `owner_decisions.py:537`). `RegimePromotionDecision` accepts `role=FEATURE_GENERATOR`
with `status=MODEL_FEATURE` (`assert_lawful_role` only checks a *minimum* status). The CLI's refusal
(`scripts/ifvg_regime_promotion.py:36-41,140-141`: "requires the activated IFVG_REGIME_CONTEXT_V1
block and a completed controlled feature study") is a CLI-only rule.

**Reproduction (probe P4):**
`P4 RESULT: MODEL_FEATURE persisted via the store API with no activation/controlled-study precondition: model_feature`

**Failure scenario.** Any code path (a test helper, a notebook, a future UI seam) can mint the
top-of-ladder status the plan reserves for a separate, evidence-bearing step. In V1 nothing
consumes MODEL_FEATURE beyond FEATURE_ELIGIBLE (`verify_frozen_authority` accepts both;
`CLASS_MINIMUM_STATUS` tops out at FEATURE_ELIGIBLE), so the exposure is a governance/consistency
gap rather than an execution gap today — but the kickoff's "no … promoted" rule should be
structural, not a CLI string.

**Fix.** Refuse `MODEL_FEATURE` in `persist_regime_promotion` for V1 (exact CLI text), or require
verified refs to the activated block resolution + the persisted `RegimeControlledStudy` in the
decision payload before persisting it.

---

### S6 — MEDIUM — `resolve_report_gate`'s owner-evidence half never calls `assert_owner_decision_authorizes` (no provenance-vs-scope, supersession, effectivity, or value checks)

**Evidence.** `ml/regime_stratification_gate.py:110-131`: for FEATURE_ELIGIBLE-minimum classes
it loads the owner artifact and compares only `resolved_regime_protocol_id` and
`capability_assessment_id`; it takes no `run_scope` and never consults `load_supersession_chain`.
The module docstring (`:1-11`) claims the gate refuses "an owner artifact that is not the
decision's own ratification reference" — true — but a superseded, expired, or
`synthetic_test_authorization_v1` artifact in a real run passes.

**Reachability.** Through the pipeline it is currently dead: `regime_stratification_service.py:286-295`
records `feature_only` / `cohort_model` as typed refusals and only calls `_gate` for the
descriptive classes (`needs_owner=False`). It is live for any direct caller and for the
hardening candidate that turns modeled classes into report envelopes (DEV-R6.1-10).

**Fix.** Give `resolve_report_gate` a `run_scope` parameter and call
`assert_owner_decision_authorizes(owner, protocol_envelope=..., assessment_envelope=...,
transition=transition_key(prev, status), as_of=decision.decided_at, run_scope=..., supersession_chain=load_supersession_chain(root))`
in the `needs_owner` branch; or delete the owner half until a class needs it.

---

### S7 — MINOR — The UI's pre-launch readiness omits `store_root` / `run_scope`, so a model-bearing plan with absent / superseded / mismatched frozen authority is shown "available" and the Launch handler freezes it, saves the charter + `pipeline_specs` envelopes, and spawns a job that fails at S00

**Evidence.** `scripts/ifvg_pipeline_tab.py:884` `derive_stage_plan_readiness(spec)` and `:982`
`assert_stage_plan_launchable(spec)` — both without `store_root=` / `run_scope=`, so
`regime_study_block_reason(..., store_root=None)` skips `verify_frozen_authority`
(`ml/regime_study.py:452-465`). The pipeline's own S00 passes both
(`search/pipeline.py:903-905`), so the run fails closed before any path — but the plan's
"fails BEFORE any path is constructed … before launch" is met only by the job, not by the UI,
and the immutable store gains a spec/charter entry for an unlaunchable plan.

**Fix.** Pass `store_root=Path(roots["store_root"])` and the effective run scope in both calls
(the store root is already in hand at `:1699`/`:1730`).

---

### S8 — MINOR — Evidence-document honesty: premature "corrected after the adversarial round" claims, a cited test that does not exist, and an inexact "only spawn" statement

**Evidence.** `R6.1/DEVIATIONS.md:3` ("Reconciled after the adversarial round
(`ADVERSARIAL_REVIEW_RESOLUTION.md`)") and `R6.1/ACCESS_SAFETY_EVIDENCE.md:3-5` ("corrected after
the adversarial round; … `ADVERSARIAL_REVIEW.md`") were written before this round; neither file
exists (`ls R6.1/ | grep -i adversarial` → none). `ACCESS_SAFETY_EVIDENCE.md:112` says
`ifvg/context_model.py` is "sha256 golden-tested" — no such test exists
(`grep -rn context_model tests | grep -i "sha256|golden|hashlib"` → no hit); the fact itself
holds by `git status` (untouched). `ACCESS_SAFETY_EVIDENCE.md:66` "the Launch handler stays the
only spawn" — there are two `_spawn_pipeline_job(` call sites (Launch `:1026`, Resume/Retry
`:1659`), both pre-existing at HEAD (`:617`, `:1073`); the accurate statement is "no new spawn
site". `ACCESS_SAFETY_EVIDENCE.md:78-81` and DECISIONS #82 overstate S02 (see S4);
`:52-56` overstates supersession robustness (see S2).

**Fix.** Rewrite the three sentences to what the code does; either add the golden
`context_model.py` hash test or cite `git diff HEAD --stat` instead; fix after S2/S4 land.

---

### S9 — MINOR — `threadpoolctl` is imported at module import time but is not a declared dependency

**Evidence.** `ml/regime_service.py:38` `from threadpoolctl import threadpool_limits`; `pyproject.toml`
declares `scikit-learn>=1.4` but not `threadpoolctl`; it is present only transitively
(`importlib.metadata.requires("scikit-learn")` → `threadpoolctl>=3.1.0`; installed 3.6.0).
DEV-R6.1-4 / PRE_R6_1_BASELINE state this honestly ("a scikit-learn dependency"). A future
sklearn that drops or renames the dependency breaks import of the whole regime lane.

**Fix.** Declare `threadpoolctl>=3.1` in `pyproject.toml` (no new package is installed — it is
already present) or import lazily inside `run_regime_protocol` with a hard failure message.

---

### S10 — MINOR — Supersession lock file leaks on crash (30 s `TimeoutError` forever) and unsanitized exceptions escape the CLI

**Evidence.** `owner_decisions.py:276-290`: `os.open(lock, O_CREAT|O_EXCL)` … `os.unlink` in
`finally` — a kill between `os.open` and the `try` (line 279 → 285) leaves `SUPERSESSIONS.lock`;
every later supersession then waits 30 s and raises `TimeoutError`. `scripts/ifvg_regime_promotion.py:247`
catches only `(PermissionError, ValueError, LookupError)`; `TimeoutError` and the store's
`FileExistsError` (`store.py:156,207`) print a raw traceback (the module docstring promises "a
refusal prints its sanitized reason and exits 2").

**Fix.** Stale-lock detection (age + pid) or an `O_EXCL` lock inside the same `try`; catch
`(OSError, TimeoutError)` in `main()` and print the sanitized first line.

---

### S11 — MINOR — CLI `--decided-at` is a caller string used as the effectivity `as_of` for the owner artifact

**Evidence.** `scripts/ifvg_regime_promotion.py:172` (`decided_at or datetime.now(UTC)`),
`ml/regime_store.py:607` (`as_of=decision.decided_at`), `owner_decisions.py:402-408`. An operator
can back-date a promotion into an expired owner decision's window (or forward-date into a
not-yet-effective one). `decided_at` enters the decision id, so it is at least visible in `chain`.

**Fix.** Require `decided_at >= previous decision's decided_at` (monotone chain) and
`decided_at >= owner.approved_at`; or take `decided_at` from the assessment's evidence as-of
(as S10 does) rather than from the command line.

---

### S12 — MINOR — D15 writer is not streaming (every partition is held in memory up to the 2 GiB budget); wording overstated

**Evidence.** `src/alpha_lab/propsim/event_detail.py:407-422` accumulates every partition's bytes in
`sidecars` before returning; `search_bridge.py:111-122` then hands the whole dict to
`save_or_reuse_envelope`. The budget checks themselves are correct and happen before any
publication (`:377-382` preflight rows; `:414-420` cumulative bytes), and the store publishes
atomically, so "no partial artifact" holds. "Streams … into ZSTD Parquet partitions" describes
the layout, not the memory profile: a run near `max_published_bytes` needs ~2 GiB RAM plus the
Parquet build buffers.

**Fix.** Either write partitions to the temp publication directory as they are produced
(extend `save_envelope_immutable` to accept a producer) or lower the registered V1 byte budget
and say "bounded in-memory" in the docs.

---

## What holds (checks performed)

**(1) Protected/sealed zero-counter — AFFIRMED.**
- `grep -nE '2026-06-11|2026-06-1[2-9]|2026-06-[23][0-9]|2026-0[7-9]-|2026-1[0-2]-|20260611|...'` over all 82
  new/modified `src/`, `scripts/`, `tests/` files: hits are only (a) decision/approval timestamps
  `2026-08-25…29` from which no path is derived, (b) `search/charter.py:65 _PROTECTED_BUFFER_DAY = "2026-06-11"`
  — pre-existing at HEAD (`git show HEAD:…charter.py` line 65; charter.py changed only by the D15
  field at `:245-252`), (c) `identities.py:831` / `test_identities.py:265,290` example partition
  strings dated `2026-06-04` / `2026-06-03` (before the protected day; the identities example is
  pre-existing at HEAD line 823).
- `grep -nE 'data/databento|DEFAULT_DATA_DIR|IfvgCaptureConfig|read_parquet|databento|\.parquet|Path\("data|ifvg_datasets'`
  over the set: `read_parquet` appears only in the panel materializer (`:502`, on the
  artifact-directory path just size+sha256+row verified against the manifest, after a
  directory-escape guard `:490-494`) and in the D15 reader over manifest-verified bytes;
  `store.py:38-39` roots are pre-existing constants; `scripts/ifvg_pipeline_tab.py:41`
  `PIPELINE_STATE_ROOT` pre-existing at HEAD `:41`; the smoke harness writes only under
  `tempfile.gettempdir()/ifvg_r61_smoke/<key>` (`r61_smoke_app.py:53,256-262`, rmtree confined to that base).
- `find data -type f -newermt 2026-08-25` → **0** (before and after my probes).
- No `regime_*`, `context_bar_panels`, `owner_decisions`, `fold_*`, `fold_sets`, `fold_schedules`,
  `bundle_feature_views`, `regime_stratified_reports`, `search`, `search_test`, `*search*/v1`
  directory under `data/`; the same store-name search repo-wide (excluding `.git`) → none.
- Top-level `data/` mtimes: newest 2026-08-21 (`ifvg_datasets` 2026-08-11). Only `ARCHITECTURE.md`
  (user-owned + lane transform) is newer than 2026-08-25 outside `src/tests/scripts/docs/QL-*`.
- My probes wrote only under `%TEMP%\r61_probe_*` / `%TEMP%\r61_panel_probe_*`;
  `find . -type f -mmin -12` afterwards showed only the implementer's concurrently written
  evidence files under `R6.1/`.

**(2) Panel materializer reads only through the verified pair loaders; the seam never accepts a frame.**
`materialize_context_bar_panel` (`context_bar_panel_materializer.py:474-512`): `TypeError` on any
non-`VerifiedReplayChartArtifact` (test-pinned `test_context_bar_panel.py:72`), unique manifest
entry lookup, `bars_tf.parquet` re-read from `replay.directory` after a `relative_to` escape
guard, size + `file_sha256` + row count checked against the manifest entry before interpretation
(tamper test `:74-80`); resample rule re-checked; only a `VerifiedReplayChartArtifact` (dataclass
produced by `load_verified_replay_chart_artifact(_v2)`, `replay_chart_store.py:617-677 / 1103-1163`,
which verify manifest identity, policy set, pair, and every listed file) reaches it. The pipeline
passes only a chart-id string into `context_bar_source` (`pipeline_regime.py:196`); the executor
(`regime_executor.py:65-105`) and observation seam (`regime_observation_source.py:104-148`) take
ids only and derive `source_artifact_ids` from the loaded envelope; `run_regime_protocol` refuses a
panel protocol whose pinned `panel_source_artifact_id` is not among the loaded source ids
(`regime_service.py:483-493`). The residual gap is the chart-id binding (S1), not a frame path.

**(3) Store discipline.** All ten new stores are in `SEARCH_STORE_NAMES` (`store.py:82-98`) and go
through `save_or_reuse_envelope`, whose reuse path asserts byte-equal envelope content AND
manifest-hash-equal sidecars (`store.py:291-336`) — so a same-id publication with different
sidecars fails closed and `none_v0` artifacts cannot be widened. Every new loader rehashes its
sidecar against an envelope-carried hash (`context_bar_panel_materializer.py:604-632`,
`bundle_feature_view.py:139-153`, `fold_set_artifact.py:291-294`, `regime_fold_features.py:778-784`,
`regime_oos_assignment.py:575-592`, `regime_stratification_service.py:119-128`,
`event_detail.py:464-556`). Sidecar names stay within the R5 whitelist (`store.py:117,177`).
`persist_regime_fit` reuse-by-reproduction (`regime_store.py:297-328`): reload through the
manifest-checked store → `_verify_bundle` re-transform `allclose` + re-predict equality →
envelope equality → row-set equality → returns the EXISTING artifact with no write; a tampered or
non-reproducing entry raises. Fresh entries are pre-verified from the exact bytes and withdrawn on
post-publish failure (`:341-361`). `persist_regime_promotion` (`:498-561`) loads the assessment,
re-checks the chain, re-runs `assert_lawful_promotion` with the assessment's own `gates_passed`,
and for FEATURE_ELIGIBLE+ requires a verified-loaded owner artifact that passes
`assert_owner_decision_authorizes` — a bare 64-hex ref is refused (`test_regime_store.py:341-352`,
CLI test `:196-198`); synthetic provenance is refused under the default
`full_authorized_development` scope (`:502`, tests `:379-380`, CLI `:203-205`); the pipeline maps
the charter's `SyntheticAuthorizationMarker` to `synthetic_fixture` (`pipeline_regime.py:117-123`)
and S00 refuses `full_authorized_development` under that marker (`pipeline.py:942-947`).
Readiness (`regime_study.py:452-465`), S05 (`pipeline_regime.py:229-237`), S09a (`:427-435`),
S10 (`:505-513`) and block activation (`regime_block_activation.py:177-190,280-288`) all re-verify
the exact frozen ids under the run scope. A forged supersession line fails the chain closed
(`test_owner_decisions.py:341-368`). Gaps: S2, S3, S5, S6.

**(4) No automated selection / promotion surface.** Regime Lane source: the only Streamlit widgets
are five `st_module.text_input` (`ifvg_regime_panels.py:133-153`); no button / form / toggle /
selectbox / multiselect / radio / checkbox / slider / download_button / `session_state` (grep +
`test_ifvg_pipeline_tab.py:897-917` source scan). `_spawn_pipeline_job(` call sites: 2 at HEAD
(`:617`, `:1073`), 2 now (`:1026`, `:1659`) — none added. CLI: `test_import_launches_nothing`
(`test_ifvg_regime_promotion_cli.py:85-95`, AST check that `persist_regime_promotion`/`main` are
never called at column 0; no `subprocess`); `model_feature` refused with the exact text
(`:153-173`). `IFVG_REGIME_CONTEXT_V1` stays `PLANNED` in the module registry
(`feature_blocks.py:370`); activation is a pure event over copies (`regime_block_activation.py:296-303`).
S10's STRATIFICATION_READY requires `initial_status is DESCRIPTIVE_ONLY and coverage_gates_passed
and oos_assignment_available and request.descriptive_classes` (`regime_study.py:526-545`) and is
persisted through `persist_regime_promotion`, which re-checks the assessment; `decided_at` is the
evidence as-of, never the wall clock (`:481-491`).

**(5) S14 zero fitting; S02 reproduction.** No `sklearn`/`catboost`/`joblib`/`.fit(` in
`regime_stratified_{strategy,prop,frontier}.py`, `regime_stratification_{gate,service}.py`,
`regime_report_stage.py`, `regime_assignment_sources.py` (grep → no hits);
`test_s14_performs_zero_fitting` monkeypatches every estimator's `fit` to raise during S14
(`test_pipeline_regime.py:438-489`); the stage record carries `fitting_performed: False`
(`regime_report_stage.py:208`). Prior-attempt report recovery is by verified reload of the exact
ids in the prior stage sidecar (`:96-120`); S12/S13 record exact persisted simulation ids via the
`on_simulation_persisted` hook, never a listing (`pipeline.py:1738-1760, 1796, 1857-1864`, `search_bridge.py:431-439`).
S02 reproduction compares `compute_strategy_metrics(...).model_dump()` byte-for-byte with the
persisted evaluation and raises on mismatch (`pipeline.py:1042-1062`) — with the `None` gap (S4).

**(6) D15.** Preflight row budget (`event_detail.py:377-382`) and cumulative byte budget (`:414-420`)
raise `EventDetailBudgetError` inside `build_account_event_detail`, i.e. before
`save_or_reuse_envelope` (`search_bridge.py:111-129`); publication is atomic (`store.py:160-210`);
`none_v0` ⇔ no storage/schema 0/no budget is validator-enforced on both payloads
(`simulation.py:155-166, 220-231`), the policy enters the identity (ids move — recorded), and the
store refuses different sidecars under one id. The reader verifies detail manifest ↔ envelope
identity, budget, every partition's sha256/bytes/rows/schema hash, block membership and total
order (`:464-556`). `trading_day` never enters `event_id` (`account.py:477-484`).
`tests/propsim/test_account_event_detail.py` 9 passed.

**(7) DEVIATIONS honesty.** `threadpool_limits(limits=1)` wraps the entire kernel
(`regime_service.py:450`); golden fit id pinned (`test_regime_service.py:784-789`); the double-run
test proves identical stage-result ids (`test_pipeline_regime.py:193-232`) — DEV-R6.1-4 and
DECISIONS #79 are accurate. Session rule: `_session_facts` classifies at `close − 1 µs` and
measures position from the close instant (`context_bar_panel_materializer.py:253-277`) — DEV-R6.1-5
accurate. Identity evolutions (DEV-R6.1-2 / #78) match the code (`bundle_feature_view.py:70-73`,
`simulation.py:148-166`, `charter.py:245-252`, `pipeline.py:273-276`). DEV-R6.1-8 / #82 overclaim
S02 (S4). DEV-R6.1-6 (Error-suffix names) accurate. DEV-R6.1-3: the fixture's seam ignores the
chart id (S1) — not stated.

**(8) Frozen lanes.** `git status` lists none of `context_model.py`, `context_folds.py`,
`context_labels.py`, `context_feature_view.py`, `replay_chart_store.py`, `manifest.py`,
`data_access.py`, `preparation.py` (M0–M3 lane untouched); `PRE_R6_1_*` registries pinned
(`test_feature_blocks.py:286-306`). Propsim: `make_prop_simulator` gained keyword-only params with
`none_v0` defaults (`search_bridge.py:257-259`), `persist_account_simulation` signature unchanged,
`PropAccountEventEnvelope.trading_day` defaults to `None`. `../Strategy-Core`: clean worktree
(0 entries), HEAD `a4e3303` 2026-08-05, no file newer than 2026-08-25. `../Trade-Lab`: 27 dirty
entries (pre-existing, R6 evidence says the same 27), newest mtime 2026-07-31 14:19, no file newer
than 2026-08-25. Plan package: unmodified (`git status` shows only the untracked `QL-*` tree).

**(9) Dependencies.** No change to `pyproject.toml` / lock files (`git status` clean for them);
`threadpoolctl 3.6.0`, `scikit-learn 1.7.0` present; sklearn declares `threadpoolctl>=3.1.0` (S9 note).

**(10) UI safety.** Exact-ID loads only (`load_regime_protocol/assessment/fit_assignments/promotion`,
`load_owner_decision`, `load_regime_stratified_report(+detail)` — all `load_verified_envelope` /
`load_sidecar_bytes`; 64-hex enforced at `envelope_destination`, `store.py:120-126`; stores never
listed); `load_regime_fit_assignments` never unpickles (`regime_store.py:447-472`,
`test_ui_loader_never_unpickles`); every loader failure goes through `sanitize_error`
(`ifvg_regime_panels.py:167-169, 218-221, 523-525, 627-629, 682-689, 758-774`);
`regime_stage_evidence_defaults` auto-fill reads only manifest-verified stage sidecars and surfaces
a store-integrity note on `SearchStoreError` (`study_providers.py:727-803`). FUX source scans read
both `ifvg_regime_panels.py` and `ifvg_pipeline_tab.py` (`test_ifvg_pipeline_tab.py:903-917`);
`study_providers.py` is a library module (not a script) and its new functions use no listing
(the two `iterdir()` calls at `:128,:528` are pre-existing).

**Commands / targeted checks.**
- `python -m pytest -q -p no:cacheprovider tests/agents/ifvg_search/test_owner_decisions.py tests/agents/test_ifvg_regime_promotion_cli.py tests/propsim/test_account_event_detail.py tests/agents/ifvg_search/test_context_bar_panel.py` (with `PYTHONDONTWRITEBYTECODE=1`) → **48 passed in 9.15 s**.
- `ruff check` over `regime_store.py owner_decisions.py pipeline_regime.py regime_study.py ifvg_regime_promotion.py event_detail.py context_bar_panel_materializer.py` → All checks passed; `git diff --check` → clean.
- Probes (synthetic, `%TEMP%` only): `scratchpad/probe_owner_supersession.py` (P1–P4 above),
  `scratchpad/probe_panel_chart_binding.py` (S1 above).
- Full suite NOT run (out of scope for this lens); the implementer's `_midpoint_pytest.txt` was
  still being written during this review.
