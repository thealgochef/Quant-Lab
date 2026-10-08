# IFSM MFFU context batch — task record

Updated: 2026-10-07. The owner request and bounded matrix are in
`C:/Users/gonza/Downloads/IFSM_MFFU_Context_Batch_v01.zip`; that archive remains
the immutable specification source. This record tracks implementation and run
state, not an amendment to the handoff.

## Current state

- Verified the package manifest, its 64-intent matrix and 188 direct comparison
  pairs, the repaired six-result reference `7278632babf01b08…`, its exact
  253 evaluation plus 10 warmup dates and Task B registrations. The handoff's
  `0283b…` source-binding caption conflicts with the verified repaired plan and
  result; both bind actual repaired Core patch `028b3ce4…`.
- The 186 supplied EOD level-set identities through the cutoff are vendor
  `NQ1!` continuous-front-month coordinates. Task B uses source-selected NQ
  contracts; exact roll parity has not been independently established, and
  the batch applies no futures-basis adjustment. A metadata-only audit matched
  both approved Task B catalog hashes and their selected contract/roll records.
  The available metadata contains no vendor `NQ1!` constituent/roll mapping or
  paired price evidence, so it does not establish a price-basis mismatch either.
- Implemented isolated task Core context/cap/gate/geometry/conditional-exit
  policies and funded target-observation override. Installed Core and historical
  studies remain unchanged. Added a source-bound v02 EOD lookup, full 64-intent
  plan, independent ordinary/funded worker, per-day resume, saved-result analysis,
  Lab matrix read path and result-review exporter. These are not yet a completed
  financial study.
- Focused policy, adapter, preservation, worker and Lab tests have passed; a
  registered-input preflight verified all 64 calendars and the 10+253 scope.
  One authorized warmup day ran through the new worker and stopped before
  evaluation trading. A consolidated Quant-Lab gate passed 111 tests, with 48
  passive cases skipped until the task Core path was set; all 48 then passed.
  An initial Core parity test used a pre-repair calendar reference and failed
  nine assertions; it now binds the exact repaired Core named by the prior
  six-result plan. The corrected Core gate passed 132 tests, including the
  November 28 early close, without changing Core economics. The final
  pre-fee trade-audit source then passed 219 focused Quant-Lab tests with
  the task Core selected, plus Ruff over `src tests scripts` and task Core
  `src tests`.
- MCB001/MCB025 (prior C02/C01) were reused under guarded immutable-output,
  neutral-section and source-parity proofs. Their v02 context remains a
  separately labeled posthoc sidecar, not an executed input. The managed
  runner and review exporter verify those proofs and sidecars on readback.
- Both review ZIP publishers are implemented with extracted manifest readback.
  Their actual final publications require the completed saved result and real
  Lab screenshots; neither ZIP exists for this task yet. The explicit external
  saved-result Lab route is documented in `LAB_REVIEW_ROUTE.md`.
- A read-only provisional plan build retained 64 distinct rows. Runtime
  validation deliberately refused MCB049's three-micro fee posting; no exact
  plan, approval or new financial result was saved from that probe.
- A separate read-only preflight skipped only that fee check in its own Python
  process to expose other input issues. It verified 64 sections, the 253-day
  calendar and all 10+253 registered date metadata records: 526 metadata file
  opens, zero market rows, zero protected rows and no denied date. The
  production approval/run guard was not modified.
- A focused read-only worker review found two resume-binding gaps. The terminal
  saved-result path now checks the exact approval and ordered 64 dispatches, and
  day checkpoint readback requires both stream activity rows for every completed
  evaluation date. Targeted mismatch tests and Ruff passed after these repairs.
- The prior repaired v1 review folder passes its existing manifest readback with
  36 listed payloads and 37 total files. Its prose count says 32, its repair
  caption says `0283b...` where the saved plan/result bind `028b3...`, and two
  partial-exit rows in `configurations.csv` describe micros as minis. A separate
  metadata-only v2 review folder was published and passed manifest readback:
  36 listed payloads, 37 files, manifest SHA-256 `409a3fd0...`. Thirty-two
  other payload files retain identical bytes, and no economics were regenerated.
  The old folder and saved result remain immutable.
- The settled pre-fee source passed a consolidated 287-case focused Quant-Lab
  gate: 239 passed in the general run and the 48 intentionally fixture-gated
  passive Core cases passed separately with the task Core selected. The task
  Core's selected gate remains 132 passed. Ruff passed over Quant-Lab
  `src tests scripts` and task Core `src tests`; Git diff whitespace check
  reported no errors. These are implementation checks, not a financial run.
- The saved Lab matrix now shows effective values for all 64 policies, and the
  saved result view exposes matched effects, interactions and receipt waits.
  The review publisher adds MFFU-specific trading rules and account/HTF-zone
  concentration, and verifies decision timestamps. Real result-screen
  screenshots still require a completed approved batch.
- A source-binding audit found that the frozen plan omitted the registered day
  loader and other worker/result dependencies. The plan now hashes those exact
  source files, including `prepared_store.py` and `day_artifacts.py`; a synthetic
  loader-mutation regression confirms rejection before any market read. The
  focused plan/source-review gate passed 10 tests, Ruff and whitespace checks.
  A fresh read-only provisional plan retained 64 rows and 56 bound Quant-Lab
  source files; it was not saved or approved. The focused prelaunch findings
  and limits are recorded outside the repository in
  `../Claude-Quant-Lab-Research-Artifacts/ifsm-mffu-context-batch-20261007/validation/pre_fee_implementation_review.json`.

## Owner fee decision and implementation

On October 7 the owner selected per-execution-fill fee posting: multiply the
actual filled quantity by $0.514 per micro, then round that fill's total to the
nearest cent with `ROUND_HALF_UP`. A six-micro entry posts $3.08; each
three-micro exit posts $1.54; the three fills total $6.16. Historical
ten/five/five fills remain $5.14 + $2.57 + $2.57 = $10.28. The convention is
bound as `per_fill_total_round_half_up_cent_v1` in the new plan, variant behavior
identity, worker sizing, reuse proof and result settings; the ordinary replay
and funded ledger use the same actual-fill helper. Focused arithmetic,
ordinary/funded reporting and compatibility checks precede plan approval.

The named `FEE_ROUNDING_ADDENDUM_v01.md` and companion arithmetic cases were not
present in the repo, original handoff ZIP or task-owned handoff extraction when
checked. The implementation follows the owner's direct explicit rule and
six/three/three example; it does not claim to have verified unseen case files.

The first approved source-bound plan `f35adf99…` saved the two compatible
controls and an incomplete result `2bf271db…`: its 62 new Windows workers
refused a Core import already present during process-spawn initialization. No
new configuration replay completed in that attempt. The result, approval,
checkpoints and source remain preserved in the task-owned store. The worker
initializer now permits a preimport only when its module path resolves inside
the frozen task Core. A real spawned-worker import smoke check and 25 focused
worker/plan/reuse tests passed. That source change required a new exact plan
`9ef09a18…` and approval `e054d31f…`. The managed replay sealed 63 of 64
outputs, including those same immutable C01/C02 controls, before MCB060 failed
to replace one Windows day checkpoint (`PermissionError`). The immutable
incomplete result `2b9d5039…` and the last valid MCB060 checkpoint remain
preserved. The first one-worker continuation disappeared before MCB060 dispatch
for an unknown reason. A detached same-plan, one-worker continuation verified
the 63 sealed outputs, resumed only MCB060, and saved completed result
`7a8cb062d3fc31bcf71e7015a60e4ea389988b749435609fbc15cd1beda9d052`
on October 7. No new plan, approval, or vendor-data preparation was needed.

The incomplete result exposed a review-analysis field mismatch: saved funded
trades use `account_number`, while the bound concentration/diagnostics module
looks for `account_id`. Its account concentration buckets and diagnostic
account-ID cells are therefore invalid. The saved trades, fee postings, cash
ledger, and account journeys remain intact. A separate read-only,
account-number derivation reconciles exact cents outside the approved plan;
the reviewed findings disclose the affected built-in fields and include the
corrected leader account view. Its full 425-row CSV and SHA-256 receipt stay
in the task-owned Research-Artifacts folder; the source review ZIP carries
the review-only CSV bytes as a `.txt` receipt.

## Final saved result and publication

The verified saved result contains 64/64 completed policies: 62 new and two
compatible reused controls, with zero failed dispositions. All 188 declared
matched comparisons are available. Read-only audit reconciled every account
ledger and found zero fill-fee mismatches across 12,556 funded and 15,559
ordinary saved trades. The final plan retained all 64 intents, including Q6
and QG. MCB062 leads the inspected MyFundedFutures cash ranking at $48,102.16
net received cash; recovered MCB060 ranks third at $39,937.16. This is one
historical sample and does not promote the task to Core or alter
`research/core/current.json`.

The completed comparison and reviewed findings are in ignored
`reports/ifsm_mffu_context_batch/IFSM_MFFU_Result_Review_7a8cb062d3fc31bc_v1.zip`.
It contains the standard verified comparison export, MFFU-specific analysis,
64 dispositions, 188 pairs, and two real reopened-Lab PNG captures. The
publisher and an independent readback verified all 55 ZIP members and its
SHA-256 `7f0cf7933aa83340b4db3713802a779ffa62114b94178db0106f12ea1a2f3414`.
The separate source-review ZIP is
`reports/ifsm_mffu_context_batch_7a8cb062d3fc31bc_source_review.zip`;
its own manifest and extracted readback bind the approved source, Core patch,
focused-check evidence, preserved recovery receipts, and the result-review
ZIP. Both ZIPs are final lightweight peer-review deliverables under ignored
`reports/`; working stores remain outside the repository.
