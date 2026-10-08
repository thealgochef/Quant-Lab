# Execution status: corrected batch complete

Current retrieval, verified October 8, 2026: the owner-authorized integration
and repair completed all six strategy rows and twelve funded rows across 253
evaluation dates plus ten warmup dates. The corrected result is
`7278632babf01b084c43ddb6df77332f15d8b16702c7082390553b856053d4ad`.
Evidence: sibling artifact folder
`ifsm-correct-config-full-range-v01/repair_v02/result_validation.json`.
See [the storage index](../RESEARCH_STORAGE.md) for current locations. The dated
preflight and continuation notes below remain the historical execution record.

## Historical initial preflight: stopped at the engine capability gate

Checked October 4, 2026. No strategy replay, funded replay, new approval or
dispatch was created. The six-configuration batch is **not complete**.

The supplied instruction ZIP was verified against `PACKAGE_MANIFEST.json` and
its six documents installed in this folder. Existing saved studies remain
unchanged.

## Exact launch blocker

`TASK.md` section 3 requires an already implemented engine that supports both
`scale_out_half_breakeven_hold_to_close_v1` and the five passive MenthorQ
settings. It states: "If both capabilities cannot be combined without a
substantive change, stop with the exact blocker instead of using a different
exit or an older engine silently."

The current pinned Core is commit
`709487bc85ef82a259297c4ad6f3f198f4cabbcd`, source identity
`30ffab98b2766da326a3712d2639e962e1b72241b9c3327fdb41faa95f5ec57b`.
Its actual imported section supports MenthorQ but has no `exit_policy` field.
Validating the requested partial-exit payload raises `extra_forbidden` for
`exit_policy`. The existing partial-exit source at the sibling
`strategy-core-scale-out-exit` checkout has base commit
`7c7111e398c083cf8e966e2e0c5aac8a41cc12c0` and source patch
`962ad56a36471c85b1b05ea1e106eddce7bea2dcfc3a523ee6a01ab6b1011a6f`.
It supports the requested exit but rejects the five MenthorQ context fields.

The inspected local Core checkouts and available local Git references contain
no implementation combining these capabilities. Routing changes cannot add
missing execution behavior to Core. A compatible engine implementation and
verification, beyond this task's allowed registry/runner work, is required
before this batch can be frozen and approved.

## Completed preflight work

- Resolved all six historical configurations, including the authoritative
  daily-close source and exact saved approvals for C05 and C06. Preserved their
  original identities and recorded schema differences.
- Verified the Task B archive member SHA-256
  `9001b9f387548aa229313a81d7cc08dd6b7646a2911b9e245987f295f29a64ac`
  and exact ordered 253 evaluation plus ten warmup dates against `date_scope.json`.
- Verified both prepared-store registration receipts, unique ownership of
  every requested replay date, both preparation-catalog hashes, and the saved
  firm/processing-clock hashes. This checks metadata; prepared bars and raw
  market observations were not reread.
- Verified `python scripts/run_ifsm_research_ui.py --check` succeeds for the
  current pin. Isolated schema probes establish the incompatibility above.
- Inspected existing orchestration, verified export and Lab reopen paths.
  No source changes were made; no regression suite was necessary.

## Evidence and limits

Detailed preflight evidence is under
`C:/Users/gonza/Documents/Claude-Quant-Lab-Research-Artifacts/ifsm-correct-config-full-range-v01/preflight/`:
`input_scope_audit.json`, `historical_bindings.json`, `capability_audit.json`,
and `export_lab_findings.json`. The evidence contains no new financial outcomes.

There is no completed result folder, review ZIP or completed Lab entry for this
batch. No partial four-configuration run or substituted exit was launched.
No raw market observations, protected dates, new feeds, model fitting,
commits or pushes were used. The requested comparison remains unexecuted.

## Owner-authorized continuation, October 4

The preceding sections preserve the original blocked preflight. The owner's
`ENGINE_INTEGRATION_AND_RESUME_v01.md` supersedes only its missing integration
authority. Work resumed in the same task and external artifact directory.

The existing partial-exit patch was ported onto a task-owned checkout of current
Core 709487. The actual partial execution methods match their authoritative
source; current timing, calendar and context modules remain unchanged. Core
tests passed 742 with three named real-store probes skipped, followed by 17
strengthened preservation checks. Actual funded off/on fixture checks passed
48 cases across both exit families and firms; identity exclusions are explicit
and account references reconcile. These are engineering checks, not new market
outcomes or historical full-range equality.

The runner retains exact six-row sections, input/calendar/source approval
bindings and per-day checkpoints. Saved reports cover six ordinary and twelve
funded streams without allocating funded cash by context. The focused read-only
review's evidence and resume-binding findings are being addressed before the
final plan is frozen. Batch execution and review publication remain pending.

## Delivered outcome, October 4

The preceding pending statement records the engineering phase. The original
six-configuration batch subsequently ran under one frozen plan and new approval:

- Plan: `215f63e0589c42108d77c68f7be8816458083e2fd644913ae855a3a852b00d4a`.
- Approval: `f334eddc771efc5432f365e46b72b41dcf9ade6ee5e7b4e8499ec27a40ce2d6d`.
- Immutable result: `ec33a39d953f7308d41f2be092278988c2ffbbf75d29fdfa736585a1a776ff0f`.

C03–C06 completed all 263 declared replay dates (253 evaluated plus ten warmup).
C01/C02 retain truthful failed rows over that same declared scope. Both stopped
at `ScheduledCloseCoverageError: position crossed mandatory deadline without an
executable boundary bar`; verified resumes reproduced the failure after the
November 28 checkpoint. No completed worker was replayed. Changing the frozen
calendar/closing assumptions or adding executable boundary evidence would be
required to finish those rows; no close was fabricated and no input was prepared.

The result is **Incomplete**, with six ordinary and twelve separate funded rows.
Completed accounting and activity checks pass; failed full-range economics remain
unavailable. Among completed configurations, TakeProfitTrader's net-cash leader
is S1_D160_W1_P1 ($18,722.69), and MyFundedFutures' is S0_D160_W1_P1 ($20,089.46).
These are received trader cash less account acquisitions, separately by firm.

A saved-record timestamp schema adapter repaired report assembly without
changing any instant, worker record, financial calculation, plan or approval.
Read-only Lab adapters corrected failed-row indexing, full parent-chart labels,
optional package-less diagnostics and unperformed-check wording. The existing
Lab library discovers and reopens the same verified result. Actual captures
show both firms and effective settings. All 50 frozen code/owner fingerprints
and the task Core identity remain unchanged.

The existing strict publisher created
`reports/funded_comparison/funded_comparison_ec33a39d953f7308_export_v1/` and its
review ZIP. All 37 manifest-listed payloads verify after extracting the final ZIP
outside the repository; there are no missing, mismatched or unlisted payloads.
Review ZIP SHA-256:
`119f6c0c14774234e80c70f619807cbf332b4523e2755261d068249a73940c1d`.

The separate source-review ZIP is under the original external task root:
`source_review/ifsm_correct_config_full_range_v01_source_review_v06.zip`.
Its 107 payloads and layered patches verify after extraction; SHA-256:
`d7b806d151dc390ace8ae4d4361ed3c3d9352fc90b21f53e652073cab873181c`.
Focused test scopes, repaired attempts, skips and saved-record access limits
are recorded in the packages. No full regression suite was claimed clean.

The authorized guarded inputs were used for this run. Protected-date data,
2021–2022 inputs, new feeds, model fitting, follow-on research, live activation,
default/global Core upgrades, commits and pushes were not used. Historical
results, original blocked findings, frozen proofs and previous source packages
remain preserved. This batch is closed at TASK's completed-or-truthfully-failed
finish line; no new experiment is launched.
