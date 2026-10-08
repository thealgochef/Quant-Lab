# b8db reporting follow-up — B8-01 and B8-02

## 1. Authorized outcome

Finish the two remaining reporting corrections from the independent review, prove that the corrected financial result is preserved, and deliver verifiable result/UI and source/test packages. This instruction is an implementation assignment, not a request for another plan or diagnosis. Use [ACCEPTANCE.md](ACCEPTANCE.md) and [RETURN_REQUIREMENTS.md](RETURN_REQUIREMENTS.md) as the completion contract.

The target economic result is:

`b8db8427cb2260619d7f0483358e5d0da78d718fc395f2374b6a62a1e378c186`

Its historical predecessor is:

`7a8cb062d3fc31bcf71e7015a60e4ea389988b749435609fbc15cd1beda9d052`

The reviewed reporting version is `ifsm_mffu_reporting_v5`. That is an inspection anchor, not permission to overwrite an existing version or ignore newer work. Resolve the next available immutable report revision from current project state. Do not hardcode a claimed final version before checking it.

The previous lifecycle repair is complete: 18 reused policies, 46 replays, 28 changed financial histories, and 26 changed cash totals. Do not reopen that implementation or replay those histories for these presentation defects. The reference files preserve this distinction.

## 2. Isolation without losing normal integration

Read the repository's current root/scoped instructions. Establish the actual imported reporting/UI/export source, the current worktree, any active writers, the relevant stored result, and already-completed follow-up work. Do not revert to an archived checkout, compare against an assumed clean main, or attribute unrelated dirty changes to this task.

Use a task-local worktree or the established safe isolated edit/test arrangement. Record task-only before/after bytes and actual imports. Test against isolated copies or immutable read-only result references. Preserve the production Core pin and every executed economic source binding. The pin reported by the delivered source is an audit anchor; do not reset a legitimately newer unrelated pin.

Isolation is for edits and verification. The completed fix must be wired into the **ordinary main Quant-Lab → IFVG Lab → My studies → IFSM MFFU context batch → Corrected lifecycle** workflow. Do not deliver a private port, task-specific environment variables, or a one-off viewer as the result.

A newly versioned report pointer/catalog addition is permitted through the existing supported publication mechanism. Do not modify historical result/plan/approval bytes or relabel the repaired display as a fresh financial run. Retain access to Original execution and the earlier reporting versions.

If either finding is already fixed locally, verify its actual behavior and include that work in the closeout rather than reimplementing it. Missing analysis-tool dependencies in the previous reviewer environment are not new application defects.

## 3. B8-01 — checkpoint-specific gamma origin

### Reproduce before fixing

Read [inputs/REVIEW_CHECKPOINT_PROBE.json](inputs/REVIEW_CHECKPOINT_PROBE.json) and the source excerpts. The actual anchor is MCB003 trade `e2189e7d-1151-541d-a60e-0f1e992b587a`:

- actual decision/fill: `1750068615558271393` UTC nanoseconds;
- actual checkpoint: `2025-06-16T10:10:15.558271393Z`;
- receipt context datetime: `2025-06-16T10:10:15.558271+00:00`;
- difference: 393 ns, caused by the stored representation's lower precision.

The reviewed code labels this first-target checkpoint as an annotation despite retaining the linked receipt, then the selected row/coverage or UI can reuse the entry's executed-context role. The prior report counted 1,908 funded trades with a saved conditional `target_decision`; this is a historical audit count to reconcile, not an instruction to force every record to be labeled executed without verifying it.

### Required behavior

1. Resolve each checkpoint from its own actual evidence. Prefer a verified link through the saved target decision, exact decision/fill identity, scoped trade/configuration/stream/account identity where available, and the matching source context. A receipt merely near the timestamp is not proof. Exact timestamp equality is also not the only possible identity proof when encodings have different precision.
2. Preserve the highest available precision of the actual observation/fill time. Retain the separately stored receipt context time and the reporting annotation's evaluation time. Document the existing conversion/precision relationship when used to validate a real link. Do not use floating-point time conversion to erase distinctions.
3. Do not use a blanket +/- one-microsecond tolerance, nearest-time match, or same-bar match to attach an otherwise unproven receipt. Do not round the actual event backward to make the test pass. A wrong trade, stream, account, policy, source snapshot, or ambiguous receipt must not become executed evidence.
4. Validate causal source eligibility under the saved policy, including report-boundary cases. The funded print checkpoint and the ordinary strategy's declared candle decision can legitimately have different timing contracts. Preserve them; do not force one stream onto the other's timestamps.
5. Propagate the checkpoint's own origin consistently through the normalized row, basis selection, group coverage, checkpoint cards, Trade review, and generated reporting tables. Entry-origin metadata must not overwrite first-target origin. Parent-lock origin, where displayed, is independent too.
6. Keep annotation-only observations honest: an unconditional partial exit with no saved target-context decision remains a reporting annotation; so does enrichment of a reused reference when no original decision-context evidence exists. A checkpoint not reached is not an annotation event. Missing or contradictory evidence stays explicitly unavailable/unresolved rather than receiving a fabricated origin.
7. Distinguish a source observation recorded alongside an event from a context value actually used by a conditional policy. Provenance is not proof that gamma caused the outcome. Do not invent target action receipts for unconditional policies.
8. Keep point-in-time visibility based on the actual event time and canonical event order. A first target and its context/action cannot become visible 393 ns early merely because its receipt uses microseconds. No later trades, accounts, balances, exits, or setup information should leak through the picker/cards/chart at an earlier cursor.

Find and fix the production reporting path, not just the reviewer probe or a hand-edited CSV. Useful inspected anchors are `presentation/lab/mffu_gamma.py` (`build_gamma`, `selected_rows`, `coverage`, `checkpoint_cards`) and `scripts/ifvg_lab_mffu_views.py` (`render_trade_gamma`); find their current locations locally. An additive reporting-only normalization/helper is allowed. Changing historical economic source, actual receipts, or the live execution policy is not.

### Numeric preservation

This is a provenance/linkage repair. The prior review found gamma values, selected dates, geometry, fees, and money consistent. Expected changes are checkpoint-specific origin/linkage/precision metadata and the corresponding origin coverage counts. Underlying numeric snapshots and the trade membership of economic gamma groups must remain unchanged, except any demonstrably wrong field that requires an explicit discrepancy report. Do not silently alter the as-of policy to get numerical equality.

## 4. B8-02 — one policy-description source for UI and exports

The new corrected standard export, not just an old archive, still has known-wrong labels. Read [inputs/REVIEW_STANDARD_EXPORT_LABELS.json](inputs/REVIEW_STANDARD_EXPORT_LABELS.json).

1. Derive plain-language descriptions from the verified effective configuration plus bound intent and execution evidence. Reuse/extend the improved shared label model; do not maintain a second divergent lookup table in the CSV or plot exporter. Descriptions and tests must reflect the executed plan, not stale `planned_state` wording in inherited metadata.
2. For all 64 rows, describe the position as **Micro E-mini Nasdaq-100 exposure (MNQ)** with the actual constant or regime-dependent quantity policy. State separately that NQ candles/prints supply signal and price-proxy evidence. Do not relabel the source data as actual MNQ executions.
3. For MCB057–MCB062, show the actual 5%, 7.5%, or 10% opposing-pattern-distance rule: `(1D Max − 1D Min) / 2 × fraction`, frozen at supporting-parent lock and converted to the quarter-point tick grid under the saved rounding rule. Describe fixed 80 ticks / 20 points separately as the missing/invalid-context fallback. Do not call that fallback the active maximum. Fixed-distance policies must remain correctly described as fixed.
4. Express each exit accurately: always partial, always whole at 1R, positive-gamma conditional at the first 1R checkpoint, or early-positive conditional at that checkpoint, as appropriate. Preserve the actual unknown/neutral fallback from the plan; do not infer it from the name. A conditional exit must not be labeled simply 'whole exit.'
5. Give generated chart entries a stable configuration ID plus concise distinguishing schedule/exit/geometry information. Where the displayed candidates differ by sizing, cap, or entry filter, include that difference in the label or an adjacent readable key. Labels must not falsely imply equivalence. Retain readable full labels, units, the actual top-N/full-set scope, and source-consistent cash values. Avoid hashes, raw cents, or undocumented codes in the business-facing chart.
6. Update the actual standard export path used by the application. Correct both new `configurations.csv` and generated `net_cash_by_configuration` chart output, plus directly shared surfaces. Do not rely on a manually patched post-export file or only the improved integrated UI.
7. Regenerate the affected output as a new immutable reporting/export revision referencing the same b8db economic result. Preserve all earlier archives and their labels as historical evidence. No currency, quantity, trading target, stop, date, or account computation changes are authorized.

A small shared presenter/exporter change and directly affected tests are in scope. A broader UI redesign, new settings, new features, or cleanups unrelated to these labels are not.

## 5. Protection and permitted computation

Permitted: read existing verified saved result/receipts and already-bound context; derive the reporting companion; render the normal UI; run synthetic/record-fixture tests and report regeneration. No new data acquisition is required. Keep existing date access rules; do not inspect June 11, 2026 onward market history or broaden raw-data paths.

Prohibited: financial backtests/replays (including a 'quick' new historical run), altered fees or rounding, Core/strategy/worker changes, new study approvals, new ML work, expanded dates, reconstructed missing setup evidence, changes to trading thresholds or gamma definitions, or commits/pushes. No changes to root `.claude`, `CLAUDE.md`, or `AGENTS.md` are requested.

Before work, capture relevant immutable result/child/approval/plan/receipt hashes and scoped source/pin references. After work, verify the same originals and all 64 economic projections. New report headers, origin metadata, labels, and report-catalog references can differ, but those differences must be explicitly allowlisted. Do not apply a broad comparison that ignores all changed fields.

If a genuine economic or source-context contradiction emerges, preserve it, finish independent safe work, and report the minimal failing record and the additional decision needed. Do not launch a financial rerun or suppress the contradiction just to mark the task complete. Ordinary choice of helper names, report revision, compatible test environment, or archive location is an engineering choice; no separate owner question is needed.

## 6. Execution and closeout

Reproduce → implement both corrections → run focused final-source checks → inspect affected normal-app views → regenerate versioned reports/exports → verify invariance → perform one focused independent review → seal and extract-check both delivery ZIPs. Combine review-driven fixes; rerun their affected tests, not every previously accepted feature.

Follow the repository's mandatory test rules, but this handoff does not request a new full suite. Captured focused tests must cover the actual changed imports and final source. If an earlier broad test predates a later edit, identify it as earlier and rerun the relevant affected tests on the final source. Compilation, hand-calculated JSON, and screenshots alone are not substitutes for executable behavior checks.

Update [TASKS.md](TASKS.md) with exact status/evidence. Do not claim closure at code completion. The finish line is both corrections visible in the normal dashboard and newly generated export, preserved economics, and both review ZIPs in [RETURN_REQUIREMENTS.md](RETURN_REQUIREMENTS.md).
