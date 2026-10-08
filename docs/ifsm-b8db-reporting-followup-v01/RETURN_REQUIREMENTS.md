# What the agent must return

Return two compact new archives and one short final summary. Do not resend hundreds of unchanged files or the full 20 MB financial archive solely to close these two issues. Do include the actual changed source, necessary test dependencies/fixtures, and enough evidence to review both fixes without reading unsupported terminal assertions.

Suggested filenames (choose a new suffix if already present):

- `IFSM_b8db_Reporting_Followup_Result_UI_v01.zip`
- `IFSM_b8db_Reporting_Followup_Source_Tests_v01.zip`

Both must identify the unchanged b8db economic result, the new reporting/export revision, the reviewed v5 predecessor, and the actual final source. The filenames are suggestions; identity reconciliation is mandatory.

## A. Result/UI review ZIP

Include:

1. **FIX_REPORT.md** — B8-01 and B8-02 disposition, actual root cause/fix, final report/UI location, what changed, what did not, and remaining limitations. Start with a clear completed/not-completed status for each item.
2. **CHECKPOINT_ORIGIN_AUDIT.csv** — one compact row per saved funded conditional target decision, with scoped identity, observation time, receipt time, prior/corrected role, match method and validation reason. Summary counts must also cover ordinary, annotation-only, no-target and missing-evidence categories where applicable. Do not duplicate complete price histories or full source snapshots per row.
3. **PROVENANCE_COVERAGE.json** — basis-specific totals before/after, errors/ambiguous/unavailable records, numeric context/group preservation counts, and exact source/report references. Preserve legacy annotations; do not count nonexistent checkpoints as annotations.
4. **The complete new standard `configurations.csv` for all 64** and the new `net_cash_by_configuration` chart. Include a compact all-64 label consistency check; test all six geometry rows and every conditional exit.
5. **ECONOMIC_PRESERVATION.csv/JSON** — all-64 before/after financial equality and counts; protected record hashes/projections; explicitly allowed reporting differences. Money/history changes must be zero. A single leader comparison is insufficient.
6. **A small screenshot set with an index** — normally four to six captures are enough: a recorded target with consistent origin, an annotation-only target, point-in-time before/at target (can be combined with those), readable geometry/conditional labels and the normal study/report-version route after reload. Use the final renderer. Link each to its source version, configuration/trade/checkpoint and actual evidence purpose. Do not retest every unchanged screen.
7. **MANIFEST.json** — every included payload's relative path, size and hash; explicit external references, no silent omissions.

This archive is reviewer-facing business/report evidence: no application scripts, code patches, raw ticks/bars, environments, caches, credentials or giant dumps. A changed code file belongs in B.

## B. Source/test review ZIP

Include:

1. **README.md / SOURCE_MAP.json** — pre-task and final imported source, task-only file list, baseline reference, exact commands/dependencies, and how to run the focused portable subset. Do not promise standalone execution if an undeclared original checkout is required.
2. **Task-only patch plus changed/new source and tests**, with before/after hashes. Base it on the captured actual working baseline, not guessed main. Include a small reconstruction/fixture subset for the reporter and exporter. Dependencies that cannot be packaged must be precisely declared; never fake them to obtain passes. No Core/execution change should be present.
3. **Actual minimal saved-record fixtures** for the precision case, a conditional half exit, an annotation-only target and a reused control; synthetic negative cases clearly labeled. Preserve exact event IDs/timestamps/source context, redact credentials only (none expected). Do not borrow unavailable setup evidence from another configuration.
4. **Captured failing-before/passing-after and final focused test/lint output**, command/interpreter/import/source bindings, skip/deselection reasons. If already fixed at task start, identify the archived-before reproduction separately from current verification.
5. **Read-only reviewer findings and resolutions** — concise, tied to code/tests/evidence, not a score or a claim that everything is perfect.
6. **Publication and preservation receipts** — immutable predecessor/new reporting identities, same b8db economic identity, protected file equality, normal-dashboard report reference, and source/report/screens evidence binding.
7. **A manifest and extracted-package checks** — safe unique members, hashes, row counts, links and test portability. Make cross-archive links explicit with the paired archive/member identity; include any file described as delivered.

Standalone final archive SHA-256 values should be written to a small **DELIVERY_RECEIPT.json outside both archives** after sealing them, including their actual filenames, sizes, hashes and extracted-readback results. Each ZIP's manifest covers its payload, excluding itself; do not claim the archive contains its own final hash. These are delivery records, not a new result identity.

## Final reply to the owner

State, briefly:

- B8-01: actual recorded checkpoint cases corrected; annotations still annotations; any unresolved cases and why.
- B8-02: all 64 exports/labels checked; six dynamic policies and conditional exits correctly described.
- Preservation: no economic changes, no financial replay, no Core/rule/pin changes, no new dates, no ML, no commit/push (only claim what the evidence supports).
- Final focused test outcomes and what was not rerun.
- Normal UI route and new report revision; both archive locations and hashes.

Do not finish with only 'code fixed', 'tests passed', or a link to a local-only verification log. The two archives must actually exist, extract cleanly, and contain the evidence above. Stop when this isolated follow-up is complete; ML design is the next owner discussion, not automatic follow-on work.
