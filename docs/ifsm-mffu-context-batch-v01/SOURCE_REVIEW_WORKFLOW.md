# MFFU source-review ZIP

The source-review ZIP is a separate audit deliverable from the financial
result-review ZIP. Publish it after the 64-row result, result-review ZIP, final
Core patch, and focused test receipts are frozen. The publisher in
`src/alpha_lab/propsim/funded/mffu_source_review.py` does not run research.

Call `publish_mffu_source_review` with exact paths to the saved plan, approval,
and result envelopes, the saved `result.json`, and the already published
result-review ZIP. Supply Quant-Lab and task Core checkout roots and their base
commits, an external staging root under the task-owned Research-Artifacts area,
and a new ignored `reports/*.zip` output path. Supply explicit `repo_files` and
`core_files` lists for affected tests, documentation, fixtures, and any source
outside the plan-bound runtime list. The publisher automatically includes all
plan-bound Quant-Lab runtime files and all changed task Core `src` files. It
rejects data, dependency, cache, report, raw and vendor paths, and limits file
and package sizes.

The `evidence_record_path` must point to a small JSON record with `commands`,
`test_receipts`, `failures`, `corrections`, `environment`, and `import_identity`.
The command and test-receipt lists must be nonempty. Include precise focused
test commands and outcomes, any historical failures with reproductions, the
corrections made, Python and installed/imported Core identities, and the
handoff caption discrepancy. One handoff caption names `0283b…`; the verified
repaired reference plan/result bind `028b3ce4…`. The final task Core patch may
have a third hash after task edits; record all three roles clearly and bind the
final source to actual bytes. Pass supporting small text/JSON test receipts in
`receipt_files`. No market bars, vendor ZIPs, credentials or full result JSON
belong in this source package.

The `environment` object must name `python_version`, `python_executable`,
`platform`, `quant_lab_root`, and `core_root`. The `import_identity` object must
name `strategy_core_module_file`, its `strategy_core_module_sha256`, and a
`core_source` object with `base_commit`, `branch`, and `patch_sha256`. The
publisher checks that the module is beneath the task Core source checkout and
that these hashes match the frozen plan. Evidence and receipt files must be
small files outside both repositories.

The publisher checks the three content-addressed envelopes, completed result,
64 ordered worker dispositions, dispatch and reuse mappings, plan-bound runtime
hashes, task Core source identity, and the result-review ZIP's extraction
receipt. It writes selected source bytes, scoped tracked and untracked patches,
source bindings, worker mapping, evidence and a final per-member SHA-256
manifest. It creates the ZIP in external staging, extracts and reads every
member back against the manifest, moves it to the ignored `reports/` location,
then verifies the final ZIP again. `verify_mffu_source_review_zip` provides
repeatable readback with an external extraction root.

The frozen 64-intent plan records runtime source hashes as ordered
`[absolute_path, sha256]` pairs. The publisher and verifier check those pairs
without changing their saved representation and reject malformed or duplicate
paths. Saved worker dispatches carry the equivalent hashes as a mapping. The
worker-mapping check normalizes those two shapes only for comparison and hashes
each original saved dispatch, including both reused controls. The source-review
regression covers both shapes and rejects malformed or duplicate paths.

The task Core plan's `patch_sha256` covers its entire uncommitted `src` diff and
untracked source files under the base commit. The ZIP's scoped patch file
hashes are separate manifest receipts; they should not be presented as the
Core plan identity. The ZIP also carries the exact tracked `src` diff used by
the Core identity plus every untracked source byte, so its Core identity can
be recomputed after extraction. The complete selected source bytes and the frozen plan
bindings are the reviewable source record. The separately verified financial
result-review ZIP holds cash, trade and comparison tables.
