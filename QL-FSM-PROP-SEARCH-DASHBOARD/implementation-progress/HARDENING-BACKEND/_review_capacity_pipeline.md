# Reviewer B — capacity implementation, warning policy, sequential truth, pipeline integration, test adequacy

Read-only adversarial review of the HARDENING-BACKEND tree (2026-09-02, relaunched after the
rate-limit termination; the midpoint full suite had already completed: 2078 passed under
`filterwarnings = error`). Scope: `propsim/event_detail.py`, `propsim/search_bridge.py`,
`ml/regime_stratified_prop.py`, `scripts/hardening_capacity_benchmark.py`, `pyproject.toml`,
`ifvg/dataset.py` (`concat_schema_aligned`), `search/pipeline.py` (worker policy, attempt receipt,
S00), `scripts/ifvg_pipeline_job.py`, `search/bounded_verification.py` (the report builders),
`scripts/ifvg_bounded_verification.py`, and the new/changed tests. No file was edited; no git
write; no real path; the capacity benchmark was NOT re-run (its evidence was audited instead).

## Verdict

**No blocker; no major.** The streaming writer consumes its iterable exactly once with no
whole-artifact index, the external DuckDB uniqueness check and the external summary aggregation
are real (memory-limited, spill directory, exact counts, canonical ORDER BY equal to the Python
key, row-group-aligned bytes byte-identical to `pq.write_table`, temp directories cleaned), the
benchmark measures the production paths in fresh subprocesses with a native RSS monitor and its
gate formulas are §4.4 verbatim, the warning rule is exact and the schema-aligned concat
reproduces the pre-deprecation result on every shape I threw at it, and the sequential-execution
refusal is typed and precedes job creation at every backend entry point. Two MEDIUM findings:
the bounded control-flow report reads a stage sidecar under the wrong name (so the S14
`children_skipped` evidence is never populated), and the B1 capacity projection is stated
without its dependence on events-per-path (the per-block bound is `path_block_size ×
events_per_path`; the benchmark fixes 200). Seven MINOR findings follow.

**Capacity-gate verdict: YES** — `CAPACITY_BENCHMARKS.json` agrees with the `.md`; 18 runs; the
output hash is identical across the rss run, its repeat and the tracemalloc run at every size;
`peak_rss_increase = PeakWorkingSetSize(after) − WorkingSetSize(baseline after imports)`
(baseline 0.196 GiB in every run — conservative); slope/projection/artifact/wall/Python-peak
formulas match §4.4 exactly (slope gate `1.25 × max(slope_250k→500k, 64 B/row)`; projection
`inc[1M] + max(slope_500k→1M, 0) × (max − 1M)` ≤ `min(6 GiB, 0.5 × min available)`); every
run started with ≥ 8.755 GiB available. The measured shape is the caveat of B-02.

**Access-safety verdict (my scope): AFFIRMED** — no real source path, no `data/` reference, no
date literal that derives a path; every temp directory is `tempfile.mkdtemp()` and removed in
`finally`; no store-ROOT listing in `executors.py`, `pipeline.py`, `bounded_verification.py`
(`scripts/ifvg_bounded_verification.py:102` lists the files of ONE exact store entry to hash
them — not a store listing); `find data -type f -newermt 2026-09-02` → 0 (checked by the main
agent's baseline; my probes wrote only under temp roots).

## Findings

### B-01 — MEDIUM — `bounded_verification.py:824`: S14 sidecar read under the wrong name; `children_skipped` evidence is silently empty

`_s14_component` calls `_stage_sidecar(root, entry, "regime_reports.json")`, but S14 persists
its regime record as `regime_stratified_reports.json` (`pipeline.py:2388`, record from
`regime_report_stage.build_reports`, which carries `children_skipped`). `load_json_sidecar`
reports a name the manifest does not declare as the lawful "not produced" state, so `regime`
is always `None`, `children_skipped` is always `{}`, and the `typed_skip_with_evidence` outcome
carries no typed-skip record — the §6.3 "explicit typed skip with evidence" claim degrades to
the sanitized explanation string. The synthetic fixture (no regime study) cannot detect this
(`test_bounded_verification.py` passes with the wrong name). Fix: read
`regime_stratified_reports.json`; add a test over a stratified regime run whose S14 record
carries a `children_skipped` entry and assert the component evidence carries it.

### B-02 — MEDIUM — the B1 projection depends on events-per-path, which the policy does not bound

The writer holds one path block in Python lists (17 columns × rows of the block); block rows
= `path_block_size (250) × events_per_path`. `EVENT_DETAIL_BUDGET_V1` caps total rows (10M) and
the block size (250 paths) but not events per path, so the per-row RSS slope and the 1.446 GiB
projection at 10M rows are for the benchmark's fixed 200 events/path (50,000 rows per block).
A production bootstrap walk with more events per path scales the resident block linearly
(≈ 850 MB of Python objects at 2,000 events/path) while total rows stay inside the budget.
The implementation IS block-bounded (memory ∝ events per block, never the artifact), so the
plan's structural requirement holds; the numerical claim should state its shape. Fix:
record `events_per_path` as a benchmark parameter in `CAPACITY_BENCHMARKS.md`, run one
additional worst-case shape (e.g. 1,000–2,000 events/path at 1M rows) or derive the per-block
bound analytically in the report, and consider a row-based flush guard in a future
versioned capacity policy (splitting a path block across partitions is not possible under the
current manifest contract, so this is a policy-version change, not a patch).

### B-03 — MINOR — `bounded_verification.py:717–731`: the fold outcome parses a sanitized explanation string

`_fold_component` derives the valid-fold count with `explanation.startswith("0 folds
constructible")` / `"; 0 valid"` / `re.search(r"(\d+) folds under .*?; (\d+) valid", …)`
because `fold_sample_adequacy.json` carries no `valid_fold_count` (its
`sample_adequacy_preview` has `expected_gate_outcome ∈ {pass, fail, no_valid_folds}` and
`per_fold`, and only for regime runs). The synthetic fixture exercises only the
"0 folds constructible" branch; the regex branch (a regime run: "N folds under …; M valid;
regime schedule …") is untested. Fix: persist a typed `valid_fold_count` / `fold_count` in the
S08 stage sidecar (or read `expected_gate_outcome` + `per_fold` when present) and drop the
regex; add a test over a regime run.

### B-04 — MINOR — `bounded_verification.py:683–699`: the panel `typed_insufficiency` branch is unreachable in practice

`materialize_context_bar_panel` never fails for lookback insufficiency — it materializes the
panel with per-row typed reasons (`insufficient_trading_day_lookback`,
`context_bar_panel_materializer.py:342`) — so on a five-day fixture the outcome is
`panel_materialized` and the typed-null counts are not recorded as evidence; the
`status == "failed" and "insufficient" in explanation` branch keys on a string that no
production path emits. Fix: record the panel validity reason counts (from the persisted panel
artifact or the S06 coverage sidecar) in the component evidence and remove the string match.

### B-05 — MINOR — the S00 full-development bound-bundle check has no pipeline-level test

`_stage_s00_validate`'s new `assert_authorization_bound_to_store` branch (real bundle charter
→ namespace + current witness before any stage) is covered only indirectly
(`test_authorization.py` tests the helper); the sole `full_authorized_development` reference
in the pipeline suites is a crafted publication-state test
(`test_pipeline_evidence_integrity.py:679`). Fix: a synthetic full-scope charter carrying an
owner bundle — unbound (unmarked store / stale witness) refuses at S00 before any stage runs;
bound proceeds to the readiness check.

### B-06 — MINOR — `worker_parallelism_refusal` returns `None` for `max_workers=0`

`WorkerPolicy(max_workers=0)` fails on `ge=1` before the sequential validator runs, so the
pydantic error carries no `UnsupportedWorkerParallelismError` and the extractor returns
`None`; the shim's `assert_supported_worker_parallelism(0)` does type it (`requested_workers=0`
refused). Cosmetic; `True`/`"1"` are accepted as 1 (bool/str → int), also cosmetic.

### B-07 — MINOR — the byte-budget refusal path of the summary is not independently tested

`test_attempt_temp_directory_is_cleaned_on_success_and_on_refusal` uses
`max_summary_rows=1, max_published_bytes=10` — the ROW budget refuses first, so the
`summary_path.stat().st_size > budget.max_published_bytes` branch (`regime_stratified_prop.py:925`)
and the empty-table byte branch (`:935`) are never exercised. A mutation dropping the byte
check passes the suite. Fix: a test with a generous row budget and a tiny byte budget.

### B-08 — MINOR — the writer's early `path_count` overrun is only reached through the ordinal-order check

`test_iterable_form_requires_draw_ordinal_order`'s "doubled" case declares `path_count=5` but
fails on the draw-ordinal check first; the `paths_seen > declared_paths` refusal
(`event_detail.py:~646`) has no direct test (an ordered stream longer than the declared
count). Mutation: dropping that early check passes the suite (the final `paths_seen !=
declared_paths` check still catches it after streaming — refusal before publication holds).

### B-09 — MINOR — a single oversized simulation is fully aggregated before the row-budget refusal

`build_stratified_prop_body` checks `summary_rows_total + count_rows > budget.max_summary_rows`
AFTER aggregating that simulation into DuckDB (bounded by the 512 MiB limit + spill, so memory
is safe), whereas R6.1 refused at the first partition that crossed the budget. Acceptable
(exact count, refusal before publication), but note the wasted work on a refused build.

## Probes run (all under temp directories; exact outcomes)

- **P1 streaming writer** — `iter(walks)` is the only traversal in the iterable form
  (`_pair_stream`); the sequence form materializes only what it was handed; no `list/tuple/
  sorted` over the iterable path; `paths_seen`/`rows_seen` verified against the declared
  counts after streaming; `_external_uniqueness_check` runs unconditionally for every
  publication with `SET memory_limit='256MiB'`, `SET threads=2`, `SET temp_directory=<mkdtemp>`
  and only `COUNT` queries (`fetchone`; no id list ever fetched); temp dir removed in
  `finally`; row groups `EVENT_DETAIL_ROW_GROUP_SIZE=65_536` (unchanged). Suites:
  `tests/propsim/test_account_event_detail.py` (13) + `test_event_detail_streaming.py` (8) →
  **21 passed** (byte-identical partitions to the R6.1 writer, the cross-block forgery caught
  externally and NOT in memory when the check is disabled).
- **P2 external aggregation** — `_RowGroupAlignedWriter` bytes == `pq.write_table(table,
  row_group_size=N)` for batch patterns `[1,1500,2,1000,954]`, `[3457]`, `[999,999,999,460]`,
  `[2500,957]` (N=1000, 3,457 rows, 4 row groups each): **identical**; `summary_order_sql()`
  order == `sorted(rows, key=_summary_sort_key)` on 11 adversarial rows (cluster 2 vs 10,
  NULL cluster, reasons `""`/`no_source_trade`/`Panel_pit_unassigned`, `Fee` vs `fee`, `C0`
  vs `c1`, `P1`/`p1_`/`p10`): **equal** (binary collation); one thread +
  `preserve_insertion_order` + ORDER BY over a total key (stratum is a function of cluster);
  row budget = exact `COUNT(*)` per simulation before its rows are written, final check after;
  byte budget on the written file before the bytes are read; intermediate schema typed
  (`INTERMEDIATE_SUMMARY_SCHEMA`); the cross-partition path repetition refused in SQL
  (`HAVING COUNT(DISTINCT partition_ordinal) > 1`); `StratifiedPropBody` fields / `detail`
  keys unchanged (+ `detail["summary"]["aggregation"]`). Suites:
  `test_stratified_prop_external_aggregation.py` (6) + `test_regime_stratification.py` (…)
  → green within the 74-passed targeted run.
- **P3 benchmark honesty** — fresh subprocess per (benchmark, size, mode); imports and the
  fixture set-up precede the baseline; B1 = production writer over a generator + store publish
  (`save_envelope_immutable` with the sidecar producer) + production reader; B2 = production
  `build_stratified_prop_body` over lazily generated partitions; gate formulas verified
  against `_evaluate` line by line; JSON ↔ MD consistent; 18 raw runs; all three hashes per
  size equal; min available RAM 8.755 GiB. Caveat B-02 (shape dependence); B1 events are
  slotted duck-typed objects (the writer reads attributes only; Pydantic construction cost is
  the walk source's, outside the gated path).
- **P4 warning policy** — the rule is `ignore:<exact message regex, colon as \x3a>:
  DeprecationWarning:sklearn\.linear_model\._logistic` (message anchored at the start; the
  warning is attributed to that module at `_logistic.py:456`); nothing else is ignored;
  `concat_schema_aligned` vs the silenced old `pd.concat` on 12 additional shapes (all-NA
  frame second; `Int64`; naive/tz datetimes vs None/NaT; int+float+None; bool vs NaN; absent
  middle column; all-NA object vs NaN; `category`; `uint32`; absent bool; `string` dtype) →
  **columns, dtypes and values identical, zero warnings**; goldens + `test_ifvg_dataset` +
  `test_ifvg_fsm_audit_contracts` + `test_hardening_warning_policy` green.
- **P5 sequential truth** — `WorkerPolicy.model_construct(max_workers=4)` smuggled into
  `ExecutionAttemptIdentity` is REFUSED (pydantic re-validates the nested model:
  `unsupported_worker_parallelism_v1`); `assert_supported_worker_parallelism`: `"1"`/`True`/
  `1.0` → 1, `"2"` → refused(2), `None` → refused(−1); the extractor returns `None` for
  unrelated errors and for `max_workers=0` (B-06); the shim refuses in `start`/`resume`
  before `job_dir.mkdir` and in `worker` before any store load (tests spy on `Popen` and use
  a nonexistent store root); `ifvg_search_job.py` has no workers concept; the UI slider
  still offers 1–4 (documented, out of scope); `pipeline_semantic_id` has no worker field.
- **P6 S00 seams** — the real verification branch passes `store_root`; the full-development
  branch requires the bound bundle + current witness before the stage list runs; no store
  listing in the scope; synthetic-marker charters cannot reach a `research` namespace through
  `save_charter` (semantic class + path defense); B-05 test gap.
- **P7 bounded report builder** — B-01 (wrong sidecar name), B-03 (string parse), B-04
  (unreachable branch); every outcome vocabulary is registered and `passed` is derived from
  the outcome class (a lying `passed` and an unregistered outcome are refused);
  `test_bounded_verification.py` (8) green.
- **P8 test adequacy — mutations NOT caught**: `_s14_component` sidecar name (B-01); the fold
  regex (B-03); the panel string match (B-04); dropping the summary byte-budget check (B-07);
  dropping the writer's early `path_count` overrun (B-08); removing
  `preserve_insertion_order` (determinism test likely still passes with one thread); the S00
  bound-bundle branch (B-05). Assertions that only check "no exception": none material — the
  aggregation suite compares values, order, bytes and cleanup; the streaming suite compares
  bytes and counts.

## Plan-text readings resolved

- §4.4 "prove event-ID uniqueness by the existing deterministic identity projection … if any
  source cannot prove uniqueness from its canonical key, use a disk-backed external uniqueness
  check": the writer cannot verify a foreign envelope's projection per row, so the external
  check runs unconditionally — the stronger reading (DEV-HB-10).
- §4.4 "hardening cannot pass on extrapolation alone": satisfied structurally (block/partition
  bounded, temp-cleaned) and numerically for the measured shape; B-02 asks that the shape be
  stated.
