# HARDENING-BACKEND — Adversarial Review (verbatim merge of both independent read-only reviewers)

Reviewers ran on the complete working tree after the midpoint full suite (2078 passed, 0 failed under warnings-as-errors). Both were relaunched once after a session rate limit terminated the first pair before either wrote a report; the reports below are the complete relaunched outputs, unedited. Dispositions are in `ADVERSARIAL_REVIEW_RESOLUTION.md`.

---

# Reviewer A — semantic authority, immutable evidence, access safety, Phase 3 / Phase 4 contracts (`_review_authority.md`)

# HARDENING-BACKEND — Adversarial Review A (semantic authority, immutable evidence, access safety, Phase 3/4 contracts)

Reviewer A (read-only; relaunched after the rate-limit interruption). Scope: `store_namespace.py`,
`supersession_chain.py`, `owner_decision_lock.py`, `owner_decisions.py`, `authorization.py`,
`verification.py`, `charter.py`, `executors.py`, `child_replay.py` (slice preflight), `pipeline.py`
(S00), `identities.py` (registry), `trading_calendar.py`, `verification_window.py`,
`seed_production.py`, `bounded_verification.py`, the five scripts, the regime consumers, and the
listed tests. Probes were run as throwaway scripts under `%CLAUDE_JOB_DIR%\tmp\reviewA\`
(`probe_a.py` → P1/P2, `probe_b.py` → P3/P5/P6); exact outputs are quoted below.

## Verdict

The semantic-namespace, immutable-chain and witness designs hold under every rollback / forgery /
crash probe I ran, the Phase 3 and Phase 4 contracts refuse before any path with typed reasons, and
access safety is affirmed. **No blocker.** Two MAJOR findings: (RA-01) the plan's witness rule is
NOT applied at the real MBP-1 diagnostic seam, which still authorizes over a pathname
(`search_test/v1`) plus the run envelope's own fields and never binds the store namespace or the
current supersession head; (RA-02) on Windows the liveness lock's `release()` silently fails when
any other handle has the lock body open (a polling waiter, or another process), leaving an orphaned
lock owned by a LIVE pid that is — correctly — never reclaimed: under contention the owner-decision
lock wedges until the holder process exits (reproduced deterministically). Four MEDIUM and five
MINOR findings follow. Every item is fixable inside this release.

## Findings

### RA-01 — MAJOR — the real MBP-1 diagnostic seam lacks the namespace + head-witness binding
`src/alpha_lab/agents/data_infra/ifvg/features/mbp1_coverage_diagnostic.py:203-300`
(`assert_verification_namespace`, `assert_diagnostic_authorized`) and
`scripts/ifvg_mbp1_coverage_diagnostic.py:115-130`.
Observed: the real path gates on `tuple(root.parts[-2:]) == ("search_test", "v1")` (a PATHNAME
rule — authority-defining at this seam, not defense in depth), then verifies the run envelope's
allowlist / hash / policy / coverage matrix and calls `register_program_allowlist`, and then
constructs real source paths (`config.data_dir / "NQ" / day / "mbp1.parquet"`, script line ~172)
under `VerificationReplayPolicy`. It never loads the store namespace (`require_store_namespace(...,
expected_class="test")`), never checks `authorization.store_namespace_id`, and never calls
`assert_head_witness_current` / `assert_authorization_bound_to_store`. Plan §4.2: "The same witness
rule applies to strategy, MBP-1, regime, prop, publication, seed-production, and verification
authorizations". Consequence: an UNMARKED directory whose path ends in `search_test/v1` (or a store
whose head has moved since the owner signed) can run the real diagnostic and register the program
allowlist — exactly the F-11 path-derived authority the release removes elsewhere.
Fix: in `assert_diagnostic_authorized` (before `register_program_allowlist`) call
`assert_authorization_bound_to_store(root, store_namespace_id=authorization.store_namespace_id,
supersession_head_witness=authorization.supersession_head_witness, expected_namespace_class="test")`
and `assert_namespace_deployment_coherent(root, namespace)`; keep the pathname check as defense in
depth. Add a test: an unmarked / research-class / moved-head store refuses before
`register_program_allowlist` and before any path factory call.

### RA-02 — MAJOR — `OwnerDecisionLock.release()` silently fails under a concurrent reader (Windows); the lock then wedges for the holder's lifetime
`src/alpha_lab/agents/data_infra/ifvg/search/owner_decision_lock.py:383-388` (`release`), `:340-341`
(reclaim unlink), `:360-368` (`refresh` → `os.replace`), `:260-268` (`_read` swallows `OSError`).
Observed (probe P3ix, deterministic): holder acquires; another handle opens the body for reading;
`holder.release()` returns normally but the file remains — `lock_left=True
still_exists_after_reader_closed=True`. Windows refuses `os.unlink` while any handle is open
(sharing violation → `PermissionError`), and `release()` suppresses it. Probe P3i (six racing
writers in one process, 50 ms polling): `held=1 max_overlap=1 lock_left=True` — the first holder's
release collided with a waiter's `_read()`, the orphan carries a LIVE pid with a fresh heartbeat, and
the other five writers timed out with `lock_held_by_live_holder` (a correct verdict about a wrong
lock). Cross-process waiters read the file the same way, so a second owner-decision writer on the
same machine can wedge the first process's lock until that process EXITS (then dead-pid reclaim after
the heartbeat timeout). No overlap was ever observed (mutual exclusion holds); this is a liveness
defect. `refresh()`'s `os.replace` and the reclaim unlink are exposed to the same transient error
(refresh would raise an UNTYPED `PermissionError` mid-publication).
Fix: retry `os.unlink` / `os.replace` on `PermissionError`/`OSError` with a short backoff (e.g. 20 ×
25 ms); if the release still fails raise a typed `OwnerDecisionLockError("lock_release_failed")`
(never silent); make `_read()` retry once on a transient `OSError`; add a test that holds a read
handle open across `release()` and asserts the lock is gone.

### RA-03 — MEDIUM — research-catalog activation does not re-verify the charter's bundle against the CURRENT head
`src/alpha_lab/agents/data_infra/ifvg/search/pipeline.py:2943-2999` (`activate_pipeline_result`).
Observed: activation re-derives the publication gates but never reloads the charter and never calls
`assert_authorization_bound_to_store` for `charter.payload.owner_authorization`. The bundle's witness
is verified at `save_charter` and at S00 launch only; a supersession that lands between the launch
and the activation (revoking a decision the charter relied on) does not stop the research-catalog
activation. Plan §4.2 lists "publication" among the authorizations the witness rule covers.
Fix: in `activate_pipeline_result`, load the charter (`state["search_charter_id"]`) and refuse unless
its bundle is bound to the store's namespace and the CURRENT head (`AuthorizationError` →
`PublicationError`); verification-scope stays refused first.

### RA-04 — MEDIUM — the runner hard-codes three of the eight "also prove" proofs
`scripts/ifvg_bounded_verification.py:196-206` (`proofs` dict).
Observed: `different_bytes_fail_closed`, `missing_or_corrupt_manifest_fails_closed`,
`corrupt_sidecar_fails_closed` are literal `True` with an evidence ref naming test modules;
`identical_bytes_reuse` is `all_reused` (a property of the second attempt, not a byte-level probe).
`R1BaselineGatePayload.passed` therefore partly encodes an assertion about the test suite rather than
evidence gathered on the fixture (plan §6.2 "Also prove … identical existing bytes are a pass/reuse;
same semantic ID with different bytes, missing/corrupt manifest, or corrupt sidecar fails closed").
Also `first_attempt_gates` and `second_attempt_gates` are the SAME final S15 result (the first
attempt's gate report is not captured before the second run).
Fix: capture attempt 1's `search_results` envelope (its `control_flow_gates`) before attempt 2; run
scratch-copy probes on the fixture's executed-trade table entry (copy the entry to a scratch store
under the evidence dir: identical bytes → `load_executed_trade_table` OK; flip one byte of the
sidecar → typed refusal; delete/corrupt the manifest → typed refusal) and set those proofs from the
outcomes; keep the test-suite refs as supplementary evidence.

### RA-05 — MEDIUM — the preflight's seed step swallows every exception into `seed_profile_mismatch`
`src/alpha_lab/agents/data_infra/ifvg/search/bounded_verification.py:361-366`.
Observed: `except Exception as error: raise _refuse("seed_profile_mismatch", …)` around
`seed_loader(...)`. A programming error, a corrupt manifest (`SidecarLoadError`), a missing entry or
a sandbox-refused pickle all surface as "seed profile mismatch" — a typed but WRONG reason (the
namespace / sidecar reasons are lost), and `SeedSnapshotError` for the pytz case (DEV-HB-18) would be
misreported as a profile mismatch.
Fix: catch `(SearchStoreError, SeedSnapshotError, StoreNamespaceError)` only, map `SidecarLoadError`
to a new typed reason (e.g. `seed_snapshot_unverifiable`) and let anything else propagate.

### RA-06 — MEDIUM — `synthetic_owner_decision_fixture` initializes namespaces from PRODUCTION code
`src/alpha_lab/agents/data_infra/ifvg/search/owner_decisions.py:~838-850`.
Observed: the fixture calls `initialize_test_namespace(root)` on any UNMARKED root (refusing only
research-LOOKING paths). It lives in `src`, is importable by the promotion CLI's process, and would
silently mark the real `data/ifvg_datasets/search_test/v1` store as `test` if ever pointed at it —
bypassing the explicit operator `init` the plan requires ("A one-time explicit migration command
initializes the namespace after showing the operator the intended class"). The class is correct
(`test`) but the initialization would be implicit.
Fix: move the auto-initialization into `tests/agents/ifvg_search/namespace_fixture.py` (the test
helper) and make the production fixture REQUIRE a marked `test` namespace; or at minimum refuse when
`root` resolves inside the repository's `data/` tree.

### RA-07 — MINOR — seed production proves "no prohibited outputs" by construction only
`src/alpha_lab/agents/data_infra/ifvg/search/seed_production.py:734-896`.
Observed: the runner writes exactly `seed_snapshots` + `seed_production_runs` (structurally), but
no post-condition checks that no other store directory gained an entry during the replay; the
guarantee rests on `build_ifvg_v2_capture` not persisting (true today). The Quant-Lab / Strategy-Core
source identities (git reads, not source paths) are computed before the authorization loads —
harmless, but the "verify EVERYTHING before any source path" docstring should say so.
Fix: snapshot the set of store directories (and their entry counts) under `root` before the replay
and refuse (typed `prohibited_output_written`) if anything but the two permitted stores changed;
document the identity reads.

### RA-08 — MINOR — `executors._verification_run_envelope` picks the FIRST catalogued run
`src/alpha_lab/agents/data_infra/ifvg/search/executors.py:~55-70`.
Observed: `state.verification_run_ids[0]`; with two catalogued verification runs the executor binds
the first regardless of the charter's allowlist / pipeline id (it then compares the allowlist and
profile, so a mismatch refuses, but the selection is positional). The namespace + witness check
(this release) protects authority; the selection is a usability/ambiguity issue.
Fix: select the run whose `pipeline_semantic_id` equals the semantic id being launched; refuse when
more than one matches.

### RA-09 — MINOR — the `test`-class deployment check is not applied at every real seam
`src/alpha_lab/agents/data_infra/ifvg/search/authorization.py` (`assert_authorization_bound_to_store`)
and `pipeline.py:1080-1096` (S00 full-development branch).
Observed: `assert_namespace_deployment_coherent` (a `test` namespace under a research-looking path)
runs in `executors.real_verification_context`, the preflight and the seed runner, but not in
`assert_authorization_bound_to_store` itself, so `validate_verification_run(store_root)` and the S00
full-development branch accept an incoherently deployed store. Defense in depth only (authority is
semantic), hence MINOR.
Fix: call `assert_namespace_deployment_coherent` inside `assert_authorization_bound_to_store`.

### RA-10 — MINOR — the head/namespace atomic replace shares RA-02's transient-error exposure
`src/alpha_lab/agents/data_infra/ifvg/search/store_namespace.py:191-195` (`_atomic_write_text`).
Observed: `os.replace` of the head while a concurrent reader holds `SUPERSESSIONS.head` open raises
`PermissionError` on Windows; in `publish_supersession` that is the documented "failure before step
4" (orphan record, idempotent retry), so correctness holds — but the error is untyped.
Fix: retry `os.replace` briefly and wrap a persistent failure as `StoreNamespaceError` (new reason
`supersession_head_publish_failed`).

### RA-11 — MINOR — the shortlist's "highest lifecycle/candidate/decision/trade coverage" entry uses a tuple that is not the ranking order
`src/alpha_lab/agents/data_infra/ifvg/search/verification_window.py:405-510` (DEV-HB-19/DEV-D-6).
Observed: the required entry maximizes `(trades, decisions, candidates, lifecycle rows, rank)` while
the §5.1 ranking is lexicographic over the registered order; the deviation is recorded and the
result is evidence, not a selection. No hidden score exists in `rank_logical_windows`
(`rank_key` = the registered order; ties by earlier start).
Fix: none required beyond keeping the deviation stated; consider labelling the entry's tuple in the
markdown so the owner sees the difference.

## Access-safety verdict: AFFIRMED

- `find data -type f -newermt "2026-09-02 00:00"` → 0; no `STORE_NAMESPACE.json` under `data/`;
  `data/ifvg_datasets/` holds only the pre-existing `context_pair_catalog_v1.json`, `context_views/`,
  `fsm_audit/`, `replay_chart/`, `replay_chart_catalog_v1.json`, `v2/`, `v3/` — no `search*` store.
- `register_program_allowlist(` call sites: only the two PRE-EXISTING real seams
  (`child_replay.run_baseline_verification_slice`, after validation; `mbp1_coverage_diagnostic.assert_diagnostic_authorized`);
  neither the shortlist, the seed-production lane nor the Phase 4 preflight registers anything (the
  preflight only READS an existing marker: probe P6h `rotated_window_refused`).
- No immutable store is listed by the new code (`iterdir`/`glob`/`listdir` hits: `store.py` stray-file
  check inside the store's own temp dir, and the pre-existing Trade-Lab journal loader); the catalog
  event log is the only index consulted (`executors`).
- No production `assert`; the only broad excepts are the lock's liveness helpers (never raise into
  the writer), the preflight seed step (RA-05) and the runner's table-load proof.
- Date literals in the new modules: the window guards (`2026-06-10/11/12`), the plan's June proposal
  constant, evidence notes for holiday sessions, and fixture instants — no path derives from them;
  June 11 and the sealed range are refused by `SeedProductionReplayPolicy` (P5g) and the preflight (P6d).
- The seed-production CLI builds an `IfvgCaptureConfig` OBJECT before `run_seed_production_chain`,
  which verifies the authorization before any day artifact / source path is resolved (the capture is
  only reached after `verify_seed_production_authorization`; test-proven with a monkeypatched capture).
- The Phase 4 runner refused `fail_before_path` on the real, run-less verification store without
  constructing a config, policy or path (`_load_run` → catalog + verified load → `PermissionError`).

## Probes run (exact outcomes)

- P1a unmarked research-looking path: class `None`; synthetic scope refused (defense in depth).
  P1b unmarked plain path: synthetic scope NOT refused (documented DEV-HB-3; no owner artifact can
  exist there — persist/load refuse `store_namespace_missing`). P1c a marked TEST store COPIED under a
  research-looking path keeps its id and class (`same_id=True`) and the synthetic scope is refused by
  the path check. P1d research class at a plain path refuses the synthetic scope semantically.
  P1e a research store whose namespace FILE is overwritten by a copied test file: the class reads
  `test` but the head refuses (`supersession_head_namespace_mismatch`) — the head/namespace pair is
  the bound unit; a full-store clone (namespace + head + chain) DOES carry authority elsewhere (P2k:
  witness current after `copytree`) — the stated trust boundary ("deleting or rewriting the entire
  store plus every external witness is outside the trust boundary").
- P2a newest record deleted → `supersession_record_unverifiable`; P2b head rolled back → structurally
  valid, witness refuses `supersession_head_shorter_than_witness`; P2c record rewritten →
  `supersession_record_unverifiable`; P2d head deleted → `supersession_head_missing` (consumer:
  "chain fails closed"); P2e orphan record: chain unchanged, witness current; a FORGED head adopting
  the orphan with a correct digest yields a longer valid chain and every earlier witness refuses
  (`supersession_head_witness_mismatch`) — the orphan's replacement must still be a verified owner
  artifact for `load_supersession_chain`; P2f divergent replay → `supersession_divergent_replay`;
  P2g crash before the head update → witness unchanged (2), retry advances to 3 with one record
  reused; P2h no lock left after publications; P2i unknown owner id → `store_entry_missing`
  (`SearchStoreError`, caught by every consumer); P2j legacy v1 head → `supersession_head_malformed`.
- P3i six racing writers: mutual exclusion held (`max_overlap=1`) but the lock was ORPHANED (RA-02);
  P3ix deterministic reproduction of the orphan; P3ii a live holder with a stale heartbeat is never
  reclaimed; P3iii dead pid reclaimed only after the heartbeat timeout; P3iv PID reuse reclaimed;
  P3v other host → `lock_holder_liveness_unknown`; P3vi malformed stale body → `lock_body_malformed`;
  P3vii token theft mid-publication → `lock_lost`, head not moved, thief's lock intact; P3viii a
  stale holder's release leaves the new holder's lock.
- P5a–P5i: verify passes on the exact authorization; wrong chain / inventory / profile / not-yet-
  effective / moved head / synthetic-in-research / copied-into-another-namespace all refuse with the
  typed reasons; the policy refuses June 11, the sealed range, a gap and a Saturday before any path.
- P6a–P6l (stub seed loader): a valid window passes; Sunday partition date → `date_domain_mismatch`;
  six days → `sixth_day_refused`; June 11 → `protected_or_sealed_date`; gap →
  `logical_days_not_consecutive`; synthetic marker → `synthetic_marker_refused`; foreign namespace
  witness → `supersession_head_witness_refused`; rotated marker → `rotated_window_refused`; wrong
  output root → `output_namespace_not_locked`; research-class canonical store →
  `store_namespace_refused`; Good Friday (04-02, 04-03, 04-06) is accepted by the calendar as
  consecutive logical days (refused only by the stub seed's first day); no chain record was created.
- P7 goldens + identity audit: `test_r61_fix_goldens.py` + `test_identities.py` → 20 passed; the
  seven golden identities are unchanged; re-minted identities are owner-decision artifacts
  (`store_namespace_id`), verification runs / refs, real charter bundles and attempt receipts — no
  persisted real artifact exists outside tmp roots.
- P8 calendar/shortlist: `is_logical_trading_day` = weekday ∧ not in the registered closures
  (`2026-01-01` only); `physical_partitions_for` = `(td−1 prev_utc_date, td utc_date)`;
  `session_bounds_utc` = `[td−1 18:00 ET, td 18:00 ET)` via `ZoneInfo("US/Eastern")` (DST-aware);
  `rank_logical_windows` sorts by the registered lexicographic key with earlier-start ties — no
  hidden score; the shortlist registers nothing (`register_program_allowlist_called=False`).
- Targeted suites on the current tree: see the line appended below.

## Plan-text ambiguities resolved

- "An unmarked existing store cannot authorize a write or real launch" is read as: owner-authority-
  bearing writes and real launches refuse; structural writes and synthetic-marker charters in an
  unmarked tmp root remain lawful because nothing there can carry authority (DEV-HB-3) — acceptable.
- "Refuses a missing, shorter, or different current head" is implemented as exact equality (a longer
  chain also refuses) — the conservative reading (DEV-HB-7) — acceptable; RA-03 asks for the same
  rule at activation.
- The plan's witness-rule list includes MBP-1 and publication; RA-01 / RA-03 record the two seams
  where it is not yet applied.

- Targeted suites on the current tree (store_namespace, supersession_chain, owner_decision_lock, owner_decisions, seed_production, trading_calendar, verification_window, bounded_verification (+script), authorization, verification): 104 passed in 7.67s

---

# Reviewer B — capacity, warning policy, sequential truth, pipeline integration, test adequacy (`_review_capacity_pipeline.md`)

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
