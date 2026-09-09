# DRAFT — Verification-Fixture Authorization Package (Owner Review Required)

**Status: PROPOSAL. Nothing in this document is an authorization.** This
implementation task does not grant, fabricate, self-sign, or infer the
`VerificationAuthorizationRef` (kickoff §4). R1 acceptance remains
`blocked_verification_authorization` until the owner signs decisions **21**
(canonical allowlist, coverage-evidenced) and **R-5** (coverage sign-off +
the `VerificationAuthorizationRef` itself).

## 1. What the owner is being asked to ratify

1. **The one canonical ≤5-day real verification allowlist** — used by the
   ENTIRE implementation-verification program, never rotated per release
   (V3 P0-7; a marker file enforces this at runtime).
2. **The coverage-matrix evidence** backing that date choice
   (`COVERAGE_MATRIX_PROPOSED.json`, `WINDOW_COVERAGE_SCAN.json` — built
   exclusively from the already-authorized accepted artifacts
   `v2 143b510f…` and `fsm_audit 7e55ee89…`; **no raw-source discovery
   occurred**).
3. **The baseline seed snapshot production step** (see §4) and the resulting
   `seed_snapshot_id`.
4. The immutable `VerificationAuthorizationRef` binding: policy id
   `verification_fixed_allowlist_max5_v1` + approved allowlist hash +
   coverage-matrix artifact id + seed snapshot id + approver + timestamp.

## 2. Coverage evidence and the date decision

The plan's candidate window `2026-06-04…06-10` (5 trading days:
06-04, 06-05, 06-08, 06-09, 06-10) was scored against every alternative
derivable from the accepted 138-day dataset. Key facts (post-warmup):

| Evidence | Candidate 06-04…06-10 | Best-scoring window 2026-02-06…02-11 |
|---|---|---|
| Source partitions recorded | 5/5 days | 5/5 days |
| Setup lifecycle events | 1 | 125 |
| Entry candidates | 0 | 30 |
| Eligible decisions | 0 | 9 |
| Executed trades | 0 | 9 |
| Lifecycle paths covered | 1 of 7 scored | 4 of 4 funnel paths |

Monthly funnel totals in the accepted dataset (lifecycle/candidates/
decisions/trades): Jan 153/21/6/6 · Feb 202/40/11/11 · Mar 241/36/8/8 ·
Apr 126/35/8/8 · **May 0/0/0/0 · Jun 1/0/0/0**. The doc-default strategy
produced essentially no funnel activity in the entire exposed May–June
range — recency does not buy behavioral coverage here, which is precisely
why P0-7 demoted 06-04…06-10 to a *candidate*.

**Both lawful options, stated neutrally (the choice is the owner's):**

- **Option A — keep `2026-06-04…06-10`.** Maximum chain continuity to the
  development cutoff; the real slice then proves source access, mid-chain
  seed start, sequential replay, dual-drive audit neutrality, immutable
  save/reload/reuse, and access safety — while every candidate/decision/
  trade/label lifecycle path is covered by synthetic fixtures (the
  documented fallback in TEST_MATRIX §2 when no window covers all paths).
  Requires the long seed snapshot (chain through 2026-06-03, ≈131 replay
  days, ≈35 min).
- **Option B — adopt `2026-02-06…02-11`** (store days as listed in
  `WINDOW_COVERAGE_SCAN.json`). The real slice additionally exercises the
  candidate/decision/trade/label paths on real data (30 candidates, 9
  trades), and the seed snapshot chain is ≈19 replay days (≈5 min).
  Trade-off: the window sits mid-development-history rather than at the
  cutoff edge.

Sunday-partition note: composing a Monday trading day references the prior
Sunday-evening calendar partition. Under the cached-artifacts-only slice no
raw partition path is constructed; the `ReplayInputBundle` records composing
partitions as metadata (`relative_logical_partition_key="prev_utc_date/…"`).

## 3. Draft `VerificationAuthorizationRef` payload (fields the owner signs)

```json
{
  "verification_policy_id": "verification_fixed_allowlist_max5_v1",
  "approved_allowlist_hash": "<sha256 of the newline-joined sorted approved dates>",
  "coverage_matrix_artifact_id": "a4a4835a97e1e35bca3082ad6b5be777c7459a31f17c6769212485a4ed20e51e",
  "seed_snapshot_id": "<minted by the authorized snapshot production step, §4>",
  "approved_by": "<owner>",
  "approved_at": "<UTC timestamp>",
  "content_hash": "<sha256 of this payload>"
}
```

The `coverage_matrix_artifact_id` above is the computed envelope id of
`COVERAGE_MATRIX_PROPOSED.json` **for the candidate allowlist**; choosing
Option B (or any other window) re-mints the coverage matrix for that window
before signing. The proposed allowlist hash for Option A is
`allowlist_sha256(("2026-06-04","2026-06-05","2026-06-08","2026-06-09","2026-06-10"))`.

## 4. Seed snapshot production (authorized owner action, NOT executed)

No v2 per-day seed cache exists (v2 replays always re-drive the reducer), so
the profile-matching baseline seed snapshot must be produced by ONE
development-policy chain replay of `ifvg_v2_doc_default_fresh_static_1r`
through the day before the approved allowlist start, with
`final_day_exhausts_dataset=False` (a dataset-exhausted final day alters the
end seed — proven in `tests/agents/ifvg_search/test_child_replay.py`), then
saved via `save_seed_snapshot` (immutable, profile-bound, hash-verified).
This replay reads only permitted development dates under the existing
`DevelopmentReplayPolicy`; it is nevertheless a long real-data replay and is
therefore **left as part of the owner-authorized R1 acceptance run**, not
executed during implementation (kickoff hard constraint: no full-development
replay during implementation).

## 5. What was and was not executed during implementation

- Executed: coverage-matrix computation from already-authorized immutable
  artifacts (read-only), synthetic tests (119), fail-before-path negative
  tests. Protected/sealed counters: zero; June 11 and the sealed range were
  never constructed, listed, stat-ed, opened, or read.
- NOT executed: the real five-day vertical slice, the seed snapshot
  production replay, any full-development replay/feature build/model fit/
  prop simulation/search run.
