# QL-FSM-PROP-SEARCH-DASHBOARD — R6.1 Correction-Release Plan Document Set

**Feature:** `ifvg_prop_robust_config_search_v1` — correction release after R6 (V1 KMeans regime lane, commit `179a2c9`)
**Prepared:** 2026-08-28 · **Revision 3:** 2026-08-28 (owner plan-review corrections plus final contract-closure patch applied)
**Status:** PLAN — revision 3, ready for owner approval. **Not an authorization.** No code, tests, artifacts,
catalogs, seed snapshots, replays, MBP-1 feature builds, model fits, prop simulations, or source-data runs
were performed while producing this set.

## Relationship to the other folders

| Folder | Role |
|---|---|
| `..\..\FINAL-IMPLEMENTATION-PLAN-DOCS\` | The authoritative V1 package — **unchanged**; this set is subordinate to it and cites it |
| `..\..\IMPLEMENTATION-START\` | The kickoff process rules (tests-first, adversarial round, evidence folder, commit discipline) — apply to R5B.1 and R6.1 unchanged |
| `..\R6\` | The R6 evidence this set corrects (never edited) |
| `..\R5B.1\`, `..\R6.1\` | To be created by the implementation (GATE_SUMMARY, TEST_RESULTS, ADVERSARIAL_REVIEW(+_RESOLUTION), FILES_TOUCHED, DEVIATIONS, ACCESS_SAFETY_EVIDENCE, browser-smoke, raw outputs, DRAFT_OWNER_DECISION_PROPOSALS) |

## Reading order

| # | Document | What it contains |
|---|---|---|
| 0 | `R6_OWNER_AUDIT.md` | The owner's R6 audit (input) — verbatim |
| 0b | `OWNER_PLANNING_DECISIONS_2026-08-28.md` | The owner's four binding planning answers (input) — verbatim: R5B.1 in scope, CatBoost bundle rung, the seven-feature panel block, status gates |
| 0c | `OWNER_PLAN_REVIEW_CORRECTIONS_2026-08-28.md` | The owner's review of plan revision 1 (input) — verbatim: seven blocking corrections + smaller amendments |
| 0d | `FINAL_PLAN_CORRECTIONS_2026-08-28.md` | Final contract-closure rulings applied directly to revision 3 |
| 0e | `FINAL_CONSISTENCY_AUDIT.md` | Cross-document verification of the corrected package |
| 1 | **`R6.1_IMPLEMENTATION_PLAN.md`** | **The plan (revision 3)** — audit verification; owner decisions; two-commit release mechanics; fold-safe panel/regime/CatBoost contracts; exact promotion identity; physical/channel MBP-1 coverage; bounded Parquet prop evidence; pipeline stages; tests; evidence; open blockers |

## What happens next

1. Owner reviews revision 3 and approves it (or requests changes). No implementation-architecture assumption remains open:
   the prop-event sidecar is approved as a versioned immutable account-simulation change (D15).
2. Implementation proceeds as two release-scoped commits on the feature branch — `R5B.1` then
   `R6.1` — each with its own evidence folder and adversarial round; no push, no merge;
   `implementation_status: complete` / `acceptance_status: transitively_blocked_by_R1`.
3. The owner-decision proposals for regime defaults 25/28/29/30 (per protocol: candidate grain, 5m
   panel, 15m panel) are delivered as drafts for the owner's ratification workflow; the
   `VerificationAuthorizationRef` remains the first blocker for every real-data step (including the
   bounded MBP-1 coverage diagnostic and the five-day regime mini-run).
