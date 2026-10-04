# Task B — questions and applied choices

No owner question is currently unresolved. Preparation and its registrations
are complete. Attempts 01/02 were interrupted for implementation repairs;
neither establishes a completed study. The replacement source freeze and exact
plan approval are complete. Attempt 03 completed S15 with a persisted end of
October 4, 2026 at 01:15:36 UTC, and the foreground worker exited 0. All thirteen
sequential children and 26 configuration/firm pairs finished, with flat-account,
canonical and independent-reference audits passing. The source CSV and final
cache audits passed, including the independent delivered aggregate/context
check. The single full-suite run completed with the five documented pre-existing
failures. B4 parity passed from its single capture and frozen A1 comparison.

Implementation, preparation, study, CSV reconciliation, full-suite and B4 checks
are complete as recorded here. The published delivery commit and final repository
status are recorded in `run_log.txt`; archive verification is recorded in the
external `logs/delivery_integrity.json` receipt.

| Attempt 03 identity | Verified value |
|---|---|
| Quant-Lab frozen commit | `8c46f80dc65485c70751785b8edcaf5dfc11c0d2` |
| Core frozen commit | `709487bc85ef82a259297c4ad6f3f198f4cabbcd` |
| Search ID | `519b5317ebde363e47081db22e3a022378403e2468155d1af0415cdb33290a3b` |
| Pipeline ID | `1e3018b11ed0130545aefda0aeca927d625f246d7054b0ba7847e3a5c299e37a` |
| Approval ID | `366c0e42c3fd27d89055cf31085a54684a3c28a9c41667c2352ebe8d4ff410e7` |
| Verified pipeline result | `9cbc991020f6eeed7f766de268934847ce8ec27501f4de382772d0b82282fa88` |
| Funded artifact | `1e94c2c695d0152068e78d67dd4a4df33ca7d23bb4d8d5632efe0e112b4eb924` |
| Tables artifact | `1d6205a7434f92650017a564c005e5d3316e88ca096ae42ff4c7e1fbd124540f` |

The exact plan and independent check retain thirteen configurations, 253
evaluation dates and ten explicit warmup weekdays. The joint correction checks
passed 99 focused tests and Ruff. The actual full-suite and B4 outcomes are
recorded below.

The actual source audit passed for eight aggregate CSVs and thirteen context
exports, including cash-event-month reconciliation and all four independently
identified roll dates; its receipt is external
`logs/b2_attempt03_source_csv_independent_audit_verified.json`. All 52 grouped
CSVs passed the separate
`logs/b2_attempt03_per_configuration_grouped_csv_audit.json` audit. The final
`logs/final_prepared_cache_preservation_receipt.json` check passed all 578 cache
files, preserving registrations, original preparation evidence and enriched
catalogs with zero raw reads.

The handoff assembly contains 108 files, including 73 source CSVs and the
required `stage_timings.csv`, bound by proof
`99189f482847e73d6b4a3f66ab17ac4a5755d65a89f7bd6c75afaf64402d7b11`.
Direct delivered aggregate/context CSV validation now passed in external
`logs/b2_attempt03_delivered_csv_independent_audit_final.json`, including the
required SHA-verified `stage_timings.csv`. Initial alias/checker attempt logs
remain preserved as external-validation history. B4 parity now passed.

The full suite ran exactly once on frozen Quant-Lab
`8c46f80dc65485c70751785b8edcaf5dfc11c0d2`, October 4, 2026 from
01:23:03.561491 to 01:57:04.336418 UTC: 4,153 passed, 4 skipped and exactly 5
documented pre-existing failures, with pytest exit 1. Pytest took 2,037.77 seconds;
the wrapper took 2,040.774 seconds. `test_run.log` is a byte-exact copy of the
completed source log; the external receipt is `logs/full_test_run_receipt.json`.
The failed node IDs are:

- `tests/agents/data_infra/ifvg/test_catboost_bundle_model.py::test_frozen_m0_m3_catboost_lane_is_byte_and_identity_unchanged`
- `tests/agents/ifvg_search/test_htf_cap_experiment.py::test_exact_four_effective_profiles_and_distinct_cap_identities`
- `tests/agents/test_ifvg_capture_scheme.py::test_default_tags_regression_locked`
- `tests/agents/test_ifvg_capture_scheme.py::test_default_times_round_trip_to_canonical_tags`
- `tests/agents/test_ifvg_capture_scheme.py::test_list_profiles_default_first_and_upsert`

B4 completed PASS from the single `python -B WORK/task_b_parity.py --phase after`
capture and frozen A1 `python -B WORK/compare_task_b_parity.py` comparison; both
exited 0. The comparison timestamp is October 4, 2026 at 02:02:01.911484 UTC.
The completed capture contains eight v2 tables with full-byte parity and twelve
v3 tables with exact retained schema, row order and values. The historical
profile hash remains
`e0f318732cb59d844ac14b5e3839862146e7da1f612f9884f767247f66dd39dd`.
There are no additional A1 exclusions, and forbidden access counters are zero.
The detailed evidence is `parity_report_b.md` and external
`parity/parity_result_b.json`.

The independent cached-only
[roll receipt](C:/Users/gonza/Documents/Claude-Quant-Lab-Research-Artifacts/ifsm-task-b-20261003/logs/independent_expected_roll_days.json)
binds this exact frozen source and approved replay. Its expected evaluation roll
days are June 16, September 15 and December 15, 2025, and March 16, 2026; no warmup
roll was flagged. It verified the 263 replay days and 526 replay cache
hashes/footers with zero raw/protected access. The actual source CSV audit and
separate post-study 578-file check have now passed against this inventory.

The first approved source was `fa35a974b3660d3a3fd06c1df03e907acaf2bf39`.
Its baseline child completed; its context-on child was running when the owned
worker was terminated on October 3, 2026 at 21:38:54 UTC. Its immutable outputs,
approval, atomic checkpoint and separate interruption receipt remain under the
external task workspace. This attempt is not a completed study. The confirmed
repairs use the existing fixed-target metadata fallback and register the
already implemented funded/table output envelopes. No owner policy changed.

The second approved source was `73bda927b82bc809505ed0f7bd8943047e3ea90c`,
with pipeline ID
`095e547d62027e1e3b27853f3fa1091b8bb7872d11dd51563d5759b5460bb59e`.
The owned worker was terminated on October 3, 2026 at 22:28:49 UTC for a report
provenance repair: baseline/context-on matched every compared execution column
except the profile-bound `entering_seed_hash`, which the report assertion had
treated as behavior. Its outputs, approval, atomic state and separate
interruption receipt remain preserved in the external task workspace. Its
archived plan does not authorize a replacement source. No owner policy changed.

The replacement implementation also corrects date attribution under the
existing decisions: points and roll attribution use Core's logical trading
day (Sunday evening belongs to Monday), and selected instrument/roll lookup
uses that logical day's catalog entry. EOD context retains its civil date;
cash retains the actual event's Chicago civil date. These corrections are
included in the frozen source, passed the focused checks above, and introduce
no new owner policy. Attempt 03's completed-study verification passed.

Preparation completed at 21:09:41 UTC on October 3: the new store prepared 151
of the 154 supplied weekdays; July 8, November 14 and November 20 remain missing.
The original 2026 registration retains all 138 preparation receipts and owns
114 weekdays; the new registration owns 151 weekdays. Their 578 cache files
were verified unchanged before and after the completed catalog-symbol
enrichment and cached boundary audit. This is preparation evidence, not a
post-study cache check or a completed study result.

| Question or pattern conflict | Applied choice |
|---|---|
| Historical warmup includes Sunday store dates in original 2026 evidence | Task B's supplied weekday inventory and explicit ten dates govern this new task: June 2–13, 2025. Physical Sunday tails remain provenance and do not add logical warmup dates. Historical identities remain unchanged. |
| Original 2026 cache seeds differ from the new 2025 final high/low state | Registered loading verifies each segment's original seed lineage and bytes. Reducer/context state continues across the ordered replay; it does not claim that old 2026 caches were built from 2025 seeds. |
| Contract names are stored in the source's `symbol` column | A separate index-only pass uses the verified `symbol` fallback and permits only same-ID null-to-known enrichment. Original preparation receipts and cache bytes remain preserved. |
| A Chicago civil date may contain normal and late holiday transitions | Each existing closure is assigned to its canonical logical 18:00 ET interval. Actual Core controller, funded ledger/engine and independent-reference tests cover normal, late and abbreviated boundaries. Calendars and economics remain unchanged. |
| Good Friday 2026 structural/holding difference | The existing accepted holding amendment supplies the 08:10 Chicago deadline; structural preparation remains unchanged. The calendars agree throughout 2025, so no new disagreement-based exclusion is introduced. |
| Prepared streams lack an exact scheduled-deadline candle on July 3, November 28 and December 24, 2025 | The cached-only audit records the observed coverage and its conditional limitation: a surviving position would require executable deadline coverage. These dates remain in the authorized replay. No exchange early-close claim, new calendar rule or date exclusion is inferred. |

The cash allocation and roll questions were resolved by decisions 20/21 as
recorded below. Roll-day points use the entry logical date; cash events use
their actual Chicago calendar date. Funded timed events use the existing
selected evaluation-calendar convention; processing timers continue across
gaps. Full calendar details are in `calendar_notes.md`.

## Historical start check and its resolution

Resolution of the historical start prerequisite: it was resolved on October 3, 2026
by the owner's explicit merge authorization. Both A1 merges were pushed and
their required ancestors verified; merge commits are recorded in `CHANGES.md`.
The earlier stop/check record is preserved below as history. Task B resumes
at B0 on `feature/menthorq-level-study` in both repositories.

The funded cash allocation question found during B0 is also resolved. The owner
ratified decisions 20 and 21 in this thread: no cash allocation to entry groups,
points-only NY/session/regime/slot/cell views, cash-event-month cash, a thirteenth
`ny_only` baseline configuration, and no roll-day exclusions. Both decisions are
appended in `decisions.md`; the exact study plan must contain thirteen rows.
The account-driven full stream is the funded source; no subgroup simulation or
filtered saved stream substitutes for it. At that point, work continued on B1/B2.

The following checks describe the earlier, resolved stop only. Their work-state
and blocked-status statements do not describe the current task.

Checked October 3, 2026 at 18:19:47 UTC (13:19:47 America/Chicago).

Task B is stopped before B0 under the first explicit stop condition. The Core
repository was found, but neither repository's freshly fetched `origin/main`
contains merged A1. Local `main` and `origin/main` match in each repository.

Binding instruction from the supplied `TASK_B.md`:

> Start from the merged `main` in both (A1 has been merged); if either `main` does not contain the A1 commits, stop and write `questions.md`.

| Repository | Fresh `origin/main` | A1 commit checked | `git merge-base --is-ancestor` exit |
|---|---|---|---|
| Quant-Lab | `2e3efd5d9921e1956d80144ee50d831956c7a360` | `f4ee8454651abafd97f0b3a1714e41465cf44049` (implementation) | 1 |
| Quant-Lab | `2e3efd5d9921e1956d80144ee50d831956c7a360` | `5d1e74e681e3e6dd1441d5a03423280a9ef33370` (final approval-table follow-up) | 1 |
| Core | `7c7111e398c083cf8e966e2e0c5aac8a41cc12c0` | `b062bfcf5a5209440a4b4c9d7c0ca2263f9f4cc2` | 1 |

Both `git fetch origin` commands succeeded. Git object inspection also finds
none of the following A1 paths on the respective `origin/main` trees:

- Quant-Lab: `src/alpha_lab/agents/data_infra/ifvg/menthorq_levels.py`,
  `src/alpha_lab/agents/data_infra/ifvg/menthorq_reporting.py`, and
  `docs/IFSM_MENTHORQ_LEVEL_CONTEXT.md`.
- Core: `src/strategy_core/strategies/ifvg_smc/menthorq_levels.py` and
  `tests/test_ifvg_menthorq_levels.py`.

A1 remains published on `origin/feature/menthorq-level-context`: Quant-Lab at
`5d1e74e681e3e6dd1441d5a03423280a9ef33370`, Core at
`b062bfcf5a5209440a4b4c9d7c0ca2263f9f4cc2`.

## Historical resolution required

Merge the complete A1 feature branch, including Quant-Lab's follow-ups, into
`main` in both repositories. Resume Task B after those merges are published.
The next action is a fresh fetch and verification of both merged main trees,
then creation of `feature/menthorq-level-study` in both repositories.

## Historical work state

The six required references were read in the specified order from the supplied
`ifsm_task_b_level_study.zip`. No Task B branch was created, no merge was
performed, and B0–B4 have not begun. No source code, study, approval, catalog,
prepared store, or market-data partition was changed or opened. Existing A1
work, the owner's `handoff.zip`, and the unrelated dirty Core worktree are
preserved. Only this stop record was added to the handoff.

The remaining handoff files are not produced because the task stopped at its
start prerequisite. Task B is incomplete; no study results or test/parity
claims are made.

## Historical continuation check

At 18:20:44 UTC on October 3, 2026, a second `git fetch origin` succeeded in
both repositories. The two `origin/main` commits remained unchanged; the
Quant-Lab final A1 follow-up and Core A1 implementation ancestry checks again
returned exit 1. The same start prerequisite remains unresolved. No B0–B4
work is authorized past this explicit stop condition.

At 18:21:22 UTC on October 3, 2026, the third consecutive check again fetched
both repositories successfully. `origin/main` remained at `2e3efd5` in
Quant-Lab and `7c7111e` in Core, the A1 ancestry checks returned exit 1, and the
A1 runtime modules remained absent from both main trees. The goal is blocked
pending publication of the two A1 merges, as required by Task B's start rule.
