# Open decisions and known limits - IFVG dashboard repairs

Recorded September 23, 2026 at about 11:30 PM Chicago (CDT). The owner asked for the remaining constraints to be recorded before the repair work is committed. This file is the one place for what is still open after repairs R1-R8. How each repair was reproduced, fixed and verified is in [TASKS.md](TASKS.md). The assignment is [TASK.md](TASK.md). Summaries of current behavior are under "IFVG dashboard repairs" in [ARCHITECTURE.md](../../ARCHITECTURE.md), [ML_TRAINING_WORKBENCH.md](../ML_TRAINING_WORKBENCH.md) and [pipeline_state.yaml](../pipeline_state.yaml).

Recording these items changed nothing. No saved draft, plan, approval, result, export or engine pin was edited, and no study, preparation or worker was run.

## Owner decisions still needed

Each decision blocks only the item it names.

1. **R5: the replacement minimum for one saved search.** Draft `9aa2072d…` ("Parent staleness … (copy)") saves `steps.benchmarks.strategy_gates.min_independent_days` as 2050.0, inherited from frozen draft `caca5ab4…`. That study evaluates 107 trading days (after 10 warmup days), so no configuration can pass. Review, the approval request, charter validation and worker entry now refuse it in plain words. The saved value is never clamped or replaced. Needed: the value the owner wants for this draft (107 or fewer for its current dates). The copy has not been run. The frozen original `caca5ab4…`, its completed search, charter `79485ce6…` and approval `8369bbf0…` keep 2050, and the historic charter still loads.
2. **R7: S0_D80_W1_P1 as the starting point of the other study types.** In both apps, a new Evaluate study starts on S0_D80_W1_P1 when the verified daily-close package is on the computer and re-resolves to its section hash (implementation-verification studies excepted). It is the legacy profile with its 10 saved values, section hash `86261cc9…`. Compare, Search, Prop feasibility, Strategy across firms and Full workflow still start on the legacy baseline, as before. The legacy warning shows wherever that baseline is selected. Starting those types on S0 needs the worker to treat a configuration that is not in the registry as the baseline child, which changes research semantics. Registering S0 as a profile instead would rename its identity (to `74b81e47…`). Needed: whether either change is authorized.
3. **R7: S0 as the Trade review and context-research default.** That default needs prepared S0 context data, which needs a run that is not authorized. Until then the legacy baseline stays the default there. The screen shows the warning and says why S0 is not offered. Needed: whether to authorize that run.
4. **R8: which stored data block to use.** Local market data is stored in two blocks: December 2, 2021 to March 10, 2022, and June 2, 2025 to June 10, 2026. The blocks are a little over three years apart. A range across the gap can be chosen, but the screen warns that the strategy would carry its state across the gap. Needed: which block the next longer study should use.
5. **R8: two market closures.** The registered trading calendar covers 2026 only. Two older days are left out as closures, based on their stored files:
   - December 24, 2021 (Christmas observed): the file holds 8 records, all from the previous day.
   - December 25, 2025 (Christmas Day): the file holds only the evening reopen, which belongs to the next trading day.

   The registered calendar was not changed. Needed: confirm both closures.
6. **R8: preparing days before 2026.** Prepared study inputs exist only for 2026. Preparing earlier days needs its own authorization and a separate cache location, because the current cache layout would overwrite the prepared 2026 inputs. Needed: whether to authorize it, and the storage location.
7. **R8: whether stored days before 2026 count as already authorized.** The owner authorized extending the window to the earliest locally stored, already-authorized market data. Verification and authorization records exist only for 2026. The repair nevertheless opened the window at the earliest stored file (December 2, 2021; first evidence day December 14, 2021), treating the stored pre-2026 files as covered by that authorization. That is an assumption, not a recorded fact. Running such a range stays blocked either way (see the next section). Needed: confirm, or name the earliest date to allow.

## Known limitations

- **R8: ranges before 2026 can be saved and checked, but not approved or run.** Approval is refused when a range's first replayed day, warmup included, is before January 1, 2026: "Study inputs for dates before January 1, 2026 have not been prepared …". For example, January 2-30, 2026 first replays December 22, 2025. Verification and authorization records cover 2026 only, so days before 2026 have none. The wider window accepts those days only because their stored files exist. The earliest evidence day is December 14, 2021, the first day with ten stored warmup days before it. June 11, 2026 onward is still refused at every layer.
- **R1: a saved funded comparison the running engine cannot represent opens read-only.** An example is the half-exit rule under the default pinned engine. The page shows the saved settings, the real configuration count and any exactly matching saved plan. To edit, approve or run it, start the dedicated app with the research engine: `python scripts/run_ifsm_research_ui.py --research-core <research checkout>`. The shared pin is not changed.
- **R3: funded trade review.** Movement inside a minute is not drawn. Configurations that were not in the verified strategy study have no gap zones; the caption says why.
- **R6: the first listing still takes about four minutes.** The route is Main app → New study → More study types → Feature and model study. It now verifies each saved envelope once per listing, not once per child, and reuses the listing across reruns while the store is unchanged.
  - Before: 524 to 560 seconds, every time.
  - After: about four minutes the first time (at most 4 minutes 51 seconds in the browser), then under 3 seconds.

  A faster first load needs a storage or index change, which was deferred.
- **R6: all 356 saved children list as unavailable under the current pinned engine.** This is a separate compatibility limitation. The repair did not cause it or hide it.

## Changed behavior that makes older descriptions out of date

- **Research dates use start and end date pickers.** The one-date-per-line text box is gone. Any description of typing evidence dates one per line is out of date; TASK.md and SOURCE_OBSERVATIONS.md describe it only as the observed problem. Saved date lists stay byte-identical, including lists the pickers cannot show.
- **The gap-invalidation rule is saved only on an owner edit.** Before this task, opening a saved Evaluate draft without the rule added the inherited rule, and autosave wrote it. Drafts `f3099b0b…` and `708c133a…` are examples. Now opening returns the saved settings unchanged. The rule is written explicitly only when the owner changes a fixed setting, and a missing rule still resolves to the original rule. Two September 18 tests were updated to expect this.
- **A saved draft keeps its baseline hash from the earlier engine** until the owner clicks "Update the saved baseline to the current engine".
- **Not reproduced (R4):** the "no overnight holding" wording was already correct.

## Pre-existing test failures

The final full suite ran on the final code on September 23, 2026, from 9:37 PM to 10:14 PM Chicago, with the installed pinned Strategy-Core. Result: 3,540 passed, 5 failed, 2 skipped. All five failures fail identically on the recorded pre-task tree. They lie outside the repaired areas, and this task did not fix them:

- `tests/agents/data_infra/ifvg/test_catboost_bundle_model.py::test_frozen_m0_m3_catboost_lane_is_byte_and_identity_unchanged` (frozen model bundle hash `f108eb34…`; expected `e9cd1be9…`)
- `tests/agents/ifvg_search/test_htf_cap_experiment.py::test_exact_four_effective_profiles_and_distinct_cap_identities` (`A_cap1_opp80_control`: the full effective configuration differs from the baseline)
- `tests/agents/test_ifvg_capture_scheme.py::test_default_tags_regression_locked`
- `tests/agents/test_ifvg_capture_scheme.py::test_default_times_round_trip_to_canonical_tags`
- `tests/agents/test_ifvg_capture_scheme.py::test_list_profiles_default_first_and_upsert`

The three capture-scheme tests fail on the default capture tag: `92b4ab1709c2e164` instead of the expected `035d9e14ff3276ed`. A later full suite should expect these five until they are fixed separately. Receipts: `test_receipts/full_suite/suite.log` and `test_receipts/review_fixes/pre_existing_*` in the internal engineering store.

## Read-only search incident

On September 23, 2026, during the investigation between about 6:30 PM and 8:05 PM Chicago, a read-only helper ran one recursive text search over `data/` with no file filter. It ran for about two minutes before it was stopped. It may have opened raw market-data files under `data/databento/NQ`. It cannot be ruled out that it reached days dated June 11, 2026 or later. It printed nothing and wrote nothing, and no study, preparation or dataset used it. The rule since then: never search `data/` or `data/databento` recursively.

## Where the evidence is kept

- [TASKS.md](TASKS.md): the ledger, committed with this folder.
- The internal engineering store, `Claude-Quant-Lab-Research-Artifacts/ifvg-dashboard-repairs-20260923/`: a folder beside this repository, not in git. It holds:
  - the pre-task baseline and file hashes;
  - investigations and isolated copies;
  - test receipts and full-suite logs;
  - reviewer reports and scripts;
  - 29 internal screenshots.
- The handoff package, `reports/ifvg_dashboard_repairs/ifvg_dashboard_repairs_20260923_v1/`, and its `.zip` (1,099,611 bytes, SHA-256 `f46f1e6a…f8ed`). It holds REPAIR_REPORT.md, REVIEW_FINDINGS.md, validation_summary.json, the check tables, SCREEN_EVIDENCE.md and 16 screenshots. `reports/` is ignored by git: the package stays local and must never be committed.
- The owner's original package: [TASK.md](TASK.md), [SOURCE_OBSERVATIONS.md](SOURCE_OBSERVATIONS.md), [README.md](README.md), `evidence/`, the original TASKS.md and `.claude/skills/fix-ifvg-dashboard/SKILL.md`, listed in `PACKAGE_MANIFEST.json` as delivered, before any work. TASKS.md has since become the living ledger, so its manifest hash no longer matches; the other delivered files are kept byte-exact in git (`.gitattributes`), so their manifest hashes still verify.
