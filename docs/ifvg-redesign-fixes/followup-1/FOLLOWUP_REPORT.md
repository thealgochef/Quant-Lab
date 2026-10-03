# IFVG Lab — closeout review follow-up 1: setup timing and two wording edits

September 25, 2026, evening (Chicago). This answers the owner's closeout review response,
saved verbatim as `../CLOSEOUT_REVIEW_RESPONSE.md`. It is a separate package: the delivered
closeout (`../closeout/`, `../closeout.zip`, `../closeout_source_review.zip`) is unchanged,
byte for byte, and still verifies against its manifest. The maintained definitions are now
`CALCULATION_DEFINITIONS.md` in this folder. Nothing was committed or pushed.

## 1. Outcome in brief

- **Item 1, setup timing: fixed and verified.**
  - Trade review's "How the setup formed" text dated a tap and a close through the opposing gap
    by their candle's *opening* minute, and showed them from that minute at a point in time.
    The setup key, the chart and the moment picker already waited for the candle's *close*.
  - Now all four use one instant per event: when it became known. For a tap or a
    close-through, that is its candle's close. The candle's opening minute is kept only as its
    name, as in "The 7:05 PM candle closes through it · 7:06 PM".
  - The fault was reachable on real data. In the saved study, 84 of the 374 funded trades of
    the two configurations with their own setup records had the owner's boundary. On all 84,
    the delivered page offered that minute as a moment and its text claimed the close-through
    before the candle had closed. After the fix, none do (`DEMONSTRATION.md`).
  - The defect predates the closeout. It is already in tree `811f4b3`, the closeout's own
    pre-task tree and the owner's supplied "before" version.
- **Item 2, wording: done.**
  - The Risk tab's historical-order sentence now names the fields the check compares, with no
    "identical".
  - Every living document now says the $133.51 against $126 difference is *reproduced by
    changing the seed*. The reference's seed was not recorded, so the cause is not proven.
    The fixed seed 20260923 is kept, and no seed search was made.
- **Item 3, evidence: complete.** The package holds:
  - the exact changed source and test identity;
  - the pre-fix failure, reproduced on an isolated copy of the delivered source (all
    ordinary assertion failures);
  - the post-fix focused results;
  - a readable before and at-completion demonstration.
- **Full suite: not run.** The focused checks the project requires (AGENTS.md) ran on the final
  source. The closeout's full-suite result (3,959 passed, 5 failed, 4 skipped) belongs to tree
  `76bb3b0c` and does not cover this change.

What each part is:

- **Code changes:** five source files (`review_panels.py`, `review_chart.py`, `firm_race.py`,
  `ifvg_lab_trade_review.py`, `ifvg_lab_detail_risk.py`), plus a docstring-only pointer change
  in `market.py`.
- **Tests:** one new file and three updated.
- **Documentation:** eleven existing documents, the owner's input saved verbatim, and this
  package.
- **The demonstration:** runs the application's own functions on saved records. It is not a
  financial rerun.
- **No new study, funded run or approval was made.**

## 2. Item 1 — completed-candle timing

### What was wrong

`review_panels.setup_steps` used `record.bar_open("tap_bar")` and
`record.bar_open("inversion_bar")` both as the printed time and as the point-in-time filter.
`setup_key`, `moments` and the setup chart used `bar_close`.

The strategy engine records a tap as a candle's high or low reaching the gap, and a
close-through as a candle's body close beyond the opposing gap. It records both at the
candle's close (`availability_ts_utc`) and saves no earlier observation, so neither is known
at the candle's open. The same code is in tree `811f4b3`, the closeout's pre-task tree and
the owner's supplied "before" version, so the closeout did not introduce it.

### What changed (code)

- **`review_panels.py`:**
  - New `SetupEvent(role, known_at, candle_open)` and `setup_events(record)`. A gap is known
    at its confirmation; a tap or close-through at its candle's close. `candle_open` only names
    the candle.
  - `setup_steps`, `setup_key` and `moments` read these events.
  - New `STEP_TIMES_NOTE`.
  - A candle event without a recorded close is never shown at a point in time and offers no
    moment. In full history it is listed with "(its close was not recorded)" and "—".
- **`review_chart.py`:** `setup_figure` draws the close-through's shading and number from the
  same event. The chart already used the close; now the key and the chart also agree when a
  time is missing.
- **`scripts/ifvg_lab_trade_review.py`:** the "How the setup formed" card shows "Each time is
  when that step became known: a gap when its third candle closed, a candle's touch or
  close-through when that candle closed."
- **Deliberate edge-case changes.** None of these occurs in the saved study. All 9,770
  setup records linked to its 13,923 funded trades carry both candle times and every gap
  confirmation (374 of them own records, 9,396 related context); the other 4,153 funded trades
  have no linked record. The count is in `validation_summary.json` (`record_completeness`).
  - A tap is listed with its higher-timeframe gap even when neither of its candle's times was
    recorded.
  - A close-through recorded with only its close is shown from that close.
  - A record without its higher-timeframe gap lists its tap, matching the moment the picker
    already offered.

### How it reads now

The owner's boundary case uses an own record. Its opposing gap is confirmed at 7:05 PM, the
minute the close-through candle opens; that candle closes at 7:06 PM; the entry is at
7:07 PM.

- **At the selectable 7:05 PM moment:** the text, the key and the chart show the opposing gap
  and not the close-through. The delivered text already listed "A candle closes through it ·
  7:05 PM".
- **At 7:06 PM:** all three show "The 7:05 PM candle closes through it · 7:06 PM".
- **In full history:** the tap reads "The 5:18 PM candle taps into it · Apr 12, 5:19 PM"
  (it was "Price taps into it · Apr 12, 5:18 PM").

`DEMONSTRATION.md` shows the before and after tables and a real case:

- **Configuration:** S0_D80_W1_P1 at MyFundedFutures, Account 1, entry January 15, 2:22 AM.
- **Setup:** opposing gap confirmed at 2:20 AM; the close-through candle opens at 2:20 AM and
  closes at 2:21 AM.

### A test expectation corrected, not relaxed

`test_trade_review.py::test_reference_trade_review_values` pinned the mock's times, 5:18 PM and
7:05 PM, which are the two candles' opening minutes. It now expects the saved record's own
closes, 5:19 PM and 7:06 PM, and asserts `record.bar_close` for both. The test also still
asserts that the record is related context. No saved trade or record was changed.

### Kept exactly

The actual April trade's entry at 7:07 PM, the default cursor at 7:10 PM, and the half exit at
7:10:27.251840803 PM (hidden at the cursor). Tested by `test_the_april_anchors_are_kept` and by
the existing A7 tests.

### Regressions added

In `tests/agents/ifvg_lab/test_setup_event_timing.py` (23 tests; synthetic records only):

- **The boundary:** immediately before completion (7:05 PM and one nanosecond before
  7:06 PM), at completion (7:06 PM), and the selectable earlier moment (7:05 PM, offered by
  the picker).
- **Taps:** the same checks at 5:18:59.999999999 PM and 5:19 PM.
- **Related records:** at 7:05 PM, 7:05:59.999999999 PM, 7:06 PM and the entry.
- **Every offered moment:** text, key, chart number and chart shading checked together.
- **Full history:** each step's time and each candle's name, in the steps and the key.
- **Missing completion evidence:** a close missing for either candle, a tap candle with no
  times, and a close-through recorded without its opening minute.
- **The rest:** a record without its higher-timeframe gap, the page card's text and times
  note, and the April anchors.

## 3. Item 2 — wording

### Historical-order check

`firm_race.check_original` compares three things:

- five saved summary figures as exact integers: net cash, payouts received, payout count,
  accounts bought and account costs;
- the number of trades;
- trade by trade in the saved order, each trade's net result (to the cent) and
  account-loss flag.

It compares no fill times, prices or quantities, account assignment, payout or failure times,
or setup lineage.

**Before** (Risk tab, TakeProfitTrader leader):

> Checked first: the saved order, replayed through the same simulator, gives exactly the saved
> result — $30,781.88 net cash, 13 payouts, 6 accounts, and all 114 trades identical. …

**After** (figures checked against the saved summary):

> Checked first: the saved order, replayed through the same simulator, matches the saved
> result's net cash ($30,781.88), payouts received ($31,393.88), payout count (13), accounts
> bought (6) and account costs ($612.00) exactly, and gives the same number of trades (114),
> each with the same net result and account-loss flag in the saved order. It doesn't compare
> fill times, prices or quantities, which account took each trade, payout or failure times, or
> which setup each trade came from. This historical-order check confirms the adapter
> reproduces the saved order's money and account losses; it doesn't show that other sampled
> orders, or a fresh account running the full strategy, would be modelled exactly.

Other changes from the review:

- **Failure message.** It shows money in dollars; it said raw cents. If the replay produced a
  different number of trades, it now says so instead of reporting "0 of N" matching trades
  (`Validation.replayed_trades`).
- **Where else this was fixed:**
  - `firm_race` docstrings and one test docstring;
  - definitions §14;
  - `ARCHITECTURE.md` and `ML_TRAINING_WORKBENCH.md`;
  - DECISIONS_LOG AC16, with P5.3 marked;
  - the redesign report's notice and its task-ledger row P5b.

### $133.51 against $126

- **Before** (closeout definitions §4 and AC12): "the difference is a seed difference within
  the same method."
- **After** (definitions §4): "changing only the seed under the same method reproduces the
  difference … The reference's seed was not recorded. This shows that the seed alone can
  produce the difference; it does not prove that a different seed is what produced the
  reference. The application keeps its fixed seed 20260923, and no further seed search was
  made."
- **Also changed:**
  - §17;
  - DECISIONS_LOG, with AC12 marked superseded in part and AC15 added;
  - the notices in `../handoff/FIX_REPORT.md` and the redesign report;
  - `../TASKS.md`, where F6 is now "Qualified/relabelled, verified".

## 4. Item 3 — evidence

### 4.1 Source identity

Details are in `source_identity.json` and `validation_summary.json`. The code diff is in the
separate `../followup-1_source_review.zip`.

- **HEAD:** 1b9d407 (no commit).
- **This follow-up's pre-task tree:** `ca30aafca8eb42ab6111804e81dff47a9a8bf3b6`. Its src,
  scripts and tests are identical to the delivered closeout source `76bb3b0c514b0d1fba2fcfec17beb1694943b6c7`.
- **Final source tree:** `41d2288af68bea7810c584579af3786028f3914d`.
- **Source-only identity**, which later document edits don't change:

  | Part | Object id |
  |---|---|
  | `src` | `0022d2235e98a58ddd2ffdcba75e89946feccfe8` |
  | `scripts` | `cc474e70f9e20abcc745962c22665db2cb7bc6b0` |
  | `tests` | `057161bcccf68e954bb5a1ae8e8c20c77f793b99` |
  | `pyproject.toml` | `9c0121cf9da558c31d48cb05788dc340fd44269e` |

- **Source and test files changed** (git blobs, before → after):

| File | Before | After |
|---|---|---|
| `src/…/presentation/lab/review_panels.py` | `da133375` | `a8b5536e` |
| `src/…/presentation/lab/review_chart.py` | `d2021420` | `c45a894f` |
| `src/…/presentation/lab/firm_race.py` | `f2cbe07f` | `e1c812c1` |
| `src/…/presentation/lab/market.py` (docstring pointer only) | `7f3077e2` | `814551e0` |
| `scripts/ifvg_lab_trade_review.py` | `e0abdeb2` | `d56039d4` |
| `scripts/ifvg_lab_detail_risk.py` | `ad92eced` | `2680a925` |
| `tests/agents/ifvg_lab/test_setup_event_timing.py` (new) | — | `d738df94` |
| `tests/agents/ifvg_lab/test_trade_review.py` | `e584ac02` | `6ed8a6c3` |
| `tests/agents/ifvg_lab/test_detail_risk.py` | `e8dc2541` | `50912162` |
| `tests/agents/ifvg_lab/test_firm_race.py` (docstring) | `a0ef8a57` | `3944127b` |

The source-and-test diff has SHA-256 `38c090a9…`. The files that changed after the source
freeze are listed in `validation_summary.json`; all of them are documents.

### 4.2 The pre-fix failure, reproduced on an isolated copy

The run was at 7:05 PM (`prefix_reproduction.json`).

- **Source:** tree `76bb3b0c` was extracted with `git archive` (src, scripts, tests,
  pyproject.toml; no data), and only the new regression file was added.
- **Imports:** `review_panels.py`, `review_chart.py`, `ifvg_lab_trade_review.py` and the
  `_record` fixture module were imported from the extraction, with git blobs equal to the tree.
  `SetupEvent` was absent.
- **Result:** 22 failed, 1 passed. The pass is the April-anchors retention check, which should
  pass before and after. All 22 failures are ordinary assertion failures, with no other error
  type.
- **The owner's boundary case** fails with `step_numbers [1, 2, 3, 4, 5] != [1, 2, 3, 4]` and
  `close_through_step True != False`. The key and chart fields already matched.

### 4.3 Post-fix focused results on the final source

The run was 7:08 – 7:11 PM. The source-only identity was the same at start and end, and equal
to §4.1.

| Step | Result |
|---|---|
| `python -m ruff check src tests scripts` | all checks passed (exit 0) |
| the new regression file | 23 passed |
| the directly affected files (test_ac_trade_review, test_trade_review, test_setup_records, test_review_fixes, test_detail_risk, test_firm_race, test_ac_risk, test_reference_values) | 230 passed |
| every Lab test plus the neighbouring funded and Lab-tab tests | 503 passed, 3 skipped (the three half-exit-engine tests) |
| those three in the half-exit research environment | 3 passed |
| `test_firm_race.py::test_every_saved_configuration_replays_exactly` (all 128 pairs) | PASSED |

**Mutation probes.** Twelve deliberate breakages were each applied to a scratch copy of the
final source, and the new tests caught all 12. They included:

- dating either event by its candle's open;
- falling back to the open when the close is missing;
- the chart shading from the open;
- dropping the "not recorded" form;
- removing the card's times note;
- naming the candle by its close;
- related records offering setup moments.

### 4.4 What did not run

- **The full suite.** AGENTS.md: "Do not run the full regression suite by default, especially
  for isolated UI or presentation changes … Once relevant checks pass, do not broaden testing
  without a concrete unresolved correctness concern." The focused set above covers every
  affected module and its neighbours. The closeout's full-suite receipt (3,959 passed,
  5 failed, 4 skipped, tree `76bb3b0c`) is not claimed for this change. The five
  long-standing failures and the recorded guard-teardown follow-ups are not newly fixed by
  this change.
- **No other study, funded run, approval or launch.**
- **Screens not re-captured.** The Risk tab's new sentence was not re-captured; its text is
  quoted in §3.
- **Superseded delivered screenshots.** In the closeout, S06–S08 (Trade review) show the
  superseded setup text and times, and S02, S03 and S09 (Risk tab) show the old
  "all 114 trades identical" sentence. They are kept unchanged as the delivered record. The
  changed Trade review screen was re-captured (§4.6).

### 4.5 Preservation and access

- **Preservation.** The 22,619 saved study, draft, approval, review, export and payout record
  files are byte-identical before (6:10 PM) and after (7:12 PM). The delivered closeout
  folder, its manifest and both ZIPs are unchanged.
- **Reads.** The demonstration's final runs read the saved funded result and the verified
  package's setup records (MANIFEST.json, run_context.json, configs.json and data/trades.csv,
  hash-checked). The first post-fix run (about 6:31 PM) resolved the package through
  `market.study_package_root`, which hash-reads every required package file, the stored
  one-minute bars included. Nothing was parsed or kept, and the bar file ends at the
  June 10, 2026, 4:00 PM close.
- **Not read:** no raw market data under data/databento, and nothing dated June 11, 2026 or
  later.
- **Reviewer disclosure.** One reviewer's broad search from the repository root may have
  touched files under data/ before it was stopped. Nothing from data/ was displayed or
  written. Details are in `validation_summary.json` (`access`).

### 4.6 Screens: the corrected Trade review, captured

These are captures of the running application, not synthetic renderings. The dedicated IFSM
research application ran from the final source tree on the redesign's isolated store copy
(127.0.0.1:8661, light theme), at 7:32 PM, and headless Chrome captured the whole page. No
save, approval or run control was pressed. The isolated review ledger is unchanged (SHA-256
`e861127e…`). The page text was checked for each capture.

| File | What it shows |
|---|---|
| `screenshots/R07_review_april_point_in_time_corrected.png` | The leader's April 12 trade (TakeProfitTrader, Account 6) at the default 7:10 PM cursor. The related setup context reads "The 5:18 PM candle taps into it", and the times note is under the steps. It replaces the closeout's S07 for this card. |
| `screenshots/R08_review_april_full_history_corrected.png` | The same trade in full history: "The 5:18 PM candle taps into it · Apr 12, 5:19 PM" and "The 7:05 PM candle closes through it · 7:06 PM". The minute-record line is unchanged. It replaces the closeout's S08 for this card. |
| `screenshots/R12a_review_member_boundary_2_20_am_before_completion.png` | Configuration S0_D80_W1_P1 at MyFundedFutures, Account 1 (its own setup record), entry January 15, 2:22 AM. The moment is "2:20 AM · the opposing gap has formed". The text lists steps 1–4, and the key and chart show 1 and 2. There is no close-through anywhere. |
| `screenshots/R12b_review_member_boundary_2_21_am_at_completion.png` | The same trade at "2:21 AM · the 2:20 AM candle has closed through the opposing gap". Step 5 reads "The 2:20 AM candle closes through it · 2:21 AM", and the key and chart show it as item 3. |

## 5. Independent review

Four read-only reviewers who had not written the change reviewed the frozen review tree
`0299c644` from 6:33 to 6:52 PM. Their lenses were timing correctness, test adequacy (with
mutation probes on scratch copies), wording accuracy, and scope and preservation. An
adversarial verifier then tried to refute each reviewer's findings.

- **Findings:** 33 in all, 0 Material, 18 Minor and 15 Notes.
- **Verdicts:** 26 confirmed, 7 partly confirmed and 0 refuted.
- **Dispositions:** every finding was fixed or recorded, as below.

| Id | Severity | Finding | Disposition |
|---|---|---|---|
| T1 (timing) | Minor | A tap with no recorded candle times was dropped from full history. | Fixed: listed with its gap, "(its close was not recorded)", "—"; tested. |
| T2 (timing) | Minor | A close-through with only its close got a key number but no chart marker. | Fixed: the chart reads the same event; number on the close; tested. |
| T3 (timing) | Minor | Package files cited before they existed; the definitions header's change list was incomplete. | Fixed: package produced; list complete. |
| T4 (timing) | Note | A never-known step keeps its number, so point-in-time numbering skips it. | Kept: numbers stay stable across moments (the existing rule); full history explains it. |
| T5 (timing) | Note | The invariant test did not render the chart. | Fixed: chart number and shading checked at every instant. |
| T6 (timing) | Note | CALCULATIONS.md still says "tapped April 12, 5:18 PM". | Fixed: the notice lists the timing change; §17 row. |
| T1 (tests) | Minor | The chart's shading was untested (a mutation survived). | Fixed: shading checked. |
| T2 (tests) | Minor | The full-history form of a missing close was untested. | Fixed. |
| T3 (tests) | Minor | No focused receipts yet; ledger placeholders. | Fixed: §4.3; `../TASKS.md` filled. |
| T4 (tests) | Note | The card's times note was untested. | Fixed. |
| T5 (tests) | Note | The key's wording was tested only with saved data present. | Fixed: synthetic assertion. |
| T6 (tests) | Note | Related records were not at every instant the docstring claimed. | Fixed: cases at 7:05:59.999999999 PM and 7:06 PM. |
| T7 (tests) | Note | In full history, a missing-close key item had no chart marker. | Fixed: marked on its candle, without shading. |
| T8 (tests) | Note | The pre-fix receipt checked two imported modules. | Fixed: four. |
| W1 | Minor | Living documents still said the saved order "replays exactly". | Fixed: ARCHITECTURE, redesign report notice, P5.3 and P5b marked. |
| W2 | Minor | The definitions header's change list was incomplete. | Fixed. |
| W3 | Minor | §16 generalised the missing-completion wording. | Fixed: edge cases stated precisely. |
| W4 | Minor | A trade-count mismatch was reported as "0 of N trades". | Fixed: says so; count named; tested. |
| W5 | Minor | "all 1 trade' net results" for one trade. | Fixed: no possessive; tested. |
| W6 | Minor | TASKS F6 still read "Fixed and verified" with "cause documented". | Fixed: "Qualified/relabelled, verified". |
| W7 | Minor | SCREENS.md, DATA_GAPS.md and market.py still pointed to the closeout copy. | Fixed for SCREENS.md and market.py; DATA_GAPS.md's notice concerns placeholders only, so it was left (verifier's reading). |
| W8 | Minor | The CALCULATIONS.md notice did not list the timing change. | Fixed. |
| W9 | Note | The success sentence used the caller's net figure, not the compared value. | Fixed: it shows the compared saved values; parameter removed. |
| W10 | Note | Four small wording points. | Fixed: §16 reworded, the chart now reads the event, §17 row, workbench list. |
| W11 | Note | Pending package files. | Fixed. |
| F1 | Minor | SCREENS.md pointer. | Fixed (DATA_GAPS.md: no change needed). |
| F2 | Minor | The "word for word" claim missed two edits. | Fixed. |
| F3 | Minor | Edge-case behaviour changes were undisclosed. | Fixed: §2 here, AC14, definitions §16. |
| F4 | Minor | The demonstration's first post-fix run hash-read package files, the bars included. | Recorded (§4.5); the script now opens only the setup-record loader; rerun. |
| F5 | Note | The validation plan should follow AGENTS.md and state it. | Followed: focused checks only; §4.4; §14 points to the focused run. |
| F6 | Note | market.py's docstring pointer. | Fixed. |
| F7 | Note | Two parentheticals in the CALCULATIONS.md notice. | Fixed. |
| F8 | Note | A reviewer's broad search may have touched data/. | Recorded (§4.5). |

The fixes were checked by the strengthened tests, the 12 mutation probes and the focused run in
§4.3. They were not independently re-reviewed.

**Completeness check.** A final read-only critic compared this package with the owner's request
sentence by sentence, from about 7:17 to 7:31 PM, and a verifier checked its findings.

- **G1 (Minor, confirmed).** "Pre-task" named two trees. Fixed: the defect is placed in
  `811f4b3`, the owner's "before" version, here and in AC14.
- **G2 (Minor, partly).** The delivered S06–S08 show superseded text, and the package did not
  say so. Fixed: §4.4 says so, and the changed screen was re-captured (§4.6).
- **G3 (Note, confirmed).** "Every saved record carries both candle times" was not counted.
  Fixed: counted (§2).
- **G4 (Note, partly).** The source-review README named the wrong tree for the focused run.
  Fixed: it now names both trees and their equal source-only identity.
- **G5 (Note).** The critic's own disclosure: it briefly wrote bytecode folders into the
  internal isolated copy and removed them. Recorded; the copy's files are unchanged.

## 6. Boundaries kept

- **Unchanged:** the earlier configurator, the saved three-day time-under-water preference,
  the account, exit, payout, cost and calendar rules, data permissions and the engine pins
  (`research/core`, `pyproject.toml`).
- **Not changed or touched:** no funded-simulator or engine code, and nothing under data/ or
  reports/.
- **Not done:** no economic study, full-path model, commit or push.

## 7. Still open (separate future tasks, unchanged here)

- **Engine work:** the larger full-path firm model and saving each variant's own setup
  records.
- **Measured before, not re-measured here:** the earlier feature-and-model first-load delay
  and the 356 configurations listed as unavailable.
- **Not fixed by this change:**
  - the five long-standing full-suite failures;
  - the recorded guard-teardown follow-ups (closeout `REVIEW_FINDINGS.md` rows 27–28).
- **Owner decisions:** the time-under-water limit, and whether to retire the earlier
  configurator.

## 8. What is in this package

`manifest.json` lists every file, with its size and SHA-256:

- **This report**, `FOLLOWUP_REPORT.md`.
- **`CALCULATION_DEFINITIONS.md`:** the maintained definitions. Its header lists every change
  from the closeout copy.
- **`DEMONSTRATION.md`:** before and at completion, synthetic and real.
- **`screenshots/`:** the four captures in §4.6.
- **`validation_summary.json`:**
  - trees and identity;
  - every focused command, with its times, exit codes, counts and skips;
  - what did not run;
  - the mutation probes;
  - demonstration counts and the saved-record completeness count;
  - preservation and access;
  - the test sequence;
  - the source-review attachment check.
- **`source_identity.json`.**
- **`prefix_reproduction.json`.**
- **ZIPs.** `../followup-1.zip` holds this folder and is verified by extraction. The separate
  `../followup-1_source_review.zip` holds before and after copies of every changed source and
  test file, `changes.diff`, and a replay of the owner's boundary from the extracted files
  alone. At 7:05 PM the delivered `review_panels.py` shows the close-through in the text and
  not in the key; the corrected one shows it in neither.

Scripts, logs, receipts and the isolated extraction stay in the internal folder
`../../../../Claude-Quant-Lab-Research-Artifacts/ifvg-setup-timing-followup-20260925/`.
