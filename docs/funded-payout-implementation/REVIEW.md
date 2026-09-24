# Final read-only review — instructions

Updated September 23, 2026 for the single-account configuration comparison (previous version
under `history/2026-09-23T053828-CDT/`). These are instructions for a reviewer, not completed
review findings. Actual findings are recorded in TASKS.md and the review folder's
`validation_summary.json` / RESULTS.md.

Read SPEC.md (Part A), OWNER_DECISIONS.md, TASKS.md and SCREEN_BRIEF.md. Review actual changes
against the recorded pre-task baseline in each repository. The lead provides task-only diffs or
before/after files and actual test receipts. Do not assume main is the baseline or attribute
unrelated edits to this task.

## What to check

- **Configurator coverage:** every configuration the owner selected is frozen into the plan,
  loaded by the worker as its own configuration and produces its own result for each selected
  firm. Nothing selects one profitable child to stand for the study. Poor or inactive
  configurations are not filtered out.
- **One account at a time:** at most one live funded account per configuration and firm,
  including payout-paused accounts; no substitute during processing; replacements charged once
  each, without a credit stop; no expansion after large receipts; no purchases after the cutoff.
- **Own strategy state:** early liquidation, blocked entries during payout protection or
  processing, and replacement change that pair's later opportunities; no same-event
  replacement trade; no replay of a consumed or failed signal; no state or money leakage
  between pairs. Identical trades that the rules produce independently are acceptable.
- **Price path and execution:** stop, target, account breach and daily-deadline ordering;
  stop fills under the declared convention when prints cross the stop; gap-through
  liquidation; fees at each fill and never twice; equality boundaries; which trades have
  ordered prints and which use a labeled minute approximation.
- **Money:** payouts (eligibility, day-end full-surplus request, two-business-day pause,
  4:00 PM Chicago receipt), pending at cutoff not received, receipts kept after failure,
  `net cash = receipts − account costs`, maximum shortfall.
- **Screen and export agreement:** the comparison screen and the review folder show the same
  verified values; no credit/growth tables in this mode; no scripts or raw dumps in the folder;
  adverse and zero-payout results retained.
- **History:** the September 22 pilot stays labeled as the earlier shared-signal five-account
  check and its stored result is unchanged.

Compare tests with what they actually exercise. A screenshot does not prove worker
propagation; a synthetic fixture does not prove historical price coverage.

## Output

Return: Blocking findings (file/line, requirement, concrete failing case, financial
consequence, smallest correction); Evidence checked; Checks not performed (including tests you
could not run); Owner decisions still needed. Use None where appropriate. Separate demonstrated
defects from concerns needing a new test; no style preferences as blockers.

Do not edit files, run studies or workers, place orders, access protected or new data, or
fabricate results. A clean review is not a new backtest, a certification of complete price
coverage or a future payout guarantee.
