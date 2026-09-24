# Final read-only review

Read SPEC.md, OWNER_DECISIONS.md, TASKS.md and SCREEN_BRIEF.md. Review actual changes against the recorded pre-task baseline in each repository. The lead must provide task-only diffs or before/after source evidence and relevant test receipts in the internal store. Do not assume main is the correct baseline or attribute unrelated existing edits to this task.

Focus on confirmed defects that would change money, account survival or required behavior: open-trade failure ordering, end-of-day floor movement, fees and exact boundaries, payout stop/request/pause/receipt, per-instance credits and money isolation, growth spending, resumed state, changed strategy opportunities after exits, firm limits, same-clock copied accounts and screen/export reconciliation. Check absent/no-payout/failed accounts as well as favorable ones.

Compare completed tests with what they actually exercise. A screenshot does not prove worker propagation; a passing synthetic fixture does not prove historical price coverage. Mark unsupported assumptions and minute-path uncertainty. Verify that the automatic business review folder contains no scripts, source snapshots or raw dumps while retaining necessary adverse financial records.

Return each blocking finding with exact file/line, requirement, concrete failing case or reproduction, financial consequence, and smallest scoped correction. Separate demonstrated defects from concerns requiring a new test. Do not report style preferences as financial blockers. Inspect supplied real screen evidence for the owner's labels, times, decision hierarchy and material caveats. State when actual visual evidence is unavailable.

Do not edit files, run studies, place orders, purchase accounts, access protected/new data, or fabricate results. A clean source/receipt review is not execution of the code, a new backtest, a certification of complete price coverage or a future payout guarantee.

Return: Blocking findings; Evidence checked; Checks not performed; Owner decisions still needed. Use None where appropriate. The lead agent records the review and performs any actual fixes/retests.
