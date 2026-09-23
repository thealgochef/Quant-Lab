# Strategy study pause and recovery

Updated 2026-09-10.

The existing study detail screen offers **Stop after current work** while running
and **Resume study** while interrupted. A stop is a request to finish and save
the current unit; it does not immediately terminate a replay. The worker checks
between input-verification configurations, after each replay's metrics are saved,
and between finalization stages, including after the final replay. Opening or
refreshing a paused study never starts work.

Completed and reused configurations count as successful work. Failed attempts
are shown separately and are included in the attempted total. Consequently,
147 completed plus 45 failed means 192 attempted, with 45 requiring a retry.
Preparation labels and counters report verification before replay starts. New
attempts persist start/update/end timestamps. Missing historical timing remains
unavailable rather than being inferred from a file's modification time.

## Bounded result handling

The previous parent runner retained all child result tables until the entire
batch finished. A large batch could exhaust memory before remaining children
or batch metrics completed. The runner now computes and publishes each child's
costed metrics before its completed checkpoint, then releases the full tables.
Only small metrics, gate reports and optional prop summary vectors persist in
memory across children. Prop consumers with a verified loader receive one saved
result at a time. Missing prop inputs cannot pass gates.

For a preserved Core replay missing costed metrics,
`search/saved_strategy_result.py` verifies the Core envelope and reference,
dataset identity and manifest, effective section, invariant audit, execution
table bytes/rows/keys/identity and gross stream hash. It reads only the saved
execution table. It never opens raw market data or runs the strategy. Warmup
exclusion, cost application, uncertainty estimation and every frozen gate use
the existing numerical implementations. A named baseline can have a generated
effective-section name; the requested canonical profile binds to the verified
Core profile identity, with the full effective-section hash checked separately.

## Provenance and resuming

Job-control repairs have a separate versioned source receipt in
`<state root>/<search id>/job_runtimes/<runtime id>.json`. It binds the worker,
orchestrator, recovery loader, metrics/statistics/gates and verification source
files plus the actual scoped strategy replay source identity. It does not
replace historical replay identities. The worker freshly resolves inputs on
resume and rejects changed identities for previously saved configurations.
Existing authorization, date, input-bundle and namespace checks remain required.

Completed replay tables and published metrics remain immutable. A future manual
resume reuses them and retries failed or unfinished work. A stop during resumed
preparation preserves the previous completion facts; those displayed facts do
not replace artifact verification before aggregation.

An externally killed worker may leave a lock. Do not remove it merely because
the screen appears stale. The September 10 recovery verified that the exact
worker was gone, preserved its checkpoint, checked that its lock was absent,
and verified all saved completed
artifacts, and wrote a paused operational checkpoint. Its reports are under
`../Claude-Quant-Lab-Research-Artifacts/archived-reports/ifvg_run_recovery_20260910/` (relocated September 22). No remaining study replay was launched.
