# Focused acceptance and test contract

The case IDs below define coverage, not a demand for one separate execution or screenshot per row. Batch related cases into the smallest meaningful test files. Retain failing-before/passing-after evidence for the reported defects. Use actual project functions and a compact deterministic fixture, not only copied function bodies.

## B8-01 behavioral cases

| ID | Required case | Acceptance |
|---|---|---|
| P01 | Actual MCB003 393-nanosecond anchor | The exact checkpoint is linked to its verified saved target decision despite lower-precision receipt context time. Original event/time/context bytes stay intact. |
| P02 | Recorded conditional decision retaining half | The partial-target branch uses its own saved receipt/context and provenance, not the entry's role. |
| P03 | Recorded conditional whole-at-target decision | The whole-target branch is correctly distinguished from a fixed whole-position policy or an annotation. |
| P04 | Actual unconditional partial-exit checkpoint without saved decision | Remains annotation-only at its actual checkpoint; no inferred executed target receipt. |
| P05 | Reused control enriched afterward | Reporting-only context stays labeled as such. Entry and checkpoint roles can differ legitimately. |
| P06 | Wrong/ambiguous receipt in same microsecond | Wrong trade/config/stream/account/policy/source or duplicate conflicting candidates are rejected or explicitly unresolved. Temporal closeness alone never establishes origin. |
| P07 | Point-in-time cursor | At target minus 1 ns, no target/action/context is visible; at the exact event, only facts permitted by canonical ordering appear. No time shift is introduced by display formatting. |
| P08 | Report-eligibility boundary and timezone conversion | Availability is validated against the saved contract and true decision basis. Never select a future report or round the observation backward. |
| P09 | Ordinary candle versus funded print | Keep their declared timing bases distinct. Provenance identifies an actual saved decision where verified even when its reporting checkpoint has a different specified meaning. |
| P10 | No target reached / missing saved target timestamp | Show not reached or unavailable as applicable. Do not label a nonexistent checkpoint as annotated or executed. |
| P11 | Select basis and coverage | Entry, parent-lock (where supported), and first-target cards/rows/coverage use their own origin. Each population reconciles, unavailable counts included. |
| P12 | Scientific numbers preserved | Source selections, numeric gamma values, group trade membership/results and frozen geometry are unchanged. Only approved provenance/linkage/count metadata differs. |

Reconcile all saved funded target-decision records, historically 1,908. Produce an audit keyed by result, configuration, stream and trade/decision identity. For each record report prior role, corrected role, actual event time, receipt context time/precision relationship, verification status and reason. If the current record count differs, explain the exact source/version difference rather than silently narrowing the cohort. Do not force 1,908 executed labels: verify identity, membership, and context for every record. Audit annotation-only/legacy/missing cases too and preserve them in summary counts.

The prior numerical context population is 43,130 entry/first-checkpoint reporting rows; reproduce applicable records by stable keys and explicitly account for any representation change. Aggregate origin counts may legitimately change; gamma signs/values and economic group totals are not allowed to change unnoticed.

## B8-02 output cases

| ID | Required case | Acceptance |
|---|---|---|
| E01 | All 64 configuration rows | IDs, effective policies and economics unchanged; each product/quantity description is accurate; NQ price proxy remains separate. |
| E02 | Six dynamic geometries MCB057–MCB062 | Correct schedule/fraction; half-range formula; frozen-at-lock basis; tick rounding from plan; separately named 20-point fallback. |
| E03 | Fixed-distance controls | Still explicitly fixed. Not relabeled dynamically merely because a gamma table exists. |
| E04 | Constant and regime-dependent sizing | Ten, six, and variable exposures are accurately described; no 'ten minis' interpretation of ten micros. |
| E05 | Every conditional exit policy | Correct condition and first-1R checkpoint; actual action/fallback described; no generic whole-exit substitution. |
| E06 | Regenerated chart | Readable distinct labels with IDs; stated top-N scope; cash matches saved results. At least the leader, a nearby geometry policy, a fixed reference and a conditional exit are distinguishable. |
| E07 | Shared route | Tests call the real standard exporter and shared label source. A manual CSV edit is not a fix. |
| E08 | UI/export agreement | Normal dashboard detail and downloaded/generated CSV/chart agree semantically for the same selected version. |

Use original b8db effective settings as the authority. A CSV row may gain descriptive fields; preserve machine configuration/intent bindings and label these as descriptions, not amended execution settings. Raw inherited metadata may be retained as historical, but must not masquerade as current active status.

## Preservation and UI checks

- Recheck hashes of original b8db and 7a8c envelopes, plans, approvals, child histories, context/target receipts, previously delivered exports/ZIPs, executed source and production pin. Restrict the inventory to known protected paths rather than scanning arbitrary market trees.
- Independently compare all 64 financial summary rows and scoped trade/account/cash record hashes or exact canonical projections. Expected changes in money, counts, entry/exit decisions, fees and actual timestamps: **zero**. Preserve gamma values and geometry by key, allowing only declared report metadata/description/provenance changes.
- MCB062's anchor remains 4,810,216 cents net cash; 4,972,716 cents received; 162,500 cents account costs; 13 purchased accounts; 302 funded trades; 25 received payments. This anchor supplements, not replaces, all-64 preservation.
- Open the ordinary main-dashboard study route; compare Original execution and Corrected lifecycle without overwriting either. Select a recorded conditional target, an annotation-only target, and MCB062's description/export. Confirm version and selected configuration remain correct after refresh/restart.
- Confirm target-minus-1-ns/exact-target behavior in a deterministic test, and inspect at least a before/at-target UI example at the supported cursor precision. Do not claim a browser proves nanosecond ordering if its controls are coarser.
- Retain the already-fixed account/trade point-in-time visibility and normal source-bound Market conditions, gamma and lens navigation. Focused regressions are sufficient; no new 19-screen capture matrix is required.

## Proportional verification

Capture final relevant pytest/lint commands, working directory, interpreter, actual imported files, exit status, counts, and source digest. Explicitly list skips, deselections, missing dependencies and any broader tests not rerun. Old passing counts must not be relabeled as final-source validation.

Perform one read-only review focused on linkage, basis-specific origin, export semantics, preservation, and completeness. Resolve concrete in-scope findings and retest affected paths. Do not turn documentation edits or optional cleanup suggestions into another long full-suite loop. Follow mandatory repository rules and state any resulting additional test work honestly.

For delivery, extract each completed ZIP into a clean directory. Verify unique safe paths, exact manifest membership/size/hash, readable JSON/CSV, artifact identity links, and a documented subset of runnable tests. No local-only file may be presented as included. Archive hashing must avoid circular/self-hashing claims.
