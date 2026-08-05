# IFVG Lab Restoration and Context Repair Plan

Status: implementation contract  
Frozen: 2026-07-31  
Scope: Strategy-Core and Quant-Lab only

## Objective

Repair deterministic IFVG context generation in Strategy-Core, produce immutable formula-v2
context artifacts linked to unchanged v2 candidate artifacts, and restore Quant-Lab's IFVG Lab
around a leakage-safe, exact-link experiment workflow.

Trade-Lab runtime, WebSockets, orders, execution, models, and repository contents are out of
scope.

## Frozen baselines

| Repository | Branch | Commit | Working tree at freeze |
|---|---|---|---|
| Strategy-Core | `platform-refactor` | `f7a4a3cca154364587bca811e89f51e6dc295f2e` | dirty; existing context/v2 work must be preserved |
| Quant-Lab | `platform-refactor` | `bd6825924d02f76c108e81ef1d4edb27b032d55a` | dirty; existing v2/v3/UI work must be preserved |
| Trade-Lab (reference only) | n/a | `f808b9fd284bc306769c509e27144ed7340a9985` | no changes permitted |

The freeze includes the complete `git status --short` output captured before implementation.
Pre-existing tracked and untracked files are not reset, overwritten wholesale, or treated as
disposable. The accepted January v2 artifact is
`49902280956e2f6448799b16018e65b197387c58c6ae6109d17b5a76386c9a1a`, with manifest hash
`e635d064225444f02e2f11277d01912ee96e87bf24618aaa363cc86d53af7c5f`. Formula-v1 v3 artifact
`ee6cfa9eff3c27423281cb3ef34b638f04274cd94993a44adb2cae6b579e71c7` remains immutable and
readable.

## Architecture

Current:

```text
source rows -> v2 candidates/decisions/trades
            -> formula-v1 context (calendar inferred from missing rows)
            -> dynamic pandas parquet schemas
            -> approximate/mutable experiment joins and persistence
            -> audit-only IFVG Lab router
```

Target:

```text
authorized dates + committed ExchangeMinuteSchedule + explicit source coverage
  -> unchanged v2 stream ------------------------------+
  -> formula-v2 context (seed/resume deterministic)     |
                                                        v
                     verified immutable paired v2/v3 artifacts
                                      |
                            exact candidate-stage links
                                      |
                       one immutable candidate feature view
                                      |
                     deterministic M0/M1/M2/M3 walk-forward
                                      |
                  Experiments | Replay / Verifier | Data & Audit
```

## Source-of-truth map

| Concern | Source of truth |
|---|---|
| v2 candidates, labels, decisions, executions | accepted Strategy-Core v2 ordered stream |
| TIME bars and displacement eligibility | committed `ExchangeMinuteSchedule` plus `ContextSourceCoverage` |
| Context vocabulary | `ifvg_context_v1` |
| Corrected context math | `ifvg_context_formula_v2` |
| Physical context tables | `ifvg_context_arrow_v1` registry |
| Artifact pairing | v3 manifest's exact v2 artifact ID and manifest hash |
| Experiment rows | immutable exact-linked candidate view |
| Model protocol | `ifvg_context_catboost_binary_v1` |
| Actual execution report | source v2 decisions and executed trades only |
| Protected-data rules | development-access policy and event-derived audit counters |
| UI behavior | current contracts, never legacy mutable loaders |

## Defect and repair matrix

| Defect | Root cause | Repair | Gate |
|---|---|---|---|
| false displacement `source_gap` | elapsed UTC arithmetic subtracting one maintenance window | hashed exchange-minute schedule and typed coverage evidence | calendar boundary and gap-classification tests |
| unstable Parquet schemas | dynamic flattening and pandas inference | twelve explicit ordered Arrow schemas | empty/all-null/populated identity tests |
| lost opposing-leg evidence | active setup refers to evictable pool state | immutable/pinned qualifying links | saturated eviction test |
| untested short path | positive fixtures only cover long path | synthetic short-EQH normalization fixtures | directional parity tests |
| large/nonportable seeds | duplicate transient state and OpenSSL-dependent digest handling | schema-2 compact seed and portable length-prefixed SHA-256 chain | resume parity and size gates |
| weak parity diagnostics | unordered/partial comparison | ordered stream hash, complete ID diffs, first mismatch | parity report tests |
| leaking experiment joins | nearest/setup/row-order/keep-last matching | exact candidate-stage/cursor/capture binding | adversarial join rejection tests |
| mutable experiment outputs | overwrite/delete workflow | full-SHA immutable atomic run store | duplicate refusal tests |
| router regression | audit screen replaced experiment UI dispatcher | restored three-tab shell using new contracts | Streamlit AppTest |

## Dependency order

```text
contracts and baseline freeze
  -> schedule/coverage + formula-v2 observer
    -> January parity/regeneration
      -> Arrow registry + verified paired preparation
        -> exact feature view + labels/folds/model/statistics/run store
          -> IFVG Lab UI restoration
            -> nonsealed end-to-end verification and activation
```

No downstream artifact is promoted until all upstream identities and reports verify.

## Artifact migration

- Existing v2 bytes and IDs remain unchanged.
- Formula-v1 v3 remains readable and is marked `superseded` for M2 rather than rewritten.
- Formula-v2 uses record schema 2, observer seed schema 2, and Quant-Lab
  container/dataset/manifest schema 4.
- Formula-v1 seeds are rejected by formula-v2 observers; there is no migration shim.
- Formula-v2 v3 identity includes its exact v2 reference, per-table Arrow schema hashes, and
  aggregate schema hash.
- Staging directories, checkpoints, and locks are not catalog entries. Only fully validated
  final directories can become `context_ready`.

## Authorized-data boundary

- Warmup is the frozen first ten available permitted source dates before 2026-01-13.
- Candidate evidence is 2026-01-13 through 2026-06-10.
- Latest source instant is `2026-06-10T21:00:00Z`.
- June 11 is protected; June 12 onward is sealed.
- Permitted dates are enumerated before paths are constructed. Parent directories containing
  protected entries are never listed.
- Raw evidence may be read through the cutoff. Label bars require
  `entry_ts < close_ts < cutoff`; unresolved labels are `censored_protected_boundary`.
- Every protected-buffer and sealed audit counter is derived from access events and must be zero.

## Rollout and rollback

Rollout is additive: publish Strategy-Core, update only Quant-Lab's Strategy-Core pin through
its alignment workflow, prepare and verify immutable pairs, run one deterministic experiment,
then switch the Quant-Lab catalog pointer. Trade-Lab is not updated.

Rollback reverts the Quant-Lab catalog pointer and Strategy-Core pin. Immutable old artifacts,
formula-v1 reports, and legacy read-only results remain available. No artifact is converted or
rewritten in place.

## Risks and controls

| Risk | Control |
|---|---|
| accidental protected-date access | capability-scoped path construction and event audit |
| v2 behavioral drift | byte/seed/table/ordered-stream parity gates |
| context evidence removed by bounds | pinned immutable sweep links and saturation tests |
| platform-dependent identity | canonical serialization and portable SHA-256 chain |
| leakage through feature/label timing | exact cursor links and as-of assertions |
| selection bias | fixed tiers/model/folds/thresholds/bootstrap; no adaptive search |
| misleading UI comparison | strict compatibility identity and config diff on mismatch |
| dirty-tree loss | narrow patches and before/after status review |

## Acceptance gates

1. All Strategy-Core context formula-v2 tests, parity tests, capacity gates, and repeated
   performance gates pass.
2. January generation is repeat-identical; overwrite is refused; every remaining invalid window
   reconciles to audited source evidence.
3. Every v3 table validates against the exact Arrow registry in empty, all-null, and populated
   form; hashes, PK/FK/as-of constraints, and v2 reference verify.
4. Protected-buffer and sealed access counters are all zero.
5. Candidate views use exact stage/cursor links and preserve unavailable features as explicit
   nulls; no approximate or deduplicating match is accepted.
6. Folds, labels, CatBoost protocol, metrics, uncertainty, and run persistence are deterministic.
7. Experiments is the default UI tab; Replay / Verifier and Data & Audit use verified exact IDs;
   legacy runs are read-only.
8. Repository-native pytest and Ruff suites pass. AppTest, failure/empty states, keyboard and
   viewport checks pass.

Measured identities, commands, and gate results are appended during final handoff. Development
results are never described as sealed or external validation.

## Implementation handoff — 2026-07-31

Implementation is complete through the additive code, contract, unit/integration, AppTest, and
January verification surfaces. Catalog activation is blocked by `SC-PERF-001`; no formula-v2
artifact, full development pair, experiment run, dependency pin, or catalog pointer was
published.

Implemented Strategy-Core symbols include `ExchangeMinuteSchedule`, `MinuteSlotStatus`,
`ContextSourceCoverage`, cached `eligible_counts_by_partition()`, formula-v2 displacement gap
classification, the one-tick-span equal-level policy, pinned opposing-leg evidence,
`IfvgContextObserverSeed` schema 2, and compact internal market-structure/equal-level snapshots.
The portable record chain is the locked length-prefixed SHA-256 construction. Reclaimed links,
derived membership, fired-swing complements, and inner identity copies are omitted from seeds;
unreclaimed and active-setup evidence remains retained.

Implemented Quant-Lab surfaces include the twelve-table Arrow registry and schema hashes,
verified v2/v3 pair loading, development access auditing, persisted preparation jobs, exact
candidate-stage feature views, independent label derivations, setup-safe folds, the pinned
CatBoost/statistics/reporting protocol, immutable view/run stores, and the restored
Experiments / Replay-Verifier / Data-Audit UI. Primary implementation modules are
`context_schemas.py`, `artifact_io.py`, `development_access.py`, `preparation.py`,
`context_feature_view.py`, `context_labels.py`, `context_folds.py`, `context_model.py`,
`context_statistics.py`, `context_reporting.py`, `context_run_store.py`, and
`scripts/ifvg_lab_tab.py`.

### Final gate evidence

| Gate | Result |
|---|---|
| accepted v2 parity, context validity, identity, and reconciliation | pass on the 26-date cached January chain |
| terminal state / seed | 809,170 bytes / 809,170 bytes; limits 4,194,304 / 838,860 |
| largest transition | 23,211 bytes; limit 26,214 |
| largest equal-level pool | 5 members; limit 16 |
| observer step p99 | 1.3525 ms; limit 1.6 ms |
| multi-timeframe callback p99 | 0.3990 ms; limit 8 ms |
| replay median slowdown | 199.0982%; limit 20% — **failed** |
| paired repeated-run p95 slowdown | 203.7889%; limit 25% — **failed** |
| Strategy-Core unit suite | 414 passed; Ruff clean |
| Strategy-Core frozen real-store validation | 6 passed on frozen July 2025 fixtures; 764.1 s |
| Quant-Lab full suite | 970 passed; Ruff clean |
| Streamlit AppTest/server startup | pass |
| interactive viewport/keyboard/visual inspection | open; in-app browser unavailable in this session |

The performance measurement used Windows 11, Python 3.13.1, two warmup pairs, and ten alternating
paired runs over the same preloaded 26-date source. Baseline median was 1.67937215 seconds and
context median was 5.0229727 seconds; baseline/context coefficients of variation were
0.00833637 and 0.01214146. Because the aggregate gates failed, immutable save code was never
reached and the accepted v2 and formula-v1 v3 artifacts were not modified.

### Operator commands

From Strategy-Core:

```powershell
python -m pytest tests -q
python -m pytest validation -q --durations=10
python -m ruff check src tests
```

From Quant-Lab:

```powershell
python -m pytest -q
python -m ruff check src tests scripts
python scripts/run_ifvg_context_capture.py --cached-artifacts-only
```

The final command is expected to refuse publication until both slowdown gates pass. Full
permitted preparation, deterministic experiment execution, interactive visual QA, dependency
publication, and catalog activation remain downstream acceptance work; they must not be
bypassed or described as completed.
