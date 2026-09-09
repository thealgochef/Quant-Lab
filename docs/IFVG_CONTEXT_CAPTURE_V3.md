# IFVG context capture v3

Status: implementation complete and locally verified on 2026-07-31. This is a
nonsealed measurement artifact path, not a model, evaluation, or promotion.

## Contract boundary

The v3 capture references and hash-pins the accepted v2 exploration dataset:

- dataset ID:
  `49902280956e2f6448799b16018e65b197387c58c6ae6109d17b5a76386c9a1a`;
- manifest payload SHA-256:
  `e635d064225444f02e2f11277d01912ee96e87bf24618aaa363cc86d53af7c5f`.

One Strategy-Core replay produces the unchanged v2 emissions used for strict
parity plus typed `ifvg_context_v1` records. Core v2 tables are transient parity
inputs and are never duplicated into the v3 artifact. The saver accepts only the
new normalized context/link tables and rejects outcome/performance-like report
content.

The separate v3 table enum contains:

1. `context_state`;
2. `context_capture`;
3. `context_structure_state`;
4. `context_structure_delta`;
5. `context_displacement_window`;
6. `equal_level_pool_lifecycle`;
7. `equal_level_pool_member`;
8. `equal_level_sweep_link`;
9. `context_validity_provenance`;
10. `candidate_context_link`;
11. `decision_context_link`; and
12. `trade_context_link`.

These contracts do not extend v2 enums or `flatten_emissions()`. Primary keys,
formula identity, and foreign keys are validated table by table. Candidate,
decision, and trade joins require exact IDs; setup-only, nearest-time, row-order,
and keep-last fallbacks fail closed.

## Fixed nonsealed chain

The runner authorizes only January 1-2, 4-9, 11-16, 18-23, and 25-30, 2026.
The first ten available dates through January 12 are warmup; January 13-30 are
evidence. Every other date is denied before path construction, metadata lookup,
existence checks, or reads. Source-file hashes and the zero-forbidden-access
audit enter dataset identity.

Run from the Quant-Lab repository:

```text
python scripts/run_ifvg_context_capture.py --cached-artifacts-only
```

Omit `--cached-artifacts-only` only when rebuilding trusted allowlisted bar/level
inputs is explicitly intended. The command does not train, select, tune, gate,
or evaluate a model. It emits only validity, coverage, reconciliation, capacity,
identity, and performance reports.

Artifacts are content-addressed under
`data/ifvg_datasets/v3/<dataset-id>/exploration/`. The identity includes both
accepted-v2 hashes, all context identities, ordered timeframes, 240-minute
experimental status, repository HEAD/status/source-tree hashes, exact source
hashes, allowlist partitions, and container versions. A repeated derivation must
produce the same ID; saving to an existing ID is refused rather than overwritten.

The final local derivation on 2026-07-31 produced:

- dataset ID
  `ee6cfa9eff3c27423281cb3ef34b638f04274cd94993a44adb2cae6b579e71c7`;
- 1,123 unique context captures and 1,089 unique context states;
- 21 candidate, 6 decision, and 6 executed-trade exact-ID links;
- exact accepted-v2 core parity and 27,168 validity/provenance rows; and
- a second identical derivation that resolved to the same ID and was refused
  with `FileExistsError` before overwrite.

## Capacity and performance gates

The verification runner fails closed unless all required reports pass. Limits are
5 MiB terminal observer state, 1 MiB terminal seed, 32 KiB maximum transition,
25 percent replay slowdown, 2 ms completed 1-minute observer-step p99, and 10 ms
multi-timeframe callback p99. The performance report records its measurement
policy so results cannot be compared without provenance.

For the final artifact, terminal state and seed were both 981,037 bytes, the
largest transition was 32,376 bytes, completed-1m p99 was 0.7917 ms,
multi-timeframe callback p99 was 0.2349 ms, and replay slowdown was 23.46%.
Replay slowdown uses
`fixed_source_wall_replay_observer_subtraction_v1`: the clock includes the
shared fixed-source load, v2 core replay, table/label construction, and accepted
baseline parity work, and stops before context-only normalization.

## Future M0-M3 inputs (not executed here)

The future ablation families are declarations only:

- `M0`: accepted v2 baseline inputs, referenced by the pinned v2 dataset ID;
- `M1`: `M0` plus exact-linked structure state, the seven-timeframe MTF vector,
  local 1-minute structure, summaries, and transition deltas;
- `M2`: `M1` plus exact-linked integer-tick displacement windows and canonical
  1-minute FVG summaries; and
- `M3`: `M2` plus exact-linked equal-level pool state, membership, lifecycle,
  nearest-pool context, and sweep/reclaim evidence.

A future, separately approved evaluation must establish a new dataset/model
identity, use exact candidate/decision links, predeclare purged walk-forward and
calibration behavior, and prove feature identity. This implementation performs
none of that work and makes no predictive, profitability, or promotion claim.
