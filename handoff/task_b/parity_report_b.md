# Task B B4 parity

Decision 13's frozen A1 comparison is reused unchanged.
V2 compares complete bytes; v3 uses A1 retained Arrow field order, types, nullability,
primary-key row order and exact values. No additional exclusions, rounding or ID rewrite.

Compared at UTC: 2026-10-04T02:02:01.911484+00:00
A1 reference: C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-a1-20261003
Task B after: C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-b-20261003\parity\after
Passed: True

| v2 table | A1 SHA-256 | Task B SHA-256 | Match |
|---|---|---|---|
| candidate_label | fbd7072561432ba9ea6e16d284897dd1b09f7112a207984d2b08d2e736f25630 | fbd7072561432ba9ea6e16d284897dd1b09f7112a207984d2b08d2e736f25630 | True |
| eligible_decision | 4f1aed4bd36912913af21a5e9cb4f17a040c5b6843a849ee56951d06051ffe1b | 4f1aed4bd36912913af21a5e9cb4f17a040c5b6843a849ee56951d06051ffe1b | True |
| entry_candidate | 6ae04d565489ce6b4786ea0e52e45764f46b4489b12690efcf46e9115b612d2d | 6ae04d565489ce6b4786ea0e52e45764f46b4489b12690efcf46e9115b612d2d | True |
| executed_trade | 01f101041fdd47333d9c0f325c80844859c16e07f61251cb020ab55f1f3ecc25 | 01f101041fdd47333d9c0f325c80844859c16e07f61251cb020ab55f1f3ecc25 | True |
| geometry_dossier | 6751177ed4be4229baa1d74f21078feb57a4219e14c477dc1e7525b5b4d9de87 | 6751177ed4be4229baa1d74f21078feb57a4219e14c477dc1e7525b5b4d9de87 | True |
| label_source_1m | f60acdf488d7950193466a286f866e5dd26b1bc2462a8ba01f10c7243b99f0c0 | f60acdf488d7950193466a286f866e5dd26b1bc2462a8ba01f10c7243b99f0c0 | True |
| quarantine | 4c4ae517a040f2207586c06f7cbff806f20f16cb61af0f75f7875b6c5a95b61c | 4c4ae517a040f2207586c06f7cbff806f20f16cb61af0f75f7875b6c5a95b61c | True |
| setup_lifecycle_event | 654dbae6376f60040115b5b7dfe165b2aa3000221c859b06b81fc5c14b15303b | 654dbae6376f60040115b5b7dfe165b2aa3000221c859b06b81fc5c14b15303b | True |

| v3 table | A1 retained SHA-256 | Task B retained SHA-256 | Match |
|---|---|---|---|
| context_state | 129bda2ef5a7638797771abe3213f8d488d2ad85d60af6802a2a44b4185bdd52 | 129bda2ef5a7638797771abe3213f8d488d2ad85d60af6802a2a44b4185bdd52 | True |
| context_capture | 8e2e4bdf424b875696ec6cff93e0ac767efe88898a2ebfbd1c9fbc40916aa2be | 8e2e4bdf424b875696ec6cff93e0ac767efe88898a2ebfbd1c9fbc40916aa2be | True |
| context_structure_state | 8278f66f153365c83c2122059358aad78e44e1c766794144ce736d26f0ff67d3 | 8278f66f153365c83c2122059358aad78e44e1c766794144ce736d26f0ff67d3 | True |
| context_structure_delta | 4929a80d4423bccf7ebfcb098aec6418569152260251ddc2755d39b634069889 | 4929a80d4423bccf7ebfcb098aec6418569152260251ddc2755d39b634069889 | True |
| context_displacement_window | 78acfe968d1af148cdbc8e6209d3d75ab6b2e298e9656549ebeaaf6bc41dc9ea | 78acfe968d1af148cdbc8e6209d3d75ab6b2e298e9656549ebeaaf6bc41dc9ea | True |
| equal_level_pool_lifecycle | f0a339cda1999678a34c5549f611b9a08eb7e3892fe8a1a52ec81d3f7a95d739 | f0a339cda1999678a34c5549f611b9a08eb7e3892fe8a1a52ec81d3f7a95d739 | True |
| equal_level_pool_member | 9ee00a4d1ec6a61721d1c98105274236e2e001b42619157ac6e36ab538602ee7 | 9ee00a4d1ec6a61721d1c98105274236e2e001b42619157ac6e36ab538602ee7 | True |
| equal_level_sweep_link | d058cf98b09f4acb526891ab564ec5df637d60eb88de438291d8573cb481903e | d058cf98b09f4acb526891ab564ec5df637d60eb88de438291d8573cb481903e | True |
| context_validity_provenance | 8ad7d29d4cbce97521c3cd06bdcd3ca79f37ddf1dd2fec20226bbc0938193d2c | 8ad7d29d4cbce97521c3cd06bdcd3ca79f37ddf1dd2fec20226bbc0938193d2c | True |
| candidate_context_link | 8bd37ba51d8e5a4d3ae805e51729023c61ba96319806fc0058e05822529248ba | 8bd37ba51d8e5a4d3ae805e51729023c61ba96319806fc0058e05822529248ba | True |
| decision_context_link | 9c820c399e028f7bf2bd32bbe7d98295dbc594fbaf1d79a6e68d0349f81994da | 9c820c399e028f7bf2bd32bbe7d98295dbc594fbaf1d79a6e68d0349f81994da | True |
| trade_context_link | 692f4fbdd02113cbcd29cd0110c5f43faecc8d3eb258c95078411e7417ea9813 | 692f4fbdd02113cbcd29cd0110c5f43faecc8d3eb258c95078411e7417ea9813 | True |

Profile hash A1: e0f318732cb59d844ac14b5e3839862146e7da1f612f9884f767247f66dd39dd
Profile hash Task B: e0f318732cb59d844ac14b5e3839862146e7da1f612f9884f767247f66dd39dd
Profile hash match: True

Evaluation dates: 2026-01-13, 2026-01-14, 2026-01-15, 2026-01-16, 2026-01-19, 2026-01-20, 2026-01-21, 2026-01-22, 2026-01-23, 2026-01-26
Warmup dates: 2026-01-01, 2026-01-02, 2026-01-04, 2026-01-05, 2026-01-06, 2026-01-07, 2026-01-08, 2026-01-09, 2026-01-11, 2026-01-12
Physical chain dates: 2026-01-01, 2026-01-02, 2026-01-04, 2026-01-05, 2026-01-06, 2026-01-07, 2026-01-08, 2026-01-09, 2026-01-11, 2026-01-12, 2026-01-13, 2026-01-14, 2026-01-15, 2026-01-16, 2026-01-18, 2026-01-19, 2026-01-20, 2026-01-21, 2026-01-22, 2026-01-23, 2026-01-25, 2026-01-26

Prepared-input copy receipts and date scopes match A1; forbidden counters are zero.

## Frozen A1 v3 exclusions

**context_state**: strategy_core_commit, strategy_core_source_tree_hash, mtf_strategy_core_commit, mtf_strategy_core_source_tree_hash, entering_context_seed_hash
- strategy_core_commit: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- strategy_core_source_tree_hash: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- mtf_strategy_core_commit: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- mtf_strategy_core_source_tree_hash: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- entering_context_seed_hash: Full observer-seed hash includes ContextIdentity carrying Core commit/source tree; capture_driver.py:202-204 and Core context_features.py:913-934.

**context_capture**: strategy_core_commit, strategy_core_source_tree_hash, entering_context_seed_hash
- strategy_core_commit: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- strategy_core_source_tree_hash: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- entering_context_seed_hash: Full observer-seed hash includes ContextIdentity carrying Core commit/source tree; capture_driver.py:202-204 and Core context_features.py:913-934.

**context_structure_state**: strategy_core_commit, strategy_core_source_tree_hash, entering_context_seed_hash
- strategy_core_commit: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- strategy_core_source_tree_hash: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- entering_context_seed_hash: Full observer-seed hash includes ContextIdentity carrying Core commit/source tree; capture_driver.py:202-204 and Core context_features.py:913-934.

**context_structure_delta**: strategy_core_commit, strategy_core_source_tree_hash, entering_context_seed_hash
- strategy_core_commit: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- strategy_core_source_tree_hash: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- entering_context_seed_hash: Full observer-seed hash includes ContextIdentity carrying Core commit/source tree; capture_driver.py:202-204 and Core context_features.py:913-934.

**context_displacement_window**: strategy_core_commit, strategy_core_source_tree_hash, entering_context_seed_hash
- strategy_core_commit: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- strategy_core_source_tree_hash: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- entering_context_seed_hash: Full observer-seed hash includes ContextIdentity carrying Core commit/source tree; capture_driver.py:202-204 and Core context_features.py:913-934.

**equal_level_pool_lifecycle**: strategy_core_commit, strategy_core_source_tree_hash, pool_strategy_core_commit, pool_strategy_core_source_tree_hash, entering_context_seed_hash
- strategy_core_commit: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- strategy_core_source_tree_hash: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- pool_strategy_core_commit: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- pool_strategy_core_source_tree_hash: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- entering_context_seed_hash: Full observer-seed hash includes ContextIdentity carrying Core commit/source tree; capture_driver.py:202-204 and Core context_features.py:913-934.

**equal_level_pool_member**: swing_strategy_core_commit, swing_strategy_core_source_tree_hash, entering_context_seed_hash
- swing_strategy_core_commit: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- swing_strategy_core_source_tree_hash: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- entering_context_seed_hash: Full observer-seed hash includes ContextIdentity carrying Core commit/source tree; capture_driver.py:202-204 and Core context_features.py:913-934.

**equal_level_sweep_link**: strategy_core_commit, strategy_core_source_tree_hash, entering_context_seed_hash
- strategy_core_commit: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- strategy_core_source_tree_hash: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- entering_context_seed_hash: Full observer-seed hash includes ContextIdentity carrying Core commit/source tree; capture_driver.py:202-204 and Core context_features.py:913-934.

**context_validity_provenance**: strategy_core_commit, strategy_core_source_tree_hash, entering_context_seed_hash
- strategy_core_commit: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- strategy_core_source_tree_hash: Literal Core commit/source-tree provenance (including nested mtf/pool/swing records); Core structures/context.py:100-125,159-161 and capture_driver.py normalization.
- entering_context_seed_hash: Full observer-seed hash includes ContextIdentity carrying Core commit/source tree; capture_driver.py:202-204 and Core context_features.py:913-934.

**candidate_context_link**: none

**decision_context_link**: none

**trade_context_link**: none

Capture/comparison commands and actual timestamps are preserved in run_log.txt.
The comparator does not perform a capture or replay. Named capture metadata and
generated v2/v3 outputs are its only input artifacts.
