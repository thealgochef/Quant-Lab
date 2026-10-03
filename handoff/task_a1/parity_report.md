# Task A1 parity

Decision 13 defines the comparison.

The baseline was captured before implementation on the Step 0 main code, with the installed unmodified original Core. The after run used the implemented A1 working tree and the re-pinned installed Core. All four new fields remained at their defaults in both parity runs. No replay or test was rerun when this report was expanded.

| Identity | Baseline | After |
|---|---|---|
| Quant-Lab HEAD at capture | `2e3efd5d9921e1956d80144ee50d831956c7a360` | `2e3efd5d9921e1956d80144ee50d831956c7a360`; A1 edits were uncommitted at capture |
| Core commit at capture | `7c7111e398c083cf8e966e2e0c5aac8a41cc12c0` | `b062bfcf5a5209440a4b4c9d7c0ca2263f9f4cc2` |
| Core canonical full-package source identity (`research/core/current.json`) | `9247f16e223c9226f9afa17832c8810e31ce8853bfad9a0b5452e31845377bf3` | `afa16bed5733f6d41f090bc8b8e09215464cf9a009ec1c1bc7236973b76ead01` |
| Core context-source tree hash persisted in the capture envelope | `03857ee86afa41e8211b5b4c4e88fdc6b9f7a4de64c46ce2d4a25b5ee0dca095` | `517e65f4a7e097287ebd31925b326e50f972e8f7c982bcdcf1eea69da222725c` |

The canonical package identity and context-source tree hash cover different established source sets. The latter is the actual `strategy_core_source_tree_hash` supplied to these v3 captures. Quant-Lab did not persist a separate working-tree source hash; its HEAD and the explicit uncommitted after-state are recorded here. Git reflog confirms no Quant-Lab HEAD movement between the feature checkout at 2026-10-03T03:26:22-05:00 and the two capture commands.

Ten evaluation dates: `2026-01-13`, `2026-01-14`, `2026-01-15`, `2026-01-16`, `2026-01-19`, `2026-01-20`, `2026-01-21`, `2026-01-22`, `2026-01-23`, `2026-01-26`.

Ten warmup dates: `2026-01-01`, `2026-01-02`, `2026-01-04`, `2026-01-05`, `2026-01-06`, `2026-01-07`, `2026-01-08`, `2026-01-09`, `2026-01-11`, `2026-01-12`.

The physical replay chain contains those twenty dates plus `2026-01-18` and `2026-01-25` (Sunday source partitions), for 22 physical dates. Trusted prepared day artifacts were byte-copied to the external `day_inputs` scratch directory. Both phases used the same explicit original cache-trust allowlist, while replaying only this bounded chain. No raw market files were opened.

V2 compares the complete Parquet bytes for the seven execution tables and `label_source_1m`, plus the exact historical profile hash. V3 drops only the per-table columns listed below, sorts each retained table by its existing unique primary key, combines Arrow chunks, and compares exact retained field names, field order, types, nullability, canonical row order, and values using Arrow table equality. Retained-content SHA-256 covers Arrow IPC serialization with schema metadata removed; serialization metadata is not compared. Every non-excluded column is retained, including all record IDs, numerical hashes, configuration hashes, and schema hashes. No tolerance, rounding, value imputation, ID rewrite, or additional column exclusion is applied.

Exact executed commands, including their recorded timestamps (America/Chicago offset), follow. All ran from `C:\Users\gonza\Documents\Claude-Quant-Lab`. The capture commands invoked the normal lower-level v2/v3 preparation builders with cached artifacts only and `measure_performance=False`, into the named external scratch stores. Reports were called directly; no gated research launch or benchmark replay was used.

```powershell
# 2026-10-03T03:32:26.3493174-05:00
python -B 'C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-a1-20261003\task_a1_harness.py' --phase baseline 2>&1 | Tee-Object -FilePath 'C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-a1-20261003\baseline_execution.log'
```

```powershell
# 2026-10-03T03:49:00.0192785-05:00
python -B 'C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-a1-20261003\task_a1_harness.py' --phase after 2>&1 | Tee-Object -FilePath 'C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-a1-20261003\after_execution.log'
```

```powershell
# 2026-10-03T03:50:18.4142796-05:00
python -B 'C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-a1-20261003\compare_parity.py'
```


Baseline: C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-a1-20261003\baseline
After: C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-task-a1-20261003\after

| v2 table | Before SHA-256 | After SHA-256 | Match |
|---|---|---|---|
| candidate_label | fbd7072561432ba9ea6e16d284897dd1b09f7112a207984d2b08d2e736f25630 | fbd7072561432ba9ea6e16d284897dd1b09f7112a207984d2b08d2e736f25630 | True |
| eligible_decision | 4f1aed4bd36912913af21a5e9cb4f17a040c5b6843a849ee56951d06051ffe1b | 4f1aed4bd36912913af21a5e9cb4f17a040c5b6843a849ee56951d06051ffe1b | True |
| entry_candidate | 6ae04d565489ce6b4786ea0e52e45764f46b4489b12690efcf46e9115b612d2d | 6ae04d565489ce6b4786ea0e52e45764f46b4489b12690efcf46e9115b612d2d | True |
| executed_trade | 01f101041fdd47333d9c0f325c80844859c16e07f61251cb020ab55f1f3ecc25 | 01f101041fdd47333d9c0f325c80844859c16e07f61251cb020ab55f1f3ecc25 | True |
| geometry_dossier | 6751177ed4be4229baa1d74f21078feb57a4219e14c477dc1e7525b5b4d9de87 | 6751177ed4be4229baa1d74f21078feb57a4219e14c477dc1e7525b5b4d9de87 | True |
| label_source_1m | f60acdf488d7950193466a286f866e5dd26b1bc2462a8ba01f10c7243b99f0c0 | f60acdf488d7950193466a286f866e5dd26b1bc2462a8ba01f10c7243b99f0c0 | True |
| quarantine | 4c4ae517a040f2207586c06f7cbff806f20f16cb61af0f75f7875b6c5a95b61c | 4c4ae517a040f2207586c06f7cbff806f20f16cb61af0f75f7875b6c5a95b61c | True |
| setup_lifecycle_event | 654dbae6376f60040115b5b7dfe165b2aa3000221c859b06b81fc5c14b15303b | 654dbae6376f60040115b5b7dfe165b2aa3000221c859b06b81fc5c14b15303b | True |

| v3 table | Before retained-content SHA-256 | After retained-content SHA-256 | Match |
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

Profile hash: e0f318732cb59d844ac14b5e3839862146e7da1f612f9884f767247f66dd39dd; match: True

V3 exclusions (no IDs removed):

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

The exact replay and comparison commands are reproduced above; their original full command record remains in the external working directory.

Full source configuration, exact 22 physical dates, ten evaluation days, ten warmup days, copy hashes and zero-protected-access receipts are in each phase's input_receipts.json and access_audit.json.
