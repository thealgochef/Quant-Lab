<!-- PROPOSAL — nothing here is an authorization. Regenerated after the adversarial-fix
round by `scripts/ifvg_regime_promotion.py propose` over the R6.1 browser-smoke store
(commit 6c0b60a; SYNTHETIC fixture runs under %TEMP%/ifvg_r61_smoke/d8dae9697cf5b3bb — no real
data, no real path; the protocol / assessment ids below are the smoke run's synthetic ids).
The owner's ratification workflow (`propose` over the REAL assessment, then the owner completes
the placeholders and persists the artifact) replaces this draft. A payload carrying a
placeholder cannot be persisted as an owner decision (test-pinned). -->

# Draft owner decision 25+28+29+30 — context-bar panel grain (300s)

**PROPOSAL — nothing here is an authorization.** The owner's ratification workflow replaces every placeholder; a payload carrying a placeholder cannot be persisted as an owner decision.

- decision_id: `25+28+29+30:regime_feature_eligibility`
- resolved_regime_protocol_id: `20fa067681083fba016751b5020c000f79706705b31b3fb14ce0f9fc921c9c9f`
- capability_assessment_id: `a71df8441a582775d0e5d5a9978784f5f5313a4513670127d9235448e3747ffd`
- authorized_transitions: ['stratification_ready->feature_eligible']

| Decision | Key | Proposed value (proposed_protocol_default) |
|---|---|---|
| 25 | algorithm_key | `kmeans_v1` |
| 25 | algorithm_parameters_hash | `3130dd35c2752f0652290ff8368d9dc610836738548b7b03dbc78496fb0d8fa8` |
| 25 | algorithm_parameters | `{"algorithm": "lloyd", "init": "k-means++", "max_iter": 300, "n_init": 10, "random_state": 7, "tol": 0.0001}` |
| 28 | observation_granularity | `context_bar_panel` |
| 28 | panel_interval_seconds | `300` |
| 28 | observation_stage | `entry_decision` |
| 29 | fixed_cluster_count | `3` |
| 30 | minimum_cluster_occupancy_fraction | `0.05` |
| 30 | minimum_cluster_rows_per_fold | `25` |
| 30 | minimum_bootstrap_aligned_ami_mean | `0.5` |
| 30 | minimum_training_observations | `300` |

Owner fields to complete: `author`, `approved_at` (ISO-8601 with offset), `effective_from`, optional `effective_to`, `rationale`; provenance stays `owner_signed`.

```json
{
  "approved_at": "<OWNER_TO_FILL>",
  "author": "<OWNER_TO_FILL>",
  "authorized_transitions": [
    "stratification_ready->feature_eligible"
  ],
  "capability_assessment_id": "a71df8441a582775d0e5d5a9978784f5f5313a4513670127d9235448e3747ffd",
  "decision_id": "25+28+29+30:regime_feature_eligibility",
  "decision_keys": [
    "25:regime_algorithm_baseline",
    "28:regime_grain",
    "29:cluster_counts",
    "30:occupancy_stability_gates"
  ],
  "decision_values": {
    "algorithm_key": "kmeans_v1",
    "algorithm_parameters": {
      "algorithm": "lloyd",
      "init": "k-means++",
      "max_iter": 300,
      "n_init": 10,
      "random_state": 7,
      "tol": 0.0001
    },
    "algorithm_parameters_hash": "3130dd35c2752f0652290ff8368d9dc610836738548b7b03dbc78496fb0d8fa8",
    "fixed_cluster_count": 3,
    "minimum_bootstrap_aligned_ami_mean": 0.5,
    "minimum_cluster_occupancy_fraction": 0.05,
    "minimum_cluster_rows_per_fold": 25,
    "minimum_training_observations": 300,
    "observation_granularity": "context_bar_panel",
    "observation_stage": "entry_decision",
    "panel_interval_seconds": 300
  },
  "effective_from": "<OWNER_TO_FILL>",
  "effective_to": null,
  "provenance": "owner_signed",
  "rationale": "<OWNER_TO_FILL>",
  "resolved_regime_protocol_id": "20fa067681083fba016751b5020c000f79706705b31b3fb14ce0f9fc921c9c9f",
  "reviewed_evidence_refs": [
    "a71df8441a582775d0e5d5a9978784f5f5313a4513670127d9235448e3747ffd"
  ],
  "supersedes": null
}
```
