<!-- PROPOSAL — nothing here is an authorization. Regenerated after the adversarial-fix
round by `scripts/ifvg_regime_promotion.py propose` over the R6.1 browser-smoke store
(commit 6c0b60a; SYNTHETIC fixture runs under %TEMP%/ifvg_r61_smoke/d8dae9697cf5b3bb — no real
data, no real path; the protocol / assessment ids below are the smoke run's synthetic ids).
The owner's ratification workflow (`propose` over the REAL assessment, then the owner completes
the placeholders and persists the artifact) replaces this draft. A payload carrying a
placeholder cannot be persisted as an owner decision (test-pinned). -->

# Draft owner decision 25+28+29+30 — candidate grain (candidate_stage_row @ entry_decision)

**PROPOSAL — nothing here is an authorization.** The owner's ratification workflow replaces every placeholder; a payload carrying a placeholder cannot be persisted as an owner decision.

- decision_id: `25+28+29+30:regime_feature_eligibility`
- resolved_regime_protocol_id: `a9b7888ad1ff801ad343422248bdf5b1951ddff9121a972ff23ca3814598159b`
- capability_assessment_id: `db97e9bd4a906692e366720dddbf050e94cda0bfb7090b95a29d018d2430181b`
- authorized_transitions: ['stratification_ready->feature_eligible']

| Decision | Key | Proposed value (proposed_protocol_default) |
|---|---|---|
| 25 | algorithm_key | `kmeans_v1` |
| 25 | algorithm_parameters_hash | `3130dd35c2752f0652290ff8368d9dc610836738548b7b03dbc78496fb0d8fa8` |
| 25 | algorithm_parameters | `{"algorithm": "lloyd", "init": "k-means++", "max_iter": 300, "n_init": 10, "random_state": 7, "tol": 0.0001}` |
| 28 | observation_granularity | `candidate_stage_row` |
| 28 | panel_interval_seconds | `None` |
| 28 | observation_stage | `entry_decision` |
| 29 | fixed_cluster_count | `3` |
| 30 | minimum_cluster_occupancy_fraction | `0.05` |
| 30 | minimum_cluster_rows_per_fold | `25` |
| 30 | minimum_bootstrap_aligned_ami_mean | `0.5` |
| 30 | minimum_training_observations | `150` |

Owner fields to complete: `author`, `approved_at` (ISO-8601 with offset), `effective_from`, optional `effective_to`, `rationale`; provenance stays `owner_signed`.

```json
{
  "approved_at": "<OWNER_TO_FILL>",
  "author": "<OWNER_TO_FILL>",
  "authorized_transitions": [
    "stratification_ready->feature_eligible"
  ],
  "capability_assessment_id": "db97e9bd4a906692e366720dddbf050e94cda0bfb7090b95a29d018d2430181b",
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
    "minimum_training_observations": 150,
    "observation_granularity": "candidate_stage_row",
    "observation_stage": "entry_decision",
    "panel_interval_seconds": null
  },
  "effective_from": "<OWNER_TO_FILL>",
  "effective_to": null,
  "provenance": "owner_signed",
  "rationale": "<OWNER_TO_FILL>",
  "resolved_regime_protocol_id": "a9b7888ad1ff801ad343422248bdf5b1951ddff9121a972ff23ca3814598159b",
  "reviewed_evidence_refs": [
    "db97e9bd4a906692e366720dddbf050e94cda0bfb7090b95a29d018d2430181b"
  ],
  "supersedes": null
}
```
