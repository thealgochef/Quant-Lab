# R2 — Files Touched

Status of every repo file below: **staged and committed at the R2 release
checkpoint** (hash in `GATE_SUMMARY.md`).

## New repository source files (10 inherited drafts, reviewed + repaired, plus the job shim)

```text
src/alpha_lab/agents/data_infra/ifvg/search/orchestrator.py
src/alpha_lab/agents/data_infra/ifvg/search/lineage.py
src/alpha_lab/agents/data_infra/ifvg/search/gates.py
src/alpha_lab/agents/data_infra/ifvg/search/strategy_metrics.py
src/alpha_lab/agents/data_infra/ifvg/search/frontier.py
src/alpha_lab/agents/data_infra/ifvg/search/robustness.py
src/alpha_lab/agents/data_infra/ifvg/search/insights.py
src/alpha_lab/agents/data_infra/ifvg/study/population_delta.py
src/alpha_lab/agents/data_infra/ifvg/study/funnel_delta.py
src/alpha_lab/agents/data_infra/ifvg/study/contrasts.py
scripts/ifvg_search_job.py
```

Repairs applied to the inherited drafts during R2 (all test-driven):

- `gates.py` — the eleventh threshold (`require_bootstrap_ci_excludes_zero`)
  now evaluates (was silently absent); typed failure mapping extended.
- `strategy_metrics.py` — `net_expectancy_bootstrap_ci95` extracted from the
  trading-day cluster bootstrap so the CI gate has an input.
- `lineage.py` — `LineageUniquenessReport.per_entity_kind` uses `ImmutableMap`
  (CS §0.3); the candidate entry-FVG column corrected to the real v2 schema
  (`fvg_fvg_id`); dropped-tap rows with empty setup ids no longer poison
  setup-identity completeness.
- `orchestrator.py` — per-search lock gained checkpoint heartbeats + a
  provably-stale orphan break (resume-after-kill); `run_search` exposes
  `stale_lock_seconds`; module registered in the identity-projection audit.
- `insights.py` — all seven fixed categories always render (Evidence Quality
  emits an explicit no-evidence line instead of vanishing).
- `contrasts.py` — the seed-7 paired bootstrap CI
  (`paired_cell_bootstrap_10000_seed7_v1`) attached to every metric estimate.

## New test files

```text
tests/agents/ifvg_search/test_child_audit_companion.py
tests/agents/ifvg_search/test_lineage.py
tests/agents/ifvg_search/test_orchestrator.py
tests/agents/ifvg_search/test_population_funnel_deltas.py
tests/agents/ifvg_search/test_contrasts.py
tests/agents/ifvg_search/test_gates_frontier_robustness_insights.py
tests/agents/ifvg_search/test_cohort_interpretation.py
tests/agents/ifvg_search/test_search_job_script.py
tests/agents/ifvg_search/test_setup_drill_through.py   (adversarial B-M2/B-M4)
```

## Adversarial-fix additions (same release; ADVERSARIAL_REVIEW_RESOLUTION.md)

- `search/charter.py` — `declared_contrast_ids` in the hashed charter payload
  (DEV-R2-9); `OBJECTIVE_DIRECTIONS` registry + `ObjectivePolicy` direction
  validation.
- `search/failure.py` — typed `ChildNeutralityError`.
- `search/orchestrator.py` — neutrality blocks child publication (F1);
  nonce-verified lock release + rename-to-quarantine stale break + 24h
  default threshold (F2); costed-evaluation publication + reused-child reload
  + persisted frontier (F5); real registry hash + per-combo authorization +
  reachable `blocked_invariant_failure` w/ the DEV-R2-8 exemption (F6);
  sentinel consumption + `phase_notes` (F13); registered objective directions
  + missing-objective exclusion (F16).
- `search/child_replay.py` — worker raises on failed neutrality; slice
  refuses before publication + §3.3 companion-then-publish order; zero-warmup
  slice cfg (B-M1); vacuous-link stamp (F10); neutrality-before-audit
  publication order (F14); referential-integrity fail-closed (F15);
  repo-root-anchored canonical marker (F19); cost from the registered policy
  (m9).
- `search/store.py` — sidecar-hash verification on reuse (F14);
  `lineage_reports` store.
- `search/lineage.py` — `LineageUniquenessEnvelope` + `persist_lineage_uniqueness`
  (F7); `derive_lineage_validity` (m8).
- `search/identities.py` — cross-baseline canonical-name adoption (F12);
  ImmutableMap coerce duplicate-key refusal (F11).
- `search/gates.py` — max caps fail closed when unmeasurable with trades (F9).
- `search/insights.py` — EvidenceRef vocabulary cleanup (F8, DEV-R2-7).
- `search/frontier.py` — trace wording without the forbidden substring (F18).
- `study/population_delta.py` — same-profile guard on the native basis (F4);
  corrected persistence wording (F7).
- `study/contrasts.py` — charter-anchored evaluation (F3); typed interaction
  refusal + ambiguous-grid refusal (F17); nested `ImmutableMap` estimates
  (F11).
- `fsm_audit_preparation.py` — warmup-stamp consistency refusal (B-M1).
- `scripts/ifvg_search_job.py` — context-managed job log (B-m6).

## Modified repository files

```text
src/alpha_lab/agents/data_infra/ifvg/search/identities.py
    (registry imports orchestrator/contrasts/fsm_audit_preparation;
     canonicalize_section content-equality guard — a derived child carrying
     its baseline's name is now renamed to its canonical id, fixing CS §1.3
     for every generated child)
src/alpha_lab/agents/data_infra/ifvg/search/store.py
    (+"fsm_audit_companions" store name — DEV-R2-2)
src/alpha_lab/agents/data_infra/ifvg/search/child_replay.py
    (+ChildAuditNeutralityEnvelope, build_slice_companions,
     verify_exact_drill_targets — the DEV-R1-6 companion seam closed)
src/alpha_lab/agents/data_infra/ifvg/dataset.py
    (audit-mode capture retains trace_audit_rows + stamps audit frames
     identically to the fsm-audit builder — DEV-R2-1)
src/alpha_lab/agents/data_infra/ifvg/fsm_audit_preparation.py
    (per-child neutrality-aware audit build: ChildFsmAuditBuild/Envelope,
     build_child_fsm_audit, publish_child_fsm_audit; the doc-default parity
     gate untouched)
src/alpha_lab/agents/data_infra/ifvg/replay_chart_provider.py
    (resolve_selection +setup_id exact kind)
scripts/ifvg_verifier_tab.py
    (queue_jump +setup_id; _route_pending_setup_jump → setup mode's exact
     resolver)
tests/agents/test_ifvg_verifier_tab.py
    (contract updated in the same change: setup_id queues; nearest/fuzzy
     kinds still refused; routing test added)
tests/agents/ifvg_search/conftest.py
    (+make_resolved_trades_frame — a fully valid synthetic executed-trade
     builder)
ARCHITECTURE.md · docs/README.md · docs/pipeline_state.yaml
    (R2 lane appends/status only; user hunks preserved — see GATE_SUMMARY)
```

## Never modified (verified)

All M0–M3 lane modules, all existing `src/alpha_lab/propsim/` modules, all of
Strategy-Core, all of Trade-Lab, all existing immutable artifacts/catalogs
under `data/`, `docs/DECISIONS.md` (no R2-reserved decision exists — D-041 is
R3's), `docs/ML_TRAINING_WORKBENCH.md`.
