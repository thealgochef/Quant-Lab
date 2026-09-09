# Implementation Decisions Taken — `ifvg_prop_robust_config_search_v1`

Running consolidated ledger of implementation-level decisions (engineering
defaults and seam adaptations). None is an owner ratification; every
scientific default remains `proposed_protocol_default` per `OWNER_DECISIONS.md`.

## R1

1. **Shared-file commit isolation** — pre-existing user-owned hunks in
   `ARCHITECTURE.md` / `docs/README.md` / `docs/pipeline_state.yaml` are
   never committed: release commits stage HEAD + lane appends via
   `git hash-object --path` + `update-index`; byte-identity of the surviving
   user hunks verified post-commit (151/151). `docs/ML_TRAINING_WORKBENCH.md`
   untouched.
2. **`start_after_artifact` fallback adopted** (plan-sanctioned R0 fallback)
   on `build_ifvg_v2_capture`, plus the discovered
   `final_day_exhausts_dataset=False` requirement for snapshot-producing
   prefix chains (dataset exhaustion alters the end seed). DEV-R1-1.
3. **`ArtifactProvenanceReadAdapter`** for verification reads of dev-chain
   day-artifact caches (the cache stamp pins the WRITING policy's allowlist
   hash — discovered seam); strictly read-only (worker refuses rebuilds under
   it), every authorization delegated to the ≤5-day policy. DEV-R1-2.
4. **`study/cohort.py` authored in R1** (schedule-forward; content exactly
   per CS §9.1) so cell identities never churn at R2. DEV-R1-3.
5. **Axis value ids use the full technical key** (`<key>.<token>`); the
   plan's shorter ids are marked illustrative. DEV-R1-4.
6. **`none_*_v1` sentinels** accepted alongside 64-hex resolved ids for
   decision/model resolved-id fields until R5 mints real resolved ids.
   DEV-R1-5.
7. **Slice companion builders injected** — audit/chart companions + the
   `setup_id` verifier jump are R2-assigned seams; the slice's corresponding
   gates evaluate OPEN until wired (never claimed passed). DEV-R1-6.
8. **Coverage evidence presented neutrally** — the plan's candidate window is
   funnel-quiet (May–June: zero candidates/trades in the accepted dataset);
   `2026-02-06…02-11` covers all four funnel paths. Owner chooses (21/R-5);
   nothing registered. DEV-R1-7.
9. **Baseline cell bundle key = `B0_CORE`** — DT §2's `IFVG_CORE_BASELINE_V1`
   shorthand names a block; the typed bundle registry governs bundle keys.
   DEV-R1-8.
10. **V1 cells cannot carry an execution-gating decision policy** — the
    fail-closed reading of §7A.3 (S11 blocked + policies planned-only): the
    cell validator refuses them; the comparison layer for the
    post-ratification era is tested on a documented validation-bypassed
    construction.
11. **Uniform ratification posture** — every AVAILABLE baseline value of a
    searchable axis is `pending` until owner decision 2 (including the
    unbounded parent-retest baseline).
12. **Windows lock-contention semantics** — catalog lock treats
    `PermissionError`/`OSError` (delete-pending) as contention; stale
    orphaned locks broken via mtime threshold.

## R2

13. **Search-lock liveness = checkpoint heartbeat + stale break** — the
    per-search O_EXCL lock is refreshed (mtime) at every atomic checkpoint;
    resume breaks an orphaned lock only once its heartbeat is provably older
    than `stale_lock_seconds` (default 3600 s — far beyond any child
    boundary gap). Same Windows-truthful contention class as R1 #12.
    DEV-R2-1…6 in `R2/DEVIATIONS.md` record the audit-capture retention
    seam, the companion stores/envelopes, the job-shim runner-entry rule,
    the canonical-naming content guard, and the contrast scope/orientation.
14. **Frontier objective directions** — `ObjectivePolicy.pareto_objectives`
    carries no per-metric direction; the orchestrator maps a fixed minimize
    set (`max_drawdown_r`, `time_under_water_days`, `top_day_pnl_share`,
    `top_setup_pnl_share`) and maximizes everything else. Engineering
    default, recorded here; the charter identity still pins the exact
    objective tuple.
15. **Bootstrap-CI gate semantics** — `require_bootstrap_ci_excludes_zero`
    passes iff the trading-day cluster-bootstrap 95% CI on mean net R
    excludes zero literally (low > 0 or high < 0); when the flag is off the
    check is emitted as an explicit not-required pass so every report covers
    all eleven thresholds; when required but unavailable (fewer than two
    trading days) it fails with that reason.
16. **Per-child audit gate = neutrality, never parity** — child audit
    companions publish only behind a passing `ChildAuditNeutralityReport`
    keyed to the same `core_replay_id` plus the exact funnel⇔audit
    reconciliation; the accepted doc-default parity gate stays doc-default-
    only (untouched).

## R3

17. **Deferred prop-owned pareto objectives** — at the strategy-gate stage
    the orchestrator partitions pareto objectives by `hasattr` on
    `StrategyMetrics`: strategy-owned objectives must resolve there (None
    still excludes); prop-owned objectives are deferred to the prop phase
    ONLY when a `prop_simulator` is wired, else the child is explicitly
    excluded ("objective unavailable") — never silently dropped. Discovered
    by the seam tests: the first seam draft excluded every child whose
    charter carried a prop objective before the simulator could run.
18. **ALL-legs feasibility + worst-firm merge** — one prop-gate-failing
    simulation rejects the child (conservative), and each prop objective
    enters the frontier as the WORST value across the child's simulations
    (min for maximize metrics, max for minimize). Engineering default;
    the gate thresholds themselves remain `proposed_protocol_default`.
19. **Account walk extends, never edits, the evaluation engine** — the
    lifecycle walk is additive beside `engine.py`; compatibility is proven
    by the one-contract `AccountWalk`≡`EvaluationWalk` parity fixture
    (fees/payouts disabled) instead of by modifying the accepted walker.
20. **Event total order = walk-emission order surfaced as `event_ordinal`**
    (`prop_account_event_order_v1`): same-timestamp ties resolve by the
    fixed kind precedence baked into the deterministic day walk; consumers
    sort by the global ordinal, never by timestamp alone.
21. **Projection-audit envelope extras** — identity pairs may declare
    `extra_envelope_fields` (post-materialization facts such as artifact
    content hashes); the audit test supplies placeholder facts and still
    enforces payload → id → envelope → reload determinism, so
    post-materialization fields can never enter a pre-run identity.
22. **Scenario semantics are position-relative and walk-effective** — the
    two registered intrabar policies map to two breach-observation modes
    (`unrealized_adverse_first` = MAE tested before any MFE peak-raise;
    `unrealized_favorable_first` = MFE peak-raise first); the artifact
    builder's per-bar "adverse" is the POSITION-adverse extreme (long→low,
    short→high). Same policy id = same semantics in artifact and walk;
    order-sensitive trades provably diverge (review blocker 1).
23. **One-directional breach-observation compatibility** — a firm requiring
    unrealized observation refuses realized-only walks; the reverse
    (over-conservative unrealized scenario on a realized-only firm) is
    permitted, labeled by the mode. Keeps the parity surface runnable.
24. **Bootstrap horizon rides the protocol id** — the registered family
    `day_block_bootstrap_h<horizon>_v1` puts the sampled-path length inside
    the simulation identity; the runner's former `max_days_per_path`
    argument is deleted (review blocker: same id, different results).
25. **Runner revalidation is mandatory** — `run_account_simulation`
    re-hashes every supplied policy/report/bundle against the identity
    fields pinning them and derives fidelities/event ids from artifacts;
    `supersessions` is a pure refusal registry (can only block, never change
    a number) and is the one non-identity runner kwarg.

## R4

26. **Axis→market-meaning grouping + classification projection** — the 44
    registered axes map onto the seven FUX §10.1 groups via a fixed
    presentation table (`AXIS_GROUP_ASSIGNMENTS`, unmapped keys would land
    in a tested-empty "Other"); the seven registry classifications project
    onto the five FUX §10.3 visible labels (THESIS_DEFINING renders
    "Locked Invariant" — a thesis change is a new profile via the
    interpretation selector, never an edit; RISK_POLICY_AXIS renders
    "Search Axis" with the "Prop Resimulation" chip). Presentation only;
    the registry stays the identity authority.
27. **Question↔mode compatibility matrix** — each search mode answers its
    natural FUX §8.1 question (1:1); Full Pipeline Run accepts all four;
    prop-evaluating modes refuse the strategy-only template with an
    explanation. Engineering default (DEV-R4-12).
28. **Objective templates resolve to registered metrics only** — the six
    FUX §8.2 templates pin `pareto_objectives`/`lexicographic_tie_breaks`
    drawn from `OBJECTIVE_DIRECTIONS` (strategy-only charters drop
    prop-owned objectives to keep the frontier resolvable, per #17);
    every template is stamped `proposed_protocol_default`; Custom offers
    the registered metric list, nothing free-form.
29. **Draft policy** — drafts are mutable until `mark_frozen` (the single
    permitted final write, stamping the frozen `search_id`); frozen drafts
    refuse save/discard forever and clone via deep copy; unreadable draft
    files are skipped by listings, never fatal. Draft ids are uuid4 (drafts
    mutate; content addressing would churn), path-component-validated.
30. **The state file is the run locator** — run enumeration lists the
    MUTABLE job root (64-hex dirs) + catalog display names; the frontier
    is found via the new `phase_notes["frontier_id"]` pointer
    (DEV-R4-2); costed evaluations re-derive their deterministic lookup
    id from (core_replay_id × cost policy). The immutable store roots are
    never listed (provider-tested, including the catalogued-ghost case).
31. **Funnel/stage derivation table** — Generated=children;
    Replay Valid=state∈{completed,reused}; Strategy Pass excludes the
    strategy-stage reason set {insufficient_trades, insufficient_days,
    negative_expectancy, drawdown, knife_edge} AND the reused-unevaluated
    sentinel; Prop Feasible/Robust come from the persisted frontier
    (feasible_ids/frontier_ids) when present, else prop-stage reasons +
    the prop-crash and frontier-exclusion sentinels (state-only Robust
    renders "–"). Child-stage knife_edge counts as a strategy-stage
    failure (robustness is a comparison-surface evaluation, per the
    orchestrator's phase note). Sentinels contract-tested against the
    orchestrator source (DEV-R4-13).
32. **Feasible children render "Robust Finalist"; dominance is ranking,
    not a gate** — every frontier-feasible child passed all benchmarks
    (status ✓); frontier membership orders them and the representative
    renders the Development Exploratory Representative status; a child
    excluded for an unavailable objective or an unevaluated reuse renders
    Blocked with its exact explanation.
33. **UI namespace + evidence-action budget** — the workspace reads one of
    two namespaces (research/verification) with truthful badging
    (DEV-R4-9); insight/timeline evidence actions queue exact verifier
    jumps ONLY for the four supported id kinds (setup/trade/candidate/
    decision); simulation evidence routes to the timeline; account_event/
    child/axis render copyable identities (the verifier has no jump target
    for them — recon note honored, nothing fuzzy substituted).

## R5

34. **Stage reuse is proven by identity, never assumed** — every pipeline
    stage executor is idempotent and store-reusing; a retry RE-EXECUTES the
    stage and marks it REUSED only when the freshly-minted
    `PipelineStageResultEnvelope` id equals the prior attempt's verified id
    (in-memory stages — views/labels/folds/ladder — re-derive
    deterministically instead of deserializing). Prop vectors are the one
    heavyweight exception: S12/S13 persist them as stage sidecars and a
    later attempt whose children were reused without rebuilt tables loads
    the PRIOR attempt's verified vectors instead of skipping the children.
35. **Bootstrap horizon default** — the pipeline's S13 pins
    `day_block_bootstrap_h90_v1` (`BOOTSTRAP_PROTOCOL_ID_DEFAULT`; the
    horizon rides the simulation identity per R3 #24). Engineering default,
    unratified for research.
36. **Mode-5 drafts assemble the underlying search charter** — a Full
    Pipeline Run draft maps onto `fsm_config_search` when axes were
    selected, else `single_configuration` (the pipeline runs OVER that
    charter; `search_mode` has no pipeline value by design).
37. **Bridge mode bridging (DEV-R3-11 closure)** — `make_prop_simulator`
    gained additive mode parameters with R3-identical defaults; multi-mode
    vector keys are `"<label>:<mode>"` (stress adds `":<scenario id>"`) so
    every leg enters the conservative ALL-legs rule;
    `historical_ordered_event_replay` is REFUSED at this seam (reserved for
    actual ordered streams; none is wired); scenario mode requires
    caller-supplied per-trade bar observations (never fabricated).
38. **UI stage plans + synthetic policy pinning** — the Configure phase
    offers two capability-scoped plans (strategy-only; full 16-stage).
    Prop-bearing UI launches pin `synthetic_firm_specs()` (the ONE
    canonical synthetic spec set shared with the registered synthetic
    wiring) as the spec's `account_policy_set_ids` until first-party
    contracts are owner-verified; S00 fail-closed-verifies the pins against
    the wiring.
39. **`pipeline_runner_planned` retired; `pipeline_no_runs` added** — the
    R4-era planned-capability presentation died with the R5 surface (the
    no-dead-vocabulary rule, DEV-R2-7 precedent); the dedicated no-runs
    presentation replaces the wrong-state reuse the R4 review flagged.
    `runner_executor_planned` survives with updated copy: it now names the
    operator-run boundary (real full-development charters keep NO
    registered executor).
40. **Real executors fail before path at FACTORY construction** — the
    registered baseline-verification entries refuse without a persisted
    `VerificationRunEnvelope` (discovered via the provider, stores never
    listed) BEFORE any config, policy, or source path is built; their
    child runners refuse every non-baseline child (real multi-child is
    synthetic-only by the two-path design).
41. **Ladder↔bundle mapping is tier-frozen in R5** — the pipeline's S09
    maps a bundle onto `run_supervised_ladder` only when the bundle's
    resolved feature SET equals a frozen tier bundle exactly
    (`frozen_tier_for_bundle`); anything else fails closed rather than
    improvising a feature list (bundle-parametrized ladders arrive with
    the R5B/R6 feature lanes).
42. **Search-lane comparison results carry a typed derivation id** — S14's
    persisted `ComparisonResult` names its subject through the R5-FIX typed
    reference `SearchDerivationComparisonSubject.derivation_id`: the
    canonical hash of `{kind: cross_profile_population_delta_v1, search_id,
    baseline/challenger core replay ids, changed axes}` (study-cell
    `ComparisonEnvelope`s require the children's published v2 dataset
    references, which synthetic control-flow children do not have); the
    envelope id hashes the full result content, so identical re-computed
    deltas reuse one immutable artifact. (Originally a bare `comparison_id`
    string field; retyped by R5-FIX #45.)

## §R5-FIX (post-gate fix round, 2026-08-21/22)

43. **Production registry purity** — `REGISTERED_RUNNER_ENTRIES` holds src
    executors ONLY and the production module never names a `tests.*` path
    (test-witnessed, incl. a source scan). Synthetic fixture wiring reaches
    resolution exclusively through the guarded
    `register_development_runner_entries` extension point (keys must carry
    the `synthetic` marker, production keys unshadowable, values exact
    `module:function`, validate-then-commit, idempotent-on-match, conflict
    refusal), registered by `tests/agents/ifvg_search/conftest.py` at
    import. A production worker process — where no registration ran —
    refuses the synthetic keys exactly like unregistered entries.
44. **The real scope's expected seed is evidence, not an argument** — S00
    refuses the real verification branch without
    `PipelineWiring.loaded_seed_snapshot_id_source`; the source performs
    the VERIFIED seed-snapshot store load (manifest + file hashes +
    id-hashes-payload + seed-bytes rehash + profile binding) and returns
    the loaded envelope's content-derived id, which is what
    `validate_verification_run` checks against the owner authorization.
    `expected_seed_snapshot_id` survives as an optional cross-check only;
    set-but-disagreeing values refuse before any source path.
45. **Comparison subjects are a discriminated union** —
    `ComparisonResult.subject` is `StudyCellComparisonSubject`
    (`comparison_id` naming a frozen `ComparisonEnvelope`) |
    `SearchDerivationComparisonSubject` (`derivation_id`, DECISIONS_TAKEN
    #42) discriminated on `subject_kind`; the identity domain is enforced
    by the contract, and a bare 64-hex string is no longer a lawful
    subject.

## R5B

46. **The published registry IS the activation event** — `feature_blocks.py`
    applies `with_activated_block` to the R5-era planned state at module
    level and publishes the result as `FEATURE_BLOCK_REGISTRY`; the
    pre-activation mappings stay exported (`PRE_ACTIVATION_*`) so the
    versioned event (version bump, first resolved id, registry-hash change,
    B2/B3 re-mint, B1/B4 stability) is provable forever, not a one-time
    migration. Import-time invariants assert version 2 / available / the
    hash change / the definition-level research boundary.
47. **Databento normalization defaults** — prices decode under the pinned
    `databento_fixed_price_1e9_v1` scale with a fixed per-instrument tick
    map (NQ/ES = 0.25; unknown instruments REFUSE rather than guess);
    unsigned vendor integer columns widen to int64; `source_ordinal` is
    assigned in SOURCE row order BEFORE the stable sort by the complete
    four-part key (it is the deterministic final tie-break).
48. **Sequence-gap and coverage semantics** — *(AMENDED by R5B.1 — see #67:
    the sequence-jump rule is WITHDRAWN and unrepresentable; coverage is
    evidence-based under policy v2 (#68–#75). Retained verbatim as the R5B
    historical record.)* — only a positive sequence jump
    > 1 marks the interval between the adjacent events invalid; vendor
    sequence RESETS (a decrease) are not gaps. Day coverage = 1 − (gap
    nanoseconds / evidence span); `MIN_DAY_COVERAGE_FRACTION = 0.95` is the
    engineering default below which every window of that day types
    `coverage_below_threshold`. Unratified for research like every
    engineering default.
49. **v2 anchors are completed-bar cutoffs; typed-missing precedence is
    fixed** — *(precedence AMENDED by R5B.1 #70: `sequence_gap` is withdrawn;
    `coverage_evidence_unavailable` and `declared_source_gap` replace it)* —
    the five registered lifecycle anchors map onto the candidate
    row's own timestamps (htf_tap→`tap_ts_utc`, parent_lock→`lock_ts_utc`,
    opposing→`armed_ts_utc`, inversion→`inversion_ts_utc`,
    entry→`entry_ts_utc`) as `COMPLETED_BAR_BOUNDARY` cutoffs under
    `completed_bar_boundary_exclusive_v1` (bar-close decisions: numerically
    equal feed timestamps are never assumed known before the decision).
    Missing reasons evaluate day → anchor → boundary → content:
    `no_mbp1_partition` → `coverage_below_threshold` →
    `stage_outside_coverage` → `same_timestamp_order_unavailable` →
    `sequence_gap` → `instrument_roll_boundary` →
    `minimum_event_count_not_met`; formula-edge NaNs (zero denominators,
    zero trades) stay VALID.
50. **Controlled-study shape** — the baseline arm is the challenger's OWN
    `base_bundle_key` (composition-by-extension is the preregistered block
    ablation); both arms run bundle-parametrized prevalence + logistic
    rungs (`ifvg_context_logistic_l2_v1` pinned — the CatBoost fold runner
    is tier-locked inside the frozen M0–M3 lane and refuses with exactly
    that reason at the ladder, the stage-plan readiness, and the UI); both
    arms keep the candidate view's `view_id` so `oos_row_id` pairs exactly
    across arms, with the evidence divergence pinned instead by the bundle
    view's and the study payload's `mbp1_feature_artifact_id`; cross-arm
    row identity AND a numerically identical prevalence reference are
    asserted before any delta.
51. **`BundleFeatureViewPayload.mbp1_feature_artifact_id` + ladder
    `feature_source`** — the view identity now pins joined MBP-1 evidence,
    and the ladder id binds a typed feature source ({frozen_tier} |
    {resolved_bundle, feature names}) instead of a bare tier string.
    Derived (in-memory) identities computed after this change differ from
    R5-era values; stage retries across the boundary legitimately
    re-execute (reuse is proven by identity) and no immutable artifact is
    mutated. Pre-acceptance schema evolution, recorded in DEVIATIONS.
52. **Module/mount placement** — the controlled-study workflow lives in
    `ifvg/ml/controlled_feature_study.py` (the plan's R5B file list names
    the six `features/` modules; deliverable 10 needed a lane-correct home
    beside the ladder it parametrizes), and the R5B dashboard renders as
    the "MBP-1 Order Flow (research-only offline)" expander on the
    pipeline surface plus the Configure-phase bundle/model integration
    (FUX §35 mandates the panels, not a mount point; the fixed sub-nav is
    untouched). The activation decision extends `docs/DECISIONS.md` as
    D-046 — the number after the reserved block — per the R5-era
    reservation note that promised the activation entry at R5B.
53. **DEV-R5-10 closed** — `ifvg_search_job.py` now invokes runner-entry
    factories as `factory(charter, store_root=<worker --store-root>)`,
    matching the pipeline shim's contract; the real search entry receives
    the worker's store root instead of defaulting to the canonical
    namespace, and the registered synthetic factory accepts the keyword.

## R6

54. **The assignment frame projects the plan's per-row contract onto
    KMeans-shaped columns** — the plan's `RegimeAssignment` lists
    `distances: tuple[float, ...]` and GMM-only fields (`probabilities`,
    `log_density`); the V1 frame persists the full `distances` list to
    every fold-local centroid (the registered `kmeans_v1` output),
    `assigned_distance` + `assignment_margin` (d2−d1), and the
    observation's as-of timestamp `observation_ts_utc`, with
    `assignment_entropy`/`log_density`/`outlier_score` carried as NaN
    columns (schema stable for the post-V1 algorithms; `probabilities`
    arrives with GMM). `RegimeAssignmentColumns` names exactly the
    persisted columns (review F13).
55. **Assessment/promotion are content-addressed envelopes** — the plan
    types `RegimeCapabilityAssessment`/`RegimePromotionDecision` as bare
    contracts; persisting them requires store keys, so both gained
    Payload/Envelope wrappers (ids hash content, like the coverage-report
    precedent) registered with the identity audit. Additive.
56. **Bootstrap-stability protocol** — 50 seeded refits (deterministic
    seeds 1000+i over `default_rng(7)` resamples of the REFERENCE fold's
    training matrix only — test-pinned), Hungarian-aligned to the fold fit
    with the EXACT ascending-local-id tie-break (enumerated for k ≤ 8;
    review F8), scored by adjusted mutual information; the ≥0.5
    aligned-AMI floor stays advisory (a stability-gate failure blocks
    promotion, never deletes a fit). Engineering default under the plan's
    "50 seeded refits" sketch.
57. **Panel→candidate assignment is OOS-only, lowest fold, order-free** —
    a candidate receives the frozen OUT-OF-SAMPLE (`partition == "test"`,
    valid) regime of the last COMPLETED bar at or before its as-of instant;
    when several OOS folds cover that bar the lowest `fold_index` wins,
    independent of input row order; in-sample rows are never consulted; a
    bar without an OOS assignment types `coverage_gap`; the output carries
    `regime_fit_id` and `partition`. Replaces the midpoint "first valid row
    in (row_id, fold_index) order" rule, which was neither OOS-only nor
    what the code did (review F4).
58. **Regime UI mount and the stratification boundary** — the Regime Lane
    renders as a pipeline-surface expander beside the MBP-1 panel (FUX §35
    mandates the views, not a mount point; the fixed five-entry sub-nav is
    untouched), with exact-ID loads only (stores never listed) and no
    promote/launch/rank/retrain control (source-scanned). It delivers the
    coverage, occupancy, stability, ASSIGNMENT (exact fit id: coverage by
    partition, distance/margin quantiles, the OOS regime timeline), and
    STRATIFICATION-of-the-assignment-frame views plus the promotion
    role/status view; stratified RESULT views (strategy/model/prop metrics
    by regime) are the ML §5.5 comparison classes and land with their
    studies (DEV-R6-4 — an owner-visible scoping deviation).
59. **The promotion ladder is structural** — the `RegimePromotionDecision`
    contract itself enforces one forward step at a time (blocked and
    non-promoting states — descriptive, experimental, blocked — are always
    reachable and resume the ladder from the bottom; SUPERSEDED is
    terminal; PLANNED is never a target), a chained
    `previous_decision_ref` (None exactly when the previous status is
    planned), an ISO-8601 `decided_at` with an explicit offset, a 64-hex
    `owner_ratification_ref` (the `OwnerDecisionEvidenceRef` content hash)
    from FEATURE_ELIGIBLE onward, and the ROLE ladder — execution-side
    roles (decision policy, execution-gate candidate, frozen execution
    gate) are unrepresentable in V1 and carry the exact S11 blocked
    reason; every other role requires the status that earns it.
    `persist_regime_promotion` loads the referenced assessment (must exist,
    verify, and name the protocol) and the chained previous decision, and
    re-checks the ladder with the assessment's OWN `gates_passed` — so
    FEATURE_ELIGIBLE+ over a failing or absent assessment is
    unpersistable (reviews F5/S1/S2). Corrects the midpoint entry, whose
    "at both … the contract itself" claim the code did not honor.
60. **Identity extensions after the adversarial round** — the protocol
    identity gains `pinned_parameters_hash` (the registry's pinned
    parameters, re-verified at run time — a registry edit never executes
    under an old id; review F12), `initialization_policy` is validated
    against the registered set (every planned protocol is EXPRESSIBLE),
    and `input_feature_bundle_ref` is 64-hex; the fit identity gains
    `training_feature_matrix_hash` and requires non-empty verified 64-hex
    `source_artifact_ids` (two fits over different feature values can
    never share an id; review F1); the stability report gains
    `temporal_order_policy`, `temporal_transition_count`, and
    `alignment_space`. Derived identities differ from the midpoint tree;
    no persisted regime artifact existed outside tmp roots (DEV-R6-6).
61. **Input permission is default-on** — `resolve_kmeans_protocol` and
    `run_regime_protocol` resolve `input_feature_bundle_ref` against the
    registered AVAILABLE bundles (an unknown id refuses), refuse any input
    outside that bundle, and apply the availability-stage rule from the
    feature-block registries without a caller-supplied map (a feature with
    no registered stage is refused as unprovable; reviews F2/S4). No
    panel materializer / panel fold builder ships in V1 — the panel fit
    path is proven on a synthetic 5m panel with hand-built walk-forward
    folds (DEV-R6-7).
62. **Protocol policies fail closed** — each registry entry names its
    `executable_policies`; `assert_protocol_executable` refuses (before
    any preprocessing or fit, and at the private per-fold seam too) any
    protocol whose algorithm is planned, whose version or pinned-parameter
    hash drifted, or whose policy values (`cluster_count_policy`,
    `dimensionality_reduction_policy`, `kernel_or_affinity_policy`,
    `out_of_sample_assignment_policy`, …) lie outside the executable set
    (reviews S3/S9). The store's trust boundary is recorded (integrity,
    not authenticity — DEV-R6-8; reviews S6/S7).
63. **Temporal facts come from the OOS timeline** —
    `temporal_persistence` and the `transition_matrix` are computed over
    valid TEST-partition rows ordered by `observation_ts_utc` (first
    present of `entry_ts_utc` / `feature_as_of_ts` / `bar_close_ts_utc`;
    the observation frame must carry one) WITHIN each fold — no cross-fold
    concatenation, no row-id ordering, no double counting of overlapping
    training windows (review F3); `temporal_transition_count` records the
    evidence size.
64. **Canonical ids are geometry-ranked** — the reference fold's canonical
    reporting ids are the ranks of its centroids under the lexicographic
    order of their scaled input-feature coordinates (ties by local id),
    so canonical ids depend on the fitted geometry only — never on the
    training row order or the spelling of row ids; later folds are
    Hungarian-matched onto the reference and mapped through those ranks.
    Cross-fold comparisons (alignment, recurrence, separation, descriptors)
    use the scaled INPUT-feature space only — fold-dependent
    missing-indicator columns never enter (review F9). Ids stay NOMINAL.
65. **Fit hygiene** — all-missing training rows are excluded from the fit
    and typed `source_feature_missing` (identity and coverage agree;
    review F10); duplicated observation keys are refused (review F11);
    the candidate-stage / decision-row / panel sample-adequacy minimums are
    read from the stamped `REGIME_PROPOSED_DEFAULTS` (the decision-row
    minimum is now stamped; review F14).
66. **Persistence verifies before it publishes** — `persist_regime_fit`
    binds the assignment frame to the fit (fit id, fold, protocol, and the
    estimator's own labels; review F6), verifies the EXACT bytes it will
    publish (reload from bytes → re-transform allclose → re-predict equals
    the persisted labels), publishes, re-verifies through the store, and
    withdraws a fresh entry that fails (review S8); the shared
    `load_sidecar_bytes` verifies the manifest hash, whitelists the
    sidecar name, and hashes the returned bytes (review S6); the UI loads
    JSON + Arrow only and never unpickles (review S7); both panel scripts
    joined the FUX source scans (review S5).

## R5B.1 (MBP-1 coverage-policy correction; plan §6.I / D12; owner Q1)

67. **The sequence-jump gap rule is withdrawn and unrepresentable** —
    `Mbp1SourceContract.gap_semantics` is the Literal
    `mbp1_source_coverage_declared_evidence_v2` and
    `sequence_jump_semantics = sequence_jump_diagnostic_only_v2`; the R5B
    `sequence_gap_marks_interval_invalid_v1` value fails validation. Raw
    sequence jumps/resets and `ts_recv` spacing are recorded as diagnostics
    (`Mbp1SequenceJumpDiagnostics`, `Mbp1TsRecvGapDiagnostics`) and never
    open an interval, type a reason, or reduce coverage (DECISIONS_TAKEN #48
    amended; DEV-R5B-6 superseded).
68. **Coverage policy v2 is evidence-based at an explicit scope** —
    `Mbp1EvidenceScope` (dataset/publisher/channel/instrument partition,
    physical partition key, UTC date, verified expected span); accepted
    kinds: declared partition gap manifest, `F_MAYBE_BAD_BOOK`
    (DBN flag bit 4), dataset-condition record. Positive completeness exists
    only through `compile_mbp1_partition_gap_manifest` over a verified
    `Mbp1CompletenessCompilationReport` (source inventory + owner review;
    `positive_completeness_authorized`); a manifest with
    `completeness_assertion="none"` is negative evidence only. Dataset
    conditions map to the five `vendor_*` states and can downgrade
    (degraded/pending/missing → `completeness_unknown`) but never prove
    completeness.
69. **`F_MAYBE_BAD_BOOK` opens an interval that closes only at a documented
    recovery boundary** — start = manifest-declared start, else the last
    trusted in-scope event (no bad-book / bad-`ts_recv` flag), else the
    partition's expected start; end = the first documented recovery
    boundary (manifest end / vendor recovery event / snapshot recovery /
    owner-approved), else `partition_expected_end_ts` with
    `open_uncertainty_to_partition_end=True`. The next unflagged row never
    closes it. Scope = channel partition only behind a verified channel map,
    else publisher/physical partition; never the flagged instrument.
70. **Coverage arithmetic (D12)** — per physical partition the denominator is
    `intersection(verified partition span, authorized session span)`
    (18:00 ET previous day → 17:00 ET, DST-aware); intervals from every
    evidence kind are unioned, merged once, and clipped; multiple UTC
    partitions of one trading day are measured separately and
    duration-weighted with the weakest status winning; head/tail gaps count
    only when declared; an empty span is unknown. Physical partition spans
    are HALF-OPEN `[start, end)` and, after clipping to the session span,
    must be pairwise disjoint (overlapping/duplicated partitions are refused
    — review F3); an in-session event outside every declared span refuses
    the build (the manifest contradicts the data), and session sub-spans no
    declared partition covers are recorded as `uncovered_session_intervals`
    on the day view (review F2). The materializer types
    `no_mbp1_partition` → `coverage_evidence_unavailable`
    (`completeness_unknown`) → `coverage_below_threshold` →
    `stage_outside_coverage` → `same_timestamp_order_unavailable` →
    `coverage_evidence_unavailable` (a window touching an uncovered session
    sub-span) → `declared_source_gap` → `instrument_roll_boundary` →
    `minimum_event_count_not_met`; windows are never widened or imputed and
    every candidate row is preserved.
71. **Every affected identity re-mints (DEV-R5B-3 precedent)** — the
    normalized event schema retains `publisher_id` + `flags` (new schema
    hash → new source artifact ids); `IFVG_ORDER_FLOW_MBP1_V1` is
    re-resolved as a SECOND versioned event via the new pure
    `with_reresolved_block` (block v3, `ifvg_order_flow_mbp1_formula_v2` /
    `mbp1_feature_materializer_v2`, new resolved block id, registry hash,
    B2/B3 bundle ids); the R5B activation payload keeps the historical v1
    versions (`MBP1_FORMULA_VERSION_V1`) and the R5B state is exported as
    `PRE_R5B1_*`; feature artifacts, coverage reports (`coverage_policy_id`),
    and controlled-study ids move. No persisted real artifact exists outside
    tmp/smoke roots; nothing immutable was mutated.
72. **Synthetic evidence is scope-bound; the real diagnostic fails before
    any path** — `evidence_provenance ∈ {owner_reviewed, synthetic_fixture,
    none}` rides every partition row; the pipeline's S05
    (`assert_evidence_provenance_permitted`) and
    `build_mbp1_source_artifact_from_paths` refuse synthetic provenance
    outside the synthetic marker. `scripts/ifvg_mbp1_coverage_diagnostic.py`
    is gated by the R1 real-slice gate (review S2/F9; #75): a persisted,
    verified `VerificationRunEnvelope` whose `VerificationAuthorizationRef`
    binds the requested allowlist (`allowlist_hash == sha256(allowlist) ==
    approved_allowlist_hash`), the verification policy id, the
    `search_test/v1` namespace, a verified-loaded coverage-matrix artifact
    over exactly those days, and the ONE canonical program allowlist
    (`register_program_allowlist`); the access policy must BE a
    `VerificationReplayPolicy` over the same days; the store root must END
    in `search_test/v1` (a substring is never a namespace — review S4). It
    pins `completeness_inferred_from_sequence_continuity=False`
    structurally; the synthetic path proves the report shape only. Real
    MBP-1 research and R5B acceptance stay blocked until policy v2 passes
    on the real fixture (an owner action).
73. **Positive completeness is bound to the partition content it certifies
    (review F1/S3)** — `Mbp1PartitionEvidence` carries the compilation
    report whenever its manifest asserts completeness (report id, scope,
    provenance, intervals, and `positive_completeness_authorized` must
    agree — the contract itself enforces what the store loader used to);
    the artifact builder refuses a positive claim whose
    `verified_partition_refs` do not intersect the partition's content
    hashes (`content_sha256` of the canonical bytes; the raw file sha256 on
    the real path); a provenance label without a manifest is
    unrepresentable (evidence rows and partition rows alike); the report's
    refs are SHA-patterned and `owner_review_decision_id` is a 64-hex
    decision hash; `source_document_sha256` must be one of the report's
    `evidence_refs` and never the report's own id (review S7). Binding the
    owner-review hash to a persisted owner-decision artifact is the R6.1
    owner-evidence workstream (DEV-R5B.1-2).
74. **The real read seam clips the UTC-date file to the trading day and the
    development cutoff (review S1)** — `read_mbp1_partition_frame` keeps only
    rows inside `[18:00 ET D-1, 17:00 ET D)` and before
    `DEVELOPMENT_CUTOFF_UTC` (the 18:00 ET tail of the last exposed day —
    trading day 2026-06-11, protected — is therefore structurally excluded
    before normalization or hashing); the clipped counts and the raw file's
    sha256 ride the partition row (`rows_outside_session_span`,
    `rows_after_development_cutoff`, `source_content_refs`). The previous
    UTC file's 18:00 ET → midnight portion is NOT composed in R5B.1
    (DEV-R5B.1-1): with declared evidence that span is an uncovered session
    interval whose windows type `coverage_evidence_unavailable`; without
    evidence the day is `completeness_unknown` anyway.
75. **Vendor-flag interval closure and scope are exact (reviews F5/F6/F7/F12)**
    — the flagged record is always inside the uncertainty (a boundary AT
    the flagged instant closes after it); a boundary documented beyond the
    partition end never closes the interval (fail closed to the partition
    end); recovery boundaries and dataset-condition records are bound to
    the partition scope (foreign ones refuse); merged intervals carry every
    `contributing_evidence_kinds`; a channel-scoped label requires a
    channel-scoped scope behind a verified map; the materializer refuses a
    superseded (v1) resolution and stamps versions FROM the block (F8); the
    R-6-family coverage defaults are a stamped `MBP1_PROPOSED_DEFAULTS`
    table surfaced by the MBP-1 panel (F10).

## R6.1 (the regime-lane correction; plan D1–D15; owner Q2–Q4)

76. **Every regime input is a verified-loaded artifact; every fold set is a
    persisted artifact** — `bundle_feature_views` / `context_bar_panels`
    stores + `RegimeObservationSourceRef` (`source_artifact_ids` from the
    loaded envelope only); `FoldScheduleEnvelope.fold_schedule_id` + the
    persisted `FoldSetArtifact` (candidate or panel) under `fold_schedules` /
    `fold_sets`; cross-grain studies compare schedules, never row-population
    fold-set ids (D2/D3).
77. **The descriptive study runs S09a only, but S07 stays in the plan** —
    the frozen 16-stage dependency table is unchanged; `model_protocol_id`
    may be `None` when a regime study is present (the one validator
    relaxation); the label-free candidate fold builder serves S08 when
    labels are absent (DEV-R6.1-1).
78. **Pre-acceptance identity evolution** — semantic ids (`regime_study`),
    assessment ids (per-fold stability / transitions / renamed gate),
    bundle-view ids, bundle-path ladder ids (`comparison_row_id`), block
    registry hash (panel registration), simulation / charter / protocol ids
    (D15 policy fields, default `none_v0`); fit ids and
    `resolved_regime_protocol_id` unchanged (golden constants) (DEV-R6.1-2).
79. **Single-threaded regime kernel** — `run_regime_protocol` under
    `threadpool_limits(1)` for byte-reproducible assignment tables; an
    execution-environment control, never a protocol field (DEV-R6.1-4).
80. **Panel session state = the session in force at the bar's final instant**
    (`session_of_bar_final_instant_v1`); position measured from the close
    against that session's window (DEV-R6.1-5).
81. **Fit-local model features; canonical alignment reporting-only** — the
    `RegimeFoldFeatureArtifact` is the only supervised feature source (fit k →
    fold k; the leakage test perturbs every later observation and every
    other fit); the local id is a model feature only under
    `hard_id_encoding="categorical"` (stamped default `none`); the regime
    block's resolution is minted per protocol/activation (DEV-R6.1-7).
82. **Verified reuse by reproduction, everywhere** — fits (`persist_regime_fit`
    reloads and must reproduce transform + labels; bytes never rewritten),
    reused children needed by stratified reports (S02 re-derives and must
    reproduce the persisted costed evaluation), prior-attempt S14 reports
    (verified reload) (DEV-R6.1-8). **AMENDED by the R6.1 adversarial round**
    (S4): re-derived tables are adopted ONLY when they reproduce the persisted
    costed evaluation of THIS cost policy; with none they are not this run's
    evidence (reused, gates not evaluated, out of the stratified reports) (#98).
83. **Exact persisted simulation ids reach the stratified-prop reports** —
    `make_prop_simulator(on_simulation_persisted=…)` + the attempt-invariant
    `account_simulations.json` sidecar; never a store listing (DEV-R6.1-9).
84. **Frozen model-bearing authority** — `feature_only` / `cohort_model`
    requests freeze the exact promotion / owner / assessment ids (hashed into
    the semantic id; verified at readiness, S05, S09a, S10, and at block
    activation); descriptive requests derive DESCRIPTIVE_ONLY / BLOCKED /
    STRATIFICATION_READY deterministically at the evidence as-of instant; no
    "latest" lookup exists (D1/D4/D9).
85. **Owner-decision artifacts are the ratification evidence** — decisions
    25/28/29/30 incl. the exact pinned KMeans snapshot/hash, exact protocol +
    assessment binding, store-owned fail-closed supersession, synthetic
    provenance only in the synthetic scope; `persist_regime_promotion`
    requires the artifact from FEATURE_ELIGIBLE (a bare 64-hex reference is
    not evidence) (D5). **AMENDED by the R6.1 adversarial round** (S2/F8,
    S3, S5, S6, F2, F10, S11): the supersession log is hash-chained and
    written BEFORE the replacement is published; provenance may never
    weaken along a supersession; STRATIFICATION_READY structurally requires
    passing coverage gates + an OOS assignment at persistence AND the report
    gate re-derives it from the loaded assessment; `MODEL_FEATURE` is
    unpersistable in V1; the synthetic scope / provenance are confined to
    test namespaces (P0-4 mirror); `decided_at` derives from verified
    artifacts, never the wall clock or a flag (#90, #91; DEV-R6.1-15).
86. **S14 performs zero fitting** — stratified reports are built from
    persisted S09/S10/S12/S13 artifacts only (test-enforced by monkeypatched
    estimators); the modeled classes are S09c deliverables recorded at S14
    under `delivered_by` (verified exact S09c ids) — refusals only when a
    requested modeled class was not delivered (DEV-R6.1-10, amended by
    review F6; #92).
87. **The bundle-aware CatBoost rung runs on both arms; comparison rows are
    bundle-independent** — `ifvg_context_catboost_bundle_v1` (frozen-lane
    parameters by value; native NaN; registry ∪ block-declared categoricals);
    `comparison_row_id = hash{fold_schedule_id, candidate_fold_set_id,
    fold_index, candidate_id, label_artifact_id}`; the frozen M0–M3 lane is
    byte/identity-unchanged (D13).
88. **Prop-event detail is a versioned, bounded, immutable Parquet artifact**
    — policy / storage / schema / budgets in the simulation identities and the
    `SimulationProtocol`; ZSTD partitions by path block; preflight + streaming
    budgets fail before publication; `none_v0` never widened (D15).
    **AMENDED by the R6.1 adversarial round** (S12/F16, F3): the writer
    streams literally (one path block at a time through the store's
    sidecar-producer protocol; #93), and the `stratified_prop` report carries
    the bounded report-local `account_event_regime_summary.parquet` instead
    of per-event JSON (#92).
89. **Evidence hygiene** — hermetic provider tests; browser manifest v2 bound
    to the final commit and validated by `verify_browser_manifest.py`;
    `git format-patch` source-review patches per release (§6.K; final closure
    #8).
90. **Supersession is hash-chained and precedes publication** — every
    `SUPERSESSIONS.jsonl` line is a `SupersessionRecord` chained
    `prev_line_sha256 → line_sha256` with the head in `SUPERSESSIONS.head`
    (atomic); the line is appended (lock-guarded, idempotent, stale-lock
    reclaim) BEFORE `save_or_reuse_envelope` publishes the replacement, so a
    crash leaves a dangling line the loader treats as fail-closed until the
    replacement is re-persisted; an edited / deleted / reordered line or a
    missing head refuses the whole chain; a replacement may never carry
    weaker provenance than the prior (review S2/F8; DEV-R6.1-15).
91. **Promotion governance is structural on every path** — D6 at
    `persist_regime_promotion` (STRATIFICATION_READY needs
    `coverage_gates_passed AND oos_assignment_available`) and re-derived from
    the LOADED assessment by `resolve_report_gate` (never the status label);
    the gate's owner branch re-runs `assert_owner_decision_authorizes` under
    the run scope (`run_scope` threaded from `StratificationInputs`);
    `MODEL_FEATURE_PROMOTION_REFUSAL` is one constant shared by the store and
    the CLI; `assert_run_scope_lawful_for_root` confines the synthetic scope
    and synthetic-provenance owner artifacts (persist AND load) to test
    namespaces; `--decided-at` is gone — `decided_at` = max(previous
    decision, owner `effective_from`) with a monotone chain, so the CLI's
    STRATIFICATION_READY reuses S10's exact decision (reviews F2, S3, S5, S6,
    F10, S11, S10; DEV-R6.1-15).
92. **Stratified-prop reports carry the bounded event-regime summary; the
    modeled classes are recorded as deliveries** —
    `account_event_regime_summary.parquet` (schema
    `ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA` v1; one row per simulation × path ×
    regime stratum / typed reason × event type; keyed by the exact
    `RegimeAssignmentEvidenceRef` + simulation ids; budget
    `account_event_regime_summary_budget_v1` = 5,000,000 rows / 268,435,456
    bytes, typed refusal before publication; four envelope extras — final
    ruling 7) replaces the per-event JSON; the D15 `EVENT_TYPE_PRECEDENCE` and
    the event's own `trading_day` are the only ordering/day sources;
    `StratificationInputs.delivered_by` → verified exact S09c ids in the S14
    record (reviews F3, F6, F9, F12; DEV-R6.1-10/-11/-16). Every stratified
    report id moves (pre-acceptance).
93. **The store's sidecar-producer protocol; the D15 writer streams
    literally** — `ProducedSidecar` / `SidecarProducer` /
    `write_produced_sidecar`; `save_envelope_immutable(sidecar_producer=)`
    runs the producer into the temporary publication directory, re-hashes by
    streaming, refuses any bookkeeping lie / stray / ghost / reserved name
    with nothing published; reuse re-runs the producer into a scratch
    directory and requires manifest-hash reproduction; the event-detail
    writer builds one path block at a time column-wise (memory = one block +
    a 32-byte-per-row `event_id` index); partitions, manifests and simulation
    ids are byte-identical to the in-memory path (reviews S12/F16;
    DEV-R6.1-14).
94. **Deep-frozen study summaries, honest test names, lazy thread control**
    — `FrozenTree` / `FrozenJson` make every nested summary / paired-delta
    value immutable while the canonical bytes (hence the study ids) are
    unchanged (golden probe); the supervised-study tests assert what their
    names claim and the controlled regime study runs the three-rung bundle
    ladder on both arms; `run_regime_protocol` imports `threadpoolctl`
    lazily with `THREADPOOLCTL_MISSING_MESSAGE` (undeclared, DEV-R6.1-13)
    (reviews F15, F5, S9).
95. **The panel-grain candidate as-of source is a verified bundle-view ref**
    — `execute_regime_protocol(candidate_as_of_source: RegimeObservationSourceRef)`
    refuses a `(str, frame)` tuple, a bare string or a panel-kind ref
    ("not evidence"), verified-loads the bundle view through
    `load_regime_observations`, takes the as-of instants from the LOADED
    frame, and stamps `candidate_as_of_source_ref` as the loaded envelope
    line (`CANDIDATE_AS_OF_SOURCE_REF_PATTERN =
    ^(bundle_feature_view|context_bar_panel):[0-9a-f]{64}$`); S09a passes the
    PERSISTED primary bundle-view id (review F1). Panel-grain OOS artifact ids
    move where the ref differed.
96. **S05 binds the panel to the chart it asked for** — the declared
    selection policy `panel_chart_lowest_core_replay_id_v1` over S04's
    one-chart-per-child output; `replay.artifact_id == chart_id` and the run's
    pair are refused otherwise (typed; S05 fails closed; nothing persisted);
    the S05 sidecar records the loaded `replay_chart_artifact_id`, the policy,
    `panel_source_core_replay_id`, `replay_chart_source_pair_sha256`; the
    fixture seam loads exactly the requested id (reviews S1, F13;
    DEV-R6.1-18).
97. **S08 derives the fold schedule once** — from the candidate view's
    observed trading days (charter allowlist ∩ observed); the labeled folds
    are built from the SAME days; `trading_days_source` +
    `days_without_labels` in the S08 sidecar (review F7; DEV-R6.1-17).
98. **S02 adopts a reused child's re-derived tables only by verified
    reproduction** — a persisted costed evaluation for THIS cost policy must
    exist and be reproduced byte-for-byte; otherwise "reproduction
    unverifiable; the re-derived tables are not this run's evidence" (reused;
    S03 evaluates no gates; no evaluation published; out of the stratified
    reports) (review S4; DEV-R6.1-8 amended). AMENDED by R6.1-FIX #107: a
    persisted executed-trade table now verifies the reproduction under ANY
    cost policy (DEV-R6.1-FIX-3).
99. **S14's panel PIT assigner loads persisted artifacts only** — OOS
    assignment → protocol → panel envelope + frame → every fit's assignment
    sidecar, all by exact id; the executed-trade tables are S02's this-run
    tables because no persisted trade-table artifact exists (review F14;
    DEV-R6.1-18).
100. **§9 coverage closed; one hard-id vocabulary; UI readiness verifies the
    frozen authority before persisting** — `test_s14_performs_zero_fitting`
    parametrized over descriptive / supervised-candidate / supervised-panel
    runs (estimators monkeypatched inside S14), the supervised panel two-pass
    E2E (`test_model_bearing_panel_run_executes_s09b_s09c_pit_on_the_frozen_authority`),
    the S07 / label-policy refusal test; `hard_id_encoding ∈ {none,
    fit_local_categorical_v1}` from `regime_fold_features.py` only; the tab's
    Preview and Launch handler call readiness / launchability with the store
    root and the effective run scope so an unlawful model-bearing plan is
    refused BEFORE any charter / spec envelope or spawn (reviews F4, F11, S7;
    DEV-R6.1-7).

## R6.1-FIX

101. **Verified per-fit assignment evidence is the ONLY assignment source
    (F-01 / F-02)** — `load_regime_fit_assignments` returns
    `VerifiedFitAssignments` (envelope, artifact, the frame decoded from the
    manifest-verified sidecar bytes, the SHA-256 of exactly those bytes, the
    enforced schema hash); `persist_regime_fit` reuses an existing fit only
    when the candidate assignment bytes equal the stored sidecar byte-for-byte
    (different values under one identity fail closed); the executor
    exact-loads every fit it persisted (`verified_fit_assignments_for_run`,
    id + fold re-checked) and builds the descriptive OOS artifact, the panel
    PIT assignment and S10's evidence as-of from those frames only —
    `run.assignments` never reaches a persisted artifact (an in-memory frame
    is refused by type). `RegimeOosAssignmentPayload` binds
    `regime_fit_assignment_refs` (`FitAssignmentRef`, sorted by fit id; the
    `regime_fit_ids` projection validated against them),
    `resolved_cluster_count` and a `consulted_assignments_hash` over EVERY
    consulted value (fit, row, fold, partition, local id, canonical id, the
    full distance vector, assigned distance, margin, validity, reason);
    formula `regime_oos_assignment_v2`. `RegimeAssignmentEvidenceRef` pins the
    artifact's `assignment_table_sha256` + `assignment_schema_hash`; the panel
    assigner verifies every fit's sidecar against the ref the OOS artifact
    bound. DEV-R6.1-FIX-7 / -9.
102. **Enforced assignment schemas and cross-field invariants (F-05)** —
    `FIT_ASSIGNMENT_SCHEMA` (exactly `RegimeAssignmentColumns`, pandas
    metadata stripped; hash `FIT_ASSIGNMENT_SCHEMA_HASH`) serializes every
    fit sidecar; `validate_assignment_rows(kind=fit|descriptive|model_facing)`
    holds on build, save, verified load and downstream consumption: a valid
    row carries a 64-hex fit id, `fold_index >= 0`, a lawful partition
    (`test` only for the descriptive kind), a local id in `[0, k)`, `k`
    finite distances, finite assigned distance / margin, no reason, and the
    canonical id except on the model-facing kind; an invalid row carries NO
    output value and exactly one registered reason. Every value column is
    required — the optional-column fallbacks for distances / assigned
    distance / margin are gone; a differently typed sidecar is refused on
    load.
103. **Fold-feature source identity (F-03)** — `FoldFitRef` binds
    `assignments_sidecar_sha256` + `assignment_schema_hash` whenever a fit is
    present (all three null together only for an absent fit);
    `build_regime_fold_features(fit_assignments=…)` consumes
    `Mapping[int, VerifiedFitAssignments]` only (every fit of the run
    present; id / fold / protocol re-checked; fit k's VERIFIED rows for fold
    k); S09b passes the executor's verified evidence;
    `verify_fold_fit_refs_against_store` re-checks every ref by exact id on
    every load (no listing); `validate_fold_feature_rows` (the model-facing
    invariants) holds on build, load and the ladder seam. The in-memory run
    frame is proven INERT (tampering it changes no byte and no id).
104. **Candidate as-of policy: preserve with typed missingness (F-04)** —
    `candidate_as_of_missing` is registered in
    `PANEL_ASSIGNMENT_MISSING_REASONS` (hence in the fold-feature vocabulary);
    `_as_of_ns` returns `(instants, missing_mask)`; a null stage anchor yields
    ONE invalid row with that reason on both the descriptive and the
    fold-feature path; an unparseable NON-null instant stays a hard error;
    the as-of source hash represents the null deterministically; no candidate
    disappears.
105. **Raw thin-regime net-R accounting over the normalized frame (F-09 /
    F-10B)** — `RegimeNetRAccounting` (formula `regime_net_r_accounting_v1`,
    basis `all_valid_assigned_trades_v1`) persists `trade_count_by_regime`,
    `net_r_by_regime`, `unassigned_trade_count` / `unassigned_net_r`,
    `assigned_net_r_total`, `abs_net_r_mass`, `abs_net_r_share_by_regime`,
    `signed_contribution_fraction_by_regime`, `top_regime_abs_net_r_share`,
    the `works_only_in_regime` claim (true / false / null with
    `incomplete_assignment_accounting`) and the zero-denominator reasons —
    summed with `per_trade_net_r` (`(realized − cost) / risk`, the exact
    vector `compute_strategy_metrics` averages) over EVERY valid assigned
    trade; the reportability floor governs interval / reportability metrics
    only; `CohortDescriptiveBody` mirrors the claim (validator-enforced).
    `normalized_executed_trades` is computed ONCE and drives every join,
    stratum, metric and the binding `executed_trade_table_sha256`.
    DEV-R6.1-FIX-10.
106. **Exact label identity (F-08)** — `label_artifact_content_id` binds the
    registered label policy and every consumed label / economic column
    (`LABEL_CONSUMED_COLUMNS`: candidate id, setup id, trading day, entry /
    resolution instants and availability flags, binary target, gross / net R;
    formula `label_artifact_consumed_columns_v2`), order-invariantly; S07
    mints it as the label artifact id; `ControlledFeatureStudyPayload.label_artifact_id`
    is mandatory 64-hex with `label_identity_source ∈ {label_artifact,
    content_hash_unpersisted}` — the helper form (`label_policy_id=None`)
    can never be saved, compared as an immutable study, or promoted
    (`save_controlled_feature_study` refuses it). DEV-R6.1-FIX-8.
107. **The immutable executed-trade table (F-06)** — store
    `executed_trade_tables`; `EXECUTED_TRADE_TABLE_SCHEMA_V1` is the exact
    ordered 42-column projection `core_executed_trade_exact_v1` (declared,
    typed as the v2 capture, never inferred); `ExecutedTradeTablePayload`
    derives from the core replay so `executed_trade_table_id_for` exact-loads
    without listing; the envelope binds the projection SHA-256, the raw
    core-table hash the neutrality report hashes, row count and byte size.
    S02 persists it after a fresh completion (the neutrality core-table hash
    must agree) and after verified reproduction; a reused child is adopted
    for the stratified reports only when its re-derivation reproduces the
    persisted table byte-for-byte (else the persisted costed evaluation; else
    typed `executed_trade_table_unavailable`), and a reused child's costed
    evaluation under a cost policy that has none is published from the
    VERIFIED persisted table (amends #98 — R6.1's "different cost policy →
    unverifiable" applies only when neither table nor evaluation exists);
    corrupt evidence is a typed child failure, absence a typed fact. S14
    iterates the charter's child set, verified-loads every gated child's
    table by exact id, re-checks S02's hash, binds `executed_trade_table_id`
    into the `cohort_descriptive` body and the `source_metric_refs`, and
    records `children_evidence` + typed `children_skipped`
    (`child_not_completed_or_reused`, `strategy_gates_not_passed`,
    `executed_trade_table_unavailable`) — never a silent omission.
    DEV-R6.1-FIX-2 / -3.
108. **Fail-closed prior-stage sidecars (F-07)** — the typed probe contract in
    `search/store.py`: `probe_sidecar` → `present` /
    `sidecar_not_produced_for_path` (the ONLY lawful absence, proven by the
    verified manifest) / a `SidecarLoadError` with one of eight registered
    reasons (`store_entry_missing`, `malformed_manifest`,
    `manifest_hash_mismatch`, `envelope_identity_mismatch`,
    `sidecar_missing_but_manifest_declares_it`, `sidecar_hash_mismatch`,
    `malformed_sidecar`, `unexpected_io_error`); `has_sidecar`,
    `load_optional_sidecar_bytes`, `load_json_sidecar` derive from it. The
    prop-vector, account-simulation, lineage-map, S14 report-record and S09c
    run-record recoveries use it (no broad `except Exception` around an
    immutable-evidence load remains on those paths); S15 records
    `reload_failures` (`<store>/<id>` → sanitized reason) in the publication
    block and `s15_regime_reload_failures` replaces the boolean; a halted
    attempt marks every later planned stage PENDING
    (`_mark_downstream_not_run`; S11 keeps its blocked state).
    DEV-R6.1-FIX-1 / -6.
109. **Production correctness (F-10A / F-10C / F-10D)** — the five pipeline
    wiring `assert`s raise `PipelineWiringError` (typed `RuntimeError`,
    sanitized, survives `python -O`); the fold-feature builder's two internal
    guards are typed errors; `Mbp1PartitionEvidence` requires FULL scope
    equality with its gap manifest and a positive completeness claim requires
    `partition_content_refs` plus EQUALITY between the compilation report's
    `verified_partition_refs` and the complete ordered refs (intersection is
    refused; `None` is refused for a positive claim);
    `FrozenContract.model_copy` refuses a raw value for an enum-typed field
    and the MBP-1 materializer test passes enum members (the `.value`
    strings originated in the test's `model_copy`, not in production payload
    construction). DEV-R6.1-FIX-4 / -5.
110. **Golden identities pinned; re-mints enumerated** —
    `test_r61_fix_goldens.py` pins `resolved_regime_protocol_id`, the R6
    golden fold-0 `regime_fit_id`, the frozen M0 CatBoost `resolved_hash`,
    the fixed-payload `core_replay_id` / `account_simulation_id`,
    `feature_block_registry_hash` and the `B0_CORE` bundle id at their
    `PRE_R6_1_FIX_BASELINE.md` values; the OOS-assignment, fold-feature,
    ladder / controlled-study, stratified-report and the S02 / S07 / S09 /
    S14 / S15 stage-result identities re-mint (synthetic only; nothing
    persisted outside tmp roots). DEV-R6.1-FIX-7.
111. **Stratified reports re-verify their executed-trade evidence; the net-R
    accounting is self-consistent by construction** — the stratification
    service exact-loads every child's persisted executed-trade table,
    refuses a table of another core replay or a caller frame whose projection
    bytes differ from the artifact, consumes the LOADED frame for every
    class and binds the artifact's projection-bytes hash
    (`CohortDescriptiveBody.executed_trade_table_artifact_sha256`) beside the
    normalized-frame hash; `RegimeNetRAccounting._coherent` recomputes mass,
    total, shares, top share, signed fractions, zero-denominator reasons and
    the works-only claim from `net_r_by_regime` (1e-9), `CohortDescriptiveBody`
    ties the regime strata's trade counts to the accounting, and the claim is
    FALSE when the assigned side refutes it, NULL only when unassigned trades
    prevent a supported claim, with `assigned_regime_count` persisted so a
    single-regime vacuous TRUE is visible (reviews RA-01, RA-02, RA-03;
    DEV-R6.1-FIX-13 / -14).
112. **Assignment rows are linked and arithmetically self-consistent; the
    candidate as-of provenance is strict and stage-bound** —
    `validate_assignment_rows` requires the linkage key on every row, lawful
    non-null partition / fit id / fold on invalid rows, and
    `assigned_distance == distances[local] == min` with
    `assignment_margin == d2 − d1 ≥ 0` (1e-9) on valid rows; the fit loader
    checks the sidecar's protocol column; `candidate_as_of_source_hash`
    refuses an unparseable non-null anchor and hashes a null deterministically;
    `RegimeOosAssignmentPayload.candidate_as_of_stage` is required on both
    grains (equal to the panel context's stage on the panel grain) (reviews
    RA-06, RA-07; DEV-R6.1-FIX-15 / -16).
113. **Helper label identities are full-column and stamped; the enum copy
    guard knows field shapes** — the ladder / CatBoost bundle / logistic
    helpers default to `label_artifact_content_id(None, …)` and the ladder
    run carries `label_identity_source`; `ControlledFeatureStudyPayload.label_identity_source`
    is required; `FrozenContract.model_copy` guards scalar and
    sequence-of-enum fields and passes every other annotation through
    (reviews RA-05, RA-04; DEV-R6.1-FIX-17 / -18).
114. **One canonical costed-evaluation frame; nine typed sidecar reasons;
    no S14 recovery from prior records; the publication block resets on a
    halt; reproduction re-checks the core-table hash; immutable reload
    reasons** — every costed evaluation is computed from the persisted,
    exact-loaded executed-trade projection (fresh, regime-reuse and plain
    reuse alike) and published inside per-child containment
    (`FailureReason.INVARIANT` on a store refusal); `store_entry_missing`
    means the entry directory does not exist and a manifest-less existing
    entry is the new typed reason `manifest_missing_for_existing_entry`
    (`has_envelope` raises on it; `probe_executed_trade_table` maps only
    the former to absent); `load_verified_envelope` / `load_sidecar_bytes`
    raise the typed reason at the detection point (legacy substrings kept);
    the S14 prior-reports recovery is deleted and S15 reloads every
    executed-trade table and stratified report the S14 record names;
    `_reset_publication_block` on every halt / cancel that includes S15 and
    `activate_pipeline_result` re-derives the gates from the latest attempt;
    `_reuse_child_with_regime_tables` requires projection bytes AND
    `source_core_table_hash` to reproduce, with replay counts asserted (0 for
    non-stratified reuse, 1 verified reproduction per reused child under
    stratified reporting); `_record_schema_version_for` is the one identity
    helper (a bare core id raises `PipelineWiringError`);
    `PipelineResultPayload.reload_failure_reasons` carries the S15 reload
    reasons immutably (reviews B-01…B-10; DEV-R6.1-FIX-1 / -2 / -6 amended;
    DEV-R6.1-FIX-19 … -22).

115. **Semantic store namespace (F-11, HARDENING-BACKEND)** — `search/store_namespace.py`:
    the immutable, re-verified `STORE_NAMESPACE.json` envelope (`namespace_class`
    research|test, a stable 32-hex `store_instance_id` never derived from a path,
    `authority_genesis_id`); `initialize_store_namespace` is the one-time explicit
    migration (class stated, never inferred; idempotent on identical; divergent
    refused; the genesis head written first); every owner decision, supersession
    record, authorization bundle, verification and seed-production authorization
    binds `store_namespace_id`; an unmarked store carries no owner authority;
    `path_looks_like_research_store` / `research_namespace_root` are defense in
    depth only. DEV-HB-3 / -4 / -5.
116. **Immutable supersession record chain + head witnesses (F-12)** —
    `search/supersession_chain.py` (store `owner_decision_supersessions`): one
    content-addressed record per replacement; the mandatory head commits to the
    whole chain from the genesis anchor; four-step atomic publication under the
    lock; orphan records have no authority; identical replay reuses, divergent
    replay refuses; `SupersessionHeadWitness` on every real charter bundle,
    verification and seed-production authorization; `assert_head_witness_current`
    refuses missing / shorter / different heads; the record is published BEFORE
    the replacement decision; legacy JSONL chains are refused, not migrated.
    DEV-HB-6 / -7 / -8.
117. **Liveness-aware owner-decision lock (F-13)** — `search/owner_decision_lock.py`:
    pid + process-start token + random token + host + heartbeat; reclaim only
    when the heartbeat timed out AND the holder is demonstrably dead (Win32
    `GetProcessTimes` / `/proc` starttime; PID reuse detected); live, other-host,
    malformed or unassessable holders are never reclaimed (typed timeouts);
    token re-verified before the head moves; release unlinks only its own token.
    DEV-HB-9.
118. **Authority seams** — `OwnerAuthorizationBundle` / `VerificationAuthorizationRef`
    carry `store_namespace_id` + `supersession_head_witness`;
    `assert_authorization_bound_to_store`; `validate_owner_authorization(store_root=)`,
    `validate_verification_run(..., store_root)` (a `test` namespace required),
    `save_charter` (semantic synthetic rule + bound real bundles), the real
    executors and S00 real branches verify namespace + current witness before any
    path; `synthetic_owner_decision_fixture` marks unmarked test roots explicitly.
119. **Capacity (F-17)** — the streaming event-detail writer (iterable of walk
    pairs consumed once; no whole-artifact index; unconditional disk-backed DuckDB
    uniqueness check; bounded row groups), the generator-fed bridge, the external
    DuckDB event-regime summary aggregation (intermediate Parquet partitions,
    explicit memory limit, attempt-local temp dir, exact counts, canonical ORDER
    BY, row-group-aligned writer, budgets before publication, temp cleanup) and
    the `HARDENING_CAPACITY_POLICY_V1` benchmark harness with native RSS;
    every §4.4 gate passed (`CAPACITY_BENCHMARKS.md`). DEV-HB-10 / -11 / -12.
120. **Warning policy (F-18)** — `filterwarnings = ["error", <one exact
    sklearn/SciPy rule>]`; `dataset.concat_schema_aligned` + the test-side typed
    append close the six project-owned warnings with frozen bytes unchanged;
    `WARNING_BASELINE.json` records the rule with package version, reason,
    owner and expiry. DEV-HB-15.
121. **Sequential-execution truth (F-20)** — `SUPPORTED_CHILD_WORKERS = 1`,
    `EXECUTION_MODE_V1 = sequential_children_v1`, typed
    `UnsupportedWorkerParallelismError`; `WorkerPolicy.max_workers == 1` (the R5
    `le=4` ceiling removed); `ExecutionAttemptIdentity.effective_workers` /
    `execution_mode` persisted on every attempt receipt; the job shim refuses
    `--max-workers != 1` before job creation; the UI slider is the UI plan's.
    DEV-HB-13 / -14.
122. **Logical trading-day calendar and the rebuilt shortlist (F-22)** —
    `search/trading_calendar.py` (`cme_globex_18et_weekday_v1`; `[td−1 18:00 ET,
    td 18:00 ET)` over physical partitions `(td−1, td)`; registered full closure
    2026-01-01 only — Good Friday is a trading day by evidence;
    `VerificationTradingDayRef`), `search/verification_window.py` +
    `scripts/ifvg_verification_window_shortlist.py` (already-authorized evidence
    only; the plan's lexicographic order; `owner_selection = NOT PERFORMED`;
    `register_program_allowlist` never called). Finding: the June proposal is
    INELIGIBLE (no exact verifier target); the R1 February store-day candidate
    is provisional/ineligible as stated (a Sunday partition). DEV-HB-16 / -17 / -19.
123. **Separate seed-production authorization and run (F-16 / F-21)** —
    `search/seed_production.py` + `scripts/ifvg_seed_production.py`:
    `SeedProductionReplayPolicy` (fail-before-path; store-day chain; June 11 and
    the sealed range refused), authorization / run contracts binding the namespace
    + head witness, profile + section hash, store-day chain and its logical
    subset, `snapshot_through_day`, `first_intended_verification_day`, the
    source-inventory hash, QL/SC identities, seed schema, permitted / prohibited
    outputs and owner fields; verification before any path;
    `run_seed_production_chain` (seed snapshot + access audit + run receipt
    only) proven synthetically; unsigned packets fail validation; seed timestamps
    canonicalized to stdlib UTC (a latent R1 sandbox defect recorded).
    DEV-HB-16 / -18.
124. **Bounded verification code (Phase 4)** — `search/bounded_verification.py`:
    the typed §6.1 preflight (16 refusal reasons; output root locked to
    `search_test/v1`; a coherent `test` namespace; current witness; real
    authorization; 1–5 consecutive LOGICAL days inside the window; physical
    mapping; canonical program allowlist; profile-bound continuous seed),
    `R1BaselineGateReport` (both attempts' gates + audit-mode digests + eight
    proofs; `passed` derived) and `BoundedReleaseControlFlowReport` (eight
    components typed from persisted evidence; `not_research_evidence`);
    `scripts/ifvg_bounded_verification.py` refuses `fail_before_path` without
    the owner's persisted authorization (proven on the real store). DEV-HB-20 /
    -21 / -22.
125. **Release boundary** — Phases 2, 3 (code/contracts) and 4 (code) in one
    `HARDENING-BACKEND` commit; no owner action, no real seed replay, no real
    ≤5-day run; the real verification store not initialized; `data/` unchanged;
    acceptance `transitively_blocked_by_R1`. DEV-HB-1 / -2.

## HARDENING-BACKEND-FIX (the compact backend correction; plan §§4–10; D-051)

126. **Token-safe stale-lock reclamation (HB-FIX-01)** — `search/file_mutex.py` (new; a
    private standard-library cross-process mutex: `msvcrt.locking` byte range on Windows,
    `fcntl.flock` elsewhere; the mutex file is never unlinked, carries no body and no
    authority) + `search/owner_decision_lock.py`: `_observe_stale` → `_reclaim` re-reads and
    re-evaluates age / host / pid / process-start token under the reclaim mutex and unlinks
    ONLY the byte-identical dead holder it observed; a persistent read failure is the typed
    `lock_read_failed` (never absence); `release()` raises `lock_release_failed` when it
    cannot read or verify its own lock (absent / malformed / vanished / persistent I/O);
    `_try_create` removes or quarantines the partial exclusive file before
    `lock_create_failed`; token-only release preserved.
127. **Atomic, recoverable namespace initialization (HB-FIX-02)** —
    `search/store_namespace.py`: a one-time init mutex
    (`owner_decisions/STORE_NAMESPACE.init.mutex`); five-state classification — both absent →
    the deterministic pair through temporary files + verified load; both coherent → idempotent
    or `store_namespace_divergent`; one present → exact recovery by the identical request
    (class + explicit instance id, no supersession records) else
    `incomplete_store_namespace_initialization`; both inconsistent → fail closed;
    `store_namespace_initialization_busy` while another initializer holds the mutex;
    `_verified_genesis_pair` proves `namespace.store_namespace_id == head.store_namespace_id`,
    `authority_genesis_id == head.head_sha256` and requested class == persisted class; the
    supersession store name has one source of truth (`supersession_chain.py` imports it).
128. **Public source-kind boundary (HB-FIX-03)** — `search/trading_calendar.py`:
    `SourceKind = Literal["mbp1", "trades", "legacy_verified_replay_source"]`; the private
    `_PHYSICAL_STEM_RESOLVER` is the only place the historical stem is named;
    `PhysicalSourceDescriptor` (internal: filename, era, sha256, partition key;
    `replay_bytes_only`); `assert_public_source_kind` at `trading_day_ref_from_inventory`, the
    inventory builder and `seed_production.seed_chain_source_inventory_hash`;
    `test_public_source_kind_surface.py` permits the literal exactly once in the IFVG lane.
    Re-mint: inventories / windows / seed authorizations whose payload serialized the stem.
129. **Exact regime provenance, native validation, exact OOS identity (HB-FIX-04 … -07)** —
    `ml/regime_contracts.py` (`validate_native_values`, `FIT_ASSIGNMENT_NATIVE_SPEC`,
    `SENTINEL_STRINGS`, `RegimeAssignmentEvidenceError`; `validate_assignment_rows` gains the
    provenance rules: fold + partition together, a fit id only with them, `no_oos_assignment`
    never names a fit / fold; `_fit_assignment_frame_for_schema` validates natively BEFORE
    `to_numeric` / `astype`), `ml/regime_oos_assignment.py` (three-way semantics with
    `_retained_invalid_row`; the candidate grain indexes ALL test rows; the panel PIT rule
    retains the invalid fit row in both modes; `OOS_ASSIGNMENT_NATIVE_SPEC` before
    canonicalization; the payload refuses any `assignment_schema_hash` but the registered one;
    `_assert_candidate_sets_equal`; `verify_assignment_table_bytes` decodes the Arrow bytes and
    proves schema / count / uniqueness / row invariants at save AND load),
    `ml/regime_fold_features.py` (`fold_feature_native_spec`; spine rules;
    `assert_fold_feature_spine_bound` at build and load — rows ⊆ the `FoldFitRef` binding of
    their fold; an invalid fit row without a typed reason refused),
    `ml/regime_assignment_sources.py` (`regime_for_trades` keeps the known fit / fold; only the
    trade-facing outputs are null). Re-mint: OOS / fold artifacts whose invalid rows previously
    collapsed; the KMeans fit sidecar bytes unchanged (validation added, coercion rules not).
130. **Fail-closed manifests; exact label and executed-trade evidence (HB-FIX-08 / -09)** —
    `search/store.py`: `_validate_manifest_entries` shared by every probe / load / reuse path
    (mapping entries with a bare relative file name, a lowercase 64-hex digest, a non-negative
    byte count; no duplicate or normalized-collision paths; no reserved names; the envelope
    entry exactly once), `_sidecar_path` (resolve + compare before opening;
    `sidecar_path_escape`), size + hash checks, `invalid_store_locator` distinguished from
    absence; `ml/comparison_rows.py` (`assert_unique_label_candidates` before every label hash;
    `assert_exact_label_artifact`); optional `label_policy_id` on
    `controlled_feature_study` / `regime_controlled_study` / `regime_cohort_model`, always
    passed by the pipeline's persisting seams (`regime_supervised_stage` S09c and the
    controlled MBP-1 study in `pipeline.py`); `ml/regime_stratification_service.py`
    (`StratificationEvidenceError`; the persisting service requires every child's
    `executed_trade_table_id` — `executed_trade_table_required`). Carry-forward: direct API
    callers of the three runners that pass a `label_artifact_id` without `label_policy_id` keep
    the R6.1-FIX trust semantics.
131. **Central seed canonicalization (HB-FIX-10)** — `search/child_replay.py`:
    `canonicalize_seed_datetimes` (aware → a fresh stdlib-UTC object; naive unchanged;
    dataclasses / pydantic models / named tuples / tuples / lists / dicts rebuilt),
    `canonical_seed_hash`; `save_seed_snapshot` canonicalizes unconditionally (the ONE seam);
    `seed_production._canonical_utc` delegates and the runner hashes and persists the canonical
    seed (the former "hash changed" refusal withdrawn — the canonical hash IS the identity).
    Re-mint: seeds created from non-UTC representations; UTC-represented goldens unchanged.
132. **Bounded event-detail partition (HB-FIX-11)** — `propsim/event_detail.py`:
    `EventDetailBudget.max_rows_per_partition` (omitted from serialization when absent → V1
    identities byte-identical), `EVENT_DETAIL_BUDGET_V2` (50,000 = the benchmark's measured
    row-group size; no registered ceiling lowered), `EVENT_DETAIL_PARTITION_BOUND_POLICY_V2`;
    the writer flushes at the bound even inside one path; partitions keyed
    `(path_block_id, partition_ordinal)` (`event_detail_block_000000_000.parquet`) with first /
    last event keys, rows, bytes, sha256, schema hash; cleanup of every written partition on
    refusal; the reader proves the bound, the boundary keys and the total order across
    partitions; the V1 budget loadable, refused by the writer; `search_bridge` defaults to V2;
    `scripts/hardening_capacity_benchmark.py` gained the `normal` / `skewed` / `dense` shapes
    and the resident-batch gate (PASS: maximum resident writer batch 50,000 rows at 1M rows;
    1M-row peak RSS +0.322 GiB normal / +0.322 skewed / +0.333 dense). Re-mint: simulations
    under the default (V2) budget; identities minted under an explicit V1 budget preserved.
133. **Complete authority-chain proof (HB-FIX-12)** — `search/owner_decisions.py`:
    `verify_complete_owner_authority_chain` + `CompleteOwnerAuthorityChain`;
    `load_supersession_chain` = the complete proof (every record verified from the genesis
    anchor, every superseded and replacement decision verified-loaded, the replacement pointing
    to its predecessor, one protocol, no weaker provenance, no earlier approval, no earlier
    recording, no decision superseded twice, the signed witness equal to the verified current
    head); run by `authorization.assert_authorization_bound_to_store` (charter freeze / load,
    pipeline launch, activation gate, executors, verification run, MBP-1 diagnostic),
    `seed_production._witness_current`, the bounded-verification preflight witness step and
    the regime chain loader; reasons `supersession_decision_unverifiable`,
    `supersession_transition_unlawful`, `supersession_chain_divergent` (registered in the
    namespace module). Consequence: a bundle re-signed over a chain naming non-existent
    decisions is refused (previously accepted on the structure-only witness).
134. **Focused review round and release boundary** — exactly two read-only reviewers in one
    pass (plan §3.2 / §13; `FOCUSED_REVIEW.md`): 0 blockers; RA-01 (a held lock found carrying
    another writer's token now releases as the typed `lock_release_failed`), RA-02 (a
    non-contention mutex failure is `FileMutexError` → `lock_mutex_failed` /
    `store_namespace_initialization_failed`, never a live holder or a busy initializer), RA-03
    (canonicalization covers dict keys, sets, pydantic extras, `init=False` dataclass fields
    and pandas Timestamps; `numpy.datetime64` / `NaT` / sub-microsecond carriers refused),
    RA-04 (a read failure right after exclusive creation discards the fresh lock), RB-01
    (run-level `label_identity_proof`; `assert_persistable_label_proof` at every persisting
    study save — an id without the registered policy is in-memory only), RB-02 (a fit-bearing
    fold refuses fit-less rows at build and load), RB-03 (event-detail cleanup covers a partial
    file and the manifest stage), RB-04 (the benchmark observes the writer's resident batch at
    its Parquet seam; the reader's bound is the identity-bound budget's), RB-05 (the OOS /
    panel / trade-projection seams validate the columns they consume natively before
    coercion) — all FIXED once; five carry-forwards recorded, not gates. Ten corrections in one
    `HARDENING-BACKEND-FIX` commit parented by `e56f937`; no owner action, no seed production,
    no real verification, no full pipeline, `data/` unchanged; acceptance
    `transitively_blocked_by_R1`; `backend_dev_complete_for_ui = true`,
    `ui_implementation_may_begin = true`. D-051.

135. **HARDENING-BACKEND-FIX.1 — the four remaining code issues, the capacity gate closed** —
    one focused commit `7491769` (`74917692a96b725d7e1f6f2c9ec8160e3a18afec`, parent `a5eee1a`;
    13 files, +1,092 / −101; not pushed, not merged). The independent review document was absent
    from the host; each finding was confirmed against the source before editing. (1)
    `search/owner_decision_lock.py`: `LockBody.parse` validates strictly (exact schema version;
    native positive in-range pid, never bool; 32-lowercase-hex token; string-or-null host /
    start token; aware ISO-8601 timestamps; exactly seven fields; no coercion;
    RecursionError-safe) and a malformed body stays non-reclaimable; after O_EXCL creation
    (binary mode) acquisition is proven by the exact persisted readback (missing / foreign /
    altered → `lock_lost`, malformed → `lock_body_malformed`, unreadable → `lock_read_failed`;
    nothing acquired, nothing removed); a partial exclusive file is cleaned up only when the
    readback is exactly a prefix of the bytes this writer wrote — the RA-04 discard of an
    unreadable fresh lock is withdrawn (left in place, fail closed). (2)
    `search/store_namespace.py` + CLI: the real initializer never mints an instance id — the
    first initialization of an unmarked store requires an explicit 32-hex `store_instance_id`
    (`store_instance_id_required`, before any publication; malformed ids refused before the
    mutex and as the typed CLI refusal `store_instance_id_malformed`); class-only replay of a
    marked store, the identical explicit request, explicit-id crash recovery and the serialized
    fail-closed concurrency are unchanged; `initialize_test_namespace` is the clearly separate
    disposable test-only helper minting `uuid4().hex` BEFORE the real initializer. (3)
    `ml/regime_oos_assignment.py`: `assert_native_candidate_ids` before `astype(str)` — native
    non-blank `str` only; canonical output for valid ids unchanged. (4) The unchanged harness run
    alone on the idle host: `CAPACITY_BENCHMARKS.json` `passed = true`, B1 / B2 Overall PASS (min
    available RAM 13.119 / 12.161 GiB), every limit unchanged, output hashes identical to the
    prior runs. Gates: targeted 135 / 297 / 135; full suite as-is 2,287 passed; credentials
    cleared 2,287 passed; exact-reuse / golden 93 passed; warnings as errors; ruff clean; diff-check
    clean. One read-only reviewer, one pass: PASS, two minor findings fixed (RecursionError-safe
    parse; typed CLI refusal). `backend_dev_complete_for_ui = true`,
    `ui_implementation_may_begin = true`; acceptance `transitively_blocked_by_R1`; owner
    verification and R1 acceptance remain separate later actions. D-052.
    **R3 (2026-09-04).** The owner's independent complete lock-safety review of `7491769` found
    one Windows failure path: `_win32_process_times` treated a FAILED `GetExitCodeProcess` query
    as proof of death, so a live holder whose exit-code query failed could be reclaimed. Exact
    correction (a failed query is `unknown`, never reclaimable; only a SUCCESSFUL query with an
    exit code other than STILL_ACTIVE is `dead`) in the follow-up commit `ffcf39b`
    (`ffcf39bdcfb7b4d7ca8995886bb23e682f11e30b`, parent `7491769`; release head), proven red-first
    by `test_win32_exit_code_query_failure_is_unknown` and
    `test_stale_lock_is_not_reclaimed_when_win32_exit_query_fails`; D-052 amended. Gates over the
    corrected tree: fast set 137, consumer regression 299, capacity benchmark PASS (9.644 / 10.423
    GiB available), ruff and diff-check clean; the full suites (2,287 x2) and the golden suite (93)
    were not rerun — owner decision: the correction is confined to the Win32 liveness reader.
    Patch (two-commit series) and bundle regenerated with SHA-256. `backend_dev_complete_for_ui =
    true`, `ui_implementation_may_begin = true`; acceptance `transitively_blocked_by_R1`.

## UI-1 (UI/UX redesign Phase 1)

136. **UI-1 — semantic purpose, namespace and authorization truth; satisfiability; honest
    launch and evidence** — Phase 1 of the owner-approved
    `UI-UX-REDESIGN-PLAN/IMPLEMENTATION_PLAN.md` (revision 2) over the HARDENING-BACKEND-FIX.1
    release head `ffcf39b`. Engineering decisions inside the plan's scope: (1) a synthetic
    fixture is the EVIDENCE CLASS of Implementation Verification (`EvidenceClass`), not a fourth
    purpose — the R4-era synthetic search in the verification namespace stays available under
    the VERIFICATION-ONLY badge, and real verification refuses the marker; (2) the identity-
    bearing satisfiability rules live in `validate_charter` (FSM ≥ 2 profiles, single ≤ 2,
    Universal ≥ 2 firms) and the real-charter prop-objective rule applies to non-synthetic
    charters only (synthetic fixtures declare firms through the injected wiring, as every
    existing fixture does); the presentation report applies every rule to every UI draft;
    (3) `Evaluate one configuration` is a fifth research question (owner Q4) compatible with
    the single-configuration family; (4) the purpose annotation persists on drafts (additive
    `purpose_annotation`) and as an additive catalog event kind `purpose` on frozen charters;
    (5) the real verification charter carries an `OwnerAuthorizationBundle` whose 21/R-5
    evidence ref derives from the persisted, verified `VerificationAuthorizationRef` (the
    charter shape is unchanged); (6) readiness typing maps the backend reasons —
    `supersession_head_witness_mismatch` → `stale_head`, the chain-structure refusals →
    `wrong_head`, the decision-level refusals → `superseded`, identity / class mismatch →
    `wrong_namespace`, missing envelope → `store_unmarked`; (7) the honest launch waits up to
    `LAUNCH_STATE_WAIT_SECONDS` (10 s; tests shorten it) for the worker's state file and
    otherwise renders `launch_not_started` with the job-log location; (8) the publication gates
    record `gates_store_namespace_id` (None for an unmarked store) and `gates_state_sha256`
    (stages / attempts / children), and `activate_pipeline_result` gains
    `expected_store_namespace_id`; (9) listings locate each run's store by exact charter id
    across the deployed stores (research first, then test) and fall back to the default read
    root only for display; (10) `direction_colorscale` treats an unregistered metric as
    "direction unregistered" (neutral scale, labelled), never as maximize; (11) the reconciliation
    report's `passed` becomes `True` / `False` / `None` (JSON null when nothing was evaluated) with
    `evaluated`, `gate_evaluations` and `unevaluated_reports` — additive keys; (12) drafts are still
    written on `Start new draft` / Start cards (session-only drafts are UI-2, owner Q2); the
    Verification Center in UI-1 is the readiness surface + verification-draft entry (the
    fixture → seed → final-authorization flow is UI-2); the phase radio stays until UI-6. No
    owner action, no seed production, no real verification run; acceptance
    `transitively_blocked_by_R1`. D-053.

## UI-2 (UI/UX redesign Phase 2)

137. **UI-2 — complete Verification Center, goal-derived flows, safe drafts and explicit
    reviews** — Phase 2 of the owner-approved `UI-UX-REDESIGN-PLAN/IMPLEMENTATION_PLAN.md`
    (revision 2) over the UI-1 release head `f1827e8`. Engineering decisions inside the plan's
    scope: (1) seed artifacts are never catalogued (the seed run writes only its permitted
    outputs), so the center picks external results up through `--receipt-out` receipts under the
    mutable `data/ifvg_verification_center` root and verifies every recorded id by exact load;
    (2) the real verification bundle derives from the owner's SIGNED `VerificationAuthorizationRef`
    (its content hash is the 21/R-5 decision artifact) — never from a run envelope id, which would
    make the charter depend on the run that names its pipeline spec; (3) the center has no spawn
    seam: the seed job and the bounded run are exact external commands shown only when their typed
    preconditions hold (a verified authorization; a passed preflight); (4) the seed-production
    authorization is `verified_envelope` (namespace, profile, window — explicitly not the full
    check) when the accepted inventory is unavailable to the workspace; (5) the center's Review /
    run step freezes the exact-baseline charter, persists the pipeline spec through the pipeline
    tab's own `_assemble_pipeline_spec` (the same identity the operator surface would mint) and
    registers the `VerificationRunEnvelope` through the backend's fail-before-path validation;
    (6) the Validation step stays in every research flow (dates, seed and the authorization
    checklist have no other home) and a step the flow skips contributes nothing to satisfiability
    or charter assembly; (7) the pipeline family refines by purpose (Development Research →
    feature / model evidence, Full Authorized Development → the advanced end-to-end study);
    (8) the draft schema bumps to 2 with additive fields only (schema-1 files load unchanged);
    `discard_draft` is retired (raises; deletes nothing) rather than removed; (9) the ONE delete
    control (`Delete draft permanently`, History's archived view) is a registered exception of the
    forbidden-control scan, proven typed by source; (10) the code identities of the seed packet are
    computed over the installed code (`quant_lab_replay_source_identity`,
    `strategy_core_source_identity`; read-only) and tests pin them; (11) `Saving` is not observable
    in Streamlit's synchronous model — the chip states are `Not saved yet` / `Saved` / `Autosaved`
    (drafts) and `Unsaved` / `Saved` (reviews); (12) the seed CLI gains `register-authorization`
    (persist a COMPLETED packet; placeholders are the typed refusal `packet_not_signed`) and
    `--receipt-out`. No owner action, no seed production, no real verification run, no browser
    evidence (UI-6); acceptance `transitively_blocked_by_R1`. D-054.

## UI-3 (UI/UX redesign Phase 3)

138. **UI-3 — metric and help system; research results; MBP-1 / regime presentation** —
    Phase 3 of the owner-approved `UI-UX-REDESIGN-PLAN/IMPLEMENTATION_PLAN.md` (revision 2) over
    the UI-2 release head `eb85d09`. Engineering decisions inside the plan's scope: (1) every
    metric reference is a REGISTERED code source (the selected resolved gate, the persisted
    reference Brier, the 0 skill boundary, the 0.5 chance line as a direction only, the
    calibration targets as a distance only, the stamped adequacy minimums, the report limits, the
    measured access counters, intervals crossing zero) — no threshold is invented; the
    policy-enforced `protected_*` zeros stay INFORMATIONAL (the UI-1 ruling) while the measured
    denied-attempt count is 0 PASS / nonzero FAIL; (2) the roll-up rule is fixed and
    order-sensitive (FAIL → BLOCKED → INCONCLUSIVE → WARNING → PASS → INFORMATIONAL → UNAVAILABLE)
    and a PASS under a proposed threshold rolls up as WARNING; (3) the help scan covers EVERY UI
    script including the verifier and lab tabs, admits only registered, justified, LIVE
    exemptions (sixteen navigation controls) and fails on a dead help entry or a stale exemption;
    the verifier tab receives registry help now (UI-4 regroups the controls and may reword the
    entries); (4) the detail levels keep the persisted `summary` / `analyst` / `audit` values and
    relabel them Summary / Research details / Technical identity & audit; `disclosure_level`
    delegates; (5) `context_reporting.py` and the M0–M3 computation are untouched — the adapters
    expose additive reference / calibration / fold / access fields and older stored runs render
    UNAVAILABLE where a key is absent; (6) the Context Research reconciliation banner keys on the
    Data-integrity roll-up, so a legacy `passed: True` without evaluated gates is never green;
    (7) the Results readings use the charter's resolved gates with the proposed caveat and the
    prop vector's worst firm (D-#18); (8) the ladder frame keeps AUC numeric with a separate
    `AUC reason` column; (9) the MBP-1 and regime panels instantiate their manual inputs into a
    lower Advanced diagnostics container so the summary reads their values, never render a
    second empty state for a failure the details report, and keep the regime panel's read-only
    rule (no button / form / toggle / selectbox / session-state write); (10) the wizard's gate
    inputs get static accessible names (`Resolved value for <gate>`) under the collapsed label;
    (11) `trade_frequency` / `setup_occupancy` are registered as schema-reserved, not persisted.
    No owner action, no seed production, no real verification run, no browser evidence (UI-6);
    acceptance `transitively_blocked_by_R1`. D-055.
