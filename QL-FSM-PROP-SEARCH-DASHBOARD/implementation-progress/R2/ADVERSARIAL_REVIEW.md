# R2 — Adversarial Review (two independent read-only reviewers)

Reviewer A: contracts/identity/immutability lens (CS §0–§4/§12, DT §4–§5,
TM §3.2/§3.5/§3.8/§3.9). Reviewer B: access-safety / M0–M3 compatibility /
pre-existing-surface behavior / R2 gate completeness. Both reviewers were
read-only and produced findings + verified-clean lists; no finding was
dismissed because tests were green. Resolutions in
`ADVERSARIAL_REVIEW_RESOLUTION.md`.

---

## Reviewer A report (verbatim)

# Adversarial Review — R2 of `ifvg_prop_robust_config_search_v1` (Quant-Lab)

Scope: R2 additions/modifications under `C:\Users\gonza\Documents\Claude-Quant-Lab\src\alpha_lab\agents\data_infra\ifvg\` (search/, study/, fsm_audit_preparation.py, dataset.py), the seven named test files, and `QL-FSM-PROP-SEARCH-DASHBOARD\implementation-progress\R2\DEVIATIONS.md`, audited against CONTRACTS_AND_SCHEMAS.md (CS), DELTA_TAXONOMY.md (DT), PHASED_DELIVERY.md, TEST_MATRIX.md (TM). All 63 in-scope tests pass when run read-only (`38 passed` + `25 passed`, no cache writes). Green tests were not treated as evidence of compliance.

---

## BLOCKER

### F1 — Failed/absent neutrality does not block child publication; publication order is inverted vs the §3.3 worker flow
- **Contract**: CS §3.3 "Any failed child-neutrality report blocks the audit artifact **and child publication**"; CS §1.4 "post-replay invariant/neutrality failures block publication"; CS §3.3 worker flow order: "…build requested companions → assert zero forbidden access → **publish atomically** → reload and verify".
- **Evidence**:
  - `src\alpha_lab\agents\data_infra\ifvg\search\child_replay.py:666-670` — `run_baseline_verification_slice` publishes `replay_input_bundles` and `core_replays` envelopes **before** any neutrality consultation; the only neutrality effects are the gate boolean at :697 and the companion builder's refusal (which runs *after* publication, :673-682).
  - `child_replay.py:394-460` — `run_child_replay` computes the `ChildAuditNeutralityReport` and **returns normally when `passed=False`**; no raise.
  - `src\alpha_lab\agents\data_infra\ifvg\search\orchestrator.py:441-471` — `run_search` saves the child's `core_replays` envelope and membership after `child_runner` success with **no neutrality consultation at all**; the runner seam contract (docstring :360-368) requires only `.tables` and `.gross_trade_stream_hash`, so the orchestrator is structurally unable to enforce §3.3 unless a future runner happens to raise.
  - `orchestrator.py:435-440` — reuse keys solely on `has_envelope(store_root, "core_replays", id)` and labels it "verified reuse: an immutable replay with this exact identity already exists". A failed-neutrality child that got its envelope published is therefore treated forever after as a verified, complete child (and, being immutable, can never be re-attempted without manual store surgery).
  - The only genuine refusals live in `fsm_audit_preparation.py:391-416` (`build_child_fsm_audit`) — they protect the **audit artifact** only, and fire after the child identity is already in the store.
- **Reproduction**: (1) Run `run_baseline_verification_slice` with a `replay_runner` whose two drives differ (or monkeypatch `build_neutrality_report` to `passed=False`) and `companion_builders=None` → function returns successfully; `core_replays/<id>/` and `replay_input_bundles/<id>/` exist in the store while the gate report shows `neutrality_passed: false`. (2) Call `run_search` with a `child_runner` stub (as the tests do) → child "completed", envelope + membership published, gates and frontier computed — neutrality never consulted. (3) Re-run either path → `reused=True` for the failed child.
- **Resolution**: gate publication on neutrality: in `run_baseline_verification_slice`, evaluate `result.neutrality.passed` (fail-closed on None when `dual_drive`) **before** step 7, or reorder to companions-then-publish per §3.3; make `run_child_replay` raise on `passed=False` (typed `PermissionError`); in `run_search`, either require the runner result to carry a passing neutrality report or document that the R5 runner-entry MUST raise, and add a TM row proving a failed-neutrality child ends `failed`, unpublished, and non-reusable.

---

## MAJOR

### F2 — The stale-lock break can break a LIVE run's lock; the break itself is racy; the holder's release can delete the breaker's lock
- **Contract**: TM §3.5 (resume-after-kill, lock contention — a live run must refuse a second runner); IMPLEMENTATION_PLAN §6 process controls.
- **Evidence**: `orchestrator.py:235-256` (`_try_break_stale_search_lock`: stat-age check then `os.unlink`), :258-287 (`_search_lock`), :319-321 (heartbeat = `os.utime(lock)` only inside `_checkpoint`), :348-359 (`stale_lock_seconds: float = 3600.0`). Checkpoints happen only at child boundaries; **no heartbeat occurs during `child_runner` execution or during `prewarm`** (:398-401). TM §4's own operator estimate is ≈35 min per 138-day child *per worker share*, and measured cold recompute is 72-101 s/day (≈2.8-3.9 h per 138-day child); a whole-grid prewarm is hours. Any of these exceeds 3600 s. Additionally: two waiters can both pass the age check, waiter B can `unlink` the lock waiter A just re-created (TOCTOU between :251 unlink and :277 re-acquire), and the original holder's `finally: os.unlink(lock)` (:286-287) deletes whatever lock file currently exists — i.e. the breaker's — enabling a third concurrent runner. The passing test (`tests\agents\ifvg_search\test_orchestrator.py:403-432`) only exercises a fresh lock at 3600 s and a stale break at 0 s; it cannot see this.
- **Reproduction**: start `run_search` with a `child_runner` that sleeps 3700 s; from a second process call `run_search` for the same `search_id` with defaults → the second process breaks the live lock and both write `search_state.json` and run children concurrently.
- **Resolution**: heartbeat during long phases (touch the lock from `progress_fn`/prewarm callbacks or a daemon toucher thread); raise the default threshold well above worst-case child duration or scale it from the charter's date count; replace unlink-then-create with an atomic rename-to-quarantine; have the holder verify ownership (stored pid/nonce) before unlinking on release.

### F3 — Post-hoc contrast refusal is not charter-anchored (spoofable)
- **Contract**: DT §5 "Rules enforced in code: only declared contrasts (the `contrast_id` must exist in the **frozen charter** — post-hoc contrasts are refused)".
- **Evidence**: `src\alpha_lab\agents\data_infra\ifvg\study\contrasts.py:113-117` — the check is `contrast.contrast_id not in declared_contrast_ids`, where `declared_contrast_ids` is a plain caller argument; nothing loads it from a frozen charter and nothing verifies `payload.charter_id` corresponds to the charter that supplied the list. `src\alpha_lab\agents\data_infra\ifvg\search\charter.py:188-213` — `SearchCharterPayload` has **no** declared-contrast field (the plan's §3.1 schema also lacks one, so this is a plan gap the implementation inherited rather than closed). The docstring claim "post-hoc contrasts are structurally impossible" (contrasts.py:3-5) is therefore untrue: it is parameter discipline, not structure.
- **Reproduction**: build any `DeclaredContrastEnvelope` after seeing results; call `evaluate_declared_contrast(c, declared_contrast_ids=(c.contrast_id,), ...)` → evaluates.
- **Resolution**: anchor declaration in an immutable artifact — either a `declared_contrast_ids: tuple[str, ...]` charter field (new `search_id` semantics) or a frozen per-charter contrast-registry envelope saved in the store — and have the evaluator load the declared set by `payload.charter_id` itself, refusing mismatches. Record the charter-schema addition as a deviation/plan patch.

### F4 — Cross-profile population deltas are constructible on native ids
- **Contract**: DT §4.3 "Same-profile / same-config comparisons use `native_id_exact`… Cross-profile comparisons use `profile_independent_lineage_exact` **only**"; CS §4.
- **Evidence**: `src\alpha_lab\agents\data_infra\ifvg\study\population_delta.py:176-192` — `build_native_population_delta` accepts any two table mappings and stamps `match_basis="native_id_exact"` with **no same-profile verification**, although every v2 row carries `envelope_section_config_hash`/`evaluation_config_hash` columns that would make the check one line. Because native ids embed the profile hash, feeding two different profiles produces a fabricated "0 common / everything added+removed / jaccard 0" report under a lawful-looking basis label. No caller-level guard exists in R2 and no test attempts the misuse.
- **Reproduction**: `build_native_population_delta("setup", tables_doc_default, tables_retest480_variant)` (the two chains already built in `tests\agents\ifvg_search\test_lineage.py`) → `match_basis="native_id_exact"`, jaccard 0.0, all keys "removed"/"added".
- **Resolution**: inside `build_native_population_delta`, extract the distinct `envelope_section_config_hash` (or profile hash) per side and return `_not_comparable(kind, "different profile hashes — native basis is same-profile only; use the lineage layer")` on mismatch; add the misuse test.

### F5 — Reused children are excluded from gates/frontier; the frontier is run-history-dependent and nothing (frontier, metrics, gate reports) is persisted
- **Contract**: CS §12 (`FrontierResult` … "**persisted** lexicographic tie-break trace"; deterministic frontier over the feasible set); PHASED R2 gate "enumerate→**reuse**→gates→frontier→insights"; CS §8 declares a `frontiers` store.
- **Evidence**: `orchestrator.py:481-483` — `if result is None or outcome.state not in ("completed", "reused"): continue` — reused children never have in-memory tables, so they are silently dropped from gate evaluation and `feasible_metrics`; on a full-reuse resume, `frontier=None` while phases still advance to `frontier_complete`/`search_complete` (:530-533). `build_frontier` has exactly one caller (grep-verified) and its result is returned in-memory only; the `frontiers` store (`store.py:59`) is never written; per-child `StrategyMetrics`/`StrategyGateReport` are never persisted (the checkpoint at `orchestrator.py:323-342` records only reason/explanation strings). Consequence: the same charter over the same store yields different `FrontierResult`s depending on which children happened to replay in this process, and a reused child that would pass gates vanishes from the second study's frontier. `DEVIATIONS.md` (last scoping note) acknowledges the gate-pass scoping but the compensating "results surfaces read published artifacts" cannot exist — no metric/gate/frontier artifact is published for them to read.
- **Reproduction**: run `test_reuse_within_and_across_studies` semantics with a passing runner: study 1 completes with a frontier; study 2 (same children) returns all-reused, `frontier is None`, phase `search_complete`.
- **Resolution**: persist per-child metrics + gate reports keyed by `core_replay_id` at completion (a `search_results` store entry), load them for reused children before the gate pass, and publish the `FrontierResult` (with trace) to the `frontiers` store; until then, the deviation should state explicitly that R2's in-run frontier is a synthetic-lane artifact only.

### F6 — GeneratedProfileCapability: placeholder registry hash in the production path, unreachable `blocked_invariant_failure`, and no value-capability re-enforcement at enumeration
- **Contract**: CS §1.4 (eligibility = runnable baseline + registered values + owner-authorized values + **intact locked invariants** + valid section + deterministic naming; `registry_hash` field); PHASED R2 gate "generated-profile capability enforced in orchestration (P0-D — unratified values blocked before replay)"; TM §3.1 "inert fields… refused before enumeration".
- **Evidence**: `orchestrator.py:148-155` — `enumerate_children(..., registry_hash: str = "0" * 64)`; `run_search` calls it at :386-388 without passing `registry_sha256()` (which exists, `axis_registry.py:624-635`), so **every real run mints capabilities bound to a fake registry hash**. `identities.py:645-715` — no code path returns `blocked_invariant_failure`; no invariant check exists in the capability evaluation. `orchestrator.py:221-227` — `_has_pending_values` checks only `owner_ratification_status == "pending"`; `resolve_axis_overrides` (`axis_registry.py:707-745`) checks registration + axis binding but not `capability_status`/classification; blocked values are refused only by `validate_charter` (`charter.py:276-288`), which `run_search` never re-runs — a hand-built (never-validated) charter envelope with a blocked-capability, `not_required`-ratified value enumerates and replays.
- **Reproduction**: construct `SearchCharterEnvelope.from_payload` directly (skipping `validate_charter`) with a `blocked_reducer_hardcoded` value id in `axes`; call `run_search` → the child's capability is `generated_runnable` and the runner executes. For the registry hash: any test challenger spec — `spec.capability.registry_hash == "0"*64`.
- **Resolution**: thread `registry_sha256()` through `run_search` → `enumerate_children`; call `assert_axes_authorized` per combo inside `enumerate_children` (fail-closed defense in depth); either implement the locked-invariant check (baseline invariants unchanged in the resolved section) or record in DEVIATIONS why `blocked_invariant_failure` is structurally unreachable in v1.

### F7 — Lineage collisions are not persisted anywhere, while the emitted report text claims they are
- **Contract**: CS §4 "collisions ⇒ `not_comparable` + **persisted records**"; DT §4.3 (P1-E) "a persisted `LineageCollisionRecord`, and the `LineageUniquenessReport` **accompanies every replay** used in a cross-profile comparison".
- **Evidence**: `lineage.py` builds `LineageUniquenessReport`/`LineageCollisionRecord` in memory only; grep across `src` shows no save path, no store name, no sidecar for them (SEARCH_STORE_NAMES has none; nothing writes them). Meanwhile `population_delta.py:222-227` emits the reason string "…collisions **(persisted in its LineageUniquenessReport)**" — factually wrong today. R2 ships cross-profile deltas, so the accompanying-report requirement is already in scope.
- **Reproduction**: run `test_collision_is_recorded_never_deduped`; inspect any store/filesystem location — no collision record exists anywhere durable.
- **Resolution**: persist the uniqueness report per replay (e.g., an envelope+sidecar in a `search_results`-class store keyed by `core_replay_id`) before any cross-profile delta is surfaced, or correct the reason text and add a DEVIATIONS entry deferring persistence with its landing release.

---

## MINOR

### F8 — `EvidenceRef` kinds extend beyond CS §12's registered set
CS §12 registers `setup/trade/account-event/child/simulation/axis`. `insights.py:55-61` adds `candidate`, `decision` (reachable — :246-249 uses `kind=delta.entity_kind`) and `artifact`, `report` (never constructed; dead vocabulary). No DEVIATIONS entry. Resolution: either patch the plan vocabulary via a deviation entry (candidate/decision are defensible evidence kinds) and delete the dead `artifact`/`report` members, or map candidate/decision refs onto their parent setup.

### F9 — Max-cap gates pass silently on missing observables
`gates.py:74-89` (`_maximum`: "a cap over nothing holds"). Min gates fail closed (verified), but `max_top_setup_pnl_share` passes when the setup column is absent or `setup_abs == 0` (`strategy_metrics.py:154-163`), and `max_top_day_pnl_share` when `day_abs == 0` — a concentration cap can pass without being measured while trades exist. CS §3.4 doesn't define missing-value semantics; the chosen semantics should be explicit and arguably fail-closed when `executed_trades > 0`. Evidence of partial mitigation: with zero trades `min_executed_trades` fails first.

### F10 — Vacuous drill-target pass reaches the control-flow gate
`child_replay.py:798-805` — zero entities ⇒ `resolves=True` + `vacuous_zero_targets=True`; the Path-A gate consumes only `link["resolves"]` (:698-700), so `verifier_link_resolves` can pass with zero links exercised on a funnel-quiet window. Acknowledged in DEVIATIONS ("the gate input is honest") — but the honest flag rides evidence only, not the gate. Resolution: record `vacuous_zero_targets` next to the gate result in the stamped summary, or make the gate tri-state in the report payload.

### F11 — Deep-immutability read-hole on frozen report objects (CS §0.3; TM §3.9 last clause)
`ImmutableMap.__getitem__` returns internal references; values that are plain dicts are mutable in place after construction: `ContrastResult.effect_estimate[metric]` (`contrasts.py:156-174` stores per-metric dicts) and `StrategyMetrics.trade_stats` (`strategy_metrics.py:37,67,194`). Identity payloads are clean — `test_identities.py:110-146` proves no plain mutable container is reachable on any registered payload — but TM §3.9 extends the requirement to "frozen result/report objects… only immutable wrappers or defensive copies", which these two violate (`report.effect_estimate["x"]["paired_mean_delta"] = 999` succeeds). Secondary: `ImmutableMap._coerce` collapses duplicate keys silently for list-of-pairs input (`identities.py:167-173`) while the direct constructor raises (`:117-118`) — inconsistent fail-closed behavior on the validation path. Resolution: deep-freeze values (reuse `deep_freeze`) or return copies on read; unify duplicate-key handling.

### F12 — `canonicalize_section` compares content only against the baseline of its own claimed name
`identities.py:527-534` (DEV-R2-4). A section whose name-free content equals a **different** registered baseline's content (or an unregistered-named section carrying a baseline's exact content) is renamed to `ifvg_search_profile_<hash16>` while that baseline runs under its registered name — two names for identical semantics, hence divergent record IDs (CS §1.3; TM §3.1 P0-2 "two different names for identical semantics are impossible"). Latent today: the five registered baselines differ on thesis-locked, non-searchable fields, so the state is unreachable through registered axis values; `test_baseline_profiles_keep_their_registered_names` covers only the own-name case. Resolution: compare the name-free hash against **all** registered runnable baselines and adopt the matching registered name.

### F13 — Cancel sentinel is never consumed; skipped phases record no reason
`orchestrator.py:303-308` writes the sentinel; nothing ever deletes it, and it is checked before child 0 (:408), so any resume after a cancelled run (or a late cancel after the last child) re-cancels everything forever until the file is hand-deleted. Also the comment at :507-509 claims prop/robustness phases are "recorded honestly as skipped-with-reason", but the checkpoint payload (:323-342) records only the phase string — no skip reason exists anywhere. Resolution: consume (delete) the sentinel when honored; add a `phase_notes`/skip-reason field to the state payload.

### F14 — Audit-companion reuse does not verify sidecar bytes; publication order risks an ungated crash window
`store.py:235-258` compares only envelope JSON on reuse; `publish_child_fsm_audit`'s docstring claims "reuse verifies byte-identical stored content" (`fsm_audit_preparation.py:463-465`) — untrue for the parquet/report sidecars, whose content is deliberately outside the identity (same `FsmAuditArtifactIdentity` with different table bytes silently "reuses" the old bytes). Also `build_slice_companions` publishes the audit companion (:885) before its gating neutrality report (:888-892) — a crash between the two leaves a gated artifact stored without its stored gate evidence. Resolution: on reuse, hash the new sidecars and compare against the stored manifest entries (fail closed on mismatch); publish the neutrality report first.

### F15 — `_audit_referential_integrity` fail-open edge on an empty lifecycle table
`child_replay.py:337` — `if referenced and setups and not referenced <= setups` — when the core lifecycle table is empty but audit frames reference setup ids, integrity returns True (neutrality's `audit_stamp_referential_integrity` passes), inconsistent with `verify_exact_drill_targets`, which treats the same situation as a hard failure (:759-762, and test :350-356). The audit companion would publish before the link check fails. Resolution: drop the `and setups` guard (empty set comparison already handles it: any referenced id with no setups is orphaned).

### F16 — Pareto objective direction inferred from a hardcoded set
`orchestrator.py:96-98` (`_MINIMIZE_METRICS`); the charter's `pareto_objectives` are bare names (CS §3.1 — the plan itself carries no direction field). Any minimize-flavored metric not in the frozenset (e.g., a future `cost_r`) is silently maximized. Also `orchestrator.py:496-500` drops None-valued objectives from a passing child's metric dict, which `build_frontier` then rejects with a raw ValueError mid-run (uncaught → search aborts after replays). Resolution: a registered objective→direction map with fail-closed refusal of unknown names; catch and record frontier construction failures as a typed search failure.

### F17 — Interaction refusal reuses `UnbalancedDesignError` semantics for balanced grids; strata keep-last overwrite
`contrasts.py:119-123` — a fully-crossed two-axis interaction is refused with an error whose registered meaning (DT §5) is "grid not fully crossed and no adjustment method registered" — misleading for a balanced grid. On DEV-R2-5's lawfulness: **lawful** — DT §5's enforced-rules list does not time interaction *evaluation*, the payload/identity remain fully declarable, the R2 PHASED gate does not require contrasts, and nothing silently computes; but the refusal deserves its own typed error naming the R4 landing, and §7A.19.11 must actually be re-gated at R4. Separately, `contrasts.py:144-147` silently overwrites when two cells occupy one (stratum, axis-value) grid position (`strata.setdefault(...)[value] = cell`) — ambiguity should refuse; `denominator` counts both cells while `matched_pairs` contains only the survivor.

### F18 — Frontier trace embeds a forbidden substring; `FrontierResult` schema is missing CS §12 fields with no deviation
`frontier.py:125-127` — the persisted trace line "…(not a **publishable** representative)" contains the substring "publishable", which the substring-based guards (`insights.FORBIDDEN_INSIGHT_WORDING`, future FUX-LABEL scans) will trip if the trace is ever rendered; rendering it through `_assert_wording_lawful` raises today. Also `FrontierResult` lacks CS §12's highest-payout/reliability/lowest-breach ids and per-firm champions (prop wiring is R3, but the schema divergence is unrecorded in DEVIATIONS). Resolution: rephrase the trace line ("development lane only; outer protocol required for publication"); note the R3 field additions as a planned schema extension.

### F19 — Program-allowlist canonical marker path is CWD-relative
`verification.py:260` — `canonical = SEARCH_TEST_STORE_ROOT` (a relative `Path`) resolved against the process CWD, not `repo_root`; run from a different CWD, the cross-root canonicality check silently sees no marker (per-root marker still applies). R1 code, but the R2 slice (`child_replay.py:609-610`) is its caller. Resolution: pass `repo_root / SEARCH_TEST_STORE_ROOT` from the slice.

---

## Verified clean (area — one-line evidence)

1. **Gate threshold coverage**: all 11 `ResolvedStrategyGateThresholds` fields evaluated, field set matches CS §3.4 exactly (`charter.py:72-83`, `gates.py:91-195`; test asserts 11 checks); min gates fail closed on missing values.
2. **Failure vocabulary**: `FailureReason` is exactly the 15 CS §12 values (`failure.py:14-31`); every gate/orchestrator mapping stays inside it.
3. **Canonical-naming serialization mirror**: `name_free_section_hash` (`identities.py:492-502`) is byte-for-byte SC's `ifvg_profile_hash` construction minus the popped `profile_name`; DEV-R2-4 correctly kills "one baseline name for different semantics" in the enumerate path (`test_identities.py:361-393`).
4. **Doc-default parity gate untouched**: `git diff` of `fsm_audit_preparation.py` shows only imports plus the appended per-child section; `prepare_ifvg_fsm_audit_persisted`/`compare_v2_exact` unmodified.
5. **DEV-R2-1 dataset change**: diff-verified — all new behavior is gated on `audit_rows is not None` / `audit_capture_mode != "disabled"`; the plain path is byte-identical (only an unused counter accumulates).
6. **Audit-artifact neutrality gating**: `build_child_fsm_audit` refuses absent, mismatched-id, failed neutrality and missing trace retention — the *audit artifact* half of §3.3 is enforced (the *child publication* half is F1).
7. **Identity projection**: `ChildFsmAuditEnvelope`, `ChildAuditNeutralityEnvelope`, `SearchChildMembershipEnvelope`, `DeclaredContrastEnvelope`, `CohortEnvelope` all registered with example factories; registration enforces no self-id, no forbidden display/attempt fields, exact envelope field sets; audit + tamper + mutation-walk tests pass over every pair.
8. **Deep immutability on identity payloads**: the mutation walk proves no plain dict/list/set is reachable on any registered payload; new identity-bearing contracts use `ImmutableMap`/tuples. (Report-object hole is F11.)
9. **Lineage rules**: collisions recorded, never deduped/keep-first/fuzzed; incomplete kinds disable deltas; `match_basis` mandatory; `not_comparable` reports carry no keys; divergence reasons confined to the registered 8-value vocabulary with `unattributed` fallback; source-scan test for fuzzy tokens passes.
10. **Funnel deltas**: union vocabulary, zero-filled + flagged one-sided counters, typed-None conversions, terminal-reason deltas.
11. **Insights**: seven fixed categories always render; `not_comparable` populations suppress commonality claims entirely; dynamic text passes the forbidden-wording guard; wording observational.
12. **Frontier/robustness determinism**: sorted-id O(n²) dominance, deterministic champions and tie-breaks, persisted-in-object trace lines, `outer_fold_recurrence` schema-reserved None.
13. **Contrasts mechanics**: no imputation (grid holes, conditioning violations, missing metrics all refuse); seed-7 bootstrap deterministic; `causal_language_permitted: Literal[False]`; DEV-R2-6 orientation deterministic and recorded via `matched_pairs`.
14. **Orchestrator basics**: cancel honored at child boundaries only, completed children immutable+reusable; enumeration deterministic and deduped with the ceiling enforced; blocked capability checked before reuse/replay in-process; P0-D pending-value blocking works through the synthetic marker rules (residual gaps are F2/F6/F13).
15. **Slice safety rails**: fail-before-path validation, synthetic-marker refusal, `search_test/v1` namespace pinning, program-allowlist marker, seed profile-binding and allowlist-continuity refusals, restricted seed unpickler, read-only provenance adapter with cutoff cap and `cached_artifacts_only` enforcement.
16. **DEV-R2-3 job shim**: worker refuses without `--runner-entry`; nothing launches at import.
17. **Verifier `setup_id` jump**: fourth exact kind with typed 0/multi-candidate refusals, no fallback — FUX-DRILL-compatible.
18. **Store discipline**: refuse-if-exists, tmp+`os.replace`, manifest+hash verification, reload-assert, same-id-different-content fail-closed, `fsm_audit_companions` added per DEV-R2-2 (reuse-verification overclaim is F14).
19. **Suite status**: all 63 in-scope tests pass (38 + 25) under `-p no:cacheprovider` with bytecode writes disabled.

## Deviation rulings
- **DEV-R2-1**: lawful and diff-verified. **DEV-R2-2**: lawful (reuse-verification overclaim is F14). **DEV-R2-3**: lawful. **DEV-R2-4**: lawful defect fix (residual latent gap is F12). **DEV-R2-5**: **lawful** per DT §5's text (error-semantics reservation in F17). **DEV-R2-6**: lawful.
- **Scoping note on reused-children gating**: honest about the mechanism but its compensating claim ("results surfaces read published artifacts") is not yet satisfiable — see F5.

Summary: 1 blocker, 6 majors, 12 minors. The strongest parts of the release are the identity/immutability layer, the lineage collision semantics, and the audit-artifact half of the neutrality gate; the weakest is that "blocks child publication" is currently enforced nowhere on the child-identity publication path.


---

## Reviewer B report (verbatim)

# Adversarial Review — R2 of `ifvg_prop_robust_config_search_v1` (Quant-Lab)

Reviewed: uncommitted working tree of `C:\Users\gonza\Documents\Claude-Quant-Lab`, branch `feature/ifvg-prop-robust-config-search-v1`, HEAD `3546254` (R1). Lens: protected/sealed access safety, M0–M3/immutable-artifact compatibility, pre-existing-surface preservation, R2-gate completeness. All paths below are relative to the repo root unless absolute. I did not execute any repo code; every claim is from static evidence. I did not verify the author's "pytest green / ruff clean" claims myself.

---

## FINDINGS

### MAJOR-1 — Contradictory `is_warmup` stamping inside the child audit companion, and the pending real slice will publish it into immutable artifacts

- **Finding:** `build_slice_companions` calls `build_child_fsm_audit(..., warmup_days=0)` while the capture that produced the audit frames ran under `cfg.warmup_days` (default **10**). The audit-channel/trace rows are stamped `is_warmup = chain_index < cfg.warmup_days` at capture time, but the companion's `DAY_FUNNEL` table is stamped from the separately passed `warmup_days`. On any chain shorter than `cfg.warmup_days` (the 3-day synthetic chain, and the real ≤5-day Path-A slice), the same published artifact says every audit row **is warmup** and every `DAY_FUNNEL` row **is not**. Independently, the slice's v2 dataset stamps all evidence rows `is_warmup=True` and its invariant audit will report `evidence_capture_count = 0` for a slice whose entire point is "0 warmup + ≤5 evidence days" (TEST_MATRIX §1 Path A).
- **Severity:** major (data-integrity defect in a new immutable-artifact type; it will land in the real Path-A acceptance artifact when the owner authorization arrives).
- **Citation:** PHASED_DELIVERY.md R2 modified-list ("per-child neutrality-aware audit build"); TEST_MATRIX §1 Path A "0 real warmup days + ≤5 evidence days".
- **Evidence:** `dataset.py:554` (`is_warmup = chain_index < cfg.warmup_days`, audit-enabled drive); `child_replay.py:880` (`warmup_days=0` hardcoded); `config.py:131` (`warmup_days: int = 10` default; slice cfg does not override); `reporting.py:511-515` (invariant audit counts from `is_warmup`); contrast `dataset.py:941/953/1011` (untouched builder self-consistent on `cfg.warmup_days`).
- **Reproduction:** run `test_child_audit_companion.py::test_child_audit_build_assembles_and_reconciles`, inspect `HTF_TAP["is_warmup"]` (all True) vs `DAY_FUNNEL["is_warmup"]` (all False). No test checks this consistency.
- **Resolution:** derive both stamps from one source — pass `cfg.warmup_days` into `build_child_fsm_audit` (and set the slice cfg's `warmup_days=0`, matching Path A), or add a fail-closed consistency check in `build_child_fsm_audit`.

### MAJOR-2 — `resolve_selection(setup_id=…)` is untested and unreachable from the tab flow, despite the gate requiring "contract tests updated in the same change"

- **Finding:** The new provider branch (single-candidate resolve, zero-candidate refusal, multi-candidate refusal with exact count) has **no test anywhere**. `tests/agents/test_ifvg_replay_chart_provider.py:307-311` still tests only the R1 kinds. It is also dead code for the only current caller: the verifier tab intercepts `setup_id` jumps in `_route_pending_setup_jump` **before** `_apply_pending_jump` ever calls `resolve_selection`.
- **Severity:** major (PHASED R2 modified-list: "`setup_id` jump kind, **contract tests updated in the same change**"; FUX-DRILL-001 R2 part is the I-level provider contract).
- **Evidence:** `replay_chart_provider.py:408-423`; `ifvg_verifier_tab.py:113-126`; grep shows no `setup_id` usage in provider tests.
- **Resolution:** add provider contract tests for the three setup outcomes and the exactly-one-ID rule including `setup_id`; document that the tab path uses setup mode and this branch serves programmatic/results callers.

### MAJOR-3 — Reused children are excluded from gates/frontier; a fully-reused (cross-study or resumed) search completes with no metrics, no gate reports, and `frontier=None`; nothing persists metrics/frontier/insights

- **Finding:** `run_search` computes metrics/gates only for children whose tables came from this session's `child_runner` invocations. A `reused` child never gets metrics or a gate report (`orchestrator.py:481-483`): (a) the second study in cross-study reuse finishes `search_complete` with zero gate evaluations and no frontier; (b) a resumed-after-kill run produces a *different* final result than an uninterrupted run (§3.5 "repeat-identical" cannot hold at the result level); (c) the `frontiers` and `insights` stores are never written — metrics/gates/frontier are in-memory only, so reuse has nothing to reload by design.
- **Severity:** major (the R2 gate's E2E chain "enumerate→**reuse**→gates→frontier→insights" is only proven piecewise; reuse and gates/frontier are never proven together and are structurally incompatible in the current code).
- **Citation:** PHASED_DELIVERY.md:69; TEST_MATRIX §3.5.
- **Evidence:** `orchestrator.py:481-483`, `:435-440`; grep: no writes to `"frontiers"`/`"insights"`; `test_orchestrator.py:296-308`, `:458-470`.
- **Resolution:** persist per-child result references (the costed-evaluation store exists for exactly this) and reload for reused children before the gate pass, or mark gates/frontier for reused children explicitly skipped rather than silently gate-free.

### MAJOR-4 — "Funnel-delta → exact-setup drill-through into the verifier" has no end-to-end proof; the chain's only entity source is population-delta divergence, and its verifier link is the untested branch from MAJOR-2

- **Finding:** `FunnelDeltaReport` carries counters only (consistent with DT §4.4). The exact-setup drill evidence comes from `PopulationDeltaReport.first_divergence.entity_key` + insights' `EvidenceRef(kind="setup")` + the verifier's `setup_id` jump. Each link has a unit test, but no test drives the chain end to end.
- **Severity:** major as gate-completeness (a named R2 gate item; the weakest-covered of the nine).
- **Evidence:** `funnel_delta.py:33-38`; `insights.py:277-297`; `test_ifvg_verifier_tab.py:57-70` (router tested with a literal string, not an artifact-derived id); no integration test crosses the modules.
- **Resolution:** one integration test: population delta over the hand-built populations → `first_divergence.entity_key` → `queue_jump`/router → provider resolution over a context fixture, asserting exact resolution and sanitized failure for a missing id.

### MINOR findings

- **m1.** Vacuous drill-target pass reaches the gate boolean; the acceptance runner must require `target_count > 0` for the real slice (`child_replay.py:823-828` vs `:960`).
- **m2.** Sidecar reuse is not byte-verified; `publish_child_fsm_audit` docstring overclaims (`store.py:249-256`).
- **m3.** Prop/robustness phases advance without skip annotations; the "skipped-with-reason" comment is not reflected in the state payload (`orchestrator.py:507-514`).
- **m4.** Failed-replay child path (`except Exception` → `FailureReason.REPLAY` + sanitize) untested.
- **m5.** "Repeat-identical" (§3.5) untested at the orchestrator level (no same-charter re-run asserting identical published envelopes).
- **m6.** Job shim: `--runner-entry` accepts any importable `module:function` (registry-gate no later than R4 UI wiring); `start` arm untested; AST import-launch test weak; `job.log` handle never closed; POSIX detach flags are 0 (cosmetic).
- **m7.** Change-set hygiene: stale user doc hunks (2026-07-30/31) entangled in the working tree with R2's doc updates (must be separated at commit); `dataset.py` is an out-of-list modified surface for R2 (documented as DEV-R2-1 but note in the release evidence).
- **m8.** Lineage P1-E sub-clauses only implicit; `lineage_valid` is caller-asserted — no function derives lineage validity from a changed-axis set; "entry-thesis change → not comparable" unproven (a changed family yields a lawful-looking Jaccard-0 delta unless the caller passes `lineage_valid=False`).
- **m9.** Unsourced cost literal `0.514` in `build_slice_companions` (`child_replay.py:872`) — should come from the registered cost policy.

---

## VERIFIED CLEAN

1. **Pre-existing modified modules**: dataset.py audit-disabled path byte-identical (all four pre-existing `build_ifvg_v2_capture` callers use the default disabled mode); `prepare_ifvg_fsm_audit_persisted` and the doc-default parity gate untouched; `resolve_selection`'s three existing kinds behave identically; verifier tab adds no forbidden control; the router leaves candidate/decision/trade jumps queued; tab contract tests updated in the same change for the queue/router half.
2. **Date/data safety**: zero date literals ≥ 2026-06-11 in any new R2 module or test (June-11/12 literals that exist are R1-committed refusal tests); no listing/glob/walk in new modules; store lookups exact-ID with allowlisted store names and 64-hex validation; no reference to real data roots (the shim's default store root is the sanctioned search_test namespace; every test passes tmp roots); all writes under caller-supplied roots; `data/ifvg_datasets/search*/` and `data/ifvg_search_jobs/` do not exist — no full-development run occurred.
3. **Job shim safety**: no argv combination reaches a replay without `--runner-entry`; search-id validated before any path; traversal refused; Popen list-form exact interpreter/script; import launches nothing.
4. **R2 gate mapping**: eight of nine gate bullets mapped to passing proofs (detail in report); the weak bullet is the drill-through chain (MAJOR-4); strategy-only-cell rows are R1's `test_study_cell.py` (§3.9 satisfied; §3.8's prop-sim half correctly deferred to R3 by the row's own gate).
5. **M0–M3 / propsim / cross-repo / decisions**: `git status` clean over every M0–M3 lane module, all of `src/alpha_lab/propsim/`, `scripts/run_ifvg_experiment.py`, `scripts/ifvg_lab_tab.py`; `docs/DECISIONS.md` untouched; Strategy-Core clean at `a4e3303`; Trade-Lab's dirty files are the user's unrelated platform-refactor work; `registered_identity_pairs` pulls the four new envelope pairs into the projection-audit net with no import cycle; docs-in-same-change posture stated honestly.

## Summary

No blockers. Four majors: the warmup-stamp contradiction, the untested/unreachable `resolve_selection(setup_id=…)` branch, reused children structurally excluded from gates/frontier with nothing persisted, and no end-to-end funnel-delta→setup→verifier proof. Nine minors, mostly coverage and hygiene. Access safety is genuinely clean; M0–M3, propsim, Strategy-Core, Trade-Lab, and `docs/DECISIONS.md` are untouched. R2 acceptance remains correctly transitively blocked on the R1 owner fixture authorization.
