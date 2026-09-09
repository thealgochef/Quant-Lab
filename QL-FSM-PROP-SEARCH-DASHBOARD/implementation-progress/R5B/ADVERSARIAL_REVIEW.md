# R5B — Adversarial Review (two independent read-only reviewers)

Per kickoff §10: independent reviewers attempted to falsify the release
against `CONTRACTS_AND_SCHEMAS.md`, `TEST_MATRIX.md`, `PHASED_DELIVERY.md`,
`DELTA_TAXONOMY.md`, `ML_REGIME_CONTRACT_PLAN.md`, the protected/sealed
access rules, and the M0–M3/immutable-artifact boundaries. Reviewers
produced findings only (no edits). Dispositions and post-fix verification:
`ADVERSARIAL_REVIEW_RESOLUTION.md`.

---

## Reviewer 1 — contract fidelity (verdict: 0 blockers, 3 majors, 8 minors)

**F1 (MAJOR)** — Evidence↔identity binding across the MBP-1 layer is
caller-asserted, never verified: `materialize_mbp1_features` pins
`mbp1_source_artifact_id` without checking `events_by_day` against the
coverage rows' `content_sha256`; `save_mbp1_source_artifact` never
cross-checks sidecar bytes vs coverage hashes; `build_bundle_feature_view`
and the controlled study pin `mbp1_feature_artifact_id` without verifying
the passed frame hashes to that artifact's `feature_table_sha256` (tests
exploited this with fabricated `"d"*64` ids). A mismatched (source, events)
pair mints a persisted feature artifact with untruthful provenance.
Citation: CS §0.1 ("pin … by its **verified ID**"); the repo's own R5-FIX
standard ("a caller-provided string is NOT evidence").

**F2 (MAJOR)** — `ladder_id` under-binds MBP-1 evidence: the bundle path's
`feature_source` carried only {resolved bundle id, names}; bundle
resolution is artifact-independent and `view_id` is unchanged by frame
replacement, so two challenger ladders over DIFFERENT MBP-1 feature
artifacts shared one surfaced `ladder_id` with different predictions.
Citation: CS §0.2; the release's own DEV-R5B-3 invariant.

**F3 (MAJOR)** — The mandated §3.8/DT §6.2 proof "a same-ts after-stage
event cannot change any feature" was not implemented at the MATERIALIZER
level: no test mutated an after-stage same-timestamp event and asserted
feature-frame equality, and no test exercised `materialize_mbp1_features`
with an exact-key `cutoff_builder` at all (the admission-level membership
test alone left the claim vacuous at the feature level).

**F4 (MINOR)** — No sensitivity test that changing a window definition
(comparator/bounds/min-count/missingness) mints a new
`resolved_feature_block_id` (TEST_MATRIX §3.10 rows).

**F5 (MINOR)** — The feature-layer legacy-provenance guard is decorative:
`_require_no_legacy_provenance` has no runtime ingress (no `source_kind`
channel exists in the package), and the guard test called the private
function directly rather than proving a surface property.

**F6 (MINOR)** — The `+inf` contract validator inspected only element [0]
of the exact key (`("123", "+inf", 0, 0)` validated), and the "no +inf
anywhere" source scan covered only the features package (not
`search/pipeline.py` or the ml modules that construct/consume cutoffs).

**F7 (MINOR)** — `events_stored` participated in the hashed source-artifact
identity: byte-identical evidence yielded TWO artifact ids (stored vs
referenced), forking every downstream feature/study identity. Citation:
CS §0.1 (operational facts are envelope facts).

**F8 (MINOR)** — `candidate_anchor_hash` omitted `setup_id`, which is a
column of the produced feature table — two anchor frames differing only in
setup labeling collided on one payload identity while producing different
tables (caught only by the store's same-id/different-content backstop).

**F9 (MINOR)** — In the model-path join (evidence columns off), a cohort
candidate absent from the feature artifact received bare NaN with no
registered missing reason recorded anywhere; the join docstring overclaimed
the reason attachment.

**F10 (MINOR)** — Deliverable 12's real five-day MBP-1 control-flow
verification is not runnable in this release (the real runner entry wires
no `mbp1_evidence_source`; a real MBP-1 plan fails S00) — declared as
DEV-R5B-1; listed because the deliverable text is not met inside R5B.

**F11 (MINOR)** — `ControlledFeatureStudyPayload` embeds materialization
RESULTS (arm summaries, the paired Brier delta) in the hashed payload,
argued to contradict the recipe/envelope split the same release establishes
for `Mbp1FeatureArtifactPayload`.

Checks attacked that SURVIVED (reviewer's list): comparator directions and
bound handling; the frozen 9-window registry; completed-bar strict-`<` with
ambiguity propagation and never-widened windows; the activation event
(version 2, first mint, registry-hash change, pre-activation provable,
published registry provably the event's output); deep-book guard and MBP-10
unrepresentability; the exact one-to-one join with cohort preservation;
S00 seam refusal; S05 persist-then-join; S09 logistic pinning; readiness
launch-blocking; deterministic retry re-derivation; S11 permanently blocked
with no path from the activated block to any execution-gating surface; the
hand-computed formula fixtures (independently re-derived — OFI −7,
absorption 5/1.5, intensities, microprice all check out).

---

## Reviewer 2 — safety/access (verdict: 0 blockers, 0 majors, 1 medium, 5 minors; protected/sealed zero-counter **AFFIRMED**)

**S1 (MEDIUM)** — Deeper-book partitions passed the real-source read seam:
`assert_schema_names_match` is a subset check and `pq.read_table` read the
WHOLE file, so an mbp-10-shaped parquet (every pinned column plus
`bid_px_01…`) passed the gate with its deeper columns materialized in
memory before being discarded. Exposure-side boundary held (the pinned
schemas represent level-00 only); the ingestion side did not distinguish
mbp-1 from deeper-book files. Practical exploitability low (seam reachable
only through an access policy; no shipped real wiring reaches it).

**S2 (MINOR)** — Deep-book regex blind spots: `mbp5`/`mbp2…mbp9` (single
digit ≥2) and level-index names (`bid_px_01`) were unmatchable — the regex
tracked the plan's literal MBP-10 wording, weaker than the §9 "MBP-1 is the
maximum" rule it defends. (No such identifier existed in any real content.)

**S3 (MINOR)** — `_require_no_legacy_provenance` had no runtime ingress
(see F5), and `build_mbp1_source_artifact_from_paths` called with empty
days + `policy=None` completed degenerately (no path constructed, no byte
read — the policy check was per-day rather than top-of-function).

**S4 (MINOR)** — Changed files outside the declared scope: `docs/DECISIONS.md`
(D-046 — additive, governance-required, but outside the enumerated
D-039–D-045 range) and five collateral test files (each the legitimate
R5→R5B inversion of suites for modules the release modified); the
progress-tree helpers (`stage_shared_docs.py` — the owner should be aware a
commit built with it intentionally differs from the worktree — and the
smoke app, %TEMP%-only).

**S5 (MINOR, observational)** — The Trade-Lab WORKING TREE is dirty with
~27 pre-existing user modifications, every mtime ≤ 2026-07-31 (a month
before this session); Strategy-Core is clean; nothing in this repo's diff
lies under either lane. The frozen-lane rule holds for this release; the
pre-existing user work should be acknowledged in the gate evidence.

**S6 (MINOR)** — `_mbp1_default_ids` swallowed store-integrity failures
silently: a tampered S05 sidecar rendered identically to "no runs yet"
(fail-closed and leak-free, but an integrity failure downgraded to a
cosmetic blank on the auto-fill path).

Verified clean (reviewer's evidence-backed list): no protected/sealed
literals anywhere in the changed set; authorize-before-path proven with
zero factory calls; no real runs (the real executors wire no MBP-1 seam;
S00's real branch validates before any source path; real artifacts persist
no event bytes); all test writes tmp-rooted with the repo's real data
namespaces untouched on disk (no writes today; `search/v1` and
`search_test/v1` do not exist); order-flow boundary structural
(`can_affect_execution=False`, import-time boundary invariant, S11 blocked,
zero launch/promote/sealed controls in the panels, no Trade-Lab import);
UI reads funnel through 64-hex exact-ID validation with no store listing;
the shim's store-root pass-through cannot widen access (the factory still
fail-before-paths on the persisted envelope; the policy re-refuses
protected dates independently); none of the frozen M0–M3/propsim modules
in the diff.
