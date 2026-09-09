# R6 — Access-Safety Evidence

Implementer's audit, corrected after the adversarial round; independently
corroborated by the safety-lens reviewer (`ADVERSARIAL_REVIEW.md`,
Reviewer 2 — protected/sealed zero-counter **AFFIRMED**).

## Protected/sealed counters: ZERO

- The regime lane constructs NO source path anywhere: its inputs are
  in-memory frames (candidate views / synthetic panels) supplied by the
  caller. No file open, list, stat, or read of any real namespace exists
  in the seven new modules; the only filesystem I/O is the verified store
  protocol on caller-supplied roots (tmp roots in every test; the
  content-addressed `%TEMP%` scratch in the browser smoke).
- Fixtures use synthetic business days from `2026-01-05` (bdate ranges)
  only — no protected (2026-06-11) or sealed (≥ 2026-06-12) date literal
  exists in the R6 set (reviewer grep over all 13 files: no hit; the only
  post-June literals are decision timestamps from which no path is derived).
- On disk (reviewer): `find data -type f -newermt 2026-08-25` → 0 files;
  `data/ifvg_datasets/search` and `search_test` do not exist; no
  `regime_*` store directory exists anywhere in the repo.
- The five-day real mini-run (ML §9) is the SAME owner-blocked real slice
  as every real half since R1: this layer "never constructs real paths
  itself" by design; the real half waits on the
  `VerificationAuthorizationRef` (standing first blocker).

## No automated selection or promotion (kickoff §9 ML/regime)

- Cluster count is `fixed_k` only; `inner_train_only_selection` (and PCA,
  kernels, every non-`centroid_predict_v1` OOS policy) is
  schema-expressible and execution-REFUSED — `assert_protocol_executable`
  fails closed before any preprocessing or fit (review S3); no code path
  selects k, features, thresholds, or algorithms.
- Promotion is STRUCTURAL after the round (reviews F5/S1/S2): the
  `RegimePromotionDecision` contract itself enforces one ladder step,
  `previous_decision_ref` chaining, ISO-8601 `decided_at`, a 64-hex
  `OwnerDecisionEvidenceRef` content hash from `FEATURE_ELIGIBLE` onward,
  and the ROLE ladder (execution-side roles — decision policy,
  execution-gate candidate, frozen execution gate — are unrepresentable
  in V1 and carry the exact S11 reason); `persist_regime_promotion` then
  loads the referenced assessment from the store (must exist, verify, and
  name the protocol) and re-checks the ladder with the assessment's OWN
  `gates_passed`, so `FEATURE_ELIGIBLE`+ over a failing or absent
  assessment is unpersistable. The midpoint wording ("at both … the
  contract itself") was an overclaim; this is what the code does now.
- Diagnostics are report BUILDERS; no consumer API retrains, disables, or
  promotes from any score; internal scores (silhouette) are
  descriptive-only fields.
- The sample-adequacy/occupancy/rows/AMI gates BLOCK promotion; they never
  shrink k or delete fits.
- `IFVG_REGIME_CONTEXT_V1` remains a PLANNED feature block (resolver
  refusal re-tested this release): no regime output can enter a
  predictive bundle. S11 remains blocked with the exact reason; nothing in
  R6 feeds a model-gated replay.

## Input permission is structural (reviews F2/S4)

- `input_feature_bundle_ref` is a 64-hex identity that must resolve to a
  registered AVAILABLE bundle; every input feature must belong to it.
- The availability-stage rule is DEFAULT-ON from the feature-block
  registries (a feature with no registered stage is refused as unprovable);
  the resolver and `run_regime_protocol` both apply it.
- Every fit identity pins verified `source_artifact_ids` (never empty) and
  the hash of the training feature matrix (review F1).

## V1 algorithm boundary

`kmeans_v1` is the single implemented algorithm (import-time invariant;
every planned entry executes no policy — import-time invariant);
`GaussianMixture`, `SpectralClustering`, `Nystroem`, and `MiniBatchKMeans`
are imported NOWHERE in the R6 set — planned entries exist as registry
rows with fail-closed refusals only. No torch/keras/gym/deep-learning
import exists (7B.22-19). Registry drift is unexecutable: the protocol's
`pinned_parameters_hash` and `algorithm_version` are re-verified against
the registry at run time.

## Store discipline and UI safety

- The four regime stores follow the manifest protocol via
  `save_or_reuse_envelope`; a fit's exact bytes are verified (reload from
  bytes → re-transform allclose → re-predict equals the persisted labels)
  BEFORE publication, then re-verified through the store, and a fresh
  entry that fails the post-publish check is withdrawn (review S8); the
  assignment frame must be THIS fit's (review F6); every reference is
  manifest-relative (relocation-tested); tamper on any sidecar fails
  closed on load.
- `load_sidecar_bytes` (shared store) now verifies the manifest hash,
  whitelists the sidecar name (no traversal, no store-owned names), and
  hashes the bytes it returns (review S6).
- Trust boundary: store integrity is not authenticity (DEV-R6-8); the UI
  never unpickles (`load_regime_fit_assignments` — JSON + Arrow only).
- All test writes are `tmp_path`-rooted; no repo data namespace gained
  files (safety reviewer's on-disk check).
- The Regime Lane panel renders registry state and exact-ID loads only
  (64-hex validation at the store layer; stores never listed); errors are
  sanitized; no button/form/toggle/select control exists (source-scanned
  by test); no launch/promote/rank/retrain/live/sealed control exists;
  `_spawn_pipeline_job` call sites are unchanged (the expander only
  reads); both panel scripts are now under the FUX source scans (review
  S5).

## Frozen lanes

None of the M0–M3 lane modules, no propsim module, and no
Strategy-Core/Trade-Lab path changed in R6 (the Trade-Lab worktree's
pre-existing user modifications — 27 entries, newest mtime 2026-07-31 —
remain untouched; Strategy-Core clean). The plan package is unmodified.
