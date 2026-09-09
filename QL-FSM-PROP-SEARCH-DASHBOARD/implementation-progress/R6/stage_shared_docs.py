"""Stage the three shared docs as HEAD + R6 lane transforms ONLY.

Same mechanism as ``../R5B/stage_shared_docs.py`` (user hunks never enter
the release commit; ``--apply-worktree`` post-commit replays the same
transforms so the surviving diff is the user's pre-existing hunks only).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ARCH_APPEND = """
R6 additions (V1 KMeans regime lane — protocol/fit/capability/promotion
split per V3 P1-3; the GMM/minibatch/spectral/Nyström implementations are
the post-V1 regime-expansion release):

- **Regime contracts** (`ifvg/ml/regime_contracts.py`): the four-way split —
  `RegimeProtocolPayload` (algorithm key ≠ resolved id; the registry's
  pinned parameters hashed into the identity; the P1-B panel grain in the
  ACTUAL schema: `CONTEXT_BAR_PANEL` requires interval/source/as-of
  fields, `CANDIDATE_STAGE_ROW`/`DECISION_ROW` refuse them; a 64-hex
  bundle reference; pinned seed 7; `fit_scope="per_training_fold"`),
  role-free `RegimeFitPayload` (verified non-empty source artifact ids +
  the training-feature-matrix hash — different feature values are a
  different fit identity), `RegimeCapabilityAssessment`
  (coverage/occupancy/stability over EXACT fit + fold identities with the
  applied gate values), and `RegimePromotionDecision` — the contract
  itself enforces one ladder step at a time, a chained
  `previous_decision_ref`, an ISO-8601 decision time, a 64-hex owner
  ratification reference from `FEATURE_ELIGIBLE` onward, and the ROLE
  ladder (execution-side roles are unrepresentable in V1 with the exact
  S11 reason); it never touches a numerical identity, and
  `persist_regime_promotion` re-checks the ladder against the referenced
  assessment's own `gates_passed` (a FEATURE_ELIGIBLE decision over a
  failing or absent assessment is unpersistable). `assert_no_regime_leakage`
  refuses every outcome/label/payout column and — from the feature-block
  registries, by default — every feature available only after the
  protocol's observation stage; every input must belong to the referenced
  bundle; every scientific default (KMeans baseline, k=3, the
  occupancy/rows/AMI gates, the sample-adequacy minimums, the grain
  baseline) is stamped `proposed_protocol_default` in
  `REGIME_PROPOSED_DEFAULTS`.
- **Algorithm registry** (`regime_algorithms.py`): `kmeans_v1` is the ONE
  implemented V1 algorithm (pinned k-means++ / n_init 10 / lloyd / seed 7);
  `minibatch_kmeans_v1`, `gaussian_mixture_v1`,
  `spectral_clustering_train_only_v1` (train-fold-only affinity,
  observation/memory caps, forced `BLOCKED_NO_OOS_ASSIGNMENT`, the
  mandatory training-only warning text), `nystrom_kmeans_v1`, and the
  surrogate-assignment entry are registered PLANNED with fail-closed
  refusals carrying the exact status/reason (P1-C) — no spectral or
  Nyström fit is callable anywhere in V1, and every planned protocol
  POLICY (`inner_train_only_selection`, PCA, kernels, non-centroid OOS
  policies) is refused by `assert_protocol_executable` before any
  preprocessing or fit.
- **Fold-local service** (`regime_preprocessing.py` · `regime_service.py` ·
  `regime_alignment.py` · `regime_diagnostics.py`): the fixed per-fold
  pipeline (median imputer + indicators → optional train-fitted p01/p99
  winsorizer → standard scaler) whose fit API accepts a FOLD and slices
  internally (no row-subset parameter; all-missing training rows never
  enter a fit; duplicated observation keys refuse); per valid fold the
  pinned KMeans fits on training rows and assigns train+test rows
  deterministically (distances to every centroid, margin d2−d1, the
  observation's as-of timestamp); rows are PRESERVED with typed reasons
  (`source_feature_missing`/`fold_invalid`/`coverage_gap`); Hungarian
  alignment is reporting-only over NOMINAL, geometry-ranked canonical ids
  in the scaled input-feature space with the exact ascending-local-id
  tie-break (prediction-hash invariance test-proven); the stability report
  carries seeded bootstrap aligned-AMI, per-cluster agreement, the
  OUT-OF-SAMPLE timeline's temporal persistence and transition matrix
  (ordered by observation timestamp within each fold), fold-to-fold
  recurrence, centroid separation, and descriptive-only silhouette; the
  sample-adequacy gate (≥150 candidate-stage/decision-row, ≥300 panel
  training rows per fold) and the occupancy/rows gates block PROMOTION —
  k is never shrunk. The panel→candidate assignment consults only
  out-of-sample assignments of the last COMPLETED bar at or before each
  candidate's as-of instant (lowest fold wins; 5m/15m tested end to end);
  earlier candidates and bars without an OOS assignment are typed
  `coverage_gap`.
- **Persistence** (`regime_store.py` + four new store names on the
  manifest protocol): dual-format fitted artifacts (canonical JSON
  parameter payload hashed as `fitted_parameter_payload_hash` + joblib)
  bound to their own assignment frame and verified from the exact bytes
  BEFORE publication (reload → re-transform `np.allclose` → re-predict
  equals the persisted labels), then re-verified through the store;
  manifest-relative references only and relocation-proof reload (P1-4);
  the shared `load_sidecar_bytes` verifies the manifest hash, whitelists
  the sidecar name, and hashes the returned bytes; assessments and
  promotion decisions persist as content-addressed envelopes — a
  promotion provably changes no fit identity.
- **Regime Lane UI** (`scripts/ifvg_regime_panels.py`, mounted as a
  pipeline-surface expander beside the MBP-1 panel): the algorithm
  registry with planned entries visible-disabled and the mandatory
  spectral warning; the proposal-stamp table; the exact-ID model card
  (grain identity incl. panel fields, input bundle, fixed-k stamp,
  OOS/alignment policies, coverage + per-fold coverage, fit identities,
  nominal-id occupancy, stability with per-cluster agreement and centroid
  profiles, the transition matrix, and the insufficient-sample blocked
  state), the exact-fit-id assignment / stratification / OOS-timeline
  view, and the promotion role/status view — read-only (no promote,
  launch, rank, or retrain control exists; the UI never unpickles).
  Stratified RESULT views (metrics by regime) are the ML §5.5 comparison
  classes and land with their studies. `IFVG_REGIME_CONTEXT_V1` remains a
  planned feature block and S11 remains blocked: no regime output can
  reach a predictive bundle or execution surface in V1.
"""

README_OLD = """`scripts/ifvg_mbp1_panels.py`) implementations are complete;"""
README_NEW = """`scripts/ifvg_mbp1_panels.py`), and R6 (the V1 KMeans regime lane:
`ifvg/ml/regime_{contracts,algorithms,preprocessing,service,alignment,
diagnostics,store}.py`, ML fixtures 2 + 4-KMeans, and the Regime Lane
panel `scripts/ifvg_regime_panels.py` — kmeans_v1 only; the
GMM/minibatch/spectral/Nyström implementations are the post-V1
regime-expansion release) implementations are complete;"""

YAML_OLD = (
    '    R6_kmeans_regime_lane: {implementation_status: "not_started", '
    'acceptance_status: "transitively_blocked_by_R1"}'
)
YAML_NEW = (
    '    R6_kmeans_regime_lane: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1"}'
)

TRANSFORMS: dict[str, list[tuple[str | None, str]]] = {
    "ARCHITECTURE.md": [(None, ARCH_APPEND)],
    "docs/README.md": [(README_OLD, README_NEW)],
    "docs/pipeline_state.yaml": [(YAML_OLD, YAML_NEW)],
}


def _git(*args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args], input=data, capture_output=True, check=True
    )
    return result.stdout


def _transformed(head: str, transforms: list[tuple[str | None, str]]) -> str:
    staged = head
    for old, new in transforms:
        if old is None:
            staged = staged.rstrip("\n") + "\n" + new
        else:
            if old not in staged:
                raise SystemExit("transform anchor missing at HEAD")
            staged = staged.replace(old, new)
    return staged


def main() -> int:
    apply_worktree = "--apply-worktree" in sys.argv[1:]
    for path, transforms in TRANSFORMS.items():
        if apply_worktree:
            worktree = Path(path).read_text(encoding="utf-8")
            updated = worktree
            for old, new in transforms:
                if old is None:
                    if new not in updated:
                        updated = updated.rstrip("\n") + "\n" + new
                elif old in updated:
                    updated = updated.replace(old, new)
            Path(path).write_text(updated, encoding="utf-8", newline="\n")
            print(f"worktree {path} updated with the R6 lane transforms")
            continue
        head = _git("show", f"HEAD:{path}").decode("utf-8")
        staged = _transformed(head, transforms)
        blob = _git(
            "hash-object", "-w", "--stdin", "--path", path,
            data=staged.encode("utf-8"),
        ).decode("ascii").strip()
        _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        print(f"staged {path} = HEAD + R6 lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
