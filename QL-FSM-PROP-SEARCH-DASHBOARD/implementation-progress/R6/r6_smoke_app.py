"""Live browser-smoke harness for the R6 Regime Lane surfaces.

Shell-only launcher (never part of the repo's importable surface): builds
one COMPLETED synthetic 16-stage pipeline run under a CONTENT-ADDRESSED
scratch directory (the same evidence-tied reuse discipline as the R5/R5B
harnesses — the key covers git commit + a source-tree digest of the
exercised surfaces + the fixture's ``pipeline_semantic_id``; stale keys are
cleared; reuse requires the persisted evidence to VALIDATE), then persists
two ``kmeans_v1`` regime runs over ML fixture 2 into that run's
verification store — a HEALTHY one (n=600; every gate passes) and an
UNDER-SAMPLED one (n=170; the sample-adequacy gate blocks promotion) — and
serves the real ``render_pipeline_run`` so a browser can walk the Regime
Lane expander (algorithm registry with planned-disabled entries + the
mandatory spectral warning, the proposal-stamp table, and the exact-ID
model card: grain identity, coverage, nominal-id occupancy, stability, the
transition matrix, the insufficient-sample blocked state) over genuine
persisted artifacts. The exact ids to paste are shown in the page header
and written to ``smoke_manifest.json``.

Run from the repo root:

    python -m streamlit run \
        QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R6/r6_smoke_app.py \
        --server.port 8599 --server.headless true

Everything is written under %TEMP%/ifvg_r6_smoke/<key>; the repo's data/
tree is untouched. No launch, promote, rank, or retrain control exists on
the served surface.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for entry in (str(_REPO / "src"), str(_REPO / "scripts"), str(_REPO)):
    if entry not in sys.path:
        sys.path.insert(0, str(entry))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="R6 Regime Lane smoke", layout="wide")

_SCRATCH_BASE = Path(tempfile.gettempdir()) / "ifvg_r6_smoke"

_DIGEST_GLOBS = (
    ("src/alpha_lab/agents/data_infra/ifvg", "**/*.py"),
    ("src/alpha_lab/propsim", "**/*.py"),
    ("scripts", "ifvg_*.py"),
    ("tests/agents/ifvg_search", "*.py"),
    ("tests/agents/data_infra/ifvg/ml_fixtures", "*.py"),
)


def _git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _source_tree_digest() -> str:
    entries: list[tuple[str, str]] = []
    for base, pattern in _DIGEST_GLOBS:
        root = _REPO / base
        for path in sorted(root.glob(pattern)):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append((path.relative_to(_REPO).as_posix(), digest))
    entries.append(
        ("r6_smoke_app.py", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    )
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _prepared_evidence_validates(fixture: dict) -> dict | None:
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
        PipelineResultEnvelope,
        read_pipeline_state,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope

    semantic = fixture["semantic"]
    state = read_pipeline_state(fixture["state_root"], semantic.pipeline_semantic_id)
    if state is None:
        return None
    terminal = {"completed", "reused", "blocked"}
    stages = state.get("stages") or {}
    planned = [name for name, entry in stages.items() if (entry or {}).get("in_plan")]
    if not planned or any(
        (stages[name] or {}).get("status") not in terminal for name in planned
    ):
        return None
    publication = state.get("publication") or {}
    result_id = publication.get("pipeline_result_id")
    if not result_id:
        return None
    try:
        load_verified_envelope(
            fixture["store_root"], "search_results", result_id, PipelineResultEnvelope
        )
    except Exception:
        return None
    return {"state": state, "pipeline_result_id": result_id}


def _persist_regime_runs(store_root: Path) -> dict[str, dict[str, str]]:
    """Persist the healthy + under-sampled kmeans_v1 runs (fixture 2) into
    the verification store; every artifact is save-or-reuse verified."""

    from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
        known_cluster_fixture,
    )

    from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
    from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
        ObservationGranularity,
        RegimePromotionDecision,
        RegimePromotionDecisionEnvelope,
        RegimeRole,
        RegimeStatus,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
        resolve_kmeans_protocol,
        run_regime_protocol,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
        persist_regime_assessment,
        persist_regime_fit,
        persist_regime_promotion,
        persist_regime_protocol,
    )

    bundle = resolve_bundle("B0_CORE").resolved_feature_bundle_id
    results: dict[str, dict[str, str]] = {}
    for label, n, winsorization in (
        ("healthy", 600, "none"),
        ("undersampled", 170, "clip_p01_p99_train_fitted_v1"),
    ):
        fixture = known_cluster_fixture(k=3, n=n)
        folds = build_context_folds(
            fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
        )
        protocol = resolve_kmeans_protocol(
            input_feature_bundle_ref=bundle,
            resolved_input_features=fixture.regime_input_features,
            winsorization_policy=winsorization,
        )
        run = run_regime_protocol(
            fixture.view.frame,
            folds,
            protocol,
            source_artifact_ids=(fixture.view.view_id,),
            bootstrap_refits=10,
        )
        persist_regime_protocol(store_root, protocol)
        fit_ids: list[str] = []
        for fold_fit in run.fold_fits:
            fold_assignments = run.assignments[
                run.assignments["fold_index"] == fold_fit.fold_index
            ]
            persist_regime_fit(
                store_root,
                fold_fit,
                fold_assignments,
                observation_frame=fixture.view.frame,
            )
            fit_ids.append(fold_fit.fit_envelope.regime_fit_id)
        persist_regime_assessment(store_root, run.assessment)
        decision = RegimePromotionDecisionEnvelope.from_payload(
            RegimePromotionDecision(
                resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
                role=RegimeRole.DESCRIPTIVE_ONLY,
                status=RegimeStatus.DESCRIPTIVE_ONLY,
                previous_status=RegimeStatus.PLANNED,
                previous_decision_ref=None,
                capability_assessment_ref=run.assessment.regime_capability_assessment_id,
                owner_ratification_ref=None,
                decided_at="2026-08-28T00:00:00Z",
            )
        )
        persist_regime_promotion(store_root, decision)
        results[label] = {
            "protocol_id": protocol.resolved_regime_protocol_id,
            "assessment_id": run.assessment.regime_capability_assessment_id,
            "fit_ids": ",".join(fit_ids),
            "first_fit_id": fit_ids[0],
            "decision_id": decision.regime_promotion_decision_id,
            "gates_passed": str(run.assessment.payload.gates_passed),
            "gate_failures": ",".join(run.assessment.payload.gate_failures),
        }
    panel_protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=bundle,
        resolved_input_features=known_cluster_fixture(k=3, n=60).regime_input_features,
        observation_granularity=ObservationGranularity.CONTEXT_BAR_PANEL,
        panel_interval_seconds=300,
        panel_source_artifact_id="b" * 64,
        panel_as_of_policy_id="completed_bars_last_at_or_before_v1",
    )
    persist_regime_protocol(store_root, panel_protocol)
    results["panel_grain_protocol"] = {
        "protocol_id": panel_protocol.resolved_regime_protocol_id,
        "assessment_id": "",
        "fit_ids": "",
        "first_fit_id": "",
        "decision_id": "",
        "gates_passed": "n/a (grain identity only)",
        "gate_failures": "",
    }
    return results


@st.cache_resource
def _prepared_run() -> dict:
    from tests.agents.ifvg_search.pipeline_fixture import build_pipeline_fixture

    from alpha_lab.agents.data_infra.ifvg.search.charter import save_charter
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import run_pipeline
    from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope

    commit = _git_commit()
    tree_digest = _source_tree_digest()
    probe = build_pipeline_fixture(_SCRATCH_BASE / "_probe")
    pipeline_semantic_id = probe["semantic"].pipeline_semantic_id
    key_material = {
        "commit": commit,
        "source_tree_digest": tree_digest,
        "pipeline_semantic_id": pipeline_semantic_id,
    }
    key = hashlib.sha256(
        json.dumps(key_material, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    _SCRATCH_BASE.mkdir(parents=True, exist_ok=True)
    for stale in _SCRATCH_BASE.iterdir():
        if stale.name != key:
            shutil.rmtree(stale, ignore_errors=True)

    scratch = _SCRATCH_BASE / key
    scratch.mkdir(parents=True, exist_ok=True)
    fixture = build_pipeline_fixture(scratch)
    assert fixture["semantic"].pipeline_semantic_id == pipeline_semantic_id

    evidence = _prepared_evidence_validates(fixture)
    reused = evidence is not None
    if evidence is None:
        save_charter(fixture["store_root"], fixture["charter"])
        save_or_reuse_envelope(
            fixture["store_root"], "pipeline_specs", fixture["semantic"]
        )
        run_pipeline(
            fixture["semantic"],
            fixture["charter"],
            store_root=fixture["store_root"],
            state_root=fixture["state_root"],
            wiring=fixture["wiring"],
            worker_policy=fixture["worker_policy"],
        )
        evidence = _prepared_evidence_validates(fixture)
        if evidence is None:
            raise RuntimeError(
                "the freshly built pipeline run does not validate — refusing "
                "to serve unverified smoke evidence"
            )
    regime = _persist_regime_runs(fixture["store_root"])

    manifest = {
        **key_material,
        "scratch_key": key,
        "scratch_root": str(scratch),
        "reused_validated_evidence": reused,
        "pipeline_result_id": evidence["pipeline_result_id"],
        "regime": regime,
        "prepared_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "streamlit_version": st.__version__,
    }
    (scratch / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**fixture, "manifest": manifest}


fixture = _prepared_run()

import ifvg_pipeline_tab as pipeline_tab  # noqa: E402

pipeline_tab.PIPELINE_STATE_ROOT = fixture["state_root"]
roots = {
    "store_root": fixture["store_root"],
    "state_root": fixture["state_root"],
    "draft_root": Path(fixture["manifest"]["scratch_root"]) / "drafts",
    "namespace": "verification",
}
roots["draft_root"].mkdir(parents=True, exist_ok=True)

manifest = fixture["manifest"]
st.title("R6 smoke — Regime Lane over a completed synthetic 16-stage run")
st.caption(
    f"commit `{manifest['commit'][:12]}` · tree "
    f"`{manifest['source_tree_digest'][:12]}` · pipeline "
    f"`{manifest['pipeline_semantic_id'][:12]}` · scratch key "
    f"`{manifest['scratch_key']}` · prepared {manifest['prepared_at_utc']}"
)
with st.expander("Smoke ids to paste into the Regime model card (exact-ID loads)"):
    for label, ids in manifest["regime"].items():
        st.write(f"**{label}** — gates_passed={ids['gates_passed']} {ids['gate_failures']}")
        st.code(ids["protocol_id"], language=None)
        if ids["assessment_id"]:
            st.code(ids["assessment_id"], language=None)
        if ids["first_fit_id"]:
            st.code(ids["first_fit_id"], language=None)
        if ids["decision_id"]:
            st.code(ids["decision_id"], language=None)
pipeline_tab.render_pipeline_run(st, roots=roots)
