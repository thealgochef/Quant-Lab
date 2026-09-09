"""Live browser-smoke harness for the R6.1 regime surfaces (plan §6.K).

Shell-only launcher (never part of the repo's importable surface): builds
THREE completed synthetic 16-stage pipeline runs under a CONTENT-ADDRESSED
scratch directory — the descriptive CANDIDATE-grain regime study, the
descriptive 5m PANEL-grain regime study, and the model-bearing CANDIDATE
study frozen to a synthetic owner decision + FEATURE_ELIGIBLE promotion
taken over the descriptive run's own assessment (the two-pass workflow) —
and serves the real ``render_pipeline_run`` over one of them so a browser
can walk the Configure / Preview / Monitor regime surfaces and the
auto-filled Regime Lane (registry, model card with per-fold stability and
grain-specific transitions, assignment / stratification views, the five
stratified-report classes, the owner-decision view, promotion refusals, the
panel-grain card, the MBP-1 coverage-evidence view, the CatBoost ladder
rows). Every regime artifact is produced BY the pipeline (no post-hoc
persistence). The scratch key covers the git commit + a source-tree digest of
the exercised surfaces + the three ``pipeline_semantic_id``s; stale keys are
cleared; reuse requires the persisted evidence to VALIDATE.

Run from the repo root (the server log must capture stdout+stderr):

    python -m streamlit run \
        QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R6.1/r61_smoke_app.py \
        --server.port 8611 --server.headless true --logger.level=info

Select the served run with ``?run=candidate`` (default), ``?run=panel``, or
``?run=supervised``. Everything is written under %TEMP%/ifvg_r61_smoke/<key>;
the repo's data/ tree is untouched. No launch, promote, rank, or retrain
control exists on the served surface. Every prepared phase and served view
logs a ``[r61-smoke]`` marker line (the manifest validator asserts them).
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

st.set_page_config(page_title="R6.1 regime smoke", layout="wide")

_SCRATCH_BASE = Path(tempfile.gettempdir()) / "ifvg_r61_smoke"
_MARKER = "[r61-smoke]"

_DIGEST_GLOBS = (
    ("src/alpha_lab/agents/data_infra/ifvg", "**/*.py"),
    ("src/alpha_lab/propsim", "**/*.py"),
    ("scripts", "ifvg_*.py"),
    ("tests/agents/ifvg_search", "*.py"),
    ("tests/agents/data_infra/ifvg", "test_regime_*.py"),
    ("tests/agents/data_infra/ifvg/ml_fixtures", "*.py"),
    ("tests/agents", "test_ifvg_pipeline_tab.py"),
)


def _log(message: str) -> None:
    print(f"{_MARKER} {message}", flush=True)
    print(f"{_MARKER} {message}", file=sys.stderr, flush=True)


def _git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _source_tree_digest() -> str:
    entries: list[tuple[str, str]] = []
    for base, pattern in _DIGEST_GLOBS:
        root = _REPO / base
        for path in sorted(root.glob(pattern)):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            entries.append(
                (path.relative_to(_REPO).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
            )
    entries.append(("r61_smoke_app.py", hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()


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
    if not planned or any((stages[name] or {}).get("status") not in terminal for name in planned):
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


def _sidecar(fixture: dict, state: dict, stage_value: str, name: str) -> dict:
    from alpha_lab.agents.data_infra.ifvg.search.store import load_sidecar_bytes

    entry = state["stages"][stage_value]
    return json.loads(
        load_sidecar_bytes(
            fixture["store_root"], "pipeline_stage_results", entry["stage_result_id"], name
        )
    )


def _run_fixture(fixture: dict, label: str) -> dict:
    from alpha_lab.agents.data_infra.ifvg.search.charter import save_charter
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import run_pipeline
    from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope

    evidence = _prepared_evidence_validates(fixture)
    reused = evidence is not None
    if evidence is None:
        _log(f"phase=run label={label} action=run_pipeline")
        save_charter(fixture["store_root"], fixture["charter"])
        save_or_reuse_envelope(fixture["store_root"], "pipeline_specs", fixture["semantic"])
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
                f"the freshly built {label} pipeline run does not validate — refusing to "
                "serve unverified smoke evidence"
            )
    else:
        _log(f"phase=run label={label} action=reuse_validated_evidence")
    state = evidence["state"]
    diagnostics = _sidecar(
        fixture, state, "10_generate_predictions_and_diagnostics", "regime_diagnostics.json"
    )
    run = _sidecar(fixture, state, "09_train_models", "regime_run.json")
    reports = _sidecar(
        fixture, state, "14_build_frontier_and_insights", "regime_stratified_reports.json"
    )
    return {
        "reused_validated_evidence": reused,
        "pipeline_semantic_id": fixture["semantic"].pipeline_semantic_id,
        "pipeline_result_id": evidence["pipeline_result_id"],
        "protocol_id": diagnostics["resolved_regime_protocol_id"],
        "assessment_id": diagnostics["regime_capability_assessment_id"],
        "first_fit_id": (diagnostics["regime_fit_ids"] or [""])[0],
        "oos_assignment_id": diagnostics["regime_oos_assignment_id"],
        "decision_id": diagnostics["decisions"][-1]["regime_promotion_decision_id"],
        "final_status": diagnostics["final_status"],
        "authority_source": diagnostics["authority_source"],
        "gates_passed": str(diagnostics["gates"]["gates_passed"]),
        "gate_failures": ", ".join(diagnostics["gates"]["gate_failures"]),
        "fold_feature_artifact_id": run.get("S09b", {}).get("regime_fold_feature_artifact_id", ""),
        "controlled_study_id": run.get("S09c", {}).get("regime_controlled_study_id", ""),
        "cohort_model_study_id": run.get("S09c", {}).get("regime_cohort_model_study_id", ""),
        "stratified_report_ids": dict(reports.get("reports_by_class", {})),
        "report_refusals": dict(reports.get("refusals", {})),
    }


def _frozen_authority(fixture: dict, facts: dict) -> tuple[str, str, str]:
    """The synthetic owner decision + FEATURE_ELIGIBLE promotion over the
    descriptive run's OWN assessment (the two-pass workflow; synthetic scope)."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
        RegimePromotionDecision,
        RegimePromotionDecisionEnvelope,
        RegimeRole,
        RegimeStatus,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
        load_regime_assessment,
        load_regime_promotion,
        load_regime_protocol,
        persist_regime_promotion,
    )
    from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (
        synthetic_owner_decision_fixture,
    )

    root = fixture["store_root"]
    protocol = load_regime_protocol(root, facts["protocol_id"])
    assessment = load_regime_assessment(root, facts["assessment_id"])
    ready = load_regime_promotion(root, facts["decision_id"])
    owner = synthetic_owner_decision_fixture(
        root,
        protocol=protocol,
        assessment=assessment,
        approved_at="2026-08-26T00:00:00+00:00",
        effective_from="2026-08-26T00:00:00+00:00",
    )
    eligible = RegimePromotionDecisionEnvelope.from_payload(
        RegimePromotionDecision(
            resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
            role=RegimeRole.FEATURE_GENERATOR,
            status=RegimeStatus.FEATURE_ELIGIBLE,
            previous_status=RegimeStatus.STRATIFICATION_READY,
            previous_decision_ref=ready.regime_promotion_decision_id,
            capability_assessment_ref=assessment.regime_capability_assessment_id,
            owner_ratification_ref=owner.owner_decision_artifact_id,
            decided_at="2026-08-28T13:00:00+00:00",
        )
    )
    persist_regime_promotion(root, eligible, run_scope="synthetic_fixture")
    _log("phase=authority action=synthetic_owner_decision_and_feature_eligible_persisted")
    return (
        eligible.regime_promotion_decision_id,
        owner.owner_decision_artifact_id,
        assessment.regime_capability_assessment_id,
    )


@st.cache_resource
def _prepared_runs() -> dict:
    from tests.agents.ifvg_search.pipeline_fixture import build_pipeline_fixture

    commit = _git_commit()
    tree_digest = _source_tree_digest()
    probe_root = _SCRATCH_BASE / "_probe"
    probe_ids = {
        shape: build_pipeline_fixture(probe_root / shape, regime_study=shape)[
            "semantic"
        ].pipeline_semantic_id
        for shape in ("candidate", "panel")
    }
    key_material = {"commit": commit, "source_tree_digest": tree_digest, **probe_ids}
    key = hashlib.sha256(json.dumps(key_material, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    _SCRATCH_BASE.mkdir(parents=True, exist_ok=True)
    for stale in _SCRATCH_BASE.iterdir():
        if stale.name != key:
            shutil.rmtree(stale, ignore_errors=True)
    scratch = _SCRATCH_BASE / key
    scratch.mkdir(parents=True, exist_ok=True)
    _log(f"phase=prepare commit={commit[:12]} tree={tree_digest[:12]} key={key}")

    fixtures: dict[str, dict] = {}
    facts: dict[str, dict] = {}
    candidate = build_pipeline_fixture(scratch / "candidate", regime_study="candidate")
    assert candidate["semantic"].pipeline_semantic_id == probe_ids["candidate"]
    fixtures["candidate"] = candidate
    facts["candidate"] = _run_fixture(candidate, "candidate")
    panel = build_pipeline_fixture(scratch / "panel", regime_study="panel")
    assert panel["semantic"].pipeline_semantic_id == probe_ids["panel"]
    fixtures["panel"] = panel
    facts["panel"] = _run_fixture(panel, "panel")
    authority = _frozen_authority(candidate, facts["candidate"])
    supervised = build_pipeline_fixture(
        scratch / "supervised", regime_study="candidate_supervised", regime_authority=authority
    )
    supervised["store_root"] = candidate["store_root"]  # the frozen authority lives here
    fixtures["supervised"] = supervised
    facts["supervised"] = _run_fixture(supervised, "supervised")
    facts["supervised"]["frozen_authority"] = {
        "regime_promotion_decision_id": authority[0],
        "owner_decision_artifact_id": authority[1],
        "required_capability_assessment_id": authority[2],
    }
    manifest = {
        "commit": commit,
        "source_tree_digest": tree_digest,
        "scratch_key": key,
        "scratch_root": str(scratch),
        "runs": facts,
        "prepared_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "streamlit_version": st.__version__,
    }
    (scratch / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _log("phase=prepare action=smoke_manifest_written")
    return {"fixtures": fixtures, "manifest": manifest}


prepared = _prepared_runs()
selected = st.query_params.get("run", "candidate")
if selected not in prepared["fixtures"]:
    selected = "candidate"
fixture = prepared["fixtures"][selected]
manifest = prepared["manifest"]

import ifvg_pipeline_tab as pipeline_tab  # noqa: E402

pipeline_tab.PIPELINE_STATE_ROOT = fixture["state_root"]
roots = {
    "store_root": fixture["store_root"],
    "state_root": fixture["state_root"],
    "draft_root": Path(manifest["scratch_root"]) / "drafts" / selected,
    "namespace": "verification",
}
roots["draft_root"].mkdir(parents=True, exist_ok=True)
facts = manifest["runs"][selected]
_log(f"phase=serve run={selected} pipeline={facts['pipeline_semantic_id'][:12]}")

st.title(f"R6.1 smoke — regime study ({selected}) over a completed synthetic 16-stage run")
st.caption(
    f"commit `{manifest['commit'][:12]}` · tree `{manifest['source_tree_digest'][:12]}` · "
    f"pipeline `{facts['pipeline_semantic_id'][:12]}` · status `{facts['final_status']}` "
    f"({facts['authority_source']}) · scratch key `{manifest['scratch_key']}` · prepared "
    f"{manifest['prepared_at_utc']} · switch runs with ?run=candidate|panel|supervised"
)
with st.expander("Smoke ids (exact-ID loads; auto-filled below)"):
    for key_name in (
        "pipeline_semantic_id",
        "protocol_id",
        "assessment_id",
        "first_fit_id",
        "oos_assignment_id",
        "decision_id",
        "fold_feature_artifact_id",
        "controlled_study_id",
        "cohort_model_study_id",
    ):
        if facts.get(key_name):
            st.write(f"**{key_name}**")
            st.code(facts[key_name], language=None)
    for class_name, ids in facts["stratified_report_ids"].items():
        st.write(f"**stratified report — {class_name}**")
        for report_id in ids:
            st.code(report_id, language=None)
    if facts["report_refusals"]:
        st.write("**report refusals (typed, never raised)**")
        st.json(facts["report_refusals"])
pipeline_tab.render_pipeline_run(st, roots=roots)
