"""Read-only data providers for the study workspace UI (R4).

CS §12 names the provider surface the UI consumes; the scripts stay thin
widget layers (FUX-MOD-001) and every read here is store/exact-ID-safe:

* run enumeration reads the MUTABLE job-state root and catalog event log —
  the immutable store roots are never listed;
* every artifact load goes through ``load_verified_envelope`` (manifest
  re-verification) by exact 64-hex id;
* absent artifacts return ``None`` — the UI renders the §31
  ``artifact_unavailable`` state, nothing is fabricated;
* the R2→R4 obligation is enforced here: cross-profile deltas are
  constructible ONLY through :func:`prepare_cross_profile_deltas`, which
  persists both sides' lineage-uniqueness reports first.

Errors surfaced to the UI are sanitized by the caller through
``ifvg_ui_common.sanitize_error``; providers raise typed errors and never
render.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .search.catalog import read_catalog_events, rebuild_catalog_index
from .search.charter import CostPolicy, SearchCharterEnvelope
from .search.lineage import (
    NativeLineageMap,
    derive_lineage_validity,
    persist_lineage_uniqueness,
)
from .search.orchestrator import (
    SearchFrontierEnvelope,
    _child_evaluation_envelope,
    _load_child_evaluation,
    read_search_state,
)
from .search.store import (
    has_envelope,
    load_sidecar_bytes,
    load_verified_envelope,
)
from .study.population_delta import (
    PopulationDeltaReport,
    build_lineage_population_delta,
)

__all__ = [
    "SearchRunSummary",
    "list_search_runs",
    "catalog_annotations",
    "list_catalogued_envelope_ids",
    "load_contract_summaries",
    "load_search_state",
    "load_charter",
    "load_frontier_for_state",
    "load_child_metrics",
    "costed_evaluation_id_for",
    "ACCOUNT_EVENTS_SIDECAR",
    "WALK_SUMMARY_SIDECAR",
    "load_account_simulation_events",
    "load_prop_vectors",
    "VerificationAuthorizationState",
    "verification_authorization_state",
    "PipelineRunSummary",
    "list_pipeline_runs",
    "load_comparison_results_for_search",
    "load_ladder_diagnostics",
    "mbp1_stage_evidence_defaults",
    "prepare_cross_profile_deltas",
    "CrossProfileDeltas",
]

_HEX64 = 64


@dataclass(frozen=True)
class SearchRunSummary:
    search_id: str
    phase: str
    display_name: str
    child_count: int
    archived: bool = False


def load_search_state(state_root: Path, search_id: str) -> dict[str, Any] | None:
    """The atomic ``search_state.json`` payload (None when absent)."""

    return read_search_state(Path(state_root), search_id)


def catalog_annotations(store_root: Path) -> Mapping[str, Mapping[str, Any]]:
    """artifact_id → mutable catalog annotations (display name, note, …)."""

    events, _torn = read_catalog_events(Path(store_root))
    index = rebuild_catalog_index(events)
    return index.get("entries", {})


def _display_name(entry: Mapping[str, Any]) -> str | None:
    """Normalize a catalog display-name payload (bare string or dict)."""

    payload = entry.get("display_name")
    if isinstance(payload, Mapping):
        payload = payload.get("display_name")
    return str(payload) if payload else None


def list_search_runs(
    state_root: Path, store_root: Path
) -> tuple[SearchRunSummary, ...]:
    """Every search the JOB root knows, newest state first.

    The job root is mutable operational state (listing it is fine); the
    immutable stores are never listed. Display names come from the mutable
    catalog; the technical ``search_id`` stays authoritative (FUX §5.5).
    """

    state_root = Path(state_root)
    if not state_root.exists():
        return ()
    annotations = catalog_annotations(store_root)
    summaries: list[tuple[float, SearchRunSummary]] = []
    for child in state_root.iterdir():
        if len(child.name) != _HEX64 or not child.is_dir():
            continue
        state = read_search_state(state_root, child.name)
        if state is None:
            continue
        entry = annotations.get(child.name, {})
        summaries.append(
            (
                (child / "search_state.json").stat().st_mtime,
                SearchRunSummary(
                    search_id=child.name,
                    phase=str(state.get("phase") or "unknown"),
                    display_name=_display_name(entry) or child.name[:12] + "…",
                    child_count=len(state.get("children") or ()),
                    archived=bool(entry.get("archived", False)),
                ),
            )
        )
    summaries.sort(key=lambda pair: pair[0], reverse=True)
    return tuple(summary for _mtime, summary in summaries)


def list_catalogued_envelope_ids(
    store_root: Path, store_name: str
) -> tuple[tuple[str, str], ...]:
    """(envelope_id, display_name) for every CATALOGUED artifact of a store.

    The catalog event log is the one mutable index (CS §8); the immutable
    store roots are never listed. Only ids that both appear in the catalog
    AND verify present in the store are returned — publishers register
    artifacts by appending a ``display_name`` event.
    """

    root = Path(store_root)
    entries = catalog_annotations(root)
    found: list[tuple[str, str]] = []
    for artifact_id, entry in sorted(entries.items()):
        if entry.get("archived"):
            continue
        if len(artifact_id) == _HEX64 and has_envelope(root, store_name, artifact_id):
            found.append(
                (artifact_id, _display_name(entry) or artifact_id[:12] + "…")
            )
    return tuple(found)


def load_charter(store_root: Path, search_id: str) -> SearchCharterEnvelope | None:
    if not has_envelope(store_root, "charters", search_id):
        return None
    return load_verified_envelope(
        Path(store_root), "charters", search_id, SearchCharterEnvelope
    )


def load_frontier_for_state(
    store_root: Path, state: Mapping[str, Any] | None
) -> SearchFrontierEnvelope | None:
    """The persisted frontier via the state file's exact pointer note."""

    if not state:
        return None
    frontier_id = (state.get("phase_notes") or {}).get("frontier_id")
    if not frontier_id or not has_envelope(store_root, "frontiers", frontier_id):
        return None
    return load_verified_envelope(
        Path(store_root), "frontiers", str(frontier_id), SearchFrontierEnvelope
    )


def costed_evaluation_id_for(core_replay_id: str, cost_policy: CostPolicy) -> str:
    """The deterministic costed-evaluation envelope id (§1.5 lookup key)."""

    return _child_evaluation_envelope(core_replay_id, cost_policy).costed_evaluation_id


def load_child_metrics(
    store_root: Path, core_replay_id: str, cost_policy: CostPolicy
):
    """The published study-independent ``StrategyMetrics`` (None when absent)."""

    envelope = _child_evaluation_envelope(core_replay_id, cost_policy)
    return _load_child_evaluation(Path(store_root), envelope.costed_evaluation_id)


def load_contract_summaries(store_root: Path) -> tuple[dict[str, Any], ...]:
    """§11 card summaries for every catalogued, store-verified firm contract.

    Only compiled ``PropFirmContractEnvelope`` artifacts registered in the
    catalog appear — presets and uncompiled sources are NOT contracts and
    are never rendered as such. ``launchable`` is a UI hint only: real
    studies additionally require ``first_party_verified`` at charter
    validation (the fail-closed layer, never this summary).
    """

    from alpha_lab.propsim.firm_contracts import (  # noqa: PLC0415
        PropFirmContractEnvelope,
    )

    root = Path(store_root)
    cards: list[dict[str, Any]] = []
    for contract_id, display_name in list_catalogued_envelope_ids(
        root, "prop_contracts"
    ):
        envelope = load_verified_envelope(
            root, "prop_contracts", contract_id, PropFirmContractEnvelope
        )
        payload = envelope.payload
        evaluation = payload.evaluation
        drawdown = getattr(evaluation, "drawdown", None)
        daily = getattr(evaluation, "daily_loss", None)
        limits = getattr(evaluation, "contract_limits", None)
        payout = payload.payout
        fees = payload.fees
        synthetic = payload.verification_status == "synthetic_fixture_verified"
        superseded = payload.verification_status == "superseded"
        # FUX §11: unverified/blocked contracts render NO launchable
        # checkbox. Only the two VERIFIED ladder endpoints are selectable —
        # the intermediates (evidence compiled / owner review pending) and
        # superseded contracts render their reason instead.
        launchable = payload.verification_status in (
            "synthetic_fixture_verified",
            "first_party_verified",
        )
        launch_block_reason = None
        if superseded:
            launch_block_reason = "contract is superseded"
        elif not launchable:
            launch_block_reason = (
                f"verification status is {payload.verification_status!r} — "
                "not verified; the owner review must complete before this "
                "contract is selectable"
            )
        cards.append(
            {
                "firm_contract_id": envelope.firm_contract_id,
                "display_name": display_name,
                "firm": payload.firm,
                "account_type": payload.account_type,
                "account_size_label": payload.account_size_label,
                "contract_version": payload.contract_version,
                "effective_date": payload.effective_date,
                "verification_status": payload.verification_status,
                "source_status": payload.source_status,
                "drawdown_rule": _summary(drawdown),
                "daily_rule": _summary(daily),
                "contract_limits": _summary(limits),
                "payout_rules": _summary(payout),
                "fees": _summary(fees),
                "post_payout": _summary(
                    getattr(payout, "post_payout_threshold_rule", None)
                ),
                "path_capabilities": ", ".join(
                    sorted(
                        {
                            capability
                            for requirement in payload.rule_path_requirements
                            for capability in getattr(
                                requirement, "required_path_capabilities", ()
                            )
                        }
                    )
                )
                or "closed-trade evidence",
                "contract_evidence_bundle_id": envelope.contract_evidence_bundle_id,
                "supersession": (
                    "superseded" if superseded else "not superseded"
                ),
                "synthetic": synthetic,
                "launchable": launchable,
                "launch_block_reason": launch_block_reason,
            }
        )
    return tuple(cards)


def _summary(value: Any) -> str:
    if value is None:
        return "—"
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")
        parts = [
            f"{key}={payload[key]}"
            for key in sorted(payload)
            if payload[key] not in (None, (), [], {})
        ]
        return "; ".join(parts[:6]) + ("; …" if len(parts) > 6 else "")
    return str(value)


#: Sidecar names for persisted account simulations (fixtures/R5 write them
#: through ``save_envelope_immutable(..., extra_files=...)``; the store's
#: manifest protocol hashes and re-verifies both on load).
ACCOUNT_EVENTS_SIDECAR = "account_events.json"
WALK_SUMMARY_SIDECAR = "walk_summary.json"


def load_account_simulation_events(
    store_root: Path, account_simulation_id: str
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]] | None:
    """(walk summary, ordered event envelopes) for one exact simulation id.

    Events are returned in ``event_ordinal`` order — the envelope total
    order, never chart-timestamp sorting (FUX §28). ``None`` when the
    artifact is absent; manifest verification failures raise. The envelope
    itself is load-VERIFIED first (manifest hash + identity), so the
    sidecars are only trusted behind a verified manifest.
    """

    from alpha_lab.propsim.simulation import (  # noqa: PLC0415
        AccountSimulationEnvelope,
    )

    root = Path(store_root)
    if not has_envelope(root, "account_simulations", account_simulation_id):
        return None
    load_verified_envelope(
        root, "account_simulations", account_simulation_id, AccountSimulationEnvelope
    )
    summary = json.loads(
        load_sidecar_bytes(
            root, "account_simulations", account_simulation_id, WALK_SUMMARY_SIDECAR
        )
    )
    events = json.loads(
        load_sidecar_bytes(
            root, "account_simulations", account_simulation_id, ACCOUNT_EVENTS_SIDECAR
        )
    )
    ordered = tuple(
        sorted(events, key=lambda event: int(event["event_ordinal"]))
    )
    return summary, ordered


def load_prop_vectors(
    store_root: Path,
) -> Mapping[str, Mapping[str, dict[str, Any]]]:
    """core_replay_id → {simulation label → walk summary} for every
    CATALOGUED account simulation.

    The ``walk_summary.json`` sidecar (written by fixtures now; by the R5
    pipeline's S13/S15 stages in production) carries the
    ``payout_reliability_vector`` mapping, the simulation mode, and any
    optional ``survival_curve`` / ``payout_samples`` blocks the results
    surfaces render. Absent artifacts simply do not appear — the UI shows
    its §31 states instead of fabricating rows.
    """

    from alpha_lab.propsim.simulation import (  # noqa: PLC0415
        AccountSimulationEnvelope,
    )

    root = Path(store_root)
    vectors: dict[str, dict[str, dict[str, Any]]] = {}
    for simulation_id, display_name in list_catalogued_envelope_ids(
        root, "account_simulations"
    ):
        envelope = load_verified_envelope(
            root, "account_simulations", simulation_id, AccountSimulationEnvelope
        )
        summary = json.loads(
            load_sidecar_bytes(
                root, "account_simulations", simulation_id, WALK_SUMMARY_SIDECAR
            )
        )
        summary["account_simulation_id"] = simulation_id
        summary.setdefault("simulation_mode", envelope.payload.simulation_mode)
        vectors.setdefault(envelope.payload.core_replay_id, {})[display_name] = (
            summary
        )
    return vectors


@dataclass(frozen=True)
class VerificationAuthorizationState:
    """WHETHER the exact owner-approved ``VerificationAuthorizationRef``
    exists (FUX §14.2) — derived from the repository, never hardcoded."""

    exists: bool
    detail: str
    verification_run_ids: tuple[str, ...] = ()


def verification_authorization_state(
    store_root: Path,
) -> VerificationAuthorizationState:
    """Scan the persisted verification runs for a real authorization ref.

    CS §6 binds the ref inside ``VerificationRunPayload`` before any source
    path is constructed, so a load-verified ``VerificationRunEnvelope`` is
    the repository's evidence that the owner-approved ref exists. With none
    persisted, the state is truthfully *missing* — blocking the real run,
    never code authoring or synthetic states.
    """

    from .search.verification import VerificationRunEnvelope  # noqa: PLC0415

    root = Path(store_root)
    found: list[str] = []
    for run_id, _display in list_catalogued_envelope_ids(
        root, "verification_runs"
    ):
        envelope = load_verified_envelope(
            root, "verification_runs", run_id, VerificationRunEnvelope
        )
        if envelope.payload.verification_authorization is not None:
            found.append(run_id)
    if found:
        return VerificationAuthorizationState(
            exists=True,
            detail=(
                "an owner-approved VerificationAuthorizationRef is bound "
                f"inside {len(found)} persisted verification run(s)"
            ),
            verification_run_ids=tuple(found),
        )
    return VerificationAuthorizationState(
        exists=False,
        detail=(
            "no persisted verification run carries an owner-approved "
            "VerificationAuthorizationRef (owner decisions 21/R-5 pending)"
        ),
    )


@dataclass(frozen=True)
class CrossProfileDeltas:
    """Population deltas whose lineage-uniqueness precondition is proven."""

    baseline_lineage_report_id: str
    challenger_lineage_report_id: str
    lineage_valid: bool
    lineage_invalid_reason: str | None
    reports: Mapping[str, PopulationDeltaReport]


def prepare_cross_profile_deltas(
    store_root: Path,
    *,
    baseline_map: NativeLineageMap,
    challenger_map: NativeLineageMap,
    changed_axis_keys: tuple[str, ...],
    entity_kinds: Sequence[str] = ("setup", "candidate", "decision", "trade"),
) -> CrossProfileDeltas:
    """The ONLY cross-profile delta constructor the UI may use.

    Persists both sides' lineage-uniqueness reports into the immutable
    ``lineage_reports`` store BEFORE any delta is built (the R2→R4
    obligation, DT §4.3 P1-E), derives lineage validity from the registered
    axis specs (never a hand-asserted boolean), and lets the delta layer
    disable non-comparable populations rather than fuzzing them.
    """

    baseline_envelope, _ = persist_lineage_uniqueness(
        store_root, baseline_map.uniqueness_report
    )
    challenger_envelope, _ = persist_lineage_uniqueness(
        store_root, challenger_map.uniqueness_report
    )
    lineage_valid, invalid_reason = derive_lineage_validity(changed_axis_keys)
    reports = {
        kind: build_lineage_population_delta(
            kind,
            baseline_map,
            challenger_map,
            lineage_valid=lineage_valid,
            lineage_invalid_reason=invalid_reason,
        )
        for kind in entity_kinds
    }
    return CrossProfileDeltas(
        baseline_lineage_report_id=baseline_envelope.lineage_report_id,
        challenger_lineage_report_id=challenger_envelope.lineage_report_id,
        lineage_valid=lineage_valid,
        lineage_invalid_reason=invalid_reason,
        reports=reports,
    )


@dataclass(frozen=True)
class PipelineRunSummary:
    """One pipeline run discovered from the MUTABLE job root (stores are
    exact-ID-only and never listed — same locator rule as search runs)."""

    pipeline_semantic_id: str
    run_scope: str
    current_stage: str | None
    attempt_count: int
    publication_state: str
    failed: bool


def list_pipeline_runs(state_root: Path) -> tuple[PipelineRunSummary, ...]:
    from .search.pipeline import read_pipeline_state  # noqa: PLC0415

    root = Path(state_root)
    if not root.exists():
        return ()
    summaries: list[tuple[float, PipelineRunSummary]] = []
    for entry in root.iterdir():
        if not entry.is_dir() or len(entry.name) != _HEX64:
            continue
        state = read_pipeline_state(root, entry.name)
        if state is None:
            continue
        stages = dict(state.get("stages") or {})
        failed = any(
            isinstance(row, dict) and row.get("status") == "failed"
            for row in stages.values()
        )
        publication = dict(state.get("publication") or {})
        state_file = entry / "pipeline_state.json"
        try:
            mtime = state_file.stat().st_mtime
        except OSError:
            mtime = 0.0
        summaries.append(
            (
                mtime,
                PipelineRunSummary(
                    pipeline_semantic_id=str(state.get("pipeline_semantic_id")),
                    run_scope=str(state.get("run_scope") or "unknown"),
                    current_stage=state.get("current_stage"),
                    attempt_count=len(state.get("attempts") or ()),
                    publication_state=str(publication.get("state") or "not_prepared"),
                    failed=failed,
                ),
            )
        )
    summaries.sort(key=lambda pair: pair[0], reverse=True)
    return tuple(summary for _mtime, summary in summaries)


def load_comparison_results_for_search(
    store_root: Path, state: Mapping[str, Any] | None
) -> tuple[Any, ...]:
    """Persisted ComparisonResult envelopes named by a pipeline state's S14
    stage outputs (DEV-R4-16: the UI consumes persisted contracts, exact-ID
    loads only — nothing is rebuilt at render time)."""

    from .study.comparison_contracts import ComparisonResultEnvelope  # noqa: PLC0415

    if not state:
        return ()
    stages = dict(state.get("stages") or {})
    s14 = dict(stages.get("14_build_frontier_and_insights") or {})
    results = []
    for artifact_id in s14.get("output_artifact_ids") or ():
        try:
            results.append(
                load_verified_envelope(
                    Path(store_root),
                    "search_results",
                    str(artifact_id),
                    ComparisonResultEnvelope,
                )
            )
        except Exception:  # noqa: BLE001 — frontier/insight ids share the list
            continue
    return tuple(results)


def load_ladder_diagnostics(
    store_root: Path, state: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """The persisted S10 supervised-ladder diagnostics for a pipeline run
    (manifest-verified stage-result sidecar; exact-ID access only)."""

    if not state:
        return None
    stages = dict(state.get("stages") or {})
    s10 = dict(stages.get("10_generate_predictions_and_diagnostics") or {})
    stage_result_id = s10.get("stage_result_id")
    if not stage_result_id or s10.get("status") not in ("completed", "reused"):
        return None
    try:
        raw = load_sidecar_bytes(
            Path(store_root),
            "pipeline_stage_results",
            str(stage_result_id),
            "supervised_ladder.json",
        )
    except Exception:  # noqa: BLE001 — absence is a rendered state, not a crash
        return None
    return json.loads(raw.decode("utf-8"))


def mbp1_stage_evidence_defaults(
    state_root: Path, store_root: Path, pipeline_id: str | None
) -> tuple[dict[str, str], str | None]:
    """Exact MBP-1 artifact ids from a run's persisted stage evidence (R5B).

    Reads the S05 stage result's manifest-verified sidecar for the
    ``__mbp1_evidence__`` block and the S09 outputs for the controlled-study
    id. Returns ``(defaults, note)``: absence of evidence yields empty
    defaults silently, but a VERIFICATION failure on persisted evidence
    (tampered sidecar/manifest) yields a note the UI must surface — an
    integrity failure never renders as a cosmetic blank (safety review S6).
    """

    from .search.pipeline import read_pipeline_state  # noqa: PLC0415
    from .search.store import SearchStoreError  # noqa: PLC0415

    defaults: dict[str, str] = {}
    if not pipeline_id:
        return defaults, None
    state = read_pipeline_state(Path(state_root), str(pipeline_id))
    if not state:
        return defaults, None
    s05 = dict((state.get("stages") or {}).get("05_materialize_feature_views") or {})
    result_id = s05.get("stage_result_id")
    if not result_id:
        return defaults, None
    try:
        sidecar = json.loads(
            load_sidecar_bytes(
                Path(store_root),
                "pipeline_stage_results",
                str(result_id),
                "bundle_feature_views.json",
            ).decode("utf-8")
        )
    except SearchStoreError:
        return {}, (
            "the selected run's persisted MBP-1 stage evidence failed store "
            "verification; enter exact artifact ids manually"
        )
    except Exception:  # noqa: BLE001 — absence renders as empty inputs
        return defaults, None
    evidence = dict(sidecar.get("__mbp1_evidence__") or {})
    if evidence.get("coverage_report_id"):
        defaults["coverage_report_id"] = str(evidence["coverage_report_id"])
    if evidence.get("feature_artifact_id"):
        defaults["feature_artifact_id"] = str(evidence["feature_artifact_id"])
    s09 = dict((state.get("stages") or {}).get("09_train_models") or {})
    outputs = list(s09.get("output_artifact_ids") or ())
    if evidence and outputs:
        defaults["controlled_study_id"] = str(outputs[0])
    return defaults, None


# ── R6.1: the regime study surfaces (Monitor expander + Regime Lane auto-fill) ─


def _stage_sidecar(
    store_root: Path, state: Mapping[str, Any] | None, stage_value: str, name: str
) -> dict[str, Any] | None:
    """One manifest-verified stage-result sidecar of a completed/reused stage,
    or None when the stage has no verified result yet (absence is a rendered
    state, never a crash)."""

    if not state:
        return None
    stages = dict(state.get("stages") or {})
    entry = dict(stages.get(stage_value) or {})
    stage_result_id = entry.get("stage_result_id")
    if not stage_result_id or entry.get("status") not in ("completed", "reused"):
        return None
    try:
        raw = load_sidecar_bytes(
            Path(store_root), "pipeline_stage_results", str(stage_result_id), name
        )
    except Exception:  # noqa: BLE001 — absence renders as a state
        return None
    return json.loads(raw.decode("utf-8"))


def load_regime_diagnostics(
    store_root: Path, state: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """The persisted S10 regime diagnostics (``regime_diagnostics.json``):
    request, frozen authority refs, exact ids, gates, decisions, sub-steps."""

    return _stage_sidecar(
        store_root, state, "10_generate_predictions_and_diagnostics", "regime_diagnostics.json"
    )


def load_regime_run_facts(
    store_root: Path, state: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """The persisted S09 regime run record (``regime_run.json``): the S09a
    facts and, for model-bearing runs, the S09b/S09c records."""

    return _stage_sidecar(store_root, state, "09_train_models", "regime_run.json")


def load_regime_report_index(
    store_root: Path, state: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """The persisted S14 stratified-report index (``regime_stratified_reports.json``):
    report ids by class, the recorded per-class refusals, and ``delivered_by``
    — the modeled classes delivered by S09c (class -> exact study id), which
    are NOT refusals."""

    return _stage_sidecar(
        store_root, state, "14_build_frontier_and_insights", "regime_stratified_reports.json"
    )


def regime_stage_evidence_defaults(
    state_root: Path, store_root: Path, pipeline_id: str | None
) -> tuple[dict[str, str], str | None]:
    """Exact regime artifact ids from a run's persisted stage evidence (R6.1).

    The Regime Lane copy of ``mbp1_stage_evidence_defaults``: reads the S10
    stage result's manifest-verified ``regime_diagnostics.json`` for the
    protocol / assessment / first fit / final decision ids and the S14
    ``regime_stratified_reports.json`` for the first stratified report id.
    Returns ``(defaults, note)``: absence yields empty defaults silently, but
    a VERIFICATION failure on persisted evidence yields a note the UI must
    surface — an integrity failure never renders as a cosmetic blank.
    """

    from .search.pipeline import read_pipeline_state  # noqa: PLC0415
    from .search.store import SearchStoreError  # noqa: PLC0415

    defaults: dict[str, str] = {}
    if not pipeline_id:
        return defaults, None
    state = read_pipeline_state(Path(state_root), str(pipeline_id))
    if not state:
        return defaults, None
    stages = dict(state.get("stages") or {})
    s10 = dict(stages.get("10_generate_predictions_and_diagnostics") or {})
    result_id = s10.get("stage_result_id")
    if not result_id:
        return defaults, None
    try:
        diagnostics = json.loads(
            load_sidecar_bytes(
                Path(store_root),
                "pipeline_stage_results",
                str(result_id),
                "regime_diagnostics.json",
            ).decode("utf-8")
        )
    except SearchStoreError:
        return {}, (
            "the selected run's persisted regime stage evidence failed store "
            "verification; enter exact artifact ids manually"
        )
    except Exception:  # noqa: BLE001 — absence renders as empty inputs
        return defaults, None
    if diagnostics.get("resolved_regime_protocol_id"):
        defaults["protocol_id"] = str(diagnostics["resolved_regime_protocol_id"])
    if diagnostics.get("regime_capability_assessment_id"):
        defaults["assessment_id"] = str(diagnostics["regime_capability_assessment_id"])
    fit_ids = list(diagnostics.get("regime_fit_ids") or ())
    if fit_ids:
        defaults["fit_id"] = str(fit_ids[0])
    decisions = list(diagnostics.get("decisions") or ())
    if decisions:
        defaults["decision_id"] = str(decisions[-1].get("regime_promotion_decision_id") or "")
    s14 = dict(stages.get("14_build_frontier_and_insights") or {})
    s14_result = s14.get("stage_result_id")
    if s14_result:
        try:
            index = json.loads(
                load_sidecar_bytes(
                    Path(store_root),
                    "pipeline_stage_results",
                    str(s14_result),
                    "regime_stratified_reports.json",
                ).decode("utf-8")
            )
        except SearchStoreError:
            return {}, (
                "the selected run's persisted stratified-report evidence failed "
                "store verification; enter exact artifact ids manually"
            )
        except Exception:  # noqa: BLE001 — a run without S14 reports
            index = {}
        report_ids = list(index.get("report_ids") or ())
        if report_ids:
            defaults["report_id"] = str(report_ids[0])
    return {key: value for key, value in defaults.items() if value}, None
