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
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    "ArtifactScope",
    "artifact_scope_for_charter",
    "AuthorizationReadiness",
    "resolve_store_namespace",
    "verification_authorization_readiness",
    "verification_owner_bundle",
    "verification_bundle_from_signed_ref",
    "owner_authorization_readiness",
    # UI-2 — the Verification Center read models
    "SEED_AUTHORIZATION_RECEIPT",
    "SEED_RUN_RECEIPT",
    "CENTER_RECORD_FILE",
    "ShortlistState",
    "load_window_shortlist",
    "shortlist_window",
    "coverage_matrix_from_shortlist_window",
    "InventoryState",
    "load_source_inventory",
    "VerificationCenterRecord",
    "load_verification_center_record",
    "save_verification_center_record",
    "pick_up_seed_receipts",
    "SeedAuthorizationState",
    "seed_authorization_state",
    "SeedSnapshotState",
    "seed_snapshot_state",
    "SeedReceiptState",
    "seed_receipt_state",
    "SignedRefState",
    "load_signed_verification_ref",
    "register_verification_run",
    "PreflightState",
    "bounded_preflight_state",
    "verification_stage_rows",
    "verification_evidence_summary",
    "owner_authorization_bundle_from_store",
    "locate_charter_store",
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
    #: UI-1: the run's OWN store (located by its exact charter id across the
    #: known roots), the store's verified namespace class (``None`` for an
    #: unmarked or unreadable store — never a path guess) and the artifact's
    #: evidence scope (a synthetic-marker or verification charter is
    #: verification-only). The badge keys on the artifact, never a radio.
    store_root: str | None = None
    namespace_class: str | None = None
    verification_only: bool | None = None


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


def locate_charter_store(
    charter_id: str, store_roots: Sequence[Path]
) -> Path | None:
    """The FIRST known store holding the exact charter id (exact-id probe;
    the stores are never listed), or ``None``."""

    for root in store_roots:
        try:
            if has_envelope(Path(root), "charters", str(charter_id)):
                return Path(root)
        except (ValueError, OSError):
            continue
    return None


def _namespace_class_or_none(store_root: Path) -> str | None:
    """``research`` / ``test`` for a marked store; ``None`` when unmarked or
    unreadable (a corrupt envelope is reported by the namespace state, never
    guessed here)."""

    from .search.store_namespace import StoreNamespaceError, namespace_class_of  # noqa: PLC0415

    try:
        return namespace_class_of(Path(store_root))
    except (StoreNamespaceError, OSError):
        return None


def _verification_only_for(store_root: Path, charter_id: str) -> bool | None:
    try:
        charter = load_charter(Path(store_root), charter_id)
    except Exception:  # noqa: BLE001 — a corrupt charter is reported by the loader
        return None
    if charter is None:
        return None
    return artifact_scope_for_charter(charter).verification_only


def list_search_runs(
    state_root: Path,
    store_root: Path,
    *,
    store_roots: Sequence[Path] | None = None,
) -> tuple[SearchRunSummary, ...]:
    """Every search the JOB root knows, newest state first.

    The job root is mutable operational state (listing it is fine); the
    immutable stores are never listed. Display names come from the mutable
    catalog; the technical ``search_id`` stays authoritative (FUX §5.5).
    UI-1: each run is annotated with the store that holds its exact charter
    (``store_root`` first, then every extra ``store_roots`` entry), that
    store's verified namespace class and the artifact's evidence scope — the
    verification badge and the read location derive from the artifact.
    """

    state_root = Path(state_root)
    if not state_root.exists():
        return ()
    roots: list[Path] = [Path(store_root)]
    for extra in store_roots or ():
        if Path(extra) not in roots:
            roots.append(Path(extra))
    annotations_by_root = {root: catalog_annotations(root) for root in roots}
    summaries: list[tuple[float, SearchRunSummary]] = []
    for child in state_root.iterdir():
        if len(child.name) != _HEX64 or not child.is_dir():
            continue
        state = read_search_state(state_root, child.name)
        if state is None:
            continue
        located = locate_charter_store(child.name, roots)
        annotation_root = located if located is not None else roots[0]
        entry = annotations_by_root[annotation_root].get(child.name, {})
        summaries.append(
            (
                (child / "search_state.json").stat().st_mtime,
                SearchRunSummary(
                    search_id=child.name,
                    phase=str(state.get("phase") or "unknown"),
                    display_name=_display_name(entry) or child.name[:12] + "…",
                    child_count=len(state.get("children") or ()),
                    archived=bool(entry.get("archived", False)),
                    store_root=str(located) if located is not None else None,
                    namespace_class=(
                        _namespace_class_or_none(located) if located is not None else None
                    ),
                    verification_only=(
                        _verification_only_for(located, child.name)
                        if located is not None
                        else None
                    ),
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
    #: UI-1: the store holding the run's exact charter, its verified
    #: namespace class and the charter's evidence scope (see SearchRunSummary)
    search_charter_id: str | None = None
    store_root: str | None = None
    namespace_class: str | None = None
    verification_only: bool | None = None


def list_pipeline_runs(
    state_root: Path, *, store_roots: Sequence[Path] | None = None
) -> tuple[PipelineRunSummary, ...]:
    from .search.pipeline import read_pipeline_state  # noqa: PLC0415

    root = Path(state_root)
    if not root.exists():
        return ()
    roots = [Path(item) for item in (store_roots or ())]
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
        charter_id = state.get("search_charter_id")
        located = (
            locate_charter_store(str(charter_id), roots) if charter_id and roots else None
        )
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
                    search_charter_id=str(charter_id) if charter_id else None,
                    store_root=str(located) if located is not None else None,
                    namespace_class=(
                        _namespace_class_or_none(located) if located is not None else None
                    ),
                    verification_only=(
                        _verification_only_for(located, str(charter_id))
                        if located is not None
                        else None
                    ),
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


# ─────────────────────────────────────────────────────────────────────────────
# UI-1 — verified namespace, artifact scope and TYPED authorization readiness
# (plan §8 "study_providers": read-only adapters over the backend contracts;
# no duplicate authorization, namespace or date logic lives here)
# ─────────────────────────────────────────────────────────────────────────────


def resolve_store_namespace(store_root: Path, *, expected_class: str | None = None):
    """The store's SEMANTIC namespace state from its verified envelope
    (``verified`` / ``unmarked`` / ``corrupt`` / ``deployment_incoherent`` /
    ``class_mismatch``); a local path never defines authority."""

    from .presentation.run_purpose import namespace_state_for_store  # noqa: PLC0415

    return namespace_state_for_store(Path(store_root), expected_class=expected_class)


@dataclass(frozen=True)
class ArtifactScope:
    """What a frozen charter IS, derived from the artifact itself (never from
    a session selector): its evidence class, the run scope its date policy
    and authorization imply, and whether it is verification-only."""

    evidence_class: str  # synthetic_fixture | real
    run_scope: str  # synthetic_fixture | verification_5d | full_authorized_development
    verification_only: bool
    label: str


def artifact_scope_for_charter(charter_envelope) -> ArtifactScope:
    from .search.authorization import SyntheticAuthorizationMarker  # noqa: PLC0415

    payload = charter_envelope.payload
    if isinstance(payload.owner_authorization, SyntheticAuthorizationMarker):
        return ArtifactScope(
            evidence_class="synthetic_fixture",
            run_scope="synthetic_fixture",
            verification_only=True,
            label="Synthetic fixture — verification only, never research evidence",
        )
    if payload.date_policy.access_policy_id == "verification_fixed_allowlist_max5_v1":
        return ArtifactScope(
            evidence_class="real",
            run_scope="verification_5d",
            verification_only=True,
            label="Real ≤5-day verification slice — verification only",
        )
    return ArtifactScope(
        evidence_class="real",
        run_scope="full_authorized_development",
        verification_only=False,
        label="Owner-authorized development data",
    )


@dataclass(frozen=True)
class AuthorizationReadiness:
    """The TYPED readiness of the real authorization a computation path
    requires (plan §9 Phase 1). ``status`` is one of
    ``presentation.run_purpose.AUTHORIZATION_READINESS_STATUSES``; the UI
    renders every one of them and never collapses them into "present"."""

    authorization_class: str  # verification_authorization_ref | owner_authorization_bundle
    status: str
    detail: str
    evidence_ids: tuple[str, ...] = ()
    missing_decision_keys: tuple[str, ...] = ()
    store_namespace_id: str | None = None


_HEAD_REASON_TO_STATUS: Mapping[str, str] = {
    "supersession_head_witness_mismatch": "stale_head",
    "supersession_head_shorter_than_witness": "wrong_head",
    "supersession_chain_broken": "wrong_head",
    "supersession_record_unverifiable": "wrong_head",
    "supersession_decision_unverifiable": "superseded",
    "supersession_transition_unlawful": "superseded",
    "supersession_chain_divergent": "superseded",
    "store_namespace_identity_mismatch": "wrong_namespace",
    "store_namespace_class_mismatch": "wrong_namespace",
    "store_namespace_deployment_incoherent": "store_incoherent",
    "store_namespace_missing": "store_unmarked",
}


def _namespace_readiness_block(
    store_root: Path, *, authorization_class: str, expected_class: str | None
) -> AuthorizationReadiness | None:
    namespace = resolve_store_namespace(store_root, expected_class=expected_class)
    if namespace.status == "verified":
        return None
    status = {
        "unmarked": "store_unmarked",
        "corrupt": "store_corrupt",
        "deployment_incoherent": "store_incoherent",
        "class_mismatch": "wrong_namespace",
    }.get(namespace.status, "unavailable")
    return AuthorizationReadiness(
        authorization_class=authorization_class,
        status=status,
        detail=namespace.detail,
        store_namespace_id=namespace.store_namespace_id,
    )


def verification_authorization_readiness(
    store_root: Path,
    *,
    baseline_profile_name: str | None = None,
    baseline_section_config_hash: str | None = None,
    allowlist: Sequence[str] | None = None,
) -> AuthorizationReadiness:
    """Typed readiness of the owner's ``VerificationAuthorizationRef`` for the
    real ≤5-day slice: the store must be a verified, coherently deployed
    ``test`` namespace; a persisted verification run must carry a ref bound to
    THIS namespace whose signed head witness is CURRENT (the complete owner
    authority chain proof); and, when given, the run's baseline profile /
    section hash and allowlist must match the draft. Every failure keeps its
    typed reason; nothing here asserts readiness."""

    from .data_access import allowlist_sha256  # noqa: PLC0415
    from .search.authorization import (  # noqa: PLC0415
        AuthorizationError,
        assert_authorization_bound_to_store,
    )
    from .search.verification import VerificationRunEnvelope  # noqa: PLC0415

    root = Path(store_root)
    authorization_class = "verification_authorization_ref"
    try:
        blocked = _namespace_readiness_block(
            root, authorization_class=authorization_class, expected_class="test"
        )
        if blocked is not None:
            return blocked
        namespace = resolve_store_namespace(root, expected_class="test")
        candidates = list_catalogued_envelope_ids(root, "verification_runs")
    except Exception as error:  # noqa: BLE001 — the lookup itself failed (typed, sanitized later)
        return AuthorizationReadiness(
            authorization_class=authorization_class,
            status="unavailable",
            detail=f"the verification-run lookup failed: {type(error).__name__}",
        )
    if not candidates:
        return AuthorizationReadiness(
            authorization_class=authorization_class,
            status="missing",
            detail=(
                "no persisted verification run carries an owner-approved "
                "VerificationAuthorizationRef bound to this store (owner decisions 21/R-5)"
            ),
            store_namespace_id=namespace.store_namespace_id,
        )
    failures: list[AuthorizationReadiness] = []
    ready_ids: list[str] = []
    for run_id, _display in candidates:
        try:
            envelope = load_verified_envelope(
                root, "verification_runs", run_id, VerificationRunEnvelope
            )
        except Exception as error:  # noqa: BLE001 — a corrupt entry is reported, not trusted
            failures.append(
                AuthorizationReadiness(
                    authorization_class,
                    "store_corrupt",
                    f"verification run {run_id[:12]}… failed verification "
                    f"({type(error).__name__})",
                    (run_id,),
                    store_namespace_id=namespace.store_namespace_id,
                )
            )
            continue
        payload = envelope.payload
        ref = payload.verification_authorization
        if ref.store_namespace_id != namespace.store_namespace_id:
            failures.append(
                AuthorizationReadiness(
                    authorization_class,
                    "wrong_namespace",
                    f"verification run {run_id[:12]}… authorizes another store namespace "
                    f"({ref.store_namespace_id[:12]}… ≠ {namespace.store_namespace_id[:12]}…)",
                    (run_id,),
                    store_namespace_id=namespace.store_namespace_id,
                )
            )
            continue
        try:
            assert_authorization_bound_to_store(
                root,
                store_namespace_id=ref.store_namespace_id,
                supersession_head_witness=ref.supersession_head_witness,
                expected_namespace_class="test",
            )
        except AuthorizationError as error:
            reason = str(getattr(error, "reason", "") or "")
            failures.append(
                AuthorizationReadiness(
                    authorization_class,
                    _HEAD_REASON_TO_STATUS.get(reason, "wrong_head"),
                    f"verification run {run_id[:12]}…: {error}",
                    (run_id,),
                    store_namespace_id=namespace.store_namespace_id,
                )
            )
            continue
        if baseline_profile_name is not None and (
            payload.baseline_profile_id != baseline_profile_name
            or (
                baseline_section_config_hash is not None
                and payload.baseline_section_config_hash != baseline_section_config_hash
            )
        ):
            failures.append(
                AuthorizationReadiness(
                    authorization_class,
                    "wrong_profile",
                    f"verification run {run_id[:12]}… authorizes baseline "
                    f"{payload.baseline_profile_id!r}, not {baseline_profile_name!r} "
                    "(or a different resolved section hash)",
                    (run_id,),
                    store_namespace_id=namespace.store_namespace_id,
                )
            )
            continue
        expected_hash = allowlist_sha256(tuple(payload.allowlist))
        if (
            payload.allowlist_hash != expected_hash
            or ref.approved_allowlist_hash != payload.allowlist_hash
            or (allowlist is not None and tuple(payload.allowlist) != tuple(allowlist))
        ):
            failures.append(
                AuthorizationReadiness(
                    authorization_class,
                    "wrong_source",
                    f"verification run {run_id[:12]}… authorizes the allowlist "
                    f"{list(payload.allowlist)}, which does not match the draft's dates "
                    "(one canonical allowlist; never rotated)",
                    (run_id,),
                    store_namespace_id=namespace.store_namespace_id,
                )
            )
            continue
        ready_ids.append(run_id)
    if ready_ids:
        return AuthorizationReadiness(
            authorization_class,
            "ready",
            f"{len(ready_ids)} persisted verification run(s) carry an owner-approved "
            "VerificationAuthorizationRef bound to this store's verified namespace and "
            "current supersession head",
            tuple(ready_ids),
            store_namespace_id=namespace.store_namespace_id,
        )
    first = failures[0]
    if len(failures) > 1:
        first = AuthorizationReadiness(
            first.authorization_class,
            first.status,
            first.detail + f" (+{len(failures) - 1} other run(s) refused)",
            tuple(item for failure in failures for item in failure.evidence_ids),
            store_namespace_id=first.store_namespace_id,
        )
    return first


def verification_bundle_from_signed_ref(ref, requirement_set):
    """UI-2: the computation-path-scoped ``OwnerAuthorizationBundle`` of a REAL
    verification charter, derived from the owner's SIGNED
    ``VerificationAuthorizationRef`` ALONE. The 21/R-5 evidence ref names the
    signed reference's own content hash as the decision artifact (the ref IS
    the owner's artifact) — never a run-envelope id: the persisted run names
    the frozen pipeline spec, which derives from the charter, which carries
    this bundle, so a bundle bound to the run id would be circular."""

    from .search.authorization import (  # noqa: PLC0415
        VERIFICATION_FIXTURE_DECISION_KEY,
        OwnerAuthorizationBundle,
        OwnerDecisionEvidenceRef,
    )

    evidence = OwnerDecisionEvidenceRef(
        decision_id=VERIFICATION_FIXTURE_DECISION_KEY,
        decision_artifact_id=ref.content_hash,
        content_hash=ref.content_hash,
        author=ref.approved_by,
        approved_at=ref.approved_at,
        effective_from=ref.approved_at,
        reviewed_evidence_refs=(ref.coverage_matrix_artifact_id, ref.seed_snapshot_id),
    )
    return OwnerAuthorizationBundle(
        requirement_set_id=requirement_set.requirement_set_id,
        decision_refs={VERIFICATION_FIXTURE_DECISION_KEY: evidence},
        store_namespace_id=ref.store_namespace_id,
        supersession_head_witness=ref.supersession_head_witness,
    )


def verification_owner_bundle(
    store_root: Path, readiness: AuthorizationReadiness, requirement_set
):
    """The bundle a REAL verification charter carries, assembled ONLY from a
    ``ready`` readiness over the persisted, verified ``VerificationAuthorizationRef``
    (UI-2: derived from the signed ref itself — run-independent). ``None``
    unless ready."""

    from .search.verification import VerificationRunEnvelope  # noqa: PLC0415

    if readiness.status != "ready" or not readiness.evidence_ids:
        return None
    run_id = readiness.evidence_ids[0]
    envelope = load_verified_envelope(
        Path(store_root), "verification_runs", run_id, VerificationRunEnvelope
    )
    return verification_bundle_from_signed_ref(
        envelope.payload.verification_authorization, requirement_set
    )


def _owner_decision_evidence(store_root: Path, namespace_id: str):
    """decision key → (artifact id, evidence ref) for every CATALOGUED,
    verified owner decision artifact of THIS namespace (exact-id loads)."""

    from .search.owner_decisions import OwnerDecisionArtifactEnvelope  # noqa: PLC0415

    found: dict[str, tuple[str, Any]] = {}
    for artifact_id, _display in list_catalogued_envelope_ids(
        Path(store_root), "owner_decisions"
    ):
        try:
            envelope = load_verified_envelope(
                Path(store_root), "owner_decisions", artifact_id, OwnerDecisionArtifactEnvelope
            )
        except Exception:  # noqa: BLE001 — a corrupt artifact is never evidence
            continue
        if envelope.payload.store_namespace_id != namespace_id:
            continue
        for key in (envelope.payload.decision_id, *envelope.payload.decision_keys):
            found.setdefault(str(key), (artifact_id, envelope.evidence_ref()))
    return found


def owner_authorization_readiness(
    store_root: Path, requirement_set, *, expected_class: str | None = "research"
) -> AuthorizationReadiness:
    """Typed readiness of the computation-path-scoped owner bundle for a
    research purpose: the store must be a verified namespace of the expected
    class; every required decision key needs a catalogued, verified owner
    decision artifact of THIS namespace; the current supersession head must
    be provable. Absent artifacts are ``missing`` with the exact keys —
    never inferred from a path, a file name or a synthetic marker."""

    from .search.store_namespace import StoreNamespaceError  # noqa: PLC0415
    from .search.supersession_chain import current_supersession_head_witness  # noqa: PLC0415

    root = Path(store_root)
    authorization_class = "owner_authorization_bundle"
    required = tuple(
        requirement.decision_key for requirement in requirement_set.payload.requirements
    )
    try:
        blocked = _namespace_readiness_block(
            root, authorization_class=authorization_class, expected_class=expected_class
        )
        if blocked is not None:
            return blocked
        namespace = resolve_store_namespace(root, expected_class=expected_class)
        assert namespace.store_namespace_id is not None
        evidence = _owner_decision_evidence(root, namespace.store_namespace_id)
    except Exception as error:  # noqa: BLE001
        return AuthorizationReadiness(
            authorization_class,
            "unavailable",
            f"the owner-decision lookup failed: {type(error).__name__}",
        )
    missing = tuple(key for key in required if key not in evidence)
    if missing:
        return AuthorizationReadiness(
            authorization_class,
            "missing",
            "no registered owner-decision evidence artifact exists for "
            f"{len(missing)} required decision(s): {', '.join(missing)}",
            missing_decision_keys=missing,
            store_namespace_id=namespace.store_namespace_id,
        )
    try:
        current_supersession_head_witness(root)
    except StoreNamespaceError as error:
        return AuthorizationReadiness(
            authorization_class,
            _HEAD_REASON_TO_STATUS.get(error.reason, "wrong_head"),
            str(error),
            store_namespace_id=namespace.store_namespace_id,
        )
    return AuthorizationReadiness(
        authorization_class,
        "ready",
        f"every required decision ({len(required)}) has a verified owner-decision artifact "
        "of this namespace; the bundle binds the current supersession head",
        tuple(evidence[key][0] for key in required),
        store_namespace_id=namespace.store_namespace_id,
    )


def owner_authorization_bundle_from_store(
    store_root: Path, requirement_set, *, expected_class: str | None = "research"
):
    """The bundle for a ``ready`` research readiness (``None`` otherwise):
    the persisted evidence refs bound to the store's verified namespace and
    its CURRENT head witness."""

    from .search.authorization import OwnerAuthorizationBundle  # noqa: PLC0415
    from .search.supersession_chain import current_supersession_head_witness  # noqa: PLC0415

    readiness = owner_authorization_readiness(
        store_root, requirement_set, expected_class=expected_class
    )
    if readiness.status != "ready" or readiness.store_namespace_id is None:
        return None
    root = Path(store_root)
    evidence = _owner_decision_evidence(root, readiness.store_namespace_id)
    witness = current_supersession_head_witness(root)
    return OwnerAuthorizationBundle(
        requirement_set_id=requirement_set.requirement_set_id,
        decision_refs={
            requirement.decision_key: evidence[requirement.decision_key][1]
            for requirement in requirement_set.payload.requirements
        },
        store_namespace_id=readiness.store_namespace_id,
        supersession_head_witness=witness,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI-2 — Verification Center read models (plan §5.3 / §8 / §9 Phase 2)
#
# Every function here is a READ-ONLY adapter over the backend contracts of
# HARDENING-BACKEND Phase 3 / 4 (the shortlist, seed production, the signed
# verification authorization, the bounded-run preflight) or a mutable,
# non-semantic center record. Immutable stores are never listed — every
# artifact resolves by EXACT id; a local path never defines authority; nothing
# here signs, produces a seed or launches a run.
# ─────────────────────────────────────────────────────────────────────────────

#: The receipt files the seed-production CLI writes with ``--receipt-out``
#: under the center root — the refresh seam for the external seed steps.
SEED_AUTHORIZATION_RECEIPT = "seed_production_authorization.receipt.json"
SEED_RUN_RECEIPT = "seed_production_run.receipt.json"
#: The mutable, non-semantic center record (provisional window + recorded ids).
CENTER_RECORD_FILE = "verification_center.json"

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64_RE.fullmatch(value))


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid4().hex}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _has_envelope_or_corrupt(root: Path, store_name: str, envelope_id: str) -> bool:
    """``has_envelope`` that reports a corrupt entry (directory without its
    manifest) as PRESENT — the caller's typed load then names the corruption."""

    try:
        return has_envelope(Path(root), store_name, envelope_id)
    except Exception:  # noqa: BLE001 — an existing corrupt entry
        return True


# ── the shortlist ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ShortlistState:
    status: str  # available | missing | corrupt
    detail: str
    source_path: str
    shortlist_id: str | None = None
    shortlist: Any = None
    generated_at_utc: str | None = None


def load_window_shortlist(path: Path) -> ShortlistState:
    """The persisted shortlist document (``LOGICAL_WINDOW_COVERAGE_SCAN.json``)
    re-validated through its typed contract; the recorded shortlist id must
    hash the document. Missing and corrupt are typed states."""

    from .search.verification_window import VerificationWindowShortlist  # noqa: PLC0415

    source = Path(path)
    if not source.exists():
        return ShortlistState(
            "missing",
            "no shortlist document exists at the configured evidence location",
            str(source),
        )
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        shortlist = VerificationWindowShortlist.model_validate(document["shortlist"])
        recorded = document.get("shortlist_id")
        if recorded is not None and recorded != shortlist.shortlist_id:
            raise ValueError("the recorded shortlist id does not hash the document")
    except Exception as error:  # noqa: BLE001 — typed, sanitized at render
        return ShortlistState(
            "corrupt",
            f"the shortlist document failed verification ({type(error).__name__})",
            str(source),
        )
    generated = document.get("generated_at_utc")
    return ShortlistState(
        "available",
        f"{shortlist.candidate_window_count} candidate windows, "
        f"{shortlist.eligible_window_count} eligible; owner selection "
        f"{shortlist.owner_selection}",
        str(source),
        shortlist_id=shortlist.shortlist_id,
        shortlist=shortlist,
        generated_at_utc=str(generated) if generated else None,
    )


def shortlist_window(shortlist, days: Sequence[str]):
    """The EXACT window (same logical days) of the shortlist, or ``None``."""

    target = tuple(str(day) for day in days)
    for score in shortlist.ranked_windows:
        if tuple(score.days) == target:
            return score
    for entry in shortlist.entries:
        if tuple(entry.window.days) == target:
            return entry.window
    return None


def coverage_matrix_from_shortlist_window(shortlist, window):
    """The coverage matrix of ONE shortlisted window, derived from the
    shortlist's own per-day coverage rows (already-authorized evidence; no
    table is re-read). Content-addressed like every envelope; nothing is
    persisted here."""

    from .data_access import allowlist_sha256  # noqa: PLC0415
    from .search.verification import (  # noqa: PLC0415
        CoverageMatrixEnvelope,
        CoverageMatrixPayload,
        DayCoverageRow,
    )

    rows = tuple(
        DayCoverageRow(
            trading_day=row.logical_trading_day,
            source_partition_recorded=bool(row.source_partitions_present),
            setup_lifecycle_rows=int(row.setup_lifecycle_rows),
            entry_candidate_rows=int(row.entry_candidate_rows),
            eligible_decision_rows=int(row.eligible_decision_rows),
            executed_trade_rows=int(row.executed_trade_rows),
            candidate_label_rows=int(row.candidate_label_rows),
            audit_event_rows=None,  # the shortlist records coverage, not counts
            replay_chart_available=None,
        )
        for row in window.coverage
    )
    lifecycle_paths = {
        "setup_activation": any(row.setup_lifecycle_rows for row in rows),
        "entry_candidate": any(row.entry_candidate_rows for row in rows),
        "eligible_decision": any(row.eligible_decision_rows for row in rows),
        "execution_resolution": any(row.executed_trade_rows for row in rows),
        "candidate_labeling": any(row.candidate_label_rows for row in rows),
        "audit_events": any(bool(row.audit_day_covered) for row in window.coverage),
        "replay_chart": False,
    }
    uncovered = sorted(path for path, covered in lifecycle_paths.items() if not covered)
    days = tuple(window.days)
    payload = CoverageMatrixPayload(
        evidence_source_dataset_id=shortlist.evidence_source_dataset_id,
        evidence_source_manifest_sha256=shortlist.evidence_source_manifest_sha256,
        candidate_allowlist=days,
        candidate_allowlist_hash=allowlist_sha256(days),
        rows=rows,
        lifecycle_paths_covered=lifecycle_paths,
        uncovered_paths_note=(
            "all scored lifecycle paths are covered by the selected window"
            if not uncovered
            else "not covered by the selected window (derived from the shortlist rows): "
            + ", ".join(uncovered)
        ),
    )
    return CoverageMatrixEnvelope.from_payload(payload)


# ── the source inventory (already-authorized manifest evidence) ──────────────


@dataclass(frozen=True)
class InventoryState:
    status: str  # available | missing | corrupt
    detail: str
    source_path: str
    inventory: dict[str, tuple[str, str]] | None = None
    partition_count: int = 0


def load_source_inventory(path: Path) -> InventoryState:
    """``{physical_utc_date: (public kind, content sha256)}`` from an accepted
    manifest's ``identity.permitted_source_hashes`` (or a permitted-hash list /
    day map, the seed CLI's accepted forms). No raw source is discovered."""

    from .search.trading_calendar import inventory_from_permitted_source_hashes  # noqa: PLC0415

    source = Path(path)
    if not source.exists():
        return InventoryState(
            "missing",
            "no accepted-manifest inventory exists at the configured location",
            str(source),
        )
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(document, dict) and "identity" in document:
            document = document["identity"]["permitted_source_hashes"]
        if isinstance(document, list):
            inventory = inventory_from_permitted_source_hashes(document)
        elif isinstance(document, dict):
            inventory = {
                str(day): (str(entry[0]), str(entry[1])) for day, entry in document.items()
            }
        else:
            raise ValueError("unrecognized inventory document shape")
    except Exception as error:  # noqa: BLE001
        return InventoryState(
            "corrupt",
            f"the inventory document failed to load ({type(error).__name__})",
            str(source),
        )
    return InventoryState(
        "available",
        f"{len(inventory)} physical partitions from the accepted inventory",
        str(source),
        inventory=inventory,
        partition_count=len(inventory),
    )


# ── the mutable center record ────────────────────────────────────────────────


@dataclass
class VerificationCenterRecord:
    """Mutable, non-semantic authoring state of the Verification Center: the
    owner's PROVISIONAL window and the exact ids recorded from the external
    seed steps. Never an authorization; never part of an identity."""

    schema_version: int = 1
    store_namespace_id: str | None = None
    provisional_window: dict[str, Any] | None = None
    seed_production_authorization_id: str | None = None
    seed_production_run_id: str | None = None
    seed_snapshot_id: str | None = None
    updated_at_utc: str = ""


def load_verification_center_record(center_root: Path) -> VerificationCenterRecord:
    """The record, or an EMPTY one when the file is absent or unreadable
    (an unreadable record is never trusted)."""

    path = Path(center_root) / CENTER_RECORD_FILE
    if not path.exists():
        return VerificationCenterRecord()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        known = set(VerificationCenterRecord.__dataclass_fields__)
        return VerificationCenterRecord(
            **{key: value for key, value in dict(payload).items() if key in known}
        )
    except Exception:  # noqa: BLE001 — a corrupt record is an empty record
        return VerificationCenterRecord()


def save_verification_center_record(
    center_root: Path, record: VerificationCenterRecord, *, now_fn=_utc_now_iso
) -> Path:
    record.updated_at_utc = now_fn()
    path = Path(center_root) / CENTER_RECORD_FILE
    _write_json_atomic(path, asdict(record))
    return path


def pick_up_seed_receipts(
    center_root: Path, record: VerificationCenterRecord
) -> tuple[VerificationCenterRecord, tuple[str, ...]]:
    """Read the receipt files the external seed CLI wrote under the center
    root and record their EXACT ids (64-hex only). Returns the updated record
    (a copy) and one human note per receipt file seen; an unreadable receipt
    is reported, never trusted, never fatal."""

    notes: list[str] = []
    updated = replace(record)
    root = Path(center_root)
    expectations = (
        (
            SEED_AUTHORIZATION_RECEIPT,
            (("seed_production_authorization_id", "seed_production_authorization_id"),),
        ),
        (
            SEED_RUN_RECEIPT,
            (
                ("seed_snapshot_id", "seed_snapshot_id"),
                ("seed_production_run_id", "seed_production_run_id"),
            ),
        ),
    )
    for filename, fields in expectations:
        path = root / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("receipt is not an object")
        except Exception as error:  # noqa: BLE001
            notes.append(f"{filename}: unreadable receipt ({type(error).__name__}); ignored")
            continue
        picked: list[str] = []
        for receipt_key, record_field in fields:
            value = payload.get(receipt_key)
            if _is_hex64(value):
                setattr(updated, record_field, value)
                picked.append(f"{record_field} {value[:12]}…")
            else:
                notes.append(f"{filename}: no exact {receipt_key} in the receipt; ignored")
        if picked:
            notes.append(f"{filename}: picked up " + ", ".join(picked))
    return updated, tuple(notes)


# ── the seed lane by exact id ────────────────────────────────────────────────


@dataclass(frozen=True)
class SeedAuthorizationState:
    status: str
    detail: str
    authorization_id: str | None = None
    provenance: str | None = None
    chain_first_day: str | None = None
    chain_last_day: str | None = None
    chain_day_count: int = 0
    first_intended_verification_day: str | None = None
    baseline_profile_name: str | None = None


def seed_authorization_state(
    store_root: Path,
    authorization_id: str | None,
    *,
    baseline_profile_name: str,
    section_hash: str,
    first_intended_verification_day: str,
    inventory: Mapping[str, tuple[str, str]] | None = None,
    quant_lab_source_identity: str | None = None,
    strategy_core: tuple[str, str] | None = None,
    now: str | None = None,
) -> SeedAuthorizationState:
    """Typed state of the recorded seed-production authorization: exact-id
    load, namespace, profile / section, the intended first verification day;
    with the inventory (and the code identities) the COMPLETE backend
    verification runs (``verify_seed_production_authorization``) and every
    refusal keeps its typed reason. Without the inventory the state is
    ``verified_envelope`` — explicitly not the full check."""

    from .search.seed_production import (  # noqa: PLC0415
        SEED_PRODUCTION_AUTHORIZATION_STORE,
        SeedProductionAuthorizationEnvelope,
        SeedProductionAuthorizationError,
        seed_chain_source_inventory_hash,
        verify_seed_production_authorization,
    )

    root = Path(store_root)
    if not authorization_id:
        return SeedAuthorizationState(
            "not_recorded",
            "no seed-production authorization id is recorded yet (register the completed "
            "packet with the seed CLI, then refresh)",
        )
    if not _is_hex64(authorization_id) or not _has_envelope_or_corrupt(
        root, SEED_PRODUCTION_AUTHORIZATION_STORE, authorization_id
    ):
        return SeedAuthorizationState(
            "not_found",
            f"no seed-production authorization {str(authorization_id)[:12]}… exists in this "
            "store (exact-id load; the store is never listed)",
            authorization_id=authorization_id,
        )
    try:
        envelope = load_verified_envelope(
            root,
            SEED_PRODUCTION_AUTHORIZATION_STORE,
            authorization_id,
            SeedProductionAuthorizationEnvelope,
        )
    except Exception as error:  # noqa: BLE001
        return SeedAuthorizationState(
            "corrupt",
            f"seed-production authorization {authorization_id[:12]}… failed verification "
            f"({type(error).__name__})",
            authorization_id=authorization_id,
        )
    payload = envelope.payload
    chain = tuple(payload.ordered_seed_chain_replay_days)
    facts = dict(
        authorization_id=envelope.seed_production_authorization_id,
        provenance=str(payload.provenance),
        chain_first_day=chain[0],
        chain_last_day=chain[-1],
        chain_day_count=len(chain),
        first_intended_verification_day=payload.first_intended_verification_day,
        baseline_profile_name=payload.baseline_profile_name,
    )
    namespace = resolve_store_namespace(root, expected_class="test")
    if (
        namespace.status == "verified"
        and payload.store_namespace_id != namespace.store_namespace_id
    ):
        return SeedAuthorizationState(
            "wrong_namespace", "the authorization names another store namespace", **facts
        )
    if (
        payload.baseline_profile_name != baseline_profile_name
        or payload.resolved_section_config_hash != section_hash
    ):
        return SeedAuthorizationState(
            "profile_mismatch",
            "the authorization binds a different baseline profile / section hash",
            **facts,
        )
    if payload.first_intended_verification_day != first_intended_verification_day:
        return SeedAuthorizationState(
            "window_mismatch",
            f"the authorization's first intended verification day "
            f"{payload.first_intended_verification_day} is not the selected window's first "
            f"day {first_intended_verification_day}",
            **facts,
        )
    if inventory is None:
        return SeedAuthorizationState(
            "verified_envelope",
            "the persisted authorization verified (namespace, profile, window); the source "
            "inventory hash and the code identities were NOT checked because the accepted "
            "inventory is not available to this workspace",
            **facts,
        )
    try:
        inventory_hash = seed_chain_source_inventory_hash(chain, inventory)
        verify_seed_production_authorization(
            root,
            envelope,
            expected_profile_name=baseline_profile_name,
            expected_section_config_hash=section_hash,
            expected_chain_replay_days=chain,
            expected_source_inventory_hash=inventory_hash,
            expected_quant_lab_source_identity=quant_lab_source_identity,
            expected_strategy_core=strategy_core,
            now=now or _utc_now_iso(),
        )
    except SeedProductionAuthorizationError as error:
        return SeedAuthorizationState(error.reason, str(error), **facts)
    except ValueError as error:
        return SeedAuthorizationState("source_inventory_mismatch", str(error), **facts)
    return SeedAuthorizationState(
        "verified",
        "the persisted authorization passed the complete backend verification (namespace, "
        "current head witness, profile, chain, inventory hash, code identities, schema, "
        "effectivity)",
        **facts,
    )


@dataclass(frozen=True)
class SeedSnapshotState:
    status: str
    detail: str
    seed_snapshot_id: str | None = None
    seed_hash: str | None = None
    first_replay_day: str | None = None
    snapshot_through_day: str | None = None
    chain_date_count: int = 0
    profile_name: str | None = None


def seed_snapshot_state(
    store_root: Path,
    seed_snapshot_id: str | None,
    *,
    section_hash: str,
    first_replay_day: str,
    profile_name: str,
) -> SeedSnapshotState:
    """Typed state of the recorded seed snapshot through the VERIFIED,
    profile-bound loader the runner itself uses."""

    from .search.child_replay import SeedSnapshotError, load_seed_snapshot  # noqa: PLC0415

    root = Path(store_root)
    if not seed_snapshot_id:
        return SeedSnapshotState(
            "not_recorded",
            "no seed snapshot id is recorded yet (run the authorized seed job, then refresh)",
        )
    if not _is_hex64(seed_snapshot_id) or not _has_envelope_or_corrupt(
        root, "seed_snapshots", seed_snapshot_id
    ):
        return SeedSnapshotState(
            "missing",
            f"no seed snapshot {str(seed_snapshot_id)[:12]}… exists in this store "
            "(exact-id load)",
            seed_snapshot_id=seed_snapshot_id,
        )
    try:
        envelope, _chain_start = load_seed_snapshot(
            root, seed_snapshot_id, expected_section_config_hash=section_hash
        )
    except SeedSnapshotError as error:
        return SeedSnapshotState(
            "profile_mismatch", str(error), seed_snapshot_id=seed_snapshot_id
        )
    except Exception as error:  # noqa: BLE001
        return SeedSnapshotState(
            "corrupt",
            f"seed snapshot {seed_snapshot_id[:12]}… failed verification "
            f"({type(error).__name__})",
            seed_snapshot_id=seed_snapshot_id,
        )
    payload = envelope.payload
    facts = dict(
        seed_snapshot_id=envelope.seed_snapshot_id,
        seed_hash=payload.seed_hash,
        first_replay_day=payload.first_replay_day,
        snapshot_through_day=payload.snapshot_through_day,
        chain_date_count=int(payload.chain_date_count),
        profile_name=payload.profile_name,
    )
    if payload.profile_name != profile_name:
        return SeedSnapshotState(
            "profile_mismatch",
            f"the seed is bound to profile {payload.profile_name!r}, not {profile_name!r}",
            **facts,
        )
    if payload.first_replay_day != first_replay_day:
        return SeedSnapshotState(
            "discontinuous",
            f"the seed's first replay day {payload.first_replay_day} does not continue into "
            f"the selected window starting {first_replay_day}",
            **facts,
        )
    return SeedSnapshotState(
        "verified",
        "verified, profile-bound seed snapshot continuous with the selected window",
        **facts,
    )


@dataclass(frozen=True)
class SeedReceiptState:
    status: str
    detail: str
    run_id: str | None = None
    seed_snapshot_id: str | None = None
    authorization_id: str | None = None
    provenance: str | None = None
    chain_replay_day_count: int = 0
    logical_trading_day_count: int = 0
    first_intended_verification_day: str | None = None
    verification_evidence_footprint_days: int | None = None
    access_audit_sha256: str | None = None


def seed_receipt_state(
    store_root: Path, run_id: str | None, *, expected_seed_snapshot_id: str | None = None
) -> SeedReceiptState:
    """Typed state of the recorded seed-production run receipt (exact id)."""

    from .search.seed_production import (  # noqa: PLC0415
        SEED_PRODUCTION_RUN_STORE,
        SeedProductionRunEnvelope,
    )

    root = Path(store_root)
    if not run_id:
        return SeedReceiptState(
            "not_recorded", "no seed-production run receipt id is recorded yet"
        )
    if not _is_hex64(run_id) or not _has_envelope_or_corrupt(
        root, SEED_PRODUCTION_RUN_STORE, run_id
    ):
        return SeedReceiptState(
            "missing",
            f"no seed-production run receipt {str(run_id)[:12]}… exists in this store",
            run_id=run_id,
        )
    try:
        envelope = load_verified_envelope(
            root, SEED_PRODUCTION_RUN_STORE, run_id, SeedProductionRunEnvelope
        )
    except Exception as error:  # noqa: BLE001
        return SeedReceiptState(
            "corrupt",
            f"seed-production run receipt {run_id[:12]}… failed verification "
            f"({type(error).__name__})",
            run_id=run_id,
        )
    payload = envelope.payload
    facts = dict(
        run_id=envelope.seed_production_run_id,
        seed_snapshot_id=payload.seed_snapshot_id,
        authorization_id=payload.seed_production_authorization_id,
        provenance=str(payload.provenance),
        chain_replay_day_count=int(payload.chain_replay_day_count),
        logical_trading_day_count=int(payload.logical_trading_day_count),
        first_intended_verification_day=payload.first_intended_verification_day,
        verification_evidence_footprint_days=int(payload.verification_evidence_footprint_days),
        access_audit_sha256=payload.access_audit_sha256,
    )
    if (
        expected_seed_snapshot_id is not None
        and payload.seed_snapshot_id != expected_seed_snapshot_id
    ):
        return SeedReceiptState(
            "mismatch", "the run receipt names another seed snapshot", **facts
        )
    return SeedReceiptState(
        "verified",
        f"verified receipt: {payload.chain_replay_day_count} chain days replayed, "
        f"{payload.verification_evidence_footprint_days} verification-evidence days "
        "(a separately authorized preparation action)",
        **facts,
    )


# ── the owner's signed verification authorization ────────────────────────────


@dataclass(frozen=True)
class SignedRefState:
    status: str
    detail: str
    source_path: str
    ref: Any = None


def load_signed_verification_ref(
    path: Path,
    *,
    store_root: Path,
    expected_seed_snapshot_id: str,
    expected_allowlist_hash: str,
) -> SignedRefState:
    """The owner's COMPLETED ``VerificationAuthorizationRef`` read from the
    named file (a bare ref, or a packet wrapping ``verification_authorization_ref``)
    and validated TYPED against this store's verified namespace and current
    head, the verified seed and the selected window. Never persisted here."""

    from .search.authorization import (  # noqa: PLC0415
        AuthorizationError,
        VerificationAuthorizationRef,
        assert_authorization_bound_to_store,
    )
    from .search.seed_production import OWNER_PLACEHOLDER  # noqa: PLC0415

    source = Path(path)
    if not source.exists():
        return SignedRefState(
            "file_missing",
            "no completed verification authorization file exists at the named location",
            str(source),
        )
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(document, dict) and "verification_authorization_ref" in document:
            document = document["verification_authorization_ref"]
        if not isinstance(document, dict):
            raise ValueError("the file is not a reference object")
    except Exception as error:  # noqa: BLE001
        return SignedRefState(
            "malformed",
            f"the file could not be read as a reference ({type(error).__name__})",
            str(source),
        )
    if any(
        isinstance(value, str) and OWNER_PLACEHOLDER in value for value in document.values()
    ):
        return SignedRefState(
            "unsigned",
            "the reference still carries an owner placeholder (approved_by / approved_at / "
            "content_hash) — only the owner completes it, outside this workspace",
            str(source),
        )
    try:
        ref = VerificationAuthorizationRef.model_validate(
            {k: v for k, v in document.items() if k in VerificationAuthorizationRef.model_fields}
        )
    except Exception as error:  # noqa: BLE001
        return SignedRefState(
            "malformed",
            f"the reference failed contract validation ({type(error).__name__})",
            str(source),
        )
    namespace = resolve_store_namespace(Path(store_root), expected_class="test")
    if namespace.status != "verified":
        return SignedRefState(
            "store_unmarked" if namespace.status == "unmarked" else "wrong_namespace",
            namespace.detail,
            str(source),
            ref=ref,
        )
    if ref.store_namespace_id != namespace.store_namespace_id:
        return SignedRefState(
            "wrong_namespace",
            "the reference authorizes another store namespace",
            str(source),
            ref=ref,
        )
    try:
        assert_authorization_bound_to_store(
            Path(store_root),
            store_namespace_id=ref.store_namespace_id,
            supersession_head_witness=ref.supersession_head_witness,
            expected_namespace_class="test",
        )
    except AuthorizationError as error:
        reason = str(getattr(error, "reason", "") or "")
        return SignedRefState(
            _HEAD_REASON_TO_STATUS.get(reason, "wrong_head"), str(error), str(source), ref=ref
        )
    if ref.seed_snapshot_id != expected_seed_snapshot_id:
        return SignedRefState(
            "seed_mismatch",
            "the reference names a different seed snapshot than the verified seed",
            str(source),
            ref=ref,
        )
    if ref.approved_allowlist_hash != expected_allowlist_hash:
        return SignedRefState(
            "allowlist_mismatch",
            "the reference's approved allowlist hash is not the selected window's hash "
            "(one canonical allowlist; never rotated)",
            str(source),
            ref=ref,
        )
    return SignedRefState(
        "valid",
        f"completed reference by {ref.approved_by} at {ref.approved_at}, bound to this "
        "store's verified namespace and current head, the verified seed and the selected "
        "window",
        str(source),
        ref=ref,
    )


# ── run registration, preflight, monitor ─────────────────────────────────────


def register_verification_run(
    store_root: Path,
    *,
    ref,
    pipeline_semantic_id: str,
    allowlist: Sequence[str],
    seed_snapshot_id: str,
    baseline_profile_name: str,
    baseline_section_config_hash: str,
    coverage_matrix_artifact_id: str,
    display_name: str,
):
    """Persist (save-or-reuse) the ``VerificationRunEnvelope`` binding the
    owner's signed reference to the frozen pipeline spec — VALIDATED through
    the backend's fail-before-path check first (a mismatch persists nothing).
    Catalogued by a display-name event so the typed readiness discovers it.
    Nothing here launches; ``register_program_allowlist`` is never called."""

    from .data_access import allowlist_sha256  # noqa: PLC0415
    from .search.catalog import append_catalog_event  # noqa: PLC0415
    from .search.store import save_or_reuse_envelope  # noqa: PLC0415
    from .search.verification import (  # noqa: PLC0415
        VerificationRunEnvelope,
        VerificationRunPayload,
        validate_verification_run,
    )

    root = Path(store_root)
    days = tuple(str(day) for day in allowlist)
    envelope = VerificationRunEnvelope.from_payload(
        VerificationRunPayload(
            pipeline_semantic_id=pipeline_semantic_id,
            verification_authorization=ref,
            allowlist=days,
            allowlist_hash=allowlist_sha256(days),
            seed_snapshot_id=seed_snapshot_id,
            baseline_profile_id=baseline_profile_name,
            baseline_section_config_hash=baseline_section_config_hash,
            coverage_matrix_artifact_id=coverage_matrix_artifact_id,
        )
    )
    validate_verification_run(
        envelope,
        expected_pipeline_semantic_id=pipeline_semantic_id,
        expected_baseline_profile_id=baseline_profile_name,
        expected_baseline_section_config_hash=baseline_section_config_hash,
        expected_seed_snapshot_id=ref.seed_snapshot_id,
        authorization=ref,
        store_root=root,
    )
    stored, _reused = save_or_reuse_envelope(root, "verification_runs", envelope)
    if stored.verification_run_id not in catalog_annotations(root):
        append_catalog_event(
            root,
            kind="display_name",
            artifact_id=stored.verification_run_id,
            payload={"display_name": display_name},
        )
    return stored


@dataclass(frozen=True)
class PreflightState:
    status: str  # passed | refused | unavailable
    reason: str | None
    detail: str
    record: Any = None


def bounded_preflight_state(
    store_root: Path,
    repo_root: Path,
    run_envelope,
    *,
    section_hash: str,
    authorization=None,
    logical_day_refs=None,
) -> PreflightState:
    """The §6.1 preflight as a typed read model: every check runs BEFORE any
    source path exists; a refusal keeps the backend's reason."""

    from .search.bounded_verification import (  # noqa: PLC0415
        BoundedVerificationRefusalError,
        preflight_bounded_verification,
    )

    payload = run_envelope.payload
    try:
        record = preflight_bounded_verification(
            store_root=Path(store_root),
            repo_root=Path(repo_root),
            verification_run=run_envelope,
            authorization=(
                authorization
                if authorization is not None
                else payload.verification_authorization
            ),
            pipeline_semantic_id=payload.pipeline_semantic_id,
            baseline_profile_id=payload.baseline_profile_id,
            baseline_section_config_hash=section_hash,
            logical_day_refs=logical_day_refs,
        )
    except BoundedVerificationRefusalError as error:
        return PreflightState("refused", error.reason, str(error))
    except PermissionError as error:
        return PreflightState("refused", "fail_before_path", str(error))
    except Exception as error:  # noqa: BLE001
        return PreflightState(
            "unavailable", None, f"the preflight could not run ({type(error).__name__})"
        )
    return PreflightState(
        "passed",
        None,
        f"every §6.1 check passed ({len(record.checks)} checks); no source path was "
        "constructed",
        record=record,
    )


def verification_stage_rows(state: Mapping[str, Any] | None):
    """The monitor rows of the resolved verification plan ONLY (stages outside
    the plan are never listed)."""

    from .study_presentation import derive_pipeline_stage_rows  # noqa: PLC0415

    if not state:
        return ()
    return tuple(row for row in derive_pipeline_stage_rows(state) if row.in_plan)


_EVIDENCE_REPORTS: tuple[str, ...] = (
    "R1_BASELINE_GATES.json",
    "BOUNDED_RELEASE_CONTROL_FLOW.json",
)


def verification_evidence_summary(evidence_dir: Path) -> dict[str, Any]:
    """The ``passed`` flags and gate ids of the evidence reports the bounded
    runner writes (``R1_BASELINE_GATES.json``, ``BOUNDED_RELEASE_CONTROL_FLOW.json``).
    A missing folder is ``present = False``; an unreadable report is reported
    as ``passed = None`` (never assumed)."""

    root = Path(evidence_dir)
    summary: dict[str, Any] = {"present": root.is_dir(), "reports": {}}
    if not summary["present"]:
        return summary
    for name in _EVIDENCE_REPORTS:
        path = root / name
        if not path.exists():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            payload = document.get("payload", document) if isinstance(document, dict) else {}
            passed = payload.get("passed") if isinstance(payload, Mapping) else None
            gate_ids: list[str] = []
            for key in ("first_attempt_gates", "second_attempt_gates"):
                gates = payload.get(key) if isinstance(payload, Mapping) else None
                if isinstance(gates, Mapping):
                    results = gates.get("results")
                    if isinstance(results, Mapping):
                        gate_ids.extend(str(gate) for gate in results)
            components = payload.get("components") if isinstance(payload, Mapping) else None
            if isinstance(components, list):
                gate_ids.extend(
                    str(item.get("component")) for item in components if isinstance(item, Mapping)
                )
            summary["reports"][name] = {
                "passed": passed if isinstance(passed, bool) else None,
                "gate_ids": tuple(dict.fromkeys(gate_ids)),
            }
        except Exception as error:  # noqa: BLE001 — reported, never assumed
            summary["reports"][name] = {
                "passed": None,
                "error": type(error).__name__,
                "gate_ids": (),
            }
    return summary
