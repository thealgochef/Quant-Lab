"""The Verification Center (UI-2; plan §5.3, §7, §9 Phase 2; FUX §14.4).

One state-driven surface for Implementation Verification with NO research
steps: purpose / readiness → the logical trading-day fixture (LOGICAL days
and their PHYSICAL partitions displayed separately) → the seed-production
authorization packet, its registration receipt, the exact seed job command
and the verified seed → the final verification authorization packet and the
owner's completed reference → review / run (the charter frozen from the
SIGNED reference, the pipeline spec, the registered run, the §6.1 preflight
and the exact bounded-run command) → the monitor of the resolved seed /
verification stages only. No prop, risk, research benchmark, feature / model
or publication control exists here.

Everything semantic is consumed read-only from the backend contracts
through ``study_providers``; the center never signs an owner artifact, never
produces a seed, never launches (no spawn seam exists in this module) and
never registers the program allowlist. External steps expose their EXACT
command, packet path and refresh semantics; their results are picked up from
receipt files under the mutable center root. A local path never defines
authority: every id is exact-loaded and verified on render.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st
from ifvg_ui_common import (
    STATE_PREFIX,
    cli_escape_hatch,
    identity_block,
    render_empty_state,
    sanitize_error,
    sanitize_select,
    verification_badge,
)

from alpha_lab.agents.data_infra.ifvg.config import (
    FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
    V2_DATASET_DIR,
)
from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.presentation.help_registry import help_text
from alpha_lab.agents.data_infra.ifvg.presentation.run_purpose import (
    PURPOSE_DESCRIPTIONS,
    RunPurpose,
)
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    derive_authorization_requirements,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
    EXECUTION_MODE_V1,
    SUPPORTED_CHILD_WORKERS,
)
from alpha_lab.agents.data_infra.ifvg.search.seed_production import (
    PROHIBITED_SEED_OUTPUTS,
    SeedProductionAuthorizationError,
    build_seed_production_packet,
    build_verification_authorization_packet,
    render_seed_production_packet_markdown,
    render_verification_packet_markdown,
)
from alpha_lab.agents.data_infra.ifvg.search.trading_calendar import (
    PERMITTED_WINDOW_LAST_DAY,
    PROTECTED_BUFFER_DAY,
    SEALED_START_DAY,
)
from alpha_lab.agents.data_infra.ifvg.study_providers import (
    SEED_AUTHORIZATION_RECEIPT,
    SEED_RUN_RECEIPT,
    AuthorizationReadiness,
    InventoryState,
    PreflightState,
    SeedAuthorizationState,
    SeedReceiptState,
    SeedSnapshotState,
    ShortlistState,
    SignedRefState,
    VerificationCenterRecord,
    bounded_preflight_state,
    coverage_matrix_from_shortlist_window,
    load_signed_verification_ref,
    load_source_inventory,
    load_verification_center_record,
    load_window_shortlist,
    pick_up_seed_receipts,
    register_verification_run,
    resolve_store_namespace,
    save_verification_center_record,
    seed_authorization_state,
    seed_receipt_state,
    seed_snapshot_state,
    shortlist_window,
    verification_authorization_readiness,
    verification_bundle_from_signed_ref,
    verification_evidence_summary,
    verification_stage_rows,
)

__all__ = [
    "DEFAULT_PROFILE",
    "INVENTORY_MANIFEST_PATH",
    "SHORTLIST_DOCUMENT_PATH",
    "SEED_PACKET_JSON",
    "SIGNED_REF_FILE",
    "VERIFICATION_PACKET_JSON",
    "EVIDENCE_DIR_NAME",
    "render_verification_center",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Operational evidence locations (module attributes; tests inject tmp files).
#: Neither defines authority: the shortlist is re-validated through its typed
#: contract and the inventory is the accepted manifest's permitted hashes.
SHORTLIST_DOCUMENT_PATH = (
    _REPO_ROOT
    / "QL-FSM-PROP-SEARCH-DASHBOARD"
    / "implementation-progress"
    / "HARDENING-BACKEND"
    / "LOGICAL_WINDOW_COVERAGE_SCAN.json"
)
INVENTORY_MANIFEST_PATH = (
    _REPO_ROOT / V2_DATASET_DIR / FSM_AUDIT_ACCEPTED_V2_DATASET_ID / "exploration" / "manifest.json"
)
DEFAULT_PROFILE = "ifvg_v2_doc_default_fresh_static_1r"

#: The center's mutable files (packets, the owner's completed reference,
#: receipts) — never authority, never an identity.
SEED_PACKET_JSON = "SEED_PRODUCTION_PACKET.json"
SEED_PACKET_MD = "SEED_PRODUCTION_PACKET.md"
VERIFICATION_PACKET_JSON = "VERIFICATION_PACKET.json"
VERIFICATION_PACKET_MD = "VERIFICATION_PACKET.md"
COVERAGE_MATRIX_JSON = "COVERAGE_MATRIX.json"
SIGNED_REF_FILE = "VERIFICATION_AUTHORIZATION_REF.signed.json"
EVIDENCE_DIR_NAME = "R1-VERIFICATION-EVIDENCE"

_VC = f"{STATE_PREFIX}vc_"
_START = f"{STATE_PREFIX}start_"
_DRAFT_KEY = f"{STATE_PREFIX}draft_id"
_TOP_N = 12


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _rel(path: Path) -> str:
    """A project-relative rendering for commands (never authority)."""

    try:
        return str(Path(path).resolve().relative_to(_REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Resolution — every typed state, read-only, on each render
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CenterContext:
    roots: Mapping[str, Any]
    store_root: Path
    repo_root: Path
    center_root: Path
    profile: str
    section_hash: str
    namespace: Any
    record: VerificationCenterRecord
    receipt_notes: tuple[str, ...]
    shortlist: ShortlistState
    window: Any
    window_note: str | None
    inventory: InventoryState
    seed_authorization: SeedAuthorizationState | None
    seed_receipt: SeedReceiptState | None
    seed_snapshot: SeedSnapshotState | None
    coverage_matrix: Any
    signed_ref: SignedRefState | None
    readiness: AuthorizationReadiness
    run_envelope: Any
    preflight: PreflightState | None


def _code_identities(repo_root: Path) -> tuple[str | None, tuple[str, str] | None, str | None]:
    """The Quant-Lab replay-source identity and the Strategy-Core identity of
    the CODE this workspace runs (read-only repository queries). ``None`` with
    a note when they cannot be computed — nothing is guessed."""

    from ifvg_study_wizard import _strategy_core_root  # noqa: PLC0415

    from alpha_lab.agents.data_infra.ifvg.search.identities import (  # noqa: PLC0415
        quant_lab_replay_source_identity,
        strategy_core_source_identity,
    )

    try:
        quant_lab = quant_lab_replay_source_identity(repository_root=_REPO_ROOT)
        strategy_core = strategy_core_source_identity(repository_root=_strategy_core_root())
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        return None, None, f"code identities unavailable: {sanitize_error(error)}"
    return quant_lab, strategy_core, None


def _resolve(roots: Mapping[str, Any]) -> CenterContext:
    from ifvg_study_tab import roots_for_purpose  # noqa: PLC0415

    purpose_roots = roots_for_purpose(roots, RunPurpose.IMPLEMENTATION_VERIFICATION)
    store_root = Path(purpose_roots["store_root"])
    repo_root = Path(roots.get("repo_root") or _REPO_ROOT)
    center_root = Path(
        roots.get("verification_center_root") or _REPO_ROOT / "data/ifvg_verification_center"
    )
    resolved = resolve_profile_config({"profile_name": DEFAULT_PROFILE})
    section_hash = resolved.section_config_hash
    namespace = resolve_store_namespace(store_root, expected_class="test")
    record = load_verification_center_record(center_root)
    updated, notes = pick_up_seed_receipts(center_root, record)
    if (
        updated.seed_production_authorization_id,
        updated.seed_production_run_id,
        updated.seed_snapshot_id,
    ) != (
        record.seed_production_authorization_id,
        record.seed_production_run_id,
        record.seed_snapshot_id,
    ):
        save_verification_center_record(center_root, updated)
    record = updated
    shortlist = load_window_shortlist(SHORTLIST_DOCUMENT_PATH)
    window = None
    window_note = None
    provisional = record.provisional_window or {}
    if provisional.get("days") and shortlist.status == "available":
        window = shortlist_window(shortlist.shortlist, tuple(provisional["days"]))
        if window is None:
            window_note = (
                "the recorded provisional window is not in the current shortlist — record it again"
            )
        elif provisional.get("shortlist_id") not in (None, shortlist.shortlist_id):
            window_note = (
                "the window was recorded under an earlier shortlist document; the current "
                "document still lists it"
            )
    elif provisional.get("days"):
        window_note = (
            "the shortlist document is unavailable, so the recorded window cannot be re-verified"
        )
    inventory = load_source_inventory(INVENTORY_MANIFEST_PATH)
    seed_authorization = seed_receipt = seed_snapshot = None
    coverage_matrix = None
    signed_ref = None
    if window is not None:
        first_day = str(window.days[0])
        quant_lab, strategy_core, _note = _code_identities(repo_root)
        seed_authorization = seed_authorization_state(
            store_root,
            record.seed_production_authorization_id,
            baseline_profile_name=DEFAULT_PROFILE,
            section_hash=section_hash,
            first_intended_verification_day=first_day,
            inventory=inventory.inventory,
            quant_lab_source_identity=quant_lab,
            strategy_core=strategy_core,
        )
        seed_receipt = seed_receipt_state(
            store_root,
            record.seed_production_run_id,
            expected_seed_snapshot_id=record.seed_snapshot_id,
        )
        seed_snapshot = seed_snapshot_state(
            store_root,
            record.seed_snapshot_id,
            section_hash=section_hash,
            first_replay_day=first_day,
            profile_name=DEFAULT_PROFILE,
        )
        coverage_matrix = coverage_matrix_from_shortlist_window(shortlist.shortlist, window)
        if seed_snapshot.status == "verified":
            signed_ref = load_signed_verification_ref(
                center_root / SIGNED_REF_FILE,
                store_root=store_root,
                expected_seed_snapshot_id=str(record.seed_snapshot_id),
                expected_allowlist_hash=allowlist_sha256(tuple(window.days)),
            )
    readiness = verification_authorization_readiness(
        store_root,
        baseline_profile_name=DEFAULT_PROFILE,
        baseline_section_config_hash=section_hash,
        allowlist=tuple(window.days) if window is not None else None,
    )
    run_envelope = None
    preflight = None
    if readiness.status == "ready" and readiness.evidence_ids:
        from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
            load_verified_envelope,
        )
        from alpha_lab.agents.data_infra.ifvg.search.verification import (  # noqa: PLC0415
            VerificationRunEnvelope,
        )

        try:
            run_envelope = load_verified_envelope(
                store_root, "verification_runs", readiness.evidence_ids[0], VerificationRunEnvelope
            )
        except Exception:  # noqa: BLE001 — the readiness already typed the failure
            run_envelope = None
        if run_envelope is not None:
            preflight = bounded_preflight_state(
                store_root, repo_root, run_envelope, section_hash=section_hash
            )
    return CenterContext(
        roots=purpose_roots,
        store_root=store_root,
        repo_root=repo_root,
        center_root=center_root,
        profile=DEFAULT_PROFILE,
        section_hash=section_hash,
        namespace=namespace,
        record=record,
        receipt_notes=notes,
        shortlist=shortlist,
        window=window,
        window_note=window_note,
        inventory=inventory,
        seed_authorization=seed_authorization,
        seed_receipt=seed_receipt,
        seed_snapshot=seed_snapshot,
        coverage_matrix=coverage_matrix,
        signed_ref=signed_ref,
        readiness=readiness,
        run_envelope=run_envelope,
        preflight=preflight,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Purpose & readiness
# ─────────────────────────────────────────────────────────────────────────────


def _readiness_rows(ctx: CenterContext) -> list[tuple[str, str, str]]:
    window = ctx.window
    if window is not None:
        fixture = (
            "selected",
            " · ".join(window.days) + (f" — {ctx.window_note}" if ctx.window_note else ""),
        )
        covered = bool(
            window.hard_constraints["all_source_partitions_present_and_hash_addressable"]
        )
        partitions = (
            "hash_addressable" if covered else "incomplete",
            f"{len(window.trading_day_refs) * 2} physical partitions from the accepted inventory"
            if covered
            else "a physical partition of the window is not in the accepted inventory",
        )
    else:
        fixture = ("not_selected", ctx.window_note or "no provisional window recorded")
        partitions = ("not_evaluated", "select a window first")
    seed_authorization = (
        (ctx.seed_authorization.status, ctx.seed_authorization.detail)
        if ctx.seed_authorization is not None
        else ("not_evaluated", "select a window first")
    )
    seed_job = (
        (ctx.seed_receipt.status, ctx.seed_receipt.detail)
        if ctx.seed_receipt is not None
        else ("not_evaluated", "select a window first")
    )
    seed = (
        (ctx.seed_snapshot.status, ctx.seed_snapshot.detail)
        if ctx.seed_snapshot is not None
        else ("not_evaluated", "select a window first")
    )
    if ctx.preflight is not None:
        bounded = (ctx.preflight.status, ctx.preflight.detail)
    else:
        bounded = ("not_evaluated", "the preflight runs once a registered run exists")
    return [
        ("Semantic store namespace", ctx.namespace.status, ctx.namespace.detail),
        ("Logical-day fixture", *fixture),
        ("Physical partition coverage", *partitions),
        ("Seed-production authorization", *seed_authorization),
        ("Seed job", *seed_job),
        ("Verified seed", *seed),
        (
            "Final verification authorization (VerificationAuthorizationRef)",
            ctx.readiness.status,
            ctx.readiness.detail,
        ),
        ("Bounded-run readiness", *bounded),
    ]


def _render_readiness(st_module, ctx: CenterContext) -> None:
    st_module.subheader("1 · Purpose & readiness")
    st_module.caption(PURPOSE_DESCRIPTIONS[RunPurpose.IMPLEMENTATION_VERIFICATION])
    for label, status, detail in _readiness_rows(ctx):
        st_module.write(f"{label}: **{status}** — {sanitize_error(detail)}")
    if ctx.namespace.store_namespace_id:
        identity_block(st_module, "store_namespace_id", ctx.namespace.store_namespace_id)
    st_module.caption(
        f"Execution mode: {EXECUTION_MODE_V1} — sequential in V1; effective workers: "
        f"{SUPPORTED_CHILD_WORKERS}. Run scope `verification_5d`; namespace class `test`; "
        "verification_control_flow_gates_v1 only; no research interpretation, no publication."
    )
    for note in ctx.receipt_notes:
        st_module.caption(f"Receipt pickup — {sanitize_error(note)}")
    requirement_set = derive_authorization_requirements("verification_5d", (), None, (), ())
    st_module.markdown("**Authorization requirement (real slice)**")
    for requirement in requirement_set.payload.requirements:
        st_module.write(f"- `{requirement.decision_key}` — {requirement.reason}")
    if ctx.namespace.status != "verified":
        render_empty_state(st_module, "store_namespace_unverified", detail=ctx.namespace.detail)
        cli_escape_hatch(
            st_module,
            "python scripts/ifvg_store_namespace.py init --store-root "
            f"{_rel(ctx.store_root)} --namespace-class test --confirm",
            reason="initialize the verification store explicitly as a test namespace",
        )
    if ctx.window is None and ctx.readiness.status != "ready":
        # the pre-flow state (no window yet): the typed readiness of the final
        # authorization is shown here; once a window exists step 5 owns it
        if ctx.readiness.status in ("missing", "store_unmarked"):
            render_empty_state(
                st_module,
                "verification_authorization_missing",
                detail=f"{ctx.readiness.status}: {ctx.readiness.detail}",
            )
        else:
            render_empty_state(st_module, "authorization_not_ready", detail=ctx.readiness.detail)
    st_module.markdown("**Synthetic fixture (machinery proof)**")
    st_module.caption(
        "A synthetic fixture proves the machinery in the test namespace and needs no "
        "owner authorization; it never becomes verification evidence. The real ≤5-day "
        "slice is prepared in the steps below and never uses a synthetic marker."
    )
    if st_module.button(
        "Start a verification draft",
        key=f"{_START}verify_draft",
        help=help_text("center.start_verification_draft"),
    ):
        from ifvg_study_tab import (  # noqa: PLC0415
            TASK_CARDS,
            request_route,
            start_draft_from_card,
            stash_session_draft,
        )

        card = next(card for card in TASK_CARDS if card.card_id == "verify_implementation")
        try:
            draft = start_draft_from_card(card, ctx.roots)
        except Exception as error:  # noqa: BLE001
            st_module.error(f"Draft could not be created: {sanitize_error(error)}")
        else:
            stash_session_draft(st_module, draft)
            request_route(st_module, "new_study")
            st_module.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Fixture — logical trading days and physical partitions, separately
# ─────────────────────────────────────────────────────────────────────────────


def _window_label(score) -> str:
    return " · ".join(score.days)


def _render_fixture(st_module, ctx: CenterContext) -> None:
    st_module.subheader("2 · Fixture — logical trading-day window")
    shortlist = ctx.shortlist
    if shortlist.status != "available":
        render_empty_state(st_module, "shortlist_unavailable", detail=shortlist.detail)
        cli_escape_hatch(
            st_module,
            "python scripts/ifvg_verification_window_shortlist.py --out-dir "
            f"{_rel(Path(shortlist.source_path).parent)}",
            reason="rebuild the shortlist from already-authorized evidence (no raw source is read)",
        )
        return
    listing = shortlist.shortlist
    st_module.caption(
        f"{shortlist.detail} · generated {shortlist.generated_at_utc or '—'} · ranking "
        "lexicographic in the §5.1 order with no hidden score · owner selection "
        f"{listing.owner_selection} · program allowlist registered: "
        f"{listing.register_program_allowlist_called}"
    )
    identity_block(st_module, "shortlist_id", shortlist.shortlist_id or "")
    st_module.markdown("**Required entries**")
    st_module.table(
        {
            "entry": [entry.label for entry in listing.entries],
            "rank": [entry.rank for entry in listing.entries],
            "window (logical days)": [_window_label(entry.window) for entry in listing.entries],
            "eligible": ["yes" if entry.window.eligible else "no" for entry in listing.entries],
            "seed chain (store days)": [
                entry.window.seed_chain_replay_day_count for entry in listing.entries
            ],
            "failed constraints": [
                ", ".join(entry.window.ineligibility_reasons) or "—" for entry in listing.entries
            ],
        }
    )
    ranked = listing.ranked_windows[:_TOP_N]
    with st_module.expander(
        f"Ranking trace (top {len(ranked)} of {listing.candidate_window_count})"
    ):
        st_module.dataframe(
            [
                {
                    "#": index + 1,
                    "window": _window_label(score),
                    "eligible": score.eligible,
                    **{name: value for name, value in score.rank_components.items()},
                    "failed constraints": ", ".join(score.ineligibility_reasons) or "—",
                }
                for index, score in enumerate(ranked)
            ],
            hide_index=True,
            width="stretch",
        )
    eligible = [score for score in listing.ranked_windows if score.eligible]
    options = [_window_label(score) for score in eligible]
    if not options:
        st_module.warning("No eligible window exists in the shortlist; nothing can be selected.")
    else:
        sanitize_select(st_module, f"{_VC}window", options)
        chosen = st_module.selectbox(
            "Eligible windows (consecutive logical trading days; ineligible windows are listed "
            "above with their failed constraint and cannot be selected)",
            options,
            key=f"{_VC}window",
            help=(
                "The owner's selection is PROVISIONAL and recorded in the center only; the "
                "shortlist document keeps owner_selection = NOT PERFORMED and no allowlist "
                "is registered here."
            ),
        )
        if st_module.button(
            "Record provisional window",
            key=f"{_VC}record_window",
            help="Records the provisional window for the packets below; registers nothing.",
        ):
            score = next(score for score in eligible if _window_label(score) == chosen)
            rank = next(
                (
                    index + 1
                    for index, item in enumerate(listing.ranked_windows)
                    if item.days == score.days
                ),
                None,
            )
            record = ctx.record
            record.store_namespace_id = ctx.namespace.store_namespace_id
            record.provisional_window = {
                "shortlist_id": shortlist.shortlist_id,
                "days": list(score.days),
                "label": f"rank {rank}" if rank is not None else "shortlisted",
                "recorded_at_utc": _utc_now(),
            }
            save_verification_center_record(ctx.center_root, record)
            st_module.rerun()
    window = ctx.window
    if window is None:
        render_empty_state(st_module, "window_not_selected", detail=ctx.window_note)
        return
    st_module.markdown(
        f"**Provisional window:** {_window_label(window)} "
        f"({ctx.record.provisional_window.get('label', '')})"
    )
    if ctx.window_note:
        st_module.warning(ctx.window_note)
    st_module.markdown("**Logical trading days** (the allowlist domain; the 18:00 ET roll)")
    st_module.table(
        {
            "logical trading day": [ref.logical_trading_day for ref in window.trading_day_refs],
            "session open (UTC)": [ref.session_open_ts_utc for ref in window.trading_day_refs],
            "session close (UTC)": [ref.session_close_ts_utc for ref in window.trading_day_refs],
            "lifecycle rows": [row.setup_lifecycle_rows for row in window.coverage],
            "candidates": [row.entry_candidate_rows for row in window.coverage],
            "decisions": [row.eligible_decision_rows for row in window.coverage],
            "executed trades": [row.executed_trade_rows for row in window.coverage],
            "exact verifier targets": [row.exact_verifier_targets for row in window.coverage],
            "audit day covered": [
                "yes" if row.audit_day_covered else "no" for row in window.coverage
            ],
        }
    )
    st_module.markdown(
        "**Physical partitions** (the store-day files each logical day is composed from — "
        "td−1 `prev_utc_date` and td `utc_date`; content hashes from the accepted inventory)"
    )
    st_module.table(
        {
            "logical day": [
                day_ref.logical_trading_day
                for day_ref in window.trading_day_refs
                for _ref in day_ref.ordered_source_partition_refs
            ],
            "physical UTC date": [
                ref.physical_utc_date
                for day_ref in window.trading_day_refs
                for ref in day_ref.ordered_source_partition_refs
            ],
            "partition key": [
                ref.relative_logical_partition_key
                for day_ref in window.trading_day_refs
                for ref in day_ref.ordered_source_partition_refs
            ],
            "source kind": [
                ref.source_kind
                for day_ref in window.trading_day_refs
                for ref in day_ref.ordered_source_partition_refs
            ],
            "content sha256": [
                ref.content_sha256
                for day_ref in window.trading_day_refs
                for ref in day_ref.ordered_source_partition_refs
            ],
        }
    )
    st_module.caption(
        f"Exclusions (read-only): the protected buffer {PROTECTED_BUFFER_DAY} and the sealed "
        f"range from {SEALED_START_DAY} are excluded from every window; the permitted window "
        f"ends {PERMITTED_WINDOW_LAST_DAY}. The seed chain is the canonical STORE-DAY chain "
        f"{window.seed_chain_replay_days[0]} … {window.seed_chain_replay_days[-1]} "
        f"({window.seed_chain_replay_day_count} store days) — a separately authorized "
        "preparation action, never part of the ≤5-day verification evidence footprint."
    )
    if ctx.coverage_matrix is not None:
        identity_block(
            st_module,
            "coverage_matrix_artifact_id (derived from the shortlist rows)",
            ctx.coverage_matrix.coverage_matrix_id,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Seed — packet, registration receipt, the exact job, the verified seed
# ─────────────────────────────────────────────────────────────────────────────


def _render_seed(st_module, ctx: CenterContext) -> None:
    st_module.subheader("3 · Seed — separately authorized seed production")
    window = ctx.window
    if window is None:
        st_module.caption("Select and record a provisional window first (step 2).")
        return
    first_day = str(window.days[0])
    st_module.markdown("**A · The unsigned seed-production packet**")
    st_module.caption(
        "The packet binds the store namespace and head witness, the baseline profile, the "
        "canonical store-day chain through the day before the window, the expected "
        "source-inventory hash and the code identities; its owner fields are placeholders "
        "that fail validation until the owner completes them outside this workspace."
    )
    packet_path = ctx.center_root / SEED_PACKET_JSON
    inventory = ctx.inventory
    if inventory.status != "available":
        st_module.warning(
            f"Accepted inventory unavailable — {sanitize_error(inventory.detail)}; build the "
            "packet with the CLI from the accepted manifest instead."
        )
    quant_lab, strategy_core, identity_note = _code_identities(ctx.repo_root)
    if identity_note:
        st_module.warning(identity_note)
    can_build = (
        inventory.status == "available"
        and ctx.namespace.status == "verified"
        and quant_lab is not None
        and strategy_core is not None
    )
    if st_module.button(
        "Prepare the seed-production packet",
        key=f"{_VC}build_seed_packet",
        disabled=not can_build,
        help="Writes the UNSIGNED packet under the center root; nothing is persisted to a store.",
    ):
        try:
            packet = build_seed_production_packet(
                ctx.store_root,
                baseline_profile_name=ctx.profile,
                resolved_section_config_hash=ctx.section_hash,
                first_intended_verification_day=first_day,
                inventory=inventory.inventory or {},
                quant_lab_source_identity=str(quant_lab),
                strategy_core_commit=str(strategy_core[0]),
                strategy_core_source_identity=str(strategy_core[1]),
            )
        except (SeedProductionAuthorizationError, ValueError) as error:
            st_module.error(f"Packet not built: {sanitize_error(error)}")
        else:
            _write_text(packet_path, json.dumps(packet, indent=2, sort_keys=True) + "\n")
            _write_text(
                ctx.center_root / SEED_PACKET_MD,
                render_seed_production_packet_markdown(
                    packet, title="Seed-production authorization packet (UNSIGNED)"
                ),
            )
            st_module.success(
                f"Unsigned packet written: {packet['seed_chain_replay_day_count']} store days "
                f"({packet['payload']['ordered_seed_chain_replay_days'][0]} … "
                f"{packet['payload']['snapshot_through_day']}), "
                f"{packet['logical_trading_day_count']} logical trading days, first intended "
                f"verification day {packet['payload']['first_intended_verification_day']}."
            )
            st_module.rerun()
    if packet_path.exists():
        st_module.caption(f"Packet on disk: `{_rel(packet_path)}` (and the Markdown twin).")
    cli_escape_hatch(
        st_module,
        "python scripts/ifvg_seed_production.py packet --store-root "
        f"{_rel(ctx.store_root)} --profile-name {ctx.profile} --first-verification-day "
        f"{first_day} --inventory-json {_rel(INVENTORY_MANIFEST_PATH)} "
        "--quant-lab-source-identity <identity> --strategy-core-commit <commit> "
        f"--strategy-core-source-identity <identity> --out-dir {_rel(ctx.center_root)}",
        reason="the equivalent CLI packet build",
    )
    st_module.write(
        "Permitted outputs of the seed run: seed snapshot, access audit, run receipt. "
        "Prohibited outputs: " + ", ".join(PROHIBITED_SEED_OUTPUTS) + "."
    )

    st_module.markdown("**B · Owner signature and registration (outside this workspace)**")
    st_module.caption(
        "The owner completes approved_by / approved_at / effective_from / owner_decision_refs "
        "in a copy of the packet and registers it; the receipt below is picked up on refresh."
    )
    cli_escape_hatch(
        st_module,
        "python scripts/ifvg_seed_production.py register-authorization --store-root "
        f"{_rel(ctx.store_root)} --packet-json "
        f"{_rel(ctx.center_root / 'SEED_PRODUCTION_PACKET.signed.json')} "
        f"--receipt-out {_rel(ctx.center_root / SEED_AUTHORIZATION_RECEIPT)}",
        reason="register the COMPLETED packet through the backend's fail-closed seam",
    )
    authorization = ctx.seed_authorization
    assert authorization is not None
    st_module.write(
        f"Seed-production authorization: **{authorization.status}** — "
        f"{sanitize_error(authorization.detail)}"
    )
    if authorization.status in ("verified", "verified_envelope"):
        identity_block(
            st_module, "seed_production_authorization_id", authorization.authorization_id or ""
        )
        st_module.table(
            {
                "provenance": [authorization.provenance],
                "chain (store days)": [
                    f"{authorization.chain_first_day} … {authorization.chain_last_day} "
                    f"({authorization.chain_day_count})"
                ],
                "first intended verification day": [authorization.first_intended_verification_day],
                "baseline profile": [authorization.baseline_profile_name],
            }
        )
        st_module.markdown("**C · The explicit seed job (a REAL replay; external)**")
        st_module.caption(
            f"Replays the {authorization.chain_day_count}-store-day chain under the "
            "seed_production_explicit_chain_v1 policy; a separately authorized preparation "
            "action with zero verification-evidence days; it writes only the seed snapshot, "
            "the access audit and the run receipt. The receipt is picked up on refresh."
        )
        cli_escape_hatch(
            st_module,
            "python scripts/ifvg_seed_production.py run --store-root "
            f"{_rel(ctx.store_root)} --authorization-id {authorization.authorization_id} "
            f"--profile-name {ctx.profile} --inventory-json {_rel(INVENTORY_MANIFEST_PATH)} "
            f"--receipt-out {_rel(ctx.center_root / SEED_RUN_RECEIPT)}",
            reason="the explicit, separately authorized seed job (never started from this page)",
        )
    else:
        render_empty_state(
            st_module,
            "seed_production_not_authorized",
            detail=f"{authorization.status}: {authorization.detail}",
        )
    if st_module.button(
        "Refresh",
        key=f"{_VC}refresh_seed",
        help=help_text("center.refresh_seed"),
    ):
        st_module.rerun()

    st_module.markdown("**D · The verified seed**")
    receipt = ctx.seed_receipt
    snapshot = ctx.seed_snapshot
    assert receipt is not None and snapshot is not None
    st_module.write(f"Seed job receipt: **{receipt.status}** — {sanitize_error(receipt.detail)}")
    if receipt.status == "verified":
        identity_block(st_module, "seed_production_run_id", receipt.run_id or "")
        st_module.table(
            {
                "provenance": [receipt.provenance],
                "chain days replayed": [receipt.chain_replay_day_count],
                "logical trading days": [receipt.logical_trading_day_count],
                "verification-evidence days": [receipt.verification_evidence_footprint_days],
                "access audit sha256": [receipt.access_audit_sha256],
            }
        )
    st_module.write(f"Seed snapshot: **{snapshot.status}** — {sanitize_error(snapshot.detail)}")
    if snapshot.status == "verified":
        identity_block(st_module, "seed_snapshot_id", snapshot.seed_snapshot_id or "")
        identity_block(st_module, "seed_hash", snapshot.seed_hash or "")
        st_module.table(
            {
                "profile": [snapshot.profile_name],
                "snapshot through": [snapshot.snapshot_through_day],
                "first replay day": [snapshot.first_replay_day],
                "chain date count": [snapshot.chain_date_count],
            }
        )
    else:
        render_empty_state(
            st_module, "seed_missing", detail=f"{snapshot.status}: {snapshot.detail}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4 · Final authorization — the packet and the owner's completed reference
# ─────────────────────────────────────────────────────────────────────────────


def _display_inventory_hash(window) -> str:
    """A DISPLAY-ONLY digest of the window's partition refs (never registered)."""

    return canonical_contract_sha256(
        {
            "window": list(window.days),
            "partitions": [
                [
                    ref.physical_utc_date,
                    ref.relative_logical_partition_key,
                    ref.source_kind,
                    ref.content_sha256,
                ]
                for day_ref in window.trading_day_refs
                for ref in day_ref.ordered_source_partition_refs
            ],
        }
    )


def _render_final_authorization(st_module, ctx: CenterContext) -> None:
    st_module.subheader("4 · Final authorization — VerificationAuthorizationRef")
    window = ctx.window
    snapshot = ctx.seed_snapshot
    if window is None or snapshot is None:
        st_module.caption("Select and record a provisional window first (step 2).")
        return
    if snapshot.status != "verified":
        render_empty_state(
            st_module,
            "seed_missing",
            detail="the final verification packet is buildable only after a verified, "
            "profile-matching seed continuous with the window exists",
        )
        return
    packet_path = ctx.center_root / VERIFICATION_PACKET_JSON
    st_module.markdown("**E · The unsigned final packet**")
    if st_module.button(
        "Prepare the final verification packet",
        key=f"{_VC}build_final_packet",
        help=(
            "Buildable only after the verified seed: the corrected logical allowlist, the "
            "shortlist and coverage-matrix ids, the seed's provenance and the blank owner "
            "fields. Nothing is persisted to a store."
        ),
    ):
        try:
            packet = build_verification_authorization_packet(
                ctx.store_root,
                seed_snapshot_id=str(snapshot.seed_snapshot_id),
                logical_window=tuple(window.days),
                resolved_section_config_hash=ctx.section_hash,
                baseline_profile_name=ctx.profile,
                shortlist_id=str(ctx.shortlist.shortlist_id),
                coverage_matrix_artifact_id=ctx.coverage_matrix.coverage_matrix_id,
                expected_source_inventory_hash=_display_inventory_hash(window),
                seed_production_run_id=ctx.record.seed_production_run_id,
                inventory=ctx.inventory.inventory,
            )
        except (SeedProductionAuthorizationError, ValueError) as error:
            st_module.error(f"Packet not built: {sanitize_error(error)}")
        else:
            _write_text(packet_path, json.dumps(packet, indent=2, sort_keys=True) + "\n")
            _write_text(
                ctx.center_root / VERIFICATION_PACKET_MD,
                render_verification_packet_markdown(
                    packet, title="Final verification authorization packet (UNSIGNED)"
                ),
            )
            _write_text(
                ctx.center_root / COVERAGE_MATRIX_JSON,
                json.dumps(ctx.coverage_matrix.model_dump(mode="json"), indent=2, sort_keys=True)
                + "\n",
            )
            st_module.success("Unsigned final packet written (owner fields blank).")
            st_module.rerun()
    if packet_path.exists():
        st_module.caption(
            f"Packet on disk: `{_rel(packet_path)}` (Markdown twin and the derived coverage "
            "matrix beside it). Complete `verification_authorization_ref` — approved_by, "
            "approved_at, content_hash — in a copy saved as "
            f"`{_rel(ctx.center_root / SIGNED_REF_FILE)}`."
        )
    st_module.markdown("**F · The owner's completed reference (validated, never persisted here)**")
    signed = ctx.signed_ref
    if signed is None:
        return
    st_module.write(f"Completed reference: **{signed.status}** — {sanitize_error(signed.detail)}")
    if signed.status == "valid" and signed.ref is not None:
        identity_block(st_module, "content_hash", signed.ref.content_hash)
        identity_block(st_module, "approved_allowlist_hash", signed.ref.approved_allowlist_hash)
    elif signed.status in ("file_missing", "unsigned"):
        render_empty_state(
            st_module, "final_authorization_unsigned", detail=f"{signed.status}: {signed.detail}"
        )
    else:
        render_empty_state(
            st_module, "authorization_not_ready", detail=f"{signed.status}: {signed.detail}"
        )
    if st_module.button(
        "Validate the completed reference",
        key=f"{_VC}validate_ref",
        help=help_text("center.validate_reference"),
    ):
        st_module.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 5 · Review & run — freeze from the signed ref, register the run, preflight
# ─────────────────────────────────────────────────────────────────────────────


def _exact_baseline_charter(ctx: CenterContext, bundle):
    """The exact-baseline verification charter (mirrors the wizard's assembly
    for the verification scope; no challengers, no firms, verification gates)."""

    from ifvg_study_wizard import (  # noqa: PLC0415
        _PROP_GATE_DEFAULTS,
        _ROBUSTNESS_GATE_DEFAULTS,
        _commit_of,
        _strategy_core_root,
    )

    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (  # noqa: PLC0415
        SEARCH_AXIS_REGISTRY_V1,
        AxisClassification,
        registry_sha256,
    )
    from alpha_lab.agents.data_infra.ifvg.search.charter import (  # noqa: PLC0415
        CostPolicy,
        DatePolicy,
        ObjectivePolicy,
        ResolvedPropGateThresholds,
        ResolvedRobustnessGateThresholds,
        ResolvedStrategyGateThresholds,
        SearchCharterPayload,
        SearchMode,
        SimulationProtocol,
    )

    window = ctx.window
    return SearchCharterPayload(
        search_mode=SearchMode.SINGLE_CONFIGURATION,
        baseline_profile_name=ctx.profile,
        baseline_section_config_hash=ctx.section_hash,
        axes={},
        locked_invariants_registry_sha256=registry_sha256(),
        measured_only_fields=tuple(
            key
            for key, spec in SEARCH_AXIS_REGISTRY_V1.items()
            if spec.classification is AxisClassification.MEASUREMENT_ONLY
        ),
        blocked_capabilities=tuple(
            key
            for key, spec in SEARCH_AXIS_REGISTRY_V1.items()
            if spec.classification is AxisClassification.BLOCKED
        ),
        authorized_firm_contract_ids=(),
        authorized_risk_policy_ids=(),
        authorized_withdrawal_policy_ids=(),
        objective_policy=ObjectivePolicy(
            feasibility_gates=ResolvedStrategyGateThresholds(min_session_stability_score=0.5),
            prop_feasibility_gates=ResolvedPropGateThresholds(**_PROP_GATE_DEFAULTS),
            robustness_gates=ResolvedRobustnessGateThresholds(**_ROBUSTNESS_GATE_DEFAULTS),
            pareto_objectives=("net_expectancy_r",),
            lexicographic_tie_breaks=("profit_factor", "max_drawdown_r", "core_replay_id"),
        ),
        date_policy=DatePolicy(
            replay_dates=tuple(window.days),
            warmup_dates=(),
            access_policy_id="verification_fixed_allowlist_max5_v1",
        ),
        simulation_protocol=SimulationProtocol(
            modes=("historical_closed_trade",),
            stress_scenario_ids=(),
            trade_path_capability_policy_id="path_capability_policy_v1",
            clock_policy_id="historical_calendar_clock_v1",
        ),
        max_child_count=1,
        seed=7,
        cost_policy=CostPolicy(),
        strategy_core_commit=_commit_of(_strategy_core_root()),
        quant_lab_commit=_commit_of(_REPO_ROOT),
        source_artifact_ids=(),
        owner_authorization=bundle,
    )


def _freeze_and_register(st_module, ctx: CenterContext) -> None:
    """The explicit handler: charter (from the SIGNED ref) → pipeline spec →
    the registered run. Persists into the verified test store only; launches
    nothing; never calls register_program_allowlist."""

    from ifvg_pipeline_tab import _assemble_pipeline_spec  # noqa: PLC0415

    from alpha_lab.agents.data_infra.ifvg.search.catalog import (
        append_catalog_event,  # noqa: PLC0415
    )
    from alpha_lab.agents.data_infra.ifvg.search.charter import (  # noqa: PLC0415
        CharterValidationError,
        SearchCharterEnvelope,
        save_charter,
        validate_charter,
    )
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
        PipelineSemanticIdentity,
        assert_stage_plan_launchable,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (
        save_or_reuse_envelope,  # noqa: PLC0415
    )

    signed = ctx.signed_ref
    assert signed is not None and signed.ref is not None
    requirement_set = derive_authorization_requirements("verification_5d", (), None, (), ())
    bundle = verification_bundle_from_signed_ref(signed.ref, requirement_set)
    try:
        payload = _exact_baseline_charter(ctx, bundle)
        if "unknown" in (payload.strategy_core_commit, payload.quant_lab_commit):
            raise CharterValidationError(
                "source-commit provenance could not be resolved; freezing is refused rather "
                "than stamping 'unknown' into the immutable charter"
            )
        validate_charter(payload, as_of_utc=_utc_now(), store_root=ctx.store_root)
        charter = SearchCharterEnvelope.from_payload(payload)
        spec = _assemble_pipeline_spec(payload, charter.search_id, fields={"full_plan": False})
        assert_stage_plan_launchable(spec, store_root=ctx.store_root, run_scope="verification_5d")
        save_charter(ctx.store_root, charter)
        semantic = PipelineSemanticIdentity.from_payload(spec)
        save_or_reuse_envelope(ctx.store_root, "pipeline_specs", semantic)
        run = register_verification_run(
            ctx.store_root,
            ref=signed.ref,
            pipeline_semantic_id=semantic.pipeline_semantic_id,
            allowlist=tuple(ctx.window.days),
            seed_snapshot_id=str(ctx.record.seed_snapshot_id),
            baseline_profile_name=ctx.profile,
            baseline_section_config_hash=ctx.section_hash,
            coverage_matrix_artifact_id=signed.ref.coverage_matrix_artifact_id,
            display_name=f"verification run · {' · '.join(ctx.window.days)}",
        )
    except CharterValidationError as error:
        st_module.error(f"Charter cannot freeze (fail-closed): {sanitize_error(error)}")
        return
    except PermissionError as error:
        st_module.error(f"Registration refused (fail-before-path): {sanitize_error(error)}")
        return
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.error(f"Freeze failed: {sanitize_error(error)}")
        return
    try:
        append_catalog_event(
            ctx.store_root,
            kind="purpose",
            artifact_id=charter.search_id,
            payload={
                "purpose": RunPurpose.IMPLEMENTATION_VERIFICATION.value,
                "evidence_class": "real",
                "run_scope": "verification_5d",
                "namespace_class": "test",
            },
        )
    except Exception as error:  # noqa: BLE001 — annotation is non-semantic
        st_module.caption(f"purpose annotation not recorded: {sanitize_error(error)}")
    st_module.success(
        "Verification charter frozen from the signed reference, pipeline spec persisted, "
        "run registered (verified reuse on repeat)."
    )
    identity_block(st_module, "search_id (charter)", charter.search_id)
    identity_block(st_module, "pipeline_semantic_id", semantic.pipeline_semantic_id)
    identity_block(st_module, "verification_run_id", run.verification_run_id)
    st_module.rerun()


def _render_review_run(st_module, ctx: CenterContext) -> None:
    st_module.subheader("5 · Review & run — the bounded ≤5-day verification")
    window = ctx.window
    if window is None:
        st_module.caption("Select and record a provisional window first (step 2).")
        return
    st_module.table(
        {
            "exact baseline": [f"{ctx.profile} · section {ctx.section_hash[:16]}…"],
            "allowlist (logical days)": [" · ".join(window.days)],
            "physical partitions": [
                ", ".join(
                    ref.physical_utc_date
                    for day_ref in window.trading_day_refs
                    for ref in day_ref.ordered_source_partition_refs
                )
            ],
            "seed": [ctx.record.seed_snapshot_id or "— (not recorded)"],
            "drive": ["dual drive (audit-disabled and audit-enabled compared by neutrality)"],
            "release-control checks": [
                "verification_control_flow_gates_v1 · R1 baseline gate report · bounded "
                "release control-flow report"
            ],
            "artifacts": [
                "core replays · executed-trade tables · search results (verification-only stamps)"
            ],
            "execution": [f"{EXECUTION_MODE_V1} · effective workers {SUPPORTED_CHILD_WORKERS}"],
        }
    )
    readiness = ctx.readiness
    st_module.write(
        f"Final verification authorization (VerificationAuthorizationRef): "
        f"**{readiness.status}** — {sanitize_error(readiness.detail)}"
    )
    if readiness.status != "ready":
        signed = ctx.signed_ref
        if signed is not None and signed.status == "valid":
            st_module.caption(
                "The completed reference validated: freezing the exact-baseline charter from "
                "it, persisting the pipeline spec and registering the run makes the readiness "
                "ready. Nothing launches here."
            )
            if st_module.button(
                "Freeze the verification charter and register the run",
                key=f"{_VC}freeze_register",
                type="primary",
                help=(
                    "Charter (bundle derived from the SIGNED reference, never a synthetic "
                    "marker) → pipeline spec → VerificationRunEnvelope; verified reuse on repeat."
                ),
            ):
                _freeze_and_register(st_module, ctx)
        elif readiness.status in ("missing", "store_unmarked"):
            render_empty_state(
                st_module,
                "verification_authorization_missing",
                detail=f"{readiness.status}: {readiness.detail}",
            )
        else:
            render_empty_state(
                st_module,
                "authorization_not_ready",
                detail=f"{readiness.status}: {readiness.detail}",
            )
        return
    run = ctx.run_envelope
    if run is not None:
        identity_block(st_module, "verification_run_id", run.verification_run_id)
        identity_block(st_module, "pipeline_semantic_id", run.payload.pipeline_semantic_id)
        try:
            from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
                PipelineSemanticIdentity,
            )
            from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
                load_verified_envelope,
            )

            semantic = load_verified_envelope(
                ctx.store_root,
                "pipeline_specs",
                run.payload.pipeline_semantic_id,
                PipelineSemanticIdentity,
            )
            identity_block(
                st_module, "search_id (charter)", semantic.payload.search_charter_id or ""
            )
        except Exception as error:  # noqa: BLE001
            st_module.caption(f"pipeline spec not loadable: {sanitize_error(error)}")
    preflight = ctx.preflight
    if preflight is None:
        st_module.caption("The preflight could not be evaluated for the registered run.")
        return
    if preflight.status == "passed" and preflight.record is not None:
        st_module.success(f"Preflight passed — {sanitize_error(preflight.detail)}")
        st_module.table(
            {
                "check": list(preflight.record.checks.keys()),
                "result": ["✓ passed" for _ in preflight.record.checks],
            }
        )
        st_module.caption(
            f"Logical days {' · '.join(preflight.record.logical_trading_days)} → physical "
            f"partitions {', '.join(preflight.record.physical_partition_dates)}; seed "
            f"{preflight.record.seed_snapshot_id[:12]}…; allowlist hash "
            f"{preflight.record.allowlist_hash[:12]}…"
        )
        st_module.markdown("**The exact bounded run (external; starts only from this command)**")
        pipeline_state_root = _pipeline_state_root()
        cli_escape_hatch(
            st_module,
            "python scripts/ifvg_bounded_verification.py run --store-root "
            f"{_rel(ctx.store_root)} --repo-root {_rel(ctx.repo_root)} --state-root "
            f"{_rel(pipeline_state_root)} --evidence-dir "
            f"{_rel(ctx.center_root / EVIDENCE_DIR_NAME)}",
            reason=(
                "the REAL ≤5-day bounded verification: two attempts (fresh, then verified "
                "reuse), the R1 baseline gate report and the bounded release control-flow "
                "report; the program allowlist is registered by the runner only"
            ),
        )
    else:
        render_empty_state(
            st_module,
            "preflight_refused",
            detail=f"{preflight.reason or preflight.status}: {preflight.detail}",
        )
    if st_module.button(
        "Run the preflight again",
        key=f"{_VC}preflight_again",
        help=help_text("center.preflight_again"),
    ):
        st_module.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 6 · Monitor — only the resolved seed / verification stages; no Publish
# ─────────────────────────────────────────────────────────────────────────────


def _pipeline_state_root() -> Path:
    import ifvg_pipeline_tab as pipeline_tab  # noqa: PLC0415

    return Path(pipeline_tab.PIPELINE_STATE_ROOT)


def _render_monitor(st_module, ctx: CenterContext) -> None:
    st_module.subheader("6 · Monitor — seed and verification stages")
    receipt = ctx.seed_receipt
    seed_row = {
        "Stage": "Seed production (separately authorized preparation)",
        "Status": (
            "✓ Verified seed"
            if receipt is not None and receipt.status == "verified"
            else f"· {receipt.status}"
            if receipt is not None
            else "· not evaluated"
        ),
        "Explanation": receipt.detail if receipt is not None else "select a window first",
        "Stage result": (receipt.run_id or "")[:12] if receipt is not None else "",
    }
    run = ctx.run_envelope
    if run is None:
        st_module.table([seed_row])
        st_module.caption(
            "No registered verification run yet: the verification stages appear once the "
            "run is registered (step 5)."
        )
        return
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
        read_pipeline_state,  # noqa: PLC0415
    )

    pipeline_id = run.payload.pipeline_semantic_id
    state = read_pipeline_state(_pipeline_state_root(), pipeline_id)
    rows = [seed_row, *(row.as_row() for row in verification_stage_rows(state))]
    st_module.table(rows)
    if state is None:
        st_module.caption(
            "No pipeline state for this run yet — the bounded run has not been started "
            "(step 5 shows the exact command)."
        )
    else:
        attempts = list(state.get("attempts") or ())
        if attempts:
            latest = dict(attempts[-1])
            st_module.caption(
                f"Attempt {len(attempts)} started {latest.get('started_at', '—')} · execution "
                f"mode {latest.get('execution_mode', EXECUTION_MODE_V1)} · effective workers "
                f"{latest.get('effective_workers', SUPPORTED_CHILD_WORKERS)}"
            )
        else:
            st_module.caption(
                f"Execution mode {EXECUTION_MODE_V1} · effective workers {SUPPORTED_CHILD_WORKERS}"
            )
        publication = dict(state.get("publication") or {})
        if publication.get("pipeline_result_id"):
            identity_block(
                st_module,
                "pipeline_result_id (verification-only stamps)",
                str(publication["pipeline_result_id"]),
            )
        counters = [
            child.get("access_audit")
            for child in state.get("children") or ()
            if isinstance(child, Mapping) and child.get("access_audit")
        ]
        if counters:
            st_module.markdown("**Access counters** (from the children's persisted access audits)")
            st_module.json(counters[0] if len(counters) == 1 else counters)
        else:
            st_module.caption(
                "Access counters: no access-audit evidence persisted yet (never assumed zero)."
            )
    summary = verification_evidence_summary(ctx.center_root / EVIDENCE_DIR_NAME)
    st_module.markdown("**Gate evidence** (the evidence folder the bounded runner writes)")
    if not summary["present"]:
        st_module.caption(
            f"No evidence folder at `{_rel(ctx.center_root / EVIDENCE_DIR_NAME)}` yet."
        )
    else:
        for name, report in summary["reports"].items():
            st_module.write(
                f"- `{name}`: passed = **{report.get('passed')}**"
                + (f" · gates {', '.join(report['gate_ids'])}" if report.get("gate_ids") else "")
            )
    st_module.caption(
        "Exact verifier links resolve from the Replay / Verifier tab for the verified pair "
        "once the replay-chart artifact of the run exists (S04). This surface has no Publish "
        "route: verification results are never activated into the research catalog."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def render_verification_center(st_module=st, *, roots: Mapping[str, Any]) -> None:
    st_module.subheader("Verification Center")
    verification_badge(st_module)
    try:
        ctx = _resolve(roots)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.error(f"Verification Center could not resolve its state: {sanitize_error(error)}")
        return
    _render_readiness(st_module, ctx)
    _render_fixture(st_module, ctx)
    _render_seed(st_module, ctx)
    _render_final_authorization(st_module, ctx)
    _render_review_run(st_module, ctx)
    _render_monitor(st_module, ctx)
