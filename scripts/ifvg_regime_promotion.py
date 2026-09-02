"""Regime promotion CLI (R6.1 §6.E) — ``chain``, ``propose``, ``promote``.

* ``chain``   — walk one promotion decision back to its first decision
  (exact-ID verified store loads; never a "latest" lookup).
* ``propose`` — write the owner-decision PROPOSAL markdown for one exact
  protocol + assessment (decisions 25/28/29/30 with the pinned KMeans
  parameter snapshot); placeholders FAIL validation, so a draft can never be
  persisted by accident — nothing here is an authorization.
* ``promote`` — persist ONE promotion decision over its verified evidence:
  ``--to stratification_ready`` (the structural gates — coverage gates AND
  an OOS assignment, D6 — are re-checked by ``persist_regime_promotion``;
  the decision time is the previous decision's, so the result reproduces
  S10's own deterministic decision id and is a reuse) or
  ``--to feature_eligible`` (the protocol, the verified owner-decision
  artifact, and the assessment are loaded and verified by
  ``persist_regime_promotion`` under ``--run-scope``; synthetic provenance
  is lawful only in ``synthetic_fixture``, and that scope only in a test
  namespace). ``decided_at`` is NEVER a caller string or the wall clock: it
  is derived from verified artifacts — the previous decision's ``decided_at``
  and, for a ratified status, the owner artifact's ``effective_from``
  (whichever is later; the chain is monotone). ``model_feature`` is refused
  by the store and the CLI alike with one exact text: it requires the
  activated IFVG_REGIME_CONTEXT_V1 block and a completed controlled feature
  study.

Importing this module launches nothing; every store access is an exact-ID
verified load; a refusal (authorization, evidence, store integrity, lock
timeout, filesystem error) prints its sanitized reason and exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (  # noqa: E402
    MODEL_FEATURE_PROMOTION_REFUSAL,
)

__all__ = ["MODEL_FEATURE_REFUSAL", "build_parser", "main"]

#: One exact text, shared with ``persist_regime_promotion`` (S5).
MODEL_FEATURE_REFUSAL = MODEL_FEATURE_PROMOTION_REFUSAL
_TARGETS = ("stratification_ready", "feature_eligible", "model_feature")
_RUN_SCOPES = ("synthetic_fixture", "verification_5d", "full_authorized_development")


def _instant(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _emit(payload: dict) -> None:
    print(json.dumps(payload, sort_keys=True))


def _chain(root: Path, decision_id: str) -> list[dict]:
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (  # noqa: PLC0415
        load_regime_promotion,
    )
    from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (  # noqa: PLC0415
        OwnerDecisionRefusalError,
        load_owner_decision,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
        SearchStoreError,
    )

    rows: list[dict] = []
    seen: set[str] = set()
    cursor: str | None = decision_id
    while cursor is not None:
        if cursor in seen:
            raise ValueError("the promotion chain loops; refusing")
        seen.add(cursor)
        decision = load_regime_promotion(root, cursor)
        payload = decision.payload
        owner_evidence: str | None = None
        if payload.owner_ratification_ref is not None:
            # S3: a ratified step is shown with its owner artifact's exact-ID
            # verified-load outcome in THIS root (a synthetic artifact in the
            # research namespace, or a tampered/missing one, is "refused").
            try:
                owner = load_owner_decision(root, payload.owner_ratification_ref)
            except (SearchStoreError, OwnerDecisionRefusalError) as error:
                owner_evidence = "refused: " + str(error).splitlines()[0][:300]
            else:
                owner_evidence = f"verified ({owner.payload.provenance})"
        rows.append(
            {
                "regime_promotion_decision_id": cursor,
                "resolved_regime_protocol_id": payload.resolved_regime_protocol_id,
                "role": payload.role.value,
                "status": payload.status.value,
                "previous_status": payload.previous_status.value,
                "previous_decision_ref": payload.previous_decision_ref,
                "capability_assessment_ref": payload.capability_assessment_ref,
                "owner_ratification_ref": payload.owner_ratification_ref,
                "owner_evidence": owner_evidence,
                "decided_at": payload.decided_at,
            }
        )
        cursor = payload.previous_decision_ref
    rows.reverse()  # oldest first
    return rows


def _propose(
    root: Path, protocol_id: str, assessment_id: str, out: Path, transition: str
) -> dict:
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (  # noqa: PLC0415
        load_regime_assessment,
        load_regime_protocol,
    )
    from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (  # noqa: PLC0415
        build_owner_decision_proposal,
        render_owner_decision_proposal_markdown,
    )

    protocol = load_regime_protocol(root, protocol_id)
    assessment = load_regime_assessment(root, assessment_id)
    if assessment.payload.resolved_regime_protocol_id != protocol.resolved_regime_protocol_id:
        raise ValueError("the assessment does not belong to the protocol")
    # HARDENING-BACKEND §4.1: the proposal names the target store's VERIFIED
    # semantic namespace when the store is marked (else the owner placeholder).
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (  # noqa: PLC0415
        StoreNamespaceError,
        load_store_namespace,
    )

    try:
        store_namespace_id: str | None = load_store_namespace(root).store_namespace_id
    except StoreNamespaceError:
        store_namespace_id = None
    proposal = build_owner_decision_proposal(
        protocol, assessment, transition=transition, store_namespace_id=store_namespace_id
    )
    title = (
        "PROPOSAL — owner decisions 25/28/29/30 for regime protocol "
        f"{protocol_id[:12]}… (nothing here is an authorization)"
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_owner_decision_proposal_markdown(proposal, title=title),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": "proposal_written",
        "authorization": "none — PROPOSAL; placeholders fail validation",
        "path": str(out),
        "resolved_regime_protocol_id": protocol_id,
        "capability_assessment_id": assessment_id,
        "transition": transition,
    }


def _promote(
    root: Path,
    *,
    protocol_id: str,
    assessment_id: str,
    to: str,
    previous_decision_id: str | None,
    owner_decision_id: str | None,
    run_scope: str,
) -> dict:
    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (  # noqa: PLC0415
        RegimePromotionDecision,
        RegimePromotionDecisionEnvelope,
        RegimeRole,
        RegimeStatus,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (  # noqa: PLC0415
        load_regime_promotion,
        persist_regime_promotion,
    )
    from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (  # noqa: PLC0415
        load_owner_decision,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
        SearchStoreError,
    )

    if to == "model_feature":
        raise PermissionError(MODEL_FEATURE_REFUSAL)
    target = RegimeStatus(to)
    if previous_decision_id is None:
        raise ValueError(
            f"promotion to {to} requires --previous-decision-id (the ladder is a chain "
            "of persisted decisions; the first decision is derived by the pipeline's S10)"
        )
    if target is RegimeStatus.FEATURE_ELIGIBLE and not owner_decision_id:
        raise ValueError(
            "promotion to feature_eligible requires --owner-decision-id (the verified "
            "owner-decision artifact id; a bare reference is not evidence)"
        )
    previous = load_regime_promotion(root, previous_decision_id)
    if previous.payload.resolved_regime_protocol_id != protocol_id:
        raise ValueError("the previous decision names a different regime protocol")
    # decided_at is derived from VERIFIED artifacts only (never the wall clock,
    # never a caller string): the previous decision's decided_at — which for
    # stratification_ready reproduces S10's deterministic decision exactly —
    # and, for a ratified status, the owner artifact's effective_from
    # (whichever is later; the store enforces the monotone chain).
    decided_at = previous.payload.decided_at
    decided_at_source = "previous_decision.decided_at"
    if target is RegimeStatus.FEATURE_ELIGIBLE:
        try:
            owner = load_owner_decision(root, str(owner_decision_id))
        except SearchStoreError as error:
            raise ValueError(
                "--owner-decision-id is not a verified owner-decision artifact of this store "
                "(a bare 64-hex reference is not evidence)"
            ) from error
        if _instant(owner.payload.effective_from) > _instant(decided_at):
            decided_at = owner.payload.effective_from
            decided_at_source = "owner_decision.effective_from"
    role = (
        RegimeRole.FEATURE_GENERATOR
        if target is RegimeStatus.FEATURE_ELIGIBLE
        else RegimeRole.STRATIFICATION_ONLY
    )
    envelope = RegimePromotionDecisionEnvelope.from_payload(
        RegimePromotionDecision(
            resolved_regime_protocol_id=protocol_id,
            role=role,
            status=target,
            previous_status=previous.payload.status,
            previous_decision_ref=previous_decision_id,
            capability_assessment_ref=assessment_id,
            owner_ratification_ref=(
                owner_decision_id if target is RegimeStatus.FEATURE_ELIGIBLE else None
            ),
            decided_at=decided_at,
        )
    )
    _stored, reused = persist_regime_promotion(root, envelope, run_scope=run_scope)
    return {
        "status": "reused" if reused else "persisted",
        "regime_promotion_decision_id": envelope.regime_promotion_decision_id,
        "regime_status": target.value,
        "role": role.value,
        "previous_decision_id": previous_decision_id,
        "owner_decision_artifact_id": owner_decision_id,
        "run_scope": run_scope,
        "decided_at": decided_at,
        "decided_at_source": decided_at_source,
        "note": (
            "a promotion changes no protocol or fit identity; feature_eligible carries "
            "no research weight until a model-bearing study freezes this exact id"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ifvg_regime_promotion",
        description="R6.1 regime promotion: chain / propose / promote (exact-ID stores).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    chain = sub.add_parser("chain", help="walk a promotion decision back to its first decision")
    chain.add_argument("--store-root", required=True)
    chain.add_argument("--decision-id", required=True)
    propose = sub.add_parser("propose", help="write the owner-decision PROPOSAL markdown")
    propose.add_argument("--store-root", required=True)
    propose.add_argument("--protocol-id", required=True)
    propose.add_argument("--assessment-id", required=True)
    propose.add_argument("--out", required=True)
    propose.add_argument(
        "--transition",
        default="stratification_ready->feature_eligible",
        choices=("stratification_ready->feature_eligible", "feature_eligible->model_feature"),
    )
    promote = sub.add_parser("promote", help="persist one promotion decision")
    promote.add_argument("--store-root", required=True)
    promote.add_argument("--protocol-id", required=True)
    promote.add_argument("--assessment-id", required=True)
    promote.add_argument("--to", required=True, choices=_TARGETS)
    promote.add_argument("--previous-decision-id", default=None)
    promote.add_argument("--owner-decision-id", default=None)
    promote.add_argument("--run-scope", default="full_authorized_development", choices=_RUN_SCOPES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.store_root)
    try:
        if args.command == "chain":
            _emit({"chain": _chain(root, args.decision_id)})
        elif args.command == "propose":
            _emit(
                _propose(
                    root, args.protocol_id, args.assessment_id, Path(args.out), args.transition
                )
            )
        else:
            _emit(
                _promote(
                    root,
                    protocol_id=args.protocol_id,
                    assessment_id=args.assessment_id,
                    to=args.to,
                    previous_decision_id=args.previous_decision_id,
                    owner_decision_id=args.owner_decision_id,
                    run_scope=args.run_scope,
                )
            )
    except (PermissionError, ValueError, LookupError, OSError, TimeoutError) as error:
        # PermissionError / FileExistsError / lock timeouts are OSError or
        # TimeoutError subclasses; SearchStoreError is a ValueError — every
        # refusal prints one sanitized line, never a traceback.
        reason = str(error).splitlines()[0][:800] if str(error) else type(error).__name__
        _emit({"status": "refused", "reason": reason})
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
