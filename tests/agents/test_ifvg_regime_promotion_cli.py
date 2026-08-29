"""R6.1 — the regime promotion CLI (plan §6.E): importing launches nothing;
``chain`` / ``propose`` / ``promote`` are exact-ID store actions;
``model_feature`` is refused with the exact text; FEATURE_ELIGIBLE persists
only over a verified owner-decision artifact in a lawful run scope."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import ifvg_regime_promotion as cli  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.ml.regime_executor import (  # noqa: E402
    execute_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (  # noqa: E402
    load_regime_promotion,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_study import (  # noqa: E402
    RegimeStudyRequest,
    build_s10_decisions,
)
from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (  # noqa: E402
    synthetic_owner_decision_fixture,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (  # noqa: E402
    REGIME_INPUT_FEATURES,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_observation_source import (  # noqa: E402
    persisted_candidate_source,
)

_SOURCE = Path(cli.__file__).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    root = tmp_path_factory.mktemp("regime_cli")
    source = persisted_candidate_source(root)
    result = execute_regime_protocol(
        root,
        protocol=source.protocol,
        observation_source=source.source_ref,
        fold_set_artifact_id=source.fold_set_envelope.fold_set_artifact_id,
        bootstrap_refits=2,
    )
    request = RegimeStudyRequest(
        input_feature_bundle_key="B0_CORE",
        resolved_input_features=REGIME_INPUT_FEATURES,
        stratified_reporting_requested=True,
        comparison_classes_requested=("cohort_descriptive",),
    )
    decisions = build_s10_decisions(
        root,
        protocol=result.protocol,
        assessment=result.run.assessment,
        request=request,
        decided_at="2026-08-28T12:00:00+00:00",
    )
    return {
        "root": root,
        "protocol_id": result.protocol.resolved_regime_protocol_id,
        "assessment_id": result.regime_capability_assessment_id,
        "protocol": result.protocol,
        "assessment": result.run.assessment,
        "first": decisions[0],
        "ready": decisions[-1],
    }


def _run(args: list[str], capsys) -> tuple[int, dict]:
    code = cli.main(args)
    out = capsys.readouterr().out.strip().splitlines()[-1]
    return code, json.loads(out)


def test_import_launches_nothing() -> None:
    tree = ast.parse(_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if "persist_regime_promotion" in name or "main(" in name:
                assert getattr(node, "col_offset", 0) > 0
    assert "if __name__" in _SOURCE
    assert "subprocess" not in _SOURCE
    assert "advisory" not in _SOURCE


def test_chain_walks_exact_decisions_and_refuses_unknown_ids(lane, capsys) -> None:
    ready_id = lane["ready"].regime_promotion_decision_id
    code, payload = _run(
        ["chain", "--store-root", str(lane["root"]), "--decision-id", ready_id], capsys
    )
    assert code == 0
    chain = payload["chain"]
    assert [row["status"] for row in chain] == ["descriptive_only", "stratification_ready"]
    assert chain[0]["previous_decision_ref"] is None
    assert chain[1]["previous_decision_ref"] == lane["first"].regime_promotion_decision_id
    code, payload = _run(
        ["chain", "--store-root", str(lane["root"]), "--decision-id", "f" * 64], capsys
    )
    assert code == 2 and payload["status"] == "refused"


def test_propose_writes_a_draft_that_is_not_an_authorization(lane, tmp_path, capsys) -> None:
    out = tmp_path / "drafts" / "proposal.md"
    code, payload = _run(
        [
            "propose",
            "--store-root",
            str(lane["root"]),
            "--protocol-id",
            lane["protocol_id"],
            "--assessment-id",
            lane["assessment_id"],
            "--out",
            str(out),
        ],
        capsys,
    )
    assert code == 0 and payload["status"] == "proposal_written"
    text = out.read_text(encoding="utf-8")
    assert "PROPOSAL — nothing here is an authorization" in text
    assert "<OWNER_TO_FILL>" in text
    assert "| 25 | algorithm_key | `kmeans_v1` |" in text
    assert "| 29 | fixed_cluster_count | `3` |" in text
    # a mismatched assessment refuses before writing anything
    code, payload = _run(
        [
            "propose",
            "--store-root",
            str(lane["root"]),
            "--protocol-id",
            lane["protocol_id"],
            "--assessment-id",
            "9" * 64,
            "--out",
            str(tmp_path / "never.md"),
        ],
        capsys,
    )
    assert code == 2 and not (tmp_path / "never.md").exists()


def test_model_feature_is_refused_with_the_exact_text(lane, capsys) -> None:
    code, payload = _run(
        [
            "promote",
            "--store-root",
            str(lane["root"]),
            "--protocol-id",
            lane["protocol_id"],
            "--assessment-id",
            lane["assessment_id"],
            "--to",
            "model_feature",
            "--previous-decision-id",
            lane["ready"].regime_promotion_decision_id,
        ],
        capsys,
    )
    assert code == 2
    assert payload["reason"] == cli.MODEL_FEATURE_REFUSAL
    assert "requires the activated IFVG_REGIME_CONTEXT_V1 block" in payload["reason"]


def test_feature_eligible_requires_verified_owner_evidence_in_a_lawful_scope(
    lane, capsys
) -> None:
    root = lane["root"]
    base = [
        "promote",
        "--store-root",
        str(root),
        "--protocol-id",
        lane["protocol_id"],
        "--assessment-id",
        lane["assessment_id"],
        "--to",
        "feature_eligible",
        "--previous-decision-id",
        lane["ready"].regime_promotion_decision_id,
    ]
    # no owner id → refused before any store write
    code, payload = _run(base, capsys)
    assert code == 2 and "requires --owner-decision-id" in payload["reason"]
    # a bare 64-hex reference is not evidence
    code, payload = _run([*base, "--owner-decision-id", "f" * 64], capsys)
    assert code == 2 and "not a verified owner-decision artifact" in payload["reason"]
    owner = synthetic_owner_decision_fixture(
        root, protocol=lane["protocol"], assessment=lane["assessment"]
    )
    # synthetic provenance is refused in the default (real) run scope
    code, payload = _run([*base, "--owner-decision-id", owner.owner_decision_artifact_id], capsys)
    assert code == 2 and "synthetic_fixture run scope" in payload["reason"]
    # …and persists in the synthetic scope
    code, payload = _run(
        [
            *base,
            "--owner-decision-id",
            owner.owner_decision_artifact_id,
            "--run-scope",
            "synthetic_fixture",
        ],
        capsys,
    )
    assert code == 0 and payload["status"] == "persisted"
    # decided_at came from VERIFIED artifacts (the previous decision's own
    # instant is the later of the two candidates here), never the wall clock
    assert payload["decided_at"] == lane["ready"].payload.decided_at
    assert payload["decided_at_source"] == "previous_decision.decided_at"
    stored = load_regime_promotion(root, payload["regime_promotion_decision_id"])
    assert stored.payload.status.value == "feature_eligible"
    assert stored.payload.owner_ratification_ref == owner.owner_decision_artifact_id
    assert stored.payload.previous_decision_ref == lane["ready"].regime_promotion_decision_id
    # the chain now has three exact decisions, oldest first
    new_id = payload["regime_promotion_decision_id"]
    code, chained = _run(["chain", "--store-root", str(root), "--decision-id", new_id], capsys)
    assert code == 0
    assert [row["status"] for row in chained["chain"]] == [
        "descriptive_only",
        "stratification_ready",
        "feature_eligible",
    ]
    # the ratified step shows its owner artifact's exact-ID verified-load
    # outcome in THIS root (S3)
    assert [row["owner_evidence"] for row in chained["chain"]] == [
        None,
        None,
        "verified (synthetic_test_authorization_v1)",
    ]
    # stratification_ready needs its previous decision (the ladder is a chain)
    code, payload = _run(
        [
            "promote",
            "--store-root",
            str(root),
            "--protocol-id",
            lane["protocol_id"],
            "--assessment-id",
            lane["assessment_id"],
            "--to",
            "stratification_ready",
        ],
        capsys,
    )
    assert code == 2 and "requires --previous-decision-id" in payload["reason"]


def test_decided_at_is_derived_from_verified_artifacts_and_reproduces_s10(lane, capsys) -> None:
    """F10 / S11: ``--decided-at`` does not exist; ``promote --to
    stratification_ready`` reproduces S10's deterministic decision exactly
    (a reuse, never a second decision under another instant)."""

    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "promote",
                "--store-root",
                str(lane["root"]),
                "--protocol-id",
                lane["protocol_id"],
                "--assessment-id",
                lane["assessment_id"],
                "--to",
                "stratification_ready",
                "--decided-at",
                "2026-08-28T13:00:00+00:00",
            ]
        )
    code, payload = _run(
        [
            "promote",
            "--store-root",
            str(lane["root"]),
            "--protocol-id",
            lane["protocol_id"],
            "--assessment-id",
            lane["assessment_id"],
            "--to",
            "stratification_ready",
            "--previous-decision-id",
            lane["first"].regime_promotion_decision_id,
        ],
        capsys,
    )
    assert code == 0 and payload["status"] == "reused"
    assert payload["regime_promotion_decision_id"] == lane["ready"].regime_promotion_decision_id
    assert payload["decided_at"] == lane["first"].payload.decided_at
    assert payload["decided_at_source"] == "previous_decision.decided_at"


def test_synthetic_scope_is_refused_in_the_research_namespace(lane, tmp_path, capsys) -> None:
    """S3 (probe P3, P0-4 mirror): a research-shaped root refuses
    ``--run-scope synthetic_fixture`` — a caller string never unlocks
    synthetic authority outside a test namespace."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
        persist_regime_assessment,
        persist_regime_promotion,
        persist_regime_protocol,
    )

    research = tmp_path / "data" / "ifvg_datasets" / "search" / "v1"
    persist_regime_protocol(research, lane["protocol"])
    persist_regime_assessment(research, lane["assessment"])
    persist_regime_promotion(research, lane["first"])
    code, payload = _run(
        [
            "promote",
            "--store-root",
            str(research),
            "--protocol-id",
            lane["protocol_id"],
            "--assessment-id",
            lane["assessment_id"],
            "--to",
            "stratification_ready",
            "--previous-decision-id",
            lane["first"].regime_promotion_decision_id,
            "--run-scope",
            "synthetic_fixture",
        ],
        capsys,
    )
    assert code == 2 and "confined to test namespaces" in payload["reason"]
    # the real scope persists the structural decision in the same root
    code, payload = _run(
        [
            "promote",
            "--store-root",
            str(research),
            "--protocol-id",
            lane["protocol_id"],
            "--assessment-id",
            lane["assessment_id"],
            "--to",
            "stratification_ready",
            "--previous-decision-id",
            lane["first"].regime_promotion_decision_id,
        ],
        capsys,
    )
    assert code == 0 and payload["status"] == "persisted"


def test_lock_timeouts_and_filesystem_errors_exit_sanitized(lane, capsys, monkeypatch) -> None:
    """S10: a supersession-log lock timeout or a filesystem error prints one
    sanitized line and exits 2 — never a traceback."""

    from alpha_lab.agents.data_infra.ifvg.ml import regime_store

    owner = synthetic_owner_decision_fixture(
        lane["root"], protocol=lane["protocol"], assessment=lane["assessment"]
    )
    args = [
        "promote",
        "--store-root",
        str(lane["root"]),
        "--protocol-id",
        lane["protocol_id"],
        "--assessment-id",
        lane["assessment_id"],
        "--to",
        "feature_eligible",
        "--previous-decision-id",
        lane["ready"].regime_promotion_decision_id,
        "--owner-decision-id",
        owner.owner_decision_artifact_id,
        "--run-scope",
        "synthetic_fixture",
    ]
    for error in (
        TimeoutError("owner-decision supersession log is locked by another writer\ndetail"),
        FileExistsError("an immutable entry already exists"),
    ):

        def _raise(*_a, _error=error, **_k):
            raise _error

        monkeypatch.setattr(regime_store, "persist_regime_promotion", _raise)
        code, payload = _run(args, capsys)
        assert code == 2 and payload["status"] == "refused"
        assert payload["reason"] == str(error).splitlines()[0]
