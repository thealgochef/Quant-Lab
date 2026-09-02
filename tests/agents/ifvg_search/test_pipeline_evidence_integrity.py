"""R6.1-FIX §3.7 / §3.8 / §3.9 inside the 16-stage pipeline (findings F-06,
F-07, F-10A).

* S02 persists every completed child's executed-trade table artifact by the
  identity derived from the core replay; S14 verified-loads it by that id
  and binds it into every stratified report; a reused child is verified
  against the persisted table bytes (never dropped silently), a missing
  table is a typed ``children_skipped`` record, a corrupt table a typed
  child failure;
* a tampered prior-attempt stage sidecar is a typed stage FAILURE — never
  "not produced";
* S15 records every reload failure by reason;
* production wiring gaps raise ``PipelineWiringError`` (no ``assert``).

Adversarial round (reviews B-01 … B-09): every costed evaluation derives
from the ONE verified persisted projection (provenance-independent bytes);
a manifest-less entry is typed corruption; a drifted re-derivation is
refused (projection bytes AND raw core-table hash); a disagreeing
neutrality hash fails the child; a halted attempt resets the publication
block and activation re-derives the gates; S15 records a real post-S14
tamper in the immutable pipeline result and the publication gates fail.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.ml.regime_stratification_service import (
    load_regime_stratified_report,
)
from alpha_lab.agents.data_infra.ifvg.search import pipeline as pipeline_module
from alpha_lab.agents.data_infra.ifvg.search.charter import SearchCharterEnvelope
from alpha_lab.agents.data_infra.ifvg.search.executed_trade_table import (
    EXECUTED_TRADE_TABLE_SIDECAR,
    EXECUTED_TRADE_TABLE_STORE,
    executed_trade_table_id_for,
    load_executed_trade_table,
)
from alpha_lab.agents.data_infra.ifvg.search.failure import PipelineWiringError
from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import _child_evaluation_envelope
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
    PUBLICATION_GATE_IDS,
    PipelineResultEnvelope,
    PipelineSemanticIdentity,
    PublicationError,
    QuantLabPipelineStage,
    StageStatus,
    activate_pipeline_result,
    read_pipeline_state,
    run_pipeline,
    run_publication_gates,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SearchStoreError,
    SidecarLoadError,
    envelope_destination,
    has_envelope,
    load_sidecar_bytes,
    load_verified_envelope,
)
from tests.agents.ifvg_search.pipeline_fixture import build_pipeline_fixture
from tests.agents.ifvg_search.test_pipeline_regime import (  # noqa: F401
    completed_candidate,
)

S = QuantLabPipelineStage


def _run(fixture, **overrides):
    return run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
        **overrides,
    )


def _state(fixture, result=None):
    result = result or fixture["result"]
    return read_pipeline_state(fixture["state_root"], result.pipeline_semantic_id)


def _sidecar(fixture, stage, name, result=None) -> dict:
    entry = _state(fixture, result)["stages"][stage.value]
    return json.loads(
        load_sidecar_bytes(
            fixture["store_root"], "pipeline_stage_results", entry["stage_result_id"], name
        )
    )


def _other_cost_policy(fixture, *, state_root: Path, cost: float):
    charter = fixture["charter"]
    other_payload = charter.payload.model_copy(
        update={
            "cost_policy": charter.payload.cost_policy.model_copy(
                update={"cost_points_round_turn": cost}
            )
        }
    )
    other_charter = SearchCharterEnvelope.from_payload(other_payload)
    other_spec = fixture["semantic"].payload.model_copy(
        update={
            "search_charter_id": other_charter.search_id,
            "cost_policy_sha256": canonical_contract_sha256(other_payload.cost_policy),
        }
    )
    return {
        **fixture,
        "charter": other_charter,
        "semantic": PipelineSemanticIdentity.from_payload(other_spec),
        "state_root": state_root,
    }


# ── §3.7 the executed-trade table artifact in S02 / S14 ─────────────────────


def test_s02_persists_the_executed_trade_table_and_s14_verified_loads_it(
    completed_candidate,  # noqa: F811
):
    root = completed_candidate["store_root"]
    rows = _state(completed_candidate)["children"]
    assert rows and all(row["state"] == "completed" for row in rows)
    record = _sidecar(
        completed_candidate, S.S14_BUILD_FRONTIER_AND_INSIGHTS, "regime_stratified_reports.json"
    )
    assert record["children_skipped"] == {}
    evidence = record["children_evidence"]
    for row in rows:
        core = row["core_replay_id"]
        table_id = executed_trade_table_id_for(core, record_schema_version=2)
        assert row["executed_trade_table_id"] == table_id
        assert has_envelope(root, EXECUTED_TRADE_TABLE_STORE, table_id)
        loaded = load_executed_trade_table(root, table_id)
        assert evidence[core] == {
            "executed_trade_table_id": table_id,
            "executed_trade_table_sha256": loaded.envelope.executed_trade_table_sha256,
            "row_count": loaded.envelope.row_count,
        }
    assert set(evidence) == {row["core_replay_id"] for row in rows} == set(record["children"])
    for report_id in record["reports_by_class"]["cohort_descriptive"]:
        report = load_regime_stratified_report(root, report_id)
        body = report.payload.body
        assert body.executed_trade_table_id == executed_trade_table_id_for(
            body.core_replay_id, record_schema_version=2
        )
        assert body.executed_trade_table_id in report.payload.source_metric_refs


def test_reused_children_verify_against_the_persisted_table_and_never_vanish_from_s14(
    completed_candidate,  # noqa: F811
    tmp_path,
):
    """The F-06 gap: a reused child under a cost policy with no persisted
    evaluation used to vanish from S14. Now the re-derived tables are
    verified byte-for-byte against the persisted executed-trade table, the
    evaluation is published from verified bytes, and every child stays in
    the reports."""

    root = completed_candidate["store_root"]
    other = _other_cost_policy(completed_candidate, state_root=tmp_path / "other_state", cost=1.25)
    result = _run(other)
    assert result.stage_statuses[S.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS.value] == (
        StageStatus.COMPLETED.value
    )
    rows = read_pipeline_state(other["state_root"], result.pipeline_semantic_id)["children"]
    assert rows
    for row in rows:
        assert row["state"] == "reused"
        assert row["replay_invocations"] == 1
        assert "executed-trade table" in row["explanation"]
        assert "byte" in row["explanation"]
        evaluation = _child_evaluation_envelope(
            row["core_replay_id"], other["charter"].payload.cost_policy
        )
        assert has_envelope(root, "costed_evaluations", evaluation.costed_evaluation_id)
    record = _sidecar(
        other, S.S14_BUILD_FRONTIER_AND_INSIGHTS, "regime_stratified_reports.json", result
    )
    assert record["children_skipped"] == {}
    assert set(record["children"]) == {row["core_replay_id"] for row in rows}
    assert record["reports_by_class"]["cohort_descriptive"]


def test_a_tampered_trade_table_is_refused_not_skipped(completed_candidate, tmp_path):  # noqa: F811
    root = completed_candidate["store_root"]
    rows = _state(completed_candidate)["children"]
    victim = rows[0]["core_replay_id"]
    table_id = executed_trade_table_id_for(victim, record_schema_version=2)
    sidecar = (
        envelope_destination(root, EXECUTED_TRADE_TABLE_STORE, table_id)
        / EXECUTED_TRADE_TABLE_SIDECAR
    )
    original = sidecar.read_bytes()
    other = _other_cost_policy(completed_candidate, state_root=tmp_path / "tamper_state", cost=1.5)
    try:
        sidecar.write_bytes(original[:-4] + b"\x00" * 4)
        result = _run(other)
        state = read_pipeline_state(other["state_root"], result.pipeline_semantic_id)
        by_id = {row["core_replay_id"]: row for row in state["children"]}
        assert by_id[victim]["state"] == "failed"
        assert "executed-trade table" in by_id[victim]["explanation"]
        assert "sidecar_hash_mismatch" in by_id[victim]["explanation"]
        others = [row for core, row in by_id.items() if core != victim]
        assert others and all(row["state"] == "reused" for row in others)
        record = _sidecar(
            other, S.S14_BUILD_FRONTIER_AND_INSIGHTS, "regime_stratified_reports.json", result
        )
        # the corrupt table is a TYPED child-stage failure at S02; S14 records
        # the failed child under children_skipped — never a silent omission
        assert record["children_skipped"][victim] == "child_not_completed_or_reused"
        assert victim not in record["children"]
    finally:
        sidecar.write_bytes(original)


def test_children_skipped_is_persisted_with_a_typed_reason(completed_candidate, tmp_path):  # noqa: F811
    """A reused child whose persisted table is ABSENT and whose re-derivation
    cannot be verified (no evaluation under this cost policy) is recorded
    under ``children_skipped`` with the typed reason — never silently omitted."""

    import shutil

    root = completed_candidate["store_root"]
    rows = _state(completed_candidate)["children"]
    victim = rows[-1]["core_replay_id"]
    table_id = executed_trade_table_id_for(victim, record_schema_version=2)
    directory = envelope_destination(root, EXECUTED_TRADE_TABLE_STORE, table_id)
    parked = tmp_path / "parked_table"
    shutil.move(str(directory), str(parked))
    other = _other_cost_policy(completed_candidate, state_root=tmp_path / "skip_state", cost=1.75)
    try:
        result = _run(other)
        state = read_pipeline_state(other["state_root"], result.pipeline_semantic_id)
        by_id = {row["core_replay_id"]: row for row in state["children"]}
        assert by_id[victim]["state"] == "reused"
        assert "executed_trade_table_unavailable" in by_id[victim]["explanation"]
        evaluation = _child_evaluation_envelope(victim, other["charter"].payload.cost_policy)
        assert not has_envelope(root, "costed_evaluations", evaluation.costed_evaluation_id)
        record = _sidecar(
            other, S.S14_BUILD_FRONTIER_AND_INSIGHTS, "regime_stratified_reports.json", result
        )
        assert record["children_skipped"][victim] == "executed_trade_table_unavailable"
        assert victim not in record["children"]
        assert len(record["children"]) == len(rows) - 1
    finally:
        shutil.move(str(parked), str(directory))


# ── §3.8 fail-closed prior-attempt sidecars ─────────────────────────────────


def _tamper(root: Path, stage_result_id: str, name: str):
    path = envelope_destination(root, "pipeline_stage_results", stage_result_id) / name
    original = path.read_bytes()
    path.write_bytes(original[:-2] + b"  ")
    return path, original


@pytest.mark.parametrize(
    ("stage", "name", "failing_stage"),
    [
        (S.S12_RUN_PROP_HISTORICAL_REPLAYS, "prop_vectors.json", S.S12_RUN_PROP_HISTORICAL_REPLAYS),
        (
            S.S13_RUN_BOOTSTRAP_AND_STRESS,
            "account_simulations.json",
            S.S13_RUN_BOOTSTRAP_AND_STRESS,
        ),
        (
            S.S14_BUILD_FRONTIER_AND_INSIGHTS,
            "regime_stratified_reports.json",
            S.S14_BUILD_FRONTIER_AND_INSIGHTS,
        ),
        # the S09 record is re-derived byte-identically on the retry, so the
        # tamper is met at S09's own republication (typed refusal, S09 fails)
        (S.S09_TRAIN_MODELS, "regime_run.json", S.S09_TRAIN_MODELS),
    ],
)
def test_prior_attempt_sidecar_tamper_fails_the_stage_closed(
    completed_candidate,  # noqa: F811
    tmp_path,
    stage,
    name,
    failing_stage,
):
    root = completed_candidate["store_root"]
    # a private copy of the state so the module fixture's attempt history is untouched
    import shutil

    state_root = tmp_path / "state"
    shutil.copytree(completed_candidate["state_root"], state_root)
    fixture = {**completed_candidate, "state_root": state_root}
    entry = _state(completed_candidate)["stages"][stage.value]
    path, original = _tamper(root, entry["stage_result_id"], name)
    try:
        result = _run(fixture, operational_retry_reason="sidecar tamper probe")
        statuses = result.stage_statuses
        assert statuses[failing_stage.value] == StageStatus.FAILED.value, statuses
        # every later planned stage of THIS attempt is pending, never a stale
        # "completed" carried over from the prior attempt
        later = [
            stage_value
            for stage_value in statuses
            if stage_value > failing_stage.value
            and stage_value != S.S11_RUN_FROZEN_MODEL_GATED_REPLAYS.value
        ]
        assert later and all(statuses[value] == StageStatus.PENDING.value for value in later), (
            statuses
        )
        state = read_pipeline_state(state_root, result.pipeline_semantic_id)
        explanation = state["stages"][failing_stage.value]["explanation"]
        # the typed reason of the store probe, or the store's own hash refusal
        # when the tampered sidecar is first met at (re)publication time
        assert "sidecar_hash_mismatch" in explanation or "failed verification" in explanation
        assert "not produced" not in explanation
    finally:
        path.write_bytes(original)


def test_lineage_sidecar_tamper_fails_closed_before_the_delta_stage(tmp_path):
    """A plain (non-regime) pipeline: the prior attempt's lineage sidecars
    live in S02's stage result, which the retry re-derives byte-identically —
    a tampered sidecar is met at S02's republication as a typed refusal, the
    attempt halts, and every later stage (the cross-profile deltas included)
    is PENDING for this attempt; nothing downstream ever reads the tampered
    evidence as "not produced"."""

    fixture = build_pipeline_fixture(tmp_path / "plain")
    first = _run(fixture)
    fixture = {**fixture, "result": first}
    root = fixture["store_root"]
    rows = _state(fixture)["children"]
    entry = _state(fixture)["stages"][S.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS.value]
    path, original = _tamper(
        root, entry["stage_result_id"], f"lineage_map_{rows[0]['core_replay_id']}.json"
    )
    try:
        result = _run(fixture, operational_retry_reason="lineage tamper probe")
        statuses = result.stage_statuses
        assert statuses[S.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS.value] == StageStatus.FAILED.value
        assert statuses[S.S14_BUILD_FRONTIER_AND_INSIGHTS.value] == StageStatus.PENDING.value
        assert statuses[S.S15_VERIFY_AND_PUBLISH.value] == StageStatus.PENDING.value
        state = read_pipeline_state(fixture["state_root"], result.pipeline_semantic_id)
        explanation = state["stages"][S.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS.value]["explanation"]
        assert "failed verification" in explanation
        assert "not produced" not in explanation
    finally:
        path.write_bytes(original)


def test_s15_records_reload_failures_by_reason(completed_candidate, tmp_path, monkeypatch):  # noqa: F811
    import shutil

    state_root = tmp_path / "state"
    shutil.copytree(completed_candidate["state_root"], state_root)
    fixture = {**completed_candidate, "state_root": state_root}
    real = pipeline_module.load_verified_envelope

    def _broken(root, store_name, envelope_id, envelope_cls):
        if store_name == "insights":
            raise SearchStoreError(f"manifest hash mismatch for {store_name}/{envelope_id}")
        return real(root, store_name, envelope_id, envelope_cls)

    monkeypatch.setattr(pipeline_module, "load_verified_envelope", _broken)
    result = _run(fixture, operational_retry_reason="reload failure probe")
    state = read_pipeline_state(state_root, result.pipeline_semantic_id)
    assert result.stage_statuses[S.S15_VERIFY_AND_PUBLISH.value] in (
        StageStatus.COMPLETED.value,
        StageStatus.REUSED.value,
    )
    failures = state["publication"]["reload_failures"]
    assert failures and all(key.startswith("insights/") for key in failures)
    assert all("manifest hash mismatch" in reason for reason in failures.values())
    assert state["publication"]["control_flow_gates_passed"] is False


# ── §3.9 production wiring checks are typed errors, never asserts ─────────────


def test_wiring_gaps_raise_typed_errors(tmp_path) -> None:
    fixture = build_pipeline_fixture(tmp_path)
    context = pipeline_module._RunContext(
        semantic=fixture["semantic"],
        charter=fixture["charter"],
        wiring=dataclasses.replace(
            fixture["wiring"],
            audit_builder=None,
            chart_builder=None,
            candidate_view_source=None,
            label_builder=None,
            mbp1_evidence_source=None,
        ),
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        state={"stages": {}},
    )
    for stage in (
        pipeline_module._stage_s03_audit,
        pipeline_module._stage_s04_charts,
        pipeline_module._stage_s05_feature_views,
        pipeline_module._stage_s07_labels,
    ):
        with pytest.raises(PipelineWiringError):
            stage(context)
    with pytest.raises(PipelineWiringError):
        pipeline_module._ensure_mbp1_evidence(context)
    assert issubclass(PipelineWiringError, RuntimeError)
    # no production assert statement survives in the pipeline or the fold-feature builder
    for module_path in (
        Path(pipeline_module.__file__),
        Path(pipeline_module.__file__).parent.parent / "ml" / "regime_fold_features.py",
    ):
        source = module_path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*assert ", source, flags=re.MULTILINE), module_path.name


# ── adversarial round (reviews B-01 … B-09) ──────────────────────────────────


def test_costed_evaluations_are_provenance_independent_across_run_orders(tmp_path):
    """Review B-01: one ``costed_evaluation_id`` carries ONE set of bytes
    whatever run order produced it. Regime run under cost P1 → plain run
    under cost P2 on the same store (children reused; the evaluation is
    published from the persisted projection) → regime run under cost P2
    (children re-derived and verified against the persisted table): S02
    completes, every child is reused, and every evaluation — P1 and P2 —
    reproduces byte-for-byte from the persisted executed-trade table, the
    ONE canonical evaluation frame."""

    from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
    from alpha_lab.agents.data_infra.ifvg.search.strategy_metrics import compute_strategy_metrics

    regime = build_pipeline_fixture(tmp_path / "regime", regime_study="candidate")
    first = _run(regime)
    assert first.stage_statuses[S.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS.value] == (
        StageStatus.COMPLETED.value
    )
    store = regime["store_root"]
    plain = {**build_pipeline_fixture(tmp_path / "plain"), "store_root": store}
    plain_p2 = _other_cost_policy(plain, state_root=tmp_path / "plain_p2_state", cost=1.25)
    second = _run(plain_p2)
    assert second.stage_statuses[S.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS.value] == (
        StageStatus.COMPLETED.value
    )
    rows_second = read_pipeline_state(plain_p2["state_root"], second.pipeline_semantic_id)[
        "children"
    ]
    assert rows_second
    assert all(
        row["state"] == "reused" and row["replay_invocations"] == 0 for row in rows_second
    )
    # the P2 evaluations were published from the persisted projection (the
    # plain reuse holds no raw tables; S14's prop merge later rewrites the
    # row explanation, so the evidence is the store, not the text)
    for row in rows_second:
        evaluation = _child_evaluation_envelope(
            row["core_replay_id"], plain_p2["charter"].payload.cost_policy
        )
        assert has_envelope(store, "costed_evaluations", evaluation.costed_evaluation_id)
    regime_p2 = _other_cost_policy(regime, state_root=tmp_path / "regime_p2_state", cost=1.25)
    third = _run(regime_p2)
    assert third.stage_statuses[S.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS.value] == (
        StageStatus.COMPLETED.value
    ), third.stage_statuses
    rows_third = read_pipeline_state(regime_p2["state_root"], third.pipeline_semantic_id)[
        "children"
    ]
    assert {row["core_replay_id"] for row in rows_third} == {
        row["core_replay_id"] for row in rows_second
    }
    assert all(
        row["state"] == "reused" and row["replay_invocations"] == 1 for row in rows_third
    )
    for fixture in (regime, regime_p2):
        policy = fixture["charter"].payload.cost_policy
        for row in rows_third:
            core = row["core_replay_id"]
            table = load_executed_trade_table(
                store, executed_trade_table_id_for(core, record_schema_version=2)
            )
            expected = compute_strategy_metrics(
                {RecordTable.EXECUTED_TRADE: table.frame},
                cost_points=policy.cost_points_round_turn,
                evaluation_config_hash=canonical_contract_sha256(
                    {"core_replay_id": core, "cost_policy": policy.model_dump(mode="json")}
                ),
            )
            stored = load_sidecar_bytes(
                store,
                "costed_evaluations",
                _child_evaluation_envelope(core, policy).costed_evaluation_id,
                "strategy_metrics.json",
            )
            assert stored == (
                json.dumps(expected.model_dump(mode="json"), sort_keys=True) + "\n"
            ).encode("utf-8")
    record = _sidecar(
        regime_p2, S.S14_BUILD_FRONTIER_AND_INSIGHTS, "regime_stratified_reports.json", third
    )
    assert record["children_skipped"] == {}
    assert set(record["children"]) == {row["core_replay_id"] for row in rows_third}


def test_a_manifest_less_table_entry_is_a_typed_child_failure(completed_candidate, tmp_path):  # noqa: F811
    """Review B-02: an entry directory without its manifest is corrupt — never
    "absent"; the reused child fails typed at S02 and S14 records it as not
    completed, not as ``executed_trade_table_unavailable``."""

    root = completed_candidate["store_root"]
    rows = _state(completed_candidate)["children"]
    victim = rows[1]["core_replay_id"]
    table_id = executed_trade_table_id_for(victim, record_schema_version=2)
    manifest = envelope_destination(root, EXECUTED_TRADE_TABLE_STORE, table_id) / "manifest.json"
    original = manifest.read_bytes()
    other = _other_cost_policy(
        completed_candidate, state_root=tmp_path / "manifestless_state", cost=1.35
    )
    try:
        manifest.unlink()
        with pytest.raises(SidecarLoadError) as info:
            has_envelope(root, EXECUTED_TRADE_TABLE_STORE, table_id)
        assert info.value.reason == "manifest_missing_for_existing_entry"
        result = _run(other)
        state = read_pipeline_state(other["state_root"], result.pipeline_semantic_id)
        by_id = {row["core_replay_id"]: row for row in state["children"]}
        assert by_id[victim]["state"] == "failed"
        assert "manifest_missing_for_existing_entry" in by_id[victim]["explanation"]
        assert "executed_trade_table_unavailable" not in by_id[victim]["explanation"]
        others = [row for core, row in by_id.items() if core != victim]
        assert others and all(row["state"] == "reused" for row in others)
        record = _sidecar(
            other, S.S14_BUILD_FRONTIER_AND_INSIGHTS, "regime_stratified_reports.json", result
        )
        assert record["children_skipped"][victim] == "child_not_completed_or_reused"
    finally:
        manifest.write_bytes(original)


def _drifted_wiring(fixture, victim: str, mutate):
    real = fixture["wiring"].child_runner

    def _runner(*, spec, core_replay_id):
        from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable  # noqa: PLC0415

        result = real(spec=spec, core_replay_id=core_replay_id)
        if core_replay_id == victim:
            mutate(result.tables[RecordTable.EXECUTED_TRADE])
        return result

    return dataclasses.replace(fixture["wiring"], child_runner=_runner)


def test_a_drifted_runner_cannot_reproduce_the_persisted_table(completed_candidate, tmp_path):  # noqa: F811
    """Reviews B-05 / B-06(1): a re-derivation that drifts INSIDE the
    projection (one realized outcome) or OUTSIDE it (a column the projection
    drops → a different raw core-table hash) is refused typed — never adopted
    as "reproduced byte-for-byte"."""

    rows = _state(completed_candidate)["children"]
    victim = rows[0]["core_replay_id"]

    def _inside(trades):
        trades.loc[0, "realized_ticks"] = int(trades.loc[0, "realized_ticks"]) + 1

    def _outside(trades):
        trades["geometry_entry_ticks"] = 7.0

    for cost, mutate, marker in (
        (1.45, _inside, "byte-for-byte"),
        (1.55, _outside, "source core-table hash"),
    ):
        other = _other_cost_policy(
            completed_candidate, state_root=tmp_path / f"drift_{cost}", cost=cost
        )
        other = {**other, "wiring": _drifted_wiring(other, victim, mutate)}
        result = _run(other)
        state = read_pipeline_state(other["state_root"], result.pipeline_semantic_id)
        by_id = {row["core_replay_id"]: row for row in state["children"]}
        assert by_id[victim]["state"] == "failed", by_id[victim]
        assert marker in by_id[victim]["explanation"], by_id[victim]["explanation"]
        others = [row for core, row in by_id.items() if core != victim]
        assert others
        assert all(row["state"] == "reused" and row["replay_invocations"] == 1 for row in others)
        record = _sidecar(
            other, S.S14_BUILD_FRONTIER_AND_INSIGHTS, "regime_stratified_reports.json", result
        )
        assert record["children_skipped"][victim] == "child_not_completed_or_reused"


def test_a_neutrality_report_disagreeing_with_the_table_hash_fails_the_child(tmp_path):
    """Review B-06(3): a fresh child whose neutrality report declares an
    executed-trade core-table hash different from the persisted table's
    source hash is a typed child failure (the other children complete)."""

    from types import SimpleNamespace

    fixture = build_pipeline_fixture(tmp_path / "neutrality")
    real = fixture["wiring"].child_runner
    victims: dict[str, str] = {}

    def _runner(*, spec, core_replay_id):
        result = real(spec=spec, core_replay_id=core_replay_id)
        if not victims:
            victims["id"] = core_replay_id
            result.neutrality = SimpleNamespace(
                passed=True, audit_disabled_core_table_hashes={"executed_trade": "0" * 64}
            )
        return result

    fixture = {**fixture, "wiring": dataclasses.replace(fixture["wiring"], child_runner=_runner)}
    result = _run(fixture)
    state = read_pipeline_state(fixture["state_root"], result.pipeline_semantic_id)
    by_id = {row["core_replay_id"]: row for row in state["children"]}
    victim = victims["id"]
    assert by_id[victim]["state"] == "failed"
    assert "core table hash" in by_id[victim]["explanation"]
    others = [row for core, row in by_id.items() if core != victim]
    assert others and all(row["state"] == "completed" for row in others)


def test_a_halted_attempt_resets_the_publication_block_and_activation_re_derives_the_gates(
    completed_candidate,  # noqa: F811
    tmp_path,
):
    """Review B-04: after a later attempt fails closed, the publication block
    of the earlier attempt is gone (gates None, no result id), the gates
    fail, and activation refuses — and a recorded all-true checklist never
    authorizes an activation on its own: the gates are RE-DERIVED from the
    latest attempt's stage statuses at activation time."""

    import shutil

    state_root = tmp_path / "state"
    shutil.copytree(completed_candidate["state_root"], state_root)
    fixture = {**completed_candidate, "state_root": state_root}
    root = fixture["store_root"]
    semantic_id = completed_candidate["result"].pipeline_semantic_id
    gates = run_publication_gates(state_root, semantic_id, store_root=root)
    assert all(gates.values()), gates
    prior_result_id = read_pipeline_state(state_root, semantic_id)["publication"][
        "pipeline_result_id"
    ]
    entry = _state(completed_candidate)["stages"][S.S12_RUN_PROP_HISTORICAL_REPLAYS.value]
    path, original = _tamper(root, entry["stage_result_id"], "prop_vectors.json")
    try:
        result = _run(fixture, operational_retry_reason="publication reset probe")
        assert result.stage_statuses[S.S12_RUN_PROP_HISTORICAL_REPLAYS.value] == (
            StageStatus.FAILED.value
        )
        state = read_pipeline_state(state_root, semantic_id)
        assert state["publication"] == {
            "state": "prepared_not_published",
            "gates": None,
            "activated": False,
            "pipeline_result_id": None,
            "control_flow_gates_passed": False,
            "reload_failures": {},
        }
        gates = run_publication_gates(state_root, semantic_id, store_root=root)
        assert not any(gates.values()), gates
        # the verification scope refuses first (its own boundary); the gate
        # re-derivation is exercised below under a crafted development scope
        with pytest.raises(PublicationError, match="can never activate"):
            activate_pipeline_result(state_root, semantic_id, store_root=root)
        # defense in depth: a crafted all-true checklist over a non-terminal
        # latest attempt (development scope, so the scope refusal does not
        # apply) is refused because the gates are re-derived at activation
        crafted = read_pipeline_state(state_root, semantic_id)
        crafted["run_scope"] = "full_authorized_development"
        crafted["publication"] = {
            "state": "gates_passed",
            "gates": {name: True for name in PUBLICATION_GATE_IDS},
            "activated": False,
            "pipeline_result_id": prior_result_id,
            "control_flow_gates_passed": True,
            "reload_failures": {},
        }
        (state_root / semantic_id / pipeline_module._STATE_FILENAME).write_text(
            json.dumps(crafted), encoding="utf-8"
        )
        with pytest.raises(PublicationError, match="latest attempt"):
            activate_pipeline_result(state_root, semantic_id, store_root=root)
        assert read_pipeline_state(state_root, semantic_id)["publication"]["activated"] is False
    finally:
        path.write_bytes(original)


def test_s15_records_a_real_artifact_tamper_and_the_publication_gates_fail(
    completed_candidate,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """Reviews B-06(4) / B-06(6) / B-09 / B-03: bytes of the persisted OOS
    assignment and of one executed-trade table are tampered AFTER S14 ran in
    this attempt (every earlier stage re-verifies what it republishes, so a
    tamper that survives to S15 has to land between S14 and S15). S15 records
    both by ``<store>/<id>`` with the typed reason, fails the control-flow
    gate, carries the record in the immutable pipeline result, and the
    publication gates fail."""

    import shutil

    state_root = tmp_path / "state"
    shutil.copytree(completed_candidate["state_root"], state_root)
    fixture = {**completed_candidate, "state_root": state_root}
    root = fixture["store_root"]
    semantic_id = completed_candidate["result"].pipeline_semantic_id
    diagnostics = _sidecar(
        completed_candidate, S.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS, "regime_diagnostics.json"
    )
    oos_id = diagnostics["regime_oos_assignment_id"]
    rows = _state(completed_candidate)["children"]
    table_id = executed_trade_table_id_for(rows[0]["core_replay_id"], record_schema_version=2)
    targets = (
        envelope_destination(root, "regime_oos_assignments", oos_id)
        / "regime_oos_assignments.arrow",
        envelope_destination(root, EXECUTED_TRADE_TABLE_STORE, table_id)
        / EXECUTED_TRADE_TABLE_SIDECAR,
    )
    originals = {path: path.read_bytes() for path in targets}
    real_s14 = pipeline_module._STAGE_EXECUTORS[S.S14_BUILD_FRONTIER_AND_INSIGHTS]

    def _s14_then_tamper(context):
        outputs = real_s14(context)
        for path, original in originals.items():
            path.write_bytes(original[:-4] + b"\x00" * 4)
        return outputs

    monkeypatch.setitem(
        pipeline_module._STAGE_EXECUTORS, S.S14_BUILD_FRONTIER_AND_INSIGHTS, _s14_then_tamper
    )
    try:
        result = _run(fixture, operational_retry_reason="post-S14 tamper probe")
        state = read_pipeline_state(state_root, semantic_id)
        assert result.stage_statuses[S.S15_VERIFY_AND_PUBLISH.value] == (
            StageStatus.COMPLETED.value
        ), state["stages"][S.S15_VERIFY_AND_PUBLISH.value]["explanation"]
        failures = state["publication"]["reload_failures"]
        assert f"regime_oos_assignments/{oos_id}" in failures, failures
        assert f"executed_trade_tables/{table_id}" in failures, failures
        assert all(
            "sidecar_hash_mismatch" in reason or "failed verification" in reason
            for reason in failures.values()
        ), failures
        assert state["publication"]["control_flow_gates_passed"] is False
        # B-09: the reasons are part of the IMMUTABLE pipeline result
        reloaded = load_verified_envelope(
            root,
            "search_results",
            state["publication"]["pipeline_result_id"],
            PipelineResultEnvelope,
        )
        assert dict(reloaded.payload.reload_failure_reasons) == failures
        assert reloaded.payload.control_flow_gates.passed is False
        gates = run_publication_gates(state_root, semantic_id, store_root=root)
        assert gates["control_flow_gates_passed"] is False and not all(gates.values())
    finally:
        for path, original in originals.items():
            path.write_bytes(original)


def test_delivered_by_s09c_fails_closed_on_a_tampered_s09_record(completed_candidate):  # noqa: F811
    """Review B-06(2): the S09c delivery lookup reads the run's OWN verified
    S09 record — a tampered record is a typed failure; in-memory S09c results
    take precedence and never touch the store."""

    from types import SimpleNamespace

    from alpha_lab.agents.data_infra.ifvg.ml import regime_report_stage

    root = completed_candidate["store_root"]
    state = _state(completed_candidate)
    context = SimpleNamespace(state=state, store_root=root)
    assert regime_report_stage._delivered_by_s09c(context, {}) == {}
    entry = state["stages"][S.S09_TRAIN_MODELS.value]
    path, original = _tamper(root, entry["stage_result_id"], "regime_run.json")
    try:
        with pytest.raises(SidecarLoadError) as info:
            regime_report_stage._delivered_by_s09c(context, {})
        assert info.value.reason == "sidecar_hash_mismatch"
        delivered = regime_report_stage._delivered_by_s09c(
            context,
            {
                "controlled_study": SimpleNamespace(
                    envelope=SimpleNamespace(regime_controlled_study_id="a" * 64)
                )
            },
        )
        assert delivered == {"feature_only": "a" * 64}
    finally:
        path.write_bytes(original)
    assert not hasattr(regime_report_stage, "_prior_reports_record")
