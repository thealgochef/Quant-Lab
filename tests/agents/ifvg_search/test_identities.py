"""Identity, projection-audit, and deep-immutability suites (TEST_MATRIX §3.1/§3.8/§3.9)."""

from __future__ import annotations

import copy

import pytest

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    CoreStrategyReplayIdentity,
    CoreStrategyReplayPayload,
    FrozenContract,
    ImmutableMap,
    ReplayAccessAuthorizationRef,
    ReplayExecutionAccessAudit,
    ReplayInputBundlePayload,
    SearchChildMembership,
    _example_core_replay_payload,
    _example_input_bundle_payload,
    _example_partition,
    build_replay_input_bundle,
    canonical_contract_sha256,
    canonical_profile_id_for,
    canonicalize_section,
    name_free_section_hash,
    registered_identity_pairs,
)


def _bundle_payload() -> ReplayInputBundlePayload:
    return _example_input_bundle_payload()


def _core_payload(**updates) -> CoreStrategyReplayPayload:
    return _example_core_replay_payload().model_copy(update=updates)


def _core_id(**updates) -> str:
    return CoreStrategyReplayIdentity.from_payload(_core_payload(**updates)).core_replay_id


# ── identity-projection audit (P0-C / V3 P0-1) ───────────────────────────────


def test_identity_projection_audit_all_pairs() -> None:
    pairs = registered_identity_pairs()
    names = {pair.name for pair in pairs}
    assert {
        "ReplayInputBundle",
        "CoreStrategyReplay",
        "AuthorizationRequirementSet",
        "SearchCharter",
        "VerificationRun",
        "CoverageMatrix",
        "SeedSnapshot",
        "Cohort",
        "StudyCell",
        "Comparison",
        "FeatureBlockResolution",
        "FeatureBundleResolution",
    } <= names
    for pair in pairs:
        # payload → id → envelope → reload determinism
        assert pair.example_factory is not None, f"{pair.name} lacks an audit example"
        payload = pair.example_factory()
        envelope = pair.envelope_cls.from_payload(payload)
        envelope_id = getattr(envelope, pair.id_field)
        assert envelope_id == canonical_contract_sha256(payload)
        reloaded = pair.envelope_cls.model_validate(envelope.model_dump(mode="json"))
        assert getattr(reloaded, pair.id_field) == envelope_id
        # a tampered id fails closed
        dumped = envelope.model_dump(mode="json")
        dumped[pair.id_field] = "0" * 64
        with pytest.raises(ValueError):
            pair.envelope_cls.model_validate(dumped)


def test_payloads_cannot_carry_display_or_attempt_fields() -> None:
    from alpha_lab.agents.data_infra.ifvg.search.identities import (
        _FORBIDDEN_PAYLOAD_FIELDS,
    )

    for pair in registered_identity_pairs():
        fields = set(pair.payload_cls.model_fields)
        assert pair.id_field not in fields, pair.name
        assert not (fields & _FORBIDDEN_PAYLOAD_FIELDS), pair.name


# ── deep immutability (V3 P1-1, mutation-adversarial) ────────────────────────


def test_immutable_map_defensively_copies_and_sorts() -> None:
    class Holder(FrozenContract):
        values: ImmutableMap[str, tuple[int, ...]]

    source = {"b": (1, 2), "a": (3,)}
    holder = Holder(values=source)
    identity_before = canonical_contract_sha256(holder)
    source["c"] = (0,)  # mutating the input mapping cannot reach the model
    assert canonical_contract_sha256(holder) == identity_before
    assert list(holder.values) == ["a", "b"]  # canonically sorted iteration
    with pytest.raises(TypeError):
        holder.values["z"] = (9,)  # type: ignore[index]
    # serialization emits sorted (key, value) records and round-trips exactly
    dumped = holder.model_dump(mode="json")
    assert dumped["values"] == [["a", [3]], ["b", [1, 2]]]
    assert canonical_contract_sha256(Holder.model_validate(dumped)) == identity_before


def _walk_and_attack(value, path=""):
    """Recursively visit every reachable container; attempt mutation on maps
    and PROVE no plain mutable container (dict/list/set) is reachable at all."""

    from pydantic import BaseModel  # noqa: PLC0415

    if isinstance(value, ImmutableMap):
        with pytest.raises(TypeError):
            value["__attack__"] = 1  # type: ignore[index]
        for key, item in value.items():
            _walk_and_attack(item, f"{path}[{key!r}]")
        return
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            _walk_and_attack(getattr(value, name), f"{path}.{name}")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _walk_and_attack(item, f"{path}[{index}]")
        return
    assert not isinstance(value, (dict, list, set, bytearray)), (
        f"mutable container reachable on an identity payload at {path}"
    )


def test_mutation_after_identity_cannot_change_payload_hash() -> None:
    for pair in registered_identity_pairs():
        payload = pair.example_factory()
        identity_before = canonical_contract_sha256(payload)
        dumped = payload.model_dump()
        mutated = copy.deepcopy(dumped)
        assert canonical_contract_sha256(payload) == identity_before
        del mutated
        # walk EVERY reachable nested structure: attempt mutation on each map
        # and prove no plain mutable container exists anywhere (CS §0.3)
        _walk_and_attack(payload, pair.name)
        assert canonical_contract_sha256(payload) == identity_before


def test_ql_replay_source_identity_scoped_tree_fixture(tmp_path) -> None:
    """§3.8 QL replay-source sensitivity on a REAL scoped source tree."""

    import subprocess

    from alpha_lab.agents.data_infra.ifvg.search.identities import (
        quant_lab_replay_source_identity,
    )

    repo = tmp_path / "repo"
    scoped = repo / "lane" / "driver.py"
    outside = repo / "docs" / "notes.md"
    scoped.parent.mkdir(parents=True)
    outside.parent.mkdir(parents=True)
    scoped.write_text("driver = 1", encoding="utf-8")
    outside.write_text("notes", encoding="utf-8")
    for command in (
        ["git", "init", "-q"],
        ["git", "add", "."],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    scope = ("lane/driver.py",)
    base = quant_lab_replay_source_identity(repository_root=repo, source_paths=scope)
    # out-of-scope changes never move the scoped identity
    outside.write_text("changed notes", encoding="utf-8")
    assert quant_lab_replay_source_identity(repository_root=repo, source_paths=scope) == base
    # scoped changes ALWAYS move it (dirty evidence + tree hash)
    scoped.write_text("driver = 2", encoding="utf-8")
    assert quant_lab_replay_source_identity(repository_root=repo, source_paths=scope) != base


# ── ReplayInputBundle content sensitivity + portability (P0-A / V3 P0-4) ─────


def test_input_bundle_changes_with_partition_bytes() -> None:
    base = _bundle_payload()
    changed = base.model_copy(
        update={
            "ordered_source_partitions": (
                base.ordered_source_partitions[0].model_copy(
                    update={"content_sha256": "9" * 64}
                ),
            )
        }
    )
    assert canonical_contract_sha256(base) != canonical_contract_sha256(changed)


def test_input_bundle_changes_with_day_artifact_manifest() -> None:
    base = _bundle_payload()
    changed = base.model_copy(
        update={
            "ordered_day_artifacts": (
                base.ordered_day_artifacts[0].model_copy(
                    update={"manifest_payload_sha256": "9" * 64}
                ),
            )
        }
    )
    assert canonical_contract_sha256(base) != canonical_contract_sha256(changed)


def test_input_bundle_is_portable_and_normalizes_reorder_only_input() -> None:
    partition_a = _example_partition()
    partition_b = partition_a.model_copy(
        update={
            "source_partition_id": "databento/NQ/2026-06-03/mbp1",
            "source_partition_utc_date": "2026-06-03",
            "relative_logical_partition_key": "prev_utc_date/mbp1",
        }
    )
    kwargs = dict(
        authorized_date_set_id="example_date_set_v1",
        day_artifacts=_bundle_payload().ordered_day_artifacts,
        source_contract_id="databento_nq_v1",
        source_schema_era_id="mbp1_era_v1",
        access_authorization=_bundle_payload().access_authorization,
    )
    forward = build_replay_input_bundle(
        source_partitions=(partition_a, partition_b), **kwargs
    )
    reordered = build_replay_input_bundle(
        source_partitions=(partition_b, partition_a), **kwargs
    )
    assert forward.replay_input_bundle_id == reordered.replay_input_bundle_id


def test_two_partitions_of_one_trading_day_never_collide() -> None:
    partition_a = _example_partition()
    partition_b = partition_a.model_copy(
        update={
            "source_partition_id": "databento/NQ/2026-06-03/mbp1",
            "source_partition_utc_date": "2026-06-03",
            "relative_logical_partition_key": "prev_utc_date/mbp1",
        }
    )
    bundle = build_replay_input_bundle(
        authorized_date_set_id="example_date_set_v1",
        source_partitions=(partition_a, partition_b),
        day_artifacts=_bundle_payload().ordered_day_artifacts,
        source_contract_id="databento_nq_v1",
        source_schema_era_id="mbp1_era_v1",
        access_authorization=_bundle_payload().access_authorization,
    )
    assert len(bundle.payload.ordered_source_partitions) == 2
    swapped = bundle.payload.model_copy(
        update={
            "ordered_source_partitions": (
                bundle.payload.ordered_source_partitions[0].model_copy(
                    update={"content_sha256": "8" * 64}
                ),
                bundle.payload.ordered_source_partitions[1],
            )
        }
    )
    assert canonical_contract_sha256(swapped) != canonical_contract_sha256(bundle.payload)


def test_identical_canonical_keys_are_refused_not_guessed() -> None:
    partition = _example_partition()
    with pytest.raises(ValueError, match="not canonicalizable|identical canonical"):
        ReplayInputBundlePayload(
            authorized_date_set_id="x",
            ordered_source_partitions=(
                partition,
                partition.model_copy(update={"content_sha256": "9" * 64}),
            ),
            ordered_day_artifacts=(),
            source_contract_id="databento_nq_v1",
            source_schema_era_id="era",
            access_authorization=_bundle_payload().access_authorization,
        )


def test_absolute_paths_never_enter_a_bundle() -> None:
    partition = _example_partition().model_copy(
        update={"source_partition_id": "C:\\data\\databento\\NQ"}
    )
    with pytest.raises(ValueError, match="absolute paths"):
        ReplayInputBundlePayload(
            authorized_date_set_id="x",
            ordered_source_partitions=(partition,),
            ordered_day_artifacts=(),
            source_contract_id="databento_nq_v1",
            source_schema_era_id="era",
            access_authorization=_bundle_payload().access_authorization,
        )


# ── core-replay identity rule (§1.2) ─────────────────────────────────────────


def test_core_replay_identity_sensitivity() -> None:
    base = _core_id()
    assert _core_id(replay_input_bundle_id="9" * 64) != base
    assert _core_id(quant_lab_replay_source_identity="9" * 64) != base
    assert _core_id(strategy_core_commit="9" * 40) != base
    assert _core_id(strategy_core_source_identity="9" * 64) != base
    assert _core_id(resolved_section_config_hash="9" * 64) != base
    assert _core_id(warmup_seed_identity="snapshot:" + "9" * 16) != base
    assert _core_id(anchor_policy="other_anchor_v9") != base


def test_core_replay_identity_independence_from_study_and_resources() -> None:
    payload = _core_payload()
    identity = CoreStrategyReplayIdentity.from_payload(payload)
    membership_a = SearchChildMembership(
        parent_search_id="1" * 64,
        child_ordinal=0,
        axis_value_ids={"parent_retest_timeout_1m_bars": "parent_retest_timeout_1m_bars.none"},
        core_replay_id=identity.core_replay_id,
        comparison_role="baseline",
    )
    membership_b = membership_a.model_copy(
        update={"parent_search_id": "2" * 64, "child_ordinal": 7, "comparison_role": "challenger"}
    )
    # two studies, one replay: memberships differ, core id does not
    assert membership_a.core_replay_id == membership_b.core_replay_id
    assert membership_a.parent_search_id != membership_b.parent_search_id
    # runtime access audits never enter the identity (V3 P0-5)
    audit_a = ReplayExecutionAccessAudit(
        core_replay_id=identity.core_replay_id,
        execution_attempt_ref="attempt-1",
        event_chain_sha256="a" * 64,
        counters={"protected_buffer": 0},
    )
    audit_b = audit_a.model_copy(
        update={"execution_attempt_ref": "attempt-2", "event_chain_sha256": "b" * 64}
    )
    assert audit_a.core_replay_id == audit_b.core_replay_id
    assert "event_chain_sha256" not in CoreStrategyReplayPayload.model_fields
    assert "counters" not in CoreStrategyReplayPayload.model_fields


def test_preflight_authorization_is_identity_but_runtime_audit_is_not() -> None:
    base = _bundle_payload()
    changed_preflight = base.model_copy(
        update={
            "access_authorization": ReplayAccessAuthorizationRef(
                access_policy_id="verification_fixed_allowlist_max5_v1",
                authorized_date_set_id="other_set",
                expected_source_inventory_hash="9" * 64,
            )
        }
    )
    assert canonical_contract_sha256(base) != canonical_contract_sha256(changed_preflight)


# ── canonical naming (§1.3) ──────────────────────────────────────────────────


def test_canonical_profile_naming_is_study_independent() -> None:
    from strategy_core.strategies.ifvg_smc.section import (
        default_ifvg_smc_section,
        ifvg_profile_hash,
    )

    base = default_ifvg_smc_section()
    overrides = {"qualification_mode": "custom_profile", "parent_retest_timeout_1m_bars": 480}
    child_a = base.model_validate(
        {**base.model_dump(mode="json"), "profile_name": "study_a_child", **overrides}
    )
    child_b = base.model_validate(
        {**base.model_dump(mode="json"), "profile_name": "study_b_child", **overrides}
    )
    # different study names → same name-free hash, same canonical id
    assert name_free_section_hash(child_a) == name_free_section_hash(child_b)
    assert canonical_profile_id_for(child_a) == canonical_profile_id_for(child_b)
    canonical_a = canonicalize_section(child_a)
    canonical_b = canonicalize_section(child_b)
    assert canonical_a.profile_name == canonical_b.profile_name
    assert canonical_a.profile_name.startswith("ifvg_search_profile_")
    # hence identical section hashes → identical record ids downstream
    assert ifvg_profile_hash(canonical_a) == ifvg_profile_hash(canonical_b)
    # the name-free hash really excludes ONLY the name
    assert name_free_section_hash(base) != name_free_section_hash(child_a)


def test_baseline_profiles_keep_their_registered_names() -> None:
    from strategy_core.strategies.ifvg_smc.section import default_ifvg_smc_section

    base = default_ifvg_smc_section()
    assert canonicalize_section(base).profile_name == "ifvg_v2_doc_default_fresh_static_1r"


# ── generated-profile capability (P0-D) ──────────────────────────────────────


def test_generated_profile_capability_paths() -> None:
    from strategy_core.strategies.ifvg_smc.section import default_ifvg_smc_section

    from alpha_lab.agents.data_infra.ifvg.search.identities import (
        evaluate_generated_profile_capability,
    )

    base = default_ifvg_smc_section()
    child = canonicalize_section(
        base.model_validate(
            {
                **base.model_dump(mode="json"),
                "profile_name": "x",
                "qualification_mode": "custom_profile",
                "parent_retest_timeout_1m_bars": 480,
            }
        )
    )
    common = dict(
        axis_value_ids={"parent_retest_timeout_1m_bars": "parent_retest_timeout_1m_bars.480"},
        registry_hash="a" * 64,
        authorization_ref="auth-1",
    )
    runnable = evaluate_generated_profile_capability(
        baseline_profile_id="ifvg_v2_doc_default_fresh_static_1r",
        authorization_state="authorized",
        section=child,
        **common,
    )
    assert runnable.status == "generated_runnable"
    blocked_base = evaluate_generated_profile_capability(
        baseline_profile_id="ifvg_v2_ict_clean_fresh_static_1r",
        authorization_state="authorized",
        section=child,
        **common,
    )
    assert blocked_base.status == "blocked_base_profile"
    unregistered_base = evaluate_generated_profile_capability(
        baseline_profile_id="no_such_profile",
        authorization_state="authorized",
        section=child,
        **common,
    )
    # absence from the FIXED registry blocks a BASELINE, never a generated child
    assert unregistered_base.status == "blocked_base_profile"
    unratified = evaluate_generated_profile_capability(
        baseline_profile_id="ifvg_v2_doc_default_fresh_static_1r",
        authorization_state="missing",
        section=child,
        **common,
    )
    assert unratified.status == "blocked_owner_decision"
    invalid = evaluate_generated_profile_capability(
        baseline_profile_id="ifvg_v2_doc_default_fresh_static_1r",
        authorization_state="authorized",
        section=None,
        section_error="ValidationError: bad field",
        **common,
    )
    assert invalid.status == "blocked_invalid_section"


def test_companion_identities_version_separately() -> None:
    from alpha_lab.agents.data_infra.ifvg.search.identities import (
        FsmAuditArtifactIdentity,
        ReplayChartArtifactIdentity,
    )

    core = "3" * 64
    audit_v1 = FsmAuditArtifactIdentity(
        core_replay_id=core,
        audit_schema_version=1,
        audit_contract_fingerprint="f" * 64,
        neutrality_mechanism_id="dual_drive_ab_v1",
    )
    audit_v2 = audit_v1.model_copy(update={"audit_schema_version": 2})
    chart = ReplayChartArtifactIdentity(
        core_replay_id=core,
        replay_chart_schema_version=1,
        range_policy_id="candidate_range_v1",
        stage_gating_policy_id="stage_gate_ordinal_cursor_ts_v2",
    )
    assert audit_v1.core_replay_id == audit_v2.core_replay_id == chart.core_replay_id
    assert audit_v1 != audit_v2
