"""The Phase 4 operator runner (``scripts/ifvg_bounded_verification.py``):
importing launches nothing; without the owner's persisted verification run
both ``preflight`` and ``run`` refuse BEFORE any source path (typed
``fail_before_path``); with a persisted, bound run the preflight passes on a
synthetic store and no ``data/`` file is ever written.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
from strategy_core.strategies.ifvg_smc.state import IFVG_SEED_SCHEMA_VERSION, seed_hash

from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.catalog import append_catalog_event
from alpha_lab.agents.data_infra.ifvg.search.child_replay import (
    DaySeedsRecord,
    SeedSnapshotPayload,
    save_seed_snapshot,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SEARCH_TEST_STORE_ROOT,
    save_or_reuse_envelope,
)
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import initialize_test_namespace
from alpha_lab.agents.data_infra.ifvg.search.verification import (
    VerificationRunEnvelope,
    VerificationRunPayload,
)
from tests.agents.ifvg_search.namespace_fixture import verification_authorization_ref

_PROFILE = "ifvg_v2_doc_default_fresh_static_1r"
_WINDOW = ("2026-06-04", "2026-06-05")


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location(
        "ifvg_bounded_verification", Path("scripts/ifvg_bounded_verification.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_import_launches_nothing_and_unauthorized_stores_refuse_before_path(
    script, tmp_path, capsys
) -> None:
    repo_root = tmp_path / "repo"
    store_root = repo_root / SEARCH_TEST_STORE_ROOT
    store_root.mkdir(parents=True)
    for command in ("preflight", "run"):
        code = script.main(
            [command, "--store-root", str(store_root), "--repo-root", str(repo_root),
             "--state-root", str(tmp_path / "state"), "--evidence-dir", str(tmp_path / "ev")]
        )
        payload = json.loads(capsys.readouterr().out)
        assert code == 2 and payload["status"] == "refused"
        assert payload["reason"] == "fail_before_path"
        assert "VerificationAuthorizationRef" in payload["detail"]
    assert not (tmp_path / "ev").exists()
    assert not (tmp_path / "state").exists()


def test_preflight_passes_on_a_persisted_bound_run(script, tmp_path, synthetic_chain, capsys):
    repo_root = tmp_path / "repo"
    store_root = repo_root / SEARCH_TEST_STORE_ROOT
    store_root.mkdir(parents=True)
    initialize_test_namespace(store_root)
    resolved = resolve_profile_config({"profile_name": _PROFILE})
    replace(IfvgCaptureConfig(), section=resolved.section)
    seed = synthetic_chain[1].end_seed
    snapshot = save_seed_snapshot(
        store_root,
        SeedSnapshotPayload(
            profile_name=_PROFILE,
            resolved_section_config_hash=seed.profile_hash,
            seed_schema_version=IFVG_SEED_SCHEMA_VERSION,
            seed_hash=seed_hash(seed),
            snapshot_through_day="2026-06-03",
            first_replay_day=_WINDOW[0],
            entering_day_seeds=DaySeedsRecord(
                prev_day="2026-06-03", prev_full_hl=(100, 50), prev_ny_day=None, prev_ny_hl=None
            ),
            chain_policy_id="development_explicit_dates_before_path_v2",
            chain_date_count=2,
            strategy_core_commit="c" * 40,
        ),
        seed,
    )
    authorization = verification_authorization_ref(
        store_root,
        approved_allowlist_hash=allowlist_sha256(_WINDOW),
        coverage_matrix_artifact_id="b" * 64,
        seed_snapshot_id=snapshot.seed_snapshot_id,
    )
    run = VerificationRunEnvelope.from_payload(
        VerificationRunPayload(
            pipeline_semantic_id="a" * 64,
            verification_authorization=authorization,
            allowlist=_WINDOW,
            allowlist_hash=allowlist_sha256(_WINDOW),
            seed_snapshot_id=snapshot.seed_snapshot_id,
            baseline_profile_id=_PROFILE,
            baseline_section_config_hash=resolved.section_config_hash,
            coverage_matrix_artifact_id="b" * 64,
        )
    )
    save_or_reuse_envelope(store_root, "verification_runs", run)
    append_catalog_event(
        store_root, kind="display_name", artifact_id=run.verification_run_id,
        payload={"display_name": "synthetic bound run"},
    )
    code = script.main(
        ["preflight", "--store-root", str(store_root), "--repo-root", str(repo_root)]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0, payload
    assert payload["status"] == "preflight_passed"
    assert payload["preflight"]["logical_trading_days"] == list(_WINDOW)
    assert all(value for _key, value in payload["preflight"]["checks"])  # ImmutableMap pairs
    # nothing under the repository's real data tree was touched
    assert not list(Path("data").glob("ifvg_datasets/search_test/**/STORE_NAMESPACE.json"))
