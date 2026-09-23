"""Small verified extracts of five original acceptance candidates and their events."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.audit_contracts import AuditTable
from alpha_lab.agents.data_infra.ifvg.b0_projection import B0ProjectionSource, project_b0_candidates
from alpha_lab.agents.data_infra.ifvg.context_feature_view import (
    M0_FEATURES,
    TIER_FEATURE_REGISTRY,
    CandidateFeatureView,
)

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "b0_real_selected_stages_v2.json"


def _frame(records):
    frame = pd.DataFrame(records)
    for column in frame:
        if column.endswith("_ts_utc"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def real_b0_fixture():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    source = B0ProjectionSource(
        payload["source"],
        {AuditTable(key): _frame(rows) for key, rows in payload["audit_tables"].items()},
    )
    return _frame(payload["candidates"]), source, _frame(payload["decision_bars"]), payload


def real_b0_view():
    candidates, source, bars, _payload = real_b0_fixture()
    frame, evidence = project_b0_candidates(
        candidates, source, decision_bars=bars, advertised_features=M0_FEATURES,
    )
    frame["entry_ts_utc"] = frame["envelope_ts_utc"]
    frame["feature_as_of_ts"] = frame["envelope_ts_utc"]
    return CandidateFeatureView(
        "real-fixture-view", "real-fixture-pair", "real-fixture-registry", frame,
        dict(TIER_FEATURE_REGISTRY), "descriptive_only", evidence,
    )
