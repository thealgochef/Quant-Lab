"""R6.1-FIX §3.6 (F-08) — exact label identity for persisted controlled
studies: the label artifact id binds EVERY label/economic column the study
consumes plus the registered label policy; an unpersisted helper run may
carry the full consumed-column content hash but is stamped
``content_hash_unpersisted`` and can never be saved."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_lab.agents.data_infra.ifvg.ml.comparison_rows import (
    LABEL_CONSUMED_COLUMNS,
    label_artifact_content_id,
    label_content_hash,
)
from alpha_lab.agents.data_infra.ifvg.ml.controlled_feature_study import (
    ControlledFeatureStudyPayload,
    load_controlled_feature_study,
    run_controlled_mbp1_study,
    save_controlled_feature_study,
)
from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import LOGISTIC_PROTOCOL_ID
from tests.agents.data_infra.ifvg.test_controlled_feature_study import study_inputs  # noqa: F401


def test_label_artifact_content_id_binds_every_consumed_column(study_inputs) -> None:  # noqa: F811
    _view, labels, _folds, _envelope, _frame = study_inputs
    base = label_artifact_content_id("policy_v1", labels)
    assert len(base) == 64
    assert label_artifact_content_id("policy_v2", labels) != base
    assert set(LABEL_CONSUMED_COLUMNS) >= {
        "candidate_id",
        "binary_target",
        "gross_r",
        "net_r",
        "trading_day",
        "setup_id",
        "entry_ts_utc",
        "resolution_ts_utc",
    }
    for column in LABEL_CONSUMED_COLUMNS:
        changed = labels.copy()
        value = changed.loc[changed.index[0], column]
        if column == "binary_target":
            replacement = 0 if value else 1
        elif isinstance(value, str):
            replacement = f"{value}x"
        elif isinstance(value, bool | np.bool_):
            replacement = not bool(value)
        else:
            replacement = float(value) + 1.0
        changed[column] = changed[column].astype(object)
        changed.loc[changed.index[0], column] = replacement
        assert label_artifact_content_id("policy_v1", changed) != base, column
    # the narrow pair hash is NOT the label artifact id
    assert label_content_hash(labels) != base
    # a frame lacking a consumed column is refused
    with pytest.raises(ValueError, match="gross_r"):
        label_artifact_content_id("policy_v1", labels.drop(columns=["gross_r"]))


def test_persisted_studies_require_the_exact_label_artifact_id(study_inputs, tmp_path):  # noqa: F811
    view, labels, folds, envelope, full_frame = study_inputs
    helper = run_controlled_mbp1_study(
        view,
        labels,
        folds,
        challenger_bundle_key="B2_CORE_ORDER_FLOW",
        mbp1_features=full_frame,
        mbp1_feature_artifact=envelope,
    )
    payload = helper.envelope.payload
    assert payload.label_identity_source == "content_hash_unpersisted"
    assert payload.label_artifact_id == label_artifact_content_id(None, labels)
    with pytest.raises(PermissionError, match="content_hash_unpersisted"):
        save_controlled_feature_study(tmp_path / "helper", helper)
    assert not (tmp_path / "helper").exists()
    exact = label_artifact_content_id("synthetic_fixture_labels_v1", labels)
    persisted = run_controlled_mbp1_study(
        view,
        labels,
        folds,
        challenger_bundle_key="B2_CORE_ORDER_FLOW",
        mbp1_features=full_frame,
        mbp1_feature_artifact=envelope,
        label_artifact_id=exact,
    )
    assert persisted.envelope.payload.label_identity_source == "label_artifact"
    assert persisted.envelope.payload.label_artifact_id == exact
    assert persisted.envelope.controlled_feature_study_id != (
        helper.envelope.controlled_feature_study_id
    )
    save_controlled_feature_study(tmp_path / "exact", persisted)
    reloaded = load_controlled_feature_study(
        tmp_path / "exact", persisted.envelope.controlled_feature_study_id
    )
    assert reloaded.payload.label_artifact_id == exact
    # the payload itself refuses a missing or malformed label artifact id
    fields = payload.model_dump()
    fields["label_artifact_id"] = None
    with pytest.raises(ValueError):
        ControlledFeatureStudyPayload(**fields)
    fields["label_artifact_id"] = "not-a-hash"
    with pytest.raises(ValueError):
        ControlledFeatureStudyPayload(**fields)
    fields["label_artifact_id"] = "a" * 64
    fields["label_identity_source"] = "made_up"
    with pytest.raises(ValueError):
        ControlledFeatureStudyPayload(**fields)
    assert payload.model_protocol_id == LOGISTIC_PROTOCOL_ID
