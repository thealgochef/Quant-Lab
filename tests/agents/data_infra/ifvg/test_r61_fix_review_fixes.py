"""R6.1-FIX adversarial round — the main agent's fixes (reviews RA-04, RA-05).

* RA-04: the ``FrozenContract.model_copy`` enum guard recognises the SHAPE of
  an enum-typed field — a scalar ``Enum`` / ``Enum | None`` takes a member, a
  homogeneous ``tuple[Enum, ...]`` takes a tuple of members (each element
  checked), every other annotation passes through untouched; every
  registered identity payload's sequence-of-enum field survives a lawful
  copy.
* RA-05: every helper path (the supervised ladder, the CatBoost bundle rung,
  the logistic rung) defaults the label identity to the FULL consumed-column
  content hash and the ladder run is stamped ``content_hash_unpersisted``;
  ``ControlledFeatureStudyPayload.label_identity_source`` is required.
"""

from __future__ import annotations

from enum import StrEnum

import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.ml.comparison_rows import (
    label_artifact_content_id,
    label_content_hash,
)
from alpha_lab.agents.data_infra.ifvg.ml.controlled_feature_study import (
    ControlledFeatureStudyPayload,
    _example_controlled_study_payload,
)
from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import LOGISTIC_PROTOCOL_ID
from alpha_lab.agents.data_infra.ifvg.ml.supervised_ladder import run_supervised_ladder
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    FrozenContract,
    _enum_field_shape,
    registered_identity_pairs,
)
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
    PipelineSemanticSpecPayload,
    QuantLabPipelineStage,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_supervised import (
    M0_FEATURES,
    class_balanced_supervised_fixture,
)


class _Kind(StrEnum):
    A = "a"
    B = "b"


class _Other(StrEnum):
    X = "x"


class _Payload(FrozenContract):
    one: _Kind
    many: tuple[_Kind, ...] = ()
    maybe: _Kind | None = None
    either: _Kind | _Other | None = None
    name: str = "x"


# ── RA-04 ────────────────────────────────────────────────────────────────────


def test_enum_field_shapes_are_recognised():
    assert _enum_field_shape(_Kind) == (_Kind, "scalar")
    assert _enum_field_shape(_Kind | None) == (_Kind, "scalar")
    assert _enum_field_shape(tuple[_Kind, ...]) == (_Kind, "sequence")
    assert _enum_field_shape(list[_Kind]) == (_Kind, "sequence")
    assert _enum_field_shape(tuple[_Kind, ...] | None) == (_Kind, "sequence")
    # a union of two enums, a mapping and a plain type pass through
    assert _enum_field_shape(_Kind | _Other | None) == (None, None)
    assert _enum_field_shape(dict[str, _Kind]) == (None, None)
    assert _enum_field_shape(str) == (None, None)
    assert _enum_field_shape(tuple[str, ...]) == (None, None)


def test_model_copy_guard_accepts_lawful_members_and_containers_of_members():
    payload = _Payload(one=_Kind.A)
    assert payload.model_copy(update={"one": _Kind.B}).one is _Kind.B
    copied = payload.model_copy(update={"many": (_Kind.A, _Kind.B)})
    assert copied.many == (_Kind.A, _Kind.B)
    assert payload.model_copy(update={"many": ()}).many == ()
    assert payload.model_copy(update={"maybe": None}).maybe is None
    assert payload.model_copy(update={"maybe": _Kind.A}).maybe is _Kind.A
    assert payload.model_copy(update={"name": "y"}).name == "y"
    # a union of two enums is not guarded by the copy seam (pydantic owns it)
    assert payload.model_copy(update={"either": _Other.X}).either is _Other.X


def test_model_copy_guard_refuses_raw_strings_in_scalar_and_sequence_fields():
    payload = _Payload(one=_Kind.A)
    with pytest.raises(TypeError, match="enum"):
        payload.model_copy(update={"one": "b"})
    with pytest.raises(TypeError, match="enum"):
        payload.model_copy(update={"many": ("a", "b")})
    with pytest.raises(TypeError, match="enum"):
        payload.model_copy(update={"many": (_Kind.A, "b")})
    with pytest.raises(TypeError, match="enum"):
        payload.model_copy(update={"many": "ab"})
    with pytest.raises(TypeError, match="enum"):
        payload.model_copy(update={"maybe": "a"})


def test_every_registered_payload_survives_a_lawful_sequence_of_enum_copy():
    """The six identity-bearing ``tuple[Enum, ...]`` fields the reviewer
    enumerated (e.g. ``PipelineSemanticSpecPayload.stage_plan``) — and any
    other registered payload's sequence-of-enum field — accept a copy that
    re-supplies their own value."""

    checked = 0
    for pair in registered_identity_pairs():
        example = pair.example_factory()
        for name, field in type(example).model_fields.items():
            enum_cls, shape = _enum_field_shape(field.annotation)
            if enum_cls is None or shape != "sequence":
                continue
            value = getattr(example, name)
            assert example.model_copy(update={name: value}) == example
            checked += 1
            if value:
                with pytest.raises(TypeError, match="enum"):
                    example.model_copy(update={name: tuple(str(v) for v in value)})
    assert checked >= 1
    assert _enum_field_shape(PipelineSemanticSpecPayload.model_fields["stage_plan"].annotation) == (
        QuantLabPipelineStage,
        "sequence",
    )


# ── RA-05 ────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ladder_inputs():
    fixture = class_balanced_supervised_fixture()
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    return fixture, folds


def test_helper_ladder_runs_bind_the_full_consumed_column_hash_and_are_stamped(ladder_inputs):
    fixture, folds = ladder_inputs
    labels = fixture.labeled_candidates
    helper = run_supervised_ladder(fixture.view, labels, folds, tier=fixture.tier)
    assert helper.label_identity_source == "content_hash_unpersisted"
    exact = label_artifact_content_id("synthetic_fixture_labels_v1", labels)
    stamped = run_supervised_ladder(
        fixture.view, labels, folds, tier=fixture.tier, label_artifact_id=exact
    )
    assert stamped.label_identity_source == "label_artifact"
    assert label_artifact_content_id(None, labels) != label_content_hash(labels)
    # a caller cannot mislabel: an explicit source must be one of the two
    with pytest.raises(ValueError, match="label_identity_source"):
        run_supervised_ladder(
            fixture.view,
            labels,
            folds,
            tier=fixture.tier,
            label_artifact_id=exact,
            label_identity_source="made_up",
        )
    # a caller may declare the helper form explicitly even with an id in hand
    declared = run_supervised_ladder(
        fixture.view,
        labels,
        folds,
        tier=fixture.tier,
        label_artifact_id=label_artifact_content_id(None, labels),
        label_identity_source="content_hash_unpersisted",
    )
    assert declared.label_identity_source == "content_hash_unpersisted"


def test_bundle_ladder_feature_source_carries_the_full_column_label_hash(ladder_inputs):
    fixture, folds = ladder_inputs
    labels = fixture.labeled_candidates
    features = tuple(M0_FEATURES)
    run = run_supervised_ladder(
        fixture.view,
        labels,
        folds,
        bundle_features=features,
        bundle_ref="b" * 64,
        protocols=("reference_prevalence_v1", LOGISTIC_PROTOCOL_ID),
    )
    assert run.feature_source["label_artifact_id"] == label_artifact_content_id(None, labels)
    assert run.feature_source["label_artifact_id"] != label_content_hash(labels)
    assert run.label_identity_source == "content_hash_unpersisted"


def test_controlled_study_payload_requires_the_label_identity_source():
    fields = _example_controlled_study_payload().model_dump()
    assert fields["label_identity_source"] == "label_artifact"
    fields.pop("label_identity_source")
    with pytest.raises(ValueError, match="label_identity_source"):
        ControlledFeatureStudyPayload(**fields)
