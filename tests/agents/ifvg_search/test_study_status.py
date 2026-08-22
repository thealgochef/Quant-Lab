"""Presentation status contracts (CS §13; FUX §§5, 31; FUX-LABEL-001)."""

from __future__ import annotations

from alpha_lab.agents.data_infra.ifvg.study_status import (
    DEV_BADGE_TEXT,
    EMPTY_STATE_PRESENTATIONS,
    FORBIDDEN_DISPLAY_PHRASES,
    FULL_SCOPE_WARNING_TEXT,
    HEATMAP_GLYPHS,
    NO_PASS_SENTENCE,
    NOT_RUN_STRATEGY_GATE_TEXT,
    RESULT_SCOPE_LABELS,
    ROUTE_LABELS,
    STATUS_PRESENTATIONS,
    TIMELINE_MARKERS,
    VERIFICATION_BADGE_TEXT,
    DisclosureLevel,
    EmptyStateKey,
    ResultScope,
    StudyStatusKey,
    StudyWorkspaceRoute,
    status_presentation,
)


def test_status_vocabulary_matches_cs13_exactly() -> None:
    assert [key.value for key in StudyStatusKey] == [
        "draft",
        "frozen",
        "queued",
        "running",
        "replay_failed",
        "strategy_rejected",
        "prop_rejected",
        "robust_finalist",
        "selected_representative",
        "superseded",
        "blocked",
    ]
    assert [level.value for level in DisclosureLevel] == [
        "summary",
        "analyst",
        "audit",
    ]
    assert [scope.value for scope in ResultScope] == [
        "candidate_research",
        "actual_executed_strategy",
        "prop_historical_closed_trade",
        "prop_1m_scenario",
        "prop_ordered_event_replay",
        "bootstrap_simulation",
        "stress_simulation",
    ]
    assert [route.value for route in StudyWorkspaceRoute] == [
        "new_study",
        "active_runs",
        "results",
        "history",
        "context_research",
    ]


def test_every_status_renders_glyph_word_and_help() -> None:
    """FUX §5.4: glyph + visible word + semantic class; never color-only."""

    assert set(STATUS_PRESENTATIONS) == set(StudyStatusKey)
    glyphs = set()
    for key in StudyStatusKey:
        presentation = status_presentation(key)
        assert presentation.glyph.strip()
        assert presentation.visible_label.strip()
        assert presentation.semantic_class.strip()
        assert len(presentation.help_text) > 20
        glyphs.add((presentation.glyph, presentation.semantic_class))
    # failure-class statuses stay glyph-distinguishable from success ones
    assert (
        STATUS_PRESENTATIONS[StudyStatusKey.ROBUST_FINALIST].glyph
        != STATUS_PRESENTATIONS[StudyStatusKey.REPLAY_FAILED].glyph
    )


def test_selected_representative_renders_the_development_label() -> None:
    presentation = status_presentation(StudyStatusKey.SELECTED_REPRESENTATIVE)
    assert presentation.visible_label == DEV_BADGE_TEXT
    assert DEV_BADGE_TEXT == "Development Exploratory Representative"


def test_exact_required_copy() -> None:
    assert NO_PASS_SENTENCE == "No configuration passed all benchmarks."
    assert VERIFICATION_BADGE_TEXT == "VERIFICATION ONLY — not research evidence"
    assert NOT_RUN_STRATEGY_GATE_TEXT == "Not run — strategy gate failed"
    assert FULL_SCOPE_WARNING_TEXT == (
        "This will run the full authorized development pipeline.\n"
        "It is not an implementation verification run."
    )
    assert ROUTE_LABELS[StudyWorkspaceRoute.NEW_STUDY] == "New Study"
    assert (
        RESULT_SCOPE_LABELS[ResultScope.PROP_1M_SCENARIO]
        == "Prop 1m Scenario / Approximation"
    )


def test_heatmap_and_timeline_glyphs_are_exact() -> None:
    """FUX §20 / §28 — the exact glyph classes and marker shapes."""

    assert HEATMAP_GLYPHS == {
        "stable_plateau": "◼",
        "knife_edge_point": "▲",
        "failed_region": "✕",
        "insufficient_data": "·",
        "blocked_cell": "⊘",
    }
    assert TIMELINE_MARKERS == {
        "payout": "▽",
        "fee": "◇",
        "breach": "✕",
        "replacement": "□",
    }


def test_every_section31_state_is_registered() -> None:
    expected = {
        "no_configurations_pass",
        "no_verified_firm_contract",
        "blocked_search_axis",
        "insufficient_sample",
        "child_replay_failed",
        "prop_not_run_strategy_gate",
        "no_model_result",
        "artifact_unavailable",
        "protected_range_refusal",
        "verification_authorization_missing",
        "feature_block_planned",
        "regime_algorithm_planned",
        "runner_executor_planned",
        # R5: the Full Pipeline Run surface exists — the R4-era
        # "pipeline_runner_planned" capability state is retired (no dead
        # vocabulary) and the dedicated no-runs presentation replaces it
        "pipeline_no_runs",
        "lineage_not_comparable",
        "browser_qa_unavailable",
    }
    assert set(EMPTY_STATE_PRESENTATIONS) == expected
    for state_id, presentation in EMPTY_STATE_PRESENTATIONS.items():
        assert presentation.heading.strip(), state_id
        assert len(presentation.explanation) > 30, state_id
        assert isinstance(presentation.key, EmptyStateKey)
    # both planned states share the CAPABILITY_PLANNED key by design
    assert (
        EMPTY_STATE_PRESENTATIONS["feature_block_planned"].key
        is EmptyStateKey.CAPABILITY_PLANNED
    )
    assert (
        EMPTY_STATE_PRESENTATIONS["regime_algorithm_planned"].key
        is EmptyStateKey.CAPABILITY_PLANNED
    )


def test_forbidden_phrases_absent_from_every_visible_string() -> None:
    """FUX §5.3 + TEST_MATRIX §3.7 P1-6 forbidden-wording scan."""

    visible: list[str] = [NO_PASS_SENTENCE, VERIFICATION_BADGE_TEXT]
    for presentation in STATUS_PRESENTATIONS.values():
        visible.extend((presentation.visible_label, presentation.help_text))
    for presentation in EMPTY_STATE_PRESENTATIONS.values():
        visible.extend(
            (
                presentation.heading,
                presentation.explanation,
                presentation.next_action or "",
            )
        )
    visible.extend(RESULT_SCOPE_LABELS.values())
    for text in visible:
        lowered = text.lower()
        for phrase in FORBIDDEN_DISPLAY_PHRASES:
            assert phrase not in lowered, (phrase, text)
        # standalone “Best”/“Winner”/“Validated” titles are prohibited too
        for banned_title in ("best", "winner"):
            assert not lowered.startswith(banned_title), text
