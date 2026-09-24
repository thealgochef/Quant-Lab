"""R5 — an unreachable minimum independent-day threshold is refused before launch.

The strategy metric counts distinct trading days among executed trades with
warmup excluded, so it can never exceed the evaluated (non-warmup) date count.
Every launch layer refuses a threshold above that count with a plain-English
explanation; none clamps or rewrites the saved value, stored charters stay
loadable, and the five-day verification policy is exempt.

All stores, drafts and dates here are synthetic and disposable. The observed
case (2,050 days over 107 evaluated dates after 10 warmup days) is modelled
with synthetic dates; the owner's saved draft is never read or modified.
"""

from __future__ import annotations

import copy
import hashlib
from datetime import date, timedelta

import pytest

from alpha_lab.agents.data_infra.ifvg.development_access import FROZEN_WARMUP_DATES
from alpha_lab.agents.data_infra.ifvg.presentation.charter_satisfiability import (
    StudyGoal,
    evaluate_charter_satisfiability,
)
from alpha_lab.agents.data_infra.ifvg.presentation.run_purpose import (
    EvidenceClass,
    RunPurpose,
)
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    CharterValidationError,
    DatePolicy,
    SearchCharterEnvelope,
    _example_charter_payload,
    save_charter,
    validate_charter,
)
from alpha_lab.agents.data_infra.ifvg.search.strategy_approval import STORE
from alpha_lab.agents.data_infra.ifvg.study_drafts import load_draft, new_draft, save_draft
from alpha_lab.agents.data_infra.ifvg.study_presentation import validate_benchmarks_step
from tests.agents.ifvg_search.test_strategy_approval import build_approved_study
from tests.agents.ifvg_search.test_strategy_approval_review import (  # noqa: F401
    _prepare,
    _record,
    request_fixture,
)
from tests.agents.test_ifvg_strategy_approval_ui import (  # noqa: F401
    _approval_paths,
    _button,
    _text,
    configured_study,
)

_AS_OF = "2026-09-23T00:00:00Z"
_TECHNICAL_TOKENS = ("namespace", "unmarked", "schema", "manifest", "runner-entry")


def _synthetic_evidence_dates(count: int) -> tuple[str, ...]:
    """``count`` lawful logical days from 2026-01-13 (Saturdays skipped).
    Synthetic: they model the evaluated-date COUNT, not the owner's list."""

    out: list[str] = []
    day = date(2026, 1, 13)
    while len(out) < count:
        if day.weekday() != 5:
            out.append(day.isoformat())
        day += timedelta(days=1)
    assert out[-1] <= "2026-06-10"
    return tuple(out)


_OBSERVED_EVIDENCE = _synthetic_evidence_dates(107)
_OBSERVED_MESSAGE = (
    "Minimum independent trading days is 2,050, but this study evaluates only 107 "
    "trading days (after 10 warmup days). No configuration can pass. Change this "
    "threshold to at most 107 before running; the saved value has not been changed."
)


def _development_charter(threshold, evidence=_OBSERVED_EVIDENCE):
    """A synthetic FSM search charter under the development access policy."""

    example = _example_charter_payload()
    return example.model_copy(
        update={
            "objective_policy": example.objective_policy.model_copy(
                update={
                    "pareto_objectives": ("net_expectancy_r",),
                    "feasibility_gates": example.objective_policy.feasibility_gates.model_copy(
                        update={"min_independent_days": threshold}
                    ),
                }
            ),
            "date_policy": DatePolicy(
                replay_dates=(*FROZEN_WARMUP_DATES, *evidence),
                warmup_dates=FROZEN_WARMUP_DATES,
                access_policy_id="development_explicit_dates_before_path_v2",
            ),
        }
    )


def _satisfiability(**overrides):
    base = dict(
        goal=StudyGoal.FSM_SEARCH,
        search_mode="fsm_config_search",
        axis_selections={"parent_retest_timeout_1m_bars": ("parent_retest_timeout_1m_bars.240",)},
        baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
        pareto_objectives=("net_expectancy_r",),
        tie_breaks=("profit_factor", "core_replay_id"),
        selected_contract_ids=(),
        launchable_contract_ids=(),
        purpose=RunPurpose.DEVELOPMENT_RESEARCH,
        evidence_class=EvidenceClass.REAL,
        prop_gates_configured=False,
        robustness_gates_configured=False,
    )
    base.update(overrides)
    return evaluate_charter_satisfiability(**base)


def _day_rule(report):
    return next(
        (rule for rule in report.rules if rule.rule_id == "independent_day_threshold_reachable"),
        None,
    )


def _assert_plain_english(text: str) -> None:
    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import _TECHNICAL

    assert not _TECHNICAL.search(text), text
    assert not any(token in text.lower() for token in _TECHNICAL_TOKENS), text


# ── shared helper ─────────────────────────────────────────────────────────────


def test_evaluated_count_excludes_warmup_and_repeats():
    from alpha_lab.agents.data_infra.ifvg.search.charter_day_threshold import (
        evaluated_date_count,
    )

    evidence = ("2026-01-13", "2026-01-14")
    assert evaluated_date_count((*FROZEN_WARMUP_DATES, *evidence), FROZEN_WARMUP_DATES) == 2
    assert evaluated_date_count((*evidence, *evidence), ()) == 2
    assert evaluated_date_count(FROZEN_WARMUP_DATES, FROZEN_WARMUP_DATES) == 0


@pytest.mark.parametrize(
    ("threshold", "passes"),
    [(0, True), (1, True), (106, True), (107, True), (107.0, True), (108, False), (2050, False)],
)
def test_helper_boundaries(threshold, passes):
    from alpha_lab.agents.data_infra.ifvg.search.charter_day_threshold import (
        check_day_threshold,
    )

    check = check_day_threshold(
        min_independent_days=threshold,
        replay_dates=(*FROZEN_WARMUP_DATES, *_OBSERVED_EVIDENCE),
        warmup_dates=FROZEN_WARMUP_DATES,
        access_policy_id="development_explicit_dates_before_path_v2",
    )
    assert check.applies and check.evaluated_dates == 107 and check.warmup_dates == 10
    assert check.passed is passes
    assert check.threshold == threshold  # reported, never clamped
    if not passes:
        assert "at most 107" in check.problem
        _assert_plain_english(check.problem)
    else:
        _assert_plain_english(check.detail)


def test_no_dates_selected_explains_instead_of_crashing():
    from alpha_lab.agents.data_infra.ifvg.search.charter_day_threshold import (
        check_day_threshold,
        draft_day_threshold_check,
    )

    for replay, warmup in (((), ()), (FROZEN_WARMUP_DATES, FROZEN_WARMUP_DATES)):
        check = check_day_threshold(
            min_independent_days=20,
            replay_dates=replay,
            warmup_dates=warmup,
            access_policy_id="development_explicit_dates_before_path_v2",
        )
        assert check.evaluated_dates == 0
        assert "no evaluation dates are selected yet" in check.problem
        _assert_plain_english(check.problem)
    # an empty or partly filled draft is explained, not a crash (charter default 20)
    for steps in ({}, {"validation": {"run_scope": "full_authorized_development"}}):
        check = draft_day_threshold_check(steps)
        assert check.threshold == 20
        assert "no evaluation dates are selected yet" in check.problem
    report = _satisfiability(strategy_gates={}, replay_dates=(), warmup_dates=())
    assert not report.passed
    assert "no evaluation dates are selected yet" in _day_rule(report).detail


def test_verification_policy_is_exempt():
    from alpha_lab.agents.data_infra.ifvg.search.charter_day_threshold import (
        check_day_threshold,
        draft_day_threshold_check,
    )

    check = check_day_threshold(
        min_independent_days=2050,
        replay_dates=("2026-06-04", "2026-06-05"),
        warmup_dates=(),
        access_policy_id="verification_fixed_allowlist_max5_v1",
    )
    assert not check.applies and check.passed
    assert draft_day_threshold_check(
        {
            "benchmarks": {"strategy_gates": {"min_independent_days": 2050.0}},
            "validation": {"run_scope": "verification_5d", "real_dates": ["2026-06-04"]},
        }
    ).passed
    example = _example_charter_payload()  # the five-day verification policy
    verification = example.model_copy(
        update={
            "objective_policy": example.objective_policy.model_copy(
                update={
                    "feasibility_gates": example.objective_policy.feasibility_gates.model_copy(
                        update={"min_independent_days": 2050}
                    )
                }
            )
        }
    )
    validate_charter(verification, as_of_utc=_AS_OF)
    report = _satisfiability(
        purpose=RunPurpose.IMPLEMENTATION_VERIFICATION,
        evidence_class=EvidenceClass.SYNTHETIC_FIXTURE,
        strategy_gates={"min_independent_days": 2050},
        replay_dates=("2026-06-04",),
    )
    assert report.passed and _day_rule(report) is None


# ── the observed saved-draft case (synthetic fixture) ─────────────────────────


def test_observed_draft_case_is_explained_and_the_saved_value_is_untouched(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.search.charter_day_threshold import (
        draft_day_threshold_check,
    )

    draft = new_draft("fsm_config_search", display_name="Observed threshold case")
    draft.steps["benchmarks"] = {
        "prop_gates": {},
        "strategy_gates": {"min_independent_days": 2050.0, "min_executed_trades": 30.0},
    }
    draft.steps["validation"] = {
        "evidence_class": "real",
        "run_scope": "full_authorized_development",
        "real_dates": list(_OBSERVED_EVIDENCE),
        "warmup_dates": list(FROZEN_WARMUP_DATES),
        "seed": 7,
        "worker_limit": 1,
    }
    path = save_draft(tmp_path, draft)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    steps_before = copy.deepcopy(draft.steps)

    check = draft_day_threshold_check(draft.steps)
    assert check.problem == _OBSERVED_MESSAGE
    _assert_plain_english(check.problem)
    # 2050.0 is a whole number: the thresholds step accepts it; the date
    # comparison belongs to review, approval, freeze and the worker
    assert validate_benchmarks_step(draft.steps["benchmarks"]) == {}
    report = _satisfiability(
        strategy_gates=draft.steps["benchmarks"]["strategy_gates"],
        replay_dates=draft.steps["validation"]["real_dates"],
        warmup_dates=draft.steps["validation"]["warmup_dates"],
    )
    assert not report.passed
    assert [rule.rule_id for rule in report.failures] == ["independent_day_threshold_reachable"]
    assert report.failures[0].detail == _OBSERVED_MESSAGE

    assert draft.steps == steps_before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    reloaded = load_draft(tmp_path, draft.draft_id)
    saved = reloaded.steps["benchmarks"]["strategy_gates"]["min_independent_days"]
    assert saved == 2050.0 and isinstance(saved, float)


def test_satisfiability_boundaries_and_backward_compatibility():
    evidence = _OBSERVED_EVIDENCE
    equal = _satisfiability(
        strategy_gates={"min_independent_days": 107.0},
        replay_dates=(*FROZEN_WARMUP_DATES, *evidence),
        warmup_dates=FROZEN_WARMUP_DATES,
    )
    assert equal.passed and _day_rule(equal).passed
    _assert_plain_english(_day_rule(equal).detail)
    over = _satisfiability(
        strategy_gates={"min_independent_days": 108},
        replay_dates=evidence,
        warmup_dates=FROZEN_WARMUP_DATES,
    )
    assert not over.passed and "at most 107" in _day_rule(over).detail
    # warmup dates listed among the replay dates are never counted
    warmup_only = _satisfiability(
        strategy_gates={"min_independent_days": 1},
        replay_dates=FROZEN_WARMUP_DATES,
        warmup_dates=FROZEN_WARMUP_DATES,
    )
    assert not warmup_only.passed
    # callers that do not supply the gates keep the previous rule set
    assert _day_rule(_satisfiability()) is None


# ── charter validation (freeze, save_charter, worker entry, child boundary) ───


@pytest.mark.parametrize(("threshold", "refused"), [(107, False), (108, True), (2050, True)])
def test_validate_charter_boundaries(threshold, refused):
    payload = _development_charter(threshold)
    if not refused:
        validate_charter(payload, as_of_utc=_AS_OF)
        return
    with pytest.raises(CharterValidationError, match="at most 107") as error:
        validate_charter(payload, as_of_utc=_AS_OF)
    _assert_plain_english(str(error.value))
    assert payload.objective_policy.feasibility_gates.min_independent_days == threshold


def test_observed_charter_message_is_exact():
    with pytest.raises(CharterValidationError) as error:
        validate_charter(_development_charter(2050), as_of_utc=_AS_OF)
    assert str(error.value) == _OBSERVED_MESSAGE


def test_historic_charter_with_unreachable_threshold_still_loads(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.study_providers import load_charter

    root = tmp_path / "search_test" / "v1"
    envelope = SearchCharterEnvelope.from_payload(_development_charter(2050))
    saved, reused = save_charter(root, envelope)  # a synthetic charter publishes unvalidated
    assert not reused
    loaded = load_charter(root, saved.search_id)
    assert loaded is not None and loaded.search_id == envelope.search_id
    assert loaded.payload.objective_policy.feasibility_gates.min_independent_days == 2050
    assert len(loaded.payload.date_policy.replay_dates) == 117
    assert SearchCharterEnvelope.from_payload(loaded.payload).search_id == envelope.search_id
    # inspectable, but never launchable
    with pytest.raises(CharterValidationError, match="at most 107"):
        validate_charter(loaded.payload, as_of_utc=_AS_OF)


def test_real_charter_publication_refuses_before_saving(tmp_path):
    root, payload, _approval, _requirements = build_approved_study(
        tmp_path, min_independent_days=2
    )
    envelope = SearchCharterEnvelope.from_payload(payload)
    with pytest.raises(CharterValidationError, match="evaluates only 1 trading day"):
        save_charter(root, envelope)
    assert not (root / "charters").exists()


def test_worker_entry_refuses_before_any_data_path(tmp_path, monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.search import strategy_executor as executor

    root, payload, _approval, _requirements = build_approved_study(
        tmp_path, min_independent_days=2
    )

    def forbidden(*args, **kwargs):
        pytest.fail("the worker must refuse before any date policy or data path")

    for name in ("DevelopmentReplayPolicy", "load_day_artifacts", "load_strategy_approval"):
        monkeypatch.setattr(executor, name, forbidden)
    with pytest.raises(CharterValidationError) as error:
        executor.search_strategy_development_entry(
            SearchCharterEnvelope.from_payload(payload), store_root=root
        )
    assert str(error.value) == (
        "Minimum independent trading days is 2, but this study evaluates only 1 trading "
        "day (after 10 warmup days). No configuration can pass. Change this threshold to "
        "at most 1 before running; the saved value has not been changed."
    )
    assert payload.objective_policy.feasibility_gates.min_independent_days == 2


def test_worker_entry_accepts_a_threshold_equal_to_the_evaluated_count(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.search import strategy_executor as executor

    root, payload, _approval, _requirements = build_approved_study(
        tmp_path, min_independent_days=1
    )
    wiring = executor.search_strategy_development_entry(
        SearchCharterEnvelope.from_payload(payload), store_root=root
    )
    assert set(wiring) == {"identity_resolver", "child_runner"}


# ── approval request: blocked, nothing saved ──────────────────────────────────


def test_approval_request_is_blocked_before_cache_checks_and_never_saved(
    request_fixture, monkeypatch  # noqa: F811
):
    from alpha_lab.agents.data_infra.ifvg.search import strategy_approval_review as service

    root, fields, requirements, cfg = request_fixture
    assert not _prepare(request_fixture).blockers  # 1 evaluated date, threshold 1
    raised = copy.deepcopy(fields)
    raised["objective_policy"]["feasibility_gates"]["min_independent_days"] = 2
    fixture = (root, raised, requirements, cfg)

    def forbidden(*args, **kwargs):
        pytest.fail("an impossible threshold must block before any cache check")

    monkeypatch.setattr(service, "_cache_review", forbidden)
    review = _prepare(fixture)
    assert review.blockers == (
        "Minimum independent trading days is 2, but this study evaluates only 1 trading "
        "day (after 10 warmup days). No configuration can pass. Change this threshold to "
        "at most 1 before running; the saved value has not been changed.",
    )
    assert review.configuration_rows == ()
    assert review.intent["objective_policy"]["feasibility_gates"]["min_independent_days"] == 2
    with pytest.raises(ValueError, match="Approval is blocked: Minimum independent trading"):
        _record(fixture, review)
    assert not (root / STORE).exists()


# ── the research wizard's approval panel (headless AppTest) ───────────────────


def test_research_wizard_shows_the_reason_and_keeps_run_disabled(configured_study):  # noqa: F811
    roots, draft, render = configured_study
    saved = load_draft(roots["draft_root"], draft.draft_id)
    saved.steps["benchmarks"]["strategy_gates"]["min_independent_days"] = 3
    save_draft(roots["draft_root"], saved)
    at = render()
    text = _text(at)
    assert (
        "Minimum independent trading days is 3, but this study evaluates only 2 trading "
        "days (after 10 warmup days). No configuration can pass. Change this threshold to "
        "at most 2 before running; the saved value has not been changed."
    ) in text
    assert _button(at, "Run study").disabled
    assert all(widget.label != "Save study approval" for widget in at.button)
    assert not _approval_paths(roots)
    stored = load_draft(roots["draft_root"], draft.draft_id)
    assert stored.steps["benchmarks"]["strategy_gates"]["min_independent_days"] == 3


# ── thresholds step: whole-number count gates (refused, never rounded) ────────


def test_thresholds_step_refuses_fractional_or_negative_count_gates():
    fields = {
        "strategy_gates": {
            "min_independent_days": 20.5,
            "min_executed_trades": -1.0,
            "max_time_under_water_days": 45.0,
            "min_net_expectancy_r": -0.25,
        },
        "robustness_gates": {"minimum_plateau_width": 1.5},
    }
    before = copy.deepcopy(fields)
    errors = validate_benchmarks_step(fields)
    assert set(errors) == {
        "strategy_gates.min_independent_days",
        "strategy_gates.min_executed_trades",
        "robustness_gates.minimum_plateau_width",
    }
    for message in errors.values():
        assert "whole number of zero or more" in message
        _assert_plain_english(message)
    assert fields == before
    assert validate_benchmarks_step(
        {"strategy_gates": {"min_independent_days": 2050.0, "min_executed_trades": 0}}
    ) == {}
