"""A11 and the response's section 3: the two approval paths compared by identity.

Review and approve (``scripts/ifvg_lab_new_funded.py``) and the earlier configurator
(``scripts/ifvg_funded_comparison_study.py``) are checked on the same ten saved drafts as
``test_new_funded.test_both_approval_paths_accept_and_refuse_the_same_drafts`` (both
build them with ``approval_case_specs``). Equal accept/refuse answers are necessary but not
sufficient. For every draft either path accepts on the running engine, this module also
compares the effective plan each path would approve (plan id, the ordered configurations
with their resolved identities, firms, dates, sizes and costs, processing clock), what each
would record as the approval and what each would dispatch.

Four more approvable drafts (``IDENTITY_CASES``: one firm only, another size and cost,
five settings varied, and on the half-exit engine a half exit with its own micro size and
cost) make sure several DISTINCT plans are compared, not only one. Every refused draft is
opened on both real pages, whose approval controls must be disabled or absent, with no
writer called.

Both screens run in Streamlit's headless AppTest against temporary stores, and their
approval and run buttons are really pressed. Every writer is a CAPTURING fake keyed by
(store root, plan id): ``save_plan``, ``record_owner_approval`` (which, like the real one,
approves only a plan saved in the same store), ``mark_frozen`` and ``_spawn`` record their
arguments and write nothing, and ``find_approval`` answers from the approvals captured in
the store it is asked about. The real launch check (``dispatch_problem``) and the real
``_freeze_and_launch`` run on those fakes. The real writers in their home modules are
replaced by functions that fail the test if they are ever called. Nothing is saved outside
``tmp_path``, nothing is approved in a real store and nothing is launched; no file under
``tmp_path`` is added or changed after the drafts are saved.

``core_source_identity`` is replaced by the test's ``FAKE_CORE`` (patch hash of 64 ones), so
every plan and approval id here is test-only and never equals a real approval's.

The matrix is printed once as ``APPROVAL_MATRIX_JSON {...}`` (run pytest with ``-s``): an
object with the engine, the fake-Core note and one entry per case.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_setup as fs  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.study_drafts import save_draft  # noqa: E402
from tests.agents.ifvg_lab.test_new_funded import (  # noqa: E402
    ALL_HOURS,
    DAYS,
    FAKE_CORE,
    HALF_EXIT_ENGINE,
    PACKAGE_GATES,
    S0,
    _earlier_accepts,
    _open,
    _review_accepts,
    _source,
    _text,
    _variation_draft,
    approval_case_draft,
    approval_case_specs,
)

PINNED, HALF_EXIT = "pinned", "half-exit"
ENGINE = HALF_EXIT if HALF_EXIT_ENGINE else PINNED
BASE = "S0_D80_W1_P1"
#: S0_D80_W1_P1's own gap rule, the other registered rule, today's withdrawal trigger
OWN_GAP, OTHER_GAP, TODAY_USD = fs.GAP_CLOSE, fs.GAP_WICK, 500
FIRMS = ["takeprofittrader", "myfundedfutures"]

#: the documented rule: only the first two drafts are approvable on the pinned engine; the
#: half-exit engine also approves the saved half-exit plan
ACCEPTED = {
    PINNED: frozenset({"no New funded comparison settings", "its settings at today's behavior"}),
    HALF_EXIT: frozenset({"no New funded comparison settings", "its settings at today's behavior",
                          "the half exit"}),
}

#: the ten saved drafts (evidence file): variation selections around S0_D80_W1_P1, the New
#: funded comparison settings saved under their own key (None: the key is absent), the firms
APPROVAL_CASES = [
    {"id": name,
     "inputs": {"source": "the verified daily-close strategy study", "base": BASE,
                "variation_selections": selections,
                "new_funded_comparison_settings": redesign,
                "firm_keys": list(firms) if firms is not None else FIRMS,
                "variation_sizes": {}},
     "expected": {"accepted_on_pinned_engine": name in ACCEPTED[PINNED],
                  "accepted_on_half_exit_engine": name in ACCEPTED[HALF_EXIT],
                  "plan_configurations": 64 if name == "the half exit" else 2}}
    for name, (selections, redesign, firms) in approval_case_specs(
        OWN_GAP, OTHER_GAP, TODAY_USD).items()
]

_HOURS = {"enabled_entry_sessions": [S0, ALL_HOURS]}
#: more approvable drafts, so several DISTINCT plans are compared on each engine (evidence
#: file); ``variation_sizes`` override the saved sizes (1 E-mini at $5.14 per fill, 10 micros
#: at $0.514 per fill for half exits)
IDENTITY_CASES = [
    {"id": "extra: MyFundedFutures only",
     "inputs": {"base": BASE, "variation_selections": _HOURS, "variation_sizes": {},
                "firm_keys": ["myfundedfutures"]},
     "expected": {"accepted_on_pinned_engine": True, "accepted_on_half_exit_engine": True,
                  "plan_configurations": 2}},
    {"id": "extra: two E-minis at $4.50 per fill",
     "inputs": {"base": BASE, "variation_selections": _HOURS,
                "variation_sizes": {"whole_quantity": 2, "whole_cost_mills": 4500},
                "firm_keys": FIRMS},
     "expected": {"accepted_on_pinned_engine": True, "accepted_on_half_exit_engine": True,
                  "plan_configurations": 2}},
    {"id": "extra: five settings varied",
     "inputs": {"base": BASE,
                "variation_selections": {
                    **_HOURS,
                    "tp_r_multiple": ["tp_r_multiple.1.0", "tp_r_multiple.2.0",
                                      "tp_r_multiple.3.0"],
                    "htf_timeframes": ["htf_timeframes.1H-4H", "htf_timeframes.1H"],
                    "parent_timeframes": ["parent_timeframes.1m-3m-5m-10m-15m-30m",
                                          "parent_timeframes.1m-5m-10m-15m-30m"],
                    "enable_shorts": ["enable_shorts.false", "enable_shorts.true"]},
                "variation_sizes": {}, "firm_keys": FIRMS},
     "expected": {"accepted_on_pinned_engine": True, "accepted_on_half_exit_engine": True,
                  "plan_configurations": 48}},
    {"id": "extra: half exit with 4 micros at $0.60 per fill",
     "inputs": {"base": BASE,
                "variation_selections": {"enabled_entry_sessions": [S0],
                                         "tp_r_multiple": ["tp_r_multiple.1.0",
                                                           "tp_r_multiple.3.0"],
                                         "exit_policy": [fs.WHOLE_EXIT, fs.HALF_EXIT]},
                "variation_sizes": {"half_quantity": 4, "half_cost_mills": 600},
                "firm_keys": FIRMS},
     "expected": {"accepted_on_pinned_engine": False, "accepted_on_half_exit_engine": True,
                  "plan_configurations": 3}},  # the half exit at 3R is left out
]
_EXPECTED_KEY = {PINNED: "accepted_on_pinned_engine", HALF_EXIT: "accepted_on_half_exit_engine"}
FAKE_CORE_NOTE = ("core_source_identity is replaced by the test's FAKE_CORE (patch_sha256 of 64 "
                  "ones), so these plan and approval ids are test-only and never equal a real "
                  "approval's.")

#: the shared count follows the saved membership (evidence file)
COUNT_CASES = [
    {"id": "one configuration",
     "inputs": {"variation_selections": {"enabled_entry_sessions": [S0]}},
     "expected": {"configurations": 1}},
    {"id": "two entry hours",
     "inputs": {"variation_selections": {"enabled_entry_sessions": [S0, ALL_HOURS]}},
     "expected": {"configurations": 2}},
    {"id": "two entry hours x two targets",
     "inputs": {"variation_selections": {"enabled_entry_sessions": [S0, ALL_HOURS],
                                         "tp_r_multiple": ["tp_r_multiple.1.0",
                                                           "tp_r_multiple.3.0"]}},
     "expected": {"configurations": 4}},
    {"id": "two entry hours x three targets x two directions",
     "inputs": {"variation_selections": {"enabled_entry_sessions": [S0, ALL_HOURS],
                                         "tp_r_multiple": ["tp_r_multiple.1.0",
                                                           "tp_r_multiple.2.0",
                                                           "tp_r_multiple.3.0"],
                                         "enable_shorts": ["enable_shorts.false",
                                                           "enable_shorts.true"]}},
     "expected": {"configurations": 12}},
]

_UNCHANGED = ("Needs your decision. The saved limit is 3 trading days under water; it stays as "
              "saved, and this page doesn't propose another value.")
#: the time-under-water limit is a neutral pending decision (evidence file)
UNDER_WATER_CASES = [
    {"id": "saved limit, nothing typed",
     "inputs": {"source_study_limit_days": 3, "saved_in_draft": {}, "typed": {}},
     "expected": {"status": "decision", "text": "3", "changed": False, "message": _UNCHANGED}},
    {"id": "the saved value typed again",
     "inputs": {"source_study_limit_days": 3, "saved_in_draft": {},
                "typed": {"max_days_under_water": "3"}},
     "expected": {"status": "decision", "text": "3", "changed": False, "message": _UNCHANGED}},
    {"id": "a different typed value",
     "inputs": {"source_study_limit_days": 3, "saved_in_draft": {},
                "typed": {"max_days_under_water": "5"}},
     "expected": {"status": "decision", "text": "5", "changed": True,
                  "message": "Needs your decision. You entered 5; the saved limit is 3. This page "
                             "doesn't propose a value."}},
    {"id": "a different value saved in the draft",
     "inputs": {"source_study_limit_days": 3, "saved_in_draft": {"max_days_under_water": 20},
                "typed": {}},
     "expected": {"status": "decision", "text": "20", "changed": True,
                  "message": "Needs your decision. You entered 20; the saved limit is 3. This page "
                             "doesn't propose a value."}},
]

REVIEW_AGREE, REVIEW_RUN = "ifvg_lab_v1_approve_agree", "ifvg_lab_v1_approve_run"
EARLIER_AGREE, EARLIER_APPROVE = "ifvg_fcmp_agree", "ifvg_fcmp_approve"
EARLIER_RUN = "ifvg_fcmp_run"
DRAFT_KEY = "ifvg_study_v1_draft_id"


# ── capturing fakes ───────────────────────────────────────────────────────


def _steps_digest(draft: Any) -> str:
    return hashlib.sha256(json.dumps(draft.steps, sort_keys=True, default=str)
                          .encode("utf-8")).hexdigest()


def _store_key(store_root: Any) -> str:
    return str(Path(store_root))


class Recorder:
    """Capturing fakes for every writer on both approval paths (nothing is written).

    Each driven path (``start``) gets its own event list and its own captured plans and
    approvals, keyed by (store root, plan id), so the second path saves and approves
    afresh. As in the real store, an approval needs the plan saved in the SAME store, and
    ``find_approval`` finds only approvals of the store it is asked about. A writer that
    runs while no path is being driven fails the test.
    """

    def __init__(self) -> None:
        self.path: str | None = None
        self.events: dict[str, list[tuple]] = {}
        self.plans: dict[str, dict[tuple[str, str], Any]] = {}
        self.approvals: dict[str, dict[tuple[str, str], Any]] = {}

    def start(self, path: str) -> None:
        self.path = path
        self.events[path] = []
        self.plans[path] = {}
        self.approvals[path] = {}

    def stop(self) -> None:
        self.path = None

    def _log(self, *event: Any) -> None:
        if self.path is None:
            raise AssertionError(f"{event[0]} ran while no approval was being driven")
        self.events[self.path].append(event)

    def install(self, monkeypatch, screen) -> None:
        from alpha_lab.propsim.funded.comparison_plan import (
            FundedComparisonApprovalEnvelope,
            FundedComparisonApprovalPayload,
        )

        real_freeze = screen._freeze_and_launch
        real_dispatch = screen.dispatch_problem
        real_approve = screen._approve_and_run

        def save_plan(store_root, envelope):
            store, plan_id = _store_key(store_root), envelope.funded_comparison_plan_id
            self._log("save_plan", store, envelope)
            self.plans[self.path].setdefault((store, plan_id), envelope)
            return plan_id

        def record_owner_approval(store_root, plan_id, *, approved_on, channel, statement,
                                  scope):
            store = _store_key(store_root)
            if self.path is None or (store, plan_id) not in self.plans[self.path]:
                # as the real recorder: approve only a plan saved in this same store
                raise ValueError("approve a saved plan only")
            # the exact record the real function would store (validated, never written)
            envelope = FundedComparisonApprovalEnvelope.from_payload(
                FundedComparisonApprovalPayload(
                    funded_comparison_plan_id=plan_id, approved_on=approved_on,
                    channel=channel, statement=statement, scope=scope))
            self._log("record_owner_approval", store, envelope)
            self.approvals[self.path].setdefault((store, plan_id), envelope)
            return envelope.funded_comparison_approval_id

        def find_approval(store_root, plan_id):
            if self.path is None:
                return None
            return self.approvals[self.path].get((_store_key(store_root), plan_id))

        def mark_frozen(root, draft, *, search_id, **_kwargs):
            self._log("mark_frozen", str(root), draft.draft_id, search_id, _steps_digest(draft))
            return draft

        def spawn(command):
            self._log("_spawn", [str(part) for part in command])
            return 4242

        def freeze_and_launch(st_module, draft, envelope, roots):
            self._log("_freeze_and_launch", draft.draft_id, envelope.funded_comparison_plan_id)
            return real_freeze(st_module, draft, envelope, roots)

        def dispatch_problem(roots, draft_id, envelope):
            problem = real_dispatch(roots, draft_id, envelope)
            self._log("dispatch_problem", draft_id, envelope.funded_comparison_plan_id, problem)
            return problem

        def approve_and_run(st_module, draft, envelope, roots, scope, *, source):
            if self.path is not None:  # the plan the earlier configurator shows for approval
                self.events[self.path].append(("shown", draft.draft_id, envelope, scope))
            return real_approve(st_module, draft, envelope, roots, scope, source=source)

        # the launch waits up to ten seconds for the worker's first progress: skip the wait
        clock = itertools.count(0.0, 60.0)
        fake_time = SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda _seconds: None)
        for name, fake in (("save_plan", save_plan),
                           ("record_owner_approval", record_owner_approval),
                           ("find_approval", find_approval), ("mark_frozen", mark_frozen),
                           ("_spawn", spawn), ("_freeze_and_launch", freeze_and_launch),
                           ("dispatch_problem", dispatch_problem),
                           ("_approve_and_run", approve_and_run), ("time", fake_time)):
            monkeypatch.setattr(screen, name, fake)


@pytest.fixture
def ac_env(tmp_path, monkeypatch):
    source = _source()
    if source is None:
        pytest.skip("the verified daily-close study archive is not available locally")
    # the screens that bind the draft writer by name are imported before the guards below,
    # so a later import can never keep a guard after this test
    import ifvg_funded_comparison_study as screen
    import ifvg_funded_study
    import ifvg_study_wizard
    import ifvg_workspace

    from alpha_lab.agents.data_infra.ifvg import study_drafts
    from alpha_lab.propsim.funded import comparison_study, core_identity

    roots = {
        "repo_root": Path(__file__).resolve().parents[3], "store_root": tmp_path / "store",
        "store_roots": {"research": tmp_path / "store"},
        "draft_root": tmp_path / "drafts", "state_root": tmp_path / "jobs",
        "pipeline_state_root": tmp_path / "pipelines",
        "funded_comparison_state_root": tmp_path / "comparison_jobs",
        "reports_root": tmp_path / "reports",
    }
    monkeypatch.setattr(ifvg_workspace, "_TEST_ROOTS", roots, raising=False)
    monkeypatch.setattr(core_identity, "core_source_identity", lambda: dict(FAKE_CORE))
    called: list[str] = []

    def forbidden(name):
        def fail(*args, **kwargs):
            called.append(name)
            raise AssertionError(f"the real {name} must never run in this test")

        return fail

    # the real writers, wherever they might be reached from
    for module, name in ((comparison_study, "save_plan"),
                         (comparison_study, "record_owner_approval"),
                         (study_drafts, "mark_frozen")):
        monkeypatch.setattr(module, name, forbidden(f"{module.__name__}.{name}"))
    for module in (ifvg_funded_study, ifvg_study_wizard):  # their own by-name bindings
        monkeypatch.setattr(module, "mark_frozen",
                            forbidden(f"{study_drafts.__name__}.mark_frozen"))
    recorder = Recorder()
    recorder.install(monkeypatch, screen)
    return {"roots": roots, "source": source, "sources": {source.package.run_id: source},
            "called": called, "rec": recorder, "tmp": tmp_path}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) \
        if root.exists() else []


def _snapshot(root: Path) -> dict[str, str]:
    """Every file under ``root`` with its SHA-256 (the no-write check)."""

    return {name: _digest(root / name) for name in _files(root)}


# ── what each path would approve, record and dispatch ────────────────────


def _membership(envelope) -> dict[str, Any]:
    """The effective plan field by field: the configurations an approval covers."""

    plan = envelope.payload
    source = plan.source.model_dump(mode="json")
    return {
        "plan_id": envelope.funded_comparison_plan_id,
        "base_configuration": plan.base_configuration,
        "configurations": [
            {"name": v.name, "axis_value_ids": [list(pair) for pair in v.axis_value_ids],
             "resolved_section_config_hash": v.resolved_section_config_hash,
             "exit_policy": v.exit_policy, "instrument": v.instrument, "quantity": v.quantity,
             "cost_per_contract_mills": v.cost_per_contract_mills,
             "in_verified_study": v.in_verified_study,
             "cache_configuration": v.cache_configuration}
            for v in plan.variants],
        "firms": [p.firm_key for p in plan.firm_profiles],
        "source": {k: v for k, v in source.items()
                   if k not in ("warmup_dates", "evaluation_dates", "cutoff_utc")},
        "dates": {"warmup": source["warmup_dates"], "evaluation": source["evaluation_dates"],
                  "cutoff_utc": source["cutoff_utc"]},
        "processing": plan.processing.model_dump(mode="json"),
        "core_source": plan.core_source.model_dump(mode="json"),
        "whole_payload": plan.model_dump(mode="json"),
    }


def _named(events, name):
    return [e for e in events if e[0] == name]


def _approval(events, draft_id: str) -> dict[str, Any]:
    """What the path recorded as the approval (the approval record binds the plan id only;
    the draft is bound at launch, where ``dispatch_problem`` re-reads and rebuilds it)."""

    saves = _named(events, "save_plan")
    ((_, store, record),) = _named(events, "record_owner_approval")
    saved = saves[0]
    payload = record.payload
    return {"approval_id": record.funded_comparison_approval_id,
            "plan_id": payload.funded_comparison_plan_id,
            "approved_on": payload.approved_on, "approved_by": payload.approved_by,
            "channel": payload.channel, "statement": payload.statement, "scope": payload.scope,
            "store_root": store, "saved_plan_store_root": saved[1],
            "saved_plan_id": saved[2].funded_comparison_plan_id,
            "source_binding": saved[2].payload.source.model_dump(mode="json"),
            "draft_id": draft_id}


def _dispatch(events) -> dict[str, Any]:
    """What the path dispatched: the checked launch, the frozen draft and the worker command."""

    ((_, draft_id, plan_id),) = _named(events, "_freeze_and_launch")
    ((_, checked_draft, checked_plan, problem),) = _named(events, "dispatch_problem")
    ((_, root, frozen_draft, search_id, steps),) = _named(events, "mark_frozen")
    ((_, command),) = _named(events, "_spawn")
    return {"draft_id": draft_id, "plan_id": plan_id,
            "dispatch_check": [checked_draft, checked_plan, problem],
            "frozen": [root, frozen_draft, search_id, steps], "command": command,
            "plan_saved_before_launch": [[e[1], e[2].funded_comparison_plan_id]
                                         for e in _named(events, "save_plan")]}


def _normalised_command(command: list[str], env) -> list[str]:
    """The worker command with this computer's paths replaced by placeholders."""

    swaps = ((sys.executable, "<python>"), (str(env["tmp"]), "<tmp>"),
             (str(env["roots"]["repo_root"]), "<repo>"))
    out = []
    for part in command:
        for real, placeholder in swaps:
            part = part.replace(real, placeholder)
        out.append(part.replace("\\", "/"))
    return out


def _writer_sequence(events) -> list[str]:
    return [e[0] for e in events if e[0] != "shown"]


def _enabled(at, kind: str, key: str) -> bool:
    """Whether a control is on the page AND can be used (absent counts as not enabled)."""

    try:
        return not getattr(at, kind)(key=key).disabled
    except KeyError:
        return False


def _present(at, kind: str, key: str) -> bool:
    try:
        getattr(at, kind)(key=key)
    except KeyError:
        return False
    return True


def _drive_review(env, draft_id: str):
    """Review and approve: tick the box and press "Record approval and run" (AppTest)."""

    rec = env["rec"]
    rec.start("review")
    try:
        at = _open("approve_funded", draft_id)
        offered = _enabled(at, "checkbox", REVIEW_AGREE)
        at.checkbox(key=REVIEW_AGREE).check().run()
        assert not at.exception, at.exception
        label = at.checkbox(key=REVIEW_AGREE).label
        assert not at.button(key=REVIEW_RUN).disabled
        at.button(key=REVIEW_RUN).click().run()
        assert not at.exception, at.exception
    finally:
        rec.stop()
    return label, at.session_state[DRAFT_KEY], _text(at), offered


def _drive_earlier(env, draft_id: str):
    """The earlier configurator: tick, "Record my approval", then "Run funded comparison"."""

    rec = env["rec"]
    rec.start("earlier")
    try:
        at = _open("new", draft_id)
        offered = _enabled(at, "checkbox", EARLIER_AGREE)
        at.checkbox(key=EARLIER_AGREE).check().run()
        assert not at.exception, at.exception
        assert not at.button(key=EARLIER_APPROVE).disabled
        at.button(key=EARLIER_APPROVE).click().run()
        assert not at.exception, at.exception
        if at.button(key=EARLIER_RUN).disabled:  # the page reruns itself after recording
            at.run()
        assert not at.button(key=EARLIER_RUN).disabled
        at.button(key=EARLIER_RUN).click().run()
        assert not at.exception, at.exception
    finally:
        rec.stop()
    return _text(at), offered


def _refused_pages(env, name: str, draft_id: str) -> dict[str, Any]:
    """Both real pages for a draft the helper models refuse (AppTest; nothing pressed).

    The pages run their own gating (the earlier page's ``_render_variations``: firm
    filtering, sizes, the engine identity, left-out combinations). Returns whether each
    offers an approval control, and every writer event each page produced.
    """

    rec, out = env["rec"], {}
    for page, screen, controls in (
            ("review", "approve_funded", (("checkbox", REVIEW_AGREE), ("button", REVIEW_RUN))),
            ("earlier", "new", (("checkbox", EARLIER_AGREE), ("button", EARLIER_APPROVE),
                                ("button", EARLIER_RUN)))):
        path = f"refused {page}: {name}"
        rec.start(path)
        try:
            at = _open(screen, draft_id)
        finally:
            rec.stop()
        out[page] = any(_enabled(at, kind, key) for kind, key in controls)
        out[page + "_present"] = [key for kind, key in controls if _present(at, kind, key)]
        out[page + "_writes"] = _writer_sequence(rec.events[path])
        out[page + "_captured"] = sorted(rec.plans[path]) + sorted(rec.approvals[path])
    return out


def _count_in(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1).replace(",", "")) if match else None


def _shown_counts(env, draft) -> dict[str, int | None]:
    """The configuration count each screen shows for one SAVED draft (fix F4).

    My studies: ``ifvg_lab_library._draft_check`` then ``library.draft_summary`` (the Drafts
    row's text). New funded comparison, as ``render_funded_setup`` branches: an editable
    draft shows ``ifvg_lab_new_funded._choices_count`` on the page state it builds (what
    ``_plan_panel`` prints); a draft this engine can't represent opens read-only and shows
    ``funded_setup.plan_count`` of the saved draft (``_setup_read_only``'s first sentence).
    Review and approve: ``funded_setup.plan_count`` of ``_review_state`` (its tile and
    approval sentence).
    """

    import ifvg_lab_library as library_screen
    import ifvg_lab_new_funded as nf

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import library as lib

    check = library_screen._draft_check(draft.draft_id, str(draft.updated_at_utc), draft)
    my_studies = _count_in(lib.draft_summary(draft, check), r"^(?:Up to )?([\d,]+) configuration")
    state = nf._review_state(env["roots"], draft, env["sources"])
    named, named_problem = nf._named()
    page = nf._Setup(roots=env["roots"], root=Path(env["roots"]["draft_root"]), draft=draft,
                     persisted=True, source=state.source, base_ids=state.base_ids,
                     offered=fs.offered_options(state.base_ids), named=named,
                     named_problem=named_problem)
    if state.check.runnable:
        choices = fs.choices_from_draft(draft, base_ids=state.base_ids, offered=page.offered)
        setup = nf._choices_count(page, choices).total
    else:
        setup = fs.plan_count(state.check, draft).total
    review = fs.plan_count(state.check, state.draft).total
    return {"my_studies": my_studies, "new_funded_comparison": setup, "review_and_approve": review}


def _saved_cases(env) -> list[dict[str, Any]]:
    """The ten F11 drafts and the identity drafts, saved in the temporary draft folder."""

    source = env["source"]
    base_ids = dict(source.by_name[BASE].axis_value_ids)
    own_gap = fs.base_value(fs.GAP_AXIS, base_ids)
    other_gap = fs.GAP_WICK if own_gap != fs.GAP_WICK else fs.GAP_CLOSE
    today = fs.today_trigger_usd()
    # the evidence constant is exactly the ten drafts the F11 test builds from this source
    assert (own_gap, other_gap, today) == (OWN_GAP, OTHER_GAP, TODAY_USD)
    specs = approval_case_specs(own_gap, other_gap, today)
    assert [c["id"] for c in APPROVAL_CASES] == list(specs) and len(specs) == 10
    out = []
    for case_set, cases in (("ten", APPROVAL_CASES), ("identity", IDENTITY_CASES)):
        for case in cases:
            inputs = case["inputs"]
            if case_set == "ten":
                selections, redesign, firms = specs[case["id"]]
                assert (inputs["variation_selections"],
                        inputs["new_funded_comparison_settings"]) == (selections, redesign)
            else:
                selections, redesign, firms = (inputs["variation_selections"], None,
                                               inputs["firm_keys"])
            draft = approval_case_draft(source, selections, redesign, firms)
            draft.steps["review"]["funded_comparison"]["variation"].update(
                inputs["variation_sizes"])
            out.append({"name": case["id"], "set": case_set, "draft": draft,
                        "path": save_draft(env["roots"]["draft_root"], draft),
                        "expected": case["expected"][_EXPECTED_KEY[ENGINE]],
                        "expected_other": case["expected"][_EXPECTED_KEY[_OTHER_ENGINE]],
                        "configurations": case["expected"]["plan_configurations"],
                        "firms": inputs["firm_keys"]})
    return out


_OTHER_ENGINE = PINNED if ENGINE == HALF_EXIT else HALF_EXIT
_EMPTY_ROW = {"review_page_approval_enabled": None, "earlier_page_approval_enabled": None,
              "plan_id_review": None, "plan_id_earlier": None, "membership_equal": None,
              "approval_equal": None, "dispatch_equal": None, "configuration_count": None,
              "membership_size": None, "approval_id": None, "approval_scope": None,
              "approval_statement": None, "dispatch_command": None}


# ── the ten-case matrix ───────────────────────────────────────────────────


def test_a11_both_paths_approve_record_and_dispatch_the_identical_plan(ac_env):
    """A11 (response section 3): the ten-case expected/actual matrix with plan identity.

    For each of the ten saved drafts (and the four ``IDENTITY_CASES``): the documented
    expectation, Review and approve's answer (``_review_accepts``: no blocker on
    ``_review_state``) and the earlier configurator's (``_earlier_accepts``: "Record my
    approval" enabled). A refused draft is then opened on both real pages (AppTest), whose
    approval controls must be disabled or absent, with no writer called. For every accepted
    draft both screens are opened in AppTest and their buttons pressed: Review and
    approve's "Record approval and run" (``_approval_card`` -> ``_record_and_run``), the
    earlier configurator's "Record my approval" and then "Run funded comparison"
    (``_approve_and_run``). Their effective plans, captured approvals (store-keyed) and
    dispatches (real ``_freeze_and_launch`` and ``dispatch_problem`` on capturing fakes)
    must be identical; at least two distinct plans are compared; and the configuration
    count shared by My studies, New funded comparison and Review and approve must equal the
    plan's membership.
    """

    import ifvg_lab_new_funded as nf

    from alpha_lab.propsim.funded.comparison_draft import check_saved_comparison, saved_settings
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    env = ac_env
    roots = env["roots"]
    cases = _saved_cases(env)
    before = _snapshot(env["tmp"])  # after the drafts are saved, nothing may be added
    rows, details, refused = [], {}, {}
    for case in cases:
        name, draft, path = case["name"], case["draft"], case["path"]
        draft_id = path.parent.name
        review = _review_accepts(env, draft, env["sources"])
        earlier = _earlier_accepts(env, draft, env["sources"])
        check = check_saved_comparison(saved_settings(draft), {}, env["sources"])
        row = {"case": name, "case_set": case["set"], "expected_accept": case["expected"],
               "review_accept": review, "earlier_accept": earlier, "engine": ENGINE,
               "executed": True, **_EMPTY_ROW,
               "configuration_count": fs.plan_count(check, draft).total}
        if not (review and earlier):
            pages = _refused_pages(env, name, draft_id)
            row.update({"review_page_approval_enabled": pages["review"],
                        "earlier_page_approval_enabled": pages["earlier"]})
            refused[name] = {**pages, "runnable_here": check.runnable}
            rows.append(row)
            continue
        state = nf._review_state(roots, draft, env["sources"])
        label, review_draft_id, review_text, review_on = _drive_review(env, draft_id)
        earlier_text, earlier_on = _drive_earlier(env, draft_id)
        events = env["rec"].events
        shown_earlier = [e for e in events["earlier"] if e[0] == "shown"]
        approved = {p: _named(events[p], "save_plan")[0][2] for p in ("review", "earlier")}
        members = {p: _membership(approved[p]) for p in approved}
        approvals = {"review": _approval(events["review"], review_draft_id),
                     "earlier": _approval(events["earlier"], shown_earlier[0][1])}
        dispatches = {p: _dispatch(events[p]) for p in ("review", "earlier")}
        counts = {**_shown_counts(env, draft),
                  "review_approval_sentence": _count_in(
                      label, r"exact plan: ([\d,]+) configuration")}
        row.update({
            "review_page_approval_enabled": review_on,
            "earlier_page_approval_enabled": earlier_on,
            "plan_id_review": approvals["review"]["plan_id"],
            "plan_id_earlier": approvals["earlier"]["plan_id"],
            "membership_equal": members["review"] == members["earlier"],
            "approval_equal": approvals["review"] == approvals["earlier"],
            "dispatch_equal": dispatches["review"] == dispatches["earlier"],
            "configuration_count": counts["review_and_approve"],
            "membership_size": len(members["review"]["configurations"]),
            "approval_id": approvals["review"]["approval_id"],
            "approval_scope": approvals["review"]["scope"],
            "approval_statement": approvals["review"]["statement"],
            "dispatch_command": _normalised_command(dispatches["review"]["command"], env),
        })
        details[name] = {"case": case, "shown_review": state.envelope,
                         "shown_earlier_ids": {e[2].funded_comparison_plan_id
                                               for e in shown_earlier},
                         "members": members, "approvals": approvals,
                         "dispatches": dispatches, "counts": counts,
                         "events": {p: _writer_sequence(events[p])
                                    for p in ("review", "earlier")},
                         "draft_id": draft_id, "texts": (review_text, earlier_text)}
        rows.append(row)
    for case in cases:  # an expectation that differs on the engine not running here
        if case["expected_other"] != case["expected"]:
            rows.append({"case": case["name"], "case_set": case["set"],
                         "expected_accept": case["expected_other"], "review_accept": None,
                         "earlier_accept": None, "engine": _OTHER_ENGINE,
                         "executed": f"needs the {_OTHER_ENGINE} engine", **_EMPTY_ROW})
    print("APPROVAL_MATRIX_JSON " + json.dumps({
        "engine": ENGINE, "fake_core": True,
        "fake_core_identity": {k: FAKE_CORE[k] for k in ("base_commit", "branch",
                                                         "patch_sha256")},
        "note": FAKE_CORE_NOTE, "cases": rows}))

    executed = [row for row in rows if row["executed"] is True]
    assert len([row for row in executed if row["case_set"] == "ten"]) == 10
    for row in executed:  # the answers (the F11 check), then what the real pages offer
        expected = row["expected_accept"]
        assert (row["review_accept"], row["earlier_accept"]) == (expected, expected), row
        assert (row["review_page_approval_enabled"],
                row["earlier_page_approval_enabled"]) == (expected, expected), row
    assert {row["case"] for row in executed
            if row["case_set"] == "ten" and row["review_accept"]} == ACCEPTED[ENGINE]
    for name, pages in refused.items():  # nothing was saved, approved or run behind them
        assert pages["review_writes"] == pages["earlier_writes"] == [], (name, pages)
        assert pages["review_captured"] == pages["earlier_captured"] == [], (name, pages)
        # the approval area really rendered: Review and approve always shows it, switched
        # off; the earlier page shows it switched off for a draft it can edit, and none at
        # all for a draft this engine can't represent (read-only view)
        assert pages["review_present"] == [REVIEW_AGREE, REVIEW_RUN], (name, pages)
        assert pages["earlier_present"] == (
            [EARLIER_AGREE, EARLIER_APPROVE, EARLIER_RUN] if pages["runnable_here"] else []), (
            name, pages)
    store = _store_key(roots["store_root"])
    for name, info in details.items():
        case, draft_id = info["case"], info["draft_id"]
        members, approvals, dispatches = info["members"], info["approvals"], info["dispatches"]
        plan_id = members["review"]["plan_id"]
        # each path approves the plan it shows, rebuilt from the same saved draft
        assert info["shown_review"].funded_comparison_plan_id == plan_id, name
        assert info["shown_earlier_ids"] == {plan_id}, name  # on every rerun of that page
        assert members["review"] == members["earlier"], name
        configurations = members["review"]["configurations"]
        assert [c["name"] for c in configurations] == sorted(c["name"] for c in configurations)
        assert len(configurations) == case["configurations"], name
        # the saved firms, sizes and costs, exactly
        assert members["review"]["firms"] == [k for k in FIRM_PROFILES if k in case["firms"]]
        sizes = saved_settings(case["draft"])["variation"]
        for config in configurations:
            half = config["exit_policy"] != "fixed_target_v1"
            assert (config["instrument"], config["quantity"],
                    config["cost_per_contract_mills"]) == (
                ("micro", sizes["half_quantity"], sizes["half_cost_mills"]) if half
                else ("mini", sizes["whole_quantity"], sizes["whole_cost_mills"])), (
                name, config)
            if half:  # a half exit exists only at the 1R target
                assert dict(map(tuple, config["axis_value_ids"])).get(
                    "tp_r_multiple", fs.ONE_R) == fs.ONE_R, (name, config)
        assert members["review"]["dates"]["evaluation"] == list(
            env["source"].evaluation_dates)
        # the same approval record: plan, owner, channel, statement, scope, source, draft
        assert approvals["review"] == approvals["earlier"], name
        assert approvals["review"]["plan_id"] == plan_id
        assert approvals["review"]["draft_id"] == draft_id
        assert approvals["review"]["store_root"] == store  # approved in the store saved in
        assert approvals["review"]["saved_plan_store_root"] == store
        # the same dispatch: checked launch, frozen draft, worker command (tmp paths only)
        assert dispatches["review"] == dispatches["earlier"], name
        dispatch = dispatches["review"]
        assert dispatch["dispatch_check"] == [draft_id, plan_id, None]
        assert dispatch["frozen"][1:3] == [draft_id, plan_id]
        assert dispatch["plan_saved_before_launch"] == [[store, plan_id], [store, plan_id]]
        assert _normalised_command(dispatch["command"], env) == [
            "<python>", "<repo>/scripts/ifvg_funded_comparison_job.py", "start",
            "--plan-id", plan_id, "--store-root", "<tmp>/store",
            "--state-root", "<tmp>/comparison_jobs", "--reports-root", "<tmp>/reports"]
        for path_name, sequence in info["events"].items():
            assert sequence == ["save_plan", "record_owner_approval", "_freeze_and_launch",
                                "dispatch_problem", "save_plan", "mark_frozen", "_spawn"], (
                path_name, sequence)
        # one count, from the saved selections, on every screen: the plan's membership
        assert set(info["counts"].values()) == {len(configurations)}, info["counts"]
    for text in (t for info in details.values() for t in info["texts"]):
        assert "requested but has not reported progress yet" in text  # no worker was started
    # at least two DISTINCT plans were compared; the two first F11 drafts describe one plan
    # (the New funded comparison settings at today's behavior don't enter the plan)
    plans = {row["case"]: row["plan_id_review"] for row in executed if row["review_accept"]}
    assert plans["no New funded comparison settings"] == plans[
        "its settings at today's behavior"]
    assert len(set(plans.values())) == len(plans) - 1 >= 2, plans
    # nothing saved, approved or launched anywhere but the capturing fakes
    assert env["called"] == []
    assert _snapshot(env["tmp"]) == before  # no file added or changed under tmp_path
    for key in ("store_root", "funded_comparison_state_root", "state_root",
                "pipeline_state_root", "reports_root"):
        assert _files(roots[key]) == [], key
    assert {Path(f).parts[0] for f in _files(roots["draft_root"])} == {
        case["path"].parent.name for case in cases}  # only the drafts under test


def test_a11_the_capturing_fakes_respect_the_store_root(monkeypatch, tmp_path):
    """A11: the fakes behave like the real store: an approval needs the plan saved in the
    SAME store root, a lookup finds only that store's approvals, and nothing is written."""

    screen = SimpleNamespace(**dict.fromkeys((
        "_freeze_and_launch", "dispatch_problem", "_approve_and_run", "save_plan",
        "record_owner_approval", "find_approval", "mark_frozen", "_spawn", "time")))
    recorder = Recorder()
    recorder.install(monkeypatch, screen)
    plan_id = "a" * 64
    plan = SimpleNamespace(funded_comparison_plan_id=plan_id)
    one, other = tmp_path / "store_one", tmp_path / "store_two"
    approval = {"approved_on": "2026-09-25", "channel": "study_screen",
                "statement": "Approved on the study screen by the owner: a test plan.",
                "scope": "one test configuration"}
    with pytest.raises(AssertionError, match="no approval was being driven"):
        screen.save_plan(one, plan)
    recorder.start("probe")
    assert screen.save_plan(one, plan) == plan_id
    with pytest.raises(ValueError, match="approve a saved plan only"):
        screen.record_owner_approval(other, plan_id, **approval)  # saved in another store
    approval_id = screen.record_owner_approval(one, plan_id, **approval)
    assert screen.find_approval(one, plan_id).funded_comparison_approval_id == approval_id
    assert screen.find_approval(other, plan_id) is None
    assert screen.find_approval(str(one), plan_id) is not None  # a str or Path root alike
    recorder.stop()
    assert screen.find_approval(one, plan_id) is None  # outside a driven approval
    assert _files(tmp_path) == []


def test_a11_the_shared_count_follows_the_saved_membership(ac_env):
    """A11 (F4): the count is derived from the saved selections, never a fixed number.

    For drafts of 1, 2, 4 and 12 configurations, My studies
    (``ifvg_lab_library._draft_check`` + ``library.draft_summary``), New funded comparison
    (``ifvg_lab_new_funded._choices_count``) and Review and approve
    (``funded_setup.plan_count`` of ``_review_state``) each show the number of
    configurations in the plan Review and approve would approve (``_review_state``'s
    envelope) and in the plan the launch check rebuilds from the saved draft
    (``rebuild_saved_plan``, what ``dispatch_problem`` compares with).
    """

    import ifvg_lab_new_funded as nf

    from alpha_lab.propsim.funded.comparison_draft import (
        rebuild_saved_plan,
        saved_settings,
        saved_study_selections,
    )

    env = ac_env
    for case in COUNT_CASES:
        draft = _variation_draft(env["source"], case["inputs"]["variation_selections"])
        save_draft(env["roots"]["draft_root"], draft)
        state = nf._review_state(env["roots"], draft, env["sources"])
        rebuilt, skipped, _unavailable = rebuild_saved_plan(
            env["source"], saved_settings(draft), saved_study_selections(draft))
        expected = case["expected"]["configurations"]
        assert skipped == []
        assert len(state.envelope.payload.variants) == len(rebuilt.payload.variants) == expected
        assert state.envelope.funded_comparison_plan_id == rebuilt.funded_comparison_plan_id
        counts = _shown_counts(env, draft)
        assert set(counts.values()) == {expected}, (case["id"], counts)
    assert env["called"] == [] and _files(env["roots"]["store_root"]) == []


# ── the time-under-water limit: a neutral pending decision ───────────────


def test_a11_under_water_limit_is_a_neutral_pending_decision():
    """A11: the saved three-day limit stays as saved; the page proposes no value (no 18–20)."""

    defaults = fs.gate_defaults(PACKAGE_GATES)
    assert defaults["max_days_under_water"] == 3  # the saved requirement is unchanged
    for case in UNDER_WATER_CASES:
        inputs, expected = case["inputs"], case["expected"]
        assert defaults["max_days_under_water"] == inputs["source_study_limit_days"]
        rows = fs.gate_rows(inputs["saved_in_draft"], defaults, trading_days=DAYS,
                            typed=inputs["typed"])
        row = next(r for r in rows if r.spec.key == "max_days_under_water")
        assert (row.status, row.text, row.changed, row.message) == (
            expected["status"], expected["text"], expected["changed"], expected["message"]), case
        assert "18" not in row.message and "data points" not in row.message
        # a changed limit still blocks approval as a changed check; the value is never swapped
        assert ("gates_changed" in [b.key for b in fs.gate_blockers(rows)]) == row.changed
    assert not hasattr(fs, "_UNDER_WATER_HINT")


def test_a11_under_water_message_when_the_source_study_saved_no_limit():
    """A11: with no saved limit, a typed value says so plainly (never "the saved limit is .")."""

    defaults = {k: v for k, v in fs.gate_defaults(PACKAGE_GATES).items()
                if k != "max_days_under_water"}
    rows = fs.gate_rows({"max_days_under_water": 20}, defaults, trading_days=DAYS)
    row = next(r for r in rows if r.spec.key == "max_days_under_water")
    assert row.status == "decision"
    assert row.message == ("Needs your decision. You entered 20; the source study saved no "
                           "limit. This page doesn't propose a value.")


# ── excluded combinations: the half exit exists only at 1R ───────────────


def test_a11_review_lists_half_exits_only_at_1r_never_at_other_targets(ac_env):
    """A11: Review and approve never fabricates a half exit at a target other than 1R.

    The saved half-exit study selects 2 x 2 x 2 x 2 x 2 x 3 = 96 combinations; the 32 half
    exits at 2R or 3R are left out. Review and approve (``_review_state`` rows; on the
    pinned engine read-only, from the engine-independent count) lists 64 configurations,
    16 of them half exits, all at 1R, and every screen counts 64, never 96. On the
    half-exit engine the plan it would approve holds the same 64 (and no half exit away
    from 1R), and the earlier configurator states the 32 left-out combinations before its
    approval control.
    """

    import ifvg_lab_new_funded as nf

    env = ac_env
    draft = _variation_draft(env["source"])  # the saved 64-configuration study
    path = save_draft(env["roots"]["draft_root"], draft)
    before = _digest(path)
    state = nf._review_state(env["roots"], draft, env["sources"])
    strategy = {tuple(sorted((k, v) for k, v in row.items()
                             if k not in (fs.GAP_AXIS, fs.TRIGGER_AXIS))) for row in state.rows}
    assert len(state.rows) == len(strategy) == 64
    half = [row for row in state.rows if row["exit_policy"] == fs.HALF_EXIT]
    assert len(half) == 16 and {row["tp_r_multiple"] for row in half} == {fs.ONE_R}
    assert set(_shown_counts(env, draft).values()) == {64}
    review = _open("approve_funded", path.parent.name)
    text = _text(review)
    assert "Showing 8 of 64" in text and "96 configuration" not in text
    if HALF_EXIT_ENGINE:
        variants = state.envelope.payload.variants
        assert len(variants) == 64
        assert {dict(v.axis_value_ids).get("tp_r_multiple", fs.ONE_R) for v in variants
                if v.exit_policy != "fixed_target_v1"} == {fs.ONE_R}
        # a plan this engine can approve on Review and approve; the next test checks that
        # the page explains the 32 left-out combinations before that approval
        assert not review.checkbox(key=REVIEW_AGREE).disabled
        earlier = _open("new", path.parent.name)
        warning = " ".join(str(w.value) for w in earlier.warning)
        assert ("32 combination(s) cannot run and are not in the plan: the half exit is taken "
                "at 1R, so it applies only to the 1R target.") in warning
    else:  # read-only here: approval is off, so no plan can be approved from this page
        assert review.checkbox(key=REVIEW_AGREE).disabled
    assert _digest(path) == before and env["called"] == []


@pytest.mark.skipif(not HALF_EXIT_ENGINE, reason="needs the half-exit engine: on the pinned "
                    "engine the half-exit study is read-only and can't be approved")
def test_a11_review_explains_the_left_out_half_exits_before_approval(ac_env):
    """A11: before approving a new plan, Review and approve says which combinations are left
    out and why (the half exit is taken only at 1R), as the setup page does."""

    env = ac_env
    path = save_draft(env["roots"]["draft_root"], _variation_draft(env["source"]))
    review = _open("approve_funded", path.parent.name)
    text = _text(review)
    assert "32 combinations left out" in text and "the half exit is taken at 1R" in text
    assert env["called"] == []
