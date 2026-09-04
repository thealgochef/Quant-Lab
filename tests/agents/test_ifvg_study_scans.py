"""FUX-SAFE-001 / FUX-LABEL-001 source scans over every R4 UI surface, the
launch-only-in-handler proof, and the pure chart-builder units
(FUX-PERF-001 budgets/omissions; FUX §4.1 fallbacks)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

_UI_SCRIPTS = (
    "ifvg_ui_common.py",
    "ifvg_study_tab.py",
    "ifvg_study_wizard.py",
    "ifvg_active_runs_tab.py",
    "ifvg_results_tab.py",
    "ifvg_results_compare.py",
    "ifvg_results_charts.py",
    "ifvg_pipeline_tab.py",
    "ifvg_mbp1_panels.py",  # R5B panel (coverage gap closed by R6 review S5)
    "ifvg_regime_panels.py",  # R6 Regime Lane panel
)

_SRC_MODULES = (
    "study_status.py",
    "study_presentation.py",
    "study_drafts.py",
    "study_providers.py",
)


def _sources() -> dict[str, str]:
    sources = {
        name: (_REPO / "scripts" / name).read_text(encoding="utf-8")
        for name in _UI_SCRIPTS
    }
    base = _REPO / "src/alpha_lab/agents/data_infra/ifvg"
    for name in _SRC_MODULES:
        sources[name] = (base / name).read_text(encoding="utf-8")
    return sources


_BUTTON_LABEL = re.compile(r"button\(\s*[\"']([^\"']+)[\"']")


def test_no_forbidden_control_labels_anywhere() -> None:
    """FUX-SAFE-001: no delete/sealed/recapture/promote/unlock control, no
    allow_sealed, no order/live/serving control."""

    forbidden_words = ("delete", "sealed", "recapture", "promote", "unlock")
    for name, source in _sources().items():
        for label in _BUTTON_LABEL.findall(source):
            for word in forbidden_words:
                assert word not in label.lower(), (name, label)
        assert "allow_sealed" not in source, name
        for control in ("place order", "go live", "activate serving"):
            assert control not in source.lower(), (name, control)


def test_forbidden_display_wording_absent_from_ui_sources() -> None:
    """FUX-LABEL-001: Production Ready / Live Ready / robust representative /
    publishable / 'exact historical' never appear; assumed paths say
    scenario/approximation."""

    registry_block = re.compile(
        r"FORBIDDEN_DISPLAY_PHRASES:.*?\)\n", re.DOTALL
    )
    for name, source in _sources().items():
        # the scan registry itself and negated truthfulness statements
        # ("never exact historical …") are required content; the scan
        # targets affirmative display claims only
        source = registry_block.sub("", source)
        scannable = "\n".join(
            line
            for line in source.lower().splitlines()
            if "never exact historical" not in line
            and "not exact historical" not in line
        )
        for phrase in (
            "production ready",
            "live ready",
            "robust representative",
            '"publishable"',
            "exact historical",
        ):
            assert phrase not in scannable, (name, phrase)
    compare_source = _sources()["ifvg_results_compare.py"]
    assert "scenario/approximation" in compare_source
    assert "match_basis" in compare_source


def test_no_fuzzy_or_nearest_time_fallback_in_any_ui_source() -> None:
    """FUX-DRILL-001 source scan: no matching IMPLEMENTATION exists.

    Negated prose ("no fuzzy fallback") is required wording; the scan
    targets identifier-shaped code: calls, assignments, or attribute names
    built from fuzzy/nearest/keep-last matching."""

    code_token = re.compile(
        r"\b(?:fuzzy\w*|nearest_\w+|keep_last\w*)\s*[(=.]", re.IGNORECASE
    )
    for name, source in _sources().items():
        match = code_token.search(source)
        assert match is None, (name, match.group(0) if match else None)


def test_no_raw_override_editor_exists() -> None:
    """FUX §10.5: no free-form section_overrides control, no JSON/YAML
    editor. The only text_areas are the full-scope DATE lists (validated
    ISO dates — never overrides)."""

    sources = _sources()
    wizard = sources["ifvg_study_wizard.py"]
    # the wizard never touches section_overrides at all — values resolve
    # exclusively through the typed axis registry at enumeration time
    assert "section_overrides" not in wizard
    text_area_labels = re.findall(
        r'text_area\(\s*[\"\']([^\"\']+)', wizard
    )
    # UI-1: the frozen warmup prefix is read-only (derived from the backend
    # contract), so the ONE remaining text area is the evidence-date list,
    # validated field-by-field against the logical-day calendar
    assert len(text_area_labels) == 1
    assert all("dates" in label.lower() for label in text_area_labels)
    assert "logical trading days" in text_area_labels[0]
    for name, source in sources.items():
        assert "json.loads" not in source or name in (
            "study_drafts.py",
            "study_providers.py",
        ), name  # only the draft/provider stores parse THEIR OWN files
        if name.startswith("ifvg_"):
            assert "yaml" not in source.lower(), name


def test_process_launch_exists_only_in_the_designated_seams() -> None:
    """FUX-WIZ-012/FUX-PIPE-003 analogue: subprocess use is confined to the
    wizard's _spawn_search_job (+ read-only git rev-parse) and the job shim."""

    sources = _sources()
    for name, source in sources.items():
        if name == "ifvg_study_wizard.py":
            assert source.count("subprocess.Popen") == 1
            popen_at = source.index("subprocess.Popen")
            spawn_at = source.index("def _spawn_search_job")
            next_def = source.index("\ndef ", spawn_at + 1)
            assert spawn_at < popen_at < next_def  # inside the seam only
            assert source.count("subprocess.run") == 1  # git rev-parse only
            assert '"git", "rev-parse", "HEAD"' in source
        elif name == "ifvg_pipeline_tab.py":
            # FUX-PIPE-003: one detached seam, inside _spawn_pipeline_job only
            assert source.count("subprocess.Popen") == 1
            popen_at = source.index("subprocess.Popen")
            spawn_at = source.index("def _spawn_pipeline_job")
            next_def = source.index("\ndef ", spawn_at + 1)
            assert spawn_at < popen_at < next_def
            assert "subprocess.run" not in source
        else:
            assert "subprocess" not in source, name
    for shim_name in ("ifvg_search_job.py", "ifvg_pipeline_job.py"):
        shim = (_REPO / "scripts" / shim_name).read_text(encoding="utf-8")
        assert shim.count("subprocess.Popen") == 1, shim_name


def test_session_namespace_is_the_contracted_prefix() -> None:
    """FUX §3.2: EVERY session-state write — literal or f-string — lands
    under ifvg_study_v1_*; the ONLY legacy-prefix write is the registered
    verifier-jump interaction (safety review F5: f-string keys included)."""

    known_prefix_names = {
        "STATE_PREFIX",
        "PIPELINE_STATE_PREFIX",
        "_W",
        "_MON",
        "_RES",
        "_CMP",
        "_PIPE",
        "_PHASE_KEY",
        "_SELECTED_KEY",
        "ROUTE_KEY",
        "_PENDING_ROUTE_KEY",
        "_DRAFT_KEY",
        "_ACTIVE_DRAFT_KEY",
        "_SEARCH_KEY",
        "_RESULTS_SEARCH_KEY",
        "_SELECTED_CONFIG_KEY",
        "stage_key",
        "confirm_key",
        "size_key",
        "page_key",
        "gate_cache_key",  # per-pipeline gate cache (adversarial m-6)
        "key",
    }
    sources = _sources()
    assert 'STATE_PREFIX = "ifvg_study_v1_"' in sources["ifvg_ui_common.py"]
    assert (
        'PIPELINE_STATE_PREFIX = "ifvg_pipeline_v1_"'
        in sources["ifvg_ui_common.py"]
    )
    write_pattern = re.compile(r"session_state\[\s*(f?)([\"'])(.*?)\2\s*\]\s*=")
    var_pattern = re.compile(r"session_state\[\s*([A-Za-z_][\w.]*)\s*\]\s*=")
    for name, source in sources.items():
        for is_f, _quote, key in write_pattern.findall(source):
            if is_f:
                # every f-string key must interpolate a known workspace
                # prefix variable as its FIRST component
                first = re.match(r"\{([A-Za-z_][\w.]*)\}", key)
                assert first is not None, (name, key)
                variable = first.group(1).split(".")[-1]
                assert variable in known_prefix_names, (name, key)
            else:
                assert (
                    key.startswith("ifvg_study_v1_")
                    or key.startswith("ifvg_pipeline_v1_")
                    or key == "ifvg_context_v1_replay_pair"
                ), (name, key)
        for variable in var_pattern.findall(source):
            assert variable.split(".")[-1] in known_prefix_names, (
                name,
                variable,
            )
    assert "ifvg_context_v1_replay_pair" in sources["ifvg_ui_common.py"]


def test_no_standalone_best_winner_validated_titles() -> None:
    """FUX §5.3: Best / Winner / Validated / Production Ready / Live Ready
    never appear as display titles in any UI source (F17)."""

    title_pattern = re.compile(
        r'[\"\'](Best|Winner|Validated|Production Ready|Live Ready)\b'
    )
    for name, source in _sources().items():
        match = title_pattern.search(source)
        assert match is None, (name, match.group(0) if match else None)


# ── pure chart builders (FUX §§19-23, 28, 33) ──────────────────────────────


def test_frontier_builder_budgets_and_honest_omissions() -> None:
    from ifvg_results_charts import STUDY_LAYER_BUDGETS, build_frontier_figure

    rows = [
        {
            "config_id": f"{index:064d}"[:64],
            "name": f"cfg{index}",
            "expected_net_payout_90d": 100.0 * index,
            "payout_probability_per_rolling_30d": 0.5,
            "breach_probability_90d": 0.2,
            "median_account_lifetime_days": 40.0,
            "feasible": index % 7 != 0,
        }
        for index in range(300)
    ]
    figure, omissions = build_frontier_figure(rows, selected_id=rows[1]["config_id"])
    assert len(figure.data) == 1
    plotted = len(figure.data[0].x)
    assert plotted <= STUDY_LAYER_BUDGETS["frontier_points"]
    lines = omissions.summary_lines()
    assert any("budget" in line for line in lines)  # truncation reported
    assert any("infeasible" in line for line in lines)  # never silently plotted
    assert "★" in figure.data[0].text  # selection marker


def test_survival_builder_uses_step_lines_and_distinct_dashes() -> None:
    from ifvg_results_charts import build_survival_figure

    curves = [
        {"label": f"firm{index}", "days": [0, 30, 60], "survival": [1, 0.9, 0.8]}
        for index in range(3)
    ]
    figure, _omissions = build_survival_figure(curves)
    dashes = [trace.line.dash for trace in figure.data]
    assert len(set(dashes)) == 3  # dash pattern rides WITH color
    assert all(trace.line.shape == "hv" for trace in figure.data)


def test_timeline_builder_marker_shapes_and_display_timezone() -> None:
    from ifvg_results_charts import build_account_timeline_figure

    events = [
        {
            "event_type": "equity_update",
            "event_ts_utc": "2026-01-06T15:00:00+00:00",
            "event_ordinal": 0,
            "event_id": "e0",
            "payload": {"new_equity": 50_000.0},
        },
        {
            "event_type": "daily_halt",
            "event_ts_utc": "2026-01-06T16:00:00+00:00",
            "event_ordinal": 1,
            "event_id": "eh0",
            "payload": {"halt_reason": "dll", "threshold_value": 48_900.0},
        },
        {
            "event_type": "payout",
            "event_ts_utc": "2026-01-06T21:00:00+00:00",
            "event_ordinal": 2,
            "event_id": "e1",
            "source_trade_id": "t1",
            "payload": {"trader_amount": 300.0},
        },
        {
            "event_type": "daily_halt",
            "event_ts_utc": "2026-01-07T14:00:00+00:00",
            "event_ordinal": 3,
            "event_id": "eh1",
            "payload": {"halt_reason": "dll", "threshold_value": 49_100.0},
        },
        {
            "event_type": "phase_transition",
            "event_ts_utc": "2026-01-07T14:30:00+00:00",
            "event_ordinal": 4,
            "event_id": "ep0",
            "payload": {
                "from_phase": "evaluation",
                "to_phase": "funded",
                "reason": "target met",
            },
        },
        {
            "event_type": "breach",
            "event_ts_utc": "2026-01-07T15:00:00+00:00",
            "event_ordinal": 5,
            "event_id": "e2",
            "payload": {"breach_reason": "dll", "threshold_value": 1.0,
                        "observed_equity": 47_000.0},
        },
    ]
    figure, _omissions = build_account_timeline_figure(
        events,
        eligibility_windows=(
            ("2026-01-06T18:00:00+00:00", "2026-01-07T18:00:00+00:00"),
        ),
    )
    symbols = {trace.marker.symbol for trace in figure.data if trace.marker}
    assert "triangle-down-open" in symbols  # payout ▽ shape twin
    assert "x" in symbols  # breach ✕ shape twin
    texts = [t for trace in figure.data if trace.text for t in trace.text]
    assert "▽" in texts and "✕" in texts
    # FUX §28 (adversarial F15): the daily-loss threshold is a separate LINE
    dll = next(t for t in figure.data if t.name == "daily-loss threshold")
    assert "lines" in dll.mode
    # and phase transitions appear on the chart with their labels
    phases = next(t for t in figure.data if t.name == "phase transition")
    assert any("evaluation→funded" in t for t in phases.text)
    assert "America/New_York" in figure.layout.xaxis.title.text


def test_sanitizer_redacts_unc_and_quoted_space_paths() -> None:
    """FUX §34 (adversarial F6-safety / F20): UNC paths and quoted paths
    containing spaces are redacted."""

    from ifvg_ui_common import sanitize_error

    unc = sanitize_error(r"read failed: \\server\share\secret\file.parquet")
    assert "server" not in unc and "<path>" in unc
    spaced = sanitize_error(
        "boom at 'C:\\Program Files\\Quant Lab\\config.json' during load"
    )
    assert "Program" not in spaced and "<path>" in spaced
    drive = sanitize_error("open C:\\Users\\x\\y.txt failed token=abc")
    assert "Users" not in drive and "token=<redacted>" in drive


def test_heatmap_builder_draws_glyphs_not_color_alone() -> None:
    from ifvg_results_charts import build_sensitivity_heatmap

    cells = [
        {"row_value": "a", "col_value": "x", "value": 1.0,
         "cell_class": "stable_plateau", "sample_count": 30},
        {"row_value": "a", "col_value": "y", "value": None,
         "cell_class": "insufficient_data", "sample_count": 0},
        {"row_value": "b", "col_value": "x", "value": -1.0,
         "cell_class": "failed_region", "sample_count": 12},
        {"row_value": "b", "col_value": "y", "value": 0.2,
         "cell_class": "blocked_cell", "sample_count": 5},
    ]
    figure, _ = build_sensitivity_heatmap(
        cells, row_axis="axis A", col_axis="axis B", metric_label="net E[R]"
    )
    glyphs = {glyph for row in figure.data[0].text for glyph in row}
    assert {"◼", "·", "✕", "⊘"} <= glyphs


def test_payout_distribution_marks_p10_most_prominently() -> None:
    from ifvg_results_charts import build_payout_distribution_figure

    figure, _ = build_payout_distribution_figure(
        [0.0, 100.0, 500.0, 900.0],
        horizon_label="90-day",
        quantiles={"mean": 375.0, "median": 300.0, "p10": 30.0, "p90": 860.0},
    )
    shapes = figure.layout.shapes
    p10 = next(
        shape
        for shape, annotation in zip(
            shapes, figure.layout.annotations, strict=False
        )
        if "P10" in annotation.text
    )
    widths = [shape.line.width for shape in shapes]
    assert p10.line.width == max(widths)  # the lower tail leads visually


def test_capability_fallbacks_preserve_semantics(monkeypatch, tmp_path) -> None:
    """FUX-A11Y-004: without st.fragment the SAME plain body renders."""

    apptest = pytest.importorskip("streamlit.testing.v1")
    import ifvg_study_tab as study_tab

    from tests.agents.ifvg_search.study_ui_fixture import build_completed_search

    fixture = build_completed_search(
        tmp_path, with_prop=False, with_contract=False
    )
    monkeypatch.setattr(study_tab, "STORE_ROOT_RESEARCH", fixture["store_root"])
    monkeypatch.setattr(study_tab, "STATE_ROOT", fixture["state_root"])
    monkeypatch.setattr(study_tab, "DRAFT_ROOT", fixture["draft_root"])

    def _app_no_fragment() -> None:
        # AppTest re-executes this source fresh: import everything here.
        import ifvg_active_runs_tab as monitor  # noqa: F811
        import ifvg_study_tab as study_tab
        import streamlit as st

        class _NoFragment:
            def __getattr__(self, name):
                if name == "fragment":
                    raise AttributeError(name)
                return getattr(st, name)

        monitor.render_active_runs(
            _NoFragment(), roots=study_tab.workspace_roots(st)
        )

    at = apptest.AppTest.from_function(_app_no_fragment, default_timeout=120)
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert "Refresh" in labels  # the manual fallback carries the semantics
    assert "Generated · 4" in labels  # the same plain body rendered
