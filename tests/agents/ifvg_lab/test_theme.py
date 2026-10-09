"""The IFVG Lab follows the application's theme: one palette in two versions.

The light palette is the mocks' one; the dark palette answers the owner's report
that the screens ignored the dark theme (white text on white panels, an
invisible button to reopen the rail). Every screen color goes through the
palette, so no screen file may carry a color literal.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from alpha_lab.agents.data_infra.ifvg.presentation.lab import html as h  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme  # noqa: E402

HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")
LITERAL = re.compile(r"#[0-9A-Fa-f]{3,8}\b|\brgba?\(")


@pytest.fixture
def resolver():
    """Set the palette resolver for one test and always restore the default."""

    def _set(value):
        theme.set_theme_resolver(lambda: value)

    try:
        yield _set
    finally:
        theme.set_theme_resolver(None)


def test_both_palettes_name_the_same_colors_and_publish_them_as_variables():
    assert set(theme.DARK_COLORS) == set(theme.COLORS)
    for key in theme.COLORS:
        assert theme.css_var(key) == f"var(--lab-{key.replace('_', '-')})"
        for colors in (theme.COLORS, theme.DARK_COLORS):
            assert HEX.fullmatch(colors[key]) or colors[key].startswith("rgba("), key
    light, dark = theme.theme_css_vars("light"), theme.theme_css_vars("dark")
    for key, value in theme.COLORS.items():
        assert f"--lab-{key.replace('_', '-')}:{value};" in light
        assert f"--lab-{key.replace('_', '-')}:{theme.DARK_COLORS[key]};" in dark
    # the aliases the first screens used stay defined
    for alias in ("--lab-soft", "--lab-chart", "--lab-header", "--lab-control", "--lab-body2"):
        assert f"{alias}:var(--lab-" in light
    with pytest.raises(KeyError):
        theme.css_var("no_such_color")


def test_the_stylesheet_carries_both_themes_and_no_light_only_surface():
    css = theme.WORKSPACE_CSS
    assert ':root[data-lab-theme="dark"] {' in css
    assert "--lab-panel:#FFFFFF" in css and "--lab-panel:#1E2126" in css
    # after the variable blocks, the rules name colors only through variables
    # (the one literal left is the rail's chosen-item text, white on the dark rail
    # in both themes)
    rules = css.split("/* page")[1]
    assert set(HEX.findall(rules)) <= {"#FFFFFF"}
    assert rules.count("#FFFFFF") == 1
    # the framework's page background stays on .stApp (the theme probe reads it)
    assert ('[data-testid="stAppViewContainer"], [data-testid="stMain"] {\n'
            "  background: var(--lab-ground)") in css
    assert ".stApp { color: var(--lab-ink);" in css
    assert re.search(r"\.stApp\s*\{[^}]*background", css) is None
    # the header's controls are drawn readable in both themes
    assert '[data-testid="stExpandSidebarButton"]' in css
    assert '[data-testid="stMainMenu"] button' in css


def test_palette_and_chart_layout_follow_the_resolver(resolver):
    assert theme.active_theme() == "light"
    assert theme.palette() is theme.COLORS
    assert theme.chart_layout()["plot_bgcolor"] == theme.COLORS["chart_ground"]
    resolver("dark")
    assert theme.active_theme() == "dark"
    assert theme.palette() is theme.DARK_COLORS
    layout = theme.chart_layout()
    assert layout["plot_bgcolor"] == theme.DARK_COLORS["chart_ground"]
    assert layout["font"]["color"] == theme.DARK_COLORS["body"]
    assert layout["hoverlabel"]["bgcolor"] == theme.DARK_COLORS["panel"]
    assert theme.rgba("blue", 0.06) == "rgba(141,178,226,0.06)"
    resolver("light")
    assert theme.rgba("blue", 0.06) == "rgba(29,78,137,0.06)"
    resolver("something else")
    assert theme.active_theme() == "light"

    def broken():
        raise RuntimeError("no page")

    theme.set_theme_resolver(broken)
    assert theme.active_theme() == "light"


def test_html_builders_write_no_color_literal():
    pieces = [
        h.icon("check"), h.status_line("ok"), h.status_line("careful", kind="warn"),
        h.alert("first", "rest"), h.bars([("a", 0.2, "20%"), ("b", 0.7, "70%"), ("c", None, "")]),
        h.verdict_card("Edge", "Holds", "blue", "text"),
        h.finding_row("High", "orange", "title", "text", "next"),
    ]
    for piece in pieces:
        assert not LITERAL.search(str(piece)), piece
    assert "var(--lab-orange)" in str(h.bars([("b", 0.7, "70%")]))
    assert "var(--lab-blue)" in str(h.bars([("a", 0.2, "20%")]))


class _Box:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _fake_st(theme_type, session=None):
    html_out: list[str] = []
    return SimpleNamespace(
        html=html_out.append, container=lambda **_k: _Box(),
        session_state=session if session is not None else {},
        context=SimpleNamespace(theme=SimpleNamespace(type=theme_type)),
        _html=html_out)


def test_inject_theme_starts_on_the_reported_theme_and_keeps_the_probes_answer(monkeypatch):
    import ifvg_lab_ui

    # a light page: the stylesheet alone
    fake = _fake_st("light")
    monkeypatch.setattr(ifvg_lab_ui, "_theme_probe", lambda **_k: SimpleNamespace(theme=None))
    ifvg_lab_ui.inject_theme(fake)
    assert len(fake._html) == 1 and theme.WORKSPACE_CSS in fake._html[0]
    assert ":root:not([data-lab-theme])" not in fake._html[0]
    assert theme.THEME_KEY not in fake.session_state
    # a dark page: the dark variables apply from the first paint
    fake = _fake_st("dark")
    ifvg_lab_ui.inject_theme(fake)
    assert ":root:not([data-lab-theme]) { --lab-ground:#141619;" in fake._html[0]
    # the probe's answer wins and is kept for the following runs
    fake = _fake_st("light")
    monkeypatch.setattr(ifvg_lab_ui, "_theme_probe", lambda **_k: SimpleNamespace(theme="dark"))
    ifvg_lab_ui.inject_theme(fake)
    assert fake.session_state[theme.THEME_KEY] == "dark"
    fake = _fake_st("light", session={theme.THEME_KEY: "dark"})
    monkeypatch.setattr(ifvg_lab_ui, "_theme_probe", lambda **_k: SimpleNamespace(theme=None))
    ifvg_lab_ui.inject_theme(fake)
    assert ":root:not([data-lab-theme]) { --lab-ground:#141619;" in fake._html[0]
    # a stand-in without a probe or context still gets the stylesheet
    bare = SimpleNamespace(html=[].append)
    monkeypatch.setattr(ifvg_lab_ui, "_theme_probe",
                        lambda **_k: (_ for _ in ()).throw(TypeError("no registry")))
    ifvg_lab_ui.inject_theme(bare)


def test_the_probe_reads_the_framework_background_and_reruns_once_per_change():
    import ifvg_lab_ui

    js = ifvg_lab_ui._THEME_JS
    assert "document.querySelector('.stApp')" in js and "getComputedStyle" in js
    assert "setAttribute('data-lab-theme', current)" in js
    assert "setTriggerValue('theme', current)" in js
    assert "current !== known && current !== state.sent" in js
    assert "MutationObserver" in js and "prefers-color-scheme: dark" in js
    assert "ifvg_lab_theme_probe" in theme.WORKSPACE_CSS  # hidden, takes no room


def test_components_restore_a_new_runtime_registry_without_hiding_api_errors(monkeypatch):
    import ifvg_lab_ui as ui
    from streamlit.errors import StreamlitAPIException

    registrations = []
    mounts = []

    def register(name, **definition):
        registrations.append((name, definition))

        def mount(**kwargs):
            mounts.append(kwargs)
            return SimpleNamespace(action="next")

        return mount

    def absent(**kwargs):
        raise StreamlitAPIException("Component 'ifvg_lab_clickable' is not registered.")

    monkeypatch.setattr(ui.st.components.v2, "component", register)
    monkeypatch.setattr(ui, "_clickable", absent)
    assert ui.clickable("<button>Next</button>", key="header") == "next"
    assert registrations == [("ifvg_lab_clickable", {
        "js": ui._CLICK_JS, "isolate_styles": False,
    })]
    assert mounts[0]["key"] == f"{ui.PREFIX}click_header"
    assert mounts[0]["data"] == "<button>Next</button>"

    def invalid(**kwargs):
        raise StreamlitAPIException("invalid callback")

    monkeypatch.setattr(ui, "_clickable", invalid)
    with pytest.raises(StreamlitAPIException, match="invalid callback"):
        ui.clickable("<button>Next</button>", key="header")
    assert len(registrations) == 1


#: the semantic chart colors the Developer replay charts keep on both themes
_KEPT_CHART_LITERALS = {"#4C78A8", "#9467BD", "#E45756", "#2CA02C", "#8C8C8C", "#2E9990",
                        "#B8860B", "#D62728"}
#: the two session-band washes (``_BAND_COLORS``: engine blue, documented orange at 7%);
#: semantic like ``ZONE_COLORS``, so they keep one value on both themes
_KEPT_CHART_WASHES = ("rgba(76,120,168,0.07)", "rgba(230,159,0,0.07)")


@pytest.mark.parametrize("path", sorted(
    [p.relative_to(REPO).as_posix() for p in (REPO / "scripts").glob("ifvg_lab_*.py")]
    + [p.relative_to(REPO).as_posix() for p in
       (REPO / "src/alpha_lab/agents/data_infra/ifvg/presentation/lab").glob("*.py")
       if p.name != "theme.py"]))
def test_no_screen_file_hard_codes_a_color(path):
    text = (REPO / path).read_text(encoding="utf-8")
    found = set(HEX.findall(text))
    if path.endswith("ifvg_lab_charts.py"):
        found -= _KEPT_CHART_LITERALS
    assert not found, f"{path} still hard-codes {sorted(found)}"
    # only VALUE literals count: an rgba(...) whose first argument is a number. A palette
    # lookup such as theme.rgba("panel", 0.35) or the charts' _rgba(color, a) helper is a call.
    # Transparent/black values are theme-free; everything else must come from the palette.
    rgba = [m for m in re.findall(r"\brgba?\(\s*\d[^)]*\)", text)
            if not re.fullmatch(r"rgba?\(\s*0\s*,\s*0\s*,\s*0\s*(,\s*0(\.\d+)?)?\s*\)", m)]
    if path.endswith("ifvg_lab_charts.py"):
        rgba = [m for m in rgba if m not in _KEPT_CHART_WASHES]
    assert not rgba, f"{path} still hard-codes {rgba}"
