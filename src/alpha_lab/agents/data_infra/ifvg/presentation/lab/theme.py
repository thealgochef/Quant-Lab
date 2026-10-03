"""Design tokens for the IFVG Lab redesign (``DESIGN_SYSTEM.md``), as CSS and a chart style.

The CSS is injected only by the IFVG workspace, so other pages of the main app
keep their own look. Values come from the mock source files; when a value here
and a mock disagree, the mock source wins. Blue and orange carry meaning
together with a word or a position, never by color alone.

Two palettes, one set of names
------------------------------
The screens follow the application's own theme (Settings → Choose app theme,
or the system setting). ``COLORS`` is the light palette from the mocks and
``DARK_COLORS`` its dark counterpart. Every color is published as a CSS
variable (``--lab-<name>``) defined for both themes, and :func:`palette` gives
the active palette to Python code that must hold a real color value (Plotly
figures). Screen code must not write a color literal: CSS and inline styles use
:func:`css_var`, chart code reads :func:`palette` when it builds the figure.

The active theme is decided outside this module (``ifvg_lab_ui.inject_theme``
reads it from the page); :func:`set_theme_resolver` connects the two. Without a
resolver the palette is light, which is what the headless tests see.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = [
    "RAIL_LINK_CSS",
    "COLORS",
    "DARK_COLORS",
    "FONT_MONO",
    "FONT_SANS",
    "FONT_SERIF",
    "THEME_KEY",
    "WORKSPACE_CSS",
    "active_theme",
    "chart_layout",
    "css_var",
    "palette",
    "rgba",
    "set_theme_resolver",
    "style_chart",
    "theme_css_vars",
]

#: session key holding the theme the page reported ("light" or "dark")
THEME_KEY = "ifvg_lab_v1_theme"

COLORS: dict[str, str] = {
    "ground": "#F4F2EC",
    "panel": "#FFFFFF",
    "soft_panel": "#F7F5F0",
    "chart_ground": "#FBFAF7",
    "header_row": "#EFECE4",
    "rule": "#DAD6CC",
    "light_rule": "#E8E4DA",
    "grid": "#F0EDE6",
    "control_border": "#BDB8AC",
    "ink": "#16181B",
    "body": "#2A2D31",
    "body_2": "#3C4046",
    "muted": "#54585E",
    "blue": "#1D4E89",
    "blue_dark": "#143A67",
    "blue_light": "#E4ECF6",
    "blue_band": "#DCE6F2",
    "blue_shade": "#B7CBE3",
    "blue_mid": "#A9C0DC",
    "blue_line": "#6F8FB5",
    "orange": "#A34A12",
    "orange_dark": "#6E300B",
    "orange_darker": "#4F2308",
    "orange_light": "#F8E9DD",
    "orange_shade": "#F6E4D6",
    "orange_mid": "#EBC5A8",
    "rail": "#16181B",
    "rail_active": "#2B2E33",
    "rail_text": "#E9E6DE",
    "rail_note": "#B3AFA6",
    "leader_row": "#F3F6FA",
    "risk_row": "#FBF4EE",
    "dashed": "#A9A498",
    "sample_path": "#B8B4AB",
    # surfaces and states the mocks imply but do not name
    "on_ink": "#FFFFFF",  # text on an ink-colored fill (primary buttons, chosen switch)
    "hover": "#F7F5F0",  # a clickable row under the pointer
    "track": "#EEEBE4",  # empty part of a share bar
    "disabled_bg": "#E5E2DA",
    "disabled_text": "#5F6268",
    "disabled_border": "#D5D1C7",
    "shade": "rgba(22, 24, 27, 0.06)",  # a light wash over a surface
    "zero_line": "#8A857A",  # a chart's zero or midnight line
    "sample_line": "#7C8797",  # one resampled path among many
}

#: the same names for the dark theme: dark warm surfaces, light ink, the blue and
#: orange lightened so they read on dark panels, tints turned into dark washes
DARK_COLORS: dict[str, str] = {
    "ground": "#141619",
    "panel": "#1E2126",
    "soft_panel": "#23272D",
    "chart_ground": "#1A1D22",
    "header_row": "#272B31",
    "rule": "#363B42",
    "light_rule": "#2C3137",
    "grid": "#262A30",
    "control_border": "#50565E",
    "ink": "#EDEBE5",
    "body": "#D8D5CE",
    "body_2": "#C3C0B8",
    "muted": "#A19E96",
    "blue": "#8DB2E2",
    "blue_dark": "#BBD2EF",
    "blue_light": "#1F2E44",
    "blue_band": "#253652",
    "blue_shade": "#33507A",
    "blue_mid": "#4E729F",
    "blue_line": "#6F8FB5",
    "orange": "#E4894C",
    "orange_dark": "#F1BA8F",
    "orange_darker": "#F7D0AF",
    "orange_light": "#3A2517",
    "orange_shade": "#43291A",
    "orange_mid": "#7A4A2A",
    "rail": "#0B0C0E",
    "rail_active": "#23262B",
    "rail_text": "#E9E6DE",
    "rail_note": "#9F9C93",
    "leader_row": "#1F2733",
    "risk_row": "#2B2219",
    "dashed": "#5D626A",
    "sample_path": "#5C6068",
    "on_ink": "#16181B",
    "hover": "#262A30",
    "track": "#2B2F35",
    "disabled_bg": "#2A2E34",
    "disabled_text": "#9CA0A7",
    "disabled_border": "#3A3F46",
    "shade": "rgba(237, 235, 229, 0.08)",
    "zero_line": "#80848B",
    "sample_line": "#7F8A9A",
}

assert set(DARK_COLORS) == set(COLORS), "both palettes name the same colors"

FONT_SERIF = "'Newsreader', Georgia, serif"
FONT_SANS = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, 'Cascadia Mono', Consolas, monospace"

_FONTS_URL = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&"
    "family=IBM+Plex+Sans:wght@400;500;600&"
    "family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap"
)

#: shorter names the first screens used; kept as aliases of the systematic ones
_ALIASES = {
    "soft": "soft_panel", "chart": "chart_ground", "header": "header_row",
    "control": "control_border", "body2": "body_2", "line": "light_rule",
    "sample": "sample_path",
}


def _var_name(key: str) -> str:
    return f"--lab-{key.replace('_', '-')}"


def css_var(key: str) -> str:
    """``var(--lab-…)`` for a palette color, for CSS text and inline styles."""

    if key not in COLORS:
        raise KeyError(f"unknown lab color: {key}")
    return f"var({_var_name(key)})"


def theme_css_vars(theme: str) -> str:
    """The ``--lab-*`` declarations of one theme (``light`` or ``dark``)."""

    colors = DARK_COLORS if theme == "dark" else COLORS
    parts = [f"{_var_name(key)}:{value};" for key, value in colors.items()]
    parts += [f"{_var_name(alias)}:var({_var_name(key)});" for alias, key in _ALIASES.items()]
    return "".join(parts)


# ── the active theme ──────────────────────────────────────────────────────

_resolver: Callable[[], str | None] | None = None


def set_theme_resolver(resolver: Callable[[], str | None] | None) -> None:
    """Install the function that names the active theme (``ifvg_lab_ui`` does)."""

    global _resolver
    _resolver = resolver


def active_theme() -> str:
    """``"dark"`` when the page runs on the dark theme, otherwise ``"light"``."""

    if _resolver is None:
        return "light"
    try:
        value = _resolver()
    except Exception:  # no page (a headless test, an import-time call)
        return "light"
    return "dark" if value == "dark" else "light"


def palette(theme: str | None = None) -> dict[str, str]:
    """The palette of the active theme (or of the named one), for chart code."""

    theme = theme or active_theme()
    return DARK_COLORS if theme == "dark" else COLORS


def rgba(key: str, alpha: float, theme: str | None = None) -> str:
    """A palette color with transparency, as ``rgba(r, g, b, a)`` for chart code."""

    value = palette(theme)[key]
    if value.startswith("rgba("):
        value = "#" + "".join(f"{int(part):02X}" for part in value[5:-1].split(",")[:3])
    r, g, b = (int(value[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha:g})"


# ── the stylesheet ────────────────────────────────────────────────────────

_v = css_var

#: the rail's "Other workspaces" links, one way on every page: shipped with the rail
#: itself (``ifvg_lab_ui.rail``) and in the page stylesheet, full names, regular weight
RAIL_LINK_CSS = (
    'section[data-testid="stSidebar"] [data-testid="stPageLink"] a { padding: 6px 14px; }'
    'section[data-testid="stSidebar"] [data-testid="stPageLink"] p, '
    'section[data-testid="stSidebar"] [data-testid="stPageLink"] span { '
    "white-space: normal !important; overflow: visible !important; "
    "text-overflow: clip !important; max-width: none !important; font-size: 14px !important; "
    "font-weight: 400 !important; line-height: 1.3 !important; }"
)

WORKSPACE_CSS = f"""
@import url('{_FONTS_URL}');
:root {{
  {theme_css_vars("light")}
  --lab-serif:{FONT_SERIF}; --lab-sans:{FONT_SANS}; --lab-mono:{FONT_MONO};
}}
:root[data-lab-theme="dark"] {{ {theme_css_vars("dark")} }}
:root[data-lab-theme="dark"] {{ color-scheme: dark; }}
:root[data-lab-theme="light"] {{ color-scheme: light; }}
/* page: the ground sits on the view container, so the framework's own background
   stays readable on .stApp (that is how the page tells light from dark) */
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{
  background: {_v('ground')} !important;
}}
.stApp {{ color: {_v('ink')}; font-family: var(--lab-sans); }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stAppDeployButton"] {{ display: none !important; }}
/* the header's own controls read in both themes: the button that reopens a
   collapsed rail is drawn as a rail-colored pill, the menu in ink */
[data-testid="stExpandSidebarButton"] {{
  background: {_v('rail')} !important; border-radius: 8px !important; padding: 2px !important;
  margin-left: 4px; box-shadow: 0 0 0 1px {_v('rule')};
}}
[data-testid="stExpandSidebarButton"] span, [data-testid="stExpandSidebarButton"] svg {{
  color: {_v('rail_text')} !important; fill: {_v('rail_text')} !important;
}}
[data-testid="stExpandSidebarButton"]:hover {{ background: {_v('rail_active')} !important; }}
[data-testid="stMainMenu"] button, [data-testid="stMainMenu"] button span,
[data-testid="stMainMenu"] button svg,
[data-testid="stToolbarActions"] button, [data-testid="stToolbarActions"] button span {{
  color: {_v('ink')} !important; fill: {_v('ink')} !important;
}}
[data-testid="stMainBlockContainer"] {{
  max-width: 1440px; padding: 40px 48px 64px 48px;
}}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp select,
.stApp button {{ font-family: var(--lab-sans); }}
.stApp code {{ font-family: var(--lab-mono); }}
.stApp h1, .stApp h2, .stApp h3 {{
  font-family: var(--lab-serif); font-weight: 600; color: {_v('ink')};
  letter-spacing: -0.2px;
}}
/* framework text follows the palette (the framework's own color would be the
   other theme's when the page is drawn on the opposite palette) */
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stMain"] [data-testid="stCaptionContainer"],
.stApp [data-testid="stMain"] [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stMain"] [data-testid="stHeadingWithActionElements"],
.stApp [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stMain"] [data-testid="stCheckbox"] p,
.stApp [data-testid="stMain"] [data-testid="stRadio"] p,
.stApp [data-testid="stMain"] [data-testid="stExpander"] summary p,
.stApp [data-testid="stMain"] [data-testid="stExpander"] summary span,
.stApp [data-testid="stMain"] [data-testid="stMetricLabel"],
.stApp [data-testid="stMain"] [data-testid="stMetricValue"] {{ color: {_v('ink')}; }}
.stApp [data-testid="stMain"] [data-testid="stCaptionContainer"],
.stApp [data-testid="stMain"] [data-testid="stCaptionContainer"] p {{ color: {_v('muted')}; }}
/* left rail */
section[data-testid="stSidebar"] {{
  background: {_v('rail')} !important; min-width: 232px !important;
  max-width: 232px !important;
}}
section[data-testid="stSidebar"] > div {{ background: {_v('rail')}; }}
section[data-testid="stSidebar"] * {{ color: {_v('rail_text')}; }}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
  padding: 24px 16px 24px 16px; display: flex; flex-direction: column;
  min-height: calc(100vh - 80px);
}}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{ display: none; }}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {{
  background: {_v('rail_active')} !important; border-radius: 8px !important;
}}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] span,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg {{
  color: {_v('rail_text')} !important; fill: {_v('rail_text')} !important;
}}
.st-key-ifvg_lab_rail {{ min-height: calc(100vh - 110px); }}
{RAIL_LINK_CSS}
.st-key-ifvg_lab_rail > div:last-child {{ margin-top: auto; }}
.st-key-ifvg_lab_rail button {{
  width: 100%; justify-content: flex-start; text-align: left; border: none !important;
  background: transparent !important; color: {_v('rail_text')} !important;
  padding: 12px 14px !important; min-height: 44px; border-radius: 8px !important;
  font-size: 15px !important; box-shadow: none !important;
}}
.st-key-ifvg_lab_rail button p {{ font-size: 15px !important; }}
.st-key-ifvg_lab_rail button[kind="primary"],
.st-key-ifvg_lab_rail button[data-testid="stBaseButton-primary"] {{
  background: {_v('rail_active')} !important; color: #FFFFFF !important; font-weight: 500;
}}
.st-key-ifvg_lab_rail button:hover {{ background: {_v('rail_active')} !important; }}
.st-key-ifvg_lab_rail button:focus-visible {{ outline: 2px solid {_v('rail_text')}; }}
.lab-rail-title {{
  font-family: var(--lab-serif); font-size: 28px; font-weight: 600;
  color: {_v('rail_text')} !important; margin: 4px 0 20px 4px;
}}
.lab-rail-note {{ font-size: 13px; line-height: 1.5; color: {_v('rail_note')} !important;
  margin: 24px 0 0 4px; }}
/* the theme probe (ifvg_lab_ui.inject_theme) takes no room */
.st-key-ifvg_lab_theme_probe {{ display: none; }}
/* buttons */
.stApp [data-testid="stMain"] button[data-testid="stBaseButton-primary"] {{
  background: {_v('ink')}; border: 1px solid {_v('ink')}; color: {_v('on_ink')};
  border-radius: 8px; min-height: 44px; font-weight: 500;
}}
.stApp [data-testid="stMain"] button[data-testid="stBaseButton-primary"] p {{
  color: {_v('on_ink')}; }}
.stApp [data-testid="stMain"] button[data-testid="stBaseButton-secondary"] {{
  background: {_v('panel')}; border: 1px solid {_v('control_border')}; color: {_v('ink')};
  border-radius: 8px; min-height: 40px; font-weight: 500;
}}
.stApp [data-testid="stMain"] button[data-testid="stBaseButton-secondary"] p {{
  color: {_v('ink')}; }}
.stApp [data-testid="stMain"] button[data-testid="stBaseButton-tertiary"] {{
  color: {_v('blue')}; text-decoration: underline; padding: 0; min-height: 0;
}}
.stApp [data-testid="stMain"] button[data-testid="stBaseButton-tertiary"] p {{
  color: {_v('blue')}; }}
.stApp [data-testid="stMain"] button:disabled,
.stApp [data-testid="stMain"] button:disabled p {{
  background: {_v('disabled_bg')} !important; color: {_v('disabled_text')} !important;
  border-color: {_v('disabled_border')} !important;
}}
.stApp [data-testid="stMain"] button:disabled p {{ background: transparent !important; }}
/* segmented switches: joined buttons, selected is ink with white text */
.stApp [data-testid="stButtonGroup"] button {{
  min-height: 40px; font-weight: 500; white-space: nowrap;
  background: {_v('panel')}; color: {_v('ink')}; border-color: {_v('control_border')};
}}
.stApp [data-testid="stButtonGroup"] button p {{ color: {_v('ink')}; }}
.stApp [data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
.stApp [data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] {{
  background: {_v('ink')} !important; color: {_v('on_ink')} !important;
  border-color: {_v('ink')} !important;
}}
.stApp [data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] p {{
  color: {_v('on_ink')} !important;
}}
/* the design blue in place of the framework's default red, on every page with the rail
   (earlier wizards, study pages and result pages included) */
.stApp [data-testid="stProgress"] [role="progressbar"] > div > div {{
  background-color: {_v('light_rule')} !important; }}
.stApp [data-testid="stProgress"] [role="progressbar"] > div > div > div {{
  background-color: {_v('blue')} !important; }}
.stApp label[data-baseweb="radio"]:has(input:checked) > div:first-child {{
  background-color: {_v('blue')} !important; border-color: {_v('blue')} !important; }}
.stApp label[data-baseweb="checkbox"]:has(input:checked) > span {{
  background-color: {_v('blue')} !important; border-color: {_v('blue')} !important; }}
.stApp label[data-baseweb="checkbox"]:has(input[role="switch"]:checked) > div {{
  background-color: {_v('blue_light')} !important; }}
.stApp label[data-baseweb="checkbox"]:has(input[role="switch"]:checked) > div > div {{
  background-color: {_v('blue')} !important; }}
.stApp [data-testid="stSlider"] [role="slider"] {{
  background-color: {_v('blue')} !important; border-color: {_v('blue')} !important; }}
.stApp [data-testid="stSliderThumbValue"], .stApp [data-testid="stSliderTickBarMin"],
.stApp [data-testid="stSliderTickBarMax"] {{ color: {_v('blue_dark')} !important; }}
.stApp [data-baseweb="tag"] {{
  background-color: {_v('blue_light')} !important; color: {_v('blue_dark')} !important; }}
.stApp [data-baseweb="tag"] span, .stApp [data-baseweb="tag"] svg {{
  color: {_v('blue_dark')} !important; fill: {_v('blue_dark')} !important; }}
.stApp button[data-testid="stBaseButton-pillsActive"] {{
  background: {_v('blue_light')} !important; border-color: {_v('blue')} !important; }}
.stApp button[data-testid="stBaseButton-pillsActive"] p {{
  color: {_v('blue_dark')} !important; }}
.stApp button[data-testid="stBaseButton-pills"] {{
  background: {_v('panel')}; border-color: {_v('control_border')}; }}
.stApp button[data-testid="stBaseButton-pills"] p {{ color: {_v('ink')}; }}
.stApp [data-baseweb="tab-highlight"] {{ background-color: {_v('blue')} !important; }}
.stApp button[role="tab"][aria-selected="true"] p {{ color: {_v('blue_dark')} !important; }}
.stApp [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover,
.stApp [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:focus-visible {{
  border-color: {_v('blue')} !important; color: {_v('blue_dark')} !important; }}
.stApp [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover p {{
  color: {_v('blue_dark')} !important; }}
.stApp [data-baseweb="input"]:focus-within, .stApp [data-baseweb="textarea"]:focus-within,
.stApp [data-baseweb="select"] > div:focus-within,
.stApp [data-baseweb="base-input"]:focus-within {{
  border-color: {_v('blue')} !important; }}
.stApp [data-baseweb="calendar"] [aria-selected="true"] div,
.stApp [data-baseweb="calendar"] [aria-selected="true"]::after {{
  background-color: {_v('blue')} !important; border-color: {_v('blue')} !important; }}
/* switches: a horizontal radio drawn as joined buttons; selected is ink with white text */
[class*="st-key-ifvg_lab_switch"] [role="radiogroup"] {{ display: flex; flex-direction: row;
  flex-wrap: nowrap; gap: 0; }}
[class*="st-key-ifvg_lab_switch_firm"] [role="radiogroup"] {{ justify-content: flex-end; }}
[class*="st-key-ifvg_lab_switch"] [role="radiogroup"] > label {{ margin: 0 0 0 -1px;
  padding: 8px 14px; min-height: 40px; border: 1px solid {_v('control_border')};
  background: {_v('panel')};
  border-radius: 0; cursor: pointer; align-items: center; white-space: nowrap; }}
[class*="st-key-ifvg_lab_switch"] [role="radiogroup"] > label:first-child {{
  border-radius: 8px 0 0 8px; margin-left: 0; }}
[class*="st-key-ifvg_lab_switch"] [role="radiogroup"] > label:last-child {{
  border-radius: 0 8px 8px 0; }}
[class*="st-key-ifvg_lab_switch"] [role="radiogroup"] > label > div:first-of-type {{
  display: none; }}
[class*="st-key-ifvg_lab_switch"] [role="radiogroup"] > label p {{ font-size: 14px;
  font-weight: 500; color: {_v('ink')}; margin: 0; }}
[class*="st-key-ifvg_lab_switch"] [role="radiogroup"] > label:has(input:checked) {{
  background: {_v('ink')}; border-color: {_v('ink')}; }}
[class*="st-key-ifvg_lab_switch"] [role="radiogroup"] > label:has(input:checked) p {{
  color: {_v('on_ink')} !important; }}
[class*="st-key-ifvg_lab_switch"] [role="radiogroup"] > label:has(input:focus-visible) {{
  outline: 2px solid {_v('blue')}; outline-offset: 1px; }}
/* detail tabs: the same radio drawn as text tabs with a 3 px ink underline */
[class*="st-key-ifvg_lab_tabs"] [role="radiogroup"] {{ gap: 2px; width: 100%;
  border-bottom: 1px solid {_v('rule')}; }}
[class*="st-key-ifvg_lab_tabs"] [role="radiogroup"] > label,
[class*="st-key-ifvg_lab_tabs"] [role="radiogroup"] > label:first-child,
[class*="st-key-ifvg_lab_tabs"] [role="radiogroup"] > label:last-child {{
  border: none; border-bottom: 3px solid transparent; border-radius: 0; background: transparent;
  margin: 0; padding: 10px 12px; max-width: 150px; white-space: normal; }}
[class*="st-key-ifvg_lab_tabs"] [role="radiogroup"] > label p {{ font-size: 15px;
  font-weight: 400; color: {_v('body_2')}; line-height: 1.25; white-space: normal; }}
[class*="st-key-ifvg_lab_tabs"] [role="radiogroup"] > label:has(input:checked) {{
  background: transparent; border-bottom: 3px solid {_v('ink')}; }}
[class*="st-key-ifvg_lab_tabs"] [role="radiogroup"] > label:has(input:checked) p {{
  color: {_v('ink')} !important; font-weight: 600; }}
/* inputs */
.stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"] > div,
.stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
.stApp [data-baseweb="textarea"], .stApp [data-baseweb="textarea"] > div {{
  background: {_v('panel')}; border-color: {_v('control_border')}; border-radius: 8px;
}}
.stApp [data-baseweb="select"] > div *, .stApp [data-baseweb="input"] input,
.stApp [data-baseweb="base-input"] input, .stApp [data-baseweb="textarea"] textarea,
.stApp [data-testid="stNumberInput"] input, .stApp [data-testid="stDateInput"] input {{
  color: {_v('ink')}; -webkit-text-fill-color: {_v('ink')}; }}
.stApp [data-baseweb="input"] input::placeholder, .stApp [data-baseweb="textarea"]
  textarea::placeholder {{ color: {_v('muted')}; -webkit-text-fill-color: {_v('muted')};
  opacity: 1; }}
.stApp [data-baseweb="select"] svg {{ fill: {_v('muted')}; color: {_v('muted')}; }}
.stApp [data-testid="stNumberInput"] button {{ background: {_v('soft_panel')};
  color: {_v('ink')}; border-color: {_v('control_border')}; }}
/* the dropdown lists and calendars open in the theme's own popover; they inherit
   the palette so a chosen value never reads white on white */
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"], [data-baseweb="calendar"] {{
  background: {_v('panel')}; color: {_v('ink')}; }}
[data-baseweb="popover"] [role="option"], [data-baseweb="menu"] li {{ color: {_v('ink')}; }}
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] [role="option"]:hover, [data-baseweb="menu"] li:hover {{
  background: {_v('soft_panel')}; }}
.stApp [data-testid="stWidgetLabel"] p {{ font-size: 13px; color: {_v('muted')}; }}
/* readable selections: a chosen value is shown in full, never cut off */
.stApp [data-baseweb="tag"] {{ max-width: none !important; height: auto !important;
  background: {_v('blue_light')} !important; border: 1px solid {_v('blue')} !important;
  border-radius: 8px !important; padding-top: 4px; padding-bottom: 4px; }}
.stApp [data-baseweb="tag"] span {{ white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important; max-width: none !important; color: {_v('blue_dark')}; }}
.stApp [data-baseweb="select"] [data-baseweb="tag"] svg {{ fill: {_v('blue_dark')}; }}
/* expanders and bordered containers as cards */
.stApp [data-testid="stExpander"] details {{
  background: {_v('panel')}; border: 1px solid {_v('rule')}; border-radius: 10px;
}}
.stApp [data-testid="stExpander"] summary:hover {{ color: {_v('blue')}; }}
.stApp [data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {_v('rule')}; }}
[class*="st-key-ifvg_lab_card"] {{
  background: {_v('panel')}; border: 1px solid {_v('rule')}; border-radius: 12px;
  padding: 22px 24px;
}}
/* framework tables, code and notices on the palette's surfaces */
.stApp [data-testid="stMain"] [data-testid="stTable"] table,
.stApp [data-testid="stMain"] [data-testid="stTable"] th,
.stApp [data-testid="stMain"] [data-testid="stTable"] td {{
  color: {_v('ink')}; border-color: {_v('light_rule')}; }}
.stApp [data-testid="stMain"] [data-testid="stTable"] th {{ background: {_v('header_row')}; }}
.stApp [data-testid="stMain"] [data-testid="stCode"] pre,
.stApp [data-testid="stMain"] [data-testid="stMarkdownContainer"] code {{
  background: {_v('soft_panel')}; color: {_v('ink')}; }}
.stApp [data-testid="stMain"] [data-testid="stAlertContainer"] p,
.stApp [data-testid="stMain"] [data-testid="stAlertContainer"] li {{ color: inherit; }}
.stApp [data-testid="stMain"] [data-testid="stTooltipIcon"] svg {{ color: {_v('muted')}; }}
/* plotly charts sit on white cards */
.stApp [data-testid="stPlotlyChart"] {{ background: transparent; }}
/* HTML building blocks (lab/html.py) */
.lab {{ font-family: var(--lab-sans); color: {_v('ink')}; }}
.lab-mono {{ font-family: var(--lab-mono); }}
.lab-serif {{ font-family: var(--lab-serif); font-weight: 600; }}
.lab-muted {{ color: {_v('muted')}; }}
.lab a {{ color: {_v('blue')}; }}
.lab-crumb {{ font-size: 14px; color: {_v('muted')}; }}
.lab-h1 {{ margin: 0; font-family: var(--lab-serif); font-size: 40px; font-weight: 600;
  letter-spacing: -0.3px; line-height: 1.15; color: {_v('ink')}; }}
.lab-h1.detail {{ font-size: 34px; }}
.lab-h2 {{ margin: 0; font-family: var(--lab-serif); font-size: 26px; font-weight: 600;
  color: {_v('ink')}; line-height: 1.2; }}
.lab-h3 {{ margin: 0; font-family: var(--lab-serif); font-size: 22px; font-weight: 600;
  color: {_v('ink')}; line-height: 1.25; }}
.lab-sub {{ font-size: 16px; color: {_v('body_2')}; }}
.lab-line {{ font-size: 14px; color: {_v('muted')}; line-height: 1.45; }}
.lab-card {{ background: {_v('panel')}; border: 1px solid {_v('rule')}; border-radius: 12px;
  padding: 22px 24px; display: flex; flex-direction: column; gap: 12px; box-sizing: border-box;
  min-width: 0; }}
.lab-card.soft {{ background: {_v('chart_ground')}; border: 1px dashed {_v('dashed')}; }}
.lab-card-title {{ margin: 0; font-family: var(--lab-serif); font-size: 22px; font-weight: 600;
  color: {_v('ink')}; }}
.lab-card-title.sans {{ font-family: var(--lab-sans); font-size: 16px; }}
.lab-grid {{ display: grid; gap: 12px; }}
.lab-grid.g16 {{ gap: 16px; }}
.lab-tile {{ background: {_v('panel')}; border: 1px solid {_v('rule')}; border-radius: 12px;
  padding: 16px 18px; display: flex; flex-direction: column; gap: 4px; min-width: 0; }}
.lab-tile-label {{ font-size: 13px; color: {_v('muted')}; }}
.lab-tile-value {{ font-family: var(--lab-mono); font-size: 22px; color: {_v('ink')};
  overflow-wrap: anywhere; }}
.lab-tile-value.big {{ font-size: 26px; font-weight: 500; }}
.lab-tile-caption {{ font-size: 13px; color: {_v('muted')}; line-height: 1.45; }}
.lab-overline {{ font-size: 13px; font-weight: 600; letter-spacing: 0.6px;
  text-transform: uppercase; }}
.lab-badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 13px;
  font-weight: 600; white-space: nowrap; }}
.lab-badge.blue {{ background: {_v('blue_light')}; color: {_v('blue_dark')}; }}
.lab-badge.orange {{ background: {_v('orange_light')}; color: {_v('orange_dark')}; }}
.lab-badge.neutral {{ background: {_v('header_row')}; color: {_v('body')}; }}
.lab-status {{ display: flex; align-items: center; gap: 12px; background: {_v('panel')};
  border: 1px solid {_v('rule')}; border-radius: 10px; padding: 12px 16px; font-size: 14px;
  color: {_v('body')}; }}
.lab-alert {{ display: flex; gap: 12px; background: {_v('orange_light')};
  border: 1px solid {_v('orange_mid')}; border-radius: 12px; padding: 16px 20px;
  color: {_v('orange_darker')}; font-size: 14px; line-height: 1.55; }}
.lab-alert b {{ color: {_v('orange_darker')}; }}
.lab-note {{ background: {_v('soft_panel')}; border-radius: 8px; padding: 12px 14px;
  font-size: 14px; color: {_v('body_2')}; line-height: 1.5; }}
.lab-note.blue {{ background: {_v('blue_light')}; color: {_v('blue_dark')}; }}
.lab-note.orange {{ background: {_v('orange_light')}; color: {_v('orange_dark')}; }}
.lab-placeholder {{ font-family: var(--lab-mono); color: {_v('muted')}; }}
.lab-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
.lab-table th {{ background: {_v('header_row')}; font-size: 13px; font-weight: 600;
  color: {_v('muted')}; text-align: left; padding: 10px 12px; vertical-align: bottom; }}
.lab-table th.num, .lab-table td.num {{ text-align: right; }}
.lab-table td {{ padding: 12px 12px; border-top: 1px solid {_v('light_rule')};
  vertical-align: middle; color: {_v('ink')}; }}
.lab-table td.mono, .lab-table td.num {{ font-family: var(--lab-mono); }}
.lab-table tr.leader td {{ background: {_v('leader_row')}; }}
.lab-table tr.risk td {{ background: {_v('risk_row')}; }}
.lab-table tr.clickable {{ cursor: pointer; }}
.lab-table tr.clickable:hover td {{ background: {_v('hover')}; }}
.lab-table tr.clickable:focus-visible {{ outline: 2px solid {_v('blue')}; }}
.lab-table .group th {{ background: {_v('header_row')}; text-align: center; color: {_v('blue')};
  border-bottom: 2px solid {_v('blue')}; }}
.lab-table .group th.plain {{ color: {_v('muted')};
  border-bottom: 2px solid {_v('control_border')}; }}
.lab-table .group th.blank {{ border-bottom: none; }}
.lab-table.plain th {{ background: transparent; border-bottom: 1px solid {_v('rule')}; }}
.lab-table-wrap {{ background: {_v('panel')}; border: 1px solid {_v('rule')};
  border-radius: 12px; overflow: hidden; }}
.lab-table-wrap.scroll {{ overflow-x: auto; }}
.lab-table-foot {{ display: flex; justify-content: space-between; align-items: center;
  padding: 12px 24px; border-top: 1px solid {_v('light_rule')}; font-size: 14px;
  color: {_v('body_2')}; }}
.lab-cell-main {{ font-weight: 600; font-size: 14px; }}
/* ranking: a configuration name stays on at most two lines at 1,440 pixels */
.lab-ranking .lab-table td, .lab-ranking .lab-table th {{ padding-left: 8px;
  padding-right: 8px; }}
.lab-ranking .lab-table td.num {{ font-size: 13px; white-space: nowrap; }}
.lab-ranking .lab-cell-main, .lab-ranking .lab-cell-sub {{ line-height: 1.35; }}
.lab-cell-sub {{ font-size: 13px; color: {_v('muted')}; margin-top: 2px;
  font-family: var(--lab-sans); }}
.lab-kv {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
.lab-kv td {{ padding: 9px 0; border-top: 1px solid {_v('light_rule')}; vertical-align: top; }}
.lab-kv td:first-child {{ color: {_v('body_2')}; width: 42%; padding-right: 16px; }}
.lab-kv td.mono {{ font-family: var(--lab-mono); }}
.lab-bar-row {{ display: flex; flex-direction: column; gap: 6px; }}
.lab-bar-head {{ display: flex; justify-content: space-between; font-size: 14px;
  color: {_v('body')}; }}
.lab-bar-track {{ height: 8px; background: {_v('track')}; border-radius: 999px;
  overflow: hidden; }}
.lab-bar-fill {{ height: 8px; border-radius: 999px; }}
.lab-chip {{ display: inline-flex; align-items: center; gap: 10px; padding: 10px 14px;
  background: {_v('blue_light')}; border: 1px solid {_v('blue')}; border-radius: 8px;
  font-size: 14px; color: {_v('blue_dark')}; }}
.lab-chip.add {{ background: {_v('panel')}; border: 1px dashed {_v('dashed')};
  color: {_v('ink')}; }}
.lab-dl {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }}
.lab-finding {{ display: grid; grid-template-columns: 110px 1fr 1fr; gap: 20px; padding: 14px 0;
  border-top: 1px solid {_v('light_rule')}; }}
.lab-legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px;
  color: {_v('body_2')}; }}
.lab-legend span.sw {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px;
  margin-right: 6px; vertical-align: -1px; }}
@media (max-width: 900px) {{
  [data-testid="stMainBlockContainer"] {{ padding: 24px 16px 48px 16px; }}
  .lab-grid {{ grid-template-columns: 1fr !important; }}
  .lab-finding {{ grid-template-columns: 1fr; gap: 8px; }}
}}
"""


#: line icons, drawn as CSS masks filled with the text color (st.html strips inline SVG)
_ICON_PATHS = {
    "check": "<path d='M20 6L9 17l-5-5'/>",
    "warn": "<circle cx='12' cy='12' r='9'/><path d='M12 7v6'/><path d='M12 16.5v.5'/>",
    "lock": ("<rect x='5' y='11' width='14' height='10' rx='2'/>"
             "<path d='M8 11V7a4 4 0 0 1 8 0v4'/>"),
    "triangle": "<path d='M12 3l10 18H2z'/><path d='M12 10v5'/><path d='M12 18v.5'/>",
    "left": "<path d='M15 18l-6-6 6-6'/>",
    "right": "<path d='M9 18l6-6-6-6'/>",
}


def _icon_css() -> str:
    rules = [".lab-icon { display: inline-block; flex-shrink: 0; background-color: currentColor;"
             " -webkit-mask: var(--lab-icon) no-repeat center / contain;"
             " mask: var(--lab-icon) no-repeat center / contain; vertical-align: -3px; }"]
    for name, body in _ICON_PATHS.items():
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
               "stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
               f"{body}</svg>").replace("<", "%3C").replace(">", "%3E")
        rules.append(f'.lab-icon-{name} {{ --lab-icon: url("data:image/svg+xml;utf8,{svg}"); }}')
    return "\n".join(rules) + "\n"


WORKSPACE_CSS += _icon_css()


def chart_layout(*, height: int = 320, x_title: str | None = None, y_title: str | None = None,
                 money_axis: str | None = "y", show_legend: bool = False,
                 theme: str | None = None) -> dict[str, Any]:
    """Plotly layout following the chart rules: chart ground, light grid, muted titled axes.

    Colors come from the active palette (or ``theme``), read when the figure is
    built, so a chart drawn after a theme change takes the new palette.
    """

    c = palette(theme)
    axis = {
        "gridcolor": c["grid"], "zeroline": False, "linecolor": c["rule"],
        "tickfont": {"family": FONT_MONO, "size": 12, "color": c["muted"]},
        "title": {"font": {"family": FONT_SANS, "size": 12, "color": c["muted"]}},
        "automargin": True,
    }
    layout: dict[str, Any] = {
        "height": height,
        "margin": {"l": 8, "r": 16, "t": 12, "b": 8},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": c["chart_ground"],
        # a trace without its own color takes a design color, never the framework's red
        "colorway": [c["blue"], c["ink"], c["blue_mid"], c["orange"], c["muted"]],
        "font": {"family": FONT_SANS, "size": 13, "color": c["body"]},
        "showlegend": show_legend,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right",
                   "x": 1, "font": {"size": 12, "color": c["body_2"]}},
        "xaxis": {**axis, "title": {**axis["title"], "text": x_title or ""}},
        "yaxis": {**axis, "title": {**axis["title"], "text": y_title or ""}},
        "hoverlabel": {"font": {"family": FONT_SANS, "size": 13, "color": c["ink"]},
                       "bgcolor": c["panel"], "bordercolor": c["rule"]},
    }
    if money_axis:
        layout[f"{money_axis}axis"]["tickprefix"] = "$"
        layout[f"{money_axis}axis"]["tickformat"] = "~s"
    return layout


def style_chart(fig: Any, **kwargs: Any) -> Any:
    """Apply :func:`chart_layout` to a Plotly figure (returns the figure)."""

    fig.update_layout(**chart_layout(**kwargs))
    return fig
