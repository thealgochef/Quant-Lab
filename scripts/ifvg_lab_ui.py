"""Shared Streamlit shell for the redesigned IFVG Lab (both applications).

- :func:`inject_theme` adds the design tokens once per page run, on the palette
  of the theme the page shows (light or dark, following Settings → Choose app
  theme or the system setting);
- :func:`rail` draws the dark left navigation (My studies, Trade review, New study);
- :func:`show` renders escaped HTML from ``presentation.lab.html``;
- :func:`clickable` renders HTML whose ``data-action`` elements send their action
  back to Python (Streamlit components v2, no page reload, no new session);
- :func:`switch` and :func:`tabs` are the design system's segmented switch and
  text tabs;
- :func:`funded_study` caches one verified saved result per process (read only).

Session keys of the redesign use the ``ifvg_lab_v1_`` prefix.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import streamlit as st

from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme as _theme
from alpha_lab.agents.data_infra.ifvg.presentation.lab.html import Markup
from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import (
    THEME_KEY,
    WORKSPACE_CSS,
    theme_css_vars,
)

__all__ = [
    "PREFIX",
    "NOTE",
    "active_theme",
    "clickable",
    "funded_study",
    "inject_theme",
    "plot",
    "rail",
    "show",
    "switch",
    "tabs",
]

PREFIX = "ifvg_lab_v1_"
NOTE = "All times Chicago, 12-hour"
RAIL_ITEMS = ("My studies", "Trade review", "New study")
_RAIL_HELP = {
    "My studies": "Every study from both applications, with their results and drafts.",
    "Trade review": "Review individual trades on their stored candles.",
    "New study": "Set up a new funded comparison or another study type.",
}

_CLICK_JS = """
export default function(component) {
  const { data, setTriggerValue, parentElement } = component;
  let root = parentElement.querySelector(':scope > .lab-click-root');
  if (!root) {
    root = document.createElement('div');
    root.className = 'lab-click-root';
    parentElement.appendChild(root);
  }
  root.innerHTML = data || '';
  const send = (el) => setTriggerValue('action', el.getAttribute('data-action'));
  root.querySelectorAll('[data-action]').forEach((el) => {
    el.addEventListener('click', (event) => { event.preventDefault(); send(el); });
    el.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); send(el); }
    });
  });
}
"""

_clickable = st.components.v2.component("ifvg_lab_clickable", js=_CLICK_JS,
                                        isolate_styles=False)


_THEME_JS = r"""
export default function(component) {
  const { data, setTriggerValue, parentElement } = component;
  const root = document.documentElement;
  const known = (data && data.theme) || null;
  const parse = (text) => {
    const m = /rgba?\(\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)(?:[,\s/]+([\d.]+))?/.exec(text || '');
    if (!m) return null;
    const alpha = m[4] === undefined ? 1 : parseFloat(m[4]);
    if (alpha === 0) return null;
    return [m[1], m[2], m[3]].map(Number);
  };
  const luminance = ([r, g, b]) => {
    const lin = (v) => {
      v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
  };
  // the framework paints its own background on .stApp (the page ground sits on
  // the view container above it), so that color tells light from dark
  const detect = () => {
    const app = document.querySelector('.stApp');
    const rgb = app ? parse(getComputedStyle(app).backgroundColor) : null;
    if (rgb) return luminance(rgb) < 0.5 ? 'dark' : 'light';
    const media = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
    return media && media.matches ? 'dark' : 'light';
  };
  const state = parentElement.__labTheme
    || (parentElement.__labTheme = { sent: null, known: null });
  if (state.known !== known) { state.known = known; state.sent = null; }
  const apply = () => {
    const current = detect();
    if (root.getAttribute('data-lab-theme') !== current) {
      root.setAttribute('data-lab-theme', current);
    }
    if (current !== known && current !== state.sent) {
      state.sent = current;
      setTriggerValue('theme', current);
    }
  };
  apply();
  if (state.cleanup) state.cleanup();
  const app = document.querySelector('.stApp');
  const observer = new MutationObserver(apply);
  if (app) observer.observe(app, { attributes: true, attributeFilter: ['class', 'style'] });
  const media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  if (media && media.addEventListener) media.addEventListener('change', apply);
  const timer = setInterval(apply, 1500);
  state.cleanup = () => {
    observer.disconnect();
    if (media && media.removeEventListener) media.removeEventListener('change', apply);
    clearInterval(timer);
  };
  return state.cleanup;
}
"""

_theme_probe = st.components.v2.component("ifvg_lab_theme", js=_THEME_JS,
                                          isolate_styles=False)


def _script_theme() -> str | None:
    """The theme of the running page: what its probe confirmed, else what it reported."""

    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is None:  # no page: a headless call
        return None
    value = st.session_state.get(THEME_KEY)
    if value in ("light", "dark"):
        return value
    try:
        return st.context.theme.type
    except Exception:
        return None


_theme.set_theme_resolver(_script_theme)


def active_theme() -> str:
    """``"dark"`` or ``"light"``: the palette this page run draws with."""

    return _theme.active_theme()


def _reported_theme(st_module) -> str:
    try:
        value = st_module.session_state.get(THEME_KEY)
    except Exception:
        value = None
    if value not in ("light", "dark"):
        try:
            value = st_module.context.theme.type
        except Exception:
            value = None
    return "dark" if value == "dark" else "light"


def _probe_theme(theme: str, st_module) -> str | None:
    """Mount the page probe; returns the theme it reported this run, if it differed."""

    try:
        with st_module.container(key="ifvg_lab_theme_probe"):
            result = _theme_probe(data={"theme": theme}, key=f"{PREFIX}theme_probe",
                                  on_theme_change=lambda: None)
    except Exception:  # headless test runner (no component registry), or a stand-in
        return None
    try:
        value = getattr(result, "theme", None)
    except Exception:
        return None
    return value if value in ("light", "dark") else None


def inject_theme(st_module=st) -> None:
    """Add the design tokens once per page run, on the palette of the page's theme.

    The application's theme (Settings → Choose app theme, or the system
    setting) chooses between the light and the dark palette. The framework
    reports its theme with the run; a probe in the page confirms it from what
    is actually drawn, switches the palette at once when the theme changes and
    asks for one rerun so the charts follow. Nothing is saved.
    """

    theme = _reported_theme(st_module)
    css = WORKSPACE_CSS
    if theme == "dark":
        # the dark palette from the first paint, before the probe marks the page
        css += f":root:not([data-lab-theme]) {{ {theme_css_vars('dark')} }}"
    st_module.html(f"<style>{css}</style>")
    confirmed = _probe_theme(theme, st_module)
    if confirmed:
        with contextlib.suppress(Exception):  # a stand-in without session state
            st_module.session_state[THEME_KEY] = confirmed


def show(markup: Markup | str, st_module=st) -> None:
    """Render static HTML built by ``presentation.lab.html`` (no scripts)."""

    st_module.html(str(markup))


def clickable(markup: Markup | str, *, key: str, st_module=st) -> str | None:
    """Render HTML; returns the ``data-action`` of the element clicked this run, if any."""

    try:
        result = _clickable(data=str(markup), key=f"{PREFIX}click_{key}",
                            on_action_change=lambda: None)
    except TypeError:
        # headless test runner (no component registry): show the same markup, static
        st_module.html(str(markup))
        return None
    try:
        return getattr(result, "action", None)
    except Exception:  # an unrendered result in headless tests
        return None


#: the main application's other workspaces (set by scripts/dashboard.py), shown in the rail
_OTHER_PAGES: list[Any] = []


def set_other_pages(pages: Sequence[Any]) -> None:
    """Pages of the main application reachable from the rail (none in the dedicated app)."""

    _OTHER_PAGES[:] = list(pages)


def rail(active: str, st_module=st) -> str | None:
    """The dark left rail; returns the destination clicked this run (None otherwise)."""

    chosen = None
    with st_module.sidebar, st_module.container(key="ifvg_lab_rail"):
        st_module.html('<div class="lab-rail-title">IFVG Lab</div>')
        for item in RAIL_ITEMS:
            if st_module.button(item, key=f"{PREFIX}rail_{item.replace(' ', '_')}",
                                type="primary" if item == active else "secondary",
                                help=_RAIL_HELP[item], width="stretch"):
                chosen = item
        with st_module.container(key="ifvg_lab_rail_bottom"):
            if _OTHER_PAGES:
                from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import (
                    RAIL_LINK_CSS,
                )

                # shipped with the rail, so the links look the same on every page
                st_module.html(f"<style>{RAIL_LINK_CSS}</style>"
                               '<div class="lab-rail-note" style="margin-top:0">'
                               "Other workspaces</div>")
                for page in _OTHER_PAGES:
                    try:
                        st_module.page_link(page, label=str(getattr(page, "title", page)))
                    except Exception:  # a page object this Streamlit cannot link
                        continue
            st_module.html(f'<div class="lab-rail-note">{NOTE}</div>')
    return chosen


def switch(label: str, options: Sequence[Any], *, key: str, value: Any = None,
           default: Any = None, format_func: Callable[[Any], str] = str,
           help: str | None = None, st_module=st) -> Any:  # noqa: A002
    """Segmented switch (2–4 joined buttons) that follows an outside ``value``.

    ``value`` is the shared selection (for example the firm every funded view
    shows); a click here changes it, a click on the selected option keeps it,
    and a change made on another screen is shown here on the next run.
    """

    state_key = f"{PREFIX}{key}"
    seen_key = f"{state_key}__seen"
    options = list(options)
    seen = st_module.session_state.get(seen_key)
    target = (value if value in options else seen if seen in options
              else default if default in options else options[0])
    widget = st_module.session_state.get(state_key)
    if widget in options and widget != seen:  # clicked since the last run
        target = widget
    st_module.session_state[state_key] = target
    # a horizontal radio drawn as joined buttons (theme: st-key-ifvg_lab_switch_*);
    # it can't be deselected, and headless tests can drive it like any radio
    with st_module.container(key=f"ifvg_lab_switch_{key}"):
        chosen = st_module.radio(
            label, options, key=state_key, format_func=format_func, horizontal=True,
            help=help or f"Choose the {label.lower()}.", label_visibility="collapsed")
    chosen = chosen if chosen in options else target
    st_module.session_state[seen_key] = chosen
    return chosen


def pending_switch(key: str, options: Sequence[Any], value: Any, st_module=st) -> Any:
    """What :func:`switch` will return this run — for content drawn above the switch."""

    state_key = f"{PREFIX}{key}"
    widget = st_module.session_state.get(state_key)
    if widget in list(options) and widget != st_module.session_state.get(f"{state_key}__seen"):
        return widget
    return value


def tabs(options: Sequence[str], *, key: str, value: str | None = None,
         st_module=st) -> str:
    """Text tabs with an ink underline on the selected one (only that tab renders)."""

    with st_module.container(key=f"ifvg_lab_tabs_{key}"):
        return switch("Detail section tabs", options, key=f"tabs_{key}", value=value,
                      help="Choose which part of this configuration's detail to show.",
                      st_module=st_module)


def plot(fig: Any, *, key: str, st_module=st) -> None:
    st_module.plotly_chart(fig, key=f"{PREFIX}plot_{key}", width="stretch",
                           config={"displayModeBar": False, "responsive": True})


@st.cache_resource(show_spinner="Opening the saved study…", max_entries=8)
def _open_study(store_root: str, result_id: str, source_signature: tuple = ()):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import open_funded_study

    return open_funded_study(Path(store_root), result_id)


def funded_study(store_root: Path | str, result_id: str):
    """The verified saved result and its plan, opened once per process (read only)."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.external_catalog import (
        resolve_registered_result,
    )

    root = Path(store_root).resolve()
    published = resolve_registered_result(result_id, external_store_root=root)
    signature = ()
    if published is not None:
        pointer = published["catalog_binding"]
        signature = (pointer["binding_id"], pointer["reporting_definition_version"],
                     pointer["view_sha256"])
    return _open_study(str(root), result_id, signature)
