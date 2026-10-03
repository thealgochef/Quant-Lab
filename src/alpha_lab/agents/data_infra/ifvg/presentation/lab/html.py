"""Escaped HTML building blocks for the redesigned screens (``DESIGN_SYSTEM.md`` components).

Every builder returns :class:`Markup` (HTML that is already safe to insert).
Plain ``str`` arguments are always escaped; pass :class:`Markup` to insert
HTML built by another builder. Elements carrying ``data-action`` become
clickable when rendered through the workspace's clickable renderer, which
sends the action text back to Python; nothing here runs code.

Colors are palette variables (``var(--lab-…)``, see ``theme.css_var``), so the
same markup reads on the light and the dark theme.
"""

from __future__ import annotations

import html as _html
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Column",
    "Markup",
    "Row",
    "alert",
    "badge",
    "bars",
    "breadcrumb",
    "card",
    "cell_two_lines",
    "esc",
    "finding_row",
    "grid",
    "icon",
    "join",
    "kv_table",
    "legend",
    "link",
    "note",
    "page_header",
    "placeholder",
    "section_title",
    "status_line",
    "table",
    "tile",
    "verdict_card",
]


class Markup(str):
    """HTML that is already escaped; inserted as is by the builders."""

    __slots__ = ()


def esc(value: Any) -> Markup:
    """Escape any value for HTML (``Markup`` passes through unchanged)."""

    if isinstance(value, Markup):
        return value
    return Markup(_html.escape("" if value is None else str(value), quote=True))


def join(parts: Iterable[Any], sep: str = "") -> Markup:
    return Markup(esc(sep).join(esc(p) for p in parts))


def _attr(value: Any) -> str:
    return _html.escape(str(value), quote=True)


_ICON_NAMES = ("check", "warn", "lock", "triangle", "left", "right")


def icon(name: str, *, color: str = "var(--lab-blue)", size: int = 18) -> Markup:
    """A line icon drawn by the stylesheet (``lab-icon-*`` classes in ``theme.py``).

    ``st.html`` removes inline SVG, so the shape is a CSS mask filled with the
    element's text color; it renders the same through every renderer.
    """

    name = name if name in _ICON_NAMES else "check"
    return Markup(f'<span class="lab-icon lab-icon-{name}" aria-hidden="true" '
                  f'style="color:{_attr(color)};width:{int(size)}px;height:{int(size)}px">'
                  "</span>")


def link(text: Any, action: str) -> Markup:
    """An in-app link: clicking sends ``action`` to Python (clickable renderer only)."""

    return Markup(f'<a href="#" data-action="{_attr(action)}">{esc(text)}</a>')


def breadcrumb(parts: Sequence[tuple[Any, str | None]]) -> Markup:
    """``My studies / Funded variation study / Rank 1 at TakeProfitTrader``."""

    items = [link(text, action) if action else esc(text) for text, action in parts]
    return Markup(f'<div class="lab-crumb">{Markup(" / ").join(items)}</div>')


def page_header(title: Any, *, crumbs: Sequence[tuple[Any, str | None]] = (),
                subtitle: Any = None, meta: Any = None, detail: bool = False) -> Markup:
    out = ['<div class="lab" style="display:flex;flex-direction:column;gap:8px">']
    if crumbs:
        out.append(breadcrumb(crumbs))
    out.append(f'<h1 class="lab-h1{" detail" if detail else ""}">{esc(title)}</h1>')
    if subtitle:
        out.append(f'<div class="lab-sub">{esc(subtitle)}</div>')
    if meta:
        out.append(f'<div class="lab-line">{esc(meta)}</div>')
    out.append("</div>")
    return Markup("".join(out))


def section_title(text: Any, *, size: str = "h2", right: Any = None) -> Markup:
    title = f'<div class="lab-{size}">{esc(text)}</div>'
    if right is None:
        return Markup(f'<div class="lab">{title}</div>')
    return Markup('<div class="lab" style="display:flex;justify-content:space-between;'
                  f'align-items:baseline;gap:16px">{title}<div class="lab-line">{esc(right)}'
                  '</div></div>')


def badge(text: Any, tone: str = "neutral") -> Markup:
    tone = tone if tone in ("blue", "orange", "neutral") else "neutral"
    return Markup(f'<span class="lab-badge {tone}">{esc(text)}</span>')


def placeholder(text: Any) -> Markup:
    """Muted mono placeholder for a value the saved records do not hold."""

    return Markup(f'<span class="lab-placeholder">{esc(text)}</span>')


def card(body: Any, *, title: Any = None, right: Any = None, soft: bool = False,
         sans_title: bool = False, style: str = "") -> Markup:
    head = ""
    if title is not None or right is not None:
        klass = "lab-card-title sans" if sans_title else "lab-card-title"
        head = ('<div style="display:flex;justify-content:space-between;align-items:baseline;'
                f'gap:16px"><div class="{klass}">{esc(title or "")}</div>'
                f'<div class="lab-line">{esc(right or "")}</div></div>')
    return Markup(f'<div class="lab lab-card{" soft" if soft else ""}" style="{_attr(style)}">'
                  f"{head}{esc(body)}</div>")


def tile(label: Any, value: Any, caption: Any = None, *, big: bool = False) -> Markup:
    cap = f'<div class="lab-tile-caption">{esc(caption)}</div>' if caption else ""
    return Markup(f'<div class="lab lab-tile"><div class="lab-tile-label">{esc(label)}</div>'
                  f'<div class="lab-tile-value{" big" if big else ""}">{esc(value)}</div>'
                  f"{cap}</div>")


def grid(items: Sequence[Any], columns: int | str, *, gap: int = 12) -> Markup:
    template = (f"repeat({columns}, minmax(0, 1fr))" if isinstance(columns, int) else columns)
    body = "".join(esc(i) for i in items)
    return Markup(f'<div class="lab lab-grid" style="grid-template-columns:{_attr(template)};'
                  f'gap:{gap}px">{body}</div>')


def status_line(text: Any, *, kind: str = "check") -> Markup:
    color = "var(--lab-orange)" if kind in ("warn", "triangle", "lock") else "var(--lab-blue)"
    return Markup(f'<div class="lab lab-status">{icon(kind, color=color)}'
                  f'<span style="flex-grow:1">{esc(text)}</span></div>')


def alert(first: Any, rest: Any = None, *, kind: str = "triangle") -> Markup:
    more = f'<div style="margin-top:4px">{esc(rest)}</div>' if rest else ""
    return Markup(f'<div class="lab lab-alert" role="alert">{icon(kind, color="var(--lab-orange)")}'
                  f"<div><b>{esc(first)}</b>{more}</div></div>")


def note(text: Any, tone: str = "") -> Markup:
    return Markup(f'<div class="lab lab-note {tone}">{esc(text)}</div>')


def cell_two_lines(main: Any, sub: Any = None) -> Markup:
    sub_html = f'<div class="lab-cell-sub">{esc(sub)}</div>' if sub else ""
    return Markup(f'<div class="lab-cell-main">{esc(main)}</div>{sub_html}')


@dataclass(frozen=True)
class Column:
    key: str
    label: str
    align: str = "left"  # left / right
    mono: bool = False
    width: str | None = None


@dataclass(frozen=True)
class Row:
    cells: dict[str, Any]
    tint: str | None = None  # "leader" / "risk"
    action: str | None = None
    label: str | None = None  # accessible name for a clickable row
    extra: dict[str, Any] = field(default_factory=dict)


def table(columns: Sequence[Column], rows: Sequence[Row], *,
          groups: Sequence[tuple[str, int, str]] = (), foot: Any = None,
          plain: bool = False, wrap: bool = True, caption: str | None = None) -> Markup:
    """A design-system table; ``groups`` = (label, column span, style: blue/plain/blank)."""

    head = []
    if groups:
        cells = "".join(f'<th colspan="{span}" class="{style}">{esc(label)}</th>'
                        for label, span, style in groups)
        head.append(f'<tr class="group">{cells}</tr>')
    ths = []
    for c in columns:
        width = f' style="width:{_attr(c.width)}"' if c.width else ""
        klass = ' class="num"' if c.align == "right" else ""
        ths.append(f'<th scope="col"{klass}{width}>{esc(c.label)}</th>')
    head.append(f"<tr>{''.join(ths)}</tr>")
    body = []
    for row in rows:
        klass = [row.tint] if row.tint else []
        attrs = ""
        if row.action:
            klass.append("clickable")
            attrs = (f' data-action="{_attr(row.action)}" role="button" tabindex="0"'
                     f' aria-label="{_attr(row.label or "Open")}"')
        tds = []
        for c in columns:
            value = row.cells.get(c.key, "")
            cls = []
            if c.align == "right":
                cls.append("num")
            elif c.mono:
                cls.append("mono")
            tds.append(f'<td class="{" ".join(cls)}">{esc(value)}</td>')
        body.append(f'<tr class="{" ".join(klass)}"{attrs}>{"".join(tds)}</tr>')
    cap = f"<caption class=\"lab-line\" style=\"text-align:left\">{esc(caption)}</caption>" \
        if caption else ""
    html_table = (f'<table class="lab-table{" plain" if plain else ""}">{cap}'
                  f"<thead>{''.join(head)}</thead><tbody>{''.join(body)}</tbody></table>")
    footer = f'<div class="lab-table-foot">{esc(foot)}</div>' if foot else ""
    if not wrap:
        return Markup(f'<div class="lab" style="overflow-x:auto">{html_table}{footer}</div>')
    return Markup(f'<div class="lab lab-table-wrap scroll">{html_table}{footer}</div>')


def kv_table(rows: Sequence[tuple[Any, Any] | tuple[Any, Any, bool]]) -> Markup:
    """Two-column label/value list (values in mono when the third item is true)."""

    out = []
    for row in rows:
        label, value = row[0], row[1]
        mono = len(row) > 2 and bool(row[2])  # type: ignore[misc]
        out.append(f'<tr><td>{esc(label)}</td><td class="{"mono" if mono else ""}">'
                   f"{esc(value)}</td></tr>")
    return Markup(f'<table class="lab-kv">{"".join(out)}</table>')


def bars(rows: Sequence[tuple[Any, float | None, Any]], *, threshold: float = 0.5) -> Markup:
    """Share bars (label, share 0–1 or None, value text); orange at ``threshold`` and above."""

    out = []
    for label, share, text in rows:
        if share is None:
            fill = ""
        else:
            color = "var(--lab-orange)" if share >= threshold else "var(--lab-blue)"
            width = max(0.0, min(1.0, share)) * 100
            fill = (f'<div class="lab-bar-fill" style="width:{width:.1f}%;'
                    f'background:{color}"></div>')
        out.append(f'<div class="lab-bar-row"><div class="lab-bar-head"><span>{esc(label)}</span>'
                   f'<span class="lab-mono">{esc(text)}</span></div>'
                   f'<div class="lab-bar-track">{fill}</div></div>')
    return Markup('<div class="lab" style="display:flex;flex-direction:column;gap:14px">'
                  + "".join(out) + "</div>")


def legend(items: Sequence[tuple[str, Any]], *, line: bool = False) -> Markup:
    out = []
    for color, label in items:
        shape = (f'<span class="sw" style="background:{_attr(color)};height:3px;width:16px;'
                 'vertical-align:3px"></span>' if line else
                 f'<span class="sw" style="background:{_attr(color)}"></span>')
        out.append(f"<span>{shape}{esc(label)}</span>")
    return Markup(f'<div class="lab lab-legend">{"".join(out)}</div>')


def verdict_card(part: Any, status: Any, tone: str, text: Any) -> Markup:
    status_html = badge(status, tone) if status else badge("Not shown", "neutral")
    return Markup(
        '<div class="lab lab-tile" style="padding:18px 20px;gap:8px">'
        '<div style="display:flex;justify-content:space-between;align-items:center;gap:8px">'
        f'<div style="font-size:14px;font-weight:600">{esc(part)}</div>{status_html}</div>'
        f'<div style="font-size:14px;line-height:1.5;color:var(--lab-body)">{esc(text)}</div>'
        '</div>')


def finding_row(severity: Any, tone: str, title: Any, text: Any, next_step: Any) -> Markup:
    return Markup(
        f'<div class="lab-finding"><div>{badge(severity, tone)}</div>'
        '<div style="display:flex;flex-direction:column;gap:4px">'
        f'<div style="font-size:15px;font-weight:600">{esc(title)}</div>'
        f'<div style="font-size:14px;line-height:1.5;color:var(--lab-body)">{esc(text)}</div></div>'
        '<div style="font-size:14px;line-height:1.5;color:var(--lab-body-2)">'
        f'<span style="font-weight:600;color:var(--lab-ink)">Next step:</span> '
        f'{esc(next_step)}</div>'
        "</div>")
