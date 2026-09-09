"""Shared widget helpers for the FSM/prop study workspace (R4; FUX §4; UI-3 §6.6).

Thin, reusable Streamlit fragments over the pure presentation contracts in
``study_status.py`` / ``study_presentation.py`` and the UI-3 presentation
package (metric registry, roll-ups, helper text, labels). Every helper is
AppTest-friendly (plain functions over an injected ``st_module``), every
error string passes :func:`sanitize_error`, and no helper launches work —
rendering is always side-effect-free (FUX §34).

UI-3 adds the three detail levels (Summary → Research details → Technical
identity & audit) over the persisted ``DisclosureLevel`` vocabulary, the
identity reveal, the metric and roll-up cards built from typed readings,
and the glossary expander.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from alpha_lab.agents.data_infra.ifvg.presentation.help_registry import (
    glossary_markdown,
    help_for_metric,
    help_text,
)
from alpha_lab.agents.data_infra.ifvg.presentation.metric_registry import MetricReading
from alpha_lab.agents.data_infra.ifvg.presentation.rollups import (
    SECTION_LABELS,
    SectionRollup,
)
from alpha_lab.agents.data_infra.ifvg.presentation.status_vocabulary import (
    UiStatus,
    status_chip,
)
from alpha_lab.agents.data_infra.ifvg.presentation.workspace_mode import technical_details_enabled
from alpha_lab.agents.data_infra.ifvg.study_status import (
    DEV_BADGE_TEXT,
    EMPTY_STATE_PRESENTATIONS,
    RESULT_SCOPE_LABELS,
    VERIFICATION_BADGE_TEXT,
    DisclosureLevel,
    ResultScope,
    StudyStatusKey,
    status_presentation,
)

__all__ = [
    "STATE_PREFIX",
    "PIPELINE_STATE_PREFIX",
    "SESSION_DRAFT_KEY",
    "DETAIL_LEVEL_LABELS",
    "sanitize_error",
    "sanitize_select",
    "display_metric",
    "status_badge",
    "dev_only_badge",
    "verification_badge",
    "disclosure_level",
    "detail_levels",
    "identity_block",
    "identity_reveal",
    "status_chip_line",
    "metric_card",
    "rollup_card",
    "glossary_expander",
    "cli_escape_hatch",
    "render_empty_state",
    "paginate_controls",
    "result_scope_caption",
    "queue_replay_drilldown",
]

#: The workspace session-state namespace (FUX §3.2). The existing
#: ``ifvg_context_v1_*`` keys are written only by the registered cross-lane
#: interactions (the verifier jump in :func:`queue_replay_drilldown`).
STATE_PREFIX = "ifvg_study_v1_"

#: FUX §3.2 — the Full Pipeline Run surface's dedicated session namespace.
PIPELINE_STATE_PREFIX = "ifvg_pipeline_v1_"

#: UI-2 (owner Q2): the SESSION-ONLY draft (``asdict(StudyDraft)``) a Start
#: card or "Start new draft" creates — no file exists until the first Save
#: Draft or the first valid Next persists it.
SESSION_DRAFT_KEY = f"{STATE_PREFIX}session_draft"

#: UI-3 (plan §6.6): the three detail levels every major screen renders,
#: keyed by the PERSISTED disclosure vocabulary (unchanged session values).
DETAIL_LEVEL_LABELS: Mapping[str, str] = {
    DisclosureLevel.SUMMARY.value: "Summary",
    DisclosureLevel.ANALYST.value: "Research details",
    DisclosureLevel.AUDIT.value: "Technical identity & audit",
}
_LEVEL_BY_LABEL = {label: level for level, label in DETAIL_LEVEL_LABELS.items()}

#: Quoted paths first (spaces inside quotes survive a bare-token pass),
#: then bare drive-letter / UNC / rooted tokens.
_QUOTED_PATH = re.compile(r"(['\"])(?:[A-Za-z]:\\|\\\\|/)[^'\"]*\1")
_WINDOW_PATH = re.compile(r"(?:[A-Za-z]:\\|\\\\|/)[^\s'\"]+")
_SECRET = re.compile(r"(?i)(token|secret|api[_-]?key)\s*[:=]\s*[^\s,}]+")


def sanitize_error(error: BaseException | str) -> str:
    """Path-, secret-, and traceback-free single-line error text (FUX §34)."""

    message = str(error)
    message = _QUOTED_PATH.sub("<path>", message)
    message = _WINDOW_PATH.sub("<path>", message)
    message = _SECRET.sub(r"\1=<redacted>", message)
    message = message.splitlines()[0] if message else "unknown error"
    return message[:800]


def sanitize_select(st_module, key: str, options: Sequence[str]) -> None:
    """Drop a stale session-state selection the option list no longer has."""

    if key in st_module.session_state and st_module.session_state[key] not in options:
        del st_module.session_state[key]


def display_metric(value: Any, *, percent: bool = False) -> str:
    """Compact metric text: '—' for missing, fixed precision otherwise."""

    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)) and not percent:
        return f"{value:,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if percent:
        return f"{number * 100.0:.1f}%"
    return f"{number:,.2f}"


def status_badge(st_module, key: StudyStatusKey | str, *, help_text: bool = True) -> None:
    """Glyph + visible word + help — color never carries meaning alone."""

    presentation = status_presentation(key)
    st_module.markdown(
        f"**{presentation.glyph} {presentation.visible_label}**",
        help=presentation.help_text if help_text else None,
    )


def dev_only_badge(st_module) -> None:
    """The persistent, non-dismissible development badge (FUX §5.3)."""

    st_module.warning(
        f"Development lane — selections are shown as **{DEV_BADGE_TEXT}**. "
        "No profitability, validation, promotion, production, or "
        "live-readiness claim is made or permitted.",
        icon="🧪",
    )


def verification_badge(st_module) -> None:
    """The non-dismissible verification-scope badge (FUX §14.2)."""

    st_module.error(f"**{VERIFICATION_BADGE_TEXT}**", icon="🔒")


def detail_levels(st_module, *, key: str = f"{STATE_PREFIX}disclosure") -> str:
    """Summary | Research details | Technical identity & audit (UI-3 §6.6).

    Returns the PERSISTED level value (``summary`` / ``analyst`` / ``audit``)
    so every caller's level logic is unchanged; the labels are the UI-3
    tiers. The selection persists in the session under ``key``.
    """

    labels = list(DETAIL_LEVEL_LABELS.values())
    sanitize_select(st_module, key, labels)
    selected = st_module.radio(
        "Detail level",
        labels,
        horizontal=True,
        key=key,
        help=help_text("common.detail_level"),
    )
    return _LEVEL_BY_LABEL[str(selected)]


def disclosure_level(st_module, *, key: str = f"{STATE_PREFIX}disclosure") -> str:
    """The R4 seam (FUX §5.1) — delegates to :func:`detail_levels`."""

    return detail_levels(st_module, key=key)


def identity_block(st_module, label: str, value: str) -> None:
    """A copyable full technical identity (FUX §5.5). Never truncated."""

    if not technical_details_enabled():
        return
    st_module.caption(label)
    st_module.code(value or "—", language=None)


def identity_reveal(st_module, label: str, value: str, *, human: str | None = None) -> None:
    """UI-3 §6.6: the human label first, the full copyable id beneath."""

    caption = f"{human} — {label}" if human else label
    identity_block(st_module, caption, value)


def status_chip_line(st_module, status: UiStatus | str, text: str) -> None:
    """One ``glyph + word — text`` line; color never carries meaning alone."""

    st_module.markdown(f"{status_chip(status)} — {text}")


def metric_card(st_module, reading: MetricReading, *, caption: bool = True) -> None:
    """A registry-described metric: human name, value, status chip, the
    interpretation (reference, sample, caveat) and the 'why' help."""

    st_module.metric(
        reading.human_name,
        reading.display,
        help=help_for_metric(reading.technical_key),
    )
    if caption:
        text = f"{reading.chip} · {reading.interpretation}"
        if reading.caveat:
            text += f" · caveat: {reading.caveat}"
        st_module.caption(text)


def rollup_card(st_module, rollup: SectionRollup) -> None:
    """One section roll-up: status chip + section, the main reason, and the
    next thing to inspect (plan §6.3)."""

    with st_module.container(border=True):
        st_module.markdown(
            f"**{rollup.chip} · {SECTION_LABELS[rollup.section]}** — {rollup.main_reason}"
        )
        st_module.caption(
            f"Inspect next: {rollup.inspect_next} · {rollup.reading_count} reading(s), "
            f"{rollup.passed_count} gated pass(es)"
        )


def glossary_expander(st_module, *, expanded: bool = False) -> None:
    """The searchable-by-browser glossary of the plan's sixteen terms."""

    with st_module.expander("Glossary", expanded=expanded):
        st_module.markdown(glossary_markdown())


def cli_escape_hatch(st_module, command: str, *, reason: str) -> None:
    """Show the exact project-relative CLI command; never executes (FUX §16.7)."""

    if not technical_details_enabled():
        return
    st_module.caption(f"CLI fallback — {reason}. Run from the repository root:")
    st_module.code(command, language="text")


def render_empty_state(
    st_module, state_id: str, *, detail: str | None = None
) -> None:
    """One §31 intentional empty/blocked/failure state, fully sanitized."""

    if not technical_details_enabled():
        st_module.warning(
            {
                "purpose_unresolved": (
                    "This study's research scope is unresolved. Confirm its intended scope "
                    "before continuing."
                ),
                "authorization_not_ready": (
                    "Required research authorization is not ready. Save the study and return "
                    "when authorization is available."
                ),
                "store_namespace_unverified": (
                    "Research storage could not be verified. Running is unavailable until "
                    "the storage evidence is restored."
                ),
                "runner_unavailable": (
                    "The required research runner is unavailable. Your saved study can be "
                    "resumed when that dependency is ready."
                ),
                "runner_executor_planned": (
                    "This workflow's research runner is not available. "
                    "Its saved settings are retained."
                ),
                "launch_not_started": (
                    "The worker has not saved a startup status. The study is not confirmed "
                    "as running; refresh progress before trying again."
                ),
            }.get(
                state_id,
                "Required evidence is unavailable. Restore the missing evidence and refresh "
                "to continue.",
            )
        )
        return
    presentation = EMPTY_STATE_PRESENTATIONS[state_id]
    st_module.subheader(presentation.heading)
    st_module.write(presentation.explanation)
    if detail:
        st_module.caption(sanitize_error(detail))
    if presentation.next_action:
        st_module.info(f"Next action: {presentation.next_action}")
    with st_module.expander("Audit disclosure"):
        st_module.write(f"Owning gate: {presentation.owning_gate or '—'}")
        st_module.write(f"State key: `{presentation.key.value}`")


def paginate_controls(
    st_module,
    total: int,
    *,
    key: str,
    page_size_options: tuple[int, ...] = (25, 50, 100),
) -> tuple[int, int]:
    """Indexed pagination controls → (start, end) row slice (FUX §33)."""

    from alpha_lab.agents.data_infra.ifvg.study_presentation import page_slice

    size_key, page_key = f"{key}_page_size", f"{key}_page"
    size = int(
        st_module.selectbox(
            "Rows per page",
            page_size_options,
            key=size_key,
            help=help_text("common.rows_per_page"),
        )
    )
    n_pages = max(1, -(-total // size))
    page = int(
        st_module.number_input(
            "Page",
            min_value=1,
            max_value=n_pages,
            step=1,
            key=page_key,
            help=help_text("common.page"),
        )
    )
    start, end, _ = page_slice(total, page - 1, size)
    st_module.caption(f"Rows {start + 1}–{end} of {total}")
    return start, end


def result_scope_caption(st_module, scope: ResultScope | str) -> None:
    """The mandatory explicit result-scope label (FUX §5.2)."""

    label = RESULT_SCOPE_LABELS[ResultScope(scope)]
    st_module.caption(f"Result scope: **{label}**")


def queue_replay_drilldown(
    st_module,
    kind: str,
    value: str,
    *,
    pair_label: str | None = None,
    toast: Callable[[str], None] | None = None,
) -> bool:
    """Queue an EXACT verifier jump (FUX §27).

    Writes the existing verifier pair key (when a catalog pair label is
    supplied) plus the exact ``queue_jump`` for one of the four supported id
    kinds. No setup/time, nearest-time, row-order, keep-last, or fuzzy
    fallback exists; unresolved ids terminate in a sanitized warning inside
    the verifier itself.
    """

    try:
        from ifvg_verifier_tab import queue_jump  # noqa: PLC0415

        if pair_label is not None:
            st_module.session_state["ifvg_context_v1_replay_pair"] = pair_label
        queue_jump(kind, value)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.warning(f"Exact jump unavailable: {sanitize_error(error)}")
        return False
    notify = toast if toast is not None else getattr(st_module, "toast", None)
    st_module.session_state["ifvg_study_v1_open_review"] = True
    if notify is not None:
        notify(
            f"Loaded {kind} {value[:12]}… — open the Replay / Verifier tab."
            if technical_details_enabled() else "Supporting case selected in Trade review."
        )
    return True
