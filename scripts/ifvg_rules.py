"""Shared plain-English strategy rules for the focused IFVG workspace."""

from __future__ import annotations

from alpha_lab.agents.data_infra.ifvg.presentation.study_rules import load_study_rules


def render_strategy_rules(
    st,
    study,
    roots,
    *,
    compact=False,
    selected_core_replay_id=None,
    show_heading=True,
):
    description = load_study_rules(study, roots, selected_core_replay_id)
    if show_heading:
        st.markdown("**Strategy rules**")
    if description.is_preview:
        st.caption("Preview · Based on your saved draft settings.")
    if study.kind == "context" and not compact:
        st.caption(
            "These rules describe the underlying strategy. "
            "This study tests predictions about its hypothetical outcomes."
        )
    elif study.kind == "pipeline" and not compact:
        st.caption(
            "These rules describe the underlying strategy. "
            "Model predictions and account simulations are evaluated separately."
        )
    bullets = description.preview_bullets if compact else description.detail_bullets
    if bullets:
        st.markdown("\n".join(f"- {bullet}" for bullet in bullets))
    if description.variations and not compact:
        st.markdown("**What this study changes**")
        st.markdown("\n".join(f"- {bullet}" for bullet in description.variations))
    for issue in description.issues:
        st.caption(issue)
    return description
