"""Research explanations are separate from the developer control registry."""

from .workspace_mode import technical_details_enabled

_HELP = {
    "context.artifact_pair": "Select a prepared strategy configuration to inspect its evidence.",
    "verifier.exact_candidate_id": "Select an entry opportunity to inspect its evidence.",
    "verifier.exact_setup_id": "Select any setup, including those without entry opportunities.",
    "verifier.mode": (
        "Full history includes the outcome. Point-in-time review hides evidence "
        "that was not yet available."
    ),
    "verifier.stage_scrubber": "Show only evidence available at or before the selected event.",
    "verifier.setup_stage": "Show only evidence available at or before the selected setup event.",
    "verifier.range": "Choose which part of the setup or trade the chart covers.",
    "verifier.day_range": "Limit the list to the selected trading days.",
    "verifier.outcome_filter": "Filter opportunities by recorded execution state or outcome.",
    "verifier.session_filter": "Filter by the session of the entry opportunity.",
    "verifier.include_warmup": "Warmup examples are excluded from research and execution results.",
    "verifier.setup_warmup": "Warmup setups are excluded from research and execution results.",
    "verifier.review_notes": "Record your observations about this selected case.",
    "verifier.review_tags": "Add optional labels to your review.",
    "verifier.detail_verdict": "Optionally record a judgment about one part of the setup.",
    "verifier.download_ledger": "Download the saved reviews for this case.",
    "verifier.layer_structure": "Show price structure over the selected chart.",
    "verifier.layer_displacement": "Show displacement measurements available at the selected time.",
    "verifier.layer_pools": "Show equal-high and equal-low price pools.",
    "verifier.layer_sessions": "Show session boundaries on the chart.",
    "verifier.layer_projection": "Project higher-timeframe zones onto the execution chart.",
}


def help_text(control_id: str) -> str:
    if technical_details_enabled():
        from .help_registry import help_text as technical_help

        return technical_help(control_id)
    return _HELP.get(
        control_id, "Adjust the selected review. This does not change the saved research evidence."
    )
