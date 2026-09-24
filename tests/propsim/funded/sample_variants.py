"""Extra SYNTHETIC result variants for the funded results screen and export tests.

Every variant here is SYNTHETIC: hand-made prices on hand-made dates. They
exercise empty and bad-outcome states only; they are not historical results.
"""

from __future__ import annotations

from tests.propsim.funded.builders import execution, make_inputs, path


def all_failed_no_payout_result() -> dict:
    """Every account in both firms is lost on its first trade; no payout ever.

    One large losing trade on the first day breaches all ten starting
    accounts; no monthly credit arrives before the cutoff, so every slot
    waits for a credit and nothing is received.
    """

    from alpha_lab.propsim.funded.campaign import resumed_equivalence, run_campaign
    from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600
    from alpha_lab.propsim.funded.plan import (
        FUNDED_QUESTION,
        MATERIAL_LIMITATIONS,
        PILOT_OWNER_DECISIONS,
    )
    from alpha_lab.propsim.funded.result import build_result, validate_result

    day = "2026-01-13"
    ex = execution("loss", entry=f"{day}T15:00:00Z", exit=f"{day}T16:00:00Z",
                   move_usd=-2_500, day=day, reason="stop", stop_usd=2_500)
    points = [(f"{day}T15:10:00Z", -1_000), (f"{day}T15:20:00Z", -2_100)]
    inputs = make_inputs([ex], [path(ex, points)], days=[day, "2026-01-14"],
                         start="2026-01-12T23:00:00Z", cutoff="2026-01-14T22:00:00Z",
                         cost_per_side_cents=514, processing=TWO_BUSINESS_DAYS_FED_1600)
    instances = run_campaign(inputs)
    result = build_result(
        inputs, instances,
        run_identity={"label": "SYNTHETIC ENGINEERING SAMPLE — not historical"},
        price_evidence={"policy": "synthetic_fixture", "trades_with_ordered_prints": 1,
                        "trades_with_minute_approximation": 0},
        context={
            "funded_plan_id": "1" * 64, "purpose": "engineering_sample",
            "question": FUNDED_QUESTION,
            "source": {"description": "Synthetic losing fixture (not a study)",
                       "profile_id": "SYNTHETIC"},
            "owner_decisions": [d.model_dump(mode="json") for d in PILOT_OWNER_DECISIONS],
            "limitations": list(MATERIAL_LIMITATIONS),
        },
    )
    result["validation"] = validate_result(result, instances, resumed_equivalence(inputs))
    return result
