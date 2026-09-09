"""Regression coverage for research evidence and exact saved-review links."""

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def test_saving_review_with_model_details_preserves_the_selected_trade(monkeypatch):
    import ifvg_research_review as research
    import ifvg_verifier_tab as verifier

    class Screen:
        def subheader(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def caption(self, *args, **kwargs):
            pass

        def checkbox(self, *args, **kwargs):
            return True

        def expander(self, *args, **kwargs):
            return nullcontext()

        def dataframe(self, *args, **kwargs):
            pass

    selected = pd.Series({"decision_id": "exact-decision", "trade_id": "exact-trade"})
    evidence = SimpleNamespace(
        candidate_id="exact-candidate",
        execution=None,
        mode="full",
        stage_gates={},
        counterfactual_labels=(),
        model={"M1": {"probability": 0.6}, "M2": {"probability": 0.7}},
    )
    pair = {"v2_dataset_id": "exact-v2", "v3_dataset_id": "exact-v3"}
    context = SimpleNamespace(
        replay=SimpleNamespace(artifact_id="exact-chart"),
        pair_ref=SimpleNamespace(as_dict=lambda: pair),
    )
    saved = []
    monkeypatch.setattr(verifier, "list_reviews", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(verifier, "append_review", lambda **kwargs: saved.append(kwargs))
    monkeypatch.setattr(verifier, "technical_details_enabled", lambda: False)

    def save_review(_screen, *, on_save, **kwargs):
        on_save(reviewer="Researcher", verdicts={}, tags=[], notes="Reviewed the evidence")

    monkeypatch.setattr(verifier, "_review_form", save_review)
    research.candidate_panel(Screen(), context, evidence, selected)

    assert len(saved) == 1
    assert saved[0]["candidate_id"] == "exact-candidate"
    assert saved[0]["decision_id"] == "exact-decision"
    assert saved[0]["trade_id"] == "exact-trade"
    assert saved[0]["replay_chart_artifact_id"] == "exact-chart"
    assert saved[0]["pair_ref"] == pair
