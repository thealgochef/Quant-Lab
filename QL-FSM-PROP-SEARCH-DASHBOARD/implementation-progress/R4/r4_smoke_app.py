"""R4 browser-smoke entry: ONLY the IFVG Lab shell (Experiments workspace).

Runs the real three-tab shell (Experiments | Replay / Verifier | Data &
Audit) with the R4 study workspace inside Experiments, against the real
repository roots — no other dashboard tabs execute, nothing launches, and
all surfaces render their honest empty/blocked states when no production
artifacts exist. Evidence tool only (never part of the app).

    streamlit run QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R4/r4_smoke_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for entry in (str(_REPO / "scripts"), str(_REPO / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="R4 smoke — IFVG Lab", layout="wide")

from ifvg_lab_tab import render_ifvg_lab_tab  # noqa: E402

render_ifvg_lab_tab()
