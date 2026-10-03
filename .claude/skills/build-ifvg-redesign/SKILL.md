---
name: build-ifvg-redesign
description: Build the owner-approved IFVG Lab dashboard redesign from the mock screens in docs/ifvg-dashboard-redesign — funded results, the six-tab configuration detail, trade review with real candles, the study library, and the new-study setup and review steps — wired to real study data. No study launches, no strategy or engine changes, no new market data.
disable-model-invocation: true
---

# Build the IFVG Lab redesign

This is the owner's implementation request for the dashboard redesign. The repair task (`/fix-ifvg-dashboard`, R1–R8) comes first; this task builds on it.

Locate the current Quant-Lab project root. Read its applicable project instructions, then read these project-relative files completely, in this order:

1. `docs/ifvg-dashboard-redesign/TASK.md` — the assignment, boundaries, build phases and handoff
2. `docs/ifvg-dashboard-redesign/DECISION_RULES.md` — how to decide anything the mocks and task don't cover
3. `docs/ifvg-dashboard-redesign/SCREENS.md` — every screen and panel, where its data comes from, and what's already built
4. `docs/ifvg-dashboard-redesign/CALCULATIONS.md` — exact definitions and reference values for every number
5. `docs/ifvg-dashboard-redesign/DESIGN_SYSTEM.md` — colors, type, components and chart conventions
6. `docs/ifvg-dashboard-redesign/mocks/README.md` — then look at every image in `mocks/images/`
7. `docs/ifvg-dashboard-redesign/TASKS.md` — the progress ledger you maintain

The mock images are the north star for layout, order, grouping and wording. Numbers come from real data through CALCULATIONS.md, never from the images. When something isn't covered, follow DECISION_RULES.md, log the decision, and keep going — do not stop to ask unless DECISION_RULES.md says to.

Work phase by phase, verify each phase in the running application, keep TASKS.md current, and produce the handoff described in TASK.md. Do not stop at a plan. Do not launch a study, change strategy or account logic, download data, or touch June 11, 2026 onward. If one phase is blocked, record the exact blocker and continue with the next independent phase.
