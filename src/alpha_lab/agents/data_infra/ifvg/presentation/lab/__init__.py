"""IFVG Lab redesign (September 2026): shared, Streamlit-free building blocks.

The redesigned screens (``docs/ifvg-dashboard-redesign/``) read saved study
records only. This package holds what every screen shares and what can be
tested without a browser:

- ``format``: money, percentages, prices and Chicago 12-hour times, one way.
- ``html``: escaped HTML building blocks drawn from the design system (cards,
  tiles, badges, tables, alerts, chips, placeholders).
- ``theme``: the color and type tokens as CSS, and the chart template.
- ``funded_data``: read-only access to one saved funded comparison result
  (pairs, ordered trades, the study's trading calendar, daily results).
- ``funded_measures`` and the other calculation modules: every number defined
  in ``docs/ifvg-dashboard-redesign/CALCULATIONS.md``.

Nothing here writes a store, launches work or changes a money figure; firms are
never added together.
"""
