"""Pure presentation metadata for the IFVG Lab UI (UI-1; plan §6).

Everything here is Streamlit-free, deterministic and non-research-bearing:
the one status vocabulary every screen maps onto, the presentation-only run
purpose (which derives the run scope, the semantic namespace class and the
authorization class the actual computation path requires), and the charter
satisfiability report that refuses contradictory drafts BEFORE freeze. No
module in this package mints an identity, touches a store on import, or
duplicates a backend contract — namespace ids, authorization readiness and
execution truth are consumed from ``ifvg.search`` through read-only calls.
"""
