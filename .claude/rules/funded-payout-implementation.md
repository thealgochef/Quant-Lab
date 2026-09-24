# Funded payout implementation

Current scope (September 23, 2026): the single-account configuration comparison in `docs/funded-payout-implementation/AGENT_SCOPE_UPDATE.md`, reconciled into `SPEC.md` (Part A is current; Part B is the deferred five-account budgeted mode). Confirmed choices are in `OWNER_DECISIONS.md`; progress, open work and active jobs are in `TASKS.md`. `CODE_MAP.md` separates verified current anchors from older inspection notes.

In this mode each resolved configuration and each selected firm is one separate comparison with at most one live funded account at a time. Failures are replaced at $102 / $125 with no credit limit. There is no five-account start, monthly credit, growth, copied account or pooled portfolio. A payout-processing account is alive and is never replaced or supplemented. Do not ask again about settled money rules or the two-business-day clock.

Read applicable repository instructions, including AGENTS.md. Do not replace existing instructions or silently resolve a conflict in money or access rules. Check running jobs before a launch, retry or new writer. Do not manufacture approvals, launch unrequested sweeps, use new dates or change live settings. Use normal permissions.

A saved comparison draft that the running Strategy-Core cannot represent opens read-only and must never be re-saved from that app. A half-exit draft can be edited, approved or run only in `python scripts/run_ifsm_research_ui.py --research-core <checkout>`. Any other unrepresentable draft (missing study, unknown firm or value) stays read-only; clone it and edit the copy, which needs its own approval. A launch must match the identical approved plan rebuilt from the saved draft (ARCHITECTURE.md, IFVG dashboard repairs).

Explain outcomes in plain English with full firm names, full chart timeframes and Chicago times in 12-hour AM/PM format. Rank by received cash after all account costs; account longevity and trade frequency are not winning criteria. Label assumptions and unverified price paths.

Separate documentation, code changes, passing tests, simulated historical results and unverified work. Review exports contain only the specified documents and necessary financial evidence; scripts and raw dumps stay internal.
