# IFSM — one full-range comparison of the intended configurations

This is a ready-written task for the local agent, not a completed backtest.

Place this package's `docs/ifsm-correct-config-full-range-v01/` folder under the Quant-Lab project root. It is a new folder and replaces no existing `.claude`, `CLAUDE.md`, `AGENTS.md`, saved study, or prior task.

Send the agent this one line:

`Read docs/ifsm-correct-config-full-range-v01/TASK.md and execute the one six-configuration batch exactly as specified.`

Alternatively attach TASK.md together with the three JSON support files to the same agent session. No command launcher or new skill is required.

The batch is six established configurations, each measured separately at both firms, over Task B's exact 253 evaluation dates plus ten warmup dates. MenthorQ stays annotation-only in this proposal: this measures the intended configurations before testing restrictive context filters again. The earlier old-baseline drought investigation is deferred.

Files:
- TASK.md — full execution, scope, evidence and completion instruction.
- RUN_REQUEST.json — explicit six-row request and invariants; NOT a direct application import.
- date_scope.json — exact copied Task B dates and source identity.
- source_references.json — four complete historical bindings and two historical reference locators. The agent must still resolve and verify the current worker's complete settings.
- PACKAGE_MANIFEST.json — package integrity, not research approval or test evidence.

Historical reference configuration names are not guarantees of future performance. The new date range uses one continuous earlier initialization; a January–June slice is not a fresh replay of the older study.
