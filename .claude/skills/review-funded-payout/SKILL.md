---
name: review-funded-payout
description: Review the funded payout implementation without making changes, using its saved requirements and task evidence.
disable-model-invocation: true
model: claude-opus-5-5
effort: medium
---
Use the `funded-payout-reviewer` agent to review the implementation against `docs/funded-payout-implementation/REVIEW.md`, which covers the current single-account configuration comparison. Provide its source roots, baselines and change/test evidence from TASKS.md. Inspect the findings and report unresolved blockers and unperformed checks. Never run a study, worker or financial simulation as a review side effect. Do not modify code or treat missing evidence as a pass. If the helper is unavailable, perform the same read-only file review and disclose that it was not delegated.
