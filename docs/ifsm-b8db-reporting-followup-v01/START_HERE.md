# Isolated reporting follow-up — b8db

This package is for the same agent that completed the IFSM MFFU repair. It authorizes implementation and verification of **B8-01 and B8-02 only**, followed by a compact delivery. It is not another research phase.

Extract this ZIP into the Quant-Lab project root. It adds only `docs/ifsm-b8db-reporting-followup-v01/`. Leave earlier task folders, instructions, archives, and approved study records intact. Send the contents of [AGENT_START.txt](AGENT_START.txt) to the agent.

## Reading order

1. [TASK.md](TASK.md): scope, implementation boundaries, and finish line.
2. [ACCEPTANCE.md](ACCEPTANCE.md): precise behavior and focused validation cases.
3. [RETURN_REQUIREMENTS.md](RETURN_REQUIREMENTS.md): the two review ZIPs to return.
4. [REFERENCE_BINDINGS.json](REFERENCE_BINDINGS.json): exact reviewed result/plan/archive identities and baseline counts.
5. [inputs/REVIEW_FEEDBACK.md](inputs/REVIEW_FEEDBACK.md), [inputs/REVIEW_SOURCE_EXCERPTS.md](inputs/REVIEW_SOURCE_EXCERPTS.md), and [inputs/REVIEW_CHECKPOINT_PROBE.json](inputs/REVIEW_CHECKPOINT_PROBE.json).

[CHECKPOINT_CASES.json](CHECKPOINT_CASES.json) separates the actual saved example from synthetic test recipes. [TASKS.md](TASKS.md) is the new task's progress checklist; the agent maintains it.

## Only two outstanding fixes

- **B8-01:** checkpoint-specific gamma provenance. Match the actual decision/fill despite a known timestamp representation difference; stop borrowing the entry's provenance for later checkpoint cards and coverage.
- **B8-02:** standard export labels. Use the same authoritative policy descriptions in the normal UI, configuration CSV, and generated charts: micro exposure, NQ price proxy, dynamic distance formula, fallback, and conditional exit behavior.

No financial replay, Core execution change, new gamma policy, model training, fee change, or new dashboard is authorized. Rebuild affected reports from existing saved records and retain the normal My studies route.

The original b8db economic result must stay unchanged. Publish a new **reporting/export version**, not a new financial experiment and not an overwrite of either delivered archive. Return one result/UI ZIP and one source/test ZIP.

## Evidence included

`inputs/` contains exact byte copies from the last independent review. Its 64-row financial reference concerns the corrected b8db result, not the earlier 7a8c result. It is a secondary regression reference; the saved verified ledger remains the accounting authority.

`inputs/reviewer_probe_original.py` is preserved reviewer evidence, not an application patch or a ready-to-run project command. Its container paths and isolated-function setup must not be mistaken for your runtime. Use the real reporting functions in your own focused tests; retain the probe's stated limitations.

The agent already has the two reviewed application archives and their stores. They are identified by filename and SHA-256 in REFERENCE_BINDINGS.json rather than copied into this small handoff. Reuse the existing verified sources and data; do not ask the owner to reconstruct them.

Creating this handoff performed no application changes, application tests, financial replay, or model fit. PACKAGE_CHECKS.json records only handoff checks.
