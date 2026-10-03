# Start the redesign fixes

Merge this ZIP's `.claude` and `docs` folders into the existing Quant-Lab project root. It adds one new skill and one new task folder; it does not replace your instructions, the repair task, the redesign task, or the redesign's handoff.

Then, in Claude Code opened in that project, use:

```text
/fix-ifvg-redesign-review
```

Do not start it while another agent is editing the same project files.

## What's in the folder

| File | Purpose |
|---|---|
| `TASK.md` | The assignment: fixes F1–F11, boundaries, verification, handoff |
| `SOURCE_OBSERVATIONS.md` | What the review saw for each fix and where, with evidence |
| `evidence/` | 12 crops from the redesign handoff screenshots showing each visible problem |
| `TASKS.md` | Progress ledger the agent maintains |

If this folder already contains progress, keep it rather than overwriting it with the starting checklist.

## Package checks and limits

The package builder checked relative paths, skill frontmatter, file references, image decoding and file digests. It did not run Claude Code, operate your dashboard, or edit your repositories.

The skill uses the official project-skill format with manual invocation (`disable-model-invocation: true`): https://code.claude.com/docs/en/skills
