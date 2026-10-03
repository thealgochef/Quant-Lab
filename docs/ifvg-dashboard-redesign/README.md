# Start the redesign build

Merge this ZIP's `.claude` and `docs` folders into the existing Quant-Lab project root. It adds one new skill and one new task folder; it does not replace your existing instructions, settings, skills, the repair task, or funded-task documents.

Run the repair task first and let it finish. Then, in Claude Code opened in that project, use:

```text
/build-ifvg-redesign
```

Do not start it while another agent is editing the same project files.

## What's in the folder

| File | Purpose |
|---|---|
| `TASK.md` | The complete assignment: goal, boundaries, build phases, verification, handoff |
| `DECISION_RULES.md` | How the agent decides anything the mocks or task don't cover |
| `SCREENS.md` | Every screen and panel: what it shows, where the data comes from, what already exists |
| `CALCULATIONS.md` | Exact definitions for every number, with reference values to test against |
| `DESIGN_SYSTEM.md` | Colors, fonts, components, chart conventions, wording rules |
| `mocks/README.md` | Index of the mock screens and which values in them are real or placeholders |
| `mocks/images/` | 17 rendered mock screens, including interactive states |
| `mocks/source/` | The mock source files; serve the folder locally to click through them |
| `TASKS.md` | Progress ledger the agent maintains |

To click through the mocks yourself: in `mocks/source`, run `python -m http.server 8765` and open `http://localhost:8765/Main.dc.html`. They need a local server; opening the files directly won't work.

If this folder already contains progress, keep it rather than overwriting it with the starting checklist.

## Package checks and limits

The package builder checked relative paths, skill frontmatter, file references, image decoding and file digests. It did not run Claude Code, operate your dashboard, or edit your repositories.

The skill uses the official project-skill format with manual invocation (`disable-model-invocation: true`): https://code.claude.com/docs/en/skills
