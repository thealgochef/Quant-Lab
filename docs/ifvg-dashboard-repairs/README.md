# Start the repair-only task

Merge this ZIP's `.claude` and `docs` folders into the existing Quant-Lab project root. It adds a new skill and a new task folder; it does not replace your existing instructions, settings, skills, or funded-task documents.

In Claude Code opened in that project, use:

```text
/fix-ifvg-dashboard
```

Do not start it while another agent is editing the same project files. Finish or coordinate that work first. No `.cmd` file, model change, permission bypass, or long prompt paste is required.

The complete prompt is `TASK.md`. `SOURCE_OBSERVATIONS.md` and eight actual prior screenshots supply its evidence. `TASKS.md` is maintained by the agent. Existing economic runs must not be repeated or modified merely to verify these repairs. Stop before the larger redesign or new analytics.

If this exact repair folder already contains progress, preserve it rather than overwrite it with the initial checklist.

## Package checks and limits

The delivery builder checks relative paths, skill frontmatter, file references, screenshot decoding, and payload digests. These checks do not execute Claude Code, operate your dashboard, edit your repositories, or validate any application repair.

The skill follows the official project-skill format, with manual invocation (`disable-model-invocation: true`), checked September 23, 2026:
https://code.claude.com/docs/en/skills
