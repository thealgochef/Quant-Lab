# Documentation basis and package scope

Checked September 22, 2026. These are design choices for this task, not measured claims that one prompting format is optimal.

- The supplied developer guide recommends a complete assignment, an explicit finish line, purposeful stop/continue instructions, a file-backed checklist, bounded delegation with evidence checking, concrete design exclusions and a review. TASK.md, TASKS.md, the scoped rule, SCREEN_BRIEF.md and the read-only reviewer implement those ideas. The optional suggestion to treat prior answers as permanently settled is deliberately omitted: financial evidence can reveal a previous mistake.
  https://claude.dev/blog/getting-the-most-out-of-opus-5-5/
- The supplied platform guide recommends starting at medium effort, disclosing verification gaps and checking completion rather than interpreting a text progress report as success. This package uses an interactive client, not an unattended loop; no indefinite automatic continuation or raw model-request token settings are added.
  https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5
- Documented launcher features used here: initial prompt, explicit model/effort, default permission mode, and optional named agent. No default system prompt is replaced, no installation is automated, and no broad tool grant is bundled.
  https://code.claude.com/docs/en/cli-reference
- The model is explicitly pinned to claude-opus-5-5. The documented minimum client is 2.1.280; the launcher checks the installed version but cannot prove entitlement or guard against a provider-controlled model substitution. The implementation agent should disclose any actual model change rather than claim it stayed on the selected model.
  https://code.claude.com/docs/en/model-config
- Reusable manual skills stay short and point to the complete saved task. Their disable-model-invocation setting keeps them from automatically starting unrelated implementation work. The launcher instead provides an explicit owner-initiated request to read TASK.md. It does not depend on slash-command expansion.
  https://code.claude.com/docs/en/skills
- Path-scoped rules add working guidance without replacing CLAUDE.md, AGENTS.md or settings. Large specification files are read for this task, not imported on every project startup. Additional-repository instructions load only after access is granted.
  https://code.claude.com/docs/en/memory
- The final helper has only Read, Grep and Glob. Its tool definition prevents file edits/shell execution; actual tests remain the implementation agent's responsibility.
  https://code.claude.com/docs/en/sub-agents

## Preservation

SPEC.md and CODE_MAP.md are unchanged copies of the prior package's Version 3 requirements and historical inspection map. Their digests are in PACKAGE_MANIFEST.json. No new business choice is implied by this package, and no current local repository was inspected or modified to create it. Startup files use a new task-directory name to avoid overwriting the prior kit's owner decisions or progress; the agent must reconcile those records when present.

## What was checked here

The archive is checked for valid paths, expected file inventory, readable text/frontmatter, required links, source-copy hashes, and excluded existing-settings filenames. Windows batch syntax receives static checks. No Windows execution, authenticated Claude session, repository implementation, application tests or financial replay is claimed by building this package.

The future completed-run review folder is a different artifact. It retains the specification's documents-and-data-only requirement; these development launchers, skills and source instructions must not be copied into that business review folder.
