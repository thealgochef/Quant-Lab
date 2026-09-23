# Optional IFSM research engine

This directory contains a 71 KB incremental Git bundle and a 16 KB source
manifest. The bundle adds six commits to the published Strategy-Core prerequisite
`a4e3303179ac6a1088aecaaa3482934cf1aec4d7`; it contains no complete repository,
market data, saved studies or dependencies. Its final commit is the original
`38825ed86f3e3940515cdc28d9df567ddccc0b70`, without rewriting research history.

From Quant-Lab, after its ordinary dependency installation:

```powershell
python scripts/prepare_ifsm_research_core.py
python scripts/run_ifsm_research_ui.py --check
python scripts/run_ifsm_research_ui.py
```

Git and network access to the public prerequisite commit are required for the
first preparation. The script creates an external checkout under sibling
`Claude-Quant-Lab-Research-Artifacts/ifsm-research-core/<commit>/`. It checks the
bundle checksum, prerequisite, exact final commit, every scoped source byte and
the original source identity. Existing destinations are verified without being
modified. If an interrupted preparation left an incomplete destination, use a
new explicit destination; no existing evidence is automatically deleted.

For a custom external location, pass `--destination PATH` to preparation and
`--core PATH` to the launcher, or set `IFSM_RESEARCH_CORE=PATH`. The launcher also
retains the previous sibling `Strategy-Core-daily-close` as a fallback when the
default external checkout is absent. No package installation or live pin changes.
Historical study identities and archives remain unchanged; the cleanup move of
older working directories is recorded separately in the project's relocation index.

The source manifest preserves the original Windows line endings, including one
mixed-ending source file. This is necessary because research identities hash
exact working bytes. Preparation reproduces those same bytes on Linux too; it
does not define a new identity or reinterpret earlier research.

Review all changes using standard Git commands in the prepared external
checkout, without unpacking source into Quant-Lab:

```text
git log --reverse a4e3303179ac6a1088aecaaa3482934cf1aec4d7..38825ed86f3e3940515cdc28d9df567ddccc0b70
git diff a4e3303179ac6a1088aecaaa3482934cf1aec4d7 38825ed86f3e3940515cdc28d9df567ddccc0b70
```

CI retains the ordinary installed-Core job and adds an independent research job
that prepares this engine, verifies the launcher and runs the focused research
contracts. `IFSM_REQUIRE_RESEARCH_CORE=1` turns missing research capabilities
into a test-session error; that job cannot silently pass by skipping them.
