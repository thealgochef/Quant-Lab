# Current Strategy-Core setup

## One Core pin for Quant-Lab (2026-10-03)

Ordinary Quant-Lab and the IFSM research screen use the same exact Strategy-Core
commit: `b062bfcf5a5209440a4b4c9d7c0ca2263f9f4cc2`. Task A1 authorizes this re-pin
from `7c7111e398c083cf8e966e2e0c5aac8a41cc12c0`. `pyproject.toml` pins the installed dependency and
`current.json` records that same commit and its verified source identity.
The current identity is
`afa16bed5733f6d41f090bc8b8e09215464cf9a009ec1c1bc7236973b76ead01`, using Git's
canonical LF source bytes consistently on Windows and Linux.
Package metadata remains `0.1.0`; the exact commit and source bytes establish
the version pairing.

This commit is published on Strategy-Core branch `feature/menthorq-level-context`.
It adds the optional runtime MenthorQ snapshot and entry gates described in
[Task A1's contract](../../docs/IFSM_MENTHORQ_LEVEL_CONTEXT.md). The default-off
10-date replay preserves all eight v2 table bytes and the historical profile hash;
context content matches after the documented source identity exclusions. New work in either screen can use its gap invalidation choices,
daily-close policy and corrected Chicago schedules. Older configurations retain
their original missing-policy defaults. Execution corrections and newer state
schemas can change replay results and prevent reuse of an older checkpoint.

## Install and prepare the current engine

From Quant-Lab:

```powershell
python -m pip install -e ".[dev]"
python -m pip install --force-reinstall --no-deps "strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@b062bfcf5a5209440a4b4c9d7c0ca2263f9f4cc2"
python scripts/prepare_ifsm_research_core.py
python scripts/run_ifsm_research_ui.py --check
python scripts/run_ifsm_research_ui.py
```

The explicit Core reinstall updates an existing installation even when its
package metadata still says `0.1.0`.

Preparation fetches the exact public commit in `current.json`, then verifies its
source files and identity. Git and network access are required for the first
preparation. The checkout lives outside Quant-Lab under sibling
`Claude-Quant-Lab-Research-Artifacts/ifsm-research-core/<commit>/`. By default, the
ordinary installed package and IFSM use this same verified source for provenance. Source
parity is checked; package version `0.1.0` alone cannot satisfy that check.

Existing destinations are verified without modification. If an interrupted
preparation left an incomplete destination, use a new explicit destination;
existing evidence is not automatically deleted. For a custom external location,
pass `--destination PATH` to preparation and `--core PATH` to the IFSM launcher,
or set `IFSM_RESEARCH_CORE=PATH`. These overrides select the IFSM checkout only;
ordinary wheel installations still need the verified default managed checkout.
An editable Core installation can provide its matching source checkout directly.
The launcher does not fall back to an older sibling checkout when its current
checkout is missing.

Restart existing Python study screens and workers after upgrading so their
imports use the new pin. Preparation and `--check` launch no replay or fitting.
The normal dashboard and IFSM retain their existing study workspaces. This
upgrade does not change Trade-Lab's dependency or authorize strategy activation.

CI checks the installed dependency and the prepared source for the same current
commit, then exercises the relevant ordinary and research contracts.
`IFSM_REQUIRE_RESEARCH_CORE=1` turns missing research capabilities into a
test-session error rather than silently skipping their tests.

## Preserve historical studies

The September 22 current pin `7c7111e398c083cf8e966e2e0c5aac8a41cc12c0` and its
source identity `9247f16e223c9226f9afa17832c8810e31ce8853bfad9a0b5452e31845377bf3`
remain historical identities. Task A1 does not rewrite their saved studies.

`a4e3303179ac6a1088aecaaa3482934cf1aec4d7` was the ordinary installed pin before
this upgrade. `38825ed86f3e3940515cdc28d9df567ddccc0b70` was the isolated September 18
research pin, with source identity
`9db57357b1702b7022592980644eb20f320d19fcf0e309a4431836717d6a1eb1`. Both commits
remain in Core history. They are retained for historical reproduction, not used
as the default engine for new Quant-Lab work.

The original `manifest.json` and 71 KB `ifsm-daily-close.bundle` remain unchanged
as archival evidence. That incremental bundle adds six commits to the published
`a4e3303` prerequisite and ends at the original `38825ed` commit. It contains no
complete repository, market data, saved studies or dependencies. The original
manifest preserves exact Windows source bytes, including mixed line endings,
because historical identities include those bytes. Current preparation uses
`current.json`; it does not repurpose the archival manifest or bundle.

Never assign the current source identity to an old execution, rewrite its
approvals or resume an incompatible checkpoint. A new replay receives its own
source-bound identity and rebuilds state from its authorized input history when
required. Historical settings alone do not reproduce the original source and
state. Existing source archives and relocation records remain available.

To inspect the engine changes in a full Strategy-Core checkout (the prepared
checkout intentionally fetches only the selected commit):

```text
git log --reverse a4e3303179ac6a1088aecaaa3482934cf1aec4d7..7c7111e398c083cf8e966e2e0c5aac8a41cc12c0
git diff a4e3303179ac6a1088aecaaa3482934cf1aec4d7 7c7111e398c083cf8e966e2e0c5aac8a41cc12c0
```
