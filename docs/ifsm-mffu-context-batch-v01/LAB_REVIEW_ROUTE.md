# Reopen the external MFFU result in IFVG Lab

The completed batch stays in its task-owned Research-Artifacts store. To review
its saved 64-intent matrix and result in the existing IFVG Lab, opt into one
exact external result for the dedicated IFSM UI process. This does not change
the Lab's ordinary research and test store roots.

Set both environment variables in the PowerShell process that starts the UI:

```powershell
$env:IFSM_MFFU_REVIEW_STORE_ROOT = 'C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-mffu-context-batch-20261007\<batch-store>'
$env:IFSM_MFFU_REVIEW_RESULT_ID = '<saved-64-character-result-id>'
python scripts/run_ifsm_research_ui.py --research-core 'C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-mffu-context-batch-20261007\core' --check
python scripts/run_ifsm_research_ui.py --research-core 'C:\Users\gonza\Documents\Claude-Quant-Lab-Research-Artifacts\ifsm-mffu-context-batch-20261007\core' --port 8502
```

Open `http://127.0.0.1:8502/?view=funded&app=ifsm&result=<saved-64-character-result-id>`.
Expand **Saved 64-intent matrix and status** for the matrix screenshot. Capture
the completed-result view separately. Save both real PNG screenshots in the
task-owned working area outside the repository for the result-review publisher.

The route applies only to `app=ifsm` and the exact configured result ID. It
resolves the store path outside this repository, loads the verified saved result
and frozen MFFU plan, and checks the complete 64-row result before displaying
it. The external review target cannot offer the Lab's **Publish the review
folder again** action. Other links keep their existing default store routes.
Clear both environment variables to remove the external route; setting only
one fails closed.

The result-review publisher takes these two PNGs as `--matrix-png` and
`--result-png`, plus a reviewed findings file. Its staging root stays outside
the repository and its final ZIP goes under the ignored `reports/` directory.
