# IFVG Lab — theme fix report

September 25, 2026. This answers your report after opening the application: the side navigation was gone and could not be reopened, text read white on white in many places, and dark mode had disappeared. Before-and-after screenshots are in `screenshots/` with the prefix `T1_`. Decisions FX17–FX19 are appended to `docs/ifvg-dashboard-redesign/handoff/DECISIONS_LOG.md`, and the dark palette is documented in `docs/ifvg-dashboard-redesign/DESIGN_SYSTEM.md`.

- No study was launched, re-run, resumed or extended.
- No strategy, engine, account, cost, calendar or payout rule changed.
- No saved record was written. Every browser check ran on the redesign's isolated copy of the stores, and no save, approve or run button was pressed.

## What was wrong

Your application follows the Windows theme (Settings → Choose app theme → "Use system setting"), so it was drawing the framework's **dark** theme. The redesigned screens, however, forced the **light** palette: white cards, white inputs, a light page. The framework kept its dark-theme text (near white) on the widgets it draws itself, so:

1. **Selects, inputs and framework text were white on white.** The search box, the Sort and Status selects, every picker on Trade review, the saved-settings table on the read-only setup page, and more.
2. **The rail could not be reopened.** The sidebar collapses when its "«" button is clicked or when the window is narrow. The header button that reopens it uses the framework's dark-theme icon color, a light gray, which was invisible on the forced light page. The menu button had the same problem.
3. **Dark mode looked gone.** The theme setting was still there, but the IFVG Lab ignored it.

Every earlier browser check had started the test browser with the light theme forced, so none of this was seen. That was the gap, and it is closed below.

I reproduced all three on the isolated copy with the dark theme before changing anything (`T1_before_dark_*.png`).

## What changed

**One palette, two versions.** Every color name in the design system now has a light value (the mocks' palette, unchanged) and a dark value. The screens never write a color anymore: stylesheets and inline styles name the color (`var(--lab-panel)`, `var(--lab-ink)`, …) and chart code reads the active palette when it builds the figure. About 320 hard-coded color values across 15 screen and presentation files were replaced this way; a test now fails if any screen file carries a color value again.

**The page follows the framework's theme.** The shell reads the theme the framework reports with each run, and a small probe in the page confirms it from what is actually drawn, marks the page so the variables switch at once, and asks for one rerun when the theme changes so the charts follow. Choosing Light or Dark in Settings switches the whole IFVG Lab without a reload; "Use system setting" follows Windows.

**The rail.** Both applications now open with the rail expanded. If you collapse it, the reopen button at the top left is drawn as a dark pill with a light "»" on both themes, and the menu button takes the text color of the page.

**Design system.** The dark values are listed in `DESIGN_SYSTEM.md` under "Dark theme". Two chart colors that were literals (a chart's zero or midnight line; one resampled path and its legend) are now named tokens in both palettes.

## Verified in the running application

Three scenarios on the isolated copy, both applications, 28 screens each: the framework's dark theme as a custom base, the system setting with Windows dark (your case), and the light theme. Each screen was captured at 1,440 pixels and every visible text run was measured against its background (WCAG contrast).

| Scenario | Screens | Screens with no text under 4.5:1 | What remained |
|---|---|---|---|
| System dark | 28 | 23 | disabled buttons at 4.15:1 (since raised to 5.2:1); the framework's own blue info box on the read-only setup page at 4.09:1 |
| Custom dark base | 28 | 23 | the same items |
| Light | 28 | 22 | disabled buttons at 3.95:1 (a value the mocks implied; since raised to 4.7:1); the framework's own warning and error boxes on the earlier study pages at 4.1–4.3:1 |

Screens covered: My studies (all five tabs), the collapsed rail, New study and the earlier study-type chooser, an earlier strategy study page, a model study page, a continued draft, Trade review (both applications, full history and point in time), Funded results (both firms, Details open), all six detail tabs, New funded comparison (blank and read-only draft), Review and approve, and the Settings dialog.

**Theme switch, end to end.** With real mouse clicks through Settings → Choose app theme: My studies light → Dark, Funded results dark → Light, Trade review dark → Light (the candle charts' background went from the dark chart ground to the light one and the pickers from dark panels to white), and Risk and simulation light → Dark (the fan and payout-race charts re-drew on the dark palette). The framework stored each choice, so a reload keeps it. `T1_switch_review_*.png` shows Trade review before and after.

**Tests.** Lint is clean. The lab, workspace, dedicated-application and trade-review suites pass: 414 passed, 1 skipped (the half-exit-engine test that needs a checkout this computer does not have). New tests: `tests/agents/ifvg_lab/test_theme.py` (36 tests: both palettes complete, the stylesheet carries both, the palette and charts follow the theme, the shell starts on the reported theme and keeps the probe's answer, no screen file hard-codes a color) plus one to four tests per converted module. **Full suite on the final code**, run last (started September 25, 2026, 1:47 AM Chicago; 46 minutes 37 seconds): lint clean; **3,846 passed, 5 failed, 3 skipped**. The five failures are exactly the five pre-existing ones the repair and fix tasks recorded (the frozen CatBoost bundle, the HTF cap experiment, and the three capture-scheme default-tag tests); the three skips need checkouts this computer does not have. Log: the internal evidence folder's `test_receipts/full_suite.log`.

## Changes that are more than a color swap (for your information)

- Charts on the Developer replay tab now sit on the design's chart ground with the design's grid and text colors on both themes. Their zone colors and the good/bad green and red are unchanged.
- Two internal shapes changed names to stop cached objects from holding color values: the payouts timeline part carries a palette key instead of a color, and the replay chart's level style table holds the key "muted" for its grays. No other code read them.
- The alert icon on New funded comparison is drawn with a CSS mask filled from the palette instead of a fixed-color image, so it follows the theme.

## Limits

- The framework's own notice boxes (info, warning, error) keep the framework's colors; on the dark theme one info box measures 4.09:1, just under the 4.5:1 target, and remains readable.
- The Developer replay charts and the earlier verifier charts reachable from Trade review's "Verified context" and "Setups not taken" keep their own chart styling; they are readable on both themes but not restyled.
- The other workspaces of the main application (ML Training, Dashboard Compatibility, Strategy Analysis) carry no rail and keep their own look, as decided in FX9.

## Files changed

- Shell and tokens: `src/alpha_lab/agents/data_infra/ifvg/presentation/lab/theme.py`, `.../lab/html.py`, `scripts/ifvg_lab_ui.py`, `scripts/dashboard.py`, `scripts/ifsm_research_ui.py`.
- Screens: `scripts/ifvg_lab_library.py`, `ifvg_lab_new_funded.py`, `ifvg_lab_trade_review.py`, `ifvg_lab_funded.py`, `ifvg_lab_charts.py`, `ifvg_lab_detail_{summary,payouts,risk,trades,market,settings}.py`, `.../lab/review_chart.py`.
- Tests: new `tests/agents/ifvg_lab/test_theme.py`; additions in `test_library.py`, `test_new_funded.py`, `test_trade_review.py`, `test_detail_risk.py`, `test_detail_payouts_settings.py`, `test_detail_trades.py`, `test_detail_market.py`, `tests/agents/test_ifvg_lab_tab.py`.
- Documents: `DESIGN_SYSTEM.md` (dark theme), `DECISIONS_LOG.md` (FX17–FX19), `ARCHITECTURE.md`, this folder's `TASKS.md` and `FIX_REPORT.md` pointer.
- Internal evidence (never in the repository): `../Claude-Quant-Lab-Research-Artifacts/ifvg-theme-fix-20260925/` holds every capture and contrast scan for the three scenarios, the switch recordings, the capture and scan scripts, the conversion contract, the conversion workflow's journal and the full-suite log.
