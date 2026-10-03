# Design system

Taken directly from the mock source files in `mocks/source/`. When a value here and a mock disagree, the mock source wins.

## Color

| Token | Hex | Use |
|---|---|---|
| Ground | #F4F2EC | Page background |
| Panel | #FFFFFF | Cards, tables, charts |
| Soft panel | #F7F5F0 | Stat boxes and placeholders inside cards |
| Chart ground | #FBFAF7 | Plot areas |
| Header row | #EFECE4 | Table headers |
| Rule | #DAD6CC | Card borders |
| Light rule | #E8E4DA, #F0EDE6 | Row dividers, grid lines |
| Control border | #BDB8AC | Inputs, secondary buttons |
| Ink | #16181B | Text, primary buttons, left rail, "what actually happened" lines |
| Body text | #2A2D31, #3C4046 | Paragraphs, secondary text |
| Muted text | #54585E | Labels, captions (keeps 4.5:1 contrast on the ground) |
| Blue | #1D4E89 | Good, primary data, paid, winners, up candles |
| Blue dark | #143A67 | Text on light blue |
| Blue light | #E4ECF6, #DCE6F2, #A9C0DC, #6F8FB5 | Selected chips, bands, zones, secondary lines |
| Orange | #A34A12 | Risk, adverse, died, losers, down candles, limits |
| Orange dark | #6E300B, #4F2308 | Text on light orange |
| Orange light | #F8E9DD, #F6E4D6, #EBC5A8 | Warnings, risk shading, falling conditions |
| Left rail | #16181B, active item #2B2E33, text #E9E6DE, note #B3AFA6 | Navigation |

Blue and orange were chosen so they stay distinguishable for red-green color blindness. Never carry meaning by color alone: pair it with a word (Pass, Watch, High) or a position.

### Dark theme (added September 25, 2026)

The screens follow the application's own theme: Settings → Choose app theme, or the system setting. The table above is the light palette. On the dark theme every token keeps its name and its role and takes the value below, so a screen written against the tokens reads on both. The rail stays dark on both themes. The source of truth is `presentation/lab/theme.py` (`COLORS` and `DARK_COLORS`); screens never write a color value, only `var(--lab-<token>)` in CSS and inline styles, or `palette()[...]` when a Plotly figure needs a real value.

| Token | Dark | Note |
|---|---|---|
| Ground | #141619 | Page background |
| Panel | #1E2126 | Cards, tables, charts, inputs |
| Soft panel | #23272D | Stat boxes, option rows |
| Chart ground | #1A1D22 | Plot areas |
| Header row | #272B31 | Table headers, neutral badges |
| Rule / Light rule / Grid | #363B42 / #2C3137 / #262A30 | Borders, dividers, grid lines |
| Control border | #50565E | Inputs, secondary buttons |
| Ink | #EDEBE5 | Text, primary buttons, chosen switch; text on an ink fill is #16181B ("on ink") |
| Body text | #D8D5CE, #C3C0B8 | Paragraphs, secondary text |
| Muted text | #A19E96 | Labels, captions (6.6:1 on a panel) |
| Blue | #8DB2E2 | Good, primary data, links (7.5:1 on a panel) |
| Blue dark | #BBD2EF | Text on the blue tint |
| Blue light | #1F2E44, #253652, #33507A, #4E729F | Tint, bands, zones, secondary lines |
| Orange | #E4894C | Risk, adverse, limits (6.4:1 on a panel) |
| Orange dark | #F1BA8F, #F7D0AF | Text on the orange tint |
| Orange light | #3A2517, #43291A, #7A4A2A | Warning tint, risk shading, alert border |
| Leader row / Risk row | #1F2733 / #2B2219 | Tinted table rows |
| Left rail | #0B0C0E, active item #23262B, text #E9E6DE, note #9F9C93 | Navigation |

Two chart tokens exist in both palettes: **zero line** (#8A857A light, #80848B dark) for a chart's zero or midnight line, and **sample line** (#7C8797 light, #7F8A9A dark) for one resampled path and its legend.

## Type

- Headings and titles: Newsreader (serif), 600 weight — page titles 34–40 px, section titles 22–26 px.
- Body and labels: IBM Plex Sans — body 14–16 px, captions 12–13 px, overline labels 13 px uppercase with 0.6 px letter spacing.
- Numbers, prices, money and codes: IBM Plex Mono.
- Fallbacks: Georgia for the serif, the system sans for Plex Sans, the system monospace for Plex Mono. If the app can't load web fonts, use the fallbacks; don't substitute another display face.

## Components

- **Card:** white, 1 px #DAD6CC border, 12 px corners, 22–26 px padding, serif title.
- **Tile:** card with a 13 px muted label above a large mono number and an optional caption.
- **Status line:** one row, check icon, one sentence, "Details" button that expands.
- **Segmented switch:** two to four joined buttons, selected is ink with white text, 40–44 px tall, text never wraps.
- **Tabs:** text tabs with a 3 px ink underline on the selected one.
- **Chip:** light blue fill, blue border, full value text, remove button with an accessible name; "+ Add" is a dashed chip.
- **Badge:** pill; blue light for pass and holds, orange light for watch, high, medium and new; neutral for info.
- **Tables:** header row #EFECE4, 13 px bold muted headers, numbers right-aligned in mono, leader or highlighted row tinted #F3F6FA or #FBF4EE.
- **Placeholder value:** muted mono text in brackets or a plain sentence ("Not in this study's export").
- **Alert:** orange-light box with icon, a bold first sentence, then one explanatory sentence.
- **Buttons:** primary ink with white text; secondary white with #BDB8AC border; minimum 40–44 px tall.

## Charts

- Plot on #FBFAF7 with light grid lines; axis labels muted, 12–13 px; axes always titled in words.
- Money axes in $ with k for thousands; time axes in Chicago 12-hour.
- Resampled fans: 5th–95th band #DCE6F2, middle half #A9C0DC, typical line blue 3 px, sample paths thin grey, actual path ink dashed. Label end values at the right edge with words (bad, typical, good).
- Limits and thresholds: solid orange line with a label on the line; area past a limit tinted orange light.
- Distributions: bars blue light, the typical bar blue, tail bars past a risk threshold orange; percentage labels on bars; vertical markers for bad, typical, good and actual.
- Candles: up blue, down orange; zones as translucent fills with labels or numbered markers plus a key; entry as an ink triangle; exits as blue dots; the result in an ink box.
- Every chart has a one-sentence caption that says what to take from it, using the actual numbers.

## Words

Plain English, spelled out. Money like $30,781.88 in tables and $30.8k on axes. Times like "April 12, 2026, 7:07 PM"; short forms "Apr 12, 7:07 PM" in tables. Settings are shown by their meaning, never their internal key.
