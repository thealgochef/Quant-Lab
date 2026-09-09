# Frontend Retention Verification — Original Detailed Plan to Final Authority Set

**Status:** passed  
**Purpose:** prove that the frontend and user-experience detail from the original codebase-specific plan and planning brief V4 §10.1–§10.30 remains implementation authority after the backend amendment sequence.

## Sources compared

- original detailed frontend section in the first implementation plan;
- planning brief V4 frontend requirements §10.1–§10.30;
- final `FRONTEND_UX_CONTRACT.md`;
- final `IMPLEMENTATION_PLAN.md` §4;
- final `PHASED_DELIVERY.md` R4/R5/R5B/R6/hardening;
- final `TEST_MATRIX.md` §3.11;
- final implementation kickoff prompt.

## Result

**No original frontend requirement was intentionally dropped.** The final authority set now retains the full detail in `FRONTEND_UX_CONTRACT.md`, rather than relying on the compressed three-paragraph summary that existed before the V5 frontend-authority restoration patch.

## Retention matrix

| Original design area | Final authoritative location | Result |
|---|---|---|
| Pure Streamlit + Plotly; no React revival | FUX §§3–4 | retained |
| IFVG Lab top-level tabs and Experiments radio sub-navigation | FUX §3; FUX-IA-* | retained |
| Existing M0–M3 Context Research delegation | FUX §§3.2, 36 | retained |
| Focused UI modules, thin scripts, pure `src` presentation logic | FUX §4; IMPLEMENTATION_PLAN §4.2 | retained |
| Streamlit capability fallbacks | FUX §4.1; FUX-A11Y-004 | restored explicitly |
| Five study modes | FUX §6; FUX-WIZ-004 | retained |
| Eight-step objective-first wizard | FUX §§7–15; FUX-WIZ-* | retained in full |
| Progress, breadcrumb, Next/Back validation | FUX §7 | retained |
| Disk drafts, autosave, Save Draft, restore, History | FUX §§7, 29 | retained |
| Clone as New Search and immutable original | FUX §§7, 15, 29 | retained |
| Four research questions and six templates | FUX §8 | retained |
| Human baseline card and copyable technical identity | FUX §9 | retained |
| Market-meaning axis groups and complete card contents | FUX §10 | retained |
| Locked/blocked axes have no widgets | FUX §10.3 | retained |
| Analysis Filter vs Requires New Sequential Replay | FUX §10.4 | retained |
| Prop contract cards and verification status | FUX §11 | retained and updated to final contract-status ladder |
| Risk policy templates and universal/per-firm hierarchy | FUX §12 | retained and expanded to complete account policy sets |
| Strategy/prop/robustness benchmark groups | FUX §13 | retained |
| Validation boundary fields and verification/full scopes | FUX §14 | retained and updated to `VerificationAuthorizationRef` |
| Combination preview and replay-vs-prop cost distinction | FUX §15 | retained |
| Exact freeze/launch and typed full-scope confirmation | FUX §15 | retained |
| Active Runs polling, checklist, keyboard funnel, child table | FUX §16 | retained in full |
| Safe cancel and manual/CLI fallbacks | FUX §§16.6–16.7 | retained |
| Results disclosure, scope, and gross/cost/net labels | FUX §§5, 17 | retained |
| Overview questions/cards and exact no-pass copy | FUX §18 | retained; development title updated intentionally |
| Frontier axes and accessible selectbox twin | FUX §19 | retained |
| Sensitivity heatmap metrics and glyph classes | FUX §20 | retained |
| Firm matrix, survival curves, payout distributions | FUX §§21–23 | retained |
| Configuration explorer presets, sticky columns, pagination | FUX §24 | retained |
| Human baseline-diff names | FUX §24 | retained |
| Dimension ribbon and four comparison panels | FUX §25 | retained and updated with `match_basis` |
| Exact affected setup/trade/event IDs | FUX §§25, 27 | retained |
| Seven deterministic insight categories | FUX §26 | retained |
| Account timeline lines, shading, markers, linked events | FUX §28 | retained |
| Full Pipeline Configure/Preview/Launch/Monitor/Retry/Publish | FUX §30; FUX-PIPE-* | retained in full |
| `prepared_not_published` and verify-then-activate | FUX §30.6 | retained |
| Status vocabulary and forbidden wording | FUX §§5.3–5.4 | retained with truthful development label |
| Intentional empty/blocked/failure states | FUX §31; FUX-STATE-001 | retained and expanded |
| Keyboard, labels, contrast, responsive tables/charts | FUX §32 | retained in full |
| Required 1440×900, 1024×768, 768×1024, 390×844 QA | FUX §32.2; FUX-A11Y-003 | restored explicitly |
| Screenshot and keyboard evidence gate | FUX §32.4; hardening | restored explicitly |
| Indexed pagination and honest chart truncation | FUX §33 | retained |
| Sanitized failures and no protected/live controls | FUX §34 | retained |

## Intentional terminology updates—not losses

The following source wording was changed only to conform to the final contracts:

| Earlier wording | Final authoritative wording |
|---|---|
| Robust Representative / Selected Representative as a development title | Development Exploratory Representative |
| Exact historical 1m intrabar path | Historical 1m scenario / approximation |
| Native-ID commonality across changed profiles | Exact lineage plus displayed `match_basis`; unsupported comparison disabled |
| One universal authorization checklist | Computation-path-scoped authorization and `VerificationAuthorizationRef` for the real slice |
| MBP-1 feature availability in the pipeline release | R5 readiness; R5B versioned offline/research-only activation |
| Spectral/Nyström in initial release | planned/disabled in V1; post-V1 numerical implementation |

## Implementation authority chain

The implementation agent must use:

```text
README.md
  -> IMPLEMENTATION_PLAN.md
  -> FRONTEND_UX_CONTRACT.md
  -> PHASED_DELIVERY.md
  -> TEST_MATRIX.md
  -> supporting backend/ML/owner/audit documents
```

`FSM-PLAN-DOCUMENT.md` remains provenance-only. The original detailed plan is evidence for this retention verification but is not needed during implementation because its frontend requirements are now fully restated in the current authority set.

## Final conclusion

The corrected package is complete for implementation planning with respect to frontend and user experience. Acceptance still depends on the code satisfying the `FUX-*` gates, including interactive browser, keyboard, responsive, and screenshot evidence.
