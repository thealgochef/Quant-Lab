# Saved approval for a strategy search

Updated: 2026-09-08.

An owner can approve a saved strategy-only search without starting it. The
research store must first have an explicitly initialized research namespace.
The initialization instance is recorded before initialization so an interrupted
setup can retry the identical request.

`search/strategy_approval.py` persists a `StrategySearchApprovalEnvelope` in
`strategy_search_approvals`. It records the owner's statement and reviewed
references, effective time, namespace, requirement set, full resolved charter
content excluding authorization, and the cache's creation date allowlist.
This is local evidence of an explicit owner decision, not cryptographic identity
verification or evidence that a backtest or baseline verification passed.

The approval covers decisions 1, 2, 7 and R-2 for that exact strategy-only search.
It includes the unbounded baseline and each challenger, thresholds, cost policy,
dates and warmup, seed, registry version and source commits. Provider lookup
matches the exact resolved content before assembling the existing owner bundle.
A changed study needs matching approval; a rename does not change its content.
Approval records are immutable and catalogued. Existing regime approvals and the
supersession chain retain their separate contracts.

The wizard's **Run study** button uses current persisted readiness. Saving or
loading approval does not freeze a charter, create a worker, or replay data.
An explicit Run freezes the charter and dispatches the registered
`search_strategy_development_v1` worker. Other full-development workflows retain
their existing execution restrictions.

The worker rechecks the exact approval, namespace and current authority chain
before accessing inputs. It loads trusted cached day artifacts for the selected
replay dates, preserves the fixed ten-day warmup, resolves every configuration
independently and constructs content-addressed replay identities. The read-only
provenance adapter checks the caches against their creation allowlist while its
inner policy confines actual reads to the selected dates. It does not rebuild
caches or widen source authorization. Missing or untrusted cache evidence refuses
execution.

The worker saves its startup checkpoint before resolving the input identities,
so a long cache check is visible in progress. An input-verification exception
saves a failed checkpoint and cannot reach a child replay.

Each child uses the existing sequential replay and dual-drive neutrality check.
The existing companion publisher accepts explicit dates/warmup for this path;
the verification wrapper retains zero-warmup behavior. Results use the existing
costed metrics, gates, immutable stores, exact membership and safe-boundary
cancellation/resume. New core publications include a manifest-verified
`artifact_reference.json` sidecar pointing to their v2 evidence.

The saved parent-staleness study was approved for January 13–June 10, 2026:
107 research days, ten warmup dates, baseline plus 240/360/480, and seed 7.
The owner explicitly reserved launching it for themselves. Browser evidence
confirmed Run enabled on Review; no launch was clicked. A metadata-only preflight
checked all 117 dates with no missing cache, provenance or entering-seed issues.
This does not establish the outcome or acceptance of a real verification run.

Regression coverage includes exact-scope matching, immutable roundtrips, changed
settings, missing/corrupt/copied/not-yet-effective approval, no execution during
factory construction, four distinct fixture children, and reuse on resume. See
`tests/agents/ifvg_search/test_strategy_approval.py` and the browser report at
`reports/ifvg_browser_acceptance/STRATEGY_APPROVAL_REPORT.md`.
