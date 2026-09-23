# Saved approval for a strategy search

Updated: 2026-09-08.

An owner can approve a saved strategy-only search without starting it. The
research store must first have an explicitly initialized research namespace.
The initialization instance is recorded before initialization so an interrupted
setup can retry the identical request.

In **Configure study → Review**, the **Study approval** panel displays every
configuration (including automatic baseline combinations), research dates,
warmup, costs, target, seed, and thresholds. It checks local bars/levels metadata,
cache provenance and warmup continuity without replaying or fitting. Missing or
inconsistent evidence is shown as a specific blocker. Full input verification
still occurs in the runner.

Review those terms, enter **Reviewer name**, check **I approve this exact
strategy study**, and click **Save study approval**. This saves the existing
exact-request approval and registers it for readiness lookup; it starts no work.
Then click **Run study** when ready. The confirmation is tied to the request's
identity, so changing settings clears the confirmation and requires a matching
approval. Recording rechecks the request and metadata before saving.

This panel supports strategy configuration searches and exact
`single_configuration` studies with one registered value per selected setting
and exactly one child. Evaluate One's fixed settings do not add a baseline or
Cartesian comparisons; ordinary searches keep their existing comparison rules.
One approval covers the
axes, their values, strategy thresholds and the strategy-search workflow; it
does not add model, regime or prop-firm authority. Existing approvals and
completed results are preserved. A cloned or edited study with a changed
request cannot reuse an earlier approval merely because its name is similar.

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

The isolated IFSM UI uses the preserved research Core in its own process and
separate store. Approval and replay identify the actual imported Core checkout;
an installed package may use the sibling checkout only after source parity is
verified. Its fixed historical preparation reference is a verified metadata-only
source for the original cache creation allowlist. It never transfers the prior
study's approval, output namespace or input-reading authority.

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
