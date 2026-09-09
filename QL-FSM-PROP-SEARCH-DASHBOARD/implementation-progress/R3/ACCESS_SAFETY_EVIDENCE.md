# R3 — Access-safety evidence

Scope: the complete R3 change set (12 new propsim modules, 6 new/1 renewed
test suites, the 2 modified search modules, 2 modified search tests, shared-doc
lane appends). Commands run 2026-08-19 from the repo root; corroborated by the
independent safety-lens adversarial review (`ADVERSARIAL_REVIEW.md`).

## 1. Protected/sealed date literals — ZERO in the R3 change set

```
grep -rn "2026-06-11\|2026-06-1[2-9]" src/alpha_lab/propsim/ tests/propsim/ \
        tests/agents/ifvg_search/test_prop_seam.py
```

Hits: 6 — ALL inside `tests/propsim/test_loaders.py` (+ its pycache), a
**pre-existing, committed, July PROP-SIM-window file outside the R3 change
set** (synthetic journal fixture rows stamped `2026-06-16`, written only under
`tmp_path`; no real store path is constructed there). Every R3-authored file:
**zero** hits for `2026-06-11` and the sealed range.

The only date-like strings in R3 files are: synthetic fixture trading days in
the `2026-01-*` family (inherited R1/R2 convention), and three
`2026-08-18` **metadata timestamps** in `test_calendar_and_evidence.py`
(synthetic source-document `retrieved_at_utc`/`approved_at`/`effective_from`
fields for the evidence-compiler fixture — document metadata, not market-data
access).

## 2. No real data-store paths, no listing, tmp-only writes

```
grep -rn "data/ifvg_datasets\|data\\ifvg" tests/propsim/*.py \
        tests/agents/ifvg_search/test_prop_seam.py        → 0 hits
```

Every R3 test writes exclusively under pytest `tmp_path`. No R3 module or
test constructs, lists, stats, opens, or reads a real data-store directory.

## 3. No network access, no scraped prop-firm rules

```
grep -rn "requests\.\|urllib\|http://\|https://\|socket\." <12 new modules>  → 0 hits
```

Contract evidence is synthetic only (`contract_evidence.py` compiles
hand-built synthetic source documents; the status ladder makes
`first_party_verified` unreachable from synthetic evidence — test-enforced).
No live prop-firm rule was fetched, scraped, or embedded.

## 4. Repository boundaries

- Strategy-Core: **no file modified** (v1 hard constraint) — R3 touches only
  Quant-Lab paths.
- Trade-Lab: **no file modified**; propsim imports nothing from Trade-Lab
  (the July `loaders.py` evidence-mode reader is untouched).
- M0–M3 lane modules: untouched; full-repo suite green (see
  `TEST_RESULTS.md`).
- Existing immutable artifacts/catalogs: no R3 code path opens them; all
  simulation/store operations in tests run on synthetic objects in tmp.

## 5. No full-development runs

No full-development replay, feature materialization, model fit, configuration
search, prop search, or bootstrap research run was executed. All R3
verification is synthetic-fixture verification (`pytest` on hand-built
streams); the prop engine performance rows run on synthetic day blocks.

## Counters

| Counter | Value |
|---|---|
| June 11 2026 path constructions/listings/stats/opens/reads | 0 |
| Sealed-range (≥2026-06-12) constructions/… in the R3 change set | 0 (6 pre-existing committed synthetic-fixture literals in July `test_loaders.py`, tmp-only, outside this release) |
| Real data-store files created/modified | 0 |
| Network fetches | 0 |
| Full-development pipeline invocations | 0 |
