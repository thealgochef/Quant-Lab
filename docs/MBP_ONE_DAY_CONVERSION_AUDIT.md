# Bounded MBP-1 conversion audit

The 2026-09-09 audit established exact conversion equivalence for the two physical
files associated with logical day **2026-02-23**. It did not establish source
completeness, authorize a receipt, unblock R5B, or change the research scope.
The completed acceptance run and its original MBP findings remain preserved.

The reproducible command, run from the repository root, is:

```powershell
python scripts/audit_mbp1_conversion_day.py --logical-day 2026-02-23 --archive data/databento/GLBX-20260711-EEDSMFU845.zip --inventory reports/r5_r6_acceptance/20260908/mbp_evidence/partition_integrity.jsonl --output reports/b0_projection_repair/20260909/mbp_audit/verified_v1
```

Choose a fresh output directory for another verification; completed evidence is
never overwritten. The source archive and original parquet files remain read-only.
The event audit decodes only the two associated physical dates, 2026-02-22 and
2026-02-23, including their outside-session rows to establish whole-file conversion.
Both archived dataset conditions are `available`, meaning no known coarse vendor
issue, not certified completeness.

## Conversion and ordering

| Physical date | DBN rows decoded | Spread rows excluded | Original parquet rows compared |
|---|---:|---:|---:|
| 2026-02-22 | 260,364 | 204 | 260,160 |
| 2026-02-23 | 14,535,850 | 20,951 | 14,514,899 |
| Total | 14,796,214 | 21,155 | 14,775,059 |

The audit uses the installed `DBNStore.to_df` defaults (mapped symbols, UTC
nanosecond timestamps, floating dollar prices), streaming 200,000 raw rows at a
time. It applies exactly the historical importer's `symbol`-contains-`-` exclusion,
then compares every scalar in all 20 columns and every retained ordinal with the
original parquet. No sorting, deduplication or front-month filter occurs in this
conversion step. The importer actually retains all single-leg contracts.

Both files passed exact equality. Input SHA-256 values match the preserved
acceptance inventory; source size and modification time remained unchanged.
Parquet serialization bytes need not match a newly written file to prove scalar
equivalence. This reproduces a transformation compatible with the installed
importer; it does not identify the historical import process or its version.

There are 26,408 adjacent event-time reversals and 404,875 adjacent event-time
equalities across the two physical streams. There are zero receive-time reversals
and 687,618 adjacent receive-time equalities. These are identical in original DBN
conversion and parquet order; they are not a conversion sort defect or proof of
feed completeness. The normalization probes verify the pinned stable total order
`ts_event, ts_recv, sequence, source_ordinal` and retain source ordinals. Probes
contain the first bounded real records of each logical subspan, not all normalized
events. No full logical-day model features were built.

The authorized logical span is 2026-02-22 23:00 UTC through 2026-02-23 22:00 UTC,
end-exclusive. There are 14,650,411 single-leg rows in that span before instrument
selection. The dominant trade instrument is NQH6 / 42002475 in both physical
subspans; their selected counts are 202,665 and 12,920,020 (13,122,685 total).
These are event counts, not coverage fractions, feature rows, or usable folds.

## Snapshot handling and remaining uncertainty

The February 23 file contains five opening snapshot rows, all flags `168`:
`F_LAST=128`, `F_SNAPSHOT=32`, and `F_BAD_TS_RECV=8`. Their receive timestamps are
midnight, while event timestamps precede midnight. The five displayed level-00
book states (prices, quantities and order counts) equal the corresponding final
states in the preceding file. Three also share that prior record's event timestamp
and sequence. NQU6 and NQH6 do not, so strict event redundancy is **not proved**.
Detailed paired records are retained in each physical audit JSON.

The current research reader excludes these records by clipping each physical file
to its own event-time UTC boundaries before normalization. The audit leaves this
policy unchanged. A regression verifies that a midnight receive-time snapshot is
not backdated into the prior interval or counted as an extra ordinary add. The
observed identical displayed book states support a narrow boundary-state finding;
they do not establish general snapshot recovery semantics or recover omitted events.

Neither physical file has observed `F_MAYBE_BAD_BOOK` rows. That absence and the
vendor's `available` condition are not a completeness declaration. Exact expected
instrument/channel coverage, attributable gaps and recovery intervals, complete
feature windows, and a reviewed extraction declaration remain unresolved.
There is no extrapolation to the other acceptance dates and no completeness receipt.

## Coarse vendor warnings now reach coverage

`search/mbp1_vendor_conditions.py` reads only metadata from preserved matching
`GLBX-*.zip` packages, verifies the metadata and condition documents against their
vendor manifest, and restricts returned typed conditions to the requested dates.
It opens no DBN members. Multiple packages cannot clear a negative warning by
merely adding an `available` record; the worst observed condition is retained.
This is local package integrity, not remote vendor authentication.

`research_mbp1.py` now forwards the checksum-bound condition record to the verified
coverage evidence reader, exposes negative warnings even without a receipt, and
freezes condition records with preflight provenance. A positive owner-reviewed
local receipt therefore cannot silently suppress `degraded`, `pending` or `missing`
conditions. The existing coverage kernel downgrades that physical scope to
`completeness_unknown` and zero claimed coverage. Missing receipts still block R5B.

The previously documented degraded physical dates remain March 15, March 16,
April 10 and May 24. Their propagation was checked using already-scoped metadata
only, under condition document SHA-256
`5f0e3c21419d4e1519477ac98d0cabebbb7b9b3a0fa134db1c4091e59cf3e136`.
No additional event dates were decoded. These remain coarse dataset warnings,
not measured NQ-specific gap intervals.

## Evidence and checks

Generated evidence is under
`reports/b0_projection_repair/20260909/mbp_audit/`. The `verified_v1/` directory
contains the final script's run, extracted compressed source copies, per-file
comparison and paired snapshot JSON, raw/normalized ordering probes, and
`audit_summary.json` with runtime versions and input hashes. The initial method
check and its executed source snapshot remain at the parent directory; the final
verification repeats exactly the same event dates, without expanding the scope.
`vendor_warning_propagation.json` records the separate metadata-only warning check.

Focused pytest checks cover scalar drift, row reorder/loss/addition, midnight
snapshot clipping, condition checksum failure, metadata-only access, degraded
coverage despite a positive receipt, non-clearing negative evidence, bounded
loading, and existing coverage semantics. The executed checks and their results
are recorded in the generated audit report. Lint is scoped to the changed files.
