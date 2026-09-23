"""Checksum-bound, coarse vendor warnings; never a completeness declaration."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from ..features.mbp1_coverage_evidence import Mbp1DatasetConditionRecord

_SEVERITY = {"available": 0, "pending": 1, "degraded": 2, "missing": 3}


def load_archived_dataset_conditions(repo_root, physical_dates):
    """Read metadata only from preserved GLBX packages, restricted to scope.

    A vendor dataset warning applies at dataset/UTC-date granularity, regardless
    of any narrower local instrument receipt. Multiple archives cannot silently
    clear a warning: retain the worst condition until recovery is separately
    established. The condition document and query metadata must match their
    preserved vendor manifest. This verifies local package integrity, not vendor
    authentication or DBN conversion, and never opens a DBN member.
    """
    requested = frozenset(physical_dates)
    selected, warnings = {}, []
    for path in sorted((Path(repo_root) / "data/databento").glob("GLBX-*.zip")):
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if not {"manifest.json", "metadata.json", "condition.json"} <= names:
                warnings.append(f"Vendor condition metadata unavailable in {path.name}")
                continue
            raw_metadata = archive.read("metadata.json")
            metadata = json.loads(raw_metadata)
            query = metadata.get("query", {})
            if query.get("dataset") != "GLBX.MDP3" or query.get("schema") != "mbp-1":
                continue
            manifest = json.loads(archive.read("manifest.json"))
            declared = {item["filename"]: item for item in manifest["files"]}
            raw_condition = archive.read("condition.json")
            for name, content in (
                ("metadata.json", raw_metadata),
                ("condition.json", raw_condition),
            ):
                actual = hashlib.sha256(content).hexdigest()
                item = declared.get(name, {})
                if item.get("hash") != f"sha256:{actual}" or item.get("size") != len(content):
                    raise ValueError(f"vendor metadata checksum/size mismatch: {path.name}/{name}")
            condition_sha = hashlib.sha256(raw_condition).hexdigest()
            seen = set()
            for row in json.loads(raw_condition):
                day = row["date"]
                if day not in requested:
                    continue
                if day in seen:
                    raise ValueError(f"duplicate vendor condition date in {path.name}: {day}")
                seen.add(day)
                record = Mbp1DatasetConditionRecord(
                    dataset="GLBX.MDP3",
                    utc_date=day,
                    condition=row["condition"],
                    source_document_sha256=condition_sha,
                    recorded_at=row["last_modified_date"],
                )
                prior = selected.get(day)
                if prior is None or (_SEVERITY[record.condition], record.source_document_sha256) > (
                    _SEVERITY[prior.condition],
                    prior.source_document_sha256,
                ):
                    selected[day] = record
    for day, record in sorted(selected.items()):
        if record.condition != "available":
            warnings.append(
                f"MBP-1 vendor dataset condition {record.condition}: {day}; "
                "affected coverage remains unknown, including with a positive local receipt "
                f"(condition SHA256 {record.source_document_sha256})"
            )
    return selected, warnings
