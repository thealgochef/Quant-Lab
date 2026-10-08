"""Small review exports and safe manifests; working files stay outside reports."""

from __future__ import annotations

import ast
import csv
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from .benchmark import frame_from_rows, save_json
from .runtime import sha_file


def export_rows(folder, name, rows, *, parquet=False):
    flat = [{k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
             for k, v in row.items()} for row in rows]
    path = folder / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in flat for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat)
    if parquet and flat:
        frame_from_rows(flat).to_parquet(folder / f"{name}.parquet", index=False)


def export_business(report, folder):
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "report.json", report)
    (folder / "RESULTS.md").write_text(report["summary_markdown"], encoding="utf-8")
    for ref, jobs in report["datasets"].items():
        for job, rows in jobs.items():
            export_rows(folder / "datasets", f"{ref}_{job}", rows, parquet=True)
    for name in ("fold_metrics", "pooled_metrics", "predictions", "mean_controls", "fits"):
        export_rows(folder / "predictive", name, report["benchmark"][name])
    export_rows(folder / "predictive", "paired_differences", report["paired_diagnostics"])
    export_rows(folder / "predictive", "largest_continuations", report["largest_continuations"])
    export_rows(folder, "coverage", report["coverage"])
    for name, rows in report["economics"]["economic_result"]["tables"].items():
        export_rows(folder / "funded", name, rows)
    export_rows(folder / "funded", "six_lens_measures", report["economics"]["lenses"])
    export_rows(folder / "funded", "model_coverage", report["economics"]["operation_coverage"])
    save_json(folder / "funded/reconciliation.json", report["economics"]["financial_validation"])
    save_json(folder / "scored_start_seeds.json", report["seeds"])
    for name, contract in report["contracts"].items():
        save_json(folder / "contracts" / f"{name}.json", contract)
    rows = []
    for payload in report["operations"].values():
        for name, state in payload["streams"].items():
            rows.extend({"operation": name, **row} for row in (
                *state["entry_rows"].values(), *state["checkpoint_rows"].values()))
    export_rows(folder / "funded", "actual_decisions", rows)


def write_manifest(folder, bindings):
    files = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        count = None
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                count = max(0, sum(1 for _ in csv.reader(handle)) - 1)
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq

            count = pq.read_metadata(path).num_rows
        files.append({"path": path.relative_to(folder).as_posix(), "bytes": path.stat().st_size,
                      "sha256": sha_file(path), "rows": count})
    save_json(folder / "MANIFEST.json", {"bindings": bindings, "files": files,
        "manifest_excludes_itself": True})


def zip_and_extract(folder, archive, extracted):
    if archive.exists() or extracted.exists():
        raise FileExistsError("review archives and clean extractions are never overwritten")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(folder).as_posix())
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or any(
            Path(n).is_absolute() or ".." in Path(n).parts or ":" in n for n in names
        ):
            raise PermissionError("archive has unsafe or duplicate member paths")
        z.extractall(extracted)
    manifest = json.loads((extracted / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {f["path"] for f in manifest["files"]} | {"MANIFEST.json"}
    actual = {p.relative_to(extracted).as_posix() for p in extracted.rglob("*") if p.is_file()}
    if expected != actual:
        raise PermissionError("archive member set differs from its manifest")
    for file in manifest["files"]:
        path = extracted / file["path"]
        if sha_file(path) != file["sha256"] or path.stat().st_size != file["bytes"]:
            raise PermissionError("extracted review file differs")
    return {"path": str(archive), "bytes": archive.stat().st_size,
            "sha256": sha_file(archive), "extracted_readback": "all files verified",
            "files": len(manifest["files"])}


def copy_models(benchmark_root, destination):
    benchmark = json.loads((benchmark_root / "benchmark.json").read_text(encoding="utf-8"))
    for record in benchmark["fits"]:
        if record["model_id"] is None:
            continue
        source, target = benchmark_root / record["fit_path"], destination / record["fit_path"]
        target.mkdir(parents=True, exist_ok=False)
        for name in ("fit.json", "model.cbm"):
            if (source / name).exists():
                shutil.copyfile(source / name, target / name)


def verify_columnar(folder, report):
    for ref, jobs in report["datasets"].items():
        for job, rows in jobs.items():
            frame = pd.read_parquet(folder / "datasets" / f"{ref}_{job}.parquet")
            if frame.row_id.tolist() != [r["row_id"] for r in rows]:
                raise PermissionError("export row identity differs")
            for column in ("decision_ns", "label_start_ns", "label_end_ns", "label_available_ns"):
                if frame[column].tolist() != [r[column] for r in rows]:
                    raise PermissionError("columnar timestamp precision changed")


def copy_source_closure(repo, core, destination, starting_files):
    """Static project-import closure plus package initializers; never dependency trees."""
    roots = [repo / "src", repo, repo / "scripts", core / "src"]
    pending, included = list(starting_files), set()

    def resolve(module):
        relative = Path(*module.split("."))
        for root in roots:
            for candidate in (root / relative.with_suffix(".py"), root / relative / "__init__.py"):
                if candidate.is_file():
                    return candidate.resolve()
        return None

    while pending:
        path = Path(pending.pop()).resolve()
        if path in included:
            continue
        included.add(path)
        owner = next(r for r in roots if path.is_relative_to(r))
        parts = list(path.relative_to(owner).with_suffix("").parts)
        package = parts[:-1]
        for i in range(1, len(parts)):
            init = owner.joinpath(*parts[:i], "__init__.py")
            if init.is_file():
                pending.append(init)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (package[:len(package) - node.level + 1] if node.level else [])
                module = ".".join([*base, *([node.module] if node.module else [])])
                names = [module, *[f"{module}.{alias.name}" for alias in node.names]]
            for name in names:
                dependency = resolve(name)
                if dependency:
                    pending.append(dependency)
    for path in sorted(included):
        relative = (Path("core") / path.relative_to(core) if path.is_relative_to(core)
                    else path.relative_to(repo))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return sorted(str(p) for p in included)
