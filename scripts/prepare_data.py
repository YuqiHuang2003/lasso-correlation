#!/usr/bin/env python3
"""Acquire frozen inputs explicitly, retaining byte-identical historical metadata.

This publication-only wrapper leaves the original downloaders unchanged. Importing
it never downloads anything. Run it explicitly (for example through ``make data``).
Canonical source.json records identify the historical study snapshot; actual new
acquisition times are recorded separately in ignored data/acquisition_log.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


DATASETS = {
    "news": ("docs/data_manifest_v2_raw.json", "data/raw",
             {"online_news_popularity.zip", "OnlineNewsPopularity.csv"}),
    "uji": ("docs/data_manifest_v2_benchmarks_ujiindoorloc.json",
            "data/benchmarks/ujiindoorloc",
            {"ujiindoorloc.zip", "trainingData.csv", "validationData.csv"}),
    "leukemia": ("docs/data_manifest_v2_benchmarks_leukemia.json",
                 "data/benchmarks/leukemia",
                 {"leu.bz2", "leu.t.bz2", "leukemia.libsvm"}),
}
HISTORICAL_NOTICE = (
    "Canonical source.json is copied byte-for-byte from the frozen historical "
    "manifest. Its retrieved_at_utc identifies the original study acquisition, "
    "not this new acquisition. Actual new acquisition times are recorded here."
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(root: Path, relative: str | Path) -> Path:
    """Reject every symlink component, including dangling destination links."""
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Only repository-relative paths are allowed")
    location = root
    for part in relative.parts:
        location = location / part
        if location.is_symlink():
            raise ValueError(f"Symlink paths are not allowed: {relative}")
    if not location.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes repository: {relative}")
    return location


def _basename(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError("Manifest file names must be plain basenames")
    if name in {".", "..", "source.json"}:
        raise ValueError("Manifest file names must be raw-data basenames")
    return name


def _expected_files(dataset: str, record: dict) -> dict:
    if dataset == "news":
        files = {
            "online_news_popularity.zip": {"sha256": record["archive_sha256"],
                                          "bytes": record["archive_bytes"]},
            "OnlineNewsPopularity.csv": {"sha256": record["csv_sha256"],
                                        "bytes": record["csv_bytes"]},
        }
    else:
        files = record["files"]
    if not isinstance(files, dict):
        raise ValueError("Manifest files must be a mapping")
    for name, expected in files.items():
        _basename(name)
        if (not isinstance(expected, dict)
                or not re.fullmatch(r"[0-9a-f]{64}", str(expected.get("sha256", "")))
                or type(expected.get("bytes")) is not int or expected["bytes"] < 0):
            raise ValueError(f"Invalid frozen hash or size for {name}")
    if set(files) != DATASETS[dataset][2]:
        raise ValueError(f"Frozen manifest must cover every required {dataset} raw file")
    return files


def _check_file(path: Path, expected: dict) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected a regular data file: {path.name}")
    if path.stat().st_size != expected["bytes"] or _sha(path) != expected["sha256"]:
        raise ValueError(f"Existing/downloaded file differs from frozen hash: {path.name}")


def _local_state(root: Path, plan: dict) -> tuple[list[str], bool]:
    missing = []
    folder = _safe_path(root, plan["folder"])
    if folder.exists() and not folder.is_dir():
        raise ValueError(f"Expected a data directory: {plan['folder']}")
    for name, expected in plan["files"].items():
        path = _safe_path(root, Path(plan["folder"]) / name)
        if path.exists():
            _check_file(path, expected)
        else:
            missing.append(name)
    source = _safe_path(root, Path(plan["folder"]) / "source.json")
    if source.exists() and (not source.is_file() or source.read_bytes() != plan["source_bytes"]):
        raise ValueError("Existing source.json differs from frozen historical metadata; inspect before replacing")
    return missing, not source.exists()


def _download_to_staging(root: Path, dataset: str, folder: Path) -> dict:
    """Call only original helpers, always in an empty disposable directory."""
    filename = "download_data.py" if dataset == "news" else "download_benchmarks.py"
    path = _safe_path(root, Path("scripts") / filename)
    spec = importlib.util.spec_from_file_location(f"_frozen_acquisition_{dataset}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load original downloader: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    helper = getattr(module, {"news": "download", "uji": "download_uji",
                              "leukemia": "download_leukemia"}[dataset])
    return helper(folder, force=False)


def _write_missing(root: Path, relative: Path, body_or_path: bytes | Path) -> None:
    path = _safe_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _safe_path(root, relative)
    expected = (hashlib.sha256(body_or_path).hexdigest()
                if isinstance(body_or_path, bytes) else _sha(body_or_path))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=".input-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            if isinstance(body_or_path, bytes):
                output.write(body_or_path)
            else:
                with body_or_path.open("rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if _sha(temporary) != expected:
            raise ValueError(f"Incomplete staged copy: {path.name}")
        _safe_path(root, relative)
        # Same-directory hard linking publishes complete content atomically and
        # refuses any destination created since the preflight. Never replace it.
        os.link(temporary, path, follow_symlinks=False)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_log(root: Path) -> dict:
    path = _safe_path(root, "data/acquisition_log.json")
    if not path.exists():
        return {"schema_version": 1, "canonical_source_notice": HISTORICAL_NOTICE, "events": []}
    if not path.is_file():
        raise ValueError("Acquisition log must be a regular JSON file")
    log = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(log, dict) or log.get("schema_version") != 1
            or log.get("canonical_source_notice") != HISTORICAL_NOTICE
            or not isinstance(log.get("events"), list)):
        raise ValueError("Unexpected acquisition log; inspect rather than overwrite")
    return log


def _write_log(root: Path, log: dict) -> None:
    path = _safe_path(root, "data/acquisition_log.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".acquisition-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(log, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _safe_path(root, "data/acquisition_log.json")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _complete_pending(root: Path, log: dict, plans: dict) -> bool:
    changed = False
    for event in log["events"]:
        if event.get("publication_state") != "pending":
            continue
        datasets = event["datasets"]
        if not set(datasets).issubset(plans):
            continue
        for name, evidence in datasets.items():
            expected = hashlib.sha256(plans[name]["source_bytes"]).hexdigest()
            if evidence["historical_manifest_sha256"] != expected:
                raise ValueError("Pending acquisition has a different frozen manifest")
        if all(_local_state(root, plans[name]) == ([], False) for name in datasets):
            event["publication_state"] = "completed"
            event["publication_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            event["completion_basis"] = "Frozen inputs now validated; publication may have completed on a later retry."
            changed = True
    return changed


def prepare_data(root: str | Path, datasets=None) -> dict:
    """Validate the whole selection before networking; never overwrite bad inputs."""
    root = Path(root).resolve(strict=True)
    selected = tuple(DATASETS) if datasets is None else tuple(datasets)
    if not selected or len(set(selected)) != len(selected) or any(d not in DATASETS for d in selected):
        raise ValueError("Select distinct datasets from news, uji, leukemia")
    plans = {}
    for dataset in selected:
        manifest, folder, _ = DATASETS[dataset]
        source_bytes = _safe_path(root, manifest).read_bytes()
        record = json.loads(source_bytes)
        plan = {"manifest": manifest, "folder": folder, "source_bytes": source_bytes,
                "files": _expected_files(dataset, record)}
        plan["missing"], plan["source_missing"] = _local_state(root, plan)
        plan["downloaded"] = bool(plan["missing"])
        plans[dataset] = plan
    log = _read_log(root)  # A bad/symlinked log is rejected before downloading.
    downloaded = any(plan["missing"] for plan in plans.values())
    started = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="lasso-frozen-data-") as temporary:
        staging_root = Path(temporary).resolve()
        for dataset, plan in plans.items():
            if not plan["missing"]:
                continue
            folder = staging_root / dataset
            _download_to_staging(root, dataset, folder)
            # The helper's new source.json is deliberately not canonicalized.
            # Independently check every raw payload against the historical manifest.
            for name, expected in plan["files"].items():
                _check_file(_safe_path(staging_root, Path(dataset) / name), expected)
        # Revalidate existing inputs after all downloads, before copying any data.
        for plan in plans.values():
            plan["missing"], plan["source_missing"] = _local_state(root, plan)
        changed = downloaded or any(plan["source_missing"] for plan in plans.values())
        if changed:
            # Persist the real acquisition window before publishing any raw file.
            # A crash then leaves a truthful pending record that an offline retry
            # can complete, rather than silently losing acquisition provenance.
            completed = datetime.now(timezone.utc).isoformat()
            log["events"].append({
                "preparation_started_at_utc": started,
                "network_acquisition_completed_at_utc": completed if downloaded else None,
                "network_used": downloaded, "publication_state": "pending",
                "datasets": {name: {"downloaded": plan["downloaded"],
                                    "raw_files_to_publish": plan["missing"],
                                    "canonical_source_restoration_planned": plan["source_missing"],
                                    "historical_manifest": plan["manifest"],
                                    "historical_manifest_sha256": hashlib.sha256(plan["source_bytes"]).hexdigest()}
                             for name, plan in plans.items()},
            })
            _write_log(root, log)
        for dataset, plan in plans.items():
            for name in plan["missing"]:
                _write_missing(root, Path(plan["folder"]) / name, staging_root / dataset / name)
            if plan["source_missing"]:
                _write_missing(root, Path(plan["folder"]) / "source.json", plan["source_bytes"])
            if _local_state(root, plan) != ([], False):
                raise ValueError("Frozen inputs changed during publication; inspect and retry")
    if _complete_pending(root, log, plans):
        _write_log(root, log)
    return {"network_used": downloaded, "canonical_source_notice": HISTORICAL_NOTICE,
            "datasets": {name: {"copied_raw_files": plan["missing"],
                                 "canonical_source_restored": plan["source_missing"],
                                 "canonical_source_sha256": hashlib.sha256(plan["source_bytes"]).hexdigest()}
                         for name, plan in plans.items()},
            "acquisition_log": "data/acquisition_log.json" if log["events"] else None,
            "pending_acquisitions": sum(event.get("publication_state") == "pending" for event in log["events"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = prepare_data(args.root, None if args.dataset == "all" else (args.dataset,))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
