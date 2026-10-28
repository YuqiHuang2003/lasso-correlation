#!/usr/bin/env python3
"""Verify the complete v2 evidence and create a data-free research bundle.

This verifies saved coefficients against regenerated inputs; it never reruns
an optimization algorithm. Nonconverged and failed runs remain visible.
"""
from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_variable] = "1"

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_v2
from lasso_study.data_v2 import make_synthetic, make_uji, make_leukemia
from lasso_study.solvers import certificate

STAGES = ("synthetic", "uji", "leukemia")
RETURNED_STATUSES = {"converged", "time_limit", "max_iter", "budget_exceeded", "not_converged"}
EXCLUDED_PARTS = {".venv", ".modern_deps", ".report_deps", ".git", ".agents", ".codex",
                  ".aws", "build", "__pycache__", "preview", "assets", "development", "data"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".so", ".o", ".a", ".aux", ".out", ".bbl", ".blg", ".nbc", ".nbi"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path):
    require(path.is_file(), f"Missing required evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def expected_grid(config: dict) -> dict:
    """Map every declared cell to its details and source dataset."""
    expected = {stage: {} for stage in STAGES}
    settings = config["synthetic"]
    for seed in settings["seeds"]:
        for rho in settings["rhos"]:
            for layout in settings["layouts"]:
                for sparsity in settings["sparsities"]:
                    dataset_id = f"syn_seed{seed}_r{rho:g}_s{sparsity}_{layout}"
                    for factor in settings["lambda_factors"]:
                        for method in config["methods"]:
                            run_id = f"{dataset_id}_c{factor:g}_{method}"
                            expected["synthetic"][run_id] = {
                                "dataset": "synthetic", "seed": seed, "rho": rho,
                                "sparsity": sparsity, "layout": layout, "lambda_factor": factor,
                                "method": method, "dataset_id": dataset_id}
    for seed in config["uji"]["seeds"]:
        metadata = read(ROOT / f"results/v2/uji/datasets/uji_seed{seed}.json")
        selected = metadata["selected_indices"]
        require(len(selected) == 3 and len(set(selected)) == 3,
                f"UJI seed {seed} must have three unique selected lambda indices")
        require(metadata["best_index"] in selected, f"Missing best-validation lambda for UJI seed {seed}")
        require(len(metadata["candidates"]) == config["uji"]["validation_candidates"],
                f"UJI seed {seed}: candidate count differs from configuration")
        require(metadata["best_index"] == int(np.argmin([c["validation_mse"] for c in metadata["candidates"]])),
                f"UJI seed {seed}: recorded best lambda is not the validation minimum")
        for candidate in metadata["candidates"]:
            require(candidate["normalized_gap"] <= 1e-7, "Uncertified UJI selection candidate")
        for index in selected:
            for method in config["methods"]:
                run_id = f"uji_seed{seed}_i{index}_{method}"
                expected["uji"][run_id] = {
                    "dataset": "uji", "seed": seed, "validation_index": index,
                    "selected_by_validation": index == metadata["best_index"],
                    "method": method, "dataset_id": f"uji_seed{seed}"}
    for ratio in config["leukemia"]["lambda_ratios"]:
        for repeat in config["leukemia"]["repeats"]:
            for method in config["methods"]:
                run_id = f"leukemia_a{ratio:g}_rep{repeat}_{method}"
                expected["leukemia"][run_id] = {
                    "dataset": "leukemia", "repeat": repeat, "seed": None,
                    "lambda_ratio": ratio, "method": method, "dataset_id": "leukemia"}
    counts = {stage: len(cells) for stage, cells in expected.items()}
    require(counts == {"synthetic": 1440, "uji": 36, "leukemia": 36},
            f"This v2 report expects its frozen 1440/36/36 grid, got {counts}")
    return expected


def check_certificate(values: dict, label: str, required_tol: float | None = None) -> None:
    fields = ("primal", "dual", "gap", "raw_gap", "normalized_gap", "dual_feasibility", "kkt_inf")
    require(all(key in values and np.isfinite(values[key]) for key in fields),
            f"Missing or nonfinite certificate field: {label}")
    scale = max(1., abs(values["primal"]), abs(values["dual"]))
    raw = values["primal"] - values["dual"]
    require(raw >= -1e-11 * scale, f"Negative primal-dual gap: {label}")
    require(np.isclose(values["raw_gap"], raw, rtol=1e-7, atol=1e-11 * scale), f"Inconsistent raw gap: {label}")
    require(np.isclose(values["gap"], max(0., raw), rtol=1e-7, atol=1e-11 * scale), f"Inconsistent gap: {label}")
    require(np.isclose(values["normalized_gap"], values["gap"] / scale, rtol=1e-7, atol=1e-11),
            f"Inconsistent normalization: {label}")
    require(-1e-12 <= values["dual_feasibility"] <= 1. + 1e-10, f"Dual infeasibility: {label}")
    require(values["kkt_inf"] >= -1e-12, f"Negative KKT violation magnitude: {label}")
    if required_tol is not None:
        require(values["normalized_gap"] <= required_tol,
                f"Claimed success does not meet the common gap tolerance: {label}")


def regenerated_dataset(stage: str, details: dict, config: dict) -> tuple[dict, dict, np.ndarray, np.ndarray, str]:
    if stage == "synthetic":
        params = {key: config["synthetic"][key] for key in ("n", "p", "block_size", "noise_std", "n_test")}
        data = make_synthetic(seed=details["seed"], rho=details["rho"], sparsity=details["sparsity"],
                              layout=details["layout"], **params)
        data_hash = data["metadata"]["hashes"]["X"] + data["metadata"]["hashes"]["y"]
    elif stage == "uji":
        data = make_uji(seed=details["seed"])
        X, y = np.asfortranarray(data["X_train"]), data["y_train"]
        data_hash = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
    else:
        data = make_leukemia()
        X, y = np.asfortranarray(data["X"]), data["y"]
        data_hash = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
    X = np.asfortranarray(data.get("X", data.get("X_train")))
    y = data.get("y", data.get("y_train"))
    metadata = read(ROOT / "results/v2" / stage / "datasets" / (details["dataset_id"] + ".json"))
    for key, value in data["metadata"].items():
        require(key in metadata and metadata[key] == value,
                f"Dataset metadata differs from regenerated input: {stage}/{details['dataset_id']}:{key}")
    return data, metadata, X, y, data_hash


def verify_runs(config: dict, source_signature: str, config_hash: str, expected: dict) -> dict:
    overview = {}
    failures = []
    returned_nonconverged = []
    checked = 0
    max_reference_gap = 0.
    max_certificate_difference = 0.
    for stage in STAGES:
        folder = ROOT / "results/v2" / stage
        require(read(folder / "environment.json")["v2_numerical_signature"] == source_signature,
                f"Stage numerical signature mismatch: {stage}")
        require(read(folder / "environment.json")["v2_config_sha256"] == config_hash,
                f"Stage configuration signature mismatch: {stage}")
        require(read(folder / "config.json") == config, f"Stage configuration differs: {stage}")
        versions = read(folder / "warmup.json")["modern"]["versions"]
        require(all(versions.get(name) == expected_version for name, expected_version in
                    {"skglm": "0.5", "numba": "0.68.1", "llvmlite": "0.50.0"}.items()),
                f"Modern baseline versions differ from the locked environment: {stage}")
        read(folder / "timing.json")  # A completed stage must have saved its elapsed time.
        run_paths = {p.stem: p for p in sorted((folder / "runs").glob("*.json"))}
        failure_paths = {p.stem: p for p in sorted((folder / "failures").glob("*.json"))}
        require(set(run_paths) | set(failure_paths) == set(expected[stage]),
                f"Missing or unexpected grid cells in {stage}: "
                f"missing={sorted(set(expected[stage]) - (set(run_paths) | set(failure_paths)))}, "
                f"unexpected={sorted((set(run_paths) | set(failure_paths)) - set(expected[stage]))}")
        grouped = defaultdict(list)
        for run_id, details in expected[stage].items():
            grouped[details["dataset_id"]].append((run_id, details))
        statuses = Counter()
        stage_failed = 0
        for dataset_id, cells in grouped.items():
            data, metadata, X, y, expected_data_hash = regenerated_dataset(stage, cells[0][1], config)
            lambda_max = float(np.max(np.abs(X.T @ y)) / len(y))
            for run_id, details in cells:
                if run_id in failure_paths:
                    failure = read(failure_paths[run_id])
                    require(failure.get("run_id") == run_id, f"Failure run ID mismatch: {run_id}")
                    for key, value in details.items():
                        if key != "dataset_id":
                            require(failure.get(key) == value, f"Failure grid field mismatch: {run_id}:{key}")
                    require(failure.get("numerical_signature") == source_signature
                            and failure.get("config_sha256") == config_hash
                            and failure.get("dataset_hash") == expected_data_hash,
                            f"Failure evidence signature mismatch: {run_id}")
                    require(bool(failure.get("error_type")) and bool(failure.get("error")),
                            f"Failure lacks error detail: {run_id}")
                    failures.append({"stage": stage, "run_id": run_id, "error_type": failure["error_type"],
                                     "error": failure["error"], "record": str(failure_paths[run_id].relative_to(ROOT)),
                                     "later_returned_run_exists": run_id in run_paths})
                if run_id not in run_paths:
                    statuses["failed"] += 1
                    stage_failed += 1
                    continue
                run = read(run_paths[run_id])
                require(run.get("run_id") == run_id, f"Run ID or filename mismatch: {run_id}")
                for key, value in details.items():
                    if key != "dataset_id":
                        require(run.get(key) == value, f"Grid field mismatch: {run_id}:{key}")
                require(run.get("numerical_signature") == source_signature, f"Numerical source mismatch: {run_id}")
                require(run.get("config_sha256") == config_hash, f"Configuration mismatch: {run_id}")
                require(run.get("dataset_hash") == expected_data_hash, f"Regenerated data hash mismatch: {run_id}")
                require((run["n"], run["p"]) == X.shape, f"Wrong dimensions: {run_id}")
                require(np.isclose(run["lambda_max"], lambda_max, rtol=1e-12, atol=1e-13), f"lambda_max mismatch: {run_id}")
                if stage == "synthetic":
                    lam = details["lambda_factor"] * config["synthetic"]["noise_std"] * np.sqrt(
                        2 * np.log(config["synthetic"]["p"]) / config["synthetic"]["n"])
                elif stage == "uji":
                    lam = metadata["candidates"][details["validation_index"]]["lambda"]
                else:
                    lam = details["lambda_ratio"] * lambda_max
                require(np.isclose(run["lambda"], lam, rtol=1e-12, atol=1e-13), f"Lambda mismatch: {run_id}")
                require(np.isclose(run["lambda_ratio"], run["lambda"] / lambda_max, rtol=1e-12, atol=1e-13),
                        f"Relative lambda mismatch: {run_id}")
                status = run["status"]
                require(status in RETURNED_STATUSES, f"Unrecognized status that needs explicit treatment: {run_id}:{status}")
                statuses[status] += 1
                history = run["history"]
                require(bool(history) and history[-1] == run["final"], f"Missing/inconsistent final iterate: {run_id}")
                require(all(h["time"] >= 0 for h in history)
                        and all(a["time"] <= b["time"] for a, b in zip(history, history[1:])),
                        f"Nonmonotone checkpoint timestamps: {run_id}")
                require(run["elapsed_sec"] + 1e-8 >= history[-1]["time"], f"Solver time omits final diagnostics: {run_id}")
                require(0 <= run["diagnostics_sec"] <= run["elapsed_sec"] + 1e-8, f"Invalid diagnostic time: {run_id}")
                for point in history:
                    check_certificate(point, f"{run_id}:checkpoint")
                final = history[-1]
                if status == "converged":
                    check_certificate(final, run_id, config["solver"]["tol"])
                else:
                    returned_nonconverged.append({"stage": stage, "run_id": run_id, "status": status,
                                                  "normalized_gap": final["normalized_gap"],
                                                  "elapsed_sec": run["elapsed_sec"]})
                for tolerance in config["thresholds"]:
                    hit = next((h for h in history if h["normalized_gap"] <= tolerance), None)
                    require(run["threshold_hits"].get(str(tolerance)) == hit,
                            f"Threshold hit differs from saved trajectory: {run_id}:{tolerance}")
                coefficient_path = folder / "coefficients" / (run_id + ".npz")
                require(coefficient_path.is_file() and sha(coefficient_path) == run["coefficient_sha256"],
                        f"Coefficient file hash mismatch: {run_id}")
                with np.load(coefficient_path, allow_pickle=False) as coefficients:
                    w = coefficients["coef"]
                require(w.shape == (X.shape[1],) and w.dtype == np.float64 and np.isfinite(w).all(),
                        f"Invalid coefficients or non-float64 dtype: {run_id}")
                recomputed = certificate(X, y, w, run["lambda"])
                check_certificate(recomputed, f"{run_id}:recomputed")
                if status == "converged":
                    require(recomputed["normalized_gap"] <= config["solver"]["tol"] + 1e-12,
                            f"Recomputed success certificate failed: {run_id}")
                for key, actual in recomputed.items():
                    require(np.isclose(actual, final[key], rtol=1e-7, atol=1e-9),
                            f"Saved/recomputed certificate differs: {run_id}:{key}")
                    max_certificate_difference = max(max_certificate_difference, abs(actual - final[key]))
                check_certificate(run["reference"], f"{run_id}:reference", 1e-7)
                max_reference_gap = max(max_reference_gap, run["reference"]["normalized_gap"])
                require(np.isclose(run["objective_difference_to_reference"], final["primal"] - run["reference"]["primal"],
                                   rtol=1e-7, atol=1e-10), f"Reference difference inconsistent: {run_id}")
                if stage == "leukemia":
                    require("test_mse" not in run and "support_sensitivity" not in run,
                            f"Leukemia optimization benchmark incorrectly claims prediction/recovery evidence: {run_id}")
                if run["method"] == "skglm":
                    require(run["threshold_curve_comparable"] is False,
                            f"Library endpoint observations mislabeled as a dense trajectory: {run_id}")
                    require(run["certificate_passed"] == (final["normalized_gap"] <= config["solver"]["tol"]),
                            f"Library certificate flag inconsistent: {run_id}")
                checked += 1
            del data, X, y
        overview[stage] = {"declared_cells": len(expected[stage]), "returned_runs": len(run_paths),
                           "failed_without_returned_run": stage_failed, "status_counts": dict(statuses),
                           "source_datasets_regenerated": len(grouped),
                           "complete_grid_accounting": True}
    return {"stages": overview, "verified_coefficient_files": checked,
            "max_saved_vs_recomputed_certificate_difference": max_certificate_difference,
            "max_reference_normalized_gap": max_reference_gap,
            "failed_attempts": failures, "returned_nonconverged_runs": returned_nonconverged,
            "failure_record_count": len(failures),
            "failure_policy": "Failures and returned nonconverged runs remain in the archive. A recovered failed attempt is retained separately. Grid completeness counts each declared cell once.",
            "reference_verification_scope": "Stored feasible-dual certificate consistency, feasibility ratio and normalized gap; reference coefficient vectors were not saved, so reference fits are not rerun here."}


def verify_analysis(config: dict, config_hash: str, source_signature: str, runs: dict) -> dict:
    path = ROOT / "results/v2/summary.json"
    summary = read(path)
    integrity = summary["integrity"]
    require(summary["config"] == config and integrity["config_sha256"] == config_hash,
            "Analysis uses a different configuration")
    require(integrity["numerical_signatures"] == [source_signature], "Analysis numerical signature differs")
    missing_total = 0
    for stage in STAGES:
        current = runs["stages"][stage]
        reported = integrity["counts"][stage]
        missing = current["declared_cells"] - current["returned_runs"]
        stage_failure_records = sum(row["stage"] == stage for row in runs["failed_attempts"])
        require(reported["observed_runs"] == current["returned_runs"]
                and reported["expected_runs"] == current["declared_cells"]
                and reported["exception_records"] == stage_failure_records,
                f"Stale analysis run/failure counts: {stage}")
        require(reported["duplicate_cells"] == 0 and reported["unexpected_cells"] == 0
                and reported["missing_cells"] == missing,
                f"Analysis grid accounting differs from current files: {stage}")
        require(missing == current["failed_without_returned_run"], f"Unaccounted analysis cells: {stage}")
        missing_total += missing
    expected_designs = (len(config["synthetic"]["seeds"]) * len(config["synthetic"]["rhos"])
                        * len(config["synthetic"]["layouts"]) * len(config["synthetic"]["sparsities"]))
    require(integrity["synthetic_design_count"] == integrity["expected_synthetic_designs"] == expected_designs,
            "Analysis synthetic-design metadata is incomplete or stale")
    if missing_total == 0:
        require(integrity["complete_grid"] is True and not integrity["issues"],
                "Analysis remains partial or inconsistent; regenerate it before packaging")
    else:
        # Explicit failed cells can form a reviewable, incomplete-outcome report.
        # Permit only the expected missing-run issues, never provenance errors.
        permitted = set()
        for stage in STAGES:
            current = runs["stages"][stage]
            missing = current["failed_without_returned_run"]
            if missing:
                permitted.add(f"{stage}: {current['returned_runs']} runs, expected {current['declared_cells']}")
                permitted.add(f"{stage}: missing {missing} Cartesian grid cells")
        require(integrity["complete_grid"] is False and integrity["allow_partial"] is True
                and set(integrity["issues"]) == permitted,
                "Incomplete analysis contains issues beyond explicitly recorded failed cells")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "integrity": integrity,
            "all_declared_cells_have_outcome_records": True,
            "all_cells_returned_iterates": missing_total == 0}


def verify_reproduction(config: dict, source_signature: str, config_hash: str) -> dict:
    environment = read(ROOT / "results/v2/reproduce/environment.json")
    require(environment["v2_numerical_signature"] == source_signature
            and environment["v2_config_sha256"] == config_hash,
            "Reproduction used a different numerical source or configuration")
    require(read(ROOT / "results/v2/reproduce/config.json") == config, "Reproduction configuration differs")
    reproduction = read(ROOT / "results/v2/reproduce/reproduction.json")
    expected = {f"syn_seed0_r{rho:g}_s{sparsity}_{layout}_c1_{method}"
                for rho, sparsity, layout in ((0., 5, "dispersed"), (.9, 40, "clustered"))
                for method in config["methods"]}
    rows = reproduction["checks"]
    require(len(rows) == 8 and {r["run_id"] for r in rows} == expected,
            "Representative reproduction must contain exactly the eight declared method/condition cases")
    require(reproduction["all_passed"] is True and all(row["passed"] is True for row in rows),
            "Representative reproduction did not pass")
    require(all(0 <= r["coef_max_difference"] < 1e-7 and 0 <= r["objective_difference"] < 1e-8 for r in rows),
            "Reproduction differences exceed the declared thresholds")
    read(ROOT / "results/v2/reproduce/timing.json")
    return reproduction


def verify_report() -> dict:
    pdf = ROOT / "report/report_zh_v2.pdf"
    require(pdf.is_file() and (ROOT / "report/main_v2.tex").is_file(), "Missing final v2 PDF or editable TeX")
    recorded = read(ROOT / "report/validation_v2.json")
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("Install requirements-report.txt before validating the PDF") from error
    pages = len(PdfReader(str(pdf)).pages)
    size = pdf.stat().st_size
    require(1 <= pages <= 10 and size < 20_000_000, "Report exceeds the assignment's page or size limit")
    require(recorded["pdf_pages"] == pages and recorded["pdf_bytes"] == size,
            "Saved v2 report validation does not match the actual PDF")
    return dict(recorded, pdf_sha256=sha(pdf), tex_sha256=sha(ROOT / "report/main_v2.tex"),
                independent_pdf_page_check=pages)


def resource_totals() -> dict:
    v1, v2, startup = {}, {}, {}
    for timing in sorted((ROOT / "results").glob("*/timing.json")):
        if timing.parent.name == "v2":
            continue
        value = read(timing)["wall_sec"]
        require(np.isfinite(value) and value >= 0, f"Invalid resource timing: {timing}")
        v1[timing.parent.name] = value
    for timing in sorted((ROOT / "results/v2").glob("*/timing.json")):
        value = read(timing)["wall_sec"]
        require(np.isfinite(value) and value >= 0, f"Invalid resource timing: {timing}")
        v2[timing.parent.name] = value
    for path in sorted((ROOT / "results/v2").glob("*/warmup.json")):
        warmup = read(path)
        value = warmup["cd_setup_sec"] + warmup["modern"]["elapsed_sec"]
        require(np.isfinite(value) and value >= 0, f"Invalid warmup timing: {path}")
        startup[path.parent.name] = value
    experiment_total = sum(v1.values()) + sum(v2.values())
    startup_total = sum(startup.values())
    return {"v1_recorded_stage_wall_sec": v1, "v2_recorded_stage_wall_sec": v2,
            "v2_separately_recorded_warmup_sec": startup,
            "total_recorded_stage_wall_sec": experiment_total,
            "total_recorded_warmup_sec_outside_v2_stage_timers": startup_total,
            "total_recorded_cpu_wall_sec": experiment_total + startup_total,
            "total_recorded_cpu_wall_hours": (experiment_total + startup_total) / 3600,
            "threads": 1, "gpu_hours": 0,
            "scope": "Recorded v1 and v2 experiment-stage wall times, plus v2 warmup excluded by its stage timers. Includes smoke/pilot/reproduction. Excludes installation, reading, coding, analysis, report generation, packaging and any unrecorded or overwritten earlier attempts. Wall time is not summed per-thread CPU accounting."}


def copy_source_manifests() -> list[dict]:
    records = []
    for source in sorted((ROOT / "data").rglob("source.json")):
        suffix = "_".join(source.parent.relative_to(ROOT / "data").parts)
        target = ROOT / "docs" / f"data_manifest_v2_{suffix}.json"
        target.write_bytes(source.read_bytes())
        records.append({"original": str(source.relative_to(ROOT)), "archived_metadata": str(target.relative_to(ROOT)),
                        "sha256": sha(target)})
    require(any("ujiindoorloc" in r["original"] for r in records)
            and any("leukemia" in r["original"] for r in records), "Missing benchmark source/license manifests")
    return records


def allowed_relative_path(relative: Path | PurePosixPath) -> bool:
    return (not set(relative.parts).intersection(EXCLUDED_PARTS)
            and relative.suffix.lower() not in EXCLUDED_SUFFIXES
            and relative.name != "chinese.ttf")


def bundle_files() -> list[Path]:
    files = []
    for name in ("README.md", "requirements.txt", "requirements-modern.txt", "requirements-report.txt"):
        path = ROOT / name
        require(path.is_file(), f"Missing reproduction file: {name}")
        files.append(path)
    if (ROOT / ".gitignore").is_file():
        files.append(ROOT / ".gitignore")
    for directory in ("src", "scripts", "configs", "docs", "results", "figures", "report", "tests"):
        for path in sorted((ROOT / directory).rglob("*")):
            relative = path.relative_to(ROOT)
            # These redirected orchestration logs can change while this bundle
            # is being made, including this program's final stdout. Numerical
            # experiment logs remain part of the payload.
            live_console = str(relative) in {"results/v2_package_console.log", "results/v2_report_console.log"}
            if path.is_file() and not path.is_symlink() and not live_console and allowed_relative_path(relative):
                files.append(path)
    return sorted(set(files))


def package(verification: dict, source_signature: str, config_hash: str) -> dict:
    files = bundle_files()
    contents = {str(path.relative_to(ROOT)): sha(path) for path in files}
    manifest = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "numerical_signature": source_signature, "config_sha256": config_hash,
                "archive_prefix": "comp6704_lasso_v2/", "files": contents,
                "excluded": "All raw/derived input data directories; environments, build products, preview/assets/development directories, caches, generated Chinese font and live v2 report/package console logs. Numerical experiment logs and saved result coefficients are included as research evidence.",
                "self_hash_scope": "This manifest lists payload files; it does not recursively hash itself. package_v2.json is an external archive-hash sidecar."}
    manifest_payload = json_bytes(manifest)
    deliverables = ROOT / "deliverables"
    deliverables.mkdir(exist_ok=True)
    archive = deliverables / "comp6704_research_bundle_v2.zip"
    temporary = archive.with_suffix(".zip.tmp")
    prefix = manifest["archive_prefix"]
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in files:
            zipped.write(path, arcname=prefix + str(path.relative_to(ROOT)))
        zipped.writestr(prefix + "manifest_v2.json", manifest_payload)
    with zipfile.ZipFile(temporary) as zipped:
        require(zipped.testzip() is None, "ZIP CRC validation failed")
        names = zipped.namelist()
        require(len(names) == len(set(names)) == len(contents) + 1, "Unexpected or duplicate ZIP entries")
        for relative, expected_hash in contents.items():
            entry = prefix + relative
            require(entry in names and hashlib.sha256(zipped.read(entry)).hexdigest() == expected_hash,
                    f"ZIP payload hash mismatch: {relative}")
        require(zipped.read(prefix + "manifest_v2.json") == manifest_payload, "ZIP manifest differs from verified manifest")
        for entry in names:
            parts = PurePosixPath(entry)
            require(not parts.is_absolute() and ".." not in parts.parts
                    and entry.startswith(prefix) and allowed_relative_path(parts),
                    f"Excluded or unsafe ZIP entry: {entry}")
    temporary.replace(archive)
    (deliverables / "manifest_v2.json").write_bytes(manifest_payload)
    result = {"archive": archive.name, "bytes": archive.stat().st_size, "sha256": sha(archive),
              "payload_files": len(contents), "zip_entries_including_manifest": len(contents) + 1,
              "verified": True, "zip_payload_sha256_verified": True,
              "numerical_signature": source_signature, "config_sha256": config_hash,
              "verification_sha256": sha(ROOT / "results/v2/verification.json"),
              "manifest_sha256": sha(deliverables / "manifest_v2.json"),
              "failure_record_count": verification["runs"]["failure_record_count"],
              "returned_nonconverged_run_count": len(verification["runs"]["returned_nonconverged_runs"]),
              "contains_input_data": False}
    dump(deliverables / "package_v2.json", result)
    return result


def main() -> None:
    config_path = ROOT / "configs/study_v2.json"
    config = read(config_path)
    source_signature, config_hash = run_v2.signature(), sha(config_path)
    read(ROOT / "results/v2/protocol_lock.json")
    require((ROOT / "results/v2/summary.json").is_file(), "Run scripts/analyze_v2.py before packaging")
    with threadpool_limits(limits=1):
        expected = expected_grid(config)
        runs = verify_runs(config, source_signature, config_hash, expected)
        analysis = verify_analysis(config, config_hash, source_signature, runs)
        reproduction = verify_reproduction(config, source_signature, config_hash)
        report = verify_report()
    sources = copy_source_manifests()
    verification = {
        "verified_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "numerical_signature": source_signature, "config_sha256": config_hash,
        "runs": runs, "representative_reproduction": reproduction, "report": report,
        "resources": resource_totals(), "data_source_manifests": sources,
        "analysis": analysis,
        "identity_preference": "Name and student ID remain blank as explicitly requested; author review and submission remain with the user.",
        "verification_scope": "Regenerate inputs and recompute common certificates from saved coefficient files; no optimization fits rerun during packaging. Success statuses must meet the common normalized dual gap; other statuses are retained explicitly.",
        "recomputed_success_gap_roundoff_guard": 1e-12,
    }
    dump(ROOT / "results/v2/verification.json", verification)
    result = package(verification, source_signature, config_hash)
    print(json.dumps({"package": result, "stages": runs["stages"],
                      "recorded_cpu_wall_hours": verification["resources"]["total_recorded_cpu_wall_hours"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
