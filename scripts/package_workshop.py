#!/usr/bin/env python3
"""Package the completed, independently checked workshop study.

This is a packaging gate, not a numerical verifier: it requires fresh full
verify_workshop reports, verified analyses, representative reproduction and
stable PDF reports. It never fits models, regenerates data, or draws figures.
Do not run while any timed stage, analysis or report build is in progress.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import zipfile
import zlib
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "comp6704_workshop/"
STAGES = ("smoke", "pilot", "confirm", "real")
RESULT_STAGES = (*STAGES, "reproduce")
CHUNK = 1024 * 1024
REQUIRED_SCRIPTS = (
    "run_workshop.py", "run_workshop_followup.py", "analyze_workshop.py",
    "verify_workshop.py", "reproduce_workshop.py", "audit_workshop_feasibility.py",
    "build_workshop_brief.py", "build_workshop_update.py", "build_report.py",
    "download_benchmarks.py", "diagnose_workshop_references.py",
    "analyze_workshop_reference_impact.py", "package_workshop.py",
)
PROHIBITED_PARTS = {".venv", ".modern_deps", ".report_deps", ".git", "__pycache__", "preview", "assets", "build", "data"}
PROHIBITED_SUFFIXES = {".so", ".dll", ".dylib", ".o", ".a", ".pyc", ".pyo", ".ttf", ".otf", ".zip", ".bz2", ".libsvm"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            value.update(block)
    return value.hexdigest()


def path_in_root(relative):
    relative = Path(relative)
    require(not relative.is_absolute(), f"Expected workspace-relative evidence path: {relative}")
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), f"Missing/invalid evidence path: {relative}")
    return path


def check_inputs(provenance, checked, label):
    require(isinstance(provenance, dict) and bool(provenance), f"Missing input hashes: {label}")
    for relative, expected in provenance.items():
        path = path_in_root(relative)
        if relative in checked:
            require(checked[relative] == expected, f"Conflicting input hash: {relative}")
        require(digest(path) == expected, f"{label} input changed: {relative}")
        checked[relative] = expected


def json_signature(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def current_signature(environment, stage, checked):
    """Recompute the frozen source/dependency signature without importing solvers."""
    base = environment if stage in {"smoke", "pilot"} else environment["base_provenance"]
    source_hashes = base["source_hashes"]
    versions = base["dependency_versions"]
    check_inputs(source_hashes, checked, f"{stage} numerical source")
    for name, expected in versions.items():
        require(importlib.metadata.version(name) == expected, f"Installed {name} differs from the frozen numerical environment")
    base_signature = json_signature({"source_hashes": source_hashes, "dependency_versions": versions})
    if stage in {"smoke", "pilot"}:
        signature = base_signature
    else:
        followup_path = ROOT / "scripts/run_workshop_followup.py"
        followup_hash = digest(followup_path)
        require(environment["base_pilot_numerical_signature"] == base_signature
                and environment["followup_source_sha256"] == followup_hash, f"Follow-up source changed: {stage}")
        checked[str(followup_path.relative_to(ROOT))] = followup_hash
        signature = json_signature({"base_pilot_numerical_signature": base_signature,
                                    "followup_source_sha256": followup_hash,
                                    "config_sha256": environment["config_sha256"]})
    require(signature == environment["numerical_signature"] == environment["source_signature"], f"Numerical signature changed: {stage}")
    return signature


def validate_stage(stage, checked):
    base = ROOT / "results/workshop" / stage
    paths = {name: base / f"{name}.json" for name in ("verification", "analysis", "completion", "environment", "warmup")}
    for path in paths.values():
        require(path.is_file(), f"Finish validation/analysis before packaging: {path.relative_to(ROOT)}")
    verification, analysis, completion, environment, warmup = [read(paths[name]) for name in paths]
    require(verification.get("stage") == analysis.get("stage") == completion.get("stage") == stage, f"Stage labels differ: {stage}")
    require(verification.get("all_passed") is True and verification.get("numerical_recomputed") is True
            and verification.get("numerical_passed") is True and verification.get("complete_grid") is True,
            f"A full successful numerical audit is required: {stage}")
    require(verification["verifier_script_sha256"] == digest(ROOT / "scripts/verify_workshop.py"), f"Verifier changed after validation: {stage}")
    check_inputs(verification["input_provenance"], checked, f"{stage} verification")
    require(analysis.get("integrity_verified") is True
            and analysis.get("analysis_script_sha256") == digest(ROOT / "scripts/analyze_workshop.py"), f"Final verified analysis is stale: {stage}")
    check_inputs(analysis["input_provenance"], checked, f"{stage} analysis")
    signature = current_signature(environment, stage, checked)
    config_path = ROOT / "configs" / ("workshop_pilot.json" if stage in {"smoke", "pilot"} else f"workshop_{stage}.json")
    config_hash = digest(config_path)
    require(verification["numerical_signature"] == completion["numerical_signature"] == signature
            and verification["config_sha256"] == completion["config_sha256"] == environment["config_sha256"] == config_hash,
            f"Stage source/configuration hashes differ: {stage}")
    require(analysis["numerical_signatures"] == [signature] and analysis["config_sha256"] == [config_hash], f"Analysis source/configuration differs: {stage}")
    expected_targets, expected_runs = verification["expected_targets"], verification["expected_trajectories"]
    require(analysis["n_targets"] == completion["expected_targets"] == completion["recorded_targets"] == expected_targets
            and analysis["n_runs"] == completion["expected_trajectories"] == completion["recorded_trajectories"] == expected_runs,
            f"Stage counts differ: {stage}")
    # Hash checks catch changed/deleted files; this exact set check catches new
    # unverified records added after the audit, including duplicate grid cells.
    for folder, suffix in (("datasets", "json"), ("references", "json"), ("reference_coefficients", "npz"),
                           ("runs", "json"), ("trajectories", "npz"), ("failures", "json")):
        actual = {str(path.relative_to(ROOT)) for path in (base / folder).glob(f"*.{suffix}")}
        recorded = {name for name in verification["input_provenance"]
                    if Path(name).parent == Path("results/workshop") / stage / folder and name.endswith("."+suffix)}
        require(actual == recorded, f"New/missing artifacts since verification: {stage}/{folder}")
    runs = [read(path) for path in sorted((base / "runs").glob("*.json"))]
    statuses = dict(Counter(run["status"] for run in runs))
    require(statuses == verification["trajectory_status_counts"] == completion["trajectory_status_counts"] == analysis["statuses"], f"Run statuses differ: {stage}")
    require(len({run["run_id"] for run in runs}) == expected_runs and len({run["target_id"] for run in runs}) == expected_targets,
            f"Duplicate trajectory/target identifiers: {stage}")
    refs = [read(path) for path in sorted((base / "references").glob("*.json"))]
    accepted = sum(bool(ref["reference_accepted"]) for ref in refs)
    require(accepted == completion["reference_accepted_targets"] == analysis["reference_targets_accepted"], f"Reference acceptance counts differ: {stage}")
    for path in paths.values():
        checked[str(path.relative_to(ROOT))] = digest(path)
    checked[str(config_path.relative_to(ROOT))] = config_hash
    return {"stage": stage, "expected_targets": expected_targets, "expected_trajectories": expected_runs,
            "all_passed": True, "numerical_signature": signature, "config_sha256": config_hash,
            "trajectory_status_counts": statuses, "reference_accepted_targets": accepted,
            "reference_uncertain_targets": expected_targets-accepted,
            "preserved_exception_records": verification["preserved_exception_records"],
            "unreached_threshold_counts": verification["unreached_threshold_counts"],
            "completion_wall_sec": completion["wall_sec_this_invocation"],
            "warmup_sec_outside_stage_timer": float(warmup["cd_setup_sec"]) + float(warmup["modern"]["elapsed_sec"]),
            "verification_wall_sec": verification["verification_wall_sec"]}


def validate_reproduction(confirm, checked):
    base = ROOT / "results/workshop/reproduce"
    path = base / "reproduction.json"
    require(path.is_file(), "Representative reproduction has not finished")
    report = read(path)
    require(report.get("passed") is True and report.get("source_signature_unchanged") is True
            and report.get("original_inputs_unchanged") is True, "Representative reproduction did not pass")
    require(report["reproducer_script_sha256"] == digest(ROOT / "scripts/reproduce_workshop.py"), "Reproducer changed after the fresh fits")
    require(report["numerical_signature"] == report["source_signature"] == confirm["numerical_signature"]
            and report["config_sha256"] == confirm["config_sha256"], "Reproduction source/configuration differs from confirmation")
    require(report["expected_targets"] == len(report["targets"]) == 2
            and report["expected_new_trajectory_fits"] == report["trajectory_fit_attempts"] == report["trajectory_fits_completed"] == len(report["runs"]) == 4
            and report["new_reference_fits"] == 0 and all(run["passed"] and run["new_fit"] for run in report["runs"]),
            "The predeclared four fresh trajectory solves are incomplete")
    check_inputs(report["input_sha256"], checked, "Representative reproduction")
    expected_run_ids = {run["run_id"] for run in report["runs"]}
    require(len(expected_run_ids) == 4 and {p.stem for p in (base / "runs").glob("*.json")} == expected_run_ids
            and {p.stem for p in (base / "trajectories").glob("*.npz")} == expected_run_ids,
            "Reproduction artifact grid differs from its acceptance report")
    for row in report["runs"]:
        run_path = path_in_root(row["reproduced_run_path"])
        require(run_path.parent == base / "runs", "Reproduction run escaped its stage")
        run = read(run_path)
        trajectory_path = (base / run["trajectory_path"]).resolve()
        require(trajectory_path.is_relative_to(base.resolve()) and trajectory_path.is_file(), "Invalid reproduced trajectory path")
        require(digest(trajectory_path) == run["trajectory_sha256"] == row["reproduced_trajectory_sha256"], "Reproduction coefficient artifact changed")
        require(run["numerical_signature"] == confirm["numerical_signature"] and run["config_sha256"] == confirm["config_sha256"], "Reproduced run provenance differs")
        checked[str(run_path.relative_to(ROOT))] = digest(run_path)
        checked[str(trajectory_path.relative_to(ROOT))] = digest(trajectory_path)
    checked[str(path.relative_to(ROOT))] = digest(path)
    return {"passed": True, "fresh_targets": 2, "fresh_trajectory_fits": 4, "new_reference_fits": 0,
            "references_reused_without_fitting": True, "wall_sec": report["wall_sec"],
            "reproduction_report_sha256": checked[str(path.relative_to(ROOT))]}


def validate_pdfs(checked):
    # Lazy import: packaging needs a page-count reader but no plotting backend.
    if (ROOT / ".report_deps").is_dir():
        sys.path.insert(0, str(ROOT / ".report_deps"))
    import pymupdf
    expected = {
        "pilot_brief_zh.pdf": ("pilot_brief_validation.json", "pilot_brief_content_zh.json"),
        "research_update_zh.pdf": ("research_update_validation.json", "research_update_content_zh.json"),
    }
    reports = []
    for filename, (validation_name, content_name) in expected.items():
        path = ROOT / "report/workshop" / filename
        validation_path, content_path = path.with_name(validation_name), path.with_name(content_name)
        require(path.is_file() and validation_path.is_file() and content_path.is_file(), f"Missing final report evidence: {filename}")
        validation = read(validation_path)
        check_inputs(validation["input_provenance"], checked, filename)
        actual_hash, actual_bytes = digest(path), path.stat().st_size
        with pymupdf.open(path) as document:
            pages = len(document)
            require(pages == validation["pdf_pages"] and 5 <= pages <= 6, f"Unexpected report page count: {filename}")
            if "page_text_chars" in validation:
                require([len(page.get_text()) for page in document] == validation["page_text_chars"], f"Report text changed: {filename}")
        require(actual_bytes == validation["pdf_bytes"] and actual_hash == validation["pdf_sha256"]
                and actual_bytes < 20_000_000, f"Report bytes/hash differs from validation: {filename}")
        require(validation.get("identity_complete") is False, "Report should retain blank name/student ID")
        if "content_sha256" in validation:
            require(digest(content_path) == validation["content_sha256"], f"Report content changed: {filename}")
        for item in (path, validation_path, content_path):
            checked[str(item.relative_to(ROOT))] = digest(item)
        reports.append({"path": str(path.relative_to(ROOT)), "pages": pages, "bytes": actual_bytes,
                        "sha256": actual_hash, "validation_sha256": digest(validation_path),
                        "visual_review_status": validation.get("visual_review_status", "See workshop review documentation")})
    return reports


def validate_uji(checked):
    path = ROOT / "data/benchmarks/ujiindoorloc/source.json"
    require(path.is_file(), "Missing local UJI source/license metadata")
    source = read(path)
    require(source["dataset"] == "UJIIndoorLoc" and source["license"] == "CC BY 4.0", "Unexpected UJI source/license snapshot")
    for filename, record in source["files"].items():
        artifact = (path.parent / filename).resolve()
        require(artifact.is_relative_to(path.parent.resolve()) and artifact.is_file()
                and artifact.stat().st_size == record["bytes"] and digest(artifact) == record["sha256"],
                f"Local UJI input differs from the verified source snapshot: {filename}")
        checked[str(artifact.relative_to(ROOT))] = record["sha256"]
    checked[str(path.relative_to(ROOT))] = digest(path)
    return path, source


def resources(stages, reproduction):
    path = ROOT / "results/v2/verification.json"
    require(path.is_file(), "Missing preserved v1/v2 resource verification snapshot")
    snapshot = read(path)
    prior = snapshot["resources"]
    prior_sec = float(prior["total_recorded_cpu_wall_sec"])
    require(math.isclose(prior_sec/3600, prior["total_recorded_cpu_wall_hours"], rel_tol=1e-12), "Prior resource arithmetic differs")
    stage_wall = sum(stage["completion_wall_sec"] for stage in stages.values())
    warmups = sum(stage["warmup_sec_outside_stage_timer"] for stage in stages.values())
    reproduction_wall = float(reproduction["wall_sec"])
    numerical_audit = sum(stage["verification_wall_sec"] for stage in stages.values())
    total = prior_sec + stage_wall + warmups + reproduction_wall
    result = {"prior_v1_v2_recorded_wall_sec": prior_sec, "prior_v1_v2_recorded_wall_hours": prior_sec/3600,
              "workshop_completion_wall_sec": {name: item["completion_wall_sec"] for name, item in stages.items()},
              "workshop_warmup_sec_outside_stage_timers": {name: item["warmup_sec_outside_stage_timer"] for name, item in stages.items()},
              "workshop_reproduction_actual_wall_sec": reproduction_wall,
              "total_recorded_study_and_reproduction_wall_sec": total,
              "total_recorded_study_and_reproduction_wall_hours": total/3600,
              "workshop_numerical_verification_wall_sec_separately": numerical_audit,
              "total_including_workshop_numerical_verification_wall_sec": total+numerical_audit,
              "total_including_workshop_numerical_verification_wall_hours": (total+numerical_audit)/3600,
              "threads": 1, "gpu_hours": 0,
              "scope": "Recorded stage wall time plus separately recorded warmup and the four-fit reproduction invocation. Validation is separately shown and also included in the extended total. These are wall-clock accounts, not summed CPU core-hours. Installation, reading, coding, analysis, PDF/figure generation and packaging are excluded; unrecorded/overwritten attempts are not reconstructed."}
    frozen = {"original_verification_path": str(path.relative_to(ROOT)), "original_verification_sha256": digest(path),
              "original_verified_at": snapshot.get("verified_at"), "resources": prior,
              "scope": "Resource-only snapshot of the previous course/v2 verification; its numerical results and reports are not part of this workshop bundle."}
    return result, frozen


def allowed(relative):
    path = Path(relative)
    return (not path.is_absolute() and ".." not in path.parts and not set(path.parts) & PROHIBITED_PARTS
            and path.suffix.lower() not in PROHIBITED_SUFFIXES
            and not path.name.endswith("_console.log") and not path.name.endswith((".tmp", ".part")))


def payload_files(uji_source, prior_snapshot, audit):
    files = {}
    def add(path, relative=None):
        require(path.is_file() and not path.is_symlink(), f"Missing/symlink payload: {path}")
        name = relative or str(path.relative_to(ROOT))
        require(allowed(name), f"Forbidden archive payload: {name}")
        require(name not in files, f"Duplicate archive payload: {name}")
        files[name] = {"path": path, "origin": str(path.relative_to(ROOT))}
    def virtual(name, body, origin):
        require(allowed(name) and name not in files, f"Invalid generated payload: {name}")
        files[name] = {"body": body, "origin": origin}
    for path in sorted((ROOT / "src/lasso_study").glob("*")):
        if path.suffix in {".py", ".c"}:
            add(path)
    for path in sorted((ROOT / "tests").rglob("*.py")):
        if "__pycache__" not in path.parts:
            add(path)
    for name in REQUIRED_SCRIPTS:
        add(ROOT / "scripts" / name)
    for path in sorted((ROOT / "scripts").glob("*workshop*.py")):
        if str(path.relative_to(ROOT)) not in files:
            add(path)
    for path in sorted((ROOT / "configs").glob("workshop_*.json")):
        add(path)
    add(ROOT / "configs/skglm_environment.json")
    for name in ("requirements.txt", "requirements-modern.txt", "requirements-report.txt"):
        add(ROOT / name)
    for path in sorted((ROOT / "docs").glob("workshop*.md")):
        add(path)
    reproduction_doc = ROOT / "docs/workshop_reproduction.md"
    virtual("README.md", reproduction_doc.read_bytes(), str(reproduction_doc.relative_to(ROOT)))
    virtual("docs/workshop_uji_source.json", uji_source.read_bytes(), str(uji_source.relative_to(ROOT)))
    virtual("docs/workshop_prior_resources.json", encoded(prior_snapshot), "resource-only snapshot of results/v2/verification.json")
    virtual("results/workshop/package_verification.json", encoded(audit), "generated packaging gate evidence")
    for stage in RESULT_STAGES:
        folder = ROOT / "results/workshop" / stage
        require(folder.is_dir(), f"Missing workshop evidence folder: {stage}")
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".json", ".csv", ".npz", ".txt", ".log"} and allowed(str(path.relative_to(ROOT))):
                add(path)
        for path in sorted((ROOT / "figures/workshop" / stage).rglob("*")):
            if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".svg"} and allowed(str(path.relative_to(ROOT))):
                add(path)
    for path in sorted((ROOT / "report/workshop").glob("*")):
        if path.is_file() and path.suffix.lower() in {".pdf", ".json"}:
            add(path)
    # Both workshop builders carry their checked bibliography entries inline;
    # render_pdf does not call the shared renderer's course bib_entries function.
    # Thus the unrelated course references.bib/report assets are not dependencies.
    return files


def metadata(item):
    digest_value, crc, size = hashlib.sha256(), 0, 0
    if "body" in item:
        chunks = [item["body"]]
        stream = None
    else:
        stream = item["path"].open("rb")
        chunks = iter(lambda: stream.read(CHUNK), b"")
    try:
        for chunk in chunks:
            digest_value.update(chunk); crc = zlib.crc32(chunk, crc); size += len(chunk)
    finally:
        if stream is not None:
            stream.close()
    return {"sha256": digest_value.hexdigest(), "crc32": f"{crc & 0xffffffff:08x}", "bytes": size, "origin": item["origin"]}


def build_archive(output, files, audit):
    records = {name: metadata(item) for name, item in sorted(files.items())}
    manifest = {"format": "comp6704-workshop-manifest-1", "archive_prefix": PREFIX,
                "generated_at": audit["checked_at"], "payload_files": len(records),
                "uncompressed_payload_bytes": sum(item["bytes"] for item in records.values()),
                "files": records, "manifest_self_hash_scope": "This manifest lists every payload except itself; the outer package record hashes the manifest and ZIP."}
    manifest_body = encoded(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for name, item in sorted(files.items()):
                archive_name = PREFIX + name
                if "body" in item:
                    archive.writestr(archive_name, item["body"])
                else:
                    archive.write(item["path"], archive_name)
            archive.writestr(PREFIX + "manifest_workshop.json", manifest_body)
        expected_names = {PREFIX + name for name in records} | {PREFIX + "manifest_workshop.json"}
        with zipfile.ZipFile(temporary, "r") as archive:
            require(len(archive.namelist()) == len(set(archive.namelist())) and set(archive.namelist()) == expected_names, "Archive path set differs from manifest")
            for name, record in records.items():
                info = archive.getinfo(PREFIX + name)
                value, crc, size = hashlib.sha256(), 0, 0
                with archive.open(info) as stream:
                    for block in iter(lambda: stream.read(CHUNK), b""):
                        value.update(block); crc = zlib.crc32(block, crc); size += len(block)
                require(size == info.file_size == record["bytes"] and value.hexdigest() == record["sha256"]
                        and f"{crc & 0xffffffff:08x}" == record["crc32"] and info.CRC == crc & 0xffffffff,
                        f"ZIP SHA256/CRC payload verification failed: {name}")
            require(archive.read(PREFIX + "manifest_workshop.json") == manifest_body, "Embedded manifest differs")
        # Reject concurrent report/log/evidence changes instead of shipping a
        # manifest for an earlier version. Active console logs are never payloads.
        for name, item in files.items():
            require(metadata(item) == records[name], f"Payload changed during packaging: {name}")
        check_inputs(audit["checked_inputs_sha256"], {}, "Final packaging stability")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    manifest_path = output.with_name("manifest_workshop.json")
    manifest_path.write_bytes(manifest_body)
    package = {"packaged_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "all_passed": True,
               "archive": str(output.relative_to(ROOT)), "archive_prefix": PREFIX,
               "zip_bytes": output.stat().st_size, "zip_sha256": digest(output),
               "zip_entries": len(records)+1, "payload_files": len(records),
               "payload_bytes": manifest["uncompressed_payload_bytes"], "every_payload_sha256_and_crc_readback_passed": True,
               "manifest_path": str(manifest_path.relative_to(ROOT)), "manifest_sha256": hashlib.sha256(manifest_body).hexdigest(),
               "research_stages": audit["research_stages"], "reproduction": audit["reproduction"], "reports": audit["reports"],
               "resources": audit["resources"], "scope": audit["scope"]}
    output.with_name("package_workshop.json").write_bytes(encoded(package))
    return package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "deliverables/comp6704_workshop_research_bundle.zip")
    parser.add_argument("--check-only", action="store_true", help="Check all final inputs; do not compress or write an archive")
    args = parser.parse_args()
    if (ROOT / ".modern_deps").is_dir():
        sys.path.insert(0, str(ROOT / ".modern_deps"))
    output = args.output.resolve()
    require(output.is_relative_to((ROOT / "deliverables").resolve()) and output.suffix == ".zip", "Workshop archive must be a ZIP under deliverables")
    require(output.name != "comp6704_research_bundle_v2.zip", "Do not overwrite the frozen course/v2 archive")
    checked = {}
    stages = {stage: validate_stage(stage, checked) for stage in STAGES}
    reproduction = validate_reproduction(stages["confirm"], checked)
    reports = validate_pdfs(checked)
    uji_path, _ = validate_uji(checked)
    resource_record, prior_snapshot = resources(stages, reproduction)
    checked["results/v2/verification.json"] = prior_snapshot["original_verification_sha256"]
    checked["scripts/package_workshop.py"] = digest(Path(__file__))
    checked["docs/workshop_reproduction.md"] = digest(ROOT / "docs/workshop_reproduction.md")
    audit = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "all_passed": True,
             "packager_sha256": digest(Path(__file__)), "research_stages": stages,
             "reproduction": reproduction, "reports": reports, "resources": resource_record,
             "checked_inputs_sha256": checked,
             "scope": "Workshop-only source, tests, registered configurations, checked saved evidence and reports. Raw input data, original assignment/course reports, environments, compiled binaries, fonts, previews and active console logs are excluded. Derived coefficient NPZ evidence is included. Nonconverged research trajectories and uncertain references are preserved, not required to disappear."}
    files = payload_files(uji_path, prior_snapshot, audit)
    if args.check_only:
        print(json.dumps({"all_passed": True, "candidate_payload_files": len(files), "compressed": False,
                          "research_stages": stages, "reproduction": reproduction,
                          "reports": reports, "resources": resource_record}, ensure_ascii=False, indent=2))
        return
    package = build_archive(output, files, audit)
    print(json.dumps(package, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
