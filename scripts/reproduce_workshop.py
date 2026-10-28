#!/usr/bin/env python3
"""Reproduce four registered workshop trajectories without refitting references.

The default invocation regenerates two confirmation designs, checks their frozen
hashes, and performs FISTA/CD once per design. Existing reproduction trajectories
are never silently reused. ``--check-only`` validates frozen files/signatures
without generating designs, loading coefficient arrays, warmup, fitting or writes.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter
import traceback
from zoneinfo import ZoneInfo

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from run_workshop import cache_guard, dump, run_trajectory, sha
from run_workshop_followup import followup_signature, followup_targets, followup_target_id
from lasso_study.workshop_data import make_workshop_design
from lasso_study.solvers import certificate, lipschitz_constant


TARGET_IDS = (
    "block_seed200_rho0_dispersed_strong",
    "toeplitz_seed200_rho0.9_clustered_mixed_weak",
)
METHODS = ("fista", "cd")
COEFFICIENT_TOL = 1e-7
OBJECTIVE_TOL = 1e-8


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contained_path(base, relative):
    """Resolve frozen relative artifact paths without escaping their stage."""
    base = Path(base).resolve()
    relative = Path(relative)
    require(not relative.is_absolute(), f"Artifact path is not relative: {relative}")
    resolved = (base / relative).resolve()
    require(resolved.is_relative_to(base), f"Artifact path escapes stage: {relative}")
    require(resolved.is_file(), f"Missing frozen artifact: {resolved}")
    return resolved


def preflight(config_path, source, output):
    """Only JSON/file hashing and dependency metadata; no numerical operations."""
    source, output = source.resolve(), output.resolve()
    require(source == (ROOT / "results/workshop/confirm").resolve(),
            "The reproduction source must be the registered confirmation stage")
    require(output.is_relative_to((ROOT / "results/workshop").resolve())
            and output != (ROOT / "results/workshop").resolve(),
            "Output must be a separate directory under results/workshop")
    protected = [(ROOT / "results/workshop" / stage).resolve()
                 for stage in ("smoke", "pilot", "confirm", "real")]
    require(not any(output == path or output.is_relative_to(path)
                    or path.is_relative_to(output) for path in protected),
            "Reproduction output overlaps a frozen or active study stage")
    require(not output.exists() or not any(output.iterdir()),
            "Reproduction output is not empty; select a fresh --output directory")

    config = read_json(config_path)
    require(config.get("stage") == "confirm", "Only the confirmation config can be reproduced")
    registered = {followup_target_id(params): params
                  for params in followup_targets(config, "confirm")}
    require(all(identifier in registered for identifier in TARGET_IDS),
            "The two predeclared targets are absent from the confirmation grid")
    signature, provenance = followup_signature(config_path)
    config_hash = sha(config_path)
    paths = [config_path, source / "environment.json", source / "completion.json",
             source / "config.json", source / "effective_config.json"]
    environment, completion = read_json(paths[1]), read_json(paths[2])
    require(read_json(paths[3]) == config and read_json(paths[4]) == config,
            "Frozen confirmation configuration snapshots differ from original config")
    for label, record in (("environment", environment), ("completion", completion)):
        require(record.get("numerical_signature") == signature
                and record.get("config_sha256") == config_hash,
                f"Confirmation {label} source/configuration signature differs from current followup_signature")
    require(completion.get("stage") == "confirm"
            and completion.get("recorded_targets") == completion.get("expected_targets") == len(registered)
            and completion.get("recorded_trajectories") == completion.get("expected_trajectories")
            == len(registered) * len(config["methods"]),
            "Confirmation is not complete; wait for its registered grid to finish")

    targets = []
    for identifier in TARGET_IDS:
        params = registered[identifier]
        dataset_path = source / "datasets" / (identifier + ".json")
        reference_path = source / "references" / (identifier + ".json")
        dataset, refs = read_json(dataset_path), read_json(reference_path)
        data_hash = dataset["dataset_hash"]
        for label, record in (("dataset", dataset), ("reference", refs)):
            cache_guard(record, signature, config_hash, data_hash)
            require(record.get("target_id") == identifier and record.get("params") == params,
                    f"Frozen {label} target identity differs: {identifier}")
        require(refs.get("reference_accepted") is True,
                f"The frozen numerical reference is uncertain: {identifier}")
        reference_coefficients = contained_path(source, refs["coefficient_path"])
        require(reference_coefficients.relative_to(source)
                == Path("reference_coefficients") / (identifier + ".npz"),
                "Unexpected reference coefficient location")
        require(sha(reference_coefficients) == refs["coefficient_sha256"],
                f"Frozen reference coefficient hash differs: {identifier}")
        runs = {}
        paths.extend((dataset_path, reference_path, reference_coefficients))
        for method in METHODS:
            run_id = identifier + "_" + method
            run_path = source / "runs" / (run_id + ".json")
            record = read_json(run_path)
            cache_guard(record, signature, config_hash, data_hash)
            require(record.get("target_id") == identifier and record.get("run_id") == run_id
                    and record.get("method") == method and record.get("params") == params,
                    f"Frozen trajectory identity differs: {run_id}")
            require(record.get("status") == "converged"
                    and record["final"]["normalized_gap"] <= config["solver"]["tol"],
                    f"Original representative trajectory did not converge: {run_id}")
            trajectory_path = contained_path(source, record["trajectory_path"])
            require(sha(trajectory_path) == record["trajectory_sha256"],
                    f"Frozen trajectory coefficient hash differs: {run_id}")
            runs[method] = {"record": record, "path": run_path, "arrays_path": trajectory_path}
            paths.extend((run_path, trajectory_path))
        targets.append({"identifier": identifier, "params": params, "dataset": dataset,
                        "dataset_path": dataset_path, "refs": refs, "reference_path": reference_path,
                        "reference_coefficients": reference_coefficients, "runs": runs})
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    return {"config": config, "signature": signature, "provenance": provenance,
            "config_hash": config_hash, "targets": targets, "inputs": inputs}


def copy_artifact(original, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original, destination)
    require(sha(original) == sha(destination), f"Copied reference artifact hash differs: {destination}")


def compare_run(data, original, reproduced, original_coefficients, new_coefficients, tol):
    require(original_coefficients.shape == new_coefficients.shape == (data["X"].shape[1],),
            "Final coefficient dimensions differ")
    require(np.isfinite(original_coefficients).all() and np.isfinite(new_coefficients).all(),
            "Non-finite final coefficient")
    coefficient_delta = float(np.max(np.abs(new_coefficients - original_coefficients)))
    objective_delta = abs(float(reproduced["final"]["primal"]) - float(original["final"]["primal"]))
    checked = certificate(data["X"], data["y"], new_coefficients, data["metadata"]["lambda"])
    stored_objective_delta = abs(checked["primal"] - reproduced["final"]["primal"])
    status_matches = reproduced["status"] == original["status"]
    passed = (coefficient_delta <= COEFFICIENT_TOL and objective_delta <= OBJECTIVE_TOL
              and status_matches and reproduced["status"] == "converged"
              and checked["normalized_gap"] <= tol
              and checked["dual_feasibility"] <= 1 + 1e-12
              and stored_objective_delta <= OBJECTIVE_TOL)
    return {"passed": bool(passed), "coefficient_linf_difference": coefficient_delta,
            "absolute_primal_difference": objective_delta,
            "normalized_primal_difference": objective_delta / max(1., abs(checked["primal"]),
                                                                      abs(original["final"]["primal"])),
            "recomputed_vs_saved_primal_difference": stored_objective_delta,
            "original_status": original["status"], "reproduced_status": reproduced["status"],
            "status_matches": status_matches,
            "original_final_primal": original["final"]["primal"],
            "reproduced_final_primal": reproduced["final"]["primal"],
            "original_final_normalized_gap": original["final"]["normalized_gap"],
            "reproduced_certificate": checked,
            "original_iterations": original["iterations"], "reproduced_iterations": reproduced["iterations"],
            "original_elapsed_sec": original["elapsed_sec"],
            "reproduced_elapsed_sec": reproduced["elapsed_sec"],
            "support_by_threshold": {
                str(tau): {"same_support": bool(np.array_equal(np.abs(original_coefficients) > tau,
                                                               np.abs(new_coefficients) > tau)),
                           "support_symmetric_difference": int(np.sum((np.abs(original_coefficients) > tau)
                                                                        != (np.abs(new_coefficients) > tau)))}
                for tau in original["support_thresholds"]},
            "interpretation": "Coefficient, objective, status and common certificate checks; timing equality is not required."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/workshop_confirm.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/workshop/reproduce")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config_path, source, output = args.config.resolve(), ROOT / "results/workshop/confirm", args.output.resolve()
    invocation_start = perf_counter()
    frozen = preflight(config_path, source, output)
    config, signature, config_hash = frozen["config"], frozen["signature"], frozen["config_hash"]
    if args.check_only:
        print(json.dumps({"preflight_passed": True, "targets": list(TARGET_IDS), "planned_new_trajectory_fits": 4,
                          "planned_new_reference_fits": 0, "numerical_signature": signature,
                          "config_sha256": config_hash, "data_generated": False,
                          "fits_run": 0, "result_files_written": 0}, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    report = {"stage": "reproduce", "source_stage": "confirm", "passed": False,
              "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "timezone": "Asia/Shanghai", "source_path": str(source.relative_to(ROOT)),
              "config_path": str(config_path.relative_to(ROOT)), "config_sha256": config_hash,
              "numerical_signature": signature, "source_signature": signature,
              "reproducer_script_sha256": sha(Path(__file__)),
              "expected_targets": 2, "expected_new_trajectory_fits": 4,
              "trajectory_fit_attempts": 0, "trajectory_fits_completed": 0,
              "new_reference_fits": 0, "reused_reference_targets": 0,
              "frozen_reference_solutions_reused": 0,
              "tolerances": {"coefficient_linf": COEFFICIENT_TOL,
                             "absolute_primal": OBJECTIVE_TOL,
                             "normalized_gap": config["solver"]["tol"]},
              "input_sha256": frozen["inputs"], "targets": [], "runs": [],
              "interpretation": ["Four fresh trajectory solves; the two frozen numerical references per target are reused, not refitted.",
                                 "Numerical agreement does not prove exact minimizer support or statistical recovery.",
                                 "Wall time includes regeneration, validation, trajectory diagnostics and artifact writes; it is not a fresh timing benchmark."]}
    try:
        dump(output / "config.json", config)
        dump(output / "effective_config.json", config)
        copy_artifact(config_path, output / "original_config.json")
        with threadpool_limits(limits=1):
            dump(output / "environment.json", dict(frozen["provenance"], stage="reproduce",
                 numerical_signature=signature, source_signature=signature, config_sha256=config_hash,
                 python=sys.version, platform=platform.platform(), threads=1, threadpools=threadpool_info(),
                 reproducer_script_sha256=report["reproducer_script_sha256"],
                 references_reused_without_fitting=True))
            for target in frozen["targets"]:
                identifier, params = target["identifier"], target["params"]
                target_start = perf_counter()
                setup_start = perf_counter()
                data = make_workshop_design(config, **params)
                data["X"] = np.asfortranarray(data["X"], dtype=np.float64)
                preprocessing_sec = perf_counter() - setup_start
                hashes = data["metadata"]["hashes"]
                data_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
                require(hashes == target["dataset"]["hashes"]
                        and data_hash == target["dataset"]["dataset_hash"],
                        f"Regenerated design hash differs: {identifier}")
                require(data["metadata"]["lambda"] == target["dataset"]["lambda"],
                        f"Regenerated lambda differs: {identifier}")
                with np.load(target["reference_coefficients"], allow_pickle=False) as arrays:
                    first, second = arrays["ref_sklearn"].copy(), arrays["ref_skglm"].copy()
                    require(np.array_equal(arrays["w_true"], data["w_true"]),
                            f"Frozen reference truth differs: {identifier}")
                require(first.shape == second.shape == data["w_true"].shape
                        and np.isfinite(first).all() and np.isfinite(second).all(),
                        f"Invalid frozen reference coefficient dimensions/values: {identifier}")
                copy_artifact(target["reference_path"], output / "references" / (identifier + ".json"))
                copy_artifact(target["reference_coefficients"],
                              output / target["refs"]["coefficient_path"])
                report["reused_reference_targets"] += 1
                report["frozen_reference_solutions_reused"] += 2
                setup_start = perf_counter()
                L = lipschitz_constant(data["X"])
                svd_sec = perf_counter() - setup_start
                require(abs(L - target["dataset"]["L"]) <= 1e-12 * max(1., abs(L)),
                        f"Regenerated Lipschitz constant differs: {identifier}")
                dump(output / "datasets" / (identifier + ".json"),
                     dict(data["metadata"], target_id=identifier, params=params,
                          numerical_signature=signature, config_sha256=config_hash, dataset_hash=data_hash,
                          preprocessing_sec=preprocessing_sec, L=L, lipschitz_setup_sec=svd_sec,
                          reproduction_of=str(target["dataset_path"].relative_to(ROOT))))
                for method in METHODS:
                    run_id = identifier + "_" + method
                    require(not (output / "runs" / (run_id + ".json")).exists()
                            and not (output / "trajectories" / (run_id + ".npz")).exists(),
                            f"Refusing cached reproduction instead of a fresh fit: {run_id}")
                    report["trajectory_fit_attempts"] += 1
                    fit_start = perf_counter()
                    reproduced = run_trajectory(data, config, output, identifier, params, method,
                                                target["refs"], first, second, L, svd_sec,
                                                signature, config_hash, data_hash)
                    call_wall_sec = perf_counter() - fit_start
                    report["trajectory_fits_completed"] += 1
                    with np.load(target["runs"][method]["arrays_path"], allow_pickle=False) as arrays:
                        old_coef = arrays["final_coef"].copy()
                    new_path = contained_path(output, reproduced["trajectory_path"])
                    require(sha(new_path) == reproduced["trajectory_sha256"],
                            f"New coefficient artifact hash differs: {run_id}")
                    with np.load(new_path, allow_pickle=False) as arrays:
                        new_coef = arrays["final_coef"].copy()
                    comparison = compare_run(data, target["runs"][method]["record"], reproduced,
                                             old_coef, new_coef, config["solver"]["tol"])
                    report["runs"].append(dict(comparison, target_id=identifier, method=method,
                         run_id=run_id, new_fit=True, trajectory_call_wall_sec=call_wall_sec,
                         original_run_path=str(target["runs"][method]["path"].relative_to(ROOT)),
                         reproduced_run_path=str((output / "runs" / (run_id + ".json")).relative_to(ROOT)),
                         original_trajectory_sha256=target["runs"][method]["record"]["trajectory_sha256"],
                         reproduced_trajectory_sha256=reproduced["trajectory_sha256"]))
                report["targets"].append({"target_id": identifier, "params": params,
                     "regenerated_hashes_match": True, "dataset_hash": data_hash,
                     "reference_coefficients_sha256": target["refs"]["coefficient_sha256"],
                     "references_reused_without_fitting": True, "L": L,
                     "preprocessing_sec": preprocessing_sec, "lipschitz_setup_sec": svd_sec,
                     "target_wall_sec": perf_counter() - target_start})
        signature_after, _ = followup_signature(config_path)
        report["source_signature_unchanged"] = signature_after == signature
        report["original_inputs_unchanged"] = all(sha(ROOT / name) == expected
                                                    for name, expected in frozen["inputs"].items())
        report["passed"] = (len(report["targets"]) == 2 and len(report["runs"]) == 4
                            and report["trajectory_fit_attempts"] == report["trajectory_fits_completed"] == 4
                            and report["new_reference_fits"] == 0
                            and all(run["passed"] for run in report["runs"])
                            and report["source_signature_unchanged"] and report["original_inputs_unchanged"])
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
    report["wall_sec"] = perf_counter() - invocation_start
    report["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    dump(output / "reproduction.json", report)
    print(json.dumps({"passed": report["passed"], "new_trajectory_fits": report["trajectory_fits_completed"],
                      "new_reference_fits": report["new_reference_fits"], "wall_sec": report["wall_sec"],
                      "report": str(output / "reproduction.json")}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
