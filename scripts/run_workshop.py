#!/usr/bin/env python3
"""Independent workshop pilot with two numerical references and saved iterates.

Original course/v2 numerical modules and artifacts are read, never rewritten.
Reference agreement is numerical evidence, not proof of exact Lasso support.
Gap Safe is an offline zero-exclusion diagnostic, not a screened solver.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback
import warnings
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lasso_study.workshop_data import make_workshop_design
from lasso_study.solvers import certificate, lipschitz_constant, solve, warmup_cd
from lasso_study.modern_solver import solve_modern, warmup_modern


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_signature():
    names = ["data.py", "data_v2.py", "workshop_data.py", "solvers.py", "cd_kernel.c", "modern_solver.py"]
    hashes = {f"src/lasso_study/{name}": sha(ROOT / "src/lasso_study" / name) for name in names}
    hashes["scripts/run_workshop.py"] = sha(Path(__file__))
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "scikit-learn", "skglm", "numba", "llvmlite", "threadpoolctl")}
    payload = {"source_hashes": hashes, "dependency_versions": versions}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def support_counts(selected, truth):
    tp = int(np.sum(selected & truth)); fp = int(np.sum(selected & ~truth)); fn = int(np.sum(~selected & truth))
    precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "nnz": int(selected.sum()),
            "precision": precision, "recall": recall, "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "support_error": fp + fn, "exact_support": bool(np.array_equal(selected, truth))}


def error_decomposition(selected, reference, truth):
    """Binary accounting relative to an explicitly numerical reference."""
    current, statistical = support_counts(selected, truth), support_counts(reference, truth)
    harm_fp = int(np.sum(selected & ~reference & ~truth))
    harm_fn = int(np.sum(~selected & reference & truth))
    repair_fp = int(np.sum(~selected & reference & ~truth))
    repair_fn = int(np.sum(selected & ~reference & truth))
    harm, repair = harm_fp + harm_fn, repair_fp + repair_fn
    numerical = int(np.sum(selected != reference))
    if current["support_error"] != statistical["support_error"] + harm - repair or numerical != harm + repair:
        raise AssertionError("Binary support-error identity failed")
    return dict(current, reference_support_error=statistical["support_error"], reference_f1=statistical["f1"],
                numerical_support_error=numerical, numerical_extra=int(np.sum(selected & ~reference)),
                numerical_missing=int(np.sum(~selected & reference)), numerical_harm=harm, numerical_repair=repair,
                numerical_harm_fp=harm_fp, numerical_harm_fn=harm_fn,
                numerical_repair_fp=repair_fp, numerical_repair_fn=repair_fn,
                same_support_as_reference=numerical == 0,
                same_recovery_counts_as_reference=all(current[name] == statistical[name] for name in ("tp", "fp", "fn")))


def reference_agreement(first, second, first_coef, second_coef, settings, taus):
    by_tau = {}
    for tau in taus:
        left, right = np.abs(first_coef) > tau, np.abs(second_coef) > tau
        left_sign = np.where(left, np.sign(first_coef), 0.)
        right_sign = np.where(right, np.sign(second_coef), 0.)
        by_tau[str(tau)] = {"same_support": bool(np.array_equal(left, right)),
                            "support_symmetric_difference": int(np.sum(left != right)),
                            "same_thresholded_signs": bool(np.array_equal(left_sign, right_sign)),
                            "sklearn_nnz": int(left.sum()), "skglm_nnz": int(right.sum())}
    left, right = first["certificate"], second["certificate"]
    coef_delta = float(np.max(np.abs(first_coef - second_coef)))
    objective_delta = abs(left["primal"] - right["primal"]) / max(1., abs(left["primal"]), abs(right["primal"]))
    both_accurate = all(run["certificate"]["normalized_gap"] <= settings["normalized_gap_tol"]
                        and run["certificate"]["dual_feasibility"] <= 1 + 1e-12 for run in (first, second))
    both_within_budget = all(run["status"] == "converged" for run in (first, second))
    accepted = (both_accurate and both_within_budget
                and coef_delta <= settings["coef_linf_agreement_tol"]
                and objective_delta <= settings["normalized_objective_agreement_tol"]
                and all(row["same_support"] and row["same_thresholded_signs"] for row in by_tau.values()))
    return {"reference_accepted": bool(accepted),
            "status": "numerical_reference_agreement" if accepted else "reference_uncertain",
            "both_pass_common_precision": both_accurate, "both_within_observed_budget": both_within_budget,
            "coefficient_linf_difference": coef_delta, "normalized_primal_difference": objective_delta,
            "by_support_threshold": by_tau,
            "interpretation": "Agreement of two high-accuracy numerical implementations; not a proof of exact minimizer support or statistical recovery."}


def reference_geometry(X, y, coef, lam, taus):
    correlations = X.T @ (y - X @ coef) / len(y)
    output = {}
    for tau in taus:
        active = np.abs(coef) > tau
        if active.any():
            gram = X[:, active].T @ X[:, active] / len(y)
            eigenvalues = np.linalg.eigvalsh(gram)
            rank_cutoff = np.finfo(float).eps * max(gram.shape) * max(1., float(np.max(np.abs(eigenvalues))))
            geometry = {"active_gram_min_eigenvalue": float(eigenvalues[0]),
                        "active_gram_max_eigenvalue": float(eigenvalues[-1]),
                        "active_gram_numerical_rank": int(np.sum(eigenvalues > rank_cutoff)),
                        "active_gram_rank_cutoff": rank_cutoff,
                        "active_gram_condition": float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > rank_cutoff else None,
                        "beta_min": float(np.min(np.abs(coef[active]))),
                        "active_kkt_violation": float(np.max(np.abs(correlations[active] - lam * np.sign(coef[active]))))}
        else:
            geometry = {"active_gram_min_eigenvalue": None, "active_gram_max_eigenvalue": None,
                        "active_gram_numerical_rank": 0, "active_gram_rank_cutoff": None,
                        "active_gram_condition": None, "beta_min": None, "active_kkt_violation": 0.}
        geometry.update(nnz=int(active.sum()),
                        inactive_kkt_margin=float(lam - np.max(np.abs(correlations[~active]))) if (~active).any() else None)
        output[str(tau)] = geometry
    return output


def gap_safe_zero_mask(X, y, coef, lam, absolute_gap, column_norms=None):
    """Numerical Gap Safe sphere test for zero exclusion, not full support.

    theta=r/(n*scale), ||theta-theta*|| <= sqrt(2*gap/n),
    so |X_j.T theta| + ||X_j|| sqrt(2*gap/n) < lambda excludes j.
    A roundoff guard is conservative but is not interval-arithmetic validation.
    """
    residual = y - X @ coef
    correlations = X.T @ residual / len(y)
    scale = max(1., float(np.max(np.abs(correlations))) / lam) * (1. + 4e-15)
    norms = np.linalg.norm(X, axis=0) if column_norms is None else column_norms
    radius = np.sqrt(2 * max(0., absolute_gap) / len(y))
    guard = 64 * np.finfo(float).eps * max(1., lam, float(np.max(np.abs(correlations))))
    return np.abs(correlations) / scale + norms * radius < lam - guard


def target_parameters(config, stage):
    if stage == "smoke":
        return list(config["smoke"]["targets"])
    targets = [dict(seed=seed, family=geometry["family"], rho=geometry["rho"], layout=layout, amplitude=amplitude)
               for seed, geometry, layout, amplitude in product(config["seeds"], config["geometries"],
                   config["layouts"], config["amplitudes"])]
    keys = {(row["seed"], row["family"], row["rho"], row["layout"], row["amplitude"]) for row in targets}
    if len(keys) != len(targets):
        raise ValueError("Duplicate parameter cells in the registered workshop grid")
    geometries = {(row["family"], row["rho"]) for row in config["geometries"]}
    if ("block", 0.) in geometries and ("toeplitz", 0.) in geometries:
        raise ValueError("Identical rho=0 families must share one baseline, not duplicate targets")
    if len(targets) != config["expected_pilot_targets"]:
        raise ValueError("Registered target count disagrees with the geometry grid")
    if len(targets) * len(config["methods"]) != config["expected_pilot_trajectories"]:
        raise ValueError("Registered trajectory count disagrees with the geometry grid")
    return targets


def target_id(params):
    return f"{params['family']}_seed{params['seed']}_rho{params['rho']:g}_{params['layout']}_{params['amplitude']}"


def cache_guard(saved, source_sig, config_hash, data_hash):
    if (saved.get("numerical_signature") != source_sig or saved.get("config_sha256") != config_hash
            or saved.get("dataset_hash") != data_hash):
        raise RuntimeError("Cached workshop result has different source/configuration/data")


def fit_references(data, config, output, identifier, params, source_sig, config_hash, data_hash):
    path = output / "references" / (identifier + ".json")
    coefficients_path = output / "reference_coefficients" / (identifier + ".npz")
    if path.exists():
        saved = json.loads(path.read_text())
        cache_guard(saved, source_sig, config_hash, data_hash)
        if sha(coefficients_path) != saved["coefficient_sha256"]:
            raise RuntimeError("Reference coefficient artifact hash mismatch")
        with np.load(coefficients_path, allow_pickle=False) as arrays:
            return saved, arrays["ref_sklearn"].copy(), arrays["ref_skglm"].copy()
    X, y = data["X"], data["y"]
    lam = data["metadata"]["lambda"]
    settings = config["references"]
    start = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        model = Lasso(alpha=lam, fit_intercept=False, tol=settings["sklearn_tol"],
                      max_iter=settings["sklearn_max_iter"], selection="cyclic").fit(X, y)
    first_coef = model.coef_.copy()
    cert = certificate(X, y, first_coef, lam)
    elapsed = perf_counter() - start
    first_passed = cert["normalized_gap"] <= settings["normalized_gap_tol"]
    first = {"method": "sklearn_cyclic_cd", "certificate": cert, "elapsed_sec": elapsed,
             "certificate_passed": first_passed, "budget_exceeded": elapsed > settings["time_limit"],
             "status": "budget_exceeded" if elapsed > settings["time_limit"] else "converged" if first_passed else "not_converged",
             "iterations": int(model.n_iter_), "iteration_unit": "complete sklearn coordinate epochs",
             "library_tol": settings["sklearn_tol"], "library_max_iter": settings["sklearn_max_iter"],
             "time_limit_mode": "observed after return; no interruption within sklearn fit",
             "warnings": [str(item.message) for item in caught]}
    second = solve_modern(X, y, lam, tol=settings["normalized_gap_tol"],
                          time_limit=settings["time_limit"], **settings["modern"])
    second_coef = second.pop("coef")
    agreement = reference_agreement(first, second, first_coef, second_coef, settings, config["support_thresholds"])
    geometry_start = perf_counter()
    geometry = {"sklearn": reference_geometry(X, y, first_coef, lam, config["support_thresholds"]),
                "skglm": reference_geometry(X, y, second_coef, lam, config["support_thresholds"])}
    geometry_sec = perf_counter() - geometry_start
    saved = dict(agreement, target_id=identifier, params=params, **params,
                 numerical_signature=source_sig, source_signature=source_sig,
                 config_sha256=config_hash, dataset_hash=data_hash, **{"lambda": lam},
                 sklearn=first, skglm=second, geometry=geometry,
                 geometry_diagnostics_sec_outside_fit_timers=geometry_sec,
                 reported_reference_calls=2, actual_library_fit_attempts=1 + second["attempt_count"])
    write_start = perf_counter()
    coefficients_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(coefficients_path, ref_sklearn=first_coef, ref_skglm=second_coef, w_true=data["w_true"])
    saved.update(coefficient_path=str(coefficients_path.relative_to(output)), coefficient_sha256=sha(coefficients_path),
                 coefficient_write_sec=perf_counter() - write_start)
    dump(path, saved)
    print(json.dumps({"reference": identifier, "status": saved["status"],
                      "sklearn_gap": cert["normalized_gap"],
                      "skglm_gap": second["certificate"]["normalized_gap"]}), flush=True)
    return saved, first_coef, second_coef


def run_trajectory(data, config, output, identifier, params, method, refs, first_coef, second_coef,
                   L, svd_sec, source_sig, config_hash, data_hash):
    run_id = identifier + "_" + method
    path = output / "runs" / (run_id + ".json")
    arrays_path = output / "trajectories" / (run_id + ".npz")
    if path.exists():
        saved = json.loads(path.read_text())
        cache_guard(saved, source_sig, config_hash, data_hash)
        if sha(arrays_path) != saved["trajectory_sha256"]:
            raise RuntimeError("Stored trajectory hash mismatch")
        return saved
    snapshots = []
    callback_total_sec = callback_snapshot_sec = 0.
    truth = data["w_true"]
    true_mask = np.abs(truth) > 0
    accepted = refs["reference_accepted"]

    def callback(w):
        nonlocal callback_total_sec, callback_snapshot_sec
        start = perf_counter()
        copy_start = perf_counter()
        snapshots.append(w.copy())
        callback_snapshot_sec += perf_counter() - copy_start
        by_tau = {}
        for tau in config["support_thresholds"]:
            selected = np.abs(w) > tau
            counts = support_counts(selected, true_mask)
            counts["exact_sign"] = bool(np.array_equal(np.where(selected, np.sign(w), 0), np.sign(truth)))
            if accepted:
                counts.update(error_decomposition(selected, np.abs(first_coef) > tau, true_mask))
            by_tau[str(tau)] = counts
        result = dict(by_tau[str(config["support_threshold"])], support_by_threshold=by_tau,
                      relative_coef_error=float(np.linalg.norm(w - truth) / np.linalg.norm(truth)))
        callback_total_sec += perf_counter() - start
        return result

    result = solve(data["X"], data["y"], data["metadata"]["lambda"], method,
                   L=L, metric_callback=callback, **config["solver"])
    final_coef = result.pop("coef")
    if len(result["history"]) != len(snapshots) or not np.array_equal(final_coef, snapshots[-1]):
        raise AssertionError("Stored coefficient snapshots do not align with solver history")
    post_start = perf_counter()
    matrix = np.stack(snapshots)
    norms = np.linalg.norm(data["X"], axis=0)
    union = np.zeros(matrix.shape[1], dtype=bool)
    safe_masks = []
    for i, (w, row) in enumerate(zip(matrix, result["history"])):
        safe = gap_safe_zero_mask(data["X"], data["y"], w, data["metadata"]["lambda"], row["gap"], norms)
        union |= safe
        safe_masks.append(np.packbits(union))
        row.update(checkpoint_index=i, gap_safe_zero_count_current=int(safe.sum()),
                   gap_safe_zero_count=int(union.sum()))
        if accepted:
            reference_selected = np.abs(first_coef) > config["support_threshold"]
            row["gap_safe_excluded_reference_nonzero_count"] = int(np.sum(union & reference_selected))
            n_reference_zero = int(np.sum(~reference_selected))
            row["gap_safe_reference_zero_coverage"] = (int(np.sum(union & ~reference_selected)) / n_reference_zero
                                                       if n_reference_zero else None)
    post_sec = perf_counter() - post_start
    result.update(target_id=identifier, run_id=run_id, params=params, **params, method=method,
                  numerical_signature=source_sig, source_signature=source_sig, config_sha256=config_hash,
                  dataset_hash=data_hash, **{"lambda": data["metadata"]["lambda"]},
                  n=len(data["y"]), p=data["X"].shape[1],
                  iteration_unit="complete cyclic epoch" if method == "cd" else "proximal iteration",
                  support_threshold=config["support_threshold"], support_thresholds=config["support_thresholds"],
                  reference_info={"target_id": identifier, "reference_accepted": accepted, "status": refs["status"],
                                  "path": f"references/{identifier}.json",
                                  "interpretation": refs["interpretation"]},
                  lipschitz_setup_sec=svd_sec,
                  cold_start_sec=result["elapsed_sec"] + (svd_sec if method == "fista" else 0),
                  callback_total_sec_within_solver_timer=callback_total_sec,
                  callback_snapshot_copy_sec_within_solver_timer=callback_snapshot_sec,
                  offline_gap_safe_and_snapshot_stack_sec=post_sec,
                  gap_safe_scope="offline cumulative zero-exclusion only; solver updates are not screened; not complete-support identification or interval-arithmetic certification",
                  threshold_curve_comparable=True)
    result["threshold_hits"] = {str(t): next((row for row in result["history"] if row["normalized_gap"] <= t), None)
                                for t in config["gap_thresholds"]}
    result["final"] = result["history"][-1]
    write_start = perf_counter()
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, coef_history=matrix,
                        iterations=np.asarray([row["iteration"] for row in result["history"]], dtype=np.int64),
                        times=np.asarray([row["time"] for row in result["history"]], dtype=np.float64),
                        final_coef=final_coef, w_true=truth, ref_sklearn=first_coef, ref_skglm=second_coef,
                        support_masks_packed=np.packbits(np.abs(matrix) > config["support_threshold"], axis=1),
                        negative_sign_masks_packed=np.packbits(matrix < -config["support_threshold"], axis=1),
                        gap_safe_zero_masks_packed=np.stack(safe_masks))
    result.update(trajectory_path=str(arrays_path.relative_to(output)), trajectory_sha256=sha(arrays_path),
                  trajectory_bytes=arrays_path.stat().st_size, artifact_write_sec_outside_solver_timer=perf_counter() - write_start,
                  packed_mask_bitorder="big; use np.unpackbits(..., axis=1)[:, :p]")
    dump(path, result)
    print(json.dumps({"run": run_id, "status": result["status"], "gap": result["final"]["normalized_gap"],
                      "sec": result["elapsed_sec"], "checkpoints": len(snapshots)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "pilot"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/workshop_pilot.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["threads"] != 1:
        raise ValueError("The pilot protocol requires exactly one numerical thread")
    effective = dict(config)
    if args.stage == "smoke":
        effective.update({name: value for name, value in config["smoke"].items() if name != "targets"})
    output = ROOT / "results/workshop" / args.stage
    output.mkdir(parents=True, exist_ok=True)
    config_hash = sha(args.config)
    with threadpool_limits(limits=1):
        start = perf_counter(); warmup_cd(); cd_setup = perf_counter() - start
        modern_setup = warmup_modern()
        source_sig, provenance = source_signature()
        env_path = output / "environment.json"
        if env_path.exists():
            previous = json.loads(env_path.read_text())
            if previous["numerical_signature"] != source_sig or previous["config_sha256"] != config_hash:
                raise RuntimeError("Existing workshop stage uses different source/configuration; select a separate stage directory")
        else:
            dump(env_path, dict(provenance, numerical_signature=source_sig, source_signature=source_sig,
                                config_sha256=config_hash, stage=args.stage,
                                started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), timezone="Asia/Shanghai",
                                python=sys.version, platform=platform.platform(), threads=1, threadpools=threadpool_info(),
                                compiler=subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0]))
        dump(output / "config.json", config)
        dump(output / "effective_config.json", effective)
        # Keep the original setup evidence on a cache-only invocation.
        if not (output / "warmup.json").exists():
            dump(output / "warmup.json", {"cd_setup_sec": cd_setup, "modern": modern_setup})
        stage_start = perf_counter(); runs = []; references = []; designs = target_parameters(config, args.stage)
        for params in designs:
            identifier = target_id(params)
            try:
                setup = perf_counter()
                data = make_workshop_design(effective, **params)
                data["X"] = np.asfortranarray(data["X"], dtype=np.float64)
                preprocessing_sec = perf_counter() - setup
                hashes = data["metadata"]["hashes"]
                data_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
                setup = perf_counter(); L = lipschitz_constant(data["X"]); svd_sec = perf_counter() - setup
                ds_path = output / "datasets" / (identifier + ".json")
                if ds_path.exists():
                    cache_guard(json.loads(ds_path.read_text()), source_sig, config_hash, data_hash)
                else:
                    dump(ds_path, dict(data["metadata"], target_id=identifier, params=params,
                                       numerical_signature=source_sig, config_sha256=config_hash, dataset_hash=data_hash,
                                       preprocessing_sec=preprocessing_sec, L=L, lipschitz_setup_sec=svd_sec))
                ref, first, second = fit_references(data, effective, output, identifier, params, source_sig, config_hash, data_hash)
                references.append(ref)
                order = np.random.default_rng(params["seed"] + int(params["rho"] * 100)
                                             + (1000 if params["layout"] == "clustered" else 0)).permutation(config["methods"])
                for method in order:
                    runs.append(run_trajectory(data, effective, output, identifier, params, str(method), ref, first, second,
                                               L, svd_sec, source_sig, config_hash, data_hash))
            except Exception as error:
                dump(output / "failures" / (identifier + ".json"),
                     dict(params, target_id=identifier, error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc(), cumulative_stage_wall_sec=perf_counter() - stage_start,
                          numerical_signature=source_sig, config_sha256=config_hash))
                raise
        dump(output / "completion.json", {
            "stage": args.stage, "numerical_signature": source_sig, "config_sha256": config_hash,
            "expected_targets": len(designs), "recorded_targets": len(references),
            "expected_trajectories": len(designs) * len(config["methods"]), "recorded_trajectories": len(runs),
            "expected_reference_calls": len(designs) * 2, "reference_calls": len(references) * 2,
            "registered_geometries": config["geometries"],
            "rho_zero_handling": config["rho_zero_handling"],
            "actual_reference_library_fit_attempts": sum(ref["actual_library_fit_attempts"] for ref in references),
            "reference_accepted_targets": sum(ref["reference_accepted"] for ref in references),
            "trajectory_status_counts": {status: sum(run["status"] == status for run in runs) for status in sorted({run["status"] for run in runs})},
            "wall_sec_this_invocation": perf_counter() - stage_start,
            "observational_scope": "Each data seed is a replication unit. Checkpoints are repeated observations on one trajectory, not independent samples."})


if __name__ == "__main__":
    main()
