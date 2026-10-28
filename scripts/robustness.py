#!/usr/bin/env python3
"""Check sample curvature and sensitivity to diagnostic checkpoint frequency.

Run after the main grid, with no competing timed workload:
    .venv/bin/python scripts/robustness.py

Use --structure-only or --diagnostics-only to run one part. This script does
not alter the numerical implementation or the frozen primary configuration.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from run_study import callback_for, dump, environment, numerical_signature, sha
from lasso_study.data import make_synthetic
from lasso_study.solvers import solve, warmup_cd


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(values)), "min": float(values.min()),
            "max": float(values.max()), "n": int(len(values))}


def synthetic_parameters(config):
    return {key: value for key, value in config["synthetic"].items()
            if key not in {"seeds", "rhos", "lambda_ratios"}}


def matrix_hash(data):
    return hashlib.sha256(data["X"].tobytes() + data["y"].tobytes()).hexdigest()


def checked_baseline(run_id, expected_config_sha, expected_matrix_hash=None):
    path = ROOT / "results/main/runs" / (run_id + ".json")
    if not path.is_file():
        raise RuntimeError(f"Main grid incomplete; missing {path}")
    saved = load_json(path)
    if saved["config_sha256"] != expected_config_sha:
        raise RuntimeError(f"Configuration differs from main-grid run: {run_id}")
    if saved.get("numerical_signature") != numerical_signature():
        raise RuntimeError(f"Numerical source differs from main-grid run: {run_id}")
    if expected_matrix_hash is not None and saved["dataset_hash"] != expected_matrix_hash:
        raise RuntimeError(f"Reconstructed data differ from main-grid run: {run_id}")
    return saved


def make_structure(config, config_sha):
    """One dense SVD for each of 20 independent problem designs."""
    rows = []
    base = synthetic_parameters(config)
    for seed in config["synthetic"]["seeds"]:
        for rho in config["synthetic"]["rhos"]:
            data = make_synthetic(seed=seed, rho=rho, **base)
            X = data["X"]
            problem_hash = matrix_hash(data)
            dataset_id = f"synthetic_s{seed}_rho{rho:g}"
            primary_metadata = load_json(ROOT / "results/main/datasets" / (dataset_id + ".json"))
            if problem_hash != primary_metadata["matrix_hash"]:
                raise RuntimeError(f"Reconstructed matrix differs: {dataset_id}")
            start = perf_counter()
            singular_values = np.linalg.svd(X, compute_uv=False)
            svd_sec = perf_counter() - start
            maximum_eigenvalue = float(singular_values[0] ** 2 / X.shape[0])
            minimum_eigenvalue = (float(singular_values[-1] ** 2 / X.shape[0])
                                 if X.shape[0] >= X.shape[1] else 0.0)
            condition = maximum_eigenvalue / minimum_eigenvalue if minimum_eigenvalue > 0 else None
            block_size = config["synthetic"]["block_size"]
            # Exactly one prespecified pair in every block, not a post-hoc
            # choice of the most correlated pair; this estimates mean rho.
            pairs = [(start, start + 1) for start in range(0, X.shape[1], block_size)]
            if block_size < 2:
                pairs = []
            correlations = [float(X[:, first] @ X[:, second] / X.shape[0])
                            for first, second in pairs]
            runtime = {}
            for method in config["methods"]:
                runs = []
                by_lambda = {}
                for ratio in config["synthetic"]["lambda_ratios"]:
                    run_id = f"{dataset_id}_a{ratio:g}_{method}"
                    saved = checked_baseline(run_id, config_sha, problem_hash)
                    runs.append(saved)
                    by_lambda[str(ratio)] = {
                        "run_id": run_id, "elapsed_sec": saved["elapsed_sec"],
                        "cold_start_sec": saved["cold_start_sec"],
                        "diagnostic_fraction": saved["diagnostics_sec"] / saved["elapsed_sec"],
                        "iterations": saved["iterations"], "status": saved["status"],
                        "normalized_gap": saved["history"][-1]["normalized_gap"],
                    }
                runtime[method] = {
                    "elapsed_sec_across_lambda": summary([run["elapsed_sec"] for run in runs]),
                    "status_counts": dict(Counter(run["status"] for run in runs)),
                    "by_lambda_ratio": by_lambda,
                }
            row = {
                "dataset_id": dataset_id, "seed": seed, "rho": rho,
                "matrix_hash": problem_hash, "n": X.shape[0], "p": X.shape[1],
                "sample_hessian_condition": condition,
                "sample_hessian_min_eigenvalue": minimum_eigenvalue,
                "sample_hessian_max_eigenvalue": maximum_eigenvalue,
                "solver_L_with_roundoff_margin": maximum_eigenvalue * (1.0 + 1e-12),
                "primary_L": primary_metadata["L"],
                "population_block_condition": (1 + (block_size - 1) * rho) / (1 - rho),
                "within_block_correlation": {
                    "sampling_rule": "first two columns of every block; no cross-block pairs",
                    "pair_count": len(pairs), "pairs": pairs,
                    "correlations": correlations,
                    "mean": float(np.mean(correlations)) if correlations else None,
                    "min": float(np.min(correlations)) if correlations else None,
                    "max": float(np.max(correlations)) if correlations else None,
                },
                "svd_sec": svd_sec, "algorithm_runtime": runtime,
            }
            rows.append(row)
            print(json.dumps({"structure": dataset_id, "condition": condition,
                              "sample_mean_rho": row["within_block_correlation"]["mean"]}), flush=True)
    grouped = []
    for rho in config["synthetic"]["rhos"]:
        matching = [row for row in rows if row["rho"] == rho]
        grouped.append({"rho": rho,
                        "sample_hessian_condition": summary([row["sample_hessian_condition"] for row in matching]),
                        "sample_hessian_min_eigenvalue": summary([row["sample_hessian_min_eigenvalue"] for row in matching]),
                        "sample_hessian_max_eigenvalue": summary([row["sample_hessian_max_eigenvalue"] for row in matching]),
                        "sample_within_block_correlation": summary([row["within_block_correlation"]["mean"] for row in matching])})
    output = {
        "config_sha256": config_sha, "numerical_signature": numerical_signature(),
        "definitions": {
            "sample_hessian": "H=X.T@X/n after training-column standardization",
            "sample_hessian_condition": "lambda_max(H)/lambda_min(H)=s_max(X)^2/s_min(X)^2",
            "population_block_condition": "(1+(block_size-1)*rho)/(1-rho), before sample standardization",
            "solver_L": "sample lambda_max(H) times (1+1e-12)",
            "runtime": "measured main-grid solver times, joined by exact data/config/source hashes",
            "interpretation": "Population and finite-sample condition numbers are distinct. Runtime summaries across lambda are descriptive; use by_lambda_ratio for controlled comparisons.",
        },
        "datasets": rows, "by_rho": grouped,
    }
    dump(ROOT / "results/structure.json", output)
    return output


def compact_run(run):
    last = run["history"][-1]
    return {"status": run["status"], "elapsed_sec": run["elapsed_sec"],
            "cold_start_sec": run["cold_start_sec"], "iterations": run["iterations"],
            "diagnostics_sec": run["diagnostics_sec"],
            "diagnostic_fraction": run["diagnostics_sec"] / run["elapsed_sec"],
            "primal": last["primal"], "normalized_gap": last["normalized_gap"],
            "relative_coef_error": last["relative_coef_error"], "f1": last["f1"],
            "threshold_hits": run["threshold_hits"]}


def make_diagnostic_sensitivity(config, config_sha, sparse_check_every):
    """Rerun 12 representative solves with fewer checkpoints and matched data."""
    baseline_check_every = config["solver"]["check_every"]
    if sparse_check_every <= baseline_check_every:
        raise ValueError("Sparse check frequency must exceed the baseline check_every")
    warmup_cd()
    comparisons = []
    solver_config = dict(config["solver"], check_every=sparse_check_every)
    destination = ROOT / "results/robustness/check20_runs"
    if sparse_check_every != 20:
        destination = ROOT / "results/robustness" / f"check{sparse_check_every}_runs"
    base = synthetic_parameters(config)
    for rho in (0.0, 0.9):
        data = make_synthetic(seed=0, rho=rho, **base)
        data["X"] = np.asfortranarray(data["X"])
        problem_hash = matrix_hash(data)
        dataset_id = f"synthetic_s0_rho{rho:g}"
        primary_metadata = load_json(ROOT / "results/main/datasets" / (dataset_id + ".json"))
        if problem_hash != primary_metadata["matrix_hash"]:
            raise RuntimeError(f"Reconstructed matrix differs: {dataset_id}")
        L = primary_metadata["L"]
        for ratio in (0.01, 0.1):
            # Preserve the primary randomized algorithm order for these tasks.
            order = np.random.default_rng(int(rho * 100) + int(ratio * 1000)).permutation(config["methods"])
            for method in order:
                run_id = f"{dataset_id}_a{ratio:g}_{method}"
                baseline = checked_baseline(run_id, config_sha, problem_hash)
                path = destination / (run_id + ".json")
                if path.is_file():
                    sparse = load_json(path)
                    if (sparse.get("config_sha256") != config_sha
                            or sparse.get("numerical_signature") != numerical_signature()
                            or sparse.get("dataset_hash") != problem_hash
                            or sparse.get("solver_config") != solver_config):
                        raise RuntimeError(f"Refusing to reuse mismatched robustness run: {path}")
                else:
                    sparse = solve(data["X"], data["y"], baseline["lambda"], method,
                                   L=L, metric_callback=callback_for(data, config), **solver_config)
                    sparse.pop("coef")
                    sparse.update(run_id=run_id, config_sha256=config_sha,
                                  numerical_signature=numerical_signature(), dataset_hash=problem_hash,
                                  rho=rho, seed=0, lambda_ratio=ratio, method=str(method),
                                  **{"lambda": baseline["lambda"]}, solver_config=solver_config,
                                  cold_start_sec=sparse["elapsed_sec"] + (
                                      baseline["lipschitz_setup_sec"] if method != "cd" else 0))
                    sparse["threshold_hits"] = {
                        str(tol): next((row for row in sparse["history"] if row["normalized_gap"] <= tol), None)
                        for tol in config["thresholds"]}
                    dump(path, sparse)
                comparisons.append({
                    "run_id": run_id, "seed": 0, "rho": rho, "lambda_ratio": ratio,
                    "method": str(method),
                    "baseline": compact_run(baseline), "sparse": compact_run(sparse),
                    "elapsed_ratio_sparse_over_baseline": sparse["elapsed_sec"] / baseline["elapsed_sec"],
                    "final_primal_difference": sparse["history"][-1]["primal"] - baseline["history"][-1]["primal"],
                    "iteration_difference": sparse["iterations"] - baseline["iterations"],
                })
                print(json.dumps({"diagnostics": run_id, "status": sparse["status"],
                                  "sparse_sec": sparse["elapsed_sec"],
                                  "baseline_sec": baseline["elapsed_sec"]}), flush=True)
    rankings = []
    for rho in (0.0, 0.9):
        for ratio in (0.01, 0.1):
            matching = [row for row in comparisons if row["rho"] == rho and row["lambda_ratio"] == ratio]
            for tol in config["thresholds"]:
                key = str(tol)
                comparable = all(row[frequency]["threshold_hits"][key] is not None
                                 for row in matching for frequency in ("baseline", "sparse"))
                baseline_ranking = sparse_ranking = None
                if comparable:
                    baseline_ranking = [row["method"] for row in sorted(
                        matching, key=lambda row: row["baseline"]["threshold_hits"][key]["time"])]
                    sparse_ranking = [row["method"] for row in sorted(
                        matching, key=lambda row: row["sparse"]["threshold_hits"][key]["time"])]
                rankings.append({"rho": rho, "lambda_ratio": ratio, "normalized_gap_target": tol,
                                 "comparable": comparable, "baseline_ranking": baseline_ranking,
                                 "sparse_ranking": sparse_ranking,
                                 "rankings_match": baseline_ranking == sparse_ranking if comparable else None})
    comparable_rankings = [row for row in rankings if row["comparable"]]
    final_rankings = [row for row in comparable_rankings if row["normalized_gap_target"] == config["solver"]["tol"]]
    output = {
        "config_sha256": config_sha, "numerical_signature": numerical_signature(),
        "baseline_check_every": baseline_check_every, "sparse_check_every": sparse_check_every,
        "solver_settings_except_check_frequency": config["solver"],
        "timing_interpretation": "All times include diagnostics and recovery/prediction callbacks. Threshold times are the first checked values below the target, hence sampling upper bounds; no crossing-time interpolation is used.",
        "limitations": "One extra timing per representative problem; no timing confidence interval. A changed ranking can reflect checkpoint granularity, changed stopping iteration, or timing variability. These are 12 solves on seed 0, not a rerun of all 180 primary solves.",
        "comparisons": comparisons, "threshold_rankings": rankings,
        "comparable_ranking_count": len(comparable_rankings),
        "matching_ranking_count": sum(row["rankings_match"] for row in comparable_rankings),
        "comparable_final_tolerance_count": len(final_rankings),
        "matching_final_tolerance_count": sum(row["rankings_match"] for row in final_rankings),
    }
    dump(ROOT / "results/diagnostic_sensitivity.json", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/study.json")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--structure-only", action="store_true")
    modes.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--check-every", type=int, default=20)
    args = parser.parse_args()
    config = load_json(args.config)
    config_sha = sha(args.config)
    with threadpool_limits(limits=1):
        dump(ROOT / "results/robustness/environment.json", environment())
        started = perf_counter()
        if not args.diagnostics_only:
            make_structure(config, config_sha)
        if not args.structure_only:
            make_diagnostic_sensitivity(config, config_sha, args.check_every)
        dump(ROOT / "results/robustness/timing.json", {"wall_sec": perf_counter() - started,
             "structure_requested": not args.diagnostics_only,
             "diagnostics_requested": not args.structure_only})


if __name__ == "__main__":
    main()
