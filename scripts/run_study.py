#!/usr/bin/env python3
"""Run pilot, main grid, real-data validation, smoke test or reproduction."""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lasso_study.data import make_synthetic, make_real
from lasso_study.solvers import solve, certificate, lipschitz_constant, warmup_cd


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for folder in ("src", "scripts", "configs")
            for p in sorted((ROOT / folder).rglob("*")) if p.is_file() and p.suffix in (".py", ".json", ".c")}


def numerical_signature():
    paths = [ROOT / "src/lasso_study" / name for name in ("data.py", "solvers.py", "cd_kernel.c")]
    paths.append(Path(__file__).resolve())
    return hashlib.sha256("".join(sha(p) for p in paths).encode()).hexdigest()


def environment():
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "matplotlib", "pandas", "threadpoolctl"):
        versions[package] = importlib.metadata.version(package)
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), "unknown")
    return {"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "timezone": "Asia/Shanghai", "python": sys.version, "platform": platform.platform(),
            "cpu": cpu, "logical_cpus": os.cpu_count(), "threads": 1,
            "threadpools": threadpool_info(), "versions": versions, "source_hashes": source_hashes(),
            "cd_kernel": "C, gcc -O3, ctypes; one call per complete cyclic sweep",
            "compiler": subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0],
            "numerical_signature": numerical_signature()}


def support(w, truth, threshold):
    selected, actual = np.abs(w) > threshold, np.abs(truth) > 0
    tp = int(np.sum(selected & actual))
    precision = tp / max(1, int(selected.sum()))
    recall = tp / max(1, int(actual.sum()))
    f1 = 2 * precision * recall / max(1e-300, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "nnz": int(selected.sum())}


def callback_for(data, config):
    if "w_true" in data:
        truth = data["w_true"]
        def callback(w):
            values = support(w, truth, config["support_threshold"])
            values.update(relative_coef_error=float(np.linalg.norm(w - truth) / np.linalg.norm(truth)),
                          test_mse=float(np.mean((data["X_test"] @ w - data["y_test"]) ** 2)))
            return values
        return callback
    return lambda w: {"nnz": int(np.sum(np.abs(w) > config["support_threshold"])),
                      "validation_mse": float(np.mean((data["X_val"] @ w - data["y_val"]) ** 2))}


def reference(X, y, lam):
    start = perf_counter()
    model = Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=200000,
                  selection="cyclic").fit(X, y)
    values = certificate(X, y, model.coef_, lam)
    if values["normalized_gap"] > 1e-8 or values["dual_feasibility"] > 1 + 1e-12:
        raise RuntimeError("Reference solver did not obtain a sufficiently accurate feasible certificate")
    values.update(elapsed_sec=perf_counter() - start, iterations=int(model.n_iter_),
                  sklearn_dual_gap=float(model.dual_gap_))
    return values


def run_one(data, config, destination, run_id, details, L, setup_sec, ref):
    path = destination / "runs" / (run_id + ".json")
    if path.exists():
        saved = json.loads(path.read_text())
        if (saved["config_sha256"] != details["config_sha256"] or saved["dataset_hash"] != details["dataset_hash"]
                or saved.get("numerical_signature") != numerical_signature()):
            raise RuntimeError(f"Refusing to reuse mismatched run: {path}")
        return saved
    X = data.get("X", data.get("X_train"))
    y = data.get("y", data.get("y_train"))
    result = solve(X, y, details["lambda"], details["method"], L=L,
                   metric_callback=callback_for(data, config), **config["solver"])
    coef = result.pop("coef")
    last = result["history"][-1]
    result.update(details, run_id=run_id, reference=ref, lipschitz_setup_sec=setup_sec,
                  numerical_signature=numerical_signature(),
                  objective_above_reference=last["primal"] - ref["primal"],
                  # SVD is needed by each proximal solve in a cold invocation; CD does not need it.
                  cold_start_sec=result["elapsed_sec"] + (setup_sec if details["method"] != "cd" else 0))
    result["threshold_hits"] = {}
    for tol in config["thresholds"]:
        hit = next((row for row in result["history"] if row["normalized_gap"] <= tol), None)
        result["threshold_hits"][str(tol)] = hit
    if "w_true" in data:
        result["support_sensitivity"] = {str(t): support(coef, data["w_true"], t)
                                         for t in config["support_sensitivity"]}
    else:
        result["test_mse"] = float(np.mean((data["X_test"] @ coef - data["y_test"]) ** 2))
        result["test_rmse"] = float(np.sqrt(result["test_mse"]))
    coefficient_path = destination / "coefficients" / (run_id + ".npz")
    coefficient_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(coefficient_path, coef=coef)
    result["coefficient_sha256"] = sha(coefficient_path)
    dump(path, result)
    print(json.dumps({"run": run_id, "status": result["status"], "sec": round(result["elapsed_sec"], 4),
                      "iterations": result["iterations"], "gap": last["normalized_gap"]}), flush=True)
    return result


def synthetic(config, destination, stage, config_hash):
    params = config["synthetic"]
    seeds = [0] if stage in ("pilot", "smoke") else params["seeds"]
    rhos = [0.0, 0.9] if stage in ("pilot", "smoke") else params["rhos"]
    ratios = [0.01, 0.1] if stage == "pilot" else ([0.05] if stage == "smoke" else params["lambda_ratios"])
    base = {k: v for k, v in params.items() if k not in ("seeds", "rhos", "lambda_ratios")}
    if stage == "smoke":
        base.update(n=200, p=100, sparsity=4, block_size=20, n_test=100)
    warmup_start = perf_counter()
    warmup_cd()
    dump(destination / "jit_setup.json", {"elapsed_sec": perf_counter() - warmup_start})
    for seed in seeds:
        for rho in rhos:
            setup_start = perf_counter()
            data = make_synthetic(seed=seed, rho=rho, **base)
            data["X"] = np.asfortranarray(data["X"])
            preprocess_sec = perf_counter() - setup_start
            X, y = data["X"], data["y"]
            matrix_hash = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
            setup_start = perf_counter()
            L = lipschitz_constant(X)
            setup_sec = perf_counter() - setup_start
            ds_id = f"synthetic_s{seed}_rho{rho:g}"
            metadata = dict(data["metadata"], matrix_hash=matrix_hash, L=L,
                            preprocess_sec=preprocess_sec, lipschitz_setup_sec=setup_sec)
            dump(destination / "datasets" / (ds_id + ".json"), metadata)
            lambda_max = float(np.max(np.abs(X.T @ y)) / len(y))
            for ratio in ratios:
                lam = lambda_max * ratio
                ref = reference(X, y, lam)
                methods = list(np.random.default_rng(seed + int(rho * 100) + int(ratio * 1000)).permutation(config["methods"]))
                for method in methods:
                    details = {"dataset": "synthetic", "seed": seed, "rho": rho, "lambda_ratio": ratio,
                               "lambda": lam, "lambda_max": lambda_max, "method": method,
                               "dataset_hash": matrix_hash, "config_sha256": config_hash}
                    run_one(data, config, destination, f"{ds_id}_a{ratio:g}_{method}", details, L, setup_sec, ref)


def real(config, destination, config_hash):
    for seed in config["real"]["seeds"]:
        setup_start = perf_counter()
        data = make_real(seed=seed)
        data["X_train"] = np.asfortranarray(data["X_train"])
        preprocess_sec = perf_counter() - setup_start
        X, y = data["X_train"], data["y_train"]
        matrix_hash = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
        lambda_max = float(np.max(np.abs(X.T @ y)) / len(y))
        candidate_ratios = np.geomspace(config["real"]["validation_min_ratio"], 1.0,
                                       config["real"]["validation_ratios"])
        selection = []
        selection_start = perf_counter()
        for ratio in candidate_ratios:
            model = Lasso(alpha=lambda_max * ratio, fit_intercept=False, tol=1e-10,
                          max_iter=200000, selection="cyclic").fit(X, y)
            cert = certificate(X, y, model.coef_, lambda_max * ratio)
            if cert["normalized_gap"] > 1e-8:
                raise RuntimeError("Validation candidate did not converge to the required reference accuracy")
            selection.append({"ratio": float(ratio), "lambda": float(lambda_max * ratio),
                              "validation_mse": float(np.mean((data["X_val"] @ model.coef_ - data["y_val"]) ** 2)),
                              "normalized_gap": cert["normalized_gap"]})
        best = int(np.argmin([row["validation_mse"] for row in selection]))
        # Pick the best validation value and its nearest grid neighbours.
        selected = sorted(sorted(range(len(selection)), key=lambda j: (abs(j - best), j))[:3])
        selection_sec = perf_counter() - selection_start
        setup_start = perf_counter()
        L = lipschitz_constant(X)
        setup_sec = perf_counter() - setup_start
        dump(destination / "datasets" / f"real_s{seed}.json", dict(data["metadata"], matrix_hash=matrix_hash,
             preprocess_sec=preprocess_sec, lipschitz_setup_sec=setup_sec, L=L, lambda_max=lambda_max,
             selection=selection, best_index=best, selected_indices=selected, selection_sec=selection_sec))
        for idx in selected:
            ratio, lam = selection[idx]["ratio"], selection[idx]["lambda"]
            ref = reference(X, y, lam)
            methods = list(np.random.default_rng(seed + idx).permutation(config["methods"]))
            for method in methods:
                details = {"dataset": "real", "seed": seed, "rho": None, "lambda_ratio": ratio,
                           "lambda": lam, "lambda_max": lambda_max, "method": method,
                           "dataset_hash": matrix_hash, "config_sha256": config_hash,
                           "selected_by_validation": idx == best, "validation_index": idx}
                run_one(data, config, destination, f"real_s{seed}_i{idx}_{method}", details, L, setup_sec, ref)


def reproduce(config, destination):
    checks = []
    for rho in (0.0, 0.9):
        params = {k: v for k, v in config["synthetic"].items() if k not in ("seeds", "rhos", "lambda_ratios")}
        data = make_synthetic(seed=0, rho=rho, **params)
        X, y = np.asfortranarray(data["X"]), data["y"]
        L = lipschitz_constant(X)
        lam = 0.05 * np.max(np.abs(X.T @ y)) / len(y)
        for method in config["methods"]:
            run_id = f"synthetic_s0_rho{rho:g}_a0.05_{method}"
            original = json.loads((ROOT / "results/main/runs" / (run_id + ".json")).read_text())
            coef = np.load(ROOT / "results/main/coefficients" / (run_id + ".npz"))["coef"]
            rerun = solve(X, y, lam, method, L=L, metric_callback=callback_for(data, config), **config["solver"])
            delta = float(np.max(np.abs(coef - rerun["coef"])))
            objective_delta = abs(original["history"][-1]["primal"] - rerun["history"][-1]["primal"])
            checks.append({"run_id": run_id, "coef_max_difference": delta, "objective_difference": objective_delta,
                           "passed": bool(delta <= 1e-7 and objective_delta <= 1e-8 and rerun["status"] == "converged")})
    dump(destination / "reproduction.json", {"checks": checks, "all_passed": all(x["passed"] for x in checks)})
    print(json.dumps(checks, indent=2), flush=True)
    if not all(x["passed"] for x in checks):
        raise SystemExit("Reproduction check failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["smoke", "pilot", "main", "real", "reproduce"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/study.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    destination = ROOT / "results" / args.stage
    destination.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=config["threads"]):
        env = environment()
        env_path = destination / "environment.json"
        if env_path.exists():
            old_env = json.loads(env_path.read_text())
            if old_env.get("numerical_signature") != env["numerical_signature"]:
                raise RuntimeError("Existing stage uses different numerical code; choose a separate output directory")
        else:
            dump(env_path, env)
        dump(destination / "config.json", config)
        start = perf_counter()
        if args.stage in ("pilot", "main", "smoke"):
            synthetic(config, destination, args.stage, sha(args.config))
        elif args.stage == "real":
            real(config, destination, sha(args.config))
        else:
            reproduce(config, destination)
        dump(destination / "timing.json", {"wall_sec": perf_counter() - start})


if __name__ == "__main__":
    main()
