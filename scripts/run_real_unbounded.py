#!/usr/bin/env python3
"""Table 2: the four solvers on the real data tasks without a time budget.

The main protocol (configs/study_v2.json) stops every solver after 20 s, and
ISTA reaches that budget on both tasks while its duality gap still decreases.
This script runs ISTA, FISTA, CD and skglm on the Table 2 settings with the
same tolerance, step size, checkpoint interval, nnz callback and skglm
settings, but with no practical time or iteration cap, so that every method
reaches normalized gap <= 1e-6 and all times come from one machine.

Settings: UJI, the validation-selected lambda of each of the 3 splits.
Leukemia, lambda / lambda_max = 0.01, three timing repeats.
Requires `make data`, `make install-modern` and GCC.
Writes results/v2/real_unbounded.json and prints the Table 2 medians.

Usage:
    .venv/bin/python scripts/run_real_unbounded.py [--methods ista fista cd skglm] [--tasks uji leukemia]
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / ".modern_deps")]
from lasso_study.data_v2 import make_leukemia, make_uji  # noqa: E402
from lasso_study.modern_solver import solve_modern, warmup_modern  # noqa: E402
from lasso_study.solvers import lipschitz_constant, solve, warmup_cd  # noqa: E402
from run_v2 import callback  # noqa: E402

TIME_LIMIT = 4 * 3600.0
MAX_ITER = 10**9
OUTPUT = ROOT / "results/v2/real_unbounded.json"


def run(method, data, X, y, lam, L, config):
    settings = dict(config["solver"], time_limit=TIME_LIMIT, max_iter=MAX_ITER)
    if method == "skglm":
        result = solve_modern(X, y, lam, L=L, metric_callback=callback(data, config),
                              **settings, **config["modern"])
    else:
        result = solve(X, y, lam, method, L=L, metric_callback=callback(data, config), **settings)
    coef = result["coef"]
    record = {"method": method, "lambda": lam, "status": result["status"],
              "iterations": int(result["iterations"]), "elapsed_sec": float(result["elapsed_sec"]),
              "final_normalized_gap": float(result["history"][-1]["normalized_gap"]),
              "nnz": int(np.sum(np.abs(coef) > config["support_threshold"]))}
    if "X_test" in data:
        record["test_mse"] = float(np.mean((data["X_test"] @ coef - data["y_test"]) ** 2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=["ista", "fista", "cd", "skglm"])
    parser.add_argument("--tasks", nargs="+", default=["uji", "leukemia"])
    parser.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2], help="Leukemia timing repeats to run")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/study_v2.json").read_text())
    warmup_cd()
    warmup_modern()
    records = json.loads(OUTPUT.read_text())["runs"] if OUTPUT.exists() else []
    records = [r for r in records if r["method"] not in args.methods or r["task"] not in args.tasks
               or (r["task"] == "leukemia" and int(r["setting"][3:]) not in args.repeats)]

    def save(record):
        records.append(record)
        print(json.dumps(record), flush=True)
        OUTPUT.write_text(json.dumps({"time_limit_sec": TIME_LIMIT, "max_iter": MAX_ITER,
                                      "tol": config["solver"]["tol"], "runs": records}, indent=2) + "\n")

    uji_runs = pd.read_csv(ROOT / "results/v2/uji_runs.csv")
    selected = uji_runs[(uji_runs.method == "ista") & (uji_runs.selected_by_validation == True)]  # noqa: E712
    for _, row in (selected.sort_values("seed").iterrows() if "uji" in args.tasks else []):
        data = make_uji(seed=int(row.seed))
        data["X_train"] = X = np.asfortranarray(data["X_train"])
        y, L = data["y_train"], lipschitz_constant(X)
        for method in args.methods:
            save(dict(run(method, data, X, y, float(row["lambda"]), L, config),
                      task="uji", setting=f"seed{int(row.seed)}"))

    if "leukemia" not in args.tasks:
        return summarize(records)
    data = make_leukemia()
    data["X"] = X = np.asfortranarray(data["X"])
    y = data["y"]
    lam = 0.01 * float(np.max(np.abs(X.T @ y)) / len(y))
    L = lipschitz_constant(X)
    for repeat in args.repeats:
        for method in args.methods:
            save(dict(run(method, data, X, y, lam, L, config), task="leukemia", setting=f"rep{repeat}"))

    summarize(records)


def summarize(records):
    table = pd.DataFrame(records)
    summary = table.groupby(["task", "method"]).agg(
        reached=("status", lambda s: f"{(s == 'converged').sum()}/{len(s)}"),
        time=("elapsed_sec", "median"), iterations=("iterations", "median"),
        nnz=("nnz", "median"), test_mse=("test_mse", "median"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
