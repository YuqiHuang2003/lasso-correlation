#!/usr/bin/env python3
"""Empirical convergence rates of ISTA, FISTA, and CD, reported in Section 5.1.

Reads the saved synthetic runs (results/v2/synthetic_runs.csv) at s=20, c=1 and
reports, for each method, layout, and rho, the median number of iterations
(epochs for CD) needed to move the normalized duality gap from 1e-2 to 1e-4
and from 1e-4 to 1e-6. Equal counts for the two steps indicate a linear rate.

It then rebuilds the same designs, fits the Lasso with scikit-learn, and
compares the ISTA count with the linear-rate prediction (L/mu) ln 100, where
mu is the smallest eigenvalue of the Gram matrix X_A^T X_A / n on the true
support (mu_true) or on the support of the fitted solution (mu_fit).
No data download is needed.

Usage:
    .venv/bin/python scripts/convergence_rate.py
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lasso_study.data_v2 import make_synthetic  # noqa: E402


def min_eigenvalue(X, columns):
    gram = X[:, columns].T @ X[:, columns] / X.shape[0]
    return float(np.linalg.eigvalsh(gram)[0])


def main():
    runs = pd.read_csv(ROOT / "results/v2/synthetic_runs.csv")
    runs = runs[(runs.sparsity == 20) & (runs.lambda_factor == 1.0)].copy()
    runs["first"] = runs["iteration_to_0.0001"] - runs["iteration_to_0.01"]
    runs["second"] = runs["iteration_to_1e-06"] - runs["iteration_to_0.0001"]
    steps = (runs[runs.method.isin(["ista", "fista", "cd"])]
             .groupby(["method", "layout", "rho"])[["first", "second", "iterations"]].median())
    print("Median iterations per two decades of the gap (s=20, c=1)")
    print(steps.to_string())

    ista = runs[runs.method == "ista"]
    rows = []
    for _, run in ista.iterrows():
        data = make_synthetic(seed=int(run.seed), rho=float(run.rho), sparsity=20, layout=run.layout)
        X, y = data["X"], data["y"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coef = Lasso(alpha=float(run["lambda"]), fit_intercept=False, tol=1e-12,
                         max_iter=200000).fit(X, y).coef_
        fitted = np.flatnonzero(np.abs(coef) > 1e-6)
        true = np.flatnonzero(data["w_true"])
        L = float(run.design_L)
        rows.append({"layout": run.layout, "rho": run.rho, "observed": run.second,
                     "fitted_size": len(fitted),
                     "predicted_true": L / min_eigenvalue(X, true) * np.log(100),
                     "predicted_fit": L / min_eigenvalue(X, fitted) * np.log(100)})
    table = pd.DataFrame(rows).groupby(["layout", "rho"]).median()
    print("\nISTA iterations from 1e-4 to 1e-6 against (L/mu) ln 100 (medians over seeds)")
    print(table.round(1).to_string())


if __name__ == "__main__":
    main()
