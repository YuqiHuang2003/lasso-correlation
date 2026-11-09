#!/usr/bin/env python3
"""Rank check of equicorrelation sets on the news and gene control targets.

Reproduces Table 5 of the report. For each of the 72 confirmation targets
(36 news, 36 gene) and each reference implementation (sklearn, skglm), the
script rebuilds the design, obtains a solution with normalized duality gap at
most 1e-10, forms candidate equicorrelation sets, and checks whether the
corresponding submatrix X_E has full column rank. Tibshirani (2013) shows that
the Lasso solution is unique when X_E has full column rank.

Saved reference coefficients are reused when the repository contains them.
The other targets are refitted with the frozen reference settings in
configs/reference_audit_v1.json. Requires `make data` and `make install-modern`.

Usage:
    .venv/bin/python scripts/check_equicorrelation_rank.py
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from itertools import product
import json
from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.linear_model import Lasso

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".modern_deps")]
from lasso_study.audit_data_v1 import load_audit_source, make_audit_design  # noqa: E402
from lasso_study.modern_solver import solve_modern  # noqa: E402
from lasso_study.solvers import certificate  # noqa: E402

COEF_DIR = ROOT / "results/reference_audit_v1/confirm/reference_coefficients"
NEAR_TOLERANCES = (1e-8, 1e-6)
GAP_ALLOWANCE = 1e-12


def references(data, settings):
    """Return the two high-accuracy solutions for one target."""
    X, y, lam = data["X"], data["y"], data["metadata"]["lambda"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w_sklearn = Lasso(alpha=lam, fit_intercept=False, tol=settings["sklearn_tol"],
                          max_iter=settings["sklearn_max_iter"], selection="cyclic").fit(X, y).coef_
    w_skglm = solve_modern(X, y, lam, tol=settings["normalized_gap_tol"],
                           time_limit=settings["time_limit"], **settings["modern"])["coef"]
    return {"sklearn": np.asarray(w_sklearn), "skglm": np.asarray(w_skglm)}


def rank_record(X, columns):
    if len(columns) == 0:
        return {"size": 0, "rank": 0, "full_column_rank": True, "smin": None}
    singular = np.linalg.svd(X[:, columns], compute_uv=False)
    rank = int(np.linalg.matrix_rank(X[:, columns]))
    return {"size": int(len(columns)), "rank": rank, "full_column_rank": rank == len(columns),
            "smin": float(singular[-1]), "smin_over_smax": float(singular[-1] / singular[0])}


def candidate_sets(X, y, w, lam):
    """Near-equicorrelation sets and an outer set derived from the duality gap.

    The residual dual D(u) is (1/n)-strongly concave, so ||u - u*||^2 <= 2 n G
    for any feasible u. Every j in the true equicorrelation set therefore
    satisfies |x_j^T u| + ||x_j|| sqrt(2 n G) >= n * lam.
    """
    n = X.shape[0]
    cert = certificate(X, y, w, lam)
    residual = y - X @ w
    u = residual / max(1.0, np.max(np.abs(X.T @ residual)) / (n * lam))
    score = np.abs(X.T @ residual) / n / lam
    sets = {f"near_E_{tol:g}": np.flatnonzero(score >= 1 - tol) for tol in NEAR_TOLERANCES}
    radius = np.linalg.norm(X, axis=0) * np.sqrt(2 * n * cert["gap"])
    sets["gap_outer_E"] = np.flatnonzero(np.abs(X.T @ u) + radius >= n * lam * (1 - GAP_ALLOWANCE))
    return cert, sets


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reference_audit_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/equicorrelation_rank_check.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    settings = config["references"]

    designs, outputs = {}, []
    for source in ("news", "gene"):
        pool = load_audit_source(source)
        seeds = config["sources"][source]["confirm_seeds"]
        for seed, c, amplitude in product(seeds, config["c_values"], config["amplitudes"]):
            data = make_audit_design(config, seed, source, amplitude, c, source_data=pool)
            X, y, lam = data["X"], data["y"], data["metadata"]["lambda"]
            key = f"{source}{seed}"
            if key not in designs:
                singular = np.linalg.svd(X, compute_uv=False)
                designs[key] = {"shape": list(X.shape), "rank": int(np.linalg.matrix_rank(X)),
                                "smallest_singular": float(singular[-1])}
            target = f"{source}_seed{seed}_c{c:g}_{amplitude}"
            saved = COEF_DIR / f"{target}.npz"
            if saved.is_file():
                with np.load(saved) as arrays:
                    refs = {"sklearn": arrays["ref_sklearn"].copy(), "skglm": arrays["ref_skglm"].copy()}
                origin = "saved"
            else:
                refs = references(data, settings)
                origin = "refitted"
            for method, w in refs.items():
                cert, sets = candidate_sets(X, y, w, lam)
                record = {"source": source, "target": target, "method": method, "coefficients": origin,
                          "normalized_gap": cert["normalized_gap"], "active_size": int(np.sum(w != 0))}
                record.update({name: rank_record(X, cols) for name, cols in sets.items()})
                outputs.append(record)
            print(f"{target}: {origin}", flush=True)

    summary = {}
    for source in ("news", "gene"):
        rows = [r for r in outputs if r["source"] == source]
        keys = [f"{source}{s}" for s in config["sources"][source]["confirm_seeds"]]
        summary[source] = {
            "outputs": len(rows),
            "max_normalized_gap": max(r["normalized_gap"] for r in rows),
            "full_matrix_shape": designs[keys[0]]["shape"],
            "full_matrix_rank": sorted({designs[k]["rank"] for k in keys}),
        }
        for name in [f"near_E_{tol:g}" for tol in NEAR_TOLERANCES] + ["gap_outer_E"]:
            sizes = [r[name]["size"] for r in rows]
            summary[source][name] = {
                "full_column_rank_outputs": sum(r[name]["full_column_rank"] for r in rows),
                "size_range": [min(sizes), max(sizes)],
                "minimum_smallest_singular": min(r[name]["smin"] for r in rows if r[name]["smin"] is not None),
            }
    args.output.write_text(json.dumps({"designs": designs, "summary": summary, "outputs": outputs}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
