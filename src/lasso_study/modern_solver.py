"""An isolated skglm AndersonCD context baseline for the same Lasso objective.

The package's subdifferential/KKT tolerance is not our normalized dual gap.
Every library fit starts at zero and is checked with the project's unchanged
certificate. A fixed, at-most-three-step tolerance ladder may require refits;
all refits and posterior diagnostics are included in measured elapsed time.
Only the initial point and fit endpoints are observed, so this wrapper must not
be used to claim precise time-to-accuracy curves for the library solver.
"""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable

import numpy as np
from threadpoolctl import threadpool_limits

from .solvers import certificate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODERN_DEPS = PROJECT_ROOT / ".modern_deps"
METHOD = "skglm"
_API = None
_WARMUP = None


def _api():
    """Load only optional packages from the isolated project dependency tree."""
    global _API
    if _API is None:
        if MODERN_DEPS.is_dir() and str(MODERN_DEPS) not in sys.path:
            sys.path.insert(0, str(MODERN_DEPS))
        os.environ.setdefault("NUMBA_NUM_THREADS", "1")
        try:
            from skglm.datafits import Quadratic
            from skglm.penalties import L1
            from skglm.solvers import AndersonCD
        except ImportError as error:
            raise ImportError(
                "Install the isolated optional packages listed in "
                "configs/skglm_environment.json before using the skglm baseline"
            ) from error
        _API = AndersonCD, Quadratic, L1
    return _API


def warmup_modern() -> dict:
    """Compile float64 Fortran-array kernels once, outside benchmark timers.

    Keep this returned setup cost in the experiment metadata. It is a separate
    software-initialization cost, not an unreported zero-cost cold start.
    """
    global _WARMUP
    if _WARMUP is not None:
        return dict(_WARMUP, performed=False, elapsed_sec=0.0)
    start = perf_counter()
    AndersonCD, Quadratic, L1 = _api()
    rng = np.random.default_rng(731)
    X = np.asfortranarray(rng.normal(size=(40, 16)), dtype=np.float64)
    y = np.ascontiguousarray(X[:, 0] - X[:, 1] + .2 * rng.normal(size=40))
    with threadpool_limits(limits=1):
        solver = AndersonCD(max_iter=30, max_epochs=1000, p0=10, tol=1e-8,
                            fit_intercept=False, warm_start=False, verbose=0)
        solver.solve(X, y, Quadratic(), L1(.01),
                     w_init=np.zeros(X.shape[1]), Xw_init=np.zeros(X.shape[0]))
    elapsed = perf_counter() - start
    _WARMUP = {
        "performed": True, "elapsed_sec": elapsed, "initial_warmup_sec": elapsed,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("skglm", "numba", "llvmlite", "numpy", "scipy", "scikit-learn")},
        "objective": "||Xw-y||^2/(2n)+lambda*||w||_1; intercept disabled",
        "array_signature": "float64 Fortran-contiguous X, contiguous y",
        "method": "skglm AndersonCD with working sets and Anderson acceleration",
    }
    return dict(_WARMUP)


def solve_modern(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    *,
    tol: float = 1e-6,
    max_iter: int = 100000,
    time_limit: float = 120.0,
    check_every: int = 5,
    metric_callback: Callable[[np.ndarray], dict] | None = None,
    library_max_iter: int = 100,
    max_epochs: int = 50000,
    library_tolerances: tuple[float, ...] = (1e-4, 1e-6, 1e-8),
    L: float | None = None,
) -> dict:
    """Fit from zero and accept only the common posterior dual certificate.

    max_iter is accepted for compatibility; the actual outer working-set cap
    is min(max_iter, library_max_iter), independently capped at 100 by default.
    max_epochs caps each working-set subproblem. These are not full CD epochs.
    check_every and L are compatibility inputs; the library controls its own
    checks and coordinate step sizes. They do not change library diagnostics.

    The library has no wall-clock callback. A time budget is checked before and
    after each fit and cannot interrupt an in-progress fit. If measured total
    time exceeds it, status is 'budget_exceeded', even if the final coefficient
    vector passes the common certificate. All observations remain in the output.
    """
    if tol < 0 or not np.isfinite(tol) or time_limit < 0 or not np.isfinite(time_limit):
        raise ValueError("tol and time_limit must be finite and nonnegative")
    if (max_iter < 0 or library_max_iter < 0 or max_epochs < 1 or check_every < 1
            or int(max_iter) != max_iter or int(library_max_iter) != library_max_iter):
        raise ValueError("Invalid iteration/checkpoint caps")
    if not 1 <= len(library_tolerances) <= 3:
        raise ValueError("The preregistered tolerance ladder must have one to three values")
    if any(value <= 0 or not np.isfinite(value) for value in library_tolerances):
        raise ValueError("Library tolerances must be positive and finite")
    if any(second >= first for first, second in zip(library_tolerances, library_tolerances[1:])):
        raise ValueError("Library tolerances must be strictly decreasing")
    # JIT/loading happens before the run clock and is returned separately.
    warmup = warmup_modern()
    AndersonCD, Quadratic, L1 = _api()
    start = perf_counter()
    X = np.asfortranarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != len(y) or min(X.shape) == 0:
        raise ValueError("X and y have incompatible or empty shapes")
    if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(lam) or lam <= 0:
        raise ValueError("Inputs must be finite and lam must be positive")
    outer_cap = min(int(max_iter), int(library_max_iter))
    w = np.zeros(X.shape[1], dtype=np.float64)
    history = []
    attempts = []
    diagnostics_sec = 0.0
    total_outer_iterations = 0

    def checkpoint(attempt):
        nonlocal diagnostics_sec
        diagnostic_start = perf_counter()
        row = certificate(X, y, w, lam)
        if metric_callback is not None:
            metrics = metric_callback(w.copy())
            reserved = set(row) | {"iteration", "time", "attempt"}
            if reserved.intersection(metrics):
                raise ValueError("Metric callback keys overlap solver diagnostics")
            row.update(metrics)
        diagnostics_sec += perf_counter() - diagnostic_start
        row.update(iteration=total_outer_iterations, attempt=attempt, time=perf_counter() - start)
        history.append(row)
        return row

    with threadpool_limits(limits=1):
        row = checkpoint(0)
        if row["normalized_gap"] > tol and outer_cap > 0:
            for attempt, library_tol in enumerate(library_tolerances, start=1):
                if perf_counter() - start >= time_limit:
                    break
                fit_start = perf_counter()
                solver = AndersonCD(max_iter=outer_cap, max_epochs=int(max_epochs),
                                    p0=min(10, X.shape[1]), tol=float(library_tol),
                                    ws_strategy="subdiff", fit_intercept=False,
                                    warm_start=False, verbose=0)
                # Every tolerance attempt is independent and starts at zero.
                w, objectives, stop_crit = solver.solve(
                    X, y, Quadratic(), L1(float(lam)),
                    w_init=np.zeros(X.shape[1], dtype=np.float64),
                    Xw_init=np.zeros(X.shape[0], dtype=np.float64))
                fit_sec = perf_counter() - fit_start
                total_outer_iterations += len(objectives)
                row = checkpoint(attempt)
                attempts.append({
                    "attempt": attempt, "library_tol": float(library_tol),
                    "initialization": "zero coefficients and zero fitted values",
                    "fit_elapsed_sec": fit_sec,
                    "elapsed_sec_including_posterior_diagnostics": perf_counter() - fit_start,
                    "cumulative_elapsed_sec": row["time"],
                    "outer_working_set_iterations": int(len(objectives)),
                    "library_returned_stop_crit": float(stop_crit),
                    "library_stop_crit_definition": "maximum subdifferential violation; not normalized dual gap",
                    "library_objectives_without_timestamps": np.asarray(objectives).tolist(),
                    "normalized_gap": row["normalized_gap"],
                    "passed_common_certificate": row["normalized_gap"] <= tol,
                })
                if row["normalized_gap"] <= tol or perf_counter() - start >= time_limit:
                    break
    elapsed = perf_counter() - start
    certificate_passed = row["normalized_gap"] <= tol
    budget_exceeded = elapsed > time_limit
    if budget_exceeded:
        status = "budget_exceeded"
    elif certificate_passed:
        status = "converged"
    elif outer_cap == 0:
        status = "max_iter"
    else:
        status = "not_converged"
    return {
        "method": METHOD, "coef": w, "history": history, "status": status,
        "certificate": {key: row[key] for key in
                        ("primal", "dual", "gap", "raw_gap", "normalized_gap", "dual_feasibility", "kkt_inf")},
        "certificate_passed": certificate_passed, "budget_exceeded": budget_exceeded,
        "elapsed_sec": elapsed, "diagnostics_sec": diagnostics_sec,
        "iterations": total_outer_iterations, "iteration_unit": "outer working-set iterations across all attempts",
        "attempts": attempts, "attempt_count": len(attempts),
        "library_tolerances": list(library_tolerances),
        "library_max_iter": outer_cap, "library_max_epochs": int(max_epochs),
        "time_limit_sec": time_limit,
        "time_limit_mode": "checked between/after fits; no interruption within an in-progress library fit",
        "history_sampling": "initial point and fit endpoints only; no internal time-to-accuracy curve",
        "threshold_curve_comparable": False,
        "check_every_compatibility_input": check_every,
        "L_compatibility_input_used": False,
        "jit_warmup_sec": warmup["elapsed_sec"],
        "warmup_metadata": warmup,
        "objective_increases": None,
    }
