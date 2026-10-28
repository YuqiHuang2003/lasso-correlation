"""Reference implementations for the equally scaled, unweighted Lasso.

The objective is ||X w - y||^2 / (2 n) + lam * ||w||_1. All methods
start at zero. A coordinate-descent iteration means one complete sweep.
The timer includes algorithm setup and all checkpoint diagnostics, but excludes
the shared Lipschitz calculation and the one-time C-kernel warmup. Checkpoint
times are observed upper bounds on time to reach an accuracy, not interpolated
crossing times.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from typing import Callable

import numpy as np


_CD_FUNCTION = None
_CD_LIBRARY = None


def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Proximal map of threshold * ||.||_1."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def lipschitz_constant(X: np.ndarray) -> float:
    """Compute ||X||_2^2 / n by dense SVD, with a roundoff safety margin.

    This is deliberately a reliable spectral calculation, rather than a fixed
    number of power iterations, which can underestimate the required constant.
    Reuse its result across proximal algorithms on the same design matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
        raise ValueError("X must be a finite, nonempty two-dimensional matrix")
    singular_values = np.linalg.svd(X, compute_uv=False)
    return float((singular_values[0] ** 2 / X.shape[0]) * (1.0 + 1e-12))


def certificate(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> dict:
    """Primal/feasible-dual certificate and unnormalized KKT violation.

    The residual-space dual is max_u (2 y^T u - ||u||^2)/(2 n),
    subject to ||X^T u||_inf <= n * lam. Scaling the current residual
    produces a feasible point for every iterate, including the zero iterate.
    """
    if lam <= 0 or not np.isfinite(lam):
        raise ValueError("lam must be positive and finite")
    n = X.shape[0]
    residual = y - X @ w
    correlation = (X.T @ residual) / n
    maximum_correlation = float(np.max(np.abs(correlation)))
    # Inflate a scaling factor at the boundary very slightly for feasibility
    # in finite precision. This only makes the lower bound more conservative.
    scale = max(1.0, maximum_correlation / lam) * (1.0 + 4e-15)
    u = residual / scale
    primal = float(residual @ residual / (2 * n) + lam * np.abs(w).sum())
    dual = float((2 * (y @ u) - u @ u) / (2 * n))
    raw_gap = primal - dual
    # A truly negative gap signals an implementation/numerical failure. Only
    # harmless final-bit cancellation can be clipped to zero.
    rounding_tolerance = 1e-11 * max(1.0, abs(primal), abs(dual))
    if raw_gap < -rounding_tolerance:
        raise FloatingPointError("primal objective fell below feasible dual bound")
    gap = max(0.0, raw_gap)
    violations = np.maximum(np.abs(correlation) - lam, 0.0)
    active = w != 0
    violations[active] = np.abs(correlation[active] - lam * np.sign(w[active]))
    return {
        "primal": primal,
        "dual": dual,
        "gap": gap,
        "raw_gap": raw_gap,
        "normalized_gap": gap / max(1.0, abs(primal), abs(dual)),
        "dual_feasibility": maximum_correlation / (lam * scale),
        "kkt_inf": float(violations.max()),
    }


def _load_cd_kernel():
    """Build our readable C source locally, then load it through ctypes.

    The source and binary remain in this project. Compilation is a one-time
    environment cost, outside solver timers, just as loading BLAS is for the
    proximal methods. No external numerical solver is used for CD updates.
    """
    global _CD_FUNCTION, _CD_LIBRARY
    if _CD_FUNCTION is not None:
        return _CD_FUNCTION
    source = Path(__file__).with_name("cd_kernel.c")
    build = Path(__file__).resolve().parents[2] / "build"
    build.mkdir(exist_ok=True)
    library_path = build / "cd_kernel.so"
    if not library_path.exists() or library_path.stat().st_mtime_ns < source.stat().st_mtime_ns:
        with tempfile.NamedTemporaryFile(dir=build, suffix=".so", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            subprocess.run(
                ["gcc", "-O3", "-std=c99", "-shared", "-fPIC", str(source), "-o", str(temporary_path)],
                check=True, capture_output=True, text=True, timeout=60,
            )
            os.replace(temporary_path, library_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    _CD_LIBRARY = ctypes.CDLL(str(library_path))
    function = _CD_LIBRARY.cd_sweep
    function.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags=("F_CONTIGUOUS", "ALIGNED")),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED")),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED")),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED")),
        ctypes.c_int64, ctypes.c_int64, ctypes.c_double,
    ]
    function.restype = None
    _CD_FUNCTION = function
    return function


def _cd_sweep(X, residual, w, column_norms, lam):
    _load_cd_kernel()(X, residual, w, column_norms, X.shape[0], X.shape[1], lam)


def warmup_cd() -> None:
    """Compile/load and invoke the C kernel before measured experiments."""
    X = np.ones((2, 2), dtype=np.float64, order="F")
    _cd_sweep(X, np.zeros(2), np.zeros(2), np.ones(2), 1.0)


def solve(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    method: str,
    *,
    L: float | None = None,
    tol: float = 1e-6,
    max_iter: int = 100000,
    time_limit: float = 120.0,
    check_every: int = 5,
    metric_callback: Callable[[np.ndarray], dict] | None = None,
) -> dict:
    """Solve Lasso and return its final coefficient vector plus diagnostics.

    method is 'ista', 'fista', or 'cd'. A supplied L must be a Lipschitz upper
    bound; the caller is responsible for obtaining it reliably. If omitted for
    a proximal method it is computed by SVD and timed separately. Any supplied
    callback runs at each certificate checkpoint and is included in timing.
    A time limit is checked between updates; the final diagnostic and an update
    already in progress can exceed the limit. Unconverged iterates are retained.
    """
    method = method.lower()
    if method not in {"ista", "fista", "cd"}:
        raise ValueError("method must be ista, fista, or cd")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != len(y) or min(X.shape) == 0:
        raise ValueError("X and y have incompatible or empty shapes")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must be finite")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be positive and finite")
    if not np.isfinite(tol) or tol < 0 or max_iter < 0 or check_every < 1:
        raise ValueError("invalid tolerance or iteration/checkpoint limit")
    if not np.isfinite(time_limit) or time_limit < 0:
        raise ValueError("time_limit must be finite and nonnegative")
    lipschitz_sec = 0.0
    if method != "cd":
        if L is None:
            setup_start = perf_counter()
            L = lipschitz_constant(X)
            lipschitz_sec = perf_counter() - setup_start
        if not np.isfinite(L) or L < 0:
            raise ValueError("L must be finite and nonnegative")
        # X=0 gives a constant smooth term. Any positive step solves its L1
        # penalty at zero, so L=1 is a convenient harmless convention.
        if L == 0.0:
            L = 1.0
    jit_warmup_sec = 0.0
    if method == "cd":
        warmup_start = perf_counter()
        warmup_cd()
        jit_warmup_sec = perf_counter() - warmup_start

    start = perf_counter()
    n, p = X.shape
    w = np.zeros(p, dtype=np.float64)
    z = w.copy()
    momentum = 1.0
    if method == "cd":
        # Column-major storage is necessary for contiguous coordinate updates.
        X = np.asfortranarray(X)
        residual = y.copy()
        column_norms = np.einsum("ij,ij->j", X, X) / n
    history: list[dict] = []
    diagnostics_sec = 0.0
    objective_increases = 0

    def checkpoint(iteration: int) -> dict:
        nonlocal diagnostics_sec, objective_increases
        diagnostic_start = perf_counter()
        row = certificate(X, y, w, lam)
        if metric_callback is not None:
            metrics = metric_callback(w.copy())
            if set(metrics).intersection(row) or {"iteration", "time"}.intersection(metrics):
                raise ValueError("metric callback keys overlap solver diagnostics")
            row.update(metrics)
        if history and row["primal"] > history[-1]["primal"] + 1e-12 * max(
            1.0, abs(history[-1]["primal"])
        ):
            objective_increases += 1
        diagnostics_sec += perf_counter() - diagnostic_start
        row["iteration"] = iteration
        row["time"] = perf_counter() - start
        history.append(row)
        return row

    row = checkpoint(0)
    iteration = 0
    status = "max_iter"
    if row["normalized_gap"] <= tol:
        status = "converged"
    else:
        for iteration in range(1, max_iter + 1):
            if perf_counter() - start >= time_limit:
                iteration -= 1
                status = "time_limit"
                break
            if method == "cd":
                _cd_sweep(X, residual, w, column_norms, lam)
            else:
                gradient = (X.T @ (X @ z - y)) / n
                next_w = soft_threshold(z - gradient / L, lam / L)
                if method == "fista":
                    next_momentum = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
                    z = next_w + ((momentum - 1) / next_momentum) * (next_w - w)
                    momentum = next_momentum
                else:
                    z = next_w
                w = next_w
            if iteration % check_every == 0:
                row = checkpoint(iteration)
                if row["normalized_gap"] <= tol:
                    status = "converged"
                    break
        if history[-1]["iteration"] != iteration:
            row = checkpoint(iteration)
        # A final checkpoint can demonstrate convergence even after an update
        # cap. It is still valid evidence of reaching the declared tolerance.
        if row["normalized_gap"] <= tol:
            status = "converged"
    elapsed_sec = perf_counter() - start
    return {
        "coef": w,
        "history": history,
        "status": status,
        "iterations": iteration,
        "elapsed_sec": elapsed_sec,
        "diagnostics_sec": diagnostics_sec,
        "lipschitz_sec": lipschitz_sec,
        "jit_warmup_sec": jit_warmup_sec,
        "cd_backend": "C99 via ctypes" if method == "cd" else None,
        "objective_increases": objective_increases,
        "L": None if L is None else float(L),
    }
