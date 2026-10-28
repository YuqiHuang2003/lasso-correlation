"""Mathematical and independent-solver checks, not implementation mirrors."""

import numpy as np
import pytest
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits

from lasso_study.solvers import certificate, lipschitz_constant, solve, warmup_cd


@pytest.fixture(scope="module", autouse=True)
def stable_threads():
    with threadpool_limits(limits=1):
        warmup_cd()
        yield


@pytest.mark.parametrize("method", ["ista", "fista", "cd"])
def test_orthogonal_closed_form(method):
    rng = np.random.default_rng(5)
    n, p = 60, 15
    Q, _ = np.linalg.qr(rng.normal(size=(n, p)))
    X = np.asfortranarray(np.sqrt(n) * Q)
    y = rng.normal(size=n)
    lam = 0.03
    correlation = X.T @ y / n
    expected = np.sign(correlation) * np.maximum(np.abs(correlation) - lam, 0)
    result = solve(X, y, lam, method, tol=1e-12, check_every=1)
    assert result["status"] == "converged"
    np.testing.assert_allclose(result["coef"], expected, atol=1e-10)
    assert result["history"][-1]["dual_feasibility"] <= 1 + 1e-12


@pytest.mark.parametrize("method", ["ista", "fista", "cd"])
def test_correlated_matches_sklearn_objective(method):
    rng = np.random.default_rng(25)
    n, p = 160, 40
    common = rng.normal(size=(n, 1))
    X = np.asfortranarray(0.8 * common + 0.6 * rng.normal(size=(n, p)))
    truth = np.zeros(p)
    truth[[1, 5, 21]] = [1.5, -2, 1]
    y = X @ truth + 0.2 * rng.normal(size=n)
    lam = 0.05
    reference = Lasso(alpha=lam, tol=1e-12, fit_intercept=False, max_iter=200000).fit(X, y)
    optimum = certificate(X, y, reference.coef_, lam)["primal"]
    result = solve(X, y, lam, method, L=lipschitz_constant(X), tol=1e-9,
                   max_iter=100000, time_limit=30, check_every=5)
    assert result["status"] == "converged"
    last = result["history"][-1]
    assert abs(last["primal"] - optimum) <= 2e-8
    np.testing.assert_allclose(result["coef"], reference.coef_, atol=2e-6)
    assert all(row["dual_feasibility"] <= 1 + 1e-12 for row in result["history"])
    assert all(row["raw_gap"] >= -1e-11 for row in result["history"])
    assert last["kkt_inf"] < 1e-6


@pytest.mark.parametrize("method", ["ista", "fista", "cd"])
def test_lambda_max_zero_solution(method):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(40, 18))
    y = rng.normal(size=40)
    lam_max = np.max(np.abs(X.T @ y)) / len(y)
    result = solve(X, y, lam_max, method, tol=1e-12)
    assert result["status"] == "converged"
    assert result["iterations"] == 0
    np.testing.assert_array_equal(result["coef"], np.zeros(X.shape[1]))


def test_limits_preserve_unconverged_record():
    X = np.eye(4)
    y = np.array([2., -1., 3., 0.])
    timed = solve(X, y, 0.1, "ista", time_limit=0)
    assert timed["status"] == "time_limit"
    assert timed["iterations"] == 0
    assert len(timed["history"]) == 1
    capped = solve(X, y, 0.1, "ista", max_iter=0)
    assert capped["status"] == "max_iter"
    assert capped["history"][-1]["normalized_gap"] > 1e-6


def test_callback_and_final_checkpoint():
    rng = np.random.default_rng(32)
    X = rng.normal(size=(30, 10))
    y = rng.normal(size=30)
    result = solve(X, y, .01, "ista", max_iter=2, check_every=5, tol=0,
                   metric_callback=lambda w: {"squared_norm": float(w @ w)})
    assert [r["iteration"] for r in result["history"]] == [0, 2]
    assert all("squared_norm" in r for r in result["history"])
    assert result["history"][-1]["time"] <= result["elapsed_sec"]


@pytest.mark.parametrize("method", ["ista", "fista", "cd"])
def test_zero_columns(method):
    X = np.zeros((12, 4))
    y = np.arange(12, dtype=float)
    result = solve(X, y, .1, method)
    assert result["status"] == "converged"
    assert result["history"][-1]["normalized_gap"] < 1e-12
