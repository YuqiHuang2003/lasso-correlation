"""Independent objective/solution checks for the optional package baseline."""
import numpy as np
from sklearn.linear_model import Lasso

from lasso_study.modern_solver import solve_modern
from lasso_study.solvers import certificate


def test_modern_orthogonal_closed_form():
    rng = np.random.default_rng(681)
    Q, _ = np.linalg.qr(rng.normal(size=(70, 18)))
    X = np.asfortranarray(np.sqrt(len(Q)) * Q)
    y = X[:, 1] * 2 - X[:, 4] + .1 * rng.normal(size=len(Q))
    lam = .12
    products = X.T @ y / len(y)
    exact = np.sign(products) * np.maximum(np.abs(products) - lam, 0)
    result = solve_modern(X, y, lam, tol=1e-6)
    assert result["status"] == "converged"
    np.testing.assert_allclose(result["coef"], exact, atol=1e-8)
    assert result["certificate"]["normalized_gap"] <= 1e-6
    assert result["threshold_curve_comparable"] is False


def test_modern_correlated_objective_matches_sklearn():
    rng = np.random.default_rng(104)
    X = np.asfortranarray(.95 * rng.normal(size=(180, 1)) + .31 * rng.normal(size=(180, 45)))
    truth = np.zeros(45)
    truth[[0, 11, 31]] = [2, -1.5, 1]
    y = X @ truth + .2 * rng.normal(size=len(X))
    lam = .035
    oracle = Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=200000).fit(X, y)
    expected = certificate(X, y, oracle.coef_, lam)["primal"]
    result = solve_modern(X, y, lam, tol=1e-6,
                          metric_callback=lambda w: {"coef_error": float(np.linalg.norm(w - truth))})
    assert result["status"] == "converged"
    assert abs(result["certificate"]["primal"] - expected) < 1e-7
    np.testing.assert_allclose(result["coef"], oracle.coef_, atol=2e-4)
    assert result["certificate"]["dual_feasibility"] <= 1 + 1e-12
    assert len(result["attempts"]) <= 3
    assert result["history"][-1]["time"] <= result["elapsed_sec"]
    assert all(a["initialization"].startswith("zero") for a in result["attempts"])
    assert "coef_error" in result["history"][-1]
    assert result["library_max_iter"] == 100


def test_modern_lambda_max_is_zero_solution():
    rng = np.random.default_rng(901)
    X = np.asfortranarray(rng.normal(size=(35, 70)))
    y = rng.normal(size=35)
    lam = float(np.max(np.abs(X.T @ y)) / len(y))
    result = solve_modern(X, y, lam, tol=1e-6)
    assert result["status"] == "converged"
    assert result["attempt_count"] == 0
    assert result["iterations"] == 0
    np.testing.assert_array_equal(result["coef"], np.zeros(X.shape[1]))
    assert result["certificate"]["normalized_gap"] <= 1e-6
