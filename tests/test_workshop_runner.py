"""Mathematical checks for observational workshop diagnostics."""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_workshop import error_decomposition, gap_safe_zero_mask, reference_agreement, target_parameters
from lasso_study.solvers import certificate


def test_harm_repair_identity_with_both_types_of_changes():
    truth = np.array([1, 1, 0, 0], dtype=bool)
    reference = np.array([1, 0, 1, 0], dtype=bool)
    selected = np.array([0, 1, 1, 1], dtype=bool)
    result = error_decomposition(selected, reference, truth)
    assert result["reference_support_error"] == 2
    assert result["numerical_harm"] == 2
    assert result["numerical_repair"] == 1
    assert result["support_error"] == 3
    assert result["numerical_support_error"] == 3


def test_reference_consensus_requires_precision_and_support_agreement():
    settings = dict(normalized_gap_tol=1e-10, coef_linf_agreement_tol=1e-7,
                    normalized_objective_agreement_tol=1e-9)
    row = dict(status="converged", certificate=dict(normalized_gap=1e-12, dual_feasibility=1., primal=1.))
    coef = np.array([1., 0.])
    assert reference_agreement(row, row, coef, coef, settings, [1e-6, 1e-4])["reference_accepted"]
    failed = dict(row, certificate=dict(row["certificate"], normalized_gap=1e-7))
    assert not reference_agreement(row, failed, coef, coef, settings, [1e-6, 1e-4])["reference_accepted"]
    changed = np.array([1., .01])
    assert not reference_agreement(row, row, coef, changed, settings, [1e-6, 1e-4])["reference_accepted"]


def test_gap_safe_sphere_excludes_only_known_zero_coordinates():
    X = 2 * np.eye(4)
    y = np.array([3., 0., .1, -1.])
    lam = .2
    corr = X.T @ y / len(y)
    exact = np.sign(corr) * np.maximum(np.abs(corr) - lam, 0.)
    gap = certificate(X, y, exact, lam)["gap"]
    excluded = gap_safe_zero_mask(X, y, exact, lam, gap)
    np.testing.assert_array_equal(excluded, np.array([False, True, True, False]))


def test_registered_grid_has_one_shared_independent_feature_baseline():
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/workshop_pilot.json").read_text())
    targets = target_parameters(config, "pilot")
    assert len(targets) == 36
    assert len(targets) * len(config["methods"]) == 72
    assert not any(row["family"] == "toeplitz" and row["rho"] == 0 for row in targets)
    assert {tuple(row[name] for name in ("family", "rho")) for row in targets} == {
        ("block", 0.), ("block", .9), ("toeplitz", .9)}
