"""Scientific design checks, including pairing and prevention of data leakage."""
import numpy as np
import pytest

from lasso_study.data import array_sha256, make_real, make_synthetic


def test_synthetic_pairing_coordinates_and_independent_test():
    kwargs = dict(seed=7, n=400, p=100, sparsity=4, block_size=10, n_test=200)
    independent = make_synthetic(rho=0.0, **kwargs)
    correlated = make_synthetic(rho=0.9, **kwargs)
    repeated = make_synthetic(rho=0.9, **kwargs)
    assert correlated["metadata"]["hashes"] == repeated["metadata"]["hashes"]
    np.testing.assert_array_equal(independent["w_true"], correlated["w_true"])
    support = np.flatnonzero(correlated["w_true"])
    assert len(np.unique(support // 10)) == 4
    np.testing.assert_allclose(correlated["X"].mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(correlated["X"].std(axis=0), 1.0, atol=1e-14)
    assert abs(correlated["y"].mean()) < 1e-14
    assert correlated["X_test"].shape == (200, 100)
    assert np.max(abs(correlated["X_test"].mean(axis=0))) > 0.02
    # Population noise scale is derived from signal, rather than outcome labels.
    expected = np.var(correlated["X"] @ correlated["w_true"]) / 10.0
    assert correlated["metadata"]["noise_std"] ** 2 == pytest.approx(expected)
    assert array_sha256(independent["X"]) != array_sha256(correlated["X"])


def test_synthetic_correlation_is_controlled():
    d = make_synthetic(seed=2, rho=0.6, n=12000, p=40, sparsity=2,
                       block_size=10, n_test=20)
    corr = np.corrcoef(d["X"], rowvar=False)
    assert corr[0, 1] == pytest.approx(0.6, abs=0.035)
    assert abs(corr[0, 20]) < 0.035


def test_invalid_distinct_block_support_is_rejected():
    with pytest.raises(ValueError, match="Distinct-block"):
        make_synthetic(p=40, block_size=20, sparsity=3)


def test_real_transforms_use_only_training_rows(tmp_path):
    import pandas as pd

    n = 100
    rng = np.random.default_rng(12)
    table = {"url": [f"article-{i}" for i in range(n)],
             "timedelta": np.arange(n), "shares": np.arange(n) + 1}
    for j in range(58):
        table[f"feature_{j}"] = rng.normal(size=n)
    table["feature_57"] = np.ones(n)  # A constant is removed using train alone.
    frame = pd.DataFrame(table)
    path = tmp_path / "real.csv"
    frame.to_csv(path, index=False)
    original = make_real(seed=3, csv_path=path)
    splits = original["metadata"]["split_indices"]
    train, val, test = [np.array(splits[k]) for k in ("train", "val", "test")]
    assert len(set(train) & set(val)) == len(set(train) & set(test)) == len(set(val) & set(test)) == 0
    assert set(train) | set(val) | set(test) == set(range(n))
    assert original["metadata"]["used_features"] == 57
    np.testing.assert_allclose(original["X_train"].mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(original["X_train"].std(axis=0), 1, atol=1e-14)
    frame.loc[np.r_[val, test], "feature_0"] += 1e6
    frame.loc[np.r_[val, test], "shares"] += 1e9
    frame.to_csv(path, index=False)
    altered = make_real(seed=3, csv_path=path)
    np.testing.assert_array_equal(original["X_train"], altered["X_train"])
    np.testing.assert_array_equal(original["y_train"], altered["y_train"])
    assert original["metadata"]["train_mean"] == altered["metadata"]["train_mean"]
    assert original["metadata"]["train_y_mean"] == altered["metadata"]["train_y_mean"]

