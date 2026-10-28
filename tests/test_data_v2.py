"""Test the v2 scientific controls and training-only benchmark preparation."""
import json

import numpy as np
import pytest

from lasso_study.data_v2 import make_leukemia, make_synthetic, make_uji


def test_synthetic_fixed_noise_and_nested_paired_supports():
    datasets = {}
    options = dict(seed=3, n=70, p=400, block_size=10, n_test=50, noise_std=1.)
    for rho in (0., .9):
        for layout in ("dispersed", "clustered"):
            for sparsity in (5, 20, 40):
                datasets[rho, layout, sparsity] = make_synthetic(
                    rho=rho, layout=layout, sparsity=sparsity, **options)
    first = datasets[0., "dispersed", 5]
    for (rho, layout, sparsity), data in datasets.items():
        meta = data["metadata"]
        assert meta["hashes"]["noise"] == first["metadata"]["hashes"]["noise"]
        assert meta["hashes"]["noise_test"] == first["metadata"]["hashes"]["noise_test"]
        assert meta["hashes"]["ordered_signs"] == first["metadata"]["hashes"]["ordered_signs"]
        reference = datasets[rho, "dispersed", 5]
        np.testing.assert_array_equal(data["X"], reference["X"])
        np.testing.assert_array_equal(data["X_test"], reference["X_test"])
        assert np.count_nonzero(data["w_true"]) == sparsity
        support = meta["support"]
        assert len(set(support)) == sparsity
        _, counts = np.unique(np.asarray(support) // 10, return_counts=True)
        np.testing.assert_array_equal(counts, np.full(len(counts), 1 if layout == "dispersed" else 5))
        largest = datasets[rho, layout, 40]["metadata"]
        assert support == largest["support"][:sparsity]
        assert meta["signs"] == largest["signs"][:sparsity]
        np.testing.assert_array_equal(data["w_true"][support], meta["signs"])
        np.testing.assert_allclose(data["X"].mean(axis=0), 0., atol=1e-14)
        np.testing.assert_allclose(data["X"].std(axis=0), 1., atol=1e-14)
        assert abs(data["y"].mean()) < 1e-14
        assert meta["snr_population"] == pytest.approx(np.var(data["X"] @ data["w_true"]))
        recovered_noise = data["y"] + meta["train_y_mean"] - data["X"] @ data["w_true"]
        reference_noise = first["y"] + first["metadata"]["train_y_mean"] - first["X"] @ first["w_true"]
        np.testing.assert_allclose(recovered_noise, reference_noise, atol=1e-13)
        json.dumps(meta, allow_nan=False)
    assert first["metadata"]["hashes"]["X"] != datasets[.9, "dispersed", 5]["metadata"]["hashes"]["X"]


def test_synthetic_repeatability_and_layout_validation():
    a = make_synthetic(seed=7, rho=.6, layout="clustered", sparsity=20, n=50, p=200, n_test=30)
    b = make_synthetic(seed=7, rho=.6, layout="clustered", sparsity=20, n=50, p=200, n_test=30)
    assert a["metadata"]["hashes"] == b["metadata"]["hashes"]
    with pytest.raises(ValueError, match="multiple of five"):
        make_synthetic(layout="clustered", sparsity=6)
    with pytest.raises(ValueError, match="capacity"):
        make_synthetic(layout="dispersed", sparsity=60)


def test_uji_no_label_or_heldout_transform_leakage(tmp_path):
    import pandas as pd

    rng = np.random.default_rng(5)
    n = 100
    frame = pd.DataFrame(rng.integers(-100, 0, size=(n, 520)),
                         columns=[f"WAP{i:03d}" for i in range(1, 521)])
    frame["WAP519"] = 100  # Official not-detected value, constant on training.
    frame["WAP520"] = frame["WAP001"]  # Exact redundant predictor.
    frame["LONGITUDE"] = rng.normal(-7500, 10, size=n)
    frame["LATITUDE"] = frame["LONGITUDE"] * 3
    frame["BUILDINGID"] = rng.integers(0, 3, size=n)
    frame["USERID"] = np.arange(n)
    path = tmp_path / "trainingData.csv"
    frame.to_csv(path, index=False)
    original = make_uji(seed=1, csv_path=path)
    meta = original["metadata"]
    assert meta["used_features"] == 518
    assert meta["removed_constant_features"] == ["WAP519"]
    assert meta["removed_duplicate_features"] == {"WAP520": "WAP001"}
    assert "LONGITUDE" not in meta["feature_names"] and "LATITUDE" not in meta["feature_names"]
    heldout = np.r_[meta["split_indices"]["val"], meta["split_indices"]["test"]]
    frame.loc[heldout, "WAP519"] = -40  # Must not change the fitted constant-column decision.
    frame.loc[heldout, "WAP520"] += 40  # Nor may held-out values change duplicate selection.
    frame.loc[heldout, "WAP001"] += 1000
    frame.loc[heldout, "LONGITUDE"] += 100000
    frame.to_csv(path, index=False)
    altered = make_uji(seed=1, csv_path=path)
    np.testing.assert_array_equal(original["X_train"], altered["X_train"])
    np.testing.assert_array_equal(original["y_train"], altered["y_train"])
    assert altered["metadata"]["feature_names"] == meta["feature_names"]
    assert altered["metadata"]["train_mean"] == meta["train_mean"]
    np.testing.assert_allclose(original["X_train"].std(axis=0), 1., atol=1e-14)
    assert abs(original["y_train"].mean()) < 1e-10
    indices = [set(v) for v in meta["split_indices"].values()]
    assert not (indices[0] & indices[1] or indices[0] & indices[2] or indices[1] & indices[2])
    assert set.union(*indices) == set(range(n))


def test_leukemia_is_fixed_full_sample_optimization_problem(tmp_path):
    path = tmp_path / "leukemia.libsvm"
    lines = [f"{1 if i < 30 else -1} 1:{i+1} 7129:{(i % 9) - 3}" for i in range(72)]
    path.write_text("\n".join(lines) + "\n")
    data = make_leukemia(path)
    assert data["X"].shape == (72, 7129)
    assert data["y"].shape == (72,)
    assert data["metadata"]["has_true_support"] is False
    assert set(data) == {"X", "y", "metadata"}
    assert "X_test" not in data and "w_true" not in data
    np.testing.assert_allclose(data["X"].mean(axis=0), 0., atol=1e-14)
    assert abs(data["y"].mean()) < 1e-14


def test_benchmark_source_hash_is_enforced(tmp_path):
    path = tmp_path / "trainingData.csv"
    path.write_text("intentionally modified data")
    (tmp_path / "source.json").write_text(json.dumps({"files": {"trainingData.csv": {"sha256": "0" * 64}}}))
    with pytest.raises(ValueError, match="source.json"):
        make_uji(csv_path=path)
