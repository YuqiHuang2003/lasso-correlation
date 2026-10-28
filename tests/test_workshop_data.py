"""Small scientific-control tests for the separate workshop data protocol."""
import json

import numpy as np
import pytest

from lasso_study.data_v2 import make_synthetic as make_v2
from lasso_study.workshop_data import make_real_design, make_workshop_design


SMALL = {"n": 80, "p": 100, "sparsity": 5, "block_size": 20,
         "noise_std": 1., "lambda_factor": 1.}


def test_pairing_across_families_rho_layout_and_amplitude():
    outputs = {}
    for family in ("block", "toeplitz"):
        for rho in (0., .9):
            for layout in ("dispersed", "clustered"):
                for amplitude in ("strong", "mixed_weak"):
                    outputs[family, rho, layout, amplitude] = make_workshop_design(
                        SMALL, 100, family, rho, layout, amplitude)
    base = outputs["block", 0., "dispersed", "strong"]
    for (family, rho, layout, amplitude), data in outputs.items():
        meta = data["metadata"]
        assert meta["hashes"]["noise"] == base["metadata"]["hashes"]["noise"]
        assert meta["hashes"]["innovations"] == base["metadata"]["hashes"]["innovations"]
        assert meta["hashes"]["ordered_signs"] == base["metadata"]["hashes"]["ordered_signs"]
        np.testing.assert_array_equal(data["X"], outputs[family, rho, "dispersed", "strong"]["X"])
        assert meta["support"] == outputs["block", 0., layout, "strong"]["metadata"]["support"]
        assert meta["signs"] == base["metadata"]["signs"]
        _, count = np.unique(np.array(meta["support"]) // 20, return_counts=True)
        np.testing.assert_array_equal(count, np.full(len(count), 1 if layout == "dispersed" else 5))
        assert np.count_nonzero(data["w_true"]) == 5
        assert meta["lambda"] == pytest.approx(np.sqrt(2*np.log(100)/80))
        noise = data["y"] + meta["train_y_mean"] - data["X"] @ data["w_true"]
        base_noise = base["y"] + base["metadata"]["train_y_mean"] - base["X"] @ base["w_true"]
        np.testing.assert_allclose(noise, base_noise, atol=1e-13)
        np.testing.assert_allclose(data["X"].mean(axis=0), 0., atol=1e-14)
        np.testing.assert_allclose(data["X"].std(axis=0), 1., atol=1e-14)
        assert abs(data["y"].mean()) < 1e-14
        json.dumps(meta, allow_nan=False)
    np.testing.assert_array_equal(base["X"], outputs["toeplitz", 0., "dispersed", "strong"]["X"])
    np.testing.assert_array_equal(base["y"], outputs["toeplitz", 0., "dispersed", "strong"]["y"])


def test_block_family_matches_frozen_v2_latent_and_support_construction():
    data = make_workshop_design(SMALL, 101, "block", .6, "clustered", "strong")
    v2 = make_v2(seed=101, rho=.6, n=80, p=100, sparsity=5, block_size=20,
                 noise_std=1., n_test=10, layout="clustered")
    for key in ("X", "y", "w_true"):
        np.testing.assert_array_equal(data[key], v2[key])
    assert data["metadata"]["hashes"]["noise"] == v2["metadata"]["hashes"]["noise"]


def test_mixed_weak_uses_every_fourth_support_position_and_shared_signs():
    settings = dict(SMALL, p=200, sparsity=10)
    strong = make_workshop_design(settings, 102, "toeplitz", .9, "dispersed", "strong")
    mixed = make_workshop_design(settings, 102, "toeplitz", .9, "dispersed", "mixed_weak")
    positions = mixed["metadata"]["support"]
    np.testing.assert_array_equal(np.sign(strong["w_true"]), np.sign(mixed["w_true"]))
    np.testing.assert_array_equal(np.abs(mixed["w_true"][positions]), [1, 1, 1, .25, 1, 1, 1, .25, 1, 1])
    assert mixed["metadata"]["weak_positions_in_support_1based"] == [4, 8]
    assert strong["metadata"]["lambda"] == mixed["metadata"]["lambda"]


def test_ar1_population_correlation_and_stationary_first_coordinate():
    settings = {"n": 6000, "p": 40, "sparsity": 2, "block_size": 20}
    data = make_workshop_design(settings, 17, "toeplitz", .7, "dispersed", "strong")
    corr = np.corrcoef(data["X"], rowvar=False)
    assert corr[0, 1] == pytest.approx(.7, abs=.035)
    assert corr[0, 2] == pytest.approx(.7**2, abs=.035)
    assert corr[19, 20] == pytest.approx(.7, abs=.035)  # No artificial block boundary.
    assert corr[0, 10] == pytest.approx(.7**10, abs=.035)
    # Before column standardization the initial coordinate also has variance 1.
    assert data["metadata"]["train_scale"][0] == pytest.approx(1., abs=.035)


def _source_fixture():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(180, 111))
    X[:, 7] = .125  # Repeated float whose naive std can have roundoff residue.
    return {"X_train": X, "y_train": rng.normal(size=180),
            "metadata": {"seed": 0, "feature_names": [f"WAP{i:03d}" for i in range(1, 112)],
                         "csv_sha256": "fixture_source_hash", "license": "fixture only",
                         "split_indices": {"train": list(range(1000, 1180))}}}


def test_real_design_ignores_original_response_and_preserves_column_provenance():
    source = _source_fixture()
    config = {"real_n": 80, "real_sparsity": 5, "p": 1000, "block_size": 20}
    first = make_real_design(config, 100, "dispersed", "strong", source=source)
    source["y_train"] = np.full(180, np.nan)
    changed_response = make_real_design(config, 100, "dispersed", "strong", source=source)
    for key in ("X", "y", "w_true"):
        np.testing.assert_array_equal(first[key], changed_response[key])
    meta = first["metadata"]
    assert meta["n"] == 80 and meta["p"] == 100 and meta["sparsity"] == 5
    assert meta["source_original_response_used"] is False
    assert meta["removed_constant_source_columns_0based"] == [7]
    assert meta["column_counts"] == {"source": 111, "after_constant_removal": 110,
                                    "after_config_p_cap": 110, "after_complete_block_truncation": 100}
    assert meta["retained_source_columns_0based"] == [j for j in range(101) if j != 7]
    assert len(meta["removed_block_tail_source_columns_0based"]) == 10
    assert meta["retained_original_wap_columns_0based"] == meta["retained_source_columns_0based"]
    assert meta["source_original_csv_row_indices"] == [1000+j for j in meta["source_pool_row_indices"]]
    assert np.min(np.ptp(first["X"], axis=0)) > 0
    np.testing.assert_allclose(first["X"].mean(axis=0), 0., atol=1e-14)
    np.testing.assert_allclose(first["X"].std(axis=0), 1., atol=1e-14)
    json.dumps(meta, allow_nan=False)


def test_real_preprocessing_uses_only_selected_rows_and_pairing_ignores_layout():
    source = _source_fixture()
    config = {"n_real": 80, "real_sparsity": 5, "p": 1000, "block_size": 20}
    first = make_real_design(config, 102, "dispersed", "strong", source=source)
    other_rows = np.setdiff1d(np.arange(180), first["metadata"]["source_pool_row_indices"])
    altered = {**source, "X_train": source["X_train"].copy()}
    altered["X_train"][other_rows] += 10000.
    repeated = make_real_design(config, 102, "dispersed", "strong", source=altered)
    clustered = make_real_design(config, 102, "clustered", "mixed_weak", source=source)
    np.testing.assert_array_equal(first["X"], repeated["X"])
    np.testing.assert_array_equal(first["y"], repeated["y"])
    np.testing.assert_array_equal(first["X"], clustered["X"])
    assert first["metadata"]["hashes"]["noise"] == clustered["metadata"]["hashes"]["noise"]
    assert first["metadata"]["signs"] == clustered["metadata"]["signs"]
    source["metadata"]["seed"] = 2
    with pytest.raises(ValueError, match="fixed"):
        make_real_design(config, 102, source=source)


def test_invalid_protocol_and_insufficient_real_capacity_are_explicit():
    with pytest.raises(ValueError, match="Conflicting"):
        make_workshop_design({**SMALL, "sigma": 2.}, 100, "block", .9, "clustered", "strong")
    with pytest.raises(ValueError, match="protocol fixes"):
        make_workshop_design({**SMALL, "weak_every": 3}, 100, "block", .9, "clustered", "mixed_weak")
    with pytest.raises(ValueError, match="support does not fit"):
        make_real_design({"real_n": 80, "real_sparsity": 10, "p": 1000, "block_size": 20},
                         100, source=_source_fixture())
