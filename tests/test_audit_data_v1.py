"""Small controls for external-design pairing and X-only grouping; no optimizers."""
import json

import numpy as np
import pytest

import lasso_study.audit_data_v1 as audit_data
from lasso_study.audit_data_v1 import (
    exact_duplicate_columns, load_audit_source, make_audit_design,
    near_proportional_groups_chunked,
)


def source_fixture(source="news"):
    rng = np.random.default_rng(917)
    n = 80 if source == "news" else 72
    X = rng.normal(size=(n, 13))
    X[:, 1] = .125
    X[:, 3] = X[:, 2]
    X[:, 4] = -X[:, 2]
    return {"X_pool": X, "y_original": np.full(n, 123456.),
            "metadata": {"source": source, "source_seed": 0, "license": "fixture only",
                         "feature_names": [f"feature_{j}" for j in range(X.shape[1])],
                         "source_original_row_indices_0based": list(range(1000, 1000+n)),
                         "source_original_predictor_indices_0based": list(range(100, 113))}}


def test_pairing_truth_and_fixed_lambda_across_amplitude_and_c():
    pool = source_fixture()
    config = {"news_n": 24}
    base = make_audit_design(config, 400, "news", "strong", .5, source_data=pool)
    for amplitude in ("strong", "mixed_weak"):
        for c in (.5, 1.):
            current = make_audit_design(config, 400, "news", amplitude, c, source_data=pool)
            meta = current["metadata"]
            np.testing.assert_array_equal(current["X"], base["X"])
            assert meta["support"] == base["metadata"]["support"]
            assert meta["signs"] == base["metadata"]["signs"]
            assert meta["hashes"]["noise"] == base["metadata"]["hashes"]["noise"]
            assert meta["p"] == 12  # no block truncation or duplicate deletion
            assert meta["lambda"] == pytest.approx(c*np.sqrt(2*np.log(12)/24))
            support = meta["support"]
            expected = [1, 1, 1, .25, 1] if amplitude == "mixed_weak" else [1]*5
            np.testing.assert_array_equal(np.abs(current["w_true"][support]), expected)
            assert np.count_nonzero(current["w_true"]) == 5
            noise = current["y"] + meta["train_y_mean"] - current["X"] @ current["w_true"]
            first_noise = base["y"]+base["metadata"]["train_y_mean"]-base["X"] @ base["w_true"]
            np.testing.assert_allclose(noise, first_noise, atol=2e-15)
            np.testing.assert_allclose(current["X"].mean(axis=0), 0., atol=2e-15)
            np.testing.assert_allclose(current["X"].std(axis=0), 1., atol=2e-15)
            assert abs(current["y"].mean()) < 2e-15
            json.dumps(meta, allow_nan=False)
        a = make_audit_design(config, 400, "news", amplitude, .5, source_data=pool)
        b = make_audit_design(config, 400, "news", amplitude, 1., source_data=pool)
        np.testing.assert_array_equal(a["y"], b["y"])


def test_response_exclusion_selected_row_only_scaling_and_source_indices():
    source = source_fixture()
    first = make_audit_design({"news_n": 24}, 401, "news", "strong", 1., source_data=source)
    indices = first["metadata"]["source_pool_row_indices"]
    source["y_original"] = np.full(80, np.nan)
    other = np.setdiff1d(np.arange(80), indices)
    source["X_pool"][other] += 100000.
    again = make_audit_design({"news_n": 24}, 401, "news", "strong", 1., source_data=source)
    for key in ("X", "y", "w_true"):
        np.testing.assert_array_equal(first[key], again[key])
    meta = first["metadata"]
    assert meta["source_original_response_used"] is False
    assert meta["source_original_row_indices_0based"] == [1000+j for j in indices]
    assert meta["removed_constant_source_columns_0based"] == [1]
    assert meta["retained_source_columns_0based"] == [0]+list(range(2, 13))
    assert meta["retained_original_predictor_indices_0based"] == [100]+list(range(102, 113))
    assert meta["exact_duplicates"]["equal_nontrivial_group_count"] >= 1


def test_gene_keeps_all_rows_original_order_and_fixed_X_across_seeds():
    source = source_fixture("gene")
    first = make_audit_design({}, 500, "gene", "strong", .5, source_data=source)
    next_seed = make_audit_design({}, 501, "gene", "mixed_weak", 1., source_data=source)
    np.testing.assert_array_equal(first["X"], next_seed["X"])
    assert first["metadata"]["source_pool_row_indices"] == list(range(72))
    assert first["metadata"]["n"] == 72 and first["metadata"]["p"] == 12
    assert first["metadata"]["hashes"]["noise"] != next_seed["metadata"]["hashes"]["noise"]
    with pytest.raises(ValueError, match="complete"):
        make_audit_design({"gene_n": 60}, 500, "gene", "strong", 1., source_data=source)


def test_source_loader_uses_fixed_X_pool_and_discards_legacy_target(tmp_path, monkeypatch):
    path = tmp_path / "news.csv"
    path.write_text("url,timedelta,feature_a,feature_b,shares\n", encoding="utf-8")
    X = np.array([[1., 2.], [3., 4.], [5., 6.]])
    target = [1., 2., 3.]
    calls = []
    def fake_real(seed, csv_path):
        calls.append((seed, csv_path))
        return {"X_train": X, "y_train": np.asarray(target),
                "metadata": {"feature_names": ["feature_a", "feature_b"],
                             "split_indices": {"train": [7, 2, 9]}, "split_sha256": {"train": "fixed"},
                             "train_y_mean": float(target[0]), "license": "fixture"}}
    monkeypatch.setattr(audit_data, "make_real", fake_real)
    original = load_audit_source("news", path=path)
    target[:] = [np.nan, np.nan, np.nan]
    changed = load_audit_source("news", path=path)
    np.testing.assert_array_equal(original["X_pool"], changed["X_pool"])
    assert original["metadata"] == changed["metadata"]
    assert calls == [(0, path), (0, path)]
    assert original["metadata"]["source_original_row_indices_0based"] == [7, 2, 9]
    assert original["metadata"]["source_original_predictor_indices_0based"] == [0, 1]
    assert not any(key.startswith("y") for key in original)


def test_exact_equal_and_negative_groups_use_bytes_and_do_not_merge_linear_combinations():
    a = np.array([1., -2., 4., -3.])
    b = np.array([-1., 2., 3., -4.])
    X = np.column_stack([a, a, -a, b, a+b])
    exact = exact_duplicate_columns(X)
    assert exact["equal_groups"] == [[0, 1]]
    assert exact["signed_groups"] == [{"anchor_column_0based": 0, "columns_0based": [0, 1, 2],
                                       "member_signs_from_anchor": [1, 1, -1]}]


def test_full_pair_scan_direct_anchor_rule_negative_scale_and_cache():
    a = np.array([1., -2., 4., -3., 5., 2.])
    b = np.array([-1., 2., 3., -4., 2., 7.])
    X = np.column_stack([a, 2*a, -3*a, b, a+b])
    result = near_proportional_groups_chunked(X, block_size=2)
    assert result["all_unordered_pairs_screened"] is True
    assert result["unique_pairs_scanned"] == 10
    assert result["block_size"] == 2
    assert result["diagnostics"]["peak_gram_bytes_upper_bound"] == 32
    assert result["groups_by_tolerance"]["1e-12"] == result["groupings"]["1e-12"]["groups"]
    for tolerance, details in result["groupings"].items():
        assert [group["columns_0based"] for group in details["groups"]] == [[0, 1, 2], [3], [4]]
        np.testing.assert_allclose(details["groups"][0]["member_scales_from_anchor"], [1., 2., -3.])
        assert max(details["groups"][0]["relative_l2_residuals_to_anchor"]) <= float(tolerance)
    repeated = near_proportional_groups_chunked(X, block_size=2)
    assert repeated["cache_hit"] is True and repeated["scan_elapsed_sec"] == 0
    assert repeated["groupings"] == result["groupings"]
    repeated["groupings"]["1e-12"]["groups"][0]["columns_0based"].append(99)
    assert 99 not in near_proportional_groups_chunked(X, block_size=2)["groupings"]["1e-12"]["groups"][0]["columns_0based"]
    json.dumps(result, allow_nan=False)


def test_near_groups_do_not_chain_through_a_non_anchor_member():
    # b is within tol of a, c is within tol of b but outside tol of anchor a.
    a = np.array([1., 0., 0., 0.])
    b = np.array([1., 7e-7, 0., 0.])
    c = np.array([1., 14e-7, 0., 0.])
    result = near_proportional_groups_chunked(np.column_stack([a, b, c]), tolerances=(1e-6,), block_size=2)
    assert [g["columns_0based"] for g in result["groupings"]["1e-06"]["groups"]] == [[0, 1], [2]]


def test_invalid_protocol_and_group_inputs_are_explicit():
    source = source_fixture()
    with pytest.raises(ValueError, match="fixes"):
        make_audit_design({"noise_std": 2.}, 400, "news", "strong", 1., source_data=source)
    with pytest.raises(ValueError, match="c in"):
        make_audit_design({}, 400, "news", "strong", .2, source_data=source)
    source["metadata"]["source_seed"] = 2
    with pytest.raises(ValueError, match="fixed"):
        make_audit_design({"news_n": 24}, 400, "news", "strong", 1., source_data=source)
    with pytest.raises(ValueError, match="source_seed"):
        load_audit_source("news", source_seed=3)
    with pytest.raises(ValueError, match="nonzero"):
        near_proportional_groups_chunked(np.column_stack([np.ones(4), np.zeros(4)]), block_size=2)
