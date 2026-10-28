"""Small exact-coordinate replay controls; no model fitting or full-study replay."""
from collections import Counter
import copy
import hashlib
import json

import numpy as np
import pytest

import lasso_study.column_permutation_v1 as module
from lasso_study.column_permutation_v1 import (
    inverse_coefficients, load_sources, make_permutations, make_target,
    permuted_data, target_id, target_parameters,
)
from lasso_study.data import array_sha256
from lasso_study.solvers import certificate


def test_registered_grid_and_source_prefixed_ids():
    params = target_parameters({})
    assert len(params) == len({target_id(p) for p in params}) == 56
    assert Counter(p["source"] for p in params) == {"uji": 40, "news": 8, "gene": 8}
    assert all(p["c"] == 1. and p["layout"] is not None for p in params if p["source"] == "uji")
    assert all(p["layout"] is None for p in params if p["source"] != "uji")
    assert target_id(dict(source="uji", seed=300, layout="dispersed", amplitude="strong", c=1.)) == "uji_seed300_dispersed_strong"
    assert target_id(dict(source="news", seed=401, layout=None, amplitude="strong", c=.5)) == "news_seed401_c0.5_strong"


def test_config_grid_cannot_silently_drop_conditions():
    with pytest.raises(ValueError, match="seeds"):
        target_parameters({"uji_seeds": [300]*10})
    with pytest.raises(ValueError, match="three"):
        target_parameters({"sources": ["uji", "news"]})
    with pytest.raises(ValueError, match="c levels"):
        target_parameters({"external_c": [1.]})


def test_permutations_are_distinct_shared_json_safe_and_reproducible():
    first = make_permutations(7); again = make_permutations(7)
    assert first == again
    assert [r["order_id"] for r in first] == ["identity", "perm_0", "perm_1", "perm_2", "perm_3", "perm_4"]
    assert len({tuple(r["permutation"]) for r in first}) == 6
    assert first[0]["permutation"] == list(range(7))
    for item in first:
        permutation = np.array(item["permutation"], dtype=np.int64)
        inverse = np.array(item["inverse_permutation"], dtype=np.int64)
        np.testing.assert_array_equal(permutation[inverse], np.arange(7))
        assert item["permutation_sha256"] == array_sha256(permutation)
        assert item["inverse_sha256"] == array_sha256(inverse)
    json.dumps(first, allow_nan=False)


def test_small_factorial_capacity_and_deterministic_replacement():
    small = make_permutations(3)
    assert len({tuple(r["permutation"]) for r in small}) == 6
    assert any(r["replacement_used"] for r in small)
    assert all(r["seed"] >= 705 for r in small if r["replacement_used"])
    assert small == make_permutations(3)
    with pytest.raises(ValueError, match="three columns"):
        make_permutations(2)
    with pytest.raises(ValueError, match="five distinct"):
        make_permutations(7, seeds=[700]*5)


def test_inverse_scatter_and_objective_certificate_invariance():
    rng = np.random.default_rng(17); X = np.asfortranarray(rng.normal(size=(12, 7)))
    y = rng.normal(size=12); truth = np.array([1., 0., -.25, 0., 0., 1., 0.]); coefficient = rng.normal(size=7)
    data = {"X": X, "y": y, "w_true": truth, "metadata": {"lambda": .2}}
    expected = certificate(X, y, coefficient, .2)
    for item in make_permutations(7):
        permutation = np.array(item["permutation"])
        changed = permuted_data(data, permutation)
        np.testing.assert_array_equal(changed["X"], X[:, permutation])
        np.testing.assert_array_equal(changed["w_true"], truth[permutation])
        np.testing.assert_array_equal(changed["y"], y)
        assert changed["X"].flags.f_contiguous and changed["X"].dtype == np.float64
        permuted_coef = coefficient[permutation]
        np.testing.assert_array_equal(inverse_coefficients(permuted_coef, permutation), coefficient)
        batch = np.vstack([permuted_coef, truth[permutation]])
        np.testing.assert_array_equal(inverse_coefficients(batch, permutation), np.vstack([coefficient, truth]))
        np.testing.assert_allclose(changed["X"] @ permuted_coef, X @ coefficient, atol=2e-15)
        actual = certificate(changed["X"], changed["y"], permuted_coef, changed["metadata"]["lambda"])
        for key in ("primal", "dual", "gap", "normalized_gap", "dual_feasibility", "kkt_inf"):
            assert actual[key] == pytest.approx(expected[key], rel=2e-14, abs=2e-14)
    # The caller can mutate its permuted copy without touching frozen originals.
    changed["y"][0] += 1.
    assert changed["y"][0] != y[0]


def test_invalid_coordinate_maps_are_rejected():
    for permutation in ([0, 0, 2], [0, 1, 3], [0., 1., 2.], [True, False, True]):
        with pytest.raises(ValueError, match="permutation"):
            inverse_coefficients(np.ones(3), permutation)
    with pytest.raises(ValueError, match="finite"):
        inverse_coefficients([1., np.nan], [1, 0])


def historical_fixture(tmp_path, params):
    base = tmp_path / params["source"]; (base/"datasets").mkdir(parents=True); (base/"references").mkdir()
    (base/"reference_coefficients").mkdir()
    rng = np.random.default_rng(19); X = np.asfortranarray(rng.normal(size=(9, 6)))
    y = rng.normal(size=9); truth = np.array([1., 0., -.25, 1., 0., 1.])
    hashes = {key: array_sha256(value) for key, value in (("X", X), ("y", y), ("w_true", truth))}
    generated = {"X": X, "y": y, "w_true": truth, "metadata": {"hashes": hashes, "lambda": .2,
        "n": 9, "p": 6, "seed": params["seed"], "amplitude": params["amplitude"], "layout": params["layout"],
        "source": params["source"], "c": params["c"]}}
    identifier = target_id(params); cp = base/"reference_coefficients"/(identifier+".npz")
    np.savez_compressed(cp, ref_sklearn=truth, ref_skglm=truth+.01, w_true=truth)
    data_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    config_path = base/"baseline_config.json"; config_path.write_text('{"fixture": true}')
    config_hash = module.file_sha256(config_path)
    ds = {**generated["metadata"], "dataset_hash": data_hash, "numerical_signature": "fixture_sig", "config_sha256": config_hash}
    ref = {"lambda": .2, "dataset_hash": data_hash, "numerical_signature": "fixture_sig", "config_sha256": config_hash,
           "coefficient_path": f"reference_coefficients/{identifier}.npz", "coefficient_sha256": module.file_sha256(cp), "reference_accepted": False}
    (base/"datasets"/(identifier+".json")).write_text(json.dumps(ds)); (base/"references"/(identifier+".json")).write_text(json.dumps(ref))
    contexts = {params["source"]: {"source": params["source"], "prepared": {"fixture": True}, "baseline_config": {"fixture": True},
                                   "baseline_config_path": config_path, "baseline_config_sha256": config_hash, "baseline_path": base}}
    return generated, contexts, base


def test_uji_and_external_replay_history_and_detect_tampering(tmp_path, monkeypatch):
    for params in (dict(source="uji", seed=300, layout="dispersed", amplitude="strong", c=1.),
                   dict(source="news", seed=401, layout=None, amplitude="mixed_weak", c=.5)):
        generated, contexts, base = historical_fixture(tmp_path, params)
        if params["source"] == "uji":
            monkeypatch.setattr(module, "make_real_design", lambda *args, **kwargs: copy.deepcopy(generated))
        else:
            monkeypatch.setattr(module, "make_audit_design", lambda *args, **kwargs: copy.deepcopy(generated))
        data = make_target({}, params, contexts)
        for name in ("X", "y", "w_true"):
            np.testing.assert_array_equal(data[name], generated[name])
        assert data["historical_reference"]["metadata"]["reference_accepted"] is False  # uncertain old refs remain available
        np.testing.assert_array_equal(data["historical_reference"]["ref_sklearn"], generated["w_true"])
        assert data["origin"]["old_target_id"] == target_id(params)
        generated["metadata"]["lambda"] = .3
        with pytest.raises(ValueError, match="lambda"):
            make_target({}, params, contexts)
        generated["metadata"]["lambda"] = .2
        generated["X"][0, 0] += 1.
        with pytest.raises(ValueError, match="recorded hashes"):
            make_target({}, params, contexts)


def test_load_sources_prepares_each_fixed_source_once(tmp_path, monkeypatch):
    uji_config = tmp_path/"uji.json"; ext_config = tmp_path/"external.json"
    uji_config.write_text('{"kind":"uji"}'); ext_config.write_text('{"kind":"external"}')
    calls = []
    def uji_loader(seed):
        calls.append(("uji", seed)); return {"fixture": "uji"}
    def external_loader(source, source_seed):
        calls.append((source, source_seed)); return {"fixture": source}
    monkeypatch.setattr(module, "make_uji", uji_loader); monkeypatch.setattr(module, "load_audit_source", external_loader)
    sources = load_sources({"uji_config_path": str(uji_config), "external_config_path": str(ext_config)})
    assert calls == [("uji", 0), ("news", 0), ("gene", 0)]
    assert sources["uji"]["baseline_config"] == {"kind": "uji"}
    assert sources["news"]["baseline_config"] == sources["gene"]["baseline_config"] == {"kind": "external"}
    assert sources["gene"]["baseline_config_sha256"] == module.file_sha256(ext_config)


def test_reference_only_config_change_is_rejected_before_replay(tmp_path, monkeypatch):
    params = dict(source="news", seed=401, layout=None, amplitude="strong", c=.5)
    generated, contexts, _ = historical_fixture(tmp_path, params)
    monkeypatch.setattr(module, "make_audit_design", lambda *args, **kwargs: copy.deepcopy(generated))
    make_target({}, params, contexts)
    # This change need not alter generated X/y or lambda, but it is a different
    # frozen solver configuration and must never silently pass the replay gate.
    contexts["news"]["baseline_config_path"].write_text('{"fixture": true, "references": {"tol": 0.1}}')
    with pytest.raises(ValueError, match="config hashes differ"):
        make_target({}, params, contexts)
