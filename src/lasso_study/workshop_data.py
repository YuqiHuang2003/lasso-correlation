"""Paired, fixed-noise designs for the independent workshop study.

The frozen v1/v2 generators are imported only for public hashing and the UJI
source preparation. Their numerical code and configurations are unchanged.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .data import array_sha256
from .data_v2 import make_uji


def _value(config: dict, primary: str, alias: str, default):
    if primary in config and alias in config and config[primary] != config[alias]:
        raise ValueError(f"Conflicting {primary}/{alias} settings.")
    return config.get(primary, config.get(alias, default))


def _integer(name: str, value, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _settings(config: dict) -> dict:
    settings = {
        "n": _integer("n", config.get("n", 500), 2),
        "p": _integer("p", config.get("p", 1000), 2),
        "sparsity": _integer("sparsity", _value(config, "sparsity", "s", 20)),
        "block_size": _integer("block_size", config.get("block_size", 20)),
        "noise_std": float(_value(config, "noise_std", "sigma", 1.)),
        "lambda_factor": float(_value(config, "lambda_factor", "c", 1.)),
        "weak_amplitude": float(config.get("weak_amplitude", .25)),
        "weak_every": _integer("weak_every", config.get("weak_every", 4)),
    }
    if settings["p"] % settings["block_size"]:
        raise ValueError("p must be a multiple of block_size.")
    if not np.isfinite(settings["noise_std"]) or settings["noise_std"] <= 0:
        raise ValueError("noise_std must be positive and finite.")
    if not np.isfinite(settings["lambda_factor"]) or settings["lambda_factor"] <= 0:
        raise ValueError("lambda_factor must be positive and finite.")
    if settings["weak_amplitude"] != .25 or settings["weak_every"] != 4:
        raise ValueError("This workshop protocol fixes weak_amplitude=0.25 and weak_every=4.")
    return settings


def _streams(seed: int) -> list[np.random.Generator]:
    seed = _integer("seed", seed, 0)
    return [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(8)]


def _truth(rngs, p: int, sparsity: int, block_size: int, layout: str,
           amplitude: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if layout not in {"dispersed", "clustered"}:
        raise ValueError("layout must be dispersed or clustered.")
    if amplitude not in {"strong", "mixed_weak"}:
        raise ValueError("amplitude must be strong or mixed_weak.")
    per_group = 1 if layout == "dispersed" else 5
    groups = p // block_size
    if block_size < per_group or sparsity % per_group or sparsity > per_group * groups:
        raise ValueError("Requested support does not fit the layout: dispersed needs s <= p/block_size; "
                         "clustered needs five signals per group and s divisible by five.")
    # Identical stream assignments/order to v2 make_synthetic: layout changes
    # neither the complete group order nor the independent ordered sign vector.
    group_order = rngs[4].permutation(groups)
    within = np.stack([rngs[4].permutation(block_size) for _ in range(groups)])
    complete_support = (group_order[:, None] * block_size + within[group_order, :per_group]).reshape(-1)
    support = complete_support[:sparsity]
    ordered_signs = rngs[5].choice(np.array([-1., 1.]), size=p)
    magnitudes = np.ones(sparsity, dtype=np.float64)
    if amplitude == "mixed_weak":
        magnitudes[3::4] = .25
    coefficients = np.zeros(p, dtype=np.float64)
    coefficients[support] = ordered_signs[:sparsity] * magnitudes
    return coefficients, support, ordered_signs, magnitudes, group_order


def _standardize(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not np.isfinite(raw).all() or np.any(np.ptp(raw, axis=0) == 0):
        raise ValueError("Design must be finite and contain no constant columns before standardization.")
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0, ddof=0)
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("Design has an invalid training scale.")
    # Match the frozen v2 column-major design representation and the solver
    # input convention, including reproducible matrix-vector accumulation.
    return np.asfortranarray((raw - mean) / scale, dtype=np.float64), mean, scale


def _response(X: np.ndarray, rngs, settings: dict, seed: int, family: str,
              rho: float | None, layout: str, amplitude: str, mean: np.ndarray,
              scale: np.ndarray, feature_metadata: dict) -> dict:
    n, p = X.shape
    w_true, support, ordered_signs, magnitudes, group_order = _truth(
        rngs, p, settings["sparsity"], settings["block_size"], layout, amplitude)
    standard_noise = rngs[6].normal(size=n)
    noise = settings["noise_std"] * standard_noise
    signal = X @ w_true
    raw_y = signal + noise
    y_mean = float(raw_y.mean())
    y = np.asarray(raw_y - y_mean, dtype=np.float64)
    signal_variance, noise_variance = float(np.var(signal)), float(np.var(noise))
    lam = float(settings["lambda_factor"] * settings["noise_std"] * np.sqrt(2*np.log(p)/n))
    hashes = {
        "X": array_sha256(X), "y": array_sha256(y), "w_true": array_sha256(w_true),
        "support_order": array_sha256(support), "ordered_signs": array_sha256(ordered_signs),
        "noise": array_sha256(noise), "standard_noise": array_sha256(standard_noise),
        "magnitudes": array_sha256(magnitudes), "group_order": array_sha256(group_order),
    }
    hashes.update(feature_metadata.pop("hashes", {}))
    metadata = {
        "kind": "workshop_design", "protocol": "fixed_sigma_absolute_lambda_workshop_v1",
        "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": int(seed), "family": family, "rho": rho, "layout": layout, "amplitude": amplitude,
        "n": n, "p": p, "sparsity": settings["sparsity"], "s": settings["sparsity"],
        "block_size": settings["block_size"], "support": support.tolist(), "support_order": support.tolist(),
        "support_sorted": sorted(support.tolist()), "signs": ordered_signs[:settings["sparsity"]].tolist(),
        "magnitudes": magnitudes.tolist(), "true_coefficients": w_true[support].tolist(),
        "weak_positions_in_support_1based": (np.flatnonzero(magnitudes < 1) + 1).tolist(),
        "noise_std": settings["noise_std"], "sigma": settings["noise_std"],
        "noise_scaling": "Fixed sigma; same seed shares the Gaussian noise realization across families/rho/layout/amplitude at matching n",
        "lambda_factor": settings["lambda_factor"], "c": settings["lambda_factor"], "lambda": lam,
        "lambda_rule": "c*sigma*sqrt(2*log(p)/n); independent of signal and lambda_max",
        "signal_var_train": signal_variance, "noise_var_realized_train": noise_variance,
        "snr_population": signal_variance / settings["noise_std"]**2,
        "snr_realized_train": signal_variance / noise_variance,
        "train_mean": mean.tolist(), "train_scale": scale.tolist(), "train_y_mean": y_mean,
        "coefficient_coordinates": "X centered and population-standardized on this generated training design; ddof=0",
        "rng": "NumPy PCG64 via SeedSequence(seed).spawn(8)",
        "rng_stream_roles": {"0": "block common factors", "1": "independent feature innovations for both synthetic families",
                             "2": "real-design row subsampling", "3": "reserved",
                             "4": "group/within-group support order", "5": "ordered coefficient signs",
                             "6": "response Gaussian noise", "7": "reserved"},
        "layout_groups_scope": "Consecutive index blocks define support layout. Only family=block has independent covariance blocks; Toeplitz and real-design blocks are bookkeeping partitions.",
        "hashes": hashes,
        **feature_metadata,
    }
    return {"X": X, "y": y, "w_true": w_true, "metadata": metadata}


def make_workshop_design(config: dict, seed: int, family: str, rho: float,
                         layout: str, amplitude: str) -> dict:
    """Create block or stationary AR(1) Gaussian predictors and known sparse truth.

    At rho=0 the two families deliberately give identical paired designs, rather
    than independent replications. The AR(1) first column equals its standard
    normal innovation; subsequent columns use rho*x_prev+sqrt(1-rho**2)*e.
    """
    settings = _settings(config)
    if family not in {"block", "toeplitz"}:
        raise ValueError("family must be block or toeplitz; use make_real_design for UJI.")
    if not np.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("rho must lie in [0,1).")
    n, p, block_size = settings["n"], settings["p"], settings["block_size"]
    rngs = _streams(seed)
    common = rngs[0].normal(size=(n, p // block_size))
    innovations = rngs[1].normal(size=(n, p))
    if family == "block":
        raw = np.sqrt(rho) * np.repeat(common, block_size, axis=1) + np.sqrt(1-rho) * innovations
        covariance = "Population variance 1; correlation rho inside each block, 0 between blocks"
    else:
        raw = np.empty_like(innovations)
        raw[:, 0] = innovations[:, 0]
        for column in range(1, p):
            raw[:, column] = rho * raw[:, column-1] + np.sqrt(1-rho**2) * innovations[:, column]
        covariance = "Stationary AR(1): population correlation rho**abs(j-k), including across layout-block boundaries"
    X, mean, scale = _standardize(raw)
    feature_metadata = {
        "design_covariance": covariance, "features_are_synthetic": True,
        "rho_zero_family_pairing": "block and toeplitz use exactly the same innovations, so rho=0 yields identical X",
        "hashes": {"raw_X": array_sha256(raw), "independent_train": array_sha256(innovations),
                   "innovations": array_sha256(innovations), "latent_train": array_sha256(common)},
    }
    return _response(X, rngs, settings, seed, family, float(rho), layout, amplitude, mean, scale, feature_metadata)


def make_real_design(config: dict, seed: int, layout: str = "dispersed", amplitude: str = "strong", *,
                     source: dict | None = None) -> dict:
    """Semisynthetic continuous response on a fixed UJI seed-0 training-X pool.

    Default real_n=1000 and real_sparsity=15 intentionally differ from the main
    synthetic arm. Original UJI responses are parsed by source preparation but
    never used here to generate y, select predictors/rows or choose parameters.
    The optional source argument supports isolated small, leakage-check tests.
    """
    settings = _settings(config)
    settings["n"] = _integer("real_n", _value(config, "real_n", "n_real", 1000), 2)
    settings["sparsity"] = _integer("real_sparsity", config.get("real_sparsity", 15))
    if source is None:
        source = make_uji(seed=0)
    source_metadata = source.get("metadata", {})
    if source_metadata.get("seed", 0) != 0:
        raise ValueError("The workshop real-design source is fixed to make_uji(seed=0).")
    pool = np.asarray(source["X_train"], dtype=np.float64)
    if pool.ndim != 2 or not np.isfinite(pool).all() or settings["n"] > len(pool):
        raise ValueError("Real-design pool must be finite and contain enough rows for sampling without replacement.")
    rngs = _streams(seed)
    selected_rows = rngs[2].choice(len(pool), size=settings["n"], replace=False)
    sampled = pool[selected_rows]
    # ptp identifies exact constants even if summation roundoff gives a tiny
    # nonzero standard deviation for repeated floating-point values.
    constant_mask = np.ptp(sampled, axis=0) == 0
    nonconstant_indices = np.flatnonzero(~constant_mask)
    capped_count = min(len(nonconstant_indices), settings["p"])
    retained_count = capped_count // settings["block_size"] * settings["block_size"]
    if retained_count == 0:
        raise ValueError("Not enough nonconstant source features to form one complete layout block.")
    source_columns = nonconstant_indices[:retained_count]
    after_cap = nonconstant_indices[:capped_count]
    tail_columns = after_cap[retained_count:]
    cap_columns = nonconstant_indices[capped_count:]
    X, mean, scale = _standardize(sampled[:, source_columns])
    settings["p"] = retained_count
    names = source_metadata.get("feature_names", [f"source_column_{j}" for j in range(pool.shape[1])])
    if len(names) != pool.shape[1]:
        raise ValueError("Source feature names do not match the X_train column count.")
    selected_names = [names[j] for j in source_columns]
    original_wap_indices = [int(name[3:])-1 if name.startswith("WAP") and name[3:].isdigit() else None
                            for name in selected_names]
    source_split_rows = source_metadata.get("split_indices", {}).get("train")
    original_rows = ([int(source_split_rows[j]) for j in selected_rows]
                     if source_split_rows is not None else None)
    feature_metadata = {
        "features_are_synthetic": False, "source_dataset": "UJIIndoorLoc seed-0 training design",
        "source_page": source_metadata.get("source_page", "https://archive.ics.uci.edu/dataset/310/ujiindoorloc"),
        "source_license": source_metadata.get("license", "CC BY 4.0 for official UJIIndoorLoc"),
        "source_doi": source_metadata.get("doi", "10.24432/C5MS59"),
        "source_csv_sha256": source_metadata.get("csv_sha256"), "source_preprocessing_seed": 0,
        "source_training_split_sha256": source_metadata.get("split_sha256", {}).get("train"),
        "source_pool_shape": list(pool.shape), "source_pool_feature_names": names,
        "source_original_response_used": False,
        "response_generation": "Fresh continuous Gaussian linear response X*w_true + sigma*epsilon; original real response not used for this semisynthetic task",
        "source_features_scope": "Real WiFi RSSI-derived standardized WAP columns; official non-detection code retained by source preparation",
        "row_sampling": "SeedSequence stream 2; uniform choice without replacement from fixed seed-0 training pool",
        "source_pool_row_indices": selected_rows.tolist(), "source_original_csv_row_indices": original_rows,
        "column_pipeline": "subsample rows, remove exact constant columns, retain first min(config.p, available) in source WAP order, truncate final incomplete index block, re-center/re-scale on sampled rows",
        "column_counts": {"source": pool.shape[1], "after_constant_removal": len(nonconstant_indices),
                          "after_config_p_cap": capped_count, "after_complete_block_truncation": retained_count},
        "removed_constant_source_columns_0based": np.flatnonzero(constant_mask).tolist(),
        "removed_p_cap_source_columns_0based": cap_columns.tolist(),
        "removed_block_tail_source_columns_0based": tail_columns.tolist(),
        "retained_source_columns_0based": source_columns.tolist(), "feature_names": selected_names,
        "retained_original_wap_columns_0based": original_wap_indices,
        "real_design_comparison_scope": "Independent robustness arm with real_sparsity default 15 and real_n default 1000; not a difficulty-matched comparison with synthetic n500/p1000/s20",
        "hashes": {"source_pool_X": array_sha256(pool), "source_pool_row_indices": array_sha256(selected_rows),
                   "sampled_source_X": array_sha256(sampled), "retained_source_columns": array_sha256(source_columns),
                   "raw_X": array_sha256(sampled[:, source_columns])},
    }
    return _response(X, rngs, settings, seed, "real_design", None, layout, amplitude, mean, scale, feature_metadata)
