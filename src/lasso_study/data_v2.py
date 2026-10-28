"""Version 2 paired recovery design and literature-aligned real benchmarks.

This module is separate from the frozen v1 data generator. Synthetic noise has
a fixed standard deviation across correlation, sparsity and support layout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import array_sha256, file_sha256


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = ROOT / "data/benchmarks"
UJI_PAGE = "https://archive.ics.uci.edu/dataset/310/ujiindoorloc"
LEUKEMIA_PAGE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#leukemia"


def _verify_source_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Run scripts/download_benchmarks.py first; missing {path}")
    actual = file_sha256(path)
    source_path = path.parent / "source.json"
    if source_path.is_file():
        record = json.loads(source_path.read_text(encoding="utf-8"))
        expected = record.get("files", {}).get(path.name, {}).get("sha256")
        if expected is not None and actual != expected:
            raise ValueError(f"Benchmark content does not match source.json: {path}")
    return actual


def _fit_standardizer(train: np.ndarray, *others: np.ndarray, deduplicate: bool = False,
                      drop_constants: bool = True):
    """Fit feature selection, centering and population scale using train only."""
    train = np.asarray(train, dtype=np.float64)
    if train.ndim != 2 or min(train.shape) == 0 or not np.isfinite(train).all():
        raise ValueError("Training predictors must be a nonempty finite matrix.")
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    constant = scale == 0.
    keep = ~constant if drop_constants else np.ones(train.shape[1], dtype=bool)
    duplicates = {}
    if deduplicate:
        seen = {}
        for column in np.flatnonzero(keep):
            values = np.ascontiguousarray(train[:, column])
            fingerprint = hashlib.sha256(values.tobytes()).digest()
            candidates = seen.get(fingerprint, [])
            duplicate_of = next((j for j in candidates if np.array_equal(train[:, j], values)), None)
            if duplicate_of is not None:
                keep[column] = False
                duplicates[int(column)] = int(duplicate_of)
            else:
                seen.setdefault(fingerprint, []).append(int(column))
    if not np.any(keep):
        raise ValueError("No nonconstant distinct training predictors remain.")
    safe_scale = np.where(constant, 1., scale)
    transformed = []
    for values in (train, *others):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != train.shape[1] or not np.isfinite(values).all():
            raise ValueError("All predictor arrays must be finite with matching columns.")
        transformed.append(np.asarray((values[:, keep] - mean[keep]) / safe_scale[keep], dtype=np.float64))
    return transformed, mean, scale, keep, np.flatnonzero(constant), duplicates


def make_synthetic(seed: int = 0, rho: float = 0., n: int = 500, p: int = 1000,
                   sparsity: int = 20, block_size: int = 20, noise_std: float = 1.,
                   n_test: int = 1000, layout: str = "dispersed") -> dict:
    """Paired block correlation, nested sparsity, and dispersed/clustered support.

    For fixed seed and dimensions, X and X_test do not depend on sparsity or
    layout. The standard Gaussian noise realization does not depend on rho,
    sparsity or layout. Only the explicit noise_std multiplies it. Supports are
    nested in increasing sparsity separately for each layout. The ordered sign
    sequence is shared across layouts; w_true is in training-standardized
    coordinates. Clustered support always has exactly five signals per block.
    """
    if not 0 <= rho < 1 or not np.isfinite(rho):
        raise ValueError("rho must lie in [0, 1).")
    if min(n, p, sparsity, block_size, n_test) < 1 or n < 2 or p % block_size:
        raise ValueError("Positive dimensions and p divisible by block_size are required.")
    if not np.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be positive and finite.")
    if layout not in {"dispersed", "clustered"}:
        raise ValueError("layout must be dispersed or clustered.")
    n_blocks = p // block_size
    signals_per_active_block = 1 if layout == "dispersed" else 5
    if block_size < signals_per_active_block or sparsity % signals_per_active_block:
        raise ValueError("Clustered support requires a multiple of five signals and block_size >= 5.")
    if sparsity > n_blocks * signals_per_active_block:
        raise ValueError("Requested sparsity exceeds the chosen layout's block capacity.")
    rngs = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(8)]
    common = rngs[0].normal(size=(n, n_blocks))
    independent = rngs[1].normal(size=(n, p))
    common_test = rngs[2].normal(size=(n_test, n_blocks))
    independent_test = rngs[3].normal(size=(n_test, p))
    raw_X = np.sqrt(rho) * np.repeat(common, block_size, axis=1) + np.sqrt(1-rho) * independent
    raw_test = np.sqrt(rho) * np.repeat(common_test, block_size, axis=1) + np.sqrt(1-rho) * independent_test
    arrays, mean, scale, retained, _, _ = _fit_standardizer(raw_X, raw_test)
    if not np.all(retained):
        raise ValueError("Unexpected constant synthetic predictor.")
    X, X_test = arrays
    block_order = rngs[4].permutation(n_blocks)
    within_block_orders = np.stack([rngs[4].permutation(block_size) for _ in range(n_blocks)])
    if layout == "dispersed":
        ordered_support = block_order * block_size + within_block_orders[block_order, 0]
    else:
        ordered_support = (block_order[:, None] * block_size
                           + within_block_orders[block_order, :5]).reshape(-1)
    support = ordered_support[:sparsity]
    ordered_signs = rngs[5].choice(np.array([-1., 1.]), size=p)
    w_true = np.zeros(p, dtype=np.float64)
    w_true[support] = ordered_signs[:sparsity]
    standard_noise = rngs[6].normal(size=n)
    standard_noise_test = rngs[7].normal(size=n_test)
    noise = noise_std * standard_noise
    noise_test = noise_std * standard_noise_test
    signal, signal_test = X @ w_true, X_test @ w_true
    y_uncentered = signal + noise
    y_mean = float(y_uncentered.mean())
    y = np.asarray(y_uncentered - y_mean, dtype=np.float64)
    y_test = np.asarray(signal_test + noise_test - y_mean, dtype=np.float64)
    signal_variance = float(np.var(signal, ddof=0))
    realized_noise_variance = float(np.var(noise, ddof=0))
    metadata = {
        "kind": "synthetic_v2", "protocol": "fixed_sigma_nested_support_v2",
        "seed": int(seed), "rho": float(rho), "n": int(n), "p": int(p),
        "n_test": int(n_test), "sparsity": int(sparsity), "s": int(sparsity),
        "layout": layout, "block_size": int(block_size), "n_blocks": n_blocks,
        "signals_per_active_block": signals_per_active_block,
        "support": support.tolist(), "support_order": support.tolist(),
        "signs": ordered_signs[:sparsity].tolist(), "true_coefficients": w_true[support].tolist(),
        "noise_std": float(noise_std), "noise_scaling": "fixed sigma; never rescaled by signal",
        "signal_var_train": signal_variance, "noise_var_realized_train": realized_noise_variance,
        "snr_population": signal_variance / noise_std**2,
        "snr_realized_train": signal_variance / realized_noise_variance,
        "train_y_mean": y_mean, "train_mean": mean.tolist(), "train_scale": scale.tolist(),
        "coefficient_coordinates": "training-centered and population-standardized X; ddof=0",
        "rng": "NumPy PCG64; SeedSequence(seed).spawn(8)",
        "pairing": "Same seed: shared design components/noise across rho, s and layout; nested support within layout; shared ordered signs across layouts",
        "hashes": {"X": array_sha256(X), "y": array_sha256(y), "w_true": array_sha256(w_true),
                   "X_test": array_sha256(X_test), "y_test": array_sha256(y_test),
                   "support_order": array_sha256(support), "ordered_signs": array_sha256(ordered_signs),
                   "noise": array_sha256(noise), "noise_test": array_sha256(noise_test),
                   "latent_train": array_sha256(common), "independent_train": array_sha256(independent)},
    }
    return {"X": X, "y": y, "w_true": w_true, "X_test": X_test, "y_test": y_test,
            "metadata": metadata}


def make_uji(seed: int = 0, csv_path: str | Path | None = None) -> dict:
    """Longitude regression using only WAP features from UCI trainingData.csv.

    Fit exact constant/duplicate-column removal on the 60% training subset;
    apply its retained columns and scale to validation/test. The official 100
    non-detection code remains unchanged before standardization. No location or
    identity metadata enters X. The external official validation set is unused.
    """
    import pandas as pd

    path = Path(csv_path) if csv_path is not None else BENCHMARK_ROOT / "ujiindoorloc/trainingData.csv"
    csv_hash = _verify_source_file(path)
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    wap_names = [f"WAP{i:03d}" for i in range(1, 521)]
    if "LONGITUDE" not in frame or not set(wap_names).issubset(frame.columns):
        raise ValueError("Expected official UJI schema with LONGITUDE and WAP001..WAP520.")
    raw_X = frame[wap_names].to_numpy(dtype=np.float64)
    target = frame["LONGITUDE"].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all():
        raise ValueError("Longitude must be finite; no silent label imputation.")
    permutation = np.random.default_rng(seed).permutation(len(frame))
    n_train, n_val = int(.6 * len(frame)), int(.2 * len(frame))
    splits = {"train": permutation[:n_train], "val": permutation[n_train:n_train+n_val],
              "test": permutation[n_train+n_val:]}
    arrays, mean, scale, keep, constant_columns, duplicates = _fit_standardizer(
        *(raw_X[index] for index in splits.values()), deduplicate=True)
    y_mean = float(target[splits["train"]].mean())
    metadata = {
        "kind": "uji_v2", "name": "UJIIndoorLoc longitude", "seed": int(seed),
        "source_page": UJI_PAGE, "doi": "10.24432/C5MS59", "license": "CC BY 4.0",
        "csv_sha256": csv_hash, "n_rows": len(frame), "pool": "official trainingData.csv only",
        "target": "LONGITUDE", "target_units": "original UCI coordinate units, without target rescaling",
        "raw_features": 520, "used_features": int(keep.sum()),
        "feature_names": [name for name, retained in zip(wap_names, keep) if retained],
        "removed_constant_features": [wap_names[j] for j in constant_columns],
        "removed_duplicate_features": {wap_names[j]: wap_names[k] for j, k in duplicates.items()},
        "excluded_non_wap_fields": [name for name in frame.columns if name not in wap_names],
        "nondetection_code": "official RSSI value 100 retained before standardization",
        "feature_selection_fit": "training subset only; exact constants then exact duplicate columns",
        "split_rule": "default_rng(seed).permutation; floor(0.6*n), floor(0.2*n), remainder",
        "split_sizes": {key: len(index) for key, index in splits.items()},
        "split_indices": {key: index.tolist() for key, index in splits.items()},
        "split_sha256": {key: array_sha256(index) for key, index in splits.items()},
        "train_mean": mean.tolist(), "train_scale": scale.tolist(), "train_y_mean": y_mean,
        "evaluation_scope": "IID random held-out longitude prediction within official training pool; not cross-device, temporal or external-test performance",
        "has_true_support": False,
    }
    result = {"metadata": metadata}
    for (label, index), X in zip(splits.items(), arrays):
        result[f"X_{label}"] = X
        result[f"y_{label}"] = np.asarray(target[index] - y_mean, dtype=np.float64)
    return result


def make_leukemia(path: str | Path | None = None) -> dict:
    """All 72 LIBSVM observations as a fixed optimization-only Lasso problem."""
    from sklearn.datasets import load_svmlight_file

    location = Path(path) if path is not None else BENCHMARK_ROOT / "leukemia/leukemia.libsvm"
    content_hash = _verify_source_file(location)
    sparse_X, labels = load_svmlight_file(str(location), n_features=7129, zero_based=False)
    raw_X = sparse_X.toarray().astype(np.float64, copy=False)
    if raw_X.shape != (72, 7129) or len(np.unique(labels)) != 2 or not np.isfinite(labels).all():
        raise ValueError("Expected the merged 72 x 7129, two-label LIBSVM leukemia benchmark.")
    arrays, mean, scale, keep, constants, _ = _fit_standardizer(raw_X, drop_constants=False)
    y_mean = float(labels.mean())
    X, y = arrays[0], np.asarray(labels - y_mean, dtype=np.float64)
    values, counts = np.unique(labels, return_counts=True)
    metadata = {
        "kind": "leukemia_v2", "name": "Leukemia fixed least-squares optimization benchmark",
        "source_page": LEUKEMIA_PAGE, "file_sha256": content_hash,
        "license": "Not specified on checked LIBSVM dataset page; local download only, no redistribution",
        "n": 72, "p": 7129, "raw_features": 7129, "used_features": int(keep.sum()),
        "labels": {str(float(value)): int(count) for value, count in zip(values, counts)},
        "target": "numeric binary class label, centered over all 72 observations",
        "train_y_mean": y_mean, "train_mean": mean.tolist(), "train_scale": scale.tolist(),
        "constant_columns_retained_as_zero": constants.tolist(),
        "split": "none; merge official 38/34 files in that order, use all 72 for one fixed optimization task",
        "evaluation_scope": "solver objective/gap/timing/sparsity only; no regression application, held-out prediction, causal or true-support claim",
        "has_true_support": False,
        "hashes": {"X": array_sha256(X), "y": array_sha256(y)},
    }
    return {"X": X, "y": y, "metadata": metadata}
