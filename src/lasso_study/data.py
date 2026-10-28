"""Paired synthetic problems and leakage-free public regression data splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_URL = "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip"
DATA_PAGE = "https://archive.ics.uci.edu/dataset/332/online+news+popularity"


def array_sha256(array: np.ndarray) -> str:
    """Hash shape, dtype and values so a problem or split can be identified."""
    a = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(a.shape).encode("ascii"))
    digest.update(a.dtype.str.encode("ascii"))
    digest.update(a.tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _standardize(train: np.ndarray, *others: np.ndarray):
    mean = train.mean(axis=0)
    scale = train.std(axis=0, ddof=0)
    keep = scale > 0.0
    if not np.any(keep):
        raise ValueError("Training data has no nonconstant features.")
    arrays = [np.asarray((a[:, keep] - mean[keep]) / scale[keep], dtype=np.float64)
              for a in (train, *others)]
    return arrays, mean, scale, keep


def make_synthetic(
    seed: int = 0,
    rho: float = 0.0,
    n: int = 2000,
    p: int = 1000,
    sparsity: int = 20,
    block_size: int = 20,
    snr: float = 10.0,
    n_test: int = 1000,
) -> dict:
    """Generate equicorrelated blocks with active coefficients in distinct blocks.

    The true coefficient vector is defined in the *training-standardized*
    coordinates. The same seed pairs latent factors, independent components,
    support, signs and Gaussian noise across values of rho. SNR is the ratio of
    training signal variance to the population variance of the Gaussian noise;
    realized finite-sample SNR is reported separately.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1).")
    if min(n, p, sparsity, block_size, n_test) < 1 or n < 2 or snr <= 0:
        raise ValueError("Dimensions, sparsity, block size and SNR must be positive.")
    if p % block_size:
        raise ValueError("p must be a multiple of block_size.")
    n_groups = p // block_size
    if sparsity > n_groups:
        raise ValueError("Distinct-block support requires sparsity <= p/block_size.")
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(8)]
    common = streams[0].normal(size=(n, n_groups))
    independent = streams[1].normal(size=(n, p))
    common_test = streams[2].normal(size=(n_test, n_groups))
    independent_test = streams[3].normal(size=(n_test, p))
    raw_train = np.sqrt(rho) * np.repeat(common, block_size, axis=1)
    raw_train += np.sqrt(1.0 - rho) * independent
    raw_test = np.sqrt(rho) * np.repeat(common_test, block_size, axis=1)
    raw_test += np.sqrt(1.0 - rho) * independent_test
    arrays, mean, scale, keep = _standardize(raw_train, raw_test)
    if not np.all(keep):
        raise ValueError("Unexpected constant synthetic feature.")
    X, X_test = arrays
    groups = streams[4].choice(n_groups, size=sparsity, replace=False)
    support = groups * block_size + streams[4].integers(block_size, size=sparsity)
    w_true = np.zeros(p, dtype=np.float64)
    w_true[support] = streams[5].choice(np.array([-1.0, 1.0]), size=sparsity)
    signal = X @ w_true
    signal_test = X_test @ w_true
    noise_std = float(np.sqrt(np.var(signal, ddof=0) / snr))
    noise = noise_std * streams[6].normal(size=n)
    noise_test = noise_std * streams[7].normal(size=n_test)
    y_raw = signal + noise
    y_mean = float(y_raw.mean())
    y = np.asarray(y_raw - y_mean, dtype=np.float64)
    y_test = np.asarray(signal_test + noise_test - y_mean, dtype=np.float64)
    metadata = {
        "kind": "synthetic",
        "seed": int(seed), "rng": "NumPy PCG64 via SeedSequence.spawn(8)",
        "rho": float(rho), "n": int(n), "p": int(p), "n_test": int(n_test),
        "sparsity": int(sparsity), "block_size": int(block_size),
        "support": support.tolist(), "true_coefficients": w_true[support].tolist(),
        "coefficient_coordinates": "training-standardized X (population std, ddof=0)",
        "snr_definition": "var(training signal) / Gaussian population noise variance",
        "snr_target": float(snr), "noise_std": noise_std,
        "snr_realized_train": float(np.var(signal) / np.var(noise)),
        "train_mean": mean.tolist(), "train_scale": scale.tolist(),
        "train_y_mean": y_mean,
        "hashes": {"X": array_sha256(X), "y": array_sha256(y),
                   "w_true": array_sha256(w_true),
                   "X_test": array_sha256(X_test), "y_test": array_sha256(y_test)},
    }
    return {"X": X, "y": y, "w_true": w_true,
            "X_test": X_test, "y_test": y_test, "metadata": metadata}


def make_real(seed: int = 0, csv_path: str | Path | None = None) -> dict:
    """Split Online News Popularity 60/20/20, fitting all transforms on train.

    Random splits assess an IID held-out task, not future article forecasting.
    The URL identifier and relative publication-time field are excluded. The
    target is log1p(shares). Its training mean is subtracted from all splits,
    equivalent to an unpenalized intercept when X is centered.
    """
    import pandas as pd

    path = Path(csv_path) if csv_path is not None else PROJECT_ROOT / "data/raw/OnlineNewsPopularity.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run scripts/download_data.py first; missing {path}")
    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    if not {"url", "timedelta", "shares"}.issubset(frame.columns):
        raise ValueError("Unexpected UCI CSV schema: required columns missing.")
    names = [c for c in frame.columns if c not in {"url", "timedelta", "shares"}]
    if len(names) != 58:
        raise ValueError(f"Expected 58 numeric predictors, found {len(names)}.")
    X_raw = frame[names].to_numpy(dtype=np.float64)
    shares = frame["shares"].to_numpy(dtype=np.float64)
    if not np.isfinite(X_raw).all() or not np.isfinite(shares).all() or np.any(shares < 0):
        raise ValueError("Nonfinite predictors or invalid share counts; no silent imputation.")
    target = np.log1p(shares)
    n_rows = len(frame)
    permutation = np.random.default_rng(seed).permutation(n_rows)
    n_train, n_val = int(0.6 * n_rows), int(0.2 * n_rows)
    train = permutation[:n_train]
    val = permutation[n_train:n_train + n_val]
    test = permutation[n_train + n_val:]
    arrays, mean, scale, keep = _standardize(X_raw[train], X_raw[val], X_raw[test])
    y_mean = float(target[train].mean())
    splits = {"train": train, "val": val, "test": test}
    metadata = {
        "kind": "real", "name": "UCI Online News Popularity", "seed": int(seed),
        "source_url": DATA_URL, "source_page": DATA_PAGE,
        "license": "CC BY 4.0", "doi": "10.24432/C5NS3V",
        "csv_sha256": file_sha256(path), "n_rows": n_rows,
        "raw_features": len(names), "used_features": int(keep.sum()),
        "feature_names": [name for name, retained in zip(names, keep) if retained],
        "removed_constant_features": [name for name, retained in zip(names, keep) if not retained],
        "excluded_fields": ["url", "timedelta"], "target": "log1p(shares)",
        "split_rule": "default_rng(seed).permutation; floor(0.6*n)/floor(0.2*n)/remainder",
        "split_rng": "NumPy PCG64", "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_indices": {k: v.tolist() for k, v in splits.items()},
        "split_sha256": {k: array_sha256(v) for k, v in splits.items()},
        "train_mean": mean.tolist(), "train_scale": scale.tolist(),
        "train_y_mean": y_mean,
        "evaluation_scope": "IID random held-out prediction; not temporal forecasting or causal inference",
    }
    result = {"metadata": metadata}
    for (label, index), matrix in zip(splits.items(), arrays):
        result[f"X_{label}"] = matrix
        result[f"y_{label}"] = np.asarray(target[index] - y_mean, dtype=np.float64)
    return result

