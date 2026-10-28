"""External-design controls for the frozen reference-audit v1 protocol.

Original responses are discarded at source preparation. All nonconstant
predictors survive the new sampling step, including exact/near duplicates.
Near-proportional grouping is an optional X-only post-processing operation;
the generator does not call it or construct a feature Gram matrix.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import array_sha256, file_sha256, make_real
from .data_v2 import make_leukemia


ROOT = Path(__file__).resolve().parents[2]
_GROUP_CACHE: OrderedDict = OrderedDict()
_GROUP_CACHE_MAX = 8


def _integer(name, value, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _matrix(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1 or not np.isfinite(X).all():
        raise ValueError("X must be a finite two-dimensional matrix with at least two rows.")
    return X


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def exact_duplicate_columns(X):
    """Record bitwise equal and sign-reversed float64 columns in O(n*p).

    Signed groups give relations to one anchor, not a quadratic list of pairs.
    Signed zero is significant because the rule is stored-byte equality.
    """
    X = _matrix(X)
    buckets = {}
    keys = []
    for j in range(X.shape[1]):
        key = np.ascontiguousarray(X[:, j]).tobytes()
        keys.append(key)
        buckets.setdefault(key, []).append(j)
    equal = sorted((indices for indices in buckets.values() if len(indices) > 1), key=lambda item: item[0])
    visited, signed = set(), []
    for j, key in enumerate(keys):
        if key in visited:
            continue
        negative = np.ascontiguousarray(-X[:, j]).tobytes()
        same, opposite = buckets[key], buckets.get(negative, []) if negative != key else []
        same_members = set(same)
        members = sorted(same + opposite)
        visited.add(key); visited.add(negative)
        if len(members) > 1:
            signed.append({"anchor_column_0based": j, "columns_0based": members,
                           "member_signs_from_anchor": [1 if member in same_members else -1 for member in members]})
    return {"equal_groups": equal, "signed_groups": signed,
            "equal_nontrivial_group_count": len(equal), "signed_nontrivial_group_count": len(signed),
            "columns_in_equal_groups": sum(map(len, equal)),
            "negative_members_relative_to_anchor": sum(sign < 0 for group in signed for sign in group["member_signs_from_anchor"]),
            "rule": "Stored float64 column bytes equal anchor bytes or bytes of its unary negation; no columns deleted"}


def load_audit_source(source, *, source_seed=0, path=None):
    """Load the fixed source X pool, without returning the original response.

    ``news`` reuses the frozen make_real(seed=0) training-X transform. ``gene``
    reuses all 72 merged LIBSVM rows. The legacy loaders parse their response
    fields, but no response value or response statistic enters the returned
    pool, sampling, column selection, or simulated-response construction.
    """
    if source not in {"news", "gene"}:
        raise ValueError("source must be news or gene.")
    if _integer("source_seed", source_seed) != 0:
        raise ValueError("The audit protocol fixes source_seed=0.")
    location = Path(path) if path is not None else ROOT / (
        "data/raw/OnlineNewsPopularity.csv" if source == "news" else "data/benchmarks/leukemia/leukemia.libsvm")
    if not location.is_file():
        raise FileNotFoundError(f"Missing frozen local source: {location}")
    actual_hash = file_sha256(location)
    record_path = location.parent / "source.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    expected = record.get("csv_sha256") if source == "news" else record.get("files", {}).get(location.name, {}).get("sha256")
    if expected is not None and actual_hash != expected:
        raise ValueError("Source input differs from its local source.json snapshot.")
    if source == "news":
        loaded = make_real(seed=0, csv_path=location)
        old = loaded["metadata"]
        X = np.asfortranarray(loaded["X_train"], dtype=np.float64)
        names = list(old["feature_names"])
        with location.open(newline="", encoding="utf-8") as stream:
            header = [name.strip() for name in next(csv.reader(stream))]
        original_names = [name for name in header if name not in {"url", "timedelta", "shares"}]
        original_columns = [original_names.index(name) for name in names]
        original_rows = list(old["split_indices"]["train"])
        prep = "Frozen make_real(seed=0), train-only exact-constant removal and ddof=0 standardization; source training pool only"
        license_text = record.get("license", old.get("license", "CC BY 4.0"))
        split_hash = old["split_sha256"]["train"]
    else:
        loaded = make_leukemia(path=location)
        old = loaded["metadata"]
        X = np.asfortranarray(loaded["X"], dtype=np.float64)
        names = [f"gene_{j+1:04d}" for j in range(X.shape[1])]
        original_rows, original_columns = list(range(len(X))), list(range(X.shape[1]))
        prep = "Frozen make_leukemia merged 38/34 order, all 72 rows, center and ddof=0 scale; old zero constants will be removed in final-sample preparation"
        license_text = record.get("license", old.get("license", "Not specified on checked LIBSVM page; local only, no redistribution"))
        split_hash = array_sha256(np.arange(len(X), dtype=np.int64))
    _matrix(X)
    metadata = {"source": source, "source_seed": 0, "source_original_response_used": False,
                "source_preparation": prep, "source_pool_shape": list(X.shape), "feature_names": names,
                "source_original_row_indices_0based": original_rows,
                "source_original_predictor_indices_0based": original_columns,
                "input_path": str(location.resolve().relative_to(ROOT)) if location.resolve().is_relative_to(ROOT) else location.name,
                "input_sha256": actual_hash,
                "source_record_sha256": file_sha256(record_path) if record_path.is_file() else None,
                "source_record": record, "source_split_sha256": split_hash,
                "source_pool_X_sha256": array_sha256(X), "license": license_text,
                "source_page": record.get("source_page", old.get("source_page")),
                "source_scope": "News has a fixed source training pool with new row subsamples; gene is one fixed X with new support/noise draws; sources are not difficulty matched"}
    metadata["source_preparation_sha256"] = _json_hash(metadata)
    return {"X_pool": X, "metadata": metadata}


def make_audit_design(config, seed, source, amplitude, c, *, source_data=None):
    """Generate one paired sparse response on a retained external feature design."""
    if source not in {"news", "gene"} or amplitude not in {"strong", "mixed_weak"}:
        raise ValueError("Use source news/gene and amplitude strong/mixed_weak.")
    seed = _integer("seed", seed)
    sparsity = _integer("sparsity", config.get("sparsity", 5), 1)
    sigma = float(config.get("noise_std", 1.))
    if sparsity != 5 or sigma != 1. or config.get("weak_every", 4) != 4 or config.get("weak_amplitude", .25) != .25:
        raise ValueError("The frozen audit protocol fixes sparsity=5, noise_std=1, weak_every=4, weak_amplitude=0.25.")
    if isinstance(c, (bool, np.bool_)) or not np.isfinite(c) or float(c) not in {.5, 1.}:
        raise ValueError("The frozen audit protocol fixes c in {0.5, 1.0}.")
    data = source_data if source_data is not None else load_audit_source(source)
    pool = _matrix(data["X_pool"])
    old = data["metadata"]
    if old.get("source", source) != source or old.get("source_seed", 0) != 0:
        raise ValueError("The source pool must match the requested source and fixed seed 0.")
    rngs = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)]
    if source == "news":
        n = _integer("news_n", config.get("news_n", 1000), 2)
        if n > len(pool):
            raise ValueError("Not enough news source rows for sampling without replacement.")
        rows = rngs[0].choice(len(pool), size=n, replace=False)
    else:
        n = _integer("gene_n", config.get("gene_n", len(pool)), 2)
        if n != len(pool):
            raise ValueError("The gene arm retains the complete source pool in its original order.")
        rows = np.arange(n, dtype=np.int64)
    sampled = pool[rows]
    keep = np.ptp(sampled, axis=0) != 0
    columns, constants = np.flatnonzero(keep), np.flatnonzero(~keep)
    if len(columns) < sparsity:
        raise ValueError("Too few nonconstant features for the registered sparse truth.")
    retained = sampled[:, columns]
    mean, scale = retained.mean(axis=0), retained.std(axis=0, ddof=0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Final-sample feature scale is invalid.")
    X = np.asfortranarray((retained-mean)/scale, dtype=np.float64)
    p = X.shape[1]
    support = rngs[1].permutation(p)[:sparsity]
    signs = rngs[2].choice(np.array([-1., 1.]), size=sparsity)
    magnitudes = np.ones(sparsity)
    if amplitude == "mixed_weak":
        magnitudes[3::4] = .25
    truth = np.zeros(p, dtype=np.float64)
    truth[support] = signs*magnitudes
    standard_noise = rngs[3].normal(size=n)
    noise = sigma*standard_noise
    signal = X @ truth
    raw_y = signal+noise
    y_mean = float(raw_y.mean())
    y = np.asarray(raw_y-y_mean, dtype=np.float64)
    names = old.get("feature_names", [f"source_column_{j}" for j in range(pool.shape[1])])
    source_rows = old.get("source_original_row_indices_0based", list(range(len(pool))))
    source_columns = old.get("source_original_predictor_indices_0based", list(range(pool.shape[1])))
    if len(names) != pool.shape[1] or len(source_rows) != len(pool) or len(source_columns) != pool.shape[1]:
        raise ValueError("Source row/column provenance dimensions differ from X_pool.")
    hashes = {"X": array_sha256(X), "y": array_sha256(y), "w_true": array_sha256(truth),
              "source_pool_X": array_sha256(pool), "sampled_raw_X": array_sha256(sampled),
              "sampled_rows": array_sha256(rows), "retained_columns": array_sha256(columns),
              "support_order": array_sha256(support), "ordered_signs": array_sha256(signs),
              "noise": array_sha256(noise), "standard_noise": array_sha256(standard_noise),
              "magnitudes": array_sha256(magnitudes)}
    signal_var, noise_var = float(np.var(signal)), float(np.var(noise))
    metadata = {"kind": "reference_audit_external_design_v1", "protocol": "reference_audit_protocol_v1",
                "generator_source_sha256": file_sha256(Path(__file__)), "source": source, "seed": seed,
                "source_seed": 0, "amplitude": amplitude, "c": float(c), "lambda_factor": float(c),
                "n": n, "p": p, "sparsity": sparsity, "noise_std": sigma,
                "lambda": float(c*sigma*np.sqrt(2*np.log(p)/n)),
                "lambda_rule": "c*sigma*sqrt(2*log(final_p)/final_n); no signal or lambda_max scaling",
                "support": support.tolist(), "support_order": support.tolist(), "support_sorted": sorted(support.tolist()),
                "signs": signs.tolist(), "magnitudes": magnitudes.tolist(), "true_coefficients": truth[support].tolist(),
                "support_rule": "First five entries of a uniform feature permutation; no artificial layout blocks",
                "weak_positions_in_support_1based": (np.flatnonzero(magnitudes < 1)+1).tolist(),
                "train_mean": mean.tolist(), "train_scale": scale.tolist(), "train_y_mean": y_mean,
                "signal_var_train": signal_var, "noise_var_realized_train": noise_var,
                "snr_population": signal_var/sigma**2, "snr_realized_train": signal_var/noise_var,
                "source_original_response_used": False, "source_pool_shape": list(pool.shape),
                "source_pool_row_indices": rows.tolist(), "source_original_row_indices_0based": [int(source_rows[j]) for j in rows],
                "retained_source_columns_0based": columns.tolist(), "removed_constant_source_columns_0based": constants.tolist(),
                "retained_original_predictor_indices_0based": [int(source_columns[j]) for j in columns],
                "feature_names": [names[j] for j in columns],
                "column_pipeline": "Select rows; remove exact constants; retain every other column in source order; center and ddof=0 scale on this sample",
                "column_counts": {"source": pool.shape[1], "removed_constants": len(constants), "retained": p},
                "source_preparation_sha256": old.get("source_preparation_sha256", _json_hash(old)),
                "source_input_sha256": old.get("input_sha256"), "source_record_sha256": old.get("source_record_sha256"),
                "source_split_sha256": old.get("source_split_sha256"), "license": old.get("license"),
                "source_page": old.get("source_page"),
                "coefficient_coordinates": "Final-sample centered and population-standardized retained X, ddof=0",
                "rng": "NumPy PCG64 via SeedSequence(seed).spawn(4)",
                "rng_stream_roles": {"0": "news row sampling; gene retains rows", "1": "support permutation",
                                     "2": "ordered signs", "3": "fixed-sigma Gaussian noise"},
                "pairing": "Same source/seed shares X, rows, support, signs and noise across c/amplitude; c changes only lambda",
                "statistical_unit": "News seed changes rows/support/noise; gene seed changes support/noise conditional on one fixed X; sources not difficulty matched",
                "exact_duplicates": exact_duplicate_columns(X),
                "near_proportional_groups": "Deferred to X-only post-processing; not run during source/response preparation",
                "hashes": hashes}
    return {"X": X, "y": y, "w_true": truth, "metadata": metadata}


def near_proportional_groups_chunked(X, tolerances=(1e-12, 1e-10, 1e-14), block_size=256):
    """Scan all unordered pairs in blocks, then greedily verify anchor members.

    Full cosine candidate screening is conservative at floating-point precision.
    Every admitted member independently passes ||x_j-a*x_anchor||/||x_j||<=tol;
    negative a is allowed. No transitive chaining, response, truth or fit enters
    this operation. Working arrays are at most B-by-p booleans and B-by-B
    floats, never a dense p-by-p floating-point Gram matrix for p>B.
    """
    started = perf_counter()
    X = _matrix(X)
    block_size = _integer("block_size", block_size, 1)
    tolerances = tuple(float(tol) for tol in tolerances)
    if not tolerances or len(set(tolerances)) != len(tolerances) or any(not np.isfinite(tol) or not 0 < tol < 1 for tol in tolerances):
        raise ValueError("tolerances must be distinct finite numbers strictly between 0 and 1.")
    X_hash = array_sha256(X)
    cache_key = (X_hash, tolerances, block_size)
    if cache_key in _GROUP_CACHE:
        cached = copy.deepcopy(_GROUP_CACHE.pop(cache_key))
        _GROUP_CACHE[cache_key] = copy.deepcopy(cached)
        cached.update(cache_hit=True, scan_elapsed_sec=0., elapsed_sec=perf_counter()-started)
        cached["diagnostics"].update(cache_hit=True, wall_sec=cached["elapsed_sec"], scan_wall_sec=0.)
        return cached
    n, p = X.shape
    B = min(block_size, p)
    norms = np.empty(p)
    for j in range(0, p, B):
        norms[j:j+B] = np.linalg.norm(X[:, j:j+B], axis=0)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Grouping requires nonzero finite feature norms; remove constants first.")
    eps = np.finfo(np.float64).eps
    gamma_n = n*eps/max(1-n*eps, eps)
    rounding_guard = 16*gamma_n+8*eps
    cutoff = max(0., float(np.sqrt(max(0., 1-max(tolerances)**2)))-rounding_guard)
    labels = np.full((len(tolerances), p), -1, dtype=np.int64)
    groups = [[] for _ in tolerances]
    candidate_count = pair_count = cosine_entries = direct_checks = 0
    scan_started = perf_counter()
    for start in range(0, p, B):
        stop = min(start+B, p)
        left = X[:, start:stop]/norms[start:stop]
        candidates = np.zeros((stop-start, p), dtype=bool)
        left_indices = np.arange(start, stop)
        for other in range(start, p, B):
            end = min(other+B, p)
            right = X[:, other:end]/norms[other:end]
            cosine = left.T @ right
            np.abs(cosine, out=cosine)
            eligible = cosine >= cutoff
            upper = np.arange(other, end)[None, :] > left_indices[:, None]
            eligible &= upper
            candidates[:, other:end] = eligible
            candidate_count += int(np.count_nonzero(eligible))
            pair_count += int(np.count_nonzero(upper))
            cosine_entries += cosine.size
            # Release before computing the next RHS, so two cosine blocks
            # cannot coexist during Python assignment evaluation.
            del cosine, eligible, upper, right
        for local, anchor in enumerate(range(start, stop)):
            active = [k for k in range(len(tolerances)) if labels[k, anchor] < 0]
            if not active:
                continue
            current = {k: {"anchor_column_0based": anchor, "columns_0based": [anchor],
                           "member_scales_from_anchor": [1.], "relative_l2_residuals_to_anchor": [0.]} for k in active}
            for k in active:
                labels[k, anchor] = len(groups[k])
            denominator = float(X[:, anchor] @ X[:, anchor])
            for member in np.flatnonzero(candidates[local]):
                needed = [k for k in active if labels[k, member] < 0]
                if not needed:
                    continue
                scale = float((X[:, anchor] @ X[:, member])/denominator)
                residual = float(np.linalg.norm(X[:, member]-scale*X[:, anchor])/norms[member])
                direct_checks += 1
                for k in needed:
                    if residual <= tolerances[k]:
                        current[k]["columns_0based"].append(int(member))
                        current[k]["member_scales_from_anchor"].append(scale)
                        current[k]["relative_l2_residuals_to_anchor"].append(residual)
                        labels[k, member] = len(groups[k])
            for k in active:
                groups[k].append(current[k])
        del left, candidates
    if pair_count != p*(p-1)//2 or np.any(labels < 0):
        raise AssertionError("Incomplete all-pair screening or group partition.")
    scan_sec = perf_counter()-scan_started
    grouped = {}
    for tolerance, members, assignment in zip(tolerances, groups, labels):
        sizes = [len(group["columns_0based"]) for group in members]
        grouped[str(tolerance)] = {"tolerance": tolerance, "groups": members, "column_to_group": assignment.tolist(),
                                   "group_count": len(members), "nontrivial_group_count": sum(size > 1 for size in sizes),
                                   "maximum_group_size": max(sizes), "columns_in_nontrivial_groups": sum(size for size in sizes if size > 1)}
    result = {"X_sha256": X_hash, "n": n, "p": p, "tolerances": list(tolerances), "block_size": B,
              "groups_by_tolerance": {str(tolerance): members for tolerance, members in zip(tolerances, groups)},
              "groupings": grouped, "all_unordered_pairs_screened": True,
              "unique_pairs_scanned": pair_count, "cosine_entries_computed": cosine_entries,
              "cosine_candidate_pairs": candidate_count, "direct_anchor_residual_checks": direct_checks,
              "cosine_candidate_cutoff": cutoff, "cosine_rounding_guard": rounding_guard,
              "working_array_bytes_upper_bound": int(8*(2*n*B+B*B+3*p+len(tolerances)*p)+B*p+2*B*B),
              "memory_bound_scope": "Conservative bound for block cosine/normalized input blocks, candidate bool mask and label/norm vectors; excludes caller input, result Python objects and BLAS-internal scratch",
              "rule": "Full unordered-pair blocked absolute cosine screening; ascending-column greedy anchor groups; every admitted member directly passes no-intercept relative residual; negative scale allowed; no transitive chaining",
              "floating_point_boundary": "Candidate guard accommodates dot/normalization roundoff; final groups are numerical residual groups, not proofs of exact proportionality or mathematical solution non-uniqueness",
              "cache_hit": False, "scan_elapsed_sec": scan_sec, "original_scan_elapsed_sec": scan_sec,
              "cache_scope": "Process-local LRU of eight X-hash/tolerance/block-size results; call once per distinct X in post-processing",
              "elapsed_sec": perf_counter()-started}
    result["diagnostics"] = {"wall_sec": result["elapsed_sec"], "scan_wall_sec": scan_sec,
                             "block_size": B, "peak_gram_bytes_upper_bound": int(8*B*B),
                             "working_array_bytes_upper_bound": result["working_array_bytes_upper_bound"],
                             "candidate_pairs_checked": direct_checks, "cosine_candidate_pairs": candidate_count,
                             "unique_pairs_scanned": pair_count, "all_unordered_pairs_screened": True,
                             "cache_hit": False, "cosine_candidate_cutoff": cutoff, "cosine_rounding_guard": rounding_guard}
    _GROUP_CACHE[cache_key] = copy.deepcopy(result)
    while len(_GROUP_CACHE) > _GROUP_CACHE_MAX:
        _GROUP_CACHE.popitem(last=False)
    return result
