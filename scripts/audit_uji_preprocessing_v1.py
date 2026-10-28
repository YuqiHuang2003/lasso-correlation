#!/usr/bin/env python3
"""Matrix-only audit of UJI row sampling, preprocessing and three WAP columns.

Reads WAP predictors only: no original response, generated response, coefficients,
reference fit or optimization is used. Existing study artifacts are read-only.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
from datetime import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lasso_study.data import array_sha256, file_sha256
from lasso_study.data_v2 import _fit_standardizer
from lasso_study.workshop_data import _standardize, _streams

REPRESENTATIVES = ("WAP220", "WAP424", "WAP466")
WAPS = tuple(f"WAP{j:03d}" for j in range(1, 521))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def bits(column):
    return np.ascontiguousarray(column, dtype=np.float64).tobytes()


def deduplication(X, names):
    """Strict float64 bytes, including signed zeros; ascending-column anchors."""
    positive, signed = {}, {}
    same_of, signed_of = {}, {}
    same_root, signed_root = {}, {}
    for j, name in enumerate(names):
        encoded = bits(X[:, j]); digest = hashlib.sha256(encoded).digest()
        candidates = positive.get(digest, [])
        anchor = next((k for k in candidates if bits(X[:, k]) == encoded), None)
        if anchor is None:
            positive.setdefault(digest, []).append(j)
            same_root[name] = name
        else:
            same_of[name] = names[anchor]; same_root[name] = names[anchor]
        match = next(((k, sign) for k, sign in signed.get(digest, [])
                      if bits(sign * X[:, k]) == encoded), None)
        if match is None:
            signed.setdefault(digest, []).append((j, 1))
            neg_digest = hashlib.sha256(bits(-X[:, j])).digest()
            signed.setdefault(neg_digest, []).append((j, -1))
            signed_root[name] = name
        else:
            anchor, sign = match
            signed_of[name] = {"anchor": names[anchor], "sign": sign}
            signed_root[name] = names[anchor]
    pairs = {}
    for left, right in combinations(REPRESENTATIVES, 2):
        if left not in same_root or right not in same_root:
            pairs[left + "/" + right] = {"both_present": False}
        else:
            pairs[left + "/" + right] = {"both_present": True,
                "same_deduplication_class": same_root[left] == same_root[right],
                "same_or_opposite_deduplication_class": signed_root[left] == signed_root[right]}
    return {"n_columns": len(names), "same_only_removed": len(same_of),
            "same_or_opposite_removed": len(signed_of), "same_only_mapping": same_of,
            "same_or_opposite_mapping": signed_of, "representative_pair_coverage": pairs,
            "scope": "Strict stored float64 bytes; constants included only if this representation retains them."}


def rank_diagnostic(X, names):
    if X.shape[1] == 0:
        return {"columns": list(names), "n_columns": 0, "singular_values": [],
                "rank_default": 0, "rank_relative_1e12": 0,
                "default_cutoff": 0., "relative_1e12_cutoff": 0.}
    singular = np.linalg.svd(X, compute_uv=False)
    cutoff = np.finfo(float).eps * max(X.shape) * singular[0]
    relative = 1e-12 * singular[0]
    return {"columns": list(names), "n_columns": len(names), "singular_values": singular.tolist(),
            "default_cutoff": float(cutoff), "rank_default": int(np.sum(singular > cutoff)),
            "relative_1e12_cutoff": float(relative), "rank_relative_1e12": int(np.sum(singular > relative)),
            "scope": "Only these representative columns; numerical ranks are not exact mathematical ranks."}


def pair_diagnostic(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xn, yn = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    scale = float(x @ y / (xn * xn)) if xn else None
    residual = y - scale*x if scale is not None else None
    xc, yc = x - x.mean(), y - y.mean()
    xcn, ycn = float(np.linalg.norm(xc)), float(np.linalg.norm(yc))
    slope = float(xc @ yc / (xcn*xcn)) if xcn else None
    intercept = float(y.mean() - slope*x.mean()) if slope is not None else None
    affine = y - (slope*x + intercept) if slope is not None else None
    return {"bitwise_same": bits(x) == bits(y), "bitwise_opposite": bits(x) == bits(-y),
            "x_constant": bool(np.ptp(x) == 0), "y_constant": bool(np.ptp(y) == 0),
            "proportional_scale_y_from_x": scale,
            "proportional_relative_l2_residual": float(np.linalg.norm(residual)/yn) if residual is not None and yn else None,
            "proportional_max_absolute_residual": float(np.max(np.abs(residual))) if residual is not None else None,
            "affine_slope_y_from_x": slope, "affine_intercept_y_from_x": intercept,
            "affine_relative_l2_residual_centered_y": float(np.linalg.norm(affine)/ycn) if affine is not None and ycn else None,
            "affine_max_absolute_residual": float(np.max(np.abs(affine))) if affine is not None else None,
            "centered_cosine": float(xc @ yc/(xcn*ycn)) if xcn and ycn else None}


def audit_view(identifier, phase, raw, raw_names, actual_X, actual_names,
               statuses, original_rows, pool_rows=None, saved_metadata=None):
    representative = []; pair_rows = []
    for name in REPRESENTATIVES:
        values = raw[:, raw_names.index(name)]
        unique, counts = np.unique(values, return_counts=True)
        mode_index = int(np.argmax(counts))
        representative.append({"feature": name, "retained_in_actual_matrix": name in actual_names,
             "pipeline_status": statuses[name], "n_rows": len(values),
             "n_detected_raw_ne_100": int(np.sum(values != 100)),
             "n_nondetected_raw_eq_100": int(np.sum(values == 100)),
             "n_unique": len(unique), "mode_raw": float(unique[mode_index]),
             "mode_count": int(counts[mode_index]), "nonmode_count": int(len(values)-counts[mode_index]),
             "value_counts": [[float(v), int(n)] for v, n in zip(unique, counts)]})
    raw_indices = [raw_names.index(name) for name in REPRESENTATIVES]
    present = [name for name in REPRESENTATIVES if name in actual_names]
    available_indices = [actual_names.index(name) for name in present]
    pair_results = {}
    for left, right in combinations(REPRESENTATIVES, 2):
        result = {"raw_before_selection": pair_diagnostic(raw[:, raw_names.index(left)], raw[:, raw_names.index(right)]),
                  "both_retained": left in actual_names and right in actual_names}
        if result["both_retained"]:
            result["actual_standardized_matrix"] = pair_diagnostic(
                actual_X[:, actual_names.index(left)], actual_X[:, actual_names.index(right)])
        pair_results[left + "/" + right] = result
        for representation in ("raw_before_selection", "actual_standardized_matrix"):
            if representation in result:
                pair_rows.append(dict(view=identifier, phase=phase, left=left, right=right,
                                      representation=representation, both_retained=result["both_retained"],
                                      **result[representation]))
    nonconstant = [j for j in range(raw.shape[1]) if np.ptp(raw[:, j]) != 0]
    dedup = {"raw_all_available": deduplication(raw, raw_names),
             "raw_nonconstant": deduplication(raw[:, nonconstant], [raw_names[j] for j in nonconstant]),
             "actual_standardized": deduplication(actual_X, actual_names)}
    result = {"view": identifier, "phase": phase, "n_rows": len(raw),
              "raw_p": raw.shape[1], "actual_p": actual_X.shape[1],
              "raw_row_sha256": array_sha256(np.asarray(original_rows, dtype=np.int64)),
              "actual_X_sha256": array_sha256(actual_X), "representatives": representative,
              "pairs": pair_results, "deduplication": dedup,
              "rank_raw_representatives": rank_diagnostic(raw[:, raw_indices], REPRESENTATIVES),
              "rank_actual_retained_representatives": rank_diagnostic(actual_X[:, available_indices], present),
              "saved_metadata_verified": saved_metadata}
    columns_csv = [dict(view=identifier, phase=phase, **{k:v for k,v in row.items() if k != "value_counts"})
                   for row in representative]
    duplicate_csv = [dict(view=identifier, phase=phase, representation=label,
                          n_rows=len(raw), n_columns=row["n_columns"],
                          same_only_removed=row["same_only_removed"],
                          same_or_opposite_removed=row["same_or_opposite_removed"])
                     for label, row in dedup.items()]
    rows_csv = [dict(view=identifier, audit_row_0based=i, original_csv_row_0based=int(row),
                     source_pool_row_0based=int(pool_rows[i]) if pool_rows is not None else "")
                for i,row in enumerate(original_rows)]
    return result, columns_csv, pair_rows, duplicate_csv, rows_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/reference_audit_v1/uji_preprocessing")
    args = parser.parse_args(); started = perf_counter()
    output = args.output.resolve()
    require(output == (ROOT / "results/reference_audit_v1/uji_preprocessing").resolve(), "Use the dedicated audit output")
    source_path = ROOT / "data/benchmarks/ujiindoorloc/source.json"
    manifest_path = ROOT / "docs/data_manifest_v2_benchmarks_ujiindoorloc.json"
    csv_path = source_path.parent / "trainingData.csv"
    source = json.loads(source_path.read_text()); manifest = json.loads(manifest_path.read_text())
    require(file_sha256(source_path) == file_sha256(manifest_path), "source.json differs from saved source manifest")
    checked_files = {str(source_path.relative_to(ROOT)): file_sha256(source_path),
                     str(manifest_path.relative_to(ROOT)): file_sha256(manifest_path)}
    for name in ("trainingData.csv", "ujiindoorloc.zip"):
        path = source_path.parent / name; actual = file_sha256(path)
        require(actual == source["files"][name]["sha256"], f"Source hash mismatch: {name}")
        checked_files[str(path.relative_to(ROOT))] = actual
    config_path = ROOT / "configs/workshop_real.json"
    config = json.loads(config_path.read_text())
    checked_files[str(config_path.relative_to(ROOT))] = file_sha256(config_path)
    result = {"kind": "matrix_only_preprocessing_audit_v1", "new_optimization_fits": 0,
              "original_response_read_or_used": False, "generated_responses": 0,
              "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "input_sha256": checked_files, "audit_script_sha256": file_sha256(Path(__file__)),
              "representatives": list(REPRESENTATIVES), "nondetection_code": 100,
              "old_row_seeds": list(range(300,310)), "new_row_probe_seeds": list(range(600,610)),
              "row_probe_scope": "Matrix row-sampling probes, not new response generation or reference fitting",
              "numerical_definitions": {"rank_default": "eps*max(n,p)*largest_singular_value",
                  "rank_secondary": "1e-12*largest_singular_value",
                  "affine_residual": "||y-(a*x+b)|| / ||y-mean(y)||; undefined for constant y",
                  "proportional_residual": "||y-a*x|| / ||y||",
                  "bitwise_dedup": "Stored float64 bytes, signed zeros retained; ascending-column keep-first"},
              "views": []}
    collected = [[], [], [], []]
    def add(*args, **kwargs):
        view, *tables = audit_view(*args, **kwargs)
        result["views"].append(view)
        for target, rows in zip(collected, tables): target.extend(rows)
        print(json.dumps({"matrix_audit_view": view["view"], "n":view["n_rows"], "p":view["actual_p"]}), flush=True)
    with threadpool_limits(limits=1):
        # Only WAP columns are parsed; the source response and metadata fields
        # are not loaded into a DataFrame or consulted for row/column selection.
        raw = pd.read_csv(csv_path, usecols=list(WAPS))[list(WAPS)].to_numpy(dtype=np.float64)
        require(raw.shape == (19937,520) and np.isfinite(raw).all(), "Unexpected official WAP matrix")
        full_arrays, _, _, full_keep, _, _ = _fit_standardizer(raw, deduplicate=False)
        full_names = [name for name,keep in zip(WAPS,full_keep) if keep]
        add("official_training_full", "official_full", raw, list(WAPS), full_arrays[0], full_names,
            {name: "retained" if name in full_names else "constant" for name in REPRESENTATIVES},
            np.arange(len(raw)))
        permutation = np.random.default_rng(0).permutation(len(raw))
        train_rows = permutation[:int(.6*len(raw))]
        source_raw = raw[train_rows]
        arrays, _, _, keep, constants, duplicates = _fit_standardizer(source_raw, deduplicate=True)
        pool = arrays[0]
        pool_names = [name for name, retained in zip(WAPS, keep) if retained]
        expected_pool_meta_path = ROOT / "results/workshop/real/datasets/uji_seed300_dispersed_strong.json"
        expected_pool_meta = json.loads(expected_pool_meta_path.read_text())
        checked_files[str(expected_pool_meta_path.relative_to(ROOT))] = file_sha256(expected_pool_meta_path)
        require(array_sha256(pool) == expected_pool_meta["hashes"]["source_pool_X"], "Reconstructed source pool hash differs")
        require(array_sha256(train_rows) == expected_pool_meta["source_training_split_sha256"], "Source training split hash differs")
        pool_status = {}
        for name in REPRESENTATIVES:
            original = WAPS.index(name)
            pool_status[name] = ("retained" if keep[original] else "constant" if original in constants
                                 else "exact_duplicate_of_"+WAPS[duplicates[original]])
        add("source_seed0_training_pool", "source_pool", source_raw, list(WAPS), pool, pool_names,
            pool_status, train_rows, np.arange(len(train_rows)), saved_metadata={"source_pool_hash_matches":True})
        result["source_pool"] = {"n":len(pool), "p":pool.shape[1], "feature_names":pool_names,
             "X_sha256":array_sha256(pool), "train_rows_sha256":array_sha256(train_rows),
             "source_removed_constants":[WAPS[j] for j in constants],
             "source_removed_duplicates":{WAPS[j]:WAPS[k] for j,k in duplicates.items()}}
        for seed in list(range(300,310)) + list(range(600,610)):
            phase = "old_row_sample" if seed < 600 else "new_row_probe"
            chosen = _streams(seed)[2].choice(len(pool), size=config["real_n"], replace=False)
            sampled = pool[chosen]
            constant_mask = np.ptp(sampled, axis=0) == 0
            nonconstant = np.flatnonzero(~constant_mask)
            capped = min(len(nonconstant), config["p"])
            retained = capped // config["block_size"] * config["block_size"]
            columns = nonconstant[:retained]
            X, _, _ = _standardize(sampled[:,columns])
            actual_names = [pool_names[j] for j in columns]
            statuses = {}
            for name in REPRESENTATIVES:
                if name not in pool_names: statuses[name] = "already_removed_from_source_pool_"+pool_status[name]
                else:
                    j = pool_names.index(name)
                    statuses[name] = ("constant_in_row_sample" if constant_mask[j] else "retained" if j in columns
                                      else "p_cap" if j in nonconstant[capped:] else "incomplete_block_tail")
            saved = None
            if seed < 600:
                metadata_path = ROOT / "results/workshop/real/datasets" / f"uji_seed{seed}_dispersed_strong.json"
                metadata = json.loads(metadata_path.read_text())
                checked_files[str(metadata_path.relative_to(ROOT))] = file_sha256(metadata_path)
                require(chosen.tolist() == metadata["source_pool_row_indices"], f"Old row index mismatch: {seed}")
                require(columns.tolist() == metadata["retained_source_columns_0based"], f"Old column selection mismatch: {seed}")
                require(array_sha256(X) == metadata["hashes"]["X"], f"Old standardized X hash mismatch: {seed}")
                saved = {"row_indices_match":True, "retained_columns_match":True, "X_hash_matches":True,
                         "metadata_path":str(metadata_path.relative_to(ROOT))}
            add(f"seed{seed}_rows1000", phase, raw[train_rows[chosen]], list(WAPS), X, actual_names,
                statuses, train_rows[chosen], chosen, saved_metadata=saved)
            result["views"][-1]["column_pipeline"] = {"source_p":pool.shape[1],
                "after_constant_removal":len(nonconstant), "after_p_cap":capped, "actual_p":retained,
                "removed_constant_source_indices":np.flatnonzero(constant_mask).tolist(),
                "removed_p_cap_source_indices":nonconstant[capped:].tolist(),
                "removed_incomplete_tail_source_indices":nonconstant[retained:capped].tolist(),
                "retained_source_indices":columns.tolist(), "actual_feature_names":actual_names}
    require(all(file_sha256(ROOT/name)==expected for name,expected in checked_files.items()), "An input changed during audit")
    result["boundaries"] = ["All diagnostics refer to the downloaded source and these sampled/preprocessed matrices only.",
         "Row probe seeds are not new fitted response replications; original y is never read or used.",
         "Raw pre-selection counts may describe a column removed from actual X; removed columns are never forced into X.",
         "Strict byte equality, approximate proportionality, affine relations and general linear dependence differ.",
         "Numerical ranks/near-dependence do not establish exact mathematical rank or multiple Lasso minimizers.",
         "No finding identifies an error in the original UJI data paper or another study's full design."]
    result["wall_sec"] = perf_counter()-started
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"audit.json",result)
    for name,rows in zip(("representative_columns.csv","representative_pairs.csv","deduplication_counts.csv","row_indices.csv"),collected):
        write_csv(output/name,rows)
    print(json.dumps({"passed":True,"views":len(result["views"]),"new_optimization_fits":0,
                      "wall_sec":result["wall_sec"],"output":str(output)},indent=2))


if __name__ == "__main__":
    main()
