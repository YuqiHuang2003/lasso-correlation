#!/usr/bin/env python3
"""Validate v2 results and rebuild tables/figures without rerunning a solver.

Only synthetic/uji/leukemia production runs enter these summaries. Smoke, pilot
and reproduction runs are never pooled. skglm has endpoint observations, not a
comparable time-to-accuracy history, so it is excluded from threshold analyses.
"""
from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/comp6704-v2-matplotlib")

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import product
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("synthetic", "uji", "leukemia")
BASIC = ("ista", "fista", "cd")
METHODS = BASIC + ("skglm",)
LABELS = {"ista": "ISTA", "fista": "FISTA", "cd": "Cyclic CD", "skglm": "skglm AndersonCD"}
COLORS = {"ista": "#386cb0", "fista": "#d95f02", "cd": "#13856f", "skglm": "#8056a5"}
MARKERS = {"ista": "o", "fista": "s", "cd": "^", "skglm": "D"}
STYLES = {"ista": ":", "fista": "--", "cd": "-", "skglm": "-."}
SYNTHETIC_KEY = ("seed", "rho", "sparsity", "layout", "lambda_factor", "method")
CELL_KEY = ("rho", "sparsity", "layout", "lambda_factor", "method")
DESIGN_KEY = ("seed", "rho", "sparsity", "layout")
RECOVERY = ("precision", "recall", "f1", "fdr", "tp", "fp", "fn", "nnz",
            "exact_support", "exact_sign", "active_sign_accuracy", "group_precision",
            "group_recall", "group_f1", "selected_groups", "true_groups", "relative_coef_error",
            "test_mse", "signal_prediction_mse")
CERTIFICATE = ("primal", "dual", "gap", "raw_gap", "normalized_gap", "dual_feasibility", "kkt_inf")
DIAGNOSTICS = ("signal_var_train", "noise_std", "noise_var_realized_train", "snr_population",
               "snr_realized_train", "active_gram_min_eigenvalue", "active_gram_condition",
               "sample_irrepresentable_score", "raw_population_signal_variance_formula",
               "lipschitz_setup_sec", "preprocessing_sec", "L")
FIGURE_STATUS_NOTE = ""


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_records(folder: Path, child: str) -> list[dict]:
    return [read(path) for path in sorted((folder / child).glob("*.json"))]


def finite(value) -> bool:
    return isinstance(value, (int, float, np.number, bool)) and math.isfinite(float(value))


def stats(values) -> dict:
    values = np.asarray([float(x) for x in values if finite(x)], dtype=float)
    if not len(values):
        return dict(n=0, mean=None, median=None, q25=None, q75=None, min=None, max=None)
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75)),
            "min": float(values.min()), "max": float(values.max())}


def final(run: dict) -> dict:
    if isinstance(run.get("final"), dict):
        return run["final"]
    history = run.get("history", [])
    return history[-1] if history else {}


def value(run: dict, metric: str):
    return final(run).get(metric, run.get(metric))


def converged(run: dict, tolerance: float) -> bool:
    gap = final(run).get("normalized_gap")
    return run.get("status") == "converged" and finite(gap) and gap <= tolerance


def comparable_history(run: dict) -> bool:
    # Check the method as well as the flag, to avoid treating library endpoints
    # as a trajectory even if a future writer mistakenly sets the flag True.
    return run.get("method") in BASIC and run.get("threshold_curve_comparable") is True


def threshold_hit(run: dict, tolerance: float):
    if not comparable_history(run):
        return None
    for key, observation in run.get("threshold_hits", {}).items():
        if math.isclose(float(key), tolerance, rel_tol=1e-12, abs_tol=0.):
            return observation
    return next((h for h in run.get("history", [])
                 if finite(h.get("normalized_gap")) and h["normalized_gap"] <= tolerance), None)


def normalized(item):
    return round(float(item), 12) if isinstance(item, (float, np.floating)) else item


def key_for(row: dict, fields) -> tuple:
    return tuple(normalized(row.get(k)) for k in fields)


def group_by(rows: list[dict], fields) -> dict:
    result = defaultdict(list)
    for row in rows:
        result[key_for(row, fields)].append(row)
    return result


def select(rows: list[dict], **criteria) -> list[dict]:
    return [r for r in rows if all(normalized(r.get(k)) == normalized(v) for k, v in criteria.items())]


def expected_and_integrity(runs, datasets, failures, snapshots, config, allow_partial, config_hash):
    """Require the actual Cartesian grid, not merely the correct row count."""
    synthetic = config["synthetic"]
    expected_syn = set(product(synthetic["seeds"], synthetic["rhos"], synthetic["sparsities"],
                               synthetic["layouts"], synthetic["lambda_factors"], config["methods"]))
    expected_leukemia = set(product(config["leukemia"]["lambda_ratios"],
                                   config["leukemia"]["repeats"], config["methods"]))
    expected_designs = set(product(synthetic["seeds"], synthetic["rhos"],
                                   synthetic["sparsities"], synthetic["layouts"]))
    issues, warnings = [], []
    observed = {"synthetic": [key_for(r, SYNTHETIC_KEY) for r in runs["synthetic"]],
                "leukemia": [key_for(r, ("lambda_ratio", "repeat", "method")) for r in runs["leukemia"]]}
    expected_uji = set()
    uji_metadata = {d.get("seed"): d for d in datasets["uji"]}
    for seed in config["uji"]["seeds"]:
        metadata = uji_metadata.get(seed)
        if metadata is None:
            issues.append(f"missing UJI selection metadata for seed {seed}")
            continue
        indices = metadata.get("selected_indices", [])
        if len(indices) != 3 or len(set(indices)) != 3 or metadata.get("best_index") not in indices:
            issues.append(f"invalid UJI selected_indices/best_index for seed {seed}")
        expected_uji.update(product([seed], indices, config["methods"]))
        for r in select(runs["uji"], seed=seed):
            if bool(r.get("selected_by_validation")) != (r.get("validation_index") == metadata.get("best_index")):
                issues.append(f"UJI best-lambda flag mismatch: {r.get('run_id')}")
    observed["uji"] = [key_for(r, ("seed", "validation_index", "method")) for r in runs["uji"]]
    expected = {"synthetic": expected_syn, "uji": expected_uji, "leukemia": expected_leukemia}
    expected_counts = {"synthetic": len(expected_syn),
                       "uji": len(config["uji"]["seeds"]) * 3 * len(config["methods"]),
                       "leukemia": len(expected_leukemia)}
    counts = {}
    for dataset in STAGES:
        snapshot = snapshots[dataset]
        if snapshot["config"] is not None and snapshot["config"] != config:
            issues.append(f"{dataset}: frozen configuration differs from the analysis configuration")
        env = snapshot["environment"]
        if runs[dataset] and env is None:
            issues.append(f"{dataset}: missing environment/source provenance record")
        elif env is not None and env.get("v2_config_sha256") != config_hash:
            issues.append(f"{dataset}: original configuration hash differs between environment records")
        points = observed[dataset]
        duplicates = [list(k) for k, n in Counter(points).items() if n > 1]
        missing = expected[dataset] - set(points)
        unexpected = set(points) - expected[dataset]
        counts[dataset] = {"observed_runs": len(points), "expected_runs": expected_counts[dataset],
                           "missing_cells": len(missing), "unexpected_cells": len(unexpected),
                           "duplicate_cells": len(duplicates),
                           "exception_records": len(failures[dataset])}
        if len(points) != expected_counts[dataset]:
            issues.append(f"{dataset}: {len(points)} runs, expected {expected_counts[dataset]}")
        if missing:
            issues.append(f"{dataset}: missing {len(missing)} Cartesian grid cells")
        if unexpected:
            issues.append(f"{dataset}: {len(unexpected)} unexpected grid cells")
        if duplicates:
            issues.append(f"{dataset}: {len(duplicates)} duplicate grid cells")
        if len({r.get("run_id") for r in runs[dataset]}) != len(points):
            issues.append(f"{dataset}: duplicate or missing run IDs")
        for r in runs[dataset]:
            if r.get("dataset") != dataset:
                issues.append(f"dataset label mismatch: {r.get('run_id')}")
            if r.get("config_sha256") != config_hash:
                issues.append(f"configuration hash mismatch: {r.get('run_id')}")
            if not r.get("history") or not all(finite(final(r).get(k)) for k in CERTIFICATE):
                issues.append(f"missing/nonfinite final certificate: {r.get('run_id')}")
            if r.get("status") == "converged" and not converged(r, config["solver"]["tol"]):
                issues.append(f"converged status contradicts final certificate: {r.get('run_id')}")
            if r.get("method") == "skglm" and r.get("threshold_curve_comparable") is not False:
                issues.append(f"skglm endpoint history incorrectly marked comparable: {r.get('run_id')}")
        signatures = {r.get("numerical_signature") for r in runs[dataset]}
        if None in signatures or len(signatures) > 1:
            issues.append(f"{dataset}: missing or multiple numerical source signatures")
        if signatures and env is not None and signatures != {env.get("v2_numerical_signature")}:
            issues.append(f"{dataset}: run and environment numerical signatures differ")
        # All methods must see the same problem, including lambda, not simply
        # matching the same nominal factors in the grid.
        problem_fields = {"synthetic": SYNTHETIC_KEY[:-1], "uji": ("seed", "validation_index"),
                          "leukemia": ("lambda_ratio", "repeat")}[dataset]
        for _, group in group_by(runs[dataset], problem_fields).items():
            if len({(r.get("dataset_hash"), r.get("lambda"), r.get("n"), r.get("p")) for r in group}) != 1:
                issues.append(f"{dataset}: different data/penalties among paired methods")
        known_ids = {r.get("run_id") for r in runs[dataset]}
        current_exceptions = [r for r in failures[dataset] if r.get("run_id") not in known_ids]
        if current_exceptions:
            warnings.append(f"{dataset}: {len(current_exceptions)} unresolved exception records; cells remain missing")
    all_signatures = {r.get("numerical_signature") for stage in STAGES for r in runs[stage]}
    if len(all_signatures) > 1:
        issues.append("production datasets contain different numerical source signatures")
    design_points = [key_for(d, DESIGN_KEY) for d in datasets["synthetic"]]
    if len(design_points) != len(set(design_points)) or set(design_points) != expected_designs:
        issues.append(f"synthetic design metadata incomplete/duplicated: {len(set(design_points))}/{len(expected_designs)}")
    design_map = {key_for(d, DESIGN_KEY): d for d in datasets["synthetic"]}
    for r in runs["synthetic"]:
        d = design_map.get(key_for(r, DESIGN_KEY))
        if d is None:
            issues.append(f"missing design metadata for {r.get('run_id')}")
        elif r.get("dataset_hash") != d.get("hashes", {}).get("X", "") + d.get("hashes", {}).get("y", ""):
            issues.append(f"design/run hash mismatch: {r.get('run_id')}")
    for _, group in group_by(datasets["synthetic"], ("seed", "rho")).items():
        if len({d.get("hashes", {}).get("X") for d in group}) > 1:
            issues.append("synthetic X is not paired across sparsity/layout")
    for _, group in group_by(datasets["synthetic"], ("seed",)).items():
        for name in ("noise", "noise_test", "ordered_signs"):
            if len({d.get("hashes", {}).get(name) for d in group}) > 1:
                issues.append(f"synthetic {name} is not paired across design factors")
    issues = list(dict.fromkeys(issues))
    integrity = {"complete_grid": not issues, "allow_partial": bool(allow_partial), "issues": issues,
                 "warnings": warnings, "counts": counts, "config_sha256": config_hash,
                 "numerical_signatures": sorted(s for s in all_signatures if s is not None),
                 "synthetic_design_count": len(set(design_points)), "expected_synthetic_designs": len(expected_designs)}
    if issues and not allow_partial:
        raise SystemExit("Incomplete/inconsistent v2 results: " + "; ".join(issues))
    return integrity, design_map


def run_table(runs, design_map, config):
    rows = []
    base_fields = ("run_id", "dataset", "seed", "repeat", "rho", "sparsity", "layout", "lambda_factor",
                   "lambda", "lambda_ratio", "lambda_max", "method", "n", "p", "status", "iterations",
                   "iteration_unit", "elapsed_sec", "cold_start_sec", "diagnostics_sec", "lipschitz_setup_sec",
                   "objective_difference_to_reference", "validation_index", "selected_by_validation", "test_mse",
                   "signal_prediction_mse", "validation_mse", "attempt_count", "certificate_passed", "budget_exceeded",
                   "evaluation_sec_outside_solver_timer", "threshold_curve_comparable", "dataset_hash",
                   "config_sha256", "numerical_signature")
    for run in runs:
        row = {k: run.get(k) for k in base_fields}
        row["converged_common_criterion"] = converged(run, config["solver"]["tol"])
        row.update({"final_" + k: v for k, v in final(run).items()})
        row["diagnostics_fraction"] = run.get("diagnostics_sec", 0.) / max(run.get("elapsed_sec", 0.), 1e-300)
        for k, v in run.get("reference", {}).items():
            if finite(v):
                row["reference_" + k] = v
        for t in config["thresholds"]:
            hit = threshold_hit(run, t)
            row[f"observed_time_to_{t:g}"] = None if hit is None else hit.get("time")
            row[f"iteration_to_{t:g}"] = None if hit is None else hit.get("iteration")
        d = design_map.get(key_for(run, DESIGN_KEY), {}) if run.get("dataset") == "synthetic" else {}
        row.update({"design_" + k: d.get(k) for k in DIAGNOSTICS})
        rows.append(row)
    return rows


def aggregate_cells(runs, config):
    rows = []
    for keys, group in sorted(group_by(runs, CELL_KEY).items()):
        good = [r for r in group if converged(r, config["solver"]["tol"])]
        row = dict(zip(CELL_KEY, keys))
        row.update(n_runs=len(group), n_converged=len(good), n_unconverged=len(group)-len(good),
                   expected_seeds=len(config["synthetic"]["seeds"]),
                   status_counts=dict(Counter(r["status"] for r in group)))
        for metric in ("elapsed_sec", "cold_start_sec", "diagnostics_sec", "iterations"):
            row[metric] = stats(r.get(metric) for r in good)
        for metric in RECOVERY:
            row[metric] = stats(value(r, metric) for r in group)
        row["recovery_converged_only"] = {k: stats(value(r, k) for r in good) for k in RECOVERY}
        for metric in ("exact_support", "exact_sign"):
            available = [r for r in group if value(r, metric) is not None]
            completed = [r for r in good if value(r, metric) is not None]
            row[metric + "_successes"] = sum(bool(value(r, metric)) for r in available)
            row[metric + "_rate"] = (row[metric + "_successes"] / len(available) if available else None)
            row[metric + "_converged_rate"] = (sum(bool(value(r, metric)) for r in completed)/len(completed)
                                                     if completed else None)
        row["lambda"] = stats(r.get("lambda") for r in group)
        row["lambda_ratio"] = stats(r.get("lambda_ratio") for r in group)
        rows.append(row)
    return rows


def paired_contrasts(runs, config, factor, low, high, fixed):
    output = []
    for keys, group in sorted(group_by(runs, fixed).items()):
        low_map = {r["seed"]: r for r in select(group, **{factor: low})}
        high_map = {r["seed"]: r for r in select(group, **{factor: high})}
        pairs = [(low_map[s], high_map[s]) for s in sorted(set(low_map) & set(high_map))]
        good = [(a, b) for a, b in pairs if converged(a, config["solver"]["tol"])
                and converged(b, config["solver"]["tol"])]
        row = dict(zip(fixed, keys))
        row.update(factor=factor, low=low, high=high, n_pairs=len(pairs), n_converged_pairs=len(good))
        for metric in ("elapsed_sec", "cold_start_sec"):
            row[metric + "_high_over_low"] = stats(b[metric]/a[metric] for a, b in good if a[metric] > 0)
        row["recovery_high_minus_low"] = {metric: stats(value(b, metric)-value(a, metric) for a, b in pairs
                    if finite(value(a, metric)) and finite(value(b, metric))) for metric in RECOVERY}
        output.append(row)
    return output


def layout_rho_interactions(runs, config):
    """Difference in the rho effect between clustered and dispersed support."""
    low, high = min(config["synthetic"]["rhos"]), max(config["synthetic"]["rhos"])
    result = []
    fields = ("sparsity", "lambda_factor", "method")
    for keys, group in sorted(group_by(runs, fields).items()):
        cases = []
        for _, by_seed in group_by(group, ("seed",)).items():
            four = {(r["layout"], r["rho"]): r for r in by_seed}
            needed = [(layout, rho) for layout in ("dispersed", "clustered") for rho in (low, high)]
            if all(k in four for k in needed):
                cases.append([four[k] for k in needed])
        good = [case for case in cases if all(converged(r, config["solver"]["tol"]) for r in case)]
        row = dict(zip(fields, keys))
        row.update(low_rho=low, high_rho=high, n_paired_quadruples=len(cases), n_converged_quadruples=len(good))
        row["recovery_difference_in_differences"] = {
            metric: stats((value(d, metric)-value(c, metric))-(value(b, metric)-value(a, metric))
                          for a, b, c, d in cases if all(finite(value(r, metric)) for r in (a, b, c, d)))
            for metric in ("f1", "recall", "group_f1", "relative_coef_error", "test_mse", "exact_support")}
        row["time_rho_ratio_clustered_over_dispersed"] = stats(
            (d["elapsed_sec"]/c["elapsed_sec"])/(b["elapsed_sec"]/a["elapsed_sec"])
            for a, b, c, d in good if min(a["elapsed_sec"], b["elapsed_sec"], c["elapsed_sec"]) > 0)
        result.append(row)
    return result


def quality(runs, tolerance):
    if not runs:
        return {"n_runs": 0, "status_counts": {}}
    observations = [h for r in runs for h in r.get("history", [])]
    normalized_deltas = [abs(r["objective_difference_to_reference"])/max(1., abs(r["reference"]["primal"]))
                         for r in runs if finite(r.get("objective_difference_to_reference")) and r.get("reference")]
    def extreme(values, mode):
        values = [float(x) for x in values if finite(x)]
        return (max(values) if mode == "max" else min(values)) if values else None
    return {"n_runs": len(runs), "n_converged": sum(converged(r, tolerance) for r in runs),
            "status_counts": dict(Counter(r["status"] for r in runs)),
            "max_final_normalized_gap": extreme((final(r).get("normalized_gap") for r in runs), "max"),
            "max_converged_final_normalized_gap": extreme((final(r).get("normalized_gap") for r in runs if converged(r, tolerance)), "max"),
            "max_reference_normalized_gap": extreme((r.get("reference", {}).get("normalized_gap") for r in runs), "max"),
            "max_normalized_objective_difference_to_reference": extreme(normalized_deltas, "max"),
            "max_absolute_objective_difference_to_reference": extreme((abs(r["objective_difference_to_reference"])
                          for r in runs if finite(r.get("objective_difference_to_reference"))), "max"),
            "max_dual_feasibility_all_checkpoints": extreme((h.get("dual_feasibility") for h in observations), "max"),
            "min_raw_gap_all_checkpoints": extreme((h.get("raw_gap") for h in observations), "min"),
            "max_final_kkt_inf": extreme((final(r).get("kkt_inf") for r in runs), "max"),
            "diagnostic_fraction": stats(r.get("diagnostics_sec", 0.)/max(r.get("elapsed_sec", 0.), 1e-300) for r in runs),
            "objective_increases_at_recorded_checkpoints": stats(r.get("objective_increases") for r in runs),
            "runs_with_observed_objective_increases": sum((r.get("objective_increases") or 0) > 0 for r in runs)}


def structural_summary(records):
    output = []
    for keys, group in sorted(group_by(records, ("rho", "sparsity", "layout")).items()):
        row = dict(zip(("rho", "sparsity", "layout"), keys))
        row["n_designs"] = len(group)
        row.update({metric: stats(d.get(metric) for d in group) for metric in DIAGNOSTICS})
        row["irrepresentable_score_below_one_count"] = sum(finite(d.get("sample_irrepresentable_score"))
                                                          and d["sample_irrepresentable_score"] < 1 for d in group)
        output.append(row)
    by_design = [{k: d.get(k) for k in DESIGN_KEY + DIAGNOSTICS +
                 ("group_signal_counts", "group_sign_sums", "full_hessian_singular", "full_hessian_rank_upper_bound")}
                 for d in records]
    return {"by_cell": output, "by_design": by_design,
            "interpretation": "True-support Gram/irrepresentable values are diagnostics, not finite-sample recovery guarantees; full Hessians are singular for p>n-1."}


def threshold_tables(runs, config):
    detailed, aggregated = [], []
    eligible = [r for r in runs if comparable_history(r)]
    for run in eligible:
        for tolerance in config["thresholds"]:
            hit = threshold_hit(run, tolerance)
            row = {k: run.get(k) for k in SYNTHETIC_KEY + ("run_id", "status")}
            row.update(threshold=tolerance, reached=hit is not None, observation_type="first observed basic-solver checkpoint")
            if hit is not None:
                row.update({k: v for k, v in hit.items() if finite(v)})
            detailed.append(row)
    for keys, group in sorted(group_by(eligible, CELL_KEY).items()):
        for tolerance in config["thresholds"]:
            hits = [threshold_hit(r, tolerance) for r in group]
            found = [h for h in hits if h is not None]
            row = dict(zip(CELL_KEY, keys))
            row.update(threshold=tolerance, n_runs=len(group), n_hits=len(found),
                       n_final_unconverged=sum(not converged(r, config["solver"]["tol"]) for r in group))
            row.update({metric: stats(h.get(metric) for h in found)
                        for metric in ("time", "iteration") + RECOVERY if metric not in ("test_mse", "signal_prediction_mse")})
            aggregated.append(row)
    return detailed, aggregated


def sensitivity_summary(runs, config):
    detailed = []
    numerical = [float(t) for t in config["support_sensitivity"] if float(t) <= 1e-4]
    ranges, numeric_ranges = [], []
    for r in runs:
        data = r.get("support_sensitivity", {})
        values = []
        small = []
        for threshold, metrics in data.items():
            t = float(threshold)
            row = {k: r.get(k) for k in CELL_KEY}
            row.update(seed=r.get("seed"), threshold=t, final_status=r.get("status"), **metrics)
            detailed.append(row)
            if finite(metrics.get("f1")):
                values.append(metrics["f1"])
                if t in numerical:
                    small.append(metrics["f1"])
        if values:
            ranges.append(max(values)-min(values))
        if len(small) == len(numerical):
            numeric_ranges.append(max(small)-min(small))
    groups = []
    for keys, rows in sorted(group_by(detailed, ("threshold", "layout", "method")).items()):
        entry = dict(zip(("threshold", "layout", "method"), keys))
        entry.update(n_runs=len(rows), threshold_role="numerical sensitivity" if keys[0] <= 1e-4 else "prespecified hard-threshold supplement")
        entry["metrics_pooled_over_rho_s_lambda"] = {m: stats(r.get(m) for r in rows)
            for m in ("f1", "recall", "precision", "fdr", "group_f1", "exact_support", "exact_sign", "nnz")}
        groups.append(entry)
    return {"primary_threshold": config["support_threshold"], "numerical_thresholds": numerical,
            "by_threshold_layout_method": groups, "within_run_f1_range_all_thresholds": stats(ranges),
            "within_run_f1_range_numerical_thresholds": stats(numeric_ranges),
            "runs_with_numerical_f1_change": sum(x > 1e-12 for x in numeric_ranges),
            "runs_with_any_f1_change": sum(x > 1e-12 for x in ranges),
            "interpretation": "0.01/0.05 change the statistical estimator by fixed post-thresholding; no threshold was selected using test performance."}


def agreement_summary(runs, config):
    rows = []
    for keys, group in sorted(group_by(runs, SYNTHETIC_KEY[:-1]).items()):
        good = [r for r in group if converged(r, config["solver"]["tol"])]
        if len(good) < 2:
            continue
        row = dict(zip(SYNTHETIC_KEY[:-1], keys))
        row.update(n_converged_methods=len(good))
        for metric in ("f1", "recall", "group_f1", "relative_coef_error", "nnz"):
            values = [value(r, metric) for r in good if finite(value(r, metric))]
            row[metric + "_range"] = max(values)-min(values) if values else None
        rows.append(row)
    return {"by_problem": rows, "f1_range": stats(r["f1_range"] for r in rows),
            "coefficient_error_range": stats(r["relative_coef_error_range"] for r in rows),
            "interpretation": "Agreement uses only methods passing the final common criterion; partial-method sets are identified by their count."}


def modern_summary(runs, tolerance):
    rows = [r for r in runs if r.get("method") == "skglm"]
    return {"n_runs": len(rows), "n_converged": sum(converged(r, tolerance) for r in rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "attempt_count": stats(r.get("attempt_count", len(r.get("attempts", []))) for r in rows),
            "attempt_count_histogram": dict(Counter(str(r.get("attempt_count", len(r.get("attempts", [])))) for r in rows)),
            "library_tolerance_attempt_counts": dict(Counter(str(a.get("library_tol")) for r in rows for a in r.get("attempts", []))),
            "certificate_passed_but_not_converged_status": sum(bool(r.get("certificate_passed")) and not converged(r, tolerance) for r in rows),
            "threshold_curves_excluded": len(rows),
            "interpretation": "All tolerance-ladder refits and posterior checks are included in elapsed time. Library fits cannot be interrupted at the exact wall-clock cap."}


def real_summary(runs, config, dataset):
    tol = config["solver"]["tol"]
    best = [r for r in runs if r.get("selected_by_validation")] if dataset == "uji" else runs
    fields = ("method",) if dataset == "uji" else ("lambda_ratio", "method")
    output = []
    for keys, group in sorted(group_by(best, fields).items()):
        good = [r for r in group if converged(r, tol)]
        row = dict(zip(fields, keys))
        row.update(n_runs=len(group), n_converged=len(good), status_counts=dict(Counter(r["status"] for r in group)))
        for metric in ("elapsed_sec", "cold_start_sec", "iterations"):
            row[metric] = stats(r.get(metric) for r in good)
        for metric in ("nnz", "primal", "normalized_gap", "test_mse", "validation_mse"):
            row[metric] = stats(value(r, metric) for r in group)
        output.append(row)
    selected_rows = [{k: (value(r, k) if k in ("nnz", "normalized_gap", "primal") else r.get(k))
                      for k in ("run_id", "seed", "repeat", "method", "lambda", "lambda_ratio", "validation_index",
                                "validation_mse", "test_mse", "nnz", "primal", "normalized_gap", "elapsed_sec", "cold_start_sec", "status")}
                     for r in best]
    return {"aggregated": output, "selected_rows": selected_rows, "n_all_runs": len(runs),
            "n_selected_rows": len(best), "quality": quality(runs, tol),
            "scope": ("UJI longitude random held-out regression; validation selects lambda, test is not used for selection."
                      if dataset == "uji" else "One fixed 72-observation binary-label squared-loss optimization task; repeats describe timing, not new datasets or prediction validation.")}


def status_table(runs, failures, config):
    result = []
    for dataset in STAGES:
        records = list(runs[dataset])
        run_ids = {r.get("run_id") for r in records}
        records += [dict(r, status="exception") for r in failures[dataset] if r.get("run_id") not in run_ids]
        for keys, rows in sorted(group_by(records, ("method", "status")).items()):
            result.append({"dataset": dataset, "method": keys[0], "status": keys[1], "n_runs": len(rows),
                           "n_common_criterion_converged": sum(converged(r, config["solver"]["tol"]) for r in rows),
                           "n_certificate_passed": sum(finite(final(r).get("normalized_gap"))
                                                      and final(r)["normalized_gap"] <= config["solver"]["tol"] for r in rows),
                           "elapsed_sec_median_all_status": stats(r.get("elapsed_sec") for r in rows)["median"]})
    return result


def save_figure(fig, directory: Path, name: str):
    directory.mkdir(parents=True, exist_ok=True)
    if FIGURE_STATUS_NOTE:
        fig.text(.995, .005, FIGURE_STATUS_NOTE, ha="right", va="bottom", fontsize=7, color="#b82020",
                 bbox={"facecolor": "white", "edgecolor": "#b82020", "alpha": .9, "pad": 2})
    fig.savefig(directory / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(directory / (name + ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def decorate(ax):
    ax.grid(True, color="#dddddd", alpha=.65, linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)


def axes_grid(config, title):
    settings = config["synthetic"]
    fig, axes = plt.subplots(len(settings["layouts"]), len(settings["sparsities"]),
                             figsize=(11.8, 6.5), squeeze=False, constrained_layout=True)
    for i, layout in enumerate(settings["layouts"]):
        for j, sparsity in enumerate(settings["sparsities"]):
            ax = axes[i, j]
            ax.set_title(f"{layout.capitalize()}, s={sparsity}")
            ax.set_xticks(settings["rhos"])
            ax.set_xlabel(r"Within-block correlation $\rho$")
            decorate(ax)
    fig.suptitle(title)
    return fig, axes


def plot_timing(runs, config, figures, metric="elapsed_sec", name="timing"):
    factor = 1.
    label = "Solver time" if metric == "elapsed_sec" else "Solver + proximal SVD setup"
    fig, axes = axes_grid(config, label + "; c=1, median and IQR of converged paired seeds")
    for i, layout in enumerate(config["synthetic"]["layouts"]):
        for j, sparsity in enumerate(config["synthetic"]["sparsities"]):
            ax = axes[i, j]
            censored = []
            for method in config["methods"]:
                measures = []
                for rho in config["synthetic"]["rhos"]:
                    group = select(runs, layout=layout, sparsity=sparsity, rho=rho, lambda_factor=factor, method=method)
                    good = [r for r in group if converged(r, config["solver"]["tol"])]
                    measures.append(stats(r.get(metric) for r in good))
                    unfinished = [r for r in group if not converged(r, config["solver"]["tol"])]
                    if unfinished:
                        censored.append(f"{method} rho={rho:g}: {len(unfinished)}/{len(group)} not converged")
                        boundary = max(config["solver"]["time_limit"], max(r.get(metric, 0.) for r in unfinished))
                        ax.scatter([rho], [boundary], marker="x", color=COLORS[method], s=26)
                arrays = [[s[k] if s["n"] else np.nan for s in measures] for k in ("median", "q25", "q75")]
                ax.plot(config["synthetic"]["rhos"], arrays[0], color=COLORS[method], marker=MARKERS[method],
                        linestyle=STYLES[method], linewidth=1.5, markersize=4, label=LABELS[method])
                ax.fill_between(config["synthetic"]["rhos"], arrays[1], arrays[2], color=COLORS[method], alpha=.12)
            ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel("Time at common final criterion (s)")
            if censored:
                ax.text(.02, .03, "\n".join(censored), transform=ax.transAxes, fontsize=5.8,
                        va="bottom", bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    save_figure(fig, figures, name)


def plot_recovery(runs, config, figures, metric, ylabel, name):
    unfinished = sum(not converged(r, config["solver"]["tol"]) for r in select(runs, lambda_factor=1.))
    fig, axes = axes_grid(config, f"{ylabel}; c=1, final iterates, median/IQR (unconverged: {unfinished})")
    for i, layout in enumerate(config["synthetic"]["layouts"]):
        for j, sparsity in enumerate(config["synthetic"]["sparsities"]):
            ax = axes[i, j]
            for method in config["methods"]:
                summaries = [stats(value(r, metric) for r in select(runs, layout=layout, sparsity=sparsity,
                                  rho=rho, lambda_factor=1., method=method)) for rho in config["synthetic"]["rhos"]]
                arrays = [[s[k] if s["n"] else np.nan for s in summaries] for k in ("median", "q25", "q75")]
                ax.plot(config["synthetic"]["rhos"], arrays[0], color=COLORS[method], marker=MARKERS[method],
                        linestyle=STYLES[method], linewidth=1.4, markersize=4, markerfacecolor="none", label=LABELS[method])
                ax.fill_between(config["synthetic"]["rhos"], arrays[1], arrays[2], color=COLORS[method], alpha=.07)
                for rho in config["synthetic"]["rhos"]:
                    group = select(runs, layout=layout, sparsity=sparsity, rho=rho, lambda_factor=1., method=method)
                    if any(not converged(r, config["solver"]["tol"]) for r in group):
                        ax.text(rho, 1.025, "*", color=COLORS[method], ha="center", fontsize=9)
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.set_ylim(-.035, 1.07)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    save_figure(fig, figures, name)


def plot_heatmap(runs, config, figures, metric, title, name):
    settings = config["synthetic"]
    fig, axes = plt.subplots(1, len(settings["layouts"]), figsize=(9.4, 3.9), squeeze=False, constrained_layout=True)
    image = None
    for ax, layout in zip(axes.flat, settings["layouts"]):
        values = np.full((len(settings["sparsities"]), len(settings["rhos"])), np.nan)
        for i, s in enumerate(settings["sparsities"]):
            for j, rho in enumerate(settings["rhos"]):
                group = select(runs, method="cd", lambda_factor=1., layout=layout, sparsity=s, rho=rho)
                available = [value(r, metric) for r in group if finite(value(r, metric))]
                if available:
                    values[i, j] = float(np.mean(available) if metric == "exact_support" else np.median(available))
                    text = f"{values[i,j]:.2f}\n(n={len(available)})"
                    if any(not converged(r, config["solver"]["tol"]) for r in group):
                        text += "*"
                else:
                    text = "missing"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="white" if finite(values[i, j]) and values[i, j] > .65 else "#24323b")
        image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(len(settings["rhos"])), [f"{r:g}" for r in settings["rhos"]])
        ax.set_yticks(range(len(settings["sparsities"])), settings["sparsities"])
        ax.set_xlabel(r"Within-block correlation $\rho$")
        ax.set_ylabel("True nonzero coefficients s")
        ax.set_title(layout.capitalize())
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), fraction=.035, label=title)
    fig.suptitle(f"{title}: cyclic CD, c=1; * includes an unconverged final iterate")
    save_figure(fig, figures, name)


def plot_tradeoff(runs, config, figures):
    settings = config["synthetic"]
    factor_colors = dict(zip(settings["lambda_factors"], ("#3269a8", "#15946e", "#d67924")))
    s_markers = dict(zip(settings["sparsities"], ("o", "^", "s")))
    fig, axes = plt.subplots(2, len(settings["layouts"]), figsize=(10.0, 7.0), squeeze=False, constrained_layout=True)
    for col, layout in enumerate(settings["layouts"]):
        for row, metric in enumerate(("f1", "group_f1")):
            ax = axes[row, col]
            for s in settings["sparsities"]:
                for rho in settings["rhos"]:
                    points = []
                    for factor in settings["lambda_factors"]:
                        group = select(runs, method="cd", layout=layout, sparsity=s, rho=rho, lambda_factor=factor)
                        x, y = stats(value(r, "relative_coef_error") for r in group), stats(value(r, metric) for r in group)
                        if x["n"] and y["n"]:
                            points.append((x["median"], y["median"]))
                            ax.scatter(x["median"], y["median"], s=35, marker=s_markers[s], color=factor_colors[factor], alpha=.85)
                    if len(points) > 1:
                        ax.plot(*zip(*points), color="#aaaaaa", linewidth=.6, alpha=.65, zorder=0)
            ax.set_xlabel("Median relative coefficient error")
            ax.set_ylabel("Median feature F1" if metric == "f1" else "Median group F1")
            ax.set_ylim(-.03, 1.04)
            ax.set_title(layout.capitalize())
            decorate(ax)
    handles = ([Line2D([], [], color=color, marker="o", linestyle="none", label=f"c={c:g}")
                for c, color in factor_colors.items()] +
               [Line2D([], [], color="#555555", marker=marker, linestyle="none", label=f"s={s}")
                for s, marker in s_markers.items()])
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    unfinished = sum(not converged(r, config["solver"]["tol"]) for r in select(runs, method="cd"))
    fig.suptitle(f"Regularization tradeoffs: CD final iterates (unconverged: {unfinished}); lines join c values")
    save_figure(fig, figures, "coefficient_support_tradeoff")


def plot_convergence(runs, config, figures):
    settings = config["synthetic"]
    rho_values = (min(settings["rhos"]), max(settings["rhos"]))
    sparsities = (min(settings["sparsities"]), max(settings["sparsities"]))
    seed = settings["seeds"][0]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.7), squeeze=False, constrained_layout=True)
    for ax, (s, rho) in zip(axes.flat, product(sparsities, rho_values)):
        for method in BASIC:
            group = select(runs, seed=seed, sparsity=s, rho=rho, layout="clustered", lambda_factor=1., method=method)
            if not group or not comparable_history(group[0]):
                continue
            run = group[0]
            h = [h for h in run.get("history", []) if finite(h.get("time")) and h["time"] > 0 and finite(h.get("normalized_gap"))]
            if h:
                ax.loglog([x["time"] for x in h], [max(1e-16, x["normalized_gap"]) for x in h],
                          color=COLORS[method], linewidth=1.5,
                          label=LABELS[method] + (" (unconverged)" if not converged(run, config["solver"]["tol"]) else ""))
        ax.axhline(config["solver"]["tol"], color="#888888", linestyle=":", linewidth=.8)
        ax.set_title(f"Clustered, s={s}, rho={rho:g}")
        ax.set_xlabel("Elapsed solver time including diagnostics (s)")
        ax.set_ylabel("Normalized feasible duality gap")
        decorate(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(f"Observed convergence: seed {seed}, c=1; basic solvers only")
    save_figure(fig, figures, "convergence")


def plot_accuracy(runs, config, figures):
    settings = config["synthetic"]
    tolerances = sorted(config["thresholds"], reverse=True)
    fig, axes = plt.subplots(2, len(settings["layouts"]), figsize=(9.6, 6.3), squeeze=False, constrained_layout=True)
    for col, layout in enumerate(settings["layouts"]):
        for row, metric in enumerate(("relative_coef_error", "f1")):
            ax = axes[row, col]
            for method in BASIC:
                group = select(runs, sparsity=max(settings["sparsities"]), rho=max(settings["rhos"]),
                               layout=layout, lambda_factor=1., method=method)
                points = []
                for t in tolerances:
                    hits = [threshold_hit(r, t) for r in group]
                    points.append(stats(h.get(metric) for h in hits if h is not None))
                arrays = [[s[k] if s["n"] else np.nan for s in points] for k in ("median", "q25", "q75")]
                ax.plot(tolerances, arrays[0], marker=MARKERS[method], color=COLORS[method], linewidth=1.5, label=LABELS[method])
                ax.fill_between(tolerances, arrays[1], arrays[2], color=COLORS[method], alpha=.13)
                for t, s in zip(tolerances, points):
                    if 0 < s["n"] < len(settings["seeds"]):
                        ax.annotate(f"n={s['n']}", (t, s["median"]), fontsize=6)
            ax.set_xscale("log")
            ax.set_xlim(max(tolerances)*1.4, min(tolerances)/1.4)
            ax.set_xticks(tolerances)
            ax.set_xlabel("First observed normalized-gap threshold")
            ax.set_ylabel("Relative coefficient error" if row == 0 else "Feature F1")
            ax.set_title(layout.capitalize())
            if row == 1:
                ax.set_ylim(-.03, 1.04)
            decorate(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(f"Optimization accuracy and recovery: s={max(settings['sparsities'])}, rho={max(settings['rhos']):g}, c=1")
    save_figure(fig, figures, "accuracy_recovery")


def main():
    global FIGURE_STATUS_NOTE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, help="Default: frozen synthetic/config.json, otherwise configs/study_v2.json")
    parser.add_argument("--allow-partial", action="store_true", help="Development only; missing/inconsistent results remain explicitly marked")
    parser.add_argument("--skip-figures", action="store_true", help="Write validation and tables without plotting")
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / "results/v2"
    config_path = args.config or (results / "synthetic/config.json" if (results / "synthetic/config.json").is_file()
                                  else root / "configs/study_v2.json")
    config = read(config_path)
    snapshots = {stage: {"config": read(results / stage / "config.json")
                        if (results / stage / "config.json").is_file() else None,
                        "environment": read(results / stage / "environment.json")
                        if (results / stage / "environment.json").is_file() else None}
                 for stage in STAGES}
    # run_v2 hashes the original input file, then writes a reformatted JSON
    # snapshot. The snapshot has the same values but different bytes. Recover
    # its recorded original hash instead of hashing those reformatted bytes.
    env = snapshots["synthetic"]["environment"]
    frozen_default = args.config is None and (results / "synthetic/config.json").is_file()
    config_hash = (env["v2_config_sha256"] if frozen_default and env and env.get("v2_config_sha256")
                   else hashlib.sha256(config_path.read_bytes()).hexdigest())
    runs = {stage: load_records(results / stage, "runs") for stage in STAGES}
    datasets = {stage: load_records(results / stage, "datasets") for stage in STAGES}
    failures = {stage: load_records(results / stage, "failures") for stage in STAGES}
    integrity, design_map = expected_and_integrity(runs, datasets, failures, snapshots, config, args.allow_partial, config_hash)
    FIGURE_STATUS_NOTE = "DEVELOPMENT: INCOMPLETE/INCONSISTENT GRID" if not integrity["complete_grid"] else ""
    synthetic = runs["synthetic"]
    thresholds, threshold_summary = threshold_tables(synthetic, config)
    statuses = status_table(runs, failures, config)
    for stage in STAGES:
        write_csv(results / (stage + "_runs.csv"), run_table(runs[stage], design_map, config))
    write_csv(results / "threshold_recovery.csv", thresholds)
    write_csv(results / "status_table.csv", statuses)
    summary = {
        "protocol_version": config.get("protocol_version"), "integrity": integrity,
        "definitions": {
            "converged": "status=converged AND final normalized gap <= configured tolerance; passed certificates over a library budget are not silently accepted as converged timings",
            "time": "Median/IQR only among converged runs. Final-iterate recovery retains unconverged runs, with counts and converged-only summaries alongside.",
            "uncertainty": "Five paired synthetic seeds describe instance variation, not confidence intervals. Leukemia repeats describe timing on one fixed dataset.",
            "thresholds": "Basic solver checkpoints only; observed crossing upper bounds, not interpolation. skglm initial/fit endpoints are excluded.",
            "cold_start": "Solver elapsed plus dense SVD for ISTA/FISTA; data preparation and one-time library compilation/loading are separate.",
            "exact_recovery": "Empirical success counts/rates over the observed seeds, not a high-precision probability estimate.",
            "structural_diagnostics": "Computed using known synthetic truth; not finite-sample guarantees. Global Hessian condition is not reported for p>n."},
        "group_metric_baseline": [{"layout": layout, "sparsity": s,
                "true_groups": s if layout == "dispersed" else s // 5,
                "all_groups_selected_f1": 2 * (s if layout == "dispersed" else s // 5) /
                    (config["synthetic"]["p"] // config["synthetic"]["block_size"] +
                     (s if layout == "dispersed" else s // 5))}
                for layout in config["synthetic"]["layouts"] for s in config["synthetic"]["sparsities"]],
        "config": config,
        "by_cell": aggregate_cells(synthetic, config),
        "paired_rho_high_over_low": paired_contrasts(synthetic, config, "rho", min(config["synthetic"]["rhos"]),
                max(config["synthetic"]["rhos"]), ("sparsity", "layout", "lambda_factor", "method")),
        "paired_sparsity_high_over_low": paired_contrasts(synthetic, config, "sparsity", min(config["synthetic"]["sparsities"]),
                max(config["synthetic"]["sparsities"]), ("rho", "layout", "lambda_factor", "method")),
        "paired_clustered_over_dispersed": paired_contrasts(synthetic, config, "layout", "dispersed", "clustered",
                ("rho", "sparsity", "lambda_factor", "method")),
        "rho_layout_interactions": layout_rho_interactions(synthetic, config),
        "threshold_recovery": threshold_summary,
        "structure": structural_summary(datasets["synthetic"]),
        "support_sensitivity": sensitivity_summary(synthetic, config),
        "algorithm_agreement": agreement_summary(synthetic, config),
        "quality": {stage: quality(runs[stage], config["solver"]["tol"]) for stage in STAGES},
        "quality_by_method": {stage: {m: quality(select(runs[stage], method=m), config["solver"]["tol"])
                                      for m in config["methods"]} for stage in STAGES},
        "skglm": {stage: modern_summary(runs[stage], config["solver"]["tol"]) for stage in STAGES},
        "status_table": statuses,
        "uji": real_summary(runs["uji"], config, "uji"),
        "leukemia": real_summary(runs["leukemia"], config, "leukemia"),
        "setup": {stage: {k: stats(d.get(k) for d in datasets[stage]) for k in
                            ("preprocessing_sec", "lipschitz_setup_sec", "selection_sec", "n", "p", "used_features")}
                  for stage in STAGES},
    }
    dump(results / "summary.json", summary)
    if synthetic and not args.skip_figures:
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.6, "axes.titlesize": 10,
                             "axes.labelsize": 8.3, "legend.fontsize": 8, "pdf.fonttype": 42})
        figures = root / "figures/v2"
        plot_timing(synthetic, config, figures)
        plot_timing(synthetic, config, figures, "cold_start_sec", "timing_cold_start")
        plot_recovery(synthetic, config, figures, "f1", "Feature F1", "feature_f1")
        plot_recovery(synthetic, config, figures, "recall", "Feature recall", "feature_recall")
        plot_recovery(synthetic, config, figures, "group_f1", "Group F1", "group_f1")
        plot_heatmap(synthetic, config, figures, "f1", "Median feature F1", "sparsity_rho_f1")
        plot_heatmap(synthetic, config, figures, "exact_support", "Exact-support fraction", "sparsity_rho_exact_support")
        plot_tradeoff(synthetic, config, figures)
        plot_convergence(synthetic, config, figures)
        plot_accuracy(synthetic, config, figures)
    print(json.dumps({"integrity": integrity, "summary": str(results / "summary.json"),
                      "synthetic_cells": len(summary["by_cell"]), "figures_written": bool(synthetic and not args.skip_figures)}, indent=2))


if __name__ == "__main__":
    main()
