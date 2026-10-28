#!/usr/bin/env python3
"""Post-hoc real-design numerical-reference diagnostics; no optimization.

Includes every registered real target, accepted or uncertain. Reconstructs one
predictor design per seed, then each response with the original noise stream.
Numerical rank and near-proportionality use explicit tolerances. Neither equal
objectives nor a rank diagnostic is a proof of mathematical non-uniqueness.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / ".modern_deps")]
from lasso_study.data import array_sha256
from lasso_study.data_v2 import make_uji
from lasso_study.workshop_data import make_real_design
from lasso_study.solvers import certificate
from run_workshop import dump, sha
from run_workshop_followup import followup_targets, followup_target_id, followup_signature


def stats(values):
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return {"n": int(len(x)), "median": float(np.median(x)) if len(x) else None,
            "min": float(x.min()) if len(x) else None, "max": float(x.max()) if len(x) else None}


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def column_identity(metadata, index):
    return {"column_0based": int(index), "feature_name": metadata["feature_names"][index],
            "source_pool_column_0based": metadata["retained_source_columns_0based"][index],
            "original_wap_column_0based": metadata["retained_original_wap_columns_0based"][index]}


def exact_duplicate_groups(X):
    positive, negative = defaultdict(list), defaultdict(list)
    for j in range(X.shape[1]):
        positive[X[:, j].tobytes()].append(j)
        negative[(-X[:, j]).tobytes()].append(j)
    same = [indices for indices in positive.values() if len(indices) > 1]
    opposite = []
    for representation, first in positive.items():
        second = negative.get(representation, [])
        if second and min(first) < min(second):
            opposite.append({"first": first, "opposite": second})
    return same, opposite


def x_only_near_proportional_groups(X, tolerances=(1e-12, 1e-10, 1e-14)):
    """Deterministic anchor grouping using X alone, never coefficients/truth.

    Greedy anchors follow the retained WAP column order. Every member is tested
    directly against its anchor, so transitive links cannot weaken the stated
    residual criterion. The coarse cosine filter only avoids impossible pairs;
    final membership uses the explicitly formed regression residual vector.
    """
    norms = np.linalg.norm(X, axis=0)
    cosine = (X.T @ X) / (norms[:, None] * norms[None, :])
    pair_fit = {}
    for anchor in range(X.shape[1]):
        candidates = np.flatnonzero(np.abs(cosine[anchor]) >= 1 - 1e-8)
        for member in candidates[candidates > anchor]:
            alpha = float(X[:, anchor] @ X[:, member] / (norms[anchor] ** 2))
            residual = float(np.linalg.norm(X[:, member] - alpha * X[:, anchor]) / norms[member])
            pair_fit[anchor, int(member)] = alpha, residual
    output = {}
    for tolerance in tolerances:
        remaining = set(range(X.shape[1])); groups = []
        while remaining:
            anchor = min(remaining)
            members, scales, residuals = [anchor], [1.], [0.]
            for member in sorted(remaining - {anchor}):
                relation = pair_fit.get((anchor, member))
                if relation is not None and relation[1] <= tolerance:
                    members.append(member); scales.append(relation[0]); residuals.append(relation[1])
            remaining.difference_update(members)
            groups.append({"anchor_column_0based": anchor, "columns_0based": members,
                           "member_scales_from_anchor": scales, "relative_l2_residuals_to_anchor": residuals})
        output[str(tolerance)] = groups
    return output


def aggregate_reference_groups(groups, left, right, metadata, taus):
    first, second, nontrivial = [], [], []
    for group in groups:
        columns = group["columns_0based"]
        scales = np.asarray(group["member_scales_from_anchor"])
        a, b = float(scales @ left[columns]), float(scales @ right[columns])
        first.append(a); second.append(b)
        if len(columns) > 1:
            nontrivial.append(dict(group, feature_names=[metadata["feature_names"][j] for j in columns],
                                   aggregated_coef_sklearn=a, aggregated_coef_skglm=b, coefficient_difference=a-b,
                                   original_group_l1_sklearn=float(np.abs(left[columns]).sum()),
                                   original_group_l1_skglm=float(np.abs(right[columns]).sum())))
    first, second = np.asarray(first), np.asarray(second)
    return {"group_count": len(groups), "nontrivial_group_count": len(nontrivial),
            "aggregated_coefficient_linf_difference": float(np.max(np.abs(first-second))),
            "aggregated_coefficient_l2_difference": float(np.linalg.norm(first-second)),
            "support_comparisons": {str(tau): {
                "same_group_support": bool(np.array_equal(np.abs(first)>tau, np.abs(second)>tau)),
                "group_support_symmetric_difference": int(np.sum((np.abs(first)>tau) != (np.abs(second)>tau))),
                "sklearn_selected_groups": int(np.sum(np.abs(first)>tau)),
                "skglm_selected_groups": int(np.sum(np.abs(second)>tau))} for tau in taus},
            "nontrivial_group_aggregates": nontrivial}


def union_geometry(X, left, right, tau):
    union = np.flatnonzero((np.abs(left) > tau) | (np.abs(right) > tau))
    if not len(union):
        return {"support_threshold": tau, "union_size": 0, "union_indices": [], "numerical_rank": 0,
                "rank_deficient_at_reported_tolerance": False, "minimum_singular_value_X_over_sqrt_n": None,
                "minimum_gram_eigenvalue": None, "difference_rayleigh_curvature": None,
                "difference_energy_fraction_in_small_singular_subspace": None}
    scaled = X[:, union] / np.sqrt(len(X))
    _, values, vt = np.linalg.svd(scaled, full_matrices=False)
    cutoff = max(scaled.shape) * np.finfo(float).eps * values[0]
    rank = int(np.sum(values > cutoff))
    gram = scaled.T @ scaled
    eigenvalues = np.linalg.eigvalsh(gram)
    delta = left[union] - right[union]
    squared_norm = float(delta @ delta)
    prediction = scaled @ delta
    near = values <= max(cutoff, values[0] * 1e-8)
    projection = float(np.sum((vt[near] @ delta) ** 2))
    if scaled.shape[1] > scaled.shape[0]:
        projection += max(0., squared_norm - float(np.sum((vt @ delta) ** 2)))
    return {"support_threshold": tau, "union_size": int(len(union)), "union_indices": union.tolist(),
            "numerical_rank": rank, "rank_tolerance": float(cutoff),
            "rank_deficient_at_reported_tolerance": rank < len(union),
            "minimum_singular_value_X_over_sqrt_n": float(values[-1]) if scaled.shape[1] <= scaled.shape[0] else 0.,
            "maximum_singular_value_X_over_sqrt_n": float(values[0]),
            "smallest_singular_value_above_rank_tolerance": float(values[values > cutoff][-1]) if rank else None,
            "minimum_gram_eigenvalue": float(eigenvalues[0]), "maximum_gram_eigenvalue": float(eigenvalues[-1]),
            "difference_rayleigh_curvature": float(prediction @ prediction / squared_norm) if squared_norm else None,
            "difference_energy_fraction_in_small_singular_subspace": min(1., projection / squared_norm) if squared_norm else None}


def diagnose(config_path):
    start = perf_counter(); base = ROOT / "results/workshop/real"
    provenance = {}

    def read(path):
        provenance[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())

    config = read(config_path)
    expected = followup_targets(config, "real")
    targets = {followup_target_id(params): params for params in expected}
    env, completion = read(base / "environment.json"), read(base / "completion.json")
    config_snapshot = read(base / "effective_config.json")
    if config_snapshot != config:
        raise RuntimeError("Real configuration differs from its frozen snapshot")
    signature, signature_info = followup_signature(config_path)
    if env["numerical_signature"] != signature or env["config_sha256"] != sha(config_path):
        raise RuntimeError("Real stage numerical/config signature differs from current frozen sources")
    if completion["recorded_targets"] != len(targets) or completion["expected_targets"] != len(targets):
        raise RuntimeError("Real stage is incomplete")
    observed = {path.stem for path in (base / "references").glob("*.json")}
    if observed != set(targets):
        raise RuntimeError("Reference target set differs from registered real grid")
    source = make_uji(seed=0)
    designs = {}; seed_diagnostics = {}; records = []; csv_rows = []; coordinate_rows = []; pair_rows = []
    groupings = {}; grouped_csv_rows = []; x_group_csv_rows = []
    for identifier, params in targets.items():
        metadata = read(base / "datasets" / (identifier + ".json"))
        refs = read(base / "references" / (identifier + ".json"))
        cp = base / "reference_coefficients" / (identifier + ".npz")
        provenance[str(cp.relative_to(ROOT))] = sha(cp)
        if provenance[str(cp.relative_to(ROOT))] != refs["coefficient_sha256"]:
            raise RuntimeError(f"Reference coefficient SHA mismatch: {identifier}")
        with np.load(cp, allow_pickle=False) as arrays:
            left, right, truth = (arrays[name].copy() for name in ("ref_sklearn", "ref_skglm", "w_true"))
        seed = params["seed"]
        if seed not in designs:
            baseline = make_real_design(config, seed, config["layouts"][0], config["amplitudes"][0], source=source)
            X = np.asfortranarray(baseline["X"])
            noise_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(8)[6])
            noise = config["noise_std"] * noise_rng.normal(size=len(X))
            designs[seed] = X, noise
            same, opposite = exact_duplicate_groups(X)
            groupings[seed] = x_only_near_proportional_groups(X)
            grouping_records = {}
            for tolerance, groups in groupings[seed].items():
                nontrivial = [group for group in groups if len(group["columns_0based"]) > 1]
                grouping_records[tolerance] = {
                    "group_count": len(groups), "nontrivial_group_count": len(nontrivial),
                    "singleton_columns": [group["anchor_column_0based"] for group in groups if len(group["columns_0based"]) == 1],
                    "nontrivial_groups": nontrivial,
                    "maximum_member_anchor_relative_residual": max(max(group["relative_l2_residuals_to_anchor"]) for group in groups),
                    "maximum_absolute_scale_magnitude_deviation_from_one": max(max(abs(abs(a)-1) for a in group["member_scales_from_anchor"]) for group in groups)}
                for group in nontrivial:
                    for member, scale, residual in zip(group["columns_0based"], group["member_scales_from_anchor"], group["relative_l2_residuals_to_anchor"]):
                        x_group_csv_rows.append({"seed": seed, "relative_residual_tolerance": float(tolerance),
                                                 "anchor_column_0based": group["anchor_column_0based"],
                                                 "anchor_feature": metadata["feature_names"][group["anchor_column_0based"]],
                                                 **column_identity(metadata, member), "scale_from_anchor": scale,
                                                 "relative_residual_to_anchor": residual})
            seed_diagnostics[str(seed)] = {
                "n": len(X), "p": X.shape[1], "exact_equal_float64_column_groups": same,
                "exact_opposite_float64_column_groups": opposite,
                "exact_group_feature_names": [[metadata["feature_names"][j] for j in group] for group in same],
                "x_only_groupings": grouping_records}
        X, noise = designs[seed]
        raw_y = X @ truth + noise
        y = np.asarray(raw_y - raw_y.mean(), dtype=np.float64)
        if (array_sha256(X) != metadata["hashes"]["X"] or array_sha256(y) != metadata["hashes"]["y"]
                or array_sha256(truth) != metadata["hashes"]["w_true"]):
            raise RuntimeError(f"Seed-cached design/response reconstruction differs: {identifier}")
        delta = left - right
        prediction_left, prediction_right = X @ left, X @ right
        prediction_delta = X @ delta
        lam = metadata["lambda"]
        certificates = {"sklearn": certificate(X, y, left, lam), "skglm": certificate(X, y, right, lam)}
        l1_left, l1_right = float(np.abs(left).sum()), float(np.abs(right).sum())
        primary = config["support_threshold"]
        geometries = {str(tau): union_geometry(X, left, right, tau) for tau in config["support_thresholds"]}
        aggregated = {tolerance: aggregate_reference_groups(groups, left, right, metadata, config["support_thresholds"])
                      for tolerance, groups in groupings[seed].items()}
        for tolerance, group_result in aggregated.items():
            flat_group = {"target_id": identifier, "seed": seed, "layout": params["layout"],
                          "amplitude": params["amplitude"], "reference_accepted": refs["reference_accepted"],
                          "relative_residual_tolerance": float(tolerance),
                          **{name: value for name, value in group_result.items() if not isinstance(value, (dict, list))}}
            for tau, comparison in group_result["support_comparisons"].items():
                flat_group.update({f"tau_{tau}_{name}": value for name, value in comparison.items()})
            grouped_csv_rows.append(flat_group)
        significant = np.flatnonzero(np.abs(delta) > 1e-6)
        ranked = sorted(significant.tolist(), key=lambda j: (-abs(delta[j]), j))[:20]
        coordinates = []
        for j in ranked:
            values, counts = np.unique(X[:, j], return_counts=True)
            modal = values[np.argmax(counts)]
            nonmodal = np.flatnonzero(X[:, j] != modal)
            row = dict(column_identity(metadata, j), ref_sklearn=float(left[j]), ref_skglm=float(right[j]),
                       coefficient_difference=float(delta[j]), target_id=identifier,
                       distinct_stored_values=int(len(values)), dominant_value_fraction=float(counts.max()/len(X)),
                       nonmodal_sample_row_count=int(len(nonmodal)), nonmodal_sample_rows_first20=nonmodal[:20].tolist())
            coordinates.append(row); coordinate_rows.append(row)
        pairs = []
        for offset, j in enumerate(ranked):
            xj = X[:, j]; normj = float(np.linalg.norm(xj))
            for k in ranked[offset+1:]:
                xk = X[:, k]; normk = float(np.linalg.norm(xk))
                alpha = float(xk @ xj / (normk * normk))
                residual = xj - alpha * xk
                relative = float(np.linalg.norm(residual) / normj)
                corr = float((xj @ xk) / (normj * normk))
                pair = {"target_id": identifier, "column_j_0based": int(j), "column_k_0based": int(k),
                        "feature_j": metadata["feature_names"][j], "feature_k": metadata["feature_names"][k],
                        "original_wap_j_0based": metadata["retained_original_wap_columns_0based"][j],
                        "original_wap_k_0based": metadata["retained_original_wap_columns_0based"][k],
                        "correlation": corr, "exact_equal_stored_float64": bool(np.array_equal(xj, xk)),
                        "exact_opposite_stored_float64": bool(np.array_equal(xj, -xk)),
                        "best_proportional_scale_j_from_k": alpha, "proportional_relative_l2_residual": relative,
                        "proportional_max_abs_residual": float(np.max(np.abs(residual))),
                        "raw_column_difference_max_abs": float(np.max(np.abs(xj-xk))),
                        "delta_j": float(delta[j]), "delta_k": float(delta[k]),
                        "two_coordinate_prediction_delta_rmse": float(np.linalg.norm(xj*delta[j]+xk*delta[k])/np.sqrt(len(X)))}
                pairs.append(pair)
        pairs.sort(key=lambda row: (row["proportional_relative_l2_residual"],
                                    -abs(row["delta_j"])-abs(row["delta_k"]), row["column_j_0based"], row["column_k_0based"]))
        reported_pairs = pairs[:10]
        pair_rows.extend(reported_pairs)
        adjacency = {j: set() for j in ranked}
        for pair in pairs:
            if pair["proportional_relative_l2_residual"] < 1e-8:
                j, k = pair["column_j_0based"], pair["column_k_0based"]
                adjacency[j].add(k); adjacency[k].add(j)
        components = []
        remaining = set(ranked)
        while remaining:
            stack = [min(remaining)]; component = set()
            while stack:
                j = stack.pop()
                if j not in component:
                    component.add(j); stack.extend(adjacency[j] - component)
            remaining -= component
            if len(component) >= 2:
                indices = sorted(component)
                components.append({"columns_0based": indices,
                                   "feature_names": [metadata["feature_names"][j] for j in indices],
                                   "coefficient_differences": delta[indices].tolist(),
                                   "raw_coefficient_difference_sum": float(delta[indices].sum()),
                                   "group_prediction_delta_rmse": float(np.linalg.norm(X[:, indices]@delta[indices])/np.sqrt(len(X))),
                                   "group_l1_norm_sklearn": float(np.abs(left[indices]).sum()),
                                   "group_l1_norm_skglm": float(np.abs(right[indices]).sum())})
        objective_delta = abs(certificates["sklearn"]["primal"] - certificates["skglm"]["primal"])
        row = {"target_id": identifier, **params, "reference_accepted": refs["reference_accepted"],
               "coefficient_linf_difference": float(np.max(np.abs(delta))), "coefficient_l2_difference": float(np.linalg.norm(delta)),
               "prediction_delta_rmse": float(np.linalg.norm(prediction_delta)/np.sqrt(len(X))),
               "prediction_delta_max_abs": float(np.max(np.abs(prediction_delta))),
               "prediction_relative_l2_difference": float(np.linalg.norm(prediction_delta)/max(1., np.linalg.norm(prediction_left), np.linalg.norm(prediction_right))),
               "l1_norm_sklearn": l1_left, "l1_norm_skglm": l1_right, "l1_norm_absolute_difference": abs(l1_left-l1_right),
               "objective_sklearn": certificates["sklearn"]["primal"], "objective_skglm": certificates["skglm"]["primal"],
               "absolute_objective_difference": objective_delta,
               "normalized_objective_difference": objective_delta/max(1., abs(certificates["sklearn"]["primal"]), abs(certificates["skglm"]["primal"])),
               "n_times_objective_sklearn": len(X)*certificates["sklearn"]["primal"],
               "n_times_objective_skglm": len(X)*certificates["skglm"]["primal"],
               "significant_coefficient_difference_count": int(len(significant)),
               "support_agreement": {str(tau): bool(np.array_equal(np.abs(left)>tau,np.abs(right)>tau)) for tau in config["support_thresholds"]},
               "union_active_geometry": geometries, "recomputed_certificates": certificates,
               "x_only_group_aggregation": aggregated,
               "significant_coordinates_top20": coordinates, "closest_proportional_pairs_top10": reported_pairs,
               "near_proportional_connected_components_among_changed_coordinates": components,
               "changed_coordinates_include_exact_equal_pair": any(pair["exact_equal_stored_float64"] for pair in pairs),
               "changed_coordinates_include_near_proportional_pair": any(pair["proportional_relative_l2_residual"] < 1e-8 for pair in pairs)}
        records.append(row)
        flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for tau, geometry in geometries.items():
            flat[f"union_numerical_rank_tau_{tau}"] = geometry["numerical_rank"]
            flat[f"union_size_tau_{tau}"] = geometry["union_size"]
            flat[f"union_sigma_min_tau_{tau}"] = geometry["minimum_singular_value_X_over_sqrt_n"]
            flat[f"delta_small_subspace_fraction_tau_{tau}"] = geometry["difference_energy_fraction_in_small_singular_subspace"]
        csv_rows.append(flat)
        print(json.dumps({"target": identifier, "accepted": refs["reference_accepted"],
                          "coef_delta": row["coefficient_linf_difference"], "prediction_delta_rmse": row["prediction_delta_rmse"],
                          "union_rank": geometries[str(primary)]["numerical_rank"],
                          "union_size": geometries[str(primary)]["union_size"]}), flush=True)
    representative = max(records, key=lambda row: (row["coefficient_linf_difference"], row["target_id"]))
    measures = ("coefficient_linf_difference", "prediction_delta_rmse", "prediction_relative_l2_difference",
                "l1_norm_absolute_difference", "absolute_objective_difference", "normalized_objective_difference")
    summary = {
        "n_targets": len(records), "n_seeds": len(designs),
        "reference_accepted_targets": sum(row["reference_accepted"] for row in records),
        "reference_uncertain_targets": sum(not row["reference_accepted"] for row in records),
        "metric_summaries": {name: stats([row[name] for row in records]) for name in measures},
        "support_agreement_counts": {str(tau): sum(row["support_agreement"][str(tau)] for row in records) for tau in config["support_thresholds"]},
        "numerically_rank_deficient_union_counts": {str(tau): sum(row["union_active_geometry"][str(tau)]["rank_deficient_at_reported_tolerance"] for row in records) for tau in config["support_thresholds"]},
        "changed_coordinates_with_exact_equal_pair_targets": sum(row["changed_coordinates_include_exact_equal_pair"] for row in records),
        "changed_coordinates_with_near_proportional_pair_targets": sum(row["changed_coordinates_include_near_proportional_pair"] for row in records),
        "x_only_group_aggregation": {tolerance: {
            "n_targets": len(records),
            "aggregated_coefficient_linf_difference": stats([row["x_only_group_aggregation"][tolerance]["aggregated_coefficient_linf_difference"] for row in records]),
            "support_agreement_counts": {str(tau): sum(row["x_only_group_aggregation"][tolerance]["support_comparisons"][str(tau)]["same_group_support"] for row in records)
                                         for tau in config["support_thresholds"]},
            "support_agreement_descriptive_fraction": {str(tau): sum(row["x_only_group_aggregation"][tolerance]["support_comparisons"][str(tau)]["same_group_support"] for row in records)/len(records)
                                                      for tau in config["support_thresholds"]}}
                                      for tolerance in ("1e-12", "1e-10", "1e-14")},
        "representative_target": representative["target_id"],
    }
    report = {"diagnostic_type": "post-hoc numerical near-dependence investigation; no optimizer runs",
              "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "timezone": "Asia/Shanghai",
              "diagnostic_script_sha256": sha(Path(__file__)), "numerical_signature": signature,
              "config_sha256": sha(config_path), "signature_provenance": signature_info,
              "input_provenance": provenance, "summary": summary, "targets": records, "by_seed_design": seed_diagnostics,
              "representative_selection_rule": "Maximum coefficient Linf difference over all registered 40 targets, accepted and uncertain included; ties by target ID. This is post-hoc illustration, not independent confirmation.",
              "numerical_thresholds": {"significant_coefficient_abs_difference": 1e-6,
                                       "candidate_columns_max": 20, "reported_closest_pairs_max": 10,
                                       "small_singular_subspace_relative_threshold": 1e-8,
                                       "near_proportional_pair_relative_residual_threshold": 1e-8,
                                       "x_only_grouping_primary_residual_tolerance": 1e-12,
                                       "x_only_grouping_sensitivity_tolerances": [1e-10, 1e-14],
                                       "svd_rank_rule": "max(n, union_size) * machine_epsilon * largest singular value of X_union/sqrt(n)"},
              "x_only_grouping_rule": "Deterministic ascending-column anchors, using X only. Add member j iff ||X_j-a_j X_anchor||/||X_j|| <= tolerance with least-squares a_j. Each member is checked directly against its anchor; singletons retained. Aggregate coefficients by sum_j a_j*w_j. No truth, coefficients, acceptance flag, or outcome is used to construct groups.",
              "boundaries": ["Numerical near-dependence is diagnosed at reported floating-point thresholds; mathematical rank and non-uniqueness are not proved.",
                             "Exactly equal stored float64 columns are a property of this sampled/preprocessed representation; two small-gap numerical solutions still do not constitute an exact optimality certificate.",
                             "No reference is promoted to exact truth or chosen as the correct solution. All registered accepted and uncertain targets are included.",
                             "Source rows, response noise and predictor columns are shared across conditions within each seed; 40 targets are not 40 independent datasets.",
                             "Pair diagnostics inspect the top 20 coordinates by absolute coefficient disagreement, not all possible near dependencies."],
              "wall_sec": perf_counter()-start}
    dump(base / "reference_diagnostics.json", report)
    write_csv(base / "reference_diagnostics.csv", csv_rows)
    write_csv(base / "reference_difference_coordinates.csv", coordinate_rows)
    write_csv(base / "reference_difference_pairs.csv", pair_rows)
    write_csv(base / "reference_group_diagnostics.csv", grouped_csv_rows)
    write_csv(base / "reference_x_groups.csv", x_group_csv_rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/workshop_real.json")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        report = diagnose(args.config)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
