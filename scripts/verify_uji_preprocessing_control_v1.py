#!/usr/bin/env python3
"""Verify the completed UJI mechanism control without any optimizer calls.

Run only after the new reference-fit timers have finished. The verifier reads
old artifacts, reconstructs their fixed designs, independently derives column
groups and mappings, and recomputes certificates for saved coefficients.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
import hashlib
import importlib.metadata
from itertools import product
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import traceback
from zoneinfo import ZoneInfo

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".modern_deps")]
from lasso_study.data import array_sha256
from lasso_study.data_v2 import make_uji
from lasso_study.workshop_data import make_real_design
from lasso_study.solvers import certificate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self):
        self.inputs, self.errors, self.warnings, self.checks = {}, [], [], 0

    def check(self, condition, label):
        self.checks += 1
        if not bool(condition):
            self.errors.append(label)

    def track(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise RuntimeError(f"Invalid/missing evidence path: {path}")
        key, value = str(path.relative_to(ROOT)), sha(path)
        self.check(key not in self.inputs or self.inputs[key] == value, "Input changed during verification: "+key)
        self.inputs[key] = value
        return value

    def read(self, path):
        self.track(path)
        return json.loads(Path(path).read_text())

    def equal(self, actual, expected, label, atol=3e-12):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict), label+": mapping required")
            if isinstance(actual, dict):
                for key, value in expected.items():
                    self.check(key in actual, label+": missing "+str(key))
                    if key in actual:
                        self.equal(actual[key], value, label+"/"+str(key), atol)
        elif isinstance(expected, (list, tuple)):
            self.check(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label+": list differs")
            if isinstance(actual, (list, tuple)) and len(actual) == len(expected):
                for index, value in enumerate(expected):
                    self.equal(actual[index], value, label+f"/{index}", atol)
        elif isinstance(expected, (float, np.floating)):
            self.check(actual is not None and math.isclose(float(actual), float(expected), rel_tol=5e-10, abs_tol=atol), label+": numeric difference")
        else:
            self.check(actual == expected, label+": difference")


def equal_column_groups(X):
    seen, groups = {}, []
    for column in range(X.shape[1]):
        key = np.ascontiguousarray(X[:, column], dtype=np.float64).tobytes()
        if key not in seen:
            seen[key] = len(groups); groups.append([])
        groups[seen[key]].append(column)
    return groups


def direct_near_groups(X, tolerance):
    """Independent greedy all-candidate residual search; no cosine prefilter."""
    norms = np.linalg.norm(X, axis=0); assigned = np.zeros(X.shape[1], dtype=bool); groups = []
    for anchor in range(X.shape[1]):
        if assigned[anchor]:
            continue
        members = [anchor]; assigned[anchor] = True
        denominator = float(X[:, anchor] @ X[:, anchor])
        for column in range(anchor+1, X.shape[1]):
            if assigned[column]:
                continue
            scale = float((X[:, anchor] @ X[:, column])/denominator)
            if np.linalg.norm(X[:, column]-scale*X[:, anchor])/norms[column] <= tolerance:
                members.append(column); assigned[column] = True
        groups.append(members)
    return groups


def describe(X, partition):
    out = []
    for members in partition:
        anchor = members[0]; denominator = float(X[:, anchor] @ X[:, anchor])
        scales = [1. if j == anchor else float((X[:, anchor] @ X[:, j])/denominator) for j in members]
        residuals = [float(np.linalg.norm(X[:, j]-a*X[:, anchor])) for j, a in zip(members, scales)]
        out.append({"anchor_column_0based": anchor, "columns_0based": members, "member_scales_from_anchor": scales,
                    "residual_l2_norms": residuals,
                    "relative_l2_residuals_to_anchor": [r/np.linalg.norm(X[:, j]) for j, r in zip(members, residuals)]})
    return out


def aggregate(groups, weights):
    return np.array([sum(float(a)*float(weights[j]) for j, a in zip(group["columns_0based"], group["member_scales_from_anchor"]))
                     for group in groups], dtype=np.float64)


def mapping_values(X, y, groups, weights, lam):
    anchors = [group["anchor_column_0based"] for group in groups]; theta = aggregate(groups, weights)
    old_fit = X @ weights; mapped_fit = X[:, anchors] @ theta
    scaled_mass = cancellation = signed_scale = error_bound = representation_bound = 0.
    for index, group in enumerate(groups):
        for column, scale, residual in zip(group["columns_0based"], group["member_scales_from_anchor"], group["residual_l2_norms"]):
            mass = abs(float(weights[column]))
            scaled_mass += abs(scale)*mass
            signed_scale += (abs(scale)-1)*mass
            error_bound += abs(abs(scale)-1)*mass
            representation_bound += mass*residual
    old_l1, mapped_l1 = float(np.abs(weights).sum()), float(np.abs(theta).sum())
    cancellation = scaled_mass-mapped_l1
    old_residual, mapped_residual = y-old_fit, y-mapped_fit
    return theta, {"signal_or_fit_mapping_rmse": float(np.linalg.norm(old_fit-mapped_fit)/np.sqrt(len(X))),
                   "mapping_rmse_triangle_bound": representation_bound/np.sqrt(len(X)),
                   "old_l1_norm": old_l1, "mapped_l1_norm": mapped_l1, "l1_norm_change": mapped_l1-old_l1,
                   "l1_scale_change": signed_scale, "l1_cancellation_amount": cancellation, "l1_scale_error_bound": error_bound,
                   "l1_change_lower_bound": -error_bound-cancellation, "l1_change_upper_bound": error_bound-cancellation,
                   "absolute_l1_change_upper_bound": error_bound+cancellation,
                   "penalty_change": lam*(mapped_l1-old_l1), "penalty_change_lower_bound": lam*(-error_bound-cancellation),
                   "penalty_change_upper_bound": lam*(error_bound-cancellation),
                   "l1_accounting_residual": (mapped_l1-old_l1)-(signed_scale-cancellation),
                   "old_objective": float(np.dot(old_residual, old_residual)/(2*len(X))+lam*old_l1),
                   "mapped_objective": float(np.dot(mapped_residual, mapped_residual)/(2*len(X))+lam*mapped_l1)}


def support_score(coef, truth_mask, tau):
    selected = np.abs(coef) > tau
    tp = int(np.count_nonzero(selected & truth_mask)); fp = int(np.count_nonzero(selected & ~truth_mask))
    fn = int(np.count_nonzero(~selected & truth_mask)); denominator = 2*tp+fp+fn
    return {"tp": tp, "fp": fp, "fn": fn, "nnz": int(selected.sum()), "support_error": fp+fn,
            "f1": float(Fraction(2*tp, denominator)) if denominator else 0.}


def pair_scores(first, second, truth, taus):
    return {str(tau): {"sklearn": support_score(first, np.abs(truth)>tau, tau),
                       "skglm": support_score(second, np.abs(truth)>tau, tau),
                       "support_symmetric_difference_count": int(np.count_nonzero((np.abs(first)>tau) != (np.abs(second)>tau)))}
            for tau in taus}


def qualification(refs, first, second, settings, taus):
    reasons = []; by_reference = {}
    for method in ("sklearn", "skglm"):
        row = refs[method]; cert = row["certificate"]
        failures = []
        finite = all(np.isfinite(cert.get(key, np.nan)) for key in ("primal", "dual", "gap", "normalized_gap", "dual_feasibility"))
        if not finite:
            failures.append("nonfinite_objective_or_certificate")
        if not np.isfinite(cert.get("normalized_gap", np.nan)) or cert["normalized_gap"] > settings["normalized_gap_tol"]:
            failures.append("normalized_gap_above_tolerance")
        if not np.isfinite(cert.get("dual_feasibility", np.nan)) or cert["dual_feasibility"] > 1+1e-12:
            failures.append("dual_not_feasible")
        if not np.isfinite(row.get("elapsed_sec", np.nan)) or row["elapsed_sec"] > settings["time_limit"]:
            failures.append("observed_budget_exceeded")
        if row["status"] != "converged":
            failures.append("library_result_not_common_converged_status")
        by_reference[method] = {"eligible": not failures, "reasons": failures,
                                "normalized_gap": cert["normalized_gap"], "elapsed_sec": row["elapsed_sec"]}
        reasons.extend(method+":"+item for item in failures)
    pa, pb = (refs[method]["certificate"]["primal"] for method in ("sklearn", "skglm"))
    consensus_reasons = list(reasons)
    if np.max(np.abs(first-second)) > settings["coef_linf_agreement_tol"]:
        consensus_reasons.append("coefficient_linf_disagreement")
    if abs(pa-pb)/max(1., abs(pa), abs(pb)) > settings["normalized_objective_agreement_tol"]:
        consensus_reasons.append("normalized_objective_disagreement")
    for tau in taus:
        a, b = np.abs(first)>tau, np.abs(second)>tau
        if not np.array_equal(a, b):
            consensus_reasons.append(f"support_disagreement_tau={tau}")
        if not np.array_equal(np.where(a, np.sign(first), 0.), np.where(b, np.sign(second), 0.)):
            consensus_reasons.append(f"thresholded_sign_disagreement_tau={tau}")
    return not reasons, not consensus_reasons, {"accuracy": reasons, "consensus": consensus_reasons, "by_reference": by_reference}


def verify(config_path):
    started = perf_counter(); audit = Audit(); config = audit.read(config_path)
    base = ROOT / "results/uji_preprocessing_control_v1"; old_base = ROOT / config["baseline_path"]
    env = audit.read(base / "environment.json"); completion = audit.read(base / "completion.json")
    old_config = audit.read(ROOT / config["baseline_config_path"])
    audit.equal(config["references"], old_config["references"], "Frozen reference settings")
    payload_keys = ("base_numerical_signature", "base_provenance", "driver_sha256", "config_sha256", "group_helper_sha256", "frozen_input_hashes")
    payload = {key: env[key] for key in payload_keys}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    audit.equal(signature, env["numerical_signature"], "Control numerical signature")
    audit.equal(env["config_sha256"], audit.track(config_path), "Control config hash")
    audit.equal(env["driver_sha256"], audit.track(ROOT / "scripts/run_uji_preprocessing_control_v1.py"), "Control driver hash")
    audit.equal(env["group_helper_sha256"], audit.track(ROOT / "src/lasso_study/audit_data_v1.py"), "Grouping helper hash")
    for relative, value in env["frozen_input_hashes"].items():
        audit.equal(audit.track(ROOT / relative), value, "Old input unchanged "+relative)
    for relative, value in env["base_provenance"]["source_hashes"].items():
        audit.equal(audit.track(ROOT / relative), value, "Frozen helper source "+relative)
    for name, version in env["base_provenance"]["dependency_versions"].items():
        audit.equal(importlib.metadata.version(name), version, "Numerical dependency "+name)
    audit.equal(hashlib.sha256(json.dumps(env["base_provenance"], sort_keys=True).encode()).hexdigest(), env["base_numerical_signature"], "Base signature payload")
    for optional in ("protocol_lock.json", "warmup.json", "config.json", "setup.json"):
        if (base / optional).is_file():
            audit.read(base / optional)
    params_list = [dict(seed=seed, layout=layout, amplitude=amplitude) for seed, layout, amplitude in product(
        config["selected_seeds"], config["layouts"], config["amplitudes"])]
    targets = {f"uji_seed{p['seed']}_{p['layout']}_{p['amplitude']}": p for p in params_list}
    expected = {identifier+"__"+arm for identifier, arm in product(targets, config["arms"])}
    actual = {path.stem for path in (base / "controls").glob("*.json")}
    audit.check(actual == expected and len(targets) == config["expected_old_targets"], "Registered 12-target/48-control grid")
    audit.equal(completion["recorded_controls"], len(expected), "Completed controls")
    source = make_uji(seed=0)
    import pandas as pd
    raw_path = ROOT / config["raw_uji_csv"]; raw_hash = audit.track(raw_path)
    raw_frame = pd.read_csv(raw_path, usecols=[f"WAP{j:03}" for j in range(1, 521)])
    partitions = {}; rows_out = []; mapping_maxima = defaultdict(float); new_calls = 0; statuses = Counter()
    reserved_calls = 0; unknown_attempts = 0; actual_library_attempts = 0; new_reference_ids = set()
    preparation_paths = set(); reconstruction_times = {}; mapping_sec = write_sec = 0.
    for identifier, params in targets.items():
        ds_path = old_base / "datasets" / (identifier+".json"); old_ds = audit.read(ds_path)
        rp = old_base / "references" / (identifier+".json"); old_ref = audit.read(rp)
        cp = old_base / "reference_coefficients" / (identifier+".npz"); audit.equal(audit.track(cp), old_ref["coefficient_sha256"], identifier+": old coefficient SHA")
        with np.load(cp, allow_pickle=False) as arrays:
            first, second, truth = (arrays[key].copy() for key in ("ref_sklearn", "ref_skglm", "w_true"))
        data = make_real_design(old_config, params["seed"], params["layout"], params["amplitude"], source=source)
        X, y = data["X"], data["y"]; lam = old_ds["lambda"]
        audit.equal(data["metadata"]["hashes"], old_ds["hashes"], identifier+": old X/y generation hashes")
        audit.check(np.array_equal(truth, data["w_true"]), identifier+": old truth")
        audit.equal(data["metadata"]["lambda"], lam, identifier+": regenerated lambda", atol=0.)
        audit.equal(old_ref["lambda"], lam, identifier+": old reference lambda", atol=0.)
        audit.equal(old_ds["source_csv_sha256"], raw_hash, identifier+": raw input SHA")
        old_data_hash = hashlib.sha256(json.dumps(old_ds["hashes"], sort_keys=True).encode()).hexdigest()
        audit.equal(old_ref["dataset_hash"], old_data_hash, identifier+": old reference data hash")
        seed = params["seed"]
        if seed not in partitions:
            names = [f"WAP{j+1:03}" for j in old_ds["retained_original_wap_columns_0based"]]
            raw = raw_frame.loc[old_ds["source_original_csv_row_indices"], names].to_numpy(dtype=np.float64)
            partitions[seed] = {"raw_hash": array_sha256(raw),
                "baseline": describe(X, [[j] for j in range(X.shape[1])]),
                "final_float64_exact": describe(X, equal_column_groups(X)),
                "A_raw_exact": describe(X, equal_column_groups(raw)),
                "B_near": describe(X, direct_near_groups(X, config["near_relative_residual_tolerance"]))}
        for arm in config["arms"]:
            control_id = identifier+"__"+arm; record = audit.read(base / "controls" / (control_id+".json"))
            groups = partitions[seed][arm]; anchors = [g["anchor_column_0based"] for g in groups]
            audit.equal(record["groups"], groups, control_id+": independent raw/exact/near partition and residuals", atol=2e-14)
            audit.equal(record["retained_old_columns"], anchors, control_id+": retained anchors")
            audit.equal(record["deleted_old_columns"], sorted(set(range(X.shape[1]))-set(anchors)), control_id+": deleted identities")
            audit.equal(record["old_y_sha256"], array_sha256(y), control_id+": response unchanged")
            audit.equal(record["old_lambda"], lam, control_id+": lambda unchanged", atol=0.)
            audit.equal(record["raw_RSSI_subsample_hash"], partitions[seed]["raw_hash"], control_id+": raw row/column selection")
            audit.equal(record["numerical_signature"], signature, control_id+": signature")
            audit.equal(record["old_reference_sha256"], sha(rp), control_id+": old reference immutable")
            audit.equal(record["old_reference_coefficients_sha256"], sha(cp), control_id+": old coefficient immutable")
            audit.equal(record["old_dataset_sha256"], sha(ds_path), control_id+": old metadata immutable")
            preparation = base / record["shared_group_preparation_path"]
            if preparation not in preparation_paths:
                prep = audit.read(preparation); preparation_paths.add(preparation)
                audit.equal(prep["X_sha256"], array_sha256(X), control_id+": shared X-only preparation")
                audit.equal(prep["numerical_signature"], signature, control_id+": shared preparation source")
            reconstruction_times.setdefault(identifier, record["old_dataset_reconstruction_sec"])
            audit.equal(record["old_dataset_reconstruction_sec"], reconstruction_times[identifier], control_id+": shared reconstruction timing", atol=0.)
            mapping_sec += record["mapping_diagnostics_sec"]; write_sec += record["mapping_artifact_write_sec"]
            reserved_calls += record["public_reference_call_slots_reserved"]
            unknown_attempts += record["new_public_reference_calls"] is None
            if record["new_actual_library_fit_attempts"] is not None:
                actual_library_attempts += record["new_actual_library_fit_attempts"]
            mapped = {}
            for role, weights in (("truth", truth), ("old_sklearn", first), ("old_skglm", second)):
                theta, diag = mapping_values(X, y, groups, weights, lam); mapped[role] = theta
                audit.equal(record["mapping_diagnostics"][role], diag, control_id+": mapping/L1 "+role)
                audit.check(abs(diag["l1_accounting_residual"]) <= 5e-12, control_id+": L1 accounting identity "+role)
                audit.check(diag["l1_change_lower_bound"]-5e-12 <= diag["l1_norm_change"] <= diag["l1_change_upper_bound"]+5e-12,
                            control_id+": signed L1 bounds "+role)
                audit.check(abs(diag["l1_norm_change"]) <= diag["absolute_l1_change_upper_bound"]+5e-12, control_id+": absolute L1 bound "+role)
                audit.check(diag["signal_or_fit_mapping_rmse"] <= diag["mapping_rmse_triangle_bound"]+5e-12, control_id+": representation triangle bound "+role)
                for metric in ("signal_or_fit_mapping_rmse", "mapping_rmse_triangle_bound", "absolute_l1_change_upper_bound"):
                    saved_metric = record["mapping_diagnostics"][role][metric]
                    mapping_maxima[arm+"/"+role+"/"+metric] = max(mapping_maxima[arm+"/"+role+"/"+metric], saved_metric)
            mp = base / "mappings" / (control_id+".npz")
            audit.equal(audit.track(mp), record["mapping_artifact_sha256"], control_id+": mapping NPZ SHA")
            with np.load(mp, allow_pickle=False) as arrays:
                canonical_mapped_truth = arrays["mapped_w_true"].copy()
                expected_arrays = {"old_w_true": truth, "mapped_w_true": mapped["truth"], "old_ref_sklearn": first,
                                   "old_ref_skglm": second, "mapped_old_ref_sklearn": mapped["old_sklearn"],
                                   "mapped_old_ref_skglm": mapped["old_skglm"], "retained_old_columns": np.asarray(anchors, dtype=np.int64)}
                for key, value in expected_arrays.items():
                    audit.check(arrays[key].shape == value.shape and np.allclose(arrays[key], value, rtol=1e-13, atol=2e-14), control_id+": NPZ "+key)
            audit.equal(record["original_variable_scores"], pair_scores(first, second, truth, config["support_thresholds"]), control_id+": original coordinate scores")
            audit.equal(record["mapped_old_output_scores"], pair_scores(mapped["old_sklearn"], mapped["old_skglm"], mapped["truth"], config["support_thresholds"]), control_id+": mapped old outputs")
            reduced_X = np.asfortranarray(X[:, anchors]); identity = len(anchors) == X.shape[1] and np.array_equal(reduced_X, X) and np.array_equal(mapped["truth"], truth)
            fresh_hashes = {"X": array_sha256(reduced_X), "y": array_sha256(y), "w_true": array_sha256(canonical_mapped_truth)}
            # Mapping sums can differ by a few ulps between independent Python
            # accumulation and the driver's dot. Canonical stored truth is
            # hashed only after the independent numerical mapping check above.
            reduced_hash = hashlib.sha256(json.dumps(fresh_hashes, sort_keys=True).encode()).hexdigest()
            audit.equal(record["reduced_input_hashes"], fresh_hashes, control_id+": retained-input hashes")
            audit.equal(record["reduced_dataset_hash"], reduced_hash, control_id+": retained-input combined hash")
            status = record["status"]; statuses[status] += 1
            ref, a, b = None, None, None
            if record["reference_origin"] == "old_frozen_reference":
                audit.check(identity and record["new_public_reference_calls"] == 0, control_id+": reuse only identical problem")
                ref, a, b = old_ref, first, second
            elif record["reference_origin"] == "new_reduced_problem_fit":
                audit.check(not identity and arm in {"A_raw_exact", "B_near"}, control_id+": only changed A/B may fit")
                ref_path = base / record["reduced_reference_path"]; ref = audit.read(ref_path)
                new_reference_ids.add(control_id)
                audit.equal(record["reduced_reference_sha256"], sha(ref_path), control_id+": reduced reference SHA")
                npz = base / "reference_coefficients" / (control_id+".npz")
                audit.equal(audit.track(npz), ref["coefficient_sha256"], control_id+": new reference NPZ hash")
                with np.load(npz, allow_pickle=False) as arrays:
                    a, b = arrays["ref_sklearn"].copy(), arrays["ref_skglm"].copy()
                    audit.check(np.array_equal(arrays["w_true"], canonical_mapped_truth), control_id+": new truth coordinates")
                audit.equal(ref["dataset_hash"], reduced_hash, control_id+": reduced data hash")
                audit.equal(ref["lambda"], lam, control_id+": new reference lambda", atol=0.)
                audit.equal(ref["numerical_signature"], signature, control_id+": new reference source")
                audit.check(record["new_public_reference_calls"] == 2, control_id+": public call accounting")
                new_calls += 2
            else:
                audit.check(status in {"budget_not_run", "protocol_review_required", "incomplete_reference_attempt_no_retry", "reference_error"}
                            and record["new_public_reference_calls"] in {0, None},
                            control_id+": unresolved arm must remain explicit")
            eligible = accepted = None
            scores = {}
            if ref is not None:
                eligible, accepted, reasons = qualification(ref, a, b, config["references"], config["support_thresholds"])
                audit.equal(ref["reference_accepted"], accepted, control_id+": independently checked consensus")
                audit.equal(record["reference_accepted"], accepted, control_id+": arm consensus")
                audit.equal(record["reference_consensus"], accepted, control_id+": arm consensus alias")
                audit.equal(record["qualification_failure_reasons"], reasons, control_id+": detailed qualification reasons")
                if "accuracy_eligible" in record:
                    audit.equal(record["accuracy_eligible"], eligible, control_id+": arm accuracy eligibility")
                for method, coef in (("sklearn", a), ("skglm", b)):
                    cert = certificate(reduced_X, y, coef, lam)
                    audit.equal(ref[method]["certificate"], {key:cert[key] for key in ("primal", "dual", "gap", "raw_gap", "normalized_gap", "dual_feasibility", "kkt_inf")},
                                control_id+": recomputed certificate "+method, atol=5e-12)
                scores = pair_scores(a, b, mapped["truth"], config["support_thresholds"])
                audit.equal(record["reduced_coordinate_scores"], scores, control_id+": reduced coordinate scores")
            sizes = np.array([len(g["columns_0based"]) for g in groups]); counts = np.array([np.count_nonzero(truth[g["columns_0based"]]) for g in groups])
            for tau in config["support_thresholds"]:
                any_truth = counts > 0; net_truth = np.abs(mapped["truth"]) > tau
                flags = {"groups_with_any_true_variable": int(any_truth.sum()), "net_true_nonzero_groups": int(net_truth.sum()),
                         "original_true_variable_count": int(np.count_nonzero(truth)),
                         "groups_containing_multiple_true_variables": int(np.count_nonzero(counts > 1)),
                         "true_variable_identities_merged_count": int(np.count_nonzero(truth)-any_truth.sum()),
                         "deleted_columns_with_true_nonzero_coefficient": int(np.count_nonzero(truth[record["deleted_old_columns"]])),
                         "groups_with_true_variables_but_net_below_tau": int(np.count_nonzero(any_truth & ~net_truth))}
                audit.equal(record["mapped_truth_support_diagnostics"][str(tau)], flags, control_id+": net/contains-any truth")
                pair = scores.get(str(tau)); delta = abs(pair["sklearn"]["f1"]-pair["skglm"]["f1"]) if pair else None
                rows_out.append({"control_id": control_id, "old_target_id": identifier, **params, "arm": arm, "tau": tau,
                    "status": status, "accuracy_eligible": eligible, "consensus_accepted": accepted,
                    "old_p": X.shape[1], "new_p": len(anchors), "singletons": int(np.count_nonzero(sizes == 1)),
                    "nontrivial_groups": int(np.count_nonzero(sizes > 1)), "maximum_group_size": int(sizes.max()),
                    "lost_coordinate_identities": X.shape[1]-len(anchors), "true_variables_in_nontrivial_groups": int(counts[sizes > 1].sum()),
                    "groups_with_multiple_true_variables": int(np.count_nonzero(counts > 1)),
                    "deleted_columns_with_true_nonzero_coefficient": flags["deleted_columns_with_true_nonzero_coefficient"],
                    "extra_true_variables_merged": flags["true_variable_identities_merged_count"],
                    "net_cancellation_groups": flags["groups_with_true_variables_but_net_below_tau"],
                    "reduced_scores": pair, "f1_difference_if_accuracy_eligible": delta if eligible else None,
                    "contains_any_scores": {"sklearn": support_score(a, any_truth, tau), "skglm": support_score(b, any_truth, tau)} if pair else None})
    audit.equal(completion["stored_known_new_public_reference_calls"], new_calls, "Stored known new public reference calls")
    audit.equal(completion["reserved_public_reference_call_slots"], reserved_calls, "Stored reserved public reference calls")
    audit.equal(completion["unknown_partial_reference_attempt_controls"], unknown_attempts, "Unknown partial attempt controls")
    audit.check(new_calls <= reserved_calls <= config["maximum_new_public_reference_calls"], "At-most-48 reserved public reference calls")
    audit.check({p.stem for p in (base / "references").glob("*.json")} == new_reference_ids
                and {p.stem for p in (base / "reference_coefficients").glob("*.npz")} == new_reference_ids, "Exact new reference artifact grid")
    audit.check({p.stem for p in (base / "mappings").glob("*.npz")} == expected, "Exact mapping artifact grid")
    attempts = [audit.read(path) for path in sorted((base / "reference_attempts").glob("*.json"))]
    audit.equal(sum(row["reserved_public_reference_calls"] for row in attempts), reserved_calls, "Durable attempt reservation total")
    audit.check({row["control_id"] for row in attempts} == {identifier for identifier in expected if (base / "reference_attempts" / (identifier+".json")).is_file()}, "Attempt identifiers")
    for attempt in attempts:
        identifier = attempt["control_id"]
        audit.equal(attempt["numerical_signature"], signature, identifier+": attempt signature")
        if identifier in new_reference_ids:
            audit.equal(attempt["state"], "completed", identifier+": completed durable attempt")
            audit.equal(attempt["reference_sha256"], sha(base / "references" / (identifier+".json")), identifier+": attempt result SHA")
    invocations = [audit.read(path) for path in sorted((base / "invocations").glob("*.json"))]
    relevant = [r for r in invocations if r["numerical_signature"] == signature]
    if relevant:
        audit.equal(sum(r["new_public_reference_call_slots_reserved"] for r in relevant), reserved_calls, "Invocation public-call reservation ledger")
        audit.equal(sum(r["new_actual_library_fit_attempts"] for r in relevant), actual_library_attempts, "Actual library-fit attempt ledger")
        audit.equal(completion["new_actual_library_fit_attempts_this_invocation"], relevant[-1]["new_actual_library_fit_attempts"], "Final invocation library attempts")
    audit.equal(completion["accuracy_eligible_controls"], sum(r["accuracy_eligible"] is True for r in rows_out if r["tau"] == config["support_thresholds"][0]), "Accuracy coverage total")
    audit.equal(completion["consensus_controls"], sum(r["consensus_accepted"] is True for r in rows_out if r["tau"] == config["support_thresholds"][0]), "Consensus coverage total")
    summaries = []
    for arm, tau in product(config["arms"], config["support_thresholds"]):
        group = [r for r in rows_out if r["arm"] == arm and r["tau"] == tau]
        eligible = [r for r in group if r["accuracy_eligible"] is True]
        summaries.append({"arm": arm, "tau": tau, "planned_targets": len(targets), "recorded_targets": len(group),
            "accuracy_eligible_targets": len(eligible), "consensus_accepted_targets": sum(r["consensus_accepted"] is True for r in group),
            "unrun_or_unqualified_targets": len(targets)-len(eligible),
            "f1_different_eligible_targets": sum(r["f1_difference_if_accuracy_eligible"] > 1e-12 for r in eligible),
            "maximum_eligible_f1_difference": max((r["f1_difference_if_accuracy_eligible"] for r in eligible), default=None),
            "status_counts": dict(Counter(r["status"] for r in group)),
            "old_p_range": [min(r["old_p"] for r in group), max(r["old_p"] for r in group)],
            "new_p_range": [min(r["new_p"] for r in group), max(r["new_p"] for r in group)],
            "lost_coordinate_identities_range": [min(r["lost_coordinate_identities"] for r in group), max(r["lost_coordinate_identities"] for r in group)],
            "true_variables_in_nontrivial_groups_max": max(r["true_variables_in_nontrivial_groups"] for r in group),
            "deleted_true_nonzero_coordinates_max": max(r["deleted_columns_with_true_nonzero_coefficient"] for r in group),
            "targets_with_deleted_true_nonzero_coordinates": sum(r["deleted_columns_with_true_nonzero_coefficient"] > 0 for r in group),
            "extra_true_variables_merged_max": max(r["extra_true_variables_merged"] for r in group),
            "net_cancellation_groups_max": max(r["net_cancellation_groups"] for r in group)})
    for relative, value in list(audit.inputs.items()):
        audit.check(sha(ROOT/relative) == value, "Evidence unchanged at completion: "+relative)
    return {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "all_passed": not audit.errors,
            "verifier_script_sha256": sha(Path(__file__)), "numerical_signature": signature, "config_sha256": sha(config_path),
            "checks": audit.checks, "issues": audit.errors, "warnings": audit.warnings, "input_provenance": audit.inputs,
            "n_old_targets_reconstructed": len(targets), "n_controls_checked": len(expected), "new_public_reference_calls": new_calls,
            "reserved_public_reference_call_slots": reserved_calls, "unknown_partial_reference_attempt_controls": unknown_attempts,
            "actual_library_fit_attempts": actual_library_attempts,
            "arm_summary": summaries, "control_score_rows": rows_out, "mapping_metric_maxima": dict(mapping_maxima),
            "recorded_invocation_wall_sec": sum(r["wall_sec"] for r in relevant), "verification_wall_sec": perf_counter()-started,
            "separately_recorded_components_sec": {"old_design_reconstruction": sum(reconstruction_times.values()),
                "X_only_group_preparation": sum(json.loads(path.read_text())["preparation_sec"] for path in preparation_paths),
                "mapping_diagnostics": mapping_sec, "mapping_npz_write": write_sec},
            "boundaries": ["Post-hoc selected old targets; no independent confirmation, no optimizer calls.",
                           "Direct exhaustive greedy anchor search independently checks near groups for these small UJI designs.",
                           "L1/triangle accounting is checked with explicit float64 residual tolerance, not interval-arithmetic guarantees.",
                           "Raw-variable and reduced net-coordinate scores are different tasks; contains-any truth is separate.",
                           "Budget is a returned-call soft budget; wall time is not a hard interruption certificate.",
                           "No claim of exact problem equivalence, correct unique support, mathematical multi-solution proof or novelty."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/uji_preprocessing_control_v1.json")
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=1):
            result = verify(args.config.resolve())
    except Exception:
        result = {"all_passed": False, "verifier_script_sha256": sha(Path(__file__)), "issues": [traceback.format_exc()]}
    output = ROOT / "results/uji_preprocessing_control_v1/verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key:result.get(key) for key in ("all_passed", "checks", "n_old_targets_reconstructed", "n_controls_checked", "new_public_reference_calls", "arm_summary", "issues")}, ensure_ascii=False, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
