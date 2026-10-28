#!/usr/bin/env python3
"""Bounded post-hoc UJI preprocessing control; frozen old artifacts are read-only.

--check-only hashes the registered inputs without data reconstruction or fitting.
Normal execution fits only changed A/B problems, at most 48 public calls. Exact
float64 deduplication is a diagnostic: unexpected deletions require review.
"""
from __future__ import annotations

import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter, time_ns
import traceback
from zoneinfo import ZoneInfo

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"scripts"), str(ROOT/"src"), str(ROOT/".modern_deps")]
from run_workshop import dump, sha, fit_references, source_signature as base_signature, support_counts
from lasso_study.data import array_sha256
from lasso_study.data_v2 import make_uji
from lasso_study.workshop_data import make_real_design
from lasso_study.audit_data_v1 import near_proportional_groups_chunked
from lasso_study.modern_solver import warmup_modern


def target_parameters(config):
    rows = [dict(seed=seed, layout=layout, amplitude=amplitude)
            for seed, layout, amplitude in product(config["selected_seeds"], config["layouts"], config["amplitudes"])]
    if len(rows) != config["expected_old_targets"] or len({tuple(row.values()) for row in rows}) != len(rows):
        raise ValueError("Post-hoc control grid differs from the registered 12 old targets")
    return rows


def old_target_id(params):
    return f"uji_seed{params['seed']}_{params['layout']}_{params['amplitude']}"


def control_signature(config_path):
    config = json.loads(Path(config_path).read_text())
    base, evidence = base_signature()
    baseline = ROOT/config["baseline_path"]
    raw_path = ROOT/config["raw_uji_csv"]
    raw_sha = sha(raw_path)
    files = [ROOT/config["protocol_path"], ROOT/config["baseline_config_path"], baseline/"environment.json", raw_path]
    for params in target_parameters(config):
        identifier = old_target_id(params)
        dataset = json.loads((baseline/"datasets"/(identifier+".json")).read_text())
        if dataset["source_csv_sha256"] != raw_sha:
            raise RuntimeError("Actual raw RSSI hash differs from the frozen old dataset")
        files += [baseline/"datasets"/(identifier+".json"), baseline/"references"/(identifier+".json"),
                  baseline/"reference_coefficients"/(identifier+".npz")]
    inputs = {str(path.relative_to(ROOT)): sha(path) for path in files}
    payload = {"base_numerical_signature": base, "base_provenance": evidence,
               "driver_sha256": sha(Path(__file__)), "config_sha256": sha(config_path),
               "group_helper_sha256": sha(ROOT/"src/lasso_study/audit_data_v1.py"), "frozen_input_hashes": inputs}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def exact_groups(matrix):
    """Byte-exact same columns only; earliest retained column anchors the group."""
    seen = {}; groups = []
    for j in range(matrix.shape[1]):
        representation = np.ascontiguousarray(matrix[:, j], dtype=np.float64).tobytes()
        if representation not in seen:
            seen[representation] = len(groups); groups.append([j])
        else:
            groups[seen[representation]].append(j)
    return groups


def describe_groups(X, member_groups):
    result = []
    for members in member_groups:
        anchor = min(members); column = X[:, anchor]
        denominator = float(column@column)
        scales = []; residuals = []
        for member in members:
            scale = 1. if member == anchor else float(column@X[:, member]/denominator)
            scales.append(scale)
            residuals.append(float(np.linalg.norm(X[:, member]-scale*column)))
        result.append({"anchor_column_0based": anchor, "columns_0based": list(members),
                       "member_scales_from_anchor": scales, "residual_l2_norms": residuals,
                       "relative_l2_residuals_to_anchor": [residual/np.linalg.norm(X[:, member]) for residual,member in zip(residuals,members)]})
    if sorted(j for group in result for j in group["columns_0based"]) != list(range(X.shape[1])):
        raise AssertionError("Column groups must partition every old column exactly once")
    return result


def aggregate(groups, coefficients):
    return np.asarray([np.asarray(group["member_scales_from_anchor"])@coefficients[group["columns_0based"]]
                       for group in groups], dtype=np.float64)


def mapping_diagnostics(X, y, groups, coefficients, lam):
    anchors = [group["anchor_column_0based"] for group in groups]
    theta = aggregate(groups, coefficients)
    old_fit = X@coefficients; mapped_fit = X[:, anchors]@theta
    error = old_fit-mapped_fit
    scale_change = cancellation = scale_bound = representation_bound = 0.
    for i, group in enumerate(groups):
        weights = coefficients[group["columns_0based"]]
        scales = np.asarray(group["member_scales_from_anchor"])
        weighted_abs = np.abs(scales*weights).sum()
        cancellation += max(0., float(weighted_abs-abs(theta[i])))
        scale_change += float(((np.abs(scales)-1)*np.abs(weights)).sum())
        scale_bound += float((np.abs(np.abs(scales)-1)*np.abs(weights)).sum())
        representation_bound += float(np.abs(weights)@np.asarray(group["residual_l2_norms"]))
    old_l1, mapped_l1 = float(np.abs(coefficients).sum()), float(np.abs(theta).sum())
    n = len(X); old_residual = y-old_fit; mapped_residual = y-mapped_fit
    return {"signal_or_fit_mapping_rmse": float(np.linalg.norm(error)/np.sqrt(n)),
            "mapping_rmse_triangle_bound": representation_bound/np.sqrt(n),
            "old_l1_norm": old_l1, "mapped_l1_norm": mapped_l1,
            "l1_norm_change": mapped_l1-old_l1, "l1_scale_change": scale_change,
            "l1_cancellation_amount": cancellation, "l1_scale_error_bound": scale_bound,
            "l1_change_lower_bound": -scale_bound-cancellation, "l1_change_upper_bound": scale_bound-cancellation,
            "absolute_l1_change_upper_bound": scale_bound+cancellation,
            "penalty_change": lam*(mapped_l1-old_l1),
            "penalty_change_lower_bound": lam*(-scale_bound-cancellation),
            "penalty_change_upper_bound": lam*(scale_bound-cancellation),
            "l1_accounting_residual": (mapped_l1-old_l1)-(scale_change-cancellation),
            "old_objective": float(old_residual@old_residual/(2*n)+lam*old_l1),
            "mapped_objective": float(mapped_residual@mapped_residual/(2*n)+lam*mapped_l1)}


def scores(first, second, truth, taus):
    output = {}
    for tau in taus:
        masks = [np.abs(coef)>tau for coef in (first,second)]
        output[str(tau)] = {"sklearn": support_counts(masks[0], np.abs(truth)>tau),
                            "skglm": support_counts(masks[1], np.abs(truth)>tau),
                            "support_symmetric_difference_count": int(np.sum(masks[0]!=masks[1]))}
    return output


def truth_identity_diagnostics(groups, truth, taus):
    """Identity loss and net cancellation are different from numerical accuracy."""
    mapped = aggregate(groups, truth)
    original_true_count = int(np.count_nonzero(truth))
    group_true_counts = [int(np.count_nonzero(truth[group["columns_0based"]])) for group in groups]
    return {str(tau): {
        "original_true_variable_count": original_true_count,
        "groups_with_any_true_variable": int(sum(count > 0 for count in group_true_counts)),
        "groups_containing_multiple_true_variables": int(sum(count > 1 for count in group_true_counts)),
        "true_variable_identities_merged_count": int(sum(max(0, count-1) for count in group_true_counts)),
        "deleted_columns_with_true_nonzero_coefficient": int(sum(
            np.count_nonzero(truth[[j for j in group["columns_0based"] if j != group["anchor_column_0based"]]])
            for group in groups)),
        "net_true_nonzero_groups": int(np.count_nonzero(np.abs(mapped) > tau)),
        "groups_with_true_variables_but_net_below_tau": int(sum(
            count > 0 and abs(mapped[i]) <= tau for i, count in enumerate(group_true_counts)))
    } for tau in taus}


def qualification(refs, config):
    """Accuracy/budget eligibility is recorded independently of pair consensus."""
    settings = config["references"]
    by_reference, reasons = {}, []
    for name in ("sklearn", "skglm"):
        row = refs[name]; cert = row["certificate"]; failures = []
        if not all(np.isfinite(cert.get(key, np.nan)) for key in ("primal", "dual", "gap", "normalized_gap", "dual_feasibility")):
            failures.append("nonfinite_objective_or_certificate")
        if not np.isfinite(cert.get("normalized_gap", np.nan)) or cert["normalized_gap"] > settings["normalized_gap_tol"]:
            failures.append("normalized_gap_above_tolerance")
        if not np.isfinite(cert.get("dual_feasibility", np.nan)) or cert["dual_feasibility"] > 1 + 1e-12:
            failures.append("dual_not_feasible")
        if not np.isfinite(row.get("elapsed_sec", np.nan)) or row["elapsed_sec"] > settings["time_limit"]:
            failures.append("observed_budget_exceeded")
        if row.get("status") != "converged":
            failures.append("library_result_not_common_converged_status")
        by_reference[name] = {"eligible": not failures, "reasons": failures,
                              "normalized_gap": cert.get("normalized_gap"), "elapsed_sec": row.get("elapsed_sec")}
        reasons.extend(name+":"+reason for reason in failures)
    consensus = list(reasons)
    for key, tolerance, label in (("coefficient_linf_difference", "coef_linf_agreement_tol", "coefficient_linf_disagreement"),
                                  ("normalized_primal_difference", "normalized_objective_agreement_tol", "normalized_objective_disagreement")):
        if not np.isfinite(refs[key]) or refs[key] > settings[tolerance]:
            consensus.append(label)
    for tau in config["support_thresholds"]:
        row = refs["by_support_threshold"][str(tau)]
        if not row["same_support"]:
            consensus.append(f"support_disagreement_tau={tau}")
        if not row["same_thresholded_signs"]:
            consensus.append(f"thresholded_sign_disagreement_tau={tau}")
    accepted = not consensus
    if bool(refs["reference_accepted"]) != accepted:
        raise AssertionError("Frozen helper consensus flag differs from explicit qualification")
    return {"accuracy_eligible": not reasons, "reference_accepted": accepted, "reference_consensus": accepted,
            "qualification_failure_reasons": {"accuracy": reasons, "consensus": consensus, "by_reference": by_reference}}


def unevaluated_qualification(reason):
    return {"accuracy_eligible": None, "reference_accepted": None, "reference_consensus": None,
            "qualification_failure_reasons": {"accuracy": [reason], "consensus": [reason], "by_reference": {}}}


def validate_cached_control(saved, output, signature, config_hash):
    if saved["numerical_signature"] != signature or saved["config_sha256"] != config_hash:
        raise RuntimeError("Cached control signature/configuration differs")
    artifact = output/"mappings"/(saved["control_id"]+".npz")
    if sha(artifact) != saved["mapping_artifact_sha256"]:
        raise RuntimeError("Cached mapping artifact hash differs")
    if saved.get("reduced_reference_path"):
        ref_path = output/saved["reduced_reference_path"]
        if sha(ref_path) != saved["reduced_reference_sha256"]:
            raise RuntimeError("Cached reduced reference record hash differs")
        ref = json.loads(ref_path.read_text())
        if (ref["numerical_signature"] != signature or ref["config_sha256"] != config_hash
                or ref["dataset_hash"] != saved["reduced_dataset_hash"]
                or sha(output/ref["coefficient_path"]) != ref["coefficient_sha256"]):
            raise RuntimeError("Cached reduced reference inputs/coefficient artifact differ")


def main():
    invocation_start = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/uji_preprocessing_control_v1.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    targets = target_parameters(config); signature, provenance = control_signature(args.config)
    if args.check_only:
        print(json.dumps({"old_targets":len(targets),"arms":config["arms"],"maximum_new_public_reference_calls":48,
                          "numerical_signature":signature,"config_sha256":provenance["config_sha256"],
                          "new_fits":0,"data_generated":False,"result_files_written":0},indent=2)); return
    output = ROOT/"results/uji_preprocessing_control_v1"; output.mkdir(parents=True,exist_ok=True)
    baseline = ROOT/config["baseline_path"]; old_config = json.loads((ROOT/config["baseline_config_path"]).read_text())
    if config["references"] != old_config["references"]:
        raise ValueError("Mechanism control must retain the frozen reference settings")
    env_path = output/"environment.json"
    if env_path.exists() and json.loads(env_path.read_text())["numerical_signature"] != signature:
        raise RuntimeError("Existing control uses different frozen inputs")
    started = invocation_start; records=[]; new_calls=0; actual_attempts=0; invocation=str(time_ns())
    ledger = output/"invocations"
    old_invocations = [json.loads(path.read_text()) for path in ledger.glob("*.json")]
    if any(row.get("state") != "finished" for row in old_invocations):
        raise RuntimeError("Unfinished invocation requires review; automatic retry could hide consumed budget")
    previous_wall = sum(row["wall_sec"] for row in old_invocations if row["numerical_signature"] == signature)
    attempt_dir = output/"reference_attempts"
    previous_calls = sum(row["reserved_public_reference_calls"] for path in attempt_dir.glob("*.json")
                         if (row:=json.loads(path.read_text()))["numerical_signature"] == signature)
    dump(ledger/(invocation+".json"), {"numerical_signature":signature,"state":"running","wall_sec":0.,
                                      "new_public_reference_call_slots_reserved":0})
    try:
        with threadpool_limits(limits=1):
            warmup = warmup_modern()
            if not env_path.exists():
                dump(env_path,dict(provenance,numerical_signature=signature,started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),threads=1))
                dump(output/"config.json",config); dump(output/"warmup.json",warmup)
                (output/"protocol_snapshot.md").write_bytes((ROOT/config["protocol_path"]).read_bytes())
            import pandas as pd
            raw_path=ROOT/config["raw_uji_csv"]
            raw_csv_sha=sha(raw_path)
            preparation=perf_counter()
            raw_frame=pd.read_csv(raw_path,usecols=[f"WAP{j:03}" for j in range(1,521)])
            raw_load_sec=perf_counter()-preparation
            preparation=perf_counter();source=make_uji(seed=0);source_load_sec=perf_counter()-preparation
            mapping_cache={}
            dump(output/"setup.json", {"raw_RSSI_load_sec":raw_load_sec,"frozen_UJI_source_load_sec":source_load_sec,
                                       "warmup_path":"warmup.json","raw_csv_sha256":raw_csv_sha})
            for params in targets:
                identifier=old_target_id(params)
                old_ds=json.loads((baseline/"datasets"/(identifier+".json")).read_text())
                old_refs=json.loads((baseline/"references"/(identifier+".json")).read_text())
                if (old_refs["dataset_hash"] != old_ds["dataset_hash"]
                        or old_refs["numerical_signature"] != old_ds["numerical_signature"]
                        or old_refs["config_sha256"] != old_ds["config_sha256"]
                        or old_ds["config_sha256"] != sha(ROOT/config["baseline_config_path"])
                        or old_ds["numerical_signature"] != json.loads((baseline/"environment.json").read_text())["numerical_signature"]):
                    raise RuntimeError("Baseline reference/dataset/configuration provenance differs")
                cp=baseline/"reference_coefficients"/(identifier+".npz")
                if sha(cp)!=old_refs["coefficient_sha256"]: raise RuntimeError("Baseline coefficient hash mismatch")
                with np.load(cp,allow_pickle=False) as arrays:
                    first,second,truth=(arrays[name].copy() for name in ("ref_sklearn","ref_skglm","w_true"))
                reconstruction_start=perf_counter()
                data=make_real_design(old_config,params["seed"],params["layout"],params["amplitude"],source=source)
                reconstruction_sec=perf_counter()-reconstruction_start
                X,y=data["X"],data["y"]; lam=old_ds["lambda"]
                if (data["metadata"]["hashes"]!=old_ds["hashes"] or not np.array_equal(truth,data["w_true"])
                        or data["metadata"]["lambda"] != lam or old_refs["lambda"] != lam):
                    raise RuntimeError("Baseline reconstruction differs")
                if old_ds["source_csv_sha256"]!=raw_csv_sha: raise RuntimeError("Raw RSSI source SHA mismatch")
                seed=params["seed"]
                if seed not in mapping_cache:
                    prep_start=perf_counter()
                    rows=old_ds["source_original_csv_row_indices"]
                    names=[f"WAP{j+1:03}" for j in old_ds["retained_original_wap_columns_0based"]]
                    raw=raw_frame.loc[rows,names].to_numpy(dtype=np.float64)
                    raw_groups=describe_groups(X,exact_groups(raw))
                    float_groups=describe_groups(X,exact_groups(X))
                    grouped=near_proportional_groups_chunked(X,tolerances=(config["near_relative_residual_tolerance"],),
                                                            block_size=config["group_scan_block_size"])
                    near_members=[group["columns_0based"] for group in grouped["groups_by_tolerance"][str(config["near_relative_residual_tolerance"])]]
                    near_groups=describe_groups(X,near_members)
                    mapping_cache[seed]={"baseline":describe_groups(X,[[j] for j in range(X.shape[1])]),
                                         "final_float64_exact":float_groups,"A_raw_exact":raw_groups,"B_near":near_groups,
                                         "mapping_preparation_sec":perf_counter()-prep_start,"raw_RSSI_subsample_hash":array_sha256(raw),
                                         "near_grouping_diagnostics":grouped["diagnostics"]}
                    dump(output/"group_preparation"/(str(seed)+".json"),
                         {"seed":seed,"numerical_signature":signature,"X_sha256":array_sha256(X),
                          "preparation_sec":mapping_cache[seed]["mapping_preparation_sec"],
                          "near_grouping_diagnostics":grouped["diagnostics"],
                          "scope":"One X-only preparation per seed, shared across paired layouts/amplitudes"})
                for arm in config["arms"]:
                    control_id=identifier+"__"+arm; groups=mapping_cache[seed][arm]
                    anchors=[group["anchor_column_0based"] for group in groups]
                    mapped_truth=aggregate(groups,truth); mapped_first=aggregate(groups,first); mapped_second=aggregate(groups,second)
                    reduced_X=np.asfortranarray(X[:,anchors])
                    identity=len(anchors)==X.shape[1] and np.array_equal(reduced_X,X) and np.array_equal(mapped_truth,truth)
                    metadata_path=output/"controls"/(control_id+".json")
                    if metadata_path.exists():
                        saved=json.loads(metadata_path.read_text())
                        validate_cached_control(saved,output,signature,provenance["config_sha256"])
                        records.append(saved);continue
                    mapping_start=perf_counter()
                    mapping={name:mapping_diagnostics(X,y,groups,coef,lam) for name,coef in
                             (("truth",truth),("old_sklearn",first),("old_skglm",second))}
                    for group in groups:
                        group["feature_names"]=[old_ds["feature_names"][j] for j in group["columns_0based"]]
                        group["original_wap_columns_0based"]=[old_ds["retained_original_wap_columns_0based"][j] for j in group["columns_0based"]]
                    flags=truth_identity_diagnostics(groups,truth,config["support_thresholds"])
                    reduced_hashes={"X":array_sha256(reduced_X),"y":array_sha256(y),"w_true":array_sha256(mapped_truth)}
                    data_hash=hashlib.sha256(json.dumps(reduced_hashes,sort_keys=True).encode()).hexdigest()
                    record={"control_id":control_id,"old_target_id":identifier,"params":params,"arm":arm,
                            "numerical_signature":signature,"config_sha256":provenance["config_sha256"],
                            "old_reference_path":str((baseline/"references"/(identifier+".json")).relative_to(ROOT)),
                            "old_reference_sha256":sha(baseline/"references"/(identifier+".json")),
                            "old_reference_coefficients_sha256":sha(cp),"old_dataset_sha256":sha(baseline/"datasets"/(identifier+".json")),
                            "old_p":X.shape[1],"new_p":len(anchors),"retained_old_columns":anchors,
                            "deleted_old_columns":sorted(set(range(X.shape[1]))-set(anchors)),"groups":groups,
                            "old_lambda":lam,"response_unchanged":True,"old_y_sha256":array_sha256(y),
                            "reduced_input_hashes":reduced_hashes,"reduced_dataset_hash":data_hash,
                            "old_dataset_reconstruction_sec":reconstruction_sec,
                            "shared_group_preparation_path":f"group_preparation/{seed}.json",
                            "raw_RSSI_subsample_hash":mapping_cache[seed]["raw_RSSI_subsample_hash"],
                            "mapping_diagnostics":mapping,"mapped_truth_support_diagnostics":flags,
                            "original_variable_scores":scores(first,second,truth,config["support_thresholds"]),
                            "mapped_old_output_scores":scores(mapped_first,mapped_second,mapped_truth,config["support_thresholds"]),
                            "mapping_diagnostics_sec":perf_counter()-mapping_start,
                            "new_public_reference_calls":0,"new_actual_library_fit_attempts":0,"public_reference_call_slots_reserved":0,
                            "interpretation":"Post-hoc mechanism control; original-variable and reduced-coordinate F1 are different tasks. Keeping y/lambda does not prove exact objective equivalence."}
                    if arm=="final_float64_exact" and not identity:
                        record.update(status="protocol_review_required",reference_origin="no_new_fit")
                        record.update(unevaluated_qualification("unexpected_final_float64_exact_deletion_requires_review"))
                    elif identity:
                        record.update(status="baseline_reused_identical_problem",reference_origin="old_frozen_reference",
                                      reduced_coordinate_scores=record["original_variable_scores"])
                        record.update(qualification(old_refs,config))
                    else:
                        control_data={"X":reduced_X,"y":y,"w_true":mapped_truth,
                                      "metadata":{"lambda":lam,"hashes":reduced_hashes}}
                        new_params=dict(params,arm=arm)
                        ref_path=output/"references"/(control_id+".json")
                        attempt_path=attempt_dir/(control_id+".json")
                        was_cached=ref_path.exists()
                        budget_exhausted=(previous_wall+perf_counter()-started>=config["wall_budget_sec"] or
                                          previous_calls+new_calls+2>config["maximum_new_public_reference_calls"])
                        if not was_cached and attempt_path.exists():
                            record.update(status="incomplete_reference_attempt_no_retry",reference_origin="none",
                                          new_public_reference_calls=None,new_actual_library_fit_attempts=None,
                                          public_reference_call_slots_reserved=2)
                            record.update(unevaluated_qualification("prior_partial_reference_attempt_requires_review"))
                        elif not was_cached and budget_exhausted:
                            record.update(status="budget_not_run",reference_origin="none")
                            record.update(unevaluated_qualification("bounded_phase_budget_no_fit"))
                        else:
                            if not was_cached:
                                dump(attempt_path,{"control_id":control_id,"numerical_signature":signature,
                                     "config_sha256":provenance["config_sha256"],"dataset_hash":data_hash,"invocation":invocation,
                                     "reserved_public_reference_calls":2,"state":"started"})
                                new_calls+=2
                            try:
                                ref,r1,r2=fit_references(control_data,config,output,control_id,new_params,signature,provenance["config_sha256"],data_hash)
                            except Exception as error:
                                record.update(status="reference_error",reference_origin="failed_new_fit" if not was_cached else "invalid_reference_cache",
                                              new_public_reference_calls=None,new_actual_library_fit_attempts=None,
                                              public_reference_call_slots_reserved=2,
                                              error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc())
                                record.update(unevaluated_qualification("reference_exception_partial_attempt_count_unknown"))
                                if not was_cached:
                                    failed=json.loads(attempt_path.read_text());failed.update(state="failed_no_retry",error_type=type(error).__name__)
                                    dump(attempt_path,failed)
                            else:
                                if not was_cached:
                                    actual_attempts+=ref["actual_library_fit_attempts"]
                                    completed=json.loads(attempt_path.read_text())
                                    completed.update(state="completed",actual_library_fit_attempts=ref["actual_library_fit_attempts"],
                                                     reference_sha256=sha(ref_path))
                                    dump(attempt_path,completed)
                                record.update(status=ref["status"],reference_origin="new_reduced_problem_fit",
                                              new_public_reference_calls=2,new_actual_library_fit_attempts=ref["actual_library_fit_attempts"],
                                              public_reference_call_slots_reserved=2,reference_loaded_from_cache=was_cached,
                                              reduced_coordinate_scores=scores(r1,r2,mapped_truth,config["support_thresholds"]),
                                              reduced_reference_path=f"references/{control_id}.json",reduced_reference_sha256=sha(ref_path))
                                record.update(qualification(ref,config))
                    write_start=perf_counter()
                    artifact=output/"mappings"/(control_id+".npz");artifact.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(artifact,old_w_true=truth,mapped_w_true=mapped_truth,
                                        old_ref_sklearn=first,old_ref_skglm=second,
                                        mapped_old_ref_sklearn=mapped_first,mapped_old_ref_skglm=mapped_second,
                                        retained_old_columns=np.asarray(anchors,dtype=np.int64))
                    record["mapping_artifact_sha256"]=sha(artifact)
                    record["mapping_artifact_write_sec"]=perf_counter()-write_start
                    dump(metadata_path,record);records.append(record)
                    print(json.dumps({"control":control_id,"status":record["status"],"old_p":X.shape[1],"new_p":len(anchors),"new_public_calls":record["new_public_reference_calls"]}),flush=True)
            dump(output/"completion.json",{"numerical_signature":signature,"expected_old_targets":len(targets),
                  "expected_controls":len(targets)*len(config["arms"]),"recorded_controls":len(records),
                  "new_public_reference_calls_this_invocation":new_calls,"new_actual_library_fit_attempts_this_invocation":actual_attempts,
                  "stored_known_new_public_reference_calls":sum(row["new_public_reference_calls"] or 0 for row in records),
                  "unknown_partial_reference_attempt_controls":sum(row["new_public_reference_calls"] is None for row in records),
                  "reserved_public_reference_call_slots":sum(row["public_reference_call_slots_reserved"] for row in records),
                  "accuracy_eligible_controls":sum(row["accuracy_eligible"] is True for row in records),
                  "consensus_controls":sum(row["reference_consensus"] is True for row in records),
                  "complete_grid":len(records)==len(targets)*len(config["arms"]),
                  "wall_sec_this_invocation":perf_counter()-started,"previous_recorded_wall_sec":previous_wall,
                  "posthoc":True,"interpretation":"No independent confirmation or exact mathematical equivalence claim; original-coordinate results remain unchanged"})
    finally:
        dump(ledger/(invocation+".json"),{"numerical_signature":signature,"wall_sec":perf_counter()-started,"state":"finished",
             "new_public_reference_call_slots_reserved":new_calls,"new_actual_library_fit_attempts":actual_attempts,
             "scope":"Measured invocation wall time, soft checks between new fits; abrupt termination may prevent final ledger write"})


if __name__=="__main__":main()
