#!/usr/bin/env python3
"""Frozen column-order follow-up, with independent single-method qualification.

The check-only entry hashes existing inputs and builds a pure metadata manifest.
It never regenerates a design, warms a solver, fits, or writes result files.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter, time_ns
import traceback
import warnings
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"scripts"), str(ROOT/"src"), str(ROOT/".modern_deps")]
from run_workshop import dump, sha, source_signature as base_signature
from lasso_study.data import array_sha256
from lasso_study.solvers import certificate
from lasso_study.modern_solver import solve_modern, warmup_modern
from lasso_study.audit_data_v1 import near_proportional_groups_chunked


def _data_api():
    from lasso_study import column_permutation_v1 as module
    return module


def target_parameters(config):
    return _data_api().target_parameters(config)


def target_id(params):
    return _data_api().target_id(params)


def call_id(identifier, order_id, method):
    return identifier+"__"+order_id+"__"+method


def build_manifest(config, targets=None):
    """Order-outer schedule; positions, sources and methods follow the protocol."""
    targets = target_parameters(config) if targets is None else targets
    by_source = {source: [row for row in targets if row["source"] == source]
                 for source in config["sources"]}
    calls = []
    for order_index, order in enumerate(config["execution_order_ids"]):
        offset = order_index % len(config["sources"])
        sources = config["sources"][offset:]+config["sources"][:offset]
        for position in range(max(map(len, by_source.values()))):
            methods = config["methods"] if (position+order_index) % 2 == 0 else list(reversed(config["methods"]))
            for source in sources:
                if position >= len(by_source[source]):
                    continue
                params = by_source[source][position]
                identifier = target_id(params)
                for method in methods:
                    calls.append({"call_index": len(calls), "call_id": call_id(identifier, order, method),
                                  "target_id": identifier, "params": params, "source": source,
                                  "order_id": order,
                                  "permutation_order_id": "identity" if order == "identity_repeat" else order,
                                  "is_primary_order": order in config["main_order_ids"],
                                  "is_identity_repeat": order == "identity_repeat", "method": method})
    if len(calls) != config["expected_public_calls"] or len({row["call_id"] for row in calls}) != len(calls):
        raise ValueError("Column-order call grid is incomplete or duplicated")
    primary = sum(row["is_primary_order"] for row in calls)
    if (primary != config["expected_primary_public_calls"]
            or len(calls)-primary != config["expected_qc_public_calls"]
            or len(calls) > config["maximum_public_reference_calls"]):
        raise ValueError("Primary/QC call counts differ from the frozen protocol")
    return calls


def _old_paths(config, params):
    base = ROOT/config["baseline_paths"][params["source"]]
    identifier = target_id(params)
    return {"dataset": base/"datasets"/(identifier+".json"),
            "reference": base/"references"/(identifier+".json"),
            "coefficients": base/"reference_coefficients"/(identifier+".npz"),
            "environment": base/"environment.json"}


def _registered_orders(config, p):
    orders = _data_api().make_permutations(p, seeds=tuple(config["permutation_seeds"]))
    if [row["order_id"] for row in orders] != config["main_order_ids"]:
        raise ValueError("Data helper permutation IDs differ from the protocol")
    if config["replacement_seed_start"] != max(config["permutation_seeds"])+1:
        raise ValueError("Replacement seed policy differs from the frozen data API")
    return orders


def column_signature(config_path):
    """Lock code, dependencies, protocol, actual orders, schedule and old artifacts."""
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    targets = target_parameters(config)
    manifest = build_manifest(config, targets)
    base, previous = base_signature()
    files = {ROOT/config["protocol_path"], ROOT/config["uji_config_path"], ROOT/config["external_config_path"]}
    baseline_configs = {source: ROOT/config["uji_config_path" if source=="uji" else "external_config_path"]
                        for source in config["sources"]}
    if any(json.loads(path.read_text())["references"]!=config["references"] for path in set(baseline_configs.values())):
        raise RuntimeError("Solver/qualification settings differ from a frozen historical configuration")
    source_files={"uji":ROOT/"data/benchmarks/ujiindoorloc/trainingData.csv",
                  "news":ROOT/"data/raw/OnlineNewsPopularity.csv",
                  "gene":ROOT/"data/benchmarks/leukemia/leukemia.libsvm"}
    source_file_hashes={source:sha(path) for source,path in source_files.items()}
    files.update(source_files.values())
    designs = {}
    for params in targets:
        paths = _old_paths(config, params); files.update(paths.values())
        dataset = json.loads(paths["dataset"].read_text())
        reference = json.loads(paths["reference"].read_text())
        if (sha(paths["coefficients"]) != reference["coefficient_sha256"]
                or dataset["dataset_hash"] != reference["dataset_hash"]
                or dataset["numerical_signature"] != reference["numerical_signature"]
                or dataset["config_sha256"] != reference["config_sha256"]
                or dataset["config_sha256"] != sha(baseline_configs[params["source"]])
                or dataset["lambda"] != reference["lambda"]):
            raise RuntimeError("Frozen historical dataset/reference provenance differs")
        recorded_source_hash=dataset["source_csv_sha256"] if params["source"]=="uji" else dataset["source_input_sha256"]
        if recorded_source_hash!=source_file_hashes[params["source"]]:
            raise RuntimeError("Actual source file differs from the frozen historical input")
        X_hash = dataset["hashes"]["X"]
        orders = _registered_orders(config, dataset["p"])
        if X_hash in designs and designs[X_hash]["orders"] != orders:
            raise RuntimeError("Shared X received inconsistent permutations")
        designs[X_hash] = {"X_sha256": X_hash, "p": dataset["p"], "orders": orders}
        grouping = ROOT/config["baseline_paths"][params["source"]]/"grouping"/(X_hash+".json")
        if params["source"] != "uji" and grouping.exists():
            files.add(grouping)
    if len(targets) != config["expected_targets"] or len(designs) != config["expected_distinct_designs"]:
        raise ValueError("Target or distinct-design count differs from the registration")
    input_hashes = {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}
    module_path = ROOT/"src/lasso_study/column_permutation_v1.py"
    payload = {"base_numerical_signature": base, "base_provenance": previous,
               "driver_sha256": sha(Path(__file__)), "data_module_sha256": sha(module_path),
               "audit_data_sha256": sha(ROOT/"src/lasso_study/audit_data_v1.py"),
               "config_sha256": sha(config_path), "protocol_sha256": sha(ROOT/config["protocol_path"]),
               "frozen_input_hashes": input_hashes,
               "manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
               "permutation_registry_sha256": hashlib.sha256(json.dumps(designs, sort_keys=True).encode()).hexdigest()}
    payload["old_input_hashes"] = input_hashes
    payload["source_hashes"] = dict(previous["source_hashes"], **{
        "scripts/run_column_permutation_v1.py":payload["driver_sha256"],
        "src/lasso_study/column_permutation_v1.py":payload["data_module_sha256"],
        "src/lasso_study/audit_data_v1.py":payload["audit_data_sha256"]})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def original_coefficients(coef_permuted, permutation):
    p = len(coef_permuted)
    permutation = np.asarray(permutation, dtype=np.int64)
    if permutation.shape != (p,) or not np.array_equal(np.sort(permutation), np.arange(p)):
        raise ValueError("Invalid feature permutation")
    mapped = np.empty(p, dtype=np.float64)
    mapped[permutation] = coef_permuted
    return mapped


def compare_invariance(X, X_permuted, coef_permuted, coef_original, permutation,
                       permuted_cert, original_cert, atol=5e-12, rtol=5e-12):
    """Compare two coordinate descriptions of one returned coefficient vector."""
    metrics = {}
    for key in ("primal", "dual", "gap", "raw_gap", "normalized_gap", "dual_feasibility", "kkt_inf"):
        left, right = float(permuted_cert[key]), float(original_cert[key])
        difference = abs(left-right); tolerance = atol+rtol*max(abs(left), abs(right))
        metrics[key] = {"absolute_difference": difference, "allowed_difference": tolerance,
                        "passed": bool(np.isfinite([left, right]).all() and difference <= tolerance)}
    left, right = X_permuted@coef_permuted, X@coef_original
    prediction_error = float(np.max(np.abs(left-right)))
    prediction_scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    metrics["prediction"] = {"maximum_absolute_difference": prediction_error,
                             "rmse": float(np.linalg.norm(left-right)/np.sqrt(len(X))),
                             "allowed_difference": atol+rtol*prediction_scale,
                             "passed": bool(np.isfinite(left).all() and np.isfinite(right).all()
                                            and prediction_error <= atol+rtol*prediction_scale)}
    mapping_passed = bool(np.array_equal(coef_original[np.asarray(permutation)], coef_permuted))
    l1_difference = abs(float(np.abs(coef_original).sum())-float(np.abs(coef_permuted).sum()))
    l1_limit = atol+rtol*max(float(np.abs(coef_original).sum()), float(np.abs(coef_permuted).sum()))
    metrics["l1"] = {"absolute_difference": l1_difference, "allowed_difference": l1_limit,
                     "passed": bool(np.isfinite(l1_difference) and l1_difference <= l1_limit)}
    return {"passed": bool(mapping_passed and all(row["passed"] for row in metrics.values())),
            "inverse_mapping_bitwise_passed": mapping_passed, "atol": atol, "rtol": rtol, "metrics": metrics}


def independent_qualification(coef, cert, library_status, elapsed_sec, settings):
    precision_reasons = []
    if not np.isfinite(coef).all():
        precision_reasons.append("nonfinite_coefficient")
    if not all(np.isfinite(cert.get(key, np.nan)) for key in
               ("primal", "dual", "gap", "raw_gap", "normalized_gap", "dual_feasibility", "kkt_inf")):
        precision_reasons.append("nonfinite_objective_or_certificate")
    if not np.isfinite(cert.get("normalized_gap", np.nan)) or cert["normalized_gap"] > settings["normalized_gap_tol"]:
        precision_reasons.append("normalized_gap_above_tolerance")
    if not np.isfinite(cert.get("dual_feasibility", np.nan)) or cert["dual_feasibility"] > 1+1e-12:
        precision_reasons.append("dual_not_feasible")
    reasons = list(precision_reasons)
    status_passed = library_status == "converged"
    budget_passed = bool(np.isfinite(elapsed_sec) and 0 <= elapsed_sec <= settings["time_limit"])
    if not status_passed:
        reasons.append("library_result_not_common_converged_status")
    if not budget_passed:
        reasons.append("observed_public_budget_exceeded")
    return {"accuracy_eligible": not reasons,"precision_eligible":not precision_reasons,
            "budget_eligible":budget_passed,"library_status_eligible":status_passed,
            "qualification_failure_reasons": reasons}


def score(coef, truth, tau):
    selected, actual = np.abs(coef) > tau, np.abs(truth) > 0
    tp = int(np.sum(selected & actual)); fp = int(np.sum(selected & ~actual)); fn = int(np.sum(~selected & actual))
    value = Fraction(2*tp, 2*tp+fp+fn) if 2*tp+fp+fn else Fraction(0)
    return {"tp": tp, "fp": fp, "fn": fn, "nnz": int(selected.sum()), "f1": float(value),
            "f1_exact": {"numerator": value.numerator, "denominator": value.denominator},
            "support_indices": np.flatnonzero(selected).tolist(),
            "thresholded_signs": np.where(selected, np.sign(coef), 0.).astype(int).tolist()}


def compare_outputs(X, fresh, previous, fresh_cert, previous_cert, config):
    """Historical/repeat QC uses existing coefficient/objective/support criteria."""
    settings = config["references"]
    linf = float(np.max(np.abs(fresh-previous)))
    objective_delta = abs(fresh_cert["primal"]-previous_cert["primal"])/max(1.,abs(fresh_cert["primal"]),abs(previous_cert["primal"]))
    by_tau = {}
    for tau in config["support_thresholds"]:
        left, right = np.abs(fresh)>tau, np.abs(previous)>tau
        signed_left, signed_right = np.where(left,np.sign(fresh),0.), np.where(right,np.sign(previous),0.)
        by_tau[str(tau)] = {"same_support": bool(np.array_equal(left,right)),
                            "same_thresholded_signs": bool(np.array_equal(signed_left,signed_right)),
                            "support_symmetric_difference": int(np.count_nonzero(left!=right)),
                            "thresholded_sign_difference": int(np.count_nonzero(signed_left!=signed_right))}
    prediction_delta = X@(fresh-previous)
    passed = bool(linf<=settings["coef_linf_agreement_tol"]
                  and objective_delta<=settings["normalized_objective_agreement_tol"]
                  and all(row["same_support"] and row["same_thresholded_signs"] for row in by_tau.values()))
    return {"passed": passed, "coefficient_linf_difference": linf,
            "normalized_primal_difference": objective_delta,
            "prediction_delta_rmse": float(np.linalg.norm(prediction_delta)/np.sqrt(len(X))),
            "prediction_delta_max_abs": float(np.max(np.abs(prediction_delta))),
            "certificate_absolute_differences": {key: abs(float(fresh_cert[key])-float(previous_cert[key]))
                                                 for key in ("primal","dual","gap","normalized_gap","dual_feasibility","kkt_inf")},
            "by_support_threshold": by_tau}


def run_single_method(data, order, method, config):
    """One public call, measured through inverse mapping and both certificates."""
    X,y = data["X"],data["y"]; lam = data["metadata"]["lambda"]
    preparation = perf_counter()
    permutation = np.asarray(order["permutation"],dtype=np.int64)
    Xp = np.asfortranarray(X[:,permutation],dtype=np.float64)
    permuted_input_hashes={"X":array_sha256(Xp),"y":array_sha256(y),
                           "w_true":array_sha256(np.asarray(data["w_true"][permutation],dtype=np.float64))}
    preparation_sec = perf_counter()-preparation
    settings = config["references"]
    start = perf_counter()
    if method == "sklearn":
        fit_start = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            model = Lasso(alpha=lam,fit_intercept=False,tol=settings["sklearn_tol"],
                          max_iter=settings["sklearn_max_iter"],selection="cyclic",warm_start=False).fit(Xp,y)
        w = model.coef_.copy()
        fit_elapsed = perf_counter()-fit_start
        library = {"method":"sklearn_cyclic_cd","iterations":int(model.n_iter_),
                   "iteration_unit":"complete cyclic coordinate epochs","library_tol":settings["sklearn_tol"],
                   "library_max_iter":settings["sklearn_max_iter"],"attempt_count":1,
                   "attempts":[{"attempt":1,"library_tol":settings["sklearn_tol"],"fit_elapsed_sec":fit_elapsed,
                                "initialization":"zero coefficients (new estimator, warm_start=False)"}],
                   "warnings":[str(row.message) for row in caught]}
        library_status = None
    elif method == "skglm":
        library = solve_modern(Xp,y,lam,tol=settings["normalized_gap_tol"],time_limit=settings["time_limit"],**settings["modern"])
        w = library.pop("coef"); library_status = library["status"]
    else:
        raise ValueError("Unregistered method")
    solver_elapsed = perf_counter()-start
    diagnostics_start = perf_counter()
    mapped = original_coefficients(w,permutation)
    permuted_cert = certificate(Xp,y,w,lam)
    original_cert = certificate(X,y,mapped,lam)
    invariance = compare_invariance(X,Xp,w,mapped,permutation,permuted_cert,original_cert,
                                   config["invariance_atol"],config["invariance_rtol"])
    diagnostics_sec = perf_counter()-diagnostics_start
    elapsed = perf_counter()-start
    if method == "sklearn":
        library_status = "converged" if permuted_cert["normalized_gap"]<=settings["normalized_gap_tol"] else "not_converged"
    qualified = independent_qualification(mapped,original_cert,library_status,elapsed,settings)
    if not invariance["passed"]:
        qualified["accuracy_eligible"] = False
        qualified["qualification_failure_reasons"].append("permutation_invariance_failure")
    status = "budget_exceeded" if elapsed>settings["time_limit"] else "converged" if qualified["accuracy_eligible"] else "not_converged"
    return {"coef_permuted":w,"coef_original":mapped,"certificate":original_cert,"certificate_original":original_cert,
            "permuted_certificate":permuted_cert,"invariance":invariance,"library_status":library_status,
            "permuted_input_hashes":permuted_input_hashes,
            "status":status,**qualified,"elapsed_sec":elapsed,"public_elapsed_sec":elapsed,
            "solver_elapsed_sec":solver_elapsed,"posterior_mapping_diagnostics_sec":diagnostics_sec,
            "permuted_X_preparation_sec":preparation_sec,"actual_library_fit_attempts":int(library["attempt_count"]),
            "library_result":library,"initialization":"independent zero start; no warm starts across orders or attempts",
            "time_limit_mode":"observed full public elapsed; no interruption within native fit"}


def _validate_groups(X, grouped, tolerances):
    if grouped["X_sha256"] != array_sha256(X):
        raise RuntimeError("Group record belongs to another original X")
    for tolerance in tolerances:
        groups = grouped["groups_by_tolerance"][str(tolerance)]
        if sorted(j for row in groups for j in row["columns_0based"]) != list(range(X.shape[1])):
            raise RuntimeError("Original-coordinate groups do not partition columns")
        for row in groups:
            anchor = row["anchor_column_0based"]
            for member,scale in zip(row["columns_0based"],row["member_scales_from_anchor"]):
                residual = np.linalg.norm(X[:,member]-scale*X[:,anchor])/np.linalg.norm(X[:,member])
                if residual > tolerance:
                    raise RuntimeError("Saved group member fails direct original-X anchor residual")


def validate_cached_run(saved, call, output, signature, config_hash, dataset_hash, permutation_hash):
    if any(saved.get(key)!=value for key,value in
           (("numerical_signature",signature),("config_sha256",config_hash),("dataset_hash",dataset_hash),
            ("permutation_sha256",permutation_hash),("call_id",call["call_id"]),("call_index",call["call_index"]))):
        raise RuntimeError("Cached call input/signature/schedule differs")
    artifact = output/saved["coefficient_path"] if saved.get("coefficient_path") else None
    if artifact is not None and sha(artifact)!=saved["coefficient_sha256"]:
        raise RuntimeError("Cached coefficient artifact hash differs")
    reservation_path=output/"reservations"/(call["call_id"]+".json")
    if not reservation_path.exists():
        raise RuntimeError("Cached call has no persistent public-call reservation")
    reservation=json.loads(reservation_path.read_text())
    if reservation["numerical_signature"]!=signature or reservation["dataset_hash"]!=dataset_hash:
        raise RuntimeError("Cached reservation belongs to another input/signature")
    if reservation.get("run_sha256") and reservation["run_sha256"]!=sha(output/"runs"/(call["call_id"]+".json")):
        raise RuntimeError("Cached run record hash differs from its completed reservation")


def validate_preparation(output, signature, config_hash):
    path=output/"preparation.json"
    record=json.loads(path.read_text())
    if record["numerical_signature"]!=signature or record["config_sha256"]!=config_hash:
        raise RuntimeError("Prepared artifacts belong to a different frozen configuration")
    for relative,expected in record["artifact_hashes"].items():
        if sha(output/relative)!=expected:
            raise RuntimeError("Prepared artifact differs from its pre-fit lock: "+relative)
    return record


def main():
    invocation_start = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/column_permutation_v1.json")
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--prepare-only",action="store_true",
                        help="Regenerate and save locked inputs/orders/groups, without warmup or fitting")
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    if args.check_only and args.prepare_only:
        raise ValueError("Choose check-only or prepare-only")
    targets = target_parameters(config); calls = build_manifest(config,targets)
    signature,provenance = column_signature(args.config)
    if args.check_only:
        print(json.dumps({"targets":len(targets),"expected_distinct_designs":config["expected_distinct_designs"],
                          "public_calls":len(calls),"primary_calls":sum(row["is_primary_order"] for row in calls),
                          "qc_calls":sum(row["is_identity_repeat"] for row in calls),
                          "numerical_signature":signature,"config_sha256":provenance["config_sha256"],
                          "manifest_sha256":provenance["manifest_sha256"],"permutation_registry_sha256":provenance["permutation_registry_sha256"],
                          "data_generated":False,"new_fits":0,"result_files_written":0},indent=2));return
    if config["threads"]!=1:
        raise ValueError("Column permutation protocol requires one numerical thread")
    output = ROOT/config["output_path"]; output.mkdir(parents=True,exist_ok=True)
    environment_path = output/"environment.json"
    if environment_path.exists() and json.loads(environment_path.read_text())["numerical_signature"]!=signature:
        raise RuntimeError("Result directory has a different frozen signature")
    if (output/"stage_stop.json").exists():
        raise RuntimeError("Frozen stage_stop prevents further fits; review requires a new protocol/version")
    if not args.prepare_only:
        if not (output/"preparation.json").exists():
            raise RuntimeError("Run --prepare-only, then lock actual prepared artifacts before fitting")
        validate_preparation(output,signature,provenance["config_sha256"])
        lock_path=output/"protocol_lock.json"
        if not lock_path.exists():
            raise RuntimeError("Prepared inputs require the root protocol_lock.json before fitting")
        locked=json.loads(lock_path.read_text())
        if locked.get("numerical_signature")!=signature or locked.get("preparation_sha256")!=sha(output/"preparation.json"):
            raise RuntimeError("Root protocol lock differs from the actual prepared artifact manifest")
    ledger = output/"invocations"; reservations = output/"reservations"
    old_invocations = [json.loads(path.read_text()) for path in ledger.glob("*.json")]
    if any(row.get("state")!="finished" for row in old_invocations):
        raise RuntimeError("Incomplete invocation leaves wall budget unknown; automatic resume requires review")
    previous_wall = sum(row["wall_sec"] for row in old_invocations)
    previous_reserved = len(list(reservations.glob("*.json")))
    invocation = str(time_ns());new_reserved=0;new_actual_attempts=0;unknown_actual_attempt_calls=[];budget_stop=False;stopped=False
    dump(ledger/(invocation+".json"),{"state":"running","numerical_signature":signature,"wall_sec":0.})
    try:
        with threadpool_limits(limits=1):
            if not environment_path.exists():
                dump(environment_path,dict(provenance,numerical_signature=signature,source_signature=signature,
                     started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),python=sys.version,platform=platform.platform(),
                     threads=1,threadpools=threadpool_info()))
                dump(output/"config.json",config)
                (output/"protocol_snapshot.md").write_bytes((ROOT/config["protocol_path"]).read_bytes())
            manifest = {"numerical_signature":signature,"config_sha256":provenance["config_sha256"],
                        "manifest_sha256":provenance["manifest_sha256"],"expected_public_calls":len(calls),
                        "targets":targets,"calls":calls}
            if (output/"manifest.json").exists() and json.loads((output/"manifest.json").read_text())!=manifest:
                raise RuntimeError("Saved fixed manifest differs")
            if not (output/"manifest.json").exists():dump(output/"manifest.json",manifest)
            prepare_start = perf_counter();api = _data_api();sources = api.load_sources(config)
            source_preparation_sec = perf_counter()-prepare_start;data_by_id={};designs={}
            for params in targets:
                start = perf_counter();data = api.make_target(config,params,sources)
                identifier = target_id(params);X_hash = array_sha256(data["X"])
                if X_hash!=data["metadata"]["hashes"]["X"]:
                    raise RuntimeError("Regenerated original X hash differs")
                data_hash = hashlib.sha256(json.dumps(data["metadata"]["hashes"],sort_keys=True).encode()).hexdigest()
                if X_hash not in designs:
                    orders = _registered_orders(config,data["X"].shape[1])
                    permutation_record = {"numerical_signature":signature,"X_sha256":X_hash,
                                          "p":data["X"].shape[1],"orders":orders}
                    permutation_path = output/"permutations"/(X_hash+".json")
                    if permutation_path.exists() and json.loads(permutation_path.read_text())!=permutation_record:
                        raise RuntimeError("Saved permutations differ")
                    if not permutation_path.exists():dump(permutation_path,permutation_record)
                    group_start=perf_counter();tolerances=tuple(dict.fromkeys([config["grouping_tolerance"]]+config["grouping_sensitivity_tolerances"]))
                    existing = data["origin"].get("existing_grouping_path")
                    group_path=output/"groups"/(X_hash+".json")
                    if group_path.exists():
                        grouping=json.loads(group_path.read_text());group_origin=grouping["origin"]
                        if grouping["numerical_signature"]!=signature:raise RuntimeError("Prepared grouping source signature differs")
                    elif existing and (ROOT/existing).exists():
                        grouping=json.loads((ROOT/existing).read_text());group_origin={"kind":"existing_frozen_record","path":existing,"sha256":sha(ROOT/existing)}
                    else:
                        grouping=near_proportional_groups_chunked(data["X"],tolerances=tolerances)
                        group_origin={"kind":"original_X_only_pre_fit_computation"}
                    _validate_groups(data["X"],grouping,tolerances)
                    if not group_path.exists():
                        dump(group_path,dict(grouping,numerical_signature=signature,origin=group_origin,
                             preparation_sec=perf_counter()-group_start,coordinate_system="original frozen columns"))
                    designs[X_hash]={"X":data["X"],"orders":{row["order_id"]:row for row in orders},
                                     "permutation_path":str(permutation_path.relative_to(output)),
                                     "permutation_record_sha256":sha(permutation_path)}
                else:
                    data["X"] = designs[X_hash]["X"]
                data_by_id[identifier]=data
                dataset_record=dict(data["metadata"],target_id=identifier,params=params,origin=data["origin"],
                                    numerical_signature=signature,config_sha256=provenance["config_sha256"],dataset_hash=data_hash,
                                    preprocessing_sec=perf_counter()-start)
                dataset_path=output/"datasets"/(identifier+".json")
                if dataset_path.exists():
                    saved=json.loads(dataset_path.read_text())
                    if (saved["numerical_signature"]!=signature or saved["dataset_hash"]!=data_hash
                            or saved["origin"]!=data["origin"]):
                        raise RuntimeError("Saved regenerated dataset provenance differs")
                else:
                    dump(dataset_path,dataset_record)
            if len(designs)!=config["expected_distinct_designs"]:
                raise RuntimeError("Unexpected number of regenerated X designs")
            if args.prepare_only:
                prepared_paths=[environment_path,output/"manifest.json",output/"config.json",output/"protocol_snapshot.md"]
                for directory in ("datasets","permutations","groups"):
                    prepared_paths+=sorted((output/directory).glob("*.json"))
                prepared={"numerical_signature":signature,"config_sha256":provenance["config_sha256"],
                          "targets":len(targets),"distinct_designs":len(designs),"public_calls":len(calls),
                          "artifact_hashes":{str(path.relative_to(output)):sha(path) for path in prepared_paths},
                          "new_fits":0,"warmup_performed":False,
                          "scope":"Actual original-coordinate data manifests, groups and six orders saved before root lock"}
                if (output/"preparation.json").exists():
                    validate_preparation(output,signature,provenance["config_sha256"])
                    if json.loads((output/"preparation.json").read_text())!=prepared:
                        raise RuntimeError("Prepared artifact list changed")
                else:dump(output/"preparation.json",prepared)
                dump(output/"setup"/(invocation+".json"),{"mode":"prepare_only","source_preparation_sec":source_preparation_sec,
                     "all_preparation_sec":perf_counter()-prepare_start,"warmup_performed":False,"new_fits":0})
                print(json.dumps({"prepared":True,"targets":len(targets),"distinct_designs":len(designs),"public_calls":len(calls),
                                  "numerical_signature":signature,"preparation_sha256":sha(output/"preparation.json"),
                                  "new_fits":0,"warmup_performed":False},indent=2))
                return
            missing = any(not (output/"runs"/(row["call_id"]+".json")).exists() for row in calls)
            time_available=previous_wall+perf_counter()-invocation_start<config["wall_budget_sec"]
            warmup=warmup_modern() if missing and time_available else {"performed":False,"elapsed_sec":0.,"reason":"all calls cached or phase budget exhausted"}
            if not (output/"warmup.json").exists():dump(output/"warmup.json",warmup)
            dump(output/"setup"/(invocation+".json"),{"source_preparation_sec":source_preparation_sec,
                 "all_preparation_sec":perf_counter()-prepare_start,"warmup":warmup,"distinct_designs":len(designs)})
            for call in calls:
                identifier=call["target_id"];data=data_by_id[identifier]
                X_hash=data["metadata"]["hashes"]["X"];design=designs[X_hash]
                order=design["orders"][call["permutation_order_id"]]
                data_hash=hashlib.sha256(json.dumps(data["metadata"]["hashes"],sort_keys=True).encode()).hexdigest()
                run_path=output/"runs"/(call["call_id"]+".json");reservation_path=reservations/(call["call_id"]+".json")
                if run_path.exists():
                    saved=json.loads(run_path.read_text())
                    validate_cached_run(saved,call,output,signature,provenance["config_sha256"],data_hash,order["permutation_sha256"])
                    if (saved.get("invariance",{}).get("passed") is False or
                            call["order_id"]=="identity" and saved.get("baseline_comparison",{}).get("passed") is False):
                        dump(output/"stage_stop.json",dict(call,numerical_signature=signature,reason="cached_correctness_or_history_QC_failure"))
                        stopped=True;break
                    continue
                common=dict(call,numerical_signature=signature,source_signature=signature,config_sha256=provenance["config_sha256"],
                            dataset_hash=data_hash,input_hashes=data["metadata"]["hashes"],**{"lambda":data["metadata"]["lambda"]},
                            permutation_sha256=order["permutation_sha256"],inverse_sha256=order["inverse_sha256"],
                            permutation_path=design["permutation_path"],permutation_record_sha256=design["permutation_record_sha256"],
                            historical_reference_path=data["historical_reference"]["path"],origin=data["origin"])
                if reservation_path.exists():
                    prior=json.loads(reservation_path.read_text())
                    if prior["numerical_signature"]!=signature or prior["dataset_hash"]!=data_hash:
                        raise RuntimeError("Prior partial reservation input differs")
                    dump(run_path,dict(common,status="prior_partial_call_no_retry",accuracy_eligible=None,
                         precision_eligible=None,budget_eligible=None,library_status_eligible=None,
                         qualification_failure_reasons=["reserved_call_without_completed_output"],
                         actual_library_fit_attempts=prior.get("actual_library_fit_attempts")))
                    continue
                if (previous_wall+perf_counter()-invocation_start>=config["wall_budget_sec"]
                        or previous_reserved+new_reserved>=config["maximum_public_reference_calls"]):
                    budget_stop=True;break
                reservation=dict(common,invocation=invocation,reserved_public_reference_calls=1,state="started")
                dump(reservation_path,reservation);new_reserved+=1
                call_start=perf_counter()
                try:
                    result=run_single_method(data,order,call["method"],config)
                except Exception as error:
                    unknown_actual_attempt_calls.append(call["call_id"])
                    failure=dict(common,status="call_exception",accuracy_eligible=None,
                                 precision_eligible=None,budget_eligible=None,library_status_eligible=None,
                                 qualification_failure_reasons=["call_exception_partial_attempt_count_unknown"],
                                 error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc(),
                                 actual_library_fit_attempts=None,elapsed_sec_before_exception=perf_counter()-call_start)
                    dump(run_path,failure);reservation.update(state="failed_no_retry",error_type=type(error).__name__)
                    dump(reservation_path,reservation)
                    # A posterior floating-point failure invalidates the correctness QC.
                    if isinstance(error,FloatingPointError):
                        dump(output/"stage_stop.json",dict(common,reason="posterior_certificate_exception",error_message=str(error)))
                        stopped=True;break
                    continue
                new_actual_attempts+=result["actual_library_fit_attempts"]
                reservation.update(state="fit_returned",actual_library_fit_attempts=result["actual_library_fit_attempts"],
                                   public_elapsed_sec=result["public_elapsed_sec"])
                dump(reservation_path,reservation)
                first=result.pop("coef_permuted");mapped=result.pop("coef_original")
                post_start=perf_counter();result["scores"]={str(tau):score(mapped,data["w_true"],tau) for tau in config["support_thresholds"]}
                historical=data["historical_reference"]
                if call["order_id"] in {"identity","identity_repeat"}:
                    old_coef=historical["ref_"+call["method"]]
                    old_cert=historical["metadata"][call["method"]]["certificate"]
                    result["baseline_comparison"]=compare_outputs(data["X"],mapped,old_coef,result["certificate"],old_cert,config)
                    result["baseline_comparison"]["status"]="pass" if result["baseline_comparison"]["passed"] else "baseline_reproduction_anomaly"
                if call["is_identity_repeat"]:
                    identity_path=output/"runs"/(call_id(identifier,"identity",call["method"])+".json")
                    if identity_path.exists():
                        identity=json.loads(identity_path.read_text())
                        if identity.get("coefficient_path"):
                            with np.load(output/identity["coefficient_path"],allow_pickle=False) as arrays:identity_coef=arrays["coef_original"].copy()
                            repeat=compare_outputs(data["X"],mapped,identity_coef,result["certificate"],identity["certificate"],config)
                            repeat["status"]=("pass" if repeat["passed"] else "fail") if result["accuracy_eligible"] and identity["accuracy_eligible"] else "undecidable"
                            result["repeat_comparison"]=repeat
                        else:result["repeat_comparison"]={"status":"undecidable","reason":"identity_coefficient_unavailable"}
                    else:result["repeat_comparison"]={"status":"undecidable","reason":"identity_call_unavailable"}
                result["historical_comparison_and_scoring_sec"]=perf_counter()-post_start
                write_start=perf_counter();artifact=output/"coefficients"/(call["call_id"]+".npz");artifact.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(artifact,coef_permuted=first,coef_original=mapped,w_true=data["w_true"],
                                    permutation=np.asarray(order["permutation"],dtype=np.int64),
                                    inverse_permutation=np.asarray(order["inverse_permutation"],dtype=np.int64))
                result.update(coefficient_path=str(artifact.relative_to(output)),coefficient_sha256=sha(artifact),
                              coef_path=str(artifact.relative_to(output)),
                              coefficient_write_sec=perf_counter()-write_start)
                saved=dict(common,**result);dump(run_path,saved)
                reservation.update(state="completed",actual_library_fit_attempts=result["actual_library_fit_attempts"],run_sha256=sha(run_path))
                dump(reservation_path,reservation)
                reason=("permutation_invariance_failure" if not result["invariance"]["passed"] else
                        "baseline_reproduction_anomaly" if call["order_id"]=="identity" and not result["baseline_comparison"]["passed"] else None)
                print(json.dumps({"call":call["call_id"],"index":call["call_index"],"status":result["status"],
                                  "accuracy_eligible":result["accuracy_eligible"],"gap":result["certificate"]["normalized_gap"],
                                  "elapsed_sec":result["elapsed_sec"],"stage_stop":reason}),flush=True)
                if reason:
                    dump(output/"stage_stop.json",dict(common,reason=reason,run_path=str(run_path.relative_to(output)),run_sha256=sha(run_path)))
                    stopped=True;break
            saved_runs=[json.loads(path.read_text()) for path in (output/"runs").glob("*.json")]
            dump(output/"completion.json",{"numerical_signature":signature,"config_sha256":provenance["config_sha256"],
                 "expected_targets":len(targets),"expected_public_calls":len(calls),"recorded_calls":len(saved_runs),
                 "complete_grid":len(saved_runs)==len(calls),"primary_recorded_calls":sum(row["is_primary_order"] for row in saved_runs),
                 "qc_recorded_calls":sum(row["is_identity_repeat"] for row in saved_runs),
                 "accuracy_eligible_calls":sum(row.get("accuracy_eligible") is True for row in saved_runs),
                 "status_counts":dict(Counter(row["status"] for row in saved_runs)),"budget_stop":budget_stop,"stage_stopped":stopped,
                 "new_public_call_slots_reserved":new_reserved,"total_public_call_slots_reserved":previous_reserved+new_reserved,
                 "new_known_actual_library_fit_attempts":new_actual_attempts,"unknown_partial_attempt_calls":unknown_actual_attempt_calls,
                 "previous_recorded_wall_sec":previous_wall,"wall_sec_this_invocation":perf_counter()-invocation_start})
    finally:
        dump(ledger/(invocation+".json"),{"state":"finished","numerical_signature":signature,
             "wall_sec":perf_counter()-invocation_start,"new_public_call_slots_reserved":new_reserved,
             "new_known_actual_library_fit_attempts":new_actual_attempts,"unknown_partial_attempt_calls":unknown_actual_attempt_calls,
             "budget_stop":budget_stop,"stage_stopped":stopped,
             "scope":"Full invocation wall time including preflight/preparation/warmup/output; no retries of reserved incomplete calls"})
    if stopped:raise SystemExit("Frozen correctness/history QC stopped the stage; retained outputs require review")
    if budget_stop:raise SystemExit("Frozen phase budget reached; unattempted manifest calls remain unresolved")


if __name__ == "__main__":
    main()
