#!/usr/bin/env python3
"""Reference qualification and F1 consequences on new external designs.

No FISTA/CD trajectory, SVD step-size estimate, feature grouping, or p-by-p
Gram is computed here. Frozen helpers store the two reference implementations.
All raw outputs remain visible, including noneligible and nonconsensus targets.
--check-only performs no data generation, warmup, fitting, or result writes.
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
from itertools import product
import json
from pathlib import Path
import platform
import sys
from time import perf_counter, time_ns
import traceback
from zoneinfo import ZoneInfo

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / ".modern_deps")]
from run_workshop import cache_guard, dump, fit_references, sha, source_signature as base_signature
from lasso_study.data import array_sha256
from lasso_study.modern_solver import warmup_modern


def _data_api():
    from lasso_study.audit_data_v1 import load_audit_source, make_audit_design
    return load_audit_source, make_audit_design


def _normalize_stage(stage):
    stage = stage.removeprefix("external_")
    if stage not in {"pilot", "confirm"}:
        raise ValueError("Stage must be pilot/confirm (external_ aliases are accepted)")
    return stage


def target_parameters(config, stage):
    stage = _normalize_stage(stage)
    if config["threads"] != 1 or set(config["sources"]) != {"news", "gene"}:
        raise ValueError("The frozen audit uses exactly news/gene and one numerical thread")
    targets = []
    all_seeds = set()
    for source, settings in config["sources"].items():
        pilot, confirm = settings["pilot_seeds"], settings["confirm_seeds"]
        if len(set(pilot + confirm)) != len(pilot) + len(confirm):
            raise ValueError(f"Duplicated or overlapping exploratory/confirmation seeds for {source}")
        if all_seeds.intersection(pilot + confirm):
            raise ValueError("Source-specific seed ranges overlap")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in pilot + confirm):
            raise ValueError("Seeds must be nonnegative integers")
        all_seeds.update(pilot + confirm)
        targets += [dict(source=source, seed=seed, amplitude=amplitude, c=float(c))
                    for seed, c, amplitude in product(settings[stage + "_seeds"], config["c_values"], config["amplitudes"])]
    identities = {(row["source"], row["seed"], row["amplitude"], row["c"]) for row in targets}
    if len(identities) != len(targets) or len(targets) != config["expected_targets"][stage]:
        raise ValueError("Audit target grid differs from the registered count")
    if len(targets) * 2 != config["expected_public_reference_calls"][stage]:
        raise ValueError("Audit public reference-call count differs from its grid")
    return targets


def target_id(params):
    return f"{params['source']}_seed{params['seed']}_c{params['c']:g}_{params['amplitude']}"


def audit_signature(config_path):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    protocol_path = (ROOT / config["protocol_path"]).resolve()
    if not protocol_path.is_relative_to(ROOT):
        raise ValueError("Audit protocol must be saved within this project")
    previous, provenance = base_signature()
    extension = {"base_numerical_signature": previous,
                 "audit_data_sha256": sha(ROOT / "src/lasso_study/audit_data_v1.py"),
                 "audit_driver_sha256": sha(Path(__file__)), "config_sha256": sha(config_path),
                 "protocol_path": str(protocol_path.relative_to(ROOT)), "protocol_sha256": sha(protocol_path)}
    signature = hashlib.sha256(json.dumps(extension, sort_keys=True).encode()).hexdigest()
    return signature, dict(extension, base_provenance=provenance)


def f1_fraction(tp, fp, fn):
    denominator = 2 * int(tp) + int(fp) + int(fn)
    return Fraction(2 * int(tp), denominator) if denominator else Fraction(0)


def exact_delta_flags(first, second, eligible, deltas):
    difference = abs(first - second)
    return {str(delta): bool(difference > Fraction(str(delta))) if eligible else None for delta in deltas}


def score(coef, truth, tau):
    selected, actual = np.abs(coef) > tau, np.abs(truth) > 0
    tp = int(np.sum(selected & actual)); fp = int(np.sum(selected & ~actual)); fn = int(np.sum(~selected & actual))
    fraction = f1_fraction(tp, fp, fn)
    thresholded = np.where(selected, np.sign(coef), 0.)
    return {"tp": tp, "fp": fp, "fn": fn, "nnz": int(selected.sum()),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "f1": float(fraction),
            "f1_exact": {"numerator": fraction.numerator, "denominator": fraction.denominator},
            "support_indices": np.flatnonzero(selected).tolist(),
            "exact_true_support": bool(np.array_equal(selected, actual)),
            "true_sign_error": int(np.sum(thresholded != np.sign(truth))),
            "exact_true_sign": bool(np.array_equal(thresholded, np.sign(truth)))}, fraction


def qualification(refs, config):
    settings = config["references"]
    by_reference, accuracy_reasons = {}, []
    for name in ("sklearn", "skglm"):
        row = refs[name]; cert = row["certificate"]; reasons = []
        if not all(np.isfinite(cert.get(key, np.nan)) for key in ("primal", "dual", "gap", "normalized_gap", "dual_feasibility")):
            reasons.append("nonfinite_objective_or_certificate")
        if not np.isfinite(cert.get("normalized_gap", np.nan)) or cert["normalized_gap"] > settings["normalized_gap_tol"]:
            reasons.append("normalized_gap_above_tolerance")
        if not np.isfinite(cert.get("dual_feasibility", np.nan)) or cert["dual_feasibility"] > 1 + 1e-12:
            reasons.append("dual_not_feasible")
        if not np.isfinite(row.get("elapsed_sec", np.nan)) or row["elapsed_sec"] > settings["time_limit"]:
            reasons.append("observed_budget_exceeded")
        if row.get("status") != "converged":
            reasons.append("library_result_not_common_converged_status")
        by_reference[name] = {"eligible": not reasons, "reasons": reasons, "status": row.get("status"),
                              "elapsed_sec": row.get("elapsed_sec"), "normalized_gap": cert.get("normalized_gap")}
        accuracy_reasons.extend(name + ":" + reason for reason in reasons)
    consensus_reasons = list(accuracy_reasons)
    if refs["coefficient_linf_difference"] > settings["coef_linf_agreement_tol"]:
        consensus_reasons.append("coefficient_linf_disagreement")
    if refs["normalized_primal_difference"] > settings["normalized_objective_agreement_tol"]:
        consensus_reasons.append("normalized_objective_disagreement")
    for tau in config["support_thresholds"]:
        comparison = refs["by_support_threshold"][str(tau)]
        if not comparison["same_support"]:
            consensus_reasons.append(f"support_disagreement_tau={tau}")
        if not comparison["same_thresholded_signs"]:
            consensus_reasons.append(f"thresholded_sign_disagreement_tau={tau}")
    eligible = not accuracy_reasons
    if bool(refs["reference_accepted"]) != (eligible and not consensus_reasons):
        raise AssertionError("Frozen helper consensus flag differs from explicit qualification reasons")
    return {"accuracy_eligible": eligible, "reference_accepted": bool(refs["reference_accepted"]),
            "qualification_failure_reasons": {"accuracy": accuracy_reasons, "consensus": consensus_reasons,
                                              "by_reference": by_reference}}


def evaluate_pair(data, refs, first, second, config):
    started = perf_counter()
    result = qualification(refs, config)
    eligible = result["accuracy_eligible"]
    scores = {}
    deltas = list(dict.fromkeys([config["primary_delta"]] + config["delta_sensitivity"]))
    for tau in config["support_thresholds"]:
        left, lf = score(first, data["w_true"], tau)
        right, rf = score(second, data["w_true"], tau)
        difference = abs(lf-rf)
        selected_left, selected_right = np.abs(first) > tau, np.abs(second) > tau
        signed_left = np.where(selected_left, np.sign(first), 0.)
        signed_right = np.where(selected_right, np.sign(second), 0.)
        scores[str(tau)] = {
            "sklearn": left, "skglm": right,
            "raw_absolute_f1_difference": float(difference), "raw_signed_f1_difference_skglm_minus_sklearn": float(rf-lf),
            "absolute_f1_difference_exact": {"numerator": difference.numerator, "denominator": difference.denominator},
            "eligible_absolute_f1_difference": float(difference) if eligible else None,
            "delta_exceeded": exact_delta_flags(lf, rf, eligible, deltas),
            "support_symmetric_difference_indices": np.flatnonzero(selected_left != selected_right).tolist(),
            "support_symmetric_difference_count": int(np.sum(selected_left != selected_right)),
            "sign_difference_count": int(np.sum(signed_left != signed_right)),
            "observed_two_output_f1_range": [float(min(lf, rf)), float(max(lf, rf))],
        }
    primary = scores[str(config["support_threshold"])]
    delta_prediction = data["X"] @ (first-second)
    result.update(scores_by_tau=scores, primary_support_threshold=config["support_threshold"],
                  primary_absolute_f1_difference=primary["eligible_absolute_f1_difference"],
                  primary_delta_exceeded=primary["delta_exceeded"][str(config["primary_delta"])],
                  prediction_difference_rmse=float(np.linalg.norm(delta_prediction)/np.sqrt(len(data["y"]))),
                  l1_norm_difference_abs=float(abs(np.abs(first).sum()-np.abs(second).sum())),
                  evaluation_sec_outside_reference_timers=perf_counter()-started,
                  interpretation="Raw output scores are retained for every target; the primary contrast requires accuracy+budget eligibility and does not condition on consensus. The observed two-output range is not a bound over all minimizers.")
    return result


def seed_outcomes(evaluations, targets, config):
    lookup = {row["target_id"]: row for row in evaluations}
    deltas = list(dict.fromkeys([config["primary_delta"]] + config["delta_sensitivity"]))
    output = []
    for source, seed in sorted({(params["source"], params["seed"]) for params in targets}):
        expected = [target_id(params) for params in targets if params["source"] == source and params["seed"] == seed]
        available = [lookup[identifier] for identifier in expected if identifier in lookup]
        states = {}
        for delta in deltas:
            key = str(delta)
            flags = [row["scores_by_tau"][str(config["support_threshold"])]["delta_exceeded"][key] for row in available]
            state = "event" if any(value is True for value in flags) else "no_event" if len(available) == len(expected) and all(value is False for value in flags) else "unresolved"
            states[key] = state
        output.append({"source": source, "seed": seed, "planned_target_count": len(expected),
                       "observed_target_count": len(available),
                       "accuracy_eligible_targets": sum(row["accuracy_eligible"] for row in available),
                       "primary_state": states[str(config["primary_delta"])], "states_by_delta": states})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "confirm", "external_pilot", "external_confirm"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reference_audit_v1.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(); stage = _normalize_stage(args.stage)
    config = json.loads(args.config.read_text())
    targets = target_parameters(config, stage)
    signature, provenance = audit_signature(args.config)
    config_hash = provenance["config_sha256"]
    if args.check_only:
        print(json.dumps({"stage": stage, "targets": len(targets), "public_reference_calls": 2*len(targets),
                          "numerical_signature": signature, "config_sha256": config_hash,
                          "protocol_sha256": provenance["protocol_sha256"], "data_generated": False,
                          "new_fits": 0, "result_files_written": 0}, indent=2))
        return
    output = ROOT / "results/reference_audit_v1" / stage
    output.mkdir(parents=True, exist_ok=True)
    env_path = output / "environment.json"
    if env_path.exists():
        previous = json.loads(env_path.read_text())
        if previous["numerical_signature"] != signature or previous["config_sha256"] != config_hash:
            raise RuntimeError("Existing audit stage has different frozen source/config/protocol")
    invocation = f"{stage}_{time_ns()}"
    invocation_start = perf_counter()
    ledger = ROOT / "results/reference_audit_v1/invocations"
    previous_wall = sum(row.get("wall_sec", 0.) for path in ledger.glob("*.json")
                        if (row := json.loads(path.read_text())).get("numerical_signature") == signature)
    new_targets = reused_targets = new_actual_attempts = 0
    unknown_reference_attempt_targets = []
    evaluations = []; references = []; budget_stop = False
    try:
        with threadpool_limits(limits=1):
            missing_refs = any(not (output / "references" / (target_id(params)+".json")).exists() for params in targets)
            setup_start = perf_counter()
            warmup = warmup_modern() if missing_refs else {"performed": False, "elapsed_sec": 0., "reason": "all references already cached"}
            setup_sec = perf_counter()-setup_start
            if not env_path.exists():
                dump(env_path, dict(provenance, numerical_signature=signature, source_signature=signature,
                                    stage=stage, external_stage_label="external_"+stage,
                                    started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), timezone="Asia/Shanghai",
                                    python=sys.version, platform=platform.platform(), threads=1, threadpools=threadpool_info()))
            dump(output / "config.json", config); dump(output / "effective_config.json", config)
            for snapshot, original_path, expected_hash in (
                (output / "config_original.json", args.config, config_hash),
                (output / "protocol_snapshot.md", ROOT / config["protocol_path"], provenance["protocol_sha256"]),
            ):
                if snapshot.exists() and sha(snapshot) != expected_hash:
                    raise RuntimeError("Original-text configuration/protocol snapshot hash mismatch")
                if not snapshot.exists():
                    snapshot.write_bytes(Path(original_path).read_bytes())
            if not (output / "warmup.json").exists():
                dump(output / "warmup.json", {"modern": warmup, "setup_sec": setup_sec,
                      "scope": "One engineering JIT warmup fit excluded from per-target reference times; no C-baseline warmup is needed"})
            load_source, make_design = _data_api()
            sources = {}
            for params in targets:
                identifier = target_id(params)
                if previous_wall + perf_counter()-invocation_start >= config["additional_wall_budget_sec"]:
                    budget_stop = True
                    break
                try:
                    phase = "source_or_data_preparation"
                    source = params["source"]
                    if source not in sources:
                        start = perf_counter()
                        sources[source] = load_source(source, source_seed=config["sources"][source]["source_seed"])
                        source_record = dict(sources[source]["metadata"], source=source,
                                             source_pool_shape=list(sources[source]["X_pool"].shape),
                                             source_pool_X_sha256=array_sha256(sources[source]["X_pool"]),
                                             load_preprocessing_sec=perf_counter()-start,
                                             numerical_signature=signature, config_sha256=config_hash)
                        source_path = output / "sources" / (source+".json")
                        if source_path.exists():
                            saved_source = json.loads(source_path.read_text())
                            if any(saved_source.get(name) != source_record[name] for name in
                                   ("source_pool_X_sha256", "numerical_signature", "config_sha256")):
                                raise RuntimeError("Cached source pool differs from current frozen source")
                        else:
                            dump(source_path, source_record)
                    start = perf_counter()
                    data = make_design(config, **params, source_data=sources[source])
                    data["X"] = np.asfortranarray(data["X"], dtype=np.float64)
                    prep_sec = perf_counter()-start
                    data_hash = hashlib.sha256(json.dumps(data["metadata"]["hashes"], sort_keys=True).encode()).hexdigest()
                    ds_path = output / "datasets" / (identifier+".json")
                    if ds_path.exists():
                        cache_guard(json.loads(ds_path.read_text()), signature, config_hash, data_hash)
                    else:
                        dump(ds_path, dict(data["metadata"], target_id=identifier, params=params, **params,
                                           numerical_signature=signature, source_signature=signature,
                                           config_sha256=config_hash, protocol_sha256=provenance["protocol_sha256"],
                                           dataset_hash=data_hash, preprocessing_sec=prep_sec))
                    was_cached = (output / "references" / (identifier+".json")).exists()
                    phase = "cached_reference_loading" if was_cached else "new_reference_fitting"
                    refs, first, second = fit_references(data, config, output, identifier, params,
                                                        signature, config_hash, data_hash)
                    references.append(refs)
                    if was_cached:
                        reused_targets += 1
                    else:
                        new_targets += 1; new_actual_attempts += refs["actual_library_fit_attempts"]
                    phase = "evaluation"
                    ref_path = output / "references" / (identifier+".json")
                    coef_path = output / "reference_coefficients" / (identifier+".npz")
                    eval_path = output / "evaluation" / (identifier+".json")
                    inputs = {"dataset_sha256": sha(ds_path), "reference_json_sha256": sha(ref_path),
                              "reference_coefficients_sha256": sha(coef_path)}
                    if eval_path.exists():
                        evaluation = json.loads(eval_path.read_text())
                        cache_guard(evaluation, signature, config_hash, data_hash)
                        if evaluation["input_hashes"] != inputs:
                            raise RuntimeError("Cached evaluation inputs have changed")
                    else:
                        evaluation = dict(evaluate_pair(data, refs, first, second, config),
                                          target_id=identifier, params=params, **params,
                                          numerical_signature=signature, source_signature=signature,
                                          config_sha256=config_hash, protocol_sha256=provenance["protocol_sha256"],
                                          dataset_hash=data_hash, input_hashes=inputs)
                        dump(eval_path, evaluation)
                    evaluations.append(evaluation)
                    print(json.dumps({"target": identifier, "accuracy_eligible": evaluation["accuracy_eligible"],
                                      "reference_accepted": evaluation["reference_accepted"],
                                      "eligible_delta_f1": evaluation["primary_absolute_f1_difference"],
                                      "primary_event": evaluation["primary_delta_exceeded"], "cached": was_cached}), flush=True)
                except Exception as error:
                    if phase == "new_reference_fitting":
                        unknown_reference_attempt_targets.append(identifier)
                    dump(output / "failures" / (identifier+".json"),
                         dict(params, target_id=identifier, error_type=type(error).__name__, error=str(error),
                              traceback=traceback.format_exc(), numerical_signature=signature, config_sha256=config_hash,
                              protocol_sha256=provenance["protocol_sha256"],
                              phase=phase,
                              reference_attempt_count_unknown=phase == "new_reference_fitting",
                              wall_sec_before_exception=perf_counter()-invocation_start))
                    raise
            outcomes = seed_outcomes(evaluations, targets, config)
            dump(output / "completion.json", {
                "stage": stage, "external_stage_label": "external_"+stage, "numerical_signature": signature,
                "config_sha256": config_hash, "protocol_sha256": provenance["protocol_sha256"],
                "expected_targets": len(targets), "recorded_targets": len(evaluations),
                "expected_public_reference_calls": len(targets)*2, "stored_public_reference_calls": len(references)*2,
                "stored_actual_library_fit_attempts": sum(row["actual_library_fit_attempts"] for row in references),
                "new_reference_targets_this_invocation": new_targets, "reused_reference_targets_this_invocation": reused_targets,
                "new_public_reference_calls_this_invocation": new_targets*2,
                "new_actual_library_fit_attempts_this_invocation": new_actual_attempts,
                "accuracy_eligible_targets": sum(row["accuracy_eligible"] for row in evaluations),
                "consensus_accepted_targets": sum(row["reference_accepted"] for row in evaluations),
                "failure_reason_counts": dict(Counter(reason for row in evaluations for reason in row["qualification_failure_reasons"]["consensus"])),
                "seed_outcomes": outcomes, "budget_stop": budget_stop,
                "prior_recorded_wall_sec": previous_wall, "wall_sec_this_invocation": perf_counter()-invocation_start,
                "complete_grid": len(evaluations) == len(targets),
                "interpretation": "Primary F1 contrasts use all accuracy-eligible targets, not only consensus accepted. Seeds are source-specific paired replication units; gene has one fixed design. No group analysis or optimizer trajectories were run."})
    finally:
        dump(ledger / (invocation+".json"), {"stage": stage, "invocation": invocation,
             "numerical_signature": signature, "config_sha256": config_hash,
             "protocol_sha256": provenance["protocol_sha256"], "wall_sec": perf_counter()-invocation_start,
             "new_reference_targets": new_targets, "new_actual_library_fit_attempts": new_actual_attempts,
             "unknown_reference_attempt_targets": unknown_reference_attempt_targets,
             "fit_count_scope": "Attempt count sums completed reference records; a reference exception can leave an explicitly unknown partial attempt count, never assumed zero",
             "budget_stop": budget_stop,
             "scope": "Measured invocation wall time including preparation/warmup/output; abrupt process termination can prevent this final ledger write"})
    if budget_stop:
        raise SystemExit("Soft phase budget reached; unattempted targets remain unresolved and are retained in the planned grid")


if __name__ == "__main__":
    main()
