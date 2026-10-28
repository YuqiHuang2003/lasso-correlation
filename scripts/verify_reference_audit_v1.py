#!/usr/bin/env python3
"""Independent integrity and numerical checks for the external reference audit.

No optimizer, full trajectory or feature grouping is run. Default verification
regenerates the already-recorded responses and recomputes two certificates per
target. Run after timed reference fits. --structure-only cannot yield a fully
numerically verified all_passed result. Original artifacts are never rewritten.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback
from zoneinfo import ZoneInfo

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(ROOT / ".modern_deps")]
from run_reference_audit_v1 import audit_signature, target_id
from run_workshop import dump, sha
from verify_workshop import Audit, compare_certificate
from lasso_study.data import array_sha256
from lasso_study.solvers import certificate


def expected_grid(config, stage):
    targets = []
    for source, settings in config["sources"].items():
        seeds = settings[stage + "_seeds"]
        if len(set(seeds)) != len(seeds) or set(settings["pilot_seeds"]) & set(settings["confirm_seeds"]):
            raise ValueError("Repeated or exploratory/confirmation-overlapping source seeds")
        for seed, c, amplitude in product(seeds, config["c_values"], config["amplitudes"]):
            targets.append(dict(source=source, seed=seed, amplitude=amplitude, c=float(c)))
    result = {target_id(params): params for params in targets}
    if len(result) != len(targets) or len(targets) != config["expected_targets"][stage]:
        raise ValueError("Registered exact target grid is inconsistent")
    return result


def counts_and_fraction(coef, truth, tau):
    selected, actual = np.abs(coef) > tau, np.abs(truth) > 0
    tp = int(np.sum(selected & actual)); fp = int(np.sum(selected & ~actual)); fn = int(np.sum(~selected & actual))
    den = 2*tp + fp + fn
    f1 = Fraction(2*tp, den) if den else Fraction(0)
    signs = np.where(selected, np.sign(coef), 0.)
    return {"tp": tp, "fp": fp, "fn": fn, "nnz": int(selected.sum()),
            "precision": tp/max(1,tp+fp), "recall": tp/max(1,tp+fn), "f1": float(f1),
            "f1_exact": {"numerator": f1.numerator, "denominator": f1.denominator},
            "support_indices": np.flatnonzero(selected).tolist(),
            "exact_true_support": bool(np.array_equal(selected, actual)),
            "true_sign_error": int(np.sum(signs != np.sign(truth))),
            "exact_true_sign": bool(np.array_equal(signs, np.sign(truth)))}, f1, selected, signs


def accuracy_conditions(record, settings):
    cert = record["certificate"]
    return (record.get("status") == "converged"
            and np.isfinite(record.get("elapsed_sec", np.nan))
            and record["elapsed_sec"] <= settings["time_limit"]
            and all(np.isfinite(cert.get(name, np.nan)) for name in
                    ("primal", "dual", "gap", "normalized_gap", "dual_feasibility"))
            and cert["normalized_gap"] <= settings["normalized_gap_tol"]
            and cert["dual_feasibility"] <= 1 + 1e-12)


def verify_evaluation(audit, evaluation, refs, first, second, truth, config, identifier):
    eligible = bool(accuracy_conditions(refs["sklearn"], config["references"])
                    and accuracy_conditions(refs["skglm"], config["references"]))
    audit.check(evaluation.get("accuracy_eligible") is eligible, f"Accuracy eligibility mismatch: {identifier}")
    audit.check(evaluation.get("reference_accepted") == refs["reference_accepted"], f"Consensus/evaluation flag mismatch: {identifier}")
    reasons = evaluation.get("qualification_failure_reasons", {})
    audit.check(bool(reasons.get("accuracy")) == (not eligible), f"Accuracy failure-reason coverage mismatch: {identifier}")
    audit.check(bool(reasons.get("consensus")) == (not refs["reference_accepted"]), f"Consensus failure-reason coverage mismatch: {identifier}")
    expected_accuracy_reasons = []
    for method in ("sklearn", "skglm"):
        row = refs[method]; cert = row["certificate"]; failed = []
        if not all(np.isfinite(cert.get(name, np.nan)) for name in ("primal", "dual", "gap", "normalized_gap", "dual_feasibility")):
            failed.append("nonfinite_objective_or_certificate")
        if not np.isfinite(cert.get("normalized_gap", np.nan)) or cert["normalized_gap"] > config["references"]["normalized_gap_tol"]:
            failed.append("normalized_gap_above_tolerance")
        if not np.isfinite(cert.get("dual_feasibility", np.nan)) or cert["dual_feasibility"] > 1+1e-12:
            failed.append("dual_not_feasible")
        if not np.isfinite(row.get("elapsed_sec", np.nan)) or row["elapsed_sec"] > config["references"]["time_limit"]:
            failed.append("observed_budget_exceeded")
        if row.get("status") != "converged":
            failed.append("library_result_not_common_converged_status")
        expected_accuracy_reasons.extend(method+":"+reason for reason in failed)
        method_reasons = reasons.get("by_reference", {}).get(method, {})
        audit.check(set(method_reasons.get("reasons", [])) == set(failed)
                    and method_reasons.get("eligible") == (not failed)
                    and method_reasons.get("status") == row.get("status"), f"Per-reference qualification reason mismatch: {identifier}/{method}")
    audit.check(set(reasons.get("accuracy", [])) == set(expected_accuracy_reasons), f"Accuracy reason stratification mismatch: {identifier}")
    expected_consensus_reasons = list(expected_accuracy_reasons)
    all_consensus = eligible
    delta_values = list(dict.fromkeys([config["primary_delta"]] + config["delta_sensitivity"]))
    expected_scores = {}
    for tau in config["support_thresholds"]:
        stored = evaluation["scores_by_tau"][str(tau)]
        left, lf, lm, ls = counts_and_fraction(first, truth, tau)
        right, rf, rm, rs = counts_and_fraction(second, truth, tau)
        for method, expected in (("sklearn", left), ("skglm", right)):
            for name, value in expected.items():
                if isinstance(value, float):
                    good = np.isclose(stored[method].get(name, np.nan), value, rtol=1e-12, atol=1e-12)
                else:
                    good = stored[method].get(name) == value
                audit.check(good, f"Score mismatch: {identifier}/tau{tau}/{method}/{name}")
        difference = abs(lf-rf)
        exact = {"numerator": difference.numerator, "denominator": difference.denominator}
        audit.check(stored["absolute_f1_difference_exact"] == exact
                    and stored["raw_absolute_f1_difference"] == float(difference),
                    f"Exact Fraction F1 difference mismatch: {identifier}/tau{tau}")
        audit.check(stored["raw_signed_f1_difference_skglm_minus_sklearn"] == float(rf-lf),
                    f"Signed F1 difference mismatch: {identifier}/tau{tau}")
        audit.check(stored["eligible_absolute_f1_difference"] == (float(difference) if eligible else None),
                    f"Ineligible target was zero-filled, omitted or contrast is wrong: {identifier}/tau{tau}")
        flags = {str(delta): bool(difference > Fraction(str(delta))) if eligible else None for delta in delta_values}
        audit.check(stored["delta_exceeded"] == flags, f"Strict rational delta event mismatch: {identifier}/tau{tau}")
        audit.check(stored["support_symmetric_difference_indices"] == np.flatnonzero(lm != rm).tolist()
                    and stored["support_symmetric_difference_count"] == int(np.sum(lm != rm))
                    and stored["sign_difference_count"] == int(np.sum(ls != rs)),
                    f"Variable/sign identity disagreement mismatch: {identifier}/tau{tau}")
        comparison = refs["by_support_threshold"][str(tau)]
        audit.check(comparison["same_support"] == bool(np.array_equal(lm, rm))
                    and comparison["same_thresholded_signs"] == bool(np.array_equal(ls, rs)),
                    f"Raw reference support/sign agreement mismatch: {identifier}/tau{tau}")
        all_consensus &= bool(np.array_equal(lm, rm) and np.array_equal(ls, rs))
        if not np.array_equal(lm, rm):
            expected_consensus_reasons.append(f"support_disagreement_tau={tau}")
        if not np.array_equal(ls, rs):
            expected_consensus_reasons.append(f"thresholded_sign_disagreement_tau={tau}")
        expected_scores[str(tau)] = flags
    coef_delta = float(np.max(np.abs(first-second)))
    pa, pb = refs["sklearn"]["certificate"]["primal"], refs["skglm"]["certificate"]["primal"]
    objective_delta = abs(pa-pb)/max(1., abs(pa), abs(pb))
    all_consensus &= (coef_delta <= config["references"]["coef_linf_agreement_tol"]
                      and objective_delta <= config["references"]["normalized_objective_agreement_tol"])
    if coef_delta > config["references"]["coef_linf_agreement_tol"]:
        expected_consensus_reasons.append("coefficient_linf_disagreement")
    if objective_delta > config["references"]["normalized_objective_agreement_tol"]:
        expected_consensus_reasons.append("normalized_objective_disagreement")
    audit.check(set(reasons.get("consensus", [])) == set(expected_consensus_reasons),
                f"Consensus failure-reason stratification mismatch: {identifier}")
    audit.check(np.isclose(coef_delta, refs["coefficient_linf_difference"], rtol=1e-12, atol=1e-14)
                and np.isclose(objective_delta, refs["normalized_primal_difference"], rtol=1e-12, atol=1e-15),
                f"Stored coefficient/objective agreement magnitudes differ: {identifier}")
    audit.check(bool(refs["reference_accepted"]) == bool(all_consensus), f"Consensus qualification contradicts artifacts: {identifier}")
    primary = evaluation["scores_by_tau"][str(config["support_threshold"])]
    audit.check(evaluation["primary_absolute_f1_difference"] == primary["eligible_absolute_f1_difference"]
                and evaluation["primary_delta_exceeded"] == primary["delta_exceeded"][str(config["primary_delta"])],
                f"Primary endpoint fields differ from primary tau: {identifier}")
    return eligible, expected_scores


def _verify_stage(stage, reconstruct, config_path):
    started = perf_counter(); audit = Audit()
    base = ROOT / "results/reference_audit_v1" / stage
    config_path = Path(config_path) if config_path else ROOT / "configs/reference_audit_v1.json"
    config = audit.read(config_path); config_hash = sha(config_path)
    targets = expected_grid(config, stage)
    env, completion = audit.read(base/"environment.json"), audit.read(base/"completion.json")
    audit.check(audit.read(base/"config.json") == config and audit.read(base/"effective_config.json") == config,
                "Configuration snapshots do not match frozen input values")
    signature, provenance = audit_signature(config_path)
    for name, row in (("environment", env), ("completion", completion)):
        audit.check(row.get("numerical_signature") == signature and row.get("config_sha256") == config_hash
                    and row.get("protocol_sha256") == provenance["protocol_sha256"], f"{name} provenance mismatch")
    audit.artifact(base/"config_original.json", config_hash)
    audit.artifact(base/"protocol_snapshot.md", provenance["protocol_sha256"])
    audit.artifact(ROOT/config["protocol_path"], provenance["protocol_sha256"])
    complete = True
    for directory, suffix in (("datasets", "json"), ("references", "json"), ("reference_coefficients", "npz"), ("evaluation", "json")):
        found = {path.stem for path in (base/directory).glob("*."+suffix)}
        good = found == set(targets)
        complete &= good
        audit.check(good, f"{directory}: missing={sorted(set(targets)-found)}, unexpected={sorted(found-set(targets))}")
    audit.check(completion.get("expected_targets") == completion.get("recorded_targets") == len(targets)
                and completion.get("complete_grid") is True, "Completion count/grid mismatch")
    audit.check(completion.get("expected_public_reference_calls") == completion.get("stored_public_reference_calls") == 2*len(targets),
                "Stored public reference-call count mismatch")
    source_records = {}; source_data = {}; designs = []; evaluations = {}; reference_records = []
    counts = Counter(); statuses = Counter(); exact_events = {}; per_target = []
    if reconstruct:
        from lasso_study.audit_data_v1 import load_audit_source, make_audit_design
    for source in config["sources"]:
        source_records[source] = audit.read(base/"sources"/(source+".json"))
        audit.check(source_records[source].get("source") == source
                    and source_records[source].get("numerical_signature") == signature
                    and source_records[source].get("config_sha256") == config_hash,
                    f"Source pool metadata provenance mismatch: {source}")
        if reconstruct:
            source_data[source] = load_audit_source(source, source_seed=config["sources"][source]["source_seed"])
            audit.check(array_sha256(source_data[source]["X_pool"]) == source_records[source]["source_pool_X_sha256"],
                        f"Regenerated source pool differs: {source}", "numerical")
    for identifier, params in targets.items():
        try:
            ds_path = base/"datasets"/(identifier+".json"); rp = base/"references"/(identifier+".json")
            ep = base/"evaluation"/(identifier+".json"); cp = base/"reference_coefficients"/(identifier+".npz")
            ds, refs, evaluation = audit.read(ds_path), audit.read(rp), audit.read(ep)
            data_hash = hashlib.sha256(json.dumps(ds["hashes"], sort_keys=True).encode()).hexdigest()
            for name, record in (("dataset", ds), ("references", refs), ("evaluation", evaluation)):
                audit.check(record.get("target_id") == identifier and record.get("params") == params
                            and all(record.get(key) == value for key, value in params.items()), f"{name} target identity mismatch: {identifier}")
                audit.check(record.get("numerical_signature") == record.get("source_signature") == signature
                            and record.get("config_sha256") == config_hash and record.get("dataset_hash") == data_hash,
                            f"{name} data/source/config provenance mismatch: {identifier}")
                if name != "references":
                    audit.check(record.get("protocol_sha256") == provenance["protocol_sha256"], f"{name} protocol hash mismatch: {identifier}")
            audit.check(evaluation["input_hashes"] == {"dataset_sha256": sha(ds_path), "reference_json_sha256": sha(rp),
                                                       "reference_coefficients_sha256": sha(cp)}, f"Evaluation input hashes differ: {identifier}")
            audit.artifact(cp, refs["coefficient_sha256"])
            with np.load(cp, allow_pickle=False) as arrays:
                first, second, truth = (arrays[name].copy() for name in ("ref_sklearn", "ref_skglm", "w_true"))
            audit.check(all(array.dtype == np.float64 and array.shape == (ds["p"],) and np.isfinite(array).all()
                            for array in (first, second, truth)), f"Invalid coefficient arrays: {identifier}")
            audit.check(array_sha256(truth) == ds["hashes"]["w_true"], f"Truth coefficient hash mismatch: {identifier}")
            audit.check(ds["source_original_response_used"] is False and ds["sparsity"] == 5
                        and ds["n"] == (config["news_n"] if params["source"] == "news" else config["gene_n"]),
                        f"Sample scope/source-label usage mismatch: {identifier}")
            eligible, events = verify_evaluation(audit, evaluation, refs, first, second, truth, config, identifier)
            exact_events[identifier] = events; evaluations[identifier] = evaluation
            modern = refs["skglm"]; attempts = modern["attempts"]
            audit.check(modern["attempt_count"] == len(attempts) <= 3
                        and refs["actual_library_fit_attempts"] == 1+len(attempts), f"Actual attempt count mismatch: {identifier}")
            audit.check([row["library_tol"] for row in attempts] == config["references"]["modern"]["library_tolerances"][:len(attempts)]
                        and all(row["initialization"].startswith("zero") for row in attempts), f"Tolerance-ladder/zero initialization mismatch: {identifier}")
            audit.check(modern["elapsed_sec"]+1e-12 >= sum(row["fit_elapsed_sec"] for row in attempts),
                        f"Refit time not included cumulatively: {identifier}")
            if reconstruct:
                data = make_audit_design(config, **params, source_data=source_data[params["source"]])
                data["X"] = np.asfortranarray(data["X"])
                audit.check(data["metadata"]["hashes"] == ds["hashes"] and np.array_equal(data["w_true"], truth)
                            and data["metadata"]["lambda"] == ds["lambda"] == refs["lambda"],
                            f"Regenerated data/response/lambda differ: {identifier}", "numerical")
                for method, coef in (("sklearn", first), ("skglm", second)):
                    cert = certificate(data["X"], data["y"], coef, ds["lambda"])
                    compare_certificate(audit, cert, refs[method]["certificate"], identifier+"/"+method)
                    if refs[method]["status"] == "converged":
                        audit.check(cert["normalized_gap"] <= config["references"]["normalized_gap_tol"]+5e-13,
                                    f"Reference converged status contradicts regenerated certificate: {identifier}/{method}", "numerical")
                    counts["reference_certificates_recomputed"] += 1
                predicted = float(np.linalg.norm(data["X"]@(first-second))/np.sqrt(len(data["y"])))
                audit.check(np.isclose(predicted, evaluation["prediction_difference_rmse"], rtol=1e-9, atol=1e-12),
                            f"Prediction difference mismatch: {identifier}", "numerical")
                counts["designs_regenerated"] += 1
            counts["targets_checked"] += 1; counts["accuracy_eligible_targets"] += eligible
            counts["consensus_accepted_targets"] += bool(refs["reference_accepted"])
            counts["actual_library_fit_attempts"] += refs["actual_library_fit_attempts"]
            statuses.update(refs[method]["status"] for method in ("sklearn", "skglm"))
            designs.append(ds); reference_records.append(refs)
            per_target.append({"target_id": identifier, "accuracy_eligible": eligible,
                               "reference_accepted": refs["reference_accepted"], "fraction_events": events})
        except Exception as error:
            audit.errors["structure"].append(f"Cannot verify {identifier}: {type(error).__name__}: {error}")
            audit.warnings.append(traceback.format_exc())
    by_seed = defaultdict(list)
    for ds in designs:
        by_seed[ds["source"], ds["seed"]].append(ds)
    for (source, seed), rows in by_seed.items():
        for name in ("X", "support_order", "ordered_signs", "noise", "standard_noise", "sampled_rows", "retained_columns"):
            audit.check(all(name in row["hashes"] for row in rows) and len({row["hashes"].get(name) for row in rows}) == 1,
                        f"Broken or missing source/seed pairing for {source}/{seed}/{name}")
        for amplitude in config["amplitudes"]:
            paired = [row for row in rows if row["amplitude"] == amplitude]
            audit.check(len({row["hashes"]["y"] for row in paired}) == 1
                        and len({row["hashes"]["w_true"] for row in paired}) == 1,
                        f"Changing c changed y/truth for {source}/{seed}/{amplitude}")
    gene_hashes = {ds["hashes"]["X"] for ds in designs if ds["source"] == "gene"}
    audit.check(len(gene_hashes) <= 1, "Gene design changed across conditional response seeds")
    planned_seed_states = []
    delta_values = list(dict.fromkeys([config["primary_delta"]]+config["delta_sensitivity"]))
    for source, settings in config["sources"].items():
        for seed in settings[stage+"_seeds"]:
            identifiers = [identifier for identifier, params in targets.items() if params["source"] == source and params["seed"] == seed]
            present = [evaluations[identifier] for identifier in identifiers if identifier in evaluations]
            states = {}
            for delta in delta_values:
                flags = [exact_events[identifier][str(config["support_threshold"])][str(delta)]
                         for identifier in identifiers if identifier in exact_events]
                states[str(delta)] = ("event" if any(flag is True for flag in flags) else "no_event"
                                      if len(flags) == len(identifiers) and all(flag is False for flag in flags) else "unresolved")
            planned_seed_states.append({"source": source, "seed": seed, "planned_target_count": len(identifiers),
                                         "observed_target_count": len(present),
                                         "accuracy_eligible_targets": sum(row["accuracy_eligible"] for row in present),
                                         "primary_state": states[str(config["primary_delta"])], "states_by_delta": states})
    recorded_states = completion.get("seed_outcomes", [])
    recorded_state_map = {(row["source"], row["seed"]): row for row in recorded_states}
    expected_state_map = {(row["source"], row["seed"]): row for row in planned_seed_states}
    audit.check(len(recorded_state_map) == len(recorded_states) and recorded_state_map == expected_state_map,
                "Completion seed-level event/non-event/unresolved accounting differs")
    audit.check(completion.get("accuracy_eligible_targets") == counts["accuracy_eligible_targets"]
                and completion.get("consensus_accepted_targets") == counts["consensus_accepted_targets"]
                and completion.get("stored_actual_library_fit_attempts") == counts["actual_library_fit_attempts"],
                "Completion qualification/attempt summaries differ from raw records")
    failure_files = list((base/"failures").glob("*.json"))
    for path in failure_files:
        audit.read(path)
    structural = not audit.errors["structure"]; numerical = not audit.errors["numerical"] if reconstruct else None
    return {"stage": stage, "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "timezone": "Asia/Shanghai", "verifier_script_sha256": sha(Path(__file__)),
            "numerical_signature": signature, "config_sha256": config_hash, "protocol_sha256": provenance["protocol_sha256"],
            "complete_grid": complete, "structural_passed": structural, "numerical_recomputed": reconstruct,
            "numerical_passed": numerical, "all_passed": bool(complete and structural and reconstruct and numerical),
            "expected_targets": len(targets), "statistics": dict(counts), "reference_status_counts": dict(statuses),
            "seed_outcomes": planned_seed_states, "per_target": per_target,
            "preserved_exception_records": len(failure_files), "checks": dict(audit.checks),
            "issues": audit.errors, "warnings": audit.warnings, "input_provenance": audit.inputs,
            "verification_wall_sec": perf_counter()-started,
            "boundaries": ["No reference is required to pass consensus; accuracy-ineligible and reference-uncertain outputs remain in the grid.",
                           "Default verification regenerates stored responses and recomputes two certificates per target; it does not perform new optimizer fits, SVD or grouping.",
                           "Fraction comparisons are checked independently from integer TP/FP/FN; exact equality at delta is not an event.",
                           "This is an artifact/numerical audit, not a certificate of exact support, uniqueness, statistical recovery or novelty."]}


def verify_stage(stage, *, reconstruct=True, config_path=None):
    stage = stage.removeprefix("external_")
    if stage not in {"pilot", "confirm"}:
        raise ValueError("Expected pilot or confirm")
    with threadpool_limits(limits=1):
        return _verify_stage(stage, reconstruct, config_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+", choices=("pilot", "confirm", "external_pilot", "external_confirm"))
    parser.add_argument("--config", type=Path, default=ROOT/"configs/reference_audit_v1.json")
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args(); passed = True
    for raw_stage in args.stages:
        stage = raw_stage.removeprefix("external_")
        try:
            report = verify_stage(stage, reconstruct=not args.structure_only, config_path=args.config)
        except Exception as error:
            report = {"stage": stage, "all_passed": False, "complete_grid": False, "structural_passed": False,
                      "numerical_recomputed": False, "issues": {"fatal": [f"{type(error).__name__}: {error}"]},
                      "traceback": traceback.format_exc(), "verifier_script_sha256": sha(Path(__file__))}
        dump(ROOT/"results/reference_audit_v1"/stage/"verification.json", report)
        passed &= bool(report.get("structural_passed") if args.structure_only else report["all_passed"])
        print(json.dumps({name: report.get(name) for name in ("stage", "all_passed", "statistics", "issues")}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
