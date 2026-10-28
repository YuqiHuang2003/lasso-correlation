#!/usr/bin/env python3
"""Verify workshop artifacts without running an optimizer.

Run only after timed experiments finish. Default validation regenerates designs
and performs certificate matrix-vector products at each reference, final point,
and first observation of each of the five gap targets. It does not fit a model,
compute an SVD, or change original results. --structure-only skips regeneration
and certificate recomputation and cannot produce an all_passed numerical audit.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter
from datetime import datetime
import hashlib
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
from run_workshop import (dump, sha, source_signature, target_parameters, target_id,
                          reference_agreement)
from run_workshop_followup import followup_signature, followup_targets, followup_target_id
from lasso_study.data import array_sha256
from lasso_study.data_v2 import make_uji
from lasso_study.workshop_data import make_workshop_design, make_real_design
from lasso_study.solvers import certificate


class Audit:
    def __init__(self):
        self.errors = {"structure": [], "numerical": []}
        self.warnings = []
        self.checks = Counter()
        self.inputs = {}

    def check(self, condition, message, category="structure"):
        self.checks[category] += 1
        if not bool(condition):
            self.errors[category].append(message)
        return bool(condition)

    def read(self, path):
        self.inputs[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())

    def artifact(self, path, expected):
        actual = sha(path)
        self.inputs[str(path.relative_to(ROOT))] = actual
        self.check(actual == expected, f"Artifact SHA256 mismatch: {path}")


def matches_params(record, params):
    return record.get("params") == params and all(record.get(key) == value for key, value in params.items())


def compare_certificate(audit, actual, saved, label):
    for name in ("primal", "dual", "gap", "raw_gap", "normalized_gap", "dual_feasibility", "kkt_inf"):
        atol = 5e-13 if name == "normalized_gap" else 5e-12
        audit.check(name in saved and np.isclose(actual[name], saved.get(name, np.nan), rtol=5e-11, atol=atol),
                    f"Recomputed certificate mismatch {label}/{name}", "numerical")
    audit.check(actual["dual_feasibility"] <= 1 + 1e-12, f"Infeasible recomputed dual {label}", "numerical")


def validate_support_history(audit, run, coefs, truth, reference, accepted, config, masks):
    """Check every stored support metric, including the binary H/R partition."""
    history = run["history"]; label = run["run_id"]
    actual = np.abs(truth) > 0
    checked = 0
    for tau in config["support_thresholds"]:
        selected = np.abs(coefs) > tau
        ref = np.abs(reference) > tau
        tp = np.sum(selected & actual, axis=1)
        fp = np.sum(selected & ~actual, axis=1)
        fn = np.sum(~selected & actual, axis=1)
        nnz = np.sum(selected, axis=1)
        f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
        exact = np.all(selected == actual, axis=1)
        exact_sign = np.all(np.where(selected, np.sign(coefs), 0.) == np.sign(truth), axis=1)
        harm = np.sum((ref == actual) & (selected != actual), axis=1)
        repair = np.sum((ref != actual) & (selected == actual), axis=1)
        disagreement = np.sum(selected != ref, axis=1)
        ref_error = int(np.sum(ref != actual))
        audit.check(np.all(fp + fn == ref_error + harm - repair) and np.all(disagreement == harm + repair),
                    f"Support H/R identity failed: {label}, tau={tau}")
        for index, row in enumerate(history):
            stored = row.get("support_by_threshold", {}).get(str(tau), {})
            expected = {"tp": int(tp[index]), "fp": int(fp[index]), "fn": int(fn[index]),
                        "nnz": int(nnz[index]), "support_error": int(fp[index] + fn[index]),
                        "f1": float(f1[index]), "exact_support": bool(exact[index]), "exact_sign": bool(exact_sign[index])}
            if accepted:
                expected.update(reference_support_error=ref_error, numerical_harm=int(harm[index]),
                                numerical_repair=int(repair[index]), numerical_support_error=int(disagreement[index]),
                                same_support_as_reference=bool(disagreement[index] == 0))
            for name, value in expected.items():
                good = name in stored and (np.isclose(stored[name], value, rtol=1e-12, atol=1e-12)
                                           if name == "f1" else stored[name] == value)
                audit.check(good, f"Checkpoint support metric mismatch {label}/{index}/tau{tau}/{name}")
                if tau == config["support_threshold"]:
                    audit.check(name in row and row[name] == stored.get(name),
                                f"Top-level support metric differs from primary tau {label}/{index}/{name}")
            checked += 1
        if tau == config["support_threshold"]:
            audit.check(np.array_equal(masks["support_masks_packed"], np.packbits(selected, axis=1)),
                        f"Packed support mask mismatch: {label}")
            audit.check(np.array_equal(masks["negative_sign_masks_packed"], np.packbits(coefs < -tau, axis=1)),
                        f"Packed sign mask mismatch: {label}")
    safe_packed = masks["gap_safe_zero_masks_packed"]
    p = coefs.shape[1]
    audit.check(safe_packed.shape == (len(history), (p + 7) // 8), f"Gap Safe packed-mask shape mismatch: {label}")
    safe = np.unpackbits(safe_packed, axis=1)[:, :p].astype(bool)
    audit.check(np.all(safe[:-1] <= safe[1:]), f"Cumulative Gap Safe masks decreased: {label}")
    audit.check(all(row.get("gap_safe_zero_count") == int(mask.sum()) for row, mask in zip(history, safe)),
                f"Gap Safe zero counts differ from stored masks: {label}")
    if accepted:
        ref_selected = np.abs(reference) > config["support_threshold"]
        excluded = np.sum(safe & ref_selected, axis=1)
        audit.check(all(row.get("gap_safe_excluded_reference_nonzero_count") == int(count)
                        for row, count in zip(history, excluded)), f"Gap Safe/reference overlap count mismatch: {label}")
        if np.any(excluded):
            audit.warnings.append(f"Numerical Gap Safe excludes an observed reference nonzero in {label}; inspect margins")
    return checked


def _verify_stage(stage, *, reconstruct=True, config_path=None):
    started = perf_counter(); audit = Audit()
    base = ROOT / "results/workshop" / stage
    default = "workshop_pilot.json" if stage in {"smoke", "pilot"} else f"workshop_{stage}.json"
    original_path = Path(config_path) if config_path else ROOT / "configs" / default
    original = audit.read(original_path)
    config_hash = sha(original_path)
    config = dict(original)
    if stage == "smoke":
        config.update({name: value for name, value in original["smoke"].items() if name != "targets"})
    saved_config = audit.read(base / "config.json")
    effective = audit.read(base / "effective_config.json")
    completion = audit.read(base / "completion.json")
    env = audit.read(base / "environment.json")
    audit.check(saved_config == original and effective == config, "Frozen configuration snapshots differ from original config")
    if stage in {"smoke", "pilot"}:
        params_list = target_parameters(original, stage)
        identify = target_id
        current_signature, _ = source_signature()
    else:
        params_list = followup_targets(original, stage)
        identify = followup_target_id
        current_signature, _ = followup_signature(original_path)
    signature = env.get("numerical_signature")
    audit.check(bool(signature) and signature == current_signature, "Current frozen numerical source/dependencies differ from stage environment")
    audit.check(env.get("config_sha256") == config_hash and completion.get("config_sha256") == config_hash,
                "Environment/completion original config hashes differ")
    audit.check(completion.get("numerical_signature") == signature, "Completion/source signature mismatch")
    audit.check(all(record.get("source_signature", signature) == signature for record in (env, completion)),
                "Duplicate environment/completion source-signature fields disagree")
    targets = {identify(params): params for params in params_list}
    expected_runs = {identifier + "_" + method for identifier in targets for method in config["methods"]}
    files = {name: {path.stem for path in (base / name).glob("*." + suffix)}
             for name, suffix in (("datasets", "json"), ("references", "json"), ("reference_coefficients", "npz"),
                                  ("runs", "json"), ("trajectories", "npz"))}
    for folder in files:
        expected = expected_runs if folder in {"runs", "trajectories"} else set(targets)
        audit.check(files[folder] == expected,
                    f"{folder} grid differs: missing={sorted(expected-files[folder])}, unexpected={sorted(files[folder]-expected)}")
    audit.check(completion.get("expected_targets") == completion.get("recorded_targets") == len(targets),
                "Declared target count differs from exact grid")
    audit.check(completion.get("expected_trajectories") == completion.get("recorded_trajectories") == len(expected_runs),
                "Declared trajectory count differs from exact grid")
    statistics = Counter(); statuses = Counter(); reference_statuses = Counter(); missing_thresholds = Counter()
    source = None
    if reconstruct and stage == "real":
        source = make_uji(seed=0)
    for identifier, params in targets.items():
        try:
            ds = audit.read(base / "datasets" / (identifier + ".json"))
            refs = audit.read(base / "references" / (identifier + ".json"))
            data_hash = hashlib.sha256(json.dumps(ds["hashes"], sort_keys=True).encode()).hexdigest()
            for label, record in (("dataset", ds), ("reference", refs)):
                audit.check(record.get("target_id") == identifier and matches_params(record, params),
                            f"{label} target/params mismatch: {identifier}")
                audit.check(record.get("numerical_signature") == signature and record.get("config_sha256") == config_hash
                            and record.get("dataset_hash") == data_hash, f"{label} provenance mismatch: {identifier}")
                audit.check(record.get("source_signature", signature) == signature,
                            f"{label} duplicate source-signature field differs: {identifier}")
            rp = base / "reference_coefficients" / (identifier + ".npz")
            audit.artifact(rp, refs["coefficient_sha256"])
            with np.load(rp, allow_pickle=False) as arrays:
                ref1, ref2, truth = (arrays[name].copy() for name in ("ref_sklearn", "ref_skglm", "w_true"))
            audit.check(all(a.dtype == np.float64 and a.shape == (ds["p"],) and np.isfinite(a).all()
                            for a in (ref1, ref2, truth)), f"Invalid reference/truth arrays: {identifier}")
            audit.check(array_sha256(truth) == ds["hashes"]["w_true"], f"Stored truth hash differs from dataset: {identifier}")
            agreement = reference_agreement(refs["sklearn"], refs["skglm"], ref1, ref2,
                                            config["references"], config["support_thresholds"])
            audit.check(refs.get("reference_accepted") == agreement["reference_accepted"] and refs.get("status") == agreement["status"],
                        f"Reference acceptance contradicts stored coefficients/certificates: {identifier}")
            reference_statuses[refs["status"]] += 1
            statistics["reference_targets"] += 1
            statistics["reference_accepted_targets"] += bool(refs["reference_accepted"])
            data = None
            if reconstruct:
                if stage == "real":
                    data = make_real_design(config, params["seed"], params["layout"], params["amplitude"], source=source)
                else:
                    data = make_workshop_design(config, **params)
                data["X"] = np.asfortranarray(data["X"], dtype=np.float64)
                audit.check(data["metadata"]["hashes"] == ds["hashes"] and np.array_equal(data["w_true"], truth),
                            f"Regenerated data/truth hashes differ: {identifier}", "numerical")
                audit.check(data["metadata"]["lambda"] == ds["lambda"] == refs["lambda"],
                            f"Regenerated lambda differs: {identifier}", "numerical")
                fresh = []
                for name, coef in (("sklearn", ref1), ("skglm", ref2)):
                    cert = certificate(data["X"], data["y"], coef, ds["lambda"])
                    compare_certificate(audit, cert, refs[name]["certificate"], identifier + "/" + name)
                    if refs[name]["status"] == "converged":
                        audit.check(cert["normalized_gap"] <= config["references"]["normalized_gap_tol"] + 5e-13,
                                    f"Reference converged status contradicts recomputed gap: {identifier}/{name}", "numerical")
                    fresh.append(dict(refs[name], certificate=cert))
                    statistics["reference_certificates_recomputed"] += 1
                fresh_agreement = reference_agreement(fresh[0], fresh[1], ref1, ref2, config["references"], config["support_thresholds"])
                audit.check(fresh_agreement["reference_accepted"] == refs["reference_accepted"],
                            f"Reference acceptance changes on regenerated data: {identifier}", "numerical")
                statistics["designs_regenerated"] += 1
            for method in config["methods"]:
                run_id = identifier + "_" + method
                run = audit.read(base / "runs" / (run_id + ".json"))
                audit.check(run.get("run_id") == run_id and run.get("target_id") == identifier
                            and run.get("method") == method and matches_params(run, params), f"Run identity mismatch: {run_id}")
                audit.check(run.get("numerical_signature") == signature and run.get("config_sha256") == config_hash
                            and run.get("dataset_hash") == data_hash, f"Run provenance mismatch: {run_id}")
                audit.check(run.get("source_signature", signature) == signature,
                            f"Run duplicate source-signature field differs: {run_id}")
                audit.check(run.get("reference_info", {}).get("reference_accepted") == refs["reference_accepted"]
                            and run.get("reference_info", {}).get("path") == f"references/{identifier}.json"
                            and run.get("reference_info", {}).get("target_id") == identifier
                            and run.get("reference_info", {}).get("status") == refs["status"],
                            f"Run reference link/acceptance mismatch: {run_id}")
                cp = base / "trajectories" / (run_id + ".npz")
                audit.artifact(cp, run["trajectory_sha256"])
                with np.load(cp, allow_pickle=False) as arrays:
                    stored = {name: arrays[name].copy() for name in arrays.files}
                coefs = stored["coef_history"]; history = run["history"]
                audit.check(coefs.dtype == np.float64 and coefs.shape == (len(history), ds["p"]) and np.isfinite(coefs).all(),
                            f"Invalid coefficient history shape/dtype/values: {run_id}")
                audit.check(np.array_equal(stored["ref_sklearn"], ref1) and np.array_equal(stored["ref_skglm"], ref2)
                            and np.array_equal(stored["w_true"], truth), f"Embedded references/truth differ from independent artifacts: {run_id}")
                audit.check(np.array_equal(stored["iterations"], [h["iteration"] for h in history])
                            and np.array_equal(stored["times"], [h["time"] for h in history]), f"History iteration/time alignment mismatch: {run_id}")
                audit.check(np.all(np.diff(stored["iterations"]) > 0) and np.all(np.diff(stored["times"]) >= 0),
                            f"Nonincreasing iteration or decreasing time: {run_id}")
                audit.check(np.array_equal(coefs[-1], stored["final_coef"]) and run["final"] == history[-1], f"Final checkpoint mismatch: {run_id}")
                audit.check(all(h.get("checkpoint_index") == i for i, h in enumerate(history)), f"Checkpoint index alignment mismatch: {run_id}")
                audit.check(run.get("status") in {"converged", "max_iter", "time_limit"}, f"Unknown basic-solver status: {run_id}")
                certificate_fields = ("primal", "dual", "gap", "raw_gap", "normalized_gap", "dual_feasibility", "kkt_inf")
                audit.check(all(all(name in h and np.isfinite(h[name]) for name in certificate_fields)
                                and h["dual_feasibility"] <= 1 + 1e-12 and h["gap"] >= 0 and h["normalized_gap"] >= 0
                                and np.isclose(h["normalized_gap"], h["gap"] / max(1., abs(h["primal"]), abs(h["dual"])),
                                               rtol=1e-12, atol=1e-14) for h in history),
                            f"Invalid or internally inconsistent stored certificates: {run_id}")
                if run["status"] == "converged":
                    audit.check(history[-1]["normalized_gap"] <= config["solver"]["tol"],
                                f"Converged status contradicts saved final gap: {run_id}")
                audit.check(np.isfinite(run["elapsed_sec"]) and history[-1]["time"] <= run["elapsed_sec"]
                            and run["diagnostics_sec"] <= run["elapsed_sec"], f"Invalid timing components: {run_id}")
                statistics["support_threshold_checkpoint_rows_checked"] += validate_support_history(
                    audit, run, coefs, truth, ref1, refs["reference_accepted"], config, stored)
                selected_indices = {len(history) - 1}
                for target_gap in config["gap_thresholds"]:
                    indices = [i for i, h in enumerate(history) if h["normalized_gap"] <= target_gap]
                    saved_hit = run["threshold_hits"].get(str(target_gap))
                    if indices:
                        selected_indices.add(indices[0])
                        audit.check(saved_hit == history[indices[0]], f"First threshold observation mismatch: {run_id}/{target_gap}")
                    else:
                        audit.check(saved_hit is None, f"Unreached threshold incorrectly has an observation: {run_id}/{target_gap}")
                        missing_thresholds[f"{method}:{target_gap}"] += 1
                if reconstruct:
                    for index in sorted(selected_indices):
                        cert = certificate(data["X"], data["y"], coefs[index], ds["lambda"])
                        compare_certificate(audit, cert, history[index], f"{run_id}/{index}")
                        statistics["trajectory_certificates_recomputed"] += 1
                        if index == len(history)-1 and run["status"] == "converged":
                            audit.check(cert["normalized_gap"] <= config["solver"]["tol"] + 5e-13,
                                        f"Converged run fails recomputed tolerance: {run_id}", "numerical")
                statistics["trajectories_checked"] += 1
                statuses[run["status"]] += 1
        except Exception as error:
            audit.errors["structure"].append(f"Cannot validate target {identifier}: {type(error).__name__}: {error}")
            audit.warnings.append(traceback.format_exc())
    failures = [audit.read(path) for path in sorted((base / "failures").glob("*.json"))]
    if failures:
        audit.warnings.append(f"Preserved {len(failures)} exception records; expected-grid and artifact checks determine current completeness")
    structural_passed = not audit.errors["structure"]
    numerical_passed = not audit.errors["numerical"] if reconstruct else None
    return {
        "stage": stage, "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "timezone": "Asia/Shanghai",
        "verifier_script_sha256": sha(Path(__file__)), "numerical_signature": signature, "config_sha256": config_hash,
        "complete_grid": all(files[folder] == (expected_runs if folder in {"runs", "trajectories"} else set(targets)) for folder in files),
        "expected_targets": len(targets), "expected_trajectories": len(expected_runs),
        "structural_passed": structural_passed, "numerical_recomputed": reconstruct, "numerical_passed": numerical_passed,
        "all_passed": bool(structural_passed and reconstruct and numerical_passed),
        "checks": dict(audit.checks), "statistics": dict(statistics), "trajectory_status_counts": dict(statuses),
        "reference_status_counts": dict(reference_statuses), "unreached_threshold_counts": dict(missing_thresholds),
        "preserved_exception_records": len(failures), "issues": audit.errors, "warnings": audit.warnings,
        "input_provenance": audit.inputs, "verification_wall_sec": perf_counter()-started,
        "resource_scope": "Single-thread validation; default regeneration/certificate matvecs are new validation cost, not optimization reruns or original benchmark timing.",
        "boundaries": ["Nonconverged trajectories and uncertain numerical references are preserved, not required to pass the research endpoint.",
                       "Only the two reference certificates, final iterates and five first-hit checkpoints are recomputed; support/H-R/masks are checked at every stored checkpoint.",
                       "Gap Safe packed masks/counts are checked structurally; the sphere test is not numerically recomputed at all checkpoints.",
                       "This validates saved numerical evidence and bookkeeping; it does not certify exact Lasso support, future support stability, statistical recovery, or novelty."]}


def verify_stage(stage, *, reconstruct=True, config_path=None):
    """Single-thread public API; no optimizer or experiment rerun is called."""
    with threadpool_limits(limits=1):
        return _verify_stage(stage, reconstruct=reconstruct, config_path=config_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+", choices=("smoke", "pilot", "confirm", "real"))
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--config", type=Path, help="Single-stage original input configuration override")
    args = parser.parse_args()
    if args.config is not None and len(args.stages) != 1:
        parser.error("--config requires exactly one stage")
    passed = True
    with threadpool_limits(limits=1):
        for stage in args.stages:
            try:
                report = verify_stage(stage, reconstruct=not args.structure_only, config_path=args.config)
            except Exception as error:
                report = {"stage": stage, "all_passed": False, "complete_grid": False,
                          "structural_passed": False, "numerical_recomputed": False,
                          "issues": {"fatal": [f"{type(error).__name__}: {error}"]}, "traceback": traceback.format_exc(),
                          "verifier_script_sha256": sha(Path(__file__))}
            dump(ROOT / "results/workshop" / stage / "verification.json", report)
            passed &= bool(report.get("structural_passed") if args.structure_only else report["all_passed"])
            print(json.dumps({name: report.get(name) for name in
                             ("stage", "all_passed", "complete_grid", "structural_passed", "numerical_recomputed", "statistics", "issues")}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
