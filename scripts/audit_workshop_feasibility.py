#!/usr/bin/env python3
"""Audit existing v2 support observations without running an optimizer.

Reads stored JSON certificates/metrics and final NPZ coefficients. It never
regenerates a design, computes a gradient, or fits a model. Agreement among
observed numerical solutions is not a certificate of exact Lasso support.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASIC = ("ista", "fista", "cd")
KEYS = ("seed", "rho", "sparsity", "layout", "lambda_factor")
TRANSITIONS = (("1e-2_to_1e-6", .01, 1e-6), ("1e-4_to_1e-6", .0001, 1e-6))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"n": int(len(values)), "median": float(np.median(values)) if len(values) else None,
            "q25": float(np.quantile(values, .25)) if len(values) else None,
            "q75": float(np.quantile(values, .75)) if len(values) else None,
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None}


def key(row):
    return tuple(row[name] for name in KEYS)


def hit(run, target):
    for label, observation in run.get("threshold_hits", {}).items():
        if float(label) == target:
            return observation
    return next((row for row in run["history"] if row["normalized_gap"] <= target), None)


def count_summary(rows):
    valid = [row for row in rows if row["available"]]
    return {
        "n_runs": len(rows), "n_available": len(valid), "n_missing": len(rows)-len(valid),
        "different_checkpoints": sum(row["different_checkpoint"] for row in valid),
        "tp_fp_fn_counts_changed": sum(row["counts_changed"] for row in valid),
        "f1_increased": sum(row["f1_direction"] == "increased" for row in valid),
        "f1_decreased": sum(row["f1_direction"] == "decreased" for row in valid),
        "f1_unchanged": sum(row["f1_direction"] == "unchanged" for row in valid),
        "exact_support_flag_changed": sum(row["exact_support_flag_changed"] for row in valid),
        "additional_observed_time_sec": describe([row["additional_observed_time_sec"] for row in valid]),
        "additional_fraction_of_tight_threshold_time": describe(
            [row["additional_fraction_of_tight_threshold_time"] for row in valid]),
        "absolute_f1_change": describe([abs(row["f1_change"]) for row in valid]),
    }


def audit(root):
    folder = root / "results/v2/synthetic"
    config_path = folder / "config.json"
    config = read(config_path)
    params = config["synthetic"]
    threshold = config["support_threshold"]
    methods = config["methods"]
    expected = set(product(params["seeds"], params["rhos"], params["sparsities"],
                           params["layouts"], params["lambda_factors"]))
    designs = {}
    inputs = {config_path}
    env_path = folder / "environment.json"
    environment = read(env_path)
    inputs.add(env_path)
    for path in sorted((folder / "datasets").glob("*.json")):
        data = read(path)
        design_key = (data["seed"], data["rho"], data["sparsity"], data["layout"])
        if design_key in designs:
            raise RuntimeError(f"Duplicate design metadata: {design_key}")
        designs[design_key] = data
        inputs.add(path)
    problems = defaultdict(dict)
    checkpoint_rows = []
    schema = Counter()
    source_signatures, config_signatures = set(), set()
    reference_gaps = []
    statuses = Counter()
    for path in sorted((folder / "runs").glob("*.json")):
        run = read(path)
        problem = key(run)
        method = run["method"]
        if problem not in expected or method not in methods or method in problems[problem]:
            raise RuntimeError(f"Unexpected or duplicate run: {path}")
        inputs.add(path)
        source_signatures.add(run["numerical_signature"])
        config_signatures.add(run["config_sha256"])
        statuses[run["status"]] += 1
        reference_gaps.append(run["reference"]["normalized_gap"])
        schema["reference_records"] += 1
        schema["references_with_coef_or_support"] += any(
            name in run["reference"] for name in ("coef", "coefficients", "support", "selected_indices"))
        schema["runs_with_internal_checkpoint_coef_or_support"] += any(
            any(name in row for name in ("coef", "coefficients", "support", "selected_indices"))
            for row in run["history"])
        design = designs[problem[:4]]
        if run["dataset_hash"] != design["hashes"]["X"] + design["hashes"]["y"]:
            raise RuntimeError(f"Run/design hash mismatch: {path}")
        truth = set(design["support"])
        coefficient_path = folder / "coefficients" / (run["run_id"] + ".npz")
        if sha(coefficient_path) != run["coefficient_sha256"]:
            raise RuntimeError(f"Stored coefficient hash mismatch: {coefficient_path}")
        with np.load(coefficient_path, allow_pickle=False) as archive:
            selected = set(np.flatnonzero(np.abs(archive["coef"]) > threshold).tolist())
        inputs.add(coefficient_path)
        final = run["final"]
        observed = (len(selected & truth), len(selected - truth), len(truth - selected), len(selected))
        recorded = tuple(final[name] for name in ("tp", "fp", "fn", "nnz"))
        if observed != recorded:
            raise RuntimeError(f"Final coefficient/support metrics disagree: {path}")
        problems[problem][method] = {"support": selected, "normalized_gap": final["normalized_gap"],
                                    "f1": final["f1"], "tp": final["tp"], "fp": final["fp"], "fn": final["fn"]}
        if method not in BASIC:
            continue
        if run.get("threshold_curve_comparable") is not True:
            raise RuntimeError(f"Basic run is not marked trajectory-comparable: {path}")
        for transition, early_target, late_target in TRANSITIONS:
            early, late = hit(run, early_target), hit(run, late_target)
            row = dict(zip(KEYS, problem), run_id=run["run_id"], method=method,
                       transition=transition, available=early is not None and late is not None)
            if row["available"]:
                if early["normalized_gap"] > early_target or late["normalized_gap"] > late_target:
                    raise RuntimeError(f"Stored threshold observation fails its target: {path}")
                change = late["f1"] - early["f1"]
                early_counts = {name: early[name] for name in ("tp", "fp", "fn", "nnz")}
                late_counts = {name: late[name] for name in ("tp", "fp", "fn", "nnz")}
                extra = late["time"] - early["time"]
                if extra < -1e-12:
                    raise RuntimeError(f"Threshold observation time decreased: {path}")
                row.update(early_target=early_target, tight_target=late_target,
                           early_iteration=early["iteration"], tight_iteration=late["iteration"],
                           different_checkpoint=early["iteration"] != late["iteration"],
                           counts_changed=early_counts != late_counts,
                           early_counts=early_counts, tight_counts=late_counts,
                           early_f1=early["f1"], tight_f1=late["f1"], f1_change=change,
                           f1_direction="increased" if change > 1e-12 else "decreased" if change < -1e-12 else "unchanged",
                           exact_support_flag_changed=early["exact_support"] != late["exact_support"],
                           additional_observed_time_sec=extra,
                           additional_fraction_of_tight_threshold_time=extra / late["time"])
            checkpoint_rows.append(row)
    if set(problems) != expected or any(set(rows) != set(methods) for rows in problems.values()):
        raise RuntimeError("The original synthetic v2 Cartesian grid is incomplete")
    if len(source_signatures) != 1 or len(config_signatures) != 1:
        raise RuntimeError("Multiple numerical source/configuration signatures in the original grid")
    if source_signatures != {environment["v2_numerical_signature"]}:
        raise RuntimeError("Run/environment numerical signature mismatch")
    if config_signatures != {environment["v2_config_sha256"]}:
        raise RuntimeError("Run/environment original configuration signature mismatch")

    agreement_rows = []
    pair_counts = {f"{first}_vs_{second}": {"n_problems": 0, "different_support_sets": 0,
                                         "symmetric_difference_total": 0, "symmetric_difference_max": 0}
                   for first, second in combinations(methods, 2)}
    for problem, rows in sorted(problems.items()):
        differences = []
        for first, second in combinations(methods, 2):
            difference = len(rows[first]["support"] ^ rows[second]["support"])
            differences.append(difference)
            pair = pair_counts[f"{first}_vs_{second}"]
            pair["n_problems"] += 1
            pair["different_support_sets"] += difference > 0
            pair["symmetric_difference_total"] += difference
            pair["symmetric_difference_max"] = max(pair["symmetric_difference_max"], difference)
        agreement_rows.append(dict(zip(KEYS, problem), all_observed_final_supports_identical=max(differences) == 0,
                                   maximum_pairwise_symmetric_difference=max(differences),
                                   observed_final_normalized_gaps={method: row["normalized_gap"] for method, row in rows.items()},
                                   observed_final_f1={method: row["f1"] for method, row in rows.items()}))
    transitions = {}
    for label, _, _ in TRANSITIONS:
        rows = [row for row in checkpoint_rows if row["transition"] == label]
        transitions[label] = {
            "overall": count_summary(rows),
            "by_method": {method: count_summary([row for row in rows if row["method"] == method]) for method in BASIC},
            "count_change_cases": [row for row in rows if row["available"] and row["counts_changed"]],
        }
    manifest = [{"path": str(path.relative_to(root)), "sha256": sha(path), "bytes": path.stat().st_size}
                for path in sorted(inputs)]
    combined = hashlib.sha256("\n".join(entry["path"] + "\0" + entry["sha256"] for entry in manifest).encode()).hexdigest()
    return {
        "audit_type": "read-only existing-v2 workshop feasibility audit; zero new optimizer runs",
        "audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "timezone": "Asia/Shanghai",
        "audit_script_sha256": sha(Path(__file__)),
        "input_provenance": {"combined_manifest_sha256": combined,
                             "manifest_digest_rule": "SHA256 of newline-joined relative_path + NUL + file_sha256, sorted by path",
                             "input_files": len(manifest), "input_bytes": sum(entry["bytes"] for entry in manifest),
                             "original_v2_numerical_signatures": sorted(source_signatures),
                             "original_v2_config_sha256": sorted(config_signatures),
                             "frozen_config_snapshot_sha256": sha(config_path), "manifest": manifest},
        "support_threshold": threshold, "methods": methods, "run_status_counts": dict(statuses),
        "n_problems": len(problems), "n_final_coefficient_files": sum(len(rows) for rows in problems.values()),
        "final_support_agreement": {
            "problems_with_all_observed_final_supports_identical": sum(row["all_observed_final_supports_identical"] for row in agreement_rows),
            "problems_with_any_observed_final_support_disagreement": sum(not row["all_observed_final_supports_identical"] for row in agreement_rows),
            "pairwise": pair_counts, "by_problem": agreement_rows,
            "interpretation": "Comparison is between observed numerical final supports at the fixed threshold, not against a proved exact minimizer support."},
        "checkpoint_transitions": transitions,
        "storage_feasibility": {
            "reference_schema_counts": dict(schema), "reference_normalized_gap": describe(reference_gaps),
            "available": ["true support and signed coefficients in design metadata", "all final numerical coefficient vectors",
                          "basic-method first-observed checkpoint times, gaps, TP/FP/FN/F1 and sign/group summaries", "high-accuracy reference objective/certificate"],
            "missing_in_current_run_records": ["reference coefficient vector or selected support indices",
                                               "coefficient vector or selected support indices at intermediate checkpoints"],
            "requires_new_numerical_work": ["obtain and store high-accuracy reference coefficients/support, with posterior checks",
                                           "replay selected basic trajectories with coefficient/support snapshots at the three target checkpoints"],
            "cannot_be_inferred": ["TP/FP/FN agreement does not prove support-set agreement because variable identities may swap",
                                   "agreement of all final numerical supports does not prove equality to exact Lasso support",
                                   "a numerical reference objective/gap alone does not supply reference coefficient identities"],
        },
        "limitations": ["Existing observations were already inspected; subsequent hypotheses are exploratory and need independent confirmation.",
                        "Times are observed checkpoint times, hence upper bounds on threshold crossing; no optimizer was rerun to refine them.",
                        "No conclusion about novelty, exact minimizer support, statistical guarantees or universal support stability is made."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root)
    destination = args.output or root / "results/workshop/feasibility.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"output": str(destination), "n_problems": result["n_problems"],
                      "identical_observed_final_supports": result["final_support_agreement"]["problems_with_all_observed_final_supports_identical"],
                      "checkpoint_changes": {label: value["overall"] for label, value in result["checkpoint_transitions"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
