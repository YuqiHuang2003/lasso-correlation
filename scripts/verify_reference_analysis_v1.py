#!/usr/bin/env python3
"""Independently check saved external-audit analysis, without optimizer calls.

Recount supports from NPZ, regenerate fixed X/y for fitted/primal diagnostics,
and independently rebuild seed/oracle summaries. Group partitions and direct
anchor residuals are checked, but the all-pair search for absent groups is not
rerun. This script does not import the analysis implementation under review.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
import hashlib
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
sys.path.insert(0, str(ROOT / "src"))
from lasso_study.audit_data_v1 import load_audit_source, make_audit_design


class Audit:
    def __init__(self):
        self.errors, self.inputs, self.checks = [], {}, 0

    def check(self, condition, message):
        self.checks += 1
        if not bool(condition):
            self.errors.append(message)

    def track(self, path):
        path = Path(path)
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = str(path.relative_to(ROOT))
        self.check(relative not in self.inputs or self.inputs[relative] == value, "Input changed while checking: "+relative)
        self.inputs[relative] = value
        return value

    def read(self, path):
        self.track(path)
        return json.loads(Path(path).read_text())

    def equal(self, actual, expected, label, atol=2e-14):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict), label+": expected a mapping")
            if isinstance(actual, dict):
                for key, value in expected.items():
                    self.check(key in actual, label+": missing "+str(key))
                    if key in actual:
                        self.equal(actual[key], value, label+"/"+str(key), atol)
        elif isinstance(expected, (list, tuple)):
            self.check(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label+": list length differs")
            if isinstance(actual, (list, tuple)) and len(actual) == len(expected):
                for index, value in enumerate(expected):
                    self.equal(actual[index], value, label+"/"+str(index), atol)
        elif isinstance(expected, (float, np.floating)):
            self.check(actual is not None and math.isclose(float(actual), float(expected), rel_tol=5e-11, abs_tol=atol), label+": value differs")
        else:
            self.check(actual == expected, label+": value differs")


def score(selected, truth):
    tp = int(np.count_nonzero(selected & truth)); fp = int(np.count_nonzero(selected & ~truth))
    fn = int(np.count_nonzero(~selected & truth)); denominator = 2*tp+fp+fn
    exact = Fraction(2*tp, denominator) if denominator else Fraction(0)
    return {"tp": tp, "fp": fp, "fn": fn, "nnz": int(selected.sum()), "support_error": fp+fn, "f1": float(exact)}, exact


def contrast(first, second, truth, tau, eligible, deltas):
    selected_a, selected_b = np.abs(first) > tau, np.abs(second) > tau
    a, exact_a = score(selected_a, truth); b, exact_b = score(selected_b, truth)
    delta = abs(exact_a-exact_b)
    return {"sklearn": a, "skglm": b, "raw_absolute_f1_difference": float(delta),
            "eligible_absolute_f1_difference": float(delta) if eligible else None,
            "delta_fraction_numerator": delta.numerator, "delta_fraction_denominator": delta.denominator,
            "support_symmetric_difference": int(np.count_nonzero(selected_a != selected_b)),
            "sign_difference": int(np.count_nonzero(np.where(selected_a, np.sign(first), 0.) != np.where(selected_b, np.sign(second), 0.))),
            "delta_exceeded": {str(d): bool(delta > Fraction(str(d))) if eligible else None for d in deltas}}


def qualify(refs, a, b, config):
    settings = config["references"]; by_ref = {}; reasons = []
    for method in ("sklearn", "skglm"):
        row, failures = refs[method], []
        certificate = row["certificate"]
        if not all(np.isfinite(certificate.get(key, np.nan)) for key in ("primal", "dual", "gap", "normalized_gap", "dual_feasibility")):
            failures.append("nonfinite_objective_or_certificate")
        if not np.isfinite(certificate.get("normalized_gap", np.nan)) or certificate["normalized_gap"] > settings["normalized_gap_tol"]:
            failures.append("normalized_gap_above_tolerance")
        if not np.isfinite(certificate.get("dual_feasibility", np.nan)) or certificate["dual_feasibility"] > 1+1e-12:
            failures.append("dual_not_feasible")
        if not np.isfinite(row.get("elapsed_sec", np.nan)) or row["elapsed_sec"] > settings["time_limit"]:
            failures.append("observed_budget_exceeded")
        if row.get("status") != "converged":
            failures.append("library_result_not_common_converged_status")
        by_ref[method] = {"eligible": not failures, "reasons": failures, "status": row["status"],
                          "elapsed_sec": row["elapsed_sec"], "normalized_gap": certificate["normalized_gap"]}
        reasons.extend(method+":"+item for item in failures)
    consensus = list(reasons)
    if np.max(np.abs(a-b)) > settings["coef_linf_agreement_tol"]:
        consensus.append("coefficient_linf_disagreement")
    pa, pb = (refs[method]["certificate"]["primal"] for method in ("sklearn", "skglm"))
    if abs(pa-pb)/max(1., abs(pa), abs(pb)) > settings["normalized_objective_agreement_tol"]:
        consensus.append("normalized_objective_disagreement")
    for tau in config["support_thresholds"]:
        sa, sb = np.abs(a) > tau, np.abs(b) > tau
        if not np.array_equal(sa, sb):
            consensus.append(f"support_disagreement_tau={tau}")
        if not np.array_equal(np.where(sa, np.sign(a), 0.), np.where(sb, np.sign(b), 0.)):
            consensus.append(f"thresholded_sign_disagreement_tau={tau}")
    return not reasons, not consensus, {"accuracy": reasons, "consensus": consensus, "by_reference": by_ref}


def wilson(k, n):
    z2 = 1.959963984540054**2
    center = (k+z2/2)/(n+z2)
    radius = math.sqrt(z2*(k*(n-k)/n+z2/4))/(n+z2)
    return [max(0., center-radius), min(1., center+radius)]


def expected_seed_summaries(rows, deltas):
    cells = defaultdict(list)
    for row in rows:
        cells[row["source"], row["seed"], row["tau"]].append(row)
    events = []
    for (source, seed, tau), group in sorted(cells.items()):
        if {(r["amplitude"], r["c"]) for r in group} != set(product(("strong", "mixed_weak"), (.5, 1.))):
            raise RuntimeError("Non-unique seed condition grid")
        for delta in deltas:
            eligible = [r for r in group if r["accuracy_eligible"]]
            state = "event" if any(r["delta_exceeded"][str(delta)] for r in eligible) else "no_event" if len(eligible) == 4 else "undecidable"
            events.append({"source": source, "seed": seed, "tau": tau, "delta": delta, "state": state,
                           "n_targets": 4, "n_eligible": len(eligible)})
    summary = []
    for source, tau, delta in sorted({(r["source"], r["tau"], r["delta"]) for r in events}):
        group = [r for r in events if (r["source"], r["tau"], r["delta"]) == (source, tau, delta)]
        counts = Counter(r["state"] for r in group); n = len(group)
        summary.append({"source": source, "tau": tau, "delta": delta, "n_seed_blocks": n,
                        "n_event": counts["event"], "n_no_event": counts["no_event"], "n_undecidable": counts["undecidable"],
                        "observed_identification_range": [counts["event"]/n, (counts["event"]+counts["undecidable"])/n],
                        "range_is_confidence_interval": False,
                        "seed_event_wilson95_if_fully_observed": wilson(counts["event"], n) if not counts["undecidable"] else None})
    return events, summary


def expected_oracles(rows, tolerance):
    cells = defaultdict(list)
    for row in rows:
        cells[row["source"], row["seed"], row["amplitude"], row["tau"]].append(row)
    result = []
    for key, group in sorted(cells.items()):
        record = dict(zip(("source", "seed", "amplitude", "tau"), key))
        if not all(r["accuracy_eligible"] for r in group):
            record.update(state="undecidable", sklearn_preferred_c=None, skglm_preferred_c=None, has_any_tie=None, both_tied=None)
        else:
            preferred = {}
            for method in ("sklearn", "skglm"):
                maximum = max(r[method]["f1"] for r in group)
                preferred[method] = sorted(r["c"] for r in group if maximum-r[method]["f1"] <= tolerance)
            same = preferred["sklearn"] == preferred["skglm"]
            strict = all(len(value) == 1 for value in preferred.values())
            record.update(state="same_set" if same else "strict_singleton_reversal" if strict else "tie_involved_difference",
                          sklearn_preferred_c=preferred["sklearn"], skglm_preferred_c=preferred["skglm"],
                          has_any_tie=any(len(value) > 1 for value in preferred.values()), both_tied=all(len(value) > 1 for value in preferred.values()))
        result.append(record)
    return result


def group_mapping(groups, X, tolerance, audit, label):
    p = X.shape[1]; indices = []; assignment = np.full(p, -1, dtype=np.int64); scales = np.zeros(p)
    sizes = []
    for k, group in enumerate(groups):
        columns = group["columns_0based"]; coefficients = group["member_scales_from_anchor"]
        anchor = group["anchor_column_0based"]
        audit.check(bool(columns) and columns[0] == anchor and columns == sorted(columns), label+": anchor/order")
        audit.check(len(columns) == len(coefficients) == len(group["relative_l2_residuals_to_anchor"]), label+": member lengths")
        for column, coefficient, saved_residual in zip(columns, coefficients, group["relative_l2_residuals_to_anchor"]):
            indices.append(column); assignment[column] = k; scales[column] = coefficient
            if column == anchor:
                audit.equal(coefficient, 1., label+": anchor scale")
                audit.equal(saved_residual, 0., label+": anchor residual")
            else:
                actual = np.linalg.norm(X[:, column]-coefficient*X[:, anchor])/np.linalg.norm(X[:, column])
                audit.check(actual <= tolerance*(1+1e-9), label+": direct residual exceeds tolerance")
                audit.equal(saved_residual, float(actual), label+": direct residual", atol=1e-15)
        sizes.append(len(columns))
    audit.check(sorted(indices) == list(range(p)), label+": incomplete/duplicate partition")
    return assignment, scales, np.asarray(sizes)


def verify_stage(stage, config, audit, source_cache):
    base = ROOT / "results/reference_audit_v1" / stage
    analysis = audit.read(base / "analysis.json"); verification = audit.read(base / "verification.json")
    audit.check(analysis["integrity_verified"] and verification["all_passed"] and verification["numerical_recomputed"], stage+": verified status")
    audit.equal(analysis["analysis_script_sha256"], audit.track(ROOT / "scripts/analyze_reference_audit_v1.py"), stage+": analysis source")
    for relative, expected_hash in analysis["input_provenance"].items():
        audit.equal(audit.track(ROOT / relative), expected_hash, stage+": provenance "+relative)
    expected = {(source, seed, amplitude, c) for source, definition in config["sources"].items()
                for seed, amplitude, c in product(definition[stage+"_seeds"], config["amplitudes"], config["c_values"])}
    taus = config["support_thresholds"]; deltas = list(dict.fromkeys([config["primary_delta"]]+config["delta_sensitivity"]))
    raw = {(r["source"], r["seed"], r["amplitude"], r["c"], r["tau"]): r for r in analysis["raw_scores"]}
    audit.check(len(raw) == len(analysis["raw_scores"]) == len(expected)*len(taus), stage+": raw score grid")
    audit.check(set(raw) == {(*key, tau) for key, tau in product(expected, taus)}, stage+": registered conditions")
    grouped = {(r["target_id"], r["tau"], r["grouping_tolerance"]): r for r in analysis["group_scores"]}
    comparisons = {r["target_id"]: r for r in analysis["reference_comparisons"]}
    audit.check(analysis["all_group_evaluations_completed"], stage+": group evaluations omitted")
    recomputed = []; mappings = {}; profiles = {}; comparison_rows = []
    for key in sorted(expected):
        first_row = raw[(*key, taus[0])]; identifier = first_row["target_id"]
        refs = audit.read(base / "references" / (identifier+".json"))
        ds = audit.read(base / "datasets" / (identifier+".json"))
        npz = base / "reference_coefficients" / (identifier+".npz"); audit.track(npz)
        with np.load(npz, allow_pickle=False) as saved:
            a, b, truth = (saved[name].copy() for name in ("ref_sklearn", "ref_skglm", "w_true"))
        eligible, accepted, reasons = qualify(refs, a, b, config)
        audit.equal(refs["reference_accepted"], accepted, identifier+": consensus qualification")
        params = dict(zip(("source", "seed", "amplitude", "c"), key))
        source = params["source"]
        if source not in source_cache:
            source_cache[source] = load_audit_source(source)
        data = make_audit_design(config, **params, source_data=source_cache[source])
        audit.equal(data["metadata"]["hashes"], ds["hashes"], identifier+": regenerated hashes")
        audit.check(np.array_equal(data["w_true"], truth), identifier+": saved truth")
        X = data["X"]; difference = a-b; fitted = X @ difference
        primal_a, primal_b = (refs[method]["certificate"]["primal"] for method in ("sklearn", "skglm"))
        comp = {"coefficient_linf_difference": float(np.max(np.abs(difference))), "coefficient_l2_difference": float(np.linalg.norm(difference)),
                "fitted_difference_rmse": float(np.linalg.norm(fitted)/np.sqrt(len(X))), "fitted_difference_linf": float(np.max(np.abs(fitted))),
                "l1_norm_absolute_difference": float(abs(np.abs(a).sum()-np.abs(b).sum())),
                "primal_absolute_difference": abs(primal_a-primal_b),
                "primal_normalized_difference": abs(primal_a-primal_b)/max(1., abs(primal_a), abs(primal_b)),
                "sklearn_normalized_gap": refs["sklearn"]["certificate"]["normalized_gap"],
                "skglm_normalized_gap": refs["skglm"]["certificate"]["normalized_gap"], "qualification_failure_reasons": reasons}
        audit.equal(comparisons[identifier], comp, identifier+": coefficient/fitted/objective comparison", atol=1e-15)
        comparison_rows.append(dict(source=source, **{k:v for k,v in comp.items() if not isinstance(v, dict)}))
        for method, coefficient in (("sklearn", a), ("skglm", b)):
            residual = X @ coefficient-data["y"]
            primal = float(np.dot(residual, residual)/(2*len(X))+ds["lambda"]*np.abs(coefficient).sum())
            audit.equal(refs[method]["certificate"]["primal"], primal, identifier+"/"+method+": independently recomputed primal", atol=5e-12)
        for tau in taus:
            saved_row = raw[(*key, tau)]
            row = dict(params, target_id=identifier, n=ds["n"], p=ds["p"], tau=tau, accuracy_eligible=eligible, reference_accepted=accepted,
                       **contrast(a, b, truth != 0, tau, eligible, deltas))
            audit.equal(saved_row, row, identifier+f": raw tau={tau}")
            recomputed.append(row)
        xhash = ds["hashes"]["X"]
        if xhash not in mappings:
            groups_file = audit.read(base / "grouping" / (xhash+".json"))
            audit.check(groups_file["all_unordered_pairs_screened"] and groups_file["unique_pairs_scanned"] == ds["p"]*(ds["p"]-1)//2,
                        identifier+": reported all-pair coverage")
            audit.equal(analysis["grouping_diagnostics"][xhash], groups_file["diagnostics"], identifier+": grouping diagnostic copy")
            mappings[xhash] = {}
            for tolerance, groups in groups_file["groups_by_tolerance"].items():
                mappings[xhash][float(tolerance)] = group_mapping(groups, X, float(tolerance), audit, identifier+"/groups"+tolerance)
        for tolerance, (assignment, scales, sizes) in mappings[xhash].items():
            count = len(sizes)
            aggregate = lambda w: np.bincount(assignment, weights=w*scales, minlength=count)
            ag_a, ag_b, ag_truth = aggregate(a), aggregate(b), aggregate(truth)
            actual_counts = np.bincount(assignment, weights=(truth != 0).astype(float), minlength=count).astype(int)
            gross = np.bincount(assignment, weights=np.abs(truth*scales), minlength=count)
            for tau in taus:
                net = np.abs(ag_truth) > tau; contains_any = actual_counts > 0
                details = contrast(ag_a, ag_b, net, tau, eligible, deltas)
                details.update(evaluation_object="group_net_effect", observed_absolute_f1_difference=details["raw_absolute_f1_difference"],
                    group_count=count, singleton_groups=int(np.count_nonzero(sizes == 1)), nontrivial_groups=int(np.count_nonzero(sizes > 1)),
                    columns_in_nontrivial_groups=int(sizes[sizes > 1].sum()), lost_coordinate_identities=len(truth)-count,
                    maximum_group_size=int(sizes.max()), true_variables_in_nontrivial_groups=int(actual_counts[sizes > 1].sum()),
                    groups_with_multiple_true_variables=int(np.count_nonzero(actual_counts > 1)),
                    full_cancellation_groups=int(np.count_nonzero(contains_any & ~net)),
                    partial_or_full_cancellation_groups=int(np.count_nonzero(gross > np.abs(ag_truth)+1e-12)),
                    contains_any_true_groups=int(contains_any.sum()), effective_true_groups=int(net.sum()),
                    contains_any_scores={"sklearn": score(np.abs(ag_a)>tau, contains_any)[0], "skglm": score(np.abs(ag_b)>tau, contains_any)[0]},
                    aggregated_coefficient_linf_difference=float(np.max(np.abs(ag_a-ag_b))),
                    aggregation_changes_representation=bool(np.any(sizes > 1)))
                audit.equal(grouped[identifier, tau, tolerance], details, identifier+f": grouped tau={tau}/residual={tolerance}")
                profiles[source, tolerance] = {"group_count": count, "singletons": details["singleton_groups"], "nontrivial_groups": details["nontrivial_groups"],
                                               "maximum_group_size": details["maximum_group_size"], "lost_coordinate_identities": details["lost_coordinate_identities"],
                                               "true_variables_in_nontrivial_groups": details["true_variables_in_nontrivial_groups"], "full_cancellation_groups": details["full_cancellation_groups"]}
    audit.check(len(grouped) == len(expected)*len(taus)*3, stage+": grouped condition grid")
    events, seed_summary = expected_seed_summaries(recomputed, deltas)
    audit.equal(analysis["seed_events"], events, stage+": seed events")
    audit.equal(analysis["seed_summary"], seed_summary, stage+": three-state and Wilson summaries")
    oracle = expected_oracles(recomputed, config["oracle_c_tie_tolerance"])
    audit.equal(analysis["oracle_c_preferences"], oracle, stage+": oracle sets and ties")
    audit.equal(analysis["oracle_preference_counts"], dict(Counter(r["state"] for r in oracle)), stage+": oracle counts")
    audit.equal(analysis["oracle_any_tie_count"], sum(r["has_any_tie"] is True for r in oracle), stage+": any tie count")
    audit.equal(analysis["oracle_both_tied_count"], sum(r["both_tied"] is True for r in oracle), stage+": both tied count")
    sources = []
    for source in config["sources"]:
        rows = [r for r in recomputed if r["source"] == source and r["tau"] == config["support_threshold"]]
        eligible_rows = [r for r in rows if r["accuracy_eligible"]]
        item = {"source": source, "targets": len(rows), "independent_seed_blocks": len({r["seed"] for r in rows}),
                "accuracy_eligible_targets": len(eligible_rows), "full_consensus_targets": sum(r["reference_accepted"] for r in rows),
                "f1_different_targets": sum(r["delta_exceeded"]["0.0"] is True for r in eligible_rows),
                "primary_event_targets": sum(r["delta_exceeded"][str(config["primary_delta"])] is True for r in eligible_rows),
                "max_observed_eligible_f1_difference": max((r["eligible_absolute_f1_difference"] for r in eligible_rows), default=None),
                "n_values": sorted({r["n"] for r in rows}), "p_values": sorted({r["p"] for r in rows})}
        sources.append(item)
    audit.equal(analysis["source_summary"], sources, stage+": source summary")
    oracle_summary = []
    for source, tau in product(config["sources"], taus):
        rows = [r for r in oracle if r["source"] == source and r["tau"] == tau]
        oracle_summary.append({"source": source, "tau": tau, "comparisons": len(rows), "states": dict(Counter(r["state"] for r in rows)),
                               "any_tie": sum(r["has_any_tie"] is True for r in rows), "both_tied": sum(r["both_tied"] is True for r in rows),
                               "sklearn_preferred_sets": dict(Counter(",".join(map(str, r["sklearn_preferred_c"])) if r["sklearn_preferred_c"] is not None else "undecidable" for r in rows))})
    comparison_maxima = {source: {metric: max(r[metric] for r in comparison_rows if r["source"] == source)
                                  for metric in comparison_rows[0] if metric != "source"} for source in config["sources"]}
    return {"stage": stage, "source_summary": sources, "raw_score_rows_recounted": len(recomputed), "group_score_rows_recounted": len(grouped),
            "seed_summary": seed_summary, "oracle_summary": oracle_summary,
            "group_profiles": [{"source": source, "tolerance": tolerance, **details} for (source, tolerance), details in sorted(profiles.items())],
            "reference_comparison_maxima": comparison_maxima, "distinct_X_groupings": len(mappings)}


def main():
    started = perf_counter(); audit = Audit(); stages = []
    with threadpool_limits(limits=1):
        config = audit.read(ROOT / "configs/reference_audit_v1.json"); source_cache = {}
        for stage in ("pilot", "confirm"):
            try:
                stages.append(verify_stage(stage, config, audit, source_cache))
            except Exception:
                audit.errors.append(stage+": fatal verification exception\n"+traceback.format_exc())
    for relative, value in list(audit.inputs.items()):
        audit.check(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == value, "Input changed before completion: "+relative)
    result = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "all_passed": not audit.errors,
              "verifier_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checks": audit.checks,
              "stages": stages, "issues": audit.errors, "input_provenance": audit.inputs, "wall_sec": perf_counter()-started,
              "scope": ["Independent NPZ recount and Fraction F1/delta comparisons; no import of analysis implementation and no optimization.",
                        "Fixed generator reconstructs X/y for saved hash, fit-difference and primal checks; no reference refits.",
                        "Stored groups are checked as partitions, directly against anchors and through independent bincount aggregation; the search for absent near-proportional pairs is not independently rerun.",
                        "This checks bookkeeping/numerical summaries, not exact support, mathematical uniqueness, global solution-set bounds or novelty."]}
    output = ROOT / "results/reference_audit_v1/analysis_verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"all_passed": result["all_passed"], "checks": result["checks"], "issues": result["issues"],
                      "stages": [{k:s[k] for k in ("stage", "source_summary", "oracle_summary")} for s in stages]}, ensure_ascii=False, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
