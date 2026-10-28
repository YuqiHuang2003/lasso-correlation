#!/usr/bin/env python3
"""Independent saved-artifact verification; never runs an optimizer.

Numerical verification must run after the timed column-permutation experiment.
--self-check uses only a tiny synthetic matrix. --structure-only never loads
external design matrices and cannot produce numerical all_passed=True.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from fractions import Fraction
import hashlib
import importlib.metadata
from itertools import combinations, product
import json
import math
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import traceback
from zoneinfo import ZoneInfo

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/".modern_deps")]
from lasso_study.data import array_sha256
from lasso_study.solvers import certificate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


class Audit:
    def __init__(self):
        self.checks = Counter(); self.issues = []; self.inputs = {}; self.warnings = []

    def check(self, condition, label, kind="structure"):
        self.checks[kind] += 1
        if not bool(condition):
            self.issues.append({"kind": kind, "message": label})

    def track(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise RuntimeError("Missing/outside-project evidence: "+str(path))
        key = str(path.relative_to(ROOT)); actual = sha(path)
        self.check(key not in self.inputs or self.inputs[key] == actual, "Input changed while reading: "+key)
        self.inputs[key] = actual
        return actual

    def read(self, path):
        self.track(path)
        return json.loads(Path(path).read_text())

    def equal(self, actual, expected, label, atol=5e-12, rtol=5e-12, kind="structure"):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict), label+": mapping expected", kind)
            if isinstance(actual, dict):
                for key, value in expected.items():
                    self.check(key in actual, label+": missing "+str(key), kind)
                    if key in actual:
                        self.equal(actual[key], value, label+"/"+str(key), atol, rtol, kind)
        elif isinstance(expected, (list, tuple)):
            self.check(isinstance(actual, (list, tuple)) and len(actual) == len(expected), label+": sequence differs", kind)
            if isinstance(actual, (list, tuple)) and len(actual) == len(expected):
                for index, value in enumerate(expected):
                    self.equal(actual[index], value, label+f"/{index}", atol, rtol, kind)
        elif isinstance(expected, (float, np.floating)):
            self.check(actual is not None and abs(float(actual)-float(expected)) <= atol+rtol*max(abs(float(actual)), abs(float(expected))),
                       label+": numeric difference", kind)
        else:
            self.check(actual == expected, label+": differs", kind)


def independent_targets(config):
    """Derive the declared selection without importing the new data/runner API."""
    targets = []
    for seed, layout, amplitude in product(config["uji_seeds"], config["layouts"], config["amplitudes"]):
        targets.append({"target_id": f"uji_seed{seed}_{layout}_{amplitude}", "source": "uji", "seed": seed,
                        "layout": layout, "amplitude": amplitude, "c": 1.})
    for source in ("news", "gene"):
        for seed, amplitude, c in product(config["external_seeds"][source], config["amplitudes"], config["external_c"]):
            targets.append({"target_id": f"{source}_seed{seed}_c{c:g}_{amplitude}", "source": source, "seed": seed,
                            "c": float(c), "amplitude": amplitude, "layout": None})
    if len(targets) != 56 or len({row["target_id"] for row in targets}) != 56:
        raise ValueError("Registered selection must contain the 56 distinct old targets")
    return targets


def independent_schedule(config, targets):
    by_source = {source: [row for row in targets if row["source"] == source] for source in config["sources"]}
    schedule = []
    for order_index, order_id in enumerate(config["execution_order_ids"]):
        sources = config["sources"][order_index % 3:]+config["sources"][:order_index % 3]
        for position in range(max(map(len, by_source.values()))):
            for source in sources:
                if position >= len(by_source[source]):
                    continue
                target = by_source[source][position]
                methods = config["methods"] if (position+order_index) % 2 == 0 else config["methods"][::-1]
                for method in methods:
                    schedule.append({"target_id": target["target_id"], "source": source, "order_id": order_id,
                                     "method": method, "is_qc_repeat": order_id == "identity_repeat"})
    return schedule


def independently_generated_permutations(p, seeds, replacement_start):
    orders = {"identity": np.arange(p, dtype=np.int64)}; used = {tuple(orders["identity"])}
    actual_seeds = {"identity": None}; replacement = replacement_start
    for index, planned_seed in enumerate(seeds):
        seed = planned_seed
        while True:
            candidate = np.random.default_rng(seed).permutation(p).astype(np.int64)
            if tuple(candidate) not in used:
                break
            seed = replacement; replacement += 1
        key = f"perm_{index}"; orders[key] = candidate; actual_seeds[key] = seed; used.add(tuple(candidate))
    return orders, actual_seeds


def independent_score(coef, truth, tau):
    selected, true = np.abs(coef) > tau, np.abs(truth) > 0
    tp = int(np.count_nonzero(selected & true)); fp = int(np.count_nonzero(selected & ~true))
    fn = int(np.count_nonzero(~selected & true)); denominator = 2*tp+fp+fn
    exact = Fraction(2*tp, denominator) if denominator else Fraction(0)
    signs = np.where(selected, np.sign(coef), 0.)
    return {"tp": tp, "fp": fp, "fn": fn, "nnz": int(np.count_nonzero(selected)),
            "f1": float(exact), "f1_exact": {"numerator": exact.numerator, "denominator": exact.denominator},
            "support_indices": np.flatnonzero(selected).tolist(), "thresholded_signs": signs.astype(int).tolist()}, exact, selected, signs


def eligibility(cert, elapsed, config):
    settings = config["references"]
    finite = all(np.isfinite(cert.get(key, np.nan)) for key in
                 ("primal", "dual", "gap", "raw_gap", "normalized_gap", "dual_feasibility", "kkt_inf"))
    accurate = bool(finite and cert["normalized_gap"] <= settings["normalized_gap_tol"]
                    and cert["dual_feasibility"] <= 1+1e-12)
    within = bool(np.isfinite(elapsed) and 0 <= elapsed <= settings["time_limit"])
    return accurate, within, accurate and within


def observed_events(coefficients, truth, tau, expected_count, deltas):
    """Only eligible outputs enter observed ranges; partial agreement is unresolved."""
    scored = [independent_score(coef, truth, tau) for coef in coefficients]
    support_change = any(not np.array_equal(a[2], b[2]) for a, b in combinations(scored, 2))
    complete = len(scored) == expected_count
    support_state = "event" if support_change else "no_event" if complete else "undecidable"
    exact_range = max((row[1] for row in scored), default=Fraction(0))-min((row[1] for row in scored), default=Fraction(0))
    states = {str(delta): "event" if len(scored) >= 2 and exact_range > Fraction(str(delta))
              else "no_event" if complete else "undecidable" for delta in deltas}
    return {"eligible_order_count": len(scored), "support_state": support_state,
            "f1_range": float(exact_range) if scored else None,
            "f1_range_exact": {"numerator": exact_range.numerator, "denominator": exact_range.denominator} if scored else None,
            "f1_states_by_delta": states,
            "maximum_pairwise_support_difference": max((int(np.count_nonzero(a[2] != b[2]))
                                                         for a, b in combinations(scored, 2)), default=0) if scored else None}


def close_value(first, second, config):
    first = np.asarray(first); second = np.asarray(second)
    return bool(first.shape == second.shape and np.isfinite(first).all() and np.isfinite(second).all()
                and np.max(np.abs(first-second), initial=0.) <= config["invariance_atol"]
                + config["invariance_rtol"]*max(np.max(np.abs(first), initial=0.), np.max(np.abs(second), initial=0.)))


def coefficient_comparison(first, second, X, y, lam, taus, config):
    """Historical/repeat QC in original coordinates, without optimizing."""
    left = certificate(X, y, first, lam); right = certificate(X, y, second, lam)
    linf = float(np.max(np.abs(first-second)))
    objective_delta = abs(left["primal"]-right["primal"])/max(1., abs(left["primal"]), abs(right["primal"]))
    by_tau = {}
    for tau in taus:
        am, bm = np.abs(first) > tau, np.abs(second) > tau
        an, bn = np.where(am, np.sign(first), 0.), np.where(bm, np.sign(second), 0.)
        by_tau[str(tau)] = {"same_support": bool(np.array_equal(am, bm)),
                            "same_thresholded_signs": bool(np.array_equal(an, bn)),
                            "support_symmetric_difference": int(np.count_nonzero(am != bm)),
                            "thresholded_sign_difference": int(np.count_nonzero(an != bn))}
    passed = (linf <= config["references"]["coef_linf_agreement_tol"]
              and objective_delta <= config["references"]["normalized_objective_agreement_tol"]
              and all(row["same_support"] and row["same_thresholded_signs"] for row in by_tau.values()))
    return {"coefficient_linf_difference": linf, "normalized_primal_difference": objective_delta,
            "prediction_delta_rmse": float(np.linalg.norm(X@(first-second))/np.sqrt(len(y))),
            "by_support_threshold": by_tau, "passed": bool(passed)}


def frozen_target_evidence(audit, config, targets, reconstruct):
    """Read old evidence and optionally regenerate it using the frozen generators."""
    uji_config = audit.read(ROOT/config["uji_config_path"])
    external_config = audit.read(ROOT/config["external_config_path"])
    audit.check(config["uji_seeds"] == list(range(300, 310)), "Full old UJI seed range must be retained")
    for source in ("news", "gene"):
        selected = sorted(external_config["sources"][source]["confirm_seeds"])[:2]
        audit.check(config["external_seeds"][source] == selected, "External selection must use first two frozen confirm seeds: "+source)
    audit.check(config["references"] == uji_config["references"] == external_config["references"],
                "Solver settings must equal both frozen predecessors")
    sources, x_cache, result = {}, {}, {}
    if reconstruct:
        from lasso_study.data_v2 import make_uji
        from lasso_study.workshop_data import make_real_design
        from lasso_study.audit_data_v1 import load_audit_source, make_audit_design
        sources["uji"] = make_uji(seed=0)
        audit.track(ROOT/"data/benchmarks/ujiindoorloc/trainingData.csv")
        for source in ("news", "gene"):
            sources[source] = load_audit_source(source, source_seed=0)
            audit.track(ROOT/sources[source]["metadata"]["input_path"])
    for target in targets:
        identifier, source = target["target_id"], target["source"]
        base = ROOT/config["baseline_paths"][source]
        ds_path = base/"datasets"/(identifier+".json"); ref_path = base/"references"/(identifier+".json")
        cp = base/"reference_coefficients"/(identifier+".npz")
        ds, refs = audit.read(ds_path), audit.read(ref_path)
        audit.check(audit.track(cp) == refs["coefficient_sha256"], identifier+": historical coefficient hash")
        ds_hash = hashlib.sha256(json.dumps(ds["hashes"], sort_keys=True).encode()).hexdigest()
        audit.check(ds["dataset_hash"] == refs["dataset_hash"] == ds_hash, identifier+": old data/reference link")
        audit.check(ds["lambda"] == refs["lambda"], identifier+": old lambda/reference link")
        with np.load(cp, allow_pickle=False) as arrays:
            coefficients = {name: arrays[key].copy() for name, key in
                            (("sklearn", "ref_sklearn"), ("skglm", "ref_skglm"), ("truth", "w_true"))}
        audit.check(all(coef.dtype == np.float64 and coef.shape == (ds["p"],) and np.isfinite(coef).all()
                        for coef in coefficients.values()), identifier+": finite old coefficient dimensions")
        audit.check(array_sha256(coefficients["truth"]) == ds["hashes"]["w_true"], identifier+": old truth hash")
        item = {"target": target, "old_metadata": ds, "old_references": refs,
                "historical_coefficients": coefficients, "origin": {"dataset": ds_path, "references": ref_path, "coefficients": cp}}
        if reconstruct:
            if source == "uji":
                data = make_real_design(uji_config, target["seed"], target["layout"], target["amplitude"], source=sources[source])
            else:
                data = make_audit_design(external_config, target["seed"], source, target["amplitude"], target["c"], source_data=sources[source])
            audit.check(data["metadata"]["hashes"] == ds["hashes"], identifier+": regenerated complete hash dictionary", "numerical")
            audit.check(data["metadata"]["lambda"] == ds["lambda"], identifier+": regenerated lambda", "numerical")
            input_key = "source_csv_sha256" if source == "uji" else "source_input_sha256"
            audit.check(data["metadata"][input_key] == ds[input_key], identifier+": unchanged entire raw source file", "numerical")
            audit.check(data["metadata"]["generator_source_sha256"] == ds["generator_source_sha256"],
                        identifier+": unchanged frozen data generator", "numerical")
            audit.check(np.array_equal(data["w_true"], coefficients["truth"]), identifier+": regenerated truth", "numerical")
            for name in ("X", "y", "w_true"):
                audit.check(array_sha256(data[name]) == ds["hashes"][name], identifier+": regenerated "+name+" hash", "numerical")
            xhash = ds["hashes"]["X"]
            if xhash not in x_cache:
                x_cache[xhash] = data["X"]
            data["X"] = x_cache[xhash]; item["data"] = data
        result[identifier] = item
    for source, count in (("uji", 10), ("news", 2), ("gene", 1)):
        observed = {row["old_metadata"]["hashes"]["X"] for row in result.values() if row["target"]["source"] == source}
        audit.check(len(observed) == count, f"{source}: distinct-X count must equal {count}")
    grouped = defaultdict(list)
    for row in result.values():
        grouped[row["target"]["source"], row["target"]["seed"]].append(row)
    for (source, seed), rows in grouped.items():
        audit.check(len(rows) == 4 and len({row["old_metadata"]["hashes"]["X"] for row in rows}) == 1,
                    f"{source}/{seed}: four paired targets must share X")
        if source in ("news", "gene"):
            for amplitude in config["amplitudes"]:
                paired = [row for row in rows if row["target"]["amplitude"] == amplitude]
                audit.check(len({row["old_metadata"]["hashes"]["y"] for row in paired}) == 1
                            and len({row["old_metadata"]["hashes"]["w_true"] for row in paired}) == 1,
                            f"{source}/{seed}/{amplitude}: c must not change response or truth")
    return result


def read_run(audit, config, base, planned, evidence, permutation, reconstruct):
    """Check one scheduled call from its saved arrays, independently of runner helpers."""
    call_id = planned["target_id"]+"__"+planned["order_id"]+"__"+planned["method"]
    path = base/"runs"/(call_id+".json")
    if not path.exists():
        return {"call_id": call_id, "planned": planned, "present": False, "eligible": None}
    row = audit.read(path)
    for key in ("target_id", "order_id", "method", "source"):
        audit.check(row.get(key) == planned[key], call_id+": scheduled "+key)
    audit.check(row.get("call_id") == call_id, call_id+": call identifier")
    item = {"call_id": call_id, "planned": planned, "present": True, "record": row, "eligible": None}
    npz = base/"coefficients"/(call_id+".npz")
    if not row.get("coefficient_path"):
        audit.check(row.get("accuracy_eligible") in (False, None), call_id+": absent coefficients cannot be eligible")
        return item
    expected_sha = row.get("coefficient_sha256", row.get("coefficient_artifact_sha256"))
    audit.check(expected_sha is not None and audit.track(npz) == expected_sha, call_id+": coefficient artifact hash")
    with np.load(npz, allow_pickle=False) as arrays:
        coef_permuted, coef_original, saved_perm, inverse, truth = (arrays[name].copy() for name in
            ("coef_permuted", "coef_original", "permutation", "inverse_permutation", "w_true"))
    p = evidence["old_metadata"]["p"]
    audit.check(all(coef.dtype == np.float64 and coef.shape == (p,) and np.isfinite(coef).all()
                    for coef in (coef_original, coef_permuted, truth)), call_id+": finite coefficient arrays")
    audit.check(saved_perm.dtype.kind in "iu" and saved_perm.shape == (p,)
                and np.array_equal(np.sort(saved_perm), np.arange(p)), call_id+": valid integer permutation")
    audit.check(np.array_equal(saved_perm, permutation), call_id+": frozen shared permutation")
    audit.check(np.array_equal(inverse, np.argsort(saved_perm)), call_id+": inverse permutation")
    audit.check(np.array_equal(coef_original[saved_perm], coef_permuted), call_id+": exact coefficient inverse mapping")
    audit.check(np.array_equal(truth, evidence["historical_coefficients"]["truth"]), call_id+": unchanged original truth")
    for tau in config["support_thresholds"]:
        score, fraction, _, _ = independent_score(coef_original, truth, tau)
        audit.equal(row["scores"][str(tau)], score, call_id+f": tau={tau} independently counted score")
    elapsed = row.get("public_elapsed_sec", row.get("elapsed_sec", np.nan))
    raw_cert = row["certificate"]
    accurate, within, eligible = eligibility(raw_cert, elapsed, config)
    solver_status = row.get("solver_status", row.get("library_status", row.get("status")))
    eligible = bool(eligible and solver_status == "converged" and row["invariance"]["passed"])
    audit.check(row.get("accuracy_eligible") is eligible, call_id+": individual precision/budget eligibility")
    for key in ("precision_eligible", "certificate_eligible", "accuracy_passed"):
        if key in row:
            audit.check(row[key] is accurate, call_id+": independent precision flag "+key)
    for key in ("budget_eligible", "within_public_budget"):
        if key in row:
            audit.check(row[key] is within, call_id+": independent budget flag "+key)
    audit.check(row["library_status_eligible"] is bool(solver_status == "converged"), call_id+": independent library status flag")
    item.update(coef=coef_original, eligible=eligible, accurate=accurate, within_budget=within)
    if reconstruct:
        data = evidence["data"]; X, y, lam = data["X"], data["y"], evidence["old_metadata"]["lambda"]
        Xp = np.asfortranarray(X[:, saved_perm])
        expected_inputs = {"X": array_sha256(Xp), "y": array_sha256(y), "w_true": array_sha256(truth[saved_perm])}
        audit.check(row["permuted_input_hashes"] == expected_inputs, call_id+": unchanged values in permuted input coordinates", "numerical")
        original_cert = certificate(X, y, coef_original, lam)
        permuted_cert = certificate(Xp, y, coef_permuted, lam)
        for name in original_cert:
            audit.check(close_value(original_cert[name], raw_cert[name], config), call_id+": regenerated original certificate/"+name, "numerical")
            audit.check(close_value(permuted_cert[name], row["permuted_certificate"][name], config),
                        call_id+": regenerated permuted certificate/"+name, "numerical")
        invariance = (close_value(X@coef_original, Xp@coef_permuted, config)
                      and close_value(np.abs(coef_original).sum(), np.abs(coef_permuted).sum(), config)
                      and all(close_value(original_cert[key], permuted_cert[key], config) for key in original_cert))
        audit.check(row["invariance"]["passed"] is bool(invariance), call_id+": recomputed coordinate invariance", "numerical")
        if eligible:
            audit.check(original_cert["normalized_gap"] <= config["references"]["normalized_gap_tol"],
                        call_id+": independently recomputed gap must meet strict threshold", "numerical")
        item["recomputed_certificate"] = original_cert
    return item


def verify_registration(audit, config, config_path, base, targets, evidence):
    env = audit.read(base/"environment.json"); manifest = audit.read(base/"manifest.json")
    audit.check(audit.read(base/"config.json") == config, "Configuration snapshot matches frozen file")
    audit.check(audit.track(base/"protocol_snapshot.md") == audit.track(ROOT/config["protocol_path"]), "Protocol snapshot exact bytes")
    params = {row["target_id"]: {key: value for key, value in row.items() if key != "target_id"} for row in targets}
    expected_calls = []
    for index, row in enumerate(independent_schedule(config, targets)):
        expected_calls.append({"call_index": index, "call_id": row["target_id"]+"__"+row["order_id"]+"__"+row["method"],
                               "target_id": row["target_id"], "params": params[row["target_id"]], "source": row["source"],
                               "order_id": row["order_id"], "method": row["method"],
                               "permutation_order_id": "identity" if row["is_qc_repeat"] else row["order_id"],
                               "is_primary_order": not row["is_qc_repeat"], "is_identity_repeat": row["is_qc_repeat"]})
    audit.check(manifest["calls"] == expected_calls, "Entire fixed source/order/method call schedule")
    audit.check(manifest["targets"] == list(params.values()), "Frozen full target list/order")
    audit.check(manifest["expected_public_calls"] == len(expected_calls) == 784, "Full public call denominator")
    audit.check(sum(row["is_primary_order"] for row in expected_calls) == 672, "Main six-order denominator")
    audit.check(sum(row["is_identity_repeat"] for row in expected_calls) == 112, "QC-only repeat denominator")
    mh = hashlib.sha256(json.dumps(expected_calls, sort_keys=True).encode()).hexdigest()
    audit.check(mh == manifest["manifest_sha256"] == env["manifest_sha256"], "Manifest canonical content hash")
    registry = {}; permutations = {}
    for identifier, item in evidence.items():
        metadata = item["old_metadata"]; xhash = metadata["hashes"]["X"]
        if xhash in registry:
            continue
        stored = audit.read(base/"permutations"/(xhash+".json"))
        audit.check(stored["X_sha256"] == xhash and stored["p"] == metadata["p"], identifier+": permutation design identity")
        orders, actual_seeds = independently_generated_permutations(metadata["p"], config["permutation_seeds"], config["replacement_seed_start"])
        audit.check([row["order_id"] for row in stored["orders"]] == config["main_order_ids"], xhash+": exactly six main order IDs")
        for order in stored["orders"]:
            order_id = order["order_id"]; perm = np.asarray(order["permutation"], dtype=np.int64)
            inverse = np.asarray(order["inverse_permutation"], dtype=np.int64)
            audit.check(np.array_equal(perm, orders[order_id]), xhash+"/"+order_id+": independently generated permutation")
            audit.check(np.array_equal(inverse, np.argsort(perm)), xhash+"/"+order_id+": inverse")
            audit.check(order["seed"] == actual_seeds[order_id], xhash+"/"+order_id+": deterministic replacement seed")
            audit.check(order["permutation_sha256"] == array_sha256(perm) and order["inverse_sha256"] == array_sha256(inverse),
                        xhash+"/"+order_id+": actual permutation hashes")
        registry[xhash] = {"X_sha256": xhash, "p": metadata["p"], "orders": stored["orders"]}
        permutations[xhash] = orders
    audit.check(len(registry) == 13, "Exactly 13 distinct shared permutation registries")
    registry_hash = hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest()
    audit.check(registry_hash == env["permutation_registry_sha256"], "Locked actual permutation registry hash")
    payload_keys = ("base_numerical_signature", "base_provenance", "driver_sha256", "data_module_sha256", "audit_data_sha256",
                    "config_sha256", "protocol_sha256", "frozen_input_hashes", "manifest_sha256", "permutation_registry_sha256",
                    "old_input_hashes", "source_hashes")
    payload = {key: env[key] for key in payload_keys}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    audit.check(signature == env["numerical_signature"] == env["source_signature"] == manifest["numerical_signature"], "Independent numerical signature payload")
    for key, path in (("driver_sha256", ROOT/"scripts/run_column_permutation_v1.py"),
                      ("data_module_sha256", ROOT/"src/lasso_study/column_permutation_v1.py"),
                      ("audit_data_sha256", ROOT/"src/lasso_study/audit_data_v1.py"),
                      ("config_sha256", config_path), ("protocol_sha256", ROOT/config["protocol_path"])):
        audit.check(payload[key] == audit.track(path), "Locked current input "+key)
    for key, expected in payload["frozen_input_hashes"].items():
        audit.check(audit.track(ROOT/key) == expected, "Frozen input hash: "+key)
    audit.check(payload["old_input_hashes"] == payload["frozen_input_hashes"], "Old-input lock alias covers exact frozen inputs")
    for key, expected in payload["source_hashes"].items():
        audit.check(audit.track(ROOT/key) == expected, "Locked complete numerical source hash: "+key)
    previous = payload["base_provenance"]
    audit.check(hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest() == payload["base_numerical_signature"],
                "Frozen numerical-base signature")
    for key, expected in previous["source_hashes"].items():
        audit.check(audit.track(ROOT/key) == expected, "Frozen numerical helper: "+key)
    for key, expected in previous["dependency_versions"].items():
        audit.check(importlib.metadata.version(key) == expected, "Frozen dependency version: "+key)
    preparation = audit.read(base/"preparation.json"); lock = audit.read(base/"protocol_lock.json")
    audit.check(lock["numerical_signature"] == signature and lock["preparation_sha256"] == sha(base/"preparation.json"),
                "Root lock covers actual prefit artifact manifest")
    audit.check(preparation["new_fits"] == 0 and preparation["warmup_performed"] is False,
                "Prepared inputs were frozen before warmup and experimental calls")
    for relative, expected in preparation["artifact_hashes"].items():
        audit.check(audit.track(base/relative) == expected, "Prefit prepared artifact hash: "+relative)
    return expected_calls, permutations, signature


def verify_budget_ledger(audit, config, base, calls, runs, signature):
    planned = {row["call_id"]: row for row in calls}; reservations = {}; invocations = []
    for path in sorted((base/"reservations").glob("*.json")):
        row = audit.read(path); identifier = path.stem; reservations[identifier] = row
        audit.check(identifier in planned and row["call_id"] == identifier, identifier+": reservation planned identity")
        audit.check(row["numerical_signature"] == signature and row["reserved_public_reference_calls"] == 1,
                    identifier+": reservation signature and one-slot accounting")
        audit.check(row["state"] in {"started", "fit_returned", "completed", "failed_no_retry"}, identifier+": explicit reservation state")
        if row["state"] == "completed":
            audit.check((base/"runs"/(identifier+".json")).exists(), identifier+": completed reservation has a saved call")
            audit.check(row["run_sha256"] == sha(base/"runs"/(identifier+".json")), identifier+": completed call hash")
        if row["state"] == "failed_no_retry":
            audit.check(identifier in runs and runs[identifier]["record"]["accuracy_eligible"] is None,
                        identifier+": unknown failed call is retained")
    for path in sorted((base/"invocations").glob("*.json")):
        row = audit.read(path); invocations.append(row)
        audit.check(row["state"] == "finished", "No unfinished invocation can conceal consumed wall budget: "+path.name)
        audit.check(row["numerical_signature"] == signature, "Invocation frozen signature: "+path.name)
        audit.check(np.isfinite(row["wall_sec"]) and row["wall_sec"] >= 0, "Finite nonnegative invocation wall time")
    audit.check(len(reservations) <= config["maximum_public_reference_calls"] == 784, "Persistent public-slot cap")
    audit.check(sum(row["new_public_call_slots_reserved"] for row in invocations) == len(reservations), "Invocation/reservation public-slot totals")
    known_attempts = sum(item["record"].get("actual_library_fit_attempts") or 0 for item in runs.values() if item["present"])
    audit.check(sum(row["new_known_actual_library_fit_attempts"] for row in invocations) == known_attempts,
                "Actual internal library-attempt ledger equals saved completed outputs")
    for identifier, item in runs.items():
        if item["present"]:
            audit.check(identifier in reservations, identifier+": saved call requires prefit reservation")
            if "coef" in item:
                row = item["record"]; library = row["library_result"]; attempts = library["attempts"]
                count = row["actual_library_fit_attempts"]
                audit.check(count == library["attempt_count"] == len(attempts), identifier+": library attempt count")
                expected_tols = ([config["references"]["sklearn_tol"]] if row["method"] == "sklearn"
                                 else config["references"]["modern"]["library_tolerances"][:len(attempts)])
                audit.check([attempt["library_tol"] for attempt in attempts] == expected_tols,
                            identifier+": fixed tolerance ladder")
                audit.check(all(attempt["initialization"].startswith("zero") for attempt in attempts), identifier+": every attempt starts at zero")
                audit.check(row["public_elapsed_sec"]+1e-12 >= sum(attempt["fit_elapsed_sec"] for attempt in attempts),
                            identifier+": public budget includes every internal fit")
    return {"reserved_public_call_slots": len(reservations), "known_actual_library_fit_attempts": known_attempts,
            "recorded_invocations": len(invocations), "recorded_total_wall_sec": sum(row["wall_sec"] for row in invocations)}


def verify_saved_outputs(config_path, reconstruct=True):
    started = perf_counter(); audit = Audit(); config_path = Path(config_path)
    config = audit.read(config_path); base = ROOT/config["output_path"]; targets = independent_targets(config)
    evidence = frozen_target_evidence(audit, config, targets, reconstruct)
    calls, permutations, signature = verify_registration(audit, config, config_path, base, targets, evidence)
    expected_ids = {row["call_id"] for row in calls}; found_ids = {path.stem for path in (base/"runs").glob("*.json")}
    audit.check(found_ids <= expected_ids, "No unexpected call is outside the frozen manifest")
    expected_datasets = {row["target_id"] for row in targets}
    audit.check({path.stem for path in (base/"datasets").glob("*.json")} == expected_datasets, "All 56 regenerated dataset manifests retained")
    for identifier, item in evidence.items():
        saved = audit.read(base/"datasets"/(identifier+".json")); old = item["old_metadata"]
        audit.check(saved["hashes"] == old["hashes"] and saved["lambda"] == old["lambda"], identifier+": saved original data and lambda")
        audit.check(saved["numerical_signature"] == signature, identifier+": saved dataset signature")
        origin = saved["origin"]
        for origin_key, file_key in (("dataset", "dataset"), ("reference", "references"), ("coefficient", "coefficients")):
            audit.check(origin[origin_key+"_sha256"] == sha(item["origin"][file_key]), identifier+": unchanged historical "+origin_key)
    runs = {}; per_target = []; counts = Counter()
    for call in calls:
        item = evidence[call["target_id"]]; xhash = item["old_metadata"]["hashes"]["X"]
        try:
            run = read_run(audit, config, base, call, item, permutations[xhash][call["permutation_order_id"]], reconstruct)
            if run["present"]:
                row = run["record"]
                audit.check(row["numerical_signature"] == signature and row["config_sha256"] == sha(config_path), call["call_id"]+": frozen signature/config")
                audit.check(row["input_hashes"] == item["old_metadata"]["hashes"] and row["lambda"] == item["old_metadata"]["lambda"], call["call_id"]+": unchanged input problem")
                audit.check(row["is_primary_order"] is call["is_primary_order"] and row["is_identity_repeat"] is call["is_identity_repeat"], call["call_id"]+": main/QC classification")
            runs[call["call_id"]] = run
            counts["present_calls"] += run["present"]; counts["eligible_calls"] += run["eligible"] is True
            counts["saved_coefficient_vectors"] += "coef" in run
            counts["certificates_recomputed"] += 2*int(reconstruct and "coef" in run)
        except Exception as error:
            audit.check(False, f"Cannot verify {call['call_id']}: {type(error).__name__}: {error}")
            audit.warnings.append(traceback.format_exc())
            runs[call["call_id"]] = {"call_id": call["call_id"], "planned": call, "present": False, "eligible": None}
    deltas = list(dict.fromkeys([config["primary_delta"]]+config["delta_sensitivity"]))
    for target in targets:
        identifier = target["target_id"]; item = evidence[identifier]; truth = item["historical_coefficients"]["truth"]
        for method in config["methods"]:
            main = [runs[identifier+"__"+order+"__"+method] for order in config["main_order_ids"]]
            identity = runs[identifier+"__identity__"+method]; repeat = runs[identifier+"__identity_repeat__"+method]
            history_status = "undecidable"; repeat_status = "undecidable"
            if reconstruct and "coef" in identity:
                data = item["data"]
                expected = coefficient_comparison(identity["coef"], item["historical_coefficients"][method], data["X"], data["y"], item["old_metadata"]["lambda"], config["support_thresholds"], config)
                stored = identity["record"]["baseline_comparison"]
                audit.equal(stored, expected, identifier+"/"+method+": historical original-order QC", kind="numerical")
                history_status = "pass" if expected["passed"] and identity["eligible"] else "fail" if not expected["passed"] else "undecidable"
            elif "coef" in identity:
                history_status = "pass" if identity["eligible"] and identity["record"]["baseline_comparison"]["passed"] else "undecidable"
            if "coef" in identity and "coef" in repeat:
                if reconstruct:
                    data = item["data"]
                    expected = coefficient_comparison(repeat["coef"], identity["coef"], data["X"], data["y"], item["old_metadata"]["lambda"], config["support_thresholds"], config)
                    audit.equal(repeat["record"]["repeat_comparison"], expected, identifier+"/"+method+": repeated original-order QC", kind="numerical")
                    passed = expected["passed"]
                else:
                    passed = repeat["record"]["repeat_comparison"]["passed"]
                repeat_status = ("pass" if passed else "fail") if identity["eligible"] and repeat["eligible"] else "undecidable"
                audit.check(repeat["record"]["repeat_comparison"]["status"] == repeat_status, identifier+"/"+method+": repeat QC eligibility")
            for tau in config["support_thresholds"]:
                events = observed_events([row["coef"] for row in main if row["eligible"] is True], truth, tau, 6, deltas)
                qc = history_status == repeat_status == "pass"
                events.update(target_id=identifier, source=target["source"], seed=target["seed"], method=method, tau=tau,
                              historical_qc_status=history_status, repeat_qc_status=repeat_status,
                              qc_support_state=events["support_state"] if qc else "undecidable",
                              qc_f1_states_by_delta={key: value if qc else "undecidable" for key, value in events["f1_states_by_delta"].items()})
                per_target.append(events)
    budget = verify_budget_ledger(audit, config, base, calls, runs, signature)
    completion = audit.read(base/"completion.json")
    audit.check(completion["recorded_calls"] == len(found_ids), "Completion emitted-call count")
    audit.check(completion["accuracy_eligible_calls"] == counts["eligible_calls"], "Completion independent eligibility count")
    audit.check(completion["total_public_call_slots_reserved"] == budget["reserved_public_call_slots"], "Completion persistent slot count")
    complete = len(found_ids) == 784
    audit.check(completion["complete_grid"] is complete, "Completion full/partial grid flag")
    audit.check(complete or completion["budget_stop"] or completion["stage_stopped"], "Unattempted calls require explicit budget/QC stop")
    for key, expected in list(audit.inputs.items()):
        audit.check(sha(ROOT/key) == expected, "Input unchanged after verification: "+key)
    structural = not any(row["kind"] == "structure" for row in audit.issues)
    numerical = not any(row["kind"] == "numerical" for row in audit.issues) if reconstruct else None
    return {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "verifier_script_sha256": sha(Path(__file__)),
            "numerical_signature": signature, "config_sha256": sha(config_path), "complete_grid": complete,
            "structural_passed": structural, "numerical_recomputed": reconstruct, "numerical_passed": numerical,
            "all_passed": bool(structural and reconstruct and numerical), "expected_targets": 56,
            "expected_primary_public_calls": 672, "expected_qc_public_calls": 112, "expected_public_calls": 784,
            "regenerated_targets": 56 if reconstruct else 0, "distinct_designs": 13,
            "counts": dict(counts), "budget": budget, "per_target_method_tau": per_target,
            "checks": dict(audit.checks), "issues": audit.issues, "warnings": audit.warnings,
            "input_provenance": audit.inputs, "verification_wall_sec": perf_counter()-started,
            "boundaries": ["No optimizer, regrouping or plot is run by this verifier.",
                           "A valid partial experiment may pass artifact integrity while complete_grid remains false; missing planned outputs retain undecidable events.",
                           "Only the six main orders enter support/F1 events; original-order repeats are independent QC."]}


def merged_state(states, count):
    if any(value == "event" for value in states):
        return "event"
    return "no_event" if len(states) == count and all(value == "no_event" for value in states) else "undecidable"


def verify_analysis_outputs(config_path):
    """Recount saved scores/events/blocks without regenerating X or certificates."""
    started = perf_counter(); audit = Audit(); config = audit.read(config_path); base = ROOT/config["output_path"]
    numeric = audit.read(base/"verification.json"); analysis = audit.read(base/"analysis.json")
    audit.check(numeric["all_passed"] is True and numeric["verifier_script_sha256"] == sha(Path(__file__)), "Current independent numerical verification required")
    audit.check(analysis["analysis_script_sha256"] == audit.track(ROOT/"scripts/analyze_column_permutation_v1.py"), "Analysis source hash")
    audit.check(analysis["numerical_signature"] == numeric["numerical_signature"], "Analysis/numerical frozen signature")
    for relative, expected in analysis["input_provenance"].items():
        audit.check(audit.track(ROOT/relative) == expected, "Analysis input hash: "+relative)
    targets = independent_targets(config); expected_conditions = {}; expected_events = []; expected_qc = {}
    deltas = list(dict.fromkeys([config["primary_delta"]]+config["delta_sensitivity"]))
    verified_events = {(row["target_id"], row["method"], row["tau"]): row for row in numeric["per_target_method_tau"]}
    for target in targets:
        identifier = target["target_id"]
        old_base = ROOT/config["baseline_paths"][target["source"]]
        old_ds = audit.read(old_base/"datasets"/(identifier+".json"))
        old_cp = old_base/"reference_coefficients"/(identifier+".npz"); audit.track(old_cp)
        with np.load(old_cp, allow_pickle=False) as arrays:
            truth = arrays["w_true"].copy()
        grouping = audit.read(base/"groups"/(old_ds["hashes"]["X"]+".json"))
        in_group = np.zeros(len(truth), dtype=bool)
        for group in grouping["groups_by_tolerance"][str(config["grouping_tolerance"])]:
            if len(group["columns_0based"]) > 1:
                in_group[group["columns_0based"]] = True
        for method in config["methods"]:
            good = {}; all_rows = {}
            for order in config["execution_order_ids"]:
                rp = base/"runs"/(identifier+"__"+order+"__"+method+".json")
                if not rp.exists():
                    continue
                row = audit.read(rp); all_rows[order] = row
                if row.get("accuracy_eligible") is True and row.get("coefficient_path"):
                    cp = base/row["coefficient_path"]; audit.track(cp)
                    with np.load(cp, allow_pickle=False) as arrays:
                        coefficient = arrays["coef_original"].copy()
                    if order in config["main_order_ids"]:
                        good[order] = coefficient
            for tau in config["support_thresholds"]:
                verified = verified_events[identifier, method, tau]
                calculated = observed_events(list(good.values()), truth, tau, 6, deltas)
                audit.check(calculated["support_state"] == verified["support_state"]
                            and calculated["f1_states_by_delta"] == verified["f1_states_by_delta"], identifier+"/"+method+": independent raw events agree with full verifier")
                qc_pass = verified["repeat_qc_status"] == "pass"; history_pass = verified["historical_qc_status"] == "pass"
                qc = qc_pass and history_pass
                expected_qc[identifier, method] = {"repeat_qc_state": verified["repeat_qc_status"], "repeat_qc_pass": qc_pass, "history_pass": history_pass}
                signed = {order: independent_score(coef, truth, tau)[3] for order, coef in good.items()}
                masks = {order: np.abs(coef) > tau for order, coef in good.items()}
                union = np.zeros(len(truth), dtype=bool); max_sign = 0; max_coef = 0.; max_objective = 0.; identity_differences = {}
                for left, right in combinations(good, 2):
                    changed = masks[left] != masks[right]; union |= changed
                    max_sign = max(max_sign, int(np.count_nonzero(signed[left] != signed[right])))
                    max_coef = max(max_coef, float(np.max(np.abs(good[left]-good[right]))))
                    max_objective = max(max_objective, abs(all_rows[left]["certificate"]["primal"]-all_rows[right]["certificate"]["primal"]))
                    if left == "identity":
                        identity_differences[right] = int(np.count_nonzero(changed))
                fractions = [independent_score(coef, truth, tau)[1] for coef in good.values()]
                low = min(fractions) if fractions else None; high = max(fractions) if fractions else None
                span = high-low if fractions else None
                expected = {"n_eligible": len(good), "expected_main_outputs": 6,
                            "f1_min": float(low) if low is not None else None, "f1_max": float(high) if high is not None else None,
                            "observed_f1_range": float(span) if span is not None else None,
                            "range_fraction_numerator": span.numerator if span is not None else None,
                            "range_fraction_denominator": span.denominator if span is not None else None,
                            "f1_event_states": calculated["f1_states_by_delta"], "support_event_state": calculated["support_state"],
                            "qc_support_event_state": calculated["support_state"] if qc else "undecidable",
                            "qc_f1_event_states": {key: value if qc else "undecidable" for key, value in calculated["f1_states_by_delta"].items()},
                            "repeat_qc_state": verified["repeat_qc_status"], "repeat_qc_pass": qc_pass, "history_pass": history_pass,
                            "n_nontrivial_group_columns": int(np.count_nonzero(in_group)),
                            "maximum_pairwise_support_difference": calculated["maximum_pairwise_support_difference"] or 0,
                            "maximum_pairwise_sign_difference": max_sign, "maximum_pairwise_coefficient_linf_difference": max_coef,
                            "maximum_pairwise_primal_absolute_difference": max_objective,
                            "support_difference_from_identity": identity_differences,
                            "changed_coordinate_union": np.flatnonzero(union).tolist(),
                            "changed_in_nontrivial_groups": int(np.count_nonzero(union & in_group)),
                            "changed_outside_nontrivial_groups": int(np.count_nonzero(union & ~in_group))}
                expected_conditions[identifier, method, tau] = expected
                raw_events = [("support", calculated["support_state"])]+[("f1_gt_"+str(delta), calculated["f1_states_by_delta"][str(delta)]) for delta in deltas]
                for metric, raw in raw_events:
                    for layer, state in (("raw", raw), ("qc", raw if qc else "undecidable")):
                        expected_events.append({"target_id": identifier, "source": target["source"], "seed": target["seed"],
                                                "method": method, "tau": tau, "metric": metric, "layer": layer, "state": state})
    observed = {(row["target_id"], row["method"], row["tau"]): row for row in analysis["conditions"]}
    audit.check(len(observed) == len(analysis["conditions"]) == len(expected_conditions) == 224, "All target/method/tau conditions retained once")
    for key, expected in expected_conditions.items():
        audit.equal(observed[key], expected, "/".join(map(str, key))+": independent analysis condition")
    event_fields = ("target_id", "source", "seed", "method", "tau", "metric", "layer", "state")
    actual_events = {tuple(row[field] for field in event_fields) for row in analysis["target_events"]}
    expected_event_set = {tuple(row[field] for field in event_fields) for row in expected_events}
    audit.check(len(actual_events) == len(analysis["target_events"]) and actual_events == expected_event_set, "Complete raw and QC-limited target-event grid")
    block_fields = ("source", "seed", "method", "tau", "metric", "layer")
    block_members = defaultdict(list)
    for row in expected_events:
        block_members[tuple(row[field] for field in block_fields)].append(row["state"])
    expected_blocks = {key: merged_state(values, 4) for key, values in block_members.items()}
    actual_blocks = {tuple(row[field] for field in block_fields): row for row in analysis["seed_blocks"]}
    audit.check(set(actual_blocks) == set(expected_blocks) and len(actual_blocks) == len(analysis["seed_blocks"]), "Complete same-method seed-block grid")
    for key, state in expected_blocks.items():
        audit.check(actual_blocks[key]["state"] == state and actual_blocks[key]["n_targets"] == 4, str(key)+": independently merged four-target block")
    union_fields = ("source", "seed", "tau", "metric", "layer"); union_members = defaultdict(list)
    for key, state in expected_blocks.items():
        row = dict(zip(block_fields, key)); union_members[tuple(row[field] for field in union_fields)].append(state)
    expected_unions = {key: merged_state(states, 2) for key, states in union_members.items()}
    actual_unions = {tuple(row[field] for field in union_fields): row for row in analysis["method_union_seed_blocks"]}
    audit.check(set(actual_unions) == set(expected_unions), "Cross-method union is stored separately")
    for key, state in expected_unions.items():
        audit.check(actual_unions[key]["state"] == state and actual_unions[key]["n_methods"] == 2, str(key)+": independently merged two-method union")
    for key, metric in (("support_work_trigger_methods", "support"), ("primary_f1_work_trigger_methods", "f1_gt_"+str(config["primary_delta"]))):
        seed_sets = defaultdict(set)
        for values, state in expected_blocks.items():
            source, seed, method, tau, block_metric, layer = values
            if source == "uji" and tau == config["support_threshold"] and block_metric == metric and layer == "qc" and state == "event":
                seed_sets[method].add(seed)
        audit.check(analysis[key] == sorted(method for method, seeds in seed_sets.items() if len(seeds) >= 2), "Same-implementation two-seed work trigger: "+key)
    for row in analysis["repeat_qc"]:
        audit.equal(row, expected_qc[row["target_id"], row["method"]], "Independent repeated-original-order QC aggregate")
    for relative, expected in list(audit.inputs.items()):
        audit.check(sha(ROOT/relative) == expected, "Analysis evidence unchanged during verification: "+relative)
    return {"all_passed": not audit.issues, "numerical_signature": numeric["numerical_signature"],
            "verifier_script_sha256": sha(Path(__file__)), "analysis_script_sha256": analysis["analysis_script_sha256"],
            "analysis_sha256": sha(base/"analysis.json"), "independent_conditions": len(expected_conditions),
            "independent_target_events": len(expected_events), "independent_seed_blocks": len(expected_blocks),
            "independent_method_union_seed_blocks": len(expected_unions), "optimizer_calls": 0, "external_matrices_loaded": 0,
            "checks": dict(audit.checks), "issues": audit.issues, "input_provenance": audit.inputs,
            "verification_wall_sec": perf_counter()-started}


def self_check():
    X = np.asfortranarray(np.array([[1., 2., 3.], [-1., 4., 0.], [2., 0., -2.]]))
    original = np.array([.5, 0., -1.]); permutation = np.array([2, 0, 1], dtype=np.int64)
    permuted = original[permutation]; restored = np.empty_like(original); restored[permutation] = permuted
    assert np.array_equal(restored, original)
    np.testing.assert_allclose(X@original, X[:, permutation]@permuted, atol=0., rtol=0.)
    inverse = np.argsort(permutation); assert np.array_equal(permutation[inverse], np.arange(3))
    truth = np.array([1., 0., -1.]); score, exact, _, _ = independent_score(original, truth, 1e-6)
    assert (score["tp"], score["fp"], score["fn"], exact) == (2, 0, 0, Fraction(1))
    assert not (Fraction(1, 50) > Fraction("0.02"))
    assert observed_events([original], truth, 1e-6, 6, [.02])["support_state"] == "undecidable"
    assert observed_events([original]*6, truth, 1e-6, 6, [.02])["support_state"] == "no_event"
    changed = original.copy(); changed[1] = 1.
    assert observed_events([original, changed], truth, 1e-6, 6, [.02])["support_state"] == "event"
    assert observed_events([original], truth, 1e-6, 6, [.02])["f1_states_by_delta"]["0.02"] == "undecidable"
    # Exercise the saved-array adapter on tiny local fixtures, never a fit.
    config = {"references": {"normalized_gap_tol": 1e-10, "time_limit": 20.},
              "support_thresholds": [1e-6, 1e-4], "invariance_atol": 5e-12, "invariance_rtol": 5e-12}
    y = X@truth+.1; lam = .2
    cert = certificate(X, y, original, lam); perm_cert = certificate(np.asfortranarray(X[:, permutation]), y, permuted, lam)
    accurate, within, eligible = eligibility(cert, .1, config)
    planned = {"target_id": "fixture", "source": "fixture", "order_id": "perm_0", "method": "sklearn"}
    call_identifier = "fixture__perm_0__sklearn"
    class FixtureAudit(Audit):
        def track(self, path):
            return sha(path)
    with tempfile.TemporaryDirectory(prefix="column-verifier-selfcheck-") as temporary:
        base = Path(temporary); (base/"runs").mkdir(); (base/"coefficients").mkdir()
        artifact = base/"coefficients"/(call_identifier+".npz")
        np.savez_compressed(artifact, coef_original=original, coef_permuted=permuted, permutation=permutation,
                            inverse_permutation=inverse, w_true=truth)
        record = dict(planned, call_id=call_identifier, coefficient_path="coefficients/"+artifact.name,
                      coefficient_sha256=sha(artifact), certificate=cert, permuted_certificate=perm_cert,
                      public_elapsed_sec=.1, library_status="converged", status="converged" if eligible else "not_converged",
                      accuracy_eligible=eligible, precision_eligible=accurate, budget_eligible=within,
                      library_status_eligible=True, invariance={"passed": True},
                      scores={str(tau): independent_score(original, truth, tau)[0] for tau in config["support_thresholds"]},
                      permuted_input_hashes={"X": array_sha256(np.asfortranarray(X[:, permutation])),
                                            "y": array_sha256(y), "w_true": array_sha256(truth[permutation])})
        write_json(base/"runs"/(call_identifier+".json"), record)
        fixture_audit = FixtureAudit()
        item = read_run(fixture_audit, config, base, planned,
                        {"old_metadata": {"p": 3, "lambda": lam}, "historical_coefficients": {"truth": truth},
                         "data": {"X": X, "y": y}}, permutation, True)
        assert not fixture_audit.issues, fixture_audit.issues
        assert item["eligible"] is eligible
    print(json.dumps({"self_check_passed": True, "optimizer_calls": 0, "external_designs_loaded": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/column_permutation_v1.json")
    parser.add_argument("--structure-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--analysis-only", action="store_true", help="Verify saved analysis from already-verified coefficients, without reloading X")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    config = json.loads(args.config.read_text()); base = ROOT/config["output_path"]
    filename = "analysis_verification.json" if args.analysis_only else "verification_structure.json" if args.structure_only else "verification.json"
    try:
        with threadpool_limits(limits=1):
            report = (verify_analysis_outputs(args.config) if args.analysis_only else
                      verify_saved_outputs(args.config, reconstruct=not args.structure_only))
    except Exception as error:
        report = {"all_passed": False, "structural_passed": False, "verifier_script_sha256": sha(Path(__file__)),
                  "issues": [{"kind": "fatal", "message": f"{type(error).__name__}: {error}"}], "traceback": traceback.format_exc()}
    write_json(base/filename, report)
    print(json.dumps({key: report.get(key) for key in ("all_passed", "complete_grid", "counts", "budget", "checks", "issues")}, indent=2))
    if not (report.get("structural_passed") if args.structure_only and not args.analysis_only else report.get("all_passed")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
