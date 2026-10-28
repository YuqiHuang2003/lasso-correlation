#!/usr/bin/env python3
"""Independent confirmation/real-design driver using frozen pilot helpers.

--check-only validates the grid and file/dependency signatures without data
generation, matrix operations, warmup, fitting, or result writes. Run stages
sequentially after reviewing the exploratory pilot; this driver is not an
automation and does not start either follow-up stage by its mere existence.
"""
from __future__ import annotations

import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback
from zoneinfo import ZoneInfo

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from run_workshop import (cache_guard, dump, fit_references, run_trajectory, sha,
                          source_signature as pilot_source_signature, target_id as pilot_target_id)
from lasso_study.workshop_data import make_workshop_design, make_real_design
from lasso_study.data_v2 import make_uji
from lasso_study.solvers import lipschitz_constant, warmup_cd
from lasso_study.modern_solver import warmup_modern


def followup_signature(config_path):
    """Extend the immutable pilot signature without changing its definition."""
    # Optional packages live in this isolated target. Adding the metadata search
    # path performs no imports from those packages, JIT compilation or fitting.
    optional = ROOT / ".modern_deps"
    if optional.is_dir() and str(optional) not in sys.path:
        sys.path.insert(0, str(optional))
    base_sig, base_provenance = pilot_source_signature()
    extension = {"base_pilot_numerical_signature": base_sig,
                 "followup_source_sha256": sha(Path(__file__)),
                 "config_sha256": sha(config_path)}
    signature = hashlib.sha256(json.dumps(extension, sort_keys=True).encode()).hexdigest()
    return signature, dict(extension, base_provenance=base_provenance)


def followup_targets(config, stage):
    """Pure parameter-grid validation: never touches designs or solvers."""
    if stage not in {"confirm", "real"} or config.get("stage") != stage:
        raise ValueError("The selected stage and configuration stage must agree")
    seeds = config["seeds"]
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if set(seeds) & set(config.get("excluded_exploratory_seeds", [])):
        raise ValueError("Follow-up seeds overlap the excluded exploratory seeds")
    if config["threads"] != 1 or config["methods"] != ["fista", "cd"]:
        raise ValueError("The registered follow-up uses one thread and FISTA/CD")
    if stage == "confirm":
        geometries = {(row["family"], row["rho"]) for row in config["geometries"]}
        if len(geometries) != len(config["geometries"]):
            raise ValueError("Duplicate geometry conditions")
        if ("block", 0.) in geometries and ("toeplitz", 0.) in geometries:
            raise ValueError("The identical rho=0 designs must share one baseline")
        if geometries != {("block", 0.), ("block", .9), ("toeplitz", .9)}:
            raise ValueError("Confirmation must retain the three registered pilot geometries")
        targets = [dict(seed=seed, family=geometry["family"], rho=geometry["rho"], layout=layout, amplitude=amplitude)
                   for seed, geometry, layout, amplitude in product(seeds, config["geometries"],
                                                                    config["layouts"], config["amplitudes"])]
    else:
        if config.get("family") != "uji" or config.get("rho") is not None:
            raise ValueError("Real-design follow-up must use family=uji and rho=null")
        if config.get("source_preprocessing_seed") != 0 or config.get("source_response_used") is not False:
            raise ValueError("The real arm fixes source preprocessing seed zero and excludes its original response")
        if config.get("real_n") != 1000 or config.get("real_sparsity") != 15:
            raise ValueError("The registered real arm uses 1000 rows and 15 signals")
        targets = [dict(seed=seed, family="uji", rho=None, layout=layout, amplitude=amplitude)
                   for seed, layout, amplitude in product(seeds, config["layouts"], config["amplitudes"])]
    identities = {(row["seed"], row["family"], row["rho"], row["layout"], row["amplitude"]) for row in targets}
    if len(identities) != len(targets):
        raise ValueError("Duplicate follow-up parameter cells")
    if len(targets) != config["expected_targets"]:
        raise ValueError("Registered target count differs from the actual parameter grid")
    if len(targets) * len(config["methods"]) != config["expected_trajectories"]:
        raise ValueError("Registered trajectory count differs from the actual parameter grid")
    return targets


def followup_target_id(params):
    if params["family"] == "uji":
        return f"uji_seed{params['seed']}_{params['layout']}_{params['amplitude']}"
    return pilot_target_id(params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("confirm", "real"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config or ROOT / "configs" / ("workshop_confirm.json" if args.stage == "confirm" else "workshop_real.json")
    config = json.loads(config_path.read_text())
    targets = followup_targets(config, args.stage)
    source_sig, provenance = followup_signature(config_path)
    config_hash = sha(config_path)
    if args.check_only:
        print(json.dumps({"stage": args.stage, "targets": len(targets),
                          "trajectories": len(targets) * len(config["methods"]),
                          "reference_public_calls": len(targets) * 2,
                          "seeds": config["seeds"], "numerical_signature": source_sig,
                          "base_pilot_numerical_signature": provenance["base_pilot_numerical_signature"],
                          "config_sha256": config_hash,
                          "data_generated": False, "fits_run": 0, "result_files_written": 0}, indent=2))
        return
    output = ROOT / "results/workshop" / args.stage
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        env_path = output / "environment.json"
        if env_path.exists():
            previous = json.loads(env_path.read_text())
            if previous["numerical_signature"] != source_sig or previous["config_sha256"] != config_hash:
                raise RuntimeError("Existing follow-up stage has different source/configuration; preserve it separately before rerunning")
        setup_start = perf_counter(); warmup_cd(); cd_setup = perf_counter() - setup_start
        modern_setup = warmup_modern()
        if not env_path.exists():
            dump(env_path, dict(provenance, numerical_signature=source_sig, source_signature=source_sig,
                                config_sha256=config_hash, stage=args.stage,
                                started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), timezone="Asia/Shanghai",
                                python=sys.version, platform=platform.platform(), threads=1, threadpools=threadpool_info(),
                                compiler=subprocess.check_output(["gcc", "--version"], text=True).splitlines()[0]))
        dump(output / "config.json", config)
        dump(output / "effective_config.json", config)
        if not (output / "warmup.json").exists():
            dump(output / "warmup.json", {"cd_setup_sec": cd_setup, "modern": modern_setup})
        stage_start = perf_counter()
        source = None
        source_setup_sec = 0.
        if args.stage == "real":
            setup_start = perf_counter()
            source = make_uji(seed=0)
            source_setup_sec = perf_counter() - setup_start
            dump(output / "real_source.json", {
                "source_preparation_sec": source_setup_sec,
                "source_csv_sha256": source["metadata"]["csv_sha256"],
                "source_training_split_sha256": source["metadata"]["split_sha256"]["train"],
                "source_feature_names": source["metadata"]["feature_names"],
                "scope": "Fixed seed-0 UJI training predictor pool; original response is not used for the semisynthetic target"})
        runs, references = [], []
        for params in targets:
            identifier = followup_target_id(params)
            try:
                setup_start = perf_counter()
                if args.stage == "confirm":
                    data = make_workshop_design(config, **params)
                else:
                    data = make_real_design(config, params["seed"], params["layout"], params["amplitude"], source=source)
                    # This is a stage label only: preserve the data module's
                    # original family name and all numerical data/hashes.
                    data["metadata"] = dict(data["metadata"], generator_family=data["metadata"].get("family"), family="uji")
                data["X"] = np.asfortranarray(data["X"], dtype=np.float64)
                preprocessing_sec = perf_counter() - setup_start
                data_hash = hashlib.sha256(json.dumps(data["metadata"]["hashes"], sort_keys=True).encode()).hexdigest()
                setup_start = perf_counter(); L = lipschitz_constant(data["X"]); svd_sec = perf_counter() - setup_start
                ds_path = output / "datasets" / (identifier + ".json")
                if ds_path.exists():
                    cache_guard(json.loads(ds_path.read_text()), source_sig, config_hash, data_hash)
                else:
                    dump(ds_path, dict(data["metadata"], target_id=identifier, params=params,
                                       numerical_signature=source_sig, config_sha256=config_hash, dataset_hash=data_hash,
                                       preprocessing_sec=preprocessing_sec, L=L, lipschitz_setup_sec=svd_sec))
                ref, first, second = fit_references(data, config, output, identifier, params, source_sig, config_hash, data_hash)
                references.append(ref)
                order_seed = params["seed"] + (int(params["rho"] * 100) if params["rho"] is not None else 0)
                order_seed += 1000 if params["layout"] == "clustered" else 0
                for method in np.random.default_rng(order_seed).permutation(config["methods"]):
                    runs.append(run_trajectory(data, config, output, identifier, params, str(method), ref, first, second,
                                               L, svd_sec, source_sig, config_hash, data_hash))
            except Exception as error:
                dump(output / "failures" / (identifier + ".json"),
                     dict(params, target_id=identifier, error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc(), cumulative_stage_wall_sec=perf_counter() - stage_start,
                          numerical_signature=source_sig, config_sha256=config_hash))
                raise
        dump(output / "completion.json", {
            "stage": args.stage, "numerical_signature": source_sig, "source_signature": source_sig,
            "base_pilot_numerical_signature": provenance["base_pilot_numerical_signature"], "config_sha256": config_hash,
            "expected_targets": len(targets), "recorded_targets": len(references),
            "expected_trajectories": len(targets) * len(config["methods"]), "recorded_trajectories": len(runs),
            "expected_reference_calls": len(targets) * 2, "reference_calls": len(references) * 2,
            "actual_reference_library_fit_attempts": sum(ref["actual_library_fit_attempts"] for ref in references),
            "reference_accepted_targets": sum(ref["reference_accepted"] for ref in references),
            "trajectory_status_counts": {status: sum(run["status"] == status for run in runs)
                                          for status in sorted({run["status"] for run in runs})},
            "registered_geometries": config.get("geometries", [{"family": "uji", "rho": None}]),
            "source_preparation_sec": source_setup_sec, "wall_sec_this_invocation": perf_counter() - stage_start,
            "reference_call_count_scope": "Counts stored per-target calls; cache reuse does not imply new fitting in this invocation",
            "observational_scope": "Independent follow-up seeds are the replication unit; conditions are paired and checkpoints are not independent samples.",
            "interpretation": "Independent checking of exploratory observations, not a promise of a theoretical guarantee or novelty."})


if __name__ == "__main__":
    main()
