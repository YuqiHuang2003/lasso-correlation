#!/usr/bin/env python3
"""Build the Chinese research update from completed workshop evidence.

Read-only with respect to numerical artifacts: this script never constructs a
design, fits a model, or reruns an optimizer. Run it only after sequential timed
stages and their analyses/figures have finished. The existing report renderer is
loaded lazily, after the saved grids and provenance have been checked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("pilot", "confirm", "real")
LABEL = {"pilot": "先导", "confirm": "独立确认", "real": "真实设计"}
METHOD = {"fista": "FISTA", "cd": "循环CD"}
PRIMARY_TAU = 1e-6
COARSE_GAP = 1e-2
INTERMEDIATE_GAP = 1e-4
MAIN_GAP = 1e-6
TIGHT_GAP = 1e-10


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def near(a, b):
    return a is not None and b is not None and math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=0.)


def quantile(sorted_values, q):
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def stats(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not values:
        return {"n": 0, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {"n": len(values), "median": statistics.median(values),
            "q25": quantile(values, .25), "q75": quantile(values, .75),
            "min": values[0], "max": values[-1]}


def fmt(value, places=3):
    if value is None or not math.isfinite(float(value)):
        return "未观测"
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.{places}f}"


def ratio(numerator, denominator, percent=True):
    text = f"{numerator}/{denominator}"
    if denominator and percent:
        text += f" ({100*numerator/denominator:.1f}%)"
    elif not denominator:
        text += "（无可用记录）"
    return text


def interval(values, places=3, multiplier=1.):
    item = values if isinstance(values, dict) else stats(values)
    if not item["n"]:
        return "未观测"
    return (f"{fmt(multiplier*item['median'], places)} "
            f"[{fmt(multiplier*item['q25'], places)}, {fmt(multiplier*item['q75'], places)}]")


def numeric_range(values, integer=False):
    values = list(values)
    if not values:
        return "未记录"
    low, high = min(values), max(values)
    display = (lambda value: str(int(value))) if integer else (lambda value: fmt(value, 6))
    return display(low) if low == high else f"{display(low)}–{display(high)}"


def record_value(record, key):
    return record.get(key, record.get("params", {}).get(key))


def condition(record):
    return tuple(record_value(record, key) for key in ("seed", "family", "rho", "layout", "amplitude"))


def registered_conditions(config, stage):
    geometries = config["geometries"] if stage != "real" else [{"family": config["family"], "rho": config["rho"]}]
    return {(int(seed), geometry["family"], geometry["rho"], layout, amplitude)
            for seed, geometry, layout, amplitude in product(
                config["seeds"], geometries, config["layouts"], config["amplitudes"])}


def references():
    """The six checked closest works; do not alter the course bibliography."""
    return [
        {"key": "kumar2024", "text": "Kumar, A.; Bhattacharyya, S.; Bouchard, K. Numerical characterization of support recovery in sparse regression with correlated design. Communications in Statistics—Simulation and Computation 53(3):1504–1518, 2024卷期；2022-03-31在线发表。",
         "url": "https://doi.org/10.1080/03610918.2022.2050392"},
        {"key": "fercoq2015", "text": "Fercoq, O.; Gramfort, A.; Salmon, J. Mind the duality gap: safer rules for the Lasso. ICML, PMLR 37:333–342, 2015.",
         "url": "https://proceedings.mlr.press/v37/fercoq15.html"},
        {"key": "bastounis2023i", "text": "Bastounis, A.; Cucker, F.; Hansen, A. C. When can you trust feature selection?—I: A condition-based analysis of LASSO and generalised hardness of approximation. arXiv:2312.11425, 2023；核查版本为预印本。",
         "url": "https://arxiv.org/abs/2312.11425"},
        {"key": "bastounis2023ii", "text": "Bastounis, A.; Cucker, F.; Hansen, A. C. When can you trust feature selection?—II: On the effects of random data on condition in statistics and optimisation. arXiv:2312.11429, 2023；核查版本为预印本。",
         "url": "https://arxiv.org/abs/2312.11429"},
        {"key": "walwil2025", "text": "Walwil, I.; Fercoq, O. The smoothed duality gap as a stopping criterion. Mathematical Programming Computation 17:653–697, 2025.",
         "url": "https://doi.org/10.1007/s12532-025-00284-0"},
        {"key": "moufad2025", "text": "Moufad, B.; Bannier, P.-A.; Bertrand, Q.; Klopfenstein, Q.; Massias, M. skglm: Improving scikit-learn for Regularized Generalized Linear Models. Journal of Machine Learning Research 26(149):1–6, 2025.",
         "url": "https://www.jmlr.org/papers/v26/24-0008.html"},
        {"key": "bu2023", "text": "Bu, H.; Nagata, T.; Onodera, K.; Arima, S. Sparse Reluctant Interaction Modelling with Convex Hull Approximation—For Accurate Modelling of Highly Correlated Variables Based on Diverse Nearly Optimal Solutions. Journal of Japan Industrial Management Association 73(4E):223–233, 2023.",
         "url": "https://doi.org/10.11221/jima.73.223"},
        {"key": "jarret2025", "text": "Jarret, A.; Kashani, S.; Rué-Queralt, J.; Hurley, P.; Fageot, J.; Simeoni, M. PolyCLEAN: Atomic optimization for super-resolution imaging and uncertainty estimation in radio interferometry. Astronomy & Astrophysics 693:A225, 2025.",
         "url": "https://doi.org/10.1051/0004-6361/202450970"},
    ]


def load_stage(stage):
    """Reject stale/incomplete evidence without requiring every solver to converge."""
    base = ROOT / "results/workshop" / stage
    essential = [base / name for name in ("analysis.json", "effective_config.json", "config.json", "completion.json")]
    for path in essential:
        require(path.is_file(), f"Stage {stage} is not ready: missing {path.relative_to(ROOT)}")
    summary, config, saved_config, completion = [read(path) for path in essential]
    require(summary["stage"] == completion["stage"] == stage, f"Wrong stage label in {base}")
    require(summary.get("integrity_verified") is True
            and summary.get("analysis_script_sha256") == sha(ROOT / "scripts/analyze_workshop.py"),
            f"Final verified analysis is missing or stale: {stage}")
    require(config == saved_config, f"Effective/saved configuration differs: {stage}")
    expected = registered_conditions(config, stage)
    planned_runs = {(cell, method) for cell in expected for method in config["methods"]}
    require(len(config["seeds"]) == len(set(config["seeds"])), f"Repeated seed: {stage}")
    require(near(config["support_threshold"], PRIMARY_TAU), f"Primary tau changed: {stage}")
    for value in (COARSE_GAP, INTERMEDIATE_GAP, MAIN_GAP, TIGHT_GAP):
        require(any(near(value, threshold) for threshold in config["gap_thresholds"]), f"Missing planned gap {value}: {stage}")
    for value in (PRIMARY_TAU, 1e-4):
        require(any(near(value, threshold) for threshold in config["support_thresholds"]), f"Missing support threshold {value}: {stage}")
    require(completion["expected_targets"] == completion["recorded_targets"] == len(expected), f"Incomplete target grid: {stage}")
    require(completion["expected_trajectories"] == completion["recorded_trajectories"] == len(planned_runs), f"Incomplete trajectory grid: {stage}")
    declared_targets = config.get("expected_targets", config.get("expected_pilot_targets"))
    declared_runs = config.get("expected_trajectories", config.get("expected_pilot_trajectories"))
    require(declared_targets == len(expected) and declared_runs == len(planned_runs), f"Declared grid disagrees with factors: {stage}")
    require(summary["n_targets"] == len(expected) and summary["n_runs"] == len(planned_runs), f"Stale analysis counts: {stage}")
    require(summary["numerical_signatures"] == [completion["numerical_signature"]]
            and summary["config_sha256"] == [completion["config_sha256"]], f"Analysis signatures differ: {stage}")
    original_config = ROOT / "configs" / f"workshop_{stage}.json"
    require(original_config.is_file() and sha(original_config) == completion["config_sha256"], f"Configuration hash differs: {stage}")
    require(read(original_config) == config, f"Original/saved configuration differs: {stage}")
    provenance = dict(summary["input_provenance"])
    for relative, expected_hash in provenance.items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(base.resolve()) and path.is_file() and sha(path) == expected_hash,
                f"Stale or invalid analysis evidence: {relative}")
    runs = [read(path) for path in sorted((base / "runs").glob("*.json"))]
    require(len(runs) == len(planned_runs), f"New/missing run files: {stage}")
    require(len({run["run_id"] for run in runs}) == len(runs), f"Duplicate run IDs: {stage}")
    actual_grid = [(condition(run), record_value(run, "method")) for run in runs]
    require(len(set(actual_grid)) == len(actual_grid) and set(actual_grid) == planned_runs, f"Saved grid differs from registered factors: {stage}")
    target_ids = sorted({run["target_id"] for run in runs})
    require(len(target_ids) == len(expected), f"Target IDs do not match registered conditions: {stage}")
    refs = [read(base / "references" / f"{target}.json") for target in target_ids]
    designs = [read(base / "datasets" / f"{target}.json") for target in target_ids]
    for group in (runs, refs):
        require(all(item.get("numerical_signature", item.get("source_signature")) == completion["numerical_signature"]
                    and item["config_sha256"] == completion["config_sha256"] for item in group), f"Mixed record provenance: {stage}")
    require({condition(design) for design in designs} == expected, f"Design metadata grid differs: {stage}")
    refs_by_target = {item["target_id"]: item for item in refs}
    require(all(run["dataset_hash"] == refs_by_target[run["target_id"]]["dataset_hash"] for run in runs), f"Run/reference data hash differs: {stage}")
    accepted = sum(bool(ref.get("reference_accepted")) for ref in refs)
    require(accepted == summary["reference_targets_accepted"] == completion["reference_accepted_targets"], f"Reference acceptance count differs: {stage}")
    statuses = dict(Counter(run["status"] for run in runs))
    require(statuses == summary["statuses"] == completion["trajectory_status_counts"], f"Status counts differ: {stage}")
    finals = summary["finals"]
    for tau in config["support_thresholds"]:
        rows = [row for row in finals if near(row["tau"], tau)]
        require(len(rows) == len(runs) and {row["run_id"] for row in rows} == {run["run_id"] for run in runs},
                f"Missing/duplicate final support records at tau={tau}: {stage}")
    for design in designs:
        real = stage == "real"
        require(design["n"] == (config["real_n"] if real else config["n"])
                and design["sparsity"] == (config["real_sparsity"] if real else config["sparsity"]), f"Wrong design dimensions: {stage}")
        require(design["p"] > 0 and design["p"] % config["block_size"] == 0, f"Invalid retained feature count: {stage}")
        if real:
            require(design["source_preprocessing_seed"] == 0 and design["source_original_response_used"] is False,
                    "Real arm must use the fixed source pool without the original response")
        else:
            require(design["p"] == config["p"], f"Synthetic feature count differs: {stage}")
    failures = []
    for path in sorted((base / "failures").glob("*.json")):
        failure = read(path)
        identifier = failure.get("target_id", path.stem)
        methods = {run["method"] for run in runs if run["target_id"] == identifier}
        failures.append({"path": str(path.relative_to(ROOT)), "target_id": identifier,
                         "error_type": failure.get("error_type"), "error": failure.get("error"),
                         "has_all_registered_trajectories_now": methods == set(config["methods"])})
        provenance[str(path.relative_to(ROOT))] = sha(path)
    for path in essential + [original_config] + [base / "datasets" / f"{target}.json" for target in target_ids]:
        provenance[str(path.relative_to(ROOT))] = sha(path)
    for optional in ("environment.json", "warmup.json", "real_source.json"):
        path = base / optional
        if path.is_file():
            provenance[str(path.relative_to(ROOT))] = sha(path)
    lock_path = base / "protocol_lock.json"
    require(lock_path.is_file(), f"Missing saved protocol lock: {stage}")
    lock = read(lock_path)
    require(lock["numerical_signature"] == completion["numerical_signature"]
            and lock["config_sha256"] == completion["config_sha256"], f"Protocol lock differs: {stage}")
    if stage in {"confirm", "real"}:
        require(any(near(item.get("loose_gap"), INTERMEDIATE_GAP) and near(item.get("tight_gap"), MAIN_GAP)
                    for item in lock["primary_comparisons"]), f"Primary registered cost comparison differs: {stage}")
    provenance[str(lock_path.relative_to(ROOT))] = sha(lock_path)
    return {"stage": stage, "base": base, "summary": summary, "config": config, "completion": completion,
            "runs": runs, "references": refs, "designs": designs, "failures": failures,
            "provenance": provenance, "registered_targets": len(expected), "registered_runs": len(planned_runs)}


def endpoint(stage, gap, tau=PRIMARY_TAU, method=None):
    planned = [run for run in stage["runs"] if method is None or run["method"] == method]
    observed = [row for row in stage["summary"]["checkpoints"]
                if near(row["tau"], tau) and near(row["gap_threshold"], gap)
                and (method is None or row["method"] == method)]
    require(len({row["run_id"] for row in observed}) == len(observed), "Duplicate threshold checkpoint")
    good = [row for row in observed if row["reference_consensus"]]
    bad = [row for row in good if not row["same_support_as_reference"]]
    swaps = [row for row in good if row["same_recovery_counts_as_reference"] and not row["same_support_as_reference"]]
    return {"stage": stage["stage"], "gap": gap, "tau": tau, "method": method,
            "registered": len(planned), "reached": len(observed), "not_reached": len(planned)-len(observed),
            "accepted_reference_reached": len(good), "reached_reference_uncertain": len(observed)-len(good),
            "support_matches": sum(row["same_support_as_reference"] for row in good),
            "count_matches": sum(row["same_recovery_counts_as_reference"] for row in good),
            "same_count_different_support": len(swaps), "different_support_seeds": sorted({row["seed"] for row in bad}),
            "same_count_different_support_seeds": sorted({row["seed"] for row in swaps}),
            "absolute_f1_difference": stats(abs(row["f1"]-row["reference_f1"]) for row in good)}


def departures(stage, method):
    group = [row for row in stage["summary"]["finals"] if near(row["tau"], PRIMARY_TAU) and row["method"] == method]
    good = [row for row in group if row["reference_consensus"]]
    lost = [row for row in good if row["observed_reference_match_was_later_lost"]]
    ever = [row for row in good if row["first_observed_reference_match_time"] is not None]
    return {"method": method, "tau": PRIMARY_TAU, "registered": len(group), "reference_consensus": len(good),
            "ever_observed_match": len(ever), "later_lost": len(lost), "later_lost_seeds": sorted({row["seed"] for row in lost})}


def transition(stage, method, loose, tight):
    """Use each observed trajectory once at the primary tau only."""
    matches = [item for item in stage["summary"]["transition_summary"]
               if item["method"] == method and near(item["tau"], PRIMARY_TAU)
               and near(item["loose_gap"], loose) and near(item["tight_gap"], tight)]
    require(len(matches) <= 1, "Duplicate transition summary")
    all_runs = sum(run["method"] == method for run in stage["runs"])
    empty = stats([])
    row = matches[0] if matches else {"n_runs": 0, "supports_different": 0, "counts_different": 0,
                                    "support_swaps_with_same_counts": 0, "extra_observed_sec": empty,
                                    "extra_fraction_of_tight_time": empty}
    return {"stage": stage["stage"], "method": method, "loose_gap": loose, "tight_gap": tight,
            "registered": all_runs, "both_reached": row["n_runs"], "not_both_reached": all_runs-row["n_runs"],
            "supports_different": row["supports_different"], "counts_different": row["counts_different"],
            "same_count_support_changes": row["support_swaps_with_same_counts"],
            "extra_observed_sec": row["extra_observed_sec"], "fraction_of_tighter_checkpoint_time": row["extra_fraction_of_tight_time"]}


def stage_audit(stage):
    endpoints = [endpoint(stage, gap, tau, method) for gap, tau, method in product(
        (COARSE_GAP, INTERMEDIATE_GAP, MAIN_GAP, TIGHT_GAP), stage["config"]["support_thresholds"], stage["config"]["methods"])]
    diagnostics = []
    for method in stage["config"]["methods"]:
        group = [run for run in stage["runs"] if run["method"] == method]
        diagnostics.append({"method": method, "elapsed_sec": stats(run["elapsed_sec"] for run in group),
                            "diagnostic_sec": stats(run["diagnostics_sec"] for run in group),
                            "diagnostic_fraction": stats(run["diagnostics_sec"]/run["elapsed_sec"] for run in group if run["elapsed_sec"] > 0),
                            "callback_snapshot_copy_sec": stats(run["callback_snapshot_copy_sec_within_solver_timer"] for run in group)})
    return {"stage": stage["stage"], "independent_seeds": stage["config"]["seeds"],
            "registered_targets": stage["registered_targets"], "registered_runs": stage["registered_runs"],
            "statuses": stage["summary"]["statuses"], "reference_accepted_targets": stage["summary"]["reference_targets_accepted"],
            "reference_uncertain_targets": stage["registered_targets"]-stage["summary"]["reference_targets_accepted"],
            "saved_failure_records": stage["failures"], "endpoints": endpoints,
            "primary_tau_departures": [departures(stage, method) for method in stage["config"]["methods"]],
            "transitions": [transition(stage, method, loose, tight) for method, (loose, tight) in product(
                stage["config"]["methods"], ((COARSE_GAP, MAIN_GAP), (INTERMEDIATE_GAP, MAIN_GAP), (MAIN_GAP, TIGHT_GAP)))],
            "diagnostics": diagnostics,
            "offline_gap_safe_stack_sec": sum(run["offline_gap_safe_and_snapshot_stack_sec"] for run in stage["runs"]),
            "artifact_write_sec": sum(run["artifact_write_sec_outside_solver_timer"] for run in stage["runs"]),
            "stage_wall_sec": stage["completion"]["wall_sec_this_invocation"],
            "n": sorted({design["n"] for design in stage["designs"]}),
            "p": sorted({design["p"] for design in stage["designs"]}),
            "sparsity": sorted({design["sparsity"] for design in stage["designs"]})}


def load_reference_diagnostics(real):
    """Read the completed post-hoc diagnostic without recomputing matrices."""
    path = real["base"] / "reference_diagnostics.json"
    require(path.is_file(), "Missing final real-design reference diagnostic")
    diagnostic = read(path)
    require(diagnostic["diagnostic_script_sha256"] == sha(ROOT / "scripts/diagnose_workshop_references.py"),
            "Reference diagnostic script changed after its saved analysis")
    require(diagnostic["numerical_signature"] == real["completion"]["numerical_signature"]
            and diagnostic["config_sha256"] == real["completion"]["config_sha256"], "Reference diagnostic source/configuration differs")
    for relative, expected in diagnostic["input_provenance"].items():
        source = (ROOT / relative).resolve()
        require(source.is_relative_to(ROOT) and source.is_file() and sha(source) == expected,
                f"Reference diagnostic input changed: {relative}")
    require(diagnostic["summary"]["n_targets"] == real["registered_targets"]
            and diagnostic["summary"]["reference_accepted_targets"] == real["summary"]["reference_targets_accepted"],
            "Reference diagnostic target/acceptance denominator differs")
    require("1e-12" in diagnostic["summary"].get("x_only_group_aggregation", {}),
            "Wait for the final X-only group aggregation diagnostic before rendering")
    return diagnostic, path


def load_reference_recovery_impact(real):
    path = real["base"] / "reference_recovery_impact.json"
    require(path.is_file(), "Missing final numerical-reference recovery impact analysis")
    result = read(path)
    require(result["analysis_script_sha256"] == sha(ROOT / "scripts/analyze_workshop_reference_impact.py")
            and result["source_verification_sha256"] == sha(real["base"] / "verification.json"),
            "Reference recovery impact script or verification changed")
    for relative, expected in result["input_provenance"].items():
        source = (ROOT / relative).resolve()
        require(source.is_relative_to(real["base"].resolve()) and source.is_file() and sha(source) == expected,
                f"Reference recovery impact input changed: {relative}")
    require(all(result["summaries"][str(tau)]["n_targets"] == real["registered_targets"]
                for tau in (PRIMARY_TAU, 1e-4)), "Reference recovery impact denominator differs")
    return result, path


def threshold_finding(stage):
    row = endpoint(stage, INTERMEDIATE_GAP)
    lead = f"{LABEL[stage['stage']]}在gap≤{INTERMEDIATE_GAP:g}、主τ={PRIMARY_TAU:g}处"
    if row["support_matches"] == row["registered"] and row["registered"]:
        return lead + f"全部{row['registered']}条预注册轨迹均已到达、参考获共识且逐索引支持匹配。"
    if row["accepted_reference_reached"]:
        result = (lead + f"的逐索引匹配为{ratio(row['support_matches'],row['accepted_reference_reached'])}"
                  f"（仅已达且参考共识）；全注册观测匹配yield为{ratio(row['support_matches'],row['registered'])}。")
    else:
        result = lead + "没有已达且参考获共识的可用记录，不能评价参考支持充分性。"
    if row["not_reached"] or row["reached_reference_uncertain"]:
        result += f"未达{row['not_reached']}条、已达但参考不确定{row['reached_reference_uncertain']}条；不能据此称全网格一致。"
    return result


def coarse_sentence(stage):
    pieces = []
    for method in stage["config"]["methods"]:
        row = endpoint(stage, COARSE_GAP, method=method)
        pieces.append(f"{METHOD.get(method,method)}逐索引同{row['support_matches']}/{row['accepted_reference_reached']}、"
                      f"TP/FP/FN同{row['count_matches']}/{row['accepted_reference_reached']}，"
                      f"其中同计数不同索引{row['same_count_different_support']}条"
                      f"（涉及{len(row['same_count_different_support_seeds'])}个种子）")
    return f"较粗gap={COARSE_GAP:g}、主τ下：" + "；".join(pieces) + "。分母均为已达且参考共识轨迹。"


def departure_sentence(stage):
    pieces = []
    for method in stage["config"]["methods"]:
        row = departures(stage, method)
        pieces.append(f"{METHOD.get(method,method)}为{row['later_lost']}/{row['reference_consensus']}条"
                      f"（涉及{len(row['later_lost_seeds'])}个种子；曾匹配{row['ever_observed_match']}条）")
    return "首次观测匹配后又离开：" + "；".join(pieces) + "。仅主τ逐方法计数，不跨τ重复累加。"


def support_table(stage):
    rows = []
    for method, tau in product(stage["config"]["methods"], stage["config"]["support_thresholds"]):
        row = endpoint(stage, INTERMEDIATE_GAP, tau, method)
        rows.append([f"{METHOD.get(method,method)} / {tau:g}", f"{row['reached']}/{row['registered']}",
                     str(row["accepted_reference_reached"]), f"{row['support_matches']} / {row['count_matches']}",
                     str(row["same_count_different_support"]), ratio(row["support_matches"], row["registered"])])
    return ["方法 / τ", "已达 / 注册", "已达且共识n", "支持同 / 计数同", "同计数异支持", "匹配yield"], rows


def yield_caption(stage):
    geometry = ("三种几何分别成线，强/弱幅度分线型；各线汇合两种布局和配对种子。"
                if stage == "confirm" else "真实设计各线汇合两种布局和配对种子，幅度设置分线。")
    return (f"图{1 if stage == 'confirm' else 2}　{LABEL[stage]}的已观测参考支持匹配yield："
            "分子是已达该gap、参考共识且逐索引匹配的轨迹，分母是该图线全部预注册轨迹。"
            "未达档位或参考不确定不计作已观测匹配，也不据此认定支持错误。" + geometry)


def make_pages(stages, audit, diagnostic, recovery_impact):
    p = lambda text: ["p", text]
    small = lambda text: ["small", text]
    table = lambda heads, rows: ["table", heads, rows]
    image = lambda stage, stem, caption, height: ["image", f"workshop/{stage}/{stem}", caption, height]
    confirm, real = stages["confirm"], stages["real"]
    stage_rows = []
    for stage in STAGES:
        item = stages[stage]
        config = item["config"]
        dims = (f"{numeric_range((d['n'] for d in item['designs']), integer=True)} / "
                f"{numeric_range((d['p'] for d in item['designs']), integer=True)} / "
                f"{numeric_range((d['sparsity'] for d in item['designs']), integer=True)}")
        stage_rows.append([LABEL[stage], f"{len(config['seeds'])}个：{min(config['seeds'])}–{max(config['seeds'])}",
                           f"{item['registered_targets']} / {item['registered_runs']}", dims,
                           f"{item['summary']['reference_targets_accepted']} / {item['registered_targets']}"])
    # A same-count/different-identity observation must actually recur, not be assumed.
    coarse = {stage: endpoint(stages[stage], COARSE_GAP) for stage in ("confirm", "real")}
    if all(row["same_count_different_support"] for row in coarse.values()):
        recurrence = "粗gap下“计数一致掩盖变量身份差异”在两独立臂均有观测。"
    elif any(row["same_count_different_support"] for row in coarse.values()):
        present = "、".join(LABEL[stage] for stage, row in coarse.items() if row["same_count_different_support"])
        absent = "、".join(LABEL[stage] for stage, row in coarse.items() if not row["same_count_different_support"])
        recurrence = f"该粗gap现象出现在{present}，在{absent}的可用记录中未复现，不能声称跨臂普遍成立。"
    else:
        recurrence = "两独立臂在粗gap的可用记录中均未观测到“计数一致但身份不同”；先导现象未在此设置复现。"
    complete_all = all(endpoint(stages[stage], INTERMEDIATE_GAP)["support_matches"] == stages[stage]["registered_runs"]
                       for stage in ("confirm", "real"))
    implication = ("常用的1e-4档位在本轮保持稳定，是限制更强主张的负信号；不能将较粗档位差异写成“1e-4普遍不可靠”。"
                   if complete_all else "1e-4档位的充分性须按已达、参考共识和逐条件记录限定；不确定或未达记录不能自动算作失配，也不能省略。")
    confirmed_intermediate = endpoint(confirm, INTERMEDIATE_GAP)
    tighter_sentences = []
    for item in (confirm, real):
        row = endpoint(item, MAIN_GAP)
        tighter_sentences.append(f"{LABEL[item['stage']]}在gap≤{MAIN_GAP:g}处支持匹配"
                                 f"{row['support_matches']}/{row['accepted_reference_reached']}条可用轨迹")
    pages = [[
        ["title", "固定gap与支持评价：独立确认研究进展"],
        p("研究进展报告 · 候选经验贡献审计\n姓名：________________　学号：________________\n日期：2026年10月28日（Asia/Shanghai）"),
        ["h", "1　主要结果与证据范围"],
        p("问题是：固定Lasso优化精度下，相对生成真支持的评价是否已经稳定，而数值参考的逐变量支持仍未匹配？本轮在先导之后加入未见种子的合成确认和真实设计半合成验证，并记录继续求解的观测成本。"),
        p(threshold_finding(confirm) + threshold_finding(real)
          + f"合成确认在1e-4处有{confirmed_intermediate['accepted_reference_reached']-confirmed_intermediate['support_matches']}条逐索引反例；"
          + "，".join(tighter_sentences) + "。"),
        p(recurrence + implication),
        table(["阶段", "独立种子", "目标 / 轨迹", "n / p / 真非零", "参考共识目标"], stage_rows),
        p(f"独立确认的重复单位是{len(confirm['config']['seeds'])}个数据种子，真实设计臂是{len(real['config']['seeds'])}个子样本/噪声种子。"
          f"不能把{confirm['registered_targets']}个条件目标或{real['registered_targets']}个真实设计目标当作彼此独立样本；同一种子的几何、布局、幅度和方法均配对，轨迹检查点是重复观察。"),
        p("本报告是可否定的经验进展，不认定新颖性或发表级贡献已经成立。数值参考共识不等于精确支持；追加观测时间不等于一个可部署停止规则保证节省的时间。"),
    ]]
    failure_rows = []
    for stage in STAGES:
        item, derived = stages[stage], audit[stage]
        status = "；".join(f"{name}={count}" for name, count in sorted(derived["statuses"].items()))
        recovered = sum(f["has_all_registered_trajectories_now"] for f in item["failures"])
        failure_rows.append([LABEL[stage], status, str(derived["reference_uncertain_targets"]),
                             f"{len(item['failures'])} / {recovered}",
                             str(derived["registered_runs"]-endpoint(item, TIGHT_GAP)["reached"])])
    reference_list = confirm["references"] + real["references"]
    coef_max = max(ref["coefficient_linf_difference"] for ref in reference_list)
    objective_max = max(ref["normalized_primal_difference"] for ref in reference_list)
    config = confirm["config"]
    counts = [design["column_counts"] for design in real["designs"]]
    real_column_text = (f"固定seed-0训练池原有{numeric_range((c['source'] for c in counts),True)}列；"
                        f"抽样后去常量保留{numeric_range((c['after_constant_removal'] for c in counts),True)}列，"
                        f"再按配置上限和完整{real['config']['block_size']}列索引块保留p={numeric_range((d['p'] for d in real['designs']),True)}。")
    pages.append([
        ["h", "2　协议、端点与完整分母"],
        ["eq", r"F(w)=\frac{\|Xw-y\|_2^2}{2n}+\lambda\|w\|_1,\quad\lambda=c\sigma\sqrt{2\log(p)/n}"],
        p(f"合成确认保留block ρ=0、block ρ=0.9、Toeplitz ρ=0.9三个几何；ρ=0两族设计完全相同，只注册一个共同基线。"
          f"训练标准化后定义β，σ={config['noise_std']:g}、c={config['lambda_factor']:g}固定。dispersed每索引组至多一个信号，clustered每活跃组五个；"
          f"strong幅度1，mixed_weak每第{config['weak_every']}个有序信号幅度{config['weak_amplitude']:g}。固定绝对λ不随信号或λmax缩放。"),
        small(f"合成λ={numeric_range(d['lambda'] for d in confirm['designs'])}；真实臂λ={numeric_range(d['lambda'] for d in real['designs'])}，随抽样保留p略变。每个固定目标跨方法使用同一λ，不能说全研究共用一个绝对λ。"),
        p("真实设计来自UJIIndoorLoc的WAP信号列；固定源预处理seed=0，仅从训练池抽取新子样本。原响应不用于此半合成任务、选变量或调参数；响应由预设稀疏真值和固定Gaussian噪声生成。"
          + real_column_text + "抽样行及逐步列索引、常量剔除和尾块剔除均保留。索引块只是支持布局分区，不能称为真实独立协方差块。此臂n、p和真非零数不同，不与合成臂宣称同难度直接横比。"),
        p("参考支持端点要求|w|>τ下逐索引匹配数值参考；真实支持评价端点使用离线TP、FP、FN与参考完全相同的代理。后者不保证变量身份相同，也不代表恢复了真支持；尚无预先定义的实用评价容许误差δ。主τ=1e-6，τ=1e-4作敏感性审计；τ、统计惩罚λ、迭代gap与固定float64位宽各自不同。"),
        p(f"scikit-learn循环CD与skglm[[moufad2025]]须共同达到归一gap≤{config['references']['normalized_gap_tol']:g}、"
          f"在预算内完成、系数差≤{config['references']['coef_linf_agreement_tol']:g}、归一目标差≤{config['references']['normalized_objective_agreement_tol']:g}，"
          f"并在两τ下支持和符号一致。确认+真实臂全部参考对（含未共识）最大系数差{fmt(coef_max)}、归一目标差{fmt(objective_max)}；这些检查不是可信精确支持证书。"),
        table(["阶段", "全部轨迹状态", "参考不确定目标", "异常记录 / 已补齐", "未达1e-10轨迹"], failure_rows),
        small("异常记录即使后续补齐也保留；超时是有效状态，未完成者不作为快解。下页同时列全注册、已达、已达且参考共识分母。条件内Wilson区间仅适用于固定条件/方法、已达且参考共识的种子；不将跨条件或配对方法合并成独立二项样本。"),
    ])
    ds = diagnostic["summary"]
    metrics = ds["metric_summaries"]
    targets = diagnostic["targets"]
    accepted_targets = [row for row in targets if row["reference_accepted"]]
    uncertain_targets = [row for row in targets if not row["reference_accepted"]]
    rank_key = str(PRIMARY_TAU)
    rank_bad_accepted = sum(row["union_active_geometry"][rank_key]["rank_deficient_at_reported_tolerance"] for row in accepted_targets)
    rank_bad_uncertain = sum(row["union_active_geometry"][rank_key]["rank_deficient_at_reported_tolerance"] for row in uncertain_targets)
    grouping = ds["x_only_group_aggregation"]["1e-12"]
    representative = next(row for row in targets if row["target_id"] == ds["representative_target"])
    representative_geometry = representative["union_active_geometry"][rank_key]
    representative_groups = representative["x_only_group_aggregation"]["1e-12"]["nontrivial_group_aggregates"]
    representative_group = max(representative_groups, key=lambda row: abs(row["aggregated_coef_sklearn"]))
    representative_text = (f"按最大系数分歧选的事后示例{representative['target_id']}中，"
                           + "/".join(representative_group["feature_names"])
                           + f"近比例但非逐位相同；活跃并集{representative_geometry['union_size']}列、数值秩{representative_geometry['numerical_rank']}，"
                           f"最小奇异值{fmt(representative_geometry['minimum_singular_value_X_over_sqrt_n'])}。"
                           "机制说明限于本轮抽样和预处理后的设计。")
    diagnostic_text = (f"后验机制核对：原坐标{ds['reference_uncertain_targets']}/{ds['n_targets']}个目标参考不确定；"
                       f"其中{rank_bad_uncertain}/{len(uncertain_targets)}的双参考活跃并集在报告SVD阈值下数值秩亏，"
                       f"获共识者为{rank_bad_accepted}/{len(accepted_targets)}。全部目标预测RMSE差最大{fmt(metrics['prediction_delta_rmse']['max'])}，"
                       f"L1范数差{fmt(metrics['l1_norm_absolute_difference']['max'])}，目标差{fmt(metrics['absolute_objective_difference']['max'])}，"
                       f"但系数差可达{fmt(metrics['coefficient_linf_difference']['max'])}。仅用X按列顺序anchor、相对回归残差≤1e-12建近比例组后，"
                       f"两τ的组支持均{grouping['support_agreement_counts'][rank_key]}/{ds['n_targets']}一致，"
                       f"聚合系数差最大{fmt(grouping['aggregated_coefficient_linf_difference']['max'])}；1e-10/1e-14敏感性相同。"
                       "这是事后改变比较对象的机制对照，不能撤销原坐标不确定、等同找回真实变量或证明数学多解；不能外推原始全UJI或已有论文。")
    impact = recovery_impact["summaries"][str(PRIMARY_TAU)]
    sensitivity_impact = recovery_impact["summaries"][str(1e-4)]
    impact_case = impact["maximum_case"]
    impact_text = (f"评价后果：使用两套达到共同精度与预算的候选参考系数，主τ下{impact['targets_with_different_f1']}/{impact['n_targets']}个目标的真实支持F1随实现而变，"
                   f"最大差{100*impact['maximum_absolute_f1_difference']:.2f}个百分点；"
                   f"示例{impact_case['target_id']}为{impact_case['sklearn']['f1']:.5f}与{impact_case['skglm']['f1']:.5f}，"
                   f"敏感τ下为{sensitivity_impact['targets_with_different_f1']}/{sensitivity_impact['n_targets']}且最大差同为"
                   f"{100*sensitivity_impact['maximum_absolute_f1_difference']:.2f}个百分点。"
                   f"这是全部目标、{len(real['config']['seeds'])}个配对种子的事后比较，不证明哪套更正确或数学多解。")
    for stage in ("confirm", "real"):
        item = stages[stage]
        heads, rows = support_table(item)
        coarse_sensitivity = endpoint(item, COARSE_GAP, 1e-4)
        pages.append([
            ["h", f"{3 if stage == 'confirm' else 4}　{LABEL[stage]}：充分性与路径不稳定"],
            image(stage, "support_adequacy", yield_caption(stage), 245 if stage == "confirm" else 195),
            table(heads, rows),
            p(f"表中gap固定为{INTERMEDIATE_GAP:g}；“支持同/计数同”分别指逐索引支持和TP/FP/FN计数相对参考一致，匹配yield=支持同/全注册。"
              "其余注册记录可能是已知失配、未达档位或参考不确定，不能混称支持错误。"),
            p(coarse_sentence(item)),
            p(f"输出阈值敏感性：在τ=1e-4的粗gap处，同计数异支持为{coarse_sensitivity['same_count_different_support']}/"
              f"{coarse_sensitivity['accepted_reference_reached']}条可用轨迹；主τ与敏感性τ不重复计入总样本数。"),
            p(departure_sentence(item)),
            *([small(diagnostic_text), small(representative_text), small(impact_text)] if stage == "real" else []),
            small("首次匹配以及最后一次观测不一致之后的时间，只描述已保存检查点；不是未来迭代不再改变的保证。图汇合布局/种子展示描述趋势，完整逐条件的分母与Wilson区间保留在analysis.json，避免用汇合比例推断独立总体概率。"),
        ])
    cost_rows = []
    for stage in ("confirm", "real"):
        for row in audit[stage]["transitions"]:
            if not near(row["loose_gap"], INTERMEDIATE_GAP):
                continue
            cost_rows.append([f"{LABEL[stage]} / {METHOD.get(row['method'],row['method'])}",
                              f"{row['both_reached']}/{row['registered']}", interval(row["extra_observed_sec"],5),
                              interval(row["fraction_of_tighter_checkpoint_time"],1,100) + "%",
                              f"{row['supports_different']} / {row['counts_different']}"])
    diagnostics = []
    for stage in ("confirm", "real"):
        for row in audit[stage]["diagnostics"]:
            diagnostics.append(f"{LABEL[stage]}{METHOD.get(row['method'],row['method'])}诊断中位"
                               f"{fmt(row['diagnostic_sec']['median'])}秒（占{fmt(100*row['diagnostic_fraction']['median'],1)}%）")
    pages.append([
        ["h", "5　伤害、偶然修正与追加观测成本"],
        image("confirm", "harm_repair", "图3　独立确认：主τ下H/R平均变量数，仅已达且参考共识记录。H是参考正确而当前错误，R是参考错误而当前偶然正确；分母可随gap变化。", 145),
        image("real", "harm_repair", "图4　真实设计：与图3相同记账。E当前=E参考+H−R，逐索引参考差异=H+R；这是二元选择恒等式，不是新定理、F1分解或估计器改进。", 145),
        table(["阶段 / 方法", "均已达 / 注册", "追加秒中位[IQR]", "占紧档时间[IQR]", "支持变 / 计数变"], cost_rows),
        small("主表采用注册比较1e-4→1e-6。百分比逐轨迹定义为(t紧−t松)/t紧，列中位及四分位，不是“相对松档多花百分之几”。分母只含两档均到达者，含参考不确定目标的轨迹；此处仅比较同一轨迹两档的支持/真支持计数变化。未两档均达者保留为删失，不能当成零秒或快解。1e-2→1e-6及1e-6→1e-10补充成本保留于内容JSON。检查间隔使时间只是首次已观测档位之差，不能保证实际部署节省。"),
        small("；".join(diagnostics) + "。证书/支持诊断和回调快照复制均已计入解算时间，复制是子项，不重复相加。"
              f"确认+真实臂的离线Gap Safe/快照堆叠合计{fmt(sum(audit[s]['offline_gap_safe_stack_sec'] for s in ('confirm','real')))}秒，"
              f"文件写出{fmt(sum(audit[s]['artifact_write_sec'] for s in ('confirm','real')))}秒另记；参考拟合、SVD、预热亦单列。"),
    ])
    pages.append([
        ["h", "6　候选经验贡献、负信号与后续门槛"],
        p("最近邻已覆盖大量核心问题：Kumar等比较相关设计恢复[[kumar2024]]；Gap Safe连接gap与安全排除[[fercoq2015]]；Bastounis I/II研究可信支持、条件、恢复与精度成本[[bastounis2023i]][[bastounis2023ii]]；Walwil与Fercoq研究停止量及成本[[walwil2025]]。Bu等已利用多组近最优Lasso解研究相关变量选择[[bu2023]]，PolyCLEAN已用对偶饱和区域描述支持不确定性[[jarret2025]]；本轮仅审计此基准的参考资格，不把多支持或近依赖作为新发现。"),
        p("候选经验价值是：在固定float64、固定绝对λ及配对条件下，将逐索引参考支持、真支持评价计数、路径曾匹配后离开和绝对秒数同时审计，并用独立种子及真实设计检验适用范围。H/R恒等式不是理论贡献，参考共识不是精确解证明；输入近似/可变位宽的可信计算问题也不同于本轮迭代容差实验。"),
        p(implication + "若差异仅在非常粗档位出现、在确认或真实设计不复现，或绝对追加时间太小、记录成本占比高，则不支持实用停止策略贡献。若只在少数挑出的条件成立，也应保留no-go或限定为适用性/负面结果。"),
        p("下一步应先预注册实用评价误差δ和可接受绝对成本门槛，再决定是否需要新设计或新停止规则。不得依据生成真值在线停机，不将首次已观测匹配当未来保证，也不把本轮确认包装成已成立的发表级创新。仍须进一步核查新颖性、外部设计适用范围以及不同支持阈值下的结论。"),
        ["h2", "参考文献"], ["refs"],
        ["h2", "证据与重建"],
        small("输入：results/workshop/{pilot,confirm,real}/analysis.json与completion.json；原始逐轨迹、参考和系数证据原位保留。先完成各阶段analyze_workshop.py，再运行python scripts/build_workshop_update.py。最近邻定位见docs/workshop_nearest_work.md。内容JSON保留逐方法分母、失败清单和时间统计，validation记录证据哈希；此脚本不运行优化或改写冻结课程实验。"),
    ])
    return pages, {"intermediate_gap_fully_observed_and_matching_both_followups": complete_all,
                   "coarse_same_count_different_support": coarse,
                   "reference_diagnostics_summary": ds,
                   "reference_recovery_impact": recovery_impact["summaries"],
                   "reference_diagnostics_rank_deficiency": {"uncertain": rank_bad_uncertain, "uncertain_total": len(uncertain_targets),
                                                              "accepted": rank_bad_accepted, "accepted_total": len(accepted_targets)},
                   "candidate_interpretation": recurrence + implication}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "report/workshop/research_update_zh.pdf")
    parser.add_argument("--check-only", action="store_true", help="Validate evidence and compose content in memory; do not render or write")
    parser.add_argument("--no-preview", action="store_true", help="Skip preview image exports when rebuilding")
    args = parser.parse_args()
    stages = {stage: load_stage(stage) for stage in STAGES}
    seed_sets = {stage: set(item["config"]["seeds"]) for stage, item in stages.items()}
    require(not seed_sets["pilot"] & seed_sets["confirm"] and not seed_sets["pilot"] & seed_sets["real"]
            and not seed_sets["confirm"] & seed_sets["real"], "Follow-up seed sets are not independent of earlier stages")
    provenance = {}
    for item in stages.values():
        provenance.update(item["provenance"])
    for stage, stem in product(("confirm", "real"), ("support_adequacy", "harm_repair")):
        path = ROOT / "figures/workshop" / stage / f"{stem}.png"
        require(path.is_file(), f"Follow-up figure is not ready: {path.relative_to(ROOT)}")
        require(path.stat().st_mtime_ns >= (stages[stage]["base"] / "analysis.json").stat().st_mtime_ns,
                f"Figure predates its analysis; rebuild stage figures: {path.relative_to(ROOT)}")
        provenance[str(path.relative_to(ROOT))] = sha(path)
    notes = ROOT / "docs/workshop_nearest_work.md"
    require(notes.is_file(), "Missing checked nearest-work notes")
    provenance[str(notes.relative_to(ROOT))] = sha(notes)
    diagnostic, diagnostic_path = load_reference_diagnostics(stages["real"])
    provenance[str(diagnostic_path.relative_to(ROOT))] = sha(diagnostic_path)
    provenance["scripts/diagnose_workshop_references.py"] = sha(ROOT / "scripts/diagnose_workshop_references.py")
    recovery_impact, recovery_impact_path = load_reference_recovery_impact(stages["real"])
    provenance[str(recovery_impact_path.relative_to(ROOT))] = sha(recovery_impact_path)
    provenance["scripts/analyze_workshop_reference_impact.py"] = sha(ROOT / "scripts/analyze_workshop_reference_impact.py")
    ambiguity_notes = ROOT / "docs/workshop_reference_ambiguity_literature.md"
    if ambiguity_notes.is_file():
        provenance[str(ambiguity_notes.relative_to(ROOT))] = sha(ambiguity_notes)
    findings_notes = ROOT / "docs/workshop_findings_zh.md"
    if findings_notes.is_file():
        provenance[str(findings_notes.relative_to(ROOT))] = sha(findings_notes)
    provenance[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    audit = {stage: stage_audit(item) for stage, item in stages.items()}
    pages, interpretation = make_pages(stages, audit, diagnostic, recovery_impact)
    citation_entries = references()
    content = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "pages": pages,
               "references": citation_entries, "stage_audits": audit, "derived_interpretation": interpretation,
               "input_provenance": provenance,
               "replication_unit": "Data seed; conditions, methods, thresholds, and checkpoints are paired repeated observations"}
    if args.check_only:
        print(json.dumps({"evidence_valid": True, "planned_pages": len(pages),
                          "stages": {stage: {key: item[key] for key in ("registered_targets", "registered_runs", "reference_uncertain_targets", "statuses")}
                                     for stage, item in audit.items()}, "rendered": False}, ensure_ascii=False, indent=2))
        return
    # Importing the renderer can create no numerical results. Rendering and image
    # exports occur only here, after the timed stages are explicitly finished.
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_report import render_pdf
    import pymupdf

    output = args.output.resolve()
    require(output.is_relative_to((ROOT / "report/workshop").resolve()), "Keep this update in report/workshop")
    require(output.name != "pilot_brief_zh.pdf", "The frozen pilot brief must not be overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    content_path = output.with_name("research_update_content_zh.json")
    content_path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    captions = ["三个阶段的注册设计与完成证据", "全部轨迹状态、参考不确定与异常记录",
                "独立确认：gap≤1e-4的两个支持端点", "真实设计：gap≤1e-4的两个支持端点",
                "主支持阈值下的追加观测时间与端点变化"]
    render_pdf(pages, citation_entries, output, table_captions=captions)
    with pymupdf.open(output) as document:
        require(5 <= len(document) <= 6, f"Update has {len(document)} pages; adjust layout before delivery")
        require(output.stat().st_size < 20_000_000, "Update exceeds 20 MB")
        page_chars = [len(page.get_text()) for page in document]
        require(all(count > 80 for count in page_chars), "An unexpected blank/overflow page was generated")
        if not args.no_preview:
            preview = output.parent / "preview"
            preview.mkdir(exist_ok=True)
            for index, page in enumerate(document):
                page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).save(str(preview / f"research_update_{index+1:02d}.png"))
        validation = {"generated_at": content["generated_at"], "pdf_pages": len(document),
                      "pdf_bytes": output.stat().st_size, "pdf_sha256": sha(output), "page_text_chars": page_chars,
                      "identity_complete": False, "identity_display": "Name and student ID deliberately blank",
                      "references": len(citation_entries), "stages": {stage: {key: item[key] for key in (
                          "independent_seeds", "registered_targets", "registered_runs", "statuses", "reference_accepted_targets",
                          "reference_uncertain_targets", "saved_failure_records", "n", "p", "sparsity")} for stage, item in audit.items()},
                      "numerical_signatures": {stage: item["summary"]["numerical_signatures"] for stage, item in stages.items()},
                      "config_sha256": {stage: item["summary"]["config_sha256"] for stage, item in stages.items()},
                      "input_provenance": provenance, "content_sha256": sha(content_path),
                      "previews_exported": not args.no_preview,
                      "visual_review_status": "Pending visual inspection of the exported pages",
                      "interpretation": "Exploratory research progress; no exact-reference, universal sufficiency, established novelty, or deployable-saving claim"}
    output.with_name("research_update_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "pdf_pages": validation["pdf_pages"], "pdf_bytes": validation["pdf_bytes"],
                      "stages": {stage: {"targets": item["registered_targets"], "trajectories": item["registered_runs"]}
                                 for stage, item in audit.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
