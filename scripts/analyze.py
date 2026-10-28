#!/usr/bin/env python3
"""Validate the completed experiment grid and regenerate figures and summaries.

Only results/main and results/real enter the analysis. Pilot, smoke and
reproduction experiments are never pooled into the reported comparisons.
"""
from __future__ import annotations

import os
for _thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/comp6704-matplotlib")

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ista", "fista", "cd")
LABELS = {"ista": "ISTA", "fista": "FISTA", "cd": "Cyclic CD"}
COLORS = {"ista": "#2460a8", "fista": "#c54b28", "cd": "#19836d"}
MARKERS = {"ista": "o", "fista": "s", "cd": "^"}


def load_runs(folder: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((folder / "runs").glob("*.json"))]


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def stats(values) -> dict:
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(values):
        return {"n": 0, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {"n": int(len(values)), "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)), "q75": float(np.quantile(values, 0.75)),
            "min": float(values.min()), "max": float(values.max())}


def match(rows: list[dict], **conditions) -> list[dict]:
    return [row for row in rows if all(row.get(key) == value for key, value in conditions.items())]


def threshold_hit(run: dict, tolerance: float):
    return next((hit for key, hit in run["threshold_hits"].items()
                 if np.isclose(float(key), tolerance, rtol=1e-12, atol=0)), None)


def converged(run: dict, tolerance: float = 1e-6) -> bool:
    return run["status"] == "converged" and run["history"][-1]["normalized_gap"] <= tolerance


def validate(main: list[dict], real: list[dict], config: dict, allow_partial: bool) -> dict:
    settings = config["synthetic"]
    expected_main = {(seed, rho, ratio, method)
                     for seed in settings["seeds"] for rho in settings["rhos"]
                     for ratio in settings["lambda_ratios"] for method in config["methods"]}
    observed_main = [(r["seed"], r["rho"], r["lambda_ratio"], r["method"]) for r in main]
    observed_real = [(r["seed"], r["validation_index"], r["method"]) for r in real]
    issues = []
    if len(set(observed_main)) != len(observed_main):
        issues.append("duplicate synthetic grid cells")
    if set(observed_main) - expected_main:
        issues.append("unexpected synthetic grid cells")
    missing = len(expected_main - set(observed_main))
    if missing:
        issues.append(f"missing {missing} synthetic grid cells")
    expected_real_count = len(config["real"]["seeds"]) * 3 * len(config["methods"])
    if len(real) != expected_real_count:
        issues.append(f"real run count is {len(real)}, expected {expected_real_count}")
    if len(set(observed_real)) != len(observed_real):
        issues.append("duplicate real grid cells")
    for seed in config["real"]["seeds"]:
        seed_rows = match(real, seed=seed)
        indexes = {r["validation_index"] for r in seed_rows}
        if len(indexes) != 3:
            issues.append(f"real seed {seed} has {len(indexes)} lambda values, expected 3")
        for index in indexes:
            if {r["method"] for r in match(seed_rows, validation_index=index)} != set(config["methods"]):
                issues.append(f"incomplete real methods for seed {seed}, lambda index {index}")
        if sum(bool(r["selected_by_validation"]) for r in seed_rows) != len(config["methods"]):
            issues.append(f"real seed {seed} lacks one common best lambda per method")
    hashes = {r["config_sha256"] for r in main + real}
    if len(hashes) > 1:
        issues.append("multiple configuration hashes in the main/real runs")
    if issues and not allow_partial:
        raise SystemExit("Incomplete or inconsistent experiment results: " + "; ".join(issues))
    return {"complete_grid": not issues, "allow_partial": bool(allow_partial), "issues": issues,
            "main_runs": len(main), "expected_main_runs": len(expected_main),
            "real_runs": len(real), "expected_real_runs": expected_real_count,
            "config_hashes": sorted(hashes)}


def flat_runs(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        row = {key: run.get(key) for key in (
            "run_id", "dataset", "seed", "rho", "lambda_ratio", "lambda", "lambda_max",
            "method", "status", "iterations", "elapsed_sec", "cold_start_sec",
            "diagnostics_sec", "lipschitz_setup_sec", "objective_above_reference",
            "objective_increases", "selected_by_validation", "validation_index",
            "test_mse", "test_rmse", "dataset_hash", "config_sha256")}
        row.update({"final_" + key: value for key, value in run["history"][-1].items()})
        row["reference_primal"] = run["reference"]["primal"]
        row["reference_normalized_gap"] = run["reference"]["normalized_gap"]
        row["diagnostics_fraction"] = run["diagnostics_sec"] / max(run["elapsed_sec"], 1e-300)
        for tolerance in (1e-2, 1e-4, 1e-6):
            hit = threshold_hit(run, tolerance)
            row[f"time_to_{tolerance:g}"] = None if hit is None else hit["time"]
            row[f"iteration_to_{tolerance:g}"] = None if hit is None else hit["iteration"]
        rows.append(row)
    return rows


def save_figure(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(directory / (stem + ".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)


def decorate_axis(ax) -> None:
    ax.grid(True, which="major", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.spines[["right", "top"]].set_visible(False)


def draw_convergence(runs: list[dict], figures: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.7), sharey=True, constrained_layout=True)
    for ax, (rho, ratio) in zip(axes.flat, ((0., .01), (0., .1), (.9, .01), (.9, .1))):
        selected = match(runs, seed=0, rho=rho, lambda_ratio=ratio)
        for method in METHODS:
            rows = match(selected, method=method)
            if not rows:
                continue
            run = rows[0]
            h = run["history"]
            times = np.array([point["time"] for point in h])
            gaps = np.array([max(point["normalized_gap"], 1e-16) for point in h])
            valid = times > 0
            label = LABELS[method] + (" (censored)" if not converged(run) else "")
            ax.loglog(times[valid], gaps[valid], color=COLORS[method], label=label, linewidth=1.7)
        ax.axhline(1e-6, color="#777777", linestyle=":", linewidth=1)
        ax.set_title(rf"$\rho={rho:g}$, $\lambda/\lambda_{{max}}={ratio:g}$")
        ax.set_xlabel("Elapsed solver time (s)")
        ax.set_ylabel("Normalized duality gap")
        ax.set_ylim(1e-8, 2.)
        decorate_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("Convergence on paired synthetic problems (seed 0)")
    save_figure(fig, figures, "convergence")


def draw_time(runs: list[dict], config: dict, figures: Path, metric: str, stem: str) -> None:
    ratios, rhos = config["synthetic"]["lambda_ratios"], config["synthetic"]["rhos"]
    fig, axes = plt.subplots(1, len(ratios), figsize=(11., 3.8), sharey=True, constrained_layout=True)
    for ax, ratio in zip(np.atleast_1d(axes), ratios):
        for method in METHODS:
            medians, lowers, uppers = [], [], []
            for rho in rhos:
                all_rows = match(runs, rho=rho, lambda_ratio=ratio, method=method)
                completed = [r[metric] for r in all_rows if converged(r, config["solver"]["tol"])]
                st = stats(completed)
                medians.append(st["median"] if st["n"] else np.nan)
                lowers.append(st["q25"] if st["n"] else np.nan)
                uppers.append(st["q75"] if st["n"] else np.nan)
                censored = len(all_rows) - st["n"]
                if censored:
                    # Display a lower-bound marker at the time budget for censored solves.
                    boundary = config["solver"]["time_limit"]
                    ax.scatter([rho], [boundary], marker="x", color=COLORS[method], zorder=5)
                    ax.annotate(f"{LABELS[method]}: {censored} censored", (rho, boundary),
                                xytext=(2, 4), textcoords="offset points", fontsize=6)
            ax.plot(rhos, medians, marker=MARKERS[method], color=COLORS[method],
                    linewidth=1.6, markersize=5, label=LABELS[method])
            ax.fill_between(rhos, lowers, uppers, color=COLORS[method], alpha=0.15)
        ax.set_title(rf"$\lambda/\lambda_{{max}}={ratio:g}$")
        ax.set_xlabel(r"Within-block correlation $\rho$")
        ax.set_xticks(rhos)
        ax.set_yscale("log")
        decorate_axis(ax)
    axes[0].set_ylabel("Time to normalized gap $10^{-6}$ (s)")
    axes[-1].legend(frameon=False)
    title = "Solver time" if metric == "elapsed_sec" else "Cold-start time (solver + SVD for proximal methods)"
    fig.suptitle(title + "; median and IQR across five paired seeds")
    save_figure(fig, figures, stem)


def draw_recovery(runs: list[dict], config: dict, figures: Path) -> None:
    ratios, rhos = config["synthetic"]["lambda_ratios"], config["synthetic"]["rhos"]
    fig, axes = plt.subplots(2, len(ratios), figsize=(11., 6.4), sharex=True,
                             sharey="row", constrained_layout=True)
    for column, ratio in enumerate(ratios):
        for row, (metric, label) in enumerate((("relative_coef_error", "Relative coefficient error"),
                                               ("f1", "Support F1"))):
            ax = axes[row, column]
            for method in METHODS:
                summaries = [stats(r["history"][-1][metric]
                                   for r in match(runs, rho=rho, lambda_ratio=ratio, method=method))
                             for rho in rhos]
                values = [[s[key] if s["n"] else np.nan for s in summaries]
                          for key in ("median", "q25", "q75")]
                ax.plot(rhos, values[0], color=COLORS[method], marker=MARKERS[method],
                        linewidth=1.6, markersize=4, label=LABELS[method])
                ax.fill_between(rhos, values[1], values[2], color=COLORS[method], alpha=0.15)
            if row == 0:
                ax.set_title(rf"$\lambda/\lambda_{{max}}={ratio:g}$")
            else:
                ax.set_xlabel(r"Within-block correlation $\rho$")
                ax.set_ylim(-.03, 1.03)
            if column == 0:
                ax.set_ylabel(label)
            ax.set_xticks(rhos)
            decorate_axis(ax)
    axes[0, -1].legend(frameon=False)
    censored = sum(not converged(r) for r in runs)
    fig.suptitle(f"Recovery at the final iterate; median and IQR (unconverged runs: {censored})")
    save_figure(fig, figures, "recovery")


def draw_accuracy_recovery(runs: list[dict], config: dict, figures: Path) -> None:
    tolerances = sorted(config["thresholds"], reverse=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True,
                             sharey="row", constrained_layout=True)
    missing = 0
    for column, rho in enumerate((0., .9)):
        for row, (metric, label) in enumerate((("relative_coef_error", "Relative coefficient error"),
                                               ("f1", "Support F1"))):
            ax = axes[row, column]
            for method in METHODS:
                group = match(runs, rho=rho, lambda_ratio=.05, method=method)
                summaries = []
                for tol in tolerances:
                    hits = [threshold_hit(run, tol) for run in group]
                    if row == 0:
                        missing += sum(hit is None for hit in hits)
                    summaries.append(stats(hit[metric] for hit in hits if hit is not None))
                med, lo, hi = [[s[key] if s["n"] else np.nan for s in summaries]
                               for key in ("median", "q25", "q75")]
                ax.plot(tolerances, med, color=COLORS[method], marker=MARKERS[method],
                        label=LABELS[method], linewidth=1.6, markersize=5)
                ax.fill_between(tolerances, lo, hi, color=COLORS[method], alpha=.15)
                for tol, s in zip(tolerances, summaries):
                    if s["n"] and s["n"] != len(config["synthetic"]["seeds"]):
                        ax.annotate(f"n={s['n']}", (tol, s["median"]), fontsize=6)
            ax.set_xscale("log")
            ax.set_xlim(max(tolerances) * 1.5, min(tolerances) / 1.5)
            ax.set_xticks(tolerances)
            if row == 0:
                ax.set_title(rf"$\rho={rho:g}$")
            else:
                ax.set_xlabel("First observed normalized-gap threshold")
            if column == 0:
                ax.set_ylabel(label)
            decorate_axis(ax)
    axes[0, -1].legend(frameon=False)
    fig.suptitle(r"Recovery versus optimization accuracy ($\lambda/\lambda_{max}=0.05$)"
                 + f"\nMedian and IQR; missing threshold hits: {missing}")
    save_figure(fig, figures, "accuracy_recovery")


def aggregate_main(runs: list[dict], config: dict) -> list[dict]:
    rows = []
    for rho in config["synthetic"]["rhos"]:
        for ratio in config["synthetic"]["lambda_ratios"]:
            for method in METHODS:
                group = match(runs, rho=rho, lambda_ratio=ratio, method=method)
                finished = [r for r in group if converged(r, config["solver"]["tol"])]
                row = {"rho": rho, "lambda_ratio": ratio, "method": method,
                       "n_runs": len(group), "n_converged": len(finished),
                       "n_unconverged": len(group) - len(finished)}
                for key in ("elapsed_sec", "cold_start_sec", "iterations"):
                    row[key] = stats(r[key] for r in finished)
                for key in ("relative_coef_error", "precision", "recall", "f1", "test_mse", "nnz"):
                    row[key] = stats(r["history"][-1][key] for r in group)
                rows.append(row)
    return rows


def time_comparisons(runs: list[dict], config: dict) -> dict:
    rho_ratios, method_ratios = [], []
    for ratio in config["synthetic"]["lambda_ratios"]:
        for method in METHODS:
            row = {"lambda_ratio": ratio, "method": method}
            for metric in ("elapsed_sec", "cold_start_sec"):
                paired = []
                for seed in config["synthetic"]["seeds"]:
                    low = match(runs, rho=0., lambda_ratio=ratio, method=method, seed=seed)
                    high = match(runs, rho=.9, lambda_ratio=ratio, method=method, seed=seed)
                    if low and high and converged(low[0]) and converged(high[0]):
                        paired.append(high[0][metric] / low[0][metric])
                row[metric + "_high_over_low_paired"] = stats(paired)
                lo_values = [r[metric] for r in match(runs, rho=0., lambda_ratio=ratio, method=method) if converged(r)]
                hi_values = [r[metric] for r in match(runs, rho=.9, lambda_ratio=ratio, method=method) if converged(r)]
                row[metric + "_ratio_of_medians"] = (float(np.median(hi_values) / np.median(lo_values))
                                                                 if lo_values and hi_values else None)
            rho_ratios.append(row)
        for rho in config["synthetic"]["rhos"]:
            row = {"rho": rho, "lambda_ratio": ratio}
            for metric in ("elapsed_sec", "cold_start_sec"):
                paired = []
                for seed in config["synthetic"]["seeds"]:
                    ista = match(runs, rho=rho, lambda_ratio=ratio, method="ista", seed=seed)
                    fista = match(runs, rho=rho, lambda_ratio=ratio, method="fista", seed=seed)
                    if ista and fista and converged(ista[0]) and converged(fista[0]):
                        paired.append(ista[0][metric] / fista[0][metric])
                row[metric + "_ista_over_fista_paired"] = stats(paired)
            method_ratios.append(row)
    return {"rho_0.9_over_0": rho_ratios, "ista_over_fista": method_ratios,
            "ratio_interpretation": "Ratios only include paired runs that both converged; >1 means slower numerator."}


def threshold_summary(runs: list[dict], config: dict) -> list[dict]:
    summaries = []
    for rho in (0., .9):
        for method in METHODS:
            group = match(runs, rho=rho, lambda_ratio=.05, method=method)
            for tol in config["thresholds"]:
                hits = [threshold_hit(run, tol) for run in group]
                available = [h for h in hits if h is not None]
                row = {"rho": rho, "lambda_ratio": .05, "method": method, "threshold": tol,
                       "n_runs": len(group), "n_hits": len(available)}
                for key in ("time", "iteration", "relative_coef_error", "precision", "recall", "f1", "test_mse"):
                    row[key] = stats(hit[key] for hit in available)
                summaries.append(row)
    return summaries


def support_summary(runs: list[dict], config: dict) -> dict:
    rows = []
    for threshold in config["support_sensitivity"]:
        for method in METHODS:
            group = match(runs, method=method)
            values = [next(v for k, v in r["support_sensitivity"].items()
                           if float(k) == threshold) for r in group]
            row = {"threshold": threshold, "method": method, "n_runs": len(group)}
            for metric in ("precision", "recall", "f1", "nnz"):
                row[metric] = stats(v[metric] for v in values)
            rows.append(row)
    deltas = []
    for run in runs:
        f1_values = [v["f1"] for v in run["support_sensitivity"].values()]
        deltas.append(max(f1_values) - min(f1_values))
    return {"by_method_and_threshold": rows, "within_run_f1_range": stats(deltas),
            "runs_with_any_f1_change": sum(d > 1e-12 for d in deltas)}


def quality_summary(runs: list[dict]) -> dict:
    if not runs:
        return {}
    relative_deltas = [abs(r["objective_above_reference"]) / max(1., abs(r["reference"]["primal"]))
                       for r in runs]
    return {
        "max_absolute_objective_delta_to_reference": max(abs(r["objective_above_reference"]) for r in runs),
        "max_normalized_absolute_objective_delta_to_reference": max(relative_deltas),
        "max_final_normalized_gap": max(r["history"][-1]["normalized_gap"] for r in runs),
        "max_reference_normalized_gap": max(r["reference"]["normalized_gap"] for r in runs),
        "max_dual_feasibility_ratio_all_checkpoints": max(h["dual_feasibility"] for r in runs for h in r["history"]),
        "min_raw_gap_all_checkpoints": min(h["raw_gap"] for r in runs for h in r["history"]),
        "max_final_kkt_inf": max(r["history"][-1]["kkt_inf"] for r in runs),
        "all_status_counts": dict(Counter(r["status"] for r in runs)),
    }


def real_summary(runs: list[dict]) -> dict:
    best = [r for r in runs if r["selected_by_validation"]]
    summaries = []
    for method in METHODS:
        group = match(best, method=method)
        row = {"method": method, "n_best_validation_runs": len(group),
               "n_converged": sum(converged(r) for r in group)}
        for metric in ("elapsed_sec", "cold_start_sec", "test_mse", "test_rmse"):
            row[metric] = stats(r[metric] for r in group)
        for metric in ("validation_mse", "nnz", "normalized_gap"):
            row[metric] = stats(r["history"][-1][metric] for r in group)
        summaries.append(row)
    all_summaries = []
    for method in METHODS:
        group = match(runs, method=method)
        all_summaries.append({"method": method, "n_runs": len(group),
                              "n_converged": sum(converged(r) for r in group),
                              "elapsed_sec": stats(r["elapsed_sec"] for r in group if converged(r)),
                              "test_mse": stats(r["test_mse"] for r in group)})
    return {"best_validation_by_method": summaries, "all_three_lambdas_by_method": all_summaries,
            "best_lambda_rows": [{"seed": r["seed"], "method": r["method"],
                                  "lambda_ratio": r["lambda_ratio"], "lambda": r["lambda"],
                                  "validation_index": r["validation_index"],
                                  "validation_mse": r["history"][-1]["validation_mse"],
                                  "test_mse": r["test_mse"], "nnz": r["history"][-1]["nnz"],
                                  "elapsed_sec": r["elapsed_sec"], "cold_start_sec": r["cold_start_sec"],
                                  "normalized_gap": r["history"][-1]["normalized_gap"]} for r in best],
            "mse_units": "Squared log1p(shares), centered by training mean; not raw share counts"}


def write_real_tex(path: Path, runs: list[dict]) -> None:
    selected = sorted((r for r in runs if r["selected_by_validation"]),
                      key=lambda r: (r["seed"], METHODS.index(r["method"])))
    lines = [r"\begin{tabular}{rlrrrrrr}", r"\toprule",
             r"Seed & Method & $\lambda/\lambda_{\max}$ & NNZ & Val. MSE & Test MSE & Time (s) & Gap \\",
             r"\midrule"]
    for r in selected:
        h = r["history"][-1]
        lines.append(f"{r['seed']} & {LABELS[r['method']]} & {r['lambda_ratio']:.4f} & {h['nnz']} & "
                     f"{h['validation_mse']:.4f} & {r['test_mse']:.4f} & {r['elapsed_sec']:.3f} & "
                     f"{h['normalized_gap']:.1e}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Development only: mark incomplete grids explicitly instead of failing.")
    args = parser.parse_args()
    root = args.root.resolve()
    results, figures = root / "results", root / "figures"
    config_path = results / "main/config.json"
    config = json.loads((config_path if config_path.exists() else root / "configs/study.json").read_text())
    main_runs, real_runs = load_runs(results / "main"), load_runs(results / "real")
    integrity = validate(main_runs, real_runs, config, args.allow_partial)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "pdf.fonttype": 42})
    write_csv(results / "main_runs.csv", flat_runs(main_runs))
    real_rows = flat_runs(real_runs)
    write_csv(results / "real_runs.csv", real_rows)
    write_csv(results / "real_best_validation.csv", [r for r in real_rows if r["selected_by_validation"]])
    hits = []
    for run in main_runs + real_runs:
        for tolerance, point in run["threshold_hits"].items():
            hits.append(dict(run_id=run["run_id"], dataset=run["dataset"], seed=run["seed"],
                             rho=run["rho"], lambda_ratio=run["lambda_ratio"], method=run["method"],
                             threshold=float(tolerance), reached=point is not None,
                             **({} if point is None else point)))
    write_csv(results / "threshold_hits.csv", hits)
    write_real_tex(results / "real_table.tex", real_runs)
    datasets = [json.loads(path.read_text()) for path in sorted((results / "main/datasets").glob("*.json"))]
    real_datasets = [json.loads(path.read_text()) for path in sorted((results / "real/datasets").glob("*.json"))]
    diagnostics = []
    for dataset, rows in (("synthetic", main_runs), ("real", real_runs)):
        for method in METHODS:
            group = match(rows, method=method)
            diagnostics.append({"dataset": dataset, "method": method,
                                "fraction": stats(r["diagnostics_sec"] / max(r["elapsed_sec"], 1e-300) for r in group),
                                "seconds": stats(r["diagnostics_sec"] for r in group)})
    fista = match(main_runs, method="fista")
    summary = {
        "integrity": integrity,
        "timing_definition": "Solver timers include initialization and certificate/metric callbacks. Cold-start adds dense SVD to each proximal solve; excludes common data preprocessing and one-time compiled CD library setup.",
        "threshold_definition": "First observed checkpoint at or below threshold; observed upper bounds, no interpolation.",
        "uncertainty_definition": "Median and interquartile range across fixed paired seeds; descriptive spread, not confidence intervals.",
        "main_grid": aggregate_main(main_runs, config),
        "time_comparisons": time_comparisons(main_runs, config),
        "threshold_recovery": threshold_summary(main_runs, config),
        "support_sensitivity": support_summary(main_runs, config),
        "quality": {"synthetic": quality_summary(main_runs), "real": quality_summary(real_runs)},
        "diagnostics": diagnostics,
        "setup": {"synthetic_lipschitz_sec": stats(d["lipschitz_setup_sec"] for d in datasets),
                  "synthetic_preprocess_sec": stats(d["preprocess_sec"] for d in datasets),
                  "real_lipschitz_sec": stats(d["lipschitz_setup_sec"] for d in real_datasets),
                  "real_preprocess_sec": stats(d["preprocess_sec"] for d in real_datasets),
                  "real_validation_selection_sec": stats(d["selection_sec"] for d in real_datasets)},
        "fista_objective_increases": {"definition": "Increases between recorded certificate checkpoints, not every internal iteration",
                                      "n_runs": len(fista),
                                      "runs_with_increases": sum(r["objective_increases"] > 0 for r in fista),
                                      "total_observed_increases": sum(r["objective_increases"] for r in fista),
                                      "count_per_run": stats(r["objective_increases"] for r in fista)},
        "real": real_summary(real_runs),
    }
    dump(results / "summary.json", summary)
    if main_runs:
        draw_convergence(main_runs, figures)
        draw_time(main_runs, config, figures, "elapsed_sec", "correlation_time")
        draw_time(main_runs, config, figures, "cold_start_sec", "correlation_cold_start")
        draw_recovery(main_runs, config, figures)
        draw_accuracy_recovery(main_runs, config, figures)
    print(json.dumps({"integrity": integrity, "summary": str(results / "summary.json"),
                      "figures": str(figures)}, indent=2))


if __name__ == "__main__":
    main()

