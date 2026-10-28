#!/usr/bin/env python3
"""Build an exploratory Chinese workshop brief from the completed pilot audit.

This script reads saved analysis/evidence only. It does not fit models or change
the course reports, frozen numerical modules, or their experiment settings.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_report import render_pdf
import pymupdf


METHODS = {"fista": "FISTA", "cd": "循环CD"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def stats(values):
    values = np.asarray([float(x) for x in values if x is not None and np.isfinite(x)])
    if not len(values):
        return {"n": 0, "median": None, "q25": None, "q75": None, "min": None, "max": None}
    return {"n": int(len(values)), "median": float(np.median(values)),
            "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75)),
            "min": float(values.min()), "max": float(values.max())}


def fmt(value, decimals=3):
    if value is None or not np.isfinite(value):
        return "未观测"
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.{decimals}f}"


def fraction(numerator, denominator):
    return f"{numerator}/{denominator}" + (f" ({100*numerator/denominator:.1f}%)" if denominator else "（无可用记录）")


def distribution(values, decimals=3):
    s = stats(values)
    return (f"{fmt(s['median'], decimals)} [{fmt(s['q25'], decimals)}, {fmt(s['q75'], decimals)}]"
            if s["n"] else "未观测")


def entries():
    """Checked bibliographic metadata; no mutation of the main bibliography."""
    return [
        {"key": "kumar2024", "text": "Kumar, A.; Bhattacharyya, S.; Bouchard, K. Numerical characterization of support recovery in sparse regression with correlated design. Communications in Statistics—Simulation and Computation 53(3):1504–1518, 2024卷期；2022-03-31在线发表。",
         "url": "https://doi.org/10.1080/03610918.2022.2050392"},
        {"key": "fercoq2015", "text": "Fercoq, O.; Gramfort, A.; Salmon, J. Mind the duality gap: safer rules for the Lasso. ICML, PMLR 37:333–342, 2015.",
         "url": "https://proceedings.mlr.press/v37/fercoq15.html"},
        {"key": "bastounis2023i", "text": "Bastounis, A.; Cucker, F.; Hansen, A. C. When can you trust feature selection?—I: A condition-based analysis of LASSO and generalised hardness of approximation. arXiv:2312.11425, 2023，核查版本为预印本。",
         "url": "https://arxiv.org/abs/2312.11425"},
        {"key": "bastounis2023ii", "text": "Bastounis, A.; Cucker, F.; Hansen, A. C. When can you trust feature selection?—II: On the effects of random data on condition in statistics and optimisation. arXiv:2312.11429, 2023，核查版本为预印本。",
         "url": "https://arxiv.org/abs/2312.11429"},
        {"key": "walwil2025", "text": "Walwil, I.; Fercoq, O. The smoothed duality gap as a stopping criterion. Mathematical Programming Computation 17:653–697, 2025.",
         "url": "https://doi.org/10.1007/s12532-025-00284-0"},
        {"key": "moufad2025", "text": "Moufad, B.; Bannier, P.-A.; Bertrand, Q.; Klopfenstein, Q.; Massias, M. skglm: Improving scikit-learn for Regularized Generalized Linear Models. Journal of Machine Learning Research 26(149):1–6, 2025.",
         "url": "https://www.jmlr.org/papers/v26/24-0008.html"},
    ]


def validated_inputs(base: Path):
    analysis_path = base / "analysis.json"
    require(analysis_path.is_file(), "Pilot analysis is not complete: missing analysis.json")
    summary = read(analysis_path)
    require(summary.get("stage") == "pilot", "This brief requires the registered pilot, not smoke/confirm results")
    config = read(base / "effective_config.json")
    completion = read(base / "completion.json")
    require(summary["n_targets"] == completion["recorded_targets"] == config["expected_pilot_targets"],
            "Incomplete pilot target count")
    require(summary["n_runs"] == completion["recorded_trajectories"] == config["expected_pilot_trajectories"],
            "Incomplete pilot trajectory count")
    require(summary["numerical_signatures"] == [completion["numerical_signature"]]
            and summary["config_sha256"] == [completion["config_sha256"]],
            "Pilot analysis/provenance signatures differ")
    provenance = dict(summary["input_provenance"])
    for relative, expected in provenance.items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(base.resolve()) and path.is_file() and sha(path) == expected,
                f"Stale or invalid pilot evidence: {relative}")
    runs = [read(path) for path in sorted((base / "runs").glob("*.json"))]
    require(len(runs) == summary["n_runs"], "New/stale run records require analysis regeneration")
    require(len({run["run_id"] for run in runs}) == len(runs), "Duplicate pilot run identifiers")
    target_ids = sorted({run["target_id"] for run in runs})
    references = [read(base / "references" / (target + ".json")) for target in target_ids]
    designs = [read(base / "datasets" / (target + ".json")) for target in target_ids]
    for path in (analysis_path, base / "config.json", base / "effective_config.json", base / "completion.json"):
        provenance[str(path.relative_to(ROOT))] = sha(path)
    for target in target_ids:
        path = base / "datasets" / (target + ".json")
        provenance[str(path.relative_to(ROOT))] = sha(path)
    require(all(d["n"] == config["n"] and d["p"] == config["p"]
                and d["sparsity"] == config["sparsity"] for d in designs), "Design/config dimensions differ")
    return summary, config, completion, runs, references, designs, provenance


def endpoint_rows(summary, config, gap):
    rows = []
    metrics = []
    for tau in config["support_thresholds"]:
        for method in config["methods"]:
            all_observed = [r for r in summary["checkpoints"] if r["method"] == method
                            and r["tau"] == tau and r["gap_threshold"] == gap]
            good = [r for r in all_observed if r["reference_consensus"]]
            same = sum(r["same_support_as_reference"] for r in good)
            counts = sum(r["same_recovery_counts_as_reference"] for r in good)
            swaps = sum(r["same_recovery_counts_as_reference"] and not r["same_support_as_reference"] for r in good)
            diff = stats(abs(r["f1"] - r["reference_f1"]) for r in good)
            metrics.append({"method": method, "tau": tau, "gap": gap, "observed": len(all_observed),
                            "reference_consensus_n": len(good), "support_matches": same,
                            "count_matches": counts, "same_count_different_support": swaps,
                            "absolute_f1_difference": diff})
            rows.append([f"{METHODS.get(method, method)} / {tau:g}", str(len(good)), fraction(same, len(good)),
                         fraction(counts, len(good)), str(swaps), fmt(diff["median"], 4)])
    return rows, metrics


def build_pages(summary, config, completion, runs, references, designs):
    p = lambda text: ["p", text]
    eq = lambda text: ["eq", text]
    table = lambda headers, rows: ["table", headers, rows]
    image = lambda stem, caption, height: ["image", "workshop/pilot/" + stem, caption, height]
    primary_gap = next((float(t) for t in config["gap_thresholds"] if np.isclose(t, 1e-6, rtol=1e-12, atol=0)), None)
    require(primary_gap is not None, "Registered pilot does not contain the planned 1e-6 audit threshold")
    coarse_gap = max(config["gap_thresholds"])
    support_table, endpoint_metrics = endpoint_rows(summary, config, coarse_gap)
    primary_tau = config["support_threshold"]
    main_rows = [r for r in summary["checkpoints"] if r["gap_threshold"] == primary_gap
                 and r["tau"] == primary_tau and r["reference_consensus"]]
    main_support = sum(r["same_support_as_reference"] for r in main_rows)
    main_counts = sum(r["same_recovery_counts_as_reference"] for r in main_rows)
    main_swaps = sum(r["same_recovery_counts_as_reference"] and not r["same_support_as_reference"] for r in main_rows)
    n = len(main_rows)
    coarse_rows = [r for r in summary["checkpoints"] if r["gap_threshold"] == coarse_gap
                   and r["tau"] == primary_tau and r["reference_consensus"]]
    coarse_support = sum(r["same_support_as_reference"] for r in coarse_rows)
    coarse_counts = sum(r["same_recovery_counts_as_reference"] for r in coarse_rows)
    coarse_swaps = sum(r["same_recovery_counts_as_reference"] and not r["same_support_as_reference"] for r in coarse_rows)
    intermediate_gap = next((float(t) for t in config["gap_thresholds"] if np.isclose(t, 1e-4, rtol=1e-12, atol=0)), None)
    intermediate_rows = [r for r in summary["checkpoints"] if r["gap_threshold"] == intermediate_gap
                         and r["tau"] == primary_tau and r["reference_consensus"]]
    intermediate_support = sum(r["same_support_as_reference"] for r in intermediate_rows)
    observed_departures = []
    departure_sentences = []
    for method in config["methods"]:
        rows = [r for r in summary["finals"] if r["tau"] == primary_tau and r["method"] == method
                and r["reference_consensus"]]
        count = sum(r["observed_reference_match_was_later_lost"] for r in rows)
        observed_departures.append({"method": method, "tau": primary_tau,
                                    "eligible_trajectories": len(rows), "observed_match_later_lost": count})
        departure_sentences.append(f"{METHODS.get(method,method)}为{count}/{len(rows)}")
    if n and main_support == n:
        finding = (f"在主支持阈值τ={primary_tau:g}、首次观测归一化gap≤{primary_gap:g}处，"
                   f"{fraction(main_support, n)}条可用轨迹已逐索引匹配数值参考，"
                   "本轮没有在这一档位观测到两个充分性端点的分离。普通区域稳定本身应作为结果保留。")
    elif n:
        finding = (f"在主支持阈值τ={primary_tau:g}、首次观测归一化gap≤{primary_gap:g}处，"
                   f"逐索引支持匹配为{fraction(main_support, n)}，真支持TP/FP/FN计数匹配为"
                   f"{fraction(main_counts, n)}；其中{main_swaps}条计数相同但变量索引不同。"
                   "这是指定网格的观测差异，尚不足以验证新停止规则或证明普遍充分性。")
    else:
        finding = "主gap档位没有同时具备已达检查点与数值参考共识的记录；不能据此评价充分性。"
    finding += (f"更粗gap={coarse_gap:g}处，逐索引匹配{coarse_support}/{len(coarse_rows)}，"
                f"计数匹配{coarse_counts}/{len(coarse_rows)}，其中{coarse_swaps}条计数相同而索引不同。")
    if intermediate_gap is not None:
        finding += (f"在gap={intermediate_gap:g}处已观测到{intermediate_support}/{len(intermediate_rows)}逐索引匹配；"
                    "不能把粗档位差异改写为常用容差普遍不可靠。")
    accepted = summary["reference_targets_accepted"]
    status_text = "；".join(f"{key}={value}" for key, value in sorted(summary["statuses"].items()))
    geometries = "、".join(f"{g['family']} ρ={g['rho']:g}" for g in config["geometries"])
    lambdas = [d["lambda"] for d in designs]
    lambda_text = fmt(lambdas[0], 6) if min(lambdas) == max(lambdas) else f"{fmt(min(lambdas),6)}–{fmt(max(lambdas),6)}"
    settings_rows = [
        ["目标与轨迹", f"{summary['n_targets']}个固定目标；{summary['n_runs']}条轨迹；" + "/".join(METHODS.get(m,m) for m in config["methods"])],
        ["独立种子", ", ".join(str(s) for s in config["seeds"]) + "；因素和算法均配对，轨迹检查点不是独立重复"],
        ["设计与维度", f"{geometries}；n={config['n']}，p={config['p']}，真非零数={config['sparsity']}"],
        ["布局与幅度", f"{', '.join(config['layouts'])}；strong幅度1，mixed_weak每第{config['weak_every']}位幅度{config['weak_amplitude']:g}"],
        ["固定噪声与惩罚", f"σ={config['noise_std']:g}，c={config['lambda_factor']:g}，λ={lambda_text}；不随信号方差或λmax缩放"],
        ["精度与观察间隔", f"最终gap目标={config['solver']['tol']:g}；每{config['solver']['check_every']}次近端迭代/完整CD轮检查；单轨迹上限{config['solver']['time_limit']:g}秒"],
    ]
    pages = [[
        ["title", "固定对偶间隙下的支持充分性审计"],
        p("Workshop先导研究简报 · 探索性结果\n姓名：________________　学号：________________\n日期：2026年10月28日（Asia/Shanghai）"),
        ["h", "1　研究问题与当前证据"],
        p("同一Lasso目标求到某个固定gap时，相对生成真支持的评价是否已与高精度数值参考一致，而逐变量支持仍未匹配？本研究将这两个端点分开记录，同时量化继续计算到更紧gap的观测成本。它不提出新算法，也不预设较松容差一定不充分。"),
        eq(r"F(w)=\frac{\|Xw-y\|_2^2}{2n}+\lambda\|w\|_1,\quad\lambda=c\sigma\sqrt{2\log(p)/n}"),
        p(finding),
        table(["因素", "预先固定的pilot协议"], settings_rows),
        p(f"状态记录：{status_text}。数值参考获共识的目标为{accepted}/{summary['n_targets']}；其余目标单列不确定，不计入参考充分性分母。block与Toeplitz在ρ=0时共享完全相同设计，故只保留一个共同基线。"),
        p("独立信息来自数据种子；不同几何、幅度、布局及两条算法轨迹均有配对关系。下文比例只是预设网格中的描述频率，不作为独立二项试验的总体概率或置信区间。"),
    ]]
    coef_agreement = stats(r["coefficient_linf_difference"] for r in references)
    objective_agreement = stats(r["normalized_primal_difference"] for r in references)
    pages.append([
        ["h", "2　两个端点、数值参考与记账定义"],
        p("记S*为生成真支持，Sref,τ为数值参考在|w|>τ下的支持，St,τ为检查点支持。实验能直接计算前者与参考之间的差别，但没有可信精确证书时，不能把Sref,τ改名为精确Lasso最优支持。τ是输出支持阈值，λ是统计惩罚，gap是迭代停止精度；本轮运算始终为float64，未改变浮点位宽。"),
        eq(r"\bar g=\frac{P-D}{\max(1,|P|,|D|)}"),
        p("参考支持端点要求St,τ与Sref,τ逐索引相同，符号另行保存。真实支持评价端点在本先导使用更严格的离线代理：相对S*的TP、FP、FN计数与参考完全相同。它既不等于恢复S*，也不保证变量索引相同；本轮尚未预注册实际评价容许误差δ，不能将该代理升级成可部署的“已经足够”决策。"),
        p(f"参考由scikit-learn循环CD与skglm各自求解[[moufad2025]]，共同gap目标{config['references']['normalized_gap_tol']:g}；要求系数无穷范数差≤{config['references']['coef_linf_agreement_tol']:g}、归一目标值差≤{config['references']['normalized_objective_agreement_tol']:g}，并在预定τ下支持和符号一致。当前最大系数差={fmt(coef_agreement['max'])}，最大归一目标差={fmt(objective_agreement['max'])}。两实现共识是数值证据，仍不是精确支持或可信计算证明。"),
        ["h2", "数值偏差可能造成伤害，也可能偶然修正统计选择"],
        p("H(t)统计参考选择正确、当前选择错误的变量；R(t)统计参考选择错误、当前反而选择正确的变量。令E为相对S*的误选加漏选，则有下述二元选择记账恒等式。它不是新定理，也不能直接用于F1或三状态符号错误。"),
        eq(r"E_t=E_{\mathrm{ref}}+H(t)-R(t),\qquad|S_{t,\tau}\,\triangle\,S_{\mathrm{ref},\tau}|=H(t)+R(t)"),
        p(f"主τ={primary_tau:g}下，首次观测匹配后又在后续检查点离开的轨迹，" + "、".join(departure_sentences)
          + "。此处每条轨迹只按主τ计一次，不将跨τ记录叠加成独立样本。首次匹配不意味着此后不变；“最后一次观测不一致之后”也只描述保存区间，不能保证未来所有迭代。"),
        p("Gap Safe在本轮仅作离线零变量排除诊断[[fercoq2015]]：equicorrelation set、未排除集合、数值非零支持及生成真支持不可互换，覆盖率不等于恢复率。"),
    ])
    missing_at_table = config["expected_pilot_trajectories"] - sum(
        r["tau"] == primary_tau and r["gap_threshold"] == coarse_gap for r in summary["checkpoints"])
    pages.append([
        ["h", "3　固定gap的观测充分性"],
        image("support_adequacy", "图1　首次观测参考支持匹配yield：已达gap、参考共识且逐索引匹配数/全部预注册轨迹数。未达或参考不确定不算观测匹配，也不据此认定支持错误。三种几何分线、强/弱幅度分线型，每线汇合两布局及配对种子；不是独立总体成功概率。", 290),
        table(["方法 / τ", "参考共识且已达n", "逐索引支持同", "TP/FP/FN同", "同计数不同支持", "|F1−F1ref|中位"], support_table),
        p(f"表中固定观测gap档位为{coarse_gap:g}，用于显示更粗精度下的端点差异。在主τ下，{missing_at_table}条轨迹尚无这一档位的已达检查点；数值参考未获共识的目标也不进入比率分母。τ由" + "、".join(f"{t:g}" for t in config["support_thresholds"]) + "分别审计，不把极小非零系数自动当成可靠统计发现。"),
        p("同计数而不同索引可由变量替换或伤害与修正的抵消产生。即使F1不变，也应检查逐索引差异；即使逐索引匹配参考，仍可能重复参考的统计误选。完整逐目标、逐检查点与符号记录保留在分析JSON/CSV中。"),
    ])
    transition_rows = []
    tighter = min(config["gap_thresholds"])
    for tau in config["support_thresholds"]:
        for method in config["methods"]:
            selected = next((r for r in summary["transition_summary"] if r["method"] == method and r["tau"] == tau
                             and r["loose_gap"] == primary_gap and r["tight_gap"] == tighter), None)
            if selected is None:
                transition_rows.append([f"{METHODS.get(method,method)} / {tau:g}", "0", "未观测", "未观测", "未观测", "未观测"])
            else:
                transition_rows.append([f"{METHODS.get(method,method)} / {tau:g}", str(selected["n_runs"]),
                                        f"{selected['supports_different']} / {selected['counts_different']}",
                                        str(selected["support_swaps_with_same_counts"]),
                                        fmt(selected["extra_observed_sec"]["median"]),
                                        f"{100*selected['extra_fraction_of_tight_time']['median']:.1f}%"])
    cost_sentences = []
    cost_metrics = []
    for method in config["methods"]:
        group = [run for run in runs if run["method"] == method]
        diagnostic_fraction = [run["diagnostics_sec"] / run["elapsed_sec"] for run in group if run["elapsed_sec"] > 0]
        copy_fraction = [run["callback_snapshot_copy_sec_within_solver_timer"] / run["elapsed_sec"]
                         for run in group if run["elapsed_sec"] > 0]
        diagnostic, snapshot = stats(diagnostic_fraction), stats(copy_fraction)
        cost_metrics.append({"method": method, "diagnostic_fraction": diagnostic, "snapshot_copy_fraction": snapshot,
                             "elapsed_sec": stats(run["elapsed_sec"] for run in group)})
        cost_sentences.append(f"{METHODS.get(method,method)}诊断占解算时间中位{100*diagnostic['median']:.1f}%"
                              f"（系数快照复制子项{100*snapshot['median']:.2f}%）")
    offline = sum(r["offline_gap_safe_and_snapshot_stack_sec"] for r in runs)
    writes = sum(r["artifact_write_sec_outside_solver_timer"] for r in runs)
    pages.append([
        ["h", "4　伤害、偶然修正与追加观测成本"],
        image("harm_repair", f"图2　主τ={primary_tau:g}下的平均H/R变量数，汇合预设条件；仅用已达相应gap且数值参考有共识的轨迹。两者是相对真值的离线记账，不能用“修正”声称改进了估计器。", 205),
        table(["方法 / τ", "两档均已达n", "支持变 / 计数变", "同计数换位", "追加秒中位", "占紧档时间"], transition_rows),
        p(f"表中从首次观测gap≤{primary_gap:g}至首次≤{tighter:g}计算追加时间。它包含途中诊断，且受检查间隔影响，是观测上界差；它不是可保证节省的时间，也没有基于未来真值实现一个停止策略。表的分母仅限两档都已到达的轨迹，未完成轨迹不作为快解。"),
        p("；".join(cost_sentences) + f"。上述诊断和支持/快照回调均已计入解算计时，不能再相加。离线Gap Safe及快照堆叠共{fmt(offline)}秒、文件写出共{fmt(writes)}秒另记；参考拟合、SVD与环境预热也分开记录。"),
        p("旧课程实验的末端支持一致、计数变化稀少，是反对预写“固定gap普遍失败”的已有负信号。本先导允许同样的稳定结果；若两个端点的差别没有实际评价或可复现成本意义，应接受no-go或按负面适用性结果报告。不能仅因少数近退化例或H/R恒等式成立就扩张结论。"),
    ])
    pages.append([
        ["h", "5　最近邻、局限与下一步条件"],
        p("相关设计下的支持恢复数值研究已有系统比较[[kumar2024]]；其2024卷期对应2022年在线发表。Gap Safe已把gap与安全排除及有限识别联系起来[[fercoq2015]]。Bastounis I给出可信支持计算的条件分析[[bastounis2023i]]，II已连接相关随机设计、统计恢复、数值精度与计算成本[[bastounis2023ii]]；两篇本次均按2023预印本引用。Walwil与Fercoq已系统研究停止量及误差界/成本[[walwil2025]]。本轮不能声称首次联结这些层面。"),
        p("当前范围只是固定目标与float64求解轨迹的经验审计。没有穷尽新颖性检索，没有精确支持可信证书，没有独立确认集，也未建立实用容许误差δ或可部署停止规则。支持阈值敏感性、共识失败、未达档位和记录开销均应保留；先导种子与后验选点不能充当新的独立验证。"),
        p("下一步须先冻结可否定主张、δ和实际成本门槛，再使用未见种子与预定设计族独立确认；普通与弱信号区域都应保留。若分离比例接近零、代价差缺乏实际意义或只依赖挑出的极端目标，则停止扩网格。workshop展示、非正式短稿与正式收录状态应分别说明，当前结果不构成发表或接收保证。"),
        ["h2", "参考文献"],
        ["refs"],
        ["h2", "证据与重建"],
        p("主证据：results/workshop/pilot/analysis.json；逐检查点系数与两参考系数保留在独立workshop目录。重建：python scripts/analyze_workshop.py pilot，随后python scripts/build_workshop_brief.py。最近邻原文定位见docs/workshop_nearest_work.md。本简报由保存数据动态生成，不修改课程报告与第二版冻结结果。"),
    ])
    derived = {"primary_gap": primary_gap, "primary_tau": primary_tau, "primary_usable_rows": n,
               "primary_support_matches": main_support, "primary_count_matches": main_counts,
               "primary_same_count_different_support": main_swaps, "endpoints": endpoint_metrics,
               "coarse_gap": coarse_gap, "coarse_usable_rows": len(coarse_rows),
               "coarse_support_matches": coarse_support, "coarse_count_matches": coarse_counts,
               "coarse_same_count_different_support": coarse_swaps,
               "intermediate_gap": intermediate_gap, "intermediate_usable_rows": len(intermediate_rows),
               "intermediate_support_matches": intermediate_support,
               "primary_tau_observed_departures": observed_departures,
               "costs": cost_metrics, "offline_gap_safe_stack_sec": offline, "artifact_write_sec": writes}
    return pages, derived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "report/workshop/pilot_brief_zh.pdf")
    args = parser.parse_args()
    base = ROOT / "results/workshop/pilot"
    summary, config, completion, runs, references, designs, provenance = validated_inputs(base)
    for name in ("support_adequacy", "harm_repair"):
        path = ROOT / "figures/workshop/pilot" / (name + ".png")
        require(path.is_file(), f"Missing final pilot figure: {path}")
        provenance[str(path.relative_to(ROOT))] = sha(path)
    notes = ROOT / "docs/workshop_nearest_work.md"
    require(notes.is_file(), "Missing checked nearest-work notes")
    provenance[str(notes.relative_to(ROOT))] = sha(notes)
    pages, derived = build_pages(summary, config, completion, runs, references, designs)
    citations = entries()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    captions = ["预设先导研究设计", "较粗gap处的两个充分性代理（仅数值参考共识记录）",
                "从主gap档位收紧到最终档位的观测变化与成本"]
    content_path = output.with_name("pilot_brief_content_zh.json")
    content_path.write_text(json.dumps({"pages": pages, "references": citations, "derived": derived},
                                      ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    render_pdf(pages, citations, output, table_captions=captions)
    document = pymupdf.open(output)
    require(1 <= len(document) <= 6, f"Brief has {len(document)} pages; shorten layout before delivery")
    require(output.stat().st_size < 20_000_000, "Brief exceeds 20 MB")
    preview = output.parent / "preview"
    preview.mkdir(exist_ok=True)
    for index, page in enumerate(document):
        page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).save(str(preview / f"pilot_brief_{index+1:02d}.png"))
    validation = {"generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                  "stage": "pilot", "pdf_pages": len(document), "pdf_bytes": output.stat().st_size,
                  "pdf_sha256": sha(output), "identity_complete": False,
                  "identity_display": "Name and student ID deliberately blank",
                  "n_targets": summary["n_targets"], "n_trajectories": summary["n_runs"],
                  "reference_accepted_targets": summary["reference_targets_accepted"],
                  "references": len(citations), "numerical_signatures": summary["numerical_signatures"],
                  "config_sha256": summary["config_sha256"], "input_provenance": provenance,
                  "page_text_chars": [len(page.get_text()) for page in document],
                  "pdf_renderer": "Existing ReportLab/Matplotlib renderer; PDF previews exported for visual review",
                  "interpretation": "Exploratory audit; numerical-reference consensus is not exact support, and observed extra time is not guaranteed deployable saving"}
    output.with_name("pilot_brief_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
                                                           encoding="utf-8")
    print(json.dumps({"output": str(output), "pdf_pages": validation["pdf_pages"],
                      "pdf_bytes": validation["pdf_bytes"], "n_targets": summary["n_targets"],
                      "n_trajectories": summary["n_runs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
