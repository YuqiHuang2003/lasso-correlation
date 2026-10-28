#!/usr/bin/env python3
"""Build a Chinese draft PDF and editable TeX from the saved, checked results."""
from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/comp6704-matplotlib")
import argparse
import html
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / ".report_deps").is_dir():
    sys.path.insert(0, str(ROOT / ".report_deps"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.mathtext import math_to_image
import pymupdf
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def bib_entries():
    text = (ROOT / "report/references.bib").read_text()
    entries = []
    for key, body in re.findall(r"@\w+\{([^,]+),(.*?)(?=\n@|\Z)", text, re.S):
        fields = {name: value for name, value in re.findall(r"^\s*(\w+)\s*=\s*\{(.*)\},?\s*$", body, re.M)}
        clean = lambda s: s.replace("{$\\ell_1$}", "ℓ1").replace("$\\ell_1$", "ℓ1").replace(r"{\.Z}","Ż").replace(r"{\l}","ł").replace(r"\&","&").replace("{", "").replace("}", "").replace("--", "–")
        author = clean(fields.get("author", "")).replace(" and ", "; ")
        venue = clean(fields.get("journal", fields.get("booktitle", fields.get("howpublished", ""))))
        if fields.get('archivePrefix','').lower()=='arxiv':
            venue='arXiv:'+fields.get('eprint','')+'（预印本）'
        citation = f"{author}. {clean(fields['title'])}. {venue}, {fields['year']}"
        if "volume" in fields:
            citation += f", {fields['volume']}"
        if "pages" in fields:
            citation += f":{clean(fields['pages'])}"
        citation += "."
        url = "https://doi.org/" + fields["doi"] if "doi" in fields else fields.get("url", "")
        entries.append({"key": key, "text": citation, "url": url, "year":int(fields['year'])})
    return entries


def build_pages(summary, structure, sensitivity, author):
    display_name = author['name'].strip() or '________________'
    display_student_id = author['student_id'].strip() or '________________'
    grid = summary["main_grid"]
    def g(rho, ratio, method="cd"):
        return next(x for x in grid if x["rho"] == rho and x["lambda_ratio"] == ratio and x["method"] == method)
    def th(rho, method, tol):
        return next(x for x in summary["threshold_recovery"] if x["rho"] == rho and x["method"] == method and x["threshold"] == tol)
    med = lambda d: d["median"]
    t0, t9 = g(0, .05), g(.9, .05)
    rtimes = summary["time_comparisons"]["rho_0.9_over_0"]
    ratio_vals = [x["elapsed_sec_high_over_low_paired"]["median"] for x in rtimes]
    real = summary["real"]["best_validation_by_method"]
    q = summary["quality"]
    fista = summary["fista_objective_increases"]
    p, eq, image, table = (lambda s: ["p", s]), (lambda s: ["eq", s]), (lambda name, cap, h: ["image", name, cap, h]), (lambda head, rows: ["table", head, rows])
    pages = []
    pages.append([
        ["title", "特征相关性对 Lasso 求解效率与稀疏恢复的影响"],
        p(f"COMP6704 个人作业 · 中文研究报告草稿\n姓名：{display_name}　学号：{display_student_id}\n实验及草稿日期：2026 年 10 月 28 日（Asia/Shanghai）"),
        ["h", "摘要"],
        p(f"本文在统一 Lasso 目标和对偶间隙精度下，比较 ISTA、FISTA 与循环坐标下降（CD）。分组相关合成实验包含 4 档相关性、3 档相对正则化和 5 个配对种子，共 180 次求解；另以 UCI 新闻数据完成 27 次真实回归比较。全部运行达到归一化对偶间隙 1e-6。相关性从 0 增至 0.9 后，九组算法与正则化组合的配对耗时比中位数介于 {min(ratio_vals):.1f}–{max(ratio_vals):.1f}。CD 在本组合成实现中最快；FISTA 的有限精度优势依赖数据与精度。更小优化间隙不保证恢复指标逐步改善，且弱正则化下相关性升高时支持 F1 反而增加。结果强调应区分求解误差、系数误差与支持选择，并保留实现和数据机制对结论的限制。"),
        ["h", "1　研究动机与问题"],
        p("相关特征可能携带相近的信息，使某些方向的曲率减小，也使多个变量在预测上相互替代。Lasso 通过 ℓ1 惩罚产生稀疏系数，但“较容易算出最优解”和“较准确找到真实变量”并非同一件事。本文研究：（1）相关性是否增加达到同一精度的成本；（2）FISTA 的加速是否总带来实际优势；（3）提高优化精度是否总能改善恢复。项目应用凸性、梯度步长、非光滑正则化、坐标最小化与对偶性等课程相关概念。个人项目研究凸非光滑回归，区别于小组的 grokking 神经网络优化器课题。"),
        ["h", "2　相关工作"],
        p("Lasso 将参数收缩和变量选择结合[[tibshirani1996]]。阈值化迭代[[daubechies2004]]与 FISTA[[beck2009]]建立了近端方法的基础；ISTA/FISTA 的经典界描述目标残差，不能直接推断支持恢复或实际时间排名。坐标下降的可分非光滑收敛分析[[tseng2001]]与正则化路径实现[[friedman2010]]为本项目的残差增量更新提供依据。"),
        p("模型选择一致性[[zhao2006]]和带噪恢复阈值[[wainwright2009]]揭示了协方差、信号、样本量和正则化条件的重要性。弹性网的分组效应[[zou2005]]说明相关变量选择受惩罚结构影响。后续工作包括对偶间隙安全筛选[[fercoq2015]][[ndiaye2017]]，以及坐标外推和现代求解库。本文对这三种基础算法进行受控比较，不提出新算法；参考库只核对目标值，不充当计时基线。")
    ])
    pages.append([
        ["h", "3　问题建模与误差区分"],
        eq(r"\min_{w\in\mathbb{R}^p}P(w),\quad P(w)=\frac{1}{2n}\|Xw-y\|_2^2+\lambda\|w\|_1,\quad \lambda>0"),
        p("式(1)的平方损失凸且光滑，ℓ1 项凸但在零点不可微。λ>0 使水平集有界，有限维情形下存在最优解；当 X 列满秩时解唯一。训练特征中心化并缩放至单位总体标准差，y 减训练均值，等价于保留未惩罚截距。验证和测试沿用训练变换。"),
        eq(r"\nabla f(w)=X^\top(Xw-y)/n,\quad L=\|X\|_2^2/n,\quad \lambda_{\max}=\|X^\top y\|_\infty/n"),
        p("合成机制为 y=Xw*+ε，真实系数定义在训练标准化坐标。优化残差比较当前目标与同一 λ 的最优目标；恢复误差比较当前系数与 w*。两者参照物不同，Lasso 最优解仍含噪声和正则化偏差。真实数据没有 w*，不报告其支持恢复率。"),
        eq(r"E_{\mathrm{coef}}=\frac{\|w-w^\star\|_2}{\|w^\star\|_2},\quad \widehat S=\{j:|w_j|>10^{-6}\},\quad F_1=\frac{2\,\mathrm{Prec}\,\mathrm{Rec}}{\mathrm{Prec}+\mathrm{Rec}}"),
        ["h", "3.1　相关结构与曲率"],
        eq(r"x_{i,hj}^{\mathrm{raw}}=\sqrt{\rho}\,z_{i,h}+\sqrt{1-\rho}\,e_{i,hj},\quad z,e\sim\mathcal{N}(0,1)"),
        eq(r"\Sigma_b=(1-\rho)I+\rho\mathbf{1}\mathbf{1}^\top,\quad \kappa(\Sigma_b)=\frac{1+(b-1)\rho}{1-\rho},\quad b=20"),
        p("不同组独立，组内公共变量与独立变量共同控制相关性。总体块的谱特征值为 1−ρ 和 1+19ρ；这一代数关系为求解困难提供结构解释。标准化后有限样本 Hessian 为 XᵀX/n，其条件数必须另测，不能直接套用总体值。活跃变量结构也会影响算法速度。"),
        ["h", "3.2　可计算的优化证书"],
        eq(r"D(v)=\frac{2y^\top v-\|v\|_2^2}{2n},\quad \|X^\top v\|_\infty\leq n\lambda"),
        eq(r"r=y-Xw,\quad c=\max\{1,\|X^\top r\|_\infty/(n\lambda)\},\quad v=r/c"),
        eq(r"0\leq P(w)-P^*\leq G=P(w)-D(v),\quad \widetilde G=G/\max\{1,|P|,|D|\}"),
        p("可行对偶点给出最优目标的下界；该尺度下的 v 不能与除以 n 的另一种对偶变量混用。实现对缩放因子加入浮点裕量，检查可行性和 KKT 残差，仅裁剪舍入级负间隙。归一化间隙是本项目的停止约定，并非统计恢复证书。")
    ])
    pages.append([
        ["h", "4　三种求解方法"],
        ["h2", "4.1　软阈值与 ISTA"],
        eq(r"S_\tau(a)=\operatorname{sign}(a)\max\{|a|-\tau,0\},\quad w^{k+1}=S_{\lambda/L}(w^k-\nabla f(w^k)/L)"),
        p("软阈值是 ℓ1 惩罚的近端映射，可由一维二次距离加绝对值惩罚的正、负、零三段最小化得到。使用有效 Lipschitz 上界时，ISTA 的二次上界步骤保证目标不增。各算法统一从零初始化。"),
        ["h2", "4.2　FISTA"],
        eq(r"w^{k+1}=S_{\lambda/L}(z^k-\nabla f(z^k)/L),\quad t_{k+1}=(1+\sqrt{1+4t_k^2})/2"),
        eq(r"z^{k+1}=w^{k+1}+\frac{t_k-1}{t_{k+1}}(w^{k+1}-w^k),\quad z^0=w^0=0,\ t_0=1"),
        p("标准 FISTA 在外推点求梯度，增加向量级运算，但不强制每步目标单调。本文不使用重启。对可达最优值的凸复合目标，光滑项梯度 Lipschitz 连续、近端步骤精确并采用有效 L 时，经典界为[[beck2009]]："),
        eq(r"P(w^k_{\mathrm{ISTA}})-P^*\leq\frac{L\|w^0-\widehat{w}_\lambda\|_2^2}{2k},\quad P(w^k_{\mathrm{FISTA}})-P^*\leq\frac{2L\|w^0-\widehat{w}_\lambda\|_2^2}{(k+1)^2}"),
        p("O(1/k) 与 O(1/k²) 是最坏情形目标残差上界，不要求强凸性，也不等于可计算对偶间隙的精确速率；它们不保证所有有限精度和实现中的时间排名。"),
        ["h2", "4.3　循环坐标下降"],
        eq(r"a_j=\|X_j\|_2^2/n,\quad q_j=X_j^\top r/n+a_jw_j,\quad w_j^+=S_\lambda(q_j)/a_j"),
        eq(r"r\leftarrow r-X_j(w_j^+-w_j),\quad w_j\leftarrow w_j^+"),
        p("固定其余坐标后精确求解一维子问题，依次更新全部特征为一个 epoch。增量残差避免每次重算 Xw。对于本目标，可分 ℓ1、光滑二次项、紧水平集及非零列的一维严格凸子问题满足相应收敛分析条件[[tseng2001]][[friedman2010]]；不据此宣称任意相关性下具有统一线性收敛常数。"),
        table(["方法", "一次计数", "稠密主要成本", "额外机制"], [["ISTA", "近端迭代", "O(np)", "可靠步长"], ["FISTA", "近端迭代", "O(np)", "外推向量"], ["CD", "完整 epoch", "O(np)", "增量残差"]]),
        p("三者除存储 X 外使用 O(n+p) 工作存储。相同的大 O 不代表相同常数或访存成本。L 由稠密 SVD 计算，近端方法共享其结果；其额外成本在独立启动口径中加回。CD 使用可读 C99 内核，近端方法调用 NumPy/BLAS，性能结论限定于这些实现。")
    ])
    pages.append([
        ["h", "5　实验协议与公平性"],
        table(["合成因素", "固定设置"], [["规模", "训练 n=2000、p=1000；独立测试 n=1000"], ["支持与信号", "20 个非零，位于 20 个不同组，系数 ±1"], ["相关性", "ρ=0、0.3、0.6、0.9；每组 20 列"], ["正则化", "λ/λmax=0.01、0.05、0.1；λmax 用训练数据计算"], ["随机性", "种子 0–4；相同种子跨 ρ 配对随机来源"], ["噪声", "高斯；训练信号方差/噪声总体方差=10"], ["主网格", "4×3×5×3=180 次求解"]]),
        p("先生成原始特征，再用训练参数标准化，并在该坐标定义真系数。噪声尺度使用训练信号方差，有限样本实现的信噪比另记，不能声称每组样本方差比恰好等于 10。相同相对 λ 不保证不同 ρ 的绝对 λ 相同，因此相关性比较也包含这一规则下惩罚强度的变化。"),
        ["h2", "5.1　真实数据与参数选择"],
        p("UCI Online News Popularity[[fernandes2015]] 官方 CSV 实测 39644 行、61 列。排除 url、timedelta 和作为目标的 shares 后，使用 58 个数值特征；目标为 log1p(shares)。数据许可为 CC BY 4.0，CSV SHA256 与官方 ZIP、许可归属均已保存。官方页面行数元数据与文件实测值不同，本文采用文件实测值。"),
        p("三个划分种子 0–2 各固定训练/验证/测试为 23786/7928/7930。仅训练集估计均值和标准差。在 20 个对数等距 λ/λmax∈[0.001,1] 上用高精度参考解按验证 MSE 选最佳点，再取最佳点及距离最近的两点，让三算法共用三档 λ。测试集只计算最终模型误差，共 27 次比较，不再合并重拟合。随机划分评估 IID 留出预测，不代表未来新闻预测。"),
        ["h2", "5.2　精度、计时与记录"],
        p("单 CPU 线程、float64、零初始。每 5 步/epoch 计算证书和指标，以归一化间隙≤1e-6 停止；另提取首次达到 1e-2、1e-4 的检查点。每次上限 120 秒、100000 步，所有失败或超时状态保留。检查点之间的真实穿越时间未知，图中时间是观测上界。"),
        p("求解时间包含初始化与全部证书/指标回调，单独记录诊断开销。精确 SVD、数据准备、验证选择、参考求解另计；独立启动时间为近端求解加 SVD，CD 无需 SVD。C 编译/加载属于单次环境成本。算法顺序按固定种子打乱，结果显示五种子的中位数与四分位距，后者是描述性离散程度，不是置信区间。"),
        p("用 scikit-learn Lasso 在相同缩放下核对目标和证书，参考间隙要求≤1e-8。支持阈值预定 1e-6，另查 1e-8/1e-4。18 项测试覆盖正交闭式解、相关设计参考解、λmax 零解、可行性、停止行为、种子配对和预处理隔离。"),
        p("硬件：Intel Xeon Gold 6278C @2.60GHz，可见 8 个逻辑 CPU，BLAS/OpenMP 固定 1 线程；Python 3.13.13、NumPy 2.5.1、SciPy 1.18.0、scikit-learn 1.9.0，GCC 15.2.0。共享机器未做 CPU 独占绑定，因此小时间差不宜过度解释。")
    ])
    pages.append([
        ["h", "6　求解成本与相关性"],
        image("convergence", "图 1　种子 0 的代表性间隙轨迹。横轴为含诊断的求解秒数；CD 的 epoch 与近端迭代含义不同。", 225),
        image("correlation_time", "图 2　达到 1e-6 的求解耗时，中位数与四分位距，三个分面对应相对正则化。180/180 收敛。", 173),
        p(f"在 λ/λmax=0.05 时，ρ=0 到 0.9 的中位求解耗时分别由 ISTA {g(0,.05,'ista')['elapsed_sec']['median']:.3f} 增至 {g(.9,.05,'ista')['elapsed_sec']['median']:.3f} 秒、FISTA {g(0,.05,'fista')['elapsed_sec']['median']:.3f} 增至 {g(.9,.05,'fista')['elapsed_sec']['median']:.3f} 秒、CD {t0['elapsed_sec']['median']:.3f} 增至 {t9['elapsed_sec']['median']:.3f} 秒。相关性明显增加本组问题的计算成本；结合样本曲率诊断可解释这一趋势，但不是对所有 Lasso 设计的定理。"),
        p(f"低相关下，FISTA 的高精度时间可能高于 ISTA；ρ=0.9 时两种近端方法的差距反向。标准 FISTA 在 {fista['runs_with_increases']}/{fista['n_runs']} 次运行中出现检查点间目标上升，共观察 {fista['total_observed_increases']} 次。这是记录频率下的观察次数，不是逐迭代总数。CD 在本组合成实现与精度下最快，不能据此把实现差异全部归因于算法理论。")
    ])
    pages.append([
        ["h", "7　恢复效果与优化精度"],
        image("recovery", "图 3　最终系数相对误差与支持 F1。曲线重叠反映相同目标精度下三种求解器的统计结果接近。", 235),
        image("accuracy_recovery", "图 4　λ/λmax=0.05，首次观测到不同间隙阈值时的恢复指标。精度提高不保证指标单调改善。", 220),
        p(f"例如 CD 在 λ/λmax=0.05 时，ρ=0 的中位相对系数误差为 {med(g(0,.05)['relative_coef_error']):.3f}，ρ=0.9 为 {med(g(.9,.05)['relative_coef_error']):.3f}。但弱惩罚 0.01 下，支持 F1 由 {med(g(0,.01)['f1']):.3f} 升至 {med(g(.9,.01)['f1']):.3f}，同时系数误差由 {med(g(0,.01)['relative_coef_error']):.3f} 增至 {med(g(.9,.01)['relative_coef_error']):.3f}。这是不同恢复指标衡量不同性质的实测反例。"),
        p(f"在高相关、相对 λ=0.05 的 ISTA 轨迹中，首次达到 1e-2/1e-4/1e-6 时，中位系数误差为 {med(th(.9,'ista',.01)['relative_coef_error']):.3f}/{med(th(.9,'ista',.0001)['relative_coef_error']):.3f}/{med(th(.9,'ista',.000001)['relative_coef_error']):.3f}。粗精度阶段的未完全求解状态可显著影响指标；接近最优后则主要受目标本身的偏差、噪声与相关结构限制。")
    ])
    real_rows = sorted(summary["real"]["best_lambda_rows"], key=lambda x: (x["seed"], ["ista", "fista", "cd"].index(x["method"])))
    pages.append([
        ["h", "8　真实数据与数值核查"],
        p("表 3 仅列验证选择的最佳 λ，每个划分在三算法之间共享。其余两档及全部 27 次求解保存在 real_runs.csv，未按测试误差挑选模型。MSE 单位为 log1p(shares) 的平方，不能解读为原始分享次数误差。"),
        table(["种子", "方法", "λ/λmax", "非零数", "验证 MSE", "测试 MSE", "秒", "归一间隙"],
              [[str(x["seed"]), x["method"].upper(), f"{x['lambda_ratio']:.4f}", str(x["nnz"]), f"{x['validation_mse']:.4f}", f"{x['test_mse']:.4f}", f"{x['elapsed_sec']:.3f}", f"{x['normalized_gap']:.1e}"] for x in real_rows]),
        p("三个种子的最佳验证参数下，各方法的测试 MSE 接近，耗时排名可能与合成数据不同。这符合它们求解相同目标的预期；真实数据中强相关、共线性及活跃集结构与合成机制不同，不能按低维/高维标签直接推断排序。真实非零系数数目只是模型稀疏度，不代表找到的真实因素数量。"),
        ["h2", "8.1　精度与曲率证据"],
        p(f"207/207 次主比较均达到目标。合成数据相对参考目标的最大归一化绝对差为 {q['synthetic']['max_normalized_absolute_objective_delta_to_reference']:.2e}，真实数据为 {q['real']['max_normalized_absolute_objective_delta_to_reference']:.2e}。全部检查点最大对偶约束比为 {max(q['synthetic']['max_dual_feasibility_ratio_all_checkpoints'],q['real']['max_dual_feasibility_ratio_all_checkpoints']):.12f}，未超过可行上界 1。"),
        table(["ρ", "总体块条件数", "样本 Hessian 条件数中位数", "组内相关均值中位数"],
              [[f"{x['rho']:g}", f"{(1+19*x['rho'])/(1-x['rho']):.2f}", f"{x['sample_hessian_condition']['median']:.1f}", f"{x['sample_within_block_correlation']['median']:.3f}"] for x in structure["by_rho"]]),
        p(f"支持阈值从 1e-8 到 1e-4 的检查中，{summary['support_sensitivity']['runs_with_any_f1_change']}/180 次运行的 F1 有变化，最大变化 {summary['support_sensitivity']['within_run_f1_range']['max']:.4f}。这项检查只衡量声明范围内的数值阈值敏感性，不检验特征选择的总体稳定性。"),
        ["h2", "8.2　诊断频率敏感性"],
        p(f"将记录间隔从 5 提高至 20，在种子 0 的低/高相关、两档 λ、三算法上另跑 12 次。最终 1e-6 精度的 {sensitivity['matching_final_tolerance_count']}/{sensitivity['comparable_final_tolerance_count']} 组算法排序一致；三个精度合计 {sensitivity['matching_ranking_count']}/{sensitivity['comparable_ranking_count']} 组一致。粗精度排序可能受采样粒度影响，不能把有限检查点时间当作精确的阈值穿越时间。每种设置仅补一次计时，不构成计时置信区间。")
    ])
    pages.append([
        ["h", "9　讨论、局限与结论"],
        image("correlation_cold_start", "图 5　近端方法加回精确 SVD 的独立启动口径；CD 不需要 SVD。数据准备与环境加载成本另记。", 165),
        p("对研究问题一，本实验中增大相关性显著增加达到统一精度的时间，样本曲率也更不均衡；但相关性、相对 λ 规则、活跃集和有限样本共同决定难度，不能宣称 ρ 是唯一原因。对问题二，FISTA 的优势依赖精度与设计：最坏情形理论界并不强制每个有限实例更快，标准外推允许非单调。对问题三，优化精度改善到足够高后仍留下统计误差，不保证系数与支持指标同步或单调改善。"),
        p("弱惩罚下大量小系数带来假阳性，相关性增加时支持 F1 可因选择规模变化而提高，同时系数误差变大。因此应同时报告 precision、recall、F1、系数误差和预测误差，而非选取单个有利指标。不同算法在相同目标精度下的最终统计指标接近，进一步表明主要统计差异来自问题与正则化，而非算法名称。"),
        p("局限包括：仅五个合成种子，四分位距不是显著性检验；真支持限制为不同组各一个变量，未覆盖同组多个真信号；每个配对设置没有独立多次计时；NumPy/BLAS 与 C99 内核常数不同，机器未独占；相同相对 λ 下绝对 λ 随数据变化；只有一个真实数据集，目标变换和随机划分限定其解释，且无法评估真实支持。本文不证明新算法优越性，也不把表中的排名推广到稀疏矩阵、路径热启动、筛选或其他硬件。"),
        p("结论是应把计算精度与统计恢复分开评估。在本项目规定的机制与实现下，CD 的合成求解时间较低，FISTA 对高相关设计的加速存在但并非普遍有限精度优势，相关性对恢复的影响依赖评价指标及正则化水平。后续可检验同组多信号、固定绝对 λ 和更多真实数据；这些不属于本次已完成实验。"),
        ["h2", "复现与代码获取"],
        p("代码与完整结果随本地 comp6704_research_bundle.zip 提供。主入口 scripts/run_study.py，图表一条命令 scripts/analyze.py；README 记录安装、GCC、数据下载、快速检查及完整步骤。每次运行保存配置、数据哈希、数值源文件签名、轨迹、系数和环境；六个代表性设置已独立重跑核对。提交时需把实际可访问的代码下载位置写入最终报告。"),
        p("资源记录以 results/*/timing.json 为准；必做实验在 2026 年 10 月 28 日提前完成，未使用 GPU，未开展可选扩展。报告属于待作者核验的中文草稿，姓名和学号按要求留空，最终代码获取链接待定稿时补入；课程截止为 2026 年 11 月 29 日 23:59。实验实现、运行和本草稿整理使用了 Codex 辅助。")
    ])
    pages.append([["h", "参考文献"], ["refs"], ["h2", "数据与源材料"],
                  p("数据源为 UCI 官方 Online News Popularity，DOI 10.24432/C5NS3V，CC BY 4.0。下载文件及处理元数据见 data/raw/source.json 和 docs/data_source.md。课程约束来自所附 COMP6704 Individual Assignment 2026 Fall，研究网格、预算与内部日期来自用户研究计划，两者在 docs/execution_plan.md 中区分。")])
    return pages


def render_pdf(pages, entries, target, table_captions=None):
    font_path = ROOT / "report/chinese.ttf"
    if not font_path.exists():
        font_path.write_bytes(pymupdf.Font("cjk").buffer)
    pdfmetrics.registerFont(TTFont("Chinese", str(font_path)))
    base = ParagraphStyle("body", fontName="Chinese", fontSize=9.7, leading=15.5, spaceAfter=6, wordWrap="CJK")
    styles = {
        "p": base,
        "title": ParagraphStyle("title", parent=base, fontSize=17, leading=25, alignment=TA_CENTER, spaceAfter=12),
        "h": ParagraphStyle("heading", parent=base, fontSize=13, leading=20, spaceBefore=4, spaceAfter=8),
        "h2": ParagraphStyle("subheading", parent=base, fontSize=10.5, leading=17, spaceBefore=5, spaceAfter=6),
        "caption": ParagraphStyle("caption", parent=base, fontSize=8.2, leading=12, textColor=colors.HexColor("#455468"), spaceAfter=7),
        "small": ParagraphStyle("small", parent=base, fontSize=7.9, leading=11.5, spaceAfter=5),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8.2, leading=12, spaceAfter=0),
    }
    numbers = {e["key"]: i + 1 for i, e in enumerate(entries)}
    def text(s):
        s = re.sub(r"\[\[(.*?)\]\]", lambda m: f"[{numbers[m.group(1)]}]", s)
        return html.escape(s).replace("\n", "<br/>")
    story = []
    assets = ROOT / "report/assets"
    assets.mkdir(exist_ok=True)
    width = 487
    equation_number = 0
    table_number = 0
    table_captions = table_captions or ["三种方法的稠密计算成本", "合成实验固定设置", "真实数据最佳验证参数结果", "总体结构与有限样本曲率"]
    for page_index, page in enumerate(pages):
        if page_index:
            story.append(PageBreak())
        for index, block in enumerate(page):
            kind = block[0]
            if kind in styles:
                story.append(Paragraph(text(block[1]), styles[kind]))
            elif kind == "eq":
                path = assets / f"equation_{page_index}_{index}.png"
                math_to_image("$" + block[1] + "$", str(path), dpi=220, format="png", color="#182433")
                from PIL import Image as PILImage
                with PILImage.open(path) as im:
                    iw, ih = im.size
                scale = min((width - 34) / iw, 0.43)
                equation_number += 1
                equation_table = Table([[Image(str(path), width=iw * scale, height=ih * scale),
                                         Paragraph(f"({equation_number})", styles["small"])]],
                                       colWidths=[width-29,29])
                equation_table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                                                    ("TOPPADDING",(0,0),(-1,-1),0),
                                                    ("BOTTOMPADDING",(0,0),(-1,-1),0)]))
                story.append(equation_table)
                story.append(Spacer(1, 9))
            elif kind == "image":
                path = ROOT / "figures" / (block[1] + ".png")
                from PIL import Image as PILImage
                with PILImage.open(path) as im:
                    iw, ih = im.size
                scale = min(width / iw, block[3] / ih)
                story.append(Image(str(path), width=iw * scale, height=ih * scale))
                story.append(Paragraph(text(block[2]), styles["caption"]))
            elif kind == "table":
                header, rows = block[1], block[2]
                table_number += 1
                story.append(Paragraph(f"表 {table_number}　{table_captions[table_number-1]}", styles["caption"]))
                raw = [header] + rows
                cells = [[Paragraph(text(str(c)), styles["cell"]) for c in row] for row in raw]
                if len(header) == 2:
                    widths = [88, width - 88]
                elif len(header) == 8:
                    widths = [30, 42, 56, 42, 71, 71, 57, 118]
                    widths = [x * width / sum(widths) for x in widths]
                else:
                    widths = [width / len(header)] * len(header)
                t = Table(cells, colWidths=widths, repeatRows=1, hAlign="CENTER")
                t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef4")),
                                      ("LINEABOVE", (0, 0), (-1, 0), .6, colors.HexColor("#6e7e90")),
                                      ("LINEBELOW", (0, 0), (-1, 0), .4, colors.HexColor("#aab6c4")),
                                      ("LINEBELOW", (0, -1), (-1, -1), .6, colors.HexColor("#6e7e90")),
                                      ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 5),
                                      ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
                story.extend([t, Spacer(1, 9)])
            elif kind == "refs":
                for i, e in enumerate(entries):
                    link = f' <link href="{html.escape(e["url"],quote=True)}" color="#315d86">原文</link>' if e["url"] else ""
                    story.append(Paragraph(f"[{i+1}] " + text(e["text"]) + link, styles["small"]))
            else:
                raise ValueError(kind)
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Chinese", 8)
        canvas.setFillColor(colors.HexColor("#68788c"))
        canvas.drawString(54, 27, "COMP6704 · 中文草稿 · 2026-10-28")
        canvas.drawRightString(541, 27, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(target), pagesize=(595.28,841.89), leftMargin=54, rightMargin=54,
                            topMargin=42, bottomMargin=43, title=pages[0][0][1], author="COMP6704 report draft")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def render_tex(pages, target, table_captions=None):
    (target.parent / "figures").mkdir(exist_ok=True)
    def escape(s):
        citations = []
        def cite(m):
            citations.append(r"\cite{" + m.group(1) + "}")
            return f"CITATIONTOKEN{len(citations)-1}END"
        s = re.sub(r"\[\[(.*?)\]\]", cite, s)
        for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
            s = s.replace(a,b)
        for a, b in {"ρ":r"\ensuremath{\rho}","λ":r"\ensuremath{\lambda}","ℓ":r"\ensuremath{\ell}","ᵀ":r"\ensuremath{^\top}","²":r"\ensuremath{^2}","≤":r"\ensuremath{\leq}","∈":r"\ensuremath{\in}","ε":r"\ensuremath{\varepsilon}","×":r"\ensuremath{\times}","→":r"\ensuremath{\to}","−":"-"}.items():
            s=s.replace(a,b)
        for i,c in enumerate(citations):
            s=s.replace(f"CITATIONTOKEN{i}END",c)
        return s.replace("\n",r"\\ ")
    lines = [r"\documentclass[UTF8,fontset=fandol,10pt]{ctexart}",
             r"\usepackage[a4paper,left=19mm,right=19mm,top=16mm,bottom=16mm]{geometry}",
             r"\usepackage{amsmath,amssymb,graphicx,booktabs,tabularx,hyperref}",
             r"\hypersetup{colorlinks=true,urlcolor=blue,citecolor=blue}",
             r"\setlength{\parindent}{0pt}\setlength{\parskip}{4pt}",
             r"\setlength{\emergencystretch}{2em}",r"\begin{document}"]
    table_number = 0
    table_captions = table_captions or ["三种方法的稠密计算成本", "合成实验固定设置", "真实数据最佳验证参数结果", "总体结构与有限样本曲率"]
    for i,page in enumerate(pages):
        if i:lines.append(r"\clearpage")
        for block in page:
            k=block[0]
            if k=="title":lines.append(r"\begin{center}\Large\bfseries "+escape(block[1])+r"\end{center}")
            elif k=="h":lines.append(r"\section*{"+escape(block[1])+"}")
            elif k=="h2":lines.append(r"\subsection*{"+escape(block[1])+"}")
            elif k=="p":lines.append(escape(block[1])+"\n")
            elif k=="eq":lines.append(r"\begin{equation}"+block[1]+r"\end{equation}")
            elif k=="image":
                figure_target=target.parent / "figures" / (block[1]+".pdf")
                figure_target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT / "figures" / (block[1]+".pdf"), figure_target)
                lines.append(r"\begin{center}\includegraphics[width=\linewidth,height="+f"{block[3]*.35278:.1f}"+r"mm,keepaspectratio]{figures/"+block[1]+r".pdf}\\{\small "+escape(block[2])+r"}\end{center}")
            elif k=="table":
                n=len(block[1]);cols="l"+"X"*(n-1)
                table_number += 1
                lines += [r"\begin{center}{\small 表 "+str(table_number)+"　"+table_captions[table_number-1]+r"}\\\small\begin{tabularx}{\linewidth}{"+cols+"}",r"\toprule",
                          " & ".join(escape(str(c)) for c in block[1])+r" \\ \midrule"]
                lines += [" & ".join(escape(str(c)) for c in row)+r" \\" for row in block[2]]
                lines += [r"\bottomrule\end{tabularx}\end{center}"]
            elif k=="refs":lines += [r"\renewcommand{\refname}{}\bibliographystyle{unsrt}\bibliography{references}"]
    lines.append(r"\end{document}")
    target.write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",default="")
    parser.add_argument("--student-id",default="")
    args=parser.parse_args()
    summary=read("results/summary.json")
    if not summary["integrity"]["complete_grid"]:
        raise SystemExit("Cannot build the report from partial results")
    author={"name":args.name,"student_id":args.student_id}
    pages=build_pages(summary,read("results/structure.json"),read("results/diagnostic_sensitivity.json"),author)
    entries=bib_entries()
    # Include verified later developments when available, keeping numbering from BibTeX order.
    keys={e['key'] for e in entries}
    later=[]
    if 'bertrand2021' in keys:later.append("坐标下降的 Anderson 外推[[bertrand2021]]")
    if 'moufad2025' in keys:later.append("2025 年的 skglm 求解库[[moufad2025]]")
    if later:pages[0][-1][1] += "更近期工作包括"+"与".join(later)+"，本文引用其说明实现发展，不声称已复现其加速结果。"
    (ROOT/'report/content_zh.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2))
    render_pdf(pages,entries,ROOT/'report/report_zh.pdf')
    render_tex(pages,ROOT/'report/main.tex')
    doc=pymupdf.open(ROOT/'report/report_zh.pdf')
    if len(doc)>10:raise SystemExit(f"Report has {len(doc)} pages; reduce layout before delivery")
    if (ROOT/'report/report_zh.pdf').stat().st_size>=20_000_000:raise SystemExit('Report exceeds 20 MB')
    checks={'pdf_pages':len(doc),'pdf_bytes':(ROOT/'report/report_zh.pdf').stat().st_size,
            'identity_complete':bool(args.name.strip() and args.student_id.strip()),
            'identity_display':'blank lines for unspecified fields, as requested by the user',
            'page_text_chars':[len(page.get_text()) for page in doc], 'references':len(entries),
            'pdf_renderer':'ReportLab; all pages raster-previewed',
            'tex_validation':'Source provided; desktop standalone compiler lacks multifile project support. Compile with full XeLaTeX/BibTeX environment.'}
    (ROOT/'report/validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    preview=ROOT/'report/preview';preview.mkdir(exist_ok=True)
    for i,page in enumerate(doc):
        page.get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).save(str(preview/f'page_{i+1:02}.png'))
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
