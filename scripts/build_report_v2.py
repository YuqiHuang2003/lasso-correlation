#!/usr/bin/env python3
"""Generate the Chinese v2 report from completed correlation x sparsity results."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_report import bib_entries,render_pdf,render_tex,pymupdf

LABELS={'ista':'ISTA','fista':'FISTA','cd':'CD','skglm':'skglm'}
CAPTIONS=['2023年以来的核心研究及本项目取舍','方法与实际计算口径','新版合成实验因素',
          'UJI验证选择模型的跨划分结果','Leukemia固定目标的三次计时中位数']


def read(path):return json.loads((ROOT/path).read_text())


def make_pages(summary,author):
    cfg=summary['config'];grid=summary['by_cell'];quality=summary['quality']
    def cell(rho,s=20,layout='dispersed',c=1.,method='cd'):
        return next(x for x in grid if x['rho']==rho and x['sparsity']==s and x['layout']==layout and x['lambda_factor']==c and x['method']==method)
    def diag(rho,s=40,layout='dispersed'):
        return next(x for x in summary['structure']['by_cell'] if x['rho']==rho and x['sparsity']==s and x['layout']==layout)
    def hit(layout,tol,method='ista'):
        return next(x for x in summary['threshold_recovery'] if x['rho']==.9 and x['sparsity']==40 and x['layout']==layout and x['lambda_factor']==1 and x['method']==method and x['threshold']==tol)
    def m(d):return d['median']
    def f(value,precision=3):return '未达标' if value is None else f'{value:.{precision}f}'
    p=lambda t:['p',t];h=lambda t:['h',t];h2=lambda t:['h2',t];eq=lambda t:['eq',t]
    image=lambda name,cap,height:['image','v2/'+name,cap,height]
    table=lambda head,rows:['table',head,rows]
    display_name=author['name'].strip() or '________________';display_id=author['student_id'].strip() or '________________'
    main=quality['synthetic'];low=cell(0,40);high=cell(.9,40)
    clustered=cell(.9,40,'clustered')
    pages=[]
    pages.append([
        ['title','特征相关性对 Lasso 求解效率与稀疏恢复的影响'],
        p(f'COMP6704 个人作业 · 中文研究报告草稿（第二版）\n姓名：{display_name}　学号：{display_id}\n日期：2026 年 10 月 28 日（Asia/Shanghai）'),
        h('摘要'),
        p(f"本文把特征相关性、真实非零数量和支持布局共同作为实验因素，比较优化效率与稀疏恢复。新版固定噪声、单个信号幅度和三档绝对正则化，在 n=500、p=1000 下构造360个目标，由ISTA、FISTA、循环坐标下降与skglm求解，共1440次运行，其中{main['n_converged']}次在预算内通过统一1e-6归一化对偶间隙。恢复评价覆盖误选、漏选、精确支持、符号与组支持，而不以非零数代替恢复质量。另用近期文献采用的UJIIndoorLoc与Leukemia完成72次真实设计比较。结果显示，求解成本与恢复效果应按相关性、稀疏度、布局及惩罚联合解释；达到同一目标精度不意味着不同数据上的真实支持同样容易恢复。"),
        h('1　研究问题与近期研究定位'),
        p('题目保持“特征相关性对Lasso求解效率与稀疏恢复的影响”。研究问题为：（1）固定噪声、系数幅度及惩罚时，相关性和真实非零数如何共同影响求解时间？（2）信号分散或集中在同组时，误选、漏选及符号恢复是否相同？（3）进一步缩小优化间隙能改善多少恢复效果？项目应用凸性、梯度步长、非光滑正则化、坐标最小化和对偶性，区别于小组的grokking神经网络优化器课题。'),
        p('近期求解研究已关注正则化路径与模型选择[[tay2023glmnet]]、排序稀疏惩罚的混合更新[[larsson2023slope]]、大规模变量筛选[[zogala2024screening]]和相关设计下的路径计时[[yang2024adelie]]。这些工作说明实际效率还受热启动、活跃集、罚项及预处理影响。本文保留三种基础方法解释机制，并增加skglm作为现代实现参照。'),
        p('恢复方面，2025年的符号恢复设计研究强调效应方向和设计结构[[stallrich2025]]；部分旋转设计理论区分不同列集合之间的几何[[buhai2025]]；平均情形研究还显式处理相关设计与污染[[minsker2025]]。skglm的软件基准[[moufad2025]]和2026年核心集研究[[zou2026coreset]]进一步提示应联合检查速度、目标质量与稀疏性。本文据此重设计实验，不宣称提出新算法或首次研究相关性。')
    ])
    pages.append([
        h('2　近期文献的数据、内容与采用边界'),
        table(['工作与年份','论文的数据或设计','本项目采用的启发与边界'],[
            ['glmnet，2023[[tay2023glmnet]]','模拟计时；Spambase分类','完整路径、热启动和模型选择，与独立单目标计时分开'],
            ['SLOPE CD，2023[[larsson2023slope]]','相关合成；bcTCGA、news20、rcv1、Rhee2006','检查稀疏真值及多种误差；不把SLOPE算法当作普通Lasso CD'],
            ['MI-Lasso，2024[[zogala2024screening]]','rat-eye 120×31098；超高维Eurostat','区分计算活跃集和真实支持；区分索引构建与求解成本，近似查询不套用严格safe保证'],
            ['adelie，2024预印本[[yang2024adelie]]','相关高斯路径；Leukemia、UJIIndoorLoc、Prostate','采用其两套真实设计；主实验仍为独立目标，不声称复现完整路径'],
            ['符号恢复设计，2025[[stallrich2025]]','两水平超饱和设计，含已知/未知符号','增加支持布局和符号指标，不把条件性结果或猜想当作普遍保证'],
            ['部分旋转，2025预印本[[buhai2025]]','半随机相关设计；主要为理论','记录活跃Gram及相对方向诊断；未施加论文的随机旋转'],
            ['污染与相关设计，2025[[minsker2025]]','5列相关块，n1000、p1100，稀疏污染','借鉴分组和支持/符号控制；本轮不实现鲁棒增广目标'],
            ['skglm，2025[[moufad2025]]','MEG/Lasso；癌症/Cox；药效/Group Lasso','实际加入安装版本0.5；库容差与后验统一精度分开'],
            ['Lasso核心集，2026[[zou2026coreset]]','Synthetic、mediamill、CTs、mnist8m','真实设计配模拟响应；行抽样近似不同于完整目标的精度，不实现抽样']]),
        h2('2.1　由文献落实到实验的改变'),
        p('第一版只有一个真实非零数，且改变设计时同时按信号方差调节噪声、按λmax调节惩罚，难以单独解释恢复差异。第二版增加s=5/20/40和两种支持布局，固定σ=1与绝对λ；真实数据改为文献中的UJI回归和Leukemia高维设计。原版180次合成与Online News结果保留为初步实验，不混入本版统计。'),
        p('每个固定目标共用数据、λ和零初始。skglm包含工作集与外推，提供当代软件背景；它的内部检查频率和停止量不同，不能将速度差全部归因于CD更新。文献中“筛选后留下多少计算变量”和“找回多少真信号”是不同问题，本实验分别记录。')
    ])
    pages.append([
        h('3　模型、相关结构与稀疏恢复'),
        eq(r'\min_{w\in\mathbb{R}^p}P(w),\qquad P(w)=\frac{\|Xw-y\|_2^2}{2n}+\lambda\|w\|_1'),
        p('平方损失凸且光滑，ℓ1项凸但在零点不可微；λ>0保证最优解存在。X和y按训练数据中心化，X按训练总体标准差缩放，验证/测试沿用参数，相当于保留未惩罚截距。合成真系数定义在该标准化坐标，取值±1。'),
        eq(r'x^{\mathrm{raw}}_{i,hj}=\sqrt{\rho}\,z_{i,h}+\sqrt{1-\rho}\,e_{i,hj},\quad y=Xw^\star+\varepsilon,\quad\varepsilon_i\sim\mathcal{N}(0,1)'),
        p('组大小20，不同组独立；dispersed每个活跃组一个信号，clustered每个活跃组五个信号。相同种子跨ρ、s、布局共用设计分量、噪声实现与有序符号，同一布局的支持随s嵌套。固定σ和单个信号幅度保留了总信号能量及符号抵消的变化，SNR逐例记录。'),
        eq(r'\Sigma_b=(1-\rho)I+\rho\mathbf{1}\mathbf{1}^\top,\quad V_g=(1-\rho)k_g+\rho m_g^2'),
        p('k_g为组内真信号数，m_g为其符号和。上式是理想单位方差总体块的代数关系，不能替代训练标准化后的实测信号方差。混合符号可使高相关组内信号抵消；相同ρ并不代表相同恢复难度。'),
        eq(r'E_{\mathrm{coef}}=\|w-w^\star\|_2/\|w^\star\|_2,\quad \widehat S_\tau=\{j:|w_j|>\tau\},\quad \tau=10^{-6}'),
        eq(r'\mathrm{Prec}=TP/(TP+FP),\quad\mathrm{Rec}=TP/s,\quad F_1=2TP/(2TP+FP+FN),\quad\mathrm{FDR}=FP/(TP+FP)'),
        p('同时记录精确支持恢复、阈值后整个符号向量完全正确、真活跃符号正确率，以及组precision/recall/F1。空选择约定precision/recall/F1/FDR均为0，必须结合选择数解释。组支持允许同组替代，不能代替真实变量恢复；两布局的真组数量不同，也不能仅按组F1横向判优。五种子的精确恢复显示成功数/5，不作高精度概率估计。'),
        eq(r'I_S=\|G_{S^cS}G_{SS}^{-1}\operatorname{sign}(w^\star_S)\|_\infty,\quad G=X^\top X/n'),
        p('由于p>n且中心化，全Hessian奇异。采用真活跃Gram的最小特征值/条件数与样本I_S作为事后几何诊断，不作为算法先验，也不把I_S<1当作有限样本恢复保证。模型预测误差、系数误差、支持和符号指标各自回答不同问题。')
    ])
    pages.append([
        h('4　求解方法、收敛与统一证书'),
        eq(r'\nabla f(w)=X^\top(Xw-y)/n,\quad L=\|X\|_2^2/n,\quad S_\tau(a)=\operatorname{sign}(a)\max(|a|-\tau,0)'),
        eq(r'w^{k+1}_{\mathrm{ISTA}}=S_{\lambda/L}(w^k-\nabla f(w^k)/L)'),
        p('软阈值是一维二次距离加绝对值惩罚的近端解。有效Lipschitz上界保证ISTA目标不增。FISTA在外推点z更新，t从1开始：'),
        eq(r'w^{k+1}=S_{\lambda/L}(z^k-\nabla f(z^k)/L),\quad t_{k+1}=(1+\sqrt{1+4t_k^2})/2'),
        eq(r'z^{k+1}=w^{k+1}+\frac{t_k-1}{t_{k+1}}(w^{k+1}-w^k)'),
        p('对具有最优解的凸复合目标，光滑项梯度Lipschitz连续、近端精确且步长有效时，ISTA/FISTA的经典目标残差上界分别为O(1/k)、O(1/k²)[[beck2009]]。上界不等于实测对偶间隙的精确斜率，也不保证每个有限精度更快；本轮FISTA不重启。'),
        eq(r'a_j=\|X_j\|_2^2/n,\quad w_j^+=S_\lambda(X_j^\top r/n+a_jw_j)/a_j,\quad r\leftarrow r-X_j(w_j^+-w_j)'),
        p('循环CD精确最小化每个一维子问题，并增量维护残差；完整遍历为一个epoch[[friedman2010]]。可分ℓ1、光滑二次项、紧水平集和非零列的一维严格凸性质对应其收敛条件[[tseng2001]]，不宣称任意相关设计有统一线性常数。'),
        table(['方法','主更新/迭代口径','本轮实现'],[['ISTA/FISTA','每步O(np)，额外O(n+p)','NumPy/BLAS；共同SVD步长'],['循环CD','每完整epoch O(np)','C99增量残差，无SVD'],['skglm','外层工作集轮数，不等于epoch','0.5版AndersonCD，工作集+外推']]),
        eq(r'D(v)=(2y^\top v-\|v\|_2^2)/(2n),\quad r=y-Xw,\quad v=r/\max\{1,\|X^\top r\|_\infty/(n\lambda)\}'),
        eq(r'\|X^\top v\|_\infty\leq n\lambda,\quad0\leq P(w)-P^*\leq P(w)-D(v),\quad \widetilde G=\frac{P-D}{\max(1,|P|,|D|)}'),
        p('统一终点为归一化间隙≤1e-6。skglm本地KKT容差不等于该间隙，固定按1e-4/1e-6/1e-8从零重算直到后验合格，全部尝试累计计时；每次最多100个外层工作集轮，每子问题最多50000个内部epoch，最多拟合3次。只观察拟合端点，不混画密集精度轨迹。')
    ])
    pages.append([
        h('5　实验协议与复现'),
        table(['因素','冻结设置'],[['规模','n500、p1000；独立测试n1000；组大小20'],['相关性/稀疏度','ρ=0/0.3/0.6/0.9；s=5/20/40'],['布局/符号','每活跃组1或5个信号；配对混合±1'],['噪声/惩罚','固定σ1；λ=cσ√(2log(p)/n)，c=0.5/1/2'],['重复/方法','5个配对种子；ISTA、FISTA、CD、skglm'],['主网格','4×3×2×3×5×4=1440次求解'],['计算/基本上限','float64，单线程，零初始；基本方法20秒、100000步']]),
        p('三档λ约0.08311/0.16623/0.33245，同一c跨所有结构使用相同绝对惩罚；λmax和λ/λmax仅记录。固定σ与系数幅度使噪声不再随s或符号抵消改变；总体SNR自然变化，所以结论应表述为固定单个信号、噪声与惩罚的结构效应。参考求解器在相同缩放下独立获得更高精度目标。'),
        p('UJIIndoorLoc[[uji2014]]使用官方19937条训练数据池，目标LONGITUDE，仅WAP001–520为输入。三个随机60/20/20划分为11962/3987/3988；先划分再删除训练常量/完全重复列并标准化，实际保留452/452/451列。全池删除55常量可得文献的465列，但本轮不使用全池预处理。100的未探测编码保留；位置、楼层、身份和时间字段不进入X。12点相对λ网格在验证集选最佳及邻近两点，四算法共36次；测试不调参。'),
        p('Leukemia来自LIBSVM官方38+34条文件，合并72×7129并列标准化；二元标签编码后用于同一平方损失优化。三档相对λ、三次固定目标计时、四算法共36次，不报告泛化性能或真实支持恢复率。两数据均来自近期adelie普通Lasso实验；我们的划分、清洗和独立目标协议不等于论文的完整路径复现。'),
        p('基本算法每5步/epoch检查，阈值时间为首次观察上界。计时含初始化和诊断，预测/阈值敏感性另记；近端SVD、数据准备、C/Numba首次编译分别记录。基本算法在更新间检查20秒，skglm只能在库拟合间/返回后检查，超额明确标记。未收敛不混作成功时间，恢复统计同时显示状态与成功子集。'),
        p('32次代表性试跑全部收敛后冻结主网格；26项正确性测试通过，涵盖解析解、参考目标、证书、停止、配对和预处理。硬件Xeon Gold 6278C @2.60GHz，BLAS/OpenMP/Numba均1线程，Python3.13.13；NumPy2.5.1、SciPy1.18.0、skglm0.5。共享机器未独占，种子IQR是实例差异，非计时置信区间。')
    ])
    pages.append([
        h('6　相关性与稀疏度对求解成本的共同影响'),
        image('timing','图1　c=1的求解时间；行区分布局、列区分s，五种子中位数与IQR。skglm只比较终点总调用时间。',245),
        image('convergence','图2　种子0、clustered、c=1的代表轨迹，仅三种基本算法；迭代/epoch不等价。',215),
        p(f"在dispersed、s=40、c=1下，ρ=0→0.9时，ISTA中位耗时为{f(m(cell(0,40,method='ista')['elapsed_sec']))}→{f(m(cell(.9,40,method='ista')['elapsed_sec']))}秒，FISTA为{f(m(cell(0,40,method='fista')['elapsed_sec']))}→{f(m(cell(.9,40,method='fista')['elapsed_sec']))}，CD为{f(m(low['elapsed_sec']))}→{f(m(high['elapsed_sec']))}。这些是本设计与实现的条件性比较。"),
        p(f"在相同ρ=0.9、dispersed、c=1时，s=5与40的CD中位耗时分别为{f(m(cell(.9,5)['elapsed_sec']))}与{f(m(high['elapsed_sec']))}秒。现代基线含工作集、外推及容差重跑，迭代数不能与基本方法直接比较。更多信号改变活跃集和几何，仍须按布局与λ读图；加回SVD的独立启动图和全部未收敛计数另存，不只挑成功设置。")
    ])
    pages.append([
        h('7　稀疏恢复：找对变量、符号与组'),
        image('sparsity_rho_f1','图3　c=1、CD的特征F1中位数，分别展示两布局下ρ×s交互；其余方法保留完整逐次记录。',180),
        image('sparsity_rho_exact_support','图4　CD、c=1的精确支持恢复比例；每格仅5个种子，等价于成功数/5，不是高精度总体概率。',180),
        p(f"dispersed、c=1、s=40的CD中位F1为ρ=0时{m(low['f1']):.3f}、ρ=0.9时{m(high['f1']):.3f}；高相关clustered同设置为{m(clustered['f1']):.3f}。同时，估计非零数分别为{m(low['nnz']):.0f}、{m(high['nnz']):.0f}、{m(clustered['nnz']):.0f}，应结合precision={m(high['precision']):.3f}与recall={m(high['recall']):.3f}解释高相关分散设置，而非只看是否稀疏。"),
        p(f"以CD、c=1为例，低相关分散s=5的精确支持成功{cell(0,5)['exact_support_successes']}/5、完整符号成功{cell(0,5)['exact_sign_successes']}/5；高相关分散s=40分别为{high['exact_support_successes']}/5和{high['exact_sign_successes']}/5。这些离散计数比F1更严格，不能由较高平均F1直接推断完全恢复。"),
        p(f"在ρ=0.9、s=40时，两布局真活跃Gram条件数中位数分别为{m(diag(.9,40,'dispersed')['active_gram_condition']):.1f}与{m(diag(.9,40,'clustered')['active_gram_condition']):.1f}，样本不可表示分数为{m(diag(.9,40,'dispersed')['sample_irrepresentable_score']):.2f}与{m(diag(.9,40,'clustered')['sample_irrepresentable_score']):.2f}。不同诊断可朝不同方向变化；不能用某一个数单独解释恢复。"),
        p('组内替代可使组F1高于特征F1，但两布局真组数为s与s/5，基准难度不同。“选择全部50组”在s40分散布局也会得到较高组F1。因此报告分别保留真组数、组选中数、误选/漏选与完整符号指标；不把预测接近等同于找到相同变量。')
    ])
    pages.append([
        h('8　正则化、优化精度与恢复的区别'),
        image('coefficient_support_tradeoff','图5　CD终止解的五种子中位数；线连接相同ρ、s下三档c。系数误差、特征F1与组F1各有含义，不据测试结果挑阈值。',225),
        image('accuracy_recovery','图6　s40、ρ0.9、c1，首次观测到三个间隙阈值时的系数与特征恢复；skglm无同等密度轨迹，未混入。',215),
        p(f"例如dispersed的ISTA首次达到1e-2/1e-4/1e-6时，中位系数误差为{m(hit('dispersed',.01)['relative_coef_error']):.3f}/{m(hit('dispersed',.0001)['relative_coef_error']):.3f}/{m(hit('dispersed',.000001)['relative_coef_error']):.3f}。这些检查点相对于同一个λ目标；跨λ比较还包含正则化偏差，不能全部归因于求解精度。"),
        p(f"支持阈值1e-8/1e-6/1e-4的数值敏感性中，{summary['support_sensitivity']['runs_with_numerical_f1_change']}/{main['n_runs']}次运行F1有变化，最大变化为{summary['support_sensitivity']['within_run_f1_range_numerical_thresholds']['max']:.4f}。0.01/0.05是额外预定硬阈值，会改变估计支持，不作为Lasso原始支持或事后最优阈值。")
    ])
    uji={x['method']:x for x in summary['uji']['aggregated']}
    leu={(x['lambda_ratio'],x['method']):x for x in summary['leukemia']['aggregated']}
    pages.append([
        h('9　真实设计验证、讨论与局限'),
        table(['方法','UJI成功/3','终点测试MSE','终点非零数','成功求解秒'],
              [[LABELS[k],f"{uji[k]['n_converged']}/{uji[k]['n_runs']}",f(m(uji[k]['test_mse']),2),f(m(uji[k]['nnz']),0),f(m(uji[k]['elapsed_sec']))] for k in cfg['methods']]),
        p('上表按验证集最佳λ汇总中位数，MSE单位为原始经度坐标的平方。时间只汇总达标运行；MSE与非零数含20秒截断在内的所有终点，不据未达标预测比较同精度优劣。各档逐次记录在CSV，跨划分IQR在summary.json。随机留出不代表跨设备、时间外推或官方外部测试表现，真实支持未知。'),
        table(['方法','Leukemia λ/λmax=.01 秒','.1 秒','.3 秒'],
              [[LABELS[k]]+[f(m(leu[(a,k)]['elapsed_sec'])) for a in cfg['leukemia']['lambda_ratios']] for k in cfg['methods']]),
        p(f"Leukemia每个目标重复三次仅描述计时。新版各数据成功数为合成{quality['synthetic']['n_converged']}/{quality['synthetic']['n_runs']}、UJI{quality['uji']['n_converged']}/{quality['uji']['n_runs']}、Leukemia{quality['leukemia']['n_converged']}/{quality['leukemia']['n_runs']}。合成对参考目标的最大归一化差为{quality['synthetic']['max_normalized_objective_difference_to_reference']:.2e}；全部证书、原始gap、KKT、失败状态和配置签名可追溯。"),
        h2('9.1　结论与解释范围'),
        p('相关性对求解和恢复的影响需要与真实非零数、支持布局、符号和惩罚共同分析。三个基本求解器主要改变达到给定目标精度的成本；统计恢复则还受目标偏差、噪声和设计几何限制。现代工作集/外推的性能优势属于整个实现，不能由本实验拆分为单一机制。'),
        p('限制包括：每格仅五个种子；除Leukemia外没有逐实例多次计时；基本/库算法的内部诊断频率不同；全局SNR随信号数与符号抵消变化；组指标有真组占比差异；固定三档惩罚不是统计最优调参。真实数据不提供真支持。本轮没有实现全路径、筛选索引、SLOPE、核心集或鲁棒污染目标；这些文献用于设计和定位，不构成其复现。')
    ])
    pages.append([
        h('参考文献'),['refs'],h2('代码获取与复现'),
        p('随附comp6704_research_bundle_v2.zip提供源码、配置、逐次日志、系数、图表、数据下载及文献核查。主要入口为scripts/run_v2.py；scripts/analyze_v2.py从结果重建图表，scripts/build_report_v2.py重建本文。配置、哈希、依赖、26项测试、代表性重跑和资源记录见results/v2/verification.json及README。原始Leukemia文件不重新分发，按官方链接下载。最终对阅卷人可访问的代码位置待作者定稿补入。'),
        p('姓名和学号按要求留空。报告正文先用中文，PDF不超过10页且小于20MB；课程截止为2026年11月29日23:59。本稿实现、运行和整理使用Codex辅助，仍需作者核验与提交。')
    ])
    return pages


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--name',default='');parser.add_argument('--student-id',default='');args=parser.parse_args()
    summary=read('results/v2/summary.json')
    if not summary['integrity']['complete_grid']:raise SystemExit('Cannot report a partial or inconsistent v2 grid')
    pages=make_pages(summary,{'name':args.name,'student_id':args.student_id})
    ordered=[]
    def strings(value):
        if isinstance(value,str):
            yield value
        elif isinstance(value,(list,tuple)):
            for item in value:
                yield from strings(item)
    for page in pages:
        for block in page:
            for text in strings(block):
                for key in re.findall(r'\[\[(.*?)\]\]',text):
                    if key not in ordered:ordered.append(key)
    entry_map={e['key']:e for e in bib_entries()};entries=[entry_map[k] for k in ordered]
    recent=[e for e in entries if e['year']>=2023]
    if len(recent)<8:raise RuntimeError('Recent literature coverage regressed')
    (ROOT/'report/content_zh_v2.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2))
    pdf=ROOT/'report/report_zh_v2.pdf';tex=ROOT/'report/main_v2.tex'
    render_pdf(pages,entries,pdf,table_captions=CAPTIONS);render_tex(pages,tex,table_captions=CAPTIONS)
    doc=pymupdf.open(pdf)
    if len(doc)>10 or pdf.stat().st_size>=20_000_000:raise RuntimeError(f'Layout exceeds course limit: {len(doc)} pages, {pdf.stat().st_size} bytes')
    validation={'protocol_version':2,'pdf_pages':len(doc),'pdf_bytes':pdf.stat().st_size,'identity_complete':bool(args.name.strip() and args.student_id.strip()),
                'identity_display':'blank unspecified fields as requested','references':len(entries),'recent_research_references_2023plus':len(recent),
                'reference_keys':ordered,'page_text_chars':[len(page.get_text()) for page in doc],
                'pdf_renderer':'ReportLab, mathematical figures from Matplotlib; previews exported for inspection',
                'tex_validation':'Editable multi-file source; compile in a full XeLaTeX/BibTeX environment'}
    (ROOT/'report/validation_v2.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    preview=ROOT/'report/preview/v2';preview.mkdir(parents=True,exist_ok=True)
    for i,page in enumerate(doc):page.get_pixmap(matrix=pymupdf.Matrix(1.2,1.2)).save(str(preview/f'page_{i+1:02}.png'))
    # Keep the previously delivered links pointed at the current draft.
    shutil.copyfile(pdf,ROOT/'report/report_zh.pdf');shutil.copyfile(tex,ROOT/'report/main.tex')
    shutil.copyfile(ROOT/'report/validation_v2.json',ROOT/'report/validation.json')
    print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
