#!/usr/bin/env python3
"""Read-only Chinese brief for the verified column-order mechanism study."""
from pathlib import Path
import hashlib
import json
import os
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/comp6704-column-brief-mpl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base=ROOT/'results/column_permutation_v1';config=read(ROOT/'configs/column_permutation_v1.json')
    for name in ('verification.json','analysis_verification.json'):
        verification=read(base/name)
        if not verification['all_passed'] or verification['verifier_script_sha256']!=sha(ROOT/'scripts/verify_column_permutation_v1.py'):
            raise RuntimeError('Current independent verification is required: '+name)
        for relative,digest in verification.get('input_provenance',{}).items():
            if sha(ROOT/relative)!=digest:raise RuntimeError('Changed verified input: '+relative)
    analysis=read(base/'analysis.json');completion=read(base/'completion.json')
    if analysis['analysis_script_sha256']!=sha(ROOT/'scripts/analyze_column_permutation_v1.py'):
        raise RuntimeError('Analysis source changed')
    from build_report import render_pdf,pymupdf
    import matplotlib.pyplot as plt
    import numpy as np
    p=lambda value:['p',value]
    h=lambda value:['h',value]
    table=lambda head,rows:['table',head,rows]
    summary=analysis['source_summary'];primary=analysis['primary_tau']
    source_rows=[[r['source'],r['method'],str(r['targets']),str(r['qc_support_states'].get('event',0)),
                  str(r['qc_primary_f1_states'].get('event',0)),f"{100*r['maximum_observed_f1_range']:.2f}"] for r in summary]
    block_rows=[]
    for method in config['methods']:
        selected=[r for r in analysis['seed_summary'] if r['source']=='uji' and r['method']==method
                  and r['tau']==primary and r['layer']=='qc']
        support=next(r for r in selected if r['metric']=='support')
        f1=next(r for r in selected if r['metric']=='f1_gt_0.02')
        block_rows.append([method,f"{support['event']}/10",f"{f1['event']}/10",str(support['undecidable'])])
    cases=[max((r for r in analysis['conditions'] if r['method']==method and r['tau']==primary),
               key=lambda r:r['observed_f1_range'] or 0) for method in config['methods']]
    case=max(cases,key=lambda r:r['observed_f1_range'] or 0)
    case_scores=[r for r in analysis['output_scores'] if r['target_id']==case['target_id'] and r['method']==case['method']
                 and r['tau']==primary and r['is_primary']]
    case_scores.sort(key=lambda r:config['main_order_ids'].index(r['order_id']))
    case_rows=[[r['order_id'],r['tp'],r['fp'],r['fn'],r['nnz'],f"{r['f1']:.6f}"] for r in case_scores]
    dataset=read(base/'datasets'/(case['target_id']+'.json'))
    names=[dataset['feature_names'][j] for j in case['changed_coordinate_union']]
    max_range=max(r['maximum_observed_f1_range'] for r in summary)
    wall=completion['previous_recorded_wall_sec']+completion['wall_sec_this_invocation']
    smallfig=ROOT/'figures/column_permutation_v1/source_overview.png'
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    for ax,metric,title in zip(axes,('qc_support_states','qc_primary_f1_states'),('Support changes','F1 range > 0.02')):
        values=[r[metric].get('event',0) for r in summary]
        bars=ax.bar(np.arange(len(summary)),values,color=['#376c98','#b66f47']*3)
        ax.bar_label(bars,labels=[f"{v}/{r['targets']}" for v,r in zip(values,summary)],fontsize=8)
        ax.set_xticks(range(len(summary)),[r['source']+'\n'+r['method'] for r in summary],fontsize=7)
        ax.set_ylim(0,max([1]+values)*1.2);ax.set_title(title);ax.set_ylabel('Targets; paired conditions')
    fig.savefig(smallfig,dpi=190);fig.savefig(smallfig.with_suffix('.pdf'));plt.close(fig)
    references=[
        {'key':'kim2025','text':'Kim, D.; Lee, J.; Yun, C. Provable Benefit of Random Permutations over Uniform Sampling in Stochastic Coordinate Descent. ICML, PMLR267:30480–30547, 2025.','url':'https://proceedings.mlr.press/v267/kim25x.html'},
        {'key':'trust2023','text':'Bastounis, A.; Cucker, F.; Hansen, A. C. When can you trust feature selection?—I: A condition-based analysis of LASSO and generalised hardness of approximation. arXiv:2312.11425, 2023（所读v1预印本）.','url':'https://arxiv.org/abs/2312.11425'},
        {'key':'bu2023','text':'Bu, H.; Nagata, T.; Onodera, K.; Arima, S. Sparse Reluctant Interaction Modelling with Convex Hull Approximation—For Accurate Modelling of Highly Correlated Variables Based on Diverse Nearly Optimal Solutions. JIMA73(4E):223–233, 2023.','url':'https://doi.org/10.11221/jima.73.223'},
        {'key':'skglm2025','text':'Moufad, B. et al. skglm: Improving scikit-learn for Regularized Generalized Linear Models. JMLR26(149):1–6, 2025.','url':'https://www.jmlr.org/papers/v26/24-0008.html'},
        {'key':'knockpy','text':'knockpy 1.3.4. 作者官方fit_lasso实现与API：拟合前排列输入列并恢复系数身份。访问日期2026-10-28；软件实现来源。','url':'https://amspector100.github.io/knockpy/_modules/knockpy/knockoff_stats.html#fit_lasso'},
        {'key':'nouraie2026','text':'Nouraie等. Stability selection via variable decorrelation. Statistics and Computing36:160, 2026.','url':'https://doi.org/10.1007/s11222-026-10916-7'},
    ]
    pages=[[
        ['title','同一Lasso目标下的列顺序与稀疏恢复评分'],
        p('列顺序机制实验v1 · 中文研究简报\n姓名：________________　学号：________________\n实验完成：2026年10月28日（Asia/Shanghai）'),
        h('本轮结果'),
        p(f'保持X/y/λ和原变量身份不变，仅改变最终特征矩阵的列输入顺序，两个实现均在UJI部分目标出现支持和真支持F1变化；最大观测F1极差约{100*max_range:.2f}个百分点。全部784个输出通过独立精度检查，112次原顺序重复均一致。news/gene对照未观察到变化。本轮补上了此前删列/聚合改变评价对象的证据缺口，但结果仍限于所测设计及有限排列。'),
        h('1　实验问题与冻结范围'),
        p('研究主题仍为“特征相关性对Lasso求解效率与稀疏恢复的影响”。本阶段检验同一实现是否对输入列顺序敏感，不比较不同统计估计器，不改响应、惩罚、行、标准化或真实系数。输出逆映回原变量坐标，证书与评分均在原坐标重算。'),
        table(['来源','目标/不同X','固定设计与配对'],[
            ['UJI','40 / 10','全部旧seed300–309，两布局×两幅度；n1000、p360–380'],
            ['news','8 / 2','旧confirm seed401/402，两幅度×两c；1000×58'],
            ['gene','8 / 1','旧confirm seed501/502，共用72×7129 X；两幅度×两c']]),
        p('每份X采用原序加5个固定随机排列（seed700–704）。每target/method另重复原序一次，只作QC，不进入六主顺序的范围。总672主调用+112重复QC=784，实际库拟合864次；21项新阶段小测试分别通过。目标、实际排列、原坐标组和清单在首次拟合前锁定。'),
        p('sklearn循环CD与skglm固定版本、float64、单线程、每次从零。单输出独立按归一化gap≤1e-10、可行性及20秒软预算准入，不按跨实现共识预筛。主τ=1e-6，敏感性τ=1e-4；F1极差严格>.02使用整数TP/FP/FN的精确有理数比较。')
    ],[
        h('2　原变量支持与评分的顺序敏感性'),
        table(['来源','实现','目标','支持变化','F1极差>.02','最大极差/百分点'],source_rows),
        ['image','column_permutation_v1/source_overview','图1　QC通过后的事件目标数。40或8个目标是配对条件，不能作为同等数量的独立设计。',175],
        table(['实现','UJI支持事件seed块','F1>.02事件seed块','未决块'],block_rows),
        p('每seed块含4个条件，任一事件使块为事件，4个全部无事件才为无事件，其余未决。两实现分别统计；不能把两个实现各一个seed拼成同一实现跨两个seed的复现。两实现均通过预定机制研究工作门槛，非显著性或发表门槛。'),
        p('原先完整双参考共识的10个UJI目标，两实现在本次六顺序下均无支持/F1变化；全部顺序事件位于原先未决的30个目标。这个分层与既有资格审计相符，未显示“原本已通过共识的目标被本次排列推翻”；也不能据有限排列保证任意顺序稳定。')
    ],[
        h('3　代表案例：变量身份不变，F1仍改变'),
        p(f"案例{case['target_id']}，实现{case['method']}，主τ。该例按最大观测F1范围展示，为事后代表案例；发生频数以第2页完整分母为准。"),
        table(['顺序','TP','FP','FN','非零数','F1'],case_rows),
        p(f"六输出F1从{case['f1_min']:.6f}到{case['f1_max']:.6f}，极差为{case['range_fraction_numerator']}/{case['range_fraction_denominator']}≈{100*case['observed_f1_range']:.2f}个百分点。六输出间最大系数L∞差约{case['maximum_pairwise_coefficient_linf_difference']:.3f}，最大拟合RMSE差约{case['maximum_pairwise_fitted_rmse_difference']:.2e}，最大目标绝对差约{case['maximum_pairwise_primal_absolute_difference']:.2e}；不同最大值不必来自同一输出对。"),
        p('发生支持变化的原变量包括：'+', '.join(names)+'。没有删列或把它们改成组标签后评分。旧真支持、y和λ保持不变，原序重复QC通过。'),
        h('4　变化的位置及边界'),
        table(['来源/实现','按目标计的组内变化坐标','组外变化坐标'],[
            [r['source']+'/'+r['method'],r['target_coordinate_union_inside_groups'],r['target_coordinate_union_outside_groups']]
            for r in analysis['group_change_summary']]),
        p('组仅由原X按预先固定的近比例规则确定，未在每个置换后重新选锚。表中是各目标内部变化坐标并集的计数之和，同一原始WAP在不同目标中可重复计数，不是独立变量样本数。所有本轮变化均落在近比例非单例组中，这是机制一致性证据，不证明其为所有可能顺序效应的唯一来源。'),
        p('相近拟合和目标不自动确定支持；本实验观测的是高精度浮点输出，不证明精确数学多解，也未枚举整个最优/近最优解集。')
    ],[
        h('5　新增证据与已有工作的关系'),
        p('坐标访问规则影响优化路径和速度已有直接研究。2025年的随机排列坐标下降理论[[kim2025]]每epoch重新洗牌，研究的目标与指标也不同；本实验是每次拟合固定一个输入排列，不检验其速率定理，也不提出新随机坐标方法。'),
        p('有限精度支持计算可信度[[trust2023]]、相关变量下多种近最优表示及TP/FP/F1[[bu2023]]、现代working-set求解框架[[skglm2025]]均有既有研究。本轮增加的是具体处理链中，在数学目标与原变量评价不变、重复QC及共同精度受控时，对同实现支持/F1变化幅度、覆盖及位置的量化，不以“顺序有影响”宣称创新。'),
        p('更直接的已有实现是knockpy：拟合前随机排列Lasso输入列并恢复身份，以处理对称性[[knockpy]]。2026年的变量去相关研究也展示顺序影响选择，但先进行随顺序改变的正交化[[nouraie2026]]，与本次保持原目标不同。这些先例均须承认，不能把输入置换及逆映射流程本身作为新方法。'),
        p('结论比此前跨实现差异或降维对照更直接，但没有改变外部未复现的边界：所选news/gene均稳定，旧共识通过的UJI目标也稳定。下一阶段若发展短稿，应围绕这类实际冗余设计的评价资格和处理成本，继续核实相对既有方法的增量；本轮不自动增加排列或启动另一实验网格。'),
        h('6　完整性、计算与复现'),
        p(f'784/784输出精度合格、112/112历史核对通过、112/112重复QC通过，无预算未运行或未知部分失败。准备与运行计算墙钟合计{wall:.3f}秒（阶段软预算7200秒），包含预热；独立核验另计，不含开发/人工审阅。该记录不用于求解器速度排名。'),
        p('入口：TODO.md；docs/column_permutation_protocol_v1.md；results/column_permutation_v1/。运行清单、实际排列、原/输入坐标系数、双坐标证书、原始及QC限定三态均保留。独立数值验收与分析验收分别为verification.json和analysis_verification.json，均未重新优化。全目标科学图另存figures/column_permutation_v1/support_f1_ranges.pdf。'),
        h('近期参考文献'),['refs']
    ]]
    out=ROOT/'report/column_permutation_v1';out.mkdir(parents=True,exist_ok=True)
    content={'pages':pages,'references':references}
    (out/'content_zh.json').write_text(json.dumps(content,ensure_ascii=False,indent=2))
    target=out/'column_order_zh.pdf'
    render_pdf(pages,references,target,table_captions=['固定目标与重复单位','原坐标事件及幅度（主τ）','同一实现的种子块复现','代表案例的六个主输出','固定原坐标组中的变化位置'])
    doc=pymupdf.open(target)
    if len(doc)!=4:raise RuntimeError('Unexpected report overflow')
    preview=out/'preview';preview.mkdir(exist_ok=True)
    for i,page in enumerate(doc):
        if not page.get_text().strip():raise RuntimeError('Empty PDF page')
        page.get_pixmap(matrix=pymupdf.Matrix(1.3,1.3)).save(preview/f'page_{i+1:02}.png')
    validation={'pages':len(doc),'bytes':target.stat().st_size,'pdf_sha256':sha(target),
        'builder_sha256':sha(Path(__file__)),'identity_complete':False,'optimizer_calls':0,
        'analysis_sha256':sha(base/'analysis.json'),'verification_sha256':sha(base/'verification.json'),
        'analysis_verification_sha256':sha(base/'analysis_verification.json'),'renderer':'ReportLab/Matplotlib; no TeX compilation'}
    (out/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    print(json.dumps(validation,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
