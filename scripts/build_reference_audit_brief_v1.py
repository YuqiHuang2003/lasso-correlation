#!/usr/bin/env python3
"""Build a Chinese evidence update from verified saved results; no optimization."""
from pathlib import Path
import hashlib
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/comp6704-reference-brief-mpl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / 'results/reference_audit_v1'
    verification = read(base / 'analysis_verification.json')
    if not verification['all_passed']:
        raise RuntimeError('Independent analysis verification is required')
    if verification['verifier_script_sha256'] != sha(ROOT / 'scripts/verify_reference_analysis_v1.py'):
        raise RuntimeError('Independent verifier changed')
    for relative, digest in verification['input_provenance'].items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError('Changed independently verified input: ' + relative)
    analyses = {stage: read(base / stage / 'analysis.json') for stage in ('pilot', 'confirm')}
    for stage, analysis in analyses.items():
        if analysis['analysis_script_sha256'] != sha(ROOT / 'scripts/analyze_reference_audit_v1.py'):
            raise RuntimeError('Stale analysis: ' + stage)
        for relative, digest in analysis['input_provenance'].items():
            if sha(ROOT / relative) != digest:
                raise RuntimeError('Changed analysis input: ' + relative)
    control_base = ROOT / 'results/uji_preprocessing_control_v1'
    control_verification = read(control_base / 'verification.json')
    if not control_verification['all_passed']:
        raise RuntimeError('Independent control verification is required')
    if control_verification['verifier_script_sha256'] != sha(ROOT / 'scripts/verify_uji_preprocessing_control_v1.py'):
        raise RuntimeError('Independent control verifier changed')
    for relative, digest in control_verification.get('input_provenance', {}).items():
        if sha(ROOT / relative) != digest:
            raise RuntimeError('Changed control input: ' + relative)
    controls = [read(path) for path in (control_base / 'controls').glob('*.json')]
    if len(controls) != 48:
        raise RuntimeError('Incomplete post-hoc control grid')
    control_rows = []
    arm_labels = {'baseline':'原坐标基线', 'final_float64_exact':'最终X精确去重',
                  'A_raw_exact':'原始RSSI精确去重', 'B_near':'近比例留锚'}
    for arm, label in arm_labels.items():
        values = [r for r in controls if r['arm'] == arm]
        if len(values) != 12 or not all(r['accuracy_eligible'] for r in values):
            raise RuntimeError('Unexpected control coverage')
        differences = sum(r['reduced_coordinate_scores']['1e-06']['sklearn']['f1'] !=
                          r['reduced_coordinate_scores']['1e-06']['skglm']['f1'] for r in values)
        deleted = sorted({r['old_p']-r['new_p'] for r in values})
        control_rows.append([label, f'{sum(r["reference_accepted"] for r in values)}/12',
                             f'{differences}/12', f'{min(deleted)}–{max(deleted)}'])
    near_controls = [r for r in controls if r['arm'] == 'B_near']
    deleted_truth = sum(r['mapped_truth_support_diagnostics']['1e-06']['deleted_columns_with_true_nonzero_coefficient'] > 0
                        for r in near_controls)
    mapping_error = max(r['mapping_diagnostics']['truth']['signal_or_fit_mapping_rmse'] for r in near_controls)
    from build_report import render_pdf, pymupdf
    p = lambda value: ['p', value]
    h = lambda value: ['h', value]
    table = lambda head, rows: ['table', head, rows]
    confirm = analyses['confirm']
    sources = confirm['source_summary']
    rows = []
    for stage in ('pilot', 'confirm'):
        for source in analyses[stage]['source_summary']:
            rows.append([('先导' if stage == 'pilot' else '确认') + '/' + source['source'],
                         source['targets'], source['accuracy_eligible_targets'],
                         source['full_consensus_targets'], source['f1_different_targets']])
    comparison_rows = []
    for source in sources:
        values = [r for r in confirm['reference_comparisons'] if r['source'] == source['source']]
        comparison_rows.append([source['source']] + [f'{max(r[k] for r in values):.3e}' for k in
            ('coefficient_linf_difference', 'fitted_difference_rmse', 'primal_absolute_difference')])
    oracle = confirm['oracle_preference_counts']
    interval = next(r['seed_event_wilson95_if_fully_observed'] for r in confirm['seed_summary']
                    if r['tau'] == 1e-6 and r['delta'] == .02)
    references = [
        {'key': 'trust1', 'text': 'Bastounis, A.; Cucker, F.; Hansen, A. C. When can you trust feature selection?—I: A condition-based analysis of LASSO and generalised hardness of approximation. arXiv:2312.11425, 2023（预印本）.', 'url': 'https://arxiv.org/abs/2312.11425'},
        {'key': 'trust2', 'text': 'Bastounis, A.; Cucker, F.; Hansen, A. C. When can you trust feature selection?—II: On the effects of random data on condition in statistics and optimisation. arXiv:2312.11429, 2023（预印本）.', 'url': 'https://arxiv.org/abs/2312.11429'},
        {'key': 'bu', 'text': 'Bu, H.; Nagata, T.; Onodera, K.; Arima, S. Sparse Reluctant Interaction Modelling with Convex Hull Approximation—For Accurate Modelling of Highly Correlated Variables Based on Diverse Nearly Optimal Solutions. JIMA 73(4E):223–233, 2023.', 'url': 'https://doi.org/10.11221/jima.73.223'},
        {'key': 'polyclean', 'text': 'Jarret, A. et al. PolyCLEAN: Atomic optimization for super-resolution imaging and uncertainty estimation in radio interferometry. Astronomy & Astrophysics 693:A225, 2025.', 'url': 'https://doi.org/10.1051/0004-6361/202450970'},
        {'key': 'news', 'text': 'UCI Machine Learning Repository. Online News Popularity. 数据来源及许可记录见本项目source清单.', 'url': 'https://archive.ics.uci.edu/dataset/332/online+news+popularity'},
        {'key': 'gene', 'text': 'LIBSVM Data: Classification (Binary Class), Leukemia. 本轮使用72×7129设计，原分类标签不作模拟响应.', 'url': 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#leukemia'},
    ]
    pages = [[
        ['title', 'Lasso数值参考审计：\n外部确认结果与研究主张收窄'],
        p('COMP6704研究进展 · 中文证据简报\n姓名：________________　学号：________________\n2026年10月28日（Asia/Shanghai）'),
        h('本轮结论'),
        p('两个外部来源的72个确认目标均通过双参考精度与完整共识检查，两个支持阈值下均未出现支持、符号或真支持F1分歧。此前UJI评分依赖参考表示的现象，未通过本轮预定的跨来源继续条件。当前停止扩大外部网格，将主张收窄为特定行抽样与预处理链的案例审计；尚不能声称已形成可发表贡献。'),
        h('1　问题与冻结设计'),
        p('研究主题仍为“特征相关性对Lasso求解效率与稀疏恢复的影响”。本阶段专门检验：同一目标的高精度数值参考是否因系数表示不同而改变恢复评价；若不同，是否影响结论。代码、配置与协议在新外部拟合前保存哈希，先导与确认分开。'),
        table(['来源', '使用矩阵', '确认单位与限制'], [
            ['Online News[[news]]', '每种子1000×58', '9个种子；从固定旧训练池抽行，重新生成支持与响应'],
            ['Leukemia[[gene]]', '固定72×7129', '9个响应种子共用一个X，不能称为9个独立真实矩阵']]),
        p('每种子配对两个信号幅度与c=.5/1，共4个条件。真非零数5，σ=1，λ=cσ√(2log(p)/n)；不使用原标签作为本轮响应。先导8目标，确认72目标；两个来源此前用于课程其他任务，不称为从未见过的盲测数据。'),
        p('参考实现为scikit-learn循环CD和skglm。先各自检查归一化对偶间隙≤1e-10及可行性、预算，再查系数/目标/支持/符号共识。主τ=1e-6、敏感性τ=1e-4；主事件是绝对F1差严格大于.02，另查0/.01/.05。主F1比较包含所有精度合格目标，不预先按共识筛选。')
    ], [
        h('2　外部结果：没有复现评分依赖'),
        table(['阶段/来源', '目标数', '精度合格', '完整共识', 'F1不同'], rows),
        ['image', 'reference_audit_v1/confirm/qualification', '图1　确认目标的资格覆盖与主事件。每来源36目标来自9个种子的配对条件，不是36个独立设计。', 190],
        p(f'两个τ、全部δ均无事件。每来源均为0/9个确认seed事件、0个未决；Wilson95%区间约[0,{100*interval[1]:.1f}%]。该区间按种子计算，gene只解释为固定X及本响应机制条件下的事件率。负结果不证明总体不存在问题，亦不证明解唯一。'),
        table(['来源', '最大系数L∞差', '最大拟合差RMSE', '最大目标绝对差'], comparison_rows),
        p(f'离线oracle-c偏好比较（两来源×9种子×两幅度×两τ）共{sum(oracle.values())}条，全部为同一偏好集合，无平局或反转。这是已知真值下预定两档c的比较，不是CV、可部署调参或不同统计估计器排名。'),
        p('数值一致不等于统计恢复好：先导gene四个条件F1约.056–.095，两实现仍一致。所有目标的资格、失败原因、系数/拟合/目标差和原变量评分均保留。')
    ], [
        h('3　UJI案例：依赖抽样与去重位置'),
        p('旧UJI40个目标都通过双参考精度，只有10个通过完整共识；主τ下15/40的真支持F1随两个已观测参考改变，最大差约5.56个百分点。它是原变量评分差异，不应误称为相对生成真值的“参考误差”。'),
        table(['矩阵视图', '代表WAP220/424/466关系'], [
            ['官方完整19937行', '三列数值秩3，未表现为近比例'],
            ['固定训练池11962行', '三列数值秩3，未表现为近比例'],
            ['部分1000行子样本', '仅留下同一行的稀少检出，中心标准化后近比例'],
            ['另一些子样本', '可保留秩3，或部分/全部变成常量而被删除']]),
        p('矩阵审计复现旧10个X哈希，并检查另10个行探针；新探针没有生成响应或拟合，不能算新增求解确认。在seed305/308，原始RSSI子样本的220/466列精确相同，但旧实际X经训练池标准化、抽样及再次标准化后非逐位相同。seed300中不同检出幅度又会产生非精确重复的近比例关系。'),
        p('因此，普通去重是否覆盖该关系取决于表示和执行位置。不能概括为“常规去重无效”，也不能由删掉一对重复列推断所有评分分歧已经解决。浮点数值秩与近比例诊断均不是数学多解证明，不评价原始UJI或其他论文的完整基准有错。'),
        h('4　评价方案的收益与代价'),
        p('单参考提供某一实现返回的表示；双参考同时报告资格覆盖及未决目标；两个已观测评分的范围不等于完整解集的全局上下界。只分析共识目标会使旧UJI从40个缩至10个，必须公开这个覆盖损失。'),
        p('X-only近比例分组在旧UJI使聚合支持40/40一致，但其对象为组净效应∑aβ，原变量身份被合并。组中含任意真变量与组净效应不为零是两种不同真值。新外部在三个分组容差下全部为单例组，没有证明分组额外收益；不可把组评分当原变量恢复改善。'),
        table(['处理方式', '完整共识', '主τ的F1分歧', '每设计删列'], control_rows),
        p('机制对照固定旧seed300/305/308×两布局×两幅度。48次新公共参考调用完成24个A/B问题，全部精度合格；基线和最终X精确去重复用原问题。种子事后选择，不能用其分歧比例推断发生率。原始精确去重未消除分歧，近比例留锚在本小对照中得到完整共识。'),
        p(f'留锚保持旧y/λ，但更换坐标及参数化；无噪信号映射RMSE最大{mapping_error:.2e}，仍不等于严格目标等价。{deleted_truth}/12目标删除的原坐标含生成真信号；降维评分只对映射净真值计算，不与原变量F1相减宣称改善。')
    ], [
        h('5　文献定位与当前贡献边界'),
        p('2023年Bastounis等已用非活跃KKT裕量、活跃曲率及最小非零系数研究Lasso支持计算可信度，并联系随机设计、统计恢复与精度成本[[trust1]][[trust2]]。两个浮点实现达成共识不是其意义下的可信证书。'),
        p('Bu等2023已研究相关变量下的多种近最优解及变量选择[[bu]]；PolyCLEAN 2025以最优拟合/对偶结构描述支持不确定性[[polyclean]]。因此“目标接近不代表支持一致”“需要观察单个系数向量以外的输出”均不是本项目新意。本项目没有覆盖整个解集，也没有实现新可信支持算法。'),
        p('当前可核实的增量是具体处理链的案例、其评分后果、预处理位置和有界机制对照，附外部未复现结果和完整分母。它有复核与工程报告价值，但尚未确认足够新颖。现有材料可保留为技术报告或有限范围的经验短稿候选；不能靠新增图表、引用或更强标题补足贡献。'),
        h('6　复核入口'),
        p('完整规则：docs/reference_audit_protocol_v1.md；结果：results/reference_audit_v1/{pilot,confirm}/；独立验收：两阶段verification.json及analysis_verification.json；逐项执行状态：TODO.md。原课程第二版与workshop结果保留，不覆盖。'),
        p('本简报只读取已保存结果，不重新优化。分析脚本输出raw_scores、reference_comparisons、seed_events、oracle_c_preferences及group_scores表。环境、来源、代码和输入哈希随结果保存；gene分组分块扫描全部列对，不生成全浮点Gram。'),
        h('参考文献与数据'),
        ['refs']
    ]]
    out = ROOT / 'report/reference_audit_v1'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'content_zh.json').write_text(json.dumps({'pages': pages, 'references': references}, ensure_ascii=False, indent=2))
    target = out / 'external_audit_zh.pdf'
    render_pdf(pages, references, target, table_captions=['外部设计与统计单位', '完整资格覆盖', '确认阶段数值差异', 'UJI代表关系与抽样边界', '事后机制对照：不同表示的参考一致性'])
    document = pymupdf.open(target)
    if len(document) != len(pages):
        raise RuntimeError('Unexpected overflow: inspect PDF pagination')
    preview = out / 'preview'
    preview.mkdir(exist_ok=True)
    page_info = []
    for index, page in enumerate(document):
        if not page.get_text().strip():
            raise RuntimeError('Empty PDF page')
        page.get_pixmap(matrix=pymupdf.Matrix(1.3, 1.3)).save(preview / f'page_{index+1:02}.png')
        page_info.append({'page': index+1, 'text_characters': len(page.get_text())})
    validation = {'pages': len(document), 'bytes': target.stat().st_size, 'pdf_sha256': sha(target),
                  'builder_sha256': sha(Path(__file__)), 'identity_complete': False, 'page_info': page_info,
                  'analysis_sha256': {stage: sha(base/stage/'analysis.json') for stage in analyses},
                  'analysis_verification_sha256': sha(base/'analysis_verification.json'),
                  'control_verification_sha256': sha(control_base/'verification.json'),
                  'renderer': 'ReportLab/Matplotlib; no TeX compilation', 'optimization_calls': 0}
    (out / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2))
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
