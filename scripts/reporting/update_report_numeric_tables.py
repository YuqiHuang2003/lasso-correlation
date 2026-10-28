"""Build report tables from frozen results; does not fit models."""
from pathlib import Path
import json,csv,hashlib,statistics
ROOT=Path(__file__).resolve().parents[2]
def read(p): return json.loads((ROOT/p).read_text())
s=read('results/v2/summary.json');a=read('results/column_permutation_v1/analysis.json')
with (ROOT/'results/v2/synthetic_runs.csv').open() as f: runs=list(csv.DictReader(f))
lines=['## 14 论文数值表与完整列顺序明细','', '本节由保存的JSON/CSV生成，生成脚本为 `scripts/reporting/update_report_numeric_tables.py`。未进行新拟合。时间单位秒，F1原始分数取值0至1，F1极差另注明百分点。中位数后的方括号表示[第25百分位，第75百分位]，不是置信区间。事件比例是条件目标的描述比例；统计重复仍按种子块解释。','']
def text(t):lines.extend([t,''])
def table(head,rows):
 lines.append('| '+' | '.join(head)+' |');lines.append('| '+' | '.join(['---']*len(head))+' |')
 for row in rows:lines.append('| '+' | '.join(map(str,row))+' |')
 lines.append('')
def f(v):return '—' if v is None else f'{v:.6g}'
def agg(d):return '—' if not d or d.get('n',0)==0 else f"{f(d['median'])} [{f(d['q25'])}, {f(d['q75'])}]"
def stats(vals):
 v=sorted(vals);q=statistics.quantiles(v,n=4,method='inclusive') if len(v)>1 else [v[0]]*3
 return f'{f(statistics.median(v))} [{f(q[0])}, {f(q[2])}]'
text('### 14.1 合成v2算法总体耗时')
text('每算法360条运行，涵盖全部ρ、s、布局和λ条件，均达到gap≤1e-6。这里是全网格混合分布，便于说明总体成本；固定条件的比较见下一表。cold-start仅按原记录加回相关初始化，具体口径见第6节。来源：`results/v2/synthetic_runs.csv`。')
table(['算法','达标/运行','求解时间中位数 [Q25,Q75]','cold-start时间中位数 [Q25,Q75]','最大最终归一化gap'],[[m,f'{len(r)}/360',stats([float(x['elapsed_sec']) for x in r]),stats([float(x['cold_start_sec']) for x in r]),f(max(float(x['final_normalized_gap']) for x in r))] for m in ['ista','fista','cd','skglm'] for r in [[x for x in runs if x['method']==m]]])
text('### 14.2 固定条件的算法耗时及相关性配对比')
text('固定s=20、c=1；每行5个配对种子。表覆盖两布局和ρ端点，选择规则由研究变量确定。时间比为每种子ρ=0.9/ρ=0的比值再汇总，不是两个中位数相除。来源：`summary.json/by_cell`与`paired_rho_high_over_low`。')
rows=[]
for layout in ['dispersed','clustered']:
 for m in ['ista','fista','cd','skglm']:
  cells=[next(x for x in s['by_cell'] if x['layout']==layout and x['method']==m and x['sparsity']==20 and x['lambda_factor']==1 and x['rho']==rho) for rho in [0,0.9]]
  pair=next(x for x in s['paired_rho_high_over_low'] if x['layout']==layout and x['method']==m and x['sparsity']==20 and x['lambda_factor']==1)
  rows.append([layout,m,*[agg(x['elapsed_sec']) for x in cells],agg(pair['elapsed_sec_high_over_low'])])
table(['布局','算法','ρ=0时间','ρ=0.9时间','配对时间倍数'],rows)
text('### 14.3 合成恢复相图的数值底表')
text('固定基础CD、c=1，覆盖全部ρ×s×布局，共24个条件，每条件5种子。precision/recall/F1/系数误差为种子中位数；精确支持列为成功次数/5。四算法最终支持一致，故用CD表达支持指标，系数误差仍标明实现。来源：`summary.json/by_cell`。')
table(['布局','s','ρ','precision','recall','F1','精确支持','相对系数误差'],[[x['layout'],x['sparsity'],x['rho'],*[f(x[k]['median']) for k in ['precision','recall','f1']],f"{x['exact_support_successes']}/5",f(x['relative_coef_error']['median'])] for x in s['by_cell'] if x['method']=='cd' and x['lambda_factor']==1])
text('### 14.4 课程真实任务数值')
text('UJI表采用每划分验证集选出的λ，共3个划分。时间只汇总达标运行；MSE与非零数保留全部终止输出，包括超时。MSE单位为原经度响应单位的平方；此处不是后续模拟响应支持实验。来源：`summary.json/uji/aggregated`。')
table(['算法','达标/划分','达标时间','测试MSE','非零数'],[[x['method'],f"{x['n_converged']}/{x['n_runs']}",agg(x['elapsed_sec']),agg(x['test_mse']),agg(x['nnz'])] for x in s['uji']['aggregated']])
text('Leukemia原标签平方损失计时：每λ比例与算法3次固定目标计时重复。来源：`summary.json/leukemia/aggregated`。')
table(['λ/λmax','算法','达标/重复','达标时间'],[[x['lambda_ratio'],x['method'],f"{x['n_converged']}/{x['n_runs']}",agg(x['elapsed_sec'])] for x in s['leukemia']['aggregated']])
text('### 14.5 列顺序阈值敏感性完整汇总')
text('每目标6个主输出。各事件分别在对应τ下计算；F1极差严格大于列示百分点。所有行均完整且重复QC通过。最大支持差为同一目标内最大两两对称差再跨目标取最大。来源：`results/column_permutation_v1/analysis.json/conditions`。')
rows=[]
for tau in [1e-6,1e-4]:
 for source in ['uji','news','gene']:
  for m in ['sklearn','skglm']:
   r=[x for x in a['conditions'] if x['tau']==tau and x['source']==source and x['method']==m]
   assert len(r)==(40 if source=='uji' else 8) and all(x['n_eligible']==6 and x['repeat_qc_pass'] for x in r)
   rows.append([source,m,tau,len(r),sum(x['support_event_state']=='event' for x in r),*[sum(x['f1_event_states'][k]=='event' for x in r) for k in ['0.0','0.01','0.02','0.05']],f(100*max(x['observed_f1_range'] for x in r)),max(x['maximum_pairwise_support_difference'] for x in r)])
table(['来源','实现','τ','目标','支持事件','F1变化','>1百分点','>2百分点','>5百分点','最大极差/百分点','最大支持差'],rows)
text('### 14.6 UJI列顺序分层及种子覆盖')
text('主τ=1e-6。布局与幅度的分层属于同一批目标的描述；同一目标只在每套分层内出现一次。范围均值是每目标F1极差的算术平均。')
rows=[]
for m in ['sklearn','skglm']:
 for field in ['layout','amplitude','historical_reference_accepted']:
  for v in sorted(set(x[field] for x in a['conditions'] if x['source']=='uji'),key=str):
   r=[x for x in a['conditions'] if x['source']=='uji' and x['method']==m and x['tau']==1e-6 and x[field]==v]
   rows.append([m,field,v,len(r),sum(x['support_event_state']=='event' for x in r),sum(x['f1_event_states']['0.02']=='event' for x in r),f(100*statistics.mean(x['observed_f1_range'] for x in r))])
table(['实现','分层变量','值','目标','支持事件','F1>2百分点','平均极差/百分点'],rows)
table(['实现','τ','指标','事件块','无事件块','未决块','总块'],[[x['method'],x['tau'],x['metric'],x['event'],x['no_event'],x['undecidable'],x['n_seed_blocks']] for x in a['seed_summary'] if x['source']=='uji' and x['layer']=='qc' and x['metric'] in ['support','f1_gt_0.02']])
text('### 14.7 UJI全部40目标的列顺序结果')
text('主τ=1e-6。每个实现依次列出六次F1的最小值、最大值、极差（百分点）、最大两两支持差。此表保留所有零变化目标；F1最小/最大是六个已观测输出的范围。来源同上。')
rows=[]
for tid in sorted(x for x in a['target_ids'] if x.startswith('uji')):
 row=[tid]
 for m in ['sklearn','skglm']:
  x=next(x for x in a['conditions'] if x['target_id']==tid and x['method']==m and x['tau']==1e-6)
  row.extend([f(x['f1_min']),f(x['f1_max']),f(100*x['observed_f1_range']),x['maximum_pairwise_support_difference']])
 rows.append(row)
assert len(rows)==40
table(['目标','sklearn F1低','F1高','极差/百分点','支持差','skglm F1低','F1高','极差/百分点','支持差'],rows)
text('### 14.8 列顺序数值扰动幅度及组内外位置')
text('主τ，每个数值先在同一目标的六输出间取最大，再在来源/实现内取最大；不同列的最大值可能来自不同目标或输出对。坐标计数为每目标变化坐标并集的累加。')
table(['来源','实现','最大系数L∞差','最大拟合RMSE差','最大目标绝对差','组内变化坐标累计','组外累计'],[[source,m,*[f(max(x[k] for x in r)) for k in ['maximum_pairwise_coefficient_linf_difference','maximum_pairwise_fitted_rmse_difference','maximum_pairwise_primal_absolute_difference']],sum(x['changed_in_nontrivial_groups'] for x in r),sum(x['changed_outside_nontrivial_groups'] for x in r)] for source in ['uji','news','gene'] for m in ['sklearn','skglm'] for r in [[x for x in a['conditions'] if x['source']==source and x['method']==m and x['tau']==1e-6]]])
text('### 14.9 本节生成与取数记录')
text('表格保留约6位有效数字，百分点评分极差乘100后显示；阈值事件采用源文件的精确事件状态，不能用显示后四舍五入的分数重新决定事件。重新生成本节只读冻结结果并替换本节，不改变前文或任何实验产物。')
for name in ['results/v2/summary.json','results/v2/synthetic_runs.csv','results/column_permutation_v1/analysis.json']:
 text(f"`{name}` SHA256：`{hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}`。")
p=ROOT/'docs/REPORT_MASTER_BRIEF_ZH.md';original=p.read_text();marker='## 14 论文数值表与完整列顺序明细'
p.write_text(original.split(marker)[0].rstrip()+'\n\n'+'\n'.join(lines).rstrip()+'\n')
print(f'Updated {p}: {len(lines)} generated lines')
