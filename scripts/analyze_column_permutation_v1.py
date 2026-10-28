#!/usr/bin/env python3
"""Analyze saved column-order outputs without optimizing any model."""
from __future__ import annotations

import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/comp6704-column-mpl')
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
from time import perf_counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.modern_deps')]


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def metrics(coef,truth,tau):
    selected=np.abs(coef)>tau;target=truth!=0
    tp=int(np.sum(selected&target));fp=int(np.sum(selected&~target));fn=int(np.sum(~selected&target))
    denominator=2*tp+fp+fn
    return dict(tp=tp,fp=fp,fn=fn,nnz=int(selected.sum()),f1=float(Fraction(2*tp,denominator)) if denominator else 0.)


def exact_f1(score):
    den=2*score['tp']+score['fp']+score['fn']
    return Fraction(2*score['tp'],den) if den else Fraction(0)


def event_state(n_eligible,n_expected,event):
    if event:return 'event'
    return 'no_event' if n_eligible==n_expected else 'undecidable'


def block_state(states,expected):
    if any(state=='event' for state in states):return 'event'
    return 'no_event' if len(states)==expected and all(state=='no_event' for state in states) else 'undecidable'


def qc_limited_state(raw,qc_pass,history_pass):
    return raw if qc_pass and history_pass else 'undecidable'


def score_range(scores,deltas,n_expected=6):
    values=[exact_f1(score) for score in scores]
    low=min(values) if values else None;high=max(values) if values else None
    span=high-low if values else None
    return {'n_eligible':len(values),'expected_main_outputs':n_expected,
            'f1_min':float(low) if low is not None else None,'f1_max':float(high) if high is not None else None,
            'observed_f1_range':float(span) if span is not None else None,
            'range_fraction_numerator':span.numerator if span is not None else None,
            'range_fraction_denominator':span.denominator if span is not None else None,
            'f1_event_states':{str(delta):event_state(len(values),n_expected,
                len(values)>=2 and span>Fraction(str(delta))) for delta in deltas}}


def continued_methods(blocks,source='uji',tau=1e-6,metric='support',layer='qc'):
    counts=Counter(r['method'] for r in blocks if r['source']==source and r['tau']==tau
                   and r['metric']==metric and r['layer']==layer and r['state']=='event')
    return sorted(method for method,n in counts.items() if n>=2)


def csv_rows(path,rows):
    if not rows:return
    keys=sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def make_blocks(events):
    grouped=defaultdict(list)
    fields=('source','seed','method','tau','metric','layer')
    for row in events:grouped[tuple(row[k] for k in fields)].append(row['state'])
    blocks=[dict(zip(fields,key),state=block_state(states,4),n_targets=len(states),
                 target_state_counts=dict(Counter(states))) for key,states in sorted(grouped.items())]
    union_grouped=defaultdict(list)
    union_fields=('source','seed','tau','metric','layer')
    for row in blocks:union_grouped[tuple(row[k] for k in union_fields)].append(row['state'])
    unions=[dict(zip(union_fields,key),state=block_state(states,2),n_methods=len(states))
            for key,states in sorted(union_grouped.items())]
    summary=[]
    fields2=('source','method','tau','metric','layer')
    for key in sorted({tuple(r[k] for k in fields2) for r in blocks}):
        values=[r['state'] for r in blocks if tuple(r[k] for k in fields2)==key]
        counts=Counter(values)
        summary.append(dict(zip(fields2,key),n_seed_blocks=len(values),event=counts['event'],
                            no_event=counts['no_event'],undecidable=counts['undecidable']))
    return blocks,unions,summary


def draw(result,base):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    targets=result['target_ids'];methods=result['methods']
    rows={(r['target_id'],r['method']):r for r in result['conditions'] if r['tau']==result['primary_tau']}
    support=np.full((len(targets),len(methods)),np.nan);f1=support.copy()
    for i,target in enumerate(targets):
        for j,method in enumerate(methods):
            row=rows[(target,method)]
            if row['n_eligible']==6:
                support[i,j]=row['maximum_pairwise_support_difference']
                f1[i,j]=row['observed_f1_range']
    fig,axes=plt.subplots(1,2,figsize=(10,18),constrained_layout=True)
    for ax,values,title in zip(axes,(support,f1),('Maximum support difference','Observed F1 range')):
        cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#c9cdd2')
        finite=values[np.isfinite(values)]
        plotted=ax.imshow(values,aspect='auto',cmap=cmap,vmin=0,vmax=max(float(finite.max()) if len(finite) else 0.,.001))
        ax.set_xticks(range(len(methods)),methods);ax.set_yticks(range(len(targets)),targets,fontsize=6.3)
        ax.set_title(title);fig.colorbar(plotted,ax=ax,shrink=.45)
        for i in range(len(targets)):
            for j in range(len(methods)):
                value=values[i,j];label='?' if np.isnan(value) else str(int(value)) if title.startswith('Maximum') else f'{value:.3f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=6.5,color='#172635')
    fig.suptitle('Column-order follow-up: same estimator and original variable coordinates\nGray: incomplete primary outputs; repeats excluded from ranges')
    out=ROOT/'figures/column_permutation_v1';out.mkdir(parents=True,exist_ok=True)
    for extension in ('png','pdf'):fig.savefig(out/('support_f1_ranges.'+extension),dpi=170)
    plt.close(fig)


def analyze():
    started=perf_counter();base=ROOT/'results/column_permutation_v1'
    verification=read(base/'verification.json')
    if not verification.get('all_passed'):raise RuntimeError('Independent numerical verification is required')
    if verification['verifier_script_sha256']!=sha(ROOT/'scripts/verify_column_permutation_v1.py'):
        raise RuntimeError('Independent verifier changed')
    provenance={str((base/'verification.json').relative_to(ROOT)):sha(base/'verification.json')}
    for relative,digest in verification.get('input_provenance',{}).items():
        if sha(ROOT/relative)!=digest:raise RuntimeError('Changed verified input: '+relative)
    config_path=ROOT/'configs/column_permutation_v1.json';config=read(config_path)
    manifest=read(base/'manifest.json')
    def track(path):provenance[str(path.relative_to(ROOT))]=sha(path)
    track(config_path);track(base/'manifest.json')
    from lasso_study.column_permutation_v1 import load_sources,make_target,target_id
    from threadpoolctl import threadpool_limits
    deltas=list(dict.fromkeys([config['primary_delta']]+config['delta_sensitivity']))
    conditions=[];events=[];pairwise=[];cross=[];qc_rows=[];all_output_scores=[]
    sources=load_sources(config)
    with threadpool_limits(limits=1):
        for params in manifest['targets']:
            identifier=target_id(params);data=make_target(config,params,sources)
            X,truth=data['X'],data['w_true'];dp=base/'datasets'/(identifier+'.json');ds=read(dp);track(dp)
            if ds['hashes']['X']!=data['metadata']['hashes']['X']:raise RuntimeError('Analysis replay differs')
            gp=base/'groups'/(ds['hashes']['X']+'.json');scan=read(gp);track(gp)
            groups=scan['groups_by_tolerance'][str(config['grouping_tolerance'])]
            in_group=np.zeros(X.shape[1],dtype=bool)
            for group in groups:
                if len(group['columns_0based'])>1:in_group[group['columns_0based']]=True
            outputs={}
            for method in config['methods']:
                by_order={}
                for order in config['execution_order_ids']:
                    rp=base/'runs'/(identifier+'__'+order+'__'+method+'.json')
                    if not rp.exists():continue
                    run=read(rp);track(rp);coef=None
                    if run.get('coefficient_path'):
                        ap=base/run['coefficient_path'];track(ap)
                        if sha(ap)!=run['coefficient_sha256']:raise RuntimeError('Changed coefficient artifact')
                        with np.load(ap,allow_pickle=False) as arrays:coef=arrays['coef_original'].copy()
                    by_order[order]={'run':run,'coef':coef,'eligible':run.get('accuracy_eligible') is True and coef is not None}
                outputs[method]=by_order
                identity=by_order.get('identity');repeat=by_order.get('identity_repeat')
                history_pass=bool(identity and identity['eligible'] and identity['run'].get('baseline_comparison',{}).get('passed'))
                repeat_state=repeat['run'].get('repeat_comparison',{}).get('status','undecidable') if repeat else 'undecidable'
                qc_pass=bool(identity and identity['eligible'] and repeat and repeat['eligible'] and repeat_state=='pass')
                qc_rows.append(dict(params,target_id=identifier,method=method,repeat_qc_state=repeat_state,
                                    repeat_qc_pass=qc_pass,history_pass=history_pass))
                for tau in config['support_thresholds']:
                    meta=dict(params,target_id=identifier,method=method,tau=tau,
                              historical_reference_accepted=data['origin']['historical_reference_accepted'])
                    good=[(order,by_order[order]) for order in config['main_order_ids'] if order in by_order and by_order[order]['eligible']]
                    scores=[metrics(value['coef'],truth,tau) for _,value in good]
                    for order,value in by_order.items():
                        if value['coef'] is not None:
                            all_output_scores.append(dict(meta,order_id=order,is_primary=order in config['main_order_ids'],
                                                          accuracy_eligible=value['eligible'],**metrics(value['coef'],truth,tau)))
                    row=dict(meta,**score_range(scores,deltas),repeat_qc_state=repeat_state,repeat_qc_pass=qc_pass,
                             history_pass=history_pass,n_nontrivial_group_columns=int(in_group.sum()))
                    masks={order:np.abs(value['coef'])>tau for order,value in good}
                    signs={order:np.where(masks[order],np.sign(value['coef']),0.) for order,value in good}
                    fits={order:X@value['coef'] for order,value in good}
                    changed=np.zeros(len(truth),dtype=bool);max_support=max_sign=0;max_coef=max_fit=max_objective=0.
                    baseline_differences={}
                    for (left,a),(right,b) in combinations(good,2):
                        difference=masks[left]!=masks[right];changed|=difference
                        n_support=int(difference.sum());n_sign=int(np.sum(signs[left]!=signs[right]))
                        coefficient=float(np.max(np.abs(a['coef']-b['coef'])))
                        fitted=float(np.sqrt(np.mean((fits[left]-fits[right])**2)))
                        objective=abs(a['run']['certificate']['primal']-b['run']['certificate']['primal'])
                        max_support=max(max_support,n_support);max_sign=max(max_sign,n_sign)
                        max_coef=max(max_coef,coefficient);max_fit=max(max_fit,fitted);max_objective=max(max_objective,objective)
                        pairwise.append(dict(meta,left_order=left,right_order=right,support_difference=n_support,
                            sign_difference=n_sign,coefficient_linf_difference=coefficient,fitted_difference_rmse=fitted,
                            primal_absolute_difference=objective,changed_in_nontrivial_groups=int(np.sum(difference&in_group)),
                            changed_outside_nontrivial_groups=int(np.sum(difference&~in_group))))
                        if left=='identity':baseline_differences[right]=n_support
                    support_state=event_state(len(good),6,max_support>0)
                    row.update(support_event_state=support_state,qc_support_event_state=qc_limited_state(support_state,qc_pass,history_pass),
                        qc_f1_event_states={key:qc_limited_state(state,qc_pass,history_pass) for key,state in row['f1_event_states'].items()},
                        maximum_pairwise_support_difference=max_support,maximum_pairwise_sign_difference=max_sign,
                        maximum_pairwise_coefficient_linf_difference=max_coef,maximum_pairwise_fitted_rmse_difference=max_fit,
                        maximum_pairwise_primal_absolute_difference=max_objective,
                        support_difference_from_identity=baseline_differences,changed_coordinate_union=np.flatnonzero(changed).tolist(),
                        changed_in_nontrivial_groups=int(np.sum(changed&in_group)),changed_outside_nontrivial_groups=int(np.sum(changed&~in_group)))
                    conditions.append(row)
                    for metric,raw in [('support',support_state)]+[('f1_gt_'+str(delta),row['f1_event_states'][str(delta)]) for delta in deltas]:
                        for layer,state in [('raw',raw),('qc',qc_limited_state(raw,qc_pass,history_pass))]:
                            events.append(dict(meta,metric=metric,layer=layer,state=state))
            for order in config['main_order_ids']:
                a=outputs['sklearn'].get(order);b=outputs['skglm'].get(order)
                if a and b and a['eligible'] and b['eligible']:
                    for tau in config['support_thresholds']:
                        ma=np.abs(a['coef'])>tau;mb=np.abs(b['coef'])>tau
                        cross.append(dict(params,target_id=identifier,order_id=order,tau=tau,
                            support_difference=int(np.sum(ma!=mb)),coefficient_linf_difference=float(np.max(np.abs(a['coef']-b['coef']))),
                            f1_difference=float(abs(exact_f1(metrics(a['coef'],truth,tau))-exact_f1(metrics(b['coef'],truth,tau))))))
    blocks,unions,block_summary=make_blocks(events)
    source_summary=[];strata=[]
    for source in config['sources']:
        for method in config['methods']:
            a=[r for r in conditions if r['source']==source and r['method']==method and r['tau']==config['support_threshold']]
            def summary(values):
                return dict(targets=len(values),complete_primary_targets=sum(r['n_eligible']==6 for r in values),
                    raw_support_states=dict(Counter(r['support_event_state'] for r in values)),
                    qc_support_states=dict(Counter(r['qc_support_event_state'] for r in values)),
                    raw_primary_f1_states=dict(Counter(r['f1_event_states'][str(config['primary_delta'])] for r in values)),
                    qc_primary_f1_states=dict(Counter(r['qc_f1_event_states'][str(config['primary_delta'])] for r in values)),
                    maximum_observed_f1_range=max((r['observed_f1_range'] for r in values if r['observed_f1_range'] is not None),default=None),
                    maximum_support_difference=max((r['maximum_pairwise_support_difference'] for r in values),default=None))
            source_summary.append(dict(source=source,method=method,**summary(a)))
            for old_acceptance in (True,False):
                values=[r for r in a if r['historical_reference_accepted']==old_acceptance]
                if values:strata.append(dict(source=source,method=method,historical_reference_accepted=old_acceptance,**summary(values)))
    groups_summary=[]
    for source in config['sources']:
        for method in config['methods']:
            a=[r for r in conditions if r['source']==source and r['method']==method and r['tau']==config['support_threshold']]
            groups_summary.append(dict(source=source,method=method,targets=len(a),
                target_coordinate_union_inside_groups=sum(r['changed_in_nontrivial_groups'] for r in a),
                target_coordinate_union_outside_groups=sum(r['changed_outside_nontrivial_groups'] for r in a),
                targets_with_outside_changes=sum(r['changed_outside_nontrivial_groups']>0 for r in a)))
    result=dict(numerical_signature=manifest['numerical_signature'],integrity_verified=True,
        analysis_script_sha256=sha(Path(__file__)),input_provenance=provenance,primary_tau=config['support_threshold'],
        primary_delta=config['primary_delta'],target_ids=[target_id(p) for p in manifest['targets']],methods=config['methods'],
        conditions=conditions,target_events=events,seed_blocks=blocks,method_union_seed_blocks=unions,
        seed_summary=block_summary,source_summary=source_summary,historical_consensus_strata=strata,
        group_change_summary=groups_summary,repeat_qc=qc_rows,output_scores=all_output_scores,
        pairwise_comparisons=pairwise,cross_method_comparisons=cross,
        support_work_trigger_methods=continued_methods(blocks),
        primary_f1_work_trigger_methods=continued_methods(blocks,metric='f1_gt_'+str(config['primary_delta'])),
        all_identity_history_passed=all(r['history_pass'] for r in qc_rows),
        all_identity_repeats_passed=all(r['repeat_qc_pass'] for r in qc_rows),
        analysis_wall_sec=perf_counter()-started,
        interpretation=['Main six orders and repeated-identity QC have separate denominators.',
            'All metrics compare original variable coordinates on the same frozen mathematical target.',
            'Raw observed events are retained even when repeat QC leaves attribution unresolved.',
            'Observed score ranges do not bound the full minimizer or near-minimizer set.',
            'Seeds, paired conditions, methods and orders are not independent observations.',
            'This mechanism follow-up neither proves mathematical nonuniqueness nor establishes novelty.'])
    dump(base/'analysis.json',result)
    flat=[]
    for row in conditions:
        value={k:v for k,v in row.items() if not isinstance(v,(dict,list))}
        value.update({'raw_f1_state_'+k:v for k,v in row['f1_event_states'].items()})
        value.update({'qc_f1_state_'+k:v for k,v in row['qc_f1_event_states'].items()});flat.append(value)
    for name,rows in [('conditions',flat),('target_events',events),('seed_blocks',blocks),('seed_summary',block_summary),
                      ('method_union_seed_blocks',unions),('output_scores',all_output_scores),('repeat_qc',qc_rows),
                      ('pairwise_comparisons',pairwise),('cross_method_comparisons',cross),
                      ('historical_consensus_strata',strata),('group_change_summary',groups_summary)]:
        csv_rows(base/(name+'.csv'),rows)
    draw(result,base)
    print(json.dumps({k:result[k] for k in ('source_summary','support_work_trigger_methods','primary_f1_work_trigger_methods',
                                          'all_identity_history_passed','all_identity_repeats_passed')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();analyze()
