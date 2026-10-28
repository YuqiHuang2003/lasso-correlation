#!/usr/bin/env python3
"""Prospective external-design audit: scores, coverage and representation loss.

Run after all timed reference stages. No optimizer is called. Grouped scores
change the evaluation object and are never presented as raw-variable gains.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault('MPLCONFIGDIR','/tmp/comp6704-reference-audit-mpl')
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'.modern_deps')]
from run_reference_audit_v1 import audit_signature,target_parameters,target_id
from lasso_study.audit_data_v1 import load_audit_source,make_audit_design,near_proportional_groups_chunked
from lasso_study.data import array_sha256


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def metrics(mask,truth):
    tp=int(np.sum(mask&truth));fp=int(np.sum(mask&~truth));fn=int(np.sum(~mask&truth))
    den=2*tp+fp+fn
    return dict(tp=tp,fp=fp,fn=fn,nnz=int(mask.sum()),support_error=fp+fn,
                f1=float(Fraction(2*tp,den)) if den else 0.)


def exact_f1(m):
    den=2*m['tp']+m['fp']+m['fn']
    return Fraction(2*m['tp'],den) if den else Fraction(0)


def pair_scores(a,b,truth,tau,eligible,deltas):
    ma=np.abs(a)>tau;mb=np.abs(b)>tau
    x=metrics(ma,truth);y=metrics(mb,truth);delta=abs(exact_f1(x)-exact_f1(y))
    return dict(sklearn=x,skglm=y,raw_absolute_f1_difference=float(delta),
                eligible_absolute_f1_difference=float(delta) if eligible else None,
                delta_fraction_numerator=delta.numerator,delta_fraction_denominator=delta.denominator,
                support_symmetric_difference=int(np.sum(ma!=mb)),
                sign_difference=int(np.sum(np.where(ma,np.sign(a),0)!=np.where(mb,np.sign(b),0))),
                delta_exceeded={str(d):bool(delta>Fraction(str(d))) if eligible else None for d in deltas})


def group_scores(groups,a,b,truth,tau,eligible,deltas):
    agg=lambda w:np.array([np.dot(w[g['columns_0based']],g['member_scales_from_anchor']) for g in groups])
    x,y,t=agg(a),agg(b),agg(truth)
    net=np.abs(t)>tau
    any_truth=np.array([bool(np.any(truth[g['columns_0based']]!=0)) for g in groups])
    sizes=np.array([len(g['columns_0based']) for g in groups])
    counts=np.array([int(np.sum(truth[g['columns_0based']]!=0)) for g in groups])
    gross=np.array([float(np.abs(truth[g['columns_0based']]*g['member_scales_from_anchor']).sum()) for g in groups])
    result=pair_scores(x,y,net,tau,eligible,deltas)
    result.update(evaluation_object='group_net_effect',observed_absolute_f1_difference=result['raw_absolute_f1_difference'],
        group_count=len(groups),singleton_groups=int(np.sum(sizes==1)),nontrivial_groups=int(np.sum(sizes>1)),
        columns_in_nontrivial_groups=int(sizes[sizes>1].sum()),lost_coordinate_identities=int(len(truth)-len(groups)),
        maximum_group_size=int(sizes.max()),
        true_variables_in_nontrivial_groups=int(counts[sizes>1].sum()),groups_with_multiple_true_variables=int(np.sum(counts>1)),
        full_cancellation_groups=int(np.sum(any_truth&~net)),
        partial_or_full_cancellation_groups=int(np.sum(gross>np.abs(t)+1e-12)),
        contains_any_true_groups=int(any_truth.sum()),effective_true_groups=int(net.sum()),
        contains_any_scores={'sklearn':metrics(np.abs(x)>tau,any_truth),'skglm':metrics(np.abs(y)>tau,any_truth)},
        aggregated_coefficient_linf_difference=float(np.max(np.abs(x-y))),
        aggregation_changes_representation=bool(np.any(sizes>1)))
    return result


def csv_rows(path,rows):
    if not rows:return
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def flat_score(row):
    out={k:v for k,v in row.items() if not isinstance(v,dict)}
    for method in ('sklearn','skglm'):
        out.update({method+'_'+k:v for k,v in row[method].items()})
    out.update({'delta_gt_'+str(k):v for k,v in row['delta_exceeded'].items()})
    return out


def wilson(k,n):
    z=1.959963984540054;den=1+z*z/n;p=k/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-half),min(1.,center+half)]


def seed_events(rows,deltas):
    grouped=defaultdict(list)
    for row in rows:grouped[(row['source'],row['seed'],row['tau'])].append(row)
    out=[]
    for (source,seed,tau),values in sorted(grouped.items()):
        if len(values)!=4:raise RuntimeError('Seed must contain four registered amp/c conditions')
        for delta in deltas:
            eligible=[r for r in values if r['accuracy_eligible']]
            positive=any(r['delta_exceeded'][str(delta)] for r in eligible)
            state='event' if positive else 'no_event' if len(eligible)==4 else 'undecidable'
            out.append(dict(source=source,seed=seed,tau=tau,delta=delta,state=state,n_targets=4,n_eligible=len(eligible)))
    summary=[]
    for source,tau,delta in sorted({(r['source'],r['tau'],r['delta']) for r in out}):
        a=[r for r in out if (r['source'],r['tau'],r['delta'])==(source,tau,delta)]
        counts=Counter(r['state'] for r in a);n=len(a)
        summary.append(dict(source=source,tau=tau,delta=delta,n_seed_blocks=n,n_event=counts['event'],
            n_no_event=counts['no_event'],n_undecidable=counts['undecidable'],
            observed_identification_range=[counts['event']/n,(counts['event']+counts['undecidable'])/n],
            range_is_confidence_interval=False,
            seed_event_wilson95_if_fully_observed=wilson(counts['event'],n) if not counts['undecidable'] else None))
    return out,summary


def oracle_preferences(rows,tolerance):
    groups=defaultdict(list)
    for r in rows:groups[(r['source'],r['seed'],r['amplitude'],r['tau'])].append(r)
    output=[]
    for key,rs in sorted(groups.items()):
        if len(rs)!=2 or {r['c'] for r in rs}!={.5,1.}:raise RuntimeError('Oracle c comparison needs both registered levels')
        row=dict(zip(('source','seed','amplitude','tau'),key))
        if not all(r['accuracy_eligible'] for r in rs):
            row.update(state='undecidable',sklearn_preferred_c=None,skglm_preferred_c=None,has_any_tie=None,both_tied=None)
        else:
            sets={}
            for method in ('sklearn','skglm'):
                best=max(r[method]['f1'] for r in rs)
                sets[method]=sorted(r['c'] for r in rs if abs(r[method]['f1']-best)<=tolerance)
            state=('same_set' if sets['sklearn']==sets['skglm'] else
                   'strict_singleton_reversal' if len(sets['sklearn'])==len(sets['skglm'])==1 else 'tie_involved_difference')
            row.update(state=state,sklearn_preferred_c=sets['sklearn'],skglm_preferred_c=sets['skglm'],
                       has_any_tie=any(len(x)>1 for x in sets.values()),both_tied=all(len(x)>1 for x in sets.values()))
        output.append(row)
    return output


def plot(summary,stage):
    out=ROOT/'figures/reference_audit_v1'/stage;out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,source in zip(axes,('news','gene')):
        r=[x for x in summary['raw_scores'] if x['source']==source and x['tau']==1e-6]
        labels=['Planned','Accuracy eligible','Full consensus','F1 delta > 0.02']
        counts=[len(r),sum(x['accuracy_eligible'] for x in r),sum(x['reference_accepted'] for x in r),
                sum(x['delta_exceeded']['0.02'] is True for x in r)]
        bars=ax.bar(range(4),counts,color=['#aab7c4','#356ba0','#4b9879','#bd6945'])
        ax.bar_label(bars);ax.set_xticks(range(4),labels,rotation=18,ha='right',fontsize=8)
        ax.set_ylim(0,max(counts)*1.18);ax.set_title(source);ax.set_ylabel('Registered targets (paired conditions)')
    fig.suptitle('External design audit: qualification and raw-variable score consequences')
    for suffix in ('png','pdf'):fig.savefig(out/f'qualification.{suffix}',dpi=180)
    plt.close(fig)


def analyze(stage,skip_groups=False):
    started=perf_counter();base=ROOT/'results/reference_audit_v1'/stage
    vp=base/'verification.json';v=read(vp)
    if not v.get('all_passed'):raise RuntimeError('Complete numerical verification is required')
    if v.get('verifier_script_sha256')!=sha(ROOT/'scripts/verify_reference_audit_v1.py'):
        raise RuntimeError('Verifier changed after the saved verification')
    for relative,value in v['input_provenance'].items():
        if sha(ROOT/relative)!=value:raise RuntimeError('Verified input changed: '+relative)
    cp=ROOT/'configs/reference_audit_v1.json';config=read(cp);sig,_=audit_signature(cp)
    if sig!=v['numerical_signature']:raise RuntimeError('Current source differs from verified numerical source')
    params=target_parameters(config,stage);expected={target_id(x):x for x in params}
    if {p.stem for p in (base/'references').glob('*.json')}!=set(expected):raise RuntimeError('Incomplete reference grid')
    deltas=list(dict.fromkeys([config['primary_delta']]+config['delta_sensitivity']))
    provenance={str(vp.relative_to(ROOT)):sha(vp)};raw=[];grouped=[];comparisons=[];source_cache={};x_cache={};group_cache={}
    def track(p):provenance[str(p.relative_to(ROOT))]=sha(p)
    for identifier,param in expected.items():
        rp=base/'references'/f'{identifier}.json';ep=base/'evaluation'/f'{identifier}.json'
        npz=base/'reference_coefficients'/f'{identifier}.npz';dp=base/'datasets'/f'{identifier}.json'
        refs,e,ds=read(rp),read(ep),read(dp)
        for p in (rp,ep,npz,dp):track(p)
        with np.load(npz,allow_pickle=False) as arrays:
            a,b,truth=(arrays[k].copy() for k in ('ref_sklearn','ref_skglm','w_true'))
        eligible=e['accuracy_eligible'];accepted=refs['reference_accepted']
        meta=dict(param,target_id=identifier,n=ds['n'],p=ds['p'],accuracy_eligible=eligible,reference_accepted=accepted,
                  snr_population=ds.get('snr_population'),lambda_value=ds['lambda'])
        for tau in config['support_thresholds']:
            scores=pair_scores(a,b,truth!=0,tau,eligible,deltas)
            saved=e['scores_by_tau'][str(tau)]
            if not np.isclose(scores['raw_absolute_f1_difference'],saved['raw_absolute_f1_difference'],atol=1e-15,rtol=0):
                raise RuntimeError('Independent score recomputation disagrees')
            if scores['delta_exceeded']!=saved['delta_exceeded']:raise RuntimeError('Exact delta event disagreement')
            raw.append(dict(meta,tau=tau,evaluation_object='raw_variables',**scores))
        source=param['source'];xkey=(source,param['seed'])
        if xkey not in x_cache:
            if source not in source_cache:source_cache[source]=load_audit_source(source)
            data=make_audit_design(config,**param,source_data=source_cache[source])
            if data['metadata']['hashes']['X']!=ds['hashes']['X']:raise RuntimeError('Grouping design hash differs')
            x_cache[xkey]=data['X']
        X=x_cache[xkey];xhash=array_sha256(X)
        if xhash!=ds['hashes']['X']:raise RuntimeError('Paired grouping design differs from target X')
        difference=a-b
        fitted_difference=X@difference
        primal_a=refs['sklearn']['certificate']['primal'];primal_b=refs['skglm']['certificate']['primal']
        comparisons.append(dict(meta,coefficient_linf_difference=float(np.max(np.abs(difference))),
            coefficient_l2_difference=float(np.linalg.norm(difference)),
            fitted_difference_rmse=float(np.sqrt(np.mean(fitted_difference**2))),
            fitted_difference_linf=float(np.max(np.abs(fitted_difference))),
            l1_norm_absolute_difference=float(abs(np.abs(a).sum()-np.abs(b).sum())),
            primal_absolute_difference=abs(primal_a-primal_b),
            primal_normalized_difference=abs(primal_a-primal_b)/max(1.,abs(primal_a),abs(primal_b)),
            sklearn_normalized_gap=refs['sklearn']['certificate']['normalized_gap'],
            skglm_normalized_gap=refs['skglm']['certificate']['normalized_gap'],
            qualification_failure_reasons=e['qualification_failure_reasons']))
        if skip_groups:continue
        if xhash not in group_cache:
            group_cache[xhash]=near_proportional_groups_chunked(X)
            gp=base/'grouping'/f'{xhash}.json';dump(gp,group_cache[xhash]);track(gp)
        scan=group_cache[xhash]
        for tolerance,groups in scan['groups_by_tolerance'].items():
            if sorted(j for g in groups for j in g['columns_0based'])!=list(range(len(truth))):raise RuntimeError('Invalid grouping partition')
            for tau in config['support_thresholds']:
                grouped.append(dict(meta,tau=tau,grouping_tolerance=float(tolerance),
                                    **group_scores(groups,a,b,truth,tau,eligible,deltas)))
    events,seed_summary=seed_events(raw,deltas);oracle=oracle_preferences(raw,config['oracle_c_tie_tolerance'])
    source_summary=[]
    for source in config['sources']:
        a=[r for r in raw if r['source']==source and r['tau']==config['support_threshold']]
        good=[r for r in a if r['accuracy_eligible']]
        source_summary.append(dict(source=source,targets=len(a),independent_seed_blocks=len({r['seed'] for r in a}),
            accuracy_eligible_targets=len(good),full_consensus_targets=sum(r['reference_accepted'] for r in a),
            f1_different_targets=sum(r['delta_exceeded']['0.0'] is True for r in good),
            primary_event_targets=sum(r['delta_exceeded'][str(config['primary_delta'])] is True for r in good),
            max_observed_eligible_f1_difference=max((r['eligible_absolute_f1_difference'] for r in good),default=None),
            n_values=sorted({r['n'] for r in a}),p_values=sorted({r['p'] for r in a})))
    track(cp)
    result=dict(stage=stage,integrity_verified=True,numerical_signature=sig,config_sha256=sha(cp),
        analysis_script_sha256=sha(Path(__file__)),primary_tau=config['support_threshold'],primary_delta=config['primary_delta'],
        raw_scores=raw,group_scores=grouped,reference_comparisons=comparisons,
        source_summary=source_summary,seed_events=events,seed_summary=seed_summary,
        oracle_c_preferences=oracle,oracle_preference_counts=dict(Counter(r['state'] for r in oracle)),
        oracle_any_tie_count=sum(r['has_any_tie'] is True for r in oracle),
        oracle_both_tied_count=sum(r['both_tied'] is True for r in oracle),
        grouping_diagnostics={key:value['diagnostics'] for key,value in group_cache.items()},
        all_group_evaluations_completed=not skip_groups,input_provenance=provenance,analysis_wall_sec=perf_counter()-started,
        interpretation=['Paired conditions are not independent datasets. Gene uses one fixed X.',
            'Primary score contrasts include all accuracy-eligible targets, regardless of full consensus.',
            'Seed-level identification ranges reflect unresolved conditions, not confidence intervals.',
            'Oracle c preferences use known synthetic truth and are not operational parameter selection.',
            'Group net-effect and contains-any truths are distinct; grouped scores change the evaluation object.',
            'Observed two-solution score ranges are not global bounds on a solution set.'])
    dump(base/'analysis.json',result)
    csv_rows(base/'raw_scores.csv',[flat_score(r) for r in raw])
    csv_rows(base/'reference_comparisons.csv',comparisons)
    csv_rows(base/'seed_events.csv',events);csv_rows(base/'oracle_c_preferences.csv',oracle)
    csv_rows(base/'group_scores.csv',[flat_score(r) for r in grouped])
    plot(result,stage)
    print(json.dumps({k:result[k] for k in ('stage','source_summary','oracle_preference_counts','grouping_diagnostics')},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('pilot','confirm'))
    parser.add_argument('--skip-groups',action='store_true',help='Development only; no final group-scheme claims')
    args=parser.parse_args()
    with threadpool_limits(limits=1):analyze(args.stage,args.skip_groups)


if __name__=='__main__':main()
