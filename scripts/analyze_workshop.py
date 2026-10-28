#!/usr/bin/env python3
"""Audit observed support identities and recovery along independent trajectories.

This is descriptive analysis. A numerical reference consensus is not a proof
of exact support, and checkpoint times are not deployable stopping rules.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
os.environ.setdefault('MPLCONFIGDIR', '/tmp/comp6704-matplotlib-workshop')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10)
TAUS = (1e-6, 1e-4)
CELL = ('family', 'rho', 'layout', 'amplitude', 'method', 'tau', 'gap_threshold')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def stats(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(a):
        return dict(n=0, mean=None, median=None, q25=None, q75=None, min=None, max=None)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                q25=float(np.quantile(a, .25)), q75=float(np.quantile(a, .75)),
                min=float(a.min()), max=float(a.max()))


def wilson(successes, n):
    if not n:
        return dict(n=0, successes=0, rate=None, lower=None, upper=None)
    z=1.959963984540054; rate=successes/n; den=1+z*z/n
    center=(rate+z*z/(2*n))/den
    half=z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
    return dict(n=n,successes=int(successes),rate=rate,lower=max(0.,center-half),upper=min(1.,center+half))


def support_counts(selected, truth):
    tp = int(np.sum(selected & truth)); fp = int(np.sum(selected & ~truth))
    fn = int(np.sum(~selected & truth)); den = 2 * tp + fp + fn
    return dict(tp=tp, fp=fp, fn=fn, nnz=int(selected.sum()),
                f1=2 * tp / den if den else 0., support_error=fp + fn)


def errors(selected, ref, truth):
    result = support_counts(selected, truth)
    stat = support_counts(ref, truth)
    harmful = int(np.sum((ref == truth) & (selected != truth)))
    repair = int(np.sum((ref != truth) & (selected == truth)))
    numerical = int(np.sum(selected != ref))
    if result['support_error'] != stat['support_error'] + harmful - repair:
        raise AssertionError('Support error accounting identity failed')
    if numerical != harmful + repair:
        raise AssertionError('Binary support disagreement partition failed')
    return dict(result, reference_support_error=stat['support_error'],
                reference_f1=stat['f1'], numerical_support_error=numerical,
                numerical_harm=harmful, numerical_repair=repair,
                same_support_as_reference=numerical == 0,
                same_recovery_counts_as_reference=all(result[k] == stat[k] for k in ('tp', 'fp', 'fn')))


def sign_errors(coef, reference, truth, tau):
    current=np.where(np.abs(coef)>tau,np.sign(coef),0.)
    ref=np.where(np.abs(reference)>tau,np.sign(reference),0.)
    actual=np.sign(truth)
    return dict(numerical_sign_error=int(np.sum(current!=ref)),
                true_sign_error=int(np.sum(current!=actual)),
                reference_sign_error=int(np.sum(ref!=actual)))


def observed_times(coefs, history, reference, truth, tau):
    masks = np.abs(coefs) > tau; ref = np.abs(reference) > tau
    matches = np.all(masks == ref, axis=1)
    first = np.flatnonzero(matches)
    bad = np.flatnonzero(~matches)
    after = 0 if not len(bad) else int(bad[-1] + 1)
    last_time = float(history[-1]['time'])
    stable_observed_time = float(history[after]['time']) if after < len(history) else None
    left_after_first = bool(len(first) and np.any(~matches[first[0]:]))
    return dict(first_observed_reference_match_time=float(history[first[0]]['time']) if len(first) else None,
                time_after_last_observed_mismatch=stable_observed_time,
                observed_reference_match_was_later_lost=left_after_first,
                final_matches_reference=bool(matches[-1]),
                remaining_fraction_after_last_observed_mismatch=(last_time-stable_observed_time)/last_time
                if stable_observed_time is not None and last_time > 0 else None)


def write_csv(path, records):
    if not records:
        return
    fields = sorted(set().union(*(r.keys() for r in records)))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)


def plot(summary, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = summary['checkpoints']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, method, tau in zip(axes.ravel(), ('fista', 'cd', 'fista', 'cd'), (1e-6, 1e-6, 1e-4, 1e-4)):
        geometries=sorted({(r['family'],r['rho']) for r in rows},key=str)
        for geom_index, (family,rho) in enumerate(geometries):
            marker=('o','s','^','D')[geom_index%4]
            data = [r for r in rows if r['method'] == method and r['tau'] == tau and r['family'] == family and r['rho']==rho]
            if not data:
                continue
            for amplitude, ls in (('strong','-'), ('mixed_weak','--')):
                sub = [r for r in data if r['amplitude'] == amplitude]
                xs=[]; ys=[]
                planned=[r for r in summary['finals'] if r['method']==method and r['tau']==tau
                         and r['family']==family and r['rho']==rho and r['amplitude']==amplitude]
                for tol in THRESHOLDS:
                    good = [r for r in sub if r['gap_threshold'] == tol and r['reference_consensus']]
                    if planned:
                        xs.append(tol); ys.append(sum(r['same_support_as_reference'] for r in good)/len(planned))
                if xs:
                    geometry_label=family if rho is None else f'{family},rho={rho:g}'
                    ax.plot(xs, ys, marker=marker, linestyle=ls, label=geometry_label+'/'+amplitude)
        ax.set_xscale('log'); ax.invert_xaxis(); ax.set_ylim(-.03,1.03)
        ax.set_title(f'{method.upper()}, support threshold={tau:g}')
        ax.set_xlabel('First observed normalized gap threshold')
        ax.set_ylabel('Observed reference-match yield / all planned')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Reference-match yield: reached gap and numerical consensus; layouts pooled')
    for ext in ('png','pdf'):
        fig.savefig(output/f'support_adequacy.{ext}', dpi=180)
    plt.close(fig)

    fig, axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,method in zip(axes,('fista','cd')):
        sub=[r for r in rows if r['method']==method and r['tau']==1e-6 and r['reference_consensus']]
        means=[]
        for tol in THRESHOLDS:
            a=[r for r in sub if r['gap_threshold']==tol]
            means.append((np.mean([r['numerical_harm'] for r in a]) if a else np.nan,
                          np.mean([r['numerical_repair'] for r in a]) if a else np.nan))
        x=np.arange(len(THRESHOLDS)); vals=np.asarray(means)
        ax.bar(x-.18,vals[:,0],.36,label='Numerical harm')
        ax.bar(x+.18,vals[:,1],.36,label='Numerical repair')
        ax.set_xticks(x,[f'{t:g}' for t in THRESHOLDS]); ax.set_ylabel('Mean number of variables')
        ax.set_xlabel('First observed gap threshold'); ax.set_title(method.upper()); ax.legend()
    fig.suptitle('Numerical support errors: reached thresholds with reference consensus')
    for ext in ('png','pdf'):
        fig.savefig(output/f'harm_repair.{ext}',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for ax,method in zip(axes,('fista','cd')):
        geometries=sorted({(r['family'],r['rho']) for r in rows},key=str)
        for index,(family,rho) in enumerate(geometries):
            for amplitude,ls in (('strong','-'),('mixed_weak','--')):
                values=[]
                for tol in THRESHOLDS:
                    a=[r['gap_safe_reference_zero_coverage'] for r in rows
                       if r['method']==method and r['tau']==1e-6 and r['family']==family
                       and r['rho']==rho and r['amplitude']==amplitude and r['gap_threshold']==tol
                       and r['reference_consensus'] and r['gap_safe_reference_zero_coverage'] is not None]
                    values.append(float(np.median(a)) if a else np.nan)
                label=family if rho is None else f'{family},rho={rho:g}'
                ax.plot(THRESHOLDS,values,linestyle=ls,marker=('o','s','^')[index%3],label=label+'/'+amplitude)
        ax.set_xscale('log');ax.invert_xaxis();ax.set_ylim(-.03,1.03)
        ax.set_xlabel('First observed normalized gap threshold')
        ax.set_ylabel('Median excluded / numerical-reference zeros')
        ax.set_title(method.upper());ax.grid(alpha=.2);ax.legend(fontsize=6.5)
    fig.suptitle('Offline cumulative safe-zero coverage at reached thresholds')
    for ext in ('png','pdf'):
        fig.savefig(output/f'gap_safe_coverage.{ext}',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('smoke','pilot','confirm','real'))
    parser.add_argument('--allow-unverified',action='store_true',help='Development only; not for final reports')
    args=parser.parse_args(); base=ROOT/'results/workshop'/args.stage
    verification_path=base/'verification.json'
    verified=False
    if verification_path.exists():
        verification=load(verification_path)
        verified=bool(verification.get('all_passed') and verification.get('numerical_recomputed'))
        if verified:
            for relative, expected_hash in verification['input_provenance'].items():
                if digest(ROOT/relative)!=expected_hash:
                    raise RuntimeError(f'Verification inputs changed: {relative}')
    if not verified and not args.allow_unverified:
        raise SystemExit('Run scripts/verify_workshop.py for this stage before final analysis')
    completion=load(base/'completion.json')
    if completion['expected_targets']!=completion['recorded_targets'] or completion['expected_trajectories']!=completion['recorded_trajectories']:
        raise RuntimeError('Stage completion declares an incomplete grid')
    paths=sorted((base/'runs').glob('*.json'))
    if not paths:
        raise SystemExit(f'No saved runs: {base}')
    rows=[]; finals=[]; transitions=[]; provenance={}; statuses=defaultdict(int)
    for p in (base/'completion.json',base/'config.json',base/'environment.json',verification_path):
        if p.exists():
            provenance[str(p.relative_to(ROOT))]=digest(p)
    numerical_signatures=set(); config_hashes=set(); reference_by_target={}
    for path in paths:
        r=load(path); provenance[str(path.relative_to(ROOT))]=digest(path)
        run_id=r['run_id']; target=r['target_id']; history=r['history']
        cp=base/'trajectories'/f'{run_id}.npz'
        rp=base/'references'/f'{target}.json'
        refs=load(rp); reference_by_target[target]=refs
        if digest(cp)!=r['trajectory_sha256'] or digest(base/'reference_coefficients'/f'{target}.npz')!=refs['coefficient_sha256']:
            raise RuntimeError(f'Coefficient artifact hash mismatch: {run_id}')
        for source in (cp,rp,base/'reference_coefficients'/f'{target}.npz'):
            provenance[str(source.relative_to(ROOT))]=digest(source)
        with np.load(cp) as arrays:
            coefs=arrays['coef_history']; truth=arrays['w_true']; ref=arrays['ref_sklearn']; ref2=arrays['ref_skglm']
            if len(history)!=len(coefs) or len(coefs)==0:
                raise RuntimeError(f'Checkpoint alignment failed: {run_id}')
            if not np.array_equal(arrays['iterations'],np.asarray([h['iteration'] for h in history])):
                raise RuntimeError(f'Iteration alignment failed: {run_id}')
            if not np.array_equal(arrays['final_coef'],coefs[-1]):
                raise RuntimeError(f'Final checkpoint mismatch: {run_id}')
            coefs=coefs.copy();truth=truth.copy();ref=ref.copy();ref2=ref2.copy()
        statuses[r['status']]+=1
        numerical_signatures.add(r.get('numerical_signature',r.get('source_signature','unknown')))
        config_hashes.add(r.get('config_sha256','unknown'))
        details={key:r.get(key,r.get('params',{}).get(key)) for key in ('family','rho','layout','amplitude','method','seed')}
        details.update(n=r['n'],p=r['p'],lambda_value=r['lambda'])
        details.update(run_id=run_id,target_id=target,status=r['status'])
        for tau in TAUS:
            consensus=bool(np.array_equal(np.abs(ref)>tau,np.abs(ref2)>tau))
            accepted=bool(refs.get('reference_accepted',False)) and consensus
            # Consensus is numerical evidence, never an exact-minimizer certificate.
            common=dict(details,tau=tau,reference_consensus=accepted,
                        raw_reference_support_agreement=consensus)
            geometry=refs.get('geometry',{}).get('sklearn',{}).get(str(tau),{})
            common.update(reference_beta_min=geometry.get('beta_min'),
                          reference_inactive_kkt_margin=geometry.get('inactive_kkt_margin'),
                          reference_active_gram_condition=geometry.get('active_gram_condition'))
            final=dict(common,**errors(np.abs(coefs[-1])>tau,np.abs(ref)>tau,np.abs(truth)>0),
                       **sign_errors(coefs[-1],ref,truth,tau),
                       **observed_times(coefs,history,ref,truth,tau),elapsed_sec=r['elapsed_sec'],
                       diagnostics_fraction=r['diagnostics_sec']/r['elapsed_sec'] if r['elapsed_sec'] else 0.,
                       callback_snapshot_copy_sec=r['callback_snapshot_copy_sec_within_solver_timer'],
                       offline_gap_safe_sec=r['offline_gap_safe_and_snapshot_stack_sec'],
                       final_gap=history[-1]['normalized_gap'])
            finals.append(final)
            found={}
            for tol in THRESHOLDS:
                candidates=[i for i,h in enumerate(history) if h['normalized_gap']<=tol]
                if not candidates:
                    continue
                i=candidates[0];found[tol]=i;selected=np.abs(coefs[i])>tau
                item=dict(common,gap_threshold=tol,checkpoint_index=i,iteration=history[i]['iteration'],
                          time=history[i]['time'],observed_gap=history[i]['normalized_gap'],
                          gap_safe_zero_count=history[i].get('gap_safe_zero_count'),
                          gap_safe_reference_zero_coverage=history[i].get('gap_safe_reference_zero_coverage') if tau==1e-6 else None,
                          gap_safe_excluded_reference_nonzero_count=history[i].get('gap_safe_excluded_reference_nonzero_count') if tau==1e-6 else None,
                          **errors(selected,np.abs(ref)>tau,np.abs(truth)>0),
                          **sign_errors(coefs[i],ref,truth,tau))
                rows.append(item)
            for loose,tight in ((1e-2,1e-6),(1e-4,1e-6),(1e-6,1e-10)):
                if loose not in found or tight not in found:
                    continue
                a,b=found[loose],found[tight];ma=np.abs(coefs[a])>tau;mb=np.abs(coefs[b])>tau
                ca=support_counts(ma,np.abs(truth)>0);cb=support_counts(mb,np.abs(truth)>0)
                t0=history[a]['time'];t1=history[b]['time']
                transitions.append(dict(common,loose_gap=loose,tight_gap=tight,
                    supports_different=bool(np.any(ma!=mb)),symmetric_difference=int(np.sum(ma!=mb)),
                    counts_different=any(ca[k]!=cb[k] for k in ('tp','fp','fn')),
                    support_swap_with_same_counts=bool(np.any(ma!=mb)) and all(ca[k]==cb[k] for k in ('tp','fp','fn')),
                    f1_change=cb['f1']-ca['f1'],extra_observed_sec=t1-t0,
                    extra_fraction_of_tight_time=(t1-t0)/t1 if t1>0 else 0.))
    if len(numerical_signatures)!=1 or len(config_hashes)!=1:
        raise RuntimeError('Mixed numerical signatures or configurations')
    grouped=defaultdict(list); planned_groups=defaultdict(list)
    for row in finals:
        planned_groups[tuple(row[k] for k in CELL[:-1])].append(row)
    for row in rows:
        grouped[tuple(row[k] for k in CELL)].append(row)
    cells=[]
    expected_cell_keys=[key+(threshold,) for key in planned_groups for threshold in THRESHOLDS]
    for key in sorted(expected_cell_keys,key=str):
        group=grouped.get(key,[])
        good=[r for r in group if r['reference_consensus']]
        planned=planned_groups[key[:-1]]
        item=dict(zip(CELL,key));item.update(n_runs=len(group),n_reference_consensus=len(good),
            n_registered_runs=len(planned),n_gap_reached=len(group),n_gap_not_reached=len(planned)-len(group),
            n_registered_reference_consensus=sum(r['reference_consensus'] for r in planned),
            n_reached_reference_uncertain=len(group)-len(good),
            n_support_matches=sum(r['same_support_as_reference'] for r in good),
            n_recovery_count_matches=sum(r['same_recovery_counts_as_reference'] for r in good))
        item['observed_reference_match_yield_vs_planned']=item['n_support_matches']/len(planned)
        if len({r['seed'] for r in group})!=len(group):
            raise RuntimeError('Repeated data seed within a method/condition cell')
        item['reference_support_match_wilson95']=wilson(item['n_support_matches'],len(good))
        item['recovery_count_match_wilson95']=wilson(item['n_recovery_count_matches'],len(good))
        for metric in ('f1','reference_f1','numerical_support_error','numerical_harm','numerical_repair','time','gap_safe_reference_zero_coverage'):
            item[metric]=stats([r[metric] for r in good])
        cells.append(item)
    transition_summary=[]
    groups=defaultdict(list)
    for row in transitions:
        key=tuple(row[k] for k in ('method','tau','loose_gap','tight_gap'))
        groups[key].append(row)
    for key,g in sorted(groups.items(),key=lambda kv:str(kv[0])):
        item=dict(zip(('method','tau','loose_gap','tight_gap'),key));item.update(n_runs=len(g),
            supports_different=sum(r['supports_different'] for r in g),counts_different=sum(r['counts_different'] for r in g),
            support_swaps_with_same_counts=sum(r['support_swap_with_same_counts'] for r in g),
            extra_observed_sec=stats([r['extra_observed_sec'] for r in g]),
            extra_fraction_of_tight_time=stats([r['extra_fraction_of_tight_time'] for r in g]))
        transition_summary.append(item)
    summary=dict(stage=args.stage,integrity_verified=verified,analysis_script_sha256=digest(Path(__file__)),n_runs=len(paths),n_targets=len(reference_by_target),
        statuses=dict(statuses),numerical_signatures=sorted(numerical_signatures),config_sha256=sorted(config_hashes),
        reference_targets_accepted=sum(bool(r.get('reference_accepted',False)) for r in reference_by_target.values()),
        cells=cells,transition_summary=transition_summary,finals=finals,checkpoints=rows,
        input_provenance=provenance,
        interpretation=['Exploratory results; independent data seeds are the replication unit.',
        'Numerical consensus does not prove exact support or statistical recovery.',
        'Observed additional time is descriptive; no deployable stopping-time saving is established.',
        'Threshold hit and last mismatch times refer only to observed checkpoints.',
        'Wilson intervals apply within condition and method, conditional on accepted numerical references AND reaching that gap; checkpoints and paired methods are not independent replicates.',
        'The support-adequacy figure uses all planned runs as denominator for observed reference-match yield; unknown references and unreached gaps are not counted as observed matches.',
        'Support-error accounting is a binary identity, not a new theorem.'])
    if args.stage=='pilot' and (len(paths)!=72 or len(reference_by_target)!=36):
        raise RuntimeError('Pilot is incomplete: expected 36 distinct targets and 72 trajectories')
    if len(paths)!=completion['recorded_trajectories'] or len(reference_by_target)!=completion['recorded_targets']:
        raise RuntimeError('Saved files do not match the declared completed grid')
    (base/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    write_csv(base/'checkpoint_comparisons.csv',rows);write_csv(base/'support_transitions.csv',transitions)
    write_csv(base/'final_support_comparisons.csv',finals)
    plot(summary,ROOT/'figures/workshop'/args.stage)
    print(json.dumps({k:summary[k] for k in ('stage','n_runs','n_targets','statuses','reference_targets_accepted','transition_summary')},indent=2))


if __name__=='__main__':
    main()
