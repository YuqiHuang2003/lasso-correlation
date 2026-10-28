#!/usr/bin/env python3
"""Run literature-aligned correlation x sparsity experiments, separately from v1."""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings
import numpy as np
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from lasso_study.data_v2 import make_synthetic, make_uji, make_leukemia
from lasso_study.solvers import solve,certificate,lipschitz_constant,warmup_cd
from lasso_study.modern_solver import solve_modern,warmup_modern
from run_study import dump,sha,environment


def signature():
    paths=[ROOT/'src/lasso_study'/p for p in ('data.py','data_v2.py','solvers.py','cd_kernel.c','modern_solver.py')]
    paths.append(Path(__file__))
    return hashlib.sha256(''.join(sha(p) for p in paths).encode()).hexdigest()


def support_metrics(w,truth,threshold,block_size=20):
    selected=np.abs(w)>threshold; actual=np.abs(truth)>0
    tp=int(np.sum(selected&actual));fp=int(np.sum(selected&~actual));fn=int(np.sum(~selected&actual))
    precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn)
    gs=selected.reshape(-1,block_size).any(axis=1);ga=actual.reshape(-1,block_size).any(axis=1)
    gt=int(np.sum(gs&ga));gp=gt/max(1,int(gs.sum()));gr=gt/max(1,int(ga.sum()))
    estimated_sign=np.where(selected,np.sign(w),0.)
    return {'precision':precision,'recall':recall,'f1':2*precision*recall/max(1e-300,precision+recall),
            'fdr':fp/max(1,tp+fp),'tp':tp,'fp':fp,'fn':fn,'nnz':int(selected.sum()),
            'exact_support':bool(np.array_equal(selected,actual)),
            'exact_sign':bool(np.array_equal(estimated_sign,np.sign(truth))),
            'active_sign_accuracy':float(np.mean(estimated_sign[actual]==np.sign(truth[actual]))),
            'group_precision':gp,'group_recall':gr,'group_f1':2*gp*gr/max(1e-300,gp+gr),
            'selected_groups':int(gs.sum()),'true_groups':int(ga.sum()),
            'relative_coef_error':float(np.linalg.norm(w-truth)/np.linalg.norm(truth))}


def callback(data,config):
    if 'w_true' in data:
        return lambda w:support_metrics(w,data['w_true'],config['support_threshold'],config['synthetic']['block_size'])
    return lambda w:{'nnz':int(np.sum(np.abs(w)>config['support_threshold']))}


def structure(data,block_size):
    X,truth=data['X'],data['w_true'];active=np.flatnonzero(truth);inactive=np.flatnonzero(truth==0)
    gram=X[:,active].T@X[:,active]/len(X);ev=np.linalg.eigvalsh(gram)
    score=(X[:,inactive].T@X[:,active]/len(X))@np.linalg.solve(gram,np.sign(truth[active]))
    grouped=truth.reshape(-1,block_size);counts=np.count_nonzero(grouped,axis=1);signs=grouped.sum(axis=1)
    rho=data['metadata']['rho']
    population_var=float(np.sum((1-rho)*counts+rho*signs**2))
    return {'active_gram_min_eigenvalue':float(ev[0]),'active_gram_condition':float(ev[-1]/ev[0]),
            'sample_irrepresentable_score':float(np.max(np.abs(score))),
            'full_hessian_rank_upper_bound':min(X.shape[0]-1,X.shape[1]),
            'full_hessian_singular':bool(X.shape[1]>X.shape[0]-1),
            'group_signal_counts':counts.tolist(),'group_sign_sums':signs.tolist(),
            'raw_population_signal_variance_formula':population_var,
            'population_formula_scope':'unit-scale population block formula; training sample standardization changes exact finite-sample covariance'}


def reference(X,y,lam):
    started=perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        fit=Lasso(alpha=lam,fit_intercept=False,tol=1e-11,max_iter=200000,selection='cyclic').fit(X,y)
    cert=certificate(X,y,fit.coef_,lam)
    if cert['normalized_gap']>1e-7:
        raise RuntimeError(f"Reference did not converge: {cert['normalized_gap']}")
    return dict(cert,elapsed_sec=perf_counter()-started,iterations=int(fit.n_iter_),warnings=[str(x.message) for x in caught])


def run_one(data,config,output,run_id,details,L,svd_sec,ref,config_hash,source_sig):
    path=output/'runs'/(run_id+'.json')
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('numerical_signature')!=source_sig or old['config_sha256']!=config_hash or old['dataset_hash']!=details['dataset_hash']:
            raise RuntimeError(f'Refusing mismatched cached result {path}')
        return old
    X=data.get('X',data.get('X_train'));y=data.get('y',data.get('y_train'))
    method=details['method']
    try:
        if method=='skglm':
            result=solve_modern(X,y,details['lambda'],L=L,metric_callback=callback(data,config),**config['solver'],**config['modern'])
        else:
            result=solve(X,y,details['lambda'],method,L=L,metric_callback=callback(data,config),**config['solver'])
            result['iteration_unit']='complete cyclic epoch' if method=='cd' else 'proximal iteration'
            result['threshold_curve_comparable']=True
        coef=result.pop('coef');last=result['history'][-1]
        result.update(details,run_id=run_id,config_sha256=config_hash,numerical_signature=source_sig,
                      lipschitz_setup_sec=svd_sec,reference=ref,
                      objective_difference_to_reference=last['primal']-ref['primal'],
                      cold_start_sec=result['elapsed_sec']+(svd_sec if method in ('ista','fista') else 0.))
        result['threshold_hits']={str(t):next((h for h in result['history'] if h['normalized_gap']<=t),None) for t in config['thresholds']}
        evaluation_start=perf_counter()
        if 'w_true' in data:
            result['support_sensitivity']={str(t):support_metrics(coef,data['w_true'],t,config['synthetic']['block_size']) for t in config['support_sensitivity']}
            result['test_mse']=float(np.mean((data['X_test']@coef-data['y_test'])**2))
            result['signal_prediction_mse']=float(np.mean((data['X_test']@(coef-data['w_true']))**2))
        elif 'X_test' in data:
            result['test_mse']=float(np.mean((data['X_test']@coef-data['y_test'])**2))
            result['validation_mse']=float(np.mean((data['X_val']@coef-data['y_val'])**2))
        result['evaluation_sec_outside_solver_timer']=perf_counter()-evaluation_start
        result['final']=last
        cp=output/'coefficients'/(run_id+'.npz');cp.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(cp,coef=coef);result['coefficient_sha256']=sha(cp)
        dump(path,result)
        print(json.dumps({'run':run_id,'status':result['status'],'sec':round(result['elapsed_sec'],4),
                          'gap':last['normalized_gap'],'f1':last.get('f1')}),flush=True)
        return result
    except Exception as error:
        dump(output/'failures'/(run_id+'.json'),dict(details,run_id=run_id,error_type=type(error).__name__,error=str(error),
             numerical_signature=source_sig,config_sha256=config_hash))
        raise


def synthetic(config,output,stage,config_hash,source_sig):
    params=config['synthetic'];base={k:params[k] for k in ('n','p','block_size','noise_std','n_test')}
    seeds=[0] if stage in ('pilot','smoke') else params['seeds']
    rhos=[0.,.9] if stage in ('pilot','smoke') else params['rhos']
    sparsities=[5,40] if stage=='pilot' else ([5] if stage=='smoke' else params['sparsities'])
    factors=[1.] if stage in ('pilot','smoke') else params['lambda_factors']
    if stage=='smoke':base.update(n=80,p=100,n_test=100)
    for seed in seeds:
        for rho in rhos:
            L=None;svd_sec=None;design_hash=None
            for layout in params['layouts']:
                for sparsity in sparsities:
                    prepared=perf_counter()
                    data=make_synthetic(seed=seed,rho=rho,sparsity=sparsity,layout=layout,**base)
                    X=np.asfortranarray(data['X']);data['X']=X;y=data['y']
                    preprocessing_sec=perf_counter()-prepared
                    if design_hash is None:
                        design_hash=data['metadata']['hashes']['X'];t=perf_counter();L=lipschitz_constant(X);svd_sec=perf_counter()-t
                    elif design_hash!=data['metadata']['hashes']['X']:
                        raise RuntimeError('Design changed across paired sparsity/layout factors')
                    dataset_id=f'syn_seed{seed}_r{rho:g}_s{sparsity}_{layout}'
                    geom=structure(data,base['block_size'])
                    dump(output/'datasets'/(dataset_id+'.json'),dict(data['metadata'],**geom,
                         preprocessing_sec=preprocessing_sec,L=L,lipschitz_setup_sec=svd_sec))
                    lambda_max=float(np.max(np.abs(X.T@y))/len(y))
                    for factor in factors:
                        lam=factor*base['noise_std']*np.sqrt(2*np.log(base['p'])/base['n'])
                        ref=reference(X,y,lam)
                        methods=np.random.default_rng(seed+int(100*rho)+sparsity+int(factor*10)+(0 if layout=='dispersed' else 1000)).permutation(config['methods'])
                        for method in methods:
                            details={'dataset':'synthetic','seed':seed,'rho':rho,'sparsity':sparsity,'layout':layout,
                                     'lambda_factor':factor,'lambda':float(lam),'lambda_max':lambda_max,'lambda_ratio':float(lam/lambda_max),
                                     'method':str(method),'dataset_hash':data['metadata']['hashes']['X']+data['metadata']['hashes']['y'],
                                     'n':len(y),'p':X.shape[1]}
                            run_one(data,config,output,f'{dataset_id}_c{factor:g}_{method}',details,L,svd_sec,ref,config_hash,source_sig)


def uji(config,output,config_hash,source_sig):
    settings=config['uji']
    for seed in settings['seeds']:
        t=perf_counter();data=make_uji(seed=seed);prep=perf_counter()-t
        X=np.asfortranarray(data['X_train']);data['X_train']=X;y=data['y_train']
        lm=float(np.max(np.abs(X.T@y))/len(y));data_hash=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()
        candidates=[];select_start=perf_counter()
        for ratio in np.geomspace(settings['min_ratio'],1.,settings['validation_candidates']):
            model=Lasso(alpha=lm*ratio,fit_intercept=False,tol=1e-10,max_iter=100000).fit(X,y)
            cert=certificate(X,y,model.coef_,lm*ratio)
            if cert['normalized_gap']>1e-7:raise RuntimeError('UJI selection candidate did not converge')
            candidates.append({'ratio':float(ratio),'lambda':float(lm*ratio),'validation_mse':float(np.mean((data['X_val']@model.coef_-data['y_val'])**2)),
                               'normalized_gap':cert['normalized_gap']})
        best=int(np.argmin([c['validation_mse'] for c in candidates]));selected=sorted(sorted(range(len(candidates)),key=lambda j:(abs(j-best),j))[:3])
        selection_sec=perf_counter()-select_start;t=perf_counter();L=lipschitz_constant(X);svd_sec=perf_counter()-t
        dump(output/'datasets'/f'uji_seed{seed}.json',dict(data['metadata'],candidates=candidates,best_index=best,selected_indices=selected,
              selection_sec=selection_sec,preprocessing_sec=prep,L=L,lipschitz_setup_sec=svd_sec))
        for index in selected:
            candidate=candidates[index];ref=reference(X,y,candidate['lambda'])
            for method in np.random.default_rng(seed+index).permutation(config['methods']):
                details={'dataset':'uji','seed':seed,'method':str(method),'lambda':candidate['lambda'],'lambda_ratio':candidate['ratio'],
                         'lambda_max':lm,'validation_index':index,'selected_by_validation':index==best,'dataset_hash':data_hash,'n':len(y),'p':X.shape[1]}
                run_one(data,config,output,f'uji_seed{seed}_i{index}_{method}',details,L,svd_sec,ref,config_hash,source_sig)


def leukemia(config,output,config_hash,source_sig):
    t=perf_counter();data=make_leukemia();prep=perf_counter()-t
    X=np.asfortranarray(data['X']);data['X']=X;y=data['y'];lm=float(np.max(np.abs(X.T@y))/len(y))
    data_hash=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest();t=perf_counter();L=lipschitz_constant(X);svd_sec=perf_counter()-t
    dump(output/'datasets'/'leukemia.json',dict(data['metadata'],preprocessing_sec=prep,L=L,lipschitz_setup_sec=svd_sec))
    for ratio in config['leukemia']['lambda_ratios']:
        lam=ratio*lm;ref=reference(X,y,lam)
        for repeat in config['leukemia']['repeats']:
            for method in np.random.default_rng(repeat+int(ratio*1000)).permutation(config['methods']):
                details={'dataset':'leukemia','repeat':repeat,'seed':None,'method':str(method),'lambda':lam,'lambda_ratio':ratio,
                         'lambda_max':lm,'dataset_hash':data_hash,'n':len(y),'p':X.shape[1],'has_true_support':False}
                run_one(data,config,output,f'leukemia_a{ratio:g}_rep{repeat}_{method}',details,L,svd_sec,ref,config_hash,source_sig)


def reproduce(config,output):
    checks=[];p=config['synthetic'];base={k:p[k] for k in ('n','p','block_size','noise_std','n_test')}
    for rho,sparsity,layout in ((0.,5,'dispersed'),(.9,40,'clustered')):
        data=make_synthetic(seed=0,rho=rho,sparsity=sparsity,layout=layout,**base);X=np.asfortranarray(data['X']);L=lipschitz_constant(X)
        lam=p['noise_std']*np.sqrt(2*np.log(p['p'])/p['n'])
        for method in config['methods']:
            run_id=f'syn_seed0_r{rho:g}_s{sparsity}_{layout}_c1_{method}';folder=ROOT/'results/v2/synthetic'
            old=json.loads((folder/'runs'/(run_id+'.json')).read_text());coef=np.load(folder/'coefficients'/(run_id+'.npz'))['coef']
            result=(solve_modern(X,data['y'],lam,metric_callback=callback(data,config),**config['solver'],**config['modern']) if method=='skglm'
                    else solve(X,data['y'],lam,method,L=L,metric_callback=callback(data,config),**config['solver']))
            delta=float(np.max(np.abs(result['coef']-coef)));objdelta=abs(result['history'][-1]['primal']-old['final']['primal'])
            checks.append({'run_id':run_id,'coef_max_difference':delta,'objective_difference':objdelta,
                           'passed':bool(delta<1e-7 and objdelta<1e-8 and result['status']==old['status'])})
    dump(output/'reproduction.json',{'checks':checks,'all_passed':all(x['passed'] for x in checks)})
    if not all(x['passed'] for x in checks):raise RuntimeError('Reproduction check failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['smoke','pilot','synthetic','uji','leukemia','reproduce'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/study_v2.json')
    args=parser.parse_args();config=json.loads(args.config.read_text());output=ROOT/'results/v2'/args.stage
    output.mkdir(parents=True,exist_ok=True)
    config_hash=sha(args.config);source_sig=signature()
    with threadpool_limits(limits=1):
        env_path=output/'environment.json'
        if env_path.exists() and json.loads(env_path.read_text()).get('v2_numerical_signature')!=source_sig:
            raise RuntimeError('Stage contains a different v2 numerical source; archive it before rerunning')
        env=environment();env.update(v2_numerical_signature=source_sig,v2_config_sha256=config_hash)
        if not env_path.exists():dump(env_path,env)
        dump(output/'config.json',config)
        t=perf_counter();warmup_cd();cd_sec=perf_counter()-t;modern=warmup_modern()
        dump(output/'warmup.json',{'cd_setup_sec':cd_sec,'modern':modern})
        started=perf_counter()
        if args.stage in ('smoke','pilot','synthetic'):synthetic(config,output,args.stage,config_hash,source_sig)
        elif args.stage=='uji':uji(config,output,config_hash,source_sig)
        elif args.stage=='leukemia':leukemia(config,output,config_hash,source_sig)
        else:reproduce(config,output)
        dump(output/'timing.json',{'wall_sec':perf_counter()-started})


if __name__=='__main__':main()
