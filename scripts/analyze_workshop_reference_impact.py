#!/usr/bin/env python3
"""Measure recovery-score differences between the two saved numerical references.

No fitting or grouping is performed. Targets are paired within ten data seeds.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=ROOT/'results/workshop/real'
    verification=json.loads((base/'verification.json').read_text())
    if not verification.get('all_passed'):
        raise RuntimeError('Complete numerical verification is required')
    provenance={};rows=[]
    paths=sorted((base/'reference_coefficients').glob('*.npz'))
    if len(paths)!=40:
        raise RuntimeError('The registered real-design arm requires 40 targets')
    for path in paths:
        relative=str(path.relative_to(ROOT));actual=sha(path)
        if verification['input_provenance'][relative]!=actual:
            raise RuntimeError('Verified input coefficient artifact changed')
        provenance[relative]=actual
        with np.load(path,allow_pickle=False) as arrays:
            truth=arrays['w_true']!=0
            for tau in (1e-6,1e-4):
                metrics=[]
                for key in ('ref_sklearn','ref_skglm'):
                    selected=np.abs(arrays[key])>tau
                    tp=int(np.sum(selected&truth));fp=int(np.sum(selected&~truth));fn=int(np.sum(~selected&truth))
                    metrics.append(dict(tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.))
                rows.append(dict(target_id=path.stem,tau=tau,sklearn=metrics[0],skglm=metrics[1],
                                 absolute_f1_difference=abs(metrics[0]['f1']-metrics[1]['f1'])))
    summaries={}
    for tau in (1e-6,1e-4):
        selected=[r for r in rows if r['tau']==tau]
        summaries[str(tau)]=dict(n_targets=len(selected),
            targets_with_different_f1=sum(r['absolute_f1_difference']>1e-12 for r in selected),
            maximum_absolute_f1_difference=max(r['absolute_f1_difference'] for r in selected),
            maximum_case=max(selected,key=lambda r:r['absolute_f1_difference']))
    result=dict(analysis_script_sha256=sha(Path(__file__)),source_verification_sha256=sha(base/'verification.json'),
        input_provenance=provenance,summaries=summaries,targets=rows,
        scope='Post-hoc paired numerical-reference comparison on all 40 registered targets; no new fits. Ten seeds, not 40 independent datasets. Differences do not identify the correct reference or prove mathematical non-uniqueness.')
    (base/'reference_recovery_impact.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
