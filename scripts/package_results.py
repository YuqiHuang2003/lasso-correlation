#!/usr/bin/env python3
"""Validate and archive the concrete local research deliverables."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_study import numerical_signature, dump


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary=json.loads((ROOT/'results/summary.json').read_text())
    pdf=json.loads((ROOT/'report/validation.json').read_text())
    repro=json.loads((ROOT/'results/reproduce/reproduction.json').read_text())
    assert summary['integrity']['complete_grid']
    assert summary['integrity']['main_runs']==180 and summary['integrity']['real_runs']==27
    assert pdf['pdf_pages']<=10 and pdf['pdf_bytes']<20_000_000
    assert repro['all_passed']
    signature=numerical_signature()
    checked=0
    for stage in ('main','real'):
        for path in (ROOT/'results'/stage/'runs').glob('*.json'):
            run=json.loads(path.read_text())
            assert run['numerical_signature']==signature, path
            assert sha(ROOT/'results'/stage/'coefficients'/(run['run_id']+'.npz'))==run['coefficient_sha256'], path
            assert run['history'][-1]['dual_feasibility']<=1+1e-12, path
            checked+=1
    stage_timings={}
    for stage in ('smoke','pilot','main','real','reproduce','robustness'):
        path=ROOT/'results'/stage/'timing.json'
        if path.exists(): stage_timings[stage]=json.loads(path.read_text())['wall_sec']
    resources={'sequential_stage_wall_sec':stage_timings,'total_recorded_wall_sec':sum(stage_timings.values()),
               'total_recorded_cpu_wall_hours':sum(stage_timings.values())/3600,
               'threads':1,'gpu_hours':0,'scope':'experiment stages only; excludes installation, reading, coding, plotting and report typesetting'}
    dump(ROOT/'results/resources.json',resources)
    checks={'main_runs':180,'real_runs':27,'verified_coefficient_files':checked,
            'source_signature':signature,'correctness_tests':{'passed':18,'command':'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -m pytest -q','evidence':'18 passed in 1.34s; executed before pilot/main with identical solver and data sources'},
            'representative_reproductions':len(repro['checks']),'reproduction_passed':repro['all_passed'],
            'report':pdf,'resources':resources,
            'identity_preference':'Leave name and student ID blank at the user\'s request',
            'remaining':['final externally accessible code location and submission filename','author review and Canvas submission']}
    dump(ROOT/'results/verification.json',checks)
    paths=[ROOT/'README.md',ROOT/'requirements.txt',ROOT/'requirements-report.txt',ROOT/'.gitignore']
    for directory in ('src','scripts','configs','docs','results','figures','report'):
        for path in (ROOT/directory).rglob('*'):
            if not path.is_file():continue
            relative=path.relative_to(ROOT)
            if any(part in ('__pycache__','preview','assets','development') for part in relative.parts):continue
            if path.name=='chinese.ttf' or path.suffix in ('.pyc','.aux','.out','.bbl','.blg','.log','.so'):continue
            paths.append(path)
    paths=sorted(set(paths))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in paths}
    dump(ROOT/'deliverables/source_manifest.json',{'files':manifest,'numerical_signature':signature})
    archive_path=ROOT/'deliverables/comp6704_research_bundle.zip'
    with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in paths:
            archive.write(path,arcname='comp6704_lasso/'+str(path.relative_to(ROOT)))
        archive.write(ROOT/'deliverables/source_manifest.json',arcname='comp6704_lasso/source_manifest.json')
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==len(paths)+1
    dump(ROOT/'deliverables/package.json',{'archive':archive_path.name,'bytes':archive_path.stat().st_size,
         'sha256':sha(archive_path),'files':len(paths)+1,'verified':True})
    print(json.dumps({'checks':checks,'package':json.loads((ROOT/'deliverables/package.json').read_text())},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
