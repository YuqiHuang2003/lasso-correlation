#!/usr/bin/env python3
"""Package the verified research increment without rerunning optimizers."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    required = [(ROOT/'results/reference_audit_v1/analysis_verification.json', 'scripts/verify_reference_analysis_v1.py'),
                (ROOT/'results/uji_preprocessing_control_v1/verification.json', 'scripts/verify_uji_preprocessing_control_v1.py')]
    for path, verifier in required:
        verification = json.loads(path.read_text())
        if not verification['all_passed']:
            raise RuntimeError('Unverified results: '+str(path))
        if verification['verifier_script_sha256'] != sha(ROOT/verifier):
            raise RuntimeError('Changed verifier: '+verifier)
        for relative, digest in verification.get('input_provenance', {}).items():
            if sha(ROOT/relative) != digest:
                raise RuntimeError('Stale verified input: '+relative)
    validation = json.loads((ROOT/'report/reference_audit_v1/validation.json').read_text())
    pdf = ROOT/'report/reference_audit_v1/external_audit_zh.pdf'
    if sha(pdf) != validation['pdf_sha256']:
        raise RuntimeError('Changed PDF after validation')
    if validation['builder_sha256'] != sha(ROOT/'scripts/build_reference_audit_brief_v1.py'):
        raise RuntimeError('Changed report builder')
    if validation['analysis_verification_sha256'] != sha(required[0][0]):
        raise RuntimeError('Changed analysis verification after report build')
    if validation['control_verification_sha256'] != sha(required[1][0]):
        raise RuntimeError('Changed control verification after report build')
    for stage, digest in validation['analysis_sha256'].items():
        if sha(ROOT/f'results/reference_audit_v1/{stage}/analysis.json') != digest:
            raise RuntimeError('Stale report analysis')
    files = set()
    def add(path):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py','.c','.h','.json','.md','.csv','.npz','.pdf','.png','.txt','.bib','.ttf'}:
            files.add(path)
    for folder in ('src','scripts','tests','configs','docs','results/reference_audit_v1',
                   'results/uji_preprocessing_control_v1','figures/reference_audit_v1'):
        for path in (ROOT/folder).rglob('*'):
            add(path)
    for path in (ROOT/'report/reference_audit_v1').iterdir():
        add(path)
    for path in (ROOT/'data').rglob('source.json'):
        add(path)
    for path in (ROOT/'data').rglob('*manifest*.json'):
        add(path)
    old = ROOT/'results/workshop/real'
    for name in ('environment.json','config.json','effective_config.json','completion.json'):
        add(old/name)
    config = json.loads((ROOT/'configs/uji_preprocessing_control_v1.json').read_text())
    for seed in config['selected_seeds']:
        for layout in config['layouts']:
            for amplitude in config['amplitudes']:
                identifier = f'uji_seed{seed}_{layout}_{amplitude}'
                for folder, extension in [('datasets','.json'),('references','.json'),('reference_coefficients','.npz')]:
                    add(old/folder/(identifier+extension))
    for path in ROOT.glob('requirements*.txt'):
        add(path)
    for name in ('README.md','TODO.md','report/chinese.ttf','report/references.bib'):
        add(ROOT/name)
    manifest = {str(path.relative_to(ROOT)):sha(path) for path in sorted(files)}
    explanation = ('研究增量包：含外部审计、UJI处理对照及其所需的12个旧基线目标；不含原始数据、'
                   '环境、旧完整轨迹或课程/旧workshop完整PDF。旧完整研究包单独保留。'
                   '本包没有发布或提交到外部平台。')
    payload = {'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               'scope':explanation,'file_count':len(manifest),'files':manifest}
    output = ROOT/'deliverables'
    output.mkdir(exist_ok=True)
    target = output/'comp6704_reference_audit_v1.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in manifest:
            archive.write(ROOT/relative, relative)
        archive.writestr('MANIFEST.json',json.dumps(payload,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('ZIP CRC validation failed')
        for relative, digest in manifest.items():
            if hashlib.sha256(archive.read(relative)).hexdigest() != digest:
                raise RuntimeError('ZIP payload mismatch: '+relative)
    (output/'manifest_reference_audit_v1.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    info = {'path':str(target.relative_to(ROOT)), 'bytes':target.stat().st_size,
            'sha256':sha(target), 'payload_files':len(manifest), 'verified_payload_hashes':True,
            'scope':explanation,'optimization_calls':0}
    (output/'package_reference_audit_v1.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print(json.dumps(info,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
