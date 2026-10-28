#!/usr/bin/env python3
"""Package the verified column-order research increment; never refit models."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base=ROOT/'results/column_permutation_v1'
    for name in ('verification.json','analysis_verification.json'):
        verification=json.loads((base/name).read_text())
        if not verification['all_passed'] or verification['verifier_script_sha256']!=sha(ROOT/'scripts/verify_column_permutation_v1.py'):
            raise RuntimeError('Current independent verification is required: '+name)
        for relative,digest in verification['input_provenance'].items():
            if sha(ROOT/relative)!=digest:raise RuntimeError('Stale verified input: '+relative)
    validation=json.loads((ROOT/'report/column_permutation_v1/validation.json').read_text())
    checks={'pdf_sha256':ROOT/'report/column_permutation_v1/column_order_zh.pdf',
            'builder_sha256':ROOT/'scripts/build_column_permutation_brief_v1.py',
            'analysis_sha256':base/'analysis.json','verification_sha256':base/'verification.json',
            'analysis_verification_sha256':base/'analysis_verification.json'}
    if any(validation[key]!=sha(path) for key,path in checks.items()):raise RuntimeError('Stale report inputs')
    files=set()
    def add(path):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in {'.py','.c','.h','.json','.md','.csv','.npz','.pdf','.png','.txt','.bib','.ttf'}:
            files.add(path)
    for directory in ('src','scripts','tests','configs','docs','results/column_permutation_v1','figures/column_permutation_v1'):
        for path in (ROOT/directory).rglob('*'):add(path)
    for path in (ROOT/'report/column_permutation_v1').iterdir():add(path)
    environment=json.loads((base/'environment.json').read_text())
    for relative in environment['frozen_input_hashes']:
        if not relative.startswith('data/'):
            add(ROOT/relative)
    for path in (ROOT/'data').rglob('source.json'):add(path)
    for path in (ROOT/'data').rglob('*manifest*.json'):add(path)
    for path in ROOT.glob('requirements*.txt'):add(path)
    for name in ('README.md','TODO.md','report/chinese.ttf','report/references.bib'):add(ROOT/name)
    manifest={str(path.relative_to(ROOT)):sha(path) for path in sorted(files)}
    scope=('列顺序研究增量：包含本阶段全部784输出、分析、中文简报、代码、协议及56个旧目标的必要基线记录；'
           '不含原始数据、安装环境或旧完整轨迹/报告。官方数据按下载脚本获取，旧完整包另存。未发布或提交外部平台。')
    payload={'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'scope':scope,
             'file_count':len(manifest),'files':manifest}
    out=ROOT/'deliverables';out.mkdir(exist_ok=True);target=out/'comp6704_column_permutation_v1.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for relative in manifest:archive.write(ROOT/relative,relative)
        archive.writestr('MANIFEST.json',json.dumps(payload,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:raise RuntimeError('ZIP CRC validation failed')
        for relative,digest in manifest.items():
            if hashlib.sha256(archive.read(relative)).hexdigest()!=digest:raise RuntimeError('ZIP content mismatch: '+relative)
    (out/'manifest_column_permutation_v1.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    info={'path':str(target.relative_to(ROOT)),'bytes':target.stat().st_size,'sha256':sha(target),
          'payload_files':len(manifest),'verified_payload_hashes':True,'optimizer_calls':0,'scope':scope}
    (out/'package_column_permutation_v1.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print(json.dumps(info,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
