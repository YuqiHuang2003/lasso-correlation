#!/usr/bin/env python3
"""Check byte-identical scientific files without datasets or scientific packages."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'repository_snapshot.json').read_text())
    issues = []
    for relative, record in manifest['files'].items():
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            issues.append({'file': relative, 'reason': 'missing file or symlink'})
        elif path.stat().st_size != record['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            issues.append({'file': relative, 'reason': 'snapshot bytes changed'})
    result = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, capture_output=True, check=True)
    forbidden = ('.venv/', '.modern_deps/', '.report_deps/', '.aws/', '.codex/', 'data/raw/',
                 'data/benchmarks/', 'deliverables/', 'publication/', 'publication_support/')
    for relative in result.stdout.decode().split('\0'):
        if not relative:
            continue
        if relative.startswith(forbidden) or '__pycache__/' in relative or relative.endswith(('.pyc', '.so')):
            issues.append({'file': relative, 'reason': 'excluded local/input/build file is tracked'})
        if Path(relative).name in {'assignment.pdf', 'assignment_extracted.txt', 'user_research_plan_2026-10-28.txt', 'id_rsa', 'chinese.ttf'}:
            issues.append({'file': relative, 'reason': 'excluded document or credential is tracked'})
    print(json.dumps({'all_passed': not issues, 'snapshot_files': len(manifest['files']),
                      'raw_data_required': False, 'optimizer_calls': 0, 'issues': issues}, indent=2))
    return bool(issues)


if __name__ == '__main__':
    sys.exit(main())
