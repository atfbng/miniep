"""Static isolated-directory checks; does not import or execute teaching code."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def module_path(root: Path, name: str) -> Path | None:
    base = root.joinpath(*name.split('.'))
    if base.with_suffix('.py').is_file():
        return base.with_suffix('.py')
    if (base / '__init__.py').is_file():
        return base / '__init__.py'
    return None


def declared_names(path: Path) -> set[str]:
    result = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            result.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            result.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
    return result


def source_from_git(repo: Path, name: str, revision: str) -> str:
    return subprocess.check_output(['git', 'show', revision + ':' + name], cwd=repo, text=True)


def check(repo: Path, courses: list[int] | None = None) -> dict:
    rows = []
    source_map = json.loads((repo / 'code/source_map.json').read_text())
    entry_map = {}
    for files in source_map.values():
        for target, data in files.items():
            if data.get('source', '').startswith('examples/'):
                entry_map[data['source']] = target
    with tempfile.TemporaryDirectory(prefix='miniep-course-isolation-') as directory:
        for number in (courses if courses is not None else range(14)):
            original = repo / 'code' / f'course{number}'
            assert original.is_dir(), original
            assert not any(x.is_symlink() for x in original.rglob('*')), original
            isolated = Path(directory) / f'lesson-{number}'
            shutil.copytree(original, isolated, ignore=shutil.ignore_patterns('__pycache__', '*.egg-info', 'results', '.pytest_cache'))
            requirement_files = [
                isolated / 'requirements.txt',
                *sorted(isolated.glob('requirements*.txt')),
            ]
            requirements = set()
            for requirement_file in dict.fromkeys(requirement_files):
                requirements.update(
                    re.split(r'[<>=!~\[; ]', line.strip())[0].replace('-', '_')
                    for line in requirement_file.read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith('#')
                )
            local_roots = {x.name for x in isolated.iterdir() if x.is_dir()}
            count = 0
            for file in isolated.rglob('*.py'):
                tree = ast.parse(file.read_text(), filename=str(file))
                count += 1
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports = [(alias.name, []) for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        assert not node.level, (file, 'unexpected relative import')
                        imports = [(node.module, [a.name for a in node.names])]
                    else:
                        imports = []
                    for name, symbols in imports:
                        top = name.split('.')[0]
                        if top in local_roots:
                            target = module_path(isolated, name)
                            assert target is not None, (file, name)
                            for symbol in symbols:
                                assert symbol in declared_names(target) or module_path(isolated, name + '.' + symbol), (file, name, symbol)
                        else:
                            assert top in sys.stdlib_module_names or top in requirements, (file, 'undeclared dependency', name)
                    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == 'parents':
                        assert isinstance(node.slice, ast.Constant) and node.slice.value == 1, (file, 'bootstrap escapes course')
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        assert not re.search(r'code/course\d|\.\./course\d|examples/\d', node.value), (file, 'cross-course or stale example path')
                        if node.value.startswith('test/') and node.value.endswith('.py'):
                            assert (isolated / node.value).is_file(), (file, node.value)
            for target, data in source_map.get(str(number), {}).items():
                file = isolated / target
                assert hashlib.sha256(file.read_bytes()).hexdigest() == data['sha256'], file
                if data.get('origin') in ('new', 'adapted'):
                    assert data.get('description'), (file, 'missing new-source description')
                    continue
                if file.suffix != '.py':
                    assert file.read_bytes() == subprocess.check_output(['git', 'show', data['source_revision'] + ':' + data['source']], cwd=repo)
                    continue
                old = source_from_git(repo, data['source'], data['source_revision'])
                if data['source'] == 'tests/test_advanced_lessons.py':
                    methods = {node.name: ast.dump(node) for node in ast.walk(ast.parse(old)) if isinstance(node, ast.FunctionDef)}
                    for node in ast.walk(ast.parse(file.read_text())):
                        if isinstance(node, ast.FunctionDef):
                            assert ast.dump(node) == methods[node.name], (file, node.name)
                else:
                    for source, replacement in entry_map.items():
                        old = old.replace(source, replacement)
                    old = old.replace('python -m unittest tests.', 'python -m unittest checks.')
                    assert ast.dump(ast.parse(old)) == ast.dump(ast.parse(file.read_text())), file
            main = isolated / 'test' / ('check_environment.py' if number == 0 else 'demo.py')
            assert main.is_file(), main
            rows.append({'course': number, 'python_files': count, 'isolated_import_graph_complete': True, 'source_semantics_preserved': True if 0 < number < 13 else None, 'source_status': 'new integration' if number == 13 else 'migrated' if number else 'new environment example'})
    return {'mode': 'static-only', 'courses': rows, 'not_executed': ['teaching examples', 'unit tests', 'GPU/NCCL', 'benchmarks', 'profiling']}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--courses', type=int, nargs='+', help='check only selected course numbers')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    report = check(repo, args.courses)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"PASS: {len(report['courses'])} isolated course directories; Python syntax, local imports, dependencies, paths and source metadata checked.")
    print('Teaching examples, tests and GPU workloads were not executed.')


if __name__ == '__main__':
    main()
