"""Portable checks and run preparation for the supplied U-GMA review archive.

This helper leaves the archived scientific code and original run records intact.
Summary reconstruction uses the standard library and requires no training.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import urllib.request


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_digest(data):
    return digest(data.replace(b'\r\n', b'\n'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def ett_check(path, spec):
    data = path.read_bytes()
    actual = canonical_digest(data)
    if actual != spec['lf_sha256']:
        raise ValueError(f'Unexpected ETT contents: {path}')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.reader(handle))
    if rows[0] != spec['columns'] or len(rows)-1 != spec['rows']:
        raise ValueError(f'Unexpected ETT dimensions: {path}')
    return {'rows': len(rows)-1, 'numeric_channels': len(rows[0])-1,
            'sha256': digest(data), 'lf_sha256': actual}


def fetch_data(root, sources):
    destination = root/'external/ett/ETT-small'
    destination.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, spec in sources['ett'].items():
        target = destination/(name+'.csv')
        if not target.exists():
            with urllib.request.urlopen(spec['pinned_url'], timeout=45) as response:
                data = response.read(20_000_000)
            if canonical_digest(data) != spec['lf_sha256']:
                raise ValueError(f'Downloaded ETT contents differ: {name}')
            target.write_bytes(data)
        report[name] = ett_check(target, spec)
    return report


def verify(root, sources, catalog):
    manifest = read(root/'SHA256.json')
    failures = [name for name, expected in manifest.items()
                if not (root/name).is_file() or digest((root/name).read_bytes()) != expected]
    if failures:
        raise ValueError('Archive file differences: '+', '.join(failures[:12]))
    datasets = {name: ett_check(root/'external/ett/ETT-small'/(name+'.csv'), spec)
                for name, spec in sources['ett'].items()}
    for row in catalog:
        if digest((root/row['record']).read_bytes()) != row['record_sha256']:
            raise ValueError('Run record changed: '+row['id'])
    return {'archive_files_verified': len(manifest), 'indexed_runs_verified': len(catalog), 'datasets': datasets}


def summarize(root, catalog, output):
    gains, rmse_gains, groups = [], [], {}
    for row in catalog:
        if row['group'] != 'r0':
            continue
        result = read(root/row['record'])['results']
        base, corrected = result['backbone']['mae'], result['enhanced']['mae']
        gain = 100*(base-corrected)/base
        gains.append(gain)
        rmse_gains.append(100*(result['backbone']['rmse']-result['enhanced']['rmse'])/result['backbone']['rmse'])
        key = f"{row['dataset']}/{row['backbone']}/Q{row['horizon']}"
        groups.setdefault(key, []).append(gain)
    if len(gains) != 80 or len(groups) != 16 or any(len(v) != 5 for v in groups.values()):
        raise ValueError('Expected the 80-run R0 comparison with five seeds per setting')
    result = {'r0': {'runs': len(gains), 'improved_runs': sum(x > 0 for x in gains),
        'mean_mae_reduction_percent': statistics.mean(gains),
        'mean_rmse_reduction_percent': statistics.mean(rmse_gains),
        'settings': {k: {'n': len(v), 'mean_gain_percent': statistics.mean(v),
                          'sample_sd_gain_percent': statistics.stdev(v)} for k, v in sorted(groups.items())}}}
    def grouped_csv(name, predicate, group_columns, value_column):
        groups = {}
        with (root/'paper_evidence'/name).open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if predicate(row):
                    key = '/'.join(row[k] for k in group_columns)
                    groups.setdefault(key, []).append(float(row[value_column]))
        if any(len(v) != 5 for v in groups.values()):
            raise ValueError('Expected five independent seeds: '+name)
        return {k: {'n': len(v), 'mean': statistics.mean(v), 'sample_sd': statistics.stdev(v)} for k, v in sorted(groups.items())}
    result['relu_smm_clean_spearman'] = grouped_csv('risk_scores_per_seed.csv',
        lambda r: r['kind'] == 'clean' and r['score'] == 'risk', ['dataset', 'backbone'], 'rho')
    result['tafas_delayed_complete_target_mae'] = grouped_csv('tta_per_seed.csv',
        lambda r: r['method'] == 'tafas_delayed_full_labels', ['dataset'], 'mae')
    write_json(output/'summary.json', result)
    return result


def export_configs(root, catalog, group, output):
    selected = [r for r in catalog if r['group'] == group]
    expected = {'r0': 80, 'relu': 20, 'softplus': 20}[group]
    if len(selected) != expected:
        raise ValueError('Unexpected group size')
    config_dir = output/'configs'/group
    config_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        record = read(root/row['record'])
        cfg = copy.deepcopy(record['config'])
        cfg['data']['path'] = str(root/'external/ett/ETT-small'/(row['dataset']+'.csv'))
        if cfg['data'].get('reference_path'):
            raise ValueError('This exporter covers the unchanged ETT inputs')
        cfg['output_dir'] = str(output/'runs'/row['id'])
        if (Path(cfg['output_dir'])/'run.json').exists():
            raise ValueError('This rerun already has a result; choose a new --output directory')
        write_json(config_dir/(row['id']+'.json'), cfg)
    return {'group': group, 'configurations': len(selected), 'directory': str(config_dir),
            'changed_fields': ['data.path', 'output_dir'],
            'training_started': False}


def inspect_iot(path):
    import numpy as np
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.ndim != 2:
        raise ValueError('Expected a time-by-variable array')
    return {'file': str(path), 'rows': int(array.shape[0]), 'columns': int(array.shape[1]),
            'dtype': str(array.dtype), 'all_values_finite': bool(np.isfinite(array).all()),
            'sha256': digest(path.read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fetch-data', 'verify', 'summary', 'configs', 'inspect-iot'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--group', choices=['r0', 'relu', 'softplus'], default='r0')
    parser.add_argument('--file', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root/'reproduction_outputs'
    if output == root/'results' or root/'results' in output.parents:
        parser.error('Choose an output directory outside the archived results folder')
    sources = read(root/'reproduction/data_sources.json')
    catalog = read(root/'reproduction/run_catalog.json')
    if args.command == 'fetch-data':
        result = fetch_data(root, sources)
    elif args.command == 'verify':
        result = verify(root, sources, catalog)
    elif args.command == 'summary':
        result = summarize(root, catalog, output)
    elif args.command == 'configs':
        result = export_configs(root, catalog, args.group, output)
    else:
        if args.file is None:
            parser.error('--file is required for inspect-iot')
        result = inspect_iot(args.file)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
