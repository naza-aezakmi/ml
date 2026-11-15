"""Audit the bundled production atlas, without downloading data or ML dependencies.

Run from this repository: python3 scripts/audit_production.py --api ../api
Pass --output docs/production-data-audit.json to update the checked-in snapshot.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def audit(api: Path) -> dict:
    directory = api / 'terrain-atlas'
    index_file = directory / 'relief/index.json'
    screen_file = directory / 'relief/field-screen-v1.gz'
    manifest = json.loads(index_file.read_text())
    screen = gzip.decompress(screen_file.read_bytes())
    assert len(screen) == manifest['worlds']['earth']['count'], 'Invalid screen length'
    worlds = {}
    for name, world in manifest['worlds'].items():
        parts = [part for part in world['tiles'] if part.get('file')]
        files = [directory / 'relief' / name / part['file'] for part in parts]
        assert all(path.is_file() for path in files), f'Missing {name} tiles'
        worlds[name] = {
            'source': world['label'], 'source_url': world['url'],
            'native_spacing_approx_m_at_equator': world['resolution'],
            'global_grid_cells': world['count'],
            'stored_windows': sum(part['count'] for part in parts),
            'profile_samples': sum(part['count'] for part in parts) * manifest['samples'] ** 2,
            'compressed_tile_files': len(files),
            'compressed_tile_bytes': sum(path.stat().st_size for path in files),
        }
    return {
        'scope': 'Production Moon/Mars terrain retrieval, separate from experimental clustering',
        'samples_per_axis': manifest['samples'],
        'samples_per_window': manifest['samples'] ** 2,
        'window_width_m': manifest['step'],
        'sample_spacing_m': manifest['step'] / (manifest['samples'] - 1),
        'height_quantization_m': manifest['scale'],
        'worlds': worlds,
        'earth_screen': {
            'eligible_windows': sum(bool(value & 1) for value in screen),
            'eligible_usa_windows': sum(value & 3 == 3 for value in screen),
            'flag_frequencies': dict(sorted(Counter(screen).items())),
        },
        'sha256': {
            'relief/index.json': hashlib.sha256(index_file.read_bytes()).hexdigest(),
            'relief/field-screen-v1.gz': hashlib.sha256(screen_file.read_bytes()).hexdigest(),
            'relief-worker.js': hashlib.sha256((directory / 'relief-worker.js').read_bytes()).hexdigest(),
        },
        'interpretation': 'Counts describe bundled derived windows and samples, not independent observations or raw mission archive size. Source DEM pixels can be reused across windows.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api', type=Path, default=Path('../api'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    serialized = json.dumps(audit(args.api), ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end='')
