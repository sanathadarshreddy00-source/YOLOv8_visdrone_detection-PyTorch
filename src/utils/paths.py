"""Central path loader.

Loads `configs/paths.yaml` and exposes Path objects for commonly used project
locations so scripts can import `src.utils.paths` and use `paths.IMAGES` etc.
"""
from pathlib import Path
import yaml
import os


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"


def _load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"paths config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


_cfg = _load_config()


def _make_path(p):
    if p is None:
        return None
    return (PROJECT_ROOT / p).resolve()


# Data paths
IMAGES = _make_path(_cfg.get('data', {}).get('images'))
ANNOTATIONS = _make_path(_cfg.get('data', {}).get('annotations'))
LABELS = _make_path(_cfg.get('data', {}).get('labels'))
DATASET = _make_path(_cfg.get('data', {}).get('dataset'))
SPLITS = _make_path(_cfg.get('data', {}).get('splits'))

# Output paths
PREDICTIONS = _make_path(_cfg.get('outputs', {}).get('predictions'))
VERIFICATION = _make_path(_cfg.get('outputs', {}).get('verification'))
VIDEOS = _make_path(_cfg.get('outputs', {}).get('videos'))

# Logging / runs
LOGS = _make_path(_cfg.get('logging', {}).get('logs'))
RUNS_PROJECT = _make_path(_cfg.get('logging', {}).get('runs_project'))

# Misc
BACKUPS = _make_path(_cfg.get('misc', {}).get('backups'))
TMP = _make_path(_cfg.get('misc', {}).get('temp'))


def ensure_dirs(*paths):
    """Create directories for the provided Path objects if they don't exist."""
    for p in paths:
        if p is None:
            continue
        Path(p).mkdir(parents=True, exist_ok=True)


__all__ = [
    'PROJECT_ROOT', 'IMAGES', 'ANNOTATIONS', 'LABELS', 'DATASET', 'SPLITS',
    'PREDICTIONS', 'VERIFICATION', 'VIDEOS', 'LOGS', 'RUNS_PROJECT', 'BACKUPS', 'TMP',
    'ensure_dirs'
]
