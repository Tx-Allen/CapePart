"""Common helpers for loading configs and building dataset annotations."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional

from Config.default import get_cfg_defaults


def load_config(path: Optional[str] = None):
    """Load a configuration file or return the default cfg clone."""

    if path is None:
        return get_cfg_defaults()

    spec_path = Path(path)
    if not spec_path.is_file():
        raise FileNotFoundError(spec_path)

    module_name = spec_path.stem
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f'Unable to load config module from {path}')

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)  # type: ignore[attr-defined]

    if hasattr(module, 'cfg'):
        return module.cfg.clone()
    if hasattr(module, 'get_cfg_defaults'):
        return module.get_cfg_defaults()
    raise AttributeError('Config file must expose cfg or get_cfg_defaults')


def maybe_build_annotations(cfg) -> None:
    """Build train/val/test json splits automatically if needed."""

    if not (cfg.DATASET.USE_JSON and getattr(cfg.DATASET, 'AUTO_BUILD', False)):
        return

    root_path = Path(cfg.DATASET.ROOT)
    ratio = cfg.DATASET.TRAIN_SPLIT_RATIO if cfg.DATASET.TRAIN_SPLIT_RATIO is not None else 0.8
    ratio = max(0.0, min(1.0, float(ratio)))
    train_json = root_path / cfg.DATASET.TRAIN_JSON
    val_json = root_path / cfg.DATASET.VAL_JSON
    test_json = root_path / cfg.DATASET.TEST_JSON

    need_build = not train_json.exists() or not val_json.exists()
    if need_build:
        from tools.auto_split_json import generate_splits

        print(
            f"[INFO] Auto building train/val json at {root_path} "
            f"(ratio={ratio}, seed={cfg.DATASET.SPLIT_SEED})"
        )
        generate_splits(root_path, train_json, val_json, ratio, cfg.DATASET.SPLIT_SEED, test_json)

    cfg.DATASET.TRAIN = cfg.DATASET.TRAIN_JSON
    cfg.DATASET.VAL = cfg.DATASET.VAL_JSON
    cfg.DATASET.TEST = cfg.DATASET.TEST_JSON
