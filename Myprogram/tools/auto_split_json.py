import argparse
import json
import random
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.episode_utils import (  # noqa: E402
    collect_samples,
    group_samples_by_subclass,
    samples_to_episodes,
)


def split_samples(samples, ratio: float, seed: int):
    grouped = group_samples_by_subclass(samples)
    train_samples = []
    val_samples = []
    rng = random.Random(seed)
    for key, entries in grouped.items():
        entries_sorted = sorted(entries, key=lambda e: e.stem)
        rng.shuffle(entries_sorted)
        if len(entries_sorted) <= 1:
            train_samples.extend(entries_sorted)
            continue
        split_idx = int(round(len(entries_sorted) * ratio))
        split_idx = max(1, min(split_idx, len(entries_sorted) - 1))
        train_samples.extend(entries_sorted[:split_idx])
        val_samples.extend(entries_sorted[split_idx:])
    return train_samples, val_samples


def write_json(path: Path, episodes):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)
    print(f'Wrote {len(episodes)} episodes to {path}')


def generate_splits(root: Path, train_path: Path, val_path: Path, ratio: float, seed: int,
                    test_path: Path | None = None) -> Tuple[int, int]:
    samples = collect_samples(root)
    train_samples, val_samples = split_samples(samples, ratio, seed)
    train_eps = samples_to_episodes(train_samples)
    val_eps = samples_to_episodes(val_samples)
    write_json(train_path, train_eps)
    write_json(val_path, val_eps)
    if test_path is not None:
        write_json(test_path, val_eps)
    return len(train_eps), len(val_eps)


def main():
    parser = argparse.ArgumentParser(description='按子类均匀拆分生成 train/val JSON')
    parser.add_argument('--root', required=True, help='数据根目录')
    parser.add_argument('--train', required=True, help='train JSON 输出路径')
    parser.add_argument('--val', required=True, help='val JSON 输出路径')
    parser.add_argument('--ratio', type=float, default=0.8, help='train 占比 (0-1)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--test', type=str, default='', help='可选 test JSON 输出路径（默认与val相同）')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(root)

    test_path = Path(args.test) if args.test else None
    generate_splits(root, Path(args.train), Path(args.val), args.ratio, args.seed, test_path)


if __name__ == '__main__':
    main()
