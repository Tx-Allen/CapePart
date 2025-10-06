import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataloader.Dataloader import FewShotSegmentationDataset
from tools.config_utils import load_config, maybe_build_annotations
from tools.visualize_utils import ensure_shape, save_visualization


def build_dataset(cfg, split: str, force_ann: str | None):
    if split == 'train':
        ann_file = cfg.DATASET.TRAIN
    elif split == 'val':
        ann_file = cfg.DATASET.VAL
    elif split == 'test':
        ann_file = cfg.DATASET.TEST
    else:
        raise ValueError(split)
    if force_ann:
        ann_file = force_ann

    dataset = FewShotSegmentationDataset(
        cfg,
        cfg.DATASET.ROOT,
        ann_file,
        split=split,
        is_train=False,  # 强制顺序访问，避免随机采样
        episodes=cfg.DATASET.VAL_EPISODES,
        num_shots=cfg.DATASET.NUM_SHOTS,
        num_queries=cfg.DATASET.NUM_QUERIES,
        img_size=cfg.DATASET.IMG_SIZE,
        mean=cfg.DATASET.MEAN,
        std=cfg.DATASET.STD,
    )
    return dataset


def save_episode(dataset, index: int, cfg, output_dir: Path):
    sample = dataset[index]
    support_images = sample['support_images']
    support_masks = sample['support_masks']
    query_images = sample['query_images']
    query_masks = sample['query_masks']
    meta = sample['meta']

    output_dir.mkdir(parents=True, exist_ok=True)
    mean = cfg.DATASET.MEAN
    std = cfg.DATASET.STD

    support_images, support_masks = ensure_shape(support_images, support_masks)
    query_images, query_masks = ensure_shape(query_images, query_masks)

    support_paths = meta.get('support_paths', [])
    query_paths = meta.get('query_paths', [])

    save_visualization(
        output_dir,
        'support',
        support_images,
        support_masks,
        support_paths,
        mean=mean,
        std=std,
        threshold=0.5,
        suffix='overlay',
    )
    save_visualization(
        output_dir,
        'query',
        query_images,
        query_masks,
        query_paths,
        mean=mean,
        std=std,
        threshold=0.5,
        suffix='overlay',
    )

    with open(output_dir / 'meta.txt', 'w', encoding='utf-8') as f:
        f.write(str(meta))


def main():
    parser = argparse.ArgumentParser(description='可视化 few-shot episode')
    parser.add_argument('--config', type=str, default=None, help='可选配置文件路径')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--ann', type=str, default=None, help='自定义 JSON 文件名（位于 root 下）')
    parser.add_argument('--index', type=int, default=0, help='Episode 索引')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    args = parser.parse_args()

    cfg = load_config(args.config)
    maybe_build_annotations(cfg)
    dataset = build_dataset(cfg, args.split, args.ann)
    output_dir = Path(args.output)

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f'Episode index {args.index} out of range (0, {len(dataset) - 1})')

    save_episode(dataset, args.index, cfg, output_dir)
    print(f'Episode {args.index} saved to {output_dir}')


if __name__ == '__main__':
    main()
