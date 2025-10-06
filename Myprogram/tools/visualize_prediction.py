import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Backbone import CapeFormerSegmentation
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
        is_train=False,
        episodes=cfg.DATASET.VAL_EPISODES,
        num_shots=cfg.DATASET.NUM_SHOTS,
        num_queries=cfg.DATASET.NUM_QUERIES,
        img_size=cfg.DATASET.IMG_SIZE,
        mean=cfg.DATASET.MEAN,
        std=cfg.DATASET.STD,
    )
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Run model inference on a single episode and visualize predictions')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output', type=str, required=True, help='Output directory for visualizations')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--ann', type=str, default=None, help='Override annotation file name under dataset root')
    parser.add_argument('--index', type=int, default=0, help='Episode index to visualize')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for masks')
    parser.add_argument('--device', type=str, default=None, help='Computation device (e.g., cuda:0 or cpu)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg.SYSTEM.DEVICE = args.device

    maybe_build_annotations(cfg)

    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu')

    dataset = build_dataset(cfg, args.split, args.ann)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f'Episode index {args.index} out of range (0, {len(dataset) - 1})')

    sample = dataset[args.index]

    support_images, support_masks = ensure_shape(sample['support_images'], sample['support_masks'])
    query_images, query_masks = ensure_shape(sample['query_images'], sample['query_masks'])

    model = CapeFormerSegmentation(cfg)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = (
        checkpoint.get('state_dict')
        or checkpoint.get('model')
        or checkpoint
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        support_batch = support_images.unsqueeze(0).to(device)
        support_mask_batch = support_masks.unsqueeze(0).to(device)
        query_batch = query_images.unsqueeze(0).to(device)
        logits = model(support_batch, support_mask_batch, query_batch)
        preds = torch.sigmoid(logits).cpu()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mean = cfg.DATASET.MEAN
    std = cfg.DATASET.STD

    support_paths = sample['meta'].get('support_paths', [])
    query_paths = sample['meta'].get('query_paths', [])

    save_visualization(
        output_dir,
        'support',
        support_images,
        support_masks,
        support_paths,
        mean=mean,
        std=std,
        threshold=0.5,
        suffix='gt_overlay',
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
        suffix='gt_overlay',
    )

    save_visualization(
        output_dir,
        'query_pred',
        query_images,
        preds[0],
        query_paths,
        mean=mean,
        std=std,
        threshold=args.threshold,
        suffix='pred_overlay',
    )

    with open(output_dir / 'meta.txt', 'w', encoding='utf-8') as f:
        f.write(str(sample['meta']))

    print(f"Episode {args.index} predictions saved to {output_dir}")


if __name__ == '__main__':
    main()
