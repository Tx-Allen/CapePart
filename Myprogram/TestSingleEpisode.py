from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from Backbone import CapeFormerSegmentation
from Dataloader.Dataloader import FewShotSegmentationDataset
from tools.config_utils import load_config


PALETTE: List[tuple[int, int, int]] = [
    (239, 71, 111),
    (255, 209, 102),
    (6, 214, 160),
    (17, 138, 178),
    (7, 59, 76),
    (255, 127, 80),
    (144, 190, 109),
    (67, 170, 139),
    (249, 132, 239),
    (255, 196, 61),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a single-episode inference with custom support/query paths.'
    )
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='训练好的权重文件路径')
    parser.add_argument('--support-image', type=str, required=True, help='Support 图像路径')
    parser.add_argument('--support-mask', type=str, required=True, help='Support 多通道 mask 路径')
    parser.add_argument('--query-image', type=str, required=True, help='Query 图像路径')
    parser.add_argument('--device', type=str, default=None, help='推理设备（如 cuda:0）')
    parser.add_argument('--output-dir', type=str, default='./single_episode_output', help='输出目录')
    parser.add_argument('--threshold', type=float, default=None, help='掩码二值化阈值，默认读取配置')
    parser.add_argument('--alpha', type=float, default=0.6, help='可视化叠加时的颜色透明度 [0, 1]')
    parser.add_argument(
        '--part-names',
        type=str,
        default='',
        help='可选的部件名称，使用逗号分隔，与mask通道一一对应',
    )
    return parser.parse_args()


def build_transforms(cfg):
    """Create image/mask transforms consistent with training configuration."""

    img_size = cfg.DATASET.IMG_SIZE
    mean = cfg.DATASET.MEAN
    std = cfg.DATASET.STD
    image_transform = FewShotSegmentationDataset.build_image_transform(img_size, mean, std)
    mask_transform = FewShotSegmentationDataset.build_mask_transform(img_size)
    return image_transform, mask_transform


def _tensorize_mask(mask_img: Image.Image, mask_transform: transforms.Compose) -> torch.Tensor:
    """Convert a PIL mask into a normalized float tensor."""

    tensor = mask_transform(mask_img).float()
    if tensor.max() > 0:
        tensor = tensor / tensor.max()
    return tensor.squeeze(0)


def load_support(mask_path: Path, image_path: Path, mask_transform: transforms.Compose, image_transform: transforms.Compose):
    """Load support image/masks and return both PIL image and tensors."""

    support_image = Image.open(image_path).convert('RGB')
    support_tensor = image_transform(support_image)
    mask_channels = FewShotSegmentationDataset.decode_multi_channel_mask(str(mask_path))
    part_tensors = [_tensorize_mask(mask_img, mask_transform) for mask_img in mask_channels]
    if not part_tensors:
        raise ValueError(f'在 {mask_path} 中未解析到任何有效的mask通道')
    support_masks = torch.stack(part_tensors)
    return support_image, support_tensor, support_masks


def load_query(image_path: Path, image_transform: transforms.Compose):
    """Load and tensorize the query image."""

    query_image = Image.open(image_path).convert('RGB')
    query_tensor = image_transform(query_image)
    return query_image, query_tensor


def prepare_episode(support_tensor: torch.Tensor, support_masks: torch.Tensor, query_tensor: torch.Tensor):
    """Expand tensors to match the model's `[B, S/Q, ...]` expectations."""

    support_images = support_tensor.unsqueeze(0).unsqueeze(0)
    support_masks = support_masks.unsqueeze(0).unsqueeze(0)
    query_images = query_tensor.unsqueeze(0).unsqueeze(0)
    return support_images, support_masks, query_images


def resolve_part_names(raw: str, count: int) -> List[str]:
    """Resolve user-provided part names and pad/truncate to match channels."""

    names = [item.strip() for item in raw.split(',') if item.strip()]
    if not names:
        return [f'part_{idx}' for idx in range(count)]
    if len(names) != count:
        print(
            f"[WARN] 提供的部件名称数量({len(names)})与mask通道数({count})不一致，自动裁剪或补齐默认名称。"
        )
        names = (names + [f'part_{idx}' for idx in range(count)])[:count]
    return names


def build_palette(num_parts: int) -> List[tuple[int, int, int]]:
    """Repeat a compact color palette until it covers all parts."""

    if num_parts <= len(PALETTE):
        return PALETTE[:num_parts]
    palette = []
    idx = 0
    while len(palette) < num_parts:
        palette.append(PALETTE[idx % len(PALETTE)])
        idx += 1
    return palette


def save_predictions(
    masks: torch.Tensor,
    output_dir: Path,
    part_names: Sequence[str],
    palette: Sequence[tuple[int, int, int]],
    original_image: Image.Image,
    threshold: float,
    alpha: float,
):
    """Persist per-part masks and an overlay visualization to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)

    probs = torch.sigmoid(masks)
    resized_probs = F.interpolate(
        probs.unsqueeze(0),
        size=original_image.size[::-1],
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)

    np.savez_compressed(output_dir / 'pred_masks.npz', masks=resized_probs.cpu().numpy())

    binary = resized_probs > threshold
    base = np.array(original_image.convert('RGB'), dtype=np.float32)

    for idx in range(binary.shape[0]):
        mask_array = (binary[idx].cpu().numpy() * 255).astype(np.uint8)
        mask_img = Image.fromarray(mask_array, mode='L')
        mask_img.save(output_dir / f'{part_names[idx]}_mask.png')

        color = np.array(palette[idx], dtype=np.float32)
        region = binary[idx].cpu().numpy()
        base[region] = (1 - alpha) * base[region] + alpha * color

    overlay = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    overlay.save(output_dir / 'overlay.png')


@torch.no_grad()
def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg.SYSTEM.DEVICE = args.device
    if args.threshold is not None:
        cfg.EVAL.THRESHOLD = args.threshold

    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu')

    model = CapeFormerSegmentation(cfg)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    image_transform, mask_transform = build_transforms(cfg)

    support_image_path = Path(args.support_image)
    support_mask_path = Path(args.support_mask)
    query_image_path = Path(args.query_image)

    _, support_tensor, support_masks = load_support(
        support_mask_path,
        support_image_path,
        mask_transform,
        image_transform,
    )
    query_image, query_tensor = load_query(query_image_path, image_transform)

    support_images, support_masks_tensor, query_images = prepare_episode(
        support_tensor,
        support_masks,
        query_tensor,
    )

    support_images = support_images.to(device)
    support_masks_tensor = support_masks_tensor.to(device)
    query_images = query_images.to(device)

    outputs = model(support_images, support_masks_tensor, query_images)
    preds = outputs[0, 0]

    threshold = float(getattr(cfg.EVAL, 'THRESHOLD', 0.5))
    part_names = resolve_part_names(args.part_names, preds.shape[0])
    palette = build_palette(preds.shape[0])

    save_predictions(
        preds.cpu(),
        Path(args.output_dir),
        part_names,
        palette,
        query_image,
        threshold,
        max(0.0, min(1.0, args.alpha)),
    )

    print(f'完成推理，结果已保存至 {args.output_dir}')


if __name__ == '__main__':
    main()
