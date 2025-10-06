"""Shared helpers for episode visualization scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def unnormalize(img_tensor: torch.Tensor, mean, std) -> np.ndarray:
    img = img_tensor.clone().cpu().numpy()
    for channel, (mu, sigma) in enumerate(zip(mean, std)):
        img[channel] = img[channel] * sigma + mu
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    return np.transpose(img, (1, 2, 0))


def color_palette(num_parts: int):
    base_colors = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 128, 255),
    ]
    colors = []
    for idx in range(num_parts):
        if idx < len(base_colors):
            colors.append(base_colors[idx])
        else:
            rng = np.random.default_rng(idx)
            colors.append(tuple(int(value) for value in rng.integers(0, 255, size=3)))
    return colors


def overlay_masks(
    image: np.ndarray,
    masks: torch.Tensor,
    *,
    alpha: float = 0.5,
    threshold: float = 0.5,
) -> Image.Image:
    colored = image.copy().astype(np.float32)
    masks_np = masks.detach().cpu().numpy()
    num_parts = masks_np.shape[0]
    colors = color_palette(num_parts)

    for part_idx in range(1, num_parts):  # skip background
        mask = masks_np[part_idx] > threshold
        if not mask.any():
            continue
        color = np.array(colors[part_idx], dtype=np.float32)
        colored[mask] = colored[mask] * (1 - alpha) + color * alpha

    return Image.fromarray(colored.astype(np.uint8))


def ensure_shape(images: torch.Tensor, masks: torch.Tensor):
    if images.dim() == 5:
        img_tensor = images[0]
    elif images.dim() == 4:
        img_tensor = images
    else:
        raise ValueError(f'Unexpected image tensor shape: {images.shape}')

    if masks.dim() == 5:
        mask_tensor = masks[0]
    elif masks.dim() == 4:
        mask_tensor = masks
    else:
        raise ValueError(f'Unexpected mask tensor shape: {masks.shape}')

    return img_tensor, mask_tensor


def save_visualization(
    output_dir: Path,
    prefix: str,
    images: torch.Tensor,
    masks: torch.Tensor,
    paths,
    *,
    mean,
    std,
    threshold: float,
    suffix: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    shots = images.shape[0]
    for shot_idx in range(shots):
        img_np = unnormalize(images[shot_idx], mean, std)
        overlay = overlay_masks(img_np, masks[shot_idx], threshold=threshold)
        stem = Path(paths[shot_idx]).stem if paths and shot_idx < len(paths) else f'{prefix}_{shot_idx}'
        Image.fromarray(img_np).save(output_dir / f'{prefix}_{stem}_image.png')
        overlay.save(output_dir / f'{prefix}_{stem}_{suffix}.png')
