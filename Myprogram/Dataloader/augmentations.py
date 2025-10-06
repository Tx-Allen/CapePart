"""Episode-level augmentation helpers."""

from __future__ import annotations

import random
from typing import List

import torch
from PIL import Image

from utils import mask_area, random_erase


class EpisodeAugmentor:
    """Bundle augmentation and mask post-processing logic."""

    def __init__(
        self,
        *,
        is_train: bool,
        color_jitter=None,
        flip_prob: float = 0.0,
        mask_dropout_prob: float = 0.0,
        min_mask_pixels: int = 0,
        mask_erase_ratio: float = 0.0,
    ) -> None:
        self.is_train = is_train
        self.color_jitter = color_jitter
        self.flip_prob = float(flip_prob)
        self.mask_dropout_prob = float(mask_dropout_prob)
        self.min_mask_pixels = int(max(min_mask_pixels, 0))
        self.mask_erase_ratio = float(mask_erase_ratio)

    def apply(self, image: Image.Image, masks: List[Image.Image], *, is_support: bool):
        if not self.is_train:
            return image, masks

        aug_img = image
        aug_masks = masks
        if self.color_jitter is not None:
            aug_img = self.color_jitter(aug_img)
        if random.random() < self.flip_prob:
            aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
            aug_masks = [mask.transpose(Image.FLIP_LEFT_RIGHT) for mask in aug_masks]
        return aug_img, aug_masks

    def refine_mask_tensor(self, tensor: torch.Tensor, *, is_support: bool) -> torch.Tensor:
        if not self.is_train:
            return tensor

        refined = tensor
        if self.min_mask_pixels > 0 and mask_area(refined) < self.min_mask_pixels:
            refined = torch.zeros_like(refined)
        if is_support and self.mask_erase_ratio > 0:
            refined = random_erase(refined.clone(), self.mask_erase_ratio)
        return refined

    def apply_dropout(self, parts: torch.Tensor) -> torch.Tensor:
        if not self.is_train or self.mask_dropout_prob <= 0:
            return parts
        dropout_mask = torch.rand(parts.size(0)) < self.mask_dropout_prob
        if dropout_mask.any():
            parts = parts.clone()
            parts[dropout_mask] = 0
        return parts
