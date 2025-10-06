"""Multi-scale fusion utilities for CapeFormer."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFeatureFusion(nn.Module):
    """Fuse backbone features from multiple stages into a single map.

    The module projects each stage to a common dimensionality and
    resamples them to the resolution of a reference feature (default: the
    deepest stage). The fused output is the average of the aligned feature
    maps, providing richer context for downstream heads while keeping the
    interface identical to the original single-scale output.
    """

    def __init__(self, in_channels: Dict[str, int], out_channels: int, reference: str | None = None):
        super().__init__()
        if not in_channels:
            raise ValueError('MultiScaleFeatureFusion requires at least one feature map.')

        self.reference = reference or list(in_channels.keys())[-1]
        ordered = OrderedDict(sorted(in_channels.items(), key=lambda item: item[0]))
        self.lateral_convs = nn.ModuleDict({
            name: nn.Conv2d(channels, out_channels, kernel_size=1)
            for name, channels in ordered.items()
        })

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.reference not in features:
            raise KeyError(f'Reference feature "{self.reference}" not found in input keys {list(features.keys())}.')

        reference_feat = features[self.reference]
        ref_hw = reference_feat.shape[-2:]

        fused = None
        for name, conv in self.lateral_convs.items():
            if name not in features:
                continue
            feat = conv(features[name])
            if feat.shape[-2:] != ref_hw:
                feat = F.interpolate(feat, size=ref_hw, mode='bilinear', align_corners=False)
            fused = feat if fused is None else fused + feat

        if fused is None:
            raise RuntimeError('No valid features provided for fusion.')

        fused = fused / len(self.lateral_convs)
        return fused
