import torch
import torch.nn as nn
import torch.nn.functional as F

from .capeformer_encoder import CapeFormerBackbone
from .transformer import CapeFormerTransformer, QuerySupportJointEncoder
from .mask_head import CapeFormerMaskHead
from .position_encoding import PositionEmbeddingSine
from .fusion import InformationFusion
from .multiscale import MultiScaleFeatureFusion


class CapeFormerSegmentation(nn.Module):
    """CapeFormer few-shot part segmentation model."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.backbone = CapeFormerBackbone(pretrained=cfg.MODEL.PRETRAINED or None)

        use_multiscale = bool(getattr(cfg.MODEL, 'USE_MULTISCALE_FUSION', False))
        if use_multiscale:
            reference = getattr(cfg.MODEL, 'MULTISCALE_REFERENCE', 'res5')
            self.multiscale_fusion = MultiScaleFeatureFusion(
                self.backbone.feature_channels,
                out_channels=self.backbone.out_channels,
                reference=reference,
            )
        else:
            self.multiscale_fusion = None

        self.input_proj = nn.Conv2d(
            in_channels=self.backbone.out_channels,
            out_channels=cfg.MODEL.D_MODEL,
            kernel_size=1,
        )

        self.transformer = CapeFormerTransformer(
            d_model=cfg.MODEL.D_MODEL,
            nhead=cfg.MODEL.NHEAD,
            num_encoder_layers=cfg.MODEL.NUM_ENCODER_LAYERS,
            num_decoder_layers=cfg.MODEL.NUM_DECODER_LAYERS,
            dim_feedforward=cfg.MODEL.DIM_FEEDFORWARD,
            dropout=cfg.MODEL.DROPOUT,
        )

        default_joint_layers = getattr(cfg.MODEL, 'JOINT_ENCODER_LAYERS', 3)
        joint_layers = int(default_joint_layers) if default_joint_layers else 0
        self.joint_encoder = (
            QuerySupportJointEncoder(
                d_model=cfg.MODEL.D_MODEL,
                nhead=cfg.MODEL.NHEAD,
                num_layers=joint_layers,
                dim_feedforward=cfg.MODEL.DIM_FEEDFORWARD,
                dropout=cfg.MODEL.DROPOUT,
            )
            if joint_layers > 0
            else None
        )

        self.pos_encoder = PositionEmbeddingSine(
            cfg.MODEL.D_MODEL // 2,
            normalize=True,
        )

        self.mask_head = CapeFormerMaskHead(cfg.MODEL.D_MODEL, cfg.MODEL.MASK_DIM)
        use_info_fusion = bool(getattr(cfg.MODEL, 'USE_INFO_FUSION', False))
        self.info_fusion = (
            InformationFusion(cfg.MODEL.D_MODEL) if use_info_fusion else None
        )

    def forward(self, support_images, support_masks, query_images):
        """Predict query masks using support/query pairs.

        Args:
            support_images: Tensor shaped ``[B, S, C, H, W]``.
            support_masks: Tensor shaped ``[B, S, P, H, W]``.
            query_images: Tensor shaped ``[B, Q, C, H, W]``.
        """

        b, s, c, h, w = support_images.shape
        _, q, _, _, _ = query_images.shape
        num_parts = support_masks.shape[2]

        if num_parts == 0:
            raise ValueError('support_masks 的部件通道数为 0，无法计算原型。')

        prototypes = self._compute_prototypes(support_images, support_masks)
        prototypes = prototypes.unsqueeze(1).repeat(1, q, 1, 1)
        prototypes = prototypes.view(b * q, num_parts, -1)

        query_images = query_images.view(b * q, c, h, w)
        query_features = self._extract_features(query_images)
        projected = self.input_proj(query_features)

        pos = self.pos_encoder(projected)

        if self.joint_encoder is not None:
            projected, prototypes = self.joint_encoder(projected, pos, prototypes)
            pos = self.pos_encoder(projected)

        if prototypes.shape[1] != num_parts:
            raise RuntimeError(
                f'Prototype 部件维度与原始掩码不一致: expected {num_parts}, got {prototypes.shape[1]}'
            )

        decoder_out, memory = self.transformer(projected, pos, prototypes)
        if self.info_fusion is not None:
            memory = self.info_fusion(projected, memory)

        masks = self.mask_head(decoder_out, memory)
        masks = F.interpolate(masks, size=(h, w), mode='bilinear', align_corners=False)
        masks = masks.view(b, q, num_parts, h, w)
        return masks

    def _compute_prototypes(self, support_images, support_masks):
        """Compute feature prototypes for each part from the support set."""

        b, s, c, h, w = support_images.shape
        num_parts = support_masks.shape[2]

        support_images = support_images.view(b * s, c, h, w)
        support_features = self._extract_features(support_images)
        support_features = self.input_proj(support_features)

        _, dim, fh, fw = support_features.shape
        support_features = support_features.view(b, s, dim, fh, fw)

        resized_masks = F.interpolate(
            support_masks.view(b * s * num_parts, 1, h, w),
            size=(fh, fw),
            mode='bilinear',
            align_corners=False,
        ).view(b, s, num_parts, fh, fw)

        feature_expand = support_features.unsqueeze(2)
        mask_expand = resized_masks.unsqueeze(3)
        masked = feature_expand * mask_expand
        masked = masked.view(b, s, num_parts, dim, -1).sum(-1)

        denom = resized_masks.view(b, s, num_parts, -1).sum(-1).clamp(min=1e-6)
        protos = masked / denom.unsqueeze(-1)
        protos = protos.mean(1)
        return protos

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract backbone features with optional multi-scale fusion."""

        if self.multiscale_fusion is None:
            return self.backbone(images)

        features = self.backbone(images, return_dict=True)
        return self.multiscale_fusion(features)
