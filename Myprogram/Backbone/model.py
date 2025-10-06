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
    # 初始化CapeFormer分割模型的各个模块
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg # 保存配置

        # Backbone
        self.backbone = CapeFormerBackbone( #构建骨干网络
            pretrained=cfg.MODEL.PRETRAINED or None
            ) # 如果有预训练权重路径则加载，否则为 None
        self.multiscale_fusion = None
        if bool(getattr(cfg.MODEL, 'USE_MULTISCALE_FUSION', False)):
            reference = getattr(cfg.MODEL, 'MULTISCALE_REFERENCE', 'res5')
            self.multiscale_fusion = MultiScaleFeatureFusion(
                self.backbone.feature_channels,
                out_channels=self.backbone.out_channels,
                reference=reference,
            )
        
        # 1x1卷积将骨干输出映射到Transformer所需维度
        self.input_proj = nn.Conv2d( # 卷积：通道映射
            self.backbone.out_channels, # 输入通道=骨干输出通道
            cfg.MODEL.D_MODEL, # 输出通道=Transformer 的 d_model
            kernel_size=1 #卷积核大小1x1
            )
        
        # 初始化Transformer，用CapeFormer参数
        self.transformer = CapeFormerTransformer( # 初始化Transformer
            d_model=cfg.MODEL.D_MODEL, # Transformer token/特征的通道维
            nhead=cfg.MODEL.NHEAD, # 多头注意力头数
            num_encoder_layers=cfg.MODEL.NUM_ENCODER_LAYERS, # 编码器层数
            num_decoder_layers=cfg.MODEL.NUM_DECODER_LAYERS, # 解码器层数（解码 P 个“部件原型”）
            dim_feedforward=cfg.MODEL.DIM_FEEDFORWARD, # 前馈网络隐藏层维度
            dropout=cfg.MODEL.DROPOUT, # dropout概率
        )

        joint_layers = getattr(cfg.MODEL, 'JOINT_ENCODER_LAYERS', 0) # 是否启用联合精炼层

        # 联合编码器模块
        self.joint_encoder = (
            QuerySupportJointEncoder( 
                d_model=cfg.MODEL.D_MODEL, # 维度同 Transformer
                nhead=cfg.MODEL.NHEAD, # 头数同 Transformer
                num_layers=joint_layers, # 联合编码层数
                dim_feedforward=cfg.MODEL.DIM_FEEDFORWARD, # 前馈维度同 Transformer
                dropout=cfg.MODEL.DROPOUT, # dropout 同 Transformer
            )
            if joint_layers and joint_layers > 0 # 启用条件：层数大于0
            else None # 否则不启用
        )

        # 生成正弦位置编码，为特征提供空间参考
        self.pos_encoder = PositionEmbeddingSine(
            cfg.MODEL.D_MODEL // 2, # 位置编码维度为 d_model 的一半, 二维（x、y）正弦位置编码的总通道数 = 2 × num_pos_feats, 才能使最终位置编码通道数 = d_model，与主干投影后的特征对齐。
            normalize=True # 归一化到 [0, 2pi] 范围, 可减少分辨率依赖、稳定训练，并让最低频分量正好覆盖一个完整正弦/余弦周期。
            )
        
        # 掩码预测头，将Transformer输出映射到像素掩码
        self.mask_head = CapeFormerMaskHead(cfg.MODEL.D_MODEL, cfg.MODEL.MASK_DIM)
        self.info_fusion = InformationFusion(cfg.MODEL.D_MODEL) if getattr(cfg.MODEL, 'USE_INFO_FUSION', False) else None

    # 前向计算：使用Support集生成原型并预测查询掩码, 编排数据流
    def forward(self, support_images, support_masks, query_images):
        '''
        support_images: [B, S, C, H, W]
        support_masks: [B, S, P, H, W]
        query_images: [B, Q, C, H, W]
        '''
        b, s, c, h, w = support_images.shape # 读取批量(B)、shots(S)、通道(C)、高宽(H,W)
        _, q, _, _, _ = query_images.shape # 读取查询张数 Q（B/C/H/W 与 support 对齐的约定）
        num_parts = support_masks.shape[2] # P：部件数量（mask 的通道/类别数）

        if num_parts == 0:
            raise ValueError('support_masks 的部件通道数为 0，无法计算原型。')

        # 利用Support集计算每个part的原型
        prototypes = self._compute_prototypes( # 计算每个部件的“特征原型”
            support_images, support_masks # 输出形状通常为 [B, P, d_model]
        )

        # 为每个查询样本复制原型，并调整为Transformer期望形状
        prototypes = (
            prototypes.unsqueeze(1) # 在第1维扩展以匹配查询数
            .repeat(1, q, 1, 1) # 复制 q 份原型
            .view(b * q, num_parts, -1) # 调整形状为 [B*Q, P, d_model] 
        )

        # 展平Query图像并提取骨干特征
        query_images = query_images.view(b * q, c, h, w) # 将 [B,Q,C,H,W] 展成 [B·Q,C,H,W] 以便送入 CNN
        query_features = self._extract_features(query_images) # Backbone 提取特征 → [B·Q, C_back, H', W']
        projected = self.input_proj(query_features) # 1x1卷积映射到 d_model → [B·Q, d_model, H', W']
        
        # 生成位置编码
        pos = self.pos_encoder(projected)

        # 如果开启了联合编码
        if self.joint_encoder is not None:
            projected, prototypes = self.joint_encoder(projected, pos, prototypes) # 精炼查询特征和原型
            pos = self.pos_encoder(projected)  # 精炼后重新生成位置编码以保持对齐
        if prototypes.shape[1] != num_parts:
            raise RuntimeError(
                f'Prototype 部件维度与原始掩码不一致: expected {num_parts}, got {prototypes.shape[1]}'
            )

        # Transformer输出decoder token及记忆特征
        decoder_out, memory = self.transformer(projected, pos, prototypes)
        if self.info_fusion is not None:
            memory = self.info_fusion(projected, memory)
        # 掩码头生成稠密掩码
        masks = self.mask_head(decoder_out, memory)
        masks = F.interpolate(masks, size=(h, w), mode='bilinear', align_corners=False)
        # 恢复批次、查询与part维度
        masks = masks.view(b, q, num_parts, h, w)
        return masks

    # 内部函数：根据支持集生成类别原型向量
    def _compute_prototypes(self, support_images, support_masks):
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

        # 匹配维度后进行掩码加权求和
        feature_expand = support_features.unsqueeze(2)  # [b, s, 1, dim, fh, fw]
        mask_expand = resized_masks.unsqueeze(3)        # [b, s, p, 1, fh, fw]
        masked = feature_expand * mask_expand           # [b, s, p, dim, fh, fw]
        masked = masked.view(b, s, num_parts, dim, -1).sum(-1)

        denom = resized_masks.view(b, s, num_parts, -1).sum(-1).clamp(min=1e-6)
        protos = masked / denom.unsqueeze(-1)
        protos = protos.mean(1)
        return protos

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.multiscale_fusion is None:
            return self.backbone(images)
        features = self.backbone(images, return_dict=True)
        return self.multiscale_fusion(features)
