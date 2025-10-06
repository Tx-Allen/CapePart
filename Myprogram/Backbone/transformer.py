import math  # 提供正弦位置编码所需的数学函数

import torch
import torch.nn as nn


class CapeFormerTransformer(nn.Module):
    # 初始化CapeFormer所需的Transformer结构
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Transformer Encoder，处理查询图像特征
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Transformer Decoder，处理支持集提示信息
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        # 记录embedding维度，供后续模块查询
        self.d_model = d_model

    # 前向计算：编码特征并解码查询提示
    def forward(self, src, pos_embed, query_embed, key_padding_mask=None):
        # 记录输入的维度信息
        b, c, h, w = src.shape
        # 将特征图展平到[HW, B, C]形状以适配官方实现
        src_flat = src.flatten(2).permute(2, 0, 1)
        pos_flat = pos_embed.flatten(2).permute(2, 0, 1)

        # 编码阶段融合位置编码，得到全局表征
        memory = self.encoder(src_flat + pos_flat, mask=None)

        # 将查询向量转成[num_query, B, C]以匹配PyTorch接口
        query = query_embed.permute(1, 0, 2)
        tgt = torch.zeros_like(query)  # 目标初始化为零向量
        # 解码阶段将支持集提示与编码记忆进行交互
        hs = self.decoder(tgt + query, memory, memory_key_padding_mask=key_padding_mask)

        # 将编码记忆还原成[B, C, H, W]供后续掩码预测使用
        memory = memory.permute(1, 2, 0).view(b, c, h, w)
        # 调整decoder输出为[B, Q, C]
        hs = hs.transpose(0, 1)
        return hs, memory


class QuerySupportJointEncoder(nn.Module):
    """Query-Support联合精炼编码器"""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 使用标准TransformerEncoder层同时建模query像素与support token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model  # 记录特征维度，便于生成1D位置编码

    # 构造1D正弦位置编码，用于区别不同support部件
    def _build_support_pos(self, num_parts: int, batch_size: int, device, dtype):
        if num_parts == 0:
            return torch.zeros(0, batch_size, self.d_model, device=device, dtype=dtype)
        positions = torch.arange(num_parts, device=device, dtype=dtype).unsqueeze(1)  # [P,1]
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype) *
            (-math.log(10000.0) / max(self.d_model, 1))
        )  # [d_model/2]
        sinusoid = positions * div_term  # [P, d_model/2]
        pos = torch.zeros(num_parts, self.d_model, device=device, dtype=dtype)
        pos[:, 0::2] = torch.sin(sinusoid)
        pos[:, 1::2] = torch.cos(sinusoid)
        return pos.unsqueeze(1).repeat(1, batch_size, 1)  # [P, B, C]

    # 前向：将query特征与support token拼接后联合精炼
    def forward(self, query_feats: torch.Tensor, pos_embed: torch.Tensor, support_tokens: torch.Tensor):
        """
            query_feats: [B, C, H, W] 查询图像的空间特征
            pos_embed: [B, C, H, W] 查询特征对应的位置编码
            support_tokens: [B, P, C] 支持集生成的部件原型
            refined_feats: [B, C, H, W] 精炼后的查询特征
            refined_tokens: [B, P, C] 精炼后的部件token
        """
        b, c, h, w = query_feats.shape
        hw = h * w
        dtype = query_feats.dtype
        device = query_feats.device

        query_seq = query_feats.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        pos_seq = pos_embed.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        query_seq = query_seq + pos_seq  # 注入空间位置信息

        num_parts = support_tokens.shape[1]
        support_seq = support_tokens.permute(1, 0, 2)  # [P, B, C]
        support_pos = self._build_support_pos(num_parts, b, device, dtype)
        support_seq = support_seq + support_pos

        fused = torch.cat([query_seq, support_seq], dim=0)  # [(HW+P), B, C]
        refined = self.encoder(fused)

        refined_query = refined[:hw].permute(1, 2, 0).view(b, c, h, w)
        refined_support = refined[hw:].permute(1, 0, 2)  # [B, P, C]
        return refined_query, refined_support
