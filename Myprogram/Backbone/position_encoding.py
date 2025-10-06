import math
import torch
import torch.nn as nn


class PositionEmbeddingSine(nn.Module):
    """Sine-cosine positional embeddings (DETR-style)."""

    # 初始化正弦位置编码超参数
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()

        # 记录特征维度与温度系数，用于后续计算
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize

        # 当需要归一化时决定缩放范围
        if scale is not None and normalize is False:
            raise ValueError('when normalize is False scale should be None')
        self.scale = 2 * math.pi if scale is None else scale

    # 前向计算：生成与输入尺寸匹配的位置编码
    def forward(self, tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:

        # 如果未提供mask，则构造全有效像素的掩码
        if mask is None:
            mask = torch.zeros((tensor.shape[0], tensor.shape[2], tensor.shape[3]), dtype=torch.bool, device=tensor.device)
        
        # 取反得到有效区域，后续累积坐标
        not_mask = ~mask
        
        # 分别计算纵向、横向的累计坐标
        y_embed = not_mask.cumsum(1, dtype=tensor.dtype)
        x_embed = not_mask.cumsum(2, dtype=tensor.dtype)
        
        if self.normalize:
            # 归一化到指定尺度，保证不同尺寸图像一致
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # 依据特征维度生成频率序列
        dim_t = torch.arange(self.num_pos_feats, dtype=tensor.dtype, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 计算正弦编码，并沿最后一维拼接
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        
        # 拼合纵横方向并调整为[B, C, H, W]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
