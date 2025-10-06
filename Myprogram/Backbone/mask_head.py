import torch  # 引入张量运算库，用于掩码计算与einsum
import torch.nn as nn  # 提供神经网络模块基类与层实现


class CapeFormerMaskHead(nn.Module):
    """将Transformer输出的token特征解码成像素级掩码。"""

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        # 1×1卷积：把Transformer记忆特征从in_channels压缩到hidden_dim，便于与token向量做内积
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        # 两层线性层 + ReLU：把每个decoder token映射成掩码查询向量
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 第1层线性变换，提升组合能力
            nn.ReLU(inplace=True),  # 非线性激活，增加表达能力，避免线性可分限制
            nn.Linear(hidden_dim, hidden_dim),  # 第2层线性变换，输出与特征图同维度的查询向量
        )

    def forward(self, decoder_output: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """根据解码器输出与记忆特征生成掩码。

        Args:
            decoder_output: [B, Q, C]  # B批次、Q部件token、C隐藏维度
            memory: [B, C, H, W]  # Transformer编码后的空间特征
        Returns:
            [B, Q, H, W]  # 每个token对应的像素掩码
        """
        # 通过1×1卷积把空间特征映射到hidden_dim，与token向量处于同一特征空间
        mask_features = self.proj(memory)
        # 将decoder的每个token投影到掩码空间，得到查询向量
        mask_embed = self.mask_embed(decoder_output)
        # einsum实现按token对特征图逐像素加权：token向量 · 每个像素的特征=掩码数值
        masks = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)
        # 输出仍为B×部件×H×W张量，供上游插值或阈值处理
        return masks
