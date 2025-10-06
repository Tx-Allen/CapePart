import torch
import torch.nn as nn


class InformationFusion(nn.Module):
    """融合编码器前后的多尺度特征，增强掩码预测的上下文信息。"""

    def __init__(self, channels: int):
        super().__init__()
        # 轻量级瓶颈结构：先对输入进行1x1投影，再通过3x3卷积混合信息
        self.query_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.memory_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self.norm = nn.GroupNorm(32, channels)

    def forward(self, query_feat: torch.Tensor, memory_feat: torch.Tensor) -> torch.Tensor:
        """将transformer前的查询特征与编码后的记忆特征进行融合。

        Args:
            query_feat: [B, C, H, W] Transformer输入的查询特征（可为联合精炼后的结果）
            memory_feat: [B, C, H, W] Transformer编码器输出的记忆特征
        Returns:
            [B, C, H, W] 融合后的记忆表示
        """
        query_emb = self.query_proj(query_feat)
        memory_emb = self.memory_proj(memory_feat)
        fused = torch.cat([query_emb, memory_emb], dim=1)
        out = self.fuse(fused)
        out = self.norm(out + memory_emb)  # 残差连接保持稳定
        return out
