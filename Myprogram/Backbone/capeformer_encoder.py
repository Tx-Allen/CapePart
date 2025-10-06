from typing import Dict

import torch
import torch.nn as nn
from torchvision.models import resnet50


class CapeFormerBackbone(nn.Module):
    """ResNet-50 backbone compatible with CapeFormer checkpoints."""

    # 初始化ResNet骨干，支持加载CapeFormer预训练权重
    def __init__(self, pretrained: str | None = None):
        super().__init__()
        # 创建基础的ResNet50模型，保持与CapeFormer结构一致
        self.body = resnet50(weights=None)

        if pretrained:
            # 若配置提供预训练文件，则在CPU加载权重避免显存占用
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                # 某些权重文件将真实参数存放在state_dict键下，需要取出
                state_dict = state_dict['state_dict']
            # 消除DataParallel等带来的前缀，方便与当前模型对齐
            cleaned = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state_dict.items()}
            # 加载权重并反馈缺失/多余键，便于检查兼容性
            missing, unexpected = self.body.load_state_dict(cleaned, strict=False)
            if missing:
                print(f'[CapeFormerBackbone] missing keys: {missing}')
            if unexpected:
                print(f'[CapeFormerBackbone] unexpected keys: {unexpected}')

        # 构建四阶段ResNet结构，用于提取逐层特征
        self.stem = nn.Sequential(
            self.body.conv1,  # 初始卷积层，提取低层纹理
            self.body.bn1,    # 归一化保证数值稳定
            self.body.relu,   # 非线性激活增强表达能力
            self.body.maxpool,  # 最大池化降低特征图分辨率
        )
        self.layer1 = self.body.layer1  # 第一层残差块
        self.layer2 = self.body.layer2  # 第二层残差块
        self.layer3 = self.body.layer3  # 第三层残差块
        self.layer4 = self.body.layer4  # 第四层残差块

        # CapeFormer期望的输出通道数
        self.out_channels = 2048
        self.feature_channels: Dict[str, int] = {
            'res2': 256,
            'res3': 512,
            'res4': 1024,
            'res5': 2048,
        }

    # 前向传播：提取图像特征图
    def forward(self, x: torch.Tensor, *, return_dict: bool = False):
        # 通过stem模块获取基础特征
        x = self.stem(x)
        # 依次经过四个残差阶段，累积语义信息
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        if return_dict:
            return {
                'res2': c2,
                'res3': c3,
                'res4': c4,
                'res5': c5,
            }
        # 返回最终高语义特征图
        return c5
