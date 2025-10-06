import torch

# 计算预测掩码与真值的IoU
def compute_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    # 对预测掩码做Sigmoid并根据阈值二值化
    pred = (pred_mask.sigmoid() > threshold).float()
    # 规范化真值掩码为0/1
    target = (target_mask > 0.5).float()
    # 计算交集像素点数
    intersection = (pred * target).sum(dim=(-1, -2))
    # 计算并集并加上极小值防止除0
    union = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) - intersection + 1e-6
    return intersection / union

# 计算预测掩码与真值的Dice系数
def compute_dice(pred_mask: torch.Tensor, target_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    # 将预测掩码二值化
    pred = (pred_mask.sigmoid() > threshold).float()
    target = (target_mask > 0.5).float()
    # 交集用于计算Dice分子
    intersection = (pred * target).sum(dim=(-1, -2))
    # 分母为预测与真值像素和，加入极小值防止除0
    denom = pred.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) + 1e-6
    return 2 * intersection / denom
