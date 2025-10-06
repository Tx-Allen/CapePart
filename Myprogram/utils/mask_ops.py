import torch

def mask_area(mask: torch.Tensor) -> torch.Tensor:
    """计算掩码的像素面积，mask 形状为 [H, W] 或 [P, H, W]。"""
    if mask.dim() == 3:
        return mask.float().sum(dim=(-1, -2))
    return mask.float().sum()


def is_valid_mask(mask: torch.Tensor, min_area: float) -> bool:
    """判断掩码是否满足最小面积约束。"""
    return mask_area(mask) >= min_area


def random_erase(mask: torch.Tensor, erase_ratio: float = 0.1) -> torch.Tensor:
    """对掩码做随机遮挡，返回同形状张量。"""
    if erase_ratio <= 0:
        return mask
    flat = mask.view(-1)
    num_pixels = flat.numel()
    num_erase = int(num_pixels * erase_ratio)
    if num_erase <= 0:
        return mask
    indices = torch.randperm(num_pixels, device=mask.device)[:num_erase]
    flat[indices] = 0
    return mask
