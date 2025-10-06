import argparse  # 引入命令行参数解析库，用于自定义评估流程
import torch  # 深度学习张量库，用于模型推理与设备管理

from Backbone import CapeFormerSegmentation  # few-shot 分割模型主体
from Dataloader.Dataloader import build_dataloader  # 构建 episodic 数据加载器
from utils import compute_iou, compute_dice  # 评估指标函数，衡量模型表现
from tools.config_utils import load_config, maybe_build_annotations


# 读取评估所需配置
# 解析评估脚本参数
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CapeFormer segmentation model')
    parser.add_argument('--config', type=str, default=None)  # 指定配置文件路径
    parser.add_argument('--checkpoint', type=str, required=True)  # 待评估的权重文件
    parser.add_argument('--device', type=str, default=None)  # 可覆盖默认设备
    parser.add_argument('--threshold', type=float, default=None)  # IoU/Dice 二值化阈值
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=None)  # 额外覆盖配置项
    return parser.parse_args()


# 评估入口：加载模型并计算指标
def main():
    args = parse_args()  # 解析命令行参数
    cfg = load_config(args.config)  # 根据入参加载配置
    if args.opts:
        cfg.merge_from_list(args.opts)  # 支持命令行覆盖任意配置项
    if args.device:
        cfg.SYSTEM.DEVICE = args.device  # 覆盖设备设置
    if args.threshold:
        cfg.EVAL.THRESHOLD = args.threshold  # 调整评估阈值

    maybe_build_annotations(cfg)

    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu')  # 最终推理设备

    model = CapeFormerSegmentation(cfg)  # 构建模型实例
    ckpt = torch.load(args.checkpoint, map_location='cpu')  # 加载训练好的权重
    state_dict = ckpt.get('model', ckpt)  # 兼容直接存state_dict或完整字典
    model.load_state_dict(state_dict, strict=False)  # 加载模型参数
    model.to(device)  # 移动到目标设备
    model.eval()  # 设置为评估模式（关闭dropout/bn更新）

    loader = build_dataloader(cfg, 'test', False)  # 构造测试集 episode 数据

    iou_scores, dice_scores = [], []  # 持续累积每个 episode 的评估指标

    with torch.no_grad():  # 评估不需要梯度，节省显存
        for batch in loader:  # 遍历所有测试 episode
            support_images = batch['support_images'].to(device)  # 支持图像张量
            support_masks = batch['support_masks'].to(device)  # 支持掩码张量
            query_images = batch['query_images'].to(device)  # 查询图像张量
            query_masks = batch['query_masks'].to(device)  # 查询掩码（监督信号）

            logits = model(support_images, support_masks, query_images)  # 获得预测掩码
            pred_flat = logits.view(-1, logits.shape[-2], logits.shape[-1])  # 展平成[N,H,W]便于计算指标
            target_flat = query_masks.view(-1, query_masks.shape[-2], query_masks.shape[-1])  # 同样展平
            iou = compute_iou(pred_flat, target_flat, threshold=cfg.EVAL.THRESHOLD)  # 计算IoU
            dice = compute_dice(pred_flat, target_flat, threshold=cfg.EVAL.THRESHOLD)  # 计算Dice
            iou_scores.append(iou.mean().item())  # 记录每个episode的平均IoU
            dice_scores.append(dice.mean().item())  # 记录每个episode的平均Dice

    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0  # 汇总平均IoU
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0  # 汇总平均Dice
    print(f'IoU: {mean_iou:.4f}, Dice: {mean_dice:.4f}')  # 输出评估结果


if __name__ == '__main__':  # 脚本入口
    main()  # 执行评估流程
