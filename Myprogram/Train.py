import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from Backbone import CapeFormerSegmentation
from Dataloader.Dataloader import build_dataloader
from tools.train_loop import train_one_epoch, validate_one_epoch, save_checkpoint
from tools.config_utils import load_config, maybe_build_annotations
from utils.logger import create_logger


def parse_args():
    parser = argparse.ArgumentParser(description='MyProgram few-shot part segmentation training')
    parser.add_argument('--config', type=str, default=None, help='外部配置脚本')
    parser.add_argument('--model-dir', type=str, default='./Checkpoints', help='checkpoint 输出目录')
    parser.add_argument('--log-dir', type=str, default='./Logs', help='日志与 TensorBoard 输出目录')
    parser.add_argument('--resume', type=str, default='', help='从 checkpoint 恢复训练')
    parser.add_argument('--device', type=str, default=None, help='运行设备，如 cuda:0')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖配置中的 epoch 数')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=None, help='其他配置项覆盖，如 KEY VALUE')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.epochs:
        cfg.TRAIN.EPOCHS = args.epochs
    if args.device:
        cfg.SYSTEM.DEVICE = args.device
    cfg.TRAIN.CHECKPOINT_DIR = args.model_dir

    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    log_dir = Path(args.log_dir)
    logger = create_logger(log_dir)
    writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard'))

    logger.info(f'Configuration:\n{cfg}')

    maybe_build_annotations(cfg)

    model = CapeFormerSegmentation(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR
    )
    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_iou = 0.0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            logger.info(f'Loading checkpoint from {resume_path}')
            checkpoint = torch.load(resume_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint.get('epoch', 0)
            best_iou = checkpoint.get('best_iou', 0.0)
        else:
            logger.warning(f'Resume file {resume_path} not found, start from scratch!')

    train_loader = build_dataloader(cfg, 'train', True)
    val_loader = build_dataloader(cfg, 'val', False)

    logger.info('Start training...')
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        train_one_epoch(
            cfg, train_loader, model, optimizer, criterion, device, epoch, logger, writer
        )
        scheduler.step()

        if (epoch + 1) % cfg.TRAIN.EVAL_EPOCH_FREQ == 0:
            metrics = validate_one_epoch(
                cfg, val_loader, model, criterion, device, epoch, logger, writer
            )
            current_iou = metrics['iou']
            is_best = current_iou > best_iou
            if is_best:
                best_iou = current_iou

            ckpt_path = save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_iou': best_iou,
                    'cfg': cfg.dump(),
                },
                Path(cfg.TRAIN.CHECKPOINT_DIR),
                is_best,
            )
            logger.info(
                f"Checkpoint saved to {ckpt_path} | best IoU so far: {best_iou:.4f}"
            )

    writer.close()
    logger.info('Training finished.')


if __name__ == '__main__':
    main()
