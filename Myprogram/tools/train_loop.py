import time
from pathlib import Path

import torch

from utils import AverageMeter, compute_iou, compute_dice


def train_one_epoch(cfg, loader, model, optimizer, criterion, device, epoch, logger, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    model.train()
    end = time.time()

    for step, batch in enumerate(loader):
        data_time.update(time.time() - end)

        support_images = batch['support_images'].to(device)
        support_masks = batch['support_masks'].to(device)
        query_images = batch['query_images'].to(device)
        query_masks = batch['query_masks'].to(device)

        if cfg.DEBUG.ENABLE and cfg.DEBUG.PRINT_TENSOR_SHAPES and step == 0:
            logger.info(
                "[DEBUG] support_images %s, support_masks %s, query_images %s, query_masks %s",
                tuple(support_images.shape), tuple(support_masks.shape),
                tuple(query_images.shape), tuple(query_masks.shape)
            )

        optimizer.zero_grad()
        preds = model(support_images, support_masks, query_images)
        loss = criterion(preds, query_masks)

        if cfg.DEBUG.ENABLE and cfg.DEBUG.PRINT_STATS and step == 0:
            logger.info(
                "[DEBUG] preds stats min %.4f max %.4f mean %.4f",
                preds.min().item(), preds.max().item(), preds.mean().item()
            )

        loss.backward()
        if cfg.TRAIN.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
        optimizer.step()

        loss_meter.update(loss.item(), query_images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if step % cfg.TRAIN.LOG_INTERVAL == 0:
            msg = (
                f"Epoch [{epoch}] Step [{step}/{len(loader)}] "
                f"Loss {loss_meter.val:.4f} (avg {loss_meter.avg:.4f}) "
                f"Data {data_time.val:.3f}s Batch {batch_time.val:.3f}s"
            )
            logger.info(msg)
        if writer is not None:
            writer.add_scalar('train/loss', loss.item(), epoch * len(loader) + step)

    logger.info(
        f"Epoch [{epoch}] training done. Avg loss {loss_meter.avg:.4f}, "
        f"avg batch time {batch_time.avg:.3f}s, avg data time {data_time.avg:.3f}s"
    )


@torch.no_grad()
def validate_one_epoch(cfg, loader, model, criterion, device, epoch, logger, writer=None):
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()
    dice_meter = AverageMeter()

    model.eval()
    for step, batch in enumerate(loader):
        support_images = batch['support_images'].to(device)
        support_masks = batch['support_masks'].to(device)
        query_images = batch['query_images'].to(device)
        query_masks = batch['query_masks'].to(device)

        if cfg.DEBUG.ENABLE and cfg.DEBUG.PRINT_TENSOR_SHAPES and step == 0:
            logger.info(
                "[DEBUG][VAL] support %s, masks %s, query %s, gt %s",
                tuple(support_images.shape), tuple(support_masks.shape),
                tuple(query_images.shape), tuple(query_masks.shape)
            )

        preds = model(support_images, support_masks, query_images)
        loss = criterion(preds, query_masks)
        loss_meter.update(loss.item(), query_images.size(0))

        pred_flat = preds.view(-1, preds.shape[-2], preds.shape[-1])
        gt_flat = query_masks.view(-1, query_masks.shape[-2], query_masks.shape[-1])
        iou = compute_iou(pred_flat, gt_flat, threshold=cfg.EVAL.THRESHOLD)
        dice = compute_dice(pred_flat, gt_flat, threshold=cfg.EVAL.THRESHOLD)
        iou_meter.update(iou.mean().item(), query_images.size(0))
        dice_meter.update(dice.mean().item(), query_images.size(0))

        if step % cfg.TRAIN.LOG_INTERVAL == 0:
            logger.info(
                f"Val Epoch [{epoch}] Step [{step}/{len(loader)}] "
                f"Loss {loss_meter.val:.4f} IoU {iou_meter.val:.4f} Dice {dice_meter.val:.4f}"
            )

    logger.info(
        f"Validation Epoch [{epoch}] -> loss {loss_meter.avg:.4f}, "
        f"IoU {iou_meter.avg:.4f}, Dice {dice_meter.avg:.4f}"
    )
    if writer is not None:
        writer.add_scalar('val/loss', loss_meter.avg, epoch)
        writer.add_scalar('val/iou', iou_meter.avg, epoch)
        writer.add_scalar('val/dice', dice_meter.avg, epoch)

    return {
        'loss': loss_meter.avg,
        'iou': iou_meter.avg,
        'dice': dice_meter.avg,
    }


def save_checkpoint(state: dict, out_dir: Path, is_best: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = 'best' if is_best else f"epoch_{state['epoch']}"
    path = out_dir / f'checkpoint_{tag}.pth'
    torch.save(state, path)
    return path
