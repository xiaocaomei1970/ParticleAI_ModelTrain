"""Flow Field 训练主脚本"""
import csv
import json
import os
import sys
import time
import argparse
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .flow_dataset import FlowDataset
from .flow_head import FlowModel
from .flow_loss import FlowLoss
from .eval_masks import evaluate_batch_v1, evaluate_batch_v1_labels


def _check_flow_dir(flow_dir, img_dir, split_name):
    """验证 flow 目录存在且包含 .npy 文件，否则给出明确的错误提示。"""
    if not os.path.isdir(flow_dir):
        print(f'\n{"=" * 60}')
        print(f'  ERROR: Flow directory not found: {flow_dir}')
        print(f'  Run convert_labels_to_flows first to generate GT flow fields:')
        print(f'    python -m model_flow.data.convert_labels_to_flows \\')
        print(f'        --img-dir {img_dir} \\')
        print(f'        --label-dir {img_dir} \\')
        print(f'        --out-dir {flow_dir}')
        print(f'{"=" * 60}\n')
        sys.exit(1)

    npy_count = len([f for f in os.listdir(flow_dir) if f.endswith('.npy')])
    if npy_count == 0:
        print(f'\n{"=" * 60}')
        print(f'  ERROR: No .npy files in {flow_dir}')
        print(f'  The directory exists but is empty. Run convert_labels_to_flows:')
        print(f'    python -m model_flow.data.convert_labels_to_flows \\')
        print(f'        --img-dir {img_dir} \\')
        print(f'        --label-dir {img_dir} \\')
        print(f'        --out-dir {flow_dir}')
        print(f'{"=" * 60}\n')
        sys.exit(1)


def _fast_forward_scheduler(scheduler, current_step):
    """将 scheduler 推进到已完成的 optimizer step，屏蔽恢复阶段的 PyTorch 顺序警告。"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`')
        for _ in range(current_step):
            scheduler.step()


def _build_train_loader(train_dataset, cfg, batch_size):
    """构建训练 DataLoader。解冻后可用更小 batch 重建以降低显存峰值。"""
    return DataLoader(train_dataset, batch_size=batch_size,
                      shuffle=True, num_workers=cfg.num_workers,
                      pin_memory=True)


def _val_collate(batch):
    """val DataLoader collate：images/flows 正常 stack，labels 和 resize_info 保持 list。"""
    import torch
    images = torch.stack([item[0] for item in batch])
    flows = torch.stack([item[1] for item in batch])
    if len(batch[0]) > 2:
        labels = [item[2] for item in batch]
        resize_infos = [item[3] for item in batch]
    else:
        labels = []
        resize_infos = []
    return images, flows, labels, resize_infos


def _estimate_total_train_steps(train_dataset, cfg):
    """按动态 batch 策略估算完整训练总 step 数，用于 LR schedule。"""
    stage1_epochs = max(cfg.unfreeze_backbone_after_epoch - 1, 0)
    stage2_epochs = max(cfg.max_epochs - cfg.unfreeze_backbone_after_epoch + 1, 0)
    stage1_steps = int(np.ceil(len(train_dataset) / cfg.batch_size))
    stage2_batch = getattr(cfg, 'unfreeze_batch_size', cfg.batch_size)
    stage2_steps = int(np.ceil(len(train_dataset) / stage2_batch))
    return max(stage1_epochs * stage1_steps + stage2_epochs * stage2_steps, 1)


def _build_optimizer_scheduler(model, cfg, total_steps, current_step=0,
                              backbone_unfrozen=False):
    """构建 optimizer 和 scheduler（P0-3 重构）。

    根据 backbone_unfrozen 决定 backbone 参数组的 lr:
      - 未解冻: backbone lr=0（冻结阶段，参数 requires_grad 已为 False，此组无实际更新）
      - 已解冻: backbone lr = base_lr * unfreeze_backbone_lr_ratio

    给每个 param group 添加 'name' 字段用于日志和调试。
    返回 (optimizer, scheduler)。
    """
    neck_head_params = [
        p for p in list(model.neck.parameters()) + list(model.flow_head.parameters())
        if p.requires_grad
    ]

    groups = [{
        'name': 'neck_head',
        'params': neck_head_params,
        'lr': cfg.base_lr,
        'weight_decay': cfg.weight_decay,
    }]

    if backbone_unfrozen:
        backbone_params = [p for p in model.backbone.parameters()
                           if p.requires_grad]
        bb_lr = cfg.base_lr * cfg.unfreeze_backbone_lr_ratio
        groups.append({
            'name': 'backbone',
            'params': backbone_params,
            'lr': bb_lr,
            'weight_decay': cfg.weight_decay,
        })
    elif cfg.freeze_backbone:
        # 冻结阶段 - backbone 参数 requires_grad=False，此组无实际更新
        groups.append({
            'name': 'backbone',
            'params': model.backbone.parameters(),
            'lr': 0.0,
            'weight_decay': cfg.weight_decay,
        })

    optimizer = torch.optim.AdamW(groups)

    # 构建 CosineAnnealing + Warmup 调度器
    min_lr = cfg.base_lr * cfg.min_lr_ratio if hasattr(cfg, 'min_lr_ratio') else 0.0

    # 保护小数据集: 确保 warmup < total_steps, 且 Cosine T_max >= 1
    if total_steps <= cfg.warmup_iters:
        actual_warmup = max(1, total_steps // 2)
        cosine_t_max = max(1, total_steps - actual_warmup)
        print(f"  WARNING: total_steps ({total_steps}) <= warmup_iters "
              f"({cfg.warmup_iters}); "
              f"using actual_warmup={actual_warmup}, cosine_T_max={cosine_t_max}")
    else:
        actual_warmup = cfg.warmup_iters
        cosine_t_max = total_steps - actual_warmup

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0,
        total_iters=actual_warmup)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_t_max,
        eta_min=min_lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine],
        milestones=[actual_warmup])

    # 将 scheduler 步进到 current_step（用于恢复/重建时）
    _fast_forward_scheduler(scheduler, current_step)

    return optimizer, scheduler


def _print_lr(optimizer, batch_idx, train_loader_len, epoch, cfg, loss,
              flow_l, cell_l):
    """打印训练日志，包含各 param group 的 LR（P0-3 改进）。"""
    parts = []
    for g in optimizer.param_groups:
        name = g.get('name', '?')
        parts.append(f'{name}={g["lr"]:.6f}')
    lr_str = ' '.join(parts)
    print(f'Epoch [{epoch}/{cfg.max_epochs}] '
          f'Batch [{batch_idx}/{train_loader_len}] '
          f'Loss: {loss:.4f} '
          f'flow={flow_l:.4f} cellprob={cell_l:.4f} '
          f'LR: {lr_str}')


# V1 训练指标字段。best checkpoint 以 mask_instance_f1（来自真实 GT labels）为主判据。
# mask_proxy_* 字段为 flow-derived 诊断对照项，不用作模型选择。
METRIC_FIELDS = [
    'epoch',
    'global_step',
    'train_loss',
    'train_batch_size',
    'backbone_unfrozen',
    'epoch_seconds',
    'val_loss',
    'val_flow_loss',
    'val_cellprob_loss',
    'cellprob_iou_pixel',
    'cellprob_iou_image_mean',
    'mask_pix_iou',
    'mask_instance_f1',
    'mask_precision',
    'mask_recall',
    'mask_boundary_iou',
    'mask_over_split_count',
    'mask_n_pred',
    'mask_n_gt',
    'mask_proxy_f1',
    'mask_proxy_count_error',
    'mask_proxy_area_error',
    'best_metric_name',
    'best_metric',
    'is_best',
]


def _write_metric_row(row, csv_path='checkpoints/training_metrics.csv',
                      jsonl_path='checkpoints/training_metrics.jsonl'):
    """Append one epoch/validation metric row to durable CSV and JSONL files."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    normalized = {field: row.get(field, '') for field in METRIC_FIELDS}
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized)

    with open(jsonl_path, 'a', encoding='utf-8') as jsonl_file:
        jsonl_file.write(json.dumps(normalized, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--allow-cpu', action='store_true',
                        help='允许在无 CUDA 时回退 CPU 训练（正式 V1 训练不应使用）')
    args = parser.parse_args()

    cfg = Config()
    if not torch.cuda.is_available() and args.device == 'cuda':
        if args.allow_cpu:
            print('WARNING: CUDA not available, falling back to CPU (--allow-cpu was set).')
            args.device = 'cpu'
        else:
            print('ERROR: CUDA not available and --allow-cpu not set.')
            print('V1 formal training requires GPU. Use --allow-cpu only for debugging.')
            sys.exit(1)
    cfg.device = args.device

    print(f'Device: {cfg.device}')
    print(f'Backbone: {cfg.backbone_name}')
    print(f'Input size: {cfg.input_size}')

    # 训练前硬门禁：manifest 和 readiness report 必须存在
    manifest_csv = os.path.join(cfg.data_root, 'dataset_manifest.csv')
    manifest_jsonl = os.path.join(cfg.data_root, 'dataset_manifest.jsonl')
    if not os.path.exists(manifest_csv) and not os.path.exists(manifest_jsonl):
        print(f'ERROR: dataset_manifest.csv or .jsonl not found at {cfg.data_root}/')
        print('Run init_dataset_manifest and check_dataset_manifest (step 8) first.')
        sys.exit(1)
    report_path = os.path.join(cfg.data_root, 'dataset_readiness_report.md')
    if not os.path.exists(report_path):
        print(f'ERROR: dataset_readiness_report.md not found at {report_path}')
        print('Run dataset_readiness_report (step 8) and run_pretrain_gates (step 9) first.')
        sys.exit(1)

    # P0-1: 验证 flow 目录存在，给出明确错误提示
    _check_flow_dir(cfg.train_flow_dir, cfg.train_img_dir, 'train')
    _check_flow_dir(cfg.val_flow_dir, cfg.val_img_dir, 'val')

    # 数据集
    train_dataset = FlowDataset(
        cfg.train_img_dir, cfg.train_flow_dir, cfg.input_size,
        cfg.mean, cfg.std, cfg.pad_value, augment=True,
        aug_flip_prob=cfg.aug_flip_prob,
        aug_rotate_prob=cfg.aug_rotate_prob,
        aug_jitter_prob=cfg.aug_jitter_prob,
        aug_scale_prob=cfg.aug_scale_prob,
        aug_scale_range=cfg.aug_scale_range)
    val_dataset = FlowDataset(
        cfg.val_img_dir, cfg.val_flow_dir, cfg.input_size,
        cfg.mean, cfg.std, cfg.pad_value, augment=False,
        label_dir=cfg.val_img_dir)

    train_batch_size = cfg.batch_size
    train_loader = _build_train_loader(train_dataset, cfg, train_batch_size)
    val_loader = DataLoader(val_dataset, batch_size=2,
                             shuffle=False, num_workers=2,
                             collate_fn=_val_collate)
    total_train_steps = _estimate_total_train_steps(train_dataset, cfg)

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples:   {len(val_dataset)}')
    print(f'Train batch size: {train_batch_size} '
          f'(after unfreeze: {getattr(cfg, "unfreeze_batch_size", train_batch_size)})')

    # 模型
    model = FlowModel(cfg)
    model = model.to(cfg.device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Params: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable')

    # 损失
    loss_fn = FlowLoss(
        flow_weight=cfg.flow_loss_weight,
        flow_scale=cfg.flow_scale,
        cellprob_weight=cfg.cellprob_loss_weight,
        flow_loss_mode=cfg.flow_loss_mode,
        cellprob_pos_weight_mode=cfg.cellprob_pos_weight_mode,
        cellprob_pos_weight_max=cfg.cellprob_pos_weight_max,
    )

    # 非 Cellpose 官方模式警告
    if cfg.flow_loss_mode != "all_pixels" or cfg.cellprob_pos_weight_mode != "none":
        print('WARNING: non-Cellpose loss mode enabled; '
              'results are not official-aligned baseline.')

    # ── 构建 optimizer + scheduler（P0-3 重构）──
    optimizer, scheduler = _build_optimizer_scheduler(
        model, cfg, total_train_steps, current_step=0, backbone_unfrozen=False)

    # 恢复
    start_epoch = 1
    best_val_metric = -1.0
    backbone_unfrozen = False
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=cfg.device)
        model.load_state_dict(ckpt['model_state_dict'])
        backbone_unfrozen = ckpt.get('backbone_unfrozen', False)
        if backbone_unfrozen:
            model.backbone.unfreeze_stages([2, 3])
            train_batch_size = getattr(cfg, 'unfreeze_batch_size', cfg.batch_size)
            train_loader = _build_train_loader(train_dataset, cfg, train_batch_size)
            print(f'Resumed with unfrozen backbone; train batch size={train_batch_size}')
        # 使用重建方式恢复，确保 base_lrs 一致
        current_step = ckpt.get('global_step', ckpt['epoch'] * len(train_loader))
        global_step = current_step
        optimizer, scheduler = _build_optimizer_scheduler(
            model, cfg, total_train_steps,
            current_step=current_step,
            backbone_unfrozen=backbone_unfrozen)
        # 恢复 AdamW 状态 (momentum/variance)
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as e:
                print(f'Warning: Could not restore optimizer state: {e}')
                print('  Continuing with fresh optimizer state (LR schedule preserved)')
        start_epoch = ckpt['epoch'] + 1
        best_val_metric = ckpt.get('best_val_metric',
                                   ckpt.get('best_val_iou', -1.0))

    for epoch in range(start_epoch, cfg.max_epochs + 1):
        # P0-3: 阶段2解冻 backbone → 重建 optimizer + scheduler
        if (not backbone_unfrozen and
                epoch >= cfg.unfreeze_backbone_after_epoch and
                hasattr(model.backbone, 'unfreeze_stages')):
            print('\n' + '=' * 55)
            print('Unfreezing backbone stages 2,3...')
            model.backbone.unfreeze_stages([2, 3])
            backbone_unfrozen = True

            new_batch_size = getattr(cfg, 'unfreeze_batch_size', cfg.batch_size)
            if new_batch_size < train_batch_size:
                train_batch_size = new_batch_size
                train_loader = _build_train_loader(train_dataset, cfg, train_batch_size)
                print(f'  Train batch size reduced to {train_batch_size} '
                      f'to avoid OOM after backbone unfreeze')
                if cfg.device == 'cuda':
                    torch.cuda.empty_cache()

            # 重建 optimizer + scheduler，AdamW 动量重置（新旧 param_groups 不一致无法迁移）
            optimizer, scheduler = _build_optimizer_scheduler(
                model, cfg, total_train_steps,
                current_step=global_step,
                backbone_unfrozen=True)
            print('  Optimizer reset: AdamW momentum reinitialized for all param groups')

            for g in optimizer.param_groups:
                print(f'  {g["name"]}: lr={g["lr"]:.6f}')
            print('=' * 55 + '\n')

        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for batch_idx, (images, flows) in enumerate(train_loader):
            images = images.to(cfg.device)
            flows = flows.to(cfg.device)

            pred = model(images)
            loss, flow_l, cell_l = loss_fn(pred, flows)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                _print_lr(optimizer, batch_idx, len(train_loader),
                         epoch, cfg, loss.item(), flow_l.item(), cell_l.item())

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - epoch_start
        print(f'Epoch [{epoch}/{cfg.max_epochs}] Avg Loss: {avg_loss:.4f} '
              f'Time: {elapsed:.1f}s')

        metric_row = {
            'epoch': epoch,
            'global_step': global_step,
            'train_loss': avg_loss,
            'train_batch_size': train_batch_size,
            'backbone_unfrozen': backbone_unfrozen,
            'epoch_seconds': elapsed,
            'best_metric': best_val_metric,
            'is_best': False,
        }

        # 每个 epoch 都保存 latest，避免非验证 epoch 崩溃时回退过多。
        os.makedirs('checkpoints', exist_ok=True)
        latest_ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_loss,
            'best_val_metric': best_val_metric,
            'backbone_unfrozen': backbone_unfrozen,
            'global_step': global_step,
            'train_batch_size': train_batch_size,
        }
        torch.save(latest_ckpt, 'checkpoints/latest.pth')

        # 验证 + 保存
        if epoch % cfg.val_interval == 0 or epoch == cfg.max_epochs:
            model.eval()
            val_loss = 0.0
            val_flow_loss = 0.0
            val_cellprob_loss = 0.0
            val_iou_sum = 0.0
            val_iou_pixel_inter = 0.0
            val_iou_pixel_union = 0.0
            val_count = 0
            all_preds = []
            all_flows = []
            all_labels = []
            all_resize_infos = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch[0].to(cfg.device)
                    flows = batch[1].to(cfg.device)
                    pred = model(images)
                    loss, flow_l, cell_l = loss_fn(pred, flows)
                    val_loss += loss.item()
                    val_flow_loss += flow_l.item()
                    val_cellprob_loss += cell_l.item()

                    all_preds.append(pred.cpu().numpy())
                    all_flows.append(flows.cpu().numpy())
                    if len(batch) > 2 and len(batch[2]) > 0:
                        all_labels.extend([lab.numpy() for lab in batch[2]])
                        all_resize_infos.extend(batch[3])

                    # 逐像素 cellprob IoU (logit > 0 → foreground)
                    pred_cp = (pred[:, 2] > 0).float()
                    gt_cp = (flows[:, 0] > 0.5).float()
                    intersection = (pred_cp * gt_cp).sum(dim=(1, 2))
                    union = ((pred_cp + gt_cp) > 0).float().sum(dim=(1, 2))
                    # image-mean IoU (逐图累加, 避免最后 batch 权重偏大)
                    for b in range(images.size(0)):
                        iou = (intersection[b] / union[b].clamp(min=1)).item()
                        val_iou_sum += iou
                    # pixel-weighted IoU (总交集 / 总并集)
                    val_iou_pixel_inter += intersection.sum().item()
                    val_iou_pixel_union += union.sum().item()
                    val_count += images.size(0)

            val_loss /= max(len(val_loader), 1)
            val_flow_loss /= max(len(val_loader), 1)
            val_cellprob_loss /= max(len(val_loader), 1)
            val_iou_image = val_iou_sum / max(val_count, 1)
            val_iou_pixel = (val_iou_pixel_inter / max(val_iou_pixel_union, 1.0))

            print(f'  Val Loss: {val_loss:.4f} '
                  f'(flow={val_flow_loss:.4f} cellprob={val_cellprob_loss:.4f})')
            print(f'  cellprob IoU: pixel={val_iou_pixel:.4f} image_mean={val_iou_image:.4f}')

            # ── Mask-level 评估（依赖 val_compute_masks 配置）──
            val_mask_f1 = -1.0
            has_gt_labels = False
            mask_metrics_gt = {}
            mask_metrics_proxy = {}
            if cfg.val_compute_masks and len(all_preds) > 0:
                preds_concat = np.concatenate(all_preds, axis=0)

                # 主判据：基于真实 GT labels 的 mask instance F1
                if all_labels:
                    mask_metrics_gt = evaluate_batch_v1_labels(
                        preds_concat, all_labels, all_resize_infos,
                        target_size=cfg.input_size,
                        cellprob_threshold=0.0,
                        flow_threshold=cfg.fd_flow_threshold,
                    )
                    print(f'  Mask (GT labels): F1={mask_metrics_gt["instance_f1"]:.4f} '
                          f'P={mask_metrics_gt["precision"]:.4f} R={mask_metrics_gt["recall"]:.4f} '
                          f'BIoU={mask_metrics_gt["boundary_iou_mean"]:.4f} '
                          f'(pred={int(mask_metrics_gt["n_pred"])} gt={int(mask_metrics_gt["n_gt"])})')
                    val_mask_f1 = mask_metrics_gt['instance_f1']
                    has_gt_labels = True
                else:
                    print('  ERROR: no GT labels loaded for validation. '
                          'V1 requires val *_labels.png for mask metrics.')
                    sys.exit(1)

                # 诊断项：flow-derived GT mask（仅用于对照，不用于 best.pth 选择）
                flows_concat = np.concatenate(all_flows, axis=0)
                mask_metrics_proxy = evaluate_batch_v1(
                    preds_concat, flows_concat,
                    cellprob_threshold=0.0,
                    flow_threshold=cfg.fd_flow_threshold,
                )
                print(f'  Mask proxy (flow-derived): F1={mask_metrics_proxy["instance_f1"]:.4f} '
                      f'(pred={int(mask_metrics_proxy["n_pred"])} gt={int(mask_metrics_proxy["n_gt"])})')
            else:
                if not cfg.val_compute_masks:
                    print('  ERROR: val_compute_masks must be True for V1 training. '
                          'best checkpoint requires mask_instance_f1 from GT labels.')
                    sys.exit(1)

            # V1 best checkpoint 主判据：真实 GT label 的 mask_instance_f1
            best_metric_name = 'mask_instance_f1'
            val_metric = val_mask_f1

            # 先判断是否刷新 best，再构造 checkpoint（确保 best.pth 中 best_val_metric 为最新值）
            is_best = val_metric > best_val_metric
            if is_best:
                best_val_metric = val_metric

            metric_row.update({
                'val_loss': val_loss,
                'val_flow_loss': val_flow_loss,
                'val_cellprob_loss': val_cellprob_loss,
                'cellprob_iou_pixel': val_iou_pixel,
                'cellprob_iou_image_mean': val_iou_image,
                'best_metric_name': best_metric_name,
                'best_metric': val_metric,
                'is_best': is_best,
            })
            if val_mask_f1 >= 0:
                if not has_gt_labels:
                    import sys
                    sys.exit("ERROR: mask evaluation requires GT labels; "
                             "flow pairs gate must be run before training. "
                             "No GT labels found for validation set.")
                metric_row.update({
                    'mask_instance_f1': mask_metrics_gt['instance_f1'],
                    'mask_precision': mask_metrics_gt['precision'],
                    'mask_recall': mask_metrics_gt['recall'],
                    'mask_pix_iou': mask_metrics_gt['pix_iou'],
                    'mask_boundary_iou': mask_metrics_gt['boundary_iou_mean'],
                    'mask_n_pred': mask_metrics_gt['n_pred'],
                    'mask_n_gt': mask_metrics_gt['n_gt'],
                    'mask_over_split_count': mask_metrics_gt.get('over_split_proxy_count', 0),
                    'mask_proxy_f1': mask_metrics_proxy['instance_f1'],
                    'mask_proxy_count_error': mask_metrics_proxy['count_error'],
                    'mask_proxy_area_error': mask_metrics_proxy['mean_area_error'],
                })

            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_metric': val_metric,
                'best_val_metric': best_val_metric,
                'best_metric_name': best_metric_name,
                'backbone_unfrozen': backbone_unfrozen,
                'global_step': global_step,
                'train_batch_size': train_batch_size,
            }
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(ckpt, f'checkpoints/epoch_{epoch}.pth')
            torch.save(ckpt, 'checkpoints/latest.pth')

            if is_best:
                torch.save(ckpt, 'checkpoints/best.pth')
                print(f'  New best model ({best_metric_name}={val_metric:.4f}, '
                      f'cellprob_IoU_pixel={val_iou_pixel:.4f}, '
                      f'img_mean={val_iou_image:.4f})')

        _write_metric_row(metric_row)

    print(f'Training complete! Best val metric: {best_val_metric:.4f}')


if __name__ == '__main__':
    main()
