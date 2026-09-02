"""FlowDynamics 参数网格搜索（P2-5）

在固定验证子集上，对 FlowDynamics 关键参数做网格搜索，
按实例级指标（Pixel IoU、Recall、Precision、实例数差异）选择最优组合。

用法:
    python -m model_flow.tune_flow_dynamics \
        --checkpoint checkpoints/best.pth \
        --img-dir data/particles/val \
        --flow-dir data/particles/flows_val \
        --samples temp/stratified_val_samples.txt

注意: 需要在有 PyTorch 和 cellpose 的环境中运行。
"""
import os
import json
import argparse
import itertools

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ..config import Config
from ..eval_masks import _unwrap, evaluate_masks_v1
from ..flow_head import FlowModel
from ..utils import imread_unicode, imread_unchanged


def preprocess(img_bgr, cfg):
    """与 inference.py 一致的预处理。"""
    h, w = img_bgr.shape[:2]
    scale = cfg.input_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = cfg.input_size - new_h
    pad_w = cfg.input_size - new_w
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    img = cv2.copyMakeBorder(
        img, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT, value=(cfg.pad_value,) * 3)

    img = img[:, :, ::-1].astype(np.float32) / 255.0
    mean_arr = np.array(cfg.mean, dtype=np.float32)
    std_arr = np.array(cfg.std, dtype=np.float32)
    img = (img - mean_arr) / std_arr
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor, (scale, pad_left, pad_top, new_h, new_w)


def compute_cellpose_masks(dy, dx, cellprob_prob, niter, flow_threshold, min_size):
    """使用 Cellpose compute_masks 生成实例 mask（基准）。"""
    from cellpose.dynamics import compute_masks
    dP = np.stack([dy, dx], axis=0)
    labels = _unwrap(compute_masks(
        dP, cellprob_prob, niter=niter,
        cellprob_threshold=0.5,  # 概率空间
        flow_threshold=flow_threshold,
        min_size=min_size,
        max_size_fraction=0.5,
        device=torch.device('cpu'),
    ))
    return labels


def crop_padded_mask_to_original(mask, preprocess_meta, original_shape):
    """将 input_size padded 空间的 mask 裁剪并还原到原图尺寸。"""
    _, pad_left, pad_top, new_h, new_w = preprocess_meta
    orig_h, orig_w = original_shape
    valid = mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    return cv2.resize(valid.astype(np.int32), (orig_w, orig_h),
                      interpolation=cv2.INTER_NEAREST)


def compute_cpp_equiv_masks(dy, dx, cellprob_prob, niter, cellprob_threshold_logit,
                             flow_threshold, min_size,
                             max_size_fraction=0.5,
                             boundary_particle_policy='include',
                             edge_touch_margin_px=1.0):
    """使用 C++ FlowDynamics 的 Python 等效实现（参数搜索用）。

    核心 Euler 积分复用在 euler_core.py，本函数只负责参数搜索特有的
    cellpose remove_bad_flow_masks 和 mean logit 过滤。
    """
    from .euler_core import (euler_integrate_to_labels,
                              remove_large_masks,
                              fill_holes_labels,
                              filter_by_size_and_boundary,
                              filter_by_mean_logit)

    # ── prob → logit + 前景遮罩 ──
    cellprob_logit = np.log(np.maximum(cellprob_prob, 1e-7) /
                            np.maximum(1 - cellprob_prob, 1e-7))
    fg_mask = (cellprob_logit > 0.0)

    labels = euler_integrate_to_labels(dy, dx, fg_mask, niter)

    # ── max_size_fraction（先于 flow QC，与 C++ 顺序一致）──
    labels = remove_large_masks(labels, max_size_fraction)

    # ── remove_bad_flow_masks（使用 euler_core Python 同源实现, 与 C++ 等价）──
    if flow_threshold > 0 and labels.max() > 0:
        from .euler_core import remove_bad_flow_masks as euler_remove_bad
        labels = euler_remove_bad(labels, dy, dx, flow_threshold)

    # ── 尺寸 + 边界过滤 + mean logit 过滤 ──
    labels = fill_holes_labels(labels)
    labels = filter_by_size_and_boundary(
        labels, min_size, max_size_fraction,
        boundary_particle_policy, edge_touch_margin_px)
    labels = filter_by_mean_logit(labels, cellprob_logit, cellprob_threshold_logit)

    return labels


def score_metrics(metrics):
    """V1 selection score: instance correctness and contour overlap only."""
    split_penalty = min(metrics['over_split_proxy_count'] / max(metrics['n_gt'], 1), 1.0)
    count_balance = 1.0 - abs(metrics['n_pred'] - metrics['n_gt']) / max(metrics['n_gt'], 1)
    count_balance = max(0.0, count_balance)
    return (
        0.35 * metrics['instance_f1'] +
        0.25 * metrics['boundary_iou_mean'] +
        0.20 * metrics['mask_iou_mean'] +
        0.05 * metrics['recall'] +
        0.05 * metrics['precision'] +
        0.05 * count_balance +
        0.05 * (1.0 - split_penalty)
    )


def selection_key(row):
    """Tie-break order from the V1 plan."""
    return (
        row['score'],
        row['avg_instance_f1'],
        row['avg_boundary_iou_mean'],
        row['avg_mask_iou_mean'],
        row['avg_recall'],
        row['avg_precision'],
        -row['avg_over_split_proxy_count'],
        -abs(row['avg_n_pred'] - row['avg_n_gt']),
    )


def collect_image_files(img_dir, samples_file='', files_arg='', limit=0):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    available = [
        f for f in sorted(os.listdir(img_dir))
        if os.path.splitext(f)[1].lower() in exts
    ]
    available_set = set(available)

    requested = []
    if samples_file:
        with open(samples_file, 'r', encoding='utf-8') as handle:
            requested.extend([
                line.strip() for line in handle
                if line.strip() and not line.strip().startswith('#')
            ])
    if files_arg:
        requested.extend([
            item.strip() for item in files_arg.split(',')
            if item.strip()
        ])

    if requested:
        missing = [name for name in requested if name not in available_set]
        if missing:
            print(f"ERROR: {len(missing)} requested sample files not found in {img_dir}:")
            for name in missing[:20]:
                print(f"  {name}")
            raise SystemExit(1)
        return requested

    if limit and limit > 0:
        print("WARNING: --limit without --samples/--files uses filename order. "
              "Use a stratified sample list for formal parameter selection.")
        return available[:limit]
    return available


def main():
    parser = argparse.ArgumentParser(description='FlowDynamics 参数网格搜索')
    parser.add_argument('--checkpoint', default='checkpoints/best.pth')
    parser.add_argument('--img-dir', default='data/particles/val')
    parser.add_argument('--flow-dir', default='data/particles/flows_val')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit image count when no --samples/--files is provided. 0=all.')
    parser.add_argument('--samples', default='',
                        help='Text file containing image filenames to evaluate, one per line.')
    parser.add_argument('--files', default='',
                        help='Comma-separated image filenames to evaluate.')
    parser.add_argument('--gt-labels-dir', default='',
                        help='Directory of *_labels.png original human-annotated masks. '
                             'When provided, uses these as GT for parameter selection '
                             'instead of 5× flow-computed masks. Strongly recommended '
                             'for formal parameter search.')
    parser.add_argument('--allow-label-fallback', action='store_true',
                        help='Allow fallback to 5× flow-computed GT when *_labels.png '
                             'is missing or empty. Not recommended for formal parameter '
                             'search; use only for debugging or when labels are incomplete.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', '-o', default='temp/flow_dynamics_best_params.json',
                        help='输出最优参数 JSON')
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device

    # 加载模型
    print(f'Loading model: {args.checkpoint}')
    model = FlowModel(cfg)
    model = model.to(cfg.device)
    ckpt = torch.load(args.checkpoint, map_location=cfg.device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.neck.load_state_dict(ckpt['neck'])
        model.flow_head.load_state_dict(ckpt['flow_head'])
    model.eval()

    # 收集图片
    img_files = collect_image_files(args.img_dir, args.samples, args.files, args.limit)

    print(f'Evaluating on {len(img_files)} images...')
    if not img_files:
        raise SystemExit('No images selected for parameter search.')

    # 定义搜索空间
    param_grid = {
        'cellprob_threshold_logit': [-4, -3, -2, -1, 0],
        'flow_threshold': [0.2, 0.3, 0.4, 0.5, 0.6],
        'min_size': [ 30, 50, 80, 120],
    }

    # ── 生成 GT masks ──
    # 优先使用原始 *_labels.png (人工标注轮廓), 不可用时回退到 5×GT flow 反推 mask
    from cellpose.dynamics import compute_masks

    use_label_gt = bool(args.gt_labels_dir)
    strict_labels = use_label_gt and not args.allow_label_fallback

    gt_masks = {}
    gt_source_by_image: dict[str, str] = {}
    label_count = 0
    flow_fallback_count = 0
    missing_images: list[str] = []

    for fname in tqdm(img_files):
        name = os.path.splitext(fname)[0]

        if use_label_gt:
            label_path = os.path.join(args.gt_labels_dir, name + '_labels.png')
            if os.path.exists(label_path):
                label_img = imread_unchanged(label_path)
                if label_img is not None and label_img.max() > 0:
                    gt_masks[fname] = label_img.astype(np.int32)
                    gt_source_by_image[fname] = 'original_label'
                    label_count += 1
                    continue
            # label 缺失或为空
            if strict_labels:
                missing_images.append(fname)
                continue
            flow_fallback_count += 1

        # 回退: 5×GT flow → compute_masks
        flow_path = os.path.join(args.flow_dir, name + '.npy')
        if not os.path.exists(flow_path):
            continue
        flow = np.load(flow_path)
        # flow channels: [cellprob, dy, dx]
        gt_cp = flow[0]
        gt_dP = 5.0 * np.stack([flow[1], flow[2]], axis=0).astype(np.float64)
        gt_mask = _unwrap(compute_masks(
            gt_dP, gt_cp,
            cellprob_threshold=0.0,
            flow_threshold=0.4,
            min_size=1,
        ))
        gt_masks[fname] = gt_mask
        if use_label_gt:
            gt_source_by_image[fname] = 'flow_fallback'
        else:
            gt_source_by_image[fname] = 'flow_computed'

    # strict 模式: 有缺失 label 直接失败
    if strict_labels and missing_images:
        print(f'\nERROR: --gt-labels-dir is set but {len(missing_images)} image(s) '
              f'lack valid *_labels.png. Formal parameter search requires original '
              f'labels for every sample. Missing:')
        for fname in missing_images[:20]:
            print(f'  {fname}')
        if len(missing_images) > 20:
            print(f'  ... {len(missing_images) - 20} more')
        print('Use --allow-label-fallback to permit flow-computed GT instead, '
              'but this is not recommended for formal search.')
        raise SystemExit(1)

    # 构建动态 gt_source_info (反映实际使用的 GT 来源)
    if use_label_gt:
        if flow_fallback_count > 0:
            gt_source_info = (f'original labels ({label_count} images) '
                              f'+ 5× flow fallback ({flow_fallback_count} images)')
        else:
            gt_source_info = f'original labels (*_labels.png, {label_count} images)'
    else:
        gt_source_info = '5× flow compute_masks'

    # 记录 GT 实例统计
    n_instances = sum(len(np.unique(m)) - 1 for m in gt_masks.values())
    print(f'  GT source: {gt_source_info}')
    print(f'  GT instances: {n_instances}')
    if use_label_gt and flow_fallback_count > 0:
        print(f'  WARNING: {flow_fallback_count} image(s) missing labels, '
              f'fell back to flow-computed GT. '
              f'Remove --allow-label-fallback to require labels for all samples.')

    # 对每张图推理一次
    print('\n--- Model inference ---')
    all_results = {}
    for fname in tqdm(img_files):
        if fname not in gt_masks:
            continue

        img_path = os.path.join(args.img_dir, fname)
        img_bgr = imread_unicode(img_path)
        if img_bgr is None:
            continue

        tensor, (scale, pad_left, pad_top, new_h, new_w) = preprocess(img_bgr, cfg)
        tensor = tensor.to(cfg.device)

        with torch.no_grad():
            flow_s4 = model(tensor)

        flow_s4 = flow_s4[0].cpu().numpy()
        flow_full = np.zeros((3, cfg.input_size, cfg.input_size), dtype=np.float32)
        for c in range(3):
            flow_full[c] = cv2.resize(flow_s4[c],
                                       (cfg.input_size, cfg.input_size),
                                       interpolation=cv2.INTER_LINEAR)

        dy = flow_full[0]
        dx = flow_full[1]
        cellprob_logit = flow_full[2]

        # 屏蔽 padding
        pad_mask = np.zeros((cfg.input_size, cfg.input_size), dtype=np.float32)
        pad_mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = 1.0
        cellprob_logit[pad_mask < 0.5] = -100.0

        cellprob_prob = 1.0 / (1.0 + np.exp(-cellprob_logit))
        all_results[fname] = (
            dy, dx, cellprob_prob,
            (scale, pad_left, pad_top, new_h, new_w),
            img_bgr.shape[:2],
        )

    # 网格搜索
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, vals))
              for vals in itertools.product(*param_grid.values())]

    all_scores = []

    print(f'\n--- Grid search ({len(combos)} combinations) ---')
    for combo in tqdm(combos, desc='Searching'):
        scores = []
        metrics = []
        for fname, (dy, dx, cp_prob, preprocess_meta, original_shape) in all_results.items():
            pred = compute_cpp_equiv_masks(
                dy, dx, cp_prob,
                niter=cfg.fd_niter,
                cellprob_threshold_logit=combo['cellprob_threshold_logit'],
                flow_threshold=combo['flow_threshold'],
                min_size=combo['min_size'],
                max_size_fraction=cfg.fd_max_size_fraction,
            )
            gt = gt_masks[fname]
            pred = crop_padded_mask_to_original(pred, preprocess_meta, original_shape)
            s = evaluate_masks_v1(pred, gt)
            count_balance = 1.0 - abs(s['n_pred'] - s['n_gt']) / max(s['n_gt'], 1)
            combo_score = score_metrics(s)
            scores.append(combo_score)
            metrics.append({**s, 'count_balance': count_balance, 'score': combo_score})

        avg_score = np.mean(scores) if scores else 0.0
        score_std = float(np.std(scores)) if scores else 0.0
        avg_metrics = {}
        metric_keys = [
            'pix_iou', 'mask_iou_mean', 'boundary_iou_mean',
            'recall', 'precision', 'instance_f1',
            'over_split_proxy_count',
            'count_balance', 'n_pred', 'n_gt',
        ]
        for key in metric_keys:
            avg_metrics[f'avg_{key}'] = float(np.mean([m[key] for m in metrics])) if metrics else 0.0
        all_scores.append({
            **combo,
            'score': float(avg_score),
            'score_std': score_std,
            **avg_metrics,
        })

    # 输出结果
    all_scores.sort(key=selection_key, reverse=True)
    if not all_scores:
        raise SystemExit('No parameter combinations were scored. Check flow files and samples.')
    best_row = all_scores[0] if all_scores else {}
    best_combo = {
        'cellprob_threshold_logit': best_row.get('cellprob_threshold_logit'),
        'flow_threshold': best_row.get('flow_threshold'),
        'min_size': best_row.get('min_size'),
    }
    best_score = float(best_row.get('score', 0.0))
    print(f'\n{"=" * 70}')
    print(f'Top 5 parameter combinations:')
    print(f'{"Rank":<6} {"cellprob_thr":<16} {"flow_thr":<12} {"min_size":<10} '
          f'{"Score":<10} {"F1":<8} {"BIoU":<8} {"MIoU":<8} {"Recall":<8}')
    print('-' * 70)
    for i, s in enumerate(all_scores[:5]):
        print(f'{i+1:<6} {s["cellprob_threshold_logit"]:<16} '
              f'{s["flow_threshold"]:<12} {s["min_size"]:<10} '
              f'{s["score"]:.4f}   {s["avg_instance_f1"]:.4f} '
              f'{s["avg_boundary_iou_mean"]:.4f} '
              f'{s["avg_mask_iou_mean"]:.4f} '
              f'{s["avg_recall"]:.4f}')

    top_score = all_scores[0]['score'] if all_scores else 0.0
    tied = [s for s in all_scores if abs(s['score'] - top_score) < 1e-9]
    if len(tied) > 1:
        print(f'\nWARNING: {len(tied)} parameter combinations tie for the top score '
              f'({top_score:.6f}). Treat best_params as a candidate, not a unique optimum.')

    print(f'\nBest parameters:')
    for k, v in best_combo.items():
        print(f'  {k}: {v} (score={best_score:.4f})')

    # 保存结果
    result = {
        'best_params': best_combo,
        'best_score': best_score,
        'gt_source': gt_source_info,
        'gt_stats': {
            'total_evaluated': len(gt_masks),
            'by_source': {
                'original_label': label_count,
                'flow_fallback': flow_fallback_count,
                'flow_only': len(gt_masks) - label_count - flow_fallback_count,
            },
            'strict_mode': strict_labels,
            'missing_label_images': missing_images,
        },
        'selection_rule': (
            'Sort by V1 contour score, instance F1, boundary IoU, mask IoU, '
            'recall, precision, lower over-split count, and count balance.'
        ),
        'all_results': all_scores[:10],
        'search_space': {k: list(v) for k, v in param_grid.items()},
        'samples': img_files,
        'tie_count_at_best_score': len(tied),
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2, default=float)
    print(f'\nResults saved to: {args.out}')


if __name__ == '__main__':
    main()
