"""Mask-level 验证指标：使用 Cellpose compute_masks() 评估实例分割质量。

V1 指标的公共函数（equivalent_diameter_px / contour_boundary / binary_iou /
percentile / evaluate_masks_v1）在这里统一定义，供训练验证、参数搜索
和离线验收复用，避免多处各自实现导致行为不一致。
"""
import cv2
import numpy as np
from cellpose.dynamics import compute_masks


def _unwrap(result):
    """兼容 Cellpose compute_masks 返回值。

    Cellpose 4.1.1 返回单个 ndarray，旧版本返回 (masks, flows) 元组。
    """
    return result[0] if isinstance(result, tuple) else result


# ═══════════════════════════════════════════════════════════════
# 公共度量函数
# ═══════════════════════════════════════════════════════════════

def equivalent_diameter_px(mask: np.ndarray) -> float:
    area = float(mask.sum())
    return float(np.sqrt(4.0 * area / np.pi)) if area > 0 else 0.0


def contour_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return ((mask.astype(np.uint8) > 0) & (eroded == 0)).astype(np.uint8)


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def percentile(vals, q):
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


# ═══════════════════════════════════════════════════════════════
# V1 实例级指标（参数搜索 / 训练验证 / 离线验收共用）
# ═══════════════════════════════════════════════════════════════

def _match_instances(pred: np.ndarray, gt: np.ndarray, iou_threshold: float = 0.5):
    """贪心一对一实例匹配，供 evaluate_masks_v1 和离线验收共用。

    Returns:
        matches: [(pred_id, gt_id, iou), ...]
        candidate_counts_by_gt: {gt_id: count} 每 GT 有多少 pred 候选 (IoU >= threshold)
        n_pred, n_gt: int
    """
    ids_pred = [i for i in np.unique(pred) if i > 0]
    ids_gt = [i for i in np.unique(gt) if i > 0]

    candidate_counts_by_gt = {gid: 0 for gid in ids_gt}
    candidates = []
    for pid in ids_pred:
        pm = (pred == pid).astype(np.uint8)
        for gid in ids_gt:
            gm = (gt == gid).astype(np.uint8)
            iou_val = binary_iou(pm, gm)
            if iou_val >= iou_threshold:
                candidates.append((iou_val, pid, gid))
                candidate_counts_by_gt[gid] += 1

    candidates.sort(reverse=True)
    matched_pred_ids = set()
    matched_gt_ids = set()
    matches = []
    for iou_val, pid, gid in candidates:
        if pid in matched_pred_ids or gid in matched_gt_ids:
            continue
        matched_pred_ids.add(pid)
        matched_gt_ids.add(gid)
        matches.append((pid, gid, iou_val))

    return matches, candidate_counts_by_gt, len(ids_pred), len(ids_gt)


SMALL_PARTICLE_DIAMETER_THRESHOLD_PX = 32.0
OUTPUT_STRIDE = 4


def evaluate_masks_v1(pred: np.ndarray, gt: np.ndarray,
                      small_diameter_threshold: float = SMALL_PARTICLE_DIAMETER_THRESHOLD_PX) -> dict:
    """单张图 V1 实例级指标。

    Args:
        pred: (H, W) int32, 预测实例 mask, 0=背景
        gt:   (H, W) int32, GT 实例 mask, 0=背景
        small_diameter_threshold: 小颗粒等效圆直径阈值 (px), 默认 32。

    Returns:
        dict 包含 pix_iou, mask_iou_mean, boundary_iou_mean, instance_f1,
        precision, recall, small_particle_recall_lt32px,
        equivalent_diameter_mae_px, area_absolute_relative_error_mean,
        over_split_proxy_count, psd_d10/d50/d90_bias_px, n_pred, n_gt。
    """
    pred_b = (pred > 0).astype(np.int32)
    gt_b = (gt > 0).astype(np.int32)
    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    pix_iou = inter / union if union > 0 else 0.0

    ids_gt = [i for i in np.unique(gt) if i > 0]

    matches, candidate_counts_by_gt, n_pred, n_gt = _match_instances(pred, gt)

    if n_pred == 0 and n_gt == 0:
        return {
            'pix_iou': 1.0, 'mask_iou_mean': 1.0, 'boundary_iou_mean': 1.0,
            'recall': 1.0, 'precision': 1.0, 'instance_f1': 1.0,
            'small_particle_recall_lt32px': 1.0,
            'equivalent_diameter_mae_px': 0.0,
            'area_absolute_relative_error_mean': 0.0,
            'over_split_proxy_count': 0,
            'psd_d10_bias_px': 0.0, 'psd_d50_bias_px': 0.0, 'psd_d90_bias_px': 0.0,
            'n_pred': 0, 'n_gt': 0,
        }

    precision = len(matches) / max(n_pred, 1)
    recall = len(matches) / max(n_gt, 1)
    instance_f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    matched_gt_ids = {gid for _, gid, _ in matches}

    mask_ious = []
    boundary_ious = []
    diameter_errors = []
    area_errors = []
    gt_matched_diameters = []
    pred_matched_diameters = []
    for pid, gid, iou_val in matches:
        pm = (pred == pid).astype(np.uint8)
        gm = (gt == gid).astype(np.uint8)
        gd = equivalent_diameter_px(gm)
        pd = equivalent_diameter_px(pm)
        mask_ious.append(iou_val)
        boundary_ious.append(binary_iou(contour_boundary(pm), contour_boundary(gm)))
        diameter_errors.append(abs(pd - gd))
        area_errors.append(abs(float(pm.sum()) - float(gm.sum())) / max(float(gm.sum()), 1.0))
        gt_matched_diameters.append(gd)
        pred_matched_diameters.append(pd)

    small_gt = [
        gid for gid in ids_gt
        if equivalent_diameter_px((gt == gid).astype(np.uint8)) < small_diameter_threshold
    ]
    small_recall = (
        len(set(small_gt) & matched_gt_ids) / len(small_gt)
        if small_gt else 1.0
    )
    over_split = sum(max(0, count - 1) for count in candidate_counts_by_gt.values())

    return {
        'pix_iou': float(pix_iou),
        'mask_iou_mean': float(np.mean(mask_ious)) if mask_ious else 0.0,
        'boundary_iou_mean': float(np.mean(boundary_ious)) if boundary_ious else 0.0,
        'recall': float(recall),
        'precision': float(precision),
        'instance_f1': float(instance_f1),
        'small_particle_recall_lt32px': float(small_recall),
        'equivalent_diameter_mae_px': float(np.mean(diameter_errors)) if diameter_errors else 0.0,
        'area_absolute_relative_error_mean': float(np.mean(area_errors)) if area_errors else 0.0,
        'over_split_proxy_count': int(over_split),
        'psd_d10_bias_px': percentile(pred_matched_diameters, 10) - percentile(gt_matched_diameters, 10),
        'psd_d50_bias_px': percentile(pred_matched_diameters, 50) - percentile(gt_matched_diameters, 50),
        'psd_d90_bias_px': percentile(pred_matched_diameters, 90) - percentile(gt_matched_diameters, 90),
        'n_pred': n_pred,
        'n_gt': n_gt,
    }


# ═══════════════════════════════════════════════════════════════
# 旧版训练验证（保留向后兼容）
# ═══════════════════════════════════════════════════════════════

def _compute_instance_metrics(pred_masks: np.ndarray, gt_masks: np.ndarray,
                              pred_cellprob: np.ndarray, gt_cellprob: np.ndarray) -> dict:
    """计算预测 mask 与 GT mask 之间的指标。

    Args:
        pred_masks: (H, W) int16, 0=背景, >0=实例ID
        gt_masks: (H, W) int16, 0=背景, >0=实例ID
        pred_cellprob: (H, W) float32, 预测 cellprob logits
        gt_cellprob: (H, W) float32, GT cellprob (二值, >0.5=前景)

    Returns:
        dict: 包含 pix_iou, instance_f1, count_error, mean_area_error
    """
    pred_fg = (pred_masks > 0).astype(np.float32)
    gt_fg = (gt_cellprob > 0.5).astype(np.float32)

    # PixIoU (前景 vs GT 前景)
    inter = (pred_fg * gt_fg).sum()
    union = ((pred_fg + gt_fg) > 0).sum()
    pix_iou = inter / max(union, 1)

    # 实例匹配 (IoU > 0.5 视为匹配)
    pred_ids = [i for i in np.unique(pred_masks) if i > 0]
    gt_ids = [i for i in np.unique(gt_masks) if i > 0]

    if len(pred_ids) > 0 and len(gt_ids) > 0:
        # ── Greedy bipartite matching (与 tune_flow_dynamics.py 一致) ──
        candidates = []
        for pid in pred_ids:
            p_mask = (pred_masks == pid)
            for gid in gt_ids:
                g_mask = (gt_masks == gid)
                inter_ij = (p_mask * g_mask).sum()
                union_ij = ((p_mask + g_mask) > 0).sum()
                iou_ij = inter_ij / max(union_ij, 1)
                if iou_ij >= 0.5:
                    candidates.append((iou_ij, pid, gid))

        candidates.sort(reverse=True, key=lambda x: x[0])
        matched_pred = set()
        matched_gt = set()
        matched_pairs = []
        for iou_val, pid, gid in candidates:
            if pid in matched_pred or gid in matched_gt:
                continue
            matched_pred.add(pid)
            matched_gt.add(gid)
            matched_pairs.append((pid, gid, iou_val))

        tp = len(matched_pairs)
        fp = len(pred_ids) - tp
        fn = len(gt_ids) - len(matched_gt)
    elif len(pred_ids) > 0:
        fp = len(pred_ids)
        fn = 0
        matched_pairs = []
    elif len(gt_ids) > 0:
        fp = 0
        fn = len(gt_ids)
        matched_pairs = []
    else:
        tp = fp = fn = 0
        matched_pairs = []

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    instance_f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Count error
    count_error = abs(len(pred_ids) - len(gt_ids))

    # Area error (仅对匹配的实例，与 tune_flow_dynamics.py 一致)
    area_errors = []
    for pid, gid, _iou_val in matched_pairs:
        p_area = (pred_masks == pid).sum()
        g_area = (gt_masks == gid).sum()
        area_errors.append(abs(p_area - g_area) / max(g_area, 1))
    mean_area_error = np.mean(area_errors) if area_errors else 0.0

    return {
        'pix_iou': float(pix_iou),
        'instance_f1': float(instance_f1),
        'precision': float(precision),
        'recall': float(recall),
        'count_error': float(count_error),
        'mean_area_error': float(mean_area_error),
        'n_pred': len(pred_ids),
        'n_gt': len(gt_ids),
    }


def evaluate_batch(pred_flows: np.ndarray, gt_flows: np.ndarray,
                   cellprob_threshold: float = 0.0,
                   flow_threshold: float = 0.4,
                   min_size: int = 1) -> dict:
    """对一批预测和 GT flow 计算 mask-level 指标。

    在 stride-4 分辨率上运行 Cellpose compute_masks()，避免上采样开销。
    使用 GT flow 作为输入运行 compute_masks() 可获得近似 GT 实例掩码。

    Args:
        pred_flows: (B, 3, H, W) float32, 预测 [dy, dx, cellprob]
        gt_flows: (B, 3, H, W) float32, GT [cellprob, dy, dx]
        cellprob_threshold: compute_masks 的 cellprob 阈值 (logit 空间)
        flow_threshold: remove_bad_flow_masks 阈值
        min_size: 最小 mask 面积 (stride-4 分辨率下用 1 避免过度过滤)

    Returns:
        dict: 各指标的平均值
    """
    B = pred_flows.shape[0]
    metrics_list = []

    for b in range(B):
        # 预测 flow: pred = [dy, dx, cellprob] → dP = (2, H, W)
        pred_dP = np.stack([
            pred_flows[b, 0].astype(np.float64),
            pred_flows[b, 1].astype(np.float64),
        ], axis=0)
        pred_cp = pred_flows[b, 2].astype(np.float32)

        # GT flow: gt = [cellprob, dy, dx] → dP = (2, H, W)
        # ×5: labels_to_flows() 输出流值在 [-1,1]，训练 target 为 5×GT，
        # compute_masks() 内部对输入 dP 做 ÷5，因此重建 GT mask 需乘回 5
        gt_dP = 5.0 * np.stack([
            gt_flows[b, 1].astype(np.float64),
            gt_flows[b, 2].astype(np.float64),
        ], axis=0)
        gt_cp = gt_flows[b, 0].astype(np.float32)

        # 在 stride-4 分辨率上运行 compute_masks
        pred_masks = _unwrap(compute_masks(
            pred_dP, pred_cp,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
        ))
        gt_masks = _unwrap(compute_masks(
            gt_dP, gt_cp,
            cellprob_threshold=0.0,  # GT: 接受所有前景
            flow_threshold=flow_threshold,
            min_size=min_size,
        ))

        metrics = _compute_instance_metrics(pred_masks, gt_masks, pred_cp, gt_cp)
        metrics_list.append(metrics)

    # 平均
    avg = {}
    keys = ['pix_iou', 'instance_f1', 'precision', 'recall',
            'count_error', 'mean_area_error', 'n_pred', 'n_gt']
    for k in keys:
        avg[k] = float(np.mean([m[k] for m in metrics_list]))
    return avg


def evaluate_batch_v1_labels(pred_flows: np.ndarray, gt_labels: list[np.ndarray],
                              resize_infos: list, target_size: int = 1024,
                              cellprob_threshold: float = 0.0,
                              flow_threshold: float = 0.4,
                              min_size: int = 1) -> dict:
    """训练期 V1 batch 评估：使用真实 GT labels 作为 GT mask。

    GT label 经过与训练 image/flow 完全相同的 resize+pad 路径对齐到模型输出尺寸。
    mask_instance_f1 作为 best checkpoint 主判据。

    Args:
        pred_flows: (B, 3, H, W) float32, 预测 [dy, dx, cellprob] at stride 4
        gt_labels:  list of (H_orig, W_orig) int32, 真实实例标签（全分辨率）
        resize_infos: list of (scale, pad_top, pad_left, new_h, new_w)
        target_size: 模型输入尺寸 (默认 1024)
    """
    B = pred_flows.shape[0]
    s4 = target_size // 4
    if len(gt_labels) != B:
        raise ValueError(f"pred_flows batch={B} but gt_labels has {len(gt_labels)} items")
    metrics_list = []

    for b in range(B):
        pred_dP = np.stack([
            pred_flows[b, 0].astype(np.float64),
            pred_flows[b, 1].astype(np.float64),
        ], axis=0)
        pred_cp = pred_flows[b, 2].astype(np.float32)

        pred_masks = _unwrap(compute_masks(
            pred_dP, pred_cp,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
        ))

        # GT label 对齐：与 _resize_flow 使用相同的 resize+pad 路径
        gt_mask = gt_labels[b]
        scale, pad_top, pad_left, new_h, new_w = resize_infos[b]
        pad_h = target_size - new_h
        pad_w = target_size - new_w

        # 等比缩放
        gt_resized = cv2.resize(
            gt_mask.astype(np.float32), (new_w, new_h),
            interpolation=cv2.INTER_NEAREST)
        # 中心 padding
        gt_padded = cv2.copyMakeBorder(
            gt_resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
            cv2.BORDER_CONSTANT, value=0)
        # 降采样到 stride-4（与 cellprob 使用相同的 INTER_NEAREST）
        gt_mask_s4 = cv2.resize(
            gt_padded, (s4, s4),
            interpolation=cv2.INTER_NEAREST).astype(gt_mask.dtype)

        metrics = evaluate_masks_v1(pred_masks, gt_mask_s4, small_diameter_threshold=8.0)
        metrics['equivalent_diameter_mae_px'] *= OUTPUT_STRIDE
        metrics_list.append(metrics)

    avg = {}
    keys = [
        'pix_iou', 'mask_iou_mean', 'boundary_iou_mean',
        'recall', 'precision', 'instance_f1',
        'small_particle_recall_lt32px',
        'equivalent_diameter_mae_px',
        'area_absolute_relative_error_mean',
        'false_positive_count', 'false_negative_count',
        'over_split_proxy_count', 'n_pred', 'n_gt',
    ]
    for k in keys:
        vs = [m[k] for m in metrics_list if k in m]
        avg[k] = float(np.mean(vs)) if vs else 0.0
    avg['count_error'] = float(abs(avg.get('n_pred', 0) - avg.get('n_gt', 0)))
    avg['mean_area_error'] = avg.get('area_absolute_relative_error_mean', 0.0)
    return avg


def evaluate_batch_v1(pred_flows: np.ndarray, gt_flows: np.ndarray,
                      cellprob_threshold: float = 0.0,
                      flow_threshold: float = 0.4,
                      min_size: int = 1) -> dict:
    """训练期 V1 batch 评估：在与 evaluate_batch 相同的 stride-4 分辨率上
    运行 compute_masks()，但使用 evaluate_masks_v1() 计算完整的 V1 实例指标。

    Args:
        pred_flows: (B, 3, H, W) float32, 预测 [dy, dx, cellprob]
        gt_flows:   (B, 3, H, W) float32, GT [cellprob, dy, dx]
    """
    B = pred_flows.shape[0]
    metrics_list = []

    for b in range(B):
        pred_dP = np.stack([
            pred_flows[b, 0].astype(np.float64),
            pred_flows[b, 1].astype(np.float64),
        ], axis=0)
        pred_cp = pred_flows[b, 2].astype(np.float32)

        gt_dP = 5.0 * np.stack([
            gt_flows[b, 1].astype(np.float64),
            gt_flows[b, 2].astype(np.float64),
        ], axis=0)
        gt_cp = gt_flows[b, 0].astype(np.float32)

        pred_masks = _unwrap(compute_masks(
            pred_dP, pred_cp,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
        ))
        gt_masks = _unwrap(compute_masks(
            gt_dP, gt_cp,
            cellprob_threshold=0.0,
            flow_threshold=flow_threshold,
            min_size=min_size,
        ))

        # stride-4 下小颗粒阈值按比例缩小: 32 px → 8 px @ stride 4
        metrics = evaluate_masks_v1(pred_masks, gt_masks, small_diameter_threshold=8.0)
        # 直径类指标从 stride-4 像素乘回原始像素，与文档 V1 门槛可比
        metrics['equivalent_diameter_mae_px'] *= OUTPUT_STRIDE
        metrics['psd_d10_bias_px'] *= OUTPUT_STRIDE
        metrics['psd_d50_bias_px'] *= OUTPUT_STRIDE
        metrics['psd_d90_bias_px'] *= OUTPUT_STRIDE
        # 补充旧版 evaluate_batch 兼容字段
        metrics['count_error'] = float(abs(metrics['n_pred'] - metrics['n_gt']))
        metrics['mean_area_error'] = metrics['area_absolute_relative_error_mean']
        metrics_list.append(metrics)

    avg = {}
    keys = [
        'pix_iou', 'mask_iou_mean', 'boundary_iou_mean',
        'recall', 'precision', 'instance_f1',
        'small_particle_recall_lt32px',
        'equivalent_diameter_mae_px',
        'area_absolute_relative_error_mean',
        'over_split_proxy_count',
        'psd_d10_bias_px', 'psd_d50_bias_px', 'psd_d90_bias_px',
        'count_error', 'mean_area_error',
        'n_pred', 'n_gt',
    ]
    for k in keys:
        avg[k] = float(np.mean([m[k] for m in metrics_list]))
    return avg
