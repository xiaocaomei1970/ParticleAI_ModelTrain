"""GT Flow Field 生成：从实例标签图 → (cellprob, dy, dx)。

使用 Cellpose 的 labels_to_flows() 进行热扩散 PDE 求解。
"""
import cv2
import numpy as np
import torch
from cellpose.dynamics import labels_to_flows


def _segmentation_to_mask(ann: dict, h: int, w: int) -> np.ndarray | None:
    """将单个 COCO annotation 的 segmentation 转为 (H, W) 二值 mask。

    支持 polygon list、RLE（压缩/非压缩）、空 segmentation（bbox fallback）。
    iscrowd=1 的 crowd 标注跳过，返回 None。
    """
    if ann.get('iscrowd', 0) == 1:
        return None  # crowd 标注不参与实例 mask 生成

    segm = ann.get('segmentation', [])

    # ── polygon list ──
    if isinstance(segm, list):
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in segm:
            if len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        if mask.sum() == 0:
            # polygon 全部无效，回退到 bbox
            bbox = ann.get('bbox', [])
            if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                x, y, bw, bh = map(int, bbox)
                x = max(0, x); y = max(0, y)
                bw = min(bw, w - x); bh = min(bh, h - y)
                mask[y:y + bh, x:x + bw] = 1
        return mask

    # ── RLE (dict with 'counts' and 'size') ──
    if isinstance(segm, dict) and 'counts' in segm:
        counts = segm.get('counts', [])
        try:
            from pycocotools import mask as cocomask
            # P2-F6: counts 类型显式分支
            if isinstance(counts, (str, bytes)):
                return cocomask.decode(segm).astype(np.uint8)
            elif isinstance(counts, list):
                rle = cocomask.frPyObjects(segm, h, w)
                if isinstance(rle, list):
                    rle = cocomask.merge(rle)
                return cocomask.decode(rle).astype(np.uint8)
            else:
                return None
        except ImportError:
            pass
        if isinstance(counts, list):
            mask = np.zeros(h * w, dtype=np.uint8)
            pos = 0
            val = 0
            for cnt in counts:
                mask[pos:pos + cnt] = val
                pos += cnt
                val = 1 - val
            return mask.reshape((h, w), order='F')
        elif isinstance(counts, (str, bytes)):
            raise ImportError(
                "Compressed RLE detected but pycocotools is not installed. "
                "Install it via: pip install pycocotools"
            )
        return None

    # ── 空 segmentation，bbox fallback ──
    bbox = ann.get('bbox', [])
    if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
        x, y, bw, bh = map(int, bbox)
        mask = np.zeros((h, w), dtype=np.uint8)
        x = max(0, x); y = max(0, y)
        bw = min(bw, w - x); bh = min(bh, h - y)
        mask[y:y + bh, x:x + bw] = 1
        return mask

    return None


def generate_flow_field(label_map: np.ndarray) -> np.ndarray:
    """从实例标签图生成 flow field。

    Args:
        label_map: (H, W) int32, 0=背景, 1..N=实例ID

    Returns:
        flow: (3, H, W) float32
            flow[0] = cellprob (binary foreground mask, 0/1; P3-2: 与 Cellpose 官方训练一致)
            flow[1] = dy (垂直流场, [-1,1])
            flow[2] = dx (水平流场, [-1,1])
    """
    if label_map.max() == 0:
        h, w = label_map.shape
        return np.zeros((3, h, w), dtype=np.float32)

    flows = labels_to_flows(
        [label_map.astype(np.int32)],
        device=torch.device('cpu'),
    )

    # flows[0] shape: (4, H, W)
    #   channel 0: original labels
    #   channel 1: cell probability / binary foreground mask (P3-2: 修正注释)
    #   channel 2: dy (垂直流场)
    #   channel 3: dx (水平流场)
    f = flows[0]

    # 二值化前景掩码: 与 Cellpose 官方训练对齐
    # 官方 _loss_fn_seg: criterion2(y[:,-1], (lbl[:,-3] > 0.5).to(y.dtype))
    cellprob = (f[1] > 0.5).astype(np.float32)
    dy = f[2].astype(np.float32)
    dx = f[3].astype(np.float32)

    return np.stack([cellprob, dy, dx], axis=0)


def generate_flow_from_coco(anns: list, h: int, w: int) -> np.ndarray:
    """从 COCO annotation 列表生成 flow field。

    支持 polygon / RLE / bbox fallback 等多种 segmentation 格式。
    iscrowd=1 的 crowd 标注自动跳过。

    Args:
        anns: COCO annotations for one image
        h, w: 图像尺寸

    Returns:
        flow: (3, H, W) float32
    """
    label_map = np.zeros((h, w), dtype=np.int32)
    instance_id = 0

    for ann in anns:
        mask = _segmentation_to_mask(ann, h, w)
        if mask is None or mask.sum() == 0:
            continue
        instance_id += 1
        label_map[mask > 0] = instance_id

    if label_map.max() == 0:
        return np.zeros((3, h, w), dtype=np.float32)

    return generate_flow_field(label_map)
