"""Flow Field 推理脚本 (PyTorch + FlowDynamics)

用法:
    python -m model_flow.inference test.jpg --checkpoint checkpoints/best.pth
    python -m model_flow.inference test.jpg --checkpoint checkpoints/best.pth --out result.png
"""
import os
import sys
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .flow_head import FlowModel
from .utils import imread_unicode


def preprocess(img_bgr, cfg):
    """预处理: resize, pad 114, BGR→RGB, /255, normalize"""
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

    # BGR → RGB, float32, /255, normalize
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    mean_arr = np.array(cfg.mean, dtype=np.float32)
    std_arr = np.array(cfg.std, dtype=np.float32)
    img = (img - mean_arr) / std_arr

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor, (scale, pad_left, pad_top)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input image path')
    parser.add_argument('--checkpoint', default='checkpoints/best.pth')
    parser.add_argument('--out', '-o', help='Output visualization path')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device

    # 加载模型
    print(f'Loading model from {args.checkpoint}...')
    t0 = time.time()
    model = FlowModel(cfg)
    model = model.to(cfg.device)
    ckpt = torch.load(args.checkpoint, map_location=cfg.device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        # 兼容旧格式 checkpoint (仅 neck + flow_head)
        model.neck.load_state_dict(ckpt['neck'])
        model.flow_head.load_state_dict(ckpt['flow_head'])
    model.eval()
    print(f'Loaded in {time.time() - t0:.1f}s')

    # 加载并预处理图像
    img_bgr = imread_unicode(args.input)
    orig_h, orig_w = img_bgr.shape[:2]
    tensor, (scale, pad_left, pad_top) = preprocess(img_bgr, cfg)
    tensor = tensor.to(cfg.device)

    # 推理
    t0 = time.time()
    with torch.no_grad():
        flow_s4 = model(tensor)  # (1, 3, 256, 256)
    print(f'Inference: {time.time() - t0:.1f}s')

    # Upsample to full resolution
    t0 = time.time()
    flow_s4 = flow_s4[0].cpu().numpy()  # (3, 256, 256)
    flow_full = np.zeros((3, cfg.input_size, cfg.input_size), dtype=np.float32)
    for c in range(3):
        flow_full[c] = cv2.resize(flow_s4[c], (cfg.input_size, cfg.input_size),
                                   interpolation=cv2.INTER_LINEAR)

    # 提取 dy, dx, cellprob
    dy = flow_full[0]        # 垂直流场
    dx = flow_full[1]        # 水平流场
    cellprob = flow_full[2]  # cellprob (logits)

    # 屏蔽 padding 区域: 将 cellprob 设为极小值, 避免边界假颗粒
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    pad_mask = np.zeros((cfg.input_size, cfg.input_size), dtype=np.float32)
    pad_mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = 1.0
    # 将 pad 区域的 cellprob 设为 -100 (sigmoid ≈ 3.7e-44, 极不可能作为前景)
    cellprob[pad_mask < 0.5] = -100.0

    # FlowDynamics 后处理
    # 注: Python 推理使用 Cellpose compute_masks (包含 remove_bad_flow_masks),
    #     C++ 推理使用 FlowDynamics (现已包含等效 bad-flow 过滤)。
    #     两者均与 Cellpose 行为对齐，参数相同则结果一致。
    from cellpose.dynamics import compute_masks
    from .eval_masks import _unwrap

    dP = np.stack([dy, dx], axis=0)  # (2, H, W)
    # cellprob logits → probability
    cellprob_prob = 1.0 / (1.0 + np.exp(-cellprob))

    labels = _unwrap(compute_masks(
        dP, cellprob_prob,
        niter=cfg.fd_niter,
        cellprob_threshold=cfg.inference_cellprob_threshold,
        flow_threshold=cfg.fd_flow_threshold,
        min_size=cfg.fd_min_size,
        max_size_fraction=cfg.fd_max_size_fraction,
        device=torch.device('cpu'),
    ))
    print(f'FlowDynamics: {time.time() - t0:.1f}s')

    # ── 坐标逆映射: padded 1024² → 原图 (评审V3-#2) ──
    # 1) 裁剪 padding 区域 (new_h, new_w 已在上面计算)
    labels_cropped = labels[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    # 2) resize 到原图尺寸 (nearest 保持整数 ID)
    labels_orig = cv2.resize(labels_cropped, (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)

    n_particles = labels_orig.max()
    print(f'Detected: {n_particles} particles')

    # 可视化 (在原图上)
    if args.out and n_particles > 0:
        import matplotlib.pyplot as plt
        from matplotlib import patches

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(img_bgr[:, :, ::-1])
        colors = plt.cm.rainbow(np.linspace(0, 1, n_particles + 1))

        for inst_id in range(1, n_particles + 1):
            mask = (labels_orig == inst_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                best = max(contours, key=cv2.contourArea)
                color = colors[inst_id][:3]
                x, y, w, h = cv2.boundingRect(best)
                rect = patches.Rectangle(
                    (x, y), w, h, linewidth=1,
                    edgecolor=color, facecolor='none')
                ax.add_patch(rect)

        ax.set_title(f'{n_particles} particles')
        ax.axis('off')
        fig.savefig(args.out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {args.out}')

    print('Done.')


if __name__ == '__main__':
    main()
