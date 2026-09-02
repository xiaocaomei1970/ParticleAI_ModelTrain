"""FlowDynamics 对齐测试 v3 — 严格匹配 Cellpose compute_masks 内部逻辑。

默认模式使用 Cellpose CPSAM 模型的 flow field 输出作为输入，验证 C++
FlowDynamics 后处理与 Cellpose compute_masks 的后处理一致性。
训练后模式使用 --checkpoint 加载本项目训练出的 best.pth，验证真实训练
模型输出在 Python/Cellpose 后处理和 C++ FlowDynamics 后处理之间的一致性。

关键发现 (来自 compute_masks 源码):
  p_final = follow_flows(dP * (cellprob > cellprob_threshold) / 5., inds=inds, ...)

即:
  1. dP 先 /5.0
  2. 背景像素 (cellprob <= threshold) 的 dP 置零
  3. cellprob threshold 在概率空间 (Cellpose 默认 0.0)

用法:
    python -m model_flow.align_flow_dynamics --samples alignment_samples.txt --img-dir data/particles/train --limit 3
    python -m model_flow.align_flow_dynamics --checkpoint checkpoints/best.pth --img-dir data/particles/val --limit 20 --use-cpp-alignment
"""
import os
import argparse
import time
import json

import cv2
import numpy as np
import torch

from ..config import Config
from ..eval_masks import _unwrap
from ..flow_head import FlowModel
from ..utils import imread_unicode, imread_unchanged, imwrite_unicode


# ═══════════════════════════════════════════════════════════════
# Three-layer alignment architecture:
#   Layer 1 — Cellpose reference: official compute_masks (oracle)
#   Layer 2 — FlowDynamics core: Euler + get_masks_torch + max_size + flow_qc + fill/min_size
#   Layer 3 — FlowDynamics deployment extras: mean logit + boundary policy (project-specific)
# ═══════════════════════════════════════════════════════════════

def compute_cellpose_reference(dP, cellprob_prob, niter=200,
                                flow_threshold=0.4, min_size=50,
                                max_size_fraction=0.5):
    """Layer 1: Official Cellpose compute_masks — oracle / ground truth.

    Args:
        dP: (2, H, W) raw flow field.
        cellprob_prob: (H, W) probability values (NOT logits), i.e. sigmoid output.
                       Threshold 0.5 is used to match logit>0.0 semantics.
        max_size_fraction: passed to Cellpose to match project parameter.
    """
    from cellpose.dynamics import compute_masks as cm
    return _unwrap(cm(dP, cellprob_prob, niter=niter,
                      cellprob_threshold=0.5,  # prob space: 0.5 ⇔ logit 0.0
                      flow_threshold=flow_threshold,
                      min_size=min_size,
                      max_size_fraction=max_size_fraction,
                      device=torch.device('cpu')))


def compute_flowdynamics_python_equiv(
        dy, dx, cellprob_prob, niter,
        flow_threshold, min_size,
        cellprob_threshold_logit,
        max_size_fraction,
        boundary_particle_policy='include',
        edge_touch_margin_px=1.0):
    """Layer 2+3: Mirror of C++ FlowDynamics complete deployment path.

    Order: Euler+get_masks → max_size → remove_bad_flow_masks → fill+min_size
           → mean logit → boundary (debug-only; use --use-cpp-alignment for release).
    """
    from .euler_core import (euler_integrate_to_labels, remove_large_masks,
                              fill_holes_labels, filter_by_size_and_boundary,
                              filter_by_mean_logit)
    import numpy as np

    cellprob_logit = np.log(np.maximum(cellprob_prob, 1e-7) /
                            np.maximum(1.0 - cellprob_prob, 1e-7))
    fg_mask = (cellprob_logit > 0.0)

    # --- Layer 2: FlowDynamics core (align to Cellpose) ---
    labels = euler_integrate_to_labels(dy, dx, fg_mask, niter)
    labels = remove_large_masks(labels, max_size_fraction)

    if flow_threshold > 0 and labels.max() > 0:
        from cellpose.dynamics import remove_bad_flow_masks
        dP_raw = np.stack([dy, dx], axis=0)
        labels = remove_bad_flow_masks(labels, dP_raw,
                                        threshold=flow_threshold,
                                        device=torch.device('cpu'))

    labels = fill_holes_labels(labels)

    # min_size after fill (Cellpose: fill_holes_and_remove_small_masks)
    if min_size > 0 and labels.max() > 0:
        for lbl in np.unique(labels):
            if lbl == 0: continue
            if (labels == lbl).sum() < min_size:
                labels[labels == lbl] = 0

    # --- Layer 3: FlowDynamics deployment extras (project-specific) ---
    labels = filter_by_mean_logit(labels, cellprob_logit, cellprob_threshold_logit)
    labels = filter_by_size_and_boundary(
        labels, min_size=0, max_size_fraction=max_size_fraction,
        boundary_particle_policy=boundary_particle_policy,
        edge_touch_margin_px=edge_touch_margin_px)

    return labels


# ═══════════════════════════════════════════════════════════════
# 对齐测试
# ═══════════════════════════════════════════════════════════════

def compute_masks_cellpose(dP, cellprob_prob, niter=200, cellprob_threshold=0.0,
                           flow_threshold=0.4, min_size=50, max_size_fraction=0.5):
    """调用 Cellpose 官方 compute_masks (已导出)."""
    from cellpose.dynamics import compute_masks as cm
    return _unwrap(cm(dP, cellprob_prob, niter=niter,
                      cellprob_threshold=cellprob_threshold,
                      flow_threshold=flow_threshold,
                      min_size=min_size,
                      max_size_fraction=max_size_fraction,
                      device=torch.device('cpu')))


def load_flow_params(path, cfg):
    """读取 tune_flow_dynamics.py 输出的 best_params，未提供时使用 Config 默认值。"""
    params = {
        'niter': cfg.fd_niter,
        'cellprob_threshold_logit': cfg.fd_cellprob_threshold,
        'cellprob_threshold_probability': cfg.inference_cellprob_threshold,
        'flow_threshold': cfg.fd_flow_threshold,
        'min_size': cfg.fd_min_size,
        'max_size_fraction': cfg.fd_max_size_fraction,
    }
    if not path:
        return params

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    best = data.get('best_params', data)
    if 'cellprob_threshold_logit' in best:
        params['cellprob_threshold_logit'] = float(best['cellprob_threshold_logit'])
    if 'flow_threshold' in best:
        params['flow_threshold'] = float(best['flow_threshold'])
    if 'min_size' in best:
        params['min_size'] = int(best['min_size'])
    if 'niter' in best:
        params['niter'] = int(best['niter'])
    if 'max_size_fraction' in best:
        params['max_size_fraction'] = float(best['max_size_fraction'])
    return params


def collect_samples(img_dir, samples_path, files_arg, limit):
    """收集待对齐样本。优先 files，其次 samples 文件，最后从 img_dir 自动取图。"""
    if files_arg:
        samples = [x.strip() for x in files_arg.split(',') if x.strip()]
    elif samples_path and os.path.exists(samples_path):
        with open(samples_path, 'r', encoding='utf-8') as f:
            samples = [line.strip() for line in f if line.strip()]
    else:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        samples = sorted([
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])

    if limit > 0:
        samples = samples[:limit]
    return samples


def preprocess_for_model(img_bgr, cfg):
    """与 inference.py 一致的训练模型预处理。"""
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
    img = (img - np.array(cfg.mean, dtype=np.float32)) / np.array(cfg.std, dtype=np.float32)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor, (pad_left, pad_top, new_h, new_w)


def load_trained_model(checkpoint, cfg):
    """加载训练 checkpoint，返回 eval 模型。"""
    model = FlowModel(cfg)
    model = model.to(cfg.device)
    ckpt = torch.load(checkpoint, map_location=cfg.device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.neck.load_state_dict(ckpt['neck'])
        model.flow_head.load_state_dict(ckpt['flow_head'])
    model.eval()
    return model


def infer_trained_model_flow(model, img_bgr, cfg):
    """用训练模型生成 padded input_size 空间的 dy/dx/cellprob logits。"""
    tensor, (pad_left, pad_top, new_h, new_w) = preprocess_for_model(img_bgr, cfg)
    tensor = tensor.to(cfg.device)

    with torch.no_grad():
        flow_s4 = model(tensor)[0].cpu().numpy()

    flow_full = np.zeros((3, cfg.input_size, cfg.input_size), dtype=np.float32)
    for c in range(3):
        flow_full[c] = cv2.resize(flow_s4[c], (cfg.input_size, cfg.input_size),
                                  interpolation=cv2.INTER_LINEAR)

    dy = flow_full[0]
    dx = flow_full[1]
    cellprob_logit = flow_full[2]

    pad_mask = np.zeros((cfg.input_size, cfg.input_size), dtype=np.float32)
    pad_mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = 1.0
    cellprob_logit[pad_mask < 0.5] = -100.0
    return dy, dx, cellprob_logit


def pixel_iou(a, b):
    a_b = (a > 0).astype(np.int32); b_b = (b > 0).astype(np.int32)
    inter = (a_b & b_b).sum(); union = (a_b | b_b).sum()
    return inter / union if union > 0 else 1.0


def instance_stats(a, b):
    ids_a = set(np.unique(a)) - {0}; ids_b = set(np.unique(b)) - {0}
    if not ids_a and not ids_b: return 1.0, 1.0, len(ids_a), len(ids_b)
    if not ids_a or not ids_b: return 0.0, 0.0, len(ids_a), len(ids_b)
    matches = 0
    for ia in ids_a:
        ma = (a == ia).astype(np.uint8)
        best = 0.0
        for ib in ids_b:
            mb = (b == ib).astype(np.uint8)
            inter = (ma & mb).sum(); union = (ma | mb).sum()
            if union > 0: best = max(best, inter / union)
        if best > 0.5: matches += 1
    recall = matches / len(ids_a)
    precision = matches / len(ids_b) if ids_b else 0
    return recall, precision, len(ids_a), len(ids_b)


def _run_real_cpp_alignment(cpp_align_exe, dy, dx, cellprob_logit,
                           niter, flow_threshold, min_size,
                           cellprob_threshold_logit, max_size_fraction,
                           out_dir, base_name, **kwargs):
    """P0-2: 调用真实 C++ flow_dynamics_align.exe 进行对齐验证。

    将 dy/dx/cellprob 写为 float32 二进制文件，调用 C++ 可执行文件，
    读取输出 PNG mask 并返回 labels。

    kwargs:
        keep_temp: bool, 保留中间文件 (默认 False)
        timeout: int, 超时秒数 (默认 300)
    """
    import subprocess
    import tempfile

    H, W = dy.shape

    # 创建临时目录存放中间文件 (使用项目 temp/ 目录)
    tmp_dir = os.path.join(os.getcwd(), 'temp', 'cpp_align', base_name)
    os.makedirs(tmp_dir, exist_ok=True)

    dy_bin = os.path.join(tmp_dir, f'{base_name}_dy.bin')
    dx_bin = os.path.join(tmp_dir, f'{base_name}_dx.bin')
    cp_bin = os.path.join(tmp_dir, f'{base_name}_cp.bin')
    out_png = os.path.join(tmp_dir, f'{base_name}_cpp_mask.png')

    # 写入 float32 二进制文件
    dy.astype(np.float32).tofile(dy_bin)
    dx.astype(np.float32).tofile(dx_bin)
    cellprob_logit.astype(np.float32).tofile(cp_bin)

    # 自动检测 C++ 可执行文件路径
    if not cpp_align_exe:
        # 搜索常见构建目录
        inference_cpp_dir = os.path.join(
            os.path.dirname(__file__), '..', 'inference_cpp')
        candidates = [
            os.path.join(inference_cpp_dir, 'build', 'Release',
                         'flow_dynamics_align.exe'),
            os.path.join(inference_cpp_dir, 'build', 'Debug',
                         'flow_dynamics_align.exe'),
            os.path.join(os.getcwd(), 'temp', 'review_build_inference', 'Release',
                         'flow_dynamics_align.exe'),
            os.path.join(os.getcwd(), 'temp', 'review_build_inference', 'Debug',
                         'flow_dynamics_align.exe'),
        ]
        for c in candidates:
            if os.path.exists(c):
                cpp_align_exe = c
                break

    if not cpp_align_exe or not os.path.exists(cpp_align_exe):
        raise FileNotFoundError(
            f'C++ alignment executable not found: {cpp_align_exe}\n'
            f'Build it with: cd model_flow/inference_cpp && mkdir build && cd build && '
            f'cmake .. && cmake --build . --config Release\n'
            f'Or specify path with --cpp-align-exe')

    # 调用 C++ 可执行文件
    cmd = [
        cpp_align_exe,
        dy_bin, dx_bin, cp_bin,
        str(H), str(W),
        out_png,
        '--niter', str(niter),
        '--flow_threshold', str(flow_threshold),
        '--min_size', str(min_size),
        '--cellprob_threshold', str(cellprob_threshold_logit),
        '--max_size_fraction', str(max_size_fraction),
    ]
    keep_temp = kwargs.get('keep_temp', False)
    timeout_sec = kwargs.get('timeout', 300)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                encoding='utf-8', errors='replace',
                                timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        print(f'  WARNING: C++ alignment timed out ({timeout_sec}s). '
              f'Intermediate files kept at: {tmp_dir}')
        raise RuntimeError(
            f'C++ alignment timed out after {timeout_sec}s. '
            f'Output kept at: {out_png}')
    except Exception:
        print(f'  WARNING: C++ alignment failed. '
              f'Intermediate files kept at: {tmp_dir}')
        raise

    if result.returncode != 0:
        print(f'  WARNING: C++ alignment failed (exit={result.returncode}). '
              f'Intermediate files kept at: {tmp_dir}')
        raise RuntimeError(f'C++ alignment failed:\n{result.stderr}')

    # 读取输出 mask
    cpp_mask = imread_unchanged(out_png)
    if cpp_mask is None:
        raise RuntimeError(f'C++ alignment produced no output: {out_png}')

    # 清理临时文件 (成功时, 除非 --keep-temp)
    if not keep_temp:
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
    else:
        print(f'  Intermediate files kept at: {tmp_dir}')

    if cpp_mask.dtype != np.uint16:
        cpp_mask = cpp_mask.astype(np.int32)
    return cpp_mask.astype(np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', default='alignment_samples.txt')
    parser.add_argument('--files', default='',
                        help='逗号分隔的样本文件名列表，优先级高于 --samples')
    parser.add_argument('--img-dir', default='data/particles/train')
    parser.add_argument('--out-dir', default='alignment_results')
    parser.add_argument('--report', default='',
                        help='JSON alignment report path. Defaults to <out-dir>/alignment_report.json')
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--checkpoint', default='',
                        help='训练后的 best.pth；提供后使用训练模型输出做对齐验证')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--flow-params', default='',
                        help='tune_flow_dynamics.py 输出的 flow_dynamics_best_params.json')
    parser.add_argument('--use-cpp-alignment', action='store_true',
                        help='P0-2: 调用编译后的 flow_dynamics_align.exe 进行真实 C++ 对齐验证')
    parser.add_argument('--cpp-align-exe', default='',
                        help='C++ 对齐测试可执行文件路径 (默认 auto-detect)')
    parser.add_argument('--keep-temp', action='store_true',
                        help='保留 C++ 对齐中间文件 (dy.bin/dx.bin/cp.bin/cpp_mask.png)')
    parser.add_argument('--timeout', type=int, default=300,
                        help='C++ 对齐超时秒数 (默认 300)')
    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device
    flow_params = load_flow_params(args.flow_params, cfg)
    samples = collect_samples(args.img_dir, args.samples, args.files, args.limit)

    if args.checkpoint:
        print(f'Loading trained FlowModel: {args.checkpoint}')
        model = load_trained_model(args.checkpoint, cfg)
        source_label = 'trained model'
    else:
        print(f'Loading CPSAM model...')
        from cellpose import models
        cpsam = models.CellposeModel(gpu=False, model_type='cpsam')
        model = None
        source_label = 'CPSAM'

    print(f'Testing {len(samples)} images.\n')
    os.makedirs(args.out_dir, exist_ok=True)

    all_ious = []
    all_diff_ratios = []
    all_instance_diffs = []
    per_image_reports = []

    for fname in samples:
        print(f'\n── {fname} ──')
        img_path = os.path.join(args.img_dir, fname)
        img = imread_unicode(img_path)
        if img is None: continue

        if model is not None:
            print(f'  FlowModel inference...')
            t0 = time.time()
            dy, dx, cp_raw = infer_trained_model_flow(model, img, cfg)
            print(f'  FlowModel: {time.time()-t0:.1f}s')
        else:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f'  CPSAM...')
            t0 = time.time()
            _, flows, _ = cpsam.eval(img_gray, channels=[0, 0],
                                     diameter=0, compute_masks=False)
            print(f'  CPSAM: {time.time()-t0:.0f}s')

            dy = flows[1][0].astype(np.float32)
            dx = flows[1][1].astype(np.float32)
            cp_raw = flows[2].astype(np.float32)
            if cp_raw.ndim > 2:
                cp_raw = cp_raw.squeeze()

            max_side = max(dy.shape)
            if max_side > 512:
                s = 512.0 / max_side
                dy = cv2.resize(dy, (int(dy.shape[1]*s), int(dy.shape[0]*s)), cv2.INTER_LINEAR)
                dx = cv2.resize(dx, (int(dx.shape[1]*s), int(dx.shape[0]*s)), cv2.INTER_LINEAR)
                cp_raw = cv2.resize(cp_raw, (int(cp_raw.shape[1]*s), int(cp_raw.shape[0]*s)), cv2.INTER_LINEAR)
                print(f'  Resized: {dy.shape[1]}x{dy.shape[0]}')

        dP = np.stack([dy, dx], axis=0)
        cp_prob = 1.0 / (1.0 + np.exp(-cp_raw))  # logit → prob
        cellprob_logit_core = cp_raw  # raw logits for foreground mask
        print(f'  cp_prob: min={cp_prob.min():.4f} max={cp_prob.max():.4f} mean={cp_prob.mean():.4f}')
        print(f'  cp_raw:  min={cp_raw.min():.2f} max={cp_raw.max():.2f}')

        # ── Layer 1: Cellpose reference (oracle) ──
        print('  Cellpose reference (oracle)...')
        t0 = time.time()
        masks_ref = compute_cellpose_reference(
            dP, cp_prob,
            niter=flow_params['niter'],
            flow_threshold=flow_params['flow_threshold'],
            min_size=flow_params['min_size'],
            max_size_fraction=flow_params['max_size_fraction'])
        t_ref = time.time() - t0
        n_ref = len(np.unique(masks_ref)) - 1
        print(f'  Reference: {n_ref} instances ({t_ref:.0f}s)')

        # ── Layer 2+3: C++ FlowDynamics or Python mirror ──
        if args.use_cpp_alignment:
            backend_label = 'C++ FlowDynamics (real_cpp)'
            print(f'  {backend_label}...')
            masks_candidate = _run_real_cpp_alignment(
                cpp_align_exe=args.cpp_align_exe,
                dy=dy, dx=dx, cellprob_logit=cp_raw,
                niter=flow_params['niter'],
                flow_threshold=flow_params['flow_threshold'],
                min_size=flow_params['min_size'],
                cellprob_threshold_logit=flow_params['cellprob_threshold_logit'],
                max_size_fraction=flow_params['max_size_fraction'],
                out_dir=args.out_dir, base_name=os.path.splitext(fname)[0],
                keep_temp=args.keep_temp, timeout=args.timeout)
            t_candidate = 0.0
            n_candidate = len(np.unique(masks_candidate)) - 1
            backend = 'real_cpp'
        else:
            backend_label = 'FlowDynamics Python mirror (debug_only)'
            backend = 'python_mirror'
            print(f'  {backend_label}...')
            t0 = time.time()
            masks_candidate = compute_flowdynamics_python_equiv(
                dy, dx, cp_prob,
                niter=flow_params['niter'],
                flow_threshold=flow_params['flow_threshold'],
                min_size=flow_params['min_size'],
                cellprob_threshold_logit=flow_params['cellprob_threshold_logit'],
                max_size_fraction=flow_params['max_size_fraction'])
            t_candidate = time.time() - t0
            n_candidate = len(np.unique(masks_candidate)) - 1
        print(f'  {backend_label}: {n_candidate} instances ({t_candidate:.0f}s)')

        # ── core_vs_cellpose: FlowDynamics core (stop before deployment extras) ──
        from .euler_core import euler_integrate_to_labels, remove_large_masks, fill_holes_labels
        fg_core = (cellprob_logit_core > 0.0)
        masks_core = euler_integrate_to_labels(dy, dx, fg_core, flow_params['niter'])
        masks_core = remove_large_masks(masks_core, flow_params['max_size_fraction'])
        if flow_params['flow_threshold'] > 0 and masks_core.max() > 0:
            from cellpose.dynamics import remove_bad_flow_masks
            dP_raw = np.stack([dy, dx], axis=0)
            masks_core = remove_bad_flow_masks(masks_core, dP_raw,
                                                threshold=flow_params['flow_threshold'],
                                                device=torch.device('cpu'))
        masks_core = fill_holes_labels(masks_core)
        if flow_params['min_size'] > 0 and masks_core.max() > 0:
            for lbl in np.unique(masks_core):
                if lbl == 0: continue
                if (masks_core == lbl).sum() < flow_params['min_size']:
                    masks_core[masks_core == lbl] = 0
        n_core = len(np.unique(masks_core)) - 1

        core_pix_iou = pixel_iou(masks_ref, masks_core)
        core_recall, core_precision, _, _ = instance_stats(masks_ref, masks_core)
        core_diff = int((masks_ref > 0) != (masks_core > 0)).sum()

        # ── deployment_vs_cellpose: full mirror incl project extras ──
        pix_iou = pixel_iou(masks_ref, masks_candidate)
        recall, precision, _, _ = instance_stats(masks_ref, masks_candidate)
        all_ious.append(pix_iou)

        print(f'  core_vs_cellpose:     PixIoU={core_pix_iou:.4f} Recall={core_recall:.3f} Prec={core_precision:.3f} ref={n_ref} core={n_core}')
        print(f'  deployment_vs_ref:    PixIoU={pix_iou:.4f} Recall={recall:.3f} Prec={precision:.3f} ref={n_ref} candidate={n_candidate}')

        ref_b = (masks_ref > 0); candidate_b = (masks_candidate > 0)
        diff = (ref_b != candidate_b).sum()
        diff_ratio = 100 * diff / masks_ref.size
        instance_diff = abs(int(n_ref) - int(n_candidate))
        all_diff_ratios.append(diff_ratio)
        all_instance_diffs.append(instance_diff)

        diff_image_path = ''
        if diff > 0:
            viz = np.zeros((*masks_ref.shape, 3), dtype=np.uint8)
            viz[ref_b & candidate_b, 1] = 255
            viz[ref_b & ~candidate_b, 2] = 255
            viz[~ref_b & candidate_b, 0] = 255
            out_p = os.path.join(args.out_dir, os.path.splitext(fname)[0] + '_diff.png')
            imwrite_unicode(out_p, viz)
            diff_image_path = out_p
            print(f'  Diff image: {out_p}')

        per_image_reports.append({
            'file': fname,
            'source': source_label,
            'backend': backend,
            'reference_instances': int(n_ref),
            'candidate_instances': int(n_candidate),
            'core_instances': int(n_core),
            'pixel_iou': float(pix_iou),
            'recall': float(recall),
            'precision': float(precision),
            'diff_pixels': int(diff),
            'diff_ratio_percent': float(diff_ratio),
            'instance_diff': int(instance_diff),
            'core_pixel_iou': float(core_pix_iou),
            'core_recall': float(core_recall),
            'core_precision': float(core_precision),
            'diff_image': diff_image_path,
        })

    print(f'\n{"="*60}')
    if not all_ious:
        print('No images evaluated.')
        report_path = args.report or os.path.join(args.out_dir, 'alignment_report.json')
        with open(report_path, 'w', encoding='utf-8') as handle:
            json.dump({
                'source': source_label,
                'backend': backend if args.use_cpp_alignment else 'python_mirror',
                'image_count': 0,
                'flow_params': flow_params,
                'summary': {'failed': True, 'reason': 'no_images_evaluated'},
                'failed_samples': [],
                'images': [],
            }, handle, ensure_ascii=False, indent=2)
        print(f'  Report: {report_path}')
        return
    print(f'Summary ({len(all_ious)} images, source={source_label}): '
          f'Mean pixel IoU = {np.mean(all_ious):.4f}')
    below_iou = sum(1 for i in all_ious if i < 0.95)
    high_diff = sum(1 for r in all_diff_ratios if r >= 0.1)
    high_inst = sum(1 for d in all_instance_diffs if d > 2)
    print(f'  PixIoU < 0.95: {below_iou}/{len(all_ious)}')
    print(f'  Diff pixels >= 0.1%: {high_diff}/{len(all_ious)}')
    print(f'  Instance diff > 2: {high_inst}/{len(all_ious)}')
    failed = below_iou > 0 or high_diff > 0 or high_inst > 0
    print(f'  {"WARNING: needs fixing" if failed else "Alignment OK"}.')

    report_path = args.report or os.path.join(args.out_dir, 'alignment_report.json')
    report = {
        'source': source_label,
        'backend': backend if args.use_cpp_alignment else 'python_mirror',
        'image_count': len(all_ious),
        'flow_params': flow_params,
        'summary': {
            'mean_pixel_iou': float(np.mean(all_ious)),
            'mean_diff_ratio_percent': float(np.mean(all_diff_ratios)),
            'mean_instance_diff': float(np.mean(all_instance_diffs)),
            'pix_iou_below_0_95_count': int(below_iou),
            'diff_pixels_ge_0_1_percent_count': int(high_diff),
            'instance_diff_gt_2_count': int(high_inst),
            'failed': bool(failed),
        },
        'failed_samples': [
            row for row in per_image_reports
            if row['pixel_iou'] < 0.95
            or row['diff_ratio_percent'] >= 0.1
            or row['instance_diff'] > 2
        ],
        'images': per_image_reports,
    }
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f'  Report: {report_path}')


if __name__ == '__main__':
    main()
