"""Flow Field 数据集: 加载图片 + 预计算的 GT flow field (.npy)

数据增强: 水平/垂直翻转, 90° 倍旋转, 亮度/对比度抖动。
"""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import imread_unicode, long_path


class FlowDataset(Dataset):
    """加载图片和预计算的 GT flow field。

    目录结构:
        img_dir/
            image1.jpg
            image2.png
            ...
        flow_dir/
            image1.npy   # (3, H, W) float32: [cellprob, dy, dx]
            image2.npy
            ...
    """

    def __init__(self, img_dir: str, flow_dir: str,
                 input_size: int = 1024,
                 mean: tuple = (0.485, 0.456, 0.406),
                 std: tuple = (0.229, 0.224, 0.225),
                 pad_value: int = 114,
                 augment: bool = False,
                 aug_flip_prob: float = 0.5,
                 aug_rotate_prob: float = 0.3,
                 aug_jitter_prob: float = 0.3,
                 aug_scale_prob: float = 0.0,
                 aug_scale_range: tuple = (1.0, 1.0),
                 label_dir: str = "",):
        self.img_dir = img_dir
        self.flow_dir = flow_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.pad_value = pad_value
        self.augment = augment
        self.aug_flip_prob = aug_flip_prob
        self.aug_rotate_prob = aug_rotate_prob
        self.aug_jitter_prob = aug_jitter_prob
        self.aug_scale_prob = aug_scale_prob
        self.aug_scale_range = aug_scale_range

        # 收集 img_dir 下所有图片文件, 且对应的 .npy 文件存在
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        self.samples = []
        missing_flows = []
        missing_labels = []
        for fname in sorted(os.listdir(img_dir)):
            name, ext = os.path.splitext(fname)
            if ext.lower() in exts and not name.endswith('_labels'):
                npy_path = os.path.join(flow_dir, name + '.npy')
                if os.path.exists(npy_path):
                    label_path = ""
                    if label_dir:
                        label_path = os.path.join(label_dir, name + '_labels.png')
                        if not os.path.exists(label_path):
                            missing_labels.append(fname)
                            continue
                    self.samples.append((fname, npy_path, label_path))
                else:
                    missing_flows.append(fname)

        if missing_labels:
            raise RuntimeError(
                f'{len(missing_labels)} images in {img_dir} '
                f'have no matching *_labels.png in {label_dir}. '
                f'All val samples must have reviewed GT labels.')
        if missing_flows:
            msg = (f'{len(missing_flows)} images in {img_dir} '
                   f'have no matching .npy in {flow_dir}.')
            if len(missing_flows) <= 10:
                msg += '\n  ' + '\n  '.join(missing_flows)
            raise RuntimeError(
                f'{msg}\n'
                f'Run convert_labels_to_flows before training, or run '
                f'verify_training_pairs.py to diagnose.')

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image/npy pairs found in {img_dir} / {flow_dir}. "
                f"Run precompute_flows.py first."
            )

    def __len__(self):
        return len(self.samples)

    def _apply_augmentation(self, img_bgr, flow_full):
        """对图片和 flow field 同步做数据增强。

        flow_full: (3, H, W) float32, channel order [cellprob, dy, dx]
        """

        # ── 水平翻转 ──
        if random.random() < self.aug_flip_prob:
            img_bgr = cv2.flip(img_bgr, 1)
            flow_full[2] = -flow_full[2]  # dx 取反
            flow_full = flow_full[:, :, ::-1].copy()  # 翻转 + copy 避免负步长
            img_bgr = img_bgr.copy()

        # ── 垂直翻转 ──
        if random.random() < self.aug_flip_prob:
            img_bgr = cv2.flip(img_bgr, 0)
            flow_full[1] = -flow_full[1]  # dy 取反
            flow_full = flow_full[:, ::-1, :].copy()
            img_bgr = img_bgr.copy()

        # ── 旋转 90°/180°/270° ──
        if random.random() < self.aug_rotate_prob:
            k = random.randint(1, 3)  # 1=90°, 2=180°, 3=270°
            # 图像旋转
            if k == 1:
                img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
            elif k == 2:
                img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_180)
            else:
                img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # flow field 旋转 (cellprob 不变, dy/dx 向量同步旋转)
            cellprob = flow_full[0]
            dy = flow_full[1]
            dx = flow_full[2]

            if k == 1:  # 90° 顺时针: dy' = dx, dx' = -dy
                dy_rot = np.rot90(dx, 3).copy()
                dx_rot = -np.rot90(dy, 3).copy()
                cp_rot = np.rot90(cellprob, 3).copy()
            elif k == 2:  # 180°: dy' = -dy, dx' = -dx
                dy_rot = -np.rot90(dy, 2).copy()
                dx_rot = -np.rot90(dx, 2).copy()
                cp_rot = np.rot90(cellprob, 2).copy()
            else:  # 270° (逆时针90°): dy' = -dx, dx' = dy
                dy_rot = -np.rot90(dx, 1).copy()
                dx_rot = np.rot90(dy, 1).copy()
                cp_rot = np.rot90(cellprob, 1).copy()

            flow_full = np.stack([cp_rot, dy_rot, dx_rot], axis=0)

        # ── 亮度/对比度抖动 ──
        if random.random() < self.aug_jitter_prob:
            alpha = 1.0 + random.uniform(-0.15, 0.15)  # 对比度: 0.85 ~ 1.15
            beta = random.uniform(-25.5, 25.5)          # 亮度: -10% ~ +10% (BGR 0-255)
            img_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

        # ── 随机缩放 (P2-4) ──
        if (self.aug_scale_prob > 0 and
                self.aug_scale_range[0] < self.aug_scale_range[1] and
                random.random() < self.aug_scale_prob):
            scale = random.uniform(self.aug_scale_range[0],
                                   self.aug_scale_range[1])
            h, w = img_bgr.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            # 图像缩放
            img_bgr = cv2.resize(img_bgr, (new_w, new_h),
                                 interpolation=cv2.INTER_LINEAR)
            # flow field 同步缩放:
            #   cellprob (binary): INTER_NEAREST 保持二值
            #   dy/dx: Cellpose 的 flows 是归一化方向场 (|dP|≈1)，
            #     图像缩放不改变方向向量模长，仅需 INTER_LINEAR 空间插值
            cellprob = flow_full[0:1]  # (1, H, W)
            dy = flow_full[1:2]        # (1, H, W)
            dx = flow_full[2:3]        # (1, H, W)
            cp_scaled = cv2.resize(cellprob[0], (new_w, new_h),
                                   interpolation=cv2.INTER_NEAREST)
            dy_scaled = cv2.resize(dy[0], (new_w, new_h),
                                   interpolation=cv2.INTER_LINEAR)
            dx_scaled = cv2.resize(dx[0], (new_w, new_h),
                                   interpolation=cv2.INTER_LINEAR)
            flow_full = np.stack([cp_scaled, dy_scaled, dx_scaled], axis=0)

        return img_bgr, flow_full

    def _resize_pad(self, img_bgr, target_size):
        """Resize 保持宽高比, 用 pad_value 填充, 再 normalize"""
        h, w = img_bgr.shape[:2]
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = target_size - new_h
        pad_w = target_size - new_w
        pad_top = pad_h // 2
        pad_left = pad_w // 2

        img = cv2.copyMakeBorder(
            img, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
            cv2.BORDER_CONSTANT, value=(self.pad_value,) * 3)

        # BGR → RGB, float32, /255, normalize
        img = img[:, :, ::-1]
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        return img, (scale, pad_top, pad_left, new_h, new_w)

    def _resize_flow(self, flow_fullres, resize_info, target_size):
        """将全分辨率 flow field 变换为与训练目标对齐的 stride-4 版本。

        resize_info = (scale, pad_top, pad_left, new_h, new_w) — 来自 _resize_pad 的返回值，
        避免 _resize_pad 和 _resize_flow 各自计算 padding 产生 1px 偏差。

        cellprob: INTER_AREA 直接下采样到 stride-4
        dy/dx:   INTER_LINEAR 下采样到 stride-4（与 Cellpose resize target 语义对齐）
        """
        scale, pad_top, pad_left, new_h, new_w = resize_info
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        s4 = target_size // 4

        # Resize 到 input_size: cellprob 用 INTER_NEAREST 保持二值（与 Cellpose 一致），
        # dy/dx 用 INTER_LINEAR 平滑插值
        chs_resized = []
        # cellprob (channel 0): 二值掩码，用 INTER_NEAREST
        ch = cv2.resize(flow_fullres[0], (new_w, new_h),
                        interpolation=cv2.INTER_NEAREST)
        ch = cv2.copyMakeBorder(
            ch, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
            cv2.BORDER_CONSTANT, value=0)
        chs_resized.append(ch)
        # dy, dx (channel 1, 2): 连续 flow 向量，用 INTER_LINEAR
        for c in (1, 2):
            ch = cv2.resize(flow_fullres[c], (new_w, new_h),
                            interpolation=cv2.INTER_LINEAR)
            ch = cv2.copyMakeBorder(
                ch, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
                cv2.BORDER_CONSTANT, value=0)
            chs_resized.append(ch)
        flow_aligned = np.stack(chs_resized, axis=0)  # (3, target_size, target_size)

        # 下采样到 stride-4
        flow_s4 = np.zeros((3, s4, s4), dtype=np.float32)

        # cellprob: INTER_AREA 一步降到 stride 4
        flow_s4[0] = cv2.resize(flow_aligned[0], (s4, s4),
                                interpolation=cv2.INTER_AREA)

        # dy/dx: INTER_LINEAR 降到 stride 4（向量化，避免双层循环）
        for c in (1, 2):
            flow_s4[c] = cv2.resize(flow_aligned[c], (s4, s4),
                                    interpolation=cv2.INTER_LINEAR)

        return flow_s4

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fname = sample[0]
        npy_path = sample[1]
        label_path = sample[2] if len(sample) > 2 else ""
        img_path = os.path.join(self.img_dir, fname)

        # 加载图像
        img_bgr = imread_unicode(img_path)
        if img_bgr is None:
            raise RuntimeError(f"Cannot read image: {img_path}")

        # 加载 GT flow field (全分辨率)
        flow_full = np.load(long_path(npy_path))  # (3, H, W)
        img_h, img_w = img_bgr.shape[:2]
        if flow_full.ndim != 3 or flow_full.shape[0] != 3:
            raise RuntimeError(
                f"Flow shape must be (3,H,W), got {flow_full.shape}: {fname}")
        if flow_full.shape[1:] != (img_h, img_w):
            raise RuntimeError(
                f"Flow size {flow_full.shape[1:]} != image size {(img_h, img_w)}: {fname}")
        if flow_full.dtype != np.float32:
            raise RuntimeError(
                f"Flow dtype must be float32, got {flow_full.dtype}: {fname}")
        if not np.isfinite(flow_full).all():
            raise RuntimeError(f"Flow contains NaN/Inf: {fname}")

        # 加载 GT label（验证期使用，不做增强）
        label_tensor = None
        if label_path and not self.augment:
            from .utils import imread_unchanged
            label_data = imread_unchanged(label_path)
            if label_data is None:
                raise RuntimeError(f"Cannot read GT label: {label_path}")
            if label_data.ndim == 3 and label_data.shape[2] == 1:
                label_data = label_data[:, :, 0]
            if label_data.shape[:2] != (img_h, img_w):
                raise RuntimeError(
                    f"Label size {label_data.shape[:2]} != image size {(img_h, img_w)}: {fname}")
            label_tensor = torch.from_numpy(label_data.astype(np.int32))

        # 数据增强 (与 flow 同步)
        if self.augment:
            img_bgr, flow_full = self._apply_augmentation(img_bgr, flow_full)

        # 预处理图像
        img_tensor, resize_info = self._resize_pad(img_bgr, self.input_size)

        # 预处理 flow: 对齐 + 下采样到 stride 4（复用 _resize_pad 的尺寸信息）
        flow_s4 = self._resize_flow(flow_full, resize_info, self.input_size)

        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1)  # HWC → CHW
        flow_tensor = torch.from_numpy(flow_s4)  # (3, s4, s4)

        if label_tensor is not None:
            # resize_info 用于 evaluate_batch_v1_labels 将 GT label 对齐到模型输出尺寸
            return img_tensor, flow_tensor, label_tensor, resize_info
        return img_tensor, flow_tensor
