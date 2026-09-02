"""大图切 tile 训练数据生成。

仅对长边 >1536 的图片启用。按 V1 固定规则切 tile：
  tile_size=1024, overlap=256, core_margin=128

输出 tile 命名规则：{原图basename}_tile_{row}_{col}.{ext}
对应的 label 命名：{原图basename}_tile_{row}_{col}_labels.png

用法:
    python -m model_flow.data.create_tiled_training_data ^
        --img-dir temp/staging/images ^
        --label-dir temp/staging/labels ^
        --out-dir temp/tiled_data
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..utils import imread_unicode, imread_unchanged, imwrite_unicode, long_path

TILE_SIZE = 1024
OVERLAP = 256
LONG_SIDE_LIMIT = 1536


def should_tile(img_path: str) -> bool:
    """返回 True 表示需要切图。"""
    img = imread_unicode(img_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    return max(h, w) > LONG_SIDE_LIMIT


def tile_positions(h: int, w: int):
    """生成 (y, x, tile_h, tile_w) 的 tile 位置列表。

    滑动步长 1024 - 256 = 768，保证 overlap=256。
    最后一块对齐右下角。
    """
    step = TILE_SIZE - OVERLAP

    def axis_starts(length: int) -> list[int]:
        if length <= TILE_SIZE:
            return [0]
        max_start = length - TILE_SIZE
        starts = list(range(0, max_start + 1, step))
        if starts[-1] != max_start:
            starts.append(max_start)
        return starts

    positions = []
    for y_start in axis_starts(h):
        for x_start in axis_starts(w):
            y_end = min(y_start + TILE_SIZE, h)
            x_end = min(x_start + TILE_SIZE, w)
            positions.append((y_start, x_start, y_end - y_start, x_end - x_start))
    return positions


def process_one(img_path: str, label_path: str, out_dir: str) -> int:
    """处理一张大图，返回生成的 tile 数量。"""
    img = imread_unicode(img_path)
    if img is None:
        return 0

    labels = imread_unchanged(label_path)
    if labels is None:
        labels = np.zeros(img.shape[:2], dtype=np.uint16)

    h, w = img.shape[:2]
    base = Path(img_path).stem
    ext = Path(img_path).suffix
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    count = 0
    for row_idx, col_idx, th, tw in tile_positions(h, w):
        # 只在 tile 有前景内容时输出
        tile_labels = labels[row_idx:row_idx + th, col_idx:col_idx + tw]
        if tile_labels.max() == 0:
            continue

        tile_img = img[row_idx:row_idx + th, col_idx:col_idx + tw]

        # resize/pad 到 TILE_SIZE
        need_pad_h = TILE_SIZE - th
        need_pad_w = TILE_SIZE - tw
        if need_pad_h > 0 or need_pad_w > 0:
            # Match C++ tiled inference: preserve tile origin and pad only
            # bottom/right for edge tiles.
            pad_top = 0
            pad_bottom = need_pad_h
            pad_left = 0
            pad_right = need_pad_w
            tile_img = cv2.copyMakeBorder(tile_img, pad_top, pad_bottom,
                                           pad_left, pad_right,
                                           cv2.BORDER_CONSTANT, value=114)
            tile_labels = cv2.copyMakeBorder(tile_labels.astype(np.uint16),
                                              pad_top, pad_bottom,
                                              pad_left, pad_right,
                                              cv2.BORDER_CONSTANT, value=0)

        tile_name = f"{base}_tile_{row_idx}_{col_idx}"
        tile_img_path = out / f"{tile_name}{ext}"
        tile_label_path = out / f"{tile_name}_labels.png"

        imwrite_unicode(long_path(str(tile_img_path)), tile_img)
        imwrite_unicode(long_path(str(tile_label_path)),
                        tile_labels.astype(np.uint16),
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="大图切 tile 训练数据生成")
    parser.add_argument("--img-dir", required=True, help="原图目录")
    parser.add_argument("--label-dir", required=True, help="标签目录（*_labels.png）")
    parser.add_argument("--out-dir", required=True, help="输出 tile 目录")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)

    images = sorted([f for f in img_dir.iterdir()
                     if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')])

    total_tiles = 0
    tiled_images = 0
    for img_path in tqdm(images):
        if not should_tile(str(img_path)):
            continue

        stem = img_path.stem
        label_path = label_dir / f"{stem}_labels.png"
        if not label_path.exists():
            print(f"  SKIP (no label): {img_path.name}")
            continue

        n = process_one(str(img_path), str(label_path), str(args.out_dir))
        if n > 0:
            total_tiles += n
            tiled_images += 1

    print(f"\nDone: {total_tiles} tiles from {tiled_images} large images -> {args.out_dir}")


if __name__ == "__main__":
    main()
