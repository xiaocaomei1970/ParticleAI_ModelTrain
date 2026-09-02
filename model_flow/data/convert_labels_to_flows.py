"""批量转换: PNG uint16 标签图 → .npy flow field

用法:
    python -m model_flow.data.convert_labels_to_flows \
        --label-dir ./reviewed_labels/ \
        --out-dir data/particles/flows_train/
"""
import csv
import os
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from cellpose.dynamics import labels_to_flows
from tqdm import tqdm

from ..utils import imread_unchanged, long_path


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def collect_image_bases(img_dir: str) -> set[str]:
    if not img_dir:
        return set()
    return {
        os.path.splitext(name)[0]
        for name in os.listdir(img_dir)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS
        and not os.path.splitext(name)[0].endswith('_labels')
    }


def validate_label_files(label_dir: str, img_dir: str = '') -> list[str]:
    label_files = sorted([
        f for f in os.listdir(label_dir)
        if f.endswith('_labels.png')
    ])
    if not label_files:
        raise RuntimeError(
            f"No *_labels.png files found in {label_dir}. "
            "Reviewed label files must be named <image_basename>_labels.png."
        )

    label_bases = {f[:-len('_labels.png')] for f in label_files}
    if len(label_bases) != len(label_files):
        raise RuntimeError("Duplicate label basenames found.")

    if img_dir:
        image_bases = collect_image_bases(img_dir)
        labels_without_images = sorted(label_bases - image_bases)
        images_without_labels = sorted(image_bases - label_bases)
        if labels_without_images or images_without_labels:
            print("\nLabel/image basename mismatch:")
            print(f"  labels without images: {len(labels_without_images)}")
            for name in labels_without_images[:20]:
                print(f"    {name}_labels.png")
            print(f"  images without labels: {len(images_without_labels)}")
            for name in images_without_labels[:20]:
                print(f"    {name}")
            raise RuntimeError(
                "Label files must match image basenames before converting to flow."
            )

    return label_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-dir', required=True,
                        help='PNG 标签图目录 (审核修订后的)')
    parser.add_argument('--out-dir', required=True,
                        help='输出 .npy flow field 目录')
    parser.add_argument('--img-dir', default='',
                        help='可选：原图目录。提供后会严格检查 <basename>_labels.png 与原图 basename 一一对应。')
    parser.add_argument('--tile-manifest', default='',
                        help='可选：tile staging manifest。提供后，tile_role=background_negative '
                             '的全 0 tile 会被允许并生成全 0 flow。')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 读取 background_negative tile 白名单
    bg_tile_stems: set[str] = set()
    if args.tile_manifest:
        if not os.path.exists(args.tile_manifest):
            raise SystemExit(f"ERROR: --tile-manifest file not found: {args.tile_manifest}")
        with open(args.tile_manifest, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if row.get('tile_role') == 'background_negative':
                    stem = Path(row.get('file_name', '')).stem
                    if stem:
                        bg_tile_stems.add(stem)

    try:
        label_files = validate_label_files(args.label_dir, args.img_dir)
    except RuntimeError as exc:
        print(f'ERROR: {exc}')
        raise SystemExit(1)
    print(f'Found {len(label_files)} label maps')
    if bg_tile_stems:
        print(f'  (background_negative tiles whitelisted: {len(bg_tile_stems)})')

    t0 = time.time()
    generated = 0
    validation_failures = 0
    for fname in tqdm(label_files):
        # 读取标签图并校验
        label_path = os.path.join(args.label_dir, fname)
        labels = imread_unchanged(label_path)
        if labels is None:
            print(f'  ERROR: cannot read {fname}, skipping')
            validation_failures += 1
            continue

        label_stem = fname[:-len('_labels.png')]
        allow_empty = label_stem in bg_tile_stems

        from ..utils import validate_instance_label
        label_errors = validate_instance_label(labels, max_area_fraction=0.8,
                                               allow_empty=allow_empty)
        if label_errors:
            validation_failures += 1
            for err in label_errors:
                print(f'  ERROR: {fname}: {err}')
            continue

        # 转为 int32
        if labels.ndim == 3 and labels.shape[2] == 1:
            labels = labels[:, :, 0]
        if labels.dtype != np.int32:
            labels = labels.astype(np.int32)

        if labels.max() == 0:
            # 无颗粒，生成空 flow
            h, w = labels.shape
            flow = np.zeros((3, h, w), dtype=np.float32)
        else:
            # labels_to_flows: (4, H, W) → [labels, cell_probability, dy, dx]
            # cell_probability 由距离变换经非线性归一化得到，非原始距离值
            flows = labels_to_flows([labels], device=torch.device('cpu'))
            f = flows[0]
            # cell_probability 二值化: 与 Cellpose 官方训练对齐 (threshold=0.5)
            cellprob = (f[1] > 0.5).astype(np.float32)
            dy = f[2].astype(np.float32)
            dx = f[3].astype(np.float32)
            flow = np.stack([cellprob, dy, dx], axis=0)

        # 保存 .npy
        base = fname[:-len('_labels.png')]
        out_path = long_path(os.path.join(args.out_dir, base + '.npy'))
        np.save(out_path, flow)
        generated += 1

    elapsed = time.time() - t0
    print(f'Done! {generated} flow fields in {elapsed:.0f}s '
          f'({elapsed/max(generated,1):.1f}s per image)')
    print(f'Output: {args.out_dir}/')
    if validation_failures > 0:
        raise SystemExit(
            f'ERROR: {validation_failures} label(s) failed validation. '
            'Fix the labels above before proceeding.')
    print('')
    print('Next for V1: verify image/flow pairs and dataset readiness:')
    print('  python scripts/verify_training_pairs.py')
    print('  python -m model_flow.manifest.check_dataset_manifest \\')
    print('      --manifest data/particles/dataset_manifest.csv \\')
    print('      --base-dir . --require-flow-for-splits train,val')
    print('')
    print('For new reviewed data, use the V1 staging -> split -> flow pipeline;')
    print('do not bypass manifest review with the legacy merge helper.')


if __name__ == '__main__':
    main()
