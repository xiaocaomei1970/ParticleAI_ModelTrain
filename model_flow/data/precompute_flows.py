"""从 COCO 标注生成 GT flow field (.npy)

用法:
    python -m model_flow.precompute_flows \
        --ann data/particles/annotations/instances_train.json \
        --img-dir data/particles/train \
        --out-dir data/particles/flows_train

    python -m model_flow.precompute_flows \
        --ann data/particles/annotations/instances_val.json \
        --img-dir data/particles/val \
        --out-dir data/particles/flows_val
"""
# V1 legacy / COCO 辅助 — V1 正式训练使用 convert_labels_to_flows.py，不使用此脚本。
import os
import sys
import json
import argparse
import time

import numpy as np
import cv2
from tqdm import tqdm

from .gt_flows import generate_flow_from_coco
from ..utils import long_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', required=True, help='COCO annotation JSON')
    parser.add_argument('--img-dir', required=True, help='Image directory')
    parser.add_argument('--out-dir', required=True, help='Output .npy directory')
    parser.add_argument('--limit', type=int, default=0, help='Max images (0=all)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.ann, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # img_id → annotations
    img_to_anns = {}
    for ann in coco['annotations']:
        img_to_anns.setdefault(ann['image_id'], []).append(ann)

    images = coco['images']
    if args.limit > 0:
        images = images[:args.limit]

    print(f'Processing {len(images)} images...')
    t0 = time.time()

    missing_imgs = 0
    for img_info in tqdm(images):
        img_id = img_info['id']
        anns = img_to_anns.get(img_id, [])
        if not anns:
            continue

        # 验证图片文件存在
        img_path = os.path.join(args.img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            missing_imgs += 1
            continue

        h, w = img_info['height'], img_info['width']
        flow = generate_flow_from_coco(anns, h, w)

        fname = os.path.splitext(img_info['file_name'])[0]
        out_path = long_path(os.path.join(args.out_dir, fname + '.npy'))
        np.save(out_path, flow)

    elapsed = time.time() - t0
    count = len(os.listdir(args.out_dir))
    if missing_imgs > 0:
        print(f'Warning: {missing_imgs} images referenced in COCO JSON but missing on disk, skipped.')
    print(f'Done! Generated {count} flow fields in {elapsed:.1f}s '
          f'({elapsed / max(count, 1):.1f}s per image)')
    print(f'Output: {args.out_dir}/')


if __name__ == '__main__':
    main()
