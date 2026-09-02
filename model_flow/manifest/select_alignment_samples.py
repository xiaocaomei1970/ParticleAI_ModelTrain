"""Legacy: 筛选对齐验证用的典型图片样本（基于 COCO JSON）。
Superseded by select_alignment_samples_by_manifest.py for V1.
Use select_alignment_samples_by_manifest.py with dataset_manifest.csv instead.

基于数据集标注统计，按 6 个维度打分排序，输出 10-20 张候选图片。

用法:
    python -m model_flow.select_alignment_samples \
        --ann data/particles/annotations/instances_train.json \
        --img-dir data/particles/train \
        --out-samples alignment_samples.txt \
        --top 20
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import cv2
import numpy as np

from ..utils import imread_unicode


def image_metrics(anns, h, w):
    """计算单张图片的多样性指标"""
    n = len(anns)
    if n == 0:
        return None

    areas = []
    aspects = []
    diameters = []
    # 检查边界颗粒
    edge_touching = 0

    for ann in anns:
        bbox = ann.get('bbox', [0, 0, 0, 0])
        x, y, bw, bh = bbox
        area = ann.get('area', bw * bh)
        areas.append(area)

        eq_diam = np.sqrt(4 * area / np.pi) if area > 0 else 0
        diameters.append(eq_diam)

        aspect = max(bw, bh) / max(min(bw, bh), 1) if bw > 0 and bh > 0 else 1
        aspects.append(aspect)

        # 边界接触检测
        if (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1):
            edge_touching += 1

    # 粘连程度: 扩展 bbox 后计算重叠
    adhesion_pairs = 0
    if n > 1:
        expanded = []
        for ann in anns:
            x, y, bw, bh = ann.get('bbox', [0, 0, 0, 0])
            margin = max(3, min(bw, bh) * 0.1)
            expanded.append([
                max(0, x - margin), max(0, y - margin),
                min(w, x + bw + margin), min(h, y + bh + margin)
            ])
        for i in range(len(expanded)):
            for j in range(i + 1, len(expanded)):
                xi1, yi1, xi2, yi2 = expanded[i]
                xj1, yj1, xj2, yj2 = expanded[j]
                if min(xi2, xj2) > max(xi1, xj1) and min(yi2, yj2) > max(yi1, yj1):
                    adhesion_pairs += 1

    adhesion_ratio = adhesion_pairs / max(n * (n - 1) / 2, 1)

    return {
        'n_particles': n,
        'area_median': float(np.median(areas)),
        'dia_median': float(np.median(diameters)),
        'aspect_max': float(np.max(aspects)),
        'adhesion_ratio': float(adhesion_ratio),
        'edge_touching': edge_touching,
    }


from .dataset_manifest_utils import infer_microscope_type as detect_microscope_type


def score_image(metrics):
    """为图片的多样性打分 (越高越好)。"""
    score = 0.0
    n = metrics['n_particles']

    # 密度梯度分: 偏离中位数的程度
    if n < 20:
        score += 1.0  # 低密度
    elif n < 80:
        score += 0.5  # 中密度
    else:
        score += 1.0  # 高密度 (罕见有价值)

    # 粘连程度
    if metrics['adhesion_ratio'] < 0.01:
        score += 0.3
    elif metrics['adhesion_ratio'] > 0.3:
        score += 1.0  # 高度粘连更有代表性

    # 粒径极端值
    if metrics['dia_median'] < 20:
        score += 0.5
    elif metrics['dia_median'] > 80:
        score += 0.5

    # 长宽比极端
    if metrics['aspect_max'] > 2.5:
        score += 1.0

    # 边界接触
    if metrics['edge_touching'] > 0:
        score += 1.0

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', required=True, help='COCO annotation JSON')
    parser.add_argument('--img-dir', default='', help='Image directory')
    parser.add_argument('--out-samples', default='alignment_samples.txt',
                        help='Output candidate list file')
    parser.add_argument('--top', type=int, default=20, help='Max candidates')
    args = parser.parse_args()

    with open(args.ann, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    img_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # 收集所有图片的指标
    all_metrics = []
    for img_info in coco['images']:
        img_id = img_info['id']
        anns = img_to_anns.get(img_id, [])
        if not anns:
            continue

        h, w = img_info['height'], img_info['width']
        m = image_metrics(anns, h, w)
        if m is None:
            continue

        m['img_id'] = img_id
        m['file_name'] = img_info['file_name']
        m['microscope'] = detect_microscope_type(img_info['file_name'])
        m['h'] = h
        m['w'] = w
        all_metrics.append(m)

    # 计算全局参考值
    all_dia = [m['dia_median'] for m in all_metrics if m['dia_median'] > 0]

    # 打分
    for m in all_metrics:
        m['score'] = score_image(m)

    # ── 策略：按显微镜类型分组 → 每组选前 N → 剩余名额按分数补齐 ──
    # 按类型分组
    by_micro = defaultdict(list)
    for m in all_metrics:
        by_micro[m['microscope']].append(m)

    # 每组内按分数排序
    for typ in by_micro:
        by_micro[typ].sort(key=lambda x: x['score'], reverse=True)

    types_sorted = sorted(by_micro.keys())
    per_type_target = max(1, args.top // len(types_sorted))  # 每种类型至少这么多

    selected = []
    selected_files = set()

    # 第一轮: 每种类型选 per_type_target 张
    for typ in types_sorted:
        if len(selected) >= args.top:
            break
        for m in by_micro[typ]:
            if len(selected) >= args.top:
                break
            if m['file_name'] in selected_files:
                continue
            if len([s for s in selected if s['microscope'] == typ]) >= per_type_target + 1:
                break
            selected.append(m)
            selected_files.add(m['file_name'])

    # 第二轮: 剩余名额按分数填充（排除已选的）
    remaining = [m for m in all_metrics if m['file_name'] not in selected_files]
    remaining.sort(key=lambda x: x['score'], reverse=True)
    while len(selected) < args.top and remaining:
        selected.append(remaining.pop(0))
        selected_files.add(selected[-1]['file_name'])

    # 按显微镜类型排序输出
    selected.sort(key=lambda x: (x['microscope'], -x['score']))

    # 输出
    print(f"Dataset: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"\nSelected {len(selected)} candidates for alignment testing:\n")
    print(f"{'#':<4} {'Score':<8} {'Particles':<10} {'Diam':<10} {'Adhesion':<10} {'Microscope':<14} {'File'}")
    print("-" * 100)

    with open(args.out_samples, 'w', encoding='utf-8') as out:
        for i, m in enumerate(selected):
            line = (f"{i+1:<4} {m['score']:<8.1f} {m['n_particles']:<10} "
                    f"{m['dia_median']:<10.1f} {m['adhesion_ratio']:<10.2f} "
                    f"{m['microscope']:<14} {m['file_name']}")
            print(line)
            out.write(m['file_name'] + '\n')

    print(f"\nCandidate list saved to: {args.out_samples}")
    print("Review and confirm these files, then run alignment test.")


if __name__ == '__main__':
    main()
