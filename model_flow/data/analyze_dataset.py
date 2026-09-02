"""数据集分析: 颗粒形态分布 + 密度统计。

用法:
    python -m model_flow.analyze_dataset \
        --ann data/particles/annotations/instances_train.json \
        --img-dir data/particles/train

    python -m model_flow.analyze_dataset \
        --ann data/particles/annotations/instances_val.json \
        --img-dir data/particles/val
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from ..utils import imread_unicode


def particle_metrics(ann, h, w):
    """计算单个颗粒的形态指标."""
    segm = ann.get('segmentation', [])
    bbox = ann['bbox']  # [x, y, w, h]

    area = ann.get('area', bbox[2] * bbox[3])
    bw, bh = bbox[2], bbox[3]

    # 长宽比
    aspect = max(bw, bh) / max(min(bw, bh), 1)

    # 如果有 polygon segmentation，可以计算更精确的指标
    convexity = -1.0
    solidity = -1.0
    perimeter = -1.0

    # RLE segmentation 不支持 polygon 分析，跳过
    if isinstance(segm, dict):
        pass
    elif isinstance(segm, list) and segm:
        for poly in segm:
            if isinstance(poly, list) and len(poly) >= 6:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                actual_area = cv2.contourArea(pts)
                if actual_area > 0:
                    hull = cv2.convexHull(pts)
                    hull_area = cv2.contourArea(hull)
                    peri = cv2.arcLength(pts, True)
                    convexity = hull_area / max(actual_area, 1)
                    solidity = actual_area / max(hull_area, 1)
                    perimeter = peri
                    area = actual_area
                break

    # 等效直径
    eq_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0

    return {
        'area': area,
        'width': bw,
        'height': bh,
        'aspect_ratio': aspect,
        'eq_diameter': eq_diameter,
        'convexity': convexity,
        'solidity': solidity,
        'perimeter': perimeter,
    }


def image_adhesion_score(anns, img_h, img_w):
    """计算图像的粘连程度: 统计有重叠 bbox 或近距离的颗粒对."""
    if len(anns) < 2:
        return 0, 0

    bboxes = []
    for ann in anns:
        x, y, w, h = ann['bbox']
        # 扩展 bbox 以检测近邻
        margin = max(3, min(w, h) * 0.1)
        bboxes.append([
            max(0, x - margin),
            max(0, y - margin),
            min(img_w, x + w + margin),
            min(img_h, y + h + margin),
        ])

    adhesion_pairs = 0
    total_pairs = 0
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            total_pairs += 1
            # IoU or overlap check
            xi1, yi1, xi2, yi2 = bboxes[i]
            xj1, yj1, xj2, yj2 = bboxes[j]
            inter_w = min(xi2, xj2) - max(xi1, xj1)
            inter_h = min(yi2, yj2) - max(yi1, yj1)
            if inter_w > 0 and inter_h > 0:
                adhesion_pairs += 1

    return adhesion_pairs, total_pairs


def analyze(ann_file, img_dir, limit=0):
    """主分析函数."""

    with open(ann_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    images = coco['images']
    if limit > 0:
        images = images[:limit]

    img_to_anns = defaultdict(list)
    for ann in coco['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # 收集指标
    all_metrics = []  # per-particle
    img_counts = []   # particles per image
    img_areas = []    # total particle area per image
    adhesion_counts = []

    for img_info in tqdm(images, desc='Analyzing'):
        img_id = img_info['id']
        anns = img_to_anns.get(img_id, [])
        if not anns:
            continue

        h, w = img_info['height'], img_info['width']
        img_area = h * w

        for ann in anns:
            m = particle_metrics(ann, h, w)
            m['image_id'] = img_id
            all_metrics.append(m)

        img_counts.append(len(anns))
        total_particle_area = sum(
            ann.get('area', ann['bbox'][2] * ann['bbox'][3]) for ann in anns)
        img_areas.append(total_particle_area / max(img_area, 1))

        n_adh, n_total = image_adhesion_score(anns, h, w)
        adhesion_counts.append((n_adh, n_total))

    # ════════════════════════════════════════════════════════
    # 统计输出
    # ════════════════════════════════════════════════════════

    areas = [m['area'] for m in all_metrics]
    aspects = [m['aspect_ratio'] for m in all_metrics if m['aspect_ratio'] > 0]
    diameters = [m['eq_diameter'] for m in all_metrics if m['eq_diameter'] > 0]
    convexities = [m['convexity'] for m in all_metrics if m['convexity'] > 0]
    solidities = [m['solidity'] for m in all_metrics if m['solidity'] > 0]

    adhesion_ratios = [n_adh / max(n_total, 1) for n_adh, n_total in adhesion_counts]

    print(f"\n{'='*60}")
    print(f"Dataset Analysis: {os.path.basename(ann_file)}")
    print(f"{'='*60}")
    print(f"  Images:          {len(images)}")
    print(f"  Annotated imgs:  {len(img_counts)}")
    print(f"  Total particles: {len(all_metrics)}")
    print(f"  Avg particles/img: {np.mean(img_counts):.1f}")
    print(f"  Particle coverage: {np.mean(img_areas)*100:.1f}%")

    print(f"\n── Particle Morphology ──")
    print(f"  {'Metric':<20} {'Min':>10} {'Median':>10} {'Max':>10}")
    print(f"  {'-'*50}")
    if areas:
        a = sorted(areas)
        print(f"  {'Area (px)':<20} {a[0]:>10.1f} {a[len(a)//2]:>10.1f} {a[-1]:>10.1f}")
    if diameters:
        d = sorted(diameters)
        print(f"  {'Eq Diameter (px)':<20} {d[0]:>10.1f} {d[len(d)//2]:>10.1f} {d[-1]:>10.1f}")
    if aspects:
        asp = sorted(aspects)
        print(f"  {'Aspect Ratio':<20} {asp[0]:>10.2f} {asp[len(asp)//2]:>10.2f} {asp[-1]:>10.2f}")
    if convexities:
        c = sorted(convexities)
        print(f"  {'Convexity':<20} {c[0]:>10.3f} {c[len(c)//2]:>10.3f} {c[-1]:>10.3f}")
    if solidities:
        s = sorted(solidities)
        print(f"  {'Solidity':<20} {s[0]:>10.3f} {s[len(s)//2]:>10.3f} {s[-1]:>10.3f}")

    print(f"\n── Density / Adhesion ──")
    print(f"  Particles per image: {np.min(img_counts)} ~ {np.median(img_counts):.0f} ~ {np.max(img_counts)}")
    print(f"  Adhesion ratio:      {np.mean(adhesion_ratios)*100:.1f}% (higher = more crowded)")
    print(f"  Pct images with adhesion > 10%: "
          f"{100 * sum(1 for r in adhesion_ratios if r > 0.1) / max(len(adhesion_ratios), 1):.1f}%")

    # Flow field 适用性评估
    print(f"\n── Flow Field Suitability ──")
    if aspects:
        high_aspect = sum(1 for v in aspects if v > 3.0)
        print(f"  Particles with aspect > 3.0: {high_aspect} "
              f"({100 * high_aspect / max(len(aspects), 1):.1f}%)")
        if high_aspect / max(len(aspects), 1) > 0.3:
            print(f"  ⚠ WARNING: High proportion of elongated particles.")
            print(f"    Flow field may struggle — consider data filtering.")
        else:
            print(f"  OK: Most particles are compact, suitable for flow field.")

    if convexities:
        low_conv = sum(1 for v in convexities if 0 < v < 1.1)
        print(f"  Near-convex particles (convexity < 1.1): {low_conv} "
              f"({100 * low_conv / max(len(convexities), 1):.1f}%)")

    print(f"\n{'='*60}")
    print(f"Analysis complete.")
    print(f"{'='*60}")

    return {
        'n_images': len(images),
        'n_particles': len(all_metrics),
        'avg_per_image': np.mean(img_counts),
        'area_min_median_max': (np.min(areas), np.median(areas), np.max(areas)) if areas else (0, 0, 0),
        'diameter_median': np.median(diameters) if len(diameters) > 0 else 0,
        'aspect_median': np.median(aspects) if len(aspects) > 0 else 0,
        'adhesion_ratio_mean': np.mean(adhesion_ratios) if adhesion_ratios else 0,
        'all_metrics': all_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze particle dataset')
    parser.add_argument('--ann', required=True, help='COCO annotation JSON')
    parser.add_argument('--img-dir', default='', help='Image directory')
    parser.add_argument('--limit', type=int, default=0, help='Max images (0=all)')
    parser.add_argument('--out', '-o', default='', help='Save metrics to JSON')
    args = parser.parse_args()

    results = analyze(args.ann, args.img_dir, args.limit)

    if args.out:
        # 保存关键指标 (不含 all_metrics 避免过大)
        save_results = {k: v for k, v in results.items() if k != 'all_metrics'}
        # 转换 numpy 类型
        for k, v in save_results.items():
            if isinstance(v, (np.floating, np.integer)):
                save_results[k] = float(v)
            elif isinstance(v, tuple):
                save_results[k] = [float(x) for x in v]
        with open(args.out, 'w') as f:
            json.dump(save_results, f, indent=2, default=float)
        print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
