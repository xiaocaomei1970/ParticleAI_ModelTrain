"""Analyze reviewed uint16 instance label maps without relying on COCO JSON.

This is intended for newly reviewed pre-label data before it is merged into the
full training set. It reads *_labels.png files directly, checks optional image
basename pairing, and reports particle size/shape statistics that are useful for
deciding whether the new complete dataset covers the required particle scenes.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..utils import imread_unchanged


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def collect_images(img_dir: str) -> dict[str, Path]:
    if not img_dir:
        return {}
    root = Path(img_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    return {
        path.stem: path
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def collect_labels(label_dir: str) -> dict[str, Path]:
    root = Path(label_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    labels = {
        path.name[:-len('_labels.png')]: path
        for path in sorted(root.iterdir())
        if path.is_file() and path.name.endswith('_labels.png')
    }
    if not labels:
        raise RuntimeError(f"No *_labels.png files found in {label_dir}")
    return labels


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def analyze_one(label_path: Path) -> dict:
    labels = imread_unchanged(str(label_path))
    if labels is None:
        raise RuntimeError(f"Cannot read label map: {label_path}")
    if labels.ndim == 3:
        labels = cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY)
    if not np.issubdtype(labels.dtype, np.integer):
        raise RuntimeError(f"Label map must be integer type: {label_path}")

    labels = labels.astype(np.int32, copy=False)
    height, width = labels.shape
    image_area = max(height * width, 1)
    particle_ids = [int(v) for v in np.unique(labels) if v > 0]

    particle_metrics = []
    edge_touching = 0
    for particle_id in particle_ids:
        mask = (labels == particle_id).astype(np.uint8)
        area = int(mask.sum())
        if area <= 0:
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        perimeter = float(cv2.arcLength(contour, True))
        contour_area = float(cv2.contourArea(contour))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        equivalent_diameter = float(np.sqrt(4.0 * area / np.pi))
        aspect_ratio = float(max(w, h) / max(min(w, h), 1))
        circularity = float(4.0 * np.pi * area / max(perimeter * perimeter, 1e-8))
        solidity = float(contour_area / max(hull_area, 1e-8))

        touches_edge = x <= 0 or y <= 0 or (x + w) >= width or (y + h) >= height
        if touches_edge:
            edge_touching += 1

        particle_metrics.append({
            'id': particle_id,
            'area': area,
            'equivalent_diameter': equivalent_diameter,
            'bbox_width': int(w),
            'bbox_height': int(h),
            'aspect_ratio': aspect_ratio,
            'perimeter': perimeter,
            'circularity': circularity,
            'solidity': solidity,
            'edge_touching': touches_edge,
        })

    areas = [m['area'] for m in particle_metrics]
    diameters = [m['equivalent_diameter'] for m in particle_metrics]
    aspects = [m['aspect_ratio'] for m in particle_metrics]
    circularities = [m['circularity'] for m in particle_metrics]
    solidities = [m['solidity'] for m in particle_metrics]

    return {
        'file': label_path.name,
        'height': height,
        'width': width,
        'particle_count': len(particle_metrics),
        'foreground_fraction': float((labels > 0).sum() / image_area),
        'edge_touching_count': edge_touching,
        'area_min': percentile(areas, 0),
        'area_p10': percentile(areas, 10),
        'area_median': percentile(areas, 50),
        'area_p90': percentile(areas, 90),
        'area_max': percentile(areas, 100),
        'diameter_min': percentile(diameters, 0),
        'diameter_p10': percentile(diameters, 10),
        'diameter_median': percentile(diameters, 50),
        'diameter_p90': percentile(diameters, 90),
        'diameter_max': percentile(diameters, 100),
        'aspect_median': percentile(aspects, 50),
        'aspect_p90': percentile(aspects, 90),
        'circularity_median': percentile(circularities, 50),
        'solidity_median': percentile(solidities, 50),
        '_areas': areas,
        '_diameters': diameters,
    }


def summarize(per_image: list[dict]) -> dict:
    counts = [row['particle_count'] for row in per_image]
    fg = [row['foreground_fraction'] for row in per_image]
    diameters = []
    areas = []
    for row in per_image:
        diameters.extend(row.get('_diameters', []))
        areas.extend(row.get('_areas', []))

    total_particles = int(sum(counts))
    diameter_lt4_count = sum(1 for value in diameters if value < 4.0)
    diameter_lt8_count = sum(1 for value in diameters if value < 8.0)

    return {
        'n_images': len(per_image),
        'total_particles': total_particles,
        'particles_per_image_min': percentile(counts, 0),
        'particles_per_image_median': percentile(counts, 50),
        'particles_per_image_max': percentile(counts, 100),
        'foreground_fraction_median': percentile(fg, 50),
        'diameter_global_p10': percentile(diameters, 10),
        'diameter_global_p50': percentile(diameters, 50),
        'diameter_global_p90': percentile(diameters, 90),
        'diameter_sample_median': percentile(diameters, 50),
        'area_global_p10': percentile(areas, 10),
        'area_global_p50': percentile(areas, 50),
        'area_global_p90': percentile(areas, 90),
        'area_sample_median': percentile(areas, 50),
        'diameter_lt4_count': int(diameter_lt4_count),
        'diameter_lt4_ratio': float(diameter_lt4_count / total_particles) if total_particles else 0.0,
        'diameter_lt8_count': int(diameter_lt8_count),
        'diameter_lt8_ratio': float(diameter_lt8_count / total_particles) if total_particles else 0.0,
        'edge_touching_total': int(sum(row['edge_touching_count'] for row in per_image)),
    }


def public_image_summary(row: dict) -> dict:
    return {
        key: value for key, value in row.items()
        if not key.startswith('_')
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze reviewed *_labels.png maps without COCO JSON.')
    parser.add_argument('--label-dir', required=True,
                        help='Directory containing <basename>_labels.png files.')
    parser.add_argument('--img-dir', default='',
                        help='Optional original image directory; enables strict basename pairing checks.')
    parser.add_argument('--out', default='temp/label_dataset_analysis.json',
                        help='Output JSON report path.')
    args = parser.parse_args()

    try:
        labels = collect_labels(args.label_dir)
        images = collect_images(args.img_dir)
        if images:
            missing_labels = sorted(set(images) - set(labels))
            labels_without_images = sorted(set(labels) - set(images))
            if missing_labels or labels_without_images:
                print("Label/image basename mismatch:")
                print(f"  images without labels: {len(missing_labels)}")
                for name in missing_labels[:20]:
                    print(f"    {name}")
                print(f"  labels without images: {len(labels_without_images)}")
                for name in labels_without_images[:20]:
                    print(f"    {name}_labels.png")
                raise RuntimeError("Fix label/image basenames before analysis.")

        per_image = []
        for _, label_path in tqdm(labels.items(), desc='Analyzing labels'):
            per_image.append(analyze_one(label_path))

        report = {
            'label_dir': args.label_dir,
            'img_dir': args.img_dir,
            'summary': summarize(per_image),
            'images': [public_image_summary(row) for row in per_image],
        }

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

        summary = report['summary']
        print("\nLabel dataset analysis OK.")
        print(f"  images: {summary['n_images']}")
        print(f"  total particles: {summary['total_particles']}")
        print("  particles/image: "
              f"{summary['particles_per_image_min']:.0f} / "
              f"{summary['particles_per_image_median']:.0f} / "
              f"{summary['particles_per_image_max']:.0f}")
        print(f"  diameter <4px: {summary['diameter_lt4_count']} "
              f"({summary['diameter_lt4_ratio']:.1%})")
        print(f"  diameter <8px: {summary['diameter_lt8_count']} "
              f"({summary['diameter_lt8_ratio']:.1%})")
        print(f"  report: {out_path}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
