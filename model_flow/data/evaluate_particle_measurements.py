"""Evaluate predicted instance contours against GT labels.

This module intentionally reports only segmentation/contour quality metrics.
Particle size, area, and physical-unit analysis belong to downstream analysis
software and are not acceptance criteria for the training solution.

Usage:
    python -m model_flow.data.evaluate_particle_measurements ^
        --pred-label-dir temp/pred_labels ^
        --gt-label-dir data/particles/val ^
        --out temp/contour_report.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..utils import imread_unchanged
from ..eval_masks import evaluate_masks_v1, _match_instances


def compute_metrics_for_image(pred_path: str, gt_path: str) -> dict:
    """Compute contour-focused metrics for one predicted label image."""
    pred = imread_unchanged(pred_path)
    gt = imread_unchanged(gt_path)
    if pred is None or gt is None:
        return {}

    metrics = evaluate_masks_v1(pred, gt)
    matches, _, _, _ = _match_instances(pred, gt)
    n_matched = len(matches)
    false_positive_count = max(0, metrics['n_pred'] - n_matched)
    false_negative_count = max(0, metrics['n_gt'] - n_matched)
    review_required = (
        metrics['instance_f1'] < 0.90 or
        metrics['mask_iou_mean'] < 0.85 or
        metrics['boundary_iou_mean'] < 0.85 or
        metrics['recall'] < 0.90 or
        metrics['precision'] < 0.90
    )

    return {
        "n_gt": metrics['n_gt'],
        "n_pred": metrics['n_pred'],
        "n_matched": n_matched,
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "instance_f1": metrics['instance_f1'],
        "mask_iou_mean": metrics['mask_iou_mean'],
        "boundary_iou_mean": metrics['boundary_iou_mean'],
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "over_split_proxy_count": metrics['over_split_proxy_count'],
        "review_required": review_required,
    }


def main():
    parser = argparse.ArgumentParser(description="实例轮廓与 GT 重合度验收")
    parser.add_argument("--pred-label-dir", required=True, help="预测标签目录")
    parser.add_argument("--gt-label-dir", required=True, help="GT 标签目录")
    parser.add_argument("--out", default="temp/contour_report.json",
                        help="输出 JSON 报告路径")
    args = parser.parse_args()

    pred_dir = Path(args.pred_label_dir)
    gt_dir = Path(args.gt_label_dir)

    pred_files = sorted(pred_dir.glob("*_labels.png"))
    if not pred_files:
        print("No *_labels.png files found in", args.pred_label_dir)
        return

    per_image = []
    for pred_file in tqdm(pred_files):
        gt_file = gt_dir / pred_file.name
        if not gt_file.exists():
            continue
        metrics = compute_metrics_for_image(str(pred_file), str(gt_file))
        if metrics:
            metrics["file"] = pred_file.stem.replace("_labels", "")
            per_image.append(metrics)

    report = {
        "n_images": len(per_image),
        "summary": {},
        "images": per_image,
    }
    if per_image:
        report["summary"] = {
            "total_gt": sum(row["n_gt"] for row in per_image),
            "total_pred": sum(row["n_pred"] for row in per_image),
            "total_matched": sum(row["n_matched"] for row in per_image),
            "precision_mean": float(np.mean([row["precision"] for row in per_image])),
            "recall_mean": float(np.mean([row["recall"] for row in per_image])),
            "instance_f1_mean": float(np.mean([row["instance_f1"] for row in per_image])),
            "mask_iou_mean": float(np.mean([row["mask_iou_mean"] for row in per_image])),
            "boundary_iou_mean": float(np.mean([row["boundary_iou_mean"] for row in per_image])),
            "false_positive_count": sum(row["false_positive_count"] for row in per_image),
            "false_negative_count": sum(row["false_negative_count"] for row in per_image),
            "over_split_proxy_count": sum(row["over_split_proxy_count"] for row in per_image),
            "review_required_count": sum(
                1 for row in per_image if row.get("review_required", False)),
            "review_required_ratio": float(
                sum(1 for row in per_image if row.get("review_required", False))
                / len(per_image)),
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=float)

    summary = report["summary"]
    print(f"Contour evaluation: {report['n_images']} images")
    print(f"  GT instances: {summary.get('total_gt', 0)}")
    print(f"  Pred instances: {summary.get('total_pred', 0)}")
    print(f"  Matched (IoU>0.5): {summary.get('total_matched', 0)}")
    print(f"  Instance F1: {summary.get('instance_f1_mean', 0):.3f}")
    print(f"  Mask IoU: {summary.get('mask_iou_mean', 0):.3f}")
    print(f"  Boundary IoU: {summary.get('boundary_iou_mean', 0):.3f}")
    print(f"  False positives: {summary.get('false_positive_count', 0)}")
    print(f"  False negatives: {summary.get('false_negative_count', 0)}")
    print(f"  Over-split proxy: {summary.get('over_split_proxy_count', 0)}")
    print(f"  Review required: {summary.get('review_required_count', 0)}/{report['n_images']} "
          f"({summary.get('review_required_ratio', 0):.1%})")
    print(f"  Report: {args.out}")


if __name__ == "__main__":
    main()
