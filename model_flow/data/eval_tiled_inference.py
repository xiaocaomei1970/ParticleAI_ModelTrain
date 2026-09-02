"""大图 holdout 轮廓验证：对 holdout 大图执行 tile 推理，与 GT 对比输出指标。

仅用于模型与 C++ 后处理验收，不进入下游分析软件。
依赖 C++ flow_inference.exe 的 --tile 模式。

用法:
    python -m model_flow.data.eval_tiled_inference ^
        --img-dir temp/holdout_images ^
        --gt-label-dir temp/holdout_labels ^
        --onnx-dir experiments/2026-05-v2/onnx ^
        --out temp/tiled_holdout_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..utils import imread_unchanged
from .evaluate_particle_measurements import compute_metrics_for_image


def find_flow_inference():
    """Auto-detect C++ flow_inference.exe."""
    candidates = [
        Path("model_flow") / "inference_cpp" / "build" / "Release" / "flow_inference.exe",
        Path("model_flow") / "inference_cpp" / "build" / "Debug" / "flow_inference.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    raise FileNotFoundError(
        "flow_inference.exe not found. "
        "Build with: cd model_flow/inference_cpp && cmake -S . -B build && cmake --build build --config Release"
    )


def validate_recipe_before_cpp(recipe_path: str) -> None:
    """Run the JSON Schema recipe gate before passing a recipe to C++."""
    recipe = Path(recipe_path)
    if not recipe.is_file():
        raise FileNotFoundError(f"analysis_recipe.json not found: {recipe}")

    project_root = Path(__file__).resolve().parents[2]
    validator = project_root / "scripts" / "validate_recipe.py"
    if not validator.is_file():
        raise FileNotFoundError(f"validate_recipe.py not found: {validator}")

    result = subprocess.run(
        [sys.executable, str(validator), str(recipe)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if result.returncode != 0:
        message = result.stdout.strip() or result.stderr.strip()
        raise RuntimeError(
            "analysis_recipe.json failed schema validation before C++ inference:\n"
            f"{message}")


def main():
    parser = argparse.ArgumentParser(description="大图 holdout tile 验证")
    parser.add_argument("--img-dir", required=True, help="holdout 图片目录")
    parser.add_argument("--gt-label-dir", required=True, help="GT 标签目录（*_labels.png）")
    parser.add_argument("--onnx-dir", required=True, help="ONNX 模型目录")
    parser.add_argument("--out", default="temp/tiled_holdout_eval.json")
    parser.add_argument("--cpp-exe", default="", help="C++ flow_inference.exe 路径（自动查找）")
    parser.add_argument("--recipe", default="",
                        help="analysis_recipe.json 路径，用于覆盖 flow_inference_config 默认参数")
    args = parser.parse_args()

    exe = args.cpp_exe or find_flow_inference()
    if args.recipe:
        validate_recipe_before_cpp(args.recipe)

    img_dir = Path(args.img_dir)
    gt_dir = Path(args.gt_label_dir)
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    tmp_dir = os.path.join(os.getcwd(), "temp", "tiled_eval")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        per_image = []
        for img_path in images:
            stem = img_path.stem
            gt_path = gt_dir / f"{stem}_labels.png"
            if not gt_path.exists():
                print(f"  SKIP (no GT): {img_path.name}")
                continue

            # 调用 C++ tile 推理
            out_sub = os.path.join(tmp_dir, stem)
            cmd = [
                exe,
                str(img_path), args.onnx_dir, out_sub,
                "--tile",
            ]
            if args.recipe:
                cmd.extend(["--recipe", args.recipe])
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=600)
            if result.returncode != 0:
                print(f"  ERROR ({img_path.name}): {result.stderr[:200]}")
                continue

            # 读取预测标签
            pred_label_path = Path(out_sub) / f"{stem}_labels.png"
            if not pred_label_path.exists():
                print(f"  SKIP (no output): {img_path.name}")
                continue

            m = compute_metrics_for_image(str(pred_label_path), str(gt_path))
            if m:
                m["file"] = img_path.name
                tile_report_path = Path(out_sub) / "tile_merge_report.json"
                if tile_report_path.exists():
                    with tile_report_path.open("r", encoding="utf-8") as handle:
                        m["tile_merge_report"] = json.load(handle)
                per_image.append(m)

        # 汇总
        report = {
            "n_images": len(per_image),
            "summary": {},
            "images": per_image,
        }
        if per_image:
            report["summary"] = {
                "total_gt": sum(r["n_gt"] for r in per_image),
                "total_pred": sum(r["n_pred"] for r in per_image),
                "total_matched": sum(r["n_matched"] for r in per_image),
                "precision_mean": float(np.mean([r["precision"] for r in per_image])),
                "recall_mean": float(np.mean([r["recall"] for r in per_image])),
                "instance_f1_mean": float(np.mean([r["instance_f1"] for r in per_image])),
                "mask_iou_mean": float(np.mean([r["mask_iou_mean"] for r in per_image])),
                "boundary_iou_mean": float(np.mean([r["boundary_iou_mean"] for r in per_image])),
                "false_positive_count": sum(r["false_positive_count"] for r in per_image),
                "false_negative_count": sum(r["false_negative_count"] for r in per_image),
                "over_split_proxy_count": sum(r["over_split_proxy_count"] for r in per_image),
                "tile_total_tiles": sum(
                    r.get("tile_merge_report", {}).get("total_tiles", 0)
                    for r in per_image),
                "tile_duplicates_merged": sum(
                    r.get("tile_merge_report", {}).get("duplicates_merged", 0)
                    for r in per_image),
                "tile_halo_discarded": sum(
                    r.get("tile_merge_report", {}).get("halo_discarded", 0)
                    for r in per_image),
                "review_required_count": sum(1 for r in per_image if r.get("review_required", False)),
                "review_required_ratio": float(sum(1 for r in per_image if r.get("review_required", False)) / len(per_image)),
            }

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=float)

        s = report["summary"]
        print(f"\nTiled holdout evaluation: {report['n_images']} images")
        print(f"  GT: {s.get('total_gt', 0)}  Pred: {s.get('total_pred', 0)}  "
              f"Matched: {s.get('total_matched', 0)}")
        print(f"  Instance F1: {s.get('instance_f1_mean', 0):.3f}")
        print(f"  Mask IoU: {s.get('mask_iou_mean', 0):.3f}")
        print(f"  Boundary IoU: {s.get('boundary_iou_mean', 0):.3f}")
        print(f"  False positives: {s.get('false_positive_count', 0)}")
        print(f"  False negatives: {s.get('false_negative_count', 0)}")
        print(f"  Over-split proxy: {s.get('over_split_proxy_count', 0)}")
        print(f"  Tiles: {s.get('tile_total_tiles', 0)}  "
              f"Merged: {s.get('tile_duplicates_merged', 0)}  "
              f"Halo discarded: {s.get('tile_halo_discarded', 0)}")
        print(f"  Review required: {s.get('review_required_count', 0)}/{report['n_images']} "
              f"({s.get('review_required_ratio', 0):.1%})")
        print(f"  Report: {args.out}")
    finally:
        pass


if __name__ == "__main__":
    main()
