"""Legacy: Convert 02_NIST_模拟SEM颗粒 dataset to COCO JSON format.
Superseded by prepare_training_data.py for V1. Do not use directly in V1 pipeline.

- 6 sets (set1~set6), each with 567 intensity images and 1 shared binary mask
- Filter: noise <= 34 AND contrast > 6 → 135 images per set, 810 total
- binary mask → connectedComponents → instance contours → polygon segmentation
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from model_flow.utils import imread_unicode, imread_unchanged, imwrite_unicode, long_path

HERE = Path(__file__).resolve().parents[3]
SRC_BASE = r"E:\MyProjects\已标注数据集\02_NIST_模拟SEM颗粒"
INTENSITY_DIR = os.path.join(SRC_BASE, "intensity_sets")
MASK_DIR = os.path.join(SRC_BASE, "mask_sets", "masks")
DST_IMG_DIR = str(HERE / "data" / "particles" / "train")
DST_JSON = str(HERE / "data" / "particles" / "annotations" / "02_nist.json")
PREFIX = "nist"

NOISE_MAX = 34
CONTRAST_MIN = 6


def parse_filename(fname):
    """从 set{N}_cex_noise_{NNN}_contrast_{NNN}.tiff 提取参数"""
    name = fname.replace(".tiff", "")
    parts = name.split("_")
    set_num = int(parts[0][3:])
    noise = int(parts[3])
    contrast = int(parts[5])
    return set_num, noise, contrast


def extract_instances_from_mask(mask):
    """从二值 mask 提取所有实例的 polygon + bbox。

    Returns:
        list[dict]: 每个实例 {"segmentation": [...], "bbox": [...], "area": float}
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

    instances = []

    for label_id in range(1, num_labels):
        inst_mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if len(contour) < 3:
                continue
            pts = contour.flatten().tolist()
            if len(pts) < 6:
                continue
            contour_2d = contour.reshape(-1, 2)
            area = float(cv2.contourArea(contour_2d))
            if area < 4:
                continue
            x, y, bw, bh = cv2.boundingRect(contour_2d)

            instances.append({
                "segmentation": [pts],
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "area": area,
            })

    return instances


def main():
    os.makedirs(DST_IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DST_JSON), exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "particle", "supercategory": "none"}],
    }

    img_id = 0
    ann_id = 0

    for set_num in range(1, 7):
        set_name = f"set{set_num}"
        intensity_dir = os.path.join(INTENSITY_DIR, set_name)
        mask_path = os.path.join(
            MASK_DIR, f"mask_set{set_num}_cex_noise_000_contrast_100.tiff"
        )

        if not os.path.isdir(intensity_dir):
            print(f"Warning: {intensity_dir} not found, skipping set{set_num}")
            continue
        if not os.path.exists(mask_path):
            print(f"Warning: {mask_path} not found, skipping set{set_num}")
            continue

        # ── 读 mask，提取实例（每个 set 只做一次）──
        mask = imread_unchanged(mask_path)
        if mask is None:
            print(f"Warning: cannot read {mask_path}, skipping set{set_num}")
            continue

        instances = extract_instances_from_mask(mask)
        print(
            f"set{set_num}: mask loaded, {len(instances)} instances extracted"
        )

        if len(instances) == 0:
            print(f"  No instances found in mask, skipping set{set_num}")
            continue

        # ── 过滤 intensity 图 ──
        filtered = []
        for fname in sorted(os.listdir(intensity_dir)):
            if not fname.endswith(".tiff"):
                continue
            _, noise, contrast = parse_filename(fname)
            if noise <= NOISE_MAX and contrast > CONTRAST_MIN:
                filtered.append((fname, noise, contrast))

        print(
            f"  Filtered: {len(filtered)} images "
            f"(noise <= {NOISE_MAX}, contrast > {CONTRAST_MIN})"
        )

        # ── 为过滤后的每张图生成 COCO 条目 ──
        for fname, noise, contrast in filtered:
            src_img_path = os.path.join(intensity_dir, fname)

            new_name = (
                f"{PREFIX}_{set_name}_"
                f"noise_{noise:03d}_contrast_{contrast:03d}.png"
            )
            dst_img_path = os.path.join(DST_IMG_DIR, new_name)

            # 复制图片（转为单通道 PNG）
            if not os.path.exists(dst_img_path):
                img = imread_unicode(src_img_path)
                if img is None:
                    print(f"  Warning: cannot read {fname}, skipping")
                    continue
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imwrite_unicode(long_path(dst_img_path), img)

            img_id += 1
            coco["images"].append({
                "id": img_id,
                "file_name": new_name,
                "height": 512,
                "width": 512,
            })

            # 为每个实例创建 annotation
            for inst in instances:
                ann_id += 1
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "segmentation": inst["segmentation"],
                    "bbox": inst["bbox"],
                    "area": inst["area"],
                    "iscrowd": 0,
                })

    # ── 写 COCO JSON ──
    with open(DST_JSON, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"\nDone!")
    print(f"  Images:      {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Output JSON: {DST_JSON}")
    print(f"  Output imgs: {DST_IMG_DIR}/")


if __name__ == "__main__":
    main()
