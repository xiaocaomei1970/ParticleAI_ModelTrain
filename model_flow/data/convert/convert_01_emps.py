"""Legacy: Convert emps-main (Electron Microscopy Particle Segmentation) to COCO JSON.
Superseded by prepare_training_data.py for V1. Do not use directly in V1 pipeline.

- 465 SEM images + uint16 instance segmaps
- All images → data/particles/train/, intermediate JSON → annotations/01_emps.json
- merge_coco.py 将在第 2 步统一做 train/val 分层划分
- Each unique segmap value = one particle instance
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from model_flow.utils import imread_unicode, imread_unchanged, imwrite_unicode, long_path

HERE = Path(__file__).resolve().parents[3]
SRC_BASE = r"E:\MyProjects\已标注数据集\emps-main"
SRC_IMG = os.path.join(SRC_BASE, "images")
SRC_SEG = os.path.join(SRC_BASE, "segmaps")
DST_IMG_DIR = str(HERE / "data" / "particles" / "train")
DST_JSON = str(HERE / "data" / "particles" / "annotations" / "01_emps.json")
PREFIX = "emps"


def extract_instances(segmap):
    """从 uint16 segmap 提取实例。每个唯一非零值 = 一个实例。"""
    instances = []
    instance_ids = np.unique(segmap)
    instance_ids = instance_ids[instance_ids != 0]

    for inst_id in instance_ids:
        inst_mask = (segmap == inst_id).astype(np.uint8)
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
            if area < 1:
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

    # ── collect all segmaps ──
    seg_files = sorted(os.listdir(SRC_SEG))
    print(f"Found {len(seg_files)} segmaps")

    images = []
    annotations = []

    for fname in seg_files:
        seg_path = os.path.join(SRC_SEG, fname)
        img_path = os.path.join(SRC_IMG, fname)

        if not os.path.exists(img_path):
            print(f"  Warning: missing image {fname}, skipping")
            continue

        # ── read segmap (uint16) ──
        segmap = imread_unchanged(seg_path)
        if segmap is None:
            print(f"  Warning: cannot read segmap {fname}, skipping")
            continue

        instances = extract_instances(segmap)
        if len(instances) == 0:
            continue

        # ── copy image ──
        new_name = f"{PREFIX}_{fname}"
        dst_img = os.path.join(DST_IMG_DIR, new_name)
        if not os.path.exists(dst_img):
            img = imread_unicode(img_path)
            if img is None:
                print(f"  Warning: cannot read image {fname}, skipping")
                continue
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imwrite_unicode(long_path(dst_img), img)

        # ── accumulate ──
        h, w = segmap.shape[:2]
        ann_start = len(annotations)
        for inst in instances:
            annotations.append({
                "segmentation": inst["segmentation"],
                "bbox": inst["bbox"],
                "area": inst["area"],
            })

        images.append({
            "file_name": new_name,
            "height": h,
            "width": w,
            "ann_start": ann_start,
            "ann_end": len(annotations),
        })

    # ── assign IDs ──
    for i, img in enumerate(images):
        img["id"] = i + 1
        for k in range(img["ann_start"], img["ann_end"]):
            annotations[k]["id"] = k + 1
            annotations[k]["image_id"] = img["id"]
            annotations[k]["category_id"] = 1
            annotations[k]["iscrowd"] = 0

    # ── build final COCO ──
    coco_images = []
    for img in images:
        coco_images.append({
            "id": img["id"],
            "file_name": img["file_name"],
            "height": img["height"],
            "width": img["width"],
        })

    coco = {
        "images": coco_images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "particle", "supercategory": "none"}],
    }

    with open(DST_JSON, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"Done! {len(coco_images)} images, {len(annotations)} annotations")
    print(f"  Output JSON: {DST_JSON}")
    print(f"  Output imgs: {DST_IMG_DIR}/")


if __name__ == "__main__":
    main()
