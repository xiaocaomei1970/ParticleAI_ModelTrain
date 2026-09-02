"""Legacy: Convert 04_nNPipe_HRTEM催化颗粒 dataset to COCO JSON format.
Superseded by prepare_training_data.py for V1. Do not use directly in V1 pipeline.

- Use only au_ge (93 samples).
- pd_c 300kX/400kX is excluded because particle boundaries are too ambiguous
  against the carbon support background for reliable particle-size training.
- Binary mask (background=0, particle=255), 1:1 mapping
- connectedComponents → contours → polygon + bbox
- Filter: pixel area >= 16
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from model_flow.utils import imread_unicode, imwrite_unicode, long_path

HERE = Path(__file__).resolve().parents[3]
EXPERIMENTAL = (
    r"E:\MyProjects\已标注数据集\04_nNPipe_HRTEM催化颗粒"
    r"\nNPipe_resources\nNPipe_resources\experimental_images"
)
DST_IMG_DIR = str(HERE / "data" / "particles" / "train")
DST_JSON = str(HERE / "data" / "particles" / "annotations" / "04_nnp.json")
MIN_AREA = 16


def extract_instances(mask):
    """从二值 mask (0=背景, 255=颗粒) 提取实例。过滤面积 < MIN_AREA。

    Returns:
        instances: list[dict] with keys segmentation, bbox, area
        skipped_small: int, number of small instances removed
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

    instances = []
    skipped_small = 0

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
            if area < MIN_AREA:
                skipped_small += 1
                continue
            x, y, bw, bh = cv2.boundingRect(contour_2d)

            instances.append({
                "segmentation": [pts],
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "area": area,
            })

    return instances, skipped_small


def process_subset(img_rel_dir, gt_rel_dir, prefix):
    """处理一个子集。

    Returns:
        images: list[dict] with keys file_name, height, width, ann_count
        annotations: list[dict] flat list of annotations
        skipped_small: int
    """
    img_dir = os.path.join(EXPERIMENTAL, img_rel_dir)
    gt_dir = os.path.join(EXPERIMENTAL, gt_rel_dir)

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".tif")])

    images = []
    annotations = []
    total_skipped = 0

    for fname in img_files:
        name = os.path.splitext(fname)[0]
        gt_path = os.path.join(gt_dir, f"{name}_gt.tif")

        if not os.path.exists(gt_path):
            print(f"  Warning: no GT for {fname}, skipping")
            continue

        # ── read mask ──
        mask = imread_unicode(gt_path)
        if mask is None:
            print(f"  Warning: cannot read GT {gt_path}, skipping")
            continue

        instances, skipped = extract_instances(mask)
        total_skipped += skipped

        if len(instances) == 0:
            continue

        # ── copy image ──
        new_name = f"{prefix}_{name}.png"
        dst_img = os.path.join(DST_IMG_DIR, new_name)
        if not os.path.exists(dst_img):
            img = imread_unicode(os.path.join(img_dir, fname))
            if img is None:
                print(f"  Warning: cannot read {fname}, skipping")
                continue
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imwrite_unicode(long_path(dst_img), img)

        # ── accumulate ──
        ann_start = len(annotations)
        for inst in instances:
            annotations.append({
                "segmentation": inst["segmentation"],
                "bbox": inst["bbox"],
                "area": inst["area"],
            })

        images.append({
            "file_name": new_name,
            "height": 2048,
            "width": 2048,
            "ann_start": ann_start,
            "ann_end": len(annotations),
        })

    return images, annotations, total_skipped


def main():
    os.makedirs(DST_IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DST_JSON), exist_ok=True)

    subsets = [
        ("au_ge",     r"au_ge\img",     r"au_ge\gt",    "nnp_auge"),
    ]

    all_images = []
    all_annotations = []
    total_skipped = 0

    for label, img_rel, gt_rel, prefix in subsets:
        print(f"Processing {label}...")
        imgs, anns, skipped = process_subset(img_rel, gt_rel, prefix)
        # Adjust annotation offsets for cross-subset merging
        base = len(all_annotations)
        for img in imgs:
            img["ann_start"] += base
            img["ann_end"] += base
        all_images.extend(imgs)
        all_annotations.extend(anns)
        total_skipped += skipped
        print(f"  {len(imgs)} images, {len(anns)} annotations, "
              f"{skipped} small (<{MIN_AREA}px) removed")

    # ── assign final IDs ──
    for i, img in enumerate(all_images):
        img["id"] = i + 1

    for j, img in enumerate(all_images):
        for k in range(img["ann_start"], img["ann_end"]):
            all_annotations[k]["id"] = k + 1
            all_annotations[k]["image_id"] = img["id"]
            all_annotations[k]["category_id"] = 1
            all_annotations[k]["iscrowd"] = 0

    # ── clean up temporary keys ──
    coco_images = []
    for img in all_images:
        coco_images.append({
            "id": img["id"],
            "file_name": img["file_name"],
            "height": img["height"],
            "width": img["width"],
        })

    coco = {
        "images": coco_images,
        "annotations": all_annotations,
        "categories": [{"id": 1, "name": "particle", "supercategory": "none"}],
    }

    with open(DST_JSON, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)

    print(f"\nDone!")
    print(f"  Images:      {len(coco_images)}")
    print(f"  Annotations: {len(all_annotations)}")
    print(f"  Small removed: {total_skipped}")
    print(f"  Output JSON: {DST_JSON}")


if __name__ == "__main__":
    main()
