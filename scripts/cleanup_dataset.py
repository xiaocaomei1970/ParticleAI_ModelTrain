"""清理数据集：修复 bad bbox，删除 mask/冗余文件，uint16→uint8，同步 JSON

用法:
    # 预览 (不实际修改)
    python cleanup_dataset.py --dry-run

    # 执行清理
    python cleanup_dataset.py
"""
# V1 legacy / COCO 辅助 — V1 数据使用 manifest 管理，不使用 COCO JSON 清理流程。
import json
import os
import sys
import argparse

import cv2
import numpy as np

from pathlib import Path

from model_flow.utils import imread_unicode, imwrite_unicode, long_path

HERE = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = str(HERE / "data" / "particles" / "annotations")
TRAIN_IMG_DIR = str(HERE / "data" / "particles" / "train")
VAL_IMG_DIR = str(HERE / "data" / "particles" / "val")


def convert_uint16_to_uint8(img_dir):
    """将 16-bit TIFF 转为 8-bit，避免 OpenCV 警告"""
    converted = 0
    for fname in os.listdir(img_dir):
        path = os.path.join(img_dir, fname)
        img = imread_unicode(path)
        if img is None:
            continue
        if img.dtype == np.uint16:
            # 归一化到 0-255
            img = (img.astype(np.float32) / img.max() * 255).astype(np.uint8)
            imwrite_unicode(path, img)
            converted += 1
    return converted


def remove_mask_files(img_dir):
    """删除误入图片目录的 mask 文件（_mask.tif 等）"""
    removed = 0
    for fname in os.listdir(img_dir):
        if "_mask." in fname.lower():
            os.remove(long_path(os.path.join(img_dir, fname)))
            removed += 1
    return removed


def fix_bboxes(data):
    """修复 bad bbox：从 segmentation 重建零宽高 bbox"""
    valid_anns = []
    bad_count = 0
    for ann in data["annotations"]:
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            bad_count += 1
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            seg = ann.get("segmentation", [])
            if seg and isinstance(seg[0], list) and len(seg[0]) >= 6:
                xs = seg[0][0::2]
                ys = seg[0][1::2]
                x, y = min(xs), min(ys)
                w, h = max(xs) - x, max(ys) - y
                if w > 0 and h > 0:
                    ann["bbox"] = [float(x), float(y), float(w), float(h)]
                    valid_anns.append(ann)
                else:
                    bad_count += 1
            else:
                bad_count += 1
        else:
            valid_anns.append(ann)
    return valid_anns, bad_count


def sync_json_with_disk(json_file, img_dir):
    """删除 JSON 中对应图片不存在的条目"""
    path = os.path.join(ANNOTATIONS_DIR, json_file)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    existing = set(os.listdir(img_dir))
    old_imgs = len(data["images"])
    old_anns = len(data["annotations"])

    data["images"] = [img for img in data["images"] if img["file_name"] in existing]
    valid_ids = {img["id"] for img in data["images"]}
    data["annotations"] = [ann for ann in data["annotations"] if ann["image_id"] in valid_ids]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    return old_imgs - len(data["images"]), old_anns - len(data["annotations"])


def clean_split(json_file, img_dir):
    """对单个 split 执行完整清理流程"""
    # 1. uint16 → uint8 转换
    uint16_cvt = convert_uint16_to_uint8(img_dir)

    # 2. 删除 mask 文件
    masks_removed = remove_mask_files(img_dir)

    # 3. 同步 JSON（删除已不存在文件的条目）
    imgs_removed, anns_removed = sync_json_with_disk(json_file, img_dir)

    # 3. 修复 bad bbox
    path = os.path.join(ANNOTATIONS_DIR, json_file)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_anns, bad_count = fix_bboxes(data)
    data["annotations"] = valid_anns

    # 删除无标注图片
    valid_fnames = {img["file_name"] for img in data["images"]}
    for fname in os.listdir(img_dir):
        if fname not in valid_fnames:
            os.remove(long_path(os.path.join(img_dir, fname)))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"  uint16→uint8 converted: {uint16_cvt}")
    print(f"  Mask files removed: {masks_removed}")
    print(f"  JSON entries removed: {imgs_removed} images, {anns_removed} annotations")
    print(f"  Bad bboxes fixed/skipped: {bad_count}")
    print(f"  Final: {len(data['images'])} images, {len(data['annotations'])} annotations")

    return masks_removed, bad_count, len(data["images"]), len(data["annotations"])


def main():
    parser = argparse.ArgumentParser(description='清理数据集')
    parser.add_argument('--dry-run', action='store_true',
                        help='预览模式，不实际修改任何文件')
    args = parser.parse_args()

    if args.dry_run:
        print("=" * 60)
        print("  DRY RUN — 不会修改任何文件")
        print("=" * 60)

    print("=== Cleaning train ===")
    if not args.dry_run:
        clean_split("instances_train.json", TRAIN_IMG_DIR)
    else:
        # dry-run: 仅统计
        path = os.path.join(ANNOTATIONS_DIR, "instances_train.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Current: {len(data['images'])} images, {len(data['annotations'])} annotations")
        train_imgs = len(os.listdir(TRAIN_IMG_DIR))
        print(f"  Files on disk: {train_imgs}")

    print("\n=== Cleaning val ===")
    if not args.dry_run:
        clean_split("instances_val.json", VAL_IMG_DIR)
    else:
        path = os.path.join(ANNOTATIONS_DIR, "instances_val.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Current: {len(data['images'])} images, {len(data['annotations'])} annotations")
        val_imgs = len(os.listdir(VAL_IMG_DIR))
        print(f"  Files on disk: {val_imgs}")

    # 中间 JSON 列表
    intermediate = [f for f in os.listdir(ANNOTATIONS_DIR)
                    if f not in ("instances_train.json", "instances_val.json")]
    if intermediate:
        print(f"\n  Intermediate JSONs that would be removed: {len(intermediate)}")
        for f in intermediate:
            print(f"    - {f}")
        if not args.dry_run:
            for f in intermediate:
                os.remove(long_path(os.path.join(ANNOTATIONS_DIR, f)))
                print(f"  Removed: {f}")

    if args.dry_run:
        print(f"\n  Dry run complete. Run without --dry-run to apply changes.")
    else:
        print(f"\nDone. Train: {len(os.listdir(TRAIN_IMG_DIR))} images, Val: {len(os.listdir(VAL_IMG_DIR))} images")


if __name__ == '__main__':
    main()
