"""合并所有中间 COCO JSON，统一 ID，并按来源分组划分 train/val。

NIST 模拟 SEM 每个 set 共享同一个颗粒几何 mask，只改变 noise/contrast。
因此 NIST 必须按 set 整组划分，避免相同几何布局同时出现在 train 和 val。
"""
# V1 legacy / COCO 辅助 — V1 使用 manifest + split_dataset_by_manifest，不使用 COCO 合并。
import json
import os
import random
import shutil
import glob
import re

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ANNOTATIONS_DIR = str(HERE / "data" / "particles" / "annotations")
TRAIN_IMG_DIR = str(HERE / "data" / "particles" / "train")
VAL_IMG_DIR = str(HERE / "data" / "particles" / "val")
TRAIN_FLOW_DIR = str(HERE / "data" / "particles" / "flows_train")
VAL_FLOW_DIR = str(HERE / "data" / "particles" / "flows_val")
SPLIT_RATIO = 0.8  # 80% train, 20% val
SEED = 42

random.seed(SEED)

os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(TRAIN_FLOW_DIR, exist_ok=True)
os.makedirs(VAL_FLOW_DIR, exist_ok=True)

# 收集所有中间 JSON（排除已生成的最终文件，保证幂等性）
FINAL_JSONS = {"instances_train.json", "instances_val.json"}
json_files = sorted(
    f for f in glob.glob(os.path.join(ANNOTATIONS_DIR, "*.json"))
    if os.path.basename(f) not in FINAL_JSONS
)

all_images = []
all_annotations = []
seen_filenames = set()
global_old_id = 0

for jf in json_files:
    with open(jf, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 重新映射此 JSON 内部的 image_id → global_old_id
    local_to_global = {}
    skipped_mask = 0
    for img in data["images"]:
        fname = img["file_name"]
        if fname in seen_filenames:
            continue
        # 防御：跳过 mask 文件（不应该出现，但以防万一）
        if "_mask." in fname.lower():
            skipped_mask += 1
            continue
        seen_filenames.add(fname)
        global_old_id += 1
        local_to_global[img["id"]] = global_old_id
        img["_global_id"] = global_old_id
        all_images.append(img)

    if skipped_mask > 0:
        print(f"  Skipped {skipped_mask} mask files from {jf}")

    for ann in data["annotations"]:
        if ann["image_id"] in local_to_global:
            ann["image_id"] = local_to_global[ann["image_id"]]
            all_annotations.append(ann)

print(f"Total unique images: {len(all_images)}")
print(f"Total annotations: {len(all_annotations)}")

# 建立全局 old image_id → annotations 的映射
old_img_id_to_anns = {}
for ann in all_annotations:
    oid = ann["image_id"]
    if oid not in old_img_id_to_anns:
        old_img_id_to_anns[oid] = []
    old_img_id_to_anns[oid].append(ann)

# 只保留有标注的图片
valid_images = [img for img in all_images if img["_global_id"] in old_img_id_to_anns]
print(f"Images with annotations: {len(valid_images)}")

# 检查哪些图片文件实际存在 (同时检查 train 和 val, 保证幂等)
img_files_exist = set(os.listdir(TRAIN_IMG_DIR))
if os.path.isdir(VAL_IMG_DIR):
    img_files_exist |= set(os.listdir(VAL_IMG_DIR))
valid_images_existing = [img for img in valid_images if img["file_name"] in img_files_exist]
print(f"Images with existing files: {len(valid_images_existing)}")
missing = len(valid_images) - len(valid_images_existing)
if missing > 0:
    print(f"  Missing files: {missing}")

valid_images = valid_images_existing

if len(valid_images) == 0:
    print("ERROR: No valid images found!")
    exit(1)

# 显微镜类型检测
def detect_microscope_type(fname):
    fname_l = fname.lower()
    if fname_l.startswith('nist_'):
        return 'NIST'
    if 'emps' in fname_l or fname_l.startswith('01_'):
        return 'SEM'
    if fname_l.startswith('tio2_sem_'):
        return 'SEM'
    if fname_l.startswith('tio2_tsem_'):
        return 'TSEM'
    if 'cryo' in fname_l or fname_l.startswith('03_'):  # 03_ 为计划中的 CryoEM 数据集，convert_03_*.py 尚未实现
        return 'CryoEM'
    if 'nnp' in fname_l or fname_l.startswith('04_'):
        return 'TEM'
    return 'Other'


def split_group_key(img):
    """返回划分分组 key。

    NIST 按 set 整组划分；其他来源按单张图片划分。
    """
    fname = img['file_name'].lower()
    match = re.match(r'nist_set(\d+)_', fname)
    if match:
        return f'nist_set{match.group(1)}'
    return img['file_name']


def split_grouped_images(group):
    """按 group key 做稳定随机划分，并尽量接近 SPLIT_RATIO。"""
    grouped = {}
    for img in group:
        grouped.setdefault(split_group_key(img), []).append(img)

    keys = sorted(grouped.keys())
    random.shuffle(keys)

    # 对 NIST 这种少量大组，按组数 round 可避免 6 组时只取 4 组训练导致比例过低。
    target_train_groups = int(round(len(keys) * SPLIT_RATIO))
    target_train_groups = min(max(target_train_groups, 1), len(keys) - 1) if len(keys) > 1 else len(keys)
    train_keys = set(keys[:target_train_groups])

    train_part = []
    val_part = []
    for key in keys:
        if key in train_keys:
            train_part.extend(grouped[key])
        else:
            val_part.extend(grouped[key])
    return train_part, val_part


def sync_files_for_split(train_images, val_images, src_train_dir, src_val_dir):
    """按新 split 同步文件位置，支持从旧 train/val 双向移动。"""
    train_names = {img["file_name"] for img in train_images}
    val_names = {img["file_name"] for img in val_images}

    def ensure_in_target(fname, target_dir, other_dir):
        target_path = os.path.join(target_dir, fname)
        other_path = os.path.join(other_dir, fname)
        if os.path.exists(target_path):
            return
        if os.path.exists(other_path):
            shutil.move(other_path, target_path)

    for fname in sorted(train_names):
        ensure_in_target(fname, src_train_dir, src_val_dir)
    for fname in sorted(val_names):
        ensure_in_target(fname, src_val_dir, src_train_dir)

    # 清理 split 外残留文件。只删除另一侧已有目标归属的文件，避免误删未知文件。
    for fname in sorted(train_names):
        stray = os.path.join(src_val_dir, fname)
        if os.path.exists(stray):
            os.remove(stray)
    for fname in sorted(val_names):
        stray = os.path.join(src_train_dir, fname)
        if os.path.exists(stray):
            os.remove(stray)

# ── 按显微镜类型分层划分 ──
# 先分组
by_type = {}
for img in valid_images:
    mtype = detect_microscope_type(img['file_name'])
    by_type.setdefault(mtype, []).append(img)

train_images = []
val_images = []

for mtype in sorted(by_type.keys()):
    group = by_type[mtype]
    train_part, val_part = split_grouped_images(group)
    train_images.extend(train_part)
    val_images.extend(val_part)
    print(f'  {mtype}: {len(group)} total → {len(train_part)} train / {len(val_part)} val')

# 合并后再次 shuffle，确保跨类型混合
random.shuffle(train_images)
random.shuffle(val_images)

print(f'\nTrain: {len(train_images)} images')
print(f'Val:   {len(val_images)} images')

# 重新分配 image_id
old_to_new_img = {}
new_train_images = []
new_val_images = []

for i, img in enumerate(train_images):
    old_to_new_img[img["_global_id"]] = (i + 1, "train")
    new_train_images.append({
        "id": i + 1,
        "file_name": img["file_name"],
        "height": img["height"],
        "width": img["width"],
    })

for i, img in enumerate(val_images):
    old_to_new_img[img["_global_id"]] = (i + 1, "val")
    new_val_images.append({
        "id": i + 1,
        "file_name": img["file_name"],
        "height": img["height"],
        "width": img["width"],
    })

# 重新分配 annotation_id，分配 annotations 到对应的 split
train_annotations = []
val_annotations = []

for img in train_images:
    oid = img["_global_id"]
    for ann in old_img_id_to_anns.get(oid, []):
        new_img_id, _ = old_to_new_img[oid]
        train_annotations.append({
            "id": len(train_annotations) + 1,
            "image_id": new_img_id,
            "category_id": 1,
            "segmentation": ann["segmentation"],
            "bbox": ann["bbox"],
            "area": ann["area"],
            "iscrowd": ann.get("iscrowd", 0),
        })

for img in val_images:
    oid = img["_global_id"]
    for ann in old_img_id_to_anns.get(oid, []):
        new_img_id, _ = old_to_new_img[oid]
        val_annotations.append({
            "id": len(val_annotations) + 1,
            "image_id": new_img_id,
            "category_id": 1,
            "segmentation": ann["segmentation"],
            "bbox": ann["bbox"],
            "area": ann["area"],
            "iscrowd": ann.get("iscrowd", 0),
        })

# 写入 COCO JSON
categories = [{"id": 1, "name": "particle", "supercategory": "none"}]

train_coco = {
    "images": new_train_images,
    "annotations": train_annotations,
    "categories": categories,
}
val_coco = {
    "images": new_val_images,
    "annotations": val_annotations,
    "categories": categories,
}

# 同步图片和已预计算 flow 的 split 位置，保证重复执行后目录与 JSON 一致。
sync_files_for_split(train_images, val_images, TRAIN_IMG_DIR, VAL_IMG_DIR)
flow_train_images = [
    {"file_name": os.path.splitext(img["file_name"])[0] + ".npy"}
    for img in train_images
]
flow_val_images = [
    {"file_name": os.path.splitext(img["file_name"])[0] + ".npy"}
    for img in val_images
]
sync_files_for_split(flow_train_images, flow_val_images, TRAIN_FLOW_DIR, VAL_FLOW_DIR)

with open(os.path.join(ANNOTATIONS_DIR, "instances_train.json"), "w", encoding="utf-8") as f:
    json.dump(train_coco, f, ensure_ascii=False)
with open(os.path.join(ANNOTATIONS_DIR, "instances_val.json"), "w", encoding="utf-8") as f:
    json.dump(val_coco, f, ensure_ascii=False)

print(f"\nDone! Train: {len(train_images)} images, {len(train_annotations)} annotations")
print(f"      Val:   {len(val_images)} images, {len(val_annotations)} annotations")
print(f"Train JSON: {os.path.join(ANNOTATIONS_DIR, 'instances_train.json')}")
print(f"Val JSON:   {os.path.join(ANNOTATIONS_DIR, 'instances_val.json')}")
