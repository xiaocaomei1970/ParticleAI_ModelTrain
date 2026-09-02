"""Audit COCO JSON metadata and the local particle training dataset.

This script is an auxiliary dataset audit. The hard training gate is
verify_training_pairs.py, because FlowDataset trains from image/.npy pairs
rather than COCO annotations.

verify_dataset.py still matters after packaging and extracting on ModelScope:
it confirms the annotation JSON files were included, JSON-referenced images are
present, common COCO fields are sane, NIST grouped split did not leak, and the
extracted image/flow files still look consistent.
"""
# V1 legacy / COCO 辅助 — V1 数据不使用 COCO 格式，此脚本仅供旧 pipeline 审计参考。
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent  # project root
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model_flow.utils import imread_unicode

DATA_ROOT = HERE / "data" / "particles"
ANNOTATIONS_DIR = DATA_ROOT / "annotations"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def collect_image_files(image_dir: Path) -> dict[str, Path]:
    if not image_dir.is_dir():
        return {}
    return {
        path.name: path
        for path in sorted(image_dir.iterdir())
        if is_image_file(path)
    }


def collect_flow_files(flow_dir: Path) -> dict[str, Path]:
    if not flow_dir.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(flow_dir.iterdir())
        if path.is_file() and path.suffix == ".npy"
    }


def valid_segmentation(segmentation) -> bool:
    if isinstance(segmentation, list):
        if not segmentation:
            return False
        for polygon in segmentation:
            if not isinstance(polygon, list) or len(polygon) < 6:
                return False
        return True
    if isinstance(segmentation, dict):
        return "counts" in segmentation and "size" in segmentation
    return False


def load_json(split_name: str, json_file: str):
    path = ANNOTATIONS_DIR / json_file
    if not path.exists():
        print(f"\n=== {split_name} ===")
        print(f"  BLOCKER: {json_file} not found at {path}")
        return None, 1
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle), 0


def check_json_split(split_name: str, image_subdir: str, flow_subdir: str,
                     json_file: str) -> int:
    blockers = 0
    image_dir = DATA_ROOT / image_subdir
    flow_dir = DATA_ROOT / flow_subdir
    data, missing_json = load_json(split_name, json_file)
    blockers += missing_json
    if data is None:
        return blockers

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    disk_images = collect_image_files(image_dir)
    disk_image_names = set(disk_images)
    json_image_names = {image.get("file_name", "") for image in images}

    print(f"\n=== {split_name} ===")
    print(f"  JSON: {json_file}")
    print(f"  Categories: {categories}")
    print(f"  Images in JSON: {len(images)}")
    print(f"  Image files on disk: {len(disk_images)}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Avg annotations/image: {len(annotations) / max(len(images), 1):.1f}")

    mask_files = [name for name in disk_image_names if "_mask." in name.lower()]
    print(f"  Mask-like files in image dir: {len(mask_files)}")
    if mask_files:
        print("  WARNING: mask-like files are present in the image directory.")
        for name in sorted(mask_files)[:10]:
            print(f"    - {name}")

    missing_images = sorted(json_image_names - disk_image_names)
    extra_images = sorted(disk_image_names - json_image_names)
    print(f"  Missing image files referenced by JSON: {len(missing_images)}")
    print(f"  Extra image files not in JSON: {len(extra_images)}")
    if missing_images:
        blockers += 1
        for name in missing_images[:20]:
            print(f"    missing image: {name}")
    if extra_images:
        print("  NOTE: extra images can be valid for training if they have same-basename .npy flows.")
        for name in extra_images[:20]:
            print(f"    extra image: {name}")

    image_ids = {image.get("id") for image in images}
    annotation_image_ids = {annotation.get("image_id") for annotation in annotations}
    orphan_annotations = annotation_image_ids - image_ids
    print(f"  Orphan annotations: {len(orphan_annotations)}")
    if orphan_annotations:
        blockers += 1

    bad_segmentations = sum(
        1 for annotation in annotations
        if not valid_segmentation(annotation.get("segmentation"))
    )
    bad_bboxes = sum(
        1 for annotation in annotations
        if not isinstance(annotation.get("bbox"), (list, tuple))
        or len(annotation["bbox"]) != 4
        or annotation["bbox"][2] <= 0
        or annotation["bbox"][3] <= 0
    )
    print(f"  Bad segmentations: {bad_segmentations}")
    print(f"  Bad bboxes: {bad_bboxes}")
    if bad_segmentations or bad_bboxes:
        blockers += 1

    unreadable = [
        name for name, path in disk_images.items()
        if imread_unicode(str(path)) is None
    ]
    print(f"  Unreadable image files: {len(unreadable)}")
    if unreadable:
        blockers += 1
        for name in unreadable[:20]:
            print(f"    unreadable image: {name}")

    sizes = {
        (image.get("height", 0), image.get("width", 0))
        for image in images
    }
    print(f"  Unique JSON image sizes: {len(sizes)} (sample: {list(sizes)[:5]})")

    blockers += check_flow_pairs(split_name, disk_images, flow_dir)
    return blockers


def check_flow_pairs(split_name: str, disk_images: dict[str, Path],
                     flow_dir: Path) -> int:
    blockers = 0
    if not flow_dir.is_dir():
        print(f"\n  {split_name} flow pairing:")
        print(f"    BLOCKER: flow dir not found: {flow_dir}")
        return 1

    image_bases = {Path(name).stem for name in disk_images}
    flows = collect_flow_files(flow_dir)
    flow_bases = set(flows)
    missing_flows = sorted(image_bases - flow_bases)
    orphan_flows = sorted(flow_bases - image_bases)

    print(f"\n  {split_name} flow pairing:")
    print(f"    .npy files: {len(flow_bases)}")
    print(f"    Images:     {len(image_bases)}")
    print(f"    Missing .npy (image has no flow): {len(missing_flows)}")
    print(f"    Orphan .npy (flow has no image):  {len(orphan_flows)}")
    if missing_flows:
        blockers += 1
        for name in missing_flows[:20]:
            print(f"      missing flow: {name}.npy")
    if orphan_flows:
        blockers += 1
        for name in orphan_flows[:20]:
            print(f"      orphan flow: {flows[name].name}")

    bad_flow_shapes = []
    for name in sorted(image_bases & flow_bases):
        try:
            flow = np.load(flows[name], mmap_mode="r")
            if flow.ndim != 3 or flow.shape[0] != 3:
                bad_flow_shapes.append((flows[name].name, tuple(flow.shape)))
        except Exception as exc:
            bad_flow_shapes.append((flows[name].name, f"load error: {exc}"))
    print(f"    Bad flow shapes: {len(bad_flow_shapes)}")
    if bad_flow_shapes:
        blockers += 1
        for name, shape in bad_flow_shapes[:20]:
            print(f"      bad flow: {name} shape={shape}")

    return blockers


def nist_set_ids(json_file: str) -> set[str]:
    path = ANNOTATIONS_DIR / json_file
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    ids = set()
    for image in data.get("images", []):
        match = re.match(r"nist_set(\d+)_", image.get("file_name", "").lower())
        if match:
            ids.add(match.group(1))
    return ids


def main() -> None:
    blockers = 0
    blockers += check_json_split("Train", "train", "flows_train", "instances_train.json")
    blockers += check_json_split("Val", "val", "flows_val", "instances_val.json")

    train_nist_sets = nist_set_ids("instances_train.json")
    val_nist_sets = nist_set_ids("instances_val.json")
    nist_overlap = train_nist_sets & val_nist_sets
    print("\n  NIST grouped split:")
    print(f"    Train sets: {sorted(train_nist_sets)}")
    print(f"    Val sets:   {sorted(val_nist_sets)}")
    print(f"    Overlap:    {sorted(nist_overlap)}")
    if nist_overlap:
        print("  BLOCKER: NIST sets overlap between train and val. Re-run merge_coco.py.")
        blockers += 1

    if ANNOTATIONS_DIR.is_dir():
        print(f"\nAnnotation JSONs: {sorted(os.listdir(ANNOTATIONS_DIR))}")
    else:
        print(f"\nBLOCKER: annotation directory not found: {ANNOTATIONS_DIR}")
        blockers += 1

    if blockers:
        print(f"\nDataset audit FAILED with {blockers} blocker group(s).")
        sys.exit(1)
    print("\nDataset audit OK.")


if __name__ == "__main__":
    main()
