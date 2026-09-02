"""Legacy: Convert TiO2 SEM/TSEM particle masks to COCO JSON.
Superseded by prepare_training_data.py for V1. Do not use directly in V1 pipeline.

- SEM: original image + manual 4-connected binary mask, 40 samples, 1024x768
- TSEM: original image cropped to TSEM binary mask size, 40 samples, 1024x712
- Binary mask foreground is non-zero; instances are extracted with 4-connectivity.
- Outputs images into data/particles/train/ and annotations/05_tio2.json.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

from model_flow.utils import imread_unicode, imread_unchanged, imwrite_unicode, long_path

HERE = Path(__file__).resolve().parents[3]
SRC_BASE = Path(r"E:\MyProjects\已标注数据集\TiO2")
SRC_IMAGE_BASE = SRC_BASE / "Electron Microscopy Images"
SRC_MASK_BASE = SRC_BASE / "Electron Microscopy Image Masks"

SEM_IMAGE_DIR = SRC_IMAGE_BASE / "SEM"
TSEM_IMAGE_DIR = SRC_IMAGE_BASE / "TSEM"
SEM_MASK_DIR = SRC_MASK_BASE / "TiO2_Masks_Manual_4connected"
TSEM_MASK_DIR = SRC_MASK_BASE / "TiO2_Masks_TSEM"

DST_IMG_DIR = HERE / "data" / "particles" / "train"
DST_JSON = HERE / "data" / "particles" / "annotations" / "05_tio2.json"

MIN_AREA = 4
CONNECTIVITY = 4


def extract_instances_from_binary_mask(mask: np.ndarray) -> tuple[list[dict], int]:
    """Extract particle instances from a binary mask using 4-connectivity."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    binary = (mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary, connectivity=CONNECTIVITY)

    instances = []
    skipped_small = 0

    for label_id in range(1, num_labels):
        instance_mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if len(contour) < 3:
                continue
            points = contour.flatten().tolist()
            if len(points) < 6:
                continue

            contour_2d = contour.reshape(-1, 2)
            area = float(cv2.contourArea(contour_2d))
            if area < MIN_AREA:
                skipped_small += 1
                continue

            x, y, width, height = cv2.boundingRect(contour_2d)
            instances.append({
                "segmentation": [points],
                "bbox": [float(x), float(y), float(width), float(height)],
                "area": area,
            })

    return instances, skipped_small


def write_training_image(
    source_path: Path,
    destination_name: str,
    target_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Copy an image into the training image directory as grayscale PNG."""
    image = imread_unicode(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read: {source_path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if target_shape is not None:
        target_height, target_width = target_shape
        image_height, image_width = image.shape[:2]
        if image_height < target_height or image_width < target_width:
            raise ValueError(
                f"Cannot crop {source_path.name} from "
                f"{(image_height, image_width)} to {target_shape}"
            )
        image = image[:target_height, :target_width]

    destination_path = DST_IMG_DIR / destination_name
    imwrite_unicode(long_path(str(destination_path)), image)

    height, width = image.shape[:2]
    return height, width


def append_sample(
    coco: dict,
    image_path: Path,
    mask_path: Path,
    output_name: str,
) -> tuple[int, int]:
    """Append one image and its extracted annotations to a COCO dict."""
    mask = imread_unchanged(str(mask_path))
    instances, skipped_small = extract_instances_from_binary_mask(mask)
    if not instances:
        print(f"  Warning: no valid instances in {mask_path.name}, skipped")
        return 0, skipped_small

    mask_height, mask_width = mask.shape[:2]
    image_height, image_width = write_training_image(
        image_path,
        output_name,
        target_shape=(mask_height, mask_width),
    )
    if (image_height, image_width) != (mask_height, mask_width):
        raise ValueError(
            f"Image/mask shape mismatch for {output_name}: "
            f"image={(image_height, image_width)}, mask={(mask_height, mask_width)}"
        )

    image_id = len(coco["images"]) + 1
    coco["images"].append({
        "id": image_id,
        "file_name": output_name,
        "height": image_height,
        "width": image_width,
    })

    for instance in instances:
        annotation_id = len(coco["annotations"]) + 1
        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": instance["segmentation"],
            "bbox": instance["bbox"],
            "area": instance["area"],
            "iscrowd": 0,
        })

    return len(instances), skipped_small


def convert_sem(coco: dict) -> tuple[int, int, int]:
    """Convert TiO2 SEM samples."""
    sample_count = 0
    annotation_count = 0
    skipped_small_count = 0

    for mask_path in sorted(SEM_MASK_DIR.glob("*_m.tif")):
        stem = mask_path.stem.removesuffix("_m")
        image_path = SEM_IMAGE_DIR / f"{stem}.tif"
        if not image_path.exists():
            print(f"  Warning: missing SEM image for {mask_path.name}")
            continue

        output_name = f"tio2_sem_{stem}.png"
        added, skipped_small = append_sample(coco, image_path, mask_path, output_name)
        if added:
            sample_count += 1
            annotation_count += added
        skipped_small_count += skipped_small

    return sample_count, annotation_count, skipped_small_count


def convert_tsem(coco: dict) -> tuple[int, int, int]:
    """Convert TiO2 TSEM samples from original images cropped to mask size."""
    sample_count = 0
    annotation_count = 0
    skipped_small_count = 0

    mask_paths = [
        path for path in sorted(TSEM_MASK_DIR.glob("*.tif"))
        if path.is_file()
    ]

    for mask_path in mask_paths:
        tsem_stem = mask_path.stem
        image_path = TSEM_IMAGE_DIR / f"{tsem_stem}.tif"
        if not image_path.exists():
            print(f"  Warning: missing TSEM image for {mask_path.name}")
            continue

        output_name = f"tio2_tsem_{tsem_stem}.png"
        added, skipped_small = append_sample(coco, image_path, mask_path, output_name)
        if added:
            sample_count += 1
            annotation_count += added
        skipped_small_count += skipped_small

    return sample_count, annotation_count, skipped_small_count


def main():
    os.makedirs(DST_IMG_DIR, exist_ok=True)
    os.makedirs(DST_JSON.parent, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "particle", "supercategory": "none"}],
    }

    print("=== Converting TiO2 SEM ===")
    sem_images, sem_annotations, sem_skipped = convert_sem(coco)
    print(
        f"  SEM: {sem_images} images, {sem_annotations} annotations, "
        f"{sem_skipped} small (<{MIN_AREA}px) removed"
    )

    print("=== Converting TiO2 TSEM ===")
    tsem_images, tsem_annotations, tsem_skipped = convert_tsem(coco)
    print(
        f"  TSEM: {tsem_images} images, {tsem_annotations} annotations, "
        f"{tsem_skipped} small (<{MIN_AREA}px) removed"
    )

    with open(DST_JSON, "w", encoding="utf-8") as file:
        json.dump(coco, file, ensure_ascii=False)

    print("\nDone!")
    print(f"  Images:      {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Output JSON: {DST_JSON}")
    print(f"  Output imgs: {DST_IMG_DIR}")


if __name__ == "__main__":
    main()
