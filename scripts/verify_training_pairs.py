"""Verify the image/flow pairs actually used by FlowDataset training.

The training loader does not read COCO annotations. It scans image files under
data/particles/train and data/particles/val, then keeps only images that have a
same-basename .npy file under flows_train or flows_val. This script is the hard
gate for a "complete training dataset" before packaging or training.
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent  # project root
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model_flow.utils import imread_unicode

DATA_ROOT = HERE / "data" / "particles"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(image_dir: Path) -> dict[str, Path]:
    if not image_dir.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        and not path.stem.endswith('_labels')
    }


def collect_flows(flow_dir: Path) -> dict[str, Path]:
    if not flow_dir.is_dir():
        return {}
    return {
        path.stem: path
        for path in sorted(flow_dir.iterdir())
        if path.is_file() and path.suffix == ".npy"
    }


def check_split(split_name: str, image_dir: Path, flow_dir: Path) -> bool:
    """验证 image/flow 配对、可读性、尺寸一致性。返回 True 表示全部通过。"""
    image_dir = Path(image_dir)
    flow_dir = Path(flow_dir)
    print(f"\n=== {split_name} training pairs ===")
    print(f"  image dir: {image_dir}")
    print(f"  flow dir:  {flow_dir}")

    ok = True
    if not image_dir.is_dir():
        print("  ERROR: image directory does not exist.")
        return False
    if not flow_dir.is_dir():
        print("  ERROR: flow directory does not exist.")
        return False

    images = collect_images(image_dir)
    flows = collect_flows(flow_dir)
    missing_flows = sorted(set(images) - set(flows))
    orphan_flows = sorted(set(flows) - set(images))

    print(f"  images: {len(images)}")
    print(f"  flows:  {len(flows)}")
    print(f"  paired: {len(set(images) & set(flows))}")
    print(f"  missing flows: {len(missing_flows)}")
    print(f"  orphan flows:  {len(orphan_flows)}")

    if not images:
        print("  ERROR: no training images found.")
        ok = False
    if not flows:
        print("  ERROR: no .npy flow files found.")
        ok = False
    if missing_flows:
        ok = False
        for name in missing_flows[:20]:
            print(f"    missing flow for image: {images[name].name}")
        if len(missing_flows) > 20:
            print(f"    ... {len(missing_flows) - 20} more")
    if orphan_flows:
        ok = False
        for name in orphan_flows[:20]:
            print(f"    orphan flow without image: {flows[name].name}")
        if len(orphan_flows) > 20:
            print(f"    ... {len(orphan_flows) - 20} more")

    unreadable = []
    bad_flow_shape = []
    bad_flow_dtype = []
    bad_flow_value = []
    size_mismatches = []
    for name in sorted(set(images) & set(flows)):
        img = imread_unicode(str(images[name]))
        if img is None:
            unreadable.append(images[name].name)
            ok = False
            continue
        try:
            flow = np.load(flows[name], mmap_mode="r")
            if flow.ndim != 3 or flow.shape[0] != 3:
                bad_flow_shape.append((flows[name].name, tuple(flow.shape)))
                ok = False
            elif flow.dtype != np.float32:
                bad_flow_dtype.append((flows[name].name, str(flow.dtype)))
                ok = False
            elif not np.isfinite(flow).all():
                bad_flow_value.append(flows[name].name)
                ok = False
            # 硬校验：image 空间尺寸 == flow 空间尺寸
            if img.shape[:2] != flow.shape[1:]:
                size_mismatches.append(
                    (name, img.shape[:2], flow.shape[1:]))
                ok = False
        except Exception as exc:
            bad_flow_shape.append((flows[name].name, f"load error: {exc}"))
            ok = False

    print(f"  unreadable images: {len(unreadable)}")
    print(f"  bad flow shapes:   {len(bad_flow_shape)}")
    print(f"  bad flow dtypes:   {len(bad_flow_dtype)}")
    print(f"  bad flow values:   {len(bad_flow_value)}")
    print(f"  size mismatches:   {len(size_mismatches)}")
    for name in unreadable[:10]:
        print(f"    unreadable image: {name}")
    for name, shape in bad_flow_shape[:10]:
        print(f"    bad flow: {name} shape={shape}")
    for name, dtype in bad_flow_dtype[:10]:
        print(f"    bad flow dtype: {name} dtype={dtype}")
    for name in bad_flow_value[:10]:
        print(f"    bad flow values (NaN/Inf): {name}")
    for name, img_sz, fl_sz in size_mismatches[:10]:
        print(f"    size mismatch: {name} image={img_sz} flow={fl_sz}")

    return ok


def main() -> None:
    ok = True
    ok = ok and check_split("Train", DATA_ROOT / "train", DATA_ROOT / "flows_train")
    ok = ok and check_split("Val", DATA_ROOT / "val", DATA_ROOT / "flows_val")

    # ── 跨 split 泄漏检测 ──
    print("\n=== Cross-split leak check ===")
    train_images = collect_images(DATA_ROOT / "train")
    val_images = collect_images(DATA_ROOT / "val")
    train_names = set(train_images)
    val_names = set(val_images)
    overlap = train_names & val_names
    if overlap:
        ok = False
        print(f"  ERROR: {len(overlap)} filename(s) appear in both train and val:")
        for name in sorted(overlap)[:20]:
            print(f"    {name}")
        if len(overlap) > 20:
            print(f"    ... {len(overlap) - 20} more")
        print("  This causes data leakage and must be fixed before training.")
    else:
        print(f"  No filename overlap (train={len(train_names)}, val={len(val_names)}).")
    # Also check for pixel-level duplicates across all images
    import hashlib
    train_hashes: dict[str, list[str]] = {}
    for stem, path in train_images.items():
        img = imread_unicode(str(path))
        if img is not None:
            h = hashlib.md5(img.tobytes()).hexdigest()
            train_hashes.setdefault(h, []).append(stem)
    val_hashes: dict[str, list[str]] = {}
    for stem, path in val_images.items():
        img = imread_unicode(str(path))
        if img is not None:
            h = hashlib.md5(img.tobytes()).hexdigest()
            val_hashes.setdefault(h, []).append(stem)
    pixel_leaks = []
    for h, train_stems_list in train_hashes.items():
        if h in val_hashes:
            pixel_leaks.append((train_stems_list, val_hashes[h]))
    if pixel_leaks:
        ok = False
        print(f"  ERROR: {len(pixel_leaks)} pixel-level duplicate(s) across train/val "
              f"(different filenames, same decoded pixel content):")
        for train_stems_list, val_stems_list in pixel_leaks[:10]:
            print(f"    train: {train_stems_list}  val: {val_stems_list}")
        if len(pixel_leaks) > 10:
            print(f"    ... {len(pixel_leaks) - 10} more")
        print("  This causes data leakage and must be fixed before training.")

    if not ok:
        print("\nTraining pair verification FAILED.")
        sys.exit(1)
    print("\nTraining pair verification OK.")


if __name__ == "__main__":
    main()
