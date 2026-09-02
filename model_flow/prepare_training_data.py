"""统一准备训练数据：扫描子目录、检测标注格式、转换为 * _labels.png、汇聚到 staging 目录。

用法:
    # 全量扫描
    python -m model_flow.prepare_training_data ^
        --src-root E:\MyProjects\已标注数据集 ^
        --subset * ^
        --out-img-dir temp/staging/images ^
        --out-label-dir temp/staging/labels ^
        --out-manifest temp/staging/dataset_manifest_draft.csv

    # 增量追加（步骤 3 新增子目录）
    python -m model_flow.prepare_training_data ^
        --src-root E:\MyProjects\已标注数据集 ^
        --subset my_new_data_sem ^
        --out-img-dir temp/staging/images ^
        --out-label-dir temp/staging/labels ^
        --out-manifest temp/staging/dataset_manifest_draft.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .utils import (imread_unicode, imread_unchanged, imwrite_unicode, long_path,
                     validate_instance_label)


class DataPreparationError(RuntimeError):
    """Raised when strict data preparation detects a blocking input issue."""


# ── 显微镜类型后缀 → microscope_type ──
SUFFIX_MAP = {
    "_sem": "SEM",
    "_tem": "TEM",
    "_tsem": "TSEM",
    "_optical": "optical",
}

# ── 格式检测规则（基于目录名子串匹配） ──
FORMAT_RULES = [
    ("emps", "emps"),   # EMPS: segmaps + images
    ("NIST", "nist"),   # NIST: shared binary mask
    ("nNPipe", "nnp"),  # nNPipe: experimental_images
    ("TiO2", "tio2"),   # TiO2
]


def detect_format(dir_name: str) -> str:
    """根据目录名检测标注格式。找不到返回 'generic'。"""
    for keyword, fmt in FORMAT_RULES:
        if keyword.lower() in dir_name.lower():
            return fmt
    return "generic"


def detect_microscope_type(dir_name: str) -> str:
    """从目录名后缀识别显微镜类型，默认 'unknown'。"""
    dir_name_lower = dir_name.lower()
    for suffix, mtype in SUFFIX_MAP.items():
        if dir_name_lower.endswith(suffix):
            return mtype
    return "unknown"


def tile_rule(long_side: int) -> bool:
    """长边 >1536 视为大图。"""
    return long_side > 1536


def is_large_image(img_path: str) -> bool:
    """读取图片尺寸并判断是否大图。"""
    img = imread_unicode(img_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    return tile_rule(max(h, w))


# ═══════════════════════════════════════════════════════════════
# 格式转换核心
# ═══════════════════════════════════════════════════════════════

def _save_label_mask(labels: np.ndarray, out_label_path: str):
    """保存 uint16 labels。确保目录存在。"""
    os.makedirs(os.path.dirname(out_label_path), exist_ok=True)
    imwrite_unicode(long_path(out_label_path), labels.astype(np.uint16),
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])


def _copy_image(src_img: str, dst_img: str):
    """复制图片，确保目录存在。"""
    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    shutil.copy2(src_img, long_path(dst_img))


def convert_emps(subset_path: Path, out_img_dir: str, out_label_dir: str,
                 force: bool, prefix: str = "",
                 strict_pairing: bool = True) -> list[dict]:
    """EMPS 格式：uint16 segmap 直接作为标签图。"""
    src_img_dir = subset_path / "images"
    src_seg_dir = subset_path / "segmaps"
    if not src_img_dir.is_dir() or not src_seg_dir.is_dir():
        return []

    rows = []
    for seg_name in sorted(os.listdir(str(src_seg_dir))):
        seg_path = str(src_seg_dir / seg_name)
        img_path = str(src_img_dir / seg_name)

        if not os.path.exists(img_path):
            continue

        seg = imread_unchanged(seg_path)
        if seg is None or seg.max() == 0:
            continue

        # 校验 image 与 segmap 尺寸一致
        img_data = imread_unicode(img_path)
        if img_data is not None and seg.shape[:2] != img_data.shape[:2]:
            print(f"  WARNING: EMPS segmap size {seg.shape[:2]} != image size "
                  f"{img_data.shape[:2]} for {seg_name}, skipping")
            if strict_pairing:
                raise DataPreparationError(
                    f"EMPS size mismatch: {seg_name} segmap={seg.shape[:2]} image={img_data.shape[:2]}")
            continue

        base = f"{prefix}_{Path(seg_name).stem}" if prefix else Path(seg_name).stem
        dst_img = os.path.join(out_img_dir, base + Path(img_path).suffix.lower())
        dst_label = os.path.join(out_label_dir, base + "_labels.png")

        if not force and os.path.exists(dst_label):
            continue

        _copy_image(img_path, dst_img)
        _save_label_mask(seg, dst_label)

        rows.append({
            "file_name": base + Path(img_path).suffix.lower(),
            "label_file": base + "_labels.png",
            "image_path": dst_img,
            "label_path": dst_label,
            "is_large_image": str(is_large_image(img_path)).lower(),
            "notes": "auto_instance_from_binary_mask_requires_review",
        })
    return rows


def _conncomp_label(mask: np.ndarray, min_area: int = 1) -> np.ndarray:
    """二值 mask → 连通分量 → uint16 标签图。"""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = (mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(binary, connectivity=8)
    label_out = np.zeros(labels.shape, dtype=np.uint16)

    # 跳过背景 0，过滤小面积
    for lid in range(1, num):
        area = (labels == lid).sum()
        if area >= min_area:
            label_out[labels == lid] = lid
    return label_out


def convert_mask_only(subset_path: Path, out_img_dir: str, out_label_dir: str,
                      force: bool, prefix: str = "",
                      img_subdir: str = "", mask_subdir: str = "",
                      filter_fn=None, strict_pairing: bool = True) -> list[dict]:
    """通用 mask 格式：图片目录 + 独立 mask 文件 → 连通分量生成标签。"""
    src_img_dir = subset_path / img_subdir if img_subdir else subset_path
    src_mask_dir = subset_path / mask_subdir if mask_subdir else subset_path
    if not src_img_dir.is_dir() or not src_mask_dir.is_dir():
        return []

    # 收集 mask 文件, 建立精确 stem 映射
    # 规则: 图片 foo.ext 对应标签 foo_labels.png
    # mask stem 去掉 _labels 后缀作为 key; 也保留原始 stem 作为 fallback
    mask_by_key: dict[str, str] = {}
    key_conflicts: list[str] = []
    for mf in sorted(os.listdir(str(src_mask_dir))):
        if not mf.lower().endswith((".png", ".tif", ".tiff", ".jpg")):
            continue
        m_stem = Path(mf).stem
        if m_stem.endswith("_labels"):
            key = m_stem[:-len("_labels")]
        else:
            key = m_stem
        if key in mask_by_key:
            key_conflicts.append(key)
        mask_by_key[key] = str(src_mask_dir / mf)
    if key_conflicts:
        message = (f"duplicate label key(s) in {src_mask_dir}: "
                   f"{key_conflicts[:10]}")
        print(f"  ERROR: {message}")
        if strict_pairing:
            raise DataPreparationError(message)
        return []

    rows = []
    missing_labels: list[str] = []
    for img_name in sorted(os.listdir(str(src_img_dir))):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            continue
        stem = Path(img_name).stem
        img_path = str(src_img_dir / img_name)

        # 精确 stem 匹配 (不再使用 substring)
        mask_path = mask_by_key.get(stem)
        if mask_path is None:
            missing_labels.append(img_name)
            continue

        # 可选过滤
        if filter_fn and not filter_fn(stem, img_path):
            continue

        mask = imread_unchanged(mask_path)
        if mask is None:
            if strict_pairing:
                raise DataPreparationError(f"cannot read label file: {mask_path}")
            continue

        # 校验 image 与 mask 尺寸一致
        img_check = imread_unicode(img_path)
        if img_check is not None and mask.shape[:2] != img_check.shape[:2]:
            msg = (f"mask size {mask.shape[:2]} != image size {img_check.shape[:2]}"
                   f" for {img_name}")
            print(f"  WARNING: {msg}")
            if strict_pairing:
                raise DataPreparationError(msg)
            continue

        labels = _conncomp_label(mask, min_area=4)
        if labels.max() == 0:
            continue

        base = f"{prefix}_{stem}" if prefix else stem
        dst_img = os.path.join(out_img_dir, base + Path(img_path).suffix.lower())
        dst_label = os.path.join(out_label_dir, base + "_labels.png")

        if not force and os.path.exists(dst_label):
            continue

        _copy_image(img_path, dst_img)
        _save_label_mask(labels, dst_label)

        rows.append({
            "file_name": base + Path(img_path).suffix.lower(),
            "label_file": base + "_labels.png",
            "image_path": dst_img,
            "label_path": dst_label,
            "is_large_image": str(is_large_image(img_path)).lower(),
            "notes": "auto_instance_from_binary_mask_requires_review",
        })

    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} image(s) in {src_img_dir} "
              f"have no matching label file in {src_mask_dir}")
        if len(missing_labels) <= 10:
            for name in missing_labels:
                print(f"    missing label for: {name}")
        if strict_pairing:
            raise DataPreparationError(
                f"strict pairing enabled; {len(missing_labels)} image(s) in "
                f"{src_img_dir} have no matching label file in {src_mask_dir}. "
                f"Use --no-strict-pairing only for explicit debugging.")
    return rows


def convert_nist(subset_path: Path, out_img_dir: str, out_label_dir: str,
                 force: bool, prefix: str = "nist",
                 strict_pairing: bool = True) -> list[dict]:
    """NIST 格式：intensity_sets + mask_sets/masks。"""
    intensity_dir = subset_path / "intensity_sets"
    mask_dir = subset_path / "mask_sets" / "masks"
    if not intensity_dir.is_dir() or not mask_dir.is_dir():
        return []

    def nist_filter(stem: str, _img_path: str) -> bool:
        try:
            parts = stem.split("_")
            noise = int(parts[3])
            contrast = int(parts[5])
            return noise <= 34 and contrast > 6
        except (IndexError, ValueError):
            return False

    return convert_mask_only(
        subset_path, out_img_dir, out_label_dir, force,
        prefix=prefix, img_subdir="intensity_sets",
        mask_subdir=str(mask_dir.relative_to(subset_path)),
        filter_fn=nist_filter,
        strict_pairing=strict_pairing,
    )


def convert_nnp(subset_path: Path, out_img_dir: str, out_label_dir: str,
                force: bool, prefix: str = "nnp",
                strict_pairing: bool = True) -> list[dict]:
    """nNPipe 格式：experimental_images 下各子集。"""
    exp_dir = subset_path / "nNPipe_resources" / "nNPipe_resources" / "experimental_images"
    if not exp_dir.is_dir():
        return []

    rows = []
    for subdir in sorted(os.listdir(str(exp_dir))):
        img_dir = exp_dir / subdir / "image"
        mask_dir = exp_dir / subdir / "target"
        if not img_dir.is_dir() or not mask_dir.is_dir():
            continue

        sub_prefix = f"{prefix}_{subdir}" if prefix != subdir else subdir
        sub_rows = convert_mask_only(
            exp_dir / subdir, out_img_dir, out_label_dir, force,
            prefix=sub_prefix, img_subdir="image", mask_subdir="target",
            strict_pairing=strict_pairing,
        )
        rows.extend(sub_rows)
    return rows


def convert_tio2(subset_path: Path, out_img_dir: str, out_label_dir: str,
                 force: bool, prefix: str = "tio2",
                 strict_pairing: bool = True) -> list[dict]:
    """TiO2 格式：假设有 image 和 mask 子目录。"""
    for img_sub, mask_sub in [("image", "mask"), ("images", "masks"), ("img", "gt")]:
        rows = convert_mask_only(
            subset_path, out_img_dir, out_label_dir, force,
            prefix=prefix, img_subdir=img_sub, mask_subdir=mask_sub,
            strict_pairing=strict_pairing,
        )
        if rows:
            return rows
    return []


def convert_generic(subset_path: Path, out_img_dir: str, out_label_dir: str,
                    force: bool, prefix: str = "",
                    strict_pairing: bool = True) -> list[dict]:
    """通用格式：目录下已有 *_labels.png，直接验证复制。"""
    rows = []
    missing_labels: list[str] = []
    for fname in sorted(os.listdir(str(subset_path))):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            continue
        stem = Path(fname).stem
        # 跳过 *_labels.png 自身，避免将其当作原图而寻找 _labels_labels.png
        if stem.endswith("_labels"):
            continue
        img_path = str(subset_path / fname)
        label_path = str(subset_path / f"{stem}_labels.png")
        if not os.path.exists(label_path):
            missing_labels.append(fname)
            continue

        # 校验标签合法性（含尺寸一致性）
        try:
            label_data = imread_unchanged(label_path)
        except Exception as exc:
            print(f"  WARNING: cannot read label {label_path}: {exc}")
            if strict_pairing:
                raise DataPreparationError(f"cannot read label: {label_path}") from exc
            continue
        label_errors = validate_instance_label(label_data, max_area_fraction=0.8)
        if label_errors:
            for err in label_errors:
                print(f"  WARNING: {label_path}: {err}")
            if strict_pairing:
                raise DataPreparationError(
                    f"invalid label {label_path}: {'; '.join(label_errors)}")
            continue

        # 硬校验：image 和 label 空间尺寸必须一致
        try:
            img_data = imread_unicode(img_path)
            if img_data is None:
                raise DataPreparationError(f"cannot read image: {img_path}")
            if img_data.shape[:2] != label_data.shape[:2]:
                raise DataPreparationError(
                    f"size mismatch: image {img_data.shape[:2]} vs "
                    f"label {label_data.shape[:2]} — {fname}")
        except DataPreparationError:
            raise
        except Exception as exc:
            print(f"  WARNING: size check failed for {fname}: {exc}")
            if strict_pairing:
                raise DataPreparationError(
                    f"size validation failed for {fname}") from exc
            continue

        base = f"{prefix}_{stem}" if prefix else stem
        dst_img = os.path.join(out_img_dir, base + Path(img_path).suffix.lower())
        dst_label = os.path.join(out_label_dir, base + "_labels.png")

        if not force and os.path.exists(dst_label):
            continue

        _copy_image(img_path, dst_img)
        os.makedirs(os.path.dirname(dst_label), exist_ok=True)
        shutil.copy2(label_path, long_path(dst_label))

        rows.append({
            "file_name": base + Path(img_path).suffix.lower(),
            "label_file": base + "_labels.png",
            "image_path": dst_img,
            "label_path": dst_label,
            "is_large_image": str(is_large_image(img_path)).lower(),
            "label_status": "unknown",
        })

    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} generic image(s) in {subset_path} "
              "have no matching *_labels.png")
        if len(missing_labels) <= 10:
            for name in missing_labels:
                print(f"    missing label for: {name}")
        if strict_pairing:
            raise DataPreparationError(
                f"strict pairing enabled; {len(missing_labels)} generic image(s) "
                f"in {subset_path} have no matching *_labels.png.")
    return rows


CONVERTERS = {
    "emps": convert_emps,
    "nist": convert_nist,
    "nnp": convert_nnp,
    "tio2": convert_tio2,
    "generic": convert_generic,
}


# ═══════════════════════════════════════════════════════════════
# Manifest
# ═══════════════════════════════════════════════════════════════

MANIFEST_HEADER = [
    "record_id", "image_path", "file_name", "split",
    "source", "microscope_type",
    "polarity", "particle_morphology", "shape_type", "density_level", "size_distribution",
    "quality_level", "adhesion_level",
    "label_status", "label_path", "flow_path", "notes",
    "is_large_image",
]


def _append_manifest(manifest_path: str, rows: list[dict]):
    """追加行到 CSV，文件不存在则写表头；按 record_id/file_name 去重。"""
    exists = os.path.exists(manifest_path)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    seen_keys = set()
    if exists:
        with open(manifest_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seen_keys.add((row.get("record_id", ""), row.get("file_name", "")))
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            # 补齐缺失字段
            for field in MANIFEST_HEADER:
                row.setdefault(field, "")
            # 自动时间戳
            if not row.get("source"):
                row["source"] = f"prepared_{datetime.date.today().isoformat()}"
            key = (row.get("record_id", ""), row.get("file_name", ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            writer.writerow(row)


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def process_subset(subset_name: str, src_root: str,
                   out_img_dir: str, out_label_dir: str,
                   manifest_path: str, force: bool,
                   strict_pairing: bool = True) -> tuple[int, int]:
    """处理一个子目录，返回 (成功图片数, 总图片数)。"""
    subset_path = Path(src_root) / subset_name
    if not subset_path.is_dir():
        print(f"  SKIP: {subset_name} (no such directory)")
        return 0, 0

    fmt = detect_format(subset_name)
    mtype = detect_microscope_type(subset_name)
    converter = CONVERTERS.get(fmt, convert_generic)

    prefix = subset_name  # 使用完整子集名避免不同子集（如 batch_sem/batch_tem）前缀冲突

    # 尝试转换格式已知的子目录
    try:
        rows = converter(subset_path, out_img_dir, out_label_dir, force, prefix=prefix,
                         strict_pairing=strict_pairing)

        # 格式未知：尝试通用格式（已有 *_labels.png）
        if not rows and fmt == "generic":
            rows = convert_generic(subset_path, out_img_dir, out_label_dir, force, prefix=prefix,
                                   strict_pairing=strict_pairing)
    except DataPreparationError as exc:
        print(f"  {subset_name}: format={fmt}, type={mtype}, ERROR: {exc}")
        raise

    # 写入 manifest
    if rows:
        total = len(rows)
        valid = sum(1 for r in rows if os.path.exists(r["image_path"]) and os.path.exists(r["label_path"]))
        for row in rows:
            row["microscope_type"] = mtype
            row["record_id"] = os.path.splitext(row["file_name"])[0]
            row["particle_morphology"] = "unknown"
            row["shape_type"] = "unknown"
            row["density_level"] = "unknown"
            row["size_distribution"] = "unknown"
            row["quality_level"] = "unknown"
            row["adhesion_level"] = "unknown"
            row.setdefault("label_status", "prelabelled")

        _append_manifest(manifest_path, rows)
        print(f"  {subset_name}: format={fmt}, type={mtype}, rows={len(rows)} (valid={valid})")
        return valid, total
    else:
        print(f"  {subset_name}: format={fmt}, type={mtype}, no valid rows found")
        if strict_pairing:
            raise DataPreparationError(
                f"{subset_name}: no valid rows generated in strict/default mode.")
        return 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified training data preparation tool")
    parser.add_argument("--src-root", required=True,
                        help="已标注数据集根目录")
    parser.add_argument("--subset", default="*",
                        help='子目录: "*" 扫描全部，空格分隔指定目录则增量追加')
    parser.add_argument("--out-img-dir", default="temp/staging/images",
                        help="输出图片目录")
    parser.add_argument("--out-label-dir", default="temp/staging/labels",
                        help="输出标签目录")
    parser.add_argument("--out-manifest", default="temp/staging/dataset_manifest_draft.csv",
                        help="输出 manifest CSV 路径")
    parser.add_argument("--force", action="store_true",
                        help="覆盖已存在的同名文件（默认跳过）")
    parser.add_argument("--no-strict-pairing", action="store_true",
                        help="允许图片缺少对应标签时仅 warning 跳过（默认：缺失标签时失败）")
    args = parser.parse_args()
    strict_pairing = not args.no_strict_pairing

    # --force --subset *: 全量重建，清空旧 manifest 避免残留过期行
    if args.force and args.subset == "*" and os.path.exists(args.out_manifest):
        os.remove(args.out_manifest)
        print(f"  (--force --subset *: removed old manifest {args.out_manifest})")

    src_root = Path(args.src_root)
    if not src_root.is_dir():
        print(f"ERROR: --src-root not found: {args.src_root}")
        sys.exit(1)

    # 解析 --subset
    if args.subset == "*":
        subsets = sorted([
            d.name for d in src_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
    else:
        subsets = [s.strip() for s in args.subset.split() if s.strip()]

    if not subsets:
        print("ERROR: no subsets to process.")
        sys.exit(1)

    print(f"Processing {len(subsets)} subsets from {src_root}...")
    print(f"  Output: img={args.out_img_dir}, label={args.out_label_dir}")
    print(f"  Manifest: {args.out_manifest}")
    print(f"  Force: {args.force}")
    print()

    total_valid = 0
    total_all = 0
    try:
        for name in tqdm(subsets, desc="Subsets"):
            valid, all_ = process_subset(
                name, args.src_root,
                args.out_img_dir, args.out_label_dir,
                args.out_manifest, args.force,
                strict_pairing=strict_pairing,
            )
            total_valid += valid
            total_all += all_
    except DataPreparationError as exc:
        print(f"\nERROR: data preparation failed: {exc}")
        sys.exit(1)

    print(f"\nDone: {total_valid} valid images from {total_all} total in {len(subsets)} subsets")
    print(f"  Manifest: {args.out_manifest} (edit scene fields before training)")
    if strict_pairing and total_valid == 0:
        print("ERROR: strict/default data preparation produced zero valid images.")
        sys.exit(1)


if __name__ == "__main__":
    main()
