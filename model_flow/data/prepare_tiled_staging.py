"""生成可直接划分 train/val 的 V1 staging 数据。

非大图样本原样复制；训练/验证大图样本切成 1024 tile，并为每个 tile
生成 manifest 行；holdout 样本保留完整原图用于端到端 tile 推理验收。
输出 manifest 可直接传给 split_dataset_by_manifest。
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from ..manifest.dataset_manifest_utils import read_manifest, write_manifest
from ..utils import imread_unicode, imread_unchanged, imwrite_unicode, long_path
from .create_tiled_training_data import (
    LONG_SIDE_LIMIT,
    OVERLAP,
    TILE_SIZE,
    should_tile,
    tile_positions,
)

# 用于描述 tile 核心区域（唯一非重叠部分），向下游 tile 合并提供元数据。
# 与截断实例剔除逻辑（boundary distance 0）无关 — 截断检测必须使用裁剪边界本身，
# 不能用 core_margin 替代，否则会漏掉真正被切断的实例。
TILE_CORE_MARGIN = 128


def copy_non_large_sample(
    row: dict,
    image_path: Path,
    label_path: Path,
    out_image_dir: Path,
    out_label_dir: Path,
) -> dict:
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    # ── 硬校验：image 和 label 空间尺寸必须一致 ──
    try:
        from model_flow.utils import imread_unicode, imread_unchanged
        img = imread_unicode(str(image_path))
        if img is None:
            raise RuntimeError(f"cannot read image: {image_path}")
        lab = imread_unchanged(str(label_path))
        if lab is None:
            raise RuntimeError(f"cannot read label: {label_path}")
        if img.shape[:2] != lab.shape[:2]:
            raise RuntimeError(
                f"size mismatch: image {img.shape[:2]} vs label {lab.shape[:2]} "
                f"for {image_path.name}")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"failed to size-validate {image_path.name}: {exc}") from exc

    output_image_path = out_image_dir / image_path.name
    output_label_path = out_label_dir / label_path.name
    shutil.copy2(image_path, output_image_path)
    shutil.copy2(label_path, output_label_path)

    source_is_large_image = row.get("is_large_image", "") or "false"
    output_row = dict(row)
    output_row.update({
        "image_path": str(output_image_path).replace("\\", "/"),
        "file_name": output_image_path.name,
        "label_path": str(output_label_path).replace("\\", "/"),
        "is_large_image": source_is_large_image,
        "is_tile": "false",
        "source_image_path": row.get("image_path", "") or image_path.name,
        "tile_x": "",
        "tile_y": "",
        "tile_width": "",
        "tile_height": "",
        "tile_overlap": "",
        "tile_core_margin": "",
        "tile_role": "",
    })
    return output_row


def make_tile_rows(
    row: dict,
    image_path: Path,
    label_path: Path,
    out_image_dir: Path,
    out_label_dir: Path,
    background_tile_ratio: float = 0.0,
    rng: random.Random | None = None,
) -> list[dict]:
    image = imread_unicode(str(image_path))
    if image is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    labels = imread_unchanged(str(label_path))
    if labels is None:
        raise RuntimeError(f"无法读取标签: {label_path}")

    height, width = image.shape[:2]
    if labels.shape[:2] != (height, width):
        raise RuntimeError(
            f"图片与标签尺寸不一致: image={image.shape[:2]}, labels={labels.shape[:2]} ({image_path})")
    source_is_large_image = (
        "true"
        if row.get("is_large_image") == "true" or max(height, width) > LONG_SIDE_LIMIT
        else "false"
    )
    tile_rows: list[dict] = []
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    for tile_y, tile_x, tile_height, tile_width in tile_positions(height, width):
        tile_labels = labels[tile_y:tile_y + tile_height, tile_x:tile_x + tile_width].copy()
        removed_truncated = 0

        # P0-10: 剔除接触 tile 裁剪边界的截断实例 (保守策略: 接触任一边即剔除)
        if tile_labels.max() > 0:
            instance_ids = [v for v in np.unique(tile_labels) if v > 0]
            for inst_id in instance_ids:
                inst_mask = (tile_labels == inst_id)
                ys, xs = np.nonzero(inst_mask)
                if len(ys) == 0:
                    continue
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                touches_crop_edge = (
                    y0 <= 0 or y1 >= tile_height - 1
                    or x0 <= 0 or x1 >= tile_width - 1
                )
                if touches_crop_edge:
                    tile_labels[inst_mask] = 0
                    removed_truncated += 1

        if tile_labels.max() == 0:
            # P0-11: 按比例保留空背景 tile 作为硬负样本
            if background_tile_ratio > 0 and rng is not None:
                if rng.random() >= background_tile_ratio:
                    continue
                # 保留此空 tile，标记为 background_negative
                is_background_tile = True
            else:
                continue
        else:
            is_background_tile = False

        tile_image = image[tile_y:tile_y + tile_height, tile_x:tile_x + tile_width]
        pad_height = TILE_SIZE - tile_height
        pad_width = TILE_SIZE - tile_width
        if pad_height > 0 or pad_width > 0:
            # Keep padding identical to C++ tiled inference. The final
            # row/column tile is aligned to the bottom/right boundary; padding
            # only happens when the image is shorter than TILE_SIZE on an axis.
            pad_top = 0
            pad_bottom = pad_height
            pad_left = 0
            pad_right = pad_width
            tile_image = cv2.copyMakeBorder(
                tile_image, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=114)
            tile_labels = cv2.copyMakeBorder(
                tile_labels.astype(np.uint16), pad_top, pad_bottom,
                pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        tile_stem = f"{image_path.stem}_tile_{tile_y}_{tile_x}"
        tile_image_path = out_image_dir / f"{tile_stem}{image_path.suffix}"
        tile_label_path = out_label_dir / f"{tile_stem}_labels.png"
        imwrite_unicode(long_path(str(tile_image_path)), tile_image)
        imwrite_unicode(
            long_path(str(tile_label_path)),
            tile_labels.astype(np.uint16),
            [cv2.IMWRITE_PNG_COMPRESSION, 3],
        )

        output_row = dict(row)
        output_row.update({
            "record_id": f"{row.get('record_id') or image_path.stem}__tile_{tile_y}_{tile_x}",
            "image_path": str(tile_image_path).replace("\\", "/"),
            "file_name": tile_image_path.name,
            "split": "unassigned",
            "label_path": str(tile_label_path).replace("\\", "/"),
            "flow_path": "",
            "is_large_image": source_is_large_image,
            "is_tile": "true",
            "source_image_path": row.get("image_path", "") or image_path.name,
            "tile_x": str(tile_x),
            "tile_y": str(tile_y),
            "tile_width": str(tile_width),
            "tile_height": str(tile_height),
            "tile_overlap": str(OVERLAP),
            "tile_core_margin": str(TILE_CORE_MARGIN),
            "tile_role": "background_negative" if is_background_tile else "train_tile",
            "notes": (
                f"removed_truncated_instances={removed_truncated}"
                if removed_truncated > 0 else ""
            ),
        })
        tile_rows.append(output_row)

    return tile_rows


def resolve_input_path(directory: Path, file_name: str, explicit_path: str) -> Path:
    path = Path(explicit_path) if explicit_path else Path(file_name)
    if path.is_absolute() and path.exists():
        return path
    return directory / file_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 V1 可划分 staging：非大图复制，大图切 tile 并同步 manifest。")
    parser.add_argument("--manifest", required=True,
                        help="输入 draft manifest，例如 temp/staging/dataset_manifest_draft.csv")
    parser.add_argument("--img-dir", required=True, help="输入图片目录")
    parser.add_argument("--label-dir", required=True, help="输入标签目录")
    parser.add_argument("--out-img-dir", required=True, help="输出图片/tile 目录")
    parser.add_argument("--out-label-dir", required=True, help="输出标签/tile label 目录")
    parser.add_argument("--out-manifest", required=True, help="输出 tile 后 manifest")
    parser.add_argument("--background-tile-ratio", type=float, default=0.05,
                        help="空背景 tile 保留比例 (默认 0.05), 0 表示全部跳过")
    args = parser.parse_args()

    image_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    out_image_dir = Path(args.out_img_dir)
    out_label_dir = Path(args.out_label_dir)
    rows = read_manifest(args.manifest)

    rng = random.Random(42)
    output_rows: list[dict] = []
    large_count = 0
    tile_count = 0
    bg_tile_count = 0
    copied_count = 0

    for row in rows:
        file_name = row.get("file_name", "")
        if not file_name:
            continue
        image_path = resolve_input_path(image_dir, file_name, row.get("image_path", ""))
        label_path = label_dir / f"{Path(file_name).stem}_labels.png"
        if not image_path.exists():
            raise FileNotFoundError(f"找不到图片: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"找不到标签: {label_path}")

        actual_is_large = should_tile(str(image_path))
        is_holdout = (row.get("split") == "holdout"
                      or row.get("tile_role") == "holdout_full_image")
        if is_holdout:
            output_row = copy_non_large_sample(
                row, image_path, label_path, out_image_dir, out_label_dir)
            output_row.update({
                "split": "holdout",
                "is_large_image": str(actual_is_large).lower(),
                "tile_role": "holdout_full_image",
            })
            output_rows.append(output_row)
            copied_count += 1
            continue

        is_large = row.get("is_large_image") == "true" or actual_is_large
        if is_large:
            large_count += 1
            tile_rows = make_tile_rows(
                row, image_path, label_path, out_image_dir, out_label_dir,
                background_tile_ratio=args.background_tile_ratio, rng=rng)
            output_rows.extend(tile_rows)
            tile_count += len(tile_rows)
            bg_tile_count += sum(
                1 for r in tile_rows if r.get("tile_role") == "background_negative")
        else:
            output_rows.append(
                copy_non_large_sample(row, image_path, label_path, out_image_dir, out_label_dir))
            copied_count += 1

    write_manifest(output_rows, args.out_manifest)
    print(
        f"Done: copied={copied_count}, large_images={large_count}, "
        f"tiles={tile_count}, background_tiles={bg_tile_count}, "
        f"long_side_limit={LONG_SIDE_LIMIT}, "
        f"background_tile_ratio={args.background_tile_ratio}, "
        f"manifest={args.out_manifest}")


if __name__ == "__main__":
    main()
