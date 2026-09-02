"""Create an initial dataset manifest from image directories."""
from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_manifest_utils import (
    MANIFEST_FIELDS,
    collect_image_files,
    infer_microscope_type,
    make_relative,
    read_manifest,
    write_manifest,
)

# 从 base manifest 复制的字段（场景字段 + tile 诊断字段）
# label_status 从 base manifest 复制，尊重步骤 4 的人工复核结果。
# 未在 base manifest 中的行使用 --label-status CLI 默认值。
# 步骤 8b gate (--require-reviewed-for-splits train,val) 会拦截所有非 reviewed 行。
_COPY_FIELDS = [
    'microscope_type', 'polarity', 'particle_morphology', 'shape_type',
    'density_level', 'size_distribution', 'quality_level',
    'adhesion_level', 'label_status', 'notes',
    # tile 诊断字段 — 正式 manifest 必须保留 tile 来源和坐标信息
    'is_large_image', 'is_tile', 'source_image_path',
    'tile_x', 'tile_y', 'tile_width', 'tile_height',
    'tile_overlap', 'tile_core_margin', 'tile_role',
]


def find_matching_file(directory: str, stem: str, suffix: str) -> str:
    if not directory:
        return ''
    candidate = Path(directory) / f'{stem}{suffix}'
    if candidate.is_file():
        return str(candidate)
    return ''


def build_base_lookup(base_manifest_path: str) -> dict[str, dict]:
    """从 base manifest 构建 {file_name: row} 查找表。

    用于将用户手动补齐的场景字段复制到新生成的正式 manifest。
    """
    if not base_manifest_path:
        return {}
    rows = read_manifest(base_manifest_path)
    return {row.get('file_name', '').strip(): row for row in rows
            if row.get('file_name', '').strip()}


def add_rows_for_split(
    rows: list[dict],
    split: str,
    image_dir: str,
    flow_dir: str,
    label_dir: str,
    base_dir: Path,
    source: str,
    label_status: str,
    base_lookup: dict[str, dict] | None = None,
) -> None:
    if not image_dir:
        return
    for image_path in collect_image_files(image_dir):
        flow_path = find_matching_file(flow_dir, image_path.stem, '.npy')
        label_path = find_matching_file(label_dir, image_path.stem, '_labels.png')
        row = {field: '' for field in MANIFEST_FIELDS}
        row.update({
            'record_id': image_path.stem,
            'image_path': make_relative(image_path, base_dir),
            'file_name': image_path.name,
            'split': split,
            'source': source,
            'microscope_type': infer_microscope_type(image_path.name),
            'polarity': 'unknown',
            'particle_morphology': 'unknown',
            'shape_type': 'unknown',
            'density_level': 'unknown',
            'size_distribution': 'unknown',
            'quality_level': 'unknown',
            'adhesion_level': 'unknown',
            'label_status': label_status,
            'label_path': make_relative(Path(label_path), base_dir) if label_path else '',
            'flow_path': make_relative(Path(flow_path), base_dir) if flow_path else '',
        })

        # 从 base manifest 复制用户已补齐的场景字段
        if base_lookup and image_path.name in base_lookup:
            base_row = base_lookup[image_path.name]
            for field in _COPY_FIELDS:
                val = str(base_row.get(field, '') or '').strip()
                if val and val != 'unknown':
                    row[field] = val
        rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Initialize dataset_manifest.csv/jsonl from image directories.')
    parser.add_argument('--out', required=True, help='Output manifest path.')
    parser.add_argument('--base-dir', default='.',
                        help='Base directory used for relative paths.')
    parser.add_argument('--train-img-dir', default='')
    parser.add_argument('--val-img-dir', default='')
    parser.add_argument('--test-img-dir', default='')
    parser.add_argument('--holdout-img-dir', default='')
    parser.add_argument('--train-flow-dir', default='')
    parser.add_argument('--val-flow-dir', default='')
    parser.add_argument('--test-flow-dir', default='')
    parser.add_argument('--holdout-flow-dir', default='')
    parser.add_argument('--label-dir', default='',
                        help='Optional shared directory containing *_labels.png files.')
    parser.add_argument('--train-label-dir', default='',
                        help='Optional train label directory; defaults to --label-dir, then train image dir.')
    parser.add_argument('--val-label-dir', default='',
                        help='Optional val label directory; defaults to --label-dir, then val image dir.')
    parser.add_argument('--test-label-dir', default='',
                        help='Optional test label directory; defaults to --label-dir, then test image dir.')
    parser.add_argument('--holdout-label-dir', default='',
                        help='Optional holdout label directory; defaults to --label-dir, then holdout image dir.')
    parser.add_argument('--source', default='current_project')
    parser.add_argument('--label-status', default='unknown',
                        choices=['unlabelled', 'prelabelled', 'reviewed', 'unknown'])
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite output manifest if it already exists.')
    parser.add_argument('--base-manifest', default='',
                        help='Base manifest (draft) whose manually-edited scene fields '
                             'will be copied to matching file_name rows.')
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists, pass --overwrite: {out_path}")

    base_dir = Path(args.base_dir)
    base_lookup = build_base_lookup(args.base_manifest) if args.base_manifest else {}
    rows: list[dict] = []
    add_rows_for_split(rows, 'train', args.train_img_dir, args.train_flow_dir,
                       args.train_label_dir or args.label_dir or args.train_img_dir,
                       base_dir, args.source, args.label_status,
                       base_lookup=base_lookup)
    add_rows_for_split(rows, 'val', args.val_img_dir, args.val_flow_dir,
                       args.val_label_dir or args.label_dir or args.val_img_dir,
                       base_dir, args.source, args.label_status,
                       base_lookup=base_lookup)
    add_rows_for_split(rows, 'test', args.test_img_dir, args.test_flow_dir,
                       args.test_label_dir or args.label_dir or args.test_img_dir,
                       base_dir, args.source, args.label_status,
                       base_lookup=base_lookup)
    add_rows_for_split(rows, 'holdout', args.holdout_img_dir,
                       args.holdout_flow_dir,
                       args.holdout_label_dir or args.label_dir or args.holdout_img_dir,
                       base_dir,
                       args.source, args.label_status,
                       base_lookup=base_lookup)

    if not rows:
        raise SystemExit("No images found. Provide at least one image directory.")

    write_manifest(rows, out_path)
    print("Dataset manifest initialized.")
    print(f"  rows: {len(rows)}")
    print(f"  output: {out_path}")
    print("Next: edit scene fields and label_status before formal training.")


if __name__ == '__main__':
    main()
