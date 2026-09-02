"""Validate dataset_manifest.csv/jsonl before formal training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset_manifest_utils import (
    read_manifest, validate_manifest, parse_split_list,
    V1_REQUIRED_SCENE_FIELDS,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='检查 dataset_manifest 字段、枚举、路径和 Flow 配对。')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--base-dir', default='.')
    parser.add_argument('--require-flow-for-splits', default='',
                        help='逗号分隔；哪些 split 必须已有 flow_path。数据准备早期默认不强制。')
    parser.add_argument('--require-reviewed-for-splits', default='',
                        help='逗号分隔；哪些 split 必须 label_status==reviewed。'
                             'V1 正式训练使用 train,val。')
    parser.add_argument('--strict-scene-fields', action='store_true',
                        help='启用后, V1 必填场景字段 (particle_morphology/'
                             'density_level/size_distribution/quality_level/adhesion_level/'
                             'is_large_image) 中任何 unknown 或空值均报 error。'
                             'microscope_type 仅用于成像域记录和分层分析，不作为硬 gate。')
    parser.add_argument('--out', default='temp/dataset_manifest_check.json')
    args = parser.parse_args()

    try:
        rows, extra_fields_rows = read_manifest(
            args.manifest, collect_extra_fields=True)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    strict_fields = V1_REQUIRED_SCENE_FIELDS if args.strict_scene_fields else []
    errors, warnings, summary = validate_manifest(
        rows,
        base_dir=args.base_dir,
        require_flow_for_splits=parse_split_list(args.require_flow_for_splits),
        strict_unknown_for=strict_fields,
        require_reviewed_for_splits=parse_split_list(args.require_reviewed_for_splits),
        extra_fields_rows=extra_fields_rows,
    )

    report = {
        'manifest': args.manifest,
        'summary': summary,
        'errors': errors,
        'warnings': warnings,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("数据集清单检查 Dataset manifest check")
    print(f"  行数 rows: {summary['total_images']}")
    print(f"  错误 errors: {len(errors)}")
    print(f"  警告 warnings: {len(warnings)}")
    print(f"  报告 report: {out_path}")

    if errors:
        print("\n错误 Errors:")
        for error in errors[:50]:
            print(f"  - {error}")
        if len(errors) > 50:
            print(f"  ... {len(errors) - 50} more")
        raise SystemExit(1)

    if warnings:
        print("\n警告 Warnings:")
        for warning in warnings[:30]:
            print(f"  - {warning}")
        if len(warnings) > 30:
            print(f"  ... {len(warnings) - 30} more")

    print("\n数据集清单检查通过 Manifest check OK.")


if __name__ == '__main__':
    main()
