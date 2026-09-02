"""Generate a formal-training readiness report from dataset_manifest."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .dataset_manifest_utils import (
    FIELD_DESCRIPTIONS_ZH,
    V1_MINIMUM_REQUIREMENTS,
    V1_REQUIRED_SCENE_FIELDS,
    check_v1_minimum_requirements,
    field_label,
    parse_split_list,
    read_manifest,
    validate_manifest,
)


def counter_for(rows: list[dict], field: str) -> Counter:
    return Counter(row.get(field, '') or 'unknown' for row in rows)


def markdown_table(counter: Counter) -> str:
    lines = ['| 取值 Value | 数量 Count |', '| --- | ---: |']
    for key, count in sorted(counter.items(), key=lambda item: str(item[0])):
        lines.append(f'| {key} | {count} |')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='生成正式训练前的数据集就绪报告。')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--base-dir', default='.')
    parser.add_argument('--out', default='data/particles/dataset_readiness_report.md')
    parser.add_argument('--require-flow-for-splits', default='train,val')
    parser.add_argument('--require-reviewed-for-splits', default='train,val')
    args = parser.parse_args()

    try:
        rows, extra_fields_rows = read_manifest(
            args.manifest, collect_extra_fields=True)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    errors, warnings, summary = validate_manifest(
        rows,
        base_dir=args.base_dir,
        require_flow_for_splits=parse_split_list(args.require_flow_for_splits),
        strict_unknown_for=V1_REQUIRED_SCENE_FIELDS,
        require_reviewed_for_splits=parse_split_list(args.require_reviewed_for_splits),
        extra_fields_rows=extra_fields_rows,
    )

    # V1 最低数量和覆盖要求
    v1_errors, v1_warnings = check_v1_minimum_requirements(rows)
    errors.extend(v1_errors)
    warnings.extend(v1_warnings)

    lines = [
        '# 数据集就绪报告 Dataset Readiness Report',
        '',
        f'- 数据集清单 Manifest: `{args.manifest}`',
        f'- 图片总数 Total images: {len(rows)}',
        f'- Flow 路径数 Flow paths: {summary["flow_path_count"]}',
        f'- 标签路径数 Label paths: {summary["label_path_count"]}',
        f'- 阻断问题 Blockers: {len(errors)}',
        f'- 警告 Warnings: {len(warnings)}',
        '',
        '## 数据划分覆盖 Split Coverage',
        '',
        markdown_table(counter_for(rows, 'split')),
        '',
        '## 字段说明 Field Notes',
        '',
        '- size_distribution 只用于训练数据管理、分层划分和诊断；不作为最终轮廓验收指标。',
        '- microscope_type 只记录显微镜来源或成像域，用于分层划分、分层分析和诊断；不作为模型推理或 C++ 后处理正确性的硬性门禁。',
        '- dataset_manifest 不记录 has_scale/pixel_size/pixel_size_unit/scale_source；比例尺不属于训练 manifest。',
    ]

    for field, title in [
        ('microscope_type', '显微镜类型 Microscope Type'),
        ('particle_morphology', '颗粒形态 Particle Morphology'),
        ('polarity', '明暗极性 Polarity'),
        ('shape_type', '形态类型 Shape Type'),
        ('density_level', '密度等级 Density Level'),
        ('size_distribution', '粒径分布 Size Distribution'),
        ('quality_level', '图像质量 Quality Level'),
        ('adhesion_level', '粘连程度 Adhesion Level'),
        ('label_status', '标注状态 Label Status'),
    ]:
        description = FIELD_DESCRIPTIONS_ZH.get(field, '')
        if description:
            lines.extend(['', f'## {title}', '', description, '', markdown_table(counter_for(rows, field))])
        else:
            lines.extend(['', f'## {title}', '', markdown_table(counter_for(rows, field))])

    if errors:
        lines.extend(['', '## 阻断问题 Blockers', ''])
        lines.extend(f'- {error}' for error in errors)
    if warnings:
        lines.extend(['', '## 警告 Warnings', ''])
        lines.extend(f'- {warning}' for warning in warnings)

    # V1 最低数量要求表
    train_n = sum(1 for r in rows if r.get("split") == "train")
    val_n = sum(1 for r in rows if r.get("split") == "val")
    holdout_n = sum(1 for r in rows if r.get("split") == "holdout")
    large_sources = len({r.get("source_image_path") or r.get("image_path") or ""
                         for r in rows if r.get("is_large_image") == "true"})
    holdout_large = len({r.get("source_image_path") or r.get("image_path") or ""
                          for r in rows if r.get("is_large_image") == "true"
                          and r.get("split") == "holdout"})

    lines.extend([
        '',
        '## V1 最低数量要求 V1 Minimum Requirements',
        '',
        '| 要求 Requirement | 最低 Minimum | 实际 Actual | 状态 Status |',
        '| --- | ---: | ---: | --- |',
        f'| train 样本数 | {V1_MINIMUM_REQUIREMENTS["train_min_samples"]} | {train_n} | '
        f'{"PASS" if train_n >= V1_MINIMUM_REQUIREMENTS["train_min_samples"] else "FAIL"} |',
        f'| val 样本数 | {V1_MINIMUM_REQUIREMENTS["val_min_samples"]} | {val_n} | '
        f'{"PASS" if val_n >= V1_MINIMUM_REQUIREMENTS["val_min_samples"] else "FAIL"} |',
        f'| holdout 样本数 | {V1_MINIMUM_REQUIREMENTS["holdout_min_samples"]} | {holdout_n} | '
        f'{"PASS" if holdout_n >= V1_MINIMUM_REQUIREMENTS["holdout_min_samples"] else "FAIL"} |',
        f'| 大图源图数 | {V1_MINIMUM_REQUIREMENTS["large_image_source_min"]} | {large_sources} | '
        f'{"PASS" if large_sources >= V1_MINIMUM_REQUIREMENTS["large_image_source_min"] else "FAIL"} |',
        f'| holdout 大图数 | {V1_MINIMUM_REQUIREMENTS["holdout_large_image_min"]} | {holdout_large} | '
        f'{"PASS" if holdout_large >= V1_MINIMUM_REQUIREMENTS["holdout_large_image_min"] else "FAIL"} |',
        '',
    ])

    lines.extend([
        '',
        '## 正式训练门禁 Formal Training Gate',
        '',
        '- 只有阻断问题为 0 时才允许进入正式训练。',
        '- V1 最低数量要求必须全部 PASS。',
        '- 正式材料/粉体颗粒图像训练前，颗粒形态、size_distribution、密度、粘连程度、图像质量和 is_large_image 不得为空或 unknown。',
        '- 显微镜类型或成像域应尽量记录，用于分层分析；无法可靠确认时可为 unknown，不阻断训练或 C++ 推理正确性验收。',
        '- train/val 样本必须为 label_status=reviewed。',
        '- 最终验收只评价实例轮廓与 GT 轮廓重合度，不评价粒径、面积或粒度分布。',
        '',
    ])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines), encoding='utf-8')

    print("数据集就绪报告已生成 Dataset readiness report written.")
    print(f"  行数 rows: {len(rows)}")
    print(f"  阻断 blockers: {len(errors)}")
    print(f"  警告 warnings: {len(warnings)}")
    print(f"  输出 output: {out_path}")
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
