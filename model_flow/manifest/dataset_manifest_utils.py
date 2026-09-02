"""Shared helpers for dataset manifest tools.

The manifest is not model input. It is the index used before formal training to
track scene coverage, reviewed labels, flow pairing, and split assignments.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Union, Tuple


V1_REQUIRED_SCENE_FIELDS = [
    'particle_morphology',
    'density_level',
    'size_distribution',
    'quality_level',
    'adhesion_level',
    'is_large_image',
]

# 文档明确禁止出现在训练 manifest 中的比例尺相关字段
FORBIDDEN_MANIFEST_FIELDS = {
    'has_scale', 'pixel_size', 'pixel_size_unit', 'scale_source',
}

FORBIDDEN_FIELD_LABELS_ZH = {f: f for f in FORBIDDEN_MANIFEST_FIELDS}
FORBIDDEN_FIELD_LABELS_ZH.update({
    'has_scale': '是否有比例尺',
    'pixel_size': '像素尺寸',
    'pixel_size_unit': '像素尺寸单位',
    'scale_source': '比例尺来源',
})


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

MANIFEST_FIELDS = [
    'record_id',
    'image_path',
    'file_name',
    'split',
    'source',
    'microscope_type',
    'polarity',
    'particle_morphology',
    'shape_type',
    'density_level',
    'size_distribution',
    'quality_level',
    'adhesion_level',
    'label_status',
    'label_path',
    'flow_path',
    'notes',
    'is_large_image',
    # tile 相关
    'is_tile',
    'source_image_path',
    'tile_x',
    'tile_y',
    'tile_width',
    'tile_height',
    'tile_overlap',
    'tile_core_margin',
    'tile_role',
]

FIELD_LABELS_ZH = {
    'record_id': '记录编号',
    'image_path': '图片路径',
    'file_name': '文件名',
    'split': '数据划分',
    'source': '数据来源',
    'microscope_type': '显微镜类型',
    'polarity': '明暗极性',
    'particle_morphology': '颗粒形态',
    'shape_type': '形态类型（兼容旧字段）',
    'density_level': '密度等级',
    'size_distribution': '粒径分布',
    'quality_level': '图像质量',
    'adhesion_level': '粘连程度',
    'label_status': '标注状态',
    'label_path': '标签路径',
    'flow_path': 'Flow 路径',
    'notes': '备注',
    'is_large_image': '是否大图',
    'is_tile': '是否切片',
    'source_image_path': '源图片路径',
    'tile_x': '切片左上角 X',
    'tile_y': '切片左上角 Y',
    'tile_width': '切片宽度',
    'tile_height': '切片高度',
    'tile_overlap': '切片重叠',
    'tile_core_margin': '切片核心边距',
    'tile_role': '切片角色',
}

FIELD_DESCRIPTIONS_ZH = {
    'microscope_type': (
        '材料/粉体图片的显微镜来源或成像域记录，用于分层划分、分层分析和诊断；'
        '不作为模型推理或 C++ 后处理正确性的硬性门禁。无法可靠确认时可为 unknown。'
    ),
    'particle_morphology': '整张图中颗粒主导形态，用于分层划分和分层验收。',
    'size_distribution': '按模型输入尺度下颗粒等效圆直径的分布标注，只用于训练数据管理。',
    'density_level': '颗粒之间的空间密集程度，与颗粒大小无关。',
    'adhesion_level': '颗粒接触、粘连或重叠的程度，用于验证粘连拆分能力。',
    'quality_level': '图像对比度、模糊、噪声纹理等质量分层。',
}

ENUMS = {
    'split': {'train', 'val', 'test', 'holdout', 'unassigned'},
    'microscope_type': {'SEM', 'TEM', 'TSEM', 'optical', 'CryoEM', 'unknown'},
    'polarity': {
        'bright_background_dark_particles',
        'dark_background_bright_particles',
        'mixed',
        'unknown',
    },
    'particle_morphology': {
        'round_like', 'irregular', 'flake_like', 'fiber_like', 'porous',
        'mixed', 'unknown',
    },
    'shape_type': {
        'round_like', 'irregular', 'flake_like', 'fiber_like', 'porous',
        'mixed', 'unknown',
    },
    'density_level': {'sparse', 'medium', 'dense', 'unknown'},
    'size_distribution': {'small', 'medium', 'large', 'wide', 'unknown'},
    'quality_level': {
        'clear',
        'low_contrast',
        'blurred',
        'noisy_texture',
        'unknown',
    },
    'adhesion_level': {'none', 'few_touching', 'many_touching', 'unknown'},
    'label_status': {'unlabelled', 'prelabelled', 'reviewed', 'unknown'},
    'is_large_image': {'true', 'false'},
    'is_tile': {'true', 'false'},
    'tile_role': {
        '',
        'train_tile',
        'validation_tile',
        'background_negative',
        'holdout_full_image',
    },
}


def parse_split_list(value: str, sep: str = ',') -> list[str]:
    """Parse a comma-separated list of split names, trimming whitespace."""
    return [item.strip() for item in value.split(sep) if item.strip()]


def is_jsonl(path: Path) -> bool:
    return path.suffix.lower() in {'.jsonl', '.ndjson'}


def read_manifest(path: str | Path,
                  collect_extra_fields: bool = False
                  ) -> Union[list[dict], Tuple[list[dict], list[dict]]]:
    """读取 manifest 并返回标准化后的行列表。

    当 collect_extra_fields=True 时返回 (rows, extra_fields_warnings) 元组，
    其中 extra_fields_warnings 是 [{row_index, field}] 列表。
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"找不到数据集清单 dataset_manifest: {manifest_path}。"
            "V1 不会默认使用旧 data/particles 数据，请先按方案准备并生成新的 manifest。")

    extra_fields_warnings: list[dict] = []

    if is_jsonl(manifest_path):
        rows = []
        with manifest_path.open('r', encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSONL at line {line_number}: {exc}") from exc
                extra_collector: list[str] = []
                normalized = normalize_row(raw_row, extra_collector)
                for field_name in extra_collector:
                    extra_fields_warnings.append(
                        {'row': line_number, 'field': field_name})
                rows.append(normalized)
        return (rows, extra_fields_warnings) if collect_extra_fields else rows

    with manifest_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        all_extra: list[dict] = []
        rows_out: list[dict] = []
        for row_index, raw_row in enumerate(reader, start=2):  # row 1 = header
            extra_collector: list[str] = []
            normalized = normalize_row(raw_row, extra_collector)
            for field_name in extra_collector:
                all_extra.append({'row': row_index, 'field': field_name})
            rows_out.append(normalized)
        return (rows_out, all_extra) if collect_extra_fields else rows_out


def write_manifest(rows: list[dict], path: str | Path) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [normalize_row(row) for row in rows]

    if is_jsonl(manifest_path):
        with manifest_path.open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        return

    with manifest_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS,
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def normalize_row(row: dict,
                  extra_fields_collector: Union[list, None] = None) -> dict:
    normalized = {field: str(row.get(field, '') or '').strip()
                  for field in MANIFEST_FIELDS}
    normalized['is_large_image'] = normalize_bool(normalized['is_large_image'])
    normalized['is_tile'] = normalize_bool(normalized['is_tile'])
    if not normalized['particle_morphology'] and normalized['shape_type']:
        normalized['particle_morphology'] = normalized['shape_type']
    if not normalized['shape_type'] and normalized['particle_morphology']:
        normalized['shape_type'] = normalized['particle_morphology']

    if extra_fields_collector is not None:
        row_keys_lower = {k.lower() for k in row.keys() if k.strip()}
        for key in row_keys_lower:
            if key not in {f.lower() for f in MANIFEST_FIELDS}:
                extra_fields_collector.append(key)

    return normalized


def field_label(field: str) -> str:
    label = FIELD_LABELS_ZH.get(field, field)
    return f"{label} ({field})"


def normalize_bool(value: str) -> str:
    text = str(value or '').strip().lower()
    if text in {'1', 'true', 'yes', 'y'}:
        return 'true'
    if text in {'0', 'false', 'no', 'n'}:
        return 'false'
    # 空值不默认转为 false; 保留空字符串让下游 enum 校验捕获并报错。
    # is_large_image / is_tile 应由脚本自动填写, 不应在正式 manifest 中为空。
    if text == '':
        return ''
    return text


def collect_image_files(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return [
        path for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        and not path.stem.endswith("_labels")
    ]


def infer_microscope_type(file_name: str) -> str:
    name = file_name.lower()
    name = re.sub(r'_tile_\d+_\d+(?=\.[^.]+$|$)', '', name)
    if name.startswith('tio2_tsem_'):
        return 'TSEM'
    if 'emps' in name or name.startswith('nist_') or name.startswith('tio2_sem_'):
        return 'SEM'
    if 'cryo' in name:
        return 'CryoEM'
    if 'nnp' in name or 'tem' in name:
        return 'TEM'
    return 'unknown'


def make_relative(path: Path, base_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(base_dir.resolve())).replace('\\', '/')
    except ValueError:
        return str(path)


def resolve_manifest_path(path_text: str, base_dir: str | Path = '.') -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def validate_manifest(
    rows: list[dict],
    base_dir: str | Path = '.',
    require_flow_for_splits: Iterable[str] = ('train', 'val'),
    strict_unknown_for: Iterable[str] = (),
    require_reviewed_for_splits: Iterable[str] = (),
    extra_fields_rows: Iterable[dict] = (),
) -> tuple[list[str], list[str], dict]:
    errors: list[str] = []
    warnings: list[str] = []
    required_flow = set(require_flow_for_splits)

    record_ids = Counter(row.get('record_id', '') for row in rows)
    image_paths = Counter(row.get('image_path', '') for row in rows)

    for value, count in record_ids.items():
        if value and count > 1:
            errors.append(f"重复的 {field_label('record_id')}: {value}")
    for value, count in image_paths.items():
        if value and count > 1:
            errors.append(f"重复的 {field_label('image_path')}: {value}")

    for index, row in enumerate(rows, start=2):
        prefix = f"row {index} ({row.get('record_id') or row.get('file_name')})"
        if not row.get('record_id', ''):
            errors.append(f"{prefix}: 缺少 {field_label('record_id')}")
        if not row.get('image_path', ''):
            errors.append(f"{prefix}: 缺少 {field_label('image_path')}")
        if not row.get('file_name', ''):
            errors.append(f"{prefix}: 缺少 {field_label('file_name')}")

        for field, allowed in ENUMS.items():
            value = row.get(field, '')
            if value not in allowed:
                errors.append(
                    f"{prefix}: {field_label(field)} 必须是 {sorted(allowed)} 之一，当前为 '{value}'")

        image_path_value = row.get('image_path', '')
        image_path = resolve_manifest_path(image_path_value, base_dir)
        if image_path_value and not image_path.is_file():
            errors.append(f"{prefix}: 找不到 {field_label('image_path')}: {image_path_value}")

        split = row.get('split', '')
        if split in required_flow and not row.get('flow_path'):
            errors.append(f"{prefix}: split='{split}' 时缺少 {field_label('flow_path')}")
        if row.get('flow_path'):
            flow_path = resolve_manifest_path(row['flow_path'], base_dir)
            if not flow_path.is_file():
                errors.append(f"{prefix}: 找不到 {field_label('flow_path')}: {row['flow_path']}")

        if row.get('label_path'):
            label_path = resolve_manifest_path(row['label_path'], base_dir)
            if not label_path.is_file():
                errors.append(f"{prefix}: 找不到 {field_label('label_path')}: {row['label_path']}")

    summary = summarize_manifest(rows)
    strict_set = set(strict_unknown_for) if strict_unknown_for else set()
    for field in [
        'microscope_type',
        'particle_morphology',
        'density_level',
        'size_distribution',
        'quality_level',
        'adhesion_level',
        'is_large_image',
    ]:
        unknown_count = summary['by_field'].get(field, {}).get('unknown', 0)
        if unknown_count:
            msg = (f"{field_label(field)} 有 {unknown_count} 行为 unknown；"
                   f"正式训练前应补齐。")
            if field in strict_set:
                errors.append(msg)
            else:
                warnings.append(msg)

    # 检测 label_status reviewed 约束
    reviewed_splits = set(require_reviewed_for_splits) if require_reviewed_for_splits else set()
    if reviewed_splits:
        for field in ['label_status', 'split']:
            pass  # placeholder
        for index, row in enumerate(rows, start=2):
            prefix = f"row {index} ({row.get('record_id') or row.get('file_name')})"
            split = row.get('split', '')
            label_status = row.get('label_status', '')
            if split in reviewed_splits and label_status != 'reviewed':
                errors.append(
                    f"{prefix}: split='{split}' 要求 {field_label('label_status')} 为 'reviewed'，"
                    f"当前为 '{label_status}'。正式训练前所有 train/val 样本必须完成人工复核。")

    # 检测 strict_unknown_for 中的空字符串或 unknown
    if strict_set:
        for index, row in enumerate(rows, start=2):
            prefix = f"row {index} ({row.get('record_id') or row.get('file_name')})"
            for field in strict_set:
                value = str(row.get(field, '') or '').strip()
                if value == '' or value == 'unknown':
                    errors.append(
                        f"{prefix}: {field_label(field)} 不能为空或 'unknown'，"
                        f"当前为 '{value}'。正式训练前必须补齐。")

    # 检测禁止字段（比例尺相关字段不应出现在训练 manifest 中）
    forbidden_detected: dict[str, set] = {}
    for entry in extra_fields_rows:
        field_lower = str(entry.get('field', '')).lower()
        if field_lower in FORBIDDEN_MANIFEST_FIELDS:
            row_index = entry.get('row', '?')
            forbidden_detected.setdefault(field_lower, set()).add(row_index)

    for field_lower, row_set in forbidden_detected.items():
        label = FORBIDDEN_FIELD_LABELS_ZH.get(field_lower, field_lower)
        errors.append(
            f"{label} ({field_lower}) 不应出现在训练 manifest 中（共 {len(row_set)} 行，"
            f"如 row {sorted(row_set, key=lambda x: (str(x) if isinstance(x, int) else x))[:5]})。"
            f"比例尺信息属于 analysis_recipe.json，不属于 dataset_manifest。")

    return errors, warnings, summary


def summarize_manifest(rows: list[dict]) -> dict:
    summary = {'total_images': len(rows), 'by_field': {}}
    for field in [
        'split',
        'microscope_type',
        'polarity',
        'particle_morphology',
        'shape_type',
        'density_level',
        'size_distribution',
        'quality_level',
        'adhesion_level',
        'is_large_image',
        'label_status',
    ]:
        summary['by_field'][field] = dict(Counter(row.get(field, '') for row in rows))
    flow_count = sum(1 for row in rows if row.get('flow_path'))
    label_count = sum(1 for row in rows if row.get('label_path'))
    summary['flow_path_count'] = flow_count
    summary['label_path_count'] = label_count
    return summary


# ── V1 最低数据要求 ──

V1_MINIMUM_REQUIREMENTS = {
    "train_min_samples": 200,
    "val_min_samples": 30,
    "holdout_min_samples": 10,
    "large_image_source_min": 5,
    "holdout_large_image_min": 3,
}


def check_v1_minimum_requirements(rows):
    """检查 V1 最低数据数量和覆盖要求，返回 (errors, warnings)。"""
    errors = []
    warnings = []

    def split_rows(split_name):
        return [r for r in rows if r.get("split") == split_name]

    train_rows = split_rows("train")
    val_rows = split_rows("val")
    holdout_rows = split_rows("holdout")

    # 样本数
    for name, actual, minimum in [
        ("train", len(train_rows), V1_MINIMUM_REQUIREMENTS["train_min_samples"]),
        ("val", len(val_rows), V1_MINIMUM_REQUIREMENTS["val_min_samples"]),
        ("holdout", len(holdout_rows), V1_MINIMUM_REQUIREMENTS["holdout_min_samples"]),
    ]:
        if actual < minimum:
            errors.append(
                f"V1 最低 {name} 样本数不足: {actual} < {minimum}")

    # 大图源图数 (按 source_image_path 或 image_path 分组)
    all_large_sources = set()
    holdout_large_sources = set()
    for row in rows:
        if row.get("is_large_image") != "true":
            continue
        source = row.get("source_image_path") or row.get("image_path") or ""
        all_large_sources.add(source)
        if row.get("split") == "holdout":
            holdout_large_sources.add(source)

    if len(all_large_sources) < V1_MINIMUM_REQUIREMENTS["large_image_source_min"]:
        errors.append(
            f"V1 大图源图数不足: {len(all_large_sources)}"
            f" < {V1_MINIMUM_REQUIREMENTS['large_image_source_min']}")
    if len(holdout_large_sources) < V1_MINIMUM_REQUIREMENTS["holdout_large_image_min"]:
        errors.append(
            f"V1 holdout 大图数不足: {len(holdout_large_sources)}"
            f" < {V1_MINIMUM_REQUIREMENTS['holdout_large_image_min']}")

    # 显微镜类型 / 成像域仅用于分层分析，不作为 V1 推理正确性的硬 gate。
    micro_types = {r.get("microscope_type", "") for r in rows if r.get("split") in ("train", "val")}
    known_micro_types = {v for v in micro_types if v and v != "unknown"}
    if not known_micro_types:
        warnings.append(
            "train/val 未记录可靠显微镜类型或成像域；这不阻断训练，"
            "但会降低分层分析和数据覆盖诊断的可信度。")

    # train/val 全部 reviewed
    for r in train_rows + val_rows:
        if r.get("label_status") != "reviewed":
            errors.append(
                f"V1 要求 train/val label_status=reviewed，"
                f"但 {r.get('record_id')} 为 '{r.get('label_status')}'")
            break  # 只报告一次

    # 场景字段非空非 unknown
    for field in V1_REQUIRED_SCENE_FIELDS:
        for r in train_rows + val_rows:
            val = str(r.get(field, "") or "").strip()
            if val == "" or val == "unknown":
                errors.append(
                    f"V1 场景字段 {field_label(field)} 存在空值或 unknown: "
                    f"{r.get('record_id')} = '{val}'")
                break  # 每种字段只报告一次

    return errors, warnings
