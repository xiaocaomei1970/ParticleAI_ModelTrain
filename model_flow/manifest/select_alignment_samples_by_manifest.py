"""Select FlowDynamics tuning samples from dataset_manifest metadata."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .dataset_manifest_utils import read_manifest


def parse_fields(value: str) -> list[str]:
    return [item.strip() for item in value.split(',') if item.strip()]


def sample_score(row: dict) -> float:
    score = 0.0
    if row.get('density_level') in {'sparse', 'dense'}:
        score += 1.0
    if row.get('size_distribution') in {'small', 'wide', 'large'}:
        score += 1.0
    if row.get('quality_level') in {'low_contrast', 'blurred', 'noisy_texture'}:
        score += 1.0
    if row.get('adhesion_level') in {'few_touching', 'many_touching'}:
        score += 1.0
    morphology = row.get('particle_morphology') or row.get('shape_type')
    if morphology in {'irregular', 'flake_like', 'fiber_like', 'porous', 'mixed'}:
        score += 0.5
    return score


def group_key(row: dict, fields: list[str]) -> tuple:
    return tuple(row.get(field, 'unknown') or 'unknown' for field in fields)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create a stratified sample list from dataset_manifest.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out-samples', default='temp/stratified_val_samples.txt')
    parser.add_argument('--out-report', default='',
                        help='Optional JSON report of candidate/selected counts per stratum.')
    parser.add_argument('--split', default='val')
    parser.add_argument('--top', type=int, default=40)
    parser.add_argument('--group-fields',
                        default='microscope_type,particle_morphology,density_level,size_distribution,adhesion_level,quality_level',
                        help='Comma-separated manifest fields for stratification.')
    parser.add_argument('--output-field', choices=['file_name', 'image_path'],
                        default='file_name')
    parser.add_argument('--require-flow', action='store_true',
                        help='Only sample rows with non-empty flow_path.')
    args = parser.parse_args()

    if args.top <= 0:
        raise SystemExit("--top must be > 0")

    rows = [
        row for row in read_manifest(args.manifest)
        if row.get('split') == args.split
    ]
    if args.require_flow:
        rows = [row for row in rows if row.get('flow_path')]
    if not rows:
        raise SystemExit(f"No rows found for split={args.split}")

    fields = parse_fields(args.group_fields)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row, fields)].append(row)
    group_total_counts = {key: len(value) for key, value in groups.items()}

    for group_rows in groups.values():
        group_rows.sort(key=lambda row: (-sample_score(row), row['file_name']))

    selected: list[dict] = []
    selected_ids = set()
    group_items = sorted(groups.items(), key=lambda item: str(item[0]))

    while len(selected) < args.top:
        added = False
        for _, group_rows in group_items:
            while group_rows and group_rows[0]['record_id'] in selected_ids:
                group_rows.pop(0)
            if not group_rows:
                continue
            row = group_rows.pop(0)
            selected.append(row)
            selected_ids.add(row['record_id'])
            added = True
            if len(selected) >= args.top:
                break
        if not added:
            break

    selected.sort(key=lambda row: (
        row.get('microscope_type', ''),
        row.get('density_level', ''),
        row.get('size_distribution', ''),
        row.get('file_name', ''),
    ))

    out_path = Path(args.out_samples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as handle:
        for row in selected:
            handle.write(row[args.output_field] + '\n')

    selected_group_counts: dict[tuple, int] = defaultdict(int)
    for row in selected:
        selected_group_counts[group_key(row, fields)] += 1

    out_report = args.out_report
    if not out_report:
        out_report = str(out_path.with_suffix('.report.json'))
    report_path = Path(out_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    group_report = []
    for key in sorted(group_total_counts, key=str):
        group_report.append({
            'group': {field: value for field, value in zip(fields, key)},
            'candidate_count': group_total_counts[key],
            'selected_count': selected_group_counts.get(key, 0),
        })
    with report_path.open('w', encoding='utf-8') as handle:
        json.dump({
            'manifest': args.manifest,
            'split': args.split,
            'group_fields': fields,
            'candidate_count': len(rows),
            'selected_count': len(selected),
            'groups': group_report,
        }, handle, ensure_ascii=False, indent=2)

    print("Manifest-stratified samples selected.")
    print(f"  split: {args.split}")
    print(f"  candidates: {len(rows)}")
    print(f"  groups: {len(groups)}")
    print(f"  selected: {len(selected)}")
    print(f"  output: {out_path}")
    print(f"  report: {report_path}")
    print("\nGroup coverage:")
    for item in group_report[:30]:
        group_text = " | ".join(item['group'].values())
        print(
            f"  {group_text}: candidates={item['candidate_count']} "
            f"selected={item['selected_count']}")
    if len(group_report) > 30:
        print(f"  ... {len(group_report) - 30} more groups in report")
    print("\nPreview:")
    for index, row in enumerate(selected[:20], start=1):
        print(
            f"  {index:02d}. {row['file_name']} | {row['microscope_type']} | "
            f"{row.get('particle_morphology') or row.get('shape_type')} | "
            f"{row['density_level']} | {row['size_distribution']} | "
            f"{row['quality_level']} | score={sample_score(row):.1f}")


if __name__ == '__main__':
    main()
