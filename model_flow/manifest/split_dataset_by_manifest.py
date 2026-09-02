"""按 manifest 场景字段分层划分 train/val，并复制图片和标签。

用法:
    # 已分配 split（仅复制）
    python -m model_flow.manifest.split_dataset_by_manifest \
        --manifest data/particles/dataset_manifest.csv \
        --output-root data/particles

    # 按场景分层分配 val（步骤 4.1）
    python -m model_flow.manifest.split_dataset_by_manifest \
        --manifest temp/staging/dataset_manifest_draft.csv \
        --src-img-dir temp/staging/images --src-label-dir temp/staging/labels \
        --output-root data/particles \
        --val-ratio 0.1 --min-val-samples 30
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

from .dataset_manifest_utils import read_manifest


def copy_file(src: Path, dst: Path, overwrite: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"  WARNING: source not found: {src}")
        return
    if not overwrite and dst.exists():
        return
    shutil.copy2(src, dst)


def assign_stratified_val(rows: list[dict], val_ratio: float,
                          seed: int = 42,
                          min_val_samples: int = 0) -> set[str]:
    """按 V1 场景字段分层，每组内随机抽选 val 比例的 record_id。

    大图切出的 tile 必须以源图为单位划分，避免同一源图的重叠 tile
    同时出现在 train 和 val，造成像素级泄露。
    """
    rng = random.Random(seed)
    source_units: dict[str, dict] = {}
    for row in rows:
        source_key = row.get("source_image_path") or row.get("image_path") or row["record_id"]
        if source_key not in source_units:
            source_units[source_key] = {
                "key": (
                    row.get("microscope_type", "unknown") or "unknown",
                    row.get("particle_morphology") or row.get("shape_type") or "unknown",
                    row.get("size_distribution", "unknown") or "unknown",
                    row.get("adhesion_level", "unknown") or "unknown",
                    row.get("density_level", "unknown") or "unknown",
                    row.get("quality_level", "unknown") or "unknown",
                ),
                "record_ids": [],
            }
        source_units[source_key]["record_ids"].append(row["record_id"])

    groups = defaultdict(list)
    for source_key, source_unit in source_units.items():
        key = source_unit["key"]
        groups[key].append(source_key)

    val_source_keys = set()
    all_source_keys: list[str] = []
    small_groups: list[str] = []
    for key, source_keys in sorted(groups.items()):
        n_val_raw = int(round(len(source_keys) * val_ratio)) if len(source_keys) else 0
        # 当 val_ratio > 0 且组内源图 >= 4 时, 至少选 1 个进 val 保证覆盖;
        # 小组 (<=3) 按比例计算, 不强取, 以免该场景全部进入 val 导致 train 缺失。
        if len(source_keys) >= 4 and val_ratio > 0:
            n_val = max(1, n_val_raw) if len(source_keys) > 0 else 0
        else:
            n_val = n_val_raw
            if n_val == 0 and len(source_keys) > 0:
                small_groups.append(
                    f"  {key}: {len(source_keys)} source(s), all → train")
        n_val = min(n_val, len(source_keys))
        rng.shuffle(source_keys)
        all_source_keys.extend(source_keys)
        for source_key in source_keys[:n_val]:
            val_source_keys.add(source_key)

    if small_groups:
        print("  INFO: the following small stratification groups produced "
              "zero val samples (consider adding more data):")
        for msg in small_groups:
            print(msg)

    def build_val_ids() -> set[str]:
        selected_ids = set()
        for selected_source_key in val_source_keys:
            for record_id in source_units[selected_source_key]["record_ids"]:
                selected_ids.add(record_id)
        return selected_ids

    if min_val_samples > 0:
        current_val_ids = build_val_ids()
        # 区分小组源(≤3)和大组源, 优先从大组回填, 保护稀有场景
        small_group_sources: set[str] = set()
        for key, source_keys in groups.items():
            if len(source_keys) <= 3:
                for sk in source_keys:
                    small_group_sources.add(sk)

        remaining_large = [k for k in all_source_keys
                           if k not in val_source_keys
                           and k not in small_group_sources]
        remaining_small = [k for k in all_source_keys
                           if k not in val_source_keys
                           and k in small_group_sources]
        rng.shuffle(remaining_large)
        rng.shuffle(remaining_small)

        # 第一轮: 从大组回填
        for source_key in remaining_large:
            if len(current_val_ids) >= min_val_samples:
                break
            val_source_keys.add(source_key)
            current_val_ids.update(source_units[source_key]["record_ids"])

        # 第二轮: 仍不足时从小组回填, 记录受影响的稀有场景
        small_backfill: list[str] = []
        for source_key in remaining_small:
            if len(current_val_ids) >= min_val_samples:
                break
            val_source_keys.add(source_key)
            current_val_ids.update(source_units[source_key]["record_ids"])
            small_backfill.append(source_key)

        if small_backfill:
            print("  WARNING: min-val-samples backfill pulled the following "
                  "small-scene sources into val; train may lack these scenes:")
            for sk in small_backfill:
                skey = source_units[sk]["key"]
                print(f"    {skey}: source={sk}")

    val_ids = set()
    for source_key in val_source_keys:
        for record_id in source_units[source_key]["record_ids"]:
            val_ids.add(record_id)

    # 后检查: 是否有场景的 train 源图数为 0 (全部进了 val)?
    train_source_keys = set(all_source_keys) - val_source_keys
    val_only_scenes: list[str] = []
    for key, source_keys in sorted(groups.items()):
        train_in_group = [sk for sk in source_keys if sk in train_source_keys]
        if not train_in_group:
            val_in_group = [sk for sk in source_keys if sk in val_source_keys]
            if val_in_group:
                val_only_scenes.append(
                    f"  {key}: {len(val_in_group)}/{len(source_keys)} "
                    f"source(s) in val, 0 in train")

    if val_only_scenes:
        print("  WARNING: the following scenes exist only in val "
              "(0 train sources). Model will not learn these scenes:")
        for msg in val_only_scenes:
            print(msg)

    return val_ids


def count_val_source_units(rows: list[dict], val_ids: set[str]) -> int:
    """统计进入 val 的源图单元数量，用于日志提示。"""
    val_sources = set()
    for row in rows:
        if row.get("record_id") in val_ids:
            val_sources.add(row.get("source_image_path") or row.get("image_path") or row["record_id"])
    return len(val_sources)


def is_auto_split_candidate(row: dict) -> bool:
    """Return True when a row may be assigned to train/val automatically."""
    if row.get("split") == "holdout":
        return False
    if row.get("tile_role") == "holdout_full_image":
        return False
    return True


def check_source_split_leakage(rows: list[dict], row_splits: dict[str, str]) -> list[str]:
    """返回跨 split 的源图列表。正常自动划分不应产生任何结果。"""
    source_to_splits = defaultdict(set)
    for row in rows:
        record_id = row.get("record_id", "")
        split = row_splits.get(record_id)
        if not split:
            continue
        source_key = row.get("source_image_path") or row.get("image_path") or record_id
        source_to_splits[source_key].add(split)
    return [source_key for source_key, splits in source_to_splits.items()
            if len(splits) > 1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Split dataset by manifest and materialize directories.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--base-dir', default='.')
    parser.add_argument('--splits', default='train,val',
                        help='Comma-separated splits.')
    parser.add_argument('--require-flow-for-splits', default='',
                        help='Comma-separated splits that must have flow_path. Keep empty before flow generation.')
    parser.add_argument('--overwrite', action='store_true')

    # New for step 4.1: source lookup and val auto-split
    parser.add_argument('--src-img-dir', default='',
                        help='Source image directory (overrides manifest image_path resolution).')
    parser.add_argument('--src-label-dir', default='',
                        help='Source label directory (for copying *_labels.png).')
    parser.add_argument('--val-ratio', type=float, default=0.0,
                        help='Auto-split: assign this fraction to val by scene stratification. '
                             'When > 0, assigns eligible rows to train/val while preserving '
                             'holdout rows for explicit --splits holdout materialization.')
    parser.add_argument('--min-val-samples', type=int, default=0,
                        help='Auto-split: keep adding source units until val has at least this '
                             'many rows when possible. Use 30 for V1 formal training.')

    args = parser.parse_args()

    try:
        rows = read_manifest(args.manifest)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    requested_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    require_flow = {s.strip() for s in args.require_flow_for_splits.split(",") if s.strip()}
    output_root = Path(args.output_root)

    # 自动分配 val
    val_assigned: set[str] = set()
    auto_split = args.val_ratio > 0.0
    if auto_split:
        auto_rows = [row for row in rows if is_auto_split_candidate(row)]
        auto_candidate_ids = {row.get("record_id", "") for row in auto_rows}
        val_assigned = assign_stratified_val(
            auto_rows, args.val_ratio, min_val_samples=args.min_val_samples)
        n_train = len(auto_rows) - len(val_assigned)
        n_val = len(val_assigned)
        n_holdout_skipped = len(rows) - len(auto_rows)
        n_val_sources = count_val_source_units(auto_rows, val_assigned)
        print(f"Auto split (val_ratio={args.val_ratio}): "
              f"train={n_train} val={n_val} val_source_units={n_val_sources} "
              f"holdout_skipped={n_holdout_skipped}")
        if n_val < 3:
            print("  WARNING: val samples < 3, consider increasing --val-ratio")
        if args.min_val_samples > 0 and n_val < args.min_val_samples:
            print(f"  WARNING: val samples < --min-val-samples ({args.min_val_samples}); "
                  "dataset is too small or grouped source units are too coarse.")

    # ── 目录安全清理 ──
    output_root = output_root.resolve()
    dirs_to_clear: list[Path] = []
    for split in requested_splits:
        dirs_to_clear.append(output_root / split)
        dirs_to_clear.append(output_root / f"flows_{split}")

    if args.overwrite:
        for d in dirs_to_clear:
            d = d.resolve()
            dr = output_root.resolve()
            try:
                d.relative_to(dr)
            except ValueError:
                raise RuntimeError(f"refusing to clear directory outside output_root: {d}")
            if d.is_dir():
                import shutil as _shutil
                _shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)
                print(f"  cleared: {d}")
    else:
        non_empty = [d for d in dirs_to_clear if d.is_dir() and any(d.iterdir())]
        if non_empty:
            names = [str(d.relative_to(output_root)) for d in non_empty]
            raise SystemExit(
                f"输出目录非空，请使用 --overwrite 进行受控重建: {', '.join(names)}")

    counts = {split: 0 for split in requested_splits}
    row_splits: dict[str, str] = {}
    for row in rows:
        # 决定 split
        if auto_split:
            if row.get("record_id", "") in auto_candidate_ids:
                split = "val" if row["record_id"] in val_assigned else "train"
            else:
                split = row.get("split")
        else:
            split = row.get("split")
        if split not in requested_splits:
            continue
        row_splits[row.get("record_id", "")] = split

        if split in require_flow and not row.get("flow_path"):
            raise RuntimeError(f"flow_path required for {split}: {row['record_id']}")

        # 图片
        if args.src_img_dir:
            src_img = Path(args.src_img_dir) / row["file_name"]
        else:
            src_img = Path(args.base_dir) / row["image_path"]
        dst_img = output_root / split / row["file_name"]
        copy_file(src_img, dst_img, overwrite=args.overwrite)

        # .npy flow
        if row.get("flow_path"):
            flow_src = Path(args.base_dir) / row["flow_path"]
            flow_name = Path(row["file_name"]).stem + ".npy"
            flow_dst = output_root / f"flows_{split}" / flow_name
            copy_file(flow_src, flow_dst, overwrite=args.overwrite)

        # *_labels.png
        if args.src_label_dir:
            label_name = Path(row["file_name"]).stem + "_labels.png"
            label_src = Path(args.src_label_dir) / label_name
            label_dst = output_root / split / label_name
            copy_file(label_src, label_dst, overwrite=args.overwrite)

        counts[split] += 1

    print("\nCopied:")
    for split in sorted(counts):
        print(f"  {split}: {counts[split]} images")
    leaked_sources = check_source_split_leakage(rows, row_splits)
    if leaked_sources:
        raise SystemExit(
            "同一源图的 tile 被分到了多个 split，请检查 manifest/source_image_path: "
            + ", ".join(leaked_sources[:10]))
    if not any(counts.values()):
        raise SystemExit("No rows copied. Check --splits and manifest.")


if __name__ == "__main__":
    main()
