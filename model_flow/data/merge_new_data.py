"""安全合并新训练数据到 data/particles/

通过像素级 MD5 检查图像内容是否已存在：
  - 已存在 → 用新数据覆盖旧数据（新标注可能更完整）
  - 不存在 → 新增

val 分流策略:
  - 无 --manifest: 基于内容哈希随机分配 (向后兼容)
  - 有 --manifest: 按数据集元数据分层抽样, 确保每种场景类型在 val 中都有代表

用法:
    # 预览（不实际写入）
    python -m model_flow.merge_new_data \
        --src-img-dir ./staging_images/ \
        --src-flow-dir ./staging_flows/ \
        --dry-run

    # 执行合并（随机 val 分流）
    python -m model_flow.merge_new_data \
        --src-img-dir ./staging_images/ \
        --src-flow-dir ./staging_flows/

    # 执行合并（按 manifest 分层 val 分流）
    python -m model_flow.merge_new_data \
        --src-img-dir ./staging_images/ \
        --src-flow-dir ./staging_flows/ \
        --val-ratio 0.1 \
        --manifest data/particles/dataset_manifest.csv
"""
import os
import sys
import hashlib
import argparse
import random
import struct
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..utils import imread_unicode, long_path


def compute_pixel_hash(img_path):
    """计算解码后原始像素的 MD5。

    返回 None 表示图片无法读取。
    基于像素内容而非文件字节，因此不同格式/压缩级别的相同图像会得到相同 hash。
    """
    img = imread_unicode(img_path)
    if img is None:
        return None
    return hashlib.md5(img.tobytes()).hexdigest()


def build_existing_index(img_dir):
    """扫描已有图片目录，构建 {pixel_hash: (filename, path)} 索引。

    如果多张图片 hash 相同（真正的重复），只保留第一张。
    """
    if not os.path.isdir(img_dir):
        return {}
    index = {}
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for fname in sorted(os.listdir(img_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in exts:
            continue
        path = os.path.join(img_dir, fname)
        h = compute_pixel_hash(path)
        if h is None:
            print(f"  Warning: cannot read existing image {fname}, skipped")
            continue
        if h in index:
            old_fname, _ = index[h]
            print(f"  Warning: duplicate content found in existing data: "
                  f"{fname} == {old_fname}, keeping {old_fname}")
            continue
        index[h] = (fname, path)
    return index


def collect_new_samples(src_img_dir, src_flow_dir):
    """收集新数据：图片 + 对应 .npy flow field 的配对列表。

    跳过无对应 .npy 的图片，跳过无对应图片的 .npy。
    """
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    samples = []
    skipped_no_flow = []
    skipped_no_img = []

    for fname in sorted(os.listdir(src_img_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in exts:
            continue
        name = os.path.splitext(fname)[0]
        img_path = os.path.join(src_img_dir, fname)
        flow_path = os.path.join(src_flow_dir, name + '.npy')
        if not os.path.exists(flow_path):
            skipped_no_flow.append(fname)
            continue
        samples.append((fname, name, img_path, flow_path))

    # 检查孤立的 .npy
    for fname in sorted(os.listdir(src_flow_dir)):
        if not fname.endswith('.npy'):
            continue
        name = os.path.splitext(fname)[0]
        # 检查是否有对应图片
        found = False
        for ext in exts:
            if os.path.exists(os.path.join(src_img_dir, name + ext)):
                found = True
                break
        if not found:
            skipped_no_img.append(fname)

    return samples, skipped_no_flow, skipped_no_img


def _build_manifest_lookup(manifest_path, group_fields):
    """Build {file_name: group_key} from manifest for stratified val splitting.

    Returns empty dict when manifest_path is empty/None.
    """
    if not manifest_path:
        return {}
    from ..manifest.dataset_manifest_utils import read_manifest
    rows = read_manifest(manifest_path)
    lookup = {}
    for row in rows:
        fname = row.get('file_name', '').strip()
        if not fname:
            continue
        key = '|'.join(
            str(row.get(f, '') or 'unknown').strip()
            for f in group_fields
        )
        lookup[fname] = key
    return lookup


def _assign_stratified_val(samples, manifest_lookup, val_ratio, seed):
    """Determine which new samples go to val using stratified per-group sampling.

    samples: list of (fname, name, img_path, flow_path) for NEW samples only
    manifest_lookup: {file_name: group_key} from manifest (may be incomplete)
    val_ratio: float, fraction to assign to val
    seed: int, for reproducible group-level shuffles

    Returns: set of fname assigned to val
    """
    groups = defaultdict(list)
    unknown = []
    for fname, name, img_path, flow_path in samples:
        key = manifest_lookup.get(fname)
        if key:
            groups[key].append(fname)
        else:
            unknown.append(fname)

    rng = random.Random(seed)
    val_assigned = set()

    for key, fnames in sorted(groups.items()):
        group_size = len(fnames)
        n_val = max(1, int(round(group_size * val_ratio))) if group_size > 0 else 0
        n_val = min(n_val, group_size)
        shuffled = sorted(fnames)
        rng.shuffle(shuffled)
        for fname in shuffled[:n_val]:
            val_assigned.add(fname)

    # Unknown (not in manifest): fall back to content-hash random
    if unknown:
        n_val_unknown = max(1, int(round(len(unknown) * val_ratio))) if unknown else 0
        n_val_unknown = min(n_val_unknown, len(unknown))
        # Use deterministic shuffle by filename for reproducibility
        unknown_sorted = sorted(unknown)
        rng.shuffle(unknown_sorted)
        for fname in unknown_sorted[:n_val_unknown]:
            val_assigned.add(fname)

    return val_assigned


def main():
    parser = argparse.ArgumentParser(
        description='安全合并新训练数据（基于图像内容去重）')
    parser.add_argument('--src-img-dir', required=True,
                        help='新图片目录')
    parser.add_argument('--src-flow-dir', required=True,
                        help='新 .npy flow field 目录')
    parser.add_argument('--target-img-dir', default='data/particles/train',
                        help='目标图片目录 (默认: data/particles/train)')
    parser.add_argument('--target-flow-dir', default='data/particles/flows_train',
                        help='目标 flow 目录 (默认: data/particles/flows_train)')
    parser.add_argument('--val-ratio', type=float, default=0.0,
                        help='新增样本分配到验证集的比例 (0.0=全部进train, 0.1=10%%进val, P3-5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，用于稳定切分 train/val (P3-5)')
    parser.add_argument('--manifest', default='',
                        help='dataset_manifest.csv/jsonl 路径。提供后按场景分层分配 val，'
                             '确保各场景类型在 val 中都有代表。')
    parser.add_argument('--val-group-fields',
                        default='microscope_type,density_level,size_distribution',
                        help='分层字段 (逗号分隔), 默认: microscope_type,density_level,size_distribution')
    parser.add_argument('--dry-run', action='store_true',
                        help='预览模式，不实际写入')
    args = parser.parse_args()

    # 验证源目录
    if not os.path.isdir(args.src_img_dir):
        print(f'Error: src-img-dir not found: {args.src_img_dir}')
        sys.exit(1)
    if not os.path.isdir(args.src_flow_dir):
        print(f'Error: src-flow-dir not found: {args.src_flow_dir}')
        sys.exit(1)

    # 收集新数据
    samples, skipped_no_flow, skipped_no_img = collect_new_samples(
        args.src_img_dir, args.src_flow_dir)

    if not samples:
        print('No valid image/.npy pairs found.')
        sys.exit(1)

    # 构建已有图片索引
    print(f'Indexing existing images in {args.target_img_dir}...')
    existing = build_existing_index(args.target_img_dir)
    print(f'  {len(existing)} existing images indexed.')

    # P3-5: 验证集分流 — 准备 val 目录和哈希种子
    val_img_dir = args.target_img_dir.replace('/train', '/val').replace('\\train', '\\val')
    val_flow_dir = args.target_flow_dir.replace('/flows_train', '/flows_val').replace('\\flows_train', '\\flows_val')
    do_split = args.val_ratio > 0.0

    if do_split:
        val_existing = build_existing_index(val_img_dir)
        print(f'  {len(val_existing)} existing val images indexed.')
    else:
        val_existing = {}

    # 确保目标目录存在
    if not args.dry_run:
        os.makedirs(args.target_img_dir, exist_ok=True)
        os.makedirs(args.target_flow_dir, exist_ok=True)
        if do_split:
            os.makedirs(val_img_dir, exist_ok=True)
            os.makedirs(val_flow_dir, exist_ok=True)

    replaced = []
    added_to_train = []
    added_to_val = []
    skipped_bad_img = []

    print(f'\nProcessing {len(samples)} new samples...')
    if do_split:
        print(f'  val_ratio={args.val_ratio}, seed={args.seed}')

    # ── P3-6: 按 manifest 分层分配 val ──
    manifest_lookup = {}
    val_assigned = set()
    use_stratified_split = do_split and bool(args.manifest)
    if use_stratified_split:
        group_fields = [
            item.strip() for item in args.val_group_fields.split(',') if item.strip()
        ]
        manifest_lookup = _build_manifest_lookup(args.manifest, group_fields)
        if manifest_lookup:
            # 预先计算所有新样本(不含覆盖)的 val 分配
            new_only = [
                (fname, name, img_path, flow_path)
                for fname, name, img_path, flow_path in samples
            ]
            val_assigned = _assign_stratified_val(
                new_only, manifest_lookup, args.val_ratio, args.seed)
            n_manifest_hits = sum(
                1 for fname, _, _, _ in new_only if fname in manifest_lookup)
            print(f'  Stratified val split: manifest hits={n_manifest_hits}/{len(new_only)}, '
                  f'val_assigned={len(val_assigned)}')
            # 打印各组 val 分配详情
            from collections import Counter
            group_val = Counter()
            group_total = Counter()
            for fname, _, _, _ in new_only:
                key = manifest_lookup.get(fname, '(not in manifest)')
                group_total[key] += 1
                if fname in val_assigned:
                    group_val[key] += 1
            for key in sorted(group_total):
                print(f'    {key}: {group_val[key]}/{group_total[key]} → val')
        else:
            print('  Manifest provided but no entries matched; '
                  'falling back to content-hash random.')

    for fname, name, img_path, flow_path in tqdm(samples):
        # 计算新图片 hash
        h = compute_pixel_hash(img_path)
        if h is None:
            skipped_bad_img.append(fname)
            continue

        if h in existing:
            # ── 内容已存在：覆盖 ──
            old_fname, old_path = existing[h]
            old_name = os.path.splitext(old_fname)[0]
            old_flow = os.path.join(args.target_flow_dir, old_name + '.npy')
            old_img = old_path

            if args.dry_run:
                replaced.append((fname, old_fname))
                # 更新索引：新 hash 可能不变，但文件名变了
                # dry-run 不更新索引
                continue

            # 删除旧文件
            if os.path.exists(old_img):
                os.remove(long_path(old_img))
            if os.path.exists(old_flow):
                os.remove(long_path(old_flow))

            # 写入新文件
            dst_img = os.path.join(args.target_img_dir, fname)
            dst_flow = os.path.join(args.target_flow_dir, name + '.npy')
            shutil.copy2(img_path, long_path(dst_img))
            shutil.copy2(flow_path, long_path(dst_flow))

            # 更新索引：新文件名替代旧文件名
            existing[h] = (fname, dst_img)
            replaced.append((fname, old_fname))

        else:
            # ── 新图片：直接添加（P3-5/P3-6: 按 val_ratio 分流）──
            # 优先使用 manifest 分层分配, 回退到内容哈希随机
            if use_stratified_split and val_assigned:
                to_val = fname in val_assigned
            else:
                # 基于内容哈希稳定分配 train/val（相同内容总是分配到同一集合）
                import struct
                hash_bytes = bytes.fromhex(h) if len(h) >= 8 else hashlib.md5(h.encode()).digest()
                hash_int = struct.unpack('<Q', hash_bytes[:8])[0]
                rng = random.Random(hash_int + args.seed)
                to_val = do_split and rng.random() < args.val_ratio

            if to_val:
                target_img = val_img_dir
                target_flow = val_flow_dir
                target_existing = val_existing
            else:
                target_img = args.target_img_dir
                target_flow = args.target_flow_dir
                target_existing = existing

            if args.dry_run:
                added_to_val.append(fname) if to_val else added_to_train.append(fname)
                continue

            dst_img = os.path.join(target_img, fname)
            dst_flow = os.path.join(target_flow, name + '.npy')

            # P1-1: 检查目标文件名是否已被占用 (同名但不同内容)
            if os.path.exists(long_path(dst_img)):
                base, ext = os.path.splitext(fname)
                counter = 1
                while os.path.exists(long_path(dst_img)):
                    new_fname = f"{base}_{counter}{ext}"
                    dst_img = os.path.join(target_img, new_fname)
                    dst_flow = os.path.join(target_flow,
                                            f"{base}_{counter}.npy")
                    counter += 1
                print(f"  Renamed: {fname} -> {new_fname} (target name already used by different content)")

            shutil.copy2(img_path, long_path(dst_img))
            shutil.copy2(flow_path, long_path(dst_flow))

            target_existing[h] = (os.path.basename(dst_img), dst_img)
            if to_val:
                added_to_val.append(os.path.basename(dst_img))
            else:
                added_to_train.append(os.path.basename(dst_img))

    # ── 报告 ──
    print(f'\n{"=" * 55}')
    if args.dry_run:
        print('  DRY RUN — 未实际写入')
    print(f'  新增 (train): {len(added_to_train)}')
    if do_split:
        print(f'  新增 (val):   {len(added_to_val)}')
    print(f'  覆盖: {len(replaced)}')
    if skipped_no_flow:
        print(f'  跳过 (无 .npy): {len(skipped_no_flow)}')
    if skipped_no_img:
        print(f'  跳过 (无图片): {len(skipped_no_img)}')
    if skipped_bad_img:
        print(f'  跳过 (无法读取): {len(skipped_bad_img)}')
    print(f'{"=" * 55}')

    if replaced:
        print(f'\n覆盖详情:')
        for new_fname, old_fname in replaced:
            print(f'  {new_fname} → 覆盖 {old_fname} (内容相同)')

    if skipped_no_flow:
        print(f'\n无对应 .npy 的图片:')
        for f in skipped_no_flow:
            print(f'  {f}')

    if skipped_no_img:
        print(f'\n无对应图片的 .npy:')
        for f in skipped_no_img:
            print(f'  {f}')

    print(f'\n目标目录 (train): {args.target_img_dir}')
    print(f'目标 flow (train): {args.target_flow_dir}')
    if do_split:
        print(f'目标目录 (val):   {val_img_dir}')
        print(f'目标 flow (val):   {val_flow_dir}')


if __name__ == '__main__':
    main()
