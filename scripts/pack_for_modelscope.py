"""Pack the complete training project into a .tar.gz for ModelScope Notebook."""

import os
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# For importing model_flow
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model_flow.utils import imread_unicode

DATA_ROOT = os.path.join(HERE, "data", "particles")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXCLUDE_EXTS = {".pyc", ".pth", ".onnx", ".tar.gz", ".gz", ".zip"}
EXCLUDE_DIRS = {"__pycache__", "checkpoints", "onnx", ".git", "third_party",
                "experiments", "temp", "models--timm--convnext_small.dinov3_lvd1689m",
                ".playwright-mcp", ".claude"}

CRITICAL_FILES = [
    "model_flow",
    "scripts",
    "data/particles/train",
    "data/particles/val",
    "data/particles/flows_train",
    "data/particles/flows_val",
    "analysis_recipe.schema.json",
    "setup_env_modelscope.sh",
    "requirements_modelscope.txt",
    "models--timm--convnext_small.dinov3_lvd1689m",
]

FILES_TO_INCLUDE = [
    "analysis_recipe.schema.json",
]


def should_exclude_file(name, include_annotations=False):
    if name.startswith("."):
        return True
    if not include_annotations and name.endswith(".json"):
        return True
    if os.path.splitext(name)[1].lower() in EXCLUDE_EXTS:
        return True
    # Also explicitly include some critical JSONs
    for inc in FILES_TO_INCLUDE:
        if name == inc or name.endswith("/" + inc):
            return False
    return False


def should_exclude_dir(name):
    if name.startswith("."):
        return True
    if name in EXCLUDE_DIRS:
        return True
    return False


def add_path(tar, full_path, arcname, include_annotations=False):
    if os.path.isfile(full_path):
        if should_exclude_file(os.path.basename(full_path), include_annotations):
            return
        tar.add(full_path, arcname=arcname)
    elif os.path.isdir(full_path):
        for root, dirs, files in os.walk(full_path):
            dirs[:] = [d for d in dirs if not should_exclude_dir(d)]
            for fname in files:
                if should_exclude_file(fname, include_annotations):
                    continue
                src = os.path.join(root, fname)
                rel = os.path.relpath(src, os.path.dirname(full_path))
                tar.add(src, arcname=os.path.join(arcname, rel))


def verify_training_pairs():
    """验证 image/flow 配对、可读性、尺寸一致性。复用共享 check_split。"""
    from scripts.verify_training_pairs import check_split

    print("Verifying image/flow pairs before packaging...")
    ok = check_split("train", os.path.join(DATA_ROOT, "train"),
                     os.path.join(DATA_ROOT, "flows_train"))
    ok = check_split("val", os.path.join(DATA_ROOT, "val"),
                     os.path.join(DATA_ROOT, "flows_val")) and ok
    return ok


def find_dataset_manifest():
    for name in ("dataset_manifest.csv", "dataset_manifest.jsonl"):
        path = os.path.join(DATA_ROOT, name)
        if os.path.isfile(path):
            return path
    return None


def verify_dataset_manifest():
    manifest_path = find_dataset_manifest()
    if not manifest_path:
        print("  ERROR: data/particles/dataset_manifest.csv or .jsonl is required for formal training.")
        return False

    print(f"Verifying dataset manifest: {os.path.relpath(manifest_path, HERE)}")
    from model_flow.manifest.dataset_manifest_utils import (
        read_manifest, validate_manifest, check_v1_minimum_requirements,
        V1_REQUIRED_SCENE_FIELDS)

    try:
        rows, extra_fields_rows = read_manifest(
            manifest_path, collect_extra_fields=True)
        errors, warnings, summary = validate_manifest(
            rows,
            base_dir=str(HERE),
            require_flow_for_splits=("train", "val"),
            strict_unknown_for=V1_REQUIRED_SCENE_FIELDS,
            require_reviewed_for_splits=("train", "val"),
            extra_fields_rows=extra_fields_rows,
        )
    except Exception as exc:
        print(f"  ERROR: failed to read manifest: {exc}")
        return False

    if errors:
        print(f"  ERROR: manifest has {len(errors)} blocker(s).")
        for error in errors[:20]:
            print(f"    {error}")
        return False

    if warnings:
        print(f"  WARNING: manifest has {len(warnings)} warning(s).")
        for warning in warnings[:10]:
            print(f"    {warning}")

    # V1 硬性要求: 最低数量和覆盖
    v1_errors, v1_warnings = check_v1_minimum_requirements(rows)
    if v1_errors:
        print(f"  ERROR: V1 minimum requirements not met ({len(v1_errors)} failure(s)):")
        for e in v1_errors[:20]:
            print(f"    {e}")
        return False
    if v1_warnings:
        for w in v1_warnings[:10]:
            print(f"  WARNING: {w}")

    train_n = summary.get('by_field', {}).get('split', {}).get('train', 0)
    val_n = summary.get('by_field', {}).get('split', {}).get('val', 0)
    print(f"  Manifest OK: {len(rows)} rows (train={train_n}, val={val_n})")
    return True


def verify_critical_files():
    """检查关键文件/目录是否存在，以及 DINOv3 离线权重。"""
    print("Checking critical files...")
    ok = True
    for item in CRITICAL_FILES:
        path = os.path.join(HERE, item)
        if not os.path.exists(path):
            print(f"  MISSING: {item}")
            ok = False
            continue
        print(f"  OK: {item}")

    # DINOv3 offline smoke
    try:
        import torch
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HERE))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from model_flow.backbone import DINOv3Backbone
        backbone = DINOv3Backbone(
            model_name="convnext_small.dinov3_lvd1689m",
            pretrained=True, freeze=True)
        _ = backbone(torch.randn(1, 3, 256, 256))
        print("  OK: DINOv3 weights loadable")
    except Exception as exc:
        print(f"  FAIL: DINOv3 weights: {exc}")
        ok = False

    return ok


def package_data_particles(tar, include_annotations=False):
    """将 data/particles 下训练所需子目录打包。"""
    for subdir in ["train", "val", "flows_train", "flows_val"]:
        src = os.path.join(DATA_ROOT, subdir)
        if os.path.isdir(src):
            add_path(tar, src, f"data/particles/{subdir}",
                     include_annotations=include_annotations)

    # manifest 和 report
    for fname in ["dataset_manifest.csv", "dataset_manifest.jsonl",
                  "dataset_readiness_report.md"]:
        path = os.path.join(DATA_ROOT, fname)
        if os.path.isfile(path):
            tar.add(path, arcname=f"data/particles/{fname}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Pack training project for ModelScope Notebook.")
    parser.add_argument("--output", default="particles_flow_train.tar.gz",
                        help="Output tar.gz path.")
    parser.add_argument("--include-annotations", action="store_true",
                        help="Include annotation JSON files.")
    args = parser.parse_args()

    output = os.path.join(HERE, args.output) if not os.path.isabs(args.output) else args.output

    print("=" * 60)
    print("Packaging V1 training project for ModelScope")
    print("=" * 60)

    # 1. 验证关键文件
    if not verify_critical_files():
        print("\nERROR: Critical files missing.")
        sys.exit(1)

    # 2. 验证 training pairs
    print()
    if not verify_training_pairs():
        print("\nERROR: Training pair verification failed.")
        sys.exit(1)

    # 3. 验证 manifest
    print()
    if not verify_dataset_manifest():
        print("\nERROR: Dataset manifest verification failed.")
        sys.exit(1)

    # 4. 打包
    print()
    print(f"Creating {output} ...")
    with tarfile.open(output, "w:gz") as tar:
        # 核心代码
        add_path(tar, os.path.join(HERE, "model_flow"), "model_flow",
                 include_annotations=args.include_annotations)
        add_path(tar, os.path.join(HERE, "scripts"), "scripts",
                 include_annotations=args.include_annotations)

        # 特殊文件
        for fname in FILES_TO_INCLUDE:
            path = os.path.join(HERE, fname)
            if os.path.isfile(path):
                tar.add(path, arcname=fname)

        # 环境
        for fname in ["setup_env_modelscope.sh", "requirements_modelscope.txt"]:
            path = os.path.join(HERE, fname)
            if os.path.isfile(path):
                tar.add(path, arcname=fname)

        # 数据
        package_data_particles(tar, include_annotations=args.include_annotations)

    size_mb = os.path.getsize(output) / (1024 * 1024)
    print(f"\nPackage created: {output} ({size_mb:.1f} MB)")
    print()
    print("On ModelScope Notebook:")
    print(f"  tar -xzf {os.path.basename(output)}")
    print("  bash setup_env_modelscope.sh")
    print("  python scripts/verify_training_pairs.py")
    print("  python -m model_flow.flow_train --device cuda")


if __name__ == "__main__":
    main()
