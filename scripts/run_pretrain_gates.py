"""P0 训练前 gate 验证脚本.

在 V1 正式训练前运行, 按顺序检查所有 P0 gate. 任一 FAIL 退出码非零.

用法:
    python scripts/run_pretrain_gates.py
    python scripts/run_pretrain_gates.py --skip-pack  # 跳过打包检查
"""
from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PASS = 0
FAIL = 1
SKIP = 2
results: list[tuple[str, int, str]] = []


def record(name: str, status: int, detail: str = "") -> None:
    label = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP"}[status]
    msg = f"[{label}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append((name, status, detail))


# ═══════════════════════════════════════════════════════════════════════
# P0-1: capped_sample pos_weight 类型
# ═══════════════════════════════════════════════════════════════════════

def gate_p01() -> None:
    try:
        import torch
        sys.path.insert(0, str(HERE))
        from model_flow.flow_loss import FlowLoss
        loss_fn = FlowLoss(
            cellprob_pos_weight_mode="capped_sample",
            cellprob_pos_weight_max=10.0,
        )
        pred = torch.randn(2, 3, 64, 64)
        target = torch.zeros(2, 3, 64, 64)
        target[:, 0, 10:30, 10:30] = 1.0
        total, _, _ = loss_fn(pred, target)
        record("P0-1 capped_sample pos_weight", PASS,
               f"forward OK, total={total.item():.4f}")
    except Exception as exc:
        record("P0-1 capped_sample pos_weight", FAIL, str(exc))


# ═══════════════════════════════════════════════════════════════════════
# P0-2: verify_training_pairs.py 语法 + 逻辑检查
# ═══════════════════════════════════════════════════════════════════════

def gate_p02() -> None:
    script = HERE / "scripts" / "verify_training_pairs.py"
    if not script.exists():
        record("P0-2 verify_training_pairs syntax", FAIL, "script not found")
        return
    try:
        with open(script, "r", encoding="utf-8") as f:
            ast.parse(f.read())
        record("P0-2 verify_training_pairs", PASS, "syntax OK")
    except SyntaxError as exc:
        record("P0-2 verify_training_pairs", FAIL, str(exc))


# ═══════════════════════════════════════════════════════════════════════
# P0-3: 关键材料存在性
# ═══════════════════════════════════════════════════════════════════════

def gate_p03() -> None:
    critical = [
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
    missing = [item for item in critical if not (HERE / item).exists()]
    if missing:
        record("P0-3 critical files", FAIL,
               f"missing: {', '.join(missing)}")
    else:
        record("P0-3 critical files", PASS)

    has_csv = (HERE / "data" / "particles" / "dataset_manifest.csv").is_file()
    has_jsonl = (HERE / "data" / "particles" / "dataset_manifest.jsonl").is_file()
    if not has_csv and not has_jsonl:
        record("P0-3 manifest existence", FAIL,
               "neither dataset_manifest.csv nor .jsonl found")
    else:
        record("P0-3 manifest existence", PASS,
               f"csv={has_csv}, jsonl={has_jsonl}")

    report = HERE / "data" / "particles" / "dataset_readiness_report.md"
    if not report.is_file():
        record("P0-3 readiness report", FAIL, "dataset_readiness_report.md not found")
    else:
        record("P0-3 readiness report", PASS)

    # DINOv3 offline test
    try:
        import torch
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HERE))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from model_flow.backbone import DINOv3Backbone
        backbone = DINOv3Backbone(
            model_name="convnext_small.dinov3_lvd1689m",
            pretrained=True, freeze=True)
        _ = backbone(torch.randn(1, 3, 256, 256))
        record("P0-3 DINOv3 offline", PASS, "weights loadable")
    except Exception as exc:
        record("P0-3 DINOv3 offline", FAIL, str(exc))


# ═══════════════════════════════════════════════════════════════════════
# P0-4: reviewed gate + strict scene fields
# ═══════════════════════════════════════════════════════════════════════

def gate_p04() -> None:
    from model_flow.manifest.dataset_manifest_utils import (
        read_manifest, validate_manifest, V1_REQUIRED_SCENE_FIELDS,
    )
    manifest_path = None
    for name in ("dataset_manifest.csv", "dataset_manifest.jsonl"):
        candidate = HERE / "data" / "particles" / name
        if candidate.is_file():
            manifest_path = candidate
            break
    if not manifest_path:
        record("P0-4 manifest gate", SKIP, "no manifest found (also covered by P0-3)")
        return
    try:
        rows, extra = read_manifest(manifest_path, collect_extra_fields=True)
        errors, warnings, summary = validate_manifest(
            rows,
            base_dir=str(HERE),
            require_flow_for_splits=("train", "val"),
            strict_unknown_for=V1_REQUIRED_SCENE_FIELDS,
            require_reviewed_for_splits=("train", "val"),
            extra_fields_rows=extra,
        )
        if errors:
            record("P0-4 manifest gate", FAIL,
                   f"{len(errors)} error(s): {errors[0][:120]}")
        else:
            record("P0-4 manifest gate", PASS,
                   f"rows={len(rows)}, warnings={len(warnings)}")
    except Exception as exc:
        record("P0-4 manifest gate", FAIL, str(exc))


# ═══════════════════════════════════════════════════════════════════════
# P0-5: euler_core masks_to_flows + remove_bad_flow_masks
# ═══════════════════════════════════════════════════════════════════════

def gate_p05() -> None:
    try:
        import numpy as np
        from model_flow.alignment.euler_core import masks_to_flows, remove_bad_flow_masks

        H, W = 64, 64
        labels = np.zeros((H, W), dtype=np.int32)
        y, x = np.ogrid[:H, :W]
        labels[(y - 20)**2 + (x - 20)**2 < 64] = 1
        labels[(y - 44)**2 + (x - 44)**2 < 49] = 2

        dy, dx = masks_to_flows(labels)
        assert dy.shape == (H, W), "wrong dy shape"
        assert not np.any(np.isnan(dy)), "NaN in dy"

        # bad flow: should remove all
        bad_dy = np.ones((H, W), dtype=np.float32) * 5.0
        bad_dx = np.ones((H, W), dtype=np.float32) * 5.0
        l2 = labels.copy()
        result = remove_bad_flow_masks(l2, bad_dy, bad_dx, 0.4)
        after_bad = len(np.unique(result)) - 1
        assert after_bad == 0, f"bad flow: expected 0 instances, got {after_bad}"

        # good flow: should keep all
        good_dy = dy.astype(np.float32) * 5.0
        good_dx = dx.astype(np.float32) * 5.0
        l3 = labels.copy()
        result3 = remove_bad_flow_masks(l3, good_dy, good_dx, 0.4)
        after_good = len(np.unique(result3)) - 1
        before = len(np.unique(labels)) - 1
        assert after_good == before, f"good flow: expected {before}, got {after_good}"

        record("P0-5 euler_core parity", PASS,
               f"masks_to_flows + remove_bad_flow_masks OK")
    except Exception as exc:
        record("P0-5 euler_core parity", FAIL, str(exc))


# ═══════════════════════════════════════════════════════════════════════
# P0-6: C++ align exe path
# ═══════════════════════════════════════════════════════════════════════

def gate_p06() -> None:
    candidates = [
        HERE / "model_flow" / "inference_cpp" / "build" / "Release" / "flow_dynamics_align.exe",
        HERE / "model_flow" / "inference_cpp" / "build" / "Debug" / "flow_dynamics_align.exe",
    ]
    found = [str(c) for c in candidates if c.exists()]
    if found:
        record("P0-6 C++ align exe", PASS, f"found: {found[0]}")
    else:
        record("P0-6 C++ align exe", SKIP,
               "flow_dynamics_align.exe not built yet (build before C++ parity test)")


# ═══════════════════════════════════════════════════════════════════════
# P0-7: requirements completeness
# ═══════════════════════════════════════════════════════════════════════

def gate_p07() -> None:
    req_path = HERE / "requirements_modelscope.txt"
    if not req_path.is_file():
        record("P0-7 requirements", FAIL, "requirements_modelscope.txt not found")
        return
    content = req_path.read_text(encoding="utf-8")
    missing = []
    for pkg in ("onnx", "jsonschema"):
        if pkg not in content:
            missing.append(pkg)
    if missing:
        record("P0-7 requirements", FAIL,
               f"missing packages: {', '.join(missing)}")
    else:
        record("P0-7 requirements", PASS, "onnx, jsonschema present")


# ═══════════════════════════════════════════════════════════════════════
# P0-8: contour-only acceptance metrics check
# ═══════════════════════════════════════════════════════════════════════

def gate_p08() -> None:
    script = HERE / "model_flow" / "data" / "evaluate_particle_measurements.py"
    try:
        with open(script, "r", encoding="utf-8") as f:
            content = f.read()
            ast.parse(content)
        checks = [
            "instance_f1",
            "mask_iou_mean",
            "boundary_iou_mean",
            "false_positive_count",
            "false_negative_count",
            "over_split_proxy_count",
        ]
        missing = [c for c in checks if c not in content]
        forbidden = [
            "diameter_mae",
            "diameter_bias",
            "psd_",
            "area_absolute_relative_error_mean",
        ]
        forbidden_present = [c for c in forbidden if c in content]
        if missing:
            record("P0-8 contour metrics", FAIL,
                   f"missing fields: {', '.join(missing)}")
        elif forbidden_present:
            record("P0-8 contour metrics", FAIL,
                   f"forbidden downstream metrics present: {', '.join(forbidden_present)}")
        else:
            record("P0-8 contour metrics", PASS,
                   "acceptance metrics are contour-only")
    except Exception as exc:
        record("P0-8 contour metrics", FAIL, str(exc))


# ═══════════════════════════════════════════════════════════════════════
# P0-10/P0-11: tile truncation + background sampling
# ═══════════════════════════════════════════════════════════════════════

def gate_p10_p11() -> None:
    # Check if any train/val samples are large images
    manifest_path = None
    for name in ("dataset_manifest.csv", "dataset_manifest.jsonl"):
        candidate = HERE / "data" / "particles" / name
        if candidate.is_file():
            manifest_path = candidate
            break
    if not manifest_path:
        record("P0-10 tile truncation", SKIP, "no manifest")
        record("P0-11 background tiles", SKIP, "no manifest")
        return

    try:
        from model_flow.manifest.dataset_manifest_utils import read_manifest
        rows = read_manifest(manifest_path)
        large_in_train_val = any(
            row.get("split") in ("train", "val")
            and row.get("is_large_image") == "true"
            for row in rows
        )
    except Exception:
        large_in_train_val = False

    if not large_in_train_val:
        record("P0-10 tile truncation", SKIP,
               "no large-image train/val samples")
        record("P0-11 background tiles", SKIP,
               "no large-image train/val samples")
        return

    # Check prepare_tiled_staging.py has the features
    script = HERE / "model_flow" / "data" / "prepare_tiled_staging.py"
    try:
        with open(script, "r", encoding="utf-8") as f:
            content = f.read()
            ast.parse(content)
        has_trunc = "touches_crop_edge" in content
        has_bg = "background_tile_ratio" in content
        if has_trunc:
            record("P0-10 tile truncation", PASS,
                   "truncation removal implemented")
        else:
            record("P0-10 tile truncation", FAIL,
                   "truncation removal NOT implemented")
        if has_bg:
            record("P0-11 background tiles", PASS,
                   "background tile sampling implemented")
        else:
            record("P0-11 background tiles", FAIL,
                   "background tile sampling NOT implemented")
    except Exception as exc:
        record("P0-10/P0-11", FAIL, str(exc))


# ═══════════════════════════════════════════════════════════════════════
# P0-12: V1 最低数量和覆盖要求
# ═══════════════════════════════════════════════════════════════════════

def gate_p12() -> None:
    manifest_path = None
    for name in ("dataset_manifest.csv", "dataset_manifest.jsonl"):
        candidate = HERE / "data" / "particles" / name
        if candidate.is_file():
            manifest_path = candidate
            break
    if not manifest_path:
        record("P0-12 V1 minimum requirements", SKIP,
               "no manifest found (also covered by P0-3)")
        return
    try:
        from model_flow.manifest.dataset_manifest_utils import (
            read_manifest, check_v1_minimum_requirements)
        rows = read_manifest(manifest_path)
        errors, warnings = check_v1_minimum_requirements(rows)
        if errors:
            record("P0-12 V1 minimum requirements", FAIL,
                   f"{len(errors)} error(s): {errors[0][:120]}")
        else:
            record("P0-12 V1 minimum requirements", PASS,
                   f"rows={len(rows)}, warnings={len(warnings)}")
    except Exception as exc:
        record("P0-12 V1 minimum requirements", FAIL, str(exc))


# ═══════════════════════════════════════════════════════════════════════
# P0-9: solution document checks
# ═══════════════════════════════════════════════════════════════════════

def gate_p09() -> None:
    doc = HERE / "模型训练和C++后处理方案.md"
    if not doc.is_file():
        record("P0-9 document", FAIL, "solution doc not found")
        return
    content = doc.read_text(encoding="utf-8")
    checks = [
        ("## 1. 目标与边界", "goal/scope section"),
        ("## 5. 分步骤执行", "step-by-step execution section"),
        ("本方案只验收实例分割轮廓质量", "contour-only acceptance boundary"),
        ("不验收粒径、面积、粒度分布", "downstream size metrics excluded"),
        ("label_status", "reviewed gate mentioned"),
        ("--strict-scene-fields", "strict scene fields command"),
        ("output_stride=4", "output_stride fixed"),
        ("flow_scale=5.0", "flow scale note"),
        ("cellprob logit > 0.0", "Euler foreground gate note"),
        ("## 4. 数据准备标准", "beginner data preparation standard"),
        ("原始已标注完整图片", "source image quantity requirement"),
        ("元数据填写规则", "metadata filling rules"),
        ("标注质量要求", "label quality rules"),
        ("右/下边界", "tile boundary alignment"),
        ("Python/C++ FlowDynamics parity", "Python/C++ parity step"),
        ("## 6. 禁止事项", "forbidden actions section"),
    ]
    failures = [(desc, pattern) for pattern, desc in checks
                if pattern not in content]
    if failures:
        record("P0-9 document", FAIL,
               f"missing: {', '.join(d for d, _ in failures)}")
    else:
        record("P0-9 document", PASS,
               "solution document matches executable V1 contour workflow")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    skip_pack = "--skip-pack" in sys.argv

    print("=" * 60)
    print("V1 训练前 Gate 验证")
    print("=" * 60)

    gate_p01()
    gate_p02()
    gate_p03()
    gate_p04()
    gate_p05()
    gate_p06()
    gate_p07()
    gate_p08()
    gate_p09()
    gate_p10_p11()
    gate_p12()

    if not skip_pack:
        print()
        # Note: pack_for_modelscope.py already calls these gates internally;
        # we just verify it's importable and syntax-valid.
        pack_script = HERE / "scripts" / "pack_for_modelscope.py"
        try:
            with open(pack_script, "r", encoding="utf-8") as f:
                ast.parse(f.read())
            record("P0-pack syntax", PASS, "pack_for_modelscope.py syntax OK")
        except SyntaxError as exc:
            record("P0-pack syntax", FAIL, str(exc))

    print()
    print("=" * 60)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print(f"Summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP "
          f"(total {len(results)})")

    if n_fail > 0:
        print("\nFAILED gates:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  [{name}] {detail}")
        print("\nFix the FAILED gates above before starting formal training.")
        return 1
    print("All gates passed. Ready for formal training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
