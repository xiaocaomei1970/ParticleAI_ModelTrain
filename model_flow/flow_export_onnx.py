"""ONNX 导出: backbone + neck+flow_head

用法:
    python -m model_flow.flow_export_onnx --checkpoint checkpoints/best.pth --output ./onnx/ --flow-params flow_dynamics_best_params.json
"""
import os
import argparse

import torch
import torch.nn as nn

from .config import Config
from .flow_head import FlowModel


def export_backbone(cfg, model, output_path):
    """导出 backbone.onnx (4 级特征图)，使用 checkpoint 中的权重（包含微调后的 stage）"""
    print('Exporting backbone.onnx...')
    backbone = model.backbone
    backbone.eval()

    dummy = torch.randn(1, 3, cfg.input_size, cfg.input_size)
    with torch.no_grad():
        feats = backbone(dummy)
    for i, f in enumerate(feats):
        print(f'  Stage {i}: shape={list(f.shape)}')

    torch.onnx.export(
        backbone, dummy, output_path,
        opset_version=17,
        input_names=['input'],
        output_names=['stage0', 'stage1', 'stage2', 'stage3'],
        dynamic_axes={'input': {0: 'batch'}},  # 固定 1024×1024，仅 batch 动态
    )

    import onnx
    onnx.checker.check_model(output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'  backbone.onnx OK ({size_mb:.1f} MB)')


def export_neck_head(cfg, model, output_path):
    """导出 neck_head.onnx (FPN + Flow Head)，使用 checkpoint 中的权重"""
    print('Exporting neck_head.onnx...')

    neck = model.neck
    flow_head = model.flow_head
    neck.eval()
    flow_head.eval()

    class NeckHeadWrapper(nn.Module):
        def __init__(self, neck, flow_head):
            super().__init__()
            self.neck = neck
            self.flow_head = flow_head

        def forward(self, c0, c1, c2, c3):
            feats = self.neck([c0, c1, c2, c3])
            p2 = feats[0]
            flow = self.flow_head(p2)
            return flow  # (B, 3, H/4, W/4)

    wrapper = NeckHeadWrapper(neck, flow_head)
    wrapper.eval()

    H = cfg.input_size
    dummies = (
        torch.randn(1, 96, H // 4, H // 4),
        torch.randn(1, 192, H // 8, H // 8),
        torch.randn(1, 384, H // 16, H // 16),
        torch.randn(1, 768, H // 32, H // 32),
    )

    torch.onnx.export(
        wrapper, dummies, output_path,
        opset_version=17,
        input_names=['stage0', 'stage1', 'stage2', 'stage3'],
        output_names=['flow'],  # (dy, dx, cellprob) @ stride 4
    )

    import onnx
    onnx.checker.check_model(output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'  neck_head.onnx OK ({size_mb:.1f} MB)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/best.pth')
    parser.add_argument('--output', default='./onnx/')
    parser.add_argument('--flow-params', default='',
                        help='tune_flow_dynamics.py 输出的 flow_dynamics_best_params.json')
    parser.add_argument('--allow-config-defaults', action='store_true',
                        help='允许不传 --flow-params，使用 Config 默认 FlowDynamics 参数。默认禁止，避免误导出旧参数。')
    args = parser.parse_args()

    if not args.flow_params and not args.allow_config_defaults:
        parser.error(
            '--flow-params is required. If you intentionally want Config defaults, '
            'pass --allow-config-defaults explicitly.'
        )

    cfg = Config()
    os.makedirs(args.output, exist_ok=True)

    flow_params = {
        'fd_cellprob_threshold': cfg.fd_cellprob_threshold,
        'fd_niter': cfg.fd_niter,
        'fd_min_size': cfg.fd_min_size,
        'fd_flow_threshold': cfg.fd_flow_threshold,
        'fd_max_size_fraction': cfg.fd_max_size_fraction,
    }
    if args.flow_params:
        import json
        print(f'Loading FlowDynamics params: {args.flow_params}')
        with open(args.flow_params, 'r', encoding='utf-8') as f:
            data = json.load(f)
        best = data.get('best_params', data)
        if 'cellprob_threshold_logit' in best:
            flow_params['fd_cellprob_threshold'] = float(best['cellprob_threshold_logit'])
        if 'niter' in best:
            flow_params['fd_niter'] = int(best['niter'])
        if 'min_size' in best:
            flow_params['fd_min_size'] = int(best['min_size'])
        if 'flow_threshold' in best:
            flow_params['fd_flow_threshold'] = float(best['flow_threshold'])
        if 'max_size_fraction' in best:
            flow_params['fd_max_size_fraction'] = float(best['max_size_fraction'])

    # 加载完整模型（包含可能微调过的 backbone）
    print(f'Loading checkpoint: {args.checkpoint}')
    model = FlowModel(cfg)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'  Loaded full model (epoch {ckpt.get("epoch", "?")})')
    else:
        # 兼容旧格式
        model.neck.load_state_dict(ckpt['neck'])
        model.flow_head.load_state_dict(ckpt['flow_head'])
        print('  Loaded neck + flow_head (legacy format, backbone is pretrained)')

    export_backbone(cfg, model, os.path.join(args.output, 'backbone.onnx'))
    export_neck_head(cfg, model,
                     os.path.join(args.output, 'neck_head.onnx'))

    # 保存配置 (确保输出到 ONNX 目录内而非父目录)
    import json
    output_dir = args.output.rstrip('/').rstrip('\\')
    config_path = os.path.join(output_dir, 'flow_inference_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'schema_version': 1,
            'input_size': cfg.input_size,
            'output_stride': 4,
            'fixed_input_size': True,
            'mean': list(cfg.mean),
            'std': list(cfg.std),
            'pad_value': cfg.pad_value,
            # Euler 前景门控: 硬编码 logit > 0.0 (prob > 0.5), 不可配置
            'euler_cellprob_threshold_logit': 0.0,
            'euler_cellprob_threshold_probability': 0.5,
            # 最终颗粒置信度过滤 (mean logit < this → 丢弃)
            'fd_cellprob_threshold': flow_params['fd_cellprob_threshold'],
            'fd_niter': flow_params['fd_niter'],
            'fd_min_size': flow_params['fd_min_size'],
            'fd_flow_threshold': flow_params['fd_flow_threshold'],
            'fd_max_size_fraction': flow_params['fd_max_size_fraction'],
        }, f, indent=2)
    print(f'  flow_inference_config.json saved')

    print(f'\nDone! ONNX models in {args.output}/')
    print('  backbone.onnx  - ConvNeXt-S DINOv3')
    print('  neck_head.onnx - FPN + Flow Head → (dy,dx,cellprob)@s4')
    print('  flow_inference_config.json - preprocessing + FlowDynamics params')


if __name__ == '__main__':
    main()
