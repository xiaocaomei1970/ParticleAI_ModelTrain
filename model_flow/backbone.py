"""ConvNeXt-S DINOv3 Backbone（冻结特征提取器）"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import timm


LOCAL_TIMM_CACHE_DIR = "models--timm--convnext_small.dinov3_lvd1689m"


def _configure_local_hf_cache():
    """Use the bundled timm/HuggingFace cache when the training package includes it."""
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parents[1],
    ]
    for root in candidates:
        cache_dir = root / LOCAL_TIMM_CACHE_DIR
        if cache_dir.is_dir():
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root))
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            return str(cache_dir)
    return None


class DINOv3Backbone(nn.Module):
    """封装 timm 的 ConvNeXt-S DINOv3 为多级特征提取器。

    输出 (默认 out_indices=(0,1,2,3)):
        features[0]: (B, 96,  H/4,  W/4)   stride 4  (P2 用)
        features[1]: (B, 192, H/8,  W/8)   stride 8
        features[2]: (B, 384, H/16, W/16)  stride 16
        features[3]: (B, 768, H/32, W/32)  stride 32
    """

    def __init__(self, model_name: str = "convnext_small.dinov3_lvd1689m",
                 pretrained: bool = True, freeze: bool = True,
                 out_indices: tuple = (0, 1, 2, 3)):
        super().__init__()
        if pretrained:
            _configure_local_hf_cache()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
        )
        self.out_indices = out_indices
        self.out_channels = [self.backbone.feature_info[i]["num_chs"]
                             for i in out_indices]
        self.strides = [self.backbone.feature_info[i]["reduction"]
                        for i in out_indices]

        self._is_frozen = False
        self._unfrozen_stage_indices = set()
        if freeze:
            self.freeze()

    def freeze(self):
        """冻结全部 backbone 参数"""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        self._is_frozen = True
        self._unfrozen_stage_indices.clear()

    def unfreeze_stages(self, stage_indices: list):
        """P3: 解冻指定 stage 的参数用于微调。

        stage_indices: [0,1,2,3] 对应 ConvNeXt 的 4 个 stage
        通常解冻最后 1-2 个 stage (如 [2,3] 解冻 stride 16, 32)
        """
        # timm features_only 模式: stages_0, stages_1, ...
        for si in stage_indices:
            stage = getattr(self.backbone, f'stages_{si}')
            for p in stage.parameters():
                p.requires_grad_(True)
            self._unfrozen_stage_indices.add(si)
        self._is_frozen = False

    def _keep_frozen_children_eval(self):
        """让未解冻的 backbone 子模块保持 eval，避免冻结层出现训练态随机性。"""
        for child in self.backbone.children():
            has_trainable_params = any(
                p.requires_grad for p in child.parameters(recurse=True)
            )
            if not has_trainable_params:
                child.eval()

    def train(self, mode: bool = True):
        """保持冻结部分在 eval 模式, 解冻部分正常切换"""
        super().train(mode)
        if self._is_frozen:
            self.backbone.eval()
        elif mode:
            self._keep_frozen_children_eval()
        return self

    def forward(self, x: torch.Tensor) -> list:
        """x: (B, 3, H, W) → 返回多级特征图列表"""
        if self._is_frozen:
            with torch.no_grad():
                all_feats = self.backbone(x)
        else:
            all_feats = self.backbone(x)
        return [all_feats[i] for i in self.out_indices]
