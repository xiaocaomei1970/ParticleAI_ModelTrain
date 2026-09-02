"""Flow Head: 从 FPN P2 特征预测 (dy, dx, cellprob)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块: 3×3 Conv + BN + SiLU, skip connection"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class FlowHead(nn.Module):
    """Flow Field 预测头。

    输入: P2 feature (B, 128, H/4, W/4)
    输出: (B, 3, H/4, W/4) → dy, dx, cellprob
    """

    def __init__(self, in_channels: int = 128,
                 num_blocks: int = 3,
                 out_channels: int = 3,
                 cellprob_bias_init: float = 0.0):
        """P2-2: SiLU 激活使用 'linear' gain; cellprob bias 可按前景比例初始化。

        Args:
            cellprob_bias_init: cellprob 通道 bias 初始值（logit 空间）。
                默认 0.0。建议按训练集前景比例设置，如 log(fg_ratio / (1-fg_ratio)).
        """
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResidualBlock(in_channels) for _ in range(num_blocks)
        ])
        self.head = nn.Conv2d(in_channels, out_channels, 1)

        self._init_weights(cellprob_bias_init)

    def _init_weights(self, cellprob_bias_init: float = 0.0):
        # P2-2: SiLU 无专用 gain，使用 'linear' (gain=1.0) 更合适
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                         nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 输出层 cellprob bias 初始化（dy/dx bias 保持 0）
        if cellprob_bias_init != 0.0 and self.head.bias is not None:
            with torch.no_grad():
                self.head.bias[2] = cellprob_bias_init

    def forward(self, p2: torch.Tensor) -> torch.Tensor:
        """p2: (B, 128, H/4, W/4) → (B, 3, H/4, W/4)"""
        x = self.blocks(p2)
        return self.head(x)


class FlowModel(nn.Module):
    """完整 Flow Field 模型: Backbone + Neck + FlowHead"""

    def __init__(self, cfg):
        super().__init__()
        from .backbone import DINOv3Backbone
        from .neck import LightFPN

        self.backbone = DINOv3Backbone(
            model_name=cfg.backbone_name,
            pretrained=True,
            freeze=cfg.freeze_backbone,
            out_indices=(0, 1, 2, 3),
        )
        self.neck = LightFPN(
            in_channels=self.backbone.out_channels,
            out_channels=cfg.neck_channels,
            upsample_mode=getattr(cfg, 'fpn_upsample_mode', 'nearest'),
        )
        self.flow_head = FlowHead(
            in_channels=cfg.neck_channels,
            num_blocks=cfg.flow_head_residual_blocks,
            out_channels=cfg.flow_head_out_channels,
            cellprob_bias_init=getattr(cfg, 'cellprob_bias_init', 0.0),
        )
        self.cfg = cfg

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 1024, 1024) → (B, 3, 256, 256) @ stride 4"""
        feats = self.backbone(x)
        feats = self.neck(feats)
        p2 = feats[0]  # stride 4
        return self.flow_head(p2)
