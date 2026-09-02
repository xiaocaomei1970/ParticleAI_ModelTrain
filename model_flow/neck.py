"""轻量级 Feature Pyramid Network (FPN) — 4级版本

P2 (stride 4) 新增用于高分辨率 mask 生成 (评审2-#2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightFPN(nn.Module):
    """对 ConvNeXt backbone 输出的 4 级特征做 top-down 融合。

    输入 (来自 backbone):
        C2: (B, 96,  H/4,  W/4)   stride 4  [P2 mask用]
        C3: (B, 192, H/8,  W/8)   stride 8  [P3 检测用]
        C4: (B, 384, H/16, W/16)  stride 16 [P4 检测用]
        C5: (B, 768, H/32, W/32)  stride 32 [P5 检测用]

    输出 (统一 128 通道):
        P2: (B, 128, H/4,  W/4)   stride 4  [mask_feat]
        P3: (B, 128, H/8,  W/8)   stride 8  [检测]
        P4: (B, 128, H/16, W/16)  stride 16 [检测]
        P5: (B, 128, H/32, W/32)  stride 32 [检测]
    """

    def __init__(self, in_channels: list = [96, 192, 384, 768],
                 out_channels: int = 128,
                 upsample_mode: str = "nearest"):
        super().__init__()
        self.num_levels = len(in_channels)  # 4
        self.upsample_mode = upsample_mode

        # Lateral convs: 对齐各层通道
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        # 3×3 smooth convs
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(out_channels) for _ in in_channels
        ])
        self.act = nn.SiLU(inplace=True)

    def forward(self, feats: list) -> list:
        """feats: C2, C3, C4, C5 (stride 4, 8, 16, 32)"""
        c2, c3, c4, c5 = feats

        # Top-down: 从高到低逐级融合
        mode = self.upsample_mode
        kwargs = {'align_corners': False} if mode == 'bilinear' else {}

        # P5
        p5 = self.lateral_convs[3](c5)
        # P4
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode=mode, **kwargs)
        p4 = self.lateral_convs[2](c4) + p5_up
        # P3
        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode=mode, **kwargs)
        p3 = self.lateral_convs[1](c3) + p4_up
        # P2 (新增: stride 4, 高分辨率 mask 专用)
        p3_up = F.interpolate(p3, size=c2.shape[-2:], mode=mode, **kwargs)
        p2 = self.lateral_convs[0](c2) + p3_up

        # Smooth + BN + Act
        outs = [p2, p3, p4, p5]
        outs = [self.act(self.bns[i](self.smooth_convs[i](f)))
                for i, f in enumerate(outs)]
        return outs  # [P2, P3, P4, P5]
