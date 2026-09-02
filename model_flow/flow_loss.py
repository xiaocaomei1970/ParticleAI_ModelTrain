"""Flow Field 损失: MSE(flow) + BCE(cellprob)（可配置）。

公式（Cellpose 官方模式 — 默认）:
  flow_mse       = MSE(pred_flow, 5 * GT_flow)  全像素
  cellprob_loss  = BCEWithLogits(pred_cellprob, GT_mask), 无 pos_weight
  total          = flow_weight * flow_mse + cellprob_weight * cellprob_loss

默认 flow_weight=0.5，对应 Cellpose 官方 _loss_fn_seg() 中 flow MSE 的 /2。

实验模式（通过配置切换）:
  flow_loss_mode:         "all_pixels" (Cellpose 官方, 默认) | "foreground" (仅前景, 实验)
  cellprob_pos_weight_mode: "none" (Cellpose 官方, 默认) | "sample" | "batch" | "capped_sample"

注意: 仅 GT flow 做 ×5 放大，pred flow 不缩放。
使用非 Cellpose 官方模式时，训练协议与 Cellpose v4 不一致，分割行为可能偏离。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowLoss(nn.Module):
    """Flow Field 损失（默认 Cellpose 官方行为）。

    支持 flow loss 模式和 cellprob pos_weight 模式的组合配置。
    默认配置与 Cellpose 官方 _loss_fn_seg() 等价。
    """

    def __init__(self, flow_weight: float = 0.5, flow_scale: float = 5.0,
                 cellprob_weight: float = 1.0,
                 flow_loss_mode: str = "all_pixels",
                 cellprob_pos_weight_mode: str = "none",
                 cellprob_pos_weight_max: float = 10.0):
        super().__init__()
        self.flow_weight = flow_weight
        self.flow_scale = flow_scale
        self.cellprob_weight = cellprob_weight
        self.flow_loss_mode = flow_loss_mode
        self.cellprob_pos_weight_mode = cellprob_pos_weight_mode
        self.cellprob_pos_weight_max = cellprob_pos_weight_max
        self.mse = nn.MSELoss(reduction='none')

        valid_flow_modes = {"foreground", "all_pixels"}
        valid_cp_modes = {"none", "batch", "sample", "capped_sample"}
        if flow_loss_mode not in valid_flow_modes:
            raise ValueError(f"flow_loss_mode must be one of {valid_flow_modes}")
        if cellprob_pos_weight_mode not in valid_cp_modes:
            raise ValueError(f"cellprob_pos_weight_mode must be one of {valid_cp_modes}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """pred, target: (B, 3, H, W)
        pred channels: [dy, dx, cellprob]
        target channels: [cellprob, dy, dx]  ← 注意顺序!
        """
        # target 来自 labels_to_flows: [cellprob, dy, dx]
        # pred 来自 FlowHead: [dy, dx, cellprob]
        # 需要对齐通道

        # ── 前景遮罩（共用 GT cellprob 通道）──
        gt_mask = target[:, 0]  # (B, H, W), 0/1 binary
        fg_mask = (gt_mask > 0.5).float()  # 前景=1, 背景=0
        bg_mask = 1.0 - fg_mask

        # ── Flow loss ──
        pred_flow = pred[:, :2]                       # (B, 2, H, W)
        gt_flow = target[:, 1:3] * self.flow_scale    # (B, 2, H, W) ×5

        flow_se = self.mse(pred_flow, gt_flow)  # (B, 2, H, W), no reduction
        flow_se = flow_se.mean(dim=1)  # (B, H, W), dy/dx 取均值

        if self.flow_loss_mode == "foreground":
            # 加权平均: 前景 MSE / 前景像素数，避免背景主导
            fg_count = fg_mask.sum(dim=(1, 2)).clamp(min=1)  # (B,)
            flow_loss = (flow_se * fg_mask).sum(dim=(1, 2)) / fg_count
        else:  # "all_pixels" — Cellpose 官方行为
            flow_loss = flow_se.mean(dim=(1, 2))  # Cellpose 的 /2 由默认 flow_weight=0.5 提供
        flow_loss = flow_loss.mean()  # batch 平均

        # ── Cellprob loss: 按配置选择 pos_weight 策略 ──
        pred_cellprob = pred[:, 2]  # (B, H, W) logits

        if self.cellprob_pos_weight_mode == "none":
            # Cellpose 官方: 无 pos_weight
            cellprob_loss = F.binary_cross_entropy_with_logits(
                pred_cellprob, gt_mask)
        elif self.cellprob_pos_weight_mode == "batch":
            # 整个 batch 统一 pos_weight
            n_bg = bg_mask.sum().clamp(min=1)
            n_fg = fg_mask.sum().clamp(min=1)
            pos_w = (n_bg / n_fg).clamp(max=self.cellprob_pos_weight_max)
            cellprob_loss = F.binary_cross_entropy_with_logits(
                pred_cellprob, gt_mask, pos_weight=pos_w)
        elif self.cellprob_pos_weight_mode == "sample":
            # 每个 batch 样本独立计算 pos_weight
            cellprob_loss = 0.0
            for b in range(pred.shape[0]):
                n_bg = bg_mask[b].sum().clamp(min=1)
                n_fg = fg_mask[b].sum().clamp(min=1)
                pos_w = n_bg / n_fg
                cellprob_loss += F.binary_cross_entropy_with_logits(
                    pred_cellprob[b], gt_mask[b], pos_weight=pos_w)
            cellprob_loss = cellprob_loss / pred.shape[0]
        else:  # "capped_sample"
            cellprob_loss = 0.0
            for b in range(pred.shape[0]):
                n_bg = bg_mask[b].sum().clamp(min=1)
                n_fg = fg_mask[b].sum().clamp(min=1)
                pos_w = torch.clamp(n_bg / n_fg, max=self.cellprob_pos_weight_max)
                cellprob_loss += F.binary_cross_entropy_with_logits(
                    pred_cellprob[b], gt_mask[b], pos_weight=pos_w)
            cellprob_loss = cellprob_loss / pred.shape[0]

        total = self.flow_weight * flow_loss + self.cellprob_weight * cellprob_loss
        return total, flow_loss, cellprob_loss
