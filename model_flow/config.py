"""Flow Field 训练配置"""
import torch


class Config:
    # ─── 数据集 ───
    data_root = "data/particles"
    train_img_dir = "data/particles/train"
    val_img_dir = "data/particles/val"
    train_flow_dir = "data/particles/flows_train"   # 预计算的 GT .npy
    val_flow_dir = "data/particles/flows_val"

    # ─── 输入 ───
    input_size = 1024
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    pad_value = 114  # BGR 空间填充值，经 ImageNet 归一化后各通道接近零值

    # ─── Backbone ───
    backbone_name = "convnext_small.dinov3_lvd1689m"
    backbone_out_channels = [96, 192, 384, 768]
    backbone_strides = [4, 8, 16, 32]
    freeze_backbone = True
    unfreeze_backbone_after_epoch = 50  # 应约为 max_epochs 的 50%
    unfreeze_backbone_lr_ratio = 0.1

    # ─── Neck ───
    neck_channels = 128
    fpn_upsample_mode = "nearest"  # FPN 上采样: "nearest" (主流) | "bilinear" (实验)

    # ─── Flow Head ───
    flow_head_residual_blocks = 3
    flow_head_out_channels = 3  # dy, dx, cellprob
    cellprob_bias_init = 0.0     # cellprob 输出 bias 初始值 (logit), 0=默认, 可按前景比例设置

    # ─── 训练 ───
    batch_size = 8
    unfreeze_batch_size = 4       # 解冻 backbone stage 2/3 后降低 batch，避免显存峰值 OOM
    num_workers = 4
    max_epochs = 100
    base_lr = 0.001
    weight_decay = 0.05
    warmup_iters = 500
    val_interval = 5
    val_compute_masks = True  # 验证时使用 Cellpose compute_masks 计算 mask-level 指标
    min_lr_ratio = 0.01          # CosineAnnealingLR eta_min = base_lr * min_lr_ratio

    # ─── 损失（P1-2 配置化）───
    flow_loss_weight = 0.5
    flow_scale = 5.0           # GT flow 放大倍数 (Cellpose 标准)
    cellprob_loss_weight = 1.0
    # flow loss 计算模式: "all_pixels" (Cellpose 官方行为) | "foreground" (仅前景像素, 实验项)
    flow_loss_mode = "all_pixels"
    # cellprob pos_weight 模式: "none" (Cellpose 官方行为) | "batch" | "sample" | "capped_sample" (实验项)
    cellprob_pos_weight_mode = "none"
    cellprob_pos_weight_max = 10.0  # capped_sample 模式上限

    # ─── FlowDynamics 参数 (推理时使用, Euler 积分模式) ───
    # 默认值仅用于训练期/调试占位；正式发布必须由分层参数搜索结果通过 --flow-params 导出覆盖。
    fd_cellprob_threshold = -3.0   # logit 空间, 最终颗粒置信度过滤 (mean logit < this → 丢弃)
    fd_niter = 200                 # Euler 迭代次数 (Cellpose 默认)
    fd_min_size = 50               # 调试默认值；正式发布使用分层参数搜索得到的 min_size
    fd_flow_threshold = 0.4         # remove_bad_flow_masks flow error 阈值
    fd_max_size_fraction = 0.5      # 最大 mask 面积占整图比例
    # 注: Euler 前景门控阈值为 logit > 0.0 (prob > 0.5), 硬编码于 C++ FlowDynamics,
    #     与 Cellpose compute_masks 内部行为一致, 不可配置
    inference_cellprob_threshold = 0.5  # 概率空间 (Python compute_masks)

    # ─── 数据增强 ───
    aug_flip_prob = 0.5
    aug_rotate_prob = 0.3
    aug_jitter_prob = 0.3
    aug_scale_range = (0.8, 1.2)  # 随机缩放范围 (P2-4); 禁用请设 aug_scale_prob=0
    aug_scale_prob = 0.3

    # ─── 设备 ───
    device = "cuda"
