#pragma once
#include "Particle.h"
#include "ParticleIdManager.h"
#include <opencv2/core.hpp>
#include <utility>
#include <vector>

/// Convert Cellpose flow fields to instance masks via Euler integration.
/// Behavior is intended to match Cellpose official compute_masks.
/// Historical alignment metrics are obsolete after foreground-gating fixes;
/// re-run align_flow_dynamics.py before recording release metrics.
/// bad-flow 过滤已升级为与 Cellpose masks_to_flows_gpu 一致的扩散流场实现。
class FlowDynamics {
public:
    enum class BoundaryParticlePolicy {
        Include,
        Exclude,
        MarkAndExcludeFromPsd
    };

    struct Config {
        // ── Thresholds ─────────────────────────────────────────
        float cellprobThreshold = -3.0f; // logit-space, 最终颗粒过滤 (meanLogit < this → 丢弃)

        // ── Euler integration ──────────────────────────────────
        int   niter              = 200;     // Euler 迭代次数 (Cellpose 默认 200)
        // 注: 流场前景遮罩阈值固定在 logit > 0.0 (prob > 0.5)
        //     与 Cellpose compute_masks 内部行为一致，不可配置

        // ── Flow error 过滤 (remove_bad_flow_masks 等效) ──────
        float flowThreshold      = 0.4f;    // flow error > this → 丢弃 (Cellpose 默认 0.4)

        // ── Filtering ──────────────────────────────────────────
        int   minSize           = 50;    // minimum mask area (pixels)
        float maxSizeFraction   = 0.5f;  // max mask size as fraction of image area
        // minVolume removed — get_masks_torch uses h>10 seeds + h_slc>2 dilation,
        // not the old histogram volume threshold. Kept as comment for migration reference.
        BoundaryParticlePolicy boundaryParticlePolicy = BoundaryParticlePolicy::Include;
        float edgeTouchMarginPx = 1.0f;
    };

    FlowDynamics(const Config& cfg = {});

    /// @param dy       CV_32FC1 vertical flow field (raw model output)
    /// @param dx       CV_32FC1 horizontal flow field (raw model output)
    /// @param cellprob CV_32FC1 cell probability (raw logits)
    /// @param idm      shared ID allocator (if null, uses 1-based per-call IDs)
    /// @return list of detected particles
    std::vector<Particle> computeMasks(const cv::Mat& dy, const cv::Mat& dx,
                                       const cv::Mat& cellprob,
                                       ParticleIdManager* idm = nullptr);

private:
    /// Euler integration with bilinear interpolation.
    /// @param dP  (H, W, 2) flow field [dy, dx] — raw model output, NOT normalized
    /// @param pts (N, 2) initial pixel coordinates [y, x] in [-1, 1] normalized space
    /// @return (N, 2) final positions after niter Euler steps
    cv::Mat stepsInterp(const cv::Mat& dP, const cv::Mat& pts) const;

    // ── Post-processing ────────────────────────────────────────
    /// Official Cellpose get_masks_torch clustering: rpad=20 histogram,
    /// 5x5 max-pool seed detection, 5-iteration 3x3 dilation with h_slc>2.
    /// @param rst               convergence sink linear indices (one per foreground pixel)
    /// @param foregroundIndices original foreground pixel linear indices
    /// @param H, W              image dimensions
    /// @return (H, W) CV_32SC1 label map (0=background)
    cv::Mat getMasksTorch(const std::vector<int>& rst,
                          const std::vector<int>& foregroundIndices,
                          int H, int W);
    /// Build Particle structs from label map.  Applies fillHoles,
    /// minSize/maxSize, mean-logit, and boundary-policy filters.
    /// Note: when boundaryParticlePolicy == Exclude, the excluded particle's
    /// ID is still consumed (idm->nextId() is called before the boundary check),
    /// so output IDs may be non-contiguous.  Downstream consumers must not
    /// assume contiguous IDs.
    std::vector<Particle> buildParticles(const cv::Mat& labels,
                                         const cv::Mat& cellprob,
                                         ParticleIdManager* idm);

    /// remove_bad_flow_masks 等效: 使用与 Cellpose masks_to_flows_gpu 一致的
    /// 热扩散流场生成, 对每个 mask 计算 flow error, 丢弃 error > m_cfg.flowThreshold 的 mask。
    /// flowThreshold 默认 0.4（与 Cellpose 官方一致），设为 0 可禁用过滤。
    /// @param labels  实例标签图 (会被原地修改: 未通过过滤的 mask → 0)
    /// @param rawDy   原始 dy (未经 /5 和前景遮罩)
    /// @param rawDx   原始 dx
    void removeBadFlowMasks(cv::Mat& labels, const cv::Mat& rawDy,
                            const cv::Mat& rawDx);

    /// 从实例标签图生成流场 (dy, dx)，使用与 Cellpose masks_to_flows_gpu 一致的
    /// 热扩散 PDE 迭代算法: 从 mask 中心扩散 → 扩散场梯度 → L2 归一化。
    /// @param labels  CV_32S 实例标签图 (0=背景, 1..N=实例)
    /// @return (dy, dx) 各为 CV_64FC1, 形状与 labels 相同
    /// @note  按整图 padding 后同步扩散, 与 Cellpose 官方 neighbor 更新语义一致
    std::pair<cv::Mat, cv::Mat> masksToFlows(const cv::Mat& labels);

    Config m_cfg;
    cv::Size m_imageSize;
    cv::Mat m_rawDy;   // 原始 dy (未经缩放), 供 removeBadFlowMasks 使用
    cv::Mat m_rawDx;
};
