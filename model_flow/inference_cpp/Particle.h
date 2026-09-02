#pragma once
#include <map>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

enum class EditSource { Auto, ManualAdd, Split, Merge };

struct Particle {
    // ── 身份信息（由 ISegmenter / ParticleEditor 填充）──
    int id = 0;
    int imageIndex = 0;                  // 来源图片序号（分析记录内唯一）
    cv::Mat mask;                        // 二值掩膜，原图分辨率
    std::vector<cv::Point> contour;      // 最大外轮廓
    float confidence = 0.0f;             // 模型置信度
    bool deleted = false;                // 软删除标记（编辑操作不复用 ID）
    EditSource source = EditSource::Auto;
    std::vector<int> parentIds;          // split/merge 时的来源颗粒 ID

    // ── tile provenance（用于跨 tile 去重）──
    int tileOriginX = -1;                // 来源 tile 在整图上的左上角 x, -1=未知
    int tileOriginY = -1;                // 来源 tile 在整图上的左上角 y, -1=未知

    // ── 表征参数（由 ParticleCharacterizer 填充）──
    std::map<std::string, float> metrics;

    /// 获取指定 key 的 metric 值，key 不存在时返回 0
    float metric(const std::string& key) const {
        auto it = metrics.find(key);
        return it != metrics.end() ? it->second : 0.0f;
    }
};

/// Standard metric key constants.
namespace ParticleMetrics {
    constexpr auto areaPx          = "areaPx";
    constexpr auto areaNm2         = "areaNm2";
    constexpr auto equivDiameterNm = "equivDiameterNm";
    constexpr auto circularity     = "circularity";
    constexpr auto perimeterPx     = "perimeterPx";
    constexpr auto perimeterNm     = "perimeterNm";
    constexpr auto centroidX       = "centroidX";
    constexpr auto centroidY       = "centroidY";
    constexpr auto centroidXNm     = "centroidXNm";
    constexpr auto centroidYNm     = "centroidYNm";
    constexpr auto bboxX           = "bboxX";
    constexpr auto bboxY           = "bboxY";
    constexpr auto bboxW           = "bboxW";
    constexpr auto bboxH           = "bboxH";
    constexpr auto bboxXNm         = "bboxXNm";
    constexpr auto bboxYNm         = "bboxYNm";
    constexpr auto bboxWNm         = "bboxWNm";
    constexpr auto bboxHNm         = "bboxHNm";
    constexpr auto minFeretNm      = "minFeretNm";
    constexpr auto maxFeretNm      = "maxFeretNm";
    constexpr auto aspectRatio     = "aspectRatio";
    constexpr auto solidity        = "solidity";
    constexpr auto convexHullArea  = "convexHullArea";
    constexpr auto isEdgeParticle  = "isEdgeParticle";
}
