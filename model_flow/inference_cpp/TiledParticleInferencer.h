#pragma once

#include "Particle.h"
#include "ParticleIdManager.h"
#include "FlowDynamics.h"
#include "ParticleInferencer.h"
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

/// Per-tile processing record (particles in full-image coordinates after transform).
struct TileRecord {
    int tileX = 0, tileY = 0;   // full-image origin
    int tileH = 0, tileW = 0;   // actual dims (≤ tileSize)
    int coreX0 = 0, coreY0 = 0; // core zone in tile-local coords
    int coreX1 = 0, coreY1 = 0;
    std::vector<Particle> particles; // full-image coords
};

/// Statistics written to tile_merge_report.json.
struct TileMergeStats {
    int totalTiles            = 0;
    int totalDetected         = 0;  // before core/halo filter
    int coreRetained          = 0;  // after filter
    int haloDiscarded         = 0;
    int duplicateMerged       = 0;
    int finalCount            = 0;
    double totalTimeMs        = 0.0;
};

/// Large-image tiled inference with core/halo adoption,
/// cross-tile dedup, and global ID remapping.
///
/// Designed to be reusable in the final particle-analysis software
/// — no Python dependencies, no training-specific logic.
///
/// Usage:
///   TiledParticleInferencer::Config tileCfg;
///   tileCfg.tileSize  = 1024;
///   tileCfg.overlap   = 256;
///   tileCfg.coreMargin = 128;
///
///   TiledParticleInferencer tiler(tileCfg,
///       "backbone.onnx", "neck_head.onnx",
///       inferenceSettings, flowDynamicsCfg);
///
///   auto particles = tiler.process(fullImage, "report.json", &idManager);
///   // particles are in full-image coordinates with unique IDs
///
class TiledParticleInferencer {
public:
    struct Config {
        int tileSize      = 1024;
        int overlap       = 256;
        int coreMargin    = 128;
        int longSideLimit = 1536;   // longer side ≤ this → no tiling
        int padValue      = 114;    // BGR padding value, from flow_inference_config.json
        std::vector<cv::Rect> ignoreRegions; // full main-ROI coordinates
    };

    TiledParticleInferencer(
        const Config& tileConfig,
        const std::string& backbonePath,
        const std::string& headPath,
        const FlowFieldInferenceSettings& inferSettings,
        const FlowDynamics::Config& dynConfig);

    /// Main entry.  If the image does not need tiling (shouldTile() == false),
    /// falls back to single-image inference via the existing pipeline.
    /// When reportJsonPath is non-empty, writes tile_merge_report.json.
    std::vector<Particle> process(
        const cv::Mat& inputImage,
        const std::string& reportJsonPath = "",
        ParticleIdManager* idm = nullptr);

    /// Returns true when the image should be tiled.
    bool shouldTile(const cv::Mat& img) const;

    /// Read-only access to the last run's merge statistics.
    const TileMergeStats& stats() const { return m_stats; }

private:
    struct TileRect { int x0, y0, x1, y1; };

    /// Compute the tile grid (full-image coords) for the given image dimensions.
    std::vector<TileRect> computeTileGrid(int h, int w) const;

    /// Run single-tile inference. Returns particles in padded-tile coordinates.
    std::vector<Particle> inferSingleTile(
        const cv::Mat& tileImg, int actualH, int actualW,
        const std::vector<cv::Rect>& localIgnoreRegions,
        ParticleIdManager* idm);

    /// Transform padded-tile-local particle coordinates to full-image coords.
    void transformToFull(std::vector<Particle>& particles,
                         int tx0, int ty0, int actualH, int actualW) const;

    /// Discard particles whose centroid falls outside the core zone.
    void filterCoreZone(std::vector<Particle>& particles,
                        const TileRect& tileRect,
                        int actualH, int actualW, int& kept, int& discarded);

    /// Merge duplicate particles across tiles (IoU + centroid distance + area).
    std::vector<Particle> mergeCrossTileDuplicates(
        std::vector<Particle>&& allParticles);

    void writeReport(const std::string& path);

    // ── config ──
    Config                    m_tileCfg;
    std::string               m_backbonePath;
    std::string               m_headPath;
    FlowFieldInferenceSettings m_inferCfg;
    FlowDynamics::Config      m_dynCfg;
    std::unique_ptr<FlowFieldInferencer> m_inferencer;

    // ── merge thresholds ──
    double m_iouThresh         = 0.5;
    double m_centroidDistMax   = 50.0;   // px in full-image space
    double m_areaRatioMax      = 2.0;    // max(minA, maxA) / min(minA, maxA)

    TileMergeStats m_stats;
    int m_fullW = 0, m_fullH = 0;
};
