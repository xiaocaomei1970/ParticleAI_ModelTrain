#include "TiledParticleInferencer.h"
#include "FlowDynamics.h"
#include "ParticleInferencer.h"
#include "Particle.h"
#include "ParticleIdManager.h"
#include "json.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

using json = nlohmann::json;

namespace {

cv::Rect clippedRect(const cv::Rect& rect, const cv::Size& bounds)
{
    return rect & cv::Rect(0, 0, bounds.width, bounds.height);
}

cv::Rect mapImageRectToPaddedRect(
    const cv::Rect& imageRect, float scale, int padLeft, int padTop,
    const cv::Size& paddedSize)
{
    int x0 = static_cast<int>(std::floor(imageRect.x * scale + padLeft));
    int y0 = static_cast<int>(std::floor(imageRect.y * scale + padTop));
    int x1 = static_cast<int>(std::ceil((imageRect.x + imageRect.width) * scale + padLeft));
    int y1 = static_cast<int>(std::ceil((imageRect.y + imageRect.height) * scale + padTop));
    return clippedRect(cv::Rect(x0, y0, x1 - x0, y1 - y0), paddedSize);
}

void applyIgnoreRegionsToFlow(
    cv::Mat& dy, cv::Mat& dx, cv::Mat& cellprob,
    const std::vector<cv::Rect>& ignoreRegions,
    float scale, int padLeft, int padTop)
{
    const cv::Size paddedSize = cellprob.size();
    for (const auto& region : ignoreRegions) {
        cv::Rect mapped = mapImageRectToPaddedRect(
            region, scale, padLeft, padTop, paddedSize);
        if (mapped.empty()) continue;
        cellprob(mapped).setTo(-100.0f);
        dy(mapped).setTo(0.0f);
        dx(mapped).setTo(0.0f);
    }
}

bool eraseRegionsFromParticle(Particle& particle,
                              const std::vector<cv::Rect>& ignoreRegions,
                              const cv::Size& imageSize)
{
    if (particle.mask.empty()) return false;

    if (particle.mask.size() != imageSize) {
        cv::resize(particle.mask, particle.mask, imageSize, 0, 0, cv::INTER_NEAREST);
    }

    for (const auto& region : ignoreRegions) {
        cv::Rect clipped = clippedRect(region, imageSize);
        if (!clipped.empty()) {
            particle.mask(clipped).setTo(0.0f);
        }
    }

    cv::Mat binaryMask;
    cv::threshold(particle.mask, binaryMask, 0.5, 255.0, cv::THRESH_BINARY);
    binaryMask.convertTo(binaryMask, CV_8UC1);
    if (cv::countNonZero(binaryMask) == 0) {
        return false;
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return false;
    }

    auto it = std::max_element(contours.begin(), contours.end(),
        [](const auto& a, const auto& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    particle.contour = *it;
    return true;
}

void eraseRegionsFromParticles(std::vector<Particle>& particles,
                               const std::vector<cv::Rect>& ignoreRegions,
                               const cv::Size& imageSize)
{
    if (ignoreRegions.empty()) return;
    std::vector<Particle> retained;
    retained.reserve(particles.size());
    for (auto& particle : particles) {
        if (eraseRegionsFromParticle(particle, ignoreRegions, imageSize)) {
            retained.push_back(std::move(particle));
        }
    }
    particles = std::move(retained);
}

std::vector<cv::Rect> intersectIgnoreRegionsWithTile(
    const std::vector<cv::Rect>& ignoreRegions,
    const cv::Rect& tileRect)
{
    std::vector<cv::Rect> localRegions;
    for (const auto& region : ignoreRegions) {
        cv::Rect intersection = region & tileRect;
        if (intersection.empty()) continue;
        intersection.x -= tileRect.x;
        intersection.y -= tileRect.y;
        localRegions.push_back(intersection);
    }
    return localRegions;
}

} // namespace

// ═══════════════════════════════════════════════════════════════
// Constructor
// ═══════════════════════════════════════════════════════════════

TiledParticleInferencer::TiledParticleInferencer(
    const Config& tileConfig,
    const std::string& backbonePath,
    const std::string& headPath,
    const FlowFieldInferenceSettings& inferSettings,
    const FlowDynamics::Config& dynConfig)
    : m_tileCfg(tileConfig)
    , m_backbonePath(backbonePath)
    , m_headPath(headPath)
    , m_inferCfg(inferSettings)
    , m_dynCfg(dynConfig)
{
    m_tileCfg.padValue = inferSettings.padValue;
}

// ═══════════════════════════════════════════════════════════════
// Tile-or-not decision
// ═══════════════════════════════════════════════════════════════

bool TiledParticleInferencer::shouldTile(const cv::Mat& img) const
{
    int maxSide = std::max(img.rows, img.cols);
    return maxSide > m_tileCfg.longSideLimit;
}

// ═══════════════════════════════════════════════════════════════
// Tile grid computation
// ═══════════════════════════════════════════════════════════════

std::vector<TiledParticleInferencer::TileRect>
TiledParticleInferencer::computeTileGrid(int h, int w) const
{
    std::vector<TileRect> grid;
    int step = m_tileCfg.tileSize - m_tileCfg.overlap;
    if (m_tileCfg.tileSize <= 0 || step <= 0)
        return grid;

    std::vector<int> yStarts;
    std::vector<int> xStarts;
    for (int y = 0; y < h; y += step)
        yStarts.push_back(std::min(y, std::max(0, h - m_tileCfg.tileSize)));
    for (int x = 0; x < w; x += step)
        xStarts.push_back(std::min(x, std::max(0, w - m_tileCfg.tileSize)));

    if (yStarts.empty()) yStarts.push_back(0);
    if (xStarts.empty()) xStarts.push_back(0);
    yStarts.push_back(std::max(0, h - m_tileCfg.tileSize));
    xStarts.push_back(std::max(0, w - m_tileCfg.tileSize));
    std::sort(yStarts.begin(), yStarts.end());
    std::sort(xStarts.begin(), xStarts.end());
    // Defensive de-duplication for small images and exact boundary-aligned sizes.
    yStarts.erase(std::unique(yStarts.begin(), yStarts.end()), yStarts.end());
    xStarts.erase(std::unique(xStarts.begin(), xStarts.end()), xStarts.end());

    for (int yStart : yStarts) {
        for (int xStart : xStarts) {
            int yEnd = std::min(yStart + m_tileCfg.tileSize, h);
            int xEnd = std::min(xStart + m_tileCfg.tileSize, w);
            grid.push_back({xStart, yStart, xEnd, yEnd});
        }
    }
    return grid;
}

// ═══════════════════════════════════════════════════════════════
// Single-tile inference (returns padded-tile coords)
// ═══════════════════════════════════════════════════════════════

std::vector<Particle>
TiledParticleInferencer::inferSingleTile(
    const cv::Mat& tileImg, int actualH, int actualW,
    const std::vector<cv::Rect>& localIgnoreRegions,
    ParticleIdManager* idm)
{
    // Reuse ONNX Runtime sessions across tiles; preprocessing state is updated per call.
    if (!m_inferencer) {
        m_inferencer = std::make_unique<FlowFieldInferencer>(
            m_backbonePath, m_headPath, m_inferCfg);
    }
    cv::Mat dyS4, dxS4, cpS4;
    m_inferencer->infer(tileImg, dyS4, dxS4, cpS4);

    int fullSz = m_inferencer->inputSize();
    cv::Mat dyFull, dxFull, cpFull;
    cv::resize(dyS4, dyFull, cv::Size(fullSz, fullSz), 0, 0, cv::INTER_LINEAR);
    cv::resize(dxS4, dxFull, cv::Size(fullSz, fullSz), 0, 0, cv::INTER_LINEAR);
    cv::resize(cpS4, cpFull, cv::Size(fullSz, fullSz), 0, 0, cv::INTER_LINEAR);

    // pad mask: zero-out padding area
    float scale = m_inferencer->getScale();
    int newH = static_cast<int>(actualH * scale);
    int newW = static_cast<int>(actualW * scale);
    int padTop = static_cast<int>(m_inferencer->getPadTop());
    int padLeft = static_cast<int>(m_inferencer->getPadLeft());

    if (padTop > 0)
        cpFull(cv::Rect(0, 0, fullSz, padTop)) = -100.0f;
    if (fullSz - newH - padTop > 0)
        cpFull(cv::Rect(0, fullSz - (fullSz - newH - padTop),
                        fullSz, fullSz - newH - padTop)) = -100.0f;
    if (padLeft > 0)
        cpFull(cv::Rect(0, padTop, padLeft, newH)) = -100.0f;
    if (fullSz - newW - padLeft > 0)
        cpFull(cv::Rect(fullSz - (fullSz - newW - padLeft), padTop,
                        fullSz - newW - padLeft, newH)) = -100.0f;

    applyIgnoreRegionsToFlow(dyFull, dxFull, cpFull,
                             localIgnoreRegions, scale, padLeft, padTop);

    FlowDynamics dynamics(m_dynCfg);
    auto particles = dynamics.computeMasks(dyFull, dxFull, cpFull, idm);

    cv::Rect contentRoi(padLeft, padTop, newW, newH);
    contentRoi &= cv::Rect(0, 0, fullSz, fullSz);
    for (auto& particle : particles) {
        for (auto& point : particle.contour) {
            point.x = std::clamp(
                static_cast<int>(std::round((point.x - padLeft) / scale)),
                0, std::max(0, actualW - 1));
            point.y = std::clamp(
                static_cast<int>(std::round((point.y - padTop) / scale)),
                0, std::max(0, actualH - 1));
        }
        if (!particle.mask.empty() && contentRoi.width > 0 && contentRoi.height > 0) {
            cv::Mat cropped = particle.mask(contentRoi);
            cv::Mat tileMask;
            cv::resize(cropped, tileMask, cv::Size(actualW, actualH),
                       0, 0, cv::INTER_NEAREST);
            particle.mask = tileMask;
        }
    }
    eraseRegionsFromParticles(
        particles, localIgnoreRegions, cv::Size(actualW, actualH));
    return particles;
}

// ═══════════════════════════════════════════════════════════════
// Coordinate transform: padded-tile-local → full-image
// ═══════════════════════════════════════════════════════════════

void TiledParticleInferencer::transformToFull(
    std::vector<Particle>& particles,
    int tx0, int ty0, int actualH, int actualW) const
{
    for (auto& p : particles) {
        for (auto& pt : p.contour) {
            pt.x = pt.x + tx0;
            pt.y = pt.y + ty0;
        }
        // transform mask as well
        if (!p.mask.empty()) {
            cv::Mat fullMask = cv::Mat::zeros(m_fullH, m_fullW, CV_32FC1);
            cv::Mat tileMask = p.mask;
            if (tileMask.size() != cv::Size(actualW, actualH)) {
                cv::resize(tileMask, tileMask, cv::Size(actualW, actualH),
                           0, 0, cv::INTER_NEAREST);
            }
            tileMask.copyTo(fullMask(cv::Rect(tx0, ty0, actualW, actualH)));
            p.mask = fullMask;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Core/halo filter
// ═══════════════════════════════════════════════════════════════

void TiledParticleInferencer::filterCoreZone(
    std::vector<Particle>& particles,
    const TileRect& tileRect,
    int actualH, int actualW, int& kept, int& discarded)
{
    int cx0 = tileRect.x0 <= 0 ? 0 : m_tileCfg.coreMargin;
    int cy0 = tileRect.y0 <= 0 ? 0 : m_tileCfg.coreMargin;
    int cx1 = tileRect.x1 >= m_fullW ? actualW : actualW - m_tileCfg.coreMargin;
    int cy1 = tileRect.y1 >= m_fullH ? actualH : actualH - m_tileCfg.coreMargin;

    // if coreMargin is too large for a small edge tile, default everything to core
    if (cx1 <= cx0) { cx0 = 0; cx1 = actualW; }
    if (cy1 <= cy0) { cy0 = 0; cy1 = actualH; }

    std::vector<Particle> coreOnly;
    for (auto& p : particles) {
        // compute centroid from contour
        if (p.contour.empty()) {
            coreOnly.push_back(std::move(p));
            ++kept;
            continue;
        }
        double sx = 0, sy = 0;
        for (const auto& pt : p.contour) {
            sx += pt.x;
            sy += pt.y;
        }
        double cx = sx / p.contour.size();
        double cy = sy / p.contour.size();

        if (cx >= cx0 && cx < cx1 && cy >= cy0 && cy < cy1) {
            coreOnly.push_back(std::move(p));
            ++kept;
        } else {
            ++discarded;
        }
    }
    particles = std::move(coreOnly);
}

// ═══════════════════════════════════════════════════════════════
// Cross-tile duplicate merging
// ═══════════════════════════════════════════════════════════════

static double computeMaskIoU(const cv::Mat& a, const cv::Mat& b)
{
    if (a.empty() || b.empty()) return 0.0;
    cv::Mat inter, un;
    cv::bitwise_and(a, b, inter);
    cv::bitwise_or(a, b, un);
    double sInter = cv::countNonZero(inter);
    double sUn = cv::countNonZero(un);
    return sUn > 0 ? sInter / sUn : 0.0;
}

static double centroidDist(const Particle& a, const Particle& b)
{
    if (a.contour.empty() || b.contour.empty()) return 1e9;
    double ax = 0, ay = 0, bx = 0, by = 0;
    for (const auto& pt : a.contour) { ax += pt.x; ay += pt.y; }
    for (const auto& pt : b.contour) { bx += pt.x; by += pt.y; }
    ax /= a.contour.size(); ay /= a.contour.size();
    bx /= b.contour.size(); by /= b.contour.size();
    return std::sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
}

static bool sameOrAdjacentTile(const Particle& a, const Particle& b)
{
    if (a.tileOriginX < 0 || a.tileOriginY < 0) return false;
    if (b.tileOriginX < 0 || b.tileOriginY < 0) return false;
    int dx = std::abs(a.tileOriginX - b.tileOriginX);
    int dy = std::abs(a.tileOriginY - b.tileOriginY);
    // adjacent tiles in x, y, or diagonal
    return dx <= 1024 + 256 && dy <= 1024 + 256;  // tileSize + overlap as upper bound
}

static bool bboxNearOverlapBoundary(const Particle& p, const cv::Rect& overlapRect)
{
    if (p.contour.empty()) return false;
    cv::Rect bbox = cv::boundingRect(p.contour);
    // Check if bbox touches or crosses the tile overlap boundary region
    bool nearX = (bbox.x <= overlapRect.x + overlapRect.width)
              && (bbox.x + bbox.width >= overlapRect.x);
    bool nearY = (bbox.y <= overlapRect.y + overlapRect.height)
              && (bbox.y + bbox.height >= overlapRect.y);
    return nearX || nearY;
}

static bool isSameParticle(const Particle& a, const Particle& b,
                           double iouThresh, double distThresh, double areaMax,
                           int tileSize = 1024, int overlap = 256)
{
    double iou = computeMaskIoU(a.mask, b.mask);
    if (iou > iouThresh) return true;

    // Low IoU fallback: masks overlap partially but IoU is low.
    // This happens when a particle is split across tile boundaries.
    // Only trigger if particles come from adjacent/overlapping tiles
    // and at least one touches the tile overlap boundary.
    if (iou > 0 && iou <= iouThresh) {
        if (!sameOrAdjacentTile(a, b)) return false;
        // Check proximity to overlap boundary
        int margin = tileSize - overlap;  // core margin
        cv::Rect overlapX(0, 0, 0, 0);
        if (a.tileOriginX != b.tileOriginX) {
            int ox = std::max(a.tileOriginX, b.tileOriginX);
            overlapX = cv::Rect(ox, 0, overlap, std::max(
                a.tileOriginY + tileSize, b.tileOriginY + tileSize));
        }
        cv::Rect overlapY(0, 0, 0, 0);
        if (a.tileOriginY != b.tileOriginY) {
            int oy = std::max(a.tileOriginY, b.tileOriginY);
            overlapY = cv::Rect(0, oy, std::max(
                a.tileOriginX + tileSize, b.tileOriginX + tileSize), overlap);
        }
        bool nearBoundary = false;
        if (overlapX.width > 0) nearBoundary = bboxNearOverlapBoundary(a, overlapX)
                                            || bboxNearOverlapBoundary(b, overlapX);
        if (!nearBoundary && overlapY.height > 0)
            nearBoundary = bboxNearOverlapBoundary(a, overlapY)
                        || bboxNearOverlapBoundary(b, overlapY);
        if (!nearBoundary) return false;

        // Fallback: centroid distance + area ratio
        double dist = centroidDist(a, b);
        if (dist > distThresh) return false;
        double aArea = cv::countNonZero(a.mask);
        double bArea = cv::countNonZero(b.mask);
        if (aArea <= 0 || bArea <= 0) return false;
        double ratio = std::max(aArea, bArea) / std::min(aArea, bArea);
        return ratio <= areaMax;
    }

    // IoU == 0: masks don't overlap. Use centroid distance + area ratio.
    double dist = centroidDist(a, b);
    if (dist > distThresh) return false;
    double aArea = cv::countNonZero(a.mask);
    double bArea = cv::countNonZero(b.mask);
    if (aArea <= 0 || bArea <= 0) return false;
    double ratio = std::max(aArea, bArea) / std::min(aArea, bArea);
    return ratio <= areaMax;
}

std::vector<Particle>
TiledParticleInferencer::mergeCrossTileDuplicates(
    std::vector<Particle>&& allParticles)
{
    int n = static_cast<int>(allParticles.size());
    std::vector<bool> merged(n, false);
    std::vector<Particle> result;

    for (int i = 0; i < n; ++i) {
        if (merged[i]) continue;
        Particle keeper = std::move(allParticles[i]);
        merged[i] = true;

        for (int j = i + 1; j < n; ++j) {
            if (merged[j]) continue;
            if (isSameParticle(keeper, allParticles[j],
                               m_iouThresh, m_centroidDistMax, m_areaRatioMax,
                               m_tileCfg.tileSize, m_tileCfg.overlap)) {
                // keep the higher-confidence one
                if (allParticles[j].confidence > keeper.confidence) {
                    keeper = std::move(allParticles[j]);
                    merged[i] = false; // old keeper was overridden
                }
                merged[j] = true;
                ++m_stats.duplicateMerged;
            }
        }
        result.push_back(std::move(keeper));
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════
// Report
// ═══════════════════════════════════════════════════════════════

void TiledParticleInferencer::writeReport(const std::string& path)
{
    json report;
    report["total_tiles"]          = m_stats.totalTiles;
    report["total_detected"]       = m_stats.totalDetected;
    report["core_retained"]        = m_stats.coreRetained;
    report["halo_discarded"]       = m_stats.haloDiscarded;
    report["duplicates_merged"]    = m_stats.duplicateMerged;
    report["final_particle_count"] = m_stats.finalCount;
    report["total_time_ms"]        = m_stats.totalTimeMs;

    std::ofstream ofs(path);
    ofs << report.dump(2) << std::endl;
}

// ═══════════════════════════════════════════════════════════════
// Main entry
// ═══════════════════════════════════════════════════════════════

std::vector<Particle>
TiledParticleInferencer::process(
    const cv::Mat& inputImage,
    const std::string& reportJsonPath,
    ParticleIdManager* idm)
{
    m_fullH = inputImage.rows;
    m_fullW = inputImage.cols;
    m_stats = {};

    auto t0 = std::chrono::high_resolution_clock::now();

    // ── fallback: small image, single-pass ──
    if (!shouldTile(inputImage)) {
        cv::Mat img = inputImage.clone();
        auto particles = inferSingleTile(
            img, m_fullH, m_fullW, m_tileCfg.ignoreRegions, idm);
        eraseRegionsFromParticles(
            particles, m_tileCfg.ignoreRegions, cv::Size(m_fullW, m_fullH));
        return particles;
    }

    // ── tiled path ──
    auto grid = computeTileGrid(m_fullH, m_fullW);
    m_stats.totalTiles = static_cast<int>(grid.size());

    std::vector<Particle> allParticles;
    for (const auto& rect : grid) {
        int tw = rect.x1 - rect.x0;
        int th = rect.y1 - rect.y0;

        // crop tile
        cv::Mat tileCrop = inputImage(cv::Rect(rect.x0, rect.y0, tw, th));
        // pad to tileSize
        int padRight  = m_tileCfg.tileSize - tw;
        int padBottom = m_tileCfg.tileSize - th;
        cv::Mat tileImg;
        cv::copyMakeBorder(tileCrop, tileImg,
                           0, std::max(padBottom, 0),
                           0, std::max(padRight, 0),
                           cv::BORDER_CONSTANT,
                           cv::Scalar(m_tileCfg.padValue, m_tileCfg.padValue,
                                      m_tileCfg.padValue));

        // inference
        // Per-tile IDs are temporary; assign stable global IDs after merge.
        std::vector<cv::Rect> tileIgnoreRegions = intersectIgnoreRegionsWithTile(
            m_tileCfg.ignoreRegions, cv::Rect(rect.x0, rect.y0, tw, th));
        auto tileParts = inferSingleTile(tileImg, th, tw, tileIgnoreRegions, nullptr);
        m_stats.totalDetected += static_cast<int>(tileParts.size());

        // core/halo in tile-local coordinates, before transforming to full image
        int kept = 0, discarded = 0;
        filterCoreZone(tileParts, rect, th, tw, kept, discarded);
        m_stats.coreRetained += kept;
        m_stats.haloDiscarded += discarded;

        // transform to full-image coords
        transformToFull(tileParts, rect.x0, rect.y0, th, tw);

        // collect — stamp tile origin for cross-tile dedup
        for (auto& p : tileParts) {
            p.tileOriginX = rect.x0;
            p.tileOriginY = rect.y0;
            allParticles.push_back(std::move(p));
        }
    }

    // cross-tile dedup
    allParticles = mergeCrossTileDuplicates(std::move(allParticles));
    eraseRegionsFromParticles(
        allParticles, m_tileCfg.ignoreRegions, cv::Size(m_fullW, m_fullH));

    // global ID remapping
    for (size_t i = 0; i < allParticles.size(); ++i)
        allParticles[i].id = idm ? idm->nextId() : static_cast<int>(i) + 1;

    m_stats.finalCount = static_cast<int>(allParticles.size());

    auto t1 = std::chrono::high_resolution_clock::now();
    m_stats.totalTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // report
    if (!reportJsonPath.empty())
        writeReport(reportJsonPath);

    return allParticles;
}
