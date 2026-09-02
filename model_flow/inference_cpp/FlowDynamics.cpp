#include "FlowDynamics.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <vector>
#include <tuple>

#ifndef CV_PI
#define CV_PI 3.14159265358979323846
#endif

FlowDynamics::FlowDynamics(const Config& cfg) : m_cfg(cfg) {}

// ══════════════════════════════════════════════════════════════════════════════
// 2D Max Pool helper (stride=1, same-size output, zero-padded boundary)
// ══════════════════════════════════════════════════════════════════════════════

static cv::Mat maxPool2D(const cv::Mat& src, int kernelSize)
{
    int H = src.rows, W = src.cols;
    int pad = kernelSize / 2;
    cv::Mat dst(H, W, CV_64FC1, cv::Scalar(0));
    cv::parallel_for_(cv::Range(0, H), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            const double* srcRow = src.ptr<double>(y);
            double* dstRow = dst.ptr<double>(y);
            for (int x = 0; x < W; ++x) {
                double maxVal = 0.0;
                for (int dy = -pad; dy <= pad; ++dy) {
                    int sy = y + dy;
                    if (sy < 0 || sy >= H) continue;
                    const double* sRow = src.ptr<double>(sy);
                    for (int dx = -pad; dx <= pad; ++dx) {
                        int sx = x + dx;
                        if (sx < 0 || sx >= W) continue;
                        maxVal = std::max(maxVal, sRow[sx]);
                    }
                }
                dstRow[x] = maxVal;
            }
        }
    });
    return dst;
}

// ══════════════════════════════════════════════════════════════════════════════
// Main entry — Euler integration + get_masks_torch clustering
// ══════════════════════════════════════════════════════════════════════════════

std::vector<Particle> FlowDynamics::computeMasks(
    const cv::Mat& dy, const cv::Mat& dx, const cv::Mat& cellprob,
    ParticleIdManager* idm)
{
    if (dy.empty() || dx.empty() || cellprob.empty())
        return {};

    m_imageSize = dy.size();
    int H = m_imageSize.height;
    int W = m_imageSize.width;

    CV_Assert(dy.size() == dx.size() && dx.size() == cellprob.size());

    m_rawDy = dy.clone();
    m_rawDx = dx.clone();

    // ── Cellpose normalization (pixel/(N-1)*2-1, flow*2/(N-1)) ──
    float invHm1 = 1.0f / (H - 1);
    float invWm1 = 1.0f / (W - 1);

    cv::Mat dP[2] = {dy, dx};
    cv::Mat dPMerged;
    cv::merge(dP, 2, dPMerged);

    std::vector<int> foregroundIndices;
    foregroundIndices.reserve(static_cast<size_t>(H) * W / 4);

    for (int y = 0; y < H; ++y) {
        const float* probRow = cellprob.ptr<float>(y);
        float* dPRow = dPMerged.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            bool isForeground = probRow[x] > 0.0f;
            float mask = isForeground ? (1.0f / 5.0f) : 0.0f;
            dPRow[x * 2 + 0] *= mask;
            dPRow[x * 2 + 1] *= mask;
            if (isForeground)
                foregroundIndices.push_back(y * W + x);
        }
    }

    if (foregroundIndices.empty())
        return {};

    // Normalize to [-1, 1]: pt = pixel / (N-1) * 2 - 1  (Cellpose exact)
    int foregroundCount = static_cast<int>(foregroundIndices.size());
    cv::Mat pts(foregroundCount, 2, CV_32FC1);
    for (int i = 0; i < foregroundCount; ++i) {
        int index = foregroundIndices[i];
        int y = index / W;
        int x = index % W;
        pts.at<float>(i, 0) = static_cast<float>(y) * invHm1 * 2.0f - 1.0f;
        pts.at<float>(i, 1) = static_cast<float>(x) * invWm1 * 2.0f - 1.0f;
    }

    // Euler integration
    cv::Mat finalPts = stepsInterp(dPMerged, pts);

    // Denormalize → pixel indices: (pt+1)/2*(N-1) + truncation (Cellpose exact)
    std::vector<int> rst(foregroundCount);
    for (int i = 0; i < foregroundCount; ++i) {
        float fy = (finalPts.at<float>(i, 0) + 1.0f) * 0.5f * (H - 1);
        float fx = (finalPts.at<float>(i, 1) + 1.0f) * 0.5f * (W - 1);
        int yi = std::clamp(static_cast<int>(fy), 0, H - 1);
        int xi = std::clamp(static_cast<int>(fx), 0, W - 1);
        rst[i] = yi * W + xi;
    }

    // ── Official Cellpose get_masks_torch clustering ──
    cv::Mat labels = getMasksTorch(rst, foregroundIndices, H, W);

    // ── max_size_fraction removal (Cellpose: after projection to foreground) ──
    if (m_cfg.maxSizeFraction > 0.0f) {
        double totalPx = static_cast<double>(H) * W;
        double maxPx = totalPx * m_cfg.maxSizeFraction;
        double dmin, dmax;
        cv::minMaxLoc(labels, &dmin, &dmax);
        int nLbl = static_cast<int>(dmax);
        for (int lbl = 1; lbl <= nLbl; ++lbl) {
            if (cv::countNonZero(labels == lbl) > maxPx)
                labels.setTo(0, labels == lbl);
        }
    }

    // ── remove_bad_flow_masks ──
    if (m_cfg.flowThreshold > 0.0f) {
        removeBadFlowMasks(labels, m_rawDy, m_rawDx);
    }

    return buildParticles(labels, cellprob, idm);
}

// ══════════════════════════════════════════════════════════════════════════════
// Euler integration (align_corners=False, matches Cellpose grid_sample)
// ══════════════════════════════════════════════════════════════════════════════

cv::Mat FlowDynamics::stepsInterp(const cv::Mat& dP, const cv::Mat& pts) const
{
    int H = dP.rows, W = dP.cols;
    int N = pts.rows;

    // Cellpose flow scaling: dP *= 2/(N-1)
    float scaleY = 2.0f / (H - 1);
    float scaleX = 2.0f / (W - 1);

    cv::Mat dP_norm(H, W, CV_32FC2);
    for (int y = 0; y < H; ++y) {
        const float* src = dP.ptr<float>(y);
        float* dst = dP_norm.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            dst[x * 2 + 0] = src[x * 2 + 0] * scaleY;
            dst[x * 2 + 1] = src[x * 2 + 1] * scaleX;
        }
    }

    cv::Mat pt = pts.clone();
    cv::Mat ptNext(N, 2, CV_32FC1);

    for (int iter = 0; iter < m_cfg.niter; ++iter) {
        cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; ++i) {
                float y = pt.at<float>(i, 0);
                float x = pt.at<float>(i, 1);

                // grid_sample(align_corners=False): pixel=(coord+1)*0.5*N - 0.5 (unclamped)
                float imgY = (y + 1.0f) * 0.5f * H - 0.5f;
                float imgX = (x + 1.0f) * 0.5f * W - 0.5f;

                int y0 = static_cast<int>(std::floor(imgY));
                int x0 = static_cast<int>(std::floor(imgX));
                int y1 = y0 + 1;
                int x1 = x0 + 1;

                float wy = imgY - static_cast<float>(y0);
                float wx = imgX - static_cast<float>(x0);

                // per-neighbor validity (zero padding)
                bool v00 = (y0 >= 0 && y0 < H && x0 >= 0 && x0 < W);
                bool v01 = (y0 >= 0 && y0 < H && x1 >= 0 && x1 < W);
                bool v10 = (y1 >= 0 && y1 < H && x0 >= 0 && x0 < W);
                bool v11 = (y1 >= 0 && y1 < H && x1 >= 0 && x1 < W);

                // clamped indices for safe memory access only
                int y0c = std::clamp(y0, 0, H - 1), y1c = std::clamp(y1, 0, H - 1);
                int x0c = std::clamp(x0, 0, W - 1), x1c = std::clamp(x1, 0, W - 1);

                const float* row0c = dP_norm.ptr<float>(y0c);
                const float* row1c = dP_norm.ptr<float>(y1c);

                float dy_interp = 0.0f, dx_interp = 0.0f;
                if (v00) { float w = (1 - wy) * (1 - wx); dy_interp += w * row0c[x0c * 2 + 0]; dx_interp += w * row0c[x0c * 2 + 1]; }
                if (v01) { float w = (1 - wy) * wx;        dy_interp += w * row0c[x1c * 2 + 0]; dx_interp += w * row0c[x1c * 2 + 1]; }
                if (v10) { float w = wy * (1 - wx);        dy_interp += w * row1c[x0c * 2 + 0]; dx_interp += w * row1c[x0c * 2 + 1]; }
                if (v11) { float w = wy * wx;              dy_interp += w * row1c[x1c * 2 + 0]; dx_interp += w * row1c[x1c * 2 + 1]; }

                ptNext.at<float>(i, 0) = std::clamp(y + dy_interp, -1.0f, 1.0f);
                ptNext.at<float>(i, 1) = std::clamp(x + dx_interp, -1.0f, 1.0f);
            }
        });
        cv::swap(pt, ptNext);
    }

    return pt;
}

// ══════════════════════════════════════════════════════════════════════════════
// Official Cellpose get_masks_torch clustering
// ══════════════════════════════════════════════════════════════════════════════

cv::Mat FlowDynamics::getMasksTorch(
    const std::vector<int>& rst,
    const std::vector<int>& foregroundIndices,
    int H, int W)
{
    constexpr int RPAD = 20;
    int H_pad = H + 2 * RPAD;
    int W_pad = W + 2 * RPAD;
    int N = static_cast<int>(rst.size());

    if (N == 0)
        return cv::Mat::zeros(H, W, CV_32SC1);

    // ── Build padded histogram ──
    cv::Mat hist(H_pad, W_pad, CV_64FC1, cv::Scalar(0));
    for (int i = 0; i < N; ++i) {
        int lin = rst[i];
        int y_conv = lin / W + RPAD;
        int x_conv = lin % W + RPAD;
        if (y_conv >= 0 && y_conv < H_pad && x_conv >= 0 && x_conv < W_pad)
            hist.at<double>(y_conv, x_conv) += 1.0;
    }

    // ── 5×5 max pool → seed detection ──
    cv::Mat hMax = maxPool2D(hist, 5);

    // Seeds: local maxima with height > 10
    std::vector<std::tuple<double, int, int>> seedPeaks;
    for (int y = 0; y < H_pad; ++y) {
        const double* hRow = hist.ptr<double>(y);
        const double* mxRow = hMax.ptr<double>(y);
        for (int x = 0; x < W_pad; ++x) {
            if (std::abs(hRow[x] - mxRow[x]) < 1e-10 && hRow[x] > 10.0) {
                seedPeaks.emplace_back(hRow[x], y, x);
            }
        }
    }

    if (seedPeaks.empty())
        return cv::Mat::zeros(H, W, CV_32SC1);

    // ── Per-pixel seeds, sort by peak height ascending ──
    int nSeeds = static_cast<int>(seedPeaks.size());

    std::vector<int> order(nSeeds);
    for (int i = 0; i < nSeeds; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return std::get<0>(seedPeaks[a]) < std::get<0>(seedPeaks[b]);
    });

    // ── Per-pixel seed dilation (5 rounds, 3×3 kernel, h_slc > 2) ──
    cv::Mat mask(H_pad, W_pad, CV_32SC1, cv::Scalar(0));

    for (int rank = 0; rank < nSeeds; ++rank) {
        int idx = order[rank];
        int label = rank + 1;  // larger label = larger peak, wins in amax
        int cy = std::get<1>(seedPeaks[idx]);
        int cx = std::get<2>(seedPeaks[idx]);

        int y0 = std::max(0, cy - 5);
        int y1 = std::min(H_pad, cy + 6);
        int x0 = std::max(0, cx - 5);
        int x1 = std::min(W_pad, cx + 6);
        int winH = y1 - y0;
        int winW = x1 - x0;
        if (winH <= 0 || winW <= 0) continue;

        // h_slc: local histogram
        cv::Mat hSlc(winH, winW, CV_64FC1);
        for (int wy = 0; wy < winH; ++wy) {
            const double* src = hist.ptr<double>(y0 + wy);
            double* dst = hSlc.ptr<double>(wy);
            for (int wx = 0; wx < winW; ++wx)
                dst[wx] = src[x0 + wx];
        }

        // seed mask with 1 at center
        cv::Mat seedMask(winH, winW, CV_64FC1, cv::Scalar(0));
        int relCy = cy - y0;
        int relCx = cx - x0;
        if (relCy >= 0 && relCy < winH && relCx >= 0 && relCx < winW)
            seedMask.at<double>(relCy, relCx) = 1.0;

        for (int it = 0; it < 5; ++it) {
            seedMask = maxPool2D(seedMask, 3);
            for (int wy = 0; wy < winH; ++wy) {
                double* smRow = seedMask.ptr<double>(wy);
                const double* hsRow = hSlc.ptr<double>(wy);
                for (int wx = 0; wx < winW; ++wx)
                    smRow[wx] *= (hsRow[wx] > 2.0) ? 1.0 : 0.0;
            }
        }

        // Scatter amax into global mask
        for (int wy = 0; wy < winH; ++wy) {
            const double* smRow = seedMask.ptr<double>(wy);
            int* maskRow = mask.ptr<int>(y0 + wy);
            for (int wx = 0; wx < winW; ++wx) {
                if (smRow[wx] > 0) {
                    int& cell = maskRow[x0 + wx];
                    cell = std::max(cell, label);
                }
            }
        }
    }

    // ── Crop to original size ──
    cv::Mat labels(H, W, CV_32SC1, cv::Scalar(0));
    for (int y = 0; y < H; ++y) {
        const int* src = mask.ptr<int>(y + RPAD);
        int* dst = labels.ptr<int>(y);
        for (int x = 0; x < W; ++x)
            dst[x] = src[x + RPAD];
    }

    // ── Renumber sequentially ──
    {
        double dmin2, dmax2;
        cv::minMaxLoc(labels, &dmin2, &dmax2);
        int maxLbl = static_cast<int>(dmax2);
        if (maxLbl > 0) {
            std::vector<int> lut(maxLbl + 1, 0);
            int newId = 1;
            for (int y = 0; y < H; ++y) {
                const int* row = labels.ptr<int>(y);
                for (int x = 0; x < W; ++x) {
                    int lbl = row[x];
                    if (lbl > 0 && lbl <= maxLbl && lut[lbl] == 0)
                        lut[lbl] = newId++;
                }
            }
            cv::Mat renumbered(H, W, CV_32SC1, cv::Scalar(0));
            for (int y = 0; y < H; ++y) {
                const int* src = labels.ptr<int>(y);
                int* dst = renumbered.ptr<int>(y);
                for (int x = 0; x < W; ++x)
                    dst[x] = lut[src[x]];
            }
            labels = renumbered;
        }
    }

    // ── Assign labels back to original foreground pixels ──
    cv::Mat result(H, W, CV_32SC1, cv::Scalar(0));
    for (int i = 0; i < N; ++i) {
        int srcIdx = foregroundIndices[i];
        int sinkIdx = rst[i];
        int srcY = srcIdx / W;
        int srcX = srcIdx % W;
        int sinkY = sinkIdx / W;
        int sinkX = sinkIdx % W;
        result.at<int>(srcY, srcX) = labels.at<int>(sinkY, sinkX);
    }

    return result;
}

// ══════════════════════════════════════════════════════════════════════════════
// masksToFlows: Cellpose masks_to_flows_gpu equivalent
// ══════════════════════════════════════════════════════════════════════════════

std::pair<cv::Mat, cv::Mat> FlowDynamics::masksToFlows(const cv::Mat& labels)
{
    int H = labels.rows, W = labels.cols;
    cv::Mat dy_all = cv::Mat::zeros(H, W, CV_64FC1);
    cv::Mat dx_all = cv::Mat::zeros(H, W, CV_64FC1);

    double dmin, dmax;
    cv::minMaxLoc(labels, &dmin, &dmax);
    int nLabels = static_cast<int>(dmax);
    if (nLabels <= 0) return {dy_all, dx_all};

    struct LabelStats {
        int area = 0;
        int minY = std::numeric_limits<int>::max();
        int minX = std::numeric_limits<int>::max();
        int maxY = std::numeric_limits<int>::min();
        int maxX = std::numeric_limits<int>::min();
        double sumY = 0.0;
        double sumX = 0.0;
    };
    struct Pixel { int y; int x; int label; };
    struct Center { int y; int x; };

    auto numpyRoundToInt = [](double value) -> int {
        double floorValue = std::floor(value);
        double fraction = value - floorValue;
        if (fraction < 0.5) return static_cast<int>(floorValue);
        if (fraction > 0.5) return static_cast<int>(floorValue + 1.0);
        int lower = static_cast<int>(floorValue);
        return (lower % 2 == 0) ? lower : lower + 1;
    };

    std::vector<LabelStats> stats(nLabels + 1);
    std::vector<Pixel> foregroundPixels;
    foregroundPixels.reserve(static_cast<size_t>(H) * static_cast<size_t>(W) / 4);

    cv::Mat paddedLabels = cv::Mat::zeros(H + 2, W + 2, CV_32SC1);
    for (int y = 0; y < H; ++y) {
        const int* srcRow = labels.ptr<int>(y);
        int* paddedRow = paddedLabels.ptr<int>(y + 1);
        for (int x = 0; x < W; ++x) {
            int lbl = srcRow[x];
            paddedRow[x + 1] = lbl;
            if (lbl <= 0 || lbl > nLabels) continue;
            auto& s = stats[lbl];
            s.area++; s.minY = std::min(s.minY, y); s.minX = std::min(s.minX, x);
            s.maxY = std::max(s.maxY, y); s.maxX = std::max(s.maxX, x);
            s.sumY += static_cast<double>(y); s.sumX += static_cast<double>(x);
            foregroundPixels.push_back({y + 1, x + 1, lbl});
        }
    }

    if (foregroundPixels.empty()) return {dy_all, dx_all};

    std::vector<Center> centers;
    centers.reserve(nLabels);
    int maxExtent = 0;

    for (int lbl = 1; lbl <= nLabels; ++lbl) {
        const auto& s = stats[lbl];
        if (s.area <= 0) continue;
        int height = s.maxY - s.minY + 1, width = s.maxX - s.minX + 1;
        maxExtent = std::max(maxExtent, height + width + 2);
        double localMeanY = (s.sumY - static_cast<double>(s.minY) * s.area) / s.area;
        double localMeanX = (s.sumX - static_cast<double>(s.minX) * s.area) / s.area;
        int targetLocalY = numpyRoundToInt(localMeanY);
        int targetLocalX = numpyRoundToInt(localMeanX);
        int centerY = s.minY + targetLocalY, centerX = s.minX + targetLocalX;
        if (centerY < s.minY || centerY > s.maxY || centerX < s.minX || centerX > s.maxX
            || labels.at<int>(centerY, centerX) != lbl) {
            int bestY = s.minY, bestX = s.minX;
            double bestDistance = std::numeric_limits<double>::max();
            for (int y = s.minY; y <= s.maxY; ++y) {
                const int* row = labels.ptr<int>(y);
                for (int x = s.minX; x <= s.maxX; ++x) {
                    if (row[x] != lbl) continue;
                    double ddy = static_cast<double>(y - s.minY - targetLocalY);
                    double ddx = static_cast<double>(x - s.minX - targetLocalX);
                    double dist = ddy * ddy + ddx * ddx;
                    if (dist < bestDistance) { bestDistance = dist; bestY = y; bestX = x; }
                }
            }
            centerY = bestY; centerX = bestX;
        }
        centers.push_back({centerY + 1, centerX + 1});
    }

    if (centers.empty() || maxExtent <= 0) return {dy_all, dx_all};

    int nIter = 2 * maxExtent;
    cv::Mat temperature = cv::Mat::zeros(H + 2, W + 2, CV_64FC1);
    cv::Mat nextTemperature = cv::Mat::zeros(H + 2, W + 2, CV_64FC1);

    static constexpr int neighborOffsets[9][2] = {
        {0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}, {-1,-1}, {-1,1}, {1,-1}, {1,1}
    };

    for (int iter = 0; iter < nIter; ++iter) {
        for (const auto& center : centers)
            temperature.at<double>(center.y, center.x) += 1.0;
        cv::parallel_for_(cv::Range(0, static_cast<int>(foregroundPixels.size())),
            [&](const cv::Range& range) {
                for (int i = range.start; i < range.end; ++i) {
                    const auto& pixel = foregroundPixels[static_cast<size_t>(i)];
                    double sum = 0.0;
                    for (const auto& offset : neighborOffsets) {
                        int yy = pixel.y + offset[0], xx = pixel.x + offset[1];
                        if (paddedLabels.at<int>(yy, xx) == pixel.label)
                            sum += temperature.at<double>(yy, xx);
                    }
                    nextTemperature.at<double>(pixel.y, pixel.x) = sum / 9.0;
                }
            });
        std::swap(temperature, nextTemperature);
    }

    cv::parallel_for_(cv::Range(0, static_cast<int>(foregroundPixels.size())),
        [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; ++i) {
                const auto& pixel = foregroundPixels[static_cast<size_t>(i)];
                double dyy = temperature.at<double>(pixel.y + 1, pixel.x)
                           - temperature.at<double>(pixel.y - 1, pixel.x);
                double dxx = temperature.at<double>(pixel.y, pixel.x + 1)
                           - temperature.at<double>(pixel.y, pixel.x - 1);
                double norm = std::sqrt(dyy * dyy + dxx * dxx) + 1e-60;
                dy_all.at<double>(pixel.y - 1, pixel.x - 1) = dyy / norm;
                dx_all.at<double>(pixel.y - 1, pixel.x - 1) = dxx / norm;
            }
        });

    return {dy_all, dx_all};
}

// ══════════════════════════════════════════════════════════════════════════════
// remove_bad_flow_masks — Cellpose flow_error equivalent
// ══════════════════════════════════════════════════════════════════════════════

void FlowDynamics::removeBadFlowMasks(cv::Mat& labels,
                                       const cv::Mat& rawDy,
                                       const cv::Mat& rawDx)
{
    if (m_cfg.flowThreshold <= 0.0f) return;

    int H = labels.rows, W = labels.cols;
    double dmin, dmax;
    cv::minMaxLoc(labels, &dmin, &dmax);
    int nLabels = static_cast<int>(dmax);
    if (nLabels <= 0) return;

    cv::Mat netDy, netDx;
    rawDy.convertTo(netDy, CV_64FC1, 1.0 / 5.0);
    rawDx.convertTo(netDx, CV_64FC1, 1.0 / 5.0);

    auto [maskDy, maskDx] = masksToFlows(labels);

    std::vector<double> errorSums(nLabels + 1, 0.0);
    std::vector<int> pixelCounts(nLabels + 1, 0);
    std::mutex reductionMutex;
    cv::parallel_for_(cv::Range(0, H), [&](const cv::Range& range) {
        std::vector<double> localErrorSums(nLabels + 1, 0.0);
        std::vector<int> localPixelCounts(nLabels + 1, 0);
        for (int y = range.start; y < range.end; ++y) {
            const int* labelRow = labels.ptr<int>(y);
            const double* mdyRow = maskDy.ptr<double>(y);
            const double* mdxRow = maskDx.ptr<double>(y);
            const double* ndyRow = netDy.ptr<double>(y);
            const double* ndxRow = netDx.ptr<double>(y);
            for (int x = 0; x < W; ++x) {
                int lbl = labelRow[x];
                if (lbl <= 0 || lbl > nLabels) continue;
                double ddy = mdyRow[x] - ndyRow[x];
                double ddx = mdxRow[x] - ndxRow[x];
                localErrorSums[lbl] += ddy * ddy + ddx * ddx;
                localPixelCounts[lbl]++;
            }
        }
        std::lock_guard<std::mutex> lock(reductionMutex);
        for (int lbl = 1; lbl <= nLabels; ++lbl) {
            errorSums[lbl] += localErrorSums[lbl];
            pixelCounts[lbl] += localPixelCounts[lbl];
        }
    });

    std::vector<uchar> removeLabel(nLabels + 1, 0);
    for (int lbl = 1; lbl <= nLabels; ++lbl) {
        if (pixelCounts[lbl] <= 0) continue;
        double meanErr = errorSums[lbl] / pixelCounts[lbl];
        if (meanErr > m_cfg.flowThreshold) removeLabel[lbl] = 1;
    }

    cv::parallel_for_(cv::Range(0, H), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            int* labelRow = labels.ptr<int>(y);
            for (int x = 0; x < W; ++x) {
                int lbl = labelRow[x];
                if (lbl > 0 && lbl <= nLabels && removeLabel[lbl])
                    labelRow[x] = 0;
            }
        }
    });
}

// ══════════════════════════════════════════════════════════════════════════════
// fillHoles — Cellpose fill_voids.fill equivalent
// ══════════════════════════════════════════════════════════════════════════════

static cv::Mat fillHoles(const cv::Mat& labels)
{
    double minV, maxV;
    cv::minMaxLoc(labels, &minV, &maxV);
    int nLabels = static_cast<int>(maxV);
    if (nLabels <= 0) return labels.clone();

    cv::Mat filled = labels.clone();
    int H = labels.rows, W = labels.cols;

    for (int lbl = 1; lbl <= nLabels; ++lbl) {
        cv::Mat mask = (labels == lbl);
        cv::Rect bbox = cv::boundingRect(mask);
        if (bbox.width <= 2 || bbox.height <= 2) continue;

        // Extract mask within bbox
        cv::Mat localMask = mask(bbox).clone();
        // Invert: holes become 1, label pixels become 0
        cv::Mat inverted;
        cv::bitwise_not(localMask, inverted);

        // Flood-fill from bbox edges — fill background (non-label) pixels
        cv::Mat floodMask = inverted.clone();
        for (int y = 0; y < bbox.height; ++y) {
            if (y == 0 || y == bbox.height - 1) {
                for (int x = 0; x < bbox.width; ++x) {
                    if (floodMask.at<uchar>(y, x))
                        cv::floodFill(floodMask, cv::Point(x, y), cv::Scalar(0));
                }
            } else {
                if (floodMask.at<uchar>(y, 0))
                    cv::floodFill(floodMask, cv::Point(0, y), cv::Scalar(0));
                if (floodMask.at<uchar>(y, bbox.width - 1))
                    cv::floodFill(floodMask, cv::Point(bbox.width - 1, y), cv::Scalar(0));
            }
        }

        // Holes: pixels still 1 in floodMask after flood-fill
        // Fill them in the output
        cv::Mat holeMask = floodMask > 0;
        if (cv::countNonZero(holeMask) > 0) {
            cv::Mat filledRoi = filled(bbox);
            filledRoi.setTo(lbl, holeMask);
        }
    }

    return filled;
}

// ══════════════════════════════════════════════════════════════════════════════
// Build Particle structs from label map
// ══════════════════════════════════════════════════════════════════════════════

std::vector<Particle> FlowDynamics::buildParticles(
    const cv::Mat& labels, const cv::Mat& cellprob, ParticleIdManager* idm)
{
    // Fill holes before any other processing (Cellpose: fill_voids.fill)
    cv::Mat filledLabels = fillHoles(labels);

    double minVal, maxVal;
    cv::minMaxLoc(filledLabels, &minVal, &maxVal);
    int maxLabel = static_cast<int>(maxVal);
    if (maxLabel <= 0) return {};

    int H = filledLabels.rows, W = filledLabels.cols;
    float totalArea = static_cast<float>(H) * W;
    float maxArea = totalArea * m_cfg.maxSizeFraction;

    std::vector<Particle> particles;
    particles.reserve(maxLabel);

    for (int lbl = 1; lbl <= maxLabel; ++lbl) {
        cv::Mat mask = (filledLabels == lbl);
        int area = cv::countNonZero(mask);
        if (area < m_cfg.minSize || area > maxArea) continue;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) continue;

        auto it = std::max_element(contours.begin(), contours.end(),
            [](const auto& a, const auto& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });

        Particle p;
        p.id = idm ? idm->nextId() : static_cast<int>(particles.size()) + 1;
        p.contour = *it;

        cv::Scalar meanProb = cv::mean(cellprob, mask);
        float meanLogit = static_cast<float>(meanProb[0]);
        if (meanLogit < m_cfg.cellprobThreshold) continue;

        p.confidence = 1.0f / (1.0f + std::exp(-meanLogit));

        {
            cv::Rect bbox = cv::boundingRect(*it);
            bool touchesBoundary =
                bbox.x <= m_cfg.edgeTouchMarginPx ||
                bbox.y <= m_cfg.edgeTouchMarginPx ||
                bbox.x + bbox.width >= W - 1 - m_cfg.edgeTouchMarginPx ||
                bbox.y + bbox.height >= H - 1 - m_cfg.edgeTouchMarginPx;
            p.metrics[ParticleMetrics::isEdgeParticle] = touchesBoundary ? 1.0f : 0.0f;
            if (touchesBoundary &&
                m_cfg.boundaryParticlePolicy == BoundaryParticlePolicy::Exclude) continue;
        }

        mask.convertTo(p.mask, CV_32FC1, 1.0);
        particles.push_back(std::move(p));
    }
    return particles;
}
