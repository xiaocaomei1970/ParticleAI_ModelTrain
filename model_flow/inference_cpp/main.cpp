/*
 * Flow Field 推理完整管线
 *
 * 流程:
 *   1. 加载图像
 *   2. backbone.onnx → 4 级特征图
 *   3. neck_head.onnx → (dy, dx, cellprob) @ stride 4
 *   4. bilinear upsample → 全分辨率
 *   5. FlowDynamics::computeMasks() → particles
 *   6. 保存 overlay 与 uint16 label mask，可作为人工审核预标注
 *
 * 依赖:
 *   - ParticleInferencer.h/cpp (本项目的 ONNX 推理)
 *   - FlowDynamics.h/cpp (本项目的 flow segmentation 后处理)
 *   - Particle.h (颗粒结果数据结构)
 */

#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "json.hpp"

#include "PathUtils.h"
#include "ParticleInferencer.h"
#include "FlowDynamics.h"
#include "TiledParticleInferencer.h"
#include "Particle.h"

namespace {

constexpr auto kInferenceConfigFileName = "flow_inference_config.json";

struct RuntimePaths {
    std::string imagePath;
    std::string backbonePath;
    std::string headPath;
    std::string configPath;
    std::string outDir;
    std::string recipePath;    // --recipe analysis_recipe.json
};

struct RuntimeFlags {
    bool useTiled = false;     // --tile
};

void printUsage(const std::string& executableName)
{
    std::cerr << "Usage:\n"
              << "  " << executableName << " <image> <onnx_dir> [out_dir] [--tile] [--recipe recipe.json]\n"
              << "  " << executableName << " <image> <backbone.onnx> <neck_head.onnx>\n";
    std::cerr << "\n  --tile   Enable tiled inference for large images (long side > 1536).\n"
              << "  --recipe Override flow_inference_config.json defaults with values from analysis_recipe.json.\n"
              << "\nWhen out_dir is provided, saves <basename>_result.png and "
              << "<basename>_labels.png.\n";
}

std::pair<RuntimePaths, RuntimeFlags>
parseRuntimePaths(const std::vector<std::string>& arguments)
{
    RuntimePaths paths;
    RuntimeFlags flags;

    if (arguments.size() < 3) {
        return {paths, flags};
    }

    paths.imagePath = arguments[1];

    // parse positional arguments
    int idx = 2;
    if (idx < static_cast<int>(arguments.size())) {
        std::filesystem::path arg2 = pathFromUtf8(arguments[idx]);
        if (std::filesystem::is_directory(arg2)) {
            // <onnx_dir> form
            paths.backbonePath = utf8FromPath(arg2 / "backbone.onnx");
            paths.headPath    = utf8FromPath(arg2 / "neck_head.onnx");
            paths.configPath  = utf8FromPath(arg2 / kInferenceConfigFileName);
            ++idx;
            if (idx < static_cast<int>(arguments.size()) && arguments[idx] != "--tile" && arguments[idx] != "--recipe")
                { paths.outDir = arguments[idx]; ++idx; }
        } else {
            // <backbone.onnx> <neck_head.onnx> form
            paths.backbonePath = arguments[idx]; ++idx;
            if (idx < static_cast<int>(arguments.size()))
                { paths.headPath = arguments[idx]; ++idx; }
            paths.configPath = utf8FromPath(std::filesystem::path(pathFromUtf8(paths.backbonePath)).parent_path() / kInferenceConfigFileName);
        }
    }

    // parse flags
    while (idx < static_cast<int>(arguments.size())) {
        if (arguments[idx] == "--tile") {
            flags.useTiled = true;
            ++idx;
        } else if (arguments[idx] == "--recipe" && idx + 1 < static_cast<int>(arguments.size())) {
            paths.recipePath = arguments[idx + 1];
            idx += 2;
        } else {
            ++idx;
        }
    }

    return {paths, flags};
}

const nlohmann::json& requireField(const nlohmann::json& config, const char* key)
{
    if (!config.contains(key)) {
        throw std::runtime_error(std::string("Missing required config field: ") + key);
    }
    return config.at(key);
}

int readRequiredInt(const nlohmann::json& config, const char* key)
{
    return requireField(config, key).get<int>();
}

float readRequiredFloat(const nlohmann::json& config, const char* key)
{
    return requireField(config, key).get<float>();
}

bool readRequiredBool(const nlohmann::json& config, const char* key)
{
    return requireField(config, key).get<bool>();
}

std::string readRequiredString(const nlohmann::json& config, const char* key)
{
    return requireField(config, key).get<std::string>();
}

std::array<float, 3> readRequiredFloatArray3(const nlohmann::json& config,
                                             const char* key)
{
    const auto& arrayValue = requireField(config, key);
    if (!arrayValue.is_array() || arrayValue.size() != 3)
        throw std::runtime_error(std::string(key) + " must be an array of 3 numbers");

    std::array<float, 3> values{};
    for (size_t i = 0; i < 3; ++i)
        values[i] = arrayValue.at(i).get<float>();
    return values;
}

cv::Rect clippedRect(const cv::Rect& rect, const cv::Size& bounds)
{
    return rect & cv::Rect(0, 0, bounds.width, bounds.height);
}

std::vector<cv::Rect> readIgnoreRegions(const nlohmann::json& runtime)
{
    std::vector<cv::Rect> regions;
    if (!runtime.contains("ignore_regions")) {
        return regions;
    }

    const auto& ignoreRegions = runtime["ignore_regions"];
    if (!ignoreRegions.is_array()) {
        throw std::runtime_error("resolved_parameters.runtime.ignore_regions must be an array");
    }

    for (const auto& item : ignoreRegions) {
        if (!item.is_object()) {
            throw std::runtime_error("Each ignore_regions item must be an object");
        }
        for (const char* key : {"x", "y", "width", "height"}) {
            if (!item.contains(key) || !item[key].is_number_integer()) {
                throw std::runtime_error(
                    std::string("ignore_regions item missing integer field: ") + key);
            }
        }
        int x = item["x"].get<int>();
        int y = item["y"].get<int>();
        int width = item["width"].get<int>();
        int height = item["height"].get<int>();
        if (x < 0 || y < 0 || width <= 0 || height <= 0) {
            throw std::runtime_error(
                "ignore_regions require x/y >= 0 and width/height > 0");
        }
        regions.emplace_back(x, y, width, height);
    }
    return regions;
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

void rejectDeprecatedFields(const nlohmann::json& config)
{
    static const std::vector<const char*> deprecatedKeys = {
        "final_cellprob_threshold_logit",
    };

    for (const char* key : deprecatedKeys) {
        if (config.contains(key)) {
            throw std::runtime_error(
                std::string("Deprecated config field is not allowed: ") + key);
        }
    }
}

void loadInferenceConfig(const std::string& configPath,
                         FlowFieldInferenceSettings& inferenceSettings,
                         FlowDynamics::Config& flowDynamicsConfig)
{
    std::ifstream input(pathFromUtf8(configPath));
    if (!input) {
        throw std::runtime_error("Cannot open inference config: " + configPath);
    }

    nlohmann::json config;
    input >> config;

    rejectDeprecatedFields(config);

    int schemaVersion = readRequiredInt(config, "schema_version");
    if (schemaVersion != 1) {
        std::ostringstream oss;
        oss << "Unsupported flow_inference_config schema_version: "
            << schemaVersion << " (expected 1)";
        throw std::runtime_error(oss.str());
    }

    if (!readRequiredBool(config, "fixed_input_size")) {
        throw std::runtime_error("Only fixed_input_size=true is supported.");
    }

    int outputStride = readRequiredInt(config, "output_stride");
    if (outputStride != 4) {
        std::ostringstream oss;
        oss << "Unsupported output_stride: " << outputStride
            << " (expected 4). V1 production model uses stride=4; "
            << "stride=2 is experimental and requires a different ONNX model.";
        throw std::runtime_error(oss.str());
    }

    inferenceSettings.inputSize = readRequiredInt(config, "input_size");
    inferenceSettings.padValue = readRequiredInt(config, "pad_value");
    inferenceSettings.mean = readRequiredFloatArray3(config, "mean");
    inferenceSettings.std = readRequiredFloatArray3(config, "std");

    float eulerGate = readRequiredFloat(config, "euler_cellprob_threshold_logit");
    if (std::abs(eulerGate) > 1e-6f) {
        throw std::runtime_error(
            "Only euler_cellprob_threshold_logit=0.0 is supported by current C++ FlowDynamics.");
    }
    (void)readRequiredFloat(config, "euler_cellprob_threshold_probability");

    flowDynamicsConfig.cellprobThreshold =
        readRequiredFloat(config, "fd_cellprob_threshold");
    flowDynamicsConfig.niter =
        readRequiredInt(config, "fd_niter");
    flowDynamicsConfig.minSize =
        readRequiredInt(config, "fd_min_size");
    flowDynamicsConfig.flowThreshold =
        readRequiredFloat(config, "fd_flow_threshold");
    flowDynamicsConfig.maxSizeFraction =
        readRequiredFloat(config, "fd_max_size_fraction");
}

/// Load recipe JSON and override config fields where present.
///
/// Performs structural validation before applying overrides:
///   - schema_version must be 1
///   - analysis_mode (required), advanced_overrides (required)
///   - resolved_parameters and its 4 sub-objects must exist
///   - Unknown fields in resolved_parameters.flow_dynamics produce a warning
///
/// Full JSON Schema validation is performed by the Qt application layer
/// or by the CLI script: python scripts/validate_recipe.py <recipe>
void applyRecipeOverrides(const std::string& recipePath,
                          FlowDynamics::Config& dynCfg,
                          TiledParticleInferencer::Config& tileCfg)
{
    std::ifstream input(pathFromUtf8(recipePath));
    if (!input) {
        throw std::runtime_error("Cannot open recipe: " + recipePath);
    }
    nlohmann::json recipe;
    input >> recipe;

    // ── schema_version ──
    if (!recipe.contains("schema_version")) {
        throw std::runtime_error(
            "Recipe is missing required field: schema_version");
    }
    int schemaVer = recipe["schema_version"].get<int>();
    if (schemaVer != 1) {
        std::ostringstream oss;
        oss << "Unsupported recipe schema_version: " << schemaVer
            << " (expected 1). This C++ code only supports V1 recipe format.";
        throw std::runtime_error(oss.str());
    }

    // ── analysis_mode (schema required, enum: standard|advanced) ──
    if (!recipe.contains("analysis_mode")) {
        throw std::runtime_error(
            "Recipe is missing required field: analysis_mode");
    }
    {
        std::string mode = recipe["analysis_mode"].get<std::string>();
        if (mode != "standard" && mode != "advanced") {
            std::ostringstream oss;
            oss << "Invalid analysis_mode: '" << mode
                << "' (expected 'standard' or 'advanced')";
            throw std::runtime_error(oss.str());
        }
    }

    // ── advanced_overrides.enabled (schema required, type boolean) ──
    if (!recipe.contains("advanced_overrides")) {
        throw std::runtime_error(
            "Recipe is missing required field: advanced_overrides");
    }
    {
        const auto& ao = recipe["advanced_overrides"];
        if (!ao.contains("enabled") || !ao["enabled"].is_boolean()) {
            throw std::runtime_error(
                "Recipe advanced_overrides.enabled is missing or not boolean");
        }
    }

    if (!recipe.contains("resolved_parameters")) {
        throw std::runtime_error(
            "Recipe is missing required field: resolved_parameters");
    }
    const auto& rp = recipe["resolved_parameters"];

    // ── resolved_parameters 必填子对象 ──
    static const std::vector<const char*> kRequiredResolvedFields = {
        "preprocessing", "flow_dynamics", "runtime", "statistics"
    };
    for (const char* field : kRequiredResolvedFields) {
        if (!rp.contains(field)) {
            std::ostringstream oss;
            oss << "Recipe resolved_parameters is missing required sub-field: "
                << field;
            throw std::runtime_error(oss.str());
        }
    }

    // ── flow_dynamics 未知字段警告 ──
    static const std::set<std::string> kKnownFlowDynamicsFields = {
        "euler_cellprob_threshold_logit", "fd_cellprob_threshold",
        "fd_niter", "fd_min_size", "fd_flow_threshold", "fd_max_size_fraction"
    };
    if (rp.contains("flow_dynamics")) {
        for (const auto& [key, _] : rp["flow_dynamics"].items()) {
            if (kKnownFlowDynamicsFields.find(key)
                == kKnownFlowDynamicsFields.end()) {
                std::cerr << "WARNING: unknown field in recipe "
                             "resolved_parameters.flow_dynamics: "
                          << key << " (will be ignored)\n";
            }
        }
    }

    // ── 应用覆盖值 ──
    if (rp.contains("flow_dynamics")) {
        const auto& fd = rp["flow_dynamics"];
        if (fd.contains("fd_niter"))            dynCfg.niter            = fd["fd_niter"].get<int>();
        // fd_min_size 是已换算的像素面积值 (由 Qt 应用层根据 pixel_size 和
        // user_parameters.minimum_detectable_diameter 换算)。
        // C++ 只消费此已解析的像素值, 不做物理单位换算。
        if (fd.contains("fd_min_size"))         dynCfg.minSize          = fd["fd_min_size"].get<int>();
        if (fd.contains("fd_flow_threshold"))   dynCfg.flowThreshold    = fd["fd_flow_threshold"].get<float>();
        if (fd.contains("fd_cellprob_threshold")) dynCfg.cellprobThreshold = fd["fd_cellprob_threshold"].get<float>();
        if (fd.contains("fd_max_size_fraction")) dynCfg.maxSizeFraction = fd["fd_max_size_fraction"].get<float>();
    }
    if (rp.contains("runtime")) {
        const auto& rt = rp["runtime"];
        if (rt.contains("edge_touch_margin_px"))
            dynCfg.edgeTouchMarginPx = rt["edge_touch_margin_px"].get<float>();
        if (rt.contains("boundary_particle_policy")) {
            std::string policy = rt["boundary_particle_policy"].get<std::string>();
            if (policy == "exclude") {
                dynCfg.boundaryParticlePolicy =
                    FlowDynamics::BoundaryParticlePolicy::Exclude;
            } else if (policy == "mark_and_exclude_from_psd") {
                dynCfg.boundaryParticlePolicy =
                    FlowDynamics::BoundaryParticlePolicy::MarkAndExcludeFromPsd;
            } else if (policy == "include") {
                dynCfg.boundaryParticlePolicy =
                    FlowDynamics::BoundaryParticlePolicy::Include;
            } else {
                throw std::runtime_error("Unsupported boundary_particle_policy: " + policy);
            }
            std::cout << "Recipe boundary_particle_policy: " << policy << "\n";
        }
        if (rt.contains("tile_size"))       tileCfg.tileSize      = rt["tile_size"].get<int>();
        if (rt.contains("tile_overlap"))    tileCfg.overlap       = rt["tile_overlap"].get<int>();
        if (rt.contains("tile_core_margin")) tileCfg.coreMargin   = rt["tile_core_margin"].get<int>();
        tileCfg.ignoreRegions = readIgnoreRegions(rt);
    }
}

} // namespace

int runMain(const std::vector<std::string>& arguments)
{
    if (arguments.size() < 3)
    {
        printUsage(arguments.empty() ? "flow_inference" : arguments[0]);
        return 1;
    }

    auto [paths, flags] = parseRuntimePaths(arguments);

    FlowFieldInferenceSettings inferenceSettings;
    FlowDynamics::Config flowDynamicsConfig;
    TiledParticleInferencer::Config tileCfg;
    // tileCfg defaults: tileSize=1024, overlap=256, coreMargin=128, longSideLimit=1536
    try {
        loadInferenceConfig(paths.configPath, inferenceSettings, flowDynamicsConfig);

        // apply recipe overrides
        if (!paths.recipePath.empty()) {
            applyRecipeOverrides(paths.recipePath, flowDynamicsConfig, tileCfg);
            std::cout << "Recipe overrides applied: " << paths.recipePath << "\n";
        }
        std::cout << "Config: " << paths.configPath << "\n";
        // 阈值说明: Euler 前景门控硬编码为 logit > 0.0 (prob > 0.5)，
        //   fd_cellprob_threshold 是最终颗粒置信度过滤（mean logit）
        std::cout << "FlowDynamics: niter=" << flowDynamicsConfig.niter
                  << " minSize=" << flowDynamicsConfig.minSize
                  << " flowThreshold=" << flowDynamicsConfig.flowThreshold
                  << " cellprobThreshold(final)=" << flowDynamicsConfig.cellprobThreshold
                  << " (Euler fg gate: logit>0.0, prob>0.5)\n";
        if (!tileCfg.ignoreRegions.empty()) {
            std::cout << "Runtime ignore_regions: "
                      << tileCfg.ignoreRegions.size() << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << "\n";
        return 1;
    }

    // ── 1. 加载图像 ──
    cv::Mat imgBgr = imreadUnicode(paths.imagePath, cv::IMREAD_COLOR);
    if (imgBgr.empty())
    {
        std::cerr << "Failed to load: " << paths.imagePath << "\n";
        return 1;
    }
    std::cout << "Image: " << imgBgr.cols << "x" << imgBgr.rows << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    int origH = imgBgr.rows, origW = imgBgr.cols;
    long long inferMs = 0, postMs = 0;
    std::vector<Particle> particles;

    if (flags.useTiled && TiledParticleInferencer(tileCfg, "", "", inferenceSettings, flowDynamicsConfig).shouldTile(imgBgr))
    {
        std::cout << "Tiled inference enabled (" << tileCfg.tileSize
                  << "/" << tileCfg.overlap << "/" << tileCfg.coreMargin << ")\n";
        TiledParticleInferencer tiler(tileCfg,
            paths.backbonePath, paths.headPath, inferenceSettings, flowDynamicsConfig);
        std::string reportPath;
        if (!paths.outDir.empty()) {
            std::filesystem::path outDirPath = pathFromUtf8(paths.outDir);
            std::filesystem::create_directories(outDirPath);
            reportPath = utf8FromPath(outDirPath / "tile_merge_report.json");
        }
        ParticleIdManager idm;
        particles = tiler.process(imgBgr, reportPath, &idm);
        const auto& st = tiler.stats();
        auto tEnd = std::chrono::high_resolution_clock::now();
        inferMs = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - t0).count();
        postMs = 0;
        std::cout << "Tiled: " << st.totalTiles << " tiles, "
                  << st.totalDetected << " detected, "
                  << st.coreRetained << " core, "
                  << st.duplicateMerged << " merged, "
                  << st.finalCount << " final ("
                  << st.totalTimeMs << "ms)\n";
    }
    else
    {
        // ── 单图推理 ──
        FlowFieldInferencer inferencer(
            paths.backbonePath, paths.headPath, inferenceSettings);

        cv::Mat dyS4, dxS4, cellprobS4;
        inferencer.infer(imgBgr, dyS4, dxS4, cellprobS4);
        auto t1 = std::chrono::high_resolution_clock::now();
        inferMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "ONNX inference: " << inferMs << "ms\n";

        // upsample + pad masking
        int fullSize = inferencer.inputSize();
        cv::Mat dyFull, dxFull, cellprobFull;
        cv::resize(dyS4, dyFull, cv::Size(fullSize, fullSize), 0, 0, cv::INTER_LINEAR);
        cv::resize(dxS4, dxFull, cv::Size(fullSize, fullSize), 0, 0, cv::INTER_LINEAR);
        cv::resize(cellprobS4, cellprobFull, cv::Size(fullSize, fullSize), 0, 0, cv::INTER_LINEAR);

        {
            int newH = static_cast<int>(inferencer.getOrigH() * inferencer.getScale());
            int newW = static_cast<int>(inferencer.getOrigW() * inferencer.getScale());
            int padTop = inferencer.getPadTop();
            int padLeft = inferencer.getPadLeft();
            int padBottom = fullSize - newH - padTop;
            int padRight = fullSize - newW - padLeft;
            if (padTop > 0)    cellprobFull(cv::Rect(0, 0, fullSize, padTop)) = -100.0f;
            if (padBottom > 0) cellprobFull(cv::Rect(0, fullSize - padBottom, fullSize, padBottom)) = -100.0f;
            if (padLeft > 0)   cellprobFull(cv::Rect(0, padTop, padLeft, newH)) = -100.0f;
            if (padRight > 0)  cellprobFull(cv::Rect(fullSize - padRight, padTop, padRight, newH)) = -100.0f;
        }
        applyIgnoreRegionsToFlow(dyFull, dxFull, cellprobFull,
                                 tileCfg.ignoreRegions,
                                 inferencer.getScale(),
                                 static_cast<int>(inferencer.getPadLeft()),
                                 static_cast<int>(inferencer.getPadTop()));

        FlowDynamics dynamics(flowDynamicsConfig);
        particles = dynamics.computeMasks(dyFull, dxFull, cellprobFull);
        {
            auto tPost = std::chrono::high_resolution_clock::now();
            postMs = std::chrono::duration_cast<std::chrono::milliseconds>(tPost - t1).count();
        }

        // coordinate remapping (padded → original)
        float scale_f = inferencer.getScale();
        float padLeft_f = static_cast<float>(inferencer.getPadLeft());
        float padTop_f = static_cast<float>(inferencer.getPadTop());
        int fullSz = inferencer.inputSize();
        for (auto& p : particles) {
            for (auto& pt : p.contour) {
                pt.x = static_cast<int>((pt.x - padLeft_f) / scale_f);
                pt.y = static_cast<int>((pt.y - padTop_f) / scale_f);
            }
            cv::Mat paddedMask = cv::Mat::zeros(fullSz, fullSz, CV_32FC1);
            p.mask.copyTo(paddedMask);
            int newH = static_cast<int>(origH * scale_f);
            int newW = static_cast<int>(origW * scale_f);
            int padTop_i = static_cast<int>(padTop_f);
            int padLeft_i = static_cast<int>(padLeft_f);
            cv::Rect contentRoi(padLeft_i, padTop_i, newW, newH);
            contentRoi &= cv::Rect(0, 0, fullSz, fullSz);
            cv::Mat cropped = paddedMask(contentRoi);
            cv::Mat origMask;
            cv::resize(cropped, origMask, cv::Size(origW, origH), 0, 0, cv::INTER_NEAREST);
            p.mask = origMask;
        }
        eraseRegionsFromParticles(
            particles, tileCfg.ignoreRegions, cv::Size(origW, origH));
        std::cout << "Single-image: " << particles.size() << " particles\n";
    }

    // ── output info ──
    for (size_t i = 0; i < std::min(particles.size(), size_t(5)); ++i) {
        const auto& p = particles[i];
        cv::Rect bbox = cv::boundingRect(p.contour);
        std::cout << "  #" << p.id
                  << " bbox=[" << bbox.x << "," << bbox.y
                  << "," << bbox.width << "," << bbox.height << "]"
                  << " confidence=" << p.confidence << "\n";
    }
    std::cout << "Done.\n";

    // ── 8. 保存叠加图和 uint16 标签图 (如果指定了输出目录) ──
    if (!paths.outDir.empty())
    {
        std::filesystem::path outDirPath = pathFromUtf8(paths.outDir);
        std::filesystem::create_directories(outDirPath);

        std::filesystem::path inPath = pathFromUtf8(paths.imagePath);
        std::string baseName = inPath.stem().string();
        std::string outPath = utf8FromPath(outDirPath / (baseName + "_result.png"));
        std::string labelPath = utf8FromPath(outDirPath / (baseName + "_labels.png"));
        std::string metaPath = utf8FromPath(outDirPath / (baseName + "_metadata.json"));

        cv::Mat labelMask = cv::Mat::zeros(origH, origW, CV_16UC1);
        for (const auto& p : particles)
        {
            int labelValue = std::clamp(p.id, 1, 65535);
            labelMask.setTo(static_cast<uint16_t>(labelValue), p.mask > 0.5f);
        }

        if (imwriteUnicode(labelPath, labelMask))
            std::cout << "Saved labels: " << labelPath << "\n";
        else
            std::cerr << "Failed to save labels: " << labelPath << "\n";

        // Write inference metadata JSON for traceability
        {
            nlohmann::json meta;
            meta["flow_dynamics_version"] = "1.0";
            meta["niter"] = flowDynamicsConfig.niter;
            meta["min_size"] = flowDynamicsConfig.minSize;
            meta["flow_threshold"] = flowDynamicsConfig.flowThreshold;
            meta["cellprob_threshold"] = flowDynamicsConfig.cellprobThreshold;
            meta["max_size_fraction"] = flowDynamicsConfig.maxSizeFraction;
            meta["boundary_particle_policy"] =
                flowDynamicsConfig.boundaryParticlePolicy ==
                    FlowDynamics::BoundaryParticlePolicy::Exclude ? "exclude" :
                flowDynamicsConfig.boundaryParticlePolicy ==
                    FlowDynamics::BoundaryParticlePolicy::MarkAndExcludeFromPsd ?
                    "mark_and_exclude_from_psd" : "include";
            meta["particle_count"] = static_cast<int>(particles.size());
            meta["image"] = paths.imagePath;
            meta["ignore_regions_count"] = static_cast<int>(tileCfg.ignoreRegions.size());
            if (!paths.recipePath.empty())
                meta["recipe"] = paths.recipePath;
            std::ofstream mf(pathFromUtf8(metaPath));
            mf << meta.dump(2) << std::endl;
            std::cout << "Saved metadata: " << metaPath << "\n";
        }

        // 在原图上绘制颗粒轮廓和编号
        cv::Mat overlay = imgBgr.clone();
        for (const auto& p : particles)
        {
            cv::Scalar color(0, 255, 0); // 绿色轮廓
            cv::drawContours(overlay, std::vector<std::vector<cv::Point>>{p.contour},
                             -1, color, 2);
            cv::Rect bbox = cv::boundingRect(p.contour);
            cv::putText(overlay, std::to_string(p.id),
                        cv::Point(bbox.x, bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }

        // 添加检测信息
        std::string info = std::to_string(particles.size()) + " particles ("
                           + std::to_string(inferMs + postMs) + "ms)";
        cv::putText(overlay, info, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        if (imwriteUnicode(outPath, overlay))
            std::cout << "Saved: " << outPath << "\n";
        else
            std::cerr << "Failed to save: " << outPath << "\n";
    }

    return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[])
{
    std::vector<std::string> arguments;
    arguments.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i)
        arguments.push_back(utf8FromWide(argv[i]));
    return runMain(arguments);
}
#else
int main(int argc, char* argv[])
{
    std::vector<std::string> arguments;
    arguments.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i)
        arguments.emplace_back(argv[i]);
    return runMain(arguments);
}
#endif
