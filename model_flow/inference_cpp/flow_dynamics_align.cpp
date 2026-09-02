/*
 * FlowDynamics 对齐测试工具 (P0-2)
 *
 * 用法:
 *   flow_dynamics_align.exe <dy.bin> <dx.bin> <cellprob.bin> <H> <W> <out_mask.png>
 *       [--niter 200] [--flow_threshold 0.4] [--min_size 50]
 *       [--cellprob_threshold -3.0]
 *       [--save-intermediate]
 *
 * 输入:  三个 float32 二进制文件 (H×W raw data)，分别对应 dy, dx, cellprob(logits)
 * 输出:  uint16 PNG mask (0=背景, 1..N=实例)
 * 可选:  --save-intermediate 时额外输出 remove_bad_flow_masks 前后的中间 mask
 *
 * 该工具供 Python align_flow_dynamics.py 调用，实现 C++ FlowDynamics 与
 * Cellpose compute_masks 的真实对比验证。
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "PathUtils.h"
#include "FlowDynamics.h"
#include "Particle.h"

namespace {

void printUsage(const std::string& name)
{
    std::cerr << "Usage: " << name
              << " <dy.bin> <dx.bin> <cellprob.bin> <H> <W> <out_mask.png>\n"
              << "       [--niter 200] [--flow_threshold 0.4] [--min_size 50]\n"
              << "       [--cellprob_threshold -3.0] [--max_size_fraction 0.5]\n"
              << "       [--save-intermediate]\n";
}

/// 从 float32 二进制文件读取 H×W 矩阵
cv::Mat readFloat32Binary(const std::string& path, int H, int W)
{
    std::ifstream file(pathFromUtf8(path), std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << path << "\n";
        return cv::Mat();
    }

    std::vector<float> data(H * W);
    file.read(reinterpret_cast<char*>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(float)));

    if (file.gcount() != static_cast<std::streamsize>(data.size() * sizeof(float))) {
        std::cerr << "Error: File " << path << " too small for " << H << "x" << W
                  << " (read " << file.gcount() << " bytes, expected "
                  << data.size() * sizeof(float) << ")\n";
        return cv::Mat();
    }

    cv::Mat mat(H, W, CV_32FC1);
    std::memcpy(mat.data, data.data(), data.size() * sizeof(float));
    return mat;
}

/// 将 label mask (CV_32S) 写入 uint16 PNG
bool writeLabelMask(const std::string& path, const cv::Mat& labels32s)
{
    cv::Mat u16;
    labels32s.convertTo(u16, CV_16UC1);
    return imwriteUnicode(path, u16);
}

/// 从 label mask 构建二值可视化 mask (仅用于输出调试信息)
cv::Mat buildOutlineViz(const cv::Mat& labels, int H, int W)
{
    cv::Mat viz(H, W, CV_8UC3, cv::Scalar(0, 0, 0));
    double minV, maxV;
    cv::minMaxLoc(labels, &minV, &maxV);
    int nLabels = static_cast<int>(maxV);

    for (int lbl = 1; lbl <= nLabels; ++lbl) {
        cv::Mat mask = (labels == lbl);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) continue;

        auto& best = *std::max_element(contours.begin(), contours.end(),
            [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
        cv::Scalar color(rand() % 200 + 56, rand() % 200 + 56, rand() % 200 + 56);
        cv::drawContours(viz, std::vector<std::vector<cv::Point>>{best}, -1, color, 2);
    }
    return viz;
}

} // namespace

int runMain(const std::vector<std::string>& arguments)
{
    if (arguments.size() < 7) {
        printUsage(arguments.empty() ? "flow_dynamics_align" : arguments[0]);
        return 1;
    }

    std::string dyPath = arguments[1];
    std::string dxPath = arguments[2];
    std::string cpPath = arguments[3];
    int H = std::stoi(arguments[4]);
    int W = std::stoi(arguments[5]);
    std::string outPath = arguments[6];

    // 可选参数
    FlowDynamics::Config cfg;
    bool saveIntermediate = false;
    for (size_t i = 7; i < arguments.size(); ++i) {
        std::string arg = arguments[i];
        if (arg == "--niter" && i + 1 < arguments.size())
            cfg.niter = std::stoi(arguments[++i]);
        else if (arg == "--flow_threshold" && i + 1 < arguments.size())
            cfg.flowThreshold = std::stof(arguments[++i]);
        else if (arg == "--min_size" && i + 1 < arguments.size())
            cfg.minSize = std::stoi(arguments[++i]);
        else if (arg == "--cellprob_threshold" && i + 1 < arguments.size())
            cfg.cellprobThreshold = std::stof(arguments[++i]);
        else if (arg == "--max_size_fraction" && i + 1 < arguments.size())
            cfg.maxSizeFraction = std::stof(arguments[++i]);
        else if (arg == "--save-intermediate")
            saveIntermediate = true;
    }

    auto outputParent = pathFromUtf8(outPath).parent_path();
    if (!outputParent.empty())
        std::filesystem::create_directories(outputParent);

    std::cout << "FlowDynamics Align Test\n";
    std::cout << "  Size: " << H << "x" << W << "\n";
    std::cout << "  Config: niter=" << cfg.niter
              << " flowThreshold=" << cfg.flowThreshold
              << " cellprobThreshold=" << cfg.cellprobThreshold
              << " minSize=" << cfg.minSize
              << " maxSizeFraction=" << cfg.maxSizeFraction << "\n";

    // 读取输入
    cv::Mat dy = readFloat32Binary(dyPath, H, W);
    cv::Mat dx = readFloat32Binary(dxPath, H, W);
    cv::Mat cellprob = readFloat32Binary(cpPath, H, W);

    if (dy.empty() || dx.empty() || cellprob.empty()) {
        std::cerr << "Error: Failed to read input files\n";
        return 1;
    }

    std::cout << "  dy:       min=" << *std::min_element(dy.begin<float>(), dy.end<float>())
              << " max=" << *std::max_element(dy.begin<float>(), dy.end<float>()) << "\n";
    std::cout << "  dx:       min=" << *std::min_element(dx.begin<float>(), dx.end<float>())
              << " max=" << *std::max_element(dx.begin<float>(), dx.end<float>()) << "\n";
    std::cout << "  cellprob: min=" << *std::min_element(cellprob.begin<float>(), cellprob.end<float>())
              << " max=" << *std::max_element(cellprob.begin<float>(), cellprob.end<float>()) << "\n";

    // 运行 FlowDynamics
    FlowDynamics dynamics(cfg);
    auto particles = dynamics.computeMasks(dy, dx, cellprob);

    std::cout << "  Result: " << particles.size() << " particles\n";

    // 从 particles 重建 label mask
    cv::Mat labels = cv::Mat::zeros(H, W, CV_32SC1);
    for (const auto& p : particles) {
        labels.setTo(p.id, p.mask > 0.5f);
    }

    // 写入最终 mask
    if (!writeLabelMask(outPath, labels)) {
        std::cerr << "Error: Failed to write " << outPath << "\n";
        return 1;
    }
    std::cout << "  Output: " << outPath << "\n";

    // 中间结果（可选）
    if (saveIntermediate) {
        std::string base = outPath.substr(0, outPath.find_last_of('.'));
        std::string vizPath = base + "_viz.png";
        cv::Mat viz = buildOutlineViz(labels, H, W);
        imwriteUnicode(vizPath, viz);
        std::cout << "  Intermediate (viz): " << vizPath << "\n";
    }

    std::cout << "Done.\n";
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
