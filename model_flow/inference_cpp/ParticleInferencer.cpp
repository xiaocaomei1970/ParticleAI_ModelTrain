#include "ParticleInferencer.h"
#include "PathUtils.h"
#include <iostream>

static Ort::SessionOptions createSessionOptions(bool& useDML)
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 尝试 DirectML: 优先 GPU/DirectML → CPU 回退
    // DML 要求: ORT_SEQUENTIAL + enable_mem_pattern=false
    // 需要链接 onnxruntime-directml (非标准 onnxruntime)
    try {
        opts.AppendExecutionProvider("DML", {});
        opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        opts.DisableMemPattern();
        useDML = true;
        std::cout << "[Inferencer] DirectML execution provider enabled" << std::endl;
    } catch (const Ort::Exception& e) {
        useDML = false;
        std::cerr << "[Inferencer] DirectML not available, falling back to CPU: "
                  << e.what() << std::endl;
    }

    return opts;
}

FlowFieldInferencer::FlowFieldInferencer(const std::string& backbonePath,
                                           const std::string& headPath)
    : FlowFieldInferencer(backbonePath, headPath, FlowFieldInferenceSettings{})
{
}

FlowFieldInferencer::FlowFieldInferencer(
    const std::string& backbonePath,
    const std::string& headPath,
    const FlowFieldInferenceSettings& settings)
    : m_env(ORT_LOGGING_LEVEL_WARNING, "flow_inference"),
      m_backboneSession(nullptr),
      m_headSession(nullptr),
      m_inputSize(settings.inputSize),
      m_mean{settings.mean[0], settings.mean[1], settings.mean[2]},
      m_std{settings.std[0], settings.std[1], settings.std[2]},
      m_padValue(settings.padValue)
{
    bool useDML = false;
    Ort::SessionOptions opts = createSessionOptions(useDML);

    auto wbp = widePathFromUtf8(backbonePath);
    auto whp = widePathFromUtf8(headPath);

    m_backboneSession = Ort::Session(m_env, wbp.c_str(), opts);
    m_headSession = Ort::Session(m_env, whp.c_str(), opts);

    if (useDML) {
        std::cout << "[Inferencer] Sessions created with DirectML acceleration"
                  << std::endl;
    }
}

std::vector<float> FlowFieldInferencer::preprocessImage(const cv::Mat& inputImage)
{
    int h = inputImage.rows;
    int w = inputImage.cols;

    m_origH = h;
    m_origW = w;
    m_scale = static_cast<float>(m_inputSize) / std::max(h, w);
    int new_h = static_cast<int>(h * m_scale);
    int new_w = static_cast<int>(w * m_scale);

    cv::Mat resized;
    cv::resize(inputImage, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    m_padTop = (m_inputSize - new_h) / 2;
    int pad_bottom = m_inputSize - new_h - m_padTop;
    m_padLeft = (m_inputSize - new_w) / 2;
    int pad_right = m_inputSize - new_w - m_padLeft;

    cv::Mat padded;
    // 算法思想：先 pad 114, 再 BGR→RGB + normalize (与训练一致)
    cv::copyMakeBorder(resized, padded, m_padTop, pad_bottom, m_padLeft, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(m_padValue, m_padValue, m_padValue));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; c++)
    {
        channels[c] = (channels[c] - m_mean[c]) / m_std[c];
    }

    std::vector<float> tensorData(3 * m_inputSize * m_inputSize);
    for (int c = 0; c < 3; c++)
    {
        cv::Mat continuousChannel = channels[c].isContinuous()
            ? channels[c]
            : channels[c].clone();
        std::memcpy(tensorData.data() + c * m_inputSize * m_inputSize,
                    continuousChannel.data,
                    m_inputSize * m_inputSize * sizeof(float));
    }

    return tensorData;
}

void FlowFieldInferencer::infer(const cv::Mat& inputImage,
                                 cv::Mat& dy, cv::Mat& dx, cv::Mat& cellprob)
{
    // 预处理
    std::vector<float> tensorData = preprocessImage(inputImage);

    // ── Backbone 推理 ──
    Ort::AllocatorWithDefaultOptions alloc;
    std::vector<int64_t> inputShape = {1, 3, m_inputSize, m_inputSize};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, tensorData.data(), tensorData.size(),
        inputShape.data(), inputShape.size());

    const char* backboneInputs[] = {"input"};
    const char* backboneOutputs[] = {"stage0", "stage1", "stage2", "stage3"};
    auto backboneOutput = m_backboneSession.Run(
        Ort::RunOptions{nullptr}, backboneInputs, &inputTensor, 1,
        backboneOutputs, 4);

    // ── Neck+Head 推理 ──
    const char* headInputs[] = {"stage0", "stage1", "stage2", "stage3"};
    const char* headOutputs[] = {"flow"};  // 单一输出: (1, 3, H/4, W/4)
    auto headOutput = m_headSession.Run(
        Ort::RunOptions{nullptr}, headInputs, backboneOutput.data(), 4,
        headOutputs, 1);

    // ── 解析输出 ──
    auto& flowTensor = headOutput[0];
    auto shape = flowTensor.GetTensorTypeAndShapeInfo().GetShape();
    // shape: [1, 3, H/4, W/4] → channel order: dy, dx, cellprob

    int s4 = static_cast<int>(shape[2]);  // m_inputSize / 4
    const float* flowData = flowTensor.GetTensorData<float>();

    dy = cv::Mat(s4, s4, CV_32FC1);
    dx = cv::Mat(s4, s4, CV_32FC1);
    cellprob = cv::Mat(s4, s4, CV_32FC1);

    int planeSize = s4 * s4;
    std::memcpy(dy.data, flowData, planeSize * sizeof(float));
    std::memcpy(dx.data, flowData + planeSize, planeSize * sizeof(float));
    std::memcpy(cellprob.data, flowData + 2 * planeSize, planeSize * sizeof(float));
}
