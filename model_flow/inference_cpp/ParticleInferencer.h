#pragma once

#include <array>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct FlowFieldInferenceSettings
{
    int inputSize = 1024;
    std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> std = {0.229f, 0.224f, 0.225f};
    int padValue = 114;
};

/*
 * 简要注释：FlowFieldInferencer 类用于 Flow Field 推理引擎，
 *          负责 ONNX 模型加载、图像预处理以及双阶段网络推理。
 *          输出 (dy, dx, cellprob) flow field，供 FlowDynamics 消费。
 */
class FlowFieldInferencer
{
public:
    /*
     * 简要注释：构造函数
     * 输入参数：backbonePath - 骨干网络模型路径
     *          headPath - FPN+FlowHead 模型路径
     */
    FlowFieldInferencer(const std::string& backbonePath, const std::string& headPath);
    FlowFieldInferencer(const std::string& backbonePath, const std::string& headPath,
                        const FlowFieldInferenceSettings& settings);

    /*
     * 简要注释：执行推理，输出 flow field
     * 输入参数：inputImage - 原始 BGR 格式的图像
     * 输出参数：dy - CV_32FC1 垂直流场 (H/4 × W/4)
     *          dx - CV_32FC1 水平流场
     *          cellprob - CV_32FC1 细胞概率 (logits)
     */
    void infer(const cv::Mat& inputImage,
               cv::Mat& dy, cv::Mat& dx, cv::Mat& cellprob);

    int inputSize() const { return m_inputSize; }
    float getScale() const { return m_scale; }
    float getPadLeft() const { return m_padLeft; }
    float getPadTop() const { return m_padTop; }
    int getOrigH() const { return m_origH; }
    int getOrigW() const { return m_origW; }

private:
    /*
     * 简要注释：图像预处理
     *          预处理顺序与 Python 训练端一致: resize → pad 114 → BGR2RGB → /255 → normalize
     */
    std::vector<float> preprocessImage(const cv::Mat& inputImage);

private:
    Ort::Env m_env;
    Ort::Session m_backboneSession;
    Ort::Session m_headSession;

    int m_inputSize;
    float m_mean[3];
    float m_std[3];
    int m_padValue;
    float m_scale = 1.0f;
    float m_padTop = 0.0f;
    float m_padLeft = 0.0f;
    int m_origH = 0;
    int m_origW = 0;
};
