#pragma once

/// 长路径 + Unicode 文件 I/O 工具
///
/// Windows MAX_PATH (260 字符) 限制: 超过此长度的路径需要 \\?\ 前缀。
/// cv::imread/imwrite 内部可能走 fopen/CRT，不支持 \\?\。
/// 本模块使用 std::filesystem::path + cv::imdecode/imencode 规避。
///
/// Unicode 路径支持（P0-1 修复）:
///   在 Windows 上，std::ifstream(const char*) 使用 ANSI 代码页解释文件名，
///   无法正确处理 UTF-8 编码的中文路径。本模块使用 pathFromUtf8() 将 UTF-8
///   转为 std::filesystem::path（Windows 内部为 wchar_t），确保 Unicode 路径
///   被正确传递给文件系统 API。
///
/// 用法:
///   cv::Mat img = imreadUnicode(path);
///   imwriteUnicode(path, img);
///   auto fp = pathFromUtf8(utf8Path);         // 用于 std::ifstream(fp)
///   auto wp = widePathFromUtf8(utf8Path);     // 用于 ONNX Runtime Ort::Session

#include <string>
#include <fstream>
#include <vector>
#include <filesystem>

#include <opencv2/opencv.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

inline std::string utf8FromWide(const std::wstring& widePath)
{
    if (widePath.empty())
        return {};

    int utf8Len = WideCharToMultiByte(CP_UTF8, 0,
                                      widePath.c_str(),
                                      static_cast<int>(widePath.size()),
                                      nullptr, 0, nullptr, nullptr);
    if (utf8Len <= 0)
        return {};

    std::string utf8Path(utf8Len, '\0');
    WideCharToMultiByte(CP_UTF8, 0,
                        widePath.c_str(),
                        static_cast<int>(widePath.size()),
                        utf8Path.data(), utf8Len, nullptr, nullptr);
    return utf8Path;
}

/// 将 UTF-8 编码的路径转为 std::filesystem::path。
/// Windows 内部使用 wchar_t，此函数确保中文字符被正确转换。
inline std::filesystem::path pathFromUtf8(const std::string& utf8Path)
{
    if (utf8Path.empty())
        return {};

    int wideLen = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                      utf8Path.c_str(),
                                      static_cast<int>(utf8Path.size()),
                                      nullptr, 0);
    if (wideLen <= 0) {
        // fallback: 假设已经是系统代码页
        return std::filesystem::u8path(utf8Path);
    }

    std::wstring wide(wideLen, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                        utf8Path.c_str(),
                        static_cast<int>(utf8Path.size()),
                        wide.data(), wideLen);
    return std::filesystem::path(wide);
}

/// 将 UTF-8 路径转为 std::wstring（供 ONNX Runtime Ort::Session 等需要宽字符串的 API）。
inline std::wstring widePathFromUtf8(const std::string& utf8Path)
{
    return pathFromUtf8(utf8Path).wstring();
}

inline std::string utf8FromPath(const std::filesystem::path& path)
{
    return utf8FromWide(path.wstring());
}

#else
inline std::string utf8FromWide(const std::wstring& widePath)
{
    return std::string(widePath.begin(), widePath.end());
}

inline std::filesystem::path pathFromUtf8(const std::string& utf8Path)
{
    return std::filesystem::u8path(utf8Path);
}

inline std::wstring widePathFromUtf8(const std::string& utf8Path)
{
    return pathFromUtf8(utf8Path).wstring();
}

inline std::string utf8FromPath(const std::filesystem::path& path)
{
    return path.string();
}
#endif

/// 读取图片（支持 Unicode 路径和超长路径）。
/// 使用 std::filesystem::path 打开文件，Windows 上正确支持中文路径。
inline cv::Mat imreadUnicode(const std::string& path, int flags = cv::IMREAD_COLOR)
{
    auto fp = pathFromUtf8(path);
    std::ifstream file(fp, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        return cv::Mat();
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), size))
        return cv::Mat();
    return cv::imdecode(buffer, flags);
}

/// 保存图片（支持 Unicode 路径和超长路径）。
inline bool imwriteUnicode(const std::string& path, const cv::Mat& img,
                           const std::vector<int>& params = {})
{
    std::string ext = path.substr(path.find_last_of('.'));
    std::vector<uchar> buf;
    if (!cv::imencode(ext, img, buf, params))
        return false;

    auto fp = pathFromUtf8(path);
    std::ofstream file(fp, std::ios::binary);
    if (!file.is_open())
        return false;
    file.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    return true;
}
