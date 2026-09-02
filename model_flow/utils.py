"""共享工具：长路径支持 + Unicode 图片读取"""
import os
import sys

import cv2
import numpy as np


def long_path(path):
    """为 Windows 超长路径添加 \\?\ 前缀 (>260 字符)。

    在 Windows 上，不带此前缀的路径限制为 MAX_PATH (260 字符)。
    numpy.save/load 和 Python open() 均支持 \\?\ 扩展路径（最长 32767 字符）。
    非 Windows 平台原样返回。
    """
    if sys.platform == 'win32':
        path = os.path.abspath(path)
        if not path.startswith('\\\\?\\'):
            path = '\\\\?\\' + path
    return path


def imread_unicode(path):
    """读取图片（支持 Unicode 路径和超长路径）。

    返回 numpy 数组，解码失败返回 None。文件不存在时 np.fromfile 抛 FileNotFoundError。
    """
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imread_unchanged(path):
    """读取图片，保留原始位深和通道数（支持 Unicode 路径和超长路径）。

    用于读取 uint16 标签图等需要精确像素值的场景。
    返回 numpy 数组，解码失败返回 None。文件不存在时 np.fromfile 抛 FileNotFoundError。
    """
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def validate_instance_label(label, max_area_fraction=0.8, allow_empty=False):
    """校验实例标签图，返回错误信息列表（空列表表示合法）。

    label: numpy 数组
    max_area_fraction: 单个实例占图像面积的最大比例，超过则报错
    allow_empty: 是否允许全背景标签（仅 background_negative tile 应为 True）
    """
    errors = []

    if label is None:
        errors.append("标签为 None，文件可能损坏或无法解码")
        return errors

    # 必须是单通道
    if label.ndim == 3:
        if label.shape[2] == 1:
            label = label[:, :, 0]
        elif label.shape[2] >= 3:
            errors.append(
                f"标签为 {label.shape[2]} 通道彩色图，不能作为实例标签。"
                "请提供单通道 uint16 实例标签图")
            return errors
        else:
            errors.append(f"标签通道数异常: shape={label.shape}")
            return errors

    # 必须是整数类型
    if not np.issubdtype(label.dtype, np.integer):
        errors.append(f"标签 dtype 必须为整数类型，当前为 {label.dtype}")
        return errors

    # 不能有负值
    if (label < 0).any():
        errors.append("标签包含负值")

    total_pixels = label.size
    unique_ids = np.unique(label)
    positive_ids = unique_ids[unique_ids > 0]

    # 背景必须包含 0
    if 0 not in unique_ids:
        errors.append("标签缺少背景 (id=0)")

    # 至少有一个正实例
    if len(positive_ids) == 0:
        if not allow_empty:
            errors.append("标签没有任何实例 (全为 0)")
        return errors

    # 单个实例不能覆盖过大面积
    for inst_id in positive_ids:
        area = int((label == inst_id).sum())
        fraction = area / total_pixels
        if fraction > max_area_fraction:
            errors.append(
                f"实例 id={inst_id} 覆盖图像 {fraction:.1%} 面积，"
                f"超过上限 {max_area_fraction:.0%}")

    return errors


def imwrite_unicode(path, img, *args, **kwargs):
    """保存图片（支持 Unicode 路径和超长路径）。

    额外参数透传给 cv2.imencode（如 [cv2.IMWRITE_PNG_COMPRESSION, 3]）。
    """
    ext = os.path.splitext(path)[1] or '.png'
    success, buf = cv2.imencode(ext, img, *args, **kwargs)
    if not success:
        raise IOError(f"Cannot encode image to: {path}")
    buf.tofile(path)
