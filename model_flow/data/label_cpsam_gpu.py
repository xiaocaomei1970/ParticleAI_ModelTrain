"""CPSAM GPU 批量处理：对未标注图片运行 Cellpose CPSAM，保存 PNG 标签图。

在 ModelScope GPU Notebook 上运行。

用法:
    python label_cpsam_gpu.py --img-dir ./unlabeled/ --out-dir ./cpsam_labels/
"""
import os
import argparse
import time

import cv2
import numpy as np
from tqdm import tqdm

from ..utils import imread_unicode, imwrite_unicode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', required=True, help='未标注图片目录')
    parser.add_argument('--out-dir', required=True, help='输出标签图目录')
    parser.add_argument('--model', default='cpsam',
                        help='Cellpose 模型名 (cpsam/cyto3)')
    parser.add_argument('--diameter', type=float, default=0,
                        help='颗粒直径 (0=自动)')
    parser.add_argument('--gpu', action='store_true', default=True,
                        help='使用 GPU (默认)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='强制使用 CPU')
    args = parser.parse_args()

    use_gpu = args.gpu and not args.no_gpu

    os.makedirs(args.out_dir, exist_ok=True)

    # 加载 Cellpose 模型
    from cellpose import models
    print(f'Loading model: {args.model} (GPU={use_gpu})...')
    model = models.CellposeModel(
        gpu=use_gpu, model_type=args.model)

    # 收集图片
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    img_files = sorted([
        f for f in os.listdir(args.img_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    print(f'Found {len(img_files)} images')

    t0 = time.time()
    for fname in tqdm(img_files):
        img_path = os.path.join(args.img_dir, fname)
        img = imread_unicode(img_path)
        if img is None:
            print(f'  Skip: cannot read {fname}')
            continue

        # 灰度显微图：显式转为灰度并指定 channels=[0,0]
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        # CPSAM 推理
        masks, flows, styles = model.eval(
            img_gray, diameter=args.diameter,
            channels=[0, 0],  # grayscale
        )

        # 保存 PNG uint16 标签图
        name = os.path.splitext(fname)[0]
        out_path = os.path.join(args.out_dir, name + '_labels.png')
        imwrite_unicode(out_path, masks.astype(np.uint16))

    elapsed = time.time() - t0
    count = len(os.listdir(args.out_dir))
    print(f'Done! {count} label maps in {elapsed:.0f}s '
          f'({elapsed/max(count,1):.1f}s per image)')
    print(f'Output: {args.out_dir}/')
    print('')
    print('Next: Download this directory to local PC, then run:')
    print('  label_review.exe <img_dir> <label_dir>')
    print('After review, convert labels to flows:')
    print('  python convert_labels_to_flows.py --label-dir <reviewed_dir> --out-dir data/particles/flows_train/')


if __name__ == '__main__':
    main()
