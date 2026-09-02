#!/bin/bash
# ModelScope Notebook setup for Flow Field training.
# Usage: bash setup_env_modelscope.sh
set -e

echo "=== ModelScope Flow Field training environment ==="

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=== Installing dependencies ==="
pip install -r requirements_modelscope.txt -q

echo ""
echo "=== Preparing bundled pretrained weights ==="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEIGHT_DIR="${SCRIPT_DIR}/models--timm--convnext_small.dinov3_lvd1689m"
if [ -d "$WEIGHT_DIR" ]; then
    export HUGGINGFACE_HUB_CACHE="${SCRIPT_DIR}"
    export HF_HUB_OFFLINE=1
    echo "Bundled ConvNeXt-S DINOv3 weights found and offline mode enabled (HUGGINGFACE_HUB_CACHE=${SCRIPT_DIR})."
else
    echo "WARNING: bundled ConvNeXt-S DINOv3 weights not found; timm may try to download online."
fi

echo ""
echo "=== Verifying dependencies ==="
python -c "
import os
import torch
import timm
import cellpose
import numpy as np

print(f'torch: {torch.__version__}')
print(f'timm: {timm.__version__}')
print(f'cellpose: {getattr(cellpose, \"version_str\", getattr(cellpose, \"__version__\", \"unknown\"))}')
print(f'HUGGINGFACE_HUB_CACHE: {os.environ.get(\"HUGGINGFACE_HUB_CACHE\", \"\")}')
print(f'HF_HUB_OFFLINE: {os.environ.get(\"HF_HUB_OFFLINE\", \"\")}')
m = timm.create_model('convnext_small.dinov3_lvd1689m', pretrained=True, features_only=True)
print('ConvNeXt-S DINOv3 OK')
from cellpose.dynamics import labels_to_flows
label = np.zeros((32, 32), dtype=np.uint16)
label[8:24, 8:24] = 1
flows = labels_to_flows([label], device=torch.device('cpu'))
assert len(flows) == 1 and flows[0].shape[0] >= 4, 'labels_to_flows smoke test failed'
print('cellpose labels_to_flows OK')
print('All dependencies verified!')
"

pip freeze > train_environment_freeze.txt
echo "Dependency versions written to train_environment_freeze.txt"

echo ""
echo "=== Ready ==="
echo "GT flow fields were precomputed locally and packed into this archive."
echo "Training reads .npy flow fields directly."
echo ""
echo "Train:"
echo "  cd /mnt/workspace/particles_flow_v1"
echo "  python -m model_flow.flow_train"
echo ""
echo "Background:"
echo "  nohup python -m model_flow.flow_train > train.log 2>&1 &"
echo "  tail -f train.log"
