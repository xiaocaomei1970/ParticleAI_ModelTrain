# 模型训练和C++后处理方案

## 1. 目标与边界

本方案用于从零准备 V1 数据，训练基于 ConvNeXt-S DINOv3 backbone 的 Flow Field 模型，导出 ONNX，并在 C++ 中正确执行 FlowDynamics 后处理，最终输出精细、准确的颗粒实例轮廓。V1 重点侧重于**材料/粉体领域**的颗粒显微图，不把其它行业的通用图像分割作为目标。

模型输出为 `(dy, dx, cellprob)`。训练目标来自 Cellpose `labels_to_flows()` 生成的 GT flow，GT 通道顺序为 `(cellprob, dy, dx)`。完整技术路线是：

```text
人工实例标签 -> labels_to_flows -> GT flow -> 训练模型 -> 预测 flow -> FlowDynamics -> 实例 mask/轮廓
```

本方案只验收实例分割轮廓质量，不验收粒径、面积、粒度分布、物理尺度换算或报告统计。这些属于下游粒度分析软件，不作为模型训练和 C++ 后处理正确性的判断标准。

外部数据源筛选、泛化风险、合成数据和不确定性策略见 [外部数据源与泛化策略.md](外部数据源与泛化策略.md)。本方案只引用其中的稳定规则，不把数据源调研记录并入主流程。

## 2. 固定技术契约

- 模型输入尺寸：V1 固定 1024，ONNX 输出 stride 固定为 4。
- 模型输出通道：`dy, dx, cellprob`。
- GT flow 通道：`cellprob, dy, dx`。
- 训练期 `dy/dx` 目标使用 `flow_scale=5.0`，即模型预测与 `5 * GT_flow` 计算 loss。
- 推理期 FlowDynamics 跟随 flow 前必须对 `dP` 执行 `/5`，与训练期 `flow_scale=5.0` 配套。
- Euler 前景 gate 固定为 `cellprob logit > 0.0`，不可作为调参项。
- `fd_cellprob_threshold` 是 Euler 完成后的实例级 mean-logit filter，不等同于像素级前景 gate。
- `fd_flow_threshold` 使用 Cellpose 语义：mask 内 `mean((mask_flow - net_flow/5)^2)`，单像素误差为 `ddy^2 + ddx^2`，不做平方根。
- tile 策略以当前 Python/C++ 一致实现为准：长边大于 1536 的训练/验证大图切成 1024 tile，最后一行/最后一列 tile 贴右/下边界；不足 1024 的 tile 才 padding。
- `flow_inference_config.json` 是 C++ 后处理默认参数的唯一正式来源；recipe 只能显式覆盖运行时允许覆盖的字段。

## 3. 数据 gate 摘要

数据采集、样本类型、推荐数量、场景覆盖、元数据填写和标注质量要求以 [数据准备手册.md](数据准备手册.md) 为准。本章只保留正式训练前必须阻断的 gate 摘要。

正式 V1 训练必须满足：

- 数据范围符合 [数据准备手册.md](数据准备手册.md)：只使用材料/粉体颗粒显微图，不混入非目标域图片。
- 训练、验证和 holdout 图片只包含颗粒显微视野 ROI；实际推理必须先确定主分析 ROI，ROI 内仍需排除的标尺、标尺文字、其它文字、污染或遮挡区域通过 ignore regions 传给运行时屏蔽。
- 原始已标注完整图片 ≥ 150 张；场景覆盖达到数据准备手册的最低要求。
- train 样本 ≥ 200，val 样本 ≥ 30，holdout 完整原图 ≥ 10。
- 大图源图 ≥ 5，holdout 大图 ≥ 3。
- `dense + small + round_like + none/few_touching` 这类密集小颗粒关键组合场景必须在 train、val、FlowDynamics 调参样本和 holdout 中都有真实、人工复核样本；不得只在 train 中出现，也不得只用合成图覆盖 val 或 holdout。
- 外部数据集必须先完成标签语义核验和格式转换；可视化叠加图、彩色 overlay、论文截图不得直接作为训练标签。
- 同一来源、同一论文、同一原始大图或同一图切出的 tile 不得跨 train/val/holdout 泄漏。
- train/val 全部为 `label_status=reviewed`。
- V1 必填场景字段 `particle_morphology`、`density_level`、`size_distribution`、`quality_level`、`adhesion_level`、`is_large_image` 均非空、非 `unknown`。
- `microscope_type` 只记录显微镜来源或成像域，用于分层划分、分层分析和诊断；不作为模型推理或 C++ 后处理正确性的硬性 gate，无法可靠确认时可保留 `unknown`。
- 正式 `dataset_manifest.csv` 不得出现 `has_scale`、`pixel_size`、`pixel_size_unit`、`scale_source` 等比例尺相关字段。
- 每张训练/验证图片必须有可读标签和 flow；image、label、flow 空间尺寸必须一致。

## 4. 总体流程

1. 归档上一轮 baseline，只作为对照和伪标签辅助。
2. 整理原始已标注数据目录，形成 `prepare_training_data.py` 可读取的输入。
3. 从原始已标注数据生成 staging 图片、标签和 draft manifest。
4. 人工补全并复核 manifest 场景字段与标注状态。
5. 对大图生成 tile staging，并保持 holdout 大图为完整原图。
6. 按 manifest 分层划分 train/val。
7. 从 train/val 标签生成 flow。
8. 物化 holdout，生成正式 dataset manifest，并运行严格数据 gate。
9. 运行训练前 gate 并打包训练材料。
10. 上传 GPU 环境训练模型。
11. 在验证集上调 FlowDynamics 参数。
12. 构建 C++ 后处理工具，并做 Python/C++ FlowDynamics parity。
13. 使用调参产物导出 ONNX 和 `flow_inference_config.json`。
14. 校验 recipe，并用 C++ 推理验证 ONNX 与后处理。
15. 在 holdout 上验收轮廓与 GT 的重合度。

任何 gate 失败都必须停止当前流程，修复后从失败步骤重跑；不得带着失败产物进入后续步骤。

## 5. 分步骤执行

### 步骤 1：归档 baseline

输入：

- 上一轮 checkpoint、ONNX、训练日志、调参结果、Python/C++ parity 报告。

动作：

- 将上一轮产物放入 `experiments/2026-05-first-training/`。
- 记录 baseline 指标和已知限制。

输出：

- 可追溯 baseline 目录。

通过标准：

- baseline 不作为 V1 训练目标，只用于对照、回归和伪标签辅助。

失败处理：

- 缺少关键 baseline 文件时，继续 V1 训练不受阻，但必须在实验记录中标明无法对照的内容。

### 步骤 2：整理原始已标注数据目录

输入：

- 已完成人工标注或已复核的颗粒图片与实例标签。
- 样本类型、数量、场景覆盖、标注质量和原始目录格式必须满足第 3 章摘要及 [数据准备手册.md](数据准备手册.md)。

动作：

- 按 [数据准备手册.md](数据准备手册.md) 第 6 章建立原始已标注数据根目录。
- 每个子目录表示一个来源批次或一个数据集子集，优先使用通用实例标签格式。
- 历史格式（EMPS、NIST、nNPipe、TiO2）必须满足手册列出的目录结构，才能由准备脚本自动识别。
- 接入外部数据集前必须确认标签 id 是否等价于实例 id；二值 mask、语义 mask、复用 id、polygon JSON 或可视化 overlay 必须显式转换并复核。

输出：

- 一个可作为 `--src-root` 传入的原始已标注数据根目录。

通过标准：

- 目标图片都有对应标签。
- 标签可读，且至少包含一个实例。
- 每对 image 与 label 空间尺寸（高度、宽度）严格一致。
- 同一子目录内不存在会映射到同一 stem 的重复标签。
- 原始图片数量、场景覆盖和大图数量达到第 3 章及数据准备手册的最低要求；否则只能做小样本流程验证。
- 如果使用二值 mask 历史格式，脚本会用连通分量生成实例标签；这类样本后续必须人工复核，确认粘连颗粒没有被错误合并。
- 如果外部标签复用 id 表示多个不连通颗粒，必须按数据集官方转换逻辑或等价规则拆成独立实例后再进入 staging。

失败处理：

- 缺标签、标签命名不一致、实例 id 错误、图片不可读时，先修原始目录；不要进入下一步。
- `--no-strict-pairing` 只允许临时调试使用，正式流程禁止使用。

### 步骤 3：准备 staging 数据

输入：

- 步骤 2 整理好的原始已标注数据根目录。

动作：

```powershell
python -m model_flow.prepare_training_data `
  --src-root <已标注数据根目录> `
  --subset "*" `
  --out-img-dir temp/staging/images `
  --out-label-dir temp/staging/labels `
  --out-manifest temp/staging/dataset_manifest_draft.csv `
  --force
```

输出：

- `temp/staging/images`
- `temp/staging/labels`
- `temp/staging/dataset_manifest_draft.csv`

通过标准：

- 每张进入 staging 的图片都有对应 `*_labels.png`。
- strict/default 流程中缺 label、重复 label key、无法配对或空输出必须失败。

失败处理：

- 修复原始数据命名、标签缺失或标注格式后重跑本步骤。

### 步骤 4：补全并复核 manifest

输入：

- `temp/staging/dataset_manifest_draft.csv`

动作：

- 按 [数据准备手册.md](数据准备手册.md) 第 4 章的元数据填写规则，人工补全以下 V1 必填场景字段：
  - `particle_morphology`
  - `density_level`
  - `size_distribution`
  - `quality_level`
  - `adhesion_level`
  - `is_large_image`
- 记录 `microscope_type` 作为来源/成像域分层字段；无法可靠确认时可保留 `unknown`，并在 `source` 或 `notes` 中写明来源线索。
- 对正式 train/val 候选样本，将 `label_status` 设为 `reviewed`。
- 不得在 dataset manifest 中创建或保留 `has_scale`、`pixel_size`、`pixel_size_unit`、`scale_source` 等物理尺度列。

输出：

- 场景字段完整、标注状态明确的 draft manifest。

通过标准：

- train/val 候选样本的必填场景字段无空值、无 `unknown`；`microscope_type` 可为 `unknown`，但应在 `source` 或 `notes` 中保留来源/成像域线索。
- train/val 候选样本 `label_status=reviewed`。

失败处理：

- 回到标注或元数据编辑环节，补齐字段并重新复核。

### 步骤 5：生成 tile staging

输入：

- `temp/staging/images`
- `temp/staging/labels`
- 已复核的 `temp/staging/dataset_manifest_draft.csv`

动作：

```powershell
python -m model_flow.data.prepare_tiled_staging `
  --manifest temp/staging/dataset_manifest_draft.csv `
  --img-dir temp/staging/images `
  --label-dir temp/staging/labels `
  --out-img-dir temp/staging_tiled/images `
  --out-label-dir temp/staging_tiled/labels `
  --out-manifest temp/staging_tiled/dataset_manifest_draft.csv `
  --background-tile-ratio 0.05
```

输出：

- `temp/staging_tiled/images`
- `temp/staging_tiled/labels`
- `temp/staging_tiled/dataset_manifest_draft.csv`

通过标准：

- 非大图样本原样复制。
- train/val 大图切为 1024 tile，最后一行/最后一列 tile 贴右/下边界。
- tile 行保留源图场景字段，`is_tile=true`，并写入 `source_image_path`、`tile_x`、`tile_y`、`tile_width`、`tile_height`、`tile_overlap`、`tile_core_margin`、`tile_role`。
- 空背景 tile 使用 `tile_role=background_negative`。
- holdout 样本保持完整原图，使用 `tile_role=holdout_full_image`。

失败处理：

- 修复 manifest、缺失标签或 tile 生成错误后重跑本步骤。

### 步骤 6：划分 train/val

输入：

- `temp/staging_tiled/images`
- `temp/staging_tiled/labels`
- `temp/staging_tiled/dataset_manifest_draft.csv`

动作：

```powershell
python -m model_flow.manifest.split_dataset_by_manifest `
  --manifest temp/staging_tiled/dataset_manifest_draft.csv `
  --src-img-dir temp/staging_tiled/images `
  --src-label-dir temp/staging_tiled/labels `
  --output-root data/particles `
  --splits train,val `
  --val-ratio 0.1 `
  --min-val-samples 30 `
  --overwrite
```

输出：

- `data/particles/train`
- `data/particles/val`
- 对应 train/val 标签文件

通过标准：

- 同一源图生成的多个 tile 不跨 train/val 泄漏。
- val 覆盖关键场景组合。
- 目标 split 目录必须为空，或使用 `--overwrite` 进行受控重建（会清空目标目录）。

失败处理：

- 调整 manifest 或 val 比例后重跑本步骤。
- 若目标目录非空且未传 `--overwrite`，清理后重跑。

### 步骤 7：生成 flow

输入：

- `data/particles/train`
- `data/particles/val`
- 对应 `*_labels.png`

动作：

```powershell
python -m model_flow.data.convert_labels_to_flows `
  --label-dir data/particles/train `
  --img-dir data/particles/train `
  --tile-manifest temp/staging_tiled/dataset_manifest_draft.csv `
  --out-dir data/particles/flows_train

python -m model_flow.data.convert_labels_to_flows `
  --label-dir data/particles/val `
  --img-dir data/particles/val `
  --tile-manifest temp/staging_tiled/dataset_manifest_draft.csv `
  --out-dir data/particles/flows_val
```

输出：

- `data/particles/flows_train`
- `data/particles/flows_val`

通过标准：

- 每个 train/val 图片都有对应 flow。
- flow 文件通道语义为 `(cellprob, dy, dx)`。

失败处理：

- 修复标签文件、空 mask 或配对问题后重跑本步骤。

### 步骤 8：物化 holdout，生成正式 manifest 和 readiness report

输入：

- train/val 图片、标签、flow。
- holdout 原图和 GT 标签，来自 `temp/staging_tiled/dataset_manifest_draft.csv` 中 `split=holdout` 或 `tile_role=holdout_full_image` 的行。
- `temp/staging_tiled/dataset_manifest_draft.csv`

动作：

```powershell
python -m model_flow.manifest.split_dataset_by_manifest `
  --manifest temp/staging_tiled/dataset_manifest_draft.csv `
  --src-img-dir temp/staging_tiled/images `
  --src-label-dir temp/staging_tiled/labels `
  --output-root temp/holdout_dataset `
  --splits holdout `
  --overwrite

python -m model_flow.manifest.init_dataset_manifest `
  --out data/particles/dataset_manifest.csv `
  --base-manifest temp/staging_tiled/dataset_manifest_draft.csv `
  --base-dir . `
  --train-img-dir data/particles/train `
  --val-img-dir data/particles/val `
  --holdout-img-dir temp/holdout_dataset/holdout `
  --train-flow-dir data/particles/flows_train `
  --val-flow-dir data/particles/flows_val `
  --train-label-dir data/particles/train `
  --val-label-dir data/particles/val `
  --holdout-label-dir temp/holdout_dataset/holdout `
  --label-status prelabelled `
  --overwrite

python -m model_flow.manifest.check_dataset_manifest `
  --manifest data/particles/dataset_manifest.csv `
  --base-dir . `
  --require-flow-for-splits train,val `
  --require-reviewed-for-splits train,val `
  --strict-scene-fields `
  --out temp/dataset_manifest_check.json

python -m model_flow.manifest.dataset_readiness_report `
  --manifest data/particles/dataset_manifest.csv `
  --base-dir . `
  --require-flow-for-splits train,val `
  --out data/particles/dataset_readiness_report.md
```

说明：`--label-status prelabelled` 只是未能从 `--base-manifest` 匹配到的行的失败默认值。正常正式流程中，train/val 的 `label_status=reviewed` 会从步骤 4/5 的 base manifest 继承；如果继承失败，后续 `--require-reviewed-for-splits train,val` 会拦截。

输出：

- `temp/holdout_dataset/holdout`
- `data/particles/dataset_manifest.csv`
- `temp/dataset_manifest_check.json`
- `data/particles/dataset_readiness_report.md`

通过标准：

- manifest 严格校验无 blocker。
- readiness report 与正式 manifest gate 结论一致。
- 正式 manifest 中包含 holdout 行，且 holdout 标签路径可读。
- 第 3 章数据 gate 摘要中的所有阻断项全部 PASS。

失败处理：

- 按 blocker 回到数据、标签或 manifest 编辑环节修复。

### 步骤 9：训练前 gate 和打包

输入：

- `data/particles`
- requirements、schema、离线 DINOv3 权重、代码。

动作：

```powershell
python scripts/verify_training_pairs.py
python scripts/run_pretrain_gates.py --skip-pack
python scripts/pack_for_modelscope.py --output temp/particles_flow_train.tar.gz
```

输出：

- `temp/particles_flow_train.tar.gz`

通过标准：

- 所有训练前 gate 通过。
- 打包脚本成功输出训练包。

失败处理：

- 不上传 GPU；修复失败 gate 后重跑。

### 步骤 10：GPU 训练

输入：

- 训练包。

动作：

```bash
bash setup_env_modelscope.sh
python -m model_flow.flow_train --device cuda
```

输出：

- checkpoint，包括 `best.pth`。
- 训练日志。

通过标准：

- loss 正常下降。
- 验证集实例分割指标（基于 val `*_labels.png` 真实 GT label，非 flow 反推）可用于选择 checkpoint。
- best checkpoint 的主判据为真实 GT label 的 `mask_instance_f1`。
- val 必须包含 `dense + small + round_like + none/few_touching` 关键组合样本；这类样本上的小颗粒召回、实例数量误差、边界贴合度和误合并情况需要单独查看，不能只看全局平均指标。
- val `*_labels.png` 缺失时训练必须失败，不允许回退到 flow-derived GT mask。
- 训练没有通道错位、flow 配对缺失或 NaN。

失败处理：

- 先查数据与 flow，再查训练配置；不要因单个样本直接改模型结构。

### 步骤 11：调 FlowDynamics 参数

输入：

- `best.pth`
- `data/particles/val`
- `data/particles/flows_val`
- val GT labels。

动作：

```powershell
python -m model_flow.alignment.tune_flow_dynamics `
  --checkpoint checkpoints/best.pth `
  --img-dir data/particles/val `
  --flow-dir data/particles/flows_val `
  --gt-labels-dir data/particles/val `
  --device cuda `
  --out temp/flow_dynamics_best_params.json
```

输出：

- `temp/flow_dynamics_best_params.json`

通过标准：

- 输出包含最佳参数、GT 来源、候选结果和 `tie_count_at_best_score` 等并列最优信息。
- 搜索指标只用于选择后处理参数，不替代最终 holdout 轮廓验收。
- 调参样本必须覆盖 `dense + small + round_like + none/few_touching` 关键组合；参数选择不能通过降低 Euler 前景 gate 来弥补模型未识别小颗粒的问题，Euler 前景 gate 仍遵守第 2 章固定契约。

失败处理：

- GT label 缺失时默认失败；只有调试可显式使用 `--allow-label-fallback`。

### 步骤 12：构建 C++ 工具并做 Python/C++ FlowDynamics parity

输入：

- 调参产物。
- 分层样本列表或 val 样本。
- C++ 编译环境：Qt CMake、Visual Studio 2022、third_party 下的 OpenCV/ONNX Runtime/nlohmann。

动作：

先构建 C++ 可执行文件：

```powershell
& "C:\Qt6.10.2\Tools\CMake_64\bin\cmake.exe" `
  -S model_flow/inference_cpp `
  -B model_flow/inference_cpp/build `
  -G "Visual Studio 17 2022" `
  -A x64

& "C:\Qt6.10.2\Tools\CMake_64\bin\cmake.exe" `
  --build model_flow/inference_cpp/build `
  --config Release
```

再执行 Python/C++ FlowDynamics parity：

```powershell
python -m model_flow.alignment.align_flow_dynamics `
  --checkpoint checkpoints/best.pth `
  --img-dir data/particles/val `
  --flow-params temp/flow_dynamics_best_params.json `
  --use-cpp-alignment `
  --report temp/flow_dynamics_cpp_parity.json
```

输出：

- `model_flow/inference_cpp/build/Release/flow_dynamics_align.exe`
- `model_flow/inference_cpp/build/Release/flow_inference.exe`
- `temp/flow_dynamics_cpp_parity.json`

通过标准：

- C++ 构建成功，两个可执行文件均存在且能启动。
- 同一模型输出和同一 FlowDynamics 参数下，Python mask 与 C++ mask 高度一致。
- flow error、fill holes、min size、mean-logit filter 的行为无语义偏差。

失败处理：

- 构建失败时先修 CMake、依赖路径或编译错误。
- parity 失败时先修 C++ FlowDynamics 或参数读取，不允许带着 parity 偏差导出正式 ONNX。

### 步骤 13：导出 ONNX

输入：

- `checkpoints/best.pth`
- `temp/flow_dynamics_best_params.json`

动作：

```powershell
python -m model_flow.flow_export_onnx `
  --checkpoint checkpoints/best.pth `
  --output onnx `
  --flow-params temp/flow_dynamics_best_params.json
```

输出：

- `onnx/backbone.onnx`
- `onnx/neck_head.onnx`
- `onnx/flow_inference_config.json`

通过标准：

- 导出脚本必须显式接收 `--flow-params`。
- `flow_inference_config.json` 包含预处理、`output_stride=4`、Euler gate、FlowDynamics 参数。

失败处理：

- 不使用无意识默认参数导出；修复调参产物或 checkpoint 后重跑。

### 步骤 14：C++ recipe 校验与推理

输入：

- ONNX 目录。
- `analysis_recipe.json`。
- C++ 推理程序。

动作：

```powershell
python scripts/validate_recipe.py analysis_recipe.json
```

随后使用 C++ 程序读取 ONNX 目录和 recipe 执行推理。

输出：

- C++ mask 输出。
- `_metadata.json`。
- 如使用 tile 推理，输出 `tile_merge_report.json`。

通过标准：

- recipe 通过 JSON Schema 校验。
- C++ 从 `flow_inference_config.json` 读取默认后处理参数。
- recipe override 只覆盖允许覆盖的运行时参数。

失败处理：

- schema 不通过时修 recipe；推理结果异常时先查参数读取和预处理一致性。

### 步骤 15：holdout 轮廓验收

输入：

- 已在 manifest 中标记 `split=holdout` 或 `tile_role=holdout_full_image` 的 holdout 原图。
- holdout GT `*_labels.png`。
- ONNX 目录。
- C++ `flow_inference.exe` 可执行文件。

动作：

步骤 8 已经从 tiled staging 目录中物化 holdout 输入。若 `temp/holdout_dataset/holdout` 缺失，或 holdout manifest/标签更新过，先重新物化：

```powershell
python -m model_flow.manifest.split_dataset_by_manifest `
  --manifest temp/staging_tiled/dataset_manifest_draft.csv `
  --src-img-dir temp/staging_tiled/images `
  --src-label-dir temp/staging_tiled/labels `
  --output-root temp/holdout_dataset `
  --splits holdout `
  --overwrite
```

再执行 C++ tile 推理验收：

```powershell
python -m model_flow.data.eval_tiled_inference `
  --img-dir temp/holdout_dataset/holdout `
  --gt-label-dir temp/holdout_dataset/holdout `
  --onnx-dir onnx `
  --out temp/tiled_holdout_eval.json
```

输出：

- `temp/tiled_holdout_eval.json`

通过标准：

- 使用 `instance_f1`、`precision`、`recall`、`mask_iou_mean` 评价实例 mask 与 GT 的重合度。
- 使用 `boundary_iou_mean` 评价轮廓贴合度。
- 统计 `false_positive_count`、`false_negative_count`、`over_split_proxy_count`，并人工复核粘连未分开样本。
- 对 `dense + small + round_like + none/few_touching` holdout 样本必须单独验收小颗粒召回、实例数量误差、边界外扩和大片误合并；如果这类样本失败，应优先判定为数据覆盖、模型泛化或分辨率瓶颈，而不是用非正式灰度/形态学后处理修补。
- tile 大图必须检查边缘 tile、重叠区、跨 tile 颗粒和实例去重。
- 同一 ONNX 输出和同一 FlowDynamics 参数下，Python 与 C++ mask 结果必须高度一致。

失败处理：

- 先判断是数据/标注、FlowDynamics 参数、C++ parity、tile 合并还是模型泛化问题。
- 只有确认数据和后处理正确但轮廓仍不达标时，才补充针对性数据或重训。

## 6. C++/recipe 运行契约

本章约束 C++ 推理程序如何消费 ONNX 发布包和运行时 recipe。它不替代步骤 12-15，而是给出 C++ 集成时必须遵守的边界。

- C++ 默认后处理参数必须来自 ONNX 目录内的 `flow_inference_config.json`。该文件必须由步骤 13 的导出脚本生成，不得手工散落维护。
- `flow_inference_config.json` 至少包含：`schema_version=1`、`input_size=1024`、`fixed_input_size=true`、`output_stride=4`、`mean`、`std`、`pad_value`、`euler_cellprob_threshold_logit=0.0`、`euler_cellprob_threshold_probability=0.5`、`fd_cellprob_threshold`、`fd_niter`、`fd_min_size`、`fd_flow_threshold`、`fd_max_size_fraction`。
- C++ 程序必须拒绝不支持的 `schema_version`、`fixed_input_size=false`、`output_stride != 4` 或 `euler_cellprob_threshold_logit != 0.0`，防止加载与当前 FlowDynamics 不匹配的模型。
- 运行时采用“主分析 ROI + ignore regions”两层输入机制。主分析 ROI 必须与训练输入域一致，只包含颗粒显微视野；若用户提供带信息栏的完整原图，Qt 应用层必须先确定并裁剪主分析 ROI，再调用模型和 FlowDynamics。
- ignore regions 表示主分析 ROI 内不参与分析的矩形区域，例如 ROI 内残留的标尺、标尺文字、其它文字、污染、遮挡或不希望统计的局部区域。ignore regions 使用主分析 ROI 坐标；如果用户在完整原图上框选，Qt 应用层必须减去 ROI 偏移后再传给 C++。
- C++ 不应通过推理前涂黑或涂白图片来实现 ignore regions。推荐在模型输出后、FlowDynamics 前把 ignore regions 内的 `cellprob` 置为背景低 logit、`dy/dx` 置零；FlowDynamics 输出后再把 ignore regions 内的 mask 强制清零。tile 推理时，应先将全局 ignore regions 与每个 tile 求交并映射到 tile 局部坐标，合并后再按全局 ignore mask 清零一次。
- 如需在完整原图上显示结果，应使用 ROI 偏移把 mask 坐标映射回原图；ignore regions 内保持背景。
- `analysis_recipe.json` 是运行时分析配置，不是训练 manifest。比例尺、最小统计粒径、边界颗粒策略等属于 recipe 或 Qt 应用层，不得回写到 `dataset_manifest.csv`。
- recipe override 只能覆盖 C++ 明确允许的字段：`resolved_parameters.flow_dynamics` 中的 `fd_niter`、`fd_min_size`、`fd_flow_threshold`、`fd_cellprob_threshold`、`fd_max_size_fraction`，以及 `resolved_parameters.runtime` 中的 `tile_size`、`tile_overlap`、`tile_core_margin`、`edge_touch_margin_px`、`boundary_particle_policy`、`ignore_regions`。
- `fd_min_size` 在 C++ 中是已换算的像素面积值。物理单位到像素面积的换算应由 Qt 应用层根据 recipe 的 scale 与用户输入完成；C++ 只消费换算后的像素值。
- C++ tile 推理默认使用 `tile_size=1024`、`tile_overlap=256`、`tile_core_margin=128`、`long_side_limit=1536`；如 recipe 覆盖 tile 参数，必须保留与训练/验证策略可解释的一致性，并在输出 metadata 或 tile merge report 中留痕。
- 每次正式发布或替换 C++ 后处理实现后，必须重新运行步骤 12 的 Python/C++ parity；不能沿用旧 parity 报告。

## 7. 禁止事项

- manifest 或 flow pair gate 未通过，禁止训练。
- `label_status` 不是 `reviewed` 的样本禁止进入正式 train/val。
- 未调 FlowDynamics 参数，禁止正式导出 ONNX。
- 未完成 Python/C++ parity，禁止宣称 C++ 后处理正确。
- 不得将带信息栏、比例尺文字、倍率文字、仪器 UI、报告文字或黑白边框的完整原图直接作为训练、验证、holdout 或主分析 ROI 输入。
- 不得把 ignore regions 写入训练 manifest，也不得通过修改训练图片来模拟运行时忽略区域。
- 不得把粒径、面积、粒度分布、PSD 或物理尺度换算作为本方案验收指标。
- 不得手工散落填写 C++ 后处理参数；正式参数必须来自 ONNX 配套配置和显式 recipe override。
- 不得将完整大图直接缩小到 1024 代替 tile 验证。

## 8. 最终交付物

- V1 数据 manifest 和 readiness report。
- 训练 checkpoint 和训练日志。
- FlowDynamics 调参产物。
- Python/C++ parity 报告。
- ONNX 目录：`backbone.onnx`、`neck_head.onnx`、`flow_inference_config.json`。
- C++ 推理输出、metadata 和 tile merge report。
- holdout 轮廓验收报告。

只有上述交付物全部通过对应 gate，才视为方案目标达成。
