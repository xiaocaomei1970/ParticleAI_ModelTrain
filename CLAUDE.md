## 项目

基于 **ConvNeXt-S DINOv3** backbone，训练一个 Flow Field 预测模型。模型输出每个像素的 (dy, dx, cellprob)，通过 FlowDynamics（Cellpose-inspired 自定义 flow segmentation 后处理，行为与 Cellpose 官方 `compute_masks` 对齐）生成实例分割 mask。

## 开发工具

- Qt 6.10.2，安装目录：`C:\Qt6.10.2`
- Visual Studio 2022，安装目录：`C:\Program Files\Microsoft Visual Studio\2022\Community`
- CMake，安装目录：`C:\Qt6.10.2\Tools\CMake_64`

## 第三方库和模型

- onnxruntime，目录：`third_party/onnxruntime`
- OpenCV，目录：`third_party/opencv`
- nlohmann/json，目录：`third_party/nlohmann`

## C++编程规范

- 类的成员变量加前缀 m_ ;
- 类、结构体、变量和函数命名，不要用缩写，要从命名就看出变量或函数的用途；
- 类的声明和实现要分开放在.h/cpp文件里面，不要放在同一个文件里面；
- 贯彻面向对象编程的单一职责原则；
- 同一个问题，可能需要选择不同实现方式的时候，贯彻面向对象编程的面向接口编程原则，并通过外部配置选择使用哪种实现方式；

## 规则

- 临时文件统一存放到 temp/ 目录下；

## 代码审查规范

### 审查数据管线代码的强制检查项

1. **每个 `os.listdir` / `Path.iterdir` 调用点**：列出该目录下实际会出现的所有文件类型（包括 `*_labels.png`），验证过滤条件是否完备，确认不会被标签文件、临时文件污染。
2. **每个 `os.makedirs` / `mkdir` 调用点**：确认在 `shutil.copy2` / `imwrite` 等写文件操作之前执行，避免干净目录下 `FileNotFoundError`。
3. **模拟端到端数据流**：用一个最小样例（1 张图 + 1 个标签）走通整条管线，而不是只验证孤立函数。

### 涉及第三方库数据语义的修改

1. **先读权威源码再下结论**：涉及 Cellpose 输出格式、PyTorch 张量布局、ONNX 算子行为等第三方库的数据语义时，必须先读该库的相关源码，确认数据格式和语义约定，不依赖直觉类比。
2. **验证前提而非推理链**：对"数学上正确的修复"，必须额外验证前提假设——先问"这个数据是什么含义"，再问"这个变换对不对"。
3. **Agent 发现的关键问题**：不直接执行修复，先独立确认核心前提（特别是涉及第三方库语义的），避免把 Agent 分析当最终结论。

### 优先级原则

- 数据管线的基础正确性（文件枚举、目录创建、配对验证）> 参数默认值对齐
- 阻断性 bug（会导致数据丢失、流程崩溃）> 文档格式不一致
- 语义正确性 > 数学形式正确性

