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