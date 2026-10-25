# Cone Annotation Tool · 三色锥桶标注与检测

面向 Windows 的本地 YOLO11 图片、视频检测与标注工作台，支持红、蓝、黄三类锥桶。

## 功能

- 从本机任意目录导入图片、视频；原文件保留。
- 三色置信度分别设置，支持 640 / 960 / 1280 推理尺寸。
- 图片预标注、拖动新增框、移动、缩放、撤销与人工审核。
- 视频逐帧检测，导出带英文名称的 MP4V 视频、图片、CSV、JSONL 和图片 YOLO 标签。
- 基于锥桶高度及水平视场角估算前方 Z 和左右 X（右正左负），单位米。
- 提供数据审核、按编号分组划分及 YOLO11 训练脚本。

所有媒体在本机处理。服务仅监听 `127.0.0.1`，不是公开网络服务。

## Windows 快速开始

需要 Windows 10/11 x64、Python 3.12 和联网安装依赖。无需安装 CUDA Toolkit；使用 GPU 时需要兼容驱动的 NVIDIA 显卡。

```powershell
git clone https://github.com/subing509/cone-annotation-tool.git
cd cone-annotation-tool
# CPU 版本
powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
# 或安装 CUDA 12.4 版本（不要同时执行两个安装选项）
# powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1 -CUDA
```

将你有权使用的三类检测模型放到 `cone_project/models/best.pt`。类别顺序必须是：

| 编号 | 名称 | 画面标签 |
|---|---|---|
| 0 | red 或 red_cone | red_cone |
| 1 | blue 或 blue_cone | blue_cone |
| 2 | yellow 或 yellow_cone | yellow_cone |

此仓库不分发数据集、原项目微调权重或多 GB 运行环境。通用 COCO 模型不能直接替代三类锥桶模型。未配置模型时，可以打开界面和人工标注，但不能运行检测。

双击 `start.cmd`，或运行：

```powershell
.\.venv-yolo11\Scripts\python.exe .\cone_project\tool_server.py
```

浏览器访问 <http://127.0.0.1:8765/>。命令行启动时使用 Ctrl+C 停止；后台启动后关闭浏览器不会停止服务。端口占用时请先关闭已有服务。

## 使用及结果

1. 点击“选择本机视频 / 图片”，导入后自动选中。
2. 分别填写红、蓝、黄阈值；降低阈值会增加误检风险。
3. 图片可点击“模型预标注”，人工修改后保存审核副本；视频使用“检测所选媒体”。
4. 输出位于 `cone_project/tool_data/exports/<任务编号>/`，标注媒体在 `annotated/`。
5. 导入媒体保存在 `tool_data/imports/`，请预留磁盘空间。每个导入文件最大 20 GB。

预测标签标记为未审核；导出的人工审核标签可用于后续训练。视频采用 MP4V，部分浏览器无法直接播放，请使用支持该编码的本地播放器。

## 位置估算的含义

默认锥桶高度 0.45 m、水平视场角 90° 都是假设，可在界面修改。计算假设完整锥桶可见、像素为正方形且光心居中。Z 为相机光轴方向深度；X 为左右偏移，不是地图坐标或精确地面距离。没有修正俯仰、遮挡、畸变。边缘截断或框高不足 6 像素显示 `Position: N/A`。

CSV/JSONL 每个目标包含 `x_m`、`z_m`、`dist_m`（等于 Z）、`position_valid` 和 `dist_source`；任务元数据记录所用参数。旧输出需重新检测才会应用新的参数。

## 训练

将自有 JPG 图片置于 `JPEGImage/`，同名 YOLO 检测标签置于 `label/`。标签每行为 `class_id x_center y_center width height`，坐标归一化到 0–1。

```powershell
.\.venv-yolo11\Scripts\python.exe .\cone_project\prepare_data.py
.\.venv-yolo11\Scripts\python.exe .\cone_project\resplit_data.py
```

准备可合法使用的 YOLO11n 预训练权重 `cone_project/models/yolo11n.pt`，然后运行：

```powershell
.\.venv-yolo11\Scripts\python.exe .\cone_project\train_yolo11.py --smoke
.\.venv-yolo11\Scripts\python.exe .\cone_project\train_yolo11.py
```

训练脚本要求 CUDA；支持 `--resume` 恢复含优化器状态的检查点。训练结果写入 `cone_project/runs/`。将最终三类模型复制到 `models/best.pt` 后重启服务。

划分脚本使用编号分组和相邻组边界隔离，编号不一定对应真实视频来源。建议按视频或采集场景划分训练、验证和测试，避免相邻帧泄漏。请使用新的独立视频评估效果，本仓库不宣称通用识别精度。

## 已知限制

- 侧箱等非锥桶物体可能误检为蓝锥；部分黄色锥桶可能误判为红色。需要补充困难负样本、正确标注并微调模型。
- 推理任务串行执行；CPU 检测视频可能较慢。
- 本工具面向本机使用，不提供多用户身份认证或公网部署保护。
- 模型 `.pt` 文件仅应来自可信来源。

## 开发验证

```powershell
.\.venv-yolo11\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖三色阈值、坐标校验、位置几何和没有权重时的界面接口；不需要私有数据或 GPU。

## 许可证

本项目源码采用 [GNU AGPL v3](LICENSE)（`AGPL-3.0-only`）。使用的 Ultralytics 软件也采用 AGPL-3.0；其他依赖保留各自许可证。项目许可证不授予第三方数据或权重的使用权。
