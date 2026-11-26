# YOAO Plugins

[English](README.md) | 简体中文

YOAO（You Only Annotate Once）的官方插件仓库，集中维护平台使用的推理、训练、通知渠道与格式适配器插件。

插件开发 SDK 与共享推理运行时由 [YOAO-Labs/Plugin-SDK](https://github.com/YOAO-Labs/Plugin-SDK) 维护。各插件的具体能力、配置与使用方式见对应目录内的 `README.zh-CN.md`。

## 仓库内容

### 推理插件（`plugins/inference/`）

| 插件 | 说明 |
| --- | --- |
| [LibreYOLO](plugins/inference/libreyolo/README.zh-CN.md) | 基于 LibreYOLO（YOLOv9）的目标检测推理插件，不依赖 Ultralytics |
| [YOLO](plugins/inference/yolo/README.zh-CN.md) | 基于 Ultralytics YOLO 的多任务推理插件，支持目标检测、实例分割、姿态估计、旋转框（OBB）、开放词汇检测与分割、视觉示例交互及视频跟踪 |
| [RapidOCR](plugins/inference/rapidocr/README.zh-CN.md) | 基于 RapidOCR（ONNX）的 OCR 推理插件，支持文本检测、文本识别与端到端识别 |
| [ONNX Tools](plugins/inference/onnxtools/README.zh-CN.md) | 基于 ONNX Tools 的两阶段车辆属性推理插件，支持车辆检测与车型、颜色属性分类 |
| [Grounded-SAM2](plugins/inference/grounded-sam2/README.zh-CN.md) | 基于 GroundingDINO 与 SAM 2.1 的开放词汇检测、交互式图像分割与视频跟踪插件 |
| [SAM3](plugins/inference/sam3/README.zh-CN.md) | 基于 SAM 3 与 SAM 3.1 的文本检测与分割、交互式图像分割及视频跟踪插件 |

### 训练器插件（`plugins/trainer/`）

| 插件 | 说明 |
| --- | --- |
| [Ultralytics YOLO](plugins/trainer/ultralytics/README.zh-CN.md) | 基于 Ultralytics YOLO 的目标检测与实例分割训练器插件，支持 YOLOv8、YOLO11 与 YOLO26 及离线增强预览（运行时依赖为 AGPL-3.0-only） |
| [LibreYOLO](plugins/trainer/libreyolo/README.zh-CN.md) | 基于 LibreYOLO（LibreYOLO9）的目标检测训练器插件（MIT 许可，无 Ultralytics 依赖） |

### 通知渠道插件（`plugins/notification-channel/`）

| 插件 | 说明 |
| --- | --- |
| [飞书私信（Feishu DM）](plugins/notification-channel/feishu-dm/README.zh-CN.md) | 零依赖 Node.js 飞书私信通知渠道插件，向绑定用户的 `open_id` 发送任务与作业卡片消息 |

### 格式适配器插件（`plugins/format-adapter/`）

| 插件 | 说明 |
| --- | --- |
| [LabelMe JSON](plugins/format-adapter/labelme-json/README.zh-CN.md) | 零依赖 Node.js LabelMe JSON 格式适配器插件，支持矩形框与多边形标注的导入和导出 |

## 许可证

第一方源码采用 [MIT](LICENSE) 许可证。第三方依赖、上游子模块及模型权重继续适用各自许可；本仓库不包含模型权重、训练产物、数据集和凭据。
