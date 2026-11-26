# YOAO Plugins

English | [简体中文](README.zh-CN.md)

Official plugin repository for YOAO (You Only Annotate Once), maintaining the platform's inference, trainer, notification channel, and format adapter plugins.

The plugin development SDK and shared inference runtime live in [YOAO-Labs/Plugin-SDK](https://github.com/YOAO-Labs/Plugin-SDK). See each plugin directory's `README.md` for supported capabilities, configuration, and usage.

## Repository contents

### Inference plugins (`plugins/inference/`)

| Plugin | Description |
| --- | --- |
| [LibreYOLO](plugins/inference/libreyolo/README.md) | Object detection inference plugin based on LibreYOLO (YOLOv9), with no Ultralytics dependency |
| [YOLO](plugins/inference/yolo/README.md) | Multi-task inference plugin based on Ultralytics YOLO supporting detection, instance segmentation, pose estimation, oriented bounding boxes (OBB), open-vocabulary detection and segmentation, visual exemplar prompts, and video tracking |
| [RapidOCR](plugins/inference/rapidocr/README.md) | ONNX-based OCR inference plugin built on RapidOCR for text detection, text recognition, and end-to-end OCR |
| [ONNX Tools](plugins/inference/onnxtools/README.md) | Two-stage vehicle attribute inference plugin built on ONNX Tools for vehicle detection and vehicle type and color classification |
| [Grounded-SAM2](plugins/inference/grounded-sam2/README.md) | Open-vocabulary detection, interactive image segmentation, and video tracking plugin based on GroundingDINO and SAM 2.1 |
| [SAM3](plugins/inference/sam3/README.md) | Text-prompted detection and segmentation, interactive image segmentation, and video tracking plugin based on SAM 3 and SAM 3.1 |

### Trainer plugins (`plugins/trainer/`)

| Plugin | Description |
| --- | --- |
| [Ultralytics YOLO](plugins/trainer/ultralytics/README.md) | Object detection and instance segmentation trainer plugin based on Ultralytics YOLO, supporting YOLOv8, YOLO11, YOLO26, and offline augmentation preview (runtime dependency is AGPL-3.0-only) |
| [LibreYOLO](plugins/trainer/libreyolo/README.md) | Object detection trainer plugin based on LibreYOLO (LibreYOLO9), MIT-licensed with no Ultralytics dependency |

### Notification channel plugins (`plugins/notification-channel/`)

| Plugin | Description |
| --- | --- |
| [Feishu DM](plugins/notification-channel/feishu-dm/README.md) | Zero-dependency Node.js Feishu direct message plugin that sends task and job notification cards to a bound user's `open_id` |

### Format adapter plugins (`plugins/format-adapter/`)

| Plugin | Description |
| --- | --- |
| [LabelMe JSON](plugins/format-adapter/labelme-json/README.md) | Zero-dependency Node.js LabelMe JSON format adapter supporting rectangle and polygon annotation import and export |

## License

First-party source code is licensed under [MIT](LICENSE). Third-party dependencies, upstream submodules, and model weights remain subject to their own licenses. This repository does not include model weights, training artifacts, datasets, or credentials.
