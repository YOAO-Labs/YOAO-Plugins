# Ultralytics YOLO 推理插件

[English](README.md) | 简体中文

`yoao-yolo-plugin` 是基于 Ultralytics YOLO `8.4.183` 与固定版本 Ultralytics CLIP 分支实现的 YOAO 多任务推理插件。插件遵循 YOAO ML Backend 协议 v2.2，提供模型池化的 PyTorch 推理运行时，同时支持官方预训练模型与平台受管训练模型。

## 能力与模型

| `/setup` 模型 ID | 任务 | 提示类型 | 输入 | 输出 | 支持的系列与规格 |
| --- | --- | --- | --- | --- | --- |
| `detect` | `detect` | `none` | `full_image`、`crop` | `bbox` | `yolov8`（`n/s/m/l/x`）、`yolov9`（`t/s/m/c/e`）、`yolov10`（`n/s/m/b/l/x`）、`yolo11`（`n/s/m/l/x`，默认 `yolo11`/`s`）、`yolo12`（`n/s/m/l/x`）、`yolo26`（`n/s/m/l/x`）、`rtdetr`（`l/x`） |
| `segment` | `segment` | `none` | `full_image`、`crop` | `polygon`（受管模型输出原生 RLE） | `yolov8`（`n/s/m/l/x`）、`yolov9`（`c/e`）、`yolo11`（`n/s/m/l/x`，默认 `yolo11`/`s`）、`yolo26`（`n/s/m/l/x`） |
| `pose` | `pose` | `none` | `full_image`、`crop` | `keypoint` | `yolov8`、`yolo11`（默认 `yolo11`/`s`）、`yolo26`（`n/s/m/l/x`） |
| `obb` | `obb` | `none` | `full_image`、`crop` | `rotated_bbox` | `yolov8`、`yolo11`（默认 `yolo11`/`s`）、`yolo26`（`n/s/m/l/x`） |
| `detect-world` | `detect` | `text` | `full_image`、`crop` | `bbox` | `yolo-worldv2`（默认 `s`）、`yolo-world`（`s/m/l/x`） |
| `detect-yoloe` | `detect` | `text` | `full_image`、`crop` | `bbox` | `yoloe-v8`（`s/m/l`）、`yoloe-11`（默认 `s`，`s/m/l`）、`yoloe-26`（`n/s/m/l/x`） |
| `segment-yoloe` | `segment` | `text` | `full_image`、`crop` | `polygon`（`mask` / `both`） | `yoloe-v8`（`s/m/l`）、`yoloe-11`（默认 `s`，`s/m/l`）、`yoloe-26`（`n/s/m/l/x`） |
| `exemplar-yoloe` | `interactive_seg` | `exemplar` | `full_image` | `bbox`、`polygon` | `yoloe-v8`（`s/m/l`）、`yoloe-11`（默认 `s`，`s/m/l`）、`yoloe-26`（`n/s/m/l/x`）；支持多正框提示，不支持负框 |
| `track` | `tracker` | `none` | `video` | 带 `track_id` 的 `bbox` | 复用 `detect` 系列与规格，支持 `bytetrack` 与 `botsort`（默认 BoT-SORT 关闭 ReID） |

- 合法的系列与规格组合由 `model_registry.py` 定义，不支持的组合会返回错误。
- 受管 Ultralytics 检测与实例分割训练模型会依据不可变训练回执校验，并支持平台评估请求。

## 模型权重与资产

本仓库与构建镜像均不包含模型权重。`yolo_plugin/pretrained-assets.json` 描述了 30 个受控检测与分割基础模型（`yolov8`、`yolo11`、`yolo26` × `n/s/m/l/x`）以及 6 个推理与文本编码器资产（`yolo11s-pose.pt`、`yolo11s-obb.pt`、`yolov8s-worldv2.pt`、`yoloe-11s-seg.pt`、`ViT-B-32.pt`、`mobileclip_blt.ts`）。

- 插件模式（设置 `AAP_PLUGIN_TOKEN_FILE`）下，可变权重与回执位于 `AAP_PLUGIN_DATA_DIR/checkpoints`，已校验受控资产位于 `AAP_PLUGIN_DATA_DIR/model-assets`。
- 独立模式下默认从 `YOLO_CHECKPOINTS_DIR`（`/app/checkpoints`）读取权重。
- 管理员可使用 `python -m scripts.prepare_assets`（从已审批的本地权重文件准备受控资产）或 `python -m scripts.download_weights`（按需下载闭集官方权重）。
- 设置 `YOLO_STRICT_OFFLINE=1` 可禁止 Ultralytics 联网下载权重；离线使用开放词汇模型时仍须提前准备本地文本编码器并隔离网络。

## 关键配置

容器默认入口为 `uvicorn main:app --host 0.0.0.0 --port 8003 --workers 1`。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `AAP_PLUGIN_TOKEN_FILE` | 空 | 设置后启用插件模式；私有 Bearer 令牌文件路径（不少于 32 个字符） |
| `AAP_PLUGIN_DEVICE` / `YOLO_DEVICE` | `cuda:0` | 运行设备（`cpu` 或 `cuda:0`）；插件模式读取 `AAP_PLUGIN_DEVICE` |
| `AAP_PLUGIN_DATA_DIR` / `YOLO_CHECKPOINTS_DIR` | `/app/checkpoints` | 资产根目录；插件模式使用 `AAP_PLUGIN_DATA_DIR` |
| `YOLO_MODEL_POOL_CAP` | `2` | 模型池最大常驻模型数 |
| `YOLO_BUILD_TIMEOUT` | `30` | 冷启动构建等待超时（秒） |
| `YOLO_IDLE_UNLOAD_SECONDS` | `600` | 空闲自动卸载阈值（秒，`<=0` 关闭） |
| `YOLO_IDLE_CHECK_INTERVAL` | `60` | 空闲检查周期（秒） |
| `YOLO_STRICT_OFFLINE` | `0` | 设为 `1` 时禁止联网下载权重 |
| `YOLO_TRACKER_MAX_FRAMES` | `900` | 单次视频跟踪请求最大帧数 |
| `LOG_LEVEL` | `INFO` | Python 日志级别 |

插件模式同时需要 `AAP_PLUGIN_ID`、`AAP_PLUGIN_VERSION`、`AAP_PLUGIN_REGISTRY_ID` 与 `AAP_ROUTE_REGISTRY_ID`。受管 CUDA 部署还需 `AAP_PLUGIN_GPU_RESOURCE_ID`、`AAP_PLUGIN_GPU_UUID`、`AAP_PLUGIN_GPU_KEYRING_FILE` 以及镜像内置的 `CUDA_VERSION=13.2`；仅当经审批的验收报告与摘要（`AAP_PLUGIN_QUALIFICATION_FILE`、`AAP_PLUGIN_QUALIFICATION_SHA256`）匹配当前部署时，`/setup` 才会声明受管 GPU 生命周期能力。

## 运行环境与依赖

- **Python**：`>=3.11,<3.14`
- **推理栈**：`ultralytics==8.4.183`、固定提交的 Ultralytics CLIP 分支、`torch==2.14.1+cu132`、`torchvision==0.29.1+cu132`、CUDA `13.2`
- **设备策略**：支持 CPU 与 CUDA；受管 CUDA 插件模式拒绝回退 CPU。
- **环境隔离**：顶层平铺模块（`main`、`schemas`、`predictor` 等）须安装在独立 Python 环境中，避免与其他平铺后端冲突。

## 许可证

第一方 YOAO 源码采用 [MIT](LICENSE) 许可证。运行时依赖 `ultralytics==8.4.183` 与固定提交的 Ultralytics CLIP 分支均采用 [AGPL-3.0](AGPL-3.0.txt)（上游源码仓库：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)、[ultralytics/CLIP](https://github.com/ultralytics/CLIP)）。其他传递依赖与模型权重遵循各自条款；组合运行时整体不视为 MIT，使用与分发前须独立评估 AGPL-3.0 及模型授权义务。
