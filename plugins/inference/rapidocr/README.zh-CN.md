# RapidOCR 推理插件

[English](README.md) | 简体中文

`yoao-rapidocr-plugin` 是基于 RapidOCR `3.9.0` 与 ONNX Runtime `1.23.2` 实现的 YOAO OCR 推理插件。插件遵循 YOAO ML Backend 协议 v2.2，将 `det -> cls -> rec` 流程拆分为可编排的原子模型与端到端复合模型。

## 能力与模型

| `/setup` 模型 ID | 任务 | 组合形态 | 输入 | 输出 | 可调参数 |
| --- | --- | --- | --- | --- | --- |
| `ocr-det` | `detect` | `atom` | `full_image` | `polygon` 文本框 | `box_thresh`（`0.5`）、`unclip_ratio`（`1.6`） |
| `ocr-rec` | `ocr` | `atom`（内置 `cls` 方向校正） | `crop` | `attributes.text`、`orientation`（`0`/`180`）、`language`（`universal`/`en`） | 无 |
| `ocr-e2e` | `ocr` | `composite`（`det -> cls -> rec`） | `full_image` | `polygon` + `text`、`orientation`、`language` | `box_thresh`（`0.5`）、`unclip_ratio`（`1.6`）、`text_score`（`0.5`） |

- 文本行方向分类（`cls`，0/180 度）内化在 `ocr-rec` 与 `ocr-e2e` 中，不单独暴露；平台编排也可将 `ocr-det` 与 `ocr-rec` 串联使用。
- 通过 `context.model_variants`（`{version, size, lang}`）选择变体，默认为 `v5` / `mobile` / `universal`：
  - **`ocr-det`**：PP-OCR `v5`（`mobile`、`server`）与 `v6`（`tiny`、`small`、`medium`），与语言无关。
  - **`ocr-rec` / `ocr-e2e`**：通用中英（`universal`）支持 `v5`（`mobile`、`server`）与 `v6`（`tiny`、`small`、`medium`）；英文（`en`）仅支持 `v5`（`mobile`）。
  - 内置 `cls` 按尺寸自动选择：`mobile`/`tiny`/`small` 使用 `mobile` 档，`server`/`medium` 使用 `server` 档。

## 模型权重与资产

插件共使用 RapidOCR `3.9.0` 对应的 13 个 ONNX 文件（5 个检测、2 个方向分类、6 个识别）。本仓库与构建镜像均不包含模型权重。

- 插件模式（设置 `AAP_PLUGIN_TOKEN_FILE`）下，权重位于 `AAP_PLUGIN_DATA_DIR/models`，启动时依据 `python -m plugin_assets` 生成的 `data/assets.json` 校验摘要并拒绝未列入清单的文件。
- 独立模式下，从 `RAPIDOCR_MODEL_DIR` 读取权重。
- 管理员可运行 `python -m pretrained_manifest` 查看全部 13 个文件的路径与 SHA-256 清单，或在获准环境中运行 `python -m download_models --models-dir /path/to/models` 显式准备权重。

## 关键配置

容器默认入口为 `python3 -m uvicorn main:app --host 0.0.0.0 --port 8005 --workers 1`。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RAPIDOCR_MODEL_DIR` | GPU `/app/models`；CPU `/var/lib/aap-plugin/models` | 独立模式权重根目录（插件模式改用 `AAP_PLUGIN_DATA_DIR/models`） |
| `RAPIDOCR_DEVICE` | GPU `gpu`；CPU `cpu` | 独立模式运行设备（插件模式读取 `AAP_PLUGIN_DEVICE`） |
| `RAPIDOCR_POOL_CAP` | `3` | 复合引擎池上限（每个引擎持有 `det`/`cls`/`rec` 三个 ORT session） |
| `RAPIDOCR_BUILD_TIMEOUT` | `30` | 冷启动构建等待超时（秒） |
| `RAPIDOCR_IDLE_UNLOAD_SECONDS` | `600` | 整池空闲自动卸载阈值（秒，`<=0` 关闭） |
| `RAPIDOCR_IDLE_CHECK_INTERVAL` | `60` | 空闲检查周期（秒） |
| `LOG_LEVEL` | `INFO` | Python 日志级别 |

受管插件模式同时读取 `AAP_PLUGIN_TOKEN_FILE`、`AAP_PLUGIN_DATA_DIR`、`AAP_PLUGIN_DEVICE` 与 `AAP_PLUGIN_ASSET_MANIFEST_SHA256`；CUDA 部署还需提供 `AAP_PLUGIN_REGISTRY_ID`、`AAP_ROUTE_REGISTRY_ID`、`AAP_PLUGIN_GPU_RESOURCE_ID`、`AAP_PLUGIN_GPU_UUID` 与 `AAP_PLUGIN_GPU_KEYRING_FILE`。

## 运行环境与依赖

- **Python**：`>=3.10`
- **核心依赖**：`rapidocr==3.9.0`、`numpy==2.2.6`、`opencv-python-headless==5.0.0.93`
- **CPU 与 CUDA 配置**（`pyproject.toml` 中互斥 extra）：
  - `cpu`（`Dockerfile.cpu`）：使用 `onnxruntime==1.23.2`，不包含 NVIDIA wheel。
  - `cuda`（`Dockerfile`）：基于 CUDA `12.8` 运行时镜像，使用 `onnxruntime-gpu==1.23.2` 与 `nvidia-cudnn-cu12==9.10.2.21`（支持 PP-OCRv6）。
- **回退策略**：独立 `gpu` 模式仅在明确设备错误时尝试回退 CPU；受管 CUDA 插件模式要求全部业务 session 以 `CUDAExecutionProvider` 为首 provider，拒绝回退 CPU。

## 许可证

第一方插件源码采用 [MIT](LICENSE) 许可证。上游 RapidOCR 及第三方依赖遵循各自许可。

依据 [RapidOCR 模型许可声明](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/)，`PP-OCRv6_det_small.onnx` 与 `PP-OCRv6_rec_small.onnx` 适用 Apache-2.0（须保留 PaddleOCR 归属与声明）。其余 11 个 ONNX 权重文件不在该声明覆盖范围内，使用或再分发前须逐项独立核验授权。
