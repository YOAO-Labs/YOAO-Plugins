# Grounded-SAM2 推理插件

[English](README.md) | 简体中文

`yoao-grounded-sam2-plugin` 是基于 GroundingDINO 与 SAM 2.1（`IDEA-Research/Grounded-SAM-2`）实现的 YOAO 开放词汇检测、交互式图像分割与有状态视频跟踪插件。插件遵循 YOAO ML Backend 协议 v2.2，维护独立的图像模型池与视频模型池。

## 能力与模型

| `/setup` 模型 ID | 任务 | 组合形态 | 提示类型 | 输入 | 输出 |
| --- | --- | --- | --- | --- | --- |
| `grounded-sam2-detection` | `detect` | `atom`（DINO） | `text` | `full_image`、`crop` | `bbox` |
| `grounded-sam2-segmentation` | `segment` | `composite`（DINO + SAM 2.1） | `text` | `full_image`、`crop` | `polygon`（`mask` / `both`） |
| `grounded-sam2-interactive-seg` | `interactive_seg` | `atom`（SAM 2.1，交互式） | `point`、`interactive_box`、`mask`、`scribble` | `full_image` + 提示 | `polygon`、`mask`（支持多候选与低分辨率 logits） |
| `grounded-sam2-box-seg` | `segment` | `atom`（SAM 2.1，可批处理） | `bbox` | `bbox_prompt`、`full_image` | `polygon` |
| `grounded-sam2-tracker` | `tracker` | `composite`（SAM 2.1 Video，交互式） | `point`、`bbox`、`correction_frame` | `video` + 提示 | `bbox`、`polygon`、`mask`（支持正反向窗口跟踪） |

- **SAM 2.1 规格**（`sam_variant`）：`tiny`（默认）、`small`、`base_plus`、`large`（对应 `sam2.1_hiera_{tiny,small,base_plus,large}.pt`）。
- **GroundingDINO 规格**（`dino_variant`）：`T`（默认，`groundingdino_swint_ogc.pth`）与 `B`（`groundingdino_swinb_cogcoor.pth`）。
- 图像模型池同时构造 SAM 2.1 与 GroundingDINO；视频跟踪器使用独立的 SAM 2.1 视频模型池并支持跨窗口会话状态。

## 模型权重与资产

本仓库与构建镜像均不包含模型权重或分词器文件。服务强制开启 Hugging Face 离线模式（`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`），缺失权重时直接报错而不自动联网下载。`grounded_sam2_plugin/pretrained-assets.json` 记录了 SAM 2.1、GroundingDINO 与 `bert-base-uncased` 文件的 SHA-256 摘要。

- 插件模式（设置 `AAP_PLUGIN_TOKEN_FILE`）下，权重位于 `AAP_PLUGIN_DATA_DIR/checkpoints`。
- 独立模式下，默认从 `CHECKPOINT_DIR`（`/app/checkpoints`）读取权重。
- 所有图像变体均须在 `checkpoints/bert-base-uncased/` 下准备完整的本地 BERT 文件（`config.json`、`tokenizer.json`、`tokenizer_config.json`、`vocab.txt`、`model.safetensors`）。
- 管理员可在独立的获准环境中运行 `scripts/download_checkpoints.py` 下载 SAM 2.1 与 GroundingDINO 权重；BERT 文件须另行准备。

## 关键配置

容器默认入口为 `uvicorn main:app --host 0.0.0.0 --port 8001 --workers 1`。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CHECKPOINT_DIR` | `/app/checkpoints` | 独立模式权重目录（插件模式改用 `AAP_PLUGIN_DATA_DIR/checkpoints`） |
| `SAM_VARIANT` | `tiny` | 默认 SAM 2.1 规格（`tiny`、`small`、`base_plus`、`large`） |
| `DINO_VARIANT` | `T` | 默认 GroundingDINO 规格（`T`、`B`） |
| `BOX_THRESHOLD` | `0.35` | 默认 GroundingDINO 检测框阈值 |
| `TEXT_THRESHOLD` | `0.25` | 默认 GroundingDINO 文本阈值 |
| `EMBEDDING_CACHE_SIZE` | `16` | 交互式分割图像特征缓存容量 |
| `MODEL_POOL_CAP` | `1` | 图像模型池常驻上限 |
| `VIDEO_MODEL_POOL_CAP` | `1` | 视频跟踪模型池常驻上限 |
| `MODEL_POOL_BUILD_TIMEOUT` | `30` | 图像模型冷启动等待超时（秒） |
| `VIDEO_MODEL_POOL_BUILD_TIMEOUT` | `60` | 视频模型冷启动等待超时（秒） |
| `IDLE_UNLOAD_SECONDS` | `600` | 图像模型池空闲卸载阈值（秒） |
| `VIDEO_IDLE_UNLOAD_SECONDS` | `600` | 视频模型池空闲卸载阈值（秒） |
| `VIDEO_TRACKER_MAX_WINDOW_FRAMES` | `300` | 单次视频跟踪窗口最大帧数 |
| `VIDEO_TRACKER_MAX_CONTEXT_FRAMES` | `300` | 有状态视频上下文最大帧数（大于窗口帧数时启用会话模式） |
| `LOG_LEVEL` | `INFO` | Python 日志级别 |

受管插件模式同时读取 `AAP_PLUGIN_TOKEN_FILE`、`AAP_PLUGIN_DATA_DIR`、`AAP_PLUGIN_DEVICE`、`AAP_PLUGIN_ASSET_MANIFEST_SHA256`；CUDA 部署还需 `AAP_PLUGIN_REGISTRY_ID`、`AAP_ROUTE_REGISTRY_ID`、`AAP_PLUGIN_GPU_RESOURCE_ID`、`AAP_PLUGIN_GPU_UUID`、`AAP_PLUGIN_GPU_KEYRING_FILE`、`CUDA_VERSION` 以及经审批的验收报告（`AAP_PLUGIN_QUALIFICATION_FILE` / `AAP_PLUGIN_QUALIFICATION_SHA256`）。

## 运行环境与依赖

- **Python**：`3.10`（`>=3.10,<3.11`）
- **推理栈**：`torch==2.3.1`、`torchvision==0.18.1`、CUDA `12.1`
- **子模块**：构建镜像前需递归初始化 Git 子模块 `vendor/grounded-sam-2`。
- **设备策略**：受管 CUDA 插件模式拒绝回退 CPU。

## 许可证

第一方插件源码采用 [MIT](LICENSE) 许可证。`vendor/grounded-sam-2` 中的上游 SAM 2 与 GroundingDINO 源码继续适用 Apache-2.0，镜像在 `/usr/share/licenses/grounded-sam2-vendor` 下保留上游声明（含 BSD-3-Clause `cctorch` 声明）。

SAM 2.1、GroundingDINO 与 `google-bert/bert-base-uncased` 模型卡片均声明 Apache-2.0；使用或再分发前仍须独立核验模型及第三方依赖条款。
