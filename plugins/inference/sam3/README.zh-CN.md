# SAM3 推理插件

[English](README.md) | 简体中文

`yoao-sam3-plugin` 是基于 SAM 3 与 SAM 3.1（`facebookresearch/sam3`）实现的 YOAO GPU 推理插件。插件遵循 YOAO ML Backend 协议 v2.2，支持文本检测与分割、交互式图像分割、交互式视频跟踪（PVS）及多路文本视频跟踪。

## 能力与模型

| `/setup` 模型 ID | 任务 | 组合形态 | 提示类型 | 输入 | 输出 | 权重文件 |
| --- | --- | --- | --- | --- | --- | --- |
| `sam3-detection` | `detect` | `atom` | `text`（PCS） | `full_image`、`crop` | `bbox` | `checkpoints/sam3.pt` |
| `sam3-segmentation` | `segment` | `composite` | `text`（PCS） | `full_image`、`crop` | `bbox`、`polygon`（`mask` / `both`） | `checkpoints/sam3.pt` |
| `sam3-interactive-seg` | `interactive_seg` | `atom`（交互式） | `point`、`interactive_box`、`mask`、`scribble`、`exemplar` | `full_image` + 提示 | `polygon`、`mask`（支持多候选与低分辨率 logits） | `checkpoints/sam3.pt` |
| `sam3-video-interactive-tracker` | `tracker` | `composite`（PVS，交互式） | `point`、`interactive_box`、`correction_frame` | `video` + 提示 | `bbox`、`polygon`、`mask`（正反向跟踪） | `checkpoints/sam3.pt` |
| `sam3-video-tracker` | `tracker` | `composite`（SAM 3.1 Multiplex） | `text` | `video` | `bbox`、`polygon`、`mask` | `checkpoints/sam3.1_multiplex.pt` |

- 五个模型统一使用 `model_variant=sam3`。BPE 词表内置于上游 `sam3` 包中，本地加载权重无需额外的 Hugging Face 配置文件。
- 视频跟踪以整窗口 JSON 返回结果。持久化 PVS 会话上下文绑定到对应 PVS 模型池条目，空闲卸载、驱逐或终态排空时会立即失效并释放。

## 模型权重与资产

本仓库与构建镜像均不包含模型权重，服务启动时也不会自动下载权重。`sam3_plugin/pretrained-assets.json` 记录了 `checkpoints/sam3.pt`（`facebook/sam3`）与 `checkpoints/sam3.1_multiplex.pt`（`facebook/sam3.1`）的路径、大小与 SHA-256 摘要。

- 插件模式（设置 `AAP_PLUGIN_TOKEN_FILE`）下，权重挂载到 `AAP_PLUGIN_DATA_DIR/checkpoints`。
- 独立模式下，默认从 `CHECKPOINT_DIR`（`/app/checkpoints`）读取权重。
- 上述两个 Hugging Face 仓库均为独立门控（gated）仓库。获准管理员须分别申请访问权限，并可在独立的准备环境中运行 `scripts/download_checkpoints.py`；切勿将访问凭据打入容器镜像。

## 关键配置

容器默认入口为 `uvicorn main:app --host 0.0.0.0 --port 8002 --workers 1`，镜像默认设置 `HOME=/tmp` 以兼容非 root 及只读根文件系统下的内核缓存写入。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CHECKPOINT_DIR` | `/app/checkpoints` | 独立模式权重目录（插件模式改用 `AAP_PLUGIN_DATA_DIR/checkpoints`） |
| `SAM3_SCORE_THRESHOLD` | `0.5` | 默认置信度阈值 |
| `SAM3_EMBEDDING_CACHE_SIZE` | `32` | 交互式分割图像特征缓存容量 |
| `SAM3_MODEL_POOL_BUILD_TIMEOUT` | `120` | 模型池冷启动等待超时（秒） |
| `SAM3_IDLE_UNLOAD_SECONDS` | `600` | 空闲自动卸载阈值（秒，`<=0` 关闭） |
| `SAM3_IDLE_CHECK_INTERVAL` | `60` | 空闲检查周期（秒） |
| `SAM3_VIDEO_MAX_WINDOW_FRAMES` | `16` | SAM 3.1 多路文本跟踪单窗口最大帧数 |
| `SAM3_PVS_MAX_WINDOW_FRAMES` | `16` | PVS 交互式视频跟踪单窗口最大帧数 |
| `SAM3_PVS_MAX_CONTEXT_FRAMES` | `16` | PVS 持久化上下文最大帧数（大于窗口帧数时启用会话模式） |
| `SAM3_VIDEO_USE_FA3` | `0` | 设为 `1` 时在视频推理中启用 FlashAttention-3 |
| `LOG_LEVEL` | `INFO` | Python 日志级别 |

受管插件模式同时读取 `AAP_PLUGIN_TOKEN_FILE`、`AAP_PLUGIN_DATA_DIR`、`AAP_PLUGIN_DEVICE`、`AAP_PLUGIN_ASSET_MANIFEST_SHA256`、`AAP_PLUGIN_REGISTRY_ID`、`AAP_ROUTE_REGISTRY_ID`、`AAP_PLUGIN_GPU_RESOURCE_ID`、`AAP_PLUGIN_GPU_UUID`、`AAP_PLUGIN_GPU_KEYRING_FILE` 以及经审批的验收报告（`AAP_PLUGIN_QUALIFICATION_FILE` / `AAP_PLUGIN_QUALIFICATION_SHA256`）。

## 运行环境与依赖

- **Python**：`3.12`（`>=3.12,<3.13`）
- **推理栈**：`torch==2.7.1`、`torchvision==0.22.1`、CUDA `12.8`
- **子模块**：构建镜像前需初始化 Git 子模块 `vendor/sam3`。
- **设备策略**：强制要求 CUDA；设备不可用或构建失败时直接报错，不回退 CPU。

## 许可证

第一方插件源码采用 [MIT](LICENSE) 许可证。上游 `facebookresearch/sam3` 子模块与模型权重（`facebook/sam3`、`facebook/sam3.1`）均适用 **SAM License**，而非 MIT。

在分发源码、镜像或权重前，管理员须独立取得门控访问权限并核查遵守 SAM License（含再分发条件及必留声明）。
