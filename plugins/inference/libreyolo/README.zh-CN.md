# LibreYOLO 推理插件

[English](README.md) | 简体中文

`yoao-libreyolo-plugin` 是基于原生 LibreYOLO `1.6.0` 加载器（`yoao-libreyolo-native`）实现的 YOAO 目标检测推理插件，不依赖 Ultralytics。插件使用 YOAO 检测推理 SDK（`yoao-inference-plugin-sdk`）与共享 GPU 生命周期运行时。

## 能力与模型

| `/setup` 模型 ID | 任务 | 输入 | 输出 | 变体与权重 |
| --- | --- | --- | --- | --- |
| `detect` | `detect` | `full_image`、`crop` | `bbox` | 预训练 YOLOv9 `t`（默认）、`s`、`m`、`c`；受管训练权重（`trained:<model_version_id>`） |

- 预训练变体对应 `LibreYOLO9t.pt`、`LibreYOLO9s.pt`、`LibreYOLO9m.pt` 与 `LibreYOLO9c.pt`。启动服务前必须先准备好默认变体 `t`。
- 受管训练权重使用 `libreyolo-native-state-dict-1.0` 格式，加载前会依据不可变训练回执独立校验。
- 请求级推理参数支持 `conf`（默认 `0.25`）、`iou`（默认 `0.45`）、`max_det`（默认 `300`）以及可选的 `classes` 类别过滤。

## 模型权重与资产

本仓库与构建镜像均不包含模型权重，推理过程也不会自动下载权重。插件内置版本化预训练资产清单 `yoao_libreyolo_plugin/controlled-model-assets.json`，固定记录四个官方 YOLOv9 权重的文件名、来源与 SHA-256 摘要。

在获准环境中启动服务前，可显式下载并校验所需权重：

```sh
yoao-libreyolo prepare-assets --size t --asset-dir /var/lib/aap-libreyolo/assets
```

如需暴露其他规格，可分别传入 `--size s`、`m` 或 `c`。资产清单可见不代表已获得下载、启用或再分发授权。

## 关键配置

容器默认入口为 `yoao-libreyolo serve --host 0.0.0.0 --port 8000`（以 UID/GID `10001` 运行）。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `AAP_PLUGIN_TOKEN_FILE` | 必填 | 私有 Bearer 令牌文件路径（不少于 32 个字符） |
| `AAP_PLUGIN_DEVICE` | `cpu` | 运行设备（`cpu` 或 `cuda:0`） |
| `AAP_PLUGIN_ASSET_DIR` | `/var/lib/aap-libreyolo/assets` | 已校验预训练权重目录 |
| `AAP_PLUGIN_DATA_DIR` | `/var/lib/aap-libreyolo` | 私有状态目录（受管模型保存在 `managed/` 下） |
| `AAP_PLUGIN_ASSET_ORIGINS` | 空 | 允许拉取受管模型的来源列表（逗号分隔） |
| `AAP_PLUGIN_IMAGE_HOSTS` | 空 | 允许拉取远程图像的主机列表（逗号分隔） |

受管 CUDA 部署还需提供 `AAP_PLUGIN_REGISTRY_ID`、`AAP_PLUGIN_GPU_RESOURCE_ID`、`AAP_PLUGIN_GPU_UUID`（物理 `GPU-...`）与 `AAP_PLUGIN_GPU_KEYRING_FILE`（Ed25519 公钥验签 keyring JSON）。仅当同时挂载经审批的验收报告与摘要（`AAP_PLUGIN_QUALIFICATION_FILE`、`AAP_PLUGIN_QUALIFICATION_SHA256`、`AAP_PLUGIN_OCI_ARTIFACT`、`AAP_PLUGIN_IMAGE_ID`）时，服务才会声明受管 GPU 生命周期能力。

本地 CPU 回环调试可运行 `yoao-libreyolo serve --local --device cpu --host 127.0.0.1 --port 8000`。

## 运行环境与依赖

- **Python**：`>=3.11`
- **推理栈**：`libreyolo==1.6.0`、`torch==2.14.1+cu132`、`torchvision==0.29.1+cu132`（CUDA `13.2` 基线）
- **精度与回退策略**：固定 FP32 执行，禁用 half、AMP 与 TF32；CUDA 设备校验失败时直接报错，不回退 CPU。

## 许可证

第一方插件及原生加载器源码采用 [MIT](LICENSE) 许可证。上游 [LibreYOLO 许可说明](https://www.libreyolo.com/docs/licensing) 将其库代码与 YOLO9 COCO 权重标示为 MIT，但随包附带的第三方声明（含 DINOv3）、PyTorch、NVIDIA 运行时组件以及各具体权重仓库仍适用各自许可，使用或再分发前须独立核查。
