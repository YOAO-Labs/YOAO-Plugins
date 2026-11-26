# LibreYOLO 训练器插件

[English](README.md) | 简体中文

`yoao-trainer-libreyolo`（`yoao.trainer.libreyolo`）是基于 LibreYOLO `1.6.0`（LibreYOLO9）实现的官方 MIT 许可目标检测训练器插件，遵循 `yoao-trainer/v1` 协议与 `yoao-extension-manifest/v1` 清单规范。源码、依赖锁与容器镜像均不包含任何 Ultralytics 依赖。

## 支持的能力

- **扩展点与协议**：`trainer`（`yoao-trainer/v1`，清单能力：`train`、`evaluate`、`export_weights`；不声明 `preview`）
- **上游框架**：`libreyolo` `1.6.0`
- **支持任务**：`detect`（仅目标检测）
- **支持架构（共 4 个）**：`libreyolo9t`、`libreyolo9s`、`libreyolo9m`、`libreyolo9c`（`family: "libreyolo9"`、`end2end: false`）
- **数据集契约**：`ultralytics-yolo-1`、`ultralytics-bbox-1`
- **推理兼容性**：`framework: "libreyolo"`、`min_version: "1.6.0"`、`format: "libreyolo-native-state-dict-1.0"`

## CLI 子命令

容器入口为 `yoao-trainer-libreyolo`：

```sh
# 无需 GPU 或网络，输出 TrainerCapabilitiesV1 JSON
yoao-trainer-libreyolo describe

# 校验训练配置并输出 ValidateConfigResultV1 JSON
yoao-trainer-libreyolo validate-config --spec /work/input/config-spec.json

# 执行离线确定性训练、评估并生成回执产物
yoao-trainer-libreyolo train --spec /work/input/spec.json --output /work/output
```

- 检查点读取强制使用 `torch.load(..., weights_only=True)`，并在断点恢复时将优化器、EMA、RNG 或 GradScaler 状态异常升级为硬错误。
- 训练过程在输出目录内生成 `libreyolo-data.yaml` 而不修改只读数据集挂载，在具备样本任务标识时写入 `input-membership.json`，最终评估报告标准 COCO 指标（`mAP50-95`、`mAP50`、`mAP75`、`AR100`）。

## 运行环境与依赖

- **Python**：`>=3.11`（容器镜像使用 Python `3.12`）
- **训练栈**：`libreyolo==1.6.0`、`torch==2.14.1+cu132`、`torchvision==0.29.1+cu132`、`pycocotools>=2.0.7,<3`、CUDA `13.2`（无任何 `ultralytics` 依赖）
- **资源与容器**：`manifest.json` 声明 `requires_gpu: true`、`memory_mb: 8192`。镜像以非 root 用户 UID/GID `10001` 在 `--network=none` 环境下运行，缓存目录位于 `/tmp`。
- **模型权重**：本仓库与镜像均不包含基础权重（`LibreYOLO9{t,s,m,c}.pt`），须在运行时将已审批权重只读挂载进容器。

## 许可证

第一方插件源码采用 [MIT](LICENSE) 许可证，`manifest.json` 声明为 `MIT` 且无 Ultralytics 依赖。PyTorch、CUDA、`pycocotools`、`libreyolo` 随附第三方声明及预训练权重继续适用各自许可。
