# Ultralytics YOLO 训练器插件

[English](README.md) | 简体中文

`yoao-trainer-ultralytics`（`yoao.trainer.ultralytics`）是面向 Ultralytics YOLO `8.4.183` 的官方 `yoao-trainer/v1` 训练器插件，遵循 `yoao-extension-manifest/v1` 清单规范，支持离线目标检测与实例分割训练、配置校验及纯 CPU 数据增强预览。

## 支持的能力

- **扩展点与协议**：`trainer`（`yoao-trainer/v1`，清单能力：`train`、`evaluate`、`preview`、`export_weights`）
- **上游框架**：`ultralytics` `8.4.183`
- **支持任务**：`detect`（目标检测）、`segment`（实例分割）
- **支持架构（共 30 个）**：`yolov8`、`yolo11`、`yolo26` × `n`/`s`/`m`/`l`/`x` 的检测（`{family}{scale}`）与分割（`{family}{scale}-seg`）架构；其中 `yolo26` 采用端到端检测头（固定 `nms=False`，不适用 `dfl` 损失权重）。
- **数据集契约**：`ultralytics-yolo-1`、`ultralytics-bbox-1`、`ultralytics-simple-polygon-1`、`ultralytics-instance-rle-1`
- **推理兼容性**：`framework: "ultralytics"`、`min_version: "8.4.183"`、`format: null`（实例分割使用 `aap-instance-rle-1` 输出契约）

## CLI 子命令

容器入口为 `yoao-trainer-ultralytics`：

```sh
# 输出 TrainerCapabilitiesV1 JSON（架构列表、基础权重 SHA-256 目录、参数 schema）
yoao-trainer-ultralytics describe

# 校验训练配置并输出 ValidateConfigResultV1 JSON（校验不通过时同样返回退出码 0）
yoao-trainer-ultralytics validate-config --spec /work/input/config-spec.json

# 执行离线固定批大小训练、评估并生成回执产物
yoao-trainer-ultralytics train --spec /work/input/spec.json --output /work/output

# 执行纯 CPU 确定性数据增强预览（输出 preview-*.png 与 report.json）
yoao-trainer-ultralytics preview --spec /work/input/spec.json --output /work/output
```

- `train` 在离线环境中禁用遥测并以确定性配置运行，产出支持断点续训的检查点、`input-membership.json`、单调递增的 `events.jsonl` 与原子写入的 `result.json` 回执。
- `preview` 强制在纯 CPU 下运行（`CUDA_VISIBLE_DEVICES=""`），并施加内存、CPU 时间与超时约束。

## 运行环境与依赖

- **Python**：`>=3.11,<3.14`（容器镜像使用 Python `3.12`）
- **训练栈**：`ultralytics==8.4.183`、`torch==2.14.1+cu132`、`torchvision==0.29.1+cu132`、CUDA `13.2`
- **资源与容器**：`manifest.json` 声明 `requires_gpu: true`、`memory_mb: 8192`。镜像以非 root 用户 UID/GID `10001` 在 `--network=none` 环境下运行，缓存目录位于 `/tmp`。
- **模型权重**：本仓库与镜像均不包含 30 个受控基础 `.pt` 权重，须在运行时将已审批权重只读挂载进容器。

## 许可证

第一方插件源码采用 [MIT](LICENSE) 许可证。容器运行时固定依赖 `ultralytics==8.4.183`（采用 [AGPL-3.0-only](AGPL-3.0.txt) 许可），`manifest.json` 相应声明为 `AGPL-3.0-only`。模型权重与其他第三方依赖继续适用各自条款。
