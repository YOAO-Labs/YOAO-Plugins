# ONNX Tools 推理插件

[English](README.md) | 简体中文

`yoao-onnxtools-plugin` 是基于 [onnxtools](https://github.com/yyq19990828/onnxtools) 与 ONNX Runtime `1.22.0` 实现的 YOAO 两阶段车辆属性推理插件。插件遵循 YOAO ML Backend 协议 v2.2，同时提供检测加属性的复合管道与可独立编排的检测、分类原子模型。

## 能力与模型

| `/setup` 模型 ID | 任务 | 组合形态 | 输入 | 输出 | ORT session 数 |
| --- | --- | --- | --- | --- | --- |
| `vehicle-attr` | `detect` | `composite`（RT-DETR + PP-LCNet） | `full_image`、`crop` | `bbox` + `vehicle_type`、`color` | 2 |
| `vehicle-detect` | `detect` | `atom`（仅 RT-DETR） | `full_image`、`crop` | `bbox` | 1 |
| `vehicle-attr-classify` | `classify` | `atom`（仅 PP-LCNet） | `full_image`、`crop` | `vehicle_type`、`color`（不产出几何框） | 1 |

- **车型（`vehicle_type`，13 类）**：`car`（小车）、`truck`（卡车）、`bus`（公交）、`tanker`（油罐车）、`slagcar`（渣土车）、`fire_engine`（消防车）、`mixer`（混凝土搅拌车）、`ambulance`（救护车）、`police_car`（警车）、`engineering_truck`（工程车）、`hazardous_goods_vehicle`（危险品运输车）、`manned_sweeping_vehicle`（有人扫路车）、`school_bus`（校车）。
- **颜色（`color`，11 类）**：`black`（黑）、`white`（白）、`gray`（灰）、`red`（红）、`yellow`（黄）、`green`（绿）、`blue`（蓝）、`purple`（紫）、`brown`（棕）、`pink`（粉）、`other`（其他）。
- `/setup` 为 `vehicle-attr` 与 `vehicle-attr-classify` 自报 `output_attribute_schema`，方便项目直接导入属性选项。

## 模型权重与资产

本仓库与构建镜像均不包含模型权重。`pretrained-model-assets.json` 记录了两个 ONNX 文件（检测模型 `rtdetr-2024080100.onnx` 与属性分类模型 `va_260612.onnx`）的预期大小与 SHA-256 摘要。

- 插件模式（设置 `AAP_PLUGIN_TOKEN_FILE`）下，须将已审批权重只读挂载到 `AAP_PLUGIN_DATA_DIR/models`。
- 独立模式下，从 `ONNXTOOLS_MODEL_DIR`（默认 `/app/models`）加载权重。

## 关键配置

容器默认入口为 `python3 -m uvicorn main:app --host 0.0.0.0 --port 8004 --workers 1`。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ONNXTOOLS_MODEL_DIR` | `/app/models` | 独立模式模型目录（插件模式改用 `AAP_PLUGIN_DATA_DIR/models`） |
| `ONNXTOOLS_DET_MODEL` | `rtdetr-2024080100.onnx` | 车辆检测 ONNX 文件名 |
| `ONNXTOOLS_VA_MODEL` | `va_260612.onnx` | 车辆属性分类 ONNX 文件名 |
| `ONNXTOOLS_CONF_THRES` | `0.5` | 检测置信度阈值 |
| `ONNXTOOLS_BUILD_TIMEOUT` | `30` | 冷启动构建等待超时（秒） |
| `ONNXTOOLS_IDLE_UNLOAD_SECONDS` | `600` | 整池空闲自动卸载阈值（秒，`<=0` 关闭） |
| `ONNXTOOLS_IDLE_CHECK_INTERVAL` | `60` | 空闲检查周期（秒） |
| `LOG_LEVEL` | `INFO` | Python 日志级别 |

受管插件模式同时读取 `AAP_PLUGIN_TOKEN_FILE`、`AAP_PLUGIN_DATA_DIR`、`AAP_PLUGIN_DEVICE`、`AAP_PLUGIN_REGISTRY_ID`、`AAP_ROUTE_REGISTRY_ID`、`AAP_PLUGIN_GPU_RESOURCE_ID`、`AAP_PLUGIN_GPU_UUID`、`AAP_PLUGIN_GPU_KEYRING_FILE` 以及经审批的 CUDA 生命周期验收报告（`AAP_PLUGIN_QUALIFICATION_FILE` / `AAP_PLUGIN_QUALIFICATION_SHA256`）。

## 运行环境与依赖

- **Python**：`>=3.10`
- **推理栈**：固定提交的 `onnxtools[inference]`、`onnxruntime-gpu==1.22.0`（支持 CUDA 与 CPU）、`numpy==2.2.6`、`opencv-contrib-python-headless==5.0.0.93`、CUDA `12.8` 运行时基础镜像
- **设备策略**：独立部署在 CUDA 不可用时可回退 CPU；受管 CUDA 插件模式拒绝整条业务 session 回退 CPU。
- **环境隔离**：顶层平铺模块（`main`、`schemas`、`predictor` 等）须安装在独立 Python 环境中。

## 许可证

第一方插件源码采用 [MIT](LICENSE) 许可证。上游 `onnxtools` 与第三方运行时依赖遵循各自许可。

`rtdetr-2024080100.onnx` 与 `va_260612.onnx` 的来源与许可尚未核实；`pretrained-model-assets.json` 不授予下载、使用或再分发权利，部署前须由管理员独立核查授权。
