# YOAO Plugins

本仓库仅持有官方推理实现：[LibreYOLO](plugins/inference/libreyolo/README.md)、[YOLO](plugins/inference/yolo/README.md)、[RapidOCR](plugins/inference/rapidocr/README.md) 和 [ONNX Tools](plugins/inference/onnxtools/README.md)。LibreYOLO 原生适配器位于 `plugins/inference/libreyolo/native`。第一方代码采用 [MIT](LICENSE)；第三方代码、模型权重与容器依赖继续适用各自的许可。模型权重、训练产物、数据集和凭据不提交到本仓库。

插件开发 SDK、共享推理运行时和 mask 工具由 [Plugin-SDK](https://github.com/YOAO-Labs/Plugin-SDK) 持有并发布；平台 API SDK、业务 owner、权限与持久化规则由平台持有。平台维护的任务、协议、生命周期、推理与训练契约通过固定版本和 SHA256 的公开制品消费，不依赖平台 checkout。YOLO 的哈希锁固定 [`contracts-906628f2619b`](https://github.com/YOAO-Labs/Plugin-SDK/releases/tag/contracts-906628f2619b)，其 manifest SHA256 为 `5c26a3829073324bf6f9cc29521457d296920424e5b1b3c34d672408aa956663`；ONNX Tools 与 RapidOCR 的既有锁及共享下载器仍固定 `contracts-3d6393e64630`，manifest SHA256 为 `d3749d2a7429da60c02e222c65e9c4a5ec545b050b797422263d5a3bcb02ddec`。各插件按自身锁文件构建；更新开发文档不会替换已获资格镜像的依赖。

各插件拥有其预训练模型目录（固定来源、摘要和许可信息）；平台独立管理审批、存储、缓存准备和用户 checkpoint。目录可见不代表允许下载、启用或重新分发。容器与真实业务资格独立于源码公开和 Python 包发布。

## 部署资产选择描述

`scripts/generate-selection-descriptor.py` 为固定摘要的原生部署生成独立的规范 JSON 文件，描述每个模型／变体预期使用的资产。它不替换安装的 `asset_manifest_sha256`，也不读取权重内容、下载、批准或启用模型。使用与镜像匹配的源码和插件发布清单；YOLO 的源码侧工具环境还需安装插件锁文件固定的 `aap-task-contract` 和 `aap-training-contract` wheel，LibreYOLO 使用其锁定的 `httpx` 依赖读取现有资产描述模块。

```sh
python scripts/generate-selection-descriptor.py --plugin rapidocr --version 0.1.0 \
  --artifact 'registry.example/rapidocr@sha256:<64-hex-digest>' \
  --asset-manifest-sha256 '<64-hex-installation-manifest-digest>' \
  --data-dir /private/plugin-data --output /private/rapidocr-selections.json
```

LibreYOLO、YOLO 和 ONNX Tools 分别使用 `--plugin libreyolo`、`--plugin yolo`、`--plugin onnxtools`，并传入各自的版本、镜像及安装清单摘要。输出必须是数据目录外的新文件，标准输出会打印 SHA256。JSON 使用键排序、紧凑分隔符、ASCII 转义且无末尾换行；数组确定性排序。资产路径相对于原生插件数据目录，禁止路径穿越、逃逸符号链接及冲突的 YOLO 预备资产身份。YOLO 复用训练契约读取预备清单，并遵循原生的摘要文件优先、旧文件名次之的查找顺序；预期身份始终来自插件发布清单。预备映射或存放位置改变后须重新生成。

工具只描述具有完整发布身份的组合；本地缺少权重不等于无资产需求。LibreYOLO 声明原生 `yolov9` 系列四种尺寸与 `assets/` 路径，已有安装也须显式批准；RapidOCR 每个引擎都声明全部三个 session 的资产，包含尚未安装的组合；ONNX Tools 区分检测、分类和组合流水线；YOLO 两种跟踪算法共享检测器权重，World 与 YOLOE 文本任务分别需要对应编码器，视觉示例任务不需要文本编码器。目录描述不授予使用权。平台继续管理审批、可用性、本地文件核验、项目授权及存储；未确认许可的权重不得公开分发。

## 已安装环境下的插件开发

使用公开 [SDK 发布渠道](https://github.com/YOAO-Labs/Plugin-SDK/releases/tag/python-sdk-6b0f437333b3) 安装 SDK、运行时和所需契约 wheel；验证发布 manifest 中的摘要，并以哈希锁定所有依赖。开发环境的契约安装步骤见 [Plugin-SDK](https://github.com/YOAO-Labs/Plugin-SDK#portable-contracts-and-local-development)；已安装插件继续使用自身哈希锁固定的发布制品。SDK 的通用入口为 `aap-inference-plugin`：

```sh
aap-inference-plugin init my-plugin
cd my-plugin
uv lock --python 3.11
uv sync --python 3.11 --frozen
```

通用 SDK 不准备 LibreYOLO 权重。需要官方 LibreYOLO 模型时，请先审查模型许可，然后使用 [LibreYOLO 文档](plugins/inference/libreyolo/README.md)中的 `aap-libreyolo prepare-assets`。本地开发 token 不是平台管理员凭据，本地预检不等于受管平台部署资格。

## CPU 验证

[Plugin Python acceptance](.github/workflows/plugin-python.yml) 分别运行四个插件的 CPU 测试，隔离各自的 Python 环境和依赖。YOLO 使用完整开发锁文件；RapidOCR 与 ONNX Tools 使用镜像的哈希锁定 headless 依赖及已校验契约 wheel，不引入平台 checkout。模型权重和私有 fixture 不进入 CI。

CPU 测试不授予 CUDA 资格。真实模型推理、逐 session provider、显存回收及平台安装／退役，仍须在获授权的输入和独占设备上验证具体镜像摘要。

## 构建发布输入

从干净克隆运行；隔离构建环境需安装 `build==1.3.0`、`setuptools==80.9.0` 和 `wheel==0.45.1`：

```sh
python scripts/build-plugin-artifacts.py --revision HEAD --output /private/plugin-release \
  --plugin libreyolo --plugin yolo --plugin rapidocr --plugin onnxtools
```

构建器从指定提交创建源码归档，为所选插件和 LibreYOLO native 生成 wheel 和源码分发包，并从 wheel 提取插件拥有的目录，同时生成 `manifest.json` 和 `SHA256SUMS`。默认只选择 LibreYOLO。它不会上传制品，也不包含 SDK、运行时或 mask 源码、模型权重或第三方 wheel。独立的私有哈希锁定候选包与 OCI 流程见 [LibreYOLO](plugins/inference/libreyolo/README.md)。
