# YOAO Plugins

本仓库仅持有官方推理实现：[LibreYOLO](plugins/inference/libreyolo/README.md)、[YOLO](plugins/inference/yolo/README.md)、[RapidOCR](plugins/inference/rapidocr/README.md) 和 [ONNX Tools](plugins/inference/onnxtools/README.md)。LibreYOLO 原生适配器位于 `plugins/inference/libreyolo/native`。第一方代码采用 [MIT](LICENSE)；第三方代码、模型权重与容器依赖继续适用各自的许可。模型权重、训练产物、数据集和凭据不提交到本仓库。

插件开发 SDK、共享推理运行时和 mask 工具由 [Plugin-SDK](https://github.com/YOAO-Labs/Plugin-SDK) 持有并发布；平台 API SDK、业务 owner、权限与持久化规则由平台持有。平台维护的任务、协议、生命周期、推理与训练契约通过固定版本和 SHA256 的公开制品消费，不依赖平台 checkout。契约镜像 `contracts-3d6393e64630` 的 manifest SHA256 为 `d3749d2a7429da60c02e222c65e9c4a5ec545b050b797422263d5a3bcb02ddec`。

各插件拥有其预训练模型目录（固定来源、摘要和许可信息）；平台独立管理审批、存储、缓存准备和用户 checkpoint。目录可见不代表允许下载、启用或重新分发。容器与真实业务资格独立于源码公开和 Python 包发布。

## Operator selection descriptors

`scripts/generate-selection-descriptor.py` produces a separate canonical JSON sidecar describing the expected asset identities for a specific immutable native deployment. It does not replace the installation's `asset_manifest_sha256`, hash weight files, download, approve, or enable models. Use the exact source revision and published catalogs matching the artifact; for YOLO the source-only operator environment also needs the pinned `aap-task-contract` wheel from the plugin's lockfile.

```sh
python scripts/generate-selection-descriptor.py --plugin rapidocr --version 0.1.0 \
  --artifact 'registry.example/rapidocr@sha256:<64-hex-digest>' \
  --asset-manifest-sha256 '<64-hex-installation-manifest-digest>' \
  --data-dir /private/plugin-data --output /private/rapidocr-selections.json
```

Use `--plugin yolo` or `--plugin onnxtools` with their own version, artifact and installation digest. Output must be a new file outside the data directory; stdout prints its SHA256. JSON keys use sorted order, compact separators, ASCII escaping, and no trailing newline. Arrays are deterministically sorted. Paths are relative to the native plugin data directory; the generator rejects traversal, escaping symlinks and conflicting prepared YOLO identities. For YOLO it reads the prepared cache's filename mappings to follow the native digest/legacy precedence, but identities come from the published catalog. Regenerate if the prepared mapping or placement changes.

Only combinations with complete published identities are described; missing local weights remain expected, not asset-free. RapidOCR describes all three sessions per engine, including currently uninstalled combinations; ONNX distinguishes detector, classifier and pipeline; YOLO tracker algorithms share detector weights, World and YOLOE text require their respective encoders, and visual exemplars do not require a text encoder. Catalog descriptions grant no rights. Platform policy, approval, availability, local-byte verification, project permissions and storage remain platform responsibilities; unverified weight redistribution is not authorized.

## 已安装环境下的插件开发

使用公开 [SDK 发布渠道](https://github.com/YOAO-Labs/Plugin-SDK/releases/tag/python-sdk-0.1.0) 安装 SDK、运行时和所需契约 wheel；验证发布 manifest 中的摘要，并以哈希锁定所有依赖。SDK 的通用入口为 `aap-inference-plugin`：

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
