# YOAO Plugins

YOAO（**You Only Annotate Once**）的官方推理插件、通知渠道插件与格式适配器插件在本仓库维护：

- **推理插件（`plugins/inference/`）**：[LibreYOLO](plugins/inference/libreyolo/README.md)、[YOLO](plugins/inference/yolo/README.md)、[RapidOCR](plugins/inference/rapidocr/README.md)、[ONNX Tools](plugins/inference/onnxtools/README.md)、[Grounded-SAM2](plugins/inference/grounded-sam2/README.md) 和 [SAM3](plugins/inference/sam3/README.md)。LibreYOLO 原生适配器位于 `plugins/inference/libreyolo/native`。
- **通知渠道插件（`plugins/notification-channel/`）**：[飞书私信（Feishu DM）](plugins/notification-channel/feishu-dm/README.md)（零依赖 Node.js 20+ ESM 参考实现，遵循 `yoao-notification-channel/v1` 与 `yoao-extension-manifest/v1`，支持直连档裸进程与非 root 只读 OCI 容器两种形态）。
- **格式适配器插件（`plugins/format-adapter/`）**：[LabelMe JSON](plugins/format-adapter/labelme-json/README.md)（零依赖 Node.js 20+ ESM 参考实现，遵循 `yoao-format-adapter/v1` 与 `yoao-extension-manifest/v1`，支持 `rectangle` / `polygon` 导入导出及直连档裸进程与非 root 只读 OCI 容器两种形态）。

第一方代码采用 [MIT](LICENSE)；第三方代码、模型权重与容器依赖继续适用各自的许可。模型权重、训练产物、数据集和凭据不提交到本仓库。

插件开发 SDK、共享推理运行时和 mask 工具由 [Plugin-SDK](https://github.com/YOAO-Labs/Plugin-SDK) 持有；平台 API SDK、业务 owner、权限与持久化规则由平台持有。新的候选构建使用明确提供的契约及 SDK 制品 manifest 和 SHA256，不依赖平台 checkout，也不预设新版本发布地址。历史 YOLO 锁固定 [`contracts-906628f2619b`](https://github.com/YOAO-Labs/Plugin-SDK/releases/tag/contracts-906628f2619b)，manifest SHA256 为 `5c26a3829073324bf6f9cc29521457d296920424e5b1b3c34d672408aa956663`；历史 ONNX Tools 与 RapidOCR 锁固定 `contracts-3d6393e64630`，manifest SHA256 为 `d3749d2a7429da60c02e222c65e9c4a5ec545b050b797422263d5a3bcb02ddec`。这些已发布事实不表示新命名制品已经发布，也不改变既有镜像的依赖。

各插件拥有其预训练模型目录（固定来源、摘要和许可信息）；平台独立管理审批、存储、缓存准备和用户 checkpoint。目录可见不代表允许下载、启用或重新分发。容器与真实业务资格独立于源码公开和 Python 包发布。

## 部署资产选择描述

`scripts/generate-selection-descriptor.py` 为固定摘要的原生部署生成独立的规范 JSON 文件，描述每个模型／变体预期使用的资产。它不替换安装的 `asset_manifest_sha256`，也不读取权重内容、下载、批准或启用模型。使用与镜像匹配的源码和插件发布清单；YOLO 的源码侧工具环境还需安装插件锁文件固定的 `yoao-task-contract` 和 `yoao-training-contract` wheel，LibreYOLO 使用其锁定的 `httpx` 依赖读取现有资产描述模块。

将 `PLUGIN_VERSION` 设为该安装清单中的插件版本。

```sh
python scripts/generate-selection-descriptor.py --plugin rapidocr --version "$PLUGIN_VERSION" \
  --artifact 'registry.example/rapidocr@sha256:<64-hex-digest>' \
  --asset-manifest-sha256 '<64-hex-installation-manifest-digest>' \
  --data-dir /private/plugin-data --output /private/rapidocr-selections.json
```

其他插件使用各自的 `--plugin libreyolo`、`--plugin yolo`、`--plugin onnxtools`、`--plugin grounded-sam2` 或 `--plugin sam3`，并传入其版本、镜像及安装清单摘要。输出必须是数据目录外的新文件，标准输出会打印 SHA256。JSON 使用键排序、紧凑分隔符、ASCII 转义且无末尾换行；数组确定性排序。资产路径相对于原生插件数据目录，禁止路径穿越、逃逸符号链接及冲突的 YOLO 预备资产身份。YOLO 复用训练契约读取预备清单，并遵循原生的摘要文件优先、旧文件名次之的查找顺序；预期身份始终来自插件发布清单。预备映射或存放位置改变后须重新生成。

工具只描述具有完整发布身份的组合；本地缺少权重不等于无资产需求。LibreYOLO 声明原生 `yolov9` 系列四种尺寸与 `assets/` 路径，已有安装也须显式批准；RapidOCR 每个引擎都声明全部三个 session 的资产，包含尚未安装的组合；ONNX Tools 区分检测、分类和组合流水线；YOLO 两种跟踪算法共享检测器权重，World 与 YOLOE 文本任务分别需要对应编码器，视觉示例任务不需要文本编码器。目录描述不授予使用权。平台继续管理审批、可用性、本地文件核验、项目授权及存储；未确认许可的权重不得公开分发。

Grounded-SAM2 的图像池同时构造 SAM、GroundingDINO 和 BERT，因此图像选择必须包含两个变体轴及全部依赖；视频 tracker 只选择 SAM。SAM3 区分图像／PVS 的 `sam3.pt` 与 multiplex 的 `sam3.1_multiplex.pt`，两者的访问和许可分别审批。两个目录模块均只依赖标准库。安装清单摘要取与镜像匹配且经核验的 wheel 内 `pretrained-assets.json` 的原始字节，不对 JSON 重新序列化后计算。

## 已安装环境下的插件开发

先核验 SDK 和平台契约各自的 manifest 及 wheel，再按 SDK manifest 的 `source_revision` 阅读该修订的 `packages/inference/sdk/README.md`。公开旧版的 `aap-inference-plugin` 不能用于当前候选。自定义检测插件使用作者命令 `yoao-plugin`，不是管理员的 `yoao plugins`。在独立的 Python 3.11 环境准备完整 wheelhouse 和可信哈希锁，锁中也要包含 `build`、`setuptools>=68` 和 `wheel`：

```sh
python -m pip install --no-index --find-links="$WHEELHOUSE" --require-hashes -r "$REQUIREMENTS_LOCK"
yoao-plugin init my-plugin
yoao-plugin build my-plugin
python -m pip install --no-index --no-deps my-plugin/dist/*.whl
```

通用 SDK 不准备 LibreYOLO 权重。需要官方 LibreYOLO 模型时，请先审查模型许可，然后使用 [LibreYOLO 文档](plugins/inference/libreyolo/README.md)中的 `yoao-libreyolo prepare-assets`。本地开发 token 不是平台管理员凭据，本地预检不等于受管平台部署资格。

有状态官方插件实现完整推理 HTTP 契约并复用 SDK 运行时，不缩减为仅提供检测的简化作者接口。独立模型资格检查复用 `yoao_backend_runtime.qualification_http` 的真实本地 HTTP 和签名路由准入；调用方负责应用生命周期。该工具不替代真实设备、权重批准或平台安装验收。

持久 tracker 状态由实际模型池条目持有，命令执行结束即释放 borrower；后续命令重新借用同一条目，不能按 key 静默重建旧上下文。驱逐、排空与关闭须等待真实会话清理，失败保持未知状态；每个窗口仍独立进行签名准入和完成计账。两种 SAM 保留独立依赖环境、池拓扑和上下文合同，不能互相继承资格。

## CPU 验证

[Plugin Python acceptance](.github/workflows/plugin-python.yml) 为六个插件定义独立 CPU 作业，隔离各自的 Python 环境和依赖。作业先按所需子集校验两份 manifest 和 wheel，再将 wheel 放入仓库内 `.artifacts/python/first-party/`，供各插件的相对路径依赖和锁文件使用。YOLO 与有状态插件使用各自的开发锁文件；SAM3 作业初始化固定 vendor 并应用请求级 autocast 补丁。RapidOCR 与 ONNX Tools 使用哈希锁定的 headless 依赖。每个作业还安装所构建的插件 wheel，在源码目录外检查安装后的入口和目录资源。RapidOCR 的模拟 provider 测试保留默认设备策略，CI 仅隐藏 GPU。

CI 需要仓库变量 `YOAO_CONTRACT_MANIFEST`、`YOAO_CONTRACT_MANIFEST_SHA256`、`YOAO_SDK_MANIFEST` 和 `YOAO_SDK_MANIFEST_SHA256` 指向实际可访问的制品及其摘要；缺失时作业失败。锁文件记录具体候选 wheel 的身份，提供不同制品时需由维护者核对并重新生成相应锁。模型权重和私有 fixture 不进入 CI；目录资源和开发元数据不代表模型已加载或通过资格验证。

本地从仓库根目录准备锁文件或同步环境时，也须使用实际提供的两份 manifest、摘要和兼容子集，例如：

```sh
python scripts/fetch-contracts.py --manifest "$CONTRACT_MANIFEST" --sha256 "$CONTRACT_MANIFEST_SHA256" \
  --output .artifacts/python/first-party --package yoao-task-contract --package yoao-protocol-v2 --package yoao-lifecycle-contract
python scripts/fetch-contracts.py --manifest "$SDK_MANIFEST" --sha256 "$SDK_MANIFEST_SHA256" \
  --output .artifacts/python/first-party --package yoao-backend-runtime
```

这是 RapidOCR 和 ONNX Tools 的 Python 3.10 子集；Grounded-SAM2 还需要 `yoao-mask-utils`。SAM3 使用相同的五个第一方 wheel，但运行于独立的 Python 3.12 环境。YOLO 还需要 `yoao-inference-contract` 和 `yoao-training-contract`；LibreYOLO 候选还需要这两个契约及 `yoao-inference-plugin-sdk`。后面三个包要求 Python 3.11 或更新版本，不应安装到 Python 3.10 插件环境。

CPU 测试不授予 CUDA 资格。真实模型推理、逐 session provider、显存回收及平台安装／退役，仍须在获授权的输入和独占设备上验证具体镜像摘要。

## 构建发布输入

从干净克隆运行；隔离构建环境需安装 `build==1.3.0`、`setuptools==80.9.0` 和 `wheel==0.45.1`。LibreYOLO 使用 Python 3.11，YOLO 使用 Python 3.12，RapidOCR 与 ONNX Tools 使用 Python 3.10；分别构建，不能假设同一解释器适合全部插件：

```sh
python3.11 scripts/build-plugin-artifacts.py --revision HEAD --output /private/libreyolo-release --plugin libreyolo
python3.12 scripts/build-plugin-artifacts.py --revision HEAD --output /private/yolo-release --plugin yolo
python3.10 scripts/build-plugin-artifacts.py --revision HEAD --output /private/cpu-release --plugin rapidocr --plugin onnxtools
```

有状态插件分别使用下列工具链和独立输出目录；每个构建环境还需安装 `build==1.3.0`、`wheel==0.45.1`。不能用同一个 Python／构建依赖环境混装这两个插件。

| 插件 | Python | Torch／torchvision | CUDA | 构建 setuptools |
| --- | --- | --- | --- | --- |
| Grounded-SAM2 | 3.10 | 2.3.1／0.18.1 | 12.1 | 70.3.0 |
| SAM3 | 3.12 | 2.7.1／0.22.1 | 12.8 | 80.9.0 |

```sh
python3.10 scripts/build-plugin-artifacts.py --revision HEAD \
  --output /private/grounded-sam2-release --plugin grounded-sam2
python3.12 scripts/build-plugin-artifacts.py --revision HEAD \
  --output /private/sam3-release --plugin sam3
```

构建器从指定提交创建源码归档，为所选插件生成 wheel 和源码分发包，仅在选择 LibreYOLO 时附带其 native 包，并从 wheel 提取插件拥有的目录，同时生成 `manifest.json` 和 `SHA256SUMS`。默认只选择 LibreYOLO。manifest 从同一提交记录 vendor 子模块的 URL 与固定 commit；第一方源码归档不包含子模块内容。构建模型镜像时必须从对应提交初始化所需子模块，再按插件 Dockerfile 应用固定补丁，不能将 vendor 快照复制进主仓。

通过明确指定的 manifest 路径或 URL 和摘要取得契约或 SDK wheel；文件须与 manifest 位于同一目录，`--package` 可重复以只选择兼容的发行包。不指定过滤器时下载 manifest 中所有 wheel：

```sh
python scripts/fetch-contracts.py --manifest /private/contracts/manifest.json \
  --sha256 '<manifest-sha256>' --output /private/contracts-wheels \
  --package yoao-task-contract --package yoao-protocol-v2 --package yoao-lifecycle-contract
```

LibreYOLO 的私有候选构建使用 Linux x86_64 glibc、Python 3.11、上述构建依赖及 `hatchling==1.27.0`；提供本地契约和 SDK manifest 文件或包含 `manifest.json` 的制品目录，以及各自的实际摘要：

```sh
python3.11 scripts/build-libreyolo-candidate.py --revision HEAD --output /private/libreyolo-candidate.tar \
  --contracts-manifest /private/contracts --contracts-sha256 '<contracts-manifest-sha256>' \
  --sdk-manifest /private/sdk --sdk-sha256 '<sdk-manifest-sha256>' \
  --dependency-cache /private/wheel-cache
```

构建器校验 manifest 和选中 wheel 的摘要，结合第三方哈希锁组装 wheelhouse 与 `requirements.lock`；第三方依赖获取仍需网络。`--dependency-cache` 可省略，源码目录不会作为 editable 运行依赖。候选包不上传，制品生成、安装成功或 CPU 测试均不构成公开发布或模型／GPU 资格。

第一方发布输入（`build-plugin-artifacts.py`）不含第三方 wheel；私有离线候选（`build-libreyolo-candidate.py`）会装入匹配环境的依赖 wheel，但不包含模型权重。两种构建均不上传制品，也不授予 OCI 镜像或权重的再分发许可。
