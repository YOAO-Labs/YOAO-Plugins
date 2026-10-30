# rapidocr-backend

RapidOCR（ONNX）ML backend，平台首个真实 OCR backend。把 RapidOCR 的 `det → cls → rec`
三段拆为**原子能力 + 端到端编排**，对外自报三个 model：

| model_id  | 任务        | 输入 → 输出                                     | composition                 |
| --------- | ----------- | ----------------------------------------------- | --------------------------- |
| `ocr-det` | `detection` | full_image → polygon 文本框                     | atom                        |
| `ocr-rec` | `ocr`       | crop → `attributes.text`(+orientation/language) | atom（内部跑 cls 方向校正） |
| `ocr-e2e` | `ocr`       | full_image → polygon + text + orientation       | composite（det→cls→rec）    |

cls（文本行方向 0/180）语言/版本无关，内化进 rec 与 e2e、不单独暴露。平台 pipeline 可把
`ocr-det（源阶段）→ ocr-rec（下游吃 crop）` 串成编排。

## 变体（version × size × lang）

- **det**：v5 mobile/server · v6 tiny/small/medium（语言无关）
- **rec / e2e**：通用(中英) = v5 mobile/server + v6 tiny/small/medium；英文 = v5 mobile
- `cls` 按 size 选 mobile/server 档，rec/e2e 共享

经 `context.model_variants`（`{version,size,lang}`）选档，缺省取 v5/mobile/universal。

## 模型权重

13 个 ONNX 文件（det 5 + cls 2 + rec 6）来自 RapidOCR v3.9.0 ModelScope，
不进入本仓库或公开制品。仅在授权环境中运行
`python -m download_models --models-dir /path/to/models`，并在下载、使用或再分发前
逐项核验来源、摘要、许可和授权。

安装 wheel 后运行 `python -m pretrained_manifest` 可生成版本化发布者描述符：
13 个文件的路径、大小、摘要、上游来源、许可核验状态、使用该文件的模型变体
及运行时兼容性。目录和下载器是变体、路径及摘要的唯一来源；描述符不包含
管理员批准、项目授权或存储位置。`unverified` 不代表可分发或已获许可。

## Python 依赖与镜像构建

`pyproject.toml` 将 CPU 与 CUDA 声明为互斥的运行时 extra：`cpu` 使用
ONNX Runtime 1.23.2，`cuda` 使用 ONNX Runtime GPU 1.23.2 和 cuDNN 9.10.2。
两者均使用 RapidOCR 3.9.0、NumPy 2.2.6、headless OpenCV 和共享包；
CUDA/cuBLAS 由 GPU 基础镜像提供。共享 runtime 还依赖 `aap-lifecycle-contract`。

`uv.lock` 固定完整依赖图，`requirements-cpu.txt` 和
`requirements-runtime.txt` 分别固定第三方 registry artifact 哈希。
开发锁文件与镜像输入均使用 `Plugin-SDK` 的不可变公开 runtime 发布
`python-sdk-0.1.0` 和契约发布 `contracts-3d6393e64630`；
`build-inputs.py` 使用仓库的 `scripts/fetch-contracts.py` 校验契约 manifest 与 wheel 摘要，并校验 runtime wheel 摘要。

镜像以 `--no-deps --require-hashes` 安装第三方锁文件，之后安装已校验第一方 wheel
和独立构建的 backend wheel。RapidOCR metadata 要求 `opencv-python`，运行时有意只装
`opencv-python-headless`；不可补装 GUI 包。CUDA 基础镜像提供 CUDA/cuBLAS，不安装
`nvidia-cublas-cu12` wheel。stock `pip check` 因这些有意替换可能报告缺失 distribution，
不能将其视为此布局的成功门禁。

插件安装时管理员核验每个选中权重的来源和授权，按 `models/` 下的相对路径放入
安装状态目录的 `data/models/`。`python -m plugin_assets` 对每个 `--asset PATH`
要求独立的 `--asset-license PATH=LICENSE=SOURCE`，把实际许可和证据来源记录在
`data/assets.json` 中；顶层 `--license` 是插件代码许可，不代表权重许可。
启动时验证 manifest 摘要和固定权重 SHA256，并拒绝未列入清单的 ONNX 文件。
不能仅凭文件名或插件许可推定权重授权。RapidOCR 的
[模型许可声明](https://rapidai.github.io/RapidOCRDocs/main/en/model_licenses/)
明确覆盖本目录中摘要为 `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f`
的 v6 small det 和 `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884`
的 v6 small rec（Apache-2.0，保留 PaddleOCR 归属和通知）；其他 11 个文件不在该
声明列出的范围内，须单独核验，不得据此泛化。准备工具不会下载权重。

## 独立构建

在本插件仓库根目录运行；输出目录必须不存在，可移至不含平台 checkout 的主机：

```bash
python3 plugins/inference/rapidocr/build-inputs.py /tmp/rapidocr-inputs
docker build -f /tmp/rapidocr-inputs/Dockerfile.cpu -t rapidocr-cpu /tmp/rapidocr-inputs
docker build -f /tmp/rapidocr-inputs/Dockerfile -t rapidocr-gpu /tmp/rapidocr-inputs
```

CPU 使用 CPU ORT、无 NVIDIA wheel，运行时使用 `--runtime=runc` 且设置
`CUDA_VISIBLE_DEVICES=""`、`NVIDIA_VISIBLE_DEVICES=void`；不得申请 GPU。
GPU 镜像以 CUDA 12.8 runtime 和 cuDNN 9.10.2 支持 PP-OCRv6。构建不等于 GPU 资格；
实际 GPU 运行仅在独占目标设备获授权后进行。

环境变量：

| 变量                                  | 默认                                                | 说明                                                                                                                                        |
| ------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `RAPIDOCR_MODEL_DIR`                  | GPU `/app/models`；CPU `/var/lib/aap-plugin/models` | 权重根目录                                                                                                                                  |
| `RAPIDOCR_DEVICE`                     | GPU `gpu`；CPU `cpu`                                | `gpu` 优先构造 CUDA session，明确设备错误时才尝试 CPU replacement；`cpu` 只构造 CPU session。实际 provider 以 `/health` 的业务 session 为准 |
| `RAPIDOCR_POOL_CAP`                   | `3`                                                 | composite 引擎数上限；每个引擎固定持有 det/cls/rec 三个 ORT session                                                                         |
| `RAPIDOCR_BUILD_TIMEOUT`              | `30`                                                | 调用方等待冷启动的秒数；超时后 builder 仍由池跟踪                                                                                           |
| `RAPIDOCR_IDLE_UNLOAD_SECONDS`        | `600`                                               | 整池空闲卸载阈值；非正数关闭                                                                                                                |
| `RAPIDOCR_IDLE_CHECK_INTERVAL`        | `60`                                                | 空闲检查周期                                                                                                                                |
| `RAPIDOCR_MANAGED_LIFECYCLE_VERIFIED` | `0`                                                 | 部署级验收门；只有当前镜像、权重和物理 GPU 的严格证据通过后才设为 `1`                                                                       |
| `GPU_LIFECYCLE_VERIFY_KEYS_JSON`      | 空                                                  | Ed25519 验签公钥 keyring；空值保持 legacy gate                                                                                              |

## 引擎池与受管生命周期

六种合法权重三件套映射为六个动态 key。池在构造前预留 slot，同 key 冷启动
single-flight，容量满时只淘汰没有 borrower/waiter 的最旧引擎，并且先释放旧引擎的
三个 session 再构造新引擎。同引擎的阈值更新和推理由 use lock 串行；不同 key 可并行。

`/health.residency.pools.engines` 是用于仲裁的稳定聚合池 ID，动态权重 key 仍留在
`/health.pool.loaded_keys` 用于诊断。驻留判定读取每个引擎 det/cls/rec 三条完整
provider chain：任一 CUDA/TensorRT 为 true，全部明确为 CPU 才是 false，私有链缺失、
builder/清理中或清理失败为 unknown。

backend 实现 `POST /drain`、`/drain/cancel`、受管 `POST /unload`、
`/lifecycle/mode` 和 `/lifecycle/reset`。`/predict` 与 `/warmup` 在读取业务 body 前完成
admission。无 body 的 `/unload` 继续作为 legacy best-effort 兼容路径，不能单独作为
平台减账证据。

启动阶段不创建临时 ORT session。只有受 admission 保护的首个 builder 才构造业务
session。CUDA composite 构造异常只在确认为设备错误时才尝试 CPU 替代；即使替代成功，
由于无法直接观测部分构造的私有所有权链，residency 仍保持 unknown，直到一次成功的
全池清理。平台管理的 CUDA 插件使用严格设备门：业务 session 的首 provider 必须是
CUDAExecutionProvider，不允许 CPU 替代。

默认 cap=3 的部署验收会同时构建三个 composite engine，并要求 9 条 det/cls/rec 业务
session 全部以 CUDA 为首 provider。仅在获授权、独占目标设备及模型和输入批准后，
在新 GPU 镜像中运行 `/app/scripts/validate_managed_lifecycle.py`，记录实际镜像、
设备、输入和输出证据。验收脚本不得以旧镜像的证据替代。

`VALIDATION_WEIGHT_SHA256_JSON` 的 key 是验收器选中权重相对 `RAPIDOCR_MODEL_DIR` 的路径，必须与该次
v5 mobile、v5 server 和 v6 medium 三引擎所需权重精确相等。验收会在真实推理后重新读取
9 条 session 的 provider chain，因此 ORT 运行期整 session CPU fallback 也会阻断声明。正式两轮
计量前会先加载并卸载一次三引擎全池，以成熟 ORT/CUDA/cuDNN context 作为稳定回收基线。
验收失败时不能开启声明。

## 接入平台

平台按插件 manifest 注册镜像、资产摘要与能力；`resource_profile.device` 和
`batchable` 决定设备感知队列路由。实际插件认证和生命周期由共享 runtime 实施。
