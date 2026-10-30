# YOAO Plugins

YOAO 推理插件与开发工具。第一方代码采用 [MIT](LICENSE)；第三方代码、模型权重与容器依赖继续适用各自的许可，本声明不重新授权它们。模型权重、训练产物、数据集和凭据不提交到本仓库。

源码边界：插件开发 SDK、算法侧共享运行时、Mask 工具和 LibreYOLO 原生适配由本仓库持有；平台 API SDK、业务 owner、权限与持久化规则留在 `YOAO-Labs/ai-annotation-platform`。平台维护的任务、协议、生命周期、推理与训练契约通过固定版本和 SHA256 的公开制品消费，不复制维护第二份源码或依赖父目录 checkout。

GitHub Releases 中的契约制品以其 manifest 记录的来源仓库、源码提交与逐文件摘要为准；发布标签所在仓库不改变源码所有权。容器与真实业务资格独立于源码公开和 Python 包发布。

## 已安装环境下的插件开发

公开契约制品 `contracts-0a21bc402432` 归平台维护，其 manifest SHA256 为 `b38851f5e736a760021c226039562c2afc9efd21c8ab229f0bb403bab7eca292`。第一方 wheel 的发布标签为 `python-6118885dd686`。该制品发布后，使用 CPython 3.11 和 uv：

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  'https://github.com/YOAO-Labs/YOAO-Plugins/releases/download/python-6118885dd686/aap_inference_plugin_sdk-0.1.0-py3-none-any.whl' \
  'https://github.com/YOAO-Labs/YOAO-Plugins/releases/download/python-6118885dd686/aap_backend_runtime-0.1.0-py3-none-any.whl' \
  'https://github.com/YOAO-Labs/YOAO-Plugins/releases/download/contracts-0a21bc402432/aap_inference_contract-0.1.0-py3-none-any.whl#sha256=737e5d3346526e7982f8c1cff786aab8fcbacbd0c0a2a76d17a89e8fbb272b39' \
  'https://github.com/YOAO-Labs/YOAO-Plugins/releases/download/contracts-0a21bc402432/aap_lifecycle_contract-0.1.0-py3-none-any.whl#sha256=1655c1db5b7729928e1f056f21c100120f4279fe8ab19808e1322133d3ea7087' \
  'https://github.com/YOAO-Labs/YOAO-Plugins/releases/download/contracts-0a21bc402432/aap_protocol_v2-0.2.0-py3-none-any.whl#sha256=cc4db808a1bfddd1b97675745212d33f4fd090be0693308ec96b609f34e518e7' \
  'https://github.com/YOAO-Labs/YOAO-Plugins/releases/download/contracts-0a21bc402432/aap_task_contract-0.1.0-py3-none-any.whl#sha256=55adf2c7c69e0c58c86774bbe14054e1e973564553716b64215e92ff37495b6a'
.venv/bin/aap-plugin init my-plugin
cd my-plugin
uv lock
uv sync --frozen
uv run aap-plugin prepare-assets --size t --asset-dir /private/libreyolo/assets
uv run aap-plugin dev --entrypoint my-libreyolo-plugin --device cpu --asset-dir /private/libreyolo/assets --token-file /private/libreyolo/token
```

核对第一方发布 manifest，并用 lockfile 中的哈希固定全部消费方输入。`prepare-assets` 下载单独许可的模型权重，使用前须审查对应许可。token 是本地开发 token，不是平台管理员凭据。检查运行中的服务时，设置 `AAP_PLUGIN_TOKEN` 为该 token，执行 `uv run aap-plugin check --url http://127.0.0.1:8000 --fixtures /private/fixtures.json`；fixture 必须包含真实的 `multi` 和 `empty` 图像，以及经审查的预期检测、坐标和容差，具体格式见 SDK 的 `load_fixtures`。本地预检不等于受管平台部署资格。独立 OCI 构建、发布和验收见 [LibreYOLO](plugins/libreyolo/README.md)。

## 构建发布输入

从干净克隆运行；隔离的 Python 3.11 构建环境需安装 `build==1.3.0`、`setuptools==80.9.0` 和 `wheel==0.45.1`：

```sh
python scripts/build-plugin-artifacts.py --revision HEAD --output /private/plugin-release
```

构建器从指定提交创建源码归档，校验公开契约 manifest 和受控资产清单，并生成五个第一方 wheel、源码分发包、源码归档、`manifest.json` 和 `SHA256SUMS`。它不会上传制品，也不包含模型权重或第三方 wheel。独立的私有哈希锁定候选包与 OCI 流程见 [LibreYOLO](plugins/libreyolo/README.md)。运行时和 mask 包支持 Python 3.10；LibreYOLO 与开发 SDK 的目标版本为 Python 3.11。
