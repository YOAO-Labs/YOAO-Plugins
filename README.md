# YOAO Plugins

YOAO 推理插件与开发工具。第一方代码采用 [MIT](LICENSE)；第三方代码、模型权重与容器依赖继续适用各自的许可，本声明不重新授权它们。模型权重、训练产物、数据集和凭据不提交到本仓库。

源码边界：插件开发 SDK、算法侧共享运行时、Mask 工具和 LibreYOLO 原生适配由本仓库持有；平台 API SDK、业务 owner、权限与持久化规则留在 `YOAO-Labs/ai-annotation-platform`。平台维护的任务、协议、生命周期、推理与训练契约通过固定版本和 SHA256 的公开制品消费，不复制维护第二份源码或依赖父目录 checkout。

GitHub Releases 中的契约制品以其 manifest 记录的来源仓库、源码提交与逐文件摘要为准；发布标签所在仓库不改变源码所有权。容器与真实业务资格独立于源码公开和 Python 包发布。

## Installed author workflow

The public contract release `contracts-0a21bc402432` is platform-owned; its manifest SHA256 is `b38851f5e736a760021c226039562c2afc9efd21c8ab229f0bb403bab7eca292`. First-party wheels are published under `python-6118885dd686`. After that release is published, use CPython 3.11 and uv:

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
uv run aap-plugin dev --device cpu --asset-dir /private/libreyolo/assets --token-file /private/libreyolo/token
```

Review the first-party release manifest and use the lockfile's hashes to pin all consumer inputs. `prepare-assets` downloads separately licensed weights; inspect the exact license before use. The token is a local developer token, not a platform administrator credential. To check a running endpoint, set `AAP_PLUGIN_TOKEN` to that token and run `uv run aap-plugin check --url http://127.0.0.1:8000 --fixtures /private/fixtures.json`; fixtures must provide real `multi` and `empty` images and reviewed expected detections, boxes and tolerances as specified by `load_fixtures` in the SDK. Local preflight does not qualify a managed platform deployment. See [LibreYOLO](plugins/libreyolo/README.md) for independent OCI build, publishing and qualification.

## Building release inputs

From a clean clone, in an isolated Python 3.11 builder with `build==1.3.0`, `setuptools==80.9.0`, and `wheel==0.45.1`:

```sh
python scripts/build-plugin-artifacts.py --revision HEAD --output /private/plugin-release
```

The builder archives the named commit, verifies the public contract manifest and controlled asset bytes, and emits five first-party wheels, source distributions, source archive, `manifest.json` and `SHA256SUMS`. It does not upload, include weights, or bundle third-party wheels. For the separate, private hash-locked candidate and OCI workflow see [LibreYOLO](plugins/libreyolo/README.md). Runtime and mask packages support Python 3.10; LibreYOLO and the author SDK target Python 3.11.
