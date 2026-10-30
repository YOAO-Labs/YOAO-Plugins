# YOAO Plugins

YOAO 推理插件与开发工具。第一方代码采用 [MIT](LICENSE)；第三方代码、模型权重与容器依赖继续适用各自的许可，本声明不重新授权它们。模型权重、训练产物、数据集和凭据不提交到本仓库。

源码边界：插件开发 SDK、算法侧共享运行时、Mask 工具和 LibreYOLO 原生适配由本仓库持有；平台 API SDK、业务 owner、权限与持久化规则留在 `YOAO-Labs/ai-annotation-platform`。平台维护的任务、协议、生命周期、推理与训练契约通过固定版本和 SHA256 的公开制品消费，不复制维护第二份源码或依赖父目录 checkout。

GitHub Releases 中的契约制品以其 manifest 记录的来源仓库、源码提交与逐文件摘要为准；发布标签所在仓库不改变源码所有权。容器与真实业务资格独立于源码公开和 Python 包发布。
