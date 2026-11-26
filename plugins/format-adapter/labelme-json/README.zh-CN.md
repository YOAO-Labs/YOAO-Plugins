# LabelMe JSON 格式适配器插件

[English](README.md) | 简体中文

`yoao.format.labelme-json`（`format_id: "labelme-json"`）是零依赖的 Node.js（`>=20`）原生 ESM 格式适配器插件，实现 `yoao-format-adapter/v1` 协议与 `yoao-extension-manifest/v1` 清单规范。

插件负责 LabelMe JSON 与 YOAO 标准标注之间的双向转换，不持有平台数据库凭据、用户 JWT 或管理密钥，且无需任何外部网络访问。

## 能力与接口

- **扩展点与协议**：`format_adapter`（`yoao-format-adapter/v1`）
- **支持能力**：`import`、`export`
- **媒体类型**：`image`
- **支持的几何类型**：
  - `import`：将 LabelMe `rectangle`（2 点）转为 `bbox`、`polygon`（不少于 3 点）转为 `polygon`，归一化至 `[0, 1]` 坐标并应用显式标签映射。支持单个 JSON 对象或对象数组（请求体上限 `16 MiB`）。
  - `export`：将 YOAO 的 `bbox` 与 `polygon` 任务渲染为 LabelMe JSON 文件（`version: "5.0.0"`、`imageData: null`）。
- **导入选项（`option_schema`）**：`ignore_unsupported_shapes`（布尔值，默认 `true`），跳过 `rectangle` 与 `polygon` 之外的图形并返回告警；设为 `false` 时直接报错拒绝。

| 接口 | 鉴权 | 说明 |
| --- | --- | --- |
| `GET /health` | 无 | 存活检查，返回 `{"status": "ok"}` |
| `GET /setup` | Bearer 令牌 | 返回适配器元信息、能力列表与 `option_schema` |
| `POST /import/parse` | Bearer 令牌 | 将 Base64 编码的 LabelMe JSON 解析为 YOAO 标注 |
| `POST /export/render` | Bearer 令牌 | 将 YOAO 任务渲染为 Base64 编码的 LabelMe JSON 文件 |

## 运行方式

**直接以 Node.js 运行：**

```sh
cd plugins/format-adapter/labelme-json
YOAO_FORMAT_ADAPTER_TOKEN="replace-with-random-token" \
HOST=127.0.0.1 \
PORT=8092 \
node src/server.mjs
```

**受限非 root 容器运行：**

```sh
docker build -t yoao-format-labelme-json:1.0.0 plugins/format-adapter/labelme-json
docker run --rm \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --memory 128m \
  --user 1000:1000 \
  -e YOAO_FORMAT_ADAPTER_TOKEN="replace-with-random-token" \
  -p 127.0.0.1:8092:8092 \
  yoao-format-labelme-json:1.0.0
```

在 YOAO 管理端格式适配器设置（`/api/v1/admin/format-adapters`）中填写插件地址与 `YOAO_FORMAT_ADAPTER_TOKEN` 即可完成注册。

## 关键配置

| 环境变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `YOAO_FORMAT_ADAPTER_TOKEN` | 是 | 无 | YOAO 平台调用本插件的 Bearer 密钥 |
| `HOST` | 否 | `127.0.0.1`（镜像默认 `0.0.0.0`） | HTTP 监听地址 |
| `PORT` | 否 | `8092` | HTTP 监听端口（`0` 至 `65535`） |

## 运行环境与依赖

- **运行时**：Node.js `>=20.0.0`（容器镜像基于 `node:22-alpine`），零第三方 npm 运行时依赖
- **资源与容器**：`manifest.json` 声明 `memory_mb: 128` 与 `allowed_egress_hosts: []`，支持非 root 用户与只读根文件系统
- **本地测试**：在 `plugins/format-adapter/labelme-json` 目录运行 `node --test test/plugin.test.mjs`

## 许可证

采用 [MIT](../../../LICENSE) 许可证。
