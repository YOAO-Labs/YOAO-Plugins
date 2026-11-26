# 飞书私信通知渠道插件

[English](README.md) | 简体中文

`yoao-feishu-dm`（`channel_id: "feishu_dm"`）是零依赖的 Node.js（`>=20`）原生 ESM 通知渠道插件，实现 `yoao-notification-channel/v1` 协议与 `yoao-extension-manifest/v1` 清单规范。

插件接收 YOAO 平台投递的脱敏 5 字段事件信封，使用自身保管的飞书企业自建应用凭据，向绑定用户的飞书 `open_id` 发送交互式卡片消息。插件不接触平台数据库、用户 JWT、平台管理凭据或用户列表。

## 能力与接口

- **扩展点与协议**：`notification_channel`（`yoao-notification-channel/v1`）
- **支持能力**：`deliver`、`verify_address`
- **收件地址格式**：`{"open_id": "ou_..."}`（4 至 128 个字符，不含空白或控制字符）
- **允许出站主机**：`open.feishu.cn`

| 接口 | 鉴权 | 说明 |
| --- | --- | --- |
| `GET /health` | 无 | 存活检查，返回 `{"status": "ok"}` |
| `GET /setup` | Bearer 令牌 | 返回通道元信息、能力列表与 `address_schema` |
| `POST /deliver` | Bearer 令牌 | 发送通知卡片，内置基于 `delivery_id` 的内存幂等去重 |
| `POST /verify-address` | Bearer 令牌 | 发送地址验证卡片以确认用户的 `open_id` |

## 前置条件与运行方式

1. 在飞书开放平台创建企业自建应用并启用机器人能力。
2. 开通 `im:message:send_as_bot`（以应用身份发消息）权限，并将应用发布对目标成员可见。
3. 准备好 `FEISHU_APP_ID`、`FEISHU_APP_SECRET`，以及用于平台注册通道的随机 Bearer 令牌 `YOAO_CHANNEL_AUTH_TOKEN`。

**直接以 Node.js 运行：**

```sh
cd plugins/notification-channel/feishu-dm
YOAO_CHANNEL_HOST=0.0.0.0 \
YOAO_CHANNEL_PORT=8090 \
YOAO_CHANNEL_AUTH_TOKEN="replace-with-strong-random-token" \
FEISHU_APP_ID="cli_xxxxxxxx" \
FEISHU_APP_SECRET="xxxxxxxxxxxxxxxx" \
YOAO_PUBLIC_BASE_URL="https://yoao.example.com" \
node src/server.mjs
```

**受限非 root 容器运行：**

```sh
docker build -t yoao-feishu-dm:1.0.0 plugins/notification-channel/feishu-dm
docker run -d \
  --name yoao-feishu-dm \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --memory 128m \
  --user 1000:1000 \
  -p 127.0.0.1:8090:8090 \
  -e YOAO_CHANNEL_AUTH_TOKEN="replace-with-strong-random-token" \
  -e FEISHU_APP_ID="cli_xxxxxxxx" \
  -e FEISHU_APP_SECRET="xxxxxxxxxxxxxxxx" \
  -e YOAO_PUBLIC_BASE_URL="https://yoao.example.com" \
  yoao-feishu-dm:1.0.0
```

在 YOAO 管理端通知渠道设置（`/api/v1/admin/notification-channels`）中填写插件地址与 `YOAO_CHANNEL_AUTH_TOKEN` 并启用 `feishu_dm` 后，成员即可在通知偏好中绑定并验证自己的飞书 `open_id`。

## 关键配置

密钥类变量支持直接传值或通过 `*_FILE` 指定只读凭据文件路径（每组二选一，不可同时设置）。

| 环境变量 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `YOAO_CHANNEL_HOST` | 否 | `127.0.0.1`（镜像默认 `0.0.0.0`） | HTTP 监听地址 |
| `YOAO_CHANNEL_PORT` | 否 | `8090` | HTTP 监听端口（`0` 至 `65535`） |
| `YOAO_CHANNEL_AUTH_TOKEN` / `*_FILE` | 是 | 无 | YOAO 平台调用本插件的 Bearer 密钥 |
| `FEISHU_APP_ID` / `*_FILE` | 是 | 无 | 飞书企业自建应用 App ID |
| `FEISHU_APP_SECRET` / `*_FILE` | 是 | 无 | 飞书企业自建应用 App Secret |
| `FEISHU_API_BASE_URL` | 否 | `https://open.feishu.cn` | 飞书开放平台 API 基址 |
| `YOAO_PUBLIC_BASE_URL` | 否 | 空 | YOAO 外部访问基址，用于生成卡片跳转链接 |
| `FEISHU_HTTP_TIMEOUT_MS` | 否 | `8000` | 单次投递共享的飞书上游总超时毫秒数（`500` 至 `9000`） |
| `FEISHU_DEDUP_TTL_MS` | 否 | `86400000` | 内存 `delivery_id` 幂等缓存保留毫秒数（`1000` 至 `604800000`） |
| `FEISHU_DEDUP_MAX_ENTRIES` | 否 | `10000` | 内存幂等缓存最大条目数（`10` 至 `100000`） |

## 运行环境与依赖

- **运行时**：Node.js `>=20.0.0`（容器镜像基于 `node:22-alpine`），零第三方 npm 运行时依赖
- **资源与容器**：`manifest.json` 声明 `memory_mb: 128`，支持非 root 用户与只读根文件系统
- **本地测试**：在 `plugins/notification-channel/feishu-dm` 目录运行 `node --test test/*.test.mjs`

## 许可证

采用 [MIT](../../../LICENSE) 许可证。
