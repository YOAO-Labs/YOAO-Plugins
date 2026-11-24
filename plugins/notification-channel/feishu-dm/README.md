# YOAO 飞书私信通知渠道插件（零依赖 Node.js 实现）

独立于 Python `Plugin-SDK` 的参考通知渠道插件，使用 Node.js 20+ 原生 ESM 实现 `yoao-notification-channel/v1` 协议与 `yoao-extension-manifest/v1` 清单。

该插件只做一件事：接收 YOAO 平台投递的脱敏 5 字段信封（ADR-0025），使用插件自己保管的飞书自建应用凭据，向绑定用户的飞书 `open_id` 发送交互式卡片消息。插件不接触平台数据库、用户 JWT、平台管理凭据或用户列表。

## 前置条件

1. 在飞书开放平台创建一个企业自建应用并启用机器人能力。
2. 在权限管理中开通 `im:message:send_as_bot`（以应用身份发消息），将应用发布并对目标测试成员可见。
3. 准备好应用凭证 `FEISHU_APP_ID`、`FEISHU_APP_SECRET` 以及测试收件人的 `open_id`（`ou_` 开头）。
4. 准备一个由管理员生成的随机通道鉴权密钥 `YOAO_CHANNEL_AUTH_TOKEN`（平台注册通道时填写相同值）。

## 方式一：直连档运行（无需 Docker）

需要本机或内网机器安装 Node.js 20+：

```bash
cd plugins/notification-channel/feishu-dm

export YOAO_CHANNEL_HOST=0.0.0.0
export YOAO_CHANNEL_PORT=8090
export YOAO_CHANNEL_AUTH_TOKEN="replace-with-strong-random-token"
export FEISHU_APP_ID="cli_xxxxxxxx"
export FEISHU_APP_SECRET="xxxxxxxxxxxxxxxx"
export YOAO_PUBLIC_BASE_URL="https://yoao.example.com"

node src/server.mjs
```

也可通过 `*_FILE` 变量从只读凭据文件读取密钥（与直传环境变量二选一）：

- `YOAO_CHANNEL_AUTH_TOKEN_FILE`
- `FEISHU_APP_ID_FILE`
- `FEISHU_APP_SECRET_FILE`

## 方式二：安装档镜像运行（受限非 root 容器）

构建镜像（镜像标签内嵌 `io.yoao.plugin.manifest`）：

```bash
docker build -t yoao-feishu-dm:1.0.0 plugins/notification-channel/feishu-dm
```

以只读根文件系统、零特权、128 MiB 内存上限运行：

```bash
docker run -d \
  --name yoao-feishu-dm \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --memory 128m \
  --user 1000:1000 \
  -p 172.17.0.1:8090:8090 \
  -e YOAO_CHANNEL_AUTH_TOKEN="replace-with-strong-random-token" \
  -e FEISHU_APP_ID="cli_xxxxxxxx" \
  -e FEISHU_APP_SECRET="xxxxxxxxxxxxxxxx" \
  -e YOAO_PUBLIC_BASE_URL="https://yoao.example.com" \
  yoao-feishu-dm:1.0.0
```

## 在 YOAO 平台注册并启用

确保 YOAO API 与后台通知投递 worker 已配置独立加密密钥 `NOTIFICATION_CHANNEL_ENCRYPTION_KEY`；若插件运行在内网或开发期使用 HTTP，还需将插件主机加入 `NOTIFICATION_CHANNEL_HOST_ALLOWLIST`（及按需 `NOTIFICATION_CHANNEL_ALLOW_INSECURE_HTTP_HOSTS`）。

1. 超级管理员注册通道（可传 `"status": "prepared"` 先预注册，或省略直接激活；`channel_id` 与展示信息自动从插件 `GET /setup` 读取）：

```bash
curl -X POST "$YOAO_BASE_URL/api/v1/admin/notification-channels" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "direct",
    "source_tier": "official",
    "status": "prepared",
    "endpoint_url": "http://172.17.0.1:8090",
    "auth_token": "replace-with-strong-random-token"
  }'
```

2. 若上一步选择 `prepared`，超级管理员再启用通道（按返回中的 `revision` 提交 `expected_revision`，首次注册后为 `1`）：

```bash
curl -X POST "$YOAO_BASE_URL/api/v1/admin/notification-channels/feishu_dm/enable" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"expected_revision": 1}'
```

3. 平台成员绑定自己的飞书 `open_id` 并发送验证卡片：

```bash
curl -X PUT "$YOAO_BASE_URL/api/v1/notification-channels/feishu_dm/binding" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"address": {"open_id": "ou_xxxxxxxx"}, "locale": "zh-CN"}'

curl -X POST "$YOAO_BASE_URL/api/v1/notification-channels/feishu_dm/verify-address" \
  -H "Authorization: Bearer $USER_TOKEN"
```

4. 成员在通知偏好中为所需事件开启 `external_channels` 下的 `feishu_dm`（例如 `{"type": "task.rejected", "external_channels": {"feishu_dm": true}}`）。

## 环境变量一览

| 变量                                 | 必填 | 默认值                   | 说明                                           |
| ------------------------------------ | ---- | ------------------------ | ---------------------------------------------- |
| `YOAO_CHANNEL_HOST`                  | 否   | `127.0.0.1`              | 监听地址（容器镜像默认 `0.0.0.0`）             |
| `YOAO_CHANNEL_PORT`                  | 否   | `8090`                   | 监听端口                                       |
| `YOAO_CHANNEL_AUTH_TOKEN` / `*_FILE` | 是   | 无                       | YOAO 平台调用本插件时使用的 Bearer 密钥        |
| `FEISHU_APP_ID` / `*_FILE`           | 是   | 无                       | 飞书企业自建应用 App ID                        |
| `FEISHU_APP_SECRET` / `*_FILE`       | 是   | 无                       | 飞书企业自建应用 App Secret                    |
| `FEISHU_API_BASE_URL`                | 否   | `https://open.feishu.cn` | 飞书开放平台基址（隔离验收可指向替身服务）     |
| `YOAO_PUBLIC_BASE_URL`               | 否   | 空                       | YOAO 外部访问基址，用于拼接卡片跳转按钮        |
| `FEISHU_HTTP_TIMEOUT_MS`             | 否   | `8000`                   | 单次投递共享的飞书上游总超时毫秒数（500–9000） |
| `FEISHU_DEDUP_TTL_MS`                | 否   | `86400000`               | 内存 `delivery_id` 幂等缓存保留毫秒数          |
| `FEISHU_DEDUP_MAX_ENTRIES`           | 否   | `10000`                  | 内存幂等缓存最大条目数                         |

## 本地测试

```bash
node --test test/*.test.mjs
```
