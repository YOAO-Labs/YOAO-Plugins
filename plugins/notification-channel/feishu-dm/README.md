# Feishu DM Notification Channel Plugin

English | [简体中文](README.zh-CN.md)

`yoao-feishu-dm` (`channel_id: "feishu_dm"`) is a zero-dependency Node.js (`>=20`) ESM notification channel plugin implementing `yoao-notification-channel/v1` and `yoao-extension-manifest/v1`.

It receives sanitized 5-field notification envelopes from YOAO and delivers interactive Feishu cards to a bound user's `open_id` using its own Feishu enterprise app credentials. The plugin has no access to the platform database, user JWTs, admin credentials, or user lists.

## Capabilities and endpoints

- **Extension point and protocol**: `notification_channel` (`yoao-notification-channel/v1`)
- **Capabilities**: `deliver`, `verify_address`
- **Recipient address schema**: `{"open_id": "ou_..."}` (4 to 128 characters, no whitespace or control characters)
- **Allowed egress host**: `open.feishu.cn`

| Endpoint | Auth | Description |
| --- | --- | --- |
| `GET /health` | None | Liveness probe returning `{"status": "ok"}` |
| `GET /setup` | Bearer token | Returns channel metadata, capabilities, and `address_schema` |
| `POST /deliver` | Bearer token | Sends an interactive notification card with in-memory `delivery_id` deduplication |
| `POST /verify-address` | Bearer token | Sends a verification card to confirm a user's `open_id` |

## Prerequisites and running

1. Create a Feishu custom enterprise app and enable the bot capability.
2. Grant the `im:message:send_as_bot` permission and publish the app to target members.
3. Obtain `FEISHU_APP_ID`, `FEISHU_APP_SECRET`, and a random Bearer token (`YOAO_CHANNEL_AUTH_TOKEN`) shared with YOAO during channel registration.

**Bare Node.js process:**

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

**Hardened OCI container:**

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

Register the endpoint URL and `YOAO_CHANNEL_AUTH_TOKEN` in YOAO's admin notification-channel settings (`/api/v1/admin/notification-channels`), enable `feishu_dm`, then let members bind and verify their `open_id` in notification preferences.

## Configuration

Secret variables accept either direct values or `*_FILE` paths pointing to read-only files (configure only one of each pair).

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `YOAO_CHANNEL_HOST` | No | `127.0.0.1` (`0.0.0.0` in image) | HTTP listen host |
| `YOAO_CHANNEL_PORT` | No | `8090` | HTTP listen port (`0` to `65535`) |
| `YOAO_CHANNEL_AUTH_TOKEN` / `*_FILE` | Yes | None | Bearer token required from YOAO |
| `FEISHU_APP_ID` / `*_FILE` | Yes | None | Feishu custom app ID |
| `FEISHU_APP_SECRET` / `*_FILE` | Yes | None | Feishu custom app secret |
| `FEISHU_API_BASE_URL` | No | `https://open.feishu.cn` | Feishu Open Platform base URL |
| `YOAO_PUBLIC_BASE_URL` | No | Empty | Public YOAO base URL used to build card action links |
| `FEISHU_HTTP_TIMEOUT_MS` | No | `8000` | Shared Feishu upstream timeout per delivery in ms (`500` to `9000`) |
| `FEISHU_DEDUP_TTL_MS` | No | `86400000` | In-memory `delivery_id` deduplication TTL in ms (`1000` to `604800000`) |
| `FEISHU_DEDUP_MAX_ENTRIES` | No | `10000` | Maximum entries in the deduplication cache (`10` to `100000`) |

## Runtime requirements

- **Runtime**: Node.js `>=20.0.0` (container image uses `node:22-alpine`), zero third-party npm dependencies
- **Resources and container**: `manifest.json` declares `memory_mb: 128`; runs as non-root with a read-only root filesystem
- **Tests**: run `node --test test/*.test.mjs` inside `plugins/notification-channel/feishu-dm`

## License

Licensed under [MIT](../../../LICENSE).
