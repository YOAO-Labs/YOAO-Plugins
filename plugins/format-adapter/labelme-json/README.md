# LabelMe JSON Format Adapter Plugin

English | [简体中文](README.zh-CN.md)

`yoao.format.labelme-json` (`format_id: "labelme-json"`) is a zero-dependency Node.js (`>=20`) ESM format adapter plugin implementing `yoao-format-adapter/v1` and `yoao-extension-manifest/v1`.

It translates between LabelMe JSON files and canonical YOAO annotations without holding database credentials, user JWTs, or platform secrets, and requires no outbound network access.

## Capabilities and endpoints

- **Extension point and protocol**: `format_adapter` (`yoao-format-adapter/v1`)
- **Capabilities**: `import`, `export`
- **Media types**: `image`
- **Supported geometries**:
  - `import`: converts LabelMe `rectangle` (2 points) to `bbox` and `polygon` (3 or more points) to `polygon` with normalized `[0, 1]` coordinates and explicit label mapping. Accepts a single JSON object or an array of objects (request body limit `16 MiB`).
  - `export`: renders YOAO `bbox` and `polygon` tasks into LabelMe JSON (`version: "5.0.0"`, `imageData: null`).
- **Import option (`option_schema`)**: `ignore_unsupported_shapes` (boolean, default `true`) skips shapes other than `rectangle` and `polygon` with warnings; set to `false` to reject them.

| Endpoint | Auth | Description |
| --- | --- | --- |
| `GET /health` | None | Liveness probe returning `{"status": "ok"}` |
| `GET /setup` | Bearer token | Returns adapter metadata, capabilities, and `option_schema` |
| `POST /import/parse` | Bearer token | Parses base64-encoded LabelMe JSON into YOAO annotations |
| `POST /export/render` | Bearer token | Renders YOAO tasks into base64-encoded LabelMe JSON files |

## Running

**Bare Node.js process:**

```sh
cd plugins/format-adapter/labelme-json
YOAO_FORMAT_ADAPTER_TOKEN="replace-with-random-token" \
HOST=127.0.0.1 \
PORT=8092 \
node src/server.mjs
```

**Hardened OCI container:**

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

Register the endpoint URL and `YOAO_FORMAT_ADAPTER_TOKEN` in YOAO's admin format-adapter settings (`/api/v1/admin/format-adapters`).

## Configuration

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `YOAO_FORMAT_ADAPTER_TOKEN` | Yes | None | Bearer token required from YOAO |
| `HOST` | No | `127.0.0.1` (`0.0.0.0` in image) | HTTP listen host |
| `PORT` | No | `8092` | HTTP listen port (`0` to `65535`) |

## Runtime requirements

- **Runtime**: Node.js `>=20.0.0` (container image uses `node:22-alpine`), zero third-party npm dependencies
- **Resources and container**: `manifest.json` declares `memory_mb: 128` and `allowed_egress_hosts: []`; runs as non-root with a read-only root filesystem
- **Tests**: run `node --test test/plugin.test.mjs` inside `plugins/format-adapter/labelme-json`

## License

Licensed under [MIT](../../../LICENSE).
