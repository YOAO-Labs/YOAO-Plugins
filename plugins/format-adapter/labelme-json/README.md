# YOAO LabelMe JSON Format Adapter Plugin

Zero-dependency Node.js (`>=20`) reference format adapter implementing `yoao-format-adapter/v1` and `yoao-extension-manifest/v1` (`extension_point: "format_adapter"`).

## Capabilities

- `import`: translates LabelMe JSON (`rectangle` -> `bbox`, `polygon` -> `polygon`) into canonical YOAO annotations with explicit label mapping.
- `export`: renders canonical YOAO `bbox` and `polygon` tasks into LabelMe JSON files.
- Holds no database credentials, user JWTs, or platform secrets.

## Direct Tier (bare process)

```bash
YOAO_FORMAT_ADAPTER_TOKEN=replace-with-random-token HOST=127.0.0.1 PORT=8092 node src/server.mjs
```

Register with YOAO as a super admin:

```bash
POST /api/v1/admin/format-adapters
{
  "endpoint_url": "http://127.0.0.1:8092",
  "auth_token": "replace-with-random-token",
  "mode": "direct",
  "source_tier": "official",
  "activate": true
}
```

## Managed Tier (hardened OCI container)

```bash
docker build -t yoao-format-labelme-json:1.0.0 .
docker run --rm --read-only --cap-drop ALL --security-opt no-new-privileges \
  --memory 128m --user 1000:1000 \
  -e YOAO_FORMAT_ADAPTER_TOKEN=replace-with-random-token \
  -p 127.0.0.1:8092:8092 yoao-format-labelme-json:1.0.0
```

## Tests

```bash
node --test test/plugin.test.mjs
```
