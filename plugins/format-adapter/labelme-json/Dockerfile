FROM node:22-alpine

WORKDIR /app

COPY --chown=root:root package.json manifest.json ./
COPY --chown=root:root src ./src

LABEL io.yoao.plugin.manifest='{"manifest_schema":"yoao-extension-manifest/v1","extension_point":"format_adapter","plugin_id":"yoao.format.labelme-json","version":"1.0.0","display_name":"LabelMe JSON","description":"Zero-dependency Node.js LabelMe JSON rectangle and polygon format adapter.","homepage":"https://github.com/yoao-ai/yoao-plugins","publisher":"YOAO Official","license":"MIT","protocol_version":"yoao-format-adapter/v1","entrypoint":["node","src/server.mjs"],"capabilities":["import","export"],"resource":{"memory_mb":128},"extension":{"format_id":"labelme-json","adapter_version":"1.0.0","manifest_version":"1","media_types":["image"],"option_schema":{"type":"object","additionalProperties":false,"properties":{"ignore_unsupported_shapes":{"type":"boolean","default":true,"description":"Skip shape types other than rectangle and polygon."}}},"allowed_egress_hosts":[]}}'

ENV HOST=0.0.0.0
ENV PORT=8092
EXPOSE 8092

USER node
CMD ["node", "src/server.mjs"]
