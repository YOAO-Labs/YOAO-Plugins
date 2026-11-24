import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { buildNotificationCard, buildVerifyAddressCard } from "./cards.mjs";
import { loadConfig } from "./config.mjs";
import { FeishuClient } from "./feishu_client.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MANIFEST = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, "..", "manifest.json"), "utf8"),
);

export const PROTOCOL_VERSION = "yoao-notification-channel/v1";
export const CHANNEL_ID = MANIFEST.extension.channel_id;
const MAX_REQUEST_BYTES = 36 * 1024;
const MAX_ADDRESS_BYTES = 2048;

const SETUP_RESPONSE = Object.freeze({
  protocol_version: PROTOCOL_VERSION,
  channel_id: CHANNEL_ID,
  display_name: MANIFEST.display_name,
  description: MANIFEST.description,
  homepage: MANIFEST.homepage,
  capabilities: MANIFEST.capabilities,
  address_schema: MANIFEST.extension.address_schema,
});

function sha256Digest(value) {
  return crypto.createHash("sha256").update(value, "utf8").digest();
}

function bearerAuthorized(req, expectedDigest) {
  const header = req.headers.authorization;
  if (typeof header !== "string" || !header.startsWith("Bearer ")) {
    return false;
  }
  const token = header.slice("Bearer ".length);
  if (!token) {
    return false;
  }
  return crypto.timingSafeEqual(sha256Digest(token), expectedDigest);
}

function sendJson(res, statusCode, payload, extraHeaders = {}) {
  if (res.writableEnded || res.destroyed) {
    return;
  }
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  res.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": String(body.length),
    ...extraHeaders,
  });
  res.end(body);
}

async function readJsonBody(req) {
  const declaredLen = Number(req.headers["content-length"]);
  if (Number.isFinite(declaredLen) && declaredLen > MAX_REQUEST_BYTES) {
    const err = new Error("request_body_too_large");
    err.statusCode = 413;
    err.clientReason = "request_body_too_large";
    err.closeConnection = true;
    throw err;
  }
  const chunks = [];
  let total = 0;
  for await (const chunk of req) {
    total += chunk.length;
    if (total > MAX_REQUEST_BYTES) {
      const err = new Error("request_body_too_large");
      err.statusCode = 413;
      err.clientReason = "request_body_too_large";
      err.closeConnection = true;
      throw err;
    }
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString("utf8");
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    const err = new Error("invalid_json_body");
    err.statusCode = 400;
    err.clientReason = "invalid_json_body";
    throw err;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    const err = new Error("request_body_must_be_object");
    err.statusCode = 400;
    err.clientReason = "request_body_must_be_object";
    throw err;
  }
  return parsed;
}

function extractValidOpenId(address) {
  if (!address || typeof address !== "object" || Array.isArray(address)) {
    return null;
  }
  const keys = Object.keys(address);
  if (keys.length !== 1 || keys[0] !== "open_id") {
    return null;
  }
  if (Buffer.byteLength(JSON.stringify(address), "utf8") > MAX_ADDRESS_BYTES) {
    return null;
  }
  const openId = address.open_id;
  if (
    typeof openId !== "string" ||
    openId.length < 4 ||
    openId.length > 128 ||
    /[\x00-\x1f\x7f\s]/.test(openId)
  ) {
    return null;
  }
  return openId;
}

function isValidEnvelope(envelope) {
  return (
    Boolean(envelope) &&
    typeof envelope === "object" &&
    !Array.isArray(envelope) &&
    envelope.event_version === "1.0" &&
    typeof envelope.event === "string" &&
    envelope.event.length >= 1 &&
    envelope.event.length <= 60 &&
    typeof envelope.delivery_id === "string" &&
    envelope.delivery_id.length >= 1 &&
    envelope.delivery_id.length <= 64 &&
    typeof envelope.occurred_at === "string" &&
    envelope.occurred_at.length >= 1 &&
    envelope.occurred_at.length <= 64 &&
    Boolean(envelope.data) &&
    typeof envelope.data === "object" &&
    !Array.isArray(envelope.data)
  );
}

class DeliveryIdempotencyCache {
  constructor({
    ttlMs = 86_400_000,
    maxEntries = 10_000,
    now = () => Date.now(),
  } = {}) {
    this._ttlMs = ttlMs;
    this._maxEntries = maxEntries;
    this._now = now;
    this._map = new Map();
  }

  get(deliveryId) {
    const entry = this._map.get(deliveryId);
    if (!entry) {
      return null;
    }
    if (entry.expiresAt <= this._now()) {
      this._map.delete(deliveryId);
      return null;
    }
    return entry.value;
  }

  set(deliveryId, value) {
    if (this._map.has(deliveryId)) {
      this._map.delete(deliveryId);
    } else if (this._map.size >= this._maxEntries) {
      const oldestKey = this._map.keys().next().value;
      if (oldestKey !== undefined) {
        this._map.delete(oldestKey);
      }
    }
    this._map.set(deliveryId, {
      value,
      expiresAt: this._now() + this._ttlMs,
    });
  }
}

function sendOutcome(res, outcome) {
  if (outcome.status === "delivered") {
    sendJson(res, 200, outcome);
    return;
  }
  if (outcome.status === "permanent") {
    sendJson(res, 422, outcome);
    return;
  }
  const headers =
    typeof outcome.retry_after_seconds === "number"
      ? { "Retry-After": String(outcome.retry_after_seconds) }
      : {};
  sendJson(res, 503, outcome, headers);
}

function closeHeadersIfBodyUnread(req) {
  if (req.headers["content-length"] || req.headers["transfer-encoding"]) {
    return { Connection: "close" };
  }
  return {};
}

export function createPluginServer({
  config,
  feishuClient,
  now = () => Date.now(),
}) {
  const expectedTokenDigest = sha256Digest(config.channelAuthToken);
  const client =
    feishuClient ??
    new FeishuClient({
      appId: config.feishuAppId,
      appSecret: config.feishuAppSecret,
      apiBaseUrl: config.feishuApiBaseUrl,
      httpTimeoutMs: config.httpTimeoutMs,
      now,
    });
  const dedup = new DeliveryIdempotencyCache({
    ttlMs: config.dedupTtlMs,
    maxEntries: config.dedupMaxEntries,
    now,
  });
  const inflight = new Map();

  const server = http.createServer((req, res) => {
    void (async () => {
      const method = req.method ?? "GET";
      const pathname = (req.url ?? "/").split("?")[0];

      if (method === "GET" && pathname === "/health") {
        sendJson(res, 200, { status: "ok" });
        return;
      }

      if (!bearerAuthorized(req, expectedTokenDigest)) {
        sendJson(
          res,
          401,
          { status: "retryable", reason: "unauthorized" },
          closeHeadersIfBodyUnread(req),
        );
        return;
      }

      if (method === "GET" && pathname === "/setup") {
        sendJson(res, 200, SETUP_RESPONSE);
        return;
      }

      if (method === "POST" && pathname === "/deliver") {
        let body;
        try {
          body = await readJsonBody(req);
        } catch (err) {
          sendJson(
            res,
            err.statusCode ?? 400,
            {
              status: "permanent",
              reason: err.clientReason ?? "invalid_request_body",
            },
            err.closeConnection ? { Connection: "close" } : {},
          );
          return;
        }

        if (
          body.protocol_version !== PROTOCOL_VERSION ||
          body.channel_id !== CHANNEL_ID
        ) {
          sendJson(res, 422, {
            status: "permanent",
            reason: "protocol_or_channel_mismatch",
          });
          return;
        }
        const envelope = body.envelope;
        if (!isValidEnvelope(envelope)) {
          sendJson(res, 422, {
            status: "permanent",
            reason: "invalid_envelope",
          });
          return;
        }
        const openId = extractValidOpenId(body.recipient_address);
        if (!openId) {
          sendJson(res, 422, {
            status: "permanent",
            reason: "invalid_recipient_address",
          });
          return;
        }

        const deliveryId = envelope.delivery_id;
        const cached = dedup.get(deliveryId);
        if (cached) {
          sendJson(res, 200, cached);
          return;
        }

        let pending = inflight.get(deliveryId);
        if (!pending) {
          const card = buildNotificationCard({
            envelope,
            locale: body.locale,
            publicBaseUrl: config.publicBaseUrl,
          });
          pending = client.sendInteractiveCard({
            openId,
            card,
            uuid: deliveryId,
          });
          inflight.set(deliveryId, pending);
        }
        let outcome;
        try {
          outcome = await pending;
        } finally {
          if (inflight.get(deliveryId) === pending) {
            inflight.delete(deliveryId);
          }
        }

        if (outcome.status === "delivered") {
          dedup.set(deliveryId, outcome);
        }
        sendOutcome(res, outcome);
        return;
      }

      if (method === "POST" && pathname === "/verify-address") {
        let body;
        try {
          body = await readJsonBody(req);
        } catch (err) {
          sendJson(
            res,
            err.statusCode ?? 400,
            {
              status: "permanent",
              reason: err.clientReason ?? "invalid_request_body",
            },
            err.closeConnection ? { Connection: "close" } : {},
          );
          return;
        }

        if (
          body.protocol_version !== PROTOCOL_VERSION ||
          body.channel_id !== CHANNEL_ID
        ) {
          sendJson(res, 422, {
            status: "permanent",
            reason: "protocol_or_channel_mismatch",
          });
          return;
        }
        if (typeof body.verification_id !== "string" || !body.verification_id) {
          sendJson(res, 422, {
            status: "permanent",
            reason: "invalid_verification_id",
          });
          return;
        }
        const openId = extractValidOpenId(body.recipient_address);
        if (!openId) {
          sendJson(res, 422, {
            status: "permanent",
            reason: "invalid_recipient_address",
          });
          return;
        }

        const card = buildVerifyAddressCard({
          locale: body.locale,
          verificationId: body.verification_id,
        });
        const outcome = await client.sendInteractiveCard({
          openId,
          card,
          uuid: body.verification_id,
        });
        sendOutcome(res, outcome);
        return;
      }

      sendJson(
        res,
        404,
        { status: "permanent", reason: "not_found" },
        closeHeadersIfBodyUnread(req),
      );
    })().catch(() => {
      sendJson(
        res,
        500,
        { status: "retryable", reason: "internal_error" },
        { Connection: "close" },
      );
    });
  });

  server.headersTimeout = 10_000;
  server.requestTimeout = 15_000;
  server.keepAliveTimeout = 5_000;
  return server;
}

const isEntrypoint =
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isEntrypoint) {
  const config = loadConfig(process.env);
  const server = createPluginServer({ config });
  server.listen(config.port, config.host, () => {
    const addr = server.address();
    const port = typeof addr === "object" && addr ? addr.port : config.port;
    console.log(
      JSON.stringify({
        event: "yoao_feishu_dm_channel_started",
        host: config.host,
        port,
        channel_id: CHANNEL_ID,
        protocol_version: PROTOCOL_VERSION,
      }),
    );
  });

  const shutdown = () => {
    server.close(() => process.exit(0));
    server.closeIdleConnections?.();
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}
