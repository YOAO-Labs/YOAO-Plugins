import crypto from "node:crypto";
import http from "node:http";
import { fileURLToPath } from "node:url";

import { loadConfig } from "./config.mjs";
import {
  PROTOCOL_VERSION,
  SETUP_DOCUMENT,
  parseLabelMeImport,
  renderLabelMeExport,
} from "./converter.mjs";

const MAX_BODY_BYTES = 16 * 1024 * 1024;

function sendJson(res, statusCode, payload, { close = false } = {}) {
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  const headers = {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": String(body.length),
  };
  if (close) {
    headers.Connection = "close";
  }
  res.writeHead(statusCode, headers);
  res.end(body);
}

function isAuthorized(req, expectedToken) {
  const header = req.headers.authorization;
  if (typeof header !== "string" || !header.startsWith("Bearer ")) {
    return false;
  }
  const supplied = Buffer.from(header.slice("Bearer ".length), "utf8");
  const expected = Buffer.from(expectedToken, "utf8");
  if (supplied.length !== expected.length) {
    return false;
  }
  return crypto.timingSafeEqual(supplied, expected);
}

async function readJsonBody(req) {
  const chunks = [];
  let total = 0;
  for await (const chunk of req) {
    total += chunk.length;
    if (total > MAX_BODY_BYTES) {
      const err = new Error("request body exceeds 16 MiB");
      err.statusCode = 413;
      throw err;
    }
    chunks.push(chunk);
  }
  const raw = Buffer.concat(chunks).toString("utf8");
  try {
    return JSON.parse(raw);
  } catch {
    const err = new Error("request body must be valid JSON");
    err.statusCode = 400;
    throw err;
  }
}

export function createServer(config) {
  return http.createServer(async (req, res) => {
    try {
      const method = req.method || "GET";
      const pathname = new URL(req.url || "/", "http://localhost").pathname;

      if (method === "GET" && pathname === "/health") {
        sendJson(res, 200, { status: "ok" });
        return;
      }

      if (!isAuthorized(req, config.token)) {
        sendJson(res, 401, { error: "unauthorized" }, { close: true });
        return;
      }

      if (method === "GET" && pathname === "/setup") {
        sendJson(res, 200, SETUP_DOCUMENT);
        return;
      }

      if (method === "POST" && pathname === "/import/parse") {
        const body = await readJsonBody(req);
        if (!body || body.protocol_version !== PROTOCOL_VERSION) {
          sendJson(res, 400, { error: "unsupported protocol_version" });
          return;
        }
        try {
          const result = parseLabelMeImport(body);
          sendJson(res, 200, result);
        } catch (err) {
          sendJson(res, 422, { error: err.message || "invalid LabelMe input" });
        }
        return;
      }

      if (method === "POST" && pathname === "/export/render") {
        const body = await readJsonBody(req);
        if (!body || body.protocol_version !== PROTOCOL_VERSION) {
          sendJson(res, 400, { error: "unsupported protocol_version" });
          return;
        }
        try {
          const result = renderLabelMeExport(body);
          sendJson(res, 200, result);
        } catch (err) {
          sendJson(res, 422, { error: err.message || "invalid export input" });
        }
        return;
      }

      sendJson(res, 404, { error: "not_found" });
    } catch (err) {
      const status = err && Number.isInteger(err.statusCode) ? err.statusCode : 500;
      if (!res.headersSent) {
        sendJson(
          res,
          status,
          { error: err && err.message ? err.message : "internal_error" },
          { close: status === 413 },
        );
      } else {
        res.destroy();
      }
    }
  });
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const config = loadConfig();
  const server = createServer(config);
  server.listen(config.port, config.host, () => {
    console.log(
      JSON.stringify({
        event: "format_adapter_started",
        format_id: SETUP_DOCUMENT.format_id,
        host: config.host,
        port: server.address().port,
      }),
    );
  });
}
