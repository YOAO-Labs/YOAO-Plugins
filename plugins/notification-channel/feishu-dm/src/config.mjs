import fs from "node:fs";

function readSecret(env, valueKey, fileKey) {
  const direct = (env[valueKey] ?? "").trim();
  const filePath = (env[fileKey] ?? "").trim();
  if (direct && filePath) {
    throw new Error(`Configure only one of ${valueKey} or ${fileKey}`);
  }
  if (filePath) {
    const content = fs.readFileSync(filePath, "utf8").trim();
    if (!content) {
      throw new Error(`${fileKey} is empty`);
    }
    return content;
  }
  if (!direct) {
    throw new Error(`${valueKey} or ${fileKey} is required`);
  }
  return direct;
}

function normalizeBaseUrl(raw, name, { allowEmpty = false } = {}) {
  const value = (raw ?? "").trim();
  if (!value) {
    if (allowEmpty) {
      return "";
    }
    throw new Error(`${name} is required`);
  }
  if (
    /[\x00-\x1f\x7f\s]/.test(value) ||
    value.includes("?") ||
    value.includes("#")
  ) {
    throw new Error(
      `${name} must not contain whitespace, control characters, query, or fragment`,
    );
  }
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${name} must be a valid URL`);
  }
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
    throw new Error(`${name} must use http:// or https://`);
  }
  if (parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error(`${name} must not contain credentials, query, or fragment`);
  }
  const cleanPath = parsed.pathname.replace(/\/+$/, "");
  return `${parsed.origin}${cleanPath}`;
}

function parseIntInRange(raw, fallback, min, max, name) {
  if (raw === undefined || raw === null || String(raw).trim() === "") {
    return fallback;
  }
  const n = Number(raw);
  if (!Number.isInteger(n) || n < min || n > max) {
    throw new Error(`${name} must be an integer between ${min} and ${max}`);
  }
  return n;
}

export function loadConfig(env = process.env) {
  const host = (env.YOAO_CHANNEL_HOST ?? "127.0.0.1").trim() || "127.0.0.1";
  const port = parseIntInRange(
    env.YOAO_CHANNEL_PORT,
    8090,
    0,
    65535,
    "YOAO_CHANNEL_PORT",
  );
  const channelAuthToken = readSecret(
    env,
    "YOAO_CHANNEL_AUTH_TOKEN",
    "YOAO_CHANNEL_AUTH_TOKEN_FILE",
  );
  const feishuAppId = readSecret(env, "FEISHU_APP_ID", "FEISHU_APP_ID_FILE");
  const feishuAppSecret = readSecret(
    env,
    "FEISHU_APP_SECRET",
    "FEISHU_APP_SECRET_FILE",
  );
  const feishuApiBaseUrl = normalizeBaseUrl(
    env.FEISHU_API_BASE_URL ?? "https://open.feishu.cn",
    "FEISHU_API_BASE_URL",
  );
  const publicBaseUrl = normalizeBaseUrl(
    env.YOAO_PUBLIC_BASE_URL ?? "",
    "YOAO_PUBLIC_BASE_URL",
    {
      allowEmpty: true,
    },
  );
  const httpTimeoutMs = parseIntInRange(
    env.FEISHU_HTTP_TIMEOUT_MS,
    8000,
    500,
    9000,
    "FEISHU_HTTP_TIMEOUT_MS",
  );
  const dedupTtlMs = parseIntInRange(
    env.FEISHU_DEDUP_TTL_MS,
    86_400_000,
    1000,
    604_800_000,
    "FEISHU_DEDUP_TTL_MS",
  );
  const dedupMaxEntries = parseIntInRange(
    env.FEISHU_DEDUP_MAX_ENTRIES,
    10_000,
    10,
    100_000,
    "FEISHU_DEDUP_MAX_ENTRIES",
  );

  return Object.freeze({
    host,
    port,
    channelAuthToken,
    feishuAppId,
    feishuAppSecret,
    feishuApiBaseUrl,
    publicBaseUrl,
    httpTimeoutMs,
    dedupTtlMs,
    dedupMaxEntries,
  });
}
