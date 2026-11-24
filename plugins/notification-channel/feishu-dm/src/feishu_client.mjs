const PERMANENT_FEISHU_CODES = new Set([
  230001, 230002, 230013, 230072, 99992351, 99992361,
]);

const TOKEN_EXPIRED_FEISHU_CODES = new Set([
  99991661, 99991663, 99991664, 99991668,
]);

const RATE_LIMIT_FEISHU_CODES = new Set([99991400, 11232, 230020]);

const APP_PERMISSION_FEISHU_CODES = new Set([230006, 230027, 99991672]);

function clampRetryAfter(raw, fallback = 30) {
  const n = Number.parseInt(String(raw ?? ""), 10);
  if (Number.isFinite(n) && n > 0) {
    return Math.max(5, Math.min(3600, n));
  }
  return fallback;
}

function readRateLimitDelay(headers, fallback = 30) {
  const raw =
    headers?.get?.("retry-after") ?? headers?.get?.("x-ogw-ratelimit-reset");
  return clampRetryAfter(raw, fallback);
}

async function fetchJsonWithTimeout(fetchImpl, url, options, timeoutMs) {
  if (timeoutMs <= 0) {
    const err = new Error("feishu_request_timeout");
    err.name = "AbortError";
    throw err;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchImpl(url, {
      ...options,
      signal: controller.signal,
      redirect: "error",
    });
    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch {
      json = null;
    }
    return {
      status: response.status,
      headers: response.headers,
      json,
    };
  } finally {
    clearTimeout(timer);
  }
}

export class FeishuClient {
  constructor({
    appId,
    appSecret,
    apiBaseUrl = "https://open.feishu.cn",
    httpTimeoutMs = 8000,
    fetchImpl = globalThis.fetch,
    now = () => Date.now(),
  }) {
    this._appId = appId;
    this._appSecret = appSecret;
    this._apiBaseUrl = apiBaseUrl.replace(/\/+$/, "");
    this._httpTimeoutMs = httpTimeoutMs;
    this._fetch = fetchImpl;
    this._now = now;
    this._cachedToken = null;
    this._expiresAtMs = 0;
    this._inflightToken = null;
  }

  invalidateToken(expectedToken) {
    if (expectedToken === undefined || this._cachedToken === expectedToken) {
      this._cachedToken = null;
      this._expiresAtMs = 0;
    }
  }

  async getTenantAccessToken({ failedToken = null, deadlineMs = null } = {}) {
    if (failedToken !== null && this._cachedToken === failedToken) {
      this._cachedToken = null;
      this._expiresAtMs = 0;
    }
    if (this._cachedToken && this._now() < this._expiresAtMs) {
      return this._cachedToken;
    }
    if (this._inflightToken) {
      return this._inflightToken;
    }
    const timeoutMs =
      deadlineMs === null
        ? this._httpTimeoutMs
        : Math.max(0, deadlineMs - this._now());
    const promise = this._fetchTenantAccessToken(timeoutMs);
    this._inflightToken = promise;
    try {
      return await promise;
    } finally {
      if (this._inflightToken === promise) {
        this._inflightToken = null;
      }
    }
  }

  async _fetchTenantAccessToken(timeoutMs) {
    const url = `${this._apiBaseUrl}/open-apis/auth/v3/tenant_access_token/internal`;
    const res = await fetchJsonWithTimeout(
      this._fetch,
      url,
      {
        method: "POST",
        headers: { "Content-Type": "application/json; charset=utf-8" },
        body: JSON.stringify({
          app_id: this._appId,
          app_secret: this._appSecret,
        }),
      },
      timeoutMs,
    );

    if (res.status === 429) {
      const err = new Error("feishu_rate_limited");
      err.outcome = {
        status: "retryable",
        reason: "feishu_rate_limited",
        retry_after_seconds: readRateLimitDelay(res.headers, 30),
      };
      throw err;
    }
    if (res.status >= 500 || !res.json || typeof res.json !== "object") {
      const err = new Error(`feishu_token_http_${res.status}`);
      err.outcome = {
        status: "retryable",
        reason: `feishu_token_http_${res.status}`,
      };
      throw err;
    }
    if (
      res.json.code !== 0 ||
      typeof res.json.tenant_access_token !== "string"
    ) {
      const code = res.json.code ?? res.status;
      const err = new Error(`feishu_app_auth_${code}`);
      err.outcome = {
        status: "retryable",
        reason: `feishu_app_auth_${code}`,
        retry_after_seconds: 60,
      };
      throw err;
    }

    const expireSec =
      typeof res.json.expire === "number" && res.json.expire > 0
        ? res.json.expire
        : 7200;
    const ttlSec =
      expireSec > 300
        ? expireSec - 300
        : Math.max(1, Math.floor(expireSec / 2));
    this._cachedToken = res.json.tenant_access_token;
    this._expiresAtMs = this._now() + ttlSec * 1000;
    return this._cachedToken;
  }

  async sendInteractiveCard({ openId, card, uuid }) {
    const deadlineMs = this._now() + this._httpTimeoutMs;
    try {
      const token = await this.getTenantAccessToken({ deadlineMs });
      let result = await this._postMessage({
        token,
        openId,
        card,
        uuid,
        timeoutMs: deadlineMs - this._now(),
      });
      if (result._retryWithFreshToken) {
        const refreshed = await this.getTenantAccessToken({
          failedToken: token,
          deadlineMs,
        });
        result = await this._postMessage({
          token: refreshed,
          openId,
          card,
          uuid,
          timeoutMs: deadlineMs - this._now(),
        });
      }
      if (result._retryWithFreshToken) {
        return {
          status: "retryable",
          reason: "feishu_token_rejected",
          retry_after_seconds: 60,
        };
      }
      return result;
    } catch (err) {
      if (err && err.outcome) {
        return err.outcome;
      }
      if (err && err.name === "AbortError") {
        return { status: "retryable", reason: "feishu_request_timeout" };
      }
      return { status: "retryable", reason: "feishu_network_error" };
    }
  }

  async _postMessage({ token, openId, card, uuid, timeoutMs }) {
    const url = `${this._apiBaseUrl}/open-apis/im/v1/messages?receive_id_type=open_id`;
    const body = {
      receive_id: openId,
      msg_type: "interactive",
      content: JSON.stringify(card),
      uuid: String(uuid).slice(0, 50),
    };
    const res = await fetchJsonWithTimeout(
      this._fetch,
      url,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json; charset=utf-8",
        },
        body: JSON.stringify(body),
      },
      timeoutMs,
    );

    const code =
      res.json && typeof res.json === "object" ? res.json.code : undefined;

    if (
      res.status === 429 ||
      (typeof code === "number" && RATE_LIMIT_FEISHU_CODES.has(code))
    ) {
      return {
        status: "retryable",
        reason: "feishu_rate_limited",
        retry_after_seconds: readRateLimitDelay(res.headers, 30),
      };
    }

    if (typeof code === "number" && TOKEN_EXPIRED_FEISHU_CODES.has(code)) {
      return { _retryWithFreshToken: true };
    }

    if (typeof code === "number" && APP_PERMISSION_FEISHU_CODES.has(code)) {
      return {
        status: "retryable",
        reason: `feishu_app_permission_${code}`,
        retry_after_seconds: 60,
      };
    }

    if (res.status === 200 && code === 0) {
      const msgId = res.json?.data?.message_id;
      return {
        status: "delivered",
        ...(typeof msgId === "string" && msgId
          ? { provider_message_id: msgId.slice(0, 256) }
          : {}),
      };
    }

    if (typeof code === "number" && PERMANENT_FEISHU_CODES.has(code)) {
      return {
        status: "permanent",
        reason: `feishu_code_${code}`,
      };
    }

    return {
      status: "retryable",
      reason:
        typeof code === "number"
          ? `feishu_code_${code}`
          : `feishu_http_${res.status}`,
    };
  }
}
