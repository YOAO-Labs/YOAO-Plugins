import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  buildNotificationCard,
  buildVerifyAddressCard,
  escapeLarkMd,
} from "../src/cards.mjs";
import { loadConfig } from "../src/config.mjs";
import {
  createPluginServer,
  CHANNEL_ID,
  PROTOCOL_VERSION,
} from "../src/server.mjs";

const PLUGIN_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "..",
);

function listen(server) {
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      resolve(addr.port);
    });
  });
}

function close(server) {
  return new Promise((resolve) => server.close(() => resolve()));
}

function createFakeFeishuServer() {
  const state = {
    tokenCalls: 0,
    messageCalls: [],
    tokenResponse: null,
    messageHandler: null,
    messageResponses: [],
    defaultMessageResponse: {
      status: 200,
      body: { code: 0, data: { message_id: "om_fake_001" } },
    },
  };

  const server = http.createServer(async (req, res) => {
    const chunks = [];
    for await (const c of req) {
      chunks.push(c);
    }
    const raw = Buffer.concat(chunks).toString("utf8");
    const body = raw ? JSON.parse(raw) : {};
    const url = new URL(req.url, "http://127.0.0.1");

    if (
      req.method === "POST" &&
      url.pathname === "/open-apis/auth/v3/tenant_access_token/internal"
    ) {
      state.tokenCalls += 1;
      const spec = state.tokenResponse ?? {
        status: 200,
        body: {
          code: 0,
          tenant_access_token: `t-fake-token-${state.tokenCalls}`,
          expire: 7200,
        },
      };
      const payload = JSON.stringify(spec.body);
      res.writeHead(spec.status, {
        "Content-Type": "application/json",
        ...(spec.headers ?? {}),
      });
      res.end(payload);
      return;
    }

    if (req.method === "POST" && url.pathname === "/open-apis/im/v1/messages") {
      const call = {
        receiveIdType: url.searchParams.get("receive_id_type"),
        authorization: req.headers.authorization,
        body,
      };
      state.messageCalls.push(call);
      const spec = state.messageHandler
        ? state.messageHandler(call)
        : state.messageResponses.length > 0
          ? state.messageResponses.shift()
          : state.defaultMessageResponse;
      const payload = JSON.stringify(spec.body);
      res.writeHead(spec.status, {
        "Content-Type": "application/json",
        ...(spec.headers ?? {}),
      });
      res.end(payload);
      return;
    }

    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ code: 404, msg: "not found" }));
  });

  return { server, state };
}

test("loadConfig enforces required secrets, file secrets, and rejects conflicts", () => {
  assert.throws(() => loadConfig({}), /YOAO_CHANNEL_AUTH_TOKEN/);

  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "yoao-feishu-cfg-"));
  try {
    const tokFile = path.join(tmpDir, "tok");
    const appIdFile = path.join(tmpDir, "app_id");
    const secretFile = path.join(tmpDir, "app_secret");
    fs.writeFileSync(tokFile, "secret-token-from-file\n", "utf8");
    fs.writeFileSync(appIdFile, "cli_file_app", "utf8");
    fs.writeFileSync(secretFile, "secret_file_val", "utf8");

    const cfg = loadConfig({
      YOAO_CHANNEL_AUTH_TOKEN_FILE: tokFile,
      FEISHU_APP_ID_FILE: appIdFile,
      FEISHU_APP_SECRET_FILE: secretFile,
      YOAO_PUBLIC_BASE_URL: "https://yoao.example.com/",
      FEISHU_HTTP_TIMEOUT_MS: "3500",
    });
    assert.equal(cfg.channelAuthToken, "secret-token-from-file");
    assert.equal(cfg.feishuAppId, "cli_file_app");
    assert.equal(cfg.feishuAppSecret, "secret_file_val");
    assert.equal(cfg.publicBaseUrl, "https://yoao.example.com");
    assert.equal(cfg.httpTimeoutMs, 3500);

    assert.throws(
      () =>
        loadConfig({
          YOAO_CHANNEL_AUTH_TOKEN: "inline",
          YOAO_CHANNEL_AUTH_TOKEN_FILE: tokFile,
          FEISHU_APP_ID: "cli_x",
          FEISHU_APP_SECRET: "sec_x",
        }),
      /Configure only one/,
    );
    assert.throws(
      () =>
        loadConfig({
          YOAO_CHANNEL_AUTH_TOKEN: "tok",
          FEISHU_APP_ID: "cli_x",
          FEISHU_APP_SECRET: "sec_x",
          FEISHU_API_BASE_URL: "ftp://open.feishu.cn",
        }),
      /http:\/\/ or https:\/\//,
    );
    assert.throws(
      () =>
        loadConfig({
          YOAO_CHANNEL_AUTH_TOKEN: "tok",
          FEISHU_APP_ID: "cli_x",
          FEISHU_APP_SECRET: "sec_x",
          YOAO_PUBLIC_BASE_URL: "https://yoao.example.com?",
        }),
      /query, or fragment/,
    );
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test("card builders escape markdown, respect locale, and guard action_url", () => {
  assert.equal(escapeLarkMd("[A]*_B"), "\\[A\\]\\*\\_B");

  const zhCard = buildNotificationCard({
    locale: "zh-CN",
    publicBaseUrl: "https://yoao.example.com",
    envelope: {
      event_version: "1.0",
      event: "task.rejected",
      delivery_id: "11111111-1111-4111-8111-111111111111",
      occurred_at: "2026-11-24T09:00:00Z",
      data: {
        target_type: "task",
        target_id: "22222222-2222-4222-8222-222222222222",
        task_display_id: "T-123",
        project_name: "Road*Det",
        reject_reason_type: "missing_label",
        actor_name: "Reviewer",
        action_url: "/projects/p1/workbench/t1",
      },
    },
  });
  assert.equal(zhCard.header.title.content, "任务被驳回");
  assert.equal(zhCard.header.template, "red");
  assert.match(zhCard.elements[0].text.content, /T\\-123/);
  assert.match(zhCard.elements[0].text.content, /Road\\\*Det/);
  assert.equal(zhCard.elements[1].tag, "action");
  assert.equal(
    zhCard.elements[1].actions[0].url,
    "https://yoao.example.com/projects/p1/workbench/t1",
  );
  assert.equal(zhCard.elements[1].actions[0].text.content, "打开工作台");

  const enCardNoButton = buildNotificationCard({
    locale: "en-US",
    publicBaseUrl: "https://yoao.example.com",
    envelope: {
      event_version: "1.0",
      event: "export.ready",
      delivery_id: "33333333-3333-4333-8333-333333333333",
      occurred_at: "2026-11-24T09:00:00Z",
      data: {
        target_type: "export",
        target_id: "44444444-4444-4444-8444-444444444444",
        project_display_id: "P-9",
        action_url: "//evil.example.com/phish",
      },
    },
  });
  assert.equal(enCardNoButton.header.title.content, "Export ready");
  assert.equal(enCardNoButton.elements.length, 2);

  const protoCard = buildNotificationCard({
    locale: "zh-CN",
    publicBaseUrl: "https://yoao.example.com",
    envelope: {
      event_version: "1.0",
      event: "constructor",
      delivery_id: "55555555-5555-4555-8555-555555555554",
      occurred_at: "2026-11-24T09:00:00Z",
      data: {
        target_type: "task",
        target_id: "44444444-4444-4444-8444-444444444445",
        action_url: "/projects/../admin",
      },
    },
  });
  assert.equal(protoCard.header.title.content, "YOAO 通知：constructor");
  assert.equal(protoCard.header.template, "blue");
  assert.equal(protoCard.elements.length, 2);

  const verifyCard = buildVerifyAddressCard({
    locale: "en-US",
    verificationId: "55555555-5555-4555-8555-555555555555",
  });
  assert.equal(
    verifyCard.header.title.content,
    "YOAO Feishu Notification Verification",
  );
  assert.match(
    verifyCard.elements[1].elements[0].content,
    /55555555-5555-4555-8555-555555555555/,
  );
});

test("plugin HTTP endpoints enforce auth, cache tokens, deduplicate delivery_id, refresh expired tokens, and classify Feishu outcomes", async () => {
  const fakeFeishu = createFakeFeishuServer();
  const fakePort = await listen(fakeFeishu.server);
  const config = loadConfig({
    YOAO_CHANNEL_HOST: "127.0.0.1",
    YOAO_CHANNEL_PORT: "0",
    YOAO_CHANNEL_AUTH_TOKEN: "channel-secret-token",
    FEISHU_APP_ID: "cli_test_app",
    FEISHU_APP_SECRET: "sec_test_app",
    FEISHU_API_BASE_URL: `http://127.0.0.1:${fakePort}`,
    YOAO_PUBLIC_BASE_URL: "https://yoao.example.com",
  });
  const pluginServer = createPluginServer({ config });
  const pluginPort = await listen(pluginServer);
  const baseUrl = `http://127.0.0.1:${pluginPort}`;
  const authHeader = { Authorization: "Bearer channel-secret-token" };

  try {
    const healthRes = await fetch(`${baseUrl}/health`);
    assert.equal(healthRes.status, 200);
    assert.deepEqual(await healthRes.json(), { status: "ok" });

    const unauthSetup = await fetch(`${baseUrl}/setup`);
    assert.equal(unauthSetup.status, 401);

    const badTokenSetup = await fetch(`${baseUrl}/setup`, {
      headers: { Authorization: "Bearer wrong-token" },
    });
    assert.equal(badTokenSetup.status, 401);

    const setupRes = await fetch(`${baseUrl}/setup`, { headers: authHeader });
    assert.equal(setupRes.status, 200);
    const setupBody = await setupRes.json();
    assert.equal(setupBody.protocol_version, PROTOCOL_VERSION);
    assert.equal(setupBody.channel_id, CHANNEL_ID);
    assert.deepEqual(setupBody.capabilities, ["deliver", "verify_address"]);
    assert.equal(setupBody.address_schema.additionalProperties, false);

    const deliverPayload = (deliveryId) => ({
      protocol_version: PROTOCOL_VERSION,
      channel_id: CHANNEL_ID,
      locale: "zh-CN",
      recipient_address: { open_id: "ou_test_user_001" },
      envelope: {
        event_version: "1.0",
        event: "task.rejected",
        delivery_id: deliveryId,
        occurred_at: "2026-11-24T09:10:00Z",
        data: {
          target_type: "task",
          target_id: "aaaa0000-0000-4000-8000-000000000001",
          task_display_id: "T-007",
          project_name: "Demo",
          action_url: "/projects/p1/workbench/t7",
        },
      },
    });

    const d1 = "11111111-1111-4111-8111-111111111111";
    const r1 = await fetch(`${baseUrl}/deliver`, {
      method: "POST",
      headers: { ...authHeader, "Content-Type": "application/json" },
      body: JSON.stringify(deliverPayload(d1)),
    });
    assert.equal(r1.status, 200);
    assert.deepEqual(await r1.json(), {
      status: "delivered",
      provider_message_id: "om_fake_001",
    });
    assert.equal(fakeFeishu.state.tokenCalls, 1);
    assert.equal(fakeFeishu.state.messageCalls.length, 1);
    assert.equal(fakeFeishu.state.messageCalls[0].receiveIdType, "open_id");
    assert.equal(fakeFeishu.state.messageCalls[0].body.uuid, d1);
    assert.equal(
      fakeFeishu.state.messageCalls[0].body.receive_id,
      "ou_test_user_001",
    );

    const r1Dup = await fetch(`${baseUrl}/deliver`, {
      method: "POST",
      headers: { ...authHeader, "Content-Type": "application/json" },
      body: JSON.stringify(deliverPayload(d1)),
    });
    assert.equal(r1Dup.status, 200);
    assert.deepEqual(await r1Dup.json(), {
      status: "delivered",
      provider_message_id: "om_fake_001",
    });
    assert.equal(fakeFeishu.state.messageCalls.length, 1);

    const d2 = "22222222-2222-4222-8222-222222222222";
    const r2 = await fetch(`${baseUrl}/deliver`, {
      method: "POST",
      headers: { ...authHeader, "Content-Type": "application/json" },
      body: JSON.stringify(deliverPayload(d2)),
    });
    assert.equal(r2.status, 200);
    assert.equal(fakeFeishu.state.tokenCalls, 1);
    assert.equal(fakeFeishu.state.messageCalls.length, 2);

    fakeFeishu.state.messageResponses.push(
      { status: 400, body: { code: 99991663, msg: "token expired" } },
      { status: 200, body: { code: 0, data: { message_id: "om_refreshed" } } },
    );
    const d3 = "33333333-3333-4333-8333-333333333333";
    const r3 = await fetch(`${baseUrl}/deliver`, {
      method: "POST",
      headers: { ...authHeader, "Content-Type": "application/json" },
      body: JSON.stringify(deliverPayload(d3)),
    });
    assert.equal(r3.status, 200);
    assert.deepEqual(await r3.json(), {
      status: "delivered",
      provider_message_id: "om_refreshed",
    });
    assert.equal(fakeFeishu.state.tokenCalls, 2);
    assert.equal(
      fakeFeishu.state.messageCalls[fakeFeishu.state.messageCalls.length - 1]
        .authorization,
      "Bearer t-fake-token-2",
    );

    fakeFeishu.state.messageResponses.push({
      status: 400,
      body: { code: 230013, msg: "user out of app availability" },
    });
    const d4 = "44444444-4444-4444-8444-444444444444";
    const r4 = await fetch(`${baseUrl}/deliver`, {
      method: "POST",
      headers: { ...authHeader, "Content-Type": "application/json" },
      body: JSON.stringify(deliverPayload(d4)),
    });
    assert.equal(r4.status, 422);
    assert.deepEqual(await r4.json(), {
      status: "permanent",
      reason: "feishu_code_230013",
    });

    fakeFeishu.state.messageResponses.push({
      status: 429,
      headers: { "Retry-After": "45" },
      body: { code: 99991400, msg: "rate limit" },
    });
    const d5 = "55555555-5555-4555-8555-555555555555";
    const r5 = await fetch(`${baseUrl}/deliver`, {
      method: "POST",
      headers: { ...authHeader, "Content-Type": "application/json" },
      body: JSON.stringify(deliverPayload(d5)),
    });
    assert.equal(r5.status, 503);
    assert.equal(r5.headers.get("retry-after"), "45");
    assert.deepEqual(await r5.json(), {
      status: "retryable",
      reason: "feishu_rate_limited",
      retry_after_seconds: 45,
    });

    fakeFeishu.state.messageResponses.push({
      status: 403,
      body: { code: 99991672, msg: "missing im:message:send_as_bot" },
    });
    const d6 = "66666666-6666-4666-8666-666666666666";
    const r6 = await fetch(`${baseUrl}/deliver`, {
      method: "POST",
      headers: { ...authHeader, "Content-Type": "application/json" },
      body: JSON.stringify(deliverPayload(d6)),
    });
    assert.equal(r6.status, 503);
    assert.deepEqual(await r6.json(), {
      status: "retryable",
      reason: "feishu_app_permission_99991672",
      retry_after_seconds: 60,
    });

    const badAddr = await fetch(`${baseUrl}/deliver`, {
      method: "POST",
      headers: { ...authHeader, "Content-Type": "application/json" },
      body: JSON.stringify({
        ...deliverPayload("77777777-7777-4777-8777-777777777777"),
        recipient_address: { open_id: "ou_ok", extra: "bad" },
      }),
    });
    assert.equal(badAddr.status, 422);

    const oversized = await fetch(`${baseUrl}/deliver`, {
      method: "POST",
      headers: { ...authHeader, "Content-Type": "application/json" },
      body: "x".repeat(40 * 1024),
    });
    assert.equal(oversized.status, 413);
    assert.equal(oversized.headers.get("connection"), "close");

    for (const [code, expectedReason, expectedStatus] of [
      [230006, "feishu_app_permission_230006", 503],
      [230027, "feishu_app_permission_230027", 503],
      [11232, "feishu_rate_limited", 503],
      [230020, "feishu_rate_limited", 503],
      [999999, "feishu_code_999999", 503],
    ]) {
      fakeFeishu.state.messageResponses.push({
        status: 400,
        headers: code === 11232 ? { "x-ogw-ratelimit-reset": "25" } : {},
        body: { code, msg: "simulated" },
      });
      const res = await fetch(`${baseUrl}/deliver`, {
        method: "POST",
        headers: { ...authHeader, "Content-Type": "application/json" },
        body: JSON.stringify(
          deliverPayload(
            `00000000-0000-4000-8000-${String(code).padStart(12, "0")}`,
          ),
        ),
      });
      assert.equal(res.status, expectedStatus);
      const body = await res.json();
      assert.equal(body.status, "retryable");
      assert.equal(body.reason, expectedReason);
      if (code === 11232) {
        assert.equal(body.retry_after_seconds, 25);
      }
    }

    fakeFeishu.state.messageResponses.push({
      status: 200,
      body: { code: 0, data: { message_id: "om_verify_card_1" } },
    });
    const verifyRes = await fetch(`${baseUrl}/verify-address`, {
      method: "POST",
      headers: { ...authHeader, "Content-Type": "application/json" },
      body: JSON.stringify({
        protocol_version: PROTOCOL_VERSION,
        channel_id: CHANNEL_ID,
        verification_id: "88888888-8888-4888-8888-888888888888",
        recipient_address: { open_id: "ou_verify_user" },
        locale: "zh-CN",
      }),
    });
    assert.equal(verifyRes.status, 200);
    assert.deepEqual(await verifyRes.json(), {
      status: "delivered",
      provider_message_id: "om_verify_card_1",
    });
  } finally {
    await close(pluginServer);
    await close(fakeFeishu.server);
  }
});

test("app auth failure on tenant_access_token endpoint maps to retryable, never permanent", async () => {
  const fakeFeishu = createFakeFeishuServer();
  fakeFeishu.state.tokenResponse = {
    status: 400,
    body: { code: 10014, msg: "invalid app_secret" },
  };
  const fakePort = await listen(fakeFeishu.server);
  const config = loadConfig({
    YOAO_CHANNEL_HOST: "127.0.0.1",
    YOAO_CHANNEL_PORT: "0",
    YOAO_CHANNEL_AUTH_TOKEN: "tok",
    FEISHU_APP_ID: "cli_bad",
    FEISHU_APP_SECRET: "sec_bad",
    FEISHU_API_BASE_URL: `http://127.0.0.1:${fakePort}`,
  });
  const pluginServer = createPluginServer({ config });
  const pluginPort = await listen(pluginServer);
  try {
    const res = await fetch(`http://127.0.0.1:${pluginPort}/deliver`, {
      method: "POST",
      headers: {
        Authorization: "Bearer tok",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        protocol_version: PROTOCOL_VERSION,
        channel_id: CHANNEL_ID,
        locale: "zh-CN",
        recipient_address: { open_id: "ou_test_user" },
        envelope: {
          event_version: "1.0",
          event: "task.approved",
          delivery_id: "99999999-9999-4999-8999-999999999999",
          occurred_at: "2026-11-24T09:15:00Z",
          data: {
            target_type: "task",
            target_id: "bbbb0000-0000-4000-8000-000000000001",
            task_display_id: "T-009",
          },
        },
      }),
    });
    assert.equal(res.status, 503);
    assert.deepEqual(await res.json(), {
      status: "retryable",
      reason: "feishu_app_auth_10014",
      retry_after_seconds: 60,
    });
  } finally {
    await close(pluginServer);
    await close(fakeFeishu.server);
  }
});

test("concurrent expired-token failures coalesce into one refresh and handler traps unexpected throws", async () => {
  const fakeFeishu = createFakeFeishuServer();
  const fakePort = await listen(fakeFeishu.server);
  const config = loadConfig({
    YOAO_CHANNEL_HOST: "127.0.0.1",
    YOAO_CHANNEL_PORT: "0",
    YOAO_CHANNEL_AUTH_TOKEN: "tok",
    FEISHU_APP_ID: "cli_coal",
    FEISHU_APP_SECRET: "sec_coal",
    FEISHU_API_BASE_URL: `http://127.0.0.1:${fakePort}`,
  });
  const pluginServer = createPluginServer({ config });
  const pluginPort = await listen(pluginServer);

  const crashingServer = createPluginServer({
    config,
    feishuClient: {
      sendInteractiveCard() {
        return Promise.reject(new Error("boom"));
      },
    },
  });
  const crashingPort = await listen(crashingServer);

  const deliver = (port, deliveryId) =>
    fetch(`http://127.0.0.1:${port}/deliver`, {
      method: "POST",
      headers: {
        Authorization: "Bearer tok",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        protocol_version: PROTOCOL_VERSION,
        channel_id: CHANNEL_ID,
        locale: "zh-CN",
        recipient_address: { open_id: "ou_coal" },
        envelope: {
          event_version: "1.0",
          event: "task.approved",
          delivery_id: deliveryId,
          occurred_at: "2026-11-24T09:20:00Z",
          data: {
            target_type: "task",
            target_id: "cccc0000-0000-4000-8000-000000000001",
          },
        },
      }),
    });

  try {
    const warm = await deliver(
      pluginPort,
      "aaaaaaaa-0000-4000-8000-000000000001",
    );
    assert.equal(warm.status, 200);
    assert.equal(fakeFeishu.state.tokenCalls, 1);

    fakeFeishu.state.messageHandler = (call) => {
      if (call.authorization === "Bearer t-fake-token-1") {
        return { status: 400, body: { code: 99991663, msg: "expired" } };
      }
      return {
        status: 200,
        body: { code: 0, data: { message_id: `om_${call.body.uuid}` } },
      };
    };
    const results = await Promise.all([
      deliver(pluginPort, "aaaaaaaa-0000-4000-8000-000000000002"),
      deliver(pluginPort, "aaaaaaaa-0000-4000-8000-000000000003"),
      deliver(pluginPort, "aaaaaaaa-0000-4000-8000-000000000004"),
    ]);
    for (const r of results) {
      assert.equal(r.status, 200);
    }
    assert.equal(fakeFeishu.state.tokenCalls, 2);

    const crashed = await deliver(
      crashingPort,
      "aaaaaaaa-0000-4000-8000-000000000005",
    );
    assert.equal(crashed.status, 500);
    assert.equal(crashed.headers.get("connection"), "close");
    assert.deepEqual(await crashed.json(), {
      status: "retryable",
      reason: "internal_error",
    });
  } finally {
    await close(pluginServer);
    await close(crashingServer);
    await close(fakeFeishu.server);
  }
});

test("manifest.json and Dockerfile io.yoao.plugin.manifest label stay identical", () => {
  const manifestRaw = JSON.parse(
    fs.readFileSync(path.join(PLUGIN_ROOT, "manifest.json"), "utf8"),
  );
  const dockerfile = fs.readFileSync(
    path.join(PLUGIN_ROOT, "Dockerfile"),
    "utf8",
  );
  const match = dockerfile.match(
    /^LABEL\s+io\.yoao\.plugin\.manifest='(.+)'$/m,
  );
  assert.ok(
    match,
    "Dockerfile must define LABEL io.yoao.plugin.manifest='...'",
  );
  const labelRaw = JSON.parse(match[1]);
  assert.deepEqual(labelRaw, manifestRaw);
  assert.equal(manifestRaw.manifest_schema, "yoao-extension-manifest/v1");
  assert.equal(manifestRaw.extension_point, "notification_channel");
  assert.equal(manifestRaw.protocol_version, PROTOCOL_VERSION);
  assert.equal(manifestRaw.extension.channel_id, CHANNEL_ID);
  assert.equal(
    manifestRaw.extension.address_schema.additionalProperties,
    false,
  );
});
