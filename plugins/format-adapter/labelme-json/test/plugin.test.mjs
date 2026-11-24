import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { createServer } from "../src/server.mjs";
import { PROTOCOL_VERSION, SETUP_DOCUMENT } from "../src/converter.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PLUGIN_DIR = path.resolve(__dirname, "..");

async function startTestServer(token = "test-format-token") {
  const server = createServer({ host: "127.0.0.1", port: 0, token });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address();
  return {
    url: `http://127.0.0.1:${port}`,
    close: () => new Promise((resolve) => server.close(resolve)),
  };
}

test("manifest.json matches Dockerfile label and /setup metadata", () => {
  const manifest = JSON.parse(
    fs.readFileSync(path.join(PLUGIN_DIR, "manifest.json"), "utf8"),
  );
  const dockerfile = fs.readFileSync(path.join(PLUGIN_DIR, "Dockerfile"), "utf8");
  const match = dockerfile.match(/LABEL io\.yoao\.plugin\.manifest='([^']+)'/);
  assert.ok(match, "Dockerfile must embed io.yoao.plugin.manifest");
  assert.deepEqual(JSON.parse(match[1]), manifest);
  assert.equal(manifest.extension_point, "format_adapter");
  assert.equal(manifest.protocol_version, PROTOCOL_VERSION);
  assert.equal(manifest.extension.format_id, SETUP_DOCUMENT.format_id);
});

test("health, auth, setup, import/parse, and export/render round-trip", async () => {
  const token = "secret-labelme-token";
  const srv = await startTestServer(token);
  try {
    const health = await fetch(`${srv.url}/health`);
    assert.equal(health.status, 200);
    assert.deepEqual(await health.json(), { status: "ok" });

    const unauth = await fetch(`${srv.url}/setup`);
    assert.equal(unauth.status, 401);

    const setupRes = await fetch(`${srv.url}/setup`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    assert.equal(setupRes.status, 200);
    const setup = await setupRes.json();
    assert.equal(setup.format_id, "labelme-json");
    assert.deepEqual(setup.capabilities, ["import", "export"]);

    const labelme = {
      version: "5.0.0",
      imagePath: "street/001.jpg",
      imageWidth: 200,
      imageHeight: 100,
      shapes: [
        {
          label: "vehicle",
          shape_type: "rectangle",
          points: [[20, 10], [120, 60]],
        },
        {
          label: "road",
          shape_type: "polygon",
          points: [[0, 50], [200, 50], [200, 100], [0, 100]],
        },
      ],
    };
    const parseRes = await fetch(`${srv.url}/import/parse`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        protocol_version: PROTOCOL_VERSION,
        format_id: "labelme-json",
        media_type: "image",
        staged_sha256: "a".repeat(64),
        content_base64: Buffer.from(JSON.stringify(labelme), "utf8").toString("base64"),
        mapping: { labels: { vehicle: "car" } },
        options: {},
      }),
    });
    assert.equal(parseRes.status, 200);
    const parsed = await parseRes.json();
    assert.equal(parsed.items.length, 1);
    assert.equal(parsed.items[0].media_path, "street/001.jpg");
    assert.equal(parsed.items[0].annotations.length, 2);
    assert.equal(parsed.items[0].annotations[0].class_name, "car");
    assert.deepEqual(parsed.items[0].annotations[0].geometry, {
      type: "bbox",
      x: 0.1,
      y: 0.1,
      w: 0.5,
      h: 0.5,
    });
    assert.equal(parsed.items[0].annotations[1].tool_unit_id, "region");

    const exportRes = await fetch(`${srv.url}/export/render`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        protocol_version: PROTOCOL_VERSION,
        format_id: "labelme-json",
        media_type: "image",
        options: {},
        classes: ["car"],
        tasks: [
          {
            file_path: "street/001.jpg",
            width: 200,
            height: 100,
            annotations: parsed.items[0].annotations,
          },
        ],
      }),
    });
    assert.equal(exportRes.status, 200);
    const exported = await exportRes.json();
    assert.equal(exported.files.length, 1);
    assert.equal(exported.files[0].path, "001.json");
    const rendered = JSON.parse(
      Buffer.from(exported.files[0].content_base64, "base64").toString("utf8"),
    );
    assert.equal(rendered.shapes.length, 2);
    assert.equal(rendered.shapes[0].shape_type, "rectangle");

    const badDoc = {
      imagePath: "street/001.jpg",
      imageWidth: 200,
      imageHeight: 100,
      shapes: [
        { label: "car", shape_type: "rectangle", points: [["abc", 10], [120, 60]] },
      ],
    };
    const badRes = await fetch(`${srv.url}/import/parse`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        protocol_version: PROTOCOL_VERSION,
        format_id: "labelme-json",
        media_type: "image",
        staged_sha256: "b".repeat(64),
        content_base64: Buffer.from(JSON.stringify(badDoc), "utf8").toString("base64"),
        mapping: {},
        options: {},
      }),
    });
    assert.equal(badRes.status, 422);
  } finally {
    await srv.close();
  }
});
