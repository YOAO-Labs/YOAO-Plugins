export function loadConfig(env = process.env) {
  const token = (env.YOAO_FORMAT_ADAPTER_TOKEN || "").trim();
  if (!token) {
    throw new Error("YOAO_FORMAT_ADAPTER_TOKEN is required");
  }
  const host = (env.HOST || "127.0.0.1").trim() || "127.0.0.1";
  const rawPort = env.PORT === undefined ? "8092" : String(env.PORT).trim();
  const port = Number.parseInt(rawPort, 10);
  if (!Number.isInteger(port) || port < 0 || port > 65535) {
    throw new Error(`Invalid PORT: ${rawPort}`);
  }
  return Object.freeze({ host, port, token });
}
