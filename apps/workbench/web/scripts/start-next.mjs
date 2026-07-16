import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { nextCli } from "./runtime-paths.mjs";

const mode = process.argv[2];
if (mode !== "dev" && mode !== "start") {
  throw new Error("Usage: node scripts/start-next.mjs <dev|start> [Next options]");
}

const rawPort = process.env.PORT ?? "9000";
const port = Number(rawPort);
if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error("PORT must be an integer between 1 and 65535.");
}

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const frontendRoot = resolve(scriptDirectory, "..");
const child = spawn(
  process.execPath,
  [nextCli(frontendRoot), mode, "-p", String(port), ...process.argv.slice(3)],
  {
    cwd: frontendRoot,
    env: process.env,
    stdio: "inherit",
    windowsHide: false,
  },
);

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => child.kill(signal));
}
child.once("error", (error) => {
  console.error(error);
  process.exitCode = 1;
});
child.once("exit", (code, signal) => {
  process.exitCode = code ?? (signal ? 1 : 0);
});
