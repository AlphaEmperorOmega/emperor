import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

export function virtualenvPython(repositoryRoot, platform = process.platform) {
  return join(
    repositoryRoot,
    "torchenv",
    platform === "win32" ? "Scripts" : "bin",
    platform === "win32" ? "python.exe" : "python",
  );
}

export function pythonCommand(platform = process.platform) {
  return platform === "win32" ? "python.exe" : "python";
}

export function resolveRepositoryPython(
  repositoryRoot,
  { environment = process.env, platform = process.platform } = {},
) {
  if (environment.WORKBENCH_E2E_PYTHON) {
    return environment.WORKBENCH_E2E_PYTHON;
  }
  const candidate = virtualenvPython(repositoryRoot, platform);
  return existsSync(candidate) ? candidate : pythonCommand(platform);
}

export function nextCli(frontendRoot) {
  return join(frontendRoot, "node_modules", "next", "dist", "bin", "next");
}

export function matplotlibConfigDirectory() {
  return join(tmpdir(), "emperor-matplotlib");
}
