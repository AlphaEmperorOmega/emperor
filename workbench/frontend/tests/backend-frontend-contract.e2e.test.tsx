import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { QueryClient } from "@tanstack/react-query";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  logout,
  resetApiBaseUrl,
  signIn,
  useApiBaseUrl as applyApiBaseUrl,
  type WorkbenchConnectionActionEnvironment,
} from "@/features/workbench/providers/_workbench-connection-actions";
import {
  createMutationRequestOptions,
  isUnauthorizedApiError,
} from "@/lib/api/client";
import {
  observeWorkbenchAuthMode,
  workbenchConnectionRuntimeSnapshot,
} from "@/lib/api/_connection-runtime";
import {
  createConfigSnapshot,
  fetchConfigSnapshotLibrary,
} from "@/lib/api/config-snapshots";
import { fetchCapabilities, fetchHealth } from "@/lib/api/health";
import { fetchModels } from "@/lib/api/models";
import {
  createTrainingJob,
  fetchTrainingJob,
} from "@/lib/api/training-jobs";

const RUN_CONTRACT_E2E = process.env.WORKBENCH_RUN_CONTRACT_E2E === "1";
const TOKEN = "contract-e2e-token";
const REPOSITORY_ROOT = resolve(process.cwd(), "../..");
const SERVER_SCRIPT = join(
  REPOSITORY_ROOT,
  "workbench/backend/tests/contract_e2e_server.py",
);
const REPOSITORY_PYTHON = join(REPOSITORY_ROOT, "torchenv/bin/python");

type RunningBackend = {
  baseUrl: string;
  child: ChildProcessWithoutNullStreams;
  output: () => string;
  root: string;
};

const backends: RunningBackend[] = [];
const nativeFetch = globalThis.fetch;
const connectionEnvironment: WorkbenchConnectionActionEnvironment = {
  publishRuntime: () => undefined,
  queryClient: new QueryClient(),
  resetProtectedState: () => undefined,
  setIsChanging: () => undefined,
};

function getWorkbenchApiBaseUrl() {
  return workbenchConnectionRuntimeSnapshot().apiBaseUrl;
}

async function setWorkbenchApiBaseUrl(url: string) {
  const outcome = await applyApiBaseUrl(connectionEnvironment, url);
  if (!outcome.ok) {
    throw new Error(outcome.message);
  }
}

async function resetWorkbenchApiBaseUrl() {
  const outcome = await resetApiBaseUrl(connectionEnvironment);
  if (!outcome.ok) {
    throw new Error(outcome.message);
  }
}

async function setSessionAuthToken(token: string) {
  const outcome = await signIn(connectionEnvironment, token);
  if (!outcome.ok) {
    throw new Error(outcome.message);
  }
}

async function clearSessionAuthToken() {
  const outcome = await logout(connectionEnvironment);
  if (!outcome.ok) {
    throw new Error(outcome.message);
  }
}

async function unusedLoopbackPort() {
  const server = createServer();
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    server.close();
    throw new Error("Could not reserve a loopback port for the contract E2E");
  }
  const { port } = address;
  await new Promise<void>((resolve, reject) => {
    server.close((error) => (error ? reject(error) : resolve()));
  });
  return port;
}

async function waitForHealth(backend: RunningBackend) {
  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline) {
    if (backend.child.exitCode !== null) {
      throw new Error(
        `Contract backend exited during startup:\n${backend.output()}`,
      );
    }
    try {
      const response = await nativeFetch(`${backend.baseUrl}/health`);
      if (response.ok) {
        return;
      }
    } catch {
      // The socket is not listening yet.
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`Contract backend did not start:\n${backend.output()}`);
}

async function startBackend(name: string) {
  const port = await unusedLoopbackPort();
  const root = await mkdtemp(join(tmpdir(), `emperor-contract-${name}-`));
  const python =
    process.env.WORKBENCH_E2E_PYTHON ??
    (existsSync(REPOSITORY_PYTHON) ? REPOSITORY_PYTHON : "python");
  const child = spawn(
    python,
    [
      "-P",
      SERVER_SCRIPT,
      "--root",
      root,
      "--port",
      String(port),
      "--token",
      TOKEN,
      "--frontend-origin",
      window.location.origin,
    ],
    {
      cwd: REPOSITORY_ROOT,
      env: {
        ...process.env,
        MPLCONFIGDIR: "/tmp/matplotlib",
        PYTHONPATH: REPOSITORY_ROOT,
        PYTHONSAFEPATH: "1",
      },
      stdio: "pipe",
    },
  );
  let output = "";
  child.stdout.on("data", (chunk) => {
    output += String(chunk);
  });
  child.stderr.on("data", (chunk) => {
    output += String(chunk);
  });
  const backend = {
    baseUrl: `http://127.0.0.1:${port}`,
    child,
    output: () => output,
    root,
  };
  backends.push(backend);
  await waitForHealth(backend);
  return backend;
}

async function waitForExit(
  child: ChildProcessWithoutNullStreams,
  timeout: number,
) {
  if (child.exitCode !== null) {
    return true;
  }
  return new Promise<boolean>((resolve) => {
    const handleExit = () => {
      clearTimeout(timer);
      resolve(true);
    };
    const timer = setTimeout(() => {
      child.off("exit", handleExit);
      resolve(child.exitCode !== null);
    }, timeout);
    child.once("exit", handleExit);
  });
}

async function stopBackend(backend: RunningBackend) {
  if (backend.child.exitCode === null) {
    backend.child.kill("SIGTERM");
    if (!(await waitForExit(backend.child, 1_000))) {
      backend.child.kill("SIGKILL");
      await waitForExit(backend.child, 3_000);
    }
  }
  await rm(backend.root, { force: true, recursive: true });
}

async function captureError(action: () => Promise<unknown>) {
  try {
    await action();
  } catch (error) {
    return error;
  }
  throw new Error("Expected the frontend API client request to fail");
}

describe.skipIf(!RUN_CONTRACT_E2E)(
  "live backend/frontend contract",
  () => {
    let alpha: RunningBackend;
    let beta: RunningBackend;

    beforeAll(async () => {
      window.sessionStorage.clear();
      alpha = await startBackend("alpha");
      beta = await startBackend("beta");
      globalThis.fetch = (input, init) => {
        const headers = new Headers(init?.headers);
        headers.set("Origin", window.location.origin);
        return nativeFetch(input, { ...init, headers });
      };
    }, 30_000);

    afterAll(async () => {
      globalThis.fetch = nativeFetch;
      await clearSessionAuthToken();
      await resetWorkbenchApiBaseUrl();
      await Promise.all(backends.map(stopBackend));
    }, 15_000);

    it("crosses auth, mutation, origin, error, and logout boundaries", async () => {
      await setWorkbenchApiBaseUrl(alpha.baseUrl);
      expect(getWorkbenchApiBaseUrl()).toBe(alpha.baseUrl);
      await expect(fetchHealth()).resolves.toEqual({ status: "ok" });
      await expect(fetchCapabilities()).resolves.toMatchObject({
        authMode: "bearer",
        configSnapshotsEnabled: true,
        trainingEnabled: true,
        uploadsEnabled: false,
      });
      observeWorkbenchAuthMode("bearer");

      const missingTokenError = await captureError(() =>
        fetchModels(),
      );
      expect(isUnauthorizedApiError(missingTokenError)).toBe(true);
      if (isUnauthorizedApiError(missingTokenError)) {
        expect(missingTokenError).toMatchObject({
          detail: "Missing or invalid bearer credentials",
          method: "GET",
          path: "/models",
          status: 401,
        });
      }

      await setSessionAuthToken("wrong-token");
      observeWorkbenchAuthMode("bearer");
      expect(
        isUnauthorizedApiError(
          await captureError(fetchModels),
        ),
      ).toBe(true);
      await setSessionAuthToken(TOKEN);
      observeWorkbenchAuthMode("bearer");
      const models = await fetchModels();
      expect(models.models).toContainEqual({
        model: "linear",
        modelType: "linears",
      });

      const snapshot = await createConfigSnapshot(
        {
          model: "linear",
          modelType: "linears",
          name: "alpha snapshot",
          overrides: { batch_size: "2" },
          preset: "baseline",
        },
        createMutationRequestOptions(),
      );
      expect(snapshot.name).toBe("alpha snapshot");

      const job = await createTrainingJob(
        {
          datasets: ["Mnist"],
          logFolder: "contract_e2e",
          model: "linear",
          modelType: "linears",
          monitors: [],
          overrides: {},
          preset: "baseline",
        },
        createMutationRequestOptions(),
      );
      expect(job).toMatchObject({
        logTail: ["fake training log"],
        model: "linear",
        modelType: "linears",
        status: "running",
      });
      await expect(fetchTrainingJob(job.id)).resolves.toMatchObject({
        id: job.id,
        status: "running",
      });

      await setWorkbenchApiBaseUrl(beta.baseUrl);
      expect(getWorkbenchApiBaseUrl()).toBe(beta.baseUrl);
      await expect(fetchCapabilities()).resolves.toMatchObject({
        authMode: "bearer",
        trainingEnabled: true,
      });
      observeWorkbenchAuthMode("bearer");
      await expect(
        fetchModels(),
      ).resolves.toMatchObject({ models: expect.any(Array) });
      await expect(fetchConfigSnapshotLibrary()).resolves.toEqual({
        snapshots: [],
      });
      const normalizedError = await captureError(() => fetchTrainingJob(job.id));
      expect(normalizedError).toMatchObject({
        detail: `Unknown training job '${job.id}'.`,
        method: "GET",
        path: `/training/jobs/${job.id}`,
        status: 400,
      });
      expect(String(normalizedError)).toContain(beta.baseUrl);

      await setWorkbenchApiBaseUrl(alpha.baseUrl);
      observeWorkbenchAuthMode("bearer");
      await expect(
        fetchModels(),
      ).resolves.toMatchObject({ models: expect.any(Array) });
      await expect(fetchConfigSnapshotLibrary()).resolves.toMatchObject({
        snapshots: [expect.objectContaining({ id: snapshot.id })],
      });

      await clearSessionAuthToken();
      const logoutError = await captureError(() =>
        fetchModels(),
      );
      expect(isUnauthorizedApiError(logoutError)).toBe(true);
    }, 30_000);
  },
);
