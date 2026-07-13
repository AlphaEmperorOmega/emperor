import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:net";
import os from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  collectBuildPerformanceEvidence,
  createBrowserPerformanceEvidence,
  createBrowserPerformanceThresholds,
  deterministicPerformanceFailures,
} from "./performance-evidence.mjs";
import {
  matplotlibConfigDirectory,
  nextCli,
  resolveRepositoryPython,
} from "./runtime-paths.mjs";

const SCRIPT_DIRECTORY = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = resolve(SCRIPT_DIRECTORY, "..");
const REPOSITORY_ROOT = resolve(FRONTEND_ROOT, "../..");
const PYTHON = resolveRepositoryPython(REPOSITORY_ROOT);
const BACKEND_SCRIPT = resolve(
  REPOSITORY_ROOT,
  "workbench/backend/tests/browser_performance_server.py",
);
const NEXT_BINARY = nextCli(FRONTEND_ROOT);
const DEFAULTS = {
  initialWarmup: 2,
  initialRepetitions: 7,
  sessionWarmup: 2,
  sessionRepetitions: 20,
  steadyStateRepetitions: 10,
  webglWarmup: 2,
  webglRepetitions: 5,
  frameRepetitions: 60,
};

function parsePositiveInteger(value, flag) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${flag} must be a positive integer.`);
  }
  return parsed;
}

function parseArguments(argv) {
  const options = {
    ...DEFAULTS,
    chromium: process.env.CHROMIUM_BIN ?? "chromium",
  };
  const fields = {
    "--initial-warmup": "initialWarmup",
    "--initial-repetitions": "initialRepetitions",
    "--session-warmup": "sessionWarmup",
    "--session-repetitions": "sessionRepetitions",
    "--steady-state-repetitions": "steadyStateRepetitions",
    "--webgl-warmup": "webglWarmup",
    "--webgl-repetitions": "webglRepetitions",
    "--frame-repetitions": "frameRepetitions",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--help") {
      console.log(`Usage: node scripts/benchmark-browser-long-session.mjs [options]

Options:
  --initial-warmup N
  --initial-repetitions N
  --session-warmup N
  --session-repetitions N
  --steady-state-repetitions N
  --webgl-warmup N
  --webgl-repetitions N
  --frame-repetitions N
  --chromium PATH`);
      process.exit(0);
    }
    if (argument === "--chromium") {
      options.chromium = argv[index + 1];
      index += 1;
      continue;
    }
    const field = fields[argument];
    if (!field) {
      throw new Error(`Unknown argument: ${argument}`);
    }
    options[field] = parsePositiveInteger(argv[index + 1], argument);
    index += 1;
  }
  return options;
}

function percentile(values, ratio) {
  const ordered = [...values].sort((left, right) => left - right);
  if (ordered.length === 1) {
    return ordered[0];
  }
  const rank = (ordered.length - 1) * ratio;
  const lower = Math.floor(rank);
  const upper = Math.ceil(rank);
  const fraction = rank - lower;
  return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction;
}

function summarize(values) {
  if (values.length === 0) {
    return null;
  }
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const variance =
    values.length > 1
      ? values.reduce((total, value) => total + (value - mean) ** 2, 0) /
        (values.length - 1)
      : 0;
  const stdev = Math.sqrt(variance);
  return {
    coefficient_of_variation: mean === 0 ? 0 : stdev / mean,
    max: Math.max(...values),
    mean,
    median: percentile(values, 0.5),
    min: Math.min(...values),
    p95: percentile(values, 0.95),
    stdev,
  };
}

function metricMap(metrics) {
  return Object.fromEntries(metrics.metrics.map(({ name, value }) => [name, value]));
}

function metricDelta(before, after, name, multiplier = 1) {
  const beforeValue = before[name] ?? 0;
  const afterValue = after[name] ?? 0;
  const difference = afterValue >= beforeValue ? afterValue - beforeValue : afterValue;
  return difference * multiplier;
}

async function unusedLoopbackPort() {
  const server = createServer();
  await new Promise((resolvePromise, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolvePromise);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    server.close();
    throw new Error("Could not reserve a loopback port.");
  }
  const { port } = address;
  await new Promise((resolvePromise, reject) => {
    server.close((error) => (error ? reject(error) : resolvePromise()));
  });
  return port;
}

function managedProcess(command, args, options) {
  const child = spawn(command, args, {
    ...options,
    stdio: "pipe",
  });
  let output = "";
  const append = (chunk) => {
    output = `${output}${String(chunk)}`.slice(-200_000);
  };
  child.stdout.on("data", append);
  child.stderr.on("data", append);
  return { child, output: () => output };
}

async function waitForUrl(url, processHandle, label, timeout = 45_000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (processHandle.child.exitCode !== null) {
      throw new Error(`${label} exited during startup:\n${processHandle.output()}`);
    }
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
    } catch {
      // The server is not listening yet.
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 75));
  }
  throw new Error(`${label} did not become ready:\n${processHandle.output()}`);
}

async function waitForExit(child, timeout) {
  if (child.exitCode !== null) {
    return true;
  }
  return new Promise((resolvePromise) => {
    const handleExit = () => {
      clearTimeout(timer);
      resolvePromise(true);
    };
    const timer = setTimeout(() => {
      child.off("exit", handleExit);
      resolvePromise(child.exitCode !== null);
    }, timeout);
    child.once("exit", handleExit);
  });
}

async function stopProcess(processHandle) {
  if (!processHandle || processHandle.child.exitCode !== null) {
    return;
  }
  processHandle.child.kill("SIGTERM");
  if (!(await waitForExit(processHandle.child, 2_000))) {
    processHandle.child.kill("SIGKILL");
    await waitForExit(processHandle.child, 3_000);
  }
}

class CdpClient {
  constructor(url) {
    this.url = url;
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Map();
  }

  async connect() {
    this.socket = new WebSocket(this.url);
    await new Promise((resolvePromise, reject) => {
      this.socket.addEventListener("open", resolvePromise, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
    this.socket.addEventListener("message", (event) => {
      const message = JSON.parse(String(event.data));
      if (message.id) {
        const pending = this.pending.get(message.id);
        if (!pending) {
          return;
        }
        this.pending.delete(message.id);
        if (message.error) {
          pending.reject(
            new Error(`${pending.method}: ${message.error.message ?? "CDP error"}`),
          );
        } else {
          pending.resolve(message.result ?? {});
        }
        return;
      }
      for (const listener of this.listeners.get(message.method) ?? []) {
        listener(message.params ?? {});
      }
    });
    this.socket.addEventListener("close", () => {
      for (const pending of this.pending.values()) {
        pending.reject(new Error(`CDP socket closed while awaiting ${pending.method}.`));
      }
      this.pending.clear();
    });
  }

  send(method, params = {}) {
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((resolvePromise, reject) => {
      this.pending.set(id, { resolve: resolvePromise, reject, method });
      this.socket.send(JSON.stringify({ id, method, params }));
    });
  }

  on(method, listener) {
    const listeners = this.listeners.get(method) ?? [];
    listeners.push(listener);
    this.listeners.set(method, listeners);
    return () => {
      this.listeners.set(
        method,
        (this.listeners.get(method) ?? []).filter((candidate) => candidate !== listener),
      );
    };
  }

  once(method, timeout = 30_000) {
    return new Promise((resolvePromise, reject) => {
      const remove = this.on(method, (params) => {
        clearTimeout(timer);
        remove();
        resolvePromise(params);
      });
      const timer = setTimeout(() => {
        remove();
        reject(new Error(`Timed out waiting for ${method}.`));
      }, timeout);
    });
  }

  close() {
    this.socket?.close();
  }
}

function installPageInstrumentation(apiBaseUrl) {
  try {
    window.localStorage.clear();
    window.localStorage.setItem("emperor.workbench.apiBaseUrl", apiBaseUrl);
  } catch {
    // about:blank has no storage; the script runs again in the app document.
  }

  const state = {
    errors: [],
    layoutShifts: [],
    longTasks: [],
    reactCommits: [],
    webgl: {
      contextsCreated: 0,
      contextsLost: 0,
      created: {},
      deleted: {},
      live: {},
    },
  };
  Object.defineProperty(window, "__EMPEROR_BROWSER_PERFORMANCE__", {
    configurable: true,
    value: state,
  });
  performance.setResourceTimingBufferSize(5_000);

  const errorText = (value) => {
    try {
      return value instanceof Error ? `${value.name}: ${value.message}` : String(value);
    } catch {
      return "unprintable browser error";
    }
  };
  window.addEventListener("error", (event) => {
    state.errors.push(errorText(event.error ?? event.message));
  });
  window.addEventListener("unhandledrejection", (event) => {
    state.errors.push(errorText(event.reason));
  });

  try {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        state.longTasks.push({ duration: entry.duration, startTime: entry.startTime });
      }
    }).observe({ entryTypes: ["longtask"] });
  } catch {
    // Long Task timing is not exposed by every browser build.
  }
  try {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (!entry.hadRecentInput) {
          state.layoutShifts.push({ startTime: entry.startTime, value: entry.value });
        }
      }
    }).observe({ type: "layout-shift", buffered: true });
  } catch {
    // Layout Instability timing is not exposed by every browser build.
  }

  let rendererId = 0;
  window.__REACT_DEVTOOLS_GLOBAL_HOOK__ = {
    checkDCE() {},
    inject(renderer) {
      rendererId += 1;
      this.renderers.set(rendererId, renderer);
      return rendererId;
    },
    onCommitFiberRoot(id, root, priorityLevel, didError) {
      state.reactCommits.push({
        didError: Boolean(didError),
        rendererId: id,
        startTime: performance.now(),
      });
    },
    onCommitFiberUnmount() {},
    onScheduleFiberRoot() {},
    rendererInterfaces: new Map(),
    renderers: new Map(),
    supportsFiber: true,
  };

  const contexts = new WeakSet();
  const resources = new Map();
  const deletedResources = new Map();
  const resourceMethods = {
    buffer: ["createBuffer", "deleteBuffer"],
    framebuffer: ["createFramebuffer", "deleteFramebuffer"],
    program: ["createProgram", "deleteProgram"],
    query: ["createQuery", "deleteQuery"],
    renderbuffer: ["createRenderbuffer", "deleteRenderbuffer"],
    sampler: ["createSampler", "deleteSampler"],
    shader: ["createShader", "deleteShader"],
    texture: ["createTexture", "deleteTexture"],
    transformFeedback: ["createTransformFeedback", "deleteTransformFeedback"],
    vertexArray: ["createVertexArray", "deleteVertexArray"],
  };
  for (const type of Object.keys(resourceMethods)) {
    resources.set(type, new WeakSet());
    deletedResources.set(type, new WeakSet());
    state.webgl.created[type] = 0;
    state.webgl.deleted[type] = 0;
    state.webgl.live[type] = 0;
  }

  const patchPrototype = (prototype) => {
    if (!prototype || prototype.__emperorPerformancePatched) {
      return;
    }
    Object.defineProperty(prototype, "__emperorPerformancePatched", { value: true });
    for (const [type, [createName, deleteName]] of Object.entries(resourceMethods)) {
      const originalCreate = prototype[createName];
      const originalDelete = prototype[deleteName];
      if (typeof originalCreate === "function") {
        prototype[createName] = function (...args) {
          const resource = originalCreate.apply(this, args);
          const seen = resources.get(type);
          if (resource && !seen.has(resource)) {
            seen.add(resource);
            state.webgl.created[type] += 1;
            state.webgl.live[type] += 1;
          }
          return resource;
        };
      }
      if (typeof originalDelete === "function") {
        prototype[deleteName] = function (resource, ...args) {
          const seen = resources.get(type);
          const deleted = deletedResources.get(type);
          if (resource && seen.has(resource) && !deleted.has(resource)) {
            deleted.add(resource);
            state.webgl.deleted[type] += 1;
            state.webgl.live[type] -= 1;
          }
          return originalDelete.call(this, resource, ...args);
        };
      }
    }
  };
  patchPrototype(window.WebGLRenderingContext?.prototype);
  patchPrototype(window.WebGL2RenderingContext?.prototype);

  const originalGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (...args) {
    const context = originalGetContext.apply(this, args);
    const kind = String(args[0] ?? "").toLowerCase();
    if (context && (kind === "webgl" || kind === "webgl2" || kind === "experimental-webgl")) {
      if (!contexts.has(context)) {
        contexts.add(context);
        state.webgl.contextsCreated += 1;
        this.addEventListener(
          "webglcontextlost",
          () => {
            state.webgl.contextsLost += 1;
          },
          { once: true },
        );
      }
    }
    return context;
  };

  const normalize = (value) => String(value ?? "").replace(/\s+/g, " ").trim();
  const accessibleName = (element) =>
    normalize(
      element.getAttribute("aria-label") ||
        (element.getAttribute("aria-labelledby")
          ? document.getElementById(element.getAttribute("aria-labelledby"))?.textContent
          : "") ||
        element.textContent,
    );
  const isVisible = (element) =>
    element.getClientRects().length > 0 &&
    getComputedStyle(element).visibility !== "hidden";
  const scope = (selector) => (selector ? document.querySelector(selector) : document);
  state.clickButton = (name, scopeSelector = null, contains = false) => {
    const root = scope(scopeSelector);
    const expected = normalize(name).toLowerCase();
    const buttons = [...(root?.querySelectorAll("button, [role='button']") ?? [])];
    const button = buttons.find((candidate) => {
      const candidateName = accessibleName(candidate).toLowerCase();
      const nameMatches = contains
        ? candidateName.includes(expected)
        : candidateName === expected;
      return nameMatches && !candidate.disabled && isVisible(candidate);
    });
    if (!button) {
      return { ok: false, candidates: buttons.map(accessibleName).slice(0, 80) };
    }
    button.click();
    return { ok: true, name: accessibleName(button) };
  };
  state.clickWorkspace = (name) => {
    const navigation = document.querySelector("nav[aria-label='Workspace']");
    const expected = normalize(name).toLowerCase();
    const button = [...(navigation?.querySelectorAll("button") ?? [])].find(
      (candidate) => accessibleName(candidate).toLowerCase() === expected,
    );
    if (!button) {
      return { ok: false };
    }
    button.click();
    return { ok: true };
  };
  state.clickCombobox = (label) => {
    const expected = normalize(label).toLowerCase();
    const controls = [...document.querySelectorAll("button[role='combobox']")];
    const control = controls.find(
      (candidate) =>
        accessibleName(candidate).toLowerCase() === expected &&
        isVisible(candidate),
    );
    if (!control) {
      return { ok: false, candidates: controls.map(accessibleName) };
    }
    control.click();
    return { ok: true };
  };
  state.clickOption = (label, contains = false) => {
    const expected = normalize(label).toLowerCase();
    const options = [...document.querySelectorAll("[role='option']")];
    const option = options.find((candidate) => {
      const candidateName = accessibleName(candidate).toLowerCase();
      const nameMatches = contains
        ? candidateName.includes(expected)
        : candidateName === expected;
      return nameMatches && isVisible(candidate);
    });
    if (!option) {
      return { ok: false, candidates: options.map(accessibleName) };
    }
    option.click();
    return { ok: true, name: accessibleName(option) };
  };
  state.options = () =>
    [...document.querySelectorAll("[role='option']")]
      .filter(isVisible)
      .map(accessibleName);
  state.setInput = (label, value) => {
    const expected = normalize(label).toLowerCase();
    const inputs = [...document.querySelectorAll("input, textarea")];
    const input = inputs.find(
      (candidate) =>
        accessibleName(candidate).toLowerCase() === expected &&
        isVisible(candidate),
    );
    if (!input) {
      return { ok: false, candidates: inputs.map(accessibleName) };
    }
    const prototype = input instanceof HTMLTextAreaElement
      ? HTMLTextAreaElement.prototype
      : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, "value").set.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    return { ok: true };
  };
}

async function evaluate(client, expression) {
  const response = await client.send("Runtime.evaluate", {
    awaitPromise: true,
    expression,
    returnByValue: true,
    userGesture: true,
  });
  if (response.exceptionDetails) {
    throw new Error(
      response.exceptionDetails.exception?.description ??
        response.exceptionDetails.text ??
        "Browser evaluation failed.",
    );
  }
  return response.result?.value;
}

async function waitForExpression(client, expression, label, timeout = 30_000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    try {
      if (await evaluate(client, `Boolean(${expression})`)) {
        return;
      }
    } catch {
      // A navigation can briefly invalidate the execution context.
    }
    await new Promise((resolvePromise) => setTimeout(resolvePromise, 50));
  }
  throw new Error(`Timed out waiting for ${label}.`);
}

async function pageMethod(client, method, args = []) {
  return evaluate(
    client,
    `window.__EMPEROR_BROWSER_PERFORMANCE__.${method}(...${JSON.stringify(args)})`,
  );
}

async function requirePageMethod(client, method, args, label) {
  const result = await pageMethod(client, method, args);
  if (!result?.ok) {
    throw new Error(`${label} was not available: ${JSON.stringify(result)}`);
  }
  return result;
}

async function clickWorkspace(client, workspace) {
  await requirePageMethod(
    client,
    "clickWorkspace",
    [workspace],
    `${workspace} workspace control`,
  );
  await waitForExpression(
    client,
    `[...document.querySelectorAll("nav[aria-label='Workspace'] button")].some(
      (button) => button.textContent.trim() === ${JSON.stringify(workspace)} &&
        button.getAttribute("aria-current") === "page"
    )`,
    `${workspace} workspace selection`,
  );
  const readiness = {
    Model:
      "document.querySelector('.react-flow')?.getClientRects().length > 0",
    Training:
      "document.querySelector('#training-workspace')?.getClientRects().length > 0",
    Logs: "document.body.innerText.includes('Historical Scalars')",
  }[workspace];
  await waitForExpression(client, readiness, `${workspace} workspace`);
}

async function selectDropdown(client, label, option, contains = false) {
  await requirePageMethod(client, "clickCombobox", [label], `${label} combobox`);
  await waitForExpression(
    client,
    "window.__EMPEROR_BROWSER_PERFORMANCE__.options().length > 0",
    `${label} options`,
  );
  await requirePageMethod(
    client,
    "clickOption",
    [option, contains],
    `${label} option ${option}`,
  );
}

async function pageSnapshot(client) {
  return evaluate(
    client,
    `(() => {
      const state = window.__EMPEROR_BROWSER_PERFORMANCE__;
      const navigation = performance.getEntriesByType("navigation")[0];
      const paints = Object.fromEntries(
        performance.getEntriesByType("paint").map((entry) => [entry.name, entry.startTime]),
      );
      const webglCanvasCount = [...document.querySelectorAll("canvas")].filter((canvas) =>
        ["webgl", "webgl2", "experimental-webgl"].some((kind) => {
          try {
            return Boolean(canvas.getContext(kind));
          } catch {
            return false;
          }
        }),
      ).length;
      return {
        cumulativeLayoutShift: state.layoutShifts.reduce(
          (total, entry) => total + entry.value,
          0,
        ),
        errors: [...state.errors],
        firstReactCommitMs: state.reactCommits[0]?.startTime ?? null,
        longTaskCount: state.longTasks.length,
        longTaskTotalMs: state.longTasks.reduce(
          (total, entry) => total + entry.duration,
          0,
        ),
        navigation: navigation
          ? {
              domContentLoadedEventEnd: navigation.domContentLoadedEventEnd,
              domInteractive: navigation.domInteractive,
              loadEventEnd: navigation.loadEventEnd,
              responseEnd: navigation.responseEnd,
              transferSize: navigation.transferSize,
            }
          : null,
        now: performance.now(),
        paints,
        reactCommitCount: state.reactCommits.length,
        resourceTransferSize: performance
          .getEntriesByType("resource")
          .reduce((total, entry) => total + (entry.transferSize ?? 0), 0),
        webgl: JSON.parse(JSON.stringify(state.webgl)),
        webglCanvasCount,
      };
    })()`,
  );
}

async function performanceMetrics(client) {
  return metricMap(await client.send("Performance.getMetrics"));
}

async function operationSnapshot(client) {
  return {
    metrics: await performanceMetrics(client),
    page: await pageSnapshot(client),
  };
}

function operationDelta(label, before, after, durationMs) {
  return {
    duration_ms: durationMs,
    label,
    layout_count: metricDelta(before.metrics, after.metrics, "LayoutCount"),
    layout_duration_ms: metricDelta(
      before.metrics,
      after.metrics,
      "LayoutDuration",
      1_000,
    ),
    long_task_count: after.page.longTaskCount - before.page.longTaskCount,
    long_task_duration_ms: after.page.longTaskTotalMs - before.page.longTaskTotalMs,
    react_commits: after.page.reactCommitCount - before.page.reactCommitCount,
    recalc_style_count: metricDelta(
      before.metrics,
      after.metrics,
      "RecalcStyleCount",
    ),
    recalc_style_duration_ms: metricDelta(
      before.metrics,
      after.metrics,
      "RecalcStyleDuration",
      1_000,
    ),
    script_duration_ms: metricDelta(
      before.metrics,
      after.metrics,
      "ScriptDuration",
      1_000,
    ),
    task_duration_ms: metricDelta(
      before.metrics,
      after.metrics,
      "TaskDuration",
      1_000,
    ),
  };
}

async function measureOperation(client, label, action) {
  const before = await operationSnapshot(client);
  const startedAt = await evaluate(client, "performance.now()");
  await action();
  const endedAt = await evaluate(client, "performance.now()");
  const after = await operationSnapshot(client);
  return operationDelta(label, before, after, endedAt - startedAt);
}

async function measureNavigation(client, frontendUrl) {
  await client.send("Network.clearBrowserCache");
  const loaded = client.once("Page.loadEventFired");
  await client.send("Page.navigate", { url: frontendUrl });
  await loaded;
  await waitForExpression(
    client,
    "document.querySelector('nav[aria-label=Workspace]') && " +
      "document.querySelector(\"button[role=combobox][aria-label='model']\") && " +
      "document.querySelector('.react-flow')",
    "initial Model workspace",
    45_000,
  );
  const workspaceReadyMs = await evaluate(client, "performance.now()");
  await new Promise((resolvePromise) => setTimeout(resolvePromise, 250));
  const page = await pageSnapshot(client);
  const after = await performanceMetrics(client);
  const heap = await client.send("Runtime.getHeapUsage");
  const dom = await client.send("Memory.getDOMCounters");
  return {
    cumulative_layout_shift: page.cumulativeLayoutShift,
    dom_content_loaded_ms: page.navigation?.domContentLoadedEventEnd ?? null,
    dom_counters: dom,
    first_contentful_paint_ms: page.paints["first-contentful-paint"] ?? null,
    first_paint_ms: page.paints["first-paint"] ?? null,
    first_react_commit_ms: page.firstReactCommitMs,
    heap_used_bytes: heap.usedSize,
    layout_count: after.LayoutCount ?? 0,
    layout_duration_ms: (after.LayoutDuration ?? 0) * 1_000,
    load_event_ms: page.navigation?.loadEventEnd ?? null,
    long_task_count: page.longTaskCount,
    long_task_duration_ms: page.longTaskTotalMs,
    react_commits: page.reactCommitCount,
    recalc_style_count: after.RecalcStyleCount ?? 0,
    recalc_style_duration_ms: (after.RecalcStyleDuration ?? 0) * 1_000,
    resource_transfer_bytes: page.resourceTransferSize,
    script_duration_ms: (after.ScriptDuration ?? 0) * 1_000,
    task_duration_ms: (after.TaskDuration ?? 0) * 1_000,
    workspace_ready_ms: workspaceReadyMs,
  };
}

function summarizeFields(samples, fields) {
  return Object.fromEntries(
    fields.map((field) => [
      field,
      summarize(
        samples
          .map((sample) => sample[field])
          .filter((value) => typeof value === "number" && Number.isFinite(value)),
      ),
    ]),
  );
}

function normalizedApiPath(url) {
  return new URL(url).pathname
    .replace(
      /[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}/gi,
      ":id",
    )
    .replace(/[a-f0-9]{16,64}/gi, ":id");
}

function summarizeApiEntries(entries) {
  const grouped = new Map();
  for (const entry of entries) {
    const path = normalizedApiPath(entry.name);
    const existing = grouped.get(path) ?? [];
    existing.push(entry);
    grouped.set(path, existing);
  }
  return Object.fromEntries(
    [...grouped.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([path, pathEntries]) => [
        path,
        {
          count: pathEntries.length,
          duration_ms: summarize(pathEntries.map((entry) => entry.duration)),
          transfer_bytes: pathEntries.reduce(
            (total, entry) => total + entry.transferSize,
            0,
          ),
        },
      ]),
  );
}

async function collectHeapCheckpoint(client, label) {
  await client.send("HeapProfiler.collectGarbage");
  await new Promise((resolvePromise) => setTimeout(resolvePromise, 100));
  return {
    ...(await client.send("Runtime.getHeapUsage")),
    ...(await client.send("Memory.getDOMCounters")),
    label,
  };
}

async function runWorkspaceCycle(client, preset) {
  return measureOperation(client, `workspace_cycle_${preset}`, async () => {
    await clickWorkspace(client, "Model");
    await selectDropdown(client, "preset", preset);
    await waitForExpression(
      client,
      `document.querySelector("button[role=combobox][aria-label='preset']")
        ?.textContent.toLowerCase().includes(${JSON.stringify(preset.toLowerCase())}) &&
       !document.body.innerText.includes("preview building") &&
       document.querySelector('.react-flow')`,
      `graph for ${preset}`,
      45_000,
    );
    await clickWorkspace(client, "Training");
    await clickWorkspace(client, "Logs");
    await waitForExpression(
      client,
      "document.querySelector(\"button[aria-label^='Explain metric']\")",
      "scalar chart info control",
      30_000,
    );
    await evaluate(
      client,
      "document.querySelector(\"button[aria-label^='Explain metric']\").click()",
    );
    await waitForExpression(client, "document.querySelector('[role=dialog]')", "metric dialog");
    await requirePageMethod(
      client,
      "clickButton",
      ["Close metric info", "[role=dialog]", false],
      "Close metric info control",
    );
    await waitForExpression(
      client,
      "!document.querySelector('[role=dialog]')",
      "closed metric dialog",
    );
  });
}

async function selectBrowserPerformanceLogs(client) {
  await clickWorkspace(client, "Logs");
  await waitForExpression(
    client,
    `[...document.querySelectorAll("[role='combobox']")].some((candidate) =>
      candidate.getAttribute("aria-label")?.toLowerCase().startsWith("experiments"),
    )`,
    "Experiments log filter",
    30_000,
  );
  const opened = await evaluate(
    client,
    `(() => {
      const control = [...document.querySelectorAll("button[role='combobox']")]
        .find((candidate) =>
          candidate.getAttribute("aria-label")?.toLowerCase().startsWith("experiments"),
        );
      if (!control) {
        return false;
      }
      control.click();
      return true;
    })()`,
  );
  if (!opened) {
    throw new Error("Experiments log filter was not available.");
  }
  await waitForExpression(
    client,
    "document.querySelector(\"[role=option][aria-label='browser_performance']\")",
    "browser_performance log experiment option",
  );
  await requirePageMethod(
    client,
    "clickOption",
    ["browser_performance", false],
    "browser_performance log experiment option",
  );
  await waitForExpression(
    client,
    "document.querySelector(\"button[aria-label^='Explain metric']\")",
    "seeded scalar charts",
    30_000,
  );
  await clickWorkspace(client, "Model");
}

async function completeTrainingJob(client) {
  return measureOperation(client, "training_job_completion", async () => {
    await clickWorkspace(client, "Training");
    await requirePageMethod(
      client,
      "clickButton",
      ["New folder", null, false],
      "New folder control",
    );
    await requirePageMethod(
      client,
      "setInput",
      ["New log folder", "browser_performance_training"],
      "New log folder input",
    );
    await waitForExpression(
      client,
      "[...document.querySelectorAll('button')].some((button) => " +
        "button.textContent.includes('Start Training') && !button.disabled)",
      "enabled Start Training control",
      30_000,
    );
    const submittedJobCount = await evaluate(
      client,
      `performance.getEntriesByType("resource").filter((entry) =>
        new URL(entry.name).pathname === "/training/jobs"
      ).length`,
    );
    await requirePageMethod(
      client,
      "clickButton",
      ["Start Training", null, false],
      "Start Training control",
    );
    await waitForExpression(
      client,
      `performance.getEntriesByType("resource").filter((entry) =>
          new URL(entry.name).pathname === "/training/jobs"
        ).length > ${submittedJobCount} ||
        document.querySelector("[role=dialog]")?.innerText.includes("Confirm Grid Search")`,
      "Training Job request or grid confirmation",
      30_000,
    );
    const requiresConfirmation = await evaluate(
      client,
      `document.querySelector("[role=dialog]")
        ?.innerText.includes("Confirm Grid Search") ?? false`,
    );
    if (requiresConfirmation) {
      await requirePageMethod(
        client,
        "clickButton",
        ["Start Training", "[role=dialog]", false],
        "confirmed grid Training Job",
      );
    }
    await waitForExpression(
      client,
      `performance.getEntriesByType("resource").filter((entry) =>
        new URL(entry.name).pathname === "/training/jobs"
      ).length > ${submittedJobCount}`,
      "submitted Training Job request",
      30_000,
    );
    await waitForExpression(
      client,
      `[...document.querySelectorAll("h1")]
        .find((heading) => heading.textContent.trim() === "Training")
        ?.parentElement?.innerText.toLowerCase().includes("completed")`,
      "completed Training Job",
      30_000,
    );
  });
}

async function setFileInput(client, filePath) {
  const response = await client.send("Runtime.evaluate", {
    expression: "document.querySelector(\"input[type='file']\")",
    returnByValue: false,
  });
  const objectId = response.result?.objectId;
  if (!objectId) {
    throw new Error("Log import file input was not found.");
  }
  await client.send("DOM.setFileInputFiles", {
    files: [filePath],
    objectId,
  });
}

async function importLogs(client, archivePath) {
  return measureOperation(client, "log_archive_import", async () => {
    await requirePageMethod(
      client,
      "clickButton",
      ["Import Logs", null, false],
      "Import Logs header control",
    );
    await waitForExpression(
      client,
      "document.querySelector(\"[role=dialog] input[type='file']\")",
      "log import file input",
    );
    await setFileInput(client, archivePath);
    await waitForExpression(
      client,
      "document.body.innerText.includes('browser-performance-import.zip')",
      "selected import archive",
    );
    await requirePageMethod(
      client,
      "clickButton",
      ["Import Logs", "[role=dialog]", false],
      "Import Logs submit control",
    );
    await waitForExpression(
      client,
      "document.querySelector('[role=dialog] [role=status]')",
      "successful log import",
      30_000,
    );
    await requirePageMethod(
      client,
      "clickButton",
      ["Close import logs dialog", "[role=dialog]", false],
      "Close import logs dialog control",
    );
    await waitForExpression(
      client,
      "!document.querySelector('[role=dialog]')",
      "closed log import dialog",
    );
  });
}

async function prepareNeuronGraph(client) {
  await clickWorkspace(client, "Model");
  await selectDropdown(client, "model type", "neuron", true);
  await waitForExpression(
    client,
    "document.querySelector(\"button[role=combobox][aria-label='model']\") && " +
      "!document.querySelector(\"button[role=combobox][aria-label='model']\").disabled",
    "neuron model selection",
  );
  const modelName = await evaluate(
    client,
    "document.querySelector(\"button[role=combobox][aria-label='model']\").textContent.trim()",
  );
  if (!modelName) {
    await selectDropdown(client, "model", "linear", true);
  }
  await waitForExpression(
    client,
    "document.querySelector(\"button[aria-label='Open 3D cluster view']\")",
    "3D cluster control",
    45_000,
  );
}

async function measureWebglCycle(client, frameRepetitions, label) {
  const before = await operationSnapshot(client);
  const beforeWebgl = before.page.webgl;
  const startedAt = await evaluate(client, "performance.now()");
  await requirePageMethod(
    client,
    "clickButton",
    ["Open 3D cluster view", null, false],
    "Open 3D cluster view control",
  );
  await waitForExpression(
    client,
    "document.querySelector(\"[role=dialog][aria-label^='3D neuron cluster'] canvas\")",
    "3D WebGL canvas",
    30_000,
  );
  await new Promise((resolvePromise) => setTimeout(resolvePromise, 350));
  const frameResult = await evaluate(
    client,
    `(async () => {
      const canvas = document.querySelector(
        "[role=dialog][aria-label^='3D neuron cluster'] canvas",
      );
      const context = canvas.getContext("webgl2") || canvas.getContext("webgl");
      if (!context) {
        throw new Error("WebGL context was not created.");
      }
      const debugInfo = context.getExtension("WEBGL_debug_renderer_info");
      const renderer = debugInfo
        ? context.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL)
        : context.getParameter(context.RENDERER);
      const vendor = debugInfo
        ? context.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL)
        : context.getParameter(context.VENDOR);
      const samples = [];
      canvas.dispatchEvent(
        new WheelEvent("wheel", { bubbles: true, cancelable: true, deltaY: 4 }),
      );
      await new Promise(requestAnimationFrame);
      let previous = performance.now();
      for (let index = 0; index < ${frameRepetitions}; index += 1) {
        canvas.dispatchEvent(
          new WheelEvent("wheel", {
            bubbles: true,
            cancelable: true,
            deltaY: index % 2 === 0 ? 4 : -4,
          }),
        );
        await new Promise(requestAnimationFrame);
        const current = performance.now();
        samples.push(current - previous);
        previous = current;
      }
      return { renderer, samples, vendor };
    })()`,
  );
  await requirePageMethod(
    client,
    "clickButton",
    ["Close 3D cluster view", "[role=dialog]", false],
    "Close 3D cluster view control",
  );
  await waitForExpression(
    client,
    "!document.querySelector(\"[role=dialog][aria-label^='3D neuron cluster']\")",
    "closed 3D cluster dialog",
  );
  await new Promise((resolvePromise) => setTimeout(resolvePromise, 850));
  await client.send("HeapProfiler.collectGarbage");
  const endedAt = await evaluate(client, "performance.now()");
  const after = await operationSnapshot(client);
  const afterWebgl = after.page.webgl;
  return {
    ...operationDelta(label, before, after, endedAt - startedAt),
    canvas_count_after_close: after.page.webglCanvasCount,
    contexts_created: afterWebgl.contextsCreated - beforeWebgl.contextsCreated,
    contexts_lost: afterWebgl.contextsLost - beforeWebgl.contextsLost,
    frame_intervals_ms: frameResult.samples,
    renderer: frameResult.renderer,
    resource_created: Object.fromEntries(
      Object.keys(afterWebgl.created).map((type) => [
        type,
        afterWebgl.created[type] - beforeWebgl.created[type],
      ]),
    ),
    resource_deleted: Object.fromEntries(
      Object.keys(afterWebgl.deleted).map((type) => [
        type,
        afterWebgl.deleted[type] - beforeWebgl.deleted[type],
      ]),
    ),
    vendor: frameResult.vendor,
  };
}

async function runBenchmark(options) {
  if (!existsSync(resolve(FRONTEND_ROOT, ".next/BUILD_ID"))) {
    throw new Error("Production build not found. Run npm run build first.");
  }
  if (!existsSync(NEXT_BINARY)) {
    throw new Error("Existing frontend environment is required.");
  }

  const temporaryRoot = await mkdtemp(join(os.tmpdir(), "emperor-browser-performance-"));
  const backendPort = await unusedLoopbackPort();
  const frontendPort = await unusedLoopbackPort();
  const debuggingPort = await unusedLoopbackPort();
  const frontendUrl = `http://127.0.0.1:${frontendPort}`;
  const backendUrl = `http://127.0.0.1:${backendPort}`;
  let backend;
  let frontend;
  let chromium;
  let pageClient;
  let browserClient;

  try {
    backend = managedProcess(
      PYTHON,
      [
        "-P",
        BACKEND_SCRIPT,
        "--root",
        temporaryRoot,
        "--port",
        String(backendPort),
        "--frontend-origin",
        frontendUrl,
      ],
      {
        cwd: REPOSITORY_ROOT,
        env: {
          ...process.env,
          MPLCONFIGDIR: matplotlibConfigDirectory(),
          PYTHONPATH: REPOSITORY_ROOT,
          PYTHONSAFEPATH: "1",
        },
      },
    );
    await waitForUrl(`${backendUrl}/health`, backend, "Browser benchmark backend");

    frontend = managedProcess(
      process.execPath,
      [NEXT_BINARY, "start", "-H", "127.0.0.1", "-p", String(frontendPort)],
      { cwd: FRONTEND_ROOT, env: process.env },
    );
    await waitForUrl(frontendUrl, frontend, "Production frontend");

    chromium = managedProcess(
      options.chromium,
      [
        "--headless=new",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--enable-unsafe-swiftshader",
        `--remote-debugging-port=${debuggingPort}`,
        `--user-data-dir=${resolve(temporaryRoot, "chromium-profile")}`,
        "--window-size=1440,1000",
        "about:blank",
      ],
      { cwd: REPOSITORY_ROOT, env: process.env },
    );
    await waitForUrl(
      `http://127.0.0.1:${debuggingPort}/json/version`,
      chromium,
      "Chromium DevTools endpoint",
    );
    const versionEndpoint = await (
      await fetch(`http://127.0.0.1:${debuggingPort}/json/version`)
    ).json();
    const targets = await (
      await fetch(`http://127.0.0.1:${debuggingPort}/json/list`)
    ).json();
    const pageTarget = targets.find((target) => target.type === "page");
    if (!pageTarget?.webSocketDebuggerUrl) {
      throw new Error("Chromium page target was not available.");
    }
    browserClient = new CdpClient(versionEndpoint.webSocketDebuggerUrl);
    pageClient = new CdpClient(pageTarget.webSocketDebuggerUrl);
    await browserClient.connect();
    await pageClient.connect();

    const diagnostics = {
      console_errors: [],
      failed_requests: [],
      page_exceptions: [],
    };
    pageClient.on("Runtime.consoleAPICalled", (event) => {
      if (event.type === "error" || event.type === "assert") {
        diagnostics.console_errors.push(
          event.args.map((argument) => argument.value ?? argument.description).join(" "),
        );
      }
    });
    pageClient.on("Runtime.exceptionThrown", (event) => {
      diagnostics.page_exceptions.push(
        event.exceptionDetails?.exception?.description ?? event.exceptionDetails?.text,
      );
    });
    pageClient.on("Network.loadingFailed", (event) => {
      if (!event.canceled && !String(event.errorText).includes("ERR_ABORTED")) {
        diagnostics.failed_requests.push({
          error: event.errorText,
          requestId: event.requestId,
        });
      }
    });

    await Promise.all([
      pageClient.send("DOM.enable"),
      pageClient.send("HeapProfiler.enable"),
      pageClient.send("Memory.setPressureNotificationsSuppressed", { suppressed: true }),
      pageClient.send("Network.enable"),
      pageClient.send("Page.enable"),
      pageClient.send("Performance.enable", { timeDomain: "timeTicks" }),
      pageClient.send("Runtime.enable"),
    ]);
    await pageClient.send("Page.addScriptToEvaluateOnNewDocument", {
      source: `(${installPageInstrumentation.toString()})(${JSON.stringify(backendUrl)})`,
    });
    await pageClient.send("Network.setCacheDisabled", { cacheDisabled: true });

    for (let index = 0; index < options.initialWarmup; index += 1) {
      await measureNavigation(pageClient, frontendUrl);
    }
    const initialSamples = [];
    for (let index = 0; index < options.initialRepetitions; index += 1) {
      initialSamples.push(await measureNavigation(pageClient, frontendUrl));
    }
    await pageClient.send("Network.setCacheDisabled", { cacheDisabled: false });

    await clickWorkspace(pageClient, "Model");
    await requirePageMethod(pageClient, "clickCombobox", ["preset"], "preset combobox");
    await waitForExpression(
      pageClient,
      "window.__EMPEROR_BROWSER_PERFORMANCE__.options().length > 0",
      "preset option inventory",
    );
    const presetOptions = await pageMethod(pageClient, "options");
    await requirePageMethod(
      pageClient,
      "clickOption",
      [presetOptions[0], false],
      "preset dropdown close",
    );

    await selectBrowserPerformanceLogs(pageClient);

    for (let index = 0; index < options.sessionWarmup; index += 1) {
      const preset = presetOptions[(index + 1) % presetOptions.length];
      await runWorkspaceCycle(pageClient, preset);
    }
    const heapBeforeSession = await collectHeapCheckpoint(pageClient, "before_session");
    const sessionSamples = [];
    let heapMidSession = null;
    for (let index = 0; index < options.sessionRepetitions; index += 1) {
      const preset =
        presetOptions[(index + options.sessionWarmup + 1) % presetOptions.length];
      sessionSamples.push(await runWorkspaceCycle(pageClient, preset));
      if (index + 1 === Math.ceil(options.sessionRepetitions / 2)) {
        heapMidSession = await collectHeapCheckpoint(pageClient, "mid_session");
      }
    }
    const heapAfterSession = await collectHeapCheckpoint(pageClient, "after_session");

    const steadyStateSamples = [];
    for (let index = 0; index < options.steadyStateRepetitions; index += 1) {
      const preset =
        presetOptions[(index + options.sessionWarmup + 1) % presetOptions.length];
      steadyStateSamples.push(await runWorkspaceCycle(pageClient, preset));
    }
    const heapAfterSteadyState = await collectHeapCheckpoint(
      pageClient,
      "after_steady_state",
    );

    const trainingResult = await completeTrainingJob(pageClient);
    const importResult = await importLogs(
      pageClient,
      resolve(temporaryRoot, "browser-performance-import.zip"),
    );
    const heapAfterWorkflows = await collectHeapCheckpoint(
      pageClient,
      "after_training_and_import",
    );

    await prepareNeuronGraph(pageClient);
    for (let index = 0; index < options.webglWarmup; index += 1) {
      await measureWebglCycle(
        pageClient,
        options.frameRepetitions,
        `webgl_warmup_${index + 1}`,
      );
    }
    const webglSamples = [];
    for (let index = 0; index < options.webglRepetitions; index += 1) {
      webglSamples.push(
        await measureWebglCycle(
          pageClient,
          options.frameRepetitions,
          `webgl_${index + 1}`,
        ),
      );
    }
    const heapAfterWebgl = await collectHeapCheckpoint(pageClient, "after_webgl");

    const pageEnvironment = await evaluate(
      pageClient,
      `(() => ({
        deviceMemoryGiB: navigator.deviceMemory ?? null,
        devicePixelRatio: window.devicePixelRatio,
        hardwareConcurrency: navigator.hardwareConcurrency,
        language: navigator.language,
        screen: { height: screen.height, width: screen.width },
        userAgent: navigator.userAgent,
        viewport: { height: innerHeight, width: innerWidth },
      }))()`,
    );
    const browserVersion = await browserClient.send("Browser.getVersion");
    const systemInfo = await browserClient.send("SystemInfo.getInfo");
    const buildBudgetsResult = collectBuildPerformanceEvidence(
      resolve(FRONTEND_ROOT, ".next"),
    ).budgets;
    const apiEntries = await evaluate(
      pageClient,
      `performance.getEntriesByType("resource")
        .filter((entry) => entry.name.startsWith(${JSON.stringify(backendUrl)}))
        .map((entry) => ({
          duration: entry.duration,
          initiatorType: entry.initiatorType,
          name: entry.name,
          transferSize: entry.transferSize,
        }))`,
    );
    const apiSummary = summarizeApiEntries(apiEntries);

    const initialSummary = summarizeFields(initialSamples, [
      "cumulative_layout_shift",
      "dom_content_loaded_ms",
      "first_contentful_paint_ms",
      "first_paint_ms",
      "first_react_commit_ms",
      "heap_used_bytes",
      "layout_count",
      "layout_duration_ms",
      "load_event_ms",
      "long_task_count",
      "long_task_duration_ms",
      "react_commits",
      "recalc_style_count",
      "recalc_style_duration_ms",
      "resource_transfer_bytes",
      "script_duration_ms",
      "task_duration_ms",
      "workspace_ready_ms",
    ]);
    const sessionSummary = summarizeFields(sessionSamples, [
      "duration_ms",
      "layout_count",
      "layout_duration_ms",
      "long_task_count",
      "long_task_duration_ms",
      "react_commits",
      "recalc_style_count",
      "recalc_style_duration_ms",
      "script_duration_ms",
      "task_duration_ms",
    ]);
    const steadyStateSummary = summarizeFields(steadyStateSamples, [
      "duration_ms",
      "layout_count",
      "layout_duration_ms",
      "long_task_count",
      "long_task_duration_ms",
      "react_commits",
      "recalc_style_count",
      "recalc_style_duration_ms",
      "script_duration_ms",
      "task_duration_ms",
    ]);
    const frameIntervals = webglSamples.flatMap((sample) => sample.frame_intervals_ms);
    const totalContextsCreated = webglSamples.reduce(
      (total, sample) => total + sample.contexts_created,
      0,
    );
    const totalContextsLost = webglSamples.reduce(
      (total, sample) => total + sample.contexts_lost,
      0,
    );
    const retainedHeapGrowth = heapAfterWebgl.usedSize - heapBeforeSession.usedSize;
    const retainedHeapGrowthRatio =
      heapBeforeSession.usedSize === 0
        ? 0
        : retainedHeapGrowth / heapBeforeSession.usedSize;
    const sessionHeapGrowth = heapAfterSession.usedSize - heapBeforeSession.usedSize;
    const sessionHeapGrowthRatio =
      heapBeforeSession.usedSize === 0
        ? 0
        : sessionHeapGrowth / heapBeforeSession.usedSize;
    const steadyStateHeapGrowth =
      heapAfterSteadyState.usedSize - heapAfterSession.usedSize;
    const steadyStateHeapGrowthRatio =
      heapAfterSession.usedSize === 0
        ? 0
        : steadyStateHeapGrowth / heapAfterSession.usedSize;
    const thresholds = createBrowserPerformanceThresholds({
      browserWorkflowFailures:
        diagnostics.console_errors.length +
        diagnostics.page_exceptions.length +
        diagnostics.failed_requests.length,
      buildBudgets: buildBudgetsResult,
      connectedWebglCanvases: Math.max(
        ...webglSamples.map((sample) => sample.canvas_count_after_close),
      ),
      initialCumulativeLayoutShiftP95:
        initialSummary.cumulative_layout_shift.p95,
      initialTaskDurationP95: initialSummary.task_duration_ms.p95,
      initialWorkspaceReadyP95: initialSummary.workspace_ready_ms.p95,
      inspectionApiDurationP95:
        apiSummary["/inspect"]?.duration_ms.p95 ?? Number.POSITIVE_INFINITY,
      inspectionRequestCount: apiSummary["/inspect"]?.count ?? 0,
      logImportRequestCount: apiSummary["/logs/import"]?.count ?? 0,
      longSessionDurationP95: sessionSummary.duration_ms.p95,
      scalarApiDurationP95:
        apiSummary["/logs/scalars"]?.duration_ms.p95 ?? Number.POSITIVE_INFINITY,
      scalarRequestCount: apiSummary["/logs/scalars"]?.count ?? 0,
      sessionHeapGrowth,
      steadyStateHeapGrowth,
      trainingJobRequestCount: apiSummary["/training/jobs"]?.count ?? 0,
      webglContextsCreated: totalContextsCreated,
      webglContextsLost: totalContextsLost,
      webglFrameIntervalP95: summarize(frameIntervals).p95,
    });

    const result = createBrowserPerformanceEvidence({
      api: {
        entries: apiEntries,
        summary: apiSummary,
      },
      build: {
        build_id: (await readFile(resolve(FRONTEND_ROOT, ".next/BUILD_ID"), "utf8")).trim(),
        budgets: buildBudgetsResult,
        mode: "Next.js production",
      },
      conditions: {
        browser_cache: "disabled and cleared for initial-load samples; enabled later",
        frame_repetitions_per_webgl_sample: options.frameRepetitions,
        initial_repetitions: options.initialRepetitions,
        initial_warmup: options.initialWarmup,
        long_session_cycle: [
          "Model workspace preset change and graph load",
          "Training workspace visit",
          "Logs workspace scalar chart dialog open and close",
        ],
        post_session_workflows: [
          "Training Job submission and completion through the browser Interface",
          "log archive import through the browser Interface",
        ],
        requested_window_pixels: [1440, 1000],
        session_repetitions: options.sessionRepetitions,
        session_warmup: options.sessionWarmup,
        steady_state_repetitions: options.steadyStateRepetitions,
        storage_policy:
          "fresh disposable localStorage for each initial sample; retained later",
        viewport_css_pixels: [
          pageEnvironment.viewport.width,
          pageEnvironment.viewport.height,
        ],
        webgl_workflow:
          "open, animate with wheel input, measure requestAnimationFrame intervals, and close the 3D cluster dialog",
        webgl_repetitions: options.webglRepetitions,
        webgl_warmup: options.webglWarmup,
      },
      diagnostics,
      environment: {
        browser: browserVersion,
        chromium_command: options.chromium,
        cpu: {
          logical_count: os.cpus().length,
          model: os.cpus()[0]?.model ?? "unknown",
        },
        gpu: systemInfo.gpu,
        memory_bytes: os.totalmem(),
        node: process.version,
        operating_system: {
          arch: os.arch(),
          platform: os.platform(),
          release: os.release(),
          version: os.version(),
        },
        page: pageEnvironment,
      },
      initial_load: {
        samples: initialSamples,
        summary: initialSummary,
      },
      log_import: importResult,
      long_session: {
        heap: {
          checkpoints: [
            heapBeforeSession,
            heapMidSession,
            heapAfterSession,
            heapAfterSteadyState,
            heapAfterWorkflows,
            heapAfterWebgl,
          ].filter(Boolean),
          retained_growth_bytes: retainedHeapGrowth,
          retained_growth_ratio: retainedHeapGrowthRatio,
          session_growth_bytes: sessionHeapGrowth,
          session_growth_bytes_per_cycle:
            sessionHeapGrowth / options.sessionRepetitions,
          session_growth_ratio: sessionHeapGrowthRatio,
          steady_state_growth_bytes: steadyStateHeapGrowth,
          steady_state_growth_bytes_per_cycle:
            steadyStateHeapGrowth / options.steadyStateRepetitions,
          steady_state_growth_ratio: steadyStateHeapGrowthRatio,
        },
        samples: sessionSamples,
        steady_state_samples: steadyStateSamples,
        steady_state_summary: steadyStateSummary,
        summary: sessionSummary,
      },
      thresholds,
      training_job: trainingResult,
      webgl: {
        context_disposal: {
          contexts_created: totalContextsCreated,
          contexts_lost: totalContextsLost,
        },
        frame_interval_ms: summarize(frameIntervals),
        renderer: webglSamples[0]?.renderer ?? null,
        samples: webglSamples,
        vendor: webglSamples[0]?.vendor ?? null,
      },
    });
    const deterministicFailures = deterministicPerformanceFailures(thresholds);
    return { deterministicFailures, result };
  } finally {
    pageClient?.close();
    browserClient?.close();
    await Promise.all([
      stopProcess(chromium),
      stopProcess(frontend),
      stopProcess(backend),
    ]);
    await rm(temporaryRoot, { force: true, recursive: true });
  }
}

const options = parseArguments(process.argv.slice(2));
const { deterministicFailures, result } = await runBenchmark(options);
console.log(JSON.stringify(result, null, 2));
if (deterministicFailures.length > 0) {
  console.error(
    `Deterministic browser performance checks failed: ${deterministicFailures
      .map((entry) => entry.name)
      .join(", ")}`,
  );
  process.exitCode = 1;
}
