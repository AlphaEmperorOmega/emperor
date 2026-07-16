import { existsSync, readFileSync, readdirSync } from "node:fs";
import { relative, resolve } from "node:path";
import { runInNewContext } from "node:vm";
import { gzipSync } from "node:zlib";

const deferredModules = [
  {
    exportName: "WorkbenchModelWorkspace",
    label: "Model workspace",
    routeEntry: true,
    target: "@/features/workbench/components/workbench-model-workspace",
  },
  {
    exportName: "LogsWorkspaceProvider",
    label: "Logs workspace state",
    routeEntry: true,
    target: "@/features/workbench/providers/logs-workspace-provider",
  },
  {
    exportName: "TrainingExecutionProvider",
    label: "Training execution state",
    routeEntry: true,
    target: "@/features/workbench/providers/training-execution-provider",
  },
  {
    exportName: "TrainingPanel",
    label: "Training workspace",
    routeEntry: true,
    target: "@/features/workbench/components/training-panel",
  },
  {
    exportName: "ConnectedLogsSidebarPanel",
    label: "Logs workspace view",
    routeEntry: true,
    target: "@/features/workbench/components/logs/logs-workspace",
  },
  {
    exportName: "ConnectedLogRunDetailsPanel",
    label: "Log run details",
    routeEntry: true,
    target: "@/features/workbench/components/logs/log-run-details-panel",
  },
  {
    exportName: "WorkbenchWorkspaceOverlays",
    label: "Workbench overlays",
    routeEntry: true,
    target: "@/features/workbench/components/workbench-overlays",
  },
  {
    exportName: "AppHeaderRuntimeDefaultsControls",
    label: "Runtime Defaults header controls",
    routeEntry: true,
    target:
      "@/features/workbench/components/screen/app-header-runtime-defaults-controls",
  },
  {
    exportName: "FullConfigDialog",
    label: "Full Config dialog",
    routeEntry: true,
    target: "@/features/workbench/components/config/full-config-dialog",
  },
  {
    exportName: "GraphPreviewPanel",
    label: "React Flow graph canvas",
    routeEntry: false,
    target: "@/features/workbench/components/screen/graph-canvas",
  },
  {
    exportName: "SelectedNodeDetails",
    label: "Selected node details",
    routeEntry: false,
    target: "@/features/workbench/components/graph/selected-node-details",
  },
  {
    exportName: "layoutGraph",
    label: "Dagre graph layout",
    routeEntry: false,
    target: "@/lib/graph/layout",
  },
  {
    exportName: "LazyLogScalarChart",
    label: "ECharts scalar charts",
    routeEntry: false,
    target: "@/features/workbench/components/logs/log-scalar-chart",
  },
  {
    exportName: "MonitorChartsModal",
    label: "Monitor charts",
    routeEntry: false,
    target: "@/features/workbench/components/monitor/monitor-charts-modal",
  },
  {
    exportName: "fetchLogRuns",
    label: "Logs query client",
    routeEntry: false,
    target: "@/lib/api/logs",
  },
  {
    exportName: "fetchPresets",
    label: "Model Package metadata client",
    routeEntry: false,
    target: "@/lib/api/models",
  },
  {
    exportName: "fetchConfigSnapshots",
    label: "Config Snapshot query client",
    routeEntry: false,
    target: "@/lib/api/config-snapshots",
  },
  {
    exportName: "fetchMonitorParameterStatus",
    label: "Monitor Data query client",
    routeEntry: false,
    target: "@/lib/api/monitor-data",
  },
  {
    exportName: "inspectModel",
    label: "Model Inspection query client",
    routeEntry: false,
    target: "@/lib/api/inspection",
  },
  {
    exportName: "fetchTrainingJob",
    label: "Training Job polling client",
    routeEntry: false,
    target: "@/lib/api/training-jobs",
  },
  {
    exportName: "NeuronCluster3DScene",
    label: "Three.js neuron scene",
    routeEntry: false,
    target: "@/features/workbench/components/graph/neuron-cluster-3d-scene",
  },
  {
    exportName: "ApiConnectionDialog",
    label: "API Connection dialog",
    routeEntry: false,
    target: "@/features/workbench/components/screen/api-connection-dialog",
  },
  {
    exportName: "ImportLogsDialog",
    label: "Import Logs dialog",
    routeEntry: false,
    target: "@/features/workbench/components/screen/import-logs-dialog",
  },
].map(Object.freeze);

const scalarChunkAllowedOwners = [
  "@/features/workbench/components/logs/log-scalar-chart",
  "@/features/workbench/components/monitor/monitor-charts-modal",
];

export const PERFORMANCE_EVIDENCE_POLICY = Object.freeze({
  budgets: Object.freeze({
    firstLoadBytes: 210_000,
    routeSpecificBytes: 98_000,
  }),
  requiredMeasured: Object.freeze({
    firstLoadBytes: 205_000,
    routeSpecificBytes: 93_000,
  }),
  deferredModules: Object.freeze(deferredModules),
  scalarChunkAllowedOwners: Object.freeze(scalarChunkAllowedOwners),
  schemaVersion: 1,
});

function invariant(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function readManifest(nextDirectory, name) {
  try {
    return JSON.parse(readFileSync(resolve(nextDirectory, name), "utf8"));
  } catch (error) {
    throw new Error(`Unable to read .next/${name}. Run npm run build first.`, {
      cause: error,
    });
  }
}

function gzipSize(nextDirectory, relativePath) {
  const contents = readFileSync(resolve(nextDirectory, relativePath));
  return gzipSync(contents).byteLength;
}

function sumGzipSize(nextDirectory, files) {
  return [...new Set(files)].reduce(
    (total, file) => total + gzipSize(nextDirectory, file),
    0,
  );
}

function normalizeClientChunkPath(file) {
  return file.replace(/^\/?_next\//, "").replace(/^\.next\//, "");
}

function createBuildBudgets(nextDirectory, pageFiles, routeSpecificFiles) {
  return {
    first_load: {
      budget_bytes: PERFORMANCE_EVIDENCE_POLICY.budgets.firstLoadBytes,
      gzip_bytes: sumGzipSize(nextDirectory, pageFiles),
    },
    route_specific: {
      budget_bytes: PERFORMANCE_EVIDENCE_POLICY.budgets.routeSpecificBytes,
      gzip_bytes: sumGzipSize(nextDirectory, routeSpecificFiles),
    },
  };
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function readTurbopackJavaScriptChunks(nextDirectory) {
  const chunksDirectory = resolve(nextDirectory, "static/chunks");
  invariant(
    existsSync(chunksDirectory),
    "The production Turbopack JavaScript chunk directory is missing.",
  );
  const chunks = new Map();

  function visit(directory) {
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      const absolutePath = resolve(directory, entry.name);
      if (entry.isDirectory()) {
        visit(absolutePath);
      } else if (entry.isFile() && entry.name.endsWith(".js")) {
        const relativePath = relative(nextDirectory, absolutePath).replaceAll(
          "\\",
          "/",
        );
        chunks.set(relativePath, readFileSync(absolutePath, "utf8"));
      }
    }
  }

  visit(chunksDirectory);
  invariant(
    chunks.size > 0,
    "The production Turbopack JavaScript chunk directory is empty.",
  );
  return chunks;
}

function findTurbopackLoaderIds(chunks, exportName) {
  const pattern = new RegExp(
    `\\.A\\((\\d+)\\)\\.then\\(\\(?([A-Za-z_$][\\w$]*)\\)?=>\\2\\.${escapeRegExp(exportName)}\\)`,
    "g",
  );
  const loaderIds = new Set();
  for (const source of chunks.values()) {
    for (const match of source.matchAll(pattern)) {
      loaderIds.add(match[1]);
    }
  }
  return [...loaderIds];
}

function findTurbopackAsyncLoads(chunks, loaderId) {
  const registration = new RegExp(
    `(?:^|[,\\[])${escapeRegExp(loaderId)},[A-Za-z_$][\\w$]*=>\\{`,
    "g",
  );
  const loads = new Map();

  for (const source of chunks.values()) {
    for (const match of source.matchAll(registration)) {
      const registrationBodyStart = match.index + match[0].length;
      const registrationTail = source.slice(
        registrationBodyStart,
        registrationBodyStart + 2_000,
      );
      const nextRegistration =
        /},\d+,[A-Za-z_$][\w$]*=>\{/.exec(registrationTail);
      const registrationEnd =
        nextRegistration === null
          ? registrationBodyStart + registrationTail.length
          : registrationBodyStart + nextRegistration.index + 1;
      const registrationSource = source.slice(match.index, registrationEnd);
      const filesMatch =
        /Promise\.all\((\[[^\]]*\])\.map\([A-Za-z_$][\w$]*=>[A-Za-z_$][\w$]*\.l\([A-Za-z_$][\w$]*\)\)\)/.exec(
          registrationSource,
        );
      const moduleMatch =
        /\.then\(\(\)=>[A-Za-z_$][\w$]*\((\d+)\)\)/.exec(
          registrationSource,
        );
      if (!moduleMatch) {
        continue;
      }

      let files = [];
      if (filesMatch) {
        try {
          files = JSON.parse(filesMatch[1]);
        } catch {
          continue;
        }
        if (
          !Array.isArray(files) ||
          files.some((file) => typeof file !== "string")
        ) {
          continue;
        }
      }

      const normalizedFiles = [
        ...new Set(
          files
            .filter((file) => file.endsWith(".js"))
            .map(normalizeClientChunkPath),
        ),
      ];
      const moduleId = moduleMatch[1];
      loads.set(
        `${moduleId}\0${normalizedFiles.slice().sort().join("\0")}`,
        {
          files: normalizedFiles,
          moduleId,
        },
      );
    }
  }

  return [...loads.values()];
}

function collectNamedTurbopackDeferredModules(
  nextDirectory,
  chunks,
  initialFiles,
) {
  return PERFORMANCE_EVIDENCE_POLICY.deferredModules.map(
    ({ exportName, label, routeEntry, target }) => {
      const loaderIds = findTurbopackLoaderIds(chunks, exportName);
      invariant(
        loaderIds.length > 0,
        `${label} is no longer represented by a named Turbopack dynamic import (${exportName}).`,
      );
      invariant(
        loaderIds.length === 1,
        `${label} is represented by multiple Turbopack loader identities: ${loaderIds.join(", ")}.`,
      );

      const loads = findTurbopackAsyncLoads(chunks, loaderIds[0]);
      invariant(
        loads.length > 0,
        `${label} has no emitted Turbopack JavaScript chunk.`,
      );
      const moduleIds = [...new Set(loads.map(({ moduleId }) => moduleId))];
      invariant(
        moduleIds.length === 1,
        `${label} resolves to multiple Turbopack module identities: ${moduleIds.join(", ")}.`,
      );
      const files = [
        ...new Set(loads.flatMap((load) => load.files)),
      ].sort();
      invariant(
        files.length > 0,
        `${label} resolves from an initial Turbopack module instead of an emitted deferred JavaScript chunk.`,
      );

      const initialOverlap = files.filter((file) => initialFiles.has(file));
      invariant(
        initialOverlap.length === 0,
        `${label} leaked into the initial /page chunks: ${initialOverlap.join(", ")}.`,
      );

      return {
        bytes: sumGzipSize(nextDirectory, files),
        files,
        label,
        loaderId: loaderIds[0],
        moduleIds,
        routeEntry,
        target,
      };
    },
  );
}

function readTurbopackClientReferenceManifest(nextDirectory) {
  const relativePath = "server/app/page_client-reference-manifest.js";
  const manifestPath = resolve(nextDirectory, relativePath);
  if (!existsSync(manifestPath)) {
    return null;
  }

  try {
    const context = {};
    runInNewContext(readFileSync(manifestPath, "utf8"), context, {
      timeout: 1_000,
    });
    return context.__RSC_MANIFEST?.["/page"] ?? null;
  } catch (error) {
    throw new Error(
      `Unable to read .next/${relativePath}. Run npm run build first.`,
      { cause: error },
    );
  }
}

function collectTurbopackBuildPerformanceEvidence(
  nextDirectory,
  clientReferenceManifest,
) {
  const entryJSFiles = clientReferenceManifest?.entryJSFiles;
  invariant(
    entryJSFiles &&
      typeof entryJSFiles === "object" &&
      !Array.isArray(entryJSFiles),
    "The production /page client reference manifest has no JavaScript entries.",
  );
  const entries = Object.entries(entryJSFiles);
  const pageEntryKeys = new Set(
    entries
      .filter(([entry]) => /(?:^|\/)app\/page$/.test(entry))
      .map(([entry]) => entry),
  );
  invariant(
    pageEntryKeys.size > 0,
    "The production /page JavaScript entry was not found.",
  );

  const pageFiles = [
    ...new Set(
      entries
        .flatMap(([, files]) => (Array.isArray(files) ? files : []))
        .filter((file) => typeof file === "string" && file.endsWith(".js"))
        .map(normalizeClientChunkPath),
    ),
  ];
  invariant(pageFiles.length > 0, "The production /page JavaScript entry is empty.");
  const otherEntryFiles = new Set(
    entries
      .filter(([entry]) => !pageEntryKeys.has(entry))
      .flatMap(([, files]) => (Array.isArray(files) ? files : []))
      .filter((file) => typeof file === "string" && file.endsWith(".js"))
      .map(normalizeClientChunkPath),
  );
  const routeSpecificFiles = pageFiles.filter(
    (file) => !otherEntryFiles.has(file),
  );

  const loadableManifest = readManifest(
    nextDirectory,
    "server/app/page/react-loadable-manifest.json",
  );
  invariant(
    loadableManifest &&
      typeof loadableManifest === "object" &&
      !Array.isArray(loadableManifest),
    "The production /page loadable manifest is not an object.",
  );
  const loadableEntries = Object.entries(loadableManifest);
  invariant(
    loadableEntries.length > 0,
    "The production /page loadable manifest has no dynamic imports.",
  );
  const initialFiles = new Set(pageFiles);
  const manifestModules = new Map();
  for (const [moduleId, value] of loadableEntries) {
    const files = Array.isArray(value?.files)
      ? [
          ...new Set(
            value.files
              .filter(
                (file) => typeof file === "string" && file.endsWith(".js"),
              )
              .map(normalizeClientChunkPath),
          ),
        ]
      : [];
    invariant(
      files.length > 0,
      `Turbopack dynamic module ${moduleId} has no emitted JavaScript chunk.`,
    );
    const initialOverlap = files.filter((file) => initialFiles.has(file));
    invariant(
      initialOverlap.length === 0,
      `Turbopack dynamic module ${moduleId} leaked into the initial /page chunks: ${initialOverlap.join(", ")}.`,
    );
    manifestModules.set(moduleId, files);
  }

  const chunks = readTurbopackJavaScriptChunks(nextDirectory);
  const namedDeferred = collectNamedTurbopackDeferredModules(
    nextDirectory,
    chunks,
    initialFiles,
  );
  for (const [moduleId, files] of manifestModules) {
    const owner = namedDeferred.find(({ moduleIds }) =>
      moduleIds.includes(moduleId),
    );
    invariant(
      owner,
      `Turbopack dynamic module ${moduleId} has no named deferred Module policy.`,
    );
    const unaccountedFiles = files.filter(
      (file) => !owner.files.includes(file),
    );
    invariant(
      unaccountedFiles.length === 0,
      `${owner.label} loadable manifest files are not represented by its Turbopack loader: ${unaccountedFiles.join(", ")}.`,
    );
  }

  for (const deferredModule of namedDeferred.filter(
    ({ routeEntry }) => routeEntry,
  )) {
    invariant(
      deferredModule.moduleIds.some((moduleId) =>
        manifestModules.has(moduleId),
      ),
      `${deferredModule.label} is missing from the production /page loadable manifest.`,
    );
  }

  const scalarResult = namedDeferred.find(
    ({ label }) => label === "ECharts scalar charts",
  );
  invariant(scalarResult, "The scalar chart deferred evidence is missing.");
  const scalarFiles = new Set(scalarResult.files);
  const scalarChunkOwners = namedDeferred.filter(({ files }) =>
    files.some((file) => scalarFiles.has(file)),
  );
  invariant(
    scalarChunkOwners.every(({ target }) =>
      PERFORMANCE_EVIDENCE_POLICY.scalarChunkAllowedOwners.includes(target),
    ),
    `ECharts chunks have unexpected owners: ${scalarChunkOwners.map(({ target }) => target).join(", ")}.`,
  );

  const deferred = namedDeferred.map(
    ({ bytes, files, label, target }) => ({
      bytes,
      files,
      label,
      target,
    }),
  );

  return {
    budgets: createBuildBudgets(nextDirectory, pageFiles, routeSpecificFiles),
    deferred,
    manifestKind: "next16-turbopack",
    pageFiles,
    routeSpecificFiles,
  };
}

export function formatPerformanceKilobytes(bytes) {
  return `${(bytes / 1_000).toFixed(1)} kB`;
}

export function collectBuildPerformanceEvidence(nextDirectory) {
  const turbopackManifest =
    readTurbopackClientReferenceManifest(nextDirectory);
  if (turbopackManifest) {
    return collectTurbopackBuildPerformanceEvidence(
      nextDirectory,
      turbopackManifest,
    );
  }

  const appBuildManifest = readManifest(nextDirectory, "app-build-manifest.json");
  const loadableManifest = readManifest(
    nextDirectory,
    "react-loadable-manifest.json",
  );
  const pages = appBuildManifest?.pages;
  invariant(
    pages && typeof pages === "object" && !Array.isArray(pages),
    "The production app build manifest has no pages map.",
  );

  const pageFiles = pages["/page"]?.filter(
    (file) => typeof file === "string" && file.endsWith(".js"),
  );
  invariant(pageFiles?.length, "The production /page JavaScript entry was not found.");

  const otherEntryFiles = new Set(
    Object.entries(pages)
      .filter(([entry]) => entry !== "/page")
      .flatMap(([, files]) => (Array.isArray(files) ? files : []))
      .filter((file) => typeof file === "string" && file.endsWith(".js")),
  );
  const routeSpecificFiles = pageFiles.filter(
    (file) => !otherEntryFiles.has(file),
  );
  const budgets = createBuildBudgets(
    nextDirectory,
    pageFiles,
    routeSpecificFiles,
  );

  invariant(
    loadableManifest &&
      typeof loadableManifest === "object" &&
      !Array.isArray(loadableManifest),
    "The production loadable manifest is not an object.",
  );
  const loadableEntries = Object.entries(loadableManifest);
  const initialFiles = new Set(pageFiles);
  const deferred = PERFORMANCE_EVIDENCE_POLICY.deferredModules.map(
    ({ label, target }) => {
      const entry = loadableEntries.find(([key]) =>
        key.endsWith(` -> ${target}`),
      );
      invariant(entry, `${label} is no longer represented by a dynamic import.`);

      const files = Array.isArray(entry[1]?.files)
        ? entry[1].files.filter(
            (file) => typeof file === "string" && file.endsWith(".js"),
          )
        : [];
      invariant(files.length, `${label} has no emitted JavaScript chunk.`);

      const initialOverlap = files.filter((file) => initialFiles.has(file));
      invariant(
        initialOverlap.length === 0,
        `${label} leaked into the initial /page chunks: ${initialOverlap.join(", ")}.`,
      );

      return {
        bytes: sumGzipSize(nextDirectory, files),
        files,
        label,
        target,
      };
    },
  );

  const scalarResult = deferred.find(
    ({ label }) => label === "ECharts scalar charts",
  );
  invariant(scalarResult, "The scalar chart deferred evidence is missing.");
  const scalarFiles = new Set(scalarResult.files);
  const scalarChunkOwners = loadableEntries
    .filter(([, value]) =>
      Array.isArray(value?.files)
        ? value.files.some((file) => scalarFiles.has(file))
        : false,
    )
    .map(([key]) => key);
  invariant(
    scalarChunkOwners.every((owner) =>
      PERFORMANCE_EVIDENCE_POLICY.scalarChunkAllowedOwners.some((target) =>
        owner.endsWith(` -> ${target}`),
      ),
    ),
    `ECharts chunks have unexpected owners: ${scalarChunkOwners.join(", ")}.`,
  );

  return {
    budgets,
    deferred,
    manifestKind: "next15-webpack",
    pageFiles: [...new Set(pageFiles)],
    routeSpecificFiles: [...new Set(routeSpecificFiles)],
  };
}

export function assertBuildPerformanceBudgets(evidence) {
  const { first_load: firstLoad, route_specific: routeSpecific } =
    evidence.budgets;
  invariant(
    firstLoad.gzip_bytes <= firstLoad.budget_bytes,
    `First-load JavaScript is ${formatPerformanceKilobytes(firstLoad.gzip_bytes)}; budget is ${formatPerformanceKilobytes(firstLoad.budget_bytes)}.`,
  );
  invariant(
    routeSpecific.gzip_bytes <= routeSpecific.budget_bytes,
    `Route-specific JavaScript is ${formatPerformanceKilobytes(routeSpecific.gzip_bytes)}; budget is ${formatPerformanceKilobytes(routeSpecific.budget_bytes)}.`,
  );
  invariant(
    firstLoad.gzip_bytes <=
      PERFORMANCE_EVIDENCE_POLICY.requiredMeasured.firstLoadBytes,
    `First-load JavaScript is ${formatPerformanceKilobytes(firstLoad.gzip_bytes)}; required headroom is ${formatPerformanceKilobytes(PERFORMANCE_EVIDENCE_POLICY.requiredMeasured.firstLoadBytes)} within the ${formatPerformanceKilobytes(firstLoad.budget_bytes)} budget.`,
  );
  invariant(
    routeSpecific.gzip_bytes <=
      PERFORMANCE_EVIDENCE_POLICY.requiredMeasured.routeSpecificBytes,
    `Route-specific JavaScript is ${formatPerformanceKilobytes(routeSpecific.gzip_bytes)}; required headroom is ${formatPerformanceKilobytes(PERFORMANCE_EVIDENCE_POLICY.requiredMeasured.routeSpecificBytes)} within the ${formatPerformanceKilobytes(routeSpecific.budget_bytes)} budget.`,
  );
}

const thresholdKinds = new Map([
  ...[
    "first-load gzip budget",
    "route-specific gzip budget",
    "browser workflow failures",
    "WebGL contexts disposed",
    "connected WebGL canvases after close",
    "graph inspection requests exercised",
    "scalar chart requests exercised",
    "Training Job requests exercised",
    "log import requests exercised",
  ].map((name) => [name, "deterministic"]),
  ...[
    "initial workspace-ready p95",
    "initial main-thread task p95",
    "long-session cycle p95",
    "graph-cache fill retained heap growth",
    "cache-saturated retained heap growth",
    "WebGL frame interval p95",
    "initial cumulative layout shift p95",
    "log scalar API p95",
    "graph inspection API p95",
  ].map((name) => [name, "informational"]),
]);

function threshold(name, actual, limit, unit, comparator = "maximum") {
  const kind = thresholdKinds.get(name);
  invariant(kind, `Threshold ${name} is not part of the performance policy.`);
  const passed = comparator === "minimum" ? actual >= limit : actual <= limit;
  return { actual, comparator, kind, limit, name, passed, unit };
}

export function createBrowserPerformanceThresholds(metrics) {
  return [
    threshold(
      "first-load gzip budget",
      metrics.buildBudgets.first_load.gzip_bytes,
      PERFORMANCE_EVIDENCE_POLICY.budgets.firstLoadBytes,
      "bytes",
    ),
    threshold(
      "route-specific gzip budget",
      metrics.buildBudgets.route_specific.gzip_bytes,
      PERFORMANCE_EVIDENCE_POLICY.budgets.routeSpecificBytes,
      "bytes",
    ),
    threshold(
      "browser workflow failures",
      metrics.browserWorkflowFailures,
      0,
      "count",
    ),
    threshold(
      "WebGL contexts disposed",
      metrics.webglContextsLost,
      metrics.webglContextsCreated,
      "contexts",
      "minimum",
    ),
    threshold(
      "connected WebGL canvases after close",
      metrics.connectedWebglCanvases,
      0,
      "canvases",
    ),
    threshold(
      "graph inspection requests exercised",
      metrics.inspectionRequestCount,
      1,
      "requests",
      "minimum",
    ),
    threshold(
      "scalar chart requests exercised",
      metrics.scalarRequestCount,
      1,
      "requests",
      "minimum",
    ),
    threshold(
      "Training Job requests exercised",
      metrics.trainingJobRequestCount,
      1,
      "requests",
      "minimum",
    ),
    threshold(
      "log import requests exercised",
      metrics.logImportRequestCount,
      1,
      "requests",
      "minimum",
    ),
    threshold(
      "initial workspace-ready p95",
      metrics.initialWorkspaceReadyP95,
      5_000,
      "ms",
    ),
    threshold(
      "initial main-thread task p95",
      metrics.initialTaskDurationP95,
      1_000,
      "ms",
    ),
    threshold(
      "long-session cycle p95",
      metrics.longSessionDurationP95,
      3_000,
      "ms",
    ),
    threshold(
      "graph-cache fill retained heap growth",
      metrics.sessionHeapGrowth,
      20_000_000,
      "bytes",
    ),
    threshold(
      "cache-saturated retained heap growth",
      metrics.steadyStateHeapGrowth,
      5_000_000,
      "bytes",
    ),
    threshold(
      "WebGL frame interval p95",
      metrics.webglFrameIntervalP95,
      33.4,
      "ms",
    ),
    threshold(
      "initial cumulative layout shift p95",
      metrics.initialCumulativeLayoutShiftP95,
      0.1,
      "score",
    ),
    threshold(
      "log scalar API p95",
      metrics.scalarApiDurationP95,
      1_000,
      "ms",
    ),
    threshold(
      "graph inspection API p95",
      metrics.inspectionApiDurationP95,
      2_000,
      "ms",
    ),
  ];
}

export function deterministicPerformanceFailures(thresholds) {
  return thresholds.filter(
    (entry) => entry.kind === "deterministic" && !entry.passed,
  );
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function record(value, path) {
  invariant(isRecord(value), `${path} must be an object.`);
  return value;
}

function array(value, path) {
  invariant(Array.isArray(value), `${path} must be an array.`);
  return value;
}

function string(value, path) {
  invariant(typeof value === "string", `${path} must be a string.`);
}

function nullableString(value, path) {
  invariant(value === null || typeof value === "string", `${path} is invalid.`);
}

function finiteNumber(value, path) {
  invariant(
    typeof value === "number" && Number.isFinite(value),
    `${path} must be a finite number.`,
  );
}

function validateNumberArray(value, path) {
  array(value, path).forEach((entry, index) =>
    finiteNumber(entry, `${path}[${index}]`),
  );
}

function validateStringArray(value, path) {
  array(value, path).forEach((entry, index) =>
    string(entry, `${path}[${index}]`),
  );
}

function validateNumberRecord(value, path) {
  for (const [key, entry] of Object.entries(record(value, path))) {
    finiteNumber(entry, `${path}.${key}`);
  }
}

function validateDistribution(value, path) {
  const distribution = record(value, path);
  for (const key of [
    "coefficient_of_variation",
    "max",
    "mean",
    "median",
    "min",
    "p95",
    "stdev",
  ]) {
    finiteNumber(distribution[key], `${path}.${key}`);
  }
}

function validateDistributionRecord(value, path) {
  for (const [key, distribution] of Object.entries(record(value, path))) {
    validateDistribution(distribution, `${path}.${key}`);
  }
}

function validateOperation(value, path) {
  const operation = record(value, path);
  string(operation.label, `${path}.label`);
  for (const key of [
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
  ]) {
    finiteNumber(operation[key], `${path}.${key}`);
  }
}

function validateBudget(value, name, expectedBudget) {
  record(value, `${name} build budget`);
  invariant(
    value.budget_bytes === expectedBudget,
    `${name} build budget does not match the current performance policy.`,
  );
  invariant(
    typeof value.gzip_bytes === "number" && Number.isFinite(value.gzip_bytes),
    `${name} build budget must record finite gzip bytes.`,
  );
}

function validateThreshold(value, index) {
  record(value, `Threshold ${index}`);
  invariant(typeof value.name === "string", `Threshold ${index} needs a name.`);
  invariant(typeof value.unit === "string", `Threshold ${index} needs a unit.`);
  const expectedKind = thresholdKinds.get(value.name);
  invariant(
    expectedKind && value.kind === expectedKind,
    `Threshold ${index} has a kind outside the performance policy.`,
  );
  invariant(
    value.comparator === "minimum" || value.comparator === "maximum",
    `Threshold ${index} has an invalid comparator.`,
  );
  invariant(
    typeof value.actual === "number" && !Number.isNaN(value.actual),
    `Threshold ${index} needs a numeric actual value.`,
  );
  invariant(
    typeof value.limit === "number" && !Number.isNaN(value.limit),
    `Threshold ${index} needs a numeric limit.`,
  );
  const expectedPassed =
    value.comparator === "minimum"
      ? value.actual >= value.limit
      : value.actual <= value.limit;
  invariant(
    value.passed === expectedPassed,
    `Threshold ${index} has an inconsistent passed value.`,
  );
}

export function validateBrowserPerformanceEvidence(value) {
  const evidence = record(value, "Browser performance evidence");
  invariant(
    evidence.schema_version === PERFORMANCE_EVIDENCE_POLICY.schemaVersion,
    `Browser performance evidence must use schema version ${PERFORMANCE_EVIDENCE_POLICY.schemaVersion}.`,
  );

  const api = record(evidence.api, "api");
  array(api.entries, "api.entries");
  for (const [path, summaryValue] of Object.entries(
    record(api.summary, "api.summary"),
  )) {
    const summary = record(summaryValue, `api.summary.${path}`);
    finiteNumber(summary.count, `api.summary.${path}.count`);
    validateDistribution(
      summary.duration_ms,
      `api.summary.${path}.duration_ms`,
    );
    finiteNumber(
      summary.transfer_bytes,
      `api.summary.${path}.transfer_bytes`,
    );
  }

  const build = record(evidence.build, "build");
  string(build.build_id, "build.build_id");
  string(build.mode, "build.mode");
  const budgets = record(build.budgets, "build.budgets");
  validateBudget(
    budgets.first_load,
    "First-load",
    PERFORMANCE_EVIDENCE_POLICY.budgets.firstLoadBytes,
  );
  validateBudget(
    budgets.route_specific,
    "Route-specific",
    PERFORMANCE_EVIDENCE_POLICY.budgets.routeSpecificBytes,
  );

  const conditions = record(evidence.conditions, "conditions");
  for (const key of [
    "frame_repetitions_per_webgl_sample",
    "initial_repetitions",
    "initial_warmup",
    "session_repetitions",
    "session_warmup",
    "steady_state_repetitions",
    "webgl_repetitions",
    "webgl_warmup",
  ]) {
    finiteNumber(conditions[key], `conditions.${key}`);
  }
  for (const key of ["browser_cache", "storage_policy", "webgl_workflow"]) {
    string(conditions[key], `conditions.${key}`);
  }
  for (const key of ["long_session_cycle", "post_session_workflows"]) {
    validateStringArray(conditions[key], `conditions.${key}`);
  }
  for (const key of ["requested_window_pixels", "viewport_css_pixels"]) {
    validateNumberArray(conditions[key], `conditions.${key}`);
  }

  const diagnostics = record(evidence.diagnostics, "diagnostics");
  for (const key of ["console_errors", "failed_requests", "page_exceptions"]) {
    array(diagnostics[key], `diagnostics.${key}`);
  }

  const environment = record(evidence.environment, "environment");
  string(record(environment.browser, "environment.browser").product, "environment.browser.product");
  string(environment.chromium_command, "environment.chromium_command");
  const cpu = record(environment.cpu, "environment.cpu");
  finiteNumber(cpu.logical_count, "environment.cpu.logical_count");
  string(cpu.model, "environment.cpu.model");
  const gpu = record(environment.gpu, "environment.gpu");
  array(gpu.devices, "environment.gpu.devices").forEach((device, index) =>
    string(
      record(device, `environment.gpu.devices[${index}]`).deviceString,
      `environment.gpu.devices[${index}].deviceString`,
    ),
  );
  finiteNumber(environment.memory_bytes, "environment.memory_bytes");
  string(environment.node, "environment.node");
  const operatingSystem = record(
    environment.operating_system,
    "environment.operating_system",
  );
  for (const key of ["arch", "platform", "release", "version"]) {
    string(operatingSystem[key], `environment.operating_system.${key}`);
  }
  const page = record(environment.page, "environment.page");
  invariant(
    page.deviceMemoryGiB === null ||
      (typeof page.deviceMemoryGiB === "number" &&
        Number.isFinite(page.deviceMemoryGiB)),
    "environment.page.deviceMemoryGiB is invalid.",
  );
  for (const key of [
    "devicePixelRatio",
    "hardwareConcurrency",
  ]) {
    finiteNumber(page[key], `environment.page.${key}`);
  }
  for (const key of ["language", "userAgent"]) {
    string(page[key], `environment.page.${key}`);
  }
  for (const key of ["screen", "viewport"]) {
    const dimensions = record(page[key], `environment.page.${key}`);
    finiteNumber(dimensions.height, `environment.page.${key}.height`);
    finiteNumber(dimensions.width, `environment.page.${key}.width`);
  }

  const initialLoad = record(evidence.initial_load, "initial_load");
  array(initialLoad.samples, "initial_load.samples").forEach((sample, index) =>
    record(sample, `initial_load.samples[${index}]`),
  );
  validateDistributionRecord(initialLoad.summary, "initial_load.summary");

  validateOperation(evidence.log_import, "log_import");
  validateOperation(evidence.training_job, "training_job");

  const longSession = record(evidence.long_session, "long_session");
  const heap = record(longSession.heap, "long_session.heap");
  array(heap.checkpoints, "long_session.heap.checkpoints").forEach(
    (checkpointValue, index) => {
      const checkpoint = record(
        checkpointValue,
        `long_session.heap.checkpoints[${index}]`,
      );
      string(checkpoint.label, `long_session.heap.checkpoints[${index}].label`);
      for (const key of [
        "documents",
        "jsEventListeners",
        "nodes",
        "usedSize",
      ]) {
        finiteNumber(
          checkpoint[key],
          `long_session.heap.checkpoints[${index}].${key}`,
        );
      }
    },
  );
  for (const key of [
    "retained_growth_bytes",
    "retained_growth_ratio",
    "session_growth_bytes",
    "session_growth_bytes_per_cycle",
    "session_growth_ratio",
    "steady_state_growth_bytes",
    "steady_state_growth_bytes_per_cycle",
    "steady_state_growth_ratio",
  ]) {
    finiteNumber(heap[key], `long_session.heap.${key}`);
  }
  for (const key of ["samples", "steady_state_samples"]) {
    array(longSession[key], `long_session.${key}`).forEach((sample, index) =>
      validateOperation(sample, `long_session.${key}[${index}]`),
    );
  }
  validateDistributionRecord(
    longSession.steady_state_summary,
    "long_session.steady_state_summary",
  );
  validateDistributionRecord(longSession.summary, "long_session.summary");

  const webgl = record(evidence.webgl, "webgl");
  const contextDisposal = record(
    webgl.context_disposal,
    "webgl.context_disposal",
  );
  finiteNumber(
    contextDisposal.contexts_created,
    "webgl.context_disposal.contexts_created",
  );
  finiteNumber(
    contextDisposal.contexts_lost,
    "webgl.context_disposal.contexts_lost",
  );
  validateDistribution(webgl.frame_interval_ms, "webgl.frame_interval_ms");
  nullableString(webgl.renderer, "webgl.renderer");
  nullableString(webgl.vendor, "webgl.vendor");
  array(webgl.samples, "webgl.samples").forEach((sampleValue, index) => {
    const path = `webgl.samples[${index}]`;
    const sample = record(sampleValue, path);
    validateOperation(sample, path);
    for (const key of [
      "canvas_count_after_close",
      "contexts_created",
      "contexts_lost",
    ]) {
      finiteNumber(sample[key], `${path}.${key}`);
    }
    validateNumberArray(sample.frame_intervals_ms, `${path}.frame_intervals_ms`);
    validateNumberRecord(sample.resource_created, `${path}.resource_created`);
    validateNumberRecord(sample.resource_deleted, `${path}.resource_deleted`);
    nullableString(sample.renderer, `${path}.renderer`);
    nullableString(sample.vendor, `${path}.vendor`);
  });

  const thresholds = array(evidence.thresholds, "thresholds");
  thresholds.forEach(validateThreshold);
  const thresholdNames = new Set(thresholds.map((entry) => entry.name));
  invariant(
    thresholds.length === thresholdKinds.size &&
      thresholdNames.size === thresholdKinds.size &&
      [...thresholdKinds.keys()].every((name) => thresholdNames.has(name)),
    "Evidence thresholds do not match the performance policy.",
  );
  invariant(
    thresholds.some(
      (entry) =>
        entry.name === "first-load gzip budget" &&
        entry.actual === budgets.first_load.gzip_bytes &&
        entry.limit === budgets.first_load.budget_bytes,
    ),
    "The first-load threshold does not match build evidence.",
  );
  invariant(
    thresholds.some(
      (entry) =>
        entry.name === "route-specific gzip budget" &&
        entry.actual === budgets.route_specific.gzip_bytes &&
        entry.limit === budgets.route_specific.budget_bytes,
    ),
    "The route-specific threshold does not match build evidence.",
  );

  return evidence;
}

export function createBrowserPerformanceEvidence(fields) {
  return validateBrowserPerformanceEvidence({
    ...fields,
    schema_version: PERFORMANCE_EVIDENCE_POLICY.schemaVersion,
  });
}
