import { readFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { resolve } from "node:path";

const NEXT_DIRECTORY = resolve(".next");
const FIRST_LOAD_BUDGET_BYTES = 215_000;
const ROUTE_SPECIFIC_BUDGET_BYTES = 100_000;

function readManifest(name) {
  try {
    return JSON.parse(readFileSync(resolve(NEXT_DIRECTORY, name), "utf8"));
  } catch (error) {
    throw new Error(`Unable to read .next/${name}. Run npm run build first.`, {
      cause: error,
    });
  }
}

function gzipSize(relativePath) {
  const contents = readFileSync(resolve(NEXT_DIRECTORY, relativePath));
  return gzipSync(contents).byteLength;
}

function sumGzipSize(files) {
  return [...new Set(files)].reduce((total, file) => total + gzipSize(file), 0);
}

function formatKilobytes(bytes) {
  return `${(bytes / 1_000).toFixed(1)} kB`;
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

const appBuildManifest = readManifest("app-build-manifest.json");
const loadableManifest = readManifest("react-loadable-manifest.json");
const pageFiles = appBuildManifest.pages["/page"]?.filter((file) =>
  file.endsWith(".js"),
);

assert(pageFiles?.length, "The production /page JavaScript entry was not found.");

const otherEntryFiles = new Set(
  Object.entries(appBuildManifest.pages)
    .filter(([entry]) => entry !== "/page")
    .flatMap(([, files]) => files)
    .filter((file) => file.endsWith(".js")),
);
const routeSpecificFiles = pageFiles.filter((file) => !otherEntryFiles.has(file));
const firstLoadBytes = sumGzipSize(pageFiles);
const routeSpecificBytes = sumGzipSize(routeSpecificFiles);

assert(
  firstLoadBytes <= FIRST_LOAD_BUDGET_BYTES,
  `First-load JavaScript is ${formatKilobytes(firstLoadBytes)}; budget is ${formatKilobytes(FIRST_LOAD_BUDGET_BYTES)}.`,
);
assert(
  routeSpecificBytes <= ROUTE_SPECIFIC_BUDGET_BYTES,
  `Route-specific JavaScript is ${formatKilobytes(routeSpecificBytes)}; budget is ${formatKilobytes(ROUTE_SPECIFIC_BUDGET_BYTES)}.`,
);

const deferredModules = [
  {
    label: "ECharts scalar charts",
    target: "@/features/workbench/components/logs/log-scalar-chart",
  },
  {
    label: "React Flow graph canvas",
    target: "@/features/workbench/components/screen/graph-canvas",
  },
  {
    label: "Selected node details",
    target: "@/features/workbench/components/graph/selected-node-details",
  },
  {
    label: "Dagre graph layout",
    target: "@/lib/graph/layout",
  },
  {
    label: "Three.js neuron scene",
    target: "@/features/workbench/components/graph/neuron-cluster-3d-scene",
  },
  {
    label: "Training workspace",
    target: "@/features/workbench/components/connected-training-panel",
  },
  {
    label: "Full Config dialog",
    target: "@/features/workbench/components/config/full-config-dialog",
  },
  {
    label: "API Connection dialog",
    target: "@/features/workbench/components/screen/api-connection-dialog",
  },
  {
    label: "Import Logs dialog",
    target: "@/features/workbench/components/screen/import-logs-dialog",
  },
];

const initialFiles = new Set(pageFiles);
const deferredResults = deferredModules.map(({ label, target }) => {
  const entry = Object.entries(loadableManifest).find(([key]) =>
    key.endsWith(` -> ${target}`),
  );
  assert(entry, `${label} is no longer represented by a dynamic import.`);

  const files = entry[1].files.filter((file) => file.endsWith(".js"));
  assert(files.length, `${label} has no emitted JavaScript chunk.`);

  const initialOverlap = files.filter((file) => initialFiles.has(file));
  assert(
    initialOverlap.length === 0,
    `${label} leaked into the initial /page chunks: ${initialOverlap.join(", ")}.`,
  );

  return {
    label,
    bytes: sumGzipSize(files),
    files,
  };
});

const scalarResult = deferredResults.find(
  ({ label }) => label === "ECharts scalar charts",
);
const scalarFiles = new Set(scalarResult.files);
const scalarChunkOwners = Object.entries(loadableManifest)
  .filter(([, value]) => value.files.some((file) => scalarFiles.has(file)))
  .map(([key]) => key);
const allowedScalarOwners = [
  "@/features/workbench/components/logs/log-scalar-chart",
  "@/features/workbench/components/monitor/monitor-charts-modal",
];
assert(
  scalarChunkOwners.every((owner) =>
    allowedScalarOwners.some((target) => owner.endsWith(` -> ${target}`)),
  ),
  `ECharts chunks have unexpected owners: ${scalarChunkOwners.join(", ")}.`,
);

console.log("Performance budgets passed");
console.log(
  `- First-load JavaScript: ${formatKilobytes(firstLoadBytes)} / ${formatKilobytes(FIRST_LOAD_BUDGET_BYTES)}`,
);
console.log(
  `- Route-specific JavaScript: ${formatKilobytes(routeSpecificBytes)} / ${formatKilobytes(ROUTE_SPECIFIC_BUDGET_BYTES)}`,
);
for (const { label, bytes } of deferredResults) {
  console.log(`- Deferred ${label}: ${formatKilobytes(bytes)} gzip`);
}
