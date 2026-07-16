import { mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gzipSync } from "node:zlib";
import { mkdtemp, rm } from "node:fs/promises";
import { afterEach, describe, expect, it } from "vitest";
import {
  PERFORMANCE_EVIDENCE_POLICY,
  assertBuildPerformanceBudgets,
  collectBuildPerformanceEvidence,
  createBrowserPerformanceEvidence,
  createBrowserPerformanceThresholds,
  deterministicPerformanceFailures,
  validateBrowserPerformanceEvidence,
} from "../scripts/performance-evidence.mjs";

const temporaryDirectories = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories.splice(0).map((directory) =>
      rm(directory, { force: true, recursive: true }),
    ),
  );
});

async function createBuildFixture() {
  const directory = await mkdtemp(join(tmpdir(), "performance-evidence-"));
  temporaryDirectories.push(directory);
  const chunks = new Map([
    ["shared.js", "shared runtime"],
    ["route.js", "route implementation"],
    ["scalar.js", "scalar chart implementation"],
  ]);
  const loadableManifest = {};

  PERFORMANCE_EVIDENCE_POLICY.deferredModules.forEach(({ target }, index) => {
    const file = target.endsWith("/log-scalar-chart")
      ? "scalar.js"
      : `deferred-${index}.js`;
    chunks.set(file, `deferred implementation ${index}`);
    loadableManifest[`fixture-${index} -> ${target}`] = { files: [file] };
  });
  loadableManifest[
    "fixture-monitor -> @/features/workbench/components/monitor/monitor-charts-modal"
  ] = { files: ["scalar.js"] };

  await Promise.all([
    writeFile(
      join(directory, "app-build-manifest.json"),
      JSON.stringify({
        pages: {
          "/layout": ["shared.js"],
          "/page": ["shared.js", "route.js", "route.js"],
        },
      }),
    ),
    writeFile(
      join(directory, "react-loadable-manifest.json"),
      JSON.stringify(loadableManifest),
    ),
    ...[...chunks].map(([file, contents]) =>
      writeFile(join(directory, file), contents),
    ),
  ]);

  return { chunks, directory, loadableManifest };
}

async function createTurbopackBuildFixture() {
  const directory = await mkdtemp(join(tmpdir(), "performance-evidence-next16-"));
  temporaryDirectories.push(directory);
  const chunks = new Map([
    ["static/chunks/shared.js", "shared Turbopack runtime"],
    ["static/chunks/route.js", ""],
  ]);
  const clientReferenceManifest = {
    entryJSFiles: {
      "[project]/app/layout": ["/_next/static/chunks/shared.js"],
      "[project]/app/page": [
        "/_next/static/chunks/shared.js",
        "/_next/static/chunks/route.js",
        "/_next/static/chunks/route.js",
      ],
    },
  };
  const loadableManifest = {};
  const generatedImports = [];
  const generatedLoaders = [];

  PERFORMANCE_EVIDENCE_POLICY.deferredModules.forEach(
    ({ exportName, routeEntry }, index) => {
      const loaderId = 2201 + index;
      const moduleId = 1201 + index;
      const file = `static/chunks/deferred-${index}.js`;
      chunks.set(file, `deferred ${exportName} implementation`);
      generatedImports.push(
        `e.A(${loaderId}).then(module=>module.${exportName}),{loadableGenerated:{modules:[${moduleId}]}}`,
      );
      generatedLoaders.push(
        `${loaderId},e=>{e.v(load=>Promise.all([${JSON.stringify(file)}].map(file=>e.l(file))).then(()=>load(${moduleId})))}`,
      );
      if (routeEntry) {
        loadableManifest[moduleId] = { id: moduleId, files: [file] };
      }
    },
  );
  chunks.set(
    "static/chunks/route.js",
    `${generatedImports.join(";")};[${generatedLoaders.join(",")}];`,
  );

  const manifestDirectory = join(directory, "server", "app");
  const loadableDirectory = join(manifestDirectory, "page");
  await Promise.all([
    mkdir(loadableDirectory, { recursive: true }),
    mkdir(join(directory, "static", "chunks"), { recursive: true }),
  ]);
  await Promise.all([
    writeFile(
      join(manifestDirectory, "page_client-reference-manifest.js"),
      `globalThis.__RSC_MANIFEST = globalThis.__RSC_MANIFEST || {};\nglobalThis.__RSC_MANIFEST["/page"] = ${JSON.stringify(clientReferenceManifest)};\n`,
    ),
    writeFile(
      join(loadableDirectory, "react-loadable-manifest.json"),
      JSON.stringify(loadableManifest),
    ),
    ...[...chunks].map(([file, contents]) =>
      writeFile(join(directory, file), contents),
    ),
  ]);

  return {
    chunks,
    directory,
    loadableDirectory,
    loadableManifest,
    routeFile: join(directory, "static", "chunks", "route.js"),
  };
}

function passingMetrics(buildBudgets) {
  return {
    browserWorkflowFailures: 0,
    buildBudgets,
    connectedWebglCanvases: 0,
    initialCumulativeLayoutShiftP95: 0.01,
    initialTaskDurationP95: 100,
    initialWorkspaceReadyP95: 200,
    inspectionApiDurationP95: 100,
    inspectionRequestCount: 1,
    logImportRequestCount: 1,
    longSessionDurationP95: 200,
    scalarApiDurationP95: 100,
    scalarRequestCount: 1,
    sessionHeapGrowth: 1_000,
    steadyStateHeapGrowth: 1_000,
    trainingJobRequestCount: 1,
    webglContextsCreated: 1,
    webglContextsLost: 1,
    webglFrameIntervalP95: 16.7,
  };
}

function distribution(value = 1) {
  return {
    coefficient_of_variation: 0,
    max: value,
    mean: value,
    median: value,
    min: value,
    p95: value,
    stdev: 0,
  };
}

function operation(label) {
  return {
    duration_ms: 1,
    label,
    layout_count: 0,
    layout_duration_ms: 0,
    long_task_count: 0,
    long_task_duration_ms: 0,
    react_commits: 1,
    recalc_style_count: 0,
    recalc_style_duration_ms: 0,
    script_duration_ms: 0,
    task_duration_ms: 0,
  };
}

function createArtifactFields(buildBudgets, thresholds) {
  return {
    api: { entries: [], summary: {} },
    build: {
      build_id: "fixture-build",
      budgets: buildBudgets,
      mode: "Next.js production",
    },
    conditions: {
      browser_cache: "disabled for initial samples",
      frame_repetitions_per_webgl_sample: 1,
      initial_repetitions: 1,
      initial_warmup: 1,
      long_session_cycle: ["fixture cycle"],
      post_session_workflows: ["fixture workflow"],
      requested_window_pixels: [100, 100],
      session_repetitions: 1,
      session_warmup: 1,
      steady_state_repetitions: 1,
      storage_policy: "fixture storage",
      viewport_css_pixels: [100, 100],
      webgl_repetitions: 1,
      webgl_warmup: 1,
      webgl_workflow: "fixture WebGL workflow",
    },
    diagnostics: {
      console_errors: [],
      failed_requests: [],
      page_exceptions: [],
    },
    environment: {
      browser: { product: "Chrome/fixture" },
      chromium_command: "chromium",
      cpu: { logical_count: 1, model: "fixture CPU" },
      gpu: { devices: [{ deviceString: "fixture GPU" }] },
      memory_bytes: 1,
      node: "vfixture",
      operating_system: {
        arch: "x64",
        platform: "linux",
        release: "fixture",
        version: "fixture",
      },
      page: {
        deviceMemoryGiB: null,
        devicePixelRatio: 1,
        hardwareConcurrency: 1,
        language: "en",
        screen: { height: 100, width: 100 },
        userAgent: "fixture",
        viewport: { height: 100, width: 100 },
      },
    },
    initial_load: { samples: [{}], summary: { duration_ms: distribution() } },
    log_import: operation("log_import"),
    long_session: {
      heap: {
        checkpoints: [
          {
            documents: 1,
            jsEventListeners: 1,
            label: "fixture",
            nodes: 1,
            usedSize: 1,
          },
        ],
        retained_growth_bytes: 0,
        retained_growth_ratio: 0,
        session_growth_bytes: 0,
        session_growth_bytes_per_cycle: 0,
        session_growth_ratio: 0,
        steady_state_growth_bytes: 0,
        steady_state_growth_bytes_per_cycle: 0,
        steady_state_growth_ratio: 0,
      },
      samples: [operation("session")],
      steady_state_samples: [operation("steady_state")],
      steady_state_summary: { duration_ms: distribution() },
      summary: { duration_ms: distribution() },
    },
    thresholds,
    training_job: operation("training_job"),
    webgl: {
      context_disposal: { contexts_created: 1, contexts_lost: 1 },
      frame_interval_ms: distribution(16.7),
      renderer: null,
      samples: [
        {
          ...operation("webgl"),
          canvas_count_after_close: 0,
          contexts_created: 1,
          contexts_lost: 1,
          frame_intervals_ms: [16.7],
          renderer: null,
          resource_created: { buffer: 1 },
          resource_deleted: { buffer: 1 },
          vendor: null,
        },
      ],
      vendor: null,
    },
  };
}

describe("performance evidence Module", () => {
  it("collects deduplicated build budgets and current deferred targets", async () => {
    const { chunks, directory } = await createBuildFixture();
    const evidence = collectBuildPerformanceEvidence(directory);

    expect(evidence.pageFiles).toEqual(["shared.js", "route.js"]);
    expect(evidence.routeSpecificFiles).toEqual(["route.js"]);
    expect(evidence.manifestKind).toBe("next15-webpack");
    expect(evidence.budgets).toEqual({
      first_load: {
        budget_bytes: PERFORMANCE_EVIDENCE_POLICY.budgets.firstLoadBytes,
        gzip_bytes:
          gzipSync(chunks.get("shared.js")).byteLength +
          gzipSync(chunks.get("route.js")).byteLength,
      },
      route_specific: {
        budget_bytes: PERFORMANCE_EVIDENCE_POLICY.budgets.routeSpecificBytes,
        gzip_bytes: gzipSync(chunks.get("route.js")).byteLength,
      },
    });
    expect(PERFORMANCE_EVIDENCE_POLICY.deferredModules).toContainEqual(
      expect.objectContaining({
        label: "Training workspace",
        target: "@/features/workbench/components/training-panel",
      }),
    );
    expect(PERFORMANCE_EVIDENCE_POLICY.deferredModules).toContainEqual(
      expect.objectContaining({
        label: "Training execution state",
        target: "@/features/workbench/providers/training-execution-provider",
      }),
    );
    expect(PERFORMANCE_EVIDENCE_POLICY.deferredModules).toContainEqual(
      expect.objectContaining({
        label: "Runtime Defaults header controls",
        target:
          "@/features/workbench/components/screen/app-header-runtime-defaults-controls",
      }),
    );
    expect(PERFORMANCE_EVIDENCE_POLICY.requiredMeasured).toEqual({
      firstLoadBytes: 205_000,
      routeSpecificBytes: 93_000,
    });
    expect(PERFORMANCE_EVIDENCE_POLICY.deferredModules).not.toContainEqual(
      expect.objectContaining({
        target:
          "@/features/workbench/components/connected-training-panel",
      }),
    );
    expect(evidence.deferred).toHaveLength(
      PERFORMANCE_EVIDENCE_POLICY.deferredModules.length,
    );
    expect(() => assertBuildPerformanceBudgets(evidence)).not.toThrow();
  });

  it("collects Next 16 Turbopack route entries and named deferred Modules", async () => {
    const { chunks, directory } = await createTurbopackBuildFixture();
    const evidence = collectBuildPerformanceEvidence(directory);

    expect(evidence.manifestKind).toBe("next16-turbopack");
    expect(evidence.pageFiles).toEqual([
      "static/chunks/shared.js",
      "static/chunks/route.js",
    ]);
    expect(evidence.routeSpecificFiles).toEqual([
      "static/chunks/route.js",
    ]);
    expect(evidence.budgets).toEqual({
      first_load: {
        budget_bytes: PERFORMANCE_EVIDENCE_POLICY.budgets.firstLoadBytes,
        gzip_bytes:
          gzipSync(chunks.get("static/chunks/shared.js")).byteLength +
          gzipSync(chunks.get("static/chunks/route.js")).byteLength,
      },
      route_specific: {
        budget_bytes: PERFORMANCE_EVIDENCE_POLICY.budgets.routeSpecificBytes,
        gzip_bytes: gzipSync(
          chunks.get("static/chunks/route.js"),
        ).byteLength,
      },
    });
    expect(evidence.deferred).toHaveLength(
      PERFORMANCE_EVIDENCE_POLICY.deferredModules.length,
    );
    expect(evidence.deferred).toContainEqual(
      expect.objectContaining({
        label: "Training workspace",
        target: "@/features/workbench/components/training-panel",
      }),
    );
    expect(evidence.deferred).toContainEqual(
      expect.objectContaining({
        label: "Model workspace",
        target: "@/features/workbench/components/workbench-model-workspace",
      }),
    );
    expect(() => assertBuildPerformanceBudgets(evidence)).not.toThrow();
  });

  it("rejects a Next 16 Turbopack dynamic chunk in the initial route", async () => {
    const { directory, loadableDirectory, loadableManifest } =
      await createTurbopackBuildFixture();
    loadableManifest[1201] = { files: ["static/chunks/route.js"] };
    await writeFile(
      join(loadableDirectory, "react-loadable-manifest.json"),
      JSON.stringify(loadableManifest),
    );

    expect(() => collectBuildPerformanceEvidence(directory)).toThrow(
      "leaked into the initial /page chunks",
    );
  });

  it("rejects a missing named Turbopack deferred Module", async () => {
    const { directory, routeFile } = await createTurbopackBuildFixture();
    const routeSource = await readFile(routeFile, "utf8");
    await writeFile(
      routeFile,
      routeSource.replace(
        ".then(module=>module.TrainingPanel)",
        ".then(module=>module.RenamedTrainingPanel)",
      ),
    );

    expect(() => collectBuildPerformanceEvidence(directory)).toThrow(
      "Training workspace is no longer represented by a named Turbopack dynamic import",
    );
  });

  it("requires measured bundle headroom without changing policy budgets", () => {
    const evidence = {
      budgets: {
        first_load: {
          budget_bytes: PERFORMANCE_EVIDENCE_POLICY.budgets.firstLoadBytes,
          gzip_bytes:
            PERFORMANCE_EVIDENCE_POLICY.requiredMeasured.firstLoadBytes + 1,
        },
        route_specific: {
          budget_bytes: PERFORMANCE_EVIDENCE_POLICY.budgets.routeSpecificBytes,
          gzip_bytes: 1,
        },
      },
    };

    expect(evidence.budgets.first_load.gzip_bytes).toBeLessThan(
      evidence.budgets.first_load.budget_bytes,
    );
    expect(() => assertBuildPerformanceBudgets(evidence)).toThrow(
      "required headroom",
    );
  });

  it("rejects deferred scalar chunks owned outside the allowed seam", async () => {
    const { directory, loadableManifest } = await createBuildFixture();
    loadableManifest[
      "fixture-unexpected -> @/features/workbench/components/unrelated-panel"
    ] = { files: ["scalar.js"] };
    await writeFile(
      join(directory, "react-loadable-manifest.json"),
      JSON.stringify(loadableManifest),
    );

    expect(() => collectBuildPerformanceEvidence(directory)).toThrow(
      "ECharts chunks have unexpected owners",
    );
  });

  it("keeps stable failures deterministic and machine-sensitive limits informational", () => {
    const buildBudgets = {
      first_load: {
        budget_bytes: PERFORMANCE_EVIDENCE_POLICY.budgets.firstLoadBytes,
        gzip_bytes: 100,
      },
      route_specific: {
        budget_bytes: PERFORMANCE_EVIDENCE_POLICY.budgets.routeSpecificBytes,
        gzip_bytes: 50,
      },
    };
    const thresholds = createBrowserPerformanceThresholds({
      ...passingMetrics(buildBudgets),
      browserWorkflowFailures: 1,
      webglFrameIntervalP95: 50,
    });

    expect(
      thresholds.find((entry) => entry.name === "browser workflow failures"),
    ).toMatchObject({ kind: "deterministic", passed: false });
    expect(
      thresholds.find((entry) => entry.name === "WebGL frame interval p95"),
    ).toMatchObject({ kind: "informational", passed: false });
    expect(deterministicPerformanceFailures(thresholds).map(({ name }) => name)).toEqual([
      "browser workflow failures",
    ]);
  });

  it("constructs and validates the current schema with nullable WebGL identity", () => {
    const buildBudgets = {
      first_load: {
        budget_bytes: PERFORMANCE_EVIDENCE_POLICY.budgets.firstLoadBytes,
        gzip_bytes: 100,
      },
      route_specific: {
        budget_bytes: PERFORMANCE_EVIDENCE_POLICY.budgets.routeSpecificBytes,
        gzip_bytes: 50,
      },
    };
    const thresholds = createBrowserPerformanceThresholds(
      passingMetrics(buildBudgets),
    );
    const evidence = createBrowserPerformanceEvidence(
      createArtifactFields(buildBudgets, thresholds),
    );

    expect(evidence.schema_version).toBe(
      PERFORMANCE_EVIDENCE_POLICY.schemaVersion,
    );
    expect(evidence.webgl).toMatchObject({ renderer: null, vendor: null });
    expect(validateBrowserPerformanceEvidence(evidence)).toBe(evidence);
    expect(() =>
      validateBrowserPerformanceEvidence({ ...evidence, schema_version: 999 }),
    ).toThrow("schema version");
    expect(() =>
      validateBrowserPerformanceEvidence({
        ...evidence,
        thresholds: [
          { ...thresholds[0], passed: !thresholds[0].passed },
          ...thresholds.slice(1),
        ],
      }),
    ).toThrow("inconsistent passed value");
  });
});
