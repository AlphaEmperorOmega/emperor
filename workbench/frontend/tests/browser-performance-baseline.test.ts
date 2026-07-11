import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

type Distribution = {
  coefficient_of_variation: number;
  max: number;
  mean: number;
  median: number;
  min: number;
  p95: number;
  stdev: number;
};

type Operation = {
  duration_ms: number;
  label: string;
  layout_count: number;
  layout_duration_ms: number;
  long_task_count: number;
  long_task_duration_ms: number;
  react_commits: number;
  recalc_style_count: number;
  recalc_style_duration_ms: number;
  script_duration_ms: number;
  task_duration_ms: number;
};

type Threshold = {
  actual: number;
  comparator: "maximum" | "minimum";
  kind: "deterministic" | "informational";
  limit: number;
  name: string;
  passed: boolean;
  unit: string;
};

type BrowserPerformanceBaseline = {
  api: {
    summary: Record<
      string,
      { count: number; duration_ms: Distribution; transfer_bytes: number }
    >;
  };
  build: {
    budgets: Record<
      "first_load" | "route_specific",
      { budget_bytes: number; gzip_bytes: number }
    >;
    mode: string;
  };
  conditions: {
    frame_repetitions_per_webgl_sample: number;
    initial_repetitions: number;
    initial_warmup: number;
    long_session_cycle: string[];
    post_session_workflows: string[];
    requested_window_pixels: number[];
    session_repetitions: number;
    session_warmup: number;
    steady_state_repetitions: number;
    viewport_css_pixels: number[];
    webgl_repetitions: number;
    webgl_warmup: number;
    webgl_workflow: string;
  };
  diagnostics: {
    console_errors: unknown[];
    failed_requests: unknown[];
    page_exceptions: unknown[];
  };
  environment: {
    browser: { product: string };
    cpu: { logical_count: number; model: string };
    gpu: { devices: Array<{ deviceString: string }> };
    memory_bytes: number;
    node: string;
    operating_system: { platform: string; release: string };
    page: { viewport: { height: number; width: number } };
  };
  initial_load: {
    samples: Array<Record<string, number>>;
    summary: Record<string, Distribution>;
  };
  log_import: Operation;
  long_session: {
    heap: {
      checkpoints: Array<{
        documents: number;
        jsEventListeners: number;
        label: string;
        nodes: number;
        usedSize: number;
      }>;
      retained_growth_bytes: number;
      session_growth_bytes: number;
      steady_state_growth_bytes: number;
    };
    samples: Operation[];
    steady_state_samples: Operation[];
    steady_state_summary: Record<string, Distribution>;
    summary: Record<string, Distribution>;
  };
  schema_version: number;
  thresholds: Threshold[];
  training_job: Operation;
  webgl: {
    context_disposal: {
      contexts_created: number;
      contexts_lost: number;
    };
    frame_interval_ms: Distribution;
    renderer: string;
    samples: Array<
      Operation & {
        canvas_count_after_close: number;
        contexts_created: number;
        contexts_lost: number;
        frame_intervals_ms: number[];
        resource_created: Record<string, number>;
        resource_deleted: Record<string, number>;
      }
    >;
  };
};

const baselinePath = resolve(
  process.cwd(),
  "../../docs/architecture/browser-performance-baseline-2026-07-10.json",
);
const baseline = JSON.parse(
  readFileSync(baselinePath, "utf8"),
) as BrowserPerformanceBaseline;

function expectDistribution(distribution: Distribution) {
  for (const value of Object.values(distribution)) {
    expect(Number.isFinite(value)).toBe(true);
  }
  expect(distribution.min).toBeLessThanOrEqual(distribution.median);
  expect(distribution.median).toBeLessThanOrEqual(distribution.max);
  expect(distribution.p95).toBeGreaterThanOrEqual(distribution.min);
  expect(distribution.p95).toBeLessThanOrEqual(distribution.max);
  expect(distribution.coefficient_of_variation).toBeGreaterThanOrEqual(0);
}

describe("browser performance baseline evidence", () => {
  it("records the production workload, environment, warmups, and repetitions", () => {
    expect(baseline.schema_version).toBe(1);
    expect(baseline.build.mode).toBe("Next.js production");
    expect(baseline.conditions).toMatchObject({
      frame_repetitions_per_webgl_sample: 60,
      initial_repetitions: 7,
      initial_warmup: 2,
      requested_window_pixels: [1440, 1000],
      session_repetitions: 20,
      session_warmup: 2,
      steady_state_repetitions: 10,
      webgl_repetitions: 5,
      webgl_warmup: 2,
    });
    expect(baseline.conditions.long_session_cycle).toEqual([
      "Model workspace preset change and graph load",
      "Training workspace visit",
      "Logs workspace scalar chart dialog open and close",
    ]);
    expect(baseline.conditions.post_session_workflows).toHaveLength(2);
    expect(baseline.conditions.webgl_workflow).toContain("3D cluster dialog");
    expect(baseline.conditions.viewport_css_pixels).toEqual([
      baseline.environment.page.viewport.width,
      baseline.environment.page.viewport.height,
    ]);
    expect(baseline.environment).toMatchObject({
      browser: { product: expect.stringMatching(/^Chrome\//) },
      cpu: {
        logical_count: expect.any(Number),
        model: expect.any(String),
      },
      node: expect.stringMatching(/^v\d+/),
      operating_system: {
        platform: "linux",
        release: expect.any(String),
      },
    });
    expect(baseline.environment.memory_bytes).toBeGreaterThan(0);
    expect(baseline.environment.gpu.devices[0]?.deviceString).toContain(
      "SwiftShader",
    );
  });

  it("preserves bundle budgets and records successful public workflows", () => {
    expect(baseline.build.budgets.first_load).toMatchObject({
      budget_bytes: 210_000,
    });
    expect(baseline.build.budgets.route_specific).toMatchObject({
      budget_bytes: 98_000,
    });
    for (const budget of Object.values(baseline.build.budgets)) {
      expect(budget.gzip_bytes).toBeLessThanOrEqual(budget.budget_bytes);
    }
    expect(baseline.diagnostics).toEqual({
      console_errors: [],
      failed_requests: [],
      page_exceptions: [],
    });

    const deterministic = baseline.thresholds.filter(
      (entry) => entry.kind === "deterministic",
    );
    expect(deterministic.map((entry) => entry.name)).toEqual(
      expect.arrayContaining([
        "first-load gzip budget",
        "route-specific gzip budget",
        "browser workflow failures",
        "graph inspection requests exercised",
        "scalar chart requests exercised",
        "Training Job requests exercised",
        "log import requests exercised",
        "WebGL contexts disposed",
      ]),
    );
    expect(deterministic.every((entry) => entry.passed)).toBe(true);
  });

  it("records hydration, main-thread, React commit, layout, and variance evidence", () => {
    expect(baseline.initial_load.samples).toHaveLength(
      baseline.conditions.initial_repetitions,
    );
    for (const field of [
      "first_react_commit_ms",
      "first_contentful_paint_ms",
      "workspace_ready_ms",
      "task_duration_ms",
      "script_duration_ms",
      "react_commits",
      "layout_count",
      "layout_duration_ms",
      "cumulative_layout_shift",
      "heap_used_bytes",
    ]) {
      expectDistribution(baseline.initial_load.summary[field]);
    }
  });

  it("records cache-filling and cache-saturated long-session heap evidence", () => {
    expect(baseline.long_session.samples).toHaveLength(
      baseline.conditions.session_repetitions,
    );
    expect(baseline.long_session.steady_state_samples).toHaveLength(
      baseline.conditions.steady_state_repetitions,
    );
    for (const summary of [
      baseline.long_session.summary,
      baseline.long_session.steady_state_summary,
    ]) {
      for (const field of [
        "duration_ms",
        "task_duration_ms",
        "react_commits",
        "layout_count",
        "layout_duration_ms",
      ]) {
        expectDistribution(summary[field]);
      }
    }

    expect(
      baseline.long_session.heap.checkpoints.map((checkpoint) => checkpoint.label),
    ).toEqual([
      "before_session",
      "mid_session",
      "after_session",
      "after_steady_state",
      "after_training_and_import",
      "after_webgl",
    ]);
    for (const checkpoint of baseline.long_session.heap.checkpoints) {
      expect(checkpoint.usedSize).toBeGreaterThan(0);
      expect(checkpoint.documents).toBeGreaterThan(0);
      expect(checkpoint.nodes).toBeGreaterThan(0);
      expect(checkpoint.jsEventListeners).toBeGreaterThan(0);
    }
    expect(Number.isFinite(baseline.long_session.heap.session_growth_bytes)).toBe(
      true,
    );
    expect(
      Number.isFinite(baseline.long_session.heap.steady_state_growth_bytes),
    ).toBe(true);
    expect(Number.isFinite(baseline.long_session.heap.retained_growth_bytes)).toBe(
      true,
    );
  });

  it("records real graph, scalar, Training Job, and import API activity", () => {
    for (const path of [
      "/inspect",
      "/logs/scalars",
      "/training/jobs",
      "/logs/import",
    ]) {
      const evidence = baseline.api.summary[path];
      expect(evidence.count).toBeGreaterThanOrEqual(1);
      expectDistribution(evidence.duration_ms);
    }
    expect(baseline.training_job).toMatchObject({
      label: "training_job_completion",
      react_commits: expect.any(Number),
    });
    expect(baseline.log_import).toMatchObject({
      label: "log_archive_import",
      react_commits: expect.any(Number),
    });
    expect(baseline.training_job.duration_ms).toBeGreaterThan(0);
    expect(baseline.log_import.duration_ms).toBeGreaterThan(0);
  });

  it("records WebGL frame samples and context-loss disposal", () => {
    expect(baseline.webgl.samples).toHaveLength(
      baseline.conditions.webgl_repetitions,
    );
    expect(baseline.webgl.context_disposal).toEqual({
      contexts_created: baseline.conditions.webgl_repetitions,
      contexts_lost: baseline.conditions.webgl_repetitions,
    });
    expectDistribution(baseline.webgl.frame_interval_ms);
    expect(baseline.webgl.renderer).toContain("SwiftShader");
    for (const sample of baseline.webgl.samples) {
      expect(sample.contexts_created).toBe(1);
      expect(sample.contexts_lost).toBe(1);
      expect(sample.canvas_count_after_close).toBe(0);
      expect(sample.frame_intervals_ms).toHaveLength(
        baseline.conditions.frame_repetitions_per_webgl_sample,
      );
      expect(sample.resource_created.buffer).toBeGreaterThan(0);
      expect(sample.resource_deleted.buffer).toBe(sample.resource_created.buffer);
      expect(sample.resource_deleted.program).toBe(
        sample.resource_created.program,
      );
    }
  });

  it("keeps machine-sensitive runtime thresholds informational", () => {
    const informational = baseline.thresholds.filter(
      (entry) => entry.kind === "informational",
    );
    expect(informational.length).toBeGreaterThan(0);
    expect(informational.map((entry) => entry.name)).toContain(
      "WebGL frame interval p95",
    );
    expect(informational.some((entry) => !entry.passed)).toBe(true);
  });
});
