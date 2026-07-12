export type Distribution = {
  coefficient_of_variation: number;
  max: number;
  mean: number;
  median: number;
  min: number;
  p95: number;
  stdev: number;
};

export type Operation = {
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

export type PerformanceThreshold = {
  actual: number;
  comparator: "maximum" | "minimum";
  kind: "deterministic" | "informational";
  limit: number;
  name: string;
  passed: boolean;
  unit: string;
};

export type BuildBudgets = Record<
  "first_load" | "route_specific",
  { budget_bytes: number; gzip_bytes: number }
>;

export type BuildPerformanceEvidence = {
  budgets: BuildBudgets;
  deferred: Array<{ bytes: number; files: string[]; label: string }>;
  pageFiles: string[];
  routeSpecificFiles: string[];
};

export type BrowserPerformanceMetrics = {
  browserWorkflowFailures: number;
  buildBudgets: BuildBudgets;
  connectedWebglCanvases: number;
  initialCumulativeLayoutShiftP95: number;
  initialTaskDurationP95: number;
  initialWorkspaceReadyP95: number;
  inspectionApiDurationP95: number;
  inspectionRequestCount: number;
  logImportRequestCount: number;
  longSessionDurationP95: number;
  scalarApiDurationP95: number;
  scalarRequestCount: number;
  sessionHeapGrowth: number;
  steadyStateHeapGrowth: number;
  trainingJobRequestCount: number;
  webglContextsCreated: number;
  webglContextsLost: number;
  webglFrameIntervalP95: number;
};

export type BrowserPerformanceEvidence = {
  api: {
    entries: unknown[];
    summary: Record<
      string,
      { count: number; duration_ms: Distribution; transfer_bytes: number }
    >;
  };
  build: {
    build_id: string;
    budgets: BuildBudgets;
    mode: string;
  };
  conditions: {
    browser_cache: string;
    frame_repetitions_per_webgl_sample: number;
    initial_repetitions: number;
    initial_warmup: number;
    long_session_cycle: string[];
    post_session_workflows: string[];
    requested_window_pixels: number[];
    session_repetitions: number;
    session_warmup: number;
    steady_state_repetitions: number;
    storage_policy: string;
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
    chromium_command: string;
    cpu: { logical_count: number; model: string };
    gpu: { devices: Array<{ deviceString: string }> };
    memory_bytes: number;
    node: string;
    operating_system: {
      arch: string;
      platform: string;
      release: string;
      version: string;
    };
    page: {
      deviceMemoryGiB: number | null;
      devicePixelRatio: number;
      hardwareConcurrency: number;
      language: string;
      screen: { height: number; width: number };
      userAgent: string;
      viewport: { height: number; width: number };
    };
  };
  initial_load: {
    samples: Array<Record<string, unknown>>;
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
      retained_growth_ratio: number;
      session_growth_bytes: number;
      session_growth_bytes_per_cycle: number;
      session_growth_ratio: number;
      steady_state_growth_bytes: number;
      steady_state_growth_bytes_per_cycle: number;
      steady_state_growth_ratio: number;
    };
    samples: Operation[];
    steady_state_samples: Operation[];
    steady_state_summary: Record<string, Distribution>;
    summary: Record<string, Distribution>;
  };
  schema_version: number;
  thresholds: PerformanceThreshold[];
  training_job: Operation;
  webgl: {
    context_disposal: {
      contexts_created: number;
      contexts_lost: number;
    };
    frame_interval_ms: Distribution;
    renderer: string | null;
    samples: Array<
      Operation & {
        canvas_count_after_close: number;
        contexts_created: number;
        contexts_lost: number;
        frame_intervals_ms: number[];
        renderer: string | null;
        resource_created: Record<string, number>;
        resource_deleted: Record<string, number>;
        vendor: string | null;
      }
    >;
    vendor: string | null;
  };
};

export const PERFORMANCE_EVIDENCE_POLICY: Readonly<{
  budgets: Readonly<{
    firstLoadBytes: number;
    routeSpecificBytes: number;
  }>;
  deferredModules: ReadonlyArray<
    Readonly<{ label: string; target: string }>
  >;
  scalarChunkAllowedOwners: readonly string[];
  schemaVersion: number;
}>;

export function formatPerformanceKilobytes(bytes: number): string;
export function collectBuildPerformanceEvidence(
  nextDirectory: string,
): BuildPerformanceEvidence;
export function assertBuildPerformanceBudgets(
  evidence: BuildPerformanceEvidence,
): void;
export function createBrowserPerformanceThresholds(
  metrics: BrowserPerformanceMetrics,
): PerformanceThreshold[];
export function deterministicPerformanceFailures(
  thresholds: PerformanceThreshold[],
): PerformanceThreshold[];
export function validateBrowserPerformanceEvidence(
  value: unknown,
): BrowserPerformanceEvidence;
export function createBrowserPerformanceEvidence(
  fields: Omit<BrowserPerformanceEvidence, "schema_version">,
): BrowserPerformanceEvidence;
