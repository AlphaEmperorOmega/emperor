import { render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  includeStartedExperiment: vi.fn(),
  inspection: {
    target: {
      kind: "preset" as const,
      modelPackage: { modelType: "linears", model: "linear" },
      preset: "baseline",
      experimentTask: "image-classification",
      datasets: ["Mnist"],
    } as {
      kind: "preset" | "historical-run";
      modelPackage: { modelType: string; model: string };
      preset: string;
      experimentTask: string;
      datasets: string[];
      run?: {
        runId: string;
        experiment: string;
        preset: string;
        dataset: string;
        experimentTask: string;
      };
    },
    browser: {
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "baseline",
      selectedDatasets: ["Mnist"],
    },
    presets: [
      { name: "baseline", label: "BASELINE" },
    ],
  },
  workspaceInput: vi.fn(),
  protectedAccessReady: true,
}));

vi.mock("@/features/workbench/providers/workbench-providers", () => ({
  useModelPackageInspection: () => ({
    target: mocks.inspection.target,
    browser: mocks.inspection.browser,
    options: {
      presets: mocks.inspection.presets,
    },
  }),
}));

vi.mock("@/features/workbench/providers/workbench-connection-provider", () => ({
  isWorkbenchProtectedAccessReady: () => mocks.protectedAccessReady,
  useWorkbenchCapabilities: () => ({
    capabilities: { logDeletionEnabled: true },
  }),
  useWorkbenchConnection: () => ({}),
  useRegisterWorkbenchConnectionReset: () => undefined,
}));

vi.mock("@/features/workbench/providers/training-provider", () => ({
  useActiveTrainingJob: () => ({ activeTrainingJob: undefined }),
}));

vi.mock("@/features/workbench/state/logs/_logs-chart-state", () => ({
  useLogsChartViewModel: () => ({
    content: {},
    settings: {},
    status: {},
    actions: {},
  }),
}));

vi.mock("@/features/workbench/state/logs/use-log-queries", () => ({
  useLogRunArtifactsQuery: () => ({
    data: undefined,
    error: null,
    isLoading: false,
  }),
}));

vi.mock("@/features/workbench/state/logs/_use-logs-workspace-state", () => ({
  useLogsWorkspaceImplementation: (input: unknown) => {
    mocks.workspaceInput(input);
    const idleQuery = {
      data: undefined,
      error: null,
      isLoading: false,
      isFetching: false,
    };
    const noop = vi.fn();
    return {
      includeStartedExperiment: mocks.includeStartedExperiment,
      clearForConnectionChange: noop,
      enabled: true,
      logDeletionEnabled: true,
      scopeMode: "target",
      targetScope: {
        modelType: "linears",
        model: "linear",
        preset: "BASELINE",
        datasets: ["Mnist"],
      },
      runs: [],
      visibleRuns: [],
      visibleRunIds: [],
      runsQuery: idleQuery,
      experimentsQuery: idleQuery,
      tagsQuery: idleQuery,
      experimentOptions: [],
      datasetOptions: [],
      modelOptions: [],
      presetOptions: [],
      tagOptions: [],
      confusionMatrixRateTags: [],
      selectedExperiments: new Set<string>(),
      selectedDatasets: new Set<string>(),
      selectedModels: new Set<string>(),
      selectedPresets: new Set<string>(),
      selectedTags: new Set<string>(),
      selectedTagList: [],
      collapsedMetricGroups: new Set<string>(),
      selectedRun: undefined,
      loadedRunCount: 0,
      totalRunCount: 0,
      canLoadMoreRuns: false,
      isLoadingMoreRuns: false,
      loadedScalarTagRunCount: 0,
      totalScalarTagRunCount: 0,
      canLoadMoreScalarTags: false,
      isLoadingMoreScalarTags: false,
      useCurrentTargetScope: noop,
      showAllRuns: noop,
      toggleExperiment: noop,
      toggleDataset: noop,
      toggleModel: noop,
      togglePreset: noop,
      toggleTag: noop,
      selectAllExperiments: noop,
      selectNoExperiments: noop,
      selectAllDatasets: noop,
      selectNoDatasets: noop,
      selectAllModels: noop,
      selectNoModels: noop,
      selectAllPresets: noop,
      selectNoPresets: noop,
      selectAllTags: noop,
      selectNoTags: noop,
      refreshLogLists: () => Promise.resolve(),
      loadMoreRuns: noop,
      loadMoreScalarTags: noop,
      toggleMetricGroup: noop,
      setSelectedDetailRunId: noop,
      deletion: {
        enabled: true,
        presetTargetExperiment: null,
        operation: null,
        clearForConnectionChange: noop,
        actions: {
          openExperiment: noop,
          openPreset: noop,
          cancel: noop,
          retryPlan: noop,
          confirm: noop,
        },
      },
    };
  },
}));

import {
  LogsWorkspaceProvider,
  useLogRunDetail,
  useLogsBrowser,
  useLogsCharts,
  useLogsDeletion,
} from "@/features/workbench/providers/logs-workspace-provider";

beforeEach(() => {
  mocks.includeStartedExperiment.mockReset();
  mocks.workspaceInput.mockReset();
  mocks.protectedAccessReady = true;
  mocks.inspection.target = {
    kind: "preset",
    modelPackage: { modelType: "linears", model: "linear" },
    preset: "baseline",
    experimentTask: "image-classification",
    datasets: ["Mnist"],
  };
  mocks.inspection.browser = {
    selectedModelType: "linears",
    selectedModel: "linear",
    selectedPreset: "baseline",
    selectedDatasets: ["Mnist"],
  };
  mocks.inspection.presets = [{ name: "baseline", label: "BASELINE" }];
});

describe("LogsWorkspaceProvider", () => {
  it("keeps Logs reads and deletion disabled until protected access is ready", () => {
    mocks.protectedAccessReady = false;

    render(
      <LogsWorkspaceProvider enabled>
        <span>Logs child</span>
      </LogsWorkspaceProvider>,
    );

    expect(mocks.workspaceInput).toHaveBeenLastCalledWith(
      expect.objectContaining({
        enabled: false,
        logDeletionEnabled: false,
      }),
    );
  });

  it("publishes focused browser and deletion Interface shapes", () => {
    let browserKeys: string[] = [];
    let filterKeys: string[] = [];
    let browserActionKeys: string[] = [];
    let chartKeys: string[] = [];
    let detailKeys: string[] = [];
    let deletionKeys: string[] = [];
    let deletionActionKeys: string[] = [];

    function InterfaceProbe() {
      const browser = useLogsBrowser();
      const charts = useLogsCharts();
      const detail = useLogRunDetail();
      const deletion = useLogsDeletion();
      browserKeys = Object.keys(browser).sort();
      filterKeys = Object.keys(browser.filters).sort();
      browserActionKeys = Object.keys(browser.actions).sort();
      chartKeys = Object.keys(charts).sort();
      detailKeys = Object.keys(detail).sort();
      deletionKeys = Object.keys(deletion).sort();
      deletionActionKeys = Object.keys(deletion.actions).sort();
      return null;
    }

    render(
      <LogsWorkspaceProvider enabled>
        <InterfaceProbe />
      </LogsWorkspaceProvider>,
    );

    expect(browserKeys).toEqual([
      "actions",
      "filters",
      "pagination",
      "results",
      "scope",
      "status",
    ]);
    expect(filterKeys).toEqual([
      "datasets",
      "experiments",
      "models",
      "presets",
      "tags",
    ]);
    expect(browserActionKeys).toEqual([
      "loadMoreRuns",
      "loadMoreScalarTags",
      "refresh",
      "selectAll",
      "selectNone",
      "toggleFilter",
    ]);
    expect(chartKeys).toEqual(["actions", "content", "settings", "status"]);
    expect(detailKeys).toEqual(["artifacts", "run", "status"]);
    expect(deletionKeys).toEqual([
      "actions",
      "enabled",
      "operation",
      "presetTargetExperiment",
    ]);
    expect(deletionActionKeys).toEqual([
      "cancel",
      "confirm",
      "openExperiment",
      "openPreset",
      "retryPlan",
    ]);
  });

  it("seeds every Training folder observed before the provider mounts", async () => {
    render(
      <LogsWorkspaceProvider
        enabled
        startedExperiments={["first_run", "second_run"]}
      >
        <span>Logs child</span>
      </LogsWorkspaceProvider>,
    );

    await waitFor(() => {
      expect(mocks.includeStartedExperiment.mock.calls).toEqual([
        ["first_run"],
        ["second_run"],
      ]);
    });
  });

  it("delivers each buffered Training folder only once across later additions", async () => {
    const rendered = render(
      <LogsWorkspaceProvider enabled startedExperiments={["first_run"]}>
        <span>Logs child</span>
      </LogsWorkspaceProvider>,
    );

    await waitFor(() => {
      expect(mocks.includeStartedExperiment.mock.calls).toEqual([["first_run"]]);
    });
    mocks.includeStartedExperiment.mockClear();

    rendered.rerender(
      <LogsWorkspaceProvider
        enabled
        startedExperiments={["first_run", "second_run"]}
      >
        <span>Logs child</span>
      </LogsWorkspaceProvider>,
    );

    await waitFor(() => {
      expect(mocks.includeStartedExperiment.mock.calls).toEqual([["second_run"]]);
    });
  });

  it("composes target scope from the complete historical target instead of browser task browsing", () => {
    mocks.inspection.target = {
      kind: "historical-run",
      modelPackage: { modelType: "linears", model: "linear" },
      preset: "HistoricalPreset",
      experimentTask: "image-classification",
      datasets: ["FashionMnist"],
      run: {
        runId: "historical-run",
        experiment: "historical",
        preset: "HistoricalPreset",
        dataset: "FashionMnist",
        experimentTask: "image-classification",
      },
    };
    mocks.inspection.browser = {
      selectedModelType: "bert",
      selectedModel: "linear",
      selectedPreset: "pre-norm",
      selectedDatasets: ["PennTreebank"],
    };

    render(
      <LogsWorkspaceProvider enabled>
        <span>Logs child</span>
      </LogsWorkspaceProvider>,
    );

    expect(mocks.workspaceInput).toHaveBeenLastCalledWith(
      expect.objectContaining({
        targetScope: {
          modelType: "linears",
          model: "linear",
          preset: "HistoricalPreset",
          datasets: ["FashionMnist"],
        },
      }),
    );
  });

  it("maps a preset target through its catalog label without using a divergent browser preset", () => {
    mocks.inspection.browser = {
      ...mocks.inspection.browser,
      selectedPreset: "other-browser-preset",
    };

    render(
      <LogsWorkspaceProvider enabled>
        <span>Logs child</span>
      </LogsWorkspaceProvider>,
    );

    expect(mocks.workspaceInput).toHaveBeenLastCalledWith(
      expect.objectContaining({
        targetScope: {
          modelType: "linears",
          model: "linear",
          preset: "BASELINE",
          datasets: ["Mnist"],
        },
      }),
    );
  });
});
