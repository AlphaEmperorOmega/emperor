import { render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  chartViewModel: vi.fn(),
  includeStartedExperiment: vi.fn(),
  workspaceInput: vi.fn(),
  protectedAccessReady: true,
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
  useLogsChartViewModel: mocks.chartViewModel,
}));

vi.mock("@/features/workbench/state/logs/_use-logs-workspace-state", () => ({
  useLogsWorkspaceState: (input: {
    enabled: boolean;
    logDeletionEnabled?: boolean;
  }) => {
    mocks.workspaceInput(input);
    const noop = vi.fn();
    return {
      browser: {
        filters: Object.fromEntries(
          ["experiments", "datasets", "models", "presets", "tags"].map(
            (key) => [key, { options: [], selectedValues: [] }],
          ),
        ),
        status: {
          isScanning: false,
          isRefreshing: false,
          runsError: null,
          experimentsError: null,
          tagsError: null,
        },
        results: { hasExperiments: false, hasRuns: false },
        pagination: {
          runs: {
            loaded: 0,
            total: 0,
            canLoadMore: false,
            isLoadingMore: false,
          },
          scalarTags: {
            loadedRuns: 0,
            totalRuns: 0,
            canLoadMore: false,
            isLoadingMore: false,
          },
        },
        actions: {
          toggleFilter: noop,
          selectAll: noop,
          selectNone: noop,
          refresh: () => Promise.resolve(),
          loadMoreRuns: noop,
          loadMoreScalarTags: noop,
        },
      },
      charts: {
        enabled: input.enabled,
        visibleRuns: [],
        visibleRunIds: [],
        runsLoading: false,
        hasMoreRuns: false,
        tagRecords: [],
        tagsLoading: false,
        tagsFetching: false,
        tagsRefreshing: false,
        tagOptions: [],
        selectedTagList: [],
        confusionMatrixRateTags: [],
        collapsedMetricGroups: new Set<string>(),
        loadedScalarTagRunCount: 0,
        commands: {
          refresh: noop,
          openRunDetail: noop,
          setMetricGroupExpanded: noop,
          setTagSelected: noop,
        },
      },
      detail: {
        run: undefined,
        artifacts: undefined,
        status: { isLoading: false, error: null },
      },
      deletion: {
        enabled: true,
        presetTargetExperiment: null,
        operation: null,
        actions: {
          openExperiment: noop,
          openPreset: noop,
          cancel: noop,
          retryPlan: noop,
          confirm: noop,
        },
      },
      commands: {
        includeStartedExperiment: mocks.includeStartedExperiment,
        clearForConnectionChange: noop,
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
  mocks.chartViewModel.mockReset().mockReturnValue({
    content: {},
    settings: {},
    status: {},
    actions: {},
  });
  mocks.includeStartedExperiment.mockReset();
  mocks.workspaceInput.mockReset();
  mocks.protectedAccessReady = true;
});

describe("LogsWorkspaceProvider", () => {
  it("retains the charts context while gating work to active Logs", () => {
    const rendered = render(
      <LogsWorkspaceProvider enabled={false}>
        <span>Logs child</span>
      </LogsWorkspaceProvider>,
    );
    expect(mocks.chartViewModel).toHaveBeenCalledTimes(1);
    expect(mocks.chartViewModel).toHaveBeenLastCalledWith(
      expect.objectContaining({ enabled: false }),
    );

    rendered.rerender(
      <LogsWorkspaceProvider enabled>
        <span>Logs child</span>
      </LogsWorkspaceProvider>,
    );
    expect(mocks.chartViewModel).toHaveBeenCalledTimes(2);
    expect(mocks.chartViewModel).toHaveBeenLastCalledWith(
      expect.objectContaining({ enabled: true }),
    );

    rendered.rerender(
      <LogsWorkspaceProvider enabled={false}>
        <span>Logs child</span>
      </LogsWorkspaceProvider>,
    );
    expect(mocks.chartViewModel).toHaveBeenCalledTimes(3);
    expect(mocks.chartViewModel).toHaveBeenLastCalledWith(
      expect.objectContaining({ enabled: false }),
    );

    rendered.rerender(
      <LogsWorkspaceProvider enabled>
        <span>Logs child</span>
      </LogsWorkspaceProvider>,
    );
    expect(mocks.chartViewModel).toHaveBeenCalledTimes(4);
    expect(mocks.chartViewModel).toHaveBeenLastCalledWith(
      expect.objectContaining({ enabled: true }),
    );
  });

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

  it("passes only Logs activity and deletion capability to workspace state", () => {
    render(
      <LogsWorkspaceProvider enabled>
        <span>Logs child</span>
      </LogsWorkspaceProvider>,
    );

    expect(mocks.workspaceInput).toHaveBeenLastCalledWith(
      { enabled: true, logDeletionEnabled: true },
    );
  });
});
