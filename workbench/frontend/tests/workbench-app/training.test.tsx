import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useEffect, useMemo, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LogsWorkspaceProvider } from "@/features/workbench/providers/logs-workspace-provider";
import {
  useModelPackageInspection,
  WorkbenchProviders,
} from "@/features/workbench/providers/workbench-providers";
import { useWorkbenchConnection } from "@/features/workbench/providers/workbench-connection-provider";
import {
  useActiveTrainingJob,
  useTrainingConfiguration,
  useTrainingWorkspace,
} from "@/features/workbench/providers/training-provider";
import { trainingQueryKeys } from "@/lib/query-keys";
import {
  WorkbenchWorkspaceMain,
  WorkbenchWorkspaceOverlays,
} from "@/features/workbench/components/workbench-workspaces";
import { type WorkbenchWorkspace } from "@/types/workbench";
import {
  capabilitiesResponse,
  commandField,
  deferred,
  expandedTrainingDetails,
  expandedTrainingDetailsReady,
  fullConfigSearchPopup,
  fullConfigSearchResultRow,
  inspectResponse,
  installFetchMock,
  mockTrainingJobPayload,
  openFullConfig,
  openTrainingMultiSelect,
  renderWorkbench,
  resetWorkbenchAppTestState,
  schemaResponse,
  schemaResponseWithDescriptions,
  searchSpaceResponse,
  presetsResponse,
  selectExistingTrainingLogFolder,
  selectNewTrainingLogFolder,
  selectTargetOption,
  selectTrainingMonitorOption,
  selectTrainingTargetOption,
  setTargetHiddenDimOverride,
  setTrainingMultiSelectOption,
  typeConfigFieldValue,
  waitForTargetValue,
} from "./support";

function trainingRunPlanCalls(
  fetchMock: ReturnType<typeof installFetchMock>["fetchMock"],
) {
  return fetchMock.mock.calls.filter(([input]) =>
    String(input).endsWith("/training/run-plan"),
  );
}

function trainingJobPollCalls(
  fetchMock: ReturnType<typeof installFetchMock>["fetchMock"],
) {
  return fetchMock.mock.calls.filter(([input]) =>
    String(input).endsWith("/training/jobs/job-1"),
  );
}

function trainingRunPlanRequestBodies(
  fetchMock: ReturnType<typeof installFetchMock>["fetchMock"],
) {
  return trainingRunPlanCalls(fetchMock).map(([, init]) =>
    JSON.parse(String((init as RequestInit | undefined)?.body)),
  );
}

function modelCatalogCalls(
  fetchMock: ReturnType<typeof installFetchMock>["fetchMock"],
) {
  return fetchMock.mock.calls.filter(([input]) => String(input).endsWith("/models"));
}

const layerNormSearchAxis = {
  key: "stack_layer_norm_position",
  configKey: "STACK_LAYER_NORM_POSITION",
  searchKey: "SEARCH_SPACE_STACK_LAYER_NORM_POSITION",
  label: "stack layer norm position",
  section: "Layer Stack Options",
  type: "enum",
  values: ["BEFORE", "AFTER"],
  locked: false,
  lockedValue: null,
  lockedReason: "",
  lockedByPresets: [],
  lockReasons: [],
};

const searchLockPresetsResponse = {
  ...presetsResponse,
  presets: [
    ...presetsResponse.presets,
    { name: "post-norm", label: "POST_NORM", description: "Post norm" },
    { name: "gating", label: "GATING", description: "Gating" },
  ],
};

const largePresetOptions = [
  presetsResponse.presets[0],
  ...Array.from({ length: 69 }, (_, index) => {
    const number = index + 2;
    return {
      name: `adaptive-preset-${number}`,
      label: `ADAPTIVE_PRESET_${number}`,
      description: `Adaptive preset ${number}`,
    };
  }),
];

const largePresetsResponse = {
  ...presetsResponse,
  presets: largePresetOptions,
};

function searchSpaceWithPresetLocks(url: string) {
  const selectedPresets = new Set(
    (new URL(url, "http://testserver").searchParams.get("presets") ?? "")
      .split(",")
      .filter(Boolean),
  );
  const hiddenLocked = selectedPresets.has("gating");
  const layerNormLocked = selectedPresets.has("post-norm");
  return {
    ...searchSpaceResponse,
    axes: [
      {
        ...searchSpaceResponse.axes[0],
        locked: hiddenLocked,
        lockedValue: hiddenLocked ? 128 : null,
        lockedReason: hiddenLocked
          ? "Locked by the GATING preset because this preset locks `hidden_dim`."
          : "",
        lockedByPresets: hiddenLocked ? ["GATING"] : [],
        lockReasons: hiddenLocked
          ? [
              "Locked by the GATING preset because this preset locks `hidden_dim`.",
            ]
          : [],
      },
      {
        ...layerNormSearchAxis,
        locked: layerNormLocked,
        lockedValue: layerNormLocked ? "AFTER" : null,
        lockedReason: layerNormLocked
          ? "Locked by the POST_NORM preset because this preset locks `layer_norm_position`."
          : "",
        lockedByPresets: layerNormLocked ? ["POST_NORM"] : [],
        lockReasons: layerNormLocked
          ? [
              "Locked by the POST_NORM preset because this preset locks `layer_norm_position`.",
            ]
          : [],
      },
    ],
  };
}

function renderWorkspaceOverlayHarness({
  activeWorkspace,
  children,
}: {
  activeWorkspace: WorkbenchWorkspace;
  children?: ReactNode;
}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  const fullConfigDialog = {
    isOpen: false,
    mode: "default" as const,
    scope: "model" as const,
    open: vi.fn(),
    close: vi.fn(),
  };
  const featureListDialog = {
    isOpen: false,
    open: vi.fn(),
    close: vi.fn(),
  };
  const apiConnectionDialog = {
    isOpen: false,
    open: vi.fn(),
    close: vi.fn(),
  };
  const importLogsDialog = {
    isOpen: false,
    open: vi.fn(),
    close: vi.fn(),
  };

  return render(
    <QueryClientProvider client={queryClient}>
      <WorkbenchProviders
        activeWorkspace={activeWorkspace}
        onOpenFullConfig={fullConfigDialog.open}
      >
        <LogsWorkspaceProvider enabled={activeWorkspace === "logs"}>
          {children}
          {activeWorkspace === "training" && (
            <WorkbenchWorkspaceMain
              activeWorkspace={activeWorkspace}
            />
          )}
          <WorkbenchWorkspaceOverlays
            activeWorkspace={activeWorkspace}
            fullConfigDialog={fullConfigDialog}
            featureListDialog={featureListDialog}
            apiConnectionDialog={apiConnectionDialog}
            importLogsDialog={importLogsDialog}
          />
        </LogsWorkspaceProvider>
      </WorkbenchProviders>
    </QueryClientProvider>,
  );
}

function useTrainingTestInterface() {
  const workspace = useTrainingWorkspace();
  const { setup, logFolder, runtimeDefaults, searchMetadata, status } =
    workspace.draft;
  return useMemo(
    () => ({
      ...workspace,
      draft: {
        ...workspace.draft,
        selectedModelType: setup.model.selectedType,
        selectedModel: setup.model.selected,
        selectedPrimaryPreset: setup.variants.primaryPreset,
        selectedPresets: setup.variants.selectedPresets,
        selectedExperimentTask: setup.experimentTask.selected,
        selectedSnapshotIds: setup.variants.selectedSnapshotIds,
        selectedDatasets: setup.datasets.selected,
        bulkOverrides: runtimeDefaults.active,
        configSnapshots: setup.variants.snapshots,
        configSnapshotMutation: setup.variants.snapshotMutation,
        datasetOptions: setup.datasets.options,
        experimentTaskOptions: setup.experimentTask.options,
        monitorOptions: setup.monitors.options,
        selectedMonitors: setup.monitors.selected,
        monitorsLoading: setup.monitors.isLoading,
        searchAxes: searchMetadata.axes,
        searchLoading: searchMetadata.isLoading,
        trainingEnabled: status.trainingEnabled,
        canOpenFullConfig: status.canOpenFullConfig,
        activeConfigSnapshotCount: status.activeConfigSnapshotCount,
        selectedPresetCount: status.selectedPresetCount,
        logFolder: logFolder.state,
      },
      actions: {
        ...workspace.actions,
        selectModelType: setup.model.selectType,
        selectModel: setup.model.select,
        updateSearch: searchMetadata.update,
        selectLogFolderMode: logFolder.actions.selectMode,
        selectExistingLogFolder: logFolder.actions.selectExisting,
        nameNewLogFolder: logFolder.actions.nameNew,
      },
    }),
    [logFolder, runtimeDefaults.active, searchMetadata, setup, status, workspace],
  );
}

type TrainingInterfaceSnapshot = {
  workspace: ReturnType<typeof useTrainingTestInterface>;
  configuration: ReturnType<typeof useTrainingConfiguration>;
  modelTarget: ReturnType<typeof useModelPackageInspection>;
  activeJob: ReturnType<typeof useActiveTrainingJob>;
  connection: ReturnType<typeof useWorkbenchConnection>;
};

function TrainingInterfaceProbe({
  onChange,
}: {
  onChange: (snapshot: TrainingInterfaceSnapshot) => void;
}) {
  const workspace = useTrainingTestInterface();
  const configuration = useTrainingConfiguration();
  const modelTarget = useModelPackageInspection();
  const activeJob = useActiveTrainingJob();
  const connection = useWorkbenchConnection();

  useEffect(() => {
    onChange({
      activeJob,
      connection,
      configuration,
      workspace,
      modelTarget,
    });
  }, [
    activeJob,
    connection,
    configuration,
    modelTarget,
    onChange,
    workspace,
  ]);

  return null;
}

function renderTrainingInterfaceHarness({
  activeWorkspace = "training",
  onChange,
}: {
  activeWorkspace?: WorkbenchWorkspace;
  onChange: (snapshot: TrainingInterfaceSnapshot) => void;
}) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  const renderTree = (workspace: WorkbenchWorkspace) => (
    <QueryClientProvider client={queryClient}>
      <WorkbenchProviders activeWorkspace={workspace}>
        <TrainingInterfaceProbe onChange={onChange} />
      </WorkbenchProviders>
    </QueryClientProvider>
  );
  const rendered = render(renderTree(activeWorkspace));

  return {
    ...rendered,
    rerenderWorkspace(workspace: WorkbenchWorkspace) {
      rendered.rerender(renderTree(workspace));
    },
  };
}

function TargetTrainingInputsReady({ onReady }: { onReady: () => void }) {
  const training = useTrainingTestInterface();
  const draft = training.draft;

  useEffect(() => {
    if (
      draft.selectedModel &&
      draft.selectedPrimaryPreset &&
      draft.datasetOptions.length > 0 &&
      draft.canOpenFullConfig &&
      !draft.monitorsLoading &&
      !draft.searchLoading
    ) {
      onReady();
    }
  }, [
    draft.canOpenFullConfig,
    draft.datasetOptions.length,
    draft.monitorsLoading,
    draft.searchLoading,
    draft.selectedModel,
    draft.selectedPrimaryPreset,
    onReady,
  ]);

  return null;
}

function ModelInputsReady({ onReady }: { onReady: () => void }) {
  const model = useModelPackageInspection();

  useEffect(() => {
    if (
      model.browser.selectedModel &&
      model.browser.selectedPreset &&
      model.target.datasets.length > 0 &&
      model.status.schema.isReady
    ) {
      onReady();
    }
  }, [
    model.browser.selectedModel,
    model.browser.selectedPreset,
    model.target.datasets.length,
    model.status.schema.isReady,
    onReady,
  ]);

  return null;
}

async function waitForTargetTrainingInputs(onReady: () => void) {
  await waitFor(() => {
    expect(onReady).toHaveBeenCalled();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

function trainingRunList(details: HTMLElement) {
  return within(details).getByRole("main", {
    name: "Training Run List",
  });
}

async function openTrainingFullConfig(
  user: ReturnType<typeof userEvent.setup>,
  details: HTMLElement,
) {
  await user.click(
    within(trainingRunList(details)).getByRole("button", {
      name: /open full config/i,
    }),
  );
  return screen.findByRole("dialog", {
    name: /training full configuration/i,
  });
}

async function setTrainingHiddenDimOverride(
  user: ReturnType<typeof userEvent.setup>,
  details: HTMLElement,
  value: string,
) {
  const dialog = await openTrainingFullConfig(user, details);
  await typeConfigFieldValue(user, dialog, /hidden dim/i, value);
  await user.click(within(dialog).getByRole("button", { name: /^done$/i }));
  return details;
}

async function findTrainingRunSummary(name: RegExp) {
  return screen.findByRole("status", {
    name,
  });
}

describe("WorkbenchApp Training And Preview", () => {
  beforeEach(resetWorkbenchAppTestState);

  it("mounts Training as its own workspace", async () => {
    installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    expect(screen.queryByRole("button", { name: /start training/i }))
      .not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^logs$/i }));

    expect(await screen.findByText("Historical Scalars")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /start training/i }))
      .not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^training$/i }));

    expect(await screen.findByRole("heading", { name: /^training$/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /start training/i }))
      .toBeInTheDocument();
  });

  it("requests training run plans only while the Training workspace is mounted", async () => {
    const { fetchMock } = installFetchMock();
    const modelReady = vi.fn();

    const modelRender = renderWorkspaceOverlayHarness({
      activeWorkspace: "model",
      children: <ModelInputsReady onReady={modelReady} />,
    });
    await waitForTargetTrainingInputs(modelReady);
    expect(trainingRunPlanCalls(fetchMock)).toHaveLength(0);
    modelRender.unmount();
    fetchMock.mockClear();

    const trainingReady = vi.fn();
    const trainingRender = renderWorkspaceOverlayHarness({
      activeWorkspace: "training",
      children: <TargetTrainingInputsReady onReady={trainingReady} />,
    });
    await waitForTargetTrainingInputs(trainingReady);
    await waitFor(() => {
      expect(trainingRunPlanCalls(fetchMock).length).toBeGreaterThan(0);
    });
    trainingRender.unmount();
    fetchMock.mockClear();

    const logsReady = vi.fn();
    const logsRender = renderWorkspaceOverlayHarness({
      activeWorkspace: "logs",
      children: <ModelInputsReady onReady={logsReady} />,
    });
    await waitForTargetTrainingInputs(logsReady);
    expect(modelCatalogCalls(fetchMock).length).toBeGreaterThan(0);
    expect(trainingRunPlanCalls(fetchMock)).toHaveLength(0);

    logsRender.unmount();
  });

  it("submits the current backend Run plan through the Training Interface", async () => {
    const { trainingBodies } = installFetchMock({
      trainingRunPlanResponseFactory: (_request, _requestIndex, defaultPlan) => ({
        ...defaultPlan,
        runs: defaultPlan.runs.map((run, index) =>
          index === 0
            ? {
                ...run,
                id: "server-sentinel-run",
                command: "server-sentinel-command --authoritative",
              }
            : run,
        ),
      }),
    });
    let current: TrainingInterfaceSnapshot | undefined;
    renderTrainingInterfaceHarness({
      onChange: (snapshot) => {
        current = snapshot;
      },
    });

    await waitFor(() => {
      expect(current?.workspace.plan.display?.runs[0]).toMatchObject({
        id: "server-sentinel-run",
        command: "server-sentinel-command --authoritative",
      });
    });
    act(() => {
      current?.workspace.actions.selectLogFolderMode("new");
      current?.workspace.actions.nameNewLogFolder("interface_tracer");
    });
    await waitFor(() => {
      expect(current?.workspace.plan.canStart).toBe(true);
    });

    act(() => {
      current?.workspace.actions.startJob();
    });

    await waitFor(() => {
      expect(trainingBodies).toHaveLength(1);
    });
    expect(trainingBodies[0]).toHaveProperty(
      "runPlan.runs.0.id",
      "server-sentinel-run",
    );
    expect(trainingBodies[0]).not.toHaveProperty("runPlan.runs.0.command");
    expect(trainingBodies[0]).not.toHaveProperty("runPlan.runs.0.status");
    expect(trainingBodies[0]).not.toHaveProperty("runPlan.summary");
  });

  it("exposes only the five-part Training workspace Interface and focused projections", async () => {
    installFetchMock();
    let current: TrainingInterfaceSnapshot | undefined;
    renderTrainingInterfaceHarness({
      onChange: (snapshot) => {
        current = snapshot;
      },
    });

    await waitFor(() => {
      expect(current?.workspace.plan.display).toBeDefined();
    });
    expect(Object.keys(current?.workspace ?? {}).sort()).toEqual([
      "actions",
      "dialogs",
      "draft",
      "job",
      "plan",
    ]);
    expect(current?.workspace).not.toHaveProperty("input");
    expect(current?.workspace).not.toHaveProperty("training");
    expect(current?.workspace.draft).not.toHaveProperty(
      "clearForConnectionChange",
    );
    expect(current?.workspace.actions).not.toHaveProperty("setSearch");
    expect(current?.configuration).not.toHaveProperty(
      "clearForConnectionChange",
    );
    expect(current?.configuration).not.toHaveProperty("selectModelType");
    expect(current?.configuration).not.toHaveProperty("updateSearch");
    expect(Object.keys(current?.activeJob ?? {}).sort()).toEqual([
      "activeTrainingJob",
    ]);
  });

  it("retries a failed backend Run plan through a semantic Training action", async () => {
    let shouldFail = true;
    const { fetchMock } = installFetchMock({
      trainingRunPlanResponseFactory: (_request, _requestIndex, defaultPlan) =>
        shouldFail
          ? Promise.reject(new Error("run plan unavailable"))
          : defaultPlan,
    });
    renderWorkbench();
    const user = userEvent.setup();
    await expandedTrainingDetails(user);

    const retryButton = await screen.findByRole("button", {
      name: /retry plan/i,
    });
    expect(screen.getAllByText("run plan unavailable").length).toBeGreaterThan(0);
    const failedRequestCount = trainingRunPlanCalls(fetchMock).length;

    shouldFail = false;
    await user.click(retryButton);

    await findTrainingRunSummary(
      /0\s*\/\s*1 runs;\s*0\s*\/\s*30 epochs/i,
    );
    expect(trainingRunPlanCalls(fetchMock).length).toBeGreaterThan(
      failedRequestCount,
    );
    expect(
      screen.queryByRole("button", { name: /retry plan/i }),
    ).not.toBeInTheDocument();
  });

  it("keeps an obsolete Run plan response from replacing the current draft plan", async () => {
    const obsoletePlan = deferred<unknown>();
    let obsoleteResponse: unknown;
    const { fetchMock } = installFetchMock({
      trainingRunPlanResponseFactory: (request, _requestIndex, defaultPlan) => {
        if (request.logFolder === "obsolete_plan") {
          obsoleteResponse = defaultPlan;
          return obsoletePlan.promise;
        }
        if (request.logFolder === "current_plan") {
          return {
            ...defaultPlan,
            runs: defaultPlan.runs.map((run, index) =>
              index === 0
                ? { ...run, command: "current-plan --authoritative" }
                : run,
            ),
          };
        }
        return defaultPlan;
      },
    });
    let current: TrainingInterfaceSnapshot | undefined;
    renderTrainingInterfaceHarness({
      onChange: (snapshot) => {
        current = snapshot;
      },
    });

    await waitFor(() => {
      expect(current?.workspace.plan.display).toBeDefined();
    });
    act(() => {
      current?.workspace.actions.selectLogFolderMode("new");
      current?.workspace.actions.nameNewLogFolder("obsolete_plan");
    });
    await waitFor(() => {
      expect(
        trainingRunPlanRequestBodies(fetchMock).some(
          (request) => request.logFolder === "obsolete_plan",
        ),
      ).toBe(true);
    });
    const obsoleteCall = trainingRunPlanCalls(fetchMock).find(([, init]) =>
      String((init as RequestInit | undefined)?.body).includes(
        '"logFolder":"obsolete_plan"',
      ),
    );

    act(() => {
      current?.workspace.actions.nameNewLogFolder("current_plan");
    });
    await waitFor(() => {
      expect(current?.workspace.plan.display).toMatchObject({
        logFolder: "current_plan",
      });
      expect(
        current?.workspace.plan.display?.runs[0]?.command,
      ).toBe("current-plan --authoritative");
    });
    expect(
      (obsoleteCall?.[1] as RequestInit | undefined)?.signal?.aborted,
    ).toBe(true);

    await act(async () => {
      obsoletePlan.resolve(obsoleteResponse);
      await obsoletePlan.promise;
    });

    expect(current?.workspace.plan.display).toMatchObject({
      logFolder: "current_plan",
    });
    expect(current?.workspace.plan.display?.runs[0]?.command).toBe(
      "current-plan --authoritative",
    );
  });

  it("resampling replaces only the draft Run plan and submits the replacement", async () => {
    let randomPlanResponseCount = 0;
    const { trainingBodies } = installFetchMock({
      trainingRunPlanResponseFactory: (request, _requestIndex, defaultPlan) => {
        if (request.search?.mode !== "random") {
          return defaultPlan;
        }
        randomPlanResponseCount += 1;
        return {
          ...defaultPlan,
          runs: defaultPlan.runs.map((run, index) =>
            index === 0
              ? {
                  ...run,
                  command: `random-plan-${randomPlanResponseCount} --authoritative`,
                }
              : run,
          ),
        };
      },
    });
    let current: TrainingInterfaceSnapshot | undefined;
    renderTrainingInterfaceHarness({
      onChange: (snapshot) => {
        current = snapshot;
      },
    });

    await waitFor(() => {
      expect(current?.workspace.draft.searchLoading).toBe(false);
      expect(current?.workspace.plan.display).toBeDefined();
    });
    act(() => {
      current?.workspace.actions.updateSearch({
        mode: "random",
        selectedValues: { hidden_dim: [64, 128] },
        randomSamples: 2,
      });
      current?.workspace.actions.selectLogFolderMode("new");
      current?.workspace.actions.nameNewLogFolder("resampled_plan");
    });
    await waitFor(() => {
      expect(current?.workspace.plan.canResample).toBe(true);
      expect(
        current?.workspace.plan.display?.runs[0]?.command,
      ).toMatch(/^random-plan-\d+ --authoritative$/);
    });
    const originalCommand =
      current?.workspace.plan.display?.runs[0]?.command;

    act(() => {
      current?.workspace.actions.resamplePlan();
    });
    await waitFor(() => {
      expect(
        current?.workspace.plan.display?.runs[0]?.command,
      ).toMatch(/^random-plan-\d+ --authoritative$/);
      expect(
        current?.workspace.plan.display?.runs[0]?.command,
      ).not.toBe(originalCommand);
      expect(current?.workspace.plan.canStart).toBe(true);
    });
    const replacementRunId = current?.workspace.plan.display?.runs[0]?.id;

    act(() => {
      current?.workspace.actions.startJob();
    });
    await waitFor(() => {
      expect(trainingBodies).toHaveLength(1);
    });
    expect(trainingBodies[0]).toHaveProperty(
      "runPlan.runs.0.id",
      replacementRunId,
    );
    expect(trainingBodies[0]).not.toHaveProperty("runPlan.runs.0.command");
  });

  it("rejects a large-grid confirmation retained from an obsolete draft revision", async () => {
    const largeSearchSpace = {
      ...searchSpaceResponse,
      axes: [
        {
          ...searchSpaceResponse.axes[0],
          values: Array.from({ length: 11 }, (_, index) => index + 1),
        },
        {
          ...searchSpaceResponse.axes[1],
          values: Array.from({ length: 10 }, (_, index) => index + 1),
        },
      ],
    };
    const { trainingBodies } = installFetchMock({
      searchSpaceResponse: largeSearchSpace,
    });
    let current: TrainingInterfaceSnapshot | undefined;
    renderTrainingInterfaceHarness({
      onChange: (snapshot) => {
        current = snapshot;
      },
    });

    await waitFor(() => {
      expect(current?.workspace.draft.searchLoading).toBe(false);
      expect(current?.workspace.plan.display).toBeDefined();
    });
    act(() => {
      current?.workspace.actions.updateSearch({
        mode: "grid",
        selectedValues: {
          hidden_dim: largeSearchSpace.axes[0].values,
          stack_num_layers: largeSearchSpace.axes[1].values,
        },
        randomSamples: 10,
      });
      current?.workspace.actions.selectLogFolderMode("new");
      current?.workspace.actions.nameNewLogFolder("large_grid_revision_one");
    });
    await waitFor(() => {
      expect(current?.workspace.plan.displayedRunCount).toBe(110);
      expect(current?.workspace.plan.canStart).toBe(true);
    });
    act(() => {
      current?.workspace.actions.startJob();
    });
    await waitFor(() => {
      expect(current?.workspace.dialogs.largeGridConfirmation.isOpen).toBe(true);
    });
    const obsoleteConfirm =
      current?.workspace.actions.confirmLargeGridSearch;

    act(() => {
      current?.workspace.actions.nameNewLogFolder("large_grid_revision_two");
    });
    await waitFor(() => {
      expect(current?.workspace.dialogs.largeGridConfirmation.isOpen).toBe(false);
      expect(current?.workspace.plan.display?.logFolder).toBe(
        "large_grid_revision_two",
      );
    });

    act(() => {
      obsoleteConfirm?.();
    });
    expect(trainingBodies).toHaveLength(0);
  });

  it("keeps a failed create out of active Job state and permits a fresh create", async () => {
    let shouldFail = true;
    const { trainingBodies } = installFetchMock({
      createTrainingJobResponseFactory: (request) =>
        shouldFail
          ? Promise.reject(new Error("training create unavailable"))
          : mockTrainingJobPayload(
              request as Parameters<typeof mockTrainingJobPayload>[0],
              { status: "running" },
            ),
    });
    let current: TrainingInterfaceSnapshot | undefined;
    renderTrainingInterfaceHarness({
      onChange: (snapshot) => {
        current = snapshot;
      },
    });

    await waitFor(() => {
      expect(current?.workspace.plan.display).toBeDefined();
    });
    act(() => {
      current?.workspace.actions.selectLogFolderMode("new");
      current?.workspace.actions.nameNewLogFolder("create_retry");
    });
    await waitFor(() => {
      expect(current?.workspace.plan.canStart).toBe(true);
    });
    act(() => {
      current?.workspace.actions.startJob();
    });
    await waitFor(() => {
      expect(current?.workspace.job.error).toBe(
        "training create unavailable",
      );
    });
    expect(current?.workspace.job.value).toBeUndefined();

    shouldFail = false;
    act(() => {
      current?.workspace.actions.startJob();
    });
    await waitFor(() => {
      expect(trainingBodies).toHaveLength(2);
      expect(current?.workspace.job.value).toMatchObject({
        id: "job-1",
        logFolder: "create_retry",
      });
      expect(current?.workspace.job.error).toBe("");
    });
  });

  it("does not rewrite an established Training draft after a Model workspace change", async () => {
    installFetchMock();
    let current: TrainingInterfaceSnapshot | undefined;
    renderTrainingInterfaceHarness({
      onChange: (snapshot) => {
        current = snapshot;
      },
    });

    await waitFor(() => {
      expect(current?.workspace.draft.selectedModelType).toBe("linears");
      expect(current?.workspace.draft.selectedModel).toBe("linear");
      expect(current?.workspace.draft.selectedPrimaryPreset).toBe("baseline");
    });
    act(() => {
      current?.modelTarget.actions.selectModelType("bert");
    });

    await waitFor(() => {
      expect(current?.modelTarget.browser.selectedModelType).toBe("bert");
    });
    expect(current?.workspace.draft).toMatchObject({
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPrimaryPreset: "baseline",
    });
  });

  it("seeds the Training draft from the current Model target on first open", async () => {
    installFetchMock();
    let current: TrainingInterfaceSnapshot | undefined;
    const rendered = renderTrainingInterfaceHarness({
      activeWorkspace: "model",
      onChange: (snapshot) => {
        current = snapshot;
      },
    });

    await waitFor(() => {
      expect(current?.modelTarget.browser.selectedModelType).toBe("linears");
      expect(current?.workspace.draft.selectedModel).toBe("");
    });
    act(() => {
      current?.modelTarget.actions.selectModelType("bert");
    });
    await waitFor(() => {
      expect(current?.modelTarget.browser.selectedModelType).toBe("bert");
      expect(current?.modelTarget.browser.selectedPreset).toBe("bert-baseline");
    });

    rendered.rerenderWorkspace("training");

    await waitFor(() => {
      expect(current?.workspace.draft).toMatchObject({
        selectedModelType: "bert",
        selectedModel: "linear",
        selectedPrimaryPreset: "bert-baseline",
        selectedDatasets: ["ToyText"],
      });
    });
  });

  it("retains an active Training Job across draft and workspace changes", async () => {
    installFetchMock({ trainingJobStatus: "running" });
    let current: TrainingInterfaceSnapshot | undefined;
    const rendered = renderTrainingInterfaceHarness({
      onChange: (snapshot) => {
        current = snapshot;
      },
    });

    await waitFor(() => {
      expect(current?.workspace.plan.display).toBeDefined();
    });
    act(() => {
      current?.workspace.actions.selectLogFolderMode("new");
      current?.workspace.actions.nameNewLogFolder("retained_job");
    });
    await waitFor(() => {
      expect(current?.workspace.plan.canStart).toBe(true);
    });
    act(() => {
      current?.workspace.actions.startJob();
    });
    await waitFor(() => {
      expect(current?.workspace.job.value?.id).toBe("job-1");
    });

    act(() => {
      current?.workspace.actions.selectModelType("experts");
    });
    await waitFor(() => {
      expect(current?.workspace.draft.selectedModelType).toBe("experts");
    });
    expect(current?.workspace.job.value?.id).toBe("job-1");

    rendered.rerenderWorkspace("model");

    await waitFor(() => {
      expect(current?.workspace.job.value?.id).toBe("job-1");
    });
    expect(current?.workspace.draft.selectedModelType).toBe("experts");
  });

  it("clears Training internals through the connection composition command", async () => {
    installFetchMock({ trainingJobStatus: "running" });
    let current: TrainingInterfaceSnapshot | undefined;
    renderTrainingInterfaceHarness({
      onChange: (snapshot) => {
        current = snapshot;
      },
    });

    await waitFor(() => {
      expect(current?.workspace.plan.display).toBeDefined();
    });
    act(() => {
      current?.workspace.actions.updateSearch({
        mode: "random",
        selectedValues: { hidden_dim: [64, 128] },
        randomSamples: 2,
      });
      current?.workspace.actions.selectLogFolderMode("new");
      current?.workspace.actions.nameNewLogFolder("connection_owned_clear");
    });
    await waitFor(() => {
      expect(current?.workspace.plan.canStart).toBe(true);
    });
    act(() => {
      current?.workspace.actions.startJob();
    });
    await waitFor(() => {
      expect(current?.activeJob.activeTrainingJob?.status).toBe("running");
      expect(current?.workspace.plan.search.effective.mode).toBe(
        "random",
      );
    });

    await act(async () => {
      await current?.connection.actions.useApiBaseUrl(
        "https://alternate-workbench.example.test",
      );
    });

    await waitFor(() => {
      expect(current?.activeJob.activeTrainingJob).toBeUndefined();
      expect(current?.activeJob).not.toHaveProperty("setActiveJobId");
      expect(current?.activeJob).not.toHaveProperty("onJobChange");
      expect(current?.workspace.job.value).toBeUndefined();
      expect(current?.workspace.draft.logFolder).toMatchObject({
        mode: "existing",
        existingValue: "",
        newValue: "",
      });
      expect(current?.workspace.plan.search.effective.mode).toBe(
        "off",
      );
    });
  });

  it("ignores a create response that completes after a connection change", async () => {
    const pendingCreate = deferred<unknown>();
    let createRequest: Parameters<typeof mockTrainingJobPayload>[0] | undefined;
    const { trainingBodies } = installFetchMock({
      createTrainingJobResponseFactory: (request) => {
        createRequest = request as Parameters<typeof mockTrainingJobPayload>[0];
        return pendingCreate.promise;
      },
    });
    let current: TrainingInterfaceSnapshot | undefined;
    renderTrainingInterfaceHarness({
      onChange: (snapshot) => {
        current = snapshot;
      },
    });

    await waitFor(() => {
      expect(current?.workspace.plan.display).toBeDefined();
    });
    act(() => {
      current?.workspace.actions.selectLogFolderMode("new");
      current?.workspace.actions.nameNewLogFolder("obsolete_create");
    });
    await waitFor(() => {
      expect(current?.workspace.plan.canStart).toBe(true);
    });
    act(() => {
      current?.workspace.actions.startJob();
    });
    await waitFor(() => {
      expect(trainingBodies).toHaveLength(1);
      expect(current?.workspace.job.isStarting).toBe(true);
    });

    await act(async () => {
      await current?.connection.actions.useApiBaseUrl(
        "https://new-connection.example.test",
      );
    });
    await act(async () => {
      pendingCreate.resolve(
        mockTrainingJobPayload(createRequest ?? {}, { status: "running" }),
      );
      await pendingCreate.promise;
    });

    await waitFor(() => {
      expect(current?.activeJob.activeTrainingJob).toBeUndefined();
      expect(current?.workspace.job.value).toBeUndefined();
      expect(current?.workspace.job.isStarting).toBe(false);
    });
  });

  it("keeps active job polling mounted while the Training workspace is hidden", async () => {
    const { fetchMock } = installFetchMock({ trainingJobStatus: "running" });
    let current: TrainingInterfaceSnapshot | undefined;
    const rendered = renderTrainingInterfaceHarness({
      onChange: (snapshot) => {
        current = snapshot;
      },
    });

    await waitFor(() => {
      expect(current?.workspace.plan.display).toBeDefined();
    });
    act(() => {
      current?.workspace.actions.selectLogFolderMode("new");
      current?.workspace.actions.nameNewLogFolder("hidden_polling");
    });
    await waitFor(() => {
      expect(current?.workspace.plan.canStart).toBe(true);
    });
    act(() => {
      current?.workspace.actions.startJob();
    });
    await waitFor(() => {
      expect(current?.workspace.job.value?.status).toBe("running");
      expect(trainingJobPollCalls(fetchMock).length).toBeGreaterThan(0);
    });
    const visiblePollCount = trainingJobPollCalls(fetchMock).length;

    rendered.rerenderWorkspace("logs");

    await waitFor(
      () => {
        expect(trainingJobPollCalls(fetchMock).length).toBeGreaterThan(
          visiblePollCount,
        );
      },
      { timeout: 2500 },
    );
    expect(screen.queryByRole("button", { name: /start training/i }))
      .not.toBeInTheDocument();
  });

  it("keeps the running job cancellable when cancel fails", async () => {
    const { fetchMock } = installFetchMock({
      trainingJobStatus: "running",
      cancelTrainingJobError:
        "Training job 'job-1' process survived terminate and kill.",
    });
    renderWorkspaceOverlayHarness({
      activeWorkspace: "training",
    });
    const user = userEvent.setup();

    const details = await screen.findByRole("region", {
      name: "Training workspace",
    });
    await selectNewTrainingLogFolder(user, "cancel_failure");
    await user.click(screen.getByRole("button", { name: /start training/i }));
    const runList = trainingRunList(details);
    const cancelButton = await within(runList).findByRole("button", {
      name: /^cancel$/i,
    });
    expect(cancelButton).toBeEnabled();

    await user.click(cancelButton);

    await waitFor(() => {
      expect(
        within(details)
          .getAllByRole("alert")
          .some((alert) =>
            /process survived terminate and kill/i.test(alert.textContent ?? ""),
          ),
      ).toBe(true);
    });
    expect(fetchMock.mock.calls.some(([input]) =>
      String(input).endsWith("/training/jobs/job-1/cancel"),
    )).toBe(true);
    expect(within(runList).getByRole("button", { name: /^cancel$/i }))
      .toBeEnabled();
    expect(screen.getAllByText("running").length).toBeGreaterThan(0);
  });

  it("shows cancel errors in the Training workspace header", async () => {
    installFetchMock({
      trainingJobStatus: "running",
      cancelTrainingJobError:
        "Training job 'job-1' process survived terminate and kill.",
    });
    renderWorkspaceOverlayHarness({
      activeWorkspace: "training",
    });
    const user = userEvent.setup();

    await screen.findByRole("region", { name: "Training workspace" });
    await selectNewTrainingLogFolder(user, "cancel_header_failure");
    await user.click(screen.getByRole("button", { name: /start training/i }));
    await user.click(await screen.findByRole("button", { name: /^cancel$/i }));

    await waitFor(() => {
      expect(
        screen
          .getAllByRole("alert")
          .some((alert) =>
            /process survived terminate and kill/i.test(alert.textContent ?? ""),
          ),
      ).toBe(true);
    });
    expect(screen.getByRole("button", { name: /^cancel$/i })).toBeEnabled();
  });

  it("does not let stale running polls overwrite a cancelled mutation", async () => {
    const stalePoll = deferred<unknown>();
    const { fetchMock } = installFetchMock({
      trainingJobResponseFactory: (requestIndex) =>
        requestIndex === 0
          ? stalePoll.promise
          : mockTrainingJobPayload(
              { logFolder: "stale_cancel_poll", datasets: ["Mnist"] },
              { status: "cancelled" },
            ),
    });
    const { queryClient } = renderWorkbench();
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    const removeSpy = vi.spyOn(queryClient, "removeQueries");
    const user = userEvent.setup();

    await expandedTrainingDetailsReady(user);
    await selectNewTrainingLogFolder(user, "stale_cancel_poll");
    await user.click(screen.getByRole("button", { name: /start training/i }));
    await waitFor(() => {
      expect(trainingJobPollCalls(fetchMock)).toHaveLength(1);
    });
    await user.click(await screen.findByRole("button", { name: /^cancel$/i }));

    await waitFor(() => {
      expect(screen.getAllByText("cancelled").length).toBeGreaterThan(0);
    });
    await waitFor(() => {
      expect(
        queryClient.getQueryData(trainingQueryKeys.job("job-1")),
      ).toMatchObject({ status: "cancelled" });
      expect(invalidateSpy).toHaveBeenCalled();
      expect(removeSpy).toHaveBeenCalled();
    });
    invalidateSpy.mockClear();
    removeSpy.mockClear();
    const observedStatuses: string[] = [];
    const unsubscribe = queryClient.getQueryCache().subscribe((event) => {
      if (
        event.query.queryKey[0] === "training-job" &&
        event.query.queryKey[1] === "job-1"
      ) {
        const status = (
          event.query.state.data as { status?: string } | undefined
        )?.status;
        if (status) {
          observedStatuses.push(status);
        }
      }
    });
    stalePoll.resolve(
      mockTrainingJobPayload(
        { logFolder: "stale_cancel_poll", datasets: ["Mnist"] },
        { status: "running" },
      ),
    );

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    expect(observedStatuses).not.toContain("running");
    expect(
      queryClient.getQueryData(trainingQueryKeys.job("job-1")),
    ).toMatchObject({ status: "cancelled" });
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 1100));
    });
    unsubscribe();

    expect(trainingJobPollCalls(fetchMock)).toHaveLength(1);
    expect(invalidateSpy).not.toHaveBeenCalled();
    expect(removeSpy).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: /^cancel$/i }))
      .not.toBeInTheDocument();
  });

  it("renders preset-locked fields disabled with their reason", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: schemaResponse.fields.map((field) =>
          field.key === "gate_flag"
            ? {
                ...field,
                locked: true,
                lockedValue: true,
                lockedReason:
                  "Locked by the GATING preset because this preset locks `stack_gate_flag`.",
              }
            : field,
        ),
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const closeButton = within(dialog).getByRole("button", {
      name: /close full config/i,
    });
    const headerActions = closeButton.parentElement;
    if (!(headerActions instanceof HTMLElement)) {
      throw new Error("Expected full config close button to render in the header actions");
    }
    const headerPresetBadge = within(headerActions).getByText("1 preset");
    expect(headerPresetBadge).toHaveClass("text-amber");
    expect(closeButton.previousElementSibling).toBe(headerPresetBadge);

    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer hidden stack options section, 3 fields, 0 overrides/i,
    });
    const gateAccordion = within(dialog).getByRole("button", {
      name: /gate options section, 1 field, 0 overrides, 1 preset/i,
    });
    const layerSection = layerAccordion.closest("section");
    const gateSection = gateAccordion.closest("section");
    const layerJump = within(sectionNav).getByRole("button", {
      name: /jump to layer hidden stack options/i,
    });
    const gateJump = within(sectionNav).getByRole("button", {
      name: /jump to gate options/i,
    });
    const layerNavRow = layerJump.parentElement?.parentElement;
    const gateNavRow = gateJump.parentElement?.parentElement;
    if (
      !(layerSection instanceof HTMLElement) ||
      !(gateSection instanceof HTMLElement) ||
      !(layerNavRow instanceof HTMLElement) ||
      !(gateNavRow instanceof HTMLElement)
    ) {
      throw new Error("Expected full config sections and sidebar rows to render");
    }

    expect(gateNavRow).toHaveClass(
      "border-amber/30",
      "bg-amber/[0.055]",
      "hover:bg-amber/[0.09]",
    );
    expect(within(gateNavRow).getByText("1 preset")).toHaveClass("text-amber");
    expect(layerNavRow).not.toHaveClass("border-amber/30", "bg-amber/[0.055]");
    expect(within(layerNavRow).queryByText("1 preset")).not.toBeInTheDocument();
    expect(gateAccordion).toHaveClass("bg-amber/[0.08]", "hover:bg-amber/[0.12]");
    expect(gateSection).toHaveClass("border-amber/35", "bg-amber/[0.045]");
    expect(within(gateSection).getByText("1 preset")).toHaveClass("text-amber");
    expect(layerAccordion).not.toHaveClass("bg-amber/[0.08]");
    expect(layerSection).not.toHaveClass("border-amber/35", "bg-amber/[0.045]");
    expect(within(layerSection).queryByText("1 preset")).not.toBeInTheDocument();

    const gateSwitch = within(dialog).getByRole("switch", { name: /enabled/i });
    const presetBadge = within(gateSection).getByText("preset");

    expect(gateSwitch).toBeDisabled();
    expect(presetBadge).toHaveClass("text-amber");
    expect(within(gateSection).queryByText("override")).not.toBeInTheDocument();
    expect(within(dialog).getByText(/locked by the GATING preset/i)).toBeInTheDocument();

    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });
    await user.type(search, "gate");
    const searchPopup = fullConfigSearchPopup(dialog);
    const gateSearchRow = fullConfigSearchResultRow(searchPopup, /gate flag/i);
    const searchPresetBadge = within(gateSearchRow).getByText("preset");
    const searchGateControl = within(gateSearchRow).getByRole("radiogroup", {
      name: /current value/i,
    });
    const searchGateOn = within(searchGateControl).getByRole("radio", {
      name: "On",
    });
    const searchGateOff = within(searchGateControl).getByRole("radio", {
      name: "Off",
    });

    expect(gateSearchRow).toHaveTextContent(/current\s*true/i);
    expect(gateSearchRow).not.toHaveTextContent(/default\s*false/i);
    expect(searchPresetBadge).toHaveClass("text-amber");
    expect(searchGateControl).toHaveAttribute("aria-disabled", "true");
    expect(searchGateOn).toBeDisabled();
    expect(searchGateOff).toBeDisabled();
    expect(
      within(searchGateControl).queryByRole("radio", { name: "None" }),
    ).not.toBeInTheDocument();
    expect(
      within(gateSearchRow).queryByRole("switch", {
        name: /current value/i,
      }),
    ).not.toBeInTheDocument();
    expect(
      within(gateSearchRow).queryByRole("button", {
        name: /reset search result override/i,
      }),
    ).not.toBeInTheDocument();
  });

  it("Training workspace shows the three-zone setup, run list, and status layout", async () => {
    installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);

    const setupSidebar = within(details).getByRole("complementary", {
      name: "Training Setup Sidebar",
    });
    const runList = within(details).getByRole("main", {
      name: "Training Run List",
    });
    const statusSidebar = within(details).getByRole("complementary", {
      name: "Training Status Sidebar",
    });
    const trainingHeading = within(runList).getByRole("heading", {
      name: /^training$/i,
    });
    const firstRunIndex = await within(runList).findByText("#1");
    const trainingBar = runList.firstElementChild;
    if (!(trainingBar instanceof HTMLElement)) {
      throw new Error("Expected Training bar to render inside the run list");
    }
    expect(trainingBar.contains(trainingHeading)).toBe(true);
    expect(trainingBar).not.toHaveTextContent(/\/\s*Mnist/i);
    expect(trainingBar).not.toHaveTextContent(/No metrics/i);
    expect(trainingBar).not.toHaveTextContent(/planned run/i);
    expect(trainingBar).not.toHaveTextContent(/snapshots?/i);
    expect(trainingBar).not.toHaveTextContent(/Choose log folder/i);
    expect(within(runList).queryByText("Training Runs")).not.toBeInTheDocument();
    expect(
      within(trainingBar).getByRole("button", { name: /^commands$/i }),
    ).toBeInTheDocument();
    expect(
      within(trainingBar).getByRole("button", { name: /open full config/i }),
    ).toBeInTheDocument();
    expect(
      trainingHeading.compareDocumentPosition(firstRunIndex) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      within(runList).getByRole("button", { name: /start training/i }),
    ).toBeInTheDocument();
    expect(
      setupSidebar.compareDocumentPosition(runList) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      runList.compareDocumentPosition(statusSidebar) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      within(details).queryByRole("tablist", {
        name: "Training workspace sections",
      }),
    ).not.toBeInTheDocument();

    expect(within(setupSidebar).getByText("Log Folder")).toBeInTheDocument();
    expect(within(setupSidebar).queryByText("Target")).not.toBeInTheDocument();
    const experimentTaskHeading =
      within(setupSidebar).getByText("Experiment Task");
    const modelTypeHeading = within(setupSidebar).getByText("Model Type");
    expect(experimentTaskHeading).toBeInTheDocument();
    expect(within(setupSidebar).getByText("Model Type")).toBeInTheDocument();
    expect(within(setupSidebar).getByText("Model Name")).toBeInTheDocument();
    expect(within(setupSidebar).getByText("Variants")).toBeInTheDocument();
    expect(within(setupSidebar).getByText("Datasets")).toBeInTheDocument();
    expect(within(setupSidebar).getByText("Signals")).toBeInTheDocument();
    expect(within(setupSidebar).queryByText("Overrides")).not.toBeInTheDocument();
    expect(within(setupSidebar).getByText("Grid Search")).toBeInTheDocument();
    expect(
      within(setupSidebar).queryByRole("button", { name: /open full config/i }),
    ).not.toBeInTheDocument();
    expect(
      within(setupSidebar).queryByRole("button", { name: /^config$/i }),
    ).not.toBeInTheDocument();
    expect(
      within(setupSidebar).queryByRole("button", { name: /^reset$/i }),
    ).not.toBeInTheDocument();
    expect(
      experimentTaskHeading.compareDocumentPosition(modelTypeHeading) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

    const modelSelector = within(setupSidebar).getByRole("combobox", {
      name: /^training model$/i,
    });
    const presetSelector = within(setupSidebar).getByRole("combobox", {
      name: /^presets\s+1\s*\/\s*2 selected$/i,
    });
    const datasetSelector = within(setupSidebar).getByRole("combobox", {
      name: /^training datasets\s+1\s*\/\s*2 selected$/i,
    });
    const monitorSelector = within(setupSidebar).getByRole("combobox", {
      name: /^training monitors\s+0\s*\/\s*2 selected$/i,
    });
    const searchModeControl = within(setupSidebar).getByRole("radiogroup", {
      name: /training search mode/i,
    });
    const logFolderSelector = within(setupSidebar).getByRole("combobox", {
      name: "Log experiment folder",
    });
    expect(logFolderSelector.tagName).toBe("BUTTON");
    expect(logFolderSelector).toHaveClass("h-10");
    expect(logFolderSelector.parentElement).toHaveClass("h-10", "min-h-10");
    const trainingConfigSelector = within(setupSidebar).getByRole("tablist", {
      name: /training config selector/i,
    });
    const presetsTab = within(trainingConfigSelector).getByRole("tab", {
      name: "Presets",
    });
    const snapshotsTab = within(trainingConfigSelector).getByRole("tab", {
      name: "Snapshots",
    });
    expect(presetsTab).toHaveAttribute("aria-selected", "true");
    expect(snapshotsTab).toHaveAttribute("aria-selected", "false");
    const trainingConfigPanel = within(setupSidebar).getByRole("tabpanel", {
      name: "Presets",
    });
    expect(presetsTab).toHaveAttribute(
      "aria-controls",
      trainingConfigPanel.id,
    );
    expect(setupSidebar).toContainElement(searchModeControl);
    expect(within(details).queryByText(/^Request$/)).not.toBeInTheDocument();
    expect(within(runList).getByText("#1")).toBeInTheDocument();
    expect(within(runList).getAllByText("baseline").length).toBeGreaterThan(0);
    expect(within(runList).getByText("Mnist")).toBeInTheDocument();
    expect(
      within(runList).getByRole("button", { name: /command for run 1/i }),
    ).toBeInTheDocument();
    expect(within(statusSidebar).getByText("Run Plan")).toBeInTheDocument();
    expect(await within(statusSidebar).findByText("1 planned run"))
      .toBeInTheDocument();
    expect(within(statusSidebar).getByText("30 epochs")).toBeInTheDocument();
    expect(within(statusSidebar).getByText("30 left")).toBeInTheDocument();
    expect(within(statusSidebar).getByText("Next run")).toBeInTheDocument();
    expect(
      within(statusSidebar).queryByTitle(
        "source experiment.sh --model-type linears --model linear --preset baseline --experiment-task image-classification --datasets Mnist",
      ),
    ).not.toBeInTheDocument();
    expect(within(statusSidebar).queryByText("Preview runs")).not.toBeInTheDocument();
    expect(within(statusSidebar).getByText("Log Tail")).toBeInTheDocument();
    expect(within(statusSidebar).getByText("0 lines")).toBeInTheDocument();
    expect(within(statusSidebar).getByText("No log output yet")).toBeInTheDocument();

    const logFolderDropdown = within(setupSidebar).getByRole("combobox", {
      name: "Log experiment folder",
    });
    expect(logFolderDropdown).toBe(logFolderSelector);
    expect(setupSidebar.querySelector("select[aria-label='Log experiment folder']"))
      .not.toBeInTheDocument();
    expect(logFolderDropdown).toHaveTextContent("Select folder");
    await user.click(logFolderDropdown);
    const logFolderList = await within(setupSidebar).findByRole("listbox", {
      name: "Log experiment folder options",
    });
    expect(
      within(logFolderList).getByRole("option", {
        name: "test_model (1 runs)",
      }),
    ).toHaveAttribute("aria-selected", "false");
    expect(
      within(logFolderList).getByRole("option", {
        name: "test_model_2 (1 runs)",
      }),
    ).toHaveAttribute("aria-selected", "false");
    await user.keyboard("{Escape}");
    await waitFor(() => {
      expect(
        within(setupSidebar).queryByRole("listbox", {
          name: "Log experiment folder options",
        }),
      ).not.toBeInTheDocument();
    });

    expect(within(details).queryByText("Experiment Setup")).not.toBeInTheDocument();
    expect(
      within(details).queryByRole("combobox", { name: /^training preset$/i }),
    ).not.toBeInTheDocument();
    expect(modelSelector).toHaveTextContent("linear");
    expect(presetSelector).toHaveTextContent("baseline");
    expect(
      within(details).getAllByRole("combobox", { name: /^presets\b/i }),
    ).toHaveLength(1);
    expect(datasetSelector).toHaveTextContent("Mnist");
    expect(within(setupSidebar).getByRole("button", { name: /^Primary only$/i }))
      .toBeInTheDocument();
    expect(setupSidebar).toContainElement(monitorSelector);
    expect(within(setupSidebar).getByText("0 / 2")).toBeInTheDocument();
    const datasetField = datasetSelector.parentElement?.parentElement;
    const signalsField = monitorSelector.parentElement?.parentElement;
    if (
      !(datasetField instanceof HTMLElement) ||
      !(signalsField instanceof HTMLElement)
    ) {
      throw new Error("Expected setup selectors to render inside setup fields");
    }
    expect(datasetField.parentElement).toBe(signalsField.parentElement);
    const allMonitorsButton = within(signalsField).getByRole("button", {
      name: /^All$/i,
    });
    const noneMonitorsButton = within(signalsField).getByRole("button", {
      name: /^None$/i,
    });
    expect(allMonitorsButton.parentElement).toBe(noneMonitorsButton.parentElement);
    expect(allMonitorsButton.parentElement).toHaveClass(
      "grid",
      "grid-cols-2",
      "gap-2",
    );
    expect(allMonitorsButton).toBeEnabled();
    expect(noneMonitorsButton).toBeDisabled();
    await user.click(allMonitorsButton);
    await waitFor(() => {
      expect(
        within(setupSidebar).getByRole("combobox", {
          name: /^training monitors\s+2\s*\/\s*2 selected$/i,
        }),
      ).toBeInTheDocument();
    });
    expect(within(signalsField).getByRole("button", { name: /^None$/i }))
      .toBeEnabled();
    await user.click(within(signalsField).getByRole("button", { name: /^None$/i }));
    await waitFor(() => {
      const clearedMonitorSelector = within(setupSidebar).getByRole("combobox", {
        name: /^training monitors\s+0\s*\/\s*2 selected$/i,
      });
      expect(clearedMonitorSelector).toBeInTheDocument();
      expect(clearedMonitorSelector).toHaveTextContent("0 / 2 selected");
    });
    expect(within(setupSidebar).queryByLabelText(/monitor Linear layers/i))
      .not.toBeInTheDocument();
    expect(within(details).queryByText(/^Metrics$/)).not.toBeInTheDocument();

    const { listbox: datasetList } = await openTrainingMultiSelect(
      user,
      setupSidebar,
      "Training datasets",
    );
    expect(within(datasetList).getByRole("option", { name: /Mnist/i }))
      .toHaveAttribute("aria-selected", "true");
    expect(within(datasetList).getByRole("option", { name: /Cifar 10/i }))
      .toHaveAttribute("aria-selected", "false");
    await user.keyboard("{Escape}");

    const firstDatasetButton = within(setupSidebar).getByRole("button", {
      name: /^First$/i,
    });
    const allDatasetsButton = within(setupSidebar)
      .getAllByRole("button", { name: /^All$/i })
      .find((button) => button.parentElement === firstDatasetButton.parentElement);
    if (!(allDatasetsButton instanceof HTMLElement)) {
      throw new Error("Expected dataset All button to render near First");
    }
    expect(allDatasetsButton.parentElement).toBe(firstDatasetButton.parentElement);
    expect(allDatasetsButton.parentElement).toHaveClass("grid", "grid-cols-2", "gap-2");
    await user.click(allDatasetsButton);
    await waitFor(() => {
      expect(
        within(setupSidebar).getByRole("combobox", {
          name: /^training datasets\s+2\s*\/\s*2 selected$/i,
        }),
      ).toBeInTheDocument();
    });
    await user.click(firstDatasetButton);
    await waitFor(() => {
      expect(
        within(setupSidebar).getByRole("combobox", {
          name: /^training datasets\s+1\s*\/\s*2 selected$/i,
        }),
      ).toBeInTheDocument();
    });
    expect(within(setupSidebar).queryByText(/^Overrides$/)).not.toBeInTheDocument();
    expect(within(setupSidebar).queryByText("Config fields")).not.toBeInTheDocument();
    expect(
      within(setupSidebar).queryByRole("combobox", { name: /search config fields/i }),
    ).not.toBeInTheDocument();
    expect(
      within(setupSidebar).queryByRole("navigation", { name: /training override sections/i }),
    ).not.toBeInTheDocument();
    expect(
      within(setupSidebar).queryByRole("button", {
      name: /layer hidden stack options section/i,
      }),
    ).not.toBeInTheDocument();
    expect(within(setupSidebar).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(setupSidebar).getByRole("radio", { name: /new folder/i }))
      .toBeInTheDocument();
    await user.click(within(setupSidebar).getByRole("radio", { name: /new folder/i }));
    const newLogFolderModeControl = within(setupSidebar).getByRole("radiogroup", {
      name: /log folder mode/i,
    });
    expect(setupSidebar).toContainElement(newLogFolderModeControl);
    const newLogFolderInput = within(setupSidebar).getByRole("textbox", {
      name: "New log folder",
    });
    expect(setupSidebar).toContainElement(newLogFolderInput);
    expect(newLogFolderInput).toHaveClass("h-10");
    expect(within(setupSidebar).getByText("Enter a folder name.")).toHaveClass(
      "min-h-4",
      "leading-4",
    );
    const { listbox: monitorList } = await openTrainingMultiSelect(
      user,
      setupSidebar,
      "Training monitors",
    );
    await user.click(
      within(monitorList).getByRole("option", { name: /Linear layers/i }),
    );
    expect(within(monitorList).getByRole("option", { name: /Sampler usage/i }))
      .toBeInTheDocument();
    await user.keyboard("{Escape}");
    await waitFor(() => {
      const selectedMonitorSelector = within(setupSidebar).getByRole("combobox", {
        name: /^training monitors\s+1\s*\/\s*2 selected$/i,
      });
      expect(selectedMonitorSelector).toHaveTextContent("1 / 2 selected");
      expect(selectedMonitorSelector).toHaveTextContent("Linear layers");
    });
  });

  it("training setup target changes do not change preview overrides or main target", async () => {
    const { trainingBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    let details = await expandedTrainingDetailsReady(user);
    await user.click(screen.getByRole("button", { name: /^model$/i }));
    await setTargetHiddenDimOverride(user, "128");
    expect(screen.getAllByText(/1 overrides?/i).length).toBeGreaterThan(0);
    details = await expandedTrainingDetailsReady(user);

    const { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    await user.click(
      within(listbox).getByRole("option", {
        name: /recurrent-gating-halting/i,
      }),
    );
    expect(
      within(listbox).queryByRole("button", {
        name: /make recurrent-gating-halting primary/i,
      }),
    ).not.toBeInTheDocument();
    await user.click(
      within(details).getByRole("button", {
        name: /make recurrent-gating-halting primary/i,
      }),
    );
    await user.keyboard("{Escape}");

    await user.click(screen.getByRole("button", { name: /^model$/i }));
    expect(await waitForTargetValue("preset", "baseline"))
      .toHaveTextContent("baseline");
    details = await expandedTrainingDetailsReady(user);
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^presets\s+2\s*\/\s*2 selected$/i,
        }),
      ).toHaveTextContent("recurrent-gating-halting");
    });
    await user.click(screen.getByRole("button", { name: /^model$/i }));
    expect(screen.getAllByText(/1 overrides?/i).length).toBeGreaterThan(0);
    details = await expandedTrainingDetailsReady(user);
    await user.click(screen.getByRole("button", { name: /^model$/i }));
    let dialog = await openFullConfig(user);
    await waitFor(() => {
      expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue("128");
    });
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    details = await expandedTrainingDetailsReady(user);

    await user.click(screen.getByRole("button", { name: /^model$/i }));
    await setTargetHiddenDimOverride(user, "192");
    expect(screen.getAllByText(/1 overrides?/i).length).toBeGreaterThan(0);
    details = await expandedTrainingDetailsReady(user);

    await selectTrainingTargetOption(user, "model type", "Bert");

    await user.click(screen.getByRole("button", { name: /^model$/i }));
    expect(await waitForTargetValue("model", "linear")).toHaveTextContent("linear");
    expect(screen.getAllByText(/1 overrides?/i).length).toBeGreaterThan(0);
    details = await expandedTrainingDetailsReady(user);
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^training model$/i,
        }),
      ).toHaveTextContent("linear");
      expect(
        within(details).getByRole("combobox", {
          name: /^presets\s+1\s*\/\s*1 selected$/i,
        }),
      ).toHaveTextContent("bert-baseline");
    });
    await user.click(screen.getByRole("button", { name: /^model$/i }));
    dialog = await openFullConfig(user);
    await waitFor(() => {
      expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue("192");
    });
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    details = await expandedTrainingDetailsReady(user);
    expect(
      within(details).getByRole("combobox", {
        name: /^training datasets\s+1\s*\/\s*1 selected$/i,
      }),
    ).toHaveTextContent("Toy Text");
    expect(
      within(details).queryByText(/preview overrides are ignored/i),
    ).not.toBeInTheDocument();

    await selectNewTrainingLogFolder(user, "independent_training_target");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        modelType: "bert",
        model: "linear",
        preset: "bert-baseline",
        presets: ["bert-baseline"],
        datasets: ["ToyText"],
        overrides: {},
        logFolder: "independent_training_target",
      });
    });
    await user.click(screen.getByRole("button", { name: /^model$/i }));
    expect(await waitForTargetValue("model", "linear")).toHaveTextContent("linear");
    expect(await waitForTargetValue("preset", "baseline")).toHaveTextContent("baseline");
  });

  it("Training workspace posts selected model, preset, datasets, and overrides", async () => {
    const { trainingBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    let details = await expandedTrainingDetailsReady(user);
    details = await setTrainingHiddenDimOverride(user, details, "128");
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
    );
    await selectNewTrainingLogFolder(user, "my_experiment");
    await user.click(
      within(trainingRunList(details)).getByRole("button", {
        name: /start training/i,
      }),
    );

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        presets: ["baseline"],
        datasets: ["Mnist", "Cifar10"],
        overrides: { hidden_dim: "128" },
        logFolder: "my_experiment",
        monitors: [],
      });
      expect(trainingBodies[0]).toHaveProperty("runPlan.runs");
      expect(
        (trainingBodies[0] as { runPlan: { runs: unknown[] } }).runPlan.runs,
      ).toHaveLength(2);
      expect(trainingBodies[0]).not.toHaveProperty("runPlan.summary");
    });
  });

  it("Training Full Config does not mutate Model workspace overrides", async () => {
    installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    let details = await expandedTrainingDetailsReady(user);
    details = await setTrainingHiddenDimOverride(user, details, "128");

    await waitFor(() => {
      expect(
        within(trainingRunList(details)).getByText("hidden_dim=128"),
      ).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /^model$/i }));
    expect(screen.queryByText(/1 overrides?/i)).not.toBeInTheDocument();

    const dialog = await openFullConfig(user);
    expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue("256");
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));

    details = await expandedTrainingDetailsReady(user);
    const trainingDialog = await openTrainingFullConfig(user, details);
    await user.click(
      within(trainingDialog).getByRole("button", { name: /reset overrides/i }),
    );
    await user.click(
      within(trainingDialog).getByRole("button", { name: /^done$/i }),
    );
    await waitFor(() => {
      expect(within(trainingRunList(details)).queryByText("hidden_dim=128"))
        .not.toBeInTheDocument();
    });
  });

  it("shows backend field descriptions in Training Full Configuration", async () => {
    installFetchMock({
      schemaResponse: schemaResponseWithDescriptions({
        hidden_dim:
          "Sets the hidden feature width used by the training layer stack.",
      }),
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    const dialog = await openTrainingFullConfig(user, details);
    const helpButton = within(dialog).getByRole("button", {
      name: /show description for hidden dim/i,
    });
    const tooltipId = helpButton.getAttribute("aria-describedby");
    const tooltip = tooltipId ? document.getElementById(tooltipId) : null;

    if (!(tooltip instanceof HTMLElement)) {
      throw new Error("Expected field description tooltip to render");
    }

    await user.hover(helpButton);
    expect(tooltip).toHaveTextContent("training layer stack");
    expect(tooltip).not.toHaveClass("sr-only");
    expect(tooltip).toHaveStyle({
      width: "min(22rem, calc(100vw - 2rem))",
      maxWidth: "calc(100vw - 2rem)",
      overflowWrap: "normal",
      wordBreak: "normal",
    });
    expect(tooltip).not.toHaveClass("w-max");

    await user.unhover(helpButton);
    expect(tooltip).toHaveClass("sr-only");
    fireEvent.focus(helpButton);
    expect(tooltip).not.toHaveClass("sr-only");
  });

  it("shows multiple planned training runs and posts selected presets", async () => {
    const { trainingBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    const { listbox: presetListbox } = await openTrainingMultiSelect(
      user,
      details,
      "Presets",
    );
    const baselinePreset = within(presetListbox).getByRole("option", {
      name: /baseline/i,
    });
    expect(baselinePreset).toHaveAttribute("aria-selected", "true");
    expect(baselinePreset).toHaveAttribute("aria-disabled", "true");
    await user.click(
      within(presetListbox).getByRole("option", {
        name: /recurrent-gating-halting/i,
      }),
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Mnist/i,
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
    );

    expect(
      await within(details).findByRole("combobox", {
        name: /^presets\s+2\s*\/\s*2 selected$/i,
      }),
    ).toBeInTheDocument();
    expect(within(details).getByText("4 planned runs")).toBeInTheDocument();
    await user.keyboard("{Escape}");

    const runList = trainingRunList(details);
    await waitFor(() => {
      expect(
        within(runList).getAllByRole("button", { name: /command for run/i }),
      ).toHaveLength(4);
    });
    expect(
      within(runList).getAllByText("baseline").length,
    ).toBeGreaterThanOrEqual(2);
    expect(
      within(runList).getAllByText("recurrent-gating-halting").length,
    ).toBeGreaterThanOrEqual(2);
    expect(
      within(runList).getAllByText("Mnist").length,
    ).toBeGreaterThanOrEqual(2);
    expect(
      within(runList).getAllByText("Cifar10").length,
    ).toBeGreaterThanOrEqual(2);

    await selectNewTrainingLogFolder(user, "multi_preset");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        presets: ["baseline", "recurrent-gating-halting"],
        datasets: expect.arrayContaining(["Mnist", "Cifar10"]),
        logFolder: "multi_preset",
      });
    });
  });

  it("selects all presets when more than fifty are available", async () => {
    const { fetchMock } = installFetchMock({
      presetsResponse: largePresetsResponse,
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    const allPresetsButton = within(details)
      .getAllByRole("button", { name: /^All$/i })
      .find((button) =>
        button.parentElement?.textContent?.includes("Primary only"),
      );
    if (!(allPresetsButton instanceof HTMLElement)) {
      throw new Error("Expected preset All button to render near Primary only");
    }

    await user.click(allPresetsButton);

    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^presets\s+70\s*\/\s*70 selected$/i,
        }),
      ).toBeInTheDocument();
    });
    expect(
      within(details).getAllByText("70 planned runs").length,
    ).toBeGreaterThan(0);

    await waitFor(() => {
      expect(
        trainingRunPlanRequestBodies(fetchMock).at(-1)?.presets,
      ).toHaveLength(70);
    });
    expect(trainingRunPlanRequestBodies(fetchMock).at(-1)).toMatchObject({
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      presets: largePresetOptions.map((option) => option.name),
      datasets: ["Mnist"],
    });
  });

  it("shows planned training runs and row commands before training starts", async () => {
    installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await findTrainingRunSummary(
      /0\s*\/\s*1 runs;\s*0\s*\/\s*30 epochs;\s*Next run #1 0\s*\/\s*30 epochs/i,
    );
    expect(screen.queryByRole("button", { name: /expand runs/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /expanded run view/i }))
      .not.toBeInTheDocument();

    const runList = trainingRunList(details);
    expect(
      within(runList).getByText("Pending"),
    ).toBeInTheDocument();
    expect(within(runList).getAllByText("baseline").length).toBeGreaterThan(0);
    expect(within(runList).getByText("Mnist")).toBeInTheDocument();
    expect(within(runList).getByText("0 / 30 epochs")).toBeInTheDocument();

    await user.click(
      within(runList).getByRole("button", { name: /command for run 1/i }),
    );
    const commandDialog = await screen.findByRole("dialog", {
      name: /training command/i,
    });
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --datasets Mnist",
    );
  });

  it("keeps the running preset and dataset row visible with active progress", async () => {
    let runningJob: unknown;
    installFetchMock({
      createTrainingJobResponseFactory: (request) => {
        const job = mockTrainingJobPayload(
          request as Parameters<typeof mockTrainingJobPayload>[0],
          { status: "running" },
        );
        const runs = job.runPlan.runs.map((run, index) =>
          index === 0
            ? {
                ...run,
                status: "Running",
                currentEpoch: 1,
                metrics: { loss: 0.42 },
              }
            : run,
        );
        runningJob = {
          ...job,
          epoch: 1,
          step: 4,
          metrics: { loss: 0.42 },
          runPlan: {
            ...job.runPlan,
            runs,
            summary: {
              ...job.runPlan.summary,
              runningRuns: 1,
              pendingRuns: 0,
              completedEpochs: 1,
              remainingEpochs: 29,
            },
          },
        };
        return runningJob;
      },
      trainingJobResponseFactory: () =>
        runningJob ??
        mockTrainingJobPayload(
          { logFolder: "running_progress" },
          { status: "running" },
        ),
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await selectNewTrainingLogFolder(user, "running_progress");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    const runList = trainingRunList(details);
    await waitFor(() => {
      expect(within(runList).getByText("Running")).toBeInTheDocument();
      expect(within(runList).getByText("baseline")).toBeInTheDocument();
      expect(within(runList).getByText("Mnist")).toBeInTheDocument();
      expect(within(runList).getByText("1 / 30 epochs")).toBeInTheDocument();
    });

    const statusSidebar = within(details).getByRole("complementary", {
      name: "Training Status Sidebar",
    });
    expect(within(statusSidebar).getByText("Active Run")).toBeInTheDocument();
    expect(within(statusSidebar).getByText("running")).toBeInTheDocument();
    expect(within(statusSidebar).getByText("Active run")).toBeInTheDocument();
    expect(within(statusSidebar).getByText("epoch 1 / step 4")).toBeInTheDocument();
    expect(within(statusSidebar).getByText("29 left")).toBeInTheDocument();
  });

  it("shows and submits plain preset rows alongside checked config snapshots", async () => {
    const snapshotRecord = {
      id: "snap-wide",
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      name: "Wide snapshot",
      overrides: { hidden_dim: "128", num_epochs: "5" },
      createdAt: "2026-06-01T00:00:00.000Z",
      updatedAt: "2026-06-01T00:00:00.000Z",
    };
    const { trainingBodies } = installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: [
          ...schemaResponse.fields,
          {
            key: "num_epochs",
            configKey: "NUM_EPOCHS",
            flag: "--num-epochs",
            label: "num epochs",
            section: "Training",
            sectionPath: ["Training"],
            type: "int",
            default: 30,
            nullable: false,
            choices: [],
          },
        ],
      },
      configSnapshotsResponse: {
        modelType: "linears",
        model: "linear",
        snapshots: [snapshotRecord],
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await setTrainingHiddenDimOverride(user, details, "192");
    await selectNewTrainingLogFolder(user, "mixed_snapshots");

    expect(
      await findTrainingRunSummary(
        /0\s*\/\s*1 runs;\s*0\s*\/\s*30 epochs/i,
      ),
    ).toBeInTheDocument();

    await user.click(
      within(details).getByRole("tab", { name: /^snapshots$/i }),
    );
    const { listbox: snapshotListbox } = await openTrainingMultiSelect(
      user,
      details,
      "Config snapshots",
    );
    const snapshotOption = within(snapshotListbox).getByRole("option", {
      name: /wide snapshot/i,
    });
    expect(snapshotOption).toHaveAttribute("aria-selected", "false");

    await user.click(snapshotOption);
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^config snapshots\s+1\s*\/\s*1 selected$/i,
        }),
      ).toHaveTextContent("Wide snapshot");
    });

    await user.click(snapshotOption);
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^config snapshots\s+0\s*\/\s*1 selected$/i,
        }),
      ).toBeInTheDocument();
    });
    expect(
      await findTrainingRunSummary(
        /0\s*\/\s*1 runs;\s*0\s*\/\s*30 epochs/i,
      ),
    ).toBeInTheDocument();

    await user.click(snapshotOption);
    await user.keyboard("{Escape}");

    await findTrainingRunSummary(
      /0\s*\/\s*2 runs;\s*0\s*\/\s*35 epochs/i,
    );
    const runList = trainingRunList(details);
    expect(within(runList).getByText("Wide snapshot")).toBeInTheDocument();
    expect(within(runList).getAllByText("hidden_dim=192")).toHaveLength(2);

    await user.click(
      within(runList).getByRole("button", { name: /command for run 1/i }),
    );
    let commandDialog = await screen.findByRole("dialog", {
      name: /training command/i,
    });
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --experiment-task image-classification --datasets Mnist --logdir mixed_snapshots --config --hidden-dim 192",
    );
    await user.click(
      within(commandDialog).getByRole("button", {
        name: /close training command/i,
      }),
    );

    await user.click(
      within(runList).getByRole("button", { name: /command for run 2/i }),
    );
    commandDialog = await screen.findByRole("dialog", {
      name: /training command/i,
    });
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --experiment-task image-classification --datasets Mnist --logdir mixed_snapshots --config --hidden-dim 192 --num-epochs 5",
    );
    await user.click(
      within(commandDialog).getByRole("button", {
        name: /close training command/i,
      }),
    );

    await user.click(
      within(runList).getByRole("button", { name: /^commands$/i }),
    );
    commandDialog = await screen.findByRole("dialog", {
      name: /training commands/i,
    });
    expect(
      within(commandDialog).getByRole("textbox", {
        name: /^training commands$/i,
      }),
    ).toHaveValue(
      [
        "(",
        "  set -e",
        "  source experiment.sh --model-type linears --model linear --preset baseline --experiment-task image-classification --datasets Mnist --logdir mixed_snapshots --config --hidden-dim 192",
        "  source experiment.sh --model-type linears --model linear --preset baseline --experiment-task image-classification --datasets Mnist --logdir mixed_snapshots --config --hidden-dim 192 --num-epochs 5",
        ")",
      ].join("\n"),
    );
    await user.click(
      within(commandDialog).getByRole("button", {
        name: /close training commands/i,
      }),
    );

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies).toHaveLength(1);
    });
    expect(trainingBodies[0]).toMatchObject({
      overrides: {},
      logFolder: "mixed_snapshots",
    });
    expect(trainingBodies[0]).not.toHaveProperty("search");
    expect(
      (trainingBodies[0] as { runPlan: { runs: unknown[] } }).runPlan.runs,
    ).toHaveLength(2);
    expect(trainingBodies[0]).toHaveProperty(
      "runPlan.runs.0.overrides.hidden_dim",
      "192",
    );
    expect(trainingBodies[0]).not.toHaveProperty("runPlan.runs.0.command");
    expect(trainingBodies[0]).toHaveProperty(
      "runPlan.runs.1.snapshotId",
      "snap-wide",
    );
    expect(trainingBodies[0]).toHaveProperty(
      "runPlan.runs.1.snapshotName",
      "Wide snapshot",
    );
    expect(trainingBodies[0]).toHaveProperty(
      "runPlan.runs.1.overrides.hidden_dim",
      "192",
    );
    expect(snapshotRecord.overrides).toEqual({
      hidden_dim: "128",
      num_epochs: "5",
    });
  });

  it("opens snapshot draft config from a preset row action without changing preset inclusion", async () => {
    installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    const { control, listbox } = await openTrainingMultiSelect(
      user,
      details,
      "Presets",
    );
    const baselineOption = within(listbox).getByRole("option", {
      name: /baseline/i,
    });
    const unselectedPresetOption = within(listbox).getByRole("option", {
      name: /recurrent-gating-halting/i,
    });
    expect(baselineOption).toHaveAttribute("aria-selected", "true");
    expect(baselineOption).toHaveAttribute("aria-disabled", "true");
    expect(unselectedPresetOption).toHaveAttribute("aria-selected", "false");

    await user.click(
      within(listbox).getByRole("button", {
        name: "Create snapshot from recurrent-gating-halting",
      }),
    );

    await waitFor(() => {
      expect(
        within(details).queryByRole("listbox", {
          name: /presets options/i,
        }),
      ).not.toBeInTheDocument();
    });
    const fullConfigDialog = await screen.findByRole("dialog", {
      name: /full configuration/i,
    });
    expect(
      within(fullConfigDialog).getByRole("button", { name: "Save as Snapshot" }),
    ).toBeInTheDocument();
    expect(control).toHaveTextContent("1 / 2 selected");
    expect(control).toHaveTextContent("baseline");
    expect(control).not.toHaveTextContent("recurrent-gating-halting");

    await user.click(
      within(fullConfigDialog).getByRole("button", {
        name: /close full config/i,
      }),
    );
    const allPresetsButton = within(details)
      .getAllByRole("button", { name: /^All$/i })
      .find((button) =>
        button.parentElement?.textContent?.includes("Primary only"),
      );
    if (!(allPresetsButton instanceof HTMLElement)) {
      throw new Error("Expected preset All button to render near Primary only");
    }
    const primaryOnlyButton = within(details).getByRole("button", {
      name: /^Primary only$/i,
    });
    await user.click(allPresetsButton);
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^presets\s+2\s*\/\s*2 selected$/i,
        }),
      ).toBeInTheDocument();
    });
    await user.click(primaryOnlyButton);
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^presets\s+1\s*\/\s*2 selected$/i,
        }),
      ).toBeInTheDocument();
    });
  });

  it("opens edit and duplicate config from snapshot row actions without row toggles", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: [
          ...schemaResponse.fields,
          {
            key: "num_epochs",
            configKey: "NUM_EPOCHS",
            flag: "--num-epochs",
            label: "num epochs",
            section: "Training",
            sectionPath: ["Training"],
            type: "int",
            default: 30,
            nullable: false,
            choices: [],
          },
        ],
      },
      configSnapshotsResponse: {
        modelType: "linears",
        model: "linear",
        snapshots: [
          {
            id: "snap-wide",
            modelType: "linears",
            model: "linear",
            preset: "baseline",
            name: "Wide snapshot",
            overrides: { hidden_dim: "128", num_epochs: "5" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
        ],
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await user.click(
      within(details).getByRole("tab", { name: /^snapshots$/i }),
    );
    let { control, listbox } = await openTrainingMultiSelect(
      user,
      details,
      "Config snapshots",
    );
    const snapshotOption = within(listbox).getByRole("option", {
      name: /wide snapshot/i,
    });
    expect(snapshotOption).toHaveAttribute("aria-selected", "false");
    expect(control).toHaveAccessibleName("Config snapshots 0 / 1 selected");
    expect(control).toHaveTextContent("Select snapshots");

    await user.click(
      within(listbox).getByRole("button", {
        name: "Edit snapshot Wide snapshot",
      }),
    );

    let fullConfigDialog = await screen.findByRole("dialog", {
      name: /full configuration/i,
    });
    expect(
      within(fullConfigDialog).getByRole("button", {
        name: "Save Snapshot Changes",
      }),
    ).toBeInTheDocument();
    expect(control).toHaveAccessibleName("Config snapshots 0 / 1 selected");
    expect(control).not.toHaveTextContent("Wide snapshot");
    await user.click(
      within(fullConfigDialog).getByRole("button", {
        name: /close full config/i,
      }),
    );

    ({ control, listbox } = await openTrainingMultiSelect(
      user,
      details,
      "Config snapshots",
    ));
    await user.click(
      within(listbox).getByRole("button", {
        name: "Duplicate snapshot Wide snapshot",
      }),
    );

    fullConfigDialog = await screen.findByRole("dialog", {
      name: /full configuration/i,
    });
    expect(
      within(fullConfigDialog).getByRole("button", { name: "Save as Snapshot" }),
    ).toBeInTheDocument();
    expect(control).toHaveAccessibleName("Config snapshots 0 / 1 selected");
    expect(control).not.toHaveTextContent("Wide snapshot");
  });

  it("submits a Config Snapshot-only plan after removing every base preset", async () => {
    const { trainingBodies } = installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: [
          ...schemaResponse.fields,
          {
            key: "num_epochs",
            configKey: "NUM_EPOCHS",
            flag: "--num-epochs",
            label: "num epochs",
            section: "Training",
            sectionPath: ["Training"],
            type: "int",
            default: 30,
            nullable: false,
            choices: [],
          },
        ],
      },
      configSnapshotsResponse: {
        modelType: "linears",
        model: "linear",
        snapshots: [
          {
            id: "snap-wide",
            modelType: "linears",
            model: "linear",
            preset: "baseline",
            name: "Wide snapshot",
            overrides: { hidden_dim: "128", num_epochs: "5" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
        ],
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await user.click(
      within(details).getByRole("tab", { name: /^snapshots$/i }),
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Config snapshots",
      /wide snapshot/i,
    );
    await selectNewTrainingLogFolder(user, "snapshot_only");
    await findTrainingRunSummary(
      /0\s*\/\s*2 runs;\s*0\s*\/\s*35 epochs/i,
    );
    const runList = trainingRunList(details);
    await user.click(
      within(runList).getByRole("button", {
        name: "Remove preset baseline from this run plan",
      }),
    );

    await waitFor(() => {
      expect(within(runList).getByText("Wide snapshot")).toBeInTheDocument();
      expect(
        within(runList).getAllByRole("button", {
          name: /command for run/i,
        }),
      ).toHaveLength(1);
    });

    expect(
      within(details).getByRole("combobox", {
        name: /^config snapshots\s+1\s*\/\s*1 selected$/i,
      }),
    ).toBeInTheDocument();
    await user.click(within(details).getByRole("tab", { name: "Presets" }));
    expect(
      within(details).getByRole("combobox", {
        name: /^presets\s+0\s*\/\s*2 selected$/i,
      }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies).toHaveLength(1);
    });
    expect(trainingBodies[0]).toMatchObject({
      logFolder: "snapshot_only",
      presets: ["baseline"],
      runPlan: {
        runs: [
          expect.objectContaining({
            snapshotId: "snap-wide",
            snapshotName: "Wide snapshot",
            preset: "baseline",
          }),
        ],
      },
    });
    expect(trainingBodies[0]).not.toHaveProperty("search");
  });

  it("deselects a snapshot run across datasets and syncs setup variants", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: [
          ...schemaResponse.fields,
          {
            key: "num_epochs",
            configKey: "NUM_EPOCHS",
            flag: "--num-epochs",
            label: "num epochs",
            section: "Training",
            sectionPath: ["Training"],
            type: "int",
            default: 30,
            nullable: false,
            choices: [],
          },
        ],
      },
      configSnapshotsResponse: {
        modelType: "linears",
        model: "linear",
        snapshots: [
          {
            id: "snap-wide",
            modelType: "linears",
            model: "linear",
            preset: "baseline",
            name: "Wide snapshot",
            overrides: { hidden_dim: "128", num_epochs: "5" },
            createdAt: "2026-06-01T00:00:00.000Z",
            updatedAt: "2026-06-01T00:00:00.000Z",
          },
        ],
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
    );
    await user.click(
      within(details).getByRole("tab", { name: /^snapshots$/i }),
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Config snapshots",
      /wide snapshot/i,
    );
    await findTrainingRunSummary(
      /0\s*\/\s*4 runs;\s*0\s*\/\s*70 epochs/i,
    );
    const runList = trainingRunList(details);
    expect(within(runList).getAllByText("Wide snapshot")).toHaveLength(2);

    await user.click(
      within(runList).getAllByRole("button", {
        name: "Remove snapshot Wide snapshot from this run plan",
      })[0],
    );

    await waitFor(() => {
      expect(
        within(runList).queryByText("Wide snapshot"),
      ).not.toBeInTheDocument();
      expect(
        within(runList).getAllByRole("button", {
          name: /command for run/i,
        }),
      ).toHaveLength(2);
    });

    expect(
      within(details).getByRole("combobox", {
        name: /^config snapshots\s+0\s*\/\s*1 selected$/i,
      }),
    ).toBeInTheDocument();

    await user.click(within(details).getByRole("tab", { name: "Presets" }));
    expect(
      within(details).getByRole("combobox", {
        name: /^presets\s+1\s*\/\s*2 selected$/i,
      }),
    ).toBeInTheDocument();
  });

  it("shows Resample in the Training run list for random search before start", async () => {
    const { fetchMock } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await user.click(within(details).getByRole("radio", { name: /^random$/i }));
    await user.click(within(details).getByLabelText(/^search axis hidden_dim$/i));
    await findTrainingRunSummary(
      /0\s*\/\s*2 runs;\s*0\s*\/\s*60 epochs/i,
    );

    const runPlanCallCount = trainingRunPlanCalls(fetchMock).length;
    await user.click(
      within(trainingRunList(details)).getByRole("button", { name: /^resample$/i }),
    );

    await waitFor(() => {
      expect(trainingRunPlanCalls(fetchMock).length).toBeGreaterThan(
        runPlanCallCount,
      );
    });
  });

  it("keeps completed job progress visible after the draft config changes", async () => {
    installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    let details = await expandedTrainingDetailsReady(user);
    details = await setTrainingHiddenDimOverride(user, details, "128");
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
    );
    await selectNewTrainingLogFolder(user, "completed_plan");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(
        screen.getByRole("status", {
          name: /2\s*\/\s*2 runs;\s*60\s*\/\s*60 epochs/i,
        }),
      ).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(within(details).getByText("Results")).toBeInTheDocument();
      expect(within(details).getByText("done")).toBeInTheDocument();
      expect(within(details).getByText("1 line")).toBeInTheDocument();
      expect(within(details).getByTitle("logs/completed_plan")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /^model$/i }));
    await user.click(screen.getByRole("button", { name: /^reset overrides$/i }));
    await waitFor(() => {
      expect(screen.getAllByText("0 overrides").length).toBeGreaterThan(0);
    });
    details = await expandedTrainingDetailsReady(user);
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
      false,
    );
    await user.keyboard("{Escape}");
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^training datasets\s+1\s*\/\s*2 selected$/i,
        }),
      ).toBeInTheDocument();
    });

    await findTrainingRunSummary(
      /2\s*\/\s*2 runs;\s*60\s*\/\s*60 epochs/i,
    );
    const runList = trainingRunList(details);
    expect(within(runList).getAllByText("Completed")).toHaveLength(2);
    expect(within(runList).getAllByText("hidden_dim=128")).toHaveLength(2);
    expect(within(runList).getByText("Cifar10")).toBeInTheDocument();
    expect(within(runList).getByText("Mnist")).toBeInTheDocument();
    expect(
      within(runList).queryByRole("button", { name: /^resample$/i }),
    ).not.toBeInTheDocument();
  });

  it("resets completed training progress back to the current draft plan", async () => {
    installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    let details = await expandedTrainingDetailsReady(user);
    details = await setTrainingHiddenDimOverride(user, details, "128");
    await selectNewTrainingLogFolder(user, "completed_then_reset");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await findTrainingRunSummary(
      /1\s*\/\s*1 runs;\s*30\s*\/\s*30 epochs/i,
    );
    const completedRunList = trainingRunList(details);
    expect(
      within(completedRunList).getByRole("button", { name: /^reset training$/i }),
    ).toBeInTheDocument();

    await setTrainingHiddenDimOverride(user, details, "192");
    await selectNewTrainingLogFolder(user, "after_reset");
    await user.click(
      within(trainingRunList(details)).getByRole("button", {
        name: /^reset training$/i,
      }),
    );

    await findTrainingRunSummary(
      /0\s*\/\s*1 runs;\s*0\s*\/\s*30 epochs;\s*Next run #1 0\s*\/\s*30 epochs/i,
    );
    expect(
      screen.queryByRole("button", { name: /^reset training$/i }),
    ).not.toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /start training/i })).toBeEnabled();
    });

    const runList = trainingRunList(details);
    expect(within(runList).queryByText("Completed")).not.toBeInTheDocument();
    expect(within(runList).getByText("hidden_dim=192")).toBeInTheDocument();
  });

  it("hides reset training while the next run is being created", async () => {
    let createCount = 0;
    let latestRequest: Parameters<typeof mockTrainingJobPayload>[0] | undefined;
    const pendingCreate = deferred<unknown>();
    installFetchMock({
      createTrainingJobResponseFactory: (request) => {
        latestRequest = request as Parameters<typeof mockTrainingJobPayload>[0];
        createCount += 1;
        if (createCount === 2) {
          return pendingCreate.promise;
        }
        return mockTrainingJobPayload(latestRequest, { status: "running" });
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await expandedTrainingDetailsReady(user);
    await selectNewTrainingLogFolder(user, "completed_then_pending_next");
    await user.click(screen.getByRole("button", { name: /start training/i }));
    await findTrainingRunSummary(
      /1\s*\/\s*1 runs;\s*30\s*\/\s*30 epochs/i,
    );
    expect(
      screen.getByRole("button", { name: /^reset training$/i }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /^reset training$/i }))
        .not.toBeInTheDocument();
    });
    pendingCreate.resolve(
      mockTrainingJobPayload(latestRequest ?? {}, { status: "running" }),
    );
  });

  it("starts the next training run from the changed draft plan after completion", async () => {
    const { trainingBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    let details = await expandedTrainingDetailsReady(user);
    details = await setTrainingHiddenDimOverride(user, details, "128");
    await selectNewTrainingLogFolder(user, "first_plan");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies).toHaveLength(1);
      expect(
        screen.getByRole("status", {
          name: /1\s*\/\s*1 runs;\s*30\s*\/\s*30 epochs/i,
        }),
      ).toBeInTheDocument();
    });

    await setTrainingHiddenDimOverride(user, details, "192");
    await selectNewTrainingLogFolder(user, "second_plan");
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /start training/i }))
        .toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies).toHaveLength(2);
      expect(trainingBodies[1]).toMatchObject({
        logFolder: "second_plan",
        overrides: { hidden_dim: "192" },
      });
      expect(trainingBodies[1]).toHaveProperty(
        "runPlan.runs.0.overrides.hidden_dim",
        "192",
      );
      expect(trainingBodies[1]).not.toHaveProperty("runPlan.runs.0.command");
    });
  });

  it("making a selected preset primary updates the training target and resets search state", async () => {
    const { inspectBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    let details = await expandedTrainingDetailsReady(user);
    details = await setTrainingHiddenDimOverride(user, details, "128");
    await user.click(within(details).getByRole("radio", { name: /^grid$/i }));
    await user.click(within(details).getByLabelText(/^search axis hidden_dim$/i));
    expect(within(details).getByText(/1 fixed override replaced by search axes/i))
      .toBeInTheDocument();

    const initialInspectRequestCount = inspectBodies.length;
    const { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    await user.click(
      within(listbox).getByRole("option", {
        name: /recurrent-gating-halting/i,
      }),
    );
    expect(
      await within(details).findByRole("combobox", {
        name: /^presets\s+2\s*\/\s*2 selected$/i,
      }),
    ).toBeInTheDocument();
    expect(
      within(listbox).queryByRole("button", {
        name: /make recurrent-gating-halting primary/i,
      }),
    ).not.toBeInTheDocument();

    await user.click(
      await within(details).findByRole("button", {
        name: /make recurrent-gating-halting primary/i,
      }),
    );
    await user.keyboard("{Escape}");

    await user.click(screen.getByRole("button", { name: /^model$/i }));
    expect(await waitForTargetValue("preset", "baseline"))
      .toHaveTextContent("baseline");
    expect(screen.queryByText(/1 overrides?/i)).not.toBeInTheDocument();
    details = await expandedTrainingDetailsReady(user);
    await waitFor(() => {
      expect(within(details).getByRole("radio", { name: /^off$/i }))
        .toHaveAttribute("aria-checked", "true");
    });
    expect(inspectBodies).toHaveLength(initialInspectRequestCount);

    await user.click(screen.getByRole("button", { name: /^model$/i }));
    const dialog = await openFullConfig(user);
    await user.click(within(dialog).getByRole("button", { name: /update preview/i }));
    await waitFor(() =>
      expect(inspectBodies).toHaveLength(initialInspectRequestCount + 1),
    );
    expect(inspectBodies.at(-1)).toEqual({
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      experimentTask: "image-classification",
      dataset: "Mnist",
      overrides: {},
    });
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    details = await expandedTrainingDetailsReady(user);
    expect(
      within(details).getByRole("combobox", {
        name: /^presets\s+2\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("recurrent-gating-halting");
  });

  it("removing the current primary preset keeps the target preset unchanged", async () => {
    installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    let details = await expandedTrainingDetailsReady(user);
    const { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    await user.click(
      within(listbox).getByRole("option", {
        name: /recurrent-gating-halting/i,
      }),
    );

    const currentPrimary = within(listbox).getByRole("option", { name: /baseline/i });
    expect(currentPrimary).toHaveAttribute("aria-selected", "true");
    expect(currentPrimary).not.toHaveAttribute("aria-disabled", "true");
    await user.click(currentPrimary);

    await user.click(screen.getByRole("button", { name: /^model$/i }));
    expect(await waitForTargetValue("preset", "baseline")).toHaveTextContent("baseline");
    details = await expandedTrainingDetailsReady(user);
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^presets\s+1\s*\/\s*2 selected$/i,
        }),
      ).toHaveTextContent("recurrent-gating-halting");
    });
    const { listbox: refreshedListbox } = await openTrainingMultiSelect(
      user,
      details,
      "Presets",
    );
    expect(within(refreshedListbox).getByRole("option", { name: /baseline/i }))
      .toHaveAttribute("aria-selected", "false");
  });

  it("keeps the last selected preset selected in the preset multiselect", async () => {
    installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    const { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    const onlySelectedOption = within(listbox).getByRole("option", { name: /baseline/i });

    expect(onlySelectedOption).toHaveAttribute("aria-selected", "true");
    expect(onlySelectedOption).toHaveAttribute("aria-disabled", "true");
    await user.click(onlySelectedOption);

    expect(onlySelectedOption).toHaveAttribute("aria-selected", "true");
    expect(
      within(details).getByRole("combobox", {
        name: /^presets\s+1\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("baseline");
  });

  it("keeps the last dataset selected in the dataset multiselect", async () => {
    installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    const { listbox } = await openTrainingMultiSelect(
      user,
      details,
      "Training datasets",
    );
    const selectedDataset = within(listbox).getByRole("option", { name: /Mnist/i });

    expect(selectedDataset).toHaveAttribute("aria-selected", "true");
    expect(selectedDataset).toHaveAttribute("aria-disabled", "true");
    await user.click(selectedDataset);

    expect(selectedDataset).toHaveAttribute("aria-selected", "true");
    expect(
      within(details).getByRole("combobox", {
        name: /^training datasets\s+1\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("Mnist");
  });

  it("filters training multiselect options and selects matching results", async () => {
    installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    let { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    await user.type(
      within(details).getByRole("searchbox", { name: /^search presets$/i }),
      "recurrent",
    );

    expect(within(listbox).queryByRole("option", { name: /baseline/i }))
      .not.toBeInTheDocument();
    await user.click(
      within(listbox).getByRole("option", { name: /recurrent-gating-halting/i }),
    );
    expect(
      within(details).getByRole("combobox", {
        name: /^presets\s+2\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("recurrent-gating-halting");

    await user.keyboard("{Escape}");
    ({ listbox } = await openTrainingMultiSelect(
      user,
      details,
      "Training datasets",
    ));
    await user.type(
      within(details).getByRole("searchbox", { name: /^search training datasets$/i }),
      "cifar",
    );

    expect(within(listbox).queryByRole("option", { name: /Mnist/i }))
      .not.toBeInTheDocument();
    await user.click(within(listbox).getByRole("option", { name: /Cifar 10/i }));
    expect(
      within(details).getByRole("combobox", {
        name: /^training datasets\s+2\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("Cifar 10");
  });

  it("multiplies grid planned runs by selected presets and datasets", async () => {
    installFetchMock({
      searchSpaceResponse: {
        ...searchSpaceResponse,
        axes: [
          {
            ...searchSpaceResponse.axes[0],
            values: [64, 128, 256],
          },
        ],
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await setTrainingMultiSelectOption(
      user,
      details,
      "Presets",
      /recurrent-gating-halting/i,
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Mnist/i,
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
    );

    await user.click(within(details).getByRole("radio", { name: /^grid$/i }));
    await user.click(within(details).getByLabelText(/^search axis hidden_dim$/i));

    expect(within(details).getByText("3 combinations")).toBeInTheDocument();
    expect(within(details).getAllByText("12 planned runs").length).toBeGreaterThan(0);
  });

  it("starts grid search with selected axis values and omits conflicting overrides", async () => {
    const { trainingBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    let details = await expandedTrainingDetailsReady(user);
    details = await setTrainingHiddenDimOverride(user, details, "128");
    await selectNewTrainingLogFolder(user, "grid_search");

    await user.click(within(details).getByRole("radio", { name: /^grid$/i }));
    expect(screen.getByRole("button", { name: /start training/i })).toBeDisabled();

    await user.click(within(details).getByLabelText(/^search axis hidden_dim$/i));
    await user.click(within(details).getByLabelText(/^search value hidden_dim 128$/i));

    expect(within(details).getByText(/1 fixed override replaced by search axes/i))
      .toBeInTheDocument();
    expect(screen.getByRole("button", { name: /start training/i })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        presets: ["baseline"],
        datasets: ["Mnist"],
        overrides: {},
        logFolder: "grid_search",
        monitors: [],
        search: {
          mode: "grid",
          values: { hidden_dim: [64] },
        },
      });
      expect(
        (trainingBodies[0] as { runPlan: { runs: unknown[] } }).runPlan.runs,
      ).toHaveLength(1);
      expect(trainingBodies[0]).toHaveProperty(
        "runPlan.runs.0.overrides.hidden_dim",
        64,
      );
    });
  });

  it("posts random search with the configured sample count", async () => {
    const { trainingBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await selectNewTrainingLogFolder(user, "random_search");
    await user.click(within(details).getByRole("radio", { name: /^random$/i }));
    expect(within(details).getByLabelText(/^random search samples$/i)).toHaveValue(10);

    await user.click(within(details).getByLabelText(/^search axis stack_activation$/i));
    const samplesInput = within(details).getByLabelText(/^random search samples$/i);
    await user.clear(samplesInput);
    await user.type(samplesInput, "7");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "random_search",
        search: {
          mode: "random",
          values: { stack_activation: ["RELU", "GELU"] },
          randomSamples: 7,
        },
      });
    });
  });

  it("selects all search axes and values from the grid setup", async () => {
    const { trainingBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await selectNewTrainingLogFolder(user, "all_axes_search");
    await user.click(within(details).getByRole("radio", { name: /^grid$/i }));
    await user.click(within(details).getByRole("button", { name: /^all axes$/i }));

    expect(within(details).getAllByText("3 axes").length).toBeGreaterThan(0);
    expect(within(details).getByText("8 combinations")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "all_axes_search",
        search: {
          mode: "grid",
          values: {
            hidden_dim: [64, 128],
            stack_num_layers: [2, 4],
            stack_activation: ["RELU", "GELU"],
          },
        },
      });
    });
  });

  it("skips preset-owned axes when selecting all grid search axes", async () => {
    const { trainingBodies } = installFetchMock({
      presetsResponse: searchLockPresetsResponse,
      searchSpaceResponseFactory: searchSpaceWithPresetLocks,
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await selectNewTrainingLogFolder(user, "post_norm_search");
    await setTrainingMultiSelectOption(user, details, "Presets", /post-norm/i);
    await user.click(within(details).getByRole("radio", { name: /^grid$/i }));

    expect(
      (
        await within(details).findAllByText(
          /1 preset-owned axis will be skipped for POST_NORM: stack layer norm position/i,
        )
      ).length,
    ).toBeGreaterThan(0);
    expect(
      within(details).getByLabelText(/^search axis stack_layer_norm_position$/i),
    ).toBeDisabled();
    expect(
      within(details).getByText(/Locked value: AFTER by POST_NORM/i),
    ).toBeInTheDocument();

    await user.click(within(details).getByRole("button", { name: /^all axes$/i }));

    expect(within(details).getAllByText("1 axes").length).toBeGreaterThan(0);
    expect(within(details).getByText("2 combinations")).toBeInTheDocument();
    await waitFor(() => {
      expect(
        within(details).getAllByText(
          /1 preset-owned axis will be skipped for POST_NORM/i,
        ).length,
      ).toBeGreaterThan(1);
    });

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "post_norm_search",
        presets: ["baseline", "post-norm"],
        search: {
          mode: "grid",
          values: {
            hidden_dim: [64, 128],
          },
        },
      });
      expect(
        Object.keys(
          (trainingBodies[0] as { search?: { values?: Record<string, unknown[]> } })
            .search?.values ?? {},
        ),
      ).not.toContain("stack_layer_norm_position");
    });
  });

  it("skips an already selected axis when a new preset owns it", async () => {
    const { trainingBodies } = installFetchMock({
      presetsResponse: searchLockPresetsResponse,
      searchSpaceResponseFactory: searchSpaceWithPresetLocks,
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await selectNewTrainingLogFolder(user, "gating_search");
    await user.click(within(details).getByRole("radio", { name: /^grid$/i }));
    await user.click(within(details).getByLabelText(/^search axis hidden_dim$/i));
    await user.click(
      within(details).getByLabelText(/^search axis stack_layer_norm_position$/i),
    );

    expect(screen.getByRole("button", { name: /start training/i })).toBeEnabled();

    await setTrainingMultiSelectOption(user, details, "Presets", /^gating\b/i);

    expect(
      (
        await within(details).findAllByText(
          /1 selected axis was skipped because a selected preset owns it/i,
        )
      ).length,
    ).toBeGreaterThan(0);
    expect(within(details).getAllByText("1 axes").length).toBeGreaterThan(0);
    expect(screen.getByRole("button", { name: /start training/i })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "gating_search",
        presets: ["baseline", "gating"],
        search: {
          mode: "grid",
          values: {
            stack_layer_norm_position: ["BEFORE", "AFTER"],
          },
        },
      });
      expect(
        Object.keys(
          (trainingBodies[0] as { search?: { values?: Record<string, unknown[]> } })
            .search?.values ?? {},
        ),
      ).not.toContain("hidden_dim");
    });
  });

  it("confirms large grid searches before posting", async () => {
    const largeSearchSpace = {
      ...searchSpaceResponse,
      axes: [
        {
          ...searchSpaceResponse.axes[0],
          values: Array.from({ length: 11 }, (_, index) => index + 1),
        },
        {
          ...searchSpaceResponse.axes[1],
          values: Array.from({ length: 10 }, (_, index) => index + 1),
        },
      ],
    };
    const { trainingBodies } = installFetchMock({
      searchSpaceResponse: largeSearchSpace,
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await selectNewTrainingLogFolder(user, "large_grid_search");
    await user.click(within(details).getByRole("radio", { name: /^grid$/i }));
    await user.click(within(details).getByRole("button", { name: /^all axes$/i }));

    expect(within(details).getAllByText("110 planned runs").length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: /start training/i }));

    const dialog = await screen.findByRole("dialog", { name: /confirm grid search/i });
    expect(trainingBodies).toHaveLength(0);
    expect(within(dialog).getByText(/110 training runs/i)).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /^cancel$/i }));
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /confirm grid search/i }))
        .not.toBeInTheDocument();
    });
    expect(trainingBodies).toHaveLength(0);

    await user.click(screen.getByRole("button", { name: /start training/i }));
    await user.click(
      within(await screen.findByRole("dialog", { name: /confirm grid search/i }))
        .getByRole("button", { name: /start training/i }),
    );

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "large_grid_search",
        search: {
          mode: "grid",
        },
      });
    });
  });

  it("Training workspace includes selected monitors in run-plan commands and job submissions", async () => {
    const { fetchMock, trainingBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    const details = await selectTrainingMonitorOption(user, /Linear layers/i);
    await user.keyboard("{Escape}");

    let requestCountAfterMonitorSelection = 0;
    await waitFor(() => {
      const bodies = trainingRunPlanRequestBodies(fetchMock);
      expect(bodies[bodies.length - 1]?.monitors).toEqual(["linear"]);
      requestCountAfterMonitorSelection = bodies.length;
    });
    const runList = trainingRunList(details);
    await user.click(
      within(runList).getByRole("button", { name: /command for run 1/i }),
    );
    let commandDialog = await screen.findByRole("dialog", {
      name: /training command/i,
    });
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --datasets Mnist --monitors linear",
    );
    await user.click(
      within(commandDialog).getByRole("button", {
        name: /close training command/i,
      }),
    );

    await user.click(within(runList).getByRole("button", { name: /^commands$/i }));
    commandDialog = await screen.findByRole("dialog", {
      name: /training commands/i,
    });
    expect(
      within(commandDialog).getByRole("textbox", {
        name: /^training commands$/i,
      }),
    ).toHaveValue(
      "(\n  set -e\n  source experiment.sh --model-type linears --model linear --preset baseline --datasets Mnist --monitors linear\n)",
    );
    await user.click(
      within(commandDialog).getByRole("button", {
        name: /close training commands/i,
      }),
    );

    await setTrainingMultiSelectOption(
      user,
      details,
      "Training monitors",
      /Linear layers/i,
      false,
    );
    await user.keyboard("{Escape}");
    await waitFor(() => {
      const bodies = trainingRunPlanRequestBodies(fetchMock);
      expect(bodies.length).toBeGreaterThan(requestCountAfterMonitorSelection);
      expect(bodies[bodies.length - 1]?.monitors).toEqual([]);
    });
    await user.click(
      within(runList).getByRole("button", { name: /command for run 1/i }),
    );
    commandDialog = await screen.findByRole("dialog", {
      name: /training command/i,
    });
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh --model-type linears --model linear --preset baseline --datasets Mnist",
    );
    await user.click(
      within(commandDialog).getByRole("button", {
        name: /close training command/i,
      }),
    );

    await selectTrainingMonitorOption(user, /Linear layers/i);
    await user.keyboard("{Escape}");
    await waitFor(() => {
      const bodies = trainingRunPlanRequestBodies(fetchMock);
      expect(bodies[bodies.length - 1]?.monitors).toEqual(["linear"]);
    });
    await selectNewTrainingLogFolder(user, "monitor_run");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "monitor_run",
        monitors: ["linear"],
      });
      expect(trainingBodies[0]).not.toHaveProperty("runPlan.runs.0.command");
    });
    expect(
      within(details).getByRole("combobox", {
        name: /^training monitors\s+1\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent(/Linear layers/i);
  });

  it("requires a valid log folder before starting training", async () => {
    const { trainingBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    await expandedTrainingDetailsReady(user);
    const startButton = await screen.findByRole("button", { name: /start training/i });
    expect(startButton).toBeDisabled();

    await user.click(screen.getByRole("radio", { name: /new folder/i }));
    const input = screen.getByLabelText(/^new log folder$/i);

    for (const value of [
      "my experiment",
      "my-experiment",
      "my.folder",
      "my/folder",
      "_my_folder",
      "my__folder",
    ]) {
      await user.clear(input);
      await user.type(input, value);
      expect(screen.getByRole("button", { name: /start training/i })).toBeDisabled();
      expect(screen.getByRole("alert")).toHaveTextContent(/single underscores/i);
    }

    await user.clear(input);
    await user.type(input, "my_experiment");
    expect(screen.getByRole("button", { name: /start training/i })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({ logFolder: "my_experiment" });
    });
  });

  it("disables training planning and submission when capabilities disable training", async () => {
    const { fetchMock, trainingBodies } = installFetchMock({
      capabilitiesResponse: {
        ...capabilitiesResponse,
        authMode: "bearer",
        trainingEnabled: false,
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetails(user);
    const startButton = await screen.findByRole("button", { name: /start training/i });
    expect(startButton).toBeDisabled();

    expect(within(details).getByText(/training is disabled/i)).toBeInTheDocument();

    await user.click(within(details).getByRole("radio", { name: /new folder/i }));
    await user.type(within(details).getByLabelText(/^new log folder$/i), "hosted_disabled");

    expect(startButton).toBeDisabled();
    expect(trainingBodies).toHaveLength(0);
    expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/training/run-plan")))
      .toBe(false);
  });

  it("prunes datasets when the Training Experiment Task changes", async () => {
    const { fetchMock, trainingBodies } = installFetchMock({
      datasetsResponse: {
        modelType: "linears",
        model: "linear",
        defaultExperimentTask: "image-classification",
        datasetGroups: [
          {
            experimentTask: "image-classification",
            label: "Image Classification",
            datasets: [
              { name: "Mnist", label: "Mnist", inputDim: 784, outputDim: 10 },
              {
                name: "Cifar10",
                label: "Cifar 10",
                inputDim: 3072,
                outputDim: 10,
              },
            ],
          },
          {
            experimentTask: "tabular-classification",
            label: "Tabular Classification",
            datasets: [
              { name: "Iris", label: "Iris", inputDim: 4, outputDim: 3 },
            ],
          },
        ],
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsReady(user);
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
    );
    expect(
      within(details).getByRole("combobox", {
        name: /^training datasets\s+2\s*\/\s*2 selected$/i,
      }),
    ).toBeInTheDocument();

    const experimentTask = within(details).getByRole("combobox", {
      name: /^experiment task$/i,
    });
    await user.click(experimentTask);
    await user.click(
      within(
        await within(details).findByRole("listbox", {
          name: /^experiment task options$/i,
        }),
      ).getByRole("option", { name: /Tabular Classification/i }),
    );

    await waitFor(() => {
      expect(experimentTask).toHaveTextContent("Tabular Classification");
      expect(
        within(details).getByRole("combobox", {
          name: /^training datasets\s+1\s*\/\s*1 selected$/i,
        }),
      ).toHaveTextContent("Iris");
      const requests = trainingRunPlanRequestBodies(fetchMock);
      expect(requests[requests.length - 1]).toMatchObject({
        experimentTask: "tabular-classification",
        datasets: ["Iris"],
      });
    });

    await selectNewTrainingLogFolder(user, "task_pruning");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        experimentTask: "tabular-classification",
        datasets: ["Iris"],
      });
    });
  });

  it("keeps Start Training disabled when the selected model has no datasets", async () => {
    const { trainingBodies } = installFetchMock({
      datasetsResponse: {
        modelType: "linears",
        model: "linear",
        defaultExperimentTask: "image-classification",
        datasetGroups: [
          {
            experimentTask: "image-classification",
            label: "Image Classification",
            datasets: [],
          },
        ],
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    const details = await expandedTrainingDetails(user);
    expect(within(details).getByText("No datasets for this model")).toBeInTheDocument();

    await user.click(within(details).getByRole("radio", { name: /new folder/i }));
    await user.type(within(details).getByLabelText(/^new log folder$/i), "no_dataset_run");

    expect(screen.getByRole("button", { name: /start training/i })).toBeDisabled();
    expect(trainingBodies).toHaveLength(0);
  });

  it("posts the selected existing log folder", async () => {
    const { trainingBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    await selectExistingTrainingLogFolder(user, "test_model_2");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({ logFolder: "test_model_2" });
    });
  });

  it("checks the started experiment when switching to logs", async () => {
    const { trainingBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    await selectNewTrainingLogFolder(user, "fresh_run");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({ logFolder: "fresh_run" });
    });

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    await user.click(await screen.findByRole("combobox", { name: /^experiments\b/i }));
    const experimentsList = await screen.findByRole("listbox", {
      name: "Experiments options",
    });
    expect(
      await within(experimentsList).findByRole("option", {
        name: /^fresh_run\b/i,
      }),
    ).toHaveAttribute("aria-selected", "true");
  });

  it("does not replay a started folder after the Workbench connection changes", async () => {
    const { trainingBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    await selectNewTrainingLogFolder(user, "old_connection_run");
    await user.click(screen.getByRole("button", { name: /start training/i }));
    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "old_connection_run",
      });
    });

    await user.click(
      screen.getByRole("button", { name: /api connection settings/i }),
    );
    const connectionDialog = await screen.findByRole("dialog", {
      name: /api connection/i,
    });
    const apiUrlInput = within(connectionDialog).getByLabelText("API base URL");
    await user.clear(apiUrlInput);
    await user.type(apiUrlInput, "https://replacement.example.test");
    await user.click(within(connectionDialog).getByRole("button", { name: /^use$/i }));
    await user.click(
      within(connectionDialog).getByRole("button", {
        name: /close api connection settings/i,
      }),
    );

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(await screen.findByRole("combobox", { name: /^experiments\b/i }));
    const experimentsList = await screen.findByRole("listbox", {
      name: "Experiments options",
    });
    expect(
      within(experimentsList).getByRole("option", {
        name: /^old_connection_run\b/i,
      }),
    ).toHaveAttribute("aria-selected", "false");
  });

  it("Full config Update Preview sends a new inspect request for the same selection", async () => {
    const { inspectBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    const initialRequestCount = inspectBodies.length;
    const dialog = await openFullConfig(user);
    const updatePreviewButton = within(dialog).getByRole("button", {
      name: /update preview/i,
    });

    await user.click(updatePreviewButton);
    await waitFor(() => expect(inspectBodies).toHaveLength(initialRequestCount + 1));

    await user.click(updatePreviewButton);
    await waitFor(() => expect(inspectBodies).toHaveLength(initialRequestCount + 2));
  });

  it("changing the main-menu preset preserves overrides and refreshes the preview", async () => {
    const { inspectBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    const initialRequestCount = inspectBodies.length;

    await selectTargetOption(user, "preset", "recurrent-gating-halting");

    expect(screen.getAllByText(/1 overrides?/i).length).toBeGreaterThan(0);
    expect(screen.queryByText("No overrides set")).not.toBeInTheDocument();
    await waitFor(() => expect(inspectBodies).toHaveLength(initialRequestCount + 1));
    expect(inspectBodies.at(-1)).toEqual({
      modelType: "linears",
      model: "linear",
      preset: "recurrent-gating-halting",
      experimentTask: "image-classification",
      dataset: "Mnist",
      overrides: { hidden_dim: "128" },
    });
  });

  it("resetting overrides refreshes the preview when a target is selected", async () => {
    const { inspectBodies } = installFetchMock();
    renderWorkbench();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    const requestCountBeforeReset = inspectBodies.length;

    await user.click(within(dialog).getByRole("button", { name: /reset overrides/i }));

    await waitFor(() => expect(inspectBodies).toHaveLength(requestCountBeforeReset + 1));
    expect(inspectBodies.at(-1)).toEqual({
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      experimentTask: "image-classification",
      dataset: "Mnist",
      overrides: {},
    });
  });

  it("clears the displayed graph while a preview refresh is pending", async () => {
    const nextPreview = deferred<unknown>();
    installFetchMock({
      logRunsResponse: { runs: [] },
      inspectResponseFactory: (requestIndex) =>
        requestIndex === 1 ? nextPreview.promise : inspectResponse,
    });
    renderWorkbench();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    const dialog = await openFullConfig(user);
    await user.click(within(dialog).getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(screen.queryByText("main_model.0")).not.toBeInTheDocument();
    });
    expect(screen.getByText("building")).toBeInTheDocument();

    nextPreview.resolve(inspectResponse);

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
  });

});
