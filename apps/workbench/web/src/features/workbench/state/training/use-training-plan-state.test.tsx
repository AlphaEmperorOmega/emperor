import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  fetchTrainingRunPlan: vi.fn(),
}));

vi.mock("@/lib/api/training-jobs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/training-jobs")>()),
  fetchTrainingRunPlan: mocks.fetchTrainingRunPlan,
}));

import type { ConfigField, SearchAxis } from "@/lib/api/models";
import type { TrainingRunPlan } from "@/lib/api/training-jobs";
import { type ConfigSection } from "@/lib/config";
import { type ConfigSnapshot } from "@/lib/config-snapshots";
import {
  DEFAULT_TRAINING_SEARCH_STATE,
  type TrainingSearchState,
} from "@/lib/training-search";
import { useTrainingPlanState } from "@/features/workbench/state/training/use-training-plan-state";

type PlanInput = Parameters<typeof useTrainingPlanState>[0];

const fields: ConfigField[] = [
  {
    key: "hidden_dim",
    configKey: "HIDDEN_DIM",
    flag: "--hidden-dim",
    label: "Hidden Dim",
    section: "Model",
    sectionPath: ["Model"],
    type: "int",
    default: 64,
    nullable: false,
    choices: [],
    locked: false,
  },
  {
    key: "num_epochs",
    configKey: "NUM_EPOCHS",
    flag: "--num-epochs",
    label: "Epochs",
    section: "Training",
    sectionPath: ["Training"],
    type: "int",
    default: 10,
    nullable: false,
    choices: [],
    locked: false,
  },
];

const configSections: ConfigSection[] = [{ title: "Model", fields }];

const searchAxes: SearchAxis[] = [
  {
    key: "hidden_dim",
    configKey: "HIDDEN_DIM",
    searchKey: "SEARCH_SPACE_HIDDEN_DIM",
    label: "Hidden Dim",
    section: "Model",
    type: "int",
    values: [128, 256],
    locked: false,
  },
];

const gridSearch: TrainingSearchState = {
  mode: "grid",
  selectedValues: { hidden_dim: [128, 256] },
  randomSamples: 10,
};

const randomSearch: TrainingSearchState = {
  mode: "random",
  selectedValues: { hidden_dim: [128, 256] },
  randomSamples: 1,
};

const snapshots: ConfigSnapshot[] = [
  {
    id: "wide",
    name: "Wide",
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    overrides: { hidden_dim: "128", num_epochs: "4" },
    createdAt: "2026-06-01T00:00:00.000Z",
  },
];

function runPlan({
  totalRuns = 1,
  search = null,
  runPrefix = "run",
}: {
  totalRuns?: number;
  search?: TrainingRunPlan["search"];
  runPrefix?: string;
} = {}): TrainingRunPlan {
  return {
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    presets: ["baseline"],
    experimentTask: "image-classification",
    datasets: ["Mnist"],
    overrides: {},
    search,
    logFolder: "runs",
    isRandomSearch: search?.mode === "random",
    runs: Array.from({ length: totalRuns }, (_, offset) => ({
      id: `${runPrefix}-${offset + 1}`,
      index: offset + 1,
      status: "Pending",
      preset: "baseline",
      experimentTask: "image-classification",
      dataset: "Mnist",
      changes: [],
      overrides: {},
      command: `run ${offset + 1}`,
      totalEpochs: 10,
      currentEpoch: 0,
      metrics: {},
      logDir: null,
      error: null,
      errorTraceback: null,
    })),
    summary: {
      totalRuns,
      completedRuns: 0,
      runningRuns: 0,
      pendingRuns: totalRuns,
      failedRuns: 0,
      cancelledRuns: 0,
      skippedRuns: 0,
      totalEpochs: totalRuns * 10,
      completedEpochs: 0,
      remainingEpochs: totalRuns * 10,
    },
  };
}

function planInput({
  search = DEFAULT_TRAINING_SEARCH_STATE,
  selectedSnapshots = [],
  datasets = ["Mnist"],
  hasValidLogFolder = true,
  axes = searchAxes,
  launch = vi.fn(),
  updateSearch = vi.fn(),
  activeRunPlan,
  isJobRunning = false,
}: {
  search?: TrainingSearchState;
  selectedSnapshots?: ConfigSnapshot[];
  datasets?: string[];
  hasValidLogFolder?: boolean;
  axes?: SearchAxis[];
  launch?: PlanInput["execution"]["launch"];
  updateSearch?: PlanInput["draft"]["searchMetadata"]["update"];
  activeRunPlan?: TrainingRunPlan | null;
  isJobRunning?: boolean;
} = {}): PlanInput {
  return {
    draft: {
      modelPackage: {
        modelType: "linears",
        model: "linear",
        primaryPreset: "baseline",
        selectedPresets: ["baseline"],
        selectedSnapshots,
      },
      experiment: {
        task: "image-classification",
        datasets,
        monitors: ["loss"],
        logFolder: "runs",
        hasValidLogFolder,
      },
      runtimeDefaults: {
        sections: configSections,
        overrides: { hidden_dim: "192", dropout: "0.2" },
      },
      searchMetadata: {
        value: search,
        axes,
        isLoading: false,
        update: updateSearch,
      },
    },
    availability: {
      trainingEnabled: true,
      protectedReadsEnabled: true,
    },
    execution: {
      activeRunPlan,
      isJobRunning,
      canLaunch: !isJobRunning,
      launch,
    },
  };
}

function queryClientWrapper(queryClient: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(QueryClientProvider, { client: queryClient }, children);
  };
}

function renderPlan(input: PlanInput) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return renderHook(
    ({ value }: { value: PlanInput }) => useTrainingPlanState(value),
    {
      initialProps: { value: input },
      wrapper: queryClientWrapper(queryClient),
    },
  );
}

beforeEach(() => {
  mocks.fetchTrainingRunPlan.mockReset();
});

describe("useTrainingPlanState", () => {
  it("uses the backend-authoritative Config Snapshot plan and revisions", async () => {
    const launch = vi.fn();
    const authoritativePlan = runPlan({ totalRuns: 2 });
    authoritativePlan.runs[0] = {
      ...authoritativePlan.runs[0]!,
      id: "preset-baseline-Mnist-1",
      overrides: { hidden_dim: "192", dropout: "0.2" },
      totalEpochs: 10,
    };
    authoritativePlan.runs[1] = {
      ...authoritativePlan.runs[1]!,
      id: "snapshot-wide-Mnist-2",
      snapshotId: "wide",
      snapshotName: "Wide",
      overrides: {
        hidden_dim: "192",
        num_epochs: "4",
        dropout: "0.2",
      },
      totalEpochs: 4,
    };
    authoritativePlan.snapshotRevisions = [
      { id: "wide", semanticRevision: "a".repeat(64) },
    ];
    authoritativePlan.summary.totalEpochs = 14;
    authoritativePlan.summary.remainingEpochs = 14;
    mocks.fetchTrainingRunPlan.mockResolvedValue(authoritativePlan);
    const { result } = renderPlan(
      planInput({ search: gridSearch, selectedSnapshots: snapshots, launch }),
    );

    expect(result.current.activeConfigSnapshotCount).toBe(1);
    expect(result.current.search.effective.mode).toBe("off");
    expect(result.current.search.disabledReason).toContain("fixed variants");
    await waitFor(() => {
      expect(result.current.displayRunPlan?.summary.totalRuns).toBe(2);
      expect(result.current.canStart).toBe(true);
    });
    expect(mocks.fetchTrainingRunPlan).toHaveBeenCalledWith(
      expect.objectContaining({
        snapshotIds: ["wide"],
        overrides: { hidden_dim: "192", dropout: "0.2" },
      }),
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(result.current.displayRunPlan?.summary.totalRuns).toBe(2);
    expect(result.current.displayRunPlan?.summary.remainingEpochs).toBe(14);
    expect(result.current.displayRunPlan?.runs[0].overrides).toEqual({
      hidden_dim: "192",
      dropout: "0.2",
    });
    expect(result.current.displayRunPlan?.runs[1]).toMatchObject({
      snapshotId: "wide",
      snapshotName: "Wide",
      overrides: {
        hidden_dim: "192",
        num_epochs: "4",
        dropout: "0.2",
      },
    });
    act(() => result.current.actions.start());
    expect(launch).toHaveBeenCalledWith(
      expect.objectContaining({
        modelType: "linears",
        model: "linear",
        monitors: ["loss"],
        snapshotIds: ["wide"],
        snapshotRevisions: authoritativePlan.snapshotRevisions,
      }),
    );
    expect(launch.mock.calls[0][0]).not.toHaveProperty("runPlan");
  });

  it("uses the backend-authoritative normal plan and keeps folder validity in start readiness", async () => {
    const authoritativePlan = runPlan();
    const launch = vi.fn();
    mocks.fetchTrainingRunPlan.mockResolvedValue(authoritativePlan);
    const rendered = renderPlan(
      planInput({ hasValidLogFolder: false, launch }),
    );

    await waitFor(() => {
      expect(rendered.result.current.displayRunPlan).toBe(authoritativePlan);
    });
    expect(mocks.fetchTrainingRunPlan).toHaveBeenCalledWith(
      {
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        presets: ["baseline"],
        experimentTask: "image-classification",
        datasets: ["Mnist"],
        overrides: { hidden_dim: "192", dropout: "0.2" },
        logFolder: "runs",
        monitors: ["loss"],
      },
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(rendered.result.current.canStart).toBe(false);

    rendered.rerender({
      value: planInput({ hasValidLogFolder: true, launch }),
    });
    await waitFor(() => expect(rendered.result.current.canStart).toBe(true));
    expect(mocks.fetchTrainingRunPlan).toHaveBeenCalledTimes(1);

    act(() => rendered.result.current.actions.start());
    expect(launch).toHaveBeenCalledWith(
      expect.objectContaining({
        overrides: { hidden_dim: "192", dropout: "0.2" },
        runPlan: expect.objectContaining({
          runs: [expect.objectContaining({ id: "run-1" })],
        }),
      }),
    );
  });

  it("owns grid Search Metadata derivation and submits the authoritative grid plan", async () => {
    const authoritativePlan = runPlan({
      totalRuns: 2,
      search: {
        mode: "grid",
        values: { hidden_dim: [128, 256] },
      },
    });
    mocks.fetchTrainingRunPlan.mockResolvedValue(authoritativePlan);
    const launch = vi.fn();
    const { result } = renderPlan(planInput({ search: gridSearch, launch }));

    await waitFor(() => expect(result.current.canStart).toBe(true));
    expect(result.current.search.activeAxisCount).toBe(1);
    expect(result.current.search.combinationCount).toBe(2);
    expect(result.current.search.estimatedRunCount).toBe(2);
    expect(result.current.search.conflictKeys).toEqual(["hidden_dim"]);
    expect(mocks.fetchTrainingRunPlan).toHaveBeenCalledWith(
      expect.objectContaining({
        overrides: { dropout: "0.2" },
        search: {
          mode: "grid",
          values: { hidden_dim: [128, 256] },
        },
      }),
      expect.anything(),
    );

    act(() => result.current.actions.start());
    expect(launch).toHaveBeenCalledWith(
      expect.objectContaining({
        search: {
          mode: "grid",
          values: { hidden_dim: [128, 256] },
        },
        runPlan: expect.objectContaining({
          runs: [
            expect.objectContaining({ id: "run-1" }),
            expect.objectContaining({ id: "run-2" }),
          ],
        }),
      }),
    );
  });

  it("resamples random plans by advancing their private identity", async () => {
    mocks.fetchTrainingRunPlan
      .mockResolvedValueOnce(
        runPlan({
          search: {
            mode: "random",
            values: { hidden_dim: [128, 256] },
            randomSamples: 1,
          },
          runPrefix: "sample-a",
        }),
      )
      .mockResolvedValueOnce(
        runPlan({
          search: {
            mode: "random",
            values: { hidden_dim: [128, 256] },
            randomSamples: 1,
          },
          runPrefix: "sample-b",
        }),
      );
    const { result } = renderPlan(planInput({ search: randomSearch }));

    await waitFor(() => {
      expect(result.current.displayRunPlan?.runs[0].id).toBe("sample-a-1");
      expect(result.current.canResample).toBe(true);
    });
    act(() => result.current.actions.resample());
    await waitFor(() => {
      expect(mocks.fetchTrainingRunPlan).toHaveBeenCalledTimes(2);
      expect(result.current.displayRunPlan?.runs[0].id).toBe("sample-b-1");
    });
  });

  it("retries the same failed plan request without changing draft identity", async () => {
    mocks.fetchTrainingRunPlan
      .mockRejectedValueOnce(new Error("Planner unavailable."))
      .mockResolvedValueOnce(runPlan({ runPrefix: "retry" }));
    const { result } = renderPlan(planInput());

    await waitFor(() => {
      expect(result.current.canRetry).toBe(true);
      expect(result.current.displayPlanError).toBe("Planner unavailable.");
    });
    act(() => result.current.actions.retry());
    await waitFor(() => {
      expect(mocks.fetchTrainingRunPlan).toHaveBeenCalledTimes(2);
      expect(result.current.displayRunPlan?.runs[0].id).toBe("retry-1");
      expect(result.current.canRetry).toBe(false);
    });
    expect(mocks.fetchTrainingRunPlan.mock.calls[1]?.[0]).toEqual(
      mocks.fetchTrainingRunPlan.mock.calls[0]?.[0],
    );
  });

  it("invalidates a pending large-grid confirmation when the draft revision changes", async () => {
    mocks.fetchTrainingRunPlan
      .mockResolvedValueOnce(runPlan({ totalRuns: 110 }))
      .mockResolvedValueOnce(runPlan({ totalRuns: 110, runPrefix: "changed" }));
    const launch = vi.fn();
    const rendered = renderPlan(planInput({ search: gridSearch, launch }));

    await waitFor(() => {
      expect(rendered.result.current.displayedRunCount).toBe(110);
      expect(rendered.result.current.confirmation.isRequired).toBe(true);
    });
    act(() => rendered.result.current.actions.start());
    expect(rendered.result.current.confirmation.isOpen).toBe(true);
    expect(launch).not.toHaveBeenCalled();

    rendered.rerender({
      value: planInput({
        search: gridSearch,
        datasets: ["Cifar10"],
        launch,
      }),
    });
    await waitFor(() => {
      expect(rendered.result.current.confirmation.isOpen).toBe(false);
      expect(mocks.fetchTrainingRunPlan).toHaveBeenCalledTimes(2);
    });
    act(() => rendered.result.current.actions.confirmLargeGridSearch());
    expect(launch).not.toHaveBeenCalled();
  });

  it("projects unlocked Search Metadata without making rendering derive it", () => {
    mocks.fetchTrainingRunPlan.mockResolvedValue(runPlan());
    const updateSearch = vi.fn();
    const lockedAxis: SearchAxis = {
      ...searchAxes[0],
      key: "num_epochs",
      configKey: "NUM_EPOCHS",
      searchKey: "SEARCH_SPACE_NUM_EPOCHS",
      label: "Epochs",
      values: [5, 10],
      locked: true,
      lockedByPresets: ["baseline"],
    };
    const { result } = renderPlan(
      planInput({
        search: gridSearch,
        axes: [...searchAxes, lockedAxis],
        updateSearch,
      }),
    );

    expect(result.current.search.unlockedAxes).toEqual(searchAxes);
    expect(result.current.search.unlockedAxisCount).toBe(1);
    expect(result.current.search.update).toBe(updateSearch);
    expect(result.current.search.lockSummary.lockedAxisCount).toBe(1);
    expect(result.current.search.lockWarning).toContain("preset-owned axis");
  });
});
