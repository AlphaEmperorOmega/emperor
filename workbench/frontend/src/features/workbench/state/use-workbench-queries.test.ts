import { createElement, type ReactNode } from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock the API layer so we can observe which queries fire (the `enabled`
// gating) and with which arguments (the query-key inputs) per selection.
const mocks = vi.hoisted(() => ({
  fetchHealth: vi.fn(),
  fetchCapabilities: vi.fn(),
  fetchModels: vi.fn(),
  fetchPresets: vi.fn(),
  fetchDatasets: vi.fn(),
  fetchMonitors: vi.fn(),
  fetchConfigSchema: vi.fn(),
  fetchSearchSpace: vi.fn(),
}));
vi.mock("@/lib/api", () => mocks);

import { useWorkbenchQueries } from "@/features/workbench/state/use-workbench-queries";
import { type ModelIdentity } from "@/lib/api";

function renderQueries(
  model: string,
  preset: string,
  trainingPresets: string[] = [],
  options: { includeSearchSpace?: boolean } = {},
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return renderHook(
    ({
      m,
      p,
      presets,
    }: {
      m: string;
      p: string;
      presets: string[];
    }) => useWorkbenchQueries("linears", m, p, presets, options),
    {
      initialProps: { m: model, p: preset, presets: trainingPresets },
      wrapper: ({ children }: { children: ReactNode }) =>
        createElement(QueryClientProvider, { client }, children),
    },
  );
}

beforeEach(() => {
  mocks.fetchHealth.mockReset().mockResolvedValue({ status: "ok" });
  mocks.fetchCapabilities.mockReset().mockResolvedValue({
    authMode: "none",
    trainingEnabled: true,
    trainingCancellationCapability: "unsupported",
    logDeletionEnabled: true,
    configSnapshotsEnabled: true,
    historicalLogsEnabled: true,
    liveMonitorDataEnabled: true,
    historicalMonitorDataEnabled: true,
    uploadsEnabled: false,
    maxUploadSize: null,
    dataSourcesEnabled: false,
    dataSources: [],
  });
  mocks.fetchModels
    .mockReset()
    .mockResolvedValue({ models: [{ modelType: "linears", model: "linear" }] });
  mocks.fetchPresets.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({ ...identity, presets: [] }),
  );
  mocks.fetchDatasets.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({
      ...identity,
      defaultExperimentTask: "image-classification",
      datasetGroups: [
        {
          experimentTask: "image-classification",
          label: "Image Classification",
          datasets: [],
        },
      ],
    }),
  );
  mocks.fetchMonitors.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({ ...identity, monitors: [] }),
  );
  mocks.fetchConfigSchema.mockReset().mockImplementation((identity: ModelIdentity) =>
    Promise.resolve({ ...identity, fields: [] }),
  );
  mocks.fetchSearchSpace.mockReset().mockImplementation(
    (identity: ModelIdentity, preset: string) =>
      Promise.resolve({ ...identity, preset, axes: [] }),
  );
});

describe("useWorkbenchQueries enabled gating", () => {
  it("fires only root queries when no model is selected", async () => {
    renderQueries("", "");

    await waitFor(() => expect(mocks.fetchModels).toHaveBeenCalled());
    expect(mocks.fetchHealth).toHaveBeenCalled();
    expect(mocks.fetchCapabilities).toHaveBeenCalled();
    expect(mocks.fetchPresets).not.toHaveBeenCalled();
    expect(mocks.fetchDatasets).not.toHaveBeenCalled();
    expect(mocks.fetchMonitors).not.toHaveBeenCalled();
    expect(mocks.fetchConfigSchema).not.toHaveBeenCalled();
    expect(mocks.fetchSearchSpace).not.toHaveBeenCalled();
  });

  it("fires model-scoped queries once a model is selected, but not schema", async () => {
    renderQueries("linear", "");

    await waitFor(() =>
      expect(mocks.fetchPresets).toHaveBeenCalledWith({
        modelType: "linears",
        model: "linear",
      }),
    );
    expect(mocks.fetchDatasets).toHaveBeenCalledWith({
      modelType: "linears",
      model: "linear",
    });
    expect(mocks.fetchMonitors).toHaveBeenCalledWith({
      modelType: "linears",
      model: "linear",
    });
    expect(mocks.fetchConfigSchema).not.toHaveBeenCalled();
    expect(mocks.fetchSearchSpace).not.toHaveBeenCalled();
  });

  it("fires preset-scoped queries only when both model and preset are selected", async () => {
    renderQueries("linear", "base", ["base", "post-norm"]);

    await waitFor(() =>
      expect(mocks.fetchConfigSchema).toHaveBeenCalledWith(
        { modelType: "linears", model: "linear" },
        "base",
      ),
    );
    expect(mocks.fetchSearchSpace).toHaveBeenCalledWith(
      { modelType: "linears", model: "linear" },
      "base",
      ["base", "post-norm"],
    );
  });

  it("can skip search-space when callers only need schema metadata", async () => {
    renderQueries("linear", "base", [], { includeSearchSpace: false });

    await waitFor(() =>
      expect(mocks.fetchConfigSchema).toHaveBeenCalledWith(
        { modelType: "linears", model: "linear" },
        "base",
      ),
    );
    expect(mocks.fetchSearchSpace).not.toHaveBeenCalled();
  });

  it("normalizes empty training preset selections to the selected preset", async () => {
    renderQueries("linear", "base");

    await waitFor(() =>
      expect(mocks.fetchSearchSpace).toHaveBeenCalledWith(
        { modelType: "linears", model: "linear" },
        "base",
        ["base"],
      ),
    );
  });
});
